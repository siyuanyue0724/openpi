# PICF README

这份 README 只描述当前仓库里已经实际落地的 PICF v0.4.8 core。

当前对应的设计总纲是 [`plan_readme_ray_geometry.md`](/home/siyuanyue/Documents/openpi/plan_readme_ray_geometry.md) 的 `v0.4.8 / MOVEON` 版本。
这里的口径只写“当前仓库真实实现 + 本地实际核查结果”，不把未落地项混写成已完成。

总方案基线以根目录 [`plan_readme.md`](/home/siyuanyue/Documents/openpi/plan_readme.md) 为准。
若要看这次基于 projection/ray geometry first 的修订版设计，请看
[`plan_readme_ray_geometry.md`](/home/siyuanyue/Documents/openpi/plan_readme_ray_geometry.md)。
这里不重复整份方法说明，只回答三件事：

- 当前代码已经实现了什么
- 哪些地方做了明确的工程化近似
- 哪些总纲条款当前只是“部分落地”
- 现在应该从哪里继续接训练 / serving / deployment

如果要看这轮针对 language/action collapse 的**精确重构方案**，请直接看：
[`README_semantic_prefix_refactor.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_semantic_prefix_refactor.md)。

那个文件描述的是：

- 为什么当前 `semantic_summary` 主路径不够强
- 它与 `pi0.5 / pi0.5-sonata` 的逐文件 dataflow 差异
- 为什么目标方案必须改成 token-level posterior-late semantic prefix
- 后续的实施、测试、部署顺序

## 0. 2026-04-09 严格复核

今天这轮重新按“代码、数学、数据链、交接文档”四条线做了复核。
当前可确认的事实是：

- `plan_readme_ray_geometry.md` 里 v0.4.8 的关键实现口径已经和 `src/openpi/picf/core/` 对齐
- `python -m compileall -q src/openpi/picf/core scripts/picf_core_train_smoke.py scripts/picf_core_train.py`：通过
- `pytest -q src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py src/openpi/picf/pointcloud_picf_test.py src/openpi/training/data_loader_test.py`
  - `32 passed`
- `pytest -q src/openpi/picf/paligemma/wrapper_test.py src/openpi/picf/vjepa/wrapper_test.py scripts/picf_core_train_test.py src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py`
  - `51 passed`
- 2026-04-10 云机 full-token semantic smoke：
  - 双卡 `torchrun --nproc_per_node=2`
  - `--use-foundation-backbones --use-tactile --accum-steps 1`
  - 真实 `PaliGemma(source=pi0_pytorch trainable=True) + V-JEPA + Sonata + AnyTouch`
  - `step=2` 正常完成
  - `loss_total(step1)=2.6482`
  - `loss_total(step2)=1.7631`
  - checkpoint 已落到 `/tmp/openpi-train-smoke/picf_core/picf_fulltoken_ddp_smoke/2`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py src/openpi/picf/pointcloud_picf_test.py src/openpi/training/data_loader_test.py`
  - `32 passed`
- `task_ABCD_D` 的真实 `dir/zip` 原始样本一致性复核：
  - 对训练样本索引 `0 / 1 / 10 / 100 / 1000`，
    `prompt`、`rgb_static`、`depth_static`、`robot_obs`、`rgb_gripper`、`actions`
    逐字段完全一致，最大绝对差均为 `0`
- `scripts/stageb_calvin_audit.py`
  - `--mode dataset --backend zip --split validation`：通过
  - `--mode loader --backend zip --split validation --batch-size 4 --num-workers 0`：通过
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/stageb_calvin_audit.py --mode loader --backend zip --split validation --batch-size 4 --num-workers 0`：通过
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/stageb_calvin_audit.py --mode loader --backend dir --split training --batch-size 4 --num-workers 0`：通过
- `scripts/picf_core_train_smoke.py`
  - `dir + cpu + segment_index=0`：通过
  - `zip + cpu + segment_index=0/50/100`：通过
  - 云机 `dir + cuda + --use-foundation-backbones --use-tactile`：通过
    - `visual_mode = encoder`
    - `tactile_mode = encoder`
    - `point_backbone = sonata`
    - `tactile_enabled = true`
    - `loss_total = 3.1340`
    - `loss_tactile_real = 0.5052`
    - `loss_pt = 0.6932`
    - `projective_candidate_density = 0.0340`
  - 云机 `zip + cuda + --use-foundation-backbones --use-tactile`：通过
    - `loss_total = 3.1339`
    - `loss_tactile_real = 0.5052`
    - `loss_pt = 0.6932`
- `scripts/picf_core_train.py`
  - `dir + cuda + 2-step long-run`：通过
  - `dir + cuda + resume from latest.pt`：通过
  - `zip + cuda + 1-step long-run`：通过
  - 云机 `dir + cuda + --use-foundation-backbones --use-tactile + 5-step short-run`：通过
    - `loss_total` 连续 `5` 个 step 保持有限值，并从 `3.0362 -> 2.1081`
    - `loss_tactile_real` 保持在 `0.5562 -> 0.4673`
    - `loss_pt` 保持非零，约 `0.6941 -> 0.7252`
    - `projective_candidate_density` 稳定在 `0.0336 ~ 0.0361`
    - `step_5.pt` 已落盘
  - 云机 `dir + cuda + --use-foundation-backbones --use-tactile + 1-step`：通过
    - `loss_total = 3.0362`
    - `loss_tactile_real = 0.5562`
    - `loss_pt = 0.6941`
    - `step_1.pt` 已落盘
  - 云机同一实验 `--resume` 到 `step=2`：通过
    - `loss_total = 2.7766`
    - `loss_tactile_real = 0.5200`
    - `step_2.pt` 已落盘
  - 云机 `torchrun --standalone --nnodes=1 --nproc_per_node=2 ... --use-foundation-backbones --use-tactile --num-train-steps 1`：通过
    - `loss_total = 2.9149`
    - `loss_tactile_real = 0.5379`
    - `loss_pt = 0.6930`
    - `step_1.pt` 已落盘
  - 云机默认正式配置（不再显式缩小 core 维度）：
    - 单卡 `dir + cuda + --use-foundation-backbones --use-tactile --num-train-steps 30000 --log-interval 1 --save-interval 1`
    - 运行时通过 `timeout` 提前截断，但已真实跑到 `step=57`
    - `args.json` 明确记录：
      - `hidden_dim=384`
      - `posterior_hidden_dim=384`
      - `latent_dim=112`
      - `innovation_dim=384`
      - `control_dim=384`
      - `semantic_dim=2048`
      - `semantic_cross_dim=512`
      - `future_hidden_dim=384`
      - `persistent_anchors=16`
      - `observation_anchors=24`
      - `attention_heads=8`
      - `future_vote_heads=4`
      - `warmup_steps=600`
    - `metrics.jsonl` 真实首步：
      - `loss_total = 2.8504`
      - `lr = 3.33e-07`
      - `loss_pt = 0.6939`
      - `projective_candidate_density = 0.0361`
    - 截断前最后一步 `step=57`：
      - `loss_total = 2.0958`
      - `lr = 1.90e-05`
      - `latest.pt` 与 `57/` checkpoint 目录都存在
  - 云机默认正式配置双卡 DDP：
    - `torchrun --standalone --nnodes=1 --nproc_per_node=2 ... --num-train-steps 1`
    - `args.json` 同样记录 spec 默认结构：
      - `hidden_dim=384`
      - `latent_dim=112`
      - `persistent_anchors=16`
      - `observation_anchors=24`
      - `attention_heads=8`
      - `future_vote_heads=4`
    - `metrics.jsonl` 首步：
      - `loss_total = 2.9831`
      - `loss_tactile_real = 0.5139`
      - `loss_pt = 0.6914`
      - `projective_candidate_density = 0.0363`
    - `step_1.pt` 已落盘，未出现 OOM / DDP 包装错误 / checkpoint 异常
  - 云机 `torchrun --standalone --nnodes=1 --nproc_per_node=2 ... --num-train-steps 120 --log-interval 20 --save-interval 120`：
    - 已在加入 first-step window 重采样后完整跑通
    - `120/120` checkpoint 目录已落盘
    - `metrics.jsonl` 最后一条：
      - `step = 120`
      - `loss_total = 2.6533`
      - `resampled_empty_first_step_windows = 1`
    - 这次 run 明确说明：
      - 旧的 DDP watchdog timeout 不是“多卡自己卡住”
      - 真正的根因是某个 rank 抽到的训练窗口首帧没有局部 `xyzrgb` support，导致该 rank 抛错退出，另一个 rank 卡在 `allreduce`
      - 现在 trainer 会对这种首步非法窗口做有限次 rejection sampling，不再把它升级成 NCCL 超时
  - 云机 `torchrun --standalone --nnodes=1 --nproc_per_node=2 ... --num-train-steps 200 --log-interval 20 --save-interval 200`：
    - 已跨过旧故障区间 `step≈151` 并完整跑通到 `200/200`
    - `200/200` checkpoint 目录已落盘：`/tmp/openpi-train-preflight-ddp200/picf_core/picf_ddp_preflight_200/200`
    - `metrics.jsonl` 最后一条：
      - `step = 200`
      - `loss_total = 2.4001`
      - `resampled_empty_first_step_windows = 0`
    - 这次 run 说明：
      - 旧的 reducer / NCCL 卡死不是“point_error_encoder 仍然随机掉梯度”
      - 更准确的根因是：训练器之前在 DDP 包裹的 `forward` 内部捕获首步非法窗口异常并重试；一旦某个 rank 的 `forward` 在 reducer 已建图后中途抛错，就可能把该 rank 留在 unfinished reduction 状态
      - 当前修复已把首步 `xyzrgb` 合法性检查前移到进入 DDP `forward` 之前；重采样现在只发生在纯数据预检阶段，不再污染 DDP reducer 状态
  - 2026-04-09 最终复核重跑：
    - `python -m py_compile scripts/picf_core_train.py scripts/picf_core_train_test.py src/openpi/picf/core/pipeline.py src/openpi/picf/core/training.py`：通过
    - `pytest -q src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py src/openpi/picf/pointcloud_picf_test.py src/openpi/training/data_loader_test.py scripts/picf_core_train_test.py`：`43 passed`
    - `pytest -q src/openpi/picf/paligemma/wrapper_test.py src/openpi/picf/core/pipeline_test.py scripts/picf_core_train_test.py src/openpi/picf/core/training_test.py src/openpi/picf/vjepa/wrapper_test.py`：`51 passed`
    - `pytest -q src/openpi/picf/pointcloud_picf_test.py src/openpi/training/data_loader_test.py`：`9 passed, 1 skipped`
    - 当前可以把结论写成：
      - 数学 contract：首步 rejection sampling 仅限制到合法首步支持集，没有改 `PICF` 的 posterior 定义
      - 工程 contract：DDP 预检已经前移到 distributed autograd 图之外，不再靠 reducer 内部异常恢复
      - 部署 contract：本地、GitHub、云机应保持同一 commit 后再开训
- 云机 `root@px-cloud2.matpool.com:/root/openpi` 上的 foundation-weight 复核：
  - `scripts/vjepa_ckpt_fetch.py --model vjepa2_1_vit_base_384 --out-root /root/openpi/checkpoints/foundation/vjepa2_1`：通过
  - 真实 wrapper 加载 `/root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt`：通过
    - `checkpoint_loaded = True`
    - `tokens_shape = (32, 24, 24, 768)`
  - 单独把真实 V-JEPA 权重接进 `PicfFullCore`，并在 CALVIN transition 上跑 `forward + compute_transition_loss + backward + optimizer.step()`：通过
    - `mean_loss = 2.0242`
    - `num_visual_tokens = 576`
    - `loss_visual_latent ≈ 0.359 / 0.371`
    - `loss_visual_real ≈ 0.445 / 0.450`
- AnyTouch2 权重在云机上已通过 `HF_ENDPOINT=https://hf-mirror.com + HF_TOKEN` 这条镜像路径完成复核：
  - 直连 `huggingface.co` 当前仍会 `ConnectTimeout` / `Connection reset by peer`
  - 但 `hf-mirror.com` 可正常认证：
    - `whoami-v2` 返回有效账号
    - `api/models/xxuan01/AnyTouch2-Model` 返回 `gated = auto`
  - `hf_hub_download(... filename='checkpoint-4frames.pth', endpoint='https://hf-mirror.com')`：通过
    - 落盘路径：`/root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth`
    - 文件大小约 `1.1G`
  - 真实 `AnyTouch2TactileEncoder` 加载该 checkpoint 并跑 dummy `encode_sensor_clips(...)`：通过
    - `checkpoint_loaded = True`
    - `sensor_tokens_shape = (398, 768)`
    - `pooled_dim = 3072`
- CALVIN tactile 现已确认不是“没有数据”，而是：
  - `episode_0000000.npz` 明确包含 `rgb_tactile` 与 `depth_tactile`
  - 其形状分别为：
    - `rgb_tactile = (160, 120, 6)`
    - `depth_tactile = (160, 120, 2)`
  - 也就是两路 tactile RGB 传感器按通道拼接后的 `3*K` 形式
  - 现有 [`src/openpi/picf/replay/calvin_replay.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/replay/calvin_replay.py) 已能把它拆成两路 `PicfTactilePacket`
  - CALVIN 还明确提供 `depth_gripper`；当前 point path 已经把它并入双深度几何分支，不再只依赖 `depth_static`
  - `robot_obs[0:6]` 在 CALVIN 里是 TCP pose，`robot_obs[6]` 是 gripper opening width；当前 tactile pose 也不再是假定的固定 `±1cm`，而是用 TCP-local 标定量和 opening width 动态求左右指尖中心
- 在云机上，真实 `V-JEPA + AnyTouch + CALVIN tactile replay` 组合已经完成一步训练闭环：
  - 使用 `CalvinSequentialReplay(... use_tactile=True)` 构造 observation
  - `PicfFullCore` 同时加载真实 `V-JEPA` 与真实 `AnyTouch` checkpoint
  - `forward + compute_transition_loss + backward + optimizer.step()`：通过
  - 实测：
    - `num_visual_tokens = 576`
    - `num_tactile_tokens = 2`
    - `availability_tactile = 1.0`
    - `loss_tactile_real ≈ 0.480 / 0.468`
    - `mean_loss = 2.5546`
- 真实 semantic side path 现在也已接进长期 trainer：
  - `--use-foundation-backbones` 会显式启用：
    - `semantic_mode = paligemma`
    - `semantic_source = auto`
    - `semantic_trainable = True`
  - 若本地存在 `pi05_base_pytorch/model.safetensors`，当前会优先走 `pi0_pytorch` 本地 checkpoint 路径
  - 否则回退到 HF `PaliGemmaForConditionalGeneration`
- 云机 gradient probe 已确认五条主干都真实收到梯度，不是“只把 foundation 当 frozen feature extractor”：
  - `point_backbone: trainable_params=454, params_with_grad=452, grad_norm=6.0717`
  - `visual_backbone: trainable_params=158, params_with_grad=155, grad_norm=1.0025`
  - `tactile_backbone: trainable_params=201, params_with_grad=198, grad_norm=0.2661`
  - `semantic_backbone: trainable_params=603, params_with_grad=603, grad_norm=2.2966`
  - `picf_core: trainable_params=1131, params_with_grad=1123, grad_norm=26.8773`
- 当前 cotrain 路径的两个关键工程修复也已经过云机回归：
  - `V-JEPA` trainable 路径改成“参数保留原生 fp32 + CUDA autocast 前向”，避免先前 hard-cast 到 `bf16` 时的 patch-embed bias dtype 冲突
  - `PaliGemma` trainable 路径：
    - 已修复本地 `pi0_pytorch` checkpoint 的 tied embedding 缺口
    - 已修复 batch=1 图像 pad 预处理 squeeze
    - 已修复手工 checkpoint 包裹在输入不带 grad 时的静默断梯度
    - 在 `DDP + accum_steps>1` 时会自动关闭 semantic gradient checkpointing，以避免 `mark ready twice`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/picf_core_train.py --backend dir --device cuda --num-train-steps 1`：通过
  - 在 `/tmp/openpi-train-smoke/picf_core/picf_core_train_uv_min/` 落下：
    - `args.json`
    - `metrics.jsonl`
    - `latest.pt`
    - `step_1.pt`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/picf_core_train_smoke.py --backend dir --segment-index 0 --device cpu`：通过
- `python scripts/picf_core_train_smoke.py --backend dir --segment-index 0 --device cuda`：通过
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/picf_core_train_smoke.py --backend dir --segment-index 0 --device cuda`：通过
- `python scripts/picf_core_train_smoke.py --backend zip --segment-index 0 --device cuda`：通过
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/train_pytorch.py ... --pytorch-training-precision bfloat16 --model.paligemma-variant dummy --model.action-expert-variant dummy`：通过
  - 在 `/tmp/openpi-train-smoke/pi05_calvin_sonata/smoke_calvin_train_dummy_1step_uv_bf16/1` 落 checkpoint
- follow-through 不变量复核：
  - `posterior_mu_diff = 0.0`
  - `posterior_sigma_diff = 0.0`
  - `action_diff = 0.0296`
  - `binding_col_err = 0.0`
  - `ray_norm_err = 1.19e-07`
  - `innovation_norm_t0 = 0.0`
  - `innovation_norm_t1 = 0.9829`
  - `projective_bias_nonzero = 230`
  - `fusion_attention_shape = [79, 79]`
  - `future_action_diff_teacher_vs_policy = 1.5760`
  - `prev_policy_only_mu_diff = 0.0`
  - `prev_executed_mu_diff = 0.0035`

这次对 empty first-step window 的根因还做了进一步复现：

- 复现实验按双卡 DDP 真实种子重放了前 `120` 次窗口采样
  - `rank0(seed=0)`：`0` 次 empty first-step window
  - `rank1(seed=17)`：第 `100` 次采样命中 `1` 个 empty first-step window
- 具体坏窗口是：
  - `segment_id = 11137`
  - `start_step = 196437`
  - `lang = "move the sliding door to the left"`
  - `robot_xyz = [0.1629, -0.0897, 0.6189]`
  - 当前最近点距离 `nearest_dist = 0.1286 m`
  - 但当时训练 contract 的局部 crop 半径只有 `crop_radius_m = 0.08 m`
- 更关键的是，这不是整帧 point cloud 为空，而是：
  - `total_points = 512`
  - 只是以当前 EE 为中心的局部 ROI 为空
- 同一段附近逐帧重放表明：
  - `196435 ~ 196438` 这几帧在当时的 `0.08m` 半径下都没有局部点
  - 到 `196439` 开始才重新出现局部点
  - 因此这属于“窗口起点落在 free-space pre-contact 状态”而不是点云构造器坏掉

从方法论上，这个修复不是“吞异常继续跑”，而是：

- `PICF` 在 `previous is None` 的首个 control step 上，当前 posterior 必须由当前真实 point evidence 启动
- 所以没有局部 `xyzrgb` support 的窗口起点，本来就不属于模型的合法首步支持集
- 训练器现在做的是对“合法首步支持集”进行 rejection sampling
- 对已有 `previous` 的后续控制步，局部 point support 暂时为空仍然是允许的；这和 spec 里的 carried prior / current evidence 角色分工一致
- 另外，当前实现还把这一步 rejection sampling 严格放在 DDP `forward` 之前：
  - 合法性检查只依赖当前 observation、标定和 point crop，本质是数据域判定
  - 因而它应该发生在进入 distributed autograd 图之前
  - 这也是当前双卡 `200` step 已完整跑通的直接原因

同时也有几条必须诚实写清的边界：

- 当前 full-access shell 下重新探测，`.venv` Python 与 `uv run --no-sync python` 都给出：
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() == 1`
  - `device_name = NVIDIA GeForce RTX 3070 Ti Laptop GPU`
  - `nvidia-smi` 也能正常看到 `Driver 581.95 / CUDA 13.0`
- 同一条 `dir + cpu + segment_index=0` smoke 在当前机器上仍复现了：
  - 直接 `python scripts/picf_core_train_smoke.py ...` 会给出 `cuda_runtime_available = true`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/picf_core_train_smoke.py ...` 会给出 `cuda_runtime_available = false`
  这个差异目前只影响 smoke 诊断字段，不影响 loss、梯度、数据链或 core 数学路径
- 官方 `scripts/train_pytorch.py pi05_calvin_sonata` 这条旧训练主线在当前机器上的真实边界是：
  - `uv + cuda + bfloat16 + dummy paligemma/action expert`：可完整跑完 `forward + backward + optim.step()` 并保存 checkpoint
  - `uv + cuda + bfloat16 + full`：会在 Sonata encoder 进入 forward 时 OOM
  - `uv + cuda + float32 + dummy`：会在 `optim.step()` 分配 Adam state 时 OOM
  - `uv + cuda + float32 + full`：会在 `model.to(cuda)` 时 OOM
  - `uv + cpu`：默认会因 `require_cuda=True` fail-fast；即便显式 `--model.no-require-cuda`，Sonata/spconv 仍会在 CUDA stream 路径报错，所以这条官方 Sonata 训练链当前本质上是 CUDA-only
- legacy `scripts/scaffold/scaffold_replay_smoke.py` 当前真实状态是：
  - `dir` backend 可通过
  - `zip` backend 当前会报 `RuntimeError("No scaffold states were produced.")`
- `scripts/picf_core_train.py` 这轮已经完成过一次真实 cloud DDP 最小回归：
  - `torchrun --standalone --nnodes=1 --nproc_per_node=2`
  - `dir + cuda + --use-foundation-backbones --use-tactile --num-train-steps 1`
  - checkpoint 已落到 `/tmp/openpi-train-smoke/picf_core/picf_core_foundation_ddp_ready/step_1.pt`
  - 这足以说明 launcher、checkpoint、DDP 包装和 foundation-weight 路径都能起步
  - 但它仍然不等于“已经完成正式多卡长程收敛回归”


## 1. 当前主线

当前真正的主线已经切到 [`src/openpi/picf/core/pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py) 的 `PicfFullCore`，其前向现在对齐到：

`xyzrgb point subset + V-JEPA visual map + AnyTouch tactile bundle + proprio/action/timing context`
→ `unified multimodal token field`
→ `observation anchors`
→ `persistent posterior anchors`
→ `global posterior self-attention`
→ `language-late predictive heads`
→ `explicit innovation token`
→ `action head`

当前 core 输出状态定义在 [`src/openpi/picf/core/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/contracts.py)：

- `PicfTokenFieldState`
- `PicfObservationAnchorState`
- `PicfPosteriorAnchorState`
- `PicfPredictiveState`
- `PicfPredictionCache`
- `PicfCoreState`

旧的 `support / object shell / stage2` 结构不再是主线接口。

当前最重要的工程判断是：

- current posterior 已经 language-free
- innovation 已经是显式 residual token，而不是内部 gate 代理量
- tactile / point future heads 已经是 real-signal heads
- 训练侧的一步 `t -> t+1` teacher target 与 transition loss 现在已经闭合到 [`src/openpi/picf/core/training.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/training.py)
- 但它还没有并进通用的 [`scripts/train_pytorch.py`](/home/siyuanyue/Documents/openpi/scripts/train_pytorch.py) 主训练入口


## 2. 已实现模块

### 2.1 Unified Token Field

当前 `PicfFullCore` 已经把四类输入统一投影到一个共享隐空间：

- point tokens
- visual tokens
- tactile tokens
- compact context tokens

它们经过共享 `TransformerStack` 融合后，形成当前感知 token field。

这条 token field 现在已经不是单纯的 `PE_pt + PE_img + PE_cam` 版本。
当前代码已经把 geometry-first 的 point↔visual 几何先验接进 token builder：

- point token 额外吃 `PE_proj`
- visual token 额外吃 `PE_ray`
- 无效投影点稳定走 learned null projection branch
- token field 会显式导出 `projective_geometry`，其中包含：
  - point visibility
  - projected continuous patch-grid 坐标
  - visual ray direction
  - projective compatibility
  - sparse candidate edge mask

这里再明确一次当前代码的坐标语义：

- `point_proj_grid_index / visual_grid_index`：
  连续 patch-grid 坐标，以 **patch** 为单位；当前 `PE_proj`、`PE_ray` 和 `projective_compatibility` 都使用这一套
- `point_proj_grid_norm / visual_grid_norm`：
  归一化辅助坐标；当前仅为 diagnostics / backward-compatibility 保留，不再作为主几何量

### 2.2 Observation Anchors

当前 observation anchors 由 point subset 的 FPS seeds 初始化，再从统一 token field 反复 cross-attend 读取。

几何读出遵循 point-only 原则：

- `x`
- `S`
- `a`

只从 point token 权重回读，不让语言或 tactile 直接写当前几何。

### 2.3 Persistent Posterior Anchors

当前 persistent anchors 维护：

- recurrent state `h, c`
- Gaussian posterior `mu, Sigma`
- 几何 `x, S, a`
- 软活动度 `alpha`
- soft binding `binding`
- recycle gate `recycle_gate`

当前 posterior 只依赖：

- carried prior
- 当前 observation anchor evidence

语言不进入当前 posterior。

### 2.4 Global Predictive-Innovation Module

当前 predictive / innovation 路径现在分成两层：

1. **物理预测基底**
   - predictive 分支先只在 world stream 内部做 self-attention
   - world tokens 由：
     - `posterior.tokens`
     - `posterior.global_post`
     - `proprio_token`
     - `action_cond_token`
     构成
   - 从这个 world-only state 先产生：
     - `physical_global_pred`
     - `physical_prediction_cache`
   - 下一步 innovation **只读取这份 physical prediction cache**

2. **semantic-conditioned future readout**
   - 在 `physical_prediction_cache` 固定之后，当前实现优先走
     **semantic-guided task-anchor sidecar**：
     - semantic tokens 先 condition 一组 task queries
     - conditioned queries 再从完整 `fused_tokens` 读取：
       - `task_anchor_tokens`
       - `task_global_token`
       - `instruction_token`
   - 这些 sidecar tokens 再作为 posterior-late 的任务读取工作集并入
     semantic-conditioned predictive 主干
   - 旧的 raw `semantic_prefix_tokens` 路径仍然存在，但只作为
     `legacy_semantic_prefix_enabled` 的回退开关，不再是默认的唯一语义入口
   - 这个 semantic-conditioned state 再导出：
     - `global_pred`
     - `prediction_cache`
   - 这份 cache 用于 future-head 训练读出，但**不是 innovation 的物理比较基底**

推理时读取的是上一步 `physical_prediction_cache`，与当前真实目标做显式 residual，对应产生 innovation token。

动作头读取的是：

- `posterior.tokens`
- `posterior.global_post`
- `innovation_token`
- `proprio_token`
- posterior-late task sidecar tokens：
  - `task_anchor_tokens`
  - `task_global_token`
  - `instruction_token`
- 可选 legacy `semantic_prefix_tokens`
- `control_query_tokens`

这就是当前控制路径。

### 2.5 动作条件的时序语义

这里单独说明，因为这是最容易写错的地方。

当前代码现在遵循下面的约定：

- state 现在显式缓存 `executed_action`
- current prior 优先读上一时刻 `executed_action`
- context token 里的 action 分支也优先读上一时刻 `executed_action`
- 若外部没有提供已执行动作，才回退到上一时刻 policy 输出
- future heads 在训练侧若显式提供 `action_future`，则走 teacher forcing
- future heads 在普通推理路径里默认读当前动作头输出，而不是 `observation.action`

这和总纲里的：

- carried prior 由上一动作传播
- predictive heads 推理时用当前已选动作

是一致的；同时当前回归还显式验证了：

- 只改 `previous.predictive.action`，不改 `previous.predictive.executed_action`，下一步 posterior 不变
- 改 `previous.predictive.executed_action`，下一步 posterior 会改变


## 3. 十条硬约束核查

这里把 [`plan_readme.md`](/home/siyuanyue/Documents/openpi/plan_readme.md) 的 H1-H10 逐条映射到当前代码。

### 3.1 H1 current posterior 是唯一物理 belief state

当前实现状态：已满足。

对应原因：

- posterior 只由 prior + observation-anchor evidence 决定
- semantic summary 不进入 `_posterior_update(...)`
- innovation token 不进入 current posterior
- action head 不反写 posterior

### 3.2 H2 当前感知先统一 token 化

当前实现状态：已满足。

point / visual / tactile / compact context 都先投到统一 hidden size，再进共享 token fusion transformer。
其中 point↔visual 的主对齐机制现在也已经切到 geometry-first：

- token-level `PE_proj`
- token-level `PE_ray`
- projective candidate mask

而不是把强 pairwise `L_pv` 当主力。

### 3.3 H3 语言只在 posterior 之后进入 predictive / task-readout stage

当前实现状态：已满足。

当前真正进入 downstream 融合的是 **posterior-late 的 sidecar 读取结果**：

- control 分支里，posterior 固定之后，semantic 会先 condition task queries，
  再从完整 `fused_tokens` 读出：
  - `task_anchor_tokens`
  - `task_global_token`
  - `instruction_token`
  然后与：
  - `posterior.tokens`
  - `posterior.global_post`
  - `innovation_token`
  - `proprio_token`
  - `control_query_tokens`
  一起进入 control 主干
- predictive 分支里，先固定 language-free 的 `physical_prediction_cache`，
  再把同一套 task sidecar tokens 并入 semantic-conditioned predictive 主干
- 旧的 raw semantic prefix 仍然可以通过 feature flag 参与 downstream，
  但它现在是 rollback path，不再是默认唯一入口

当前实现已经不再把 `semantic_summary` 当作主路径概念：

- `semantic_summary` 不参与 `_posterior_update(...)`
- `semantic_summary` 不参与 language-free 的 `physical_prediction_cache`
- `semantic_summary` 也不再承担 control / predictive 的主要语言入口
- 当前 action / semantic-conditioned future 依赖的是
  **token-level posterior-late task sidecar**

这一步是明确向 `pi0.5 / pi0.5-sonata` 的可靠主干靠拢：

- mixed image+text prefix 继续保留，这本身就是正确设计
- 语言不再被单个 summary bottleneck 压缩后再喂给动作主干
- semantic 不直接改写 physical posterior，而是负责 condition task queries
- task queries 再从完整 `fused_tokens` 读取任务相关对象/区域作为 sidecar
- 同时 innovation / posterior / physical basis 仍然保持 language-late，不会被当前帧语言信息污染

当前 sidecar 方案的实施与部署，统一写在：
- [`README_task_anchor_sidecar_deployment_plan.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_task_anchor_sidecar_deployment_plan.md)
- [`README_task_anchor_sidecar_followthrough.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_task_anchor_sidecar_followthrough.md)

### 3.4 H4 tactile 与 point future heads 默认预测真实信号

当前实现状态：部分满足，但主方向正确。

已满足的部分：

- tactile head 预测 low-res tactile summary + contact/force/pose auxiliary
- point head 预测 coarse occupancy target

尚未完全展开的部分：

- 还没有上到 TSDF / range / scene-flow / resampled point set

### 3.5 H5 visual head 默认 dual-head

当前实现状态：已满足。

当前 visual future head 包含：

- latent head
- real RGB summary head

### 3.6 H6 innovation 必须显式定义

当前实现状态：已满足。

innovation 来自：

- 上一步 `physical_prediction_cache`
- 当前真实 target

的显式残差；当前代码没有把任何 LSTM gate 当 innovation。

### 3.7 H7 persistent anchor 生命周期默认软机制

当前实现状态：已满足最小闭合版。

当前已经实现：

- soft binding
- dustbin
- support mass
- activity
- recycle gate

当前没有引入旧式 `active/dormant/retired` 硬状态机。

### 3.8 H8 除真实缺失和安全限制外不引入新 hard gates

当前实现状态：基本满足。

当前保留的硬规则主要是：

- point contract violation
- sensor sync invalid
- first-step pointcloud contract check
- optional normalized action output clip

其余 anchor/activity/recycle/contact 都是连续量。

### 3.9 H9 默认训练范式为单阶段端到端

当前实现状态：core 训练图已闭合，并且已经有独立的长期训练入口。

也就是说：

- 现在的 core graph 已经是单图结构
- `PicfFullCore.extract_targets(...) + compute_transition_loss(...)` 已经提供了一步 future supervision 的最小训练闭环
- `scripts/picf_core_train_smoke.py` 已经能在真实 CALVIN 帧上做 `forward + backward + optimizer.step()`
- `scripts/picf_core_train.py` 已经把这条新 core 路径接成独立的长期训练入口，支持 checkpoint / resume
- 但通用的 `scripts/train_pytorch.py` 仍然不是这条新 core 的正式训练入口

所以这里可以宣称“本地可验证的一步训练闭环与长期训练 launcher 都已完成”，但不能宣称“仓库所有训练入口都已统一到新 core”。

### 3.10 H10 推理时不要求显式 rollout，但要使用上一步预测形成当前 innovation

当前实现状态：已满足。

当前 runtime 只缓存一步 `physical_prediction_cache` 与一步 semantic-conditioned `prediction_cache`，
不做多步 imagination rollout。


## 4. 当前工程化近似

下面这些是刻意做出的实现近似，不是遗漏。

### 4.1 Gaussian 使用对角协方差实现

方法总纲允许 full covariance 记号，但当前工程实现为了稳定性和算力开销，内部使用 diagonal covariance 做 information-form fusion。

对外导出的 `Sigma` 仍然是对角矩阵形式的 `[K, D_z, D_z]`，便于后续接口保持统一。

这点需要明确：

### 4.2 动作空间当前使用 `pi0/pi0_fast_sonata` 风格的 normalized contract

当前实现状态：已满足。

当前 PICF trainer / serve 路径不再直接在模型内部使用 CALVIN 的原始 `rel_actions`
物理量级，而是：

- 训练数据入口先用 CALVIN `norm_stats.json` 对 action 做 normalize
- core 在 normalized action space 内部训练
- serving 返回动作前再做 unnormalize
- 如需做数值保护，只允许在 normalized action space 里做宽松 `clip`，
  而不是把内部动作头硬裁到固定 `0.025m / pi/18`

这么做的原因是：

- 标准 `pi0 / pi0_fast_sonata` 训练/推理链本来就是 normalized action contract
- CALVIN 原始 `rel_actions` 的统计量级明显大于旧 PICF 内部物理单位 clip 上限
- 如果不先对齐 action contract，而直接上调 `pos/rot` loss 权重，只会把优化更用力地推向一个过小的输出墙

所以当前工程结论是：

- `loss_action_pos / loss_action_rot` 的优先修复项不是“继续加权”
- 而是先保证 action normalization / unnormalization / internal clip 这一整条链和标准 `pi0/pi0_fast_sonata` 保持一致

### 4.3 auxiliary losses 现在采用分组预算，而不是让所有辅助项自由抢占总损失

当前实现状态：已满足。

当前总损失按三组处理：

- `action`
- `physical_aux`
- `semantic_aux`
- `alignment`

其中：

- tactile real loss 已拆成 `tactile_map + tactile_aux`
- tactile aux 内部又按 contact / force / indent / pressure / pose 做显式尺度处理
- `physical_aux / semantic_aux / alignment` 都会相对 detached `action_loss` 做预算上限

这条设计的目的不是“把所有辅助损失都压小”，而是：

- 保证 action 仍然是主导项
- 防止 point / tactile / semantic / alignment 在某些 batch 上突然喧宾夺主
- 同时保留 auxiliary 对 world-model 训练的持续牵引

- 这是数学上的近似，不是文档疏漏
- current fusion 仍然是 information-form，只是 precision 变成 diagonal precision

### 4.2 Point Real Head 目前输出 coarse occupancy target

当前 point real target 没有直接做 TSDF / range-image / scene-flow。
目前实现是：

- 以当前 ROI center 为中心
- 对 crop 内点集做 coarse occupancy voxelization
- 输出固定维度的 occupancy summary

这是“真实点云信号优先”的最小闭合版本，后续可扩展到更强目标。

### 4.3 Tactile Real Head 目前输出 low-res tactile summary

当前 tactile real target 不是 latent，也不是 full tactile frame reconstruction。
当前实现是：

- 各 sensor 原始 tactile RGB 的低分辨率灰度摘要
- contact / force / indent / pressure / pose 辅助量

这仍然属于 real-signal supervision，而不是 latent-only。

### 4.4 Visual Real Head 目前输出 low-res RGB summary

当前 visual real target 是轻量真实视觉目标：

- 对当前 RGB 做低分辨率 pooling

视觉 latent target 仍然保留，用于 predictive embedding。

### 4.4b Visual Latent Target 当前经过轻量 target adapter

当前 visual latent branch 不是直接拿 raw pooled visual map 做 loss，
而是先经过：

- `visual_latent_target_proj(Pool(F_t^v))`

来对齐到 future head 的 hidden dim。

这层 target adapter 是当前实现里的工程化维度适配，
README 里不能把它写成“裸 V-JEPA pooled dim 直接监督”。

### 4.5 Innovation token 当前先走 deterministic normalization

当前 innovation residual 使用的是 deterministic RMS-style normalization。

这符合总纲里允许的 deterministic fallback。

尚未实现的是：

- future-head covariance prediction
- uncertainty-aware whitening residual

### 4.6 Point↔Visual geometry-first 主链现已全部接通

这轮和 `plan_readme_ray_geometry.md` 对齐后，当前代码已经实际落地：

- token-level `PE_proj`
- token-level `PE_ray`
- visibility-aware null projection branch
- `projective_compatibility`
- `projective_candidate_mask`
- relative projective attention bias `b_{t,m,u}^{proj}`
- anchor-level `L_anc^{pv}`
- bag-level `L_{pv}^{weak}`
- attention-derived `L_{focus}^{pv}`
- point-tactile `L_{pt}`
- `\tau_{pv}` 对 `L_{pv}^{weak}` 的 softmax temperature 接线

所以当前代码的真实状态是：

- geometry-first token conditioning：已实现
- patch-unit projective compatibility：已实现
- low-support routing support gates：已实现，`R_{anc}` 现在显式乘 `\omega_{m}^{route,p}\omega_{u}^{route,v}`
- sparse radius-neighborhood candidate mask：已实现
- attention-level geometry bias：已实现
- focus loss：已实现，默认从 fusion attention slice 读出
- point-tactile alignment loss：已实现，当前使用 fingertip-centered local bag
  - 候选集 = “指尖中心半径球 + 传感器前半空间”
  - 候选过少时回退到 KNN
  - point embedding 先做高斯加权池化，再与 tactile embedding 做分类式对齐
- `FusionTransformer`：当前已经是 bias-capable attention stack，并能导出平均 attention map

### 4.7 最终 tactile 部署默认值

当前训练入口 [`scripts/picf_core_train.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py) 已切到这组默认值：

- `--stride 4`
- `--max-points 1024`
- `--crop-radius-m 0.10`
- `--point-focus-sigma-m 0.03`
- `--pt-bag-radius-m` / `--pt-bag-sigma-m`
  - 现在默认不再硬编码成固定训练值
  - 当 `tactile_mode=encoder` 且提供 `tactile_fingertip_calibration.json` 时，训练会自动优先采用 calibration 产物里的：
    - `recommended_pt_bag_radius_m`
    - `recommended_pt_bag_sigma_m`
  - 只有在 calibration 没给推荐值时，才回退到基线 `0.045 / 0.015`
- `--pt-bag-kmin 32`
- `--pt-back-slack-m 0.008`
- `--p-align-on 0.55`
- `--p-align-off 0.35`

当 `tactile_mode=encoder` 时，训练入口现在会 fail fast 检查三份离线标定产物：

- `tactile_backgrounds.npz`
- `tactile_contact_stats.json`
- `tactile_fingertip_calibration.json`

当前云机 `task_ABC_D` 的最新短跑验收状态是：

- `rgb_latent_contact_v3` contact calibration 已通过阈值顺序、负样本尾部率和 observed contact rate 检查
- `tactile_final_verify_20_v9` 与 `tactile_final_smoke_100_v10` 都已跑通
- 合并这两条 metrics 后：
  - `mean tactile_contact_prob_mean ≈ 0.664`
  - `mean tactile_active_rate ≈ 0.167`
  - `loss_pt_nonzero_rate = 1.0`
  - `loss_action / loss_total ≈ 0.696`
- 最新完整几何标定 `tactile_calib_task_ABC_D_rgb_latent_full_v8` 已把 fingertip 前半空间质量抬过验收线：
  - `front_ratio = 0.6303`
  - `d_nn_trimmed_mean = 0.01023`
  - `recommended_pt_bag_radius_m = 0.03545`
  - `recommended_pt_bag_sigma_m = 0.01182`
- 一键验收命令：
  - `python scripts/picf_tactile_acceptance_audit.py --contact-stats <tactile_contact_stats.json> --fingertip-calibration <tactile_fingertip_calibration.json> --metrics <metrics.jsonl>`
- 指尖几何标定脚本 `scripts/calvin/precompute_tactile_contact_calibration.py` 现在会先为 top-contact 帧预缓存一次 merged point cloud 支持集，再搜索 `(u_open_local, o_local)`；
  不再按“候选参数数目 × 帧数”重复建点云，这样才能稳定完成完整几何标定，而不是只能依赖早期 pilot 结果
- 同时几何搜索默认只使用高置信接触的前 `24` 帧（`--geometry-max-top-frames 24`），避免把大量边缘接触/弱接触帧也平均进 objective，拖慢搜索且稀释 `front_ratio`
- 当前几何选优也不再仅靠弱加权和，而是优先满足 `target_front_ratio`（默认 `0.60`）：
  - 若存在 `front_ratio >= 0.60` 的候选，则先在这些候选里选 `d_nn_trimmed_mean` 最小者
  - 只有在没有任何候选达到目标前半空间比例时，才回退到“尽量提高 `front_ratio`，再兼顾 `d_nn`”的次优准则

也就是说，最终部署不再允许“有 AnyTouch checkpoint，但没有背景 / contact 阈值 / 指尖几何标定”的半配置训练。

当前 `contact_context` token 也已经保留双指 laterality：

- 输入 4 维现在是 `[p_left, p_right, max(p), mean(p)]`
- 不再把左右接触概率压成重复的全局统计量
- 这样即使 tactile token 还没到 `tactile_anchor_prob_on`，policy / context 仍然能区分“左指更像接触”还是“右指更像接触”
- pseudo-contact hysteresis 的上一时刻状态现在也优先继承 `tactile_contact_gate`，而不是更严格的 `tactile_anchor_mask`
  - 这样 EMA / on-off gate 的记忆语义与接触检测本身保持一致，不会被 fusion 门槛意外收紧
- `scripts/picf_core_train.py` 的 metrics 聚合也已补齐 `loss_semantic_future_aux`
  - 旧版训练日志里如果长期看到 `loss_semantic_future_aux = 0.0`，优先怀疑的是 logging bug，而不是语义分支完全没训练
  - 当前聚合逻辑已经直接从 `outputs["loss_semantic_future_aux"]` 统计该项
- `GatedCrossAttentionRead` 现在对整个 semantic read block 使用同一个 `cross_gate`
  - 旧实现只 gate 了 cross-attention 残差，但 FF 残差始终开启
  - 这会允许 read block 在 `cross_gate ≈ 0` 时仍然学习 prompt 无关的 query-only 变换，削弱 semantic gate 的设计意图
  - 现在 `cross_gate = 0` 时整个 read block 退化为 identity；只有 gate 打开后，semantic cross-attention 和 FF 才一起参与
  - 本地回归测试已覆盖：`test_gated_cross_attention_read_is_identity_when_gate_is_closed_even_if_ff_learns`
  - 用同一个 `v24/5000` checkpoint 做 patched 单窗口诊断时，`global_minus_physical_norm` 会从旧实现的约 `0.74` 降到约 `0.006`
    - 这说明旧实现里确实存在 prompt 无关的 semantic bypass
    - 如果 patched 后 action 仍几乎不变，则说明 `5000` checkpoint 的问题更接近“semantic read gate 尚未打开”，而不是“prompt 根本没进模型”

### 4.8 当前 `L_{pv}^{weak}` 是代码对齐版近似

总纲里更理想的 `L_{pv}^{weak}` 会排除 projective neighborhood 高度重叠的 visual negatives。
当前代码先用了更简单、数值稳定的近似：

- 正例仍然是 ray-bag / projective neighborhood pooled point embedding
- negatives 当前是“其余 visual patches”
- `\tau_{pv}` 当前已经真实接入 logits temperature
- 还没有做 overlap-aware negative exclusion

这是有意保留的工程近似，不是文档漏写。

另外，当前 `sigma_proj_patches` 直接作用在连续 patch-grid 坐标上，
也就是和文档里 “`1.5 patch`” 的单位语义完全一致。
归一化 grid 坐标现在只保留为辅助导出，不再参与主 compatibility 计算。

对于 soft depth factor，当前代码还有一条显式保护：

- 如果某个可见点的逐点 depth sample 本身无效，例如采样到 NaN / 缺测区域
- 该点会回退到 `depth_factor = 1`
- 不会因为“无效深度”被误当成强几何不一致而额外压低 compatibility

当前还保留的一条工程近似是：

- `projective_candidate_mask` 已经通过连续 patch-grid 上的 sparse radius neighborhood 与 `G_{t,m,u}^{proj}>\tau_{proj}` 共同构造
- 但 `projective_compatibility` 本身仍是 dense 计算，再与 sparse neighborhood 相交
- 也就是说，当前是“dense compatibility + sparse candidate mask”的实现，而不是完全稀疏的 compatibility builder

### 4.9 Observation / Anchor read 当前用 residual cross-attention 近似实现

总纲里 observation-anchor read 和 persistent-anchor evidence read 记成了显式 `GRU_obs / GRU_anc` 更新。
当前代码实现采用的是更轻量的工程近似：

- `CrossAttentionRead`
- 残差更新
- 后接小型 self-attention / feed-forward

这两条路径当前仍满足“query 反复从当前 token / anchor 集合读证据”的主语义，
但不是和总纲完全逐式同构的 GRU 单元。

### 4.9 当前 core 在全局坐标系里裁剪，不再把 core state 写在局部坐标系

当前 `PicfFullCore` 只消费 `point_set.xyz_world`，并在全局坐标系里按 `G_t[:3,3]` 附近做 subset。

旧的 `frame_context` / `local_frame` 辅助结构仍然在仓库里保留，主要用于 legacy 路径和 builder 兼容，但当前 core 的 posterior state 不再定义在单独的 local frame 里。

### 4.10 训练侧 `t -> t+1` targets 已经闭合到 core transition loss，并已有独立长期训练入口

这是当前 README 里必须写清楚的一条。

现在已经有的是：

- current-step real targets for runtime innovation construction
- current-step future prediction heads 的张量接口
- 对外公开的 [`PicfFullCore.extract_targets(...)`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
- 一步训练损失 [`compute_transition_loss(...)`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/training.py)
- 一个可直接在真实 CALVIN 数据上跑通的训练 smoke：
  [`scripts/picf_core_train_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train_smoke.py)
- 一个可直接做 checkpoint / resume 的长期训练脚本：
  [`scripts/picf_core_train.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py)

现在还没有真正接完的是：

- 更完整的 multi-step / multi-loss 训练计划
- 正式多卡长程收敛与超参扫描

所以当前代码更准确的定位是：

- inference/state graph 已闭合
- one-step training loss graph 已闭合
- 独立长期训练入口已闭合
- foundation-weight 单卡 / resume / 最小 DDP 已闭合
- 但通用训练入口和最终多卡长程训练配方还不是最终版

### 4.10b `scripts/picf_core_train.py` 当前真实训练 contract

这条脚本现在已经是新 core 的正式长期训练入口，而且当前 contract 已经同时覆盖 `stub` 与 `foundation` 两条配置：

- 数据源不是旧 `create_data_loader(pi05_calvin_sonata)`，而是脚本内部的 `_CalvinTransitionSource`
- `_CalvinTransitionSource` 直接使用 `CalvinLangSegmentDataset(..., action_horizon=1, sample_within_segment=False)`
- 每个训练 step 会随机抽一段 CALVIN segment window，窗口长度是 `unroll_steps + 1`
- 同一个 window 内，脚本会按 `t -> t+1` 顺序重复调用 `PicfFullCore.step(...)`，并在每个 transition 上计算 `compute_transition_loss(...)`
- point / visual / tactile backbone 现在都可配置：
  - `--point-backbone rgb|sonata`
  - `--visual-mode stub|encoder`
  - `--tactile-mode stub|encoder`
  - `--use-tactile` 控制是否把 CALVIN 的 tactile packet 读入训练图
- `--use-foundation-backbones` 是推荐的 cloud 启动配置；它会把 launcher 切到：
  - `point_backbone = sonata`
  - `visual_mode = encoder`
  - `tactile_mode = encoder`
  - `use_tactile = true`
  - 并自动优先查找默认 checkpoint：
    - `checkpoints/foundation/vjepa2_1/...`
    - `checkpoints/foundation/anytouch2/checkpoint-4frames.pth`
    - `src/pretrain/SpatialLM_Sonata_encoder.pth`
- 当前 `_CalvinTransitionSource` 已经会在 `use_tactile=True` 时读取：
  - `rgb_tactile`
  - `depth_tactile`
  - `depth_gripper`
  - 并通过 `_calvin_tactile_packet(...)` 拆成两路 `PicfTactilePacket`
  - 该 packet 现在还会带 calibrated tactile background，并用 TCP-local 动态指尖中心替代固定 `±1cm`
- visual path 现在分两种：
  - `visual_mode=stub`：走 `_rgb_visual_override(...) + _NullVisualEncoder()`
  - `visual_mode=encoder`：真实实例化 `Vjepa2VisualEncoder`
- tactile path 现在分两种：
  - `tactile_mode=stub`：走 `_NullTactileEncoder()`
  - `tactile_mode=encoder`：真实实例化 `AnyTouch2TactileEncoder`
  - encoder 路径在 CALVIN 上默认要求 calibrated background；没有 background 时不再回退到 `clip[0]`
- point path 现在分两种：
  - `point_backbone=rgb`：轻量 RGB point feature
  - `point_backbone=sonata`：真实实例化 `SonataPointFeatureExtractor`
- 为了避免在真实 V-JEPA 路径里提 target 时污染视觉历史，
  [`PicfFullCore.extract_targets(...)`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
  现在会在内部对 `clip_buffer` 做 `snapshot()/restore()`

因此，这条长期训练脚本当前的准确定位是：

- 它已经真实闭合了 v0.4.8 新 core 的训练图
- 它已经支持 checkpoint / resume / foundation-weight 单卡开跑
- 它已经在云机上做过一次 `torchrun` 双卡 DDP 最小回归
- 当 checkpoint 缺失或只想做结构 smoke 时，仍然可以退回 `stub` 模式

当前准确口径不再是“foundation weights 只在 ad-hoc snippet 里能用”，而是：

- foundation weights 已经接进 `scripts/picf_core_train.py`
- CALVIN tactile 也已经接进长期 trainer
- `stub` 模式保留为本地无权重环境的回退路径
- foundation-weight 短程训练已经在云机上连续跑过 `5` 个 step，未出现：
  - NaN / Inf
  - checkpoint 写入失败
  - DDP 初始化失败
  - tactile / point 分支掉空

### 4.10c `scripts/picf_core_train.py` 的输出目录、进度条、wandb 与恢复语义

当前输出目录固定为：

- `<checkpoint-base-dir>/picf_core/<exp-name>/args.json`
- `<checkpoint-base-dir>/picf_core/<exp-name>/metrics.jsonl`
- `<checkpoint-base-dir>/picf_core/<exp-name>/diagnostics/<step>/...`
- `<checkpoint-base-dir>/picf_core/<exp-name>/wandb_id.txt`
- `<checkpoint-base-dir>/picf_core/<exp-name>/latest.pt`
- `<checkpoint-base-dir>/picf_core/<exp-name>/<N>/model.pt`
- `<checkpoint-base-dir>/picf_core/<exp-name>/<N>/optimizer.pt`
- `<checkpoint-base-dir>/picf_core/<exp-name>/<N>/metadata.pt`

其中：

- `args.json`：记录本次启动参数
- `metrics.jsonl`：按 `log_interval` 追加 step 级 JSON 指标
- `diagnostics/<step>/...`：按 `diagnostic_interval` 保存当前训练 window 的静态相机可视化
  - `gt_static_t*.png`：真实 CALVIN 当前 / 未来静态帧
  - `pred_physical_t*.png`：physical future cache 的 `visual_real` 预测上采样图
  - `pred_semantic_t*.png`：semantic-conditioned future readout 的 `visual_real` 预测上采样图
  - `gt_window_static.gif` / `pred_*_window_static.gif`：短窗口 GIF
  - `compare_grid.png`：`current | pred_physical | pred_semantic | target_next` 对比图
- `wandb_id.txt`：记录当前实验对应的 wandb run id，供 `--resume` 时继续同一 run
- `latest.pt`：轻量 latest 指针，记录最近一次 checkpoint 的 step 和目录
- `<N>/...`：按完成步数编号的原子 checkpoint 目录；当前一步 checkpoint 内包含：
  - `model.pt`
  - `optimizer.pt`
  - `metadata.pt`

这里必须明确：

- 这些 `pred_*` 图不是全分辨率视频生成结果
- 它们来自 PICF visual future head 的 `visual_real` 分支
- 当前默认是 `visual_real_grid = 4`，所以本质是 **4x4 RGB coarse prediction 的上采样可视化**
- 它适合看“模型大致在期待什么”，不等价于 CALVIN evaluator 的 policy rollout video

恢复语义是：

- `--resume`：默认优先从同一实验目录下最大的数字步目录恢复；若目录不存在，则回退读取 `latest.pt`
- `--resume-checkpoint <path>`：显式从指定 checkpoint 恢复
- `--overwrite`：清空现有实验目录并重新开始；不得与 `--resume` 同时使用

这里还有一条必须写进交接文档的细节：

- `metrics.jsonl` 当前是 append 模式
- 如果复用同一个 `exp-name` 重新起一个非 `--resume` 训练，日志里会继续追加，可能出现重复 step id
- 所以想要“干净的一条新曲线”，应当：
  - 用新的 `exp-name`
  - 或显式传 `--overwrite`

训练器当前已经对齐到原版 `scripts/train_pytorch.py` 的几个关键交互点：

- 主进程默认启用 `tqdm` 实时进度条，按 step 更新当前 `loss / lr / step / ETA`
- `--log-interval` 只控制：
  - `metrics.jsonl` 追加
  - 控制台 JSON 行输出
  - wandb scalar logging
- `--save-interval` 默认已经调成 `5000`
- wandb 默认开启：
  - `--project-name` 默认 `openpi`
  - `--wandb-run-name` 默认复用 `exp-name`
  - `--wandb-mode` 支持 `online / offline / disabled`，当前长期 trainer 默认 `offline`
  - `--no-wandb` 可显式关闭
- `--no-progress` 可关闭进度条，适合纯日志环境或回归脚本
- 长期 trainer 的 core 结构默认值现在直接对齐 `PicfCoreConfig` / v0.4.8 spec：
  - `persistent_anchors=16`
  - `observation_anchors=24`
  - `hidden_dim=384`
  - `posterior_hidden_dim=384`
  - `latent_dim=112`
  - `innovation_dim=384`
  - `control_dim=384`
  - `semantic_dim=2048`
  - `semantic_cross_dim=512`
  - `future_hidden_dim=384`
  - `fusion_layers=4`
  - `posterior_layers=2`
  - `predictive_layers=2`
  - `control_layers=2`
  - `predictive_semantic_reads=2`
  - `control_semantic_reads=2`
  - `attention_heads=8`
  - `future_vote_heads=4`
- 也就是说：
  - 现在默认值就是正式训练值
  - 任何更小的 `hidden_dim / anchor / layer` 组合都应视为显式 smoke 覆盖，而不是正式训练默认

和旧 `pi0.5 / train_pytorch.py` 训练口径对照后，这里还需要明确三条参数约束：

- 新 trainer 的“全局 batch”定义是：
  - `effective_global_batch = world_size * accum_steps`
  - 每个 rank 每个 micro-step 处理一个 `_TransitionWindow`
- 因此：
  - 双卡 DDP + `accum_steps=1` => `effective_global_batch=2`
  - 这与旧 `pi0.5` 云上命令里的 `batch-size=2` 是同一量级口径
  - 如果退回单卡、但还想保持和旧双卡命令相同的全局 batch，应该改成 `--accum-steps 2`
- 在当前两张 `A100 40GB` 的真实回归里：
  - 双卡 DDP + `accum_steps=2` => `effective_global_batch=4`：已验证通过，当前是推荐起跑档
  - 双卡 DDP + `accum_steps=4` => `effective_global_batch=8`：已验证通过，是当前“已验证的更大 batch”选项
  - `accum_steps>4` 还没有做同强度回归，不应写成“已验证”
- 当前更推荐增 `accum_steps`，而不是改 per-rank micro-batch：
  - 每个 rank 每个 micro-step 仍只处理一个 `_TransitionWindow`
  - 梯度累积几乎不增加峰值激活显存
  - 这更贴近旧 `pi0.5` “在安全范围内把全局 batch 做大”的经验
- 当前 `PICF core` 的数值口径与旧 `pi0.5` 不完全相同：
  - 旧 `pi0.5` 训练主模型常用 `bfloat16`
  - 新 `PICF core` 目前验证稳定的配置是：
    - core 主干默认 `float32`
    - `V-JEPA` trainable 路径使用 `CUDA autocast(bfloat16)` 前向
    - `Sonata` / `AnyTouch` / `PaliGemma` 当前都允许 cotrain，并按各自已验证的稳定 dtype 路径运行
  - 这不是遗漏，而是当前新 core 的已验证工程配置；如果后续要压显存，再单独做 mixed-precision 回归
- 在 `PaliGemma + DDP + accum_steps>1` 时，trainer 会自动关闭 semantic gradient checkpointing：
  - 这是为了解掉 HF PaliGemma reentrant checkpoint 在 repeated backward 下的 `mark ready twice`
  - 这不改变 PICF 数学，只是当前 DDP 累积训练的工程稳定性规则
- warmup 口径也已改回 spec：
  - `--warmup-steps` 若不显式传入，则自动取 `round(0.02 * num_train_steps)`
  - 因而默认 `30000` step 训练会使用 `600` warmup steps
  - 只有在做对照实验时才建议显式覆盖这个默认值

### 4.11 当前训练调试接口已经有哪些

当前新 core 的训练调试信息主要分三层。

第一层是 step 级 debug 字段，来自
[`PicfCoreOutput.debug`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/contracts.py) /
[`PicfFullCore.step(...)`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)：

- `num_point_tokens`
- `num_visual_tokens`
- `num_tactile_tokens`
- `support_mass_mean`
- `active_alpha_sum`
- `innovation_norm`
- `hold_triggered`
- `mean_point_visibility`
- `projective_candidate_edges`
- `projective_candidate_density`

这些字段的用途是：

- `num_point_tokens == 0`：
  当前 `xyzrgb` contract / ROI subset 基本有问题；首步这是 hard failure
- `num_visual_tokens == 0`：
  当前视觉 clip / override / encoder 路径没进来
- `support_mass_mean` 长期接近 `0`：
  observation→persistent binding 没吸住，通常先查 point token、seed、dustbin
- `active_alpha_sum` 长期塌到接近 `0`：
  recycle / activity 失衡，posterior 会进入 hold 风险区
- `innovation_norm` 首步应为 `0`，后续步若观测变化应大于 `0`
- `hold_triggered == 1`：
  当前步被 runtime supervisor 判为不安全，继续先看 `point_contract_ok / sync_valid / uncertainty / innovation`
- `mean_point_visibility` 长期接近 `0`：
  先查相机标定、点云坐标系、`CameraModel`、以及是否把点送到了错误 frame
- `projective_candidate_density` 接近 `0`：
  通常是投影错位、可见性全灭、或 `tau_proj` / `sigma_proj_patches` 设得太苛刻
- `projective_candidate_density` 接近 `1`：
  通常说明 visual grid 太粗、`sigma_proj_patches` 过大、或 candidate threshold 太松

第二层是一步训练损失分解，来自
[`PicfTransitionLossBreakdown`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/training.py)：

- `total`
- `action`
- `action_pos`
- `action_rot`
- `action_gripper`
- `visual_latent`
- `visual_real`
- `tactile_real`
- `point_real`
- `alignment`
- `anchor_pv`
- `pv_weak`
- `focus_pv`
- `availability`

这里的调试解释应按下面看：

- `availability[i] == 0`：
  该分支当前 target 不可用，此时对应 loss 为 `0` 是正常的
- `visual_latent` 恒为 `0` 且 `availability_visual_latent == 1`：
  一般说明 next-step visual map override / visual encoder 路径没接对
- `tactile_real` 恒为 `0`：
  先区分是本机用 tactile stub，还是 replay 里当前帧本来没有有效 tactile packet
- `point_real` 爆炸：
  通常先查 ROI crop、occupancy grid 尺度、point subset 是否空或过 sparse
- `action_*` 三项都为 `0`：
  常见原因是没传 `action_target`
- `alignment == 0` 且当前 visual / point availability 都正常：
  先查 `projective_candidate_edges`
- `anchor_pv` 有值但 `pv_weak` 恒为 `0`：
  往往说明当前 candidate edge 太少，或 visual token 数不足以形成有效 negatives
- `focus_pv == 0`：
  当前不再是默认预期；若长期为 `0`，优先查 fusion attention map、candidate edge 是否为空，或 visual→point slice 是否被错误 mask
- `pt == 0`：
  先区分是当前样本 tactile token 不可用，还是 tactile contact gate 接近 `0`
  当前 `L_pt` 不再在“无接触证据”时强行构造正例。
  优先级是：
  - 显式接触：`force_vec / indent_depth_m / tactile_pressure`
  - 否则退回基于 tactile history 的 pseudo-contact gate
  - 当前 pseudo-contact gate 采用 fast-on / slow-off：
    - turn-on 看当前 contact evidence，避免首帧真实接触被 EMA 延迟吞掉
    - release 仍保留 EMA 路径，避免 gate chatter
  - no-contact tactile 仍然保留在 perception/world-model 路径里，但不会再进入 fusion / anchor / `L_pt` 几何正例
  对 CALVIN 这类没有显式接触标注的数据，`loss_pt` 可能长期稀疏，但不应再因为无条件正例而卡在 `log(2)` 附近乱学。

第三层是 smoke 脚本的 JSON 输出，来自
[`scripts/picf_core_train_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train_smoke.py)：

- `device`
- `device_name`
- `cuda_available`
- `cuda_runtime_available`
- `segment_index`
- `current_step_id`
- `next_step_id`
- `loss_total`
- `loss_action`
- `loss_visual_latent`
- `loss_visual_real`
- `loss_tactile_real`
- `loss_point_real`
- `loss_alignment`
- `loss_anchor_pv`
- `loss_pv_weak`
- `loss_focus_pv`
- `loss_pt`
- `availability_visual_latent`
- `availability_visual_real`
- `availability_tactile_real`
- `availability_point_real`
- `action_grad_norm`
- `point_grad_norm`
- `projective_candidate_edges`
- `projective_candidate_density`
- `mean_point_visibility`
- `mean_point_route_gate`
- `mean_visual_route_gate`
- `mean_point_route_support`
- `mean_visual_route_support`

其中当前字段语义是：

- `cuda_available`：本次 smoke 是否实际在 CUDA 设备上执行
- `cuda_runtime_available`：脚本启动时、顶层 import 阶段看到的 CUDA 运行时可见性

若 `cuda_runtime_available` 与独立执行的 `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` 不一致，
以独立探测为准；当前这个字段只作为 smoke 诊断辅助项，不参与任何核心数学或训练图逻辑。

健康 smoke 的最低标准是：

- `loss_total` 为有限正数
- 至少 visual / point 分支 availability 为 `1`
- `action_grad_norm > 0`
- `point_grad_norm > 0`
- `optimizer.step()` 能完成

如果只想最快判断新 core 训练图是不是活的，
优先看这四个量：

- `loss_total`
- `availability_point_real`
- `action_grad_norm`
- `point_grad_norm`


## 5. 数据与接口约束

### 5.1 Point Contract

当前 core 明确要求 point input 满足 `xyzrgb` 契约。

对应类型仍然是 [`src/openpi/picf/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/contracts.py) 里的 `PicfPointCloudFrame`：

- `xyz_world` 必需
- `rgb` 必需
- `normal_world` 目前仍保留为 side field

说明：

- normals 仍然在 data contract 中保留，便于 legacy / geometry helper 复用
- 但 Sonata backbone 主输入已经严格收紧为 `xyz+rgb`
- 当前 core 也不再把 normals 当 formal posterior state

### 5.2 Runtime Meta

`RuntimeMeta` 现在在保留旧字段的同时，新增了：

- `visual_available`
- `tactile_available`
- `point_contract_ok`
- `sync_valid`

旧脚本仍可继续读原有字段，新 core 则直接使用这些更贴近 v0.4.8 的标记。

### 5.3 Semantic Input

语言默认不直接从 core 内部“隐式”拉起 PaliGemma，但当前正式 trainer 已经把真实 semantic side path 接进来了。

当前语义路径支持三种方式：

- 直接传 `semantic_override`
- 传入真实 PaliGemma outputs，由 `PaliGemmaSemanticWrapper` 做摘要
- 由长期训练入口 [`scripts/picf_core_train.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py) 显式实例化
  [`PaliGemmaSemanticEncoder`](/home/siyuanyue/Documents/openpi/src/openpi/picf/paligemma/wrapper.py)
  并对每个当前 observation 计算语义 token 流

当前真实训练路径是：

- foundation 模式下，launcher 会启用：
  - `semantic_mode = paligemma`
  - `semantic_source = auto`
  - `semantic_trainable = True`
  - `semantic_use_gripper = True`
- `semantic_source = auto` 会优先使用本地 `pi05_base_pytorch/model.safetensors`
- 若本地 checkpoint 不存在，则回退到 HF `PaliGemmaForConditionalGeneration`
- 训练器每步都会对当前 observation 计算 `semantic_override`，再传入 `PicfFullCore.step(...)`
- 这条语义路径当前已经在云机双卡 `torchrun` 下按 full-token 形态通过 smoke，不再只是本地结构验证

这和 [`plan_readme_ray_geometry.md`](/home/siyuanyue/Documents/openpi/plan_readme_ray_geometry.md) 的 language-late 口径保持一致：

- semantic tokens 不进入 current posterior
- semantic tokens 只进入 posterior 之后的 predictive / control stage
- 同一帧只改 `semantic_override` 时，`posterior_mu_diff = 0.0`、`posterior_sigma_diff = 0.0`

当前这条语义侧路的实现口径需要明确写清：

- 当前 trainer 接入的是真实 `PaliGemma`，而且 backbone 可 cotrain
- 当前 downstream 使用的是 **完整的 semantic token stream**
  - `PaliGemma` 文本 hidden states 中所有有效 token 会被保留
  - `PaliGemma` 图像 hidden states 中所有有效 token 也会被保留
  - 这些 token 保持 `semantic_dim=2048` 原生宽度
  - 它们不会先被压到 `hidden_dim`
- 当前 downstream 使用的是 **完整 semantic token stream**
  - `PaliGemma` 文本 hidden states 中所有有效 token 会被保留
  - `PaliGemma` 图像 hidden states 中所有有效 token 也会被保留
  - 这些 token 保持 `semantic_dim=2048` 原生宽度
  - 它们在 posterior 固定之后，会通过 `semantic_prefix_proj` 投到 world hidden width，
    并作为 semantic prefix 直接并入 control / predictive 主干
- 在 posterior 之后，先形成 world token 流：
  - `posterior.tokens`
  - `innovation_token`
  - `proprio_token`
- control 路径会再追加：
  - `semantic_prefix_tokens`
  - `control_query_tokens`
- predictive 分支额外再拼入：
  - `posterior.global_post`
  - `action_cond_token`
- predictive 分支现在还会显式拆成两份 cache：
  - `physical_prediction_cache`
    - 只来自 semantic 进入前的 language-free world stream
    - 下一步 innovation **只允许**读取这份 cache
  - `prediction_cache`
    - 来自并入 `semantic_prefix_tokens` 之后的 semantic-conditioned future readout
    - 它服务 future head，不反写 posterior / carried prior / innovation base
- 为降低 future head 走 semantic shortcut 的风险，当前 predictive 分支默认增加：
  - `predictive_semantic_dropout_prob = 0.1`
  - 它只作用在 posterior-late semantic prefix token 流上，不作用于 current posterior
- 这意味着当前实现更接近旧 `pi0.5` 的 full-token PaliGemma 语义能力，同时继续满足：
  - current posterior language-free
  - semantic side path language-late
  - anchor / posterior tokens 与 semantic tokens 在 downstream 是平级流，但不要求预先同宽
  - `semantic_cross_dim` / `predictive_semantic_reads` / `control_semantic_reads`
    目前只为 checkpoint / CLI 兼容保留，不再改变当前 semantic-prefix 主路径
- 2026-04-10 本地复核新增通过：
  - `66 passed` 核心回归
  - CPU 一步训练 smoke
  - 新增红线回归：
    - 改 semantic 不改 `physical_prediction_cache`
    - 改 previous semantic 不改下一步 innovation
    - semantic future auxiliary loss 会把 predictive cross-attn 保持在图中
  - 训练稳定性硬化：
    - 已移除训练主路径里残留的 GPU 布尔高级索引：
      - `projective_attention_bias` 的 `candidate_mask` gather/scatter
      - `PaliGemma` full-token path 的 `[valid]` token slicing
      - predictive semantic memory 的 `[keep]` token dropping
      - posterior update 里的 `S[valid] / a[valid]` 写回
    - 当前实现改为：
      - dense masked compute
      - right-padded prefix slicing
      - dense token dropout
      - `torch.where` 式状态更新
  - 云机 replay 复核：
    - `r6/r7` 时期对少量历史崩溃窗口序列做过 `scripts/picf_replay_windows.py` 复核
    - 这类“短序列 replay 通过”**不能**视为问题已经消失
    - 后续真实长训练 `r10` 仍在 `step≈1200` 复现了 `ScatterGatherKernel.cu:144 index out of bounds`
    - 当前更严格的排查方式是：
      - 真实双卡训练复现
      - exact-prefix replay
      - 且 `scripts/picf_replay_windows.py` 中 `rng_rank` 与 `rank_seed` 必须对齐；否则 flat-index 序列虽然一致，但模型/dropout RNG 轨迹并不一致
      - 且 exact-prefix replay 不能把 `rng_num_windows` 误当成“raw RNG draws 数”；真实训练会把 retryable 空窗直接丢掉并继续抽样，所以 replay 必须按“accepted windows + retryable resample”去重放
      - 在 `r38` 的 rank1 复盘里，训练日志中的 accepted `step=455..458` 对应 raw RNG draws `463..466`，说明前面已经发生了 `8` 次 retryable resample；如果 replay 不模拟这层 resample，就会把参数轨迹对错
      - 现已支持 `--override-sonata-disable-flash true/false` 做 Sonata flash A/B
      - 现已支持 `--split-backward-from-step N`，在目标区间把 `loss_total.backward()` 拆成各个 loss 组件逐项同步执行
      - 现已支持 `--save-checkpoint-every N --save-checkpoint-dir ...`，为 long-prefix replay 留中间 checkpoint，避免每次都从 step 1 重放
      - replay checkpoint 现已额外保存 `replay_rng.pt`；resume 时如果没有这个文件，就不能把该条恢复轨迹视为“严格 exact replay”
      - replay checkpoint 现已支持 `--max-checkpoints K` 自动裁剪旧 step 目录；云机调试默认应保持 `K<=5`，避免再次把 `/tmp` 写爆
      - replay 还必须镜像训练主循环的 **per-step LR schedule** 和 **grad clipping**；否则即便 flat-index 前缀对齐，也会因为优化器语义不同而过早偏离真实参数轨迹
      - 当 `--rng-num-windows` 与 `--checkpoint` 一起使用时，若 checkpoint 目录中存在 `replay_state.json`，replay 会恢复到保存时的 accepted-step / raw-draw / retryable-skip 计数，再继续往后重放；这一步对带 resample 的 exact-prefix replay 很关键，否则从 step 400 之后继续时会再次把 RNG 轨迹跑偏
      - 对所有 `logvar -> variance` 路径，不能写成 `torch.clamp(torch.exp(logvar), min=..., max=...)`
        - 在 `logvar` 很大时，前向会先溢出成 `inf`，再被 `clamp` 截回有限值；但反向会命中 `ExpBackward0` 的 `0 * inf -> NaN`
        - 当前实现已统一改成“先在 log-space 截断到 `[log(sigma_min2), log(sigma_max2)]`，再 `exp`”，这样前向语义等价，反向不会再制造假性非有限梯度
      - 运行策略应区分两种模式：
        - `split-backward` / `diagnose-nonfinite-by-component` 是**窄区间诊断模式**，只在已经把坏点压缩到很小区间时打开；它会显著拖慢 replay，不适合作为常规 1500+/12500+ 长跑默认配置
        - 常规稳定性验证 / 长跑应优先使用 exact-prefix replay 或真实训练，但默认关闭逐组件诊断；一旦再次出现非有限梯度，再从最近 replay checkpoint 回退到窄区间重放并打开诊断
      - 本地和云机应保持关键调试文件同步，至少包括：
        - `src/openpi/picf/core/pipeline.py`
        - `scripts/picf_replay_windows.py`
        - 对应测试和 README
      - 单窗 probe 只能说明“这个窗口本身是否天然有毒”；例如 `flat_index=797970`（训练日志中 point counts 为 `(3,1,3)`）单独做 split-backward probe 可以完全通过，所以根因更可能是“真实前缀参数状态 + retryable resample 后的 accepted-window 轨迹”
    - 当前权重结论：
      - 在 crash 定位期间**不要继续上调 action loss 权重**
      - 现有训练实现里，`loss_action = 2.0 * L_pos + 2.0 * L_rot + 2.0 * L_gripper`
      - 旧云机 `1500` step replay 的最近 `300` 条 loss 统计里，`loss_action / loss_total ≈ 0.88`
      - 同一窗口里 `loss_pt / loss_total ≈ 0.0018`；所以当前主矛盾不是“action 权重太小”，而是旧 tactile 配置几乎没有真正激活 point-tactile grounding
      - 当前 baseline 仍保持：
        - `lambda_action_pos = 2.0`
        - `lambda_action_rot = 2.0`
        - `lambda_action_gripper = 2.0`
        - `lambda_visual_latent = 0.2`
        - `lambda_visual_real = 0.1`
        - `lambda_tactile_real = 0.3`
        - `lambda_point_real = 0.3`
        - `lambda_semantic_future_aux = 0.25`
        - `lambda_anchor_pv = 0.1`
        - `lambda_pv_weak = 0.02`
        - `lambda_focus_pv = 0.0`
        - `lambda_pt = 1.0`
      - 当前没有证据支持继续上调 action 权重；在真实短跑验收里，`loss_action / loss_total ≈ 0.696`，action 已经是主导项
      - 对最终 tactile 方案，应该先看 `tactile_active_rate / pt_bag_nonempty_rate / loss_pt_nonzero_rate`，而不是先把 action 权重继续上调
      - 训练日志现在已经真实输出 `tactile_contact_prob_mean` 和 `tactile_active_rate`
      - 如需长期审计，可直接运行：`python scripts/picf_loss_audit.py --log <jsonl_or_log> --tail 300`
      - 如需做 action 权重反事实分析，可附加：`--action-pos-weight <w> --action-rot-weight <w> --action-gripper-weight <w>`
      - 如需把 calibration 产物与训练 metrics 一起过验收，可运行：`python scripts/picf_tactile_acceptance_audit.py --contact-stats <tactile_contact_stats.json> --fingertip-calibration <tactile_fingertip_calibration.json> --metrics <metrics.jsonl>`
      - 截至 `2026-04-12`，云机真实 `task_ABC_D` 上的最新短跑验收结论是：
        - `tactile_final_verify_20_v9` + `tactile_final_smoke_100_v10` 都已跑通
        - `loss_pt_nonzero_rate = 1.0`
        - `mean tactile_active_rate ≈ 0.167`
        - `mean tactile_contact_prob_mean ≈ 0.664`
        - 最新完整几何 calibration `v8` 已通过 fingertip 前半空间门槛：`front_ratio = 0.6303`
        - 当训练未显式指定 `--pt-bag-radius-m/--pt-bag-sigma-m` 时，入口现在会自动采用 calibration 提供的推荐值，而不是继续硬编码 `0.045 / 0.015`
      - `tactile_real` 当前应理解为“摘要辅助头”而不是“左右传感器逐路重建头”：
        - 每个 tactile sensor 的 RGB 会先转灰度并池化到 `4x4`
        - 然后对所有有效 sensor 的 pooled map 求平均
        - 再拼接 contact / force / indent / pressure / sensor-count fraction / pose xyz 这类 aux
        - 因此它适合作为 world-model 的弱 tactile state target，不应被解读成强 per-sensor tactile future reconstruction
- 验证级别说明：
- 以上结论来自代码路径审计、回归测试、以及云机双卡 smoke
- 它们足以支持“当前工程实现满足既定数学契约”的判断
- 但这**不是** Coq / Lean / TLA+ / model checking 意义下的机器校验形式化证明
- 当前更硬的工程规格已单独收敛到：
  - [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
  - 后续如果修改 `posterior` / `innovation` / `predictive cache` 的读写边，必须同步更新这份规格
  - 一键脚本入口：
    - `python scripts/verify_picf_contract.py`

如果以上两条显式语义路径都没有，core 才会回退到零语义 token。


## 6. Legacy 路径的地位

下面这些目录仍然保留，但已经不再代表当前主线：

- [`src/openpi/picf/scaffold`](/home/siyuanyue/Documents/openpi/src/openpi/picf/scaffold)
- [`src/openpi/picf/posterior`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior)

它们现在的角色是：

- regression reference
- old invariant / acceptance scripts 的承载层
- 某些低层点云 /视觉 helper 的复用来源

不要再把这些模块当作 v0.4.8 的主状态接口。

这里的 `legacy` 不是描述性措辞，而是当前代码引用面的事实：

- `src/openpi/picf/posterior/` 与 `src/openpi/picf/scaffold/` 的直接调用者，主要只剩 `scripts/posterior/*`、`scripts/scaffold/*` 和对应单测
- 当前主训练入口 [`scripts/train_pytorch.py`](/home/siyuanyue/Documents/openpi/scripts/train_pytorch.py) 不导入 `openpi.picf.posterior` 或 `openpi.picf.scaffold`
- 当前新主线 [`src/openpi/picf/core/`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core) 仍然没有并进 `scripts/train_pytorch.py`
- 但它现在已经有独立训练入口：
  [`scripts/picf_core_train.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py)
- 当前这条独立入口已经在真实 CALVIN 上完成：
  - `dir + cuda` 起步
  - `resume from latest.pt`
  - `zip + cuda` 起步


## 7. 关键文件

主实现：

- [`src/openpi/picf/core/config.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/config.py)
- [`src/openpi/picf/core/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/contracts.py)
- [`src/openpi/picf/core/pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline.py)
- [`src/openpi/picf/core/training.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/training.py)

输入封装与 backbones：

- [`src/openpi/picf/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/contracts.py)
- [`src/openpi/picf/pointcloud_picf.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/pointcloud_picf.py)
- [`src/openpi/picf/sonata/wrapper.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/sonata/wrapper.py)
- [`src/openpi/picf/vjepa/wrapper.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/vjepa/wrapper.py)
- [`src/openpi/picf/anytouch/wrapper.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/anytouch/wrapper.py)

回归测试：

- [`src/openpi/picf/core/pipeline_test.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/pipeline_test.py)
- [`src/openpi/picf/core/training_test.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/core/training_test.py)
- [`src/openpi/picf/pointcloud_picf_test.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/pointcloud_picf_test.py)

训练 smoke：

- [`scripts/picf_core_train_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train_smoke.py)
- [`scripts/picf_core_train.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train.py)


## 8. 当前核查结果

本次修改后，已经实际核查通过：

- `pytest -q src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py src/openpi/picf/pointcloud_picf_test.py src/openpi/training/data_loader_test.py`
- `from openpi.picf.core import ...` 顶层导入

那一组较早的最小回归结果是：

- `31 passed`

当前更完整的 core + trainer 回归见上文 5.3，最新为 `54 passed`。

当前 `pipeline_test.py` 覆盖的核心约束包括：

- unified token field / observation anchors / posterior anchors 基本张量闭合
- language-late：改语言不改 current posterior
- previous-step `physical_prediction_cache` 形成 current innovation
- semantic cache reuse
- first-step `xyzrgb` contract 检查
- point contract 失效后的 hold / zero-point-token 行为
- projective geometry 与 legacy `_project_world_points()` / `_scale_to_grid()` 一致
- `camera_model` 缺失时，visual override 仍稳定走 null projection branch
- behind-camera 点不会在 `PE_proj` 路径产生 NaN / Inf
- persistent anchor 的 `S` 更新包含 observation-center scatter 项
- 可见点若采样到无效深度，不会被 projective compatibility 错误压成 0

除此之外，这次我还额外人工核查了三件数学/工程一致性问题：

- 语言路径不会改变 current posterior
- prior/context 现在优先读取 `previous.executed_action`，而不是 `previous.policy action`
- innovation 分支在 visual latent / visual real / tactile real / point real 四路上都走显式 residual

### 8.1 训练调试命令与本机结论

新 core 的最小训练 smoke 命令是：

```bash
python scripts/picf_core_train_smoke.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split training \
  --segment-index 0 \
  --stride 4 \
  --max-points 1024 \
  --device cuda
```

若走仓库 `.venv` / `uv`，则建议：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/picf_core_train_smoke.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split training \
  --segment-index 0 \
  --stride 4 \
  --max-points 1024 \
  --device cuda
```

新 core 的长期训练命令当前应改用：

```bash
python scripts/picf_core_train.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split training \
  --checkpoint-base-dir /tmp/openpi-train \
  --exp-name picf_core_train_run \
  --num-train-steps 30000 \
  --log-interval 100 \
  --save-interval 5000 \
  --accum-steps 1 \
  --unroll-steps 2 \
  --stride 4 \
  --max-points 1024 \
  --device cuda \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --wandb-enabled \
  --wandb-mode offline \
  --use-foundation-backbones \
  --use-tactile \
  --max-empty-window-retries 32
```

说明：

- 控制台会显示 `tqdm` 实时进度条；`--log-interval 100` 只控制 JSON 指标与 wandb scalar 的刷新频率，不影响进度条本身
- 进度条里显示的 `loss=` 是**当前最后一个 optimization step 的即时值**
  - 它不是 moving average，也不是 `log_interval` 区间均值
  - 当前 trainer 是直接把最近一步的 `outputs["loss_total"]` 写到 progress postfix
  - 在 `effective_global_batch = 4`、`unroll_steps = 2`、CALVIN window 高度异质的设定下，`1.4 -> 3.4 -> 1.3 -> 3.2` 这种来回跳动仍然是正常现象
  - 判断是否在收敛，应优先看 `metrics.jsonl` 中按 `log_interval` 聚合后的均值；当前已验证的双卡 `200` step run 里，`20-step` 平均 `loss_total` 是从约 `2.69` 下降到约 `2.40`
- 如果不想上报 wandb，可改成 `--no-wandb`
- 如果只想保留文件日志、不看进度条，可改成 `--no-progress`
- 如果是在云上跑长期训练，当前推荐把 wandb 设成 `offline`，这和旧 `pi0.5` 训练 README 的工程口径保持一致

如果想在不影响训练、也不占 GPU 的前提下实时看 `metrics.jsonl` 的收敛趋势，当前可以直接用：

```bash
python scripts/picf_watch_metrics.py /mnt/checkpoints/picf_core/picf_core/<exp-name>/metrics.jsonl --follow --clear-screen
```

这个脚本只做三件事：

- 追加读取 `metrics.jsonl`
- 打印最近值与最近 `N` 条的滑动均值
- 为 `loss_total / loss_action / loss_pt / tactile_active_rate / tactile_contact_prob_mean` 画 ASCII sparkline

它不 import 训练主模块，不会触发 CUDA 初始化，所以适合直接放在 JupyterLab 的 terminal 里长期挂着。

如果只想每次刷新一次，而不是持续 follow：

```bash
python scripts/picf_watch_metrics.py /mnt/checkpoints/picf_core/picf_core/<exp-name>/metrics.jsonl

如果不需要实时刷新，而是想看更稳定的 trend，当前更推荐：

```bash
python scripts/picf_watch_metrics.py /mnt/checkpoints/picf_core/picf_core/<exp-name>/metrics.jsonl --window 50
```

如果想直接生成一张可在 JupyterLab 文件浏览器里点开的 PNG，当前可以用：

```bash
python scripts/picf_plot_metrics.py \
  /mnt/checkpoints/picf_core/picf_core/<exp-name>/metrics.jsonl \
  --smoothing-window 50 \
  --output /mnt/checkpoints/picf_core/picf_core/<exp-name>/metrics_trend.png
```

这个脚本：

- 只读 `metrics.jsonl`
- 使用 `matplotlib` 的 `Agg` 后端
- 不会加载模型，也不会占 GPU
- 默认把原始曲线用淡色叠加，再画 `50` 步 rolling mean
- 一张图里同时给出：
  - `loss_total / loss_action / loss_alignment / loss_pt`
  - 各个细分 loss
  - `tactile_active_rate / tactile_contact_prob_mean / projective_candidate_density / steps_per_sec`
```
- 上面这条命令没有再显式传 `--warmup-steps`，因为长期 trainer 默认已经按 `2% * num_train_steps` 自动换算
- 当前训练入口的默认几何配方已经切到 `--stride 4 --max-points 1024 --crop-radius-m 0.10`
- 单卡 full foundation 当前推荐 `--accum-steps 1`
  - `--accum-steps 2` 在这台 `39.5 GiB` 云机上已真实复现 `torch.OutOfMemoryError`
  - 坏点在 V-JEPA backward 重算阶段，而不是 checkpoint 缺失或轻量路径误开
- 如果后续还要继续提高点密度，应单独做显存与收敛回归
- `--max-empty-window-retries 32` 是当前推荐保留的首步窗口安全阈值
  - 它只处理“窗口首帧局部 `xyzrgb` support 为空”的情况
  - 数学上等价于对 PICF 合法首步支持集做 rejection sampling，而不是吞掉任意异常继续训练
  - 当前 trainer preflight 已和 core runtime 使用同一条 pointcloud payload
  - 也就是首步合法性检查现在同样会使用 `depth_static + depth_gripper + robot_obs + focus_centers_world=[TCP,left_tip,right_tip]`
  - 这样 rejection sampling 不会再因为旧的 static-only / 单中心 crop 而系统性低估指尖局部几何

正式开训前先确认四件事：

- `nvidia-smi` 能看到目标 GPU，且显存空闲足够
- `checkpoints/foundation/vjepa2_1/...`、`checkpoints/foundation/anytouch2/checkpoint-4frames.pth`、`src/pretrain/SpatialLM_Sonata_encoder.pth` 都存在
- CALVIN 路径使用已经存在的 `task_ABCD_D` 目录或只读 zip，不要再次解压
- 环境变量里显式设置：
  - `HF_TOKEN='YOUR_HF_TOKEN'`
  - `HUGGINGFACE_HUB_TOKEN=$HF_TOKEN`
  - `HF_ENDPOINT=https://hf-mirror.com`

同一条长期训练入口在 `uv` 下也已经真实跑通：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/picf_core_train.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split training \
  --checkpoint-base-dir /tmp/openpi-train-smoke \
  --exp-name picf_core_train_uv_min \
  --num-train-steps 1 \
  --log-interval 1 \
  --save-interval 1 \
  --accum-steps 1 \
  --unroll-steps 2 \
  --stride 4 \
  --max-points 1024 \
  --device cuda \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --warmup-steps 1 \
  --wandb-mode disabled \
  --use-foundation-backbones \
  --use-tactile \
  --max-empty-window-retries 32
```

如果要从最近 checkpoint 继续：

```bash
python scripts/picf_core_train.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split training \
  --checkpoint-base-dir /tmp/openpi-train \
  --exp-name picf_core_train_run \
  --resume \
  --num-train-steps 30000 \
  --log-interval 100 \
  --save-interval 5000 \
  --accum-steps 1 \
  --unroll-steps 2 \
  --stride 4 \
  --max-points 1024 \
  --device cuda \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --wandb-enabled \
  --wandb-mode offline \
  --use-foundation-backbones \
  --use-tactile \
  --max-empty-window-retries 32
```

如果要直接起双卡 DDP 正式训练，当前推荐命令是：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_core_train.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split training \
  --checkpoint-base-dir /tmp/openpi-train \
  --exp-name picf_core_train_ddp_run \
  --num-train-steps 30000 \
  --log-interval 100 \
  --save-interval 5000 \
  --accum-steps 2 \
  --unroll-steps 2 \
  --stride 4 \
  --max-points 1024 \
  --device cuda \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --wandb-enabled \
  --wandb-mode offline \
  --use-foundation-backbones \
  --use-tactile \
  --max-empty-window-retries 32
```

如果需要严格对齐旧 `pi0.5` 的“全局 batch=2”口径：

- 双卡 DDP：保持 `--accum-steps 1`
- 单卡：理论上可改成 `--accum-steps 2`
  - 但当前这台 `39.5 GiB` 云机上，single-GPU + full foundation + `--accum-steps 2` 已真实复现 OOM
  - 因此当前工程推荐仍是单卡 `--accum-steps 1`，或直接切双卡 DDP

这条 DDP 命令当前已经做过最小起步回归和一次 `120` step 的 post-fix 回归，但还没有做长程收敛验证；因此它适合正式试训，不应被误写成“已完成多卡配方定型”。

如果想在当前已验证范围内进一步增大 batch：

- 推荐起跑档：双卡 DDP + `--accum-steps 2`
  - `effective_global_batch = 4`
- 已验证的更大档：双卡 DDP + `--accum-steps 4`
  - `effective_global_batch = 8`
- 在没有补完更高档回归前，不要把 README 写成 `accum_steps=8` 也已验证

关于这次 DDP 真实崩溃的根因，当前已经完成过一次逐样本重放：

- 旧 run 里真正抛错的不是 NCCL，而是 rank1 在 `sample_idx = 100` 时拿到了一个非法首步窗口
- 具体窗口是：
  - `segment_id = 11137`
  - `start_step = 196437`
  - `lang = "move the sliding door to the left"`
  - `nearest_dist = 0.1286 m`
  - 当时 `crop_radius_m = 0.08 m`；当前最终 tactile 默认值已提升到 `0.10 m`
- 也就是说：
  - 整帧 point cloud 仍然有 `512` 个点
  - 但以当前 EE 为中心的局部 `0.08m` ROI 为空
  - 同段 `196435 ~ 196438` 连续几帧都为空，直到 `196439` 才重新出现局部 support
- 这属于 free-space pre-contact 状态，而不是 pointcloud builder 坏掉
- 因为 `PICF` 的首个 control step 需要用当前 point evidence 启动 posterior，所以这类窗口起点不属于模型的合法首步支持集
- 训练器现在对它做的是 rejection sampling；post-fix 的双卡 DDP `120` step run 已完整跑通，并记录到：
  - `resampled_empty_first_step_windows = 1`
  - `loss_total = 2.6533`

当前本机已经真实跑通过；这轮 v0.4.8 严格复核后的代表性结果是：

- `python ... --device cpu --backend dir --segment_index=0`：
  - `loss_total = 1.8263`
  - `loss_alignment = 0.2108`
  - `loss_anchor_pv = 1.2373`
  - `loss_pv_weak = 4.3548`
  - `loss_focus_pv = 1.1552`
  - `loss_pt = 0.0`
  - `projective_candidate_density = 0.2995`
  - `mean_point_route_gate = 0.5869`
  - `mean_visual_route_gate = 0.5274`
  - `action_grad_norm = 2.9620`
  - `point_grad_norm = 0.0640`
- `python ... --device cpu --backend zip` 在 `segment_index = 0 / 50 / 100` 上也都通过：
  - `loss_total` 约为 `1.8263 / 1.8296 / 1.7323`
  - `loss_focus_pv` 约为 `1.1552 / 1.0604 / 0.8993`
  - `loss_pt` 当前都是 `0.0`
  - `projective_candidate_density` 分别约为 `0.2995 / 0.3073 / 0.3047`
  - `mean_point_route_gate` 分别约为 `0.5869 / 0.6159 / 0.5845`
  - `mean_visual_route_gate` 分别约为 `0.5274 / 0.5799 / 0.5881`
  - 说明当前 candidate edge set 没有塌到接近 `0`，也没有退化成近似全连接
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python ... --backend dir --segment_index=0 --device cpu`：
  - `cuda_available = false`
  - `cuda_runtime_available = false`
  - `loss_total = 1.8263`
  - `loss_alignment = 0.2108`
  - `loss_focus_pv = 1.1552`
  - `loss_pt = 0.0`
- `python scripts/picf_core_train.py ... --backend dir --device cuda --num-train-steps 2 --unroll-steps 2`：
  - `step=1 loss_total = 1.0894`
  - `step=2 loss_total = 2.5853`
  - checkpoint 已落到 `/tmp/openpi-train-smoke/picf_core/picf_core_train_min/step_2.pt`
- 同一条 `scripts/picf_core_train.py` 已真实验证：
  - `--resume` 能从 `latest.pt` 接着跑到 `step=3`
  - `zip + cuda + 1-step` 能正常起步并保存 checkpoint
  - `uv + dir + cuda + 1-step` 也能正常起步并保存：
    - `/tmp/openpi-train-smoke/picf_core/picf_core_train_uv_min/args.json`
    - `/tmp/openpi-train-smoke/picf_core/picf_core_train_uv_min/metrics.jsonl`
    - `/tmp/openpi-train-smoke/picf_core/picf_core_train_uv_min/latest.pt`
    - `/tmp/openpi-train-smoke/picf_core/picf_core_train_uv_min/step_1.pt`
    - `loss_total = 1.0894`
    - `loss_action = 0.5445`
    - `loss_alignment = 0.1994`
    - `loss_anchor_pv = 1.1742`
    - `loss_focus_pv = 1.5972`
    - `loss_point_real = 0.7243`
    - `loss_visual_latent = 0.3717`
    - `loss_visual_real = 0.5389`
    - `loss_pt = 0.0`
    - `projective_candidate_density = 0.3047`
    - `grad_norm ≈ 1.0`
    - `preclip_grad_norm` 当前也会单独记录

这里的 `grad_norm` 需要明确解释：

- 当前训练日志里的 `grad_norm` 是 **clip 之后** 的值
- 默认 `--grad-clip-norm = 1.0`
- 所以长期看到 `grad_norm ≈ 1.0`，首先说明的是 clipping 经常触发，而不是训练一定异常
- 现在脚本已经额外记录 `preclip_grad_norm`
  - 判断是否应该放宽 clipping，应该看 `preclip_grad_norm`，而不是只看被截断后的 `grad_norm`

和前一轮相比，`loss_alignment / loss_anchor_pv / loss_focus_pv` 的数值口径会有明显变化，
这是因为 v0.4.8 现在已经真实接入了：

- routing support gates
- relative projective attention bias
- attention-derived `L_{focus}^{pv}`
- point-tactile `L_{pt}`

这些数字本身不是收敛标准，
但它们足以说明：

- 当前一条 `t -> t+1` 训练图已经真的前向闭合
- action head 和 point real head 都拿到了非零梯度
- `optimizer.step()` 没有因为 in-place / graph 断裂而失败
- geometry-first point↔visual 分支也已经真的参与了训练图，而不是挂空

当前这条长期训练入口还要额外记住两个非阻塞边界：

- 当前真实验证已经覆盖：
  - 单进程单卡 direct python
  - 单进程单卡 `uv run --no-sync python`
  - 云机 `torchrun --standalone --nnodes=1 --nproc_per_node=2`
- 但这仍然是“最小起步回归”，不是多卡长程收敛验证
- 当前长期训练脚本虽然已经支持完整 foundation 模式，但 `stub` 路径仍然保留：
  - 用于本地无权重 / 无 CUDA 环境的结构 smoke
  - 不应把 `stub` 结果与 foundation 结果混为一谈

如果这条 smoke 失败，优先按下面顺序排：

1. `task_ABCD_D` 路径是否正确
2. Sonata / V-JEPA / AnyTouch checkpoint 是否存在，或 `--use-foundation-backbones` 是否误开
3. `xyzrgb` point cloud builder 是否产出非空 ROI
4. `visual_map_override` 或真实 visual encoder 是否给出下一帧 visual map
5. `rgb_tactile/depth_tactile` 是否可读；若只是做结构 smoke，可显式退回 `tactile_mode=stub`
6. CUDA 是否可见

### 8.2 官方训练入口的调试口径

官方训练入口仍是
[`scripts/train_pytorch.py`](/home/siyuanyue/Documents/openpi/scripts/train_pytorch.py)，
它当前还没有切到 `PicfFullCore` 主线，
所以这里的调试口径要分清：

- 新 core smoke：
  用 [`scripts/picf_core_train_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train_smoke.py)
- 官方 CALVIN 训练链 smoke：
  用 [`scripts/train_pytorch.py`](/home/siyuanyue/Documents/openpi/scripts/train_pytorch.py)

当前本机真实结论是：

- conda 默认环境会卡在 `transformers==4.48.1`
- 官方链要求 `4.53.2`
- 因此官方链调试默认应使用 `uv run --no-sync ...`

在 `uv` 环境下，官方链已经真实走到：

- CALVIN loader
- norm stats
- Sonata checkpoint loading
- forward
- backward
- `optim.step()`
- checkpoint save

当前这台机器上的真实分界是：

- `uv + cuda + bfloat16 + dummy paligemma/action expert`：能完整跑完一步训练并保存 checkpoint
- `uv + cuda + bfloat16 + full`：会在 forward 内部 materialize Sonata encoder 时 OOM
- `uv + cuda + float32 + dummy`：会在 `optim.step()` 分配 Adam state 时 OOM
- `uv + cuda + float32 + full`：会在 `model.to(cuda)` 时 OOM
- `uv + cpu`：这条 Sonata 训练链默认 fail-fast；即便 `--model.no-require-cuda`，spconv 仍会在 CUDA stream 路径报错

所以当前继续跑不完 clean full step 的主因是：

- full model 对当前 `8GB` 显存不够友好
- Sonata/spconv 路径本身是 CUDA-only
- 不是 CALVIN datapath 或训练循环本身断裂

所以调试上应把两类失败分开看：

- 如果失败在 loader / tokenizer / ckpt / forward 前：
  这是 wiring / environment 问题
- 如果已经到 `forward/backward/optim.step` 才炸：
  本机上通常先怀疑显存，而不是先怀疑 CALVIN datapath

### 8.3 2026-04-08 本机 CALVIN 核查补充

当前机器上，已经用真实 CALVIN 数据根：

- `/home/siyuanyue/datasets/calvin/dataset/task_ABCD_D`
- `/home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip`

做过下面这些实际验证：

- `CalvinLangSegmentDataset` 在 `dir` backend 上可读，segments=`22966`
- 同一个 dataset 在 `zip` backend 上也可读，且未解压大 zip
- 对同一训练样本索引 `0 / 1 / 10 / 100 / 1000`，`dir` 与 `zip` backend 读出的：
  - `prompt`
  - `rgb_static`
  - `depth_static`
  - `robot_obs`
  - `rgb_gripper`
  - `actions`
  都逐字段完全一致，最大绝对差均为 `0`
- `create_data_loader(pi05_calvin_sonata)` 在 `dir` backend 上可实际产出：
  - tokenized prompt
  - `state (B,32)`
  - `actions (B,16,32)`
  - `pointcloud (B,M,9)`
- `create_data_loader(pi05_calvin_sonata)` 在 `zip` backend 上也可实际产出 pointcloud batch
- `scripts/calvin/calvin_discover.py` 现在能正确识别：
  - `training/lang_annotations/auto_lang_ann.npy`
  - `validation/lang_annotations/auto_lang_ann.npy`
- `scripts/stageb_calvin_audit.py` 现在支持：
  - `--calvin-root .../task_ABCD_D`
  - `--backend dir`
  - `--calvin-root .../task_ABCD_D.zip`
  - `--backend zip`
- `scripts/stageb_calvin_audit.py --mode dataset` 已在真实数据上跑通：
  - `training + dir`
  - `validation + dir`
  - `training + zip`
  - `validation + zip`
- `scripts/stageb_calvin_audit.py --mode loader` 已在真实数据上跑通：
  - `training + dir`
  - `validation + dir`
  - `training + zip`
  - `validation + zip`
  - `training + dir + num_workers=2`
  - `training + zip + num_workers=2`
- 对 `create_data_loader(pi05_calvin_sonata)` 的真实首个 batch，`dir` 与 `zip` 经过完整 transform 后仍逐字段一致：
  - `Observation.state`
  - `tokenized_prompt`
  - `tokenized_prompt_mask`
  - `images['base_0_rgb']`
  - `images['left_wrist_0_rgb']`
  - `image_masks`
  - `point_clouds['pointcloud']`
  - `point_cloud_masks['pointcloud']`
  - `action`
  的最大绝对差均为 `0`
- `scripts/scaffold/scaffold_replay_smoke.py` 当前严格复核结果：
  - `dir` backend 可通过：
    - `frames = 64`
    - `mean_num_active = 77.75`
    - `mean_num_birth = 5.328125`
    - `mean_match_ratio = 0.9403`
    - `hold_count = 0`
  - `zip` backend 当前失败：
    - `RuntimeError("No scaffold states were produced.")`
  - 因此 legacy scaffold 的 `zip` 路径不应再被写成“当前已通过”
- `uv run` 也已经在仓库 `.venv` 上验证过：
  - 需要 `UV_CACHE_DIR=/tmp/uv-cache`
  - 建议搭配 `--no-sync`
  - `uv run python scripts/stageb_calvin_audit.py --mode loader ...` 已成功通过
  - `uv run` 环境内可导入 CUDA 版 PyTorch 与 Sonata runtime
- 2026-04-09 full-access 复核已经再次确认：
  - `nvidia-smi` 可见 `NVIDIA GeForce RTX 3070 Ti Laptop GPU`
  - `.venv` Python 与 `uv run --no-sync python` 都显示 `torch.cuda.is_available() == True`
  - 当前机器实际是 `1 x RTX 3070 Ti Laptop GPU / CUDA driver 581.95`
- `scripts/posterior/posterior_replay_smoke.py` 已在真实 CALVIN + CUDA + Sonata ckpt 上跑通
- 新增的一步训练 smoke [`scripts/picf_core_train_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/picf_core_train_smoke.py) 的 2026-04-09 当前 CUDA 记录是：
  - `loss_total = 1.3037`
  - `loss_action = 0.7568`
  - `loss_visual_latent = 0.3908`
  - `loss_visual_real = 0.5406`
  - `loss_point_real = 0.7307`
  - `loss_alignment = 0.1954`
  - `loss_anchor_pv = 1.1686`
  - `loss_pv_weak = 3.9291`
  - `loss_focus_pv = 1.1608`
  - `loss_pt = 0.0`
  - `projective_candidate_density = 0.2995`
  - `mean_point_route_gate = 0.7180`
  - `mean_visual_route_gate = 0.4933`
  - `action_grad_norm = 3.3148`
  - `point_grad_norm = 0.0717`
  - 同一脚本在 `direct python + dir/cuda`、`uv + dir/cuda`、`direct python + zip/cuda` 三条路径下都已跑通
- 同一条训练 smoke 现在也已在真实 `zip` backend + CPU 上跑通：
  - `loss_total = 1.8263`
  - `loss_alignment = 0.2108`
  - `loss_anchor_pv = 1.2373`
  - `loss_pv_weak = 4.3548`
  - `loss_focus_pv = 1.1552`
  - `loss_pt = 0.0`
  - `projective_candidate_density = 0.2995`
  - `mean_point_route_gate = 0.5869`
  - `mean_visual_route_gate = 0.5274`
  - `action_grad_norm = 2.9620`
  - `point_grad_norm = 0.0640`
  - 这说明 zip 版不仅能读 batch，也能走完整个 `PICF core forward + one-step loss + backward + optimizer.step()`
  - 2026-04-09 又额外复核了 `segment_index = 50 / 100`：
    - `projective_candidate_density` 约为 `0.3073 / 0.3047`
    - `mean_point_route_gate` 约为 `0.6159 / 0.5845`
    - `mean_visual_route_gate` 约为 `0.5799 / 0.5881`
- 同一条训练 smoke 也已在真实 `dir` backend + CPU 上跑通：
  - `loss_total = 1.8263`
  - `loss_alignment = 0.2108`
  - `loss_anchor_pv = 1.2373`
  - `loss_pv_weak = 4.3548`
  - `loss_focus_pv = 1.1552`
  - `loss_pt = 0.0`
  - `projective_candidate_density = 0.2995`
  - `mean_point_route_gate = 0.5869`
  - `mean_visual_route_gate = 0.5274`
  - `action_grad_norm = 2.9620`
  - `point_grad_norm = 0.0640`
- 官方 CALVIN 主训练入口 [`scripts/train_pytorch.py`](/home/siyuanyue/Documents/openpi/scripts/train_pytorch.py) 也已经做过最小 smoke：
  - 当前默认 conda 环境卡在 `transformers==4.48.1`，而训练脚本要求 `4.53.2`
  - 仓库 `.venv` / `uv.lock` 是正确的 `4.53.2`
  - 在 `uv run --no-sync ...` 下，真实训练链已经走到：
    - CALVIN loader
    - norm stats
    - Sonata ckpt load
    - forward
    - backward
    - `optim.step()`
    - checkpoint save
  - 当前已真实跑通的是：
    - `uv + cuda + bfloat16 + dummy paligemma/action expert`
    - `step=1 loss=1.2983 grad_norm=1.22`
    - checkpoint 目录：`/tmp/openpi-train-smoke/pi05_calvin_sonata/smoke_calvin_train_dummy_1step_uv_bf16/1`
  - 当前没跑通的边界也已经明确：
    - `uv + cuda + bfloat16 + full`：在 forward 内部 materialize Sonata encoder 时 OOM，`after_model_creation` 已到 `7.47GB`
    - `uv + cuda + float32 + dummy`：在 `optim.step()` 分配 Adam state 时 OOM，`after_backward` 峰值约 `4.49GB`
    - `uv + cuda + float32 + full`：在 `model.to(cuda)` 阶段 OOM
    - `uv + cpu`：默认 `require_cuda=True` fail-fast；即便 `--model.no-require-cuda`，spconv 仍会在 `torch.cuda.current_stream()` 路径报 `No CUDA GPUs are available`
  - 所以官方训练 datapath 的真实结论是：
    - wiring / CALVIN loader / ckpt / train loop 本身是通的
    - 当前机器限制主要是 `8GB` 显存，以及 Sonata/spconv 的 CUDA-only 运行时契约
- `PicfFullCore` 也已经在真实 CALVIN 帧上做过一次前向闭合核查：
  - 使用真实 point cloud builder
  - 使用 stub tactile encoder 绕开本机缺失的 AnyTouch2 ckpt
  - current posterior / physical prediction cache / semantic-conditioned prediction cache / innovation token / action 都能正确产出

这次还做了脚本级数学核查，结果是：

- `posterior_language_invariant_mu = True`
- `posterior_language_invariant_sigma = True`
- `binding` 每列和约等于 `1.0`
- `Sigma` 的对角最小值为正
- `Sigma` 的非对角绝对值最大值为 `0.0`，说明当前实现确实是对角协方差近似
- 首步 `innovation_norm = 0.0`
- 下一步在观测变化后 `innovation_norm > 0`

需要保留的已知边界：

- `MultiheadAttention` 当前没有引入复杂 OT / set-prediction matching，只是最小 cross-attention + sinkhorn dustbin 版本
- point / tactile / visual real targets 目前都还是轻量目标，不是最终 fully structured target
- 通用 `scripts/train_pytorch.py` 还没有接到这条新 core 训练路径
- legacy `scripts/posterior/posterior_full_check.py` 在真 GPU 上虽然能跑完，但当前不是全绿：
  - `smoke_has_precision_gain = false`
  - `acceptance_pass = false`
  - 失败点集中在旧 `src/openpi/picf/posterior/` acceptance 假设，而不是新 `src/openpi/picf/core/` 主线
- `PicfFullCore` 默认会尝试拉起 `AnyTouch2TactileEncoder(allow_random_init=False)`；
  在没有 AnyTouch2 checkpoint 的本机上，需要显式注入 tactile stub / tactile override 才能做纯结构 smoke

云机上的额外工程边界：

- `/mnt` 当前是 `fuse.fx` 挂载。
- 在云机上直接从 `/mnt/calvin_data/task_ABC_D` 跑 `dir` backend，会在首个 logged step 之前把 worker 阻塞到 `D` state。
- 切到 `/mnt/calvin_data/task_ABC_D.zip` 的 `zip` backend，也仍然会在同一类 FUSE 路径上阻塞。
- 对前 `3000` 个训练 steps、`seed=42`、`world_size=2`、`unroll_steps=2` 做精确采样统计后，双 rank 只会访问大约 `17.7k` 个唯一 `episode_*.npz`，估算体积约 `5 GiB`。
- 这使得“先把首 `3000` steps 需要的 frame 预取到本地 partial mirror，再从本地路径训练”成为当前云机上最可行的稳定方案。
- partial mirror 不能只复制 `training/episode_*.npz` 和 `training/lang_annotations/auto_lang_ann.npy`；训练初始化还会从 `calvin_root/calib/cameras.json` 构造 pointcloud / visual camera contract，所以 `calib/` 也必须一并带过去。
- `scripts/picf_stage_calvin_partial_cache.py` 现已默认复制整个 `calib/` 目录，并跳过已存在文件，便于把 `3000`-step cache 向更高步数增量扩展。
- 另外，`picf_core_train.py` 会在 DDP 包装前用 `source.window(rank)` 做一次 lazy-module warmup；partial cache 也必须覆盖这两个 warmup windows。脚本现在已经把这条初始化读路径纳入 staged set。
- 云机上后续又定位到另一类非数据问题：DDP 主训练脚本在 startup 阶段打印过多 runtime-introspection 日志时，rank0 可能会在真正进入 step loop 之前长时间卡住，表面现象像“训练没起步、GPU 几乎空闲、metrics 也不落盘”。
- 当前修复是让 `scripts/picf_core_train.py` 在 DDP 下默认启用 compact startup logging，只保留必要 contract 日志；如果必须排查 startup，再显式设置 `OPENPI_VERBOSE_STARTUP_LOGS=1` 打开详细日志。
- 对应地，云机稳定性验证应优先使用本地 partial cache + `/tmp` 本地 checkpoint/log 路径；只有确认 step loop 能稳定推进后，再考虑把 checkpoint/log 落回 `/mnt`。
- `scripts/picf_replay_windows.py` 的 `split-backward` 诊断现已改成“逐组件重算 forward/backward，再执行一次真实 `loss_total.backward()`”；这样避免了单图 `retain_graph=True` 连续回传造成的诊断态额外显存峰值，更适合在 `450+` 这类高风险区间做组件级定位。
- 对应脚本见 [`scripts/picf_stage_calvin_partial_cache.py`](/home/siyuanyue/Documents/openpi/scripts/picf_stage_calvin_partial_cache.py)。


## 9. 下一步建议

如果继续往训练或部署推进，优先顺序应是：

1. 把当前 coarse occupancy / low-res tactile / low-res RGB real heads 接到正式训练 loss。
2. 把 `semantic_override` 替换成真实 PaliGemma side path。
3. 在 dataset / replay 层显式生成 `t -> t+1` 的 real targets，而不是只靠 runtime 当前帧摘要。
4. 如果 one-step prediction 稳定，再把 point real head 从 occupancy 扩到 TSDF 或 scene-flow。

不建议再回到旧的 `object shell / stage2 / staged scaffold` 主线继续加模块。
