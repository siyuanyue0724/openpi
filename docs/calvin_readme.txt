CALVIN + openpi(pi0.5_sonata) 训练与评测说明（当前环境版）

【2026-04-09 当前 shell 严格复核】
- 当前对应的 PICF 设计总纲是 `/home/siyuanyue/Documents/openpi/plan_readme_ray_geometry.md` 的 `v0.4.8 / MOVEON`
- 当前 `.venv` Python 下重新执行：
  - `python -m compileall -q src/openpi/picf/core scripts/picf_core_train_smoke.py scripts/picf_core_train.py`：通过
  - `pytest -q src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py src/openpi/picf/pointcloud_picf_test.py src/openpi/training/data_loader_test.py`
    - `31 passed`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py src/openpi/picf/pointcloud_picf_test.py src/openpi/training/data_loader_test.py`
    - `31 passed`
- 真实 `task_ABCD_D` 数据链本轮重新确认：
  - `scripts/stageb_calvin_audit.py --mode dataset --backend zip --split validation`：通过
  - `scripts/stageb_calvin_audit.py --mode loader --backend zip --split validation --batch-size 4 --num-workers 0`：通过
  - `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/stageb_calvin_audit.py --mode loader --backend zip --split validation --batch-size 4 --num-workers 0`：通过
  - `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/stageb_calvin_audit.py --mode loader --backend dir --split training --batch-size 4 --num-workers 0`：通过
  - `CalvinLangSegmentDataset` 对训练样本索引 `0 / 1 / 10 / 100 / 1000` 的 `dir/zip` 原始字段再次逐项比对，`prompt`、`rgb_static`、`depth_static`、`robot_obs`、`rgb_gripper`、`actions` 最大绝对差均为 `0`
- `scripts/picf_core_train_smoke.py` 本轮重新跑通：
  - `dir + cpu + segment_index=0`
  - `zip + cpu + segment_index=0/50/100`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/picf_core_train_smoke.py --backend dir --segment-index 0 --device cpu`
  - `dir + cuda + segment_index=0`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/picf_core_train_smoke.py --backend dir --segment-index 0 --device cuda`
  - `zip + cuda + segment_index=0`
- `scripts/picf_core_train.py` 本轮新增验证：
  - `dir + cuda + 2-step long-run`：通过
  - `dir + cuda + resume from latest.pt`：通过
  - `zip + cuda + 1-step long-run`：通过
  - `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/picf_core_train.py --backend dir --device cuda --num-train-steps 1`：通过
  - `projective_candidate_density` 约为 `0.2995 / 0.3073 / 0.3047`
  - `loss_focus_pv` 约为 `1.1552 / 1.0604 / 0.8993`
  - `loss_pt` 当前均为 `0.0`
  - `uv + dir + cuda + 1-step` 的 checkpoint 目录已真实落到：
    - `/tmp/openpi-train-smoke/picf_core/picf_core_train_uv_min/args.json`
    - `/tmp/openpi-train-smoke/picf_core/picf_core_train_uv_min/metrics.jsonl`
    - `/tmp/openpi-train-smoke/picf_core/picf_core_train_uv_min/latest.pt`
    - `/tmp/openpi-train-smoke/picf_core/picf_core_train_uv_min/step_1.pt`
- follow-through 复核结果：
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
- 当前 full-access shell 直接探测 `.venv` 与 `uv` 的 CUDA 可见性：
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() == 1`
  - `device_name = NVIDIA GeForce RTX 3070 Ti Laptop GPU`
  - `nvidia-smi` 显示 `Driver 581.95 / CUDA 13.0`
- `scripts/picf_core_train_smoke.py` 里的 `cuda_runtime_available` 只作为 smoke 诊断辅助字段；
  若它与独立执行的 `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` 不一致，
  以独立探测为准，不把它当作 core 数学或数据链放行条件
- 这轮在同一条 `dir + cpu + segment_index=0` smoke 上再次确认：
  - 直接 `python scripts/picf_core_train_smoke.py ...` 仍会报 `cuda_runtime_available = true`
  - `uv run --no-sync python scripts/picf_core_train_smoke.py ...` 则报 `cuda_runtime_available = false`
  因此它只能当辅助诊断字段，不能当通过/失败标准
- 官方 `scripts/train_pytorch.py pi05_calvin_sonata` 这条旧训练主线在当前机器上的真实边界：
  - `uv + cuda + bfloat16 + dummy paligemma/action expert`：一步训练通过并已保存 checkpoint
  - `uv + cuda + bfloat16 + full`：会在 forward 里 materialize Sonata encoder 时 OOM
  - `uv + cuda + float32 + dummy`：会在 `optim.step()` 分配 Adam state 时 OOM
  - `uv + cuda + float32 + full`：会在 `model.to(cuda)` 时 OOM
  - `uv + cpu`：默认 `require_cuda=True` fail-fast；即便显式 `--model.no-require-cuda`，Sonata/spconv 仍会在 CUDA stream 路径报错，所以这条官方训练链当前本质上是 CUDA-only
- legacy `scripts/scaffold/scaffold_replay_smoke.py` 当前真实状态：
  - `dir` backend 可通过，输出：
    - `frames = 64`
    - `mean_num_active = 77.75`
    - `mean_num_birth = 5.328125`
    - `mean_match_ratio = 0.9402755313952034`
    - `hold_count = 0`
  - `zip` backend 当前失败：
    - `RuntimeError("No scaffold states were produced.")`
  因此，旧 scaffold 的 `zip` 路径不能再写成“当前已通过”
- `scripts/picf_core_train.py` 当前虽然已经接了 `DistributedDataParallel` 与 `WORLD_SIZE/RANK/LOCAL_RANK`，
  但真实回归目前仍然只覆盖：
  - 单进程单卡 direct python
  - 单进程单卡 `uv run --no-sync python`
  不能把它误写成“当前已经做完 torchrun 多卡云上回归”

【2026-04-09 v0.4.8 / MOVEON 补充】
- 当前 `PICF core` 已经把 v0.4.8 里这轮要求补齐的数学项接进代码：
  - patch-unit projective compatibility
  - sparse projective candidate radius mask
  - low-support routing support gates
  - relative projective attention bias
  - attention-derived `L_{focus}^{pv}`
  - point-tactile `L_{pt}`
  - previous executed-action cache for carried prior / context
- 同一 core、同一帧、只改 `semantic_override` 的 follow-through 结果仍然是：
  - `posterior_mu_diff = 0.0`
  - `posterior_sigma_diff = 0.0`
  - `action_diff = 0.0296`
  - `binding_col_err = 0.0`
  - `ray_norm_err = 1.19e-07`
  - `innovation_norm_t0 = 0.0`
  - `innovation_norm_t1 = 0.9829`
  - `prev_policy_only_mu_diff = 0.0`
  - `prev_executed_mu_diff = 0.0035`
- 当前 `scripts/picf_core_train_smoke.py` 的 JSON 字段也已经补齐：
  - `loss_focus_pv`
  - `loss_pt`
  - `cuda_runtime_available`
- `scripts/picf_core_train_smoke.py` 现已固定随机 seed，下面记录的 smoke 数值可复现，不再因随机初始化漂动。

【2026-04-08 本机核查补充】
- 当前这台机器本地可用的 CALVIN 数据根是：
  /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D
- 同级还存在只读 zip：
  /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip
- 当前本机训练/审计优先使用“已解压目录 + dir backend”：
  --data.calvin_root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D
  --data.backend dir
- zip backend 也已经本地只读验证通过；但严禁再次解压 task_ABCD_D.zip。
- 当前环境里 Hugging Face 默认 cache 目录不可写；若后续需要拉新 tokenizer / 模型，请先设置：
  export TRANSFORMERS_CACHE=/tmp/huggingface
- 下文不少命令仍保留旧云端示例（/mnt/.../task_ABC_D）。在当前机器上使用时，应统一映射到 task_ABCD_D 与上述本地路径。

【2026-04-08 本机已实际打通的部分】
- `CalvinLangSegmentDataset`：
  - `dir` backend 已读通 `/home/siyuanyue/datasets/calvin/dataset/task_ABCD_D`
  - `zip` backend 已只读验证 `/home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip`
  - 对训练样本索引 `0 / 1 / 10 / 100 / 1000`，dir/zip 读出的 `prompt`、`rgb_static`、`depth_static`、`robot_obs`、`rgb_gripper`、`actions` 逐字段完全一致
- `create_data_loader(pi05_calvin_sonata)`：
  - `dir` backend 已能产出真实 batch（含 tokenized prompt / state / actions / pointcloud）
  - `zip` backend 也已能产出真实 batch
- `scripts/calvin/calvin_discover.py`：
  - 已核实能正确识别 `training/lang_annotations/auto_lang_ann.npy`
  - 不再把真实 CALVIN 误报成 “has_auto_lang_ann=false”
- `scripts/stageb_calvin_audit.py`：
  - `dataset` 模式已在真实 dir backend 跑通
  - `dataset` 模式已在真实 validation dir backend 跑通
  - `dataset` 模式已在真实 training zip backend 跑通
  - `dataset` 模式已在真实 validation zip backend 跑通
  - `loader` 模式已在真实 dir backend 跑通
  - `loader --split validation` 也已跑通
  - `loader` 模式已在真实 training zip backend 跑通
  - `loader` 模式已在真实 validation zip backend 跑通
  - `loader` 模式已在真实 training dir backend + `num_workers=2` 跑通
  - `loader` 模式已在真实 training zip backend + `num_workers=2` 跑通
- `create_data_loader(pi05_calvin_sonata)` 的真实首个 transform 后 batch：
  - `Observation.state`
  - `tokenized_prompt`
  - `tokenized_prompt_mask`
  - `images['base_0_rgb']`
  - `images['left_wrist_0_rgb']`
  - `image_masks`
  - `point_clouds['pointcloud']`
  - `point_cloud_masks['pointcloud']`
  - `action`
  在 dir/zip 两条路径上逐字段完全一致，最大绝对差均为 `0`
- `scripts/scaffold/scaffold_replay_smoke.py`：
  - 2026-04-09 当前 shell 严格复核时：
    - `dir` backend 可通过
    - `zip` backend 当前失败，报 `RuntimeError("No scaffold states were produced.")`
  - 因此它不应再被写成“当前 dir/zip 都已通过”
- `uv run`：
  - 需要显式设置 `UV_CACHE_DIR=/tmp/uv-cache`
  - 建议同时加 `--no-sync`
  - 在当前机器上，`uv run` 已成功跑通：
    - `scripts/stageb_calvin_audit.py --mode loader`
    - `pi05_calvin_sonata` 的真实 CALVIN dir backend loader
  - `uv` 自带的 `.venv` 里可导入：
    - CUDA 版 PyTorch（`torch 2.7.1+cu126`）
    - Sonata runtime
- 2026-04-09 full-access 复核已经再次确认：
  - `nvidia-smi` 可正常看到 `NVIDIA GeForce RTX 3070 Ti Laptop GPU`
  - `.venv` Python 与 `uv run --no-sync python` 都显示：
    - `torch.cuda.is_available() == True`
    - `torch.cuda.device_count() == 1`
  - `scripts/posterior/posterior_replay_smoke.py` 也曾在真实 CALVIN + CUDA + Sonata ckpt 上跑通
- 新增 `scripts/picf_core_train_smoke.py`：
  - 已在真实 `task_ABCD_D` 上完成 `forward + one-step future loss + backward + optimizer.step()`
  - 2026-04-09 当前 CUDA 结果：
    - `device = cuda`
    - `device_name = NVIDIA GeForce RTX 3070 Ti Laptop GPU`
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
  - 同一脚本已经在 `direct python + dir/cuda`、`uv + dir/cuda`、`direct python + zip/cuda` 三条路径下验证通过
  - 同一脚本已在真实 zip backend + CPU 路径验证通过：
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
  - 同一脚本已在真实 dir backend + CPU 路径验证通过：
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
  - 这说明 zip backend 不只是“能读一个 batch”，而是已经走通 `PICF core` 的一步训练 smoke
- `scripts/train_pytorch.py pi05_calvin_sonata` 官方训练链：
  - 在当前默认 conda 环境下，真实阻塞点是 `transformers==4.48.1`
  - 训练脚本显式要求 `transformers==4.53.2`
  - 仓库的 `uv.lock` 与 `.venv` 已满足 `4.53.2`
  - 在 `uv run --no-sync ...` 下，这条官方训练链已经实际走到：
    - CALVIN loader
    - norm stats
    - Sonata checkpoint 加载
    - forward
    - backward
    - `optim.step()`
    - checkpoint save
  - 当前已真实跑通的是：
    - `uv + cuda + bfloat16 + dummy paligemma/action expert`
    - `step=1 loss=1.2983 grad_norm=1.22`
    - checkpoint 目录：`/tmp/openpi-train-smoke/pi05_calvin_sonata/smoke_calvin_train_dummy_1step_uv_bf16/1`
  - 当前已确认的边界：
    - `uv + cuda + bfloat16 + full`：会在 forward 内部 materialize Sonata encoder 时 OOM，`after_model_creation` 已到 `7.47GB`
    - `uv + cuda + float32 + dummy`：会在 `optim.step()` 分配 Adam state 时 OOM，`after_backward` 峰值约 `4.49GB`
    - `uv + cuda + float32 + full`：会在 `model.to(cuda)` 时 OOM
    - `uv + cpu`：默认 `require_cuda=True` fail-fast；即便 `--model.no-require-cuda`，spconv 仍会在 `torch.cuda.current_stream()` 路径报 `No CUDA GPUs are available`
  - 这说明官方训练 datapath 本身是通的，当前本机的真实限制是 `8GB` 显存与 Sonata/spconv 的 CUDA-only 运行时契约

【2026-04-08 本机当前仍受环境限制的部分】
- `scripts/posterior/posterior_full_check.py`

这条现在已经能在本机真 GPU 上跑完，但结果不是全绿：
- `spec_pass = true`
- `point_backbone_checkpoint_loaded = true`
- `invariants_pass = true`
- `acceptance_pass = false`

当前失败原因是旧 posterior 路径本身在真实数据上给出：
- `mean_point_gate_ratio = 0.0`
- `mean_precision_gain_count = 0.0`

这说明：
- 环境和 CUDA 已经不是阻塞项
- 失败点在 legacy `src/openpi/picf/posterior/` acceptance 假设或其实现本身
- 这条旧 posterior/checker 路径不应再被当作 v0.4.8 新 core 的放行标准

另外需要明确：
- `legacy` 不是口头标签，而是当前代码引用范围的事实
- 目前 `src/openpi/picf/posterior/` 与 `src/openpi/picf/scaffold/` 的直接调用者，主要只剩：
  - `scripts/posterior/*`
  - `scripts/scaffold/*`
  - 对应的单测/回归脚本
- 当前主训练入口 `scripts/train_pytorch.py` 不导入 `openpi.picf.posterior` 或 `openpi.picf.scaffold`
- 新 `src/openpi/picf/core/` 目前也还没有并进 `scripts/train_pytorch.py`
- 但它现在已经有独立训练入口：
  - `src/openpi/picf/core/training.py`
  - `scripts/picf_core_train_smoke.py`
  - `scripts/picf_core_train.py`
- 其中 `scripts/picf_core_train.py` 已在真实 CALVIN 上完成：
  - `dir + cuda` 起步
  - `resume from latest.pt`
  - `zip + cuda` 起步

另外：
- `PicfFullCore` 默认会实例化 `AnyTouch2TactileEncoder(allow_random_init=False)`；
  在当前没有 AnyTouch2 ckpt 的机器上，如果只是做结构 smoke，需要注入 tactile stub。
- `num_workers>0` 的某些 Python DataLoader 审计在当前沙箱里会碰到 `resource_sharer` 的 socket 权限限制；
  这属于当前终端/沙箱限制，不代表云端训练会失败。

一、当前目录与环境
1. openpi 仓库：/root/openpi
2. openpi 训练/推理 Python：/root/openpi/.venv/bin/python
3. openpi 多卡训练入口：/root/openpi/.venv/bin/torchrun
4. CALVIN 仓库：/mnt/calvin
5. CALVIN 数据目录（已解压）：/mnt/calvin_data/task_ABC_D
6. openpi 训练 checkpoint：/mnt/checkpoints/pi05_calvin_sonata/abc_train_full_ddp2
7. 当前推荐评测 checkpoint：/mnt/checkpoints/pi05_calvin_sonata/abc_train_full_ddp2/23000
8. CALVIN 评测环境：micromamba 环境 calvin38

二、我们到底在训练/测试哪个集合
1. 官方 CALVIN 的 task_ABC_D 表示“ABC -> D” split，也就是：
   - 训练数据来自 A/B/C 环境
   - D 环境是 held-out 评测侧
2. 在你当前 openpi 训练里：
   - 训练命令使用 --data.split training
   - compute_norm_stats 也使用 --data.split training
   - 所以训练和 norm stats 只使用 task_ABC_D/training
3. 在你当前 CALVIN 官方 evaluator 里：
   - make_env(dataset_path) 会固定读取 dataset_path/validation
   - 也就是 /mnt/calvin_data/task_ABC_D/validation
4. 因此，你当前做的是：
   - 用 ABC training 数据训练
   - 用 D 侧 validation 数据做仿真评测
5. 所以：
   - 不是在 D 上训练
   - 当前仿真测试的是 D 侧 held-out 验证/测试集合

三、当前这套 fork 的关键实现语义
1. 基座权重：来自官方 pi05_base 转换后的 PyTorch checkpoint（/mnt/checkpoints/pi05_base_pytorch）
2. Sonata encoder：单独从 /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth 加载
3. pc_projector：没有单独 ckpt 时随机初始化后训练
4. 推理时：为了稳定，sample_actions 的 torch.compile 已做成“可通过环境变量关闭”的模式
5. 训练时：base checkpoint 采用“部分加载”，允许缺失 sonata.* 与 pc_projector.* 键

四、当前训练是否合理
1. 是。当前训练方式与 openpi 官方 PyTorch 入口一致：
   - 单机多卡通过 torchrun 启动
   - PyTorch 训练使用 bfloat16 或 float32，而不是 AMP mixed precision
2. 你现在跑通的正式训练参数是“针对 Sonata + 单/双卡 A100 显存约束做过裁剪的工程参数”，不是官方 README 的默认超参，但这是合理且必要的。
3. 当前最稳定的训练配置核心参数：
   - pytorch_training_precision=bfloat16
   - data.max-points=1024
   - data.stride=8
   - model.point-token-cap=128
   - batch-size=2（若 2 卡 DDP，则每卡 1）
   - num-workers=0
   - OPENPI_SONATA_MODE=projector

四点五、PICF v0.4.8 新 core 的云上长期训练命令
说明：这条命令对应的是 `src/openpi/picf/core/` 新主线，不是旧 `scripts/train_pytorch.py`。
当前真实验证通过的是单进程单卡版；它已经支持 checkpoint / resume。

1. 新开长期训练：
cd /root/openpi && \
export CUDA_VISIBLE_DEVICES=0 && \
export PYTHONPATH=/root/openpi/src && \
mkdir -p /mnt/checkpoints/picf_core/logs && \
/root/openpi/.venv/bin/python /root/openpi/scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABCD_D \
  --backend dir \
  --split training \
  --checkpoint-base-dir /mnt/checkpoints \
  --exp-name picf_core_train_run \
  --num-train-steps 30000 \
  --log-interval 100 \
  --save-interval 1000 \
  --accum-steps 1 \
  --unroll-steps 2 \
  --stride 8 \
  --max-points 512 \
  --device cuda \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --warmup-steps 500 \
  2>&1 | tee /mnt/checkpoints/picf_core/logs/picf_core_train_run.log

2. 继续训练（resume）：
cd /root/openpi && \
export CUDA_VISIBLE_DEVICES=0 && \
export PYTHONPATH=/root/openpi/src && \
/root/openpi/.venv/bin/python /root/openpi/scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABCD_D \
  --backend dir \
  --split training \
  --checkpoint-base-dir /mnt/checkpoints \
  --exp-name picf_core_train_run \
  --resume \
  --num-train-steps 30000 \
  --log-interval 100 \
  --save-interval 1000 \
  --accum-steps 1 \
  --unroll-steps 2 \
  --stride 8 \
  --max-points 512 \
  --device cuda \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --warmup-steps 500 \
  2>&1 | tee -a /mnt/checkpoints/picf_core/logs/picf_core_train_run.log

3. 如果云上只有只读 zip，不要再解压，改用：
  --calvin-root /mnt/calvin_data/task_ABCD_D.zip
  --backend zip

4. 这条新 core 长期训练入口当前的真实 contract：
  - 数据是脚本内部 `_CalvinTransitionSource` 直接从 `CalvinLangSegmentDataset(..., action_horizon=1, sample_within_segment=False)` 抽 `unroll_steps + 1` 帧 window
  - visual path 当前用 `_rgb_visual_override` 构造 pooled visual map，再经 `visual_map_override` 喂给 core；不是在线完整 V-JEPA encoder
  - tactile path 当前用 `_NullTactileEncoder()`；不是在线 AnyTouch2 真 ckpt
  - `metrics.jsonl` 采用 append 模式；如果复用同一个 `exp-name` 重新起一个非 `--resume` 训练，日志里会继续追加并可能出现重复 step id
  - 如果需要干净曲线，应该换新的 `exp-name`，或先清理旧实验目录

五、旧 `train_pytorch.py` 训练命令（归档参考）
说明：这一节是旧入口的归档记录，不是 `PICF v0.4.8 new core` 的当前推荐路径。只有在你明确需要维护旧 `pi05_calvin_sonata` 链路时，才看这一节。

1. 新开正式训练：
cd /root/openpi && \
export CUDA_VISIBLE_DEVICES=0,1 && \
export OPENPI_SONATA_MODE=projector && \
export OPENPI_SONATA_VALIDATE=1 && \
export WANDB_MODE=offline && \
export WANDB_DIR=/mnt/checkpoints/wandb && \
export WANDB_DISABLE_CODE=true && \
export WANDB_DISABLE_GIT=true && \
export WANDB_SILENT=true && \
export TORCHDYNAMO_DISABLE=1 && \
export OPENPI_DISABLE_TORCH_COMPILE=1 && \
export PYTHONPATH=/root/openpi/src && \
mkdir -p /mnt/checkpoints/pi05_calvin_sonata/logs /mnt/checkpoints/wandb && \
/root/openpi/.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 /root/openpi/scripts/train_pytorch.py pi05_calvin_sonata \
  --checkpoint-base-dir /mnt/checkpoints \
  --exp-name abc_train_full_ddp2 \
  --data.calvin-root /mnt/calvin_data/task_ABC_D.zip \
  --data.split training \
  --data.cameras-json-path /mnt/calvin_data/task_ABC_D/calib/cameras.json \
  --model.sonata-ckpt-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --pytorch-weight-path /mnt/checkpoints/pi05_base_pytorch \
  --pytorch-training-precision bfloat16 \
  --data.max-points 1024 \
  --data.stride 8 \
  --model.point-token-cap 128 \
  --batch-size 2 \
  --num-workers 0 \
  --num-train-steps 30000 \
  --log-interval 1000 \
  --save-interval 1000 \
  --wandb-enabled \
  2>&1 | tee /mnt/checkpoints/pi05_calvin_sonata/logs/abc_train_full_ddp2.log

2. 继续训练（resume）：
cd /root/openpi && \
export CUDA_VISIBLE_DEVICES=0,1 && \
export OPENPI_SONATA_MODE=projector && \
export OPENPI_SONATA_VALIDATE=1 && \
export WANDB_MODE=offline && \
export WANDB_DIR=/mnt/checkpoints/wandb && \
export WANDB_DISABLE_CODE=true && \
export WANDB_DISABLE_GIT=true && \
export WANDB_SILENT=true && \
export TORCHDYNAMO_DISABLE=1 && \
export OPENPI_DISABLE_TORCH_COMPILE=1 && \
export PYTHONPATH=/root/openpi/src && \
/root/openpi/.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 /root/openpi/scripts/train_pytorch.py pi05_calvin_sonata \
  --checkpoint-base-dir /mnt/checkpoints \
  --exp-name abc_train_full_ddp2 \
  --data.calvin-root /mnt/calvin_data/task_ABC_D.zip \
  --data.split training \
  --data.cameras-json-path /mnt/calvin_data/task_ABC_D/calib/cameras.json \
  --model.sonata-ckpt-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --pytorch-weight-path /mnt/checkpoints/pi05_base_pytorch \
  --pytorch-training-precision bfloat16 \
  --data.max-points 1024 \
  --data.stride 8 \
  --model.point-token-cap 128 \
  --batch-size 2 \
  --num-workers 0 \
  --num-train-steps 30000 \
  --log-interval 1000 \
  --save-interval 1000 \
  --wandb-enabled \
  --resume \
  2>&1 | tee -a /mnt/checkpoints/pi05_calvin_sonata/logs/abc_train_full_ddp2.log

六、当前推荐的评测 checkpoint
1. 当前最后一个完整可用 checkpoint：
   /mnt/checkpoints/pi05_calvin_sonata/abc_train_full_ddp2/23000
2. 不要用 24000：那一步保存 checkpoint 失败，不完整。
3. 后续可对比评测：10000 / 15000 / 23000。

七、CALVIN 仿真评测：总体思路
1. openpi 环境负责起 policy server
2. calvin38 环境负责跑 CALVIN 官方 evaluator
3. 两者通过本机 127.0.0.1:8000 通信
4. 不推荐在远程机器上优先用 GUI；建议先跑无 GUI，再选择保存视频

八、在当前环境里如何起 openpi policy server
说明：不要用 uv run，直接用 /root/openpi/.venv/bin/python，避免 uv 联网同步依赖。

命令：
cd /root/openpi && \
export CUDA_VISIBLE_DEVICES=0 && \
export PYTHONPATH=/root/openpi/src && \
export CALVIN_ROOT=/mnt/calvin_data/task_ABC_D && \
export OPENPI_SONATA_MODE=projector && \
export OPENPI_SONATA_VALIDATE=1 && \
export WANDB_MODE=disabled && \
export TORCHDYNAMO_DISABLE=1 && \
export OPENPI_DISABLE_TORCH_COMPILE=1 && \
/root/openpi/.venv/bin/python /root/openpi/scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_calvin_sonata \
  --policy.dir /mnt/checkpoints/pi05_calvin_sonata/abc_train_full_ddp2/23000

成功后会停在类似：
INFO:websockets.server:server listening on 0.0.0.0:8000
这不是卡住，而是在等待 evaluator 请求。

九、CALVIN evaluator 需要的本地改动（当前版本）
文件：/mnt/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy.py

1. 这个版本没有 --custom_lang_embeddings 参数，不要添加这个 flag。
2. 这个版本的 main() 在 custom_model 分支里有一个 bug：调用 evaluate_policy(model, env, ...) 时漏传了 epoch。
   你需要把：
   if args.custom_model:
       model = CustomModel()
       env = make_env(args.dataset_path)
       evaluate_policy(model, env, debug=args.debug)

   改成：
   if args.custom_model:
       model = CustomModel()
       env = make_env(args.dataset_path)
       epoch = os.environ.get("OPENPI_EVAL_TAG", "custom")
       evaluate_policy(
           model,
           env,
           epoch=epoch,
           eval_log_dir=args.eval_log_dir,
           debug=args.debug,
           create_plan_tsne=False,
       )

3. CustomModel 需要替换成 websocket 版本，并且把最后一维 gripper action 离散化成 ±1，且返回可写 numpy 数组。
   推荐逻辑：
   - 取 obs["rgb_obs"]["rgb_static"]
   - 取 obs["rgb_obs"]["rgb_gripper"]
   - 取 obs["depth_obs"]["depth_static"]
   - 取 obs["robot_obs"]
   - prompt 直接使用 goal（这个版本里 goal 已经是原始文本）
   - 通过 WebsocketClientPolicy(host, port) 调 openpi server
   - 取 actions[0]
   - 用 np.array(..., copy=True).reshape(-1)
   - 把 action[-1] 离散化：1.0 if >= 0 else -1.0

4. 为了先 smoke test，建议把：
   NUM_SEQUENCES = 1000
   临时改成：
   NUM_SEQUENCES = 10

十、无视频版本：傻瓜式评测教程
前提：
- openpi server 已经在另一个终端起好
- evaluate_policy.py 已经按上面说明改好

命令：
eval "$(/root/bin/micromamba shell hook -s bash)"
micromamba activate calvin38

unset DISPLAY
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export EGL_VISIBLE_DEVICES=0
export OPENPI_SERVER_HOST=127.0.0.1
export OPENPI_SERVER_PORT=8000
export OPENPI_EVAL_TAG=23000

mkdir -p /mnt/calvin_eval_logs/23000

cd /mnt/calvin/calvin_models/calvin_agent

python evaluation/evaluate_policy.py \
  --dataset_path /mnt/calvin_data/task_ABC_D \
  --custom_model \
  --eval_log_dir /mnt/calvin_eval_logs/23000 \
  --device 0

输出结果：
1. 终端会显示 sequence 成功率进度条
2. 结果会写到：
   /mnt/calvin_eval_logs/23000
3. 关键结果文件通常是：
   /mnt/calvin_eval_logs/23000/results.json

十一、有视频版本：傻瓜式评测教程
说明：只有在你已经把 evaluate_policy.py 的 rollout() 按当前本地版本加入“可选写 mp4”的逻辑后，这一节才会生效。
当前约定：
- CALVIN_SAVE_VIDEO=1 时保存视频
- CALVIN_VIDEO_DIR 指定视频目录
- 默认不设时不保存视频

命令：
eval "$(/root/bin/micromamba shell hook -s bash)"
micromamba activate calvin38

unset DISPLAY
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export EGL_VISIBLE_DEVICES=0
export OPENPI_SERVER_HOST=127.0.0.1
export OPENPI_SERVER_PORT=8000
export OPENPI_EVAL_TAG=23000
export CALVIN_SAVE_VIDEO=1
export CALVIN_VIDEO_DIR=/mnt/calvin_eval_logs/23000/videos

mkdir -p /mnt/calvin_eval_logs/23000/videos

cd /mnt/calvin/calvin_models/calvin_agent

python evaluation/evaluate_policy.py \
  --dataset_path /mnt/calvin_data/task_ABC_D \
  --custom_model \
  --eval_log_dir /mnt/calvin_eval_logs/23000 \
  --device 0

输出结果：
1. 终端会显示 sequence 成功率进度条
2. 评测 json 仍会写到：
   /mnt/calvin_eval_logs/23000
3. 视频会写到：
   /mnt/calvin_eval_logs/23000/videos
4. 当前本地 patch 版本是“成功/失败都保留视频”，并且只有显式设置 CALVIN_SAVE_VIDEO=1 才会生成视频。

十二、如何理解“当前到底在测哪个集合”
1. 训练：
   openpi 训练命令一直用 --data.split training，因此训练用的是 task_ABC_D/training，也就是 ABC 训练侧。
2. 仿真评测：
   这个版本的 evaluate_policy.py 在 make_env(dataset_path) 里固定：
   val_folder = Path(dataset_path) / "validation"
   env = get_env(val_folder, show_gui=False)
   因此仿真跑的是 task_ABC_D/validation，也就是 D 侧 held-out 评测集。
3. 结论：
   当前流程是标准的“ABC 训练 -> D 评测”。

十三、常见问题与对应处理
1. 进度条很慢，是不是没用 GPU？
   不一定。CALVIN 的 tqdm 单位是 sequence，不是单步 action。一个 sequence 最多 5 个子任务，每个子任务最多 360 env step，所以看起来会很慢。只要 EGL 检查显示 NVIDIA A100，且 openpi server 在 GPU 上占显存，就是正常的。

2. failed to EGL with glad
   重点检查：
   - unset DISPLAY
   - PYOPENGL_PLATFORM=egl
   - CUDA_VISIBLE_DEVICES / EGL_VISIBLE_DEVICES
   - libEGL.so.1 与 libEGL_nvidia.so.0

3. tacto.Sensor 报错
   当前环境里已经通过：
   - tacto 重新安装
   - numpy 降级到 1.23.5
   因此不要再随便升级 numpy。

4. gripper action 断言失败
   CALVIN 这个 env 要求 gripper action 必须严格是 ±1，而 openpi 往往输出连续值，所以必须在 CustomModel.step() 里把最后一维离散化。

5. output array is read-only
   返回给 env.step(action) 的 action 需要是可写数组，所以必须用 np.array(..., copy=True)。

6. 默认会不会乱存视频？
   不会。只有显式设置 CALVIN_SAVE_VIDEO=1 才会生成视频。

十四、当前推荐的操作顺序
1. 保持 openpi server 在 23000 checkpoint 上运行
2. 把 evaluate_policy.py 的 CustomModel / custom_model 分支 bug 修好
3. 先把 NUM_SEQUENCES 改成 10
4. 先跑“无视频版本” smoke
5. smoke 成功后，再加 CALVIN_SAVE_VIDEO=1 跑“有视频版本”
6. 再决定是否把 NUM_SEQUENCES 改回 1000

十五、当前推荐 checkpoint
优先使用：
- /mnt/checkpoints/pi05_calvin_sonata/abc_train_full_ddp2/23000

后续建议比较：
- 10000
- 15000
- 23000

十六、最短命令速查
A. 起 server：
/root/openpi/.venv/bin/python /root/openpi/scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_calvin_sonata --policy.dir /mnt/checkpoints/pi05_calvin_sonata/abc_train_full_ddp2/23000

B. 无视频评测：
python evaluation/evaluate_policy.py --dataset_path /mnt/calvin_data/task_ABC_D --custom_model --eval_log_dir /mnt/calvin_eval_logs/23000 --device 0

C. 有视频评测（前提：rollout 已加视频保存逻辑）：
export CALVIN_SAVE_VIDEO=1
export CALVIN_VIDEO_DIR=/mnt/calvin_eval_logs/23000/videos
python evaluation/evaluate_policy.py --dataset_path /mnt/calvin_data/task_ABC_D --custom_model --eval_log_dir /mnt/calvin_eval_logs/23000 --device 0
