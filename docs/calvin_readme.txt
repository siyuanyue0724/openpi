CALVIN + openpi(pi0.5_sonata) 训练与评测说明（当前环境版）

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

五、当前推荐的训练命令（2卡 DDP）
说明：如果你之后继续训练或 resume，优先用这一套。不要用 uv run，直接用 /root/openpi/.venv/bin/torchrun，避免 uv 再次联网检查依赖。

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

