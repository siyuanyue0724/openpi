# PICF Scaffold + Posterior README

这份文档描述当前仓库里已经实际落地的 PICF 支线实现。

文档目标不是重复完整方法总纲，而是回答下面几个工程问题：

- 目前到底实现到了哪一阶段
- 代码在哪些文件里
- 关键输入/输出接口是什么
- scaffold 和 posterior 现在各自做了什么
- 哪些总纲条款已经满足，哪些还没有开始
- 应该用什么脚本做回放、审计和 acceptance 检查


## 1. 当前实现边界

当前仓库中的 PICF 支线已经落地到两个阶段：

1. `scaffold`
2. `posterior` 的 point-only 版本

当前**已经实现**：

- 连续 CALVIN segment replay
- PICF 专用 depth -> pointcloud + normals
- local frame / local crop / runtime metadata
- deterministic scaffold
- geometry-gated matching
- birth / stale scaffold fallback
- point-only current prior
- point-only expert constructor
- block-diagonal information-form Gaussian fusion
- posterior replay / invariant / acceptance / spec audit 脚本

当前**还没有实现**：

- learnable support query / GRU grouping
- V-JEPA visual expert
- AnyTouch tactile expert
- predictive prior / JEPA prior
- posterior-after object shell
- semantic / context / downstream Stage 2
- 主训练链或 serving 链接入

因此，当前状态应理解为：

`geometry-first scaffold + point-only posterior-v0`

而不是全文总纲的完整 v1 core。


## 2. 设计原则

本支线刻意保持“独立新链路”，没有直接改写现有 `pi0` 主训练链。

原因有两个：

- scaffold 和 posterior 都依赖跨帧状态连续性，不能直接复用随机采样 action 训练入口
- 在 visual / tactile / object shell 还没落地之前，把 PICF 并回主链只会放大调试范围

当前实现遵循的硬边界：

- posterior 当前只允许读 `current prior + point expert`
- posterior 不读 object / semantic / context / downstream state
- stale scaffold 时 point expert 关闭，posterior 必须严格退化为 prior
- 匹配和 prior 继承仍然由 scaffold 输出的 `matched_mask / pred_idx` 驱动


## 3. 目录结构

### 3.1 PICF 核心代码

路径：[`src/openpi/picf/README.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README.md)

主要文件：

- [`src/openpi/picf/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/contracts.py)
- [`src/openpi/picf/camera_io.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/camera_io.py)
- [`src/openpi/picf/geometry.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/geometry.py)
- [`src/openpi/picf/frame_context.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/frame_context.py)
- [`src/openpi/picf/pointcloud_picf.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/pointcloud_picf.py)
- [`src/openpi/picf/replay/calvin_replay.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/replay/calvin_replay.py)
- [`src/openpi/picf/scaffold/pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/scaffold/pipeline.py)
- [`src/openpi/picf/scaffold/matching.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/scaffold/matching.py)
- [`src/openpi/picf/scaffold/birth.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/scaffold/birth.py)
- [`src/openpi/picf/scaffold/local_frame.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/scaffold/local_frame.py)
- [`src/openpi/picf/posterior/config.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/config.py)
- [`src/openpi/picf/posterior/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/contracts.py)
- [`src/openpi/picf/posterior/prior.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/prior.py)
- [`src/openpi/picf/posterior/point_expert.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/point_expert.py)
- [`src/openpi/picf/posterior/fusion.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/fusion.py)
- [`src/openpi/picf/posterior/pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/pipeline.py)

### 3.2 Scaffold 脚本

- [`scripts/scaffold/scaffold_replay_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/scaffold/scaffold_replay_smoke.py)
- [`scripts/scaffold/scaffold_invariant_audit.py`](/home/siyuanyue/Documents/openpi/scripts/scaffold/scaffold_invariant_audit.py)
- [`scripts/scaffold/scaffold_stability_eval.py`](/home/siyuanyue/Documents/openpi/scripts/scaffold/scaffold_stability_eval.py)
- [`scripts/scaffold/scaffold_acceptance_check.py`](/home/siyuanyue/Documents/openpi/scripts/scaffold/scaffold_acceptance_check.py)

### 3.3 Posterior 脚本

- [`scripts/posterior/posterior_replay_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_replay_smoke.py)
- [`scripts/posterior/posterior_invariant_audit.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_invariant_audit.py)
- [`scripts/posterior/posterior_acceptance_check.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_acceptance_check.py)
- [`scripts/posterior/posterior_spec_audit.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_spec_audit.py)
- [`scripts/posterior/posterior_full_check.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_full_check.py)


## 4. 基础数据契约

### 4.1 `PicfPointCloudFrame`

定义位置：
[`src/openpi/picf/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/contracts.py)

字段：

- `grid_coord: [N,3] int32`
- `xyz_world: [N,3] float32`
- `rgb: [N,3] float32`
- `normal_world: [N,3] float32`
- `valid_point_mask: [N] bool`
- `frame_valid: bool`

语义：

- 这是 PICF 支线使用的可变长点集
- 坐标默认已经位于 world/base 稳定参考系
- `rgb` 是归一化到 `[0,1]` 的颜色
- `normal_world` 是点法向
- `frame_valid=False` 表示本帧点云无效，scaffold 会转入 stale 逻辑

### 4.2 `RuntimeMeta`

定义位置：
[`src/openpi/picf/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/contracts.py)

字段：

- `t_v_last`
- `t_p_last`
- `t_t_last`
- `t_rgb_last`
- `b_rgb_avail`
- `rgb_proj_residual`
- `n_vis_upd`
- `v_rgb_p`
- `v_pc_scaf`
- `stale_scaffold_steps`

当前说明：

- 这里只实现了 scaffold / point-only posterior 当前需要的最小运行时元数据
- `v_rgb_p` 和 `v_pc_scaf` 已经在 scaffold 中真正生效
- tactile / visual stale gate 的完整运行时使用还没有开始

### 4.3 `PicfObservation`

定义位置：
[`src/openpi/picf/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/contracts.py)

字段：

- `rgb_static`
- `depth_static`
- `robot_obs`
- `prompt`
- `step_id`
- `segment_id`
- `timestamp_s`
- `reset_scaffold`
- `rgb_gripper`
- `point_set`
- `runtime_meta`
- `G_t`

语义：

- 这是 PICF 支线每一步 forward 的统一输入容器
- `reset_scaffold=True` 表示一个 segment 的首帧，必须重置 scaffold continuity
- `G_t` 是当前 local frame 到稳定全局 frame 的变换

### 4.4 `SupportScaffoldState`

定义位置：
[`src/openpi/picf/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/contracts.py)

字段：

- `pi_geom`
- `x`
- `n`
- `r`
- `omega`
- `active_mask`
- `pred_idx`
- `matched_mask`
- `birth_mask`
- `e_id`
- `s_qry`
- `G_t`
- `step_id`
- `segment_id`
- `runtime_meta`
- `debug`

说明：

- 这是 scaffold 输出，也是 posterior 当前阶段的直接输入
- `pred_idx / matched_mask` 负责驱动 current prior 继承
- `e_id` 当前只保留 scaffold identity bookkeeping 语义，不进入 posterior

### 4.5 `PosteriorState`

定义位置：
[`src/openpi/picf/posterior/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/contracts.py)

字段：

- `mu`
- `var_block`
- `mu_prop`
- `var_prop_block`
- `point`
- `step_id`
- `segment_id`
- `debug`

重要边界：

- 当前没有 object / semantic / context 字段
- 当前没有 visual / tactile measurement 字段
- 当前 posterior 就是 support-level point-only Gaussian belief state


## 5. 点云与局部帧

### 5.1 `CalvinDepthToPicfPointCloud`

定义位置：
[`src/openpi/picf/pointcloud_picf.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/pointcloud_picf.py)

作用：

- 从 CALVIN 相机标定和 `rgb_static / depth_static` 构造 PICF 点云
- 输出 world 坐标点、法向、颜色和体素网格坐标

关键实现：

- 从 `cameras.json` 读取内外参
- 用深度图反投影得到 `points_cam`
- 用有限差分计算 organized normals
- 对无法稳定计算的 normal，用 `-xyz_cam` 方向做 fallback
- 默认将点和法向都变换到 world/base 稳定系
- 点选取支持 `fps` 或 `linspace`

常用参数：

- `stride`
- `max_points`
- `voxel_size`
- `z_min`
- `z_max`
- `selection_mode`

### 5.2 `EndEffectorLocalFrame`

定义位置：
[`src/openpi/picf/scaffold/local_frame.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/scaffold/local_frame.py)

当前默认行为：

- 从 `robot_obs[0:3]` 读平移
- 从 `robot_obs[3:6]` 读 ZYX roll-pitch-yaw
- 生成 `G_t`

这意味着当前 local frame 仍然是一个 CALVIN-friendly 默认实现，不是最终机器人无关方案。

### 5.3 `PointFrameContext`

定义位置：
[`src/openpi/picf/frame_context.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/frame_context.py)

作用：

- 把 `point_set` 从 world/base 系变换到当前 local frame
- 截取 `crop_radius_m` 范围内的局部点集
- 给 scaffold 和 posterior 共用


## 6. 连续重放器

### `CalvinSequentialReplay`

定义位置：
[`src/openpi/picf/replay/calvin_replay.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/replay/calvin_replay.py)

设计目的：

- 为 scaffold continuity 提供顺序 replay
- 避免使用现有 action training 的单帧随机采样路径

关键行为：

- 使用 `CalvinLangSegmentDataset(... sample_within_segment=False, action_horizon=1)`
- 按 segment 内顺序逐帧返回 `PicfObservation`
- 每个 segment 首帧自动设置 `reset_scaffold=True`

这条链是整个 PICF 支线成立的前提。


## 7. Scaffold 当前实现

### 7.1 配置

定义位置：
[`src/openpi/picf/scaffold/pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/scaffold/pipeline.py)

当前使用的是：

`DeterministicScaffoldConfig`

关键参数：

- `k_support=96`
- `k_birth=12`
- `k_active=96`
- `crop_radius_m=0.08`
- `r_min_m=0.003`
- `r_max_m=0.02`
- `tau_sup=0.25`
- `tau_p_m=0.005`
- `tau_n=0.8`
- `delta_pc_scaf_s=0.15`
- `n_hold_scaf=2`
- `v_rgb_identity=False`

### 7.2 当前 scaffold 是什么

当前 scaffold 不是 learnable query / attention 版本，而是 deterministic geometry-first 版本：

1. 构造 local crop
2. 从上一帧 carry-over active supports 生成 provisional seeds
3. 从当前未覆盖点集上用 weighted FPS 生成 birth seeds
4. 对每个 seed 做 radius-limited local grouping
5. 读出 `x / n / r / omega / e_id`
6. 生成 `active_mask`
7. 做 geometry-gated matching
8. 标出 `birth_mask`
9. 在 stale 时进入 transport-only fallback

### 7.3 已修正的几个关键行为

这几条是为了避免 posterior 被 scaffold 的伪证据污染：

- 空邻域 seed 不再强行吸附最近点
- carried seeds 在进入新帧前先做 crop 过滤
- `omega` 全零时允许 active set 为空
- stale scaffold 步禁止 birth
- stale scaffold 步强制关闭 point-RGB identity refinement


## 8. Posterior 当前实现

### 8.1 当前阶段定义

当前 posterior 是：

`point-only support-level Gaussian posterior`

也就是说，它只实现总纲中的最小核心子集：

- current prior
- point expert
- information-form Gaussian fusion

当前还没有：

- visual expert
- tactile expert
- predictive prior
- object shell

### 8.2 `PosteriorConfig`

定义位置：
[`src/openpi/picf/posterior/config.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/config.py)

当前默认值：

- `dim_h=32`
- `dim_g=32`
- `dim_c=32`
- `sigma_reset=1.0`
- `q_motion_block=(0.01, 0.01, 0.05)`
- `sigma_min2=1e-4`
- `sigma_max2=10.0`
- `point_var_h=4.0`
- `point_var_g=0.05`
- `point_var_c=4.0`
- `n_min_anchors=8`
- `delta_ref_m=0.005`
- `gamma_min_pc=0.05`
- `point_radius_min_m=0.03`

解释：

- `h` 和 `c` block 当前被设置成大方差弱约束
- `g` block 是 point-only posterior-v0 的主要信息承载块
- `point_radius_min_m=0.03` 是当前 point-only 审计链使用的最小支持域半径

### 8.3 `build_current_prior`

定义位置：
[`src/openpi/picf/posterior/prior.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/prior.py)

当前逻辑：

- `matched_mask=True` 且 `pred_idx>=0` 的 slot：
  - `mu_prop <- previous.mu[pred_idx]`
  - `var_prop_block <- previous.var_block[pred_idx] + q_motion_block`
- 其他 slot：
  - `mu_prop <- 0`
  - `var_prop_block <- sigma_reset^2`

这与总纲里 current prior 的基本语义一致。

### 8.4 `build_point_expert`

定义位置：
[`src/openpi/picf/posterior/point_expert.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/point_expert.py)

当前 point expert 是 geometry-primary 的 deterministic v0：

- 只在 fresh scaffold 时工作
- 默认只对 active supports 计算
- 支持域半径取：
  - `max(r_j, seed_init_radius_m, point_radius_min_m)`
- 构造统计量：
  - `anchor_count`
  - `delta2x`
  - `gamma_n`
  - `delta_pc`
  - `gamma_pc`
- gate 条件：
  - `anchor_count >= n_min_anchors`
  - `gamma_pc_tilde >= gamma_min_pc`

当前 measurement mean 的 `g` block 由下面的几何摘要组成并零填充到 32 维：

- `x(3)`
- `n(3)`
- `r(1)`
- `delta2x(3)`
- `gamma_n(1)`
- `gamma_pc(1)`
- `anchor_norm(1)`

这意味着当前 posterior-v0 还不是最终的 SONATA full-resolution learned point expert，而是一个 geometry-primary 占位实现，用来先把数值链和接口链走通。

### 8.5 `fuse_point_only`

定义位置：
[`src/openpi/picf/posterior/fusion.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/fusion.py)

当前融合逻辑：

- 以 block 为单位做 scalar variance information fusion
- 只对 `point.gate=True` 的 slot 更新
- `point.gate=False` 时 posterior 完全保留 prior

这正是当前阶段最关键的 acceptance 语义：

`stale scaffold -> point gate off -> posterior == prior`

### 8.6 `PointOnlyPosteriorPipeline`

定义位置：
[`src/openpi/picf/posterior/pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/pipeline.py)

接口：

```python
PointOnlyPosteriorPipeline.step(
    observation: PicfObservation,
    scaffold_state: SupportScaffoldState,
    previous: PosteriorState | None = None,
) -> PosteriorState
```

内部顺序：

1. 构造 `PointFrameContext`
2. 构造 current prior
3. 构造 point expert
4. 执行 Gaussian fusion
5. 返回 `PosteriorState`

debug 输出当前包括：

- `point_gate_ratio`
- `stale_prior_match_error`
- `posterior_prior_equal_on_stale`
- `matched_prior_count`
- `reset_prior_count`
- `precision_gain_count`
- `nan_count`
- `max_abs_mu`
- `min_var_block`
- `max_var_block`


## 9. 当前与总纲的一致性说明

### 已满足的条款

- H1：当前 posterior 没有 object/context/semantic/downstream 混入
- 当前 prior 只由 transport/reset 继承
- point 仍然承担 geometry-primary 角色
- point gate 使用 anchor count 和 local resolution-aware confidence
- stale scaffold 时 point expert 关闭
- posterior 当前仍是 block-diagonal Gaussian

### 尚未实现的条款

- visual expert 全部未开始
- tactile expert 全部未开始
- predictive prior / JEPA 全部未开始
- object shell 全部未开始
- Stage 2 全部未开始

### 当前和总纲的差异

当前 scaffold 仍是 deterministic 版本，而不是总纲中的 learnable query/grouping 版本。

当前 posterior 的 point expert 仍是 geometry-summary v0，而不是最终的 SONATA full-resolution learned expert。

这两点都是当前阶段的刻意设计，不是遗漏。


## 10. 运行方式

本仓库当前 PICF 支线的运行方式统一使用 `uv`。

推荐命令形式：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync ...
```

原因：

- 走仓库的 `uv` 环境
- 避免在当前环境下额外联网同步依赖
- 避免写入默认 cache 位置时受限


## 11. 脚本入口

### 11.1 Scaffold

#### smoke

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/scaffold/scaffold_replay_smoke.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 128
```

#### invariant audit

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/scaffold/scaffold_invariant_audit.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 128
```

#### stability

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/scaffold/scaffold_stability_eval.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 128
```

#### acceptance

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/scaffold/scaffold_acceptance_check.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 128
```

### 11.2 Posterior

#### smoke

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_replay_smoke.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 256
```

#### invariant audit

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_invariant_audit.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 256
```

#### engineering acceptance

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_acceptance_check.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 256
```

#### static spec audit

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_spec_audit.py \
  --repo-root .
```

#### full check

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_full_check.py \
  --repo-root . \
  --calvin-root <CALVIN_ROOT_OR_ZIP> \
  --backend zip \
  --segments 2 \
  --max-points 256
```


## 12. 测试入口

### scaffold + posterior 全量测试

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync pytest -q \
  src/openpi/picf/pointcloud_picf_test.py \
  src/openpi/picf/replay/calvin_replay_test.py \
  src/openpi/picf/scaffold/matching_test.py \
  src/openpi/picf/scaffold/pipeline_test.py \
  src/openpi/picf/posterior/prior_test.py \
  src/openpi/picf/posterior/point_expert_test.py \
  src/openpi/picf/posterior/fusion_test.py \
  src/openpi/picf/posterior/pipeline_test.py \
  scripts/scaffold/scaffold_scripts_test.py \
  scripts/scaffold/scaffold_audit_test.py \
  scripts/posterior/posterior_scripts_test.py
```

### lint

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync ruff check src/openpi/picf scripts/scaffold scripts/posterior
```


## 13. 已验证结果

当前已经实际跑通的结果包括：

- `ruff` 通过
- scaffold + posterior 相关 `pytest` 14/14 通过
- scaffold acceptance 通过
- posterior spec audit 通过
- posterior full check 通过

posterior full check 代表性结果：

- `mean_point_gate_ratio = 0.20572916666666666`
- `mean_precision_gain_count = 19.75`
- `nan_count_total = 0`
- `stale_equal_count = 8`
- `stale_equal_violations = 0`
- `var_clip_violations = 0`
- `precision_gain_violations = 0`
- `gate_count_violations = 0`

说明：

- point-only posterior 数值链已经真正参与更新，而不是全部退化为 prior
- stale 退化语义是严格成立的
- 当前 block variance 也保持在 clip 范围内


## 14. 关于 `max_points`

当前 posterior 审计脚本默认使用：

- `max_points=256`

而 scaffold acceptance 默认仍然常用：

- `max_points=128`

原因不是“posterior 需要放宽阈值”，而是当前 synthetic mini CALVIN 数据在 `max_points=128` 下过稀，point expert 往往拿不到足够 `anchor_count`，无法代表 posterior-v0 是否工作。

因此：

- scaffold 脚本可以继续用 128 做稳定性测试
- posterior 脚本应使用 256 来测试 point gate 和 Gaussian fusion 是否真正参与


## 15. 已知限制

- 当前 `EndEffectorLocalFrame` 依赖 CALVIN 风格 `robot_obs[0:6]`
- 当前 scaffold 不是 learnable query 版本
- 当前 posterior 不是 SONATA learned point feature 版本
- 当前没有 visual/tactile，所以不代表完整多模态 posterior
- 当前 acceptance 仍然基于 synthetic mini CALVIN 与局部脚本审计，不等于真实实验台闭环通过


## 16. 下一步建议

如果按总纲继续推进，下一阶段建议是：

1. 在 posterior 内加入 visual expert
2. 保持 posterior 的 static spec audit 继续阻止 object/semantic/context 越界进入
3. 在 visual expert 落地后，把 posterior full check 扩成：
   - point-only
   - visual-only
   - point+visual
4. 再开始 tactile

不建议现在直接并回 `pi0_pytorch.py` 主链。

先把：

- visual constructor
- visual gate
- visual + point information fusion

这三件事在 PICF 独立支线上做稳，再谈并回主训练链。
