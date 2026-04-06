# PICF Scaffold + Posterior README

本文档描述当前仓库里已经实际落地的 PICF 支线实现，并明确它与总纲的对应关系。

文档目标不是重复完整方法总纲，而是回答下面这些工程问题：

- 目前到底实现到了哪一阶段
- 代码在哪里
- 输入 / 输出契约是什么
- scaffold、point posterior、visual posterior 现在各自做了什么
- 哪些总纲条款已经满足，哪些还是刻意未实现
- 应该用什么脚本做 replay / audit / acceptance
- 当前阶段是否已经可以进入 AnyTouch 部署


## 1. 当前实现边界

当前 PICF 支线已经落地到三个层次：

1. `deterministic scaffold`
2. `point-only posterior-v0`
3. `visual stage-1 posterior`：`point + V-JEPA2.1 visual`

当前**已经实现**：

- 连续 CALVIN segment replay
- PICF 专用 depth -> pointcloud + normals
- local frame / local crop / runtime metadata
- deterministic geometry-first scaffold
- geometry-gated support matching
- birth / stale scaffold fallback
- current prior
- point-only expert constructor
- V-JEPA 2.1 frozen visual encoder wrapper
- support-level visual constructor 与 visual gate
- point + visual information-form Gaussian fusion
- point-only 与 visual-stage1 两套 replay / invariant / acceptance / spec-audit 脚本
- V-JEPA 2.1 ckpt 下载脚本与统一 checkpoint 布局

当前**还没有实现**：

- learnable support query / GRU grouping
- AnyTouch tactile expert
- predictive prior / JEPA prior
- posterior-after object shell
- semantic / context / downstream Stage 2
- 主训练链或 serving 链接入

因此，当前状态应理解为：

`geometry-first scaffold + canonical support posterior(stage-1: point + visual)`

而不是全文总纲的完整 v1 core。


## 2. 设计原则

PICF 支线仍保持“独立新链路”，没有直接改写 `pi0` 主训练链。

原因：

- scaffold / posterior 依赖跨帧状态连续性，不能直接复用随机 action 训练入口
- tactile / object shell / JEPA prior 还未落地前，合入主链只会扩大调试面

当前阶段遵循的硬边界：

- current posterior 只允许读 `current prior + point expert + visual expert`
- posterior 不读 object / semantic / context / downstream state
- visual 只作为 canonical expert 进入 posterior
- visual 不得进入 support identity path
- stale scaffold 时 point expert 关闭
- stale scaffold 时 visual 只允许 center-patch fallback
- 若 stale 且无 visual measurement，则 posterior 必须严格退化为 prior

这与总纲中的 H1 / H2 / H3 / H4 是一致的：

- H1：当前 belief state 仍然只由 prior + canonical experts 决定
- H2：support identity 仍由 scaffold 的 geometry-gated continuity 定义
- H3：object shell 仍未开启，因此不存在 object 反写 posterior
- H4：semantic / context / downstream state 仍未进入 core


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
- [`src/openpi/picf/posterior/visual_expert.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/visual_expert.py)
- [`src/openpi/picf/posterior/fusion.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/fusion.py)
- [`src/openpi/picf/posterior/fusion_visual.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/fusion_visual.py)
- [`src/openpi/picf/posterior/pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/pipeline.py)
- [`src/openpi/picf/posterior/pipeline_visual.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/pipeline_visual.py)

### 3.2 V-JEPA 2.1 接入代码

- [`src/openpi/picf/vjepa/config.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/vjepa/config.py)
- [`src/openpi/picf/vjepa/history.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/vjepa/history.py)
- [`src/openpi/picf/vjepa/preprocess.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/vjepa/preprocess.py)
- [`src/openpi/picf/vjepa/wrapper.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/vjepa/wrapper.py)
- [`src/openpi/picf/vjepa/vendor/ATTRIBUTION.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/vjepa/vendor/ATTRIBUTION.md)

说明：

- `vendor/` 下 vendoring 了 V-JEPA 2.1 encoder 的最小官方子集
- runtime 不依赖 hub 下载
- `TEMP_REPO` 只是临时来源，不属于最终运行依赖

### 3.3 Scaffold 脚本

- [`scripts/scaffold/scaffold_replay_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/scaffold/scaffold_replay_smoke.py)
- [`scripts/scaffold/scaffold_invariant_audit.py`](/home/siyuanyue/Documents/openpi/scripts/scaffold/scaffold_invariant_audit.py)
- [`scripts/scaffold/scaffold_stability_eval.py`](/home/siyuanyue/Documents/openpi/scripts/scaffold/scaffold_stability_eval.py)
- [`scripts/scaffold/scaffold_acceptance_check.py`](/home/siyuanyue/Documents/openpi/scripts/scaffold/scaffold_acceptance_check.py)

### 3.4 Posterior 脚本

point-only：

- [`scripts/posterior/posterior_replay_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_replay_smoke.py)
- [`scripts/posterior/posterior_invariant_audit.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_invariant_audit.py)
- [`scripts/posterior/posterior_acceptance_check.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_acceptance_check.py)
- [`scripts/posterior/posterior_spec_audit.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_spec_audit.py)
- [`scripts/posterior/posterior_full_check.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_full_check.py)

visual stage-1：

- [`scripts/posterior/posterior_visual_replay_smoke.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_visual_replay_smoke.py)
- [`scripts/posterior/posterior_visual_invariant_audit.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_visual_invariant_audit.py)
- [`scripts/posterior/posterior_visual_acceptance_check.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_visual_acceptance_check.py)
- [`scripts/posterior/posterior_visual_stage1_spec_audit.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_visual_stage1_spec_audit.py)
- [`scripts/posterior/posterior_visual_full_check.py`](/home/siyuanyue/Documents/openpi/scripts/posterior/posterior_visual_full_check.py)

### 3.5 Checkpoint 管理

- [`scripts/vjepa_ckpt_fetch.py`](/home/siyuanyue/Documents/openpi/scripts/vjepa_ckpt_fetch.py)

默认 V-JEPA 2.1 ckpt 放在：

- [`checkpoints/foundation/vjepa2_1`](/home/siyuanyue/Documents/openpi/checkpoints)


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

- 这里只实现了 scaffold / point posterior / visual posterior 当前需要的最小 runtime metadata
- `v_rgb_p` 和 `v_pc_scaf` 已经在 scaffold 中真正生效
- `t_v_last` 已从“每步直接写当前时间”修正为“最近一次被 runtime 接受的 visual update 时间”
- tactile stale gate 的完整 runtime 使用仍未开始

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
- visual stage-1 当前只消费 `rgb_static`

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
- visual expert 只读取 `pi_geom / x / n / r / G_t / runtime_meta`，不反写 scaffold

### 4.5 `PosteriorState`

定义位置：
[`src/openpi/picf/posterior/contracts.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/contracts.py)

字段：

- `mu`
- `var_block`
- `mu_prop`
- `var_prop_block`
- `point`
- `visual`
- `step_id`
- `segment_id`
- `debug`

重要边界：

- 当前没有 object / semantic / context 字段
- 当前没有 tactile measurement 字段
- 当前 posterior 是 support-level Gaussian belief state
- point-only 管线下 `visual=None`
- visual stage-1 管线下 `visual: VisualExpertState`


## 5. 点云、局部帧与视觉历史

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
- 给 scaffold、point posterior、visual posterior 共用

### 5.4 `VisualClipBuffer`

定义位置：
[`src/openpi/picf/vjepa/history.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/vjepa/history.py)

作用：

- 在 replay / runtime 上层维护 segment-scoped visual history
- 首帧 reset
- 长度不足 `num_frames` 时做 deterministic left-pad
- 不改写 `CalvinSequentialReplay`

当前默认：

- `num_frames = 64`


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

这条链是整个 PICF 支线成立的前提，也是 visual clip buffer 能独立挂载的基础。


## 7. Scaffold 当前实现

### 7.1 配置

定义位置：
[`src/openpi/picf/scaffold/pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/scaffold/pipeline.py)

当前使用：

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

### 7.3 已修正的关键行为

- 空邻域 seed 不再强行吸附最近点
- carried seeds 在进入新帧前先做 crop 过滤
- `omega` 全零时允许 active set 为空
- stale scaffold 步禁止 birth
- stale scaffold 步强制关闭 point-RGB identity refinement
- `t_v_last` 不再被 scaffold 误写成“当前 step 时间”


## 8. Posterior 当前实现

### 8.1 当前阶段定义

当前 posterior 有两条并行管线：

1. `PointOnlyPosteriorPipeline`
2. `PointVisualPosteriorPipeline`

第一条保留为 regression baseline。第二条是当前真正的 stage-1 core。

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
- `visual_var_h=0.5`
- `visual_var_g=1.0`
- `visual_var_c=4.0`
- `n_min_anchors=8`
- `delta_ref_m=0.005`
- `gamma_min_pc=0.05`
- `point_radius_min_m=0.03`

解释：

- point expert 仍承担 geometry-primary 角色，因此 `point_var_g` 最小
- visual expert 当前主要承担 appearance / view-conditioned support summary，因此 `visual_var_h < visual_var_g`
- `c` block 目前仍不给 visual 强约束

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

这与总纲里的 current prior 语义一致。

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

当前 measurement mean 的 `g` block 由几何摘要组成并零填充到 32 维。

### 8.5 `Vjepa2VisualEncoder`

定义位置：
[`src/openpi/picf/vjepa/wrapper.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/vjepa/wrapper.py)

职责：

- 接收 `[T,H,W,3]` clip
- resize / normalize
- 运行 frozen V-JEPA 2.1 encoder
- 产出 dense token grid

当前默认部署配置：

- `model_name = "vjepa2_1_vit_base_384"`
- `img_size = 384`
- `num_frames = 64`
- `patch_size = 16`
- `tubelet_size = 2`
- `camera_name = "static"`
- 官方 checkpoint 选择语义：
  - `vit_base_384 / vit_large_384 -> ema_encoder`
  - `vit_giant_384 / vit_gigantic_384 -> target_encoder`

张量形状：

- 输入 clip：`[64,H,W,3]`
- 预处理后：`[1,64,3,384,384]`
- encoder token grid：`[32,24,24,768]`
- current visual map：`[24,24,768]`

### 8.6 `build_visual_expert`

定义位置：
[`src/openpi/picf/posterior/visual_expert.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/visual_expert.py)

当前 visual constructor 严格分两种模式：

fresh scaffold：

- support-level point-pooled visual feature
- support-center patch feature
- visibility ratio
- depth residual
- depth availability bit

stale scaffold：

- 不重新 point-pool
- 只对 transport 后 center 做 patch pooling
- point field 固定为空
- 允许 center-patch-only fallback

视觉 gate 规则：

- fresh 且有 depth residual：
  - `center_in_view`
  - `visibility > epsilon_vis`
  - `depth_residual < tau_z_m`
- fresh 且无 depth residual：
  - `center_in_view`
  - `visibility > tau_vis`
- stale：
  - `center_in_view`

工程边界：

- visual 只读 `pi_geom / x / n / r / G_t / rgb_static / depth_static`
- visual 不回写 `e_id / s_qry / pred_idx / matched_mask`
- 因此不进入 support identity path

### 8.7 `fuse_point_only`

定义位置：
[`src/openpi/picf/posterior/fusion.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/fusion.py)

当前融合逻辑：

- 以 block 为单位做 scalar variance information fusion
- 只对 `point.gate=True` 的 slot 更新
- `point.gate=False` 时 posterior 完全保留 prior

### 8.8 `fuse_point_visual`

定义位置：
[`src/openpi/picf/posterior/fusion_visual.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/fusion_visual.py)

当前融合逻辑：

- prior precision 先入
- point / visual experts 分别按 gate 加入
- 仍保持 block-diagonal Gaussian
- 不引入 cross-covariance residual

这与总纲里的 calibrated engineering approximation 是一致的。

### 8.9 `PointOnlyPosteriorPipeline`

定义位置：
[`src/openpi/picf/posterior/pipeline.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/pipeline.py)

这是 regression baseline，不动。

### 8.10 `PointVisualPosteriorPipeline`

定义位置：
[`src/openpi/picf/posterior/pipeline_visual.py`](/home/siyuanyue/Documents/openpi/src/openpi/picf/posterior/pipeline_visual.py)

接口：

```python
PointVisualPosteriorPipeline.step(
    observation: PicfObservation,
    scaffold_state: SupportScaffoldState,
    previous: PosteriorState | None = None,
) -> PosteriorState
```

内部顺序：

1. fresh 时构造 `PointFrameContext`
2. 构造 current prior
3. point expert
4. `VisualClipBuffer.push()`
5. `Vjepa2VisualEncoder.encode_clip()`
6. `build_visual_expert()`
7. `fuse_point_visual()`
8. 返回 `PosteriorState`


## 9. 当前与总纲的一致性说明

### 9.1 已满足的条款

- H1：posterior 当前没有 object/context/semantic/downstream 混入
- current prior 只由 transport/reset 继承
- point 仍承担 geometry-primary 角色
- point gate 使用 anchor count 和 local resolution-aware confidence
- stale scaffold 时 point expert 关闭
- visual 只作为 canonical expert 进入 posterior
- visual 不进入 support identity path
- stale scaffold 时 visual 仅允许 center-patch fallback
- posterior 当前仍是 block-diagonal Gaussian

### 9.2 尚未实现的条款

- tactile expert 全部未开始
- predictive prior / JEPA 全部未开始
- object shell 全部未开始
- semantic / context / Stage 2 全部未开始
- learnable query / grouping 全部未开始

### 9.3 当前和总纲的差异

- scaffold 仍是 deterministic 版本，不是 learnable query / GRU grouping
- point expert 仍是 geometry-summary v0，而不是最终的 SONATA learned point expert
- visual expert 当前只落到 single-camera `rgb_static`
- 还没有 multi-camera tempered aggregation

这些都是当前阶段的刻意设计，不是遗漏。


## 10. Checkpoint 布局

V-JEPA 2.1 ckpt 当前统一放在：

- [`checkpoints/foundation/vjepa2_1`](/home/siyuanyue/Documents/openpi/checkpoints)

默认结构：

```text
checkpoints/foundation/vjepa2_1/
  manifest.json
  vjepa2_1_vit_base_384/
    vjepa2_1_vitb_dist_vitG_384.pt
  vjepa2_1_vit_large_384/
    vjepa2_1_vitl_dist_vitG_384.pt
  ...
```

下载脚本：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/vjepa_ckpt_fetch.py \
  --model vjepa2_1_vit_base_384
```


## 11. 运行方式

PICF 支线统一推荐：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync ...
```

原因：

- 走仓库的 `uv` 环境
- 避免额外联网同步依赖
- 避免写入默认 cache 位置时受限


## 12. 脚本入口

### 12.1 Scaffold

smoke：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/scaffold/scaffold_replay_smoke.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 128
```

invariant audit：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/scaffold/scaffold_invariant_audit.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 128
```

stability：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/scaffold/scaffold_stability_eval.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 128
```

acceptance：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/scaffold/scaffold_acceptance_check.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 128
```

### 12.2 Posterior point-only

smoke：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_replay_smoke.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 256
```

invariant audit：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_invariant_audit.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 256
```

acceptance：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_acceptance_check.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 256
```

static spec audit：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_spec_audit.py \
  --repo-root .
```

full check：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_full_check.py \
  --repo-root . \
  --calvin-root <CALVIN_ROOT_OR_ZIP> \
  --backend zip \
  --segments 2 \
  --max-points 256
```

### 12.3 Posterior visual stage-1

smoke：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_visual_replay_smoke.py \
  --repo-root . \
  --calvin-root <CALVIN_ROOT_OR_ZIP> \
  --backend zip \
  --segments 2 \
  --max-points 256 \
  --checkpoint-path /abs/path/to/vjepa2_1_vitb_dist_vitG_384.pt
```

invariant audit：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_visual_invariant_audit.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> \
  --backend zip \
  --segments 2 \
  --max-points 256 \
  --checkpoint-path /abs/path/to/vjepa2_1_vitb_dist_vitG_384.pt
```

acceptance：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_visual_acceptance_check.py \
  --calvin-root <CALVIN_ROOT_OR_ZIP> \
  --backend zip \
  --segments 2 \
  --max-points 256 \
  --checkpoint-path /abs/path/to/vjepa2_1_vitb_dist_vitG_384.pt
```

visual stage-1 spec audit：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_visual_stage1_spec_audit.py \
  --repo-root .
```

full check：

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python scripts/posterior/posterior_visual_full_check.py \
  --repo-root . \
  --calvin-root <CALVIN_ROOT_OR_ZIP> \
  --backend zip \
  --segments 2 \
  --max-points 256 \
  --checkpoint-path /abs/path/to/vjepa2_1_vitb_dist_vitG_384.pt
```


## 13. 测试入口

### 13.1 核心测试

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
  src/openpi/picf/posterior/fusion_visual_test.py \
  src/openpi/picf/posterior/visual_expert_test.py \
  src/openpi/picf/posterior/pipeline_visual_test.py \
  src/openpi/picf/vjepa/history_test.py \
  src/openpi/picf/vjepa/wrapper_test.py \
  scripts/scaffold/scaffold_scripts_test.py \
  scripts/scaffold/scaffold_audit_test.py \
  scripts/posterior/posterior_scripts_test.py \
  scripts/posterior/posterior_visual_scripts_test.py
```

### 13.2 lint

```bash
UV_CACHE_DIR=/tmp/uvcache uv run --no-sync ruff check src/openpi/picf scripts/scaffold scripts/posterior
```


## 14. 已验证结果

已经实际跑通的结果包括：

- `ruff` 通过
- `python -m compileall` 通过
- scaffold + posterior + visual-stage1 相关 `pytest` 21/21 通过
- point-only posterior full check 通过
- point-only posterior spec audit 通过
- visual stage-1 spec audit 通过
- real-ckpt `point_visual` acceptance 通过
- 真实 V-JEPA 2.1 base checkpoint 已下载、校验并与本地 wrapper 精确对齐

point-only baseline 代表性结果：

- `mean_point_gate_ratio = 0.20572916666666666`
- `mean_precision_gain_count = 19.75`
- `nan_count_total = 0`

真实 V-JEPA 2.1 base checkpoint 核验结果：

- 文件大小约 `1.6G`
- `sha256 = 848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`
- checkpoint 顶层含 `encoder / ema_encoder / predictor`
- `encoder` 与 `ema_encoder` 不相同；当前 wrapper 已按官方 2.1 `vitb-384` 约定优先加载 `ema_encoder`
- 与本地 vendored encoder 对齐结果：
  - `missing_keys = 0`
  - `unexpected_keys = 0`

真实 ckpt 单次 encoder 前向（CPU）结果：

- `tokens_shape = (32, 24, 24, 768)`
- `current_shape = (24, 24, 768)`
- `finite = True`

真实 ckpt 单步 PICF point+visual posterior 结果：

- `point_gate_ratio = 0.11458333333333333`
- `visual_gate_ratio = 0.3020833333333333`
- `point_precision_gain_count = 11`
- `visual_precision_gain_count = 29`
- `nan_count = 0`

这些结果说明：

- point-only posterior 数值链仍然稳定
- visual ckpt 不是只“下载成功”，而是已经真实进入 posterior 链
- visual stage-1 的张量形状、gate 与融合数值都闭合
- 当前代码已经过一轮提交前严格核查：
  - unit / script tests
  - static spec audits
  - real checkpoint follow-through
  - linter
  - import / bytecode compile checks

真实 ckpt 8-step mini CALVIN 回放结果：

- point + visual：
  - `mean_point_gate_ratio = 0.20572916666666663`
  - `mean_visual_gate_ratio = 0.44791666666666663`
  - `mean_point_precision_gain_count = 19.75`
  - `mean_visual_precision_gain_count = 43.0`
  - `nan_count_total = 0`
  - `min_var_block = 0.03504962846636772`
  - `max_var_block = 1.0`
- visual-only：
  - `mean_point_gate_ratio = 0.0`
  - `mean_visual_gate_ratio = 0.44791666666666663`
  - `mean_point_precision_gain_count = 0.0`
  - `mean_visual_precision_gain_count = 43.0`
  - `nan_count_total = 0`
  - `min_var_block = 0.12100696563720703`
  - `max_var_block = 1.0`
- stale scaffold + visual fallback 单步检查：
  - `fresh_scaffold = False`
  - `point_precision_gain_count = 0`
  - `visual_precision_gain_count = 96`
  - `visual_gate_ratio = 1.0`
  - `posterior_prior_equal_on_stale = False`
  - `stale_prior_match_error = 0.3333333432674408`
  - `nan_count = 0`

这里最后一项非常重要：

- 在 point-only 管线里，stale scaffold 时 `posterior == prior`
- 在 visual stage-1 管线里，若 stale scaffold 但 visual 仍 fresh 且 center-patch fallback 可用，则 posterior **不应**等于 prior
- 因此 visual stage-1 的正确语义不是“所有 stale 都退 prior”，而是“stale 时 point expert 关闭；若 visual 也不可用才严格退 prior”

这与总纲中的 stale scaffold policy 和 visual transport-only fallback 是一致的。


## 15. 关于 `max_points`

当前 posterior 审计脚本默认使用：

- `max_points=256`

而 scaffold acceptance 默认仍然常用：

- `max_points=128`

原因不是“posterior 需要放宽阈值”，而是当前 synthetic mini CALVIN 数据在 `max_points=128` 下过稀，point expert 往往拿不到足够 `anchor_count`，无法代表 posterior 是否真实工作。

因此：

- scaffold 脚本继续用 128 做稳定性测试
- posterior 脚本应使用 256 测 point gate 和 Gaussian fusion
- visual stage-1 脚本默认也沿用 256


## 16. 已知限制

- 当前 `EndEffectorLocalFrame` 依赖 CALVIN 风格 `robot_obs[0:6]`
- 当前 scaffold 不是 learnable query 版本
- 当前 point expert 不是 SONATA learned point feature 版本
- 当前 visual 只接 `rgb_static` 单相机
- 当前没有 tactile，所以不代表完整多模态 posterior
- 当前 acceptance 仍然基于 mini CALVIN 与局部脚本审计，不等于真实实验台闭环通过
- 当前 CPU 下真实 V-JEPA 2.1 base 前向较慢；正式部署必须在目标 GPU 环境下重新测时延预算


## 17. 是否可以开始 AnyTouch

当前结论是：

- **可以开始 AnyTouch 部署**
- 但建议继续保持 PICF 独立支线，不要并回主训练链

原因：

- current prior、point expert、visual expert、fusion 的核心边界已经稳定
- visual 已经作为 canonical expert 进入 posterior，且不污染 support identity
- point-only baseline 仍保留为 regression anchor
- tactile 是下一个自然扩展的 canonical expert

这里的“可以开始”指的是：

- V-JEPA visual stage-1 已达到可作为下一阶段稳定基座的程度
- 架构边界已经明确，AnyTouch 不需要回头修改 visual 的接口契约
- 但这**不等于**“当前 point + visual 已经完成真实实验台闭环验收”

在进入 AnyTouch 实现前，当前 visual 阶段仍有两个保留事项：

- 目标 GPU 环境下的 `p95(state_update)` 时延预算尚未正式测量
- 真实机器人 / 真传感器 replay 尚未覆盖，仅完成 CALVIN 路径与工程级 acceptance

因此更准确的结论是：

- **可以开始 AnyTouch 的架构与实现工作**
- **不应把 current visual stage-1 误记为最终实验台闭环通过**

AnyTouch 阶段的新增约束应继续保持：

- tactile 只作为 canonical expert 进入 posterior
- tactile 不进入 support identity path
- tactile gate 必须显式 contact-aware
- tactile stale / availability 必须走 runtime metadata，不得伪造当前观测


## 18. 下一步建议

下一阶段建议按下面顺序推进：

1. 保持 point-only 与 visual-stage1 两套 full check 持续可跑
2. 引入 AnyTouch tactile wrapper，但先只做 frozen backbone + tactile constructor
3. 加入 tactile gate 与 `point + visual + tactile` Gaussian fusion
4. 新增 tactile stage-1 spec audit，继续阻止 object / semantic / context 越界
5. tactile 路线稳定后，再讨论 predictive prior / JEPA

当前**不建议**做的事：

- 不建议现在并回 `pi0_pytorch.py` 主链
- 不建议现在引入 object shell
- 不建议现在把 visual 反写 identity
- 不建议现在上 multi-camera aggregation

对 AnyTouch 来说，当前 README 已经可以直接作为下一阶段的起点文档：

- runtime metadata 的位置已明确
- posterior expert 的接口风格已固定
- visual stage-1 的 boundary 可以直接作为 tactile stage-1 的模板
