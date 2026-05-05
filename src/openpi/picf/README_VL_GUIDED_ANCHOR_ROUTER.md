# VL-Guided Anchor Router for PICF

Date: 2026-05-05
Status: live implementation and long-run safety record. The current local code implements the
default-off safety substrate plus gated live consumption in the core:
PaliGemma spatial-token metadata, resize-with-pad view metadata,
transform-aware PaliGemma-grid-to-PICF-grid mapping, `PicfVLGroundingState`,
router config, column-normalized heatmap-to-point prior helpers,
`_build_vl_grounding(...)`, role-aware observation-anchor seeding and attention
bias, task-readout point-prior fusion, and posterior-binding soft overlap bias.
The router remains disabled by default and does not alter the current canonical
training profile unless `vl_anchor_router_enabled=True`.

Latest local fix audit:

```text
/tmp/picf_vl_router_anchor_collapse_fix_audit_20260505.md
```

This document records the planned PICF-compatible grounding upgrade:

```text
PaliGemma image-language encoder
-> language-conditioned 2D heatmaps / semantic anchor proposals
-> existing PICF projective geometry maps 2D heatmaps to 3D point priors
-> PICF observation anchors, task readout, and posterior binding consume those
   priors as soft gated biases
-> PI0.5 action generation remains the final action path
```

The core rule is strict:

```text
PaliGemma must not directly overwrite posterior state.
PaliGemma must not directly replace PICF anchors.
PaliGemma must not ingest raw 3D point / tactile tokens inside the VLM backbone.
```

PaliGemma is the language-conditioned 2D grounding teacher. PICF remains the
3D, tactile, posterior-memory, world-model, and action-conditioning system.

## 0. Current Local Implementation Scope

Implemented in the current local branch:

- `PaliGemmaSemanticFeatures` can carry spatial image/text tokens and per-view
  metadata.
- `PaliGemmaViewTransform` records the `resize_with_pad` geometry needed to map
  PaliGemma heatmaps back to original image coordinates.
- `_map_pg_heatmap_to_visual_grid(...)` and `_map_pg_grid_values_to_visual_grid(...)`
  consume `PaliGemmaViewTransform`; non-square images and padded regions are
  mapped through original-camera pixel centers rather than a naive grid resize.
- `PicfVLGroundingState` exists as a typed carrier.
- `PicfCoreConfig` and `scripts/picf_core_train.py` expose default-off VL
  router knobs. The standard trainer can enable the router without ad-hoc code
  patches.
- `_point_prior_from_heatmap(...)` uses column-normalized compatibility so
  point-dense regions do not automatically win.
- invalid projection returns `valid=False` and zero point prior, not a fake
  top-left fallback.
- `_build_vl_grounding(...)` can build heatmaps/point priors/anchor proposals
  only when `vl_anchor_router_enabled=True`.
- `_build_observation_anchors(...)` consumes valid VL priors as soft role-aware
  query seeds and point-attention log-prior bias when the router is enabled.
- `_build_task_readout(...)` fuses direct task point attention with role-aware
  VL point priors through a learnable gate when the router is enabled.
- `_posterior_update(...)` consumes VL/observation point-overlap as a soft
  binding-logit bias when the router is enabled.
- point geometry now keeps a row-aligned world-frame coordinate stream:
  `PointFrameContext.points_world` -> `PicfTokenFieldState.point_positions_world`.
  Camera projection, tactile alignment, VL point-prior moments, observation
  anchor geometry, and task-readout `x/S/a` use this world-frame stream instead
  of projecting local-frame rows as if they were world coordinates.
- scene/object candidate routing now uses a role-aware projective candidate
  mask. Scene/object slots prefer global rows that are visible, depth-valid,
  and not clipped by the visual-grid border; if too few such rows exist, the
  code falls back to remaining global rows only to keep tensor contracts filled.
- VL task/interaction lift uses the strict scene candidate mask with **no**
  global fallback. If every scene row is an invalid border/depth/visibility
  candidate, the VL point prior is invalid and no-op. The global fallback is
  reserved only for observation-anchor coverage seeding.
- role-0 observation anchors keep the local/proprio/tactile seed contract.
  Static-camera VL priors are not allowed to replace role-0 seeds and are not
  added as point-attention log-prior bias for role-0 rows.
- serving anchor diagnostics now export `point_cloud.xyz_world`,
  `point_cloud.projectable_mask`, and `vl_grounding.*` heatmap / anchor-prior
  summaries so a top-left/top-border overlay can be traced back to heatmap,
  projection, point prior, or task attention.

Not implemented yet:

- VL heatmap/keypose/diversity losses
- any default behavior change in the active training profile

### 0.1A Diagnostic Fix: Top-Border Scene Anchor Collapse

The first `vl_anchor_router_enabled=True` CALVIN diagnostic run at step 500
showed a real failure mode:

```text
effector role-0 anchors:
  mostly near the gripper, as expected

scene/object role-1 anchors:
  concentrated on y ~= 0 top-border projected points
  top point rows were global-scene pool rows with near-floor/boundary geometry
```

This was not a drawing issue. The JSONL showed that task and observation
point weights were actually selecting global rows whose projected pixels were
on the top image border. The root cause was a combination of three issues:

```text
1. global_scene_context was sampled by FPS from the whole point cloud.
   Large planes and camera-boundary rows could become object-role candidates.

2. PointFrameContext.points_local was overloaded.
   The unified context concatenated local effector-frame rows and world-frame
   global scene rows into one tensor, while projective geometry assumed all
   rows were world coordinates.

3. The live router has no nonzero heatmap/keypose/diversity loss yet.
   At step 500, the PaliGemma heatmap head is still weakly constrained, so
   scene/object slots must be protected by geometric validity masks rather than
   assuming the router has learned object grounding.
```

The fix is upstream and geometric, not cosmetic:

```text
PointFrameContext
  points_local        : model-frame feature coordinates
  points_world        : row-aligned world coordinates for projection/alignment

PicfTokenFieldState
  point_positions       : existing model-frame point coordinates
  point_positions_world : row-aligned world coordinates
  point_projectable_mask: scene/object projective candidate rows

_build_projective_geometry
  consumes point_positions_world

_scene_point_candidate_mask
  keeps global-scene rows that are visible, depth-valid, and not visual-grid
  border-clipped

_fused_read_role_bias
  role 0 can read local effector/contact rows
  role 1 can read scene/object candidate rows
  role 1 falls back to global rows only when no candidate rows exist

_build_vl_grounding
  task/interaction heatmaps lift only through strict scene/object candidate rows
  with no global fallback; invalid all-border/all-invisible rows produce an
  explicit zero/invalid prior
  anchor moments use point_positions_world

_build_task_readout / _build_observation_anchors
  task and observation geometry moments use point_positions_world
```

This means future diagnostics should distinguish:

```text
point_cloud.xyz:
  model-frame coordinates used by existing point token features

point_cloud.xyz_world:
  world coordinates used by camera projection, tactile alignment, and
  anchor/task geometry

point_cloud.projectable_mask:
  rows allowed as scene/object candidates for current point-centric router

vl_grounding.task_heatmap / effector_heatmap / interaction_heatmap:
  PaliGemma heatmap distributions after transform-aware mapping to the PICF
  visual grid

vl_grounding.anchor_point_prior:
  actual point priors consumed by observation/task/posterior gates
```

The current compatibility target remains:

```text
vl_anchor_router_enabled=False
-> current PICF behavior is unchanged
```

### 0.1B Long-Run Safety Closeout: Strict Lift, Local Effector Anchors

The long-run launch contract is stricter than the early diagnostic branch:

```text
VL lift path:
  _build_vl_grounding(...)
  -> _scene_point_candidate_mask(..., fallback_to_global=False)
  -> _point_prior_from_heatmap(...)

Coverage seed path:
  _build_observation_anchors(...)
  -> _scene_point_candidate_mask(..., fallback_to_global=True)
  -> fill scene/object slots so tensor contracts remain valid

Effector observation path:
  role 0 slots use local effector/contact point rows
  role 0 slots do not receive static-camera VL seed overrides
  role 0 slots do not receive static-camera VL point-prior attention bias

Scene/object observation path:
  role 1 slots may receive valid task/interaction VL seeds and log-prior bias
  role 1 slots fall back to global FPS only for coverage, not for claiming a
  valid VL prior
```

This distinction matters because a diagnostic heatmap can be weak early in
training. A fallback scene seed is acceptable for coverage. A fallback VL prior
is not acceptable because it would pretend that a clipped border point is a
valid language-conditioned anchor.

### 0.1 Current Mathematical Guardrails

The current checked-in code implements the safe substrate and gated core
consumers. The following invariants are executable and should stay true before
the router is made default-on or trained with nonzero VL losses:

```text
1. Default-off no-op:
   vl_anchor_router_enabled=False
   -> no VL heatmap head, no VL anchor projection module, no VL gate parameters
   -> no PicfVLGroundingState is attached to the live core state

2. Transform-aware heatmap mapping:
   flat heatmaps remain non-negative and sum to one after
   resize-with-pad-aware mapping from PaliGemma grid to PICF visual grid.

3. Column-normalized 2D-to-3D lift:
   each visual cell distributes its heatmap mass across compatible points.
   Point-dense regions cannot win purely because they contain more rows.

4. Projectable-mask enforcement:
   local-frame point rows are excluded from VL lift unless explicitly converted
   into world/projectable coordinates.

5. Invalid projection is explicit:
   all-zero / invisible projective compatibility returns valid=false and a zero
   prior. It must not silently create a top-left or first-point fallback.

6. Multi-mode proposal sanity:
   weighted NMS preserves separated high-confidence modes instead of collapsing
   every proposed anchor to one point.

7. Enabled gated-consumer state:
   when vl_anchor_router_enabled=True, the core can build typed heatmap, point
   prior, and proposal tensors, consume them through observation/task/posterior
   soft gates, keep output tensor contracts stable, and expose debug keys.

8. No hard posterior overwrite:
   VL grounding may add binding-logit bias, but it must not directly write
   posterior `x/S/a/h/c/mu/Sigma`.

9. Strict scene lift:
   task/interaction VL priors use no global fallback after border, visibility,
   and depth filtering. If no row survives, the prior is invalid/no-op.

10. Local effector observation anchors:
   role-0 observation anchors are not seeded or point-biased by static-camera
   VL global priors. They stay local/proprio/tactile by construction.
```

### 0.2 Current Verification Commands

Use `uv` from the repository root:

```bash
cd /home/siyuanyue/Documents/openpi

uv run python -m py_compile \
  scripts/verify_picf_contract.py \
  scripts/picf_core_train.py \
  scripts/serve_picf_policy.py \
  src/openpi/picf/paligemma/wrapper.py \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/pipeline.py

uv run pytest -q src/openpi/picf/core/pipeline_test.py -k 'vl_point_prior_projectable or scene_point_candidate_mask or vl_slot_point_priors or vl_observation_seed or vl_grounding_enabled or effector_and_scene_anchor_roles'
uv run pytest -q src/openpi/picf/core/pipeline_test.py -k 'vl_'
uv run pytest -q src/openpi/picf/core/pipeline_test.py -k 'semantic_context_carries or scene_point_candidate or effector_and_scene_anchor_roles'
uv run pytest -q scripts/picf_core_train_test.py -k 'vl_anchor_router or scene_anchor_border or save_interval or checkpoint'
uv run pytest -q scripts/picf_core_train_test.py -k 'vl_anchor_router or conditioned_control'
uv run pytest -q src/openpi/picf/paligemma/wrapper_test.py
uv run pytest -q src/openpi/picf/core/pipeline_test.py
uv run pytest -q src/openpi/picf/core/training_test.py
uv run pytest -q scripts/picf_core_train_test.py
uv run python scripts/verify_picf_contract.py
git diff --check
```

The `-k 'vl_'` test slice is the strict mathematical smoke for the staged
router substrate. The full pipeline, wrapper, training, trainer, and verifier
commands ensure that the default-off substrate does not break the existing
PICF v2.2 runtime contract.

The first targeted command is the minimum pre-launch gate for long runs. It
checks the exact failure cases that produced earlier anchor collapse:
projectable-mask exclusion of local rows, strict-vs-coverage scene masks,
role-aware slot priors, no static-VL override of role-0 observation seeds,
enabled router tensor contracts, and effector/scene point-pool separation.

## 1. Motivation

The current PICF v2.2 code already has strong physical structure:

- point, visual, tactile, proprio, and semantic streams are built separately
- projective geometry already maps 3D points to visual grid cells
- observation anchors already maintain effector-vs-scene roles
- posterior anchors already maintain recurrent physical memory
- task readout already produces task-local geometric summaries
- PI0.5/PaliGemma remains the final action-generator path

The observed diagnostic weakness is that task/readout anchors can still collapse
onto visually salient but task-irrelevant regions. The right fix is not a
post-hoc drawing threshold and not a hard replacement of PICF anchors. The right
fix is a coherent grounding path:

```text
language-conditioned PaliGemma image tokens
-> explicit supervised spatial heatmaps
-> camera/depth-compatible projection into point space
-> multi-mode 3D anchor priors
-> soft gated PICF anchor/readout/binding bias
```

This is designed to avoid three failure modes:

- task query collapse into a single irrelevant point
- all scene slots being consumed by gripper-local evidence
- semantic grounding being unable to influence 3D anchor selection

## 2. External Design Anchors

Primary references:

- BridgeVLA: https://arxiv.org/abs/2506.07961
- PosA-VLA: https://arxiv.org/abs/2512.03724
- FALCON: https://arxiv.org/abs/2510.17439
- AnchorVLA4D: https://arxiv.org/abs/2603.12730

The implementation should borrow principles, not copy mechanisms blindly.

### 2.1 BridgeVLA Principle

Use VLM features to predict spatial 2D heatmaps, then lift the heatmaps into
3D. Do not make the VLM backbone consume raw 3D tokens as if they were normal
image-language tokens.

PICF-compatible translation:

```text
PaliGemma image tokens -> 2D heatmap
PICF projective_compatibility [P,V] -> point prior [P]
```

BridgeVLA-style reasoning maps cleanly onto PICF because PICF already owns the
point cloud and the projection matrix from points to visual grid cells.

### 2.2 PosA-VLA Principle

Use pose-conditioned anchor supervision from demonstrations. In robotics,
language grounding alone is not enough; gripper/keypose supervision provides a
physical attention target.

PICF-compatible translation:

```text
current gripper pose projection -> effector heatmap target
future gripper open/close or keypose projection -> interaction heatmap target
optional object labels/masks -> task heatmap target
```

The supervision must be role-aware so that gripper slots do not absorb all
object/task slots.

### 2.3 FALCON Principle

Keep spatial grounding on the action/PICF side rather than injecting raw spatial
tokens into the VLM backbone. This protects the language-visual pretraining
space and keeps the action-side geometry explicit.

PICF-compatible translation:

```text
PaliGemma produces grounding priors.
PICF consumes grounding priors.
PaliGemma does not become the 3D memory system.
```

### 2.4 AnchorVLA4D Principle

Long-horizon manipulation benefits from persistent anchor context. PICF already
has posterior anchors and recurrent carry, so the right use is semantic binding
bias, not direct memory overwrite.

PICF-compatible translation:

```text
VL proposals help current observation anchors bind to persistent anchors.
Persistent anchors still update through PICF physical evidence.
```

## 3. Current Code Audit

This section records the current local code facts this design must respect.

### 3.1 PaliGemma Wrapper

File: `src/openpi/picf/paligemma/wrapper.py`

Current `PaliGemmaSemanticFeatures` contains:

```python
tokens: torch.Tensor
summary: torch.Tensor
prefix_embeddings: torch.Tensor | None
prefix_pad_masks: torch.Tensor | None
prefix_att_masks: torch.Tensor | None
```

Current `encode_observation(...)` already computes:

```python
prefix_output = PaliGemma(prefix_embs, masks, positions)
image_hidden = prefix_output[:, :image_token_count, :]
text_hidden = prefix_output[:, image_token_count:, :]
```

But the wrapper currently returns only valid prefix tokens and summary. It does
not return:

```text
image_hidden tokens
text_hidden tokens
per-view image token ranges
per-view grid shapes
view names
```

Therefore the first implementation step must expose these spatial image tokens
without changing PI0.5 action generation.

### 3.2 Semantic Context

File: `src/openpi/picf/core/pipeline.py`

Current `_SemanticContext` contains:

```python
tokens: torch.Tensor
prefix_tokens: torch.Tensor
available: bool
```

When the semantic override is a `PaliGemmaSemanticFeatures`, the core currently
projects only `semantic_override.tokens`.

Therefore the router needs an expanded semantic context:

```python
summary: torch.Tensor | None
image_tokens: torch.Tensor | None
text_tokens: torch.Tensor | None
image_token_ranges: tuple[tuple[int, int], ...]
image_grid_shapes: tuple[tuple[int, int], ...]
image_view_names: tuple[str, ...]
```

These fields are read-only grounding carriers. They must not change legacy
semantic token projection when the router is disabled.

### 3.3 Projective Geometry

File: `src/openpi/picf/core/contracts.py`

Current `PicfProjectiveGeometryState` already contains:

```python
point_proj_grid_norm: Tensor          # [P,2]
point_proj_grid_index: Tensor         # [P,2]
point_visibility: Tensor              # [P]
point_depth: Tensor                   # [P]
point_depth_sample: Tensor            # [P]
point_depth_valid: Tensor             # [P]
visual_grid_norm: Tensor              # [V,2]
visual_grid_index: Tensor             # [V,2]
visual_pixel_centers: Tensor          # [V,2]
visual_ray_world: Tensor              # [V,3]
camera_origin_world: Tensor           # [3]
projective_compatibility: Tensor      # [P,V]
projective_candidate_mask: Tensor     # [P,V]
projective_attention_bias: Tensor     # [P,V]
```

File: `src/openpi/picf/core/pipeline.py`

Current `_build_projective_geometry(...)` computes:

```text
3D point p -> static-camera uv -> visual-grid coordinate
visual patch v -> ray and grid coordinate
C[p,v] = exp(-patch_distance^2 / sigma^2)
         * depth_consistency
         * visibility
```

This is the exact bridge needed for the new router:

```python
point_prior = projective_compatibility @ heatmap_on_visual_grid
```

The implementation must never create a fake point prior at pixel `(0,0)` when
projection is invalid. Invalid projection should produce:

```text
valid = false
confidence = 0
prior = all zeros or a masked no-op distribution
debug says projection_invalid
```

### 3.4 Observation Anchors

File: `src/openpi/picf/core/pipeline.py`

Current `_build_observation_anchors(...)`:

```text
first effector_observation_count slots -> role 0
remaining slots                         -> role 1 scene/object

role 0 seeds from local/effector point pool by FPS
role 1 seeds from global/scene point pool by FPS

obs_reader reads fused tokens
visual_native_reread reads visual payload
point weights produce x, S, a
```

This contract should remain. The router should only add scene/object proposals
inside the role-1 budget and keep residual coverage anchors.

Do not let VL priors consume all effector anchors. Do not let gripper priors
consume all scene anchors.

### 3.5 Task Readout

File: `src/openpi/picf/core/pipeline.py`

Current `_build_task_readout(...)`:

```text
task query tokens
-> semantic conditioner reads PaliGemma semantic tokens
-> public reader reads [fused_tokens, visual_tokens]
-> point_public_attention = fused_attention[:, :point_count]
-> local task x/S/a come from normalized point_public_attention
```

Current taskpoint math:

```python
point_weights = point_public_attention[:local_count]
point_weights = point_weights / point_weights.sum(-1, keepdim=True)
x = point_weights @ point_positions
S = weighted_cov(point_positions, point_weights, x)
a = extent_from_cov(S)
```

This means taskpoint currently depends on learned task queries and the public
read attention. It does not explicitly use language-conditioned image heatmaps.

The router should add:

```text
direct point weights from task readout
+ gated point prior from PaliGemma heatmap
-> normalized fused point weights
```

### 3.6 Posterior Update

File: `src/openpi/picf/core/pipeline.py`

Current `_posterior_update(...)`:

```text
prior from recurrent carry
observation anchors from current frame
binding logits from hidden/geometric compatibility
+ role bias
-> Sinkhorn dustbin binding
-> anchor reader evidence
-> posterior x/S/a update through binding support
```

Semantic does not currently enter posterior binding. That is a deliberate v2.2
contract. The VL router must respect this in stages:

```text
Stage 0/1:
  no posterior semantic binding

Stage 2:
  optional low-gate semantic binding bias only

Forbidden:
  direct posterior x/S/a overwrite from PaliGemma heatmap
```

### 3.7 Alignment Loss

File: `src/openpi/picf/core/training.py`

Current `compute_alignment_loss(...)` already supervises point-visual
consistency through:

```text
anchor_pv
pv_weak
focus_pv
pt
```

These losses are based on `projective_compatibility` and routing consistency.
The new VL grounding losses should be added as a separate loss family, not mixed
silently into existing alignment terms.

## 4. Design Invariants

The implementation must satisfy these invariants:

```text
I1. Router disabled -> exact old behavior.
I2. All new gates initialize near zero.
I3. PaliGemma never receives raw point/tactile/PICF anchor tokens.
I4. PaliGemma never hard-writes posterior memory.
I5. PaliGemma heatmaps are priors, not labels of absolute truth.
I6. PICF point/depth/visibility owns the 2D-to-3D lift.
I7. Role budgets remain explicit: effector/contact vs scene/object.
I8. Invalid projection is a no-op, not a fake top-left point.
I9. Diagnostics must expose raw distributions, not hide failures by thresholding.
I10. CALVIN/PI0.5 action flow remains the final action loss path.
```

Additional constraints enforced by the current staged implementation:

```text
B1. PaliGemma heatmaps are in resize-with-pad coordinates.
    Current code carries `PaliGemmaViewTransform` through `_SemanticContext` and
    samples the PaliGemma grid at the padded-image coordinates corresponding to
    PICF's original-camera pixel centers. Plain interpolation is only a legacy
    fallback for old semantic features that lack transform metadata.

B2. VL 2D-to-3D lift only applies to world-projectable point rows.
    Current PICF point rows can include local effector-frame rows and global
    scene rows. MVP masks to `point_pool_ids == 1` before applying VL priors.

B3. Prior assignment must be role-aware.
    Effector priors can only feed effector/contact slots. Task/object and
    interaction priors feed scene/object slots. Index-order copying is not a
    valid implementation.

B4. Heatmap-to-point lift uses column-normalized compatibility by default.
    This conserves heatmap mass per visual cell and avoids point-density bias.

B5. Exact no-op and near-zero gates are different contracts.
    `vl_anchor_router_enabled=False` is the exact old-path contract. Negative
    gate initialization is only approximate no-op and must be tested separately.
```

The design should be implemented so the following test passes:

```text
vl_anchor_router_enabled = False
or all router gates = 0

Then:
  observation anchors match previous implementation within tolerance
  task_readout point weights match previous implementation within tolerance
  posterior binding logits match previous implementation within tolerance
  action loss path receives the same extra_prefix_tokens as before
```

## 5. New Data Structures

### 5.1 PaliGemmaSemanticFeatures Extension

Current extension:

```python
@dataclasses.dataclass(frozen=True)
class PaliGemmaSemanticFeatures:
    tokens: torch.Tensor
    summary: torch.Tensor
    prefix_embeddings: torch.Tensor | None = None
    prefix_pad_masks: torch.Tensor | None = None
    prefix_att_masks: torch.Tensor | None = None

    image_tokens: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    image_token_ranges: tuple[tuple[int, int], ...] = ()
    image_grid_shapes: tuple[tuple[int, int], ...] = ()
    image_view_names: tuple[str, ...] = ()
    image_view_transforms: tuple[PaliGemmaViewTransform, ...] = ()
```

Shape contract:

```text
image_tokens: [V_pg_total, D_pg]
text_tokens: [T_text, D_pg]
image_token_ranges[i] = [start_i, end_i) into image_tokens
image_grid_shapes[i] = (H_pg_i, W_pg_i)
image_view_names[i] = "static" or "gripper"
image_view_transforms[i] = resize-with-pad geometry for that view
```

Implementation detail:

```text
If PaliGemma is frozen and encode_observation uses torch.inference_mode(),
return image_tokens/text_tokens through detach().clone() before giving them to
trainable heatmap heads. This avoids inference-tensor autograd metadata issues.
```

### 5.2 Semantic Context Extension

Current `_SemanticContext`:

```python
@dataclasses.dataclass(frozen=True)
class _SemanticContext:
    tokens: torch.Tensor
    prefix_tokens: torch.Tensor
    available: bool
    summary: torch.Tensor | None = None
    image_tokens: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    image_token_ranges: tuple[tuple[int, int], ...] = ()
    image_grid_shapes: tuple[tuple[int, int], ...] = ()
    image_view_names: tuple[str, ...] = ()
    image_view_transforms: tuple[Any, ...] = ()
```

Legacy callers that pass raw semantic tensors still produce the old token-only
context with `image_tokens=None`.

### 5.3 PicfVLGroundingState

New contract in `src/openpi/picf/core/contracts.py`:

```python
@dataclasses.dataclass
class PicfVLGroundingState:
    task_heatmap_logits: torch.Tensor          # [Vv]
    effector_heatmap_logits: torch.Tensor      # [Vv]
    interaction_heatmap_logits: torch.Tensor   # [Vv]

    task_heatmap: torch.Tensor                 # [Vv]
    effector_heatmap: torch.Tensor             # [Vv]
    interaction_heatmap: torch.Tensor          # [Vv]

    task_point_prior: torch.Tensor             # [P]
    effector_point_prior: torch.Tensor         # [P]
    interaction_point_prior: torch.Tensor      # [P]

    anchor_point_priors: torch.Tensor          # [Kvl,P]
    anchor_x: torch.Tensor                     # [Kvl,3]
    anchor_S: torch.Tensor                     # [Kvl,3,3]
    anchor_tokens: torch.Tensor                # [Kvl,H]
    anchor_roles: torch.Tensor                 # [Kvl]
    anchor_scores: torch.Tensor                # [Kvl]

    visual_pixel_centers: torch.Tensor | None  # [Vv,2]
    valid: torch.Tensor                        # scalar bool or [Kvl]
    confidence: torch.Tensor                   # scalar or [Kvl]
```

`Vv` is the PICF visual grid token count, not necessarily PaliGemma's raw image
token count. If PaliGemma and PICF visual grids differ, heatmaps are mapped
through `PaliGemmaViewTransform` before projection. Naive interpolation is only
a compatibility fallback for legacy features without view-transform metadata.

## 6. Config Surface

Add explicit config fields. Defaults must be no-op safe:

```python
vl_anchor_router_enabled: bool = False
vl_grounding_view: str = "static"
vl_heatmap_hidden_dim: int = 512
vl_anchor_modes: int = 6
vl_anchor_nms_radius_m: float = 0.04
vl_anchor_local_sigma_m: float = 0.04
vl_min_visible_mass: float = 1e-4
vl_heatmap_temperature: float = 1.0

vl_obs_anchor_gate_init: float = -4.0
vl_task_point_gate_init: float = -4.0
vl_posterior_bind_gate_init: float = -6.0

lambda_vl_heatmap_task: float = 0.0
lambda_vl_heatmap_effector: float = 0.0
lambda_vl_heatmap_interaction: float = 0.0
lambda_vl_point_consistency: float = 0.0
lambda_vl_anchor_diversity: float = 0.0
```

Rationale:

```text
default False / zero loss -> no behavior change
negative gate init -> router can be enabled without immediate hard takeover
loss weights explicit -> no hidden training objective change
```

Trainer CLI:

```bash
python scripts/picf_core_train.py ... \
  --vl-anchor-router-enabled \
  --vl-grounding-view static \
  --vl-anchor-modes 6 \
  --vl-anchor-nms-radius-m 0.04 \
  --vl-anchor-local-sigma-m 0.04 \
  --vl-min-visible-mass 1e-4 \
  --vl-heatmap-temperature 1.0 \
  --vl-obs-anchor-gate-init -4.0 \
  --vl-task-point-gate-init -4.0 \
  --vl-posterior-bind-gate-init -6.0 \
  --vl-prior-bias-clip 4.0
```

The CLI enforces:

```text
--vl-anchor-router-enabled requires:
  picf_mode=enabled
  semantic_mode=paligemma

picf_mode=ablated forces:
  vl_anchor_router_enabled=False
```

This is intentional. The router needs PaliGemma image tokens and PICF anchor
state; it is not meaningful in PI0.5-only ablation mode.

### 6.1 Current Long-Run Launch Contract

The current VL-router long-run is a diagnostic-safe full PICF run, not a claim
that the untrained heatmap head is already a final MAPG result. Use this profile
when the goal is to collect stable long-run checkpoints and CALVIN diagnostics
without reintroducing top-border/static-VL effector contamination:

```text
steps: 30000
checkpoint cadence: every 5000 optimizer steps
unroll_steps: 2
burnin_steps: 0
picf_mode: enabled
perception_finetune_mode: frozen
VL router: enabled, gated, no default heatmap/keypose/diversity loss
scene_anchor_border_patches: 1.0
```

Operational requirements before launch:

```text
1. repository branch is Posterior_VLA and clean except intentional runtime logs
2. local tests in Section 0.2 pass
3. cloud clone is reset to origin/Posterior_VLA
4. no old torchrun, serving, or CALVIN evaluator process is occupying GPUs or
   ports 8000/8001
5. output run name includes vlrouter/strict/unroll2/30000/ckpt5000/date/retry
```

Template:

```bash
cd /root/openpi_vlrouter_longrun
export PYTHONPATH=/root/openpi_vlrouter_longrun/src:/root/openpi_vlrouter_longrun/packages/openpi-client/src
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=picf_v22_vlrouter_strict2x40_unroll2_30000_ckpt5000_YYYYMMDD_rN
LOG=/mnt/checkpoints/picf_core/debug/${RUN}.outer.log
mkdir -p /mnt/checkpoints/picf_core/debug

nohup /root/openpi/.venv/bin/torchrun --standalone --nproc_per_node=2 \
  scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --exp-name "$RUN" \
  --num-train-steps 30000 \
  --save-interval 5000 \
  --unroll-steps 2 \
  --burnin-steps 0 \
  --picf-mode enabled \
  --training-strategy ddp \
  --perception-finetune-mode frozen \
  --visual-mode encoder \
  --visual-model-name vjepa2_1_vit_base_384 \
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --visual-feature-mode final \
  --visual-real-grid 64 \
  --point-backbone sonata \
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --semantic-mode paligemma \
  --semantic-source pi0_pytorch \
  --semantic-checkpoint-path /root/openpi/checkpoints/foundation/pi05_base_pytorch \
  --semantic-checkpoint-config-path /root/openpi/checkpoints/foundation/pi05_base_pytorch/config.json \
  --semantic-max-length 256 \
  --semantic-gradient-checkpointing \
  --tactile-mode handcrafted \
  --persistent-anchors 8 \
  --observation-anchors 16 \
  --effector-persistent-anchors 1 \
  --effector-observation-anchors 1 \
  --task-local-queries 8 \
  --task-effector-queries 1 \
  --vl-anchor-router-enabled \
  --vl-anchor-modes 6 \
  --scene-anchor-border-patches 1.0 \
  --vl-obs-anchor-gate-init -4.0 \
  --vl-task-point-gate-init -4.0 \
  --vl-posterior-bind-gate-init -6.0 \
  > "$LOG" 2>&1 &
echo $! > /mnt/checkpoints/picf_core/debug/${RUN}.pid
```

Tail:

```bash
tail -f /mnt/checkpoints/picf_core/debug/${RUN}.outer.log
tail -f /mnt/checkpoints/picf_core/picf_core/${RUN}/metrics.jsonl
```

Expected save paths:

```text
/mnt/checkpoints/picf_core/picf_core/${RUN}/5000
/mnt/checkpoints/picf_core/picf_core/${RUN}/10000
/mnt/checkpoints/picf_core/picf_core/${RUN}/15000
/mnt/checkpoints/picf_core/picf_core/${RUN}/20000
/mnt/checkpoints/picf_core/picf_core/${RUN}/25000
/mnt/checkpoints/picf_core/picf_core/${RUN}/30000
```

## 7. Pseudocode: Wrapper Changes

### 7.1 Named Views

Current wrapper has `_views(...) -> list[np.ndarray]`. Planned:

```python
def _named_views(self, observation: PicfObservation) -> list[tuple[str, np.ndarray]]:
    views = [("static", np.asarray(observation.rgb_static))]
    if self.config.include_gripper_image and observation.rgb_gripper is not None:
        views.append(("gripper", np.asarray(observation.rgb_gripper)))
    return views
```

### 7.2 Prefix Embedding With Ranges

Planned `_embed_prefix(...)` return:

```python
return (
    prefix_embs,
    prefix_pad_masks,
    prefix_att_masks,
    image_token_count,
    lang_masks,
    image_token_ranges,
    image_grid_shapes,
    image_view_names,
)
```

Pseudocode:

```python
cursor = 0
for view_name, image in self._named_views(observation):
    image_tensor = self._prepare_image(image)
    img_emb = self._apply_checkpoint(_image_embed, image_tensor)
    num_img_tokens = int(img_emb.shape[1])

    start = cursor
    end = cursor + num_img_tokens
    image_token_ranges.append((start, end))
    image_view_names.append(view_name)

    grid = infer_square_grid_or_raise(num_img_tokens)
    image_grid_shapes.append(grid)

    cursor = end
    image_token_count += num_img_tokens
```

Do not silently accept a non-square grid. If PaliGemma changes token layout, the
router needs an explicit mapping instead of guessing.

### 7.3 Returning Image Tokens

Pseudocode:

```python
prefix_output = self._apply_checkpoint(...)
image_hidden = prefix_output[:, :image_token_count, :]
text_hidden = prefix_output[:, image_token_count:, :]

features = PaliGemmaSemanticFeatures(
    tokens=_take_valid_prefix_tokens(prefix_output[0], prefix_pad_masks[0]),
    summary=_summary_from_outputs(...),
    prefix_embeddings=prefix_embs[0],
    prefix_pad_masks=prefix_pad_masks[0],
    prefix_att_masks=prefix_att_masks[0],
    image_tokens=image_hidden[0].detach().clone(),
    text_tokens=text_hidden[0].detach().clone(),
    image_token_ranges=tuple(image_token_ranges),
    image_grid_shapes=tuple(image_grid_shapes),
    image_view_names=tuple(image_view_names),
)
```

The returned image tokens are for grounding heads only. They are not appended to
PI0.5 action suffixes and do not change `compute_action_flow_loss(...)`.

## 8. Pseudocode: VL Grounding Head

Add small heads inside `PicfFullCore`:

```python
self.vl_heatmap_head = nn.Sequential(
    nn.LazyLinear(self.config.vl_heatmap_hidden_dim),
    nn.GELU(),
    nn.LayerNorm(self.config.vl_heatmap_hidden_dim),
    nn.Linear(self.config.vl_heatmap_hidden_dim, 3),
)
self.vl_anchor_token_proj = nn.LazyLinear(self.config.hidden_dim)
self.vl_task_point_gate_logit = nn.Parameter(torch.tensor(self.config.vl_task_point_gate_init))
self.vl_obs_anchor_gate_logit = nn.Parameter(torch.tensor(self.config.vl_obs_anchor_gate_init))
self.vl_posterior_bind_gate_logit = nn.Parameter(torch.tensor(self.config.vl_posterior_bind_gate_init))
```

The head outputs three heatmaps:

```text
task_heatmap:
  language target / object / destination region

effector_heatmap:
  current gripper / hand / contact support region

interaction_heatmap:
  future contact / keypose / affordance region
```

The head is intentionally small. The goal is not to turn PaliGemma into a dense
segmentation model; the goal is to produce useful language-conditioned priors
for PICF.

## 9. Pseudocode: 2D Heatmap To 3D Point Prior

### 9.1 Map PaliGemma Heatmap To PICF Visual Grid

PaliGemma image grid and PICF visual grid may differ.

```python
def _map_pg_heatmap_to_visual_grid(
    heat: Tensor,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    view_transform: PaliGemmaViewTransform | None,
) -> Tensor:
    if view_transform is None:
        # Legacy compatibility only. New PaliGemma features should always carry
        # view_transform.
        return _resize_flat_heatmap(heat, src_hw, dst_hw)

    # PICF visual pixels are in the original camera image.
    original_uv = picf_visual_grid_centers(view_transform.original_hw, dst_hw)

    # Map original camera pixel centers into the padded 224x224 PaliGemma image.
    padded_uv = resize_with_pad_forward(
        original_uv,
        scale_x=view_transform.scale_x,
        scale_y=view_transform.scale_y,
        pad_left=view_transform.pad_left,
        pad_top=view_transform.pad_top,
    )

    # Sample the PaliGemma token grid at those padded-image coordinates.
    sampled = grid_sample_pg_tokens(heat.reshape(src_hw), padded_uv, view_transform.target_hw)
    sampled = sampled.reshape(-1).clamp_min(0.0)
    return sampled / sampled.sum().clamp_min(eps)
```

The destination grid must match `token_field.projective_geometry.visual_grid_index`.
This mapping is intentionally not a naive `F.interpolate` when transform
metadata exists. It prevents padded top/bottom/left/right regions from becoming
fake task anchors.

### 9.2 Build Heatmaps

```python
img_tokens = select_view_tokens(semantic.image_tokens, semantic.image_token_ranges, view="static")
text_summary = semantic.summary or semantic.tokens.mean(dim=0)
text_summary = text_summary.expand(img_tokens.shape[0], -1)

head_in = torch.cat([img_tokens, text_summary], dim=-1)
logits = self.vl_heatmap_head(head_in)

task_logits = logits[:, 0]
eff_logits = logits[:, 1]
int_logits = logits[:, 2]

task_heat_pg = softmax(task_logits / temperature)
eff_heat_pg = softmax(eff_logits / temperature)
int_heat_pg = softmax(int_logits / temperature)
```

### 9.3 Lift To Points

Let:

```text
C = token_field.projective_geometry.projective_compatibility  # [P,Vv]
h = heatmap_on_picf_visual_grid                               # [Vv]
```

Then:

```python
point_prior = C @ h
visible_mass = point_prior.sum()
if visible_mass <= vl_min_visible_mass:
    valid = False
    point_prior = zeros_like(point_prior)
else:
    valid = True
    point_prior = point_prior / visible_mass
```

This is the BridgeVLA-style back-projection adapted to PICF's point support.

Important:

```text
Do not replace invalid projection with a uniform prior by default.
Do not replace invalid projection with top-left pixel.
Do not hide invalid projection in visualization.
```

Uniform fallback can be tested as an ablation, but not as the default contract.

## 10. Pseudocode: Multi-Mode Anchor Proposal

Direct top-1 selection is not enough. It can reproduce collapse.

Use weighted soft-NMS or weighted FPS over points:

```python
def _weighted_anchor_modes(positions: Tensor, weights: Tensor, count: int, radius_m: float) -> Tensor:
    w = weights.clamp_min(0).clone()
    chosen = []
    for _ in range(count):
        if w.max() <= eps:
            break
        idx = argmax(w)
        chosen.append(idx)

        dist2 = ((positions - positions[idx]) ** 2).sum(-1)
        suppress = exp(-dist2 / (2 * radius_m * radius_m))
        w = w * (1 - suppress)
    return tensor(chosen)
```

Build local anchor distributions:

```python
for idx in mode_indices:
    dist2 = ((positions - positions[idx]) ** 2).sum(-1)
    local_kernel = exp(-dist2 / (2 * local_sigma_m * local_sigma_m))
    role_prior = choose(task_point_prior, effector_point_prior, interaction_point_prior)
    anchor_w = local_kernel * role_prior
    anchor_w = anchor_w / anchor_w.sum().clamp_min(eps)

    anchor_x = anchor_w @ point_positions
    anchor_S = weighted_cov(point_positions, anchor_w, anchor_x)
    anchor_token = vl_anchor_token_proj(pool_features(anchor_w))
```

Role allocation should be explicit:

```text
effector/contact modes:
  from effector_point_prior and proprio/contact evidence

task/object modes:
  from task_point_prior

interaction/keypose modes:
  from interaction_point_prior

coverage modes:
  residual FPS over scene/global points
```

Default for 16 observation anchors:

```text
1-2 effector/contact slots
4 task/object slots
4 interaction/keypose slots
4 posterior-predicted/object-memory slots
2-3 residual coverage slots
```

This is a budget rule, not a hard law. The key invariant is that gripper-local
support cannot consume every scene/object slot.

## 11. Pseudocode: Observation Anchor Integration

Planned signature:

```python
def _build_observation_anchors(
    self,
    token_field: PicfTokenFieldState,
    dense_memory: _StepDenseMemory | None = None,
    vl_grounding: PicfVLGroundingState | None = None,
) -> PicfObservationAnchorState:
```

Seed policy:

```text
role 0 effector slots:
  keep current local/effector FPS and tactile/proprio behavior

role 1 scene/object slots:
  use VL task/interaction proposal modes first
  fill remaining scene slots with global FPS coverage
```

Attention bias:

```python
public_role_bias = self._fused_read_role_bias(role_ids, token_field)

if vl_grounding is valid:
    bias = zeros_like(public_role_bias)
    gate = sigmoid(self.vl_obs_anchor_gate_logit)
    point_slice = slice(0, point_count)
    scene_slot_slice = slice(effector_count, effector_count + Kvl)

    bias[scene_slot_slice, point_slice] = gate * log(anchor_point_priors + eps)
    public_role_bias = public_role_bias + bias
```

This only biases attention logits. The observation anchor still rereads point,
visual, and tactile evidence and still computes `x/S/a` from its final routing.

## 12. Pseudocode: Task Readout Integration

Planned signature:

```python
def _build_task_readout(
    self,
    token_field: PicfTokenFieldState,
    dense_memory: _StepDenseMemory,
    semantic: _SemanticContext,
    proprio_token: Tensor,
    vl_grounding: PicfVLGroundingState | None = None,
) -> PicfTaskReadoutState:
```

Current direct point attention is kept:

```python
direct = point_public_attention[:local_count]
direct = direct / direct.sum(dim=-1, keepdim=True).clamp_min(eps)
```

VL prior fusion:

```python
if vl_grounding is valid:
    vl = zeros_like(direct)
    k = min(local_count, vl_grounding.anchor_point_priors.shape[0])
    vl[:k] = vl_grounding.anchor_point_priors[:k]

    if local_count > k:
        vl[k:] = vl_grounding.task_point_prior[None, :]

    g = sigmoid(self.vl_task_point_gate_logit)
    point_weights = (1 - g) * direct + g * vl
    point_weights = point_weights / point_weights.sum(dim=-1, keepdim=True).clamp_min(eps)
else:
    point_weights = direct
```

Then keep the existing geometry:

```python
x = point_weights @ token_field.point_positions
S = weighted_cov(token_field.point_positions, point_weights, x)
a = extent_from_cov(S)
local_tokens = local_tokens + task_geom_proj(geometry_pe(x, a, S))
```

This preserves the PICF task-readout architecture while adding a language-aware
3D prior.

## 13. Pseudocode: Posterior Binding Integration

Posterior binding must be staged.

### 13.1 Stage 0/1

No posterior binding change.

```text
VL router affects task readout and observation-anchor seeding/bias only.
posterior_update remains hidden/geometric/role based.
```

This preserves the v2.2 invariant that semantic does not directly enter
posterior state.

### 13.2 Stage 2 Optional Semantic Binding Bias

Only after observation/task prior works:

```python
semantic_score = cosine(
    posterior_semantic_key[:, None, :],
    observation_anchor_semantic_token[None, :, :],
)

g = sigmoid(self.vl_posterior_bind_gate_logit)
bind_logits = bind_logits + g * semantic_score
```

Rules:

```text
binding bias is allowed
direct posterior h/c/mu/S/x/a overwrite is forbidden
semantic_key updates by EMA only under confidence/support gates
```

EMA update:

```python
if support_mass[k] > support_threshold and vl_confidence > conf_threshold:
    semantic_key[k] = ema(semantic_key[k], obs_semantic_token)
```

This makes persistent anchors easier to rebind to the same semantic object while
preserving physical evidence as the authority.

## 14. Loss Design

Losses must be explicit and independently logged.

### 14.1 Heatmap Cross Entropy

```python
def heatmap_ce(logits: Tensor, target: Tensor) -> Tensor:
    target = target.clamp_min(0)
    target = target / target.sum().clamp_min(eps)
    logp = log_softmax(logits.flatten(), dim=0)
    return -(target.flatten() * logp).sum()
```

Targets:

```text
effector heatmap:
  current end-effector projection

interaction heatmap:
  future keypose / gripper open-close event projection

task heatmap:
  object mask/bbox/label if available
  otherwise optional keypose-contact proxy
```

Do not pretend a keypose proxy is an object segmentation label. Log it as
`proxy_task_heatmap`.

### 14.2 Point Prior Consistency

Small confidence-gated term:

```python
loss_vl_point = JS(
    stopgrad(vl_anchor_point_prior),
    direct_task_point_attention
)
```

Use only when:

```text
projection valid
visible mass sufficient
heatmap entropy not degenerate
task point attention has nonzero support
```

### 14.3 Anchor Diversity

Scene/object anchors should not collapse:

```python
pairwise = cdist(scene_anchor_x, scene_anchor_x)
loss_div = exp(-pairwise / radius).mean_offdiag()
```

Apply to scene/object/task anchors, not to all effector anchors.

### 14.4 Existing Loss Compatibility

Keep existing losses:

```text
L_action
L_alignment
L_physical_aux
L_semantic_future_aux
L_tactile_aux
```

Add router losses as:

```text
L_total =
  existing_total
  + lambda_vl_heatmap_task * L_hm_task
  + lambda_vl_heatmap_effector * L_hm_eff
  + lambda_vl_heatmap_interaction * L_hm_int
  + lambda_vl_point_consistency * L_point
  + lambda_vl_anchor_diversity * L_div
```

All new weights default to zero until the router is explicitly enabled.

## 15. Training Stages

### Stage 0: No-op Integration

Goal:

```text
Add carriers, gates, and debug fields without changing behavior.
```

Expected tests:

```text
router disabled -> old outputs match
gates set to zero -> old outputs match
invalid projection -> no NaN and no top-left artifact
```

### Stage 1: Heatmap Supervision

Goal:

```text
Train PaliGemma image-token heatmap head.
PaliGemma backbone can remain frozen.
```

Trainable:

```text
vl_heatmap_head
vl_anchor_token_proj
optional small adapters
```

Frozen:

```text
PaliGemma backbone
V-JEPA/Sonata/AnyTouch if using frozen-perception profile
```

### Stage 2: PICF Router Gates

Goal:

```text
Let PICF task readout and observation anchors learn to use VL priors.
```

Enable:

```text
task point gated fusion
observation anchor gated bias
router-specific debug
```

Keep posterior semantic binding off unless Stage 1/2 diagnostics are stable.

### Stage 3: Optional Posterior Semantic Binding

Goal:

```text
Help persistent anchors rebind to semantic objects over time.
```

Enable only as low-gate binding bias, never as direct posterior overwrite.

### Stage 4: Optional PaliGemma LoRA

Only if there is enough data and Stage 1/2 show useful heatmaps:

```text
LoRA on selected PaliGemma layers
heatmap head trainable
router trainable
```

Do not jump to full PaliGemma fine-tuning by default.

## 16. Diagnostics

The diagnostics must show raw failure modes. Do not hide collapse by dropping
points below a threshold.

Required JSON fields:

```text
vl.task_heatmap_entropy
vl.effector_heatmap_entropy
vl.interaction_heatmap_entropy

vl.task_point_prior_entropy
vl.task_point_prior_max
vl.task_point_prior_visible_mass
vl.task_point_prior_argmax_pixel
vl.task_point_prior_argmax_xyz

vl.anchor_scores
vl.anchor_roles
vl.anchor_x
vl.anchor_cov_eigvals
vl.anchor_pairwise_distance_min
vl.anchor_pairwise_distance_mean

task.direct_point_entropy
task.fused_point_entropy
task.vl_gate
task.direct_x
task.fused_x

projection.valid
projection.visible_point_count
projection.projective_candidate_density
projection.invalid_reason
```

Required videos/images:

```text
1. raw colored points only
2. heatmap overlay only
3. covariance ellipses only
4. combined diagnostic view
```

For covariance ellipses:

```text
draw eigenvalue-clipped ellipses
log unclipped eigenvalues in JSON
do not mark every point with text labels
```

For top-k:

```text
show top-k pixels/points
show entropy and visible mass
do not use top-left default if invalid
```

## 17. Failure Modes And Required Guards

### 17.1 Top-left Artifact

Symptom:

```text
all colors cluster near image top-left or a constant meaningless pixel
```

Likely causes:

```text
invalid projection silently converted to zero pixel
heatmap grid order mismatch
visual_pixel_centers mismatch
all prior mass zero then normalized incorrectly
```

Required guard:

```text
if visible_mass <= eps:
  valid = false
  skip VL bias
  log invalid_reason
```

### 17.2 Gripper Collapse

Symptom:

```text
all task/object anchors remain on gripper until gripper touches target
```

Likely causes:

```text
effector supervision dominates task/object slots
role budgets are not separated
task heatmap is actually an effector heatmap
interaction heatmap lacks keypose/event target
```

Required guard:

```text
separate effector slots from scene/object slots
never use effector prior to fill all scene slots
log per-role anchor distributions
```

### 17.3 Uniform Heatmap

Symptom:

```text
heatmap entropy close to uniform
point prior follows geometry density rather than task
```

Likely causes:

```text
heatmap head untrained
text conditioning missing
state/prompt mismatch
supervision unavailable
```

Required guard:

```text
low confidence -> small or zero gate contribution
do not force posterior binding from uniform heatmap
```

### 17.4 Hard Semantic Takeover

Symptom:

```text
PICF anchors stop using point/tactile evidence and follow PaliGemma priors only
```

Likely causes:

```text
gate initialized too high
loss weights too high
direct posterior overwrite
```

Required guard:

```text
near-zero gate init
explicit gate metrics
no direct posterior overwrite
router disabled equivalence test
```

### 17.5 Multi-view Distribution Shift

Symptom:

```text
adding orthographic/rendered views hurts PI0.5 action generation
```

Cause:

```text
new view tokens changed PaliGemma action prefix distribution
```

Required guard:

```text
MVP uses current static view for grounding.
If orthographic views are added, use a separate grounding branch and do not
alter the PI0.5 action prefix unless explicitly testing that ablation.
```

## 18. Implementation Order

Recommended order:

```text
M0. Documentation and tests only.
M1. Expose PaliGemma image/text tokens and view ranges.
M2. Add PicfVLGroundingState and no-op router config.
M3. Add heatmap head and _build_vl_grounding.
M4. Enable task-readout gated point fusion.
M5. Enable observation-anchor soft seeded proposals and gated point bias.
M6. Enable posterior-binding soft overlap bias.
M7. Add diagnostic export for heatmaps, point priors, top-k, ellipses.
M8. Add heatmap supervision losses with zero default weights.
```

Do not enable router training by default before M1-M6 can prove that:

```text
heatmap shape is correct
projection is valid
invalid projection is visible in JSON
point prior is not silently top-left
router disabled equivalence holds
```

## 19. Test Plan

### 19.1 Unit Tests

Current implemented tests:

```text
test_paligemma_view_transform_records_resize_with_pad_metadata
test_vl_grounding_disabled_does_not_instantiate_router_modules
test_vl_heatmap_resize_preserves_probability_mass
test_vl_heatmap_mapping_uses_resize_with_pad_transform
test_semantic_context_carries_paligemma_view_transforms
test_vl_point_prior_uses_column_normalized_projective_mass
test_vl_point_prior_projectable_mask_excludes_local_frame_rows
test_vl_point_prior_invalid_projection_is_zero_not_top_left_fallback
test_scene_point_candidate_mask_rejects_projective_border_points
test_vl_slot_point_priors_are_role_aware
test_weighted_anchor_modes_preserve_separated_high_weight_modes
test_vl_grounding_enabled_builds_state_without_changing_default_anchor_contract
test_effector_and_scene_anchor_roles_use_separate_point_pools
```

Required before enabling default-on routing:

```text
test_paligemma_features_return_image_token_ranges
test_vl_router_disabled_exact_equivalence
test_vl_zero_gate_exact_equivalence
test_vl_observation_anchor_role_budget_preserved
test_vl_task_point_weights_normalized
test_vl_posterior_binding_bias_is_soft_and_shape_safe
```

### 19.2 Synthetic Geometry Tests

Use a toy point cloud and visual grid:

```text
single visible point projects to heatmap maximum -> prior argmax is that point
two heatmap modes -> weighted NMS returns two separated anchors
all points invisible -> router valid=false and no bias applied
depth inconsistent point -> lower prior than depth-consistent point
```

### 19.3 Regression Tests

With router disabled:

```text
pipeline smoke tests must produce old tensor shapes
PICF formal contract checker must pass
CALVIN serving path must not require new fields
PI0.5-only ablation must not instantiate router losses
```

### 19.4 Cloud Diagnostic Smoke

Minimum real-data diagnostic:

```text
run 1 CALVIN sequence
save JSONL and videos under one step-scoped /mnt folder
verify:
  anchor_debug JSON has heatmap entropy
  prediction_debug still works
  videos are non-placeholder files
  no all-zero or top-left task prior unless invalid_reason says why
```

## 20. Why This Is Not A Patchy Fix

This design is not "if bad point, hide it" and not "replace anchors with
PaliGemma." It is a consistent factorization:

```text
PaliGemma:
  language-conditioned 2D semantic grounding

Projective geometry:
  physically valid lift from 2D patches to 3D points

PICF observation anchors:
  role-aware current evidence collection

PICF posterior anchors:
  recurrent physical memory and optional semantic binding bias

PICF task readout:
  action-side task-conditioned geometry for PI0.5 prefix tokens

PI0.5 action path:
  final flow-matching action generation
```

The model remains PICF. The router only supplies task-conditioned priors and
supervision that are currently missing from the point/anchor selection path.

## 21. Non-goals

The first implementation should not:

```text
inject raw point cloud tokens into PaliGemma
replace PI0.5 action generation
overwrite posterior x/S/a from a heatmap
make CALVIN evaluation depend on a missing segmentation model
use threshold-only visualization as a correctness fix
change the default training behavior when router is disabled
```

## 22. Open Decisions Before Coding

These should be resolved before the implementation PR:

```text
1. Whether MVP uses static view only or static+gripper view.
2. Whether object/task heatmap supervision exists beyond keypose proxy.
3. Whether posterior semantic binding is postponed until after taskpoint fusion.
4. Whether orthographic BridgeVLA-style rendered views are a separate branch.
5. Which CALVIN robot pose field is the authoritative keypose projection source.
6. Which diagnostic folder schema is canonical for heatmap/top-k/ellipse outputs.
```

Recommended defaults:

```text
MVP grounding view:
  static only

initial posterior semantic binding:
  off

initial trainable modules:
  heatmap head, router gates, task/anchor adapters

initial PaliGemma backbone:
  frozen

initial router losses:
  effector and interaction heatmaps first
  task/object heatmap only when target labels or reliable proxy exists
```

## 23. Final Contract

The final mathematical contract should be:

```text
Given:
  P points with positions X in R^{P x 3}
  V visual grid cells
  C in [0,1]^{P x V} from projective geometry
  PaliGemma image tokens Z in R^{V_pg x D}
  text-conditioned summary s

The router computes:
  H_task, H_eff, H_int = HeatmapHead(Z, s)
  h_task, h_eff, h_int in Delta^{V}
  w_task = normalize(C h_task)
  w_eff  = normalize(C h_eff)
  w_int  = normalize(C h_int)
  A = MultiModeAnchorProposal(w_task, w_eff, w_int, X)

PICF consumes:
  observation-anchor seed/bias from A
  task-readout point-weight bias from A
  optional posterior-binding semantic bias from A

PICF preserves:
  recurrent posterior state update
  tactile and point evidence authority
  final PI0.5 flow-matching action path
```

If `vl_anchor_router_enabled=False`, the above reduces exactly to current PICF
v2.2 behavior.
