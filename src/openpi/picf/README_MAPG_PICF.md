# MAPG-PICF: Modality-Optional Anchor Prior Graph

Date: 2026-05-06
Repo: `/home/siyuanyue/Documents/openpi`
Status: **live implementation record for the MAPG-enabled PICF path**

This README is the implementation and math contract for the full MAPG-PICF
deployment. It supersedes the earlier design-only MAPG wording. The
point-centric VL router remains available as a lower-level compatibility
substrate, but the complete graph-level path is now represented by
`PicfAnchorPriorGraphState` and is connected to:

- PaliGemma spatial grounding tokens / heatmaps.
- V-JEPA visual support priors.
- Sonata / point-token support priors when pointcloud is valid.
- AnyTouch / tactile support priors when tactile evidence is valid.
- posterior temporal support priors when previous posterior anchors exist.
- observation anchors.
- task readout.
- posterior binding.
- conditioned-control / PI0.5 action prefix.
- MAPG graph self-supervision losses.

The runtime switch is:

```bash
--mapg-enabled
```

The graph is intentionally still gated and masked. “Full MAPG” here means the
complete dataflow and losses exist and are wired into the live model; it does
not mean every noisy or missing modality is forced to contribute on every
sample. Missing/invalid modalities are explicit no-ops.

## 1. Design Contract

The old point-centric route was:

```text
PaliGemma heatmap -> point prior -> anchors
```

MAPG changes the center of the architecture:

```text
PaliGemma / V-JEPA / Sonata / AnyTouch / posterior
-> modality-native seed priors
-> finite-round shared anchor prior graph
-> PICF observation anchors / task readout / posterior binding
-> PI0.5 action prefix
```

Each anchor is a latent support, not just a 3D point:

```text
anchor k =
  visual support over V-JEPA tokens
+ optional PaliGemma image-token support
+ optional point/Sonata support
+ optional tactile/AnyTouch support
+ optional posterior temporal support
+ role, confidence, and geometry metadata
```

This is the required architecture for datasets where:

- RGB is available but pointcloud may be missing.
- pointcloud quality varies or is noisy.
- tactile evidence is sparse and should be a probabilistic contact volume.
- posterior anchors carry temporal identity.
- similar objects must remain separate anchor instances.

## 2. Live Code Surface

### 2.1 Config

`src/openpi/picf/core/config.py` defines:

```python
mapg_enabled: bool = False
mapg_anchor_count: int = 8
mapg_message_rounds: int = 1
mapg_visual_sigma_patches: float = 2.0
mapg_tactile_sigma_m: float = 0.08
mapg_posterior_sigma_m: float = 0.08
mapg_confidence_floor: float = 0.05
mapg_assignment_sinkhorn_iters: int = 6
mapg_assignment_temperature: float = 1.0
mapg_obs_gate_init: float = -4.0
mapg_task_gate_init: float = -4.0
mapg_posterior_gate_init: float = -6.0
mapg_control_gate_init: float = -4.0
mapg_prior_bias_clip: float = 4.0
```

The default is off for backward compatibility. A MAPG run must explicitly pass
`--mapg-enabled`.

### 2.2 State Object

`src/openpi/picf/core/contracts.py` defines:

```python
PicfAnchorPriorGraphState(
    pg_priors,          # [K, V_pg] or None
    visual_priors,      # [K, V]
    point_priors,       # [K, P] or None
    tactile_priors,     # [K, T] or None
    posterior_priors,   # [K, A] or None
    anchor_tokens,      # [K, H]
    anchor_roles,       # [K]
    anchor_scores,      # [K]
    anchor_confidence,  # [K]
    anchor_x,           # [K, 3] or None
    anchor_S,           # [K, 3, 3] or None
    geometry_valid,     # [K]
    obs_slot_assignment,# [N_obs, K] or None
    task_assignment,    # [N_task, K] or None
    modality_confidence,# [K, 5]
    valid,              # scalar bool tensor
)
```

`PicfCoreState` now contains:

```python
anchor_prior_graph: PicfAnchorPriorGraphState | None
```

Observation anchors, task readout, and conditioned control expose their graph
assignments / graph tokens so tests and diagnostics can verify live
consumption.

### 2.3 PaliGemma Spatial Metadata

`src/openpi/picf/paligemma/wrapper.py` exposes:

```python
image_tokens
text_tokens
image_token_ranges
image_grid_shapes
image_view_names
image_view_transforms
```

`PaliGemmaViewTransform` records the exact `resize_with_pad` mapping:

```python
original_hw
target_hw
resized_hw
pad_top / pad_bottom / pad_left / pad_right
scale_y / scale_x
```

All PaliGemma-grid heatmaps must pass through this transform before they are
mapped to the V-JEPA visual grid or point projection support. A naive resize is
not part of the contract.

## 3. Mathematical Contract

Let modality support spaces be:

```text
pg   PaliGemma image-token grid
v    V-JEPA visual grid
p    point/Sonata token rows
t    tactile/AnyTouch active token rows
post posterior anchor rows
```

For anchor `k`, MAPG maintains distributions:

```text
p_pg^k   in Delta(V_pg)
p_v^k    in Delta(V)
p_p^k    in Delta(P)
p_t^k    in Delta(T)
p_post^k in Delta(A)
```

Not every distribution exists on every step. Missing pointcloud, no active
tactile contact, invalid projection, or missing posterior history are explicit
`None` / zero-confidence paths.

### 3.1 No Implicit Fixed Point

MAPG does **not** define:

```text
p_m^k = normalize(sum_s alpha_s->m T_s->m p_s^k)
```

as an implicit cyclic equation. The live implementation uses seed priors plus
fixed finite message passing:

```text
q_m^k = source-specific seed prior
p_m^{k,0} = normalize(q_m^k)

for r = 1..R:
  p_v^{k,r} =
    normalize(alpha_v q_v^k
      + alpha_p T_p->v p_p^{k,r-1}
      + alpha_t T_t->v p_t^{k,r-1}
      + alpha_post T_post->v p_post^{k,r-1})

  p_p^{k,r} =
    normalize(alpha_p q_p^k
      + alpha_v T_v->p p_v^{k,r-1}
      + alpha_t T_t->p p_t^{k,r-1}
      + alpha_post T_post->p p_post^{k,r-1})

  p_t^{k,r}    = normalize(q_t^k)
  p_post^{k,r} = normalize(q_post^k)

p_m^k = p_m^{k,R}
```

`R` is `mapg_message_rounds`. The deployment supports `R >= 1`; current test
coverage includes `R=2`.

The `alpha` weights are confidence gates derived from valid mass and normalized
entropy of each support distribution, floored by `mapg_confidence_floor` for
valid but intentionally broad coverage priors. Anchor token fusion uses the
same modality-confidence vector as its `beta` weights instead of equal averaging
across present modalities.

### 3.2 Compatibility Operator Direction

Every operator has shape:

```text
T_s->m: [|Omega_m|, |Omega_s|]
```

and is column-normalized over the target support when valid:

```text
sum_i T_s->m[i, j] = 1
```

So:

```text
p_m = T_s->m p_s
```

The code implements this orientation with row-major tensors. For example:

```python
# visual_priors: [K, V]
# compat_col: [P, V]
point_priors = visual_priors @ compat_col.T  # [K, P]
```

Direct tactile/posterior-to-visual operators are also live. They use static
camera rays from `PicfProjectiveGeometryState.visual_ray_world` and
`camera_origin_world` to form a world-position-to-visual Gaussian support. This
keeps tactile/posterior visual evidence alive even when point support is weak or
masked.

PaliGemma-to-visual grounding does not require point support. When projective
point compatibility is missing, MAPG still receives the PaliGemma visual
heatmaps and PaliGemma image-token priors; only the point-centric lifted priors
become explicit no-ops.

### 3.3 Visual-to-Point Projection

Visual support is lifted to point support only when projective geometry is
valid:

```text
C[p, v] >= 0
C_col[p, v] = C[p, v] / clamp(sum_p C[p, v], eps)
p_p^k = normalize(sum_v C_col[p, v] p_v^k[v])
```

Important invariants:

- `point_positions_world` is used for camera projection and geometry moments.
- local/model-frame `point_positions` are not used as camera coordinates.
- `point_projectable_mask` is strict scene/global visibility support.
- strict masks are used for lift; fallback global masks are coverage-only.
- invalid visible mass produces a no-op, not a fake uniform point prior.

### 3.4 Roles

Anchor roles are:

```text
0 effector/contact
1 task/object
2 interaction/affordance
3 coverage/context
```

Role-constrained assignment maps anchors into observation and task slots:

```text
A_obs  in R^{N_obs x K}
A_task in R^{N_task x K}
```

Effector slots are not allowed to be overwritten by static-camera VL global
seeds. Scene/task/interaction slots receive scene-compatible graph priors.

### 3.5 Posterior Safety

MAPG may bias posterior binding:

```text
binding_logits += gate * graph_binding_bias
```

MAPG never directly overwrites:

```text
posterior x/S/a/h/c/mu/Sigma
```

Posterior remains the physical memory; MAPG supplies a temporal/semantic
binding prior.

## 4. Live Dataflow

`PicfFullCore.observe_step(...)` executes:

```text
RGB/depth/tactile/proprio/instruction
-> runtime meta
-> point feature extraction when valid
-> V-JEPA visual map
-> AnyTouch tactile bundle
-> PaliGemma semantic/spatial context
-> PicfTokenFieldState
-> PicfVLGroundingState
-> PicfAnchorPriorGraphState
-> observation anchors consume graph
-> posterior update consumes graph binding bias
-> task readout consumes graph supports
-> conditioned control consumes graph tokens
-> PI0.5 flow-matching action path
```

### 4.1 Observation Anchors

Observation anchors use MAPG through:

```text
query blend:
  q_obs += gate * A_obs graph_tokens

point attention bias:
  logits_point += gate * centered_log(A_obs point_priors)

visual attention bias:
  logits_visual += gate * centered_log(A_obs visual_priors)
```

The point-centric VL router can still contribute point priors when enabled, but
MAPG supplies the graph-level multimodal prior.

### 4.2 Task Readout

Task readout uses MAPG through:

```text
local task tokens += gate * A_task graph_tokens
visual weights = normalize((1-g) direct_visual + g * A_task visual_priors)
local task tokens += gate * visual_pool_proj(pool(V-JEPA, visual weights))
point weights = normalize((1-g) direct + g * A_task point_priors)
graph_visual_weights = A_task visual_priors
graph_tactile_weights = A_task tactile_priors
```

If point support is unavailable, the graph still retains visual support and
graph tokens. Geometry moments are only valid when point support exists.
`PicfTaskReadoutState.geometry_valid` records this explicitly.

### 4.3 Conditioned Control and PI0.5

Conditioned control receives graph tokens:

```text
graph_control_tokens = gate * mapg_to_control_proj(anchor_tokens)
control_prefix = [posterior, global_post, innovation, proprio, task, graph, queries]
control_world(control_prefix)
pi_prefix_reader(control_tokens)
PI0.5 action flow matching
```

The PI0.5 action generator remains the final action path. MAPG adds action-side
context; it does not replace PI0.5.

## 5. Training Objective

The total loss remains:

```text
L = L_action + L_existing_PICF + L_VL_router + L_MAPG
```

MAPG terms are implemented in `src/openpi/picf/core/training.py`:

```text
loss_mapg_graph
loss_mapg_siglip
loss_mapg_vicreg
loss_mapg_cycle
loss_mapg_masked_modality
loss_mapg_routing
```

CLI knobs:

```bash
--lambda-mapg-siglip
--lambda-mapg-vicreg
--lambda-mapg-cycle
--lambda-mapg-masked-modality
--lambda-mapg-routing
--mapg-siglip-tau
--mapg-vicreg-var-target
--mapg-vicreg-cov-weight
```

All MAPG losses are availability/confidence-gated by the presence of graph
supports:

- no point support -> no point cycle/alignment contribution.
- no active tactile support -> no tactile positive pair.
- invalid projection -> no visual-point cycle.
- no previous posterior -> no posterior pair.
- invalid per-anchor modality rows -> excluded from SigLIP/VICReg/masked terms.
- routing consistency compares MAPG assignment priors with live observation and
  task visual/point distributions when their shapes and masks are valid.

This is not a reduced implementation. It is the mathematically correct
masked-objective behavior for modality-optional data.

### 5.1 SigLIP-Style Anchor Matching

Same-anchor cross-modal embeddings are positives:

```text
(graph_k, visual_k)
(graph_k, point_k)
(graph_k, tactile_k)
(graph_k, posterior_k)
```

Different anchors in the same frame are negatives even when they are visually
or semantically similar. This prevents instance merging.

### 5.2 Assignment And Anti-Collapse

`_mapg_slot_assignment(...)` uses role-constrained Sinkhorn-style assignment:

```text
slot roles -> role mask over anchors
log(anchor_score * anchor_confidence) / temperature
row normalization
column coverage normalization for mapg_assignment_sinkhorn_iters rounds
```

This prevents all observation/task slots from collapsing onto the single
highest-score anchor while preserving role compatibility.

### 5.2 VICReg Anti-Collapse

`loss_mapg_vicreg` keeps graph and modality embedding dimensions active and
decorrelated. It is applied to every available valid-row embedding family:
graph, visual, point, tactile, and posterior.

### 5.3 Cycle Consistency

When visual-point projection exists:

```text
p_v -> T_v->p -> p_p -> T_p->v -> p_v_cycle
JS(p_v, p_v_cycle)
```

This catches transpose mistakes, mass leakage, and projection collapse.

### 5.4 Masked Modality Prediction

Available modality embeddings predict the held-out modality through a
leave-one-modality-out cosine objective. The target modality and fused graph
embedding are excluded from the predictor side, so the target cannot leak
through its own pooled representation:

```text
{point,tactile,posterior} -> visual
{visual,tactile,posterior} -> point
{visual,point,posterior} -> tactile
{visual,point,tactile} -> posterior
```

Each path is masked when the target modality is unavailable.

### 5.5 Routing Regularization

Observation and task assignment matrices are regularized to avoid both
over-uniform routing and single-anchor collapse:

```text
entropy(A_slot)
+ JS(anchor_coverage, uniform_anchor_coverage)
+ JS(A_obs visual_priors, observation visual routing)
+ JS(A_task visual_priors, task visual weights)
+ JS(A_task point_priors, task point weights)
```

## 6. CLI

Core MAPG switches:

```bash
--mapg-enabled
--mapg-anchor-count 8
--mapg-message-rounds 2
--mapg-visual-sigma-patches 2.0
--mapg-tactile-sigma-m 0.08
--mapg-posterior-sigma-m 0.08
--mapg-confidence-floor 0.05
--mapg-assignment-sinkhorn-iters 6
--mapg-assignment-temperature 1.0
--mapg-obs-gate-init -4.0
--mapg-task-gate-init -4.0
--mapg-posterior-gate-init -6.0
--mapg-control-gate-init -4.0
--mapg-prior-bias-clip 4.0
```

Graph losses:

```bash
--lambda-mapg-siglip 0.005
--lambda-mapg-vicreg 0.001
--lambda-mapg-cycle 0.002
--lambda-mapg-masked-modality 0.002
--lambda-mapg-routing 0.001
```

VL heatmap/keypose supervision remains live and should be used with MAPG:

```bash
--vl-anchor-router-enabled
--lambda-vl-heatmap-effector 0.01
--lambda-vl-heatmap-interaction 0.01
--lambda-vl-point-consistency 0.002
--lambda-vl-anchor-diversity 0.001
```

`--lambda-vl-heatmap-task` should remain `0.0` unless real object/bbox/mask
grounding labels exist. A keypose proxy is not an object segmentation label.

## 7. Canonical MAPG 2x40GB Launch

Use this profile for the current cloud MAPG long run:

```bash
RUN=picf_v22_mapg_semtrain_tactenc_strict2x40_unroll2_30000_ckpt5000_YYYYMMDD_rN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/path/to/openpi:${PYTHONPATH}

torchrun --standalone --nproc_per_node=2 scripts/picf_core_train.py \
  --calvin-root /mnt/calvin/task_ABC_D \
  --output-dir /mnt/checkpoints/picf_core/picf_core \
  --exp-name ${RUN} \
  --num-train-steps 30000 \
  --save-interval 5000 \
  --log-interval 100 \
  --diagnostic-interval 500 \
  --training-strategy fsdp_full_shard \
  --optimizer-sharding none \
  --optimizer-checkpoint-mode auto \
  --accum-steps 1 \
  --unroll-steps 2 \
  --action-horizon 16 \
  --use-foundation-backbones \
  --perception-finetune-mode frozen \
  --point-backbone sonata \
  --visual-mode encoder \
  --visual-feature-mode hierarchical \
  --tactile-mode encoder \
  --semantic-mode paligemma \
  --semantic-trainable \
  --semantic-gradient-checkpointing \
  --semantic-use-gripper \
  --vl-anchor-router-enabled \
  --lambda-vl-heatmap-task 0.0 \
  --lambda-vl-heatmap-effector 0.01 \
  --lambda-vl-heatmap-interaction 0.01 \
  --lambda-vl-point-consistency 0.002 \
  --lambda-vl-anchor-diversity 0.001 \
  --mapg-enabled \
  --mapg-anchor-count 8 \
  --mapg-message-rounds 2 \
  --lambda-mapg-siglip 0.005 \
  --lambda-mapg-vicreg 0.001 \
  --lambda-mapg-cycle 0.002 \
  --lambda-mapg-masked-modality 0.002 \
  --lambda-mapg-routing 0.001
```

Expected startup log contracts:

```text
Training config: world_size=2 ... num_steps=30000 save_interval=5000 unroll_steps=2
VL-guided anchor router contract: enabled=True ...
MAPG anchor prior graph contract: enabled=True anchors=8 message_rounds=2 ... confidence_floor=0.05 assignment_sinkhorn_iters=6 assignment_temperature=1.0 ...
Backbone contract: point=sonata(trainable=False) visual=encoder(... trainable=False) tactile=encoder(trainable=False) semantic=paligemma(trainable=True)
```

Frozen perception means the pretrained Sonata, V-JEPA, and AnyTouch encoder
weights are frozen. PICF adapters, graph projections, graph gates, action-side
heads, posterior/task/control/world modules, VL heads, and trainable PaliGemma
semantic path remain trainable according to the CLI flags.

## 8. Diagnostics

`PicfCoreOutput.debug` emits:

```text
mapg_valid
mapg_anchor_count
mapg_visual_support_mean
mapg_point_available
mapg_tactile_available
mapg_posterior_available
```

Training logs emit:

```text
loss_mapg_graph
loss_mapg_siglip
loss_mapg_vicreg
loss_mapg_cycle
loss_mapg_masked_modality
loss_mapg_routing
```

Healthy first checks:

- `mapg_valid == 1.0` when visual tokens exist.
- `mapg_visual_support_mean` near `1.0` because each visual support row is a
  normalized distribution.
- `mapg_point_available == 1.0` when projective point support exists.
- `mapg_tactile_available == 1.0` only when active tactile tokens exist.
- `mapg_posterior_available == 1.0` after the first recurrent step.
- no NaN/Inf in MAPG losses.
- observation/task graph assignments have correct shape.
- PI0.5 flow loss still logs through the standard action terms.

## 9. Verification Commands

Compile:

```bash
uv run python -m py_compile \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/training.py \
  scripts/picf_core_train.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/config.py
```

Core tests:

```bash
uv run pytest -q \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  scripts/picf_core_train_test.py
```

Targeted MAPG tests:

```bash
uv run pytest -q \
  src/openpi/picf/core/pipeline_test.py::test_mapg_enabled_builds_full_anchor_graph_and_live_consumers \
  src/openpi/picf/core/pipeline_test.py::test_mapg_builds_paligemma_grounding_without_point_router \
  src/openpi/picf/core/training_test.py::test_transition_loss_can_enable_mapg_graph_terms \
  scripts/picf_core_train_test.py::test_build_model_and_loss_config_propagate_mapg_knobs
```

Contract / diff hygiene:

```bash
uv run python scripts/verify_picf_contract.py
git diff --check
```

The current local MAPG patch was verified with:

```text
203 passed, 26 warnings
```

The warnings are existing PyTorch/FSDP/deprecation warnings in the test
environment, not MAPG contract failures.

## 10. Implementation Files

Main code:

- `src/openpi/picf/core/contracts.py`
- `src/openpi/picf/core/config.py`
- `src/openpi/picf/core/pipeline.py`
- `src/openpi/picf/core/training.py`
- `scripts/picf_core_train.py`
- `src/openpi/picf/paligemma/wrapper.py`

Tests:

- `src/openpi/picf/core/pipeline_test.py`
- `src/openpi/picf/core/training_test.py`
- `scripts/picf_core_train_test.py`

Docs:

- `src/openpi/picf/README_MAPG_PICF.md`
- `src/openpi/picf/README_v2.2.md`
- `src/openpi/picf/README.md`
- `src/openpi/picf/README_VL_GUIDED_ANCHOR_ROUTER.md`

## 11. Non-Negotiable Invariants

- PaliGemma does not ingest raw point, tactile, or posterior tokens.
- V-JEPA visual support is first-class and exists even when pointcloud is weak.
- point support is optional and must use world-frame projection.
- tactile support is probabilistic and contact-gated.
- posterior support is temporal prior, not hard memory overwrite.
- compatibility matrices use explicit orientation and normalization.
- invalid projection is a no-op.
- role-0 effector anchors keep local/proprio/tactile seed semantics.
- graph priors enter PICF through gated soft bias / token context.
- PI0.5 flow-matching remains the final action training path.

## 12. Relationship To The VL Router

`README_VL_GUIDED_ANCHOR_ROUTER.md` now describes the lower-level
PaliGemma-guided point-centric substrate:

```text
PaliGemma heatmap -> strict projective point prior -> role-aware soft bias
```

MAPG consumes and generalizes that substrate:

```text
PaliGemma heatmap -> visual/PaliGemma seed support
visual/point/tactile/posterior supports -> graph
graph -> observation/task/posterior/control
```

Do not describe the live MAPG path as “only point-centric”. The point-centric
router is one expert branch inside the graph-level architecture.
