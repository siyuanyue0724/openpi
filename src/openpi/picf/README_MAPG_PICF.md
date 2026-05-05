# MAPG-PICF: Modality-Optional Anchor Prior Graph

Date: 2026-05-05
Status: architecture and implementation contract. This document is a detailed
construction plan for the next anchor-routing upgrade. It does **not** claim
that MAPG is fully live in the current code. The current checked-in live
implementation is the default-off point-centric VL-guided router described in
[`README_VL_GUIDED_ANCHOR_ROUTER.md`](./README_VL_GUIDED_ANCHOR_ROUTER.md).

Temporary local audit backing this document:

```text
/tmp/picf_mapg_picf_code_math_audit_20260505.md
```

## 0. Executive Contract

MAPG-PICF upgrades the previous router:

```text
PaliGemma heatmap -> point prior -> anchors
```

into a modality-optional anchor graph:

```text
PaliGemma / V-JEPA / Sonata / AnyTouch / posterior
-> modality-native priors
-> shared anchor prior graph
-> PICF observation anchors / task readout / posterior binding
-> PI0.5 action prefix
```

The core architectural change is:

```text
point cloud is no longer the only anchor center.
V-JEPA visual support becomes a first-class anchor support.
Point, tactile, PaliGemma, and posterior supports are optional experts.
```

This is necessary because the long-term data contract should support:

- RGB-only or weak-pointcloud datasets.
- noisy or missing depth / point cloud.
- dense V-JEPA 2.1 visual substrate as the always-available support.
- tactile contact as a broad probabilistic contact volume, not a single point.
- posterior anchors as temporal physical memory.
- instance-aware cross-modal alignment without merging visually similar objects.

The resulting anchor object is:

```text
anchor k =
  visual support over V-JEPA tokens
+ optional PaliGemma language-conditioned image support
+ optional point/Sonata support
+ optional tactile/AnyTouch contact support
+ optional posterior temporal support
+ role, confidence, and geometry metadata
```

## 1. Current Code Facts

This section records what the current branch already provides and what MAPG
still requires.

### 1.1 PaliGemma Spatial Tokens Already Exist

`src/openpi/picf/paligemma/wrapper.py` currently exposes enough information for
a PaliGemma grounding branch:

```python
PaliGemmaSemanticFeatures.image_tokens
PaliGemmaSemanticFeatures.text_tokens
PaliGemmaSemanticFeatures.image_token_ranges
PaliGemmaSemanticFeatures.image_grid_shapes
PaliGemmaSemanticFeatures.image_view_names
PaliGemmaSemanticFeatures.image_view_transforms
```

`PaliGemmaViewTransform` records `resize_with_pad` geometry:

```python
original_hw
target_hw
resized_hw
pad_top / pad_bottom / pad_left / pad_right
scale_y / scale_x
```

This is a hard requirement. Any PaliGemma heatmap or image-token support must be
mapped through this transform before it is compared with V-JEPA grid cells,
camera pixels, or point projections.

Do not use naive `F.interpolate(pg_heatmap, visual_hw)` as the canonical
mapping. It ignores padding and can create top-left / center artifacts.
The current point-centric VL-router substrate already implements the
transform-aware mapping path; MAPG must preserve that contract when it upgrades
from point priors to graph-level visual-native supports.

### 1.2 Existing VL Router Is Point-Centric

`PicfVLGroundingState` currently represents:

```text
PaliGemma heatmaps
-> point priors through projective geometry
-> point-anchor proposals
```

This is useful, but it is not MAPG. It still treats point support as the final
carrier. MAPG keeps this as one expert path, not the entire graph.

### 1.3 Token Field Already Has Alignment Substrate

`PicfTokenFieldState` currently includes:

```python
point_tokens
visual_tokens
tactile_tokens
fused_tokens
point_align_embeddings
visual_align_embeddings
tactile_align_embeddings
projective_geometry
point_pool_ids
tactile_positions_world
tactile_contact_logits
```

This is the right substrate for a shared anchor graph. The missing fields are:

```python
visual_hw
point_positions_model_frame
point_positions_world
point_projectable_mask
modality_available
```

The point-position split is not optional for MAPG. The current code mixes local
effector point rows and global scene point rows in the point field. Camera
projection must only use world-projectable rows.

### 1.4 V-JEPA Visual Tokens Are Already Separate

`_build_public_read_memory(...)` exposes:

```text
public_read_memory = [fused_tokens, visual_tokens]
```

where `fused_tokens` contain point/tactile/context tokens and `visual_tokens`
remain a separate dense visual memory. Current task readout can attend to
visual memory, but its final task geometry still comes from point attention.

MAPG changes this:

```text
task anchor supports are visual-native first.
point geometry is added when valid.
```

### 1.5 Posterior Binding Is Already Soft-Bias Compatible

`_posterior_update(...)` already uses:

```text
hidden similarity
+ geometry distance
+ role bias
+ Sinkhorn dustbin
+ optional VL soft binding bias
```

This is compatible with MAPG, with one strict rule:

```text
MAPG may bias posterior binding.
MAPG must not directly overwrite posterior x/S/a/h/c/mu/Sigma.
```

## 2. Design Influences

MAPG follows principles from recent VLA and self-supervised representation work:

- BridgeVLA: VLM predicts language-conditioned 2D heatmaps; heatmaps are lifted
  into 3D instead of injecting raw 3D tokens into the VLM backbone.
  <https://arxiv.org/abs/2506.07961>
- FALCON: spatial priors are consumed on the action/control side, preserving
  the pretrained VLM semantic backbone.
  <https://arxiv.org/abs/2510.17439>
- Spatial Forcing: explicit point/depth can be noisy or unavailable; RGB
  representations can be aligned to spatial representations without requiring
  explicit 3D at inference.
  <https://arxiv.org/abs/2510.12276>
- V-JEPA 2.1: dense visual features are a strong spatial-temporal substrate.
  <https://arxiv.org/abs/2603.14482>
- GeoVLA / PointVLA-style systems: 3D point features are valuable action-side
  experts, not something to force through the VLM backbone.
  <https://arxiv.org/abs/2508.09071>
- AnyTouch: tactile representation should use masked modeling and multimodal
  alignment, not brittle one-point correspondence.
  <https://arxiv.org/abs/2502.12191>
- Sonata: point SSL can suffer geometric shortcuts; MAPG must avoid relying on
  coordinate similarity alone.
  <https://arxiv.org/abs/2503.16429>
- VICReg and SigLIP: use mature anti-collapse and pairwise matching losses for
  anchor-level cross-modal alignment.
  <https://arxiv.org/abs/2105.04906>
  <https://arxiv.org/abs/2303.15343>

The implementation must borrow principles, not blindly copy mechanisms.

## 3. Non-Negotiable Invariants

MAPG is only acceptable if these invariants stay true.

```text
1. Default-off exact no-op:
   mapg_anchor_graph_enabled=False must preserve current PICF behavior.

2. PI0.5 action path remains final:
   conditioned_control.pi_prefix_tokens still feed the PI0.5 action path.

3. PaliGemma is not a raw 3D/tactile backbone:
   no raw point, tactile, posterior, or anchor tokens are inserted into
   PaliGemma as normal VLM prefix tokens.

4. V-JEPA visual support is primary:
   every anchor can exist with visual support only.

5. Point support is optional:
   point cloud improves grounding when valid, but missing/noisy point cloud
   must not collapse the graph.

6. Tactile support is probabilistic:
   tactile contact is a broad contact-volume prior, not a hard 3D point.

7. Posterior remains physical memory:
   graph priors bias binding and rereads; they do not hard-write posterior
   state.

8. Role constraints are enforced:
   effector/contact priors cannot consume all task/object slots.

9. Projection validity is explicit:
   invalid or invisible projection returns invalid/zero support, not fake
   top-left or first-point support.

10. Same-anchor positives, different-anchor negatives:
    visually similar objects must not be collapsed into one anchor.
```

## 4. Mathematical Contract

Let the modality set be:

```text
M = {pg, v, p, t, post}
```

where:

```text
pg   = PaliGemma padded image grid
v    = V-JEPA dense visual grid
p    = Sonata / point token support
t    = AnyTouch tactile token support
post = PICF posterior anchor support
```

For anchor `k`, each available modality `m` has a support distribution:

```text
p_m^k in Delta(|support_m|)
```

Compatibility operators map one modality's support to another:

```text
T_pg_to_v      : PaliGemma padded image grid -> V-JEPA grid
T_v_to_p       : V-JEPA grid -> world-projectable point rows
T_p_to_v       : point rows -> V-JEPA grid
T_t_to_v       : tactile contact volume -> V-JEPA grid
T_t_to_p       : tactile contact volume -> point rows
T_post_to_v    : posterior Gaussian/extent -> V-JEPA grid
T_post_to_p    : posterior Gaussian/extent -> point rows
T_post_to_post : posterior identity prior
```

MAPG fuses source priors into each target support:

```text
p_m^k = normalize(sum_s alpha_{s->m}^k * T_{s->m}(p_s^k))
```

`alpha` is confidence-gated and role-masked. Missing modalities set
`alpha=0`.

Each modality produces a pooled embedding:

```text
e_m^k = normalize(MLP_m(sum_i p_m^k[i] * token_m[i]))
```

The graph anchor token is:

```text
z_k = sum_m beta_m^k * e_m^k
    + role_embedding(role_k)
    + confidence_embedding(conf_k)
    + geometry_embedding(x_k, S_k) if geometry_valid_k
```

`beta` is also confidence-gated. If point geometry is invalid, the geometry
embedding is replaced by an invalid-geometry embedding rather than fake `x/S`.

## 5. Proposed State Objects

### 5.1 `PicfAnchorPriorGraphState`

Add this to `src/openpi/picf/core/contracts.py`:

```python
@dataclasses.dataclass
class PicfAnchorPriorGraphState:
    # Modality-native supports. Shapes use K graph anchors.
    pg_priors: torch.Tensor | None        # [K, V_pg]
    visual_priors: torch.Tensor | None    # [K, V]
    point_priors: torch.Tensor | None     # [K, P]
    tactile_priors: torch.Tensor | None   # [K, T]
    posterior_priors: torch.Tensor | None # [K, A]

    # Graph anchor tokens and metadata.
    anchor_tokens: torch.Tensor           # [K, H]
    anchor_roles: torch.Tensor            # [K]
    anchor_scores: torch.Tensor           # [K]
    anchor_confidence: torch.Tensor       # [K]

    # Geometry is optional.
    anchor_x: torch.Tensor | None         # [K, 3]
    anchor_S: torch.Tensor | None         # [K, 3, 3]
    geometry_valid: torch.Tensor          # [K]

    # Slot/query assignment.
    obs_slot_assignment: torch.Tensor | None # [N_obs, K]
    task_assignment: torch.Tensor | None     # [N_task, K]

    # Diagnostics.
    modality_confidence: dict[str, torch.Tensor]
    valid: torch.Tensor                   # [K]
```

Do not overload `PicfVLGroundingState` into this object. Keep the existing VL
router state as the point-centric stage and add MAPG as a graph-level state.

### 5.2 Token Field Extensions

Extend `PicfTokenFieldState`:

```python
visual_hw: tuple[int, int] | None
point_positions_model_frame: torch.Tensor | None # [P, 3]
point_positions_world: torch.Tensor | None       # [P, 3]
point_projectable_mask: torch.Tensor | None      # [P]
modality_available: dict[str, bool]
```

Required invariant:

```text
camera projection uses point_positions_world and point_projectable_mask.
PICF local geometry may still use point_positions_model_frame.
```

### 5.3 Task Readout Extensions

Extend `PicfTaskReadoutState`:

```python
visual_weights: torch.Tensor | None       # [Q, V]
tactile_weights: torch.Tensor | None      # [Q, T]
anchor_assignment: torch.Tensor | None    # [Q, K]
anchor_tokens: torch.Tensor | None        # [Q, H]
geometry_valid: torch.Tensor | None       # [Q]
```

Point-derived `x/S/a` remain valid only when `geometry_valid=True`.

## 6. Build-Time Dataflow

### 6.1 High-Level `observe_step` Order

Current simplified order:

```text
semantic
token_field
observation_anchors
posterior
task_readout
conditioned_control
```

MAPG order:

```text
semantic
token_field
anchor_prior_graph
observation_anchors(anchor_prior_graph)
posterior(observation_anchors, anchor_prior_graph)
task_readout(anchor_prior_graph)
conditioned_control(task_readout, posterior, anchor_prior_graph)
PI0.5 action path
```

Pseudocode:

```python
semantic = self._semantic_context(...)
token_field, dense_memory = self._build_token_field(...)

anchor_graph = self._build_anchor_prior_graph(
    observation=observation,
    semantic=semantic,
    token_field=token_field,
    previous=previous,
)

observation_anchors = self._build_observation_anchors(
    token_field,
    dense_memory,
    anchor_graph=anchor_graph,
)

posterior = self._posterior_update(
    previous,
    observation,
    observation_anchors,
    dense_memory,
    anchor_graph=anchor_graph,
)

task_readout = self._build_task_readout(
    token_field,
    dense_memory,
    semantic,
    proprio_token,
    anchor_graph=anchor_graph,
)

conditioned_control = self._build_conditioned_control_state(
    posterior=posterior,
    task_readout=task_readout,
    anchor_graph=anchor_graph,
)
```

### 6.2 `_build_anchor_prior_graph(...)`

Pseudocode:

```python
def _build_anchor_prior_graph(
    self,
    observation,
    semantic,
    token_field,
    previous,
) -> PicfAnchorPriorGraphState | None:
    if not self.config.mapg_anchor_graph_enabled:
        return None

    pg = self._build_paligemma_priors(semantic)
    visual = self._build_visual_native_priors(pg, semantic, token_field)
    point = self._build_optional_point_priors(visual, token_field)
    tactile = self._build_optional_tactile_priors(observation, token_field)
    post = self._build_optional_posterior_priors(previous, token_field)

    candidates = self._build_graph_candidates(
        pg_priors=pg,
        visual_priors=visual,
        point_priors=point,
        tactile_priors=tactile,
        posterior_priors=post,
    )

    assignments = self._role_constrained_anchor_sinkhorn(candidates)

    anchors = self._pool_anchor_modalities(candidates, assignments, token_field)

    return PicfAnchorPriorGraphState(...)
```

## 7. Modality Priors

### 7.1 PaliGemma Priors

Inputs:

```text
image_tokens over PaliGemma padded grid
text_tokens
view_transforms
```

Outputs:

```text
role-conditioned heatmaps:
  H_task
  H_effector
  H_interaction
```

Pseudocode:

```python
def _build_paligemma_priors(self, semantic):
    if semantic.image_tokens is None:
        return None

    static_tokens = select_view(semantic, "static")
    text_summary = semantic.text_tokens.mean(dim=0)

    logits = self.mapg_pg_heatmap_head(static_tokens, text_summary)
    h_task = softmax(logits[:, ROLE_TASK] / temperature)
    h_eff = softmax(logits[:, ROLE_EFFECTOR] / temperature)
    h_int = softmax(logits[:, ROLE_INTERACTION] / temperature)

    return pg_priors_by_role
```

Rules:

- Keep PaliGemma input distribution intact.
- Do not inject point/tactile/posterior tokens into PaliGemma.
- Use view-transform metadata for any coordinate compatibility.

### 7.2 PaliGemma Grid To V-JEPA Grid

Build `T_pg_to_v` with geometry:

```text
PaliGemma padded token center
-> inverse resize-with-pad
-> original static camera pixel
-> V-JEPA visual grid cell
```

Pseudocode:

```python
def _build_pg_to_visual_compatibility(view_transform, pg_hw, visual_hw):
    pg_centers = grid_centers(pg_hw)
    original_uv = inverse_resize_with_pad(pg_centers, view_transform)
    visual_xy = pixel_to_visual_grid(original_uv, visual_hw)
    return gaussian_grid_compatibility(pg_centers, visual_xy)
```

Then:

```python
p_v_from_pg = normalize(T_pg_to_v.T @ p_pg)
```

This path works without point cloud.

### 7.3 V-JEPA Visual Native Support

Every graph anchor should have:

```text
p_v^k over V-JEPA visual tokens
z_v^k = pool(visual_tokens, p_v^k)
```

If PaliGemma grounding is unavailable, visual supports can still be proposed
from:

- direct visual saliency heads
- posterior projection
- tactile contact projection
- coverage anchors over the visual grid

### 7.4 Optional Point / Sonata Support

Only use point support when:

```text
point_positions_world is available
point_projectable_mask has valid rows
projective compatibility has visible mass
```

Use column-normalized compatibility:

```python
C = projective_compatibility * point_projectable_mask[:, None]
C_col = C / clamp(C.sum(dim=0, keepdim=True), eps)
p_p_from_v = normalize(C_col @ p_v)
```

This prevents point-dense regions from winning only because they contain more
rows.

Point pooling:

```python
z_p^k = pool(point_tokens, p_p^k)
```

Point geometry:

```python
x_k = sum_p p_p^k[p] * point_positions_world[p]
S_k = weighted_cov(point_positions_world, p_p^k, x_k)
```

If visible mass is below threshold:

```text
point support invalid
geometry_valid=False
```

No fake point fallback is allowed.

### 7.5 Optional Tactile / AnyTouch Support

Tactile contact is not a point. It is a contact volume.

Inputs:

```text
tactile tokens
tactile contact logits/probabilities
wrist pose or tactile sensor pose
tactile positions/normals if available
```

Construct a probabilistic contact volume:

```text
center = wrist_position + d * wrist_forward
covariance = anisotropic ellipsoid
confidence = contact_probability * tactile_feature_confidence
```

Then project it into supports:

```text
p_t^k over tactile tokens
p_v_from_t^k over visual grid
p_p_from_t^k over point rows
```

Pseudocode:

```python
def _build_tactile_contact_volume_prior(observation, token_field):
    contact = sigmoid(token_field.tactile_contact_logits)
    if contact.max() < min_contact_confidence:
        return invalid

    volume = build_ellipsoid_from_wrist_or_sensor_pose(...)
    p_t = normalize(contact * tactile_role_scores)
    p_v = project_volume_to_visual_grid(volume)
    p_p = evaluate_volume_density(point_positions_world)
    return tactile_priors
```

### 7.6 Optional Posterior Support

Previous posterior anchors produce temporal priors:

```text
p_post over previous anchors
p_v_from_post through camera projection
p_p_from_post through Mahalanobis kernels over point positions
```

Pseudocode:

```python
def _build_posterior_prior(previous, token_field):
    if previous is None:
        return invalid
    for anchor in previous.posterior:
        if anchor.alpha < threshold:
            continue
        p_post = identity_or_role_prior(anchor)
        p_v = project_gaussian_to_visual(anchor.x, anchor.S)
        p_p = mahalanobis_kernel(point_positions_world, anchor.x, anchor.S)
    return posterior_priors
```

Posterior priors preserve identity and handle temporary occlusion, but do not
overwrite current observation evidence.

## 8. Shared Graph Fusion

### 8.1 Confidence-Gated Mixture

Default fusion is a mixture, not product-of-experts:

```text
p_m^k = normalize(sum_s alpha_{s->m}^k * T_{s->m}(p_s^k))
```

Reason:

```text
PoE can over-sharpen and collapse under noisy point/depth/tactile calibration.
Mixture is robust to missing or unreliable modalities.
```

PoE is allowed only as an optional high-confidence mode:

```text
log p_m^k = sum_s alpha_s * log(T_{s->m}(p_s^k) + eps)
```

### 8.2 Role-Constrained Candidate Assignment

Each candidate has:

```text
role
modality supports
score
confidence
geometry validity
```

Observation/task slots also have roles:

```text
effector/contact
task/object
interaction/affordance
posterior-memory
coverage/background
```

Use entropic Sinkhorn with role masks:

```python
cost[j, c] =
    - role_match(slot_role[j], candidate_role[c])
    - visual_score[j, c]
    - point_score[j, c] if point_valid
    - tactile_score[j, c] if tactile_valid
    - posterior_score[j, c] if posterior_valid
    + overlap_penalty[j, c]

cost[role_incompatible] = large_positive
assignment = sinkhorn(-cost / temperature)
```

This matches the existing PICF use of Sinkhorn-style posterior binding and
avoids brittle index-based proposal assignment.

## 9. PICF Consumers

### 9.1 Observation Anchors

Observation anchors receive graph priors as query additions and attention
biases:

```python
queries_j += gate_query * graph_anchor_token_j

point_logits_j += gate_p * clipped_centered_log(p_point_j)
visual_logits_j += gate_v * clipped_centered_log(p_visual_j)
tactile_logits_j += gate_t * clipped_centered_log(p_tactile_j)
```

Rules:

- Apply role masks before adding priors.
- Missing modality bias is zero.
- Clamp and center log priors before adding to logits.
- Keep current observation-anchor reader path intact when graph disabled.

### 9.2 Task Readout

Task readout should become visual-native first:

```python
direct_visual = task_visual_attention
graph_visual = assigned_graph_visual_prior
p_v_task = normalize((1 - g_v) * direct_visual + g_v * graph_visual)

task_token = local_task_token + MLP(pool(visual_tokens, p_v_task))
```

If point support is valid:

```python
p_p_task = normalize((1 - g_p) * direct_point + g_p * graph_point)
x, S, a = moments(point_positions_world, p_p_task)
task_token += MLP(pool(point_tokens, p_p_task))
task_token += geometry_pe(x, S, a)
```

If tactile support is valid:

```python
task_token += MLP(pool(tactile_tokens, p_t_task))
```

If point support is invalid, task readout remains valid through visual support.

### 9.3 Posterior Binding

Add graph priors as soft binding bias only:

```python
binding_logits += gate_sem * semantic_anchor_score
binding_logits += gate_vis * visual_support_overlap
binding_logits += gate_point * point_support_overlap
binding_logits += gate_post * posterior_identity_prior
```

Forbidden:

```text
graph.x -> posterior.x direct assignment
graph.token -> posterior.hidden direct assignment
graph.heatmap -> posterior binding hard assignment
```

### 9.4 Conditioned Control And PI0.5 Prefix

Conditioned control may consume graph anchor tokens:

```python
control_context = [
    posterior tokens,
    task readout tokens,
    graph anchor tokens,
    proprio/innovation tokens,
]
```

The final action path stays:

```text
conditioned_control.pi_prefix_tokens
-> extra_prefix_tokens
-> PI0.5 action flow matching / sampling
```

## 10. Training Losses

The graph training objective should be one coherent objective, not a pile of
temporary patch losses:

```text
L_total =
  L_action
+ L_existing_PICF
+ lambda_graph * L_MAPG
```

where:

```text
L_MAPG =
  L_anchor_siglip_or_infonce
+ L_anchor_vicreg
+ L_distribution_cycle
+ L_masked_modality_prediction
+ L_role_routing_consistency
+ optional L_pose_or_object_heatmap
```

All terms use modality availability masks, confidence masks, and role masks.

### 10.1 Anchor-Level SigLIP / InfoNCE

For each anchor `k` and modality `m`:

```python
e_m_k = normalize(MLP_m(pool(tokens_m, p_m_k)))
```

Positive pairs:

```text
same anchor, different modalities
```

Hard negatives:

```text
different anchors in the same frame, including visually similar objects
```

SigLIP-style pairwise loss:

```text
L_pos = -log sigmoid(dot(e_m^k, e_n^k) / tau)
L_neg = -log sigmoid(-dot(e_m^k, e_n^j) / tau), j != k
```

This is modality-optional and works with variable positive/negative sets.

### 10.2 VICReg Anti-Collapse

For embeddings from each available modality:

```text
L_var: each dimension std must stay above threshold
L_cov: off-diagonal covariance should be small
L_inv: same-anchor cross-modal embeddings should be close
```

This is the mature replacement for ad-hoc anchor repulsion.

### 10.3 Cycle Consistency

When compatibility maps are valid:

```text
p_v -> T_v_to_p -> p_p -> T_p_to_v -> p_v_cycle
L_cycle = JS(stopgrad(p_v), p_v_cycle)
```

Other valid cycles:

```text
visual <-> tactile
visual <-> posterior
point <-> tactile
point <-> posterior
```

Do not apply cycle loss to invalid projections.

### 10.4 Masked Modality Prediction

Randomly mask one modality support and predict its pooled embedding from the
remaining graph:

```text
mask point -> predict e_point from visual/tactile/posterior
mask tactile -> predict e_tactile from visual/point/posterior
mask visual -> predict e_visual from point/tactile/posterior
```

Loss:

```text
1 - cosine(predicted_e_m, stopgrad(e_m))
```

This makes MAPG robust to missing pointcloud or tactile streams.

### 10.5 Optional Heatmap Supervision

Use only when the target is valid.

Safe targets:

```text
current gripper projection -> effector heatmap
future keypose / gripper open-close projection -> interaction heatmap
true object mask/bbox/label -> task heatmap
```

Do not pretend a future gripper keypose is object segmentation. If true object
labels are absent, keep `task_heatmap` unsupervised or weakly supervised only.

## 11. Implementation Milestones

### M0: Documentation And Audit

Deliver:

- this README
- `/tmp/picf_mapg_picf_code_math_audit_20260505.md`
- navigation from `README_v2.2.md` and `src/openpi/picf/README.md`

No runtime behavior change.

### M1: State Contracts And Default-Off Config

Add:

```python
mapg_anchor_graph_enabled: bool = False
mapg_num_graph_anchors: int = 8
mapg_visual_primary: bool = True
mapg_use_point_support: bool = True
mapg_use_tactile_support: bool = True
mapg_use_posterior_support: bool = True
mapg_role_sinkhorn_iters: int = 4
mapg_log_prior_bias_clip: float = 4.0
mapg_min_visible_mass: float = 1e-4
```

Add dataclasses but do not consume them yet.

Required tests:

```text
config parse
model construction with enabled/disabled MAPG
default-off parameter absence or no-op
```

### M2: Visual-Native Graph

Implement:

```text
PaliGemma pg priors
T_pg_to_v with resize-with-pad metadata
visual-native anchor priors over V-JEPA tokens
visual coverage fallback
```

No pointcloud dependency.

### M3: Optional Point Support

Implement:

```text
point_positions_model_frame
point_positions_world
point_projectable_mask
T_v_to_p and T_p_to_v
column-normalized compatibility
visible-mass checks
```

No fake fallback.

### M4: Tactile Contact-Volume Support

Implement:

```text
tactile contact volume
T_t_to_v
T_t_to_p
tactile confidence gating
```

### M5: Posterior Temporal Support

Implement:

```text
T_post_to_v
T_post_to_p
posterior identity prior
posterior support confidence
```

### M6: Role-Constrained Assignment

Implement:

```text
candidate roles
slot roles
role masks
Sinkhorn candidate-to-slot assignment
coverage/background slots
```

### M7: Default-Off Consumers

Wire graph into:

```text
observation anchors
task readout
posterior binding
conditioned control
```

All gates default no-op.

### M8: Graph Losses

Add zero-default losses:

```text
anchor SigLIP / InfoNCE
VICReg
cycle consistency
masked modality prediction
role routing consistency
optional heatmap loss
```

### M9: Diagnostics

Export:

```text
per-anchor visual priors
per-anchor point priors
per-anchor tactile priors
assignment matrices
support entropy
top-k support modes
geometry validity
covariance ellipses when geometry valid
role confusion metrics
```

CALVIN diagnostic videos should include separate products:

```text
points-only overlay
heatmap-only overlay
covariance/ellipse overlay
raw JSONL
```

## 12. Verification Plan

### 12.1 Local Static And Unit Checks

Before any cloud run:

```bash
cd /home/siyuanyue/Documents/openpi

uv run python -m py_compile \
  src/openpi/picf/paligemma/wrapper.py \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/training.py \
  scripts/picf_core_train.py

uv run pytest -q src/openpi/picf/paligemma/wrapper_test.py
uv run pytest -q src/openpi/picf/core/pipeline_test.py -k 'vl_ or mapg'
uv run pytest -q src/openpi/picf/core/training_test.py -k 'alignment or mapg'
uv run pytest -q scripts/picf_core_train_test.py -k 'vl_anchor_router or mapg'
uv run python scripts/verify_picf_contract.py
git diff --check
```

### 12.2 Required New Tests

Add tests for:

```text
PaliGemma padded grid -> original pixel -> V-JEPA grid compatibility.
RGB-only graph construction.
pointcloud-missing graph construction.
pointcloud-invalid visible-mass no-op.
local-frame point rows excluded from projection.
column-normalized visual-to-point mass conservation.
role masks preventing effector prior from occupying all scene slots.
default-off exact no-op.
posterior not hard-overwritten.
VICReg loss finite and non-negative.
SigLIP/InfoNCE loss with missing modalities.
cycle loss disabled under invalid compatibility.
masked modality prediction under point/tactile dropout.
```

### 12.3 Cloud Diagnostic Run

Before long training:

```text
500-1000 diagnostic steps
checkpoint interval 250 for diagnostics only
MAPG enabled but graph losses can start at zero or very small weights
export graph JSONL
confirm losses finite
confirm gates finite
confirm visual-native anchors exist when pointcloud invalid
confirm point support invalid does not destroy task readout
```

Long-run profile should be separate from diagnostics.

## 13. What Not To Do

Do not:

```text
inject point/tactile/posterior tokens into PaliGemma backbone
replace posterior state with PaliGemma/V-JEPA predictions
make pointcloud required for anchor existence
use direct PaliGemma-to-point resize without view-transform metadata
use local-frame point rows for camera projection
let gripper/contact priors fill every task/object slot
use object-looking keypose supervision as if it were object segmentation
use only coordinate-distance repulsion as anti-collapse training
enable graph losses by default before no-op and invalid-projection tests pass
```

## 14. Current Verdict

MAPG is compatible with the current PICF codebase because the repo already has:

```text
PaliGemma spatial token metadata
V-JEPA visual tokens
Sonata point tokens
AnyTouch tactile tokens
posterior anchors
projective geometry
role-aware observation anchors
soft posterior binding bias substrate
```

But MAPG should be implemented as a graph-level state and loss family, not as a
larger version of the current point-centric VL router.

The correct interpretation is:

```text
README_VL_GUIDED_ANCHOR_ROUTER.md
  = current point-centric, default-off router substrate.

README_MAPG_PICF.md
  = next full architecture contract for modality-optional anchor graph routing.
```

No long-run training should claim to use MAPG until the milestones above are
implemented, tested, and explicitly enabled in the training command.
