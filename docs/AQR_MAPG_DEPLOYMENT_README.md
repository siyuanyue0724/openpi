# AQR-MAPG Deployment Readme

Date: 2026-05-09

This document is the proposed replacement plan after reviewing the current MAPG evidence bundle:

```text
/mnt/checkpoints/picf_core/evidence/mapg_anchor_collapse_evidence_selected_20260509_084142
```

The core conclusion is intentionally conservative:

```text
Current live MAPG behaves as control-path noise in the observed run.
The MAPG idea should not be deleted, but the current candidate-prior assignment implementation should not drive control.
The replacement should be AQR-MAPG:
Typed Support Memory + Task/Role-Conditioned Anchor Query Router.
```

This is not a cosmetic rename. It changes the routing primitive from:

```text
PaliGemma / hand-built priors -> graph anchors -> same-role slot assignment
```

to:

```text
typed support memory -> task/role-conditioned anchor queries -> support distributions -> OT competition
```

The PI0.5 action path remains unchanged.

---

## 1. Current Evidence

The evidence bundle shows a concrete failure, not a vague visual artifact.

For episode 0, step 30, task:

```text
press the button to turn on the led light
```

the dense debug reports:

```text
vl_task_heatmap entropy ~= 0.9967
vl_task_heatmap peak_to_uniform ~= 1.10
```

That is essentially a near-uniform heatmap. It does not provide a reliable language-grounded task point.

The MAPG graph anchors are not all identical:

```text
same_role_point_overlap_max  ~= 0.0323
same_role_visual_overlap_max ~= 0.0201
```

But after observation assignment:

```text
obs_graph_point row1-row15 max_abs_diff_to_row1 ~= 3.7e-09
ep0 role1_unique_centers median = 1
ep0 role1_unique_centers min    = 1
ep0 role1_unique_centers max    = 1
ep0 role1_frac_unique1          = 1.0
```

So the collapse is not mainly that every MAPG graph anchor is identical. The observed collapse is:

```text
same-role object slots consume the same graph mixture.
```

The relevant current implementation is:

```python
logits = torch.log(local_scores)[None, :].expand(int(rows.numel()), -1) / temperature
```

This makes all rows of a role group start with identical assignment logits. Column balancing / Sinkhorn-style rescaling cannot break that symmetry because no row-specific evidence is present.

---

## 2. Problem Essence

The current failure is best summarized as:

```text
near-uniform PaliGemma heatmap
+ geometry / coverage-biased MAPG basins
+ same-role assignment row symmetry
+ no direct task-keypoint supervision
= all role1 object slots collapse to one non-task-specific graph mixture
```

In this state, current MAPG is not a useful control prior. It is noise because it injects a confident-looking but task-weak graph mixture into observation/task/control.

This does not prove that all MAPG-style anchor graphs are wrong. It proves that the current live MAPG implementation is using the wrong primitive:

```text
candidate prior assignment
```

where the correct primitive should be:

```text
query-conditioned support routing
```

---

## 3. Self-Critique

### 3.1 What was wrong in the previous MAPG framing?

The previous implementation implicitly assumed that:

```text
PaliGemma heatmap + graph priors + support diversity
```

would be enough to produce task-relevant object anchors.

That assumption was too optimistic.

The evidence shows that a generic VLM heatmap can be near-uniform in this setting. If the VLM signal is weak, graph priors fall back to geometry / coverage / saliency basins. Those basins may be visually structured, but they are not necessarily task interaction points.

### 3.2 Why is increasing anchor loss not sufficient?

The failure is not just a weak regularizer. It is a symmetry problem:

```text
same role rows get the same logits.
```

If the input rows are symmetric, stronger support diversity can make graph anchors different, but it does not force object slots to choose different anchors. The slot assignment still has no reason to distinguish rows.

### 3.3 Why is a standalone keypoint extractor also unsatisfactory?

A standalone extractor would split the system into two semantics:

```text
extractor predicts points
PICF / MAPG reinterprets points
control may or may not use them
```

This increases debugging ambiguity:

```text
Was the point wrong?
Was MAPG fusion wrong?
Did task readout ignore the point?
Did posterior overwrite it?
```

The model should instead use one internal object:

```text
support distributions read by anchor queries
```

Those same distributions should feed observation, task, posterior, control, losses, and debug.

### 3.4 Why not delete MAPG entirely?

Deleting MAPG would remove a useful abstraction:

```text
anchor = multi-modal support distribution + token + optional geometry
```

That abstraction is still the right long-term object for missing-modality data. The problem is not the anchor object. The problem is how anchors are currently seeded and assigned.

---

## 4. Final Architecture: AQR-MAPG

Name:

```text
AQR-MAPG
Anchor Query Router over Typed Support Memory
```

The final dataflow should be:

```text
Observation
  RGB / pointcloud / tactile / proprio / prompt
        |
        v
Backbone writers
  PaliGemma semantic/text/image tokens
  V-JEPA dense visual tokens
  Sonata point tokens
  AnyTouch tactile tokens
  posterior tokens
        |
        v
Typed Support Memory
        |
        v
Task/role/posterior-conditioned anchor queries
        |
        v
Query-to-support cross-attention
  p_v^k    over visual tokens
  p_p^k    over point tokens
  p_t^k    over tactile tokens
  p_post^k over posterior anchors
  p_mem^k  over compact temporal memory
        |
        v
Sinkhorn / OT competition
        |
        v
Anchor states
  support distributions
  anchor token z_k
  optional geometry x/S
  confidence / role / modality validity
        |
        +--> observation anchors
        +--> task readout
        +--> posterior binding bias
        +--> conditioned control
                  |
                  v
               PI0.5 action
```

This is not a separate keypoint model. The anchor query router is the keypoint/support extractor.

---

## 5. Typed Support Memory

The memory should be typed. Do not treat it as a generic LLM-style KV cache.

```text
visual memory:
  V-JEPA dense tokens
  2D grid position
  camera/view id
  optional PaliGemma image-token aligned metadata

point memory:
  Sonata/point tokens
  xyz_world
  visibility
  projectability
  point-pool id

tactile memory:
  AnyTouch tokens
  contact probability
  sensor pose / contact volume

posterior memory:
  persistent posterior anchor tokens
  x/S
  role
  alpha/confidence

temporal compact memory:
  small set of scene/object/contact tokens
  not all previous dense tokens
```

The memory must be read through queries:

```text
complexity = O(KN) cross-attention + O(K^2) query self-attention
```

Do not use full dense self-attention over all visual/point tokens.

---

## 6. Anchor Queries

There should be two query families.

### 6.1 Physical observation queries

Purpose:

```text
stable task-neutral-ish physical anchoring for observation and posterior update
```

Query construction:

```text
q_obs_k = f(
  role_k,
  coverage_code_k,
  proprio,
  previous_posterior_anchor_k,
  local/contact code
)
```

These queries should avoid strong task-specific PaliGemma heatmap bias. They may use task text only weakly or not at all.

### 6.2 Task readout queries

Purpose:

```text
language/task-conditioned where for action readout
```

Query construction:

```text
q_task_k = f(
  role_k,
  coverage_code_k,
  task_embedding,
  PaliGemma summary,
  proprio,
  posterior_summary
)
```

These queries can use PaliGemma weak heatmap bias, but only through confidence gating.

This split prevents task text from corrupting physical posterior memory.

---

## 7. Query-to-Support Routing Contract

For modality `m`:

```text
logits_m[k, i] =
  <Wq_m q_k, Wk_m token_m_i> / sqrt(d)
  + B_pos_m[k, i]
  + B_role_m[k, i]
  + B_valid_m[i]
  + B_prior_m[k, i]
```

Then:

```text
p_m^k = softmax_or_sinkhorn(logits_m[k, :])
```

The support distributions are the primary output:

```text
p_v^k    visual support over V-JEPA grid
p_p^k    point support over world points
p_t^k    tactile/contact support
p_post^k posterior identity support
p_mem^k  temporal compact memory support
```

Anchor token:

```text
r_m^k = sum_i p_m^k[i] * V_m[i]

beta_m^k = Gate(
  q_k,
  role_k,
  modality_valid_m,
  entropy(p_m^k),
  confidence_m
)

z_k = sum_m beta_m^k * r_m^k
      + role_embedding
      + confidence_embedding
      + geometry_embedding_if_valid
```

Optional geometry:

```text
x_k = sum_j p_p^k[j] * X_world[j]
S_k = covariance(p_p^k, X_world)
```

If point support is invalid:

```text
geometry_valid = False
```

The anchor is still valid through visual/tactile/posterior support.

---

## 8. Competition: Sinkhorn / OT

Competition is required. It is not an optional regularizer.

The system must avoid:

```text
all queries attending to the same support basin
```

Use role-constrained Sinkhorn / OT over query-support or slot-anchor assignment:

```text
A = Sinkhorn((quality + query_support_compatibility + role_mask) / tau)
```

Constraints:

```text
row marginal:
  each query explains a unit of mass

column capacity:
  each support region / graph anchor cannot absorb unlimited mass

role mask:
  effector, task, interaction, coverage roles do not collapse into one role
```

The missing term in current MAPG is:

```text
query_support_compatibility[j, k]
```

Current assignment uses mostly anchor score / confidence / role mask. That is not enough.

---

## 9. PaliGemma Role

PaliGemma should not be the primary point generator.

Correct role:

```text
semantic conditioner
weak language-conditioned visual prior
optional heatmap supervision target/source
```

Incorrect role:

```text
final point oracle
primary MAPG seed
posterior writer
raw 3D/tactile consumer
```

PaliGemma heatmap bias should enter as:

```text
B_pg[k, i] = lambda_pg * c_pg * clipped_centered_log(p_pg[i] + eps)
```

where:

```text
c_pg = confidence_gate(
  entropy_norm,
  peak_to_uniform,
  temporal_stability,
  support_consistency
)
```

For a near-uniform heatmap like:

```text
entropy_norm ~= 0.9967
peak_to_uniform ~= 1.10
```

the gate should drive:

```text
c_pg ~= 0
```

so PaliGemma does not silently inject noise.

---

## 10. Training Objective

Use one coherent objective:

```text
L =
  L_action
  + L_existing_PICF
  + lambda_router * L_router
```

Router objective:

```text
L_router =
  L_keypoint_interaction
  + L_support_competition
  + L_cross_modal_siglip
  + L_vicreg
  + L_masked_modality_prediction
  + L_temporal_consistency
```

### 10.1 Keypoint / interaction loss

Directly supervise support distributions, not a side extractor.

Possible labels:

```text
effector heatmap:
  current gripper projection

interaction heatmap:
  future keypose projection
  gripper open/close transition
  contact / tactile event

task heatmap:
  object labels / masks / bboxes if available
  otherwise use conservative future-interaction proxies
```

Do not treat future keypose as object segmentation. It is an interaction proxy.

### 10.2 Support competition loss

Use kernelized normalized overlap:

```text
O_m(i,j) =
  p_m_i^T K_m p_m_j
  / sqrt((p_m_i^T K_m p_m_i)(p_m_j^T K_m p_m_j) + eps)
```

Loss:

```text
L_support =
  mean_{same-role i<j}
    usage_i usage_j confidence_i confidence_j
    sum_m gamma_m valid_m relu(O_m(i,j) - margin_m)^2
```

This is a support-level anti-collapse term. It does not replace query-specific assignment.

### 10.3 Cross-modal SigLIP

Same anchor, different modality:

```text
positive
```

Different anchors in the same frame:

```text
negative, including visually similar objects
```

Use validity masks so missing/invalid modalities do not produce dirty positives or negatives.

### 10.4 VICReg

Apply to:

```text
graph / anchor tokens
visual pooled embeddings
point pooled embeddings
tactile pooled embeddings
posterior pooled embeddings
```

Do not only apply VICReg to fused anchor tokens.

### 10.5 Masked modality prediction

Randomly mask:

```text
point
tactile
posterior
visual patches
```

Predict the masked modality pooled embedding from remaining supports:

```text
1 - cosine(pred_e_m, stopgrad(e_m))
```

This is necessary for missing-modality robustness.

---

## 11. Deployment Plan

### Phase 0: Safety freeze

Goal:

```text
stop current MAPG from injecting control-path noise
```

Current safe training setting:

```text
picf_mode=enabled
semantic_mode=paligemma
vl_anchor_router_enabled=false
mapg_enabled=false
```

This keeps the main PICF / PI0.5 path available without current MAPG graph prior.

If MAPG diagnostics are needed while control is protected, add a new explicit mode:

```text
mapg_consumer_mode = "none" | "task_only" | "full"
```

Required behavior:

```text
none:
  build/export MAPG debug only
  do not alter observation/task/posterior/control

task_only:
  allow task readout diagnostics
  do not write posterior
  do not alter observation anchors

full:
  only after AQR validation passes
```

Do not rely only on negative gate initializers. A learned gate can still become active.

### Phase 1: Task-keypoint support head inside PICF

Goal:

```text
learn where explicitly without a standalone extractor
```

Implement:

```text
task/interaction/effector heatmaps over V-JEPA grid
optional lift to point support when point projection is valid
```

Inputs:

```text
V-JEPA visual tokens
task embedding
proprio
posterior summary
```

Outputs:

```text
p_v_task
p_v_interaction
p_v_effector
```

Losses:

```text
future keypose projection
gripper open/close transition
contact/tactile event
current effector projection
multi-view consistency if available
```

Backup:

```text
if keypoint entropy remains near uniform after N steps, keep MAPG disabled and inspect labels/proxies
```

### Phase 2: Typed Support Memory

Goal:

```text
make all consumers read the same support memory
```

Implement:

```text
PicfSupportMemoryState
```

Fields:

```text
visual_tokens, visual_pos, visual_mask, visual_confidence
point_tokens, xyz_world, point_projectable_mask, point_visibility
tactile_tokens, tactile_pose, contact_prob
posterior_tokens, posterior_x, posterior_S, posterior_alpha, posterior_role
temporal_tokens, temporal_role, temporal_confidence
semantic_summary, task_embedding
```

Backup:

```text
memory writer only, no consumer changes
export memory diagnostics
```

### Phase 3: Anchor Query Router

Goal:

```text
replace candidate-prior assignment with query-conditioned support routing
```

Implement:

```text
PicfAnchorQueryRouter
```

Two query groups:

```text
physical_queries
task_queries
```

Each query cross-attends typed memory and outputs:

```text
visual_support
point_support
tactile_support
posterior_support
anchor_token
confidence
optional x/S
geometry_valid
```

Use:

```text
role-constrained Sinkhorn / OT
```

Backup:

```text
router exports supports but does not alter control until validation passes
```

### Phase 4: Controlled reintegration

Enable consumers one at a time:

```text
1. task readout only
2. conditioned control prefix only
3. observation anchors
4. posterior binding
```

Do not enable posterior writes before task readout and control diagnostics are stable.

Validation gate:

```text
role1 unique centers > 1 for multi-object frames
task heatmap entropy lower than uniform baseline
task heatmap peak_to_uniform meaningfully above 1
anchor centers temporally stable without collapsing
CALVIN eval does not regress against no-MAPG PICF baseline
```

### Phase 5: Full AQR-MAPG training

Only after Phase 4 passes:

```text
mapg_consumer_mode=full
```

Keep ablations:

```text
no_router
task_query_only
physical_query_only
no_pg_bias
no_point_support
no_tactile_support
no_temporal_memory
```

---

## 12. Backup Strategy

There must always be a clean fallback:

```text
Fallback A:
  PICF with MAPG disabled

Fallback B:
  task-keypoint head only
  no MAPG graph fusion

Fallback C:
  AQR task queries only
  no physical posterior write

Fallback D:
  AQR full
```

Promotion rule:

```text
only promote if action loss, anchor diagnostics, and CALVIN eval all improve or at least do not regress
```

Rollback rule:

```text
if role1 collapse reappears or keypoint heatmap remains near-uniform,
drop to the previous fallback immediately
```

---

## 13. Success Metrics

Do not judge only by loss.

Required diagnostics:

```text
task_heatmap_entropy
task_heatmap_peak_to_uniform
interaction_heatmap_entropy
role1_unique_centers_per_frame
same_role_support_overlap
assignment row diversity
anchor temporal jitter
anchor-to-task-object consistency
posterior dustbin mass
CALVIN success / subtask success
action loss
```

Failure thresholds:

```text
PaliGemma heatmap entropy > 0.98 and peak_to_uniform < 1.5:
  PaliGemma bias must no-op

role1_unique_centers median == 1 across multi-object frames:
  assignment/router is still collapsed

obs/task/control improves visually but action loss worsens:
  keep router debug-only
```

---

## 14. Mathematical Consistency Check

The proposed system is internally consistent because:

```text
memory tokens are typed evidence
queries are latent anchors
support distributions are normalized probability measures
Sinkhorn / OT supplies competition
anchor tokens are pooled representations of supports
geometry is an optional moment of point support
posterior memory is physical and not overwritten by task text
PaliGemma is a weak semantic bias, not a spatial oracle
```

This avoids the current inconsistency:

```text
same-role slots receive identical assignment evidence
```

and avoids the overcorrection:

```text
standalone keypoint extractor disconnected from PICF / posterior / control
```

---

## 15. Literature Alignment

The design is aligned with the following principles:

```text
Slot Attention:
  slots specialize through competitive attention.

MESH / OT Slot Attention:
  Sinkhorn / OT changes competition and tie-breaking for object-centric learning.

DETR:
  learned queries plus matching reduce duplicate object predictions.

Perceiver IO:
  latent queries read large inputs with O(KN) scaling.

TokenLearner:
  small adaptive token sets can summarize dense visual inputs.

Transporter / CLIPort / PerAct:
  manipulation requires explicit spatial where/action-point learning.

Spatial Forcing:
  explicit point/depth inputs can be noisy or unavailable, so visual-native support must remain first-class.

SigLIP / VICReg:
  pairwise cross-modal matching and anti-collapse regularization are suitable support objectives, but not replacements for query-specific routing.
```

References:

```text
Slot Attention: https://arxiv.org/abs/2006.15055
MESH / OT Slot Attention: https://arxiv.org/abs/2301.13197
DETR: https://arxiv.org/abs/2005.12872
Perceiver IO: https://arxiv.org/abs/2107.14795
TokenLearner: https://arxiv.org/abs/2106.11297
Transporter Networks: https://arxiv.org/abs/2010.14406
CLIPort: https://arxiv.org/abs/2109.12098
PerAct: https://arxiv.org/abs/2209.05451
VICReg: https://arxiv.org/abs/2105.04906
SigLIP: https://arxiv.org/abs/2303.15343
```

---

## 16. Final Recommendation

The current MAPG live path should be treated as harmful in the observed configuration.

Do:

```text
disable current MAPG consumers for new mainline training
keep MAPG evidence/debug tools
add task-keypoint supervision inside PICF
build typed support memory
replace current assignment with anchor query routing
reintegrate consumers gradually behind explicit gates
```

Do not:

```text
keep current MAPG full control path active
trust near-uniform PaliGemma heatmaps
increase anchor losses as the main fix
add only geometry repulsion
build a standalone keypoint extractor disconnected from PICF
delete the anchor/support abstraction entirely
```

The target final system is:

```text
AQR-MAPG:
Typed Support Memory
+ Task/Role-Conditioned Anchor Queries
+ Query-to-Support Cross-Attention
+ Sinkhorn / OT Competition
+ Explicit Task-Keypoint Supervision
+ PI0.5 Action Path Unchanged
```

That is the mathematically coherent backup-safe replacement for current MAPG.
