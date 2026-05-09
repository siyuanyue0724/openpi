# AQR-MAPG Direct Final Deployment Contract

Date: 2026-05-09

This document is the direct-to-final deployment contract. It intentionally does **not** propose keeping the current MAPG-v0 live path as a production intermediate.

The final target is:

```text
AQR-MAPG
Anchor Query Router over Typed Support Memory
```

The decision is:

```text
Current MAPG-v0 should be disabled as a control-path mechanism.
Do not continue PaliGemma heatmap -> point prior -> same-role assignment as the main route.
Do not create a disconnected keypoint extractor.
Deploy one integrated final router:
Typed Support Memory + Task/Role-Conditioned Anchor Queries + OT Competition + Explicit Interaction Supervision.
```

This is the final architecture contract, not a diagnostic-only router.

---

## 1. Evidence Summary

Evidence bundle:

```text
/mnt/checkpoints/picf_core/evidence/mapg_anchor_collapse_evidence_selected_20260509_084142
```

The important observed facts are:

```text
vl_task_heatmap entropy ~= 0.9967
vl_task_heatmap peak_to_uniform ~= 1.10

mapg.same_role_point_overlap_max  ~= 0.0323
mapg.same_role_visual_overlap_max ~= 0.0201

obs_graph_point row1-row15 max_abs_diff_to_row1 ~= 3.7e-09
ep0 role1_unique_centers median = 1
ep0 role1_unique_centers min    = 1
ep0 role1_unique_centers max    = 1
ep0 role1_frac_unique1          = 1.0
```

Interpretation:

```text
1. PaliGemma heatmaps are too close to uniform to be trusted as a spatial oracle.
2. MAPG graph anchors are not necessarily all identical.
3. Collapse happens after same-role assignment: all role1 slots consume the same graph mixture.
4. Current MAPG-v0 is therefore control-path noise in this configuration.
```

The root code pattern is:

```python
logits = torch.log(local_scores)[None, :].expand(int(rows.numel()), -1) / temperature
```

This creates identical assignment evidence for same-role rows. Capacity scaling cannot break symmetry without row-specific evidence.

---

## 2. Final Verdict on Current MAPG

Current MAPG-v0 should not be patched incrementally into production.

Reason:

```text
The failure is structural, not just a bad coefficient.
```

Increasing anchor losses, adding geometric repulsion, or tuning PaliGemma heatmap temperature does not fix:

```text
same-role row symmetry
weak task grounding
missing explicit interaction supervision
```

Therefore, the direct final deployment should replace the current MAPG-v0 routing primitive.

Keep:

```text
debug exporters
overlay tools
evidence analysis utilities
typed support concepts
anchor/support abstraction
```

Remove from live control:

```text
current graph-prior consumers
current PaliGemma-driven visual seed as a strong prior
current score-only same-role assignment
```

---

## 3. Final Architecture

Final dataflow:

```text
Observation
  RGB / depth / pointcloud / tactile / proprio / language
        |
        v
Frozen or partially frozen backbones
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
Task/role-conditioned anchor queries
        |
        v
Query-to-support cross-attention
        |
        v
Sinkhorn / OT competition
        |
        v
Anchor states
  p_visual^k
  p_point^k
  p_tactile^k
  p_posterior^k
  z_k
  optional x_k / S_k
  confidence_k
        |
        +--> observation anchors
        +--> task readout
        +--> posterior binding bias
        +--> conditioned control
                  |
                  v
               PI0.5 action path
```

This is not a separate keypoint extractor. The router itself is the keypoint/support selector.

---

## 4. Typed Support Memory

Create:

```python
PicfTypedSupportMemoryState
```

Required fields:

```text
visual_tokens:            [V, D]
visual_positions_2d:      [V, 2]
visual_view_ids:          [V]
visual_valid_mask:        [V]

point_tokens:             [P, D]
point_xyz_world:          [P, 3]
point_projectable_mask:   [P]
point_visibility:         [P]
point_pool_ids:           [P]

tactile_tokens:           [T, D]
tactile_contact_prob:     [T]
tactile_positions_world:  [T, 3] optional

posterior_tokens:         [M, D]
posterior_x:              [M, 3]
posterior_S:              [M, 3, 3]
posterior_alpha:          [M]
posterior_roles:          [M]

temporal_tokens:          [L, D]
temporal_roles:           [L]
temporal_confidence:      [L]

semantic_task_embedding:  [D]
paligemma_heatmap:        [V] optional
paligemma_heatmap_conf:   scalar
```

This is a support memory, not an LLM-style autoregressive KV cache. It stores typed evidence that can be queried.

Scale target:

```text
cross-attention: O(KN)
query self-attention: O(K^2)

K: 8-24 anchor queries
N: visual + point + tactile + posterior + compact memory tokens
```

No full dense-token self-attention should be introduced.

---

## 5. Anchor Queries

Create:

```python
PicfAnchorQueryRouter
```

The final model uses two query families in one router.

### 5.1 Physical queries

Purpose:

```text
stable physical supports for observation and posterior
```

Definition:

```text
q_phys_k = f(
  role_k,
  coverage_code_k,
  proprio,
  previous_posterior_summary,
  local_contact_code
)
```

These queries should not be dominated by task text.

### 5.2 Task queries

Purpose:

```text
task-conditioned interaction supports for task readout and control
```

Definition:

```text
q_task_k = f(
  role_k,
  coverage_code_k,
  task_embedding,
  PaliGemma semantic summary,
  proprio,
  posterior_summary
)
```

These queries can use PaliGemma as a weak prior if confidence is high.

### 5.3 Query identity

Every query must carry row-specific identity:

```text
role embedding
coverage code
query index embedding
posterior identity when available
task/physical type embedding
```

This directly prevents the current same-role row symmetry.

---

## 6. Query-to-Support Routing

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
p_m^k = normalize_or_sinkhorn(logits_m[k, :])
```

where:

```text
m in {visual, point, tactile, posterior, temporal}
```

Outputs:

```text
p_visual^k    over V-JEPA visual grid
p_point^k     over point tokens
p_tactile^k   over tactile/contact tokens
p_posterior^k over posterior anchors
p_temporal^k  over compact temporal memory
```

Pooled modality representation:

```text
r_m^k = sum_i p_m^k[i] * V_m[i]
```

Modality gate:

```text
beta_m^k = gate(
  q_k,
  role_k,
  modality_valid_m,
  entropy(p_m^k),
  confidence_m
)
```

Anchor token:

```text
z_k = sum_m beta_m^k * r_m^k
      + role_embedding
      + query_type_embedding
      + confidence_embedding
      + geometry_embedding_if_valid
```

Point geometry:

```text
x_k = sum_j p_point^k[j] * xyz_world[j]
S_k = covariance(p_point^k, xyz_world)
```

If point support is unavailable:

```text
geometry_valid = False
```

The anchor remains usable through visual/tactile/posterior support.

---

## 7. Sinkhorn / OT Competition

Competition must happen at routing time.

For each query group and modality:

```text
A_m = Sinkhorn((logits_m + role_mask + capacity_bias) / tau)
```

Required constraints:

```text
row marginal:
  each query receives support mass

column capacity:
  support tokens / graph regions cannot absorb unlimited anchors

role compatibility:
  physical, task, effector, interaction, coverage roles are constrained
```

For final anchor-to-consumer assignment:

```text
consumer_logits[j,k] =
  dot(consumer_query_j, z_k)
  + overlap(consumer_direct_support_j, p_anchor_k)
  + role_mask[j,k]
  + anchor_quality[k]
```

Then:

```text
consumer_assignment = Sinkhorn(consumer_logits / tau)
```

There must be no remaining path equivalent to:

```text
same role -> same logits -> same graph mixture
```

---

## 8. PaliGemma Contract

PaliGemma is retained, but only as:

```text
semantic conditioner
optional auxiliary heatmap diagnostic/source only when explicitly enabled
```

It is not:

```text
final point oracle
primary where module
posterior writer
raw 3D/tactile memory consumer
```

By default, PaliGemma heatmaps do not enter visual logits at all.
The production AQR where path is learned query-to-support attention over typed
support memory. If `aqr_pg_grounding_enabled=True` and
`aqr_pg_bias_weight>0` are both explicitly set for an ablation, PaliGemma
heatmap enters task-query visual logits only as:

```text
B_pg[k, i] =
  lambda_pg * c_pg * clipped_centered_log(p_pg[i] + eps)
```

Confidence:

```text
entropy_norm = H(p_pg) / log(N)
peak_ratio   = max(p_pg) / (1 / N)

c_pg =
  sigmoid(a * (entropy_threshold - entropy_norm))
  * sigmoid(b * (peak_ratio - peak_threshold))
  * temporal_stability
  * support_consistency
```

If:

```text
entropy_norm ~= 0.9967
peak_ratio   ~= 1.10
```

then:

```text
c_pg ~= 0
```

Therefore, in the observed failure case, PaliGemma contributes almost no spatial bias.

---

## 9. Training Objective

The final objective:

```text
L =
  L_action
  + L_picf_existing
  + lambda_router * L_router
```

Router loss:

```text
L_router =
  L_interaction_keypoint
  + L_support_competition
  + L_cross_modal_siglip
  + L_vicreg
  + L_masked_modality_prediction
  + L_temporal_consistency
  + L_geometry_aux
```

### 9.1 Interaction/keypoint loss

Supervise support distributions directly:

```text
effector support:
  current gripper projection

interaction support:
  future keypose projection
  gripper open/close event
  tactile/contact event

task support:
  object labels/masks/bboxes if available
  otherwise conservative future-interaction proxy
```

Important:

```text
future keypose is an interaction proxy, not object segmentation.
```

### 9.2 Support competition

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

This prevents support-level overlap but does not replace query-specific routing.

### 9.3 Cross-modal SigLIP

Positive:

```text
same anchor, different valid modalities
```

Negative:

```text
different anchors in same frame
```

Use validity masks for all modalities.

### 9.4 VICReg

Apply to:

```text
anchor tokens
visual pooled embeddings
point pooled embeddings
tactile pooled embeddings
posterior pooled embeddings
temporal pooled embeddings
```

Do not apply only to fused anchor tokens.

### 9.5 Masked modality prediction

Mask:

```text
visual patches
point support
tactile support
posterior support
temporal memory
```

Predict:

```text
masked pooled embedding
```

Loss:

```text
1 - cosine(pred, stopgrad(target))
```

### 9.6 Geometry auxiliary

Low weight only:

```text
d_ij^2 =
  (x_i - x_j)^T (S_i + S_j + sigma^2 I)^-1 (x_i - x_j)

L_geom =
  mean same-role valid pairs relu(margin - sqrt(d_ij^2 + eps))^2
```

Geometry is optional and must not be the primary anti-collapse mechanism.

---

## 10. Direct Deployment Implementation Checklist

This is the direct final implementation list. Do not deploy MAPG-v0 as an intermediate production state.

### 10.1 Config

Add:

```python
aqr_mapg_enabled: bool = False
aqr_query_count_physical: int = 16
aqr_query_count_task: int = 8
aqr_query_rounds: int = 2
aqr_sinkhorn_iters: int = 6
aqr_sinkhorn_temperature: float = 0.2
aqr_pg_grounding_enabled: bool = False
aqr_pg_entropy_threshold: float = 0.90
aqr_pg_peak_threshold: float = 1.50
aqr_pg_bias_weight: float = 0.0
aqr_temporal_memory_tokens: int = 32
aqr_consumer_mode: Literal["full"] = "full"
```

Deprecate for final path:

```python
mapg_enabled
vl_anchor_router_enabled as control-path driver
```

They may remain only for debug comparison.

### 10.2 State objects

Add:

```python
PicfTypedSupportMemoryState
PicfAnchorQueryRouterState
PicfAnchorRouteDiagnostics
```

The old `PicfAnchorPriorGraphState` can be kept only as a compatibility/export wrapper.

### 10.3 Core modules

Implement:

```python
_build_typed_support_memory(...)
_build_anchor_queries(...)
_route_queries_to_supports(...)
_sinkhorn_competition(...)
_pool_anchor_tokens(...)
_compute_anchor_geometry(...)
_build_aqr_anchor_state(...)
```

### 10.4 Consumer replacement

Replace current consumers:

```text
observation anchors:
  use physical query anchors

task readout:
  use task query anchors

posterior binding:
  use physical anchors only

conditioned control:
  consume task anchors + physical summary + posterior summary
```

Do not let task-conditioned anchors directly overwrite posterior memory.

### 10.5 Debug export

Export:

```text
all p_visual^k
all p_point^k
all p_tactile^k
all p_posterior^k
query ids / roles
pg confidence
Sinkhorn row entropy
Sinkhorn column usage
support overlap matrix
anchor x/S/geometry_valid
task keypoint heatmaps
overlay and rawheat PNGs
```

The file names must include:

```text
episode
step
goal text
query family
role
row
entropy
peak index
```

---

## 11. Direct Training Contract

There should be one final training run type:

```text
AQR-MAPG full
```

Do not run MAPG-v0 as a production intermediate.

Recommended command shape after implementation:

```bash
python scripts/picf_core_train.py \
  --picf-mode enabled \
  --semantic-mode paligemma \
  --mapg-enabled false \
  --vl-anchor-router-enabled false \
  --aqr-mapg-enabled true \
  --aqr-query-count-physical 16 \
  --aqr-query-count-task 8 \
  --aqr-query-rounds 2 \
  --aqr-sinkhorn-iters 6 \
  --aqr-sinkhorn-temperature 0.2 \
  --no-aqr-pg-grounding-enabled \
  --aqr-pg-bias-weight 0.0 \
  --aqr-pg-entropy-threshold 0.90 \
  --aqr-pg-peak-threshold 1.50 \
  --save-every 2500 \
  --steps 30000
```

Freezing policy:

```text
freeze V-JEPA pretrained backbone
freeze Sonata pretrained backbone
freeze AnyTouch pretrained backbone
train all link/projection/router/PICF/control adapters as before
```

This matches the desired final setup:

```text
foundation representations stable
router/adapters learn task-specific support routing
PI0.5 action path remains final action generator
```

---

## 12. Validation Harness

Although the deployment target is direct final, the implementation must include a validation harness.

This is not an intermediate architecture. It is an acceptance test.

Required eval:

```text
CALVIN eval 20
anchor overlay videos
dense JSON export
rawheat + overlay PNGs
step-level quant reports
```

Required checks:

```text
task_heatmap entropy lower than current near-uniform baseline
task_heatmap peak_to_uniform > configured threshold when confident
PaliGemma c_pg ~= 0 when near-uniform
role1_unique_centers median > 1 on multi-object frames
assignment row diversity > 1 for same-role slots
same-role support overlap below margin
anchor temporal jitter bounded
action loss not worse than no-MAPG baseline
CALVIN success not worse than no-MAPG baseline
```

Failure means:

```text
the final router implementation is wrong or insufficiently supervised
```

not:

```text
reactivate MAPG-v0
```

---

## 13. Backup Plan

Backup is operational, not architectural.

Production fallback:

```text
PICF/PI0.5 with AQR disabled
```

This is the safe fallback if final AQR-MAPG fails acceptance tests.

Do not fallback to:

```text
current MAPG-v0 full control
```

because evidence already shows it can act as noise.

Rollback rule:

```text
if role1 collapse persists
or PaliGemma weak prior still dominates when near-uniform
or CALVIN/action loss regresses,
then disable AQR and return to PICF baseline while fixing router internals.
```

---

## 14. Self-Critique of This Final Plan

### Risk 1: Query router could still collapse

Mitigation:

```text
row-specific query identity
support-level Sinkhorn competition
assignment row diversity metric
support overlap loss
hard negative cross-modal loss
```

### Risk 2: Keypoint supervision proxies can be noisy

Mitigation:

```text
separate effector / interaction / task supports
confidence masks
do not treat future keypose as object mask
multi-signal agreement required for strong labels
```

### Risk 3: PaliGemma weak prior could still inject noise

Mitigation:

```text
entropy gate
peak-to-uniform gate
temporal stability gate
support consistency gate
explicit c_pg debug export
```

### Risk 4: Point support may be missing or noisy

Mitigation:

```text
visual support remains first-class
geometry_valid explicitly tracked
point geometry auxiliary low weight
no hard dependency on x/S
```

### Risk 5: Full final implementation is larger than patching MAPG-v0

Mitigation:

```text
reuse existing tokens and PICF consumers
replace router primitive only
do not create a separate extractor
keep PI0.5 action path unchanged
```

### Risk 6: Direct final deployment could hide which component failed

Mitigation:

```text
full dense debug export
query-family-separated diagnostics
per-modality support export
acceptance metrics per layer
```

The plan is still preferable to incremental MAPG-v0 patching because it fixes the mathematical object that failed: row-specific support routing.

---

## 15. Literature Rationale

This plan is consistent with the following paper-level principles:

### Slot Attention

Slot Attention shows that object-centric slots specialize through competitive attention. The current MAPG-v0 violates that spirit because same-role rows receive identical evidence. AQR restores row-specific competitive binding.

Reference:

```text
https://arxiv.org/abs/2006.15055
```

### MESH / OT Slot Attention

MESH connects slot attention and optimal transport and improves tie-breaking through Sinkhorn/OT-style competition. AQR uses role-constrained Sinkhorn as a routing primitive, not a cosmetic loss.

Reference:

```text
https://arxiv.org/abs/2301.13197
```

### DETR

DETR supports the idea of learned queries plus matching to avoid duplicate predictions. AQR uses query identities and assignment competition for anchor separation.

Reference:

```text
https://arxiv.org/abs/2005.12872
```

### Perceiver IO

Perceiver IO supports latent/output queries reading large structured inputs with scalable cross-attention. AQR uses O(KN) query-to-memory reads instead of O(N^2) dense self-attention.

Reference:

```text
https://arxiv.org/abs/2107.14795
```

### TokenLearner

TokenLearner supports learning a small adaptive set of important tokens from dense visual inputs. AQR generalizes this idea from visual-only token selection to typed multimodal support routing.

Reference:

```text
https://arxiv.org/abs/2106.11297
```

### Transporter Networks

Transporter shows robotic manipulation can be formulated through spatial displacements / action points and benefits from explicit spatial structure. AQR explicitly supervises interaction supports instead of hoping VLM heatmaps emerge.

Reference:

```text
https://arxiv.org/abs/2010.14406
```

### CLIPort

CLIPort separates semantic what from spatial where. AQR follows this by making PaliGemma semantic/weak-prior only while typed support routing learns where.

Reference:

```text
https://arxiv.org/abs/2109.12098
```

### PerAct

PerAct frames manipulation as detecting the next best voxel action. AQR keeps the same principle: action-relevant spatial supports must be directly learned, not inferred from generic VLM heatmaps.

Reference:

```text
https://arxiv.org/abs/2209.05451
```

### BridgeVLA

BridgeVLA supports heatmap-style spatial action prediction aligned with VLM-compatible 2D inputs. AQR keeps heatmap/support distributions, but does not let weak PaliGemma heatmaps dominate without confidence.

Reference:

```text
https://arxiv.org/abs/2506.07961
```

### SpatialVLA / spatial VLA work

Spatial VLA work supports explicit spatial representation in VLA policies. AQR uses typed support memory and first-class visual/point/tactile/posterior supports to preserve spatial structure.

Reference:

```text
https://arxiv.org/abs/2501.15830
```

### SigLIP

SigLIP supports pairwise sigmoid contrastive objectives that are suitable for variable valid pairs. AQR uses validity-masked cross-modal pairwise matching.

Reference:

```text
https://arxiv.org/abs/2303.15343
```

### VICReg

VICReg provides explicit anti-collapse regularization through variance/covariance terms. AQR uses it per modality and on anchor tokens, but not as a replacement for routing competition.

Reference:

```text
https://arxiv.org/abs/2105.04906
```


---

## 17. Full Theoretical Audit: Is AQR-MAPG a Coherent Final Design?

### 17.1 Short answer

My current judgment is:

```text
+ AQR-MAPG is structurally coherent.
+ It is not a patch for only the observed cyan-point collapse.
+ It removes the broken MAPG-v0 primitive instead of preserving it as legacy baggage.
+ It has strong paper-level support at the principle level.
+ It is likely to be much more usable than MAPG-v0 for CALVIN/PICF-style manipulation.
+- It is not guaranteed to succeed without implementation and training validation.
+  The correct confidence level is strong engineering prior, not mathematical proof of success.
```

This distinction matters. The final architecture is not proven by one paper as a complete combination. The claim is that every major design object has a clear reason, a known precedent, and a direct role in fixing a general class of manipulation failures, not merely the one observed collapse.

### 17.2 Why this is not just a reaction to the observed bug

The observed bug was:

```text
same-role assignment row symmetry
```

A narrow fix would be:

```text
add random row offsets
add stronger diversity loss
add geometry repulsion
add a bigger coefficient
```

AQR-MAPG does not do that.

+The final design changes the primitive:
+
+```text
+from: candidate-prior graph + post-hoc same-role assignment
+to:   row-specific anchor queries reading typed support memory
+```
+
+This is a more general correction because it addresses all of the following failure modes:
+
+```text
+1. weak language heatmap
+2. missing pointcloud
+3. noisy pointcloud
+4. multiple similar objects
+5. tactile-only/contact-driven evidence
+6. posterior temporal identity
+7. same-role slot collapse
+8. support distribution overlap
+9. representation collapse
+10. task/physical memory contamination
+```
+
+The current evidence revealed one failure mode sharply, but the replacement is designed around the correct mathematical object for manipulation:
+
+```text
+anchor = query-conditioned support distribution over typed evidence
+```
+
+not:
+
+```text
+anchor = point selected from a weak heatmap
+```
+
+That is why this is not just a repair patch.
+
+### 17.3 Follow-through from data to architecture
+
+The data pipeline contains different evidence types:
+
+```text
+RGB images
+V-JEPA dense visual features
+PaliGemma semantic/text/image features
+point/Sonata features
+tactile/AnyTouch features
+posterior recurrent anchor state
+proprio/action context
+language task
+```
+
+These data are not naturally the same type. A point token has world geometry. A visual token has 2D location and view. A tactile token has contact semantics. A posterior token has identity and temporal persistence. A language token has task semantics but weak spatial precision.
+
+Therefore, the right common abstraction cannot be a single 3D point or a single VLM heatmap. The common abstraction must be:
+
+```text
+typed support memory
+```
+
+where each modality preserves its own support space:
+
+```text
+Omega_visual     = visual grid / view tokens
+Omega_point      = world point set
+Omega_tactile    = tactile/contact tokens
+Omega_posterior  = persistent anchor slots
+Omega_temporal   = compact memory tokens
+```
+
+An anchor is then a collection of distributions:
+
+```text
+p_visual^k     in Delta(Omega_visual)
+p_point^k      in Delta(Omega_point)
+p_tactile^k    in Delta(Omega_tactile)
+p_posterior^k  in Delta(Omega_posterior)
+p_temporal^k   in Delta(Omega_temporal)
+```
+
+This is mathematically cleaner than forcing all evidence into a point early.
+
+### 17.4 Follow-through from images to diagnosis
+
+The overlay evidence shows three distinct levels:
+
+```text
+1. VL heatmaps are visually diffuse and quantitatively near-uniform.
+2. MAPG graph priors can form visible local basins.
+3. observation role1 graph rows become identical after assignment.
+```
+
+This means the failure is not simply:
+
+```text
+the heatmap image looks wrong
+```
+
+It is:
+
+```text
+weak task evidence + non-task-specific basins + row-symmetric assignment
+```
+
+AQR-MAPG addresses each level:
+
+```text
+weak task evidence:
+  PaliGemma becomes confidence-gated weak prior; task where is supervised directly.
+
+non-task-specific basins:
+  anchor queries are task/role/proprio/posterior conditioned, not only prior-score driven.
+
+row-symmetric assignment:
+  each query carries row identity and reads typed memory through its own logits.
+```
+
+### 17.5 Mathematical contract
+
+Let each typed memory be:
+
+```text
+M_m = {(k_m^i, v_m^i, pos_m^i, mask_m^i, conf_m^i)}_{i=1}^{N_m}
+```
+
+where modality:
+
+```text
+m in {visual, point, tactile, posterior, temporal}
+```
+
+Let each anchor query be:
+
+```text
+q_k = f(role_k, query_id_k, coverage_k, task_or_physical_context, proprio, posterior_summary)
+```
+
+Routing logits:
+
+```text
+ell_m^{k,i} =
+  <Wq_m q_k, Wk_m k_m^i> / sqrt(d)
+  + b_pos_m(k,i)
+  + b_role_m(k,i)
+  + b_valid_m(i)
+  + b_prior_m(k,i)
+```
+
+Support distribution:
+
+```text
+p_m^k = Normalize_m(ell_m^{k,:})
+```
+
+where `Normalize_m` is softmax or role-constrained Sinkhorn depending on the support and query group.
+
+Pooled evidence:
+
+```text
+r_m^k = sum_i p_m^k[i] v_m^i
+```
+
+Modality reliability:
+
+```text
+beta_m^k = Gate(q_k, role_k, entropy(p_m^k), conf_m, valid_m)
+```
+
+Anchor token:
+
+```text
+z_k = sum_m beta_m^k r_m^k + e_role(k) + e_query_type(k) + e_conf(k) + e_geom(k)
+```
+
+Optional point geometry:
+
+```text
+x_k = sum_j p_point^k[j] X_j
+S_k = sum_j p_point^k[j] (X_j - x_k)(X_j - x_k)^T
+```
+
+If point support is invalid:
+
+```text
+geometry_valid_k = False
+```
+
+This remains a valid anchor because `z_k` and visual/tactile/posterior supports still exist.
+
+The important property is:
+
+```text
+q_k != q_l for different anchor rows
+```
+
+even if:
+
+```text
+role_k == role_l
+```
+
+Therefore, same-role rows do not have identical logits unless the model learns to make them identical despite regularization and competition.
+
+### 17.6 Why the design is minimal, not bloated
+
+AQR-MAPG has four necessary objects:
+
+```text
+1. typed memory
+2. anchor queries
+3. support distributions
+4. competition / matching
+```
+
+Remove typed memory, and missing-modality handling becomes ad hoc.
+
+Remove queries, and same-role slots again lack row identity.
+
+Remove support distributions, and anchors become opaque embeddings or brittle points.
+
+Remove competition, and multiple anchors can explain the same region.
+
+So these are not decorative components. They are the minimal set needed for a coherent multimodal anchor router.
+
+What is explicitly not retained:
+
+```text
+MAPG-v0 score-only same-role assignment
+PaliGemma heatmap as primary point source
+post-hoc geometry repulsion as main anti-collapse
+standalone keypoint extractor disconnected from PICF
+```
+
+That means the design is not preserving old waste products for sentimental reasons.
+
+### 17.7 Why not a standalone keypoint extractor?
+
+A standalone keypoint extractor can work as an engineering shortcut, but it creates a split objective:
+
+```text
+extractor predicts pretty points
+PICF/MAPG/control may or may not use them correctly
+```
+
+AQR-MAPG instead makes keypoint selection identical to support routing:
+
+```text
+the same support distributions used for losses are used for observation/task/posterior/control
+```
+
+This is more internally consistent.
+
+### 17.8 Why not just disable MAPG permanently?
+
+Disabling MAPG permanently is a safe baseline, but it gives up the anchor/support abstraction needed for:
+
+```text
+missing pointcloud
+tactile evidence
+posterior temporal identity
+multiple object-like supports
+task-conditioned readout
+```
+
+The correct response to MAPG-v0 failure is to replace the routing primitive, not delete the support abstraction.
+
+### 17.9 Why PaliGemma remains but cannot dominate
+
+The evidence shows that the PaliGemma heatmap can be near-uniform. Therefore, using it as the primary spatial source is unsafe.
+
+However, PaliGemma still supplies useful semantic conditioning:
+
+```text
+task text embedding
+image-language context
+weak visual prior when confident
+```
+
+The correct gate is:
+
+```text
+B_pg = lambda_pg * c_pg * centered_log(p_pg + eps)
+```
+
+where:
+
+```text
+c_pg -> 0 if entropy is high or peak ratio is low
+```
+
+This makes PaliGemma helpful when it is spatially informative and harmless when it is not.
+
+### 17.10 Paper support and what each paper actually justifies
+
+The literature support is not that one paper proves AQR-MAPG as a whole. The support is compositional and principle-level.
+
+```text
+Slot Attention:
+  supports the idea that object-like slots should specialize through competitive attention.
+  Relevance: anchor queries must compete for supports.
+
+MESH / OT Slot Attention:
+  supports Sinkhorn/OT competition and tie-breaking for object-centric slots.
+  Relevance: same-role assignment requires OT-style competition, not score-only softmax.
+
+DETR:
+  supports learned queries plus matching to avoid duplicate object predictions.
+  Relevance: anchor queries need row identity and assignment/matching.
+
+Perceiver IO:
+  supports queries reading large structured inputs through scalable cross-attention.
+  Relevance: typed support memory can be read in O(KN).
+
+TokenLearner:
+  supports adaptive selection of a small number of informative tokens from dense inputs.
+  Relevance: small anchor queries can summarize dense visual/point supports.
+
+Transporter Networks:
+  supports explicit spatial action-point modeling in manipulation.
+  Relevance: where must be directly learned.
+
+CLIPort:
+  supports separating semantic what from spatial where.
+  Relevance: PaliGemma should condition semantics, not serve as spatial oracle.
+
+PerAct:
+  supports action/keypose prediction over explicit spatial supports.
+  Relevance: action-relevant supports should be supervised.
+
+BridgeVLA:
+  supports heatmap-style VLM-compatible spatial alignment.
+  Relevance: heatmaps/support distributions are useful, but must be reliable and action-side.
+
+SigLIP:
+  supports pairwise cross-modal matching with variable pairs.
+  Relevance: same-anchor cross-modal supports can be aligned under missing modalities.
+
+VICReg:
+  supports anti-collapse via variance/covariance regularization.
+  Relevance: embeddings need anti-collapse, but this does not replace routing competition.
+```
+
+This is strong support for the design principles, not a guarantee of final empirical success.
+
+### 17.11 Failure modes the final design anticipates
+
+```text
+PaliGemma uniform heatmap:
+  confidence gate turns it off.
+
+Pointcloud missing:
+  visual/tactile/posterior supports remain valid; geometry_valid=False.
+
+Pointcloud noisy:
+  point gate lowers beta_point; geometry auxiliary stays low weight.
+
+Multiple similar objects:
+  queries have identity; Sinkhorn and hard negatives prevent duplicate collapse.
+
+Tactile-only contact evidence:
+  tactile memory is a typed support source, not forced into a brittle point.
+
+Posterior temporal drift:
+  posterior support is queried but not blindly overwritten by task text.
+
+Anchor embedding collapse:
+  SigLIP/VICReg act on per-modality pooled embeddings and anchor tokens.
+
+Support overlap collapse:
+  kernelized normalized support overlap penalizes same-role active overlap.
+```
+
+### 17.12 What would falsify the plan?
+
+The plan should be considered wrong or incomplete if, after correct implementation:
+
+```text
+1. task supports remain near-uniform despite direct supervision
+2. same-role query supports remain identical
+3. PaliGemma confidence gate fails to suppress weak heatmaps
+4. query supports look good but action loss/eval regresses
+5. temporal memory increases jitter instead of reducing it
+6. point-missing cases break visual/tactile routing
+```
+
+If these happen, the fallback is not MAPG-v0. The fallback is PICF/PI0.5 baseline plus targeted AQR debugging.
+
+### 17.13 Final confidence statement
+
+I would state the confidence as:
+
+```text
+High confidence that AQR-MAPG is the right structural replacement for MAPG-v0.
+Moderate-to-high confidence that it will be trainable in this codebase because all required typed tokens already exist.
+No claim of guaranteed success before implementation and CALVIN/action-loss validation.
+```
+
+This is the most defensible position. It avoids two bad extremes:
+
+```text
+overclaiming that architecture guarantees success
+underreacting by only patching the observed collapse
+```
+
+The reason it is the right final scheme is that it follows the data all the way through:
+
+```text
+typed evidence -> row-specific queries -> support distributions -> competition -> anchor tokens -> PICF consumers -> PI0.5 action
+```
+
+That is mathematically and operationally coherent.

---

## 16. Final Decision

Deploy this, not MAPG-v0:

```text
AQR-MAPG full final:
Typed Support Memory
+ Task/Role-Conditioned Anchor Queries
+ Query-to-Support Cross-Attention
+ Role-Constrained Sinkhorn / OT Competition
+ Explicit Interaction/Task Support Supervision
+ Confidence-Gated PaliGemma Weak Prior
+ PI0.5 Action Path Unchanged
```

This is the final coherent route because:

```text
it fixes same-role assignment symmetry
it does not rely on weak PaliGemma heatmaps
it preserves MAPG's useful anchor/support abstraction
it avoids a disconnected keypoint extractor
it scales through O(KN) query reads
it has a clear baseline fallback
```

Current MAPG-v0 should remain only as an evidence/debug comparison path.

---

## 18. 2026-05-09 Live Code Deployment Notes

The live code now exposes AQR-MAPG as the direct-final graph path:

```text
--aqr-mapg-enabled true
--mapg-enabled false
--vl-anchor-router-enabled false
```

Implemented live pieces:

```text
1. AQR query identities:
   learned physical queries + learned task queries
   + role embeddings
   + query-type embeddings
   + deterministic low-discrepancy coverage codes

2. Typed support reads:
   visual V-JEPA/PICF visual support
   point/Sonata support
   tactile/AnyTouch support
   posterior support

3. PaliGemma placement:
   PaliGemma semantic tokens condition task queries
   the PaliGemma heatmap/grounding head is off by default
   heatmaps are only generated when --aqr-pg-grounding-enabled is explicit
   heatmaps influence AQR only when --aqr-pg-bias-weight > 0
   physical queries do not get direct task heatmap overwrite

4. Support competition:
   attention weights are passed through support-level Sinkhorn-style
   row/column normalization before becoming graph priors

5. Downstream consumer compatibility:
   AQR emits the existing PicfAnchorPriorGraphState contract
   observation anchors, task readout, posterior binding, and conditioned
   control consume the graph through the same audited code path

6. Row-specific downstream assignment:
   graph slot assignment now uses slot token similarity plus direct
   point/visual support overlap in addition to role masks, anchor quality,
   confidence, and Sinkhorn capacity constraints
```

Important non-goals:

```text
1. This is not a standalone keypoint extractor.
2. This does not restore MAPG-v0 as production graph construction.
3. This does not let PaliGemma heatmaps dominate anchor placement; by default
   they are not computed and do not affect anchor placement.
4. This does not alter the final PI0.5 action generator path.
```

Current local verification:

```text
python -m py_compile \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/training.py \
  scripts/picf_core_train.py \
  scripts/serve_picf_policy.py
```

Result:

```text
passed
```

Additional 2026-05-09 code audit result:

```text
AQR is now included in all point-optional graph fallback guards.
The training startup log now prints an explicit AQR-MAPG contract line in
addition to the legacy MAPG-disabled line, so cloud audit does not have to
infer AQR state only from the raw command line.

Correct guard:
  graph_can_run_without_points = mapg_enabled or aqr_mapg_enabled

Covered sites:
  first-step missing point-cloud hard error
  first-step empty local point support hard error
  runtime hold reason for point-contract violation
```

This matters because the direct-final AQR contract is visual/support-first and
point-optional. Leaving those guards legacy-`mapg_enabled`-only would be a real
contract bug, not a cosmetic issue.

Live deployment profile:

```text
--aqr-mapg-enabled true
--mapg-enabled false
--vl-anchor-router-enabled false
--picf-mode enabled
--perception-finetune-mode frozen
--num-steps 30000
--save-interval 2500
--unroll-steps 2
```

Trainability contract:

```text
Frozen:
  V-JEPA visual backbone
  Sonata point backbone / pretrained point encoder
  AnyTouch tactile backbone / pretrained tactile encoder

Trainable:
  AQR learned physical/task queries
  AQR support readers and query self-attention
  AQR/graph consumer projections and gates
  PICF observation/task/posterior/control heads
  PaliGemma semantic path under the normal semantic trainable profile
  PI0.5 action-side trainable path
```

Historical `mapg_*` loss and assignment flag names are still reused by the
trainer for shared graph-loss plumbing. They do not enable MAPG-v0 when
`--mapg-enabled false` and `--aqr-mapg-enabled true`.

Behavioral acceptance is deliberately not certified by a one-step smoke run.
The first meaningful acceptance point is the `5000`-step checkpoint:

```text
1. run CALVIN evaluation for 20 sequences
2. export anchor health videos
3. export raw PaliGemma/AQR heatmap images
4. export transparent heatmap-over-RGB overlays
5. analyze JSON statistics for support entropy, same-role overlap,
   assignment effective anchors, geometry validity, action loss, and MAPG/AQR
   loss components
```

Until those artifacts exist, the correct statement is:

```text
The final AQR implementation is deployed and smoke-verified.
The final AQR behavior is pending checkpoint/evaluation evidence.
```

Local `scripts/picf_core_train.py --help` can currently be blocked by a
local IPython / wandb vendored pygments import assertion unrelated to AQR
syntax. Cloud training environments used for previous PICF runs should still
use the normal training entrypoint.

## 19. Current Finality Boundary

This README intentionally separates the final deployed contract from broader
research ideas:

```text
In-contract for this deployment:
  typed visual / point / tactile / posterior support memory
  learned physical and task anchor queries
  query-to-support cross-attention
  support-level Sinkhorn competition
  confidence-gated weak PaliGemma task bias
  row-specific downstream slot assignment
  point-optional graph runtime
  graph losses over matching, VICReg, cycle, masked modality, routing,
  support diversity, and geometry diversity
  unchanged PI0.5 action generator path

Out-of-contract for this deployment:
  standalone keypoint extractor
  MAPG-v0 candidate-prior graph as production path
  raw point/tactile/posterior tokens inside PaliGemma
  unbounded dense temporal KV cache
```

The live temporal support source is the existing posterior anchor memory. A
separate large temporal KV cache is intentionally not part of the final
training profile because it would increase memory/latency and reintroduce an
unclear ownership boundary. This is not a cut-down of AQR; it is the explicit
final boundary for the current PICF v2.2 deployment.
