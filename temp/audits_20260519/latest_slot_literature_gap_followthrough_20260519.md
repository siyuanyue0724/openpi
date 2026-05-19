# Latest Slot / Tactile Object-Binding Follow-Through

Date: 2026-05-19

Scope: compare current PICF-AQR-OWM v2.2 object-slot implementation against
recent object-centric, adaptive-slot, object-binding, and visuo-tactile robotics
methods, then decide what is still missing and what should not be copied into
PICF.

This is not a behavior acceptance report.  It is a code/dataflow/math audit
for the current implementation and the active A7 conditional slot initializer
probe.

## Sources Checked

Primary papers / repos used for this audit:

```text
MetaSlot, arXiv:2505.20772
  claim used here:
    fixed slot counts can split/duplicate objects;
    robust OCL benefits from VQ prototype codebooks and aggregate/deduplicate.
  local code:
    temp/paper_code_20260518/MetaSlot

QASA, arXiv:2601.12936
  claim used here:
    adaptive K and slot-quality selection should be decoupled from forcing every
    fixed slot to reconstruct/action-train.

Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?,
arXiv:2510.24709
  claim used here:
    object binding is a pairwise/quadratic low-dimensional relation, not
    guaranteed by raw hidden cosine.

SlotVLA, arXiv:2511.06754
  claim used here:
    robot VLA benefits from task-relevant object slots and relation-centric
    decoding; clean boxes/masks/tracks materially simplify supervision.

OA-WAM, arXiv:2605.06481
  claim used here:
    persistent address and time-varying content should be separated for object
    identity, but wholesale address-only transformer replacement is not a
    drop-in PICF patch.

OmniVTA / tactile-object work, 2026
  claim used here:
    tactile/contact should update the contacted object state, not become a
    separate gripper-owned object.

Implementation pattern repos:
  temp/paper_code_20260518/slot-attention-video
  temp/paper_code_20260518/AdaSlot
  temp/paper_code_20260518/object-centric-learning-framework
  temp/paper_code/slotcontrast
  temp/external_repos/SlotLifter
```

## Current PICF Dataflow Being Audited

The maintained object-binding chain is:

```text
RGB/static+wrist, V-JEPA, point, tactile, tracklet, sidecar proposals
  -> typed token field
  -> AQR graph object/contact rows
  -> object candidate assignment with background residual
  -> proposal-anchor seed before point attention
  -> graph object owner / contact bridge support
  -> posterior owner transport precision fusion
  -> active posterior object file
  -> action prefix / diagnostics
```

The current A7 probe intentionally isolates this chain:

```text
run:
  picf_a7_conditional_slot_initializer_anchor1000_20260519

scope:
  picf_trainable_scope = anchor_only
  perception_finetune_mode = frozen
  semantic/PaliGemma effectively frozen
  action loss = 0
  auxiliary losses = 0 except lambda_anchor_object_pull = 1
  role layout = object_only
  effector role 0 = disabled
  proposal/sidecar/tracklet evidence = enabled
```

This makes the probe valid for one question only:

```text
Can the object-slot path move a role-1 object owner toward accepted
sidecar/contact object evidence without help from action or semantic gradients?
```

It is not a formal action-training run.

## What Is Already Implemented

### 1. Slot competition and background/no-object residual

Mature slot systems use slot-normalized competition and explicit background or
empty capacity.  PICF does not copy a reconstruction decoder, but it implements
the belief-filter analogue:

```text
Code:
  src/openpi/picf/core/pipeline.py::_aqr_active_slot_mask
  src/openpi/picf/core/pipeline.py::_aqr_downstream_slot_weights
  src/openpi/picf/core/pipeline.py::_posterior_file_competition
  src/openpi/picf/core/pipeline.py::_posterior_birth_competition

Runtime fields:
  aqr_active_anchor_count
  aqr_active_same_role_support_overlap_max
  posterior_file_competition_active_count
  posterior_file_competition_duplicate_overlap_max
```

Verdict:

```text
Implemented in PICF-native form.
Not identical to SlotAttention reconstruction, but mathematically coherent for
a typed belief filter.
```

### 2. Sidecar/object candidate assignment with background

Modern slot methods do not treat noisy masks as truth; they use masks or boxes
as responsibility priors.  PICF does this through candidate assignment:

```text
Code:
  src/openpi/picf/core/pipeline.py::_proposal_object_candidate_assignment
  src/openpi/picf/core/pipeline.py::_object_candidate_physical_rows

Key contract:
  sidecar candidate -> eligible object row or background
  invalid fragments -> background residual
  role 0 effector -> excluded from object ownership
```

Verdict:

```text
Implemented.
Correctly rejects blind SAM as production evidence.
```

### 3. Conditional object-slot initialization before dense attention

The current failure mode was late supervision trying to pull a slot after it had
already read the wrong dense point region.  The deployed fix conditions the
slot before point attention:

```text
Code:
  src/openpi/picf/core/pipeline.py::_apply_proposal_anchor_seed_to_query_and_point_bias

Math:
  q_j <- q_j + lambda_q r_j(z_proposal - q_j)
  b_{j,i}^{point} <- b_{j,i}^{point} + lambda_seed r_j log rho_{j,i}
```

Verdict:

```text
Implemented.
This is the PICF equivalent of conditional slot/mask initialization, not a
late loss patch.
```

### 4. Pairwise/quadratic binding subspace

The object-binding paper argues that same-object identity should be read as a
pairwise relation, not raw token cosine.  PICF implements:

```text
Code:
  binding_signature_proj
  posterior_binding_signature_memory
  calibrated linear / diagonal quadratic / low-rank scores

Runtime fields:
  posterior_binding_signature_* metrics
  aqr_same_role_*_binding_signature_overlap_*
```

Verdict:

```text
Implemented structurally.
Still needs latest-artifact offline IsSameObject acceptance before claiming
the learned subspace is behaviorally solved.
```

### 5. Tactile attaches to object owner

Tactile/contact should be evidence about the contacted object, not an
independent gripper object.  PICF now supports:

```text
Code:
  src/openpi/picf/core/pipeline.py::_fused_read_role_bias
  src/openpi/picf/core/pipeline.py::_mapg_tactile_seed_priors

Config:
  tactile_attach_to_object_owner = True
```

Verdict:

```text
Implemented.
In object-only probes, role 0 is removed and tactile routes to role 1 owner.
```

### 6. Posterior owner responsibility write-through

Earlier probes showed graph anchors could find the object while posterior files
stayed elsewhere.  PICF now transports owner responsibility into posterior
geometry:

```text
Code:
  src/openpi/picf/core/pipeline.py::_posterior_owner_transport_measurement

Math:
  R_obs = A_obs<-graph R_graph
  R_post = B_post<-obs R_obs

  Lambda+ = S_std^{-1} + k c_owner S_owner^{-1}
  x+ = Lambda+^{-1}(S_std^{-1}x_std + k c_owner S_owner^{-1}x_owner)
```

Verdict:

```text
Implemented.
This is the correct belief-filter closure operator.
Current A7 metrics show it is active but not yet strong/stable enough to
declare posterior binding solved.
```

## Current A7 Probe Status

Latest one-shot remote check reached step 300 and is still running.

Compact reading:

```text
GPU:
  two GPUs active

sidecar/tracklet dataflow:
  owm_proposal_tokens ~= 1.5
  owm_tracklet_tokens ~= 80
  proposal_shape_quality_mean ~= 0.90-0.94
  object_candidate_coverage_mean ~= 0.92-0.95
  object_candidate_background_mean ~= 0.02-0.04

effector leakage:
  aqr_active_anchor_count_role_0 = 0.0
  role 0 is not the object owner in this probe

active slot duplication:
  aqr_active_same_role_support_overlap_max ~= 0.03 at 100/150,
  ~= 0.09 at 300
  active slots are not in the old hard collapse regime.

raw/inactive duplication:
  aqr_same_role_support_overlap_max can still approach 1.0.
  This is mostly reserve/inactive telemetry under the current active-file
  contract, not by itself a failure.

posterior owner transport:
  posterior_owner_transport_confidence_max ~= 0.15-0.25
  posterior_owner_transport_mass_max ~= 0.40-0.87
  posterior_owner_transport_dist_to_standard_mean ~= 0.40-0.55

main probe loss:
  loss_anchor_object_pull fluctuates:
    step 100 ~= 0.485
    step 200 ~= 0.485
    step 300 ~= 1.143
```

Interpretation:

```text
Solved relative to older failures:
  sidecar evidence reaches the graph;
  effector/blue role is excluded in object-only probe;
  active same-role object files are not all collapsing;
  tracklet/proposal fields are nonzero in this run;
  posterior owner transport is active.

Still not solved:
  posterior owner confidence is modest;
  object-pull loss is not monotonically falling;
  overlay inspection remains necessary because metrics alone do not guarantee
  the active role-1 file sits on the intended object mask.
```

Strict conclusion:

```text
Do not start a 30000-step production run from this probe alone.
Use it to decide whether the conditional initializer is directionally useful.
If overlays at 100/200/300 show orange owner still off-mask, the next repair is
quality/adaptive object-file selection, not more scalar loss weights.
```

## Remaining Gaps Versus Latest Slot Literature

### Gap A: learned adaptive slot-quality selector

Recent adaptive slot work uses learned slot quality / K-adaptive selection
instead of hand-fixed active rows.  This was the most important gap identified
before the slot-quality update below.

Status after the update:

```text
PicfSlotQualityState:
  implemented for object-quality, no-object probability, duplicate probability,
  sidecar/tracklet/contact/proposal quality, active/context/reserve graph
  gating, action-visible downstream weights, object-explanation quality, and an
  optional entropy-corrected weak loss.
```

Remaining validation:

```text
Run the 1000-step quick comprehensive slot probe and inspect whether role-1
posterior centers move onto the sidecar/contact object.  The mechanism is
deployed; behavior still needs runtime evidence.
```

### Gap B: prototype/codebook initializer

MetaSlot uses VQ prototypes and aggregate/deduplicate.  PICF currently uses
learned role queries plus sidecar conditional seeding, not a global prototype
dictionary.

Why not blindly copy:

```text
MetaSlot's VQ codebook is image-reconstruction/OCL oriented and requires
pretraining.  PICF must handle missing modalities and action-conditioned belief
states.  A global VQ codebook over all objects may harm cross-dataset scaling
if it encodes CALVIN-specific object appearance.
```

PICF-compatible version:

```text
Use prototype initializers only as optional priors:
  sidecar/contact prototype
  tracklet prototype
  language/task prototype
  posterior memory prototype

Do not replace dense typed memory or posterior belief files.
```

Should implement now:

```text
Not before the current conditional initializer acceptance.
If needed, implement as a small prototype bank for object-row initialization,
not as a full VQ-VAE reconstruction pipeline.
```

### Gap C: full dense object reconstruction / explanation objective

PICF has OEML but its losses are guarded/default-zero in current probes.

Why this matters:

```text
Slot methods often learn because every object slot must explain pixels/features
and a background slot explains the rest.  PICF's anchor-only probe only tests
object pull, so it may not provide enough pressure for clean dense ownership.
```

Why not enable blindly:

```text
Earlier runs showed too many auxiliary losses can fight action and identity.
Noisy sidecar masks should not become hard reconstruction labels.
```

PICF-compatible version:

```text
Enable object explanation only after sidecar/tracklet quality gates pass:
  feature compactness
  point compactness
  contact explanation
  duplicate penalty
  background residual

Keep all weights small and staged.
```

Should implement now:

```text
The code exists.  Do not turn it on in the current anchor-only pull probe unless
we explicitly run an object-explanation ablation.
```

### Gap D: latest-artifact offline IsSameObject probe

The code has structural binding signatures and overlay JSON can dump them, but
the latest run has not yet been accepted by an offline same/different object
probe.

Required acceptance:

```text
Positive pairs:
  same sidecar candidate
  same tracklet id
  same posterior active owner across adjacent frames
  same contact neighborhood when contact is high

Negative pairs:
  different sidecar candidates
  spatially separated tracklets/point clusters
  active owner vs background/reserve

Probe:
  score(i,k) = z_i^T W z_k or diagonal/low-rank quadratic
  report AUC and per-modality ablations
```

Should implement now:

```text
Yes as an audit, not as a training loss.
It will answer whether binding_signature is meaningful on current artifacts.
```

### Gap E: full sidecar/tracklet coverage manifest

Current A7 run shows proposal and tracklet tokens are nonzero, but production
coverage over the full training set is not yet a solved fact.

Required manifest:

```text
per segment:
  sidecar frame count
  tracklet count
  proposal quality histogram
  contact confidence histogram
  object-center stability
  failure flags
```

Should implement:

```text
Yes, before long production training that relies on sidecar/tracklet evidence.
```

## What Should Not Be Added

### Blind SAM

Status:

```text
Rejected and archived.
```

Reason:

```text
CALVIN trials showed generic SAM fragments walls, drawer sides, robot
protrusions, and visual-change regions.  This is incompatible with object-owner
belief updates.  The proposal schema remains valid only for inspected
contact/task/tracklet-aware sidecars.
```

### OA-WAM address-only transformer replacement

Status:

```text
Do not transplant directly.
```

Reason:

```text
PICF already has posterior address/content and typed evidence routers.
Replacing all attention layers with address-only resets would be a new model,
not a repair.  Keep address as gated identity prior inside the belief filter.
```

### Hard sidecar labels

Status:

```text
Reject.
```

Reason:

```text
Contact-motion masks can be noisy, partial, or include gripper artifacts.
They should define a responsibility distribution and covariance, not a hard
ground-truth object label.
```

## 2026-05-19 Update: Adaptive Slot-Quality Selector

The remaining MetaSlot/QASA maturity gap was narrowed by adding a PICF-native
adaptive selector rather than transplanting a reconstruction decoder that would
break the belief-state contract.

Implemented code:

```text
src/openpi/picf/core/contracts.py:
  PicfSlotQualityState
  PicfAnchorPriorGraphState.slot_quality

src/openpi/picf/core/config.py:
  aqr_slot_quality_enabled
  aqr_slot_quality_learned_enabled
  aqr_slot_quality_learned_scale
  aqr_slot_quality_floor
  aqr_slot_quality_context_scale
  aqr_slot_quality_duplicate_threshold
  aqr_slot_quality_target_smoothing

src/openpi/picf/core/pipeline.py:
  _build_slot_quality_state(...)
  _slot_duplicate_risk(...)
  _row_strength_vector(...)
  _aqr_active_slot_mask(..., slot_quality=...)
  _aqr_downstream_slot_weights(..., slot_quality=...)
  object_explanation quality *= slot_quality.active_weight

src/openpi/picf/core/training.py:
  lambda_slot_quality
  _slot_quality_loss(...)

scripts/picf_core_train.py:
  --lambda-slot-quality

scripts/picf_object_candidate_slot_binding_audit.py:
  slot-quality contract/runtime/loss checks

Maintained validation launcher:

```text
run_a7_slot_quality_selector_anchor1000_20260519.sh
```

This launcher is deliberately not a behavior run.  It freezes or zeroes
PaliGemma/action/pretrained perception pressure, keeps role 0 effector rows
disabled, uses inspected contact-motion sidecars plus proposal/tracklet typed
evidence, enables `lambda_slot_quality = 0.10`, dumps overlays every 50 steps,
and asks only whether the object-slot machinery can bind to the sidecar object.
```

Mathematical form:

```math
e_j =
\max(
  A^{cand}_{j,*},
  A^{owner}_{j,*},
  P^{owner-point}_{j,*},
  P^{seed}_{j,*},
  P^{task-owner}_{j,*},
  P^{proposal}_{j,*},
  P^{tracklet}_{j,*},
  P^{point}_{j,*},
  P^{tactile}_{j,*}
)
```

```math
d_j =
\max_{k\ne j}
\left[
  O^{visual}_{j,k},
  O^{point}_{j,k},
  O^{temporal}_{j,k},
  O^{proposal}_{j,k},
  \exp(-\|x_j-x_k\|^2/2\sigma^2),
  O^{candidate}_{j,k}
\right]
```

```math
y^{obj}_j = \operatorname{clip}(e_j(1-d_j)),
\quad
y^{none}_j = \operatorname{clip}(1-e_j),
\quad
y^{dup}_j = \operatorname{clip}(\max(d_j,\mathbf{1}[d_j>\tau]))
```

The learned head is a residual around these measurement targets:

```math
\ell_j =
\operatorname{logit}
\begin{bmatrix}
y^{obj}_j\\
y^{none}_j\\
y^{dup}_j
\end{bmatrix}
+
\lambda_q h_\theta([q_j,e_j,d_j,s_j,c_j])
```

and the downstream object-file gate is:

```math
w^{active}_j =
\sigma(\ell^{obj}_j)
(1-\sigma(\ell^{none}_j))
(1-\sigma(\ell^{dup}_j))
```

implemented in code as:

```text
active_weight =
  object_quality * (1 - no_object_prob) * (1 - duplicate_prob)
```

The training loss is entropy-corrected BCE:

```math
L_q =
\sum_j w_j
\left[
\operatorname{BCEWithLogits}(\ell_j,y_j)
-
H(y_j)
\right]_+
/
\sum_j w_j
```

so the zero-initialized head at the deterministic target contributes near-zero
loss and does not distort early loss comparisons.

This is aligned with QASA/MetaSlot at the level PICF can safely adopt:

```text
QASA principle:
  select quality slots instead of forcing every fixed row to be an object.

MetaSlot principle:
  aggregate/deduplicate redundant object parts and avoid splitting one object
  across many fixed slots.

PICF-native implementation:
  keep dense typed memory and posterior authority, but each row predicts
  object/no-object/duplicate quality and gates active/context/reserve use.
```

What is deliberately not copied:

```text
1. Full reconstruction decoder:
   PICF is a belief router for PI0.5, not an autoencoding OCL model.

2. VQ prototype codebook:
   Useful for object images, but risky for heterogeneous missing-modality
   datasets unless proven by a separate ablation.

3. Hard SAM masks:
   Rejected by CALVIN diagnostics as noisy, object-irrelevant fragments.
```

## Immediate Decision

Current implementation is not a shallow patch.  It captures the PICF-compatible
core of recent slot/OCL literature:

```text
conditional slot initialization
slot/object competition
background residual
active/no-object capacity
learned object/no-object/duplicate slot quality
pairwise binding subspace
posterior owner write-through
tactile-to-object owner routing
tracklet/proposal typed memory
```

The main remaining maturity gap is now artifact-level validation:

```text
1. Run the new 1000-step quick comprehensive slot probe.
2. Inspect overlay/GIF views: mask-only, mask+anchor, active-only, no-gray,
   with-gray, and debug JSON.
3. Run latest-artifact IsSameObject probe from overlay JSON/signatures.
4. Only after active role-1 posterior centers land on the object should a
   30000-step production run be started.
```

Do not start a formal 30000-step production run until step-100/200/300 overlays
show the active role-1 posterior file is actually on the task object in several
representative segments.

## Current Acceptance Gates

Pass criteria for the active A7 probe:

```text
Dataflow:
  owm_proposal_tokens > 0
  owm_tracklet_tokens > 0
  aqr_proposal_anchor_seed_row_count > 0
  aqr_object_candidate_coverage_mean > 0.85
  aqr_object_candidate_background_mean < 0.10

Role leakage:
  aqr_active_anchor_count_role_0 = 0

Active duplication:
  aqr_active_same_role_support_overlap_max < 0.20
  aqr_active_same_role_object_core_overlap_max < 0.20

Posterior ownership:
  posterior_owner_transport_applied_fraction > 0
  posterior_owner_transport_confidence_max > 0.20 preferred
  overlay active role-1 posterior center inside or near sidecar mask

Loss:
  loss_anchor_object_pull should not trend upward over 50/100/200/300.
```

Current status at step 300:

```text
Dataflow: pass.
Role leakage: pass.
Active duplication: mostly pass.
Posterior ownership: partial.
Loss trend: not pass yet; object-pull fluctuates and rose at step 300.
```

Therefore the current run is informative but not final.
