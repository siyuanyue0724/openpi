# PICF-AQR-OWM Latest Slot Final Audit

Date: 2026-05-20

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

Status:

```text
code/static latest-slot audit complete;
short action-aware smoke passed;
behavior acceptance still requires longer co-training plus CALVIN/video evidence.
```

## 1. Decision

The current implementation is not a minimal patch over a weak slot scaffold.
It has imported the PICF-compatible mathematical invariants from recent
object-centric, object-binding and visuo-tactile slot work:

```text
implemented as production path:
  slot-axis evidence competition
  object/background residual explanation
  QASA-style object/no-object/duplicate quality
  active/context/reserve file gating
  active-context duplicate demotion
  MetaSlot-style duplicate/dynamic-count principle without hard VQ truth
  Object-Binding-style pairwise/quadratic same-object subspace
  gated persistent posterior binding-signature memory
  contact-motion sidecar masks as soft measurements
  object-owner geometry transport into graph and posterior files
  tactile/contact routing to object owner rather than gripper owner
  context graph attention bias instead of destructive token pruning/scaling
```

The current implementation intentionally does not copy these mechanisms:

```text
rejected for current production path:
  blind SAM proposals
  hard visual VQ posterior truth
  full RGB/frozen-feature reconstruction decoder as action-time truth
  NeRF/multi-view SlotLifter renderer inside CALVIN training
  online IsSameObject supervised loss without same-object labels
  predictive slot losses before identity has long-run evidence
```

Strict conclusion:

```text
No new mandatory slot module was found in the latest paper-code audit.

The next useful gate is not another local raw-overlap patch.  It is longer
co-training with the same slot contract, then CALVIN/video evidence.
```

This does not mean final behavior is solved.  It means the code path is no
longer missing a known, mathematically essential slot mechanism that should be
added before the next training gate.

## 2. PICF Contract

PICF is a belief-state router, not a reconstruction-first object autoencoder.
The posterior file is:

```math
F_{t,j}
=
(
x_{t,j},
S_{t,j},
a_{t,j},
c_{t,j},
\alpha_{t,j},
r_j,
\sigma_{t,j},
\phi_{t,j}
)
```

where:

```text
x,S:
  3D geometry and covariance.

a,c:
  persistent address and current content.

alpha:
  existence / support confidence.

r:
  file role.

sigma:
  typed support over visual, temporal, point, tactile, tracklet, proposal,
  PaliGemma and sidecar evidence.

phi:
  pairwise binding signature for same-object compatibility.
```

The update remains:

```math
p(F_t \mid z_{\le t}, a_{<t}, \ell)
\propto
p(z_t \mid F_t,\ell)
\int
p(F_t \mid F_{t-1},a_{t-1})
p(F_{t-1}\mid z_{\le t-1})
dF_{t-1}.
```

Therefore an imported slot mechanism is legal only if it enters one of these
places:

```text
1. measurement evidence p(z_t | F_t, language)
2. posterior file assignment / correction / transition
3. training-only guarded auxiliary
```

Mechanisms that bypass posterior authority, assume always-present visual masks,
or convert PICF into an image reconstruction objective are not production
modules for this architecture.

## 3. Paper-Code Sources Audited

Local code snapshots:

```text
temp/paper_code_20260518/object-centric-learning-framework
temp/paper_code_20260518/slot-attention-video
temp/paper_code_20260518/AdaSlot
temp/paper_code_20260518/MetaSlot
temp/paper_code_20260518/DINO
temp/paper_code_20260518/Deformable-DETR
temp/paper_code_20260518/slotcontrast
temp/external_repos/SlotLifter
temp/external_repos/slot_refs_20260520/vit-object-binding
```

Recent papers used as design evidence:

```text
MetaSlot, 2025:
  adaptive object count and duplicate handling.

Object Binding in pretrained ViTs, NeurIPS 2025:
  IsSameObject is a pairwise/quadratic low-dimensional subspace, not just
  hidden cosine.

SlotVLA, 2025:
  object/relation slots are useful for manipulation, but must be stable before
  action decoding pressure is trusted.

STORM, 2026:
  task-aware slots should adapt around frozen foundation features, not replace
  them with a reconstruction-only perception stack.

QASA, 2026:
  slot quality and adaptive K are first-class variables.

OCRA / visuo-tactile object-centric work, 2025-2026:
  tactile/contact evidence should bind to the manipulated object, not to a
  permanent gripper-only owner.
```

Public source pointers used for the 2026-05-20 re-audit:

```text
MetaSlot:
  https://arxiv.org/abs/2505.20772

Object Binding in pretrained ViTs:
  https://arxiv.org/abs/2510.24709
  https://neurips.cc/virtual/2025/poster/119887

SlotVLA:
  https://arxiv.org/abs/2511.06754

QASA:
  https://arxiv.org/abs/2601.12936
```

## 4. Mechanism Mapping

### 4.1 Slot Attention / OCL / SAVi

External invariant:

```math
a_{j,i}
=
\operatorname{softmax}_{j}(q_j^\top k_i),
\qquad
u_j
=
\sum_i
\frac{a_{j,i}}{\sum_i a_{j,i}+\epsilon}
v_i .
```

PICF deployment:

```text
_aqr_same_role_support_competition
_proposal_object_candidate_assignment
object_candidate_background
object_candidate_row_capacity
aqr_context_slot_self_support_overlap_enabled
aqr_context_slot_active_support_overlap_enabled
```

Audit decision:

```text
covered.

PICF does not need to replace AQR with OCL SlotAttention.  It needs the slot
competition invariant under typed evidence and posterior authority, which is
now present.
```

### 4.2 MetaSlot

External invariant:

```text
fixed K is insufficient;
duplicates should be demoted or masked;
object count should be adaptive.
```

PICF deployment:

```text
PicfSlotQualityState(object_quality, no_object_prob, duplicate_prob)
slot_quality.active_weight
slot_quality.context_weight
_slot_duplicate_risk
posterior_file_competition
posterior_birth_competition
active/context/reserve file state
```

Rejected part:

```text
hard visual VQ posterior truth.
```

Reason:

```text
PICF must scale across missing point clouds, missing tactile, missing wrist
views and different camera/layout distributions.  Locking posterior identity
to a single visual prototype space would fight multimodal belief fusion.
```

Future-safe variant:

```text
multimodal prototype bank as birth/proposal/dedup prior only;
never as posterior truth.
```

### 4.3 QASA

External invariant:

```text
slot quality should be separated from the main downstream loss;
active K should be content-dependent.
```

PICF deployment:

```text
aqr_slot_quality_head
aqr_slot_quality_owner_active_floor
target_object_quality
target_no_object_prob
target_duplicate_prob
active_weight / context_weight
```

Audit decision:

```text
covered.

The learned quality head is residual and bounded; accepted owner candidates
receive an active floor, so early noisy quality cannot suppress the only valid
task object.
```

### 4.4 Object Binding / IsSameObject

External invariant:

```math
s(i,j)
=
z_i^\top W z_j
```

where the same-object signal is pairwise/quadratic and low-dimensional, not
just linear cosine.

PICF deployment:

```text
binding_signature_proj
binding_quadratic_diag
binding_low_rank_left/right
_binding_signature_quadratic_scores
_calibrate_pairwise_binding_score
posterior_binding_signature_memory_enabled
posterior_binding_signature_dispersion_gate_enabled
```

Audit decision:

```text
covered as runtime signal and posterior memory.

Not added as online supervised IsSameObject loss because current CALVIN
sidecars are weak measurements, not clean same-object labels.
```

### 4.5 Object-Centric VLA / STORM / SlotVLA

External invariant:

```text
robot manipulation benefits from task-aware object slots, but frozen
foundation features should remain intact and object slots must be stable before
full action pressure is trusted.
```

PICF deployment:

```text
sidecar/proposal/tracklet typed measurements
object_candidate_owner_assignment
object_candidate_owner_point_priors
object_candidate_owner_x
posterior_owner_transport_enabled
picf-action-prefix-stopgrad
action-aware smoke with PaliGemma trainable and V-JEPA/Sonata/AnyTouch frozen
```

Audit decision:

```text
covered.

The staged run taxonomy is intentional: anchor probes, frozen-policy slot
validation, action-aware smoke, then longer co-training.
```

### 4.6 Visuo-Tactile Object Binding

External invariant:

```text
tactile/contact observations should update the object being manipulated, not
create a permanent gripper object that steals object tokens.
```

PICF deployment:

```text
tactile_attach_to_object_owner=True
posterior_owner_transport_roles=(1,)
effector_persistent_anchors=0 in maintained launch
effector_observation_anchors=0 in maintained launch
object owner transport precision fusion
```

Audit decision:

```text
covered in the maintained launch contract.

The config default still keeps one persistent effector anchor for compatibility,
but production validation disables effector anchors.  This is documented rather
than silently hidden.
```

## 5. Why More Modules Are Not Added Now

### 5.1 Full Reconstruction Decoder

A full RGB or dense-feature decoder would optimize:

```math
L_{\text{recon}}
=
\sum_i
\left\|
z_i
-
\sum_j m_{j,i}\hat z_{j,i}
\right\|^2 .
```

This is useful in reconstruction-first object discovery.  It is risky as
PICF production truth because:

```text
1. dense V-JEPA/point/tactile context must not be pruned or rewritten;
2. action training should see belief files, not an image autoencoder;
3. sidecar masks are noisy, so reconstruction loss can overfit background or
   contact artifacts.
```

PICF already has safer guarded variants:

```text
object_explanation
slot_jepa
support_pred
future visual/tactile/point prediction auxiliaries
```

If reconstruction is revisited, it should be masked frozen-feature prediction
as a low-weight training-only auxiliary, not a posterior authority path.

### 5.2 Full MetaSlot VQ Posterior Truth

Hard VQ prototypes are not added because the posterior identity must remain
multimodal:

```math
\text{identity}
\neq
\arg\min_k
\|z^{visual}-c_k\|^2 .
```

The future-safe form is:

```math
p(birth_j \mid z_t)
\leftarrow
\operatorname{softprior}(prototype, geometry, support, language)
```

followed by the usual object-candidate assignment and posterior gates.

### 5.3 Online IsSameObject Loss

The object-binding paper justifies a pairwise/quadratic binding subspace.  It
does not justify training a supervised IsSameObject loss from noisy sidecars as
if they were clean object labels.

PICF therefore uses the pairwise subspace in runtime binding and diagnostics,
and defers an offline probe/teacher until enough inspected masks or reliable
tracklets exist.

### 5.4 Blind SAM

Blind SAM remains rejected.  It found visually plausible regions, but it did
not reliably select the task object in CALVIN and could promote irrelevant
background, drawer sides or robot protrusions.  The retained `proposal_*`
schema is for inspected contact/motion/tracklet-aware sources, not class-blind
SAM truth.

## 6. Current Runtime Evidence

Short action-aware validation:

```text
run:
  picf_a7_actionaware_after_dedup_smoke300_20260520

status:
  completed 300 steps

trainable:
  PaliGemma / semantic-action stack
  PICF AQR/posterior/task/control adapters
  action-side heads

frozen:
  V-JEPA visual encoder
  Sonata point encoder
  AnyTouch tactile encoder
```

Step 300:

```text
loss_action_default_equiv                 0.03255
loss_anchor_object_pull                   0.41526
loss_anchor_pv                            0.67010
loss_object_explanation_point             2.19150
loss_binding_consistency                  0.83391
aqr_active_same_role_support_overlap_max  0.00105
aqr_context_same_role_support_overlap_max 0.24725
aqr_downstream_same_role_support_overlap  0.27396
aqr_same_role_support_overlap_max         0.37281
posterior_recycle_rate                    0.08178
posterior_identity_switch_rate            0.10556
active_duplicate_overlap_max              0.00000
```

Interpretation:

```text
passed:
  active duplicate collapse fixed in the short action-aware gate;
  active same-role support overlap stays near zero;
  downstream overlap ends below the step50 level;
  action-equivalent loss improves under trainable PaliGemma/action pressure;
  active owner overlays remain on inspected sidecar/contact masks.

still open:
  object auxiliary losses remain task-window dependent;
  stable identity switch rebounds late in the 300-step window;
  no 30000-step run or CALVIN/video acceptance has been completed with this
  exact final contract.
```

## 7. Final Gate Policy

Do not promote by static audit alone.

Promote only if the next longer run satisfies:

```text
aqr_active_same_role_support_overlap_max:
  remains low, no sustained rebound.

aqr_downstream_same_role_support_overlap_max:
  remains below the old failure band.

posterior_recycle_rate:
  does not saturate.

posterior_identity_switch_rate_stable:
  trends down or remains bounded.

loss_action_default_equiv:
  remains comparable to or better than 4-22 ablation at matched step scale.

loss_object_explanation_point / loss_anchor_object_pull:
  bounded; no unbounded monotonic explosion.

overlays:
  active owner sits on task/contact object; gray reserve/context rows do not
  become action-visible duplicates.
```

If this fails, the next repair target should be diagnosed from these metrics,
not from raw same-role overlap alone.

## 8. 2026-05-20 Follow-up Validation Launch

After this audit was added, the next 300-step validation was launched on A7:

```text
run:
  picf_a7_latestslot_final_smoke300_20260520

launcher:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh

scope:
  full PICF slot/router/posterior/OEML stack trainable
  PaliGemma/action pressure frozen
  V-JEPA/Sonata/AnyTouch pretrained encoders frozen
  not anchor_only

purpose:
  verify the finalized latest-slot contract under the comprehensive frozen
  policy validation profile before moving back to longer co-training.
```

Review at steps 100, 200 and 300 with the gate in section 7.  Stop early only
if the active owner path clearly reproduces the old failure:

```text
active owner misses the inspected sidecar/contact object;
active same-role support overlap rebounds into the collapse band;
active duplicate posterior overlap becomes nonzero;
object explanation or anchor pull becomes unbounded;
posterior recycle saturates.
```

### Step 50 / 100 Interim Read

```text
step50:
  loss_total                         0.16501
  loss_anchor_object_pull            0.32344
  loss_anchor_pv                     0.68967
  loss_object_explanation_point      2.18930
  aqr_active_same_role_support_max   0.00180
  aqr_context_same_role_support_max  0.24625
  aqr_downstream_same_role_support   0.34446
  aqr_same_role_support_overlap_max  0.45906
  posterior_recycle_rate             0.05025
  posterior_identity_switch_stable   0.35000
  object_owner_graph_active_dist_m   0.00020
  posterior_owner_transport_dist_m   0.26555

step100:
  loss_total                         0.15579
  loss_anchor_object_pull            0.28329
  loss_anchor_pv                     0.67814
  loss_object_explanation_point      2.39027
  aqr_active_same_role_support_max   0.00139
  aqr_context_same_role_support_max  0.31601
  aqr_downstream_same_role_support   0.33344
  aqr_same_role_support_overlap_max  0.50378
  posterior_recycle_rate             0.02834
  posterior_identity_switch_stable   0.47917
  object_owner_graph_active_dist_m   0.00028
  posterior_owner_transport_dist_m   0.29751
```

Interim decision:

```text
continue to step200.

not reproduced:
  active support collapse;
  active duplicate posterior collapse;
  downstream context clone rebound;
  graph owner missing the sidecar/contact mask;
  recycle saturation.

watch:
  posterior owner transport distance is not yet improving;
  stable identity switch worsened by step100;
  object explanation point is bounded but not monotonic.
```

### Step 150 / 200 Interim Read

```text
step150:
  loss_total                         0.13430
  loss_anchor_object_pull            0.23175
  loss_anchor_pv                     0.61812
  loss_object_explanation_point      2.23347
  aqr_active_same_role_support_max   0.01938
  aqr_context_same_role_support_max  0.41421
  aqr_downstream_same_role_support   0.43333
  aqr_same_role_support_overlap_max  0.83434
  posterior_recycle_rate             0.01349
  posterior_identity_switch_stable   0.43667
  object_owner_graph_active_dist_m   0.00016
  posterior_owner_transport_dist_m   0.43777

step200:
  loss_total                         0.21833
  loss_anchor_object_pull            0.45013
  loss_anchor_pv                     0.64955
  loss_object_explanation_point      2.58956
  aqr_active_same_role_support_max   0.00592
  aqr_context_same_role_support_max  0.39596
  aqr_downstream_same_role_support   0.41356
  aqr_same_role_support_overlap_max  0.84212
  posterior_recycle_rate             0.02404
  posterior_identity_switch_stable   0.31333
  object_owner_graph_active_dist_m   0.00026
  posterior_owner_transport_dist_m   0.56751
```

Step-200 decision:

```text
continue to step300, but do not claim final behavior health from the graph
owner path alone.

confirmed healthy:
  active owner overlap remains low;
  active duplicate posterior overlap remains 0;
  graph owner is still pinned to the sidecar/contact object in the overlay;
  recycle remains low and non-saturating.

remaining risk:
  posterior owner transport distance worsens from 0.26555 to 0.56751 m;
  full same-role overlap is high because inactive/context rows can still
  overlap, although active/downstream rows are controlled;
  object explanation point and anchor pull are bounded but non-monotonic.

interpretation:
  the old active-anchor collapse is not reproduced, but the posterior file
  write-through path still needs the step300 check.  If step300 keeps graph
  ownership healthy but posterior transport remains poor, the next repair
  target is posterior owner-file transport/reliability, not proposal seeding
  or a new slot module.
```

### Step 250 Interim Read

```text
step250:
  loss_total                         0.22173
  loss_anchor_object_pull            0.45009
  loss_anchor_pv                     0.63124
  loss_object_explanation_point      2.76062
  loss_aqr_denoising                 0.76723
  aqr_active_same_role_support_max   0.05478
  aqr_context_same_role_support_max  0.40670
  aqr_downstream_same_role_support   0.45898
  aqr_same_role_support_overlap_max  0.99029
  posterior_recycle_rate             0.01073
  posterior_identity_switch_stable   0.28417
  object_owner_graph_active_dist_m   0.00033
  posterior_owner_transport_dist_m   0.62500
```

Step-250 interpretation:

```text
the script is not hung: throughput remains about 0.0899 step/s, or roughly
11.1 s/step.  The graph-side object owner remains pinned to the sidecar
object, and active duplicate overlap remains 0.  The failure mode, if it
persists at step300, is posterior owner-file transport drifting away from the
graph object owner, not missing sidecar proposals or active anchor collapse.

full same-role overlap is high again, but the active row overlap stays low;
this means inactive/context rows still clone supports and should not be used
as the primary health metric unless they enter the action graph with high
downstream weight.
```

### Step 300 Final Read

```text
step300:
  loss_total                         0.23964
  loss_anchor_object_pull            0.52026
  loss_anchor_pv                     0.61655
  loss_object_explanation_point      2.39829
  loss_aqr_denoising                 0.82561
  loss_binding_consistency           0.86607
  loss_pv_weak                       6.42380
  aqr_active_anchor_count            1.25000
  aqr_effective_anchor_count         1.22284
  aqr_active_same_role_support_max   0.11539
  aqr_context_same_role_support_max  0.42732
  aqr_downstream_same_role_support   0.52827
  aqr_same_role_support_overlap_max  0.99169
  posterior_recycle_rate             0.01157
  posterior_identity_switch_rate     0.18667
  posterior_identity_switch_stable   0.40583
  active_duplicate_overlap_max       0.00000
  active_calibrated_swap_rate        0.00000
  object_owner_graph_active_dist_m   0.00035
  posterior_owner_transport_dist_m   0.51871
  owm_tracklet_tokens                79.93
  owm_proposal_tokens                1.51
  throughput                         0.08808 step/s
```

Final short-run decision:

```text
script/runtime:
  healthy.  The 300-step validation completed in about 57 minutes, with
  throughput around 11.5 s/step.  The run was not hung or broken.

fixed relative to the old visible failures:
  active owner is on or near the sidecar/proposal object in the step300
  overlay;
  gripper/effector role is not present in the active role layout;
  active duplicate posterior overlap remains 0;
  active same-role support overlap stays far below the old collapse band;
  recycle does not saturate;
  tracklet/proposal sidecar data are present in the run.

not fully solved:
  posterior owner transport does not faithfully close the graph owner into the
  posterior file geometry; it worsens from 0.26555 m at step50 to 0.51871 m at
  step300;
  inactive/context rows still have high full same-role overlap, visible as
  a high aqr_same_role_support_overlap_max even when active rows are clean;
  anchor object pull is bounded but not decreasing through the full 300-step
  window.

next mathematically local repair:
  keep the latest-slot graph ownership design.  Do not add another proposal
  source.  The next target is the posterior write-through/transport closure:
  the object-owner graph row is correct, but its accepted object geometry is
  not yet becoming a stable posterior object file.
```

## 9. 2026-05-20 Direct Owner Write-Through Repair

Follow-through document:

```text
temp/audits_20260520/posterior_owner_transport_direct_write_through_followthrough.md
```

Code-level repair:

```text
src/openpi/picf/core/pipeline.py::_posterior_owner_transport_measurement
```

The repaired path preserves graph-candidate identity until posterior-file
selection:

```math
S_{j,g}
=
\sum_o
B^{post\leftarrow obs}_{j,o}
A^{obs\leftarrow graph}_{o,g}
r_g .
```

For each owner role, accepted graph-owner candidates are assigned to bounded
posterior files before geometry averaging:

```math
(j^\*,g^\*)
=
\operatorname{TopK}_{j,g}
\left[
S_{j,g}q_j
\right].
```

Then:

```math
\hat x_{j^\*}=x^{owner}_{g^\*},
\quad
\hat S_{j^\*}=S^{owner}_{g^\*}.
```

The old obs-averaged owner transport remains only as fallback when there is no
direct candidate/file responsibility match.

The implementation collects selected `(slot, graph, score)` triples first and
then applies out-of-place `index_copy` writes.  This keeps the responsibility
score differentiable without the in-place slice writes that break PyTorch
autograd version counters.

Metric repair:

```text
posterior_owner_transport_dist_to_standard:
  pre-fusion correction size.

posterior_owner_transport_dist_after_fusion:
  actual posterior closure residual after precision fusion.

posterior_owner_transport_active_dist_after_fusion_*:
  primary short-run acceptance metric for active owner write-through.
```

This is the PICF-compatible translation of slot responsibility write-back from
Slot Attention / SAVi / MetaSlot / QASA-style object-centric learning.  It is
not a new proposal source, not SAM, not a hard VQ visual label, and not a
reconstruction decoder.

### 9.1 A7 300-Step Validation

Run:

```text
picf_a7_owner_direct_autogradsafe_smoke300_20260520
```

Result:

```text
completed 300/300;
checkpoint saved at:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_owner_direct_autogradsafe_smoke300_20260520/300
metrics:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_owner_direct_autogradsafe_smoke300_20260520/metrics.jsonl
overlays:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_owner_direct_autogradsafe_smoke300_20260520/anchor_overlays
```

Key metric trajectory:

```text
step 50:
  posterior_owner_transport_active_dist_after_fusion_mean = 0.00426 m
  object_candidate_owner_geometry_active_dist_mean = 0.00021 m
  active/context/downstream/raw support overlap max = 0.0007 / 0.2626 / 0.3412 / 0.4752

step 200:
  posterior_owner_transport_active_dist_after_fusion_mean = 0.00236 m
  object_candidate_owner_geometry_active_dist_mean = 0.00021 m
  active/context/downstream/raw support overlap max = 0.0567 / 0.1500 / 0.3002 / 0.9961

step 300:
  posterior_owner_transport_active_dist_after_fusion_mean = 0.00305 m
  object_candidate_owner_geometry_active_dist_mean = 0.00028 m
  active/context/downstream/raw support overlap max = 0.0427 / 0.1460 / 0.3116 / 0.9977
```

Verdict:

```text
The direct owner write-through repair passes the 300-step short validation.
The previously suspicious pre-fusion owner-transport distance is confirmed to
be a correction-size signal, not the closure residual.  The after-fusion active
owner residual stays in the millimeter range, active duplicate overlap stays
zero, and downstream-visible overlap remains far below the old failure band.
Raw reserve overlap remains high and should continue to be reported separately.
```

### 9.2 Local Strict Re-Audit After Full Core Tests

The post-validation local full core test pass found and fixed two small but
important consistency issues:

```text
proposal debug dataflow:
  proposal-token debug logging was accidentally nested under tracklet debug
  logging.  It is now unconditional on tracklets, so proposal-only diagnostics
  expose owm_proposal_tokens correctly.

cache contract:
  the production cache writes active posterior files only.  The old test
  assumed every persistent row must be valid, which contradicted the active /
  context / reserve contract.  Tests now assert active cache rows and next-step
  cache reads instead of all-row validity.

binding consistency:
  the guarded temporal binding loss now uses current weights for the
  current->future term and future weights for the future->current term.  This
  removes a small future-slot-order dependence while preserving detached,
  permutation-tolerant matching.
```

Strict local validation after the repair:

```text
pipeline_test.py + training_test.py:
  133 passed

latest-slot deployment audit:
  14/14 PASS

object-candidate slot binding audit:
  ok=true, 37 checks

verify/dataflow/strict/mvtrack audits:
  PASS
```

### 9.3 Owner-Direct Final Smoke300 Interim Evidence

Run:

```text
picf_a7_owner_direct_final_smoke300_20260520
```

This is a frozen-policy slot validation, not a full action-cotrain acceptance:
the PICF slot/router/posterior/OEML path is trainable, while PaliGemma/action
pressure is disabled for the diagnostic.

Step trajectory through 300:

```text
step   total    obj_pull  point    dup     active_ov  ctx_ov  downstream_ov  raw_ov  owner_after_fusion
50     0.17531  0.35183   2.20806  0.06125 0.00066    0.2563  0.3425         0.4630  0.00411 m
100    0.14806  0.27917   2.11454  0.05390 0.00000    0.3196  0.3277         0.5439  0.00446 m
150    0.15717  0.29885   2.22975  0.05145 0.00696    0.4328  0.4397         0.8546  0.00510 m
200    0.18040  0.37486   2.04953  0.04655 0.01124    0.4373  0.4436         0.9126  0.00415 m
250    0.17601  0.34864   2.30122  0.04655 0.00461    0.4683  0.4705         0.9139  0.00493 m
300    0.20884  0.47031   1.82275  0.04410 0.00744    0.4851  0.4868         0.9461  0.00532 m
```

Read this with the maintained slot contract:

```text
active posterior:
  still healthy through step 300.  Active duplicate overlap stays zero, active
  support overlap stays near zero, and owner write-through remains in the
  millimeter range after fusion.

context/reserve:
  raw all-row overlap rises again.  This is not an active-object failure, but it
  remains a context-capacity risk and must be tracked separately from active
  posterior ownership.

unweighted diagnostics:
  aqr_denoising / support_pred drift in this frozen-policy run because their
  lambdas are zero.  They are not acceptance metrics for this smoke.

verdict:
  the direct owner write-through contract passes this 300-step smoke.  It fixes
  the previous failure where accepted sidecar/object candidates could be
  averaged away before posterior file fusion.  It does not close the separate
  raw reserve/context-overlap issue, nor does it replace action-cotrain or
  CALVIN/video acceptance.
```
