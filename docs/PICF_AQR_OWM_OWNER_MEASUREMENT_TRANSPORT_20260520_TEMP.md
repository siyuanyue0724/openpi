# PICF-AQR-OWM Owner Measurement Transport Follow-Through

Date: 2026-05-20

Status: implemented locally; A7 300-step validation pending.

## Trigger

The A7 support-aware context-dedup validation
`picf_a7_context_support_dedup_300_20260520` fixed the previous step-150
context/downstream duplicate rebound:

```text
step 300:
  aqr_active_same_role_support_overlap_max     = 0.049
  aqr_context_same_role_support_overlap_max    = 0.512
  aqr_downstream_same_role_support_overlap_max = 0.628
```

However, it did not fully solve task-object owner localization:

```text
step 300 overlay:
  sidecar/mask: switch region
  graph active row: near but not centered on mask
  posterior active file: farther from mask

step 300 metrics:
  loss_object_explanation_point ~= 6.09
  grad clipping active
```

The issue is therefore no longer generic context duplication or SAM/proposal
noise.  The remaining failure is that accepted sidecar/contact owner evidence
is not transported strongly enough into graph geometry and posterior files.

## Mathematical Diagnosis

Before this repair, an accepted owner candidate affected the graph mainly as a
support prior:

```math
p_{j,i}^{point}
\leftarrow
(1-\lambda)p_{j,i}^{read}
+
\lambda p_{j,i}^{owner}
```

The graph geometry was then computed indirectly:

```math
x_j =
\sum_i p_{j,i}^{point} X_i .
```

This is too weak for noisy multimodal binding.  If the current read prior is
diffuse or shifted, the accepted owner mask is diluted before active selection;
posterior owner transport then reuses the diluted `x_j`.

The correct belief-filter interpretation is:

```text
accepted owner candidate:
  a soft measurement z_owner = (x_owner, S_owner, confidence)

not:
  a hard label
not:
  just a weak support prior
```

The measurement center should be:

```math
x_j^{owner}
=
\frac{\sum_i m_{j,i}^{owner} X_i}
       {\sum_i m_{j,i}^{owner}}
```

and its covariance:

```math
S_j^{owner}
=
\frac{\sum_i m_{j,i}^{owner}(X_i-x_j^{owner})(X_i-x_j^{owner})^T}
       {\sum_i m_{j,i}^{owner}}
+ \epsilon I .
```

The graph geometry used for active selection becomes:

```math
\tilde x_j =
(1-\gamma)x_j + \gamma x_j^{owner}
```

for accepted owner rows only.  Posterior owner transport then uses the same
owner measurement before precision fusion:

```math
\Lambda_j^{new}
=
\Lambda_j^{standard}
+
\beta c_j \Lambda_j^{owner}.
```

This keeps posterior authority and uncertainty while preventing a validated
owner candidate from being treated as a secondary hint.

## Paper-Code Boundary

The compatible paper/code principles checked for this repair are:

```text
Slot Attention / SAVi / AdaSlot / OCL:
  object slots compete over input support; background residual absorbs invalid
  fragments.

QASA:
  slot quality should not be entangled with control/reconstruction pressure;
  accepted object-quality evidence must guide active slot selection.

SlotVLA / robot object-slot work:
  task/contact object evidence should initialize or transport object-centric
  slots as measurement hypotheses, not simply bias a global policy token.

Object Binding in pretrained ViTs:
  same-object evidence is pairwise/subspace evidence, so owner assignment
  remains gated; it is not a hard identity lock.
```

2025--2026 checks used for the final interpretation:

```text
SlotVLA (arXiv:2511.06754):
  object/relation slots are used as compact action-decoding tokens, but the
  benchmark explicitly provides box/mask/tracking labels.  This supports our
  decision to treat contact-motion sidecars as measurement evidence rather
  than expecting action loss alone to discover the task object.

STORM (arXiv:2601.20381):
  task-aware object slots are stabilized before downstream policy adaptation.
  This supports staged/frozen-policy validation and rejects direct end-to-end
  action pressure as the first test for owner localization.

OCRA (arXiv:2603.14401):
  object-centric manipulation benefits from 3D/tactile priors fused into the
  policy.  This supports contact-motion point/mask priors as multimodal owner
  evidence, but does not justify hard identity truth.

When Slots Compete (arXiv:2603.11246):
  overlapping slots should be treated as a slot-set competition/merging issue.
  This supports downstream/context duplicate filtering, not blind deletion of
  dense tokens.
```

Production decisions that remain rejected:

```text
blind SAM:
  still off and archived.

hard MetaSlot VQ posterior truth:
  still rejected.

full image reconstruction decoder as action-time truth:
  still rejected for this router.
```

## Implemented Repair

### 1. First-Class Owner Geometry

Added graph fields:

```text
PicfAnchorPriorGraphState.object_candidate_owner_x
PicfAnchorPriorGraphState.object_candidate_owner_S
```

Added implementation:

```text
src/openpi/picf/core/pipeline.py::_object_candidate_owner_geometry
```

This computes accepted owner-candidate center/covariance from
`object_candidate_owner_point_priors`.

### 2. Graph Geometry Transport

In AQR graph construction, accepted owner rows now blend their graph geometry
toward owner-candidate measurement geometry:

```text
object_candidate_owner_geometry_mix
```

The maintained validation script uses:

```text
--object-candidate-owner-geometry-mix 0.95
```

### 3. Posterior Transport Uses Owner Geometry

Posterior owner transport now reuses the explicit owner measurement geometry:

```text
posterior_owner_transport_candidate_geometry_mix
```

The maintained validation script uses:

```text
--posterior-owner-transport-candidate-geometry-mix 1.0
```

### 4. Owner-Calibrated Slot Quality

QASA-style slot quality still predicts object/no-object/duplicate, but accepted
owner rows receive a lower active-quality bound:

```text
aqr_slot_quality_owner_active_floor
```

The maintained validation script uses:

```text
--aqr-slot-quality-owner-active-floor 0.65
```

This prevents the learned/duplicate quality head from suppressing a validated
owner measurement during early training.

## New Diagnostics

Added debug metrics:

```text
object_candidate_owner_geometry_rows
object_candidate_owner_geometry_dist_mean
object_candidate_owner_geometry_dist_max
object_candidate_owner_geometry_active_dist_mean
object_candidate_owner_geometry_active_dist_max
```

Acceptance:

```text
object_candidate_owner_geometry_active_dist_mean:
  should be near zero when an accepted active owner candidate exists.

posterior_owner_transport_confidence_mean:
  should become nonzero and should not stay at the previous near-zero owner
  failure state.

active overlays:
  graph active orange row should sit on the sidecar/mask core.

posterior overlays:
  active posterior owner should move toward the same sidecar/mask core after
  transport.
```

## Local Verification

Targeted tests:

```bash
uv run pytest src/openpi/picf/core/pipeline_test.py \
  -k "owner_candidate_geometry or slot_quality_owner_floor or context_slot_downstream_weight" -q
```

New tests:

```text
test_owner_candidate_geometry_returns_mask_center_measurement
test_slot_quality_owner_floor_preserves_accepted_measurement
```

Full local contract/audit checks completed after the repair:

```text
uv run python -m py_compile ...
git diff --check
bash -n scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh
uv run pytest src/openpi/picf/core/pipeline_test.py -k "owner_candidate_geometry or slot_quality_owner_floor" -q
uv run python scripts/picf_latest_slot_deployment_audit.py --fail-on-fail
uv run python scripts/verify_picf_owm_contract.py
uv run python scripts/picf_owm_strict_diagnose.py --fail-on-fail
uv run python scripts/picf_owm_dataflow_trace.py --fail-on-fail
uv run python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
```

## A7 Validation: 2026-05-20

Run:

```text
picf_a7_owner_measurement_transport_300_20260520
```

Mode:

```text
comprehensive frozen-policy validation
not anchor-only
unroll=2
burnin=1
2-GPU DDP
action loss weight = 0
PaliGemma/action/pretrained perception effectively frozen
PICF/AQR/OWM alignment modules trainable
blind SAM disabled
contact-motion sidecar enabled
```

Step 50 result:

```text
object_candidate_owner_geometry_rows             = 1.255
object_candidate_owner_geometry_active_dist_mean = 0.000195 m
object_candidate_owner_geometry_active_dist_max  = 0.000205 m

aqr_active_same_role_support_overlap_max         = 0.0018
aqr_context_same_role_support_overlap_max        = 0.2502
aqr_downstream_same_role_support_overlap_max     = 0.3642
aqr_same_role_support_overlap_max                = 0.4586

posterior_owner_transport_applied_fraction       = 0.125
posterior_owner_transport_active_confidence_mean = 0.3226
posterior_owner_transport_active_dist_mean       = 0.2553 m

loss_total                                       = 0.1648
loss_anchor_object_pull                          = 0.3229
loss_anchor_pv                                   = 0.6898
loss_object_explanation_point                    = 2.1870
loss_slot_quality                                = 0.000036
grad_clip_applied                                = false
```

Interpretation:

```text
Fixed:
  accepted owner sidecar/mask geometry is now transported into graph anchor
  geometry.  The active graph row is essentially on top of the owner
  measurement, and active same-role overlap is far below the previous failure
  band.

Not yet proven:
  posterior file transport still reports a nonzero owner distance at step 50.
  This metric is distance between the owner-transport measurement and the
  standard posterior measurement before precision fusion; it is a correction
  magnitude / disagreement metric, not directly the final overlay error.
  It must still be checked at step 100/200/300 because unbounded disagreement
  would mean the ordinary posterior branch keeps fighting the owner evidence.

Next acceptance check:
  posterior_owner_transport_active_dist_mean may remain nonzero because it
  measures owner-vs-standard correction magnitude before precision fusion.
  It should remain bounded, while overlays and owner geometry should stay on
  the sidecar core and gradients should not clip.
```

Step 100 result:

```text
object_candidate_owner_geometry_rows             = 1.275
object_candidate_owner_geometry_active_dist_mean = 0.000216 m

aqr_active_same_role_support_overlap_max         = 0.0024
aqr_context_same_role_support_overlap_max        = 0.3577
aqr_downstream_same_role_support_overlap_max     = 0.4063
aqr_same_role_support_overlap_max                = 0.6115

posterior_owner_transport_applied_fraction       = 0.125
posterior_owner_transport_active_confidence_mean = 0.4184
posterior_owner_transport_active_dist_mean       = 0.2813 m

loss_total                                       = 0.1508
loss_anchor_object_pull                          = 0.2827
loss_anchor_pv                                   = 0.6608
loss_object_explanation_point                    = 2.1716
loss_slot_quality                                = 0.000292
grad_clip_applied                                = false
```

Overlay check:

```text
step_000100__push_the_switch_downwards__mask_active.png:
  active graph owner and active posterior owner are drawn on the sidecar mask
  / switch region.

Interpretation:
  The step-100 visual evidence supports the graph/posterior owner-localization
  repair.  The batch-level posterior distance metric is still not a success
  metric by itself; it says the owner measurement is still correcting the
  standard posterior branch by a large amount.  Therefore the run should
  continue to step 200/300 to check whether the correction stays stable rather
  than becoming a new conflict.
```

Step 150 result:

```text
object_candidate_owner_geometry_active_dist_mean = 0.000278 m
aqr_active_same_role_support_overlap_max         = 0.0422
aqr_context_same_role_support_overlap_max        = 0.4623
aqr_downstream_same_role_support_overlap_max     = 0.6048
aqr_same_role_support_overlap_max                = 0.9812
posterior_owner_transport_active_confidence_mean = 0.1935
posterior_owner_transport_active_dist_mean       = 0.5682 m
loss_total                                       = 0.1578
loss_anchor_object_pull                          = 0.2638
loss_anchor_pv                                   = 0.5780
loss_object_explanation_point                    = 2.8618
grad_clip_applied                                = false
```

Step 200 result:

```text
object_candidate_owner_geometry_active_dist_mean = 0.000469 m
aqr_active_same_role_support_overlap_max         = 0.1131
aqr_context_same_role_support_overlap_max        = 0.4986
aqr_downstream_same_role_support_overlap_max     = 0.7659
aqr_same_role_support_overlap_max                = 0.9992
posterior_owner_transport_active_confidence_mean = 0.2059
posterior_owner_transport_active_dist_mean       = 0.3590 m
loss_total                                       = 0.2697
loss_anchor_object_pull                          = 0.5630
loss_anchor_pv                                   = 0.5753
loss_object_explanation_point                    = 3.1591
grad_clip_applied                                = false
```

Overlay checks:

```text
step 150 open_the_drawer:
  active graph/posterior owners remain on the drawer sidecar mask.

step 200 slide_the_door_to_the_left:
  active graph/posterior owners remain on the sidecar mask.
```

Updated interpretation:

```text
Solved by this repair:
  accepted owner-candidate geometry now drives graph geometry and the selected
  active posterior owner can visually sit on the sidecar/mask object.

Still open:
  non-active context/downstream rows continue to reuse the same owner support.
  This raises raw/context/downstream overlap and object-explanation point loss
  even when the single active owner is correct.

Next likely repair:
  apply the same object-candidate row-capacity / duplicate-demotion principle
  to downstream/context rows, not just active selection.  The repair should not
  delete dense context tokens; it should lower the downstream weight of
  duplicate rows that explain an already-owned sidecar object.
```

## 2026-05-20 Active-Context Support Dedup Repair

The step-150/200 residual failure is now narrowed to this dataflow:

```text
accepted owner sidecar -> active owner row is correct
but
context/downstream rows can still carry nearly identical full support
```

The previous context filter suppressed rows that were duplicates under
object-core support and geometry.  That is insufficient when two rows share
the same diffuse full support while their point/proposal object-core evidence
is not identical.  In that case, the extra row is not a new object; it is a
second explanation of an already-owned object.

Mathematically, let `A` be the active owner set and `s_i` be the normalized
support distribution of row `i`.  A context row should become downstream-visible
only if it is not already explained by an active owner:

```math
\max_{a \in A}
\frac{\langle s_i, s_a \rangle}
     {\|s_i\|_1 \|s_a\|_1}
< \tau_{active-support}.
```

This is a slot-competition invariant, not an extra supervision loss.  It
preserves dense context memory and background/reserve rows, but prevents a
duplicate context row from becoming a second action-visible owner.

Implemented controls:

```text
aqr_context_slot_active_support_overlap_enabled
aqr_context_slot_active_support_overlap_threshold
```

Maintained validation script:

```text
--aqr-context-slot-active-support-overlap-enabled
--aqr-context-slot-active-support-overlap-threshold 0.50
```

New regression test:

```text
test_context_slot_downstream_weight_deduplicates_active_diffuse_support
```

The test constructs an active row and a context row with identical full visual
support but different object-core/proposal evidence.  The context row must be
suppressed by full-support active deduplication while an unrelated context row
remains available at the reserve context weight.

Validation run:

```text
picf_a7_active_support_dedup_300_20260520
```

Acceptance focus:

```text
primary:
  aqr_downstream_same_role_support_overlap_max must not reproduce the old
  0.76+ rebound while active owner remains correct.

secondary:
  loss_object_explanation_point should stop rising monotonically after
  active-owner localization succeeds.

non-goal:
  raw aqr_same_role_support_overlap_max may stay high because reserve/background
  rows keep dense context capacity.  The important metric is downstream-visible
  duplicate owner reuse.
```

Step 50:

```text
loss_total                                       = 0.1819
loss_anchor_object_pull                          = 0.3701
loss_anchor_pv                                   = 0.6911
loss_object_explanation_point                    = 2.2234
aqr_active_same_role_support_overlap_max         = 0.0017
aqr_context_same_role_support_overlap_max        = 0.2398
aqr_downstream_same_role_support_overlap_max     = 0.3327
aqr_same_role_support_overlap_max                = 0.4620
object_candidate_owner_geometry_active_dist_mean = 0.000221 m
posterior_owner_transport_active_confidence_mean = 0.3294
posterior_owner_transport_active_dist_mean       = 0.2884 m
grad_clip_applied                                = false
```

Compared to `picf_a7_owner_measurement_transport_300_20260520` step 50:

```text
context overlap:    0.2502 -> 0.2398
downstream overlap: 0.3642 -> 0.3327
active overlap:     unchanged near zero
```

Interpretation:

```text
Step 50 is directionally correct but not sufficient.  The old failure appeared
after step 150, when downstream/context rows began reusing the active owner
support.  Continue to step 100/150/200 before claiming closure.
```

Step 100:

```text
loss_total                                       = 0.1489
loss_anchor_object_pull                          = 0.2629
loss_anchor_pv                                   = 0.6344
loss_object_explanation_point                    = 2.4188
aqr_active_same_role_support_overlap_max         = 0.0045
aqr_context_same_role_support_overlap_max        = 0.3399
aqr_downstream_same_role_support_overlap_max     = 0.3779
aqr_same_role_support_overlap_max                = 0.6498
object_candidate_owner_geometry_active_dist_mean = 0.000221 m
posterior_owner_transport_active_confidence_mean = 0.2706
posterior_owner_transport_active_dist_mean       = 0.5723 m
grad_clip_applied                                = false
```

Compared to `picf_a7_owner_measurement_transport_300_20260520` step 100:

```text
context overlap:          0.3577 -> 0.3399
downstream overlap:       0.4063 -> 0.3779
loss_anchor_object_pull:  0.2827 -> 0.2629
loss_object_expl_point:   2.1716 -> 2.4188
```

Overlay:

```text
step_000100__push_the_switch_downwards__mask_active.png:
  active owner remains on the switch sidecar/mask region.
```

Interpretation:

```text
The active-context support gate improves the intended overlap metrics at
step 100 and does not break owner localization.  It has not yet improved the
point-explanation loss; this remains the key secondary metric to monitor at
step 150/200.
```

Step 150:

```text
loss_total                                       = 0.1459
loss_anchor_object_pull                          = 0.2479
loss_anchor_pv                                   = 0.5810
loss_object_explanation_point                    = 2.5368
aqr_active_same_role_support_overlap_max         = 0.0182
aqr_context_same_role_support_overlap_max        = 0.4342
aqr_downstream_same_role_support_overlap_max     = 0.4569
aqr_same_role_support_overlap_max                = 0.9017
object_candidate_owner_geometry_active_dist_mean = 0.000233 m
posterior_owner_transport_active_confidence_mean = 0.2816
posterior_owner_transport_active_dist_mean       = 0.4316 m
grad_clip_applied                                = false
```

Compared to `picf_a7_owner_measurement_transport_300_20260520` step 150:

```text
context overlap:          0.4623 -> 0.4342
downstream overlap:       0.6048 -> 0.4569
raw overlap:              0.9812 -> 0.9017
loss_anchor_object_pull:  0.2638 -> 0.2479
loss_object_expl_point:   2.8618 -> 2.5368
```

Overlay:

```text
step_000150__open_the_drawer__mask_active.png:
  active owner remains on the drawer sidecar/mask region.
```

Interpretation:

```text
This is the first strong positive result for the active-support repair.  The
old step-150 downstream clone rebound is reduced materially, owner localization
is preserved, and the point-explanation loss improves instead of worsening.
The remaining raw overlap is mostly reserve/context capacity and should not be
treated as action-visible collapse by itself.
```

Step 200:

```text
loss_total                                       = 0.1581
loss_anchor_object_pull                          = 0.3043
loss_anchor_pv                                   = 0.6352
loss_object_explanation_point                    = 2.1444
aqr_active_same_role_support_overlap_max         = 0.0287
aqr_context_same_role_support_overlap_max        = 0.4842
aqr_downstream_same_role_support_overlap_max     = 0.5005
aqr_same_role_support_overlap_max                = 0.9669
object_candidate_owner_geometry_active_dist_mean = 0.000249 m
posterior_owner_transport_active_confidence_mean = 0.3745
posterior_owner_transport_active_dist_mean       = 0.2928 m
posterior_recycle_rate                           = 0.1070
grad_clip_applied                                = false
```

Compared to `picf_a7_owner_measurement_transport_300_20260520` step 200:

```text
context overlap:          0.4986 -> 0.4842
downstream overlap:       0.7659 -> 0.5005
raw overlap:              0.9992 -> 0.9669
loss_anchor_object_pull:  0.5630 -> 0.3043
loss_object_expl_point:   3.1591 -> 2.1444
```

Overlay:

```text
step_000200__slide_the_door_to_the_left__mask_active.png:
  active graph/posterior owner remains on the sidecar/mask region.
```

Interpretation:

```text
The old step-200 failure did not reproduce.  Active owner overlap remains low,
downstream-visible duplicate reuse is materially lower, and point-explanation
loss improves instead of continuing the old monotonic rise.

This does not mean raw reserve overlap is gone.  Raw overlap remains high
because reserve/context rows still preserve dense scene capacity.  The repaired
contract is narrower and more important: rows that duplicate an active owner
should not receive action-visible downstream owner weight.

Continue to step 300 before declaring this validation closed.  If step 300
preserves the step-200 pattern, this repair should be treated as the maintained
slot-competition completion rather than a temporary patch.
```

Step 250:

```text
loss_total                                       = 0.1380
loss_anchor_object_pull                          = 0.2443
loss_anchor_pv                                   = 0.6155
loss_object_explanation_point                    = 2.2171
aqr_active_same_role_support_overlap_max         = 0.0228
aqr_context_same_role_support_overlap_max        = 0.4786
aqr_downstream_same_role_support_overlap_max     = 0.5019
aqr_same_role_support_overlap_max                = 0.9855
object_candidate_owner_geometry_active_dist_mean = 0.000200 m
posterior_owner_transport_active_confidence_mean = 0.2744
posterior_owner_transport_active_dist_mean       = 0.3634 m
posterior_recycle_rate                           = 0.1042
grad_clip_applied                                = false
```

Step 300:

```text
loss_total                                       = 0.1375
loss_anchor_object_pull                          = 0.2668
loss_anchor_pv                                   = 0.6298
loss_object_explanation_point                    = 1.7866
aqr_active_same_role_support_overlap_max         = 0.0367
aqr_context_same_role_support_overlap_max        = 0.4819
aqr_downstream_same_role_support_overlap_max     = 0.5080
aqr_same_role_support_overlap_max                = 0.9928
object_candidate_owner_geometry_active_dist_mean = 0.000176 m
posterior_owner_transport_active_confidence_mean = 0.2253
posterior_owner_transport_active_dist_mean       = 0.3910 m
posterior_recycle_rate                           = 0.0970
grad_clip_applied                                = false
```

Final 300-step interpretation:

```text
The old downstream/context clone rebound did not return through step 300.

The important repaired metrics remain bounded:
  active support overlap    <= 0.0367
  downstream support overlap ~= 0.50 after step 200 instead of rising to 0.76+
  object point loss          falls to 1.7866 by step 300

The remaining high raw overlap is not ignored, but it is now correctly
localized to reserve/context capacity rather than action-visible owner rows.
This is the expected behavior for an overcomplete fixed-capacity slot bank:
unused rows may preserve dense scene/background evidence, but they must not
compete as duplicate object owners.

Close this item as fixed for short validation.  The next acceptance level is
long-run training/eval, not another local patch on this failure mode.
```

Static checks to run before A7:

```bash
uv run python -m py_compile \
  scripts/picf_core_train.py \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/pipeline_test.py

uv run python scripts/picf_latest_slot_deployment_audit.py --fail-on-fail
git diff --check
bash -n scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh
```

## A7 Validation Plan

Run name:

```text
picf_a7_owner_measurement_transport_300_20260520
```

Use the maintained comprehensive frozen-policy validation:

```text
scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh
```

Configuration remains:

```text
not anchor-only
PaliGemma/action pressure frozen
PICF slot/router/posterior/OEML trainable
sidecar contact/motion proposals enabled
blind SAM disabled
unroll=2
burnin=1
DDP world_size=2
300 steps for validation
```

Stop/continue rule:

```text
step 100:
  if owner geometry active distance is not near zero or overlays show no
  movement toward sidecar core, stop and inspect dataflow.

step 200:
  check loss_object_explanation_point, owner transport confidence, active
  overlap, and overlays.

step 300:
  decide whether this is ready for a longer frozen-policy validation or still
  needs posterior owner transport tuning.
```
