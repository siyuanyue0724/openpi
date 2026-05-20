# Posterior Owner Direct Write-Through Follow-Through

Date: 2026-05-20

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_LATEST_SLOT_FINAL_AUDIT_20260520_TEMP.md
```

## 1. Failure Boundary

The latest 300-step validation established a narrow failure:

```text
healthy:
  graph owner candidate is on the sidecar/contact object;
  active same-role overlap stays low;
  active duplicate posterior overlap stays zero;
  recycle does not saturate.

open:
  posterior owner transport was evaluated with a pre-fusion distance metric;
  the owner candidate could still be diluted by obs-averaged binding before
  file write-through.
```

The repair target is therefore not a new proposal source, not SAM, and not a
new action loss.  It is the responsibility path:

```math
candidate \rightarrow graph\ owner \rightarrow posterior\ file.
```

In code this is a direct candidate/file assignment contract: the candidate that
won graph-owner responsibility must be the same candidate written into the
posterior file unless it is rejected by the existing lifecycle, role, confidence,
or duplicate gates.

## 2. Paper-Code Principle

The compatible slot/OCL principle is responsibility-preserving write-back:

```text
Slot Attention / SAVi:
  slots compete over inputs with inverted attention, so the object responsibility
  that selects evidence is also the responsibility that updates the slot.

MetaSlot:
  duplicate slots are removed after candidate/prototype responsibility is known;
  duplicate handling is not a downstream action-side patch.

QASA:
  slot quality and slot selection are decoupled from reconstruction pressure;
  a low-quality slot should not receive ambiguous object attribution.

Object Binding in ViTs:
  same-object identity lives in a pairwise/quadratic subspace, so candidate/file
  compatibility must be explicit rather than only hidden-cosine.
```

PICF keeps this principle but maps it into a belief filter instead of copying a
visual reconstruction stack.

## 3. Old Path

The previous posterior transport path preserved owner evidence only after
observation-anchor aggregation:

```math
R^{obs}_{o,g}
=
A^{obs\leftarrow graph}_{o,g}r_g,
\quad
m_o=\sum_g R^{obs}_{o,g},
```

```math
R^{post}_{j,o}
=
B^{post\leftarrow obs}_{j,o}m_o,
\quad
\hat x_j
=
\frac{\sum_o R^{post}_{j,o}\hat x_o}
       {\sum_o R^{post}_{j,o}}.
```

This is safe as a fallback, but it has a failure mode: the graph-candidate index
`g` disappears before posterior-file selection.  If `B` is still noisy, the
validated graph owner can be averaged with unrelated observation supports and
written to the wrong or unstable posterior row.

## 4. New Direct Candidate Write-Through

The repair preserves candidate identity until the file choice:

```math
S_{j,g}
=
\sum_o
B^{post\leftarrow obs}_{j,o}
A^{obs\leftarrow graph}_{o,g}
r_g.
```

Then each eligible owner role solves a bounded candidate/file assignment:

```math
(j^\*,g^\*)
=
\operatorname{TopK}_{j,g}
\left[
S_{j,g}
q_j
1[r_j\in\mathcal R_{owner}]
1[r_g\in\mathcal R_{owner}]
\right],
```

where:

```math
q_j
=
activity_j
(1-demoted_j)
confidence_j^{assign}
confidence_j^{owner}.
```

For selected pairs, the posterior owner measurement is the accepted graph owner
measurement:

```math
\hat x_{j^\*}=x^{owner}_{g^\*},
\quad
\hat S_{j^\*}=S^{owner}_{g^\*}.
```

The standard precision fusion remains unchanged:

```math
\Lambda_j^+
=
(S_j^{std})^{-1}
+
\kappa c_j(\hat S_j)^{-1}.
```

This is not a hard label.  It remains gated by role, lifecycle, file competition,
direct responsibility score, max-per-role, covariance, and confidence cap.
The old obs-averaged owner transport remains only as fallback when no direct
candidate/file assignment is available for that role.

Implementation detail:
the selected `(slot, graph, score)` triples are first collected as indices and
then written with out-of-place `index_copy`.  This preserves the gradient from
the selected responsibility score while avoiding autograd version-counter
breakage from in-place slice writes.

## 5. Metric Repair

The old metric:

```text
posterior_owner_transport_dist_to_standard
```

is the distance before fusion:

```math
\|\hat x_j - x_j^{std}\|.
```

Large values can mean the owner measurement is correcting a bad standard
measurement, not that write-through failed.  The runtime now also reports:

```text
posterior_owner_transport_dist_after_fusion_*
posterior_owner_transport_active_dist_after_fusion_*
```

which measure:

```math
\|\hat x_j - x_j^+\|.
```

The next validation should judge posterior closure with the after-fusion metric
and overlays, while keeping the pre-fusion metric as a correction-size signal.

## 6. Code Touchpoints

```text
src/openpi/picf/core/config.py
  posterior_owner_transport_direct_candidate_assignment
  posterior_owner_transport_direct_candidate_min_score

src/openpi/picf/core/pipeline.py
  _posterior_owner_transport_measurement
  _posterior_update

src/openpi/picf/core/contracts.py
  owner_transport_dist_after_fusion

scripts/picf_core_train.py
  CLI flags, overlay JSON, debug metrics

src/openpi/picf/core/pipeline_test.py
  test_posterior_owner_transport_uses_direct_graph_candidate_write_through
```

## 7. Validation Criteria

Short 300-step validation should pass:

```text
posterior_owner_transport_active_dist_after_fusion_mean:
  should be materially below the pre-fusion distance on owner-active frames.

object_candidate_owner_geometry_active_dist_mean:
  should stay near zero on inspected sidecar/contact frames.

aqr_active_same_role_support_overlap_max:
  should not rebound into the old collapse band.

posterior_file_competition_active_duplicate_overlap_max:
  should remain zero.

overlays:
  active owner should sit on the task/contact object; gray context rows can keep
  dense scene capacity but must not become action-visible duplicate owners.
```

## 8. What This Does Not Add

```text
no blind SAM;
no hard VQ posterior truth;
no online IsSameObject supervised loss from weak masks;
no RGB reconstruction decoder in the action path;
no permanent gripper owner file competing with the manipulated object.
```

Those remain rejected or deferred because they either need cleaner labels, weaken
modality-missing scaling, or turn the belief router into a reconstruction model.

## 9. 2026-05-20 Local Strict Re-Audit Update

The full local `pipeline_test.py + training_test.py` pass found two contract
details that were not visible in the narrower smoke tests:

```text
1. Evidence cache validity:
   current production cache writes only active posterior files.  Reserve/context
   rows may be kept as typed graph context but are not required cache rows.
   The cache test now asserts non-empty active writes and next-step cache reads,
   not all persistent rows being valid.

2. Binding-consistency temporal matching:
   the previous implementation used current-slot weights on the backward
   future-slot term.  That made the guarded loss slightly order-sensitive when
   detached future slots were permuted.  The repair separates current and future
   weights: forward current->future uses current alpha; backward future->current
   uses future support weights.  This keeps the loss permutation-tolerant before
   it is enabled in production.
```

An additional debug-only issue was fixed:

```text
owm_proposal_tokens / owm_proposal_valid_fraction are logged whenever proposal
tokens exist, independent of tracklet tokens.  This prevents proposal-only
sidecar diagnostics from producing false missing-data readings.
```

Validation after these repairs:

```text
py_compile:
  PASS

pipeline_test.py + training_test.py:
  133 passed

picf_latest_slot_deployment_audit.py --fail-on-fail:
  14/14 PASS

picf_object_candidate_slot_binding_audit.py:
  ok=true, 37 checks

verify_picf_owm_contract.py:
  PASS

picf_owm_dataflow_trace.py --fail-on-fail:
  ok=true

picf_owm_mvtrack_deep_audit.py --fail-on-fail:
  PASS

git diff --check:
  PASS
```

## 10. 2026-05-20 Owner-Direct Final 300-Step Validation

Run:

```text
picf_a7_owner_direct_final_smoke300_20260520
script:
  scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh

scope:
  frozen-policy slot validation, not action cotrain
  sonata/vjepa/anytouch pretrained branches frozen
  PaliGemma/action pressure frozen or zero-weighted for this diagnostic
  PICF slot/router/posterior/OEML path trainable
  sidecar proposal + tracklet evidence enabled
  SAM disabled
  direct candidate owner transport enabled
```

Observed trajectory through step 200:

```text
step 50:
  loss_total                                  0.17531
  loss_anchor_object_pull                    0.35183
  loss_object_explanation_point              2.20806
  loss_object_explanation_duplicate          0.06125
  active / context / downstream / raw overlap
    0.00066 / 0.25628 / 0.34246 / 0.46303
  posterior active duplicate overlap          0.00000
  owner active dist after fusion              0.00411 m
  proposal / tracklet tokens                  1.415 / 80.525

step 100:
  loss_total                                  0.14806
  loss_anchor_object_pull                    0.27917
  loss_object_explanation_point              2.11454
  loss_object_explanation_duplicate          0.05390
  active / context / downstream / raw overlap
    0.00000 / 0.31957 / 0.32772 / 0.54387
  posterior active duplicate overlap          0.00000
  owner active dist after fusion              0.00446 m
  proposal / tracklet tokens                  1.505 / 79.760

step 150:
  loss_total                                  0.15717
  loss_anchor_object_pull                    0.29885
  loss_object_explanation_point              2.22975
  loss_object_explanation_duplicate          0.05145
  active / context / downstream / raw overlap
    0.00696 / 0.43278 / 0.43967 / 0.85463
  posterior active duplicate overlap          0.00000
  owner active dist after fusion              0.00510 m
  proposal / tracklet tokens                  1.460 / 79.810

step 200:
  loss_total                                  0.18040
  loss_anchor_object_pull                    0.37486
  loss_object_explanation_point              2.04953
  loss_object_explanation_duplicate          0.04655
  active / context / downstream / raw overlap
    0.01124 / 0.43734 / 0.44363 / 0.91260
  posterior active duplicate overlap          0.00000
  owner active dist after fusion              0.00415 m
  proposal / tracklet tokens                  1.410 / 80.805

step 250:
  loss_total                                  0.17601
  loss_anchor_object_pull                    0.34864
  loss_object_explanation_point              2.30122
  loss_object_explanation_duplicate          0.04655
  active / context / downstream / raw overlap
    0.00461 / 0.46825 / 0.47050 / 0.91392
  posterior active duplicate overlap          0.00000
  owner active dist after fusion              0.00493 m
  proposal / tracklet tokens                  1.470 / 77.675

step 300:
  loss_total                                  0.20884
  loss_anchor_object_pull                    0.47031
  loss_object_explanation_point              1.82275
  loss_object_explanation_duplicate          0.04410
  active / context / downstream / raw overlap
    0.00744 / 0.48514 / 0.48680 / 0.94611
  posterior active duplicate overlap          0.00000
  owner active dist after fusion              0.00532 m
  proposal / tracklet tokens                  1.505 / 79.930
```

Interpretation after the 300-step smoke:

```text
The old active-posterior collapse is not reproduced:
  active same-role support overlap remains near zero;
  active duplicate overlap remains zero;
  owner geometry closes after fusion in millimeters;
  proposal and tracklet evidence are both present.

The raw all-row overlap still rises because reserve/context capacity remains
dense.  This is not the acceptance metric for object ownership, but it must stay
reported because it can still affect downstream graph context if context
attention is weighted too strongly.

The object losses are batch-dependent.  The point explanation and duplicate
terms improve by step 300, while anchor_object_pull is worse on the final
push-switch batch.  Therefore this smoke validates the write-through / active
ownership contract, but it does not prove that every task's owner center is
already visually perfect.  That still requires overlay inspection and a longer
action-cotrain run.
```

Acceptance verdict for this smoke:

```text
PASS for:
  direct owner write-through dataflow;
  active posterior duplicate suppression;
  millimeter-range active owner closure after fusion;
  proposal + tracklet sidecar presence;
  no role-0 effector leakage into active object ownership.

OPEN for:
  raw reserve/context overlap;
  per-task overlay quality;
  full action-cotrain behavior evidence.
```
