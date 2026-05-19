# PICF-AQR-OWM Binding Logit Calibration Audit - 2026-05-15

This is a temporary follow-through note linked from `src/openpi/picf/README_v2.2.md`.

## Problem Statement

The A7 non-truncated binding diagnostic showed a real model-level gap:

```text
posterior_binding_signature_linear_score_abs_mean     ~= 0.465
posterior_binding_signature_quadratic_score_abs_mean  ~= 0.465
posterior_binding_signature_low_rank_score_abs_mean   ~= 0.00026
posterior_identity_switch_rate                        ~= 0.70
aqr_same_role_support_overlap_max                     rose from 0.23 to 0.61 by step100
```

The diagonal quadratic score was numerically equivalent to the linear cosine
score because `binding_quadratic_diag` was initialized as `sqrt(D)`. The
low-rank score was effectively inactive because both low-rank projections were
initialized with tiny Gaussian weights. Therefore the runtime path had the
right object-binding vocabulary, but it did not implement a calibrated
IsSameObject logit.

## Paper-Code Reference

The external object-binding code (`/tmp/vit-object-binding`) implements
IsSameObject as a trained pairwise logit:

```text
DiagonalQuadraticProbe:
  logit(i,j) = Linear(h_i * h_j)

QuadraticProbe:
  logit(i,j) = h_i^T W_sym h_j + b

QuadraticFixedRankProbe:
  W_sym = (W1^T W2 + W2^T W1) / 2
  logit(i,j) = h_i^T W_sym h_j + b
```

The probe is trained with `BCEWithLogitsLoss` against pairwise instance-mask
labels. The important point for PICF is not the mask label itself; it is that
the same-object score is a calibrated pairwise logit with scale and bias. A raw
positive cosine/common-mode score is not sufficient evidence of identity.

## PICF Runtime Constraint

PICF cannot assume mask labels in CALVIN or in future large heterogeneous data.
The runtime repair therefore must not introduce a supervised mask loss. It must
convert unlabeled pairwise scores into assignment evidence only when the current
pairwise matrix contains relative structure.

## Mathematical Repair

Let

```math
S_raw =
  w_l S_linear
  + w_d S_diag_quadratic
  + w_r S_low_rank_quadratic
```

where

```math
S_linear(i,j) = cos(b_i^-, b_j^{obs})
```

and the quadratic terms follow the external paper-code probe family.

PICF now applies relative logit calibration:

```math
H_r(S) = S - rowmean(S)
H_c(S) = S - colmean(S)
S_center = S_raw - rowmean(S_raw) - colmean(S_raw) + mean(S_raw)
```

Then:

```math
if std(S_center) < sigma_min:
    S_cal = 0
else:
    S_cal = clip(S_center / std(S_center), -c, c)
```

The binding logit update is:

```math
L_bind += gate(alpha, recycle, innovation) * S_cal
```

This is a calibrated assignment logit, not a new auxiliary loss.

## Why This Is Not a Patchwork Penalty

The previous failure was not caused by a missing scalar penalty. It was caused
by using uncalibrated pairwise scores as if they were IsSameObject logits.
Double-centering removes row/column common terms that do not affect relative
assignment. The dispersion gate prevents amplifying noise when signatures have
no real separability. Orthogonal low-rank initialization makes the low-rank
paper-code family numerically live from the first step without assuming labels.

## Code Follow-Through

Implemented in:

```text
src/openpi/picf/core/config.py
  binding_signature_score_calibration_enabled
  binding_signature_score_calibration_mode
  binding_signature_score_min_std
  binding_signature_score_clip

src/openpi/picf/core/pipeline.py
  orthogonal low-rank initialization
  _calibrate_pairwise_binding_score
  calibrated combined binding score in _binding_logits
  posterior_binding_signature_calibrated_* debug metrics

src/openpi/picf/core/contracts.py
  PicfPosteriorAnchorState calibrated binding diagnostics

scripts/picf_core_train.py
  CLI/config threading and metrics logging

scripts/verify_picf_owm_contract.py
  static contract guard

scripts/picf_owm_evidence_bundle.py
  evidence bundle metric/arg keys
```

## Acceptance Metrics

The next diagnostic must check:

```text
posterior_binding_signature_low_rank_score_abs_mean is no longer near zero
posterior_binding_signature_calibrated_score_std is nonzero only when separable
posterior_binding_signature_calibrated_top1_margin_mean is positive
aqr_active_same_role_support_overlap_max does not rise toward 1.0
posterior_identity_switch_rate does not stay around 0.7
posterior_recycle_rate remains non-saturated
```

This repair does not claim to solve tracklet/proposal missing data or ordinal
rank supervision. It addresses the specific paper-code mismatch exposed by the
A7 diagnostic: runtime binding scores were structurally present but not
calibrated as pairwise same-object logits.

## Strict Verification Snapshot

Validated locally on 2026-05-15 after the calibrated A7 diagnostic was already
running:

```text
PYTHONPATH=src python scripts/picf_binding_logit_calibration_audit.py --fail-on-fail
  PASS

PYTHONPATH=src python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail
  PASS

PYTHONPATH=src python scripts/picf_owm_professor_grade_audit.py --fail-on-fail
  PASS

PYTHONPATH=src python scripts/verify_picf_owm_contract.py
  PASS

PYTHONPATH=src uv run pytest -q \
  src/openpi/picf/core/pipeline_test.py::test_binding_signature_centering_removes_common_mode \
  src/openpi/picf/core/pipeline_test.py::test_binding_signature_quadratic_scores_are_pairwise_not_plain_cosine \
  src/openpi/picf/core/pipeline_test.py::test_binding_signature_score_calibration_drops_common_mode \
  src/openpi/picf/core/pipeline_test.py::test_binding_signature_score_calibration_keeps_relative_pairs \
  src/openpi/picf/core/pipeline_test.py::test_posterior_lifecycle_calibration_protects_stable_supported_slots \
  scripts/picf_owm_same_object_probe_test.py
  8 passed
```

The test scope is intentionally narrow: it verifies the paper-code mapping,
runtime dataflow, and score-calibration math. It does not claim CALVIN behavior
acceptance.
