# Slot-JEPA Normalized Diagnostic TODO

Date: 2026-05-23

This note tracks the local fix for the large raw `loss_slot_jepa` telemetry
observed in the A7 action-dominant 30K run.  It is intentionally a diagnostic
and TODO document; it does not declare the predictive OWM loss ready for
training.

## Problem

The previous `loss_slot_jepa` metric was a raw MSE between current predictive
slot tokens and a softly matched detached future posterior target:

```text
A_ij = softmatch(1 - cos(p_i, z_j^+))
z_i^match = sum_j A_ij z_j^+
loss_slot_jepa_raw = MSE(p_i, z_i^match) plus symmetric MSE
```

The matching cost used normalized cosine, but the logged loss used unnormalized
token values.  Therefore a large raw value conflated two different failures:

```text
direction failure:
  p_i points at the wrong future posterior identity.

scale failure:
  p_i and z_i^match point in a similar direction, but their norms drift.
```

The 2026-05-23 A7 log showed raw `loss_slot_jepa` rising to very large values
while `lambda_slot_jepa=0`.  Because the loss was disabled, this did not directly
drive gradients.  It is still a valid warning that the predictive latent branch
is not calibrated.

## Local Fix

The code now keeps the existing `loss_slot_jepa` field for compatibility, but
adds scale-separated diagnostics:

```text
loss_slot_jepa_direction
  mean(1 - cos(p_i, z_i^match))

loss_slot_jepa_log_norm
  SmoothL1(log ||p_i|| - log ||z_i^match||)

loss_slot_jepa_pred_norm
  mean ||p_i||

loss_slot_jepa_target_norm
  mean ||z_j^+||

loss_slot_jepa_matched_target_norm
  mean ||z_i^match||
```

These fields are logged only.  They do not enter `loss_total` unless a future
config explicitly introduces a new normalized slot-JEPA objective.

## Interpretation Gates

Use the diagnostics as follows:

```text
raw high, direction low, log_norm high:
  scale calibration problem; do not treat as identity failure.

raw high, direction high:
  predictive slot identity / temporal matching problem.

raw high, pred_norm or target_norm exploding:
  latent norm drift; inspect posterior_residual_summary_norm and residual
  context scale before enabling any predictive objective.

direction stable and log_norm stable for a long run:
  candidate condition for a small normalized predictive auxiliary.
```

## TODO Before Enabling Predictive OWM Loss

1. Run a 300-1000 step diagnostic with the new fields present in the log.
2. Verify `loss_slot_jepa_direction` is not monotonically increasing.
3. Verify `loss_slot_jepa_log_norm` is bounded and not tracking raw MSE spikes.
4. Compare `pred_norm`, `target_norm`, and `posterior_residual_summary_norm`.
5. If direction is healthy but norm drift remains, add an explicit latent norm
   calibration or detached target normalization before enabling slot-JEPA.
6. If direction is unhealthy, keep `lambda_slot_jepa=0` and debug temporal
   identity matching before any predictive loss is allowed.

## Current Policy

```text
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
```

This policy remains correct for production-length action runs until normalized
diagnostics demonstrate stable future-posterior prediction.
