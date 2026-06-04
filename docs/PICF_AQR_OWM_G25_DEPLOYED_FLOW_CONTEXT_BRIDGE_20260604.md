# PICF-AQR-OWM G25 Deployed Flow Context Bridge

Date: 2026-06-04

Entry point:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_G24_ALL_METHOD_EXECUTION_LEDGER_20260604.md
```

Purpose:

```text
Close the remaining G24 causal gap:
  PICF context is motor-readable in a side readout,
  but it has weak effect on native PI0.5 flow action.
```

This is not another sampler-only or optimizer-only run.  Those branches are
already covered by G10-G24 and are kept as infrastructure, not repeated as the
current root fix.

## 1. Method Ledger

The requested 2025-2026 VLA method families are accounted for as follows.

| Method | Code/history status | G25 decision |
| --- | --- | --- |
| Task/dataset mixing | Implemented through CALVIN bucket/task logical batching. Tested in G10/G12/G18/G22. | Keep mandatory. Not sufficient alone. |
| Batch balancing ratios | `--calvin-bucket-weight-spec`, tested in G12-RATIO. | Keep knob. Do not repeat as root fix. |
| Temperature sampling | Tested in G12-TEMP. | Keep knob. Do not repeat as root fix. |
| Dynamic PiKE-style mixing | Implemented and tested in G12-DYN. | Keep optional. Not current root. |
| Gradient accumulation / logical batch | Implemented by logical-batch micro-steps. | Keep. Necessary but not sufficient. |
| Per-task/per-bucket normalization | `_logical_batch_loss_scales`, bucket action EMA. | Keep. Not current root. |
| PCGrad/CAGrad | Scoped implementation tested in G12. | Diagnostic only unless new gradient-cosine evidence appears. |
| Continuous action chunk | Native PI0.5 flow already uses chunked continuous actions. | Already present. |
| L1/Huber action objective | Tested in G15. | Keep optional; not sufficient. |
| Modality adapters | PICF modality projectors and action context adapter exist. | Keep. Not sufficient because native action effect is weak. |
| Action expert router/MoE proxy | Implemented in G16. | Keep off by default; full MoE deferred until native context causality is fixed. |
| Action/backbone insulation | Trainable scopes exist: backbone/model/action_head/action_adapter/head+adapter. | Keep. Not sufficient alone. |
| System2/System1 subtask split | No reliable CALVIN subtask labels in current training stream. | Defer. Would be a new labeled objective, not a 2-hour root fix. |
| Direct FAST/action-token CE | Current PI0.5 wrapper drops unused generation heads. | Not deployed in G25. Correct future branch if flow residual fails. |
| Deployed context flow residual | New in G25. | Run now. This is the unclosed causal path. |

## 2. Mathematical Contract

PI0.5 action flow training samples:

```text
x_t = t eps + (1 - t) y
u_t = eps - y
```

where:

```text
y     = target action chunk
eps   = Gaussian noise
u_t   = target flow velocity
v_base(x_t, t, prefix) = native PI0.5 predicted flow velocity
C     = bounded PICF belief/action context tokens
```

G22 showed a context-only readout can learn:

```text
R_eta(C) -> y
```

but that was a side auxiliary.  It did not change the deployed native velocity
`v_base`, so `loss_action_default_equiv` could still plateau or worsen.

G25 converts the same readout into a deployed flow residual:

```text
y_c = R_eta(C)
u_c = (x_t - y_c) / max(t, t_min)
r_c = cap_rms(u_c - v_base)
v    = v_base + sigmoid(g) r_c
```

The canonical action loss remains:

```text
L_action = ||u_t - v||_2^2
```

This preserves PI0.5 semantics:

```text
If C is useless:
  sigmoid(g) and RMS cap bound the harm.

If C is motor-readable:
  v moves toward a flow-consistent velocity, so canonical action MSE can improve.
```

The same residual is used in:

```text
training: compute_action_flow_loss(...)
sampling : sample_action_chunk(...)
```

Therefore this is a deployed-path test, not an auxiliary-only diagnostic.

## 3. Code Follow-Through

Expected code path:

```text
CLI
  --semantic-action-context-flow-residual-enabled
  --semantic-action-context-flow-residual-gate-init
  --semantic-action-context-flow-residual-time-floor
  --semantic-action-context-flow-residual-rms-cap

scripts/picf_core_train.py
  argparse -> validation -> PaliGemmaSemanticConfig -> debug metric aggregation

src/openpi/picf/paligemma/config.py
  action_context_flow_residual_* fields

src/openpi/picf/paligemma/wrapper.py
  _predict_action_chunk_from_context
  _apply_action_context_flow_residual
  compute_action_flow_loss
  sample_action_chunk

src/openpi/picf/policy.py
  maps wrapper metrics into PI debug fields
```

Required metrics:

```text
pi_context_flow_residual_enabled
pi_context_flow_residual_gate
pi_context_flow_residual_token_count
pi_context_flow_residual_rms_mean
pi_context_flow_context_velocity_rms_mean
pi_context_flow_context_target_mse
pi_context_flow_residual_time_floor
pi_context_flow_base_mse
pi_context_flow_adapted_mse
pi_context_flow_gain_mse_delta
```

## 4. Experiment Gate

Short gate:

```text
Name      : G25-C2 deployed flow residual
Machine   : 2xA100
Resume    : 11000-step G24 checkpoint
Steps     : 300 optimizer steps
Sampler   : task-uniform logical batch K=4, bucket normalization on
Scope     : action_head_and_adapter
Readout   : side readout aux weight 0.0
Residual  : enabled, gate_init=-2.0, time_floor=0.05, RMS cap on
```

Pass indicators:

```text
pi_context_flow_residual_enabled = 1
pi_context_flow_gain_mse_delta >= 0 on average after warmup, or trends positive
loss_action_default_equiv improves versus G24 C1/G22 comparable baseline
No NaN/Inf in residual metrics
```

Fail indicators:

```text
gain_mse_delta remains negative
loss_action_default_equiv worsens faster than baseline
residual target MSE remains high and gate cannot compensate
```

If G25-C2 fails, the next branch is not another sampler or optimizer run.  The
next real architecture branch is direct PI0.5 action-token/FAST-style
representation supervision or a full action expert replacement, because G22
already proved side motor-readability without native action improvement.

## 5. Anti-Repeat Rules

Do not spend another 1-2 hour gate on these unless the code has materially
changed:

```text
sampler-only task uniform / ratio / temperature
optimizer reset only
action LR only
PICF LR only
PCGrad/CAGrad only
raw overlap only
SAM proposal supervision
sidecar-only mask pull
context readout auxiliary only
router/MoE proxy only
```

They either already improved structure without action, or improved a side
diagnostic without deployed action.
