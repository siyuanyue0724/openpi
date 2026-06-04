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

## 5. Deployment / Validation Log

Code commits:

```text
ceb9c0d Add deployed PICF context flow residual
4687f6a Allow flow residual gate in checkpoint migration
```

Local validation:

```text
python -m py_compile:
  src/openpi/picf/paligemma/config.py
  src/openpi/picf/paligemma/wrapper.py
  src/openpi/picf/policy.py
  scripts/picf_core_train.py
  scripts/picf_core_train_test.py
  scripts/picf_resume_from_args_json.py

uv run pytest -q src/openpi/picf/paligemma/wrapper_test.py \
  -k 'action_context_flow_residual or trainable_scope'
  -> 10 passed

uv run pytest -q src/openpi/picf/policy_test.py \
  -k 'route_context_to_action_side_adapter'
  -> 1 passed

uv run pytest -q scripts/picf_core_train_test.py \
  -k 'load_state_dict_picf_compat_allows_new_action_context_flow_and_router_only or load_state_dict_picf_compat_skips_shape_mismatches'
  -> 2 passed
```

Remote sync:

```text
Machine    : px-cloud1:26120, 2xA100
Repo       : /root/openpi_g25_20260604
Sync path  : GitHub mirror https://gh-proxy.com/https://github.com/siyuanyue0724/openpi.git
Branch     : Posterior_VLA
Commit     : 7485a56
```

Remote validation:

```text
py_compile scripts/picf_core_train.py scripts/picf_core_train_test.py
  -> PASS

pytest scripts/picf_core_train_test.py compatibility selector
  -> 2 passed
```

Checkpoint migration note:

```text
The 11000-step source checkpoint predates:
  action_context_readout_*
  action_expert_router_*
  action_context_flow_residual_gate_logit

The compatibility loader intentionally allows only these narrow new semantic
action-context/router keys to initialize from the current model.  Any unrelated
missing key still fails.
```

Sanity gate:

```text
Run   : picf_g25_flowresidual_sanity_k2_2step_20260604
Mode  : single-card, logical_batch_task_count=2, accum_steps=2
Status: PASS, reached optimizer steps 11001-11002

Initial residual diagnostic at step 11002:
  pi_context_flow_residual_enabled = 1
  pi_context_flow_residual_gate ~= 0.119
  pi_context_flow_base_mse ~= 0.0415
  pi_context_flow_adapted_mse ~= 0.0539
  pi_context_flow_gain_mse_delta ~= -0.0124

Interpretation:
  Startup and deployed-path metrics are working.  The randomly initialized
  residual gate is initially harmful on this tiny sanity sample, so the C2
  300-step run must verify whether it learns a positive contribution.
```

Active remote diagnostic:

```text
Run     : picf_g25_flowresidual_c2_from11000_to11300_20260604
Session : picf_g25_flowresidual_c2_300_20260604
Log     : /mnt/picf_run_logs/picf_g25_flowresidual_c2_300_20260604.log
Mode    : 2-card DDP, logical_batch_task_count=4, accum_steps=2
Status  : running; verified compatibility migration and first 100 optimizer
          steps after resume
```

C2 deployed-flow metric trace:

```text
step   total     action_def  base_mse  adapt_mse  gain_delta
11010  0.10449   0.04689     0.04143   0.04689   -0.00546
11020  0.10304   0.04674     0.04394   0.04674   -0.00280
11030  0.09895   0.04486     0.04484   0.04486   -0.00002
11040  0.10943   0.04985     0.05004   0.04985    0.00019
11050  0.11354   0.05110     0.05177   0.05110    0.00067
11060  0.10627   0.04758     0.04964   0.04758    0.00206
11070  0.09079   0.04048     0.04252   0.04048    0.00204
11080  0.11495   0.05133     0.05384   0.05133    0.00251
11090  0.10579   0.04816     0.04995   0.04816    0.00178
11100  0.11939   0.05472     0.05581   0.05472    0.00109
```

Interpretation at step 11100:

```text
The residual starts harmful because the new gate/readout parameters are missing
from the source checkpoint and initialize cold.  By 11030 it is nearly neutral,
and by 11040-11100 it stays positive on the deployed action objective:

  adapted_mse < base_mse
  gain_delta > 0

This is the first direct evidence that PICF context can improve the native
PI0.5 flow path rather than only a side readout.  It is not a 30K convergence
claim yet; the remaining gate is to verify the positive delta persists through
the full 300-step C2 run and then in a longer run.  At 11100 the C2 short
criterion is met: the residual has delivered seven consecutive positive
structured-log points after the initial cold-start phase.
```

## 6. Anti-Repeat Rules

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
