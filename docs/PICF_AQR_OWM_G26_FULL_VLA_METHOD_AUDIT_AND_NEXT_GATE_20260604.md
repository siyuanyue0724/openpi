# PICF-AQR-OWM G26 Full VLA Method Audit and Next Gate

Date: 2026-06-04

Entry point:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_G24_ALL_METHOD_EXECUTION_LEDGER_20260604.md
docs/PICF_AQR_OWM_G25_DEPLOYED_FLOW_CONTEXT_BRIDGE_20260604.md
```

Purpose:

```text
Convert every requested 2025-2026 VLA training/architecture method into a
math/code/history/runtime checklist, and prevent another ungrounded "just
increase batch" or "just reset optimizer" cycle.
```

This document is intentionally stricter than a run note.  A method is allowed
to consume GPU only if it passes:

```text
1. math follow-through:
   It optimizes a defensible objective and does not hide the real action loss.

2. code follow-through:
   CLI/config -> dataflow -> model/loss -> metrics -> checkpoint path is
   identified.

3. history follow-through:
   The same hypothesis was not already rejected on the current code family.

4. scaling follow-through:
   It remains compatible with large heterogeneous future data, including
   missing modalities and new embodiments.
```

## 1. Root Problem Statement

The target multi-task objective is:

```text
L(theta) = sum_b q_b L_b(theta)
G(theta) = sum_b q_b grad_theta L_b(theta)
```

where `b` is a task/data bucket.  A small random physical batch observes:

```text
g_step = grad_theta L_b(theta)
Var[g_step] = E_b ||grad L_b - G||^2
```

Therefore balanced logical batches are necessary.  They are not sufficient in
the current code family:

```text
G10-G24:
  task coverage, per-bucket scaling, temperature/ratio/dynamic mixing,
  scoped PCGrad/CAGrad, action LR/PICF LR, optimizer reset, context suffix,
  router proxy, and side action readout were tested.

Observed boundary:
  PICF context C_picf is motor-readable in a direct readout, but the native
  PI0.5 action flow did not reliably consume it.
```

The concrete causal gap is:

```text
d A_native / d C_picf is too weak
```

where:

```text
A_native = native PI0.5 continuous flow action path
C_picf   = bounded PICF action-context tokens
```

## 2. Paper-to-Code Method Ledger

Primary sources tracked here:

```text
VLA Foundry, 2026:
  probabilistic data mixing, batch balancing ratios, gradient accumulation,
  per-dataset normalization.
  https://arxiv.org/abs/2604.19728

ABot-M0, 2026:
  multi-source robot data cleaning, balanced task/embodiment sampling, action
  standardization.
  https://arxiv.org/abs/2602.11236

PiKE, 2025:
  dynamic data mixing when task gradients conflict.
  https://arxiv.org/abs/2502.06244

OpenVLA-OFT, 2025:
  continuous action chunks, parallel decoding, L1/Huber-style continuous action
  fine-tuning.
  https://arxiv.org/abs/2502.19645

Knowledge Insulation / pi0.5, 2025:
  action expert gradients can harm VLM knowledge; action boundary and
  representation supervision matter.
  https://arxiv.org/abs/2505.23705

AdaMoE-VLA / FedVLA, 2025:
  action-specialized MoE and task/embodiment-aware expert routing.
  https://arxiv.org/abs/2510.14300
```

| Requested method | Local code status | History status | G26 decision |
| --- | --- | --- | --- |
| Task/dataset mixing | `_calvin_prompt_bucket`, `_bucket_sequence_for_logical_step` | G10/G12/G18/G22 tested | Keep mandatory; do not repeat alone. |
| Batch balancing ratios | `--calvin-bucket-weight-spec` | G12-RATIO tested | Keep knob; not root fix. |
| Temperature sampling | `_compute_bucket_sampling_weights(... temperature_alpha)` | G12-TEMP tested | Keep knob; not root fix. |
| Task-uniform / trajectory-uniform | same sampler path | G12 compared | Task-uniform remains default. |
| Without-replacement logical update | `_bucket_sequence_for_logical_step(... without_replacement)` | current G25 uses it | Keep. |
| Gradient accumulation as logical batch | `world_size * accum_steps` sequence length | K4/K8/K10/K12 attempted | Keep K4 on 2xA100 unless memory changes. |
| Per-bucket loss normalization | `_logical_batch_loss_scales` | audited and used | Keep mandatory. |
| Per-bucket action EMA scale | `_logical_action_bucket_scale` | tested, insufficient | Optional diagnostic only. |
| Dynamic PiKE-style mixing | `_dynamic_bucket_sampling_weights` | G12-DYN tested | Implemented; not the current branch. |
| PCGrad | `_pcgrad_project_and_sum` | G12-PCGrad tested | Diagnostic only. |
| CAGrad | `_cagrad_project_and_sum` | G12-CAGrad tested | Diagnostic only. |
| Whole-model PCGrad/CAGrad | intentionally absent | rejected by cost/root-cause mismatch | Do not deploy without new gradient-cosine proof. |
| Continuous action chunks | native PI0.5 flow action path | always present | Already deployed. |
| L1/Huber action objective | `--semantic-action-flow-loss` | G15 tested | Optional; not root fix. |
| Modality projectors/adapters | PICF typed modality projectors + semantic wrapper adapters | deployed | Keep. |
| Context suffix bridge | `_apply_action_context_adapter` | G17/G20 tested | Keep but insufficient alone. |
| Context-only action readout | `_compute_action_context_readout_aux` | G22 proved motor-readability | Diagnostic; not deployed-action fix. |
| Deployed flow residual | `_apply_action_context_flow_residual` | G25 active | Current only non-duplicate deployed-path test. |
| Action expert router/MoE proxy | `_apply_action_expert_router` | G16 tested | Keep off; full MoE deferred. |
| Direct FAST/action-token CE | native OpenPI FAST exists; PICF drops LM heads | not implemented in PICF | Next real branch if G25 fails. |
| Embodiment adapters/heads | mostly future hook | CALVIN single embodiment | Defer to heterogeneous robot data. |
| System2/System1 subtask split | no reliable labels in current stream | theory only | Defer; not 2-hour root fix. |

## 3. Code Follow-Through Anchors

Sampler / logical batch:

```text
scripts/picf_core_train.py
  _compute_bucket_sampling_weights
  _dynamic_bucket_sampling_weights
  _bucket_sequence_for_logical_step
  _logical_batch_loss_scales
  _logical_action_bucket_scale
```

Gradient conflict:

```text
scripts/picf_core_train.py
  _logical_batch_gradient_surgery_params
  _pcgrad_project_and_sum
  _cagrad_project_and_sum
  _assign_and_sync_gradient_surgery_grads
```

Action/context boundary:

```text
src/openpi/picf/paligemma/config.py
  action_context_adapter_*
  action_context_readout_aux_*
  action_context_flow_residual_*
  action_expert_router_*

src/openpi/picf/paligemma/wrapper.py
  _apply_action_context_adapter
  _predict_action_chunk_from_context
  _compute_action_context_readout_aux
  _apply_action_context_flow_residual
  _apply_action_expert_router
  compute_action_flow_loss
  sample_action_chunk

src/openpi/picf/policy.py
  maps all wrapper metrics to PI debug keys.
```

FAST/action-token branch:

```text
src/openpi/models/tokenizer.py
src/openpi/transforms/_base.py
src/openpi/models/pi0_fast.py
src/openpi/picf/paligemma/wrapper.py::_drop_unused_generation_heads
```

Important current fact:

```text
PICF's wrapper deliberately drops unused LM heads because the current live path
uses hidden states plus continuous flow action, not FAST generation loss.
Therefore "direct FAST/action-token CE" is not currently deployed in PICF.  It
cannot be honestly marked done until the generation/action-token objective is
reintroduced with a live loss and metrics.
```

## 4. Current G25 Runtime Gate

Remote:

```text
ssh -p 26120 root@px-cloud1.matpool.com
repo: /root/openpi_g25_20260604
session: picf_g25_flowresidual_c2_300_20260604
log: /mnt/picf_run_logs/picf_g25_flowresidual_c2_300_20260604.log
```

Current structured trace through step 11170:

```text
loss_action_default_equiv:
  11010 = 0.046894
  11170 = 0.041966
  min   = 0.039342
  max   = 0.054716
  last5 mean  ~= 0.042642
  last10 mean ~= 0.045540

pi_context_flow_gain_mse_delta:
  positive in 14/17 logged points
  last5 mean  ~= +0.003792
  last10 mean ~= +0.002916

structure:
  logical_batch_distinct_bucket_count = 4
  active_same_role_support_overlap <= 0.2375 in latest rows
  loss_anchor_pv ~= 0.49-0.51 because picf_core_lr is intentionally tiny
```

Interpretation:

```text
Resolved:
  PICF context now has positive deployed-flow effect on native PI0.5 flow MSE.

Not resolved:
  The canonical action scalar remains noisy.  This is not yet a 30K convergence
  proof, and not enough to claim all action convergence problems are solved.
```

## 5. What Must Not Be Repeated

Unless code/data materially changes, do not run:

```text
sampler-only task_uniform / temperature / ratio / trajectory
optimizer-reset-only
action-LR-only
PICF-LR-only
PCGrad/CAGrad-only
raw-overlap-only
SAM proposal supervision
sidecar-only mask pull
context-readout auxiliary-only
router-proxy-only
```

Each of these was either implemented and insufficient, or rejected because it
does not target the deployed action-path causal gap.

## 6. Next Experiments

### G26-A: finish G25 deployed-flow gate

Checklist:

```text
[x] code follow-through complete
[x] local unit tests passed in G25
[x] remote run active
[ ] reach step 11300
[ ] compute final action/gain trend
[ ] archive final result into this document and G25
```

Pass:

```text
gain_delta remains positive on average and action last-window mean is below the
G24 comparable plateau.
```

Fail:

```text
gain_delta loses positivity or action returns to the 0.05+ plateau.
```

Decision:

```text
If pass:
  start a longer G25 continuation; do not switch to another sampler-only test.

If fail or inconclusive:
  start G26-B, direct FAST/action-token representation supervision audit and
  implementation plan.  Do not spend more GPU on old sampler/optimizer branches.
```

### G26-B: direct FAST/action-token CE restoration

Purpose:

```text
Test the Knowledge-Insulation action-representation branch that is still absent
from PICF.  This is the only major requested method not yet deployed.
```

Required before GPU:

```text
[ ] inspect native OpenPI FAST tokenization and PI0_FAST loss path
[ ] decide whether PICF can restore LM/action-token heads without reintroducing
    dead FSDP parameters
[ ] add live loss metrics:
    loss_fast_action_token
    loss_fast_action_token_weighted
    fast_action_token_accuracy or token_nll
[ ] ensure continuous flow action remains canonical for action comparison
[ ] py_compile and targeted unit tests
```

Non-goal:

```text
Do not replace the whole VLM, whole action expert, or all PICF slots.  The
missing piece is representation supervision at the action boundary.
```

### G26-C: full action-expert MoE

Only after G26-B or a new gradient-cosine audit proves task-family conflict:

```text
[ ] expert routing inside action expert only
[ ] no whole-backbone MoE
[ ] per-task/bucket router telemetry
[ ] compare against G25/G26-B, not against sampler-only baselines
```

## 7. Current Answer to "Have All Methods Been Tried?"

Strict answer:

```text
No.
```

Accurate breakdown:

```text
Tried/implemented:
  data mixing, logical batch, per-bucket normalization, temperature/ratio,
  dynamic mixing, scoped PCGrad/CAGrad, continuous chunks, L1/Huber, context
  suffix, router proxy, context readout, deployed flow residual.

Not deployed:
  direct PICF-side FAST/action-token CE.

Deferred for valid reasons:
  embodiment adapters/heads: current CALVIN gate has one embodiment.
  System2/System1: no reliable subtask labels in current stream.
  full action-expert MoE: high-cost branch; should follow direct boundary
  supervision or proven gradient conflict.
```

This is the working rule for the next 2-3 hours:

```text
Finish G25.  If it is not enough, implement/audit FAST/action-token CE.  Do not
repeat already rejected data-mixing or optimizer-only tests.
```
