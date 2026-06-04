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

Final structured trace through step 11300:

```text
loss_action_default_equiv:
  11010 = 0.046894
  11300 = 0.051510
  min   = 0.037384
  max   = 0.054716
  last5 mean  ~= 0.045976
  last10 mean ~= 0.045739
  last20 mean ~= 0.044115
  last30 mean ~= 0.045467

pi_context_flow_gain_mse_delta:
  last5 mean  ~= +0.006331
  last10 mean ~= +0.006120
  last20 mean ~= +0.005056
  last30 mean ~= +0.003439

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

Latest stricter judgment:
  G25 demonstrates that the deployed residual bridge is locally useful: the
  adapted flow MSE is lower than the native base flow MSE in the latest window.
  It does not yet demonstrate a stable canonical action descent.  Therefore
  sampler-only, LR-only, optimizer-only, or PCGrad/CAGrad-only reruns are still
  disallowed as low-value repeats.

Decision boundary:
  G25-C2 finished with positive deployed-flow gain but without sustained
  canonical action descent.  Do not rerun sampler/optimizer-only branches.
  Move to direct PICF-side FAST/action-token representation supervision or an
  equivalent deployed action-representation target.
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
[x] reach step 11300
[x] compute final action/gain trend
[x] archive final result into this document and G25
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

Final G26-A result:

```text
G26-A is inconclusive/insufficient as a convergence fix.

Positive:
  pi_context_flow_gain_mse_delta stayed positive and grew stronger in the
  final window.

Negative:
  loss_action_default_equiv ended at 0.051510 and the final two rows were
  0.051004 / 0.051510.  This is not sustained action descent.

Action:
  proceed to G26-B.  The next experiment must add a direct action
  representation target that makes PICF context causally useful for action
  prediction, rather than adding another sampler or optimizer wrapper.
```

### G26-B: PICF-local FAST-style action-token CE

Purpose:

```text
Test the Knowledge-Insulation / FAST-style action-representation branch that
was still absent from PICF.  This is the only major requested method not yet
deployed after G10-G25.
```

Native FAST follow-through:

```text
OpenPI native FAST path:
  src/openpi/models/tokenizer.py::FASTTokenizer
  src/openpi/transforms/_base.py::TokenizeFASTInputs
  src/openpi/models/pi0_fast.py::Pi0FAST.compute_loss

PICF PyTorch path:
  src/openpi/picf/paligemma/wrapper.py::_drop_unused_generation_heads

Current constraint:
  PICF deliberately drops PaliGemma/GemmaExpert LM heads because its live
  action path is hidden-state + continuous PI0.5 flow.  Reintroducing native
  FAST LM CE would be a larger generation-head restoration and FSDP contract
  change, not a safe 2-hour gate.

G26-B deployed equivalent:
  Add a PICF-local action-token CE on the same bounded context readout state
  used by G22 continuous readout and G25 deployed flow residual.
```

Mathematical contract:

```text
Given PICF action context C, horizon h, action dimension d:

  r_h = Attn(q_h, C)
  z_{h,d} = Quantize(clip(a_{h,d}, -c, c), K)
  p(z_{h,d} | C) = softmax(W_token r_h)_{d}

  L_token = CE(z, p)
  L_train = L_flow + lambda_token L_token

Canonical comparison remains:

  loss_action_default_equiv = MSE(u_t, v_t)

Therefore the new loss tests whether PICF context contains action-token
information without hiding historical action MSE.
```

Implemented code:

```text
src/openpi/picf/paligemma/config.py
  action_context_token_aux_weight
  action_context_token_aux_bins
  action_context_token_aux_clip

src/openpi/picf/paligemma/wrapper.py
  action_context_token_readout_out_proj
  _action_context_readout_state
  _compute_action_context_token_aux
  compute_action_flow_loss adds token weighted loss into training_total
  DDP safety: token head is trainable only when token aux is enabled, and the
  empty-context branch returns a graph-connected zero so enabled token-head
  parameters are marked used even if one rank has no PICF context on a step.

src/openpi/picf/policy.py
  picf_action_context_token_aux_* -> pi_context_token_aux_* metrics

scripts/picf_core_train.py
  --semantic-action-context-token-aux-weight
  --semantic-action-context-token-aux-bins
  --semantic-action-context-token-aux-clip
  metric logging and config validation
```

Strict diagnostic addendum:

```text
Token classification accuracy alone can be misleading if action labels are
concentrated in a center/no-op bin.  The implementation therefore also logs:

  pi_context_token_aux_label_entropy
  pi_context_token_aux_label_majority_fraction
  pi_context_token_aux_accuracy_over_majority

These metrics do not change the training loss.  They only prevent a false
positive where a token head appears accurate by predicting the majority bin.
```

Local validation:

```text
[x] python -m py_compile scripts/picf_core_train.py
[x] python -m py_compile src/openpi/picf/paligemma/config.py
[x] python -m py_compile src/openpi/picf/paligemma/wrapper.py
[x] python -m py_compile src/openpi/picf/policy.py
[x] uv run pytest -q src/openpi/picf/paligemma/wrapper_test.py \
    -k 'action_context_token_aux or action_context_readout_aux or action_context_flow_residual or trainable_scope'
    -> 14 passed after the DDP-safety regression tests were added
[x] uv run pytest -q scripts/picf_core_train_test.py \
    -k 'action_context or load_state_dict_picf_compat or trainable_scope'
    -> 9 passed
```

Runtime correction after first remote attempt:

```text
Symptom:
  first G26-B 2xA100 K4 launch reached logical-batch setup, then emitted no
  training step while GPU1 was saturated and GPU0 was idle.

Root cause class:
  token aux added a new trainable readout head to the generic action-context
  readout scope.  Under DDP this is unsafe when the branch is disabled or when
  context availability differs by rank, because trainable parameters may be
  unused on one rank.

Fix:
  split token aux into its own trainable scope and make the empty-context path
  graph-connect a zero loss to the token head.  This is a training-contract
  fix, not a loss-weight tweak.
```

Remote G26-B gate:

```text
Resume point:
  same 11000 checkpoint family as G25-C2 for direct comparison.

Initial gate:
  300 optimizer steps on 2xA100.

Required metrics:
  pi_context_token_aux_loss
  pi_context_token_aux_accuracy
  pi_context_token_aux_weighted_total
  pi_context_flow_gain_mse_delta
  loss_action_default_equiv

Pass:
  token aux loss descends or token accuracy rises,
  flow gain remains non-negative on average,
  loss_action_default_equiv does not degrade beyond G25 final window and
  preferably moves below the G25 last-window mean.

Fail:
  token aux does not learn, or token aux learns while flow/action MSE remains
  flat or worse.  The latter means the remaining issue is not representation
  availability but deployed action-expert consumption.
```

Non-goal:

```text
Do not call this "native FAST CE".  It is a PICF-local FAST-style action-token
objective designed to test the same action-boundary representation hypothesis
without reintroducing the dropped LM heads.
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

## 8. Live G26-B Remote Gate, 2026-06-04

Remote:

```text
host: px-cloud1.matpool.com:26120
repo: /root/openpi_g25_20260604
session: picf_g26b_tokenaux_fresh_k4_300_20260604
log: /mnt/picf_run_logs/picf_g26b_tokenaux_fresh_k4_300_20260604.log
```

User tail:

```bash
ssh -i /tmp/picf_g22_key -p 26120 root@px-cloud1.matpool.com \
  'tail -f /mnt/picf_run_logs/picf_g26b_tokenaux_fresh_k4_300_20260604.log'
```

### 8.1 Checkpoint-resume correction

Earlier G26-B attempts from the `11000` checkpoint did not produce steps.  The
failure signature was not the token auxiliary loss itself:

```text
rank1 stack:
  torch.distributed.distributed_c10d.py barrier
  scripts/picf_core_train.py::_distributed_barrier
  scripts/picf_core_train.py::_load_checkpoint_sequential_across_ranks
```

Therefore the current accepted runtime gate is a no-resume 300-step mechanism
test.  It can validate that the G26-B branch is mathematically and DDP-runnable,
but it must not be used as a direct loss comparison against the 11000-step G25
checkpoint family.

### 8.2 Step-10 and step-20 evidence

Current command family:

```text
resume=false
semantic_trainable_scope=action_head_and_adapter
action_context_integration=suffix_cross_attention
action_context_tokens=24
action_context_stopgrad=true
semantic_action_context_readout_aux_weight=0
semantic_action_context_token_aux_weight=0.05
semantic_action_context_token_aux_bins=64
semantic_action_context_flow_residual_enabled=true
calvin_bucket_sampling_mode=task_uniform
logical_batch_task_count=4
logical_batch_bucket_normalization=true
calvin_bucket_sample_without_replacement=true
```

Structured rows:

```text
step 10:
  loss_action_default_equiv        = 0.151121
  pi_context_token_aux_loss        = 4.164968
  pi_context_token_aux_accuracy    = 0.017944
  pi_context_flow_gain_mse_delta   = +0.004390
  pi_context_flow_base_mse         = 0.155510
  pi_context_flow_adapted_mse      = 0.151121
  active_same_role_overlap_max     = 0.011236
  context_same_role_overlap_max    = 0.014972
  raw_same_role_overlap_max        = 0.459663
  logical_distinct_bucket_count    = 4

step 20:
  loss_action_default_equiv        = 0.150090
  pi_context_token_aux_loss        = 4.094457
  pi_context_token_aux_accuracy    = 0.051147
  pi_context_flow_gain_mse_delta   = +0.009362
  pi_context_flow_base_mse         = 0.159452
  pi_context_flow_adapted_mse      = 0.150090
  active_same_role_overlap_max     = 0.005279
  context_same_role_overlap_max    = 0.024044
  raw_same_role_overlap_max        = 0.475391
  logical_distinct_bucket_count    = 4

step 50:
  loss_action_default_equiv        = 0.114194
  pi_context_token_aux_loss        = 3.258192
  pi_context_token_aux_accuracy    = 0.784790
  pi_context_token_aux_weighted    = 0.162910
  pi_context_flow_gain_mse_delta   = +0.018128
  pi_context_flow_base_mse         = 0.132322
  pi_context_flow_adapted_mse      = 0.114194
  active_same_role_overlap_max     = 0.009878
  context_same_role_overlap_max    = 0.027558
  raw_same_role_overlap_max        = 0.465212
  logical_distinct_bucket_count    = 4

step 100:
  loss_action_default_equiv        = 0.116515
  loss_action_active7              = 0.433831
  pi_context_token_aux_loss        = 1.033541
  pi_context_token_aux_accuracy    = 0.809961
  pi_context_token_aux_weighted    = 0.051677
  pi_context_flow_gain_mse_delta   = +0.009866
  pi_context_flow_base_mse         = 0.126381
  pi_context_flow_adapted_mse      = 0.116515
  active_same_role_overlap_max     = 0.004939
  context_same_role_overlap_max    = 0.022963
  raw_same_role_overlap_max        = 0.454424
  logical_distinct_bucket_count    = 4
```

Interpretation:

```text
1. Token aux is live:
   CE starts near random log(64)=4.159 and already moves to 4.094 by step 20.
   Accuracy rises from about random 1/64 to 0.051.

2. Deployed flow residual is live:
   adapted MSE is below base MSE at both step 10 and step 20.

3. Logical batch is live:
   each logged optimizer step reports 4 distinct task buckets.

4. No convergence claim yet:
   this is fresh no-resume, so action MSE cannot be compared to the 11000-step
   checkpoint-family action numbers.  The pass/fail point remains step 100/200/300.

5. The 50-step gate passes:
   action MSE decreases, token CE decreases, token accuracy rises far above
   random, flow gain remains positive, and active/context overlap stays low.
   Continue to step 100 before accepting the mechanism for a longer run.

6. The 100-step gate passes:
   action MSE remains below the initial 0.15 band, token CE descends by more
   than 3.1 nats, token accuracy stabilizes near 0.81, flow gain stays positive,
   and active/context overlap remains low.  Continue to step 200/300 before
   launching any 30K branch.
```

### 8.3 Exhaustive method status for the user's requested list

```text
Already implemented and historically tested:
  task-uniform logical batch
  trajectory/temperature/ratio sampling
  without-replacement logical updates
  per-bucket logical loss normalization
  per-bucket action EMA scale
  PiKE-style bounded dynamic mixing
  scoped PCGrad
  scoped CAGrad
  continuous PI0.5 action chunks
  L1/Huber action objective option
  modality-specific PICF projectors/adapters
  suffix action-context bridge
  context perturbation audit
  context-only action readout auxiliary
  deployed flow residual
  action-router proxy

Currently under runtime gate:
  PICF-local FAST-style action-token CE, G26-B.

Rejected as current default:
  sampler-only reruns
  optimizer-reset-only reruns
  whole-model PCGrad/CAGrad
  SAM proposal supervision
  sidecar-only mask pull
  router-proxy-only

Deferred for scaling/data reasons:
  embodiment-specific heads/adapters: required for future heterogeneous robots,
  not a CALVIN single-embodiment root cause.

  System2/System1 subtask supervision: architecturally compatible, but current
  CALVIN stream lacks reliable subtask labels for a 2-hour root-cause gate.

Evidence-triggered only:
  full action-expert MoE.  Deploy only if G26-B proves context is action-readable
  but a fresh gradient-cosine audit still shows persistent task-family conflict
  inside the action expert.
```
