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
| Direct FAST/action-token CE | PICF-local discretized action-token auxiliary in `wrapper.py::_compute_action_context_token_aux` | G26-B 300-step no-resume gate passed; imbalance diagnostics added after the run | Keep as the current action-boundary repair; rerun short sanity with label-entropy/majority metrics before longer training. |
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

step 200:
  loss_action_default_equiv        = 0.076665
  loss_action_active7              = 0.339230
  pi_context_token_aux_loss        = 0.730703
  pi_context_token_aux_accuracy    = 0.811206
  pi_context_flow_gain_mse_delta   = +0.016048
  active_same_role_overlap_max     = 0.009379
  context_same_role_overlap_max    = 0.029280
  logical_distinct_bucket_count    = 4

step 300:
  loss_action_default_equiv        = 0.065424
  loss_action_active7              = 0.293769
  pi_context_token_aux_loss        = 0.708323
  pi_context_token_aux_accuracy    = 0.814380
  pi_context_token_aux_weighted    = 0.035416
  pi_context_flow_gain_mse_delta   = +0.011346
  pi_context_flow_base_mse         = 0.076770
  pi_context_flow_adapted_mse      = 0.065424
  active_same_role_overlap_max     = 0.002360
  context_same_role_overlap_max    = 0.018357
  raw_same_role_overlap_max        = 0.456951
  posterior_identity_switch_rate   = 0.111111
  logical_distinct_bucket_count    = 4

last 10 logged rows, steps 210-300:
  loss_action_default_equiv mean   = 0.077507
  loss_action_default_equiv min    = 0.065424
  loss_action_default_equiv max    = 0.096068
  loss_action_active7 mean         = 0.344696
  pi_context_token_aux_loss mean   = 0.721478
  pi_context_token_aux_accuracy    = 0.810073 mean
  pi_context_flow_gain_delta mean  = +0.013089
  active_same_role_overlap mean    = 0.005556
  context_same_role_overlap mean   = 0.020844
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

7. The 200-step gate passes:
   action MSE reaches 0.076665 and remains far below the 0.15 starting band.
   Token CE has not collapsed upward; flow gain stays positive.  Continue to
   step 300 for the final no-resume gate result.

8. The 300-step gate passes:
   the run ends at the best logged action MSE, 0.065424, with positive flow
   gain and low active/context overlap.  This is the first G26-family gate that
   simultaneously shows direct action-representation learning and canonical
   action MSE descent under the task-uniform K4 logical-batch recipe.
```

### 8.3 Final decision after the 300-step gate

```text
Accepted:
  G26-B is a real mechanism repair, not a sampler-only or optimizer-only
  repeat.  It targets the previously measured action-boundary failure by
  making PICF context predict discretized action chunks while keeping canonical
  action MSE visible.

Not yet accepted:
  1. A direct resume from the 11000-step G25 checkpoint family.  Previous
     G26-B resume attempts stalled in `_load_checkpoint_sequential_across_ranks`.
     This is a checkpoint-loading engineering issue that must be fixed before
     claiming this branch works as a continuation path.

  2. Token-accuracy purity.  The current remote run predates the label-entropy
     and majority-bin diagnostics added locally after the run started.  The
     action MSE and flow-gain trends are already strong, but the next run should
     include:

       pi_context_token_aux_label_entropy
       pi_context_token_aux_label_majority_fraction
       pi_context_token_aux_accuracy_over_majority

Next deployment rule:
  Do not spend more time on sampler-only, optimizer-only, PCGrad/CAGrad-only,
  SAM, or sidecar-only branches.  The next meaningful branch is:

    latest code sync -> short 100/200-step G26-B sanity with imbalance metrics
    -> if still positive, launch longer training gate.
```

### 8.4 Exhaustive method status for the user's requested list

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

Accepted by the 300-step no-resume runtime gate:
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

### 8.5 Label-imbalance sanity gate, 2026-06-04

Remote:

```text
host: px-cloud1.matpool.com:26120
repo: /root/openpi_g25_20260604
commit: 92f9197
session: picf_g26b_tokenaux_imbalance_k4_200_20260604
log: /mnt/picf_run_logs/picf_g26b_tokenaux_imbalance_k4_200_20260604.log
ckpt: /mnt/checkpoints/picf_core/picf_core/picf_g26b_tokenaux_imbalance_k4_200_20260604/200
```

Purpose:

```text
Verify that the high G26-B token accuracy is not a trivial majority-bin
classifier caused by quantized action labels being imbalanced.
```

Command family:

```text
resume=false
semantic_trainable_scope=action_head_and_adapter
semantic_action_context_token_aux_weight=0.05
semantic_action_context_token_aux_bins=64
semantic_action_context_flow_residual_enabled=true
calvin_bucket_sampling_mode=task_uniform
logical_batch_task_count=4
logical_batch_bucket_normalization=true
calvin_bucket_sample_without_replacement=true
```

Structured result:

```text
step 10:
  loss_action_default_equiv                 = 0.151121
  pi_context_token_aux_loss                 = 4.164968
  pi_context_token_aux_accuracy             = 0.017944
  pi_context_token_aux_label_majority       = 0.790698
  pi_context_token_aux_accuracy_over_majority = -0.772754
  pi_context_flow_gain_mse_delta            = +0.004387

step 50:
  loss_action_default_equiv                 = 0.114115
  pi_context_token_aux_loss                 = 3.258627
  pi_context_token_aux_accuracy             = 0.784839
  pi_context_token_aux_label_majority       = 0.792456
  pi_context_token_aux_accuracy_over_majority = -0.007617
  pi_context_flow_gain_mse_delta            = +0.018173

step 100:
  loss_action_default_equiv                 = 0.116507
  loss_action_active7                       = 0.433792
  pi_context_token_aux_loss                 = 1.033446
  pi_context_token_aux_accuracy             = 0.809961
  pi_context_token_aux_label_entropy        = 1.188019
  pi_context_token_aux_label_majority       = 0.790210
  pi_context_token_aux_accuracy_over_majority = +0.019751
  pi_context_flow_gain_mse_delta            = +0.009871
  active_same_role_overlap_max              = 0.004939
  context_same_role_overlap_max             = 0.022963

step 200:
  loss_action_default_equiv                 = 0.076705
  loss_action_active7                       = 0.339352
  pi_context_token_aux_loss                 = 0.730741
  pi_context_token_aux_accuracy             = 0.811304
  pi_context_token_aux_label_entropy        = 1.175448
  pi_context_token_aux_label_majority       = 0.791846
  pi_context_token_aux_accuracy_over_majority = +0.019458
  pi_context_flow_gain_mse_delta            = +0.016044
  active_same_role_overlap_max              = 0.009690
  context_same_role_overlap_max             = 0.028148
  raw_same_role_overlap_max                 = 0.461224
  posterior_identity_switch_rate            = 0.151389
  logical_distinct_bucket_count             = 4

last 5 logged rows:
  loss_action_default_equiv mean            = 0.083481
  loss_action_active7 mean                  = 0.364616
  pi_context_token_aux_loss mean            = 0.735669
  pi_context_token_aux_accuracy_over_majority mean = +0.019766
  pi_context_flow_gain_mse_delta mean       = +0.012943
  active_same_role_overlap_max mean         = 0.004157
  context_same_role_overlap_max mean        = 0.021504

last 10 logged rows:
  loss_action_default_equiv mean            = 0.089562
  loss_action_active7 mean                  = 0.372151
  pi_context_token_aux_loss mean            = 0.768338
  pi_context_token_aux_accuracy_over_majority mean = +0.019888
  pi_context_flow_gain_mse_delta mean       = +0.013036
  active_same_role_overlap_max mean         = 0.004327
  context_same_role_overlap_max mean        = 0.022076
```

Interpretation:

```text
1. The label distribution is indeed imbalanced:
   majority fraction is about 0.79.  Therefore raw token accuracy alone is not
   a valid success metric.

2. The token auxiliary still passes after correcting for imbalance:
   accuracy_over_majority becomes positive at step 60 and stays positive through
   step 200.  The final value is +0.019458 and the last-10 mean is +0.019888.

3. The direct representation objective is not a majority-class artifact:
   CE falls from about random log(64)=4.159 to 0.730741 while
   accuracy_over_majority is positive.

4. The deployed PI0.5 flow path remains positively affected:
   `pi_context_flow_gain_mse_delta` is positive through the final row, with
   final +0.016044 and last-10 mean +0.013036.

5. The structure health remains acceptable for this branch:
   active/context support overlap is low.  Raw same-role overlap remains around
   0.46, but this is reserve/inactive overlap and is not the active downstream
   collapse mode that blocked older runs.
```

Updated gate decision:

```text
Accepted:
  G26-B is not a sampler-only, optimizer-only, sidecar-only, or auxiliary-only
  repeat.  It is the first branch that passes:

    task-balanced logical batch
    per-bucket normalization
    deployed flow residual
    PICF-local FAST-style action-token representation pressure
    majority-baseline token diagnostic

Still not accepted:
  direct resume from the 11000-step G25 checkpoint.  That remains a separate
  checkpoint-loading/barrier engineering problem.

Next valid GPU use:
  Either fix the resume-loader barrier and run G26-B from a mature checkpoint,
  or launch a longer fresh/clean G26-B gate.  Do not rerun data-mixing-only,
  optimizer-only, PCGrad/CAGrad-only, SAM, sidecar-only, or raw-overlap-only
  branches without a new falsifiable hypothesis.
```

## 9. 30000-Step Fresh Long Gate

Launch time: 2026-06-04 19:17 China time.

Remote:

```text
host: px-cloud1.matpool.com:26120
repo: /root/openpi_g25_20260604
commit: c0be1d9
tmux: picf_g26b_tokenaux_long30k_k4_fresh_20260604
log: /mnt/picf_run_logs/picf_g26b_tokenaux_long30k_k4_fresh_20260604.log
cmd: /mnt/picf_run_logs/picf_g26b_tokenaux_long30k_k4_fresh_20260604.cmd
ckpt: /mnt/checkpoints/picf_core/picf_core/picf_g26b_tokenaux_long30k_k4_fresh_20260604
```

Decision: fresh run, not resume.

Reason:

```text
1. G26-B 200/300-step checkpoints are sanity gates, not mature training states.
   Resuming them saves little and adds checkpoint-loader risk.

2. G25/G24 mature checkpoints were produced before the final G26-B token-aux
   deployment and are not a clean test of this mechanism.

3. Direct resume from the 11000-step G25 family previously stalled in the
   sequential checkpoint-loader barrier.  That remains a separate engineering
   problem and must not be mixed into the 30000-step mechanism gate.
```

Long-run contract:

```text
num_train_steps=30000
resume=false
save_interval=500
keep_last_checkpoints=3
progress_bar=true
semantic_trainable=true
semantic_trainable_scope=action_head_and_adapter
action_context_integration=suffix_cross_attention
action_context_tokens=24
action_context_stopgrad=true
semantic_action_context_readout_aux_weight=0.0
semantic_action_context_token_aux_weight=0.05
semantic_action_context_token_aux_bins=64
semantic_action_context_token_aux_clip=1.0
semantic_action_context_flow_residual_enabled=true
semantic_action_context_flow_residual_gate_init=-2.0
semantic_action_context_flow_residual_time_floor=0.05
semantic_action_context_flow_residual_rms_cap=true
calvin_bucket_sampling_mode=task_uniform
logical_batch_task_count=4
logical_batch_bucket_normalization=true
calvin_bucket_sample_without_replacement=true
```

Initial runtime audit:

```text
DDP world size: 2
bucket names:
  block_lift, block_other, block_push, drawer, other, slider, switch_button_light
bucket target weights:
  all seven buckets are task-uniform at 1/7
```

Expected evidence:

```text
Early 0-500 steps:
  action loss should compress faster than the pre-G26 platform runs.
  token CE should fall below the majority-only baseline after warmup.
  pi_context_flow_gain_mse_delta should stay positive on logged windows.
  active/context same-role overlap should remain low.

500-3000 steps:
  the important signal is not only low action loss, but whether action loss can
  keep improving without slot-JEPA explosion, deployed-flow gain inversion, or
  active posterior collapse.

Long gate:
  this run can validate G26-B as the maintained long-training recipe only if
  it shows sustained action compression and stable structure health beyond the
  short gate.  It is not a proof before those logs exist.
```

## 10. Checkpoint Cleanup and Fresh-Run Watch Line

The June-04 remote checkpoint cleanup is archived in:

```text
docs/PICF_AQR_OWM_G26B_CKPT_CLEANUP_AND_ARCHIVE_20260604.md
remote manifest: /mnt/picf_run_logs/ckpt_cleanup_20260604_manifest.txt
```

Cleanup result:

```text
deleted checkpoint directories: 97
kept checkpoint directories:    18
```

The cleanup removed old short-test/sanity/diagnostic ckpt directories only.
It preserved April baselines, known 7000/8000/10000/11000 anchors,
G22/G24/G25/G26 key gates, and the active G26-B long-run directory.  Logs and
local docs remain the source of truth for deleted short-test weights.

Fresh-run action trend through the first checked window:

```text
step100: loss_action_default_equiv ~= 0.1165
step200: loss_action_default_equiv ~= 0.0766
step300: loss_action_default_equiv ~= 0.0654
step330: loss_action_default_equiv ~= 0.0785
```

This is slower than the older best short-window/resume PICF traces that entered
the `0.03-0.05` band around 500-1000 steps.  Because the current run is a fresh
task-uniform G26-B mechanism test, not a mature-checkpoint resume, the early
gap is not a failure by itself.  It does require a strict watch line:

```text
step500 <= 0.060:
  continue as healthy.

step500 in 0.060..0.070:
  continue to step700 and decide from trend.

step500 > 0.070:
  stop and treat G26-B fresh as not reproducing the old PICF action-compression
  speed.
```

## 11. Step500 Action-Platform Contrast Gate, 2026-06-05

Why this gate exists:

The fresh G26-B long run reached the first checkpoint, but its
`loss_action_default_equiv` did not clearly enter the old best PICF action band.
Active/context overlap remained bounded, so this gate does not retest raw
same-role overlap.  It tests whether the action platform is caused by the
training boundary or action-side LR.

Baseline checkpoint:

```text
/mnt/checkpoints/picf_core/picf_core/
picf_g26b_tokenaux_long30k_k4_fresh_20260604/500
```

Baseline log:

```text
/mnt/picf_run_logs/picf_g26b_tokenaux_long30k_k4_fresh_20260604.log
```

Contrast A: widen semantic scope.

```text
run:
  picf_cmpA_pgmodel_lr5e5_from500_20260605

changed:
  semantic_trainable_scope=model_only
  lr=5e-5
  semantic_lr_scale=1.0

result:
  failed before first optimizer step on 2xA100-40GB.

failure:
  CUDA OOM in pipeline._projective_attention_bias / projective_bias_head.
  GPU0 used about 38.5GB and then needed another 1.17GB.

decision:
  This is a capacity exclusion, not a model-quality result.  Full restored
  semantic-stack training is not a valid 2x40GB quick contrast under the
  current visual/point/PICF configuration.
```

Contrast B: keep trainable boundary, raise action/adapter LR.

```text
run:
  picf_cmpB_adapter_lr5e5_from500_20260605

changed:
  semantic_trainable_scope=action_head_and_adapter
  lr=5e-5
  min_lr=5e-5
  semantic_lr_scale=1.0

unchanged:
  task_uniform logical batch K=4
  accum_steps=2
  sidecar/proposal/tracklet path
  prefix/context action-boundary repair

status:
  running on px-cloud2:28373 from step500.
  reached step507 at about 29s/step.
  reached step550 with loss_action_default_equiv=0.072889.

tail:
  tail -f /mnt/picf_run_logs/picf_cmpB_adapter_lr5e5_from500_20260605.log
```

Interim step550 comparison:

```text
source G26-B fresh step550:
  loss_action_default_equiv = 0.0735315
  loss_action_active7       = 0.329647
  loss_anchor_pv            = 0.684810
  loss_mapg_routing         = 0.377704
  loss_slot_jepa            = 0.242171

contrast B step550:
  loss_action_default_equiv = 0.0728891
  loss_action_active7       = 0.326792
  loss_anchor_pv            = 0.682659
  loss_mapg_routing         = 0.370668
  loss_slot_jepa            = 0.242047

reading:
  B is marginally better at step550, but the delta is too small to call a
  platform fix.  Structure terms remain healthy.  Continue to step600 before
  finalizing the LR-only contrast.
```

Final step600 comparison:

```text
source G26-B fresh step600:
  loss_action_default_equiv = 0.0723489
  loss_action_active7       = 0.325999
  loss_total_minus_action   = 0.167558
  loss_anchor_pv            = 0.684632
  loss_mapg_routing         = 0.385702
  loss_slot_jepa            = 0.241136
  active/context support ov = 0.007902 / 0.017207
  context-flow gain         = 0.014663

contrast B step600:
  loss_action_default_equiv = 0.0713287
  loss_action_active7       = 0.319141
  loss_total_minus_action   = 0.190204
  loss_anchor_pv            = 0.685939
  loss_mapg_routing         = 0.386191
  loss_slot_jepa            = 0.240471
  active/context support ov = 0.010277 / 0.019966
  context-flow gain         = 0.012872

reading:
  Raising the current action_head_and_adapter LR to 5e-5 gives only a small
  action improvement at step600 and slightly worse non-action total.  The
  structure terms are not damaged, but this is not a platform-breaker.  The
  action platform is therefore not primarily an LR-only problem inside the
  current narrow trainable boundary.

decision:
  Stop contrast B after step600.  Do not continue raising LR blindly.  The
  next non-duplicate contrast must isolate task-distribution/sampler effects:
  trajectory-proportional or temperature sampling versus the current
  task-uniform K4 logical batch, with the same model boundary and action LR.
```

Acceptance rule:

```text
If B step550/600 loss_action_default_equiv clearly beats the source long-run
step550/600 window without structure-term explosion, the platform is primarily
LR-limited inside the current action_head_and_adapter trainable boundary.

If B does not beat the source window, do not keep raising LR blindly.  The next
non-duplicate contrast should isolate sampler distribution: old-style
trajectory/random sampling versus task_uniform K=4, while keeping the same
scope and LR.
```

Contrast C: keep B boundary/LR, change only sampler distribution.

```text
run:
  picf_cmpC_trajectory_lr5e5_from500_20260605

changed versus B:
  calvin_bucket_sampling_mode=trajectory

unchanged versus B:
  resume checkpoint = G26-B step500
  optimizer_checkpoint_mode=model_only
  semantic_trainable_scope=action_head_and_adapter
  lr=min_lr=5e-5
  semantic_lr_scale=1.0
  logical_batch_task_count = world_size * accum_steps = 4

purpose:
  Test whether the action platform is caused by task-uniform K4 changing the
  effective data distribution away from the trajectory-proportional action
  objective.  This is a direct VLA-Foundry/ABot-M0-style data-mixing contrast,
  not another optimizer-only rerun.

acceptance:
  If C clearly beats B/source on loss_action_default_equiv by step600 while
  preserving bounded active/context overlap, sampler distribution is a major
  root cause.  If C tracks B, the platform is not fixed by LR or target bucket
  distribution under the current narrow trainable boundary.

tail:
  tail -f /mnt/picf_run_logs/picf_cmpC_trajectory_lr5e5_from500_20260605.log
```

Interim step550 comparison:

```text
source G26-B fresh step550:
  loss_action_default_equiv = 0.0735315
  loss_action_active7       = 0.329647
  loss_anchor_pv            = 0.684810
  loss_mapg_routing         = 0.377704
  loss_slot_jepa            = 0.242171

contrast B step550:
  loss_action_default_equiv = 0.0728891
  loss_action_active7       = 0.326792
  loss_anchor_pv            = 0.682659
  loss_mapg_routing         = 0.370668
  loss_slot_jepa            = 0.242047

contrast C step550:
  loss_action_default_equiv = 0.0688465
  loss_action_active7       = 0.307578
  loss_total_minus_action   = 0.187323
  loss_anchor_pv            = 0.688112
  loss_mapg_routing         = 0.379840
  loss_slot_jepa            = 0.242743
  active/context support ov = 0.000516 / 0.013524
  context-flow gain         = 0.010745

reading:
  C is the first step500 contrast that improves action by more than noise at
  step550 while keeping active/context overlap low.  The improvement is still
  modest, so continue to step600 before deciding whether trajectory-proportional
  sampling is a major part of the action-platform root cause.
```

Final step600 comparison:

```text
source G26-B fresh step600:
  loss_action_default_equiv = 0.0723489
  loss_action_active7       = 0.325999
  loss_total_minus_action   = 0.167558
  loss_anchor_pv            = 0.684632
  loss_mapg_routing         = 0.385702
  loss_slot_jepa            = 0.241136
  active/context support ov = 0.007902 / 0.017207
  context-flow gain         = 0.014663

contrast B step600:
  loss_action_default_equiv = 0.0713287
  loss_action_active7       = 0.319141
  loss_total_minus_action   = 0.190204
  loss_anchor_pv            = 0.685939
  loss_mapg_routing         = 0.386191
  loss_slot_jepa            = 0.240471
  active/context support ov = 0.010277 / 0.019966
  context-flow gain         = 0.012872

contrast C step600:
  loss_action_default_equiv = 0.0768337
  loss_action_active7       = 0.339137
  loss_total_minus_action   = 0.195179
  loss_anchor_pv            = 0.688552
  loss_mapg_routing         = 0.386720
  loss_slot_jepa            = 0.239811
  active/context support ov = 0.003204 / 0.015557
  context-flow gain         = 0.014795

reading:
  C's step550 action improvement did not persist.  By step600 it is worse than
  both the source G26-B window and contrast B on the canonical action metric,
  while structure metrics remain healthy.  Therefore the action platform is not
  solved by trajectory-proportional sampler distribution alone under the current
  action_head_and_adapter trainable boundary.

decision:
  Stop contrast C after step600.  Do not repeat sampler-only task_uniform /
  trajectory / temperature experiments as the next step unless a new
  gradient-cosine or per-bucket-loss audit identifies a specific bucket-mixing
  failure.  The remaining non-duplicate hypotheses are:

  1. the trainable action/semantic boundary is too narrow;
  2. the action signal needs a memory-safe wider bridge or action-expert path;
  3. the logical batch must change how task losses are combined, not just which
     buckets are sampled.
```

Comparison rule:

```text
Use only:
  loss_action_default_equiv
  loss_action_active7
  per-bucket loss_action_default_equiv

Do not compare the progress-bar scalar loss or raw current loss_action with
the older 4-22 raw loss_action.
```

Contrast D: widen trainable boundary through FSDP, not direct DDP.

```text
run:
  picf_cmpD_fsdp_model_lr5e5_from500_20260605

changed versus B/C:
  training_strategy=fsdp_full_shard
  window_activation_checkpointing=True
  semantic_trainable_scope=model_only
  lr=min_lr=5e-5

purpose:
  Retest Contrast A's "trainable boundary too narrow" hypothesis without the
  direct-DDP 2x40GB OOM.  This is the next non-duplicate contrast after
  excluding LR-only and sampler-only fixes.

first attempt result:
  failed before the first JSON loss row, but not from model divergence or OOM.

failure:
  dataclasses.FrozenInstanceError: cannot assign to field 'tokens'
  inside torch.distributed.fsdp._runtime_utils._register_pre_backward_hooks.

root cause:
  PaliGemmaSemanticFeatures was a frozen dataclass returned from the semantic
  encoder.  FSDP registers backward hooks on structured output tensors by
  replacing the corresponding dataclass fields.  A frozen tensor-output
  container is therefore incompatible with this FSDP path.

fix:
  Make the semantic output dataclass tree mutable.  The first local fix made
  PaliGemmaSemanticFeatures mutable, but the px-cloud2 rerun then failed on the
  nested view-transform metadata:

    dataclasses.FrozenInstanceError: cannot assign to field 'original_hw'

  This proves FSDP recurses through tuple-contained dataclass metadata while
  registering hooks.  Therefore both PaliGemmaSemanticFeatures and
  PaliGemmaViewTransform must be mutable on this FSDP path.

decision:
  The first D attempt is a framework-contract failure, not a training result.
  The second D attempt is the same framework-contract class on nested metadata,
  not a training result.  Relaunch D only after the full mutable-output-tree fix
  before making any conclusion about whether model_only semantic capacity
  breaks the action platform.
```

## 12. Post-New-Machine Shutdown Plan, 2026-06-05

Current live old-machine run:

```text
host:
  px-cloud1:26120

tmux:
  picf_g26b_tokenaux_long30k_k4_fresh_20260604

checkpoint dir:
  /mnt/checkpoints/picf_core/picf_core/picf_g26b_tokenaux_long30k_k4_fresh_20260604

available checkpoints:
  500
  1000
```

Latest status at step1110:

```text
loss_action_default_equiv last-20 mean = 0.06820
loss_action_default_equiv last-20 min  = 0.05801
loss_action_default_equiv last-20 max  = 0.07859

loss_anchor_pv last-20 mean            = 0.67956
loss_mapg_routing last-20 mean         = 0.39075
loss_slot_jepa last-20 mean            = 0.24100

active same-role support overlap mean  = 0.00678
context same-role support overlap mean = 0.02159
posterior recycle rate near step1110   = 0.05403
```

Reading:

```text
The run is alive and structurally healthy.  It is not collapsing through active
support overlap, context overlap, slot-JEPA explosion, or recycle saturation.
However, it also has not broken the action platform: the canonical action
metric is still oscillating in roughly the 0.058-0.079 band instead of entering
the old best 0.02-0.03 action band.
```

Do not repeat:

```text
LR-only inside action_head_and_adapter:
  excluded by Contrast B.

sampler-only task_uniform/trajectory:
  excluded by Contrast C.

raw-overlap-only:
  not the current blocker; active/context overlap is already low.

fixed-window-only comparisons:
  not representative enough for production acceptance.
```

Next experiments, in priority order:

```text
1. Keep the old-machine G26-B run alive until at least step1500 unless action
   mean worsens above 0.085, non-action losses explode, or a nonfinite event
   appears.  It is useful as a continuous-platform trace, but not yet a
   candidate success.

2. When a spare machine is available, rerun Contrast D after the full FSDP
   mutable-output-tree fix:

     training_strategy=fsdp_full_shard
     semantic_trainable_scope=model_only
     lr=min_lr=5e-5
     resume_checkpoint=G26-B step500

   Acceptance: by step600, action_default must clearly beat source/B/C while
   preserving bounded active/context overlap.  This tests whether the current
   narrow trainable boundary is the real platform cause.

3. In parallel or immediately after D, run a same-code PI0.5-like ablation:

     same data split
     same logical batch K4
     same action normalization and logging
     PICF branches disabled or picf_mode=ablated

   Acceptance: if ablated PI0.5-like action falls faster than full PICF under
   the same data/sampler/logging contract, PICF/context integration is still
   action-negative.  If it also plateaus, the root is not PICF-specific and
   points back to data/action-objective or action-head capacity.

4. Only after 2 and 3 decide whether to redesign the action boundary:

   - if D works: keep PICF structure, use memory-safe wider semantic/action
     bridge;
   - if PI0.5-like works but D does not: isolate PICF-to-action injection and
     action expert capacity;
   - if neither works: investigate action objective/data distribution rather
     than adding more PICF losses.
```

Operational note:

```text
px-cloud2:28373 was shut down after backing up Contrast D log/cmd locally under:
  tmp/remote_backups/px-cloud2_28373_20260605/

The FSDP dataclass fix was committed and pushed:
  5d9af6b Make PaliGemma semantic features FSDP-safe
  cca5a59 Make PaliGemma view metadata FSDP-safe
```

## 13. PI0.5-Like K4 Ablation Control, 2026-06-05

Purpose:

```text
Test the current repository/trainer/data/logical-batch shell with PICF recurrent
semantics disabled.  This isolates whether the current G26-B action platform is
PICF/context-specific or also appears in the native PI0.5 semantic action path
under the same current CALVIN bucket sampler and logging contract.
```

Run:

```text
machine:
  px-cloud1:26120

tmux:
  picf_g26pi05like_k4_ablation_30k_20260605

repo:
  /root/openpi_g25_20260604
  commit f61887a

log:
  /mnt/picf_run_logs/picf_g26pi05like_k4_ablation_30k_20260605.log

command:
  /mnt/picf_run_logs/picf_g26pi05like_k4_ablation_30k_20260605.cmd
```

Key contract:

```text
picf_mode=ablated
MAPG/AQR/VL router disabled
native PI0.5 semantic action path
extra_prefix_tokens=None
training_strategy=fsdp_full_shard
world_size=2
accum_steps=2
effective_global_batch=4
calvin_bucket_sampling_mode=task_uniform
logical_batch_task_count=4
logical_batch_bucket_normalization=True
calvin_bucket_sample_without_replacement=True
lr=2e-4
min_lr=2e-5
warmup_steps=600
```

Trainable-boundary note:

```text
semantic_trainable_scope=backbone_only trains the PaliGemma/Gemma expert model
stack (`self.model.parameters()` in `wrapper.py::_apply_trainable_scope`) and
freezes only wrapper-local flow/time calibration heads.  This is the maintained
historical full-cotrain boundary in the current wrapper tests.  It is therefore
not a frozen-action-head probe, but it is also not exact 4-22 parity.
```

Important non-equivalence:

```text
This is not exact 2026-04-22 PI0.5 parity.

It uses the current PICF trainer shell, current task-uniform K4 logical batch,
current quantile action/prompt-state normalization, FSDP, semantic length 256,
and current CALVIN transition source.  It also disables the live PICF point /
visual / tactile branches.  Therefore it answers a narrower but critical
question:

  "Under the same current data/sampler/logging contract, does removing PICF
   semantics make action descend materially faster?"

It must not be used as a claim about official PI0.5 or historical 4-22 parity.
```

Step50 result:

```text
step                          = 50
loss_total                    = 0.1291981190
loss_action_default_equiv     = 0.1291981190
loss_action_active7           = 0.4800469577
loss_action                   = 0.1291981190
loss_total_minus_action       = 0.0
grad_norm                     = 1.6998214722
lr                            = 1.6666666667e-05
logical_batch_distinct_bucket = 4
speed                         = 0.0520 steps/sec
```

Reading:

```text
Step50 is not positive evidence yet.

It verifies that the K4 task coverage is real and that the ablation has no PICF
structure-loss contamination, because loss_total equals action.  However, action
is still worse than G26-B fresh step100 (0.1165), while the LR is still very low
inside the 600-step warmup.  Do not stop or accept the branch from step50 alone.
```

Step100 result:

```text
step                          = 100
loss_total                    = 0.0770749152
loss_action_default_equiv     = 0.0770749152
loss_action_active7           = 0.3494936228
loss_action                   = 0.0770749152
loss_total_minus_action       = 0.0
grad_norm                     = 0.8037739396
lr                            = 3.3333333333e-05
logical_batch_distinct_bucket = 4
speed                         = 0.0511 steps/sec
```

Per-bucket action-default:

```text
block_lift          0.09356
block_other         0.08351
block_push          0.06008
drawer              0.09716
other               0.07079
slider              0.04984
switch_button_light 0.09115
```

Step100 reading:

```text
Positive but not final.

The ablated PI0.5-like path is materially better than G26-B fresh step100
(0.1165) and is already near G26-B fresh step200 (0.0766), while K4 coverage is
confirmed and all PICF structure losses are zero.  This supports the hypothesis
that current PICF/context integration is action-negative or at least action
slowing under the G26-B setup.

It does not yet prove the branch will enter the old 0.02-0.03 band.  The LR is
still in warmup, and the hardest buckets (`drawer`, `switch_button_light`,
`block_lift`) remain above 0.09.  Continue to step200 before deciding whether to
stop or pivot.
```

Step150 result:

```text
step                          = 150
loss_total                    = 0.0646321550
loss_action_default_equiv     = 0.0646321550
loss_action_active7           = 0.2944722176
loss_action                   = 0.0646321550
loss_total_minus_action       = 0.0
grad_norm                     = 0.6035906076
lr                            = 5.0e-05
logical_batch_distinct_bucket = 4
```

Per-bucket action-default:

```text
block_lift          0.06804
block_other         0.06786
block_push          0.04960
drawer              0.06414
other               0.06641
slider              0.05041
switch_button_light 0.08815
```

Step150 reading:

```text
The ablated PI0.5-like path reaches the G26-B fresh step300 region by step150.
This is a stronger positive control than step100: under the same current trainer
shell and K4 logical-batch sampler, disabling PICF structure/context restores a
much faster native action descent.

This still does not prove 4-22 parity or old 0.02-0.03 convergence.  The
hardest bucket remains switch/button/light, and the run is still in warmup.
Continue to step200/300; do not start another sampler-only experiment while this
causal control is unresolved.
```

Step200 result:

```text
step                          = 200
loss_total                    = 0.0641728789
loss_action_default_equiv     = 0.0641728789
loss_action_active7           = 0.2913683653
loss_action                   = 0.0641728789
loss_action_pos               = 0.2413374484
loss_action_rot               = 0.2888981402
loss_action_gripper           = 0.4488718212
loss_total_minus_action       = 0.0
grad_norm                     = 0.4555698037
lr                            = 6.6666666667e-05
logical_batch_distinct_bucket = 4
```

Per-bucket action-default:

```text
block_lift          0.05804
block_other         0.07339
block_push          0.05623
drawer              0.06972
other               0.05717
slider              0.05109
switch_button_light 0.08940
```

Step200 reading:

```text
The control remains positive relative to G26-B step200 (0.0766) and is still
slightly better than G26-B step300 (0.0654).  But step150 -> step200 is almost
flat: 0.06463 -> 0.06417.  The run has not reproduced the historical fast
0.02-0.03 convergence band.

This refines the causal interpretation:
  - disabling PICF/context is beneficial for early action descent;
  - disabling PICF/context alone is not yet sufficient to prove 4-22 parity;
  - switch/button/light and gripper remain the visible bottlenecks at step200.

Continue to step300.  If step300 remains around 0.064, the next branch should
not repeat sampler-only methods; it should audit current trainer/action path
parity against the historical 4-22/pi0.5 definition, and separately design a
PICF bridge that has measured positive action gain before re-enabling PICF.
```

Decision rule:

```text
At step100:
  If action_default is already much lower than G26-B step100 (0.1165), PICF
  semantics/context are likely action-negative in the current integration.

  If action_default is similar or worse, keep to step200 because warmup is still
  early, but treat "remove PICF alone" as unproven.

At step200:
  Compare against G26-B step200 (0.0766).  If ablation cannot beat it, the
  action platform is not explained by PICF structure alone.

At step300:
  Compare against G26-B step300 (0.0654).  If ablation remains worse or only
  equal, next work should shift from PICF structure to action objective /
  semantic path / data mixture parity with the historical 4-22 baseline.
```

Step250/300 update, 2026-06-05:

```text
step250 action_default = 0.0656914562
step300 action_default = 0.0640842691

step300 action components:
  pos     = 0.2238104343
  rot     = 0.3115257025
  gripper = 0.4133533239

logical_batch_distinct_bucket_count = 4
loss_total_minus_action             = 0.0
```

Interpretation:

```text
The current-shell PI0.5-like ablation clearly improves over full G26-B, but it
plateaus around the G26-B step300 region rather than entering the old 4-22
0.02-0.03 band.  Therefore the live evidence is:

  PICF/context integration is action-negative under G26-B;
  but PICF/context is not the only missing variable.

The next non-repeated experiment is trainer/action-shell parity:
  old 4-22 ablated PI0.5 args overlaid onto current-code defaults
  + current task_uniform K4 logical batch
  + explicit semantic_action_context_flow_residual_enabled=false.

This is deliberately not another sampler-only, PCGrad/CAGrad, Huber/L1,
dynamic-mixing, SAM, raw-overlap, or context-only experiment; those are already
recorded as tested/diagnostic in the action descent checklist.
```

Active parity run:

```text
picf_g27_oldargs_k4_pi05ablated_300_20260605
```

G27 result:

```text
step50  action_default = 0.1529971361
step100 action_default = 0.1471613646
```

Decision:

```text
Stopped at step100.  G27 is much worse than the G26 current-shell PI0.5-like
ablation at the same step:

  G26 current-shell K4 step100 = 0.077075
  G27 old-args K4 step100      = 0.147161

This rejects the simple hypothesis that restoring old 4-22 action-shell fields
under current K4 task-balanced sampling is sufficient.  The next root-cause
branch must not repeat sampler-only, PCGrad/CAGrad, Huber/L1, dynamic-mixing,
or old-action-shell toggles.  The live evidence now points to the interaction
between the balanced K4 objective and the action representation/readout itself.
```
