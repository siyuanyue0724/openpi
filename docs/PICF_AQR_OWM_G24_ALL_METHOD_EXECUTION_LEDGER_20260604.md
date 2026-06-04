# PICF-AQR-OWM G24 All-Method Execution Ledger

Date: 2026-06-04

Entry point:

```text
src/openpi/picf/README_v2.2.md
```

Purpose:

```text
Convert every requested VLA/slot/action-convergence method into an auditable
math/code/history/experiment checklist before spending more GPU time.
```

This ledger is stricter than a run note.  A method can only move forward if it
passes all four gates:

```text
1. Math gate:
   It optimizes the intended objective
       L = sum_b q_b L_b
       G = sum_b q_b grad L_b
   without changing labels, hiding action loss, or weakening large-data scaling.

2. Code follow-through gate:
   There is an identified code path from CLI/config -> dataloader/model/loss ->
   metrics/logs/checkpoints.

3. History gate:
   The experiment record does not already reject the same hypothesis on the
   current code family.

4. Runtime gate:
   If it is still unexcluded, run only the smallest test that can falsify it.
```

## 1. Current Root-Cause Boundary

The current evidence does **not** support repeating sampler-only or
optimizer-only runs.

The desired multi-task objective is:

```text
L(theta) = sum_b q_b L_b(theta)
G(theta) = sum_b q_b grad_theta L_b(theta)
```

A small random batch estimates:

```text
g_step = grad_theta L_b(theta)
Var[g_step] = E_b ||grad L_b - G||^2
```

Therefore task-balanced logical batching is necessary.  However, G12-G23 show
it is not sufficient: the native PI0.5 action flow still plateaus near the
balanced-window `0.04` band even when task coverage, loss scaling, scoped
gradient surgery, and action-router diagnostics are present.

The decisive factor exposed by G20/G22 is:

```text
d A_native / d C_picf is too weak.
```

Where:

```text
C_picf = bounded PICF belief/context tokens
A_native = native PI0.5 continuous action flow expert
Y = target action chunk
```

G22 proves `C_picf` is motor-readable under a direct auxiliary:

```text
R_eta(C_picf) -> Y
readout_loss: 0.051960 -> 0.032896
readout_mse : 0.105277 -> 0.067227
```

But the canonical action objective worsened over the same gate:

```text
loss_action_default_equiv: 0.040747 -> 0.052298
```

This closes the “only batch/sampler/optimizer” hypothesis for the current code
family.

## 2. Paper/Method Coverage Matrix

Primary method sources used for the checklist:

```text
VLA Foundry, 2026:
  probabilistic dataset mixing, batch balancing ratios, gradient accumulation,
  data/stat normalization.
  https://arxiv.org/abs/2604.19728

ABot-M0, 2026:
  multi-source robot data cleaning, task/embodiment balancing, task-uniform
  versus trajectory-uniform sampling.
  https://arxiv.org/abs/2602.11236

PiKE, 2025:
  dynamic data mixing under task-gradient conflict.
  https://arxiv.org/abs/2502.06244

OpenVLA-OFT, 2025:
  continuous action chunks, parallel action decoding, L1/Huber-style
  continuous action fine-tuning.
  https://arxiv.org/abs/2502.19645

Knowledge Insulation / pi0.5, 2025:
  continuous action expert gradients can pollute the VLM backbone; action
  representation and action expert need a controlled boundary.
  https://arxiv.org/abs/2505.23705

AdaMoE-VLA / FedVLA, 2025-2026:
  action-specialized experts and task/embodiment-aware expert routing when
  gradient conflict remains after sane batching/loss normalization.
  https://arxiv.org/abs/2510.14300
```

The corresponding local status:

| Method family | Code path | Historical evidence | Current decision |
| --- | --- | --- | --- |
| Task-uniform logical batch | `_calvin_prompt_bucket`, `_bucket_sequence_for_logical_step` | G10/G12/G18/G22 | Keep mandatory. Already tested; not sufficient alone. |
| VLA-Foundry ratios | `--calvin-bucket-weight-spec` | G12-RATIO | Keep knob. Rejected as current action fix. |
| Temperature sampling | `--calvin-bucket-sampling-mode=temperature` | G12-TEMP | Keep knob. Rejected as current action fix. |
| Without-replacement logical update | `--calvin-bucket-sample-without-replacement` | G12/G22 | Keep default. |
| Per-bucket logical loss scaling | `_logical_batch_loss_scales` | G12 dataflow | Keep mandatory. |
| Per-bucket action EMA normalization | `_logical_action_bucket_scale` | G12/G18 | Keep optional. Not root fix. |
| Dynamic PiKE-style mixing | `_dynamic_bucket_sampling_weights` | G12-DYN | Implemented; rejected as action fix. |
| Scoped PCGrad | `_pcgrad_project_and_sum` | G12-PCGrad | Implemented; structure improved, action failed. Diagnostic only. |
| Scoped CAGrad | `_cagrad_project_and_sum` | G12-CAGrad | Implemented; structure improved, action failed. Diagnostic only. |
| Whole-model PCGrad/CAGrad | absent by design | Cost/mismatch analysis | Rejected for large-data scaling and current causal boundary. |
| Continuous action chunks | native PI0.5 action path | all action gates | Already used. |
| L1/Huber action objective | `--semantic-action-flow-loss` | G15 | Rejected as canonical action fix. |
| Action head only / adapter only | semantic trainable scopes | G13 | Rejected as action fix. |
| Action expert router/MoE proxy | `_apply_action_expert_router` | G16 | Implemented; action did not improve. Keep off by default. |
| Context suffix cross-attention | `_apply_action_context_adapter` | G17/G20 | Implemented; context perturbation showed weak causal effect. |
| Context-only action readout auxiliary | `_compute_action_context_readout_aux` | G22 | Passes motor-readability, but does not fix native action. |
| Direct FAST/action-token CE | native OpenPI exists, PICF wrapper drops generation heads | G21/G22 | Not implemented. This is the next real architecture branch if continuing. |
| Embodiment adapters/heads | mostly not needed for CALVIN | single embodiment | Defer until heterogeneous robot datasets. |
| System2/System1 subtask split | no reliable CALVIN subtask labels | theory only | Defer; not a 2-hour convergence fix. |
| Full action-expert MoE | not deployed as full expert replacement | G16 proxy failed | Defer until direct action-context supervision is restored. |

## 3. Completed Experiment Checklist

### Data mixing / logical batch

```text
[x] task-uniform bucket sampler
[x] trajectory-proportional baseline comparison
[x] temperature alpha=0.5
[x] explicit ratio spec
[x] K4 logical batch on 2xA100
[x] K8/K10/K12 hardware attempts
[x] bucket metrics and per-bucket action logging
[x] E21 exact-window diagnostic
```

Conclusion:

```text
The estimator infrastructure is real and necessary, but not sufficient.
Balanced fixed-window action remains around 0.038-0.043 in comparable probes.
```

### Loss scale / dynamic mixing

```text
[x] per-bucket `q_b/n_b` normalization
[x] per-bucket action EMA scale
[x] bounded PiKE-style dynamic q_b
[x] L1/Huber action objective
```

Conclusion:

```text
These improve logging/control and sometimes structure, but do not make native
action leave the plateau by themselves.
```

### Gradient conflict

```text
[x] scoped PCGrad over semantic/action adapter groups
[x] scoped CAGrad over semantic/action adapter groups
[x] gradient-surgery dataflow validation
[x] whole-model PCGrad/CAGrad rejection recorded
```

Conclusion:

```text
Scoped surgery is implemented, but action still worsened in the short gates.
Do not repeat unless a new gradient-cosine probe identifies a new conflict
after action-context causality is fixed.
```

### Action/context boundary

```text
[x] suffix cross-attention context bridge
[x] context perturbation audit:
    completed: none_ref/zero/token_roll/sign_flip/rms_noise/disable_picf
    pending  : none
[x] action head only
[x] action adapter only
[x] head+adapter
[x] action-expert router proxy
[x] context-only action readout auxiliary
[ ] direct FAST/action-token CE restored in PICF
```

Conclusion:

```text
This is the only remaining high-information branch.  PICF context contains
action signal, but the deployed action flow does not consume it beneficially
under the current suffix bridge.
```

## 4. Current Remote 2xA100 Gate

Remote machine:

```text
ssh -p 26120 root@px-cloud1.matpool.com
repo: /root/openpi_g22_20260604
GPUs: 2 x A100-PCIE-40GB
```

Remote status at ledger creation:

```text
GPU 0/1 free.
No tmux sessions running.
```

Validated G22 command family:

```text
WANDB_MODE=disabled PYTHONPATH=/root/openpi_g22_20260604:/root/openpi_g22_20260604/src \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/picf_resume_from_args_json.py \
  --args-json /mnt/checkpoints/picf_core/picf_core/picf_g10a_taskuniform_k4_adapter_norm_20260603_042659/args.json \
  --set resume_checkpoint=/mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000 \
  --set action_context_stopgrad=False \
  --set semantic_action_context_readout_aux_weight=0.1
```

Accepted as a real training mode:

```text
world_size=2
accum_steps=2
logical_batch_task_count=4
calvin_balanced_bucket_sampler=True
calvin_bucket_sampling_mode=task_uniform
logical_batch_bucket_normalization=True
semantic trainable scope=action_head_and_adapter
point/visual/tactile pretrains frozen
```

Rejected as a final fix:

```text
readout improves but canonical action worsens.
```

## 5. Next Valid Experiment Plan

The next test must not repeat a rejected branch.  It must answer one of these:

```text
Q1. Can direct PICF-context action supervision change the canonical action path,
    not only a side readout?

Q2. Can restoring a FAST/action-token CE or equivalent representation target
    give the action expert a non-zero useful gradient from PICF context?

Q3. If Q1/Q2 fail, is the remaining issue action-expert capacity/conflict,
    justifying action-expert MoE rather than sampler/optimizer tweaks?
```

### G24-A: Strict local/remote audit

```text
[x] py_compile modified training/model files
    scripts/picf_resume_from_args_json.py
    scripts/picf_core_train.py
    src/openpi/picf/paligemma/wrapper.py
    src/openpi/picf/paligemma/config.py
    src/openpi/picf/policy.py

[x] wrapper action-readout/router/action-adapter tests
    uv run pytest -q src/openpi/picf/paligemma/wrapper_test.py \
      -k 'action_context_readout or action_adapter_only or router'
    result: 5 passed

[x] policy action debug metric tests
    uv run pytest -q src/openpi/picf/policy_test.py -k 'action'
    result: 8 passed

[x] git diff --check

[x] strict grep/dataflow follow-through for every method above
    confirmed code paths:
      _calvin_prompt_bucket
      _compute_bucket_sampling_weights
      _dynamic_bucket_sampling_weights
      _bucket_sequence_for_logical_step
      _logical_batch_loss_scales
      _pcgrad_project_and_sum
      _cagrad_project_and_sum
      _apply_action_context_adapter
      _compute_action_context_readout_aux
      _apply_action_expert_router
      policy pi_context_readout_* / pi_action_expert_router_* metrics

[x] trainable-scope follow-through
    confirmed current G24-B is not sampler-only:
      trainable_scope=action_head_and_adapter
      trainable modules:
        action_in_proj/action_out_proj/time_mlp
        action_context_in/q/k/v/out projections
        action_context_readout q/k/v/out projections and query
      frozen modules:
        point/visual/tactile pretrains
      nearly frozen:
        PICF core lr=2e-08, effectively diagnostic/stabilized
    implication:
      If native action still fails while readout learns, the failed edge is not
      "no action-side trainable parameters"; it is that the canonical action
      objective is not forced to use the PICF context representation.

[x] native FAST follow-through
    confirmed:
      src/openpi/models/tokenizer.py has FASTTokenizer
      src/openpi/transforms/_base.py has TokenizeFASTInputs
      src/openpi/models/pi0_fast.py trains token CE over token_loss_mask
    blocking fact:
      src/openpi/picf/paligemma/wrapper.py drops the outer LM heads through
      _drop_unused_generation_heads(), so PICF cannot honestly claim deployed
      FAST/action-token CE without a new architecture branch.

[x] remote sync after local doc/code update
    remote repo: /root/openpi_g22_20260604
    remote py_compile: passed
```

### G24-B: Fast remote confirmation

Purpose:

```text
Confirm the current two-card machine can run the validated G22/G23 production
contract after local sync.  This is not a new scientific claim.
```

Run length:

```text
100-150 optimizer steps.
Stop early if readout improves while canonical action again worsens.
```

Launch:

```text
remote:
  ssh -p 26120 root@px-cloud1.matpool.com

tmux:
  picf_g24_allmethod_k4_readout_300_20260604

log:
  /mnt/picf_run_logs/picf_g24_allmethod_k4_readout_300_20260604.log

exp:
  picf_g24_allmethod_k4_readout_from11000_to11300_20260604
```

Configuration gate observed at startup:

```text
world_size=2
accum_steps=2
logical_batch_task_count=4
calvin_balanced_bucket_sampler=True
calvin_bucket_sampling_mode=task_uniform
calvin_bucket_sample_without_replacement=True
logical_batch_bucket_normalization=True
dynamic_mixing=False
gradient_surgery=off
semantic trainable scope=action_head_and_adapter
point/visual/tactile pretrains frozen
```

Observed metrics:

```text
step 11010:
  loss_action_default_equiv = 0.040727
  pi_context_readout_loss   = 0.051952
  pi_context_readout_mse    = 0.105261
  loss_total_minus_action   = 0.010230
  logical_batch_distinct_bucket_count = 4

step 11020:
  loss_action_default_equiv = 0.042568
  pi_context_readout_loss   = 0.035048
  pi_context_readout_mse    = 0.074828
  loss_total_minus_action   = 0.009554
  logical_batch_distinct_bucket_count = 4

step 11030:
  loss_action_default_equiv = 0.042933
  pi_context_readout_loss   = 0.029408
  pi_context_readout_mse    = 0.060580
  loss_total_minus_action   = 0.009229
  logical_batch_distinct_bucket_count = 4

step 11040:
  loss_action_default_equiv = 0.046909
  pi_context_readout_loss   = 0.032755
  pi_context_readout_mse    = 0.068425
  loss_total_minus_action   = 0.010005
  logical_batch_distinct_bucket_count = 4
```

Interpretation:

```text
The readout branch is learning quickly, so PICF context is not information-free.
The native PI0.5 action flow does not improve with it in the same gate; it is
slightly worse than the first measured row while the readout branch improves.
This repeats the G22 causal boundary and makes sampler-only, optimizer-only,
loss-scale-only, PCGrad/CAGrad-only, and router-only repeats low value.
```

Decision:

```text
Stopped at step 11040.  The failure condition triggered before the planned
100-150 step budget:
  - direct PICF-context readout learns quickly;
  - canonical action MSE worsens from the first measured row.

The current all-method K4 contract is therefore not a final convergence fix.
It is a diagnostic proof that the remaining high-value work is the native
action/context bridge, not another sampler or optimizer repeat.
```

### G24-C: Non-duplicate architecture branch

Required implementation before full test:

```text
[ ] inspect native OpenPI FAST/token generation code
[ ] inspect PICF wrapper generation-head removal
[ ] decide whether to restore FAST/action-token CE or implement an equivalent
    canonical action-context auxiliary that directly affects deployed action
[ ] add metrics:
    loss_action_fast_ce or equivalent
    flow_context_gain_*
    fast_context_gain_*
    action_context_grad_norm_*
[ ] test locally
[ ] then run 100-300 optimizer steps
```

This is the only branch that can honestly claim to address the current
root-cause boundary.

Concrete branch split:

```text
G24-C1: canonical-context reliance audit
  Goal:
    measure whether the deployed flow expert changes when PICF context is
    present/zeroed/rolled at identical noisy-action/time samples.
  Pass:
    context changes native flow prediction and improves canonical action MSE.
  Fail:
    context readout learns but native flow remains context-invariant.
  Why:
    This is the smallest mathematically honest test of dA_native/dC_picf.

  Completed perturbation result, same 11000 checkpoint and same stratified12
  windows.  The table uses each command's final complete summary over 24 eval
  windows because the script emits two eval_summary blocks and small residual
  nondeterminism makes cross-block comparison misleading:

    mode          loss_action_default_equiv  action_ctx_tokens  adapter_gate  adapter_resid
    none_ref      0.0311465562               28.0               0.119203      0.262958
    zero          0.0309388494               28.0               0.119203      0.143332
    token_roll    0.0311466491               28.0               0.119203      0.262958
    sign_flip     0.0310968451               28.0               0.119203      0.261461
    rms_noise     0.0309325755               28.0               0.119203      0.136885
    disable_picf  0.0308614231                0.0               0.000000      0.000000

  Current inference:
    The native action path is not mathematically context-invariant because
    adapter residual statistics change when context is zeroed or noised.
    However, zero/noise/disable do not worsen canonical action loss; disabling
    PICF context is slightly best on this exact-window audit.  Therefore the
    current action-visible PICF context is weak/neutral/noisy for the deployed
    PI0.5 flow path, despite being motor-readable by the side readout.

  Consequence:
    The next valid branch is not sampler-only, optimizer-only, PCGrad-only,
    raw-overlap-only, or bridge-strength-only.  The next branch must place
    action supervision directly on the deployed action path.

G24-C2: direct deployed-path bridge
  Goal:
    move action supervision from a side readout into the deployed action path.
  Allowed designs:
    1. restore a PyTorch-side FAST/action-token CE path; or
    2. add an equivalent canonical action-context objective whose prediction is
       part of the same suffix/adaptation path used at inference.
  Disallowed:
    another side-only readout that can succeed without changing deployed action.

G24-C3: action-expert MoE
  Trigger:
    only if G24-C2 creates a useful action-context gradient but task-family
    conflict remains visible through gradient-cosine / per-bucket action trends.
  Constraint:
    MoE may live inside the action expert only.  Do not MoE the whole VLM.
```

The next 2-3 hour budget should therefore be:

```text
1. Treat G24-C1 as completed.
2. Implement C2 because C1 shows current PICF context is not beneficial to
   native action even though it contains side-readable motor information.
3. Run a 100-300 step K4 task-uniform gate.
4. Accept only if loss_action_default_equiv improves, not merely if readout
   or auxiliary losses improve.
```

## 6. Explicit Anti-Repeat Rules

Do **not** launch these again as standalone fixes:

```text
1. task_uniform K4/K8 only;
2. temperature-only sampling;
3. explicit ratio-only sampling;
4. dynamic PiKE-only sampling;
5. EMA bucket action scaling only;
6. PCGrad/CAGrad only;
7. action-head-only or adapter-only scope;
8. action router only;
9. L1/Huber action objective only;
10. raw-overlap or anchor-only repair;
11. fixed64 probes without exact-window category controls;
12. SAM-based repairs.
```

They are either already tested, rejected on scaling/mismatch grounds, or not
causal for the current plateau.

## 7. Current Bottom Line

What is solved:

```text
1. Task-balanced logical batch infrastructure.
2. Per-bucket normalization and metric visibility.
3. Dynamic mixing and scoped gradient surgery as available diagnostics.
4. Action context contains motor-readable information under direct readout.
5. SAM is archived out of the main route.
```

What is not solved:

```text
1. Native PI0.5 action flow still does not beneficially consume PICF context.
2. Direct PICF-side FAST/action-token CE is not restored.
3. An equivalent native-flow context-gain objective has not been implemented.
4. Action-expert MoE is not justified until the action/context supervision
   branch has been tested.
```

Operational decision:

```text
The next GPU run should either be a short confirmation of G22 after sync or the
first test of a true action/context-boundary implementation.  Anything else is
likely a repeat.
```
