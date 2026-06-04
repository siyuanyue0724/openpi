# PICF-AQR-OWM Action Descent Checklist

Date: 2026-06-05

Entry point:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_G26_FULL_VLA_METHOD_AUDIT_AND_NEXT_GATE_20260604.md
docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md
docs/PICF_AQR_OWM_G12_ALL_REQUESTED_VLA_METHODS_TEST_PLAN_20260603.md
docs/PICF_AQR_OWM_8H_FULL_VLA_REPAIR_MATRIX_20260603.md
```

Purpose:

```text
Prevent another action-loss experiment from repeating an already-rejected
sampler/LR/auxiliary branch.  Every new run must state which causal factor it
changes and which prior evidence it is not duplicating.
```

## 1. Core Mathematical Question

The target action objective is a mixture over CALVIN task buckets:

```text
L(theta) = sum_b q_b L_b(theta)
G(theta) = sum_b q_b grad L_b(theta)
```

A physical micro-batch that sees only one or two buckets per optimizer step has
high task-gradient variance:

```text
Var[g] = E_b ||grad L_b(theta) - G(theta)||^2
```

The maintained repair direction is therefore:

```text
1. make every optimizer update approximate G(theta) by task-balanced logical
   batches;
2. normalize per-bucket contributions so sampled-count imbalance does not
   dominate;
3. only if balanced updates still fail, inspect the action/semantic boundary
   and action path capacity.
```

This is consistent with the 2025-2026 VLA method family:

```text
VLA Foundry:
  probabilistic dataset/task mixing, batch balancing, gradient accumulation,
  and dataset-aware normalization.

ABot-M0:
  task/embodiment balance and action standardization for heterogeneous robot
  data.

PiKE:
  dynamic mixing only after static mixing is insufficient.

Knowledge Insulation / pi0.5:
  continuous action gradients need a controlled boundary to semantic/VLM
  representations.

OpenVLA-OFT:
  continuous action chunks and simple robust action losses are strong baselines,
  but objective changes must be judged by canonical action MSE.
```

## 2. Already Implemented And Tested

These should not be repeated as standalone fixes unless the code path changed.

| Method | Code path | History result | Current decision |
| --- | --- | --- | --- |
| task-uniform logical batch | `scripts/picf_core_train.py::_compute_bucket_sampling_weights` and `_bucket_sequence_for_logical_step` | K4 coverage verified; necessary but insufficient | keep default |
| gradient accumulation as logical batch | `WORLD_SIZE * accum_steps`, logged as `logical_batch_global_micro_count` | K4 feasible on 2x40GB; K8/K12 resource-limited | keep K4 where possible |
| per-bucket normalization | `_logical_batch_loss_scales` | deployed in G12/G26 | keep mandatory |
| temperature sampling | `calvin_bucket_sampling_mode=temperature` | tested; did not fix action | config only |
| trajectory/data-proportional sampling | `calvin_bucket_sampling_mode=trajectory` | G26 Contrast C worsened action by step600 | rejected as root fix |
| explicit ratio mixing | `calvin_bucket_weight_spec` | tested; no action fix | config only |
| dynamic PiKE-style mixing | `_dynamic_bucket_sampling_weights` | tested; structure transiently improved, action not fixed | not default |
| PCGrad | `_pcgrad_project_and_sum` | tested; action worsened or did not fix | not default |
| CAGrad | `_cagrad_project_and_sum` | tested; action worsened or did not fix | not default |
| Huber/L1 action-flow objective | `semantic_action_flow_loss` | tested; canonical MSE did not improve | not root fix |
| context-only action readout | `_compute_action_context_readout_aux` | proved PICF context is motor-readable but not native-flow useful alone | diagnostic |
| deployed flow residual | `_apply_action_context_flow_residual` | G25/G26 live; not enough to break platform | keep guarded |
| PICF-local FAST/action-token auxiliary | `_compute_action_context_token_aux` | G26 short gate passed, but long action still platformed | keep guarded |
| action expert router proxy | `_apply_action_expert_router` | tested; not sufficient | off by default |
| raw-overlap-only repairs | AQR/MAPG losses and overlap filters | repeatedly failed to explain action | do not reopen alone |
| SAM proposal branch | sidecar/SAM docs | noisy and archived | keep default off |

## 3. Live Control: PI0.5-Like K4 Ablation

Current run:

```text
tmux:
  picf_g26pi05like_k4_ablation_30k_20260605

log:
  /mnt/picf_run_logs/picf_g26pi05like_k4_ablation_30k_20260605.log

contract:
  picf_mode=ablated
  native PI0.5 semantic action path
  task_uniform K4 logical batch
  per-bucket normalization
  accum_steps=2
  lr=2e-4, warmup=600
```

Trainable-boundary follow-through:

```text
semantic_trainable_scope=backbone_only

This is not action-head-only.  In
src/openpi/picf/paligemma/wrapper.py::_apply_trainable_scope,
backbone_only/model_only trains `self.model.parameters()`, which includes the
PaliGemma/Gemma action expert stack, while freezing wrapper-local
`action_in_proj`, `action_out_proj`, and time MLP calibration heads.  The unit
test `test_pi0_trainable_scope_backbone_only_matches_historical_full_cotrain_boundary`
explicitly records this as the historical full-cotrain boundary.

Therefore this control is a valid current-shell PI0.5-like action-capacity
probe.  It is still not exact 4-22 parity because trainer shell, token budget,
sampling, FSDP, and optimizer schedule differ.
```

What it changes:

```text
It removes PICF recurrent/control/future semantics while preserving the current
trainer shell and K4 sampling.  This directly tests whether G26-B action
platform is caused by PICF/context integration or by the current semantic
action path / data mixture itself.
```

Step50:

```text
loss_action_default_equiv = 0.1291981190
logical_batch_distinct_bucket_count = 4
loss_total_minus_action = 0.0
lr = 1.6666666667e-05
```

Step50 reading:

```text
Not enough evidence.  K4 coverage and action-only loss are confirmed, but the
600-step warmup is still early and the action scalar is not yet better than the
G26-B fresh reference.
```

Step100:

```text
loss_action_default_equiv = 0.0770749152
loss_action_active7 = 0.3494936228
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 3.3333333333e-05
```

Step100 reading:

```text
This is the first positive control signal.  The PI0.5-like ablation is clearly
better than G26-B fresh step100 (0.1165) and already near G26-B fresh step200
(0.0766).  Therefore "PICF semantics/context integration is action-negative or
action-slowing in G26-B" is now supported.

Still unresolved: whether this branch keeps descending toward the old
0.02-0.03 band or merely reaches the same 0.06-0.08 platform earlier.  Continue
to step200 before stopping.
```

Step150:

```text
loss_action_default_equiv = 0.0646321550
loss_action_active7 = 0.2944722176
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 5.0e-05

bucket action_default_equiv:
  block_lift          0.0680387956
  block_other         0.0678645276
  block_push          0.0495955158
  drawer              0.0641414492
  other               0.0664097131
  slider              0.0504070998
  switch_button_light 0.0881502455
```

Step150 reading:

```text
The PI0.5-like ablation has already reached the G26-B fresh step300 region
(G26-B step300 action_default ~= 0.0654) by step150.  This strengthens the
causal interpretation that the enabled PICF/context path is action-negative or
action-slowing under the current integration.  The remaining question is
whether this clean action path keeps descending toward the old 0.02-0.03 band
or stalls in the 0.05-0.06 band.
```

Step200:

```text
loss_action_default_equiv = 0.0641728789
loss_action_active7 = 0.2913683653
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 6.6666666667e-05

action components:
  pos     0.2413374484
  rot     0.2888981402
  gripper 0.4488718212

bucket action_default_equiv:
  block_lift          0.0580355006
  block_other         0.0733913263
  block_push          0.0562251920
  drawer              0.0697186454
  other               0.0571673639
  slider              0.0510929807
  switch_button_light 0.0893955454
```

Step200 reading:

```text
The branch remains action-positive relative to G26-B step200 (0.0766) and still
slightly beats G26-B step300 (0.0654).  However, step150 -> step200 is nearly
flat: 0.06463 -> 0.06417.  The hard bucket is still switch/button/light, and
the gripper component rebounded.  This rules out immediate failure, but it also
rules out "PICF removal alone instantly restores old 0.02-0.03 convergence".
Continue to step300; use step300 to decide whether this is an early plateau or
a temporary warmup/bucket-composition pause.
```

Step250:

```text
loss_action_default_equiv = 0.0656914562
loss_action_active7 = 0.2976432741
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 8.3333333333e-05

action components:
  pos     0.2287622839
  rot     0.3194335699
  gripper 0.4389153123
```

Step300:

```text
loss_action_default_equiv = 0.0640842691
loss_action_active7 = 0.2884802520
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 1.0e-04

action components:
  pos     0.2238104343
  rot     0.3115257025
  gripper 0.4133533239
```

Step300 reading:

```text
This is not a clean failure, but it is not the requested action-collapse fix.
The current-shell PI0.5-like ablation improves sharply from step50 to step150,
then stays in the 0.064-0.066 band through step300.  It proves that enabled
PICF/context was action-negative relative to a clean ablated path, but it does
not prove that K4 task-balanced sampling plus current G26 action shell recovers
the old 4-22/PI0.5 0.02-0.03 action band.

Decision: stop this branch at step300 and run a stricter parity experiment:
old 4-22 args/action shell + current K4 task-uniform logical batch.  This is
the non-repeated missing test.  If old args + K4 descends faster, the root is
the G26 action shell / optimizer / trainable-boundary delta.  If it also
plateaus, then the old train-stream metric and current balanced objective are
not directly comparable, or current code has a deeper parity regression.
```

## 3.5 Parity Control E: Old 4-22 Args + Current K4 Sampler

Run:

```text
picf_g27_oldargs_k4_pi05ablated_300_20260605
```

Purpose:

```text
Hold the new task-uniform logical-batch sampler, but return the action shell to
the archived 4-22 ablated PI0.5 args as much as the current code allows.

This specifically checks the remaining uncontrolled deltas from the G26
PI0.5-like ablation:
  semantic_lr_scale: old 0.25 vs G26 current 1.0
  grad_clip_norm: old 1.0 vs G26 current 5.0-percentile shell
  semantic_trainable_scope: old action_head_and_adapter vs G26 backbone_only
  historical action/prompt/state defaults in the 4-22 args file
  explicit semantic_action_context_flow_residual_enabled=False

This is not a repeat of task-uniform, temperature sampling, ratio mixing,
dynamic PiKE-style mixing, PCGrad/CAGrad, Huber/L1, context-only readout, SAM,
or raw-overlap repair.  Those are already recorded as implemented/tested or
diagnostic-only in the method audit.  This is a trainer/action-shell parity
test.
```

Implementation note:

```text
The old 4-22 args file predates several current trainer fields.  To avoid
false failures from missing Namespace attributes, the remote run builds a
complete args file by taking the G26-B args as a current-code default template,
then overlaying the archived 4-22 args and finally applying the K4/sampler
control overrides.  This preserves old values where they exist while filling
new fields with current defaults.

Effective critical values:
  semantic_lr_scale = 0.25
  grad_clip_norm = 1.0
  grad_clip_mode = percentile
  semantic_trainable_scope = action_head_and_adapter
  lr = 2e-4
  warmup_steps = 600
  burnin_steps = 0
  effective_unroll_steps = 3
  accum_steps = 2
  picf_mode = ablated
  calvin_bucket_sampling_mode = task_uniform
  logical_batch_task_count = 4
  logical_batch_bucket_normalization = true
```

The archived 4-22 args contained a nonzero burn-in setting.  Current code
correctly rejects `burnin_steps > 0` with `picf_mode=ablated`, because the clean
PI0.5 ablation has no PICF recurrent carry to burn in.  For the parity control,
`burnin_steps=0` is therefore a contract fix, not a tuning choice.

Tail:

```bash
ssh -i /tmp/picf_g22_key -p 26120 root@px-cloud1.matpool.com 'tail -f /mnt/picf_run_logs/picf_g27_oldargs_k4_pi05ablated_300_20260605.log'
```

Startup verification:

```text
04:47:18 remote log entered PICF Training.

world_size = 2
training_strategy = fsdp_full_shard
picf_mode = ablated
trainable_numel = 13,751,362
total_numel = 3,744,957,843
accum_steps = 2
effective_global_batch = 4
num_steps = 300
lr = 2e-4
min_lr = 2e-5
warmup = 600
unroll_steps = 2
burnin_steps = 0
effective_window_steps = 2
calvin_balanced_bucket_sampler = true
calvin_bucket_sampling_mode = task_uniform
logical_batch_task_count = 4
logical_batch_bucket_normalization = true
calvin_buckets = block_lift, block_other, block_push, drawer, other, slider, switch_button_light
```

Step50:

```text
loss_action_default_equiv = 0.1529971361
loss_action_active7 = 0.5082107782
loss_total_minus_action = 0.0
logical_batch_distinct_bucket_count = 4
lr = 1.6666666667e-05

action components:
  pos     0.4047880769
  rot     0.5090084076
  gripper 0.8160861731
```

Step50 reading:

```text
This is worse than the current-shell G26 PI0.5-like ablation at step50
(0.129198).  The old action shell did not produce the expected immediate early
drop.  Because step50 is still inside low-LR warmup, continue only to step100
as the final parity gate.  If step100 remains above the current-shell step100
(0.077075), stop this branch: the plateau is not fixed by restoring old
semantic_lr_scale/trainable-scope/clip defaults under K4 balanced sampling.
```

Step100:

```text
loss_action_default_equiv = 0.1471613646
loss_action_active7 = 0.4896215200
loss_total_minus_action = 0.0
lr = 3.3333333333e-05

progress-bar recent single-window values near step100:
  step97  0.3729
  step98  0.1508
  step99  0.0731
  step100 0.1399
```

Step100 decision:

```text
Stop G27 immediately.  It is clearly worse than the current-shell G26
PI0.5-like ablation at the same step:

  G26 current-shell K4 step100: 0.077075
  G27 old-args K4 step100:     0.147161

Therefore the action plateau is not solved by restoring old semantic_lr_scale,
old grad clip, or old action_head_and_adapter trainable scope under the current
balanced K4 sampler.  The root is not simply "G26 action shell differs from
4-22".

Current causal status:
  1. Full G26-B PICF/context slows action relative to clean ablated action.
  2. Clean current-shell ablated action improves early but plateaus around
     0.064 by step300.
  3. Old-args action shell under K4 is worse than the clean current-shell
     ablation and is rejected at step100.

Next non-repeated direction:
  keep the current-shell ablated action path as the better control;
  do not restore old action_head_and_adapter as default;
  investigate whether balanced K4 objective is intrinsically harder than the
  old online train stream, and whether the action model needs a stronger native
  action-representation objective rather than more sampler/optimizer repair.
```

## 4. Next Decision Tree

At step100/200/300 of the live ablation:

```text
If ablated action clearly beats G26-B fresh:
  PICF/context is action-negative under the current integration.  Next deploy
  a stricter Knowledge-Insulation-style action boundary and only reintroduce
  PICF context through a measured positive bridge.

If ablated action also plateaus or is worse:
  PICF is not the sole root cause.  Do not add more PICF losses.  Shift to
  training-definition parity with the historical 4-22/PI0.5 baseline:
    semantic length,
    optimizer schedule,
    action prompt/state path,
    exact CALVIN transition sampling,
    and direct PI0.5 trainer-vs-PICF-trainer shell differences.

If ablated descends early but later rebounds:
  the E21 diagnosis remains strongest: production K4 approximates the balanced
  objective too weakly.  The scalable next step is not fixed-window training;
  it is a stronger logical batch / bucket coverage implementation that remains
  usable on large heterogeneous datasets.
```

## 5. Experiment Part Checklist

```text
[x] Verify current run uses task_uniform K4.
[x] Verify current run has no PICF structure losses in loss_total.
[x] Record step50.
[x] Record step100 and compare against G26-B step100 = 0.1165.
[x] Record step150 and compare against G26-B step300 ~= 0.0654.
[x] Record step200 and compare against G26-B step200 = 0.0766.
[x] Record step300 and compare against G26-B step300 = 0.0654.
[x] Stop decisively if the branch is clearly worse after step200/300.
[x] Start old 4-22 args + current K4 sampler parity control.
[x] Record G27 step50/100 and decide whether action-shell parity fixes the
    plateau: no, rejected at step100.
```

## 6. Deployment Part Checklist

```text
[x] Keep K4 task-balanced logical batch and per-bucket normalization.
[x] Keep temperature/ratio/dynamic mixing as explicit knobs, not defaults.
[x] Keep PCGrad/CAGrad as explicit diagnostics, not defaults.
[x] Keep SAM default off and archived.
[x] Keep action-context bridge guarded and measured by canonical action MSE.
[x] Decide whether PI0.5-like ablation proves PICF/context is action-negative:
    yes, relative to G26-B fresh, but not sufficient to recover old action
    descent.
[ ] Audit current trainer parity against historical 4-22 PI0.5 before any new
    PICF architecture change.
```
