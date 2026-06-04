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
[ ] Record step200 and compare against G26-B step200 = 0.0766.
[ ] Record step300 and compare against G26-B step300 = 0.0654.
[ ] Stop decisively if the branch is clearly worse after step200/300.
[ ] If it works, preserve command/log/checkpoint and write exact next deploy.
```

## 6. Deployment Part Checklist

```text
[x] Keep K4 task-balanced logical batch and per-bucket normalization.
[x] Keep temperature/ratio/dynamic mixing as explicit knobs, not defaults.
[x] Keep PCGrad/CAGrad as explicit diagnostics, not defaults.
[x] Keep SAM default off and archived.
[x] Keep action-context bridge guarded and measured by canonical action MSE.
[ ] Decide whether PI0.5-like ablation proves PICF/context is action-negative.
[ ] If yes, deploy stricter action-boundary isolation before any new full PICF
    long run.
[ ] If no, audit current trainer parity against historical 4-22 PI0.5 before
    any new PICF architecture change.
```
