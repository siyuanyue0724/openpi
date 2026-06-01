# PICF-AQR-OWM Speed Regression Audit - 2026-05-28

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/picf_aqr_owm_202605/README.md
```

## 1. Symptom

The latest A7 full-cotrain continuation:

```text
script:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_ema7000_fullcotrain_lowersemantic_300_20260528.sh

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000
```

ran at about:

```text
39-46 sec/step
```

whereas the previous comparable full-cotrain runs were about:

```text
phase-stabilized p2 core0.005/action2: 24.7-25.5 sec/step
action-prefix EMA from 6800:              27-28 sec/step
policy-only action-semantic diagnostic:   23-24 sec/step
action-head-only diagnostic:               9-10 sec/step
```

So the slowdown is real and large enough that a 30K run should not be continued
blindly until the cause is isolated.

## 2. Ruled-Out Causes

### 2.1 Not A Simple Unfreeze Mistake

The slow run printed:

```text
world_size=2
training_strategy=fsdp_full_shard
trainable_scope=all
trainable_numel=3805808119
total_numel=4088407552
semantic=paligemma(trainable=True scope=all)
point=sonata(trainable=False)
visual=encoder(finetune_mode=frozen trainable=False)
tactile=encoder(trainable=False)
unroll_steps=2
burnin_steps=1
burnin_mode=state_only
effective_window_steps=3
```

This matches the intended normal cotrain class: PaliGemma/action interface and
PICF trainable; large perception pretrains frozen.  It is not the old
`full-pretrain-unfrozen` class that naturally costs about 40 sec/step.

### 2.2 Not Only Sidecar Size

The slow run and the previous fast action-prefix run use the same sidecar root:

```text
/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
proposal_nearest_max_gap=8
tracklet_memory_enabled=True
proposal_memory_enabled=True
```

Sidecar evidence can add some forward cost, but it cannot by itself explain
the jump from about 25-28 sec/step to about 40-46 sec/step because the same
sidecar family was already present in the faster run.

### 2.3 Not Optimizer Step Or Grad Clipping

Synchronized 3-step profiling from the slow run showed:

```text
step 7001:
  forward ~= 22.0 sec
  backward ~= 14.5 sec
  grad clip ~= 0.5 sec
  optimizer step ~= 1.8 sec

step 7002:
  forward ~= 27.6 sec
  backward ~= 15-16.6 sec
  grad clip ~= 0.3 sec
  optimizer step ~= 0.2 sec

step 7003:
  forward ~= 26.8 sec
  backward ~= 17.0 sec
  grad clip ~= 0.3 sec
  optimizer step ~= 0.2 sec
```

The slow part is model forward/backward, not optimizer maintenance.

### 2.4 Not Anchor Overlay

The profiling run used:

```text
anchor_overlay_interval=999999
```

and still reproduced the slow step time.  Overlay generation is not the root
cause of the 40+ sec/step path.

## 3. Current Hot Region

The synchronized phase timing localizes the slow forward to the transition
policy path, especially later unroll transitions:

```text
step 7003:
  burnin transition 0 ~= 2.5-4.2 sec
  transition 1 policy_forward ~= 8.4 sec
  transition 2 policy_forward ~= 12.0 sec
  transition 2 loss ~= 1.9 sec
```

Therefore the working hypothesis is:

```text
current code made the full-trainable PICF -> PI0.5 transition path heavier,
especially when gradients flow through the full transition stack.
```

The current evidence does not support:

```text
sidecar-only slowdown
optimizer-only slowdown
FSDP-only slowdown
overlay/logging-only slowdown
wrong unroll/burnin configuration
wrong frozen-pretrain configuration
```

## 4. Code Diff Suspects

Compared with the previous phase-stabilized code, the current code added or
expanded:

```text
src/openpi/picf/policy.py:
  train-time action-prefix teacher
  prefix safety / RMS / trust metrics
  inference safety wrappers

src/openpi/picf/core/pipeline.py:
  action-prefix stabilization metrics
  object-explanation quality metrics
  additional stage timing hooks

src/openpi/picf/core/training.py:
  action_prefix_trust loss component
  object-explanation quality gating

scripts/picf_core_train.py:
  semantic_trainable_scope
  object-explanation quality gates
  optimizer group grad metrics
  expanded debug metrics
```

The action-prefix EMA mechanism alone is not yet proven to be the speed root:
one older action-prefix EMA run remained about 27-28 sec/step.  The next
required test is a same-checkpoint profiling ablation with the teacher disabled.

## 5. Same-Checkpoint Ablation Results

### 5.1 Action-Prefix Teacher Off

Run:

```text
EXP=picf_speed_ablate_teacher_off_ema7000_2_20260528
RESUME_CHECKPOINT=.../picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000
ACTION_PREFIX_TEACHER_MODE=off
ACTION_PREFIX_TEACHER_BLEND=0.0
LAMBDA_ACTION_PREFIX_TRUST=0.0
NUM_TRAIN_STEPS=7002
OPENPI_DEBUG_CUDA_SYNC=1
```

Result:

```text
step7001:
  total step ~= 36.6 sec
  forward ~= 21.2 sec
  backward ~= 13.9-14.3 sec
  optimizer ~= 0.2-1.8 sec

step7002:
  total step ~= 42.4 sec
  steps_per_sec ~= 0.02236
```

Interpretation:

```text
Disabling the EMA teacher and trust loss does not return the run to the
historical 25-28 sec/step band.
```

Therefore:

```text
action-prefix teacher/trust is not the main speed-regression cause.
```

The hot region remains the full-trainable transition forward/backward path.

### 5.2 Semantic Action-Head-Only Under FSDP

Run:

```text
SEMANTIC_TRAINABLE_SCOPE=action_head_only
TRAINING_STRATEGY=fsdp_full_shard
```

Result:

```text
FAILED before training:
  ValueError: Must flatten tensors with uniform dtype but got
  torch.float32 and torch.bfloat16
```

Interpretation:

```text
This is a packaging/wrapping bug in the action-head-only semantic split under
the current root FSDP wrapper.  It does not explain the 40 sec/step slow path,
because the slow path uses semantic_trainable_scope=all and does enter
training.  It does block this specific speed ablation under FSDP.
```

Next action for the speed audit:

```text
rerun semantic action-head-only with DDP, or fix the FSDP ignored/mixed-dtype
wrapping before using this split as a production recipe.
```

### 5.3 Semantic Action-Head-Only Under DDP

Run:

```text
EXP=picf_speed_ablate_sem_actionhead_ddp_ema7000_2_20260528
RESUME_CHECKPOINT=.../picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000
SEMANTIC_TRAINABLE_SCOPE=action_head_only
TRAINING_STRATEGY=ddp
NUM_TRAIN_STEPS=7002
OPENPI_DEBUG_CUDA_SYNC=1
```

Result:

```text
trainable_numel = 454,540,039

step7001:
  total step ~= 13.7 sec
  forward ~= 11.1-11.2 sec
  backward ~= 1.8-1.9 sec
  optimizer ~= 0.26 sec

step7002:
  total step ~= 12.7 sec
  forward ~= 9.4 sec
  backward ~= 1.8 sec
  optimizer ~= 0.22 sec
```

Interpretation:

```text
The same checkpoint, same PICF transition graph, same sidecar root, and same
unroll/burnin become much faster when the full PaliGemma/Gemma semantic
backbone is not trainable.
```

This strongly localizes the speed regression to:

```text
full semantic-backbone gradient path under current full-cotrain settings.
```

The remaining nuance is historical: a previous full-semantic run with similar
reported trainable_numel ran about 25-28 sec/step.  Therefore the production
question is not simply "never train PaliGemma"; it is:

```text
why did the current full-semantic path become about 1.5-1.8x slower than the
previous full-semantic path?
```

The next comparison must inspect old-vs-current semantic FSDP wrapping and the
current policy/PICF action-prefix path around PaliGemma calls.

### 5.4 Semantic Backbone-Only / Historical Boundary

Code change:

```text
PaliGemma trainable_scope now supports:
  all
  backbone_only / model_only
  action_head_only
```

`backbone_only` restores the historical full-cotrain semantic boundary:

```text
train paligemma_with_expert
freeze wrapper-local action_in_proj/action_out_proj/time_mlp_in/time_mlp_out
```

This is not a capability removal.  It keeps PaliGemma/Gemma trainable while
excluding the wrapper-local PI0 flow calibration heads from the full semantic
FSDP gradient path.

Run:

```text
EXP=picf_speed_ablate_sem_backboneonly_ema7000_3_20260528
RESUME_CHECKPOINT=.../picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000
SEMANTIC_TRAINABLE_SCOPE=backbone_only
ACTION_PREFIX_TEACHER_MODE=off
ACTION_PREFIX_TEACHER_BLEND=0.0
LAMBDA_ACTION_PREFIX_TRUST=0.0
OPENPI_DEBUG_CUDA_SYNC=1
NUM_TRAIN_STEPS=7003
```

Result:

```text
trainable_numel ~= 3,803,642,327

step7001:
  total step ~= 33.8 sec
  forward ~= 26.0 sec
  backward ~= 6.6-6.7 sec

step7002:
  total step ~= 36.5 sec
  forward ~= 28.5 sec
  backward ~= 7.2-8.9 sec

step7003:
  total step ~= 35.1 sec
  forward ~= 23.4 sec
  backward ~= 8.8-9.1 sec
```

Interpretation:

```text
Freezing wrapper-local PI0 flow/time heads cuts roughly 7-10 sec from backward
relative to semantic_scope=all, so the trainable-scope expansion is a real
part of the regression.
```

However:

```text
forward remains 23-28 sec even after that fix.
```

Therefore the remaining speed root is not the action-prefix teacher and not
only the wrapper-local heads.  The next likely cause is semantic checkpointing
/ recompute under FSDP across the unrolled transition graph.

### 5.5 Backbone-Only With Semantic Checkpointing Explicitly Off

Script support added:

```text
SEMANTIC_GRADIENT_CHECKPOINTING=1 -> --semantic-gradient-checkpointing
SEMANTIC_GRADIENT_CHECKPOINTING=0 -> --no-semantic-gradient-checkpointing
```

Run:

```text
EXP=picf_speed_ablate_sem_backboneonly_nogc_ema7000_3_20260528
RESUME_CHECKPOINT=.../picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000
SEMANTIC_TRAINABLE_SCOPE=backbone_only
SEMANTIC_GRADIENT_CHECKPOINTING=0
ACTION_PREFIX_TEACHER_MODE=off
ACTION_PREFIX_TEACHER_BLEND=0.0
LAMBDA_ACTION_PREFIX_TRUST=0.0
OPENPI_DEBUG_CUDA_SYNC=1
OPENPI_DEBUG_PHASE_LIMIT=7003
NUM_TRAIN_STEPS=7003
```

Result:

```text
trainable_numel ~= 3,803,642,327
semantic=paligemma(trainable=True scope=backbone_only)

step7001:
  total step ~= 35.25 sec
  forward ~= 25.6-25.8 sec
  backward ~= 6.8-6.9 sec
  grad clip ~= 2.0 sec

step7002:
  total step ~= 35.81 sec
  forward ~= 26.3-26.4 sec
  backward ~= 8.8-8.9 sec
  grad clip ~= 0.15 sec

step7003:
  total step ~= 34.93 sec
  forward ~= 23.4-23.8 sec
  backward ~= 9.1-9.5 sec
  grad clip ~= 0.13 sec
```

Interpretation:

```text
Turning off semantic gradient checkpointing does not materially improve the
synchronized profiling result relative to backbone_only.
```

The remaining cost is now localized more tightly after code inspection:

```text
backbone_only reproduced the old trainable parameter set, but initially did not
reproduce the old native semantic gradient-checkpointing boundary.
```

Old phase-stabilized code did both of the following:

```text
1. train only self.model / paligemma_with_expert;
2. enable native gradient checkpointing whenever the semantic backbone trains.
```

The first local backbone_only patch restored (1) but accidentally missed (2),
which forced the trainable PaliGemma/Gemma path to run near the 40GB memory
ceiling and left the step time in the 35-40 sec band.

The code fix is therefore structural, not a hyperparameter workaround:

```text
if trainable_scope in {all, backbone_only, model_only}:
  keep native PaliGemma/Gemma gradient checkpointing eligible
if trainable_scope == backbone_only/model_only:
  still freeze wrapper-local action/time heads
```

This exactly matches the historical fast full-cotrain boundary for
`backbone_only/model_only`.

The phase timing confirms data loading, sidecar IO, optimizer, and overlay
generation are not the primary bottleneck:

```text
window_load/sample_validate < 0.5 sec
optimizer_step           < 0.3 sec
overlay disabled
```

Important nuance:

```text
OPENPI_DEBUG_CUDA_SYNC=1 makes this a synchronized profiling measurement, not
the exact production wall-clock speed.
```

Therefore the next test is a non-debug production-speed smoke with the same
fixed training boundary:

```text
SEMANTIC_TRAINABLE_SCOPE=backbone_only
SEMANTIC_GRADIENT_CHECKPOINTING=0
ACTION_PREFIX_TEACHER_MODE=off
ACTION_PREFIX_TEACHER_BLEND=0.0
LAMBDA_ACTION_PREFIX_TRUST=0.0
ANCHOR_OVERLAY_INTERVAL=999999
```

### 5.6 Root-Cause Fix: Restore Native GC For Backbone-Only

Patch:

```text
src/openpi/picf/paligemma/wrapper.py
  _trains_semantic_backbone() -> true for all/backbone_only/model_only
  __init__ enables native semantic gradient checkpointing when that is true
  _apply_trainable_scope() still freezes wrapper-local action/time heads for
  backbone_only/model_only

scripts/experiments/.../run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh
  SEMANTIC_GRADIENT_CHECKPOINTING controls explicit --semantic-gradient-checkpointing
  or --no-semantic-gradient-checkpointing
```

Local verification:

```text
python -m py_compile scripts/picf_core_train.py \
  src/openpi/picf/paligemma/wrapper.py \
  src/openpi/picf/paligemma/config.py \
  src/openpi/picf/paligemma/wrapper_test.py

python -m pytest src/openpi/picf/paligemma/wrapper_test.py -q
  36 passed

bash -n scripts/experiments/picf_aqr_owm_202605_active/\
  run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh
```

Next required remote test:

```text
SEMANTIC_TRAINABLE_SCOPE=backbone_only
SEMANTIC_GRADIENT_CHECKPOINTING=1
ACTION_PREFIX_TEACHER_MODE=off
ACTION_PREFIX_TEACHER_BLEND=0.0
LAMBDA_ACTION_PREFIX_TRUST=0.0
```

Expected result if the root cause is correctly fixed:

```text
production-speed smoke should move back toward the historical 25-28 sec/step
band, not the 35-46 sec/step regression band.
```

## 6. Required Next Checks

Run only short profiling jobs before any new long run:

```text
1. Same current code, same ckpt, non-debug production timing:
   SEMANTIC_TRAINABLE_SCOPE=backbone_only
   SEMANTIC_GRADIENT_CHECKPOINTING=0
   ACTION_PREFIX_TEACHER_MODE=off
   ACTION_PREFIX_TEACHER_BLEND=0.0
   LAMBDA_ACTION_PREFIX_TRUST=0.0
   OPENPI_DEBUG_CUDA_SYNC unset

   Decision:
     if production speed returns near 25-28 sec/step, the actionable fix is
     backbone_only semantic scope plus teacher/trust off for long training.
     if production speed remains near 35-40 sec/step, the remaining regression
     is current semantic forward/FSDP call multiplicity rather than profiler
     synchronization.

2. Old phase-stabilized code, same profiling env:
   verify forward/backward phase split for the known 25 sec/step baseline.

3. Compare old-vs-current semantic FSDP wrapper:
   check whether current runtime-leaf FSDP wrapping, ignored-state split, or
   trainable-scope handling changed all-layer materialization behavior.
```

Decision rule:

```text
If backbone_only + no semantic checkpointing is only slow under CUDA-sync
profiling:
  do not treat 35 sec/step as production speed.

If it is also slow without profiling:
  inspect semantic forward call multiplicity and current FSDP wrapping.

If old phase-stabilized code profiles near 25 sec/step with similar transition
shape:
  the regression is in the current code diff, not the data/checkpoint.
```

## 7. Current Conclusion

The user's concern is correct:

```text
the slowdown is not explained by "sidecar became larger" alone.
```

The measured bottleneck was forward/backward through the current full-cotrain
transition path when `semantic_trainable_scope=all`.

The completed same-checkpoint smoke isolated the actionable root:

```text
scope=all:
  trains paligemma_with_expert plus wrapper-local action/time heads
  observed ~= 39-46 sec/step

scope=backbone_only:
  trains paligemma_with_expert only
  freezes wrapper-local action/time heads
  preserves native semantic gradient checkpointing
  observed ~= 25-27 sec/step
```

This matches the historical fast boundary:

```text
previous phase-stabilized full-cotrain:
  semantic=paligemma(trainable=True)
  only self.model / paligemma_with_expert parameters required grad
  native semantic gradient checkpointing enabled
  ~= 24.5-25.5 sec/step
```

Therefore the production fix is:

```text
default semantic_trainable_scope = backbone_only
keep semantic_gradient_checkpointing enabled
reserve semantic_trainable_scope = all for explicit diagnostics only
```

`backbone_only` is still PaliGemma cotrain.  It does not freeze the semantic
backbone; it only excludes wrapper-local restored PI0 flow/time calibration
heads from the trainable semantic FSDP path.

Validated speed smoke:

```text
log:
  /mnt/picf_run_logs/
    picf_speed_prod_sem_backboneonly_gc_log50_ema7000_10_20260528.log

config:
  SEMANTIC_TRAINABLE_SCOPE=backbone_only
  SEMANTIC_GRADIENT_CHECKPOINTING=1
  ACTION_PREFIX_TEACHER_MODE=off
  LOG_INTERVAL=50
  ANCHOR_OVERLAY_INTERVAL=999999

result:
  step7001 ~= 26.71 sec/step
  step7002 ~= 26.49 sec/step
  step7003 ~= 25.74 sec/step
  step7004 ~= 25.15 sec/step
  step7005 ~= 25.11 sec/step
  step7006 ~= 24.99 sec/step
```

Code-level guard now applied:

```text
scripts/picf_core_train.py:
  --semantic-trainable-scope default -> backbone_only

src/openpi/picf/paligemma/config.py:
  trainable_scope default -> backbone_only

scripts/experiments/.../run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh:
  SEMANTIC_TRAINABLE_SCOPE default -> backbone_only
```

Local verification:

```text
python -m py_compile scripts/picf_core_train.py \
  src/openpi/picf/paligemma/wrapper.py \
  src/openpi/picf/paligemma/config.py \
  src/openpi/picf/paligemma/wrapper_test.py

bash -n scripts/experiments/picf_aqr_owm_202605_active/\
  run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh

python -m pytest src/openpi/picf/paligemma/wrapper_test.py \
  scripts/picf_core_train_test.py::test_normalize_train_args_canonicalizes_semantic_trainable_scope \
  scripts/picf_core_train_test.py::test_validate_train_args_rejects_invalid_semantic_trainable_scope -q

38 passed
```

Remaining caution:

```text
semantic_trainable_scope=all is not deleted because it is useful for explicit
causal diagnostics, but it must not be used as the default production 30K
training recipe.
```

Follow-up validation launched:

```text
session:
  picf_a7_backbone_scope_speedfix_300_from7000_20260528

log:
  /mnt/picf_run_logs/
    picf_a7_backbone_scope_speedfix_300_from7000_20260528.log

resume:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000

target:
  7000 -> 7300

initial observed training speed:
  step7001 ~= 24.73 sec/step
  step7002 ~= 25.70 sec/step
```
