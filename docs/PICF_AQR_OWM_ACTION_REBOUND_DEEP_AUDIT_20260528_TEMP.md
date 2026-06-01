# PICF-AQR-OWM Action Rebound Deep Audit - 2026-05-28

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_ACTION_REBOUND_CAUSAL_PLAN_20260528.md
docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_FOLLOWTHROUGH_20260528.md
docs/PICF_AQR_OWM_SPEED_REGRESSION_AUDIT_20260528.md
```

## 1. Scope

This note audits the late action-loss rebound after the preserved A7 EMA
checkpoint reached a low action basin around step 7000.  The audit has two
separate questions:

```text
Q1. Why did some 2026-05-28 relaunches become slow?
Q2. After fixing speed, does the original action rebound still reproduce under
    the same recipe?
```

These questions must not be conflated.  A run that changes the loss recipe can
test runtime sanity, but it cannot establish the rebound root cause.

## 2. Code Dataflow Follow-Through

### 2.1 Action loss path

The training path is:

```text
scripts/picf_core_train.py::_PicfWindowTrainer.forward
  policy.forward_train_transition
    semantic_encoder.compute_action_flow_loss(...)
    flow_override["total"], ["action_pos"], ["action_rot"], ["action_gripper"]
  openpi.picf.core.training.compute_transition_loss(...)
```

When `action_loss_override` is present:

```math
L_a^{default} = L_{flow}
```

and the actual action objective is:

```math
L_a
=
L_{flow}
\frac{
  \lambda_p L_p + \lambda_r L_r + \lambda_g L_g
}{
  2 L_p + 2 L_r + 2 L_g
}
```

because `PicfTransitionLossConfig` default action weights are `2,2,2`.
Therefore:

```text
ACTION_LOSS_WEIGHT=2.0 -> loss_action == loss_action_default_equiv
ACTION_LOSS_WEIGHT=1.0 -> loss_action ~= 0.5 * loss_action_default_equiv
```

Use `loss_action_default_equiv` for cross-run comparison against older 4-22 and
A7 records.

### 2.2 Non-action terms

The total loss is:

```math
L =
L_a
+ L_{physical}^{cap}
+ L_{semantic}^{cap}
+ L_{align}^{cap}
```

where:

```text
loss_total_minus_action = L - L_a
```

The old EMA run had object-scaffold decay:

```text
OBJECT_SCAFFOLD_DECAY_MODE=cosine
OBJECT_SCAFFOLD_DECAY_START_STEP=0
OBJECT_SCAFFOLD_DECAY_END_STEP=1500
OBJECT_SCAFFOLD_DECAY_FLOOR=0.03
```

Since the run resumes at step 7000, the structural/object scaffold is already
at the floor.  This is why the old EMA rebound window had:

```text
loss_total_minus_action ~= 0.010-0.012
```

A 2026-05-28 speedfix smoke accidentally used the slot-comprehensive script
directly with:

```text
ACTION_LOSS_WEIGHT=1.0
ACTION_PREFIX_TEACHER_MODE=off
ACTION_PREFIX_NORM_MODE=none
OBJECT_SCAFFOLD_DECAY_MODE=none
```

At step 7050 it produced:

```text
loss_action_default_equiv = 0.05565
loss_total_minus_action   = 0.22847
```

That run is valid as a speed/runtime sanity test but invalid as a causal
reproduction of the original EMA rebound, because the non-action budget is more
than twenty times larger than the old rebound window.

### 2.3 Optimizer and scheduler

The trainer loop computes LR from the current global step:

```python
lr = _lr_for_step(step, base_lr=args.lr, warmup_steps=args.warmup_steps,
                  min_lr=args.min_lr, total_steps=args.num_train_steps)
```

Therefore shortening `NUM_TRAIN_STEPS` changes the LR schedule even if the same
checkpoint is resumed.  Rebound gates must keep:

```text
NUM_TRAIN_STEPS=30000
```

The model-only checkpoint mode means optimizer state is not saved in production
checkpoints.  Resuming from a model-only checkpoint is a fresh-optimizer probe.
This is not wrong, but it must be recorded as a phase-boundary/fresh-optimizer
experiment, not a byte-exact continuation.

## 3. Locked Historical Evidence

### 3.1 Old EMA action-interface run

```text
run: picf_a7_actionprefix_ema_from6800_action2_30k_20260527
```

Important points:

```text
step7000:
  loss_action_default_equiv = 0.02127
  loss_total_minus_action   = 0.01021
  active_overlap            = 0.06998
  downstream_overlap        = 0.07267

step8300:
  loss_action_default_equiv = 0.04701
  loss_total_minus_action   = 0.01060
  active_overlap            = 0.09844
  downstream_overlap        = 0.10713
  prefix teacher cos        ~= 0.99999
```

Interpretation:

```text
The old failure is action-specific low-basin escape.  It is not raw reserve
overlap, not broad structural-loss explosion, and not an EMA-prefix metric
failure.
```

### 3.2 Policy-only / stationary PICF windows

Phase-stabilized policy-only from 6500->6800 had:

```text
loss_total_minus_action = 0
loss_action_default_equiv ~= 0.022-0.029
active/downstream overlap ~= 0.08-0.14
```

The 2026-05-28 EMA step7000 policy-only repeats with semantic/action trainable
also reached low action bands in short windows.  This confirms:

```text
A fixed PICF prefix is action-trainable.
```

It does not prove permanent freeze is the final answer, because the design goal
is cotrain.

### 3.3 Scalar LR-only controls

Lowering `PICF_CORE_LR_SCALE` alone weakened but did not remove the rebound:

```text
core0.02 and core0.005 both reproduced action rebound near the 7000-7350 band.
```

Therefore the remaining issue is not solved by one smaller PICF LR.

## 4. Hypothesis Matrix

| Hypothesis | Status | Evidence |
| --- | --- | --- |
| Raw same-role support overlap causes rebound | Weakened/rejected for late rebound | raw overlap stays 1.0, but active/downstream overlap stays below about 0.11 while action rebounds |
| Object/sidecar scaffold dominates action | Rejected for old EMA rebound | `loss_total_minus_action` stays about 0.010-0.012 in the rebound window |
| EMA prefix drift is the sole cause | Rejected | teacher delta RMS about 0.002 and cosine about 0.99999 while action still rebounds |
| Fixed PICF prefix cannot support action | Rejected | policy-only windows reach 0.018-0.029 action-default-equivalent bands |
| Scalar PICF LR too high is the only cause | Weakened | core0.02/core0.005 still rebound |
| Wrong semantic trainable boundary caused speed and possibly optimization contamination | Partly confirmed | `semantic_trainable_scope=all` trains wrapper-local PI0 heads and slows to 39-46 sec/step; `backbone_only` restores about 25 sec/step |
| Fresh optimizer / phase boundary changes basin dynamics | Still plausible | model-only checkpoints reinitialize optimizer; old fast drops often align with fresh optimizer phases |
| Full cotrain creates nonstationary semantic/PICF conditioning distribution | Strong remaining candidate | fixed-prefix windows work; cotrain windows can escape low action basin without visible slot collapse |

## 5. Correct Next Gate

The invalid speedfix smoke should not be used as the rebound verdict.  The next
gate must replay the old EMA recipe as closely as possible, changing only the
semantic trainability boundary required by the speed audit:

```text
resume:
  old EMA step7000 checkpoint

keep:
  NUM_TRAIN_STEPS=30000
  ACTION_LOSS_WEIGHT=2.0
  ACTION_PREFIX_NORM_MODE=rmsnorm
  ACTION_PREFIX_OUTPUT_GATE=0.70
  ACTION_PREFIX_TEACHER_MODE=ema
  ACTION_PREFIX_TEACHER_BLEND=1.0
  LAMBDA_ACTION_PREFIX_TRUST=0.02
  OBJECT_SCAFFOLD_DECAY_MODE=cosine
  OBJECT_SCAFFOLD_DECAY_END_STEP=1500
  OBJECT_SCAFFOLD_DECAY_FLOOR=0.03
  PICF_CORE_LR_SCALE=0.005

change:
  SEMANTIC_TRAINABLE_SCOPE=backbone_only
  SEMANTIC_GRADIENT_CHECKPOINTING=enabled
```

Pass/fail:

```text
Pass:
  step7050/7100/7150/7200 action_default_equiv does not climb back toward
  0.04-0.05, and total_minus_action remains about 0.010-0.015.

Fail:
  action_default_equiv reproduces the 0.04+ rebound while total_minus_action,
  active/downstream overlap, and prefix EMA remain healthy.  Then the root cause
  is genuine cotrain low-basin instability, not the speed/scope bug.
```

## 6. 2026-05-28 Correct Recipe Relaunch

The first speed-fix smoke was invalid for action-rebound causality because it
changed the loss/interface recipe.  The corrected live gate is:

```text
tmux:
  picf_a7_reboundgate_ema_recipe_scopefix_from7000_30k_20260528

log:
  /mnt/picf_run_logs/
    picf_a7_reboundgate_ema_recipe_scopefix_from7000_30k_20260528.log

resume:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000
```

Confirmed startup contract:

```text
world_size                         2
training_strategy                  fsdp_full_shard
trainable_scope                    all
num_steps                          30000
unroll_steps                       2
burnin_steps                       1
optimizer_checkpoint_mode          model-only
semantic                           paligemma(trainable=True scope=backbone_only)
Sonata / V-JEPA / AnyTouch         frozen
action_prefix_teacher_mode         ema
action_prefix_norm_mode            rmsnorm
action_prefix_output_gate          0.70
lambda_action_prefix_trust         0.02
lambda_action_pos/rot/gripper      2.0 / 2.0 / 2.0
PICF_CORE_LR_SCALE                 0.005
object_scaffold_decay_mode         cosine
object_scaffold_decay_floor        0.03
sidecar_root                       contact_motion_full_tracklets_clean_20260520
```

Confirmed runtime:

```text
step 7001-7008:
  about 26-28 sec/step
```

Confirmed checkpoint state:

```text
model.pt contains:
  core.action_prefix_teacher_tokens       shape=(4, 2048)
  core.action_prefix_teacher_initialized  1.0
```

Therefore the corrected relaunch is not silently reinitializing the EMA prefix
teacher.  It preserves the teacher state from the old EMA step7000 checkpoint.

This restores the historical speed band.  The first comparable JSON metric will
arrive at step 7050 because `LOG_INTERVAL=50`.

## 7. Current Causal State Before Step-7050

The evidence now separates two questions:

```text
speed regression:
  explained by semantic trainability boundary / wrapper-local trainable scope.
  The `backbone_only` scope restores the historical runtime band.

action rebound:
  not yet resolved by the speed fix.  It must be judged only from the corrected
  EMA recipe above.
```

The already falsified or weakened rebound hypotheses remain:

```text
raw same-role support overlap:
  not sufficient.  It remains 1.0 while action can improve and while
  active/downstream overlap stays low.

weak sidecar/object scaffold dominance:
  not sufficient in the old EMA window.  `loss_total_minus_action` stayed about
  0.010-0.013 while action rose from 0.021 to 0.047.

EMA prefix metric drift:
  not sufficient.  `pi_prefix_teacher_delta_rms` fell to about 0.002-0.004 and
  cosine stayed about 0.99999 while action still rebounded.

scalar PICF core LR only:
  not sufficient.  core0.02/core0.005 reproduced the same qualitative rebound.

fixed prefix unusability:
  weakened.  policy-only/action-head-only windows reached or maintained the
  0.018-0.029 action-default-equivalent band through the old failure point.
```

Remaining root-cause class:

```text
full cotrain low-basin instability:
  the action path reaches a sharp low-loss basin, then a moving semantic/PICF
  conditioning distribution, fresh optimizer state, or full semantic backbone
  update causes escape from that basin.
```

In local update form:

```math
\Delta L_a
\approx
-\eta_\theta \|\nabla_\theta L_a\|^2
+ \frac{1}{2}\eta_\theta^2 \operatorname{Tr}(H_\theta\Sigma_\theta)
+ \langle \nabla_z L_a, \Delta z \rangle
+ \Delta_{\text{semantic representation}}
+ \Delta_{\text{optimizer state}} .
```

The current run tests whether the prior slow branch was polluted by the
trainability boundary.  If the corrected recipe still rebounds with:

```text
loss_total_minus_action about 0.010-0.015
active/downstream overlap below about 0.15
prefix teacher delta/cos healthy
```

then the remaining root is not object binding, raw overlap, or sidecar loss.  It
is the action optimizer/representation stability problem near the low-loss
basin.

Important identifiability caveat:

```text
the step7000 checkpoint is model-only.
resuming it reinitializes AdamW optimizer state.
```

This was verified on the checkpoint directory:

```text
metadata.pt exists
model.pt exists
optimizer.pt absent
```

Therefore a no-rebound result from the corrected step7000 relaunch would not by
itself separate:

```text
A. semantic_trainable_scope=backbone_only fixed a trainability-boundary issue;
B. fresh optimizer state removed the old low-basin escape trajectory;
C. both effects contributed.
```

The old EMA run reached step7000 after about 200 optimizer steps from the 6800
handoff, so its 7350 rebound included nonzero Adam moments.  The current gate is
still necessary, but the strict causal split after a pass would require either:

```text
1. a same-checkpoint scope=all fresh-optimizer control with the old EMA recipe;
2. an optimizer-state-preserving continuation from a future checkpoint;
3. a paired reset/no-reset branch at the same low-loss model point.
```

## 8. Validation Already Re-Run Locally

Local code-contract checks:

```text
python -m py_compile
  scripts/picf_core_train.py
  scripts/serve_picf_policy.py
  scripts/calvin/evaluate_picf_policy.py
  src/openpi/picf/core/config.py
  src/openpi/picf/core/pipeline.py
  src/openpi/picf/core/training.py
  src/openpi/picf/policy.py
  src/openpi/picf/paligemma/wrapper.py
  src/openpi/picf/paligemma/config.py

python scripts/verify_picf_owm_contract.py
  PASS

python scripts/picf_owm_dataflow_trace.py --fail-on-fail
  PASS

python scripts/picf_owm_strict_diagnose.py --fail-on-fail
  PASS, with only expected runtime-artifact warnings when no metrics/eval path
  is passed.

python -m pytest
  src/openpi/picf/paligemma/wrapper_test.py
  src/openpi/picf/policy_test.py
  scripts/picf_core_train_test.py
  182 passed

python scripts/picf_latest_slot_deployment_audit.py --fail-on-fail
  16/16 PASS
```

The local checks verify the static/dataflow contract only.  They do not prove
that the live rebound is fixed; the live verdict requires the 7050/7100/7150/
7200 JSON metrics from the corrected run.

## 9. Next Decision Rule

At each JSON row, compare against old EMA and phase-p2 core0.005:

```text
old EMA:
  7000 action_default=0.02127
  7350 action_default=0.03591
  8300 action_default=0.04701

phase-p2 core0.005:
  7000 action_default=0.02128
  7350 action_default=0.03556
```

Decision:

```text
If corrected run stays near 0.021-0.030 through 7350:
  semantic trainability boundary was a major contributor; keep backbone_only
  as production scope and continue.

If corrected run repeats 0.035+ by 7350 while structural metrics are healthy:
  speed/scope is fixed, but action rebound remains.  Next intervention should
  be semantic/action optimizer stabilization, not another slot/overlap patch.

If corrected run has high total_minus_action:
  recipe is still misconfigured and must be stopped immediately.
```

## 10. 2026-05-28 FSDP Optimizer Group Bug

The first corrected EMA relaunch reached step 7050 with a healthy action point:

```text
step7050:
  loss_action_default_equiv          0.02259
  loss_total_minus_action            0.01185
  active_same_role_overlap_max       0.07677
  downstream_same_role_overlap_max   0.08620
  raw_same_role_overlap_max          1.0
  prefix_teacher_delta_rms           0.01123
  prefix_teacher_cos                 0.99980
```

This is not a rebound.  However the run exposed a stricter implementation
fault: no `lr_group_picf_core` / `grad_norm_group_picf_core` appeared in the
JSON metrics.  The log only contained:

```text
lr_group_semantic_backbone
lr_group_policy_head
```

Code follow-through found the cause:

```text
train()
  model = _wrap_model_for_training_strategy(model, ...)
  optimizer, optimizer_group_info = _build_optimizer(model, ...)

_build_optimizer()
  core_params = [p for name,p in model.named_parameters()
                 if name.startswith("core.")]
```

After FSDP wrapping, root parameter names are prefixed as:

```text
_fsdp_wrapped_module.core.*
```

Therefore `name.startswith("core.")` was false, and all remaining trainable
PICF core parameters fell through into `policy_head`.  This means the intended:

```text
PICF_CORE_LR_SCALE=0.005
```

did not actually slow the PICF core under the FSDP run.  The observed run is
therefore invalid as a slow-PICF rebound experiment, even though its first
action metric is healthy.

Patch:

```text
scripts/picf_core_train.py
  _canonical_param_owner_name(name)
    strips leading DDP/FSDP wrapper prefixes:
      module.
      _fsdp_wrapped_module.
  optimizer core/policy split uses the canonical owner name.
```

Regression:

```text
python -m pytest scripts/picf_core_train_test.py -k "optimizer or picf_core_group" -q
  16 passed

python -m py_compile scripts/picf_core_train.py scripts/picf_core_train_test.py
  PASS

remote note:
  run remote tests with PYTHONPATH=src so the checked-out repository code is
  used instead of the base /root/openpi package.
```

Consequence:

```text
stop:
  picf_a7_reboundgate_ema_recipe_scopefix_from7000_30k_20260528

restart after syncing patch:
  same checkpoint, same EMA/action/object recipe, same 30000-step horizon.

new hard gate:
  first JSON must contain lr_group_picf_core and grad_norm_group_picf_core.
  startup log must print Optimizer group rows even under compact DDP logging.
  If absent, the run is still invalid.
```

## 11. 2026-05-28 Fixed-Group Relaunch First Gate

Run:

```text
picf_a7_reboundgate_ema_recipe_scopefix_lrgroupfix_from7000_30k_20260528
```

First JSON at step 7050:

```text
loss_action_default_equiv              0.022599
loss_total_minus_action                0.010611
aqr_active_same_role_support_overlap   0.064367
aqr_downstream_same_role_support       0.076416
raw same_role_support_overlap          1.000000
loss_anchor_object_pull                0.196721
loss_anchor_pv                         0.515562
loss_slot_jepa                         1.282596
posterior_identity_switch_rate         0.207778
posterior_recycle_rate                 0.129221
pi_prefix_teacher_delta_rms            0.010998
pi_prefix_teacher_cos_to_teacher       0.999809
lr_group_picf_core                     3.175974e-7
lr_group_policy_head                   6.351949e-5
lr_group_semantic_backbone             2.223182e-5
grad_norm_group_picf_core              0.010836
grad_norm_group_policy_head            0.059713
grad_norm_group_semantic_backbone      0.256823
grad_elements_group_picf_core          116478464
grad_elements_group_policy_head        109708660
grad_param_tensors_group_picf_core     7
grad_param_tensors_group_policy_head   1
steps_per_sec                          0.03968
```

Interpretation:

```text
The hard gate passes. The PICF core is now actually on a 0.005x optimizer
timescale under FSDP. The group is non-empty and substantial rather than a
logging artifact: it carries about 116M gradient elements at the first gate.

Compared with the previous EMA run at step 7050:
  old action_default_equiv ~= 0.02610
  fixed-group action_default_equiv = 0.02260

This does not prove the late rebound is solved until the run passes the old
7350/7500 rebound window, but it proves the previous slow-core causal test was
not trustworthy and that the corrected isolation is materially better at the
first gate.
```

Second JSON at step 7100:

```text
loss_action_default_equiv              0.025269
loss_total_minus_action                0.010499
aqr_active_same_role_support_overlap   0.054673
aqr_downstream_same_role_support       0.066878
raw same_role_support_overlap          1.000000
loss_anchor_object_pull                0.178887
loss_anchor_pv                         0.510791
loss_slot_jepa                         0.884419
posterior_identity_switch_rate         0.211667
posterior_recycle_rate                 0.132983
pi_prefix_teacher_delta_rms            0.004599
pi_prefix_teacher_cos_to_teacher       0.999978
lr_group_picf_core                     3.171562e-7
lr_group_policy_head                   6.343125e-5
grad_norm_group_picf_core              0.000009
grad_norm_group_policy_head            0.000606
grad_norm_group_semantic_backbone      0.107791
steps_per_sec                          0.03821
```

7100 comparison:

```text
old EMA 7100 action_default_equiv       0.026631
fixed-group 7100 action_default_equiv   0.025269

old EMA 7100 active/downstream overlap  0.060113 / 0.084385
fixed-group 7100 active/downstream      0.054673 / 0.066878
```

Interpretation:

```text
The corrected run remains better than the old run at the second gate.  It is
not a monotonic action descent from 7050 to 7100, but it also is not the old
rebound signature: non-action loss is stable, active/downstream support overlap
improves, and the PICF core LR/grad group is now real.  The decisive window is
still 7350/7500, where the old run first crossed the clear rebound band.
```

Third JSON at step 7150:

```text
loss_action_default_equiv              0.022619
loss_total_minus_action                0.010886
aqr_active_same_role_support_overlap   0.095000
aqr_downstream_same_role_support       0.105814
raw same_role_support_overlap          1.000000
loss_anchor_object_pull                0.211574
loss_anchor_pv                         0.520893
loss_slot_jepa                         0.752890
posterior_identity_switch_rate         0.200000
posterior_recycle_rate                 0.130007
pi_prefix_teacher_delta_rms            0.004309
pi_prefix_teacher_cos_to_teacher       0.999980
lr_group_picf_core                     3.167125e-7
lr_group_policy_head                   6.334250e-5
grad_norm_group_picf_core              0.000007
grad_norm_group_policy_head            0.000500
grad_norm_group_semantic_backbone      0.233439
steps_per_sec                          0.03753
```

7150 comparison:

```text
old EMA 7150 action_default_equiv       0.027682
fixed-group 7150 action_default_equiv   0.022619

old EMA 7150 active/downstream overlap  0.094950 / 0.101576
fixed-group 7150 active/downstream      0.095000 / 0.105814
```

Interpretation:

```text
The corrected run is materially better on action at 7150 while active and
downstream overlap are approximately equal to the old run.  This argues against
"worse binding/overlap" as the reason for the action improvement.  The live
causal variable that changed is the real PICF-core optimizer timescale.
```

Fourth JSON at step 7200:

```text
loss_action_default_equiv              0.018195
loss_total_minus_action                0.009508
aqr_active_same_role_support_overlap   0.064859
aqr_downstream_same_role_support       0.062881
raw same_role_support_overlap          1.000000
loss_anchor_object_pull                0.084980
loss_anchor_pv                         0.510078
loss_slot_jepa                         0.793782
posterior_identity_switch_rate         0.200556
posterior_recycle_rate                 0.129953
pi_prefix_teacher_delta_rms            0.003813
pi_prefix_teacher_cos_to_teacher       0.999985
lr_group_picf_core                     3.162662e-7
lr_group_policy_head                   6.325324e-5
grad_norm_group_picf_core              0.000031
grad_norm_group_policy_head            0.002848
grad_norm_group_semantic_backbone      0.214125
steps_per_sec                          0.03756
```

Fifth JSON at step 7250:

```text
loss_action_default_equiv              0.026179
loss_total_minus_action                0.010832
aqr_active_same_role_support_overlap   0.060000
aqr_downstream_same_role_support       0.059797
raw same_role_support_overlap          1.000000
loss_anchor_object_pull                0.214618
loss_anchor_pv                         0.499274
loss_slot_jepa                         0.858312
posterior_identity_switch_rate         0.186667
posterior_recycle_rate                 0.129861
pi_prefix_teacher_delta_rms            0.004222
pi_prefix_teacher_cos_to_teacher       0.999980
lr_group_picf_core                     3.158174e-7
lr_group_policy_head                   6.316349e-5
grad_norm_group_picf_core              0.000008
grad_norm_group_policy_head            0.002601
grad_norm_group_semantic_backbone      0.273833
steps_per_sec                          0.03735
```

Sixth JSON at step 7300:

```text
loss_action_default_equiv              0.026738
loss_total_minus_action                0.009539
aqr_active_same_role_support_overlap   0.015000
aqr_downstream_same_role_support       0.020547
raw same_role_support_overlap          1.000000
loss_anchor_object_pull                0.090702
loss_anchor_pv                         0.496261
loss_slot_jepa                         0.807639
posterior_identity_switch_rate         0.187222
posterior_recycle_rate                 0.130920
pi_prefix_teacher_delta_rms            0.003052
pi_prefix_teacher_cos_to_teacher       0.999990
lr_group_picf_core                     3.153662e-7
lr_group_policy_head                   6.307323e-5
grad_norm_group_picf_core              0.000015
grad_norm_group_policy_head            0.003802
grad_norm_group_semantic_backbone      0.326848
steps_per_sec                          0.03760
```

7200-7300 interpretation:

```text
The fixed-group branch still has normal stochastic action variation, but it has
not reproduced the old late rebound.  Through 7300, action_default_equiv remains
in the 0.018-0.027 band, loss_total_minus_action remains around 0.0095-0.0109,
and active/downstream overlap improves sharply by 7300.  This is the first run
where the intended slow PICF-core group is actually verified under FSDP, so the
old core0.005 negative conclusion is superseded until the 7350/7500 gates are
observed.
```

Seventh to ninth JSON, step7350-7450:

```text
step7350:
  loss_action_default_equiv              0.027469
  loss_total_minus_action                0.011110
  active/downstream overlap              0.060000 / 0.063665
  raw same_role_support_overlap          1.000000
  loss_anchor_object_pull                0.240288
  loss_anchor_pv                         0.517774
  loss_slot_jepa                         0.899505
  posterior_identity_switch_rate         0.173333
  posterior_recycle_rate                 0.126964

step7400:
  loss_action_default_equiv              0.026737
  loss_total_minus_action                0.010328
  active/downstream overlap              0.050000 / 0.061811
  raw same_role_support_overlap          1.000000
  loss_anchor_object_pull                0.155417
  loss_anchor_pv                         0.513583
  loss_slot_jepa                         0.659341
  posterior_identity_switch_rate         0.184444
  posterior_recycle_rate                 0.125952

step7450:
  loss_action_default_equiv              0.027944
  loss_total_minus_action                0.011623
  active/downstream overlap              0.079741 / 0.079446
  raw same_role_support_overlap          1.000000
  loss_anchor_object_pull                0.293271
  loss_anchor_pv                         0.510429
  loss_slot_jepa                         0.621767
  posterior_identity_switch_rate         0.203333
  posterior_recycle_rate                 0.123517
```

7350-7450 interpretation:

```text
The fixed-group branch has passed the first old rebound hard point.  The old EMA
run reached action_default_equiv ~= 0.03591 at step7350, while the corrected
run is still 0.02747 at 7350 and 0.02794 at 7450.  The tail-6 mean is about
0.02554, with max 0.02794.  This is not a confirmed rebound.

The raw reserve overlap remains 1.0, but the action-relevant active/downstream
overlap stays below 0.08 in the latest row and loss_total_minus_action stays
near 0.010-0.012.  Therefore the current evidence supports the FSDP optimizer
group bug as a real cause of the earlier false slow-core result.  Final closure
still requires the old 7500/8000 rebound window.
```

Tenth and eleventh JSON, step7500-7550:

```text
step7500:
  loss_action_default_equiv              0.030833
  loss_action_active7                    0.138862
  loss_total_minus_action                0.011156
  active/downstream overlap              0.084146 / 0.083089
  loss_anchor_object_pull                0.239320
  loss_anchor_pv                         0.499513
  loss_slot_jepa                         0.701623
  posterior_identity_switch_rate         0.198333
  posterior_recycle_rate                 0.127642

step7550:
  loss_action_default_equiv              0.035953
  loss_action_active7                    0.163119
  loss_total_minus_action                0.011119
  active/downstream overlap              0.105000 / 0.109954
  loss_anchor_object_pull                0.238041
  loss_anchor_pv                         0.516689
  loss_slot_jepa                         0.789530
  posterior_identity_switch_rate         0.193333
  posterior_recycle_rate                 0.126020
```

7550 interpretation:

```text
This is the first warning row.  The strict reject rule is not yet met because
only one recent row is >= 0.035, but the local trend from 7450 -> 7500 -> 7550
is action-upward.  The non-action budget is still bounded near 0.011 and
active/downstream overlap is still far below the 0.25 structural reject gate.

Therefore the evidence is not a full structural collapse.  It is an
action-specific rebound warning.  The next decisive row is 7600:
  - if 7600 remains >= 0.035, mark fixed-group rebound reproduced and switch
    to a stronger action/PICF decoupling or phase-boundary intervention;
  - if 7600 falls back below 0.035, treat 7550 as a stochastic spike and keep
    watching 7650/8000.
```

Twelfth JSON, step7600:

```text
step7600:
  loss_action_default_equiv              0.035442
  loss_action_active7                    0.160621
  loss_total_minus_action                0.011023
  active/downstream overlap              0.060000 / 0.066035
  loss_anchor_object_pull                0.227362
  loss_anchor_pv                         0.518102
  loss_slot_jepa                         0.871144
  posterior_identity_switch_rate         0.168333
  posterior_recycle_rate                 0.128519
  lr_group_picf_core                     3.126069e-7
  lr_group_semantic_backbone             2.188248e-5
  lr_group_policy_head                   6.252138e-5
```

7600 interpretation:

```text
The strict two-row rejection criterion is now met:
  step7550 action_default_equiv = 0.035953
  step7600 action_default_equiv = 0.035442

This reproduces the late action-specific rebound even after the FSDP optimizer
group fix.  The fix remains required, because it made the 7050-7450 gate
healthier and verified the intended slow PICF-core LR, but it is not the full
root-cause repair.

The non-action budget remains about 0.011 and active/downstream overlap remains
healthy.  Therefore this is not a broad slot/structure collapse.  Combined with
the policy_semantic_only run, which reaches 0.03628 at step7550 with frozen
PICF and loss_total_minus_action=0, the causal focus moves to the semantic /
action-side low-basin stability and possible data-window alignment around the
7500-7600 rows.

Current root-cause ledger:
  raw reserve overlap: not direct cause;
  weak object/sidecar losses: not direct cause;
  FSDP core-LR grouping bug: fixed but not sufficient;
  PICF core update: not necessary, because policy_semantic_only also rebounds;
  remaining decisive split: extend action_head_only beyond 7550 and add
  explicit data-window trace.
```
