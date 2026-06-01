# PICF-AQR-OWM Action Rebound Causal Plan - 2026-05-28

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_ACTION_INTERFACE_EMA_FINAL_20260527.md
docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_FOLLOWTHROUGH_20260528.md
```

## 1. Current Locked Evidence

The A7 EMA action-prefix run was stopped after the 8300 gate to preserve the
7000 checkpoint:

```text
run:
  picf_a7_actionprefix_ema_from6800_action2_30k_20260527

preserved:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000
    picf_a7_actionprefix_ema_from6800_action2_30k_20260527/8000
```

Key trajectory:

```text
step 7000:
  loss_action_default_equiv          0.021268
  loss_total_minus_action            0.010211
  pi_prefix_teacher_delta_rms        0.018038
  active_same_role_overlap_max       0.069984

step 7500:
  loss_action_default_equiv          0.038571
  loss_total_minus_action            0.010367
  pi_prefix_teacher_delta_rms        0.002524
  active_same_role_overlap_max       0.031972

step 8000:
  loss_action_default_equiv          0.043714
  loss_total_minus_action            0.010795
  pi_prefix_teacher_delta_rms        0.002116
  active_same_role_overlap_max       0.063626

step 8300:
  loss_action_default_equiv          0.047005
  loss_total_minus_action            0.010597
  pi_prefix_teacher_delta_rms        0.002260
  pi_prefix_teacher_cos_to_teacher   0.999994
  active_same_role_overlap_max       0.098444
  downstream_same_role_overlap_max   0.107127
```

### 1.2 2026-05-28 Strict Rebound Gate Relaunch

The speed-regression audit found that the slow 39-46 sec/step runs were caused
by accidentally training wrapper-local restored PI0 flow/time heads through
`semantic_trainable_scope=all`.  The production semantic cotrain boundary is
now:

```text
semantic_trainable_scope=backbone_only
semantic_gradient_checkpointing=enabled
```

This keeps the PaliGemma/Gemma semantic backbone trainable while freezing the
wrapper-local action/time heads, matching the historical fast boundary.  A
short smoke from the same 7000 checkpoint returned to about 25 sec/step.

The first 2026-05-28 speedfix smoke used `NUM_TRAIN_STEPS=7300`.  That run is
valid for speed and runtime sanity, but not for rebound causality because the
cosine/decay schedule depends on the total horizon.  The strict rebound gate is
therefore relaunched with the same 7000 checkpoint and a 30000-step horizon:

```text
run:
  picf_a7_reboundgate_30khorizon_from7000_scopefix_20260528

resume:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000

fixed controls:
  NUM_TRAIN_STEPS=30000
  unroll_steps=2
  burnin_steps=1
  sidecar_root=/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
  semantic_trainable_scope=backbone_only
  semantic_gradient_checkpointing=enabled
  picf_core_lr_scale=0.002
  semantic_lr_scale=0.1
  policy_head_lr_scale=1.0
  action_prefix_teacher=off
  lambda_action_prefix_trust=0
```

Acceptance is not `loss_total` alone.  The rebound question is answered by:

```text
primary:
  loss_action_default_equiv must not show sustained >30% rebound from the
  local 7000-step basin.

supporting:
  loss_total_minus_action must stay bounded rather than expanding into action.
  active/downstream same_role_support_overlap should stay below about 0.15.
  posterior_recycle_rate and posterior_identity_switch_rate must not rise in
  lockstep with action loss.
  step time should remain in the historical 23-27 sec/step band.
```

### 1.1 Prior Freeze-PICF Evidence Already Matters

This causal split must not be treated as if freeze-PICF has never been tested.
The 2026-05-26 phase-stabilization record already contains a clean
`policy_only` / frozen-PICF stationarization window:

```text
phase-1 run:
  PICF_TRAINABLE_SCOPE=policy_only
  objective=action only
  structural/non-action losses=0

6550:
  loss_action_default_equiv        0.02877
  loss_total_minus_action          0.0
  active_support_overlap_max       0.130
  downstream_support_overlap_max   0.139

6600:
  loss_action_default_equiv        0.02979
  loss_total_minus_action          0.0
  active_support_overlap_max       0.109
  downstream_support_overlap_max   0.107

6650:
  loss_action_default_equiv        0.02851
  loss_total_minus_action          0.0
  active_support_overlap_max       0.130
  downstream_support_overlap_max   0.129
```

Therefore "freeze PICF fixes everything" is not an open new hypothesis.  It has
already shown that a stationary PICF prefix can be action-trainable in short
windows.  The current step7000 policy-only repeat has a narrower purpose:

```text
1. verify whether the preserved EMA step7000 checkpoint behaves the same way;
2. log per-optimizer-group gradients under the new explicit semantic scope;
3. distinguish full semantic-backbone overshoot from action-head-only limits.
```

If the repeat merely stays healthy, it confirms prior evidence rather than
discovering a new root cause.  The remaining root cause must then be searched in
the transition from frozen/stationary prefix to full cotrain, especially the
action/semantic optimizer timescale and low-loss basin stability.

## 2. What This Falsifies

### 2.1 Not raw overlap

Raw same-role overlap stays saturated, but the action-visible metrics remain
healthy:

```text
active_same_role_overlap_max      < 0.10 near step8300
downstream_same_role_overlap_max  < 0.11 near step8300
```

Therefore another global raw-overlap penalty is not the causal repair.  Raw
overlap is reserve/context telemetry under the current active-slot gate.

### 2.2 Not weak object scaffold dominance

`loss_total_minus_action` stays near `0.010-0.012` while action rebounds.  The
object scaffold floor is already weak, and non-action terms do not expand in
the rebound window.

### 2.3 Not prefix EMA failure

The teacher prefix is stable:

```text
delta_rms ~= 0.002
cos_to_teacher ~= 0.99999
trust_loss ~= 2e-7
```

This falsifies the narrow hypothesis that the late rebound was only an
action-visible prefix drift problem.  EMA is still correct as a stabilizer, but
it is not sufficient.

### 2.4 Not a metric-only bug

The canonical action flow scalar and component diagnostics move together:

```text
step 7000:
  action_default_equiv 0.0213
  action_active7       0.0959

step 8300:
  action_default_equiv 0.0470
  action_active7       0.2138
```

So the rebound is not just a logging artifact in `loss_action_default_equiv`.

## 3. Remaining Mathematical Root Cause Class

The action path sees a stabilized prefix:

```math
z^{act}_k = \bar z_k
```

and optimizes:

```math
L_a(\theta)=\ell(\pi_\theta(x,z^{act}),a).
```

At the 7000 checkpoint, `L_a` is already in a low-loss basin.  A one-step
stochastic update has expected local change:

```math
\Delta L_a
\approx
-\eta\|\nabla L_a\|^2
+ \frac{1}{2}\eta^2 \operatorname{Tr}(H\Sigma)
+ \Delta_{\text{representation}}
```

where:

```text
first term:
  useful action descent.

second term:
  optimizer/noise curvature penalty. It grows important when gradient norm is
  small near a low-loss basin.

Delta_representation:
  changes to the trainable PaliGemma/Gemma/action representation even if PICF
  prefix is stable.
```

The latest evidence indicates the remaining problem is in this class:

```text
action-side optimizer / semantic-policy co-train instability after reaching a
low-loss basin.
```

This is now more likely than another object-binding repair because object and
prefix telemetry are stable while action alone worsens.

## 4. Code Dataflow Follow-Through

Training dataflow:

```text
scripts/picf_core_train.py
  _PicfWindowTrainer.forward
    policy.forward_train_transition
      core.observe_step
      policy._training_action_prefix_tokens
      semantic_encoder.compute_action_flow_loss
      core.finalize_with_action
    compute_transition_loss
      action_loss_override = flow_override["total"]
      action_default_equiv = action_loss_override
      total = action + capped physical + capped semantic + capped alignment
```

Action prefix:

```text
src/openpi/picf/policy.py
  _training_action_prefix_tokens
    online PICF prefix -> EMA teacher buffer
    action consumes teacher when blend=1.0
    trust loss is non-action alignment only
```

Optimizer groups:

```text
scripts/picf_core_train.py
  semantic_backbone     PaliGemma/PI0 action model
  picf_core             core.* belief-router parameters
  policy_head           remaining non-core, non-semantic adapters
```

New instrumentation:

```text
grad_norm_group_semantic_backbone
grad_absmax_group_semantic_backbone
lr_group_semantic_backbone

grad_norm_group_picf_core
grad_absmax_group_picf_core
lr_group_picf_core

grad_norm_group_policy_head
grad_absmax_group_policy_head
lr_group_policy_head
```

New explicit semantic scope:

```text
--semantic-trainable-scope all
  normal cotrain.

--semantic-trainable-scope action_head_only
  freeze PaliGemma/Gemma backbone/expert and train only action_in_proj,
  action_out_proj, time_mlp_in, and time_mlp_out.
```

This replaces the previous implicit behavior where action heads could remain
trainable in frozen semantic probes without an auditable contract.

## 5. Causal Experiment Matrix

All experiments resume from the preserved step7000 checkpoint and keep
`num_train_steps=30000` so the scheduler horizon remains comparable.  Stop
early at 300-500 steps if the hypothesis is decided.

### A. PICF frozen, semantic/action trainable

Script:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_ema7000_policyonly_actionsemantic_300_20260528.sh
```

Contract:

```text
picf_trainable_scope=policy_only
semantic_trainable_scope=all
structural losses=0
```

Interpretation:

```text
action falls/stays <= 0.03:
  confirms the prior phase-1 evidence that a stationary PICF prefix is
  trainable by the action/semantic path.  This does not by itself solve the
  production cotrain problem; it narrows the fault to the cotrain transition.

action rebounds:
  root is action/semantic optimizer or prefix/context content.
```

### B. PICF frozen, action head only

Script:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_ema7000_policyonly_actionhead_300_20260528.sh
```

Contract:

```text
picf_trainable_scope=policy_only
semantic_trainable_scope=action_head_only
structural losses=0
```

Interpretation:

```text
action-head-only improves while full semantic does not:
  PaliGemma/Gemma semantic backbone update is the overshoot source.

action-head-only also rebounds:
  fixed prefix/context or action-head capacity is insufficient.
```

### C. Full cotrain, reduced semantic and PICF timescales

Script:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_ema7000_fullcotrain_lowersemantic_300_20260528.sh
```

Contract:

```text
semantic_lr_scale=0.10
picf_core_lr_scale=0.002
EMA prefix kept
object scaffold floor=0.03
```

Interpretation:

```text
action stays low:
  root is optimizer timescale/noise near a sharp low-loss basin.

action rebounds:
  current prefix/context is harmful or the action route needs a different
  conditioning architecture.
```

2026-05-28 correction:

```text
Reduced-PICF-timescale FSDP runs are valid only if their JSON contains:
  lr_group_picf_core
  grad_norm_group_picf_core
```

The old optimizer grouping happened after FSDP wrapping and checked only
`name.startswith("core.")`.  FSDP root parameters are named
`_fsdp_wrapped_module.core.*`, so wrapped core parameters could be assigned to
`policy_head`.  This means prior core0.02/core0.005 failures cannot by
themselves falsify two-timescale cotrain.  Re-run the EMA step7000 gate after
the canonical-name fix before drawing the rebound conclusion.

First fixed-group 30k-horizon relaunch gate:

```text
run:
  picf_a7_reboundgate_ema_recipe_scopefix_lrgroupfix_from7000_30k_20260528

step7050:
  loss_action_default_equiv          0.022599
  loss_total_minus_action            0.010611
  active_same_role_overlap_max       0.064367
  downstream_same_role_overlap_max   0.076416
  lr_group_picf_core                 3.175974e-7
  lr_group_policy_head               6.351949e-5
```

This validates the hard gate and improves over the prior EMA step7050 action
loss (`~0.02610`).  It does not close the rebound question until the fixed run
passes the old 7350/7500 rebound window.

Step7100 remains better than the old run:

```text
old EMA 7100:
  loss_action_default_equiv          0.026631
  active/downstream overlap          0.060113 / 0.084385

fixed-group 7100:
  loss_action_default_equiv          0.025269
  loss_total_minus_action            0.010499
  active/downstream overlap          0.054673 / 0.066878
  lr_group_picf_core                 3.171562e-7
```

This still does not prove the repair through the late window, but it removes
the immediate 7100 regression case.

Step7200-7300 fixed-group update:

```text
step7200:
  loss_action_default_equiv          0.018195
  loss_total_minus_action            0.009508
  active/downstream overlap          0.064859 / 0.062881

step7250:
  loss_action_default_equiv          0.026179
  loss_total_minus_action            0.010832
  active/downstream overlap          0.060000 / 0.059797

step7300:
  loss_action_default_equiv          0.026738
  loss_total_minus_action            0.009539
  active/downstream overlap          0.015000 / 0.020547
```

The fixed-group run still has stochastic action fluctuation, but it has not
entered the old rebound band before 7350.  Do not switch at 7300.  The next hard
decision remains:

```text
continue if 7350/7500 stay below 0.035 action_default_equiv;
reject/switch only if two consecutive logs return to >=0.035 while
loss_total_minus_action and active/downstream overlap stay healthy.
```

Step7350-7450 fixed-group update:

```text
step7350:
  loss_action_default_equiv          0.027469
  loss_total_minus_action            0.011110
  active/downstream overlap          0.060000 / 0.063665

step7400:
  loss_action_default_equiv          0.026737
  loss_total_minus_action            0.010328
  active/downstream overlap          0.050000 / 0.061811

step7450:
  loss_action_default_equiv          0.027944
  loss_total_minus_action            0.011623
  active/downstream overlap          0.079741 / 0.079446
```

The branch has now passed the first old rebound hard point.  Old EMA reached
about `0.03591` at step7350; fixed-group is `0.02747` at step7350 and `0.02794`
at step7450.  This still requires the step7500/8000 gate, but there is no
current rebound signal under the strict two-row `>=0.035` rule.

Step7500-7550 fixed-group update:

```text
step7500:
  loss_action_default_equiv          0.030833
  loss_action_active7                0.138862
  loss_total_minus_action            0.011156
  active/downstream overlap          0.084146 / 0.083089

step7550:
  loss_action_default_equiv          0.035953
  loss_action_active7                0.163119
  loss_total_minus_action            0.011119
  active/downstream overlap          0.105000 / 0.109954
```

This is a yellow warning, not a completed rejection.  The action scalar crossed
the `0.035` line once, while non-action loss and structural overlap remain
bounded.  The causal decision is now strict:

```text
if step7600 action_default_equiv >= 0.035:
  mark action rebound reproduced despite fixed FSDP LR groups;
else:
  keep running and treat step7550 as a spike until 7650/8000 confirms trend.
```

Step7600 fixed-group update:

```text
step7600:
  loss_action_default_equiv          0.035442
  loss_action_active7                0.160621
  loss_total_minus_action            0.011023
  active/downstream overlap          0.060000 / 0.066035
  lr_group_picf_core                 3.126069e-7
  lr_group_semantic_backbone         2.188248e-5
  lr_group_policy_head               6.252138e-5
```

Decision:

```text
Fixed-group rebound reproduced.  The FSDP optimizer grouping fix is necessary
engineering hygiene, but it is not the complete cure.  Since the rebound
appears while loss_total_minus_action and action-visible overlap stay bounded,
this is still an action-specific low-basin failure.
```

Step7650 fixed-group follow-through:

```text
step7650:
  loss_action_default_equiv          0.038672
  loss_action_active7                0.175419
  loss_total_minus_action            0.010442
  active/downstream overlap          0.040000 / 0.047484
  loss_slot_jepa                     1.317115
  grad_norm                          0.268751
```

This confirms the rebound is not just a single step7550/7600 logging
coincidence.  It also makes the current run low-information beyond this point:
the branch lacks `window_trace_rank*.jsonl`, so it cannot distinguish
action-head/data-window instability from semantic-backbone drift.

The most important new causal evidence is the 2026-05-28
`policyonly_actionsemantic` split:

```text
PICF_TRAINABLE_SCOPE=policy_only
loss_total_minus_action=0
semantic/action side trainable

step7550 action_default_equiv = 0.036278
```

Therefore the next root-cause target is not another raw-overlap or object-loss
patch.  The remaining split is:

```text
1. action_head_only beyond 7550/8000;
2. explicit data-window trace around 7500-7600;
3. same-window replay with semantic frozen vs trainable.
```

Immediate execution rule:

```text
Stop the fixed-group rebound run after preserving logs and ckpts.
Launch action_head_only from the same 7000 checkpoint with the same 30000-step
LR horizon and window_trace enabled.
```

Action-head trace run status:

```text
run:
  picf_a7_ema7000_policyonly_actionhead_trace_from7000_30k_20260529

step7050:
  loss_action_default_equiv          0.022951
  loss_action_active7                0.102341
  loss_total_minus_action            0.000000
  grad_norm                          0.167682
  window_trace_rank0/1               written

step7100:
  loss_action_default_equiv          0.025771
  loss_action_active7                0.115666
  loss_total_minus_action            0.000000
  grad_norm                          0.084392
  window_trace_rank0/1               written

step7150:
  loss_action_default_equiv          0.021833
  loss_action_active7                0.098441
  loss_total_minus_action            0.000000

step7200:
  loss_action_default_equiv          0.017932
  loss_action_active7                0.080687
  loss_total_minus_action            0.000000

step7250:
  loss_action_default_equiv          0.026885
  loss_action_active7                0.120204
  loss_total_minus_action            0.000000

step7300:
  loss_action_default_equiv          0.027154
  loss_action_active7                0.122115
  loss_total_minus_action            0.000000

step7350:
  loss_action_default_equiv          0.028195
  loss_action_active7                0.126068
  loss_total_minus_action            0.000000

step7400:
  loss_action_default_equiv          0.029102
  loss_action_active7                0.126918
  loss_total_minus_action            0.000000

step7450:
  loss_action_default_equiv          0.028440
  loss_action_active7                0.126261
  loss_total_minus_action            0.000000

step7500:
  loss_action_default_equiv          0.030503
  loss_action_active7                0.137044
  loss_total_minus_action            0.000000

step7550:
  loss_action_default_equiv          0.035659
  loss_action_active7                0.161547
  loss_total_minus_action            0.000000

step7600:
  loss_action_default_equiv          0.035610
  loss_action_active7                0.161139
  loss_total_minus_action            0.000000
```

This confirms the diagnostic is not a configuration-only launch.  It is writing
the exact sampled window metadata needed for the 7550/7600 causal split.  The
step7100 point still matches the prior action-head-only short diagnostic,
step7200 is strongly healthy, and step7350 stays below 0.030.  This rejects the
old 7350 EMA-style immediate rebound for action-head-only, but it is still too
early to stop or declare success because the decisive reject band begins around
step7550.

Final action-head trace decision:

```text
step7550 and step7600 both exceed 0.035.
The action-head-only branch is rejected by the same hard gate as fixed_group.
```

The important causal update is that action-head-only, fixed_group, and
policy_semantic match almost point-by-point in the 7050-7600 band.  Because
the resumed trainer starts its RNG from `args.seed + 17 * rank` rather than
fast-forwarding by `start_step`, these branches replay the same sampled-window
sequence.  The rebound is therefore more likely a deterministic train-window
difficulty/metric-distribution block than a PICF structural regression.

Next required diagnostic:

```text
near-zero-LR or no-update same-window replay from the same step7000 checkpoint.
```

If that replay produces the same 7500/7600 rise, the correct production fix is
not another slot loss.  The correct fix is to add a fixed held-out action probe
and stop interpreting raw train-window loss as stationary validation loss.

Near-zero-LR replay result:

```text
run:
  picf_a7_ema7000_policyonly_actionhead_nearzerolr_7600_20260529

lr near reject band:
  about 1e-16 to 1e-20

step7500:
  loss_action_default_equiv 0.030804

step7550:
  loss_action_default_equiv 0.036840

step7600:
  loss_action_default_equiv 0.037180
```

This closes the causal split: the model does not need to update for the
7550/7600 spike to appear.  The trainer has been repaired so sampled-window RNG
is keyed by `(seed, rank, global_step, micro_step, retry_count)` by default.
Use `--no-step-indexed-window-rng` only for reproducing legacy May-2026 rebound
diagnostics.

See:

```text
docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md
```

## 6. Decision Rules

2026-05-29 supersession: the numeric reject bands below were written before the
step-indexed sampled-window RNG repair.  They are retained as the historical
logic for the legacy replayed-window runs, but they must **not** be used as
absolute stop gates for the corrected 7000+ continuations.  The current
production gate is in
`docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md`: judge corrected
runs by step-indexed windows, structural health, non-action budget, and a future
fixed held-out action probe, not by matching the old replayed 7050/7550 scalar.

Use `loss_action_default_equiv` as the comparable action scalar within a single
sampling regime.  Also inspect `loss_action_active7` to ensure component-level
movement agrees.

Continue branch:

```text
loss_action_default_equiv <= 0.030 after 300 steps;
loss_action_active7 does not rise monotonically;
group grad norms finite and not dominated by one unexpected group;
prefix delta remains near 0.002-0.005;
active/downstream overlap remain < 0.25.
```

Reject branch:

```text
loss_action_default_equiv returns to >= 0.035 for two consecutive logs;
or action_active7 rises with it;
or semantic_backbone grad dominates while action worsens;
or PICF is frozen but action still rebounds.
```

## 7. Current Best Guess Before Running Causal Splits

The strongest current guess is:

```text
large semantic/action-side update after step7000 is too aggressive for the
already-low action basin.
```

This is not yet the final root cause.  The three scripts above are designed so
that the next 300-500 steps can distinguish:

```text
PICF co-training interference
vs
PaliGemma/Gemma semantic overshoot
vs
fixed prefix/context insufficiency.
```

Do not start another 30K final run until at least one of these causal splits
passes its gate.
