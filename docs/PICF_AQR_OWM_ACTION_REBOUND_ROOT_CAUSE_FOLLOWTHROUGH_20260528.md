# PICF-AQR-OWM Action Rebound Root-Cause Follow-Through - 2026-05-28

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_ACTION_REBOUND_CAUSAL_PLAN_20260528.md
docs/PICF_AQR_OWM_ACTION_INTERFACE_EMA_FINAL_20260527.md
docs/PICF_AQR_OWM_ACTION_REBOUND_PHASE_STABILIZATION_20260526.md
```

## 1. Scope

This note exists to prevent repeated experiments on already-weakened
hypotheses.  The target failure is the late action-specific rebound observed
after the model has already reached a low `loss_action_default_equiv` band.

The current strongest evidence is:

```text
EMA action-interface run:
  step7000 action_default_equiv = 0.021268
  step8300 action_default_equiv = 0.047005

Meanwhile:
  loss_total_minus_action stays about 0.010-0.012
  active/downstream same-role overlap stays below about 0.11
  pi_prefix_teacher_delta_rms stays about 0.002
  pi_prefix_teacher_cos_to_teacher stays about 0.99999
```

So the measured failure is not a broad object-slot collapse.  It is an
action-specific escape from a low-loss basin.

## 2. Already Excluded Or Weakened Hypotheses

### 2.1 Freeze-PICF Has Already Been Tested

The 6500->6800 phase-1 stationarization window used:

```text
PICF_TRAINABLE_SCOPE=policy_only
objective=action only
loss_total_minus_action=0
```

Observed:

```text
6550 action_default_equiv = 0.02877
6600 action_default_equiv = 0.02979
6650 action_default_equiv = 0.02851

active/downstream overlap stayed about 0.10-0.13.
```

Interpretation:

```text
Fixed PICF prefix is trainable by the PI0.5/PaliGemma action path.
```

Therefore another generic freeze-PICF run is not a new root-cause experiment.
The 2026-05-28 policy-only repeat from EMA step7000 was stopped before first
metrics because its expected information gain was low.  It only confirmed the
startup contract:

```text
trainable_scope=policy_only
semantic=paligemma(trainable=True scope=all)
structural losses=0
```

### 2.2 Raw Same-Role Overlap Is Not The Current Action Rebound Driver

In the rebound windows, raw overlap can remain saturated:

```text
aqr_same_role_support_overlap_max = 1.0
```

but action-visible metrics remain healthy:

```text
active_same_role_support_overlap_max      about 0.04-0.10
downstream_same_role_support_overlap_max  about 0.05-0.11
```

For the EMA 7000->8400 window, action loss correlates negatively with active
overlap and non-action budget:

```text
corr(action, active_overlap)       ~= -0.20
corr(action, downstream_overlap)   ~= -0.20
corr(action, total_minus_action)   ~= -0.35
```

This rejects another global raw-overlap penalty as the primary repair.

### 2.3 Weak Object Scaffold Is Not Dominating The Rebound

The rebound happens while:

```text
loss_total_minus_action ~= 0.010-0.012
```

and does not expand with the action increase.  Therefore weak sidecar/object
terms are not large enough to explain action moving from about `0.021` to
`0.047`.

### 2.4 EMA Prefix Drift Is Not Sufficient Explanation

The EMA prefix repair made the action-visible prefix stable by its own metrics:

```text
pi_prefix_teacher_delta_rms       about 0.002
pi_prefix_teacher_cos_to_teacher  about 0.99999
```

Yet action still rebounded.  This falsifies the narrow hypothesis:

```text
late rebound = only online PICF prefix drift.
```

EMA remains useful, but it is not the full cure.

### 2.5 Scalar PICF Core LR Is Not Yet A Valid Exclusion Under FSDP

Earlier notes treated the slower-core variants as evidence that scalar
`PICF_CORE_LR_SCALE` alone could not fix rebound:

```text
core0.01:
  7350 action_default_equiv ~= 0.03589

core0.005:
  7350 action_default_equiv ~= 0.03556
```

This exclusion is now downgraded.  On 2026-05-28, the FSDP training path was
found to build optimizer groups after wrapping.  Root FSDP parameter names use:

```text
_fsdp_wrapped_module.core.*
```

while the old split checked only:

```text
name.startswith("core.")
```

Therefore wrapped core parameters could fall through into `policy_head`, and
the intended `picf_core` LR scale did not necessarily apply.  Any FSDP rebound
run that lacks both:

```text
lr_group_picf_core
grad_norm_group_picf_core
```

is invalid as evidence against two-timescale cotrain.

Patch:

```text
scripts/picf_core_train.py:
  _canonical_param_owner_name strips leading module./_fsdp_wrapped_module.
  before assigning picf_core vs policy_head optimizer groups.
```

Regression:

```text
python -m pytest scripts/picf_core_train_test.py -k "optimizer or picf_core_group" -q
  16 passed
```

## 3. Code Dataflow Follow-Through

### 3.1 Trainable Scope

`scripts/picf_core_train.py::_apply_picf_trainable_scope` implements:

```text
scope=policy_only:
  freeze all model parameters whose name starts with core.
  keep non-core trainable parameters unchanged.

scope=anchor_only:
  freeze everything, then enable only anchor/router/posterior diagnostic params.

scope=all:
  keep normal trainable flags.
```

Therefore:

```text
policy_only + semantic_trainable_scope=all
```

does **not** freeze PaliGemma/Gemma.  It tests action/semantic adaptation on a
stationary PICF prefix.

The new:

```text
semantic_trainable_scope=action_head_only
```

freezes the semantic backbone and trains only:

```text
action_in_proj
action_out_proj
time_mlp_in
time_mlp_out
```

This is the correct next split.

### 3.2 Action Loss Path

Training path:

```text
_PicfWindowTrainer.forward
  policy.forward_train_transition
    core.observe_step
    _training_action_prefix_tokens
    semantic_encoder.compute_action_flow_loss
    core.finalize_with_action
  compute_transition_loss(action_loss_override=flow_override["total"])
```

Important consequence:

```text
loss_action_default_equiv is the pure PI0.5 action-flow loss.
loss_action_prefix_trust and structural/object losses enter total_minus_action.
```

So the observed action rebound is not a logging artifact caused by auxiliary
losses being mixed into the comparable action scalar.

### 3.3 Optimizer Groups

The optimizer groups are:

```text
semantic_backbone:
  semantic_encoder trainable parameters.

picf_core:
  trainable parameters whose name starts with core.

policy_head:
  remaining trainable non-core parameters.
```

The trainer now logs:

```text
grad_norm_group_semantic_backbone
grad_norm_group_picf_core
grad_norm_group_policy_head
lr_group_semantic_backbone
lr_group_picf_core
lr_group_policy_head
```

These metrics are necessary because the remaining ambiguity is no longer
"object collapse vs not".  It is which trainable group pushes the low-loss
action basin out of stability.

## 4. Mathematical Root Cause Class

Let the action model consume a PICF-derived action prefix:

```math
\hat a = \pi_\theta(x, z_\phi)
```

where:

```text
theta:
  PaliGemma/Gemma/action-side parameters.

phi:
  PICF belief-router/core parameters.

z_phi:
  action-visible PICF prefix, possibly EMA/gated/normalized.
```

Near a low-loss basin, a stochastic update changes action loss as:

```math
\Delta L_a
\approx
-\eta_\theta \|\nabla_\theta L_a\|^2
+ \frac{1}{2}\eta_\theta^2 \operatorname{Tr}(H_\theta\Sigma_\theta)
+ \left\langle \nabla_z L_a, \Delta z \right\rangle
+ \Delta_{\text{semantic-representation}} .
```

Evidence has weakened the `Delta z`-only hypothesis because EMA made
`Delta z` small while action still rebounded.  The remaining plausible terms
are therefore:

```text
1. optimizer/noise curvature near the low-loss basin;
2. full semantic-backbone representation drift;
3. action-head capacity or adapter mismatch under fixed prefix;
4. data-order / replay distribution causing optimizer escape after a sharp
   local optimum.
```

This is a narrower and more useful root-cause class than "loss conflict".

## 5. Current Causal Matrix

| Hypothesis | Status | Evidence |
| --- | --- | --- |
| Raw overlap collapse drives action rebound | Weakened | active/downstream overlap remains healthy while action worsens |
| Weak sidecar/object scaffold dominates | Weakened | `loss_total_minus_action` stays bounded around 0.010-0.012 |
| Prefix EMA drift is sole cause | Falsified | EMA prefix stable but action rebounds |
| Core LR scale alone fixes rebound | Reopened under FSDP | prior slower-core runs may have grouped wrapped `core.*` into `policy_head`; require live `lr_group_picf_core` |
| Freeze PICF is untested | False | 6500->6800 policy-only phase was healthy |
| Semantic backbone full-update overshoot | Plausible | not yet isolated against action-head-only |
| Action head/adapters insufficient under fixed prefix | Plausible | requires action-head-only result |
| Full cotrain transition creates low-basin optimizer escape | Plausible | all full-cotrain-style branches eventually rebound |

## 5.1 Literature Alignment For The Remaining Root Cause

The current diagnosis is consistent with recent object-centric/VLA guidance:

```text
Object-centric latent-action work:
  object binding helps remove distractors before policy learning, but the
  policy/action model still has its own optimizer stability problem once the
  object representation is already supplied.

MetaSlot-style variable-slot work:
  duplicate/no-object handling and adaptive count are the right structural
  repair for fixed-slot competition.  PICF already implements the production
  analogue through active/context/reserve gating, duplicate demotion, and
  background residuals; this is why raw reserve overlap is no longer the
  action failure variable.

VLA-JEPA/JEPA-VLA-style latent conditioning:
  predictive latent/context features should condition the action model without
  leaking future targets or forcing every dense token to become an object slot.
  This supports the current split between object files and dense/context
  evidence, but it does not imply adding new object losses to fix an action-only
  rebound.

Flamingo-style gated cross-attention:
  if future evidence shows action needs additional dense context, the correct
  extension is a zero/near-zero initialized gated context path, not another
  raw-overlap penalty.
```

Therefore the current branch sequence is not a hyperparameter search.  It is a
causal intervention on the trainable variable in the local update equation:

```math
\Delta L_a
\approx
-\eta\|\nabla L_a\|^2
+ \frac{1}{2}\eta^2\operatorname{Tr}(H\Sigma)
+ \langle\nabla_z L_a, \Delta z\rangle
+ \Delta_{\text{semantic representation}}.
```

The action-head-only branch sets both `Delta z` and semantic representation
drift close to zero.  The full-semantic branch keeps `Delta z` small by freezing
PICF, but re-enables semantic representation drift.  This is the necessary
next split.

Additional correlation check on the old full-cotrain 7000-8400 window:

```text
corr(action, active_same_role_overlap)      = -0.198
corr(action, downstream_same_role_overlap)  = -0.201
corr(action, total_minus_action)            = -0.350
corr(action, posterior_identity_switch)     = -0.463
corr(action, pi_prefix_teacher_delta_rms)   = -0.679
corr(action, grad_norm)                     =  0.327
```

This supports the same conclusion: the old action rebound is not explained by
the active object-structure metrics worsening.  The only positive correlation
among these coarse logs is gradient norm, which is consistent with low-loss
basin optimizer/representation instability rather than object collapse.

## 6. Active Diagnostic

The redundant policy-only full-semantic repeat was stopped before first metrics.

The non-redundant branch is:

```text
run:
  picf_a7_ema7000_policyonly_actionhead_300_20260528

scope:
  PICF frozen
  PaliGemma/Gemma backbone frozen
  action projections/time MLP trainable
  structural losses disabled

purpose:
  separate semantic-backbone overshoot from action-head capacity/optimizer
  limits under a fixed PICF prefix.
```

Runtime strategy:

```text
TRAINING_STRATEGY=ddp
```

This is deliberate for this causal split.  The branch trains only the about
2.1M PI0 action projection/time-MLP parameters.  The frozen PaliGemma/Gemma
root still contains mixed bf16/fp32 tensors, and FSDP root flattening can fail
before training with:

```text
ValueError: Must flatten tensors with uniform dtype but got torch.float32 and torch.bfloat16
```

DDP avoids that distributed-wrapper artifact while preserving the mathematical
quantity under test:

```text
Can a fixed PICF prefix plus frozen semantic backbone keep optimizing the
action head after the step7000 low-loss basin?
```

Decision rule:

```text
If action-head-only stays <=0.030:
  semantic backbone full-update is the likely overshoot source.
  next production repair: semantic trust-region / lower semantic LR / LoRA or
  head-only action polish before full semantic release.

If action-head-only rebounds:
  fixed prefix/context or action-head capacity is insufficient.
  next repair must change action-conditioning architecture or adapter capacity,
  not another object-slot loss.
```

First readout:

```text
7050:
  loss_action_default_equiv          0.02294
  loss_action_active7                0.10230
  loss_total_minus_action            0.0
  grad_norm_group_semantic_backbone  0.16666
  lr_group_semantic_backbone         6.35e-5
  active_same_role_overlap_max       0.0700
  downstream_same_role_overlap_max   0.0752
  posterior_recycle_rate             0.1293
  posterior_identity_switch_rate     0.1817

7100:
  loss_action_default_equiv          0.02577
  loss_action_active7                0.11563
  loss_total_minus_action            0.0
  grad_norm_group_semantic_backbone  0.08395
  lr_group_semantic_backbone         6.34e-5
  active_same_role_overlap_max       0.0550
  downstream_same_role_overlap_max   0.0679
  posterior_recycle_rate             0.1313
  posterior_identity_switch_rate     0.1806

7150:
  loss_action_default_equiv          0.02185
  loss_action_active7                0.09851
  loss_total_minus_action            0.0
  grad_norm_group_semantic_backbone  0.10482
  lr_group_semantic_backbone         6.33e-5
  active_same_role_overlap_max       0.0850
  downstream_same_role_overlap_max   0.0799
  posterior_recycle_rate             0.1305
  posterior_identity_switch_rate     0.1822

7200:
  loss_action_default_equiv          0.01794
  loss_action_active7                0.08072
  loss_total_minus_action            0.0
  grad_norm_group_semantic_backbone  0.11439
  lr_group_semantic_backbone         6.33e-5
  active_same_role_overlap_max       0.0856
  downstream_same_role_overlap_max   0.0801
  posterior_recycle_rate             0.1294
  posterior_identity_switch_rate     0.1944

7250:
  loss_action_default_equiv          0.02695
  loss_action_active7                0.12035
  loss_total_minus_action            0.0
  grad_norm_group_semantic_backbone  0.10900
  lr_group_semantic_backbone         6.32e-5
  active_same_role_overlap_max       0.1050
  downstream_same_role_overlap_max   0.1136
  posterior_identity_switch_rate     0.1928

7300:
  loss_action_default_equiv          0.02713
  loss_action_active7                0.12202
  loss_total_minus_action            0.0
  grad_norm_group_semantic_backbone  0.12320
  lr_group_semantic_backbone         6.31e-5
  active_same_role_overlap_max       0.0600
  downstream_same_role_overlap_max   0.0768
  posterior_identity_switch_rate     0.1833

7350:
  loss_action_default_equiv          0.02824
  loss_action_active7                0.12617
  loss_total_minus_action            0.0
```

Interpretation:

```text
The fixed-PICF + frozen-semantic-backbone + action-head-only path is alive and
is improving through 7200.  This weakens the hypothesis that the action head
alone is incapable under the preserved step7000 prefix, and it also weakens
the hypothesis that raw overlap or fixed PICF prefix content is sufficient to
force the rebound.  The 7300/7350 gate now passes relative to the old EMA
full-cotrain failure, so this branch has answered its intended question.
```

## 7. What Not To Do Next

Do not start another 30K run until the action-head-only split is read.

Do not add:

```text
more raw-overlap penalties;
new SAM/object proposal losses;
slot-JEPA/support-pred pressure;
larger unroll as the first response;
another generic policy_only confirmation run.
```

Those are not aligned with the measured failure.

## 8. Current Root-Cause Narrowing After 7350

The controlled comparison now separates three classes:

```text
EMA full cotrain, same checkpoint:
  7050 -> 7200 action_default_equiv:
    0.02610 -> 0.02684
  then later:
    7300 0.03077
    7350 0.03591
    8300 0.04701

EMA policy_only + action_head_only:
  7050 -> 7350 action_default_equiv:
    0.02294 -> 0.01794 -> 0.02824
  loss_total_minus_action:
    0.0 throughout
```

Interpretation:

```text
raw overlap:
  still 1.0, yet action improves in action-head-only.
  therefore raw reserve/context overlap is not sufficient to cause rebound.

fixed PICF prefix/content:
  same preserved step7000 prefix family is usable by a trainable action head.
  therefore the prefix is not obviously information-poisoned.

action-head capacity:
  materially weakened as the main cause: the 2.1M action projection/time-MLP
  slice remains in the healthy band through the old 7350 failure point under
  fixed PICF and frozen semantic backbone.

remaining plausible root:
  the instability is introduced when a larger trainable representation is
  allowed to move around the low-loss action basin:
    either full PaliGemma/Gemma semantic update,
    or full PICF cotrain,
    or their optimizer interaction.
```

The important scientific point is that this is no longer an object-slot
binding diagnosis.  The measured failure is a low-loss action optimizer
stability problem under moving conditioning/representation.

## 8.1 Operational Branch Handoff

After the 7350 gate, the action-head-only session was stopped and the next
non-redundant branch was started:

```text
stopped:
  picf_a7_ema7000_policyonly_actionhead_ddp_20260528

started:
  picf_a7_ema7000_policyonly_actionsemantic_20260528

contract:
  PICF_TRAINABLE_SCOPE=policy_only
  SEMANTIC_TRAINABLE_SCOPE=all
  structural losses disabled
  resume checkpoint = EMA full-cotrain step7000
```

Interpretation of this branch:

```text
If full semantic/action trainable rebounds while action_head_only passed:
  root cause = PaliGemma/Gemma semantic representation update overshoots the
  already-low action basin under the fixed PICF prefix.

If full semantic/action trainable also stays healthy:
  root cause moves back to full PICF/core cotrain drift or optimizer
  interaction between PICF core and semantic stack.
```

First readout:

```text
7050:
  loss_action_default_equiv          0.02248
  loss_action_active7                0.10063
  loss_total_minus_action            0.0
  grad_norm_group_semantic_backbone  0.2962
  active_same_role_overlap_max       0.0700
  downstream_same_role_overlap_max   0.0802
  posterior_identity_switch_rate     0.1839
  pi_prefix_teacher_delta_rms        0.00784

7100:
  loss_action_default_equiv          0.02559
  loss_action_active7                0.11469
  loss_total_minus_action            0.0
  grad_norm_group_semantic_backbone  0.1368
  active_same_role_overlap_max       0.0600
  downstream_same_role_overlap_max   0.0679
  posterior_identity_switch_rate     0.1828
  pi_prefix_teacher_delta_rms        0.00456

7150:
  loss_action_default_equiv          0.02273
  loss_action_active7                0.10268
  loss_total_minus_action            0.0
  grad_norm_group_semantic_backbone  0.2797
  active_same_role_overlap_max       0.0850
  downstream_same_role_overlap_max   0.0853
  posterior_identity_switch_rate     0.1811
  pi_prefix_teacher_delta_rms        0.00391

comparison:
  old full cotrain 7050              0.02610
  action-head-only 7050              0.02294
  old full cotrain 7150              0.02768
  action-head-only 7150              0.02185
```

Interpretation:

```text
Full semantic/action update does not fail immediately.  The branch remains
healthy through 7150 and is better than the old full-cotrain line at the same
steps, while staying close to the action-head-only line.  This is not yet
decisive: the old failure only became clear at 7300/7350, so the branch should
continue to that window unless action rises above the old failure band earlier.
```

Second readout after crossing the old failure window:

```text
step    old_full    action_head_only    full_semantic_fixed_picf
7050    0.02610     0.02294             0.02248
7100    0.02663     0.02577             0.02559
7150    0.02768     0.02185             0.02273
7200    0.02684     0.01794             0.01845
7250    0.02785     0.02695             0.02636
7300    0.03077     0.02713             0.02703
7350    0.03591     0.02824             0.02749
7400    0.03533     n/a                 0.02691
7450    0.03795     n/a                 0.02796
7500    0.03857     n/a                 0.03076
7550    n/a         n/a                 0.03628
```

At `7550`, full-semantic/fixed-PICF has started to enter the old failure band,
but later and more weakly than the old full-cotrain line.  Since this branch has:

```text
PICF_TRAINABLE_SCOPE=policy_only
SEMANTIC_TRAINABLE_SCOPE=all
loss_total_minus_action=0.0
active/downstream overlap still low
raw overlap still saturated but action-visible overlap healthy
```

the causal evidence now points away from:

```text
raw reserve overlap
object/sidecar auxiliary losses
PICF core update
fixed prefix being intrinsically unusable
```

and toward:

```text
full PaliGemma/Gemma semantic representation update can destabilize the
already-low action basin even when the PICF prefix is fixed.
```

Important nuance:

```text
action-head-only passed 7350.
full-semantic/fixed-PICF passed 7350 but began rising by 7500/7550.
old full-cotrain failed earlier and harder.
```

Therefore the hierarchy is:

```text
semantic backbone drift alone is sufficient to create the rebound later;
full PICF+semantic cotrain makes it appear earlier/stronger;
action head alone is not sufficient to create it in this window.
```

The next repair should be semantic-backbone stability, not another object-slot
loss:

```text
1. keep PICF prefix fixed or very slow around the low action basin;
2. use action-head/adapters first;
3. release full semantic backbone only through a trust-region schedule:
   lower semantic LR, smaller update budget, or adapter/LoRA-style subspace;
4. only after that reintroduce full PICF cotrain with a separate slower clock.
```

## 9. Required Next Readout

At the first logged metric rows, inspect:

```text
loss_action_default_equiv
loss_action_active7
grad_norm_group_semantic_backbone
grad_norm_group_policy_head
lr_group_semantic_backbone
lr_group_policy_head
loss_total_minus_action
active/downstream overlap
posterior_recycle_rate
posterior_identity_switch_rate
```

The group-grad metrics are decisive.  Without them, the branch only repeats
loss curves and cannot isolate the optimizer group responsible for the rebound.

The active non-redundant experiment is now:

```text
PICF frozen + full semantic/action trainable from the same step7000 checkpoint.
```

Reason:

```text
action_head_only already passed:
  fixed PICF + frozen semantic backbone + action head stayed stable through
  the old 7350 failure point.

full semantic passing too:
  points back to PICF/core cotrain drift as the rebound source.

full semantic rebounding while action_head_only passes:
  localizes the source to full PaliGemma/Gemma semantic update around the
  already-low action basin.
```
