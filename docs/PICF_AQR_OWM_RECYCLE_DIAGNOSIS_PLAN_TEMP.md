# PICF-AQR-OWM Recycle Diagnosis Plan

Source index: [`src/openpi/picf/README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)

Date: 2026-05-11
Status: active diagnosis plan for the current staged MVTrack anchor/action runs.

This document records the current evidence, mathematical interpretation, and
next experiment plan for the `posterior_recycle_rate=1.0` issue observed after
staged action cotraining. It is intentionally separate from the main MVTrack
deployment contract: the deployment contract says what the architecture should
do; this file says how to diagnose whether the posterior identity mechanism is
actually doing it during the current CALVIN training probes.

2026-05-11 update: the next diagnostic instrumentation has been deployed as
debug-only metrics. It does not change posterior math, losses, gradients, cache
read, or action conditioning. It only exports the internal quantities needed to
distinguish recycle-head saturation, dustbin/residual shortcutting,
support-mass mismatch, role-specific recycle, and address-update starvation.

## 1. Short Verdict

The staged training evidence is now strong enough to support this conclusion:

```text
Anchor-only warmup works.
Direct action cotrain collapses support structure.
Staged action cotrain is much better than direct cotrain.
Posterior recycle remains unresolved.
```

The current best candidate is:

```text
R0 anchor warmup:
  picf_mvtrack_anchor_noaction_supportdiv_acc4_lr3e4_probe300_20260510_225402

S2 staged action:
  picf_s2_stage300_a05_cache005_sd005_den0_900_20260511_0155
```

The key positive S2 result is that action can be introduced while retaining
reasonable support structure:

```text
S2 step 900:
  loss_total                         = 0.7239
  loss_action                        = 0.0174
  loss_action_active7                = 0.3147
  default-weight-equivalent action   ~= 0.0696
  aqr_same_role_support_overlap_max  = 0.4746
  last20 overlap mean                = 0.4551
  aqr_effective_anchor_count         ~= 23.4
```

The key unresolved S2 result is:

```text
posterior_recycle_rate = 1.0
```

That is not a harmless label. In the current code this metric is
`posterior.recycle_gate.mean()`. A value near 1 means the posterior update is
using the residual/recycle path for most slots instead of preserving prior
slot content through time.

## 2. Code Definition of Recycle

The live posterior update computes recycle from:

```python
recycle_in = concat(
    h_prior,
    support_mass_raw,
    var_prior.mean,
    residual_summary,
    alpha_prior,
)
recycle = sigmoid(recycle_head(recycle_in))
```

Then it mixes prior state with residual state:

```python
bar_h   = (1 - recycle) * h_prior   + recycle * res_h
bar_c   = (1 - recycle) * c_prior   + recycle * res_c
bar_mu  = (1 - recycle) * mu_prior  + recycle * res_mu
bar_var = (1 - recycle) * var_prior + recycle * res_var
```

The same gate also controls the evidence binding distribution through the
dustbin transfer:

```python
recycle_share   = recycle / (1 + recycle.sum())
binding_support = support_raw + recycle_share[:, None] * dustbin_raw[None, :]
```

Address update is intentionally conservative:

```python
rate = address_update_rate * support_mass * (1 - recycle) * exp(-kappa * innovation_risk)
slot_address = normalize((1 - rate) * base_address + rate * obs_address)
```

Therefore, when `recycle ~= 1`:

```text
content state h/c/mu/var is mostly reset from residual summary;
support/address inertia is downweighted;
slot_address barely updates toward the current observation address;
future predictive losses become hard to trust because identity continuity is weak.
```

This explains why support overlap can look acceptable while posterior identity
is still not healthy: AQR support is a current-step routing property, while
recycle is a temporal posterior-continuity property.

## 3. Evidence From Current Runs

All current staged runs use:

```text
use_foundation_backbones=true
perception_finetune_mode=frozen
picf_trainable_scope=anchor_only
unroll_steps=1
burnin_steps=0
accum_steps=4
high-risk predictive losses = 0
```

### 3.1 R0: anchor-only warmup

Run:

```text
/mnt/checkpoints/picf_core/picf_core/
  picf_mvtrack_anchor_noaction_supportdiv_acc4_lr3e4_probe300_20260510_225402
```

Result at step 300:

```text
loss_total                         = 0.7145
loss_action                        = 0
loss_alignment                     = 0.7145
loss_anchor_pv                     = 2.4138
loss_pv_weak                       = 1.6702
loss_mapg_support_diversity        = 0.0666
loss_mapg_routing                  = 1.1189
aqr_same_role_support_overlap_max  = 0.3756
last20 overlap mean                = 0.4815
aqr_effective_anchor_count         ~= 23.6
posterior_recycle_rate             = 0.3837
grad_norm                          = 0.1709
```

Interpretation:

```text
R0 proves the anchor-only objective can form non-collapsed supports.
It also proves recycle is not forced to 1.0 by logging or architecture alone.
```

### 3.2 Direct cotrain controls

R1 direct action + anchor:

```text
picf_r1_action_anchor_cache005_a1_2gpu_1500_20260511_0050
last20 overlap mean    = 0.9579
posterior_recycle_rate = 1.0
```

R2 direct action + denoise:

```text
picf_r2_action_anchor_cache005_denoise005_a1_2gpu_1500_20260511_0045
last20 overlap mean    = 0.9964
posterior_recycle_rate = 1.0
```

Interpretation:

```text
Direct cotrain can reduce scalar losses while collapsing support structure.
Denoising at 0.005 does not fix this and is worse early.
```

### 3.3 Staged action runs

S1, action weight 1.0:

```text
picf_s1_stage300_a1_cache005_sd005_den0_900_20260511_0155
last20 overlap mean    = 0.5944
posterior_recycle_rate = 1.0
```

S2, action weight 0.5:

```text
picf_s2_stage300_a05_cache005_sd005_den0_900_20260511_0155
last20 overlap mean    = 0.4551
posterior_recycle_rate = 1.0
```

S3, action weight 1.0, cache read 0:

```text
picf_s3_stage300_a1_cache000_sd005_den0_900_20260511_0155
last20 overlap mean    = 0.6720
posterior_recycle_rate = 1.0
```

D1, direct action weight 0.25:

```text
picf_d1_direct_a025_cache005_sd005_den0_600_20260511_0155
last20 overlap mean    = 0.7015
posterior_recycle_rate = 0.0
```

Interpretation:

```text
Staging is the main improvement.
S2 is the current best candidate.
Cache read is not the main collapse cause because S3 is not better than S1.
Low direct action avoids recycle=1 but does not match S2 support health.
Therefore recycle saturation is not just an action-weight scalar issue.
```

## 4. What Is Proven and What Is Not

Proven:

```text
1. Anchor-only warmup can lower overlap and preserve many effective anchors.
2. Direct action cotrain is structurally unsafe even when scalar losses drop.
3. Staged cotrain preserves support structure much better than direct cotrain.
4. Cache read 0.05 is not currently the leading cause of support collapse.
5. The displayed low S2 loss_action is not directly comparable to old runs,
   because S2 uses action lambdas 0.5 rather than the old default 2.0.
```

Not proven:

```text
1. Posterior identity continuity is healthy.
2. Predictive auxiliary losses are safe to enable.
3. Staged action 1.0 is safe for long runs.
4. Recycle=1.0 is behaviorally harmless.
```

The current action number should be interpreted as:

```text
S2 loss_action=0.0174 with action lambdas 0.5
default-equivalent loss_action ~= 0.0696
roughly comparable to old v22 runs around 300-1000 steps,
not to the old 20k-40k-step low-action-loss regime.
```

## 5. Mathematical Failure Mode

The action objective is largely indifferent to same-role anchor identity. If
several anchors attend to the same action-salient support, action prediction can
still improve. The structure objective, however, needs same-role supports to
remain separated and temporally stable.

The problematic optimization geometry is:

```math
L = L_action + L_alignment
```

where `L_action` can decrease even when:

```math
support_i ~= support_j,  i != j
```

and where the recycle path can give a cheap current-frame explanation:

```math
b_t ~= residual(o_t)
```

instead of a temporally persistent belief:

```math
b_t = U(prior(b_{t-1}), measurement(o_t)).
```

Thus scalar loss descent is not sufficient. The required health conditions are:

```text
same-role overlap stays low;
effective anchor count stays high;
recycle does not saturate;
identity switch does not rise;
action loss still trends down.
```

## 6. Why Anchors Become Unstable

Anchor instability in this system is not one bug. It is the expected failure
surface of a weakly supervised, permutation-symmetric object-slot system unless
the objective keeps measurement assignment, posterior identity, and action
credit assignment aligned.

### 6.1 Permutation symmetry

Same-role physical anchors are exchangeable unless broken by data, state, or
loss:

```math
L(Q_1,\ldots,Q_K) = L(Q_{\pi(1)},\ldots,Q_{\pi(K)})
```

for any permutation `pi` of same-role anchors. This symmetry is healthy at
initialization, but it also means the optimizer has no inherent reason to keep
two same-role anchors on two different nearby objects.

What breaks the symmetry in the current system:

```text
learned query tokens;
geometry priors;
support-diversity penalty;
posterior recurrence;
support-signature/address inertia;
task/action gradients.
```

What can re-collapse it:

```text
action gradients that only need one action-salient object;
local refinement that sharpens several anchors onto the same high-confidence support;
recycle saturation that removes previous identity inertia;
missing tracklet/proposal evidence in current dataflow;
predictive targets that are not safe under unstable slot identity.
```

Test:

```text
Run anchor-only continuation from R0/300 to 900.
If overlap remains low and recycle remains moderate, action/recurrent cotrain is the trigger.
If overlap/recycle degrade without action, the anchor/posterior structural losses are insufficient alone.
```

### 6.2 Action shortcut pressure

The action objective is not an object identity objective. A policy can often
reduce action error by reading the most task-relevant visual/PG/point region,
even if multiple same-role anchors read that same region.

The shortcut form is:

```math
support_i \approx support_j,\quad i \ne j
```

while:

```math
L_{action}(f(support_i,support_j,\ldots), a^*) \downarrow
```

This explains why direct cotrain R1/R2 can lower scalar loss while overlap
stays near 1.0. Action loss alone is not evidence that object slots are
healthy.

Tests:

```text
Compare direct action 1.0, direct action 0.25, staged action 0.5.
If only staged action keeps overlap low, the issue is not action learning itself;
it is action learning before anchors are in a good basin.
```

### 6.3 Support-diversity can oppose localization

The support-diversity term penalizes same-role overlap. That is necessary.
However, if the task contains one dominant salient region, diversity can oppose
localization:

```math
L_{support-div} \downarrow \quad wants \quad p_i^\top p_j \downarrow
```

while:

```math
L_{action}, L_{PV} \downarrow \quad may want \quad p_i,p_j \rightarrow p_{salient}
```

This is why simply increasing `lambda_mapg_support_diversity` may not solve the
problem. S5 is a direct test of that hypothesis. If S5 has higher support-div
loss and still high overlap, the system is finding a stronger localization
shortcut or the diversity loss is not targeting the active support measure that
collapses.

Tests:

```text
S5: support_div=0.10.
Add per-modality overlap metrics if S5 is inconclusive:
  visual overlap;
  temporal overlap;
  PG overlap;
  point overlap;
  local-refinement overlap.
```

### 6.4 Local refinement can sharpen collapse

Local refinement is useful because it rereads top-k typed evidence near the
anchor. But if several anchors already choose the same coarse support, local
refinement can sharpen all of them onto the same local set:

```math
\Omega_i \approx \Omega_j
\Rightarrow
LocalRead_i \approx LocalRead_j
```

This is not a reason to remove local refinement; it is a reason to inspect
whether collapse occurs before or after local refinement.

Tests:

```text
Log overlap before local refinement and after local refinement.
Run one staged action probe with local_refinement_weight=0.
If pre-local overlap is healthy but post-local overlap collapses, the local
top-k set needs diversity-aware exclusion or per-anchor non-maximum suppression.
```

### 6.5 Recycle creates a positive-feedback loop

In `_binding_logits`, previous support/address inertia is downweighted by
previous recycle:

```math
inertia_j \propto (1 - recycle_{t-1,j})
```

In address update, the address update rate is also downweighted by current
recycle:

```math
\rho_j \propto support\_mass_j (1 - recycle_{t,j})
```

Therefore high recycle can self-maintain:

```text
high recycle -> weak identity inertia -> less stable binding -> high recycle
```

This is the most important theoretical reason not to enable predictive identity
losses yet. If the posterior itself is resetting, a future-slot teacher is not
a stable identity teacher.

Tests:

```text
Log recycle logits and previous-recycle-conditioned inertia.
Log address_update_rate_mean/max.
If address_update_rate is near zero whenever recycle=1, address cannot recover identity.
```

### 6.6 Dustbin/residual shortcut

The Sinkhorn dustbin and recycle residual path are necessary because new or
unmatched evidence must enter the posterior. But this path can become a
shortcut if it is easier than maintaining slot continuity.

The current flow is:

```math
binding\_raw = Sinkhorn([slot\_logits; dustbin])
residual\_summary = \sum_i dustbin_i o_i
recycle = sigmoid(g(h^-, support\_mass, var^-, residual\_summary, alpha^-))
```

If `dustbin_raw` is large or `residual_summary` dominates, the model can
continually reconstruct posterior content from current evidence and avoid
using prior identity.

Tests:

```text
Log dustbin_raw.mean/max.
Log residual_summary_norm.
Log support_mass_raw.mean/min.
Compare R0, S2, D1.
If S2 has high dustbin/residual while R0 does not, action cotrain is opening a residual shortcut.
```

### 6.7 Frozen evidence limits

Current probes freeze foundation perception and train only anchor/PICF adapter
subsets. This is correct for isolating anchor behavior, but it also means the
system cannot improve underlying representation separability:

```math
I(Y; A_t) \le I(Y; Z_t)
```

Anchor training can route existing evidence better. It cannot create missing
fine-instance information. This matters when interpreting failures on small
nearby objects: a failed slot split can be due to anchor instability, but it can
also be due to insufficient evidence resolution.

Tests:

```text
Use support videos/overlays to distinguish:
  evidence exists but anchors collapse;
  evidence itself is not separable.
Add wrist/temporal/point view-mass diagnostics before blaming the anchor loss.
```

## 7. Loss and Module Conflict Matrix

This section lists conflicts that must be tested explicitly. Each conflict is a
real optimization possibility, not necessarily a proven current bug.

| Conflict | Mechanism | Current Evidence | Test |
| --- | --- | --- | --- |
| Action vs anchor diversity | Action can use duplicated salient support; diversity wants separation | R1/R2 loss down but overlap near 1 | Staged vs direct action; action ramp |
| Action vs recycle continuity | Current-frame residual path may lower action without identity carry | S1/S2/S3 recycle=1 | Recycle logits, dustbin, residual norm |
| Support diversity vs localization | Strong diversity may fight PV/action localization | S5 early not clearly better | Support-div sweep plus per-modality overlap |
| Local refinement vs diversity | Top-k reread can sharpen several anchors onto same local set | Not yet isolated | local_refinement_weight=0 probe; pre/post local overlap |
| Cache vs posterior | Cache can reinforce stale identity if address/recycle wrong | S3 cache=0 not better than S1, so not primary | Keep cache 0/0.05 comparison, inspect cache mass/source |
| Denoising vs anchor identity | DN target can pull anchors toward noisy pseudo peaks | R2 bad at 0.005; S4 still pending | S4 0.001 vs S2 |
| Slot-JEPA vs permutation | Future target can be wrong under slot swap/recycle | slot_jepa O(1e3) in staged runs | Keep zero until recycle stable; test permutation-invariant metric only |
| Support prediction vs unstable support | Predicts future support summary before identity is stable | diagnostic small but identity unstable | Try <=1e-4 only after recycle instrumentation |
| Binding consistency vs recycle | Identity contrast assumes carry; recycle=1 invalidates carry | recycle=1 in staged action | Wait; optionally mask recycled slots |
| Alignment budgeting vs action lambda | Lower action lambda changes auxiliary/action balance | S2 action lambda 0.5 not directly comparable | Report default-equivalent action and budget scales |

Important implication:

```text
Do not solve conflicts by adding all losses. First determine which conflict is active.
Adding all losses can make the scalar objective smoother while hiding the causal failure.
```

## 8. Testing Strategy

The goal is not to find the lowest `loss_total` in the next 10 hours. The goal
is to isolate which mechanism breaks posterior identity.

### 8.1 Minimal diagnostic axes

Use only one change per run:

```text
action weight:
  0, 0.25, 0.5, 1.0

training schedule:
  direct vs R0-staged

cache:
  0 vs 0.05

support diversity:
  0.05 vs 0.10

denoising:
  0 vs 0.001

local refinement:
  0.25 vs 0
```

### 8.2 Primary metrics

The primary metrics are:

```text
aqr_same_role_support_overlap_max
aqr_same_role_local_overlap_max
aqr_effective_anchor_count
posterior_recycle_rate
posterior_recycle_logit_mean/std/min/max
posterior_recycle_gate_std/min/max
posterior_recycle_rate_effector
posterior_recycle_rate_scene
posterior_dustbin_mass_raw
posterior_dustbin_mass_final
posterior_support_mass_raw_mean
posterior_support_mass_final_mean
posterior_prior_var_mean
posterior_prior_alpha_mean
posterior_residual_summary_norm
posterior_identity_innovation_risk
posterior_address_update_rate_mean
posterior_address_update_rate_max
loss_action_active7
loss_action
loss_action_default_equiv
loss_action_weight_scale
default-weight-equivalent loss_action
```

Interpretation:

```text
high recycle_logit_mean:
  recycle head itself is saturated; inspect support/raw dustbin inputs before
  adding any loss penalty.

high dustbin_raw or high residual_summary_norm:
  the model may be using residual recycle as a cheap current-frame explanation.

support_mass_raw low but support_mass_final high:
  dustbin-to-recycle transfer is creating apparent support after the fact.

address_update_rate near zero with recycle near one:
  slot addresses cannot recover identity even if current support improves.

effector recycle != scene recycle:
  the failure is role-specific; do not apply a global recycle penalty.

local overlap high while visual overlap is moderate:
  local refinement is sharpening multiple same-role anchors onto the same local
  evidence set.

loss_action_default_equiv:
  compare this value against the 2026-04-22 ablation baseline. `loss_action`
  remains the actual optimized weighted action term, so runs with
  `lambda_action_*=0.5` will log a much smaller raw `loss_action`; the
  default-equivalent field maps it back to the default action-lambda-2.0 scale.
```

The secondary metrics are:

```text
loss_total
loss_alignment
loss_anchor_pv
loss_pv_weak
loss_mapg_routing
loss_mapg_support_diversity
loss_aqr_denoising
loss_slot_jepa diagnostic only
loss_support_pred diagnostic only
loss_binding_consistency diagnostic only
```

### 8.3 Decision tree

If staged action 0.25 still gives `recycle=1.0`:

```text
Action magnitude is not the primary cause.
Inspect recycle logits/dustbin/support-mass path.
```

If anchor-only continuation R0 -> 900 gives `recycle=1.0`:

```text
The issue is posterior/recycle calibration under anchor structural training.
Do not proceed to action long run before fixing recycle.
```

If anchor-only continuation stays moderate but staged action gives `recycle=1.0`:

```text
Action cotrain opens the recycle shortcut.
Use action ramp or freeze recycle_head/address paths during early action.
```

If cache=0 and cache=0.05 are similar:

```text
Cache is not the first-order problem.
Do not spend time overfitting cache before recycle diagnosis.
```

If support_div=0.10 improves overlap but not recycle:

```text
Support separation and posterior identity are decoupled.
Need recycle calibration, not just stronger diversity.
```

If denoise=0.001 worsens overlap/recycle:

```text
Denoising pseudo targets are not reliable yet.
Keep denoise off until support is stable.
```

If local_refinement_weight=0 improves overlap:

```text
Local refinement is sharpening collapse.
Add diversity-aware local candidate exclusion before restoring it.
```

## 9. Candidate Fixes After Diagnosis

Do not apply these before instrumentation. They are listed to keep the response
planned and non-ad-hoc.

### 9.1 Recycle head calibration

If logits are saturated high:

```text
initialize recycle_head bias lower;
add mild recycle prior only for stable supported slots;
separate effector and scene recycle bias;
clamp recycle during staged action warmup.
```

Mathematical guard:

```math
L_{recycle-prior} = mean(alpha_j support_j (recycle_j - r_0)^2)
```

only for high-support, low-innovation slots. Do not penalize recycle for truly
unmatched or new evidence.

### 9.2 Dustbin/residual control

If dustbin/residual shortcut dominates:

```text
lower dustbin transfer into every slot;
route dustbin primarily to birth/recycle candidates;
log and optionally cap residual_summary contribution to recycle_head.
```

Guard:

```text
Do not remove dustbin. New objects/unmatched observations need it.
```

### 9.3 Action ramp

If action triggers recycle only above a threshold:

```text
R0 300 -> action 0.25 for 300-600
action 0.5 for 600-1200
action 1.0 only after overlap/recycle gates pass
```

Acceptance gate before ramp:

```text
last20 overlap < 0.65
recycle not saturated
effective anchors > 23
grad stable
```

### 9.4 Recycle-aware predictive masks

If predictive aux is later enabled:

```text
mask slots with recycle > threshold;
mask low support_mass;
mask high innovation;
use detached targets only;
start support_pred before slot_jepa.
```

Slot-JEPA should remain off until:

```text
slot_jepa diagnostic is O(1), not O(1e3);
recycle is not saturated;
identity switch remains low;
support overlap is stable.
```

## 10. Working Hypotheses

H1: Recycle head saturation.

```text
The recycle head logits may saturate after action is introduced.
Need to log recycle logits, not only sigmoid outputs.
```

H2: Dustbin/residual shortcut.

```text
The model may reduce action/alignment by routing through residual_summary
instead of preserving prior slot state.
Need to log dustbin mass, support_mass_raw, and residual_summary norm.
```

H3: Support is current-step healthy but temporally memoryless.

```text
AQR support can remain non-collapsed while posterior content resets each step.
Need to separate support health from posterior continuity.
```

H4: Positive feedback through recycle-gated inertia.

```text
High previous recycle disables support-signature inertia via (1 - recycle).
That weakens identity continuity, making future recycle more likely.
```

H5: Action weight is not the only cause.

```text
S2 action=0.5 still recycle=1.0.
D1 direct action=0.25 recycle=0.0 but worse support health.
Need a staged action=0.25 run to separate action strength from resume/stage effects.
```

H6: Predictive aux would currently amplify wrong identities.

```text
slot_jepa is thousands in staged action runs.
With lambda 1e-4 it would already contribute O(0.3), comparable to major losses.
Do not enable it before recycle is understood.
```

H7: Local refinement may sharpen same-support convergence.

```text
Local reread can improve task evidence while making multiple anchors share the
same top-k local set. Need pre/post-local overlap diagnostics or a
local_refinement_weight=0 ablation.
```

H8: Alignment-group budgeting changes objective balance when action lambdas are lowered.

```text
S2 uses action lambdas 0.5. Weighted action is lower, but alignment budget and
total-loss interpretation change. Always report default-equivalent action loss
and loss_action_active7.
```

## 11. Required Instrumentation

Add debug-only metrics before changing the training objective:

```text
posterior_recycle_logit_mean
posterior_recycle_logit_std
posterior_recycle_logit_min
posterior_recycle_logit_max
posterior_recycle_rate_effector
posterior_recycle_rate_scene
posterior_support_mass_raw_mean
posterior_support_mass_final_mean
posterior_dustbin_mass_mean
posterior_residual_summary_norm
posterior_recycle_input_var_prior_mean
posterior_recycle_input_alpha_prior_mean
posterior_recycle_input_support_mass_mean
posterior_address_update_rate_mean
posterior_address_update_rate_max
```

These metrics should be added to `pipeline.py` debug emission only. They should
not change forward behavior.

Rationale:

```text
If logits are saturated high, tune recycle head initialization/regularization.
If dustbin mass is high, inspect binding and observation-anchor coverage.
If support mass is healthy but recycle remains high, recycle head calibration
is likely wrong.
If only scene slots recycle, role-specific bias is needed.
If only effector recycles, contact/proprio/action interaction is suspect.
```

Additional pre/post local refinement instrumentation:

```text
aqr_same_role_support_overlap_pre_local
aqr_same_role_support_overlap_post_local
aqr_local_candidate_overlap_mean
aqr_local_candidate_overlap_max
```

Additional action comparability metrics:

```text
loss_action_default_equivalent
lambda_action_pos_effective
lambda_action_rot_effective
lambda_action_gripper_effective
```

## 12. Next Experiment Matrix

Do not open all predictive losses. Use the following sequence.

### E1. Staged action 0.25

Purpose:

```text
Separate action weight from staged resume effects.
```

Recipe:

```text
resume R0/300
lambda_action_pos/rot/gripper = 0.25
cache = 0.05
support_div = 0.05
denoise = 0
total steps = 900
```

Expected diagnosis:

```text
If recycle stays 1.0, action weight is not the primary trigger.
If recycle drops while overlap remains healthy, ramp action 0.25 -> 0.5.
```

### E2. Staged action 0.5 with recycle instrumentation

Purpose:

```text
Repeat S2 with recycle logits/input metrics.
```

Expected diagnosis:

```text
Confirms whether S2's recycle=1 is logit saturation, dustbin shortcut,
role-specific behavior, or support-mass mismatch.
```

### E3. Anchor-only continuation from R0

Purpose:

```text
Check whether recycle drifts to 1.0 without action over 300 -> 900.
```

Recipe:

```text
resume R0/300
action = 0
support_div = 0.05
total steps = 900
```

Expected diagnosis:

```text
If recycle rises to 1.0 without action, the issue is anchor/posterior objective
calibration rather than action conflict.
If recycle stays moderate, action coupling is implicated.
```

### E4. Denoising low-weight staged run

Purpose:

```text
Finish S4 only if recycle instrumentation is not yet available.
```

Current status:

```text
S4 early data is insufficient.
Do not infer that denoise helps unless it beats S2 on overlap and recycle.
```

### E5. Support-diversity 0.10 staged run

Purpose:

```text
Check whether stronger support diversity can replace action lowering.
```

Current status:

```text
S5 early data is not encouraging.
If overlap remains high by step 600, stop it; do not run to 900 only for scalar loss.
```

## 13. Acceptance Criteria

For any staged cotrain candidate:

```text
loss_action_active7:
  should trend down, but is not the primary acceptance metric.

default-equivalent loss_action:
  report by normalizing action lambdas before comparing historical runs.

aqr_same_role_support_overlap_max:
  last20 mean should be < 0.65 for guarded acceptance;
  < 0.50 is strong.

aqr_effective_anchor_count:
  should stay > 23 for the current 24-anchor graph.

posterior_recycle_rate:
  should not stay at 1.0.
  Target for healthy identity carry is not known yet, but persistent 1.0 is
  treated as unresolved.

posterior_identity_switch_rate:
  should remain low, but do not trust it alone when recycle=1.0.

grad_norm:
  should stay stable and mostly below clip threshold.
```

For predictive aux activation:

```text
Do not enable slot_jepa while recycle=1.0 or slot_jepa diagnostic is O(1e3).
support_pred may be tested first at <= 1e-4 only after recycle instrumentation.
binding_consistency should wait until recycle is no longer saturated.
```

## 14. What Not To Do

Do not:

```text
1. Treat low weighted loss_action as comparable across action lambda settings.
2. Claim S2 has old 20k-step action quality because loss_action=0.0174.
3. Enable slot_jepa/support_pred/binding_consistency together.
4. Add a recycle penalty before logging recycle logits and dustbin/support mass.
5. Judge anchor health from loss_total alone.
6. Judge identity health from same-role overlap alone.
7. Use old direct-cotrain results as evidence that staged cotrain fails.
```

## 15. Current Operational Tail Commands

New machine:

```bash
ssh -p 29776 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_s1_stage300_a1_cache005_sd005_den0_900_20260511_0155/metrics.jsonl
tail -f /mnt/checkpoints/picf_core/picf_core/picf_s3_stage300_a1_cache000_sd005_den0_900_20260511_0155/metrics.jsonl
tail -f /mnt/checkpoints/picf_core/picf_core/picf_s5_stage300_a1_cache005_sd010_den0_900_20260511_0155/metrics.jsonl
```

Old machine:

```bash
ssh -p 28060 root@36.139.225.68
tail -f /mnt/checkpoints/picf_core/picf_core/picf_s2_stage300_a05_cache005_sd005_den0_900_20260511_0155/metrics.jsonl
tail -f /mnt/checkpoints/picf_core/picf_core/picf_d1_direct_a025_cache005_sd005_den0_600_20260511_0155/metrics.jsonl
tail -f /mnt/checkpoints/picf_core/picf_core/picf_s4_stage300_a05_cache005_sd005_den001_900_20260511_0155/metrics.jsonl
```

## 16. Current Recommendation

Use this as the next decision rule:

```text
Main candidate:
  R0 -> S2 style staged cotrain.

Do not change:
  keep predictive aux at 0.
  keep denoising at 0 unless S4 clearly beats S2.

Prioritize:
  recycle instrumentation.
  staged action 0.25.
  anchor-only continuation from R0.

Only after recycle is understood:
  consider low-weight support_pred.
  consider low-weight binding_consistency.
  keep slot_jepa off until its diagnostic magnitude is small and identities
  are stable.
```

The working training conclusion is:

```text
Staged cotrain is viable.
S2 is the current best evidence.
Posterior recycle is the blocker for claiming identity health or enabling
predictive auxiliary losses.
```
