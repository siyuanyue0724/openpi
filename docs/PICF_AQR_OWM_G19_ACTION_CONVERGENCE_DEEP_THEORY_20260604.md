# PICF-AQR-OWM G19 Action Convergence Deep Theory Audit

Date: 2026-06-04

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

Status:

```text
root-cause theory audit complete;
do not repeat sampler/optimizer-only experiments unless this document names
a new falsifiable hypothesis.
```

This document answers the current action-convergence question after G12-G18:

```text
Why do the maintained task-balanced / logical-batch / PCGrad / action-bridge
experiments still plateau around the balanced-window 0.04 band, even though
historical train-stream rows sometimes reached 0.02?
```

It is intentionally stricter than a run log.  Every recommendation below must
pass four checks:

```text
1. mathematical target: what gradient/objective/dataflow it changes;
2. code follow-through: whether that path already exists;
3. experiment status: already tested, rejected, retained, or missing;
4. scaling compatibility: whether it remains valid for future large,
   heterogeneous, partially missing-modality datasets.
```

Related documents:

```text
docs/PICF_AQR_OWM_G12_ALL_REQUESTED_VLA_METHODS_TEST_PLAN_20260603.md
docs/PICF_AQR_OWM_G17_COMPLETE_VLA_METHOD_GATE_20260604.md
docs/PICF_AQR_OWM_G18_ALL_VLA_METHODS_FOLLOWTHROUGH_20260604.md
docs/PICF_AQR_OWM_ONE_HOUR_E21_DIAGNOSTIC_20260604.md
docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md
docs/PICF_AQR_OWM_ACTION_LOSS_WINDOW_AUDIT_20260530.md
```

## 1. Strict Facts From Code And Logs

### 1.1 The task-balanced gradient estimator is implemented

Code:

```text
scripts/picf_core_train.py
  _compute_bucket_sampling_weights()
  _bucket_sequence_for_logical_step()
  _logical_batch_loss_scales()
  _dynamic_bucket_sampling_weights()
  _pcgrad_project_and_sum()
  _cagrad_project_and_sum()
```

The maintained logical-batch estimator is:

```text
g_hat = sum_{b in B_step} (q_b / n_b) grad L_b
```

with a DDP correction:

```text
local_backward_scale = world_size * q_b / n_b
```

Therefore the current issue should not be described as:

```text
"we forgot to implement task-balanced logical batch"
```

The code already expresses the VLA-Foundry / ABot-M0 style requirement that an
optimizer step approximate the multi-task mixture gradient:

```text
G = sum_b q_b grad L_b
```

### 1.2 The sampler/mixing family is necessary but already insufficient

G12/G17/G18 already cover:

```text
task_uniform logical batch
without-replacement bucket coverage
explicit bucket ratios
temperature sampling
trajectory-proportional sampling
dynamic PiKE-style mixing
per-bucket logical loss normalization
per-bucket action EMA scaling
scoped PCGrad
scoped CAGrad
```

Observed pattern:

```text
structure metrics can improve;
action_default_equiv still fails the decisive descent gate;
large-card/K12 conditions do not reproduce stable E21-style 0.02.
```

Conclusion:

```text
Do not spend the next experiment repeating sampler-only or gradient-surgery-only
variants.  They remain required training infrastructure, not the missing
action-convergence mechanism.
```

### 1.3 The current action bridge is live but not causally beneficial

Code:

```text
src/openpi/picf/paligemma/wrapper.py
  _apply_action_context_adapter()
  _apply_action_expert_router()
  compute_action_flow_loss()
```

G17 fixed-window bridge evidence:

```text
normal_suffix first10 action ~= 0.05230
normal_suffix last10  action ~= 0.04102

no_picf_action first10 action ~= 0.05133
no_picf_action last10  action ~= 0.04082
```

Conclusion:

```text
The suffix action-context adapter is structurally safe, but current evidence
does not show that PICF context improves action loss on identical windows.
```

This is the strongest signal in the current evidence stack.  It means the
action path is either:

```text
1. ignoring PICF context;
2. receiving context in a representation that is not motor-action useful;
3. using context only after the continuous action expert has already saturated;
4. bottlenecked by action expert capacity/objective rather than object belief.
```

### 1.4 OpenPI native FAST exists, but PICF lacks the equivalent auxiliary

Code:

```text
src/openpi/models/tokenizer.py
  FASTTokenizer

src/openpi/transforms/_base.py
  TokenizeFASTInputs
  ExtractFASTActions

src/openpi/models/pi0_fast.py
  Pi0FAST.compute_loss()
```

Native FAST objective:

```text
prefix = text + state
postfix = "Action:" + FAST(action_chunk)
L_fast = CE(next_token | prefix, previous_action_tokens)
```

PICF core training currently has:

```text
continuous action chunk flow loss;
loss_action_default_equiv metric;
no deployed PICF-side loss_action_fast_ce;
no deployed action-token representation objective connected to PICF belief.
```

This is the one high-value mechanism that is literature-supported and not yet
tested in PICF.

## 2. Paper-Level Cross-Check

The requested 2025-2026 method family maps to our code as follows.

```text
VLA Foundry:
  supports task/dataset mixture control, gradient accumulation, and
  normalization as framework-level training infrastructure.
  Our code implements the corresponding bucket q_b and logical-batch scaling.

ABot-M0:
  supports task/embodiment-aware balancing and action/data standardization.
  Our CALVIN setting is single embodiment, so the relevant part is task-family
  balancing, already implemented; multi-embodiment heads remain a future
  large-data branch.

PiKE:
  supports dynamic task mixing when measured progress/lag justifies changing
  q_b(t).  Our dynamic bounded q_b path exists and was tested; it did not solve
  the action plateau alone.

OpenVLA-OFT:
  supports chunked continuous action with simple stable objectives.  Our PICF
  action path already uses chunked continuous flow and logs a canonical
  default-equivalent metric.

Knowledge Insulation / pi0.5 / FAST:
  supports separating noisy continuous action gradients from the semantic VLM
  while giving the semantic side a motor-representation objective.
  This is exactly where PICF is incomplete: continuous action exists, but the
  PICF-side FAST/action-token representation auxiliary does not.

GR00T N1 / Gemini Robotics / AdaMoE / FedVLA:
  support dual-system or expertized action modules when the action module
  itself lacks capacity or tasks conflict.  These do not justify MoE-ing the
  whole backbone first.  They justify an action-expert-only branch only after
  the action-token representation branch is tested.
```

References:

```text
VLA Foundry: https://arxiv.org/abs/2604.19728
ABot-M0: https://arxiv.org/abs/2602.11236
PiKE: https://arxiv.org/abs/2502.06244
OpenVLA-OFT: https://arxiv.org/abs/2502.19645
Knowledge Insulation: https://arxiv.org/abs/2505.23705
pi0.5: https://arxiv.org/abs/2504.16054
GR00T N1: https://arxiv.org/abs/2503.14734
FedVLA: https://arxiv.org/abs/2508.02190
AdaMoE-VLA: https://charleshen1412.github.io/AdaMoE-VLA/
FAST: https://www.physicalintelligence.company/research/fast
```

## 3. Mathematical Root-Cause Model

The current VLA path can be decomposed as:

```text
X_t: base PI0.5 visual-language-state input
Z_t: PICF multimodal evidence
B_t = F_phi(Z_t, X_t): PICF posterior belief
C_t = S_psi(B_t): bounded action-visible PICF context
I_t = H_theta(X_t, C_t): action expert conditioning
a_hat = A_theta(I_t): predicted action chunk
L_flow = ||a_hat - a||^2 or equivalent flow target
```

The action gradient through PICF context is:

```text
dL_flow/dphi =
  dL_flow/da_hat
  * da_hat/dI_t
  * dI_t/dC_t
  * dC_t/dB_t
  * dB_t/dphi
```

The evidence says:

```text
active/downstream object structure can be healthy;
sampler coverage can be correct;
gradient surgery can be active;
normal PICF context ~= no PICF context on identical action windows.
```

Therefore the currently best-supported low-gain term is:

```text
dI_t/dC_t
```

not:

```text
raw inactive same-role overlap;
blind SAM proposal quality;
logical batch estimator correctness;
generic action scalar weight;
another object pull/diversity penalty.
```

This is why the next change should not be another object-side loss.  It should
make the action expert learn a motor-relevant representation of action chunks
from semantic/PICF condition.

## 4. Why Historical 0.02 Does Not Contradict Current 0.04

The May action-window audit proves both facts:

```text
old train-stream rows reached about 0.02 and stayed low for long stretches;
old fixed/balanced replay on broader windows was about 0.04-0.05.
```

The one-hour E21 diagnostic found current fixed-window means:

```text
current_trace32 ~= 0.0380
old_exact32 ~= 0.0433
old_stratified_valid32 ~= 0.0423
```

Strict interpretation:

```text
The old 0.02 was real for the train stream, but it is not the same measurement
as current balanced/fixed-window probes.
```

The actionable target for production is not:

```text
"reproduce one historical train-stream row"
```

It is:

```text
"push balanced/fixed-window action quality below the 0.04 band while preserving
task-balanced coverage and object/belief health."
```

## 5. Insightful Recommendation

### Recommendation A: Stop repeating sampler-only experiments

Deployment status:

```text
keep task_uniform logical batch;
keep without-replacement coverage;
keep per-bucket normalization;
keep action bucket metrics;
keep PCGrad/CAGrad/dynamic mixing as diagnostics/off-by-default;
do not rerun them as primary fixes.
```

Mathematical reason:

```text
They reduce Var[g_hat] and destructive bucket interference, but they do not
create a stronger motor-representation link from C_t to action prediction.
```

### Recommendation B: Run an exact action-context causality probe before adding more modules

For identical windows/noise/time:

```text
normal C_t
zero C_t
shuffle C_t across batch
sign-flip C_t
sidecar/oracle-object C_t if available
no_picf_action
```

Acceptance:

```text
If normal ~= zero ~= shuffle, PICF context has low action causal gain.
If oracle-object C_t helps but normal C_t does not, belief quality is the issue.
If normal helps but train still plateaus, optimizer/weighting remains suspect.
```

This probe gives a falsifiable answer before a large code change.

### Recommendation C: Implement PICF-side FAST/action-token representation auxiliary

Objective:

```text
y = FASTTokenizer(a_{1:H})
L_fast = CE(p_theta(y_i | y_<i, X_t, C_t), token_loss_mask_i)
L_total = L_flow + lambda_fast * L_fast + existing guarded structure losses
```

Gradient rule:

```text
continuous action flow:
  update action expert / action adapter;
  keep PICF/VLM boundary controlled.

FAST action-token auxiliary:
  update the semantic/action bridge so it learns motor-relevant action tokens;
  do not let it dominate object belief losses.
```

Why this is coherent:

```text
It does not replace PICF belief.
It does not hard-label object masks.
It uses OpenPI-native FAST machinery.
It directly targets the missing representation link between semantic/PICF
condition and the action expert.
It scales to large heterogeneous datasets because action chunks are always
available where action labels exist; missing tactile/point/cloud modalities are
not required.
```

Required logs:

```text
loss_action_fast_ce
loss_action_flow
loss_action_default_equiv
lambda_action_fast_ce
fast_token_count
fast_loss_mask_count
picf_action_context_causal_gain
```

Initial gate:

```text
Run 100-300 steps from a clean checkpoint with:
  task_uniform logical batch on;
  per-bucket normalization on;
  SAM off;
  sidecar weak only;
  continuous action loss unchanged;
  lambda_fast small and ramped.

Pass if:
  loss_action_default_equiv shows a stronger negative slope than G17/G18;
  loss_action_fast_ce decreases;
  active/downstream overlap remains healthy;
  no regression in timing beyond the FAST head overhead budget.
```

### Recommendation D: Only after FAST fails, test action-expert capacity

If FAST does not improve balanced/fixed-window action, then the next coherent
branch is action expert capacity:

```text
action-expert-only router/MoE;
not full-backbone MoE;
not object-side patching.
```

Reason:

```text
The observed failure would then imply the condition is useful or trainable, but
the motor expert cannot use it under current capacity/objective.
```

## 6. Current No-Go List

Do not use the next GPU slot for:

```text
1. another raw same-role overlap penalty;
2. another blind SAM proposal attempt;
3. another sidecar/no-sidecar binary rerun;
4. another fixed64 comparison without exact-window category controls;
5. another scalar action-weight-only run;
6. another K12/more-card run without a new objective;
7. whole-backbone MoE;
8. full image reconstruction decoder as the next action-convergence fix.
```

Each of these either has negative evidence, insufficient causality, or weak
scaling compatibility for large heterogeneous data.

## 7. Final G19 Decision

The deepest current interpretation is:

```text
The maintained PICF system has moved from "object/belief collapse" to
"action readout / motor representation bottleneck".
```

The next non-duplicate, mathematically coherent, literature-supported branch is:

```text
PICF-side FAST/action-token representation supervision, preceded by exact
action-context causal sensitivity probes.
```

This is not a minimal patch.  It is the missing half of the action/semantic
insulation pattern already present in modern VLA training:

```text
continuous action chunks provide low-level motor regression;
FAST/action-token CE provides semantic motor representation;
bounded PICF context provides object belief;
task-balanced logical batch provides scalable multi-task gradient coverage.
```

Until that branch is tested, the strict claim should be:

```text
Training infrastructure and object-side collapse fixes are substantially
closed; balanced-window action convergence is not closed; the next root-cause
test is action representation/readout, not another sampler or anchor patch.
```
