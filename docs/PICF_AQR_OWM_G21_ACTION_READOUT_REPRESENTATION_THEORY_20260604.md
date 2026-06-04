# PICF-AQR-OWM G21 Action-Readout Representation Theory

Date: 2026-06-04

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

Status:

```text
strict theory and code-follow-through complete;
next valid implementation branch is PICF-side action-token / FAST-style
representation supervision, not another sampler-only or optimizer-only run.
```

This document records the deeper analysis requested after G20.  Its purpose is
to prevent another round of experiments that are mathematically adjacent but do
not change the failing causal path.

## 1. Hard Evidence Entering G21

### 1.1 Current action-context perturbation result

G20 evaluated the same checkpoint, same deterministic windows, same action
targets, and same suffix-cross-attention action bridge under four PICF action
context modes:

```text
none
zero
token_roll
sign_flip
```

Result:

```text
first deterministic pass:
  none       loss_action_default_equiv = 0.033852748
  zero       loss_action_default_equiv = 0.033643211
  token_roll loss_action_default_equiv = 0.033852748
  sign_flip  loss_action_default_equiv = 0.033667957

second deterministic pass:
  none       loss_action_default_equiv = 0.043279073
  zero       loss_action_default_equiv = 0.043007695
  token_roll loss_action_default_equiv = 0.043279073
  sign_flip  loss_action_default_equiv = 0.043128838
```

Therefore:

```text
L_action(A_theta(X, C)) ~= L_action(A_theta(X, perturb(C)))
```

This is stronger than "PICF context is noisy".  It says the current action
objective can be optimized almost independently of the current PICF action
context `C`.

### 1.2 Current code path

Observed code path:

```text
scripts/picf_core_train.py
  _compute_bucket_sampling_weights
  _dynamic_bucket_sampling_weights
  _bucket_sequence_for_logical_step
  _logical_batch_loss_scales
  _logical_action_bucket_scale
  _pcgrad_project_and_sum
  _cagrad_project_and_sum

src/openpi/picf/policy.py
  _action_context_tokens
  _training_action_condition_tokens
  observe

src/openpi/picf/paligemma/wrapper.py
  _apply_action_context_adapter
  _apply_action_expert_router
  compute_action_flow_loss
```

The action bridge is real:

```text
C = action context tokens from PICF conditioned_control
suffix' = suffix + gate * CrossAttention(q=suffix, k=C, v=C)
v_hat = ActionExpert(prefix, suffix')
L_flow = ||v_hat - u_t||^2
```

But the G20 perturbation result says the model behaves approximately as:

```text
ActionExpert(prefix, suffix, C) ~= ActionExpert(prefix, suffix, 0)
```

### 1.3 What is already implemented and should not be repeated alone

The following requested VLA-method families are real code and already have
experiment coverage in G12-G20:

```text
task-uniform logical batch
temperature sampling
explicit ratio sampling
trajectory-proportional sampling
bounded PiKE-style dynamic mixing
logical q_b / n_b loss scaling
per-bucket action EMA scaling
scoped PCGrad
scoped CAGrad
continuous action chunks / flow loss
suffix gated action-context cross-attention
action-expert router proxy
```

They are still useful infrastructure.  They are not the missing causal
mechanism for the current action plateau because none of them forces
`I(C; action | native PI input)` to become positive.

## 2. Mathematical Diagnosis

Let:

```text
X_t = native PI0.5 visual-language-state input
Z_t = PICF multimodal evidence
B_t = F_phi(Z_t, X_t)                 # belief router
C_t = S_psi(B_t)                      # action-visible PICF context
Y_t = target action chunk
```

Current flow action training minimizes:

```text
L_flow(theta, phi, psi)
  = E || A_theta(X_t, C_t, eps, tau) - u(Y_t, eps, tau) ||^2
```

where `A_theta` is the action expert and `u` is the flow target.

The desired gradient into PICF-side action context is:

```text
dL_flow/dpsi =
  dL_flow/dA
  * dA/dC_t
  * dC_t/dpsi
```

G20 empirically shows:

```text
dA/dC_t ~= 0
```

or, more precisely, `C_t` lies in a subspace that the current action expert can
ignore without increasing `L_flow`.  Under this condition, changing task
sampling only changes which `dL_flow/dA` is seen.  It does not repair the
nearly-zero `dA/dC_t` factor.

This explains the historical pattern:

```text
sampler / PCGrad / CAGrad:
  can improve structure or reduce gradient variance;
  cannot force the action head to use PICF context.

action-weight / LR / optimizer resets:
  can improve the native action path or short-window fitting;
  do not prove PICF context is useful.

raw same-role overlap:
  may be visually or structurally undesirable;
  is not the main action plateau cause when action loss is invariant to
  context zeroing/rolling/sign-flipping.
```

## 3. Paper-To-Code Mapping

### 3.1 VLA Foundry / ABot-M0 / PiKE

These papers support controlled data mixing and stable multi-task update
estimation:

```text
L = sum_b q_b L_b
G = sum_b q_b grad L_b
g_hat = sum_{b in step} (q_b / n_b) grad L_b
```

PICF code already implements this family through bucket weights,
without-replacement logical bucket choice, DDP-aware logical loss scaling, and
dynamic bounded bucket weights.  Therefore these papers support keeping the
current task-balanced infrastructure, but they do not justify another isolated
sampler run as the next fix.

### 3.2 OpenVLA-OFT

OpenVLA-OFT supports continuous action chunks and simple stable action
objectives.  PICF already uses continuous action chunks through
`compute_action_flow_loss`.

Implication:

```text
Do not rewrite the action objective just to become continuous.
That part is already structurally aligned.
```

### 3.3 Knowledge Insulation / pi0.5

Knowledge-insulation style VLA design says continuous action experts can
damage or bypass semantic VLM knowledge unless the action boundary is
controlled.  PICF already has stop-gradient/bounded context options.  G20 says
the opposite failure now dominates:

```text
not "action gradients are overusing PICF context";
but "action readout is not using PICF context at all".
```

The missing piece is a representation target that makes PICF context
motor-readable before the continuous flow head is expected to benefit.

### 3.4 AdaMoE / FedVLA

Action-specialized MoE is justified when action expert capacity or
task-family-specific motor primitives are the bottleneck.  Current evidence is
not yet that.  We must first make `C_t` causally useful.  If context-sensitive
FAST/action-token supervision succeeds but flow action still plateaus, then an
action-expert-only MoE/router becomes a valid second-stage branch.

## 4. Insightful Next Fix

### 4.1 Add PICF-side action-token / FAST-style representation supervision

Native OpenPI already has:

```text
src/openpi/models/tokenizer.py
  FASTTokenizer

src/openpi/transforms/_base.py
  TokenizeFASTInputs
  ExtractFASTActions

src/openpi/models/pi0_fast.py
  Pi0FAST.compute_loss
```

PICF does not yet have:

```text
loss_action_fast_ce
```

The next deployable branch should add a PICF-side action-token objective:

```text
y_1:M = FAST(Y_t)
p_i = P_eta(y_i | y_<i, X_t, C_t)
L_fast = - sum_i mask_i log p_i[y_i]

L_total =
  L_flow
  + lambda_fast(t) L_fast
  + existing guarded PICF losses
```

Key design requirements:

```text
1. C_t must be in the conditioning path of L_fast.
2. L_fast must log separately as loss_action_fast_ce.
3. perturb(C_t) must be measurable for L_fast and L_flow.
4. FAST supervision is auxiliary; L_action_default_equiv remains pure and
   comparable to historical 4-22 ablation.
5. It uses only existing action labels, not SAM/mask/tracklet labels, so it
   scales to large heterogeneous datasets.
```

Why this is not a patch:

```text
The current missing factor is dA/dC.  FAST supervision creates a second
motor-representation path with direct conditional dependence on C:

  dL_fast/dpsi =
    dL_fast/dp
    * dp/dC_t
    * dC_t/dpsi

If this path works, C_t becomes predictive of action symbols before the
continuous flow expert is asked to exploit it.  This matches the VLA pattern of
separating semantic/motor representation learning from final continuous action
generation.
```

### 4.2 Required causal metrics

Every FAST/action-token experiment must report:

```text
loss_action_fast_ce
loss_action_default_equiv
fast_context_gain_zero =
  L_fast(C=zero) - L_fast(C=none)

fast_context_gain_roll =
  L_fast(C=token_roll) - L_fast(C=none)

fast_context_gain_sign =
  L_fast(C=sign_flip) - L_fast(C=none)

flow_context_gain_zero =
  L_flow(C=zero) - L_flow(C=none)

grad_norm_action_context_fast
grad_norm_action_context_flow
```

Acceptance:

```text
Before claiming repair:
  fast_context_gain_* must become positive and stable.

Before claiming action benefit:
  flow_context_gain_* must become positive or loss_action_default_equiv must
  improve against exact no-PICF/action-context controls.
```

This avoids false positives where the auxiliary loss decreases but the action
head still ignores PICF context.

## 5. Experiment Plan

### G21-A: Local implementation gate

Checklist:

```text
[ ] add or wrap FAST tokenization for the PICF PyTorch train window
[ ] add PICF-side FAST/action-token head or compatible PaliGemma action-token path
[ ] log loss_action_fast_ce independently
[ ] keep loss_action_default_equiv unchanged
[ ] add context-perturb evaluation for L_fast
[ ] py_compile touched files
[ ] unit test shape/mask/causal context perturbation
```

Stop condition:

```text
Do not launch a GPU run if loss_action_fast_ce is absent from local logs.
```

### G21-B: 100-300 step smoke

Use:

```text
task-uniform logical batch
logical q_b/n_b normalization
SAM off
sidecar weak, if available
continuous flow action unchanged
lambda_fast small warmup, e.g. 0.05 -> 0.2
```

Acceptance:

```text
loss_action_fast_ce decreases over the first 100-300 steps.
fast_context_gain_zero > 0.
fast_context_gain_roll > 0.
loss_action_default_equiv does not regress outside the G17/G20 fixed-window band.
```

### G21-C: exact control

Run matched windows:

```text
FAST with PICF context
FAST with zero context
FAST with token_roll context
FAST with no PICF context
```

Acceptance:

```text
PICF context must help L_fast on identical windows.
If it does not, the issue is not the continuous flow objective alone; the PICF
context representation itself is not motor-readable.
```

### G21-D: second-stage only if G21-B/C pass

Only after `C_t` becomes causally useful for action-token prediction:

```text
try action-expert-only router/MoE or stronger gated bridge
```

Reject whole-backbone MoE unless future data adds multi-embodiment conflict
evidence.

## 6. Final Decision

Current best explanation:

```text
The action plateau is primarily an action-readout representation problem.
The PICF belief/context path is present and bounded, but current continuous
action flow training does not make the action expert depend on that context.
```

What to stop doing:

```text
Do not rerun sampler-only, optimizer-only, action-weight-only, raw-overlap-only,
or bridge-strength-only experiments as the next fix.
```

What to do next:

```text
Implement PICF-side FAST/action-token representation supervision with causal
context-gain metrics, then run a 100-300 step gate.
```

This is the most mathematically direct repair because it targets the missing
factor exposed by G20:

```text
current failed factor: dA/dC
new supervised factor: dp_FAST/dC
```

## 7. References

```text
VLA Foundry:
  https://arxiv.org/abs/2604.19728

ABot-M0:
  https://arxiv.org/abs/2602.11236

PiKE:
  https://arxiv.org/abs/2502.06244

OpenVLA-OFT:
  https://arxiv.org/abs/2502.19645

Knowledge Insulation:
  https://arxiv.org/abs/2505.23705

AdaMoE-VLA:
  https://arxiv.org/abs/2510.14300
```
