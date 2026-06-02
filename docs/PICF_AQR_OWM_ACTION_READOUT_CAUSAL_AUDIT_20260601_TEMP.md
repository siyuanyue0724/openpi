# PICF-AQR-OWM Action Readout Causal Audit

Date: 2026-06-01

Status: temporary root-cause plan.  This document is intentionally narrow:
it is about why PICF structure/object metrics can improve while action loss
plateaus or rebounds.  It should be read together with:

```text
docs/PICF_AQR_OWM_ACTION_PLATFORM_ROOT_CAUSE_PROTOCOL_20260531.md
docs/PICF_AQR_OWM_ACTION_LOSS_WINDOW_AUDIT_20260530.md
docs/PICF_AQR_OWM_SLOT_VLA_PAPER_MATRIX_20260522.md
```

## 1. What Is Already Ruled Out

The following explanations are no longer primary hypotheses unless the code
path materially changes:

```text
raw inactive same-role overlap:
  reserve/context files can overlap without breaking active ownership.
  Use active/downstream overlap for production health.

blind SAM proposals:
  rejected and archived.  Do not revive as a default path.

sidecar deletion:
  no-sidecar did not improve action and worsened object structure.

direct dense token append:
  worsened action by shifting the PI0.5 token layout.

fixed64 live comparison:
  not a global metric.  Use exact-window/stratified windows only for causal
  checkpoint comparisons.

action scalar weight alone:
  action weight 4.0 did not decisively beat the maintained action weight 2.0
  route on the latest E14H check.

another raw-overlap penalty:
  repeated historical tests show action can fail while active/downstream
  overlap is healthy.
```

Practical consequence:

```text
Do not launch another "more diversity / less raw overlap / delete sidecar"
experiment as the next primary test.  It is not the current bottleneck.
```

## 2. Current Mathematical Model

The training path decomposes into:

```text
Z_t
  -> PICF measurement graph E_t
  -> posterior belief B_t
  -> bounded action-visible context C_t
  -> PI0.5 action interface I_theta(X_t, C_t)
  -> flow target loss L_action
```

The action gradient into PICF is:

```text
dL_action/dphi =
  dL_action/dy
  * dy/dI
  * dI/dC_t
  * dC_t/dB_t
  * dB_t/dphi
```

The observed pattern is:

```text
structure improves:
  active/downstream overlap can be healthy
  anchor/object losses can improve
  slot/prediction diagnostics can improve

action does not reliably improve:
  rolling action plateaus or rebounds
  exact-window action moves less than structure metrics
```

This points to a low-gain link in either:

```text
dI/dC_t:
  action path ignores or underuses PICF context;

dC_t/dB_t:
  context summarization loses action-relevant belief information;

dB_t/dphi:
  PICF belief is improving in its own metrics but not in an action-relevant
  subspace.
```

The most likely current bottleneck is `dI/dC_t`, not more slot collapse.

## 3. Current Code Dataflow

### 3.1 Context Extraction

`src/openpi/picf/policy.py::_action_context_tokens(...)`:

```text
conditioned_control.tokens
  -> optional query-token removal
  -> max-token bound
  -> rms/layer normalization
  -> scalar output gate
  -> optional stop-gradient
  -> context C_t
```

This is mathematically safe for dense foundation features because it does not
drop the base PI0.5 prefix.  It only provides an additional bounded context.

### 3.2 Prefix Fusion

`PicfPi05Policy._fuse_action_context_into_prefix(...)`:

```text
Q = rmsnorm(prefix)
K = rmsnorm(context)
A = softmax(Q K^T / sqrt(d))
prefix' = RMSCap(prefix + A context)
```

Property:

```text
PI prefix length is unchanged.
PI suffix positions are unchanged.
```

Known result:

```text
prefix fusion is safe but has not shown a decisive same-window action win.
```

### 3.3 Suffix-Side Adapter

`src/openpi/picf/paligemma/wrapper.py::_apply_action_context_adapter(...)`:

```text
suffix' = suffix
        + sigmoid(g) * RMSCap(
            W_o softmax(W_q LN(suffix) (W_k LN(context))^T / sqrt(d))
            W_v context
          )
```

Property:

```text
PI0.5 native prefix/suffix layout is preserved.
PICF context can affect the action suffix directly.
```

Known result:

```text
adapter metrics are live, but adapter-only/action-head-only/backbone-only
diagnostics have not produced a decisive action win yet.
```

Therefore the next test must be causal:

```text
Does changing C_t actually change the action prediction on identical
windows/noise/time?
```

## 4. Paper Comparison

The paper evidence supports the current direction, but also warns against
adding more unrelated object losses before proving the action readout path.

### 4.1 PI0.5 / OpenVLA-OFT Style Fine-Tuning

Relevant principle:

```text
Keep the action objective primary.  Add embodiment/perception interfaces in a
controlled way; do not let auxiliary reconstruction or object losses dominate
the action path.
```

PICF implication:

```text
Action loss must remain the final arbiter.  Object/slot losses are useful only
if they make action-visible belief better.
```

Sources:

```text
PI0.5: https://arxiv.org/abs/2504.16054
OpenVLA-OFT: https://arxiv.org/abs/2504.16054 is not the same work; verify the
exact OpenVLA-OFT source before citing details in a paper draft.
```

### 4.2 JEPA-VLA / VLA-JEPA / Latent World Models

Relevant principle:

```text
Latent predictive features can help VLA only when they condition action
without future leakage and without replacing the action interface.
```

PICF implication:

```text
Dense V-JEPA tokens should be preserved as bounded context or typed evidence,
not collapsed into a few hard slots.  Predictive losses must remain guarded
until matched, detached, and scale-normalized.
```

Sources:

```text
JEPA-VLA: https://arxiv.org/abs/2602.11832
VLA-JEPA: https://arxiv.org/abs/2602.10098
```

### 4.3 Flamingo / Gated Cross-Attention Family

Relevant principle:

```text
Inject external visual/context tokens through gated cross-attention rather than
changing the pretrained language/action token layout.
```

PICF implication:

```text
Direct append is the wrong interface.  Prefix fusion or suffix-side gated
cross-attention is the coherent action-readout route.
```

This supports the current suffix-adapter design, but it does not prove the
adapter is strong enough.  The causal sensitivity probe below is required.

### 4.4 Object Binding in Pretrained ViTs / IsSameObject Probes

Relevant principle:

```text
Object binding can exist as a subspace/probe property, but it is fragile and
not guaranteed to be exposed by raw cosine similarity or by action loss alone.
```

PICF implication:

```text
binding_signature_proj and support-weighted pairwise signatures are the right
kind of structural term.  The missing piece is an offline same-object probe
that measures whether the subspace is actually separating objects.
```

Source:

```text
Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
Search/open source page must be verified before paper-grade citation.
```

### 4.5 Object-Centric Slot / MetaSlot / QASA Style Methods

Relevant principle:

```text
Modern slot systems separate object files from background/context and often use
quality/competition gates instead of forcing every dense token into an active
slot.
```

PICF implication:

```text
The active/context/reserve split is mathematically consistent.  Background
tokens should remain dense context or reserve context, not be forced into
active object files.  Raw reserve overlap is not the primary failure metric.
```

### 4.6 Tactile / Contact-Rich VLA

Relevant principle:

```text
Tactile evidence should be contact-gated and bound to the manipulated object
owner, not treated as a permanent independent gripper object.
```

PICF implication:

```text
Removing permanent gripper-object ownership and binding tactile tokens to the
active contact owner is consistent with multimodal object-file design.
```

## 5. Remaining Root Hypotheses

### H1. Action Interface Is Insensitive To PICF Context

Mathematical signature:

```text
|| pi(X, C) - pi(X, 0) || is small
and
L_action(X, C) - L_action(X, 0) is small
on identical windows/noise/time.
```

This would explain:

```text
object metrics improve but action does not.
action weight changes move scalar loss only temporarily.
suffix adapter is live but not yet effective.
```

### H2. PICF Context Is Informative But Poorly Calibrated

Mathematical signature:

```text
normal context changes action, but shuffled/noisy context changes action nearly
as much as correct context.
```

This means the action path reads PICF, but not in an object-causal way.

### H3. Context Is Useful Only As Oracle/Sidecar But Not Learned Belief

Mathematical signature:

```text
oracle sidecar/contact-owner context improves same-window action, but learned
PICF context does not.
```

This points to `dC_t/dB_t` or `dB_t/dphi`, not action adapter.

### H4. Rolling Loss Is Hiding Sample-Difficulty Changes

Mathematical signature:

```text
rolling loss worsens, exact-window loss is flat or improves.
```

This is already partly confirmed historically.  It must remain a check for any
new action claim.

## 6. Next Experiments

These are ordered to identify cause.  Do not skip directly to another 30K run.

### Experiment A: Action-Context Causal Sensitivity Probe

Use the same checkpoint, same windows, same noise, and same diffusion time.
Evaluate without optimizer update:

```text
A0 normal context
A1 zero context
A2 shuffled context across batch/windows
A3 sign-flipped or Gaussianized context with matched RMS
A4 sidecar/contact-owner oracle context
A5 prefix-only no context
A6 suffix-adapter context if enabled
```

Record:

```text
loss_action_default_equiv
action_pos / action_rot / action_gripper
||pred_normal - pred_zero||_2
||pred_normal - pred_shuffle||_2
adapter gate / entropy / residual RMS
context token count and RMS
```

Decision gates:

```text
If A0 ~= A1 and prediction delta is tiny:
  PICF context is ignored.  Do not add object losses.

If A4 improves but A0 does not:
  learned belief/context quality is the bottleneck.

If A2/A3 hurt strongly while A0 helps:
  action path uses PICF correctly; then long-run optimization/sampling is the
  remaining issue.
```

### Experiment B: Gradient Follow-Through Audit

On one accepted batch, run a single backward pass and log:

```text
||dL_action/dC_t||
||dL_action/dprefix||
||dL_action/dsuffix_adapter||
||dL_action/dPICF_core||
||dL_action/dsemantic_action_backbone||
cosine(grad_action, grad_structure) for shared PICF params
```

Decision gates:

```text
If dL/dC_t is near zero:
  action cannot learn from PICF under this interface.

If dL/dC_t is healthy but dL/dPICF_core is tiny:
  detach/gate/context summarization blocks PICF learning.

If action and structure gradients have negative cosine:
  cotrain conflict is real; use PCGrad/GradNorm-style balancing or staged
  optimizer, not scalar trial-and-error.
```

### Experiment C: Stratified Same-Window Task Buckets

Build fixed windows by CALVIN task family:

```text
push block
lift/move block
open/close drawer
press button / switch
sliding door
mixed distractor windows
```

Compare checkpoints:

```text
4-22 ablation selected checkpoints
current maintained checkpoint
latest post-repair checkpoint
optional oracle-context variant
```

Decision gates:

```text
If current wins on object-like tasks but loses on buttons/switches:
  context/object routing is task-family biased.

If current loses uniformly:
  action transfer is global.

If current equals old fixed-window but rolling loss differs:
  sampler/window distribution caused the alarm.
```

### Experiment D: Minimal Training Branches After A/B/C

Only after the causal probes:

```text
D1 context ignored:
  train action-side adapter/backbone with stronger supervised action-context
  sensitivity, keep slot losses unchanged.

D2 oracle useful, learned context weak:
  improve belief-to-context readout, not action backbone first.

D3 gradient conflict:
  apply gradient surgery or two-timescale optimizer.  This is justified only
  if cosine conflict is measured.
```

## 7. What Not To Do Next

```text
Do not add another raw overlap/diversity loss.
Do not revive blind SAM.
Do not compare rolling action rows across different sampled windows.
Do not increase action weight again without same-window causal read.
Do not append dense tokens directly to PI0.5 prefix.
Do not claim the model is final until action-context sensitivity and
gradient follow-through are measured.
```

## 8. Concrete Next-Step Plan

1. Add or run a no-update action-context probe that supports context modes:

```text
normal, zero, shuffle, noise_rms, oracle_sidecar
```

2. Add a single-batch gradient audit mode to the same probe or to
`scripts/picf_fixed_window_action_probe.py`.

3. Run the probe on:

```text
latest maintained checkpoint
latest plateau checkpoint
one historical 4-22 checkpoint if available
```

4. Only after A/B/C, choose one of:

```text
interface repair:
  if action ignores context;

belief readout repair:
  if oracle context helps but learned context does not;

optimizer/gradient balancing:
  if action/structure gradients conflict;

continue long train:
  only if same-window action is improving and context sensitivity is healthy.
```

## 9. Current Recommendation

The next experiment should be a causal readout audit, not another long train.

Reason:

```text
Long trains have repeatedly mixed together three effects:
  window distribution;
  optimizer schedule discontinuity;
  true action-readout utility.

The fastest way to stop guessing is to measure whether PICF context changes the
action prediction on identical windows/noise/time.
```

Expected runtime:

```text
1-2 hours for the first exact-window sensitivity and gradient audit if the
checkpoint and data paths are already local on the cloud machine.
```

## 10. 2026-06-01 Deployment Record

Local code change:

```text
scripts/picf_fixed_window_action_probe.py
  added --action-context-probe-mode:
    normal
    zero
    token_roll
    sign_flip
    rms_noise

src/openpi/picf/policy.py
scripts/picf_core_train.py
  pass probe diagnostics into debug metrics:
    pi_context_probe_mode_id
    pi_context_probe_delta_rms_mean
    pi_context_probe_post_rms_mean
```

Remote directory:

```text
/root/openpi_probe_current_20260529
```

The earlier smoke attempt in `/root/openpi_posterior_vla_clean` was invalid
because that worktree did not contain the current action-context policy path.
The corrected probe is running from `/root/openpi_probe_current_20260529`.

Checkpoint:

```text
/mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000
```

Window set:

```text
/mnt/picf_exact_window_probes/action_context_causal_20260601/
  stratified12_windows.jsonl
```

Selection:

```text
2 windows per bucket from:
  block+grasp
  block+push
  block+slider
  drawer
  manipulator
  slider
```

Run root:

```text
/mnt/picf_exact_window_probes/action_context_causal_20260601/
  full12_ckpt11000/
```

Launched modes:

```text
GPU0:
  normal
  zero
  sign_flip

GPU1:
  token_roll
  rms_noise
```

Pass/fail read:

```text
If normal ~= zero/token_roll/sign_flip/rms_noise:
  action path is insensitive to PICF context.

If zero/sign_flip/rms_noise materially worsen loss:
  action path uses PICF context; then the plateau is more likely optimizer,
  sampling, or learned-belief quality rather than readout disconnection.

If token_roll is close to normal but zero/noise differ:
  action path uses global context magnitude but not object-token identity/order.

If sign_flip improves:
  context direction/calibration is wrong, not merely weak.
```

## 11. Dense-Token Follow-Through Audit

User concern checked on 2026-06-01:

```text
Could PICF be discarding most dense V-JEPA / visual tokens before PI0.5,
therefore breaking V-JEPA's world-model information and making action
training look like mislabeled data?
```

Code facts:

```text
src/openpi/picf/core/pipeline.py
  temporal_visual = PicfTemporalVisualSupportState(...)
  token_field.temporal_visual.tokens has one row per temporal dense V-JEPA
  grid cell, per configured view and time slice.

src/openpi/picf/core/pipeline.py
  _build_aqr_anchor_graph reads token_field.temporal_visual.tokens through
  aqr_temporal_visual_reader and produces vjepa_temporal_priors.

src/openpi/picf/core/pipeline.py
  _build_public_read_memory concatenates:
    token_field.fused_tokens
    token_field.visual_tokens
    token_field.temporal_visual.tokens
  and _build_task_readout reads this public memory.

src/openpi/picf/core/pipeline.py
  debug["owm_temporal_visual_tokens"] reports the actual temporal dense
  support count.
```

Therefore:

```text
PICF internal evidence path does not throw away temporal V-JEPA tokens before
AQR/posterior/task-readout.  Dense temporal tokens are available to AQR and to
task public readout.
```

However the PI0.5 action-visible path is intentionally much narrower:

```text
src/openpi/picf/core/pipeline.py
  _build_conditioned_control_state compresses posterior/task/graph evidence
  into conditioned_control.tokens and pi_prefix_tokens.

src/openpi/picf/policy.py
  _action_context_tokens takes conditioned_control.tokens, optionally drops
  query tokens, then keeps only:
    context[: action_context_tokens]

Current maintained run args:
  action_context_tokens = 24
  action_context_output_gate = 0.25
  action_context_stopgrad = true
  action_context_integration = prefix_fusion

src/openpi/picf/policy.py
  prefix_fusion keeps the PI prefix length fixed:
    A = softmax(norm(prefix) norm(context)^T / sqrt(d))
    prefix' = RMSCap(prefix + A context)
```

Mathematical read:

```text
Z_dense -> AQR/posterior/task readout is not discarded.

But:

Z_dense -> C_action is a lossy bottleneck:
  thousands of dense temporal tokens
  -> AQR/posterior/task/control states
  -> 4 PI prefix tokens plus at most 24 bounded context tokens.

And with action_context_stopgrad=true:
  L_action does not directly improve the PICF context encoder that produced
  those context tokens.
```

This is not a V-JEPA-structure destruction bug.  It is an action-readout
bottleneck hypothesis:

```text
Internal belief can improve while action loss plateaus if PI0.5 either ignores
the bounded PICF context or receives it in a representation that is too weak /
too compressed / too poorly calibrated to help action prediction.
```

## 12. Probe Snapshot: Context Sensitivity

Checkpoint:

```text
/mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000
```

Exact stratified window set:

```text
/mnt/picf_exact_window_probes/action_context_causal_20260601/
  stratified12_windows.jsonl
```

Final probe results:

```text
mode        action_default_equiv  context_delta_rms  note
normal      0.0266539294          0.000000           baseline context
token_roll  0.0266539294          0.129155           same as normal
zero        0.0266503993          0.250000           context removed
rms_noise   0.0266784410          0.353523           context replaced
sign_flip   0.0265320538          0.500000           context direction inverted
```

Interpretation so far:

```text
The current action head is not strongly using PICF action-context identity.
Rolling context tokens leaves the exact-window action loss unchanged; zeroing
or replacing the context does not materially worsen the exact-window loss.

This supports the readout-bottleneck hypothesis over the "dense V-JEPA was
destroyed before PICF" hypothesis.
```

Important nuance:

```text
token_roll alone is not a strong perturbation, because prefix_fusion is
attention over a set of context vectors and does not use context-token position
ids.  The decisive probes are zero and rms_noise; both are effectively neutral.
```

Final read:

```text
The current prefix_fusion action readout is effectively neutral to PICF
context on this exact-window probe.

This does not mean PICF internal dense evidence is discarded.  It means the
action generator is not causally using the bounded context bridge under this
checkpoint/config.

Next causal direction:
  Do not tune raw overlap or sidecar first.
  Audit/repair the action readout bridge itself:
    1. verify why existing suffix_cross_attention/action-adapter experiments
       did not improve action under corrected LR/optimizer conditions;
    2. only then run a matched exact-window probe where PICF context is injected
       through the action suffix adapter rather than prefix_fusion;
    3. keep dense V-JEPA/AQR internal routing intact.
```

## 13. Prefix-Bridge Causal Probe

Reason:

```text
The context probe only says the optional 24-token context bridge is neutral.
It does not prove whether the final 4 PICF PI-prefix tokens are used.
```

Patch added:

```text
scripts/picf_fixed_window_action_probe.py
  --action-prefix-probe-mode normal|zero|token_roll|sign_flip|rms_noise

src/openpi/picf/policy.py
scripts/picf_core_train.py
  forward debug metrics:
    pi_prefix_probe_mode_id
    pi_prefix_probe_delta_rms_mean
    pi_prefix_probe_post_rms_mean
```

Launched on A7:

```text
root:
  /mnt/picf_exact_window_probes/action_prefix_causal_20260601/
  full12_ckpt11000/

tmux:
  picf_prefix_probe_g0:
    zero
    sign_flip

  picf_prefix_probe_g1:
    rms_noise
    token_roll
```

Pass/fail read:

```text
If zero/sign_flip/rms_noise PI-prefix probes materially worsen action loss:
  PICF enters action primarily through a compressed 4-token prefix bottleneck.
  The problem is not total disconnection; it is bottleneck capacity/calibration.

If prefix probes are also neutral:
  the action generator is effectively ignoring the full PICF bridge.
  Then the next repair must be a true action-side gated cross-attention bridge
  or a stronger semantic/action readout adaptation, not another anchor loss.
```

Final result:

```text
baseline normal context/prefix action_default_equiv = 0.0266539294

prefix zero:
  action_default_equiv = 0.0265611961
  pi_prefix_probe_delta_rms_mean = 0.7000002861

prefix token_roll:
  action_default_equiv = 0.0266009292
  pi_prefix_probe_delta_rms_mean = 0.0000000000

prefix sign_flip:
  action_default_equiv = 0.0282719158
  pi_prefix_probe_delta_rms_mean = 1.4000005722

prefix rms_noise:
  action_default_equiv = 0.0299760835
  pi_prefix_probe_delta_rms_mean = 0.9889795035
```

Read:

```text
Zeroing all four PICF PI-prefix tokens does not worsen action loss on the exact
window set.  Therefore the useful action signal is not currently flowing
through the PICF PI-prefix.

Sign-flipping and RMS-matched random prefix worsen loss, so the channel can
perturb PI0.5; it is not dead code.  The learned PICF prefix is just not
providing positive action information at this checkpoint/config.

This is stronger than the context probe.  It shifts the leading root cause from
"dense context bottleneck" to "action generator is not trained to use PICF
belief as a predictive action condition".

token_roll has zero measured delta here, which implies the four final prefix
tokens are effectively permutation-degenerate under the current EMA/prefix
path.  Do not use token_roll as a decisive prefix perturbation.
```

## 14. Bridge-Capacity Probe

### 14.1 Why This Is The Next Experiment

The existing history already rejects several tempting but repetitive fixes:

```text
E7:
  suffix_cross_attention + action_adapter_only did not improve action.

E8:
  suffix_cross_attention + action_head_and_adapter did not improve rolling
  action over the local 100-step gate.

E9/E10:
  backbone_only suffix path was first invalid due LR/FSDP boundary, then
  repaired with semantic_lr_scale=0.35, but still did not beat prefix_fusion.

E11/E12:
  prefix_fusion vs prefix-only showed dense action context was not the sole
  root cause.

E13:
  LR continuity fixed part of the resume mismatch.  Exact-window replay showed
  source and resumed checkpoints were nearly equal on identical windows.
```

The new context/prefix perturbation results add one sharper fact:

```text
normal PICF context/prefix is not measurably better than zero context/prefix
on the exact 12-window probe, while random/sign-flipped prefix can perturb
PI0.5.
```

Therefore the next test must answer:

```text
Given frozen PICF belief B_t and the same exact windows W, can the action-side
bridge/readout learn to use C(B_t) at all?
```

This is different from E7/E8 because it is a fixed-window capacity probe, not
a rolling train-stream diagnostic.  It removes sampler drift and structure-loss
gradient conflict from the question.

### 14.2 Mathematical Contract

Freeze the belief producer:

```text
phi := PICF core / typed evidence / posterior parameters
stopgrad C_t = C_phi(Z_{\le t})
```

Train only the action-side interface:

```text
psi := action context adapter, wrapper-local action heads, PI action readout
```

Minimize on fixed windows:

```math
\min_\psi
  \frac{1}{|W|}
  \sum_{w \in W}
  L_\text{action}\left(
    \pi_\psi(X_w, \operatorname{sg}[C_\phi(Z_w)]),
    a_w
  \right)
```

Decision:

```text
If action_default_equiv drops quickly:
  C_phi contains action-useful information.
  The production issue is training schedule/interface strength, not object
  evidence itself.

If action_default_equiv cannot overfit the fixed windows:
  C_phi is not sufficiently action-informative or the chosen adapter topology
  lacks capacity.  Another long cotrain or raw-overlap penalty is unjustified.
```

### 14.3 Implementation

New script:

```text
scripts/picf_action_bridge_capacity_probe.py
```

Key properties:

```text
fixed exact windows only;
fresh optimizer by design;
picf_trainable_scope=policy_only by default;
action_context_stopgrad=true by default;
semantic_trainable_scope=action_head_and_adapter by default;
supports prefix_fusion and suffix_cross_attention;
logs loss_action_default_equiv plus structural metrics every N steps;
records trainable parameter sample and optimizer groups;
writes JSONL + summary JSON.
```

This is a causal capacity test, not a final training recipe.

### 14.4 Local Validation

```text
python -m py_compile \
  scripts/picf_action_bridge_capacity_probe.py \
  scripts/picf_fixed_window_action_probe.py \
  scripts/picf_core_train.py \
  src/openpi/picf/policy.py

git diff --check \
  scripts/picf_action_bridge_capacity_probe.py \
  scripts/picf_fixed_window_action_probe.py \
  scripts/picf_core_train.py \
  src/openpi/picf/policy.py \
  docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md
```

Result:

```text
PASS.
```

Local runtime note:

```text
The local machine lacks torch, so runtime help/import must be validated on the
cloud training environment.  This is expected for this workspace and does not
invalidate py_compile/diff checks.
```

### 14.5 First Cloud Experiment

Use the latest exact-window probe checkpoint and window set:

```text
checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000

windows:
  /mnt/picf_exact_window_probes/action_context_causal_20260601/
  stratified12_windows.jsonl
```

Run two matched bridge-capacity probes:

```text
GPU0:
  suffix_cross_attention
  semantic_trainable_scope=action_head_and_adapter
  picf_trainable_scope=policy_only

GPU1:
  prefix_fusion
  semantic_trainable_scope=action_head_and_adapter
  picf_trainable_scope=policy_only
```

Why both:

```text
suffix_cross_attention follows Flamingo-style gated action readout;
prefix_fusion is the maintained safer PI0.5 layout-preserving interface.

If only one overfits the exact windows, choose that topology for production
repair.  If neither overfits, the belief/context itself is not action-ready.
```

### 14.6 Result

Run root:

```text
/mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12/
```

Both runs completed 200 steps with finite gradients.

Suffix-side gated adapter:

```text
first12 action_default_equiv = 0.034555
last12  action_default_equiv = 0.036517
first24 action_default_equiv = 0.034356
last24  action_default_equiv = 0.035758
first50 action_default_equiv = 0.036131
last50  action_default_equiv = 0.031151
per-window first->last: 6 improved, 6 worsened
```

Prefix fusion:

```text
first12 action_default_equiv = 0.033929
last12  action_default_equiv = 0.037542
first24 action_default_equiv = 0.033536
last24  action_default_equiv = 0.041843
first50 action_default_equiv = 0.034578
last50  action_default_equiv = 0.033912
per-window first->last: 5 improved, 7 worsened
```

Task-family pattern:

```text
improves:
  push red block
  slide blue block left
  move to drawer/store object
  push switch down

worsens:
  move up switch
  slide door left
  slide door right
  close drawer
```

Conclusion:

```text
Training only the wrapper-local action head/adapter is not sufficient.  It can
fit some object-manipulation windows but fails or worsens other task families.

This rejects:
  "just make the bridge trainable"
  "just use suffix gated cross-attention"
  "prefix_fusion is the only bottleneck"

It does not prove PICF belief is useless.  The frozen-PICF capacity probe keeps
C_phi fixed and, with action_context_stopgrad=true, cannot reshape the belief
subspace toward action.
```

### 14.7 Next Root-Cause Probe

The next experiment must test the missing gradient path:

```math
\frac{\partial L_\text{action}}{\partial \phi}
  =
  \frac{\partial L_\text{action}}{\partial C_t}
  \frac{\partial C_t}{\partial B_t}
  \frac{\partial B_t}{\partial \phi}
```

Run a fixed-window action-only micro-cotrain:

```text
picf_trainable_scope=all
semantic_trainable_scope=action_head_and_adapter
action_context_stopgrad=false
picf_core_lr_scale explicitly set to a non-negligible value
same 12 exact windows
same checkpoint /11000
```

Decision:

```text
If this drops all 12 windows:
  the root cause is over-isolation of PICF from action
  (stopgrad + very low PICF LR + action bridge trained only downstream).

If this still cannot drop:
  current belief/context does not contain enough action-predictive information
  for the PI0.5 action interface.  The fix must be belief readout redesign or
  task-family-specific evidence, not optimizer tuning.
```

### 14.8 Action-Gradient-To-PICF Probe Launch

Run root:

```text
/mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_coregrad/
```

Common contract:

```text
checkpoint = .../picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000
windows    = .../action_context_causal_20260601/stratified12_windows.jsonl
steps      = 200
lr         = 5e-5
picf_core_lr_scale = 0.2
effective PICF core lr = 1e-5
picf_trainable_scope = all
semantic_trainable_scope = action_head_and_adapter
action_context_stopgrad = false
```

Matched branches:

```text
GPU0 suffix_cross_attention
GPU1 prefix_fusion
```

Why this is not a repeated experiment:

```text
The previous bridge-capacity probe froze PICF core and detached C_t.  It tested
action readout capacity only.

This probe allows:
  dL_action/dC_t
  dC_t/dB_t
  dB_t/dphi

It directly tests whether the current maintained training recipe has
over-isolated PICF from action.
```

### 14.9 Dense-Token Follow-Through Check

This audit also checked the suspected failure mode:

```text
"PICF/slot routing may be throwing away most V-JEPA dense tokens, making the
action model behave as if data were mislabeled."
```

Code facts:

```text
src/openpi/picf/core/pipeline.py
  _build_aqr_anchor_graph(...)
    token_field.temporal_visual.tokens
      -> aqr_temporal_visual_reader
      -> vjepa_temporal_priors
      -> graph.vjepa_temporal_priors
      -> conditioned control / posterior diagnostics

logs from the current checkpoint/probe:
  owm_temporal_visual_tokens = 3456
  owm_tracklet_tokens        = 64
  owm_proposal_tokens        = 1 on the sampled sidecar window
```

Therefore the dense V-JEPA evidence is not deleted inside the PICF belief
router.  The real compression point is later:

```text
conditioned_control.tokens
  -> PicfPi05Policy._action_context_tokens(...)
  -> at most action_context_tokens = 24
  -> RMS-normalized and gated
  -> either prefix_fusion or suffix_cross_attention
```

Mathematical consequence:

```text
E_t can contain dense world-model evidence,
but the action path only receives a bounded low-rank summary C_t.
```

This explains why structure/object metrics can improve while action stays flat:

```text
PICF internal belief may improve:
  dB_t/dphi is live;

but PI0.5 action may remain insensitive:
  dI/dC_t is low-gain or C_t is not action-shaped.
```

Decision:

```text
Do not frame the next fix as "restore discarded V-JEPA tokens".
The next question is whether action gradients can shape the PICF belief summary
into an action-usable subspace.
```

### 14.10 Coregrad Early Status

The action-gradient-to-PICF probe has started and is not a duplicate of E7-E13.
At launch it reports:

```text
optimizer groups:
  semantic_backbone / action-side adapter-head subset:
    lr = 5e-5
    num_params = 8,457,249

  picf_core:
    lr = 1e-5
    num_params = 452,374,247

action_context_stopgrad = false
picf_trainable_scope    = all
```

First visible rows are not yet sufficient for a decision:

```text
suffix branch:
  step 1  action = 0.046801
  step 20 action = 0.009196
  step 30 action = 0.011702
  recent at step 30 = 0.042070

prefix branch:
  step 1  action = 0.043877
  step 20 action = 0.007243
  step 30 action = 0.011572
  recent at step 30 = 0.040592
```

Interpretation:

```text
The low single-window rows prove the optimizer is live, but the recent average
has not yet shown stable descent.  The decision gate remains step 100/200.
```

Step-100 read:

```text
suffix:
  step 80 recent = 0.028161
  step 90 recent = 0.023960
  step100 recent = 0.033206

prefix:
  step 80 recent = 0.027668
  step 90 recent = 0.024116
  step100 recent = 0.033679
```

Hard-window cause in the 12-window cycle:

```text
step70:
  move up the switch
  suffix action = 0.085010
  prefix action = 0.086742

step100:
  push the pink block towards the right
  suffix action = 0.020844
  prefix action = 0.019903
```

Decision:

```text
Do not stop at step100 as success.
The probe is live, but the recent mean is still window-cycle-sensitive.
Wait for step200 so train.jsonl contains every step and per-window first/last
cycles can be compared directly.
```

### 14.11 Probe-Rigor Patch: Deterministic Flow Eval

Important correction:

```text
Train-step action loss is stochastic because PI0/PaliGemma flow matching samples
diffusion noise and time inside compute_action_flow_loss(...).

Therefore a per-window first/last train-step comparison is useful but not a
strict same-objective comparison unless noise/time are controlled.
```

The probe script now supports deterministic before/after eval:

```text
scripts/picf_action_bridge_capacity_probe.py
  --eval-before-after
  --deterministic-eval-seed <int>
  --eval-every <N>
```

Implementation:

```text
During eval only, wrap semantic_encoder.compute_action_flow_loss with a
deterministic RNG context:

  call k uses seed = base_seed + k

This preserves the original action-flow API while making every eval pass compare
the same sampled diffusion objective on the same exact windows.
```

Validation:

```text
python -m py_compile scripts/picf_action_bridge_capacity_probe.py
git diff --check scripts/picf_action_bridge_capacity_probe.py \
  docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md

Result: PASS.
```

## 15. Literature-Backed Root-Cause Position

This section is an anti-duplication guard.  It records which paper-backed
mechanism maps to which PICF failure hypothesis.

### 15.1 JEPA-VLA / VLA-JEPA

Claim supported by the line of work:

```text
Dense predictive video features are useful when they are injected into the
action policy as an action-conditioned latent signal, not when they merely
exist somewhere upstream.
```

PICF consequence:

```text
The fact that owm_temporal_visual_tokens ~= 3456 proves dense V-JEPA evidence
exists inside PICF.  It does not prove PI0.5 action consumes it.  The causal
quantity is dL_action/dC_t and the prediction delta under context ablation.
```

Therefore:

```text
If context/prefix perturbation is neutral, the next repair is not "more V-JEPA
tokens"; it is action-readout gain/alignment.
```

### 15.2 Flamingo-Style Gated Cross-Attention

Claim supported by the line of work:

```text
External perceptual memories should be introduced through a gated adapter that
preserves the pretrained language/action token layout.
```

PICF consequence:

```text
Direct append was correctly rejected.
prefix_fusion is safe but low-capacity.
suffix_cross_attention is architecturally coherent but must be tested with
same-window before/after eval, because "wired" is not the same as "useful".
```

Therefore:

```text
If frozen-PICF suffix adapter fails, do not conclude gated attention is wrong.
Conclude that action-side adapter/head alone cannot remap the current frozen
belief context.  Then test action-gradient-to-PICF.
```

### 15.3 Object Binding / IsSameObject

Claim supported by the line of work:

```text
Object identity can be present as a probeable subspace, but raw token overlap
or raw cosine is not a reliable identity contract.
```

PICF consequence:

```text
binding_signature_proj and support-weighted binding are coherent.
raw inactive overlap is not the production failure metric.
offline IsSameObject probe remains useful, but it is not the immediate action
loss root-cause experiment.
```

Therefore:

```text
Do not run another raw-overlap penalty before measuring action-gradient
follow-through.
```

### 15.4 Modern Slot / Object-File Methods

Claim supported by the line of work:

```text
Object files should compete for foreground/object evidence while background
stays as context/reserve.  Forcing every dense token into an active slot creates
crowding and harms action-conditioned readout.
```

PICF consequence:

```text
The active/context/reserve design is not itself the bug.  The unresolved issue
is whether the action path receives the right object-conditioned summary, not
whether every background token becomes a posterior object.
```

Therefore:

```text
The correct next test is not "all tokens into slots"; it is "does the bounded
object/context summary causally reduce action loss on identical windows?"
```

## 16. Non-Repeated Experiment Ladder

The current evidence requires the following ladder.  Stop as soon as a decision
gate is hit; do not keep launching variants that test an already rejected cause.

### E15: Coregrad Completion

Question:

```text
When action gradient is allowed through C_t into PICF, can exact-window action
loss descend across task families?
```

Current run:

```text
root:
  /mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_coregrad/

branches:
  suffix_cross_attention
  prefix_fusion

contract:
  picf_trainable_scope = all
  semantic_trainable_scope = action_head_and_adapter
  action_context_stopgrad = false
  picf_core_lr_scale = 0.2
  steps = 200
```

Decision:

```text
If last50 < first50 and most per-window deltas improve:
  the root cause is over-isolated PICF/action coupling.
  Next run deterministic eval and then productionize no-stopgrad/two-timescale
  action bridge.

If first50/last50 improves only by task-family and worsens elsewhere:
  PICF context is task-biased.  Next run action-only control and then
  task-family bridge readout audit.

If no stable improvement:
  do action-only capacity control before touching slot losses.
```

### E16: Deterministic Same-Window Eval

Question:

```text
Is the apparent improvement real under identical diffusion noise/time?
```

Run only after E15:

```text
scripts/picf_action_bridge_capacity_probe.py
  --eval-before-after
  --deterministic-eval-seed 20260601
  --eval-every 100
```

Decision:

```text
If deterministic eval improves:
  accept the training signal.

If train rows improve but deterministic eval does not:
  the apparent descent is flow-sampling noise; do not trust it.
```

### E17: PI0.5 Action-Only Capacity Control

Question:

```text
Can the same action path overfit the exact windows without PICF conditioning?
```

Run if E15/E16 do not close the case:

```text
picf_mode = ablated
semantic_trainable_scope = action_head_and_adapter
same checkpoint
same 12 exact windows
same deterministic eval
```

Decision:

```text
If action-only improves but PICF-coregrad does not:
  PICF conditioning is the bottleneck/noise.

If action-only also fails:
  the bottleneck is action-flow/head capacity, LR, or exact-window objective,
  not PICF slot structure.
```

### E18: Gradient Conflict Audit

Question:

```text
Are action and structure losses fighting on shared PICF parameters?
```

Required metrics:

```text
cos(grad_action, grad_structure) on:
  conditioned_control / pi_prefix reader
  AQR readers
  binding signature projection
  posterior updater

norm ratio:
  ||grad_action|| / ||grad_structure||
```

Decision:

```text
If cosine is strongly negative:
  use PCGrad/GradNorm-style balancing or staged loss release.

If cosine is near zero:
  action is disconnected/low-gain, not conflicting.

If cosine is positive but action flat:
  bridge capacity/calibration remains the issue.
```

## 17. Current Working Rule

Until E15-E17 close:

```text
No new long 30K run should be used as root-cause evidence.
No new raw-overlap penalty should be launched.
No sidecar/SAM deletion should be repeated.
No fixed64 headline comparison should be used.
```

The only acceptable next experiments are same-window causal controls that can
separate:

```text
1. action path capacity;
2. PICF context usefulness;
3. action-gradient-to-PICF follow-through;
4. train-stream/window non-stationarity.
```

## 18. E15 Final Result: Coregrad Is Live But Not Sufficient

Run:

```text
/mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_coregrad/
```

Suffix branch:

```text
rows = 200
first12 = 0.034137
last12  = 0.036429
first24 = 0.034147
last24  = 0.036274
first50 = 0.035294
last50  = 0.031605
first100 = 0.031902
last100  = 0.031540
per-window: 5 improved, 7 worsened
```

Prefix branch:

```text
rows = 200
first12 = 0.033832
last12  = 0.036361
first24 = 0.033536
last24  = 0.035849
first50 = 0.034123
last50  = 0.030829
first100 = 0.031352
last100  = 0.030991
per-window: 5 improved, 7 worsened
```

Task-level read:

```text
improved:
  lift blue block
  push red block
  move/store object
  push switch down

worsened:
  push pink block
  slide red block
  slide blue block
  close drawer
  move up switch
  slide door left
  slide door right
```

Interpretation:

```text
Allowing action gradient into PICF is real and gives a small aggregate gain
over the 100-step window.  It does not give uniform cross-task descent.

This rejects the strongest simple hypothesis:
  "the only problem is stopgrad / over-isolation."

It also rejects using this run as permission for a blind 30K restart:
  the hard task families get worse even while aggregate first50/last50 improves.
```

Current root-cause status:

```text
dL_action/dPICF exists.
The bridge is not dead.
But the learned PICF context is not yet a task-family-stable action condition.
```

Mandatory next step:

```text
E17 action-only capacity control with deterministic eval.
```

Reason:

```text
If action-only can overfit the same 12 windows, the remaining problem is PICF
conditioning/noise.

If action-only cannot overfit the same 12 windows, the bottleneck is action
flow/head capacity or the exact-window objective, not PICF.
```

## 19. E16/E17 Deterministic Control Launch

Launched on A7 after E15 completed:

```text
root:
  /mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_det_controls/

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000

windows:
  /mnt/picf_exact_window_probes/action_context_causal_20260601/
  stratified12_windows.jsonl
```

GPU0:

```text
tmux session:
  picf_coregrad_prefix_det

branch:
  E16 deterministic coregrad repeat

config:
  picf_trainable_scope = all
  semantic_trainable_scope = action_head_and_adapter
  action_context_integration = prefix_fusion
  action_context_tokens = 24
  action_context_stopgrad = false
  picf_core_lr_scale = 0.2
  eval_before_after = true
  eval_every = 100
  deterministic_eval_seed = 20260601
```

GPU1:

```text
tmux session:
  picf_actiononly_det

branch:
  E17 PI0.5 action-only capacity control

config:
  picf_mode = ablated
  picf_trainable_scope = policy_only
  semantic_trainable_scope = action_head_and_adapter
  action_context_tokens = 0
  eval_before_after = true
  eval_every = 100
  deterministic_eval_seed = 20260601
```

Tail:

```bash
tail -f /mnt/picf_exact_window_probes/action_bridge_capacity_20260601/ckpt11000_exact12_det_controls/coregrad_prefix_det/run.log
tail -f /mnt/picf_exact_window_probes/action_bridge_capacity_20260601/ckpt11000_exact12_det_controls/actiononly_det/run.log
```

Decision read:

```text
If E16 deterministic improves but E17 does not:
  PICF action coupling is useful and should be productionized.

If E17 improves much more than E16:
  PICF context is a noisy or low-gain condition; the next repair is context
  calibration/readout, not more object losses.

If neither improves:
  the action path or exact-window action objective is the bottleneck.
```

### 19.1 E17 Launch Bug And Fix

First E17 launch failed before training:

```text
RuntimeError: Unexpected key(s) in state_dict:
  core.point_feature_extractor.*
```

Root cause:

```text
The probe applied picf_mode=ablated before model construction.  That can build a
trainer whose core lacks checkpoint submodules that are irrelevant for
action-only forward but still present in the full checkpoint.
```

Fix:

```text
For --picf-mode ablated:
  build the checkpoint-native full trainer;
  load the checkpoint;
  then set trainer.policy.picf_enabled = False and trainer.picf_mode = ablated.
```

This keeps E17 as a clean action-only control:

```text
forward path:
  _PicfWindowTrainer._forward_action_only_window
  -> PicfPi05Policy._pi05_only_train_transition
  -> semantic_encoder.compute_action_flow_loss(extra_prefix_tokens=None)
```

It does not change production training.

Retry launched:

```text
/mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_det_controls/actiononly_det_retry/

tmux:
  picf_actiononly_det_retry
```

Consequence:

```text
The currently running coregrad probe remains valid for "is the optimizer/gradient
path live?".

Before making a final "action significantly descends" claim, rerun or follow up
with deterministic eval enabled.
```

### 14.12 Next Control If Coregrad Fails

If the coregrad probe still cannot show stable step200 improvement, the next
non-duplicative control is:

```text
PI0.5 action-only capacity control on the same exact windows.
```

Reason:

```text
Current probe:
  trains action-side adapter/head plus PICF core summary C_phi.

Action-only control:
  sets picf_mode=ablated;
  keeps the same semantic/action path;
  removes PICF prefixes/context from the action objective.
```

Decision rule:

```text
If action-only overfits exact12 quickly:
  the action head and flow objective have enough local capacity.
  The bottleneck is PICF action conditioning: C_t is neutral/noisy or not
  aligned with task-family-general action.

If action-only also cannot overfit exact12:
  the bottleneck is not PICF; inspect action-path LR, flow stochasticity,
  checkpoint/action-head state, or the exact-window difficulty contract.
```

Script support added:

```text
scripts/picf_action_bridge_capacity_probe.py
  --picf-mode ablated
```

Validation:

```text
python -m py_compile scripts/picf_action_bridge_capacity_probe.py
git diff --check scripts/picf_action_bridge_capacity_probe.py

Result: PASS.
```

### 19.2 E17 Retry Result: Action-Only Capacity Is Real

Retry run:

```text
root:
  /mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_det_controls/actiononly_det_retry/

tmux:
  picf_actiononly_det_retry
```

Final deterministic eval:

```text
step 0:
  loss_action_default_equiv.mean = 0.0339394346

step 100:
  loss_action_default_equiv.mean = 0.0294068087

step 200:
  loss_action_default_equiv.mean = 0.0240519585
  repeated final eval = 0.0259943306
```

Train-row summary:

```text
first12 = 0.036580
last12  = 0.029489

first24 = 0.035654
last24  = 0.029072

first50 = 0.034910
last50  = 0.028935

first100 = 0.032614
last100  = 0.029231

per-window first->last:
  improved = 7
  worsened = 4
  flat     = 1
```

Interpretation:

```text
The action-only PI0.5 branch can reduce the same exact-window objective after
the checkpoint-native full model is loaded and PICF is disabled after load.

This rejects:
  "the local action head/adapter has no capacity";
  "the exact-window action objective is intrinsically impossible";
  "the deterministic eval machinery is broken".

It supports:
  PICF conditioning/readout is currently neutral or noisy relative to the
  native action branch on the same windows.
```

Important nuance:

```text
E17 does not prove PICF should be deleted.  It proves the action branch can
descend when PICF conditioning is removed.  The next causal question is whether
E16 with PICF gradient enabled can match this deterministic descent.  If not,
the repair target is the PICF-to-action bridge and context calibration, not
another slot-overlap loss.
```

## 20. Local Integrity Check

Local checks on 2026-06-01:

```text
python -m py_compile \
  scripts/picf_action_bridge_capacity_probe.py \
  scripts/picf_fixed_window_action_probe.py \
  scripts/picf_core_train.py \
  src/openpi/picf/policy.py \
  src/openpi/picf/core/pipeline.py \
  src/openpi/picf/paligemma/wrapper.py

python scripts/picf_action_visible_reserve_gate_audit.py

git diff --check \
  scripts/picf_action_bridge_capacity_probe.py \
  scripts/picf_fixed_window_action_probe.py \
  scripts/picf_core_train.py \
  src/openpi/picf/policy.py \
  docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md
```

Result:

```text
PASS.
```

Follow-through result:

```text
Dense temporal V-JEPA tokens are present inside PICF:
  owm_temporal_visual_tokens = 3456

Tracklet/proposal sidecar evidence is present on sampled windows:
  owm_tracklet_tokens = 62
  owm_proposal_tokens = 1

Action-visible PICF remains bounded:
  pi_action_condition_token_count = 4
  pi_context_token_count = 24
```

Therefore the current failure should not be described as dense-token deletion.
The more precise current hypothesis is:

```text
PICF keeps dense evidence internally, but the bounded PICF action bridge is not
yet a consistently positive action condition.
```

## 21. E18 Suffix Deterministic Coregrad Control

Reason:

```text
E16 tests deterministic coregrad with prefix_fusion.
E17 tests action-only and shows the local action objective can descend.

The missing non-duplicative control is deterministic suffix_cross_attention
with action gradient into PICF.  This directly tests the paper-supported
gated cross-attention route under identical windows/noise/time.
```

## 22. E16 Deterministic Prefix Coregrad Result

Run:

```text
/mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_det_controls/coregrad_prefix_det/
```

Deterministic eval:

```text
step 0:
  loss_action_default_equiv.mean = 0.0266539294

step 100:
  loss_action_default_equiv.mean = 0.0280073154

step 200:
  loss_action_default_equiv.mean = 0.0317106721
  repeated final eval = 0.0351747532
```

Train-row summary:

```text
first12 = 0.037077
last12  = 0.039430

first24 = 0.035955
last24  = 0.036508

first50 = 0.030255
last50  = 0.033731

first100 = 0.030749
last100  = 0.031499
```

Interpretation:

```text
Prefix-fusion coregrad does not pass deterministic same-window eval.
It worsens the exact objective while E17 action-only improves it.

This is stronger than a rolling-loss observation:
  same checkpoint;
  same exact windows;
  deterministic diffusion eval;
  same action-side trainable scope;
  only PICF conditioning/core path differs.
```

Current root-cause read:

```text
The native PI0.5 action path can learn the exact windows.
The current PICF prefix-fusion condition is a negative/neutral action condition.
The problem is not dense V-JEPA deletion and not action-head incapacity.
The remaining open question is whether suffix gated cross-attention can turn
PICF context into a positive condition.
```

## 23. Current Causal Conclusion Before E18 Completion

The evidence now separates three questions:

```text
Q1. Are dense V-JEPA / typed tokens deleted before PICF?
  No.  They are present inside token_field and AQR/public/task readers.

Q2. Can the action branch learn this exact-window objective?
  Yes.  E17 action-only deterministic eval improves from 0.03394 to about
  0.024-0.026 by step200.

Q3. Does current PICF prefix_fusion improve action when action gradient reaches
    PICF?
  No.  E16 deterministic eval worsens from 0.02665 to about 0.032-0.035.
```

Therefore the current root cause is not:

```text
raw overlap;
SAM/sidecar presence;
action-head incapacity;
dense-token deletion;
diffusion eval randomness.
```

The current root cause is:

```text
PICF belief/context is not yet aligned as a positive action condition for
PI0.5.  The bridge can perturb action, but the learned condition is not
causally helpful on the exact-window test.
```

This is the mathematical form:

```math
\Delta_\text{PI-only}
  =
  L_\text{action}^{200}(\pi_\psi(X))
  -
  L_\text{action}^{0}(\pi_\psi(X))
  < 0

\Delta_\text{PICF-prefix}
  =
  L_\text{action}^{200}(\pi_{\psi,\phi}(X, C_\phi))
  -
  L_\text{action}^{0}(\pi_{\psi,\phi}(X, C_\phi))
  > 0
```

So the failure is a conditional readout problem:

```math
I(C_\phi; A \mid X) \text{ is not positive in the current action interface,}
```

even though:

```math
C_\phi = f_\phi(Z_\text{dense}, Z_\text{point}, Z_\text{tactile}, ...)
```

is internally nonempty and structurally constrained.

Practical implication:

```text
If E18 suffix also fails, the next trainable recipe should stop feeding PICF
as action condition until a calibration objective proves it is positive.

The safe route would be:
  keep PICF belief/slot losses as auxiliary or diagnostic;
  train PI0.5 action path without PICF action-prefix/context injection;
  separately train a calibrated action-context adapter on exact-window gates;
  only enable PICF-to-action when zero/shuffle/noise probes show positive
  causal contribution.
```

Launch:

```text
root:
  /mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_det_controls/coregrad_suffix_det/

tmux:
  picf_coregrad_suffix_det

GPU:
  1

config:
  checkpoint = .../picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000
  windows = .../action_context_causal_20260601/stratified12_windows.jsonl
  steps = 200
  lr = 5e-5
  picf_core_lr_scale = 0.2
  picf_trainable_scope = all
  semantic_trainable_scope = action_head_and_adapter
  action_context_integration = suffix_cross_attention
  action_context_tokens = 24
  action_context_output_gate = 0.25
  action_context_stopgrad = false
  eval_before_after = true
  eval_every = 100
  deterministic_eval_seed = 20260601
```

Launch hygiene:

```text
First launch failed before import because tmux's non-login shell did not have
uv on PATH:
  bash: line 1: uv: command not found

This is an environment launch error, not an algorithm result.

Relaunch uses:
  /root/.local/bin/uv
```

Second launch note:

```text
/root/.local/bin/uv run still attempted dependency resolution in this worktree
and failed building av==14.4.0.  The already-running probes use the system
Python with installed CUDA/PyTorch:

  /usr/local/bin/python3.12
  torch 2.4.1+cu124

Final E18 relaunch uses /usr/local/bin/python3.12 directly.
```

Third launch note:

```text
The probe script argument is --window-jsonl, not --windows-jsonl.
The invalid plural form exited before source/model construction.

Final active relaunch:
  /usr/local/bin/python3.12 scripts/picf_action_bridge_capacity_probe.py
  --window-jsonl ...
  --device cuda
```

Decision:

```text
If suffix deterministic improves while prefix deterministic does not:
  the correct repair is action-side gated cross-attention, not prefix fusion.

If suffix also fails while action-only succeeds:
  PICF belief/context is currently a noisy or neutral condition for action.
  Next repair must be context calibration / oracle-vs-learned belief readout,
  not more object-loss tuning.
```

Early status:

```text
model_ready:
  elapsed_s = 145.552
  trainable_numel = 460,831,496
  semantic/action-side lr = 5e-5
  picf_core lr = 1e-5

deterministic eval step0:
  loss_action_default_equiv.mean = 0.0287604353

first train row:
  step1 action_default_equiv = 0.0352236554
```

Important read:

```text
The suffix adapter baseline is already worse than prefix_fusion step0
(0.02876 vs 0.02665), because it changes the action-side interface from the
checkpoint's maintained prefix path into the gated suffix adapter path.

Therefore E18 is not judged by absolute step0 comparison to E16.  It is judged
by whether suffix_cross_attention can learn down from its own step0 under
deterministic eval at step100/200.
```

## 24. E18 Step100 Deterministic Check

Date: 2026-06-01 CST

E18 reached the first deterministic checkpoint:

```text
step0 deterministic action:
  loss_action_default_equiv.mean = 0.0287604353
  loss_total_minus_action.mean   = 0.1848280790
  pi_context_adapter_gate.mean   = 0.1192029193
  pi_context_adapter_attention_entropy_mean.mean = 3.0992846290
  pi_context_adapter_residual_rms_mean.mean      = 0.2622944663

step100 deterministic action:
  loss_action_default_equiv.mean = 0.0286838954
  loss_total_minus_action.mean   = 0.2143019537
  pi_context_adapter_gate.mean   = 0.1191076487
  pi_context_adapter_attention_entropy_mean.mean = 2.4498629173
  pi_context_adapter_residual_rms_mean.mean      = 0.5194819892
```

Interpretation:

```text
The suffix adapter has not failed catastrophically at step100, but it has also
not shown a meaningful action transfer win.

Action improves by only about 0.0000765 absolute, or about 0.27% relative to
its own suffix step0 baseline.

At the same time, non-action/structure burden rises by about 0.0295 absolute,
attention entropy drops, and adapter residual RMS roughly doubles.  So the
adapter is becoming more selective/stronger, but this stronger context
injection is not yet causally useful for action.
```

Decision:

```text
Continue E18 to step200 before final verdict.

If step200 still does not beat step0 by a clear margin, classify suffix
cross-attention as currently uncalibrated rather than production-ready.  The
next experiment should then be:

  PICF computed and auxiliary losses available,
  but PICF action condition disabled,
  plus a separate calibration gate/probe before re-enabling PICF-to-action.

This is not an argument to delete dense tokens, sidecar, or slot structure.
It is an action-readout causal-alignment issue.
```

Implementation note for the next control:

```text
Do not use action_prefix_output_gate=0 as the "no PICF action condition" test.
That would still pass extra zero prefix tokens and can shift the action layout.

Use the explicit causal switch instead:
  picf_action_condition_enabled = false

Expected graph:
  observation -> PICF observe/finalize -> PICF auxiliary/diagnostic losses
  observation -> PI0.5 action loss with no extra PICF prefix/context tokens

This separates:
  "PICF belief can be computed and trained"
from:
  "PICF belief is currently useful as an action condition"
```

## 26. E18 Step200 Verdict

E18 reached step200:

```text
step0 deterministic:
  action = 0.0287604353
  total_minus_action = 0.1848280790

step100 deterministic:
  action = 0.0286838954
  total_minus_action = 0.2143019537

step200 deterministic:
  action = 0.0322181598
  total_minus_action = 0.1746107303
```

Verdict:

```text
suffix_cross_attention is not a production fix in the current checkpoint/state.

It briefly produced a statistically tiny action improvement at step100, then
became worse than its own step0 by step200.  This is exactly the failure mode
we needed to test: the action-side gated adapter can receive context and change
its residual, but the PICF context is not calibrated as a positive action
condition.
```

Combined causal result:

```text
E16 prefix_fusion + PICF coregrad:
  action worsens from 0.02665 to 0.03171/0.03517.

E17 PI0.5 action-only:
  action improves from 0.03394 to 0.02405/0.02599.

E18 suffix_cross_attention + PICF coregrad:
  action worsens from 0.02876 to 0.03222.
```

Therefore:

```text
The immediate action plateau is not explained by:
  dense token deletion;
  action-head incapacity;
  raw reserve overlap;
  SAM/sidecar noise;
  prefix-vs-suffix layout alone.

It is explained by:
  PICF belief/context is currently an uncalibrated or negative action
  condition.  The action path can learn when that condition is absent.
```

Next experiment:

```text
E19 native-action / PICF-auxiliary control.

Compute PICF normally, keep PICF auxiliary/diagnostic state available, but do
not pass PICF prefix/context into PI0.5 action.  This tests whether the system
can recover the E17 action descent without disabling the whole PICF runtime.
```

## 27. E19 Native Action / PICF-Auxiliary Control

Launch:

```text
root:
  /mnt/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_det_controls/native_action_picfaux_det/

tmux:
  picf_native_action_picfaux_det

tail:
  tail -f /mnt/picf_exact_window_probes/action_bridge_capacity_20260601/ckpt11000_exact12_det_controls/native_action_picfaux_det/run.log

checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000

windows:
  /mnt/picf_exact_window_probes/action_context_causal_20260601/
  stratified12_windows.jsonl

critical switch:
  picf_action_condition_enabled = false

config:
  steps = 200
  lr = 5e-5
  picf_core_lr_scale = 0.2
  picf_trainable_scope = all
  semantic_trainable_scope = action_head_and_adapter
  eval_before_after = true
  eval_every = 100
  deterministic_eval_seed = 20260601
```

Expected interpretation:

```text
If E19 improves like E17:
  PICF runtime/auxiliary computation is not the blocker.  The blocker is the
  action-visible PICF condition.

If E19 fails like E16/E18:
  the issue is not only action condition injection; it may be optimizer group,
  semantic/action-head trainability, or a probe/runtime mismatch.
```

E19 step100 status:

```text
step0 deterministic action = 0.0285684394
step100 deterministic action = 0.0290960873

This is a small regression, not a significant descent.

However E19 starts much lower than E17 action-only step0:
  E17 step0 = 0.0339394346
  E19 step0 = 0.0285684394

So step100 alone cannot distinguish:
  no room / local saturation,
from:
  PICF-aux runtime still changes optimization conditions,
from:
  semantic/action-head trainable-scope mismatch.

Continue E19 to step200 before final classification.
```

E19 step200 verdict:

```text
step0 deterministic action   = 0.0285684394
step100 deterministic action = 0.0290960873
step200 deterministic action = 0.0327226793

total_minus_action stayed essentially unchanged:
  0.1848280629 -> 0.1848275165
```

Interpretation:

```text
E19 did not reproduce E17's action-only descent.

This means the result is sharper than "PICF prefix/context injection is bad":
even with PICF action condition disabled, the PICF-enabled trainer path does
not behave like the clean ablated action-only path.

The next root-cause split must compare the actual trainer paths, not another
loss weight:
  ablated trainer path;
  PICF-enabled but no-action-condition path;
  semantic/action inputs emitted by both paths before compute_action_flow_loss.
```

New hypothesis:

```text
There is an action-path implementation difference between picf_mode=ablated
and picf_enabled/native-action-with-PICF-aux that is upstream of the explicit
extra_prefix_tokens / extra_action_context_tokens.

Candidates:
  semantic_override construction differs;
  burn-in/recurrent state differs;
  action target/window assembly differs;
  policy wrapper path changes action-flow inputs even when extra_prefix is
  None;
  the probe's fresh optimizer/trainable grouping differs after full PICF model
  materialization.
```

Do not proceed to another 30K run until this trainer-path difference is
localized.

File-level follow-through found one concrete mismatch:

```text
_forward_action_only_window(...)
  loops over all transitions in the window and does not apply burnin_steps.

PICF-enabled forward(...)
  applies train_start_index = burnin_steps
  and uses pending transition loss.

The current ckpt/args have effective_unroll_steps = 3 and burnin_steps = 1.
Therefore E17 action-only and E19 PICF-enabled native-action did not optimize
the same transition indices.
```

Immediate correction:

```text
Add probe-only --burnin-steps-override.

Run E20:
  PICF-enabled;
  picf_action_condition_enabled=false;
  burnin_steps_override=0;
  same exact12 windows;
  deterministic eval 0/100/200.

If E20 descends:
  the E19 failure was an objective-window mismatch, not PICF runtime.

If E20 still fails:
  compare semantic_override/action inputs directly between ablated and PICF
  enabled paths.
```

## 28. Literature Cross-Check: Why This Is an Action-Readout Problem

Recent object-centric and JEPA-VLA papers point to the same separation:

```text
SlotVLA (arXiv:2511.06754, revised 2026-05-06):
  dense embeddings entangle object/background cues;
  object-relation slots are compact and interpretable;
  but the paper uses object/relation representations explicitly for action
  decoding rather than assuming any upstream slot automatically helps action.

QASA (arXiv:2601.12936):
  fixed-K / K-adaptive slots need an explicit slot-quality or selection signal;
  slot-count penalties can conflict with reconstruction/assignment objectives.
  This is why another raw-overlap/count penalty is not the correct next move.

STORM object-centric robotics (arXiv:2601.20381):
  slot representations are first stabilized with visual-semantic pretraining,
  then jointly adapted with a policy.  The staged recipe exists specifically
  to avoid degenerate slot formation and semantic drift.

JEPA-VLA (arXiv:2602.11832) and VLA-JEPA (arXiv:2602.10098):
  video predictive embeddings/world-state signals are useful when they are
  action-relevant and injected into the VLA path through a calibrated action
  interface.  VLA-JEPA also warns against nuisance appearance/motion and
  leakage; future/auxiliary state must be target-side or calibrated, not an
  unverified action input.
```

Therefore the current code-level follow-through is:

```text
Keep:
  PICF belief construction;
  dense typed V-JEPA/point/tactile inputs;
  active/context/reserve gating;
  weak sidecar/object cues as guarded auxiliary evidence.

Do not repeat:
  raw-overlap-only penalties;
  fixed64 as global validation;
  blind SAM;
  direct dense append to PI0.5 prefix.

Test next:
  whether any PICF-to-action route has positive causal contribution.
  If not, action should run on native PI0.5 condition while PICF is trained and
  calibrated separately.
```

## 29. E20 Runtime Start: Burnin-Aligned Native Action / PICF-Aux

E20 is now running on A7:

```text
root: /mnt/picf_exact_window_probes/action_bridge_capacity_20260601/ckpt11000_exact12_det_controls/native_action_picfaux_burnin0_det
checkpoint: e14h step 11000
windows: exact stratified12
picf_mode: enabled
picf_action_condition_enabled: false
burnin_steps_override: 0
picf_trainable_scope: all
semantic_trainable_scope: action_head_and_adapter
picf_core_lr_scale: 0.2
```

Runtime checks:

```text
model_ready: true
pi_action_condition_token_count: 0
pi_context_token_count: 0
pi_context_adapter_token_count: 0
```

This confirms the action path is native PI0.5 while PICF observe/finalize and
PICF auxiliary objectives still run.

Initial deterministic eval:

```text
step0 loss_action_default_equiv mean: 0.0370756797
step0 loss_total_minus_action mean: 0.1759359526
```

Interpretation:

```text
The burnin-aligned PICF-enabled native-action baseline starts in the same broad
range as the action-only E17 baseline, unlike E19.  This supports the concrete
file-follow-through finding that E17 and E19 were not optimizing identical
transition indices.

The decisive criterion remains the step100/step200 deterministic eval trend:
  if E20 descends like E17, E19 was an objective-window mismatch;
  if E20 fails, compare semantic/action input tensors between ablated and
  PICF-enabled paths before any new training-policy change.
```

## 30. Root-Cause Matrix Before Any Further Experiment

This matrix is the current guard against repeating old experiments.

```text
H1 dense V-JEPA/context tokens are discarded:
  status: rejected by dataflow.
  evidence:
    token_field.temporal_visual.tokens remains dense;
    public/task readers still read visual + temporal visual memory;
    runtime metrics show owm_temporal_visual_tokens ~= 3456.
  implication:
    do not fix action plateau by blindly appending all dense tokens to PI0.5.

H2 raw same-role overlap directly causes action plateau:
  status: not supported as primary root.
  evidence:
    active/downstream overlap is separated from reserve/raw overlap;
    raw reserve overlap can remain high while active overlap is low.
  implication:
    do not add another raw-overlap penalty.

H3 SAM/sidecar noise is the immediate action-descent blocker:
  status: rejected for current exact-window causal probe.
  evidence:
    context perturbation and native-action controls fail/succeed independent
    of SAM-style proposal changes; SAM is already archived and off the main
    path.
  implication:
    do not spend the next experiment on SAM.

H4 PICF action prefix/context bridge is uncalibrated:
  status: supported.
  evidence:
    E16 prefix deterministic worsened;
    E18 suffix deterministic did not improve;
    context perturbation did not show positive action causal contribution.
  implication:
    action bridge must stay disabled or be redesigned only after a positive
    causal interface is proven.

H5 PI0.5 action head lacks local capacity:
  status: not supported.
  evidence:
    E17 ablated action-only improved on the fixed window set.
  caveat:
    E17 was not apples-to-apples with E19 because ablated mode ignored
    burnin_steps.

H6 PICF-enabled native-action path changes the optimized transition indices:
  status: supported by file-level follow-through.
  evidence:
    _forward_action_only_window optimizes every transition;
    PICF-enabled forward skips burnin transitions for training loss.
  implication:
    E20 is the decisive burnin-aligned control.

H7 PICF observe/finalize perturbs semantic/action inputs even when action
condition is disabled:
  status: pending.
  evidence needed:
    tensor-level dump of semantic_override, action_chunk_target, flow kwargs,
    and contributing transition indices from ablated vs PICF-enabled paths.
  trigger:
    only run this if E20 does not descend.
```

Mathematical statement of the current action-readout test:

```text
Let L_a(theta_s; x_t, u_t, eps, tau, c_t) be the PI0.5 flow loss, where
theta_s are trainable semantic/action parameters and c_t is the optional PICF
action context.

E16/E18 tested dL_a/dtheta_s with c_t = f_PICF(z_<=t).
E19 tested c_t = null but kept PICF-enabled transition indexing.
E20 tests c_t = null and burnin=0, aligning the transition set with E17.

If E20 improves:
  the readout failure is not native PI0.5 capacity;
  it is the calibrated use of c_t and/or burnin-stage objective definition.

If E20 fails:
  the equality
    L_a^ablated(x,u) == L_a^PICF_enabled_no_condition(x,u)
  is false at the tensor/dataflow level and must be debugged directly.
```

## 31. E20 OOM And Low-Memory E20b Control

E20 exited after step20 with CUDA OOM:

```text
last logged train step: 20
step20 recent_loss_action_default_equiv: 0.0278080029
error: CUDA out of memory while picf_trainable_scope=all
```

This does not invalidate the E20 causal design.  It means the full-PICF-gradient
version is too memory-heavy on the exact-window probe when burnin is zero.

The correct next control is not another architecture change.  It is E20b:

```text
picf_mode: enabled
picf_action_condition_enabled: false
burnin_steps_override: 0
picf_trainable_scope: policy_only
semantic_trainable_scope: action_head_and_adapter
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
```

Why this is still a valid root-cause test:

```text
When picf_action_condition_enabled=false, the action loss does not receive PICF
prefix/context tokens.  Therefore, for action descent, trainable PICF core
parameters are not required.  Freezing PICF core isolates whether merely using
the PICF-enabled recurrent trainer path changes the native PI0.5 action
objective relative to ablated action-only.
```

Decision rule:

```text
If E20b descends like E17:
  native action capacity is intact; the main failure is PICF action-context
  injection and/or full-PICF cotrain memory/optimization burden.

If E20b fails:
  the PICF-enabled path still differs from ablated action-only even with native
  action conditioning and burnin aligned; run tensor-level input dump next.
```

E20b runtime status:

```text
tmux: picf_native_action_picfaux_burnin0_policyonly_det
model_ready: true
trainable_numel: 8,457,249
pi_action_condition_token_count: 0 in eval/train rows
step0 deterministic loss_action_default_equiv: 0.0370756797
step1 train loss_action_default_equiv: 0.0299814548
step10 recent_loss_action_default_equiv: 0.0373982010
```

Step10 is not enough to classify the run because the logged window is
`move up the switch`, a known hard action bucket.  Continue to step100.

Step100 deterministic result:

```text
step0  deterministic loss_action_default_equiv: 0.0370756797
step100 deterministic loss_action_default_equiv: 0.0235597369
relative drop: about 36.5%

step0  deterministic loss_total_minus_action: 0.1759359526
step100 deterministic loss_total_minus_action: 0.2069764988
```

Decision:

```text
E20b descends strongly.  This closes the immediate "action cannot learn under
PICF runtime" hypothesis.

The root is not:
  dense V-JEPA token deletion;
  raw reserve overlap;
  sidecar/SAM noise;
  PI0.5 action-head local incapacity;
  CALVIN windows being intrinsically impossible.

The supported root is:
  the current PICF-to-action conditioning / full-PICF cotrain path can harm
  PI0.5 action optimization, while native PI0.5 action with PICF as auxiliary
  runtime remains learnable.
```

Training implication:

```text
Do not resume the branch that injects unproven PICF context into PI0.5 action.

Use native PI0.5 action conditioning as the production action path for the next
long run:
  picf_action_condition_enabled = false
  keep PICF observe/finalize and diagnostics
  keep PICF auxiliary/object structure as monitored auxiliary only
  train semantic action head/adapter normally
  reintroduce PICF-to-action context only after a separate causal interface
  probe shows positive fixed-window action contribution.
```

The rise in loss_total_minus_action during E20b is not an action blocker because
E20b is intentionally policy-only and optimizes the action scalar.  It indicates
that frozen PICF auxiliary metrics should not be interpreted as the optimization
target in this control.

## 32. E20b Follow-Through: Eval Must Be Stateless Before More Claims

The E20b step100 result is a real positive signal, but the step200 logs exposed
an evaluation-contract problem:

```text
step200 deterministic eval during loop:
  loss_action_default_equiv = 0.0370958755

step200 deterministic final eval, same checkpoint state and same seed:
  loss_action_default_equiv = 0.0436806177
```

This means the previous fixed-window probe had not fully proven deterministic
stateless evaluation.  The diffusion noise/time seed was fixed, but the eval
forward still mutates PICF runtime history:

```text
visual clip buffers:
  core.clip_buffers[view].push(...)

tactile clip buffer:
  core.tactile_buffer.push(...)

optional action prefix teacher / inference caches:
  must also be treated as mutable runtime state
```

Mathematical implication:

```text
The intended eval object is
  L_eval(theta; W, eps_seed)

The unguarded probe could instead evaluate
  L_eval(theta; W, eps_seed, h_runtime)

where h_runtime is the residual clip/tactile/action-prefix state left by the
previous train/eval pass.  If h_runtime changes between two nominally identical
eval calls, the comparison is invalid.
```

Code repair applied locally and synced to A7:

```text
1. MultiSensorTactileClipBuffer now supports snapshot()/restore().
2. picf_action_bridge_capacity_probe.py now wraps eval in _PicfRuntimeStateGuard:
   - snapshots/restores visual clip buffers;
   - snapshots/restores tactile buffers;
   - snapshots/restores action prefix teacher buffers when present;
   - snapshots/restores policy inference caches.
3. Each fixed-window eval now calls _reset_picf_runtime_buffers(trainer) before
   evaluating the window.
```

This is not a loss tweak.  It is an experimental-contract fix.  Without it, a
step100/step200 difference can be caused by mutable runtime state rather than by
the action bridge itself.

Current decisive test:

```text
E20c state-guard repeat:
  checkpoint = ckpt11000
  windows    = stratified exact12
  steps      = 1
  eval       = step0, step1, final step1
  expected   = the two step1 deterministic summaries should match up to
               numeric roundoff.

If they match:
  the probe can be trusted for short causal action-readout experiments.

If they still differ:
  the remaining nondeterminism is inside action-flow eval or another mutable
  semantic/PICF state not yet guarded; do not launch another 30k run from this
  evidence.
```

E20c result after the state-guard patch:

```text
steps = 1
step0 eval action_default_equiv = 0.0370756797
step1 eval during loop        = 0.0332281620
step1 final repeated eval     = 0.0377180163
```

The state-guard patch did not fully close deterministic eval.  Therefore the
remaining issue is narrower and more serious:

```text
Either:
  A. some mutable state beyond visual/tactile/action-prefix buffers still
     changes during eval;
or:
  B. the action-flow forward itself is not numerically deterministic enough
     under the current CUDA kernels / bf16 attention path;
or:
  C. the deterministic wrapper does not cover every random draw used by action
     loss.
```

Next discriminating test:

```text
E20d:
  steps = 0
  eval_before_after = true
  deterministic seed fixed

If the two step0 evals differ:
  the root is pure eval non-determinism or unguarded eval-side state, not
  training or optimizer.

If the two step0 evals match:
  E20c's mismatch is caused by eval-after-train state not yet guarded.
```

E20d result with state guard but RNG-reset-only flow inputs:

```text
steps = 0
first step0 eval  = 0.0370756797
second step0 eval = 0.0368241016
absolute delta    = 0.0002515782
```

This is much smaller than the E20c step1 duplicate gap, but it is still not a
strictly identical fixed mathematical objective.  The remaining likely source is
the action-flow stochastic input construction or nondeterministic CUDA kernels,
not slot collapse.

Contract upgrade applied next:

```text
_DeterministicFlowRng now injects explicit `noise` and `time` tensors into
semantic_encoder.compute_action_flow_loss during eval.

noise:
  torch.randn((1, horizon, model_action_dim), generator=seeded_generator)

time:
  inverse-CDF sample from Beta(1.5, 1.0):
    u ~ Uniform(0, 1)
    time = u^(1/1.5) * 0.999 + 0.001
```

This preserves the PI0.5 flow-loss distribution while removing hidden random
draws from the eval path.  E20e reruns the zero-step repeated eval with explicit
flow inputs.

Dense-token hypothesis status:

```text
Rejected as stated.

PICF internal dataflow keeps dense temporal visual tokens in:
  token_field.temporal_visual.tokens
  aqr_temporal_visual_reader
  _build_public_read_memory
  task readout

The lossy part is only the action-visible bridge from belief state to PI0.5.
Therefore the next repair should not be "force all dense tokens into slots" or
"append thousands of dense tokens to PI0.5"; that would break the V-JEPA-style
latent-world representation and make inference slower.  The correct object is a
guarded/calibrated readout interface, tested causally.
```

## 33. E20e Result: Eval Noise Is Small But Not Mathematically Zero

E20e reran the zero-step repeated deterministic eval after injecting explicit
action-flow `noise` and `time` tensors.

Result:

```text
checkpoint:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000

windows:
  stratified12 exact windows

step0 eval 1 loss_action_default_equiv:
  0.0370756797

step0 eval 2 loss_action_default_equiv:
  0.0368241016

absolute delta:
  0.0002515782
```

Interpretation:

```text
Explicit flow-noise/time injection does not make the full PaliGemma action
eval bit-identical.  The remaining delta is small relative to the action
changes we care about, but it is still real.

Therefore:
  use deterministic eval for trend direction and large deltas;
  do not use single 1e-4-level differences as proof;
  do not claim exact mathematical equality of duplicate evals under CUDA/bf16.
```

This is still enough to run the causal E20f control:

```text
E20f:
  same checkpoint
  same 12 exact windows
  burnin = 0
  PICF runtime enabled for aux/diagnostics
  PICF action condition disabled
  trainable = action head + adapter only
  eval = explicit flow inputs + runtime state guard
  steps = 200

Acceptance:
  if step100/step200 improve by >> 0.00025, the improvement is real;
  if they do not, the current blocker is not eval noise.
```

## 34. Paper-Consistent Readout Principle For The Next Experiments

The current experiments must stay aligned with the literature direction rather
than adding another ad-hoc anchor loss.

Relevant paper guidance:

```text
V-JEPA 2 (arXiv:2506.09985):
  dense predictive video latents are valuable as world-model evidence.
  They should not be destroyed by forcing every dense token into an active
  object slot.

JEPA-VLA (arXiv:2602.11832):
  V-JEPA-style predictive embeddings are useful as VLA conditioning.
  This supports a bounded learned conditioning interface, not raw unbounded
  dense-token append.

VLA-JEPA (arXiv:2602.10098):
  future/predictive information must be leakage-free and action-relevant.
  This supports using current belief/context as conditioning while keeping
  predictive auxiliaries detached/guarded.

OA-WAM (arXiv:2605.06481):
  object-addressable memory matters when actions refer to a particular object.
  This supports PICF object belief states, but it does not imply hard-locking
  noisy addresses or forcing all background context into object slots.

OmniVTA (arXiv:2603.19201):
  tactile/contact evidence should correct belief under contact, not become a
  free-standing always-on action shortcut.
```

Design consequence:

```text
The correct mathematical target is not:
  append all dense tokens to PI0.5
  or force all dense tokens into posterior slots
  or add another raw-overlap loss.

The correct target is:
  keep dense evidence available inside PICF and task readout;
  pass only calibrated action-relevant belief/context to PI0.5;
  prove by causality that the action path can still descend when PICF runtime
  exists but direct PICF action conditioning is removed.
```

This is why E20f is the next required experiment.  If E20f descends on the exact
12 windows, then the action platform is caused by the PICF action-visible bridge,
not by dense-token deletion or action-head incapacity.  If E20f fails, then the
remaining blocker is in the trainer/runtime state itself.

## 35. E20f Step100 Result: Action Starts Significant Descent

E20f reached the decisive step100 deterministic eval.

Configuration:

```text
checkpoint:
  ckpt11000 from e14h action4 fullopt

windows:
  stratified12 exact windows

PICF runtime:
  enabled

PICF action condition:
  disabled

burnin:
  0

trainable:
  action head + adapter only

eval:
  explicit flow noise/time
  runtime state guard
```

Result:

```text
step0  deterministic loss_action_default_equiv:
  0.0370756797

step100 deterministic loss_action_default_equiv:
  0.0235446243

relative drop:
  about 36.5%

E20e duplicate-eval noise scale:
  about 0.0002516
```

Therefore the step100 improvement is not a deterministic-eval artifact.  It is
about 54 times larger than the measured duplicate-eval noise.

Root-cause implication:

```text
Rejected:
  action head cannot learn;
  PICF runtime itself makes action impossible;
  dense V-JEPA tokens are deleted before the model can use them;
  raw reserve overlap is the immediate action blocker.

Supported:
  direct PICF-to-action conditioning is currently not calibrated as a positive
  action signal.  When that condition is removed, the native action branch
  starts descending immediately on the same exact-window test.
```

The remaining reason to continue E20f to step200 is not to prove initial
descent; that is already proven.  It is to test whether the same native-action
control remains stable after the first successful descent.

## 36. E20f Step200 Result: Initial Descent Is Real, Stability Is Not Proven

E20f completed 200 sequential single-window optimizer steps.

Deterministic all-window eval:

```text
step0:
  loss_action_default_equiv = 0.0370756797

step100:
  loss_action_default_equiv = 0.0235446243

step200 eval_every:
  loss_action_default_equiv = 0.0370589560

step200 final repeated eval:
  loss_action_default_equiv = 0.0436937434
```

Training rows also show large per-window oscillation:

```text
step140 train optimized_loss = 0.004664
step150 train optimized_loss = 0.001374
step160 train optimized_loss = 0.023568
step170 train optimized_loss = 0.039968
step180 train optimized_loss = 0.004308
step190 train optimized_loss = 0.058243
step200 train optimized_loss = 0.002693
```

Interpretation:

```text
The step100 descent is real because it is far larger than duplicate eval noise.

The step200 rebound is also meaningful as a stability warning, but its exact
value is not yet a clean scalar because repeated eval after training still has
multi-window runtime/nondeterminism sensitivity.

The dominant new hypothesis is sequential-window interference:
  each optimizer step sees only one exact window;
  the next window may undo the previous window's action adapter update;
  train rows can look excellent on the current window while the 12-window mean
  returns to baseline.
```

This does not resurrect the rejected hypotheses:

```text
Not supported as first-order causes:
  dense V-JEPA token deletion inside PICF;
  action head incapable of learning;
  PICF runtime itself making action impossible;
  raw same-role overlap alone blocking action.

Still supported:
  direct PICF action-visible conditioning is currently not calibrated;
  exact-window training must be balanced across heterogeneous task windows
  before its action-loss trajectory can be interpreted.
```

## 37. E21 Required Experiment: Balanced All-Window Gradients

E21 changes exactly one causal factor relative to E20f:

```text
E20f:
  one exact window per optimizer step

E21:
  all 12 exact windows are accumulated before each optimizer step
```

Mathematically, E20f optimizes a high-variance stochastic objective:

```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta L_{i_t}(\theta_t)
```

where `i_t` cycles through heterogeneous tasks.  E21 optimizes the exact
stratified-window mean:

```math
\theta_{t+1} = \theta_t
  - \eta \nabla_\theta \frac{1}{12}\sum_{i=1}^{12} L_i(\theta_t)
```

Decision rule:

```text
If E21 descends and stays below step0:
  root cause is sequential-window interference / forgetting, not capacity.

If E21 also rebounds:
  root cause moves to action adapter capacity, optimizer/LR geometry, or an
  uncalibrated action-visible representation even without direct PICF context.

If E21 train rows descend but eval rows do not:
  remaining blocker is eval/runtime state equivalence, not optimization.
```

This experiment avoids repeating the quarantined fixed64 mistake because it
uses the same stratified 12-window exact set and reports only causal local
capacity, not global CALVIN training quality.

## 38. Extra 2025-2026 Paper Check: Why E21 Is Not Ad-Hoc

The next experiment is also consistent with the newer slot/VLA literature:

```text
SlotVLA (arXiv:2511.06754):
  object/relation slots are useful because they reduce dense-token entanglement
  and make manipulation representations compact.  This supports PICF's belief
  routing idea, but it does not prove that a noisy PICF context should be
  injected into PI0.5 without calibration.

OBEYED-VLA (arXiv:2512.22519):
  robust VLA control benefits from disentangling object-centric perception from
  action reasoning.  This supports testing whether the action branch learns
  better when PICF perception runs but direct action-visible PICF conditioning is
  disabled.

SlotVTG (arXiv:2603.25733):
  slot adapters can improve grounding by decomposing visual tokens into abstract
  slots while preserving input-grounded sequence information.  This supports a
  bounded adapter/readout rather than raw dense append.

When Slots Compete (arXiv:2603.11246):
  fixed slot sets can produce redundant slots competing over the same entity;
  the principled repair is overlap-aware merging/aggregation of slot evidence,
  not deleting dense context or adding another unrelated action loss.
```

Therefore E21 is a principled causal test:

```text
It does not add a new module.
It does not hide the action problem behind another auxiliary.
It tests whether the existing action branch has a stable multi-task gradient
when every optimizer step sees the same stratified objective.
```

If E21 succeeds, the production implication is not "train on fixed12 windows".
It is:

```text
Use task-balanced / bucket-balanced batches or accumulation for action updates;
keep PICF action conditioning guarded until it proves positive on identical
windows/noise/time;
do not keep lowering PICF LR or tuning raw overlap as the first repair.
```

## 39. E21 Step20 Result: Balanced Gradients Restore Action Descent

E21 reached the first decisive all-window eval point.

Configuration:

```text
checkpoint:
  ckpt11000 from e14h action4 fullopt

windows:
  same stratified12 exact windows as E20f

PICF runtime:
  enabled

PICF action condition:
  disabled

burnin:
  0

trainable:
  action head + adapter only

optimizer step:
  accumulate all 12 windows before one optimizer update
```

Observed train rows:

```text
step1  train mean action = 0.036811
step5  train mean action = 0.038891
step10 train mean action = 0.035661
step15 train mean action = 0.029657
step20 train mean action = 0.027274
```

Deterministic all-window eval:

```text
step0:
  action_default_equiv = 0.0370756797

step20:
  action_default_equiv = 0.0259354621

relative drop:
  about 30.1%
```

This is the first clean positive result in the current causal chain because:

```text
same checkpoint;
same 12 exact windows;
same deterministic flow objective;
PICF runtime still enabled;
direct PICF action condition disabled;
all windows included in every optimizer step.
```

Root-cause update:

```text
Strongly supported:
  action-side capacity exists;
  the previous step200 rebound in E20f was caused by sequential-window
  interference / forgetting, not by lack of information in the native action
  path;
  direct PICF action-visible conditioning remains unproven and should stay
  guarded until it passes identical-window causal tests.

Rejected more strongly:
  "the architecture cannot make action descend";
  "dense temporal visual evidence has been destroyed";
  "raw overlap is the immediate action blocker";
  "the only repair is more object/anchor losses".
```

Production implication:

```text
Do not resume long training with arbitrary rolling single-window action updates
and then interpret 50-step rows as stationary evidence.

Use task-balanced accumulation or a bucket-balanced sampler so action updates
see a stable approximation of the mixed-task objective.

PICF action conditioning must be treated as a separate learned interface, not
assumed beneficial because structure metrics are improving.
```

E21 should continue to step40/60 only to test stability.  The core question
"can action start significant descent under balanced exact-window optimization?"
is already answered yes.

## 40. E21 Step40 Result: Descent Holds, But Not Monotonic

E21 reached the second all-window eval point.

Training rows:

```text
step25 train mean action = 0.026461
step30 train mean action = 0.027233
step35 train mean action = 0.026667
step40 train mean action = 0.034448
```

Deterministic all-window eval:

```text
step0:
  action_default_equiv = 0.0370756797

step20:
  action_default_equiv = 0.0259354621

step40:
  action_default_equiv = 0.0284216780
```

Interpretation:

```text
The improvement is still real at step40:
  step40 is about 23.3% below step0.

But the curve is not monotonic:
  step40 is worse than step20.
```

This updates the diagnosis:

```text
Balanced all-window gradients fix the first-order "action cannot descend"
failure.

They do not by themselves prove a long stable basin.  The remaining issue is
likely one of:
  action adapter/head capacity is enough to descend but not enough to keep
  improving all task families;
  lr is high for this tiny exact-window mean objective;
  the native action path can fit these windows, but direct PICF action context
  still lacks calibrated positive contribution.
```

Still do not return to raw-overlap or dense-token deletion as the primary
explanation: those hypotheses do not explain why E21 immediately improves the
same windows after only balanced updates.

## 41. Eval Guard Follow-Up: Per-Window Runtime Isolation

E21 exposed another measurement hygiene issue: final repeated eval can still
differ from the eval-every row at the same step.  The likely code-level reason
is that the previous evaluator wrapped one `_PicfRuntimeStateGuard` around the
whole 12-window eval loop.  That guarded the eval as a block, but did not force
every individual window to start from the same teacher/cache state.

Patch:

```text
scripts/picf_action_bridge_capacity_probe.py::_evaluate_prepared

Before:
  one runtime guard around all eval windows

After:
  one runtime guard per eval window
  reset clip/tactile buffers inside each per-window guard
  restore trainer train/eval mode to its previous state
```

This does not change the E21 optimization result.  It makes the next causal
probe stricter by preventing eval-loop state carryover from masquerading as a
model-quality change.

## 42. E21 Final And E22 Eval-Guard Check

E21 completed 60 balanced all-window optimizer steps.

Training rows:

```text
step45 train mean action = 0.027703
step50 train mean action = 0.026273
step55 train mean action = 0.031212
step60 train mean action = 0.023190
```

Eval-every summaries:

```text
step0:
  action_default_equiv = 0.0370756797

step20:
  action_default_equiv = 0.0259354621

step40:
  action_default_equiv = 0.0284216780

step60 eval-every:
  action_default_equiv = 0.0272792162
```

The final repeated eval at step60 was:

```text
step60 final repeated:
  action_default_equiv = 0.0366903716
```

This repeated-final mismatch means final repeated eval should not be used as
the headline metric for E21.  The eval-every rows and training rows still show
that balanced all-window gradients make action descend, but a fully bit-stable
post-training repeated eval remains unresolved.

E22 tested the per-window runtime guard with a zero-step duplicate eval:

```text
eval1 step0 action_default_equiv = 0.0370756797
eval2 step0 action_default_equiv = 0.0368241016
delta                           = -0.0002515782
```

This is the same small residual noise scale as E20e.  Therefore:

```text
E21 step20 drop:
  0.0370756797 -> 0.0259354621
  absolute drop ~= 0.0111402177
  about 44x larger than duplicate-eval noise

E21 step60 eval-every drop:
  0.0370756797 -> 0.0272792162
  absolute drop ~= 0.0097964635
  about 39x larger than duplicate-eval noise
```

Conclusion:

```text
The action path can start significant descent under balanced all-window
optimization.

The remaining unresolved part is not "can it descend"; it is how to make the
production trainer approximate this balanced objective without turning every
step into a 12-window exact diagnostic.
```

Next production-level experiment:

```text
Use bucket-balanced action accumulation in normal training:
  each optimizer update should cover a controlled mix of CALVIN task families;
  keep direct PICF action condition disabled or separately gated until it proves
  positive under identical-window tests;
  report action loss by task bucket, not only global rolling rows.
```

## 43. E23 Production-Scale Follow-Through Plan

The E21/E22 conclusion is now carried into the normal trainer rather than kept
as an exact-window-only diagnostic.

Code-level follow-through:

```text
scripts/picf_core_train.py
  _calvin_prompt_bucket(prompt)
  _CalvinTransitionSource.bucket_to_slot_indices
  _CalvinTransitionSource.balanced_bucket_slot_index(...)
  --calvin-balanced-bucket-sampler
  window_trace.prompt_bucket

scripts/experiments/.../run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh
  ACCUM_STEPS env override
  CALVIN_BALANCED_BUCKET_SAMPLER env override
  PICF_ACTION_CONDITION_ENABLED env override
```

Mathematical target:

```text
Old production update:
  g_t = grad L(theta; w_t)

where w_t may come from one task family for many adjacent optimizer updates.
This exactly matches the failure mode found in E20f: fast local descent on one
window family followed by cross-family rebound.

E23 production update:
  g_t = (1/K) sum_{k=1..K} grad L(theta; w_{t,k})

where w_{t,k} are sampled from coarse CALVIN language buckets across ranks and
gradient-accumulation micro-steps.  This approximates the balanced E21 exact
objective while keeping the normal online CALVIN dataflow.
```

Design constraints:

```text
No new model module.
No new loss.
No masking/proposal semantics added to action supervision.
No reliance on fixed64 probes.
The sampler only changes which real training windows appear in an optimizer
update.
```

Run contract:

```text
Experiment:
  picf_a7_e23_bucketbalanced_noactioncond_from11000_30k_20260601

Resume:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000

Scale:
  num_train_steps = 30000
  save_interval = 1000
  keep_last_checkpoints = 5
  log_interval = 50
  anchor_overlay_interval = 100

Control:
  calvin_balanced_bucket_sampler = true
  accum_steps = 1 on 2 GPUs, effective 2 windows/update with deterministic
  bucket rotation across steps
  window_activation_checkpointing = false
  picf_action_condition_enabled = false
  optimizer_checkpoint_mode = model-only

Reason for disabling direct PICF action condition:
  E21 proved action descent through the native action path under balanced
  updates.  The direct PICF action bridge has not yet proved positive under
  identical-window causal tests, so it remains off for this production-scale
  confirmation instead of being allowed to reintroduce an uncalibrated shortcut.

Reason for model-only optimizer resume:
  full optimizer restore plus accum_steps=2 exceeded the 40GB A100 budget during
  token-fusion attention at the first resumed update.  Keeping full optimizer
  state would require accum_steps=1 and would fail to test the E21 balanced
  gradient mechanism.  E23 therefore keeps the model state fixed at step11000
  and rebuilds Adam state so the production balanced objective can actually run.

Memory constraint:
  accum_steps=2 OOMs on 40GB A100s at the second micro-step, both without
  checkpointing and with window activation checkpointing.  Therefore E23 uses
  the strongest production-feasible form on this hardware: bucket-balanced
  rank/step rotation at accum_steps=1.  It is weaker than E21's exact
  all-window objective, but it is not legacy random sampling and can run at
  30K scale.
```

Acceptance focus for the first hundreds of steps:

```text
Primary:
  loss_action_default_equiv should show bucket-stable descent or at least avoid
  the old monotonic rebound.

Secondary:
  window_trace should contain rotating prompt_bucket values.
  action loss must be interpreted with prompt_bucket coverage, not a single
  rolling global row.

Do not over-interpret:
  raw same_role_support_overlap remains a structural diagnostic.  Under the
  current context-slot / active-slot split, it is not by itself the primary
  blocker unless active/downstream overlap and action loss degrade together.
```

## 44. E23 Stop And Cloud-Preservation Record

Date: 2026-06-02 Asia/Shanghai.

The A7 `picf_a7_e23_bucketbalanced_30k` tmux run was stopped intentionally
before shutting down the cloud machine.  This stop was not caused by a crash.

Remote stop state:

```text
session:
  picf_a7_e23_bucketbalanced_30k

log:
  /mnt/picf_run_logs/
  picf_a7_e23_bucketbalanced_noactioncond_from11000_30k_20260601.log

last structured row:
  step = 11400

metrics:
  loss_total                 = 0.0932965
  loss_action_default_equiv  = 0.0417017
  loss_action                = 0.0834034
  loss_total_minus_action    = 0.0098931
  loss_anchor_pv             = 0.4854432
  loss_anchor_object_pull    = 0.1323347
  loss_slot_jepa             = 0.6280357
  active_same_role_overlap   = 0.1100
  downstream_overlap         = 0.1076
  posterior_recycle_rate     = 0.1039
  posterior_identity_switch  = 0.2411
  speed                      = 0.0377 steps/sec
```

Interpretation:

```text
1. E23 did not reproduce E21's rapid all-window descent.
   E21 used 12 exact windows per optimizer update.  E23 used the strongest
   production-feasible 40GB A100 variant: 2 GPUs, accum_steps=1, and
   bucket-balanced rank/step rotation, giving only 2 windows per update.

2. E23 did not reproduce the old active/downstream anchor collapse.
   active/downstream same-role overlap stayed low.  Raw overlap remained 1.0,
   but this is reserve/inactive dominated under the current router split and
   is not by itself the action blocker.

3. The action scalar improved from the first reported row but stayed in the
   0.041-0.048 band.  That is better than a monotonic rebound, but weaker than
   the E21 exact-balanced diagnostic.

4. The retained conclusion is therefore not "the problem is solved".
   The retained conclusion is: cross-task gradient balance is the correct
   causal direction, but the current 2-window/update production approximation
   is too weak to fully reproduce E21.
```

Hardware implication:

```text
Current 2xA100-40GB:
  fsdp_full_shard + accum_steps=1 is already near the memory ceiling.

Tried and rejected:
  accum_steps=2 on 2x40GB, with and without window activation checkpointing.
  Both hit the 40GB budget.

Recommended next hardware route:
  4xA100-40GB with accum_steps=1.
  This gives 4 windows/update without increasing per-rank accumulation memory.

Closest E21 production route:
  12 windows/update, ideally 12 GPUs with accum_steps=1 or a larger-memory
  setup.  Do not force 2x40GB accum6 or 4x40GB accum3 as the next production
  attempt; that repeats the OOM path.
```

Cloud preservation:

```text
Remote dirty snapshot saved locally:
  /home/siyuanyue/Documents/openpi/temp/remote_a7_snapshot_20260602/
  a7_openpi_dirty_snapshot_20260602.tar.gz

Contents:
  remote git status;
  remote git diff --binary;
  small changed/untracked files copied from the remote repo.

Purpose:
  preserve cloud-only code/document changes before the A7 machine is stopped.
```

Maintained next action:

```text
Do not keep tuning raw overlap.
Do not revive fixed64 as a production proxy.
Do not treat E23 as the final 30K proof.

If stronger hardware is available:
  rerun the bucket-balanced production recipe with more windows/update,
  preferably 4 GPUs accum1 first.

If only 2 GPUs are available:
  keep E23 as the best runnable approximation, but interpret slow action
  descent as expected under a weak 2-window gradient estimator.
```

## 45. Six-GPU Follow-Through Plan

Date: 2026-06-02

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_6x40g_e23_bucketbalanced_noactioncond_from11000_30k_20260602.sh
```

Rationale:

```text
E21 exact-window diagnostic:
  12 windows per optimizer update;
  strongest observed action descent;
  not a production sampler.

E23 two-GPU production approximation:
  2 windows per optimizer update;
  action stable but slow;
  active/downstream structure healthy.

6x40GB plan:
  6 windows per optimizer update;
  no gradient accumulation;
  no extra per-rank activation memory;
  closer to E21 while preserving the normal bucket-balanced CALVIN trainer.
```

Mathematical contract:

```text
Let g_i(theta) be the per-window gradient.

2-GPU E23 estimates:
  g_hat_2 = (g_1 + g_2) / 2

6-GPU follow-through estimates:
  g_hat_6 = (1/6) sum_{i=1}^6 g_i

If task-family gradient variance dominates the action plateau, then:
  Var[g_hat_6] ~= Var[g_hat_2] / 3

This tests the root-cause hypothesis without changing the model contract.
It is not a new module and not a loss rewrite.
```

Guardrails:

```text
direct PICF action condition:
  disabled

action pressure:
  same E23/E14H action4 setting

PICF core LR:
  low constant, same maintained E23 contract

resume:
  model-only from E14H step 11000

sidecar:
  full clean contact-motion tracklet sidecar

save:
  every 1000 steps, keep last 5
```

## 46. Six-Window Follow-Through Result And 12-Window Next Step

Date: 2026-06-02

Run:

```text
picf_6x40g_e23_bucketbalanced_noactioncond_from11000_30k_20260602
resume: E14H step 11000
hardware: 6xA100-40GB
world_size: 6
accum_steps: 1
effective windows/update: 6
window_activation_checkpointing: false
```

Observed structured metrics:

```text
step 11050:
  loss_action_default_equiv = 0.0440368
  loss_total                = 0.0990952
  loss_total_minus_action   = 0.0110215
  active support overlap    = 0.164923
  downstream support overlap= 0.154397

step 11800:
  loss_action_default_equiv = 0.0404126
  loss_total                = 0.0911475
  loss_total_minus_action   = 0.0103223
  active support overlap    = 0.0700000
  downstream support overlap= 0.0693600

step 11850:
  loss_action_default_equiv = 0.0433774
  loss_total                = 0.0965711
  loss_total_minus_action   = 0.00981638
  active support overlap    = 0.0916667
  downstream support overlap= 0.0901238
  raw same-role overlap     = 1.0
```

Interpretation:

```text
The 6-window run exceeded the requested 500-step observation window.

Positive:
  active/downstream support overlap stayed healthy;
  raw overlap is still reserve/inactive dominated and is not the action blocker;
  total-minus-action stayed small and stable;
  sidecar/object pull did not destabilize the graph.

Negative:
  action did not reproduce E21-style rapid descent;
  loss_action_default_equiv only moved from 0.0440 to 0.0434 over the
  structured 11050->11850 interval, with a best transient point around 0.0404.

Conclusion:
  6 windows/update improves estimator diversity relative to 2-window E23,
  but it is not enough evidence for final 30K acceptance.
```

Next launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_6x40g_e21like_accum2_windowckpt_noactioncond_from11000_30k_20260602.sh
```

Contract:

```text
world_size = 6
accum_steps = 2
effective windows/update = 12
window_activation_checkpointing = true
training_strategy = fsdp_full_shard
optimizer_checkpoint_mode = model-only
```

Mathematical reason:

```text
Let g_i(theta) be per-window gradients.

Current 6-window run:
  g_hat_6 = (1/6) sum_i g_i

E21-like run:
  g_hat_12 = (1/12) sum_i g_i

Under task-family gradient variance, the 12-window estimator should have:
  Var[g_hat_12] ~= Var[g_hat_6] / 2

This is the closest production-compatible test of the E21 evidence on the
available 6x40GB hardware. It changes estimator quality, not model semantics.
```

Decision rule:

```text
If E21-like 12-window restores action descent while preserving active/downstream
overlap, the remaining blocker was gradient estimator/task-family mixing.

If it still plateaus near 0.04 after a comparable 500-step observation window,
the blocker is not just windows/update; reopen the action-readout objective,
sampling curriculum, and context injection design rather than continuing
long-run compute.
```

Observed startup audit, 2026-06-02:

```text
Remote:
  6xA100-40GB px-cloud2 / wEODGx

Known-good run:
  picf_6x40g_e23_bucketbalanced_noactioncond_from11000_30k_20260602
  world_size=6
  accum_steps=1
  training_strategy=fsdp_full_shard
  status=ran past the 500-step observation window

Attempted E21-like run:
  picf_6x40g_e21like_accum2_windowckpt_noactioncond_from11000_30k_20260602
  world_size=6
  accum_steps=2
  training_strategy=fsdp_full_shard
  window_activation_checkpointing=1
  status=did not reach Training config; no loss rows produced

Control diagnostic:
  picf_diag_accum2_no_wckpt_from11000_11001b_20260602
  world_size=6
  accum_steps=2
  training_strategy=fsdp_full_shard
  window_activation_checkpointing=0
  status=also did not reach Training config within the startup window

Additional check:
  standalone torchrun --nproc_per_node=6 rank sanity passed on the same host.
```

Interpretation:

```text
This is not an action-loss result and not an OOM result.

The 12-window attempt currently exposes a project startup incompatibility in the
6-rank FSDP + accum_steps=2 + model-only resume path. Since both
window_checkpointing=1 and window_checkpointing=0 variants stalled before
Training config, the first suspect is the accum=2/FSDP/resume startup path
rather than the checkpointed forward itself.

Do not cite the 12-window attempt as plateau evidence. It produced no training
metrics. Before another E21-like run, add per-rank startup tracing around:
  _build_model_sequential_across_ranks
  _wrap_model_for_training_strategy
  _build_optimizer
  _load_checkpoint_sequential_across_ranks
  the post-resume distributed barrier
```

## 47. E21b2 Exact-Window Small-Batch Probe

Date: 2026-06-02

Question:

```text
E21 proved that the action path can descend when every optimizer update sees
the full stratified 12-window objective.  It did not answer whether a smaller
update, still cycling through the same task set, is stable.

E21b2 tests exactly that missing point:
  same checkpoint;
  same stratified12 exact windows;
  same action-head/adapter-only trainable scope;
  same PICF runtime with direct PICF action conditioning disabled;
  windows_per_step = 2 instead of 12.
```

Mathematical object:

```math
g_t^{(2)}
  = \frac{1}{2}
    \left(\nabla_\theta L_{i_t}(\theta_t)
          + \nabla_\theta L_{j_t}(\theta_t)\right)
```

where the index pair cycles through the fixed 12-window set.  This differs
from E21:

```math
g_t^{(12)}
  = \nabla_\theta \frac{1}{12}\sum_{i=1}^{12} L_i(\theta_t)
```

Decision rule:

```text
If E21b2 descends on all-window deterministic eval and stays below step0:
  two-window updates may be sufficient when the sequence is deterministic and
  balanced over a short cycle.

If E21b2 train rows dip but all-window eval rebounds:
  the problem is not fixed by simply cycling through task buckets.  The action
  update itself is too task-family-biased when each optimizer step sees only
  two windows.

If E21b2 behaves like E21:
  production should focus on better bucket rotation / small-batch curriculum.

If E21b2 behaves like E20f:
  production needs an architectural or optimization mechanism that reduces
  cross-task gradient conflict at small update batches, rather than relying on
  more GPUs.
```

Remote run:

```text
host:
  px-cloud1 / Zky12J / 2xA100-40GB

worktree:
  /root/openpi_e21b2_1b07eab

run root:
  /root/picf_exact_window_probes/action_bridge_capacity_20260601/
  ckpt11000_exact12_det_controls/e21b2_windows2_policyonly_180

tmux:
  picf_e21b2_windows2

tail:
  tail -f /root/picf_exact_window_probes/action_bridge_capacity_20260601/ckpt11000_exact12_det_controls/e21b2_windows2_policyonly_180/run.log
```

Early read:

```text
step0 deterministic all-window eval:
  loss_action_default_equiv mean = 0.0370217793

step30 deterministic all-window eval:
  loss_action_default_equiv mean = 0.0333460527

step60 deterministic all-window eval:
  loss_action_default_equiv mean = 0.0342119768

train rows:
  step10 recent = 0.0326874721
  step20 recent = 0.0384596322
  step30 recent = 0.0375019123
  step40 recent = 0.0302849101
  step50 recent = 0.0332465792
  step60 recent = 0.0312920111
  step70 recent = 0.0260460178
```

Provisional interpretation:

```text
windows_per_step=2 is not immediately dead: all-window eval improves at step30.

But it is weaker than E21:
  E21 step20 reached 0.0259354621;
  E21b2 step60 is still 0.0342119768.

The step30->step60 rise already suggests that two-window updates still carry
task-pair bias.  Continue to step90/180 before final classification.
```

Step120 update:

```text
step90 deterministic all-window eval:
  loss_action_default_equiv mean = 0.0300650926

step120 deterministic all-window eval:
  loss_action_default_equiv mean = 0.0308203372

recent train rows:
  step70  windows [6,7]   loss_action_default_equiv = 0.0260460178
  step80  windows [2,3]   loss_action_default_equiv = 0.0337364814
  step90  windows [10,11] loss_action_default_equiv = 0.0330096092
  step100 windows [6,7]   loss_action_default_equiv = 0.0305383532
  step110 windows [2,3]   loss_action_default_equiv = 0.0367153912
  step120 windows [10,11] loss_action_default_equiv = 0.0291963909
```

E21b2 conclusion:

```text
windows_per_step=2 can learn, but it does not reproduce E21's fast
all-window descent.

The remaining gap is not explained by action-head incapacity:
  E21 already showed action-head/adapter capacity on the same checkpoint,
  same exact windows, and same action loss.

The remaining gap is the update estimator:
  K=2 gradients are still task-pair-biased;
  K=12 gradients approximate the intended multi-task objective on this
  diagnostic support.
```

## 48. E24 Logical-Batch Plan

Date: 2026-06-02

This section replaces the optimizer-reset interpretation with a stricter
multi-task-gradient interpretation.

Target objective:

```math
L(\theta)
  = \sum_{b \in \mathcal{B}} q_b
    \mathbb{E}_{x \sim \mathcal{D}_b}
      \left[\ell_b(\theta; x)\right],

G(\theta)
  = \nabla_\theta L(\theta)
  = \sum_{b \in \mathcal{B}} q_b
    \mathbb{E}_{x \sim \mathcal{D}_b}
      \left[\nabla_\theta \ell_b(\theta; x)\right].
```

Here `b` is not merely a dataset id.  For CALVIN it should at minimum include
coarse task family:

```text
drawer
switch_button_light
slider
block_push
block_lift
block_other
other
```

Small physical batch estimator:

```math
\hat{G}_K(\theta)
  = \frac{1}{K}\sum_{k=1}^{K}
      \nabla_\theta \ell_{b_k}(\theta; x_k).
```

If `K` covers only one or two task families, then:

```math
\mathrm{Var}[\hat{G}_K]
  =
    \frac{1}{K}\mathbb{E}_b
      \left[\lVert \nabla L_b - G \rVert^2\right]
    + \text{within-task variance}.
```

This is the observed failure mode:

```text
E20f:
  K=1 exact-window sequential updates can descend on the current window and
  then rebound on all-window eval.

E21:
  K=12 exact-window logical update descends quickly because every optimizer
  step approximates the fixed multi-task objective.

E21b2:
  K=2 exact-window logical update descends, but more slowly and with noisier
  all-window eval.

E23 production:
  bucket-balanced 2-GPU updates saw only two physical windows/update and did
  not reproduce E21's descent, even though bucket cycling was enabled.
```

Therefore the next test is not another single optimizer tweak.  It is a
controlled estimator-width test:

```text
E24a:
  exact12 windows_per_step=4
  action-head/adapter-only
  PICF action condition disabled
  step0/30/60/90/120 deterministic all-window eval

E24b:
  exact12 windows_per_step=6
  action-head/adapter-only
  PICF action condition disabled
  step0/30/60/90 deterministic all-window eval

Decision:
  If K=4/K=6 approach E21, production training must implement task-balanced
  logical updates with gradient accumulation or equivalent loss aggregation.

  If K=4/K=6 remain close to E21b2, then action readout itself needs a
  stronger but still insulated bridge before production long training.

  If K=4/K=6 regress, stop and inspect code/dataflow before any 30k run.
```

This is not a small-dataset-specific trick.  Fixed exact windows are only a
diagnostic support.  The scalable production principle is:

```text
Each optimizer step should approximate the intended task-mixture gradient.

Physical batch size may remain small, but the logical update must cover a
balanced task/modality/embodiment mixture through micro-batch accumulation,
per-task loss normalization, or a dedicated logical-batch sampler.
```

Relevant external alignment:

```text
VLA Foundry:
  supports probabilistic dataset mixing, batch balancing ratios, gradient
  accumulation, and multi-dataset normalization.  This supports controlled
  mixture construction rather than naive global random batches.

ABot-M0:
  emphasizes cleaning/standardization and compares trajectory-uniform,
  task-uniform, and embodiment-uniform balancing.  This supports treating
  task family as a first-class sampling axis.

PiKE:
  treats batch construction and adaptive task mixing as the central control
  variable under multi-task gradient conflict.  This supports measuring
  gradient conflict after fixed logical-batch baselines are established.

Knowledge Insulation / pi0.5:
  supports separating continuous action expert gradients from the VLM
  backbone.  In current code this is partially covered by
  action_head_and_adapter scope, PICF action-condition disabling, and
  action_context_stopgrad.  It does not remove the need for task-balanced
  logical updates.

OpenVLA-OFT:
  supports chunked continuous action output and stable L1/continuous action
  fine-tuning.  Current PI0.5 action path already uses continuous action
  chunks/flow; the unresolved issue is update composition, not action-token
  autoregression.
```

Code follow-through:

```text
scripts/picf_core_train.py:
  _calvin_prompt_bucket() defines coarse CALVIN task families.
  _CalvinTransitionSource.bucket_to_slot_indices builds bucket membership.
  balanced_bucket_slot_index() cycles buckets by step/rank/micro-step.
  accum_steps controls micro-batch accumulation before optimizer.step().
  tqdm progress exists for production training.

scripts/picf_action_bridge_capacity_probe.py:
  windows_per_step implements exact-window logical batch width.
  --progress now shows a stderr progress bar without corrupting JSON logs.

src/openpi/picf/policy.py:
  picf_action_condition_enabled can disable direct PICF action conditioning.
  action_context_stopgrad prevents PICF context from receiving direct action
  gradients when action context is used.

src/openpi/picf/paligemma/wrapper.py:
  action_context adapter is gated cross-attention into the action suffix.
  action output is chunked continuous action flow, not discrete AR action
  token prediction.
```

Production implication after E24:

```text
If E24 confirms K>=4 or K>=6 is sufficient, the next production run should not
simply increase GPU count.  It should make the optimizer step logical:

  for each update:
    sample K task families without replacement or by temperature weights;
    sample one or more windows per family;
    compute per-family normalized losses;
    accumulate gradients;
    step AdamW once;
    log per-family losses and gradient norms.

AdamW, scheduler, EMA, and checkpoint step counters must advance only once per
logical update.
```

Step30 read:

```text
E24a K=4:
  step0 eval mean = 0.0370217793
  step30 eval mean = 0.0348250197
  step10 recent train = 0.0357231202
  step20 recent train = 0.0350576217
  step30 recent train = 0.0356868175
  step40 recent train = 0.0337900947

E24b K=6:
  step0 eval mean = 0.0370217793
  step30 eval mean = 0.0343819553
  step10 recent train = 0.0364600931
  step20 recent train = 0.0347884616
  step30 recent train = 0.0330343677
```

Interpretation at step30:

```text
K=4/K=6 are valid and running, but they do not reproduce E21's fast descent.
E21 reached 0.0259354621 by step20 on the same exact support.

Therefore the remaining issue is not solved by "slightly more windows/update".
The likely next production ingredient is a true logical optimizer step with
per-task normalized losses and/or conflict-aware weighting, not merely more
physical windows without loss decomposition.

Continue E24 to step60 to rule out delayed convergence before changing code.
```

Step60 read:

```text
E24a K=4:
  step0 eval mean  = 0.0370217793
  step30 eval mean = 0.0348250197
  step60 eval mean = 0.0311698666

  recent train:
    step40 = 0.0337900947
    step50 = 0.0298160967
    step60 = 0.0314307816

E24b K=6:
  step0 eval mean  = 0.0370217793
  step30 eval mean = 0.0343819553
  step40 recent train = 0.0314234595
  step60 eval pending
```

Interpretation at step60:

```text
K=4 is a real positive signal: all-window eval improves monotonically through
step60.

But K=4 still does not match E21's step20 descent to 0.0259354621.  Its
behavior is closer to a slow/noisy approximation of the all-window gradient.

The correct production conclusion is therefore not "small batch is hopeless".
It is:
  naive small batch is insufficient;
  K=4/K=6 logical coverage helps;
  production must make logical optimizer steps explicit and normalized by
  task/modality, otherwise it will keep under-approximating the intended
  multi-task gradient.
```
