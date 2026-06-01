# PICF-AQR-OWM Two-Timescale Cotrain Final Plan - 2026-05-26

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/picf_aqr_owm_202605/README.md
```

Phase-boundary repair:

```text
docs/PICF_AQR_OWM_ACTION_REBOUND_PHASE_STABILIZATION_20260526.md
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_phase_stabilized_from6500_30k_20260526.sh
```

2026-05-26 update: the `PICF_CORE_LR_SCALE=0.02` branch improved the failed
core=0.05 window but still rebounded by steps 7000/7050/7100.  The maintained
next test is therefore not another scalar overlap penalty.  It is an explicit
phase boundary: freeze PICF core for 300 action-adaptation steps from the clean
6500 checkpoint, then resume cotrain with `PICF_CORE_LR_SCALE=0.01`.

2026-05-28 correction: prior FSDP two-timescale conclusions are suspect unless
the metric row contains `lr_group_picf_core` and `grad_norm_group_picf_core`.
The optimizer was built after FSDP wrapping, and wrapped core names use
`_fsdp_wrapped_module.core.*`; without canonicalizing that prefix, core
parameters could fall through to `policy_head` and ignore
`PICF_CORE_LR_SCALE`.  The two-timescale principle remains the right
mathematical contract, but it must be validated again after the optimizer-group
fix.

This note records the maintained answer to the action-rebound problem under the
principle that cotrain is valuable.  It replaces the temporary policy-only
diagnostic as a production recipe.

## 1. Decision

The final maintained training direction is:

```text
Keep cotrain.
Do not freeze PICF permanently.
Do not let structural losses move PICF at the same speed as the action policy.
Use a two-timescale optimizer:
  PaliGemma / action policy: normal action-learning speed.
  PICF belief router: slow update speed.
  object scaffold: weak shaping with decay floor.
  raw predictive losses: disabled until normalized/matched.
```

This is not a minimal patch.  It is the mathematically consistent version of
the evidence we now have:

```text
PC1 policy-only probe:
  freezing `core.*` removed moving-prefix pressure and action loss improved.

A7 long cotrain:
  action improved early, then rebounded while PICF prefix/state kept moving.

Therefore:
  the defect is not "cotrain is wrong";
  the defect is "same-timescale cotrain makes the action target nonstationary."
```

## 2. Mathematical Root Cause

Let:

```math
z_t = c_{\phi}(x_{\le t})
```

be the action-visible PICF prefix emitted by the belief router, and:

```math
\hat a_t = f_{\theta}(x_t, z_t)
```

be the PI0.5/PaliGemma action path.

The training objective is:

```math
L(\theta,\phi)
=
\lambda_a L_a(\theta; z_{\phi})
+
\lambda_s(t) L_s(\phi)
+
\lambda_r L_r(\theta,\phi)
```

where:

```text
theta: semantic/action policy parameters.
phi: PICF belief-router parameters.
L_a: action flow objective.
L_s: object/slot/sidecar scaffold losses.
L_r: small routing/quality/stability terms.
```

With one optimizer LR for all trainable non-foundation parameters, the action
loss change after one step is approximately:

```math
\Delta L_a
\approx
-\eta_\theta \lVert \nabla_\theta L_a \rVert^2
+
\eta_\phi
\left\langle
  \nabla_z L_a,
  J_{\phi}(z)\nabla_\phi L_s
\right\rangle
+
O(\eta^2)
```

The first term is desired.  The second term is the failure mode.  It is
positive when structural updates move the prefix in a direction the action
head has not adapted to.  This remains true even with
`picf_action_prefix_stopgrad`: stopgrad prevents `L_a` from directly updating
PICF through the prefix, but it does not stop `L_s` from moving the prefix
distribution that the action head consumes.

The correct constraint is therefore not:

```text
freeze PICF forever
```

but:

```math
\eta_\phi
\left|
\left\langle
  \nabla_z L_a,
  J_{\phi}(z)\nabla_\phi L_s
\right\rangle
\right|
\ll
\eta_\theta \lVert \nabla_\theta L_a \rVert^2
```

Operationally, this means:

```text
PICF may keep learning, but it must move slowly once action learning is active.
```

## 3. Paper Alignment

The maintained design is supported by the following pattern in recent work:

```text
JEPA-VLA / VLA-JEPA:
  predictive video embeddings are useful as conditioning signals or detached
  targets, but the action path must avoid future leakage and unstable targets.

OA-WAM:
  object-addressable slots are useful for robust manipulation, but address and
  content must be separated so identity does not drift with appearance.

QASA:
  slot quality and active selection should be decoupled from dense
  reconstruction pressure; reducing active slot count must not fight fidelity.

PCGrad / CAGrad:
  multi-objective training can fail from gradient conflict; the cleanest
  engineering control is to reduce or project conflicting gradients rather than
  pretend all losses have the same timescale.

Flamingo / JEPA-VLA-style gated cross-attention:
  dense context can be injected through a gated pathway instead of forcing all
  residual/background evidence into object slots.
```

PICF's current production path implements the subset that fits the belief
router contract:

```text
object-owner slots:
  active task/contact/motion owner files.

dense context:
  context/reserve routing remains visible as low-weight context, not hard
  object truth.

action prefix:
  RMS-normalized PICF prefix with action stopgrad.

cotrain:
  action/semantic path trainable, PICF core trainable at a lower LR.
```

## 4. Engineering Contract

The new training interface is:

```text
--picf-core-lr-scale <float>
--policy-head-lr-scale <float>
```

Optimizer groups are now:

```text
point_backbone:
  frozen in production recipe.

visual_backbone:
  V-JEPA frozen in production recipe.

tactile_backbone:
  AnyTouch frozen in production recipe.

semantic_backbone:
  PaliGemma/action path.  Uses --semantic-lr-scale.

picf_core:
  all trainable `core.*` belief-router parameters.  Uses
  --picf-core-lr-scale.

policy_head:
  non-semantic, non-PICF policy adapters.  Uses --policy-head-lr-scale.
```

The maintained A7 cotrain test uses:

```text
PICF_TRAINABLE_SCOPE=all
ACTION_LOSS_WEIGHT=2.0
SEMANTIC_TRAINABLE=1
SEMANTIC_LR_SCALE=0.35
PICF_CORE_LR_SCALE=0.05
POLICY_HEAD_LR_SCALE=1.0
LR=7e-5
MIN_LR=2e-5
WARMUP_STEPS=20
OBJECT_SCAFFOLD_DECAY_FLOOR=0.03
SAVE_INTERVAL=500
KEEP_LAST_CHECKPOINTS=3
LOG_INTERVAL=50
ANCHOR_OVERLAY_INTERVAL=100
```

The resulting effective LR scale is:

```text
PaliGemma/action path:
  7e-5 * 0.35 = 2.45e-5

PICF core:
  7e-5 * 0.05 = 3.5e-6
```

This preserves cotrain while making the prefix roughly seven times slower than
the action/semantic path.  It is deliberately not `policy_only`; all `core.*`
parameters remain trainable.

## 5. Loss Contract

Enabled:

```text
action flow:
  lambda_action_pos/rot/gripper = 2.0

weak object ownership:
  lambda_anchor_object_pull = 0.35, decayed by scaffold floor.

slot quality:
  weak QASA-style active-quality pressure.

object explanation:
  point/contact/duplicate/background terms remain weak and quality-gated.

context routing:
  active/context/reserve split stays enabled.
```

Disabled:

```text
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
blind SAM proposal sidecars = rejected
legacy local refinement = off by default
```

Reason:

```text
Raw predictive losses are still useful telemetry, but previous runs showed
their target scale can explode.  They should become production losses only
after normalized matched targets and future-detach acceptance pass.
```

## 6. Acceptance Bundle

The run is healthy only if all groups agree:

```text
Action:
  loss_action_default_equiv should not rebound against its own local minimum
  for several 50-step gates.

Prefix:
  pi_prefix_post_rms_mean should stay near 1.0.

Active slots:
  aqr_active_same_role_support_overlap_max < 0.25 preferred;
  aqr_downstream_same_role_support_overlap_max < 0.25 preferred.

Reserve/raw:
  aqr_same_role_support_overlap_max may be high because reserve/context rows
  are included.  It is a watch metric, not the primary failure signal.

Posterior lifecycle:
  posterior_recycle_rate and posterior_identity_switch_rate should not rise
  together with action rebound.

Auxiliary budget:
  loss_total_minus_action should remain bounded and not grow while action
  plateaus.

Visual evidence:
  active-only overlays must keep task/contact objects in active owner files.
```

## 7. Rejection Conditions

Stop and redesign if:

```text
action_default_equiv rebounds >30% from a local minimum while
loss_total_minus_action is stable;

or active/downstream overlap rises above 0.25 for multiple logs;

or posterior lifecycle metrics worsen together with action rebound;

or overlays show active owner files leaving task/contact objects;

or disabled raw slot-JEPA telemetry explodes together with prefix statistics.
```

## 8. Why This Is Not A Patchwork Fix

Rejected alternatives:

```text
Permanent policy_only:
  proves moving-prefix causality but removes PICF cotrain value.

Higher action weight only:
  can hide prefix drift by overpowering it, but does not control the moving
  conditioning distribution.

More duplicate penalties only:
  previous experiments showed active overlap can be controlled while action
  still rebounds.

Blind SAM:
  adds noisy object proposals and was rejected by visual inspection and
  diagnostics.

Full reconstruction decoder:
  may be useful later as a detached auxiliary pretraining route, but it would
  turn the belief router into a pixel reconstructor and compete with the PI0.5
  action path if enabled directly.
```

The maintained fix addresses the actual coupled system:

```text
the action path sees PICF prefix;
the prefix must remain useful and slowly adaptive;
object evidence may shape the prefix;
action must remain the dominant production objective.
```

## 9. Deployment Script

Use:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_twotimescale_cotrain_from_pc1_6000_30k_20260526.sh
```

Expected first gate:

```text
step 100:
  verify optimizer groups:
    semantic_backbone lr_scale=0.35
    picf_core lr_scale=0.05
    policy_head lr_scale=1.0 if present

step 300:
  compare action trend against A7 old run and PC1 policy-only causal probe.

step 500:
  if action still improves and active metrics remain healthy, allow long run
  to continue.
```

## 10. 6500-Step Controlled Branch

Observed A7 core=0.05 gates:

```text
step 6200:
  action_default_equiv = 0.02970
  active_overlap       = 0.0950
  downstream_overlap   = 0.0942
  identity_switch      = 0.1744

step 6500:
  action_default_equiv = 0.03792
  active_overlap       = 0.0717
  downstream_overlap   = 0.0752
  identity_switch      = 0.1933

step 6700:
  action_default_equiv = 0.03859
  active_overlap       = 0.0700
  downstream_overlap   = 0.0648
  identity_switch      = 0.2344
```

Interpretation:

```text
The old same-role support collapse is not the active failure:
  active/downstream overlap remains low.

The remaining failure is moving identity/belief state:
  identity_switch rises while action rebounds.

Therefore the next controlled intervention is not higher action weight and not
another duplicate penalty.  It is a slower PICF core timescale.
```

Controlled branch:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_twotimescale_cotrain_from6500_core002_30k_20260526.sh

resume:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_twotime_cotrain_from_pc1_6000_core005_action2_30k_20260526/6500

changed:
  PICF_CORE_LR_SCALE=0.02

unchanged:
  ACTION_LOSS_WEIGHT=2.0
  SEMANTIC_LR_SCALE=0.35
  POLICY_HEAD_LR_SCALE=1.0
  LR=7e-5
  MIN_LR=2e-5
  optimizer_checkpoint_mode=model-only
  sidecar root and scaffold decay settings
```

This makes the nominal effective LRs:

```text
PaliGemma/action semantic path:
  7e-5 * 0.35 = 2.45e-5

PICF core:
  7e-5 * 0.02 = 1.40e-6

Relative semantic/action-to-PICF speed:
  17.5x
```

Acceptance against the failed 6500->6700 window:

```text
Required:
  action_default_equiv should stay below the core=0.05 branch trend;
  identity_switch should stop rising toward 0.23+;
  active/downstream overlap should remain <0.15.

Preferred:
  action_default_equiv returns toward <=0.034 by 300-500 branch steps;
  loss_total_minus_action remains bounded near 0.010-0.013;
  slot_jepa telemetry does not re-spike above 2.3.

Reject:
  action remains >=0.038 for several logs while identity_switch remains >=0.23;
  or active/downstream overlap rises above 0.25;
  or prefix RMS deviates from 1.0.
```

If rejected, the next principled branch is:

```text
freeze PICF core for 300-500 steps from the same 6500 checkpoint;
then unfreeze with PICF_CORE_LR_SCALE=0.01-0.02.
```

That is still the same mathematical fix: make the action-visible belief state
slow enough for the action path to adapt, without abandoning cotrain.
