# PICF-AQR-OWM Action Rebound Phase Stabilization - 2026-05-26

Canonical index:

```text
src/openpi/picf/README_v2.2.md
docs/picf_aqr_owm_202605/README.md
docs/PICF_AQR_OWM_COTRAIN_TWO_TIMESCALE_FINAL_20260526.md
```

## 1. Current Failure

The controlled A7 branch

```text
picf_a7_twotime_cotrain_from6500_core002_action2_30k_20260526
```

changed only:

```text
PICF_CORE_LR_SCALE: 0.05 -> 0.02
```

and resumed from:

```text
/mnt/checkpoints/picf_core/picf_core/
  picf_a7_twotime_cotrain_from_pc1_6000_core005_action2_30k_20260526/6500
```

Observed gates:

```text
step 6700:
  action_default_equiv = 0.02233
  active_overlap       = 0.1050
  downstream_overlap   = 0.1086
  identity_switch      = 0.1989

step 7000:
  action_default_equiv = 0.03090
  active_overlap       = 0.1150
  downstream_overlap   = 0.1246
  identity_switch      = 0.1961

step 7050:
  action_default_equiv = 0.03558
  active_overlap       = 0.1000
  downstream_overlap   = 0.1074
  identity_switch      = 0.1861

step 7100:
  action_default_equiv = 0.03492
  active_overlap       = 0.1356
  downstream_overlap   = 0.1440
  identity_switch      = 0.2078
```

This is not the historical slot-collapse failure:

```text
active/downstream same-role overlap remains far below 0.9.
slot_jepa telemetry is not exploding.
prefix RMS normalization remains exactly active.
```

The remaining failure is action rebound under a still-moving action-visible
belief prefix.

## 2. Mathematical Root Cause

Let the PICF belief router expose an action prefix:

```math
z_t = c_\phi(o_{\leq t})
```

and the PI0.5/PaliGemma action path predict:

```math
\hat a_t = f_\theta(o_t, z_t).
```

The current cotrain objective is:

```math
L(\theta, \phi)
=
\lambda_a L_a(\theta; z_\phi)
+ \lambda_s(t)L_s(\phi)
+ \lambda_q L_q(\theta,\phi).
```

Even with action-prefix stop-gradient, the next optimization window sees:

```math
z_{\phi_{k+1}}(o)
\neq
z_{\phi_k}(o).
```

For the action objective, the first-order change contains:

```math
\Delta L_a
\approx
-\eta_\theta ||\nabla_\theta L_a||^2
+
\left\langle
  \nabla_z L_a,
  z_{\phi_{k+1}} - z_{\phi_k}
\right\rangle.
```

The first term is the desired action improvement.  The second term is the
moving-prefix penalty.  Lowering `PICF_CORE_LR_SCALE` reduces the second term,
but the 7000/7050/7100 gates show it does not remove it.

The root cause is therefore:

```text
same-phase cotrain still lets the action-visible belief prefix move before the
action head has fully adapted to it.
```

This is more specific than generic "multi-loss conflict".  Raw overlap is not
the current dominant cause because active/downstream overlap stays bounded.
Object scaffold noise is a contributor only insofar as it moves `z_phi` through
PICF structural updates.

## 3. Paper-Aligned Interpretation

Recent methods support a staged or target-network view rather than same-speed
end-to-end pressure:

```text
JEPA / V-JEPA:
  predictive targets are stabilized with detached or EMA target encoders.

JEPA-VLA / Flamingo-style gated fusion:
  pretrained action/VLA priors are protected by gated conditioning pathways
  instead of forcing a rapidly changing dense representation directly into the
  main action stream.

VLA-JEPA:
  future/predictive latents are supervision targets, not unstable inputs to
  the current action path.

QASA / MetaSlot:
  quality selection and duplicate/no-object control are necessary, but they do
  not by themselves solve downstream action-prefix nonstationarity.
```

PICF already has:

```text
active/context quality selection,
duplicate controls,
prefix RMS normalization,
action-prefix stop-gradient,
two-timescale LR groups.
```

The missing production element is a phase boundary:

```text
first let the action path adapt to a stationary PICF prefix;
then resume PICF cotrain on a slower timescale.
```

This is the practical analogue of an EMA/target-network schedule without
doubling inference or training compute.

## 4. Maintained Repair

Use a two-phase run from the clean pre-rebound checkpoint.

### Phase 1: Prefix Stationarization

```text
resume checkpoint:
  old core=0.05 run, step 6500

trainable:
  semantic/action policy path

frozen:
  PICF core
  frozen perception backbones remain frozen

implementation note:
  the run script passes PICF_CORE_LR_SCALE=1e-6 only because the trainer
  validates this flag as positive. PICF_TRAINABLE_SCOPE=policy_only is the
  actual freeze contract.

objective:
  action only

duration:
  300 optimizer steps, 6500 -> 6800
```

This phase directly removes `z_phi` drift:

```math
z_{\phi_{k+1}} = z_{\phi_k}
```

up to semantic-token changes already tested by the policy-only causal probe.

### Phase 2: Slow Cotrain

```text
resume checkpoint:
  phase-1 step 6800

trainable:
  semantic/action policy path
  PICF core

PICF_CORE_LR_SCALE:
  0.01

ACTION_LOSS_WEIGHT:
  2.0

checkpoint policy:
  save every 1000 steps, keep latest 5 checkpoints

object scaffold:
  weak, decayed floor
```

This keeps cotrain value but enforces:

```math
||z_{\phi_{k+1}} - z_{\phi_k}||
\ll
||\theta_{k+1} - \theta_k||.
```

## 5. Why This Is Not Patchwork

Rejected alternatives:

```text
Only lower action LR:
  slows the desired term and does not stabilize z_phi.

Only increase action weight:
  can hide prefix drift temporarily but does not reduce the moving-prefix term.

Only add duplicate/overlap penalties:
  active/downstream overlap is already bounded in the failing window.

Permanent policy_only:
  removes PICF cotrain value.

Blind extra modules:
  do not target the measured cause.
```

The phase-stabilized schedule changes the variable responsible for the
measured failure: the rate at which the action-visible belief prefix moves.

## 6. Acceptance

Phase 1 must show by 6600/6650:

```text
action_default_equiv <= 0.030
loss_total_minus_action near 0
prefix RMS finite and normalized
```

Phase 2 must show by 6900/7000:

```text
action_default_equiv does not rebound above 0.032 for consecutive logs;
active_overlap < 0.20;
downstream_overlap < 0.20;
identity_switch does not rise toward 0.23+;
loss_total_minus_action remains bounded near 0.010-0.013.
```

If rejected:

```text
freeze PICF for a longer 500-step prefix stationarization,
or keep PICF_CORE_LR_SCALE=0.005-0.01 for phase 2.
```

Do not add new object losses until this phase-boundary test is complete.

## 7. Local Verification Before A7 Restart

Executed on 2026-05-26 before restarting A7:

```text
bash -n scripts/experiments/picf_aqr_owm_202605_active/run_a7_phase_stabilized_from6500_30k_20260526.sh
python3 -m py_compile scripts/picf_core_train.py src/openpi/picf/core/pipeline.py src/openpi/picf/policy.py
python3 scripts/picf_latest_slot_deployment_audit.py --repo-root . --fail-on-fail
python3 scripts/verify_picf_owm_contract.py
python3 scripts/picf_binding_dataflow_math_audit.py --fail-on-fail
python3 scripts/picf_oeml_dataflow_audit.py --fail-on-fail
python3 scripts/picf_owm_dataflow_trace.py --fail-on-fail
python3 scripts/picf_owm_strict_diagnose.py --fail-on-fail
```

Result:

```text
script syntax: pass
critical Python compile: pass
latest-slot deployment audit: 16/16 PASS
OWM contract verifier: PASS
binding dataflow/math audit: PASS
OEML dataflow audit: 6/6 PASS
OWM dataflow trace: PASS
strict diagnosis: PASS
```

Launch note:

```text
The first restart attempt failed before training because the trainer rejected
PICF_CORE_LR_SCALE=0.0.  The script now passes 1e-6 in phase 1 only to satisfy
the positive-value validator; PICF_TRAINABLE_SCOPE=policy_only is the actual
core-freeze contract.
```

Remote continuity note:

```text
The A7 restart runs a guard session named picf_a7_phase2_guard_20260526.
It waits for the phase-1 6800 checkpoint and metrics row, then starts the
phase-2-only session with SAVE_INTERVAL=1000 and KEEP_LAST_CHECKPOINTS=5.
This prevents the run from stopping at 6800 and prevents older 500/keep3
defaults from leaking into the 30000-step cotrain.
```

First A7 restart evidence:

```text
6550:
  loss_total = loss_action_default_equiv = 0.02877
  loss_total_minus_action = 0.0
  active_support_overlap_max = 0.130
  downstream_support_overlap_max = 0.139

6600:
  loss_total = loss_action_default_equiv = 0.02979
  loss_total_minus_action = 0.0
  active_support_overlap_max = 0.109
  downstream_support_overlap_max = 0.107

Interpretation:
  phase 1 is correctly isolating the action/semantic adaptation against a
  stationary PICF core. Raw reserve overlap remains a reserve-capacity
  diagnostic, not an action-visible failure.
```

## 8. Phase-2 Watch Contract To 7300

User request on 2026-05-27:

```text
Do not stop at 6800.  Confirm phase 2 enters normal operation and watch until
at least step 7300.  If phase 2 is healthy, leave it running.  If the old
rebound recurs, stop, diagnose, patch the root cause, and restart.
```

Current observed state when this watch began:

```text
time: 2026-05-27 00:53 CST
phase 1 progress: about 6635 / 6800
phase 2: not started yet
guard: running, waiting for the 6800 checkpoint and metrics row
```

Expected timing:

```text
6800 checkpoint:
  about 2026-05-27 02:00 CST

phase 2 first metric:
  about 2026-05-27 02:25-02:35 CST, depending on checkpoint startup time

7300 metric:
  about 2026-05-27 05:20-05:45 CST
```

Phase-2 health criteria:

```text
must hold:
  phase2 session exists and writes metrics
  save_interval = 1000
  keep_last_checkpoints = 5
  pi_prefix_post_rms_mean stays near 1.0
  loss_total_minus_action stays bounded near the intended weak-core scaffold
  active/downstream same-role support overlap stays below 0.20

action rebound rejection:
  loss_action_default_equiv rising above about 0.032 for consecutive phase-2
  metrics before 7300, unless all structure metrics are simultaneously improving
  and the action rise is a single-window fluctuation.

identity rejection:
  posterior_identity_switch_rate rising toward the old 0.23+ failure band while
  action also rebounds.

if rejected:
  stop phase 2, resume from the 6800 stationarized checkpoint, and try either
  a longer policy-only stationarization or a lower PICF_CORE_LR_SCALE
  (0.005-0.01). Do not add new object/slot losses as a first response.
```

2026-05-27 01:03 CST watch update:

```text
phase 1 session:
  picf_a7_phase_stabilized_20260526

phase 2 guard:
  picf_a7_phase2_guard_20260526

phase 2 session:
  not started yet

latest phase-1 rows:
  6550:
    loss_total = 0.02877
    loss_action_default_equiv = 0.02877
    loss_total_minus_action = 0.0
    active_support_overlap_max = 0.130
    downstream_support_overlap_max = 0.139
    posterior_identity_switch_rate = 0.222
    pi_prefix_post_rms_mean = 1.0000005

  6600:
    loss_total = 0.02979
    loss_action_default_equiv = 0.02979
    loss_total_minus_action = 0.0
    active_support_overlap_max = 0.109
    downstream_support_overlap_max = 0.107
    posterior_identity_switch_rate = 0.232
    pi_prefix_post_rms_mean = 1.0000005

  6650:
    loss_total = 0.02851
    loss_action_default_equiv = 0.02851
    loss_total_minus_action = 0.0
    active_support_overlap_max = 0.130
    downstream_support_overlap_max = 0.129
    posterior_identity_switch_rate = 0.215
    pi_prefix_post_rms_mean = 1.0000005

interpretation:
  phase 1 remains a clean policy-only stationarization window.  There is no
  action-visible structural loss pressure because loss_total_minus_action is
  zero.  Active/downstream overlap is low; raw overlap remains reserve-only.
  The guard has not switched yet because the 6800 checkpoint does not exist.
```

2026-05-27 05:56 CST phase-2 watch result:

```text
phase 2 launch:
  guard detected 6800 checkpoint at 2026-05-27 02:04 CST
  phase2 tmux launched at 2026-05-27 02:04 CST
  session: picf_a7_phase_stabilized_phase2_20260526

phase-2 metrics:
  6850:
    loss_action_default_equiv = 0.02596
    loss_total_minus_action = 0.01088
    active_support_overlap_max = 0.145
    downstream_support_overlap_max = 0.144
    posterior_identity_switch_rate = 0.228
    pi_prefix_post_rms_mean = 1.0000005

  6900:
    loss_action_default_equiv = 0.02811
    loss_total_minus_action = 0.01128
    active_support_overlap_max = 0.085
    downstream_support_overlap_max = 0.086
    posterior_identity_switch_rate = 0.222

  6950:
    loss_action_default_equiv = 0.02682
    loss_total_minus_action = 0.01206
    active_support_overlap_max = 0.190
    downstream_support_overlap_max = 0.184
    posterior_identity_switch_rate = 0.216

  7000:
    loss_action_default_equiv = 0.02130
    loss_total_minus_action = 0.01144
    active_support_overlap_max = 0.110
    downstream_support_overlap_max = 0.112
    posterior_identity_switch_rate = 0.202

  7050:
    loss_action_default_equiv = 0.02602
    loss_total_minus_action = 0.01140
    active_support_overlap_max = 0.150
    downstream_support_overlap_max = 0.155
    posterior_identity_switch_rate = 0.193

  7100:
    loss_action_default_equiv = 0.02644
    loss_total_minus_action = 0.01200
    active_support_overlap_max = 0.100
    downstream_support_overlap_max = 0.112
    posterior_identity_switch_rate = 0.213

  7150:
    loss_action_default_equiv = 0.02776
    loss_total_minus_action = 0.01153
    active_support_overlap_max = 0.175
    downstream_support_overlap_max = 0.173
    posterior_identity_switch_rate = 0.213

  7200:
    loss_action_default_equiv = 0.02679
    loss_total_minus_action = 0.01047
    active_support_overlap_max = 0.141
    downstream_support_overlap_max = 0.124
    posterior_identity_switch_rate = 0.220

  7250:
    loss_action_default_equiv = 0.02792
    loss_total_minus_action = 0.01115
    active_support_overlap_max = 0.135
    downstream_support_overlap_max = 0.136
    posterior_identity_switch_rate = 0.221

  7300:
    loss_action_default_equiv = 0.03056
    loss_total_minus_action = 0.01133
    active_support_overlap_max = 0.070
    downstream_support_overlap_max = 0.067
    posterior_identity_switch_rate = 0.189
    pi_prefix_post_rms_mean = 1.0000005

decision:
  phase 2 entered normal operation and passed the 7300 watch gate.  The 7300
  action value is a single-window rise but remains below the rejection threshold
  and happens while active/downstream overlap and identity switch improve
  sharply.  This is not the old coupled failure mode where action rebounds
  together with structure instability.

next action:
  leave phase2 running.  Re-check around 7600-7800 and then at the 8000
  checkpoint.  If action rises above 0.032 for consecutive metrics while
  active/downstream overlap or identity switch also deteriorates, resume from
  6800 and reduce core update pressure further before adding any new losses.
```

2026-05-27 07:02 CST action-specific rebound addendum:

```text
additional watch:
  7350:
    loss_action_default_equiv = 0.03589
    loss_total_minus_action = 0.00968
    active_support_overlap_max = 0.0468
    downstream_support_overlap_max = 0.0448
    posterior_identity_switch_rate = 0.188

  7400:
    loss_action_default_equiv = 0.03491
    loss_total_minus_action = 0.01258
    active_support_overlap_max = 0.160
    downstream_support_overlap_max = 0.164
    posterior_identity_switch_rate = 0.211

  7450:
    loss_action_default_equiv = 0.03808
    loss_total_minus_action = 0.00978
    active_support_overlap_max = 0.0500
    downstream_support_overlap_max = 0.0585
    posterior_identity_switch_rate = 0.191

interpretation:
  This is not the old coupled failure where action rebound appears together
  with active/downstream support collapse.  The structure metrics are healthy.
  However, action loss is above the 0.032 soft gate for three consecutive
  50-step metrics.  That is an action-specific rebound and should not be
  left running as the maintained candidate.

root-cause-consistent response:
  do not add object/slot losses;
  stop the current phase-2 core0.01 run;
  resume from the same 6800 stationarized checkpoint with lower PICF core
  update pressure:

    PICF_CORE_LR_SCALE = 0.005

  Keep action weight, prefix normalization, sidecar data, unroll/burn-in,
  save policy, and trainable scope unchanged.  This isolates the only
  suspected variable: action-visible prefix drift caused by too-fast core
  updates in phase 2.
```

2026-05-27 10:38 CST core0.005 replacement result:

```text
old phase2 stopped:
  session: picf_a7_phase_stabilized_phase2_20260526
  reason: action-specific rebound at 7350/7400/7450 despite healthy structure

new phase2 launched:
  session: picf_a7_phase_stabilized_phase2_core0005_20260527
  exp: picf_a7_phase_stabilized_p2_core0005_action2_30k_20260527
  resume: phase1 6800 checkpoint
  PICF_CORE_LR_SCALE = 0.005
  trainable_scope = all
  save_interval = 1000
  keep_last_checkpoints = 5
  unroll_steps = 2
  burnin_steps = 1

core0.005 watch:
  6850:
    loss_action_default_equiv = 0.02598
    loss_total_minus_action = 0.01103
    active_support_overlap_max = 0.095
    downstream_support_overlap_max = 0.115
    posterior_identity_switch_rate = 0.198

  6900:
    loss_action_default_equiv = 0.02814
    loss_total_minus_action = 0.01039
    active_support_overlap_max = 0.080
    downstream_support_overlap_max = 0.077
    posterior_identity_switch_rate = 0.207

  6950:
    loss_action_default_equiv = 0.02683
    loss_total_minus_action = 0.00951
    active_support_overlap_max = 0.080
    downstream_support_overlap_max = 0.083
    posterior_identity_switch_rate = 0.178

  7000:
    loss_action_default_equiv = 0.02128
    loss_total_minus_action = 0.01085
    active_support_overlap_max = 0.105
    downstream_support_overlap_max = 0.115
    posterior_identity_switch_rate = 0.179

  7050:
    loss_action_default_equiv = 0.02603
    loss_total_minus_action = 0.01171
    active_support_overlap_max = 0.143
    downstream_support_overlap_max = 0.152
    posterior_identity_switch_rate = 0.186

  7100:
    loss_action_default_equiv = 0.02651
    loss_total_minus_action = 0.01018
    active_support_overlap_max = 0.059
    downstream_support_overlap_max = 0.066
    posterior_identity_switch_rate = 0.196

  7150:
    loss_action_default_equiv = 0.02779
    loss_total_minus_action = 0.01120
    active_support_overlap_max = 0.110
    downstream_support_overlap_max = 0.111
    posterior_identity_switch_rate = 0.188

  7200:
    loss_action_default_equiv = 0.02686
    loss_total_minus_action = 0.01085
    active_support_overlap_max = 0.085
    downstream_support_overlap_max = 0.097
    posterior_identity_switch_rate = 0.194

  7250:
    loss_action_default_equiv = 0.02790
    loss_total_minus_action = 0.01056
    active_support_overlap_max = 0.090
    downstream_support_overlap_max = 0.101
    posterior_identity_switch_rate = 0.202

  7300:
    loss_action_default_equiv = 0.03060
    loss_total_minus_action = 0.01149
    active_support_overlap_max = 0.104
    downstream_support_overlap_max = 0.121
    posterior_identity_switch_rate = 0.192
    pi_prefix_post_rms_mean = 1.0000005

decision:
  core0.005 phase2 passed the 7300 gate.  It did not reproduce the old
  core0.01 action-specific rebound before 7300.  The maintained run should be
  core0.005, not core0.01.

next action:
  leave core0.005 running.  Re-check around 7600-7800 and at the 8000
  checkpoint.  If action rises above 0.032 for three consecutive logged
  metrics while structure remains healthy, treat it as action-timescale
  mismatch and reduce core pressure again before adding losses.  If action
  rise couples with active/downstream overlap or identity switch, revert to
  the 6800 checkpoint and inspect dataflow before resuming.
```

## 9. Structural Fix If Core-LR Scaling Is Not Enough

Core LR scaling is a root-cause-consistent mitigation, but it is not a
mathematical closure of the rebound problem.  The real failure variable is not
the scalar learning rate itself; it is the action-visible belief-prefix drift:

```math
\Delta L_a
\approx
-\eta_\theta \|\nabla_\theta L_a\|^2
+
\left\langle
  \nabla_z L_a,\,
  z_{\phi_{k+1}} - z_{\phi_k}
\right\rangle .
```

The first term is the policy/action update helping the action loss.  The
second term is PICF moving the prefix consumed by PI0.5.  Lowering
`PICF_CORE_LR_SCALE` only shrinks this term indirectly.  If action rebounds
again while active/downstream overlap and identity metrics stay healthy, the
required structural fix is:

```text
Belief Action Stabilization Layer (BASL)
  1. maintain an EMA/teacher action-visible prefix z_bar
  2. expose the action model to a gated residual prefix
       z_action = z_bar + g * LN(P(z_online) - stopgrad(z_bar))
  3. constrain prefix drift
       L_trust = max(0, RMS(z_online - stopgrad(z_bar)) - tau)^2
  4. adapt g and/or PICF core update pressure when action rises
  5. do not change object/slot losses unless structure metrics also fail
```

This follows the same principle as modern VLA/world-model integration:

```text
world/belief features are useful, but they should enter the action pathway
through a controlled residual/gated interface, not as an uncontrolled moving
prefix.
```

The second structural fix is an auxiliary-gradient quarantine:

```text
For each logged window:
  estimate action-gradient direction on z_action
  estimate auxiliary/belief-gradient direction on the same prefix
  if cosine(action, aux) < 0:
    downweight belief/core auxiliary pressure for that window
  else:
    allow normal slow cotrain
```

This is a prefix-level, cheap variant of gradient-conflict handling.  It is
preferable to full per-parameter PCGrad/CAGrad for this codebase because the
model is large and FSDP-heavy; the conflict that matters here is specifically
the action-visible prefix, not every shared parameter.

The third structural fix is optional but likely useful for inference and
future large multimodal scaling:

```text
Unassigned dense context should not be forced into object slots.
Keep dense V-JEPA/context tokens available through a small gated cross-attention
adapter, while object slots carry high-confidence object belief.
```

This preserves dense world-model information while preventing reserve/inactive
anchors from polluting the action prefix.

Confirmed update on 2026-05-27:

```text
core0.005 phase2 reached the same action-specific failure:

  7350:
    loss_action_default_equiv = 0.03556
    loss_total_minus_action = 0.01110
    active_support_overlap_max = 0.080
    downstream_support_overlap_max = 0.088
    posterior_identity_switch_rate = 0.187
    pi_prefix_post_rms_mean = 1.0000005

The rebound reappeared while structure stayed healthy.  This falsifies
"core LR scale alone is enough" and promotes BASL/gated-prefix scheduling from
optional vNext to the next maintained repair.
```

Implemented repair contract:

```text
1. Action extra-prefix output gate
   Config:
     action_prefix_output_gate

   Runtime:
     z_action = gate * normalize(z_picf)

   Purpose:
     bounds the perturbation from the moving PICF belief prefix.  This is the
     direct analogue of gated VLA conditioning; it is not a new object loss.

2. Block-alternating PICF core LR
   CLI:
     --picf-core-lr-runtime-mode block_alternating
     --picf-core-lr-block-start-step
     --picf-core-lr-block-cycle-steps
     --picf-core-lr-block-active-steps

   Runtime:
     core updates only during short active windows; policy/action sees long
     stationary-prefix adaptation windows between core updates.

3. Metrics:
     pi_prefix_gate_mean/min/max
     picf_core_lr_runtime_multiplier
     picf_core_lr_effective_scale

Maintained script:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_basil_prefixgate_from6500_30k_20260527.sh

Default structural settings:
  ACTION_PREFIX_OUTPUT_GATE=0.70
  PICF_CORE_LR_SCALE=0.01
  PICF_CORE_LR_RUNTIME_MODE=block_alternating
  PICF_CORE_LR_BLOCK_CYCLE_STEPS=200
  PICF_CORE_LR_BLOCK_ACTIVE_STEPS=40
```

Revised decision rule:

```text
If BASL/gated-prefix keeps action below 0.032 through 7350-7600 and keeps
active/downstream overlap below 0.25:
  continue long training.

If action still rebounds while structure remains healthy:
  the missing piece is not another slot/object loss; it is a stronger
  sample-conditioned prefix target, likely a teacher-prefix or cached-prefix
  trust-region from the last stable checkpoint.

If action rebounds with structure degradation:
  revert to object/dataflow diagnostics; BASL is no longer the only variable.
```
