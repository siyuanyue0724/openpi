# PICF-AQR-OWM Action Interface EMA Stabilization - 2026-05-27

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/picf_aqr_owm_202605/README.md
```

## 1. Failure Being Repaired

The latest A7 BASL-style gate:

```text
picf_a7_basil_prefixgate_p2_blockcore_action2_30k_20260527
```

again rebounded after the clean 6500/6800 checkpoint handoff:

```text
6850 action_default_equiv = 0.025959
7000 action_default_equiv = 0.021357
7300 action_default_equiv = 0.030595
7350 action_default_equiv = 0.035776
```

The usual structural-collapse signatures were not present:

```text
active/downstream same-role overlap stayed low.
slot-JEPA telemetry did not explode.
anchor PV and object-pull did not explain the action jump.
identity switch did not worsen enough to explain the rebound.
```

This makes the live root cause narrower than generic "multi-loss conflict":

```text
the action-visible PICF prefix is still nonstationary from the PI0.5 action
generator's perspective.
```

## 2. Mathematical Root Cause

Let the PICF router produce an action prefix:

```math
z_t = C_\phi(o_{\le t})
```

and PI0.5/PaliGemma action generation consume:

```math
\hat a_t = \pi_\theta(o_t, z_t).
```

The action loss over an update is:

```math
L_a(\theta,\phi)=\ell(\pi_\theta(o_t,C_\phi(o_{\le t})),a_t).
```

Even when `--picf-action-prefix-stopgrad` prevents direct action gradients
from editing PICF through the current action pass, the next batch sees a new
prefix:

```math
C_{\phi_{k+1}}(o_{\le t}) \ne C_{\phi_k}(o_{\le t}).
```

The local action-loss change contains:

```math
\Delta L_a
\approx
-\eta_\theta\|\nabla_\theta L_a\|^2
+
\left\langle
  \nabla_z L_a,
  C_{\phi_{k+1}}(o)-C_{\phi_k}(o)
\right\rangle
+
\frac12 \Delta z^\top H_z \Delta z.
```

The first term is useful action learning.  The second and third terms are a
moving-interface penalty.  Fixed prefix gates, RMS caps, lower PICF LR, and
block updates reduce the norm of `Delta z`; they do not make the action input
stationary.  That is why core-LR and block-core variants still rebounded.

## 3. Paper-Aligned Repair

The maintained repair follows the target-network / gated-conditioning pattern
used in recent predictive VLA and multimodal systems:

```text
JEPA / V-JEPA:
  online predictors train against detached or EMA targets instead of chasing
  targets that move at the same speed.

JEPA-VLA / VLA-JEPA:
  predictive video representations are useful, but the current action path must
  not receive an unstable future/predictive target as an unguarded input.

Flamingo-style gated cross-attention:
  extra visual evidence is injected through a bounded adapter, preserving the
  pretrained language/action stream.
```

PICF's direct analogue is:

```math
\bar z_k = \beta \bar z_{k-1} + (1-\beta)\,\mathrm{sg}(z_k)
```

and the action generator consumes:

```math
z^{act}_k = \alpha \bar z_k + (1-\alpha) z_k.
```

The online PICF prefix is still trained, but action sees a slow teacher prefix:

```math
L =
L_a(\theta;z^{act})
+
L_{PICF}(\phi)
+
\lambda_{trust}
\left\|
  \mathrm{rmsnorm}(z_k)-\mathrm{rmsnorm}(\mathrm{sg}(\bar z_k))
\right\|_2^2.
```

Important contract:

```text
loss_action_default_equiv remains pure action loss.
loss_action_prefix_trust is an alignment/auxiliary term.
The trust term enters total_minus_action and alignment budget, not action.
```

This keeps all 4-22 ablation comparisons valid.

## 4. Code Deployment

Implemented files:

```text
src/openpi/picf/core/config.py
  action_prefix_teacher_mode
  action_prefix_teacher_ema_decay
  action_prefix_teacher_blend
  lambda_action_prefix_trust

src/openpi/picf/core/pipeline.py
  persistent action_prefix_teacher_tokens buffer
  persistent action_prefix_teacher_initialized buffer

src/openpi/picf/policy.py
  train-time EMA teacher prefix
  RMS-normalized trust loss
  debug metrics injected into output.debug

src/openpi/picf/core/training.py
  action_prefix_trust is a first-class non-action loss component

scripts/picf_core_train.py
  CLI flags
  jsonl metrics
  OWM debug metrics

src/openpi/picf/policy_test.py
  unit test proving second-step action consumes stale EMA teacher prefix
```

New metrics:

```text
loss_action_prefix_trust
pi_prefix_teacher_mode_enabled
pi_prefix_teacher_trust_loss
pi_prefix_teacher_trust_raw
pi_prefix_teacher_delta_rms
pi_prefix_teacher_cos_to_teacher
pi_prefix_teacher_blend
pi_prefix_teacher_ema_decay
```

## 5. Maintained Run Contract

Use the clean pre-rebound checkpoint:

```text
/mnt/checkpoints/picf_core/picf_core/
picf_a7_twotime_cotrain_from_pc1_6000_core005_action2_30k_20260526/6500
```

Recommended live gate:

```text
script:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_actionprefix_ema_from6800_30k_20260527.sh

action_prefix_teacher_mode=ema
action_prefix_teacher_ema_decay=0.99
action_prefix_teacher_blend=1.0
lambda_action_prefix_trust=0.02
action_prefix_norm_mode=rmsnorm
action_prefix_output_gate=0.70
PICF_CORE_LR_SCALE=0.005
PICF_CORE_LR_RUNTIME_MODE=constant
unroll_steps=2
burnin_steps=1
save_interval=1000
keep_last_checkpoints=5
```

Acceptance at 100/300/7350-step gates:

```text
loss_action_default_equiv should not rebound from the 0.021-0.026 band toward 0.035+.
pi_prefix_teacher_delta_rms should stay bounded rather than grow monotonically.
pi_prefix_teacher_cos_to_teacher should not collapse.
loss_action_prefix_trust should be small compared with loss_action_default_equiv.
active/downstream overlap should remain low; raw overlap alone is not a stop signal.
```

## 6. Non-Goals

This repair deliberately does not:

```text
add more raw-overlap penalties;
turn weak object sidecars into hard labels;
revive SAM as training supervision;
freeze PICF permanently;
pollute loss_action_default_equiv with prefix regularization;
replace the PI0.5 action route with a reconstruction/world-model objective.
```

Those alternatives were either already rejected by A5/A7 evidence or do not
target the measured rebound mechanism.

## 7. First Live Gate - A7 6800 to 6900

Run:

```text
picf_a7_actionprefix_ema_from6800_action2_30k_20260527
```

Local/remote validation before launch:

```text
py_compile: pass
policy/train/serve targeted pytest: pass
verify_picf_owm_contract.py: pass
picf_owm_strict_diagnose.py --fail-on-fail: pass
picf_owm_dataflow_trace.py --fail-on-fail: pass
picf_owm_mvtrack_deep_audit.py --fail-on-fail: pass
```

The broader historical `verify_picf_contract.py` passed the full core pytest
suite (`323 passed`) after the runtime-args compatibility fix, but still reports
two pre-existing static contract failures unrelated to this action-interface
repair:

```text
posterior/innovation/task-readout historical order check
live_visual_native_first_competition_path static check
```

Checkpoint compatibility:

```text
Old checkpoints do not contain:
  core.action_prefix_teacher_tokens
  core.action_prefix_teacher_initialized

The loader now allows exactly these missing EMA teacher buffers and
reinitializes optimizer state.  No broad missing/unexpected-key relaxation was
added.
```

Observed first 100 steps after resume:

```text
step 6850:
  loss_total                         0.044297
  loss_action_default_equiv          0.026323
  loss_action_active7                0.118439
  loss_total_minus_action            0.017974
  loss_action_prefix_trust           0.007077
  pi_prefix_teacher_delta_rms        0.373570
  pi_prefix_teacher_cos_to_teacher   0.823078
  active_same_role_overlap_max       0.085894
  downstream_same_role_overlap_max   0.088243
  raw_same_role_overlap_max          1.000000
  loss_anchor_object_pull            0.226191
  posterior_identity_switch_rate     0.186667

step 6900:
  loss_total                         0.038661
  loss_action_default_equiv          0.028029
  loss_action_active7                0.126445
  loss_total_minus_action            0.010632
  loss_action_prefix_trust           0.000701
  pi_prefix_teacher_delta_rms        0.128650
  pi_prefix_teacher_cos_to_teacher   0.982477
  active_same_role_overlap_max       0.085000
  downstream_same_role_overlap_max   0.088482
  raw_same_role_overlap_max          1.000000
  loss_anchor_object_pull            0.151380
  posterior_identity_switch_rate     0.178889
```

Interpretation:

```text
The EMA teacher is behaving as intended over the first 100 steps:
  teacher delta contracts by ~65.6%;
  teacher cosine rises toward 1;
  trust loss becomes negligible relative to action loss.

The action loss is not yet a success proof:
  0.0263 -> 0.0280 is a small early increase, not the old 7300-style rebound.

The structural metrics remain aligned with the repaired design:
  active/downstream overlap stays low;
  raw overlap remains high because reserve/inactive rows overlap, which is not
  action-visible under the current gate.

The next decisive gates are 7000, 7300, and 7350:
  passing 7350 without returning to ~0.035+ is the first strong evidence that
  the measured rebound mechanism was actually removed rather than delayed.
```

## 8. Second Live Gate - A7 7000 to 8300

The 7000 checkpoint was preserved and the live run was stopped after the
8300-step gate:

```text
run:
  picf_a7_actionprefix_ema_from6800_action2_30k_20260527

preserved checkpoints:
  7000
  8000
```

Observed action trajectory:

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

This falsifies the narrow conclusion that action-visible prefix
nonstationarity was the only cause of the late action rebound:

```text
EMA teacher prefix:
  fixed.  Delta collapses to about 0.002 and cosine stays about 0.99999.

Active/downstream support overlap:
  still healthy.  The old raw reserve overlap remains saturated but is not the
  action-visible collapse signal.

Non-action budget:
  stable near 0.010-0.012.

Action:
  rebounds from 0.0213 at step 7000 to 0.0470 at step 8300.
```

Updated diagnosis:

```text
The EMA action-interface repair is correct and should be kept as a stability
mechanism, but it is not sufficient.  The remaining failure is now localized
to action-side optimization/co-training after the prefix has been stabilized.
The next causal split must test whether the preserved 7000 checkpoint continues
to improve when PICF is frozen or when only the action/semantic policy side is
allowed to move.
```

Required next probes:

```text
1. from step7000, freeze PICF core and run action/semantic policy-only.
   If action falls again, the rebound is caused by co-training interference.

2. from step7000, freeze PaliGemma/semantic and train action head only.
   If action falls only in this branch, the large semantic stack is overshooting.

3. from step7000, keep full cotrain but lower action-side LR and keep PICF LR
   near zero.
   If this holds the 0.02-0.03 band, the root is optimizer noise near a sharp
   low-loss basin rather than missing object evidence.
```
