# PICF-AQR-OWM Action Rebound Root-Cause Audit - 2026-05-23

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_SCALABILITY_AND_CALVIN_REPAIR_PLAN_20260523.md
```

This note records the current diagnosis for the late-run action-loss rebound in
the May-22 action-aware PICF runs.  It is intentionally evidence-first: it does
not declare CALVIN behavior acceptance and it does not treat a disabled
telemetry loss as an active training objective.

## 1. Runs Compared

Reference PI0.5 ablation:

```text
/mnt/checkpoints/picf_core/picf_core/
  picf_v22_ablated_pi05_30000_ckpt2500_print100_20260422_r2/metrics.jsonl
```

PICF action-aware family:

```text
picf_a7_actionaware_defaultsync_long30k_ckpt500_20260521
picf_a7_actionaware_defaultsync_action2_from500_long30k_20260521
picf_a7_actionaware_defaultsync_action2_from0_long30k_20260522
picf_a7_actionaware_qgdecay_from500_long30k_20260522
picf_a7_actionaware_qgfloor003_from1500_long30k_20260522
```

CALVIN behavior gate:

```text
picf_a7_actionaware_ckpt2000_3000_video_eval_20260523
picf_a7_actionaware_ckpt2000_3000_nanfix_observe1_20260523
```

## 2. Key Numeric Evidence

Old 4-22 PI0.5 ablation:

```text
step 100:   loss_action ~= 0.24169
step 200:   loss_action ~= 0.09765
step 300:   loss_action ~= 0.07379
step 500:   loss_action ~= 0.06343
step 19300: loss_action ~= 0.01839
step 20000: loss_action ~= 0.02137
```

Best May-22 PICF action window before rebound:

```text
run:
  picf_a7_actionaware_qgfloor003_from1500_long30k_20260522

step 1550: action_default_equiv ~= 0.03574
step 1700: action_default_equiv ~= 0.02685
step 1900: action_default_equiv ~= 0.02871
```

Late rebound in the same run:

```text
step 2000: action_default_equiv ~= 0.03372
step 2250: action_default_equiv ~= 0.04975
step 2500: action_default_equiv ~= 0.05118
step 3000: action_default_equiv ~= 0.05281
step 3150: action_default_equiv ~= 0.05366
```

This is a real regression relative to the run's own best window:

```text
0.02685 -> 0.05366 ~= 99.9% rebound
```

It is not yet a proof that the architecture cannot train, but it invalidates
the claim that the current long-run loss trajectory is clean.

## 3. What Is Not The Primary Cause

### 3.1 Learning Rate

The rebound occurs while LR is nearly flat and slightly decreasing:

```text
step 1700 lr ~= 1.9866e-4
step 3150 lr ~= 1.9529e-4
```

The change is only about `1.7%`.  A pure LR-too-high explanation would normally
show stronger gradient instability or broad objective oscillation.  Here
`preclip_grad_norm` is not exploding in the rebound tail:

```text
step 1700 preclip_grad_norm ~= 0.405
step 3150 preclip_grad_norm ~= 0.249
```

Conclusion: LR may still be tunable, but it is not the root explanation for the
late action rebound.

### 3.2 Weak Object Scaffold Dominance

Earlier action-2.0 runs showed the sidecar scaffold becoming larger than the
action objective.  That was why qgdecay/qgfloor was introduced.

In the qgfloor tail, however, the scaffold is already weak:

```text
step 1700:
  action_default_equiv ~= 0.02685
  anchor_object_pull raw ~= 0.157

with floor ~=0.03 and lambda_anchor_object_pull ~=0.35:
  weighted anchor_object_pull ~= 0.00165
```

The weak object teacher is no longer large enough to explain a jump from
`0.02685` to `0.05366`.

Conclusion: qgfloor fixed the earlier scaffold-dominance problem, but did not
fix the later recurrent/prefix stability problem.

### 3.3 Raw Same-Role Overlap

Raw `aqr_same_role_support_overlap_max` is often saturated at `1.0`, but the
active-object and downstream metrics are the relevant control-path signals.

In the qgfloor rebound window:

```text
aqr_active_same_role_support_overlap_max:
  mostly 0.00-0.075

aqr_downstream_same_role_support_overlap_max:
  mostly 0.07-0.23
```

Correlation against action loss in qgfloor:

```text
posterior_recycle_rate:                 +0.851
aqr_downstream_same_role_support_overlap +0.177
aqr_active_same_role_support_overlap     -0.637
```

Conclusion: raw overlap remains a noisy reserve/context telemetry issue.  It is
not the best causal handle for the late action rebound.

## 4. Most Likely Root Cause

The strongest current root cause is:

```text
action-visible PICF prefix nonstationarity caused by posterior file lifecycle
and recurrent latent scale drift.
```

Mathematically, the action loss is optimized through a semantic/action stack
that receives an extra PICF prefix:

```math
L_a(\theta; \phi_t)
=
\mathbb{E}
\left[
  \ell\left(
    f_\theta(x_t, c_{\phi_t}(z_{\le t}, h_t)),
    a_t
  \right)
\right]
```

where:

```text
theta: PaliGemma / PI0.5 action-side trainable parameters
phi_t: PICF posterior/router/prefix parameters and recurrent state statistics
c_phi: action-visible PICF control prefix
```

Even when `picf_action_prefix_stopgrad` prevents action loss from directly
updating PICF through the prefix, the action model still trains against a
moving conditioning distribution `c_phi`.  If posterior file lifecycle,
recycle gates, address updates, or prefix norms drift after the action head has
adapted, the teacher-forced action loss can rise quickly.

Observed evidence:

```text
qgfloor step 1700:
  action_default_equiv ~= 0.02685
  posterior_recycle_rate ~= 0.063
  raw slot_jepa ~= 59.6

qgfloor step 2250:
  action_default_equiv ~= 0.04975
  posterior_recycle_rate ~= 0.128
  raw slot_jepa ~= 141.5

qgfloor step 3150:
  action_default_equiv ~= 0.05366
  posterior_recycle_rate ~= 0.119
  raw slot_jepa ~= 2.87e6
```

The disabled raw `slot_jepa` telemetry is not part of the objective because
`lambda_slot_jepa=0`.  Its explosion is still important: it indicates the
future-prediction slot state has a large norm/scale mismatch, which is
consistent with recurrent/prefix drift.

## 5. CALVIN Evidence

The earlier CALVIN action-video gate rejected both tested checkpoints:

```text
ckpt2000:
  open_drawer 0/1
  nonfinite_actions = 146 / 360
  first nonfinite step = 214
  infer_ms_p50 ~= 1590

ckpt3000:
  open_drawer 0/1
  nonfinite_actions = 293 / 360
  first nonfinite step = 67
  infer_ms_p50 ~= 1402
```

This matches the training-side diagnosis: later checkpoints can look usable
from some loss terms but fail closed-loop recurrent deployment.  The next check
is the NaN-guarded rerun:

```text
/mnt/checkpoints/picf_core/eval/
  picf_a7_actionaware_ckpt2000_3000_nanfix_observe1_20260523
```

Partial result from the guarded rerun:

```text
ckpt2000_observe1:
  records=360
  nonfinite_actions=0
  clip_changed=1
  infer_ms_mean ~= 1694.66
  infer_ms_p50  ~= 1647.80
  open_drawer: 0 / 1
```

This proves the inference recurrent safety guard removes the old irreversible
NaN-action failure for ckpt2000.  It does not make the checkpoint behavior
successful: the first CALVIN `open_drawer` subtask still fails.

The ckpt3000 guarded rerun is still in progress at the time of this note.

Acceptance rule for this rerun:

```text
1. It may still fail behavior.
2. It should not enter irreversible nonfinite action state.
3. If fallback counters fire often, that is model-quality evidence, not a pass.
```

## 6. Required Next Diagnostics

Before another 30K run is interpreted as architecture quality, log the
following action-prefix stability fields during training:

```text
pi_prefix_norm_mean
pi_prefix_norm_max
pi_prefix_rms_mean
pi_prefix_rms_max
pi_prefix_delta_l2_to_previous
pi_prefix_cosine_to_previous
pi_prefix_nonfinite_count
posterior_recycle_rate
posterior_address_update_rate_mean
loss_slot_jepa_direction
loss_slot_jepa_log_norm
loss_slot_jepa_pred_norm
loss_slot_jepa_target_norm
```

The critical correlation test is:

```text
Does action loss rebound follow prefix drift / recycle drift?
```

If yes, the correct repair is prefix/recurrent-state stabilization.  If no, the
next suspect is action-head capacity or data distribution.

## 7. Repair Direction

Preferred repair order:

```text
1. Add training-time telemetry for action-visible PICF prefix statistics.
2. Add bounded normalization at the action-prefix interface, preferably RMSNorm
   or LayerNorm on PICF prefix tokens before PaliGemma consumes them.
3. Add recycle/lifecycle damping for action-visible posterior files so action
   conditioning does not flip faster than the action head can adapt.
4. Keep lambda_slot_jepa=0 until normalized direction/log-norm diagnostics are
   stable.
5. Re-run 300-step frozen-policy and 300-step action-aware smoke before another
   30K run.
6. Only after that, run CALVIN/video gates.
```

Rejected as root fixes:

```text
raising action weight alone:
  can force short-run action loss down but does not stabilize the moving prefix

optimizing raw same-role overlap:
  targets reserve/context telemetry more than the action-visible active path

enabling slot_jepa:
  unsafe while raw norm diagnostics explode

blind SAM/proposal revival:
  previously failed sidecar quality gates and does not address recurrent prefix
  nonstationarity
```

## 8. Current Conclusion

The loss rebound is not a simple "too many losses" problem.  The current
evidence says:

```text
early owner/scaffold problems were mostly mitigated;
raw same-role overlap is not the main causal metric;
late action rebound is most consistent with recurrent PICF prefix drift and
posterior lifecycle nonstationarity;
CALVIN nonfinite-action failures are behavior-level evidence of the same class
of instability.
```

Therefore the next robust fix should be an action-prefix/recurrent-stability
gate, not another object-scaffold tweak.

## 9. 2026-05-24 Action-Prefix Stability Gate

Implementation added:

```text
src/openpi/picf/core/config.py
  action_prefix_norm_mode
  action_prefix_rms_target
  action_prefix_norm_eps
  action_prefix_value_clip

src/openpi/picf/core/pipeline.py
  _stabilize_action_prefix_tokens(...)
  pi_prefix_* debug metrics in finalize_with_action(...)

scripts/picf_core_train.py
  CLI + metric logging for pi_prefix pre/post RMS, scale, max-abs,
  and nonfinite counts
```

Mathematical contract:

```math
c'_i =
\frac{\tau c_i}{\sqrt{\frac{1}{d}\sum_k c_{ik}^2 + \epsilon}}
```

for `action_prefix_norm_mode=rmsnorm`, where `c_i` is one PICF prefix token and
`\tau=action_prefix_rms_target`.  This is an interface normalization only:

```text
It does not change the internal PICF posterior, binding, owner transport,
sidecar scaffold, or reserve/context graph.
It only bounds the conditioning distribution seen by the PI0.5 action stack.
```

Why this is not a patchwork object-loss tweak:

```text
The failure signal was action-visible recurrent prefix nonstationarity.
The repair normalizes exactly that interface.
It leaves raw reserve/context overlap alone because the correlation audit did
not identify raw overlap as the causal driver.
```

Gate launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_actionprefix_rmsnorm_gate300_20260524.sh
```

Runtime note:

```text
PaliGemma/action-aware training must use fsdp_full_shard on the A7 2xA100-40GB
node.  DDP replicated the trainable semantic/action stack and OOMed before
step 1.  This is the same memory regime that motivated the maintained
action-aware wrappers; the prefix normalization itself is not the memory driver.
```

Long-run launcher after gate:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_actionprefix_rmsnorm_long30k_20260524.sh
```

Acceptance:

```text
pi_prefix_nonfinite_count == 0
pi_prefix_post_rms_mean ~= action_prefix_rms_target
loss_action_default_equiv does not reproduce the late rebound pattern
posterior_recycle_rate does not regain the high positive action-loss coupling
CALVIN inference no longer enters irreversible nonfinite action state
```
