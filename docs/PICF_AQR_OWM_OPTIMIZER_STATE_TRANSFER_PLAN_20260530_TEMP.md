# PICF-AQR-OWM Optimizer-State / Action-Transfer Plan - 2026-05-30

Canonical context:

```text
docs/picf_aqr_owm_202605/README.md
docs/PICF_AQR_OWM_MULTIMODAL_NOISE_ABLATION_PLAN_20260530_TEMP.md
docs/PICF_AQR_OWM_ACTION_PLATFORM_ROOT_CAUSE_MAP_20260530_TEMP.md
```

## Current Attribution

The 2026-05-30 sidecar/no-sidecar/reset triangle fixed one ambiguity:

```text
full-sidecar reset step8050 action = 0.0433176
no-sidecar reset step8050 action = 0.0433337
high-PICF-LR reset step8050 action = 0.0433395
```

Therefore:

```text
sidecar deletion is not the root cause;
whole-PICF LR increase is not the root cause;
the short-run improvement belongs to checkpoint restart / optimizer-state /
resumed-dynamics effects.
```

The current root-cause class is:

```math
\text{action-side optimization/transfer bottleneck under nonstationary PICF prefix}.
```

In words, the action path consumes:

```math
u_t = \mathrm{PI0.5Prefix}(x_t, B_\phi(o_{\le t}))
```

and minimizes:

```math
L_a(\theta, \phi) =
\ell(\pi_\theta(u_t), a_t).
```

If `B_phi` moves while the action optimizer has accumulated stale moments for
the previous prefix distribution, the action head can plateau or rebound even
when structural metrics improve.

## What Is Still Unresolved

We still need to separate:

```text
H1. optimizer-state issue:
  Adam moments / scheduler state are stale for the current prefix distribution.

H2. moving-prefix issue:
  PICF changes B_phi too fast for action/semantic to track.

H3. action/semantic capacity issue:
  even with frozen B_phi, action/semantic cannot exploit the current prefix.

H4. structural evidence issue:
  action needs continued PICF co-adaptation; freezing PICF hurts.
```

## Two-Hour Experiment

### E2. Frozen-PICF Action/Semantic Transfer

Run:

```text
picf_a7_stepindexed_from8000_policyonly_actionsemantic_h30k_20260530
```

Contract:

```text
resume = same step8000 corrected checkpoint
NUM_TRAIN_STEPS = 30000
step_indexed_window_rng = true
full sidecar = enabled
PICF_TRAINABLE_SCOPE = policy_only
semantic = paligemma(trainable=True scope=backbone_only)
action/policy head = trainable
Sonata / V-JEPA / AnyTouch = frozen
structural/object losses = 0
ACTION_LOSS_WEIGHT = 2.0
```

This keeps PICF forward evidence active, but freezes the complete `core.*`
belief router.  The only trainable path is the non-core action/semantic path.

Expected observations:

```text
If action drops below the full-sidecar reset row quickly:
  moving PICF prefix is a real co-training conflict.  The next production
  schedule should use alternating phases or a slow target-prefix teacher:
  structural/PICF update, then action/semantic polish against a frozen prefix.

If action matches full-sidecar reset:
  restart/fresh optimizer state explains the short-run improvement, but freezing
  PICF is not enough.  Next test should isolate optimizer groups:
  reset action/semantic optimizer only vs reset all optimizer.

If action is worse:
  action needs active PICF co-adaptation; freezing PICF is too restrictive.
  The next fix should be two-timescale co-training rather than policy-only
  polishing.
```

Decision rows:

```text
step8050:
  minimal validity row.

step8100:
  enough to decide whether E2 is promising.

step8150:
  only needed if first two rows are mixed.
```

Stop rule:

```text
Stop immediately after step8100 if E2 clearly matches or loses to full-sidecar
reset.  Continue to 8150 only if the first two rows are ambiguous.
```

### E2a. Frozen-PICF Frozen-Semantic Action-Head Transfer

Prepared fallback:

```text
picf_a7_stepindexed_from8000_policyonly_actionhead_h30k_20260530
```

Contract:

```text
resume = same step8000 corrected checkpoint
NUM_TRAIN_STEPS = 30000
step_indexed_window_rng = true
full sidecar = enabled
PICF_TRAINABLE_SCOPE = policy_only
semantic trainable = false
action/policy head = trainable
all structural/object losses = 0
ACTION_LOSS_WEIGHT = 2.0
```

Purpose:

```text
Run only if E2 does not beat the reset baseline.  E2a removes the trainable
PaliGemma branch from the causal split, so it answers whether the action/policy
head alone can adapt to the current PICF prefix distribution.
```

Decision:

```text
If E2a improves while E2 does not:
  PaliGemma/semantic updates are slowing or interfering with short-run action
  transfer.  Use action-head polishing before semantic co-train.

If E2a also matches the reset baseline:
  action-head-only is not sufficient; the missing factor is optimizer-group
  state or a required PICF/action co-adaptation.

If E2a is worse:
  semantic co-training is needed even for short-run transfer.
```

Live deployment:

```text
host = A7 / ssh -p 28060 root@36.139.225.68
tmux = picf_a7_from8000_policyonly_actionsemantic_h30k_20260530
log = /mnt/picf_run_logs/picf_a7_stepindexed_from8000_policyonly_actionsemantic_h30k_20260530.log
metrics = /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from8000_policyonly_actionsemantic_h30k_20260530/metrics.jsonl
remote repo = /root/openpi_probe_current_20260529
```

Configuration audit:

```text
PICF core:
  frozen by --picf-trainable-scope policy_only
  confirmed log: frozen_prefix=core.

Trainable path:
  PaliGemma semantic backbone trainable
  policy/action path trainable

Structural losses:
  all zeroed for this causal isolation.

Sidecar:
  full clean sidecar enabled.

Runtime:
  entered progress at step8001;
  observed speed around 22.7 sec/step.
```

Tail commands:

```bash
ssh -p 28060 root@36.139.225.68
tmux attach -t picf_a7_from8000_policyonly_actionsemantic_h30k_20260530
tail -f /mnt/picf_run_logs/picf_a7_stepindexed_from8000_policyonly_actionsemantic_h30k_20260530.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from8000_policyonly_actionsemantic_h30k_20260530/metrics.jsonl
```

Execution note:

```text
The first several minutes had no metrics because the run was in first forward
through PaliGemma/FSDP.  SIGUSR1 stackdump confirmed it was not a deadlock:
both ranks were inside PaliGemma action-flow forward under FSDP unshard.
This is acceptable for E2, but if a faster answer is needed later, the proper
follow-up is E2a action-head-only, not a change to the structural model.
```

## Live Result Rows

```text
step8050 E2 policy-only/action-semantic:
  loss_action_default_equiv = 0.0433483
  loss_total = 0.0433483
  loss_total_minus_action = 0.0
  loss_anchor_pv = 0.503478  # diagnostic only; structural loss weight is 0
  active support overlap max = 0.115
  downstream support overlap max = 0.119
  raw reserve/all overlap remains high, but inactive/reserve rows are not the
  optimized action path in this E2 causal split.

Comparison at step8050:
  full-sidecar reset = 0.0433176
  no-sidecar reset = 0.0433337
  high-PICF-LR reset = 0.0433395
  E2 frozen-PICF action/semantic = 0.0433483

Interim read:
  E2 does not beat the full-sidecar reset row.  Freezing PICF core is not, by
  itself, the missing ingredient.  Continue to step8100 for confirmation; if it
  remains matched, stop E2 and move to optimizer-group / action-only transfer.
```

Step8100 confirmation:

```text
loss_action_default_equiv = 0.0440691
loss_total = 0.0440691
loss_total_minus_action = 0.0
loss_anchor_pv = 0.514661  # diagnostic only; structural weight is 0
active support overlap max = 0.165
downstream support overlap max = 0.154

Conclusion:
  E2 should not be continued.  It matches/slightly worsens against the reset
  baseline and therefore rules out "moving PICF core alone" as the primary
  cause of the action plateau.
```

## E2a Startup Bug And Corrected Probe

Initial E2a failed before training:

```text
RuntimeError:
  picf_trainable_scope=policy_only matched no trainable non-core parameters.
```

Root cause:

```text
The first E2a script set:
  PICF_TRAINABLE_SCOPE=policy_only
  SEMANTIC_TRAINABLE=0

policy_only freezes core.*.  With semantic_trainable=false, the semantic
wrapper also freezes all non-core parameters.  Therefore no trainable parameter
exists.  This is a script/experiment-boundary bug, not evidence about the loss.
```

Correct E2a:

```text
PICF_TRAINABLE_SCOPE=policy_only
SEMANTIC_TRAINABLE=1
SEMANTIC_TRAINABLE_SCOPE=action_head_only
TRAINING_STRATEGY=ddp

This freezes:
  PICF core
  PaliGemma/Gemma backbone/expert

It trains only:
  semantic action_in_proj
  semantic action_out_proj
  semantic time_mlp_in
  semantic time_mlp_out
```

FSDP note:

```text
FSDP full-shard is not valid for this exact split because the trainable
wrapper-local action/time heads are float32 while the frozen semantic stack is
bf16, and root flattening rejects mixed dtype tensors.  DDP is the correct
causal probe wrapper here: it preserves the exact trainable set and avoids
introducing a dtype-cast workaround that would change the action-head test.
```

Decision rule:

```text
If corrected E2a improves:
  wrapper-local PI0 flow/time heads can adapt quickly to the frozen PICF prefix;
  use it as a short action-polish phase before full cotrain.

If corrected E2a matches/worsens:
  the plateau is not solved by local action/time heads.  The next real split is
  optimizer-state transfer or full semantic-backbone co-adaptation, not another
  structural-loss tweak.
```

Live corrected E2a deployment:

```text
host = A7 / ssh -p 28060 root@36.139.225.68
tmux = picf_a7_from8000_policyonly_actionhead_h30k_20260530
log = /mnt/picf_run_logs/picf_a7_stepindexed_from8000_policyonly_actionhead_h30k_20260530.log
metrics = /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from8000_policyonly_actionhead_h30k_20260530/metrics.jsonl

Confirmed startup:
  training_strategy = ddp
  picf_trainable_scope = policy_only
  semantic_trainable_scope = action_head_only
  trainable tensors = 8
  trainable numel = 2,165,792
  trainable names =
    semantic_encoder.encoder.action_in_proj.*
    semantic_encoder.encoder.action_out_proj.*
    semantic_encoder.encoder.time_mlp_in.*
    semantic_encoder.encoder.time_mlp_out.*

Observed progress:
  entered step8001;
  early runtime around 10-11 sec/step;
  metrics expected at step8050 because log_interval=50.
```

Final E2a result:

```text
Stopped at approximately step8535 after the step8500 metric was written.

Action-default-equivalent metric rows:
  step8050 = 0.0431069
  step8100 = 0.0438258
  step8150 = 0.0457705
  step8200 = 0.0448032
  step8250 = 0.0431541
  step8300 = 0.0469103
  step8350 = 0.0414932
  step8400 = 0.0398339
  step8450 = 0.0447620
  step8500 = 0.0475998

Summary:
  min  = 0.0398339
  mean = 0.0441260
  last = 0.0475998

The run is valid as a causal probe:
  trainable tensors = 8
  trainable numel = 2,165,792
  trainable names are exactly action_in/out_proj and time_mlp_in/out.
  PICF core and PaliGemma/Gemma backbone/expert are frozen.

Decision:
  Reject action-head-only polishing as the next production stage.  It is fast
  and clean, but it does not solve the plateau; by step8500 it is worse than
  E2 and worse than the reset baselines at the same comparison point.
```

Interpretation:

```text
E2 rejected "moving PICF core alone is the cause".
E2a rejected "small wrapper-local PI0 action/time heads alone can adapt out of
the plateau".

The remaining credible causes are:
  1. optimizer-state / schedule transfer across the full semantic-backbone
     action path;
  2. full semantic-backbone co-adaptation, not just action/time heads;
  3. data/window difficulty and multimodal prefix distribution mismatch.

The next 2-hour experiment should therefore not add structural losses.  It
should compare optimizer-state/schedule transfer under the production
semantic_trainable_scope=backbone_only boundary, because that is the smallest
remaining causal split that still trains the real PI0.5 action generator.
```

## Why This Is Not Another Ad-Hoc Patch

This is not a new module and not a hidden architecture change.  It is a causal
split of the existing training dynamics:

```math
\nabla_\theta L_a(\theta, \phi)\quad \text{with fixed }\phi
```

versus:

```math
(\nabla_\theta L_a,\nabla_\phi L_a+\nabla_\phi L_{struct})
\quad \text{with moving }\phi.
```

It tests whether the action platform is caused by the moving belief router or
by the action/semantic optimizer itself.

## E3. Stratified Same-Window Decision Probe - 2026-05-30

Purpose:

```text
Produce a 1-2 hour decision on whether the current issue should be treated as
an architecture problem or as a training/evaluation-schedule problem.
```

Reason:

```text
Raw online action rows mix two terms:
  sampled-window difficulty
  actual checkpoint quality

The old 4-22 run proves low online rows can be real, but it did not save
window traces.  The current fixed64 set is explicitly quarantined as a hard
diagnostic set, not a global metric.  Therefore the next valid gate is a
balanced exact-window probe on the same windows before and after a real
production-path continuation.
```

Mathematical object:

```math
J_S(\theta) = |S|^{-1}\sum_{w\in S} L_a(\theta; w)
```

where `S` is a fixed stratified set of CALVIN `(segment,start_step)` windows.
The decision uses:

```math
\Delta J_S = J_S(\theta_{9000}) - J_S(\theta_{8000})
```

not:

```math
\hat L_{k+1}(\theta_{k+1}) - \hat L_k(\theta_k)
```

because the latter includes sampled-window estimator noise.

Remote deployment:

```text
host:
  A7 / ssh -p 28060 root@36.139.225.68

repo:
  /root/openpi_probe_current_20260529

window set:
  /mnt/picf_fixed_window_probes/stratified_96_step8000_gate9000_20260530/
  stratified_96_windows.jsonl

window contract:
  mode = stratified
  buckets = block+grasp, block+push, block+slider, drawer, slider, manipulator
  per_bucket = 16
  total windows = 96
  seed = 20260530

before checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000

after checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_from8000_gate9000_20260530/9000

tmux:
  picf_stratified96_probe_8000_vs_9000

logs:
  /mnt/picf_run_logs/step8000_stratified96_probe.log
  /mnt/picf_run_logs/gate9000_step9000_stratified96_probe.log

summaries:
  /mnt/picf_fixed_window_probes/stratified_96_step8000_gate9000_20260530/
  step8000_summary.json

  /mnt/picf_fixed_window_probes/stratified_96_step8000_gate9000_20260530/
  gate9000_step9000_summary.json
```

Decision rule:

```text
If stratified fixed-window action improves materially from step8000 to step9000:
  current production recipe is still learning; do not modify architecture yet.
  Continue with optimizer-state/schedule hygiene and CALVIN/video validation.

If fixed-window action is flat or worse while active/downstream overlap remains
healthy:
  the bottleneck is not raw overlap or immediate slot collapse.
  The next fix should target the action-visible interface / optimizer transfer:
  full optimizer-state checkpoints, production semantic backbone cotrain, and
  possibly a gated dense-context path if fixed-window and CALVIN both reject.

If fixed-window action worsens and active/downstream overlap also fails:
  return to binding/router structure.  That would be the first strong evidence
  that a structural slot fix is required.
```

This is intentionally not another new loss and not a structural patch.  It is
the shortest strict experiment that can decide whether a structural patch is
justified.

Result:

```text
Completed on A7 with 96 accepted stratified windows.

Before:
  checkpoint = step8000
  summary =
    /mnt/picf_fixed_window_probes/stratified_96_step8000_gate9000_20260530/
    step8000_summary.json

After:
  checkpoint = gate9000 step9000
  summary =
    /mnt/picf_fixed_window_probes/stratified_96_step8000_gate9000_20260530/
    gate9000_step9000_summary.json
```

Aggregate fixed-window metrics:

```text
metric                                  step8000      step9000      delta
loss_action_default_equiv               0.040120      0.040363     +0.000243
loss_total_minus_action                 0.232287      0.221902     -0.010385
loss_anchor_pv                          0.517544      0.512588     -0.004956
loss_anchor_object_pull                 0.200409      0.160392     -0.040018
loss_slot_jepa                          1.109532      0.979869     -0.129664
posterior_recycle_rate                  0.122681      0.096906     -0.025775
active_same_role_support_overlap        0.130174      0.140625     +0.010451
downstream_same_role_support_overlap    0.114342      0.127711     +0.013369
```

Paired action deltas:

```text
windows = 96
mean_delta = +0.0002427
median_delta = -0.0000236
improved = 49
worsened = 47
```

Per-bucket action deltas:

```text
block+grasp                 n=16  mean_delta=+0.002550
block+push                  n=16  mean_delta=-0.000022
block+slider                n=16  mean_delta=-0.001363
drawer                      n=16  mean_delta=-0.000972
slider                      n=16  mean_delta=+0.001479
manipulator/button/light    mixed small-n, no collapse signal
```

Decision:

```text
The step8000 -> step9000 production continuation improves structural/belief
metrics but does not materially improve same-window action.

This is not evidence for another raw-overlap/anchor-structure patch:
  anchor_pv improves;
  object_pull improves;
  slot_jepa improves;
  recycle improves;
  active/downstream overlap remains in the known low band.

The remaining bottleneck is action transfer:
  the action-visible PI0.5 path is not converting better belief/router state
  into lower action loss on the same windows.

Next fix target:
  action-visible interface and optimizer/schedule transfer, not additional
  slot/raw-overlap losses.
```

Follow-up:

```text
Do not launch another slot-structure repair based on this result.
The next 1-2 hour experiment should test an action-interface repair or optimizer
state repair against the same stratified window set.
```

## E4. Live-Prefix Action-Interface Test - 2026-05-30

Reason:

```text
E3 gives a precise failure signature:
  PICF/router structure improves;
  fixed-window action does not.

The narrowest action-interface hypothesis is that the action-visible prefix is
too isolated from action gradients.  The maintained production recipe uses
picf_action_prefix_stopgrad to protect PICF from action pressure while the
prefix is still stabilizing.  At step9000, structure is already stable enough
to test whether this guard is now preventing transfer.
```

Mathematical test:

```math
z_\phi(o) = \text{PICF action-visible prefix}
```

Production stopgrad optimizes:

```math
\nabla_\theta L_a(\pi_\theta(x, \operatorname{sg}(z_\phi(o))))
```

E4 optimizes:

```math
\nabla_{\theta,\phi} L_a(\pi_\theta(x, z_\phi(o)))
```

while keeping the EMA teacher, RMS norm, output gate, and trust loss unchanged.
This is not a new loss and not a slot repair.  It is a direct test of whether
the action path can shape the interface it consumes.

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from9000_liveprefix_300_20260530.sh
```

Single intended change:

```text
ACTION_PREFIX_STOPGRAD = 0
```

Unchanged:

```text
resume = gate9000 step9000
NUM_TRAIN_STEPS = 9300
ACTION_LOSS_WEIGHT = 2.0
semantic_trainable_scope = backbone_only
PICF_CORE_LR_SCALE = 0.005
sidecar = contact_motion_full_tracklets_clean_20260520
unroll = 2
burnin = 1
action prefix RMS/EMA/gate = unchanged
slot/router/object losses = unchanged
```

Pass/fail by step9050/9100/9200/9300:

```text
PASS if action live rows improve without active/downstream overlap or prefix
trust metrics becoming unstable.

FAIL if action remains flat or worsens while structure is still healthy.
That would reject stopgrad isolation as the immediate bottleneck and move the
next repair to a richer action-visible interface, e.g. gated dense context.

HARD FAIL if active/downstream overlap, prefix norms, NaN, or object-pull
explode.  Then live-prefix action gradients are too destructive and should not
be used without a staged schedule.
```

First live checkpoint:

```text
Remote run:
  picf_a7_stepindexed_from9000_liveprefix_9300_20260530

Step 9050:
  loss_action_default_equiv        = 0.043630
  loss_total_minus_action          = 0.009840
  loss_anchor_pv                   = 0.503114
  loss_anchor_object_pull          = 0.115485
  loss_slot_jepa                   = 0.994517
  active same-role overlap max     = 0.170000
  downstream same-role overlap max = 0.174374
  posterior_recycle_rate           = 0.093941
  pi_prefix_post_rms_mean          = 0.700001
  pi_prefix_teacher_delta_rms      = 0.001483
  loss_action_prefix_trust         = 1.28e-7
  grad_norm                        = 0.984803

Interpretation:
  Runtime is healthy: no NaN, prefix RMS/gate/teacher trust are stable, and
  active/downstream overlap is not collapsing.  However action is not an
  immediate win versus the gate run.  This keeps E4 alive as a diagnostic but
  does not yet support adopting no-stopgrad as the final recipe.

Decision rule:
  Continue to 9100/9200 unless a hard failure appears.  If action remains flat
  while structure stays healthy, reject stopgrad isolation and move the next
  repair to the action-visible interface itself rather than another slot loss.
```

Step9100 update:

```text
Step 9050 -> 9100:
  loss_action_default_equiv        0.043630 -> 0.044189
  loss_total_minus_action          0.009840 -> 0.009752
  loss_anchor_pv                   0.503114 -> 0.488012
  loss_anchor_object_pull          0.115485 -> 0.108648
  loss_slot_jepa                   0.994517 -> 0.941791
  active same-role overlap max     0.170000 -> 0.105000
  downstream same-role overlap max 0.174374 -> 0.102457
  posterior_recycle_rate           0.093941 -> 0.098593
  prefix post-RMS                  0.700001 -> 0.700001
  prefix trust loss                1.28e-7  -> 2.24e-8
  grad_norm                        0.984803 -> 0.606103

Read:
  This is not a hard failure.  Live-prefix gradients are numerically stable and
  structure improves.  It is also not an action fix: the action row worsens
  slightly while the structural rows improve.

Interim conclusion:
  E4 rejects the simple hypothesis that stopgrad isolation alone is the
  bottleneck.  The evidence now points to an action-visible interface mismatch:
  the belief state is becoming cleaner, but the PI0.5 action path is not using
  that cleaner state to reduce action loss.

Next if 9200 confirms:
  Stop treating raw overlap or slot pull as the main defect.  Test a richer
  action-visible interface that preserves dense context and exposes PICF belief
  through a gated adapter/cross-attention path, rather than only a compressed
  prefix.
```

## E5. Gated Action-Context Interface - Prepared 2026-05-30

Reason:

```text
E3 and E4 jointly isolate the current bottleneck:
  - slot/router/belief rows improve on same windows;
  - action rows do not improve;
  - releasing action-prefix stopgrad is numerically stable but still does not
    transfer structural improvement into action improvement.

This means the compressed PI prefix is likely too narrow or too transformed to
carry the improved belief state into PI0.5 action generation.
```

Mathematical contract:

```math
C_t = \text{conditioned-control token sequence}
P_t = \text{compressed PI prefix}
K_t = \operatorname{select}(C_t, k)
```

Production path:

```math
\pi(a_t | x_t, P_t)
```

E5 path:

```math
\pi(a_t | x_t, [P_t,\; g\,\operatorname{norm}(K_t)])
```

with default stop-gradient on `K_t`:

```math
\nabla_\phi L_a(g\,\operatorname{norm}(K_t)) = 0
```

unless explicitly disabled.  This keeps E5 as an action-interface test, not a
new posterior objective.  It is consistent with gated VLA conditioning and
Flamingo-style guarded context injection: the action model can attend to more
context, but the context scale and gradient path are bounded.

Implementation status:

```text
Default path remains unchanged:
  action_context_tokens = 0

New config/CLI:
  --action-context-tokens
  --picf-action-context-stopgrad / --no-picf-action-context-stopgrad
  --action-context-norm-mode
  --action-context-rms-target
  --action-context-output-gate
  --action-context-include-query-tokens

Local verification:
  py_compile:
    scripts/picf_core_train.py
    src/openpi/picf/policy.py
    src/openpi/picf/core/config.py
  pytest:
    policy pi-prefix unchanged by default
    gated action-context append path
    EMA action-prefix teacher
    action-prefix RMS/gate tests
  Result: 5 passed.

Metrics contract:
  pi_context_token_count
  pi_context_gate
  pi_context_post_rms_mean
  pi_action_condition_token_count

  These are now explicitly whitelisted in `OWM_DEBUG_METRIC_KEYS`.  The first
  E5 launch reached step9050 before this whitelist was added; that row had
  `pi_context_* = null`, so it is archived as a logging/dataflow-audit failure,
  not as valid evidence for or against the gated-context hypothesis.
```

First diagnostic if E4 is rejected:

```text
resume = gate9000 step9000 or latest healthy E4 checkpoint
steps  = 300
action_context_tokens = 24
action_context_stopgrad = true
action_context_norm_mode = rmsnorm
action_context_rms_target = 1.0
action_context_output_gate = 0.25
action_context_include_query_tokens = false

PASS:
  action improves relative to gate/E4 while structural metrics remain stable.

FAIL:
  action stays flat while structure remains healthy.  Then the bottleneck is
  likely not prefix compression alone; inspect PI0.5 action adaptation,
  scheduler/optimizer transfer, or CALVIN action target comparability.

HARD FAIL:
  action improves only by exploding prefix/context metrics or destabilizing
  active/downstream overlap.  Then context gate is too high or context tokens
  need a stronger adapter rather than direct prefix appending.
```

Effective E5 launch:

```text
Run:
  picf_a7_stepindexed_from9000_actioncontext_9300_20260530

Fix before relaunch:
  Add `pi_context_*` and `pi_action_condition_token_count` to
  OWM_DEBUG_METRIC_KEYS.

Startup audit:
  prefix_stopgrad=True
  prefix_gate=0.7
  prefix_teacher=ema
  prefix_trust=0.02
  context_tokens=24
  context_stopgrad=True
  context_norm=rmsnorm
  context_gate=0.25
  context_include_queries=False
```

Step9050:

```text
pi_context_token_count: 24
pi_context_gate: 0.25
pi_context_post_rms_mean: 1.0
pi_action_condition_token_count: 28

loss_action_default_equiv: 0.052719
loss_total_minus_action: 0.009794
loss_anchor_pv: 0.504292
loss_anchor_object_pull: 0.108055
loss_slot_jepa: 0.972492
active/downstream overlap: 0.150 / 0.137
posterior_recycle_rate: 0.0946
```

Interpretation:

```text
The action context is truly active and numerically bounded, but the first
valid row is worse than both E4 and the gate run.  Structure remains healthy.

This weakens the simple "compressed prefix lacks tokens" hypothesis.  The
remaining action-transfer bottleneck is more likely the form of the action
adapter/readout: direct prefix concatenation exposes more control tokens, but
does not teach PI0.5 how to use them.  Continue to step9100 only to rule out
single-row noise.
```

Step9100:

```text
pi_context_token_count: 24
pi_context_gate: 0.25
pi_context_post_rms_mean: 1.0
pi_action_condition_token_count: 28

loss_action_default_equiv: 0.047409
loss_total_minus_action: 0.010620
loss_anchor_pv: 0.487897
loss_anchor_object_pull: 0.189887
loss_slot_jepa: 0.967336
active/downstream overlap: 0.140 / 0.149
posterior_recycle_rate: 0.1010
```

E5 conclusion:

```text
Direct append is a negative/weak diagnostic.  It proves that action-visible
context is present, bounded, and structurally safe, but it does not recover the
action loss.  The likely failure mode is not "too few context tokens"; it is
that appending 24 extra prefix tokens changes the PI0.5 prefix length and shifts
the action suffix position layout.  This violates the pretrained action
interface even though the additional tokens are normalized.
```

E6 fixed-length fusion:

```text
Replace direct append:
  P_action = concat(P_picf, C_context)

with fixed-length residual fusion:
  A = softmax(rmsnorm(P_picf) rmsnorm(C_context)^T / sqrt(d))
  P_fused = rms_cap(P_picf + A C_context, target_rms = rms(P_picf))

Properties:
  action_condition_token_count remains equal to the PI prefix length;
  action suffix position ids are unchanged;
  dense context can rotate the prefix direction;
  upward prefix scale drift is capped;
  context remains stop-gradient and separately gated.

Test:
  `action_context_integration=prefix_fusion`
  `action_context_tokens=24`
  expect `pi_action_condition_token_count=4`, not 28.
```

E6 launch:

```text
Run:
  picf_a7_stepindexed_from9000_prefixfusion_9300_20260530

Launcher:
  scripts/experiments/picf_aqr_owm_202605_active/run_a7_stepindexed_from9000_prefixfusion_300_20260530.sh
```

E6 step9050:

```text
pi_context_token_count: 24
pi_context_gate: 0.25
pi_context_post_rms_mean: 1.0
pi_context_fused_prefix_token_count: 4
pi_context_fused_post_rms_mean: 0.699998
pi_action_condition_token_count: 4

loss_action_default_equiv: 0.043636
loss_total_minus_action: 0.009635
loss_anchor_pv: 0.504453
loss_anchor_object_pull: 0.092918
loss_slot_jepa: 0.994009
active/downstream overlap: 0.155 / 0.153
posterior_recycle_rate: 0.0944
```

Read:

```text
The fixed-length adapter repaired the E5 regression.  It exposes the same 24
context tokens while preserving the 4-token PI prefix length and the old suffix
position layout.  Step9050 is not yet a win over E4, but it is a strong
causal result: the E5 failure was caused by the append/layout form, not by
context content or structural slot instability.
```

E6 step9100:

```text
loss_action_default_equiv: 0.044162
loss_total_minus_action: 0.011239
loss_anchor_pv: 0.489809
loss_anchor_object_pull: 0.245762
loss_slot_jepa: 0.933321
active/downstream overlap: 0.135 / 0.143
pi_context_token_count: 24
pi_context_fused_prefix_token_count: 4
pi_action_condition_token_count: 4
```

Decision:

```text
Stop E6 after step9100.  It is a useful causal repair of E5, but it does not
solve the action plateau.  Continuing the 300-step training stream would mix
the adapter question with sampled-window changes.  The next diagnostic is a
same-checkpoint, same-window, no-update prefix ablation:

  gate0    : action_prefix_output_gate=0.0, no context
  gate07   : action_prefix_output_gate=0.7, no context
  fusion24 : gate=0.7, 24 context tokens, fixed-length fusion
  append24 : gate=0.7, 24 context tokens, direct append

This isolates whether PICF prefix information is helping action, ignored by
the action head, or actively harmful.
```

Probe tooling correction:

```text
`scripts/picf_fixed_window_action_probe.py` now prepends both repo root and
repo_root/src to sys.path.  Without repo_root/src, remote probes can import a
stale installed `openpi` package while using current `scripts/`, creating mixed
code/config measurements.  Any old fixed-window probe result must show the
active worktree was on PYTHONPATH before it can be treated as definitive.
```

Same-window prefix ablation result:

```text
Checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_from9000_prefixfusion_9300_20260530/9100

Windows:
  24 accepted fixed windows, identical flat indices across all rows.

Results:
  gate0     action=0.064012  total_minus_action=0.261600  anchor_pv=0.524782
  gate07    action=0.064106  total_minus_action=0.261386  anchor_pv=0.523882
  fusion24  action=0.064091  total_minus_action=0.241600  anchor_pv=0.524782
  append24  action=0.076344  total_minus_action=0.241600  anchor_pv=0.524782
```

Strict read:

```text
1. gate0 ~= gate07:
   The current checkpoint's action loss is effectively insensitive to the
   PICF action prefix gate on this fixed-window set.

2. fusion24 ~= gate0/gate07:
   Fixed-length dense context fusion avoids the append regression, but it does
   not make the current action head use PICF context.

3. append24 is worse:
   Directly appending 24 context tokens is harmful.  The likely mechanism is
   pretrained PI0.5 prefix/suffix layout shift, not context-token content.

4. Structure losses stay almost unchanged across gate0/fusion24/append24:
   The ablation isolates the action-visible interface.  It is not measuring a
   new raw-overlap or ownership collapse event.
```

Root-cause update:

```text
The remaining action plateau is not explained by raw same-role overlap alone.
The stronger causal finding is that useful PICF/OEML information is not
currently a high-leverage conditioning path for the action expert.  Adding more
slot/object losses can improve structural metrics, but it cannot solve action
plateau unless the PI action path is trained or architected to consume the
belief state.

Therefore the next fix must be action-interface level:
  preserve PI0.5 prefix length and suffix position ids;
  avoid direct append;
  either train a real action-side adapter or explicitly prove that existing
  extra-prefix tokens affect predicted actions.
```
