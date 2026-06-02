# PICF-AQR-OWM 2026-05 Experiment Archive

Date: 2026-05-23

This directory is the stable index for the May 2026 PICF-AQR-OWM experiment
chain.  Historical `*_TEMP.md` files remain in `docs/` as evidence ledgers,
but this file is the entry point for continuing the work without rereading the
entire chat history.

## Current Maintained State

Current live architecture entry:

```text
src/openpi/picf/README_v2.2.md
```

Current action-readout / balanced-gradient ledger:

```text
docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md
```

Current small-window logical-batch gate:

```text
docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md#48-e24-logical-batch-plan
```

2026-06-02 update:

```text
E21b2 showed that exact-window K=2 updates can improve deterministic
all-window action eval, but do not reproduce E21's fast K=12 descent.

The maintained interpretation is now:
  the action path has capacity;
  direct optimizer resets are not the root explanation;
  the missing production ingredient is task-balanced logical optimizer
  updates that approximate the intended multi-task gradient under small
  physical batches.

Next quick gates:
  E24a exact12 windows_per_step=4
  E24b exact12 windows_per_step=6

E24 result:
  K=4 exact-window eval improved 0.0370217793 -> 0.0276452027 by step90.
  K=6 exact-window eval improved 0.0370217793 -> 0.0307366948 by step60.

Interpretation:
  balanced logical updates are necessary and useful;
  K=4 is a better speed/update-count tradeoff than blindly increasing K;
  this is still slower than E21 K=12 step20, so production needs explicit
  task-balanced logical updates plus per-task/per-modality loss normalization.

Do not start another 30K random-small-batch run from this branch.  The next
maintained production change should implement K≈4 task-balanced logical
optimizer steps, not rely on naive random batches or optimizer resets.
```

2026-06-02 status:

```text
E21 showed that 12 exact windows per optimizer update can make action descend
quickly under balanced cross-task gradients.

E23 moved this idea into the production trainer with
--calvin-balanced-bucket-sampler, but 2xA100-40GB could only run accum_steps=1,
so the live run had 2 windows/update rather than E21's 12 windows/update.

The A7 E23 run was intentionally stopped at step11400 before cloud shutdown.
It did not crash.  It preserved healthy active/downstream overlap and avoided
the old structural-collapse failure, but it did not reproduce E21's rapid
action descent.  Treat it as a preserved production-feasible approximation,
not as final acceptance.

Remote dirty state was snapshotted locally at:
  temp/remote_a7_snapshot_20260602/a7_openpi_dirty_snapshot_20260602.tar.gz
```

Current compact state report:

```text
docs/PICF_AQR_OWM_CURRENT_STATE_20260518.md
```

Current issue tracker:

```text
docs/PICF_AQR_OWM_OPEN_ISSUE_TRACKER_20260517_TEMP.md
```

Current experiment ledger:

```text
docs/PICF_AQR_OWM_EXPERIMENT_NOTE_20260517_TASK_OWNER_SIDECAREDS_TEMP.md
```

Current slot-JEPA diagnostic TODO:

```text
docs/PICF_AQR_OWM_SLOT_JEPA_NORMALIZED_TODO_20260523.md
```

Current scalability / CALVIN repair handoff:

```text
docs/PICF_AQR_OWM_SCALABILITY_AND_CALVIN_REPAIR_PLAN_20260523.md
```

Current V-JEPA cache / 200-step action-weight gate:

```text
docs/PICF_AQR_OWM_VJEPA_CACHE_AND_ACTION200_REPORT_20260523.md
```

Note: the maintained cache is a recent temporal suffix cache for the frozen
V-JEPA maps actually consumed by PICF.  The full dense-volume cache variant was
rejected because it wrote about 216 MB per clip entry and made cold-cache
training I/O-bound.

Current CALVIN inference latency gate:

```text
docs/PICF_AQR_OWM_SCALABILITY_AND_CALVIN_REPAIR_PLAN_20260523.md#7-calvin-inference-latency-gate---2026-05-23
scripts/experiments/picf_aqr_owm_202605_active/run_a7_calvin_inference_latency_gate_20260523.sh
```

This gate compares PI0.5-only ablated serving against PICF-enabled serving on
the same A7 websocket/CALVIN path.  It is the maintained answer to the
deployment-speed question; do not infer online latency from training step time
or from debug-overlay CALVIN runs.

Current cotrain rebound / two-timescale plan:

```text
docs/PICF_AQR_OWM_COTRAIN_TWO_TIMESCALE_FINAL_20260526.md
scripts/experiments/picf_aqr_owm_202605_active/run_a7_twotimescale_cotrain_from_pc1_6000_30k_20260526.sh
```

This is the maintained answer to the May-24/25 action rebound.  `policy_only`
proved the moving-prefix failure mode but is not the production recipe.  The
production recipe remains cotrain: PaliGemma/action path is trainable at normal
speed, PICF `core.*` remains trainable at a lower LR scale, and object scaffold
losses remain weak/decayed.

Current phase-stabilized rebound repair:

```text
docs/PICF_AQR_OWM_ACTION_REBOUND_PHASE_STABILIZATION_20260526.md
docs/PICF_AQR_OWM_ACTION_INTERFACE_EMA_FINAL_20260527.md
scripts/experiments/picf_aqr_owm_202605_active/run_a7_phase_stabilized_from6500_30k_20260526.sh
scripts/experiments/picf_aqr_owm_202605_active/run_a7_basil_prefixgate_from6500_30k_20260527.sh
scripts/experiments/picf_aqr_owm_202605_active/run_a7_actionprefix_ema_from6800_30k_20260527.sh
```

This supersedes the simple core=0.02 continuation as the next live gate.  The
core=0.02 branch confirmed that lower PICF LR helps, but steps 7000/7050/7100
still rebounded.  The current repair first stationarizes the action-visible
PICF prefix with a 300-step `policy_only` phase from the clean 6500 checkpoint,
then resumes full cotrain with slow PICF core pressure.  The 2026-05-27
core=0.005 run also rebounded at 7350 while structure stayed healthy, so the
next maintained script adds gated extra-prefix conditioning and block-alternating
PICF core updates.
The BASL/block-core branch also reproduced the 7350 action rebound.  The
maintained 2026-05-27 repair is now the action-interface EMA teacher:
PI0.5 consumes a slow target prefix, PICF keeps training online, and the new
`loss_action_prefix_trust` term is logged as non-action alignment pressure so
historical `loss_action_default_equiv` comparisons remain valid.

Current speed regression audit:

```text
docs/PICF_AQR_OWM_SPEED_REGRESSION_AUDIT_20260528.md
```

The 2026-05-28 EMA7000 full-cotrain continuation regressed from the comparable
25-28 sec/step band to about 40-46 sec/step.  The audit rules out a simple
unfreeze mistake, sidecar-only explanation, optimizer-step overhead, and anchor
overlay overhead.  Profiling localizes the regression to model forward/backward
inside the full-trainable transition path.  Do not treat this slow branch as
the production 30K baseline until the short same-checkpoint ablations in that
audit are complete.

Current action-rebound root-cause split:

```text
docs/PICF_AQR_OWM_ACTION_REBOUND_CAUSAL_PLAN_20260528.md
docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md
docs/PICF_AQR_OWM_STEPINDEXED_RUN_LOCAL_AUDIT_20260529_TEMP.md
docs/PICF_AQR_OWM_DATA_WINDOW_DISTRIBUTION_AUDIT_20260529_TEMP.md
docs/PICF_AQR_OWM_ANCHOR_LR_SAMPLING_CAUSAL_PLAN_20260530_TEMP.md
docs/PICF_AQR_OWM_ACTION_PLATFORM_ROOT_CAUSE_MAP_20260530_TEMP.md
docs/PICF_AQR_OWM_MULTIMODAL_NOISE_ABLATION_PLAN_20260530_TEMP.md
docs/PICF_AQR_OWM_ACTION_LOSS_WINDOW_AUDIT_20260530.md
docs/PICF_AQR_OWM_ACTION_PLATFORM_ROOT_CAUSE_PROTOCOL_20260531.md
```

2026-05-31 maintained diagnostic rule:

```text
Do not launch another action-platform or slot-structure experiment from rolling
train loss alone.  The latest exact-window replay showed source /9100 and E13
/9200 differ by only +0.000225 action on the same 100 source windows, so the
latest apparent 0.046 rolling gap is mainly sampled-window non-stationarity,
not proof of model degradation.  New experiments must first state which
hypothesis they test, whether that hypothesis was already excluded in the May
notes, and whether the current code differs enough to justify retesting.

Use:
  docs/PICF_AQR_OWM_ACTION_PLATFORM_ROOT_CAUSE_PROTOCOL_20260531.md

Primary next gates:
  E14A future-window exact replay;
  E14B stratified held-window probe;
  E14C full-optimizer checkpoint smoke continuation.
```

Important 2026-05-30 correction:

```text
No-sidecar diagnostics must keep NUM_TRAIN_STEPS=30000 even when only running
100-300 live steps.  The first 8300-horizon no-sidecar launch changed the LR
schedule and is archived as invalid/non-comparable.
```

2026-05-30 fast exclusion map:

```text
Current phenomenon:
  continuous training can enter a 0.04-0.05 action plateau/rebound, while
  restarting from the same checkpoint can produce a short-run action drop.

Current root-cause class:
  action-side optimization / transfer bottleneck under a nonstationary PICF
  prefix, amplified by checkpoint/resume optimizer dynamics.

Experiment that fixed the attribution:
  E1 no-sidecar h30k and E1b full-sidecar-reset h30k, both from the same
  step8000 checkpoint with NUM_TRAIN_STEPS=30000.

Key row:
  full-sidecar reset step8050 action=0.0433176 anchor_pv=0.5041
  no-sidecar reset step8050 action=0.0433337 anchor_pv=0.5909
  high-PICF-LR reset step8050 action=0.0433395 anchor_pv=0.5081

Excluded as primary causes:
  sidecar/proposal/tracklet deletion: not primary; deleting worsens structure.
  whole-PICF LR too low: not primary; high-PICF-LR reset gives same action row.
  raw inactive overlap: not direct action-platform cause.
  SAM proposals: archived/rejected; do not revive as a default path.
  fixed64/fixed-window scalar probe: not representative for production judgment.
  8300-horizon no-sidecar launch: invalid because it changes LR schedule.

Still unresolved inside the root-cause class:
  whether the decisive state is Adam moments, action/semantic optimizer state,
  PICF prefix nonstationarity, action-weight timing, or a combination.

Next clean experiments:
  same ckpt with optimizer-state variants:
    keep optimizer;
    reset all optimizer;
    reset only action/semantic optimizer;
    freeze PICF and train action/semantic only.
```

Active next-stage 2h diagnostic:

```text
docs/PICF_AQR_OWM_OPTIMIZER_STATE_TRANSFER_PLAN_20260530_TEMP.md
```

The first runnable branch is `E2 frozen-PICF action/semantic transfer` from the
same step8000 checkpoint.  It keeps full sidecar and PICF forward evidence, but
freezes `core.*` and trains only the non-core action/semantic path.  This
directly tests whether the remaining platform is caused by moving PICF prefixes
or by action/semantic optimization itself.

Live A7 run:

```text
exp:
  picf_a7_stepindexed_from8000_policyonly_actionsemantic_h30k_20260530

tmux:
  picf_a7_from8000_policyonly_actionsemantic_h30k_20260530

tail:
  /mnt/picf_run_logs/picf_a7_stepindexed_from8000_policyonly_actionsemantic_h30k_20260530.log
  /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from8000_policyonly_actionsemantic_h30k_20260530/metrics.jsonl

initial audit:
  entered progress at step8001;
  observed speed around 22.7 sec/step;
  confirmed Trainable scope scope=policy_only frozen_prefix=core.
```

Historical 2026-05-29 branches showed a 7550/7600/7650 action-loss rebound even
when `loss_total_minus_action` stayed bounded.  Fixed-group, policy-semantic,
and action-head-only splits reproduced the same scalar pattern, so the failure
was not isolated to active PICF structure or to object auxiliary losses.

Near-zero-LR replay closed the root-cause diagnosis: `LR=1e-12` decayed to about
`1e-16..1e-20` and still reproduced `7550=0.03684`, `7600=0.03718`.  That means
the old resumed-run evidence was contaminated by the sampled-window stream
rather than by meaningful parameter updates.  The trainer now uses
step-indexed sampled-window RNG by default, so resumed runs no longer replay an
early-window stream.  Legacy behavior is available only via
`--no-step-indexed-window-rng` for historical reproduction.

2026-05-29 corrected relaunch:

```text
tmux:
  picf_a7_stepidx_ema7000_30k

monitor:
  picf_a7_stepidx_gate_monitor

exp:
  picf_a7_stepindexed_actionprefix_ema_from7000_30k_20260529

resume:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000

log:
  /mnt/picf_run_logs/picf_a7_stepindexed_actionprefix_ema_from7000_30k_20260529.log

metrics:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_stepindexed_actionprefix_ema_from7000_30k_20260529/metrics.jsonl
```

First corrected gates:

```text
7050 action_default=0.050193 active/downstream=0.044/0.057
7100 action_default=0.051401 active/downstream=0.070/0.078
7150 action_default=0.042441 active/downstream=0.110/0.110
7200 action_default=0.037938 active/downstream=0.080/0.078
7250 action_default=0.050864 active/downstream=0.085/0.093
7300 action_default=0.043123 active/downstream=0.080/0.084
7350 action_default=0.039770 active/downstream=0.085/0.084
7400 action_default=0.045796 active/downstream=0.065/0.068
7450 action_default=0.045965 active/downstream=0.135/0.128
7500 action_default=0.042314 active/downstream=0.165/0.175
7550 action_default=0.044523 active/downstream=0.085/0.090
```

Decision:

```text
7550 passed.  Continue to the corrected 7600/8000 gates.  Do not compare
this run's first corrected-window scalar directly to the old replayed-window
run; use the step-indexed stream and the structural gates documented in
docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md.
```

Local follow-through audit:

```text
docs/PICF_AQR_OWM_STEPINDEXED_RUN_LOCAL_AUDIT_20260529_TEMP.md
docs/PICF_AQR_OWM_DATA_WINDOW_DISTRIBUTION_AUDIT_20260529_TEMP.md
```

This file is the current checklist for the corrected run.  It records the
file-level audit, the exact mathematical stop gates, and the hypotheses that
should not be repeated unless the corrected 7550/7600 gates fail.

The data-window audit is the current answer to "what data did this run
actually read?"  It uses `window_trace_rank*.jsonl` from the cloud run to
separate sampled-window/task-mixture effects from real checkpoint degradation.
As of step7950, proposal/mask/tracklet coverage is stable and active/downstream
overlap is healthy, so the remaining action plateau cannot be blamed on missing
sidecar data or raw inactive overlap alone.

Current plateau root-cause plan:

```text
docs/PICF_AQR_OWM_CURRENT_PLATEAU_ROOT_CAUSE_PLAN_20260529_TEMP.md
scripts/picf_fixed_window_action_probe.py
```

Through step8250, the corrected step-indexed continuation stayed in the
`0.04..0.05` action band and the recent eight-row mean worsened
(`0.04445 -> 0.04826`) while active/downstream overlap remained healthy.  This
makes the live optimizer continuation low-value as a production candidate.  The
next maintained discriminator is a no-update fixed-window probe comparing the
preserved step7000 and current step8000 checkpoints on identical accepted
windows before starting another large architecture or loss rewrite.

2026-05-29 result: the fixed-window discriminator is complete.  On 64 identical
accepted windows, current step8000 improves action over current step7000
(`0.061891 -> 0.057919`), so the live-row rebound is not direct action-interface
destruction.  On the same windows, archived 4-22 PI0.5-only action means are:

```text
step7500  0.058717
step10000 0.053813
step20000 0.051038
```

Current step8000 is therefore roughly old step7500 fixed-window quality, not
old step20000 quality.  The archived train-log `0.02` scalar is not a
stationary validation target.  The next decision should be behavior/CALVIN
evaluation or a later fixed-window probe, not another raw-overlap patch.

Follow-up A7 continuation:

```text
run: picf_a7_fixedprobe_from8000_action2_30k_20260529
resume: /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000
log: /mnt/picf_run_logs/picf_a7_fixedprobe_from8000_action2_30k_20260529.log
metrics: /mnt/checkpoints/picf_core/picf_core/picf_a7_fixedprobe_from8000_action2_30k_20260529/metrics.jsonl
save interval: 1000
keep last checkpoints: 5
```

This run is allowed because the fixed-window probe showed step8000 is not a
damaged checkpoint.  The first metric row is step8050; later live-row regressions
must be checked by another fixed-window probe before changing architecture.

```text
/mnt/checkpoints/picf_core/picf_core/cleanup_manifest_20260529_action_rebound_dryrun.txt
/mnt/checkpoints/picf_core/picf_core/cleanup_deleted_20260529_action_rebound.txt
```

## Current Production-Length Run

The active A7 run as of 2026-05-23 is:

```text
tmux:
  picf_a7_qgfloor003_from1500_20260522

exp:
  picf_a7_actionaware_qgfloor003_from1500_long30k_20260522

log:
  /mnt/picf_run_logs/picf_a7_actionaware_qgfloor003_from1500_long30k_20260522.log

resume:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_actionaware_qgdecay_fast1300_from1000_long30k_20260522/1500
```

Launch contract:

```text
V-JEPA / Sonata / AnyTouch pretrained modules frozen
PaliGemma trainable
action head trainable
unroll_steps=2
burnin_steps=1
effective_global_batch=2
lambda_action_pos=2.0
lambda_action_rot=2.0
lambda_action_gripper=2.0
OBJECT_SCAFFOLD_DECAY_FLOOR=0.03
SAVE_INTERVAL=500
KEEP_LAST_CHECKPOINTS=3
LOG_INTERVAL=50
ANCHOR_OVERLAY_INTERVAL=100
```

Tail command:

```bash
ssh -p 28060 root@36.139.225.68
tail -f /mnt/picf_run_logs/picf_a7_actionaware_qgfloor003_from1500_long30k_20260522.log
```

## Step-1500 / 1700 Gate Summary

The floor-0.03 continuation was chosen after the step1500 gate because the
action lambdas were already at the traditional PICF/PI0 scale:

```text
lambda_action_pos     = 2.0
lambda_action_rot     = 2.0
lambda_action_gripper = 2.0
```

Therefore the correct action-dominant move was to reduce weak object-scaffold
pressure, not to increase action above the legacy/default action scale.

Observed floor-0.03 branch:

```text
step   total     action    non-action  action_frac  active_ov  down_ov  dup
1550   0.0462    0.0357    0.0105      77.3%        0.060      0.171    0
1600   0.0451    0.0353    0.0098      78.2%        0.025      0.106    0
1650   0.0396    0.0290    0.0107      73.1%        0.084      0.155    0
1700   0.0371    0.0269    0.0103      72.3%        0.075      0.109    0
```

Interpretation:

```text
The weight change is doing what it was designed to do:
  action is dominant;
  non-action scaffold remains bounded near 0.010;
  active/downstream overlap remains below failure bands;
  active duplicate overlap remains 0;
  action loss continues to improve.
```

Do not further increase action weight unless:

```text
action_default_equiv stalls for multiple 50-step gates,
active/downstream metrics remain healthy,
non-action remains too high,
and the floor-0.03 branch has already been observed past the next checkpoint.
```

## Paper / Theory Records

Primary paper matrix:

```text
docs/PICF_AQR_OWM_SLOT_VLA_PAPER_MATRIX_20260522.md
```

This reviews 38 papers/systems across:

```text
object binding / IsSameObject probes
MetaSlot / QASA / slot merging / temporal slot consistency
object-centric robotics and manipulation
JEPA / V-JEPA / predictive VLA
action-dominant VLA recipes
tactile / contact-rich VLA
```

Action-weight audit:

```text
docs/PICF_AQR_OWM_ACTION_DOMINANT_WEIGHT_AUDIT_20260522.md
```

This is the maintained deployment reference for action/scaffold weights.  It
contains the formal loss decomposition, the expected action-fraction ranges,
the observed `1550-1700` gate, and the rules for when to continue, pause, lower
scaffold, or run a new action-weight ablation.

Main theoretical conclusion:

```text
PICF should remain an action-dominant control belief router:
  active object files explain task/contact/motion evidence;
  dense background/context remains residual typed context;
  weak object scaffolds decay after active ownership is healthy;
  raw predictive losses remain guarded until normalized and matched.
```

## Accepted / Rejected Module Status

Accepted in current production recipe:

```text
active object owner files
context/reserve duplicate suppression
posterior file competition
owner measurement transport
sidecar contact/motion/tracklet typed evidence
object-only role layout
tactile-to-object-owner binding
PaliGemma trainable action co-training
weak object scaffold decay
```

Rejected for current production recipe:

```text
blind SAM proposals
hard sidecar mask truth
full RGB reconstruction decoder as production loss
raw slot-count penalty
raw unnormalized slot-JEPA/support prediction losses
legacy local refinement by default
```

Guarded / future:

```text
offline IsSameObject probe with sidecar/tracklet/posterior weak labels
gated dense-context cross-attention for residual V-JEPA/context tokens
normalized matched latent prediction with detached targets
deterministic frozen V-JEPA feature cache
```

Maintained token-budget decision:

```text
semantic_max_length=256 for production and large-data fine-tuning
semantic_max_length=200 only for strict PI0.5 parity / memory ablation
```

## Evidence Ledgers

Use these only when deeper historical context is needed:

```text
docs/PICF_AQR_OWM_30K_VALIDATION_AND_DENSE_CONTEXT_PLAN_20260520_TEMP.md
docs/PICF_AQR_OWM_ACTION_AWARE_SMOKE_20260520_TEMP.md
docs/PICF_AQR_OWM_CONTEXT_DEDUP_FOLLOWTHROUGH_20260520_TEMP.md
docs/PICF_AQR_OWM_OWNER_MEASUREMENT_TRANSPORT_20260520_TEMP.md
docs/PICF_AQR_OWM_ROBUST_OEML_CORE_FOLLOWTHROUGH_20260520_TEMP.md
docs/PICF_AQR_OWM_SLOT_PAPER_DATAFLOW_COMPARE_20260521_TEMP.md
docs/PICF_AQR_OWM_SLOT_PAPER_CODE_GAP_AUDIT_20260518_TEMP.md
docs/PICF_AQR_OWM_FROZEN_FEATURE_CACHE_TODO_20260522.md
docs/PICF_AQR_OWM_SPEED_REGRESSION_AUDIT_20260528.md
```

SAM archive:

```text
docs/archive/picf_aqr_owm_202605/sam_rejected_20260519/
```

## Next Gates

Short-term:

```text
Use semantic_trainable_scope=backbone_only for production PaliGemma cotrain.
This is still semantic-backbone cotrain; it only freezes wrapper-local restored
PI0 flow/time heads. semantic_trainable_scope=all is a slow diagnostic mode,
not the default 30K recipe.

Let the current floor-0.03 branch run at least to the next checkpoint.
Inspect gates at 1800 / 1900 / 2000 / 2500.
Do not modify weights again before the next checkpoint unless active/downstream
metrics fail.
Before the next CALVIN repair run, use
docs/PICF_AQR_OWM_SCALABILITY_AND_CALVIN_REPAIR_PLAN_20260523.md as the
handoff: first separate inference-speed failures from checkpoint/action-quality
failures, then decide whether V-JEPA feature cache or action/scaffold gating is
the correct intervention.
```

Current May-30 diagnostic gate:

```text
E2 frozen-PICF + train semantic/action did not improve action transfer:
  step8050 action_default_equiv = 0.0433483
  step8100 action_default_equiv = 0.0440691

Therefore do not repeat policy_only/action-semantic as a production fix.

Corrected E2a is the next 2-hour probe:
  PICF_TRAINABLE_SCOPE=policy_only
  SEMANTIC_TRAINABLE=1
  SEMANTIC_TRAINABLE_SCOPE=action_head_only
  TRAINING_STRATEGY=ddp

This is the only valid action-head-only split currently exposed by the code.
The earlier variant with SEMANTIC_TRAINABLE=0 is invalid because it has no
trainable non-core parameters.

Live handle:
  tmux = picf_a7_from8000_policyonly_actionhead_h30k_20260530
  log = /mnt/picf_run_logs/picf_a7_stepindexed_from8000_policyonly_actionhead_h30k_20260530.log
  metrics = /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from8000_policyonly_actionhead_h30k_20260530/metrics.jsonl

Result:
  stopped after step8500;
  action_default_equiv min/mean/last = 0.0398339 / 0.0441260 / 0.0475998.
  This does not beat the same-step reset baselines and should not be promoted.

Next:
  test optimizer-state / schedule transfer with the real production action path
  semantic_trainable_scope=backbone_only. Do not repeat action_head_only unless
  the question is specifically wrapper-local head capacity.
```

Current strict gate:

```text
Use the May-30 stratified same-window probe before making another architecture
change:

docs/PICF_AQR_OWM_OPTIMIZER_STATE_TRANSFER_PLAN_20260530_TEMP.md
  section: E3. Stratified Same-Window Decision Probe

Remote tmux:
  picf_stratified96_probe_8000_vs_9000

Question:
  Does the real production continuation from step8000 to step9000 improve the
  same balanced CALVIN windows?

Decision:
  If yes, continue training/schedule hygiene and CALVIN validation.
  If no, do not keep patching raw overlap; move to action-visible interface /
  optimizer-state transfer repair.
```

Outcome:

```text
Completed: stratified 96-window step8000 vs gate9000/9000 probe.

Result:
  action fixed-window mean: 0.040120 -> 0.040363
  paired action delta: +0.000243 mean, -0.000024 median
  action improved/worsened windows: 49 / 47

Structural metrics improved:
  loss_total_minus_action: 0.232287 -> 0.221902
  loss_anchor_pv:          0.517544 -> 0.512588
  loss_anchor_object_pull: 0.200409 -> 0.160392
  loss_slot_jepa:          1.109532 -> 0.979869
  posterior_recycle_rate:  0.122681 -> 0.096906

Conclusion:
  current slot/router/belief training is improving structure, but the PI0.5
  action-visible path is not converting that improvement into lower same-window
  action loss.  Do not spend the next iteration on raw-overlap or sidecar
  structure.  Next target is action-interface / optimizer-transfer repair.
```

Follow-up E4:

```text
Run:
  picf_a7_stepindexed_from9000_liveprefix_9300_20260530

Single intended change:
  ACTION_PREFIX_STOPGRAD=0

Step9050 early read:
  loss_action_default_equiv = 0.043630
  active/downstream overlap = 0.170 / 0.174
  anchor_object_pull        = 0.115485
  slot_jepa                 = 0.994517
  prefix post-RMS           = 0.700001
  trust loss                = 1.28e-7

Meaning:
  live-prefix gradients are not immediately destructive, but they have not yet
  shown action improvement.  Wait for 9100/9200/9300 before deciding whether
  stopgrad isolation was the bottleneck.
```

Step9100 read:

```text
9050 -> 9100:
  action_default_equiv: 0.043630 -> 0.044189
  active/downstream overlap: 0.170/0.174 -> 0.105/0.102
  anchor_pv: 0.503114 -> 0.488012
  anchor_object_pull: 0.115485 -> 0.108648
  slot_jepa: 0.994517 -> 0.941791

Conclusion:
  structure improves, action does not.  No-stopgrad is stable but not solving
  the action-transfer bottleneck by itself.  If 9200 confirms this, the next
  experiment should target the action-visible interface, not another slot
  overlap repair.
```

Prepared E5:

```text
Name:
  gated action-context interface

Default:
  off (`action_context_tokens=0`)

Purpose:
  test whether PI0.5 needs more of `conditioned_control.tokens` than the
  compressed `pi_prefix_tokens` can carry.

Mechanism:
  action_prefix = [pi_prefix_tokens,
                   action_context_output_gate *
                   norm(selected conditioned_control tokens)]

Safety:
  appended context is stop-gradient by default, independently RMS-normalized,
  and separately gated.  This preserves PICF belief training while exposing a
  richer action-visible interface.

Local checks:
  5 targeted tests passed and py_compile passed.

Instrumentation correction:
  The first E5 launch wrote a step9050 row without `pi_context_*` metrics
  because the action-context debug keys were not included in the trainer
  whitelist.  The run was stopped and relaunched after adding:
    pi_context_token_count
    pi_context_gate
    pi_context_post_rms_mean
    pi_action_condition_token_count

  Treat rows from the pre-correction E5 launch as invalid for action-context
  attribution.

Effective E5 step9050:
  pi_context_token_count=24
  pi_context_gate=0.25
  pi_context_post_rms_mean=1.0
  pi_action_condition_token_count=28
  action_default_equiv=0.052719
  total_minus_action=0.009794
  anchor_pv=0.504292
  anchor_object_pull=0.108055
  slot_jepa=0.972492
  active/downstream overlap=0.150/0.137

Read:
  the direct gated context is active and bounded, but does not improve action
  transfer at the first valid row.  This points away from "not enough control
  tokens in the prefix" and toward the action adapter/readout form.
```

Step9100 confirms the same direction:

```text
action_default_equiv=0.047409
total_minus_action=0.010620
anchor_pv=0.487897
anchor_object_pull=0.189887
slot_jepa=0.967336
active/downstream overlap=0.140/0.149
pi_action_condition_token_count=28
```

Conclusion:

```text
Do not continue the direct-append context route as a maintained design.  It is
bounded and structurally safe, but it changes PI0.5 prefix length and shifts
action suffix positions.  The next maintained diagnostic is fixed-length
context-to-prefix fusion: dense context is read into the existing PI prefix via
a gated/RMS-capped residual attention adapter, so PI0.5 still sees the same
number of prefix tokens.
```

E6 fixed-length fusion first read:

```text
Run: picf_a7_stepindexed_from9000_prefixfusion_9300_20260530
Step9050:
  action_default_equiv=0.043636
  total_minus_action=0.009635
  active/downstream overlap=0.155/0.153
  pi_context_token_count=24
  pi_context_fused_prefix_token_count=4
  pi_action_condition_token_count=4

Meaning:
  fixed-length fusion removes the E5 append/layout regression.  It is not yet
  proof of action improvement beyond E4, but it validates the causal diagnosis
  that direct prefix-length growth was harmful.
```

E6 step9100 and next diagnostic:

```text
Step9100:
  action_default_equiv=0.044162
  total_minus_action=0.011239
  active/downstream overlap=0.135/0.143
  pi_context_token_count=24
  pi_context_fused_prefix_token_count=4
  pi_action_condition_token_count=4

Decision:
  Stop E6 as a training-stream run.  It repaired the append regression but did
  not break the action plateau.  Run same-window no-update prefix ablation next:
  gate0 vs gate07 vs fusion24 vs append24 from the same checkpoint.
```

Fixed-window probe hygiene:

```text
`scripts/picf_fixed_window_action_probe.py` must import this repo's `src/`;
otherwise remote probes can accidentally mix current scripts with stale
installed `openpi` code.  This has been fixed.  Treat older fixed-window probe
artifacts as suspect unless their logs prove the active worktree was on
PYTHONPATH.
```

Production acceptance still requires:

```text
checkpoint health;
action_default_equiv trend versus 4-22 ablation;
anchor overlay/video review;
CALVIN evaluation;
identity switch / recycle stability;
support overlap active/downstream health.
```

## 2026-05-31 E7 Action-Side Adapter

Action-platform root-cause status:

```text
same-window gate0/gate07/fusion24:
  action loss nearly identical

same-window append24:
  action loss worse
```

Interpretation:

```text
PICF belief/context is structurally healthy enough to improve non-action
metrics, but the PI0.5 action expert is not using passive PICF prefix evidence.
Direct context append is rejected because it shifts the pretrained suffix
layout.
```

Maintained fix:

```text
action_context_integration=suffix_cross_attention

PI prefix:
  unchanged

PICF context:
  bounded side context, not appended

action suffix:
  trainable gated cross-attention reads PICF context before the PI action
  expert forward

first diagnostic launcher:
  scripts/experiments/picf_aqr_owm_202605_active/run_a7_stepindexed_from9100_suffixadapter_300_20260531.sh
```

This is the next required gate before another 30K claim.  The gate must show
adapter metrics in JSONL and either improve same-window action loss relative to
E6 or prove that downstream action-head/backbone adaptation, not PICF routing,
is the remaining bottleneck.

## 2026-06-02 Six-GPU E23 Follow-Through

Use this launcher when a 6xA100-40GB machine is available:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_6x40g_e23_bucketbalanced_noactioncond_from11000_30k_20260602.sh
```

This is the maintained hardware-scaled follow-through after E23.  It keeps the
same model/loss contract as E23, but changes the gradient estimator from
2 bucket-balanced windows/update to 6 bucket-balanced windows/update.  The point
is to test the E21/E23 causal hypothesis without adding another module or
changing the action interface again.

Detailed math and guardrails:

```text
docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md
  section 45
```
