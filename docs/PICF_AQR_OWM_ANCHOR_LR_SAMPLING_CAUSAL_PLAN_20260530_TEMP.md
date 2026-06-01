# PICF Anchor LR vs Sampling Root-Cause Plan - 2026-05-30

## Question

Does the current action plateau/rebound require raising PICF/anchor learning rate,
or is it mainly a sampled-window / shortcut / train-stream-estimator issue?

## Current Evidence

The May-29 action rebound experiments already falsified a pure anchor-LR cause:

- `policy_only + action_head_only` reproduced the old 7550/7600 spike with PICF frozen.
- near-zero-LR replay reproduced the same spike with effectively no parameter update.
- the trainer had a resume RNG defect; step-indexed window RNG now fixes replaying the early window stream.
- current production run uses `step_indexed_window_rng=True` and verified optimizer groups.

Current A7 run from step8000:

- `picf_core lr ~= 3.07e-7`, policy head `~= 6.14e-5`.
- `loss_action_default_equiv` at 8050..8200: about `0.0433, 0.0441, 0.0462, 0.0461`.
- `loss_total_minus_action` remains about `0.010..0.012`.
- active/downstream support overlap remains healthy, about `0.10..0.16`.
- `loss_anchor_pv` is near `0.50`; `loss_anchor_object_pull` oscillates.

This means the run is not an active-slot collapse.  But it also means the live
train-stream action rows are not enough to decide whether action quality is
improving.

## Mathematical Decomposition

For a log interval `k`, the train metric is:

```math
\hat L_k(\theta_k) = |B_k|^{-1}\sum_{w \in B_k} L(\theta_k; w).
```

A change in the logged scalar decomposes as:

```math
\hat L_{k+1}(\theta_{k+1}) - \hat L_k(\theta_k)
=
[\hat L_{k+1}(\theta_k) - \hat L_k(\theta_k)]
+
[\hat L_{k+1}(\theta_{k+1}) - \hat L_{k+1}(\theta_k)].
```

The first term is sampled-window difficulty / train-stream estimator noise.
The second term is the actual model update effect.  Raising anchor LR only
addresses the second term and only the PICF-core component of it.

Therefore an anchor-LR intervention is not justified until the no-update replay
shows that the same windows are easy for the checkpoint but become hard only
after updates.

## Causal Experiment Order

### E1. Same-window no-update replay from step8000

Run from the same step8000 checkpoint, same seed, step-indexed RNG, same window
stream, near-zero LR and no meaningful update.

Expected interpretation:

- If E1 matches the live action curve, the live rise is mostly sampled-window
  difficulty.  Do not raise anchor LR as a root-cause fix.
- If E1 is flat/lower while live training rises, updates are damaging the model.
  Then split whether damage is PICF/anchor or semantic/action by E2/E3.

### E2. Higher PICF-core/anchor LR branch from step8000

Only after E1 rejects sampled-window dominance.  Run the same 300-step window
stream with `PICF_CORE_LR_SCALE=0.02` or an anchor-router subgroup if available.

Interpretation:

- If anchor losses improve and action does not degrade, current 0.005 core scale
  is too conservative for anchor refinement.
- If action degrades or fixed-window action worsens, raising all PICF core LR is
  unsafe; only a narrower anchor-router LR group should be considered.

### E3. Fixed-window probe at saved checkpoints

Use `scripts/picf_fixed_window_action_probe.py` on identical accepted windows.
This is the production decision metric, not raw live rows.

## Current Decision Rule

Do not callback PICF/anchor LR in the live production run yet.  First run E1.
If GPU time is limited, stop the current live run after preserving step8000 and
run E1 because it has higher causal value than continuing the same live stream.

## E1 Execution - 2026-05-30

The low-value live continuation was stopped after preserving the step8000
checkpoint.  A same-checkpoint, same-step-indexed-window, near-zero-LR replay
was launched:

```text
run:
  picf_a7_no_update_replay_from8000_8300_20260530

resume:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000

contract:
  num_train_steps = 8300
  step_indexed_window_rng = true
  LR = 1e-12
  MIN_LR = 0
  semantic_backbone lr = 3.5e-13
  picf_core lr = 5e-15
  policy_head lr = 1e-12
```

First matched 50-step row:

```text
step8050 live continuation:
  loss_action_default_equiv = 0.0433267
  loss_total_minus_action   = 0.0124034
  active/downstream overlap = 0.13997 / 0.14601

step8050 near-zero replay:
  loss_action_default_equiv = 0.0431826
  loss_total_minus_action   = 0.0106551
  active/downstream overlap = 0.12500 / 0.12914
  picf_core lr              = 1.13e-17
```

Interpretation:

```text
delta action(live - replay) = +0.000144
```

This first gate strongly supports sampled-window/train-estimator dominance for
the 8000->8050 row.  It does not support an immediate PICF/anchor-LR callback
as the root-cause fix.  Let the replay continue through at least 8100 before
closing the causal split, but the first matched row already rules out a large
model-update contribution in this interval.

Second matched row:

```text
step8100 live continuation:
  loss_action_default_equiv = 0.0440565
  loss_total_minus_action   = 0.0104006
  active/downstream overlap = 0.15998 / 0.14432

step8100 near-zero replay:
  loss_action_default_equiv = 0.0440217
  loss_total_minus_action   = 0.0119431
  active/downstream overlap = 0.17500 / 0.16174
  picf_core lr              = 7.23e-18
```

```text
delta action(live - replay) = +0.0000348
```

The second matched row confirms the first: the live action scalar is almost
identical to the no-update replay on the same step-indexed windows.  This makes
an action-loss-driven PICF/anchor LR callback unjustified.

Third matched row:

```text
step8150 live continuation:
  loss_action_default_equiv = 0.0462482
  loss_total_minus_action   = 0.0120256
  active/downstream overlap = 0.16000 / 0.16017

step8150 near-zero replay:
  loss_action_default_equiv = 0.0458736
  loss_total_minus_action   = 0.0106064
  active/downstream overlap = 0.11499 / 0.11990
  picf_core lr              = 4.08e-18
```

```text
delta action(live - replay) = +0.000375
```

Three consecutive matched rows now agree:

```text
8050 delta = +0.000144
8100 delta = +0.0000348
8150 delta = +0.000375
```

This closes the immediate anchor-LR/action-rebound split for the inspected
8000->8150 interval.  The action train-row values are sampled-window estimator
values, not evidence that PICF/anchor updates are damaging the model.

Trace validation:

```text
rank0 live/replay first 10 flat_index values:
  [402, 7392, 2045, 4883, 4722, 6854, 1843, 4713, 3289, 2537]

rank1 live/replay first 10 flat_index values:
  [3949, 1674, 609, 754, 3614, 7662, 6561, 5053, 3566, 1581]
```

The matched-row conclusion is therefore not a false comparison between
different sampled windows.

## E1 Closure

E1 was stopped after the 8150 matched row because it had already answered the
causal question it was designed to answer.  Continuing the near-zero replay to
8300 would spend GPU time measuring the same sampled-window estimator with
almost no model update:

```text
8050 action delta live-replay = +0.000144
8100 action delta live-replay = +0.0000348
8150 action delta live-replay = +0.000375
```

These deltas are far below the previously discussed reject band.  The correct
classification for the inspected interval is:

```text
primary cause of the logged 8000->8150 action rows:
  sampled-window/train-stream estimator difficulty

not supported as primary cause:
  PICF/anchor update damage
  anchor LR too low or too high
  raw same-role overlap collapse
  missing sidecar evidence
```

This does not prove that the model is behaviorally final.  It only proves that
the next decision must be made with fixed-window probes and behavior evidence,
not with another live-row action rebound repair.

## Next Experiment Gate

```text
Launch a normal continuation from the preserved step8000 checkpoint:

  resume:
    /mnt/checkpoints/picf_core/picf_core/
    picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000

  run:
    picf_a7_stepindexed_from8000_gate9000_20260530

  unchanged production contract:
    action_weight = 2.0
    lr = 7e-5
    min_lr = 2e-5
    semantic_lr_scale = 0.35
    picf_core_lr_scale = 0.005
    semantic_trainable_scope = backbone_only
    optimizer_checkpoint_mode = model-only
    unroll = 2
    burnin = 1
    step_indexed_window_rng = true
    sidecar_root = contact_motion_full_tracklets_clean_20260520

At step9000 and then step10000, run `scripts/picf_fixed_window_action_probe.py`
on the same accepted flat-index set used for the step7000/8000 ladder.  The
live train rows are allowed to oscillate; they are not the production gate.

Decision:

  if fixed-window action improves from step8000 and active/downstream overlap,
  object-pull, sidecar coverage, and prefix-stability remain bounded:
    continue the 30K run.

  if fixed-window action worsens while the structure gates remain healthy:
    isolate semantic/action capacity or data-window curriculum; do not patch
    raw overlap or anchor LR.

  if fixed-window action improves but object-pull/owner metrics drift:
    keep the run as an action candidate but schedule a separate narrow
    router/owner ablation.  Do not raise the whole PICF core group in the
    production branch.
```

## Step8000 Continuation Launch - 2026-05-30

The normal continuation was launched on A7:

```text
tmux:
  picf_a7_stepindexed_from8000_gate9000_20260530

log:
  /mnt/picf_run_logs/picf_a7_stepindexed_from8000_gate9000_20260530.log

checkpoint dir:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_from8000_gate9000_20260530
```

Startup contract verified from the cloud log:

```text
resume checkpoint:
  .../picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000

num_steps:
  30000

effective_global_batch:
  2

unroll/burnin:
  unroll_steps = 2
  burnin_steps = 1
  burnin_mode = state_only

window RNG:
  step_indexed_window_rng = true

checkpoint/diagnostic cadence:
  save_interval = 1000
  keep_last_checkpoints = 5
  log_interval = 50
  anchor_overlay_interval = 100

optimizer groups:
  semantic_backbone lr = 2.45e-5
  picf_core lr = 3.50e-7
  policy_head lr = 7.00e-5

frozen/trainable boundary:
  Sonata point encoder = frozen
  V-JEPA visual encoder = frozen
  AnyTouch tactile encoder = frozen
  PaliGemma semantic backbone = trainable, backbone_only scope
```

This run is intentionally not an architecture repair.  It is the clean
continuation after E1 ruled out immediate update damage.  The next decisive
artifact is the fixed-window probe at step9000/10000.  Do not stop or alter
this branch based only on the first raw live train rows unless a hard runtime
fault, NaN, checkpoint failure, or active-owner collapse appears.

## Step8000 Continuation Health Check - 2026-05-30 02:06 CST

Remote status:

```text
tmux:
  picf_a7_stepindexed_from8000_gate9000_20260530 is alive

GPU:
  both A100s allocated and active

trace files:
  metrics.jsonl exists
  window_trace_rank0.jsonl exists
  window_trace_rank1.jsonl exists
```

First structured row:

```text
step                              8050
loss_total                        0.05565
loss_action_default_equiv          0.04334
loss_total_minus_action            0.01231
loss_anchor_pv                     0.50275
loss_anchor_object_pull            0.35599
loss_anchor_object_pull_graph      0.24404
loss_anchor_object_pull_posterior  0.36674
active same-role overlap           0.12995
downstream overlap                 0.12960
raw same-role overlap              1.00000
posterior_recycle_rate             0.11456
posterior_identity_switch_rate     0.20222
loss_slot_jepa                     0.87767
preclip_grad_norm                  0.44300
proposal tokens                    1.35
tracklet tokens                    59.13
speed                              about 25 sec/step
```

Interpretation:

```text
PASS runtime health:
  no NaN
  no checkpoint/log failure
  no sidecar disappearance
  no active-owner collapse
  no optimizer-group drift
  no mistaken near-zero-LR replay

WATCH:
  raw overlap remains saturated, but this is the known reserve/context
  telemetry and is not the active-owner collapse signature.

NEXT:
  let the run continue to step9000/10000, then run the fixed-window action
  probe before making another architecture/LR decision.
```

## Step9000/9200 Health Check - 2026-05-30 10:27 CST

Remote status:

```text
tmux:
  picf_a7_stepindexed_from8000_gate9000_20260530 is alive

checkpoint:
  step9000 saved
  model.pt about 9.3G

overlays:
  anchor_overlays are written every 100 steps
  variants include active_only, with_gray, mask_only, mask_active,
  mask_with_gray, and sidecar_proposals

trace:
  metrics.jsonl has 24 rows
  window_trace_rank0/1 are active
```

Recent rows:

```text
step   action   total  non_action  anchor_pv  obj_pull  active_ov  downstream_ov  recycle  id_switch  slot_jepa
8650   0.04394  0.05499       0.01105     0.51052    0.23548   0.14000    0.13306        0.12140  0.19667    2.07517
8700   0.04828  0.05986       0.01158     0.49391    0.27965   0.09946    0.10169        0.11925  0.21333    1.56140
8750   0.04360  0.05404       0.01044     0.51231    0.17580   0.09000    0.10016        0.11915  0.19778    1.33792
8800   0.04551  0.05630       0.01079     0.48955    0.20767   0.08498    0.08917        0.11897  0.18778    1.21797
8850   0.04241  0.05292       0.01050     0.49726    0.17886   0.10000    0.09664        0.10950  0.20500    1.10134
8900   0.04187  0.05171       0.00983     0.49586    0.11420   0.10000    0.09919        0.11075  0.20000    1.16894
8950   0.04415  0.05419       0.01004     0.50404    0.13306   0.16500    0.14801        0.10010  0.20722    1.05866
9000   0.04264  0.05445       0.01181     0.49996    0.30118   0.17500    0.16624        0.09728  0.21833    1.11144
9050   0.03979  0.04941       0.00963     0.50249    0.09941   0.14000    0.15902        0.09678  0.22389    1.07630
9100   0.04490  0.05445       0.00955     0.48872    0.08878   0.12500    0.11214        0.08610  0.22833    1.00258
9150   0.04471  0.05491       0.01020     0.50632    0.15725   0.13500    0.10789        0.07822  0.19611    1.04595
9200   0.04678  0.05668       0.00990     0.50192    0.12098   0.08500    0.07896        0.08850  0.17889    0.88318
```

Interpretation:

```text
PASS:
  runtime is normal
  step9000 checkpoint exists
  overlays and traces exist
  sidecar proposal/tracklet tokens are present
  active/downstream overlap remains in the healthy low band
  object-pull is materially better than the early step8050/8650 rows
  slot-JEPA diagnostic is decreasing, not exploding
  prefix teacher remains stable

WATCH:
  action is still a sampled-window train-row scalar in the 0.04 band
  raw overlap remains saturated but is not the active-owner metric
  posterior identity switch remains around 0.18..0.23 and still requires
  fixed-window/behavior validation

Decision:
  keep the run alive.  The next scientific gate is a fixed-window action probe
  for step9000/10000 rather than another architecture change.
```

## Step8000 vs Step9000 Fixed-Window Gate - 2026-05-30 11:05 CST

Probe:

```text
tmux:
  picf_fixed_probe_8000_9000_20260530

artifacts:
  /mnt/picf_fixed_window_probes/step8000_vs_9000_20260530/
    step8000_summary.json
    step9000_summary.json
    step8000_windows.jsonl
    step9000_windows.jsonl
    accepted_flat_indices.json
```

Same accepted windows:

```text
accepted_windows = 64
split = training
effective_unroll_steps = 3
sidecar = /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
```

Summary:

```text
metric                                  step8000      step9000      delta
loss_action_default_equiv               0.057919      0.057213     -0.000706
loss_total_minus_action                 0.323225      0.273089     -0.050136
loss_anchor_object_pull                 0.452008      0.304943     -0.147064
loss_anchor_pv                          0.509784      0.516780     +0.006995
loss_aqr_denoising                      1.172452      1.229527     +0.057075
loss_mapg_routing                       0.423328      0.439612     +0.016284
loss_slot_jepa                          1.175297      1.004094     -0.171203
active same-role support overlap        0.085937      0.179688     +0.093750
downstream same-role support overlap    0.093724      0.165981     +0.072257
posterior_recycle_rate                  0.123171      0.094072     -0.029099
```

Paired window counts:

```text
loss_action_default_equiv:
  improved 29 / 64
  worsened 35 / 64
  median_delta = +0.000633

posterior_recycle_rate:
  improved 63 / 64
  worsened 1 / 64

anchor_object_pull:
  improved 33 / 64
  worsened 23 / 64

active/downstream overlap:
  mean worsened, median unchanged at 0 because many rows have no same-role
  active pair.
```

Interpretation:

```text
The continuation from step8000 to step9000 is not a convincing action
optimization success.  The action mean improves only by 0.000706, fewer than
half of paired windows improve, and the paired median delta is slightly worse.

The model does improve some structural/scaffold metrics, especially recycle and
object-pull mean, but this did not transfer into a meaningful action gain over
the same windows.
```

Decision:

```text
Do not resume this branch as a blind 30K production run from action evidence.
The preserved step9000 checkpoint remains useful for analysis, but the current
recipe has not passed the fixed-window action gate.

Next choices must be explicit:
  1. run CALVIN/video eval on step8000 and step9000 if behavior evidence is
     needed before discarding the branch;
  2. run a targeted action-capacity/semantic-interface experiment;
  3. compare against the 4-22 fixed-window baseline, not the nonstationary old
     train-stream 0.02 row.
```

## E2 Controlled PICF-LR Callback - 2026-05-30

User hypothesis:

```text
The corrected step-indexed run may expose many genuinely new windows.  With
PICF_CORE_LR_SCALE=0.005, the belief/router/anchor side may adapt too slowly;
if anchors or owner routing materially change, a slightly higher core LR may
be required for the action path to benefit from new evidence.
```

Important constraint:

```text
This does not revert the E1 conclusion.  E1 proved that the immediate
8000->8150 live-row action fluctuation was mostly sampled-window estimator
difficulty, not update damage.  E2 asks a different question: whether a more
plastic PICF core improves medium-horizon same-window action and structure
gates after the clean RNG fix.
```

Experiment:

```text
run:
  picf_a7_stepindexed_from8000_picflr002_action2_30k_20260530

launcher:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from8000_picflr002_30k_20260530.sh

resume:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000

single intended intervention:
  PICF_CORE_LR_SCALE = 0.02

unchanged:
  step_indexed_window_rng = true
  action_weight = 2.0
  lr = 7e-5
  min_lr = 2e-5
  semantic_trainable_scope = backbone_only
  semantic_lr_scale = 0.35
  policy_head_lr_scale = 1.0
  sidecar_root = contact_motion_full_tracklets_clean_20260520
  action_prefix EMA / RMSNorm / output gate unchanged
  object scaffold decay unchanged
  unroll = 2
  burnin = 1
  save_interval = 1000
  keep_last_checkpoints = 5
  log_interval = 50
  anchor_overlay_interval = 100
```

Expected optimizer scale:

```text
base LR around launch:
  7e-5 before schedule decay

semantic backbone:
  0.35 * base LR

policy/action head:
  1.0 * base LR

PICF core:
  0.02 * base LR
```

Therefore PICF remains much slower than action/semantic but is no longer nearly
stationary.  This is the intended two-timescale regime: belief-state adaptation
is allowed, while the action head still dominates task optimization.

Acceptance:

```text
Pass if, by the first 100-300 live steps and the next fixed-window probe:
  loss_action_default_equiv does not degrade versus the step8000 baseline;
  fixed-window action improves more than the 0.0007 seen at step9000;
  loss_total_minus_action stays bounded near the previous 0.010-0.015 band;
  active/downstream same-role overlap remains below the collapse band;
  loss_anchor_object_pull does not explode;
  loss_slot_jepa remains bounded and non-explosive;
  lr_group_picf_core and grad_norm_group_picf_core are present and nonzero.

Fail if:
  active/downstream overlap rises into collapse;
  action fixed-window worsens while structure improves;
  PICF group metrics are missing, implying a grouping or logging bug;
  nonfinite loss, checkpoint failure, or sidecar disappearance appears.
```

This is a controlled capacity/adaptation test.  It is not evidence by itself
that the final production recipe should use `0.02`; that decision requires the
same-window probe and behavior/video gates.

Launch status:

```text
machine:
  A7 / qgE72e

tmux:
  picf_a7_from8000_picflr002_20260530

log:
  /mnt/picf_run_logs/
  picf_a7_stepindexed_from8000_picflr002_action2_30k_20260530.log

checkpoint dir:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_from8000_picflr002_action2_30k_20260530

verified startup:
  resume step = 8000
  num_steps = 30000
  effective_global_batch = 2
  save_interval = 1000
  keep_last_checkpoints = 5
  log_interval = 50
  anchor_overlay_interval = 100
  step_indexed_window_rng = true
  unroll_steps = 2
  burnin_steps = 1

verified optimizer groups:
  semantic_backbone lr = 2.45e-05
  picf_core lr = 1.40e-06
  policy_head lr = 7.00e-05

verified frozen/trainable boundary:
  Sonata = frozen
  V-JEPA = frozen
  AnyTouch = frozen
  PaliGemma semantic backbone = trainable, backbone_only
```

Initial runtime:

```text
first visible progress:
  step8001 loss ~= 0.0541
  speed ~= 26.45 sec/step
  both A100s at 100% utilization
```
