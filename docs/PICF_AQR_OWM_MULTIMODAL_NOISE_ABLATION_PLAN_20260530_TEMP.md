# PICF-AQR-OWM Multimodal Evidence Noise Ablation Plan - 2026-05-30

Canonical context:

```text
docs/PICF_AQR_OWM_ACTION_PLATFORM_ROOT_CAUSE_MAP_20260530_TEMP.md
docs/PICF_AQR_OWM_ANCHOR_LR_SAMPLING_CAUSAL_PLAN_20260530_TEMP.md
docs/PICF_AQR_OWM_ACTION_LOSS_WINDOW_AUDIT_20260530.md
```

## Why This Plan Exists

The current corrected runs show a specific pattern:

```text
structure/object metrics can improve;
active/downstream overlap can stay healthy;
action loss still platforms around the 0.04 band.
```

The correct hypothesis is not "modality count is bad."  The sharper hypothesis
is:

```text
weak multimodal evidence may be routed into high-confidence belief/prefix rows
without being causally aligned to the action target for the current window.
```

In formula form, the action path consumes:

```math
z^{act}_t = G(B_\phi(E^{visual}, E^{point}, E^{tactile}, E^{pg}, E^{sidecar}, E^{tracklet}, E^{cache}))
```

and optimizes:

```math
L_a = \ell(\pi_\theta(x_t, z^{act}_t), a_t).
```

The structural losses primarily improve:

```math
B_\phi(E) \approx \text{consistent object belief}
```

but action improves only if:

```math
I(z^{act}_t; a_t \mid x_t) > 0
```

and the extra evidence is not an action-irrelevant or wrong-object shortcut.
Therefore an evidence path can improve object-pull/support metrics while
failing to reduce action loss.

## What Has Already Been Excluded

From previous ledgers:

```text
raw reserve/context overlap:
  not the direct action-platform cause.

FSDP LR grouping:
  fixed and logged; not sufficient.

old resume RNG replay:
  fixed by step-indexed window RNG.

single-task consecutive-window explanation:
  not supported by reconstructed old stream.

blind SAM proposals:
  archived/rejected; not part of this plan.
```

## Current High-PICF-LR Branch Decision

Run:

```text
picf_a7_stepindexed_from8000_picflr002_action2_30k_20260530
```

Rows:

```text
step8050 action=0.04334 object_pull=0.318 active_ov=0.125 downstream_ov=0.119
step8100 action=0.04408 object_pull=0.245 active_ov=0.100 downstream_ov=0.092
step8150 action=0.04625 object_pull=0.112 active_ov=0.085 downstream_ov=0.086
```

Decision:

```text
Stop.  The branch improves structure but worsens action.  It answers the
PICF-LR hypothesis: raising the whole PICF core LR is not the primary action
platform fix.
```

## Experiment Family

All experiments must use:

```text
resume = step8000 corrected checkpoint
step_indexed_window_rng = true
ACTION_LOSS_WEIGHT = 2.0
semantic_trainable_scope = backbone_only
PICF_CORE_LR_SCALE = 0.005 unless the experiment explicitly says otherwise
unroll = 2
burnin = 1
same segment list as the clean full-sidecar run
log_interval = 50
anchor_overlay_interval = 100
```

### E1. No-Sidecar Optional Evidence

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
run_a7_stepindexed_from8000_nosidecar_h30k_20260530.sh
```

Invalid first launch:

```text
picf_a7_stepindexed_from8000_nosidecar_action2_300_20260530
```

Reason:

```text
It used NUM_TRAIN_STEPS=8300.  Because the training loop uses global
num_steps for the LR scheduler, resuming at step8000 under an 8300-step horizon
places the run near min_lr.  That confounds no-sidecar with a low-LR ablation.
This run must not be used for causal conclusions.
```

Cloud launch:

```text
machine:
  A7 / qgE72e

tmux:
  picf_a7_from8000_nosidecar_h30k_20260530

run:
  picf_a7_stepindexed_from8000_nosidecar_h30k_action2_20260530

log:
  /mnt/picf_run_logs/
  picf_a7_stepindexed_from8000_nosidecar_h30k_action2_20260530.log

resume:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000

duration:
  configured horizon is still 30000 for LR comparability;
  manually inspect/stop after global step 8100, 8150, or 8300.
```

Corrected live launch:

```text
2026-05-30 16:10 Asia/Shanghai

tmux:
  picf_a7_from8000_nosidecar_h30k_20260530

run:
  picf_a7_stepindexed_from8000_nosidecar_h30k_action2_20260530

validated startup:
  num_steps = 30000
  progress lr at step8001 ~= 6.18e-05
  mvtrack_sidecar_root = /mnt/picf_sidecars/empty_no_sidecar_20260530
  step_indexed_window_rng = true
  semantic = paligemma(trainable=True scope=backbone_only)
  Sonata / V-JEPA / AnyTouch = frozen

reason this is valid:
  It preserves the same global training horizon and LR schedule as the
  production branch, while deleting only optional sidecar proposal/tracklet
  evidence.
```

Tail commands:

```bash
ssh -p 28060 root@36.139.225.68
tmux attach -t picf_a7_from8000_nosidecar_h30k_20260530
tail -f /mnt/picf_run_logs/picf_a7_stepindexed_from8000_nosidecar_h30k_action2_20260530.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from8000_nosidecar_h30k_action2_20260530/metrics.jsonl
```

The launch uses a temporary cloud wrapper only because the segment-index list
is too long for a single `tmux new-session` command.  The local maintained
launcher records the same contract.

Mechanism:

```text
Use SEGMENTS from the clean full sidecar root, but point mvtrack_sidecar_root at
an empty directory.  CALVIN frames, model, optimizer settings, and sampling stay
unchanged.  Optional proposal/tracklet arrays become empty.
```

Expected runtime checks:

```text
owm_proposal_tokens ~= 0
owm_tracklet_tokens ~= 0
proposal/tracklet support metrics vanish or become zero
step8050/8100 action trend is compared against the full-sidecar step8050/8100
rows, not against the invalid 8300-horizon run.
```

Interpretation:

```text
If action improves while active/downstream structure remains acceptable:
  sidecar proposal/tracklet evidence is a net action-noise source in this
  training regime.

If action worsens:
  sidecar evidence is not the primary action platform cause and may be
  necessary for object-conditioned action.

If action is unchanged:
  the platform is downstream of sidecar, likely action/semantic interface or
  dense context utilization.
```


## E1 Live Rows

First valid row after corrected 30K-horizon launch:

```text
step8050 no-sidecar:
  loss_action_default_equiv = 0.0433337
  loss_action_active7 = 0.197095
  loss_anchor_pv = 0.590872
  loss_anchor_object_pull = 0.0
  owm_proposal_tokens = 0.0
  owm_tracklet_tokens = 0.0
  grad_norm = 0.437

strict full-sidecar branch at step8050:
  loss_action_default_equiv = 0.0425068
  loss_action_active7 = 0.192218
  loss_anchor_pv = 0.506283
  loss_anchor_object_pull = 0.338501
  owm_proposal_tokens = 1.35
  owm_tracklet_tokens = 59.13
```

Interpretation after first two valid rows:

```text
The ablation is mechanically valid: proposal/tracklet tokens are zero.

step8050:
  no-sidecar action = 0.0433337
  full-sidecar action = 0.0425068
  no-sidecar is slightly worse.

step8100:
  no-sidecar action = 0.0440745
  full-sidecar action = 0.0476807
  no-sidecar is better on action.

mean(8050, 8100):
  no-sidecar action = 0.043704
  full-sidecar action = 0.045094
  no-sidecar is modestly better on action.

structure cost:
  no-sidecar anchor_pv ~= 0.590
  full-sidecar anchor_pv ~= 0.506-0.523
  no-sidecar loses object-structure support.

Provisional conclusion:
  Sidecar is not a pure noise source.  It supports structure, but can inject
  action noise in some windows.  Continue to step8150 before closing because
  the first two rows are mixed.
```

### E2. Tactile Attachment Ablation

Only run after E1.

Goal:

```text
Separate tactile/contact evidence noise from sidecar proposal/tracklet noise.
```

Candidate change:

```text
Keep tactile encoder features available, but disable tactile-to-object-owner
attachment or reduce tactile anchor probability to zero.
```

Acceptance:

```text
If action improves and tactile/contact metrics stop dominating object-owner
rows, tactile attachment is too aggressive.
If action worsens, tactile is useful and should stay attached but may need a
stronger contact-quality gate.
```

### E3. Visual/PG/Point Only

Only run if E1 and E2 both implicate weak evidence.

Goal:

```text
Test whether the dense visual/point/PG path alone gives cleaner action
transfer.
```

This is more destructive and should not be the first test.

## Stop Rules

For each 300-step diagnostic:

```text
Stop early at 100 or 150 if:
  action is clearly worse than the full run and structure is not improved;
  nonfinite loss appears;
  proposal/tracklet/tactile ablation did not actually change the relevant
  token counts.

Continue to 300 if:
  action is close but structure changed in the intended direction;
  the first two rows are noisy but not rejected.
```

## Production Decision Rule

Do not change production from one live row.  A path is considered a real
candidate only if it passes:

```text
1. live action trend on corrected windows;
2. exact/fixed-window probe against step8000 and the current full branch;
3. active/downstream overlap health;
4. CALVIN/video behavior gate.
```

## E1 Final Decision

Valid rows:

```text
no-sidecar h30k from step8000:
  step8050 action=0.0433337 anchor_pv=0.590872 proposal=0 tracklet=0
  step8100 action=0.0440745 anchor_pv=0.590064 proposal=0 tracklet=0
  step8150 action=0.0462499 anchor_pv=0.585353 proposal=0 tracklet=0

continuous full-sidecar branch:
  step8050 action=0.0425068 anchor_pv=0.506283 proposal=1.35 tracklet=59.13
  step8100 action=0.0476807 anchor_pv=0.522711 proposal=1.14 tracklet=59.43
  step8150 action=0.0503612 anchor_pv=0.510741 proposal=1.26 tracklet=60.25

high-PICF-LR full-sidecar reset branch:
  step8050 action=0.0433395 anchor_pv=0.508142 proposal=1.35 tracklet=59.13
  step8100 action=0.0440822 anchor_pv=0.512849 proposal=1.14 tracklet=59.43
  step8150 action=0.0462490 anchor_pv=0.492873 proposal=1.26 tracklet=60.25
```

Decision:

```text
Stop no-sidecar.

The no-sidecar action sequence is almost identical to the high-PICF-LR
full-sidecar reset sequence, while its structure metric is worse.  Therefore the
action improvement versus the continuous full-sidecar branch is more likely
caused by the step8000 restart / optimizer-state change than by removing
sidecar evidence.  Sidecar is not the dominant action-platform noise source.
```

Next causal test:

```text
E1b full-sidecar-reset control:
  resume the same step8000 checkpoint;
  keep full sidecar;
  keep production PICF_CORE_LR_SCALE=0.005;
  keep NUM_TRAIN_STEPS=30000;
  run to 8100/8150.

If E1b matches no-sidecar/high-PICF action:
  optimizer restart/reset explains the apparent short-run improvement.

If E1b matches continuous full-sidecar action:
  the explanation is not optimizer reset and sidecar/PICF interactions need a
  second-order test.
```

## E1b Full-Sidecar Reset Control Launch

```text
2026-05-30 17:22 Asia/Shanghai

tmux:
  picf_a7_from8000_fullsidecar_reset_h30k_20260530

run:
  picf_a7_stepindexed_from8000_fullsidecar_reset_h30k_action2_20260530

validated startup:
  resume = step8000 corrected checkpoint
  num_steps = 30000
  progress lr at step8001 ~= 6.18e-05
  mvtrack_sidecar_root = /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
  PICF_CORE_LR_SCALE = 0.005
  action_weight = 2.0
  semantic = paligemma(trainable=True scope=backbone_only)
  Sonata / V-JEPA / AnyTouch = frozen

Purpose:
  If E1b matches the no-sidecar/high-PICF-reset action sequence, the short-run
  action improvement is explained by restart/optimizer-state dynamics rather
  than sidecar removal.
```

Tail commands:

```bash
ssh -p 28060 root@36.139.225.68
tmux attach -t picf_a7_from8000_fullsidecar_reset_h30k_20260530
tail -f /mnt/picf_run_logs/picf_a7_stepindexed_from8000_fullsidecar_reset_h30k_action2_20260530.log
tail -f /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from8000_fullsidecar_reset_h30k_action2_20260530/metrics.jsonl
```

## E1b Final Decision

Valid row:

```text
full-sidecar reset h30k from step8000:
  step8050 action=0.0433176
  anchor_pv=0.504053
  object_pull=0.295201
  proposal_tokens=1.35
  tracklet_tokens=59.13

no-sidecar reset h30k from step8000:
  step8050 action=0.0433337
  anchor_pv=0.590872
  object_pull=0.0
  proposal_tokens=0
  tracklet_tokens=0

high-PICF-LR full-sidecar reset from step8000:
  step8050 action=0.0433395
  anchor_pv=0.508142
  object_pull=0.318132
  proposal_tokens=1.35
  tracklet_tokens=59.13

continuous full-sidecar branch:
  step8050 action=0.0425068
  anchor_pv=0.506283
```

Decision:

```text
Stop E1b at step8050.

The reset branches produce the same action row regardless of sidecar deletion or
PICF LR increase.  The no-sidecar branch worsens structure.  Therefore optional
sidecar proposal/tracklet evidence is not the dominant root cause of the action
platform/rebound.  The apparent short-run action improvement belongs to restart
/ optimizer-state / resumed-dynamics effects.

Do not remove sidecar by default.  Keep sidecar as weak typed evidence with the
existing quality gates.  The next root-cause target should be action-side
optimization/transfer and checkpoint/resume optimizer dynamics, not multimodal
evidence deletion.
```
