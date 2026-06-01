# PICF-AQR-OWM Current Plateau Root-Cause Plan - 2026-05-29

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md
docs/PICF_AQR_OWM_DATA_WINDOW_DISTRIBUTION_AUDIT_20260529_TEMP.md
```

## Question

The active A7 corrected step-indexed continuation:

```text
picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529
```

resumes the preserved EMA step7000 model and runs the current production
recipe:

```text
visual_mode=encoder
tactile_mode=encoder
semantic_mode=paligemma
semantic_trainable_scope=backbone_only
perception_finetune_mode=frozen
unroll_steps=2
burnin_steps=1
ACTION_LOSS_WEIGHT=2.0
PICF_CORE_LR_SCALE=0.005
POLICY_HEAD_LR_SCALE=1.0
step_indexed_window_rng=True
sidecar=/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
```

It does not show the old raw-overlap collapse, but it also does not recover the
old `0.02` action band.  The scientific question is whether this is:

```text
A. a model/optimizer/interface defect;
B. a sampled-train-window measurement artifact;
C. a stale checkpoint basin that should be abandoned;
D. a sidecar/dataflow defect.
```

## Latest Evidence

Latest remote rows inspected through step8250:

```text
step   action_default  total_minus_action  anchor_pull  anchor_pv  active_ov  downstream_ov
7900   0.04251         0.01135             0.2657       0.5003     0.1100     0.1091
7950   0.04852         0.01094             0.2244       0.5094     0.1193     0.1123
8000   0.04428         0.01034             0.1667       0.4844     0.0750     0.0783
8050   0.04251         0.01211             0.3385       0.5063     0.1650     0.1604
8100   0.04768         0.01091             0.2189       0.5227     0.1082     0.0922
8150   0.05036         0.01274             0.3876       0.5107     0.1343     0.1313
8200   0.04346         0.01374             0.5090       0.5190     0.1373     0.1494
8250   0.05154         0.01504             0.6088       0.5166     0.1200     0.1227
```

The recent eight-row action mean worsened:

```text
mean(7900..8050) = 0.04445
mean(8100..8250) = 0.04826
delta            = +0.00381
```

This is not a useful continuation signal.

## Fixed-Window Probe Result

The decisive no-update probe was run on 64 identical accepted CALVIN training
windows.  It compares checkpoints with no backward pass, no optimizer update,
and the same flat indices:

```text
current step7000:
  checkpoint: /mnt/checkpoints/picf_core/picf_core/picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000
  mean loss_action_default_equiv: 0.061891
  mean loss_total_minus_action:   0.284341
  mean loss_anchor_object_pull:   0.342123
  mean loss_anchor_pv:            0.528139
  mean active same-role overlap:  0.062867
  mean posterior_recycle_rate:    0.131462

current step8000:
  checkpoint: /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000
  mean loss_action_default_equiv: 0.057919
  mean loss_total_minus_action:   0.323225
  mean loss_anchor_object_pull:   0.452007
  mean loss_anchor_pv:            0.509784
  mean active same-role overlap:  0.085937
  mean posterior_recycle_rate:    0.123171
```

The current model did **not** get worse on stationary action quality from
step7000 to step8000:

```text
delta action_default_equiv = -0.003971
```

However, the object-scaffold/non-action side became more expensive on the same
windows:

```text
delta total_minus_action = +0.038883
delta object_pull        = +0.109884
delta active_overlap     = +0.023071
```

This means the live train-stream "rebound" is not evidence of direct action
interface destruction.  It is better described as a combination of sampled
window mixture plus non-action scaffold drift.  Future stop/continue decisions
must use fixed-window probes rather than raw live rows.

The same accepted windows were also evaluated on the archived 2026-04-22
PI0.5-only ablation:

```text
4-22 ablation step7500  fixed action mean: 0.058717
4-22 ablation step10000 fixed action mean: 0.053813
4-22 ablation step20000 fixed action mean: 0.051038
```

Therefore the old train-log `0.02` scalar is **not** a stationary fixed-window
quality target.  On this fixed window set, the current step8000 model is roughly
at the archived 4-22 step7500 level and trails the archived step20000 model by
about `0.0069` absolute action loss.  The archived 25k checkpoint is not present
on the inspected A7 storage, so 20k is the strongest fixed-window old baseline
currently available.

## Already Excluded

The experiment record excludes several tempting but wrong explanations.

### Raw overlap is not the late-action root cause

Raw same-role overlap can be `1.0`, but active/downstream overlap is controlled:

```text
aqr_active_same_role_support_overlap_max     ~= 0.07..0.16
aqr_downstream_same_role_support_overlap_max ~= 0.08..0.16
```

Historical action-head-only, policy-semantic-only, and fixed-group runs all
reproduced the late action band while `loss_total_minus_action` was zero or
small.  Therefore another scalar overlap penalty is not a root-cause fix.

### Sidecar coverage is not collapsing

The cloud data-window audit shows stable sidecar coverage:

```text
proposal_count mean            ~= 1.17..1.33
proposal_mask_point_count mean ~= 38.7..44.7
tracklet_count mean            ~= 57.8..61.8
```

The sampled train windows do vary by task, but not enough to explain away the
whole `0.04..0.05` plateau.

### Action logging is not a display bug

For PI0.5 action-flow override:

```math
L_a^{default} = L_{flow}
```

and when `ACTION_LOSS_WEIGHT=2.0`:

```math
L_a = L_a^{default}.
```

Therefore `loss_action_default_equiv` is the correct historical comparison
scalar.

### FSDP LR grouping is not the sole cause

The wrapped-name bug was fixed by canonicalizing FSDP/DDP parameter owner names
before group assignment.  The current run logs:

```text
lr_group_policy_head      ~= 6.13e-5
lr_group_semantic_backbone~= 2.14e-5
lr_group_picf_core        ~= 3.06e-7
```

The plateau persists with the intended two-timescale split.

## Mathematical Diagnosis

For a sampled window `W_k`, the logged action row is:

```math
\bar L_a(K) =
\frac{1}{50R}\sum_{k=K-49}^{K}\sum_{r=1}^{R}
L_a(\theta_K; W_{k,r}).
```

This is not a stationary validation loss.  The actual checkpoint quality should
be:

```math
J_a(\theta) = \mathbb{E}_{W \sim \mathcal{D}_{fixed}}[L_a(\theta; W)].
```

The current live metric entangles:

```math
\bar L_a(K) - \bar L_a(K-50)
=
\underbrace{J_a(\theta_K)-J_a(\theta_{K-50})}_{model change}
+ \underbrace{\epsilon_K-\epsilon_{K-50}}_{sampled-window mixture}
+ \underbrace{\delta_K}_{stateful train-mode noise}.
```

Past near-zero-LR replay proved that a large part of the 7550/7600 spike can
occur without parameter movement.  The current run uses step-indexed RNG, so it
does not replay the exact old early windows, but it still logs train-window
estimates rather than fixed validation estimates.

The root-cause class is therefore now narrowed to:

```text
1. action-visible interface/checkpoint quality under fixed windows;
2. stale low-loss basin from the old EMA step7000 continuation;
3. live train-window measurement noise;
4. semantic/action-side optimizer stability after a model-only resume.
```

It is not currently narrowed to:

```text
raw overlap,
missing sidecar,
slot-JEPA,
another object pull scalar,
or a broad PICF structural collapse.
```

## Required Decisive Test Status

Stop using live sampled-window rows as the primary decision.  The fixed
window no-update probe is now the maintained comparison tool:

```text
scripts/picf_fixed_window_action_probe.py
```

on identical accepted flat indices for:

```text
old EMA step7000 checkpoint
current corrected step8000 checkpoint
optional current step9000 checkpoint if it exists later
```

Acceptance logic after the 2026-05-29 result:

```text
If fixed-window action improves while live rows worsen:
  do not rewrite architecture from live rows alone; treat sampled-window
  mixture as the primary explanation and monitor fixed-window validation.

If fixed-window action worsens together with non-action drift:
  continuation is damaging the action-visible interface; stop that branch.

If current fixed-window action stays far above the archived 20k PI0.5-only
ablation:
  prioritize behavior/CALVIN eval and an action-capacity diagnostic, not raw
  overlap penalties.
```

## Decision For Current Run

Current run has low optimization value after step8250:

```text
action_default recent mean worsened;
total_minus_action and anchor_object_pull increased;
active/downstream structure stayed healthy;
only checkpoint currently worth preserving is step8000.
```

If GPU time is needed, it is scientifically defensible to stop the run after
preserving the step8000 output and use the machine for:

```text
1. behavior/CALVIN evaluation of the preserved step8000 checkpoint;
2. fixed-window probes for later checkpoints if training resumes;
3. if behavior fails, an action-capacity diagnostic rather than another
   raw-overlap repair.
```

Do not launch another large architecture rewrite from this evidence alone.  The
next action must first separate checkpoint quality from sampled-window noise.

## Follow-Up Run Launched

Because fixed-window action improved from current step7000 to current step8000,
the scientifically justified continuation is from the preserved step8000 model,
not from scratch and not from another architecture rewrite.

```text
run:
  picf_a7_fixedprobe_from8000_action2_30k_20260529

resume:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000

log:
  /mnt/picf_run_logs/picf_a7_fixedprobe_from8000_action2_30k_20260529.log

metrics:
  /mnt/checkpoints/picf_core/picf_core/picf_a7_fixedprobe_from8000_action2_30k_20260529/metrics.jsonl

settings:
  NUM_TRAIN_STEPS=30000
  SAVE_INTERVAL=1000
  KEEP_LAST_CHECKPOINTS=5
  LOG_INTERVAL=50
  ANCHOR_OVERLAY_INTERVAL=100
  ACTION_LOSS_WEIGHT=2.0
  LR=7e-5
  MIN_LR=2e-5
  PICF_CORE_LR_SCALE=0.005
  SEMANTIC_TRAINABLE_SCOPE=backbone_only
  OPTIMIZER_CHECKPOINT_MODE=model-only
```

Startup sanity:

```text
resumed at step=8000;
world_size=2;
backbone contract: Sonata/V-JEPA/AnyTouch frozen, PaliGemma trainable
scope=backbone_only;
optimizer groups:
  semantic_backbone lr=2.45e-5
  picf_core lr=3.5e-7
  policy_head lr=7e-5
observed speed after startup: about 24.6 sec/step.
```

The first metric row will appear at step8050.  If later live rows worsen again,
the next decision must be a fixed-window probe at the saved checkpoint rather
than another raw-overlap repair.
