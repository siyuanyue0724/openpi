# PICF-AQR-OWM Data Window Distribution Audit - 2026-05-29

Canonical related notes:

```text
src/openpi/picf/README_v2.2.md
docs/PICF_AQR_OWM_ACTION_REBOUND_ROOT_CAUSE_20260529_TEMP.md
docs/PICF_AQR_OWM_STEPINDEXED_RUN_LOCAL_AUDIT_20260529_TEMP.md
docs/PICF_AQR_OWM_COTRAIN_TWO_TIMESCALE_FINAL_20260526.md
```

This note audits what the current A7 continuation is actually reading from
CALVIN and from the contact-motion sidecar. The purpose is to separate three
failure classes that look similar in train loss:

```text
A. model/optimizer regression;
B. stochastic task-window composition;
C. sidecar/proposal/tracklet coverage defects.
```

## Active Run Under Audit

```text
remote:  A7 / qgE72e
run:     picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529
repo:    /root/openpi_probe_current_20260529
log:     /mnt/picf_run_logs/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529.log
ckpt:    /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529
sidecar: /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
calvin:  /mnt/calvin_data/task_ABC_D
```

Confirmed runtime settings from the process command and launcher chain:

```text
visual_mode=encoder
tactile_mode=encoder
semantic_mode=paligemma
semantic_trainable=True
semantic_trainable_scope=backbone_only
perception_finetune_mode=frozen
unroll_steps=2
burnin_steps=1
burnin_mode=state_only
step_indexed_window_rng=True
accum_steps=1
world_size=2
ACTION_LOSS_WEIGHT=2.0
LR=7e-5
PICF_CORE_LR_SCALE=0.005
POLICY_HEAD_LR_SCALE=1.0
mvtrack_sidecar_root=/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
tracklet_memory_enabled=True
proposal_memory_enabled=True
tracklet_read_weight=0.0
```

Important interpretation:

```text
tracklet/proposal tensors are present in the window trace;
tracklet read weight is zero in this recipe, so tracklets are live diagnostics
and optional typed evidence, but they are not a dominant AQR reader term.
proposal/contact/mask evidence is the main sidecar scaffold used here.
```

## Dataflow Formula

The training source is `scripts/picf_core_train.py::_CalvinTransitionSource`.
It keeps an exhaustive `window_index` for diagnostics, but the actual training
length and sampler are segment-uniform:

```math
S = \{\text{selected CALVIN segments with sidecar coverage}\}
```

For each selected segment `i`:

```math
\tau_i^{min}=\text{segment.start}_i
```

```math
\tau_i^{max}=\text{segment.end}_i - (\text{unroll_steps}+\text{action_horizon}-1)
```

The training sample at global step `k`, rank `r`, micro-step `m` is:

```math
\xi_{k,r,m} = \operatorname{SeedSequence}(seed,r,k-1,m,retry,0xA7B52026)
```

```math
i \sim \operatorname{Uniform}\{1,\dots,|S|\}
```

```math
\tau \sim \operatorname{Uniform}[\tau_i^{min},\tau_i^{max})
```

Then the window is:

```math
W_{k,r,m} = (x_{\tau}, x_{\tau+1}, \dots, x_{\tau+\text{unroll}-1},
             a_{\tau:\tau+15})
```

The logged 50-step metric is an online train-stream mean:

```math
\bar L_{a,K}
= \frac{1}{50R}\sum_{k=K-49}^{K}\sum_{r=1}^{R}
L_a(\theta_k; W_{k,r,0})
```

Therefore a 50-step training row is not a fixed validation set and not a single
checkpoint replay. It is an online mean over changing parameters, stochastic
tasks, motion magnitudes, sidecar quality, and language instructions. A spike
in `loss_action_default_equiv` can be partly caused by a harder sampled-window
mixture even if the checkpoint is not worse.

## Actual Cloud Data State

Remote direct annotation audit:

```text
auto_lang_ann.npy exists
segment_count_total = 17870
selected_segments   = 7869
sampling_slots      = 7869
```

The selected segment file is:

```text
/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520/calvin_segment_indices.txt
```

The active run writes exact sampled windows to:

```text
/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/window_trace_rank0.jsonl
/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/window_trace_rank1.jsonl
```

This is better than reconstructing by code alone: the audit uses the trace
where possible and only uses RNG reconstruction as a cross-check.

## Band-Level Evidence

Exact trace rows through step7950 show 100 windows per 50-step band.

```text
band   La       nonact   activeov  proposal  mask    tracklet  bucket summary
7050   0.05019  0.01011  0.04400   1.23      42.22   60.76     block 62%, drawer 20%, slider 10%
7100   0.05140  0.01056  0.06998   1.28      43.16   60.21     block 62%, drawer 14%, slider 8%
7150   0.04244  0.01146  0.11000   1.30      42.33   60.13     block 52%, slider 20%, drawer 18%
7200   0.03794  0.01035  0.08000   1.17      38.66   60.16     block 55%, slider 17%, drawer 12%
7250   0.05086  0.01082  0.08497   1.23      40.29   58.61     block 52%, drawer 16%, button 8%
7300   0.04312  0.01116  0.08000   1.33      43.12   59.66     block 55%, drawer 18%, slider 14%
7350   0.03755  0.01044  0.05500   1.29      42.31   59.43     block 55%, drawer 18%, slider 16%
7400   0.04374  0.01043  0.11000   1.29      42.58   60.71     block 59%, slider 15%, drawer 12%
7450   0.04872  0.01103  0.10500   1.21      39.64   60.50     block 50%, drawer 20%, other 14%
7500   0.04086  0.01163  0.07218   1.32      44.69   59.95     block 57%, slider 17%, drawer 12%
7550   0.04450  0.01048  0.10500   1.18      39.75   59.98     block 54%, drawer 15%, slider 11%, button 7%
7600   0.05164  0.01215  0.11007   1.20      39.52   57.83     block 56%, slider 17%, drawer 12%
7650   0.04666  0.00995  0.06065   1.13      40.65   58.61     block 42%, slider 23%, drawer 18%
7700   0.04880  0.01189  0.11500   1.19      40.92   58.00     block 43%, drawer 21%, slider 16%
7750   0.04331  0.01083  0.08000   1.28      42.46   60.64     block 47%, slider 16%, drawer 16%
7800   0.04797  0.01136  0.09000   1.27      39.83   60.73     block 53%, drawer 21%, slider 19%
7850   0.04706  0.01043  0.13500   1.24      39.86   61.13     block 48%, drawer 25%, slider 12%
7900   0.04251  0.01135  0.11000   1.24      42.70   61.78     block 56%, slider 19%, drawer 16%
7950   0.04852  0.01094  0.11935   1.31      40.36   58.80     block 44%, drawer 19%, slider 19%
```

No sidecar coverage collapse is visible:

```text
proposal_count mean remains about 1.17-1.33;
proposal_mask_point_count mean remains about 38.66-44.69;
tracklet_count mean remains about 57.83-61.78;
active/downstream same-role overlap stays below the structural reject threshold.
```

## Correlation Check

Across logged bands 7050-7950, rough Pearson correlations against
`loss_action_default_equiv` are:

```text
nonaction budget                 +0.275
loss_anchor_object_pull          +0.284
loss_anchor_pv                   +0.177
loss_slot_jepa                   +0.165
active same-role overlap         -0.002
downstream same-role overlap     +0.046
proposal_count                   -0.208
mask_count                       -0.335
tracklet_count                   -0.320
block fraction                   -0.071
slider fraction                  -0.335
drawer fraction                  +0.235
button fraction                  +0.173
light fraction                   +0.275
switch fraction                  +0.239
```

Interpretation:

```text
active/downstream overlap is not explaining action loss in this window;
sidecar coverage is not collapsing;
slightly harder/non-block windows and lower mask/tracklet counts can raise La,
but the correlations are too weak to claim data composition as the only cause.
```

## Current Diagnosis

The current run is not reading obviously broken data.

What the data audit does explain:

```text
1. 50-step train rows are stochastic task-window estimates, not validation.
2. Some row-to-row oscillation is expected because task buckets and motion
   magnitude vary.
3. The 7600 high point coincides with higher non-action budget and harder data
   mixture, but this is partial evidence only.
```

What the data audit does not explain away:

```text
1. The action loss staying around 0.04-0.05 is still worse than the best
   historical 0.026-0.029 short window.
2. The plateau cannot be blamed on raw overlap, because active/downstream
   overlap is controlled.
3. The plateau cannot be blamed on missing sidecars, because proposal/mask and
   tracklet traces are present and stable.
```

Therefore the remaining candidate causes are:

```text
A. semantic/action side optimizer dynamics in a low-loss basin;
B. action prefix interface capacity/gating still bottlenecking PI0.5;
C. task-conditioned OOD windows not visible in 50-step global averages;
D. insufficient fixed-window validation visibility, not necessarily bad train
   sampling.
```

## Required Next Validation

Do not decide from sampled-window train rows alone. The next decisive test is a
fixed-window action probe:

```text
same fixed windows
same prompts/tasks
same sidecar fields
checkpoints: 7000, 7500, 7900/8000, next saved checkpoint
outputs: per-window action loss, bucket, prompt, sidecar counts
```

Mathematically, compare:

```math
\Delta_b(k_1,k_0)=
\mathbb E_{W\in \mathcal V_b}
[L_a(\theta_{k_1};W)-L_a(\theta_{k_0};W)]
```

for each bucket `b`. This separates true checkpoint degradation from sampled
window composition.

Stop/restart criteria:

```text
continue if fixed-window La improves or remains flat while sampled La oscillates;
restart/repair if fixed-window La worsens on most buckets, especially block and
slider, while action prefix and active overlap remain healthy.
```

## Concrete Follow-Up Commands

Remote metrics tail:

```bash
ssh -p 28060 root@36.139.225.68 \
  'tail -n 120 /mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/metrics.jsonl'
```

Remote progress tail:

```bash
ssh -p 28060 root@36.139.225.68 \
  'tail -f /mnt/picf_run_logs/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529.log'
```

Window trace files:

```bash
/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/window_trace_rank0.jsonl
/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/window_trace_rank1.jsonl
```
