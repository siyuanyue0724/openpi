# PICF-AQR-OWM Action Platform Root-Cause Protocol

Date: 2026-05-31

Status: active protocol.  Do not launch another action-platform experiment
without first mapping it to one of the hypotheses below and recording whether
the same hypothesis was already tested under materially different code.

## 1. Current Theory

The current failure mode is not "all anchors collapsed" and not "sidecar is
bad".  The strongest evidence says the live training scalar is a non-stationary
sampled-window metric, while the model/action interface still has a real
medium-horizon transfer problem.

The model should be judged with two metrics at the same time:

```text
rolling train-window metric:
  useful for online monitoring, but not stationary across windows or resumes.

exact-window / stratified-window metric:
  required for causal claims about model degradation, optimizer changes, or
  architecture/loss changes.
```

Mathematically, the rolling metric is

```text
L_roll(t) = E_{w ~ P_t(window)} [ L(theta_t; w) ]
```

where `P_t(window)` is the sampled window distribution seen by the online
trainer at that time.  A checkpoint comparison using `L_roll(t1)` and
`L_roll(t2)` is not a causal model comparison unless `P_t1 = P_t2`.

The no-update causal metric is

```text
L_fixed(theta_a, W) - L_fixed(theta_b, W),

L_fixed(theta, W) = 1 / |W| * sum_{w in W} L(theta; w).
```

This is the maintained way to decide whether a checkpoint really regressed.

## 2. Confirmed Facts

### 2.1 Old live `0.02` rows are not stationary baselines

Historical near-zero-LR replay reproduced the old 7550/7600 spike even when
learning was effectively disabled.  The old resumed runs replayed a sampled
window stream; they cannot be used as a global validation target.

### 2.2 Fixed-window probe changed the interpretation

On 64 identical accepted windows:

```text
current step7000 action_default_equiv = 0.061891
current step8000 action_default_equiv = 0.057919

4-22 PI0.5-only ablation:
  step7500  = 0.058717
  step10000 = 0.053813
  step20000 = 0.051038
```

This means current step8000 is roughly old step7500 fixed-window action
quality, not old step20000 quality.  It also means the old live `0.02` scalar
must not be treated as the production baseline.

### 2.3 Sidecar deletion is not the root cause

Same step8000 reset:

```text
full sidecar reset step8050 action = 0.0433176, anchor_pv = 0.5041
no sidecar reset step8050 action   = 0.0433337, anchor_pv = 0.5909
high PICF LR reset step8050 action = 0.0433395, anchor_pv = 0.5081
```

Deleting sidecar did not improve action and damaged object structure.

### 2.4 Dense context append was harmful, fixed-length fusion was safe

Directly appending context tokens shifted the PI0.5 action prefix/suffix
layout and worsened action.  Fixed-length prefix fusion preserved the action
token layout and repaired that regression.

Maintained rule:

```text
No direct action-prefix append.
Use fixed-length fusion or no context.
Keep PI0.5 action suffix position layout unchanged.
```

### 2.5 LR discontinuity is real but not the full explanation

Source step9100 LR was approximately:

```text
policy_head ~= 2.0e-5
semantic    ~= 7.0e-6
```

Several h30k resume diagnostics restarted this checkpoint near:

```text
policy_head ~= 5.9e-5
semantic    ~= 2.1e-5
```

Restoring LR continuity improved action, but did not by itself explain all
rolling-log differences.

### 2.6 Exact-window source/E13 comparison closed the latest false alarm

On the same 100 source E6 step9100 windows:

```text
source /9100 action = 0.041924
E13 /9200 action    = 0.042149
delta               = +0.000225 (+0.54%)
```

E13 did not materially degrade the model on identical windows.  The larger
rolling-log gap was primarily window/difficulty non-stationarity.

## 3. Excluded Or Weakened Root Causes

Do not rerun these as primary tests unless the code path has materially
changed and the change is documented.

```text
raw inactive same-role overlap:
  reserve/context rows can saturate raw overlap.  Use active/downstream overlap
  for the failure gate.

blind SAM proposals:
  visually noisy and archived.  Do not revive as a default path.

sidecar/proposal/tracklet deletion:
  not primary; removing it worsens structure.

whole-PICF LR too low:
  higher PICF LR reset matched action rows but did not move action.

direct dense context append:
  known harmful due to PI0.5 layout shift.

fixed64 as global metric:
  hard diagnostic only; not representative without an explicit stratified
  contract.

another raw overlap penalty:
  previous active/downstream gates showed action can fail while active overlap
  stays healthy.
```

## 4. Remaining Root-Cause Hypotheses

### H1. Window-distribution non-stationarity

Expected if true:

```text
rolling action rows move between 0.04 and 0.05, but exact-window comparison
shows little or no checkpoint degradation.
```

Fast test:

```text
Generate exact windows from the later rolling segment, then evaluate older and
newer checkpoints on the same windows with no update.
```

Decision:

```text
If same-window delta is small, do not change architecture or LR from rolling
rows alone.
```

### H2. Optimizer-state discontinuity

Expected if true:

```text
Model-only resumes behave differently from full optimizer resumes even with
the same LR schedule.
```

Fast test:

```text
Run a 200-300 step full-optimizer checkpoint smoke branch.  The purpose is
not final action quality; it is to verify optimizer.pt creation, resume
continuity, runtime, and no immediate structural regression.
```

Decision:

```text
All serious future phase checkpoints must use optimizer_checkpoint_mode=full.
Old model-only checkpoints remain valid for eval/export, not for claiming
continuous optimizer dynamics.
```

### H3. Action/PICF transfer bottleneck

Expected if true:

```text
object/recycle/slot structure improves on same windows, but action barely
moves.
```

Fast test:

```text
Use stratified held-window probes by task bucket and compare action changes
against object/recycle/active-slot metrics.  If structure improves without
action on the same windows, the bottleneck is action transfer, not ownership.
```

Decision:

```text
Only then consider action-side adapter/training changes.  Do not add another
slot/object loss first.
```

### H4. Sampling-family imbalance

Expected if true:

```text
some task buckets improve while others dominate the rolling loss plateau.
```

Fast test:

```text
Stratified 6-bucket probe:
  block+grasp
  block+push
  block+slider
  drawer
  slider
  manipulator(button/switch/light)
```

Decision:

```text
If one bucket dominates degradation, fix sampler or task-family weighting
before changing PICF architecture.
```

## 5. One-Hour Experiment Matrix

Every experiment must write:

```text
hypothesis;
history check;
exact command;
acceptance threshold;
early stop threshold;
where logs/metrics live.
```

### E14A. Future-window exact replay

Question:

```text
Was the latest rolling gap a harder-window artifact?
```

Procedure:

```text
1. Extract 100 exact windows from a later rolling interval.
2. Evaluate source /9100 and E13 /9200 on those exact windows.
3. Do not train.
```

Pass:

```text
same-window action delta <= 2% relative and structure metrics do not regress.
```

Fail:

```text
same-window action delta > 5% relative or structure/action degrade together.
```

### E14B. Stratified held-window probe

Question:

```text
Is the apparent platform global or concentrated in task families?
```

Procedure:

```text
Use 6 buckets, 32 windows per bucket, no update.
Compare at least:
  source /9100
  E13 /9200
  current maintained candidate if present
```

Pass:

```text
no task bucket regresses >5% action relative to source while structure is
unchanged.
```

Fail:

```text
one bucket drives most degradation; next experiment must target sampler/task
family, not global architecture.
```

### E14C. Full optimizer smoke continuation

Question:

```text
Can future continuation be made reproducible without model-only resume
artifacts?
```

Procedure:

```text
Run 200-300 steps with optimizer_checkpoint_mode=full.
Log every 50 steps.  Save at a short diagnostic interval only if disk permits;
otherwise save at step-scale boundary.
```

Pass:

```text
optimizer.pt exists at save point;
LR groups match intended values;
active/downstream overlap stays below 0.25;
loss_total_minus_action remains bounded near 0.01-0.015.
```

Fail:

```text
no optimizer.pt;
LR mismatch;
runtime > 35 sec/step without explanation;
nonfinite or structural collapse.
```

### E14D. Action-transfer split only if E14A/E14B fail

Question:

```text
Is action stuck because the PI0.5 action path cannot use the belief state?
```

Procedure:

```text
Same windows, compare:
  fixed-length fusion on/off;
  PICF frozen forward evidence vs cotrain;
  action/semantic-only update branch.
```

Pass:

```text
one split produces a same-window action gain without structure regression.
```

Fail:

```text
all splits same; root is likely data/task-family difficulty or action target
capacity, not PICF interface.
```

## 6. Runtime Monitoring Contract

For any training branch:

```text
log every 50 steps;
inspect every 50 steps until at least 100;
continue only if the branch matches its hypothesis;
stop immediately on invalid config or already-excluded failure;
do not wait for 1000 steps if the first 100 steps falsify the branch.
```

Required 50-step readout:

```text
loss_action_default_equiv
loss_total_minus_action
loss_anchor_pv
loss_anchor_object_pull
loss_slot_jepa
aqr_active_same_role_support_overlap_max
aqr_downstream_same_role_support_overlap_max
posterior_identity_switch_rate
posterior_recycle_rate
lr_group_policy_head
lr_group_semantic_backbone
lr_group_picf_core
grad_norm_group_policy_head
grad_norm_group_semantic_backbone
grad_norm_group_picf_core
```

Stop immediately if:

```text
nonfinite loss;
action worsens >30% from local minimum for two consecutive rows and exact-window
probe also confirms degradation;
active/downstream overlap >0.25 for two consecutive rows;
loss_total_minus_action >0.02 for two consecutive rows;
LR groups mismatch the written contract;
the branch repeats a previously excluded experiment without a code-change
justification.
```

## 7. Next Concrete Step

The next experiment should be E14A, not a new architecture change:

```text
1. Generate exact windows from the later rolling segment.
2. Evaluate source /9100 and E13 /9200 on exactly those windows.
3. If same-window stable, launch E14C full optimizer smoke.
4. If same-window unstable, run E14B stratified probe before changing model.
```

This ordering is intentional: it tests whether there is a real checkpoint
regression before spending another hour on training.

## 8. E14A Launch Record

Launched: 2026-05-31 12:22 Asia/Shanghai.

Hypothesis:

```text
The E13 step9200 rolling train loss is higher mostly because its sampled
windows are harder/different, not because E13 /9200 materially degraded the
source /9100 weights.
```

History check before launch:

```text
Already done:
  source /9100 and E13 /9200 were compared on the source E6 step9100 windows.
  The same-window action delta was only +0.000225.

Not yet done:
  compare the same two checkpoints on the later E13 step9200 windows.

Why this is not duplicate work:
  the previous exact replay asked whether E13 damaged old source windows.
  E14A asks whether the later E13 rolling window itself is simply harder.
```

Remote:

```text
machine:
  A7 qgE72e

tmux:
  e14a_future_exact_20260531

output:
  /mnt/picf_exact_window_probes/e14a_future_9200_20260531/

window jsonl:
  /mnt/picf_exact_window_probes/e14a_future_9200_20260531/
    e13_step9200_rank01_windows.jsonl

records:
  100 exact windows from E13 step9200 rank0/rank1 window traces
```

Checkpoints:

```text
source:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_stepindexed_from9000_prefixfusion_9300_20260530/9100

E13:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531/9200
```

Tail:

```bash
ssh -p 28060 root@36.139.225.68
tmux attach -t e14a_future_exact_20260531
tail -f /mnt/picf_exact_window_probes/e14a_future_9200_20260531/source9100_probe.log
tail -f /mnt/picf_exact_window_probes/e14a_future_9200_20260531/e13_9200_probe.log
tail -f /mnt/picf_exact_window_probes/e14a_future_9200_20260531/e14a_combined.log
```

Decision rule:

```text
If E13 /9200 is within 2% action of source /9100 on these E13 step9200
windows, the rolling train gap is a window-mix artifact and no architecture
change is justified.

If E13 /9200 is worse by >5% on the same windows, then the gap is a true
checkpoint regression.  The next test becomes optimizer-state/fullopt
continuation, not another slot loss.
```

Update 2026-05-31 13:28:

```text
The initial 100-window E14A probe entered the model but was too slow for the
one-hour diagnostic budget.  It was stopped after confirming the path works.

The active fast gate is now:
  tmux = e14a_future_exact_fast32_20260531
  windows = first 32 exact records from the same E13 step9200 trace

Reason:
  32 exact windows are sufficient for the first stop/continue gate.  If the
  fast32 delta is ambiguous or near the threshold, expand to 100 windows.
```

Update 2026-05-31 14:16:

```text
The fast32 run completed, but it is invalid for action-platform conclusions.

Observed:
  source rows = 32
  E13 rows    = 32
  GPU state   = idle after completion

Problem:
  per-window JSONL did not contain loss_action_default_equiv, loss_action,
  loss_action_active7, or loss_total.  It only contained structure/debug losses
  such as loss_anchor_object_pull.

Conclusion:
  This run can only be used as a structure-field sanity check.  It must not be
  used to decide whether action regressed or whether the rolling action gap is
  window difficulty.

Tool fix:
  scripts/picf_fixed_window_action_probe.py now marks summaries as
  missing_action_metrics and raises unless --allow-missing-action-metrics is
  explicitly passed.  This prevents future false-positive action conclusions.

Next required step:
  Rerun E14A-fast32 only after confirming the probe path returns
  loss_action_default_equiv.  If necessary, run the probe in train forward mode
  under no-update/no-backward semantics, because the fixed-window question is
  about checkpoint loss on identical windows, not optimizer updates.
```

Hard rule added after this failure:

```text
No action-platform conclusion is allowed unless the exact-window summary has:
  loss_action_default_equiv
  loss_action_active7
  loss_total
  loss_total_minus_action

If these keys are missing, the experiment is invalid and must be repaired
before any model/loss change is made.
```

Update 2026-05-31 13:55:

```text
Launched a minimal action-readout calibration before any further training:
  tmux = e14a_actionreadout_trainmode4_20260531
  windows = first 4 exact E13 step9200 windows
  mode = train
  update = none; torch.no_grad probe path still applies

Purpose:
  determine whether action metrics disappear only in eval-mode probe, or
  whether the checkpoint/action-generator path itself emits non-finite action
  losses on these exact windows.

This is a probe-chain repair, not a model experiment.  It must complete before
E14A can be used for action-platform conclusions.
```

Tail:

```bash
ssh -p 28060 root@36.139.225.68
tmux attach -t e14a_actionreadout_trainmode4_20260531
tail -f /mnt/picf_exact_window_probes/e14a_actionreadout_trainmode4_20260531/source9100_trainmode4.log
tail -f /mnt/picf_exact_window_probes/e14a_actionreadout_trainmode4_20260531/e13_9200_trainmode4.log
tail -f /mnt/picf_exact_window_probes/e14a_actionreadout_trainmode4_20260531/combined.log
```

Update 2026-05-31 14:55:

```text
The train-mode action-readout calibration also failed as an action probe.

Observed from the raw first-window debug:
  loaded_step = 9200
  output_key_count = 355
  loss_total                 = NaN
  loss_action                = NaN
  loss_action_default_equiv  = NaN
  loss_action_active7        = NaN
  loss_action_pos/rot/grip   = NaN
  loss_total_minus_action    = NaN
  loss_alignment             = NaN

Finite in the same forward:
  loss_action_prefix_trust = 0.02
  loss_anchor_object_pull  ~= 1.804
  pi_prefix_nonfinite_count = 0

Conclusion:
  This is not an eval-mode dropout issue and not a missing-key issue.  The
  exact-window action path is producing non-finite action losses while the
  structure path remains finite.  Therefore E14A is still blocked.

Immediate root-cause search:
  decompose one exact window into action target, PICF prefix/context tensors,
  Paligemma suffix output, action projection output, and action loss.  Also
  check checkpoint compatibility because the current code reports newly added
  action_context_* adapter parameters as missing when loading older checkpoints.

Probe-tool hardening:
  scripts/picf_fixed_window_action_probe.py now records scalar non-finite
  outputs with *_finite=0 and *_nonfinite=1 instead of silently dropping them.
```

Update 2026-05-31 15:22:

```text
Root cause found and repaired.

Decomposition result on E13 step9200 first exact window:
  action target finite
  noise/u_t finite
  PICF extra prefix finite
  action-context adapter disabled or removed: still NaN
  no PICF extra prefix: still NaN
  all loaded model parameters finite
  first non-finite tensor: paligemma_with_expert suffix_out_all

The deciding intervention:
  original additive attention mask negative value = -2.3819763e38
  bf16 SDPA/Gemma expert suffix forward       = all-NaN
  bf16-safe additive negative value -1e4      = finite suffix_out,
                                                finite v_t,
                                                finite action loss

Exact same checkpoint/window after -1e4:
  loss_total = 0.06056
  action projection finite
  predicted chunk finite

Therefore:
  E14A was blocked by a numerical mask bug in the action forward path, not by
  sidecar, raw overlap, PICF extra prefix, action-context adapter, target data,
  or non-finite checkpoint parameters.

Code fix:
  src/openpi/picf/paligemma/wrapper.py::_prepare_attention_masks_4d
  src/openpi/models_pytorch/pi0_pytorch.py::_prepare_attention_masks_4d
  both now use -1e4, which preserves softmax masking while avoiding bf16/fp16
  SDPA NaNs.

Validation:
  local py_compile passed;
  uv run pytest passed:
    src/openpi/picf/paligemma/wrapper_test.py
    src/openpi/picf/policy_test.py
    scripts/picf_core_train_test.py

Remote mask-fix 4-window probe:
  status = ok
  loss_action_default_equiv = 0.04175
  no missing/nonfinite action metrics

Next:
  rerun E14A fast32 source9100 vs E13 step9200 on identical windows with the
  mask fix before making any training/loss decision.
```

Update 2026-05-31 15:31:

```text
E14A fast32 rerun completed after the bf16-safe attention-mask fix.

Same exact 32 E13 step9200 future windows:

source /9100:
  loss_action_default_equiv = 0.0447027
  loss_action_active7       = 0.203660
  loss_total                = 0.241421
  loss_total_minus_action   = 0.196718
  loss_anchor_object_pull   = 0.101772
  active same-role overlap  = 0.109375
  downstream overlap        = 0.100329

E13 /9200:
  loss_action_default_equiv = 0.0428278
  loss_action_active7       = 0.195095
  loss_total                = 0.321821
  loss_total_minus_action   = 0.278993
  loss_anchor_object_pull   = 0.332967
  active same-role overlap  = 0.140625
  downstream overlap        = 0.125000

Action delta:
  E13 - source = -0.001875
  relative     = -4.19%

Conclusion:
  E13 /9200 does not show same-window action degradation on the later E13
  future windows.  It is action-better on this fixed set.  Therefore the
  previous rolling-log action gap is not valid evidence of checkpoint action
  regression.

  However, E13 has worse structure/aux burden on the same windows:
    loss_total_minus_action +0.0823
    loss_anchor_object_pull +0.2312
  active/downstream same-role overlap is still below the hard 0.25 gate.

Decision:
  Do not change model architecture or add more slot losses based on the rolling
  action gap.  The next valid step is E14C full-optimizer smoke continuation:
  prove that future continuations save/load optimizer state, preserve LR groups,
  and keep active/downstream structure gates stable.
```

## 9. E14C Launch Contract

Purpose:

```text
This is not another architecture experiment.  E14A says the apparent action
gap was not same-window action degradation.  The next remaining operational
risk is continuation integrity: model-only resumes and fresh Adam moments can
make later training rows difficult to interpret.
```

Run contract:

```text
checkpoint:
  source /9100 from picf_a7_stepindexed_from9000_prefixfusion_9300_20260530

code:
  mask-fix branch where PI0/PaliGemma additive attention masks use -1e4.

optimizer:
  OPTIMIZER_CHECKPOINT_MODE=full

LR:
  LR=2e-5
  MIN_LR=2e-5
  SEMANTIC_LR_SCALE=0.35

save:
  SAVE_INTERVAL=100
  KEEP_LAST_CHECKPOINTS=3

log:
  LOG_INTERVAL=50
  ANCHOR_OVERLAY_INTERVAL=100
```

Pass gates at first 100-300 resumed steps:

```text
optimizer.pt exists in saved checkpoint;
lr_group_policy_head ~= 2e-5;
lr_group_semantic_backbone ~= 7e-6;
nonfinite flags absent;
active/downstream same-role overlap < 0.25;
loss_total_minus_action is bounded relative to E14A fixed-window structure load;
runtime is explained and not a silent full-backbone/unintended-config run.
```

Launch record:

```text
Launched: 2026-05-31 15:36 Asia/Shanghai

remote:
  A7 qgE72e

tmux:
  e14c_fullopt_smoke_20260531

log:
  /mnt/picf_run_logs/
    picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531.log

checkpoint dir:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531

observed startup:
  checkpoint loaded from source /9100 with compatibility migration;
  optimizer state is reinitialized because source /9100 is model-only;
  future checkpoints will save full optimizer state.

observed config:
  optimizer_checkpoint_mode = full
  training_strategy         = fsdp_full_shard
  optimizer_sharding        = none
  semantic_trainable        = true, scope=backbone_only
  perception backbones      = frozen
  unroll/burnin             = 2 / 1 state_only
  policy_head_lr            = 2e-5
  semantic_backbone_lr      = 7e-6
  picf_core_lr              = 2e-8
  action_context            = prefix_fusion, 24 tokens
  save_interval             = 100
  keep_last_checkpoints     = 3

early runtime:
  step 9101-9105 entered normally at roughly 26-27 sec/step.

local validation before interpreting E14C:
  py_compile passed for:
    scripts/picf_action_nan_decompose_probe.py
    scripts/picf_fixed_window_action_probe.py
    scripts/picf_core_train.py
    src/openpi/picf/paligemma/wrapper.py
    src/openpi/models_pytorch/pi0_pytorch.py
  uv pytest passed:
    src/openpi/picf/paligemma/wrapper_test.py
    src/openpi/picf/policy_test.py
    scripts/picf_core_train_test.py
  total local tests: 192 passed

local follow-up continuity fix:
  While waiting for the remote 9200 save gate, the local checkpoint loader was
  audited.  A small consistency bug was fixed: loading through `latest.pt`,
  which redirects to a numeric checkpoint directory, now forwards the
  grad-clip controller into the recursive `_load_checkpoint` call.  This does
  not affect the running E14C process because it started from an explicit
  numeric step directory, but it prevents future `latest.pt` resumes from
  silently starting with fresh clipping history.

  Validation after this local fix:
    python -m py_compile scripts/picf_core_train.py
    uv run pytest -q scripts/picf_core_train_test.py
    result: 136 passed

local follow-up probe hardening:
  The fixed-window action probe was also tightened.  Previously it failed when
  required action metrics were entirely missing, but a mixed case where some
  windows produced finite values and some produced non-finite values could
  still leave an aggregate mean in the summary.  That is invalid for causal
  action conclusions.

  The probe now always records:
    missing_required_metrics
    nonfinite_required_metrics
    nonfinite_required_metric_counts
  and raises if any required action metric is missing or non-finite, unless the
  user explicitly opts into structure-only diagnostics with
  `--allow-missing-action-metrics`.

  Validation:
    python -m py_compile scripts/picf_fixed_window_action_probe.py
    uv run pytest -q scripts/picf_core_train_test.py
    result: 136 passed

first 50-step diagnostic:
  step = 9150
  loss_total = 0.0568085
  loss_action_default_equiv = 0.0464797
  loss_action_active7 = 0.211666
  loss_total_minus_action = 0.0103288
  loss_anchor_object_pull = 0.162422
  loss_slot_jepa = 0.835056
  active same-role support overlap = 0.165000
  downstream same-role support overlap = 0.144641
  posterior_recycle_rate = 0.102784
  posterior_identity_switch_rate = 0.218333
  grad_norm = 0.422093
  grad_clip_applied = false
  steps_per_sec = 0.0364, roughly 27.5 sec/step

interpretation:
  The first training diagnostic satisfies the structural hard gate:
    active/downstream overlap < 0.25
    loss_total_minus_action is near the expected 0.01 band
    LR groups match the contract
    no non-finite/gradient issue observed
  E14C is not complete until step 9200 writes a full optimizer checkpoint.

first save-gate diagnostic:
  step = 9200
  loss_total = 0.0566340
  loss_action_default_equiv = 0.0465813
  loss_action_active7 = 0.212002
  loss_total_minus_action = 0.0100527
  loss_anchor_object_pull = 0.132633
  loss_slot_jepa = 0.748319
  active same-role support overlap = 0.094996
  downstream same-role support overlap = 0.097500
  posterior_recycle_rate = 0.100917
  posterior_identity_switch_rate = 0.207778
  grad_norm = 0.608172
  grad_clip_applied = false
  steps_per_sec = 0.0371, roughly 26.9 sec/step

  checkpoint files:
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9200/model.pt
      size = 9,912,531,484 bytes
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9200/optimizer.pt
      size = 17,495,377,509 bytes
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9200/metadata.pt
      size = 67,187 bytes

interpretation:
  The 9200 save gate passes.  Full optimizer checkpoint writing is operational
  on the mask-fix branch.  This closes the earlier ambiguity where model-only
  resumes and fresh optimizer moments made later training rows hard to compare.

  E14C should still be observed through the next 50-step diagnostic and the
  9300 save boundary before releasing the branch, because the written optimizer
  state must be part of a stable 200-step smoke, not a single-save artifact.

second 50-step diagnostic:
  step = 9250
  loss_total = 0.0498379
  loss_action_default_equiv = 0.0402419
  loss_action_active7 = 0.183069
  loss_total_minus_action = 0.0095960
  loss_anchor_object_pull = 0.0969302
  loss_slot_jepa = 0.771886
  active same-role support overlap = 0.090000
  downstream same-role support overlap = 0.091658
  posterior_recycle_rate = 0.102332
  posterior_identity_switch_rate = 0.211111
  grad_norm = 0.321622
  grad_clip_applied = false
  steps_per_sec = 0.0321, slower row because it includes the previous
  checkpoint-save disturbance; pbar has otherwise resumed normal stepping.

interpretation:
  The 9250 diagnostic strengthens the pass case:
    action improved from 0.04658 to 0.04024;
    non-action budget stayed in the expected ~0.01 band;
    object-pull improved;
    active/downstream overlap stayed well below 0.25;
    no gradient instability appeared.

  Continue to 9300.  If 9300 writes another full optimizer checkpoint and
  preserves the same gates, E14C is complete.
```

Update 2026-05-31 17:23:

```text
E14C reached the required 9300 save boundary and passed the smoke gate.

third 50-step / second save-gate diagnostic:
  step = 9300
  loss_total = 0.0516444
  loss_action_default_equiv = 0.0414902
  loss_action_active7 = 0.188535
  loss_total_minus_action = 0.0101542
  loss_anchor_pv = 0.497281
  loss_anchor_object_pull = 0.147518
  loss_slot_jepa = 0.757773
  active same-role support overlap = 0.1299999
  downstream same-role support overlap = 0.1167628
  posterior_recycle_rate = 0.101799
  posterior_identity_switch_rate = 0.213333
  lr_group_policy_head = 2.0e-5
  lr_group_semantic_backbone = 7.0e-6
  lr_group_picf_core = 2.0e-8
  grad_norm = 0.194550
  grad_clip_applied = false
  nonfinite_metrics_count = 0

checkpoint files:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9300/model.pt
    size = 9,912,531,484 bytes
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9300/optimizer.pt
    size = 17,495,377,509 bytes
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9300/metadata.pt
    size = 67,187 bytes

latest.pt:
  step = 9300
  checkpoint_dir =
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9300

interpretation:
  E14C passes.  The mask-fix branch can run, write full optimizer state, keep
  the intended LR groups, and preserve the active/downstream structural gates
  for a 200-step smoke continuation.

  This does not prove final action quality.  It closes a narrower operational
  question: future continuation checkpoints must use
  optimizer_checkpoint_mode=full if they will be compared as continuous
  optimizer trajectories.

decision:
  The E14C training process was stopped after the 9300 checkpoint completed.
  Do not keep spending GPU on this smoke branch.  The next valid experiment is
  E14B stratified held-window probing, not another architecture/loss change.
```

## 10. E14B Launch Contract

Purpose:

```text
E14A showed that the rolling action gap was not same-window action
degradation.  E14C showed that future continuation can preserve full optimizer
state.  The remaining causal question is whether any task family is still
driving the apparent action platform.
```

Procedure:

```text
Generate explicit segment/start_step records with stratified buckets:
  block+grasp
  block+push
  block+slider
  drawer
  slider
  manipulator

First gate:
  per_bucket = 8
  total windows = 48

Compare on the exact same records:
  source /9100
  E13 /9200
  E14C /9300

Required metrics:
  loss_action_default_equiv
  loss_action_active7
  loss_total
  loss_total_minus_action
  loss_anchor_object_pull
  active/downstream same-role overlap
```

Pass:

```text
No bucket regresses by >5% action relative to source while structure is stable.
If global mean is stable but one bucket is bad, the next fix is sampler/task
family handling, not another global slot loss.
```

Fail / invalid:

```text
Any missing or non-finite required action metric invalidates the probe.
The hardened fixed-window probe now raises instead of producing an ambiguous
summary in that case.
```

Launch record:

```text
Launched: 2026-05-31 17:29 Asia/Shanghai

remote:
  A7 qgE72e

tmux:
  e14b_stratified_fast48_20260531

output:
  /mnt/picf_exact_window_probes/e14b_stratified_fast48_20260531

args:
  /mnt/checkpoints/picf_core/picf_core/
    picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/args.json

checkpoints:
  source:
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_stepindexed_from9000_prefixfusion_9300_20260530/9100
  E13:
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531/9200
  E14C:
    /mnt/checkpoints/picf_core/picf_core/
      picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9300
```

First launch correction:

```text
The initial E14B window generation used --device cpu and failed before model
evaluation:

  point_backbone=sonata currently requires CUDA.

This is a parameter-validation issue, not a model/data failure.  The generator
does not train or change windows, but it loads the same train-args contract as
the probe.  Therefore the corrected launch uses:

  --device cuda:0

Window generation then succeeded:

  records = 48
  dataset_size = 7869
  split = training

bucket count check:
  block+grasp: 8
  block+push: 8
  block+slider: 8
  drawer: 8
  slider: 8
  manipulator: 8 requested, internally composed of switch/button/light/turn
               prompt variants

source /9100 and E13 /9200 probes then entered the model concurrently on the
two A7 GPUs.  Early per-window rows include finite
loss_action_default_equiv, so the previous action-readout NaN/missing-key bug
is not recurring.
```

Tail:

```bash
ssh -p 28060 root@36.139.225.68
tail -f /mnt/picf_exact_window_probes/e14b_stratified_fast48_20260531/e14b_driver.log
tail -f /mnt/picf_exact_window_probes/e14b_stratified_fast48_20260531/source9100_probe.log
tail -f /mnt/picf_exact_window_probes/e14b_stratified_fast48_20260531/e13_9200_probe.log
tail -f /mnt/picf_exact_window_probes/e14b_stratified_fast48_20260531/e14c_9300_probe.log
```

Completion update: 2026-05-31 17:58

```text
E14B fast48 completed.

status:
  source /9100 summary = ok, accepted_windows = 48, elapsed_s = 798.771
  E13 /9200 summary    = ok, accepted_windows = 48, elapsed_s = 798.382
  E14C /9300 summary   = ok, accepted_windows = 48, elapsed_s = 447.664
  missing_required_metrics = none
  nonfinite_required_metrics = none

global action:
  source /9100 loss_action_default_equiv = 0.038311
  E13 /9200    loss_action_default_equiv = 0.038719
  E14C /9300   loss_action_default_equiv = 0.038503

global action delta vs source:
  E13 /9200  = +0.000408, +1.07%
  E14C /9300 = +0.000192, +0.50%

global structure:
  source /9100 loss_total_minus_action = 0.251353
  E13 /9200    loss_total_minus_action = 0.202297
  E14C /9300   loss_total_minus_action = 0.206825

  source /9100 loss_anchor_object_pull = 0.239483
  E13 /9200    loss_anchor_object_pull = 0.099150
  E14C /9300   loss_anchor_object_pull = 0.120058

active/downstream overlap:
  source /9100 active/downstream = 0.093750 / 0.095117
  E13 /9200    active/downstream = 0.104167 / 0.104167
  E14C /9300   active/downstream = 0.125000 / 0.120082
```

Requested-bucket action deltas:

```text
bucket          source     E13       E13 rel   E14C      E14C rel
block+grasp    0.032320   0.032548  +0.70%    0.032886  +1.75%
block+push     0.038548   0.039046  +1.29%    0.038678  +0.34%
block+slider   0.027896   0.028220  +1.16%    0.028497  +2.15%
drawer         0.033573   0.034047  +1.41%    0.033786  +0.64%
manipulator    0.051485   0.052068  +1.13%    0.052101  +1.20%
slider         0.046043   0.046386  +0.74%    0.045071  -2.11%
```

Requested-bucket structure notes:

```text
manipulator improves strongly in structure:
  source non-action/object-pull = 0.547649 / 1.088756
  E13    non-action/object-pull = 0.221800 / 0.157788
  E14C   non-action/object-pull = 0.204491 / 0.160123

block+push is the only requested bucket where E14C structure worsens
materially vs source:
  source non-action/object-pull = 0.221293 / 0.150751
  E14C   non-action/object-pull = 0.276666 / 0.308935

However, block+push action is essentially unchanged:
  source = 0.038548
  E14C   = 0.038678 (+0.34%)
```

Tool correction from E14B:

```text
The probe records originally copied the actual bucket but not bucket_request
from the explicit window JSONL.  This made the manipulator request bucket split
into switch/button/light/turn prompt variants in the first comparison table.

The current result above was recomputed by joining records with the original
window JSONL.  The tool has been fixed so future explicit-window probes preserve
bucket_request in per-window records.
```

Interpretation:

```text
E14B does not support a true action regression or a single task-family action
failure.  On the same stratified windows, E14C /9300 is within +0.5% action of
source /9100 globally, and every requested bucket is within the +/-5% action
gate.

The remaining problem is not raw overlap, blind SAM/sidecar noise, checkpoint
NaN, or a specific task bucket.  The strongest remaining interpretation is:

  1. live rolling training rows are non-stationary and must not be compared as
     checkpoint-quality scalars;
  2. structure-side auxiliary burden can move independently from action;
  3. action-quality claims require fixed/stratified probes plus CALVIN/video,
     not rolling loss alone.

Next decision:
  Do not add a new slot/object loss based on rolling loss.  If another
  experiment is needed, it should be an action-transfer split or CALVIN eval,
  not another raw-overlap or sidecar ablation.
```

## 11. E14D Action-Transfer Split Launch Contract

Date: 2026-05-31 18:30 CST

Reason:

```text
E14B did not show true same-window action regression.  It did show that
structure-side quantities can improve while action stays nearly unchanged.
The next remaining causal question is therefore whether the PI0.5 action path
is actually using the PICF belief/context, not whether another anchor/object
loss is needed.
```

Mathematical hypothesis:

```text
Let B_t be the PICF belief state and z_t = g(B_t, X_t) be the action-visible
PICF interface.

If d L_action / d B_t is small because d z_t / d B_t is weak, then object
structure can improve without action improvement:

  d L_action / d B_t
    = (d L_action / d z_t) (d z_t / d B_t).

This is an action-transfer bottleneck, not an ownership/overlap failure.
```

History check before launching:

```text
Already excluded as primary root causes:
  raw inactive same-role overlap;
  blind SAM proposals;
  sidecar deletion;
  direct dense context append;
  whole-PICF LR alone;
  fixed64 as a global validation metric.

Already completed:
  E14A: bf16 mask/action-readout bug repaired;
  E14B: stratified fast48 same-window probe;
  E14C: full optimizer smoke reached 9300 and wrote optimizer.pt.
```

Procedure:

```text
Use the same E14B stratified fast48 windows:
  /mnt/picf_exact_window_probes/e14b_stratified_fast48_20260531/
    e14b_stratified_perbucket8_windows.jsonl

Evaluate without backward/update:
  source /9100 prefix-fusion baseline;
  E14C /9300 full-optimizer smoke prefix-fusion;
  suffix-adapter action-head /9200;
  suffix-adapter backbone sem035 /9200;
  prefix-fusion sem035 /9200 as LR/interface control.
```

Acceptance:

```text
If a suffix/gated-action adapter checkpoint improves fixed-window action by
>= 2% relative without worsening total_minus_action or active/downstream
overlap, the bottleneck is the previous passive prefix interface.

If suffix/gated-action adapter is action-neutral while structure is comparable,
the current adapter does not solve transfer; next work should target action
readout/training or CALVIN closed-loop, not more slot losses.

If one task bucket regresses > 5%, inspect bucket-specific records before any
global architecture change.
```

Early stop:

```text
missing or non-finite required action metrics;
wrong args/checkpoint pairing;
same-window probe accidentally runs with updates;
GPU memory conflict causing partial records.
```

## 12. Current Root-Cause Tree While E14D Runs

Date: 2026-05-31 18:45 CST

Current evidence says the action plateau/rebound should not be treated as a
single scalar-loss tuning problem.  The useful factorization is:

```text
observations Z
  -> PICF belief B_t
  -> action-visible interface z_t
  -> PI0.5 action head / diffusion suffix
  -> action loss.
```

The only way a slot/belief improvement can reduce action loss is through:

```text
d L_action / d B_t
  = (d L_action / d z_t) (d z_t / d B_t).
```

Therefore a run can show:

```text
structure losses improve;
active/downstream overlap healthy;
sidecar/object pull healthy;
action nearly flat.
```

without proving that the object router is useless.  It proves only that the
current action-visible readout may not be sufficiently used by the PI0.5 action
path under the current optimization schedule.

Experiments not to repeat unless the code path changes materially:

```text
1. raw inactive same-role overlap penalty only:
   raw reserve overlap is structurally expected for unused reserve/context
   anchors and did not explain same-window action quality.

2. blind SAM proposal supervision:
   produced noisy boxes/masks and was archived as default-off.

3. deleting sidecar:
   no action improvement on same continuation point.

4. direct dense context append:
   known to hurt action stability and speed.

5. fixed64 as global validation:
   fixed64 is a probe for one window set, not a global CALVIN-quality scalar.

6. whole-PICF LR only:
   may change structure learning speed, but did not isolate the action-transfer
   bottleneck.
```

E14D is the current decisive probe:

```text
Compare prefix-fusion and suffix/gated action adapters on exactly the same
E14B fast48 windows, with no update.

If suffix action adapter improves fixed-window action by >= 2% and structure
metrics remain comparable:
  root cause = passive prefix/action-transfer bottleneck.
  next = train that action-transfer path, not add more slot losses.

If suffix action adapter is neutral:
  root cause is deeper action readout/training or closed-loop rollout mismatch.
  next = action-readout training split or CALVIN eval, not slot-object tuning.
```

Interim E14D status:

```text
At 24/48 accepted windows, suffix_actionhead_9200 had lower fixed-window action
than prefix_sem035_9200 by about 3.9% while active/downstream overlap stayed
comparable.  This is not final until all requested windows complete.
```

Final E14D result:

```text
fast48 fixed-window action:

source9100                  0.0383109
E14C9300 prefix-fusion       0.0385030   (+0.50% vs source)
prefix_sem035_9200          0.0410586   (+7.17% vs source)
suffix_actionhead_9200      0.0397192   (+3.68% vs source)
suffix_backbone_sem035_9200 0.0412881   (+7.77% vs source)

active same-role support overlap:

source9100                  0.09375
E14C9300 prefix-fusion       0.12500
prefix_sem035_9200          0.11458
suffix_actionhead_9200      0.09278
suffix_backbone_sem035_9200 0.12500

downstream same-role support overlap:

source9100                  0.09512
E14C9300 prefix-fusion       0.12008
prefix_sem035_9200          0.11785
suffix_actionhead_9200      0.09278
suffix_backbone_sem035_9200 0.11799

total_minus_action:

source9100                  0.25135
E14C9300 prefix-fusion       0.20683
prefix_sem035_9200          0.22874
suffix_actionhead_9200      0.25010
suffix_backbone_sem035_9200 0.21773
```

Interpretation:

```text
1. suffix_actionhead_9200 improves action relative to prefix_sem035_9200 by
   about 3.26%, and active/downstream overlap is better, so a pure passive
   prefix interface is not the only viable action-visible path.

2. suffix_actionhead_9200 does not beat source9100 or E14C9300, and its
   total_minus_action is worse than prefix_sem035/E14C.  Therefore the result
   is partial: action-side adapter can help transfer, but this checkpoint does
   not yet dominate the maintained prefix-fusion baseline.

3. suffix_backbone_sem035_9200 improves structure-side burden relative to
   prefix_sem035, but action is worse.  This repeats the central split:
   structure improvement and action improvement are not automatically aligned.

4. The next experiment must not be another raw-overlap/sidecar/object-pull
   repair.  The remaining live axis is an action-transfer training split:
   train the action-visible readout/interface with enough action pressure while
   keeping PICF belief stable enough that z_t remains stationary.
```

Decision gate:

```text
Do not claim final convergence from E14D.
Do not discard suffix action-head transfer either.
Next one-hour experiment should compare:
  A. maintained E14C/source-style prefix-fusion continuation;
  B. suffix_actionhead continuation with action-head/interface trainable and
     PICF low-LR/frozen;
on fixed/stratified windows plus live rolling loss.

If B beats A on same-window action without structure regression, keep
suffix-action transfer.  If not, keep prefix-fusion and move to CALVIN closed
loop/readout training rather than adding slot losses.
```

## 13. E14E Maintained-Prefix Continuation

Date: 2026-05-31 19:00 CST

Reason:

```text
E14D did not justify replacing the maintained prefix-fusion path with suffix
cross-attention.  The strongest checkpoint on the fixed fast48 probe remained
source9100/E14C9300 prefix-fusion, not suffix_backbone or suffix_actionhead.

The next causal question is therefore:
  can the maintained prefix-fusion/full-optimizer path continue past 9300
  without the historical action plateau/rebound?
```

Launch contract:

```text
resume:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9300

optimizer:
  full optimizer resume, not model-only reset

target:
  10300 steps first gate

unchanged maintained boundary:
  action_context_integration = prefix_fusion
  action_context_tokens      = 24
  semantic_trainable_scope   = backbone_only
  semantic_lr_scale          = 0.35
  lr = min_lr                = 2e-5
  picf_core_lr_scale         = 0.001
  action_prefix_teacher      = ema

log gates:
  every 50 steps;
  inspect 9350/9400/9450/9500 and stop early if action monotonically worsens
  while fixed/structure metrics do not explain it.
```

Interpretation:

```text
If E14E continues to reduce action with stable active/downstream overlap:
  root cause was not architecture failure; keep maintained prefix-fusion and
  train longer, with fixed-window probes as checkpoint-quality metrics.

If E14E plateaus/rebounds while structure remains healthy:
  root cause is not slot ownership.  Move to action readout/CALVIN closed-loop
  evaluation and possibly action-side training schedule, not another object
  loss.

If structure degrades together with action:
  inspect optimizer resume, sidecar/mask availability, and PICF LR before any
  architecture change.
```

Launch correction:

```text
The first launcher was accidentally written into the experiment output
directory.  The trainer clears/overwrites that directory at startup, so the
launcher and tee log were removed before any useful failure message could be
kept.  This was a launcher-placement bug, not a model/training failure.

Corrected launcher:
  /root/run_scripts/e14e_prefix_fullopt_from9300_10300_20260531.sh

Corrected log:
  /tmp/e14e_prefix_fullopt_from9300_10300_20260531.train.log
```

Startup verification:

```text
E14E resumed from the intended full-optimizer checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531/9300

trainer log:
  Resumed ... at step=9300
  optimizer_checkpoint_mode=full
  training_strategy=fsdp_full_shard
  world_size=2
  unroll_steps=2
  burnin_steps=1
  action_context_integration=prefix_fusion
  semantic_trainable=True scope=backbone_only
  visual/point/tactile pretrained backbones frozen

Measured startup:
  full optimizer/checkpoint load took about 8 minutes before the first train
  step; actual train speed then returned to about 25-26 seconds/step, which is
  the expected historical band for this configuration.
```

9350 gate:

```text
loss_action_default_equiv       0.040801
loss_total                      0.051823
loss_total_minus_action         0.011022
loss_anchor_pv                  0.494813
loss_anchor_object_pull         0.231192
loss_aqr_denoising              1.181169
loss_mapg_routing               0.425300
loss_slot_jepa                  0.763325
posterior_identity_switch_rate  0.202222
posterior_recycle_rate          0.108780
active/downstream overlap       0.110000 / 0.110675
raw same-role overlap           1.000000
semantic/PICF LR                7e-6 / 2e-8
```

Read:

```text
Do not stop at 9350.  Action did not regress relative to E14C9300
(0.041490 -> 0.040801), and active/downstream overlap remains healthy.
The high raw overlap is the known reserve/context artifact and is not a stop
criterion.  The only warning is object_pull rising versus 9300, so the next
gate must check whether it normalizes or co-moves with action degradation.
```

9400 gate:

```text
loss_action_default_equiv       0.043734
loss_total                      0.053931
loss_total_minus_action         0.010196
loss_anchor_pv                  0.498631
loss_anchor_object_pull         0.151621
loss_aqr_denoising              1.181313
loss_mapg_routing               0.429176
loss_slot_jepa                  0.764355
posterior_identity_switch_rate  0.217222
posterior_recycle_rate          0.112387
active/downstream overlap       0.130000 / 0.131565
raw same-role overlap           1.000000
semantic/PICF LR                7e-6 / 2e-8
```

Read:

```text
Do not stop from 9400 alone.  Action worsened relative to 9350, but the
structure side did not co-collapse:
  total_minus_action improved 0.011022 -> 0.010196;
  object_pull normalized 0.231192 -> 0.151621;
  active/downstream overlap remained low at about 0.13;
  raw overlap stayed at the known reserve/context ceiling.

This single row therefore does not support another raw-overlap/sidecar/slot-loss
repair.  If 9450/9500 keep increasing action while structure stays healthy, the
remaining root cause is action-visible transfer / rolling-window difficulty,
not anchor ownership collapse.  If action falls back below about 0.041-0.042,
E14E remains a valid maintained-continuation run.
```

Next hard gate:

```text
9450:
  continue if action <= 0.043 and active/downstream overlap stays <= 0.15;
  watch if action is 0.043..0.045 without structure regression;
  stop or branch if action > 0.045 and 9500 confirms the same direction.

9500:
  if action is still rising and structure remains healthy, do not add object
  losses.  Move to an action-readout/CALVIN closed-loop branch.
  if action and structure both degrade, inspect optimizer/config/dataflow.
```

9450 gate:

```text
loss_action_default_equiv       0.040929
loss_total                      0.050369
loss_total_minus_action         0.009440
loss_anchor_pv                  0.487697
loss_anchor_object_pull         0.088557
loss_anchor_object_pull_graph   0.082521
loss_anchor_object_pull_post    0.089476
loss_aqr_denoising              1.162584
loss_mapg_routing               0.415439
loss_slot_jepa                  0.723472
posterior_identity_switch_rate  0.198333
posterior_recycle_rate          0.109311
posterior_active_file_swap      0.000000
posterior_file_swap             0.183750
active/downstream overlap       0.065000 / 0.060706
active/downstream object-core   0.003491 / 0.002594
raw same-role overlap           1.000000
pi_context_entropy_mean         2.932655
pi_context_fused_post_rms_mean  0.700000
preclip_grad_norm               0.677647
```

Read:

```text
9450 rejects the immediate-rebound interpretation of 9400.

Compared with 9400:
  action improves                 0.043734 -> 0.040929;
  total improves                  0.053931 -> 0.050369;
  non-action burden improves      0.010196 -> 0.009440;
  object_pull improves            0.151621 -> 0.088557;
  slot_jepa telemetry improves    0.764355 -> 0.723472;
  active overlap improves         0.130000 -> 0.065000;
  downstream overlap improves     0.131565 -> 0.060706;
  active file swap stays zero.

Therefore E14E remains a valid continuation run.  Do not stop at 9450.
Continue to 9500.  The current evidence supports rolling-window fluctuation
plus maintained prefix-fusion stability, not structural collapse.
```

Local code/dataflow audit while E14E runs:

```text
Date: 2026-05-31

Executed locally with .venv/bin/python:
  scripts/verify_picf_owm_contract.py
  scripts/picf_owm_strict_diagnose.py --fail-on-fail
  scripts/picf_owm_dataflow_trace.py --fail-on-fail
  scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail

Result:
  contract verifier PASS
  strict diagnose ok; WARN only because no runtime metrics/CALVIN eval dir was
    passed to that local invocation
  dataflow trace ok
  MVTrack deep audit PASS
```

Action-path code follow-through:

```text
src/openpi/picf/policy.py:
  _action_context_tokens(...)
    bounds context token count;
    RMS-normalizes / caps context;
    applies an output gate;
    detaches context by default.

  _fuse_action_context_into_prefix(...)
    keeps PI prefix length fixed;
    lets prefix tokens read bounded context by attention;
    caps fused prefix RMS to original prefix scale.

  _training_action_condition_tokens(...)
    mode=prefix_fusion uses the fixed-length prefix-fusion path.
    mode=suffix_cross_attention routes context through the semantic action-side
    adapter instead of extra PI prefix tokens.

src/openpi/picf/paligemma/wrapper.py:
  _apply_action_context_adapter(...)
    lets action suffix tokens query context through gated cross-attention;
    preserves native prefix/suffix positions;
    RMS-caps the residual before adding it to suffix embeddings.

scripts/experiments/.../run_a7_stepindexed_fixedwindow_from7000_30k_20260529.sh:
  E14E inherits the full slot/sidecar/owner-transport command.
  E14E only overrides resume/full optimizer/prefix_fusion/LR-group controls.
  It is not anchor-only and not policy-only.
```

Remaining live hypothesis after code follow-through:

```text
The active object routing machinery is not the current bottleneck if
active/downstream overlap stays low.  The remaining bottleneck is whether
action gradients can exploit the belief/context channel on the rolling training
stream.  This must be tested by E14E continuation and fixed-window/CALVIN
readout, not by adding another object-side auxiliary loss.
```

Resume/window RNG audit:

```text
scripts/picf_core_train.py:
  _step_indexed_window_rng(seed, rank, step, micro_step, retry_count)
    constructs the sampler RNG from global optimizer step, rank, micro-step,
    and retry count.

  train loop:
    if step_indexed_window_rng is true:
      sample_rng = _step_indexed_window_rng(... step=int(step) ...)
      flat_index = sample_rng.integers(0, len(source))
      window = source.window(flat_index, rng=sample_rng)

  startup log:
    prints step_indexed_window_rng.

Conclusion:
  A correct E14E resume does not replay the early sampled-window stream simply
  because it resumed from a checkpoint.  Therefore "optimizer reset repeatedly
  sees the same easy early windows" is excluded unless a specific launcher
  disables step_indexed_window_rng.  The remaining rolling-loss variation is
  normal nonstationary sampled-window difficulty, not a resume RNG bug.
```

Remote E14E startup log confirms:

```text
step_indexed_window_rng=True
optimizer_checkpoint_mode=full
training_strategy=fsdp_full_shard
trainable_scope=all
effective_global_batch=2
unroll_steps=2
burnin_steps=1
semantic_lr_scale=0.35 via launcher
picf_core_lr_scale=0.001 via launcher
```

So this run is a valid maintained continuation test, not a frozen/anchor-only
probe and not a stateful-RNG replay artifact.

## Root-Cause Boundary After 9450

The current question is whether the action platform/rebound has "no root
cause" or whether the model itself is structurally wrong.  The recorded
evidence supports a narrower statement:

```text
Not supported:
  the current candidate is failing because all anchors collapsed;
  raw same-role overlap alone explains action;
  sidecar/mask data is the primary action blocker;
  direct context-token count is the primary blocker;
  resume RNG replay explains E14E;
  model-only resume artifacts explain E14E.

Still possible:
  action readout uses PICF belief weakly;
  rolling sampled windows are nonstationary and hide same-checkpoint progress;
  task-family imbalance makes action rows move without structural failure;
  medium-horizon recurrent state may need CALVIN/video acceptance, not only
    single-step train loss;
  the action optimizer can still be sensitive to phase/scheduler state.
```

Mathematically, the remaining bottleneck is the product of two maps:

```math
z_t = B_\phi(o_{\le t})                    # PICF belief / object context
\hat a_t = \pi_\theta(x_t, F_\psi(z_t))    # PI0.5 action path
```

The object-side losses mostly optimize `B_phi`.  E3/E4/E5/E6 showed that
improving `B_phi` alone does not guarantee lower action on the same windows.
Therefore the plausible failure is not:

```math
B_\phi \text{ cannot form any belief.}
```

It is instead:

```math
\partial L_a / \partial F_\psi(z_t)
```

being too weak, noisy, or task-family dependent for the action path to convert
better belief into a stable action descent.  This is a model-interface problem,
but it is not proof that the full slot/belief architecture is wrong.  It says
the final arbiter must be:

```text
same-window action probes;
stratified task-family probes;
CALVIN closed-loop/video acceptance;
continued E14E rolling stability after full optimizer resume.
```

If E14E reaches 9500/10000 with action near the 0.040-0.043 band and
active/downstream overlap low, the correct action is to continue the maintained
run and evaluate, not to add another slot-side penalty.  If action rises above
about 0.045 on consecutive 50-step gates while structure stays healthy, the
next experiment must target the action readout / task-family sampling path.  If
action and structure degrade together, then inspect dataflow, optimizer groups,
and ownership state before changing architecture.

## Remaining Hypothesis Tree

The remaining root causes must be separated by what co-moves with action.

### R1. Measurement / rolling-window effect

Signature:

```text
rolling action rises or falls;
same-window action between checkpoints is flat or improves;
structure metrics stay healthy.
```

Implication:

```text
No architecture change.  Continue or evaluate with stratified/CALVIN probes.
```

This is not hand-waving: the old 4-22 audit proved online train rows and
checkpoint replay are different mathematical objects.  A rolling row can move
because `P_t(window)` changed, not because `theta_t` got worse.

### R2. Action-transfer bottleneck

Signature:

```text
same-window structure improves;
same-window action does not;
prefix/context ablation shows the action head is weakly sensitive to PICF.
```

Implication:

```text
This is a real model-interface limitation.  The object belief router may be
healthy, but F_psi(z_t) is not yet a high-leverage input to PI0.5 action.
```

Valid repairs:

```text
fixed-length action-side adapter;
full optimizer-state continuation;
semantic-backbone co-adaptation at controlled LR;
closed-loop CALVIN validation to check whether train loss underestimates value.
```

Invalid repairs:

```text
another raw-overlap penalty;
blind SAM resurrection;
deleting sidecar/masks;
directly appending more context tokens and shifting PI0.5 action positions.
```

### R3. Task-family imbalance / specialization

Signature:

```text
one task bucket improves while another bucket worsens;
rolling rows correlate with prompt/task family;
CALVIN failures cluster by subtask.
```

Implication:

```text
Fix sampling/task weighting or use bucket-aware validation.  Do not interpret
one mixed rolling scalar as global architecture failure.
```

### R4. True belief-router structural failure

Signature:

```text
action rises;
active/downstream overlap rises;
object_pull rises;
identity switch/recycle becomes unstable;
anchor videos visibly leave task objects.
```

Implication:

```text
Then the slot/belief design is failing on the data.  Only under this signature
should the next patch target object binding, ownership, or proposal masks.
```

E14E at 9450 does not match R4.  It currently matches either R1 or an improved
R2 branch that still needs 9500/10000 and fixed-window/CALVIN readout.

## E14E 9500 Gate

Remote row:

```text
loss_action_default_equiv       0.043852
loss_total                      0.054094
loss_total_minus_action         0.010242
loss_anchor_pv                  0.492301
loss_anchor_object_pull         0.159055
loss_anchor_object_pull_graph   0.142817
loss_anchor_object_pull_post    0.163058
loss_aqr_denoising              1.162292
loss_mapg_routing               0.425243
loss_slot_jepa                  0.719046
posterior_identity_switch_rate  0.212222
posterior_recycle_rate          0.103955
posterior_active_file_swap      0.000000
posterior_file_swap             0.179375
active/downstream overlap       0.114999 / 0.103212
raw same-role overlap           1.000000
pi_context_fused_post_rms_mean  0.700000
preclip_grad_norm               0.640249
```

Read:

```text
9500 is a watch row, not a stop row.

Compared with 9450:
  action worsened                 0.040929 -> 0.043852;
  total worsened                  0.050369 -> 0.054094;
  non-action burden worsened      0.009440 -> 0.010242;
  object_pull worsened            0.088557 -> 0.159055;
  active overlap worsened but low 0.065000 -> 0.114999;
  downstream overlap worsened but low 0.060706 -> 0.103212;
  slot_jepa improved              0.723472 -> 0.719046;
  active file swap stayed zero.

This does not match R4 because active/downstream overlap remain in the
controlled band, active file swap is zero, slot_jepa improves, and prefix/context
norms are bounded.  It matches R1/R2 ambiguity:
  either 9500 saw a harder rolling window,
  or action transfer remains weak even while structure is mostly healthy.

Decision:
  continue to 9550/10000.
  Stop only if action > 0.045 on consecutive gates or if action and structure
  degrade together.  Otherwise, do not add slot-side penalties; the next
  causal check is fixed-window/CALVIN action readout.
```

## E14E 9550 Gate

Remote row:

```text
loss_action_default_equiv       0.045734
loss_total                      0.055909
loss_total_minus_action         0.010174
loss_anchor_pv                  0.494797
loss_anchor_object_pull         0.148791
loss_aqr_denoising              1.200861
loss_mapg_routing               0.423595
loss_slot_jepa                  0.681072
posterior_identity_switch_rate  0.209444
posterior_recycle_rate          0.102712
posterior_active_file_swap      0.000000
posterior_file_swap             0.173125
active/downstream overlap       0.105000 / 0.092117
raw same-role overlap           1.000000
pi_context_fused_post_rms_mean  0.700000
preclip_grad_norm               0.492948
```

Read:

```text
9550 is the first hard-watch row because action crossed 0.045.
It is not yet a stop row because the structural co-failure signature is absent:
  active/downstream overlap are still low;
  active file swap remains zero;
  recycle is stable;
  slot_jepa improves strongly from 9500;
  prefix/context RMS remains bounded.

This weakens the hypothesis that action rise is caused by renewed anchor
collapse.  It strengthens the R1/R2 ambiguity:
  rolling window or task-family difficulty may have increased;
  action transfer may remain weak even when belief metrics are controlled.

Decision:
  Continue to 9600 for confirmation.
  If 9600 action remains > 0.045, stop E14E and run same-window/family probes.
  If 9600 falls back toward <= 0.043, treat 9550 as rolling-window fluctuation.
```

## E14E 9600 Stop Gate

Remote row:

```text
loss_action_default_equiv       0.049557
loss_total                      0.060206
loss_total_minus_action         0.010649
loss_anchor_pv                  0.507855
loss_anchor_object_pull         0.192412
loss_aqr_denoising              1.219616
loss_mapg_routing               0.432375
loss_slot_jepa                  0.683749
posterior_identity_switch_rate  0.217222
posterior_recycle_rate          0.097741
posterior_active_file_swap      0.000000
posterior_file_swap             0.179375
active/downstream overlap       0.149986 / 0.146595
raw same-role overlap           1.000000
preclip_grad_norm               0.360237
```

Decision:

```text
Stop E14E.  The action row crossed the hard-watch threshold twice:
  9550: 0.045734
  9600: 0.049557

This is not a pure raw-overlap collapse because active/downstream overlap are
still below the historical failure band and active file swap remains zero.
However, 9600 is no longer a harmless fluctuation: action, object_pull,
anchor_pv, denoising, routing, and active/downstream overlap all move in the
wrong direction relative to 9450.

Preserved checkpoints:
  9400
  9500
  9600

Next required experiment:
  no-update exact-window replay on the actual E14E late windows.

Question:
  Did checkpoint quality degrade on the same windows, or did the online rolling
  stream simply become harder?

Do not launch another training branch until this replay/family split is read.
```

## E14F Exact Late-Window Replay Launch

Purpose:

```text
Separate true checkpoint degradation from rolling-window difficulty.
```

Window set:

```text
/mnt/picf_fixed_window_probes/e14e_late100_9551_9600_20260531/
  e14e_late100_9551_9600_windows.jsonl

Source:
  actual E14E window_trace_rank0/1 records for global_step 9551..9600.

Count:
  100 windows, across both ranks.
```

Task mixture:

```text
block_grasp_lift          22
drawer                    21
slider/cabinet            16
block_push_slide          14
block_rotate              11
other                      9
manipulator/button/light   7
```

Probe:

```text
ckpt9400 on GPU0
ckpt9600 on GPU1

No update.
Same windows.
Same args.json.
Same sidecar root.
Same code path as `scripts/picf_fixed_window_action_probe.py`.
```

Decision:

```text
If ckpt9600 is not worse than ckpt9400 on the same windows:
  the 9600 rolling spike is primarily window difficulty.

If ckpt9600 is materially worse on the same windows:
  E14E caused real action degradation.  Then inspect per-bucket deltas and do
  not restart training until the action-transfer/optimizer cause is isolated.
```

Result:

```text
ckpt9400 mean action on late100 = 0.046127
ckpt9600 mean action on late100 = 0.045275
delta 9600 - 9400              = -0.000852
median paired delta            = -0.000450
improved windows               = 63 / 100
worsened windows               = 37 / 100
```

Per-bucket deltas:

```text
block_grasp_lift          n=22  mean_delta=-0.000841  improved=16
block_push_slide          n=14  mean_delta=-0.002068  improved=8
block_rotate              n=11  mean_delta=-0.000641  improved=6
drawer                    n=21  mean_delta=-0.000664  improved=11
manipulator/button/light  n=7   mean_delta=-0.001912  improved=6
slider/cabinet            n=16  mean_delta=-0.000737  improved=11
other                     n=9   mean_delta=+0.000940  improved=5
```

Conclusion:

```text
The 9600 rolling row did not reflect true checkpoint degradation.
On the exact late windows that produced the apparent rebound, ckpt9600 is
slightly better than ckpt9400 overall and in nearly every task bucket.

Therefore the E14E 9600 spike is primarily a rolling-window/sample-difficulty
artifact.  It is not evidence that the current prefix-fusion/full-optimizer
model structurally collapsed.

Next:
  continue with the maintained recipe from the latest healthy checkpoint or
  launch longer validation/CALVIN.  Do not introduce another architecture patch
  based on the 9600 rolling scalar alone.
```

## E14G 30K Continuation And Checkpoint Cleanup

Decision:

```text
Proceed to the next 30K continuation from E14E step9600.
```

Reason:

```text
E14F exact-window replay separated rolling sampled-window difficulty from true
checkpoint degradation.  Step9600 is not worse than step9400 on the same late
windows, so reverting to 9400 or adding another slot-side patch would be a
response to the wrong signal.
```

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_e14g_fullopt_from9600_30k_20260531.sh
```

E14G contract:

```text
resume checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14e_prefix_fullopt_from9300_10300_20260531/9600

num_train_steps:
  30000

save interval:
  1000

keep_last_checkpoints:
  5

optimizer checkpoint mode:
  full

action interface:
  prefix_fusion
  action_context_tokens = 24
  action_context_stopgrad = true
  action_prefix_stopgrad = true
  action_prefix_teacher = ema

trainable boundary:
  PaliGemma semantic backbone only
  PI0/PaliGemma local action/policy head normal trainable path
  V-JEPA, Sonata, AnyTouch pretrain modules frozen
  PICF core at low two-timescale LR

sidecar:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
```

Acceptance gates:

```text
1. The first 100-200 resumed steps must show no NaN, no lost optimizer state,
   and no unexpected fallback to model-only checkpoint resume.
2. Rolling action rows may move with task mixture; do not stop on one sampled
   row unless exact-window replay or CALVIN/video evidence confirms real
   degradation.
3. Stop immediately for structural failure:
   active/downstream overlap returning to the historical collapse band,
   nonfinite action/PICF tensors, optimizer state not loaded, or missing
   sidecar segment coverage.
4. Long-run judgment is fixed-window/CALVIN/video plus rolling train metrics,
   not rolling train action alone.
```

Checkpoint cleanup:

```text
Remote /mnt was full before E14G.

Deleted:
  May-2026 date-marked diagnostic checkpoint/log artifacts that were not part
  of the preserved current evidence chain.

Preserved:
  all pre-May baselines, including 4-22 ablation and full-PICF baselines;
  picf_a7_e14e_prefix_fullopt_from9300_10300_20260531;
  picf_a7_e14c_fullopt_smoke_from9100_maskfix_20260531;
  picf_a7_actionprefix_ema_from6800_action2_30k_20260527.

Manifest:
  /root/ckpt_cleanup_20260531_2235/manifest.txt

Space:
  freed about 240 GiB;
  /mnt went from 100% used to about 86% used.
```

## E14H Action-Pressure Continuation From E14G Step10000

Decision:

```text
Fork a long-horizon action-pressure continuation from the complete E14G
step10000 checkpoint.
```

Reason:

```text
E14G reached step10000 with stable active/downstream overlap and no structural
collapse, but rolling action loss did not show a clear downward slope.

The next causal question is not whether raw reserve overlap is harmful.  The
active/downstream indicators remain healthy.  The question is whether the
accepted E14 interface is under-weighting action relative to the conservative
PICF/structure guard.
```

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_e14h_action4_fullopt_from10000_30k_20260601.sh
```

Contract:

```text
resume checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14g_fullopt_from9600_30k_20260531/10000

optimizer:
  full resume, not model-only, not fresh Adam

horizon:
  30000

save interval:
  1000

keep_last_checkpoints:
  5

only intended change from E14G:
  ACTION_LOSS_WEIGHT = 4.0

unchanged:
  prefix_fusion action interface
  action_context_tokens = 24
  action/prefix stopgrad
  EMA prefix teacher
  PaliGemma semantic backbone trainable scope = backbone_only
  V-JEPA / Sonata / AnyTouch frozen
  PICF core LR scale = 0.001
  semantic LR scale = 0.35
  policy head LR scale = 1.0
```

Why this is a clean causal test:

```text
Do not reset optimizer:
  avoids the known restart/fresh-Adam bonus.

Do not change PICF LR:
  avoids turning the test into a structure instability experiment.

Do not add slot penalties:
  E14G active/downstream overlap is already healthy.

Increase action pressure only:
  isolates whether action plateau is caused by under-weighting the supervised
  policy objective.
```

Expected readout after roughly 8 hours:

```text
If action falls below the E14G 0.04 band while active/downstream overlap remains
below about 0.20:
  accept action-pressure schedule as the next maintained recipe.

If action remains flat:
  action weight alone is not the bottleneck; investigate prefix utility,
  task/window distribution, or semantic/action-head adaptation capacity.

If active/downstream overlap, identity switch, or anchor_pv degrade:
  reject the higher action pressure because it is destabilizing object belief.
```

Launch confirmation:

```text
launched:
  2026-06-01 02:17 CST

tmux:
  picf_a7_e14h_action4_fullopt_from10000_30k_20260601

log:
  /mnt/picf_run_logs/picf_a7_e14h_action4_fullopt_from10000_30k_20260601.log

output:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_e14h_action4_fullopt_from10000_30k_20260601

confirmed:
  resumed from E14G step10000 at step=10000
  optimizer_checkpoint_mode = full
  training_strategy = fsdp_full_shard
  sidecar root = contact_motion_full_tracklets_clean_20260520
  action interface = prefix_fusion
  semantic backbone trainable, scope=backbone_only
  V-JEPA / Sonata / AnyTouch frozen
  semantic lr = 7e-6
  picf_core lr = 2e-8
  policy_head lr = 2e-5
  action loss weight = 4.0 by launcher contract
```

Interim read at step11150:

```text
latest row:
  step                         = 11150
  loss_action_default_equiv    = 0.044113
  loss_action                  = 0.088225
  loss_total                   = 0.098419
  loss_total_minus_action      = 0.010194
  loss_anchor_object_pull      = 0.150522
  loss_anchor_pv               = 0.490316
  loss_slot_jepa               = 0.728078
  posterior_identity_switch    = 0.215556
  posterior_recycle_rate       = 0.075800
  active same-role overlap     = 0.180000
  downstream same-role overlap = 0.156491

last-15-row action band:
  mean = 0.041827
  min  = 0.036503
  max  = 0.046027

last-15-row structure band:
  active overlap mean     = 0.151456
  downstream overlap mean = 0.142050
  anchor_pv mean          = 0.495227
  slot_jepa mean          = 0.714676
```

Interpretation:

```text
E14H has not produced a decisive action improvement.  Increasing
ACTION_LOSS_WEIGHT from 2.0 to 4.0 doubled the weighted action contribution
inside loss_total, but loss_action_default_equiv stayed in the same
approximately 0.04 rolling band.

The structure side did not collapse:
  active/downstream overlap remain below the 0.25 structural reject gate;
  anchor_pv is stable around 0.49-0.50;
  slot_jepa is finite and stable;
  identity/recycle remain in the expected E14 band.

Therefore E14H weakens the "insufficient scalar action pressure" hypothesis.
It does not support keeping ACTION_LOSS_WEIGHT=4.0 as the maintained default.
```

Maintained decision:

```text
Return production/maintained experiments to ACTION_LOSS_WEIGHT=2.0.

Reason:
  2.0 is the legacy/default-equivalent PI0/PICF action scale;
  4.0 did not create an action downtrend;
  4.0 makes loss_total less comparable to previous action-dominant runs;
  4.0 can mask whether non-action terms are bounded by making action dominate
  the total without improving the comparable action metric.

If action-pressure must be tested again, do it as an explicit ablation from a
clean full-optimizer checkpoint, not by converting E14H into the maintained
branch after 4.0-scaled Adam moments have already accumulated.
```

Root-cause update:

```text
The current failure factorization is:

  observations Z
    -> PICF belief B_t
    -> action-visible interface z_t
    -> PI0/PaliGemma action path
    -> action loss.

E14H says:
  increasing d L_total / d L_action does not by itself improve
  loss_action_default_equiv.

Together with E14A/B/C/D/G, the remaining primary hypotheses are:

  H1. action-visible transfer/readout bottleneck:
      z_t is stable but not useful enough to the PI0 action path, so
      d L_action / d B_t is small even when B_t improves.

  H2. rolling-window non-stationarity:
      50-step live rows vary with sampled task/motion/sidecar difficulty and
      must be bridged by fixed/stratified windows before declaring regression.

  H3. action-side optimizer/capacity bottleneck:
      semantic/action-head parameters may be in a low-loss basin where scalar
      reweighting alone changes gradient magnitude but not the useful update
      direction.

Do not treat raw inactive overlap, sidecar deletion, or another slot penalty as
the next primary root-cause path unless an exact-window probe contradicts this
ledger.
```

## 2026-06-01 Follow-Up: Action Readout Causal Audit

The next non-duplicative diagnostic is documented in:

```text
docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md
```

It converts the remaining action plateau question into direct causal probes:

```text
normal / zero / shuffled / RMS-matched-noise / oracle-sidecar context
on identical windows, noise, and diffusion time;

single-batch gradient follow-through from L_action into context, suffix
adapter, semantic action backbone, and PICF core;

stratified same-window task buckets against 4-22 and current checkpoints.
```

Do not start another long train as a root-cause experiment until this readout
audit has been run or explicitly waived.
