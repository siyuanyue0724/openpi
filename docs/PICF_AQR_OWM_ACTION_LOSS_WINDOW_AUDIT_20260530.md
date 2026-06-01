# PICF Action Loss Window Audit - 2026-05-30

This note records the direct code/data audit requested after the action-loss
baseline discussion.  It separates facts proven by artifacts from conclusions
that cannot be proven because the old run did not record window traces.

## Code Path

Training samples a CALVIN window as:

```text
global step, rank, micro step -> sampler RNG -> flat_index -> segment/start_step
```

The relevant implementation is:

```text
scripts/picf_core_train.py::_step_indexed_window_rng
scripts/picf_core_train.py::_CalvinTransitionSource.window
```

The current trainer writes `window_trace_rank*.jsonl`; older 4-22 logs do not.

## Direct A7 Audit Command

The following temporary script was run on A7:

```text
/tmp/picf_loss_window_audit.py
```

It read:

```text
/mnt/checkpoints/picf_core/picf_core/
  picf_v22_ablated_pi05_30000_ckpt2500_print100_20260422_r2/metrics.jsonl

/mnt/checkpoints/picf_core/debug/
  picf_v22_ablated_pi05_30000_ckpt2500_print100_20260422_r2*.log*

/mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/
    metrics.jsonl
    window_trace_rank0.jsonl
    window_trace_rank1.jsonl

/mnt/picf_fixed_window_probes/step8000_vs_9000_20260530/
  step8000_windows.jsonl
  step9000_windows.jsonl
  accepted_flat_indices.json
```

## Proven Facts

### 1. The old 4-22 low action loss was real in the train stream

From `metrics.jsonl`:

```text
old 4-22 metric rows: 202
step range: 100..20100
min loss_action: 0.018393

longest consecutive metric rows <= 0.03:
  43 metric rows, step 15800..20000, mean 0.023742

longest consecutive metric rows <= 0.04:
  77 metric rows, step 12600..20100, mean 0.028410
```

From debug tqdm rows, deduped by step using the minimum visible value:

```text
parsed rows: 12564
visible range: step 1..20155

17500 local window:
  n=101, mean=0.022101, min=0.002000, max=0.121400

19250..20200:
  n=805, mean=0.020052, min=0.001300, max=0.119000
```

Therefore it is wrong to describe the old low band as "one or two lucky easy
windows."  The train stream really contained long low-loss regions.

### 2. The old 4-22 run cannot identify which task windows caused the low band

The audit found:

```text
old 4-22 window trace exists? []
```

Therefore we cannot prove from stored artifacts whether the old low band was a
same-task cluster, a resume-window replay artifact, a true global improvement,
or a mixture of these.  The exact CALVIN windows were not logged.

### 3. Current corrected runs can identify sampled windows

The current step-indexed run recorded:

```text
window trace rows: 25
total traced windows: 2500
```

Each 50-step row contains 100 windows across two ranks, with prompts and
segment/start metadata.  Example:

```text
step 7600 action=0.051639
  buckets: block+push:12, block+turn+grasp:12, block+grasp:12,
           block+slider:11, block+turn:11

step 7350 action=0.037555
  buckets: block+turn:13, block+slider:11, block:10,
           block+grasp:10, block+drawer+grasp:8
```

The simple bucket correlations on this short trace are not causal proof, but
they show that train-row action is sensitive to the sampled task-window mixture:

```text
drawer             corr=+0.511
drawer+grasp       corr=+0.416
drawer+slider      corr=+0.374
switch+push        corr=+0.308
block+push         corr=+0.225
block+grasp        corr=-0.293
block+slider       corr=-0.291
```

### 4. Fixed-window step8000 -> step9000 did not prove action improvement

The fixed-window probe used the same 64 accepted windows for both checkpoints:

```text
step8000 mean action: 0.057919
step9000 mean action: 0.057213
delta:               -0.000706

improved windows: 29 / 64
worsened windows:  35 / 64
paired median delta: +0.000633
```

Therefore the step8000 -> step9000 continuation did not produce a meaningful
action improvement on the same fixed mixed window set.

Per bucket, the result was mixed:

```text
block+push:
  n=5, delta=-0.004311, improved=3/5

block+slider:
  n=4, delta=+0.003338, improved=1/4

block+drawer+grasp:
  n=4, delta=+0.005414, improved=1/4

switch:
  n=2, delta=+0.004059, improved=0/2

light+turn:
  n=3, delta=-0.004862, improved=1/3
```

Critical limitation:

```text
This fixed64 set is a hard diagnostic set.
It is not a representative full-CALVIN validation set.
It must not be reused as a global model-quality claim without explicit approval
and a stratified sampling contract.
```

Maintained use:

```text
allowed:
  same-window regression checks;
  reproducing a suspected action-rebound failure on the exact same windows;
  hard-case diagnostics after explicitly stating it is hard-case only.

disallowed:
  claiming current model is generally better/worse than old 4-22 from fixed64 alone;
  replacing train-stream rolling metrics or stratified probes;
  launching new fixed64 comparisons without documenting why this biased hard set is needed.
```

## 2026-05-30 Probe Import Fix

The fixed-window probe must import the active worktree, not an older `openpi`
package from the venv.  The script now prepends both:

```text
repo_root
repo_root/src
```

to `sys.path` before importing `scripts.picf_core_train`.

Reason:

```text
Without repo_root/src, remote probes can load stale `openpi.picf` code from the
venv while still loading the current `scripts/` directory.  That creates mixed
code/config measurements and can invalidate same-window causal conclusions.
```

Use all fixed-window probe results produced before this import fix only when the
log proves the active worktree was on `PYTHONPATH`.

### 5. The old 0.034 and the current 0.057 are not directly comparable

The earlier comparison between an old `0.034` exact-window replay and the
current `0.057` fixed-window probe was the wrong comparison target.  They were
not computed on the same sampled windows.

The corrected same-window bridge is:

```text
window set:
  /mnt/picf_fixed_window_probes/step8000_vs_9000_20260530/step8000_windows.jsonl

old 4-22 ckpt20000 on this exact 64-window set:
  loss_action_default_equiv = 0.050257

current ckpt8000 on this exact 64-window set:
  loss_action_default_equiv = 0.057919

current ckpt9000 on this exact 64-window set:
  loss_action_default_equiv = 0.057213
```

Therefore the rigorous same-window conclusion is not `0.034 vs 0.057`.
It is:

```text
current step8000 is worse than old 4-22 step20000 by about 15.25%
current step9000 is worse than old 4-22 step20000 by about 13.84%
```

This is a real gap, but smaller and more precise than the invalid mixed-window
comparison implied.

### 6. Why old train-stream 0.021 and old replay 0.034 can both be true

The old 4-22 `metrics.jsonl` rows are 100-step online train-stream means:

```text
old metric row at step T = mean_t L(theta_t, window_t), t in T-99..T
```

A checkpoint replay is a different mathematical object:

```text
fixed replay at checkpoint T = mean_t L(theta_T, window_t), t in selected windows
```

The audit verified this with the old 4-22 checkpoint:

```text
old metrics row step20000:
  loss_action = 0.021365

old ckpt20000 replayed on exact old source step 20000:
  loss_action_default_equiv ~= 0.021921

old ckpt20000 replayed on exact old source steps 19901..20000:
  loss_action_default_equiv = 0.034396
```

So the final source step is consistent with the logged row, but replaying the
whole preceding 100-step window under the final checkpoint is higher.  This is
expected because it evaluates earlier windows with `theta_20000`, not with the
historical online parameters `theta_19901..theta_20000`.

### 7. What fixed64 is, and why it is not the old continuous train stream

The fixed64 probe is a no-update checkpoint replay:

```text
fixed64:
  load one saved checkpoint theta_C
  load exactly 64 saved (segment, start_step, prompt) windows
  run forward only
  average L(theta_C, window_i)
  no backward
  no optimizer step
  no online parameter trajectory
```

The continuous training log is different:

```text
continuous train metric row:
  for each step t:
    sample stochastic train windows
    run forward/backward
    optimizer.step() changes theta_t -> theta_{t+1}
  every log_interval:
    average recent online outputs
```

The code path confirms random window sampling, not a sequential epoch pass:

```text
flat_index = sample_rng.integers(0, len(source))
window = source.window(flat_index, rng=sample_rng)
start_step_id = rng.integers(slot.first_valid_start_step_id, slot.valid_start_exclusive)
```

The old low band therefore does not prove that the step20000 checkpoint has
`0.02` loss on every mixed validation window.  It proves the online train stream
had low loss on the sampled windows it saw at those historical parameter states.

The current fixed64 window set is also empirically harder than the old exact
resume-window set for the old 4-22 checkpoint:

```text
old 4-22 ckpt20000 on current step8000 fixed64:
  mean    0.050257
  median  0.041608
  max     0.179390

old 4-22 ckpt20000 on old resume17500 source steps 19901..20000:
  mean    0.034396
  median  0.028566
  max     0.124717
```

High-loss examples in the current fixed64 set under old 4-22:

```text
move up the switch                              0.179390
turn on the green lamp                          0.136108
rotate the pink block 90 degrees to the right   0.127510
slide the door to the right side                0.119766
grasp the door handle and slide the door right  0.114712
toggle the light switch to turn off the light   0.104490
push the button to turn on the led              0.103822
```

This proves the same old checkpoint can score near the old online low point on
some old source windows but score around `0.05` on the current mixed fixed64
window set.  That is a data-window-set difference, not evidence that the probe
loss scalar is mismatched.

Operational rule:

```text
Treat fixed64 as quarantined hard-diagnostic evidence.
Do not reuse it as a primary action benchmark.
Use stratified fixed windows or train-stream rolling metrics for primary
training decisions.
```

## Correct Interpretation

The old `0.02` train-stream band is real, but it is not a fixed validation
number.  Since the old run lacks per-step window traces, the exact old online
100-step composition cannot be recovered fully from artifacts.

The current fixed-window result is definite: the inspected step8000 -> step9000
branch did not meaningfully improve action on the same 64 windows.  Against old
4-22 step20000 on those same 64 windows, the current branch is still behind by
roughly 14-15%.

## Non-Convergence / Rebound Evidence

The hypothesis that the old 4-22 low-loss band was caused by consecutive
same-task windows is not supported by the reconstructed old resume stream.

For the old `resume17500` stream at the offset corresponding to source steps
`19901..20000`, the exact reconstructed 200 rank-window records have:

```text
rows             200
unique prompts   151
unique buckets    25
max prompt run     1
max bucket run     3
```

So the available reconstructed old sampler stream is mixed, not a contiguous
single-task or same-prompt curriculum.  This does not recover unlogged old
online windows with byte-level certainty, but it falsifies the specific claim
that the known resume-reset stream is a long consecutive similar-task block.

The short-window fast drops are instead explained by a different verified
mechanism: optimizer/scheduler phase reset can rapidly improve action in a
local online train stream, but that improvement has repeatedly failed to remain
stable.

Archived examples:

```text
fresh optimizer from A7 step2500:
  step2700 action = 0.03155
  step3100 action = 0.03366
  step3300 action = 0.03456

mid-LR fresh optimizer from A7 step5500:
  step6400 action = 0.02853
  step6550 action = 0.04335
  step6600 action = 0.03994
  step6800 action = 0.04138

EMA prefix run:
  step7000 action = 0.02127
  step7500 action = 0.03857
  step8000 action = 0.04371
  step8300 action = 0.04701
```

These runs also recorded that active/downstream overlap and non-action budget
were controlled during the rebound.  Therefore the repeated rebound is not
explained by raw support overlap alone, weak sidecar scaffold dominance, or
action-prefix EMA drift alone.

The current step8000 -> step9000 branch adds one stationary-checkpoint fact:

```text
same fixed64 windows:
  current step8000 action = 0.057919
  current step9000 action = 0.057213
```

Thus this branch did not improve action quality on the same fixed windows.  The
mathematical diagnosis that remains consistent with all recorded evidence is:

```text
fresh optimizer / phase reset can release short online action fitting;
after the low-loss basin is reached, stochastic action/semantic updates and
mixed-task windows no longer give a stable descent direction for the saved
checkpoint;
the effect appears in action metrics while structure metrics remain mostly
controlled, so it is an action-side optimization/generalization problem rather
than a renewed anchor-overlap collapse.
```

Claims not supported by the artifacts:

```text
not proven:
  "old 4-22 had 0.02 checkpoint quality on the full dataset"

not supported:
  "old low band was just consecutive same-task windows"

proven:
  "old online train loss had a long low band"
  "old ckpt20000 is 0.050257 on the current fixed64 hard set"
  "current 8000->9000 barely improves on the same fixed64 hard set"
```

The only rigorous next step is a stratified fixed-window probe:

```text
block-only windows
drawer-only windows
slider-only windows
button/switch/light windows
mixed all-task windows
```

If a checkpoint improves on one bucket but worsens on another, the issue is
task-window specialization / curriculum balance.  If it fails on every bucket,
the issue is action-interface or optimizer capacity, not data-window mix.

## Causal Closure Plan

The previous interpretation "maybe easy windows" is too vague.  The actionable
causal test is:

```text
1. Read old args.json and checkpoint metadata.
2. Reconstruct the old non-step-indexed sampler stream from code:
   rng = np.random.default_rng(seed + 17 * rank)
   flat_index = rng.integers(0, len(segment_sampling_slots))
   start_step = rng.integers(segment.start, segment.valid_start_exclusive)
3. Simulate the candidate stream after each old resume point.
4. Compare low/high loss regions against:
   a. resume-reset stream offset,
   b. prompt/task bucket composition,
   c. fixed-window checkpoint quality by bucket.
5. Run stratified fixed-window probes on old and current checkpoints.
```

This separates four explanations:

```text
A. single task-cluster shortcut:
   low region is dominated by one task type;

B. repeated mixed-curriculum shortcut:
   low region follows the same deterministic mixed sampled-window stream after
   resume, but is not a single task cluster;

C. true global action improvement:
   fixed-window probes improve across all task buckets;

D. interface/optimizer failure:
   fixed-window probes do not improve even when live train rows are low.
```

## Old 4-22 Resume Metadata

The old checkpoint metadata was loaded with the training environment:

```text
checkpoints inspected:
  5000, 7500, 10000, 12500, 15000, 17500, 20000

metadata keys:
  step
  args
  timestamp
  checkpoint_format
  optimizer_state_saved
  optimizer_checkpoint_mode
  grad_clip_controller

optimizer_state_saved:
  False for every inspected checkpoint

rng state:
  absent from metadata
```

The debug logs show explicit resumes:

```text
resume at 12500
resume at 15000
resume at 17500
```

Therefore, under the trainer code path used here, resume does not restore
optimizer state or RNG state.  The process-level sampler RNG is re-created from
`seed + 17 * rank`.

## Old Sampler Reconstruction - First Pass

Old run args:

```text
seed = 0
world_size = 2
unroll_steps = 2
action_horizon = 16
log_interval = 100
picf_mode = ablated
visual_mode = stub
tactile_mode = stub
step_indexed_window_rng = absent
```

The CALVIN annotation file has:

```text
valid segment slots = 17870
all annotations = 17870
```

The first windows after any resume with RNG reset are deterministic.  For
rank 0:

```text
flat 15200 segment 15200 start 63458   block+slider       go slide the blue block to the left
flat 9134  segment 9134  start 460036  block+drawer+grasp go towards the blue block in the drawer and pick it up
flat 5500  segment 5500  start 407440  block+drawer       sweep the block into the drawer
flat 1344  segment 1344  start 1465179 block+grasp        grasp the pink block and lift it up
flat 3132  segment 3132  start 844159  block+grasp        pick up the blue block from the table
```

For rank 1:

```text
flat 13241 segment 13241 start 216070  block+grasp+turn   grasp the blue block and rotate it right
flat 1918  segment 1918  start 1395783 block              sweep the blue block to the right
flat 8183  segment 8183  start 293479  block+drawer+grasp go towards the blue block in the drawer and grasp it
flat 13610 segment 13610 start 920299  other              move the door all the way to the right
flat 640   segment 640   start 1185753 grasp              put the grasped object in the cabinet
```

The reset stream is not a single-task cluster.  At offsets corresponding to
hundreds or thousands of local steps after resume, 100-window buckets remain
mixed.  Examples:

```text
offset 0:
  block+grasp:17, block+slider+grasp:10, block+drawer+grasp:8,
  block:8, block+push:7, block+grasp+turn:7

offset 1500:
  block+slider:12, block+grasp:12, block+turn:10, block+push:10,
  block+slider+grasp:9, block+grasp+turn:7

offset 1750:
  block+drawer+grasp:13, block+grasp:12, block+slider:11,
  block+turn:8, block:8, block+grasp+turn:7

offset 2400:
  block+grasp:19, block+slider:10, switch+light+turn:7,
  block+turn:7, block:6, block+drawer+grasp:6
```

Updated conclusion:

```text
The old sustained 0.02 band should not be explained as "same task for hundreds
of steps."  The reconstructed old resume stream is mixed.

The stronger artifact-supported explanation is:
  repeated deterministic mixed-window curriculum after resume
  + no optimizer-state restore
  + no RNG-state restore
  + already-trained action model.

This can create long low train-stream regions without proving fixed-window
global improvement.
```

## Remaining Decisive Test

Run fixed-window probes by task bucket:

```text
old 4-22 checkpoints:
  12500, 15000, 17500, 20000

current checkpoints:
  step8000, step9000

window sets:
  all-mixed
  block-only
  drawer-only
  slider-only
  button/switch/light-only
```

Expected decisive outcomes:

```text
If old 17500/20000 beats all buckets:
  old 0.02 reflects broad action improvement.

If old 17500/20000 beats only block-like buckets:
  old 0.02 reflects task-specialized action capability.

If old train stream is 0.02 but fixed buckets stay near 0.05:
  old 0.02 is train-stream/repeated-curriculum specialization, not a stable
  checkpoint-quality estimate.
```

## Probe Bug Found - 2026-05-30

The existing fixed-window probe has an important limitation:

```text
scripts/picf_fixed_window_action_probe.py:
  window = source.window(flat_index)
```

Because no RNG is passed, `_CalvinTransitionSource.sample_window_metadata`
chooses:

```text
start_step_id = slot.first_valid_start_step_id
```

Therefore the current `--flat-index-json` probe fixes the segment id but not the
true sampled `start_step`.  It is a fixed-segment-start probe, not a true
same-window probe.

This does not invalidate the structural comparison completely, but it is not
strong enough to close the action-loss causality question.  The corrected probe
must support explicit window records:

```json
{"segment": 6409, "start_step": 1000669, "prompt": "push the red block to the left"}
```

and load that exact `(segment,start_step)` pair without RNG.

## Corrected Execution Plan

### Step 1 - Patch exact-window probe

Add a deterministic source method:

```text
window_from_metadata(segment_id, start_step_id)
```

or an equivalent probe-only path that constructs `_TransitionWindow` from the
explicit metadata.  Add `--window-jsonl` to
`scripts/picf_fixed_window_action_probe.py`.

### Step 2 - Reconstruct old resume streams

Use the preserved old code path:

```text
/root/openpi/scripts/picf_core_train.py
  rng = np.random.default_rng(args.seed + 17 * rank)
  flat_index = int(rng.integers(0, len(source)))
  window = source.window(flat_index, rng=rng)
```

Generate exact candidate windows for:

```text
resume from 12500:
  local offsets 0..2499

resume from 15000:
  local offsets 0..2499

resume from 17500:
  local offsets 0..2655
```

The old debug logs prove these resume points existed, and metadata proves no
optimizer/RNG state was saved.

### Step 3 - Evaluate matching windows

For each saved old checkpoint:

```text
12500, 15000, 17500, 20000
```

evaluate exact reconstructed windows corresponding to:

```text
first 100 after resume
offset 1500..1599 after resume
offset 1750..1849 after resume
offset 2400..2499 after resume
```

This directly tests whether old low train rows are reproduced by the replayed
resume stream.

### Step 4 - Stratified task probes

Build exact `(segment,start_step)` window sets for:

```text
block
drawer
slider
button/switch/light
mixed balanced
```

Then evaluate old and current checkpoints on all sets.

### Final Decision Rule

```text
If old 17500/20000 is low on replayed resume windows but not on balanced
stratified windows:
  root cause = resume-replayed mixed curriculum / local stream specialization.

If old 17500/20000 is low on all stratified windows:
  root cause = genuinely stronger action checkpoint.

If current checkpoints improve structure but not exact-window action:
  current recipe improves router health but not action transfer.
```

## Resume Semantics Clarification - 2026-05-30

The user reports that the 4-22 training was launched once.  The artifact-level
audit must not translate that into "manual restarts."  The verified statements
are narrower:

```text
trainer save path:
  should_save -> _save_checkpoint -> prune old checkpoints -> continue loop

trainer code does not:
  sys.exit after save
  os.exec after save
  spawn subprocess after save
  automatically call resume after every save
```

The old code only logs `Resumed from ...` in the startup resume branch:

```text
if args.resume_checkpoint is not None:
  resume_path = Path(args.resume_checkpoint)
elif args.resume:
  resume_path = _resolve_resume_path(...)
...
logging.info("Resumed from %s at step=%s", resume_path, start_step)
```

The visible archived logs contain explicit startup resumes:

```text
resume12500_20260426_r3.log:
  Resumed from .../12500 at step=12500

resume15000_20260426_r4.log:
  Resumed from .../15000 at step=15000

resume17500_20260427_r5.log:
  Resumed from .../17500 at step=17500
```

Therefore the precise conclusion is:

```text
The trainer did not auto-resume after saving.
The artifact set visible on A7 is not one uninterrupted trainer process from
0 to 20000.

This does not prove the user manually restarted it.  It proves some outer
process/session interruption/relaunch/resume path occurred in the saved
artifact history.
```

Checkpoint metadata also shows:

```text
optimizer_state_saved = False
rng_state = absent
```

So every resume in this artifact family re-creates the sampler RNG from:

```text
np.random.default_rng(args.seed + 17 * rank)
```

This is why exact-window replay is required before using old `0.02` train rows
as a claim about broad checkpoint quality.

## Corrected Probe Deployment - 2026-05-30

Local and A7 probe code now supports exact `(segment,start_step)` windows:

```text
scripts/picf_core_train.py
  _CalvinTransitionSource.window_from_metadata(segment_id, start_step_id)

scripts/picf_fixed_window_action_probe.py
  --window-jsonl

scripts/picf_generate_exact_windows.py
  --mode old-resume
  --mode stratified
```

An additional compatibility issue was found while replaying old artifacts:

```text
newer probe script + older remote PicfCoreConfig
```

The trainer now contains `_spec_default(...)` legacy fallbacks for post-4-22
fields so old `args.json` replay is not broken by newer config fields.  On A7,
the corrected Python path is:

```text
PYTHONPATH=/root/openpi_probe_current_20260529/src:/root/openpi_probe_current_20260529
```

Using only the repository root in `PYTHONPATH` incorrectly imports
`/root/openpi/src/openpi/...` from the old environment and mixes source
versions.

## Exact-Window Probe Results - 2026-05-30

### Current probe code reading old 4-22 checkpoint

The corrected probe was first run from the current audit repo against the old
4-22 step-20000 checkpoint:

```text
checkpoint:
  /mnt/checkpoints/picf_core/picf_core/
    picf_v22_ablated_pi05_30000_ckpt2500_print100_20260422_r2/20000

exact resume windows:
  /mnt/picf_exact_window_audit_20260530/
    old_resume17500_skip2400_steps100.jsonl

stratified windows:
  /mnt/picf_exact_window_audit_20260530/
    old_stratified_per32.jsonl
```

Results:

```text
old20000 on exact resume17500 offset 2400 windows:
  accepted windows = 200
  loss_action_default_equiv = 0.0343960554
  train/eval mode check = identical at 0.0343960554

old20000 on stratified per32 windows:
  accepted windows = 256
  loss_action_default_equiv = 0.0399663940
```

The train/eval-mode equality rules out dropout/mode mismatch as the reason the
probe does not reproduce the old `0.02` train-stream band.

### Old-native probe reading old 4-22 checkpoint

To exclude a current-code/old-artifact mismatch, the exact-window probe was
then run inside the old `/root/openpi` code tree with the old virtual
environment.  The only old-code patch was the probe-only exact window reader:

```text
_CalvinTransitionSource.window_from_metadata(segment_id, start_step_id)
```

Results:

```text
oldnative20000 on exact resume17500 offset 2400 windows:
  accepted windows = 200
  elapsed = 359.279 s
  loss_action_default_equiv = 0.0343960554
  loss_action_pos = 0.0953970387
  loss_action_rot = 0.2000373624
  loss_action_gripper = 0.2017751293

oldnative20000 on stratified per32 windows:
  accepted windows = 256
  elapsed = 395.258 s
  loss_action_default_equiv = 0.0399663940
  loss_action_pos = 0.1318523114
  loss_action_rot = 0.2204686777
  loss_action_gripper = 0.2094214944
```

The old-native results exactly match the current-probe results.  Therefore the
gap is not caused by newer probe code, newer model code, train/eval mode, or a
Python import mixup.

### Log-row alignment check

The first exact-window probe compared a 100-step window mean against one old
`metrics.jsonl` row.  That was still too coarse.  A follow-up probe generated
the neighboring exact windows:

```text
old20000 on resume17500 skip2300:
  source steps = 19801..19900
  100-step / 200-window mean = 0.0346597895

old20000 on resume17500 skip2400:
  source steps = 19901..20000
  100-step / 200-window mean = 0.0343960554

old20000 on resume17500 skip2500:
  source steps = 20001..20100
  100-step / 200-window mean = 0.0406551328
```

Within `skip2400`, the exact last source step is close to the old logged row:

```text
exact source_step=20000 mean over ranks = 0.0219208677
old metrics.jsonl step=20000 loss_action = 0.0213650092
```

Within `skip2500`, the exact last source step is close to the next old row:

```text
exact source_step=20100 mean over ranks = 0.0350923510
old metrics.jsonl step=20100 loss_action = 0.0341463387
```

The old trainer code shows that `metrics.jsonl` rows are not single-batch
values.  They are accumulated interval means:

```text
per rank:
  metric_accum.update_from_outputs(outputs) every training step
  averages = metric_accum.averages() at log_interval

across DDP:
  averages = reduce_mean(averages)

then:
  metric_accum = _MetricAccumulator()
```

Therefore a 4-22 row at step 20000 is the online mean over the 100 training
states in that interval, not the step20000 checkpoint evaluated on those same
100 windows.  The probe uses one checkpoint, `20000`, and replays all windows
under that single parameter state.  Those are different quantities:

```text
old metrics row:
  mean_t L(theta_t, window_t), t = 19901..20000

exact-window probe:
  mean_t L(theta_20000, window_t), t = 19901..20000
```

This is why the last step can be close to the old row while the replayed
100-step window mean is higher.  The low `0.02` historical rows are real
online training losses, but they do not imply that the saved step20000
checkpoint has `0.02` fixed-window quality on the whole preceding interval.

## Final Causal Conclusion - 2026-05-30

The verified facts are now:

```text
1. The old 4-22 metrics really contain a long low train-stream band around
   loss_action ~= 0.02.

2. The old trainer does not auto-resume after saving.  The archived artifact
   history nevertheless contains startup resumes at 12500, 15000, and 17500.
   This proves the visible artifact family is not one uninterrupted trainer
   process from 0 to 20000, but it does not prove who or what restarted it.

3. Old checkpoints do not save optimizer state or RNG state.  Any resume in
   that artifact family recreates sampler RNG from seed + 17 * rank.

4. Exact replay of the old resume17500 stream with the single saved step20000
   checkpoint gives 0.034396 on source steps 19901..20000.  This does not
   contradict the old `0.02` metrics row, because the old row is an online mean
   over changing theta_t, not a single-checkpoint replay mean.

5. Balanced stratified exact windows give 0.039966.
```

Therefore the old `0.02` rows cannot be used as a fixed validation baseline for
current training.  They are online training-trajectory interval means, not
single-checkpoint fixed-window estimates of step20000 quality.

The strict comparison baseline for action loss should now be:

```text
old 4-22 step20000 online logged interval:
  loss_action ~= 0.0214

old 4-22 step20000 checkpoint replay on exact resume-like windows:
  loss_action_default_equiv ~= 0.0344

old 4-22 step20000 checkpoint replay on balanced stratified windows:
  loss_action_default_equiv ~= 0.0400
```

Any current run should compare like with like:

```text
live online train rows -> old online train rows
single-checkpoint fixed-window probes -> old single-checkpoint fixed-window probes
balanced probes -> old balanced stratified probes
```

The `0.02` number is valid only for the online-train-loss category.

## Follow-up Rule

For all future claims about action improvement:

```text
Use live train loss only as a training-health signal.
Use exact-window probes for checkpoint quality.
Record window_trace_rank*.jsonl for every serious run.
Never compare current fixed-window/probe loss to old train-stream rows without
replaying the same exact windows.
```

## 2026-05-31 E7 Maintained Fix: Action-Side Suffix Cross-Attention

The same-window E6 probe gives a stricter root cause than "slot overlap is
bad":

```text
gate0 ~= gate07 ~= fusion24 on the same 24 windows.
append24 is worse.
```

Therefore the current bottleneck is not that PICF lacks more auxiliary slot
pressure. The bottleneck is that the PI0.5 action expert can ignore passive
PICF prefix/fused-prefix evidence, while direct append harms the pretrained
prefix/suffix layout.

The maintained E7 route is:

```text
native PI prefix:
  unchanged

PICF belief/context:
  bounded, stop-gradient by default, not appended to prefix

action suffix tokens:
  query PICF belief/context through a trainable gated cross-attention adapter

position ids:
  unchanged from native PI0.5

residual scale:
  gate initialized at sigmoid(-2) ~= 0.12
  adapter residual RMS-capped to current suffix RMS

width bridge:
  PICF context lives in PI prefix width
  action suffix lives in action-expert width
  E7 uses an explicit trainable context-in projection before cross-attention
```

Mathematically, this changes the action computation from:

```text
suffix_out = F_theta([PI_prefix, PICF_prefix, action_suffix])
```

to:

```text
action_suffix' = action_suffix
               + sigmoid(g) * RMSCap(CrossAttn(action_suffix, W_in PICF_context))

suffix_out = F_theta([PI_prefix, action_suffix'])
```

This targets the measured failure mode directly: PICF conditioning is moved to
the action side while preserving PI0.5 token layout. It is not another slot
loss and it does not make PICF a reconstruction decoder.

Guardrails:

```text
direct append:
  historical ablation only

prefix_fusion:
  safe diagnostic, but not sufficient on E6 fixed windows

suffix_cross_attention:
  maintained next diagnostic/fix

first E7 training scope:
  semantic_trainable_scope=action_adapter_only
  action_context_stopgrad=1
  picf_core_lr_scale=0.001
  training_strategy=ddp for this adapter-only diagnostic
  reason: FSDP use_orig_params=False cannot flatten mixed frozen/trainable
          tensors inside the semantic root
```

Acceptance for E7:

```text
1. adapter metrics must appear in metrics.jsonl:
   pi_context_adapter_token_count
   pi_context_adapter_gate
   pi_context_adapter_attention_entropy_mean
   pi_context_adapter_residual_rms_mean

2. checkpoint load from old runs must tolerate missing adapter weights.

3. same-window action loss must separate from gate0/fusion24 if the interface
   hypothesis is correct.

4. if action loss still does not move, the remaining root cause is downstream
   action-head/backbone adaptation rather than PICF belief routing.
```

## 2026-05-31 Prefix/Context Causal Probe

The `picf_fixed_window_action_probe.py` import path bug was fixed before this
probe: it now forces the checked-out repo root and `src/` ahead of installed
packages.  This prevents mixed stale-code measurements.

Same checkpoint and same 24 accepted windows:

```text
gate0:
  action_prefix_output_gate=0.0
  action_context_tokens=0
  loss_action_default_equiv=0.064012

gate07:
  action_prefix_output_gate=0.7
  action_context_tokens=0
  loss_action_default_equiv=0.064106

fusion24:
  action_prefix_output_gate=0.7
  action_context_tokens=24
  action_context_integration=prefix_fusion
  loss_action_default_equiv=0.064091

append24:
  action_prefix_output_gate=0.7
  action_context_tokens=24
  action_context_integration=append
  loss_action_default_equiv=0.076344
```

Conclusion:

```text
The action loss is insensitive to the normal PICF prefix and to fixed-length
context fusion, but it is harmed by direct append.  This rules out "raw overlap
alone" as the primary action plateau explanation for this checkpoint.  The
current root cause is an action-interface bottleneck: the PI action expert is
not getting useful, high-leverage conditioning from the PICF belief state.
```

Next diagnostic/fix:

```text
Do not add more slot losses as the first response.  First prove or repair the
action-visible interface:
  1. run an action-output sensitivity probe across normal/zero/random/strong
     prefix on identical noise/time;
  2. if sensitivity is near zero, move PICF belief injection from passive extra
     prefix tokens to a trainable, gated action-side adapter that preserves
     prefix length and suffix positions;
  3. keep direct append disabled except for historical ablations.
```

## 2026-05-31 E7 Runtime Read And E8 Plan

E7 runtime row at step 9150:

```text
experiment:
  picf_a7_stepindexed_from9100_suffixadapter_9400_20260531

resume:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_from9000_prefixfusion_9300_20260530/9100

contract:
  action_context_integration=suffix_cross_attention
  semantic_trainable_scope=action_adapter_only
  action_context_stopgrad=1
  picf_core_lr_scale=0.001
  training_strategy=ddp

step9150:
  loss_action_default_equiv=0.047400
  loss_total_minus_action=0.010265
  loss_anchor_pv=0.501278
  loss_anchor_object_pull=0.154533
  loss_aqr_denoising=1.202261
  loss_slot_jepa=0.916385
  active/downstream overlap=0.200000 / 0.188100
  pi_context_adapter_token_count=28
  pi_context_adapter_gate=0.119178
  pi_context_adapter_attention_entropy_mean=3.075562
  pi_context_adapter_residual_rms_mean=0.269857
```

Matched E6 reference:

```text
prefixfusion step9100:
  loss_action_default_equiv=0.044162
  loss_total_minus_action=0.011239
  active/downstream overlap=0.135000 / 0.142756
```

Read:

```text
The suffix adapter is wired correctly and non-degenerate: it has context tokens,
a nonzero gate, nonzero attention entropy, and nonzero residual RMS.  It has
not yet improved action loss over the immediately preceding prefix-fusion
baseline.

This narrows the root cause.  The remaining plausible action-side bottleneck is
not "PICF context cannot reach the suffix at all"; it is that adapter-only
capacity cannot remap PICF belief into the frozen action flow basis quickly
enough.  The next causal diagnostic must train the wrapper-local action
in/out projections and time MLP together with the suffix adapter while keeping
the PaliGemma/Gemma backbone and PICF belief effectively frozen.
```

E7 step9200 confirmation:

```text
loss_action_default_equiv=0.047333
loss_total_minus_action=0.010093
loss_anchor_pv=0.499756
loss_anchor_object_pull=0.146887
loss_aqr_denoising=1.154685
loss_mapg_routing=0.422907
loss_slot_jepa=0.862420
active/downstream overlap=0.100000 / 0.097959
pi_context_adapter_gate=0.119156
pi_context_adapter_residual_rms_mean=0.327289
```

Decision:

```text
Stop E7 after step9200.  Adapter-only improves structural diagnostics but does
not improve action over E6.  This rejects "current active overlap is the action
plateau cause" for this checkpoint and escalates to E8 action-head+adapter.
```

E8 plan:

```text
experiment:
  picf_a7_stepindexed_from9100_suffixadapter_actionhead_9400_20260531

script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from9100_suffixadapter_actionhead_300_20260531.sh

only change from E7:
  semantic_trainable_scope=action_head_and_adapter

acceptance:
  if E8 lowers action_default_equiv while E7 does not, the root cause is
  action-head/readout adaptation.  The production fix is suffix_cross_attention
  plus normal action-side trainability, not another slot/overlap loss.

  if E8 also fails, the context is still not action-informative enough and the
  next layer is full semantic backbone cotrain with suffix_cross_attention.
```

E8 step9150 first row:

```text
loss_action_default_equiv=0.047409
loss_total_minus_action=0.010263
active/downstream overlap=0.200000 / 0.188105
pi_context_adapter_gate=0.119179
pi_context_adapter_residual_rms_mean=0.270469
grad_norm_group_semantic_backbone=0.239949
```

Read:

```text
The action head/time MLP are trainable and receive much larger gradients than
E7 adapter-only, but the first 50-step row is still indistinguishable from E7.
Continue to the 9200 gate before rejection.  If 9200 remains flat, the causal
chain moves one level deeper: the frozen action expert/backbone is not able to
use the new suffix-side belief signal without semantic/action transformer
adaptation.
```

E8 step9200 confirmation:

```text
loss_action_default_equiv=0.047135
loss_total_minus_action=0.010086
loss_anchor_pv=0.499724
loss_anchor_object_pull=0.146214
loss_aqr_denoising=1.154690
loss_mapg_routing=0.422897
loss_slot_jepa=0.862604
active/downstream overlap=0.100000 / 0.097957
grad_norm_group_semantic_backbone=0.177523
```

Decision:

```text
Stop E8 after the 9200 gate.  Action head/time MLP gradients are live, but
local action-head adaptation alone does not separate action loss.  The
remaining root cause is now localized to the frozen PI action transformer /
semantic-action backbone being unable to exploit the new suffix-side PICF
belief path without cotrain.

Proceed to E9: production-style `backbone_only` semantic/action transformer
cotrain with `suffix_cross_attention`.
```

Prepared E9 fallback:

```text
experiment:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_9400_20260531

script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from9100_suffixadapter_backbone_300_20260531.sh

scope:
  semantic_trainable_scope=backbone_only
  action_context_integration=suffix_cross_attention
  picf_core_lr_scale=0.001

purpose:
  If E8 also fails, test the production-style boundary: train the restored
  PI0/PaliGemma action transformer/backbone plus suffix adapter, but keep
  wrapper-local action flow projections fixed.
```

E9 first launch failure and fix:

```text
failed run:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_9400_20260531

failure:
  ValueError: Action context adapter dimension mismatch:
  context=(1, 28, 2048) suffix=(1, 16, 1024)

root cause:
  In FSDP `backbone_only`, `action_context_in_proj` is nested-wrapped and is
  no longer an `nn.Linear` instance at runtime.  The adapter's projection bridge
  incorrectly used `isinstance(in_proj, nn.Linear)` as a contract check, so the
  production-style diagnostic crashed before the first training step.

repair:
  Treat the bridge as a generic `nn.Module`, unwrap `.module` /
  `._fsdp_wrapped_module` only for optional feature-size validation, then call
  the module and validate the projected output width.  This preserves the same
  mathematical map `context_2048 -> action_width_1024`; it only fixes wrapped
  module identity handling.

local validation:
  py_compile wrapper/policy/train passed
  wrapper_test: 40 passed
  policy_test: 16 passed
  latest-slot deployment audit: 16/16 PASS
  action-visible reserve gate audit: PASS
  verify_picf_owm_contract: PASS
  picf_owm_dataflow_trace: PASS
  git diff --check: PASS

rerun:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_retry1_9400_20260531
```

E9 retry1 result:

```text
status:
  projection bridge fix worked; run passed checkpoint resume, training config,
  optimizer group construction, and entered the first training step.

failure:
  rank0 SIGSEGV during the first production FSDP training step, before the
  first metrics row.

evidence:
  stack dump before termination showed rank1 inside autograd backward.
  Log reached optimizer groups:
    semantic_backbone lr=7e-5 num_params=1.737B
    picf_core lr=7e-8 num_params=116M
    policy_head lr=7e-5 num_params=109M

root cause update:
  `action_context_*` projections were listed as independent nested FSDP hot
  leaves.  These are small adapter/interface modules, not large transformer
  stacks.  Wrapping them as separate tiny FSDP leaves is unnecessary and caused
  a low-level full-shard/backward failure in the production backbone diagnostic.

repair:
  Remove `action_context_*` modules from `fsdp_runtime_leaf_module_specs`.
  They remain trainable and remain inside the semantic root FSDP boundary.
  The mathematical route is unchanged:
    context_2048 -> action_context_in_proj -> gated suffix cross-attention
  but the FSDP boundary is now stable: large transformer hot leaves are nested,
  small adapter projections stay in the root boundary.

validation:
  wrapper_test: 40 passed
  py_compile wrapper/policy/train: PASS
  git diff --check: PASS

rerun:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_retry2_9400_20260531
```

E9 retry2 9150 result and horizon correction:

```text
run:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_retry2_9400_20260531

status:
  FSDP/backbone path is now mechanically live.  It passed checkpoint resume,
  first backward, and wrote the 9150 metrics row.

9150:
  loss_action_default_equiv = 0.048359
  loss_total_minus_action   = 0.010331
  active/downstream overlap = 0.170000 / 0.171056
  pi_context_adapter_gate   = 0.119175
  adapter residual RMS      = 0.271960
  grad semantic_backbone    = 0.339463
  lr semantic_backbone      = 2.0088e-5
  lr picf_core              = 2.0088e-8

read:
  This is not a slot/overlap collapse: non-action budget is small and
  active/downstream overlap remains well below the structural stop line.
  It also is not an FSDP/dataflow crash anymore.

problem:
  The diagnostic itself used NUM_TRAIN_STEPS=9400 while resuming from step9100.
  That puts the run at the cosine tail (`lr ~= min_lr`) immediately after
  resume.  This violates the maintained step-indexed/fixed-window rule already
  used elsewhere: resumed diagnostics that compare action learning must keep
  the production 30K horizon, even if stopped early.

decision:
  Stop retry2 after the 9150 row.  Treat it as a mechanical/FSDP validation,
  not as a final rejection of backbone cotrain.  Patch the E9 launcher to use
  NUM_TRAIN_STEPS=30000 by default and rerun from the same step9100 checkpoint.

root-cause status:
  E7/E8 still reject another raw-overlap/slot-loss repair as the next move.
  The remaining live cause is action-interface adaptation under the proper
  production LR horizon.  The corrected E9-h30k run is the decisive next test.
```

E9-h30k 9150/9200 result:

```text
run:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_h30k_20260531

status:
  production horizon and FSDP dataflow are correct:
    num_steps=30000
    semantic_trainable_scope=backbone_only
    action_context_integration=suffix_cross_attention
    semantic_backbone lr ~= 5.94e-5
    picf_core lr ~= 5.94e-8

9150:
  action_default_equiv = 0.055658
  non_action           = 0.010807
  active/downstream    = 0.134979 / 0.136243
  slot_jepa            = 0.671574
  semantic grad        = 0.291876

9200:
  action_default_equiv = 0.055003
  non_action           = 0.010334
  active/downstream    = 0.089954 / 0.082378
  slot_jepa            = 0.584209
  semantic grad        = 0.249411
```

Read:

```text
This rejects another slot/overlap repair as the immediate root cause.  The
structure is healthier than E7/E8 while action is worse:

  E7 9200 adapter_only:          action ~= 0.047333
  E8 9200 action_head_adapter:   action ~= 0.047135
  E9-h30k 9200 backbone_only:    action ~= 0.055003

The h30k correction fixed the invalid cosine-tail LR horizon, but it also made
the large semantic/action transformer update much more aggressive
(`semantic_backbone lr ~= 5.93e-5`).  E6's stable prefix-fusion boundary used
`SEMANTIC_LR_SCALE=0.35`.  Therefore the next causal test is not a new module:
it is the same suffix adapter and same 30K horizon with a two-timescale
semantic LR boundary.
```

Next repair diagnostic:

```text
run:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_sem035_h30k_20260531

script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from9100_suffixadapter_backbone_sem035_h30k_20260531.sh

only intended change from E9-h30k:
  SEMANTIC_LR_SCALE: 1.0 -> 0.35

accept:
  action should recover toward the E7/E8 band while active/downstream overlap
  remains low and adapter residual stays nonzero.

reject:
  if action remains near 0.055 despite lower semantic LR, the suffix-side PICF
  context is still not action-useful enough and the next fix must compare
  prefix_fusion continuation versus suffix adapter under the same h30k horizon.
```

E10 9150/9200 result:

```text
run:
  picf_a7_stepindexed_from9100_suffixadapter_backbone_sem035_h30k_20260531

confirmed boundary:
  resume checkpoint       = prefixfusion step9100
  num_steps               = 30000
  action_context          = suffix_cross_attention
  semantic_trainable      = backbone_only
  semantic_lr_scale       = 0.35
  policy_head_lr_scale    = 1.0
  picf_core_lr_scale      = 0.001

9150:
  action_default_equiv = 0.048495
  non_action           = 0.011015
  active/downstream    = 0.184977 / 0.186723
  anchor_pv            = 0.503595
  object_pull          = 0.225764
  slot_jepa            = 0.729864
  recycle              = 0.093192
  adapter gate         = 0.119173
  adapter entropy      = 3.080438
  adapter residual rms = 0.273764
  semantic grad        = 0.339387
  semantic lr          = 2.079e-5

9200:
  action_default_equiv = 0.047530
  non_action           = 0.010215
  active/downstream    = 0.089956 / 0.086722
  anchor_pv            = 0.498808
  object_pull          = 0.150400
  slot_jepa            = 0.583056
  recycle              = 0.092822
  adapter gate         = 0.119129
  adapter entropy      = 2.953673
  adapter residual rms = 0.319925
  semantic grad        = 0.350327
  semantic lr          = 2.075e-5
```

Read:

```text
Lowering semantic LR fixes the E9-h30k failure mode:

  E9-h30k 9200 suffix/backbone/lr1.0:   action ~= 0.055003
  E10     9200 suffix/backbone/lr0.35:  action ~= 0.047530

But E10 only returns to the E7/E8 band and still does not recover the E6
prefix-fusion boundary:

  E7 9200 suffix adapter only:          action ~= 0.047333
  E8 9200 suffix action_head_adapter:   action ~= 0.047135
  E10 9200 suffix backbone lr0.35:      action ~= 0.047530
  E6 9100 prefix_fusion reference:      action ~= 0.044162

Therefore the current root cause is now narrowed:

  rejected:
    raw support overlap as the immediate cause
    slot_jepa as the immediate cause
    sidecar/mask quality as the immediate cause
    production-horizon cosine LR artifact
    high semantic LR as the only cause

  still live:
    suffix-side cross-attention may be a weaker action interface than
    fixed-length prefix_fusion for this PI0.5/PaliGemma action layout.

The next test must be a matched prefix_fusion h30k continuation from the same
step9100 checkpoint, keeping E10's LR boundary and changing only
`ACTION_CONTEXT_INTEGRATION`.
```

E11 matched control:

```text
run:
  picf_a7_stepindexed_from9100_prefixfusion_sem035_h30k_20260531

script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from9100_prefixfusion_sem035_h30k_20260531.sh

only intended change from E10:
  ACTION_CONTEXT_INTEGRATION: suffix_cross_attention -> prefix_fusion

accept:
  action returns toward the E6 reference band while non-action structure stays
  healthy.  Then the fix is to retire suffix-side action injection for this
  path and keep fixed-length prefix_fusion as the maintained action interface.

reject:
  action remains near E10 despite prefix_fusion.  Then the remaining cause is
  not the action context topology; it is the step9100 basin / train stream /
  optimizer state boundary, and the next experiment should compare from an
  earlier preserved checkpoint under the same prefix_fusion interface.
```

E11 launch check:

```text
remote:
  A7 / qgE72e

tmux:
  e11_prefixfusion_sem035_h30k_20260531

log:
  /mnt/picf_run_logs/
  picf_a7_stepindexed_from9100_prefixfusion_sem035_h30k_20260531.log

metrics:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_from9100_prefixfusion_sem035_h30k_20260531/metrics.jsonl

confirmed from startup log:
  resumed step              = 9100
  num_steps                 = 30000
  training_strategy         = fsdp_full_shard
  effective_global_batch    = 2
  unroll/burnin             = 2 / 1
  action_context            = prefix_fusion
  context_tokens            = 24
  context_gate              = 0.25
  semantic_trainable        = True
  semantic_trainable_scope  = backbone_only
  frozen backbones          = Sonata / V-JEPA / AnyTouch
  semantic_backbone lr      = 2.45e-5 at startup
  policy_head lr            = 7.00e-5 at startup
  picf_core lr              = 7.00e-8 at startup
  optimizer checkpoint      = model_only; optimizer state reinitialized
```

E11 9150/9200 result:

```text
run:
  picf_a7_stepindexed_from9100_prefixfusion_sem035_h30k_20260531

9150:
  action_default_equiv = 0.047375
  non_action           = 0.010323
  active/downstream    = 0.184988 / 0.176964
  anchor_pv            = 0.501754
  object_pull          = 0.159862
  slot_jepa            = 0.803113
  recycle              = 0.099084
  prefix context count = 24
  prefix fused rms     = 0.699991
  semantic grad        = 0.341518
  semantic lr          = 2.079e-5

9200:
  action_default_equiv = 0.047304
  non_action           = 0.010585
  active/downstream    = 0.114996 / 0.103346
  anchor_pv            = 0.500755
  object_pull          = 0.187458
  slot_jepa            = 0.777948
  recycle              = 0.095549
  prefix context count = 24
  prefix fused rms     = 0.699996
  semantic grad        = 0.394311
  semantic lr          = 2.075e-5
```

Read:

```text
E11 improves only marginally over E10:

  E10 9200 suffix/backbone/lr0.35:        action ~= 0.047530
  E11 9200 prefix_fusion/backbone/lr0.35: action ~= 0.047304

This rejects action-context topology as the sole root cause.  The structure
side is healthy enough for a fair test: active/downstream overlap is low,
slot_jepa is bounded, recycle is low, and gradients are finite.  The action
path still does not recover the E6 reference band:

  E6 9100 prefix_fusion reference:        action ~= 0.044162
```

Next decisive control:

```text
run:
  picf_a7_stepindexed_from9100_prefixonly_sem035_h30k_20260531

script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from9100_prefixonly_sem035_h30k_20260531.sh

only intended change from E11:
  ACTION_CONTEXT_TOKENS: 24 -> 0

why this is mathematically decisive:
  E10 changed semantic LR and kept dense suffix context.
  E11 changed context topology and kept dense context.
  E12 removes dense action context while keeping the same checkpoint, horizon,
  semantic LR, trainable boundary, and action prefix teacher.

accept:
  action clearly improves below the E10/E11/E7/E8 band and moves toward E6.
  Then dense action-context tokens are the remaining action-path noise source.

reject:
  action stays near 0.047.  Then dense context is not the root cause; the
  remaining live cause is the step9100 basin / train stream / optimizer
  boundary.  The next experiment should move to an earlier checkpoint or an
  action-only no-PICF-prefix control rather than further tuning context shape.
```

E12 launch check:

```text
remote:
  A7 / qgE72e

tmux:
  e12_prefixonly_sem035_h30k_20260531

log:
  /mnt/picf_run_logs/
  picf_a7_stepindexed_from9100_prefixonly_sem035_h30k_20260531.log

metrics:
  /mnt/checkpoints/picf_core/picf_core/
  picf_a7_stepindexed_from9100_prefixonly_sem035_h30k_20260531/metrics.jsonl

confirmed from command line:
  resumed step              = 9100
  num_steps                 = 30000
  training_strategy         = fsdp_full_shard
  effective_global_batch    = 2
  unroll/burnin             = 2 / 1
  action_context            = prefix_fusion
  context_tokens            = 0
  semantic_trainable        = True
  semantic_trainable_scope  = backbone_only
  semantic_lr_scale         = 0.35
  frozen backbones          = Sonata / V-JEPA / AnyTouch
  optimizer checkpoint      = model_only; optimizer state reinitialized

startup speed:
  first steps ~= 24-26 sec/step, faster than E11's dense-prefix 28-29 sec/step.
```

E12 9150 result:

```text
run:
  picf_a7_stepindexed_from9100_prefixonly_sem035_h30k_20260531

confirmed intervention:
  action_context_tokens = 0
  pi_context_token_count = 0

9150:
  action_default_equiv = 0.047398
  non_action           = 0.009776
  active/downstream    = 0.140000 / 0.140714
  anchor_pv            = 0.502832
  object_pull          = 0.111165
  slot_jepa            = 0.813201
  recycle              = 0.097886
  policy lr            = 5.941e-5
  semantic lr          = 2.079e-5
```

Read:

```text
Dense action context is not the root cause:

  E11 9150 dense prefix_fusion: action ~= 0.047375
  E12 9150 prefix-only:         action ~= 0.047398

The decisive mismatch is now visible in the old source metrics:

  source E6 9100 policy lr      ~= 2.006e-5
  source E6 9100 semantic lr    ~= 7.020e-6

  E10/E11/E12 9150 policy lr    ~= 5.941e-5
  E10/E11/E12 9150 semantic lr  ~= 2.079e-5

Therefore the h30k resume diagnostics were not LR-continuous with the source
checkpoint.  They reinterpreted a low-LR checkpoint as an early high-LR point on
a fresh 30K cosine schedule, producing immediate action-path drift while
structure metrics stayed healthy.
```

E13 repair:

```text
run:
  picf_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531

script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531.sh

only intended repair:
  preserve maintained prefix_fusion action interface
  restore LR continuity:
    LR=2.0e-5
    MIN_LR=2.0e-5
    SEMANTIC_LR_SCALE=0.35

accept:
  step9150 action returns toward the E6 source band around 0.044 while
  structure remains healthy.

reject:
  step9150 still stays near 0.047.  Then LR discontinuity is not sufficient and
  the next root-cause test must evaluate checkpoint-basin / stream mismatch
  with exact source-window replay.
```

E13 9150 partial result:

```text
9150:
  action_default_equiv = 0.046477
  non_action           = 0.010350
  active/downstream    = 0.200000 / 0.191951
  anchor_pv            = 0.501102
  object_pull          = 0.167834
  slot_jepa            = 0.821991
  recycle              = 0.100666
  policy lr            = 2.000e-5
  semantic lr          = 7.000e-6

Comparison:
  E10 9150 suffix/lr-discontinuous:       action ~= 0.048495
  E11 9150 prefix/lr-discontinuous:       action ~= 0.047375
  E12 9150 prefix-only/lr-discontinuous:  action ~= 0.047398
  E13 9150 prefix/LR-continuous:          action ~= 0.046477
  source E6 9100 prefix:                  action ~= 0.044162
```

Read:

```text
LR discontinuity is a real cause because restoring LR continuity improves
action by ~0.0009 versus E11/E12 and ~0.0020 versus E10.  It is not yet a full
explanation because the run has not recovered the E6 source band.

Continue to 9200:
  if action keeps descending, LR continuity is likely sufficient and should
  become the maintained resume rule.

  if action stays near 0.046+, the remaining root cause is not context shape or
  structure loss.  It is either missing optimizer state or a train-stream/window
  mismatch relative to the source E6 run.  The next test should be exact-window
  replay from the source metrics boundary, not another slot-structure change.
```

E13 9200 final read:

```text
9200:
  action_default_equiv = 0.046615
  non_action           = 0.010613
  active/downstream    = 0.095000 / 0.090000
  anchor_pv            = 0.498752
  object_pull          = 0.192729
  slot_jepa            = 0.701867
  recycle              = 0.099770
  policy lr            = 2.000e-5
  semantic lr          = 7.000e-6

Source E6 reference:
  9050 action_default_equiv = 0.043636
  9100 action_default_equiv = 0.044162

LR-discontinuous controls:
  E10 9150 suffix/backbone/sem0.35 = 0.048495
  E11 9150 prefix/backbone/sem0.35 = 0.047375
  E12 9150 prefix-only/sem0.35     = 0.047398

LR-continuous repair:
  E13 9150 prefix/LR-continuous    = 0.046477
  E13 9200 prefix/LR-continuous    = 0.046615
```

Conclusion:

```text
LR discontinuity is confirmed as a real but partial cause.  Matching the source
step9100 LR improves the resumed run versus E10/E11/E12, but it does not restore
the source 0.044 band.

The remaining gap is now constrained to two non-slot causes:

1. optimizer-state discontinuity:
   source ckpt /9100 contains only model.pt and metadata.pt; no optimizer.pt.
   _load_checkpoint() therefore reinitializes Adam moments.

2. metric-window non-equivalence:
   source run logged 9050/9100 only.  E13 logged 9150/9200 on later windows.
   The source run has no 9150/9200 rows, so source 9100 cannot be directly
   compared to E13 9150/9200 without exact-window replay.

Rejected root causes:
  dense action-context tokens, because E12 prefix-only stayed in the same band;
  suffix-vs-prefix topology, because E11 prefix only marginally helped;
  active anchor collapse, because active/downstream overlap is low at E13 9200;
  slot-JEPA explosion, because slot_jepa is bounded and falling.
```

Exact-window replay launched:

```text
window file:
  /mnt/picf_exact_window_probes/e6_step9100_rank01_windows.jsonl

records:
  100 exact windows from source E6 step9100 window_trace_rank0/1

probe:
  compare source /9100 and E13 /9200 on the same 100 windows

tmux:
  exact_probe_e6_e13_20260531

outputs:
  /mnt/picf_exact_window_probes/e6_vs_e13_9100_windows_20260531/
```

Decision rule:

```text
If source /9100 still evaluates near the source band and E13 /9200 is worse on
the same exact windows, resume training changed the weights unfavorably.  Since
LR continuity was already restored, optimizer-state reset becomes the primary
confirmed cause.

If source /9100 itself does not reproduce the source band under the exact probe,
the issue is not a resume update.  It is a train/eval-mode or probe-dataflow
contract mismatch that must be fixed before any new training claim.
```

Source /9100 exact-window result:

```text
probe windows:
  source E6 step9100 window_trace_rank0/1, 100 windows

source /9100 eval:
  loss_action_default_equiv mean = 0.041924
  std = 0.029489
  min/max = 0.001045 / 0.140138
  anchor_pv mean = 0.486638
  object_pull mean = 0.096845
  slot_jepa mean = 0.883698
  active/downstream overlap mean = 0.130000 / 0.122288
  recycle mean = 0.101549
```

Read:

```text
The source checkpoint reproduces the source training-log action band on its own
exact windows.  This eliminates the earlier fixed64 representativeness problem
for this comparison and makes the E13 /9200 same-window result decisive.
```

E13 /9200 same-window result:

```text
same 100 source E6 step9100 windows:

source /9100:
  action_default_equiv = 0.041924
  total                = 0.243286
  anchor_pv            = 0.486638
  object_pull          = 0.096845
  slot_jepa            = 0.883698
  active/downstream    = 0.130000 / 0.122288
  recycle              = 0.101549

E13 /9200:
  action_default_equiv = 0.042149
  total                = 0.245926
  anchor_pv            = 0.488568
  object_pull          = 0.107138
  slot_jepa            = 0.710429
  active/downstream    = 0.115000 / 0.118174
  recycle              = 0.100065

delta E13 - source:
  action_default_equiv = +0.000225  (+0.54%)
  total                = +0.002640  (+1.09%)
  anchor_pv            = +0.001930  (+0.40%)
  object_pull          = +0.010294  (+10.63%)
  slot_jepa            = -0.173269  (-19.61%)
  active overlap       = -0.015000
  downstream overlap   = -0.004115
  recycle              = -0.001484
```

Final read for this branch:

```text
The E13 checkpoint did not materially degrade the action solution on the same
windows.  The large-looking gap between source training step9100 (~0.044) and
E13 training step9200 (~0.0466) is mainly a window/difficulty comparison error,
not evidence that prefix_fusion, dense context, slot losses, or LR-continuous
resume destroyed the model.

Optimizer-state reset remains a resume hygiene problem and should be fixed for
future continuity, but it is not the primary explanation for the E13 9200 log
gap because E13 /9200 is nearly equal to source /9100 on identical windows.

The next root-cause work should therefore move from "architecture broke action"
to "training metric is non-stationary across sampled windows".  Future action
claims must report both:
  1. rolling train-window metrics;
  2. exact-window or stratified held-window probes.
```

Compatibility warning observed during the probe:

```text
Loaded PICF trainer checkpoint with compatibility migration.
missing_keys:
  semantic_encoder.encoder.action_context_* adapter parameters
```

Dataflow check:

```text
For prefix_fusion:
  policy._training_action_condition_tokens()
    -> _fuse_action_context_into_prefix()
    -> semantic_encoder.compute_action_flow_loss(extra_prefix_tokens=prefix)
    -> paligemma._apply_action_context_adapter(..., None)

Therefore the missing semantic action-side adapter modules are not in the
prefix_fusion forward path under this probe.  They would matter for
suffix_cross_attention, but not for the maintained E6/E11/E13 prefix_fusion
comparison.
```

Full optimizer checkpoint contract:

```text
script:
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_stepindexed_from9100_lrcontinuity_fullopt_prefixfusion_h30k_20260531.sh

purpose:
  future-proof the continuation boundary by writing optimizer.pt at saved
  checkpoints.

important limitation:
  this cannot recover the missing Adam moments from the old source /9100
  checkpoint.  It only prevents the next resume from silently becoming another
  fresh-optimizer experiment.

validation:
  bash -n passed
  git diff --check passed
  uv run pytest scripts/picf_core_train_test.py -q -k
    'optimizer_checkpoint or checkpoint_roundtrip or model_only_checkpoint or
     enabled_auto_checkpoint or should_save_optimizer'
  result:
    7 passed

default save policy:
  SAVE_INTERVAL=1000
  KEEP_LAST_CHECKPOINTS=5

Reason:
  full optimizer checkpoints are resume-critical but expensive; they should be
  saved at phase-scale boundaries by default, not every 100 steps like light
  model-only diagnostics.
```
