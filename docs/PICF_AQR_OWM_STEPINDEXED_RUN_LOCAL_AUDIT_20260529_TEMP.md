# PICF-AQR-OWM Step-Indexed Corrected Run Local Audit

Date: 2026-05-29

Run under observation:

```text
picf_a7_stepindexed_actionprefix_ema_from7000_30k_20260529
```

This note is the local follow-through audit for the corrected A7 run.  It is
not a new architecture proposal.  Its purpose is to decide whether the current
training should continue, stop, or restart from a clean checkpoint.

## Current Decision

```text
Decision at step7200: continue.

Reason:
  the corrected run has not crossed the old 7550/7600 failure band yet;
  step7200 action loss improved relative to 7050/7100;
  active/downstream overlap is low;
  non-action budget is below the stop line;
  posterior lifecycle metrics are not jointly exploding.

Do not stop on raw reserve overlap alone.
```

## Mathematical Stop Gates

Let

```math
L_a(k) = \text{loss_action_default_equiv at logged step } k
```

and let

```math
m(k) = \min_{j \le k} L_a(j).
```

A corrected-run action rebound is only actionable if the rebound happens after
the corrected local minimum:

```math
L_a(k) > 0.035
\quad\text{and}\quad
L_a(k) > 1.3\,m(k)
```

for at least two consecutive logged points after `m(k)`.  Older points before
the local minimum must not be counted as sustained rebound evidence.

Structural stop gates:

```text
non-finite loss or grad
loss_total_minus_action > 0.02
aqr_active_same_role_support_overlap_max > 0.25
aqr_downstream_same_role_support_overlap_max > 0.25
action rebound plus posterior_identity_switch_rate and posterior_recycle_rate rising together
```

## Latest Observed Corrected Metrics

```text
step7050:
  action_default_equiv 0.050193
  non_action_budget    0.010105
  active/downstream    0.043996 / 0.057294

step7100:
  action_default_equiv 0.051401
  non_action_budget    0.010558
  active/downstream    0.069976 / 0.077965

step7150:
  action_default_equiv 0.042441
  non_action_budget    0.011460
  active/downstream    0.109996 / 0.110029

step7200:
  action_default_equiv 0.037938
  non_action_budget    0.010352
  active/downstream    0.080000 / 0.077714
  verdict              continue

step7250:
  action_default_equiv 0.050864
  non_action_budget    0.010823
  active/downstream    0.084968 / 0.093113
  posterior lifecycle  identity_switch=0.180000 recycle=0.130084
  verdict              continue

step7300:
  action_default_equiv 0.043123
  non_action_budget    0.011162
  active/downstream    0.080000 / 0.084150
  posterior lifecycle  identity_switch=0.177778 recycle=0.130421
  verdict              continue
```

The old near-zero-LR replay failed around 7550/7600, so `7200` is a necessary
but insufficient pass.

Step7250 was the first logged point above the step7200 local minimum.  Step7300
fell back from 0.050864 to 0.043123 with healthy active/downstream structure, so
the first 7250 increase is not sustained rebound.  Continue to the historical
suspect band at 7550/7600.

Remote source/runtime sanity:

```text
remote repo:
  /root/openpi_action_interface_ema_20260527

remote source contains:
  def _step_indexed_window_rng
  args.step_indexed_window_rng default True
  sample_rng = _step_indexed_window_rng(...)
  source.window(flat_index, rng=sample_rng)

remote process command contains:
  --visual-mode encoder
  --tactile-mode encoder
  --semantic-mode paligemma
  --semantic-trainable
  --semantic-trainable-scope backbone_only
  --unroll-steps 2
  --burnin-steps 1
  --burnin-mode state_only
  --picf-core-lr-scale 0.005
  --lambda-action-pos/rot/gripper 2.0
  --save-interval 1000
  --keep-last-checkpoints 5
```

Therefore the active run is not accidentally using the stub visual/tactile
path, and it is not the old legacy sampled-window path.

## File-Level Audit

## Dataflow Follow-Through

The corrected training path is:

```text
global step k
  -> _step_indexed_window_rng(seed, rank, k, micro_step, retry)
  -> flat_index = sample_rng.integers(0, len(source))
  -> source.window(flat_index, rng=sample_rng)
  -> PicfObservation sequence
  -> optional sidecar tracklet/proposal fields if present
  -> recurrent burn-in using the same AQR measurement graph
  -> suffix forward with PaliGemma action-flow target
  -> PicfCoreOutput.debug
  -> PicfTransitionLossOutput
  -> loss_total / loss_action_default_equiv / loss_total_minus_action
  -> metrics.jsonl and gate monitor
  -> checkpoint retention, numeric step dirs only
```

The key corrected invariant is:

```math
\operatorname{window}(k)
\perp
\operatorname{resume\_path}
\mid
(\text{seed}, \text{rank}, k, \text{micro}, \text{retry})
```

so step `k` is comparable across a fresh run and a resumed run.

### `scripts/picf_core_train.py`

Risk checked:

```text
sampled-window RNG after checkpoint resume
optimizer/checkpoint load semantics
loss weight schedules
trainable scopes
checkpoint retention
metrics logging
anchor overlay side effects
```

Current status:

```text
step-indexed RNG is present and defaults on;
the sampled window uses the same per-step RNG for flat-index selection and
source.window(...);
checkpoint retention is numeric-step-only and does not delete diagnostics;
metrics include loss_total_minus_action and active/downstream overlap.
```

Why this matters:

```math
\text{legacy RNG} = f(\text{seed}, \text{rank})
```

made resumed step7000 consume an early stream.  The corrected sampler is:

```math
\text{rng}_k = f(\text{seed}, \text{rank}, k, \text{micro}, \text{retry})
```

so checkpoint continuation no longer replays early windows.

### `src/openpi/picf/core/pipeline.py`

Risk checked:

```text
active/context/reserve routing
posterior file competition
cache read/write order
binding support/address logic
object candidate assignment
temporal visual routing
prefix debug metrics
```

Current status:

```text
active/downstream/reserve overlap metrics are split;
reserve rows are not treated as action-visible failure;
cache read is previous-only and residual-scaled;
cache write is after posterior correction;
posterior file competition and occupancy/seed coverage are audited;
prefix norm/gate/debug metrics exist.
```

No new behavior change was made in this audit except removing unused local
variables that had no consumers.

### `src/openpi/picf/core/training.py`

Risk checked:

```text
action/non-action loss split
slot-JEPA detach and matching
binding consistency permutation tolerance
object/sidecar weak scaffold losses
PV/cycle consistency losses
```

Current status:

```text
loss_total_minus_action is logged and usable as the non-action budget gate;
high-risk predictive losses remain guarded;
sidecar/object scaffold remains weak evidence, not a hard mask label.
```

### `src/openpi/picf/policy.py`

Risk checked:

```text
PICF -> PI0.5 prefix interface
action prefix stopgrad/norm/teacher mode
inference safety guards
```

Current status:

```text
training prefix path remains explicit;
inference stabilization is separate from training;
policy tests pass.
```

### `src/openpi/picf/paligemma/wrapper.py`

Risk checked:

```text
PaliGemma action pathway compatibility
semantic trainable scope
unused import drift
```

Current status:

```text
unused CONFIG_MAPPING import removed;
wrapper tests pass.
```

### `src/openpi/picf/vjepa/wrapper.py`

Risk checked:

```text
recent temporal maps preserve time;
multiview wrist/static temporal evidence stays typed;
```

Current status:

```text
V-JEPA wrapper tests pass;
MVTrack audit confirms wrist does not inherit static projective geometry.
```

### `scripts/verify_picf_owm_contract.py`

Risk checked:

```text
contract-level regression of the step-indexed RNG and runtime-c invariants
```

Current status:

```text
trainer_window_rng_is_resume_safe exists and passes.
```

### `scripts/picf_*_audit.py`

Risk checked:

```text
static/dataflow/math audits for active reserve gating, binding calibration,
posterior file competition, posterior binding memory, object candidate routing,
OEML dataflow, and MVTrack invariants.
```

Current status:

```text
all executed audits pass in local environment.
```

## Local Test Results

```text
python -m compileall -q scripts src/openpi/picf
  PASS

uv run ruff check scripts/picf_core_train.py scripts/verify_picf_owm_contract.py \
  scripts/picf_core_train_test.py src/openpi/picf/core/pipeline.py \
  src/openpi/picf/core/training.py src/openpi/picf/policy.py \
  src/openpi/picf/paligemma/wrapper.py src/openpi/picf/vjepa/wrapper.py --select F,E9
  PASS

uv run ruff check over all changed/untracked Python files --select F,E9
  PASS after removing an unused serve_picf_policy json import.

git diff --check
  PASS

python scripts/verify_picf_owm_contract.py
  PASS

python scripts/picf_owm_strict_diagnose.py --fail-on-fail
  PASS, runtime-artifact availability warnings only

python scripts/picf_owm_dataflow_trace.py --fail-on-fail
  PASS

python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
  PASS

python -m pytest scripts/picf_core_train_test.py \
  -k "step_indexed_window_rng or normalize_train_args_enables_step_indexed_window_rng or checkpoint or optimizer" -q
  30 passed, 106 deselected

python -m pytest src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  src/openpi/picf/policy_test.py \
  src/openpi/picf/paligemma/wrapper_test.py \
  src/openpi/picf/vjepa/wrapper_test.py \
  scripts/picf_core_train_test.py -q
  334 passed, 28 warnings

bash -n over all 27 changed/untracked experiment shell scripts
  PASS
```

Warnings are third-party deprecation/FSDP single-rank warnings; no PICF failure.

Changed-file scope checked in this pass:

```text
tracked modified files: 29
changed/untracked Python files: 22
changed/untracked shell scripts: 27
changed/untracked Markdown files: 27
```

Coverage mapping:

```text
Python syntax:
  compileall over scripts and src/openpi/picf

Python fatal lint:
  ruff F/E9 over all changed/untracked Python files

Shell syntax:
  bash -n over all changed/untracked experiment shell scripts

PICF runtime semantics:
  full core/policy/wrapper/trainer pytest set
  contract verifier
  strict diagnose
  dataflow trace
  MVTrack deep audit
  professor-grade audit
  action-visible reserve gate audit
  binding dataflow/math audit
  binding calibration audit
  posterior file competition audit
  object candidate slot binding audit
```

The two newly cleaned code details in this pass were unused imports/locals only:

```text
scripts/serve_picf_policy.py:
  removed unused json import

scripts/picf_core_train.py, src/openpi/picf/core/pipeline.py,
src/openpi/picf/paligemma/wrapper.py:
  removed unused exception/local/import bindings found by ruff F/E9
```

## Historical Hypotheses Not To Repeat Blindly

```text
raw same-role overlap alone:
  repeatedly shown to be a reserve/context telemetry issue unless active or
  downstream overlap also rises.

optimizer restart alone:
  can change optimization trajectory, but near-zero-LR replay reproduced the
  old spike, so optimizer alone is not the root cause.

stronger sidecar/object scaffold alone:
  may help bootstrap ownership, but old action rebound survived cases where
  structure looked healthy; do not add more scaffold before corrected 7550/7600.

freezing PICF/policy-only alone:
  useful causal probe, but not the current first move while corrected sampling
  validation is incomplete.
```

## Next Required Observation

```text
Wait for corrected 7550 and 7600.

If pass:
  mark legacy replay-window root cause closed and continue to 8000.

If fail with active/downstream healthy:
  run fixed held-out action probe before changing architecture.

If fail with active/downstream or non-action budget unhealthy:
  restart from latest healthy checkpoint and inspect the structural path.
```

## File Cleanup Status

Remote `/mnt` cleanup was audited as dry-run only:

```text
/mnt free: about 243G
protected names:
  7000, 8000, stepindexed, actionprefix_ema_from6800, 4-22/ablation,
  full_picf, baseline, current/latest

large safe deletion candidates >= 0.5G:
  none found under the checked top-level PICF checkpoint/log/eval/overlay roots.
```

No deletion was performed.  Preserving the corrected run and the step7000
checkpoint is more important than reclaiming a small amount of uncertain space.

## 2026-05-29 Follow-Up Local Audit After Step7300

This pass was run while the corrected A7 job continued.  It was intended to
catch the class of "small but fatal" issues that can make a remote loss trace
look like a training phenomenon while the actual cause is a broken local
contract.

### Current Remote Status At Audit Start

```text
run:
  picf_a7_stepindexed_actionprefix_ema_from7000_30k_20260529

latest structured metric:
  step7300

progress-bar step:
  step7327

verdict:
  continue
```

The key loss trace remains:

```text
step7050 action_default_equiv = 0.050193
step7100 action_default_equiv = 0.051401
step7150 action_default_equiv = 0.042441
step7200 action_default_equiv = 0.037938
step7250 action_default_equiv = 0.050864
step7300 action_default_equiv = 0.043123
```

Step7250 was a high point, but step7300 fell back.  Under the current stop
contract this is not enough to stop, because the suspected old failure was a
sustained two-point rebound after the corrected local minimum.

Step7350 follow-up:

```text
loss_action_default_equiv: 0.039770
loss_total_minus_action:   0.010753
active/downstream overlap: 0.085000 / 0.083510
posterior switch/recycle:  0.183889 / 0.127943
verdict:                  continue
```

This further weakens the "immediate structural rebound" interpretation.  It
does not yet close the old 7550/7600 issue, because the old spike was localized
later.

Step7400 follow-up:

```text
loss_action_default_equiv: 0.045796
loss_total_minus_action:   0.010555
active/downstream overlap: 0.064948 / 0.068091
posterior switch/recycle:  0.177778 / 0.130156
verdict:                  continue
```

The scalar action loss is higher than 7350 but the structural gates improved.
This is exactly why the stop rule requires sustained action degradation and/or
structural corroboration rather than a single train-window scalar.

### Additional Local Checks

```text
python -m compileall -q scripts src/openpi/picf
  PASS

git diff --check
  PASS

uv run ruff check ... --select F,E9
  PASS

python scripts/verify_picf_owm_contract.py
python scripts/picf_owm_strict_diagnose.py --fail-on-fail
python scripts/picf_owm_dataflow_trace.py --fail-on-fail
  PASS

python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
  PASS

python scripts/picf_owm_professor_grade_audit.py
  PASS 16/16

python scripts/picf_action_visible_reserve_gate_audit.py
python scripts/picf_binding_dataflow_math_audit.py
python scripts/picf_binding_logit_calibration_audit.py
python scripts/picf_posterior_file_competition_audit.py
python scripts/picf_object_candidate_slot_binding_audit.py
  PASS

python -m pytest scripts/picf_core_train_test.py \
  -k "step_indexed_window_rng or normalize_train_args_enables_step_indexed_window_rng or checkpoint or optimizer or keep_last" -q
  30 passed

python -m pytest \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  src/openpi/picf/policy_test.py \
  src/openpi/picf/paligemma/wrapper_test.py \
  src/openpi/picf/vjepa/wrapper_test.py \
  scripts/picf_core_train_test.py -q
  334 passed
```

### File-Level Follow-Through

The pass specifically checked the high-risk files touched by the resume-safe
training change:

```text
scripts/picf_core_train.py:
  _normalize_train_args defaults step_indexed_window_rng to true.
  _step_indexed_window_rng keys sampling by seed, rank, global step,
  micro step, and retry count.
  training loop passes sample_rng into source.window(...).
  metrics record loss_action_default_equiv, loss_total_minus_action,
  active/downstream overlap, grad, and runtime LR multipliers.
  checkpoint pruning only removes numeric step directories inside the current
  output_dir.

src/openpi/picf/core/training.py:
  anchor_object_pull target core and quality gate remain weak-measurement
  scaffold, not hard masks.
  row weights are detached before pull loss so the model cannot evade the
  target by lowering active/object gates.
  action_default_equiv remains the comparable action metric.

src/openpi/picf/core/pipeline.py:
  action-prefix normalization is a bounded adapter stabilization step.
  inactive/reserve context remains gated out of action-visible object prefix
  paths while context memory is not deleted.
```

Two `NotImplementedError` sites were reviewed and are intentional hard stops
for unsupported synchronized multimodal geometry augmentation.  They are not
on the current remote run path.

### Decision

```text
No local code-level stop reason was found.

Continue the A7 job through 7550/7600.
If corrected 7550/7600 pass, continue to 8000 before release.
```

## 2026-05-29 Re-Run Audit Pass

The audit was repeated during live monitoring of
`picf_a7_stepindexed_actionprefix_ema_from7000_30k_20260529` to check that no
local-only or stale-document assumption was being reused.

### Executed Checks

```text
python -m compileall -q scripts src/openpi/picf
python scripts/verify_picf_owm_contract.py
python scripts/picf_owm_strict_diagnose.py --fail-on-fail
python scripts/picf_owm_dataflow_trace.py --fail-on-fail
python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
  PASS

python scripts/picf_owm_professor_grade_audit.py
  PASS 16/16

python scripts/picf_action_visible_reserve_gate_audit.py
python scripts/picf_binding_dataflow_math_audit.py
python scripts/picf_binding_logit_calibration_audit.py
python scripts/picf_posterior_file_competition_audit.py
python scripts/picf_object_candidate_slot_binding_audit.py
  PASS

python -m pytest scripts/picf_core_train_test.py \
  -k "step_indexed_window_rng or normalize_train_args_enables_step_indexed_window_rng or checkpoint or optimizer or keep_last" -q
  30 passed

python -m pytest \
  src/openpi/picf/core/pipeline_test.py \
  src/openpi/picf/core/training_test.py \
  src/openpi/picf/policy_test.py \
  src/openpi/picf/paligemma/wrapper_test.py \
  src/openpi/picf/vjepa/wrapper_test.py \
  scripts/picf_core_train_test.py -q
  334 passed
```

### Re-Checked Dataflow Claims

```text
Resume-safe sampled-window dataflow:
  train step -> _step_indexed_window_rng(seed, rank, global_step, micro_step, retry)
  -> source.window(flat_index, rng=sample_rng)
  -> metrics.jsonl step-indexed action comparison

Action-visible object context:
  active object/context/reserve split remains enforced;
  reserve rows do not enter action-visible object prefix;
  background/global context is retained, not deleted.

Binding and posterior competition:
  support-weighted binding signatures reach observation anchors;
  calibrated pairwise binding is relative, not common-mode cosine;
  duplicate posterior files are demoted to no-object mass;
  object candidates and sidecar masks stay weak evidence, not hard labels.

Checkpoint safety:
  save/keep pruning is numeric-step-only inside the current output_dir;
  protected historical checkpoint directories are not touched by trainer pruning.
```

### Manual Bug Scan Notes

```text
source.window(warmup_index):
  Reviewed at scripts/picf_core_train.py:_materialize_model_parameters.
  This call is no-grad lazy-module materialization before distributed wrapping.
  It does not participate in optimizer steps and does not define the sampled
  train-window stream.  The real training loop uses source.window(..., rng=sample_rng).

shutil.rmtree sites:
  diagnostics/<step> cleanup is limited to regenerated visual diagnostics.
  output_dir overwrite requires explicit --overwrite and rejects --resume.
  checkpoint temp/final replacement only affects the current step directory.
  keep_last pruning scans only pure numeric subdirectories inside the current
  output_dir.

NotImplementedError sites:
  multimodal_geometry augmentation remains an intentional hard stop for an
  unsupported synchronized augmentation mode and is not on the current run path.

Randomness:
  global random/np/torch seeding remains process-level setup.
  sampled windows are now keyed by seed/rank/global_step/micro_step/retry.
  retry_count changes the window only after a failed sampled-window validation.
```

### Remote Cleanup Dry Run

```text
df -h /mnt:
  /mnt has about 243G available during this audit.

du -sh /mnt/checkpoints/picf_core/picf_core/*:
  The large protected historical checkpoint roots are not selected.
  The visible 202605 top-level candidates inspected in this pass report 0-size
  placeholder/log directories or are current/protected run roots.

Action:
  No deletion performed in this pass.
  Deleting 0-size placeholders would not recover useful space and increases
  traceability risk.  Keep cleanup conservative until a later explicit manifest
  identifies large non-protected directories.
```

### Live Remote State During This Audit

```text
Latest inspected progress: 7442
Latest structured metric: 7400

step7400:
  loss_action_default_equiv: 0.0457963534
  loss_total_minus_action: 0.0105552664
  active/downstream overlap: 0.064948 / 0.068091
  posterior_identity_switch_rate: 0.177778
  posterior_recycle_rate: 0.130156
  verdict: CONTINUE
```

Step7450 live update:

```text
loss_action_default_equiv: 0.0459649861
loss_total_minus_action: 0.0110241296
active/downstream overlap: 0.134999 / 0.128150
posterior_identity_switch_rate: 0.176111
posterior_recycle_rate: 0.117888
verdict: CONTINUE
```

Step7500 live update:

```text
loss_action_default_equiv: 0.0423136577
loss_total_minus_action: 0.0118976496
active/downstream overlap: 0.164993 / 0.174715
posterior_identity_switch_rate: 0.199444
posterior_recycle_rate: 0.105055
verdict: CONTINUE
```

Step7550 live update:

```text
loss_action_default_equiv: 0.0445233509
loss_total_minus_action: 0.0112881213
active/downstream overlap: 0.084996 / 0.089842
posterior_identity_switch_rate: 0.203889
posterior_recycle_rate: 0.096984
verdict: CONTINUE
```

### Current Decision

```text
The corrected run was normal structurally through step7550, but it was stopped
as a production candidate after user review because action stayed in the
0.04-range instead of the expected 0.02-range.

This is not evidence that raw overlap / slot structure is again the direct
root cause:
  loss_total_minus_action stayed near 0.011;
  active/downstream overlap fell back below 0.10 at step7550;
  recycle decreased instead of rising with action.

It is evidence that live sampled train loss is the wrong final acceptance
metric.  The next required gate is the fixed-window no-update action probe:
  scripts/picf_fixed_window_action_probe.py

The probe compares preserved checkpoints on identical accepted flat indices.
Until that probe is run, do not claim either:
  "current model is worse than old 0.02";
  or "0.04 is only a hard-window artifact".
```
