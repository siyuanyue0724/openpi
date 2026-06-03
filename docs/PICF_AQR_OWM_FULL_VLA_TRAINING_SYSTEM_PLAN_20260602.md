# PICF-AQR-OWM Full VLA Training-System Plan

Status legend:

- `[ ]` not done
- `[~]` in progress
- `[x]` done and checked
- `[!]` blocked / intentionally deferred

This document tracks the full training-system repair requested after the
logical-batch smoke test.  The previous logical-batch plan only proved the
lowest-level estimator and K=2 production safety; it did not implement the
complete VLA Foundry / ABot-M0 / PiKE-style data mixing and normalization
system.

## Non-Negotiable Principle

The model is intended to scale to large heterogeneous robot data.  Therefore
the fix must not specialize to a tiny CALVIN subset or use brittle hand-picked
windows.  The production target is:

```text
multi-task / multi-dataset objective
  -> controlled sampling mixture
  -> per-task/per-dataset normalized estimator
  -> memory-safe logical optimizer step
  -> action-dominant but structure-aware cotrain
```

Do not claim convergence is solved until a K>2 or equivalent task-covered
optimizer step runs in production and shows action descent without structural
regression.

## Paper-Derived Requirements

The following requirements are based on the recent VLA / multi-task training
literature and should be implemented only where compatible with PICF's
belief-router design.

- VLA Foundry: supports dataset sources and ratios at dataloader time for
  dataset mixing and batch balancing across modalities.  PICF equivalent:
  explicit bucket/dataset/embodiment mixture controls, not uncontrolled global
  random sampling.

- ABot-M0: emphasizes data cleaning, standardization, and balanced sampling for
  heterogeneous robot data.  PICF equivalent: task-family balanced sampling on
  CALVIN now; dataset/embodiment hooks for future large data.

- PiKE: adaptive data mixing can be useful when task gradients are mostly
  compatible; do not add expensive gradient surgery until gradient conflict is
  measured.  PICF equivalent: static task-uniform / temperature mixture first,
  then gradient-cosine probe, then dynamic mixing if needed.

- OpenVLA-OFT / modern action-head recipes: action chunking, continuous action,
  and L1/Huber-style losses are strong baselines.  PICF already uses the PI0.5
  action path and comparable action metrics; do not rewrite the action head in
  this gate unless sampler/normalization experiments fail.

Primary references used for this gate:

```text
VLA Foundry technical report:
  dataset weighting / batch balancing ratios are first-class training inputs.

ABot-M0:
  large heterogeneous robot data requires cleaning, standardization, and
  balanced sampling over task / embodiment / dataset structure.

PiKE:
  adaptive mixing is designed for low-conflict positive-gradient regimes, so
  gradient-cosine evidence must come before dynamic mixing.

OpenVLA-OFT:
  action chunking + continuous action + L1-style objective is a strong VLA
  fine-tuning recipe; PICF should not rewrite this path before testing data
  estimator and boundary issues.

Knowledge Insulation:
  continuous action-expert gradients can harm VLM backbone transfer; PICF must
  keep action/backbone gradient boundaries explicit and measurable.
```

## Current Code Reality

- [x] Existing bucket metadata.
  `PicfCalvinWindowSource` builds coarse CALVIN task buckets from language
  prompts and stores `bucket_to_slot_indices`.

- [x] Existing deterministic bucket sampler.
  `balanced_bucket_slot_index()` currently does round-robin over sorted bucket
  names across `step/rank/micro_step`.

- [x] Existing logical-batch loss scaling.
  `_logical_batch_loss_scales()` implements:

  ```text
  scale_local = world_size * q_b / n_b
  ```

  with uniform `q_b` over selected buckets.

- [x] Full CALVIN task-bucket sampler family implemented.
  `scripts/picf_core_train.py` now supports `round_robin`,
  `task_uniform`, `trajectory`, `temperature`, and strict explicit bucket
  ratios through `--calvin-bucket-weight-spec`.

- [x] Target-distribution logical-batch loss scaling implemented.
  `_logical_batch_loss_scales()` now receives the active target `q_b` and
  records selected bucket counts plus target weights in `metrics.jsonl`.

- [!] Missing explicit per-dataset / embodiment normalization.
  CALVIN is currently one dataset / one embodiment, so this is a future-data
  hook rather than an immediate CALVIN blocker.

- [!] K>2 production proof blocked.
  Full model K=4 on 2xA100-40GB OOMs.  K=2 is safe but does not prove the
  normalization benefit.

## Experiment Checklist: 1-2 Hour Gate

The gate must test all previously proposed ideas at the level possible within
the resource limit.

Hard boundary:

```text
This gate can prove:
  sampler distributions are correct;
  target-aware per-bucket loss normalization is correct;
  K2 production path is runnable and measured;
  K4 estimator path is either runnable or resource-blocked.

This gate cannot prove:
  final 30k-step convergence;
  dynamic PiKE/PCGrad benefit;
  action-head architecture superiority;
  large heterogeneous dataset scaling beyond CALVIN.
```

If the requirement is "prove within 1 hour that the dual-card run will
converge normally", that is not scientifically available from a 50-100 step
smoke.  The honest 1-2 hour result is a deployment/readiness gate, not a final
learning guarantee.

- [x] F1 sampler distribution audit.
  Implement and run a no-model audit over 1k-10k sampled windows for:
  `round_robin`, `task_uniform`, `trajectory`, and `temperature`.
  Pass criteria:
  emitted bucket histograms match the intended mixture; no empty bucket; no
  deterministic fixed-window artifact.

  Remote result on `/root/openpi_e21b2_1b07eab` with real CALVIN sidecar
  segments, `world_size=2`, `accum_steps=1`, `steps=1024`
  (`2048` global micro samples):

  ```text
  round_robin:  max_abs=0.0003, KL=0.000001
  task_uniform: max_abs=0.0095, KL=0.000952
  trajectory:   max_abs=0.0155, KL=0.001640
  temperature alpha=0.5: max_abs=0.0152, KL=0.002135
  ```

  Audit artifacts:

  ```text
  /mnt/picf_run_logs/picf_bucket_sampler_audit_round_robin_20260602.json
  /mnt/picf_run_logs/picf_bucket_sampler_audit_task_uniform_20260602.json
  /mnt/picf_run_logs/picf_bucket_sampler_audit_trajectory_20260602.json
  /mnt/picf_run_logs/picf_bucket_sampler_audit_temperature_20260602.json
  ```

- [x] F2 memory-safe K>2 feasibility probe.
  Try production K=4 with the least intrusive memory settings:
  window activation checkpointing, progress on, overlays off for the probe,
  and otherwise the same trainable scope.  If it still OOMs, mark K=4 full as
  resource-blocked and do not claim convergence proof.

  Remote result:

  ```text
  EXP=picf_f2_k4_taskuniform_11000_to11005_20260602
  world_size=2
  accum_steps=2
  logical_batch_task_count=4
  calvin_bucket_sampling_mode=task_uniform
  logical_batch_bucket_normalization=on
  window_activation_checkpointing=on
  anchor_overlay_interval=0
  ```

  Status: resource-blocked on 2xA100-40GB.  The run entered the first resumed
  optimizer step and sampled two buckets locally:

  ```text
  micro1: sampled_bucket=block_push
  micro2: sampled_bucket=block_lift
  logical_loss_scale=0.5 / 0.5
  ```

  It then OOMed in `token_fusion` attention backward/recompute with both ranks
  at roughly the full 40GB GPU limit.  This is a memory/resource limit, not a
  sampler or loss-scale logic failure.

- [x] F3 reduced-scope K>2 estimator probe.
  If full K=4 OOMs, run K=4 with frozen heavy trainable components only to
  prove sampler + estimator + metrics over at least 50 optimizer steps.
  This is not a final training-quality proof; it only proves K>2 infrastructure.

  Remote execution notes:

  ```text
  invalid attempt 1:
    training_strategy=fsdp-full-shard
    result: CLI rejected; valid spelling is fsdp_full_shard.

  invalid attempt 2:
    used the slot-comprehensive smoke launcher default SEGMENTS=0,1,2,3
    result: only 4 segments / 3 buckets loaded; not a valid full-bucket proof.

  valid attempt 3:
    EXP=picf_f3_k4_reduced_taskuniform_fullbuckets_11000_to11060_20260602
    launcher=run_a7_e23_bucketbalanced_noactioncond_from11000_30k_20260601.sh
    world_size=2
    accum_steps=2
    logical_batch_task_count=4
    semantic_trainable=0
    action_loss_weight=0
    policy_head_lr_scale=1e-6
    bucket_names=all 7 CALVIN buckets
    status=completed to step 11060
  ```

  Required code repair discovered during this gate:

  ```text
  Bug:
    --optimizer-checkpoint-mode=model-only was parsed and logged, but
    _load_checkpoint() still loaded optimizer.pt whenever the file existed.

  Failure:
    changing the trainable set for reduced-scope probes caused FSDP optimizer
    rekey failure on PaliGemma vision-tower parameters.

  Fix:
    _load_checkpoint() and _load_checkpoint_sequential_across_ranks() now
    receive optimizer_checkpoint_mode and skip optimizer state/payload when
    mode=model_only.
  ```

  Current hard result:

  ```text
  step 11010:
    logical_batch_global_micro_count=4
    logical_batch_distinct_bucket_count=4
    loss_total=0.00904
    loss_total_minus_action=0.00904
    loss_action_weight_scale=0
    loss_anchor_object_pull=0.0511
    active/downstream overlap=0.188 / 0.159
    grad_norm=0.00245

  step 11060:
    logical_batch_global_micro_count=4
    logical_batch_distinct_bucket_count=3
    loss_total=0.00908
    loss_total_minus_action=0.00908
    loss_action_weight_scale=0
    loss_anchor_object_pull=0.0510
    active/downstream overlap=0.150 / 0.125
    grad_norm=0.00238
  ```

  Interpretation: K4 task-balanced target-normalized estimator is runnable on
  2xA100-40GB in reduced-scope mode.  This proves the infrastructure and
  estimator path beyond K2.  It still does not prove full-model K4 production
  convergence because full-model K4 OOMs.

- [x] F4 K2 production behavior sanity.
  Run 50-100 steps with `task_uniform` and compare against deterministic
  round-robin K2.  Expected:
  similar mean action loss scale, higher stochasticity, no code/path failure.
  This does not prove batch repair, but catches sampler bugs.

  Remote completed:

  ```text
  EXP=picf_f4_k2_taskuniform_11000_to11050_20260602
  world_size=2
  accum_steps=1
  logical_batch_task_count=2
  calvin_bucket_sampling_mode=task_uniform
  logical_batch_bucket_normalization=on
  logical_batch_log_bucket_metrics=on
  ```

  Hard result through step 11050:

  ```text
  step 11010:
    loss_action_default_equiv=0.0361
    loss_total_minus_action=0.0131
    active/downstream overlap=0.250 / 0.234
    raw overlap=1.000 reserve-only diagnostic

  step 11020:
    loss_action_default_equiv=0.0447
    loss_total_minus_action=0.00926
    active/downstream overlap=0.125 / 0.075
    raw overlap=1.000 reserve-only diagnostic

  step 11030:
    loss_action_default_equiv=0.0537
    loss_total_minus_action=0.00910
    active/downstream overlap=0.100 / 0.100
    raw overlap=1.000 reserve-only diagnostic

  step 11040:
    loss_action_default_equiv=0.0519
    loss_total_minus_action=0.00941
    active/downstream overlap=0.200 / 0.200
    raw overlap=1.000 reserve-only diagnostic

  step 11050:
    loss_action_default_equiv=0.04798
    loss_total_minus_action=0.00899
    loss_anchor_pv=0.47696
    loss_anchor_object_pull=0.04196
    loss_mapg_routing=0.41752
    loss_slot_jepa=0.71236
    active/downstream overlap=0.075 / 0.075
    logical_batch_global_micro_count=2
    logical_batch_distinct_bucket_count=2
  ```

  Interpretation: the K2 task-uniform path is operational and structure metrics
  are not regressing.  Action is stochastic over this short window and is not
  trending down.  This is therefore a production-path sanity check only, not
  evidence that the large-batch convergence problem is solved, because every
  optimizer step still covers only two windows.

- [x] F5 per-bucket metric diagnosis.
  For every training smoke, record:
  `bucket_*_loss_action_default_equiv`,
  `bucket_*_loss_total_minus_action`,
  `bucket_*_loss_anchor_pv`,
  `bucket_*_loss_slot_jepa`, and selected bucket counts.
  Pass criteria:
  no single task bucket dominates loss without being visible in diagnostics.

- [x] F7/F7b gradient-cosine probe.
  F7 first proved an important boundary fact: `loss_action` does not directly
  update `picf_core` under the current stop-gradient/action-boundary design.
  F7b then measured the trainable semantic/action adapter group and the PICF
  structural group separately.

  Hard result from real CALVIN buckets at checkpoint step 11000:

  ```text
  loss_action on semantic/action trainable group:
    finite_pairs=30
    negative_pairs=14
    negative_fraction=0.4667
    min_cosine=-0.3768

  loss_action on picf_core:
    finite_pairs=0
    all grad norms are zero, as intended by the action/PICF boundary.

  loss_total_minus_action on picf_core:
    finite_pairs=30
    negative_pairs=6
    negative_fraction=0.2000
    min_cosine=-0.0891
  ```

  Interpretation:

  ```text
  The main measured conflict is action-bucket conflict in the semantic/action
  trainable group, not direct action corruption of PICF core.  The structural
  PICF gradients are mildly conflicting but not strongly adversarial.

  Therefore:
    - do not add whole-model PCGrad/CAGrad;
    - do not rewrite the action expert or add MoE yet;
    - prioritize task-covered logical batches and per-bucket normalization;
    - if gradient surgery is tested later, restrict it to semantic/action
      adapter parameters and only compare it against a K>=4 logical-batch run.
  ```

## Deployment Checklist

- [x] D1 sampler mode CLI.
  Add:

  ```text
  --calvin-bucket-sampling-mode round_robin|task_uniform|trajectory|temperature
  --calvin-bucket-temperature-alpha FLOAT
  --calvin-bucket-weight-spec STRING
  ```

  `round_robin` keeps current behavior.  `task_uniform` samples bucket uniformly
  per global micro-slot.  `trajectory` samples proportional to bucket segment
  count.  `temperature` uses `q_b ∝ N_b^alpha`.

- [x] D2 explicit bucket mixture weights.
  Add a source method returning target `q_b` for the active sampler.  Logical
  loss scaling must use these `q_b`, not always uniform.

- [x] D3 no-model sampler audit script.
  `scripts/picf_calvin_bucket_sampler_audit.py` instantiates the same CALVIN
  source used by training and prints bucket histograms, expected weights,
  observed frequencies, and first sampled examples.

- [x] D4 per-bucket metrics are retained.
  The previous logical-batch metrics stay mandatory.

- [x] D5 future large-data hooks.
  Keep placeholders for dataset id and embodiment id in the plan, but do not
  invent fake CALVIN embodiment logic.

- [x] D6 remote launchers.
  Add env vars for sampler mode, temperature alpha, and weight spec.

- [x] D7 verification.
  Required checks are complete for this gate:
  `py_compile`, launcher `bash -n`, sampler audit on real remote CALVIN data,
  local FSDP partial-trainable tests, and remote startup validation with the
  correct repository `PYTHONPATH`.

## Decision Rules

- If K4 full still OOMs, the honest conclusion is:
  production dual-card convergence is not proven; use K4 reduced-scope only for
  infrastructure and plan a memory reduction or larger machine.

- If K4 reduced-scope passes and K2 task-uniform has no regressions, deploy the
  sampler family but keep long-run convergence status open.

- If per-bucket metrics show one bucket dominates action loss, fix mixture /
  normalization before another 30K.

- If gradient cosine is strongly negative, the next architecture change is
  adapter/action-expert boundary or adapter-only gradient surgery, not more
  anchor losses.

## First 1-2 Hour Execution Plan

1. Implement D1-D3 and D6.
2. Run local syntax and sampler formula checks.
3. Sync to the dual-card machine.
4. Run F1 sampler audit.
5. Run F2 K4 feasibility for a short target.
6. If F2 OOMs, immediately run F3 reduced-scope K4 and F4 K2 task-uniform.
7. Update this document with exact pass/fail and metrics.

## 2026-06-02 Implementation Notes

- [x] Local syntax verification:

  ```text
  python -m py_compile scripts/picf_core_train.py scripts/picf_calvin_bucket_sampler_audit.py
  bash -n scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh \
         scripts/experiments/picf_aqr_owm_202605_active/run_a7_e23_bucketbalanced_noactioncond_from11000_30k_20260601.sh \
         scripts/experiments/picf_aqr_owm_202605_active/run_a7_actionaware_after_dedup_smoke300_20260520.sh
  git diff --check
  ```

- [x] Local audit CLI verification:

  ```text
  python scripts/picf_calvin_bucket_sampler_audit.py --help
  ```

- [x] Sampler formula sanity:

  ```text
  task_uniform: a=b=c=1/3
  trajectory for counts 1/3/6: 0.1 / 0.3 / 0.6
  temperature alpha=0.5 for counts 1/3/6: 0.193 / 0.334 / 0.473
  explicit a=1,b=2,*=0.5: 0.286 / 0.571 / 0.143
  ```

- [x] Remote no-model sampler audit completed on real CALVIN sidecar data:

  ```text
  round_robin:  max_abs=0.0003, KL=0.000001
  task_uniform: max_abs=0.0095, KL=0.000952
  trajectory:   max_abs=0.0155, KL=0.001640
  temperature:  max_abs=0.0152, KL=0.002135
  ```

- [x] Remote full-model K4 feasibility checked and resource-blocked:
  the run reached the first optimizer step, sampled two local buckets with
  correct logical scales, then OOMed in token-fusion attention on 2xA100-40GB.

- [x] Remote K2 task-uniform production sanity completed through step 11050.
  It proves the K2 path is operational and structurally healthier than the
  bucket-only baseline, but not action convergence.

- [x] Remote F6b suffix-context/action-boundary gate completed.  F6 was
  invalid because action conditioning was disabled; F6b explicitly enabled
  action conditioning, produced nonzero context metrics, and then reproduced
  the F4 K2 trajectory almost exactly.  It is therefore a valid negative result:
  suffix-side context injection alone does not solve the action plateau.

- [x] Launcher overlay-signature bug fixed:
  `run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh` previously passed
  `--anchor-overlay-dump-signatures` even when `ANCHOR_OVERLAY_INTERVAL=0`,
  which blocks speed/memory probes at argument validation time.  The launcher
  now passes dump-signatures only when overlays are enabled.

## Full Requested-Point Coverage Matrix

This section prevents the training-system repair from being reduced to "just
increase batch".  Each requested point is tracked as deployed, intentionally
deferred, or still required.

### Deployed Now

- [x] Task/dataset-mixing surface.
  CALVIN now exposes explicit bucket sampling modes rather than implicit global
  random sampling:

  ```text
  round_robin
  task_uniform
  trajectory
  temperature
  explicit --calvin-bucket-weight-spec
  ```

- [x] Batch-balancing ratios.
  `--calvin-bucket-weight-spec` is a strict ratio contract.  A bucket must be
  covered exactly, by glob, or by `*=`; otherwise startup fails.  This avoids
  silent partial ratios.

- [x] Stratified micro-batch + gradient accumulation interface.
  `--logical-batch-task-count` and `--accum-steps` define the logical optimizer
  step.  Current dual-card production supports K=2; K=4 full-model is
  resource-blocked.

- [x] Per-task/bucket loss normalization.
  Each selected bucket is weighted by target `q_b`, not by raw micro-window
  frequency:

  ```text
  scale_b = world_size * q_b / n_b
  ```

  This is the unbiased estimator for the chosen bucket objective inside one
  optimizer step.

- [x] Per-bucket metrics.
  The log records action, non-action, anchor, object-pull, MAPG, and slot-JEPA
  metrics per bucket.  This is required before any dynamic mixing.

- [x] Action/backbone boundary guard already exists in the maintained launcher.
  The current command exposes `ACTION_PREFIX_STOPGRAD`, context stop-grad,
  semantic trainable scope, and separate optimizer groups.  This is not a new
  action-head architecture, but the boundary is configurable and measured.

### Deliberately Not Deployed In This Gate

- [!] Dynamic PiKE-style mixing.
  PiKE is appropriate after we know whether bucket gradients are compatible.
  Deploying it before gradient-cosine evidence would be another uncontrolled
  moving target.  Next gate: run the gradient-cosine probe on action/PICF
  adapters only.

- [!] PCGrad / CAGrad / gradient surgery.
  These require per-task gradients and are expensive on a large VLA.  They are
  justified only if gradient cosine is strongly negative.  Otherwise fixed
  mixture plus normalization is cleaner and more scalable.

- [!] Dataset/embodiment normalization beyond CALVIN.
  CALVIN has one dataset and one embodiment in the current run.  The correct
  large-data extension is to add `dataset_id` and `embodiment_id` to the same
  mixture/normalization contract, not to invent fake CALVIN domains.

- [!] Action-head rewrite / MoE / System-2 planning split.
  These are architecture gates, not 1-2 hour training-system gates.  The current
  evidence first tests whether the update estimator is wrong.  If K-covered
  training remains flat after sampler normalization, the next structural change
  is action-expert boundary / adapter-level gradient control, not more anchor
  losses.

### Immediate Missing Proofs

- [ ] K>2 runnable proof on full model.
  Required for a hard convergence claim.  Current 2x40GB hardware cannot run K4
  full-model without further memory reduction or more GPU memory.

- [x] Reduced-scope K4 infrastructure proof.
  Completed on 2xA100-40GB through step 11060 with full 7-bucket CALVIN
  sidecar distribution.  It proves K4 sampler/loss-scale/metrics plumbing, but
  action loss was disabled, so it is not an action convergence proof.

- [x] Gradient-cosine probe.
  F7b completed on real CALVIN buckets.  It showed meaningful negative cosine
  in the semantic/action trainable group and zero direct action gradient into
  PICF core under the current boundary.  This justifies a future adapter-only
  gradient-control gate if F8 fails; it does not justify whole-model gradient
  surgery.

## Current Deployment Recommendation

For the next dual-card production-compatible run, use:

```text
CALVIN_BALANCED_BUCKET_SAMPLER=1
CALVIN_BUCKET_SAMPLING_MODE=task_uniform
LOGICAL_BATCH_TASK_COUNT=2
LOGICAL_BATCH_BUCKET_NORMALIZATION=1
LOGICAL_BATCH_LOG_BUCKET_METRICS=1
ACCUM_STEPS=1
WINDOW_ACTIVATION_CHECKPOINTING=1
```

This is the honest 2xA100-40GB deployment.  It is mathematically cleaner than
ordinary bucket-balanced sampling because every logged update has explicit
target weights and per-bucket accounting, but it is still not the final
large-batch proof.  Full-model K4 currently requires either more memory,
reduced token budgets, or additional sharding/memory work; the K4 reduced-scope
run proves the estimator path, not full production action convergence.

For the actual "large-data scaling" target, the next complete design is:

```text
dataset_id / embodiment_id / task_bucket
  -> explicit target mixture q_g
  -> per-group normalized micro losses
  -> K>=4 covered optimizer step
  -> per-group metrics
  -> gradient-cosine gate
  -> only then dynamic mixing or adapter-level gradient surgery
```

## 2026-06-02 Strict Coverage Update: What Was Actually Done

This section is intentionally blunt.  The requested repair has two different
classes of items:

```text
training-system items:
  can be deployed and smoke-tested in 1-2 hours;

architecture items:
  require a separate model-design gate and cannot be mixed into the same short
  sampler test without destroying interpretability.
```

### User-Requested Items: Status

| requested item | current status | hard evidence | decision |
| --- | --- | --- | --- |
| probabilistic task/dataset mixing | deployed for CALVIN task buckets | F1 audit covers round_robin/task_uniform/trajectory/temperature; strict ratio spec implemented | keep |
| batch-balancing ratios | deployed for CALVIN buckets | `--calvin-bucket-weight-spec` is strict and startup-failing if incomplete | keep |
| gradient accumulation as logical batch | deployed | `--logical-batch-task-count == WORLD_SIZE * accum_steps`; K2 production runs; K4 reduced-scope runs | keep |
| per-task/bucket loss normalization | deployed | `scale_b = world_size * q_b / n_b`; per-bucket target weights logged | keep |
| per-task/bucket metrics | deployed | F3/F4 logs include selected bucket counts and per-bucket losses | keep |
| dataset/embodiment normalization | not meaningful on current CALVIN-only run | one dataset / one embodiment; no domain ids to normalize yet | add when scaling beyond CALVIN |
| action/backbone gradient boundary | partially already available | current launcher exposes prefix/context stop-grad and separate LR groups; no full knowledge-insulation action-expert rewrite | keep boundary; do not claim full KI |
| dynamic PiKE-style mixing | not deployed | requires gradient-cosine evidence first | next only if fixed mixture still fails |
| PCGrad/CAGrad | not deployed | per-task gradients are expensive and unjustified until cosine is strongly negative | defer |
| action-head rewrite / MoE / System-2 split | not deployed | multi-day architecture change; current short test is about data estimator | defer unless estimator fails |

### Why This Is Not "Just Increasing Batch"

The current implementation changes the objective estimator, not only the
physical batch size.  For a target bucket mixture `q_b`, each optimizer step
uses:

```text
L_hat = sum_{m in selected windows} scale(b_m) * ell_m
scale(b) = world_size * q_b / n_b
```

where `n_b` is the global number of selected micro-windows from bucket `b` in
the optimizer step.  DDP averages rank gradients, so the `world_size` factor is
required.  This makes duplicated buckets inside one logical step behave as one
bucket contribution rather than as multiple accidental votes.

That is the VLA-Foundry/ABot-M0 compatible part: controlled mixture plus
per-group normalization.  It is still not PiKE, PCGrad, MoE, or a new action
expert.

### Current Dual-Card Limit

On 2xA100-40GB:

```text
K2 full production:
  runnable, action-on, but only two windows/update.

K4 full production:
  OOM in token_fusion attention backward/recompute even with window activation
  checkpointing.

K4 reduced-scope:
  runnable; proves sampler/loss-scale/metrics infrastructure, not action
  convergence.
```

Therefore the honest conclusion is:

```text
The dual-card deployment is mathematically cleaner than the previous bucket
sampler, but it has not yet proven the large-batch convergence behavior.
```

### Immediate 1-2 Hour Test Contract

The current active short test is:

```text
EXP=picf_f5_k2_full_actionon_taskuniform_11000_to11100_20260602
world_size=2
accum_steps=1
logical_batch_task_count=2
calvin_bucket_sampling_mode=task_uniform
logical_batch_bucket_normalization=on
logical_batch_log_bucket_metrics=on
action_on=yes
optimizer_checkpoint_mode=model_only
```

Pass criteria for this short gate:

```text
required:
  metrics.jsonl appears by step 11010;
  logical_batch_global_micro_count=2;
  per-bucket metrics are present;
  no NaN/OOM/checkpoint-load failure;
  structure losses stay in the F4/F3 healthy range.

weak positive:
  action loss shows a lower rolling mean than F4 bucket-only/K2 over 50-100
  steps.

failure:
  no action descent and no per-bucket explanation after 100 steps;
  or structure metrics regress while action stays flat.
```

Current first hard row:

```text
step=11010
loss_action_default_equiv=0.03606
loss_total_minus_action=0.01287
loss_anchor_pv=0.52585
loss_anchor_object_pull=0.41208
loss_mapg_routing=0.45131
loss_slot_jepa=0.71327
posterior_recycle_rate=0.07565
posterior_identity_switch_rate=0.23056
active/downstream overlap=0.250 / 0.243
logical_batch_global_micro_count=2
logical_batch_distinct_bucket_count=2
steps_per_sec=0.0265
```

Interpretation:

```text
The run is live and the mathematical/logging contract is correct.
The first row is not enough to prove descent.  At this speed, 100 optimizer
steps take roughly one hour, so this is the fastest meaningful full-model
dual-card check currently available without changing trainable scope.
```

Second hard row:

```text
step=11020
loss_action_default_equiv=0.04472
loss_total_minus_action=0.00927
loss_anchor_pv=0.49687
loss_anchor_object_pull=0.06408
loss_mapg_routing=0.42873
loss_slot_jepa=0.66722
posterior_recycle_rate=0.07946
posterior_identity_switch_rate=0.23333
active/downstream overlap=0.125 / 0.125
logical_batch_global_micro_count=2
logical_batch_distinct_bucket_count=2
```

Interpretation:

```text
The structural side is improving or stable: non-action loss, anchor-PV,
object-pull, MAPG routing, and active/downstream overlap all improved from the
first row.

Action did not improve over the first row.  It moved 0.03606 -> 0.04472.  This
does not yet fail the test because the window is only 20 steps, but it already
shows that K2 task-uniform + bucket-normalization alone is not an immediate
replica of the E21/K12 descent.
```

If this fails, the next experiment is not another random LR tweak.  The next
scientific fork is:

```text
Option A:
  make K4+ full fit by reducing token budget / memory, then rerun full action
  with K>=4 task coverage;

Option B:
  run gradient-cosine diagnostics on adapter/action/PICF groups and decide
  dynamic PiKE vs adapter-level gradient surgery;

Option C:
  if action gradients demonstrably corrupt semantic/PICF features, open a
  separate Knowledge-Insulation-style action-expert boundary gate.
```

F5 stop decision:

```text
F5 was stopped after step 11020.
Reason:
  its first two rows reproduce F4 almost exactly, while F4 already ran to
  step 11050.  Continuing F5 would spend another hour duplicating an existing
  K2 full action-on result.
```

Direct comparison against the completed K2 bucket-only baseline:

```text
E27 bucket-only, no logical-batch normalization:
  step11050 action=0.04787 nonaction=0.01058 active/downstream=0.160/0.141
  step11100 action=0.04463 nonaction=0.01123 active/downstream=0.175/0.171

F4 task_uniform + target-aware bucket normalization:
  step11050 action=0.04798 nonaction=0.00899 active/downstream=0.075/0.075
```

Conclusion:

```text
The deployed sampler/normalization repair improves structure health and
per-bucket observability, but K2 alone does not improve action convergence over
the bucket-only baseline.  This rules out the claim that "dual-card K2
task-uniform normalization already solves the small-batch action plateau."
```

## F6: Action-Interface Boundary Check

Reason:

```text
F4/F5 showed that K2 task-uniform normalization improves structure but not
action.  The next one-variable test is whether the action interface itself is
the immediate bottleneck.
```

Contract:

```text
EXP=picf_f6_suffix_k2_taskuniform_norm_from11000_to11100_20260602
same checkpoint as F4/F5: step11000
same action weight: 4.0
same LR: 2e-5
same semantic LR scale: 0.35
same PICF core LR scale: 0.001
same K2 task_uniform + bucket normalization
changed only:
  ACTION_CONTEXT_INTEGRATION=suffix_cross_attention
```

This is not a generic random retry.  It tests the action/PICF boundary implied
by Knowledge-Insulation-style VLA analysis: if action gradients or PICF context
injection are the short-run bottleneck, suffix-side controlled injection should
change action behavior while keeping bucket/structure math fixed.

Pass criteria:

```text
by step 11050:
  action loss lower than F4 step11050 (0.04798), preferably trending down;
  non-action structure remains near F4 healthy range;
  no active/downstream overlap regression.
```

Failure criteria:

```text
if action remains F4-like while structure remains healthy:
  the immediate problem is not the prefix-vs-suffix action interface;
  prioritize K>=4 full coverage or gradient-cosine diagnostics.
```

### F6 Validity Audit And F6b Correction

F6 as first launched is **not** a valid suffix-context experiment.

Code/dataflow follow-through:

```text
launcher:
  run_a7_e23_bucketbalanced_noactioncond_from11000_30k_20260601.sh
  line 47 default:
    PICF_ACTION_CONDITION_ENABLED=0

policy:
  PicfPi05Policy.forward_train_transition()
    if _picf_action_condition_enabled():
      _training_action_condition_tokens(...)
    else:
      prefix_tokens=None
      use_action_adapter=False

observed F6 row:
  config log:
    context_tokens=24
    context_integration=suffix_cross_attention
  metrics:
    pi_context_token_count=0
    pi_context_adapter_token_count=0
    pi_action_condition_token_count=0
```

Conclusion:

```text
F6 did not test suffix/action-side context.  It inherited the no-action-condition
gate from E23, so context construction was bypassed.  Do not cite F6 as evidence
that suffix cross-attention helps or fails.
```

F6b fixes only the invalid gate:

```text
launcher:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_f6b_suffix_context_on_k2_taskuniform_11000_to11100_20260602.sh

fixed controls:
  same checkpoint step11000
  same K2 task_uniform + target-aware bucket normalization
  same action weight and LR
  same model-only optimizer resume

changed:
  PICF_ACTION_CONDITION_ENABLED=1
  ACTION_CONTEXT_INTEGRATION=suffix_cross_attention
```

F6b hard validation gate:

```text
by the first logged row:
  pi_context_token_count > 0
  pi_context_adapter_token_count > 0
  pi_action_condition_token_count > 0
  logical_batch_global_micro_count = 2
  logical_batch_distinct_bucket_count >= 1

if any context metric stays zero:
  stop immediately and debug the code path before any action conclusion.
```

F6b decision gate:

```text
compare against F4/F5:
  F4 step11050 action=0.04798, nonaction=0.00899,
  active/downstream overlap=0.075/0.075.

positive:
  context metrics are nonzero and rolling action improves versus F4/F5 without
  structural regression.

negative:
  context metrics are nonzero but action remains F4-like.  Then K2 action
  plateau is not explained by the prefix-vs-suffix interface alone.
```

F6b first hard row:

```text
step=11010
loss_action_default_equiv=0.03625
loss_total_minus_action=0.01314
loss_anchor_pv=0.52511
loss_anchor_object_pull=0.43836
loss_mapg_routing=0.45137
loss_slot_jepa=0.70204
active/downstream overlap=0.250 / 0.244
logical_batch_global_micro_count=2
logical_batch_distinct_bucket_count=2
pi_action_condition_token_count=28
pi_context_token_count=24
pi_context_adapter_token_count=28
pi_context_adapter_gate=0.1192
pi_context_adapter_attention_entropy_mean=3.1143
```

Interpretation:

```text
F6b is now a valid suffix-context/action-boundary test.  The first row is
dataflow-positive, but action is still F4-like at the same step.  Continue to
50/100 steps before judging whether suffix-side injection helps.
```

F6b early stop decision:

```text
step 11010 action=0.03625 structure=0.01314 active/downstream=0.250/0.244
step 11020 action=0.04496 structure=0.00926 active/downstream=0.100/0.100
step 11030 action=0.05376 structure=0.00879 active/downstream=0.075/0.050
```

Reason:

```text
F6b reproduces the F4 K2 task-uniform trajectory almost exactly:
  F4 step11010 action=0.03607
  F4 step11020 action=0.04472
  F4 step11030 action=0.05370

The suffix-context path is live, but it does not change the short-run action
behavior.  Continuing to 100 steps would duplicate F4.  Stop F6b and start F7
gradient-cosine diagnostics.
```

## Requested-Point Execution Matrix

This table is the active no-handwaving checklist for the user's full request.

| requested point | status | evidence / next action |
| --- | --- | --- |
| probabilistic task/dataset mixing | deployed for CALVIN task buckets | `round_robin`, `task_uniform`, `trajectory`, `temperature`; F1 sampler audit passed |
| explicit batch-balancing ratios | deployed | `--calvin-bucket-weight-spec`; startup fails on invalid/missing bucket specs |
| stratified logical micro-batch | deployed | `--logical-batch-task-count`; K2 full runs; K4 full OOM; K4 reduced-scope runs |
| gradient accumulation as logical batch | deployed but hardware-limited | K4 full action OOM on 2xA100-40GB; reduced-scope K4 validates estimator/dataflow only |
| per-task/bucket loss normalization | deployed | `scale_b = world_size * q_b / n_b`; target weights logged |
| per-task/bucket metrics | deployed | bucket action/non-action/anchor/JEPAs logged; F4/F5/E27 comparisons available |
| action/backbone boundary | partially deployed | prefix stopgrad, context stopgrad, separate LR groups, suffix adapter exist; F6 was invalid; F6b is the valid gate |
| cheap dynamic bucket/action weighting | deployed as F9b diagnostic | per-bucket action EMA scaling with DDP-global telemetry; v3 telemetry passed at step11010; optimization effect pending step11020+ |
| full dynamic PiKE-style mixing | not deployed | F9b is only the cheap GradNorm/PiKE-like action-scale branch, not gradient-informed dynamic sampling |
| PCGrad/CAGrad | conditionally deferred | F7b supports adapter-only conflict, not whole-model surgery; run adapter-only surgery only if F9b remains F8r-like |
| MoE action expert | not deployed | deferred until fixed mixture + boundary + cosine show task-family specialization conflict |
| full action expert rewrite | not deployed | current action chunk/flow head already exists; rewrite only if F6b/cosine prove boundary insufficiency |

Immediate plan after repaired F8/F9a/F9b:

```text
1. Do not repeat K2 suffix/context or freeze-adapter controls.
   F6b and F9a already rejected those as sufficient fixes.
2. Let F9b v3 reach step11020.
   Dataflow is already accepted at step11010; step11020 decides whether cheap
   per-bucket action EMA scaling changes optimization.
3. If F9b v3 remains F8r-like at step11020+, stop it and record "dynamic
   action-scale weighting insufficient".
4. The next non-duplicate branch is adapter-only PCGrad/CAGrad or an explicit
   Knowledge-Insulation-style action-boundary change.
5. Do not deploy MoE or a full action-head rewrite until adapter-only gradient
   surgery or boundary isolation is tested.
```

F8 launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/run_a7_f8_k4_action_adapter_taskuniform_from11000_11100_20260602.sh
```

F8 success/failure rules:

```text
Hard dataflow pass:
  logical_batch_global_micro_count = 4
  logical_batch_distinct_bucket_count >= 3 on most logged rows
  bucket target weights logged
  semantic_trainable_scope = action_head_and_adapter
  action_condition_enabled = true

Learning pass:
  loss_action_default_equiv improves against E27/F4/F6b over the same
  resumed-step window;
  bucket action losses do not show a single bucket permanently dominating;
  structural loss_total_minus_action remains stable;
  no old anchor collapse regression.

Fail fast:
  OOM at first optimizer step -> F8 is memory-blocked, not mathematically
  invalid.
  Fits but action remains F4-like -> proceed to F9 adapter-only PCGrad/CAGrad
  because F7b measured real action-semantic negative cosine.
```

F8 launch repair:

```text
First F8 launch failed before training:
  ValueError: Must flatten tensors with uniform requires_grad when
  use_orig_params=False.

Cause:
  semantic_trainable_scope=action_head_and_adapter creates a partial-trainable
  semantic root.  The existing semantic FSDP root ignored minority dtypes but
  did not ignore same-dtype frozen states, so flat-parameter FSDP saw mixed
  frozen/trainable tensors in one managed handle.

Fix:
  _fsdp_wrap_root_with_ignored_non_dominant_dtypes now collects only managed
  non-nested parameters, ignores all frozen states plus minority-dtype trainable
  states, and flattens only uniform requires_grad trainable states.

Verification:
  py_compile passed.
  pytest -q scripts/picf_core_train_test.py -k
    'fsdp_wrap_root_with_ignored_non_dominant_dtypes or
     fsdp_wrap_root_ignores_frozen_states or fsdp_wrap_kwargs or
     fsdp_root_ignored_modules'
  => 5 passed.

  After adding sampler/estimator unit coverage:
  pytest -q scripts/picf_core_train_test.py -k
    'calvin_bucket_sampling_weights or logical_batch_loss_scales or
     fsdp_wrap_root_with_ignored_non_dominant_dtypes or
     fsdp_wrap_root_ignores_frozen_states or fsdp_wrap_kwargs or
     fsdp_root_ignored_modules'
  => 8 passed.

  The new tests pin:
    - task_uniform / trajectory / temperature / explicit bucket ratios;
    - duplicated-bucket normalization without DDP;
    - DDP gradient-averaging compensation with duplicated buckets.

Remote test caveat:
  running pytest directly under `/root/openpi_e21b2_1b07eab` without launcher
  `PYTHONPATH` imports `/root/openpi/src` from the shared venv path and can
  fail collection against stale source.  Valid remote verification must set:

  PYTHONPATH=/root/openpi_e21b2_1b07eab/src:/root/openpi_e21b2_1b07eab/packages/openpi-client/src:$PYTHONPATH

  With that path, remote import confirms:
    pipeline file = /root/openpi_e21b2_1b07eab/src/openpi/picf/core/pipeline.py
    train file    = /root/openpi_e21b2_1b07eab/scripts/picf_core_train.py
  and the same 5 FSDP tests pass.

Relaunched F8 runtime confirmation:

```text
session=picf_f8_k4_action_adapter_20260602
status=running after FSDP repair
world_size=2
training_strategy=fsdp_full_shard
accum_steps=2
effective_global_batch=4
logical_batch_task_count=4
logical_batch_bucket_normalization=True
calvin_bucket_sampling_mode=task_uniform
semantic_trainable_scope=action_head_and_adapter
action_condition_enabled=True
window_activation_checkpointing=True
optimizer_checkpoint_mode=model_only

optimizer groups:
  semantic/action adapter scope: lr=2e-05 num_params=4,228,625
  picf_core:                     lr=2e-08 num_params=116,478,464
  policy_head:                   lr=2e-05 num_params=109,708,660
```

Interpretation:
  F8 is not a full-backbone cotrain run and not a DDP workaround.  It is the
  intended K4 task-covered action-adapter gate.  The very low PICF-core LR is
  deliberate for this gate: action-gradient conflict was measured in the
  semantic/action group, while direct action gradients into PICF core were
  zero by design.
```

F8 first hard row:

```text
step=11010
loss_action_default_equiv=0.04179
loss_total_minus_action=0.00902
loss_anchor_pv=0.51618
loss_anchor_object_pull=0.05377
loss_mapg_routing=0.44261
loss_slot_jepa=0.69553
posterior_recycle_rate=0.07546
posterior_identity_switch_rate=0.22639
aqr_active_same_role_support_overlap_max=0.20000
aqr_downstream_same_role_support_overlap_max=0.17906
logical_batch_global_micro_count=4
logical_batch_distinct_bucket_count=4
selected buckets=block_lift,drawer,other,switch_button_light
grad_norm=0.28330
steps_per_sec=0.0164
```

Per-bucket action signal at the same log interval:

```text
block_push=0.0280
other=0.0392
slider=0.0395
drawer=0.0396
block_other=0.0400
block_lift=0.0443
switch_button_light=0.0586
```

Interpretation:

```text
The F8 dataflow contract passes: K=4 coverage, target-aware normalization,
per-bucket accounting, action conditioning, and the partial-trainable FSDP
repair are all live in the same run.

This first row is not enough to claim learning.  Action is not an immediate
breakthrough over F4/F6b; it is between the F4 first row and later F4 rows.
Structure is healthy and substantially better than the old collapse regime.
The useful new evidence is per-bucket: switch_button_light is currently the
hardest action bucket, while block_push is easiest.  This supports the task
gradient/coverage diagnosis and argues against more anchor-only losses.

Decision rule:
  continue F8 to at least step 11020/11030.  If action does not beat the F4/F6b
  trajectory by then, K=4 task coverage alone is insufficient and F9 should be
  adapter/action-group gradient control, not another sampler-only run.
```

F8 second hard row:

```text
step=11020
loss_action_default_equiv=0.04096
loss_total_minus_action=0.01200
loss_anchor_pv=0.48919
loss_anchor_object_pull=0.32547
loss_mapg_routing=0.43892
loss_slot_jepa=0.68280
logical_batch_global_micro_count=4
logical_batch_distinct_bucket_count=3
selected buckets=block_lift,block_other,block_push,block_push
```

Comparison against the earlier K2 gates:

```text
F4 step11020 action=0.04472
F6b step11020 action=0.04496
F8 step11020 action=0.04096
```

Interpretation:

```text
F8 has a real but small action improvement over K2 at the same resumed-step
window.  It is not yet the E21-style rapid descent.  Structural metrics are
mixed: anchor_pv, MAPG routing, and slot-JEPA improved, while total-minus-action
and object_pull worsened.

The object_pull worsening is bucket-local rather than global:
  switch_button_light object_pull ~= 1.88
  block_lift object_pull ~= 0.27
  block_other object_pull ~= 0.21
  block_push object_pull ~= 0.09

This supports the bucket-conflict diagnosis.  The model does not have one
uniform failure mode; it has action/structure conflict concentrated in harder
task families.  That is exactly why per-bucket metrics are now mandatory.
```

F8 contract correction after step11030:

```text
step=11030
logical_batch_global_micro_count=4
logical_batch_distinct_bucket_count=2
selected buckets=block_lift,block_lift,block_lift,block_other
loss_action_default_equiv=0.04640
```

This invalidates F8 as a completed K4 task-coverage experiment.  The code had
implemented non-round-robin bucket modes as independent per-micro-window draws,
so a logical batch configured with K=4 could still be dominated by one task
family.  That is not the VLA Foundry / ABot-M0 style contract requested here.

Required repair before any F9/PiKE/PCGrad/MoE decision:

```text
1. choose one global bucket sequence per optimizer step;
2. default non-round-robin modes to weighted sampling without replacement;
3. map rank/micro-step to that shared sequence;
4. keep concrete segment/window sampling randomized inside the selected bucket;
5. audit per-step distinct bucket count, not only long-run bucket frequency.
```

After this repair, the valid F8 rerun must satisfy:

```text
logical_batch_global_micro_count=4
logical_batch_distinct_bucket_count=4 when K<=positive_bucket_count
calvin_bucket_sample_without_replacement=True
```

Only the repaired F8 may answer whether K4 task-balanced logical batching is
sufficient.  The pre-repair F8 rows are retained only as evidence that the
sampler contract was incomplete.

Sampler repair verification:

```text
artifact=/mnt/picf_run_logs/picf_sampler_audit_k4_without_replacement_20260602.json
world_size=2
accum_steps=2
steps=256
mode=task_uniform
calvin_bucket_sample_without_replacement=true
bucket_count=7

min_distinct_buckets_per_step=4
mean_distinct_buckets_per_step=4.0
max_distinct_buckets_per_step=4
max_abs_deviation=0.01046
KL(empirical||target)=0.00113
```

This passes the actual K4 logical-batch dataflow gate.  The next F8 launch is
therefore a valid test of task-covered logical batching rather than another
independent-sampler run.

## Final 1-2 Hour Verification Contract

This is the current execution contract.  It explicitly covers every method
requested in the training-system discussion and prevents another "batch only"
interpretation.

### Must Finish In This Gate

- [x] F6b valid action-boundary check.
  This run is complete and was stopped early because it exactly reproduced the
  F4 K2 trajectory.

  ```text
  EXP=picf_f6b_suffix_context_on_k2_taskuniform_11000_to11100_20260602
  checkpoint=step11000
  world_size=2
  accum_steps=1
  logical_batch_task_count=2
  bucket_sampling=task_uniform
  bucket_normalization=on
  action_condition_enabled=on
  action_context_integration=suffix_cross_attention
  action_context_tokens=24
  action_weight=4.0
  ```

  Hard pass for dataflow:

  ```text
  pi_context_token_count > 0
  pi_context_adapter_token_count > 0
  pi_action_condition_token_count > 0
  logical_batch_global_micro_count = 2
  per-bucket metrics present
  ```

  If the hard pass fails, no action conclusion is allowed.

- [x] F7/F7b gradient-cosine probe design and launch.
  This is the required bridge from fixed mixture to PiKE/PCGrad decisions.
  It should measure coarse gradient compatibility on trainable groups only:

  ```text
  buckets: block_lift, block_push, drawer, slider, switch_button_light, other
  parameter groups:
    policy_head
    picf_core
    semantic_backbone if trainable and memory-safe
  losses:
    action_default_equiv
    total_minus_action or structural auxiliary
  output:
    grad_norm per bucket/group
    cosine matrix per group
    negative-cosine fraction
  ```

  Decision:

  ```text
  mostly nonnegative cosine + high variance:
    use PiKE/dynamic sampling or stronger K coverage.

  strongly negative cosine on action/PICF groups:
    try adapter-only PCGrad/CAGrad or stronger action/backbone insulation.

  gradients low-rank/near-zero for PICF while action is flat:
    adjust trainable scope/LR, not sampler.
  ```

  Executed remote artifacts:

  ```text
  F7:
    output=/mnt/picf_run_logs/picf_f7_gradient_cosine_after_f6b_20260602.json
    log=/mnt/picf_run_logs/picf_f7_gradient_cosine_after_f6b_20260602.log
    result=action gradient to picf_core is zero; policy_head group name did
           not capture semantic/action trainables.

  F7b:
    output=/mnt/picf_run_logs/picf_f7b_gradient_cosine_semantic_structure_20260602.json
    log=/mnt/picf_run_logs/picf_f7b_gradient_cosine_semantic_structure_20260602.log
    groups=semantic,picf_core
    loss_keys=loss_action,loss_total_minus_action
    sampled_elements_per_group=250000
  ```

  F7b measured:

  ```text
  action semantic gradients:
    negative_fraction=0.4667
    min_cosine=-0.3768

  structural PICF gradients:
    negative_fraction=0.2000
    min_cosine=-0.0891
  ```

### Already Deployed And Tested

- [x] VLA-Foundry-style fixed mixture surface:
  `round_robin`, `task_uniform`, `trajectory`, `temperature`,
  explicit `--calvin-bucket-weight-spec`.

- [x] ABot-M0-style task-family balancing for CALVIN buckets:
  real-data sampler audit passed for all modes.

- [x] Per-bucket normalized objective estimator:
  `scale_b = world_size * q_b / n_b`.

- [x] Bucket-level logging:
  selected counts, target weights, action/non-action/anchor/JEPAs per bucket.

- [x] K4 infrastructure proof:
  reduced-scope K4 completed; full K4 action is memory-blocked on 2x40GB.

### Remaining Methods Not Yet Fully Deployed, With Reasons

- [x] Cheap dynamic bucket/action weighting.
  F9b implements per-bucket action EMA scaling.  It is a scalable diagnostic
  for loss-scale imbalance, not full PiKE.

- [!] Full dynamic PiKE-style mixing.
  Still not deployed.  It requires gradient-informed task weights, validation
  or improvement-rate feedback, and stronger safeguards against chasing noisy
  buckets.  F9b is the required cheap gate before that larger mechanism.

- [!] Whole-model PCGrad/CAGrad.
  These require per-task gradients and should only be applied to adapter/action
  groups after measured negative cosine.  Applying them blindly to a 4B-param
  VLA is not scalable.

- [ ] Adapter-only PCGrad/CAGrad.
  F7b makes this legitimate if F9b K4 logical batch with dynamic action-scale
  control still fails.  It must be limited to semantic/action adapter and
  policy/action-head parameters, not the full frozen backbone or PICF memory.

- [ ] Action-expert MoE / System-2 split.
  This remains a separate architecture gate.  The present failure mode must
  first distinguish coverage noise from gradient conflict.

### F9 Plan If F8 Fails

Do not jump directly to MoE or whole-model PCGrad.  The measured conflict is
localized:

```text
action gradients on semantic/action trainable group:
  negative_fraction=0.4667
  min_cosine=-0.3768

direct action gradients into PICF core:
  zero by boundary design
```

Therefore the next controlled sequence is:

```text
F9a adapter-insulation control:
  checkpoint=step11000
  K=4 if memory fits, otherwise K=2 with identical sampler/normalization
  semantic_trainable=0 or semantic_trainable_scope=none
  policy_head/action head trainable
  PICF core either frozen or very-low-LR, matching F8

  question:
    does removing semantic/action-adapter gradients improve action descent?

  positive result:
    action descends better than F8 -> the conflict is in the semantic/action
    adapter path; use controlled insulation or adapter-only gradient control.

  negative result:
    action stays F8-like -> the conflict is not solved by adapter insulation;
    then run adapter-only PCGrad/CAGrad or revisit action condition calibration.

F9b adapter-only gradient control:
  only after F9a or F8 shows adapter conflict remains.
  It must operate on the semantic/action adapter group, not whole model, and it
  must be compared against F8/F9a on the same checkpoint window.
```

This is consistent with Knowledge-Insulation-style VLA reasoning: isolate the
continuous-action gradient path before changing the full backbone or adding
experts.  It is also consistent with the scaling requirement: freezing or
controlling a small action-adapter group does not specialize the model to a
small CALVIN subset and remains compatible with future dataset/embodiment
mixtures.

### F8r Repaired K4 Result

The repaired F8 rerun used the fixed without-replacement logical-batch sampler:

```text
remote_exp=picf_f8r_k4_action_adapter_taskuniform_wor_11000_to11100_20260602
checkpoint=step11000
world_size=2
accum_steps=2
logical_batch_task_count=4
bucket_sampling=task_uniform
bucket_sample_without_replacement=true
bucket_normalization=true
semantic_trainable_scope=action_head_and_adapter
policy_head_lr_scale=1.0
picf_core_lr_scale=0.001
action_weight=4.0
```

Hard rows:

```text
step11010:
  logical_batch_global_micro_count=4
  logical_batch_distinct_bucket_count=4
  selected=block_other,block_push,slider,switch_button_light
  loss_action_default_equiv=0.04069
  loss_total_minus_action=0.00997
  loss_anchor_pv=0.48752
  loss_anchor_object_pull=0.14121
  loss_mapg_routing=0.42354
  loss_slot_jepa=0.68785
  active/downstream_same_role_overlap=0.09998

step11020:
  logical_batch_global_micro_count=4
  logical_batch_distinct_bucket_count=4
  selected=block_other,drawer,other,slider
  loss_action_default_equiv=0.04256
  loss_total_minus_action=0.01039
  loss_anchor_pv=0.47495
  loss_anchor_object_pull=0.17446
  loss_mapg_routing=0.41784
  loss_slot_jepa=0.68334
  active_same_role_overlap=0.07500
  downstream_same_role_overlap=0.07914

step11030:
  logical_batch_global_micro_count=4
  logical_batch_distinct_bucket_count=4
  selected=block_push,drawer,other,slider
  loss_action_default_equiv=0.04293
  loss_total_minus_action=0.00904
  loss_anchor_pv=0.47603
  loss_anchor_object_pull=0.05169
  loss_mapg_routing=0.41234
  loss_slot_jepa=0.68773
  active/downstream_same_role_overlap=0.05000
```

Conclusion:

```text
F8r dataflow passes.
K4 no-replacement task coverage alone is insufficient.
Structure metrics are healthy enough that same-role overlap is not the
immediate action bottleneck in this run.
Action remains F4/F6b-like and does not show the E21-style fast descent.
```

Therefore F9a is no longer optional.  The next controlled test is adapter
insulation:

```text
F9a:
  same checkpoint, same K4 without-replacement sampler, same bucket
  normalization;
  disable semantic/action adapter training;
  keep policy/action head trainable;
  keep PICF core at the same very-low LR or as close as validation permits.
```

If F9a improves action descent, the bottleneck is semantic/action-adapter
gradient conflict.  If F9a remains flat, the next justified step is
adapter-only gradient control or action-condition calibration, not another
sampler-only rerun.

### F9a Adapter-Insulation Result

F9a was launched after F8r with only one causal change:

```text
remote_exp=picf_f9a_k4_policy_only_taskuniform_wor_11000_to11100_20260602
checkpoint=step11000
same K4 no-replacement task-uniform sampler as F8r
same per-bucket normalization as F8r
semantic=paligemma(trainable=false scope=action_head_and_adapter)
optimizer groups:
  picf_core lr=2e-08 params=116,478,464
  policy_head lr=2e-05 params=109,708,660
```

Hard rows:

```text
step11010:
  logical_batch_global_micro_count=4
  logical_batch_distinct_bucket_count=4
  selected=block_other,block_push,slider,switch_button_light
  loss_action_default_equiv=0.04120
  loss_total_minus_action=0.00997
  loss_anchor_pv=0.48752
  loss_anchor_object_pull=0.14120
  loss_mapg_routing=0.42354
  loss_slot_jepa=0.68778
  active/downstream_same_role_overlap=0.09998
  grad_norm=0.02185

step11020:
  logical_batch_global_micro_count=4
  logical_batch_distinct_bucket_count=4
  selected=block_other,drawer,other,slider
  loss_action_default_equiv=0.04302
  loss_total_minus_action=0.01023
  loss_anchor_pv=0.47463
  loss_anchor_object_pull=0.15937
  loss_mapg_routing=0.41524
  loss_slot_jepa=0.68468
  active_same_role_overlap=0.06250
  downstream_same_role_overlap=0.06657
  grad_norm=0.00243
```

Conclusion:

```text
F9a dataflow passes.
Freezing semantic/action adapter does not improve action descent.
It also collapses total trainable gradient norm by roughly two orders of
magnitude compared with F8r, so policy_head-only adaptation is too weak for
this checkpoint/window.
```

This rejects the simple "freeze the adapter" interpretation.  The remaining
evidence is sharper:

```text
F7b: semantic/action adapter receives conflicting action gradients.
F8r: keeping adapter trainable with K4 task coverage does not descend.
F9a: freezing adapter removes useful gradient capacity and still does not
     descend.
```

Therefore the next defensible branch is controlled adapter training, not more
sampler-only or freeze-only experiments:

```text
F9b:
  keep semantic/action adapter trainable;
  keep K4 no-replacement task coverage and per-bucket normalization;
  control only the measured-conflicting trainable subspace.

Allowed implementations:
  adapter-only PCGrad/CAGrad if per-bucket gradients can be collected safely;
  or a cheaper PiKE/GradNorm-style dynamic bucket/action weighting if full
  per-bucket backward is too expensive.

Not allowed:
  whole-model PCGrad;
  MoE/action-expert rewrite before F9b;
  another K2/K4 sampler-only rerun.
```

### F9b Implementation Contract

F9b uses the cheaper scalable branch first:

```text
name=per-bucket action EMA normalization
applies_to=backward action component only
does_not_change=logged loss_action_default_equiv
default=off
diagnostic_launcher=
  scripts/experiments/picf_aqr_owm_202605_active/
  run_a7_f9b_k4_action_adapter_bucketema_from11000_11100_20260602.sh
```

Mathematical contract:

```text
Original:
  L_b = L_nonaction_b + L_action_b
  backward scale = s_logical_b

F9b:
  L_b^backward = L_nonaction_b + gamma_b * L_action_b
  backward scale = s_logical_b

where
  gamma_b = clamp(mean_c EMA[L_action_c] / EMA[L_action_b],
                  gamma_min, gamma_max)
```

This is not a metric trick:

```text
loss_action_default_equiv is logged before gamma_b and remains comparable.
Only the action gradient entering the trainable adapter/policy path is scaled.
```

Why this is the next controlled step:

```text
F7b showed action-gradient conflict in the adapter group.
F8r showed task coverage alone is insufficient.
F9a showed freezing the adapter is too weak.
F9b keeps the adapter trainable but prevents a hard/noisy bucket from dominating
the action gradient scale in Adam momentum.
```

Diagnostic defaults:

```text
LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION=1
LOGICAL_BATCH_ACTION_BUCKET_EMA_DECAY=0.90
LOGICAL_BATCH_ACTION_BUCKET_SCALE_MIN=0.50
LOGICAL_BATCH_ACTION_BUCKET_SCALE_MAX=1.50
LOGICAL_BATCH_ACTION_BUCKET_MIN_COUNT=2
```

### F9b v1 Telemetry Repair

The first F9b launch reached step 11010, but it is not a valid dynamic-scaling
acceptance result:

```text
remote_exp=picf_f9b_k4_action_adapter_bucketema_wor_11000_to11100_20260602
step11010:
  loss_action_default_equiv=0.040658
  loss_total_minus_action=0.010036
  loss_anchor_pv=0.487984
  loss_anchor_object_pull=0.147407
  loss_mapg_routing=0.426113
  loss_slot_jepa=0.689465
  distinct_buckets=4
```

This row is numerically F8r-like, but the decisive problem is telemetry:

```text
logical_batch_action_bucket_ema_normalization: missing
logical_batch_action_bucket_scale_mean/min/max: missing
per-bucket gamma_b and EMA counts: missing
```

Root cause:

```text
scripts/picf_core_train.py wrote action-bucket EMA fields into
last_logical_batch_step_info, but the final JSON record only copied the base
logical-batch fields.  Therefore gamma_b could not be audited from
metrics.jsonl.
```

Repair:

```text
All scalar keys with prefix logical_batch_action_bucket_ are now copied into
the final metrics record.
```

Required rerun:

```text
remote_exp=picf_f9b_k4_action_adapter_bucketema_v2_wor_11000_to11100_20260602
same checkpoint and training config as v1
only telemetry/dataflow repair changed
```

F9b v2 result at step11010:

```text
loss_action_default_equiv=0.040664
loss_total_minus_action=0.010257
loss_anchor_pv=0.488384
loss_anchor_object_pull=0.168356
loss_mapg_routing=0.426112
loss_slot_jepa=0.693000
distinct_buckets=4
logical_batch_action_bucket_scale_mean=0.9255
logical_batch_action_bucket_scale_min=0.7611
logical_batch_action_bucket_scale_max=1.0900
```

This proves gamma_b is active and metrics are no longer empty.  However, v2 is
still not the final F9b acceptance run because the scale telemetry was
rank-local:

```text
distinct_bucket_count=4 came from the global logical step;
scale fields only covered the main-rank local micro-windows.
```

Repair:

```text
The trainer now gathers logical_action_bucket_scale_by_bucket across ranks
before computing global mean/min/max and per-bucket scale metrics.
```

Required rerun:

```text
remote_exp=picf_f9b_k4_action_adapter_bucketema_v3_wor_11000_to11100_20260602
same checkpoint and training config as v2
only global scale telemetry repair changed
```

Acceptance:

```text
F9b v3 must show non-missing global gamma_b / EMA fields.
If gamma_b is active and action remains F8r-like at 11020+, dynamic bucket
action scaling is insufficient and the next branch must be adapter-gradient
surgery or a deeper action-boundary change, not another sampler-only run.
```

The 0.90 EMA is intentionally fast for the 100-step gate.  If positive, rerun
with 0.95-0.98 before declaring the production recipe.

F9b v3 step11010 result:

```text
remote_exp=picf_f9b_k4_action_adapter_bucketema_v3_wor_11000_to11100_20260602
step=11010
loss_action_default_equiv=0.040670
loss_action_active7=0.184472
loss_total=0.091306
loss_total_minus_action=0.009966
loss_anchor_pv=0.487524
loss_anchor_object_pull=0.141201
loss_mapg_routing=0.423544
loss_slot_jepa=0.688360
logical_batch_distinct_bucket_count=4
selected buckets=block_other, block_push, slider, switch_button_light
scale_mean=0.982459
scale_min=0.761508
scale_max=1.089727
scale_block_other=1.009608
scale_block_push=1.089727
scale_slider=1.068992
scale_switch_button_light=0.761508
```

Telemetry verdict:

```text
PASS.  v3 correctly gathers per-bucket action scale telemetry across DDP ranks.
The selected-bucket scale fields now match the global selected-bucket set.
```

Optimization verdict at step11010:

```text
Not yet positive.  The action row is still F8r/F9a-like at the first log point.
This is only a dataflow acceptance row; F9b needs step11020 before deciding
whether dynamic bucket action EMA scaling has any real optimization effect.
```

F9b v3 step11020 result:

```text
step=11020
loss_action_default_equiv=0.042547
loss_action_active7=0.193515
loss_total=0.095492
loss_total_minus_action=0.010398
loss_anchor_pv=0.474969
loss_anchor_object_pull=0.175049
loss_mapg_routing=0.417914
loss_slot_jepa=0.681345
logical_batch_distinct_bucket_count=4
selected buckets=block_other, drawer, other, slider
scale_mean=0.985564
scale_min=0.910852
scale_max=1.198483
active_same_role_support_overlap=0.075000
downstream_same_role_support_overlap=0.078443
grad_norm=0.255768
```

F9b verdict:

```text
REJECT AS OPTIMIZATION FIX.

The telemetry and K4 task coverage are correct, and structure metrics remain
healthy, but action does not descend:
  step11010 action=0.040670
  step11020 action=0.042547

This rules out the cheap GradNorm/PiKE-like per-bucket action-scale branch as a
sufficient fix.  The problem is not merely per-bucket scalar loss imbalance.
The remaining measured root is task-bucket gradient direction conflict in the
semantic/action adapter path.
```

Decision:

```text
Stop F9b v3.
Do not repeat sampler-only, freeze-adapter, or scalar bucket-weighting runs.
Next non-duplicate branch:
  adapter-only PCGrad/CAGrad on semantic/action adapter + policy/action head,
  or a stronger Knowledge-Insulation-style action-boundary gate.
```

### F9c Semantic-Only PCGrad Contract

F9c is the next valid non-duplicate test after F9b.

Mathematical target:

```text
Given one logical optimizer step with task buckets b=1..K:

  g_b = ∇_θs L_b

where θs is only the semantic/action adapter trainable group.  F9c replaces the
raw accumulated semantic/action gradient with PCGrad:

  g_i <- g_i - min(0, <g_i,g_j>) / (||g_j||^2 + eps) * g_j
  g   = Σ_i g_i

Only θs is overwritten by g.  PICF-core, structural losses, sidecar/object
losses, and all non-selected parameter groups keep the ordinary backward
gradient.
```

Why this is not a whole-model or ad hoc repair:

```text
The pre-probe measured action-gradient conflict in semantic/action params:
  loss_action semantic negative_fraction ~= 0.476
  min_cosine ~= -0.159

The same probe found no direct action gradient into PICF core and only mild
structural-loss conflict in PICF core:
  loss_total_minus_action picf_core negative_fraction ~= 0.095
  min_cosine ~= -0.0043

Therefore PCGrad is scoped to semantic/action only.  Adding whole-model PCGrad,
PICF-core PCGrad, MoE, or another sampler tweak would not follow the measured
root.
```

Implementation contract:

```text
script:
  scripts/experiments/picf_aqr_owm_202605_active/
    run_a7_f9c_k4_semantic_pcgrad_from11000_11100_20260603.sh

core args:
  --logical-batch-gradient-surgery pcgrad
  --logical-batch-gradient-surgery-groups semantic

kept identical to F8r/F9b:
  K4 task-uniform logical batch
  sample-without-replacement
  per-bucket loss normalization
  semantic_trainable_scope=action_head_and_adapter
  action weight=4.0
  resume step=11000

explicitly disabled:
  logical-batch action bucket EMA scaling
```

Telemetry required before judging optimization:

```text
logical_batch_distinct_bucket_count = 4
logical_batch_gradient_surgery_enabled = true
logical_batch_gradient_surgery_mode_id = 1
logical_batch_gradient_surgery_target_param_tensors > 0
logical_batch_gradient_surgery_local_micro_count = accum_steps on each rank
```

Optimization acceptance:

```text
F9c is positive only if action improves relative to F8r/F9b under the same
step11010-11030 gate.  A clean dataflow row alone is insufficient.

If action remains >= F8r/F9b or worsens by step11020/11030:
  reject adapter PCGrad as insufficient;
  move to explicit Knowledge-Insulation/action-boundary redesign or a stronger
  action expert path, not more sampler-only experiments.
```

### F9c Semantic-Only PCGrad Result

F9c was implemented and tested after F9b.  It is no longer a planned branch.

Implementation verification:

```text
local:
  python -m py_compile scripts/picf_core_train.py scripts/picf_core_train_test.py
  python -m py_compile scripts/picf_calvin_bucket_sampler_audit.py
  python -m py_compile scripts/picf_bucket_gradient_cosine_probe.py
  pytest -q scripts/picf_core_train_test.py -q
  bash -n run_a7_f9c_k4_semantic_pcgrad_from11000_11100_20260603.sh
  git diff --check

remote:
  py_compile passed after sync to /root/openpi_e21b2_1b07eab
  bash -n passed for the F9c launcher
```

FSDP result:

```text
training_strategy=fsdp_full_shard
logical_batch_gradient_surgery=pcgrad
logical_batch_gradient_surgery_groups=semantic
window_activation_checkpointing=on

startup:
  K4 task_uniform logical batch
  sample-without-replacement
  per-bucket normalization on
  gradient_surgery=pcgrad
  target_param_tensors=5
  target_param_numel=4,228,625

failure:
  RuntimeError: setStorage ... storage size 0
  location: activation-checkpoint recompute under FSDP during token_fusion /
            projective_bias_head backward
```

Verdict:

```text
FSDP + activation checkpoint + autograd.grad PCGrad is not a valid production
path in this implementation.  This is an engineering incompatibility, not a
learning-quality result.  Do not use FSDP PCGrad as the acceptance test.
```

DDP/no-checkpoint result:

```text
remote_exp=picf_f9c_k4_semantic_pcgrad_ddp_nochkpt_wor_11000_to11100_20260603
training_strategy=ddp
window_activation_checkpointing=off
step_time~=21.7 sec/step

step=11010
loss_action_default_equiv=0.040648
loss_action_active7=0.184378
loss_total=0.092207
loss_total_minus_action=0.010910
loss_anchor_pv=0.489586
loss_anchor_object_pull=0.227279
loss_mapg_routing=0.431503
loss_slot_jepa=0.692816
logical_batch_distinct_bucket_count=4
selected buckets=block_other, block_push, slider, switch_button_light
logical_batch_gradient_surgery_enabled=true
logical_batch_gradient_surgery_mode_id=1
logical_batch_gradient_surgery_local_micro_count=2
logical_batch_gradient_surgery_target_param_tensors=14
active_same_role_support_overlap=0.137480

step=11020
loss_action_default_equiv=0.042562
loss_action_active7=0.193588
loss_total=0.094678
loss_total_minus_action=0.009553
loss_anchor_pv=0.474841
loss_anchor_object_pull=0.098224
loss_mapg_routing=0.417150
loss_slot_jepa=0.695996
logical_batch_distinct_bucket_count=4
selected buckets=block_other, drawer, other, slider
logical_batch_gradient_surgery_enabled=true
logical_batch_gradient_surgery_target_param_tensors=14
active_same_role_support_overlap remained structurally healthy
```

Comparison:

```text
F9b v3:
  action=0.040670 -> 0.042547

F9c DDP/no-checkpoint:
  action=0.040648 -> 0.042562
```

F9c verdict:

```text
REJECT AS OPTIMIZATION FIX.

F9c proves the adapter-level PCGrad dataflow and telemetry can be made to run
under DDP, but it does not improve action descent relative to F9b/F8r.  The
remaining bottleneck is not solved by sampler-only repair, adapter freeze,
per-bucket scalar normalization, or semantic-adapter PCGrad.
```

Decision after F9c:

```text
Do not repeat:
  sampler-only F8 variants;
  adapter-freeze F9a;
  scalar bucket-action normalization F9b;
  semantic-adapter PCGrad F9c;
  whole-model PCGrad.

Next non-duplicate branch:
  explicit action-boundary / action-expert redesign following
  Knowledge-Insulation / OpenVLA-OFT style separation:
    semantic belief router remains trainable and measurable;
    continuous action gradients must not freely rewrite the semantic trunk;
    action loss should primarily update action expert / bridge / policy head;
    task-balanced logical batches remain required dataflow, not the final fix.
```

### Local Verification Notes

Passed locally:

```text
python -m py_compile scripts/picf_core_train.py scripts/picf_calvin_bucket_sampler_audit.py
bash -n selected active launchers
git diff --check
python scripts/picf_calvin_bucket_sampler_audit.py --help
```

Known local testing caveat:

```text
Directly importing scripts.picf_core_train from an inline Python sanity test
currently trips a local IPython/wandb/pygments dependency issue unrelated to
the training path.  The training script itself compiles, the audit CLI works,
and the remote trainer runs.  Do not use direct module import as the proof for
sampler math in this environment; use the no-model audit script and remote
real-data audits.
```
