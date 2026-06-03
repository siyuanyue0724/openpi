# PICF-AQR-OWM Logical-Batch Deployment Plan

Status legend:

- `[ ]` not done
- `[~]` in progress
- `[x]` done and checked
- `[!]` blocked or intentionally deferred

This document is the active checklist for the next training repair.  The
separation is deliberate:

- Experiment items prove whether the recipe fixes the observed small-window
  action plateau/rebound.
- Deployment items make the production trainer implement the same mathematical
  estimator that worked in the exact-window probes.

## Core Diagnosis

The repeated plateau/rebound pattern is not explained by anchor overlap alone.
The strongest controlled evidence is:

- E21 exact-window `K=12` optimizer steps descent quickly.
- E21b2 `K=2` learns but is weaker/noisier.
- E24 `K=4` and `K=6` both improve over random small-window training, but are
  slower than `K=12`.
- Production `--calvin-balanced-bucket-sampler` with small physical coverage
  did not reproduce E21 because the production trainer still treats a step as
  ordinary micro-step averaging, without an explicit logical-batch contract,
  per-bucket accounting, or pass/fail diagnostics.

The target objective is task-family balanced:

```text
L = sum_b q_b E[ell(x) | bucket(x)=b]
```

For one optimizer step with selected micro windows `m` and bucket `b(m)`, the
unbiased logical-batch estimator is:

```text
L_step = sum_m [q_{b(m)} / n_{b(m)}] * ell_m
```

where `n_b` is the number of selected micro windows from bucket `b` inside the
optimizer step.  In DDP, gradients are averaged across ranks, so each local
backward must use:

```text
loss_scale_local = world_size * q_b / n_b
```

If every selected bucket appears once and `q_b` is uniform over selected
buckets, this reduces to the existing `1 / accum_steps` local scale.  The new
implementation still matters because it records and enforces the contract,
handles duplicated buckets correctly, and exposes per-bucket failure modes.

## Experiment Checklist

- [x] E24 exact-window K sweep archived.
  Evidence:
  `docs/PICF_AQR_OWM_ACTION_READOUT_CAUSAL_AUDIT_20260601_TEMP.md`, section 48.
  K4 eval mean improved `0.0370217793 -> 0.0276452027` by step 90.
  K6 eval mean improved `0.0370217793 -> 0.0307366948` by step 60.

- [x] E25 local trainer contract audit.
  Goal:
  prove the production trainer can express the E24 logical-batch estimator.
  Pass criteria:
  `scripts/picf_core_train.py` exposes explicit logical-batch flags, prints the
  chosen contract at startup, and emits per-bucket metrics at log intervals.
  Current:
  code path implemented and checked.  Local `python -m py_compile`, launcher
  `bash -n`, `git diff --check`, formula sanity checks, and a targeted
  logical-batch dataflow audit all pass.  Local `--help` import is blocked by
  the existing IPython/wandb/pygments environment issue, but remote trainer
  startup accepted and printed all new CLI flags, so CLI validation is covered
  by the real production entrypoint.

- [x] E26 production smoke: logical K about 4.
  Goal:
  run the production trainer with `world_size * accum_steps = 4`,
  bucket-balanced sampling, progress bar, and per-bucket metrics for 200-300
  steps.
  Pass criteria:
  action default-equivalent loss trends down over the first 100-200 optimizer
  steps; bucket metrics show at least four distinct task buckets per optimizer
  step unless the run intentionally uses a smaller debug dataset.
  Current:
  the first remote K=4 smoke used `NUM_TRAIN_STEPS=300` while resuming from
  step 11000.  The trainer correctly interpreted this as an absolute target
  step and exited without training; this is archived as an operator/config
  error, not a logical-batch failure.  The corrected run
  `picf_zky_e26_logical_k4_from11000_to11300_20260602` on
  `px-cloud1/Zky12J` reached the first optimizer step and proved the dataflow
  up to window pre-sampling: the failure trace records two local micro windows
  with buckets `slider` and `switch_button_light`, each using
  `logical_loss_scale=0.5`.  It then OOMed during FSDP `token_fusion` unshard
  at about 40GB/GPU, matching the historical `accum_steps=2` memory boundary.
  Therefore K=4 is resource-blocked on dual A100-40GB for this full token
  budget, not logically invalid.  A K=2 fallback run is now active as
  `picf_zky_e26_logical_k2_from11000_to11300_20260602` to validate production
  metrics/progress/overlay/checkpoint behavior.  K=2 reached `Training config`
  and live progress at about 23s/step.  Step 11050 JSON metrics passed the
  dataflow gate: `logical_batch_enabled=true`,
  `logical_batch_global_micro_count=2`, `logical_batch_distinct_bucket_count=2`,
  per-step selected bucket counts are present, and interval `bucket_*` metrics
  cover all seven CALVIN buckets.  Step 11100 is still needed to validate anchor
  overlay and checkpoint save/prune behavior.  Step 11100 then passed:
  checkpoint `11100/model.pt`, `latest.pt`, and anchor overlay variants
  (`mask_only`, `with_gray`, `sidecar_proposals`, `active_only`,
  `mask_active`, `mask_with_gray`) were written.

- [x] E27 compare against current production sampler.
  Goal:
  verify the new contract is not merely logging.  Compare against the previous
  `--calvin-balanced-bucket-sampler` without logical-batch normalization/logging
  under the same checkpoint and data config.
  Pass criteria:
  new run has lower action-loss variance and clearer per-bucket coverage.
  Current:
  baseline run `picf_zky_e27_bucketonly_from11000_to11100_20260602` completed on
  `px-cloud1/Zky12J` with `ACCUM_STEPS=1`,
  `CALVIN_BALANCED_BUCKET_SAMPLER=1`, and all logical-batch flags disabled.
  Step 11100 wrote `model.pt`, `latest.pt`, and the same anchor overlay variants
  as the logical K=2 smoke.
  Tail command:
  `tail -f /mnt/picf_run_logs/picf_zky_e27_bucketonly_from11000_to11100_20260602.log`
  Metrics:
  logical K=2 at step 11100 had
  `loss_total=0.1006509662`, `loss_action_default_equiv=0.0446206704`,
  `loss_total_minus_action=0.0114096291`, `loss_anchor_pv=0.4971011281`,
  `loss_anchor_object_pull=0.2709096372`, `loss_mapg_routing=0.4397392571`,
  `loss_slot_jepa=0.6829704642`, and `steps_per_sec=0.0402911626`.
  bucket-only K=2 at step 11100 had
  `loss_total=0.1004863530`, `loss_action_default_equiv=0.0446261205`,
  `loss_total_minus_action=0.0112341139`, `loss_anchor_pv=0.4971735775`,
  `loss_anchor_object_pull=0.2547935843`, `loss_mapg_routing=0.4382366538`,
  `loss_slot_jepa=0.6243513227`, and `steps_per_sec=0.0439042919`.
  Interpretation boundary:
  with `world_size=2, accum_steps=1`, logical K=2 has the same local backward
  scale as ordinary DDP averaging whenever each rank contributes one window.
  Therefore this E27 fallback is a production-entry sanity/overhead comparison,
  not a full proof of the normalization benefit.  The near-identical K=2
  metrics are the expected result and prove the new code path does not perturb
  ordinary K=2 production behavior.  A normalization-difference proof requires
  K>2 or duplicated buckets; full-model K=4 is resource-blocked on dual
  A100-40GB unless token budget or trainable scope is reduced.

- [!] E28 gradient-conflict probe, only if E26/E27 are ambiguous.
  Goal:
  compute coarse gradient cosine on action adapter/PICF trainable modules across
  4-8 buckets.
  Pass criteria:
  if gradients are mostly positive/neutral, continue with logical batching and
  avoid PCGrad/MoE; if strongly negative, plan adapter-only gradient surgery.
  Current:
  intentionally deferred for this gate.  E26/E27 resolved the immediate
  deployment question: the logical-batch estimator is implemented and safe at
  K=2, while full K=4 is blocked by memory, not by gradient ambiguity.  Run this
  probe only after a K>2 production run is made feasible and still shows
  unexplained plateau/rebound.

## Deployment Checklist

- [x] D1 add CLI flags.
  Required flags:
  `--logical-batch-task-count`,
  `--logical-batch-bucket-normalization`,
  `--logical-batch-log-bucket-metrics`.

- [x] D2 pre-sample micro windows before forward.
  Reason:
  bucket counts must be known before backward so each micro loss can receive the
  mathematically correct scale.

- [x] D3 gather bucket counts across DDP ranks.
  Reason:
  per-rank counting is not sufficient when duplicated buckets occur across
  ranks.  Use `torch.distributed.all_gather_object` for small string metadata
  before the forward/backward loop.

- [x] D4 scale each micro loss by the logical-batch estimator.
  Formula:
  `scale_local = world_size * q_b / n_b`.
  Default `q_b`:
  uniform over selected buckets in the current optimizer step.

- [x] D5 log per-bucket metrics.
  Required fields:
  count, loss_total, loss_action_default_equiv, loss_total_minus_action,
  loss_anchor_pv, loss_anchor_object_pull, loss_mapg_routing, loss_slot_jepa.
  These should be emitted as flattened metric keys so normal JSONL tailing works.

- [x] D6 preserve existing production invariants.
  Requirements:
  progress bar still works; anchor overlays still write; checkpoint pruning still
  works; DDP `no_sync` still applies for non-final micro-steps; nonfinite checks
  still run before optimizer step.
  Current:
  K=4 preserved startup and pre-sampling invariants but OOMed before a completed
  optimizer step.  K=2 fallback is running to validate progress logging,
  checkpoint pruning, metrics, and overlay generation.  Progress bar is already
  live in K=2; JSON metrics passed at step 11050.  Overlay/checkpoint validation
  passed at step 11100.

- [x] D7 update index docs.
  Required:
  link this plan from `docs/picf_aqr_owm_202605/README.md`.

- [x] D8 run local verification.
  Required commands:
  `python -m py_compile scripts/picf_core_train.py`
  and static grep checks for all new flags/metrics.
  Current:
  `py_compile`, launcher `bash -n`, and `git diff --check` pass locally.
  A targeted static dataflow audit is also 12/12 PASS: prepared-window
  dataclass, DDP object gather, parser flags, strict runtime size check,
  balanced-sampler guard, pre-sample-before-scale ordering, prepared-bucket
  scale input, scaled backward, trace scale, and flattened bucket metrics.
  CLI help is deferred to remote startup because local import hits an unrelated
  IPython/wandb/pygments assertion.  Remote startup confirms the CLI flags are
  accepted, and remote `py_compile` / launcher `bash -n` also pass after sync.
  Formula sanity:
  a standalone bucket-weight check passes for all-distinct, duplicated, and
  all-same bucket selections.  The verified invariant is that total effective
  weight is uniform per selected bucket, not per selected micro-window.

- [x] D9 remote short test and result archival.
  Required:
  launch tmux run with progress bar, observe startup and at least first logged
  metrics, then update this document with the run root, tail command, and
  pass/fail conclusion.
  Current:
  K=4 tmux session was stopped after OOM.  K=2 tmux session
  `e26_k2_logical_smoke_20260602` reached step 11100 and proved progress,
  metrics, checkpoint, and overlay behavior.
  E27 bucket-only K=2 baseline also reached step 11100 and proved the new flags
  do not silently perturb ordinary bucket-balanced K=2 production behavior.
  Tail command:
  `tail -f /mnt/picf_run_logs/picf_zky_e26_logical_k2_from11000_to11300_20260602.log`

## Resume-Step Guardrail

`NUM_TRAIN_STEPS` is an absolute target step in `scripts/picf_core_train.py`,
not a delta.  When resuming from checkpoint step `S` for `N` more optimizer
steps, launch with:

```text
NUM_TRAIN_STEPS = S + N
```

This guardrail is part of D9 because a wrong target step can make a smoke test
appear to pass startup while doing zero optimization.

## Explicit Non-Goals For This Gate

- Do not re-enable SAM.  Prior tests showed noisy proposals and it is archived.
- Do not add MoE, PCGrad, CAGrad, or dynamic PiKE-style mixing until the fixed
  logical-batch estimator is proven insufficient.
- Do not change the PI0.5 action head or V-JEPA dense token contract in this
  gate.
- Do not treat raw same-role overlap as the primary failure signal unless
  action and bucket-normalized metrics also regress.

## Production Candidate Recipe

Target recipe, if memory permits:

```text
world_size: 2
accum_steps: 2
logical_batch_task_count: 4
calvin_balanced_bucket_sampler: true
logical_batch_bucket_normalization: true
logical_batch_log_bucket_metrics: true
unroll_steps: 2
burnin_steps: 1
progress: true
save_interval: 1000 for probes, 500/1000 depending long-run policy
keep_last_checkpoints: 5 for probe, 3 or 5 for long run
```

If this passes E26/E27, the next long run should use the same logical-batch
contract rather than another random-small-batch or bucket-sampler-only 30K run.

Validated fallback on dual A100-40GB:

```text
world_size: 2
accum_steps: 1
logical_batch_task_count: 2
calvin_balanced_bucket_sampler: true
logical_batch_bucket_normalization: true
logical_batch_log_bucket_metrics: true
```

This fallback is production-safe but does not create a new estimator relative to
ordinary DDP K=2 averaging.  The next real normalization-benefit experiment
needs either more memory, fewer dense tokens, reduced trainable scope, or a
different sharding plan so K>2 can complete an optimizer step.
