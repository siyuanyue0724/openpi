# PICF-AQR-OWM 5/6-Card FSDP E21 Reproduction Decision

Date: 2026-06-03

This note answers one narrow question:

```text
Can 5-6 A100-40GB cards plus FSDP reproduce the E21 action-descent condition
before an 8-card machine is available?
```

## Short Answer

```text
6 cards:
  possible in principle, not proven yet.
  The right target is world_size=6, accum_steps=2, K=12 logical windows/update.
  This matches the E21 window count, but only after the FSDP+accum2 startup path
  reaches Training config and emits first-step metrics.

5 cards:
  useful diagnostic only.
  world_size=5, accum_steps=2 gives K=10, which is below E21 K=12.
  It may improve over K6, but it is not a strict E21 reproduction.

FSDP:
  necessary for 5/6-card production-scale attempts, but not sufficient by
  itself.  FSDP reduces parameter/gradient/optimizer-state pressure; it does
  not make K6 mathematically equivalent to K12 and it does not remove per-rank
  activation cost.
```

Therefore:

```text
Do not run another long K5/K6 accum1 experiment as an E21 reproduction.
Run a 6-card K12 startup gate first.  If it passes, run a 100-300 step K12 gate.
Only then decide whether a 30K follow-through is justified.
```

## Evidence Already Established

### 6-card K6 result

Archived run:

```text
picf_6x40g_e23_bucketbalanced_noactioncond_from11000_30k_20260602
world_size=6
accum_steps=1
effective windows/update=6
```

Result:

```text
step11050 action_default_equiv = 0.0440368
step11800 action_default_equiv = 0.0404126
step11850 action_default_equiv = 0.0433774

active/downstream support overlap stayed healthy.
loss_total_minus_action stayed small.
action did not reproduce E21-fast descent.
```

Conclusion:

```text
K6 improves task coverage relative to K2, but K6 is not enough.
This is not evidence against the model; it is evidence against treating
world_size=6, accum=1 as the E21 equivalent.
```

### Previous 6-card K12 attempt

Attempted run:

```text
picf_6x40g_e21like_accum2_windowckpt_noactioncond_from11000_30k_20260602
world_size=6
accum_steps=2
window_activation_checkpointing=1
training_strategy=fsdp_full_shard
```

Observed:

```text
did not reach Training config;
no loss rows;
control without window checkpointing also did not reach Training config;
standalone torchrun rank sanity passed on the host.
```

Conclusion:

```text
This was a startup-path failure, not an action-loss result and not an OOM
training result.  It cannot be used to reject K12.
```

Current code is not identical to that failed attempt.  The trainer now has:

```text
_build_model_sequential_across_ranks(...) no longer serializes model build;
optimizer_checkpoint_mode=model-only genuinely skips optimizer state;
logical-batch task-count/normalization/metrics flags;
phase/startup debug logging hooks.
```

This makes a new K12 startup gate worth running.

## Mathematical Contract

The desired multi-task action objective is:

```text
L(theta) = sum_b q_b E[ell(x; theta) | bucket(x)=b]
G(theta) = sum_b q_b E[grad ell(x; theta) | bucket(x)=b]
```

For one optimizer step with selected micro-windows `m`, bucket `b(m)`, and
within-step bucket count `n_b`, the task-normalized estimator is:

```text
L_step = sum_m [q_{b(m)} / n_{b(m)}] * ell_m
```

In DDP, gradients are averaged across ranks, so the local backward scale is:

```text
scale_local = world_size * q_b / n_b
```

E21-fast behavior requires not just larger GPU count but a logical optimizer
step whose selected buckets approximate the intended task mixture.  With six
cards:

```text
world_size=6, accum_steps=1 -> K=6
world_size=6, accum_steps=2 -> K=12
```

Only the second is a real E21-window-count reproduction.

## Why FSDP Alone Is Not Enough

FSDP helps with:

```text
parameter sharding;
gradient sharding;
optimizer-state sharding;
model-only resume on changed optimizer groups.
```

FSDP does not automatically solve:

```text
per-rank activation memory for one full transition window;
token_fusion attention backward/recompute memory;
extra live gradient state during accum_steps > 1;
startup barriers or checkpoint-load compatibility;
the mathematical need for K>=10-12 task coverage.
```

So the question is not:

```text
"Can FSDP make 6 cards enough?"
```

The correct question is:

```text
"Can our current FSDP path start and train world_size=6, accum_steps=2,
strict logical-batch K12 without activation OOM or startup deadlock?"
```

## 5-Card Interpretation

5 cards can run:

```text
world_size=5, accum_steps=2 -> K=10
```

This may be worth a short diagnostic if 6-card K12 is blocked, because K10
covers all seven coarse CALVIN buckets in one logical update with some
duplicates.  But it is not strict E21:

```text
K10 < K12
fewer task samples/update
same accum2 startup/memory risk
less direct comparability to E21
```

Maintained K10 launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_5x40g_e21like_accum2_k10_startup_gate_20260603.sh
```

5 cards can also run a slower, more conservative coverage gate:

```text
world_size=5, accum_steps=3 -> K=15
```

This exceeds E21's K12 window count.  It should be used only after K10 proves
startup/dataflow health, because each optimizer update performs three
micro-forwards/backwards per rank.  It uses FSDP synchronization on every
accumulation micro-step to reduce the accumulated-gradient memory peak on 40GB
cards.

Maintained K15 launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_5x40g_e21like_accum3_k15_syncmicro_startup_gate_20260603.sh
```

5-card decision rule:

```text
1. Run K10 first for 50-100 resumed steps.
2. If K10 fails startup/OOM, do not try K15; wait for 6/8 cards or reduce model
   memory through a separate precompute project.
3. If K10 starts but action remains plateaued while structure is healthy, test
   K15 for 20-50 resumed steps.
4. Do not treat K10 as strict E21 evidence; it is a lower-coverage diagnostic.
5. Treat K15 as a slower coverage stress test, not as a free speedup.
```

## Required Gate Sequence

### Gate 1: 6-card K12 startup

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/run_6x40g_e21like_startup_gate_20260603.sh
```

Default contract:

```text
NPROC_PER_NODE=6
ACCUM_STEPS=2
LOGICAL_BATCH_TASK_COUNT=12
LOGICAL_BATCH_BUCKET_NORMALIZATION=1
LOGICAL_BATCH_LOG_BUCKET_METRICS=1
WINDOW_ACTIVATION_CHECKPOINTING=1
OPTIMIZER_CHECKPOINT_MODE=model-only
NUM_TRAIN_STEPS=11002
```

Pass:

```text
Training config is printed;
resume succeeds;
step11001 or step11002 metric row exists;
logical_batch_enabled=true;
logical_batch_global_micro_count=12;
logical_batch_distinct_bucket_count is high, ideally covering all available
coarse task buckets;
no NaN/OOM/DDP ready-twice failure.
```

Fail:

```text
no Training config -> startup/FSDP/resume bug, not learning evidence;
Training config but no first metric -> forward/backward/memory issue;
first metric but K<12 fields -> contract bug.
```

### Gate 2: 6-card K12 100-300 step quality gate

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/run_6x40g_e21like_accum2_windowckpt_noactioncond_from11000_30k_20260602.sh
```

Use a short absolute target first:

```text
NUM_TRAIN_STEPS=11300
SAVE_INTERVAL=1000
KEEP_LAST_CHECKPOINTS=2
LOG_INTERVAL=50
ANCHOR_OVERLAY_INTERVAL=100
```

Acceptance:

```text
action_default_equiv should beat the K6 interval around 0.040-0.043;
active/downstream overlap should remain healthy;
loss_total_minus_action should not explode;
per-bucket metrics should not show one task family dominating all action loss.
```

### Gate 3: only then consider 30K

Run 30K only if Gate 2 shows action descent.  If K12 still plateaus near K6,
the remaining blocker is not just card count/FSDP.  Reopen the action-readout
objective or context bridge instead of spending a long run.

## Decision Table

```text
6 cards, accum1, K6:
  already tested; not enough; do not repeat as E21 reproduction.

6 cards, accum2, K12:
  best available pre-8-card route; must pass startup gate first.

5 cards, accum2, K10:
  fallback diagnostic; not strict reproduction.

8 cards, accum1, K8:
  safer startup than accum2, but still below K12.

8 cards, accum2, K16:
  stronger than E21 window count, but inherits accum2 startup/memory risk.
```

## 4-Card Slow K12 Option

4 cards can match the E21 window count by using:

```text
world_size=4
accum_steps=3
effective windows/update=12
```

This is mathematically valid only if the three micro-batches per rank are
accumulated before a single `optimizer.step()`.  Running two or three ordinary
optimizer steps is not equivalent, because AdamW moments and parameters change
between steps.

The maintained low-card route is:

```text
FSDP full shard
window activation checkpointing
FSDP sync on every accumulation micro-step
optimizer step once per K12 logical update
```

The extra FSDP sync is deliberate.  It trades speed for lower accumulation
memory: non-final micro-steps do not sit inside FSDP `no_sync()`, so gradients
can be reduced/sharded each micro-step while AdamW still updates once after all
12 logical windows.

Launchers:

```text
scripts/experiments/picf_aqr_owm_202605_active/
  run_4x40g_e21like_accum3_syncmicro_startup_gate_20260603.sh

scripts/experiments/picf_aqr_owm_202605_active/
  run_4x40g_e21like_accum3_syncmicro_30k_20260603.sh
```

Expected tradeoff:

```text
more communication than FSDP no_sync accumulation;
roughly 3 micro-forwards/backwards per optimizer step per rank;
lower peak accumulation memory than unsynchronized FSDP no_sync;
same K12 estimator as E21 if the gate emits logical_batch_global_micro_count=12.
```

Run order:

```text
1. Run the 4-card startup gate to step11002.
2. If it emits valid K12 metrics, run a 100-300 step quality gate.
3. If action improves toward E21 and structure stays healthy, use it while
   waiting for 8 cards.
4. If startup or memory fails, do not keep weakening the model semantics; wait
   for 6/8 cards or add frozen-feature precompute as a separate speed/memory
   project.
```

## Maintained Conclusion

```text
FSDP can make 6-card E21-scale training possible, but it has not yet proved it.
The current correct move is a short K12 startup gate, not another K6 long run.
If the startup gate passes, 6-card K12 is worth a 100-300 step quality gate.
If that passes, it is the only 5/6-card route that can honestly be called an
E21 reproduction attempt.
```
