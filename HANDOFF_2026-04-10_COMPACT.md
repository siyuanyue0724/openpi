# PICF Compact Handoff 2026-04-10

This is a temporary handoff document for continuity after context compaction.

It duplicates project state that already exists across code, commits, logs, and README files. If the project state becomes stable and this handoff is no longer needed, this file should be deleted later to avoid drift.

## Repo State

- Local repo: `/home/siyuanyue/Documents/openpi`
- Branch: `Posterior_VLA`
- This file is intentionally temporary and may need to be deleted later once its content is fully reflected in the main README / plan docs.

## Current PICF Contract

- `PaliGemma` semantic stream preserves full valid image/text semantic token stream.
- `semantic_dim = 2048`
- `world hidden_dim = 256`
- `current posterior` remains language-free.
- Semantic information enters only after posterior/world state is fixed.
- Fusion is heterogeneous and posterior-late:
  - `Q_world -> KV_semantic`
- Semantic does not write back into:
  - posterior cache
  - carried prior
  - physical innovation base
- Predictive path is split:
  - `physical_prediction_cache` is produced before semantic reads
  - semantic-conditioned `prediction_cache` is produced after semantic reads
- Next-step innovation reads only `previous.predictive.physical_prediction_cache`.
- `semantic_summary` is bookkeeping / diagnostics only, not the main downstream fusion input.
- `L_pt` no longer creates unconditional positives on missing-contact data.
  It uses:
  - explicit contact if available
  - otherwise a tactile-history pseudo-contact gate
- Sonata high-risk indexing paths now have runtime bounds checks to turn opaque CUDA `ScatterGatherKernel` failures into earlier, explicit Python errors when possible.

## Key Code Anchors

- Prior input and carried state contract:
  - `src/openpi/picf/core/pipeline.py`
- Physical posterior update without semantic input:
  - `src/openpi/picf/core/pipeline.py`
- Innovation reading only physical prediction cache:
  - `src/openpi/picf/core/pipeline.py`
- Predictive split between physical cache and semantic-conditioned cache:
  - `src/openpi/picf/core/pipeline.py`
- Predictive state contract:
  - `src/openpi/picf/core/contracts.py`

The most important exact references are:

- `src/openpi/picf/core/pipeline.py#L1404`
- `src/openpi/picf/core/pipeline.py#L1458`
- `src/openpi/picf/core/pipeline.py#L1742`
- `src/openpi/picf/core/pipeline.py#L1796`
- `src/openpi/picf/core/contracts.py#L83`

## Regression Coverage

The following behaviors are covered by tests:

- semantic changes do not alter current posterior
- semantic changes do not alter `physical_prediction_cache`
- semantic changes do not alter next-step innovation base
- `semantic_summary` does not drive the main downstream readout
- prior/context uses `executed_action`

Important test anchors:

- `src/openpi/picf/core/pipeline_test.py#L231`
- `src/openpi/picf/core/pipeline_test.py#L299`
- `src/openpi/picf/core/pipeline_test.py#L348`
- `src/openpi/picf/core/pipeline_test.py#L433`

## Verification Status

Current verification level is strong engineering verification, not formal proof.

What is done:

- code-path audit against the intended mathematical contract
- local regression tests
- cloud regression tests
- local smoke training
- cloud dual-GPU smoke training
- real cloud long-run training in progress

What is not done:

- no Coq / Lean proof
- no TLA+ or model checking artifact
- no machine-checked state-transition proof

So the correct statement is:

- implementation is currently consistent with the intended mathematical contract
- no hidden semantic write-back edge has been found
- this is not a formal proof of absolute correctness

## Training-Time Visualization

Training now saves diagnostics every `500` steps.

Run currently configured:

- experiment: `picf_core_train_ddp_fulltoken_run_r5`
- log interval: `100`
- save interval: `5000`
- diagnostic interval: `500`

Diagnostics are saved under:

- `/mnt/checkpoints/picf_core/picf_core_train_ddp_fulltoken_run_r5/diagnostics/<step>/`

Expected artifacts include:

- `gt_static_t*.png`
- `pred_physical_t*.png`
- `pred_semantic_t*.png`
- `gt_window_static.gif`
- `pred_physical_window_static.gif`
- `pred_semantic_window_static.gif`
- `compare_grid.png`
- `metadata.json`

These are training-window diagnostics, not CALVIN rollout evaluator videos.

## Current Cloud Training

Cloud host:

- `ssh -p 28829 root@px-cloud2.matpool.com`

Current training process:

- experiment: `picf_core_train_ddp_fulltoken_run_r5`
- command is running under `torchrun` with 2 GPUs
- current cloud repo HEAD is also `43935e9`

Latest checked state at handoff time:

- training process alive
- recent metrics written through step `500`
- first diagnostics directory has been written:
  - `/mnt/checkpoints/picf_core/picf_core_train_ddp_fulltoken_run_r5/diagnostics/000500`
- GPU memory usage near limit but still stable:
  - `GPU0: 39055 / 40960 MiB`
  - `GPU1: 39083 / 40960 MiB`

## Useful Commands

Watch main training log:

```bash
tail -f /mnt/checkpoints/picf_core/logs/picf_core_train_ddp_fulltoken_run_r5.log
```

Watch structured metrics:

```bash
tail -f /mnt/checkpoints/picf_core/picf_core_train_ddp_fulltoken_run_r5/metrics.jsonl
```

Watch monitor log:

```bash
tail -f /mnt/checkpoints/picf_core/logs/picf_core_train_ddp_fulltoken_run_r5.monitor.log
```

List saved diagnostics directories:

```bash
find /mnt/checkpoints/picf_core/picf_core_train_ddp_fulltoken_run_r5/diagnostics -maxdepth 1 -mindepth 1 -type d | sort | tail
```

Check training processes:

```bash
ps -ef | grep picf_core_train.py | grep -v grep
```

Check GPU state:

```bash
nvidia-smi
```

## Documents To Keep Consistent

If architecture contracts change later, update all of:

- `src/openpi/picf/README.md`
- `docs/calvin_readme.txt`
- `plan_readme_ray_geometry.md`
- this file, only if it is still intentionally kept

## Recommended Next Phase

Move on to a formal audit/specification phase instead of more ad hoc narrative checking.

Recommended order:

1. Write explicit state-transition invariants.
2. Define forbidden edges:
   - semantic -> posterior
   - semantic -> carried prior
   - semantic-conditioned cache -> next innovation base
3. Define allowed edges:
   - world query -> semantic memory
   - semantic -> action/future readout only after posterior is fixed
4. Decide whether to express the contract in:
   - a precise operator contract document
   - assertions/property tests
   - TLA+ / proof-oriented artifact later

## Local Workspace Note

At handoff time, local git status contains only untracked:

- `.codex`
- `TEMP_REPO/`

No tracked local code edits are pending.
