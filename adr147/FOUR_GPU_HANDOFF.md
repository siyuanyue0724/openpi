# ADR-147 Four-GPU Handoff

## Release decision

The host-independent four-GPU package is releasable only after every checked
item below is backed by an immutable artifact. This is an execution release,
not a scientific-success claim. The two-GPU evidence establishes numerical
health, separated object rows, action preservation, the repaired diagnostic
path, and exact checkpoint recovery. It does not establish closed-loop CALVIN
success or PICF superiority.

The original two-GPU run reached step 2,000, but its optional step-2,000 causal
diagnostic diverged between ranks in an observation-only shortcut and hung in
an NCCL all-gather before the then-later checkpoint call. Consequently there is
no honest step-2,000 checkpoint or step-2,000 diagnostic artifact. That failure
was not relabeled as a pass. The release instead requires both of these
root-cause-specific replacement gates:

1. the repaired production-forward diagnostic must execute all five variants
   on both ranks without rank divergence; and
2. the released 6B/FSDP2 stack must publish a complete DCP checkpoint, cold
   restore model/optimizer/lane/RNG state, and continue with the exact expected
   next-step action loss.

Both replacement gates passed. The four-rank run starts fresh from the same
released LingBot weights; no two-rank optimizer state is reused.

### Post-release diagnostic evaluator addendum, 2026-08-08

The first four-rank production attempt completed optimizer step 250 on all
ranks, then failed closed in the factual diagnostic because training and
evaluation used different physical-track evidence predicates. Training admits
either visible mask evidence or known positive existence; the evaluator had
counted only the latter and then required visible mask mass for every match.
The repaired v4 evaluator shares the exact training assignment domain and
restricts spatial metrics to its observed-mask subset. It does not alter the
model, objective, stream, optimizer, or posterior dynamics.

The focused objective/evaluator suite passes `21`; the complete
`tests/lingbot_native` suite passes `1,383` with `11` declared optional-asset
skips. A four-rank, released-weight, three-step smoke completed all optimizer
steps and factual, zero, wrong-time, cross-batch, and wrong-row variants on all
ranks. Its distributed diagnostic SHA-256 is
`97f7347d12cfe51b99442acd11be15079e043a9e6f21a7ed05f0df9f9eb8b33f`.
The failed 250-step weights were not checkpointed and cannot be resumed; the
production curve must restart from zero.

#### Superseding carried-row addendum

The first v4 repair was necessary but not sufficient. A deterministic restart
reproduced all 250 optimizer records and PNG hashes, then failed at the same
rank-2 factual diagnostic. The real sidecar frame proves that current evidence
tracks were `[1,4,5,6,7,8,9]` while the temporal gauge assigned
`[1,3,4,5,6,7,8,9]`: track 3, `part/table/button_link`, had zero current pixels
in both cameras but was legally carried by the age-57 prior in row 7.

Schema v5 records `carried_rows` provenance from the sequence binding phase.
Evaluation now requires every current-evidence track to be assigned, rejects
any extra unproven track, and permits only proven carried identities to lack
current evidence. Such rows and anonymous reserved rows are excluded from
present-frame spatial/no-object/cardinality denominators. This does not alter
the model or training objective. Focused tests pass `28`; the complete suite
passes `1,385` with `11` optional-asset skips. The data-level receipt is
`rank2-step250-carried-row-evidence.json`. The schema-v5 released-weight smoke
then completed three optimizer updates and all five variants on every rank;
its distributed report SHA-256 is
`3efc72dcd94b5047605c0e01f50d85ad080bf920d91c1c375af474e3ad1042fc`.
The distributed ABI is accepted. The real age-57 step-250 pass remains required
before the production run is accepted.

## Persistent assets

Every unique input required after shutdown is under `/mnt`:

- candidate code: `/mnt/picf-next/worktrees/adr147-fourgpu-candidate-20260808`;
- source: `/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147`;
- LingBot weights: `/mnt/picf-next/models/lingbot-vla-v2-6b`;
- Qwen processor: `/mnt/picf-next/models/qwen3-vl-4b-instruct`;
- CALVIN data: `/mnt/calvin_data/task_ABC_D/training`;
- pinned CALVIN simulator source: `/mnt/picf-next/source-checkouts/calvin-fa03f01-clean`;
- manifests and normalization: `/mnt/picf-next/manifests`;
- physical supervision: `/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z`;
- two-GPU evidence: `/mnt/picf-next/runs/adr147-layerwise-v2-20260807T194207Z`;
- repaired diagnostic evidence: `/mnt/picf-next/runs/adr147-full-diagnostic-path-smoke-step3-fixed-20260808T142729CST`;
- schema-v5 four-rank diagnostic evidence: `/mnt/picf-next/runs/adr147-schema-v5-full-diagnostic-step3-20260808T000655CST`;
- checkpoint/resume evidence: `/mnt/picf-next/runs/adr147-checkpoint-resume-smoke-20260808T143722CST`;
- handoff receipts: `/mnt/picf-next/adr147/four_gpu_handoff_20260808`;
- exact runtime archive: `/mnt/picf-next/runtime-archives/picf-runtime-restore-probe-94305690cafb-20260808.tar`.

The runtime archive is `10,136,064,000` bytes with SHA-256
`46c480491e9b491c573fbe94eb0f4f3bcfdb7b2f9707658f4e1361152304ff82`.
It contains the accepted Python 3.12 environment, PyTorch `2.8.0+cu128`,
Transformers `4.57.3`, and FlashAttention `2.8.3`. The upstream environment has
audited package-metadata conflicts, so do not run an unconstrained resolver.
Restore the exact archive and compare its complete `pip freeze` digest.

## Frozen data and optimizer contract

Four FSDP ranks deliberately imply global batch four. This is not a hidden
continuation of the two-rank optimizer. The stream plan, representation split,
evaluation plan, execution manifest, and checkpoint provenance bind world size,
global batch, source order, seed, learning rate, clipping, and accumulation.

Frozen file hashes are:

- stream plan file: `337cd18ccf47d10341324780c500b2f7c123ae0d36f06dd992df9d377504f08d`;
- representation split file: `f23730957ddf8024a766abd673899cfa4c935ab039c7166f5b3cd0724c6ecb80`;
- evaluation plan file: `8ef2a729bda82d50c3fe47c0682f6b747d9d2ab6cf27e2ee6792a0f7eeb56b60`;
- semantic stream-plan digest: `94fb1e9417aaac7c237f5000343862f2bd260ce582881b97c28831118ab55517`.

The source split contains 73 training trajectories and 73 disjoint evaluation
trajectories; one source trajectory is unused. The training side has 5,260
segments and 312,504 transitions. A 30k four-rank run consumes 120,000 samples
and covers 34 tasks. The evaluation plan has 68 items, exactly 17 per rank.

## Matched LBOT scope

Before the main run, execute one 200-step four-rank official no-PICF LBOT. It
uses the same released weights, frozen training stream, sample order, seed,
global batch, optimizer family, learning rate, clipping, action horizon,
augmentations, flow-noise seeds, and flow-timestep seeds.

The LBOT held-out evaluations at steps 0/20/100/200 are baseline sanity evidence;
they are not directly paired with a candidate held-out snapshot. The valid
candidate-versus-LBOT comparison is the exact training prefix through step 200.
`compare_matched_lbot_action_prefix.py` rejects either run unless every rank-step
matches sample keys, frame indices, lane IDs, reset flags, source digest,
augmentation seeds, flow-noise seeds, and flow-timestep seeds. It reports action
loss at equal consumed samples in 50-step windows. This is an early optimization
comparison only, not rollout success or long-horizon superiority.

## Evidence receipts

- Two-GPU run: exactly 2,000 finite records per rank, mean action loss
  `0.26322`, mean entity loss `0.66114`, mean step time `15.1947 s`, and peak
  reserved memory `38.822 GiB`.
- Step-2,000 images: task-relevant blue rows have existence `0.926/0.945` and
  soft-IoU `0.346/0.317`; unmatched rows remain low. Rows are separated, but
  boundaries and tiny parts are not solved.
- Repaired five-variant diagnostic: distributed report SHA-256
  `1391ce384a7db3d2c1211cb2c21142145042a1fde2e0a74e7de613d4adc80cf8`.
  Both ranks completed factual, zero, wrong-time, cross-batch, and wrong-row
  using the official production forward. Three steps validate execution, not
  causal capability; the causal directions are mixed and remain a 30k gate.
- Checkpoint/cold-resume report SHA-256:
  `7a0cf3f6b206d46f2b936959d61a68c197155cc3a2b3b362a4fef0c445998df0`.
  The published checkpoint had 69 files and 51,705,291,693 bytes. After a cold
  process restart, model/optimizer/lane/RNG state restored and resumed step-2
  action losses exactly matched the uninterrupted run (`0.462890625` and
  `0.55078125`).
- CALVIN simulator source and runtime: clean root commit `fa03f01f19c65920e18cf37398a9ce859274af76`,
  clean environment commit `1431a46bd36bde5903fb6345e68b5ccc30def666`,
  with pinned pybullet, Hydra, Gym, cloudpickle, and quaternion imports passing.
- Final host-independent suite: `1,379 passed, 11 skipped`; candidate-scoped
  Ruff, format, shell syntax, and diff checks pass. The skips are optional
  relative-path source probes. The persistent LingBot checkout separately
  passes `25` source tests, and the persistent CALVIN source has its own clean
  commit/dependency/runtime receipt.

## Launch order

On the new four-A100 host:

```bash
cd /mnt/picf-next/worktrees/adr147-fourgpu-candidate-20260808

./adr147/restore_four_gpu_runtime.sh

LBOT_RUN=/mnt/picf-next/runs/adr147-lbot-4gpu-$(date +%Y%m%dT%H%M%S)
./adr147/run_matched_lbot_4gpu.sh "$LBOT_RUN"

MAIN_RUN=/mnt/picf-next/runs/adr147-layerwise-v2-4gpu-$(date +%Y%m%dT%H%M%S)
./adr147/launch_four_gpu_30k.sh \
  "$MAIN_RUN" \
  "$LBOT_RUN/official_lbot_steps_200.json"
```

The main launcher fails closed unless the LBOT report is PASS, no-PICF, four-rank,
200-step, and bound to the frozen stream, seed, learning rate, and clipping.
The production run's first 20 steps are its preflight and remain part of the
same 30k run. Metrics publish every 100 steps, visuals and causal diagnostics
every 250, and checkpoints every 2,000. After main step 200, run:

```bash
/opt/picf-runtime-restore-probe-94305690cafb/bin/python \
  adr147/compare_matched_lbot_action_prefix.py \
  --baseline-report "$LBOT_RUN/official_lbot_steps_200.json" \
  --candidate-run-dir "$MAIN_RUN" \
  --steps 200 \
  --window-size 50 \
  --output "$MAIN_RUN/matched_lbot_action_prefix_step200.json"
```

Stop only on registered integrity, non-finite, OOM, distributed divergence,
representation-collapse, repeated causal-reversal, or sustained action-
regression failures. A weak single sample is inspected, not silently promoted
to a new architectural patch.

## Checklist

- [x] Two-GPU step-2,000 numerical/action/visual evidence is complete.
- [x] Original-resolution step-2,000 images are inspected.
- [x] The old step-2,000 diagnostic/checkpoint failure is recorded, not hidden.
- [x] The repaired full-forward five-variant path passes on two released 6B ranks.
- [x] Full released 6B/FSDP2 checkpoint publication and cold resume pass.
- [x] Four-rank frozen stream, split, and evaluation plans are published.
- [x] Runtime archive and independently checked SHA-256 receipt are published.
- [x] Final candidate patch, source/assets receipt, and handoff manifest pass.
- [x] New host reports exactly four visible A100 40GB devices.
- [x] Official 200-step matched LBOT passes.
- [x] Production first-20-step in-run preflight passes.
- [x] Step-200 exact-stream LBOT comparison is published.
- [x] Step-250 anchor panels are copied locally and inspected with image vision.
- [x] Schema-v5 four-rank diagnostic smoke passes all five variants on all ranks.
- [ ] The real step-250 causal boundary passes without evaluator divergence.
- [ ] Step-2,000 checkpoint and scientific gate are accepted.

The remaining items are live four-GPU execution gates and are not yet
claimable.
