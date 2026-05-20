# PICF Contact-Motion Mask Sidecar Generation

Status: active production-prep note for the 2026-05-19 long-run gate.

This note is linked from `src/openpi/picf/README_v2.2.md`.  It records the
sidecar generation contract that must run before the next 30000-step PICF run.

## Decision

Do not start the 30000-step run from the 1000-frame diagnostic sidecar root:

```text
/mnt/picf_sidecars/contact_motion_mask_1000_20260518
```

That root is valid for diagnostics only.  It contains about 1000 frames from a
small set of CALVIN language segments.  A long run with random segment sampling
would see a mixture of:

```text
frames with inspected contact/task proposal masks
frames with nearest-age fallback proposal masks
frames with no proposal evidence
```

Such a run would confound model behavior with sidecar coverage.  Before the
next long run, generate a larger contact-motion proposal sidecar root and train
with `--calvin-segment-indices $(cat <root>/calvin_segment_indices.txt)` so the
sampling distribution is explicitly tied to the sidecar coverage.

## Mathematical Role

The sidecar is not a hard object label.  It is a weak measurement source:

```math
o_t \rightarrow m^{proposal}_{t,p}
```

where `p` indexes contact/task proposal candidates.  PICF then performs:

```math
m^{proposal}_{t,p}
\rightarrow
E_{j,p}
\rightarrow
A_{j,p}, B_p
\rightarrow
posterior correction
```

with:

```text
A_{j,p}: candidate p explained by physical slot j
B_p:     candidate p absorbed by background/no-object residual
```

The sidecar must therefore satisfy only a measurement-quality contract:

```text
1. proposals are task/contact/motion biased;
2. masks are compact enough to give useful support samples;
3. proposal evidence is soft and may be rejected by background;
4. no blind SAM root is used for current training;
5. future information is not introduced into the online observation path.
```

The maintained generator is:

```text
scripts/picf_contact_motion_sidecar_precompute.py
```

Default mode is current-frame causal scoring.  The diagnostic
`--use-segment-ee-path` option must stay off for the production-prep sidecar
unless a separate future-leakage ablation is explicitly documented.

## Production-Prep Root

The next root is:

```text
/mnt/picf_sidecars/contact_motion_mask_12000_20260519
```

The first pass targets 12000 frames with up to 96 frames per language segment.
The value is intentionally larger than the previous 1000-frame diagnostic while
remaining bounded enough to finish before the long-run launch.  If runtime is
acceptable, the same command can be resumed with a larger `--target-frames`
because the generator is atomic and uses `--skip-existing`.

## A7 Command

```bash
cd /root/openpi_posterior_vla_clean
export PYTHONPATH=scripts:src:${PYTHONPATH:-}
PY_BIN=${PY_BIN:-/usr/bin/python}
"${PY_BIN}" scripts/picf_contact_motion_sidecar_precompute.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --output-root /mnt/picf_sidecars/contact_motion_mask_12000_20260519 \
  --split training \
  --target-frames 12000 \
  --max-frames-per-segment 96 \
  --static-stride 4 \
  --gripper-stride 2 \
  --top-fraction 0.020 \
  --min-top-points 24 \
  --min-score 0.015 \
  --box-pad-px 4.0 \
  --max-proposals-per-frame 3 \
  --component-radius-px 10.0 \
  --component-min-points 6 \
  --box-percentile-low 12 \
  --box-percentile-high 88 \
  --mask-samples-per-proposal 96 \
  --preview-count 128 \
  --skip-existing
```

Run it in tmux as:

```text
picf_a7_contact_motion_mask12000_20260519
```

Log:

```text
/mnt/picf_run_logs/picf_a7_contact_motion_mask12000_20260519.log
```

## Acceptance Gates

Before using the root in a long run:

```bash
find /mnt/picf_sidecars/contact_motion_mask_12000_20260519/training -name '*.npz' | wc -l
cat /mnt/picf_sidecars/contact_motion_mask_12000_20260519/manifest.json | head
ls /mnt/picf_sidecars/contact_motion_mask_12000_20260519/previews | head
```

Required:

```text
written_frames is close to target_frames;
calvin_segment_indices.txt is non-empty;
preview masks show compact task/contact regions, not blind SAM fragments;
manifest causal_current_frame_default is true;
the long-run command passes the generated calvin_segment_indices.txt content.
```

## Long-Run Hook

The long-run command must use:

```bash
--mvtrack-sidecar-root /mnt/picf_sidecars/contact_motion_mask_12000_20260519
--mvtrack-sidecar-proposal-nearest-max-gap 8
--calvin-segment-indices "$(cat /mnt/picf_sidecars/contact_motion_mask_12000_20260519/calvin_segment_indices.txt)"
```

The nearest fallback is age-aware weak evidence.  It is not a substitute for
sidecar coverage; it only prevents exact-frame sparsity from silently becoming
`proposal_tokens=0` on nearby frames.

## 2026-05-20 Full-Sidecar Preparation Update

The 1000-frame root remains diagnostic only.  The A7 inspection found a larger
proposal-mask root already present:

```text
/mnt/picf_sidecars/contact_motion_full_20260519
  proposal/mask npz files: about 199k
  fields: proposal_centers_xy, proposal_boxes_xyxy,
          proposal_mask_xy, proposal_mask_weights, proposal_mask_offsets,
          proposal_objectness, proposal_view_ids, proposal_source_ids
  missing before repair: calvin_segment_indices.txt and long-run manifest
```

The preparation script is:

```text
scripts/picf_prepare_full_sidecar_root.py
```

It is metadata-only and CPU/IO-only.  It scans the sidecar split directory,
maps covered `episode_XXXXXXX.npz` ids back to CALVIN language segments, writes:

```text
calvin_segment_indices.txt
long_run_manifest.json
```

and verifies sampled proposal/mask/tracklet keys.  This is the required gate
before a long run can claim that sampling is tied to sidecar coverage.

Important 2026-05-20 audit fix: CALVIN `auto_lang_ann.npy` stores
`info["indx"]` in a shuffled order.  Segment coverage must therefore use
per-interval binary search over sorted sidecar episode ids.  A monotonic cursor
undercounts coverage and can turn a large proposal/mask root into a false
12-segment root.

Tracklets must be generated or merged into a clean output root separately.  The
maintained tracklet generator is:

```text
scripts/picf_tracklet_sidecar_precompute.py
```

Important 2026-05-20 repair: when this generator merges a proposal root into a
tracklet output root, it must copy the sparse mask keys too:

```text
proposal_mask_xy
proposal_mask_weights
proposal_mask_offsets
```

Without this, a "full tracklet" sidecar silently loses the owner-mask evidence
that the current slot contract depends on.

The clean production root should therefore be prepared in two stages:

```text
Stage A:
  prepare /mnt/picf_sidecars/contact_motion_full_20260519
  require proposal + mask keys
  no GPU required
  expected runtime: minutes, dominated by directory scan

Stage B:
  generate clean KLT tracklets seeded by the Stage-A proposal root into:
    /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
  no GPU required
  expected runtime: hours, dominated by CPU image decode + OpenCV KLT + IO
  run sharded across CPU processes

Stage C:
  prepare/audit the Stage-B root
  require proposal + mask + tracklet keys
  train with:
    --mvtrack-sidecar-root /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
    --calvin-segment-indices "$(cat /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520/calvin_segment_indices.txt)"
```

Do not use `tracklets_samseed_*` as the clean production root.  Those roots are
useful historical evidence that the dataflow can carry tracklets, but their
seed provenance is not the current contact-motion contract.

## 2026-05-20 Clean Full-Root Generation Status

Current A7 production-prep run:

```text
tmux session:
  picf_a7_full_tracklets_clean_20260520

proposal/mask input root:
  /mnt/picf_sidecars/contact_motion_full_20260519

clean proposal+mask+tracklet output root:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520

shards:
  8 CPU/OpenCV workers

GPU:
  not used
```

The output root was intentionally empty before launch.  Old diagnostic or
rejected roots were removed from the top-level `/mnt/picf_sidecars` namespace
so future launch scripts cannot accidentally pick a dirty root.  The only
top-level roots retained for this contract are:

```text
/mnt/picf_sidecars/contact_motion_full_20260519
/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
```

Progress log caveat:

```text
scripts/picf_tracklet_sidecar_precompute.py emits per-shard
`progress_segments`, not `segments_logged`.

Total progress = sum(progress_segments across shard logs).
```

At 2026-05-21 00:24 CST, the clean run had completed:

```text
shards:
  8 / 8 done

output:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520

npz files:
  240,758
```

The first strict physical-key gate deliberately failed:

```text
tracklet_required_present_fraction:
  1.0

proposal_required_present_fraction:
  about 0.82

proposal_mask_present_fraction:
  about 0.82
```

This is not a tracklet failure.  It reflects the sparse proposal contract:
contact/task proposal sidecars are only emitted when the motion/contact scorer
finds a credible object-region mask.  Many frames still have KLT tracklets but
no current-frame proposal.  The correct training dataflow is:

```text
current frame:
  read current tracklet rows

if current proposal/mask is absent:
  borrow nearest non-empty proposal/mask within
  --mvtrack-sidecar-proposal-nearest-max-gap

then:
  set proposal_age = temporal_gap
  let proposal_age_decay_steps downweight stale evidence
```

Do not materialize empty proposal keys as a shortcut unless the reader also
continues nearest-proposal search for zero-length proposals.  Empty keys alone
can silently block fallback and reduce proposal evidence.

The maintained final gate after generation is now coverage-aware rather than
key-presence-only:

```bash
PYTHONPATH=src python scripts/picf_prepare_full_sidecar_root.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --sidecar-root /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520 \
  --split training \
  --require-tracklet \
  --proposal-nearest-max-gap 8 \
  --min-proposal-nonempty-fraction 0.80 \
  --min-proposal-reachable-fraction 0.85 \
  --min-mask-reachable-fraction 0.85 \
  --sample-limit 1024
```

Only after that command passes may a long training run use this root.

Observed gate result at 2026-05-21 00:36 CST:

```text
ok:
  true

npz_files:
  240,758

covered_segments:
  7,869

tracklet_required_present_fraction:
  1.0

proposal_nonempty_fraction:
  0.822265625

proposal_reachable_fraction with gap=8:
  0.87890625

proposal_mask_reachable_fraction with gap=8:
  0.87890625
```

The training reader was also repaired so a current tracklet-only sidecar file
does not block nearest non-empty proposal/mask borrowing.  This is required
because the full root contains valid current tracklets on more frames than it
contains current sparse proposal masks.

## 2026-05-21 Strict Quality Audit

Purpose:

```text
Answer whether the full sidecar root is clean enough for the accepted 30K run,
and whether the earlier missing proposal/mask fields are understood.
```

Remote root audited:

```text
source proposal/mask root:
  /mnt/picf_sidecars/contact_motion_full_20260519

clean merged root:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520
```

Machine-checkable result:

```text
source proposal files:
  199,339

clean merged files:
  240,758

sampled clean files:
  8,192 deterministic evenly-spaced files

load errors:
  0

tracklet keys present:
  8,192 / 8,192 = 1.0

proposal keys present:
  6,741 / 8,192 = 0.8228759765625

mask keys present:
  6,741 / 8,192 = 0.8228759765625

missing proposal although source file exists:
  0

proposal present although source file absent:
  0

finite failures:
  0

range failures:
  0

box order failures:
  0

mask offset failures:
  0

center outside box:
  0

tracklet shape/range failures:
  0
```

This proves the missing proposal/mask fields are not a merge bug: the output
root exactly preserves the source proposal root when proposal evidence exists,
and adds KLT tracklets on additional frames.  Therefore the correct data
contract is sparse proposal/mask plus dense-ish tracklet evidence.

Distribution checks:

```text
proposal area:
  p01=0.00519, p50=0.02017, p95=0.03646, max=0.06610

proposal aspect:
  p01=0.571, p50=1.143, p95=2.000, p99=2.562

objectness:
  p01=0.0169, p50=0.5724, p95=0.7300, p99=0.7579

mask samples per proposal:
  p01=6, p50=38, p95=50, max=50

tracklets per file:
  p01=47, p50=63, p95=64, max=64

nearest proposal/mask gap histogram over sampled files:
  gap0=6741, gap1=68, gap2=66, gap3=58, gap4=48,
  gap5=46, gap6=43, gap7=36, gap8=42
```

Mask/box relationship:

```text
mask_inside_box_fraction:
  p01=0.80, p05=0.844, p50=0.958, p95=1.0

mask weighted mean vs proposal center, normalized by box diagonal:
  p50=1.19e-05, p95=8.15e-05, max=7.19e-04
```

Interpretation:

```text
1. The weighted mask center and proposal center are numerically aligned.
2. Some mask sample tails fall outside the green box because the box is a
   compact percentile support box, not a hard instance-mask hull.
3. The sparse mask samples are the primary supervision signal; the box is
   visualization / seeding / coarse geometry.
4. These sidecars are acceptable as weak owner/contact scaffold, not as
   pixel-perfect instance segmentation labels.
```

Visual audit artifacts:

```text
/mnt/picf_sidecar_quality_audit_20260521/
  mask_overlay_contact_sheet.png
  episode_*.png
  audit_visual_summary.json
```

Visual inspection of the contact sheet shows mostly compact contact/motion
regions.  Some boxes include gripper plus object edge, which is expected for
contact-generated weak masks.  This would be too noisy for hard semantic mask
supervision, but it is consistent with the current weak owner-transport and
proposal-age-decayed scaffold.
