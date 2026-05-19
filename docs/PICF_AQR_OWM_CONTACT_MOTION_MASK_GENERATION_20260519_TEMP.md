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
