# PICF-AQR-OWM May 2026 Archive Index

Date: 2026-05-18

This directory is an index for the May 2026 PICF-AQR-OWM repair and experiment
burst. Most source documents intentionally remain in `docs/` because README
links, verifier strings, and experiment notes point to their exact paths. Treat
the files below as historical evidence unless they are explicitly referenced by
`src/openpi/picf/README_v2.2.md` or
`docs/PICF_AQR_OWM_CURRENT_STATE_20260518.md`.

## Historical TEMP Documents

The May 2026 `docs/PICF_AQR_OWM*_TEMP.md` files fall into these buckets:

- Dataflow/math audits: recursive follow-through, binding math, owner gate,
  posterior file competition, birth transport, task owner routing.
- Experiment notes: loss audits, A7/A5 cloud diagnostics, sidecar/SAM trial
  notes, remote CALVIN audit.
- Theory notes: slot theory gap analysis, advanced slot audit, vNext/MVTrack
  design.
- Deployment/status notes: final deployment, strict diagnosis, MVTrack deep
  audit, current issue tracker.

Do not delete these until:

1. Any README links to them have been migrated.
2. Any verifier or audit script string references have been updated.
3. The corresponding conclusion has been copied into the current state page or
   the issue tracker.

## Rejected Blind-SAM Branch

Blind automatic SAM proposal memory is archived under:

```text
docs/archive/picf_aqr_owm_202605/sam_rejected_20260519/
scripts/archive/picf_sam_proposal_precompute_legacy.py
scripts/archive/picf_sam_proposal_dataflow_audit_legacy.py
```

Maintained conclusion:

```text
Do not use blind automatic SAM for current PICF-AQR-OWM training.
It was tested and rejected because class-agnostic masks frequently selected
wall panels, robot protrusions, drawer sides, and other causally irrelevant
fragments. The generic proposal_* schema remains available only for inspected
contact/task/tracklet-aware sidecars.
```

The training CLI rejects sidecar roots that look like legacy blind-SAM outputs
unless `--allow-legacy-blind-sam-sidecar` is passed for historical
reproduction.

## Archived Run Scripts

Ad-hoc cloud launch scripts from the repo root were moved to:

```text
scripts/experiments/picf_aqr_owm_202605_archive/
```

These scripts are preserved for provenance but are not canonical launch entry
points. Current launch recipes should live in:

- `src/openpi/picf/README_v2.2.md`
- `docs/CALVIN_VALIDATION_README.md`
- `docs/PICF_AQR_OWM_CURRENT_STATE_20260518.md`

## Cleanup Policy

- Prefer archive/index over deletion for documents that explain why a branch is
  disabled.
- Prefer deletion or physical archive for one-off root-level run scripts.
- Prefer code removal only after a guarded branch has no current tests, no
  README coverage, and no planned data-dependent use.
