# Rejected Blind-SAM Proposal Branch

Date: 2026-05-19

This directory preserves historical notes for the blind automatic SAM proposal
experiments. These files are not current operator guidance.

Maintained decision:

```text
Blind automatic SAM is rejected for current PICF-AQR-OWM training.
Do not launch production runs or anchor diagnostics with blind-SAM sidecar
roots.
```

Reason:

```text
CALVIN diagnostics showed that class-agnostic automatic SAM masks/boxes often
selected wall panels, robot protrusions, drawer sides, or other causally
irrelevant fragments. That behavior conflicts with PICF's posterior-centered
belief model because false proposal evidence can push anchor geometry away from
the task object.
```

Retained interface:

```text
proposal_* remains a generic optional sidecar schema.
It may be used only for inspected contact/task/tracklet-aware proposal sources.
It is never posterior truth and never replaces dense V-JEPA/PG/point/tactile
memory.
```

Historical reproduction:

```text
scripts/archive/picf_sam_proposal_precompute_legacy.py
scripts/archive/picf_sam_proposal_dataflow_audit_legacy.py
```

The training CLI rejects sidecar roots that look like archived blind-SAM outputs
unless `--allow-legacy-blind-sam-sidecar` is explicitly passed.
