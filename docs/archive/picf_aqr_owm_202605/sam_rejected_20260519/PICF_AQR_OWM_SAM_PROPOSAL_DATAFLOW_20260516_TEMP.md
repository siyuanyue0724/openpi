# PICF-AQR-OWM SAM/Proposal Dataflow Audit

Date: 2026-05-16
Status: code-level offline proposal sidecar deployed; online behavior still requires a proposal-precomputed run.

This document records the strict SAM/proposal design decision for PICF-AQR-OWM.
It is linked from `src/openpi/picf/README_v2.2.md` and should be read as an
operator-facing contract, not as a claim that CALVIN currently contains proposal
tokens. Current long-run metrics still show `owm_proposal_tokens=0` unless a
sidecar root is supplied.

## 1. Architectural Decision

SAM/SAM2/SAM3-style segmentation is not inserted into the differentiable PICF
training loop. The maintained design is:

```text
frozen/offline object proposal generator
-> per-frame sidecar npz proposal_* arrays
-> PicfObservation optional fields
-> PicfPseudoProposalState typed memory
-> residual-gated AQR proposal reader
-> graph.proposal_priors
-> observation/posterior proposal_signature diagnostics
```

This preserves the three PICF invariants:

```text
1. dense V-JEPA/PG/point/tactile tokens remain available;
2. proposals are additional typed evidence, not hard truth;
3. posterior correction remains authoritative.
```

## 2. Why Not Online SAM

Online SAM inside the train step would be a category error for this architecture.
It would add heavy non-PICF dependencies to the recurrent belief filter, slow
the run, and blur the boundary between observation evidence and trainable
policy state. More importantly, SAM masks are proposals, not labels. Treating
them as hard object identities would create false certainty and could lock
posterior slots onto segmentation artifacts.

The correct mathematical role is a proposal likelihood term:

```math
M^{prop}_t = \{(b_m, c_m, o_m, v_m, s_m)\}_{m=1}^{N_p}
```

where `b_m` is a normalized box, `c_m` is a center, `o_m` is proposal quality,
`v_m` is view id, and `s_m` is source id. PICF maps this to proposal tokens:

```math
z^{prop}_m =
W_p[\ c_m,\ b_m,\ o_m,\ area_m,\ view_m,\ source_m,\ \phi(c_m)\ ]
+ E_{view}(v_m) + E_{modality}
```

AQR then reads them as one typed branch:

```math
q'_j =
q_j + \lambda_{prop}\left(
\operatorname{Attn}(q_j, M^{prop}_t) - q_j
\right)
```

This is a residual evidence read. It cannot overwrite posterior identity by
itself, and it cannot discard dense visual tokens.

## 3. Paper/Code Comparison

The code snapshots used for this audit were cloned to `/tmp/picf_sam_code`:

```text
segment-anything:
  segment_anything/automatic_mask_generator.py

sam2:
  sam2/sam2_image_predictor.py

openmask3d:
  openmask3d/compute_features_single_scene.py

open3dis:
  open3dis/src/mapper.py
  tools/generate_3d_inst.py
```

Observed external dataflows:

```text
SAM:
  image -> automatic masks -> bbox + predicted_iou + stability_score

SAM2:
  image/video predictor -> point/box prompted masks -> multi-object masks

OpenMask3D:
  3D masks + point cloud + images -> per-mask features

Open3DIS:
  2D proposals + 3D proposals + depth-aware point-image mapping -> final masks
```

The PICF-compatible interpretation is:

```text
SAM/SAM2:
  generate RGB object proposals for static/gripper views.

OpenMask3D/Open3DIS:
  optional future path for lifting proposals into point/3D object evidence
  when calibrated depth/point data is available.

PICF:
  consumes normalized proposal boxes and quality scores as typed evidence.
```

## 4. Current Repository Follow-Through

### Observation Contract

`src/openpi/picf/contracts.py` exposes:

```text
proposal_centers_xy
proposal_boxes_xyxy
proposal_objectness
proposal_view_ids
proposal_source_ids
```

### Core Contract

`src/openpi/picf/core/contracts.py` exposes:

```text
PicfPseudoProposalState
PicfTokenFieldState.proposal
PicfAnchorPriorGraphState.proposal_priors
PicfPosteriorAnchorState.proposal_signature
```

### Pipeline

`src/openpi/picf/core/pipeline.py` converts proposal fields into typed tokens,
applies objectness validity, reads them through `aqr_proposal_reader`, and keeps
the read residual-gated by `proposal_read_weight`.

### Train/Replay/Serve

`src/openpi/picf/replay/calvin_replay.py`, `scripts/picf_core_train.py`, and
`scripts/serve_picf_policy.py` pass proposal arrays when present.

### New Sidecar Root

`--mvtrack-sidecar-root` now lets train/replay merge optional per-frame sidecar
npz fields without changing CALVIN data:

```text
<sidecar_root>/<split>/episode_0000000.npz
<sidecar_root>/episode_0000000.npz
```

This is the missing engineering bridge from offline SAM proposal generation to
the current training run.

### Offline Precompute Script

`scripts/picf_sam_proposal_precompute.py` runs a frozen SAM automatic mask
generator and writes proposal sidecars:

```bash
python scripts/picf_sam_proposal_precompute.py \
  --calvin-root /mnt/calvin/task_ABCD_D.zip \
  --backend zip \
  --split training \
  --output-root /mnt/picf_sidecars/sam_proposals \
  --segment-anything-repo /tmp/picf_sam_code/segment-anything \
  --sam-checkpoint /mnt/models/sam_vit_h_4b8939.pth \
  --sam-model-type vit_h \
  --views both
```

Then train with:

```bash
python scripts/picf_core_train.py ... \
  --proposal-memory-enabled \
  --mvtrack-sidecar-root /mnt/picf_sidecars/sam_proposals
```

## 5. Objectness and Filtering

For each SAM mask, objectness is:

```math
o_m = \sqrt{\max(iou_m,0)\max(stability_m,0)}
```

This rejects masks where only decoder confidence or threshold stability is high.
Boxes are normalized to `[0,1]`; centers are derived from the normalized box.

PICF still applies:

```text
proposal_confidence_floor
proposal_max_tokens
proposal_read_weight
```

These are evidence-quality gates, not labels.

## 6. Relation to Current Anchor Problems

The current failure mode has been:

```text
action loss can fall;
active anchors may remain acceptable;
raw/context/reserve anchors often overlap;
posterior identity switch remains high;
proposal tokens are production-default zero after blind-SAM diagnostics.
```

SAM proposal memory addresses only one missing evidence source: external object
boundary/objectness candidates. The blind automatic version is now opt-in only,
because A7 diagnostics showed task-owner evidence remained weak and raw overlap
could worsen even when averaged proposal tokens were nonzero. It does not by
itself solve:

```text
1. task-conditioned posterior selection;
2. tactile touched-object assignment;
3. ordinal/rank grounding;
4. identity continuity without temporal proposals/tracklets.
```

Therefore proposal evidence should be deployed as frozen typed evidence only in
explicit ablations or prompted/reranked proposal runs, and evaluated by:

```text
owm_proposal_tokens > 0
aqr_proposal_support_entropy_mean
aqr_proposal_support_max
posterior proposal_signature norm
active same-role overlap
identity switch rate
anchor overlays with and without gray/context anchors
```

## 7. Hard Guards

SAM/proposal integration is invalid if any of these are violated:

```text
1. online core imports segment_anything/sam2 directly;
2. dense visual tokens are removed because proposals exist;
3. proposal masks overwrite posterior slots;
4. proposal loss is treated as supervised object identity without labels;
5. missing proposal sidecars cause non-proposal runs to fail.
```

`scripts/picf_sam_proposal_dataflow_audit.py --fail-on-fail` checks these
contracts, plus the external code patterns when `/tmp/picf_sam_code` is present.

## 8. Verification Commands

```bash
python -m py_compile \
  scripts/picf_sam_proposal_precompute.py \
  scripts/picf_sam_proposal_dataflow_audit.py \
  scripts/picf_core_train.py \
  src/openpi/picf/replay/calvin_replay.py

python scripts/picf_sam_proposal_dataflow_audit.py \
  --external-code-root /tmp/picf_sam_code \
  --fail-on-fail
```

## 9. Open Boundaries

This deployment is complete for proposal sidecar dataflow, but not a claim that
SAM has fixed behavior. Blind SAM proposal influence is production-default off:
`owm_proposal_tokens=0` is now expected unless `--proposal-memory-enabled` and
nonzero proposal read/bridge weights are explicitly supplied. A prompted or
reranked proposal-precomputed CALVIN run is required before claiming behavioral
improvement.
