# OEML Math/Dataflow Follow-Through

Object Explanation Measurement Layer: code-level mathematical and dataflow
audit.

The layer is explicitly defined over dense typed evidence rather than sparse
proposal labels.

Date: 2026-05-19

Status: code-level deployment audit for the Object Explanation Measurement
Layer (OEML). This document is not a CALVIN behavior acceptance report.

## Why This Exists

Recent object-centric learning code does not rely on a bag of independent
overlap penalties. The recurring invariant is:

```text
dense features -> slots explain tokens with object/background masks -> object
measurements -> recurrent slot state
```

The local paper-code caches inspected for this audit show the same structure:

```text
object-centric-learning-framework / AdaSlot:
  feature -> slotz/attent -> recon, with feature reconstruction losses.

slot-attention-video / SAVi:
  corrected slots -> predicted slots -> decoder masks and reconstructions.

MetaSlot:
  adaptive/duplicate slot masks, no-object suppression, and slot competition.

DINO / Deformable-DETR:
  object queries need reference evidence and denoising/reference structure;
  raw fixed queries do not reliably discover object ownership by themselves.
```

PICF already has a posterior-centered belief filter and typed evidence router.
The gap was that AQR support rows could attend to evidence without being judged
by whether each dense token is explained by a coherent object or background
residual. OEML fills that gap while preserving posterior authority.

## Mathematical Contract

For each typed memory `m` with priors `p^m_{j,i}`, object quality `q_j`, and
background prior `rho_bg`, OEML constructs token-column ownership:

```math
r^m_{j,i}
=
\frac{q_j p^m_{j,i}}
{\rho_{bg} + \sum_l q_l p^m_{l,i}}
```

```math
r^m_{bg,i}
=
\frac{\rho_{bg}}
{\rho_{bg} + \sum_l q_l p^m_{l,i}}
```

Therefore:

```math
\sum_j r^m_{j,i} + r^m_{bg,i} = 1
```

This is the slot/OCL decoder-mask invariant adapted to PICF. It does not prune
V-JEPA tokens and does not treat SAM/proposal sidecars as ground truth.

Feature explanation quality is:

```math
\mu^m_j =
\frac{\sum_i r^m_{j,i} z^m_i}
{\sum_i r^m_{j,i}}
```

```math
L_{feat,j}
=
\frac{\sum_i r^m_{j,i}(1-\cos(z^m_i,\mu^m_j))}
{\sum_i r^m_{j,i}}
```

Point compactness uses object-owned 3D points:

```math
x_j =
\frac{\sum_i r^{point}_{j,i} x_i}{\sum_i r^{point}_{j,i}}
```

```math
L_{point,j}
=
\frac{\sum_i r^{point}_{j,i}\|x_i-x_j\|^2}
{\sigma^2 \sum_i r^{point}_{j,i}}
```

Duplicate evidence is measured as same-role mask cosine:

```math
D_{j,k}
=
\frac{\langle r_j, r_k\rangle}
{\|r_j\|\|r_k\|}
```

The guarded duplicate loss is:

```math
L_{dup} = E_{same-role}[\max(0, D_{j,k}-\delta)^2]
```

Contact explanation checks whether object masks explain tactile/contact tokens:

```math
score_{contact}
=
\frac{\sum_i c_i \min(1, \sum_j r^{tactile}_{j,i})}
{\sum_i c_i}
```

Loss weights are explicit and default to zero:

```text
lambda_object_explanation_feature = 0
lambda_object_explanation_point = 0
lambda_object_explanation_contact = 0
lambda_object_explanation_duplicate = 0
lambda_object_explanation_background = 0
```

So OEML is deployed as runtime measurement and diagnostics by default; training
pressure must be intentionally enabled for a validation run.

## Dataflow Follow-Through

Current code-level dataflow after deployment:

```text
PicfObservation
  -> token_field
       visual / temporal V-JEPA / point / tactile / tracklet / proposal tokens
  -> _build_aqr_anchor_graph
       graph.visual_priors / point_priors / tactile_priors / temporal_priors ...
  -> _build_object_explanation_measurements
       object/background masks per typed memory
       anchor_quality
       duplicate_overlap
       feature/point/contact diagnostics
       graph.object_explanation_quality
  -> _mapg_slot_assignment
       assignment score *= object_explanation_quality
  -> _build_observation_anchors
  -> _posterior_update
  -> _build_task_readout / _build_conditioned_control_state
  -> PI0.5 action path
```

The crucial invariant is that OEML can downweight poor measurements before
posterior binding, but it never overwrites posterior state and never removes
dense evidence from the action path.

## Self-Critique

This is not a magic object-label generator. It cannot guarantee "fourth object"
when the typed evidence contains no separable signal. It also does not make
blind SAM proposals production truth. It fixes the specific architectural gap:
slots must explain evidence against a background residual before they strengthen
posterior binding.

Behavior acceptance still requires a fresh run with:

```text
oeml_anchor_quality_mean/max
oeml_duplicate_overlap_max/mean
oeml_feature_variance_mean
oeml_point_spatial_variance_mean
oeml_contact_explanation_score
active same-role object-core overlap
posterior recycle / file competition metrics
anchor overlays
CALVIN/video evidence
```
