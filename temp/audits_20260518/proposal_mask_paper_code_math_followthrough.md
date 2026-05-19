# Proposal-Mask Sidecar Paper-Code / Math / Dataflow Follow-Through

Date: 2026-05-18

Scope: audit whether PICF's contact-motion proposal sidecar should be a large
box, a hard object label, or mask-like typed evidence.  This file is a local
TEMP audit, not a user-facing README.

## 1. Current PICF Dataflow

Training / sidecar path:

```text
CALVIN rgb_static + depth_static + robot state + language
  -> scripts/picf_contact_motion_sidecar_precompute.py
  -> contact/task foreground score over static-view 3D points
  -> connected components in static image coordinates
  -> proposal_centers_xy / proposal_boxes_xyxy / proposal_objectness
  -> proposal_mask_xy / proposal_mask_weights / proposal_mask_offsets
  -> .npz sidecar per frame
```

Runtime path:

```text
sidecar .npz
  -> scripts/picf_core_train.py::_MVTRACK_PROPOSAL_KEYS
  -> PicfObservation(proposal_*)
  -> PicfTokenFieldState.proposal
  -> PicfFullCore._proposal_to_point_matrix()
  -> proposal-aware point prior + proposal reader
  -> AQR anchor graph
  -> posterior correction
```

Serve / replay path:

```text
scripts/serve_picf_policy.py reads the same proposal_* keys.
```

Important invariant:

```text
The sidecar never writes posterior, never overwrites dense V-JEPA/PG/point
tokens, and never stores point-row ids. It is measurement evidence only.
```

## 2. Paper-Code Evidence

### SAM

Local repo:

```text
temp/paper_code_20260518/segment-anything
```

Relevant implementation:

```text
segment_anything/modeling/prompt_encoder.py
```

Observed code structure:

```text
_embed_boxes(boxes): boxes become sparse prompt embeddings.
_embed_masks(masks): masks become dense prompt embeddings.
forward(points, boxes, masks): returns sparse embeddings for points/boxes and
dense embeddings for masks.
```

Consequence for PICF:

```text
Using a box as the sole strong target is weaker than carrying mask-like support.
But blind SAM masks are not reliable enough to be posterior truth.  The correct
mapping is: proposal masks are dense measurement prompts/evidence.
```

### Deformable DETR

Local repo:

```text
temp/paper_code_20260518/Deformable-DETR
```

Relevant implementation:

```text
models/ops/modules/ms_deform_attn.py
models/deformable_transformer.py
```

Observed code structure:

```text
MSDeformAttn.forward(query, reference_points, input_flatten, ...)
  reference_points[..., 2] are sparse local centers.
  reference_points[..., 4] are local reference boxes.
  sampling_locations = reference_points + learned offsets.

Decoder updates reference_points iteratively and detaches refined references
between layers.
```

Consequence for PICF:

```text
Reference boxes/points should guide local evidence gathering; they should not
replace the dense token field.  PICF's mask-to-point bridge is aligned with this:
it biases sparse local point support while leaving the dense point/V-JEPA/PG
memory available.
```

### DINO / DN-DETR Line

Local repo:

```text
temp/paper_code_20260518/DINO
```

Relevant implementation:

```text
models/dino/deformable_transformer.py
```

Observed code structure:

```text
refpoint_embed represents decoder reference boxes/points.
two-stage top-k proposals can initialize refpoint/tgt.
denoising queries are concatenated with regular queries during training.
reference points are iteratively refined in the decoder.
```

Consequence for PICF:

```text
The mature pattern is not "all anchors attend everywhere equally forever"; it is
query/reference-localized evidence gathering plus iterative correction.  PICF's
proposal masks should therefore act as reference support for AQR / point priors,
not as labels that bypass posterior.
```

## 3. PICF Mathematical Contract

For proposal k, sidecar stores a compact CSR mask:

```math
\mathcal S_k = \{(u_s, w_s)\}_{s=o_k}^{o_{k+1}-1},
\quad u_s \in [0,1]^2, \quad w_s \ge 0
```

For a current point token i with static-view projected coordinate:

```math
u_i = \frac{1}{2}(g_i + 1), \quad g_i \in [-1,1]^2
```

The mask likelihood is:

```math
L_i^{mask}(k)
=
\frac{
  \sum_{s=o_k}^{o_{k+1}-1}
  w_s \exp(-\|u_i-u_s\|^2/(2\tau_m^2))
}{
  \sum_{s=o_k}^{o_{k+1}-1} w_s + \epsilon
}
```

If no mask samples are present, use the bounded soft-box fallback:

```math
L_i^{box}(k)
=
\sigma((x_i-x^0_k)/\tau_b)
\sigma((x^1_k-x_i)/\tau_b)
\sigma((y_i-y^0_k)/\tau_b)
\sigma((y^1_k-y_i)/\tau_b)
```

After visibility/depth and objectness gates:

```math
\tilde L_i(k) = L_i(k) \; v_i \; q_k
```

The proposal-to-point bridge is normalized per proposal:

```math
P(i|k) =
\frac{\tilde L_i(k)}{\sum_{i'} \tilde L_{i'}(k) + \epsilon}
```

This is a measurement prior, not a posterior assignment:

```math
b_t = Correction(Prediction(b_{t-1}, a_{t-1}), Measurement(o_t, P(i|k)))
```

## 4. Why Not a Hard Mask / Why Not Store Point Row Ids

Hard-mask failure mode:

```text
If the sidecar is noisy, a hard target would force posterior to explain a wrong
object.  This is exactly the failure observed with blind SAM proposals.
```

Point-row-id failure mode:

```text
PICF point rows depend on sampling, static/gripper fusion, max point count,
stride, and training/eval profile.  A sidecar row id would silently become stale.
```

Therefore:

```text
Store normalized image-space mask samples. Re-evaluate the evidence against
current point tokens at runtime.
```

## 5. Multi-Module Interaction Check

Dense V-JEPA:

```text
Unchanged. Proposal masks do not prune or replace video tokens.
```

PaliGemma:

```text
Unchanged. PG image support remains first-class typed evidence.
```

Point / Sonata:

```text
Receives proposal-to-point likelihood as a soft support bridge. The original
projective point geometry is still authoritative for coordinates.
```

AnyTouch:

```text
Unchanged. Tactile contact can later intersect with proposal masks, but this
patch does not fake tactile object identity.
```

Posterior:

```text
Unchanged authority. Proposal evidence can influence measurement routing but
does not directly set posterior slot identity or position.
```

Action:

```text
Unchanged in anchor-only diagnostics. The sidecar should be evaluated by anchor
health first, then by action in a later co-training run.
```

## 6. Verification Already Run

Local:

```text
python -m py_compile ... PASS
uv run pytest -q scripts/picf_contact_motion_sidecar_precompute_test.py \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py
  -> 6 passed
uv run python scripts/verify_picf_owm_contract.py
  -> PASS
synthetic _proposal_to_point_matrix audit
  -> near-mask point receives higher likelihood than far point
```

Remote A7 smoke:

```text
/mnt/picf_sidecars/contact_motion_mask_smoke_20260518
  8 sidecar frames generated
  proposal_mask_xy / weights / offsets present
```

## 7. Remaining Acceptance Gate

The implementation is code-level complete, but behavior is not accepted until
the 1000-step anchor-only diagnostic confirms:

```text
owm_proposal_tokens > 0
proposal_mask samples are read
task object overlay receives active anchor support
loss_anchor_pv does not diverge
same-role overlap does not return to pathological all-anchor overlap
proposal/point bridge metrics are nonzero
```

If those fail, the likely issue is not the mask bridge data contract; it is
the upstream contact/task score map, segment selection, or AQR anchor ownership
loss balance.
