# PICF-AQR-OWM Proposal Mask Support Follow-Through

Status: implemented locally on 2026-05-18. This is the maintained replacement
for making large proposal boxes stronger.

## Design Decision

The proposal path is object-support evidence, not posterior truth.

Rejected:

```text
large event box -> strong proposal -> hard object target
```

Maintained:

```text
contact/task score map
  -> compact connected components
  -> proposal center/box/objectness
  -> sparse soft mask samples
  -> current-frame point-token likelihood
  -> AQR proposal and point priors
  -> posterior correction
```

The sidecar does not store training point row ids because point rows can change
with stride, FPS, max-points, and gripper/static fusion. It stores normalized
static-view support samples:

```text
proposal_mask_xy:      [S, 2]
proposal_mask_weights: [S]
proposal_mask_offsets: [K + 1]
```

This is a CSR representation of K proposal masks.

## Mathematical Contract

For proposal k with sparse support samples:

```math
\mathcal S_k = \{(u_s, w_s)\}_{s=a_k}^{a_{k+1}-1}
```

and current training point-token projection u_i, the mask likelihood is:

```math
L_i^{mask}(k)
=
\frac{
  \sum_{s \in \mathcal S_k}
  w_s \exp(-\|u_i-u_s\|^2/(2\tau_m^2))
}{
  \sum_{s \in \mathcal S_k} w_s + \epsilon
}
```

If mask samples are missing, the runtime falls back to a bounded box
likelihood:

```math
L_i^{box}(b_k)=
\sigma((x_i-x^0_k)/\tau_b)
\cdot\sigma((x^1_k-x_i)/\tau_b)
\cdot\sigma((y_i-y^0_k)/\tau_b)
\cdot\sigma((y^1_k-y_i)/\tau_b)
```

The final proposal-to-point bridge remains row-normalized:

```math
P(i|k) =
\frac{L_i(k) v_i q_k}{\sum_{i'} L_{i'}(k) v_{i'} q_k + \epsilon}
```

where v_i is visibility/depth validity and q_k is proposal quality/objectness.

## Paper Alignment

Slot Attention / MONet / IODINE:

```text
soft object masks and competition are first-class;
one slot should not explain multiple unrelated objects;
background/no-object capacity must exist.
```

Deformable DETR / DINO:

```text
reference points/boxes guide local evidence gathering and refinement;
they do not replace the image/point tokens.
```

SAM:

```text
point/box prompts are prompts;
the usable object evidence is mask-like support, not the box itself.
```

PICF consequence:

```text
proposal masks are strong measurement evidence;
posterior remains authoritative;
dense V-JEPA / PG / point / tactile tokens remain intact.
```

## Code Follow-Through

Generator:

```text
scripts/picf_contact_motion_sidecar_precompute.py
```

Writes:

```text
proposal_centers_xy
proposal_boxes_xyxy
proposal_objectness
proposal_view_ids
proposal_source_ids
proposal_mask_xy
proposal_mask_weights
proposal_mask_offsets
```

Data contract:

```text
src/openpi/picf/contracts.py::PicfObservation
src/openpi/picf/core/contracts.py::PicfPseudoProposalState
```

Trainer/serve readers:

```text
scripts/picf_core_train.py::_MVTRACK_PROPOSAL_KEYS
scripts/serve_picf_policy.py
```

Runtime bridge:

```text
src/openpi/picf/core/pipeline.py::_proposal_to_point_matrix
```

Config:

```text
proposal_mask_point_tau
```

## Invariants

```text
1. Mask support is optional typed evidence.
2. Missing mask samples fall back to box likelihood.
3. Sidecar point row ids are never trusted.
4. Dense typed tokens are not pruned.
5. Posterior is not overwritten.
6. Proposal quality and validity still gate the bridge.
7. Blind SAM remains archived/off by default.
```

## Local Verification

Commands run locally:

```bash
python -m py_compile \
  scripts/picf_contact_motion_sidecar_precompute.py \
  scripts/picf_core_train.py \
  scripts/serve_picf_policy.py \
  scripts/verify_picf_owm_contract.py \
  src/openpi/picf/contracts.py \
  src/openpi/picf/core/contracts.py \
  src/openpi/picf/core/config.py \
  src/openpi/picf/core/pipeline.py

uv run pytest -q \
  scripts/picf_contact_motion_sidecar_precompute_test.py \
  scripts/verify_picf_owm_contract_test.py \
  scripts/picf_owm_evidence_bundle_test.py

uv run python scripts/verify_picf_owm_contract.py
```

Observed:

```text
py_compile PASS
pytest 6 passed
OWM verifier PASS
synthetic mask-to-point bridge audit PASS
```
