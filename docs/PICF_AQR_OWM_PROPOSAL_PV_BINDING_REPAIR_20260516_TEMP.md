# PICF-AQR-OWM Proposal/PV/Binding Repair Follow-Through

Date: 2026-05-16, updated 2026-05-17

Status: object-routed PV repair is deployed and is being diagnosed on A7.
The 2026-05-17 anchor-only run shows the active object-file path is controlled,
but reserve/context raw overlap and proposal sidecar coverage still need a
follow-up run before behavior acceptance.

## 1. Problem Being Fixed

The latest long run has two different signals:

- Action loss improves quickly.
- Active object-core overlap is mostly controlled.
- `loss_anchor_pv` rises after the early window.
- Raw same-role support overlap remains high in context/reserve anchors.
- `owm_proposal_tokens=0`, so SAM/proposal memory is not active in that run.

The A7 anchor-only diagnostic at step 200 separates these signals:

```text
loss_anchor_pv: 0.748 -> 1.186 -> 1.183 -> 1.180
loss_pv_weak: 6.330 -> 6.110 -> 5.945 -> 5.609
active same-role support overlap: 0.026 -> 0.120 -> 0.144 -> 0.143
raw same-role support overlap: 0.132 -> 0.630 -> 0.963 -> 0.991
active object-core overlap: 0.061 -> 0.079 -> 0.023 -> 0.010
owm_proposal_tokens: 0 at all logged steps
```

This means the active object-file lane is not collapsing, while the fixed
reserve/context capacity is still repeatedly reading a broad same-role support
region.  That is not evidence that V-JEPA tokens are dropped or that active
objects are ignored; it is evidence that the diagnostic must keep object-file,
context, and reserve lanes separate.

The critical issue is not "too few modules".  The issue is a mismatch between
two mathematical objects:

```text
dense PV correspondence:
  every visible point/visual token pair that projects consistently should agree.

object-slot routing:
  only a sparse subset of those pairs belongs to the object/query currently
  claimed by an active anchor.
```

The old `loss_anchor_pv` made object routing match the dense projective matrix
directly:

```math
L_{anchor\_pv}^{old}
=
\frac{1}{|C|}
\sum_{p,v}
1[(p,v)\in C]\,
P_{pv}\,
BCE(R_{pv}, P_{pv})
```

where `P_pv` is projective compatibility and `R_pv` is routing consistency
induced by object anchors.  This over-constrains object slots: a sparse object
belief is asked to reconstruct dense background/static projection edges.  Under
action cotrain, the easiest compromise is to reuse the same high-confidence
regions and let context/reserve anchors duplicate them.

## 2. Repaired Objective

Dense correspondence remains supervised by `loss_pv_weak`; object slots get a
separate object-routed gate:

```math
G_{pv}
=
\sum_j
w_j\,
\bar A^{point}_{jp}\,
\bar A^{visual}_{jv}
```

where `A^point` and `A^visual` are AQR priors and `w_j` is the downstream slot
weight.  The repaired objective is:

```math
L_{anchor\_pv}^{new}
=
\frac{1}{|C|}
\sum_{p,v}
1[(p,v)\in C]\,
P_{pv}\,
\left(\rho + (1-\rho)\,\mathrm{norm}(G_{pv})\right)\,
BCE(R_{pv}, P_{pv})
```

Default `rho=0.25`.  The floor is important.  It means:

- background/static projective evidence is not deleted;
- dense V-JEPA/PG/point tokens remain available upstream;
- only the *strong object-slot pressure* is focused on object-routed edges.

This is not SAM hard-label training.  It is a correction of the loss domain:
dense PV stays dense, object anchor PV becomes object-routed.

## 3. Paper-Code Comparison

### Object Binding in ViTs

The pulled code in `/tmp/picf_sam_code/vit-object-binding` trains pairwise
IsSameObject probes.  The important implementation pattern is:

```text
compute_batch_pairwise_similarity(probe, x, y)
labels_pairwise = labels[:, :, None] == labels[:, None, :]
```

and the README explicitly compares linear, quadratic, low-rank quadratic,
cosine, dot-product, and attention baselines.  The lesson for PICF is:

```text
same-object structure should be a projected pairwise subspace,
not raw hidden cosine and not raw attention alone.
```

PICF already follows this in the binding logit through projected
`binding_signature` terms.  The current repair applies the same philosophy to
PV alignment: dense token correspondence is not identical to object identity,
so object-slot pressure must be routed through object support.

### SAM / SAM2 / OpenMask3D / Open3DIS

The pulled SAM code exposes `bbox`, `predicted_iou`, and `stability_score`.
SAM2 exposes promptable mask prediction; OpenMask3D/Open3DIS use 2D/3D proposal
fusion and projection-aware mapping.  Their shared architectural role is:

```text
proposal/objectness evidence, not policy truth.
```

PICF therefore uses SAM-style outputs as offline `proposal_*` sidecars:

```text
proposal_centers_xy
proposal_boxes_xyxy
proposal_objectness
proposal_view_ids
proposal_source_ids
```

These enter `PicfPseudoProposalState` and `aqr_proposal_reader`.  They do not
overwrite posterior, do not replace dense V-JEPA, and do not become labels.

The trainer overlay now writes three diagnostics when proposals are present:

```text
with_gray:
  graph/posterior/task anchors plus gray context/reserve files and SAM boxes.

active_only:
  full-weight active object files plus task anchors and SAM boxes.

sam_proposals:
  static-view SAM/proposal boxes with active object files. Wrist proposals are
  kept in JSON but not projected into static pixels without extrinsics.
```

This is important for the user's current failure mode: "orange/active anchors
are not on the intended red block" must be checked against task readout anchors
and SAM boxes, not against gray reserve files alone.

## 4. Dataflow Contract

### Current Dense Evidence

```text
CALVIN frame
 -> rgb_static / rgb_gripper / depth / tactile / proprio / language
 -> V-JEPA temporal tokens, PG image/text tokens, point/tactile tokens
 -> AQR point/visual/temporal/PG readers
 -> posterior correction
 -> PI0.5 action
```

### Offline Proposal Evidence

```text
CALVIN frame
 -> frozen SAM/SAM2 proposal precompute
 -> episode_XXXXXXX.npz sidecar
 -> replay / train frame dict merges proposal_*
 -> PicfObservation.proposal_*
 -> PicfPseudoProposalState
 -> aqr_proposal_reader
 -> proposal_priors / support_signature
 -> posterior correction
```

Missing proposal files are valid no-ops.  If metrics show
`owm_proposal_tokens=0`, proposal evidence is not active and the run cannot
validate this branch.

## 5. Code-Level Changes

Deployed changes:

- `PicfTransitionLossConfig.anchor_pv_object_gate_enabled=True`.
- `PicfTransitionLossConfig.anchor_pv_object_gate_floor=0.0`.
- `PicfTransitionLossConfig.anchor_pv_object_normalize_by_object_mass=True`.
- `PicfTransitionLossConfig.aqr_denoising_confirmed_object_only=True`.
- `PicfAlignmentLossConfig` mirrors the anchor-PV fields.
- `compute_alignment_loss()` constructs an object-routed point/visual gate from
  `graph.point_priors`, `graph.visual_priors`, and active object rows.
- `loss_anchor_pv` is averaged over object-confirmed point/visual edges, not
  over all dense projective edges.
- `loss_pv_weak` remains unchanged as dense PV/background supervision.
- AQR support denoising is a guarded auxiliary and now requires confirmed
  object-candidate/proposal/point evidence in addition to active object rows.
- Anchor overlays now include task readout anchors and static SAM proposal
  boxes; JSON retains proposal and downstream-weight metadata.
- `scripts/archive/picf_sam_proposal_precompute_legacy.py --preview-root ...` can write
  proposal preview PNGs during offline sidecar generation.
- `scripts/picf_core_train.py --calvin-segment-indices ...` exists only for
  diagnostics where proposal sidecars cover a small segment subset. Production
  training should leave it unset.
- `scripts/picf_anchor_pv_object_gate_audit.py` verifies the repair path.

## 6. What This Does Not Claim

This repair does not claim that:

- SAM/proposal evidence is active without precomputed sidecars.
- ordinal / "fourth-from-left" grounding is solved.
- object identity is guaranteed without tracklet/proposal/pairwise evidence.
- action loss alone validates anchor health.

Acceptance requires a fresh run with:

```text
owm_proposal_tokens > 0 in at least the proposal-enabled diagnostic
loss_anchor_pv not rising after early action convergence
aqr_active_same_role_support_overlap_max controlled
aqr_active_same_role_object_core_overlap_max controlled
posterior_identity_switch_rate decreasing or at least not worsening
anchor overlays showing active anchors on task object / relevant context,
with SAM boxes visible for proposal-enabled samples
```

## 7. Run Gate

Before a new long run:

1. Precompute proposal sidecars for a CALVIN subset or full split.
   If the subset is sparse, run the diagnostic on the same covered segment ids
   with `--calvin-segment-indices`; otherwise `owm_proposal_tokens=0` is an
   expected sampling artifact rather than a model failure.
2. Run:

```bash
python scripts/picf_anchor_pv_object_gate_audit.py
python scripts/archive/picf_sam_proposal_dataflow_audit_legacy.py --external-code-root /tmp/picf_sam_code --fail-on-fail
python -m py_compile src/openpi/picf/core/training.py scripts/picf_core_train.py scripts/archive/picf_sam_proposal_precompute_legacy.py
```

3. Launch with:

```text
--proposal-memory-enabled
--mvtrack-sidecar-root <proposal_sidecar_root>
--anchor-pv-object-gate-enabled
--anchor-pv-object-gate-floor 0.25
```

4. Confirm startup/metrics:

```text
owm_proposal_tokens > 0
anchor_pv_object_gate_enabled=True
```

## 8. Segment-Restricted SAM Diagnostic Result

Run:

```text
picf_a7_anchor_pv_object_gate_samseg0_diag300_20260517
```

Reason:

```text
The initial A7 object-gated diagnostic sampled the full CALVIN training split
while the SAM sidecar probe covered only segment 0.  It therefore reported
owm_proposal_tokens=0 and could not validate proposal memory.  This follow-up
uses --calvin-segment-indices 0 so every sampled frame is eligible for the
precomputed proposal sidecar.
```

Step 50 metrics:

```text
loss_total                               0.5854
loss_alignment                           0.5104
loss_anchor_pv                           0.7477
loss_pv_weak                             6.3128
loss_mapg_routing                        0.6692
loss_mapg_cycle                          0.3798
loss_mapg_support_diversity              0.2067
aqr_active_same_role_support_overlap_max 0.0122
aqr_same_role_support_overlap_max        0.1915
aqr_active_same_role_object_core_overlap_max 0.0380
aqr_same_role_object_core_overlap_max    0.0868
owm_proposal_tokens                      30.6050
aqr_proposal_support_entropy_mean        0.9223
aqr_proposal_support_max                 0.2268
posterior_identity_switch_rate           0.7172
posterior_file_competition_duplicate_overlap_max 0.9923
posterior_support_mass_final_mean        0.5188
```

Interpretation:

```text
Proposal memory is active in this run.  The active object-file overlap is very
low at step 50, while raw same-role overlap is also low.  The generated
sam_proposals overlay confirms that static-view SAM/proposal boxes are visible
alongside active posterior anchors.  This validates the proposal dataflow, but
it does not prove semantic task selection or PaliGemma cotrain behavior because
this diagnostic is anchor-only, action-free, and semantic-frozen.
```

Step 50 -> 150 trend:

```text
loss_total                                0.5854 -> 0.4633
loss_anchor_pv                            0.7477 -> 0.7556
loss_pv_weak                              6.3128 -> 4.5739
loss_mapg_routing                         0.6692 -> 0.7180
aqr_active_same_role_support_overlap_max  0.0122 -> 0.2836
aqr_same_role_support_overlap_max         0.1915 -> 0.9996
aqr_active_same_role_object_core_overlap_max 0.0380 -> 0.0267
aqr_same_role_object_core_overlap_max     0.0868 -> 0.8710
owm_proposal_tokens                       30.605 -> 30.565
aqr_proposal_support_max                  0.2268 -> 0.4053
posterior_support_mass_final_mean         0.5188 -> 0.4723
```

Step 150 overlay review:

```text
The active-only and sam_proposals overlays keep active anchors around the red
block, drawer mouth, and gripper contact region.  The with_gray overlay still
contains many reserve/context anchors over broader scene regions.  Therefore the
right interpretation is not "all anchors are correct objects"; it is "active
object files are no longer globally collapsed, while inactive/context files
still create high raw same-role overlap and should not be used as the primary
success metric."
```

## 9. 2026-05-17 Soft Tactile Evidence And Active-Duplicate Repair

The segment-restricted SAM diagnostic exposed one real residual gap:

```text
tactile_contact_prob_mean ~= 0.36
tactile_active_rate       = 0.0
loss_pt                   nonzero
```

This means the calibrated tactile contact estimator was producing useful
probability mass, but the hard `tactile_anchor_prob_on` gate prevented tactile
evidence from entering the AQR/support-signature path.  Treating this as either
"tactile is fine" or "tactile is absent" would both be false positives.

The repaired contract is two-level:

```math
e_t = 1[p_t \ge p_{floor}]
```

```math
w_t =
\operatorname{clip}
\left(
  {p_t-p_{floor} \over p_{on}-p_{floor}+\epsilon},
  0,
  1
\right)
```

where `p_t` is calibrated tactile contact probability.  If `p_t >= p_on`, the
runtime keeps the dense AnyTouch patch reread.  If
`p_floor <= p_t < p_on`, the runtime contributes a scaled sensor-level tactile
token to point/tactile alignment and binding signatures.  This preserves the
mathematical role of tactile as contact evidence without letting weak contact
open the full dense tactile reader.

Implementation follow-through:

```text
PicfCoreConfig:
  tactile_evidence_prob_floor = 0.35
  tactile_anchor_prob_on      = 0.55

PicfTokenFieldState:
  tactile_evidence_mask
  tactile_evidence_weight

pipeline._build_token_field:
  selected tactile groups = tactile_evidence_mask OR tactile_anchor_mask
  hard tactile groups use dense AnyTouch tokens
  weak tactile groups use weighted sensor-level tokens

pipeline._build_observation_anchors:
  tactile routing weights now contribute to obs_binding_signature

pipeline._posterior_update:
  tactile_native_reread is evaluated only for posterior slots whose bound
  tactile routing mass is nonzero.  Slots with zero tactile mass keep a zero
  tactile evidence vector.  This avoids a false-positive path where top-k over
  all-zero tactile weights would still select arbitrary tactile groups and make
  every slot appear to have contact evidence.

debug / trainer:
  tactile_contact_prob_max
  tactile_evidence_rate
  tactile_evidence_weight_mean
  tactile_evidence_weight_max
```

Remote bring-up exposed an additional dataflow bug after lowering the hard
tactile gate:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (...x768 and 512x512)
```

Cause:

```text
dense AnyTouch patch tokens were raw backbone-dimensional tokens, while
`tactile_native_reread` and `tactile_route_reread` are PICF hidden-dimensional
readers.  The old high hard gate rarely activated dense tactile patches, so the
bug was hidden.  The soft-evidence repair correctly made tactile active enough
to expose the dimensional inconsistency.
```

Repair:

```text
PicfFullCore.tactile_patch_token_proj projects dense AnyTouch patch tokens into
PICF hidden memory before any tactile route/target reread.  Sensor-level soft
tactile evidence and dense hard-contact patches now share the same hidden-dim
typed memory contract.
```

The second repair separates raw duplicate capacity from active object-file
duplicates:

```math
D_{raw} =
\max_{i \ne j,\ r_i=r_j} overlap(i,j)
```

```math
D_{active} =
\max_{i \ne j,\ r_i=r_j,\ a_i=1,\ a_j=1} overlap(i,j)
```

Raw duplicate overlap can stay high because reserve/context files deliberately
remain available.  Acceptance should use `D_active`, exported as:

```text
posterior_file_competition_active_duplicate_overlap_max
```

This prevents the overlay/debug path from falsely failing a run only because
inactive reserve files overlap.

Acceptance after this repair:

```text
tactile_contact_prob_mean must be non-degenerate;
tactile_evidence_rate or tactile_active_rate must be non-degenerate;
posterior_file_competition_active_duplicate_overlap_max is the duplicate
  acceptance metric, while posterior_file_competition_duplicate_overlap_max is
  raw capacity telemetry;
owm_tracklet_tokens=0 remains a data-side limitation unless a tracklet sidecar
  is generated and provided.
```

## 2026-05-17 Task-Owner Proposal Bias Repair

The 2026-05-17 anchor overlays showed a different failure mode from the earlier
duplicate-file collapse.  SAM/proposal boxes were present and active object
duplicates were controlled, but the active scene object anchor did not reliably
sit on the language-specified target object.  This means the runtime had
class-agnostic objectness evidence, but not enough task-conditioned ownership
evidence.

The repair is a measurement-routing fix, not a new auxiliary loss:

```math
\pi^{task}_i =
{1 \over |Q_{task}|}
\sum_{j \in Q_{task}}
p^{visual}_{j,i}
```

where `Q_task` are task-query rows with scene-object role and
`p^{visual}_{j,i}` is the AQR visual support after the task query has read
PaliGemma text/image-conditioned evidence.

For each proposal box `p`, project the task visual prior onto the visual grid
cells inside the proposal:

```math
s_p =
obj_p^\alpha
{ \sum_i \pi^{task}_i 1[x_i \in box_p] \over \sqrt{coverage_p+\epsilon} }
```

The square-root coverage denominator prevents large boxes from winning only by
area.  The objectness exponent keeps SAM/proposal masks as weak objectness, not
ground truth.

The final bias is centered log probability:

```math
b_p =
\lambda_{prop}
\left(
\log {s_p \over \sum_q s_q + \epsilon}
-
mean_q \log {s_q \over \sum_r s_r + \epsilon}
\right)
```

This bias is added to the proposal reader for physical scene-object rows and
task scene rows.  The previous AQR round's `\pi^{task}` is also applied as a
low-amplitude visual bias for physical scene-object rows on the next AQR query
round.

Important invariants:

```text
1. The repair does not remove visual/V-JEPA tokens.
2. It does not overwrite posterior or force a hard task mask.
3. It only biases scene-object rows, not effector rows.
4. It stays static-view-only unless calibrated non-static projection is present.
5. The task-owner score is exposed in debug metrics.
```

Code follow-through:

```text
PicfCoreConfig:
  task_owner_bias_enabled
  task_owner_visual_bias_weight
  task_owner_proposal_bias_weight
  task_owner_proposal_objectness_power
  task_owner_proposal_static_only

PicfAnchorPriorGraphState:
  task_owner_visual_prior
  task_owner_proposal_score

pipeline:
  _task_owner_visual_prior
  _task_owner_visual_bias
  _proposal_scores_from_visual_prior
  _task_owner_proposal_bias
  proposal_priors included in object-core overlap / active-slot gating /
  downstream-slot weighting

trainer CLI:
  --task-owner-bias-enabled
  --task-owner-visual-bias-weight
  --task-owner-proposal-bias-weight
  --task-owner-proposal-objectness-power
  --task-owner-proposal-static-only
```

Acceptance for the next run:

```text
aqr_task_owner_proposal_score_nonzero_fraction > 0
aqr_task_owner_proposal_score_max > 0
active same-role support/object-core overlap does not regress
task overlays show at least one active scene object file on the intended object
loss_anchor_pv and loss_aqr_denoising do not both worsen from 50 to 100
```
