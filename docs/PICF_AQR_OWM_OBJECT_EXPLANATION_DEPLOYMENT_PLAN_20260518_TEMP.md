# PICF-AQR-OWM Object-Explanation Deployment Plan

Date: 2026-05-18

Status: deployment design and mathematical follow-through. 2026-05-19 update:
the code-level Object Explanation Measurement Layer (OEML) is implemented in
contracts, runtime, training metrics, CLI loss hooks, and verifier scripts. This
is still not a behavior acceptance report.

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

Related evidence ledgers:

```text
docs/PICF_AQR_OWM_SLOT_PAPER_CODE_GAP_AUDIT_20260518_TEMP.md
temp/audits_20260518/slot_object_centric_2025_2026_math_followthrough.md
docs/PICF_AQR_OWM_OPEN_ISSUE_TRACKER_20260517_TEMP.md
docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md
```

## Executive Conclusion

The current PICF-AQR-OWM code is a coherent posterior-centered typed-memory
belief router. It is not yet a complete modern object-centric slot discovery
system. The central missing piece, relative to recent slot/OCL code, is a
primary object-explanation measurement layer:

```text
dense typed evidence
  -> object slots compete to explain evidence
  -> object/background masks are trained by feature/geometry/contact prediction
  -> resulting object measurements feed AQR/posterior
  -> posterior remains authoritative
  -> PI0.5 action path remains unchanged
```

This plan intentionally does not revive blind SAM as production evidence and
does not replace PICF with a pure image-OCL model. The correct repair is to
copy mature slot invariants into the PICF belief-filter likelihood.

2026-05-19 implementation status:

```text
Implemented:
  PicfObjectExplanationState
  object/background masks over visual, temporal, point, tactile, tracklet,
  and proposal typed evidence
  graph.object_explanation_quality feedback into _mapg_slot_assignment
  OEML feature/point/contact/duplicate/background loss hooks
  default-zero CLI loss switches
  oeml_* runtime metrics
  scripts/picf_oeml_dataflow_audit.py
  temp/audits_20260519/oeml_math_dataflow_followthrough.md

Still pending:
  behavior acceptance on fresh training/eval evidence
  deliberate nonzero OEML-loss validation after runtime metrics are healthy
```

## Paper-Code Sources Used

Local code checked under `temp/paper_code_20260518/`:

```text
object-centric-learning-framework:
  ocl/perceptual_grouping.py
  ocl/models/savi.py

slot-attention-video:
  savi/modules/video.py

AdaSlot:
  ocl/perceptual_grouping.py
  ocl/conditioning.py
  ocl/decoding.py

MetaSlot:
  object_centric_bench/model/metaslot.py
  object_centric_bench/model/dinosaur.py

slotcontrast:
  slotcontrast/losses.py
  slotcontrast/modules/groupers.py

DINO:
  models/dino/dino.py
  models/dino/deformable_transformer.py
  models/dino/dn_components.py

Deformable-DETR:
  models/deformable_transformer.py
  models/ops/modules/ms_deform_attn.py
```

Current PICF code checked:

```text
src/openpi/picf/core/pipeline.py:
  _build_aqr_anchor_graph
  _aqr_same_role_support_competition
  _aqr_active_slot_mask
  _proposal_anchor_seed_transport
  _binding_logits
  _posterior_update

src/openpi/picf/core/training.py:
  _mapg_graph_loss
  _vcap_auxiliary_loss
  _matched_prediction_loss
  _binding_consistency_loss

src/openpi/picf/core/config.py:
  AQR / proposal / VCAP / active-context-reserve / OWM-loss knobs
```

Recent paper-level principles used:

```text
MetaSlot, 2025:
  variable object count, VQ prototype/codebook duplicate suppression,
  progressive-noise slot aggregation.

SlotContrast / SlotMatch, 2025:
  temporal object consistency is useful but fragile under permutation/recycle.

QASA, 2026:
  decouple slot selection quality from reconstruction pressure; do not solve
  dynamic object count with a naive count penalty alone.

Object-binding ViT, 2025:
  pretrained ViTs can contain a same-object subspace, but it must be audited or
  calibrated; existence of a latent subspace does not train PICF slots by
  itself.

DINO / Deformable-DETR:
  object queries require reference geometry, denoising/reference supervision,
  and set-style competition; a query vector alone is not enough.
```

## Current PICF Mathematical Model

PICF is a belief filter:

```math
b_t(s)
\propto
p(o_t \mid s_t)
\int p(s_t \mid s_{t-1}, a_{t-1}) b_{t-1}(s_{t-1}) ds_{t-1}
```

The current AQR measurement router approximates typed evidence assignment:

```math
p_{j,i}^{(m)}
=
\operatorname{softmax}_i
\left(
  \frac{q_j^\top k_i^{(m)}}{\sqrt d}
  + b_{j,i}^{(m)}
\right)
```

where `m` ranges over visual, temporal V-JEPA, PaliGemma image support, point,
tactile, posterior, cache, tracklet, and proposal memories.

Same-role competition currently approximates inverted slot competition:

```math
\bar p_{j,i}^{(m)}
=
\frac{p_{j,i}^{(m)}}
{\sum_{j' \in role(j)} p_{j',i}^{(m)}+\epsilon}
```

and mixes it back as a routing prior:

```math
p' = (1-\lambda_c)p + \lambda_c\bar p
```

Active/context/reserve routing approximates adaptive slot cardinality:

```math
w_j =
\begin{cases}
1, & j \in active \\
\lambda_{ctx}, & j \in context \\
0, & j \in reserve
\end{cases}
```

This is coherent as a control router, but it is not the same as object
explanation. It can demote duplicate slots after the fact, but it does not
force every dense token to be explained by exactly one object or background
slot.

## Main Structural Gap

Mature OCL systems train object files by explaining dense evidence:

```math
L_{explain}
=
\sum_i
d\left(
z_i,
\sum_{j=1}^{K} m_{j,i} f_\theta(s_j, i)
+ m_{bg,i} f_{bg}(i)
\right)
```

with an object/background partition:

```math
\sum_j m_{j,i}+m_{bg,i}=1,\quad m_{j,i}\ge 0
```

PICF currently has:

```text
good:
  typed evidence readers
  posterior authority
  same-role competition
  active/context/reserve routing
  posterior file competition
  proposal anchor seeding
  binding signature terms
  guarded OWM losses

missing:
  trained per-slot object/background masks over dense evidence
  feature reconstruction / prediction from those masks
  point/contact explanation likelihood
  trained slot-quality selection decoupled from reconstruction
  calibrated IsSameObject probe on latest artifacts
```

Therefore a rising `loss_anchor_pv`, poor task-object overlay binding, or
same-role duplication should not be treated as only a loss-weight problem. The
router lacks the central object-explanation likelihood used by mature slot
models.

## Correct Deployment Repair

Add an Object-Explanation Measurement Layer (OEML) below AQR and above the
posterior update.

### Inputs

OEML reads only typed evidence already allowed in PICF:

```text
M_visual:
  dense current V-JEPA/static visual tokens

M_temporal:
  static+wrist V-JEPA temporal tokens

M_pg:
  PaliGemma image/text support tokens

M_point:
  Sonata/depth/world point tokens

M_tactile:
  AnyTouch/contact tokens

M_tracklet:
  optional generated tracklet sidecar tokens

M_proposal:
  optional task/contact proposal sidecar masks/tokens

M_post:
  previous posterior files
```

No dense token memory is pruned. Background/context evidence remains readable.

### Slot State

Each physical file keeps:

```math
S_j =
(a_j,c_j,\mu_j,\Sigma_j,\alpha_j,r_j,\sigma_j,q_j)
```

where:

```text
a_j: persistent address prior
c_j: time-varying content
mu/Sigma: geometry belief
alpha: activity/existence
r_j: role
sigma_j: support signature
q_j: current query/content carrier
```

The address is not identity truth. It is a prior term gated by current evidence,
innovation, recycle state, and support quality.

### Object/Background Masks

For each evidence family:

```math
\ell_{j,i}^{(m)}
=
q_j^\top W_m z_i^{(m)}
+ b_{geom}(j,i)
+ b_{proposal}(j,i)
+ b_{contact}(j,i)
+ b_{tracklet}(j,i)
+ b_{role}(j,i)
```

Add a background/no-object logit:

```math
\ell_{bg,i}^{(m)} = g_{bg}(z_i^{(m)}, i)
```

Then:

```math
m_{j,i}^{(m)}
=
\operatorname{softmax}_{j \in object \cup bg}
(\ell_{j,i}^{(m)})
```

This is the critical difference from current AQR. Current AQR normalizes each
slot over tokens. OEML also forces token-wise competition over object slots and
background, so evidence must be explained by something.

### Explanation Losses

Feature explanation over frozen dense visual/temporal features:

```math
L_v =
\sum_i
\left\|
\hat z_i^{v}
- z_i^{v}
\right\|_2^2
```

where:

```math
\hat z_i^{v}
=
\sum_j m_{j,i}^{v} f_v(S_j,i)
+m_{bg,i}^{v} f_{bg}^{v}(i)
```

Point/geometry explanation:

```math
L_p =
-\sum_i
\log
\left[
\sum_j
m_{j,i}^{p}
\mathcal N(x_i;\mu_j,\Sigma_j+\Sigma_i)
+m_{bg,i}^{p}p_{bg}(x_i)
\right]
```

Contact/tactile explanation:

```math
L_c =
\operatorname{BCE}
\left(
\hat y_i^{contact},
y_i^{contact}
\right)
```

with the contact target derived from tactile sensor pose/contact likelihood or
contact-motion sidecar, not from wrist image center.

Duplicate/background quality:

```math
L_{dup}
=
\sum_{j<k, r_j=r_k}
Q_j Q_k
\operatorname{overlap}(m_j,m_k)
```

Slot quality is decoupled from reconstruction, following the QASA principle:

```math
Q_j =
f(
coverage_j,
entropy_j,
geometry_valid_j,
temporal_consistency_j,
contact_agreement_j
)
```

Selection uses `Q_j`; reconstruction still has enough capacity through
background/context so there is no naive count-vs-reconstruction conflict.

### Proposal / Tracklet / Contact Semantics

Proposal/contact/tracklet sidecars are weak measurement references:

```text
allowed:
  initialize reference geometry
  add mask likelihoods
  create weak same-object pairs
  improve slot-quality estimates

forbidden:
  overwrite posterior identity
  replace dense evidence
  become action target truth
  bypass posterior correction
```

This keeps PICF aligned with Deformable-DETR/DINO reference-query practice
without turning proposals into hard labels.

## Dataflow To Implement

### Phase 1: OEML Contracts

Add:

```text
PicfObjectExplanationState
  visual_masks
  temporal_masks
  point_masks
  tactile_masks
  proposal_masks
  tracklet_masks
  background_masks
  slot_quality
  explained_visual
  explained_point
  contact_logits
  valid
```

Wire into:

```text
PicfCoreState.object_explanation
PicfAnchorPriorGraphState.object_explanation_quality
PicfPosteriorAnchorState.object_signature
```

### Phase 2: OEML Runtime

Implement:

```text
_build_object_explanation_measurements(...)
  inputs: token_field, graph, previous, proposal/tracklet/contact evidence
  outputs: PicfObjectExplanationState
```

It must run before posterior correction but after current typed memories are
available. It may use graph priors as initialization, but it must produce its
own token-wise object/background masks.

### Phase 3: OEML Losses

Add guarded losses:

```text
loss_object_explain_visual
loss_object_explain_point
loss_object_explain_contact
loss_object_duplicate
loss_object_quality
```

Defaults:

```text
lambda_object_explain_visual = 0.0
lambda_object_explain_point = 0.0
lambda_object_explain_contact = 0.0
lambda_object_duplicate = 0.0
lambda_object_quality = 0.0
```

The deployment validation profile may enable small weights only in a frozen
anchor/object warmup. They must remain off in production until short diagnostics
pass.

### Phase 4: 1000-Step Frozen Validation

Only after Phases 1-3 compile and pass local audits:

```text
trainable_scope = anchor_only
freeze_action = true
freeze_paligemma = true
freeze_vjepa = true
freeze_sonata = true
freeze_anytouch = true
num_train_steps = 1000
overlay_interval = 50
print_interval = 50
```

Acceptance:

```text
object masks visibly bind task object / contact object
active same-role object-core overlap stays low
loss_anchor_pv does not monotonically diverge
posterior recycle is not saturated
proposal/tracklet tokens are present when sidecars are configured
gray/background masks explain background without starving object slots
```

If these fail, do not start 30k action co-training.

### Phase 5: Co-Train

Only after Phase 4:

```text
unfreeze PaliGemma connector / semantic adapter as configured
unfreeze action path
keep high-risk predictive losses at zero
keep object-explanation losses budgeted
monitor object masks and anchor overlays
```

## What Should Not Be Implemented

Rejected as non-self-consistent:

```text
blind SAM proposals as hard object truth
count-only anchor loss
stop-token-only autoregressive anchor generator
more overlap penalties without object explanation
hard deletion of gray/reserve anchors
direct wholesale MetaSlot VQ codebook transplant
direct SlotLifter/NeRF recipe for CALVIN
```

Reason:

```text
These either bypass posterior authority, lack a robot measurement likelihood,
or fail to explain dense evidence.
```

## Required Local Audits Before Any Run

Run:

```bash
PYTHONPATH=src python scripts/verify_picf_owm_contract.py
PYTHONPATH=src python scripts/picf_owm_professor_grade_audit.py --fail-on-fail
PYTHONPATH=src python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail
PYTHONPATH=src python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail
```

The future OEML implementation must add audit checks for:

```text
object_explanation_state_present
tokenwise_object_background_masks_sum_to_one
background_mask_does_not_prune_dense_tokens
proposal_contact_tracklet_are_measurement_evidence_only
object_explain_losses_default_zero
frozen_validation_profile_enables_only OEML/anchor connectors
```

## Current Deployment Decision

Do not claim full object-binding completion on the current code solely because
AQR/posterior/proposal/sidecar routing exists.

Do not start the requested 1000-step frozen action+PaliGemma validation as if
the full repair is implemented unless the OEML contracts/runtime/losses above
are present.

If a previous remote diagnostic is already running, it may continue as a
baseline evidence source. Its result should be interpreted as a current-router
baseline, not as validation of the object-explanation repair.

## Local Verification Performed

Current local checks after writing this plan:

```text
PYTHONPATH=src python scripts/verify_picf_owm_contract.py
  PASS

PYTHONPATH=src python scripts/picf_owm_professor_grade_audit.py --fail-on-fail
  PASS after correcting a stale audit needle from recycle_share to birth_share.
  The runtime code already used slot-local recycle residuals, layernorm/rmsnorm
  normalization, birth_share dustbin redistribution, dustbin_final, support
  binding, address_update_rate, and identity_innovation_risk.

PYTHONPATH=src python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail
  PASS

PYTHONPATH=src python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail
  PASS

PYTHONPATH=src python scripts/picf_owm_strict_diagnose.py --fail-on-fail
  PASS, with runtime metrics/CALVIN artifact warnings only because no metrics
  path was supplied to that script invocation.

PYTHONPATH=src python scripts/picf_owm_dataflow_trace.py --fail-on-fail
  PASS

PYTHONPATH=src python scripts/picf_owm_mvtrack_deep_audit.py --fail-on-fail
  PASS

PYTHONPATH=src python -m py_compile ...
  PASS for the touched audit script and core pipeline/training modules.
```

Interpretation:

```text
The current router implementation is internally consistent under the existing
audits.

The full object-explanation repair is not implemented yet; no 1000-step frozen
validation should be treated as validating OEML until the contracts/runtime/
losses/audits listed above exist.
```

## Final Judgment

The strongest, most self-consistent next model is:

```text
PICF-AQR-OWM + Object-Explanation Measurement Layer
```

not:

```text
PICF + more sidecar biases
PICF + blind SAM
PICF + count penalty
PICF + hard anchor deletion
```

This is a complete design target. The code is not yet complete until OEML is
implemented and the audits above pass.
