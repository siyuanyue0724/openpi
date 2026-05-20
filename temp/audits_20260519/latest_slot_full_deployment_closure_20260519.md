# PICF Latest-Slot Full Deployment Closure

Date: 2026-05-19

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

Executable audit:

```bash
PYTHONPATH=src uv run --no-sync python scripts/picf_latest_slot_deployment_audit.py --fail-on-fail
PYTHONPATH=src uv run --no-sync python scripts/picf_object_candidate_slot_binding_audit.py --json
PYTHONPATH=src uv run --no-sync python scripts/verify_picf_owm_contract.py --json
```

## 1. Final closure verdict

The current PICF implementation has deployed the belief-state-compatible core
mechanisms from recent slot / object-binding / visuo-tactile work:

```text
implemented:
  slot-axis evidence competition
  explicit object/background residual
  adaptive object/no-object/duplicate slot quality
  active/context/reserve fixed-capacity file gating
  active-context full-support duplicate demotion
  same-object pairwise binding subspace
  calibrated quadratic / low-rank binding scores
  persistent posterior binding-signature memory
  sidecar/contact object candidates as soft measurements
  accepted owner-candidate geometry as graph/posterior measurement
  object-owner posterior transport through precision fusion
  tactile/contact evidence attached to object owner, not gripper owner
  tracklet/proposal optional typed sidecar dataflow
  run taxonomy separating anchor probes from comprehensive validation

rejected as production defaults:
  blind automatic SAM proposals
  hard MetaSlot visual VQ posterior truth
  full image reconstruction decoder as PICF action-time truth
  many-view NeRF/SlotLifter renderer inside CALVIN training
  online weak IsSameObject loss without real same-object labels
  predictive slot losses before posterior object identity is stable
```

Strict wording:

```text
Code-level latest-slot deployment is complete for the PICF belief-filter
contract.

Behavior-level acceptance is not complete until the A7 anchor probe and the
queued slot-comprehensive frozen-policy validation show active posterior owners
on the inspected task/contact object, low active same-role overlap, finite
binding-signature dispersion, and nonzero sidecar/tracklet evidence when
configured.

2026-05-20 validation update:
`picf_a7_active_support_dedup_300_20260520` passed the short structural gate:
the previous downstream/context clone rebound did not reproduce through step
300.  The current next gate is longer training/eval, not another local patch on
the same failure mode.
```

This distinction is mandatory.  A static audit can prove that the architecture
is not a missing-module stub; it cannot prove that 30000-step behavior will be
healthy.

## 2. PICF mathematical object

PICF is not a pure image reconstruction slot model.  Its state is a sequential
belief file:

```math
F_{t,j}=
(
x_{t,j},
S_{t,j},
a_{t,j},
c_{t,j},
\alpha_{t,j},
r_j,
\sigma_{t,j},
\phi_{t,j}
)
```

where:

```text
x,S:
  3D geometry and covariance.

a,c:
  address and content.

alpha:
  existence / support confidence.

r:
  role / file type.

sigma:
  typed support signature over visual, temporal, point, tactile, tracklet,
  proposal and PaliGemma evidence.

phi:
  pairwise binding signature used for same-object compatibility.
```

The central update is:

```math
p(F_t \mid z_{\le t}, a_{<t}, \ell)
\propto
p(z_t \mid F_t, \ell)
\int p(F_t \mid F_{t-1}, a_{t-1})p(F_{t-1}\mid z_{\le t-1})dF_{t-1}.
```

Every deployed module must enter one of three places:

```text
1. measurement evidence p(z_t | F_t, language)
2. posterior file update / assignment
3. training-only guarded auxiliary
```

If a module creates an untyped hard truth, bypasses posterior authority, or
requires always-present modality labels, it is not compatible with the PICF
contract.

## 3. Latest paper mechanism mapping

### 3.1 Slot Attention / SAVi / AdaSlot / OCL

External invariant:

```math
a_{j,i}
=
\operatorname{softmax}_j(q_j^\top k_i),
\quad
u_j=
\sum_i
\frac{a_{j,i}}{\sum_i a_{j,i}+\epsilon}v_i .
```

PICF-native deployment:

```text
same-role support competition
object-candidate row capacity
explicit background/no-object residual
active/context/reserve gating
posterior file competition
```

Reason for not copying the external module directly:

```text
The external module is a perceptual grouping / reconstruction loop. PICF is a
typed evidence router with posterior file authority and PI0.5 action output.
Replacing AQR with a reconstruction-only loop would remove the control contract.
```

Acceptance metric:

```text
aqr_active_same_role_support_overlap_max should remain low.
aqr_same_role_support_overlap_max may be high for reserve/context rows and is
not by itself a failure.
```

### 3.2 MetaSlot

External invariant:

```text
adaptive slot count
duplicate aggregation
object prototype / codebook initialization
```

PICF-native deployment:

```text
slot_quality.object / no_object / duplicate
active/context/reserve file state
duplicate demotion
birth competition
sidecar/proposal-conditioned initialization
accepted owner-candidate geometry transport
```

Not copied:

```text
global VQ prototype codebook
```

Reason:

```text
PICF must scale to datasets with missing point, missing tactile, missing wrist,
or different camera/layout distributions. A hard visual codebook inside
posterior state would make identity depend on one visual prototype space and
can conflict with typed multimodal belief updates.
```

Permitted future use:

```text
A prototype bank may be used as a birth proposal initializer only. It must pass
through object-candidate assignment, background residual, posterior transport,
and file competition.
```

### 3.3 QASA

External invariant:

```text
slot quality must be separated from reconstruction/action pressure;
adaptive K should be learned/estimated rather than all fixed slots being active.
```

PICF-native deployment:

```text
PicfSlotQualityState
aqr_slot_quality_head
target_object_quality
target_no_object_prob
target_duplicate_prob
active_weight
context_weight
optional lambda_slot_quality
```

Why this is complete enough for PICF:

```text
PICF keeps fixed-capacity files for memory continuity, but gates which files
are active object owners. This is the belief-state analogue of adaptive slot
count. It avoids deleting dense memory while preventing reserve rows from
stealing action/prefix ownership.
```

### 3.4 Object Binding in pretrained ViTs

External invariant:

```text
same-object relation is not plain hidden cosine;
it is better decoded by pairwise / quadratic probes in a binding subspace.
```

PICF-native deployment:

```text
binding_signature_proj
support-weighted binding_signature
binding_quadratic_diag
binding_low_rank_left/right
double-center z-score score calibration
posterior_binding_signature_memory
measurement dispersion gate
```

Mathematically:

```math
s^{lin}_{j,i}=\langle \phi^-_j,\tilde\phi_i\rangle
```

```math
s^{quad}_{j,i}
=
(\phi^-_j)^\top D\tilde\phi_i
+
\frac{1}{2}
\left(
(L\phi^-_j)^\top(R\tilde\phi_i)
+
(R\phi^-_j)^\top(L\tilde\phi_i)
\right).
```

Then scores are calibrated by relative pair structure, not common-mode
magnitude:

```math
\hat S = \operatorname{clip}
\left(
\frac{H_r S H_c}{\max(\operatorname{std}(H_r S H_c), \epsilon)}
\right).
```

This is why the code rejects blind per-frame binding signature overwrite when
all current measurements are common-mode.

### 3.5 SlotVLA

External invariant:

```text
robot policies benefit from object/relation slots, boxes/masks/tracks, and
object-level temporal identity.
```

PICF-native deployment:

```text
proposal_* sidecar schema
tracklet_* sidecar schema
object_candidate_assignment
object_candidate_owner_assignment
object_candidate_owner_point_priors
object_candidate_owner_x / object_candidate_owner_S
posterior_owner_transport
anchor overlays and IsSameObject diagnostics
```

Boundary:

```text
SlotVLA-style supervision often assumes curated object boxes/masks/tracks.
CALVIN sidecars here are weak contact/task/motion evidence. Therefore PICF
must treat them as measurements with background residual, never hard labels.
```

### 3.6 OCRA / OmniVTA / MLA-style visuo-tactile object binding

External invariant:

```text
tactile/contact evidence should be attached to the contacted object, not to an
unrelated gripper token, and must be fused with vision/geometry.
```

PICF-native deployment:

```text
tactile_attach_to_object_owner = True
dense AnyTouch patch projection to PICF hidden memory
soft tactile evidence gating
tactile-to-point kernel
object-owner tactile role bias
posterior_owner_transport_roles=(1,)
precision-fused owner geometry
```

Mathematically, accepted contact/object owner geometry is fused as a measurement:

```math
\Lambda^+
=
(S^{std})^{-1}
+
\kappa c^{owner}(S^{owner})^{-1}
```

```math
x^+
=
(\Lambda^+)^{-1}
\left(
(S^{std})^{-1}x^{std}
+
\kappa c^{owner}(S^{owner})^{-1}x^{owner}
\right).
```

This is stronger and cleaner than linearly dragging an anchor center.

## 4. What is still not proven by code

These are not "missing modules" in the code. They are behavior/data gates:

```text
1. Short validation has now shown active posterior owner files staying on the
   inspected sidecar/contact object across sampled overlays.  This is closed
   for short validation, not for final 30000-step behavior.

2. Longer training/eval must still show the same under the intended
   action-aware training recipe.  A 300-step run can reject structural failures;
   it cannot prove final action performance.

3. Full sidecar/tracklet coverage is still a data-production issue. The code
   can consume proposal_* and tracklet_* arrays, but behavior depends on their
   coverage and quality.

4. Latest-artifact IsSameObject probe should be rerun on the new overlays/GIFs.

5. Fourth-object / ordinal grounding remains a weak diagnostic unless a dataset
   provides reliable instance-order evidence. The current design correctly does
   not claim to solve that information bottleneck.
```

## 5. Why not deploy everything from the external repos

The rule is not "copy every file".  The rule is:

```text
deploy the invariant if it preserves posterior belief authority and
modality-missing scaling;
reject it if it creates hard visual truth, hard labels, or action-time
reconstruction dependency.
```

Rejected items and reasons:

```text
blind SAM:
  empirically noisy in CALVIN; fragments background/robot parts and is not
  language/contact/task aware.

full MetaSlot VQ posterior:
  useful for image-centric grouping, but too visual-prototype-specific for
  heterogeneous robotics datasets.

SlotLifter NeRF renderer:
  requires many-view/rendering assumptions; CALVIN is static+wrist RGB-D,
  not a many-view object reconstruction benchmark.

full reconstruction decoder:
  would add an image-generation target to a control-time belief router and
  risks optimizing appearance instead of action-relevant object files.

online IsSameObject weak loss:
  without real same-object labels it would teach the model its own noisy
  pseudo-labels.  Keep it as an audit/probe until tracklet/object labels are
  strong enough.
```

## 6. Current validation sequence

Completed A7 short validation:

```text
picf_a7_active_support_dedup_300_20260520
class: slot-comprehensive frozen-policy short validation
```

Answered question:

```text
Can active owner localization remain correct while downstream/context duplicate
support rows are prevented from re-entering the action-visible owner set?
```

Observed answer:

```text
yes for short validation:
  active support overlap <= 0.0367
  downstream support overlap ~= 0.50 instead of the previous 0.76+ rebound
  loss_object_explanation_point falls to 1.7866 by step 300
  active overlays remain on the sidecar mask
```

Next validation:

```text
longer frozen-pretrain/action-aware training and CALVIN/video evidence.
```

Do not add another patch for this same failure mode unless downstream-visible
overlap again rises into the old failure band or overlays show active owner
localization leaving the sidecar mask.

## 7. Acceptance keys

For a short run to pass:

```text
owm_proposal_tokens > 0 when proposal sidecar is configured
owm_tracklet_tokens > 0 when tracklet sidecar is configured
aqr_object_candidate_assignment_max high
aqr_object_candidate_coverage_mean nonzero
aqr_object_candidate_background_mean low only when row-specific support exists
posterior_owner_transport_active_count >= 1 on task-object frames
posterior_owner_transport_active_confidence_mean nonzero
posterior_owner_transport_active_dist_mean low relative to proposal/mask scale
aqr_active_same_role_support_overlap_max low
aqr_active_same_role_object_core_overlap_max low
posterior_binding_signature_measurement_score_std nonzero when real pairwise
  evidence exists
posterior_binding_signature_measurement_dispersion_gate_mean rejects common-mode
  evidence when pairwise information is absent
```

For overlays:

```text
mask_only:
  sidecar/contact evidence should be visible and task-relevant.

mask_active:
  active object owner should be on or near the mask, not on gripper-only pixels.

with_gray:
  reserve/context files may exist but should not be treated as active object
  owners.

sidecar_proposals:
  proposal box/mask should be inspected for task relevance; if it is bad, the
  run tests sidecar noise handling, not object binding.
```

## 8. Final engineering conclusion

The implementation is not a toy minimal patch. It contains the major
belief-compatible mechanisms needed to align PICF with recent object-centric
slot literature:

```text
slot competition:
  deployed.

adaptive quality / dynamic active count:
  deployed in fixed-file belief form.

object/background residual:
  deployed.

pairwise same-object binding subspace:
  deployed.

persistent file identity memory:
  deployed.

object-mask/tracklet sidecar measurement:
  deployed as optional typed evidence.

tactile-to-object binding:
  deployed.

posterior owner transport:
  deployed.

run taxonomy / no-SAM guard:
  deployed.
```

The version is therefore code-level complete for the latest-slot PICF design.
The only honest remaining statement is:

```text
Short structural validation is now positive. Behavior must still be accepted
by a guarded longer co-training run and CALVIN/video evidence.
```

If those fail, the next action is not to import another external slot module
blindly.  The next action is to inspect which acceptance key failed and decide
whether the failure is sidecar quality, posterior transport strength, active
file selection, or semantic/action co-training pressure.
