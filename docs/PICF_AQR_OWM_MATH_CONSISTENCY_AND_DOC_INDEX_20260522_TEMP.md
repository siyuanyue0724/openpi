# PICF-AQR-OWM Mathematical Consistency And Document Index

Date: 2026-05-22

Status: current local mathematical/documentation gate.  This document is a
navigation layer and consistency proof.  It does not replace runtime logs,
CALVIN videos, or the 30k behavior gate.

## 1. Canonical Reading Order

Use this order when resuming the project.  Do not start from old SAM or raw
overlap threads.

```text
1. src/openpi/picf/README.md
   Directory entry.  It points to the current live README and the latest local
   full audit.

2. src/openpi/picf/README_v2.2.md
   Live architecture and operator record.  It is intentionally long because it
   contains the historical run ledger, but current entries are near the top and
   at the "current docs" handoff section.

3. docs/PICF_AQR_OWM_LOCAL_FULL_AUDIT_20260522_TEMP.md
   Current local code/math/dataflow proof.  Use this as the code-level gate
   before judging behavior.

4. docs/PICF_AQR_OWM_SLOT_PAPER_DATAFLOW_COMPARE_20260521_TEMP.md
   Current paper-to-PICF mechanism mapping.  Use this when deciding whether a
   proposed module is mathematically necessary, rejected, or a future candidate.

5. docs/PICF_AQR_OWM_OPEN_ISSUE_TRACKER_20260517_TEMP.md
   Historical issue ledger.  Use only after reading the two current docs above,
   because this file contains superseded SAM/raw-overlap hypotheses as well as
   the final dispositions.
```

## 2. Current Model Philosophy

PICF is not a generic image slot autoencoder.  It is a multimodal predictive
belief router:

```text
goal:
  maintain persistent object files that can bind evidence from visual, point,
  tactile, language, temporal, cache, proposal, and tracklet sources;

constraint:
  any modality may be missing in a large-scale heterogeneous dataset;

therefore:
  object identity cannot be a hard RGB mask, hard visual prototype, hard SAM
  label, or image reconstruction target.
```

The maintained state variable is:

```math
B_t = \{b_{t,k}\}_{k=1}^{K}
```

where each object file contains:

```text
latent token/content
geometry belief
support signature
binding signature
address/content identity state
recycle/birth confidence
typed evidence supports
```

The router is allowed to use weak sidecar/contact/tracklet/proposal evidence,
but posterior authority remains with the belief update:

```math
p(B_t \mid Z_{\le t}, u_{\le t})
\propto
p(Z_t \mid B_t)\,p(B_t \mid B_{t-1}, u_{t-1})
```

No single sidecar source is truth.

## 3. Latest Literature Mechanisms And PICF Mapping

### 3.1 Object Binding In ViTs

Reference:

```text
Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
arXiv:2510.24709 / NeurIPS 2025 Spotlight
```

Relevant mechanism:

```text
same-object information is decoded as a pairwise relation, often with
quadratic/low-rank probes; raw cosine or raw attention is not enough.
```

PICF deployment:

```text
implemented:
  support-weighted binding_signature
  binding_signature_proj
  linear + diagonal quadratic + low-rank pairwise score family
  double-centered/z-scored calibration
  posterior trust gate
  offline same-object probe script

not enabled:
  online IsSameObject BCE

reason:
  current CALVIN has no dense instance-mask labels.  Online BCE on weak labels
  would self-confirm noisy assignments.  PICF uses the paper's equation family
  as runtime evidence and keeps probe training offline.
```

Mathematical invariant:

```math
\tilde{s}_{ij}
= s_{ij}
- \mathbb E_j[s_{ij}]
- \mathbb E_i[s_{ij}]
+ \mathbb E_{ij}[s_{ij}]
```

If `std(\tilde{s})` is too small, the score becomes zero identity evidence.
This prevents common scene saliency from becoming false object identity.

### 3.2 SlotContrast

Reference:

```text
Temporally Consistent Object-Centric Learning by Contrasting Slots
CVPR 2025
```

Relevant mechanism:

```text
video object slots need temporal consistency; recurrent processing alone is not
enough for stable long-term identities.
```

PICF deployment:

```text
implemented:
  posterior file continuity metrics
  tracklet typed memory contract
  persistent binding_signature memory
  matched/pairwise guarded predictive hooks

not enabled by default:
  online slot-contrastive loss
  slot-JEPA/support-pred/binding-consistency losses

reason:
  without reliable temporal object labels or validated tracklet coverage,
  online contrastive pressure can reinforce wrong identities.  Current design
  keeps temporal consistency as posterior memory and diagnostics first.
```

### 3.3 MetaSlot

Reference:

```text
MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning
arXiv:2505.20772
```

Relevant mechanism:

```text
fixed K slots cause duplicates/background rows; duplicate slots should be
masked/demoted and the effective object count should adapt.
```

PICF deployment:

```text
implemented:
  active/context/reserve split
  posterior file competition
  owner/reserve gate
  duplicate/no-object slot quality
  bounded birth/no-object transport

not copied:
  hard VQ codebook as posterior identity

reason:
  PICF must bind cross-modal object files.  A hard visual prototype can lock
  identity to RGB appearance and fail when tactile, point, language, or temporal
  evidence is the available source.  The correct analogue is duplicate demotion
  and persistent multimodal binding memory, not visual VQ truth.
```

### 3.4 STORM / Robotics Slot Adaptation

Reference:

```text
STORM: Slot-based Task-aware Object-centric Representation for robotic
Manipulation, arXiv:2601.20381
```

Relevant mechanism:

```text
frozen foundation visual features can be adapted with lightweight, task-aware
slots; staged learning prevents degenerate slots before policy cotrain.
```

PICF deployment:

```text
implemented:
  frozen-pretrain profile for V-JEPA/Sonata/AnyTouch
  PaliGemma/action-aware staged long run
  frozen-policy structural gates before action-dominant continuation

open behavior question:
  action plateau must be judged by 30k metrics and CALVIN/video evidence, not
  by local code audit alone.
```

## 4. Current Accepted Mechanism Set

These mechanisms are considered mathematically required for the current line:

```text
typed evidence memory:
  V-JEPA multiview temporal tokens
  PaliGemma image/text support
  point evidence
  tactile/contact evidence
  optional tracklet/proposal sidecars

belief update:
  posterior authority
  owner/reserve gate
  posterior file competition
  bounded birth/dustbin transport
  trust-gated binding-signature memory

slot/object routing:
  active/context/reserve split
  low-priority context visibility
  reserve/no-object suppression
  object-candidate background residual
  task-owner proposal/point transport

binding:
  support overlap
  projected pairwise binding subspace
  calibrated relative scores
  gated address/cache identity

diagnostics:
  active-only and with-gray overlays
  raw/active/downstream/reserve overlap separation
  posterior file continuity
  object-pull graph/posterior split
  action_default_equiv compared to 2026-04-22 ablation
```

This is not a random collection of losses.  It is one decomposition:

```text
dense evidence -> weak object candidates -> active/context/reserve routing
-> posterior object files -> action prefix
```

Each loss/metric must attach to one of those edges.  If a term cannot be placed
on this graph, it should not be enabled.

## 5. Explicitly Rejected Or Guarded Mechanisms

### 5.1 Blind SAM

```text
status:
  rejected / archived

reason:
  blind automatic masks selected wall panels, robot protrusions, drawer sides,
  and other non-task fragments.  Their objectness score was not task ownership.

allowed future form:
  prompted or contact/task/tracklet-guided proposal evidence after separate
  quality validation.
```

### 5.2 Hard Sidecar Labels

```text
status:
  rejected

reason:
  sidecars are generated weak evidence.  They can guide measurement routing but
  must not become posterior truth.
```

### 5.3 Full Reconstruction Decoder In The Action Path

```text
status:
  rejected for current action path; future auxiliary only after separate review.

reason:
  full reconstruction optimizes background explanation as strongly as task
  object belief.  It can turn the control router into an image autoencoder.
```

### 5.4 Online IsSameObject BCE

```text
status:
  rejected until reliable same/different object labels exist.

reason:
  with only self-generated weak labels, it risks confirming wrong binding.
```

### 5.5 Hard Visual VQ Posterior Truth

```text
status:
  rejected for current multimodal scaling goal.

reason:
  useful as a MetaSlot image-slot mechanism, but too visual-only for missing
  modality scaling.
```

## 6. Current Runtime Gate Interpretation

Raw overlap is not the primary failure metric anymore.

```text
raw overlap:
  active + context + reserve mixture;
  can saturate because reserve/no-object rows reuse broad capacity.

active overlap:
  action-visible object owner duplication;
  this is a real failure if high.

downstream overlap:
  active plus low-priority context evidence that can reach the action prefix;
  this is a watch metric.

reserve overlap:
  expected to be high when fixed no-object capacity is demoted;
  not a stop signal alone.
```

Current stop conditions:

```text
stop if:
  active overlap enters historical collapse band;
  posterior active duplicate overlap rises;
  recycle saturates;
  object-pull graph/posterior both worsen persistently;
  overlays show active objects leaving task/contact evidence;
  action_default_equiv worsens while structure stays unstable.

do not stop only because:
  raw overlap ~= 1.0;
  disabled slot-JEPA telemetry spikes while lambda_slot_jepa=0;
  reserve rows cluster in with-gray overlay.
```

## 7. Dense Context Future Candidate

The next architecture candidate, if action plateaus while object routing stays
healthy, is a separate dense-context path:

```math
H'_\ell
=
H_\ell
+ \tanh(g_\ell)\,
\mathrm{CrossAttn}
\left(
  Q=\mathrm{LN}(H_\ell),
  K,V=\mathrm{LN}(R(C_{\mathrm{dense}}))
\right)
```

where:

```text
C_dense:
  low-confidence/context/raw V-JEPA/PG/point/task evidence;

R:
  small resampler, not an object posterior;

g_l:
  zero/near-zero initialized gate.
```

This follows the gated cross-attention design philosophy used by Flamingo-style
visual language conditioning and JEPA-VLA/VLA-JEPA-style latent predictive
conditioning.  It is not part of the current 30k acceptance test.

Acceptance rule for this future module:

```text
only add if:
  active object routing remains healthy;
  action loss plateaus relative to 2026-04-22 ablation;
  overlays suggest missing peripheral/dense context rather than object collapse.

do not add if:
  active owner binding is still unstable;
  the problem is sidecar quality;
  the only symptom is raw reserve overlap.
```

## 8. Local Verification Status

The following commands passed on 2026-05-22:

```text
uv run python scripts/picf_latest_slot_deployment_audit.py --fail-on-fail
uv run python scripts/picf_binding_dataflow_math_audit.py --fail-on-fail
uv run python scripts/picf_action_visible_reserve_gate_audit.py --fail-on-fail
uv run python scripts/picf_owm_nontruncated_paper_audit.py --fail-on-fail
uv run python scripts/verify_picf_owm_contract.py
uv run pytest -q src/openpi/picf/core/training_test.py
uv run pytest -q scripts/picf_core_train_test.py
uv run pytest -q src/openpi/picf/core/pipeline_test.py
```

The complete command log and additional audits are in:

```text
docs/PICF_AQR_OWM_LOCAL_FULL_AUDIT_20260522_TEMP.md
```

## 9. Final Code-Level Verdict

Current local verdict:

```text
mathematical contract:
  coherent;

documentation:
  now has a current canonical path;

latest-paper coverage:
  key mechanisms are mapped and either implemented, guarded, or explicitly
  rejected with reasons;

known behavior boundary:
  still requires long-run 30k metrics, overlays, and CALVIN/video evidence.
```

Do not claim:

```text
all behavior problems are solved;
fourth/fine ordinal grounding is solved;
action has surpassed the 2026-04-22 long-run baseline;
raw overlap is fixed.
```

Do claim:

```text
the current code-level architecture is internally consistent;
raw overlap is no longer the right primary failure metric;
blind SAM and hard labels are out of the maintained path;
object binding is represented through calibrated pairwise support evidence;
background/context is retained without forcing every token into an object slot.
```
