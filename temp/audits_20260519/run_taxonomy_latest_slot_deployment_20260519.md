# PICF v2.2 Run Taxonomy and Latest-Slot Deployment Audit

Date: 2026-05-19

Canonical README:

```text
src/openpi/picf/README_v2.2.md
```

Companion audits:

```text
temp/audits_20260519/latest_slot_code_gap_matrix_20260519.md
temp/audits_20260519/picf_current_slot_binding_file_by_file_audit_20260519.md
docs/PICF_AQR_OWM_OBJECT_EXPLANATION_DEPLOYMENT_PLAN_20260518_TEMP.md
docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md
```

Primary 2025-2026 sources checked:

```text
MetaSlot, arXiv:2505.20772, submitted 2025-05-27 and revised 2025-10-08:
  adaptive slot count with codebook-based duplicate aggregation.

QASA, arXiv:2601.12936, submitted 2026-01-19:
  slot-quality-guided K-adaptive selection; selection decoupled from
  reconstruction.

Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?,
arXiv:2510.24709, revised 2026-01-21:
  IsSameObject decoded by quadratic probes and represented in a low-dimensional
  binding subspace.

SlotVLA, arXiv:2511.06754, revised 2026-05-06 and accepted at ICRA 2026:
  object/relation slots for robot manipulation, but with LIBERO+ object boxes,
  masks, and temporal instance tracking.
```

## 0. Immediate Correction

The active A7 run:

```text
picf_a7_slot_quality_selector_anchor1000_20260519
```

is **not** the broad frozen-policy validation run.  Its command contains:

```text
--picf-trainable-scope anchor_only
--lambda-action-pos 0
--lambda-action-rot 0
--lambda-action-gripper 0
--lambda-slot-jepa 0
--lambda-support-pred 0
--lambda-binding-consistency 0
--lambda-aqr-denoising 0
--lambda-anchor-object-pull 1.0
--lambda-slot-quality 0.10
```

Therefore its class is:

```text
anchor_capability_probe
```

Its valid question is:

```text
Given inspected contact/task sidecar evidence, can the PICF object-anchor
router and persistent posterior owner move a task object file onto the
sidecar/contact object without action or PaliGemma pressure?
```

It must not be interpreted as:

```text
full slot/router validation
formal frozen-pretrain co-training
30k production candidate
```

The command classifier added for this purpose is:

```text
scripts/picf_run_contract_audit.py
```

## 1. Run Classes

### 1.1 Anchor Capability Probe

Purpose:

```text
isolate whether anchors/posterior files can be pulled to inspected object
evidence at all.
```

Required contract:

```text
picf_trainable_scope = anchor_only
perception_finetune_mode = frozen
semantic_trainable = false
action losses = 0
predictive identity losses = 0
sidecar/contact proposals enabled when testing object binding
overlay interval <= 50 or 100
```

Allowed nonzero losses:

```text
lambda_anchor_object_pull
lambda_slot_quality
optional tiny OEML measurement losses, only if the question is not pure anchor
pull capability
```

This run is deliberately narrow.  It can prove or disprove anchor localization
capacity, but it cannot prove production co-training stability.

### 1.2 Slot-Comprehensive Frozen-Policy Validation

Purpose:

```text
test the full PICF slot/router/posterior/OEML dataflow while freezing policy
and large pretrained perception/semantic modules.
```

Required contract:

```text
picf_trainable_scope = all
perception_finetune_mode = frozen
semantic_trainable = false
action losses = 0
picf_action_prefix_stopgrad = true
foundation backbones are real encoders, but do not use the convenience
  --use-foundation-backbones flag unless semantic_trainable is intentionally
  allowed
sidecar/contact proposals enabled
blind SAM disabled
```

Important implementation detail:

```text
scripts/picf_core_train.py::_apply_foundation_profile sets
semantic_trainable=True when --use-foundation-backbones is used.

Therefore frozen-PaliGemma validation must spell out the encoders manually:
  --point-backbone sonata
  --visual-mode encoder
  --tactile-mode encoder
  --use-tactile
  --semantic-mode paligemma
  --perception-finetune-mode frozen

and must omit --semantic-trainable.
```

This is the run the user referred to as:

```text
not the super-crippled anchor-only run;
freeze pretrain/PaliGemma/action-head pressure, but allow the PICF slot
binding/router/posterior modules to move together.
```

Recommended guarded loss set:

```text
lambda_anchor_object_pull = 0.25 to 0.50
lambda_slot_quality = 0.02 to 0.05
lambda_object_explanation_point = 0.01 to 0.05
lambda_object_explanation_duplicate = 0.005 to 0.02
lambda_object_explanation_background = 0.005 to 0.01
lambda_mapg_cycle = 0.005 to 0.01
lambda_mapg_support_diversity = 0.002 to 0.005
lambda_aqr_denoising = 0
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
```

Reason:

```text
The active object-binding failure historically appeared before predictive
identity was stable.  The comprehensive validation should test object
responsibility, posterior write-through, and evidence competition without
letting future/prediction losses create another confound.
```

### 1.3 Formal Frozen-Pretrain Co-Training

Purpose:

```text
production-style 30k candidate after the frozen-policy validation passes.
```

Required contract:

```text
picf_trainable_scope = all
perception_finetune_mode = frozen
semantic_trainable = true
action losses enabled
PaliGemma/action/PICF adaptation trainable
V-JEPA/Sonata/AnyTouch pretrained backbones frozen
sidecar/contact proposals enabled only if coverage/quality is accepted
save_interval = 2500
keep_last_checkpoints = 3
anchor overlays every 100 steps
```

Predictive losses remain gated:

```text
slot_jepa/support_pred/binding_consistency stay zero until active-owner
posterior identity is stable in short and mid-run metrics.
```

## 2. Latest Slot/OCL Mechanisms: Current PICF Status

The relevant 2025-2026 methods were grouped by mathematical invariant, not by
code-copy surface area.

### Slot Attention / AdaSlot / OCL

Invariant:

```math
a_{j,i}=\operatorname{softmax}_j(q_j^T k_i),\quad
u_j=\sum_i
\frac{a_{j,i}}{\sum_i a_{j,i}+\epsilon}v_i .
```

PICF-native deployment:

```text
same-role support competition
object-candidate row capacity
background/no-object residual
active/context/reserve gating
posterior file competition
```

Do not replace PICF with a pure image decoder slot loop because PICF is a
belief-state router with typed evidence and PI0.5 action authority.

### MetaSlot

Invariant:

```text
adaptive object count, duplicate aggregation, prototype-conditioned slot birth.
```

PICF-native deployment:

```text
slot-quality active/context/reserve routing
duplicate demotion
birth competition
sidecar/proposal-conditioned seeding
```

Not copied:

```text
full global VQ prototype codebook inside posterior state
```

Reason:

```text
A hard visual prototype codebook is not modality-missing tolerant.  PICF must
scale to datasets without point cloud, tactile, wrist, or sidecars.  A future
prototype bank may be used as a birth proposal initializer only, never as
posterior truth.
```

### QASA

Invariant:

```text
slot quality / keep-or-dustbin decision is decoupled from reconstruction and
from action loss.
```

PICF-native deployment:

```text
PicfSlotQualityState
aqr_slot_quality_head
object/no-object/duplicate logits
loss_slot_quality
active/context/reserve downstream weights
```

This is currently one of the most relevant deployed mechanisms.

### Object Binding in Pretrained ViTs

Invariant:

```text
same-object relation is a pairwise / quadratic subspace, not just hidden
cosine or geometry.
```

PICF-native deployment:

```text
binding_signature_proj
support-weighted binding_signature
quadratic/low-rank binding terms
posterior binding-signature memory
calibrated relative binding diagnostics
offline IsSameObject probe script
```

Remaining acceptance:

```text
run the latest artifact IsSameObject probe on anchor_overlays / exported
features once the current run has enough frames.
```

### SlotVLA / Robot Object-Relation Slots

Invariant:

```text
object slots and relation slots are useful for manipulation, but the published
robotics setup relies on object boxes/masks/tracks in LIBERO+.
```

PICF-native deployment:

```text
object files
relation/ordinal diagnostics
task-owner sidecars
sidecar/contact weak masks
PI0.5 action prefix over corrected belief
```

Data gap:

```text
CALVIN sidecars are weak contact-motion proposals, not curated object labels.
PICF cannot honestly claim SlotVLA-level supervision unless sidecar/tracklet
coverage is completed and validated.
```

### Visuo-Tactile / Tactile Object Binding

Invariant:

```text
tactile/contact must attach to the contacted object owner, not become a
separate gripper-owned object file.
```

PICF-native deployment:

```text
tactile_attach_to_object_owner
tactile evidence probability gates
object_candidate_owner_point_priors
posterior_owner_transport
object-only physical rows for current probes
```

Not copied:

```text
full tactile world-model pretraining / 60Hz tactile reflex controller
```

Reason:

```text
The current failure is object-owner localization under offline CALVIN evidence,
not a missing real-time tactile policy.
```

## 3. What Is Still Missing

### 3.1 Missing Evidence, Not Missing Architecture

```text
full dataset sidecar/contact-motion coverage
full dataset tracklet coverage
latest-run IsSameObject artifact probe
step-1000 GIF/manual overlay acceptance
CALVIN/video behavioral eval
```

### 3.2 Intentionally Not Deployed

```text
blind SAM proposal source
full image/object reconstruction decoder as action-time truth
full MetaSlot VQ codebook inside posterior state
SlotLifter NeRF/many-view renderer
OA-WAM transformer-layer address reset inside PI0.5
tactile VAE/reflex controller
```

These are not omitted because of effort.  They are omitted because they either
target a different data contract, need labels CALVIN does not provide, or would
violate posterior-authoritative belief routing.

## 4. Next Deployment Decision

Current A7 should finish as:

```text
anchor_capability_probe
```

After it reaches enough overlays, run:

```bash
PYTHONPATH=src uv run --no-sync python scripts/picf_run_contract_audit.py \
  --command-file /path/to/run_command.txt
PYTHONPATH=src uv run --no-sync python scripts/picf_anchor_overlay_make_gifs.py \
  --overlay-dir /mnt/checkpoints/.../anchor_overlays
PYTHONPATH=src uv run --no-sync python scripts/picf_anchor_run_diagnostic_report.py \
  /mnt/checkpoints/.../run_dir --fail-on-missing
PYTHONPATH=src uv run --no-sync python scripts/picf_owm_same_object_probe.py \
  --anchor-overlays /mnt/checkpoints/.../anchor_overlays
```

If active posterior owners remain on the sidecar/contact masks, then launch
the slot-comprehensive frozen-policy validation.  If they do not, do not
launch formal co-training; fix posterior owner closure first.

## 5. Strict Current Conclusion

The current code is **not** missing an obvious latest-slot module that should be
blindly copied wholesale.  It already contains the PICF-compatible forms of
the major invariants:

```text
slot competition
adaptive quality / active-context-reserve
background residual
duplicate demotion
same-object binding subspace
sidecar/contact object measurement
tactile-to-object owner attachment
posterior owner transport
posterior binding-signature memory
```

The unresolved question is behavioral:

```text
Do these mechanisms keep the active posterior object file on the task object
across enough steps/tasks, while action/PaliGemma pressure is absent, and then
under co-training pressure?
```

That requires the current anchor probe plus the next frozen-policy validation,
not another unprincipled module insertion.
