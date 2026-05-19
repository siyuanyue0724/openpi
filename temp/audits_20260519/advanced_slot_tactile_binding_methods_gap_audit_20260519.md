# Advanced Slot / Object-Binding / Tactile Methods Gap Audit

Date: 2026-05-19

Status: literature/code comparison for PICF-AQR-OWM slot binding.  This is the
companion to
`temp/audits_20260519/picf_current_slot_binding_file_by_file_audit_20260519.md`.

Primary sources used:

```text
MetaSlot, arXiv:2505.20772
  https://arxiv.org/abs/2505.20772
  local code: temp/paper_code_20260518/MetaSlot

QASA, arXiv:2601.12936
  https://arxiv.org/abs/2601.12936

Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?,
arXiv:2510.24709
  https://arxiv.org/abs/2510.24709

SlotVLA, arXiv:2511.06754
  https://arxiv.org/abs/2511.06754

OA-WAM, arXiv:2605.06481
  https://arxiv.org/abs/2605.06481

OmniVTA, arXiv:2603.19201
  https://arxiv.org/html/2603.19201v1

Slot Attention / SAVi / AdaSlot / OCL code for mature implementation patterns:
  temp/paper_code_20260518/slot-attention-video
  temp/paper_code_20260518/AdaSlot
  temp/paper_code_20260518/object-centric-learning-framework

SlotLifter code for 3D point/ray slot lifting:
  temp/external_repos/SlotLifter
```

## 1. What Recent Mature Methods Usually Do

### 1.1 Evidence competition is slot-normalized, not independent per query

Slot Attention and SAVi use inverted dot-product attention:

```math
a_{j,i}
=
\operatorname{softmax}_{j}
\left(
q_j^\top k_i / \sqrt{d}
\right)
```

Then each slot receives a token-normalized weighted mean:

```math
\tilde{a}_{j,i}
=
\frac{a_{j,i}}{\sum_i a_{j,i}+\epsilon},
\quad
u_j=\sum_i\tilde{a}_{j,i}v_i.
```

Implementation evidence:

```text
temp/paper_code_20260518/AdaSlot/ocl/perceptual_grouping.py:
  attn = softmax over slots/heads
  attn = attn / attn.sum over tokens
  updates = weighted token mean

temp/paper_code_20260518/slot-attention-video/savi/modules/attention.py:
  InvertedDotProductAttention uses query-axis softmax.
```

PICF match:

```text
PICF has AQR readers plus `_aqr_same_role_support_competition`.
It approximates evidence competition but is not a full reconstruction-driven
Slot Attention loop.
```

PICF gap:

```text
Object evidence can still be query/routing dependent.  It does not yet have a
global slot-explains-all-tokens decoder as a production objective.
```

### 1.2 Background / empty / no-object capacity is explicit

Mature slot systems do not force every slot to be an object:

```text
SAVi can prepend background boxes or use segmentation/background init.
SlotLifter prepends an empty/background slot in the decoder.
AdaSlot memory removes empty/background/duplicated masks.
QASA dynamically selects high-quality slots instead of forcing all K slots.
```

PICF match:

```text
PICF has dustbin/no-object rows, active/context/reserve slots, posterior file
competition, and birth competition.
```

PICF gap:

```text
The no-object decision is still a hand-designed belief-filter gate, not a
learned slot-quality selector equivalent to QASA.
```

### 1.3 Variable cardinality is handled by selection/dedup/prototypes

MetaSlot explicitly targets the fixed-slot-count problem.  Its abstract states
that fixed slot count can cause one object to split into multiple parts; it
uses a VQ prototype codebook, aggregate-and-deduplicate, and progressive noise
to stabilize aggregation.

QASA similarly argues that slot-count penalties can conflict with reconstruction
and instead decouples slot selection from reconstruction through slot quality.

PICF match:

```text
PICF has active-slot/file competition and can cap active files per role.
```

PICF gap:

```text
PICF does not have:
  global VQ prototype codebook
  learned slot-quality selector
  explicit aggregate-and-deduplicate module
  learned termination/selection objective for object files
```

This is a real maturity gap.  It does not mean PICF is wrong; it means PICF's
current active/reserve mechanism is a belief-filter analogue, not the full
MetaSlot/QASA solution.

### 1.4 Object binding is pairwise/quadratic and must be probed

The Object Binding paper argues that `IsSameObject` can be decoded from ViT
patch embeddings using a quadratic similarity probe; it reports that same-object
information is low-dimensional and can guide attention.

PICF match:

```text
binding_signature_proj
centered support-weighted signatures
linear + diagonal-quadratic + low-rank pairwise score
double-center / z-score calibration
posterior binding-signature memory
```

PICF gap:

```text
The subspace is not yet independently calibrated on latest PICF artifacts.
The offline probe exists as a script/idea, but latest-run acceptance is not done.
```

Strict implication:

```text
Do not claim the binding subspace is solved until a latest artifact probe shows
same-object vs different-object separation over V-JEPA/static/wrist/point/
tactile/sidecar evidence.
```

### 1.5 Robotics slot work often uses object annotations or explicit temporal tracking

SlotVLA is relevant because it is a 2025/2026 robotic manipulation slot paper.
Its abstract says LIBERO+ provides box/mask labels and instance-level temporal
tracking, and SlotVLA uses slot-based visual tokenization and relation-centric
decoding for task-relevant action embeddings.

PICF match:

```text
PICF has object/relationship-like roles, contact rows, task-owner proposals,
and optional tracklet/proposal sidecars.
```

PICF gap:

```text
CALVIN does not provide the same clean box/mask/temporal object annotations.
PICF therefore uses weak contact-motion/point/proposal sidecars.  This is less
supervised and more fragile.
```

Strict implication:

```text
PICF cannot honestly claim SlotVLA-level object annotation support unless
contact-motion masks / tracklets / object candidates are generated and validated
over the dataset.
```

### 1.6 Object-addressable world models split identity from content

OA-WAM decomposes frame state into a robot slot plus object slots, with a
persistent address vector and time-varying content vector.  It routes attention
through address-only keys and resets the address slice at each transformer layer.

PICF match:

```text
posterior.slot_address
posterior.slot_content
binding address terms
posterior binding-signature memory
cache address retrieval
```

PICF gap:

```text
PICF does not implement OA-WAM's exact address-only attention reset inside every
transformer layer.  PICF uses a belief-filter address/content state and gated
binding scores instead.
```

This is not necessarily a bug:

```text
OA-WAM is an action/world transformer; PICF is a typed evidence router plus
PI0.5 action prefix.  Directly copying the entire transformer-layer reset would
be a major architecture replacement, not a local slot-binding fix.
```

### 1.7 Tactile is contact-local, temporally compressed, and contact-gated

OmniVTA models tactile as spatio-temporal latent fields, emphasizes dynamic and
amplitude contact regions, predicts tactile evolution, and uses contact
probability for adaptive visuo-tactile fusion.  It also uses high-frequency
tactile observations aligned with lower-frequency visual observations.

PICF match:

```text
AnyTouch features
tactile contact probability / hysteresis
tactile group routing
tactile native reread
tactile_attach_to_object_owner
point-tactile spatial alignment
owner/contact transport into object files
```

PICF gap:

```text
PICF does not have OmniVTA's tactile VAE / tactile world-model pretraining.
It does not reconstruct a continuous tactile deformation field.
It maps tactile through calibration/contact positions and object owner routing,
which is appropriate for CALVIN but weaker than dedicated tactile-world training.
```

## 2. Module-by-Module Gap Table

| Module | Mature method pattern | PICF current | Verdict |
|---|---|---|---|
| Slot attention competition | Query-axis softmax, recurrent update, token-normalized slot updates | AQR modality readers plus same-role competition | Partial |
| Object/background explanation | Object masks plus background/no-object residual | OEML masks and dustbin exist | Partial; losses guarded |
| Adaptive cardinality | VQ prototypes / slot quality / learned active subset | Active/reserve gates and file caps | Partial; no codebook/quality selector |
| Duplicate prevention | Dedup by codebook, mask IoU, slot quality | support/geometry duplicate demotion | Partial |
| Temporal identity | Memory table, stale counter, slot contrast, tracking annotations | posterior files, binding signature memory, optional tracklets | Partial; tracklet coverage needed |
| Object identity | persistent address + content, address-only routing in OA-WAM | slot_address/content plus gated address scores | Partial |
| Same-object subspace | explicit IsSameObject probe | structural binding subspace | Needs latest probe |
| Tactile binding | contact-local tactile world modeling and gating | contact probability, tactile-to-object owner, point alignment | Correct direction; weaker |
| Robotics object labels | SlotVLA uses boxes/masks/tracks | weak sidecar masks/tracklets only | Data gap |
| Action integration | relation-centric or object-addressable action decoder | PI0.5 prefix over corrected belief | Coherent but different |

## 3. What PICF Should Not Blindly Copy

### 3.1 Do not revive blind SAM as a production source

Reason:

```text
Earlier CALVIN tests showed class-agnostic SAM boxes often select walls,
drawer sides, robot protrusions, or visually changing but task-irrelevant
regions.  That violates the object-owner belief contract.
```

Keep:

```text
generic proposal schema
contact/task/tracklet-aware sidecars
manual/diagnostic visualizations
```

Reject:

```text
blind SAM proposal memory as default
```

### 3.2 Do not force every physical row to become an object

Reason:

```text
Modern slot methods all require background/no-object/empty capacity or adaptive
selection.  Forcing all rows into object losses caused historical PICF overlap
and duplicate-file failures.
```

### 3.3 Do not treat sidecar masks as hard truth

Reason:

```text
Contact-motion masks are useful weak evidence, but noisy.  They can miss object
parts or include gripper/contact artifacts.  They should provide responsibility
and covariance, not hard labels.
```

## 4. What PICF Should Add or Finish

### 4.1 Required short-term: latest artifact IsSameObject probe

Build labels from weak but inspectable sources:

```text
same object:
  same tracklet id
  same sidecar object candidate
  same high-confidence posterior owner
  nearby point cluster under contact

different object:
  different sidecar candidate
  spatially separated point clusters
  different stable posterior owners
```

Probe:

```math
\operatorname{score}(i,k)
=
z_i^\top W z_k
```

Acceptance:

```text
AUC / accuracy over latest artifacts
layer/view/modal separation
ablation: remove projected binding component and verify binding metrics worsen
```

### 4.2 Required short-term: sidecar/tracklet coverage manifest

For each CALVIN segment:

```text
sidecar mask count
tracklet count
proposal confidence
contact confidence
covered frames
object-center stability
known failure flags
```

Acceptance:

```text
training run logs nonzero proposal/tracklet tokens for expected samples
overlay shows object candidates on manipulated objects, not background change
```

### 4.3 Required medium-term: quality-guided active object selection

PICF analogue of QASA:

```math
q_j
=
f(
support\_mass_j,
support\_entropy_j,
object\_explanation\_quality_j,
duplicate\_overlap_j,
owner\_confidence_j,
innovation_j
)
```

Then:

```math
a_j=\operatorname{TopKOrThreshold}(q_j)
```

This should replace purely rule-weighted active gates once enough diagnostics
exist.

### 4.4 Required medium-term: prototype/dedup initialization, not blind fixed slots

PICF-compatible MetaSlot analogue:

```text
keep posterior files as belief state;
add a proposal/prototype initializer for births only;
do not replace posterior files with an image-only VQ codebook.
```

Candidate:

```math
z_i^{candidate}
\rightarrow
\operatorname{VQCodebook}(z_i)
\rightarrow
\operatorname{dedup}
\rightarrow
birth proposal
```

This belongs in birth/proposal initialization, not in the persistent posterior
state itself.

### 4.5 Required long-term: tactile object assignment beyond fingertip position

PICF currently uses calibrated contact positions and point/tactile alignment.
For stronger tactile object binding:

```text
contact point -> nearest candidate object point cluster / mask / tracklet
tactile latent -> contacted object file
contact confidence -> owner transport covariance
```

This is feasible without manual labels, but requires dataset-level sidecar
coverage and probe acceptance.

## 5. Answer to the User's Specific Question

Question:

```text
Are the 2025+ tactile/object-slot binding issues completely solved?
```

Strict answer:

```text
No.
```

What is solved at code level:

```text
1. PICF no longer has only hidden-cosine/geometry binding.
2. It has projected pairwise same-object binding signatures.
3. It has object/background explanation scaffolding.
4. It has file competition and birth competition.
5. It has posterior owner transport with precision fusion.
6. It can attach tactile evidence to object owner rather than a separate
   effector object.
```

What is not solved:

```text
1. No MetaSlot/QASA-level adaptive slot cardinality and learned slot-quality
   selector.
2. No completed latest-run IsSameObject artifact probe.
3. No guaranteed dense sidecar/tracklet coverage over the full dataset.
4. No tactile-world pretraining equivalent to OmniVTA.
5. No proof that owner transport remains stable over long training.
6. No strong ordinal/fine-instance solver.
```

Final assessment:

```text
PICF is no longer a naive or obviously immature slot-binding implementation.
It is a coherent belief-state adaptation of modern object-centric principles.
However, compared with the most mature 2025-2026 slot/tactile methods, it is
still missing adaptive learned slot quality/prototype selection and latest
artifact-level binding validation.
```

