# Slot/Object-Centric Paper-Code / Math / Dataflow Follow-Through

Date: 2026-05-18

Scope: audit whether the current PICF-AQR-OWM contact/proposal/anchor path is
aligned with recent slot/object-centric methods, and identify which ideas should
be copied as invariants versus which should not be transplanted into PICF.  This
is a local TEMP audit and intentionally treats SAM as a historical negative
proposal source rather than the primary architecture reference.

## 1. Local Paper-Code Repositories Checked

```text
temp/paper_code_20260518/MetaSlot
temp/paper_code_20260518/slotcontrast
temp/paper_code_20260518/Deformable-DETR
temp/paper_code_20260518/DINO
```

Primary slot/object references:

```text
MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning
  paper: https://arxiv.org/abs/2505.20772
  code:  https://github.com/lhj-lhj/MetaSlot

SlotContrast: Distilling Object-Centric Learning from Motion and Temporal Contrast
  paper: https://arxiv.org/abs/2501.08484
  code:  https://github.com/martius-lab/slotcontrast

QASA: Quality-Guided K-Adaptive Slot Attention for Object-Centric Learning
  paper: https://arxiv.org/abs/2601.12936
```

Secondary evidence:

```text
Deformable-DETR / DINO:
  reference-point, denoising-query, local evidence sampling patterns.

SAM:
  historical negative evidence for blind class-agnostic masks in CALVIN.
  It is not the maintained PICF slot reference.
```

## 2. Paper-Code Details Actually Inspected

### 2.1 MetaSlot

Relevant files:

```text
temp/paper_code_20260518/MetaSlot/object_centric_bench/model/metaslot.py
```

Observed mechanisms:

```text
from_slots_get_initial_slots(...):
  selects slot initial states through a global VQ prototype codebook and masks
  duplicate selected codebook entries.

inverted_scaled_dot_product_attention(...):
  uses inverted slot attention, where evidence tokens compete over slots before
  normalization over tokens.

VQ.update_codebook(..., smask=...):
  updates codebook entries only with valid non-duplicate slots.
```

Mathematical pattern:

```math
z_i \rightarrow k_i, v_i
```

```math
a_{ij}
=
\operatorname{softmax}_{j}
\left(
  \frac{q_j^\top k_i}{\sqrt d}
\right)
```

then token-normalized aggregation:

```math
\tilde a_{ij}
=
\frac{a_{ij}}{\sum_{i'} a_{i'j}+\epsilon}
```

and adaptive duplicate suppression:

```math
valid_j = \mathbf 1[
  codebook\_id_j \notin \{codebook\_id_{<j}\}
]
```

PICF implication:

```text
The mature principle is not "generate many fixed anchors and hope unused ones
stay harmless".  It is: evidence must compete for slots, duplicate active slots
must be demoted, and background/context capacity must be explicit.
```

PICF already follows this principle through:

```text
_aqr_same_role_support_competition()
_aqr_active_slot_mask()
_aqr_downstream_slot_weights()
_posterior_file_competition()
posterior birth/file competition gates
```

PICF does not currently copy MetaSlot's global VQ codebook.  That is deliberate:
PICF's identity state is a posterior belief state with address/content, geometry,
uncertainty, action-conditioned transition, and recycle gates.  A pure image-OCL
codebook would not be a drop-in replacement without a new world-state likelihood.

### 2.2 SlotContrast

Relevant files:

```text
temp/paper_code_20260518/slotcontrast/slotcontrast/losses.py
temp/paper_code_20260518/slotcontrast/slotcontrast/modules/groupers.py
```

Observed mechanisms:

```text
Slot_Slot_Contrastive_Loss:
  normalizes slots, compares adjacent-time slot vectors, and applies a
  contrastive loss where same-index temporal pairs are positives.

SlotAttention:
  iterative slot update with competition over input tokens.
```

Mathematical pattern:

```math
s_{t,j} = \frac{h_{t,j}}{\|h_{t,j}\|_2}
```

```math
\ell_{j,k}
=
\frac{s_{t,j}^{\top}s_{t+1,k}}{\tau}
```

```math
L_{slotcontrast}
=
\operatorname{CE}(\ell, j)
```

PICF implication:

```text
Temporal same-object consistency is a real signal, but it is fragile under slot
permutation/recycle.  In PICF it should remain detached/guarded unless matching
is stable.  The correct maintained behavior is not to force slot-JEPA or
binding-consistency on by default; the acceptance signal should first come from
anchor health, posterior recycle, object-core overlap, and overlay inspection.
```

PICF current status:

```text
slot_jepa/support_pred/binding_consistency stay default-zero.
same-object subspace is used structurally in binding, not as a strong global
loss that can corrupt posterior identity.
```

### 2.3 QASA

QASA is used as paper-level evidence because it explicitly targets adaptive slot
cardinality.  The relevant architecture principle is:

```text
slot count should be quality/adaptivity controlled, not a blind fixed-object
claim.
```

PICF implication:

```text
The production-safe analog is active/context/reserve routing:
  active slots feed downstream action/readout strongly;
  context slots preserve low-weight environment evidence;
  reserve/dustbin slots absorb background or duplicate rows without becoming
  full object files.
```

This is why PICF should not simply delete inactive anchors.  Dense evidence
stays available; only downstream object-file pressure is gated.

## 3. Current PICF Dataflow to Compare

```text
CALVIN observation
  -> V-JEPA static/wrist temporal typed memory
  -> PaliGemma text/image support
  -> Sonata point support
  -> AnyTouch/contact support
  -> contact-motion proposal sidecar and proposal-seeded tracklets
  -> AQR typed support readers
  -> same-role support competition
  -> active/context/reserve slot filter
  -> posterior binding/correction
  -> posterior file competition / birth competition
  -> PI0.5 prefix/action path
```

Sidecar/proposal mask path:

```text
contact/task score over current-frame static-view points
  -> compact connected components
  -> proposal boxes + sparse proposal masks
  -> runtime mask-to-current-point likelihood
  -> proposal/point priors
  -> AQR / posterior correction
```

The maintained invariant:

```text
No sidecar, proposal, SAM-like mask, tracklet, or diagnostic slot is allowed to
overwrite posterior truth.  They are measurement evidence only.
```

## 4. Mathematical Consistency Check

PICF belief state:

```math
b_t(s)
\propto
p(o_t \mid s_t)
\int p(s_t \mid s_{t-1}, a_{t-1}) b_{t-1}(s_{t-1}) ds_{t-1}
```

AQR approximates the measurement assignment:

```math
p_{j,i}^{(m)}
=
\operatorname{softmax}_i
\left(
  \frac{q_j^\top k_i^{(m)}}{\sqrt d}
  + b_{j,i}^{(m)}
\right)
```

Same-role competition approximates inverted slot competition:

```math
\bar p_{j,i}
=
\frac{p_{j,i}}
{\sum_{j' \in role(j)} p_{j',i}+\epsilon}
```

Then residual mix:

```math
p'_{j,i}
=
(1-\lambda_c)p_{j,i}
+\lambda_c \bar p_{j,i}
```

Active/context/reserve routing implements adaptive cardinality:

```math
w_j =
\begin{cases}
1, & j \in A_{active} \\
\lambda_{ctx}, & j \in A_{context} \\
0, & j \in A_{reserve}
\end{cases}
```

Downstream object-file pressure is then:

```math
q^{down}_j = w_j q_j
```

Dense typed memory is not pruned:

```math
M^{vjepa}, M^{pg}, M^{point}, M^{tactile}, M^{proposal}
\text{ remain available to all valid reads.}
```

Proposal-mask point bridge:

```math
L_i^{mask}(k)
=
\frac{
  \sum_{s \in S_k} w_s
  \exp(-\|u_i-u_s\|^2/(2\tau_m^2))
}{
  \sum_{s \in S_k} w_s+\epsilon
}
```

This is a measurement prior:

```math
P(i|k) =
\frac{L_i^{mask}(k) q_k}
{\sum_{i'}L_{i'}^{mask}(k)q_k+\epsilon}
```

not a hard object label.

## 5. Why We Do Not Copy MetaSlot Wholesale

Direct MetaSlot transplant would require:

```text
global VQ prototype codebook objective
image reconstruction / object-centric OCL training loop
slot decoder semantics
dataset-level slot validity curriculum
```

PICF has a different likelihood:

```text
action-conditioned belief filter with posterior authority and typed robot
measurements.
```

The self-consistent copy is therefore not codebook-for-codebook, but invariant
for invariant:

```text
MetaSlot duplicate mask
  -> PICF active/context/reserve demotion + posterior file competition.

MetaSlot inverted attention
  -> PICF same-role support competition.

MetaSlot adaptive slot number
  -> PICF active slot max/min per role plus context/reserve routing.

SlotContrast temporal consistency
  -> PICF guarded same-object/binding audit; predictive losses stay zero until
     permutation/recycle stability is proven.
```

## 6. Failure Modes Checked Against Current Code

### Raw same-role overlap remains high

This is not by itself failure after active/context/reserve routing.  Raw rows
include reserve/dustbin/context anchors.  Acceptance uses:

```text
aqr_active_same_role_support_overlap_max
aqr_active_same_role_object_core_overlap_max
posterior_recycle_rate
posterior_identity_switch_rate
anchor overlays
```

### Background/context tokens disappearing

Rejected.  PICF does not drop dense tokens.  Reserve/context gating only reduces
downstream object-file pressure from redundant anchors.  The dense V-JEPA/PG/
point/tactile memory remains readable.

### Proposal masks becoming hard labels

Rejected.  Proposal masks affect point likelihoods and proposal reads only.
They do not set posterior identity, do not bypass posterior correction, and do
not write action targets.

### Blind SAM becoming production evidence

Rejected.  Blind SAM proposals were noisy in CALVIN and remain archived/off.
The maintained sidecar is contact/task/tracklet-aware typed evidence.

## 7. Current Acceptance Run Contract

The next validation must be a mask-sidecar, tracklet-enabled, anchor-only
diagnostic:

```text
trainable_scope = anchor_only
action/PaliGemma frozen
NPROC_PER_NODE = 2
num_train_steps = 1000
FORCE_SIDECAR = 0 if mask sidecar manifest is already complete
FORCE_TRACKLETS = 0 if tracklet manifest is already complete
```

Required metrics:

```text
owm_proposal_tokens > 0
proposal_mask fields present in sidecar npz
owm_tracklet_tokens > 0
aqr_active_same_role_support_overlap_max bounded
aqr_active_same_role_object_core_overlap_max low
posterior_recycle_rate not saturated
loss_anchor_pv not divergent
anchor overlay task-object coverage improves versus no-mask sidecar
```

If this run fails, the next fix should target proposal/contact score quality or
active/context routing thresholds, not blind SAM revival or wholesale slot-code
transplant.

## 8. Current Verdict

The maintained PICF design is aligned with recent slot/object-centric theory at
the invariant level:

```text
competition
adaptive active slots
duplicate demotion
background/context capacity
temporal consistency as guarded evidence
posterior-authoritative belief update
```

It is not mathematically correct to claim that all object-binding science is
solved before the 1000-step mask-sidecar diagnostic and later long-run behavior
acceptance.  It is also not correct to replace PICF with a pure image-slot
module.  The coherent path is the current PICF belief filter with the slot
invariants above, validated by active-slot and overlay evidence.

## 9. Proposal Reference-Anchor Transport

The step-100 mask-sidecar diagnostic exposed a narrower failure than generic
slot collapse:

```text
proposal evidence:
  present and selected.

physical anchor geometry:
  not reliably transported into the selected proposal.
```

The mature object-query lesson from Deformable-DETR/DINO-like detectors is not
that proposals should become labels. It is that query slots need explicit
reference geometry before attention can converge reliably. For PICF the
belief-filter translation is:

```math
s_p = TaskOwnerScore(p)
```

```math
M_{p,n} = P(point_n \mid proposal_p)
```

```math
\rho_{r,n}
=
Normalize_n
\left(
  M_{p_r,n}^{\gamma}
\right)
```

where `p_r` are the top task/contact proposals assigned to a small number of
physical measurement rows. These rows are mixed with existing point priors:

```math
\pi'_{r,n}
=
Normalize_n
\left(
  (1-\lambda_r)\pi_{r,n}
  +
  \lambda_r \rho_{r,n}
\right)
```

with:

```math
\lambda_r
=
\lambda_{seed} \cdot clamp(s_{p_r},0,1)
```

The associated proposal token may weakly seed row content:

```math
q'_r
=
q_r
+
\lambda_{tok}s_{p_r}
(z_{p_r}-q_r)
```

This is mathematically a measurement prior, not identity assignment:

```text
No posterior field is overwritten.
No dense token memory is pruned.
No proposal becomes a hard label.
No action target is derived from the proposal.
```

It therefore preserves the PICF invariants:

```text
typed evidence router:
  still AQR.

authoritative belief state:
  still posterior correction.

background/context capacity:
  still active/context/reserve routing.

future/world-model hooks:
  unchanged and still guarded.
```

Failure tests:

```text
If proposal_anchor_seed_point_max is zero:
  dataflow/configuration failure.

If proposal_anchor_seed_point_max is high but overlays miss the proposal:
  proposal-to-point calibration or point projection failure.

If overlays enter proposal but active overlap explodes:
  active/context/reserve competition failure.

If overlays enter proposal and overlap stays bounded:
  the reference transport repaired the current anchor-placement gap.
```
