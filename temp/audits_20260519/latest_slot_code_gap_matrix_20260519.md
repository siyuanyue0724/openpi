# Latest Slot/OCL Code Gap Matrix for PICF-AQR-OWM

Date: 2026-05-19

Scope: strict comparison between the current PICF object-slot path and the
newer slot/object-binding/visuo-tactile papers and local code snapshots under
`temp/paper_code_20260518` and `temp/external_repos`.

This document is deliberately operational: every paper mechanism is classified
as implemented, deployed in a PICF-native form, rejected for a concrete
mathematical reason, or pending because it requires data coverage / behavior
evidence.

## 0. Current Runtime Under Audit

Active remote run:

```text
picf_a7_slot_quality_selector_anchor1000_20260519
```

Launcher:

```text
run_a7_slot_quality_selector_anchor1000_20260519.sh
```

Purpose:

```text
Anchor/slot capability probe only.
PaliGemma/action losses/pretrained perception are frozen or zeroed.
Object-only physical rows consume contact/task sidecar proposals and tracklets.
Six overlay variants are written every 50 steps.
```

Step-50 evidence copied locally:

```text
temp/a7_slot_quality_step50/
```

Step-50 strict reading:

```text
sidecar evidence is present:
  owm_proposal_tokens ~= 1.41
  owm_tracklet_tokens ~= 80.5
  aqr_object_candidate_assignment_max ~= 1.0
  aqr_object_candidate_coverage_mean ~= 0.94

raw same-role overlap is controlled:
  aqr_same_role_support_overlap_max ~= 0.52
  aqr_same_role_object_core_overlap_max ~= 0.34
  active same-role support overlap in overlay JSON ~= 0.006

posterior closure is still weak:
  posterior_owner_transport_mass_mean ~= 0.012
  posterior_owner_transport_confidence_mean ~= 0.0047
  posterior_owner_transport_dist_to_standard_mean ~= 0.095 m
  active posterior file is not yet visually centered on the mask target
```

Conclusion at step 50:

```text
The current failure mode is not "sidecar did not reach the model".
It is "candidate/object measurement is not yet strongly closed into the
persistent posterior owner file".
```

Do not stop at step 50. The first actionable decision point is step 100,
because the run uses 50 warmup steps and the first overlay is exactly at the
end of warmup.

## 1. Paper-Code Mechanism Inventory

### 1.1 Slot Attention / SAVi / AdaSlot / OCL

Local code:

```text
temp/paper_code_20260518/AdaSlot/ocl/perceptual_grouping.py
temp/paper_code_20260518/object-centric-learning-framework/ocl/perceptual_grouping.py
temp/paper_code_20260518/slot-attention-video/
```

Core mechanism:

```math
\operatorname{attn}_{j,i}
=
\operatorname{softmax}_{j}
\left(q_j^\top k_i / \sqrt d\right)
```

followed by token-normalized slot updates:

```math
\tilde a_{j,i}
=
\frac{a_{j,i}}{\sum_i a_{j,i}+\epsilon},
\quad
u_j=\sum_i\tilde a_{j,i}v_i .
```

PICF status:

```text
Implemented in belief-router form, not copied as a full image decoder.
Current pieces:
  AQR typed modality readers
  same-role support competition
  object-candidate column competition with background residual
  row capacity control
  active/context/reserve gating
```

Reason not to replace PICF with the external module:

```text
The external Slot Attention module is an image/object reconstruction grouping
module. PICF is a sequential belief-state router with posterior file authority,
typed evidence, and PI0.5 action generation. Replacing AQR with a decoder-only
slot loop would remove the posterior/action contract rather than fix it.
```

Required acceptance:

```text
active overlap must remain low and active posterior owner must move toward
the sidecar/contact object; otherwise the issue is posterior closure, not
slot-axis competition.
```

### 1.2 MetaSlot

Local code:

```text
temp/paper_code_20260518/MetaSlot
temp/external_repos/MetaSlot
```

Paper mechanism:

```text
global VQ object prototype codebook
aggregate-and-deduplicate slots
progressive noise to stabilize aggregation
```

PICF status:

```text
Partially deployed in compatible form.
Implemented:
  duplicate demotion
  slot-quality duplicate/no-object/object gates
  file competition
  birth competition
  sidecar/proposal-conditioned initializers

Not deployed:
  global VQ prototype codebook
```

Why the VQ codebook is not copied now:

```text
The MetaSlot codebook is trained for object-centric image reconstruction and
dataset-level visual prototypes. PICF must remain modality-missing tolerant:
some datasets have no point cloud, no tactile, no tracklets, or different
camera layouts. A hard global visual codebook inside the posterior state would
make identity depend on one modality and can conflict with object-addressable
belief updates.
```

PICF-compatible future variant:

```text
Use a prototype bank only as a birth/proposal initializer, never as posterior
truth. It should propose candidate object rows and then pass through the same
owner transport, support competition, and posterior file gates.
```

Current verdict:

```text
Do not deploy full MetaSlot VQ into the live router before step-100/1000
posterior closure evidence. It is not the current bottleneck at step 50:
candidate assignment is already high; posterior transport is weak.
```

### 1.3 QASA

Paper mechanism:

```text
decouple slot selection from reconstruction
learn/estimate per-slot quality
select high-quality slots dynamically
avoid conflicting slot-count penalties
```

PICF status:

```text
Deployed in PICF-native form:
  PicfSlotQualityState
  aqr_slot_quality_head
  object/no-object/duplicate logits
  active/context/reserve weights
  entropy-corrected weak calibration loss
```

Mathematical form:

```math
q_j = \sigma(f_\theta[
  h_j,
  e_j,
  d_j,
  s_j,
  c_j,
  m_j
])
```

where:

```text
e_j: measurement evidence from sidecar/tracklet/point/tactile/proposals
d_j: duplicate risk from support and geometry overlap
s_j: row score
c_j: confidence
m_j: modality-confidence vector
```

The action-visible object weight is:

```math
w^{active}_j
=
q^{object}_j
(1-q^{empty}_j)
(1-q^{duplicate}_j)
```

implemented multiplicatively and floored only for eligible rows.

Current acceptance status:

```text
Code/audit pass. Behavior pending step 100/1000.
```

### 1.4 Object Binding / IsSameObject

Paper mechanism:

```text
same-object patch relation is decoded by a quadratic probe from ViT embeddings
and appears as a low-dimensional binding subspace.
```

PICF status:

```text
Deployed structurally:
  binding_signature_proj
  support-weighted signatures
  diagonal quadratic term
  low-rank pairwise term
  calibrated relative binding scores
  posterior binding-signature memory
```

Not completed:

```text
latest-run offline IsSameObject artifact probe.
```

This is not a missing architecture module. It is an acceptance test. The probe
must be run on latest overlays / exported features once step 100 or step 1000
artifacts exist.

### 1.5 SlotContrast / Temporal Identity

Local code:

```text
temp/paper_code_20260518/slotcontrast/slotcontrast/losses.py
```

Paper-code pattern:

```math
\operatorname{CE}
\left(
\frac{S_t S_{t+1}^\top}{\tau},
I
\right)
```

PICF status:

```text
Compatible but guarded. PICF has matched prediction / binding-consistency hooks
and posterior file memory, but current anchor capability probes keep these
losses zero unless explicitly testing predictive co-training.
```

Reason:

```text
Historical runs showed predictive/identity losses can become misleading before
posterior identity is stable. Enabling them during a pure anchor-position probe
would contaminate the question being tested.
```

### 1.6 SlotLifter / 3D Slot Lifting

Local code:

```text
temp/external_repos/SlotLifter/model/slot_attn.py
```

Mechanism:

```text
multi-view encoder
slot-axis attention over multi-view features
empty/background slot in the decoder
point/ray-to-slot decoding
```

PICF status:

```text
Partially mapped:
  typed point memory
  projective geometry
  sidecar/contact point support
  object candidate owner point priors
  empty/dustbin/background residuals
```

Not copied:

```text
full many-view NeRF/ray renderer.
```

Reason:

```text
CALVIN here is not a many-view NeRF reconstruction setting. The matching
PICF abstraction is point/ray evidence routed into posterior files, not an
image renderer.
```

### 1.7 SlotVLA / Robotic Object-Relation Slots

Paper mechanism:

```text
slot-attention-based object and relation representations for robotic action;
LIBERO+ supplies object masks/boxes and instance-level temporal tracking.
```

PICF status:

```text
Conceptually aligned:
  object files
  task-owner sidecars
  relation/ordinal diagnostics
  PI0.5 action prefix over corrected belief

Data gap:
  CALVIN sidecars are weak contact-motion proposals, not curated object labels.
```

Strict conclusion:

```text
PICF cannot claim SlotVLA-level object annotation support unless the sidecar /
tracklet coverage is completed and validated over the training subset.
```

### 1.8 OA-WAM / Object-Addressable World Action Model

Paper mechanism:

```text
robot slot + object slots
persistent address vector and time-varying content vector
address-only routing and address reset inside transformer layers
world/action prediction in one block-causal sequence
```

PICF status:

```text
Partially mapped:
  slot_address / slot_content
  gated address binding
  address-aware cache
  posterior files as persistent belief state
```

Not copied:

```text
address-only attention reset at every transformer layer.
```

Reason:

```text
That is a replacement of the action/world transformer internals. PICF sits
beside PI0.5 as a belief router and must not rewrite the PI0.5 generator in an
anchor capability probe.
```

### 1.9 OmniVTA / Visuo-Tactile World Modeling

Paper mechanism:

```text
self-supervised tactile encoder
two-stream visuo-tactile world model
contact-aware fusion policy
high-frequency tactile correction / reflex controller
```

PICF status:

```text
Compatible path:
  AnyTouch frozen tactile encoder
  tactile contact probability
  tactile-to-object owner attachment
  point/tactile spatial calibration
  contact evidence as owner measurement
```

Not copied:

```text
tactile VAE/world-model pretraining
60Hz closed-loop tactile controller
```

Reason:

```text
Current CALVIN/PICF training is offline policy learning. The immediate binding
failure is not a missing tactile rollout model; tactile_active_rate is zero in
the step-50 batch while sidecar/proposal evidence is active. Do not use
tactile-world pretraining to explain a non-contact target-selection failure.
```

## 2. Current Missing Items After Code Review

### 2.1 Behavior / artifact gates still pending

```text
step-100/1000 overlay verification
latest-run IsSameObject artifact probe
sidecar/tracklet coverage manifest over the intended training subset
posterior owner transport closure evidence
```

These are not missing modules. They are required evidence.

### 2.2 Real data gaps

```text
No curated object masks/boxes/tracks like SlotVLA/LIBERO+.
No full dataset sidecar/tracklet coverage yet.
No strong ordinal/fine-instance labels.
No tactile-world pretraining dataset equivalent to OmniVTA.
```

### 2.3 Modules intentionally not copied

```text
blind SAM proposal source
full VQ visual prototype codebook inside posterior state
full image/object reconstruction decoder as production action loss
many-view NeRF/ray renderer
OA-WAM transformer-layer address reset inside PI0.5
tactile VAE/reflex controller
```

Reason:

```text
They are either incompatible with the current belief-router/action contract,
require supervision/data we do not have, or target a different training
objective. Adding them now would be module dumping, not a coherent PICF fix.
```

## 3. What Should Be Done Next

### 3.1 Do now

```text
Keep the current A7 slot-quality probe running to step 100.
Generate GIFs after step 100.
Run latest-artifact overlay/IsSameObject diagnostics.
Inspect whether active posterior owner moves to the sidecar mask, not just
whether graph candidates exist.
```

### 3.2 If step 100 still fails

The likely failure is:

```text
accepted object candidate reaches transient graph rows but does not close
strongly enough into persistent posterior owner files.
```

The next code change should therefore target posterior closure, not add new
external slot modules:

```text
increase posterior owner transport reliability for accepted high-quality
candidate rows;
ensure active posterior file selection uses transported owner evidence;
keep sidecar as measurement with covariance, not hard truth;
verify by active posterior distance-to-mask and owner_transport_confidence.
```

### 3.3 Do not do now

```text
Do not re-enable blind SAM.
Do not copy full MetaSlot VQ into posterior state.
Do not enable slot-JEPA/support-pred/binding-consistency during this anchor
capability probe.
Do not treat task query visualizations as physical object success.
```

## 4. Strict Conclusion

Compared to current 2025-2026 slot/object-binding methods, PICF has now
absorbed the belief-filter-compatible core mechanisms:

```text
slot-axis/evidence competition
background/no-object reserve
adaptive slot quality
duplicate suppression
pairwise same-object binding subspace
object-address/content split
tactile-to-object owner attachment
typed sidecar/tracklet evidence
posterior file competition
```

The remaining issue is not an obvious missing paper module. The remaining
issue is behavior closure:

```text
does accepted weak object evidence actually update the persistent posterior
file that action will use?
```

That must be answered by the current run's step-100/1000 overlays, metrics, and
latest-artifact probes.

## 2026-05-19 Step-200 Probe Update

The A7 `picf_a7_slot_quality_selector_anchor1000_20260519` run changed the
interpretation of the remaining failure.  The step-200 JSON shows:

```text
prompt:
  slide the door to the left

sidecar proposal center:
  pixel approximately (142, 71)

active graph owner:
  graph index 7, role 1, pixel approximately (143, 73)

active posterior owner:
  posterior index 7, role 1, pixel approximately (141, 73)
  owner_transport_mass approximately 0.817
  owner_transport_confidence approximately 0.160
```

Thus the accepted object candidate is now reaching both the transient graph
owner and the active posterior owner file on this frame.  The older aggregate
metric `posterior_owner_transport_dist_to_standard_mean` is misleading because
it averages over inactive posterior files as well as the single active owner.
The correct acceptance metric is active-owner distance/confidence:

```text
posterior_owner_transport_active_dist_mean
posterior_owner_transport_active_confidence_mean
posterior_owner_transport_active_count
```

These diagnostics are now added to the runtime debug contract.  This is not a
new loss and not a model change; it prevents false negatives when only one
posterior file is supposed to own the current task/contact object.

Current strict reading:

```text
graph-side sidecar binding:
  working on inspected frames

active posterior owner closure:
  working on step-200 inspected frame

raw reserve/support overlap:
  still high, but expected for inactive reserve/context rows and not by itself
  a failure when active overlap is low

remaining proof:
  step-1000 trend, active-owner metrics on multiple tasks, GIF inspection, and
  latest-artifact IsSameObject probe
```
