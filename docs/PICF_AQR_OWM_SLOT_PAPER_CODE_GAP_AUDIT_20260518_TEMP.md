# PICF-AQR-OWM Slot Paper-Code Gap Audit

Date: 2026-05-18

Status: structural audit, not a training acceptance report.

This document compares the current PICF-AQR-OWM anchor/slot design against
the closest object-centric slot code that was pulled locally under
`temp/paper_code_20260518/`.

## Local Paper Code Reviewed

- `temp/paper_code_20260518/object-centric-learning-framework`
  - `ocl/perceptual_grouping.py`
  - `ocl/models/savi.py`
- `temp/paper_code_20260518/slot-attention-video`
  - `savi/modules/video.py`
- `temp/paper_code_20260518/AdaSlot`
  - `ocl/perceptual_grouping.py`
  - `ocl/conditioning.py`
  - `ocl/decoding.py`
- `temp/paper_code_20260518/MetaSlot`
  - `object_centric_bench/model/metaslot.py`
  - `object_centric_bench/model/dinosaur.py`
- `temp/paper_code_20260518/DINO`
  - `models/dino/dino.py`
  - `models/dino/deformable_transformer.py`
  - `models/dino/dn_components.py`
- `temp/paper_code_20260518/Deformable-DETR`
  - `models/deformable_transformer.py`
  - `models/ops/modules/ms_deform_attn.py`

## PICF Code Reviewed

- `src/openpi/picf/core/pipeline.py`
  - `_build_aqr_anchor_graph`
  - `_aqr_same_role_support_competition`
  - `_aqr_active_slot_mask`
  - `_proposal_anchor_seed_transport`
  - `_binding_logits`
  - `_posterior_update`
- `src/openpi/picf/core/training.py`
  - `_mapg_graph_loss`
  - `_vcap_auxiliary_loss`
  - `_matched_prediction_loss`
  - `_binding_consistency_loss`
  - guarded OWM auxiliary loss composition
- `src/openpi/picf/core/config.py`
  - active slot, VCAP, proposal seed, cache, binding, local refinement knobs

## Executive Judgment

PICF-AQR-OWM is not simply a broken implementation of Slot Attention. It is a
different architecture: a posterior-centered belief router for typed evidence.
That design is coherent for robotics control.

However, if the requirement is "slot-like object binding should emerge
reliably", then the current implementation is incomplete relative to modern
slot/object-centric systems. The missing part is not another small regularizer;
it is the core object-explanation objective used by mature slot methods.

The current implementation has many downstream guards:

- same-role support competition
- active/context/reserve filtering
- posterior file competition
- proposal seed transport
- local refinement
- binding signature terms
- VCAP scaffolding

But it still lacks a trained, primary, per-slot object explanation loop:

```text
slot_j produces mask/assignment over dense evidence
slot_j reconstructs or predicts its assigned evidence
all evidence is either explained by an object slot or by background/no-object
duplicates and empty slots are explicitly trained, not only filtered later
```

This is the main structural gap.

## Comparison: Slot Attention / OCL

In `object-centric-learning-framework/ocl/perceptual_grouping.py`, Slot
Attention performs iterative competition over slots:

```text
dots = q_slot @ k_feature
attn = softmax(dots, dim=slot_axis)
updates = attn @ v_feature
slot = GRU(updates, previous_slot)
```

Important detail: competition happens on the slot axis for every feature. This
means each feature is forced to choose which slot explains it. Mature OCL code
then typically decodes object reconstructions/masks from slots.

PICF also has competition, but it is used as a routing prior inside the AQR
belief graph:

```text
typed memory -> AQR readers -> priors -> posterior update -> action prefix
```

This is useful, but it does not force each visual/point/proposal token to be
explained by exactly one object slot under a reconstruction or prediction
loss. Therefore duplicate anchors can be suppressed or demoted, but object
decomposition is not guaranteed to emerge.

## Comparison: DINOSAUR / MetaSlot

In `MetaSlot/object_centric_bench/model/dinosaur.py`, the model:

1. freezes a dense feature backbone,
2. aggregates dense features into slots,
3. decodes slots back into feature reconstructions and masks,
4. mixes reconstructions with slot masks.

This creates an object-explanation pressure:

```math
L_{explain}
=
\left\| z_i - \sum_j m_{j,i} \hat z_{j,i} \right\|^2
```

with masks satisfying an across-slot competition:

```math
\sum_j m_{j,i} = 1
```

MetaSlot additionally introduces codebook/prototype logic and duplicate-slot
suppression: duplicate quantized slots are masked out in inference. This is a
real object-count / duplicate mechanism, not only a downstream action filter.

PICF has VCAP and active slot filtering, but:

- VCAP is disabled by default.
- VCAP auxiliary weights are zero by default.
- active/context/reserve is a downstream selection policy.
- no dense feature reconstruction objective makes the selected files explain
  the scene.

Therefore, relative to DINOSAUR/MetaSlot, PICF's current anchor subsystem is a
belief-router scaffold, not a complete object-centric decomposition model.

## Comparison: SAVi / Video Slot Models

In SAVi-style code, each time step has:

```text
feature_t
previous slots / conditioning
corrected slots = grouping(feature_t, conditioning)
decoded masks/reconstruction
next conditioning = transition(corrected slots)
```

The same slot state is repeatedly corrected and predicted over time.

PICF has a posterior transition and cache, which is conceptually related, but
the temporal slot state is not strongly constrained by object reconstruction
or predictive reconstruction. Slot-JEPA/support-prediction hooks exist, but
they are guarded and normally zero-weight. That is a rational training guard,
but it also means the slot object model is not yet a mature SAVi-style
temporal object model.

## Comparison: AdaSlot

AdaSlot addresses a problem PICF has repeatedly seen in experiments: fixed
capacity causes duplicate or unnecessary slots. AdaSlot uses dynamic slot
selection / empty-slot masking so the model can represent content-dependent
object count.

PICF has:

- active/context/reserve file filtering,
- VCAP autoregressive proposal scaffolding,
- no-object / dustbin behavior in posterior binding.

These are aligned in spirit, but they are not yet equivalent to a trained
dynamic slot-number object autoencoder:

- active filtering is mostly a routing/action-prefix decision;
- VCAP losses are disabled by default;
- empty/no-object behavior is not trained against a dense object-explanation
  target.

This is why "too many gray anchors" and "anchors competing for the same useful
region" can still occur: capacity management is downstream and partially
guarded, not the central object discovery training task.

## Comparison: DINO / Deformable-DETR

DINO / Deformable-DETR show a different but relevant lesson:

- object queries are not only learned vectors;
- they are tied to reference points/boxes;
- denoising queries are trained with known noisy targets;
- Hungarian/set losses make predicted objects compete as a set.

PICF already borrowed parts of this idea:

- proposal seed transport,
- local refinement,
- denoising-like losses,
- active slot filtering.

But the current proposal path is still not a full reference-query detection
mechanism. In recent trials, proposal support could be high while
`proposal_anchor_seed_point_max` decayed and `loss_anchor_pv` rose. That
indicates the proposal is read, but not converted into a stable physical
measurement/file.

The structural difference is:

```text
DINO-style:
  proposal/reference geometry is the query's spatial contract.

Current PICF:
  proposal/reference geometry is a bias or seed mixed into a broader AQR router.
```

For object binding, the DINO-style formulation is stronger.

## Comparison: Object-Binding ViT / IsSameObject

The 2025 object-binding ViT paper argues that self-supervised ViTs can encode
an IsSameObject relation in a low-dimensional, often quadratic/similarity-like
subspace. This supports PICF's binding-signature idea.

PICF has added:

- support signatures,
- binding signature projection,
- calibrated pairwise binding scores,
- gated address terms.

This is directionally correct. But the missing piece is calibration:

```text
We use the subspace as a binding score.
We have not fully trained/audited it as an IsSameObject predicate over latest
artifacts using weak same/different labels from tracklets, contact, point
neighborhoods, proposal masks, and posterior continuity.
```

Therefore the current binding subspace is plausible but not proven. It is not
enough by itself to make object slots emerge.

## Mathematical Diagnosis

Let dense evidence tokens be:

```math
Z = \{z_i\}_{i=1}^{N}
```

and slot files/anchors be:

```math
S = \{s_j\}_{j=1}^{K}
```

Mature slot models optimize something close to:

```math
\min_{\theta}
\sum_i
d\left(
z_i,\,
\sum_j m_{j,i} f_\theta(s_j, i)
\right)
+
\lambda\,R(M)
```

where:

```math
m_{j,i} \ge 0,\quad \sum_j m_{j,i}=1
```

This forces every token to be explained by a slot or by a background/empty
slot. Duplicate slots are costly because they do not explain more evidence.

PICF currently optimizes a control-oriented objective:

```math
L
=
L_{action}
+
L_{alignment}
+
L_{routing}
+
L_{weak\ anchor}
+
L_{guarded\ OWM}
```

AQR forms priors:

```math
p_{j,i}^{(m)}
=
\mathrm{softmax}_i(q_j^\top k_i^{(m)} + b_{j,i}^{(m)})
```

and posterior binding updates belief:

```math
b_t = U(P(b_{t-1}, a_{t-1}),\, \mathrm{AQR}(o_t))
```

This is coherent for belief-state control. But unless the weak anchor/routing
terms are strong enough to impose object explanation, nothing prevents all
useful anchors from chasing the same action-relevant region while action loss
still decreases.

That exactly matches the observed failure mode:

```text
action loss can improve,
proposal support can be high,
but active/raw same-role overlap and anchor_pv can deteriorate.
```

## Is PICF a "Truncated" or "Immature" Slot Model?

For robotics belief routing:

```text
No. PICF is a coherent posterior-centered typed-memory router.
```

For object-centric slot decomposition:

```text
Yes, it is incomplete relative to modern slot/OCL systems.
```

The missing modules are not cosmetic:

1. Per-slot object explanation decoder over dense frozen features.
2. Explicit background/empty/no-object slot trained with explanation loss.
3. Dynamic slot count / duplicate suppression trained as part of object
   discovery, not only downstream filtering.
4. Temporal slot transition/correction trained with object-level prediction.
5. Calibrated IsSameObject probe/loss over weak same/different pairs.
6. Proposal/contact/tracklet conditioning used as reference-slot
   initialization, not only as attention bias.

## What This Means For The Current Failures

### Anchor not binding to the red block

Likely cause:

```text
The red-block proposal/motion/contact evidence is present, but it is not a
primary object-explanation target. The router can use or ignore it depending on
short-term action/alignment gradients.
```

### `loss_anchor_pv` rising

Likely cause:

```text
Physical anchors are not forced to explain stable object evidence. They can
move toward action-relevant or high-response regions while violating the weak
PV geometry target.
```

### same-role raw overlap staying high

Likely cause:

```text
Same-role competition and active filtering are corrective terms, not a
complete object partition objective. They can demote duplicates downstream but
cannot guarantee distinct object files.
```

### gray/reserve anchors consuming tokens

Likely cause:

```text
Reserve/background state is not trained with a full object/background
explanation objective. It is managed by gates and filters.
```

## Recommended Architectural Correction

The clean next step is not to add another small overlap penalty. The clean next
step is to add an object-explanation head that is subordinate to the posterior
belief router:

```text
typed dense evidence
  -> object explanation slots
  -> masks / assignment over dense evidence
  -> feature/geometry/contact reconstruction or prediction
  -> AQR/posterior uses explained object slots as measurements
  -> PI0.5 action remains over posterior belief
```

### Object Explanation Head

For every physical file/anchor:

```math
m_{j,i}^{visual}, m_{j,i}^{point}, m_{j,i}^{proposal}, m_{j,i}^{tracklet}
```

Train:

```math
L_{explain}
=
\lambda_v \sum_i \|z_i^v - \sum_j m_{j,i}^v \hat z_{j,i}^v\|^2
+
\lambda_p \sum_i \|x_i^p - \sum_j m_{j,i}^p \hat x_{j,i}^p\|_{\Sigma}^{2}
+
\lambda_c L_{contact/object}
+
\lambda_{bg} L_{background}
+
\lambda_{dup} L_{duplicate}
```

The masks must compete over object slots plus a background/no-object slot:

```math
\sum_{j \in objects} m_{j,i} + m_{bg,i} = 1
```

### Proposal / Contact / Tracklet Use

Proposal/contact/tracklet sidecars should not be hard truth. They should:

1. initialize or bias object-slot reference geometry;
2. provide weak same-object pairs;
3. provide weak objectness masks;
4. decay after slot masks become stable.

This is consistent with DINO-style denoising/reference queries and
object-binding ViT same-object subspaces.

### Training Sequence

Recommended:

1. Object-explanation prewarm:
   freeze action, freeze heavy backbones, train explanation head + anchor
   connectors + binding subspace.
2. Belief-router warmup:
   enable posterior update and AQR routing, keep action low or frozen.
3. Co-train:
   unfreeze PI0.5/PaliGemma connectors with low auxiliary budgets.
4. Long run acceptance:
   evaluate action, anchor overlay, object masks, `loss_anchor_pv`,
   overlap, recycle, and IsSameObject probe.

## What Not To Do

- Do not treat SAM boxes/masks as hard object truth.
- Do not add more isolated overlap penalties without object explanation.
- Do not rely on action loss to discover objects by itself.
- Do not claim current anchors are equivalent to Slot Attention/DINOSAUR/SAVi.
- Do not remove posterior authority; object slots should feed measurements,
  not replace the posterior belief state.

## Final Audit Conclusion

The module is not merely poorly tuned. The object-binding part is structurally
weaker than modern slot/OCL systems because it lacks the central object
explanation/reconstruction/prediction loop.

PICF's belief-router architecture is defensible and useful. But if we want
robust object binding, we need to add a true object-explanation slot objective
and make proposals/contact/tracklets condition object slots rather than merely
nudging AQR attention.

This should be treated as a vNext architectural repair, not a minor loss-weight
adjustment.
