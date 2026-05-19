# PICF-AQR-OWM Slot/Binding Theory Gap Analysis

Status: TEMP theory/dataflow audit
Date: 2026-05-15
Scope: architecture split, mathematical model, code follow-through, literature comparison, current failure localization.

## 0. Executive answer

The current PICF-AQR-OWM design does not have a fatal theoretical contradiction. It is coherent as a posterior-centered POMDP belief router:

```math
b_t = U_\theta\left(P_\theta(b_{t-1}, a_{t-1}), E_\theta(o_t, \ell)\right)
```

where typed evidence is routed by AQR, persistent object files are corrected by a posterior update, and action is decoded from the corrected belief.

However, it is not yet equivalent to a mature object-centric slot system in the SlotAttention/SAVi/SlotContrast/OA-WAM sense. The remaining theoretical gaps are:

1. object-file lifecycle calibration: recycle/dustbin is learned and currently underconstrained;
2. binding subspace verification: centered support signatures are plausible, but not yet proven by an offline IsSameObject probe on our data;
3. temporal identity supervision: slot-JEPA/support/binding losses are guarded/off, so recurrent state alone may not enforce long-horizon identity;
4. objectness/background modeling: fixed AQR queries plus active/dustbin filtering is a practical object-file bank, but not a generative object/background decomposition;
5. full object-addressability: slot_address exists and influences binding/cache, but the action transformer is not forced to route only through address keys as in OA-WAM.

These are not reasons to discard the architecture. They define the exact boundary between "strong maintained baseline" and "scientifically closed object-centric world model".

## 1. Architecture split and code follow-through

### 1.1 Evidence/token field

Role:

```text
E_t = {PG image/text, V-JEPA static+wrist temporal, point/depth, tactile, tracklet/proposal if present, previous posterior, cache}
```

Code facts:

- `PicfObservation` exposes optional `tracklet_*` and `proposal_*` fields in `src/openpi/picf/contracts.py`.
- `PicfTemporalVisualSupportState`, `PicfTrackletSupportState`, `PicfPseudoProposalState`, and cache states exist in `src/openpi/picf/core/contracts.py`.
- Current CALVIN training still has `owm_tracklet_tokens=0` and `owm_proposal_tokens=0` unless the data loader supplies them.

Theory:

The evidence layer is broad enough for the desired architecture, but not every typed branch is active in the current dataset path. Missing tracklets/proposals do not break correctness; they reduce available same-object evidence.

### 1.2 AQR routing and support competition

Role:

```math
p_{j,i}^{(m)} = \operatorname{softmax}_i(q_j^T k_i^{(m)} + b_{j,i}^{(m)})
```

with same-role competition:

```math
p'_{j,i} = (1 - \lambda)p_{j,i} + \lambda\,\mathrm{ExclusiveNormalize}_{same-role}(p_{j,i})
```

Code facts:

- `_aqr_same_role_support_competition` is a measurement-routing residual, not a loss.
- `_aqr_active_slot_mask` selects a sparse active subset; redundant same-role anchors become reserve/dustbin candidates.
- Current step 100 centered run has healthy active overlap: `aqr_active_same_role_support_overlap_max=0.0390`.

Theory:

This layer is no longer the main failure. It implements the right local competition idea. The fixed bank can show many visible anchors in overlay, but only a smaller active subset should bind objects.

Risk:

If users visually inspect all graph anchors without distinguishing active/inactive, they may falsely conclude all anchors are intended object slots. The acceptance metric must prioritize active overlap and posterior slots, not raw visible anchor count.

### 1.3 Support/binding signature

Role:

```math
s_j^{(m)} = \mathrm{normalize}\left(\sum_i p_{j,i}^{(m)}\,\phi_m(z_i)\right)
```

The 2025 object-binding ViT result suggests same-object information is a pairwise/projected subspace, not necessarily raw token cosine. Our current repair centers projected token keys:

```math
\phi(z_i) = \mathrm{normalize}(Wz_i - \frac{1}{N}\sum_k Wz_k)
```

then pools with support weights.

Code facts:

- `_binding_keys(tokens, center=True)` subtracts the projected token-set mean before normalization.
- `_support_binding_signature` uses centered keys.
- Local audit `picf_binding_signature_common_mode_audit.py` shows synthetic common-mode mean cosine drops from `0.9809` to `-0.3180`.
- Runtime centered run step100 drops `aqr_same_role_obs_binding_signature_overlap_mean` to `0.5183`, from Phase B `0.9846`.

Theory:

This is mathematically well aligned with the IsSameObject paper: object binding should be decoded from a low-dimensional pairwise subspace. Centering removes global scene/modality common mode before support pooling.

Remaining gap:

We have not yet trained or evaluated a true offline IsSameObject probe on CALVIN features. Therefore we know centering reduces a measured overlap pathology, but we have not proven it recovers the best same-object subspace.

### 1.4 Posterior binding and dustbin

Role:

```math
B = \mathrm{SinkhornDustbin}(\ell)
```

where:

```math
\ell_{j,i} = h_j^T o_i - d_{geom}(j,i) + \lambda_s O_{j,i}^{support} + \lambda_b S_{j,i}^{bind} + \lambda_a A_{j,i}^{address} + biases
```

The posterior then splits observation mass into slot support and dustbin residual.

Code facts:

- `_sinkhorn_dustbin` appends one dustbin row with zero logits and equal row target.
- `_posterior_owner_active_binding_bias` masks inactive observations with a large negative bias so they cannot update persistent slots.
- `dustbin_raw` is multiplied by `owner_active` reliability before residual redistribution.
- `recycle` is produced by a learned `recycle_head` from prior state, support mass, prior variance, residual summary, and prior alpha.

Current observation:

- Phase B step100: `posterior_dustbin_mass_raw=0.0191`.
- Centered run step100: `posterior_dustbin_mass_raw=0.2325`.
- Historical hard-gate failures were much larger (`~1.88` or `~7-9`).

Theory:

The centered binding signature removes a false common-mode stability signal. That makes posterior assignment more honest but less confident, causing more mass to go to dustbin/recycle. This is not the original support-collapse failure; it is a lifecycle calibration problem.

Theoretical vulnerability:

Dustbin is a single row with a fixed prior score, and recycle is a learned reset probability without an explicit Bayesian prior on object survival. Mature tracking/filtering systems usually separate:

```text
assignment likelihood
object existence/survival
birth/death/recycle process
background/dustbin process
```

Our implementation combines these into Sinkhorn dustbin + learned recycle. That is workable but underconstrained, explaining why dustbin/recycle can drift when the binding feature distribution changes.

### 1.5 Address/content and cache

Role:

```math
S_{t,j} = (a_{t,j}, c_{t,j}, \mu_{t,j}, \Sigma_{t,j}, \alpha_{t,j})
```

with address as persistent identity key and content as time-varying state.

Code facts:

- `slot_address` is stored in posterior and cache.
- binding uses address score gated by alpha, recycle, and innovation risk.
- cache read has residual scaling and skips immediate previous posterior duplicate.

Comparison to OA-WAM:

OA-WAM enforces address/content separation more strongly: it routes cross-slot attention through address-only keys and resets the address slice at each transformer layer. Our design uses address as an auxiliary binding/cache signal; it does not structurally force every action/world attention layer to be address-keyed.

Theory:

This is not wrong, but it is weaker. The current design is a posterior belief router with address assistance, not a fully address-routed transformer world model.

### 1.6 Losses

Role:

The production path mostly relies on action + alignment/graph losses. OWM predictive losses remain guarded:

```text
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
lambda_aqr_denoising = 0
```

Code facts:

- slot-JEPA/support use detached future targets and soft matched prediction, not raw index alignment.
- binding consistency uses matched token alignment but remains off by default.

Comparison to SlotContrast / SlotMatch:

SlotContrast argues recurrent processing alone often lacks long-term slot stability unless temporal consistency is explicitly enforced. SlotMatch argues simple cosine matching can be enough for distillation and extra losses may be redundant.

Theory:

Our guarded default is correct for stability, but it means we should not claim the world-model objectives are scientifically solved. The current design can train a strong action-conditioned belief router; it does not yet provide strong temporal object identity supervision.

## 2. Literature comparison

### 2.1 Object Binding in pretrained ViTs, NeurIPS 2025

Relevant finding:

- IsSameObject is decodable with a quadratic similarity probe.
- Binding lives in a low-dimensional subspace on top of object features.
- The signal guides attention.

Implication for PICF:

- raw hidden cosine is insufficient;
- projected pairwise binding signatures are justified;
- centering common-mode is justified;
- an offline IsSameObject audit is still missing.

### 2.2 SlotContrast, CVPR 2025

Relevant finding:

- recurrent object-centric video models can lack long-term stability;
- explicit temporal contrastive slot-level loss improves temporal consistency and object discovery.

Implication for PICF:

- our high recycle/identity switch is not surprising;
- action loss alone is not guaranteed to preserve identity;
- once posterior lifecycle is calibrated, a low-weight, matched temporal consistency loss may be useful, but it should not be opened before recycle/dustbin is stable.

### 2.3 SlotMatch, 2025

Relevant finding:

- slot distillation can be simple: align teacher/student slots by cosine matching, and additional losses may be redundant.

Implication for PICF:

- avoid a large pile of auxiliary losses;
- if predictive losses are enabled, use matched detached teacher targets, not many unrelated heuristics.

### 2.4 OA-WAM, 2026

Relevant finding:

- object-addressable world-action modeling separates persistent address and time-varying content;
- address-only routing improves object reference robustness.

Implication for PICF:

- current address/content state is directionally right;
- full OA-WAM-level addressability would require deeper transformer routing changes, not just cache/binding scores;
- not necessary for the current fix, but important for any claim of fully object-addressable world model.

## 3. Does PICF have a theoretical loophole?

### Not a fatal loophole

The overall decomposition is coherent:

```math
Typed evidence \rightarrow AQR routing \rightarrow posterior correction \rightarrow action
```

This matches a POMDP belief filter and avoids using cache/future targets as truth.

### Real loophole 1: lifecycle is not independently calibrated

Current posterior update uses a single Sinkhorn dustbin plus learned recycle. It lacks explicit survival/birth/death priors:

```math
p(z_i = dustbin),\quad p(slot_j survives),\quad p(slot_j resets),\quad p(new object birth)
```

Therefore changing binding signatures changes the dustbin/recycle distribution. That is exactly what the current centered run shows.

### Real loophole 2: object binding is assumed, not independently audited

We now compute a more plausible projected binding signature, but do not yet prove that it separates same-object/different-object pairs on our data. The literature says this is measurable with IsSameObject probes. We need that audit if we want scientific closure.

### Real loophole 3: object slots are not generative masks

Advanced SlotAttention-like methods often use reconstruction/mask/object decoders to force slots to explain the scene. PICF instead uses typed support routing and action/alignment supervision. This is appropriate for robotics speed, but it cannot guarantee complete object segmentation.

### Real loophole 4: fixed query bank is not the same as discovered object count

The system has many queries, but only a sparse active subset should be interpreted as object files. Overlaying all graph anchors can look like too many anchors. This is a UX/diagnostic issue and a modeling issue: active/dustbin must remain part of the semantics.

## 4. What should be fixed now vs later

### Fix now before 30k if centered run finishes with high recycle/dustbin

1. posterior lifecycle calibration:
   - add explicit debug decomposition for dustbin source: inactive-owner mass, low-binding mass, role mismatch, occupancy mismatch;
   - make recycle depend on calibrated support confidence and top1 margin, not only learned residual summary;
   - consider a conservative survival prior so stable slots do not reset unless evidence is strongly inconsistent.

2. IsSameObject audit:
   - no training dependency;
   - use tracklet if present, point-neighborhood, contact neighborhood, high-confidence owner peaks, and posterior continuity as weak labels;
   - evaluate raw hidden, projected binding, centered projected binding, and modality-specific features.

### Do not add now as a blind fix

1. more support diversity penalties:
   active support is already healthy;
   adding more penalties targets the solved layer and may hurt posterior confidence.

2. stronger slot-JEPA/support-pred/binding losses:
   literature supports temporal consistency, but current recycle/dustbin must be stable first.

3. full OA-WAM address-only transformer rewrite:
   theoretically attractive but too large for this failure localization.

## 5. Current verdict

The architecture is strong and mathematically coherent, but it is not theoretically complete in the mature slot/object-centric sense.

Current evidence says:

```text
AQR support collapse: mostly fixed for active anchors.
Binding signature common-mode: strongly improved by centering.
Posterior lifecycle: still unresolved and now the main issue.
```

Therefore the next decision should not be "add more slot modules". It should be:

```text
finish centered diagnostic -> inspect step 150/200 -> if dustbin/recycle remains high, fix posterior lifecycle calibration before 30k.
```

## 6. Sources checked

- Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?, arXiv:2510.24709, NeurIPS 2025 Spotlight.
- Temporally Consistent Object-Centric Learning by Contrasting Slots, CVPR 2025.
- SlotMatch: Distilling Object-Centric Representations for Unsupervised Video Segmentation, arXiv:2508.03411.
- OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation, arXiv:2605.06481.
