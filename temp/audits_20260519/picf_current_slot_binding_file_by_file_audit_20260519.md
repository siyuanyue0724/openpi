# PICF Current Slot / Object / Tactile Binding File-By-File Audit

Date: 2026-05-19

Status: strict architecture/dataflow audit, not behavior acceptance.

Verdict:

```text
Has the 2025+ tactile/object-slot question been completely solved?
  NO.

Has the current PICF implementation absorbed the main mathematically relevant
principles from recent object-centric work?
  PARTIALLY YES.

Does the current code still miss material pieces compared with mature slot/OCL
systems?
  YES.
```

This file audits the local PICF implementation file by file.  The companion
file
`temp/audits_20260519/advanced_slot_tactile_binding_methods_gap_audit_20260519.md`
audits external methods and gives the gap table.

## 0. Contract Being Checked

PICF is not a pure Slot Attention image segmenter.  It is a belief-state router:

```math
b_t(s)
\propto
p(o_t\mid s_t)
\int p(s_t\mid s_{t-1},a_{t-1})b_{t-1}(s_{t-1})ds_{t-1}.
```

For each persistent object file:

```math
S_{t,j}=(a_{t,j},c_{t,j},\mu_{t,j},\Sigma_{t,j},\alpha_{t,j},r_j,\sigma_{t,j})
```

where:

```text
a: persistent address / identity descriptor
c: time-varying content
mu, Sigma: geometry belief
alpha: existence / activity
r: role
sigma: support signature over visual / temporal / point / tactile / proposal
```

Therefore the correct comparison target is:

```text
typed evidence -> slot/object responsibility -> posterior file update -> action
```

not:

```text
segmentation mask -> hard object label -> action
```

## 1. `src/openpi/picf/core/config.py`

Relevant deployed blocks:

```text
object_candidate_owner_transport_enabled
object_candidate_owner_roles
object_candidate_owner_min_share
object_candidate_owner_point_mix

object_explanation_enabled
object_explanation_background_prior
object_explanation_min_slot_quality
object_explanation_duplicate_margin

bind_support_signature_weight
bind_embedding_signature_weight
bind_quadratic_signature_weight
bind_low_rank_signature_weight
binding_signature_score_calibration_enabled
binding_signature_centering_enabled

posterior_binding_signature_memory_enabled
posterior_binding_signature_update_rate
posterior_binding_signature_dispersion_gate_enabled

posterior_file_competition_enabled
posterior_birth_competition_enabled

posterior_owner_transport_enabled
posterior_owner_transport_precision_gain
posterior_owner_transport_activates_file

legacy_local_refinement_opt_in = False
local_refinement_enabled = False
```

What this means mathematically:

```math
\ell_{bind}
=
\ell_{hidden}
\ell_{geom}
\lambda_s \ell_{support}
\lambda_q \ell_{same-object}
\lambda_a \ell_{address}
```

and current owner transport uses a measurement update, not weak interpolation:

```math
\Lambda^+
=
(S^{std})^{-1}
+
\kappa c^{owner}(S^{owner})^{-1},
```

```math
x^+
=
(\Lambda^+)^{-1}
\left((S^{std})^{-1}x^{std}
+
\kappa c^{owner}(S^{owner})^{-1}x^{owner}\right).
```

Good:

```text
1. The design now has explicit object explanation, duplicate demotion, posterior
   owner transport, and posterior binding-signature memory.
2. Blind SAM is not a production default.
3. Local refinement is archived/legacy-off instead of silently adding another
   unstable reread pressure.
4. Tactile can be attached to object owner instead of being owned by a separate
   effector file.
```

Deficits:

```text
1. There is no MetaSlot-style global VQ/prototype codebook for adaptive slot
   initialization or duplicate aggregation.
2. There is no QASA-style explicitly learned slot-quality selector decoupled
   from reconstruction/action losses.
3. The object explanation layer exists, but most of its losses are guarded or
   default-zero; it is mainly a measurement/diagnostic scaffold unless a run
   explicitly enables it.
4. Owner transport strength depends on sidecar/candidate quality and
   confidence.  Step-100 precision-fusion diagnostics still showed several-pixel
   posterior drift, so this is not behavior-proven.
```

## 2. `src/openpi/picf/core/contracts.py`

Relevant states:

```text
PicfAnchorPriorGraphState:
  visual_priors
  point_priors
  tactile_priors
  posterior_priors
  tracklet_priors
  proposal_priors
  object_candidate_assignment
  object_candidate_owner_assignment
  object_candidate_owner_point_priors
  object_explanation_quality
  binding_signature

PicfActiveProposalState:
  active_prob
  role_logits
  coverage_score
  duplicate_score
  unexplained_evidence

PicfObjectExplanationState:
  object_mask_visual / background_mask_visual
  object_mask_temporal / background_mask_temporal
  object_mask_point / background_mask_point
  object_mask_tactile / background_mask_tactile
  object_mask_tracklet / background_mask_tracklet
  object_mask_proposal / background_mask_proposal
  anchor_quality
  anchor_duplicate_overlap
  contact_explanation_score

PicfPosteriorAnchorState:
  binding_signature
  file_competition_active
  owner_transport_mass
  owner_transport_confidence
  owner_transport_applied_fraction
  owner_transport_dist_to_standard
```

Good:

```text
1. The contract now has enough fields to express object-vs-background
   responsibility, active/reserve files, same-object signatures, and posterior
   owner write-through.
2. Tracklet and proposal typed memory are first-class optional states.
3. Tactile is represented as typed evidence and contact probability rather than
   raw late-concat only.
```

Deficits:

```text
1. There is no explicit per-object mask decoder state equivalent to mature OCL
   systems that reconstruct/explain every dense token through object masks plus
   background.
2. `PicfActiveProposalState` is a proposal initializer, not a full learned
   adaptive-slot allocator with a codebook or end-to-end no-object decoder.
3. Tracklet states exist, but current CALVIN coverage depends on sidecar
   generation and manifest quality.  `owm_tracklet_tokens=0` is still possible.
4. The contract supports offline IsSameObject probing, but latest-run artifact
   probe acceptance is not yet a completed required gate.
```

## 3. `src/openpi/picf/core/pipeline.py`

### 3.1 Typed AQR evidence readers

Current flow:

```text
visual / temporal V-JEPA / PG image / point / tactile / posterior / cache /
tracklet / proposal
  -> modality readers
  -> same-role support competition
  -> anchor scores/confidence
```

Key invariant:

```math
p_{j,i}^{m}
=
\operatorname{softmax}_i(q_j^\top k_i+b_{j,i}^{m})
```

followed by same-role competition, so evidence is not meant to be cloned across
all same-role anchors.

Good:

```text
1. Dense memory is retained; object slots do not delete V-JEPA/point/tactile
   tokens.
2. Tactile bias can be scoped to object owner roles.
3. Sidecar object candidates can bias proposal/point support.
```

Deficits:

```text
1. This is still query-driven routing.  It is not a pure Slot Attention
   iteration where every evidence token competes globally over all slots in an
   object reconstruction loop.
2. The current owner/mask path depends on sidecar candidate generation.  If the
   sidecar is missing or noisy, slot binding falls back to weaker support
   signatures and geometry.
3. Role layout can still be a source of semantic mismatch if effector/contact
   rows are allowed to compete for object ownership.  Recent object-only probes
   deliberately disable role-0 effector anchors for this reason.
```

### 3.2 Binding signature subspace

Current flow:

```text
support weights + typed memory tokens
  -> binding_signature_proj
  -> centered projected keys
  -> support-weighted binding signature
  -> linear + diagonal quadratic + low-rank pairwise scores
  -> double-center / z-score calibration
```

Mathematically:

```math
z_i = \operatorname{norm}(W f_i - \bar{Wf})
```

```math
s_j = \operatorname{norm}\left(\sum_i p_{j,i}z_i\right)
```

```math
\ell_{j,k}^{same}
=
s_j^\top s_k
+
s_j^\top D s_k
+
(U s_j)^\top(V s_k).
```

Good:

```text
This directly matches the object-binding-paper principle that same-object
information is pairwise/quadratic and should be calibrated, not treated as raw
cosine over common-mode features.
```

Deficits:

```text
1. The subspace is trained only indirectly unless an offline/probe path is run.
2. There is no completed latest-artifact IsSameObject acceptance report.
3. If sidecar/object masks are wrong, the binding subspace can reinforce the
   wrong object unless gated by uncertainty and owner transport confidence.
```

### 3.3 Object explanation layer

Current flow:

```text
slot support over visual / temporal / point / tactile / tracklet / proposal
  -> object masks and background residuals
  -> anchor_quality / duplicate_overlap / contact_explanation_score
```

This is the correct invariant:

```math
\sum_j M_{j,i} + M_{\varnothing,i} = 1
```

where `M_null` is background/no-object residual.

Good:

```text
1. PICF now has an explicit background/no-object residual.
2. Object slots are not forced to own every token.
3. Duplicate overlap is measurable.
```

Deficits:

```text
1. It is not yet a full reconstruction/explanation objective like many OCL
   systems.  Several lambda weights remain zero or guarded.
2. The object masks are derived from support/proposals; they are not verified
   dense instance masks.
3. The layer is useful for diagnostics and gating, but not yet behavior-proven
   as a robust object decomposer.
```

### 3.4 Posterior file competition and birth competition

Current flow:

```text
support_raw over observation anchors
  -> same-role support/geometry duplicate test
  -> active file set
  -> demoted mass to dustbin
  -> birth competition for inactive reserve files
```

Mathematically:

```math
\operatorname{dup}_{j,k}
=
\max\left(
\cos(p_j,p_k),
\exp(-\|x_j-x_k\|^2/2\sigma^2)
\right)
```

and duplicate files are demoted:

```math
p_j^{active} = a_j p_j,\quad
p_{\varnothing}^{new}=p_{\varnothing}+\sum_j(1-a_j)p_j.
```

Good:

```text
This is the right fixed-capacity analogue of no-object/dustbin competition.
It addresses the historical "all role-1 files update from the same residual"
failure.
```

Deficits:

```text
1. Fixed capacity remains.  Files can be inactive/reserve, but the model does
   not yet learn a MetaSlot-style adaptive count from a codebook.
2. The active-file cap is a rule/gate, not a learned slot-quality selector.
3. Raw inactive overlap can remain high; this is acceptable only if inactive
   files are excluded from downstream action and object losses.
```

### 3.5 Posterior owner transport

Current flow:

```text
graph object owner responsibility
  -> observation anchor assignment
  -> posterior file binding
  -> accepted owner measurement
  -> precision fusion into posterior geometry
```

Key transport:

```math
T_{b,p}
=
\sum_o\sum_g B_{b,o}A_{o,g}P_{g,p}
```

```math
\bar{x}^{owner}_b
=
\frac{\sum_p T_{b,p}x_p}{\sum_pT_{b,p}}.
```

Then:

```math
x_b^+
=
\operatorname{KalmanFuse}(x_b^{std},S_b^{std},\bar{x}_b^{owner},S_b^{owner}).
```

Good:

```text
This is the most important recent repair.  It fixes a real gap between graph
anchors and persistent posterior files.
```

Deficits from current probe:

```text
step 50:
  active graph distance to proposal: about 0.77 px
  active posterior distance to proposal: about 0.73 px

step 100:
  active graph distance to proposal: about 1.64 px
  active posterior distance to proposal: about 6.66 px
```

Interpretation:

```text
Precision fusion is materially better than the old convex interpolation
failure, but owner confidence/mass can still weaken.  It is not yet a final
proof that posterior binding is solved.
```

## 4. `src/openpi/picf/core/training.py`

Relevant deployed loss families:

```text
loss_anchor_object_pull
loss_object_explanation_feature
loss_object_explanation_point
loss_object_explanation_contact
loss_object_explanation_duplicate
loss_object_explanation_background
loss_mapg_support_diversity
loss_anchor_pv
loss_pv_weak
loss_aqr_denoising
slot_jepa / support_pred / binding_consistency guarded hooks
```

Good:

```text
1. Object pull uses stop-gradient target from already-routed point/proposal
   evidence.
2. Row selection is detached so the model cannot trivially lower active gates
   to avoid geometry loss.
3. Object pull can be role-scoped, excluding effector rows when testing object
   ownership.
4. Matched predictive losses exist and old index-aligned loss is not the
   production default.
```

Deficits:

```text
1. `loss_anchor_object_pull` is a diagnostic/scaffold, not a final production
   behavior objective.
2. Object explanation losses are default-zero unless a run explicitly enables
   them.
3. The strongest mature OCL signal -- reconstruct/explain dense tokens through
   slots plus background -- is not fully used as a production loss.
4. `slot_jepa`, `support_pred`, and `binding_consistency` remain guarded because
   they can conflict with unstable slot assignment.
```

## 5. `scripts/picf_core_train.py`

Relevant responsibilities:

```text
build PicfObservation
load CALVIN windows
load sidecar proposal/tracklet/contact-motion artifacts
construct loss config from CLI
log anchor/object/owner metrics
write overlay JSON/PNG/GIF artifacts
```

Good:

```text
1. Owner transport precision gain is exposed and logged.
2. Blind SAM sidecars are rejected unless explicitly allowed for legacy
   reproduction.
3. Contact/task sidecar roots and segment indices can be passed explicitly.
4. Metrics now expose owner transport and active-vs-raw overlap.
```

Deficits:

```text
1. Sidecar quality and coverage remain external data dependencies.
2. The loader can still produce no tracklet/proposal evidence if manifests are
   absent or incomplete.
3. A run can be mathematically clean while still not behavior-proven; metrics
   and overlay review remain mandatory.
```

## 6. `scripts/picf_object_candidate_slot_binding_audit.py`

Current audit intent:

```text
verify object candidate -> graph owner -> posterior owner transport closure
verify active-file activation by owner transport
verify owner-priority selection
verify precision fusion instead of convex blend
```

Good:

```text
This closes several historical false positives: the graph could be correct while
posterior files stayed wrong, and the old tests would not catch it.
```

Deficit:

```text
The audit is synthetic/static.  It cannot prove CALVIN behavior, long-run
stability, or real sidecar quality.
```

## 7. Summary Table

| Capability | Current status | Strict verdict |
|---|---:|---|
| Typed multimodal evidence routing | Implemented | Strong code-level |
| Tactile-to-object owner gating | Implemented | Correct direction |
| Same-object pairwise subspace | Implemented | Needs latest-artifact probe |
| Object/background explanation | Implemented as layer | Partially trained/guarded |
| Posterior file competition | Implemented | Strong code-level |
| Posterior owner transport | Implemented with precision fusion | Improved, not behavior-proven |
| Adaptive slot count/codebook | Not implemented | Real gap |
| Full learned slot-quality selector | Not implemented | Real gap |
| Full dense reconstruction/object mask decoder | Not implemented | Deliberate gap, maybe future |
| Tracklet sidecar closure | Schema/scripts exist | Data coverage dependent |
| Blind SAM proposal path | Rejected/archived | Do not revive |
| Fourth-root/ordinal solver | Weak diagnostic | Not solved |

## 8. Strict Conclusion

The current implementation is materially more mature than the earlier v26/MVTrack
runtime:

```text
fixed:
  graph-only object binding without posterior write-through
  effector/blue role stealing object pull in the diagnostic
  convex owner interpolation
  uncalibrated common-mode binding signatures
  inactive duplicate rows entering object losses

still not fully solved:
  adaptive slot cardinality / learned prototype dedup
  full dense slot explanation as a production objective
  latest-run IsSameObject artifact acceptance
  full sidecar/tracklet coverage over the training dataset
  long-run posterior owner stability
```

Therefore the rigorous answer is:

```text
The advanced-slot/tactile binding issue is not "彻底解决".
It is partially repaired at the code-level, with the most important posterior
closure now implemented, but mature slot-method parity still has concrete gaps.
```

