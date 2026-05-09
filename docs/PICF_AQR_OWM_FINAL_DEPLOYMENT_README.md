# PICF-AQR-OWM Final Deployment Contract

Date: 2026-05-09

Status:

```text
Canonical final architecture contract and full deployment specification.
This document supersedes MAPG-v0 as the long-term graph/router direction.
It does not delete current AQR; it upgrades AQR into an object-addressable,
predictive, posterior-centered belief-state architecture.
```

Source note:

```text
The user-provided method was expected at docs/temp_method.md, but the local
file is currently empty. This README is built from the method text provided in
the conversation and from direct code inspection of the current repository.
```

## 1. Final Verdict

The final architecture should be:

```text
PICF-AQR-OWM

PICF:
  posterior-centered predictive control framework

AQR:
  anchor query router over typed support memory

OWM:
  object-addressable predictive world/belief model
```

The central invariant is:

```text
posterior is the authoritative current belief state.
typed support memory is current evidence.
AQR queries route evidence into object/task anchors.
prediction cache and evidence cache are auxiliary context.
PI0.5 remains the final action generator.
```

This is not:

```text
PaliGemma heatmap -> point prior -> control
```

It is also not:

```text
one global dense transformer that mixes every modality and hopes objects emerge.
```

The correct object is:

```text
anchor/slot =
  persistent address
  current content
  optional geometry
  uncertainty/existence
  multimodal support distributions
  posterior binding history
```

## 2. Why This Is One Coherent Architecture

The system is a neural belief filter for a partially observed robot task.

At time `t`, the robot has hidden state:

```text
s_t:
  object states
  robot state
  contact state
  task-relevant latent state
```

It observes:

```text
o_t:
  RGB/video
  language
  point/depth
  tactile
  proprio
  previous action
```

The policy should act from a belief:

```text
b_t(s) = p(s_t = s | o_<=t, a_<t, language)
```

The classical belief update is:

```text
b_t(s) proportional to p(o_t | s) * integral p(s | s', a_{t-1}) b_{t-1}(s') ds'
```

PICF already approximates this structure:

```text
prior:
  previous posterior + previous action + proprio

measurement:
  current observation anchors + dense visual/tactile/point evidence

assignment:
  binding between prior slots and current observations

correction:
  posterior update with uncertainty / precision-like fusion

innovation:
  current evidence vs previous world-only prediction

action:
  PI0.5-conditioned action path
```

Therefore the final architecture should not invent a separate world model that
bypasses PICF. It should make the existing posterior/prediction/innovation
contract more explicit and object-addressable.

## 3. Code Audit: What Already Exists

### 3.1 Anchor Graph Contract Exists

Current file:

```text
/home/siyuanyue/Documents/openpi/src/openpi/picf/core/contracts.py
```

Relevant object:

```text
PicfAnchorPriorGraphState
```

Current fields include:

```text
pg_priors
visual_priors
point_priors
tactile_priors
posterior_priors
anchor_tokens
anchor_roles
anchor_scores
anchor_confidence
anchor_x
anchor_S
geometry_valid
obs_slot_assignment
task_assignment
modality_confidence
```

Decision:

```text
Keep this as the compatibility surface for observation/task/posterior/control
consumers, but extend it for first-class temporal V-JEPA support, first-class
PaliGemma image support, cache support, slot address/content fields, and
support uncertainty.
```

### 3.2 Posterior Is Already A Belief State

Current object:

```text
PicfPosteriorAnchorState
```

Current fields include:

```text
h, c
mu, Sigma
x, S
a
alpha
contact_prob
support_mass
recycle_gate
binding
evidence_tokens
tokens
global_post
role_ids
```

Current update path:

```text
_current_prior(...)
_posterior_update(...)
_innovation(...)
make_recurrent_carry(...)
```

Key code facts:

```text
_current_prior:
  previous posterior + proprio + previous action -> prior

_posterior_update:
  prior + observation anchors + graph bias + native visual/tactile reread
  -> posterior

_innovation:
  current targets vs previous physical_prediction_cache
  -> standardized residual / innovation token

make_recurrent_carry:
  carries posterior and physical_prediction_cache forward
```

Decision:

```text
Posterior must remain authoritative. Evidence cache/KV cache may assist it but
must never replace it or bypass its correction/innovation path.
```

### 3.3 AQR Exists But Is Not Yet Full OWM

Current AQR core:

```text
aqr_physical_query_tokens
aqr_task_query_tokens
_aqr_pg_image_support_read
_aqr_competitive_support
_build_aqr_anchor_graph
```

Current strengths:

```text
learned physical/task queries
PaliGemma semantic conditioning for task queries
PaliGemma image-token support remapped onto V-JEPA grid
V-JEPA/point/tactile/posterior readers
Sinkhorn-like competitive support normalization
PI0.5 path preserved
```

Current gaps:

```text
1. pg_priors is returned as None in AQR even though the contract supports it.
2. PaliGemma image support is mostly converted into V-JEPA visual bias.
3. V-JEPA temporal tokens are collapsed to one 2D map through current_map(...).
4. aqr_temporal_memory_tokens exists in config but is not used in pipeline.
5. posterior slots do not explicitly split address from content.
6. evidence cache is not a full object-addressable typed cache.
7. world prediction is global/posterior-token level, not slot-level JEPA yet.
8. ordinal/relation grounding is not an explicit head yet.
```

Decision:

```text
AQR is the right skeleton. The final deployment upgrades it; it does not roll
back to MAPG-v0 or to PaliGemma heatmap routing.
```

## 3.4 Method-To-Code Audit Matrix

This section maps the proposed method item by item to the current codebase.

| Method claim | Current code fact | Deployment decision |
| --- | --- | --- |
| Last-two-frame JEPA evidence should be explicit | `VjepaFeatureMap.current_map(...)` returns last latent slice or mean of last two latent slices; `_visual_map(...)` passes one 2D map downstream | Replace mean-only path with first-class recent temporal V-JEPA tokens; keep mean as ablation only |
| PICF already has memory | `PicfPosteriorAnchorState` carries `h/c/mu/Sigma/x/S/alpha/binding/evidence_tokens/tokens`; recurrent carry preserves posterior | Treat posterior as authoritative belief, not as a cache to be replaced |
| Posterior predicts and corrects | `_current_prior(...)`, `_posterior_update(...)`, `_innovation(...)`, and `physical_prediction_cache` already implement prior/measurement/correction/error | Preserve and strengthen this belief-filter loop; do not let AQR/cache bypass it |
| AQR should read typed memory | Current AQR separately reads visual, point, tactile, posterior; PaliGemma semantic conditions task queries | Keep AQR as the routing skeleton and extend typed memory rather than reverting to MAPG-v0 |
| PaliGemma image tokens should be first-class | `_aqr_pg_image_support_read(...)` reads image tokens but remaps support into V-JEPA bias; AQR returns `pg_priors=None` | Fill `graph.pg_priors`; preserve per-view PaliGemma image support as a typed branch |
| PaliGemma heatmap should not dominate | `aqr_pg_grounding_enabled=False`, `aqr_pg_bias_weight=0.0` defaults already encode this | Keep heatmap off by default; use only explicit ablation/diagnostic flags |
| V-JEPA temporal support should be first-class | Config has `aqr_temporal_memory_tokens`, but it is not consumed in `pipeline.py` | Implement temporal support fields and route AQR over `[time, h, w]` tokens |
| Physical slots and task anchors should be separated | Code already has `aqr_physical_query_tokens` and `aqr_task_query_tokens`; only task queries read semantic conditioner | Freeze this as a core invariant |
| Address/content split should exist | Current posterior has latent/content-like fields, binding, recycle, and roles, but no explicit address vector | Add address vectors after verifying current binding stability |
| Cache must be subordinate to posterior | Current recurrent carry stores `physical_prediction_cache`, not a long causal evidence cache | Add bounded evidence cache with age/source/uncertainty metadata; never bypass posterior |
| Slot-level JEPA world prediction is desired | Current prediction is global/posterior-token level and produces `physical_prediction_cache` | Add slot-level JEPA after identity/binding diagnostics are stable |
| Ordinal/relation grounding is required for tasks like "fourth from left" | No explicit rank/relation head in current AQR/PICF losses | Add gated relation head later, only for high-confidence relation language |
| Fine adjacent-object selection needs more than global support | V-JEPA native map is `384/16 ~= 24x24`; current AQR can bind supports but cannot create sub-token evidence | Add point-neighborhood, temporal, and latent local refinement; do not promise impossible sub-token identity |

## 4. Literature Contract

The literature does not prove this exact codebase. It supports the principles
that should be combined in this architecture.

### 4.1 Slot/Anchor Binding

Slot Attention supports competitive object-centric binding through learned
slots. This supports making anchors first-class bindable entities rather than
hoping global attention separates instances by itself.

MESH / OT-style Slot Attention supports the need for Sinkhorn/OT competition
to improve tie-breaking and separation.

Contract implication:

```text
instance/object binding belongs in slot/anchor competition,
not in unconstrained global cross-attention alone.
```

### 4.2 Query-Based Scalable Multimodal Reads

Perceiver IO and Q-Former/Perceiver-style resamplers support using a small
query/latent set to read large structured inputs with roughly linear input
scaling.

Contract implication:

```text
typed memory + anchor queries is scaling-friendly:
  O(K * sum_m N_m) + O(K^2)
instead of
  O((sum_m N_m)^2)
```

### 4.3 JEPA/V-JEPA For VLA

JEPA-VLA argues that video predictive embeddings, especially V-JEPA 2
representations from recent frames, provide useful temporal dynamics and policy
priors for VLA models.

VLA-JEPA supports leakage-free latent state prediction: future latent
information is a target, not current input.

Contract implication:

```text
V-JEPA recent temporal embeddings should be typed support tokens.
Future states may supervise prediction but must not leak into current action.
```

### 4.4 Object-Centric World Models

Object-centric world models and slot-structured world models support predicting
future state in object/slot space rather than in a monolithic global latent.

OA-WAM supports separating object address from time-varying content for
object-addressable manipulation.

Contract implication:

```text
posterior slots should have persistent address and current content.
slot-level prediction is the correct long-term world-model objective.
```

### 4.5 Robot Spatial Grounding

CLIPort, Transporter, PerAct, RVT/RVT-2, and related manipulation work support
explicit spatial/action-relevant representations rather than relying on generic
VLM attention to discover "where" implicitly.

Contract implication:

```text
language gives what/referring intent;
AQR/slots/points/temporal evidence determine where/how-to-act.
```

## 5. Final State Definition

Each persistent physical slot should be:

```text
S_{t,j} =
  address_j
  content_{t,j}
  mu_{t,j}, Sigma_{t,j}
  x_{t,j}, S_{t,j}
  alpha_{t,j}
  contact_prob_{t,j}
  role_j
  support distributions
  uncertainty/confidence metadata
```

Meaning:

```text
address_j:
  persistent slot/object identity carrier

content_{t,j}:
  current visual/semantic/contact/dynamic content

mu, Sigma:
  latent belief distribution

x, S:
  optional metric geometry belief

alpha:
  existence / activity / visibility

support distributions:
  where current evidence came from in each typed memory
```

Important rule:

```text
address is not RoPE.
address is not just slot index.
address is persistent identity carrier managed by posterior binding/recycle.
```

RoPE or multi-dimensional positional encoding should encode:

```text
time
image h/w
view/camera
point xyz/projection
tactile sensor/taxel coordinate
language position
```

It should not encode slot identity directly.

## 6. Final Typed Support Memory

The final memory must keep typed branches first-class:

```text
M_text:
  PaliGemma text / semantic tokens

M_pg_img:
  PaliGemma image tokens per view

M_vjepa:
  recent V-JEPA temporal-spatial tokens

M_point:
  Sonata / point / metric geometry tokens

M_tactile:
  AnyTouch / contact tokens

M_proprio:
  robot proprio and gripper state tokens

M_action:
  previous action / action chunk tokens

M_post:
  posterior slot tokens

M_cache:
  bounded causal evidence cache
```

Do not reduce this to:

```text
PaliGemma image -> V-JEPA scalar bias -> everything else
```

That may be used as an auxiliary bridge, but not as the only representation of
PaliGemma image evidence.

## 7. Final Query Structure

### 7.1 Physical Queries

Physical queries maintain world state:

```text
role
coverage/address code
proprio
previous posterior summary
previous action
temporal belief context
```

They should avoid strong direct PaliGemma heatmap control.

### 7.2 Task Queries

Task queries select and refine task-relevant supports:

```text
role
coverage/address code
PaliGemma semantic tokens
PaliGemma image tokens
V-JEPA temporal support
point/tactile/posterior support
language relation/ordinal context
```

Task queries can use PaliGemma strongly for semantics, but not as a spatial
oracle. PaliGemma heatmaps remain diagnostic/ablation unless explicitly enabled
and confidence-gated.

## 8. Final Routing Equations

For each query or slot `j` and modality `m`:

```text
logits_{j,i}^{m} =
  <Wq_m q_j, Wk_m e_i^m> / sqrt(d)
  + b_role
  + b_valid
  + b_geometry
  + b_temporal
  + b_prior
```

Then:

```text
p_{j}^{m} = competitive_normalize(logits_j^m)
```

Competition should be Sinkhorn-like:

```text
row constraint:
  each active query explains support

column/capacity constraint:
  one evidence token/region should not be consumed by every same-role query

role mask:
  incompatible support is disabled
```

Modality summaries:

```text
r_j^m = sum_i p_{j,i}^m * V_m e_i^m
```

Modality gate:

```text
beta_j^m = gate(query, role, support entropy, modality validity, uncertainty)
```

Content update:

```text
content_j = sum_m beta_j^m * r_j^m
```

Posterior correction remains the owner of final belief:

```text
prior + measurement evidence -> posterior
```

## 9. V-JEPA Temporal Support: Required Change

Current code:

```text
VjepaFeatureMap.current_map(use_last_two_mean=False/True)
```

Current behavior:

```text
64-frame V-JEPA clip
-> temporal latent tokens
-> last latent slice or mean(last two latent slices)
-> one 2D visual map
```

Final behavior:

```text
recent V-JEPA temporal tokens are preserved:
  M_vjepa[tau, h, w]

default:
  last_two_tokens

optional ablations:
  last_only
  last_two_mean
  last_mean_delta
  last4_tokens
```

Implementation requirement:

```text
1. Add VjepaFeatureMap.recent_maps(n).
2. Add temporal visual token flattening with time ids.
3. Extend PicfTokenFieldState or a new typed memory state to preserve time ids.
4. Let AQR visual reader read [tau, h, w] tokens.
5. Emit temporal support diagnostics.
```

Rationale:

```text
averaging reduces noise in static scenes but smears motion/contact boundaries.
explicit temporal tokens preserve dynamics and allow AQR to choose which time
slice matters.
```

## 10. PaliGemma Image Support: Required Change

Current code:

```text
_aqr_pg_image_support_read(...)
  reads first image_token_range
  returns updated task queries and visual-grid bias

_build_aqr_anchor_graph(...)
  returns pg_priors=None
```

Final behavior:

```text
PaliGemma image support remains a first-class branch:
  p_pg_img[j, view, token]

It may also provide a weak visual-grid bias after resize-with-pad remapping,
but it must not disappear into the V-JEPA branch.
```

Implementation requirement:

```text
1. Return both updated query and pg image priors from _aqr_pg_image_support_read.
2. Iterate all image_token_ranges/views, not only index 0.
3. Preserve view dimension or record view ids.
4. Fill graph.pg_priors in AQR path.
5. Add pg_image confidence and diagnostics.
6. Extend graph losses to optionally include PG image pooled embedding.
```

PaliGemma heatmap rule:

```text
aqr_pg_grounding_enabled = False by default
aqr_pg_bias_weight = 0.0 by default
```

PaliGemma semantic/image tokens are useful. PaliGemma heatmap is not the
production where mechanism.

## 11. Posterior Authority And Cache Rule

Hard invariant:

```text
posterior is authoritative current belief.
cache is auxiliary evidence memory.
action should not bypass posterior correction.
```

Current recurrent carry:

```text
posterior
executed_action
physical_prediction_cache
```

Final evidence cache:

```text
PicfEvidenceCacheEntry:
  tokens
  source: real_observation | posterior | predicted
  slot_address
  role
  age
  uncertainty
  innovation_at_write
  modality_validity
  support summaries
```

Read gate:

```text
cache_weight = gate(query, age, uncertainty, source, current innovation)
```

If current innovation is high:

```text
downweight stale cache
trust current measurement
allow posterior correction/rebinding
```

Never do:

```text
old cache -> direct action truth
```

## 12. Address / Content Separation

Required final state:

```text
address_j:
  persistent identity carrier

content_tj:
  time-varying evidence state
```

Initial implementation:

```text
reuse posterior role_ids / h / c / tokens / binding / recycle_gate
add explicit address vectors
keep address slow-moving or fixed except recycle/birth
use address in binding keys
use content in measurement/action values
```

Do not implement address as:

```text
RoPE slot index
ordinary residual token that can be overwritten each step
```

Risk:

```text
if address-content split is added before binding is stable,
it can become a cosmetic field. Acceptance tests must verify identity stability.
```

## 13. Slot-Level JEPA World Model

This is the final predictive layer, but it must be activated after binding is
stable.

Current prediction:

```text
posterior/global physical predictive basis
-> physical_prediction_cache
-> innovation compares next observation against world-only prediction
```

Final prediction:

```text
for each physical slot j:
  predict next content
  predict next geometry distribution
  predict next support summary
  predict next contact state
```

No leakage rule:

```text
future observations may be target encoder inputs only.
student/action path sees only current/past observations.
```

Loss:

```text
L_slot_jepa =
  distance(predicted_next_slot_j, stopgrad(target_next_slot_{pi(j)}))
```

Where `pi(j)` is binding/matching.

Activation rule:

```text
do not activate full slot JEPA until posterior binding identity is stable.
start with next posterior token/content prediction, not full future rollout.
```

## 14. Ordinal / Relation Grounding

Problem:

```text
"the fourth chopstick from the left" is not saliency.
It is candidate set + ordering axis + ordinal selection.
```

Final relation head:

```text
inputs:
  task query
  language tokens
  physical slots
  slot geometry/support

outputs:
  relation axis
  pairwise relation logits
  soft rank
  selected slot scores
```

Soft rank example:

```text
score_j = u_language^T x_j
rank_j = 1 + sum_{l != j} sigmoid((score_l - score_j) / tau_rank)
```

Activation rule:

```text
enable only for high-confidence ordinal/relation language
use low weight
do not let weak pseudo-rank labels dominate action loss
```

## 15. Fine Instance Limitation And Refinement

Hard information-theoretic limit:

```text
I(target identity; anchor decision) <= I(target identity; typed memory)
```

If several chopsticks are indistinguishable in all available visual/point/
tactile/temporal evidence, no architecture can recover the correct one
reliably.

Final no-new-data refinement order:

```text
1. point-neighborhood refinement
2. temporal/posterior disambiguation
3. latent local refinement over top-k support tokens
```

High-resolution RGB crop is useful only if higher-resolution source images are
available. If the pipeline only has 384 input and 16px patches, crop cannot
invent sub-token information.

## 16. Loss Contract

Do not activate every possible loss at full weight at once.

Full final loss family:

```text
L =
  L_action
  + lambda_slot_jepa       L_slot_jepa
  + lambda_support_pred    L_support_pred
  + lambda_bind            L_binding_consistency
  + lambda_div             L_slot_diversity
  + lambda_xmod            L_cross_modal_align
  + lambda_rank            L_ordinal_relation
  + lambda_innov           L_innovation_calibration
  + lambda_mask            L_masked_modality
```

Current active/available family:

```text
L_action
MAPG/AQR SigLIP-style cross-modal matching
VICReg
cycle consistency
masked modality
routing
support overlap diversity
geometry diversity
alignment budget scaling
```

Final activation order:

```text
Stage A:
  current AQR losses + first-class PG/V-JEPA temporal support diagnostics

Stage B:
  binding consistency + address/content diagnostics

Stage C:
  slot JEPA next-token/content prediction

Stage D:
  support prediction

Stage E:
  ordinal/relation loss for high-confidence language only
```

The final architecture is not cut down; staged activation is required to avoid
turning unverified future targets and noisy weak relation labels into
optimization noise.

## 17. Full Deployment Plan

This section is the concrete full deployment plan. Every item belongs to the
final architecture; phases are validation gates, not architectural omissions.

### Phase 1: Temporal V-JEPA Typed Support

Implement:

```text
VjepaFeatureMap.recent_maps(n)
visual_temporal_tokens
visual_time_ids
visual_xy_ids
graph.vjepa_temporal_priors or equivalent support field
AQR read over temporal visual tokens
diagnostics: per-time support mass / entropy / overlap
```

Tests:

```text
unit test recent_maps shape
unit test last_two_tokens != last_two_mean contract
smoke train forward with AQR enabled
CALVIN debug export temporal support overlays
```

Acceptance:

```text
startup log states temporal mode
debug records per-time support mass
no future frame leakage
```

### Phase 2: First-Class PaliGemma Image Support

Implement:

```text
_aqr_pg_image_support_read returns pg priors
all PG image views/ranges handled
graph.pg_priors filled in AQR path
PG image support pooled embedding added to graph losses
PG image support diagnostics exported
```

Tests:

```text
unit test multi-view/range aggregation
unit test graph.pg_priors non-None when image support enabled
unit test heatmap disabled still keeps image-token support active
```

Acceptance:

```text
PaliGemma heatmap remains off by default
PaliGemma image priors are visible in debug and losses
```

### Phase 3: Posterior Address / Content Split

Implement:

```text
posterior address vectors
address/content projections
binding uses address keys and content/geometry values
recycle/birth updates address intentionally
identity stability diagnostics
```

Tests:

```text
shape/backward tests
posterior carry compatibility test
identity switch metric on CALVIN debug rollouts
```

Acceptance:

```text
address does not drift every frame
content updates with measurement
recycle explicitly resets/changes address when needed
```

### Phase 4: Posterior-Grounded Evidence Cache

Implement:

```text
PicfEvidenceCacheState
cache entries with source/age/uncertainty/slot address/innovation metadata
cache read gate
innovation-gated cache downweighting
recurrent carry compatibility
```

Tests:

```text
cache age increments
reset clears cache
innovation high -> cache weight drops
predicted-only cache cannot bypass posterior
```

Acceptance:

```text
cache improves temporal continuity without stale hallucination lock-in
```

### Phase 5: Slot-Level JEPA Prediction

Implement:

```text
slot next-content predictor
detached next-posterior target path
matching pi(j) from posterior binding/address
slot_jepa loss with small weight
no future input leakage into action path
```

Tests:

```text
teacher target detached
student path current/past only
loss finite on short unroll
```

Acceptance:

```text
innovation calibration improves
posterior correction remains active
action loss does not destabilize
```

### Phase 6: Support Prediction

Implement:

```text
predict next visual/point/tactile/posterior support summaries
support_pred loss over matched slots
masked by modality availability
```

Acceptance:

```text
support predictions are better than uniform baseline
do not force tactile contact point to match full visual object extent
```

### Phase 7: Ordinal / Relation Head

Implement:

```text
language ordinal parser / detector
relation axis head
soft rank / pairwise relation logits
high-confidence weak supervision only
```

Acceptance:

```text
rank loss active only on explicit relation language
no degradation on non-ordinal tasks
```

## 18. Current Training Profile To Preserve

Keep:

```text
PI0.5 action path unchanged
V-JEPA/Sonata/AnyTouch pretrained parts frozen under frozen-perception profile
PaliGemma semantic/action path trainable under current semantic profile
AQR/PICF adapters/router/posterior/control trainable
save interval controlled by train command
tmux/cloud training operational practice unchanged
```

Do not silently enable:

```text
legacy --mapg-enabled candidate-prior graph
--vl-anchor-router-enabled production routing
--aqr-pg-grounding-enabled production heatmap routing
--aqr-pg-bias-weight > 0 without ablation label
```

## 19. Required README / Code Navigation

Read first:

```text
/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md
/home/siyuanyue/Documents/openpi/docs/AQR_MAPG_HANDOFF_README.md
/home/siyuanyue/Documents/openpi/docs/AQR_MAPG_DIRECT_FINAL_DEPLOYMENT_README.md
/home/siyuanyue/Documents/openpi/docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md
```

Then:

```text
/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md
/home/siyuanyue/Documents/openpi/src/openpi/picf/README_FROZEN_PERCEPTION_AUGMENTATION.md
/home/siyuanyue/Documents/openpi/src/openpi/picf/README_PI05_PARITY_AUDIT.md
```

Main code:

```text
src/openpi/picf/core/contracts.py
src/openpi/picf/core/config.py
src/openpi/picf/core/pipeline.py
src/openpi/picf/core/training.py
src/openpi/picf/vjepa/wrapper.py
src/openpi/picf/vjepa/config.py
scripts/picf_core_train.py
scripts/serve_picf_policy.py
```

## 20. Acceptance Audit Checklist

Before calling the final OWM path successful, verify:

```text
1. action loss trend does not regress vs current AQR baseline
2. temporal V-JEPA supports are non-uniform and time-selective when motion/contact changes
3. graph.pg_priors is populated when PG image support is enabled
4. PaliGemma heatmap remains off unless explicit ablation
5. same-role support overlap remains controlled
6. effective anchor count is not collapsed
7. posterior identity switch rate is acceptable
8. innovation spikes cause correction, not cache lock-in
9. geometry_valid and uncertainty behave sensibly under point missing/noisy cases
10. CALVIN debug videos show anchors on task/interaction/effector regions
11. JSON metrics include support entropy, per-time support mass, PG support metrics,
    posterior identity switch, cache trust, innovation norms
12. no future observation leaks into current action input
```

## 21. Final Non-Negotiable Rules

```text
1. Posterior is authoritative belief.
2. AQR routes typed evidence; it is not a standalone keypoint extractor.
3. PaliGemma semantics/images are support sources; heatmap is not default where.
4. V-JEPA temporal embeddings are first-class evidence, not averaged away.
5. Cache is subordinate to posterior and innovation.
6. Address/content split is for identity stability, not cosmetic slot indexing.
7. World prediction is leakage-free and target-only for future observations.
8. Relation/ordinal grounding is explicit but confidence-gated.
9. Fine instance selection cannot exceed information present in typed memory.
10. PI0.5 remains the action generator.
```

## 22. Final Judgment

PICF-AQR-OWM is the correct final architecture direction:

```text
typed evidence
-> object/task anchor routing
-> posterior correction
-> predictive belief
-> innovation-aware action
```

It is mathematically coherent because every component corresponds to a belief
filter role:

```text
typed memory:
  measurement evidence

AQR slots:
  measurement routing and object/task binding

posterior:
  belief correction and identity continuity

prediction cache / slot JEPA:
  transition model

innovation:
  prediction error and correction trigger

evidence cache:
  bounded historical evidence, never current truth

PI0.5:
  action generator over current belief/context
```

This is a complete final contract. It is not a minimal patch. The deployment is
phase-gated only because posterior identity, temporal support, cache trust, and
future prediction are tightly coupled; enabling all new losses without gates
would be mathematically less coherent, not more complete.

## References

- Slot Attention: https://arxiv.org/abs/2006.15055
- MESH / OT-style Slot Attention: https://arxiv.org/abs/2301.13197
- Perceiver IO: https://arxiv.org/abs/2107.14795
- JEPA-VLA: https://arxiv.org/abs/2602.11832
- VLA-JEPA: https://arxiv.org/abs/2602.10098
- Object-Centric World Model for Language-Guided Manipulation: https://arxiv.org/abs/2503.06170
- Slot Structured World Models: https://arxiv.org/abs/2402.03326
- OA-WAM: https://arxiv.org/abs/2605.06481
- Causal World Modeling for Robot Control: https://arxiv.org/abs/2601.21998
- DreamZero / World Action Models: https://arxiv.org/abs/2602.15922
- Deformable DETR: https://arxiv.org/abs/2010.04159
- PerAct: https://arxiv.org/abs/2209.05451
- DOrA order-aware 3D grounding: https://arxiv.org/abs/2403.16539
