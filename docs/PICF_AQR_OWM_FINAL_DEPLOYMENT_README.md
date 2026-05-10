# PICF-AQR-OWM Final Deployment Contract

Date: 2026-05-09

Version:

```text
PICF-AQR-OWM v1.0
This version is complete for the code-level deployment contract.
The scripted verifier and regression set below are the required re-checks after
any follow-up change.
```

Status:

```text
Canonical final architecture contract and full deployment specification.
This document supersedes MAPG-v0 as the long-term graph/router direction.
It does not delete current AQR; it upgrades AQR into an object-addressable,
predictive, posterior-centered belief-state architecture.
```

Source:

```text
Consolidated PICF-AQR-OWM architecture contract based on the current AQR/PICF
code, posterior audit, deployment discussion, and docs/temp_method.md review.
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

Deployment verdict:

```text
GO:
  implement the complete PICF-AQR-OWM architecture as the final target.

NO-GO:
  treat a README-only or verifier-failing checkout as final OWM.

GO WITH HARD GUARDS:
  run the final implementation only after the Definition of Done passes:
  state contracts, forward wiring, tests, diagnostics, no-leakage checks, and
  CALVIN/debug evidence.
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

## 3.5 Complete Deployment Definition

This document targets a direct final implementation of the complete graph below.
When later sections mention gates, they mean reviewable build and validation
checks inside one final deployment, not a reduced architecture and not a claim
that the current code is already final OWM.

```text
Observation / prompt / previous carry
  -> PaliGemma text/image tokens
  -> V-JEPA recent temporal tokens
  -> point / tactile / proprio / previous-action tokens
  -> previous posterior slots and bounded evidence cache
  -> typed support memory
  -> physical/task AQR queries
  -> competitive multimodal support routing
  -> anchor graph state
  -> observation anchors
  -> posterior prior/binding/correction
  -> slot-level prediction and innovation
  -> task readout / ordinal selectors
  -> conditioned control tokens
  -> PI0.5 action generator
  -> recurrent carry
```

The complete system is considered deployed only when all of these objects exist
in code, are wired into the forward path, are covered by tests, and are visible
in debug/JSON diagnostics:

```text
1. first-class recent V-JEPA temporal support
2. first-class PaliGemma image-token support
3. posterior address/content split
4. posterior-grounded bounded evidence cache
5. slot-level JEPA next-state prediction
6. support prediction over typed memory summaries
7. ordinal/relation selector
8. innovation-gated correction/cache trust
9. no-leakage future-target discipline
10. PI0.5 action path unchanged
```

If one of these is absent, the code may still be a valid ablation or a safe
intermediate checkpoint, but it is not the final PICF-AQR-OWM deployment.

## 3.6 Concrete Code-Level Deployment Blueprint

This section is the implementation contract. It names the exact current files,
the new or changed interfaces, and the invariants that must remain true.

### 3.6.1 Contracts And State Objects

Current file:

```text
src/openpi/picf/core/contracts.py
```

Required extensions:

```python
@dataclasses.dataclass
class PicfTemporalVisualSupportState:
    tokens: torch.Tensor          # [N, D]
    time_ids: torch.Tensor        # [N]
    view_ids: torch.Tensor        # [N], zero for current static-only V-JEPA
    grid_index: torch.Tensor      # [N, 2]
    grid_norm: torch.Tensor       # [N, 2]
    valid: torch.Tensor


@dataclasses.dataclass
class PicfEvidenceCacheState:
    tokens: torch.Tensor                  # [H, K, D]
    slot_address: torch.Tensor            # [H, K, D_addr]
    role_ids: torch.Tensor                # [H, K]
    source_ids: torch.Tensor              # [H, K]
    age: torch.Tensor                     # [H, K]
    uncertainty: torch.Tensor             # [H, K]
    innovation_at_write: torch.Tensor     # [H, K]
    modality_validity: torch.Tensor       # [H, K, M]
    valid: torch.Tensor                   # [H, K]
```

Cache implementation rule:

```text
Use a fixed-size ring buffer in code. A dynamic tuple of entries is acceptable
for prose examples, but it should not be the main training/checkpoint contract
because it makes batching, shape checks, and recurrent carry compatibility less
stable.
```

Required changes to existing states:

```text
PicfTokenFieldState:
  add temporal_visual: PicfTemporalVisualSupportState | None

PicfAnchorPriorGraphState:
  keep pg_priors but make AQR populate it
  add vjepa_temporal_priors: torch.Tensor | None
  add cache_priors: torch.Tensor | None
  add slot_address: torch.Tensor | None
  add slot_content: torch.Tensor | None
  add support_uncertainty: torch.Tensor | None

PicfPosteriorAnchorState:
  add address: torch.Tensor | None
  add content: torch.Tensor | None
  keep h/c/tokens for backward compatibility until migration is complete

PicfPredictiveState:
  add slot_prediction_tokens: torch.Tensor | None
  add slot_prediction_supports: dict[str, torch.Tensor] | None or explicit fields
  add evidence_cache: PicfEvidenceCacheState | None

PicfRecurrentCarryState:
  carry posterior, physical_prediction_cache, and evidence_cache
```

Compatibility rule:

```text
Existing graph consumers must continue to read visual_priors / point_priors /
tactile_priors / posterior_priors / anchor_tokens. New fields are additive.
Do not break current observation/task/posterior/control consumers while adding
the OWM fields.
All newly added dataclass fields must have default None/default factory values
or every construction path must be updated in the same commit.
```

### 3.6.2 V-JEPA Temporal Support

Current files:

```text
src/openpi/picf/vjepa/wrapper.py
src/openpi/picf/vjepa/config.py
src/openpi/picf/core/pipeline.py
```

Current problem:

```text
VjepaFeatureMap.current_map(...)
  returns one [H, W, C] map:
    last latent slice, or mean(last two latent slices)

pipeline._visual_map(...)
  passes only that one map into token-field construction
```

Required implementation:

```python
@dataclasses.dataclass(frozen=True)
class VjepaFeatureMap:
    ...
    def recent_maps(self, n: int = 2) -> torch.Tensor | np.ndarray:
        # return [T_recent, H, W, C], never averaged
```

Pipeline changes:

```text
1. Keep _visual_map(...) for legacy/ablation current-map consumers.
2. Add _visual_temporal_maps(...) returning [T_recent, H, W, C].
3. In _build_token_field(...), flatten temporal maps into
   temporal_visual.tokens = [T_recent * H * W, C].
4. Preserve time_ids and grid ids.
5. AQR visual reader should read temporal_visual.tokens when present.
6. The current 2D visual_priors can be derived by summing temporal priors
   over time for backward-compatible consumers.
```

Default mode:

```text
aqr_vjepa_temporal_mode = last_two_tokens
```

Ablation modes:

```text
last_only
last_two_mean
last_two_tokens
last_mean_delta
last4_tokens
```

No-leakage invariant:

```text
Only current/past clip evidence can be used by the action path.
Future V-JEPA targets can supervise slot prediction but cannot enter AQR,
posterior, task readout, or PI0.5 action inputs for the same step.
```

Required tests:

```text
src/openpi/picf/vjepa/wrapper_test.py:
  recent_maps(n=2) shape and exact slice order
  recent_maps(n) clamps safely when fewer temporal slices exist
  current_map(use_last_two_mean=True) remains an ablation, not default temporal support

src/openpi/picf/core/pipeline_test.py:
  AQR receives temporal visual tokens
  temporal prior shape is [anchors, T_recent * H * W]
  summed temporal prior matches visual_priors row mass
  future target tensors are detached from action/current path
```

### 3.6.3 PaliGemma Image Support As A First-Class Branch

Current file:

```text
src/openpi/picf/core/pipeline.py
```

Current problem:

```text
_aqr_pg_image_support_read(...)
  only reads semantic.image_token_ranges[0]
  only uses semantic.image_view_transforms[0]
  returns updated query and visual-grid bias

_build_aqr_anchor_graph(...)
  returns pg_priors=None
```

Required implementation:

```text
1. Rename return to:
   updated_q, pg_priors, pg_visual_bias, pg_view_ids

2. Iterate all image_token_ranges / image_grid_shapes / image_view_transforms.

3. Preserve per-view support before any visual-grid projection:
   pg_priors shape should be either:
     [K, total_pg_image_tokens]
   or:
     [K, V, max_tokens_per_view] with masks.

4. Continue to optionally project PG support into V-JEPA grid as a bias,
   but do not let this projection be the only surviving PG image state.

5. Fill PicfAnchorPriorGraphState.pg_priors in AQR path.

6. Add PG image pooled embeddings to graph losses only with valid masks.
```

Required default:

```text
aqr_pg_image_support_enabled = True
aqr_pg_image_support_weight = 0.35
aqr_pg_grounding_enabled = False
aqr_pg_bias_weight = 0.0
```

Meaning:

```text
PaliGemma language/image tokens participate in query support.
PaliGemma heatmap does not determine "where" in production.
```

Required tests:

```text
pipeline_test:
  graph.pg_priors is non-None when image support is enabled and PG image tokens exist
  graph.pg_priors is None/zero only when image support is disabled or tokens absent
  multi-view PG ranges are all consumed
  heatmap disabled does not disable PG image-token support
  resize-with-pad projection still uses PaliGemmaViewTransform when producing visual bias

training_test:
  PG image pooled embedding enters SigLIP/VICReg only when valid
  invalid PG rows do not create positive/negative loss pairs
```

### 3.6.4 AQR Routing And Support Competition

Current files:

```text
src/openpi/picf/core/pipeline.py
src/openpi/picf/core/training.py
```

Current useful implementation:

```text
aqr_physical_query_tokens
aqr_task_query_tokens
aqr_task_conditioner
aqr_visual_reader / point_reader / tactile_reader / posterior_reader
_aqr_competitive_support(...)
support-overlap / geometry-diversity / masked loss family
```

Required final behavior:

```text
1. Physical queries read temporal visual/point/tactile/posterior/cache support
   and remain task-neutral-ish.

2. Task queries read PaliGemma text/image, temporal visual, point/tactile,
   posterior/cache support, and relation context.

3. Same-role supports compete through Sinkhorn-like normalization.

4. Support distributions are kept by modality; only summaries are fused into
   anchor content.

5. Cross-modal alignment operates on modality-specific pooled embeddings,
   not raw support distributions forced to be identical.
```

Anti-collapse invariant:

```text
Do not solve collapse by geometry repulsion alone.
The main axes are:
  assignment/support competition
  support overlap control
  anchor pooled-embedding alignment
  posterior identity stability
Geometry diversity remains low-weight and confidence-gated.
```

Required tests:

```text
same-role rows are not identical when query seeds/support evidence differ
support row sums are one for active anchors
column concentration is bounded after Sinkhorn iterations
support-overlap loss ignores invalid/low-confidence modalities
point-missing path still produces valid visual/PG/posterior anchors
```

### 3.6.5 Posterior Address / Content Split

Current files:

```text
src/openpi/picf/core/contracts.py
src/openpi/picf/core/pipeline.py
```

Current posterior facts:

```text
_current_prior:
  previous posterior + proprio + previous action -> prior

_posterior_update:
  binding + measurement reread + precision-like fusion -> posterior

_innovation:
  current targets vs previous physical_prediction_cache
```

Required implementation:

```text
1. Add explicit address vectors to posterior state.
2. Keep address slow-moving across normal correction.
3. Allow address reset/update only through recycle/birth gates.
4. Use address in binding keys / cache lookup keys.
5. Use content in measurement values / action conditioning.
6. Keep existing h/c/tokens until all consumers are migrated.
```

Binding rule:

```text
Address is one compatibility term, not the whole identity solution.
Binding should combine address compatibility, predicted geometry compatibility,
current content/evidence similarity, role compatibility, and innovation/recycle
gates. Otherwise a wrong identity can become overly sticky.
```

Do not implement:

```text
address = RoPE(slot_index)
address = ordinary residual token overwritten every frame
address = task-conditioned semantic token
```

Required tests:

```text
address remains stable across low-innovation consecutive frames
content changes when measurement changes
recycle gate can intentionally reset address
semantic prompt changes do not rewrite physical addresses
identity switch rate does not increase against current AQR baseline
posterior carry preserves address/content
```

### 3.6.6 Posterior-Grounded Evidence Cache

Current files:

```text
src/openpi/picf/core/contracts.py
src/openpi/picf/core/pipeline.py
```

Current state:

```text
recurrent carry preserves posterior and physical_prediction_cache.
There is no full object-addressable evidence cache yet.
```

Required implementation:

```text
1. Add PicfEvidenceCacheState to recurrent carry.
2. Write cache entries after posterior correction, not before correction.
3. Store source/age/uncertainty/address/role/innovation metadata.
4. Let AQR and task readout read only the previous-carry cache as auxiliary
   typed memory.
5. Downweight cache when current innovation is high.
6. Clear/reset cache on reset_scaffold and segment boundary.
```

Hard rule:

```text
cache cannot bypass posterior into action.
cache is read as evidence/context; posterior remains current truth.
current-step cache write happens after posterior correction and is available
only to the next recurrent step.
```

Required tests:

```text
reset clears cache
age increments monotonically
innovation high reduces cache read gate
predicted-only cache entries cannot dominate current real observation
cache missing path matches current behavior
```

### 3.6.7 Slot-Level JEPA And Support Prediction

Current files:

```text
src/openpi/picf/core/pipeline.py
src/openpi/picf/core/training.py
scripts/picf_core_train.py
```

Current state:

```text
_build_physical_predictive_basis(...) produces world-only physical prediction.
_innovation(...) compares next current targets against physical_prediction_cache.
This is global/posterior-token level, not slot-level JEPA yet.
```

Required implementation:

```text
1. Predict next content for each physical slot.
2. Predict next geometry / uncertainty summary for each slot when valid.
3. Predict next support summaries per modality, not raw dense future images.
4. Use detached next posterior/target-encoder outputs as targets.
5. Match slots through address/binding; do not assume slot index equality
   before address stability is validated.
6. Add small-weight loss knobs:
   lambda_slot_jepa
   lambda_support_pred
   lambda_binding_consistency
   lambda_innovation_calibration
```

No-leakage invariant:

```text
Future targets are stop-gradient supervision only.
They cannot enter current AQR queries, posterior correction, task readout, or
PI0.5 action input.
```

Identity guard:

```text
Slot-JEPA can be implemented in the final code immediately, but its loss must
start disabled or very small until address/content diagnostics show stable
binding. Otherwise a slot-index mismatch can train false identity continuity.
```

Required tests:

```text
future target tensors are detached
current action logits are unchanged when future target values are perturbed
slot JEPA loss is finite on short unroll
binding target masks invalid/recycled slots
slot matching does not assume raw slot-index equality before identity stability
innovation calibration decreases trust in stale predictions
```

### 3.6.8 Ordinal / Relation Grounding

Required new components:

```text
language relation detector:
  identifies left/right/front/back/nearest/farthest/first/second/third/fourth

relation axis head:
  predicts ordering axis from language + camera/world context

soft-rank head:
  ranks physical slots along that axis

selector:
  maps task anchor to ranked physical slot / support
```

Implementation rule:

```text
Only activate relation loss on high-confidence relation language.
On non-relation tasks the head may produce features but its supervised loss is off.
The axis head must make the frame explicit: camera-left, world-left, and
robot-left are different labels unless calibration says they coincide.
Relation features cannot rewrite physical posterior identity.
```

Required tests:

```text
rank loss inactive on prompts without relation/ordinal words
soft-rank is differentiable and finite
axis changes under left/right/front/back prompt changes
camera/world/robot frame choices are logged in debug metrics
relation head cannot overwrite posterior identity
```

### 3.6.9 Debug / JSON / Video Acceptance

Every final component must be visible in diagnostics. Required JSON keys:

```text
aqr_temporal_support_entropy_mean
aqr_temporal_support_time_mass_t0
aqr_temporal_support_time_mass_t1
aqr_pg_support_entropy_mean
aqr_pg_support_max
aqr_effective_anchor_count
aqr_same_role_support_overlap_max
posterior_address_drift_mean
posterior_identity_switch_rate
posterior_recycle_rate
evidence_cache_trust_mean
evidence_cache_age_mean
innovation_norm_visual
innovation_norm_point
innovation_norm_tactile
slot_jepa_loss
support_pred_loss
ordinal_loss_active
```

Required visual exports:

```text
raw temporal V-JEPA support heatmap per recent time slice
temporal support overlay on RGB
PaliGemma image-token support raw map
PaliGemma image-token support overlay on RGB
point-neighborhood support projection
posterior address/identity trajectory video
cache trust over time plot
anchor health video with role labels and task text in filename
```

Acceptance cannot be based only on action loss. The final path must pass both:

```text
1. optimization evidence:
   action loss does not regress, AQR losses finite, no collapse diagnostics.

2. behavioral/evidence evidence:
   anchors/supports follow task-relevant objects over time, innovation corrects
   errors quickly, and cache does not lock onto stale hallucinated evidence.
```

### 3.6.10 Strict Scripted Contract Diagnosis

Manual README review is not sufficient. The final OWM path must also pass the
static README-to-code verifier:

```text
python scripts/verify_picf_owm_contract.py
python scripts/verify_picf_owm_contract.py --json
```

The verifier is intentionally strict and checks the architecture contract, not
just importability:

```text
README final definition of done and posterior authority
temporal V-JEPA support contracts and recent_maps time preservation
fixed tensor evidence cache with address/age/uncertainty/innovation metadata
graph fields for temporal priors, cache priors, address/content, uncertainty
pipeline temporal-token construction and AQR temporal priors
PaliGemma image support surviving as graph.pg_priors
cache causality: previous-cache read and post-correction write
debug keys for temporal, PG, posterior identity, cache trust, innovation, ordinal
next-posterior detached teacher targets for slot-JEPA/support prediction
all final OWM loss knobs exposed
trainer propagation of next posterior teacher and OWM debug metrics
evidence-bundle coverage of OWM loss/debug metrics and verifier status
```

The evidence bundle must include the verifier snapshot so a reviewer can audit
one run directory without re-reading the entire repository:

```text
python scripts/picf_owm_evidence_bundle.py --run-dir <run_dir>
```

The produced `owm_evidence_bundle.json` must include:

```text
contract_verifier.ok
contract_verifier.checks
args_owm
latest_owm_metrics
metrics_tail[*].owm
diagnostics[*].files
audit_rules.posterior_authoritative
audit_rules.cache_auxiliary_only
audit_rules.future_targets_are_loss_only
```

This scripted diagnosis is a necessary condition for long-run deployment. It is
not sufficient by itself: real CALVIN/robot evidence must still show stable
temporal support, posterior correction after innovation spikes, and no cache
lock-in.

## 3.7 Proposal Point-By-Point Resolution

This section is the strict resolution of the method text. Every proposal item is
classified as:

```text
ADOPT:
  belongs to the final architecture and must be implemented.

ADOPT WITH GUARD:
  belongs to the final architecture but must be gated by validity, confidence,
  no-leakage, or posterior stability conditions.

REJECT:
  must not be implemented because it breaks the posterior/PICF contract or
  creates an incoherent side path.
```

### 3.7.1 Section 0.1: Last Two Frames / V-JEPA

Resolution:

```text
ADOPT.
```

Current code:

```text
VjepaFeatureMap.current_map(...)
pipeline._visual_map(...)
```

Final code action:

```text
Add recent_maps(n), temporal visual support state, temporal AQR priors, and
temporal diagnostics. Keep last-two-mean only as an explicit ablation.
```

Reason:

```text
Temporal V-JEPA evidence is measurement evidence over recent time. Averaging it
before routing erases motion/contact timing and is not equivalent to JEPA-VLA.
```

### 3.7.2 Section 0.2: Existing Posterior Memory

Resolution:

```text
ADOPT.
```

Current code:

```text
PicfPosteriorAnchorState
_current_prior(...)
_posterior_update(...)
_innovation(...)
make_recurrent_carry(...)
```

Final code action:

```text
Preserve posterior as the authoritative belief state. Add evidence cache and
slot prediction around it, not instead of it.
```

Reason:

```text
The current PICF core is already a neural belief filter. Treating it as
memoryless would be a false diagnosis and would risk bypassing the best part of
the architecture.
```

### 3.7.3 Section 1: POMDP / Belief-State Reframing

Resolution:

```text
ADOPT.
```

Current code:

```text
prior -> observation anchors -> binding/correction -> innovation -> control
```

Final code action:

```text
Make the belief-filter structure explicit in state contracts, debug keys,
tests, and README terminology. Do not add independent world-model modules that
directly feed action without posterior correction.
```

Reason:

```text
Every final module has a belief-filter role:
  typed memory = measurement
  AQR = measurement routing
  posterior = belief
  slot JEPA = transition prediction
  innovation = prediction error
  cache = bounded historical evidence
  PI0.5 = action generator
```

### 3.7.4 Section 2: Address / Content / Geometry / Uncertainty

Resolution:

```text
ADOPT WITH GUARD.
```

Current code:

```text
posterior h/c/tokens look like content
posterior mu/Sigma/x/S/alpha/recycle look like belief geometry/existence
no explicit address vector exists
```

Final code action:

```text
Add explicit address and content fields. Keep address stable across normal
updates and update/reset it only through recycle/birth logic. Use address for
identity/binding/cache keys and content for measurement/action values.
```

Guard:

```text
Do not claim object identity just because a learned slot index exists. Address
must be validated by identity-switch diagnostics and posterior carry tests.
```

### 3.7.5 Section 3: RoPE / Type / Coordinate Encoding

Resolution:

```text
ADOPT.
```

Final code action:

```text
Use coordinate/time/view positional encodings for V-JEPA, points, tactile, and
language positions. Do not encode slot identity through RoPE distance. Slot
identity is address + binding history + role + posterior carry.
```

Reason:

```text
RoPE expresses relative coordinate structure. Slot identity should not have a
fake geometric distance induced by slot index.
```

### 3.7.6 Section 4.1: Typed Token Memory

Resolution:

```text
ADOPT.
```

Current code:

```text
_SemanticContext has text/image tokens.
AQR reads visual/point/tactile/posterior separately.
PG image support is currently converted mainly to visual-grid bias.
```

Final code action:

```text
Represent PaliGemma text, PaliGemma image tokens, recent V-JEPA temporal
tokens, point, tactile, proprio/action, posterior, and cache as typed support.
Do not reduce PaliGemma image support to a scalar V-JEPA grid bias.
```

Guard:

```text
Do not assume all modalities are conditionally independent. Fusion remains
learned and confidence-calibrated to avoid double-counting RGB-derived evidence.
```

### 3.7.7 Section 4.2: Physical Slots / Task Anchors

Resolution:

```text
ADOPT.
```

Current code:

```text
aqr_physical_query_tokens
aqr_task_query_tokens
task queries get semantic conditioning
```

Final code action:

```text
Freeze this as an architectural invariant. Physical slots maintain persistent
world/posterior state. Task anchors perform language-conditioned selection,
relation reasoning, and action-relevant readout.
```

Guard:

```text
Physical slots are task-neutral-ish, not task-blind. They may encode reusable
affordance/contact evidence but must not be overwritten by a transient prompt.
```

### 3.7.8 Section 4.3: AQR Measurement Routing

Resolution:

```text
ADOPT.
```

Current code:

```text
_build_aqr_anchor_graph(...)
_aqr_competitive_support(...)
_aqr_pg_image_support_read(...)
```

Final code action:

```text
Extend AQR to route over temporal V-JEPA, PG image, point, tactile, posterior,
and cache support. Preserve support distributions by modality and fuse only
their pooled summaries/content.
```

Guard:

```text
Cross-modal losses align same-slot pooled embeddings/summaries, not raw support
maps that naturally have different extents.
```

### 3.7.9 Section 4.4: Posterior Correction

Resolution:

```text
ADOPT AS HARD INVARIANT.
```

Final code action:

```text
All current evidence, cache evidence, and world-model prediction must pass
through posterior correction/innovation before affecting action. Any direct
cache-to-action or PG/V-JEPA-to-action bypass is invalid.
```

Reason:

```text
This is the mechanism that lets PICF recover quickly when an anchor/support is
wrong: prediction error raises innovation, current measurement gets higher
trust, stale cache is downweighted, and posterior rebinds/corrects.
```

### 3.7.10 Section 5: KV / Evidence Cache

Resolution:

```text
ADOPT WITH GUARD.
```

Final code action:

```text
Add a bounded posterior-grounded evidence cache with source, age, uncertainty,
slot address, role, modality validity, and innovation metadata.
```

Guard:

```text
Cache is never truth. Cache entries are historical evidence. Current posterior
after correction remains truth.
```

### 3.7.11 Section 6: Do Not Average V-JEPA Time

Resolution:

```text
ADOPT.
```

Final code action:

```text
Make last_two_tokens the default temporal support mode. Add last_two_mean only
as ablation. Add debug showing time-slice mass so motion/contact selection can
be inspected.
```

### 3.7.12 Section 7: Fine Grounding Without High-Resolution Crop

Resolution:

```text
ADOPT WITH INFORMATION LIMIT.
```

Final code action:

```text
Implement point-neighborhood refinement, temporal/posterior disambiguation, and
latent local top-k refinement. Do not promise sub-token visual identity when no
modality contains distinguishing evidence.
```

Reason:

```text
The architecture can reduce routing/identity error. It cannot create
information absent from RGB/V-JEPA/point/tactile/posterior evidence.
```

### 3.7.13 Section 8: Ordinal / Relation Grounding

Resolution:

```text
ADOPT WITH GUARD.
```

Final code action:

```text
Add language relation detector, relation axis head, soft-rank head, and
task-anchor selector. Activate supervised rank/relation losses only for
high-confidence explicit relation language.
```

Guard:

```text
Relation head can select/read physical slots. It cannot rewrite physical slot
identity or posterior address.
```

### 3.7.14 Section 9: Slot-Level JEPA World Model

Resolution:

```text
ADOPT WITH NO-LEAKAGE GUARD.
```

Final code action:

```text
Upgrade physical_prediction_cache into slot-level next-content/support/contact
prediction. Use detached next posterior/target encoder outputs as targets.
```

Guard:

```text
Future observations are targets only. They must not enter current action,
current AQR, current posterior correction, or current task readout.
```

### 3.7.15 Section 10: Heterogeneous Data / Scaling

Resolution:

```text
ADOPT.
```

Final code action:

```text
Every typed support branch must be optional and maskable. Train with modality
dropout over PG image, point, tactile, wrist/view, temporal V-JEPA, and cache
where the dataset permits.
```

Reason:

```text
Typed memory + small query routing scales better across heterogeneous robot
data than requiring all modalities to be dense-present in one all-to-all
transformer.
```

### 3.7.16 Section 11: Loss Family

Resolution:

```text
ADOPT WITH ACTIVATION GATES.
```

Final code action:

```text
Expose final loss knobs for slot JEPA, support prediction, binding consistency,
slot diversity, cross-modal pooled alignment, ordinal relation, innovation
calibration, and masked modality. Start with low/default-zero weights and turn
on only when required diagnostics exist.
```

Reason:

```text
The final architecture is complete only with these objectives available, but
mathematical coherence requires each loss to be valid-masked and not to train
against noisy or mismatched targets.
```

### 3.7.17 Section 12: Implementation Route

Resolution:

```text
ADOPT, BUT INTERPRET AS DIRECT IMPLEMENTATION GATES.
```

Final rule:

```text
Gates are not separate deployments. They are checks for adding the complete code
safely:
  state/interface -> forward wiring -> loss -> diagnostics -> tests -> cloud run.
```

Each gate must leave the branch in a trainable state with all existing AQR/PICF
contracts intact.

### 3.7.18 Section 13: Do Not Roll Back AQR/PICF

Resolution:

```text
ADOPT.
```

Final code action:

```text
Keep current AQR as skeleton. Replace missing typed-memory/posterior-world-model
pieces. Do not roll back to MAPG-v0 candidate-prior graph or PaliGemma heatmap
where-routing.
```

### 3.7.19 Section 14: Information-Theoretic Limit

Resolution:

```text
ADOPT AS HARD LIMIT.
```

Final documentation rule:

```text
Acceptance reports must distinguish architecture failure from observation
limit. If all available modalities cannot distinguish adjacent objects, no
router can guarantee correct identity.
```

### 3.7.20 Section 15: Final One-Sentence Architecture

Resolution:

```text
ADOPT AS CANONICAL DESCRIPTION.
```

Canonical description:

```text
PICF-AQR-OWM is a typed-memory, object-addressable, JEPA-predictive
belief-state architecture where PaliGemma text/image, recent V-JEPA temporal
tokens, point/tactile/proprio/action evidence, posterior slots, and bounded
cache are routed by physical/task AQR queries into persistent address-content
slots. Posterior correction is authoritative, slot prediction is leakage-free,
and PI0.5 remains the action generator.
```

## 3.8 What Must Not Be Implemented

These are explicit rejections. They are not optional ablations.

```text
1. Do not make PaliGemma heatmap the default where mechanism.
2. Do not replace posterior with a Transformer/KV cache.
3. Do not let cache feed action as current truth.
4. Do not average V-JEPA temporal evidence as the only production path.
5. Do not force tactile point contact support to match full visual object extent.
6. Do not encode slot identity as RoPE slot index distance.
7. Do not use future observations in current action input.
8. Do not bypass PI0.5 with a separate action head.
9. Do not claim sub-token adjacent-object identity without evidence in typed memory.
10. Do not solve anchor collapse only with geometry repulsion.
```

## 3.9 End-To-End Implementation Task List

The final implementation target is one complete architecture. The work can be
cut into reviewable commits for engineering control, but those commits are not
separate production deployments. A practical decomposition is:

```text
1. State/interface commit:
   add temporal visual support state, evidence cache state, address/content
   fields, graph support fields, config flags, and no-op compatibility paths.

2. V-JEPA temporal commit:
   implement recent_maps, temporal token flattening, AQR temporal read, temporal
   priors, and no-leakage tests.

3. PaliGemma image-support commit:
   populate graph.pg_priors, support all image ranges/views, add PG diagnostics,
   and keep heatmap disabled by default.

4. AQR support/routing commit:
   route physical/task queries over all typed supports, preserve modality
   priors, extend support competition, and update graph losses with masks.

5. Posterior address/content commit:
   add address/content split, update binding/cache keys, preserve carry
   compatibility, and add identity-stability diagnostics.

6. Evidence cache commit:
   add bounded cache writes after posterior correction, cache reads as auxiliary
   typed memory, innovation-gated cache trust, reset behavior, and tests.

7. Slot JEPA/support prediction commit:
   add slot next-content/support prediction, detached targets, matching masks,
   loss knobs, and no-leakage invariance tests.

8. Ordinal/relation commit:
   add relation language detector, axis/rank heads, high-confidence weak loss,
   selector diagnostics, and non-relation no-op tests.

9. Debug/eval commit:
   add JSON keys, heatmap overlays, temporal support videos, PG support videos,
   posterior identity trajectory videos, and CALVIN evidence bundle scripts.

10. Cloud deployment commit:
   update README_v2.2, train/eval commands, tmux launch templates, checkpoint
   cadence, and acceptance checklist.
```

Every commit must keep these invariants passing:

```text
python -m py_compile src/openpi/picf/core/contracts.py
python -m py_compile src/openpi/picf/core/config.py
python -m py_compile src/openpi/picf/core/pipeline.py
python -m py_compile src/openpi/picf/core/training.py
python -m py_compile scripts/picf_core_train.py scripts/serve_picf_policy.py
git diff --check
targeted pipeline/training tests for changed contracts
```

## 3.10 File-By-File Code Audit And Deployment Map

This section is the concrete audit bridge from `docs/temp_method.md` to the
current repository. The source method is accepted as the architecture proposal,
but this README is the implementation contract: every accepted idea must land in
the files below with the stated invariants.

Status note:

```text
The "Current facts" blocks below record the pre-deployment audit findings that
motivated each change. The deployed branch is accepted only by the scripted
contract verifier and the status ledger, not by these historical findings.
```

### 3.10.1 `src/openpi/picf/core/contracts.py`

Current facts:

```text
PicfAnchorPriorGraphState:
  has pg_priors but AQR currently returns pg_priors=None
  has visual_priors / point_priors / tactile_priors / posterior_priors
  has anchor_tokens / roles / confidence / geometry
  lacks vjepa_temporal_priors, cache_priors, slot_address, slot_content,
  support_uncertainty

PicfTokenFieldState:
  has visual_tokens as one flattened 2D visual map
  lacks temporal_visual support state

PicfPosteriorAnchorState:
  has h/c/mu/Sigma/x/S/a/alpha/contact/support/binding/evidence/tokens
  lacks explicit address/content fields

PicfPredictiveState:
  has physical_prediction_cache and prediction_cache
  lacks slot_prediction_tokens, slot_prediction_supports, evidence_cache

PicfRecurrentCarryState:
  carries posterior and physical_prediction_cache
  lacks bounded object-addressable evidence cache
```

Final deployment changes:

```text
1. Add PicfTemporalVisualSupportState.
2. Add PicfEvidenceCacheState as a fixed-size recurrent ring buffer.
3. Extend PicfTokenFieldState with temporal_visual.
4. Extend PicfAnchorPriorGraphState with temporal/cache/address/content fields.
5. Extend PicfPosteriorAnchorState with address/content while preserving h/c/tokens.
6. Extend PicfPredictiveState and PicfRecurrentCarryState with evidence_cache.
```

Correctness checks:

```text
contract construction works when new fields are absent or None
old checkpoints can still instantiate current state with compatibility defaults
AQR path can expose pg_priors and vjepa_temporal_priors without breaking
observation anchors, task readout, posterior update, or control state
```

Why this is required:

```text
The source method defines anchors as belief-state objects, not keypoints.
Contracts must therefore expose the objects the math depends on:
typed temporal support, PG image support, address/content identity, cache, and
slot-level prediction. If these remain implicit, the implementation will drift
back into a candidate-prior patch rather than becoming OWM.
```

### 3.10.2 `src/openpi/picf/vjepa/wrapper.py`

Current facts:

```text
VjepaFeatureMap.current_map(use_last_two_mean=False):
  returns one [H, W, C] latent map

VjepaFeatureMap.current_map(use_last_two_mean=True):
  averages the last two latent temporal slices

No method returns recent temporal slices as separate support tokens.
```

Final deployment changes:

```text
1. Add VjepaFeatureMap.recent_maps(n=2) -> [T_recent, H, W, C].
2. Preserve temporal order from older to newer or explicitly document the order.
3. Never average inside recent_maps.
4. Keep current_map as a legacy/ablation API only.
```

Correctness checks:

```text
recent_maps(2)[-1] equals the current latest latent slice
recent_maps(2) contains two distinct slices when the clip has temporal motion
current_map(use_last_two_mean=True) is not used by the production AQR path
```

Why this is required:

```text
The final architecture needs temporal measurement evidence. Averaging time
before AQR can smear or erase motion/contact timing and can weaken posterior
correction or slot-level prediction at the required fidelity.
```

### 3.10.3 `src/openpi/picf/vjepa/config.py`

Current facts:

```text
num_frames=64
tubelet_size=2
use_last_two_mean=False
```

Final deployment changes:

```text
1. Add a production temporal support mode, e.g. aqr_vjepa_temporal_mode.
2. Make last_two_tokens the default for OWM.
3. Keep last_two_mean only as an explicit ablation value.
4. Expose T_recent as a config value, default 2, with optional last4_tokens.
```

Correctness checks:

```text
startup logs include temporal mode and T_recent
training config cannot silently use last_two_mean as the OWM production path
```

### 3.10.4 `src/openpi/picf/core/config.py`

Current facts:

```text
aqr_temporal_memory_tokens exists
the current pipeline does not actually consume it as temporal V-JEPA support
```

Final deployment changes:

```text
1. Add explicit flags for temporal V-JEPA support:
   aqr_vjepa_temporal_mode
   aqr_vjepa_recent_frames
   lambda_slot_jepa
   lambda_support_pred

2. Add explicit flags for evidence cache:
   evidence_cache_enabled
   evidence_cache_len
   evidence_cache_read_weight
   evidence_cache_innovation_downweight

3. Add explicit flags for ordinal/relation:
   ordinal_relation_enabled
   lambda_ordinal_relation
   ordinal_confidence_threshold

4. Keep legacy MAPG and PG heatmap flags separate:
   mapg_enabled must stay off for OWM
   aqr_pg_grounding_enabled must stay off unless running a labeled ablation
```

Correctness checks:

```text
configuration names distinguish temporal support, cache, prediction, relation,
and heatmap ablation clearly enough that training commands cannot accidentally
turn a non-OWM path into the main run
```

### 3.10.5 `scripts/picf_core_train.py`

Current facts:

```text
The script exposes AQR/MAPG knobs including aqr_temporal_memory_tokens.
It validates that aqr_mapg_enabled requires PICF and PaliGemma semantic mode.
It rejects enabling legacy mapg_enabled and aqr_mapg_enabled together.
It has tqdm/progress infrastructure and save_interval plumbing.
```

Final deployment changes:

```text
1. Add CLI args for every new OWM config flag.
2. Print the OWM mode summary at startup:
   temporal V-JEPA mode
   PG image support on/off
   PG heatmap off/on
   evidence cache on/off
   slot JEPA on/off
   support prediction on/off
   ordinal relation on/off

3. Add metric accumulation for:
   slot_jepa_loss
   support_pred_loss
   binding_consistency_loss
   innovation_calibration_loss
   ordinal_relation_loss
   temporal support entropy/mass
   cache trust/age

4. Keep save interval and tmux deployment operationally unchanged.
```

Correctness checks:

```text
--aqr-mapg-enabled cannot be combined with --mapg-enabled
--aqr-pg-grounding-enabled remains false in canonical OWM commands
progress bar shows step, total/action loss, OWM losses, save interval, and ckpt path
save_interval can be set to 2500 or 5000 by command without code edits
```

### 3.10.6 `src/openpi/picf/core/pipeline.py`: Visual Entry

Current facts:

```text
_visual_map(...):
  returns one 2D V-JEPA map through current_map(...)

observe_step(...):
  builds visual_map once
  passes visual_map into _build_token_field(...)
  uses visual_map for current_targets and downstream PICF state
```

Final deployment changes:

```text
1. Add _visual_temporal_maps(...).
2. In observe_step, build both:
   visual_map for legacy compatibility/current targets
   temporal_visual_maps for OWM support memory

3. Pass temporal_visual_maps into _build_token_field.
4. Store flattened temporal support in token_field.temporal_visual.
5. Ensure future target maps used by training are never passed into current
   AQR, posterior, task readout, or control.
```

Correctness checks:

```text
with OWM enabled, AQR sees temporal_visual tokens
with temporal support disabled, current behavior is unchanged
perturbing future target tensors does not change current action output
```

### 3.10.7 `src/openpi/picf/core/pipeline.py`: AQR Query Construction

Current facts:

```text
aqr_physical_query_tokens and aqr_task_query_tokens already exist.
physical and task queries already receive role/type/coverage embeddings.
task queries already receive semantic conditioning through aqr_task_conditioner.
previous posterior summary can already condition the query set.
```

Final deployment changes:

```text
1. Keep physical/task split as the OWM backbone.
2. Add address-conditioned physical query inputs once posterior addresses exist.
3. Let physical queries read temporal visual/point/tactile/posterior/cache.
4. Let task queries read text/PG image/temporal visual/point/tactile/posterior/cache.
5. Do not make task-conditioned query output overwrite physical posterior identity.
```

Correctness checks:

```text
physical query outputs remain present when prompt text changes
task query outputs change with prompt semantics
posterior address diagnostics do not spike on prompt-only changes
```

### 3.10.8 `src/openpi/picf/core/pipeline.py`: PaliGemma Image Support

Current facts:

```text
_aqr_pg_image_support_read(...):
  only reads semantic.image_token_ranges[0]
  only uses semantic.image_view_transforms[0]
  returns updated queries plus a V-JEPA visual-grid bias

_build_aqr_anchor_graph(...):
  calls _aqr_pg_image_support_read(...)
  still returns pg_priors=None
```

Final deployment changes:

```text
1. Make _aqr_pg_image_support_read return:
   updated_q
   pg_priors
   pg_visual_bias
   pg_view_ids or masks

2. Iterate all semantic.image_token_ranges.
3. Preserve PG priors before remapping to V-JEPA grid.
4. Fill PicfAnchorPriorGraphState.pg_priors.
5. Keep PG image support independent from PG heatmap grounding.
```

Correctness checks:

```text
PG image support enabled + image tokens present -> graph.pg_priors is non-None
multiple image ranges produce nonzero support in all valid ranges
turning off aqr_pg_grounding_enabled does not remove PG image-token support
```

### 3.10.9 `src/openpi/picf/core/pipeline.py`: AQR Support Routing

Current facts:

```text
_aqr_competitive_support(...) performs top-k pruning and Sinkhorn-like
normalization over current modality weights.

_build_aqr_anchor_graph(...) currently reads visual, point, tactile, and
posterior supports, then stores modality priors and anchor tokens.
```

Final deployment changes:

```text
1. Add temporal visual reader path before or beside current visual reader.
2. Add cache reader path after posterior correction/cache state exists.
3. Keep all modality priors separate.
4. Derive legacy visual_priors from temporal priors only for backward
   compatibility.
5. Extend modality_confidence to distinguish visual_temporal and PG image
   support, or add separate confidence fields.
```

Correctness checks:

```text
same-role support overlap is bounded
same-role rows are not identical when evidence/query seeds differ
temporal prior rows sum to one for active anchors
legacy visual_priors remain available for existing consumers
```

### 3.10.10 `src/openpi/picf/core/pipeline.py`: Posterior Prior And Correction

Current facts:

```text
_current_prior(...):
  previous posterior + proprio + previous action -> h/c/mu/var/x/S/a/alpha prior

_posterior_update(...):
  builds binding logits from hidden and geometry
  adds role/VL/graph biases
  uses Sinkhorn dustbin/recycle
  rereads obs/native visual/tactile evidence
  performs precision-like multi-vote fusion
  writes posterior h/c/tokens/global_post

_innovation(...):
  compares current targets against previous physical_prediction_cache
```

Final deployment changes:

```text
1. Preserve this posterior pipeline as authoritative belief.
2. Add address/content split to prior, binding, write, and carry.
3. Make cache writes happen only after posterior correction.
4. Make slot-level prediction use posterior slots as state, not raw AQR anchors.
5. Make innovation control cache trust and posterior correction gates.
```

Correctness checks:

```text
posterior update still runs without cache
high innovation does not cause stale cache to dominate correction
address remains stable under low innovation
recycle explicitly resets address/content when needed
```

### 3.10.11 `src/openpi/picf/core/pipeline.py`: Predictive Basis And No-Leakage

Current facts:

```text
_build_physical_predictive_basis(...):
  uses posterior.global_post, proprio, executed_action
  produces physical_pred_tokens, physical_global_pred, physical_prediction_cache

_build_conditioned_predictive_cache(...):
  adds future condition tokens for semantic-conditioned prediction

_innovation(...):
  intentionally compares against world-only physical_prediction_cache
```

Final deployment changes:

```text
1. Keep world-only physical prediction as the innovation reference.
2. Add slot_prediction_tokens from posterior slot tokens.
3. Add support prediction heads over detached future support summaries.
4. Do not allow future condition tokens or future target encodings into current
   AQR, posterior, task readout, conditioned control, or PI0.5.
```

Correctness checks:

```text
changing future targets changes training loss but not current action output
slot prediction loss is finite on short unrolls
innovation norms still use world-only prediction, not task-conditioned future
```

### 3.10.12 `src/openpi/picf/core/training.py`

Current facts:

```text
The training loss already includes action, VL losses, MAPG/AQR graph loss,
SigLIP, VICReg, cycle, masked-modality, routing, support diversity, geometry
diversity, and alignment budget scaling.

There is no explicit slot JEPA loss, support prediction loss,
binding-consistency loss, innovation-calibration loss, or ordinal relation loss.
```

Final deployment changes:

```text
1. Add masked loss terms:
   slot_jepa
   support_pred
   binding_consistency
   innovation_calibration
   ordinal_relation

2. Use validity masks for every new loss.
3. Cross-modal alignment must use modality-specific pooled summaries, not raw
   support distributions forced to match.
4. Budget auxiliary losses against action loss to prevent router objectives from
   overwhelming behavior learning.
```

Correctness checks:

```text
all new losses are zero/finite when branch is disabled or modality missing
non-ordinal prompts keep ordinal loss inactive
masked modalities do not create fake positive pairs
auxiliary loss budget scaling reports finite scales
```

### 3.10.13 Debug, JSON, CALVIN, And Evidence Bundle Code

Current requirement:

```text
Final OWM cannot be accepted from scalar training loss alone.
```

Final deployment changes:

```text
1. Extend JSON metrics with temporal support, PG support, posterior identity,
   cache trust, innovation, slot JEPA, support prediction, and relation flags.
2. Extend CALVIN image/video export with:
   raw temporal support maps
   temporal overlays on RGB
   PG support raw/overlay views
   point-neighborhood projection
   posterior identity trajectory
   cache trust timeline
   anchor health video with task text and step in filename
3. Add an evidence-bundle command that writes a compact 10-50 MB handoff bundle.
```

Correctness checks:

```text
debug export works when point/tactile are missing
file names include episode, step, task text, and support branch
JSON and videos can distinguish architecture failure from observation limit
```

## 3.11 Final Definition Of Done

The final PICF-AQR-OWM deployment is not complete when the README exists. It is
complete only when the code satisfies all conditions below:

```text
1. contracts expose all final states
2. V-JEPA recent temporal support is wired into token_field and AQR
3. PG image-token support is populated as graph.pg_priors
4. physical/task query split remains the production router
5. posterior address/content are carried recurrently
6. evidence cache is written after posterior correction and read only as evidence
7. slot-level prediction and support prediction are no-leakage target losses
8. ordinal/relation head is gated and cannot rewrite posterior identity
9. JSON/video diagnostics expose every branch
10. PI0.5 action path remains the final action generator
```

The minimum verification set before launching a long run is:

```text
git diff --check
python -m py_compile src/openpi/picf/core/contracts.py
python -m py_compile src/openpi/picf/core/config.py
python -m py_compile src/openpi/picf/core/pipeline.py
python -m py_compile src/openpi/picf/core/training.py
python -m py_compile scripts/picf_core_train.py
python -m py_compile scripts/verify_picf_owm_contract.py
python -m py_compile scripts/picf_owm_evidence_bundle.py
python -m py_compile scripts/serve_picf_policy.py
python scripts/verify_picf_owm_contract.py
targeted unit tests for changed contracts/readers/losses
short AQR forward smoke run
short CALVIN debug export with JSON and support videos
python scripts/picf_owm_evidence_bundle.py --run-dir <short_debug_run_dir>
```

Canonical long-run launch must preserve:

```text
legacy mapg_enabled=false
aqr_mapg_enabled=true
aqr_pg_grounding_enabled=false unless ablation
V-JEPA/Sonata/AnyTouch pretrained weights frozen under frozen-perception profile
PaliGemma semantic/image/action adapters trainable according to the current profile
AQR/PICF router/posterior/cache/prediction/relation components trainable
save interval set by command, e.g. 2500 or 5000
tmux run with visible progress bar and log tail path recorded
```

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
PicfEvidenceCacheState fixed ring buffer:
  tokens: [H, K, D]
  source: [H, K]
  slot_address: [H, K, D_addr]
  role: [H, K]
  age: [H, K]
  uncertainty: [H, K]
  innovation_at_write: [H, K]
  modality_validity: [H, K, M]
  support summaries: fixed tensor fields or masked optional tensors
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

Causal order:

```text
previous cache -> AQR/task read as weak evidence
current evidence -> posterior correction
corrected posterior -> write cache for the next step
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

Guarded activation policy for the direct final implementation:

```text
Core path:
  current AQR losses + first-class PG/V-JEPA temporal support diagnostics

Identity path:
  binding consistency + address/content diagnostics

Prediction path:
  slot JEPA next-token/content prediction

Support prediction path:
  support prediction

Relation path:
  ordinal/relation loss for high-confidence language only
```

The final architecture is implemented as one complete target. The gates above
are runtime/training guards, not separate reduced deployments. They prevent
future-target, cache, identity, and weak-relation objectives from becoming
optimization noise while still keeping the final modules present and testable.

## 17. Direct Full Deployment Plan

This section is the concrete full deployment plan. Every item belongs to the
final architecture. The numbered gates are review and acceptance gates inside a
direct-to-final implementation, not a recommendation to stop at intermediate
architectures.

### Gate 1: Temporal V-JEPA Typed Support

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

### Gate 2: First-Class PaliGemma Image Support

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

### Gate 3: Posterior Address / Content Split

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
prompt-only change stability test
recycle reset test
identity switch metric on CALVIN debug rollouts
```

Acceptance:

```text
address does not drift every frame
content updates with measurement
recycle explicitly resets/changes address when needed
```

### Gate 4: Posterior-Grounded Evidence Cache

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

### Gate 5: Slot-Level JEPA Prediction

Implement:

```text
slot next-content predictor
detached next-posterior target path
matching pi(j) from posterior binding/address
slot_jepa loss with disabled-or-small initial weight until identity diagnostics pass
no future input leakage into action path
```

Tests:

```text
teacher target detached
student path current/past only
loss finite on short unroll
slot matching mask rejects unstable/recycled identities
```

Acceptance:

```text
innovation calibration improves
posterior correction remains active
action loss does not destabilize
```

### Gate 6: Support Prediction

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

### Gate 7: Ordinal / Relation Head

Implement:

```text
language ordinal parser / detector
relation axis head
soft rank / pairwise relation logits
high-confidence weak supervision only
explicit camera/world/robot frame selection
```

Acceptance:

```text
rank loss active only on explicit relation language
no degradation on non-ordinal tasks
relation head cannot overwrite posterior address/content identity
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

This is a complete final contract. It is not a minimal patch. The implementation
should go directly to the full OWM target, while keeping training/runtime gates
for posterior identity, temporal support, cache trust, and future prediction.
Enabling every new loss at full strength without those guards would be
mathematically less coherent, not more complete.

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
