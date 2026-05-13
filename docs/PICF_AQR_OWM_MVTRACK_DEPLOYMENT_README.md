# PICF-AQR-OWM-MVTrack Deployment Contract

Version: `v2.0-runtime-c`
Date: 2026-05-10
Status: code-level MVTrack runtime complete; behavior acceptance still pending fresh run evidence.

This document is linked from `src/openpi/picf/README_v2.2.md`. It is the
canonical design contract for the next version after the current maintained
PICF-AQR-OWM baseline.

## 0. Scope

PICF-AQR-OWM v26 is the maintained code-level baseline:

```text
AQR typed routing is default.
Legacy MAPG and VL heatmap routing are off by default.
PaliGemma image support is first-class graph.pg_priors.
V-JEPA recent temporal tokens are first-class graph.vjepa_temporal_priors.
Evidence cache is posterior-grounded, residual-scaled, and skips the newest
posterior duplicate.
State-only burn-in uses the same AQR measurement graph as the suffix path.
PI0.5 action generation remains unchanged.
```

MVTrack is the next architecture contract:

```text
PICF-AQR-OWM-MVTrack:
  Multiview-Tracklet Object-Addressable Predictive Belief Router
```

It upgrades the current baseline without replacing its core invariant:

```text
posterior is authoritative current belief.
cache is auxiliary historical evidence, not truth.
AQR is the typed measurement router.
PI0.5 remains the final action generator.
future latents are detached targets only.
physical predictive-cache innovation is diagnostic/cache metadata and must not
directly alter posterior identity; posterior identity gating uses current
measurement innovation from prior geometry vs observation anchors.
```

This document is not allowed to claim behavior-level completion until the code,
verification scripts, metrics, and CALVIN/video evidence satisfy Section 16.

2026-05-10 runtime-c pass:

```text
Implemented in code:
  static+wrist V-JEPA typed temporal views
  optional train/replay/serve threading for tracklet/proposal episode fields
  tracklet typed support state and AQR reader when optional tensors are present
  support-signature identity binding
  posterior-slot-address-first cache query addressing with learned-query fallback
  measurement-innovation-gated support/address inertia and gated slot_address update
  address/content/role-aware cache attention bias with residual scaling retained
  archived top-k latent local refinement over existing typed memory, disabled by default
  optional pseudo-proposal typed memory and AQR proposal reader
  training-only support denoising auxiliary, default weight 0
  matched slot-JEPA/support prediction targets
  permutation-tolerant binding consistency temporal term
  weak ordinal rank/selection diagnostics
  anchor-only large-batch diagnostic trainability scope

Still external-data dependent:
  tracklet tensors require an upstream offline tracker/preprocessor source.
  proposal tensors require an upstream detector/proposal source; SAM/DINO is not required by the maintained runtime.
  CALVIN/video behavior acceptance requires a new run on this checkout.
```

2026-05-13 cleanup update:

```text
Local refinement is no longer part of the production-default MVTrack profile.
It is retained only as a legacy ablation path requiring explicit
legacy_local_refinement_opt_in plus positive top-k and residual weight.

Default:
  legacy_local_refinement_opt_in=false
  local_refinement_enabled=false
  local_refinement_topk=0
  local_refinement_weight=0.0

Reason:
  normalized recycle input fixed the recycle/address saturation chain, while
  local top-k reread was not necessary for early non-collapse and introduced
  extra recycle/gradient pressure in the A5/A7 attribution matrix.
```

Recycle trust-gate normalization:

```text
Default:
  recycle_normalize_residual_summary=true
  recycle_residual_norm_mode=layernorm

Ablation:
  recycle_residual_norm_mode=rmsnorm

Diagnostic only:
  recycle_residual_norm_mode=none
```

LayerNorm removes the residual norm shortcut that previously saturated recycle.
RMSNorm preserves the residual mean/DC component and is useful as a conservative
ablation. Quantile normalization is not a production forward-path option because
it is non-causal unless estimated from history, can be batch/distribution
dependent, and can saturate extreme residual evidence into rank boundaries.

2026-05-14 ownership-prior update:

```text
Default:
  aqr_ownership_prior_enabled=true
  aqr_ownership_prior_weight=0.35
  aqr_ownership_temporal_prior_weight=0.20
  aqr_ownership_prior_uniform_mix=0.05
```

Reason:
  A5/A7 task-pressure warmup showed that schedule-only fixes do not solve
  same-role ownership. The failure mode is mathematical: if two same-role AQR
  rows enter Sinkhorn with identical support logits, Sinkhorn preserves that
  equality and cannot create object ownership. The maintained fix is a
  low-amplitude role-local coverage prior added directly to visual/temporal
  support logits before AQR reads memory. This is an assignment prior, not a
  new loss or a local-refinement residual. It seeds distinct object ownership
  while preserving evidence dominance and then lets support-signature binding
  and posterior correction stabilize the result.

2026-05-14 active/dustbin capacity update:

```text
Default:
  aqr_active_slot_filter_enabled=true
  aqr_active_slot_min_per_role=1
  aqr_active_slot_max_per_role=4
  aqr_active_slot_min_confidence=0.05
  aqr_active_slot_overlap_threshold=0.75
```

Reason:
  The ownership prior fixes initial symmetry but not late collapse. The
  remaining failure is capacity mismatch: a fixed physical slot budget can be
  larger than the number of useful CALVIN scene objects, so redundant same-role
  slots are forced toward the same high-salience support. The maintained fix is
  to classify anchors into active object candidates and inactive/dustbin
  candidates before observation/task assignment. Active anchors remain subject
  to same-role diversity; inactive anchors remain recurrent/query carriers but
  are not forced to bind duplicated objects. This preserves posterior authority
  and the existing typed-evidence router while adding the missing effective-slot
  capacity control. It is not a new loss and not a hard dataset-specific object
  label.

New acceptance metrics:

```text
aqr_active_anchor_count
aqr_inactive_anchor_fraction
aqr_active_same_role_support_overlap_max
aqr_active_same_role_support_overlap_mean
aqr_active_anchor_count_role_0
aqr_active_anchor_count_role_1
aqr_active_anchor_count_role_2
aqr_active_anchor_count_role_3
```

Interpretation:
  Raw `aqr_same_role_support_overlap_max` remains useful for diagnosis, but the
  active subset is the behavioral contract. Inactive/dustbin anchors may overlap
  by design. A run fails this repair if active overlap returns to the old
  0.95-0.99 collapse band, active count collapses to one, or action/recycle
  health deteriorates while active overlap improves.

Anchor-only probe contract:

```text
scripts/picf_core_train.py
  --picf-trainable-scope all|anchor_only

all:
  normal maintained training profile.

anchor_only:
  diagnostic profile only. It freezes perception, semantic, PI0.5
  action/control, and predictive heads after lazy parameter materialization.
  It leaves typed-evidence token adapters, AQR/MVTrack readers,
  observation-anchor adapters, posterior binding/address, and
  support/cache/local evidence modules trainable.
  It also forces semantic runtime to no-grad/inference mode and disables
  window-level activation checkpointing so the large-batch probe spends memory
  and time on anchor routing rather than policy-stack gradients/recompute.
  Optional MVTrack typed-memory adapters, including tracklet and proposal token
  projectors, are still materialized during warmup even when the dataset does
  not provide those modalities. This keeps FSDP/DDP strict lazy-parameter
  checks aligned with the maintained v2 contract instead of silently depending
  on modality presence.
  Under FSDP full-shard, fully frozen root modules are expanded to their
  parameters and combined with frozen root-managed parameters through the
  single `ignored_states` API for this scope. That preserves
  `use_orig_params=False` flat-parameter uniformity without broadening the
  trainable anchor allowlist.

Purpose:
  maximize safe batch/accumulation on 2x40GB and test whether anchors can
  separate, specialize, and maintain identity under frozen evidence.

Not a claim:
  anchor_only is not final policy training and should not be judged by action
  loss alone.
```

## 1. Current Code Audit

The current code already implements the following MVTrack prerequisites.

### 1.1 AQR is the maintained graph path

Code facts:

```text
src/openpi/picf/core/config.py
  aqr_mapg_enabled = True
  mapg_enabled = False
  vl_anchor_router_enabled = False
```

Training entry facts:

```text
scripts/picf_core_train.py
  semantic_mode defaults to paligemma.
  loss defaults are sourced from PicfTransitionLossConfig.
  legacy MAPG and direct AQR cannot both be production paths.
```

### 1.2 V-JEPA temporal support is first-class

Code facts:

```text
src/openpi/picf/vjepa/wrapper.py
  VjepaFeatureMap.recent_maps(n)
    returns recent temporal latent maps without averaging time.

src/openpi/picf/core/pipeline.py
  _visual_maps(...)
    returns current visual map and temporal visual maps.

  _build_token_field(...)
    builds PicfTemporalVisualSupportState.

  _build_aqr_anchor_graph(...)
    reads temporal_visual tokens through aqr_temporal_visual_reader.
    writes graph.vjepa_temporal_priors.
```

Updated MVTrack code facts:

```text
src/openpi/picf/core/pipeline.py
  keeps per-view V-JEPA clip buffers.
  encodes static and gripper/wrist RGB when available.
  emits temporal_visual.view_ids.
  does not project wrist evidence into static-camera geometry without
  calibrated wrist extrinsics.
```

### 1.3 PaliGemma image support is first-class

Code facts:

```text
src/openpi/picf/core/pipeline.py
  _aqr_pg_image_support_read(...)
    iterates semantic.image_token_ranges.
    reads PaliGemma image tokens directly with task anchors.
    returns pg_priors.
    may also produce visual-grid bias.

  _build_aqr_anchor_graph(...)
    returns PicfAnchorPriorGraphState.pg_priors.
```

This is no longer:

```text
PaliGemma image -> V-JEPA bias -> downstream only.
```

The current dataflow is:

```text
PaliGemma text -> task query semantic conditioning.
PaliGemma image -> first-class PG image support branch.
V-JEPA current -> current visual support branch.
V-JEPA recent -> temporal visual support branch.
point/tactile/posterior/cache -> separate typed support branches.
```

### 1.4 Evidence cache is posterior-address aware and residual-gated

Code facts:

```text
src/openpi/picf/core/pipeline.py
  _previous_evidence_cache_tokens(...)
    reads previous carry only.
    skips newest posterior cache row.
    applies age, uncertainty, innovation, source, and coarse role gates.
    returns PicfCacheReadState with token, address, role, source, age,
    uncertainty, innovation, modality-validity metadata.

  _build_aqr_anchor_graph(...)
    forms cache query addresses from live previous.posterior.slot_address for
    physical slots, falling back to learned query carriers when no posterior
    address exists.
    adds address/content/role terms to the cache attention bias.
    applies evidence_cache_read_weight as residual scale:
      q <- q + lambda_cache * (ReadCache(q) - q)

  _write_evidence_cache(...)
    writes after posterior correction.
```

Remaining limitation:

```text
cache source_ids are still simple because the maintained cache stores
posterior-grounded belief rows. Rich source enums only matter if future code
also writes observation-evidence, predicted, tracklet, or proposal rows into
the cache. The current cache should therefore be described as posterior-address
aware episodic belief memory, not as a multi-source database.
```

### 1.5 Posterior remains the belief filter

Code facts:

```text
src/openpi/picf/core/pipeline.py
  _current_prior(...)
    predicts current prior from previous posterior, proprio, and action.

  _posterior_update(...)
    computes binding.
    rereads current evidence.
    performs precision-style prior + measurement correction.

  _innovation(...)
    compares current targets with previous world-only prediction cache.
```

The posterior update preserves:

```math
Lambda_t^+ = Lambda_t^- + Lambda_t^meas
eta_t^+ = eta_t^- + eta_t^meas
mu_t^+ = (Lambda_t^+)^{-1} eta_t^+
```

### 1.6 Guarded OWM losses are not production pressure

Current defaults:

```text
lambda_slot_jepa = 0.0
lambda_support_pred = 0.0
lambda_binding_consistency = 0.0
```

These hooks exist, but they must not be enabled until matched targets and
identity diagnostics are stable.

### 1.7 Ordinal is diagnostic only

Current facts:

```text
ordinal_relation_enabled = True
ordinal state can detect relation/ordinal prompts.
ordinal state cannot rewrite posterior.
no rank target or ordinal loss is active.
```

This is safe, but it does not solve "fourth from left".

## 2. Method Verdict

The MVTrack plan is compatible with current PICF-AQR-OWM because every new
piece belongs to one of three belief-state categories:

```text
evidence:
  static/wrist V-JEPA, tracklets, proposal memory, local typed rereads.

correction:
  support-signature binding, gated address compatibility, address-aware cache.

training-only teacher:
  support denoising auxiliary, matched slot-JEPA, support prediction,
  weak ordinal loss.
```

The plan is rejected if it violates any invariant:

```text
1. It cannot bypass posterior correction.
2. It cannot make cache a source of current truth.
3. It cannot put future observations in the current action path.
4. It cannot let ordinal/relation overwrite posterior identity.
5. It cannot use wrist geometry as static geometry truth without calibration.
6. It cannot claim sub-token instance recovery without evidence.
```

## 3. Mathematical Contract

The robot faces a partially observable state:

```math
s_t =
{object states, robot state, contact state, task-relevant latent state}
```

Observation:

```math
o_t =
{RGB_static, RGB_wrist, point, tactile, proprio, language, a_{t-1}}
```

The policy should act on belief:

```math
b_t(s) = p(s_t=s | o_{\le t}, a_{<t}, language)
```

The update is:

```math
b_t(s)
propto
p(o_t | s_t)
int p(s_t | s_{t-1}, a_{t-1}) b_{t-1}(s_{t-1}) ds_{t-1}
```

PICF-AQR-OWM-MVTrack maps this to:

```text
typed evidence memory
  -> AQR measurement routing
  -> posterior binding/correction
  -> innovation-aware cache and prediction
  -> PI0.5 action generation
```

No module is allowed to sit outside this loop.

## 4. Slot State

Each physical slot is:

```math
S_{t,j}
=
(a_{t,j}, c_{t,j}, mu_{t,j}, Sigma_{t,j}, alpha_{t,j}, r_j, sigma_{t,j})
```

Definitions:

```text
a_tj:
  address / identity key.

c_tj:
  time-varying content.

mu_tj, Sigma_tj:
  geometry belief and uncertainty.

alpha_tj:
  existence / visibility / support confidence.

r_j:
  role/type.

sigma_tj:
  support signature over visual, temporal, point, PG image, tactile,
  tracklet, and cache supports.
```

Address is not a hard identity. It is a prior that must be gated by current
evidence, support overlap, geometry, innovation, and recycle.

## 5. Typed Memory

AQR reads:

```text
M_text:
  PaliGemma semantic/text tokens.

M_pg_img[v]:
  PaliGemma image tokens per available view.

M_vjepa[v, tau, h, w]:
  static and wrist V-JEPA temporal tokens.

M_point:
  point/depth/geometry tokens.

M_tactile:
  tactile/contact tokens.

M_track:
  tracklet / visual trace tokens.

M_post:
  previous posterior tokens.

M_cache:
  bounded address-aware historical evidence.

M_prop:
  optional pseudo proposal tokens.
```

AQR update:

```math
q_j^{l+1}
=
q_j^l
+
sum_{m in available}
gamma_{j,m}^l Attn(q_j^l, M_t^m)
+
SelfAttn(Q^l)_j
```

Complexity remains:

```math
O(K sum_m N_m) + O(K^2)
```

and never becomes:

```math
O((sum_m N_m)^2)
```

## 6. Phase 0: Invariant Tests

Before adding new runtime features, add tests or verifier checks for:

```text
future targets are detached.
cache read weight scales residual output.
cache skips newest posterior duplicate.
missing modalities are no-op.
PaliGemma heatmap remains off by default.
PI0.5 action path is unchanged.
AQR is default and legacy MAPG is off.
state-only burn-in uses AQR graph.
```

The current maintained runtime satisfies these checks. MVTrack must preserve
them.

## 7. Phase 1: Static + Wrist V-JEPA Multiview

### 7.1 Goal

The current runtime sends static and gripper/wrist RGB through view-indexed
V-JEPA clip buffers when the gripper image is available. Wrist RGB is a typed
temporal view, not a static-camera geometry source. This is the correct split:
the wrist view is reliable local hand/object evidence, but it should not be
projected into static-camera rays unless a calibrated wrist RGB camera model is
available.

### 7.2 Config

Add:

```python
vjepa_multiview_enabled: bool = True
vjepa_views: tuple[str, ...] = ("static", "gripper")
vjepa_share_encoder_across_views: bool = True
```

The runtime-c implementation intentionally does not expose a contact-gated
wrist switch. Wrist evidence is a typed visual view whenever it is present;
contact-dependent use is learned by AQR support mass rather than a hard input
gate.

### 7.3 Contracts

Extend `PicfTemporalVisualSupportState`:

```python
@dataclasses.dataclass
class PicfTemporalVisualSupportState:
    tokens: torch.Tensor
    time_ids: torch.Tensor
    view_ids: torch.Tensor
    grid_index: torch.Tensor
    grid_hw: torch.Tensor
    current_token_count: torch.Tensor
    valid: torch.Tensor
    view_names: tuple[str, ...] = ()
    grid_hw_by_view: torch.Tensor | None = None
    source_hw_by_view: torch.Tensor | None = None
```

### 7.4 Pipeline

Replace a single static clip buffer with view-indexed buffers:

```python
self.clip_buffers = {
    "static": VisualClipBuffer(...),
    "gripper": VisualClipBuffer(...),
}
self.clip_buffer = self.clip_buffers["static"]  # compatibility alias
```

Visual maps:

```python
for view_name in self.config.vjepa_views:
    rgb = observation.rgb_static if view_name == "static" else observation.rgb_gripper
    if rgb is None:
        continue
    self.clip_buffers[view_name].push(rgb, segment_id=..., reset=...)
    fmap = self.visual_encoder.encode_clip(self.clip_buffers[view_name].get_clip())
    current_by_view[view_name] = fmap.current_map(...)
    temporal_by_view[view_name] = fmap.recent_maps(...)
```

Token:

```math
z_{v,t,h,w}
=
W f_{v,t,h,w}
+
E_modality
+
E_view(v)
+
PE_time(t)
+
PE_grid(h,w)
```

### 7.5 Guard

`wrist extrinsics` means the calibrated transform from the wrist camera frame
to the robot/world/static frame, plus the camera intrinsics needed to turn a
wrist pixel into a metric ray. CALVIN already has the stronger geometry path
for gripper depth: the point-cloud builder uses `robot_obs` and the gripper
camera `E_T_C` transform to lift gripper depth into the robot/world frame. This
is analogous to the tactile path, where sensor-local contact proposals are
placed through a sensor-to-wrist transform and the current end-effector pose.

The maintained rule is therefore:

```text
wrist RGB/V-JEPA:
  typed local visual evidence with view_id=gripper.
  not static grid truth unless calibrated RGB intrinsics/extrinsics are present.

gripper depth / tactile:
  may contribute geometry when CALVIN robot pose and calibrated local sensor
  transforms are available.
```

### 7.6 Diagnostics

Add:

```text
aqr_temporal_view_mass_static
aqr_temporal_view_mass_gripper
aqr_temporal_view_entropy_static
aqr_temporal_view_entropy_gripper
wrist_temporal_priors_nonzero_fraction
view_switch_rate
```

## 8. Phase 2: Support Signature Binding

### 8.1 Goal

Current binding uses hidden similarity and geometry. MVTrack adds support
continuity and gated address compatibility.

This phase is the support-signature identity binding upgrade.

Current base:

```math
l_{j,i}
=
lambda_h cos(h_j^-, o_i)
-
lambda_g Maha(x_i; x_j^-, S_j^-)
```

MVTrack:

```math
l_{j,i}
=
lambda_c cos(W_c c_j^-, W_o o_i)
-
lambda_g (x_i-mu_j^-)^T (Sigma_j^- + Sigma_i + sigma^2 I)^{-1} (x_i-mu_j^-)
+
lambda_s g_j O_{j,i}^{support}
+
lambda_a g_j cos(W_a a_j^-, W_a a_i^{obs})
+
b_role
+
b_graph
```

Gate:

```math
g_j = alpha_j^- (1 - recycle_j^-) exp(-kappa nu_j^-)
```

This makes address inertia strong only when the slot is stable.

### 8.2 Support overlap

```math
O_{j,i}^{support}
=
lambda_v <s_j^{visual,-}, s_i^{visual}>
+
lambda_t <s_j^{temporal,-}, s_i^{temporal}>
+
lambda_p <s_j^{point,-}, s_i^{point}>
+
lambda_pg <s_j^{pg,-}, s_i^{pg}>
+
lambda_tr <s_j^{track,-}, s_i^{track}>
```

### 8.3 Address update

```math
a_j^+
=
normalize(
(1-rho_j) a_j^-
+
rho_j sum_i B_{j,i} a_i^{obs}
)
```

with:

```math
rho_j =
clip(rho_0 support_mass_j (1-recycle_j) exp(-kappa nu_j), 0, rho_max)
```

Recycle/birth resets address:

```math
a_j^+
=
normalize(a_{role(j)}^{learned} + eta sum_i B_{j,i} a_i^{obs})
```

### 8.4 Recycle gate scale invariant

The recycle gate is a belief-filter trust/reset probability. It must not be
driven by the unbounded norm of the aggregated dustbin residual:

```math
r = sum_i d_i o_i
```

because larger residual magnitude can saturate:

```math
recycle_j = sigma(f(h_j^-, support_j, var_j, r, alpha_j^-))
```

and then force:

```math
bar h_j = (1-recycle_j)h_j^- + recycle_j h_{res}
```

for many slots at once. That is not identity correction; it is reset
dominance. The maintained gate input therefore uses:

```math
hat r = LayerNorm(r)
recycle_j = sigma(f(h_j^-, support_j, var_j, hat r, alpha_j^-))
```

The residual heads still consume the original residual summary. Only the
probability gate is normalized. This preserves residual content while making
the reset decision scale-stable.

### 8.5 Diagnostics

Add:

```text
binding_hidden_score_mean
binding_geometry_score_mean
binding_support_overlap_score_mean
binding_address_score_mean
binding_graph_bias_mean
slot_address_update_rate_mean
slot_address_reset_fraction
posterior_identity_switch_rate
posterior_recycle_rate
```

## 9. Phase 3: Address-Aware Evidence Cache

### 9.1 Goal

The v26 cache is safe and residual-gated. MVTrack makes it query-conditioned
by address/content, not only flat age/uncertainty/innovation scoring.

### 9.2 Source enum

Define:

```python
CACHE_SOURCE_POSTERIOR = 1
CACHE_SOURCE_OBS_EVIDENCE = 2
CACHE_SOURCE_PREDICTED = 3
CACHE_SOURCE_TRACKLET = 4
CACHE_SOURCE_PROPOSAL = 5
```

### 9.3 Cache read state

Add:

```python
@dataclasses.dataclass
class PicfCacheReadState:
    tokens: torch.Tensor
    slot_address: torch.Tensor
    slot_content: torch.Tensor
    role_ids: torch.Tensor
    source_ids: torch.Tensor
    age: torch.Tensor
    uncertainty: torch.Tensor
    innovation: torch.Tensor
    modality_validity: torch.Tensor
    valid: torch.Tensor
```

### 9.4 Per-query score

```math
score_{j,i}
=
lambda_a cos(W_a a_j, W_a a_i^{cache})
+
lambda_c cos(W_c c_j, W_c c_i^{cache})
+
lambda_r 1[r_j=r_i]
+
b_source(source_i)
-
lambda_age age_i
-
lambda_unc uncertainty_i
-
lambda_innov innovation_i
+
lambda_mod <m_t, m_i>
```

Read:

```math
q_j'
=
q_j
+
lambda_cache (Attn(q_j, C, bias=score_j) - q_j)
```

The v26 residual gate must remain.

### 9.5 Diagnostics

Add:

```text
cache_address_score_mean
cache_content_score_mean
cache_role_score_mean
cache_age_penalty_mean
cache_innovation_penalty_mean
cache_read_mass_by_source
cache_same_address_fraction
cache_same_role_fraction
```

## 10. Phase 4: Tracklet Typed Memory

### 10.1 Goal

Tracklets provide temporal correspondence without human instance labels. This
directly targets identity switch and adjacent-object ambiguity.

Runtime-c wires optional tracklet tensors through training, replay, and serve
paths when those tensors are present in the episode or request. Standard CALVIN
frames that do not contain tracklets remain clean no-ops. The remaining
external dependency is tracklet generation, not PICF ingestion.

### 10.2 Offline preprocessing

Use a tracker such as CoTracker3 or TAPIR over static and wrist videos.

Save:

```text
episode_tracklets.npz:
  xy: [T, N, 2]
  visibility: [T, N]
  confidence: [T, N]
  velocity: [T, N, 2]
  track_id: [N]
  view_id: [N]
  optional_feat: [T, N, C]
```

Filter:

```text
visibility mean high
confidence high
forward-backward error low
velocity jump small
near high-confidence PG/point/action/contact region preferred
```

### 10.3 Contracts

Add:

```python
@dataclasses.dataclass
class PicfTrackletSupportState:
    tokens: torch.Tensor
    xy_norm: torch.Tensor
    velocity_norm: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor
    track_ids: torch.Tensor
    view_ids: torch.Tensor
    age: torch.Tensor
    valid: torch.Tensor
```

Extend:

```text
PicfTokenFieldState.tracklet
PicfAnchorPriorGraphState.tracklet_priors
PicfPosteriorAnchorState.tracklet_signature
```

### 10.4 AQR read

```math
p_{j,n}^{track}
=
softmax_n(q_j^T k_n^{track} + b_view + b_visibility + b_confidence + b_role)
```

Tracklet priors feed:

```text
AQR graph diagnostics
support signature
posterior binding bias
local refinement candidate set
denoising pseudo targets
```

## 11. Phase 5: Latent Local Refinement

### 11.1 Goal

Refine within existing typed memory. This does not require high-resolution
original images and does not pretend to create evidence that is absent.

Candidate set:

```math
Omega_j =
TopK(p_j^{vjepa})
union TopK(p_j^{pg})
union NN_point(mu_j, r_j)
union TopK(p_j^{track})
union tactile/contact neighborhood
```

Local support:

```math
p_{j,i}^{local}
=
softmax_{i in Omega_j}
(q_j^T k_i + b_geom + b_view + b_track + b_contact + b_relation)
```

Outputs:

```text
local_priors
anchor_x_refined
anchor_S_refined
local_support_entropy
local_view_mass
```

## 12. Phase 6: Training-Only Denoising Queries

### 12.1 Goal

Improve query/anchor convergence without making pseudo labels truth.

Pseudo target sources:

```text
high-confidence PG support peak
V-JEPA temporal support peak
point cluster
tracklet cluster
tactile/contact region
action endpoint
high-confidence posterior slot
optional proposal
```

Noisy query:

```math
tilde{x}=x+epsilon_x
tilde{c}=c+epsilon_c
tilde{sigma}=sigma+epsilon_sigma
```

Loss:

```math
L_dn =
Huber(hat{mu}-stopgrad(mu))
+
lambda_v KL(stopgrad(p^v) || hat p^v)
+
lambda_p KL(stopgrad(p^p) || hat p^p)
+
lambda_tr KL(stopgrad(p^track) || hat p^track)
+
lambda_r CE(hat r, r)
```

Guard:

```text
DN queries are training-only.
DN queries do not write posterior.
DN queries do not enter action.
DN queries do not enter task readout.
```

## 13. Phase 7: Matched Slot-JEPA and Support Prediction

Runtime-c replaces the enableable slot-JEPA/support prediction path with
detached soft-matched targets rather than same-index future slots. The losses
remain default zero because identity metrics, not static code presence, decide
when they are safe to activate.

Cost:

```math
C_{j,k}
=
lambda_c (1 - cos(hat c_j, stopgrad(c_k^+)))
+
lambda_g Maha(hat mu_j, mu_k^+)
+
lambda_s d(hat sigma_j, sigma_k^+)
+
lambda_a (1 - cos(hat a_j, a_k^+))
+
lambda_r 1[r_j != r_k]
```

Matching:

```math
Pi = Sinkhorn(-C / tau)
```

Slot prediction:

```math
L_slot =
sum_{j,k} Pi_{j,k} d(hat c_j, stopgrad(c_k^+))
```

Support prediction:

```math
L_support =
sum_{j,k} Pi_{j,k} KL(stopgrad(sigma_k^+) || hat sigma_j)
```

Mask:

```text
alpha high
recycle low
innovation low
support entropy low
role compatible
```

Default:

```text
lambda_slot_jepa = 0
lambda_support_pred = 0
lambda_binding_consistency = 0
```

Enable only after identity metrics are stable.

## 14. Phase 8: Gated Weak Ordinal

### 14.1 Goal

Convert ordinal diagnostic into weak relation supervision only when reliable.

Target:

```python
@dataclasses.dataclass
class PicfOrdinalRelationTarget:
    active: torch.Tensor
    selected_slot: torch.Tensor | None
    target_rank: torch.Tensor | None
    axis: torch.Tensor | None
    frame_id: torch.Tensor
    confidence: torch.Tensor
```

Soft rank:

```math
s_j = u_l^T mu_j
```

```math
rank_j = 1 + sum_{l != j} sigmoid((s_l - s_j) / tau)
```

Weak selected slot:

```math
j^* =
argmin_j d(mu_j, action_endpoint)
+
lambda_contact contact_cost_j
-
lambda_support support_conf_j
```

Loss:

```math
L_ordinal =
Huber(rank_{j^*} - r_l)
+
lambda_pair L_pairwise
```

Guard:

```text
explicit ordinal/relation prompt only
candidate separation high
selected slot confidence high
low weight
no posterior overwrite
no cache truth write
```

## 15. Phase 9: Optional Proposal Memory

Optional sources:

```text
SAM/SAM2 masks
DINO/DINOv2 objectness
Grounding/proposal boxes
```

SAM/DINO are explicitly not required for the maintained runtime and are not
implemented in this pass. Runtime-c only provides optional proposal tensor
ingestion and typed proposal routing. If an upstream system supplies proposal
centers/boxes/objectness/source ids, PICF can consume them; if not, the proposal
branch is a no-op.

Use only as:

```text
DN pseudo target source
local refinement candidate
weak objectness bias
debug overlay
```

Never use as:

```text
ground-truth mask
posterior identity override
direct action truth
```

## 16. Required Verification

MVTrack is code-level runtime complete when the static/verifier checks below
pass on the current code. It cannot be marked behavior-complete until fresh
training/evaluation artifacts also pass. The `v2.0-runtime-c` implementation
has code paths for multiview, tracklet, support-signature binding,
address-aware cache, local refinement, optional proposal memory, training-only
support denoising, matched prediction hooks, and weak ordinal diagnostics.

### 16.1 Static/verifier checks

```text
multiview_vjepa_view_ids:
  static + gripper input produces view_ids containing 0 and 1.

wrist_no_static_geometry_leak:
  uncalibrated wrist tokens do not become static grid truth.

support_overlap_binding_effect:
  high previous/current support overlap increases binding logit.

address_binding_gated:
  high innovation/recycle downweights address score.

address_update_slow_reset:
  stable slots update address slowly; recycled slots reset address.

cache_address_retrieval:
  same address/role cache entries receive higher per-query score.

cache_skip_latest_posterior:
  newest posterior row remains skipped by cache reader.

tracklet_optional_noop:
  no tracklet data preserves v26 forward compatibility.

tracklet_priors_nonempty:
  valid tracklet data produces graph.tracklet_priors.

local_refinement_no_highres_dependency:
  local refiner uses only existing typed memory top-k.

denoising_train_only:
  when DN is implemented, eval/serve do not construct DN queries.

ordinal_noop_without_prompt:
  no ordinal prompt gives zero ordinal loss and no posterior change.

ordinal_high_confidence_only:
  low-confidence selected slots do not produce ordinal loss.

matched_jepa_permutation_invariant:
  future slot permutation does not materially change matched loss.

matched_jepa_no_future_leakage:
  future target is detached and does not enter current AQR/action.
```

### 16.2 Runtime metrics

Monitor:

```text
aqr_temporal_view_mass_static
aqr_temporal_view_mass_gripper
aqr_pg_support_entropy
aqr_tracklet_support_entropy
aqr_same_role_support_overlap_max
effective_anchor_count
binding_support_overlap_score_mean
binding_address_score_mean
cache_address_score_mean
cache_read_mass_by_source
posterior_identity_switch_rate
posterior_recycle_rate
local_support_entropy
ordinal_active_fraction
innovation_norm
```

### 16.3 CALVIN/video acceptance

Required:

```text
fresh latest-code training run
500 / 2500 / 5000 step checkpoints
20-sequence CALVIN eval
anchor overlay videos
support heatmaps
cache/identity diagnostics
comparison against v26 baseline
```

Old checkpoints from pre-MVTrack code cannot validate MVTrack.

## 17. Failure Modes

```text
wrist mass always zero:
  wrist V-JEPA path is not used.

wrist mass dominates from the first step:
  wrist may be suppressing global layout; gate by contact/proximity.

address score high but identity switch high:
  address is learning noise; reduce address weight and rely on support overlap.

cache trust high during high innovation:
  stale evidence lock-in; increase innovation penalty.

same-role overlap high:
  anchor collapse persists; inspect support diversity and denoising targets.

matched JEPA decreases but identity switches increase:
  matching cost or mask is wrong.

ordinal improves relation prompts but harms non-relation tasks:
  ordinal target is not sufficiently gated.
```

## 18. Literature Validation

These papers support the direction, but they do not prove this exact
implementation will succeed.

```text
V-JEPA 2:
  self-supervised video models support understanding, prediction, and planning;
  V-JEPA 2-AC uses robot videos for latent action-conditioned world modeling.
  https://arxiv.org/abs/2506.09985

JEPA-VLA:
  recent V-JEPA predictive embeddings help VLA action prediction.
  https://arxiv.org/abs/2602.11832

VLA-JEPA:
  future latent state should be a leakage-free target, not current input.
  https://arxiv.org/abs/2602.10098

OA-WAM:
  object-addressable slots motivate address/content separation.
  https://arxiv.org/abs/2605.06481

Selective Perception for Robot:
  task-aware multi-view/wrist routing improves VLA efficiency and control.
  https://arxiv.org/abs/2602.15543

WristWorld:
  wrist views capture fine-grained hand-object interaction and are valuable for
  manipulation.
  https://arxiv.org/abs/2510.07313

TraceVLA:
  visual traces improve spatial-temporal awareness for robotic policies.
  https://arxiv.org/abs/2412.10345

CoTracker3:
  point tracks can be learned from pseudo-labeled real videos, supporting
  tracklet memory without human instance labels.
  https://arxiv.org/abs/2410.11831

Deformable DETR:
  local reference-point attention addresses dense/global attention limitations.
  https://arxiv.org/abs/2010.04159

DN-DETR and DINO:
  denoising-style auxiliary targets and improved anchor initialization
  stabilize query-based detection training. MVTrack uses this as a guarded
  support-denoising auxiliary, not as an inference-time query path.
  https://arxiv.org/abs/2203.01305
  https://arxiv.org/abs/2203.03605

Order-aware 3D grounding:
  ordinal/relation expressions require explicit order-aware modeling.
  https://arxiv.org/abs/2403.16539
```

## 19. Information Boundary

MVTrack improves evidence use and identity continuity. It cannot break the
observation limit.

Let:

```math
Y = target object identity
Z_{\le t} = all typed memory and history available to the policy
A_t = anchor/slot/action decision
```

Then:

```math
I(Y; A_t | language) <= I(Y; Z_{\le t} | language)
```

If all modalities fail to distinguish two adjacent sub-token objects, no
architecture can guarantee correct selection. MVTrack reduces architecture and
optimization error by increasing and better organizing `Z_{\le t}`:

```text
static+wrist temporal evidence
tracklet correspondence
support signatures
address-aware cache
local rereads
denoising training
matched prediction
weak ordinal constraints
```

It does not create nonexistent information.

## 20. Execution Order

Do not implement as an unordered module list. Implement in this order:

```text
Phase 0:
  verifier invariants and no-regression tests.

Phase 1:
  static+wrist V-JEPA typed multiview.

Phase 2:
  support-signature binding and gated address update.

Phase 3:
  address-aware cache retrieval.

Phase 4:
  tracklet typed memory.

Phase 5:
  latent local refinement.

Phase 6:
  training-only support denoising auxiliary.

Phase 7:
  matched slot-JEPA/support prediction.

Phase 8:
  gated weak ordinal head.

Phase 9:
  optional SAM/DINO/proposal memory.
```

Opening rules:

```text
cache address gate:
  open only after support identity diagnostics are stable.

tracklet branch:
  safe to add as typed evidence, but train/eval must no-op when missing.

support denoising:
  implemented as a training-only support-denoising auxiliary.
  default lambda_aqr_denoising remains 0.
  no inference effect.

matched predictive losses:
  default zero until identity switch/recycle metrics are stable.

ordinal weak loss:
  last; prompt-gated and low weight only.
```

## 21. Final Verdict

MVTrack is the correct next architecture contract for the current codebase.

It is not a replacement for v26. It is a strict extension:

```text
v26:
  maintained final baseline and current training target.

MVTrack:
  static+wrist V-JEPA typed memory.
  next full architecture contract for multiview, tracklet, support-signature,
  address-aware, local-refined, predictive belief routing.
```

The plan is coherent because every addition maps to:

```text
typed evidence
binding/correction
training-only detached teacher
```

and none of it bypasses posterior authority.

MVTrack should be considered code-level runtime-complete after Section 16
static/verifier checks pass on current code. It should be considered
behavior-complete only after fresh training artifacts, videos, and CALVIN
evaluation also pass.
