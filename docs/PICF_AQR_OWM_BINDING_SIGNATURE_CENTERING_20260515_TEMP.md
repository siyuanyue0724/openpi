# PICF-AQR-OWM Binding Signature Centering Follow-Through

Status: active diagnostic repair for the 2026-05-15 A7 Phase-B failure mode.

Canonical entry point: `src/openpi/picf/README_v2.2.md`.

## Problem

Phase B strict capacity/competition reached step 100 with healthy active support
ownership:

```text
aqr_same_role_support_overlap_max        = 0.1492
aqr_active_same_role_support_overlap_max = 0.0085
aqr_active_same_role_object_core_overlap = 0.0627
loss_action_default_equiv                = 0.0736
```

But the observation binding-signature overlap stayed saturated:

```text
aqr_same_role_obs_binding_signature_overlap_mean = 0.9846
aqr_same_role_obs_binding_signature_overlap_max  = 0.9989
```

This means active duplicate ownership can be suppressed, but the projected
binding representation itself is still dominated by a common scene/modality
component. Further overlap penalties would be post-hoc; the repair must target
the representation used for binding.

## Literature Constraint

Object-binding ViT work argues that `IsSameObject` is a pairwise/quadratic
relation encoded in a low-dimensional subspace of pretrained ViT features, not
a raw token-average property:

```text
Li et al., "Does Object Binding Naturally Emerge in Large Pretrained Vision
Transformers?", arXiv:2510.24709, NeurIPS 2025 Spotlight.
```

SlotContrast shows the complementary point for object-centric video: stable
object slots require temporal consistency; recurrent processing alone is not
enough when objectives do not preserve object identity:

```text
Manasyan et al., "Temporally Consistent Object-Centric Learning by Contrasting
Slots", CVPR 2025.
```

Downloaded local paper copies for this audit:

```text
/tmp/picf_lit_20260515/object_binding_vit_2510.24709.pdf
/tmp/picf_lit_20260515/slotcontrast_2412.14295.pdf
```

## Mathematical Failure Mode

The old support signature was effectively:

```math
k_i = normalize(W z_i)
s_j = normalize(\sum_i p_{j,i} k_i)
```

If `W z_i = c + r_i`, with a large common component `c` shared by all tokens
inside a typed memory, then for different anchors `j,l`:

```math
s_j \approx normalize(c + \bar r_j)
s_l \approx normalize(c + \bar r_l)
cos(s_j,s_l) \to 1
```

This can happen even when `p_j` and `p_l` are not identical. It exactly matches
Phase B: active support ownership is distinct, but binding signatures remain
nearly collinear.

## Repair

Use a centered projected binding key inside each typed memory before support
pooling:

```math
\tilde k_i = normalize(W z_i - mean_m(W z))
s_j = normalize(\sum_i p_{j,i} \tilde k_i)
```

This removes the global scene/modality common component and keeps the binding
signature focused on relative same-object evidence. It is not a new loss, not a
manual object prior, and not a hard ownership label. It only changes the readout
geometry of an already-present binding subspace.

## Dataflow Follow-Through

```text
PicfCoreConfig
  binding_signature_centering_enabled = True
  binding_signature_centering_min_tokens = 4

PicfFullCore._binding_keys(tokens, center=True)
  project tokens through binding_signature_proj
  subtract projected token-set mean when enough tokens exist
  normalize centered projected keys

PicfFullCore._support_binding_signature(weights, tokens)
  normalize support weights
  pool centered binding keys
  normalize support signature

PicfFullCore._build_observation_anchors
  point / visual / temporal / tracklet / proposal signatures all use the same
  centered support-binding path

PicfFullCore._binding_logits
  previous.posterior.binding_signature and observation.binding_signature still
  enter as a gated pairwise score through bind_embedding_signature_weight
```

## Script Evidence

New audit script:

```bash
python scripts/picf_binding_signature_common_mode_audit.py --fail-on-fail
```

Observed local result:

```text
PASS config_exposes_centered_binding_signature
PASS pipeline_centers_binding_keys_before_support_pooling
PASS binding_logits_use_gated_pairwise_signature
PASS unit_test_covers_common_mode_case
PASS numpy_common_mode_sanity

raw_offdiag_cos_mean      = 0.9809
centered_offdiag_cos_mean = -0.3180
mean_drop                 = 1.2989
```

The NumPy sanity check proves the old formula can produce false near-1.0
same-role binding cosine from common-mode features, and that centering removes
that failure on the controlled construction.

## Runtime Diagnostic Plan

Run a 1-2 hour A7 diagnostic with:

```text
action on
unroll_steps = 2
burnin_steps = 1
strict active capacity/competition kept from Phase B
binding_signature_centering_enabled = true
slot_jepa/support_pred/binding_consistency remain 0
local_refinement remains archived/disabled
```

Acceptance at 50/100/150:

```text
aqr_active_same_role_support_overlap_max <= 0.20
aqr_active_same_role_object_core_overlap_max <= 0.20
aqr_same_role_obs_binding_signature_overlap_mean should drop materially below
  the Phase-B 0.9846 baseline
loss_action_default_equiv should not regress relative to Phase B
grad_norm must not repeat an unclipped 1000+ spike
posterior_dustbin_mass_raw remains low
posterior_recycle_rate should not climb monotonically
```

Non-acceptance:

```text
active supports remain healthy but obs-binding overlap stays >= 0.98:
  centering alone is insufficient; run the offline IsSameObject probe next.

obs-binding overlap drops but active support collapses:
  binding subspace is useful but active owner capacity/gates are too weak.

action improves but recycle/grad explodes:
  optimization schedule/clip is the next blocker, not binding geometry.
```

## Scope Boundary

This is a representation-level repair. It intentionally does not introduce:

```text
new object labels
new SAM/DINO proposal dependency
new ordinal rank supervision
new slot-JEPA or support-prediction pressure
new post-hoc overlap penalty
```

Those are separate experiments. This change is the smallest coherent repair
that follows the paper-level insight and the observed Phase-B dataflow failure.
