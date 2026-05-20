# PICF-AQR-OWM 30K Validation And Dense Context Plan

Date: 2026-05-20

Canonical entry:

```text
src/openpi/picf/README_v2.2.md
```

## Current Decision

Start the 30000-step action-aware validation from the exact slot contract that
passed:

```text
frozen-policy owner-direct smoke:
  picf_a7_owner_direct_final_smoke300_20260520

action-aware smoke:
  picf_a7_actionaware_after_dedup_smoke300_20260520
```

The long run is not a new architecture experiment.  It tests whether the
already repaired active object-owner path survives normal semantic/action
training pressure.

## 30K Trainability Contract

```text
freeze:
  V-JEPA pretrained visual encoder
  Sonata pretrained point encoder
  AnyTouch pretrained tactile encoder

train:
  PaliGemma / PI0.5 semantic-action stack
  PICF AQR / posterior / task / control adapters
  action-side and prediction heads

keep disabled:
  slot_jepa
  support_pred
  binding_consistency
  aqr_denoising
```

Rationale:

```text
The short smokes show that active object ownership and action pressure can
coexist.  They do not show that guarded predictive/denoising losses are mature
enough to be production pressure.  Therefore the 30K run should not reopen
these hooks.
```

Launcher:

```text
scripts/experiments/picf_aqr_owm_202605_active/run_a7_actionaware_ownerdirect_long30k_20260520.sh
```

Sidecar requirement:

```text
default sidecar root:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520

required file:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520/calvin_segment_indices.txt

required audit:
  scripts/picf_prepare_full_sidecar_root.py
    --require-proposal
    --require-mask
    --require-tracklet
```

The wrapper now fails closed if `calvin_segment_indices.txt` is absent and no
explicit `SEGMENTS` override is supplied.  This prevents accidental fallback to
old diagnostic roots such as `contact_motion_mask_1000_20260518`.

It inherits:

```text
unroll_steps=2
burnin_steps=1
burnin_mode=state_only
training_strategy=fsdp_full_shard
optimizer_sharding=none
log_interval=50
anchor_overlay_interval=50
save_interval=2500
keep_last_checkpoints=3
progress=true
```

## Acceptance Gates

The first 500 steps decide whether to keep the 30K run alive:

```text
active_same_role_support_overlap_max:
  should remain below 0.05 most of the time.

posterior_file_competition_active_duplicate_overlap_max:
  should remain 0.

posterior_owner_transport_active_dist_after_fusion_mean:
  should remain in the centimeter-or-better range.

aqr_downstream_same_role_support_overlap_max:
  should remain below 0.60.  Raw all-row overlap may be high.

loss_action_default_equiv:
  should not explode and should trend toward the action-aware smoke band.

overlays:
  active owner should remain on the sidecar/contact object.
```

Checkpoint gates:

```text
2500:
  first saved model; inspect action trend, overlays, active ownership, and
  context/downstream overlap.

7500:
  if action improves but owner metrics degrade, stop and repair context
  exposure before spending the full budget.

30000:
  only meaningful if active ownership and action both remain healthy; then run
  CALVIN/video acceptance.
```

## Dense Context Follow-Up

The remaining architecture issue is not active object ownership.  It is that
background and peripheral evidence are still partly represented through
slot-like reserve/context rows, which makes raw same-role overlap high.

The next clean design is:

```text
object slots:
  explain task-relevant objects with posterior identity and tactile/contact
  attachment.

dense context:
  preserve raw V-JEPA/static/wrist/temporal tokens as non-object context.

dustbin/background:
  records unexplained evidence and prevents object slots from being forced to
  explain background.
```

This matches the direction of JEPA-VLA and VLA-JEPA: predictive video
embeddings are useful because they encode task-relevant temporal dynamics while
avoiding overfitting to nuisance background variation.  In PICF terms, dense
context should be injected by a gated attention/bias path, not by converting
background into persistent object files.

Proposed next-stage mechanism, not enabled in this 30K:

```text
context_gate = sigmoid(g(object_state, instruction_state, global_vjepa_state))

action_prefix =
  active_object_files
  + context_gate * dense_context_attention(dense_vjepa_tokens)
  + low_weight_state_embedding
```

This is a follow-up because the current 30K must first answer whether the
repaired object-owner contract is stable under normal action training.  Adding
another dense-context mechanism before that would confound the acceptance
signal.

## Paper Pointers

```text
JEPA-VLA:
  https://arxiv.org/abs/2602.11832
  Supports adaptive integration of predictive video embeddings into VLA
  policies and the idea that predictive embeddings can emphasize task-relevant
  temporal dynamics.

VLA-JEPA:
  https://arxiv.org/abs/2602.10098
  Supports leakage-free latent prediction and avoiding pixel/background
  reconstruction as the main control objective.
```

## Non-Goals

```text
Do not add blind SAM.
Do not turn reserve/context rows into object-file truth.
Do not enable reconstruction decoder as action-path truth.
Do not enable slot_jepa/support_pred/binding_consistency/denoising in this 30K.
```

## 2026-05-20 Sidecar Generation Handoff

The clean full sidecar root is still being generated on A7:

```text
tmux:
  picf_a7_full_tracklets_clean_20260520

input:
  /mnt/picf_sidecars/contact_motion_full_20260519

output:
  /mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520

observed progress at 2026-05-20 22:25 CST:
  2800 / 7545 progress_segments

ETA:
  about 3.0-4.0 hours remaining
```

Do not start the 30K run before the final `--require-proposal --require-mask
--require-tracklet` audit passes on the output root.
