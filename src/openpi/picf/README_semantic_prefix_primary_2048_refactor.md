# PICF Semantic Prefix Primary 2048 Handoff

Date: 2026-04-15

This is the canonical handoff for the current PICF refactor.

The durable specification this document is expected to match is
[`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md).

It supersedes both:

- the sidecar-primary experiment lineage
- the earlier 384-width semantic-prefix-primary variant

## 1. Final Design Summary

The current intended architecture is:

- one language-free physical core
- one full semantic-prefix primary control path
- one full semantic-prefix conditioned future path
- no task-anchor sidecar in the mainline
- no semantic width compression in the mainline

The design is:

```text
observation
-> task-agnostic physical token field
-> task-agnostic observation anchors
-> language-free physical posterior
-> language-free physical predictive cache
-> next-step innovation

current control / conditioned future:
-> full PaliGemma semantic token sequence at width 2048
-> posterior/global_post/innovation/proprio appended after it
-> no semantic sidecar replacement path
```

The purpose of this document is to record:

- the audited current code path
- the mathematical contract the code is meant to satisfy
- the exact file-level implementation that now exists
- the validation matrix
- the clean deployment procedure

## 2. What Was Audited

The refactor was audited against:

- `src/openpi/models/pi0.py`
- `src/openpi/models/pi0_fast_sonata.py`
- `src/openpi/picf/paligemma/wrapper.py`
- `src/openpi/picf/core/config.py`
- `src/openpi/picf/core/contracts.py`
- `src/openpi/picf/core/pipeline.py`
- `src/openpi/picf/core/training.py`
- `scripts/picf_core_train.py`
- `scripts/picf_resume_train.py`
- `scripts/serve_picf_policy.py`
- `scripts/verify_picf_contract.py`
- `src/openpi/picf/core/pipeline_test.py`
- `scripts/picf_core_train_test.py`
- `scripts/picf_resume_train_test.py`
- `docs/CALVIN_VALIDATION_README.md`

The audit goal was:

- keep PaliGemma image+text tokenization close to `pi0.5`
- keep the semantic sequence length intact
- keep the semantic sequence width intact
- preserve PICF physical-world invariants
- remove sidecar-specific configuration and runtime state
- keep action normalization and auxiliary budgeting intact

## 3. Mathematical Contract

At control step `t`, define:

- current observation:
  - `O_t`
- current semantic token sequence from PaliGemma:
  - `S_t`
- physical recurrent state:
  - `P_t`
- previous physical predictive basis:
  - `C_{t-1}^{phys}`

The intended transition is:

```text
token_field_t = Tokenize(O_t)
anchors_t = ObservationAnchors(token_field_t)
posterior_t = PosteriorUpdate(P_{t-1}, O_t, anchors_t)
innovation_t = Innovation(C_{t-1}^{phys}, targets_t)

control_t = ControlRead(
    semantic_prefix_t,
    posterior_t,
    global_post_t,
    innovation_t,
    proprio_t,
    query_t,
)

physical_pred_t = PhysicalPredict(
    posterior_t,
    global_post_t,
    proprio_t,
    action_cond_t,
)

semantic_pred_t = ConditionedFutureRead(
    semantic_prefix_t,
    physical_pred_t,
    predictive_query_t,
)
```

The crucial invariants are:

- semantic does not affect physical observation anchors
- semantic does not affect physical posterior update
- semantic does not affect `physical_prediction_cache`
- next innovation reads only `previous.predictive.physical_prediction_cache`
- semantic affects current control and conditioned future only posterior-late
- the main semantic path keeps native width `2048`

Equivalently:

- physical world-state is task-agnostic
- semantic is a current-step conditional read path
- innovation remains a comparison between world-only prediction and reality

## 4. Why This Matches The Intended `pi0.5`-Style Semantic Path Better

`pi0.5` and `pi0_fast_sonata` do not replace the semantic main path with a tiny
fixed set of task tokens.

They:

- embed the full image+text prefix
- run that full sequence through the LLM trunk
- let the action head read from the full semantic prefix context

The current PICF design now mirrors the same main idea:

- PaliGemma receives the full mixed image+text prefix
- the full semantic token sequence is preserved
- the semantic token width remains `2048`
- PICF control and conditioned future consume that full sequence directly
- posterior tokens and `global_post` are appended as structured physical
  context, rather than replacing the semantic main path

## 5. Width Contract

The current core defaults are:

- `hidden_dim = 2048`
- `posterior_hidden_dim = 2048`
- `innovation_dim = 2048`
- `control_dim = 2048`
- `future_hidden_dim = 2048`
- `semantic_dim = 2048`
- `semantic_cross_dim = 2048`
- `attention_heads = 8`

When `semantic_dim == hidden_dim`, the mainline does not apply semantic width
compression:

- `semantic_prefix_proj = Identity()`

So the current semantic path is:

- full sequence preserved
- full width preserved
- no sidecar replacement

## 6. PaliGemma Information Bandwidth

With the current wrapper:

- image size is `224`
- patch size is `14`
- each image yields `256` PaliGemma tokens
- the current wrapper can use:
  - `rgb_static`
  - `rgb_gripper`

So image tokens contribute:

- `512` semantic tokens before text is counted

If the prompt length is `L`, the semantic sequence length is:

- `512 + L`

Under the current 2048-width contract:

- those `512 + L` semantic tokens keep width `2048`
- the full sequence enters:
  - the control trunk
  - the conditioned future trunk

Global information is therefore present in two complementary forms:

- distributed across the full semantic token sequence
- explicitly present in `posterior.global_post`

## 7. End-to-End Dataflow

### 7.1 Replay window to observation

The trainer builds a `PicfObservation` from CALVIN replay.

This includes:

- `rgb_static`
- `depth_static`
- `rgb_gripper`
- `depth_gripper`
- `robot_obs`
- optional tactile packet
- prompt
- optional action

Replay actions are normalized before entering the core when normalization is
enabled.

### 7.2 Semantic encoding

`src/openpi/picf/paligemma/wrapper.py` produces:

- `PaliGemmaSemanticFeatures.tokens`
- `PaliGemmaSemanticFeatures.summary`

The core consumes token-level semantic inputs only.
Summary-only semantic inputs are invalid.

### 7.3 Physical token field

`src/openpi/picf/core/pipeline.py` builds:

- point tokens
- visual tokens
- tactile tokens
- context tokens
- fused tokens

This stage is physical and task-agnostic.

### 7.4 Observation anchors

Observation anchors are built from:

- task-agnostic point FPS seeds
- task-agnostic reads from the fused token field

Current semantic tokens do not participate in this stage.

### 7.5 Posterior update

The posterior update consumes:

- previous physical state
- current observation anchors
- current physical observation summaries

It does not consume semantic.

### 7.6 Innovation

Innovation is built from:

- current reality targets
- `previous.predictive.physical_prediction_cache`

It does not consume:

- previous conditioned future cache
- previous semantic tokens
- previous conditioned global readout

### 7.7 Control trunk

The logical control prefix is:

```text
[
  semantic_prefix_tokens,
  posterior.tokens,
  posterior.global_post,
  innovation_token,
  proprio_token,
  control_query_tokens,
]
```

The semantic prefix is the primary sequence.
The posterior stream is appended as structured physical context.

### 7.8 Physical predictive basis

The language-free predictive basis reads:

- `posterior.tokens`
- `posterior.global_post`
- `proprio_token`
- `action_cond_token`

It outputs:

- `physical_global_pred`
- `physical_prediction_cache`

### 7.9 Conditioned future branch

The logical conditioned future prefix is:

```text
[
  semantic_prefix_tokens,
  physical_pred_tokens,
  predictive_query_tokens,
]
```

This branch may affect:

- conditioned future outputs
- current action readout through semantic conditioning

It must not affect:

- physical posterior
- physical predictive cache
- next-step innovation base

### 7.10 Serving

Serving preserves:

- normalized action contract inside the core
- unnormalized action at environment execution

## 8. Runtime State Inventory

The major runtime state groups are:

- `token_field`
- `observation_anchors`
- `posterior`
- `predictive`

Important predictive fields:

- `semantic_tokens`
- `control_tokens`
- `control_query_state`
- `action`
- `executed_action`
- `physical_global_pred`
- `physical_prediction_cache`
- `predictive_query_state`
- `global_pred`
- `prediction_cache`

No task-anchor sidecar state is part of the current runtime contract.

## 9. Losses

The major loss groups are:

- action loss:
  - `loss_action_pos`
  - `loss_action_rot`
  - `loss_action_gripper`
- physical auxiliary:
  - `loss_visual_latent`
  - `loss_visual_real`
  - `loss_tactile_real`
  - `loss_point_real`
- conditioned future auxiliary:
  - `loss_semantic_future_aux`
- alignment / routing:
  - `loss_anchor_pv`
  - `loss_pv_weak`
  - `loss_focus_pv`
  - `loss_pt`

Action is the primary optimization target.
Auxiliary groups remain budget-controlled relative to action loss.

## 10. File-Level Implementation Summary

### `src/openpi/picf/core/config.py`

Current defaults are fully 2048-wide for the core state dimensions.

### `src/openpi/picf/core/contracts.py`

No task sidecar state remains in the mainline runtime state.

### `src/openpi/picf/core/pipeline.py`

Current semantic path:

- full semantic sequence
- full semantic width
- `semantic_prefix_proj = Identity()` when `semantic_dim == hidden_dim`

Current physical path:

- observation anchors language-free
- posterior language-free
- innovation language-free

### `scripts/picf_core_train.py`

Current trainer contract:

- full semantic-prefix-primary startup logging
- percentile grad clipping defaults
- clean-start support for the current 2048-width model

### `scripts/picf_resume_train.py`

Resume plumbing remains generic, but clean-start is the intended path for this
2048 refactor.

### `scripts/verify_picf_contract.py`

The verifier now points at this document as the canonical handoff and checks:

- no sidecar path in the pipeline
- full semantic prefix directly enters control and conditioned future
- `posterior.global_post` explicitly enters control
- physical invariants remain intact

## 11. Validation Results

Latest local verification after the 2048 refactor:

- wide pytest suite:
  - `134 passed`
- `python scripts/verify_picf_contract.py`:
  - static checks: `PASS`
  - targeted invariance regressions: `10 passed`
  - core regression suite: `113 passed`
  - smoke training check: `PASS`

Latest cloud verification after syncing the 2048 refactor:

- key pytest suite:
  - `91 passed`
- `python scripts/verify_picf_contract.py --skip-smoke`:
  - static checks: `PASS`
  - targeted invariance regressions: `10 passed`
  - core regression suite: `113 passed`

## 12. Clean Training Deployment

### 12.1 Tactile calibration

If calibration files are missing, regenerate them with the full supported
script:

```bash
python scripts/calvin/precompute_tactile_contact_calibration.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --device cuda \
  --anytouch-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --output-dir /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8
```

### 12.2 Clean 4-card start

The requested launch mode for this refactor is:

- clean start
- `4 x A100 40GB`
- `accum_steps=1`
- `save_interval=5000`
- percentile grad clipping:
  - window `100`
  - percentile `75`

### 12.3 Recommended clean training command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --device cuda \
  --use-foundation-backbones \
  --use-tactile \
  --accum-steps 1 \
  --save-interval 5000 \
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --tactile-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --tactile-calibration-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_fingertip_calibration.json \
  --tactile-backgrounds-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_backgrounds.npz \
  --tactile-contact-stats-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_contact_stats.json \
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --semantic-checkpoint-path /mnt/checkpoints/pi05_base_pytorch \
  --exp-name picf_semantic_prefix_primary_2048_a4_acc1_pct75_clean_v1
```

## 13. Final Handoff

The current mainline is now:

- semantic-prefix-primary
- sidecar-free
- 2048-wide throughout the core state dimensions
- identity semantic prefix projection
- language-free physical posterior / innovation contract

This is the version that should be treated as the current PICF spec.
