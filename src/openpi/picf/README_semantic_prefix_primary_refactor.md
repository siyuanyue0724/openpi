# PICF Semantic Prefix Primary Handoff

Date: 2026-04-15

This is the canonical handoff document for the current PICF refactor.

The durable specification that this handoff is expected to match is
[`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md).

It supersedes the sidecar-primary experiment lineage. The current intended
architecture is:

- one language-free physical core
- one full semantic-prefix primary control path
- one full semantic-prefix conditioned future path
- no task-anchor sidecar in the mainline

The purpose of this document is to record:

- the audited current code path
- the mathematical contract the code is meant to satisfy
- the exact file-level implementation that now exists
- the validation plan and latest verification results
- the deployment procedure for clean training

## 1. Final Design Summary

The final design is:

```text
observation
-> task-agnostic physical token field
-> task-agnostic observation anchors
-> language-free physical posterior
-> language-free physical predictive cache
-> next-step innovation

current control / conditioned future:
-> full PaliGemma semantic token sequence
-> projected posterior-late into the control / conditioned-future trunks
-> posterior tokens / global_post / innovation / proprio appended after it
```

The design does not use a task-anchor sidecar.

The reason is simple:

- sidecar-primary compressed the semantic pathway too aggressively
- it reduced the control-side semantic bandwidth far below `pi0.5`
- it was not what the intended design asked for

The semantic path is now sequence-preserving again:

- PaliGemma still encodes the full mixed image+text prefix
- PICF still keeps the physical posterior language-free
- control and conditioned future now consume the full projected semantic token
  sequence directly

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

- match the semantic-prefix usage pattern of `pi0.5` more closely
- preserve PICF physical-world invariants
- remove sidecar-specific configuration and runtime state
- keep the action normalization and auxiliary budgeting fixes

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

Equivalently:

- physical world-state is task-agnostic
- semantic is a current-step conditional read path
- innovation remains a comparison between world-only prediction and reality

## 4. Why This Matches The Intended `pi0.5`-Style Semantic Path Better

`pi0.5` and `pi0_fast_sonata` do not compress semantics into a tiny fixed set of
task tokens before the action trunk.

They:

- embed the full image+text prefix
- run that full sequence through the LLM trunk
- let the control head read from the full semantic prefix context

The current PICF design now does the analogous thing on the PICF side:

- PaliGemma still receives the full mixed image+text prefix
- the full semantic token sequence is preserved
- PICF projects that sequence from semantic width to core hidden width
- the full projected sequence is appended into control and conditioned future

So the current semantic pathway is no longer:

- sidecar tokens replacing the semantic main path

It is now:

- full semantic-prefix primary
- physical posterior appended as extra structured world-state evidence

## 5. PaliGemma Information Bandwidth

This matters for global context.

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

The current control path no longer compresses that to a fixed tiny token set.
Instead:

- those `512 + L` tokens are projected to hidden width `384`
- the full projected sequence is injected directly into:
  - the control trunk
  - the conditioned future trunk

This means:

- control-side semantic sequence length is now on the same order as the original
  `pi0.5` design
- the remaining difference versus raw `pi0.5` is width:
  - original hidden width at the PaliGemma side is `2048`
  - PICF projects to `384`

That width compression is still much less destructive than the old sidecar-only
compression to a fixed `10` tokens.

Global information is therefore present in two complementary forms:

- distributed across the full semantic token sequence
- explicitly present in `posterior.global_post`

## 6. End-to-End Dataflow

### 6.1 Replay window to observation

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

Action targets are normalized before entering the core if action normalization
is enabled.

### 6.2 Semantic encoding

`src/openpi/picf/paligemma/wrapper.py` produces:

- `PaliGemmaSemanticFeatures.tokens`
- `PaliGemmaSemanticFeatures.summary`

The core contract only consumes:

- token-level semantic inputs

Summary-only semantic inputs are invalid.

### 6.3 Physical token field

`src/openpi/picf/core/pipeline.py` builds:

- point tokens
- visual tokens
- tactile tokens
- context tokens
- fused tokens

This stage is physical and task-agnostic.

### 6.4 Observation anchors

Observation anchors are built from the fused physical token field.

They are:

- task-agnostic
- language-free

No semantic token participates here.

### 6.5 Posterior update

Posterior update reads:

- previous physical recurrent state
- current observation anchors
- current observation geometry

It does not read semantic tokens.

Outputs include:

- `posterior.tokens`
- `posterior.global_post`
- `posterior.mu`
- `posterior.Sigma`
- `posterior.binding`

### 6.6 Innovation

Innovation is built from:

- current targets
- `previous.predictive.physical_prediction_cache`

It does not read:

- previous semantic tokens
- previous conditioned future cache
- previous conditioned future global state

### 6.7 Control trunk

Control trunk input order is:

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

This is intentional.

The semantic sequence is primary.
The physical posterior is appended as structured world-state context.

### 6.8 Physical predictive basis

Physical predictive basis input is:

```text
[
  posterior.tokens,
  posterior.global_post,
  proprio_token,
  action_cond_token,
]
```

This branch is world-only.

It produces:

- `physical_global_pred`
- `physical_prediction_cache`

### 6.9 Conditioned future branch

Conditioned future input order is:

```text
[
  semantic_prefix_tokens,
  physical_pred_tokens,
  predictive_query_tokens,
]
```

This branch uses the full semantic prefix again.

It produces:

- `predictive_query_state`
- `global_pred`
- `prediction_cache`

### 6.10 Serving

Serving uses the same core contract.

Actions are:

- predicted in normalized action space
- unnormalized before environment execution

## 7. Runtime State Inventory

Important state objects are:

- `PicfTokenFieldState`
  - fused physical token field
- `PicfObservationAnchorState`
  - task-agnostic current observation anchors
- `PicfPosteriorAnchorState`
  - language-free physical recurrent world state
- `PicfPredictiveState`
  - current control output, physical predictive basis, conditioned future readout
- `PicfPredictionCache`
  - predicted future targets

Notably absent in the final design:

- no `PicfTaskAnchorState`
- no `task_anchors` field in `PicfCoreState`
- no sidecar-only semantic runtime path

## 8. Losses

The core action loss is:

- `loss_action_pos`
- `loss_action_rot`
- `loss_action_gripper`

Action loss is computed in normalized action space.

Physical auxiliary losses supervise:

- `physical_prediction_cache.visual_latent`
- `physical_prediction_cache.visual_real`
- `physical_prediction_cache.tactile_real`
- `physical_prediction_cache.point_real`

Conditioned future losses supervise:

- `prediction_cache.*`

Alignment losses remain separate.

Auxiliary budgeting is still enabled:

- physical auxiliary group is budgeted against action loss
- semantic-conditioned future group is budgeted against action loss
- alignment group is budgeted against action loss

The percentile gradient clipping mode is outside the core math:

- it only affects trainer optimization
- it does not change the physical or semantic dataflow contract

## 9. File-Level Implementation Summary

### `src/openpi/picf/core/config.py`

Current mainline config:

- keeps semantic-prefix controls
- no task-sidecar config fields remain

### `src/openpi/picf/core/contracts.py`

Current mainline runtime state:

- no `PicfTaskAnchorState`
- no `task_anchors` in `PicfCoreState`

### `src/openpi/picf/core/pipeline.py`

Current mainline dataflow:

- no task-query modules
- no `_build_task_anchors(...)`
- no sidecar debug outputs
- control and conditioned future are full-semantic-prefix primary

### `scripts/picf_core_train.py`

Current mainline trainer:

- no sidecar CLI flags
- no sidecar config plumbing
- percentile clip supported
- semantic-prefix contract logging updated to the new primary-prefix wording

### `scripts/picf_resume_train.py`

Current mainline resume path:

- no sidecar CLI override layer
- clean resume surface

### `scripts/verify_picf_contract.py`

Current verifier checks:

- sidecar removed from pipeline
- full semantic prefix used directly in control
- full semantic prefix used directly in conditioned future
- physical posterior and innovation invariants preserved

## 10. Validation Results

Latest local validation after the semantic-prefix-primary refactor:

- `pytest -q src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py scripts/picf_core_train_test.py scripts/picf_resume_train_test.py scripts/serve_picf_policy_test.py src/openpi/picf/paligemma/wrapper_test.py src/openpi/picf/vjepa/wrapper_test.py src/openpi/picf/action_normalization_test.py src/openpi/picf/pointcloud_picf_test.py`
  - `134 passed`
- `python scripts/verify_picf_contract.py`
  - static contract checks: `PASS`
  - targeted invariance regressions: `10 passed`
  - core regression suite: `113 passed`
  - smoke training check: `PASS`

Latest cloud validation on the clean synced worktree
`/root/openpi_sync_semantic_prefix_primary_20260415`:

- `pytest -q src/openpi/picf/core/pipeline_test.py scripts/picf_core_train_test.py scripts/picf_resume_train_test.py scripts/serve_picf_policy_test.py`
  - `91 passed`
- `python scripts/verify_picf_contract.py --skip-smoke`
  - static contract checks: `PASS`
  - targeted invariance regressions: `10 passed`
  - core regression suite: `113 passed`

The targeted invariance tests now cover:

- language-late posterior invariance
- physical predictive invariance under prompt changes
- semantic conditioning of control and conditioned future
- explicit dependence on `posterior.global_post`
- full semantic-prefix sequence inclusion in both control and conditioned future
- executed-action continuity
- next-step innovation isolation from conditioned-future state

## 11. Clean Training Deployment

The clean deployment assumption is:

- do not resume an old sidecar-primary checkpoint
- start a clean run under the final semantic-prefix-primary contract

### 11.1 Tactile calibration

If cloud calibration files are missing, regenerate them with the full script:

```bash
python scripts/calvin/precompute_tactile_contact_calibration.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --device cuda \
  --anytouch-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --output-dir /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8
```

### 11.2 Safe 4-card start

For the current `4 x A100 40GB` machine, the latest empirical safe setting was:

- `accum_steps=1`
- effective global batch `4`

Higher accumulation settings on this configuration hit OOM during backward even
when startup looked fine.

### 11.3 Recommended clean training command

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
  --exp-name picf_semantic_prefix_primary_a4_acc1_pct75_clean_v1
```

This command deliberately does not use:

- sidecar flags
- sidecar rollout A/B logic
- old checkpoint resume

## 12. Final Handoff

The codebase is now aligned to the following design:

- full semantic-prefix primary control path
- full semantic-prefix conditioned-future path
- language-free physical posterior
- language-free physical predictive basis
- innovation based only on previous physical predictive cache

The design is mathematically cleaner than the sidecar-primary branch and is much
closer to the intended `pi0.5`-style semantic information path.

If future work is done, it should start from this document, not from the
obsolete sidecar handoff notes.
