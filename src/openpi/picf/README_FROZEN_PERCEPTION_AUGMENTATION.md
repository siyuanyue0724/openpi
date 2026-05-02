# PICF Frozen Perception And Augmentation Design

Date: 2026-04-30
Status: implemented operator contract for the current 2x40GB frozen-perception
PICF profile and geometry-safe augmentation policy

This document records the intended engineering contract for two related topics:

- freezing large perception backbones without changing the PICF architecture
- adding train-time augmentation without breaking PICF's multimodal geometry

It is an implementation and operator contract. The executable launch commands
are maintained in
[`docs/CALVIN_VALIDATION_README.md Section 5.1A`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51a-current-2x40gb-frozen-perception-full-picf-profile).
The architecture summary and status are maintained in
[`README_v2.2.md Section 0.1`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md#01-current-training--model-summary).

## 1. Goal

The target profile is:

```text
picf_frozen_perception_2x40
```

It is meant for 2x A100 40GB finetuning when full all-backbone cotrain is too
expensive.

The profile should preserve:

- the v2.2 PICF object model
- the world-only physical posterior and innovation boundary
- the semantic-conditioned task readout
- the canonical conditioned control state `C_t`
- the PI0.5 flow-matching final action path

The profile may change:

- which pretrained perception backbones receive gradients
- exact execution strategy, such as FSDP/DDP and checkpointing
- optional train-time augmentation, when augmentation is geometry-safe

The profile must not silently change:

- V-JEPA feature shape or feature semantics
- PI0.5 prompt/action normalization
- action horizon
- unroll/window semantics
- physical geometry coordinate frames

## 2. Recommended 2x40GB Profile

Full-BPTT 2x40GB reference recommendation:

```text
picf_mode=enabled
training_strategy=fsdp_full_shard
optimizer_sharding=none
world_size=2
accum_steps=1
unroll_steps=3
action_horizon=16
semantic_max_length=256 primary / 200 memory fallback
semantic_trainable=True
perception_finetune_mode=frozen
visual_feature_mode=hierarchical
visual_real_grid=64
persistent_anchors=8
observation_anchors=16
point_backbone_trainable=False
tactile_trainable=False
picf_augmentation_mode=photometric
picf_photometric_strength=conservative
save_interval=5000
```

Current selected sub-15s 2x40GB long-run:

```text
picf_mode=enabled
training_strategy=fsdp_full_shard
optimizer_sharding=none
world_size=2
accum_steps=1
unroll_steps=1
burnin_steps=4
burnin_mode=state_only
effective_window_steps=5
action_horizon=16
semantic_max_length=256 primary / 200 memory fallback
semantic_trainable=True
semantic_gradient_checkpointing=False
perception_finetune_mode=frozen
visual_feature_mode=hierarchical
visual_real_grid=64
persistent_anchors=8
observation_anchors=16
picf_augmentation_mode=photometric
picf_photometric_strength=conservative
save_interval=5000
```

This selected profile keeps the full PICF suffix path and PI0.5 action path
trainable, but uses four no-grad recurrent state updates before one trainable
suffix transition. It is the current runtime compromise for the sub-15s target.
It is not equivalent to full-BPTT `unroll_steps=5`.

Fast 2x40GB speed profile:

```text
picf_mode=enabled
training_strategy=fsdp_full_shard
optimizer_sharding=none
world_size=2
accum_steps=1
unroll_steps=1
burnin_steps=0
action_horizon=16
semantic_max_length=256 primary / 200 memory fallback
semantic_trainable=True
semantic_gradient_checkpointing=False
perception_finetune_mode=frozen
visual_feature_mode=hierarchical
picf_augmentation_mode=photometric
picf_photometric_strength=conservative
save_interval=5000
```

This speed profile is exact for its stated objective, but it is not equivalent
to the `unroll_steps=3` full-BPTT recurrent reference. Use it when
throughput is the priority and evaluate CALVIN quality separately before
promoting it beyond a speed probe.

Trainable modules:

- PaliGemma / PI0.5 semantic-action path
- PICF core
- task readout
- conditioned control
- prediction heads
- modality adapters / projections

Frozen modules:

- V-JEPA2.1 visual backbone
- Sonata point backbone
- AnyTouch tactile backbone

Rationale:

- V-JEPA2.1 has the largest visual activation footprint when trainable.
- Sonata and AnyTouch also carry nontrivial activation and optimizer cost.
- Freezing these perception backbones follows the common JEPA/VLA practice:
  pretrained dense perception features are kept stable, while task/adaptation
  layers learn how to use them.
- This is not a model-architecture ablation. The PICF readout and control
  routes still consume the modality evidence; only the backbone weights are
  fixed.
- `unroll_steps=3` is the current maximum validated full-BPTT window on 2x40GB.
  `unroll_steps=4` and `8` were tested and OOMed on this profile.
- Direct DDP/no-FSDP was tested on the same 2x40GB frozen-perception setup and
  OOMed in the trainable PaliGemma/Gemma forward MLP. FSDP remains required
  unless the semantic path is frozen, LoRA-adapted, shortened, or otherwise
  changed.
- The fastest capacity-preserving 2x40GB probe kept FSDP, used
  `unroll_steps=1`, and disabled semantic gradient checkpointing. It reached
  about `0.107-0.114 steps/sec` after step 1, roughly `8.8-9.3 s/step`.
- The selected `burnin_steps=4`, `burnin_mode=state_only`, `unroll_steps=1`
  run reached early steps at roughly `11.7-14.3 s/step`, while exposing the
  trainable suffix to four previous recurrent state updates.
- Conservative photometric augmentation is geometry-preserving. It changes RGB
  intensities without moving pixels, point coordinates, camera geometry, robot
  state, or action labels.
- `visual_real_grid=64` is the maintained future-RGB supervision target for
  this profile. The historical `4x4` target is now treated as a diagnostic-only
  compatibility setting. Direct `256x256` linear prediction is not used on
  2x40GB because it creates a large output/error-head footprint without a
  decoder-style visual head.
- `diagnostic_visual_upscale` should be `4` with `visual_real_grid=64`; the
  trainer automatically converts the old `64` default to `4` to keep diagnostic
  videos near `256x256`.
- `persistent_anchors=8` and `observation_anchors=16` are the current slot
  budget for frozen-perception bring-up: two effector/contact recurrent slots,
  six scene/object recurrent slots, two effector observation anchors, and
  fourteen global scene observation anchors.
- A hard background slot is intentionally not reserved by default. Background
  evidence is represented through scene/object anchors plus task global and
  instruction tokens; reserving a background slot without explicit supervision
  would reduce object capacity.

## 3. Current Code Support

Already supported:

- `--perception-finetune-mode auto|full|frozen`
- `--visual-finetune-mode frozen`
- `--visual-feature-mode auto|hierarchical|final`
- FSDP root ignoring fully frozen subtrees
- frozen V-JEPA forward under autocast without hard-casting vendor weights
- `--semantic-max-length`
- `--picf-augmentation-mode off|photometric|multimodal_geometry`
- `--picf-photometric-strength conservative|reference`
- `picf_mode=enabled|ablated`

Operational notes:

- the maintained command snippets live in
  [`docs/CALVIN_VALIDATION_README.md Section 5.1A`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51a-current-2x40gb-frozen-perception-full-picf-profile)
- named shell wrappers are optional convenience only; the CLI contract itself is
  the maintained interface
- `state_only` burn-in is implemented and selected for the current sub-15s
  2x40GB long run with `burnin_steps=4`; full-BPTT `unroll_steps=3` remains the
  quality reference when runtime is acceptable
- the `unroll_steps=1` fast profile is a throughput profile, not a hidden
  replacement for recurrent full-BPTT training

Do not rely on implicit defaults for this profile. The launch log must print:

```text
point=sonata(trainable=False)
visual=encoder(finetune_mode=frozen trainable=False)
tactile=encoder(trainable=False)
semantic=paligemma(trainable=True)
semantic_max_length=256 primary / 200 memory fallback
visual_real_grid=64
persistent_anchors=8
observation_anchors=16
picf_augmentation_mode=photometric
picf_photometric_strength=conservative
```

## 4. V-JEPA2.1 Freeze Contract

V-JEPA2.1 is valuable because its dense patch features are spatially and
temporally coherent. Freezing it is safe only if freezing does not change what
PICF receives.

Current implementation fact:

- `VjepaVisualConfig.trainable=False` disables gradients.
- hierarchical output is controlled by `visual_feature_mode`, not implicitly by
  whether the backbone is trainable

Implemented contract:

```text
trainable controls gradients.
feature_mode controls output layout.
```

Current fields:

```text
perception_finetune_mode = auto | full | frozen
visual_finetune_mode = full | frozen
visual_feature_mode = hierarchical | final
```

For full PICF compatibility, the frozen profile should prefer:

```text
visual_finetune_mode=frozen
visual_feature_mode=hierarchical
```

This keeps the downstream PICF visual payload stable while saving backward
activation and optimizer memory.

## 5. Augmentation Classification

PICF cannot blindly inherit every PI0.5 image augmentation.

PI0.5 reference behavior:

- if no point cloud is present, train-time RGB augmentation may apply:
  - crop
  - resize
  - small rotation
  - color jitter
- if point clouds are present, image-only augmentation is disabled because it
  would desynchronize RGB image tokens from point/depth geometry.

PICF full mode has point, visual, tactile, proprio, and semantic streams tied to
one physical scene. Therefore augmentation must be classified by whether it
preserves the multimodal geometry contract.

### 5.1 Safe By Default

These can be introduced first. The repo-wide CLI default remains `off`, while
the maintained 2x40GB frozen-perception profile explicitly enables
conservative `photometric`:

- brightness jitter
- contrast jitter
- saturation jitter
- mild color channel jitter
- camera noise / sensor noise that does not move pixels geometrically
- modality dropout that drops an entire optional evidence stream while keeping
  the remaining streams unchanged

Important implementation rule:

- if Sonata consumes point colors, color jitter must be applied consistently to
  both RGB images and point colors derived from those images, or it must be
  restricted to semantic/V-JEPA image inputs only and documented as such.

Recommended first production option:

```text
picf_augmentation_mode=off | photometric
```

Conservative 2x40GB default:

```text
picf_augmentation_mode=photometric
picf_photometric_strength=conservative
```

Pure no-augmentation control runs should explicitly use:

```text
picf_augmentation_mode=off
```

### 5.2 Conditionally Safe

These require a shared sampled transform and explicit accounting:

- random crop
- random affine
- small rotation
- resize with nontrivial crop window

They are safe only if all affected geometry is transformed together:

- RGB images
- depth maps
- point-cloud generation or existing point coordinates
- camera intrinsics / projection metadata
- V-JEPA patch coordinate mapping
- any task-readout geometry summaries derived from those projections

Recommended future option:

```text
picf_augmentation_mode=multimodal_geometry
```

This should remain experimental until covered by tests that verify projection
consistency.

### 5.3 Not Safe As A Default

Do not enable these in full PICF by simply copying the no-point-cloud PI0.5
augmentation:

- RGB-only crop
- RGB-only rotation
- RGB-only geometric warp

These can make training loss look better while degrading physical alignment,
because posterior anchors and task readout no longer see geometrically
consistent evidence.

## 6. Proposed Augmentation API

Recommended CLI surface:

```text
--perception-finetune-mode auto|full|frozen
--picf-augmentation-mode off|photometric|multimodal_geometry
--picf-photometric-strength conservative|reference
```

Defaults:

```text
perception_finetune_mode=auto
picf_augmentation_mode=off
picf_photometric_strength=conservative
```

The implementation should centralize augmentation before modality-specific
encoding. It should not let PaliGemma, V-JEPA, Sonata, and AnyTouch each apply
uncoordinated local augmentations.

Current implementation status:

- `off` is a no-op and preserves the existing path
- `photometric` applies geometry-preserving RGB jitter to static and wrist RGB
  before point-cloud construction, so point colors stay consistent with the
  image evidence
- `multimodal_geometry` is a reserved fail-fast mode until synchronized
  RGB/depth/point/camera transforms exist

## 7. Validation Status And Remaining Work

Implemented and validated:

- CLI support for `perception_finetune_mode=frozen`
- CLI support for `semantic_max_length=256` with `200` as a memory/parity fallback
- CLI support for `picf_augmentation_mode=off|photometric|multimodal_geometry`
- fail-fast behavior for reserved `multimodal_geometry`
- tests proving that `off` preserves the existing path
- tests proving that photometric augmentation changes image intensities without
  changing point coordinates, robot state, action labels, or camera geometry
- cloud smoke tests for the 2x40GB frozen-perception path

Remaining work before promotion beyond the current profile:

- run CALVIN comparisons for photometric-on versus photometric-off
- run CALVIN comparisons for full-BPTT `unroll_steps=3` versus
  `state_only` burn-in
- implement synchronized multimodal geometry transforms only if a future
  experiment needs crop/rotation and can transform RGB, depth, points, camera
  intrinsics, and patch coordinate metadata together

## 8. Acceptance Tests

Required tests before enabling any non-off augmentation in long runs:

- `picf_augmentation_mode=off` preserves existing training batch tensor shapes.
- `picf_augmentation_mode=off` preserves current semantic prompt/action loss
  contracts.
- `photometric` changes image intensities but not image shape.
- `photometric` does not change:
  - `robot_obs`
  - `proprio`
  - action labels
  - point coordinates
  - depth-derived geometry
  - camera intrinsics
- frozen V-JEPA mode keeps the downstream feature shape expected by the PICF
  core.
- frozen Sonata and AnyTouch parameters are absent from the optimizer.
- the launch log prints every major module's trainable/frozen status.

## 9. Operational Recommendation

For the current selected 2x40GB full-PICF experiment, use:

```text
frozen perception + conservative photometric augmentation + semantic_max_length=256
unroll_steps=1 + burnin_steps=4 + burnin_mode=state_only
action_horizon=16 + save_interval=5000
visual_real_grid=64 + persistent_anchors=8 + observation_anchors=16
```

The canonical command is maintained in
[`docs/CALVIN_VALIDATION_README.md Section 5.1A`](/home/siyuanyue/Documents/openpi/docs/CALVIN_VALIDATION_README.md#51a-current-2x40gb-frozen-perception-full-picf-profile).
The short skeleton below is only a shape reference:

```bash
cd /root/openpi_posterior_vla_clean
export PYTHONPATH=/root/openpi_posterior_vla_clean/src
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/openpi/.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --exp-name picf_frozen_perception_2x40_smoke \
  --overwrite \
  --device cuda \
  --picf-mode enabled \
  --use-foundation-backbones \
  --perception-finetune-mode frozen \
  --training-strategy fsdp_full_shard \
  --optimizer-sharding none \
  --accum-steps 1 \
  --unroll-steps 1 \
  --burnin-steps 4 \
  --burnin-mode state_only \
  --action-horizon 16 \
  --semantic-max-length 256 \
  --persistent-anchors 8 \
  --observation-anchors 16 \
  --visual-real-grid 64 \
  --diagnostic-visual-upscale 4 \
  --visual-feature-mode hierarchical \
  --picf-augmentation-mode photometric \
  --picf-photometric-strength conservative \
  --num-train-steps 10 \
  --save-interval 10 \
  --log-interval 1 \
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --visual-activation-checkpointing
```

Then run a short smoke:

```text
num_train_steps=10
save_interval=10
log_interval=1
```

Only after memory, speed, and loss are stable should `photometric` augmentation
be compared against `off` as a controlled quality experiment. Full-BPTT
`unroll_steps=3` remains the quality reference when runtime is acceptable, but
the selected 2x40GB runtime profile is the `burnin_steps=4` state-only burn-in
compromise because it preserves full PICF on the trainable suffix while staying
near the sub-15s target.
