# PICF Frozen Perception And Augmentation Design

Date: 2026-04-30
Status: design record for the next 2x40GB / frozen-perception profile

This document records the intended engineering contract for two related topics:

- freezing large perception backbones without changing the PICF architecture
- adding train-time augmentation without breaking PICF's multimodal geometry

It is a design and operator contract. It does not claim that every switch below
is already implemented in the live CLI. When a switch is listed as "needed", it
must be added before treating the profile as production-ready.

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

Default 2x40GB recommendation:

```text
picf_mode=enabled
training_strategy=fsdp_full_shard
optimizer_sharding=none
world_size=2
accum_steps=1
unroll_steps=2
action_horizon=16
semantic_max_length=200
semantic_trainable=True
visual_finetune_mode=frozen
visual_feature_mode=hierarchical
point_backbone_trainable=False
tactile_trainable=False
picf_augmentation_mode=off
```

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

## 3. Current Code Support Versus Needed Cleanup

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

Needed for a clean profile:

- a named launcher/profile for `picf_frozen_perception_2x40`
- README/CALVIN command snippets once the CLI surface is finalized

Do not rely on implicit defaults for this profile. The launch log must print:

```text
point=sonata(trainable=False)
visual=encoder(finetune_mode=frozen trainable=False)
tactile=encoder(trainable=False)
semantic=paligemma(trainable=True)
semantic_max_length=200
picf_augmentation_mode=off
```

## 4. V-JEPA2.1 Freeze Contract

V-JEPA2.1 is valuable because its dense patch features are spatially and
temporally coherent. Freezing it is safe only if freezing does not change what
PICF receives.

Current implementation fact:

- `VjepaVisualConfig.trainable=False` disables gradients.
- The live wrapper currently derives hierarchical output behavior from
  `trainable`.

Required contract:

```text
trainable controls gradients.
feature_mode controls output layout.
```

Recommended future fields:

```text
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

These can be introduced first, but should still default to `off`:

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

Default:

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

## 7. Recommended Implementation Order

1. Add a named launcher/profile for `picf_frozen_perception_2x40`.
2. Change that 2x40GB frozen-perception profile to `semantic_max_length=200`.
3. Add tests proving that `off` is bitwise/shape equivalent to the current path.
4. Add tests proving that photometric augmentation does not alter point
   coordinates, robot state, action labels, or camera geometry.
5. Only then consider a synchronized multimodal geometry augmentation mode.

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

For the next 2x40GB experiment, use:

```text
frozen perception + no augmentation + semantic_max_length=200
```

Command skeleton:

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
  --unroll-steps 2 \
  --action-horizon 16 \
  --semantic-max-length 200 \
  --visual-feature-mode hierarchical \
  --picf-augmentation-mode off \
  --num-train-steps 10 \
  --save-interval 10 \
  --log-interval 1 \
  --grad-clip-mode percentile \
  --grad-clip-percentile 75 \
  --grad-clip-window 100 \
  --semantic-gradient-checkpointing \
  --visual-activation-checkpointing
```

Then run a short smoke:

```text
num_train_steps=10
save_interval=10
log_interval=1
```

Only after memory, speed, and loss are stable should `photometric` augmentation
be enabled as a separate controlled experiment.
