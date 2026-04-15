# CALVIN Validation README

This document is the current executable validation guide for CALVIN under the
present PICF semantic-prefix-primary mixed-width codebase.

The canonical architecture handoff is:

- [`README_semantic_prefix_primary_mixedwidth_refactor.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_semantic_prefix_primary_mixedwidth_refactor.md)

The formal contract is:

- [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)

## 1. Dataset / Loader Validation

Directory backend:

```bash
python scripts/stageb_calvin_audit.py \
  --mode dataset \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split validation
```

Loader validation:

```bash
python scripts/stageb_calvin_audit.py \
  --mode loader \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --split validation \
  --batch-size 4 \
  --num-workers 0
```

Use the zip backend only if you intentionally want to validate zip loading.

## 2. Contract Validation

Run this before training changes or deployment:

```bash
python scripts/verify_picf_contract.py
```

Current contract meaning:

- semantic does not affect physical observation anchors
- semantic does not affect physical posterior
- semantic does not affect `physical_prediction_cache`
- next innovation reads only `previous.predictive.physical_prediction_cache`
- control and conditioned future directly consume the full semantic prefix at
  native width `2048`
- the physical core remains width `512`
- physical posterior / innovation / physical predictive tokens are up-projected
  into the semantic-width trunks
- `posterior.global_post` explicitly enters control

## 3. Trainer Smoke Validation

CPU smoke:

```bash
python scripts/picf_core_train_smoke.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --segment-index 0 \
  --device cpu
```

Foundation-backbone CUDA smoke:

```bash
python scripts/picf_core_train_smoke.py \
  --calvin-root /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D \
  --backend dir \
  --segment-index 0 \
  --device cuda \
  --use-foundation-backbones \
  --use-tactile
```

## 4. Tactile Calibration

If calibration files are missing, regenerate them with the full supported
script instead of fabricating placeholders:

```bash
python scripts/calvin/precompute_tactile_contact_calibration.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --device cuda \
  --anytouch-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --output-dir /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8
```

Required outputs:

- `tactile_backgrounds.npz`
- `tactile_contact_stats.json`
- `tactile_fingertip_calibration.json`

## 5. Clean Training

The current intended training path is clean-start semantic-prefix-primary with:

- semantic width `2048`
- physical core width `512`

Do not use:

- sidecar flags
- sidecar rollout A/B logic
- old sidecar-primary checkpoints
- the obsolete full-core-2048 launch path

Recommended command:

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
  --exp-name picf_semantic_prefix_primary_mixedwidth_a4_acc1_pct75_clean_v1
```

This is the current clean-start launch path for the mixed-width refactor.

## 6. Serving / Rollout

After training:

- keep the normalized action contract in trainer / serving aligned
- use the standard serving entrypoint
- then run CALVIN evaluator rollouts from the served checkpoint

The serving path must preserve:

- normalized action inside the core
- unnormalized action at environment execution

## 7. Historical Notes

`docs/calvin_readme.txt` is now an archive only.
Use this file for the current validation workflow.
