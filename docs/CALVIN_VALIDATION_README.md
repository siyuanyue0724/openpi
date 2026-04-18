# CALVIN Validation README

This document is the current executable validation guide for CALVIN under the
present PICF codebase.

The canonical architecture and deployment handoff is:

- [`README_v2.2.md`](/home/siyuanyue/Documents/openpi/src/openpi/picf/README_v2.2.md)

The formal contract is:

- [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)

Important status note:

- `README_v2.2.md` is the current local architecture and deployment document
- `PICF_FORMAL_CONTRACT.md` records the current live-code contract that existing
  regression tests enforce
- `README_v2.1.md` is retained only as the archived pre-v2.2 deployment record
- the current active document set is:
  - `src/openpi/picf/README_v2.2.md`
  - `PICF_FORMAL_CONTRACT.md`
  - `docs/CALVIN_VALIDATION_README.md`

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

Current live-code contract meaning:

- semantic wrapper restores the PI0.5 expert path and injects robot state back
  into the prompt path
- trainer primary action loss uses PI0.5 flow matching, and the training-time
  denoised chunk estimate is recovered as `x_t - t * v_t`
- serving primary action path uses the PI0.5 denoise sampler and refreshes the
  predictive cache with the sampled action chunk
- semantic does not affect physical observation anchors
- semantic does not affect physical posterior
- semantic does not affect `physical_prediction_cache`
- next innovation reads only `previous.predictive.physical_prediction_cache`
- semantic enters the core only through current-step task readout
- public task-readout memory is `fused_tokens + visual_tokens`
- task readout rereads private dense visual / tactile / point payloads
- the core now builds one canonical conditioned control state `C_t`
- PI0.5 action conditioning uses `C_t^{pi}` derived from `C_t`
- conditioned future now uses token-level physical predictive tokens plus
  future-condition tokens, not the raw semantic prefix
- live observation competition now uses native-first visual routing
- live tactile public routing now uses group-level proposal competition with
  winner-read over dense tactile group memory
- live visual innovation targets include denser native-V-JEPA latent probes
- live tactile innovation targets include dense tactile latent probes in
  addition to tactile map and auxiliaries
- live point innovation targets include native point latent probes in addition
  to occupancy
- the physical core remains width `512`, while the conditioned control/future
  trunks remain semantic-width `2048`
- physical posterior / innovation / physical predictive tokens are up-projected
  into the semantic-width conditioned-control / conditioned-future trunks
- `posterior.global_post` explicitly enters conditioned control
- core no longer uses a direct trainable `7D` action head

This section describes the deployed live baseline.

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

## 5. Historical Baseline Training Command

The command below is preserved as a historical baseline from the pre-v2.2
semantic-prefix-primary mixed-width implementation.

It is **not** the recommended canonical launch command for current deployment.

Historical command:

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

This is a preserved historical launch path from before the fully restored
PI0.5-action-stack deployment.

Current v2.2 long-run training profile:

- `--num-train-steps 30000`
- `--save-interval 2500`
- `--grad-clip-mode percentile`
- `--grad-clip-percentile 75`
- `--grad-clip-window 100`
- `--training-strategy fsdp_full_shard` for the standard 4x40GB A100 full-finetune profile
- `--optimizer-sharding none` on that FSDP path; `zero1` remains a DDP-only fallback and is not sufficient for all-backbone v2.2 finetuning

Distributed runtime note for current multi-rank bring-up:

- do not use `TORCH_DISTRIBUTED_DEBUG=DETAIL` as the default 4-GPU training
  setting
- current `scripts/picf_core_train.py` defaults DDP startup to
  `TORCH_DISTRIBUTED_DEBUG=INFO`
- if the environment injects `TORCH_DISTRIBUTED_DEBUG=DETAIL`, the trainer
  fails fast unless
  `OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL=1` is explicitly set
- reserve DETAIL for targeted distributed-runtime debugging only

## 6. Serving / Rollout

After training:

- keep the normalized action contract in trainer / serving aligned
- use the standard serving entrypoint
- then run CALVIN evaluator rollouts from the served checkpoint

The serving path must preserve:

- normalized action inside the core
- unnormalized action at environment execution

## 7. Historical Notes

Older CALVIN README fragments have been retired.
Use this file for the current validation workflow.
