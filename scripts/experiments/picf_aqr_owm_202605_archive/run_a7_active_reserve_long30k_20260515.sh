#!/usr/bin/env bash
set -euo pipefail

cd /root/openpi_posterior_vla_clean

EXP=picf_a7_active_reserve_cotrain_u2b1_a05_30000_20260515
LOG=/mnt/picf_run_logs/${EXP}.log
mkdir -p /mnt/picf_run_logs /mnt/checkpoints/picf_core
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:${PYTHONPATH:-}
PYTHON_BIN=${PYTHON_BIN:-/root/openpi/.venv/bin/python}

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/picf_core_train.py \
  --calvin-root /mnt/calvin_data/task_ABC_D \
  --backend dir \
  --split training \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --use-foundation-backbones \
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --tactile-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --tactile-backgrounds-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_backgrounds.npz \
  --tactile-calibration-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_fingertip_calibration.json \
  --tactile-contact-stats-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_contact_stats.json \
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --semantic-checkpoint-path /root/openpi/checkpoints/foundation/pi05_base_pytorch \
  --training-strategy fsdp_full_shard \
  --optimizer-sharding none \
  --optimizer-checkpoint-mode model-only \
  --perception-finetune-mode frozen \
  --picf-trainable-scope all \
  --action-horizon 16 \
  --max-points 1024 \
  --accum-steps 1 \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --warmup-steps 100 \
  --weight-decay 1e-4 \
  --semantic-lr-scale 0.25 \
  --unroll-steps 2 \
  --burnin-steps 1 \
  --burnin-mode state_only \
  --lambda-action-pos 0.50 \
  --lambda-action-rot 0.50 \
  --lambda-action-gripper 0.50 \
  --picf-action-prefix-stopgrad \
  --aux-budget-alignment-ratio 1.0 \
  --aux-budget-alignment-floor 2.0 \
  --lambda-anchor-pv 0.25 \
  --lambda-mapg-cycle 0.05 \
  --lambda-mapg-support-diversity 0.25 \
  --lambda-mapg-geometry-diversity 0.05 \
  --lambda-slot-jepa 0.0 \
  --lambda-support-pred 0.0 \
  --lambda-binding-consistency 0.0 \
  --lambda-aqr-denoising 0.0 \
  --aqr-ownership-prior-enabled \
  --aqr-ownership-prior-weight 0.70 \
  --aqr-ownership-point-prior-weight 0.70 \
  --aqr-ownership-point-prior-sigma-m 0.04 \
  --aqr-ownership-temporal-prior-weight 0.35 \
  --aqr-ownership-prior-uniform-mix 0.02 \
  --aqr-same-role-support-competition-enabled \
  --aqr-same-role-support-competition-weight 0.85 \
  --aqr-same-role-support-competition-iters 5 \
  --posterior-slot-identity-std 0.02 \
  --task-slot-identity-std 0.02 \
  --posterior-bootstrap-from-observation \
  --posterior-occupancy-prior-enabled \
  --posterior-occupancy-prior-weight 1.0 \
  --posterior-occupancy-prior-sigma-m 0.04 \
  --posterior-occupancy-prior-clip 4.0 \
  --observation-anchor-seed-point-mix 0.35 \
  --recycle-normalize-residual-summary \
  --recycle-residual-norm-mode layernorm \
  --posterior-slotwise-recycle-residual \
  --posterior-owner-active-gate-enabled \
  --posterior-owner-active-min 0.30 \
  --posterior-owner-active-bias -10000.0 \
  --no-legacy-local-refinement-opt-in \
  --no-local-refinement-enabled \
  --local-refinement-topk 0 \
  --local-refinement-weight 0.0 \
  --local-refinement-binding-weight 0.0 \
  --aqr-active-slot-filter-enabled \
  --aqr-active-slot-min-per-role 1 \
  --aqr-active-slot-max-per-role 2 \
  --aqr-active-slot-min-confidence 0.05 \
  --aqr-active-slot-overlap-threshold 0.40 \
  --aqr-active-slot-relative-score-threshold 0.80 \
  --aqr-active-slot-geometry-duplicate-enabled \
  --aqr-active-slot-geometry-duplicate-sigma-m 0.05 \
  --aqr-active-slot-geometry-duplicate-threshold 0.45 \
  --grad-clip-mode fixed \
  --grad-clip-norm 5.0 \
  --anchor-overlay-interval 50 \
  --anchor-overlay-max-anchors 64 \
  --log-interval 50 \
  --save-interval 2500 \
  --keep-last-checkpoints 3 \
  --progress \
  --wandb-mode disabled \
  --overwrite \
  --exp-name "$EXP" \
  --num-train-steps 30000 2>&1 | tee "$LOG"
