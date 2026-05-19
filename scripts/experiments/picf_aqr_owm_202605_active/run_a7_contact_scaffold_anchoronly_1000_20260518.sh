#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/openpi_posterior_vla_clean}"
cd "${REPO_ROOT}"

EXP="${EXP:-picf_a7_contact_refseed_anchoronly_1000_20260518}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_causal_1000_20260518}"
LOG="/mnt/picf_run_logs/${EXP}.log"
CALVIN_ROOT="${CALVIN_ROOT:-/mnt/calvin_data/task_ABC_D}"
PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

mkdir -p /mnt/picf_run_logs /mnt/checkpoints/picf_core "${SIDECAR_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="src:${PYTHONPATH:-}"

if [ ! -f "${SIDECAR_ROOT}/manifest.json" ] || [ "${FORCE_SIDECAR:-0}" = "1" ]; then
  "${PYTHON_BIN}" scripts/picf_contact_motion_sidecar_precompute.py \
    --calvin-root "${CALVIN_ROOT}" \
    --output-root "${SIDECAR_ROOT}" \
    --split training \
    --target-frames 1000 \
    --max-frames-per-segment 96 \
    --static-stride 4 \
    --gripper-stride 2 \
    --top-fraction 0.015 \
    --min-top-points 24 \
    --min-score 0.015 \
    --box-pad-px 4.0 \
    --max-proposals-per-frame 3 \
    --component-radius-px 10.0 \
    --component-min-points 6 \
    --box-percentile-low 12.0 \
    --box-percentile-high 88.0 \
    --mask-samples-per-proposal 96 \
    --source-id 8 \
    --preview-count 48
fi

SEGMENTS="$(tr -d '[:space:]' < "${SIDECAR_ROOT}/calvin_segment_indices.txt")"
if [ -z "${SEGMENTS}" ]; then
  echo "No generated segment indices found under ${SIDECAR_ROOT}" >&2
  exit 1
fi

if [ ! -f "${SIDECAR_ROOT}/tracklet_manifest_training.json" ] || [ "${FORCE_TRACKLETS:-0}" = "1" ]; then
  "${PYTHON_BIN}" scripts/picf_tracklet_sidecar_precompute.py \
    --calvin-root "${CALVIN_ROOT}" \
    --backend dir \
    --split training \
    --proposal-root "${SIDECAR_ROOT}" \
    --output-root "${SIDECAR_ROOT}" \
    --views both \
    --segment-indices "${SEGMENTS}" \
    --keyframe-stride 8 \
    --window-forward 15 \
    --seeds-per-view 24 \
    --proposal-seed-fraction 0.75 \
    --require-proposal-keyframe \
    --max-tracklets-per-frame 96 \
    --klt-max-error-px 18.0 \
    --confidence-decay 0.985 \
    --no-skip-existing-tracklets \
    --log-every 5
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/picf_core_train.py \
  --calvin-root "${CALVIN_ROOT}" \
  --backend dir \
  --split training \
  --calvin-segment-indices "${SEGMENTS}" \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --use-foundation-backbones \
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --tactile-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --tactile-backgrounds-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_backgrounds.npz \
  --tactile-calibration-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_fingertip_calibration.json \
  --tactile-contact-stats-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_contact_stats.json \
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --semantic-checkpoint-path /root/openpi/checkpoints/foundation/pi05_base_pytorch \
  --training-strategy ddp \
  --optimizer-sharding none \
  --optimizer-checkpoint-mode model-only \
  --perception-finetune-mode frozen \
  --picf-trainable-scope anchor_only \
  --action-horizon 16 \
  --max-points 1024 \
  --accum-steps 1 \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --warmup-steps 50 \
  --weight-decay 1e-4 \
  --unroll-steps 2 \
  --burnin-steps 1 \
  --burnin-mode state_only \
  --lambda-action-pos 0.50 \
  --lambda-action-rot 0.50 \
  --lambda-action-gripper 0.50 \
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
  --proposal-memory-enabled \
  --proposal-max-tokens 16 \
  --proposal-confidence-floor 0.01 \
  --proposal-read-weight 0.10 \
  --proposal-age-decay-steps 6.0 \
  --proposal-point-bridge-weight 0.15 \
  --proposal-point-bridge-edge-tau 0.02 \
  --proposal-anchor-seed-enabled \
  --proposal-anchor-seed-rows 2 \
  --proposal-anchor-seed-weight 0.75 \
  --proposal-anchor-seed-token-weight 0.20 \
  --proposal-anchor-seed-score-floor 0.01 \
  --proposal-anchor-seed-point-topk 96 \
  --proposal-anchor-seed-point-power 1.75 \
  --task-owner-bias-enabled \
  --task-owner-proposal-bias-weight 0.15 \
  --task-owner-proposal-point-bias-weight 0.25 \
  --task-owner-proposal-point-bridge-weight 0.15 \
  --task-owner-proposal-objectness-power 0.50 \
  --task-owner-proposal-static-only \
  --task-owner-proposal-topk 2 \
  --task-owner-proposal-score-floor 0.01 \
  --mvtrack-sidecar-root "${SIDECAR_ROOT}" \
  --mvtrack-sidecar-proposal-nearest-max-gap 3 \
  --tracklet-memory-enabled \
  --tracklet-max-tokens 96 \
  --tracklet-confidence-floor 0.05 \
  --tracklet-read-weight 0.15 \
  --aqr-ownership-prior-enabled \
  --aqr-ownership-prior-weight 0.70 \
  --aqr-ownership-point-prior-weight 0.70 \
  --aqr-ownership-point-prior-sigma-m 0.04 \
  --aqr-ownership-temporal-prior-weight 0.35 \
  --aqr-ownership-prior-uniform-mix 0.02 \
  --aqr-same-role-support-competition-enabled \
  --aqr-same-role-support-competition-weight 0.85 \
  --aqr-same-role-support-competition-iters 5 \
  --posterior-file-competition-enabled \
  --posterior-file-competition-min-per-role 1 \
  --posterior-file-competition-max-per-role 0 \
  --posterior-file-competition-min-support 0.02 \
  --posterior-file-competition-relative-score-threshold 0.0 \
  --posterior-file-competition-support-overlap-threshold 0.80 \
  --posterior-file-competition-geometry-duplicate-enabled \
  --posterior-file-competition-geometry-sigma-m 0.04 \
  --posterior-file-competition-geometry-threshold 0.70 \
  --posterior-birth-competition-enabled \
  --posterior-birth-competition-max-per-role 1 \
  --posterior-birth-competition-min-score 0.05 \
  --posterior-birth-competition-inactive-only \
  --posterior-birth-alpha-suppression-power 0.5 \
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
  --aqr-context-slot-enabled \
  --aqr-context-slot-weight 0.15 \
  --aqr-context-slot-min-confidence 0.05 \
  --aqr-context-slot-min-score 0.01 \
  --aqr-context-slot-duplicate-overlap-threshold 0.75 \
  --grad-clip-mode fixed \
  --grad-clip-norm 5.0 \
  --anchor-overlay-interval 100 \
  --anchor-overlay-max-anchors 64 \
  --anchor-overlay-dump-signatures \
  --log-interval 50 \
  --save-interval 1000 \
  --keep-last-checkpoints 1 \
  --progress \
  --wandb-mode disabled \
  --overwrite \
  --exp-name "${EXP}" \
  --num-train-steps 1000 2>&1 | tee "${LOG}"
