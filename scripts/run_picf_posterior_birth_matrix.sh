#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-a5}"
REPO_ROOT="${PICF_REPO_ROOT:-$(pwd)}"
cd "${REPO_ROOT}"

export PYTHONPATH="src:${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/openpi/.venv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
UV_BIN="${UV_BIN:-}"
if [[ -z "${UV_BIN}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  elif [[ -x /root/.local/bin/uv ]]; then
    UV_BIN=/root/.local/bin/uv
  else
    echo "[posterior-birth-matrix] ERROR: uv not found; set UV_BIN=/path/to/uv" >&2
    exit 127
  fi
fi

BASE="${PICF_CHECKPOINT_BASE:-/mnt/checkpoints/picf_core}"
LOG_BASE="${PICF_LOG_BASE:-/mnt/picf_run_logs}"
RUN_TAG="${PICF_RUN_TAG:-$(git rev-parse --short HEAD)}"
mkdir -p "${LOG_BASE}"

COMMON=(
  --calvin-root /mnt/calvin_data/task_ABC_D
  --backend dir
  --split training
  --checkpoint-base-dir "${BASE}"
  --use-foundation-backbones
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt
  --tactile-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth
  --tactile-backgrounds-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_backgrounds.npz
  --tactile-calibration-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_fingertip_calibration.json
  --tactile-contact-stats-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_contact_stats.json
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth
  --semantic-checkpoint-path /root/openpi/checkpoints/foundation/pi05_base_pytorch
  --training-strategy fsdp_full_shard
  --optimizer-sharding none
  --optimizer-checkpoint-mode model-only
  --perception-finetune-mode frozen
  --action-horizon 16
  --max-points 1024
  --accum-steps 1
  --lr 2e-4
  --min-lr 2e-5
  --warmup-steps 100
  --weight-decay 1e-4
  --semantic-lr-scale 0.25
  --lambda-anchor-pv 0.25
  --lambda-mapg-cycle 0.05
  --lambda-mapg-support-diversity 0.05
  --lambda-mapg-geometry-diversity 0.02
  --lambda-slot-jepa 0.0
  --lambda-support-pred 0.0
  --lambda-binding-consistency 0.0
  --lambda-aqr-denoising 0.0
  --posterior-slot-identity-std 0.02
  --task-slot-identity-std 0.02
  --posterior-bootstrap-from-observation
  --posterior-occupancy-prior-enabled
  --posterior-occupancy-prior-weight 1.0
  --posterior-occupancy-prior-sigma-m 0.04
  --posterior-occupancy-prior-clip 4.0
  --observation-anchor-seed-point-mix 0.35
  --recycle-normalize-residual-summary
  --recycle-residual-norm-mode layernorm
  --posterior-slotwise-recycle-residual
  --no-legacy-local-refinement-opt-in
  --no-local-refinement-enabled
  --local-refinement-topk 0
  --local-refinement-weight 0.0
  --local-refinement-binding-weight 0.0
  --aqr-active-slot-filter-enabled
  --aqr-active-slot-min-per-role 1
  --aqr-active-slot-min-confidence 0.05
  --anchor-overlay-interval 100
  --anchor-overlay-max-anchors 64
  --log-interval 50
  --save-interval 300
  --keep-last-checkpoints 2
  --progress
  --wandb-mode disabled
  --overwrite
)

run_one() {
  local exp="$1"
  local steps="$2"
  local scope="$3"
  local unroll="$4"
  local burnin="$5"
  local action_weight="$6"
  local active_max="$7"
  local active_overlap="$8"
  local log="${LOG_BASE}/${exp}.log"
  {
    echo "[posterior-birth-matrix] exp=${exp} start $(date -Is)"
    echo "[posterior-birth-matrix] repo=$(pwd) commit=$(git rev-parse --short HEAD)"
    echo "[posterior-birth-matrix] scope=${scope} steps=${steps} unroll=${unroll} burnin=${burnin} action=${action_weight} active_max=${active_max} overlap=${active_overlap}"
    "${UV_BIN}" run --no-sync --project . torchrun --standalone --nproc_per_node=2 scripts/picf_core_train.py \
      "${COMMON[@]}" \
      --exp-name "${exp}" \
      --num-train-steps "${steps}" \
      --picf-trainable-scope "${scope}" \
      --unroll-steps "${unroll}" \
      --burnin-steps "${burnin}" \
      --burnin-mode state_only \
      --lambda-action-pos "${action_weight}" \
      --lambda-action-rot "${action_weight}" \
      --lambda-action-gripper "${action_weight}" \
      --aqr-active-slot-max-per-role "${active_max}" \
      --aqr-active-slot-overlap-threshold "${active_overlap}"
    echo "[posterior-birth-matrix] exp=${exp} done $(date -Is)"
  } 2>&1 | tee -a "${log}"
}

case "${PROFILE}" in
  a5)
    run_one "picf_a5_birth_anchor_u2b1_a025_450_${RUN_TAG}" 450 anchor_only 2 1 0.25 6 0.70
    run_one "picf_a5_birth_cotrain_u2b1_a025_450_${RUN_TAG}" 450 all 2 1 0.25 6 0.70
    run_one "picf_a5_birth_cotrain_u2b1_a05_450_${RUN_TAG}" 450 all 2 1 0.50 6 0.70
    ;;
  a5_cotrain)
    run_one "picf_a5_birth_cotrain_u2b1_a025_450_${RUN_TAG}" 450 all 2 1 0.25 6 0.70
    run_one "picf_a5_birth_cotrain_u2b1_a05_450_${RUN_TAG}" 450 all 2 1 0.50 6 0.70
    ;;
  a7)
    run_one "picf_a7_birth_cotrain_u2b1_a025_600_${RUN_TAG}" 600 all 2 1 0.25 6 0.70
    run_one "picf_a7_birth_cotrain_u2b1_a05_600_${RUN_TAG}" 600 all 2 1 0.50 6 0.70
    ;;
  a7_fast)
    run_one "picf_a7_birth_cotrain_u2b1_a025_300_${RUN_TAG}" 300 all 2 1 0.25 6 0.70
    run_one "picf_a7_birth_cotrain_u2b1_a05_300_${RUN_TAG}" 300 all 2 1 0.50 6 0.70
    ;;
  *)
    echo "unknown profile: ${PROFILE}" >&2
    exit 2
    ;;
esac
