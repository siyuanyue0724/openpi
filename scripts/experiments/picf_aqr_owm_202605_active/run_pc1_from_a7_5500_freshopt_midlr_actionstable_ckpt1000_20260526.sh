#!/usr/bin/env bash
set -euo pipefail

# Current-phase action-stability probe from the maintained A7 line.
#
# Purpose:
#   Stop the low-LR PC1 control and test a less underpowered phase-boundary
#   continuation from the cleaner A7 step5500 model weights.
#
# Contract:
#   - resume model weights only from A7 step5500;
#   - reset optimizer/scheduler state;
#   - use a moderate LR below A7's 5e-5 line but above the stopped 2e-5 control;
#   - keep action pressure at the maintained action-dominant scale;
#   - keep object scaffold at the already-decayed weak floor;
#   - save every 1000 steps and retain five checkpoints for phase inspection.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_2500fresh_20260525}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export EXP="${EXP:-picf_pc1_from_a7_5500_freshopt_midlr_actionstable_ckpt1000_20260526}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_from_a5_1500_freshopt_actionpolish_30k_20260524/5500}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.25}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"

export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
export ACTION_PREFIX_RMS_TARGET="${ACTION_PREFIX_RMS_TARGET:-1.0}"
export ACTION_PREFIX_NORM_EPS="${ACTION_PREFIX_NORM_EPS:-1e-6}"
export ACTION_PREFIX_VALUE_CLIP="${ACTION_PREFIX_VALUE_CLIP:-0.0}"

# Middle LR control:
#   A7 5e-5 keeps action moving but has a wider 0.04x rebound band.
#   PC1 2e-5 is structurally safe but can become underpowered after release.
#   3.5e-5 is the next causal test before adding new machinery.
export LR="${LR:-3.5e-5}"
export MIN_LR="${MIN_LR:-1.4e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

export OBJECT_SCAFFOLD_DECAY_MODE="${OBJECT_SCAFFOLD_DECAY_MODE:-cosine}"
export OBJECT_SCAFFOLD_DECAY_START_STEP="${OBJECT_SCAFFOLD_DECAY_START_STEP:-0}"
export OBJECT_SCAFFOLD_DECAY_END_STEP="${OBJECT_SCAFFOLD_DECAY_END_STEP:-1500}"
export OBJECT_SCAFFOLD_DECAY_FLOOR="${OBJECT_SCAFFOLD_DECAY_FLOOR:-0.03}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER:-2.0}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN:-0.05}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M:-0.08}"

if [[ -z "${SEGMENTS:-}" ]]; then
  SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
  if [[ ! -f "${SEGMENT_FILE}" ]]; then
    echo "Missing sidecar segment file: ${SEGMENT_FILE}" >&2
    exit 2
  fi
  SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
  export SEGMENTS
fi

exec "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
