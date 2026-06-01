#!/usr/bin/env bash
set -euo pipefail

# Production-length action-aware validation after the 2026-05-21
# default-sync frozen-policy gate.
#
# This wrapper preserves the exact slot/OEML/owner/context contract from the
# validated comprehensive profile.  It only turns on PaliGemma/action pressure,
# uses the clean full sidecar root, and increases checkpoint cadence so early
# action plateaus can be resumed from without changing training math.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_actionaware_defaultsync_long30k_ckpt500_20260521}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Production action-aware profile: frozen V-JEPA/Sonata/AnyTouch, trainable
# PaliGemma/PICF adapters/action-side heads.
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.25}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-0.50}"

# Frequent early checkpoints for plateau intervention.  Six retained numeric
# checkpoints cover the latest 1500 training steps without growing /mnt
# unboundedly.
export SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

if [[ -z "${SEGMENTS:-}" ]]; then
  SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
  if [[ ! -f "${SEGMENT_FILE}" ]]; then
    echo "Missing sidecar segment file: ${SEGMENT_FILE}" >&2
    echo "Run scripts/picf_prepare_full_sidecar_root.py after full sidecar generation." >&2
    exit 2
  fi
  SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
  export SEGMENTS
fi

exec "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
