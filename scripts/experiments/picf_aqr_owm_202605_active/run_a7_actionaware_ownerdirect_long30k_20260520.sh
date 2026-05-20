#!/usr/bin/env bash
set -euo pipefail

# Production-relevant 30k action cotrain validation after the owner-direct
# frozen-policy smoke and action-aware smoke both passed their short gates.
#
# This wrapper intentionally does not change the slot/sidecar/OEML/owner
# transport contract.  It only lengthens the validated action-aware profile.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_actionaware_ownerdirect_long30k_20260520}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.25}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-0.50}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

if [[ -z "${SEGMENTS:-}" ]]; then
  SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
  if [[ ! -f "${SEGMENT_FILE}" ]]; then
    echo "Missing sidecar segment file: ${SEGMENT_FILE}" >&2
    echo "Run scripts/picf_prepare_full_sidecar_root.py after full sidecar generation." >&2
    exit 2
  fi
  export SEGMENTS
  SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
fi

exec "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
