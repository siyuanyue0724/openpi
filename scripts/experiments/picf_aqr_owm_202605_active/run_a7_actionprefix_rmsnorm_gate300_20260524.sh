#!/usr/bin/env bash
set -euo pipefail

# Action-prefix stability gate.
#
# Purpose:
#   Validate that the PICF-to-action prefix stays numerically stationary before
#   restarting a production-length run.  This keeps the validated slot/OEML/
#   owner/context profile and changes only the action-prefix interface contract:
#   per-token RMS normalization to a fixed target before tokens enter the
#   PI0.5 action path.
#
# Acceptance at 100/200/300 steps:
#   - no NaN/nonfinite prefix/action values;
#   - loss_action_default_equiv decreases or stays in the early-good band;
#   - pi_prefix_post_rms_mean stays close to ACTION_PREFIX_RMS_TARGET;
#   - posterior_recycle_rate does not re-enter the late-run oscillatory failure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_actionprefix_rmsnorm_gate300_20260524}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-300}"
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

export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

# Keep scaffold as a weak bootstrap signal for this short gate.  The long run
# uses the same interface normalization and may decay scaffold once the prefix
# metrics are proven stable.
export OBJECT_SCAFFOLD_DECAY_MODE="${OBJECT_SCAFFOLD_DECAY_MODE:-cosine}"
export OBJECT_SCAFFOLD_DECAY_START_STEP="${OBJECT_SCAFFOLD_DECAY_START_STEP:-0}"
export OBJECT_SCAFFOLD_DECAY_END_STEP="${OBJECT_SCAFFOLD_DECAY_END_STEP:-300}"
export OBJECT_SCAFFOLD_DECAY_FLOOR="${OBJECT_SCAFFOLD_DECAY_FLOOR:-0.03}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER:-2.0}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN:-0.05}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M:-0.08}"

if [[ -z "${SEGMENTS:-}" ]]; then
  SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
  if [[ -f "${SEGMENT_FILE}" ]]; then
    SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
    export SEGMENTS
  fi
fi

exec "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
