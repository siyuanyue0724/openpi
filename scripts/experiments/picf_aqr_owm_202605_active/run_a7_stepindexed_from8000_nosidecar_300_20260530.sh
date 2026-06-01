#!/usr/bin/env bash
set -euo pipefail

# Deprecated launcher: use
# run_a7_stepindexed_from8000_nosidecar_h30k_20260530.sh instead.
#
# This file is intentionally retained only as an audit breadcrumb for the
# invalid first launch.  It should not be used for future comparisons because
# NUM_TRAIN_STEPS=8300 changes the LR schedule at resumed step8000.
#
# Historical sidecar/noise ablation from the preserved step8000 basin.
#
# Purpose:
#   Test whether contact-motion sidecar proposal/tracklet evidence is the
#   dominant action-platform noise source.  This keeps model/optimizer/action
#   settings identical to the maintained production continuation and removes
#   only the external sidecar fields by pointing mvtrack_sidecar_root at an
#   intentionally empty directory.
#
# Important:
#   SEGMENTS are still read from the clean full sidecar root so the sampled
#   CALVIN segment set is identical to the production run.  Only optional
#   tracklet/proposal arrays disappear.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_probe_current_20260529}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export EXP="${EXP:-picf_a7_stepindexed_from8000_nosidecar_action2_300_20260530}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000}"

FULL_SIDECAR_ROOT="${FULL_SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"
EMPTY_SIDECAR_ROOT="${EMPTY_SIDECAR_ROOT:-/mnt/picf_sidecars/empty_no_sidecar_20260530}"
mkdir -p "${EMPTY_SIDECAR_ROOT}"

if [[ -z "${SEGMENTS:-}" ]]; then
  SEGMENT_FILE="${FULL_SIDECAR_ROOT}/calvin_segment_indices.txt"
  if [[ ! -f "${SEGMENT_FILE}" ]]; then
    echo "Missing full sidecar segment file: ${SEGMENT_FILE}" >&2
    exit 2
  fi
  SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
  export SEGMENTS
fi

export SIDECAR_ROOT="${SIDECAR_ROOT:-${EMPTY_SIDECAR_ROOT}}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-8300}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.005}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-backbone_only}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"

exec "${SCRIPT_DIR}/run_a7_stepindexed_fixedwindow_from7000_30k_20260529.sh"
