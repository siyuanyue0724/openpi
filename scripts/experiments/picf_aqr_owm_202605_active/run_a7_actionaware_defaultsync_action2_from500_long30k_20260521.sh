#!/usr/bin/env bash
set -euo pipefail

# Action-dominant continuation from the default-sync step-500 checkpoint.
#
# This is a deliberately stronger action-pressure stage, matching the legacy
# default-equivalent action scale recorded in README_v2.2.  It should only be
# started after the 0.50 stage reaches and saves step 500 with healthy active
# owner metrics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_EXP="${SOURCE_EXP:-picf_a7_actionaware_defaultsync_long30k_ckpt500_20260521}"
SOURCE_STEP="${SOURCE_STEP:-500}"

export EXP="${EXP:-picf_a7_actionaware_defaultsync_action2_from500_long30k_20260521}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/${SOURCE_EXP}/${SOURCE_STEP}}"

if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
  echo "Missing resume checkpoint directory: ${RESUME_CHECKPOINT}" >&2
  exit 2
fi

export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec "${SCRIPT_DIR}/run_a7_actionaware_defaultsync_long30k_ckpt500_20260521.sh"
