#!/usr/bin/env bash
set -euo pipefail

# Action-dominant continuation with weak-object-scaffold quality/decay.
#
# Intended handoff:
#   1. Let the action-2.0-from-step0 run save a clean step-500 checkpoint.
#   2. Resume from that checkpoint.
#   3. Keep action at the legacy/default-equivalent scale.
#   4. Decay contact-motion/tracklet sidecar scaffold from bootstrap teacher
#      into weak shaping evidence so it cannot dominate the long-run objective.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_EXP="${SOURCE_EXP:-picf_a7_actionaware_defaultsync_action2_from0_long30k_20260522}"
SOURCE_STEP="${SOURCE_STEP:-500}"

export EXP="${EXP:-picf_a7_actionaware_qgdecay_from500_long30k_20260522}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/${SOURCE_EXP}/${SOURCE_STEP}}"

if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
  echo "Missing resume checkpoint directory: ${RESUME_CHECKPOINT}" >&2
  echo "Wait for the source run to save step ${SOURCE_STEP}, or set SOURCE_EXP/SOURCE_STEP/RESUME_CHECKPOINT." >&2
  exit 2
fi

export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

# Production curriculum:
# - existing sidecar scaffold was useful for early ownership bootstrap;
# - after step 500 it must decay to a weak teacher, not compete with action;
# - cosine avoids an abrupt objective jump on resume.
export OBJECT_SCAFFOLD_DECAY_MODE="${OBJECT_SCAFFOLD_DECAY_MODE:-cosine}"
export OBJECT_SCAFFOLD_DECAY_START_STEP="${OBJECT_SCAFFOLD_DECAY_START_STEP:-500}"
export OBJECT_SCAFFOLD_DECAY_END_STEP="${OBJECT_SCAFFOLD_DECAY_END_STEP:-1500}"
export OBJECT_SCAFFOLD_DECAY_FLOOR="${OBJECT_SCAFFOLD_DECAY_FLOOR:-0.10}"

# Stronger target-quality curvature keeps diffuse/noisy sidecar masks as weak
# evidence while retaining compact contact-motion targets.
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER:-2.0}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN:-0.05}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M:-0.08}"

exec "${SCRIPT_DIR}/run_a7_actionaware_defaultsync_long30k_ckpt500_20260521.sh"
