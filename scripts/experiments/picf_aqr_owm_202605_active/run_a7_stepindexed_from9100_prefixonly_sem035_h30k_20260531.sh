#!/usr/bin/env bash
set -euo pipefail

# E12 action-interface root-cause control.
#
# E10 showed that lowering semantic LR repairs the high-LR action degradation,
# but action remains in the E7/E8 band.  E11 showed that switching the same
# boundary back to prefix_fusion is not enough.  This run removes dense PICF
# action-context tokens entirely while keeping the same step9100 checkpoint,
# horizon, semantic LR, action prefix teacher, and trainable boundary.
#
# Purpose:
#   isolate whether dense action-context tokens are the remaining source of
#   action-path noise.  If action improves, retire dense action context from the
#   maintained action path and keep PICF/slot evidence as belief-state side
#   information.  If action remains flat, the remaining cause is the checkpoint
#   basin / stream / optimizer boundary rather than the action-context channel.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_stepindexed_from9100_prefixonly_sem035_h30k_20260531}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"

# The single intended intervention for E12.
export ACTION_CONTEXT_TOKENS="${ACTION_CONTEXT_TOKENS:-0}"
export ACTION_CONTEXT_INTEGRATION="${ACTION_CONTEXT_INTEGRATION:-prefix_fusion}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec "${SCRIPT_DIR}/run_a7_stepindexed_from9100_suffixadapter_backbone_300_20260531.sh"
