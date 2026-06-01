#!/usr/bin/env bash
set -euo pipefail

# E13 LR-continuity repair.
#
# E10/E11/E12 proved that structure health, semantic LR scale, suffix-vs-prefix
# context topology, and dense action-context tokens do not explain the remaining
# action plateau.  The decisive mismatch is the resumed learning-rate boundary:
# the source prefix_fusion run was already near min LR at step9100
#   policy ~= 2.005e-5, semantic ~= 7.02e-6,
# while the h30k resume diagnostics restarted at
#   policy ~= 5.94e-5, semantic ~= 2.08e-5.
#
# This run preserves the maintained prefix_fusion action interface but restores
# LR continuity from the source checkpoint.  LR and MIN_LR are intentionally
# equal so the resumed run does not reinterpret the already-trained checkpoint
# as an early high-LR point on a fresh 30K cosine schedule.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"

# Match the source checkpoint's actual step9100 LR scale.
export LR="${LR:-2.0e-5}"
export MIN_LR="${MIN_LR:-2.0e-5}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"

export ACTION_CONTEXT_TOKENS="${ACTION_CONTEXT_TOKENS:-24}"
export ACTION_CONTEXT_INTEGRATION="${ACTION_CONTEXT_INTEGRATION:-prefix_fusion}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec "${SCRIPT_DIR}/run_a7_stepindexed_from9100_suffixadapter_backbone_300_20260531.sh"
