#!/usr/bin/env bash
set -euo pipefail

# E14 maintained resume contract.
#
# E10/E11/E12 showed that suffix-vs-prefix topology and dense action-context
# tokens are not the main cause of the post-9100 action gap. E13 restored the
# source LR scale and partially improved action, but did not recover the source
# 0.044 band. The remaining confirmed discontinuity is optimizer state:
# source /9100 was saved as model-only, and every h30k diagnostic resumed with
# fresh Adam moments.
#
# This script is the future-proof continuation contract. It cannot recreate the
# missing source Adam state, but every checkpoint it writes is full optimizer
# state so the next phase/resume is a real continuation, not a fresh-optimizer
# experiment accidentally compared as if it were continuous.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_stepindexed_from9100_lrcontinuity_fullopt_prefixfusion_h30k_20260531}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"

# Match source step9100 LR scale instead of restarting the checkpoint on a high
# early-30K cosine LR.
export LR="${LR:-2.0e-5}"
export MIN_LR="${MIN_LR:-2.0e-5}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"

export ACTION_CONTEXT_TOKENS="${ACTION_CONTEXT_TOKENS:-24}"
export ACTION_CONTEXT_INTEGRATION="${ACTION_CONTEXT_INTEGRATION:-prefix_fusion}"

# Full optimizer checkpoints are intentionally heavier than model-only export
# checkpoints. Save them at phase-scale intervals by default; override this for
# short diagnostics only when disk/time cost is acceptable.
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

# Critical maintained change: full optimizer state at saved checkpoints.
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-full}"

exec "${SCRIPT_DIR}/run_a7_stepindexed_from9100_suffixadapter_backbone_300_20260531.sh"
