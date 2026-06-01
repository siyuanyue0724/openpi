#!/usr/bin/env bash
set -euo pipefail

# E10 action-interface diagnostic / repair candidate.
#
# E9-h30k proved that `backbone_only + suffix_cross_attention` is mechanically
# live under the production 30K horizon, but `SEMANTIC_LR_SCALE=1.0` immediately
# worsened action loss while active/downstream overlap and slot losses improved.
# That isolates the next causal variable: high-LR semantic/action-backbone drift.
#
# This run keeps the same step9100 checkpoint, suffix-side PICF action adapter,
# policy-head LR, sidecars, unroll/burn-in, and 30K horizon, but returns the
# large semantic/action transformer LR scale to the stable E6 boundary (0.35).
# If action recovers while structural losses remain healthy, the fix is not
# another slot/overlap patch; it is a two-timescale action-backbone cotrain
# boundary with the suffix adapter.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_stepindexed_from9100_suffixadapter_backbone_sem035_h30k_20260531}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec "${SCRIPT_DIR}/run_a7_stepindexed_from9100_suffixadapter_backbone_300_20260531.sh"
