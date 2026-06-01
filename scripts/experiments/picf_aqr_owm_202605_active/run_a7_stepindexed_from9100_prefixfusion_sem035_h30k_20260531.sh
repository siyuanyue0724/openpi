#!/usr/bin/env bash
set -euo pipefail

# E11 action-interface causal control.
#
# E10 showed that lowering the semantic/action backbone LR scale from 1.0 to
# 0.35 repairs the severe E9-h30k action degradation, while the suffix-side
# cross-attention adapter still does not clearly beat the E7/E8 band or recover
# the E6 prefix-fusion boundary.  This launcher keeps E10's step9100 checkpoint,
# sidecars, 30K horizon, policy-head LR, semantic LR scale, PICF LR scale, and
# action loss weight, changing only the context injection topology:
#
#   suffix_cross_attention -> prefix_fusion
#
# If action recovers under this matched run while structure remains healthy, the
# root cause is the suffix-side action-interface topology, not raw slot overlap,
# sidecar quality, or semantic LR alone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_stepindexed_from9100_prefixfusion_sem035_h30k_20260531}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"
export ACTION_CONTEXT_INTEGRATION="${ACTION_CONTEXT_INTEGRATION:-prefix_fusion}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec "${SCRIPT_DIR}/run_a7_stepindexed_from9100_suffixadapter_backbone_300_20260531.sh"
