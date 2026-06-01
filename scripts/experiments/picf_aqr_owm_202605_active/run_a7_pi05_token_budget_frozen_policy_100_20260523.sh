#!/usr/bin/env bash
set -euo pipefail

# 2026-05-23 PI0.5-close token-budget probe.
#
# Purpose:
#   Isolate the safe PI0.5 parity token change `semantic_max_length=200` from
#   structural PICF/AQR changes. This keeps the maintained comprehensive
#   frozen-policy slot/router/posterior/OEML profile unchanged except for the
#   PaliGemma semantic prefix budget.
#
# Why not change visual-grid/AQR query count here:
#   `semantic_max_length=200` is the documented PI0.5 parity knob.  Reducing
#   visual-grid or object-query capacity would change the belief router itself
#   and would confound speed/loss comparison against prior 300-step gates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_pi05tok200_frozen_policy_100_20260523}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100}"
export SEMANTIC_MAX_LENGTH="${SEMANTIC_MAX_LENGTH:-200}"
export LOG_INTERVAL="${LOG_INTERVAL:-25}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-50}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"

exec "${SCRIPT_DIR}/run_a7_slot_qualitytarget_frozen_policy_300_20260521.sh"
