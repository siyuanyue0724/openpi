#!/usr/bin/env bash
set -euo pipefail

# 2026-05-21 strict-scope reserve/raw-overlap audit.
#
# Purpose:
#   Re-run the frozen-policy slot/router/posterior/OEML validation after the
#   active/context/reserve diagnostic scopes were made mutually exclusive:
#     active  = graph.anchor_active > 0.5
#     context = downstream_weight > eps and not active
#     reserve = downstream_weight <= eps and not active
#
# This is still not an anchor-only probe.  It freezes PaliGemma, the action
# head/losses, and pretrained perception backbones, while training the
# PICF/AQR/posterior object-file stack and sidecar/contact-motion routing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_slot_reserveaudit_strictscope_frozen_policy_300_20260521}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-300}"
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-ddp}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-0}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-1e-6}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-0.0}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

if [[ -z "${SEGMENTS:-}" ]]; then
  SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
  if [[ ! -f "${SEGMENT_FILE}" ]]; then
    echo "Missing sidecar segment file: ${SEGMENT_FILE}" >&2
    echo "Run scripts/picf_prepare_full_sidecar_root.py after full sidecar generation." >&2
    exit 2
  fi
  SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
  export SEGMENTS
fi

exec "${SCRIPT_DIR}/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh"
