#!/usr/bin/env bash
set -euo pipefail

# 2026-05-23 V-JEPA frozen-feature-cache smoke.
#
# This is the maintained action-aware profile with the conservative action
# pressure used by the current long-run family.  It tests that V-JEPA cache
# read_or_encode is objective-preserving while leaving the validated PICF slot,
# sidecar, owner, and context contract unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_vjepa_cache_action05_smoke200_20260523}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-0.50}"
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.25}"
export VJEPA_FEATURE_CACHE_MODE="${VJEPA_FEATURE_CACHE_MODE:-read_or_encode}"
export VJEPA_FEATURE_CACHE_ROOT="${VJEPA_FEATURE_CACHE_ROOT:-/mnt/picf_frozen_feature_cache/vjepa2_1_base384_hierarchical}"
export VJEPA_FEATURE_CACHE_TEMPORAL_SLICES="${VJEPA_FEATURE_CACHE_TEMPORAL_SLICES:-2}"
export VJEPA_FEATURE_CACHE_STORAGE_DTYPE="${VJEPA_FEATURE_CACHE_STORAGE_DTYPE:-bfloat16}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
