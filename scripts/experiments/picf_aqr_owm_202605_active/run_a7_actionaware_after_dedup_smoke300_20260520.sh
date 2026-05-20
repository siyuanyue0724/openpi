#!/usr/bin/env bash
set -euo pipefail

# Action-aware post-dedup smoke validation.
#
# This is the next gate after `picf_a7_active_support_dedup_300_20260520`.
# It keeps the exact slot/sidecar/OEML/owner-transport command used by the
# frozen-policy validation, but turns on the production-relevant pressure:
# PaliGemma/semantic training and action loss.  Large pretrained perception
# backbones remain frozen.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_actionaware_after_dedup_smoke300_20260520}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-300}"
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.25}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-0.50}"

exec "${SCRIPT_DIR}/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh"
