#!/usr/bin/env bash
set -euo pipefail

# Moving-prefix causal probe from the PC1 mid-LR run.
#
# Purpose:
#   Test whether the observed post-release action rebound is caused by the PICF
#   belief/prefix itself moving under structural sidecar losses.  The run keeps
#   PICF forward evidence active, but freezes the complete `core.*` PICF stack
#   and trains only the non-core policy path.
#
# Contract:
#   - resume model weights from the PC1 mid-LR local minimum when available;
#   - reset optimizer/scheduler state;
#   - freeze PICF core parameters with `picf_trainable_scope=policy_only`;
#   - disable structural/object scaffold losses so only action pressure updates
#     the policy path;
#   - inspect the first 300-500 steps as a causal diagnostic.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_2500fresh_20260525}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export EXP="${EXP:-picf_pc1_freezepicf_policyonly_from_pc1_6000_action2_30k_20260526}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_pc1_from_a7_5500_freshopt_midlr_actionstable_ckpt1000_20260526/6000}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PICF_TRAINABLE_SCOPE="${PICF_TRAINABLE_SCOPE:-policy_only}"
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.25}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"

export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
export ACTION_PREFIX_RMS_TARGET="${ACTION_PREFIX_RMS_TARGET:-1.0}"
export ACTION_PREFIX_NORM_EPS="${ACTION_PREFIX_NORM_EPS:-1e-6}"
export ACTION_PREFIX_VALUE_CLIP="${ACTION_PREFIX_VALUE_CLIP:-0.0}"

export LR="${LR:-3.5e-5}"
export MIN_LR="${MIN_LR:-1.4e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

# Disable structural losses: this is a policy-only causal test, not a new slot
# training phase.
export LAMBDA_ANCHOR_PV="${LAMBDA_ANCHOR_PV:-0.0}"
export LAMBDA_ANCHOR_OBJECT_PULL="${LAMBDA_ANCHOR_OBJECT_PULL:-0.0}"
export LAMBDA_PV_WEAK="${LAMBDA_PV_WEAK:-0.0}"
export LAMBDA_MAPG_CYCLE="${LAMBDA_MAPG_CYCLE:-0.0}"
export LAMBDA_MAPG_ROUTING="${LAMBDA_MAPG_ROUTING:-0.0}"
export LAMBDA_MAPG_SUPPORT_DIVERSITY="${LAMBDA_MAPG_SUPPORT_DIVERSITY:-0.0}"
export LAMBDA_SLOT_QUALITY="${LAMBDA_SLOT_QUALITY:-0.0}"
export LAMBDA_OBJECT_EXPLANATION_POINT="${LAMBDA_OBJECT_EXPLANATION_POINT:-0.0}"
export LAMBDA_OBJECT_EXPLANATION_CONTACT="${LAMBDA_OBJECT_EXPLANATION_CONTACT:-0.0}"
export LAMBDA_OBJECT_EXPLANATION_DUPLICATE="${LAMBDA_OBJECT_EXPLANATION_DUPLICATE:-0.0}"
export LAMBDA_OBJECT_EXPLANATION_BACKGROUND="${LAMBDA_OBJECT_EXPLANATION_BACKGROUND:-0.0}"
export OBJECT_SCAFFOLD_DECAY_MODE="${OBJECT_SCAFFOLD_DECAY_MODE:-none}"

if [[ -z "${SEGMENTS:-}" ]]; then
  SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
  if [[ ! -f "${SEGMENT_FILE}" ]]; then
    echo "Missing sidecar segment file: ${SEGMENT_FILE}" >&2
    exit 2
  fi
  SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
  export SEGMENTS
fi

exec "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
