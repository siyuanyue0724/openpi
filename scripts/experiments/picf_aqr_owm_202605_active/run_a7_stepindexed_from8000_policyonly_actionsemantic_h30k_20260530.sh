#!/usr/bin/env bash
set -euo pipefail

# Frozen-PICF action/semantic transfer probe from the preserved step8000 basin.
#
# Purpose:
#   Keep PICF forward evidence active but freeze the full `core.*` belief
#   router.  Train only the non-core policy/action/PaliGemma path.  This tests
#   whether the action plateau is caused by moving PICF prefixes or by the
#   action/semantic optimizer itself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_probe_current_20260529}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export EXP="${EXP:-picf_a7_stepindexed_from8000_policyonly_actionsemantic_h30k_20260530}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529/8000}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

if [[ -z "${SEGMENTS:-}" ]]; then
  SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
  if [[ ! -f "${SEGMENT_FILE}" ]]; then
    echo "Missing sidecar segment file: ${SEGMENT_FILE}" >&2
    exit 2
  fi
  SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
  export SEGMENTS
fi

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"
export PICF_TRAINABLE_SCOPE="${PICF_TRAINABLE_SCOPE:-policy_only}"

export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-backbone_only}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
export LR="${LR:-7e-5}"
export MIN_LR="${MIN_LR:-2e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

# PICF is frozen, so structural losses must be off.  Otherwise loss_total would
# carry non-action terms that cannot update the frozen belief router.
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
export LAMBDA_ACTION_PREFIX_TRUST="${LAMBDA_ACTION_PREFIX_TRUST:-0.0}"

export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
export ACTION_PREFIX_OUTPUT_GATE="${ACTION_PREFIX_OUTPUT_GATE:-0.70}"
export ACTION_PREFIX_TEACHER_MODE="${ACTION_PREFIX_TEACHER_MODE:-ema}"
export ACTION_PREFIX_TEACHER_EMA_DECAY="${ACTION_PREFIX_TEACHER_EMA_DECAY:-0.99}"
export ACTION_PREFIX_TEACHER_BLEND="${ACTION_PREFIX_TEACHER_BLEND:-1.0}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
