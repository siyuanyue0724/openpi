#!/usr/bin/env bash
set -euo pipefail

# Phase-stabilized cotrain from the clean pre-rebound A7 step-6500 checkpoint.
#
# Root cause addressed:
#   The action path is consuming a PICF belief prefix that still moves too
#   quickly under structural/object losses.  Lowering core LR helped but did
#   not remove the 7000/7050/7100 rebound.  This script adds an explicit phase
#   boundary before slow cotrain:
#
#     phase 1: freeze PICF core and train action/semantic against stationary
#              PICF prefixes for 300 optimizer steps.
#     phase 2: resume full cotrain with a slower PICF core timescale.
#
# This is not policy_only as a final recipe.  Policy-only is only the
# stationarization phase; the maintained phase-2 run keeps PICF cotrain.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_phase_stabilized_20260526}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export BASE_CHECKPOINT="${BASE_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_twotime_cotrain_from_pc1_6000_core005_action2_30k_20260526/6500}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"
export SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
if [[ -z "${SEGMENTS:-}" ]]; then
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
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
export LR="${LR:-7e-5}"
export MIN_LR="${MIN_LR:-2e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
export ACTION_PREFIX_RMS_TARGET="${ACTION_PREFIX_RMS_TARGET:-1.0}"
export ACTION_PREFIX_NORM_EPS="${ACTION_PREFIX_NORM_EPS:-1e-6}"
export ACTION_PREFIX_VALUE_CLIP="${ACTION_PREFIX_VALUE_CLIP:-0.0}"

export OBJECT_SCAFFOLD_DECAY_MODE="${OBJECT_SCAFFOLD_DECAY_MODE:-cosine}"
export OBJECT_SCAFFOLD_DECAY_START_STEP="${OBJECT_SCAFFOLD_DECAY_START_STEP:-0}"
export OBJECT_SCAFFOLD_DECAY_END_STEP="${OBJECT_SCAFFOLD_DECAY_END_STEP:-1500}"
export OBJECT_SCAFFOLD_DECAY_FLOOR="${OBJECT_SCAFFOLD_DECAY_FLOOR:-0.03}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER:-2.0}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN:-0.05}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M:-0.08}"

PHASE1_EXP="${PHASE1_EXP:-picf_a7_phase_stabilized_p1_policyonly_from6500_6800_20260526}"
PHASE2_EXP="${PHASE2_EXP:-picf_a7_phase_stabilized_p2_core001_action2_30k_20260526}"
PHASE1_END_STEP="${PHASE1_END_STEP:-6800}"
PHASE2_END_STEP="${PHASE2_END_STEP:-30000}"
PHASE1_CKPT="/mnt/checkpoints/picf_core/picf_core/${PHASE1_EXP}/${PHASE1_END_STEP}"
PHASE2_ONLY="${PHASE2_ONLY:-0}"

if [[ "${PHASE2_ONLY}" != "1" ]]; then
  echo "[phase-stabilized] phase 1: stationary-prefix action adaptation"
  (
    export EXP="${PHASE1_EXP}"
    export RESUME_CHECKPOINT="${BASE_CHECKPOINT}"
    export NUM_TRAIN_STEPS="${PHASE1_END_STEP}"
    export PICF_TRAINABLE_SCOPE="policy_only"
    # The trainer validates this value as strictly positive even when
    # policy_only freezes the PICF core. Keep it tiny; it is not an active
    # core update in phase 1.
    export PICF_CORE_LR_SCALE="1e-6"
    export SAVE_INTERVAL="100"
    export KEEP_LAST_CHECKPOINTS="3"
    export LOG_INTERVAL="50"
    export ANCHOR_OVERLAY_INTERVAL="100"
    # Keep phase 1 as an action-prefix stationarization probe.  PICF forward
    # remains enabled, but structural losses cannot move the prefix.
    export LAMBDA_ANCHOR_OBJECT_PULL="0.0"
    export LAMBDA_SLOT_QUALITY="0.0"
    export LAMBDA_MAPG_CYCLE="0.0"
    export LAMBDA_MAPG_SUPPORT_DIVERSITY="0.0"
    export LAMBDA_OBJECT_EXPLANATION_POINT="0.0"
    export LAMBDA_OBJECT_EXPLANATION_CONTACT="0.0"
    export LAMBDA_OBJECT_EXPLANATION_DUPLICATE="0.0"
    export LAMBDA_OBJECT_EXPLANATION_BACKGROUND="0.0"
    "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
  )
else
  echo "[phase-stabilized] phase 1 skipped: PHASE2_ONLY=1"
fi

if [[ ! -d "${PHASE1_CKPT}" ]]; then
  echo "Phase-1 checkpoint missing: ${PHASE1_CKPT}" >&2
  exit 3
fi

echo "[phase-stabilized] phase 2: slow PICF cotrain"
(
  export EXP="${PHASE2_EXP}"
  export RESUME_CHECKPOINT="${PHASE1_CKPT}"
  export NUM_TRAIN_STEPS="${PHASE2_END_STEP}"
  export PICF_TRAINABLE_SCOPE="all"
  export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.01}"
  export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
  export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
  export LOG_INTERVAL="${LOG_INTERVAL:-50}"
  export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"
  "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
)
