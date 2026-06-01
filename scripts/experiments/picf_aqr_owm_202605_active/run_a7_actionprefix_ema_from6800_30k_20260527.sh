#!/usr/bin/env bash
set -euo pipefail

# Action-interface EMA teacher repair for the persistent late action rebound.
#
# This supersedes the BASL/block-core diagnostic.  The measured failure is not
# current slot collapse: active/downstream overlap stays healthy while action
# rebounds.  Therefore the repair targets the PICF -> PI0.5 interface itself:
# PI0.5 consumes a slow EMA teacher prefix, while online PICF remains trainable
# and is softly kept near the teacher by a trust-region loss.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_action_interface_ema_20260527}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEFAULT_RESUME_6800="/mnt/checkpoints/picf_core/picf_core/picf_a7_phase_stabilized_p1_policyonly_from6500_6800_20260526/6800"
DEFAULT_RESUME_6500="/mnt/checkpoints/picf_core/picf_core/picf_a7_twotime_cotrain_from_pc1_6000_core005_action2_30k_20260526/6500"
if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  if [[ -d "${DEFAULT_RESUME_6800}" ]]; then
    export RESUME_CHECKPOINT="${DEFAULT_RESUME_6800}"
  else
    export RESUME_CHECKPOINT="${DEFAULT_RESUME_6500}"
  fi
fi

export EXP="${EXP:-picf_a7_actionprefix_ema_from6800_action2_30k_20260527}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"
export SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
if [[ -z "${SEGMENTS:-}" ]]; then
  if [[ ! -f "${SEGMENT_FILE}" ]]; then
    echo "Missing sidecar segment file: ${SEGMENT_FILE}" >&2
    exit 2
  fi
  export SEGMENTS
  SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
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

export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.005}"
export PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-constant}"

export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
export ACTION_PREFIX_RMS_TARGET="${ACTION_PREFIX_RMS_TARGET:-1.0}"
export ACTION_PREFIX_NORM_EPS="${ACTION_PREFIX_NORM_EPS:-1e-6}"
export ACTION_PREFIX_VALUE_CLIP="${ACTION_PREFIX_VALUE_CLIP:-0.0}"
export ACTION_PREFIX_OUTPUT_GATE="${ACTION_PREFIX_OUTPUT_GATE:-0.70}"
export ACTION_PREFIX_TEACHER_MODE="${ACTION_PREFIX_TEACHER_MODE:-ema}"
export ACTION_PREFIX_TEACHER_EMA_DECAY="${ACTION_PREFIX_TEACHER_EMA_DECAY:-0.99}"
export ACTION_PREFIX_TEACHER_BLEND="${ACTION_PREFIX_TEACHER_BLEND:-1.0}"
export LAMBDA_ACTION_PREFIX_TRUST="${LAMBDA_ACTION_PREFIX_TRUST:-0.02}"

export OBJECT_SCAFFOLD_DECAY_MODE="${OBJECT_SCAFFOLD_DECAY_MODE:-cosine}"
export OBJECT_SCAFFOLD_DECAY_START_STEP="${OBJECT_SCAFFOLD_DECAY_START_STEP:-0}"
export OBJECT_SCAFFOLD_DECAY_END_STEP="${OBJECT_SCAFFOLD_DECAY_END_STEP:-1500}"
export OBJECT_SCAFFOLD_DECAY_FLOOR="${OBJECT_SCAFFOLD_DECAY_FLOOR:-0.03}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER:-2.0}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN:-0.05}"
export ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M:-0.08}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec bash "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
