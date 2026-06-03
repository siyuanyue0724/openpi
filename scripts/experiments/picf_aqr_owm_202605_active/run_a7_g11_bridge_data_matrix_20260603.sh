#!/usr/bin/env bash
set -euo pipefail

# G11: bridge/data matrix after G10 rejected static sampler and simple
# policy-only boundary fixes.
#
# Contract:
#   docs/PICF_AQR_OWM_G11_FULL_VLA_BRIDGE_AND_DATA_PLAN_20260603.md
#
# Cases:
#   g11a: disable PICF action context, K4 adapter scope
#   g11b: disable PICF action context, K2 fuller semantic scope
#   g11b2: g11b resource-safe retry with FSDP + window checkpointing
#   g11c: prefix-fusion bridge, K4 adapter scope
#   g11d: append bridge negative control, K4 adapter scope
#
# Use ONLY_CASE=<case_id> to run one case.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_e21b2_1b07eab}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

export CASE_TARGET_STEP="${CASE_TARGET_STEP:-11100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-0}"

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-ddp}"
export WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-0}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"

export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-task_uniform}"
export CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"
export LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-1}"
export LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-1}"
export LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION="${LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION:-0}"
export LOGICAL_BATCH_GRADIENT_SURGERY="${LOGICAL_BATCH_GRADIENT_SURGERY:-off}"

export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-4.0}"
export ACTION_POS_WEIGHT="${ACTION_POS_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
export ACTION_ROT_WEIGHT="${ACTION_ROT_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
export ACTION_GRIPPER_WEIGHT="${ACTION_GRIPPER_WEIGHT:-${ACTION_LOSS_WEIGHT}}"

export ACTION_CONTEXT_TOKENS="${ACTION_CONTEXT_TOKENS:-24}"
export ACTION_CONTEXT_STOPGRAD="${ACTION_CONTEXT_STOPGRAD:-1}"
export ACTION_CONTEXT_NORM_MODE="${ACTION_CONTEXT_NORM_MODE:-rmsnorm}"
export ACTION_CONTEXT_RMS_TARGET="${ACTION_CONTEXT_RMS_TARGET:-1.0}"
export ACTION_CONTEXT_OUTPUT_GATE="${ACTION_CONTEXT_OUTPUT_GATE:-0.25}"
export ACTION_CONTEXT_INCLUDE_QUERY_TOKENS="${ACTION_CONTEXT_INCLUDE_QUERY_TOKENS:-0}"
export ACTION_PREFIX_STOPGRAD="${ACTION_PREFIX_STOPGRAD:-1}"
export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
export ACTION_PREFIX_OUTPUT_GATE="${ACTION_PREFIX_OUTPUT_GATE:-0.70}"
export ACTION_PREFIX_TEACHER_MODE="${ACTION_PREFIX_TEACHER_MODE:-ema}"
export ACTION_PREFIX_TEACHER_EMA_DECAY="${ACTION_PREFIX_TEACHER_EMA_DECAY:-0.99}"
export ACTION_PREFIX_TEACHER_BLEND="${ACTION_PREFIX_TEACHER_BLEND:-1.0}"
export LAMBDA_ACTION_PREFIX_TRUST="${LAMBDA_ACTION_PREFIX_TRUST:-0.02}"

export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-1.0}"
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.001}"
export PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-constant}"

export LR="${LR:-2.0e-5}"
export MIN_LR="${MIN_LR:-2.0e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

RUN_LOG_DIR="${RUN_LOG_DIR:-/mnt/picf_run_logs/g11_bridge_data_matrix_20260603}"
ONLY_CASE="${ONLY_CASE:-}"
mkdir -p "${RUN_LOG_DIR}"

run_case() {
  local case_id="$1"
  shift
  local exp_name="picf_${case_id}_$(date +%Y%m%d_%H%M%S)"
  local log_path="${RUN_LOG_DIR}/${case_id}.log"

  echo "===== ${case_id} START $(date -Is) =====" | tee -a "${RUN_LOG_DIR}/matrix_status.log"
  echo "exp=${exp_name} $*" | tee -a "${RUN_LOG_DIR}/matrix_status.log"

  set +e
  env \
    EXP="${exp_name}" \
    "$@" \
    bash "${SCRIPT_DIR}/run_a7_f8_k4_action_adapter_taskuniform_from11000_11100_20260602.sh" \
    2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e

  echo "===== ${case_id} END status=${status} $(date -Is) =====" | tee -a "${RUN_LOG_DIR}/matrix_status.log"
  return 0
}

maybe_run_case() {
  local case_id="$1"
  shift
  if [[ -n "${ONLY_CASE}" && "${ONLY_CASE}" != "${case_id}" ]]; then
    echo "===== ${case_id} SKIP due to ONLY_CASE=${ONLY_CASE} =====" | tee -a "${RUN_LOG_DIR}/matrix_status.log"
    return 0
  fi
  run_case "${case_id}" "$@"
}

maybe_run_case "g11a_no_picf_action_k4_adapter" \
  PICF_ACTION_CONDITION_ENABLED=0 \
  ACTION_CONTEXT_INTEGRATION=suffix_cross_attention \
  ACCUM_STEPS=2 \
  LOGICAL_BATCH_TASK_COUNT=4 \
  SEMANTIC_TRAINABLE_SCOPE=action_head_and_adapter \
  PICF_TRAINABLE_SCOPE=all

maybe_run_case "g11b_no_picf_action_k2_backbone" \
  PICF_ACTION_CONDITION_ENABLED=0 \
  ACTION_CONTEXT_INTEGRATION=suffix_cross_attention \
  ACCUM_STEPS=1 \
  LOGICAL_BATCH_TASK_COUNT=2 \
  SEMANTIC_TRAINABLE_SCOPE=backbone_only \
  PICF_TRAINABLE_SCOPE=all

maybe_run_case "g11b2_no_picf_action_k2_backbone_fsdp_ckpt" \
  PICF_ACTION_CONDITION_ENABLED=0 \
  ACTION_CONTEXT_INTEGRATION=suffix_cross_attention \
  ACCUM_STEPS=1 \
  LOGICAL_BATCH_TASK_COUNT=2 \
  SEMANTIC_TRAINABLE_SCOPE=backbone_only \
  PICF_TRAINABLE_SCOPE=all \
  TRAINING_STRATEGY=fsdp_full_shard \
  WINDOW_ACTIVATION_CHECKPOINTING=1 \
  OPTIMIZER_SHARDING=none

maybe_run_case "g11c_prefix_fusion_k4_adapter" \
  PICF_ACTION_CONDITION_ENABLED=1 \
  ACTION_CONTEXT_INTEGRATION=prefix_fusion \
  ACCUM_STEPS=2 \
  LOGICAL_BATCH_TASK_COUNT=4 \
  SEMANTIC_TRAINABLE_SCOPE=action_head_and_adapter \
  PICF_TRAINABLE_SCOPE=all

maybe_run_case "g11d_append_bridge_negative_control" \
  PICF_ACTION_CONDITION_ENABLED=1 \
  ACTION_CONTEXT_INTEGRATION=append \
  ACCUM_STEPS=2 \
  LOGICAL_BATCH_TASK_COUNT=4 \
  SEMANTIC_TRAINABLE_SCOPE=action_head_and_adapter \
  PICF_TRAINABLE_SCOPE=all

echo "G11 matrix completed at $(date -Is). Logs: ${RUN_LOG_DIR}"
