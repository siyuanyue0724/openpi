#!/usr/bin/env bash
set -euo pipefail

# G16: action-expert-only routing after G12-G15 rejected sampler scalar,
# local boundary, semantic-capacity, and robust-objective fixes.
#
# Contract:
#   docs/PICF_AQR_OWM_G12_ALL_REQUESTED_VLA_METHODS_TEST_PLAN_20260603.md
#
# Fixed estimator:
#   K4 task-uniform logical batch, without-replacement bucket coverage,
#   per-bucket normalization, action weight=4, PICF action context on.
#
# Variable:
#   semantic/PICF-conditioned low-rank action-expert router on suffix tokens.
#   This is not whole-VLM MoE and does not change token counts or masks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_e21b2_1b07eab}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

export CASE_TARGET_STEP="${CASE_TARGET_STEP:-11120}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-0}"

export ACCUM_STEPS="${ACCUM_STEPS:-2}"
export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-task_uniform}"
export CALVIN_BUCKET_TEMPERATURE_ALPHA="${CALVIN_BUCKET_TEMPERATURE_ALPHA:-0.0}"
export CALVIN_BUCKET_WEIGHT_SPEC="${CALVIN_BUCKET_WEIGHT_SPEC:-}"
export CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"
export LOGICAL_BATCH_TASK_COUNT="${LOGICAL_BATCH_TASK_COUNT:-4}"
export LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-1}"
export LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-1}"
export LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION="${LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION:-0}"
export LOGICAL_BATCH_DYNAMIC_MIXING="${LOGICAL_BATCH_DYNAMIC_MIXING:-0}"
export LOGICAL_BATCH_GRADIENT_SURGERY="${LOGICAL_BATCH_GRADIENT_SURGERY:-off}"

export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-4.0}"
export ACTION_POS_WEIGHT="${ACTION_POS_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
export ACTION_ROT_WEIGHT="${ACTION_ROT_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
export ACTION_GRIPPER_WEIGHT="${ACTION_GRIPPER_WEIGHT:-${ACTION_LOSS_WEIGHT}}"

export PICF_ACTION_CONDITION_ENABLED="${PICF_ACTION_CONDITION_ENABLED:-1}"
export ACTION_CONTEXT_TOKENS="${ACTION_CONTEXT_TOKENS:-24}"
export ACTION_CONTEXT_INTEGRATION="${ACTION_CONTEXT_INTEGRATION:-suffix_cross_attention}"
export ACTION_CONTEXT_STOPGRAD="${ACTION_CONTEXT_STOPGRAD:-1}"
export ACTION_CONTEXT_NORM_MODE="${ACTION_CONTEXT_NORM_MODE:-rmsnorm}"
export ACTION_CONTEXT_RMS_TARGET="${ACTION_CONTEXT_RMS_TARGET:-1.0}"
export ACTION_CONTEXT_OUTPUT_GATE="${ACTION_CONTEXT_OUTPUT_GATE:-0.25}"
export ACTION_CONTEXT_INCLUDE_QUERY_TOKENS="${ACTION_CONTEXT_INCLUDE_QUERY_TOKENS:-0}"

export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-action_head_and_adapter}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-1.0}"
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.001}"
export PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-constant}"

export SEMANTIC_ACTION_FLOW_LOSS="${SEMANTIC_ACTION_FLOW_LOSS:-mse}"
export SEMANTIC_ACTION_FLOW_HUBER_DELTA="${SEMANTIC_ACTION_FLOW_HUBER_DELTA:-1.0}"
export SEMANTIC_ACTION_FLOW_TIME_ALPHA="${SEMANTIC_ACTION_FLOW_TIME_ALPHA:-1.5}"
export SEMANTIC_ACTION_FLOW_TIME_BETA="${SEMANTIC_ACTION_FLOW_TIME_BETA:-1.0}"

export LR="${LR:-2.0e-5}"
export MIN_LR="${MIN_LR:-2.0e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-ddp}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"
export WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-0}"

RUN_LOG_DIR="${RUN_LOG_DIR:-/mnt/picf_run_logs/g16_action_expert_router_20260603}"
ONLY_CASE="${ONLY_CASE:-g16_router4_rank64_k4}"
mkdir -p "${RUN_LOG_DIR}"

run_case() {
  local case_id="$1"
  shift
  local exp_name="picf_${case_id}_$(date +%Y%m%d_%H%M%S)"
  local log_path="${RUN_LOG_DIR}/${case_id}.log"

  echo "===== ${case_id} START $(date -Is) =====" | tee -a "${RUN_LOG_DIR}/matrix_status.log"
  echo "exp=${exp_name} target_step=${CASE_TARGET_STEP} $*" | tee -a "${RUN_LOG_DIR}/matrix_status.log"

  set +e
  env \
    EXP="${exp_name}" \
    NUM_TRAIN_STEPS="${CASE_TARGET_STEP}" \
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

maybe_run_case "g16_router4_rank64_k4" \
  SEMANTIC_ACTION_EXPERT_ROUTER_ENABLED=1 \
  SEMANTIC_ACTION_EXPERT_ROUTER_EXPERTS=4 \
  SEMANTIC_ACTION_EXPERT_ROUTER_RANK=64 \
  SEMANTIC_ACTION_EXPERT_ROUTER_GATE_INIT=-2.5 \
  SEMANTIC_ACTION_EXPERT_ROUTER_TEMPERATURE=1.0

maybe_run_case "g16_router8_rank64_k4" \
  SEMANTIC_ACTION_EXPERT_ROUTER_ENABLED=1 \
  SEMANTIC_ACTION_EXPERT_ROUTER_EXPERTS=8 \
  SEMANTIC_ACTION_EXPERT_ROUTER_RANK=64 \
  SEMANTIC_ACTION_EXPERT_ROUTER_GATE_INIT=-2.5 \
  SEMANTIC_ACTION_EXPERT_ROUTER_TEMPERATURE=1.0

maybe_run_case "g16_router4_rank32_k4" \
  SEMANTIC_ACTION_EXPERT_ROUTER_ENABLED=1 \
  SEMANTIC_ACTION_EXPERT_ROUTER_EXPERTS=4 \
  SEMANTIC_ACTION_EXPERT_ROUTER_RANK=32 \
  SEMANTIC_ACTION_EXPERT_ROUTER_GATE_INIT=-2.5 \
  SEMANTIC_ACTION_EXPERT_ROUTER_TEMPERATURE=1.0

echo "G16 action-expert router matrix completed at $(date -Is). Logs: ${RUN_LOG_DIR}"
