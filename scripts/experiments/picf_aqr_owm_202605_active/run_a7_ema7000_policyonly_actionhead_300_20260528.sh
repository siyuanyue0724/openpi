#!/usr/bin/env bash
set -euo pipefail

# Causal split B: freeze PICF and freeze the PaliGemma/Gemma semantic backbone;
# train only the PI0.5 action projections/time MLPs from the preserved EMA
# step7000 checkpoint.
#
# This isolates whether the late rebound is caused by large semantic-backbone
# drift after the action path has already reached a low-loss basin.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_action_interface_ema_20260527}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export EXP="${EXP:-picf_a7_ema7000_policyonly_actionhead_300_20260528}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"
export SEGMENT_FILE="${SIDECAR_ROOT}/calvin_segment_indices.txt"
if [[ -z "${SEGMENTS:-}" ]]; then
  export SEGMENTS
  SEGMENTS="$(tr -d '\n' < "${SEGMENT_FILE}")"
fi

# This probe trains only the 2.1M PI0 action projection/time-MLP parameters.
# Use DDP by default so the causal result is not confounded by FSDP's mixed
# fp32/bf16 flattening rules inside the otherwise frozen PaliGemma root.
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-ddp}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PICF_TRAINABLE_SCOPE="${PICF_TRAINABLE_SCOPE:-policy_only}"
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-action_head_only}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-1.0}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
export LR="${LR:-7e-5}"
export MIN_LR="${MIN_LR:-2e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
export ACTION_PREFIX_OUTPUT_GATE="${ACTION_PREFIX_OUTPUT_GATE:-0.70}"
export ACTION_PREFIX_TEACHER_MODE="${ACTION_PREFIX_TEACHER_MODE:-ema}"
export ACTION_PREFIX_TEACHER_EMA_DECAY="${ACTION_PREFIX_TEACHER_EMA_DECAY:-0.99}"
export ACTION_PREFIX_TEACHER_BLEND="${ACTION_PREFIX_TEACHER_BLEND:-1.0}"
export LAMBDA_ACTION_PREFIX_TRUST="${LAMBDA_ACTION_PREFIX_TRUST:-0.0}"

export LAMBDA_ANCHOR_PV="${LAMBDA_ANCHOR_PV:-0.0}"
export LAMBDA_ANCHOR_OBJECT_PULL="${LAMBDA_ANCHOR_OBJECT_PULL:-0.0}"
export LAMBDA_MAPG_CYCLE="${LAMBDA_MAPG_CYCLE:-0.0}"
export LAMBDA_MAPG_SUPPORT_DIVERSITY="${LAMBDA_MAPG_SUPPORT_DIVERSITY:-0.0}"
export LAMBDA_SLOT_QUALITY="${LAMBDA_SLOT_QUALITY:-0.0}"
export LAMBDA_OBJECT_EXPLANATION_POINT="${LAMBDA_OBJECT_EXPLANATION_POINT:-0.0}"
export LAMBDA_OBJECT_EXPLANATION_CONTACT="${LAMBDA_OBJECT_EXPLANATION_CONTACT:-0.0}"
export LAMBDA_OBJECT_EXPLANATION_DUPLICATE="${LAMBDA_OBJECT_EXPLANATION_DUPLICATE:-0.0}"
export LAMBDA_OBJECT_EXPLANATION_BACKGROUND="${LAMBDA_OBJECT_EXPLANATION_BACKGROUND:-0.0}"
export OBJECT_SCAFFOLD_DECAY_MODE="${OBJECT_SCAFFOLD_DECAY_MODE:-none}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec "${SCRIPT_DIR}/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
