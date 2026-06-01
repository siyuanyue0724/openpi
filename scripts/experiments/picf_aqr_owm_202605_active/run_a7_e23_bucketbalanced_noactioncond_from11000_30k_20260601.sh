#!/usr/bin/env bash
set -euo pipefail

# E23 production-scale follow-through for the E21/E22 action-readout audit.
#
# E21 proved that action starts descending when every optimizer update sees a
# balanced mix of task windows.  This launcher brings that principle into the
# normal CALVIN trainer with bucket-balanced accumulation instead of fixed exact
# windows.  Direct PICF action conditioning stays disabled because it has not
# yet proved a positive causal contribution under identical-window tests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_probe_current_20260529}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"

export EXP="${EXP:-picf_a7_e23_bucketbalanced_noactioncond_from11000_30k_20260601}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

# Bucket-balanced normal training.  On two GPUs this gives 2 windows/update and
# deterministic bucket rotation across steps.  accum_steps=2 OOMs on 40GB A100s
# under the current token budget, even with window activation checkpointing.
export ACCUM_STEPS="${ACCUM_STEPS:-1}"
export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-0}"

# Preserve the E14 action-pressure setting, but remove the unproven direct PICF
# action condition so this run validates the balanced native action path first.
export PICF_ACTION_CONDITION_ENABLED="${PICF_ACTION_CONDITION_ENABLED:-0}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-4.0}"
export ACTION_POS_WEIGHT="${ACTION_POS_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
export ACTION_ROT_WEIGHT="${ACTION_ROT_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
export ACTION_GRIPPER_WEIGHT="${ACTION_GRIPPER_WEIGHT:-${ACTION_LOSS_WEIGHT}}"

export ACTION_CONTEXT_TOKENS="${ACTION_CONTEXT_TOKENS:-24}"
export ACTION_CONTEXT_INTEGRATION="${ACTION_CONTEXT_INTEGRATION:-prefix_fusion}"
export ACTION_CONTEXT_STOPGRAD="${ACTION_CONTEXT_STOPGRAD:-1}"
export ACTION_PREFIX_STOPGRAD="${ACTION_PREFIX_STOPGRAD:-1}"
export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
export ACTION_PREFIX_OUTPUT_GATE="${ACTION_PREFIX_OUTPUT_GATE:-0.70}"
export ACTION_PREFIX_TEACHER_MODE="${ACTION_PREFIX_TEACHER_MODE:-ema}"
export ACTION_PREFIX_TEACHER_EMA_DECAY="${ACTION_PREFIX_TEACHER_EMA_DECAY:-0.99}"
export ACTION_PREFIX_TEACHER_BLEND="${ACTION_PREFIX_TEACHER_BLEND:-1.0}"
export LAMBDA_ACTION_PREFIX_TRUST="${LAMBDA_ACTION_PREFIX_TRUST:-0.02}"

export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-backbone_only}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.001}"
export PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-constant}"

export LR="${LR:-2.0e-5}"
export MIN_LR="${MIN_LR:-2.0e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
# Full optimizer restore plus accum_steps=2 exceeds 40GB during token-fusion
# attention.  E23 tests the balanced production objective, so use model-only
# resume from the same weights and rebuild Adam state.
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"

exec bash "${SCRIPT_DIR}/run_a7_stepindexed_from9100_suffixadapter_backbone_300_20260531.sh"
