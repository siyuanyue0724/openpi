#!/usr/bin/env bash
set -euo pipefail

# Corrected fixed-window-gated 30K relaunch from preserved EMA step7000.
#
# This launcher is the maintained answer to the 2026-05-29 "0.02 vs 0.04"
# action-loss dispute:
# - fixed-window probing showed the old 0.02 train-stream row is not a
#   stationary baseline;
# - the old step7000 checkpoint is still the best preserved basin;
# - production judgement must use fixed-window action probes plus CALVIN/video,
#   not raw sampled-window train rows alone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_probe_current_20260529}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export EXP="${EXP:-picf_a7_stepindexed_fixedwindow_from7000_action2_30k_20260529}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_actionprefix_ema_from6800_action2_30k_20260527/7000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-backbone_only}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
export LR="${LR:-7e-5}"
export MIN_LR="${MIN_LR:-2e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.005}"
export PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-constant}"

export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
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

exec "${SCRIPT_DIR}/run_a7_actionprefix_ema_from6800_30k_20260527.sh"
