#!/usr/bin/env bash
set -euo pipefail

# E14H action-pressure continuation.
#
# Purpose:
#   Test whether the E14G post-9600 action plateau is caused by insufficient
#   action pressure rather than structural PICF collapse.
#
# Design:
#   Resume the complete E14G step10000 checkpoint, including optimizer state.
#   Keep the accepted E14 action interface and two-timescale PICF boundary.
#   Increase only ACTION_LOSS_WEIGHT from 2.0 to 4.0.
#
# Interpretation:
#   If action loss falls while active/downstream overlap stays healthy, the
#   plateau was an action-pressure schedule issue.  If action does not move,
#   the bottleneck is representation/prefix utility or sampled task difficulty,
#   not merely scalar action weighting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_probe_current_20260529}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"

export EXP="${EXP:-picf_a7_e14h_action4_fullopt_from10000_30k_20260601}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_e14g_fullopt_from9600_30k_20260531/10000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

# Preserve the accepted E14 action interface.
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

# Keep the E14 two-timescale boundary; change only action pressure.
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-backbone_only}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-0.35}"
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-4.0}"
export LR="${LR:-2.0e-5}"
export MIN_LR="${MIN_LR:-2.0e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.001}"
export PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-constant}"

# Full optimizer resume is required to avoid confusing this test with a fresh
# Adam/restart bonus.
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-full}"

exec "${SCRIPT_DIR}/run_a7_stepindexed_from9100_suffixadapter_backbone_300_20260531.sh"
