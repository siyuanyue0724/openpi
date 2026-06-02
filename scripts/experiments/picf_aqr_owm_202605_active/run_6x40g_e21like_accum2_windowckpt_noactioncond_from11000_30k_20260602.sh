#!/usr/bin/env bash
set -euo pipefail

# 6xA100-40GB E21-like action-readout follow-through.
#
# Mathematical intent:
#   E21's most reliable action descent used 12 windows per optimizer update.
#   The first 6-GPU follow-through used 6 windows/update and kept active/downstream
#   structure healthy, but action stayed near 0.040-0.046. This launcher uses
#   6 physical GPUs with accum=2, giving 12 windows/update while enabling window
#   activation checkpointing to keep the per-rank 40GB memory budget feasible.
#
# This is not a new objective and not a new module. It tests whether the remaining
# action plateau is primarily estimator variance / task-family mixing, rather
# than an AQR/OWM structural failure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_6gpu_b79761f}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"

export EXP="${EXP:-picf_6x40g_e21like_accum2_windowckpt_noactioncond_from11000_30k_20260602}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-6}"

export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

export ACCUM_STEPS="${ACCUM_STEPS:-2}"
export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-1}"

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"

exec bash "${SCRIPT_DIR}/run_a7_e23_bucketbalanced_noactioncond_from11000_30k_20260601.sh"
