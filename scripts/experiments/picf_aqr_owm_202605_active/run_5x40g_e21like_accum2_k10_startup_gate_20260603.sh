#!/usr/bin/env bash
set -euo pipefail

# 5xA100-40GB K10 startup gate.
#
# Mathematical contract:
#   world_size=5, accum_steps=2 -> 10 logical micro-windows per optimizer step.
#   This is not a strict E21 K12 replica, but it is the fastest 5-card diagnostic
#   for the same task-uniform logical-batch estimator.  Use it to verify that
#   FSDP, sidecars, bucket normalization, and action descent are healthy before
#   spending time on the slower K15 route.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_5x40g_e21like_accum2_k10_startup_gate_20260603}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-5}"

export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-11100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-0}"

export ACCUM_STEPS="${ACCUM_STEPS:-2}"
export LOGICAL_BATCH_TASK_COUNT="${LOGICAL_BATCH_TASK_COUNT:-10}"
export LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-1}"
export LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-1}"
export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-task_uniform}"
export CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-1}"
export FSDP_SYNC_EACH_ACCUM_MICRO="${FSDP_SYNC_EACH_ACCUM_MICRO:-0}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"
export OPENPI_VERBOSE_STARTUP_LOGS="${OPENPI_VERBOSE_STARTUP_LOGS:-1}"
export OPENPI_DEBUG_PHASE_LIMIT="${OPENPI_DEBUG_PHASE_LIMIT:-11101}"

exec bash "${SCRIPT_DIR}/run_6x40g_e21like_accum2_windowckpt_noactioncond_from11000_30k_20260602.sh"
