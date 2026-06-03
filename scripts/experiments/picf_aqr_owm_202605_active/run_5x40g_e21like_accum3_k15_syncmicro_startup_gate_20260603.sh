#!/usr/bin/env bash
set -euo pipefail

# 5xA100-40GB K15 startup gate.
#
# Mathematical contract:
#   world_size=5, accum_steps=3 -> 15 logical micro-windows per optimizer step.
#   This exceeds E21's K12 coverage.  It is slower than K10, but can be useful if
#   K10 starts cleanly yet still shows action plateau from insufficient task
#   coverage.  FSDP synchronizes every accumulation micro-step to reduce the
#   accumulated-gradient memory peak on 40GB cards.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_5x40g_e21like_accum3_k15_syncmicro_startup_gate_20260603}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-5}"

export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-11050}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-0}"

export ACCUM_STEPS="${ACCUM_STEPS:-3}"
export LOGICAL_BATCH_TASK_COUNT="${LOGICAL_BATCH_TASK_COUNT:-15}"
export LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-1}"
export LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-1}"
export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-task_uniform}"
export CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-1}"
export FSDP_SYNC_EACH_ACCUM_MICRO="${FSDP_SYNC_EACH_ACCUM_MICRO:-1}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"
export OPENPI_VERBOSE_STARTUP_LOGS="${OPENPI_VERBOSE_STARTUP_LOGS:-1}"
export OPENPI_DEBUG_PHASE_LIMIT="${OPENPI_DEBUG_PHASE_LIMIT:-11051}"

exec bash "${SCRIPT_DIR}/run_6x40g_e21like_accum2_windowckpt_noactioncond_from11000_30k_20260602.sh"
