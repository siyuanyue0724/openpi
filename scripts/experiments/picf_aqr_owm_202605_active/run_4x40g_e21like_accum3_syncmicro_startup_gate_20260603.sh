#!/usr/bin/env bash
set -euo pipefail

# 4xA100-40GB K12 startup gate.
#
# Mathematical contract:
#   world_size=4, accum_steps=3 -> 12 logical micro-windows per optimizer step.
#   This matches the E21 window count without needing 6/8 cards.
#
# Resource contract:
#   FSDP synchronizes each accumulation micro-step instead of using no_sync().
#   This is slower, but should lower the FSDP accumulation memory peak.  AdamW,
#   scheduler, checkpoint, and EMA still advance once per logical optimizer
#   update, so the training objective remains the E21-like K12 estimator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_4x40g_e21like_accum3_syncmicro_startup_gate_20260603}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-11002}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-0}"

export ACCUM_STEPS="${ACCUM_STEPS:-3}"
export LOGICAL_BATCH_TASK_COUNT="${LOGICAL_BATCH_TASK_COUNT:-12}"
export LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-1}"
export LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-1}"
export CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-task_uniform}"
export CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-1}"
export FSDP_SYNC_EACH_ACCUM_MICRO="${FSDP_SYNC_EACH_ACCUM_MICRO:-1}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"
export OPENPI_VERBOSE_STARTUP_LOGS="${OPENPI_VERBOSE_STARTUP_LOGS:-1}"
export OPENPI_DEBUG_PHASE_LIMIT="${OPENPI_DEBUG_PHASE_LIMIT:-11003}"

exec bash "${SCRIPT_DIR}/run_6x40g_e21like_accum2_windowckpt_noactioncond_from11000_30k_20260602.sh"
