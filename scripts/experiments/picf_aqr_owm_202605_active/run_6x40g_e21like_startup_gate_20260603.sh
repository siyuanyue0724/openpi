#!/usr/bin/env bash
set -euo pipefail

# 6xA100-40GB K12 startup gate.
#
# Purpose:
#   Before spending a long run, prove that the complete E21-like production
#   contract reaches Training config, restores the checkpoint, completes at
#   least one optimizer step, and emits logical-batch bucket metrics.
#
# This is intentionally short.  With the default step11000 checkpoint it targets
# step11002, because picf_core_train.py interprets NUM_TRAIN_STEPS as an absolute
# target step after resume.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_6x40g_e21like_startup_gate_20260603}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-11002}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-0}"

export ACCUM_STEPS="${ACCUM_STEPS:-2}"
export LOGICAL_BATCH_TASK_COUNT="${LOGICAL_BATCH_TASK_COUNT:-12}"
export LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-1}"
export LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-1}"
export CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-task_uniform}"
export CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"

export WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-1}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"
export OPENPI_VERBOSE_STARTUP_LOGS="${OPENPI_VERBOSE_STARTUP_LOGS:-1}"
export OPENPI_DEBUG_PHASE_LIMIT="${OPENPI_DEBUG_PHASE_LIMIT:-11003}"

exec bash "${SCRIPT_DIR}/run_6x40g_e21like_accum2_windowckpt_noactioncond_from11000_30k_20260602.sh"
