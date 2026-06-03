#!/usr/bin/env bash
set -euo pipefail

# F9b: controlled adapter training through per-bucket action EMA scaling.
#
# Evidence entering this run:
#   F7b: action gradients conflict in the semantic/action adapter group.
#   F8r: K4 task coverage + adapter training is dataflow-correct but action
#        remains flat.
#   F9a: freezing the adapter weakens gradients and still does not improve
#        action.
#
# Therefore F9b keeps the adapter trainable but controls the measured-conflict
# path by equalizing action-gradient scale across task buckets using a bounded
# EMA.  This is the cheap, scalable branch before implementing full per-bucket
# PCGrad/CAGrad.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_f9b_k4_action_adapter_bucketema_wor_11000_to11100_20260602}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-11100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-0}"

# Same valid F8r dataflow contract.
export ACCUM_STEPS="${ACCUM_STEPS:-2}"
export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-task_uniform}"
export CALVIN_BUCKET_TEMPERATURE_ALPHA="${CALVIN_BUCKET_TEMPERATURE_ALPHA:-0.0}"
export CALVIN_BUCKET_WEIGHT_SPEC="${CALVIN_BUCKET_WEIGHT_SPEC:-}"
export CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"
export LOGICAL_BATCH_TASK_COUNT="${LOGICAL_BATCH_TASK_COUNT:-4}"
export LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-1}"
export LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-1}"

# F9b diagnostic control.  For this short 100-step gate use a faster EMA than
# the likely production value; if positive, production can retest 0.95-0.98.
export LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION="${LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION:-1}"
export LOGICAL_BATCH_ACTION_BUCKET_EMA_DECAY="${LOGICAL_BATCH_ACTION_BUCKET_EMA_DECAY:-0.90}"
export LOGICAL_BATCH_ACTION_BUCKET_SCALE_MIN="${LOGICAL_BATCH_ACTION_BUCKET_SCALE_MIN:-0.50}"
export LOGICAL_BATCH_ACTION_BUCKET_SCALE_MAX="${LOGICAL_BATCH_ACTION_BUCKET_SCALE_MAX:-1.50}"
export LOGICAL_BATCH_ACTION_BUCKET_MIN_COUNT="${LOGICAL_BATCH_ACTION_BUCKET_MIN_COUNT:-2}"

# Keep adapter trainable, unlike F9a.
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-action_head_and_adapter}"

exec bash "${SCRIPT_DIR}/run_a7_f8_k4_action_adapter_taskuniform_from11000_11100_20260602.sh"
