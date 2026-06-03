#!/usr/bin/env bash
set -euo pipefail

# F9c: semantic/action-adapter PCGrad after F9b rejected scalar bucket scaling.
#
# Evidence entering this run:
#   F8r: K4 task coverage + per-bucket normalization is dataflow-correct, but
#        action did not descend.
#   F9a: freezing the semantic/action adapter removes useful gradient capacity
#        and still does not descend.
#   F9b: per-bucket action EMA scaling is telemetry-correct, but action still
#        worsens by step 11020.
#   F9c pre-probe: action gradients in the semantic group have many negative
#        task-bucket cosine pairs; PICF-core structural gradients do not.
#
# Therefore this branch applies PCGrad only to the semantic/action trainable
# group.  It intentionally leaves PICF core and other gradients untouched.  Do
# not enable bucket EMA scaling here; this is a single-cause diagnostic.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_f9c_k4_semantic_pcgrad_wor_11000_to11100_20260603}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-11100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-0}"

# Same valid F8r/F9b dataflow contract.
export ACCUM_STEPS="${ACCUM_STEPS:-2}"
export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-task_uniform}"
export CALVIN_BUCKET_TEMPERATURE_ALPHA="${CALVIN_BUCKET_TEMPERATURE_ALPHA:-0.0}"
export CALVIN_BUCKET_WEIGHT_SPEC="${CALVIN_BUCKET_WEIGHT_SPEC:-}"
export CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"
export LOGICAL_BATCH_TASK_COUNT="${LOGICAL_BATCH_TASK_COUNT:-4}"
export LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-1}"
export LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-1}"

# F9c diagnostic control: gradient surgery replaces scalar loss reweighting.
export LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION="${LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION:-0}"
export LOGICAL_BATCH_GRADIENT_SURGERY="${LOGICAL_BATCH_GRADIENT_SURGERY:-pcgrad}"
export LOGICAL_BATCH_GRADIENT_SURGERY_GROUPS="${LOGICAL_BATCH_GRADIENT_SURGERY_GROUPS:-semantic}"
export LOGICAL_BATCH_GRADIENT_SURGERY_EPS="${LOGICAL_BATCH_GRADIENT_SURGERY_EPS:-1e-12}"

# Keep adapter trainable, unlike F9a.
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-action_head_and_adapter}"

exec bash "${SCRIPT_DIR}/run_a7_f8_k4_action_adapter_taskuniform_from11000_11100_20260602.sh"
