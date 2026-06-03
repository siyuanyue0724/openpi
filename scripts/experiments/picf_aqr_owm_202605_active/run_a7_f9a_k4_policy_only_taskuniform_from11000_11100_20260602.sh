#!/usr/bin/env bash
set -euo pipefail

# F9a: adapter-insulation control after repaired F8r.
#
# Repaired F8r proved that K4 no-replacement task coverage and per-bucket
# normalization are correctly wired, but action did not descend.  F7b measured
# negative action-gradient cosine in the semantic/action adapter group, while
# action gradients into PICF core were zero.  F9a therefore removes only the
# semantic/action adapter training pressure and keeps the rest of the F8r
# contract unchanged.
#
# Question:
#   Does freezing the semantic/action adapter let the policy/action head recover
#   action descent on the same checkpoint window?
#
# If yes:
#   The bottleneck is semantic/action adapter gradient conflict.
# If no:
#   The bottleneck is not solved by adapter insulation; move to adapter-only
#   gradient control or action-condition calibration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_f9a_k4_policy_only_taskuniform_wor_11000_to11100_20260602}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-11100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-0}"

# Keep the repaired F8r logical-batch dataflow exactly.
export ACCUM_STEPS="${ACCUM_STEPS:-2}"
export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-task_uniform}"
export CALVIN_BUCKET_TEMPERATURE_ALPHA="${CALVIN_BUCKET_TEMPERATURE_ALPHA:-0.0}"
export CALVIN_BUCKET_WEIGHT_SPEC="${CALVIN_BUCKET_WEIGHT_SPEC:-}"
export CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"
export LOGICAL_BATCH_TASK_COUNT="${LOGICAL_BATCH_TASK_COUNT:-4}"
export LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-1}"
export LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-1}"

# Keep the action objective identical to F8r.
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-4.0}"
export ACTION_POS_WEIGHT="${ACTION_POS_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
export ACTION_ROT_WEIGHT="${ACTION_ROT_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
export ACTION_GRIPPER_WEIGHT="${ACTION_GRIPPER_WEIGHT:-${ACTION_LOSS_WEIGHT}}"

# Insulation variable under test: freeze the semantic/action adapter group.
export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-0}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-action_head_and_adapter}"

# Keep the policy/action head trainable and PICF at the same very-low LR as F8r.
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.001}"
export PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-constant}"

exec bash "${SCRIPT_DIR}/run_a7_f8_k4_action_adapter_taskuniform_from11000_11100_20260602.sh"
