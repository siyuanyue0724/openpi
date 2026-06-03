#!/usr/bin/env bash
set -euo pipefail

# Slot-comprehensive frozen-policy validation.
#
# This is deliberately not an anchor_only probe.  It freezes large pretrained
# perception and PaliGemma/action pressure, but leaves the full PICF
# slot/router/posterior/OEML stack trainable.
#
# Do not use --use-foundation-backbones here: that convenience flag sets
# semantic_trainable=True.  Spell out the real encoders manually.
#
# The maintained action-aware smoke wrapper reuses this exact command and only
# overrides SEMANTIC_TRAINABLE / SEMANTIC_LR_SCALE / ACTION_* weights.  Keep the
# slot, sidecar, owner-transport, and dedup arguments identical so the smoke
# tests action/semantic pressure rather than a different slot architecture.

REPO_ROOT="${REPO_ROOT:-/root/openpi_slot_quality_ea2c5f2}"
cd "${REPO_ROOT}"

EXP="${EXP:-picf_a7_slot_comprehensive_frozen_policy_1000_20260519}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_mask_1000_20260518}"
LOG="/mnt/picf_run_logs/${EXP}.log"
CALVIN_ROOT="${CALVIN_ROOT:-/mnt/calvin_data/task_ABC_D}"
PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SEGMENTS="${SEGMENTS:-0,1,2,3}"
TRAINING_STRATEGY="${TRAINING_STRATEGY:-ddp}"
OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"
SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-0}"
SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-backbone_only}"
SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-1e-6}"
SEMANTIC_GRADIENT_CHECKPOINTING="${SEMANTIC_GRADIENT_CHECKPOINTING:-1}"
PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-1.0}"
POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
SEMANTIC_MAX_LENGTH="${SEMANTIC_MAX_LENGTH:-256}"
VJEPA_FEATURE_CACHE_MODE="${VJEPA_FEATURE_CACHE_MODE:-off}"
VJEPA_FEATURE_CACHE_ROOT="${VJEPA_FEATURE_CACHE_ROOT:-/mnt/picf_frozen_feature_cache/vjepa}"
VJEPA_FEATURE_CACHE_TEMPORAL_SLICES="${VJEPA_FEATURE_CACHE_TEMPORAL_SLICES:-4}"
VJEPA_FEATURE_CACHE_STORAGE_DTYPE="${VJEPA_FEATURE_CACHE_STORAGE_DTYPE:-bfloat16}"
ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-0.0}"
ACTION_POS_WEIGHT="${ACTION_POS_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
ACTION_ROT_WEIGHT="${ACTION_ROT_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
ACTION_GRIPPER_WEIGHT="${ACTION_GRIPPER_WEIGHT:-${ACTION_LOSS_WEIGHT}}"
ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-none}"
ACTION_PREFIX_RMS_TARGET="${ACTION_PREFIX_RMS_TARGET:-1.0}"
ACTION_PREFIX_NORM_EPS="${ACTION_PREFIX_NORM_EPS:-1e-6}"
ACTION_PREFIX_VALUE_CLIP="${ACTION_PREFIX_VALUE_CLIP:-0.0}"
ACTION_PREFIX_OUTPUT_GATE="${ACTION_PREFIX_OUTPUT_GATE:-1.0}"
ACTION_PREFIX_TEACHER_MODE="${ACTION_PREFIX_TEACHER_MODE:-off}"
ACTION_PREFIX_TEACHER_EMA_DECAY="${ACTION_PREFIX_TEACHER_EMA_DECAY:-0.99}"
ACTION_PREFIX_TEACHER_BLEND="${ACTION_PREFIX_TEACHER_BLEND:-1.0}"
LAMBDA_ACTION_PREFIX_TRUST="${LAMBDA_ACTION_PREFIX_TRUST:-0.0}"
ACTION_PREFIX_STOPGRAD="${ACTION_PREFIX_STOPGRAD:-1}"
ACTION_CONTEXT_TOKENS="${ACTION_CONTEXT_TOKENS:-0}"
ACTION_CONTEXT_INTEGRATION="${ACTION_CONTEXT_INTEGRATION:-append}"
ACTION_CONTEXT_STOPGRAD="${ACTION_CONTEXT_STOPGRAD:-1}"
ACTION_CONTEXT_NORM_MODE="${ACTION_CONTEXT_NORM_MODE:-rmsnorm}"
ACTION_CONTEXT_RMS_TARGET="${ACTION_CONTEXT_RMS_TARGET:-1.0}"
ACTION_CONTEXT_OUTPUT_GATE="${ACTION_CONTEXT_OUTPUT_GATE:-0.25}"
ACTION_CONTEXT_INCLUDE_QUERY_TOKENS="${ACTION_CONTEXT_INCLUDE_QUERY_TOKENS:-0}"
SEMANTIC_ACTION_CONTEXT_ADAPTER_GATE_INIT="${SEMANTIC_ACTION_CONTEXT_ADAPTER_GATE_INIT:--2.0}"
SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP="${SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP:-1}"
SEMANTIC_ACTION_FLOW_LOSS="${SEMANTIC_ACTION_FLOW_LOSS:-mse}"
SEMANTIC_ACTION_FLOW_HUBER_DELTA="${SEMANTIC_ACTION_FLOW_HUBER_DELTA:-1.0}"
SEMANTIC_ACTION_FLOW_TIME_ALPHA="${SEMANTIC_ACTION_FLOW_TIME_ALPHA:-1.5}"
SEMANTIC_ACTION_FLOW_TIME_BETA="${SEMANTIC_ACTION_FLOW_TIME_BETA:-1.0}"
SEMANTIC_ACTION_EXPERT_ROUTER_ENABLED="${SEMANTIC_ACTION_EXPERT_ROUTER_ENABLED:-0}"
SEMANTIC_ACTION_EXPERT_ROUTER_EXPERTS="${SEMANTIC_ACTION_EXPERT_ROUTER_EXPERTS:-4}"
SEMANTIC_ACTION_EXPERT_ROUTER_RANK="${SEMANTIC_ACTION_EXPERT_ROUTER_RANK:-64}"
SEMANTIC_ACTION_EXPERT_ROUTER_GATE_INIT="${SEMANTIC_ACTION_EXPERT_ROUTER_GATE_INIT:--2.5}"
SEMANTIC_ACTION_EXPERT_ROUTER_TEMPERATURE="${SEMANTIC_ACTION_EXPERT_ROUTER_TEMPERATURE:-1.0}"
SEMANTIC_ACTION_EXPERT_ROUTER_RMS_CAP="${SEMANTIC_ACTION_EXPERT_ROUTER_RMS_CAP:-1}"
PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-constant}"
PICF_CORE_LR_BLOCK_START_STEP="${PICF_CORE_LR_BLOCK_START_STEP:-0}"
PICF_CORE_LR_BLOCK_CYCLE_STEPS="${PICF_CORE_LR_BLOCK_CYCLE_STEPS:-0}"
PICF_CORE_LR_BLOCK_ACTIVE_STEPS="${PICF_CORE_LR_BLOCK_ACTIVE_STEPS:-0}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-0}"
CALVIN_BUCKET_SAMPLING_MODE="${CALVIN_BUCKET_SAMPLING_MODE:-round_robin}"
CALVIN_BUCKET_TEMPERATURE_ALPHA="${CALVIN_BUCKET_TEMPERATURE_ALPHA:-0.0}"
CALVIN_BUCKET_WEIGHT_SPEC="${CALVIN_BUCKET_WEIGHT_SPEC:-}"
CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT="${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT:-1}"
LOGICAL_BATCH_TASK_COUNT="${LOGICAL_BATCH_TASK_COUNT:-0}"
LOGICAL_BATCH_BUCKET_NORMALIZATION="${LOGICAL_BATCH_BUCKET_NORMALIZATION:-0}"
LOGICAL_BATCH_LOG_BUCKET_METRICS="${LOGICAL_BATCH_LOG_BUCKET_METRICS:-0}"
LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION="${LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION:-0}"
LOGICAL_BATCH_ACTION_BUCKET_EMA_DECAY="${LOGICAL_BATCH_ACTION_BUCKET_EMA_DECAY:-0.98}"
LOGICAL_BATCH_ACTION_BUCKET_SCALE_MIN="${LOGICAL_BATCH_ACTION_BUCKET_SCALE_MIN:-0.5}"
LOGICAL_BATCH_ACTION_BUCKET_SCALE_MAX="${LOGICAL_BATCH_ACTION_BUCKET_SCALE_MAX:-1.5}"
LOGICAL_BATCH_ACTION_BUCKET_MIN_COUNT="${LOGICAL_BATCH_ACTION_BUCKET_MIN_COUNT:-2}"
LOGICAL_BATCH_DYNAMIC_MIXING="${LOGICAL_BATCH_DYNAMIC_MIXING:-0}"
LOGICAL_BATCH_DYNAMIC_MIXING_DECAY="${LOGICAL_BATCH_DYNAMIC_MIXING_DECAY:-0.95}"
LOGICAL_BATCH_DYNAMIC_MIXING_WARMUP_STEPS="${LOGICAL_BATCH_DYNAMIC_MIXING_WARMUP_STEPS:-50}"
LOGICAL_BATCH_DYNAMIC_MIXING_MIN_COUNT="${LOGICAL_BATCH_DYNAMIC_MIXING_MIN_COUNT:-2}"
LOGICAL_BATCH_DYNAMIC_MIXING_ETA="${LOGICAL_BATCH_DYNAMIC_MIXING_ETA:-0.25}"
LOGICAL_BATCH_DYNAMIC_MIXING_GAMMA="${LOGICAL_BATCH_DYNAMIC_MIXING_GAMMA:-0.5}"
LOGICAL_BATCH_DYNAMIC_MIXING_CLIP="${LOGICAL_BATCH_DYNAMIC_MIXING_CLIP:-2.0}"
LOGICAL_BATCH_DYNAMIC_MIXING_MIN_MASS_FRACTION="${LOGICAL_BATCH_DYNAMIC_MIXING_MIN_MASS_FRACTION:-0.05}"
LOGICAL_BATCH_DYNAMIC_MIXING_MAX_WEIGHT="${LOGICAL_BATCH_DYNAMIC_MIXING_MAX_WEIGHT:-0.35}"
LOGICAL_BATCH_GRADIENT_SURGERY="${LOGICAL_BATCH_GRADIENT_SURGERY:-off}"
LOGICAL_BATCH_GRADIENT_SURGERY_GROUPS="${LOGICAL_BATCH_GRADIENT_SURGERY_GROUPS:-semantic}"
LOGICAL_BATCH_GRADIENT_SURGERY_EPS="${LOGICAL_BATCH_GRADIENT_SURGERY_EPS:-1e-12}"
LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_ALPHA="${LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_ALPHA:-0.4}"
LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_ITERS="${LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_ITERS:-20}"
LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_RESCALE="${LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_RESCALE:-1}"
WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-0}"
FSDP_SYNC_EACH_ACCUM_MICRO="${FSDP_SYNC_EACH_ACCUM_MICRO:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PICF_TRAINABLE_SCOPE="${PICF_TRAINABLE_SCOPE:-all}"
LAMBDA_ANCHOR_PV="${LAMBDA_ANCHOR_PV:-0.0}"
LAMBDA_ANCHOR_OBJECT_PULL="${LAMBDA_ANCHOR_OBJECT_PULL:-0.35}"
LAMBDA_PV_WEAK="${LAMBDA_PV_WEAK:-0.0}"
LAMBDA_MAPG_CYCLE="${LAMBDA_MAPG_CYCLE:-0.01}"
LAMBDA_MAPG_ROUTING="${LAMBDA_MAPG_ROUTING:-0.0}"
LAMBDA_MAPG_SUPPORT_DIVERSITY="${LAMBDA_MAPG_SUPPORT_DIVERSITY:-0.005}"
LAMBDA_SLOT_QUALITY="${LAMBDA_SLOT_QUALITY:-0.05}"
LAMBDA_OBJECT_EXPLANATION_POINT="${LAMBDA_OBJECT_EXPLANATION_POINT:-0.02}"
LAMBDA_OBJECT_EXPLANATION_CONTACT="${LAMBDA_OBJECT_EXPLANATION_CONTACT:-0.01}"
LAMBDA_OBJECT_EXPLANATION_DUPLICATE="${LAMBDA_OBJECT_EXPLANATION_DUPLICATE:-0.01}"
LAMBDA_OBJECT_EXPLANATION_BACKGROUND="${LAMBDA_OBJECT_EXPLANATION_BACKGROUND:-0.005}"
OBJECT_SCAFFOLD_DECAY_MODE="${OBJECT_SCAFFOLD_DECAY_MODE:-none}"
OBJECT_SCAFFOLD_DECAY_START_STEP="${OBJECT_SCAFFOLD_DECAY_START_STEP:-0}"
OBJECT_SCAFFOLD_DECAY_END_STEP="${OBJECT_SCAFFOLD_DECAY_END_STEP:-0}"
OBJECT_SCAFFOLD_DECAY_FLOOR="${OBJECT_SCAFFOLD_DECAY_FLOOR:-1.0}"
ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M:-0.08}"
ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN:-0.05}"
ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER="${ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER:-1.0}"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

SEMANTIC_TRAINABLE_ARGS=()
case "${SEMANTIC_TRAINABLE}" in
  1|true|TRUE|yes|YES|on|ON)
    SEMANTIC_TRAINABLE_ARGS+=(--semantic-trainable)
    ;;
esac

SEMANTIC_GRADIENT_CHECKPOINTING_ARGS=()
case "${SEMANTIC_GRADIENT_CHECKPOINTING}" in
  1|true|TRUE|yes|YES|on|ON)
    SEMANTIC_GRADIENT_CHECKPOINTING_ARGS+=(--semantic-gradient-checkpointing)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    SEMANTIC_GRADIENT_CHECKPOINTING_ARGS+=(--no-semantic-gradient-checkpointing)
    ;;
  *)
    echo "Unsupported SEMANTIC_GRADIENT_CHECKPOINTING=${SEMANTIC_GRADIENT_CHECKPOINTING}; use 0/1." >&2
    exit 2
    ;;
esac

ACTION_PREFIX_STOPGRAD_ARGS=()
case "${ACTION_PREFIX_STOPGRAD}" in
  1|true|TRUE|yes|YES|on|ON)
    ACTION_PREFIX_STOPGRAD_ARGS+=(--picf-action-prefix-stopgrad)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ACTION_PREFIX_STOPGRAD_ARGS+=(--no-picf-action-prefix-stopgrad)
    ;;
  *)
    echo "Unsupported SEMANTIC_GRADIENT_CHECKPOINTING=${SEMANTIC_GRADIENT_CHECKPOINTING}; use 0/1." >&2
    exit 2
    ;;
esac

SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP_ARGS=()
case "${SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP}" in
  1|true|TRUE|yes|YES|on|ON)
    SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP_ARGS+=(--semantic-action-context-adapter-rms-cap)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP_ARGS+=(--no-semantic-action-context-adapter-rms-cap)
    ;;
  *)
    echo "Unsupported SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP=${SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP}; use 0/1." >&2
    exit 2
    ;;
esac

SEMANTIC_ACTION_EXPERT_ROUTER_ARGS=()
case "${SEMANTIC_ACTION_EXPERT_ROUTER_ENABLED}" in
  1|true|TRUE|yes|YES|on|ON)
    SEMANTIC_ACTION_EXPERT_ROUTER_ARGS+=(--semantic-action-expert-router-enabled)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    SEMANTIC_ACTION_EXPERT_ROUTER_ARGS+=(--no-semantic-action-expert-router-enabled)
    ;;
  *)
    echo "Unsupported SEMANTIC_ACTION_EXPERT_ROUTER_ENABLED=${SEMANTIC_ACTION_EXPERT_ROUTER_ENABLED}; use 0/1." >&2
    exit 2
    ;;
esac

SEMANTIC_ACTION_EXPERT_ROUTER_RMS_CAP_ARGS=()
case "${SEMANTIC_ACTION_EXPERT_ROUTER_RMS_CAP}" in
  1|true|TRUE|yes|YES|on|ON)
    SEMANTIC_ACTION_EXPERT_ROUTER_RMS_CAP_ARGS+=(--semantic-action-expert-router-rms-cap)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    SEMANTIC_ACTION_EXPERT_ROUTER_RMS_CAP_ARGS+=(--no-semantic-action-expert-router-rms-cap)
    ;;
  *)
    echo "Unsupported SEMANTIC_ACTION_EXPERT_ROUTER_RMS_CAP=${SEMANTIC_ACTION_EXPERT_ROUTER_RMS_CAP}; use 0/1." >&2
    exit 2
    ;;
esac

ACTION_CONTEXT_STOPGRAD_ARGS=()
case "${ACTION_CONTEXT_STOPGRAD}" in
  1|true|TRUE|yes|YES|on|ON)
    ACTION_CONTEXT_STOPGRAD_ARGS+=(--picf-action-context-stopgrad)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ACTION_CONTEXT_STOPGRAD_ARGS+=(--no-picf-action-context-stopgrad)
    ;;
  *)
    echo "Unsupported ACTION_CONTEXT_STOPGRAD=${ACTION_CONTEXT_STOPGRAD}; use 0/1." >&2
    exit 2
    ;;
esac

ACTION_CONTEXT_QUERY_ARGS=()
case "${ACTION_CONTEXT_INCLUDE_QUERY_TOKENS}" in
  1|true|TRUE|yes|YES|on|ON)
    ACTION_CONTEXT_QUERY_ARGS+=(--action-context-include-query-tokens)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ACTION_CONTEXT_QUERY_ARGS+=(--no-action-context-include-query-tokens)
    ;;
  *)
    echo "Unsupported ACTION_CONTEXT_INCLUDE_QUERY_TOKENS=${ACTION_CONTEXT_INCLUDE_QUERY_TOKENS}; use 0/1." >&2
    exit 2
    ;;
esac

PICF_ACTION_CONDITION_ARGS=()
case "${PICF_ACTION_CONDITION_ENABLED:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    PICF_ACTION_CONDITION_ARGS+=(--picf-action-condition-enabled)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    PICF_ACTION_CONDITION_ARGS+=(--no-picf-action-condition-enabled)
    ;;
  *)
    echo "Unsupported PICF_ACTION_CONDITION_ENABLED=${PICF_ACTION_CONDITION_ENABLED}; use 0/1." >&2
    exit 2
    ;;
esac

CALVIN_BALANCED_BUCKET_ARGS=()
case "${CALVIN_BALANCED_BUCKET_SAMPLER}" in
  1|true|TRUE|yes|YES|on|ON)
    CALVIN_BALANCED_BUCKET_ARGS+=(--calvin-balanced-bucket-sampler)
    CALVIN_BALANCED_BUCKET_ARGS+=(--calvin-bucket-sampling-mode "${CALVIN_BUCKET_SAMPLING_MODE}")
    CALVIN_BALANCED_BUCKET_ARGS+=(--calvin-bucket-temperature-alpha "${CALVIN_BUCKET_TEMPERATURE_ALPHA}")
    case "${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT}" in
      1|true|TRUE|yes|YES|on|ON)
        CALVIN_BALANCED_BUCKET_ARGS+=(--calvin-bucket-sample-without-replacement)
        ;;
      0|false|FALSE|no|NO|off|OFF)
        CALVIN_BALANCED_BUCKET_ARGS+=(--no-calvin-bucket-sample-without-replacement)
        ;;
      *)
        echo "Unsupported CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT=${CALVIN_BUCKET_SAMPLE_WITHOUT_REPLACEMENT}; use 0/1." >&2
        exit 2
        ;;
    esac
    if [[ -n "${CALVIN_BUCKET_WEIGHT_SPEC}" ]]; then
      CALVIN_BALANCED_BUCKET_ARGS+=(--calvin-bucket-weight-spec "${CALVIN_BUCKET_WEIGHT_SPEC}")
    fi
    ;;
  0|false|FALSE|no|NO|off|OFF)
    CALVIN_BALANCED_BUCKET_ARGS+=(--no-calvin-balanced-bucket-sampler)
    ;;
  *)
    echo "Unsupported CALVIN_BALANCED_BUCKET_SAMPLER=${CALVIN_BALANCED_BUCKET_SAMPLER}; use 0/1." >&2
    exit 2
    ;;
esac

LOGICAL_BATCH_ARGS=()
if [[ "${LOGICAL_BATCH_TASK_COUNT}" != "0" ]]; then
  LOGICAL_BATCH_ARGS+=(--logical-batch-task-count "${LOGICAL_BATCH_TASK_COUNT}")
fi
case "${LOGICAL_BATCH_BUCKET_NORMALIZATION}" in
  1|true|TRUE|yes|YES|on|ON)
    LOGICAL_BATCH_ARGS+=(--logical-batch-bucket-normalization)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    LOGICAL_BATCH_ARGS+=(--no-logical-batch-bucket-normalization)
    ;;
  *)
    echo "Unsupported LOGICAL_BATCH_BUCKET_NORMALIZATION=${LOGICAL_BATCH_BUCKET_NORMALIZATION}; use 0/1." >&2
    exit 2
    ;;
esac
case "${LOGICAL_BATCH_LOG_BUCKET_METRICS}" in
  1|true|TRUE|yes|YES|on|ON)
    LOGICAL_BATCH_ARGS+=(--logical-batch-log-bucket-metrics)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    LOGICAL_BATCH_ARGS+=(--no-logical-batch-log-bucket-metrics)
    ;;
  *)
    echo "Unsupported LOGICAL_BATCH_LOG_BUCKET_METRICS=${LOGICAL_BATCH_LOG_BUCKET_METRICS}; use 0/1." >&2
    exit 2
    ;;
esac
case "${LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION}" in
  1|true|TRUE|yes|YES|on|ON)
    LOGICAL_BATCH_ARGS+=(--logical-batch-action-bucket-ema-normalization)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    LOGICAL_BATCH_ARGS+=(--no-logical-batch-action-bucket-ema-normalization)
    ;;
  *)
    echo "Unsupported LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION=${LOGICAL_BATCH_ACTION_BUCKET_EMA_NORMALIZATION}; use 0/1." >&2
    exit 2
    ;;
esac
case "${LOGICAL_BATCH_DYNAMIC_MIXING}" in
  1|true|TRUE|yes|YES|on|ON)
    LOGICAL_BATCH_ARGS+=(--logical-batch-dynamic-mixing)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    LOGICAL_BATCH_ARGS+=(--no-logical-batch-dynamic-mixing)
    ;;
  *)
    echo "Unsupported LOGICAL_BATCH_DYNAMIC_MIXING=${LOGICAL_BATCH_DYNAMIC_MIXING}; use 0/1." >&2
    exit 2
    ;;
esac
LOGICAL_BATCH_ARGS+=(
  --logical-batch-action-bucket-ema-decay "${LOGICAL_BATCH_ACTION_BUCKET_EMA_DECAY}"
  --logical-batch-action-bucket-scale-min "${LOGICAL_BATCH_ACTION_BUCKET_SCALE_MIN}"
  --logical-batch-action-bucket-scale-max "${LOGICAL_BATCH_ACTION_BUCKET_SCALE_MAX}"
  --logical-batch-action-bucket-min-count "${LOGICAL_BATCH_ACTION_BUCKET_MIN_COUNT}"
  --logical-batch-dynamic-mixing-decay "${LOGICAL_BATCH_DYNAMIC_MIXING_DECAY}"
  --logical-batch-dynamic-mixing-warmup-steps "${LOGICAL_BATCH_DYNAMIC_MIXING_WARMUP_STEPS}"
  --logical-batch-dynamic-mixing-min-count "${LOGICAL_BATCH_DYNAMIC_MIXING_MIN_COUNT}"
  --logical-batch-dynamic-mixing-eta "${LOGICAL_BATCH_DYNAMIC_MIXING_ETA}"
  --logical-batch-dynamic-mixing-gamma "${LOGICAL_BATCH_DYNAMIC_MIXING_GAMMA}"
  --logical-batch-dynamic-mixing-clip "${LOGICAL_BATCH_DYNAMIC_MIXING_CLIP}"
  --logical-batch-dynamic-mixing-min-mass-fraction "${LOGICAL_BATCH_DYNAMIC_MIXING_MIN_MASS_FRACTION}"
  --logical-batch-dynamic-mixing-max-weight "${LOGICAL_BATCH_DYNAMIC_MIXING_MAX_WEIGHT}"
  --logical-batch-gradient-surgery "${LOGICAL_BATCH_GRADIENT_SURGERY}"
  --logical-batch-gradient-surgery-groups "${LOGICAL_BATCH_GRADIENT_SURGERY_GROUPS}"
  --logical-batch-gradient-surgery-eps "${LOGICAL_BATCH_GRADIENT_SURGERY_EPS}"
  --logical-batch-gradient-surgery-cagrad-alpha "${LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_ALPHA}"
  --logical-batch-gradient-surgery-cagrad-iters "${LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_ITERS}"
)
case "${LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_RESCALE}" in
  1|true|TRUE|yes|YES|on|ON)
    LOGICAL_BATCH_ARGS+=(--logical-batch-gradient-surgery-cagrad-rescale)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    LOGICAL_BATCH_ARGS+=(--no-logical-batch-gradient-surgery-cagrad-rescale)
    ;;
  *)
    echo "Unsupported LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_RESCALE=${LOGICAL_BATCH_GRADIENT_SURGERY_CAGRAD_RESCALE}; use 0/1." >&2
    exit 2
    ;;
esac

WINDOW_ACTIVATION_CHECKPOINTING_ARGS=()
case "${WINDOW_ACTIVATION_CHECKPOINTING}" in
  1|true|TRUE|yes|YES|on|ON)
    WINDOW_ACTIVATION_CHECKPOINTING_ARGS+=(--window-activation-checkpointing)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported WINDOW_ACTIVATION_CHECKPOINTING=${WINDOW_ACTIVATION_CHECKPOINTING}; use 0/1." >&2
    exit 2
    ;;
esac

FSDP_SYNC_EACH_ACCUM_MICRO_ARGS=()
case "${FSDP_SYNC_EACH_ACCUM_MICRO}" in
  1|true|TRUE|yes|YES|on|ON)
    FSDP_SYNC_EACH_ACCUM_MICRO_ARGS+=(--fsdp-sync-each-accum-micro)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    FSDP_SYNC_EACH_ACCUM_MICRO_ARGS+=(--no-fsdp-sync-each-accum-micro)
    ;;
  *)
    echo "Unsupported FSDP_SYNC_EACH_ACCUM_MICRO=${FSDP_SYNC_EACH_ACCUM_MICRO}; use 0/1." >&2
    exit 2
    ;;
esac

ANCHOR_OVERLAY_SIGNATURE_ARGS=()
if [[ "${ANCHOR_OVERLAY_INTERVAL}" != "0" ]]; then
  ANCHOR_OVERLAY_SIGNATURE_ARGS+=(--anchor-overlay-dump-signatures)
fi

mkdir -p /mnt/picf_run_logs /mnt/checkpoints/picf_core

export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/picf_core_train.py \
  --calvin-root "${CALVIN_ROOT}" \
  --backend dir \
  --split training \
  --calvin-segment-indices "${SEGMENTS}" \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --point-backbone sonata \
  --visual-mode encoder \
  --visual-feature-mode hierarchical \
  --tactile-mode encoder \
  --use-tactile \
  --semantic-mode paligemma \
  --semantic-source auto \
  "${SEMANTIC_TRAINABLE_ARGS[@]}" \
  --semantic-trainable-scope "${SEMANTIC_TRAINABLE_SCOPE}" \
  "${SEMANTIC_GRADIENT_CHECKPOINTING_ARGS[@]}" \
  --semantic-action-context-adapter-gate-init "${SEMANTIC_ACTION_CONTEXT_ADAPTER_GATE_INIT}" \
  "${SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP_ARGS[@]}" \
  --semantic-action-flow-loss "${SEMANTIC_ACTION_FLOW_LOSS}" \
  --semantic-action-flow-huber-delta "${SEMANTIC_ACTION_FLOW_HUBER_DELTA}" \
  --semantic-action-flow-time-alpha "${SEMANTIC_ACTION_FLOW_TIME_ALPHA}" \
  --semantic-action-flow-time-beta "${SEMANTIC_ACTION_FLOW_TIME_BETA}" \
  "${SEMANTIC_ACTION_EXPERT_ROUTER_ARGS[@]}" \
  --semantic-action-expert-router-experts "${SEMANTIC_ACTION_EXPERT_ROUTER_EXPERTS}" \
  --semantic-action-expert-router-rank "${SEMANTIC_ACTION_EXPERT_ROUTER_RANK}" \
  --semantic-action-expert-router-gate-init "${SEMANTIC_ACTION_EXPERT_ROUTER_GATE_INIT}" \
  --semantic-action-expert-router-temperature "${SEMANTIC_ACTION_EXPERT_ROUTER_TEMPERATURE}" \
  "${SEMANTIC_ACTION_EXPERT_ROUTER_RMS_CAP_ARGS[@]}" \
  --semantic-max-length "${SEMANTIC_MAX_LENGTH}" \
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --vjepa-feature-cache-mode "${VJEPA_FEATURE_CACHE_MODE}" \
  --vjepa-feature-cache-root "${VJEPA_FEATURE_CACHE_ROOT}" \
  --vjepa-feature-cache-temporal-slices "${VJEPA_FEATURE_CACHE_TEMPORAL_SLICES}" \
  --vjepa-feature-cache-storage-dtype "${VJEPA_FEATURE_CACHE_STORAGE_DTYPE}" \
  --tactile-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --tactile-backgrounds-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_backgrounds.npz \
  --tactile-calibration-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_fingertip_calibration.json \
  --tactile-contact-stats-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_contact_stats.json \
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --semantic-checkpoint-path /root/openpi/checkpoints/foundation/pi05_base_pytorch \
  --action-norm-stats-path /root/openpi_posterior_vla_clean/assets/pi05_calvin_sonata/calvin/norm_stats.json \
  --prompt-state-norm-stats-path /root/openpi_posterior_vla_clean/assets/pi05_calvin_sonata/calvin/norm_stats.json \
  --training-strategy "${TRAINING_STRATEGY}" \
  --optimizer-sharding "${OPTIMIZER_SHARDING}" \
  --optimizer-checkpoint-mode "${OPTIMIZER_CHECKPOINT_MODE}" \
  "${RESUME_ARGS[@]}" \
  --perception-finetune-mode frozen \
  --picf-trainable-scope "${PICF_TRAINABLE_SCOPE}" \
  --action-horizon 16 \
  --max-points 1024 \
  --accum-steps "${ACCUM_STEPS}" \
  "${CALVIN_BALANCED_BUCKET_ARGS[@]}" \
  "${LOGICAL_BATCH_ARGS[@]}" \
  "${WINDOW_ACTIVATION_CHECKPOINTING_ARGS[@]}" \
  "${FSDP_SYNC_EACH_ACCUM_MICRO_ARGS[@]}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --weight-decay 1e-4 \
  --semantic-lr-scale "${SEMANTIC_LR_SCALE}" \
  --picf-core-lr-scale "${PICF_CORE_LR_SCALE}" \
  --policy-head-lr-scale "${POLICY_HEAD_LR_SCALE}" \
  --unroll-steps 2 \
  --burnin-steps 1 \
  --burnin-mode state_only \
  --lambda-action-pos "${ACTION_POS_WEIGHT}" \
  --lambda-action-rot "${ACTION_ROT_WEIGHT}" \
  --lambda-action-gripper "${ACTION_GRIPPER_WEIGHT}" \
  "${PICF_ACTION_CONDITION_ARGS[@]}" \
  "${ACTION_PREFIX_STOPGRAD_ARGS[@]}" \
  --action-prefix-norm-mode "${ACTION_PREFIX_NORM_MODE}" \
  --action-prefix-rms-target "${ACTION_PREFIX_RMS_TARGET}" \
  --action-prefix-norm-eps "${ACTION_PREFIX_NORM_EPS}" \
  --action-prefix-value-clip "${ACTION_PREFIX_VALUE_CLIP}" \
  --action-prefix-output-gate "${ACTION_PREFIX_OUTPUT_GATE}" \
  --action-prefix-teacher-mode "${ACTION_PREFIX_TEACHER_MODE}" \
  --action-prefix-teacher-ema-decay "${ACTION_PREFIX_TEACHER_EMA_DECAY}" \
  --action-prefix-teacher-blend "${ACTION_PREFIX_TEACHER_BLEND}" \
  --lambda-action-prefix-trust "${LAMBDA_ACTION_PREFIX_TRUST}" \
  --action-context-tokens "${ACTION_CONTEXT_TOKENS}" \
  --action-context-integration "${ACTION_CONTEXT_INTEGRATION}" \
  "${ACTION_CONTEXT_STOPGRAD_ARGS[@]}" \
  --action-context-norm-mode "${ACTION_CONTEXT_NORM_MODE}" \
  --action-context-rms-target "${ACTION_CONTEXT_RMS_TARGET}" \
  --action-context-output-gate "${ACTION_CONTEXT_OUTPUT_GATE}" \
  "${ACTION_CONTEXT_QUERY_ARGS[@]}" \
  --picf-core-lr-runtime-mode "${PICF_CORE_LR_RUNTIME_MODE}" \
  --picf-core-lr-block-start-step "${PICF_CORE_LR_BLOCK_START_STEP}" \
  --picf-core-lr-block-cycle-steps "${PICF_CORE_LR_BLOCK_CYCLE_STEPS}" \
  --picf-core-lr-block-active-steps "${PICF_CORE_LR_BLOCK_ACTIVE_STEPS}" \
  --lambda-visual-latent 0.0 \
  --lambda-visual-real 0.0 \
  --lambda-tactile-real 0.0 \
  --lambda-point-real 0.0 \
  --lambda-semantic-future-aux 0.0 \
  --disable-aux-budgeting \
  --lambda-anchor-pv "${LAMBDA_ANCHOR_PV}" \
  --lambda-anchor-object-pull "${LAMBDA_ANCHOR_OBJECT_PULL}" \
  --anchor-object-pull-sigma-m 0.04 \
  --anchor-object-pull-confirmation-threshold 0.02 \
  --anchor-object-pull-allowed-roles 1 \
  --anchor-object-pull-graph-weight 1.0 \
  --anchor-object-pull-posterior-weight 1.0 \
  --anchor-object-pull-target-quality-sigma-m "${ANCHOR_OBJECT_PULL_TARGET_QUALITY_SIGMA_M}" \
  --anchor-object-pull-target-quality-min "${ANCHOR_OBJECT_PULL_TARGET_QUALITY_MIN}" \
  --anchor-object-pull-target-quality-power "${ANCHOR_OBJECT_PULL_TARGET_QUALITY_POWER}" \
  --lambda-pv-weak "${LAMBDA_PV_WEAK}" \
  --lambda-pt 0.0 \
  --lambda-vl-heatmap-task 0.0 \
  --lambda-vl-heatmap-effector 0.0 \
  --lambda-vl-heatmap-interaction 0.0 \
  --lambda-vl-point-consistency 0.0 \
  --lambda-vl-anchor-diversity 0.0 \
  --lambda-mapg-cycle "${LAMBDA_MAPG_CYCLE}" \
  --lambda-mapg-routing "${LAMBDA_MAPG_ROUTING}" \
  --lambda-mapg-support-diversity "${LAMBDA_MAPG_SUPPORT_DIVERSITY}" \
  --lambda-mapg-geometry-diversity 0.0 \
  --lambda-slot-jepa 0.0 \
  --lambda-support-pred 0.0 \
  --lambda-binding-consistency 0.0 \
  --lambda-aqr-denoising 0.0 \
  --lambda-slot-quality "${LAMBDA_SLOT_QUALITY}" \
  --lambda-vcap-unexplained 0.0 \
  --lambda-vcap-duplicate 0.0 \
  --lambda-vcap-count 0.0 \
  --lambda-vcap-continuity 0.0 \
  --lambda-object-explanation-feature 0.0 \
  --lambda-object-explanation-point "${LAMBDA_OBJECT_EXPLANATION_POINT}" \
  --lambda-object-explanation-contact "${LAMBDA_OBJECT_EXPLANATION_CONTACT}" \
  --lambda-object-explanation-duplicate "${LAMBDA_OBJECT_EXPLANATION_DUPLICATE}" \
  --lambda-object-explanation-background "${LAMBDA_OBJECT_EXPLANATION_BACKGROUND}" \
  --object-scaffold-decay-mode "${OBJECT_SCAFFOLD_DECAY_MODE}" \
  --object-scaffold-decay-start-step "${OBJECT_SCAFFOLD_DECAY_START_STEP}" \
  --object-scaffold-decay-end-step "${OBJECT_SCAFFOLD_DECAY_END_STEP}" \
  --object-scaffold-decay-floor "${OBJECT_SCAFFOLD_DECAY_FLOOR}" \
  --tracklet-memory-enabled \
  --tracklet-read-weight 0.0 \
  --proposal-memory-enabled \
  --mvtrack-sidecar-root "${SIDECAR_ROOT}" \
  --mvtrack-sidecar-proposal-nearest-max-gap 8 \
  --proposal-age-decay-steps 8.0 \
  --proposal-shape-quality-enabled \
  --proposal-shape-area-min 0.002 \
  --proposal-shape-area-max 0.35 \
  --proposal-shape-aspect-min 0.20 \
  --proposal-context-quality-power 0.50 \
  --proposal-point-bridge-weight 0.50 \
  --proposal-point-bridge-edge-tau 0.02 \
  --proposal-anchor-seed-enabled \
  --proposal-anchor-seed-pre-reader-enabled \
  --proposal-anchor-seed-rows 2 \
  --proposal-anchor-seed-weight 0.85 \
  --proposal-anchor-seed-token-weight 0.35 \
  --proposal-anchor-seed-score-floor 0.05 \
  --proposal-anchor-seed-point-topk 128 \
  --proposal-anchor-seed-point-power 1.5 \
  --task-owner-proposal-point-bridge-weight 0.50 \
  --task-owner-bias-enabled \
  --task-owner-visual-bias-weight 0.20 \
  --task-owner-proposal-bias-weight 0.50 \
  --task-owner-proposal-point-bias-weight 0.75 \
  --task-owner-proposal-objectness-power 0.50 \
  --task-owner-proposal-static-only \
  --task-owner-proposal-topk 4 \
  --task-owner-proposal-score-floor 0.05 \
  --object-candidate-assignment-enabled \
  --object-candidate-assignment-temperature 0.35 \
  --object-candidate-background-prior 0.25 \
  --object-candidate-background-quality-weight 2.0 \
  --object-candidate-row-support-floor 0.01 \
  --object-candidate-eligible-roles 1 \
  --object-candidate-max-rows-per-candidate 1 \
  --object-candidate-row-capacity 1.25 \
  --object-candidate-row-capacity-iters 10 \
  --object-candidate-point-weight 1.0 \
  --object-candidate-proposal-weight 0.75 \
  --object-candidate-seed-weight 1.25 \
  --object-candidate-task-owner-weight 0.50 \
  --object-candidate-anchor-score-weight 1.0 \
  --object-candidate-point-mix 0.80 \
  --object-candidate-proposal-mix 0.35 \
  --object-candidate-min-shape-quality 0.01 \
  --object-candidate-owner-transport-enabled \
  --object-candidate-owner-roles 1 \
  --object-candidate-owner-min-share 0.75 \
  --object-candidate-owner-point-mix 1.0 \
  --object-candidate-owner-geometry-mix 0.95 \
  --posterior-owner-transport-candidate-geometry-mix 1.0 \
  --object-explanation-point-core-mass 0.90 \
  --object-explanation-point-core-topk 128 \
  --object-explanation-point-loss-clip 8.0 \
  --tactile-attach-to-object-owner \
  --tactile-evidence-prob-floor 0.35 \
  --tactile-anchor-prob-on 0.55 \
  --aqr-role-layout object_only \
  --effector-persistent-anchors 0 \
  --effector-observation-anchors 0 \
  --task-effector-queries 0 \
  --aqr-ownership-prior-enabled \
  --aqr-ownership-prior-weight 0.70 \
  --aqr-ownership-point-prior-weight 0.70 \
  --aqr-ownership-point-prior-sigma-m 0.04 \
  --aqr-ownership-temporal-prior-weight 0.35 \
  --aqr-ownership-prior-uniform-mix 0.02 \
  --aqr-same-role-support-competition-enabled \
  --aqr-same-role-support-competition-weight 0.85 \
  --aqr-same-role-support-competition-iters 5 \
  --posterior-file-competition-enabled \
  --posterior-file-competition-min-per-role 1 \
  --posterior-file-competition-max-per-role 1 \
  --posterior-file-competition-min-support 0.02 \
  --posterior-file-competition-relative-score-threshold 0.0 \
  --posterior-file-competition-support-overlap-threshold 0.80 \
  --posterior-file-competition-geometry-duplicate-enabled \
  --posterior-file-competition-geometry-sigma-m 0.04 \
  --posterior-file-competition-geometry-threshold 0.70 \
  --posterior-birth-competition-enabled \
  --posterior-birth-competition-max-per-role 1 \
  --posterior-birth-competition-min-score 0.05 \
  --posterior-birth-competition-inactive-only \
  --posterior-birth-alpha-suppression-power 0.5 \
  --posterior-slot-identity-std 0.02 \
  --task-slot-identity-std 0.02 \
  --posterior-bootstrap-from-observation \
  --posterior-occupancy-prior-enabled \
  --posterior-occupancy-prior-weight 1.0 \
  --posterior-occupancy-prior-sigma-m 0.04 \
  --posterior-occupancy-prior-clip 4.0 \
  --observation-anchor-seed-point-mix 0.35 \
  --recycle-normalize-residual-summary \
  --recycle-residual-norm-mode layernorm \
  --posterior-slotwise-recycle-residual \
  --posterior-binding-signature-memory-enabled \
  --posterior-binding-signature-update-rate 0.20 \
  --posterior-binding-signature-update-max-rate 0.50 \
  --posterior-binding-signature-min-support 0.02 \
  --posterior-binding-signature-owner-weight 0.50 \
  --posterior-binding-signature-dispersion-gate-enabled \
  --posterior-binding-signature-measurement-min-std 0.05 \
  --posterior-binding-signature-measurement-margin-min 0.25 \
  --posterior-binding-signature-measurement-margin-temperature 0.10 \
  --posterior-owner-active-gate-enabled \
  --posterior-owner-active-min 0.30 \
  --posterior-owner-active-bias -10000.0 \
  --posterior-owner-transport-enabled \
  --posterior-owner-transport-roles 1 \
  --posterior-owner-transport-max-per-role 1 \
  --posterior-owner-transport-max-rate 0.85 \
  --posterior-owner-transport-precision-gain 8.0 \
  --posterior-owner-transport-min-mass 0.01 \
  --posterior-owner-transport-direct-candidate-assignment \
  --posterior-owner-transport-direct-candidate-min-score 0.01 \
  --posterior-owner-transport-assignment-floor 0.50 \
  --posterior-owner-transport-reliability-floor 0.50 \
  --posterior-owner-transport-covariance-scale 0.50 \
  --posterior-owner-transport-inactive-prior 0.35 \
  --posterior-owner-transport-activates-file \
  --posterior-owner-transport-active-threshold 0.05 \
  --no-legacy-local-refinement-opt-in \
  --no-local-refinement-enabled \
  --local-refinement-topk 0 \
  --local-refinement-weight 0.0 \
  --local-refinement-binding-weight 0.0 \
  --aqr-active-slot-filter-enabled \
  --aqr-active-slot-min-per-role 1 \
  --aqr-active-slot-max-per-role 2 \
  --aqr-active-slot-min-confidence 0.05 \
  --aqr-active-slot-overlap-threshold 0.40 \
  --aqr-active-slot-relative-score-threshold 0.80 \
  --aqr-active-slot-geometry-duplicate-enabled \
  --aqr-active-slot-geometry-duplicate-sigma-m 0.05 \
  --aqr-active-slot-geometry-duplicate-threshold 0.45 \
  --aqr-context-slot-enabled \
  --aqr-context-slot-weight 0.12 \
  --aqr-context-slot-min-confidence 0.03 \
  --aqr-context-slot-min-score 0.005 \
  --aqr-context-slot-duplicate-overlap-threshold 0.75 \
  --aqr-context-slot-deduplicate-enabled \
  --aqr-context-slot-max-per-role 8 \
  --aqr-context-slot-self-overlap-threshold 0.75 \
  --aqr-context-slot-self-support-overlap-enabled \
  --aqr-context-slot-self-support-overlap-threshold 0.70 \
  --aqr-context-slot-active-support-overlap-enabled \
  --aqr-context-slot-active-support-overlap-threshold 0.50 \
  --aqr-context-slot-quality-gate-enabled \
  --aqr-slot-quality-owner-active-floor 0.65 \
  --aqr-control-graph-attention-bias-enabled \
  --no-aqr-control-graph-token-scaling-enabled \
  --aqr-control-graph-state-embedding-enabled \
  --aqr-control-graph-bias-min 0.0001 \
  --grad-clip-mode fixed \
  --grad-clip-norm 5.0 \
  --anchor-overlay-interval "${ANCHOR_OVERLAY_INTERVAL}" \
  --anchor-overlay-max-anchors 64 \
  "${ANCHOR_OVERLAY_SIGNATURE_ARGS[@]}" \
  --log-interval "${LOG_INTERVAL}" \
  --save-interval "${SAVE_INTERVAL}" \
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}" \
  --progress \
  --wandb-mode disabled \
  --overwrite \
  --exp-name "${EXP}" \
  --num-train-steps "${NUM_TRAIN_STEPS:-1000}" 2>&1 | tee "${LOG}"
