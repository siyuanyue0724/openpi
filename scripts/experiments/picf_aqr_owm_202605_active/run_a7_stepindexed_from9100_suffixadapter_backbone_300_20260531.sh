#!/usr/bin/env bash
set -euo pipefail

# E9 action-interface diagnostic.
#
# E7/E8 isolate the suffix adapter and wrapper-local action head.  If both fail
# to move action loss, the next causal test is normal semantic/action-transformer
# adaptation: train the restored PI0/PaliGemma model with the suffix-side PICF
# adapter while keeping the wrapper-local flow heads frozen, matching the
# production "backbone_only" cotrain boundary.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_probe_current_20260529}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
export EXP="${EXP:-picf_a7_stepindexed_from9100_suffixadapter_backbone_h30k_20260531}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from9000_prefixfusion_9300_20260530/9100}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

# Keep the production 30K horizon even when this diagnostic is manually stopped
# after a few hundred resumed steps.  A short 9400 horizon resumes step9100 near
# min_lr and invalidates action-interface adaptation conclusions.
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

export ACTION_PREFIX_STOPGRAD="${ACTION_PREFIX_STOPGRAD:-1}"
export ACTION_PREFIX_NORM_MODE="${ACTION_PREFIX_NORM_MODE:-rmsnorm}"
export ACTION_PREFIX_OUTPUT_GATE="${ACTION_PREFIX_OUTPUT_GATE:-0.70}"
export ACTION_PREFIX_TEACHER_MODE="${ACTION_PREFIX_TEACHER_MODE:-ema}"
export ACTION_PREFIX_TEACHER_EMA_DECAY="${ACTION_PREFIX_TEACHER_EMA_DECAY:-0.99}"
export ACTION_PREFIX_TEACHER_BLEND="${ACTION_PREFIX_TEACHER_BLEND:-1.0}"
export LAMBDA_ACTION_PREFIX_TRUST="${LAMBDA_ACTION_PREFIX_TRUST:-0.02}"

export ACTION_CONTEXT_TOKENS="${ACTION_CONTEXT_TOKENS:-24}"
export ACTION_CONTEXT_INTEGRATION="${ACTION_CONTEXT_INTEGRATION:-suffix_cross_attention}"
export ACTION_CONTEXT_STOPGRAD="${ACTION_CONTEXT_STOPGRAD:-1}"
export ACTION_CONTEXT_NORM_MODE="${ACTION_CONTEXT_NORM_MODE:-rmsnorm}"
export ACTION_CONTEXT_RMS_TARGET="${ACTION_CONTEXT_RMS_TARGET:-1.0}"
export ACTION_CONTEXT_OUTPUT_GATE="${ACTION_CONTEXT_OUTPUT_GATE:-0.25}"
export ACTION_CONTEXT_INCLUDE_QUERY_TOKENS="${ACTION_CONTEXT_INCLUDE_QUERY_TOKENS:-0}"

export SEMANTIC_TRAINABLE="${SEMANTIC_TRAINABLE:-1}"
export SEMANTIC_TRAINABLE_SCOPE="${SEMANTIC_TRAINABLE_SCOPE:-backbone_only}"
export SEMANTIC_ACTION_CONTEXT_ADAPTER_GATE_INIT="${SEMANTIC_ACTION_CONTEXT_ADAPTER_GATE_INIT:--2.0}"
export SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP="${SEMANTIC_ACTION_CONTEXT_ADAPTER_RMS_CAP:-1}"
export SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-1.0}"

export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
export LR="${LR:-7e-5}"
export MIN_LR="${MIN_LR:-2e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export POLICY_HEAD_LR_SCALE="${POLICY_HEAD_LR_SCALE:-1.0}"
export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.001}"
export PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-constant}"

# This is the production-style semantic cotrain boundary, so keep the base
# script's FSDP default unless explicitly overridden.
export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"

exec bash "${SCRIPT_DIR}/run_a7_stepindexed_fixedwindow_from7000_30k_20260529.sh"
