#!/usr/bin/env bash
set -euo pipefail

# 6xA100-40GB production follow-through for the E21/E23 action-readout audit.
#
# Mathematical intent:
#   E21 showed that action descent improves when one optimizer update sees a
#   mixed set of task windows. E23 approximated that on 2 GPUs with bucket
#   balancing and accum=1, giving only 2 windows/update because accum>1 OOMed on
#   40GB cards. This launcher uses 6 physical GPUs with accum=1, so every update
#   sees 6 independent bucket-balanced windows without increasing per-rank
#   activation memory. It is the safest available move toward the E21 12-window
#   evidence while preserving the maintained E23 objective.
#
# Do not enable direct PICF action conditioning here. The maintained causal
# result is: native action path + prefix/context readout is valid; direct action
# conditioning has not proved positive under identical-window controls.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${REPO_ROOT:-/root/openpi_6gpu_b79761f}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"

export EXP="${EXP:-picf_6x40g_e23_bucketbalanced_noactioncond_from11000_30k_20260602}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-6}"

export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_e14h_action4_fullopt_from10000_30k_20260601/11000}"
export SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_full_tracklets_clean_20260520}"

export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

export ACCUM_STEPS="${ACCUM_STEPS:-1}"
export CALVIN_BALANCED_BUCKET_SAMPLER="${CALVIN_BALANCED_BUCKET_SAMPLER:-1}"
export WINDOW_ACTIVATION_CHECKPOINTING="${WINDOW_ACTIVATION_CHECKPOINTING:-0}"

export TRAINING_STRATEGY="${TRAINING_STRATEGY:-fsdp_full_shard}"
export OPTIMIZER_SHARDING="${OPTIMIZER_SHARDING:-none}"
export OPTIMIZER_CHECKPOINT_MODE="${OPTIMIZER_CHECKPOINT_MODE:-model-only}"

exec "${SCRIPT_DIR}/run_a7_e23_bucketbalanced_noactioncond_from11000_30k_20260601.sh"
