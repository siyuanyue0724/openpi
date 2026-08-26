#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR=$1
[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" ]] || {
  echo "RUN_DIR must be an absent persistent path beneath /mnt" >&2
  exit 2
}

REPO=${PICF_REPO:-/mnt/picf-next/adr176/source-freezes/adr176-v1-action-eval-1500-20260817T1633+0800}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr176-native}
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json}
SIDECAR_ROOT=${PICF_CALVIN_PHYSICAL_SIDECAR:-/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z}
SIDECAR_MANIFEST=${PICF_CALVIN_PHYSICAL_SIDECAR_MANIFEST:-/mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json}
VISUAL_ACCEPTANCE=${PICF_CALVIN_VISUAL_ACCEPTANCE:-/mnt/picf-next-provenance/calvin-physical-visual-review-identity-a60b7934/calvin-physical-visual-acceptance.json}
CONTRACT_ROOT=${PICF_ADR176_CONTRACT_ROOT:-/mnt/picf-next/adr176/contracts/full-modal-2gpu-30k-prefix1500-v1}
STREAM_PLAN=$CONTRACT_ROOT/stream-plan.json
REPRESENTATION_SPLIT=$CONTRACT_ROOT/representation-split.json
EVALUATION_PLAN=$CONTRACT_ROOT/evaluation-plan.json
DENSE_COVERAGE=$CONTRACT_ROOT/dense-evidence-coverage.json
CACHE_ROOT=${PICF_ADR176_CACHE_ROOT:-/mnt/picf-next/adr176/caches/full-modal-2gpu-prefix1500-v1}
CURRENT_CACHE=$CACHE_ROOT/current-grid
CURRENT_SHARD_ROOT=${PICF_CURRENT_GRID_SHARD_ROOT:-/mnt/picf-next/adr150/caches/calvin-official-30k-v1/current-filter-dino-physical-v1}
CURRENT_REPORT=${CURRENT_CACHE}.build_report.json
ANYTOUCH_CACHE=$CACHE_ROOT/anytouch
SONATA_CACHE=$CACHE_ROOT/sonata
VJEPA_CACHE=$CACHE_ROOT/vjepa
BACKWARD_PREFETCH=${PICF_FSDP2_BACKWARD_PREFETCH:-disabled}
FACTUAL_GRADIENT_STORAGE=${PICF_SEQUENTIAL_FACTUAL_GRADIENT_STORAGE:-cpu}
STALL_DIAGNOSTICS=${PICF_DISTRIBUTED_STALL_DIAGNOSTICS:-enabled}
STALL_TIMEOUT_SECONDS=${PICF_DISTRIBUTED_STALL_TIMEOUT_SECONDS:-90}
TRAINABLE_SCOPE=${PICF_TRAINABLE_SCOPE:-full-host}
CUDA_ALLOCATOR=${PICF_CUDA_ALLOCATOR:-native}
STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-1500}
FORCED_CAUSAL_DIAGNOSTIC_STEP=${PICF_ENGINEERING_FORCE_CAUSAL_DIAGNOSTIC_STEP:-0}

case "$STALL_DIAGNOSTICS" in
  enabled|disabled) ;;
  *)
    echo "PICF_DISTRIBUTED_STALL_DIAGNOSTICS must be enabled or disabled" >&2
    exit 2
    ;;
esac

case "$TRAINABLE_SCOPE" in
  full-host|frozen-vision-host) ;;
  *)
    echo "PICF_TRAINABLE_SCOPE must be full-host or frozen-vision-host" >&2
    exit 2
    ;;
esac

case "$CUDA_ALLOCATOR" in
  native|expandable-segments) ;;
  *)
    echo "PICF_CUDA_ALLOCATOR must be native or expandable-segments" >&2
    exit 2
    ;;
esac

[[ "$STOP_AFTER_STEP" =~ ^[0-9]+$ ]] && (( STOP_AFTER_STEP >= 1 && STOP_AFTER_STEP <= 1500 )) || {
  echo "PICF_STOP_AFTER_STEP must be an integer in [1, 1500]" >&2
  exit 2
}
[[ "$FORCED_CAUSAL_DIAGNOSTIC_STEP" =~ ^[0-9]+$ ]] \
  && (( FORCED_CAUSAL_DIAGNOSTIC_STEP == 0 \
    || (FORCED_CAUSAL_DIAGNOSTIC_STEP >= 3 && FORCED_CAUSAL_DIAGNOSTIC_STEP <= 32) )) || {
  echo "PICF_ENGINEERING_FORCE_CAUSAL_DIAGNOSTIC_STEP must be 0 or an integer in [3, 32]" >&2
  exit 2
}

for path in \
  "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" "$SIDECAR_ROOT" \
  "$CURRENT_CACHE" "$CURRENT_SHARD_ROOT" "$ANYTOUCH_CACHE" "$SONATA_CACHE" \
  "$VJEPA_CACHE"
do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required direct directory is absent: $path" >&2
    exit 1
  }
done
for path in \
  "$PYTHON" "$DATASET_MANIFEST" "$NORM_STATS" "$SIDECAR_MANIFEST" \
  "$VISUAL_ACCEPTANCE" "$STREAM_PLAN" "$REPRESENTATION_SPLIT" \
  "$EVALUATION_PLAN" "$DENSE_COVERAGE" "$CURRENT_CACHE/manifest.json" \
  "$CURRENT_REPORT" "$ANYTOUCH_CACHE/manifest.json" \
  "$SONATA_CACHE/manifest.json" "$VJEPA_CACHE/manifest.json"
do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required direct file is absent: $path" >&2
    exit 1
  }
done

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq 2 ]] || {
  echo "ADR-176 requires exactly two visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "ADR-176 requires two A100 40GB GPUs; observed: $row" >&2
    exit 1
  }
done

file_sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"
cd "$REPO"

TRAINING_ENTRYPOINT=tools/run_lingbot_vla2_task_independent_full.py
if [[ "$STALL_DIAGNOSTICS" == enabled ]]; then
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_DESYNC_DEBUG=1
  export TORCH_NCCL_DUMP_ON_TIMEOUT=1
  export TORCH_NCCL_ENABLE_TIMING=1
  export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
  export TORCH_NCCL_TRACE_CPP_STACK=1
  export TORCH_FR_BUFFER_SIZE=2000
  export TORCH_FR_CPP_STACK=1
  DIAGNOSTIC_ENTRYPOINT=tools/diagnose_distributed_stall.py
  [[ -f "$REPO/$DIAGNOSTIC_ENTRYPOINT" && ! -L "$REPO/$DIAGNOSTIC_ENTRYPOINT" ]] || {
    echo "distributed stall diagnostic entrypoint is absent" >&2
    exit 1
  }
  export PICF_DISTRIBUTED_STALL_TARGET="$REPO/$TRAINING_ENTRYPOINT"
  export PICF_DISTRIBUTED_STALL_TIMEOUT_SECONDS="$STALL_TIMEOUT_SECONDS"
  TRAINING_ENTRYPOINT=$DIAGNOSTIC_ENTRYPOINT
fi

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  "$TRAINING_ENTRYPOINT" \
  --phase fresh \
  --source-checkout "$SOURCE" \
  --checkpoint-dir "$CHECKPOINT" \
  --processor-dir "$PROCESSOR" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --norm-stats "$NORM_STATS" \
  --stream-plan "$STREAM_PLAN" \
  --stream-plan-sha256 "$(file_sha256 "$STREAM_PLAN")" \
  --representation-split "$REPRESENTATION_SPLIT" \
  --representation-split-sha256 "$(file_sha256 "$REPRESENTATION_SPLIT")" \
  --evaluation-plan "$EVALUATION_PLAN" \
  --evaluation-plan-sha256 "$(file_sha256 "$EVALUATION_PLAN")" \
  --physical-sidecar-root "$SIDECAR_ROOT" \
  --physical-sidecar-manifest "$SIDECAR_MANIFEST" \
  --physical-sidecar-manifest-sha256 ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d \
  --physical-visual-acceptance "$VISUAL_ACCEPTANCE" \
  --physical-visual-acceptance-sha256 6443c34b6e8180a8ec090d50ee14dbb2e9d0ad6c4a5e2fc0d9f03a1dbd156552 \
  --current-grid-cache-root "$CURRENT_CACHE" \
  --current-grid-cache-shard-root "$CURRENT_SHARD_ROOT" \
  --current-grid-cache-build-report "$CURRENT_REPORT" \
  --current-grid-cache-build-report-sha256 "$(file_sha256 "$CURRENT_REPORT")" \
  --dense-evidence-mode calvin_full_v1 \
  --dense-token-bridge lingbot_task_token_resampler_v1 \
  --dense-evidence-cache-root "$ANYTOUCH_CACHE" \
  --dense-evidence-cache-manifest-sha256 "$(file_sha256 "$ANYTOUCH_CACHE/manifest.json")" \
  --dense-evidence-cache-root "$SONATA_CACHE" \
  --dense-evidence-cache-manifest-sha256 "$(file_sha256 "$SONATA_CACHE/manifest.json")" \
  --dense-evidence-cache-root "$VJEPA_CACHE" \
  --dense-evidence-cache-manifest-sha256 "$(file_sha256 "$VJEPA_CACHE/manifest.json")" \
  --dense-evidence-coverage-plan "$DENSE_COVERAGE" \
  --dense-evidence-coverage-plan-sha256 "$(file_sha256 "$DENSE_COVERAGE")" \
  --run-dir "$RUN_DIR" \
  --load-global-step 0 \
  --stop-after-step "$STOP_AFTER_STEP" \
  --engineering-force-causal-diagnostic-step "$FORCED_CAUSAL_DIAGNOSTIC_STEP" \
  --seed 20260721 \
  --trainable-scope "$TRAINABLE_SCOPE" \
  --capacity 16 \
  --maximum-control-tokens 64 \
  --prior-gradient-control-tokens 8 \
  --posterior-architecture two_pass_v3 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0 \
  --maximum-peak-reserved-gib 39.0 \
  --maximum-optimizer-lag 8 \
  --causal-ablation-mode none \
  --entity-weight 0.08 \
  --predictive-weight 0.004 \
  --local-bptt-probability 0.0 \
  --overshoot-probability 0.0 \
  --source-mask-probability 0.10 \
  --source-mask-token-fraction 0.0625 \
  --source-prediction-mode omitted_static \
  --omitted-static-rematerialization sequential-backward \
  --fsdp2-placement selective-embedding-offload \
  --fsdp2-backward-prefetch "$BACKWARD_PREFETCH" \
  --sequential-factual-gradient-storage "$FACTUAL_GRADIENT_STORAGE" \
  --cuda-allocator "$CUDA_ALLOCATOR"
