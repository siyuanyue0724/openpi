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

REPO=${PICF_REPO:-/mnt/picf-next/adr177/source-freezes/adr177-task-addressed-full-modal-v1}
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
CONTRACT_ROOT=${PICF_ADR177_CONTRACT_ROOT:-/mnt/picf-next/adr177/contracts/full-modal-2gpu-30k-v1}
STREAM_PLAN=$CONTRACT_ROOT/stream-plan.json
REPRESENTATION_SPLIT=$CONTRACT_ROOT/representation-split.json
EVALUATION_PLAN=$CONTRACT_ROOT/evaluation-plan.json
DENSE_COVERAGE=$CONTRACT_ROOT/dense-evidence-coverage.json
DENSE_CACHE_ROOT=${PICF_ADR177_CACHE_ROOT:-/mnt/picf-next/adr150/caches/calvin-official-30k-v1}
DENSE_SUPPLEMENT_ROOT=${PICF_ADR177_SUPPLEMENT_CACHE_ROOT:-/mnt/picf-next/adr176/caches/full-modal-2gpu-prefix1500-v1}
CURRENT_CACHE_ROOT=${PICF_ADR177_CURRENT_CACHE_ROOT:-/mnt/picf-next/adr176/caches/full-modal-2gpu-prefix1500-v1}
CURRENT_CACHE=$CURRENT_CACHE_ROOT/current-grid
CURRENT_SHARD_ROOT=${PICF_CURRENT_GRID_SHARD_ROOT:-/mnt/picf-next/adr150/caches/calvin-official-30k-v1/current-filter-dino-physical-v1}
CURRENT_REPORT=${CURRENT_CACHE}.build_report.json
ANYTOUCH_CACHE=$DENSE_CACHE_ROOT/anytouch-observed-pose
SONATA_CACHE=$DENSE_CACHE_ROOT/sonata-native
VJEPA_CACHE=$DENSE_CACHE_ROOT/vjepa-final
ANYTOUCH_SUPPLEMENT=$DENSE_SUPPLEMENT_ROOT/anytouch
SONATA_SUPPLEMENT=$DENSE_SUPPLEMENT_ROOT/sonata
VJEPA_SUPPLEMENT=$DENSE_SUPPLEMENT_ROOT/vjepa
STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-30000}
BACKWARD_PREFETCH=${PICF_FSDP2_BACKWARD_PREFETCH:-disabled}
FACTUAL_GRADIENT_STORAGE=${PICF_SEQUENTIAL_FACTUAL_GRADIENT_STORAGE:-cpu}
TRAINABLE_SCOPE=${PICF_TRAINABLE_SCOPE:-frozen-vision-host}
CUDA_ALLOCATOR=${PICF_CUDA_ALLOCATOR:-expandable-segments}

[[ "$STOP_AFTER_STEP" =~ ^[0-9]+$ ]] && (( STOP_AFTER_STEP >= 8 && STOP_AFTER_STEP <= 30000 )) || {
  echo "PICF_STOP_AFTER_STEP must be an integer in [8, 30000]" >&2
  exit 2
}
for path in \
  "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" "$SIDECAR_ROOT" \
  "$CURRENT_CACHE" "$CURRENT_SHARD_ROOT" "$ANYTOUCH_CACHE" "$SONATA_CACHE" \
  "$VJEPA_CACHE" "$ANYTOUCH_SUPPLEMENT" "$SONATA_SUPPLEMENT" \
  "$VJEPA_SUPPLEMENT"
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
  "$SONATA_CACHE/manifest.json" "$VJEPA_CACHE/manifest.json" \
  "$ANYTOUCH_SUPPLEMENT/manifest.json" "$SONATA_SUPPLEMENT/manifest.json" \
  "$VJEPA_SUPPLEMENT/manifest.json"
do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required direct file is absent: $path" >&2
    exit 1
  }
done

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq 2 ]] || {
  echo "ADR-177 acceptance requires exactly two visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "ADR-177 requires two A100 40GB GPUs; observed: $row" >&2
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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ENABLE_TIMING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_TRACE_CPP_STACK=1
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_FR_CPP_STACK=1
cd "$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  tools/run_lingbot_vla2_task_independent_full.py \
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
  --dense-evidence-supplement-cache-root "$ANYTOUCH_SUPPLEMENT" \
  --dense-evidence-supplement-cache-manifest-sha256 "$(file_sha256 "$ANYTOUCH_SUPPLEMENT/manifest.json")" \
  --dense-evidence-supplement-cache-root "$SONATA_SUPPLEMENT" \
  --dense-evidence-supplement-cache-manifest-sha256 "$(file_sha256 "$SONATA_SUPPLEMENT/manifest.json")" \
  --dense-evidence-supplement-cache-root "$VJEPA_SUPPLEMENT" \
  --dense-evidence-supplement-cache-manifest-sha256 "$(file_sha256 "$VJEPA_SUPPLEMENT/manifest.json")" \
  --dense-evidence-coverage-plan "$DENSE_COVERAGE" \
  --dense-evidence-coverage-plan-sha256 "$(file_sha256 "$DENSE_COVERAGE")" \
  --run-dir "$RUN_DIR" \
  --load-global-step 0 \
  --stop-after-step "$STOP_AFTER_STEP" \
  --seed 20260721 \
  --trainable-scope "$TRAINABLE_SCOPE" \
  --capacity 16 \
  --maximum-control-tokens 64 \
  --prior-gradient-control-tokens 8 \
  --posterior-architecture two_pass_v3 \
  --picf-architecture-profile adr177_task_addressed_full_modal_v1 \
  --task-query-count 4 \
  --relation-supervision-layers 8,17,26 \
  --learning-rate 1e-4 \
  --picf-learning-rate-multiplier 2.0 \
  --modality-bridge-learning-rate-multiplier 0.5 \
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
