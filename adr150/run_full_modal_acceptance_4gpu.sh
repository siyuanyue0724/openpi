#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 action-presence|action-interventions|posterior-route|posterior-adoption-dose|dcp-uninterrupted|dcp-restored RUN_DIR" >&2
  exit 2
fi

MODE=$1
RUN_DIR=$2
SOURCE_MASK_PROBABILITY=0.10
case "$MODE" in
  action-presence)
    ACCEPTANCE_MODE=action-adoption-presence
    PHASE=fresh
    LOAD_GLOBAL_STEP=0
    STOP_AFTER_STEP=1
    [[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
      echo "action-adoption presence requires one absent run directory" >&2
      exit 1
    }
    ;;
  action-interventions)
    ACCEPTANCE_MODE=action-adoption-interventions
    PHASE=fresh
    LOAD_GLOBAL_STEP=0
    STOP_AFTER_STEP=1
    [[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
      echo "action-adoption interventions require one absent run directory" >&2
      exit 1
    }
    ;;
  posterior-route)
    ACCEPTANCE_MODE=posterior-adoption-route
    PHASE=fresh
    LOAD_GLOBAL_STEP=0
    STOP_AFTER_STEP=500
    [[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
      echo "posterior-adoption routing requires one absent run directory" >&2
      exit 1
    }
    ;;
  posterior-adoption-dose)
    ACCEPTANCE_MODE=posterior-adoption-dose
    PHASE=fresh
    LOAD_GLOBAL_STEP=0
    STOP_AFTER_STEP=200
    SOURCE_MASK_PROBABILITY=1.0
    [[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
      echo "posterior-adoption dose requires one absent run directory" >&2
      exit 1
    }
    ;;
  dcp-uninterrupted)
    ACCEPTANCE_MODE=dcp-uninterrupted
    PHASE=fresh
    LOAD_GLOBAL_STEP=0
    STOP_AFTER_STEP=2
    [[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
      echo "uninterrupted DCP acceptance requires one absent run directory" >&2
      exit 1
    }
    ;;
  dcp-restored)
    ACCEPTANCE_MODE=dcp-restored
    PHASE=resume
    LOAD_GLOBAL_STEP=1
    STOP_AFTER_STEP=2
    [[ -d "$RUN_DIR/checkpoints/global_step_1" && \
       -f "$RUN_DIR/acceptance/dcp_uninterrupted.json" ]] || {
      echo "restored DCP acceptance requires the uninterrupted step-1 artifact" >&2
      exit 1
    }
    ;;
  *)
    echo "unknown full-modal acceptance mode: $MODE" >&2
    exit 2
    ;;
esac
[[ "$RUN_DIR" == /mnt/* ]] || {
  echo "full-modal acceptance output must persist under /mnt" >&2
  exit 2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147}
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json}
SIDECAR_ROOT=${PICF_CALVIN_PHYSICAL_SIDECAR:-/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z}
SIDECAR_MANIFEST=${PICF_CALVIN_PHYSICAL_SIDECAR_MANIFEST:-/mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json}
VISUAL_ACCEPTANCE=${PICF_CALVIN_VISUAL_ACCEPTANCE:-/mnt/picf-next-provenance/calvin-physical-visual-review-identity-a60b7934/calvin-physical-visual-acceptance.json}
CONTRACT_ROOT=${PICF_ADR150_CONTRACT_ROOT:-/mnt/picf-next/adr150/contracts/calvin-official-30k-v1}
STREAM_PLAN=$CONTRACT_ROOT/four-gpu-30k.physical.stream-plan.json
REPRESENTATION_SPLIT=$CONTRACT_ROOT/four-gpu-30k.physical.split.json
EVALUATION_PLAN=$CONTRACT_ROOT/four-gpu-30k.physical.evaluation.json
DENSE_COVERAGE=$CONTRACT_ROOT/four-gpu-30k.physical.dense-evidence-coverage.json
CACHE_ROOT=${PICF_ADR150_CACHE_ROOT:-/mnt/picf-next/adr150/caches/calvin-official-30k-v1}
CURRENT_CACHE=${PICF_CURRENT_CACHE_ROOT:-$CACHE_ROOT/current-filter-dino-physical-v1}
CURRENT_REPORT=${PICF_CURRENT_CACHE_BUILD_REPORT:-${CURRENT_CACHE}.build_report.json}
ANYTOUCH_CACHE=${PICF_ANYTOUCH_CACHE_ROOT:-$CACHE_ROOT/anytouch-observed-pose}
SONATA_CACHE=${PICF_SONATA_CACHE_ROOT:-$CACHE_ROOT/sonata-native}
VJEPA_CACHE=${PICF_VJEPA_CACHE_ROOT:-$CACHE_ROOT/vjepa-final}

[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "full-modal acceptance requires an exact clean checkout" >&2
  exit 1
}
for path in \
  "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" "$SIDECAR_ROOT" \
  "$CURRENT_CACHE" "$ANYTOUCH_CACHE" "$SONATA_CACHE" "$VJEPA_CACHE"
do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "full-modal acceptance directory is absent or indirect: $path" >&2
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
    echo "full-modal acceptance file is absent or indirect: $path" >&2
    exit 1
  }
done

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq 4 ]] || {
  echo "full-modal acceptance requires exactly four visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "full-modal acceptance requires four A100 40GB GPUs; observed: $row" >&2
    exit 1
  }
done

file_sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"
cd "$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  tools/run_lingbot_vla2_task_independent_full.py \
  --phase "$PHASE" \
  --acceptance-mode "$ACCEPTANCE_MODE" \
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
  --load-global-step "$LOAD_GLOBAL_STEP" \
  --stop-after-step "$STOP_AFTER_STEP" \
  --seed 20260721 \
  --capacity 16 \
  --maximum-control-tokens 64 \
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
  --source-mask-probability "$SOURCE_MASK_PROBABILITY" \
  --source-mask-token-fraction 0.0625 \
  --source-prediction-mode omitted_static \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator native
