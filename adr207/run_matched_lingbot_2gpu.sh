#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [200|2000]" >&2
  exit 2
fi

RUN_DIR=$1
STEPS=${2:-200}
WORLD_SIZE=${PICF_WORLD_SIZE:-2}
case "$WORLD_SIZE" in
  2)
    CUDA_VISIBLE_DEVICES_VALUE=0,1
    ;;
  4)
    CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3
    ;;
  *)
    echo "PICF_WORLD_SIZE must be 2 or 4" >&2
    exit 2
    ;;
esac
case "$STEPS" in
  200)
    EVALUATION_STEPS=0,20,100,200
    ;;
  2000)
    EVALUATION_STEPS=0,20,100,200,500,1000,1500,2000
    ;;
  *)
    echo "ADR-207 matched LingBot supports only registered 200- or 2000-step curves" >&2
    exit 2
    ;;
esac

[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
  echo "ADR-207 matched LingBot run directory must be absent beneath /mnt" >&2
  exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=${PICF_ADR207_REPO:-/mnt/picf-next/adr207/source-freezes/native-query-posterior-v18}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr207-native-muon}
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json}
CONTRACT_ROOT=${PICF_ADR207_CONTRACT_ROOT:-/mnt/picf-next/adr207/contracts/native-query-posterior-${WORLD_SIZE}gpu-30k-v1}
STREAM_PLAN=$CONTRACT_ROOT/stream-plan.json
REPRESENTATION_SPLIT=$CONTRACT_ROOT/representation-split.json
EVALUATION_PLAN=$CONTRACT_ROOT/evaluation-plan.json
ATTENTION_IMPLEMENTATION=${PICF_ATTENTION_IMPLEMENTATION:-flex_cached}
RUNTIME_PYTHON_OVERLAY=${PICF_RUNTIME_PYTHON_OVERLAY:-/mnt/picf-next/adr207/python-overlays/videomt-torch280-functorch-v1}
RUNTIME_OVERLAY_RECEIPT=$RUNTIME_PYTHON_OVERLAY/overlay-receipt.json
RUNTIME_OVERLAY_RECEIPT_SHA256=857d364103403df8aafc97674e97e518acf781bc8fc080840ca11c99f25aacd0

case "$ATTENTION_IMPLEMENTATION" in
  eager|flex_cached) ;;
  *)
    echo "unsupported PICF_ATTENTION_IMPLEMENTATION: $ATTENTION_IMPLEMENTATION" >&2
    exit 2
    ;;
esac

for path in "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required direct directory is absent: $path" >&2
    exit 1
  }
done
for path in \
  "$PYTHON" "$REPO/source-freeze.receipt.json" \
  "$DATASET_MANIFEST" "$NORM_STATS" "$STREAM_PLAN" \
  "$REPRESENTATION_SPLIT" "$EVALUATION_PLAN"
do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required direct file is absent: $path" >&2
    exit 1
  }
done
[[ -d "$RUNTIME_PYTHON_OVERLAY" && ! -L "$RUNTIME_PYTHON_OVERLAY" \
  && -f "$RUNTIME_OVERLAY_RECEIPT" && ! -L "$RUNTIME_OVERLAY_RECEIPT" ]] || {
  echo "ADR-207 matched LingBot runtime overlay is absent" >&2
  exit 1
}
[[ "$(sha256sum "$RUNTIME_OVERLAY_RECEIPT" | cut -d' ' -f1)" == "$RUNTIME_OVERLAY_RECEIPT_SHA256" ]] || {
  echo "ADR-207 matched LingBot runtime overlay receipt changed" >&2
  exit 1
}
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RUNTIME_PYTHON_OVERLAY" "$PYTHON" -c \
  'import functorch, torch; assert torch.__version__ == functorch.__version__ == "2.8.0+cu128"'
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-207 matched LingBot requires the exact clean source freeze" >&2
  exit 1
}

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq "$WORLD_SIZE" ]] || {
  echo "ADR-207 matched LingBot requires exactly $WORLD_SIZE visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "ADR-207 matched LingBot requires $WORLD_SIZE A100 40GB GPUs; observed: $row" >&2
    exit 1
  }
done

file_sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO:$SOURCE:$RUNTIME_PYTHON_OVERLAY"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
cd "$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$WORLD_SIZE" \
  tools/run_lingbot_vla2_official_lbot.py \
  --source-checkout "$SOURCE" \
  --runtime-hotfix "$REPO/references/patches/lingbot_vla2_distributed_muon_collective_alignment.patch" \
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
  --physical-event-stream \
  --minimum-future-source-frames 4 \
  --maximum-control-tokens 64 \
  --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/official_lbot_steps_$STEPS.json" \
  --evaluation-steps "$EVALUATION_STEPS" \
  --steps "$STEPS" \
  --seed 20260721 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0 \
  --maximum-peak-reserved-gib 39.0 \
  --attention-implementation "$ATTENTION_IMPLEMENTATION" \
  --lingbot-compile-mode upstream-default \
  --trainable-scope full-host \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator expandable-segments
