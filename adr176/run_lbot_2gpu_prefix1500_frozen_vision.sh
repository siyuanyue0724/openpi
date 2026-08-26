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

REPO=${PICF_REPO:?set PICF_REPO to the immutable ADR-176 baseline source freeze}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr176-native}
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json}
CONTRACT_ROOT=${PICF_ADR176_CONTRACT_ROOT:-/mnt/picf-next/adr176/contracts/full-modal-2gpu-30k-prefix1500-v1}
STREAM_PLAN=$CONTRACT_ROOT/stream-plan.json
REPRESENTATION_SPLIT=$CONTRACT_ROOT/representation-split.json
EVALUATION_PLAN=$CONTRACT_ROOT/evaluation-plan.json

for path in "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required direct directory is absent: $path" >&2
    exit 1
  }
done
for path in \
  "$PYTHON" "$DATASET_MANIFEST" "$NORM_STATS" "$STREAM_PLAN" \
  "$REPRESENTATION_SPLIT" "$EVALUATION_PLAN"
do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required direct file is absent: $path" >&2
    exit 1
  }
done

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq 2 ]] || {
  echo "ADR-176 matched LBOT requires exactly two visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "ADR-176 matched LBOT requires two A100 40GB GPUs; observed: $row" >&2
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
export PYTHONPATH="$REPO/src:$REPO:$SOURCE"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
cd "$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  tools/run_lingbot_vla2_official_lbot.py \
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
  --physical-event-stream \
  --maximum-control-tokens 64 \
  --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/official_lbot_steps_1500.json" \
  --evaluation-steps 0,20,100,200,500,1000,1500 \
  --steps 1500 \
  --seed 20260721 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0 \
  --maximum-peak-reserved-gib 39.0 \
  --trainable-scope frozen-vision-host \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator expandable-segments
