#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 fresh|resume RUN_DIR STOP_AFTER_STEP LOAD_GLOBAL_STEP" >&2
  exit 2
fi

PHASE=$1
RUN_DIR=$2
STOP_AFTER_STEP=$3
LOAD_GLOBAL_STEP=$4

for value in "$STOP_AFTER_STEP" "$LOAD_GLOBAL_STEP"; do
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "training steps must be canonical non-negative decimal integers" >&2
    exit 2
  }
done
[[ "$STOP_AFTER_STEP" -gt "$LOAD_GLOBAL_STEP" && "$STOP_AFTER_STEP" -le 30000 ]] || {
  echo "stop step must be greater than load step and no greater than 30000" >&2
  exit 2
}

case "$PHASE" in
  fresh)
    [[ "$LOAD_GLOBAL_STEP" -eq 0 ]] || {
      echo "fresh layerwise_v2 must load global step zero" >&2
      exit 2
    }
    ;;
  resume)
    [[ "$LOAD_GLOBAL_STEP" -gt 0 && $((LOAD_GLOBAL_STEP % 2000)) -eq 0 ]] || {
      echo "layerwise_v2 resume requires a positive 2000-step boundary" >&2
      exit 2
    }
    ;;
  *)
    echo "phase must be fresh or resume" >&2
    exit 2
    ;;
esac

WORLD_SIZE=${PICF_WORLD_SIZE:-2}
case "$WORLD_SIZE" in
  2) GPU_LIST=0,1 ;;
  4)
    GPU_LIST=0,1,2,3
    HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr147/four_gpu_handoff_20260808}
    STREAM_PLAN=$HANDOFF/four-gpu-30k.stream-plan.json
    REPRESENTATION_SPLIT=$HANDOFF/four-gpu-30k.split.json
    for path in "$STREAM_PLAN" "$REPRESENTATION_SPLIT"; do
      [[ -f "$path" ]] || {
        echo "four-rank frozen data contract is absent: $path" >&2
        exit 1
      }
    done
    FROZEN_STREAM_ARGS=(
      --stream-plan "$STREAM_PLAN"
      --stream-plan-sha256 "$(sha256sum "$STREAM_PLAN" | cut -d' ' -f1)"
      --representation-split "$REPRESENTATION_SPLIT"
      --representation-split-sha256 "$(sha256sum "$REPRESENTATION_SPLIT" | cut -d' ' -f1)"
    )
    ;;
  *)
    echo "PICF_WORLD_SIZE must be exactly 2 or 4" >&2
    exit 2
    ;;
esac
if [[ "$WORLD_SIZE" -eq 2 ]]; then
  FROZEN_STREAM_ARGS=()
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147}
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next/manifests/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next/manifests/calvin-lingbot-norm-stats.json}
SIDECAR=${PICF_CALVIN_PHYSICAL_SIDECAR:-/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z}
SIDECAR_SHA=${PICF_CALVIN_PHYSICAL_SIDECAR_SHA256:-0198b9d184069f40f1804de411e25ffb3f3a446fcd61d5dd619e944488244ed4}
VISUAL_ACCEPTANCE=${PICF_CALVIN_VISUAL_ACCEPTANCE:-/mnt/picf-next/runs/calvin-v5-full-tail-audit-43f5c5a-20260725T074858Z/visual_acceptance.json}
VISUAL_ACCEPTANCE_SHA=${PICF_CALVIN_VISUAL_ACCEPTANCE_SHA256:-4000dc3394b3027e7cf2a75d54a88b1025314ca503dc6ec2b77f4a63b2163c86}

for path in "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" "$SIDECAR"; do
  [[ -d "$path" ]] || {
    echo "required directory is absent: $path" >&2
    exit 1
  }
done

for path in \
  "$PYTHON" \
  "$DATASET_MANIFEST" \
  "$NORM_STATS" \
  "$SIDECAR/manifest.json" \
  "$VISUAL_ACCEPTANCE"
do
  [[ -f "$path" ]] || {
    echo "required file is absent: $path" >&2
    exit 1
  }
done

EXPECTED_DIFF_SHA=${PICF_EXPECTED_GIT_DIFF_SHA256:-}
if [[ -z "$EXPECTED_DIFF_SHA" ]]; then
  [[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "cloud execution requires an exact clean PICF checkout" >&2
    exit 1
  }
else
  [[ "$EXPECTED_DIFF_SHA" =~ ^[0-9a-f]{64}$ ]] || {
    echo "PICF_EXPECTED_GIT_DIFF_SHA256 must be one lowercase SHA-256" >&2
    exit 2
  }
  [[ -z "$(git -C "$REPO" ls-files --others --exclude-standard)" ]] || {
    echo "a hash-locked execution checkout cannot contain untracked files" >&2
    exit 1
  }
  OBSERVED_DIFF_SHA=$(git -C "$REPO" diff HEAD --binary --no-ext-diff | sha256sum)
  OBSERVED_DIFF_SHA=${OBSERVED_DIFF_SHA%% *}
  [[ "$OBSERVED_DIFF_SHA" == "$EXPECTED_DIFF_SHA" ]] || {
    echo "execution checkout diff SHA-256 differs" >&2
    exit 1
  }
fi

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq "$WORLD_SIZE" ]] || {
  echo "ADR-147 requires exactly $WORLD_SIZE visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "ADR-147 requires $WORLD_SIZE A100 40GB GPUs; observed: $row" >&2
    exit 1
  }
done

mkdir -p "$RUN_DIR"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU_LIST
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$WORLD_SIZE" \
  tools/run_lingbot_vla2_task_independent_full.py \
  --phase "$PHASE" \
  --source-checkout "$SOURCE" \
  --checkpoint-dir "$CHECKPOINT" \
  --processor-dir "$PROCESSOR" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --norm-stats "$NORM_STATS" \
  "${FROZEN_STREAM_ARGS[@]}" \
  --physical-sidecar-root "$SIDECAR" \
  --physical-sidecar-manifest "$SIDECAR/manifest.json" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA" \
  --physical-visual-acceptance "$VISUAL_ACCEPTANCE" \
  --physical-visual-acceptance-sha256 "$VISUAL_ACCEPTANCE_SHA" \
  --run-dir "$RUN_DIR" \
  --load-global-step "$LOAD_GLOBAL_STEP" \
  --stop-after-step "$STOP_AFTER_STEP" \
  --posterior-architecture layerwise_v2 \
  --causal-ablation-mode none \
  --entity-weight 0.08 \
  --predictive-weight 0.0 \
  --local-bptt-probability 0.0 \
  --overshoot-probability 0.0 \
  --source-mask-probability 0.0 \
  --source-prediction-mode omitted_static \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator native
