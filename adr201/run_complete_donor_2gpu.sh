#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR=$1
REPO=${PICF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
PYTHON_OVERLAY=${PICF_RUNTIME_PYTHON_OVERLAY:-/mnt/picf-next/adr199/python-overlays/videomt-runtime-c81d600f-v1}
CHECKPOINT=${PICF_VIDEOMT_CHECKPOINT:-/mnt/picf-next/adr199/assets/videomt/yt_2019_dinov3_68.9.pth}
DINOV3_BUNDLE=${PICF_VIDEOMT_DINOV3_BUNDLE:-/mnt/picf-next/adr199/assets/dinov3-vitl16-from-videomt}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
SIDECAR=${PICF_CALVIN_PHYSICAL_SIDECAR:-/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z}
SIDECAR_MANIFEST=${PICF_CALVIN_PHYSICAL_SIDECAR_MANIFEST:-/mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json}
SIDECAR_MANIFEST_SHA256=${PICF_CALVIN_PHYSICAL_SIDECAR_MANIFEST_SHA256:-ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d}
STEPS=${PICF_STOP_AFTER_STEP:-250}
BUDGET_STEPS=${PICF_BUDGET_STEPS:-30000}
EVAL_STEPS=${PICF_EVAL_STEPS:-0,50,100,250}
EVAL_CLIPS=${PICF_EVAL_CLIPS:-4}
CHECKPOINT_EVERY=${PICF_CHECKPOINT_EVERY:-250}
SAVE_FINAL_CHECKPOINT=${PICF_SAVE_FINAL_CHECKPOINT:-1}
RESUME_CHECKPOINT=${PICF_RESUME_CHECKPOINT:-}

[[ "$RUN_DIR" == /mnt/* ]] || {
  echo "RUN_DIR must be persistent beneath /mnt" >&2
  exit 2
}
if [[ -z "$RESUME_CHECKPOINT" && -e "$RUN_DIR" ]]; then
  echo "fresh RUN_DIR must be absent: $RUN_DIR" >&2
  exit 2
fi
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "resume checkpoint is absent: $RESUME_CHECKPOINT" >&2
  exit 2
fi
for path in "$REPO" "$PYTHON_OVERLAY" "$DINOV3_BUNDLE" "$DATASET" "$SIDECAR"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required directory is absent or a symlink: $path" >&2
    exit 1
  }
done
for path in "$PYTHON" "$CHECKPOINT" "$DATASET_MANIFEST" "$SIDECAR_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required file is absent or a symlink: $path" >&2
    exit 1
  }
done
[[ "$STEPS" =~ ^[0-9]+$ ]] && (( STEPS >= 1 && STEPS <= BUDGET_STEPS )) || {
  echo "PICF_STOP_AFTER_STEP must be in [1, PICF_BUDGET_STEPS]" >&2
  exit 2
}
[[ "$SAVE_FINAL_CHECKPOINT" == 0 || "$SAVE_FINAL_CHECKPOINT" == 1 ]] || {
  echo "PICF_SAVE_FINAL_CHECKPOINT must be 0 or 1" >&2
  exit 2
}

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq 2 ]] || {
  echo "ADR-201 requires exactly two visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "ADR-201 requires two A100 40GB GPUs; observed: $row" >&2
    exit 1
  }
done

SAVE_ARGS=(--save-final-checkpoint)
if [[ "$SAVE_FINAL_CHECKPOINT" == 0 ]]; then
  SAVE_ARGS=(--no-save-final-checkpoint)
fi
RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO:$PYTHON_OVERLAY"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ENABLE_TIMING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_TRACE_CPP_STACK=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cd "$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  tools/train_videomt_calvin_complete_distributed.py \
  --checkpoint "$CHECKPOINT" \
  --dinov3-bundle "$DINOV3_BUNDLE" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --physical-sidecar-root "$SIDECAR" \
  --physical-sidecar-manifest "$SIDECAR_MANIFEST" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_MANIFEST_SHA256" \
  --output-dir "$RUN_DIR" \
  --steps "$STEPS" \
  --budget-steps "$BUDGET_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --eval-clips "$EVAL_CLIPS" \
  --accumulation-steps 4 \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  "${SAVE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
