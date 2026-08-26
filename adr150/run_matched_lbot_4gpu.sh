#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [200|2000]" >&2
  exit 2
fi

RUN_DIR=$1
STEPS=${2:-200}
case "$STEPS" in
  200)
    EVALUATION_STEPS=0,20,100,200
    ;;
  2000)
    EVALUATION_STEPS=0,20,100,200,500,1000,1500,2000
    ;;
  *)
    echo "ADR-150 matched LBOT supports only the registered 200- or 2000-step curves" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147}
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json}
CONTRACT_ROOT=${PICF_ADR150_CONTRACT_ROOT:-/mnt/picf-next/adr150/contracts/calvin-official-30k-v1}
RUNTIME_HANDOFF=${PICF_RUNTIME_HANDOFF_ROOT:-/mnt/picf-next/adr147/four_gpu_handoff_20260808}
STREAM_PLAN=$CONTRACT_ROOT/four-gpu-30k.physical.stream-plan.json
REPRESENTATION_SPLIT=$CONTRACT_ROOT/four-gpu-30k.physical.split.json
EVALUATION_PLAN=$CONTRACT_ROOT/four-gpu-30k.physical.evaluation.json
FULL_MODAL_ACTION_ADOPTION=${PICF_ADR150_FULL_MODAL_ACTION_ADOPTION:-/mnt/picf-next/adr150/acceptance/full-cache-r18-dcp-boundary-f8b5304/full_modal_action_adoption.json}

[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" ]] || {
  echo "ADR-150 matched LBOT requires one absent run directory under /mnt" >&2
  exit 1
}
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-150 matched LBOT requires an exact clean checkout" >&2
  exit 1
}
for path in \
  "$PYTHON" "$DATASET_MANIFEST" "$NORM_STATS" \
  "$STREAM_PLAN" "$REPRESENTATION_SPLIT" "$EVALUATION_PLAN" \
  "$FULL_MODAL_ACTION_ADOPTION"
do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "ADR-150 matched LBOT input is absent or indirect: $path" >&2
    exit 1
  }
done

PICF_REPO=$REPO PICF_HANDOFF_ROOT=$RUNTIME_HANDOFF \
  "$REPO/adr147/restore_four_gpu_runtime.sh"

file_sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  "$REPO/tools/run_lingbot_vla2_official_lbot.py" \
  --source-checkout "$SOURCE" \
  --checkpoint-dir "$CHECKPOINT" \
  --processor-dir "$PROCESSOR" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --norm-stats "$NORM_STATS" \
  --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/official_lbot_steps_$STEPS.json" \
  --stream-plan "$STREAM_PLAN" \
  --stream-plan-sha256 "$(file_sha256 "$STREAM_PLAN")" \
  --representation-split "$REPRESENTATION_SPLIT" \
  --representation-split-sha256 "$(file_sha256 "$REPRESENTATION_SPLIT")" \
  --evaluation-plan "$EVALUATION_PLAN" \
  --evaluation-plan-sha256 "$(file_sha256 "$EVALUATION_PLAN")" \
  --full-modal-action-adoption "$FULL_MODAL_ACTION_ADOPTION" \
  --physical-event-stream \
  --maximum-control-tokens 64 \
  --evaluation-steps "$EVALUATION_STEPS" \
  --steps "$STEPS" \
  --seed 20260721 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0 \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator native
