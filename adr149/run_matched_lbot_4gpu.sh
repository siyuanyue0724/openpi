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
    echo "ADR-149 matched LBOT supports only the registered 200- or 2000-step curves" >&2
    exit 2
    ;;
esac
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147}
HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr149/handoff_20260809}
RUNTIME_HANDOFF=${PICF_RUNTIME_HANDOFF_ROOT:-/mnt/picf-next/adr147/four_gpu_handoff_20260808}

[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" ]] || {
  echo "ADR-149 matched LBOT requires one absent run directory under /mnt" >&2
  exit 1
}
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-149 matched LBOT requires an exact clean checkout" >&2
  exit 1
}
for path in \
  "$HANDOFF/four-gpu-30k.physical.stream-plan.json" \
  "$HANDOFF/four-gpu-30k.physical.split.json" \
  "$HANDOFF/four-gpu-30k.physical.evaluation-plan.json"
do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "ADR-149 matched LBOT input is absent or indirect: $path" >&2
    exit 1
  }
done

PICF_REPO=$REPO PICF_HANDOFF_ROOT=$RUNTIME_HANDOFF \
  "$REPO/adr147/restore_four_gpu_runtime.sh"

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  "$REPO/tools/run_lingbot_vla2_official_lbot.py" \
  --source-checkout "$SOURCE" \
  --checkpoint-dir /mnt/picf-next/models/lingbot-vla-v2-6b \
  --processor-dir /mnt/picf-next/models/qwen3-vl-4b-instruct \
  --dataset-split /mnt/calvin_data/task_ABC_D/training \
  --dataset-manifest /mnt/picf-next/manifests/calvin-training-files.json \
  --norm-stats /mnt/picf-next/manifests/calvin-lingbot-norm-stats.json \
  --run-dir "$RUN_DIR" \
  --output "$RUN_DIR/official_lbot_steps_$STEPS.json" \
  --stream-plan "$HANDOFF/four-gpu-30k.physical.stream-plan.json" \
  --stream-plan-sha256 "$(sha256sum "$HANDOFF/four-gpu-30k.physical.stream-plan.json" | cut -d' ' -f1)" \
  --representation-split "$HANDOFF/four-gpu-30k.physical.split.json" \
  --representation-split-sha256 "$(sha256sum "$HANDOFF/four-gpu-30k.physical.split.json" | cut -d' ' -f1)" \
  --evaluation-plan "$HANDOFF/four-gpu-30k.physical.evaluation-plan.json" \
  --evaluation-plan-sha256 "$(sha256sum "$HANDOFF/four-gpu-30k.physical.evaluation-plan.json" | cut -d' ' -f1)" \
  --physical-event-stream \
  --maximum-control-tokens 64 \
  --evaluation-steps "$EVALUATION_STEPS" \
  --steps "$STEPS" \
  --seed 20260721 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0 \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator native
