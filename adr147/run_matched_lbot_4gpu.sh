#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR=$1
HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr147/four_gpu_handoff_20260808}
REPO=${PICF_REPO:-/mnt/picf-next/worktrees/adr147-fourgpu-candidate-20260808}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147}
PATCH=$HANDOFF/adr147-four-gpu-candidate.patch

[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" ]] || {
  echo "matched LBOT requires one absent run directory under /mnt" >&2
  exit 1
}
[[ -f "$PATCH" ]] || {
  echo "matched LBOT source patch receipt is absent" >&2
  exit 1
}
[[ -z "$(git -C "$REPO" ls-files --others --exclude-standard)" ]] || {
  echo "matched LBOT checkout cannot contain untracked files" >&2
  exit 1
}
EXPECTED_DIFF_SHA=$(sha256sum "$PATCH")
EXPECTED_DIFF_SHA=${EXPECTED_DIFF_SHA%% *}
OBSERVED_DIFF_SHA=$(git -C "$REPO" diff HEAD --binary --no-ext-diff | sha256sum)
OBSERVED_DIFF_SHA=${OBSERVED_DIFF_SHA%% *}
[[ "$OBSERVED_DIFF_SHA" == "$EXPECTED_DIFF_SHA" ]] || {
  echo "matched LBOT checkout diff SHA-256 differs" >&2
  exit 1
}

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
  --output "$RUN_DIR/official_lbot_steps_200.json" \
  --stream-plan "$HANDOFF/four-gpu-30k.stream-plan.json" \
  --stream-plan-sha256 "$(sha256sum "$HANDOFF/four-gpu-30k.stream-plan.json" | cut -d' ' -f1)" \
  --representation-split "$HANDOFF/four-gpu-30k.split.json" \
  --representation-split-sha256 "$(sha256sum "$HANDOFF/four-gpu-30k.split.json" | cut -d' ' -f1)" \
  --evaluation-plan "$HANDOFF/four-gpu-30k.evaluation-plan.json" \
  --evaluation-plan-sha256 "$(sha256sum "$HANDOFF/four-gpu-30k.evaluation-plan.json" | cut -d' ' -f1)" \
  --evaluation-steps 0,20,100,200 \
  --steps 200 \
  --seed 20260721 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0 \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator native
