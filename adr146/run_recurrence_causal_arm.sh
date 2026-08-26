#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 current_frame_branch|zero_state|recurrent_state RUN_DIR" >&2
  exit 2
fi

MODE=$1
RUN_DIR=$2
case "$MODE" in
  current_frame_branch)
    PHASE=fresh
    LOAD_GLOBAL_STEP=0
    STOP_AFTER_STEP=200
    ;;
  zero_state|recurrent_state)
    PHASE=resume
    LOAD_GLOBAL_STEP=200
    STOP_AFTER_STEP=300
    ;;
  *)
    echo "unsupported ADR-146 mode: $MODE" >&2
    exit 2
    ;;
esac

REPO=${PICF_REPO:-/mnt/picf-next/worktrees/adr146-recurrence-ablation-20260807}
PYTHON=/opt/picf-runtime-restore-probe-94305690cafb/bin/python
SOURCE=/mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2
PERSISTENT_CHECKPOINT=/mnt/picf-next/models/lingbot-vla-v2-6b
PERSISTENT_PROCESSOR=/mnt/picf-next/models/qwen3-vl-4b-instruct
HOT_MODEL_ROOT=${PICF_HOT_MODEL_ROOT:-/root/picf-hot-adr146}
if [[ -f "$HOT_MODEL_ROOT/READY" ]]; then
  CHECKPOINT=$HOT_MODEL_ROOT/lingbot-vla-v2-6b
  PROCESSOR=$HOT_MODEL_ROOT/qwen3-vl-4b-instruct
else
  CHECKPOINT=$PERSISTENT_CHECKPOINT
  PROCESSOR=$PERSISTENT_PROCESSOR
fi
DATASET=/mnt/calvin_data/task_ABC_D/training
DATASET_MANIFEST=/mnt/picf-next/manifests/calvin-training-files.json
NORM_STATS=/mnt/picf-next/manifests/calvin-lingbot-norm-stats.json
SIDECAR=/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z
SIDECAR_SHA=0198b9d184069f40f1804de411e25ffb3f3a446fcd61d5dd619e944488244ed4
VISUAL_ACCEPTANCE=/mnt/picf-next/runs/calvin-v5-full-tail-audit-43f5c5a-20260725T074858Z/visual_acceptance.json
VISUAL_ACCEPTANCE_SHA=4000dc3394b3027e7cf2a75d54a88b1025314ca503dc6ec2b77f4a63b2163c86
CACHE_ROOT=/mnt/picf-next/adr141/task-independent-full-30k/cache
PREDICTIVE=$CACHE_ROOT/predictive-v4
CURRENT=$CACHE_ROOT/current-grid-v4
PREDICTIVE_REPORT=$PREDICTIVE.build_report.json
CURRENT_REPORT=$CURRENT.build_report.json

for path in \
  "$REPO" \
  "$SOURCE" \
  "$CHECKPOINT" \
  "$PROCESSOR" \
  "$DATASET" \
  "$SIDECAR" \
  "$PREDICTIVE" \
  "$CURRENT"
do
  test -d "$path"
done
for path in \
  "$DATASET_MANIFEST" \
  "$NORM_STATS" \
  "$SIDECAR/manifest.json" \
  "$VISUAL_ACCEPTANCE" \
  "$PREDICTIVE/manifest.json" \
  "$CURRENT/manifest.json" \
  "$PREDICTIVE_REPORT" \
  "$CURRENT_REPORT"
do
  test -f "$path"
done

PREDICTIVE_REPORT_SHA=$(sha256sum "$PREDICTIVE_REPORT" | awk '{print $1}')
CURRENT_REPORT_SHA=$(sha256sum "$CURRENT_REPORT" | awk '{print $1}')

mkdir -p "$RUN_DIR"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  tools/run_lingbot_vla2_task_independent_full.py \
  --phase "$PHASE" \
  --causal-ablation-mode "$MODE" \
  --source-checkout "$SOURCE" \
  --checkpoint-dir "$CHECKPOINT" \
  --processor-dir "$PROCESSOR" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --norm-stats "$NORM_STATS" \
  --physical-sidecar-root "$SIDECAR" \
  --physical-sidecar-manifest "$SIDECAR/manifest.json" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA" \
  --physical-visual-acceptance "$VISUAL_ACCEPTANCE" \
  --physical-visual-acceptance-sha256 "$VISUAL_ACCEPTANCE_SHA" \
  --predictive-cache-root "$PREDICTIVE" \
  --predictive-cache-build-report "$PREDICTIVE_REPORT" \
  --predictive-cache-build-report-sha256 "$PREDICTIVE_REPORT_SHA" \
  --current-grid-cache-root "$CURRENT" \
  --current-grid-cache-build-report "$CURRENT_REPORT" \
  --current-grid-cache-build-report-sha256 "$CURRENT_REPORT_SHA" \
  --run-dir "$RUN_DIR" \
  --load-global-step "$LOAD_GLOBAL_STEP" \
  --stop-after-step "$STOP_AFTER_STEP" \
  --entity-weight 0.08 \
  --predictive-weight 0.0 \
  --local-bptt-probability 0.0 \
  --overshoot-probability 0.0 \
  --source-mask-probability 0.0 \
  --source-prediction-mode omitted_static \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator native
