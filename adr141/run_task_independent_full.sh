#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 fresh|resume RUN_DIR STOP_AFTER_STEP LOAD_GLOBAL_STEP ENTITY_WEIGHT PREDICTIVE_WEIGHT" >&2
  exit 2
fi

PHASE=$1
RUN_DIR=$2
STOP_AFTER_STEP=$3
LOAD_GLOBAL_STEP=$4
ENTITY_WEIGHT=$5
PREDICTIVE_WEIGHT=$6

case "$PHASE" in
  fresh|resume) ;;
  *) echo "phase must be fresh or resume" >&2; exit 2 ;;
esac

REPO=/root/picf-adr141
PYTHON=/opt/picf-runtime-restore-probe-94305690cafb/bin/python
SOURCE=/mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2
CHECKPOINT=/mnt/picf-next/models/lingbot-vla-v2-6b
PROCESSOR=/mnt/picf-next/models/qwen3-vl-4b-instruct
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
  --entity-weight "$ENTITY_WEIGHT" \
  --predictive-weight "$PREDICTIVE_WEIGHT" \
  --local-bptt-probability 0.0 \
  --source-prediction-mode omitted_static \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator native
