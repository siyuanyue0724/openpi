#!/usr/bin/env bash
set -euo pipefail

REPO=/root/picf-adr141
PYTHON=/opt/picf-runtime-restore-probe-94305690cafb/bin/python
SOURCE=/mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2
TRAINING_CONFIG="$SOURCE/configs/vla/robotwin/robotwin.yaml"
CHECKPOINT=/mnt/picf-next/models/lingbot-vla-v2-6b
DATASET=/mnt/calvin_data/task_ABC_D/training
DATASET_MANIFEST=/mnt/picf-next/manifests/calvin-training-files.json
SIDECAR=/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z
SIDECAR_SHA=0198b9d184069f40f1804de411e25ffb3f3a446fcd61d5dd619e944488244ed4
VISUAL_ACCEPTANCE=/mnt/picf-next/runs/calvin-v5-full-tail-audit-43f5c5a-20260725T074858Z/visual_acceptance.json
VISUAL_ACCEPTANCE_SHA=4000dc3394b3027e7cf2a75d54a88b1025314ca503dc6ec2b77f4a63b2163c86
ROOT=/mnt/picf-next/adr141/task-independent-full-30k
PREDICTIVE="$ROOT/cache/predictive-v4"
CURRENT="$ROOT/cache/current-grid-v4"

mkdir -p "$ROOT/logs" "$ROOT/cache"
test ! -e "$PREDICTIVE"
test ! -e "$CURRENT"

cd "$REPO"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO/src:$REPO" "$PYTHON" \
  tools/build_lingbot_calvin_predictive_cache.py \
  --source-checkout "$SOURCE" \
  --training-config "$TRAINING_CONFIG" \
  --checkpoint-dir "$CHECKPOINT" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --physical-sidecar-root "$SIDECAR" \
  --physical-sidecar-manifest "$SIDECAR/manifest.json" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA" \
  --physical-visual-acceptance "$VISUAL_ACCEPTANCE" \
  --physical-visual-acceptance-sha256 "$VISUAL_ACCEPTANCE_SHA" \
  --output-root "$PREDICTIVE" \
  --comparison-id lingbot-vla2-native-picf-full \
  --plan-seed 20260721 \
  --global-batch-size 2 \
  --total-steps 30000 \
  --local-bptt-probability 0.0 \
  --overshoot-probability 0.05 \
  --source-mask-probability 0.10 \
  --maximum-optimizer-lag 8 \
  --device cuda:0 \
  --progress-every 256 \
  >"$ROOT/logs/predictive-v4-cache.log" 2>&1 &
predictive_pid=$!

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO/src:$REPO" "$PYTHON" \
  tools/build_lingbot_calvin_current_grid_cache.py \
  --source-checkout "$SOURCE" \
  --training-config "$TRAINING_CONFIG" \
  --checkpoint-dir "$CHECKPOINT" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --physical-sidecar-root "$SIDECAR" \
  --physical-sidecar-manifest "$SIDECAR/manifest.json" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA" \
  --physical-visual-acceptance "$VISUAL_ACCEPTANCE" \
  --physical-visual-acceptance-sha256 "$VISUAL_ACCEPTANCE_SHA" \
  --output-root "$CURRENT" \
  --comparison-id lingbot-vla2-native-picf-full \
  --plan-seed 20260721 \
  --global-batch-size 2 \
  --total-steps 30000 \
  --local-bptt-probability 0.0 \
  --overshoot-probability 0.05 \
  --source-mask-probability 0.10 \
  --maximum-optimizer-lag 8 \
  --device cuda:0 \
  --progress-every 256 \
  >"$ROOT/logs/current-grid-v4-cache.log" 2>&1 &
current_pid=$!

printf '%s\n' "$predictive_pid" >"$ROOT/logs/predictive-v4-cache.pid"
printf '%s\n' "$current_pid" >"$ROOT/logs/current-grid-v4-cache.pid"

status=0
wait "$predictive_pid" || status=$?
wait "$current_pid" || status=$?
exit "$status"
