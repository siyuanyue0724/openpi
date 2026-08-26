#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${PICF_WORKTREE:-/mnt/picf-next/worktrees/adr132-j0-20260804}
STAMP=${STAMP:-$(date +%Y%m%dT%H%M%S%z)}
ROOT=${ROOT:-/mnt/picf-next/runs/adr132-k8-current-estimator-cache-$STAMP}
PRED=${PRED:-/mnt/picf-next/cache/adr132-k8-current-estimator-predictive-$STAMP}
GRID=${GRID:-/mnt/picf-next/cache/adr132-k8-current-estimator-current-grid-$STAMP}
STATE=$ROOT/state
BUNDLE=$ROOT/bundle.env
BUNDLE_TMP=$ROOT/.bundle.env.tmp

PYTHON=/opt/picf-miniconda3/envs/picf-lingbot-vla2/bin/python
SOURCE=/mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2
TRAINING_CONFIG=$SOURCE/configs/vla/robotwin/robotwin.yaml
CHECKPOINT=/root/picf-runtime-42a7ad9/models/lingbot-vla-v2-6b
DATASET=/mnt/calvin_data/task_ABC_D/training
DATASET_MANIFEST=/mnt/picf-next/manifests/calvin-training-files.json
SIDECAR=/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z
SIDECAR_SHA256=0198b9d184069f40f1804de411e25ffb3f3a446fcd61d5dd619e944488244ed4
VISUAL_ACCEPTANCE=/mnt/picf-next/runs/calvin-v5-full-tail-audit-43f5c5a-20260725T074858Z/visual_acceptance.json
VISUAL_ACCEPTANCE_SHA256=4000dc3394b3027e7cf2a75d54a88b1025314ca503dc6ec2b77f4a63b2163c86
SPLIT=/mnt/picf-next/probes/representation-k8-reset-mixture-adr121-e6dbcf6-20260731.split.json
SPLIT_SHA256=9e35c8aab44d3aa6949e188ba6e0f30308b36d4a52718f0d21a6ca673339ccef
STREAM_SHA256=c1df57aaadb2c842ce78b5db0fec0ac85a0fe20b384bf38c72010290984dbee6
TEMPORAL_SHA256=629be944ddee69cc6e0f27b7cf8578dd1a482796fec0c1fa50d6cb122ee13c7e
DATASET_TREE_SHA256=ad9d19ed35c708263f08c5d8376cf6ef80ec3d4e0e198e32611ae3b94971b58d

for path in "$ROOT" "$PRED" "$PRED.build_report.json" "$GRID" "$GRID.build_report.json"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing to reuse K8 temporal bundle artifact: $path" >&2
    exit 1
  fi
done
mkdir -p "$ROOT"
printf 'phase=cache_build\nstamp=%s\npredictive=%s\ncurrent_grid=%s\n' \
  "$STAMP" "$PRED" "$GRID" >"$STATE"

on_error() {
  status=$?
  rm -f "$BUNDLE_TMP"
  printf 'phase=failed\nexit_status=%s\n' "$status" >>"$STATE"
  exit "$status"
}
trap on_error ERR

on_signal() {
  signal=$1
  status=$2
  for pid_name in PRED_PID GRID_PID TARGET_PID TEACHER_PID TEMPORAL_PID; do
    pid=${!pid_name:-}
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  rm -f "$BUNDLE_TMP"
  printf 'phase=interrupted\nsignal=%s\nexit_status=%s\n' "$signal" "$status" >>"$STATE"
  trap - ERR INT TERM
  exit "$status"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

COMMON_ARGS=(
  --source-checkout "$SOURCE"
  --training-config "$TRAINING_CONFIG"
  --checkpoint-dir "$CHECKPOINT"
  --dataset-split "$DATASET"
  --dataset-manifest "$DATASET_MANIFEST"
  --physical-sidecar-root "$SIDECAR"
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA256"
  --physical-visual-acceptance "$VISUAL_ACCEPTANCE"
  --physical-visual-acceptance-sha256 "$VISUAL_ACCEPTANCE_SHA256"
  --comparison-id lingbot-vla2-native-picf-full
  --plan-seed 20260721
  --global-batch-size 2
  --total-steps 200
  --local-bptt-probability 0.10
  --overshoot-probability 0.05
  --source-mask-probability 0.10
  --maximum-optimizer-lag 16
  --lane-interleave-factor 8
  --reset-mixture-numerator 1
  --reset-mixture-denominator 2
  --representation-split "$SPLIT"
  --representation-split-sha256 "$SPLIT_SHA256"
)

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$WORKTREE/src:$WORKTREE" "$PYTHON" \
  "$WORKTREE/tools/build_lingbot_calvin_predictive_cache.py" \
  "${COMMON_ARGS[@]}" --output-root "$PRED" --device cuda:0 \
  --batch-size 16 --progress-every 16 >"$ROOT/predictive.log" 2>&1 &
PRED_PID=$!
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$WORKTREE/src:$WORKTREE" "$PYTHON" \
  "$WORKTREE/tools/build_lingbot_calvin_current_grid_cache.py" \
  "${COMMON_ARGS[@]}" --output-root "$GRID" --device cuda:0 \
  --batch-size 16 --progress-every 32 >"$ROOT/current-grid.log" 2>&1 &
GRID_PID=$!
printf 'predictive_pid=%s\ncurrent_grid_pid=%s\n' "$PRED_PID" "$GRID_PID" >>"$STATE"

PRED_STATUS=0
GRID_STATUS=0
wait "$PRED_PID" || PRED_STATUS=$?
wait "$GRID_PID" || GRID_STATUS=$?
if ((PRED_STATUS != 0 || GRID_STATUS != 0)); then
  printf 'phase=failed\npredictive_status=%s\ncurrent_grid_status=%s\n' \
    "$PRED_STATUS" "$GRID_STATUS" >>"$STATE"
  exit 1
fi

PYTHONPATH="$WORKTREE/src:$WORKTREE" "$PYTHON" - \
  "$PRED.build_report.json" "$GRID.build_report.json" \
  "$STREAM_SHA256" "$TEMPORAL_SHA256" <<'PY'
import json
from pathlib import Path
import sys

for raw_path in sys.argv[1:3]:
    report = json.loads(Path(raw_path).read_text(encoding="ascii"))
    if report["stream_plan_sha256"] != sys.argv[3]:
        raise RuntimeError(f"K8 cache stream changed: {raw_path}")
    if report["temporal_estimator_sha256"] != sys.argv[4]:
        raise RuntimeError(f"K8 temporal estimator changed: {raw_path}")
PY

PRED_MANIFEST_SHA256=$(sha256sum "$PRED/manifest.json" | awk '{print $1}')
GRID_MANIFEST_SHA256=$(sha256sum "$GRID/manifest.json" | awk '{print $1}')
read -r PRED_COVERAGE PRED_ENCODER PRED_QUERY < <(
  "$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); c=m["contract"]; print(c["coverage_sha256"], r["teacher_encoder_digest"], c["query_schema_sha256"])' \
    "$PRED/manifest.json" "$PRED.build_report.json"
)
read -r GRID_COVERAGE GRID_ENCODER < <(
  "$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); print(m["contract"]["coverage_sha256"], r["teacher_encoder_digest"])' \
    "$GRID/manifest.json" "$GRID.build_report.json"
)

printf 'phase=audits\n' >>"$STATE"
PYTHONPATH="$WORKTREE/src:$WORKTREE" "$PYTHON" \
  "$WORKTREE/tools/audit_lingbot_predictive_targets.py" \
  --cache-root "$PRED" \
  --cache-manifest-sha256 "$PRED_MANIFEST_SHA256" \
  --dataset-tree-sha256 "$DATASET_TREE_SHA256" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA256" \
  --encoder-digest "$PRED_ENCODER" \
  --query-schema-sha256 "$PRED_QUERY" \
  --coverage-sha256 "$PRED_COVERAGE" \
  --output "$ROOT/predictive-target-audit.json" >"$ROOT/predictive-target-audit.log" 2>&1 &
TARGET_PID=$!

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$WORKTREE/src:$WORKTREE" "$PYTHON" \
  "$WORKTREE/tools/audit_lingbot_dino_teacher_causality.py" \
  --source-checkout "$SOURCE" \
  --training-config "$TRAINING_CONFIG" \
  --checkpoint-dir "$CHECKPOINT" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --physical-sidecar-root "$SIDECAR" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA256" \
  --current-cache-root "$GRID" \
  --current-cache-manifest-sha256 "$GRID_MANIFEST_SHA256" \
  --current-coverage-sha256 "$GRID_COVERAGE" \
  --current-encoder-digest "$GRID_ENCODER" \
  --predictive-cache-root "$PRED" \
  --predictive-cache-manifest-sha256 "$PRED_MANIFEST_SHA256" \
  --predictive-query-schema-sha256 "$PRED_QUERY" \
  --predictive-coverage-sha256 "$PRED_COVERAGE" \
  --predictive-encoder-digest "$PRED_ENCODER" \
  --maximum-records 16 --batch-size 2 --device cuda:0 \
  --output "$ROOT/teacher-causality-audit.json" >"$ROOT/teacher-causality-audit.log" 2>&1 &
TEACHER_PID=$!

PYTHONPATH="$WORKTREE/src:$WORKTREE" "$PYTHON" \
  "$WORKTREE/tools/audit_lingbot_predictive_temporal_targets.py" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --physical-sidecar-root "$SIDECAR" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA256" \
  --predictive-cache-root "$PRED" \
  --predictive-cache-manifest-sha256 "$PRED_MANIFEST_SHA256" \
  --predictive-query-schema-sha256 "$PRED_QUERY" \
  --predictive-coverage-sha256 "$PRED_COVERAGE" \
  --current-cache-root "$GRID" \
  --current-cache-manifest-sha256 "$GRID_MANIFEST_SHA256" \
  --current-coverage-sha256 "$GRID_COVERAGE" \
  --predictive-encoder-digest "$PRED_ENCODER" \
  --current-encoder-digest "$GRID_ENCODER" \
  --visual-output-dir "$ROOT/visuals" \
  --output "$ROOT/predictive-temporal-audit.json" >"$ROOT/predictive-temporal-audit.log" 2>&1 &
TEMPORAL_PID=$!

TARGET_STATUS=0
TEACHER_STATUS=0
TEMPORAL_STATUS=0
wait "$TARGET_PID" || TARGET_STATUS=$?
wait "$TEACHER_PID" || TEACHER_STATUS=$?
wait "$TEMPORAL_PID" || TEMPORAL_STATUS=$?
if ((TARGET_STATUS != 0 || TEACHER_STATUS != 0 || TEMPORAL_STATUS != 0)); then
  printf 'phase=failed\ntarget_status=%s\nteacher_status=%s\ntemporal_status=%s\n' \
    "$TARGET_STATUS" "$TEACHER_STATUS" "$TEMPORAL_STATUS" >>"$STATE"
  exit 1
fi

PRED_REPORT_SHA256=$(sha256sum "$PRED.build_report.json" | awk '{print $1}')
GRID_REPORT_SHA256=$(sha256sum "$GRID.build_report.json" | awk '{print $1}')
TARGET_AUDIT_SHA256=$(sha256sum "$ROOT/predictive-target-audit.json" | awk '{print $1}')
TEACHER_AUDIT_SHA256=$(sha256sum "$ROOT/teacher-causality-audit.json" | awk '{print $1}')
TEMPORAL_AUDIT_SHA256=$(sha256sum "$ROOT/predictive-temporal-audit.json" | awk '{print $1}')
VISUAL_MANIFEST=$ROOT/visuals/visual_manifest.json
VISUAL_MANIFEST_SHA256=$(sha256sum "$VISUAL_MANIFEST" | awk '{print $1}')
printf '%s\n' \
  "BUNDLE_SCHEMA=picf-next.adr132-k8-temporal-bundle.v1" \
  "ROOT=$ROOT" \
  "PRED=$PRED" \
  "GRID=$GRID" \
  "STREAM_SHA256=$STREAM_SHA256" \
  "TEMPORAL_SHA256=$TEMPORAL_SHA256" \
  "SPLIT=$SPLIT" \
  "SPLIT_SHA256=$SPLIT_SHA256" \
  "PRED_REPORT_SHA256=$PRED_REPORT_SHA256" \
  "GRID_REPORT_SHA256=$GRID_REPORT_SHA256" \
  "TARGET_AUDIT=$ROOT/predictive-target-audit.json" \
  "TARGET_AUDIT_SHA256=$TARGET_AUDIT_SHA256" \
  "TEACHER_AUDIT=$ROOT/teacher-causality-audit.json" \
  "TEACHER_AUDIT_SHA256=$TEACHER_AUDIT_SHA256" \
  "TEMPORAL_AUDIT=$ROOT/predictive-temporal-audit.json" \
  "TEMPORAL_AUDIT_SHA256=$TEMPORAL_AUDIT_SHA256" \
  "VISUAL_MANIFEST=$VISUAL_MANIFEST" \
  "VISUAL_MANIFEST_SHA256=$VISUAL_MANIFEST_SHA256" >"$BUNDLE_TMP"
sync -f "$BUNDLE_TMP"
mv "$BUNDLE_TMP" "$BUNDLE"
sync -f "$ROOT"
BUNDLE_SHA256=$(sha256sum "$BUNDLE" | awk '{print $1}')
printf 'phase=complete\nbundle=%s\nbundle_sha256=%s\n' \
  "$BUNDLE" "$BUNDLE_SHA256" >>"$STATE"
