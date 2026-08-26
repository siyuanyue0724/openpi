#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/picf-next/adr141/task-independent-full-30k
BUILD_PID_FILE="$ROOT/logs/v4-cache-launch.pid"
PREDICTIVE="$ROOT/cache/predictive-v4"
CURRENT="$ROOT/cache/current-grid-v4"
RUNS=/mnt/picf-next/adr141/runs

test -f "$BUILD_PID_FILE"
build_pid=$(<"$BUILD_PID_FILE")
case "$build_pid" in
  ''|*[!0-9]*) echo "invalid cache-build PID" >&2; exit 2 ;;
esac

while kill -0 "$build_pid" 2>/dev/null; do
  sleep 30
done

for path in \
  "$PREDICTIVE/manifest.json" \
  "$PREDICTIVE.build_report.json" \
  "$CURRENT/manifest.json" \
  "$CURRENT.build_report.json"
do
  test -s "$path"
done

/opt/picf-runtime-restore-probe-94305690cafb/bin/python - \
  "$PREDICTIVE/manifest.json" \
  "$CURRENT/manifest.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

predictive = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
current = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if predictive.get("complete") is not True:
    raise SystemExit("predictive cache is not complete")
for name, payload in (("predictive", predictive), ("current", current)):
    if not payload.get("shards") or not isinstance(payload.get("contract"), dict):
        raise SystemExit(f"{name} cache manifest is incomplete")
    temporal_digest = payload["contract"].get("temporal_estimator_sha256")
    if not isinstance(temporal_digest, str) or len(temporal_digest) != 64:
        raise SystemExit(f"{name} cache temporal contract is invalid")
PY

mkdir -p "$RUNS" "$ROOT/logs"
stamp=$(date +%Y%m%dT%H%M%S%z)
run_dir="$RUNS/task-independent-full-30k-v4-$stamp"
log="$ROOT/logs/task-independent-full-30k-v4-$stamp.log"
nohup bash /root/picf-adr141/adr141/run_task_independent_full.sh \
  fresh "$run_dir" 30000 0 0.004 0.004 >"$log" 2>&1 < /dev/null &
training_pid=$!

printf '%s\n' "$training_pid" >"$ROOT/logs/active-v4-training.pid"
printf '%s\n' "$run_dir" >"$ROOT/logs/active-v4-run-dir.txt"
printf '%s\n' "$log" >"$ROOT/logs/active-v4-log.txt"
printf 'pid=%s run_dir=%s log=%s\n' "$training_pid" "$run_dir" "$log"
