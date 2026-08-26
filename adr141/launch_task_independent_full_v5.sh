#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/picf-next/adr141/task-independent-full-30k
RUNS=/mnt/picf-next/adr141/runs
LOGS=$ROOT/logs

if pgrep -f '[r]un_lingbot_vla2_task_independent_full.py' >/dev/null; then
  echo "refusing to launch v5 while another full trainer is active" >&2
  exit 1
fi

mkdir -p "$RUNS" "$LOGS"
stamp=$(date +%Y%m%dT%H%M%S%z)
run_dir="$RUNS/task-independent-full-30k-v5-$stamp"
log="$LOGS/task-independent-full-30k-v5-$stamp.log"

nohup bash /root/picf-adr141/adr141/run_task_independent_full.sh \
  fresh "$run_dir" 30000 0 0.08 0.08 >"$log" 2>&1 < /dev/null &
training_pid=$!

publish_pointer() {
  local value=$1
  local path=$2
  local temporary="${path}.tmp.$$"
  printf '%s\n' "$value" >"$temporary"
  mv -f "$temporary" "$path"
}

publish_pointer "$training_pid" "$LOGS/active-v5-training.pid"
publish_pointer "$run_dir" "$LOGS/active-v5-run-dir.txt"
publish_pointer "$log" "$LOGS/active-v5-log.txt"
publish_pointer "$run_dir" "$ROOT/ACTIVE_RUN_DIR"
publish_pointer "$log" "$ROOT/ACTIVE_TRAIN_LOG"
printf 'pid=%s run_dir=%s log=%s\n' "$training_pid" "$run_dir" "$log"
