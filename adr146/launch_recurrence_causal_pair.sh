#!/usr/bin/env bash
set -euo pipefail

REPO=${PICF_REPO:-/mnt/picf-next/worktrees/adr146-recurrence-ablation-20260807}
ROOT=/mnt/picf-next/adr146/recurrence-only-causal
RUNS=$ROOT/runs
LOGS=$ROOT/logs

if pgrep -f '[r]un_lingbot_vla2_task_independent_full.py' >/dev/null; then
  echo "refusing to launch ADR-146 while another full trainer is active" >&2
  exit 1
fi

mkdir -p "$RUNS" "$LOGS"
stamp=$(date +%Y%m%dT%H%M%S%z)
branch="$RUNS/branch-$stamp"
zero="$RUNS/zero-state-$stamp"
recurrent="$RUNS/recurrent-state-$stamp"
branch_log="$LOGS/branch-$stamp.log"
zero_log="$LOGS/zero-state-$stamp.log"
recurrent_log="$LOGS/recurrent-state-$stamp.log"
status="$ROOT/status-$stamp.jsonl"

record_status() {
  local phase=$1
  local state=$2
  printf '{"phase":"%s","state":"%s","timestamp":"%s"}\n' \
    "$phase" "$state" "$(date --iso-8601=seconds)" >>"$status"
}

clone_branch() {
  local output=$1
  test -d "$branch/checkpoints/global_step_200"
  test ! -e "$output"
  cp -al "$branch" "$output"
  rm -f "$output/.picf-single-writer.lock" "$output/causal_arm_manifest.json"
}

record_status branch RUNNING
PICF_REPO="$REPO" bash "$REPO/adr146/run_recurrence_causal_arm.sh" \
  current_frame_branch "$branch" >"$branch_log" 2>&1
record_status branch COMPLETE

clone_branch "$zero"
clone_branch "$recurrent"

record_status zero_state RUNNING
PICF_REPO="$REPO" bash "$REPO/adr146/run_recurrence_causal_arm.sh" \
  zero_state "$zero" >"$zero_log" 2>&1
record_status zero_state COMPLETE

record_status recurrent_state RUNNING
PICF_REPO="$REPO" bash "$REPO/adr146/run_recurrence_causal_arm.sh" \
  recurrent_state "$recurrent" >"$recurrent_log" 2>&1
record_status recurrent_state COMPLETE

printf '%s\n' "$branch" >"$ROOT/LATEST_BRANCH_RUN"
printf '%s\n' "$zero" >"$ROOT/LATEST_ZERO_RUN"
printf '%s\n' "$recurrent" >"$ROOT/LATEST_RECURRENT_RUN"
printf 'branch=%s zero=%s recurrent=%s\n' "$branch" "$zero" "$recurrent"
