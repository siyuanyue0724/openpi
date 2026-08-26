#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MANUAL_ARMS_ENV" >&2
  exit 2
fi

ENV_FILE=$1
test -f "$ENV_FILE"
# The environment file is generated locally by the ADR-146 operator and only
# contains absolute branch/arm paths without shell metacharacters.
source "$ENV_FILE"
: "${branch:?}"
: "${zero:?}"
: "${recurrent:?}"

REPO=${PICF_REPO:-/mnt/picf-next/worktrees/adr146-recurrence-ablation-20260807}
ROOT=/mnt/picf-next/adr146/recurrence-only-causal
status="$ROOT/continuation-status-$(date +%Y%m%dT%H%M%S%z).jsonl"

if pgrep -f '[r]un_lingbot_vla2_task_independent_full.py' >/dev/null; then
  echo "refusing to launch continuations while another full trainer is active" >&2
  exit 1
fi
test -f "$branch/checkpoints/global_step_200/task_independent_checkpoint.json"
test -f "$zero/STEP200_COPY_READY"

record_status() {
  local phase=$1
  local state=$2
  printf '{"phase":"%s","state":"%s","timestamp":"%s"}\n' \
    "$phase" "$state" "$(date --iso-8601=seconds)" >>"$status"
}

checkpoint_inventory() {
  local root=$1
  (
    cd "$root"
    find . -type f -printf '%P %s\n' | sort | sha256sum | awk '{print $1}'
  )
}

verify_step_200_copy() {
  local output=$1
  local source_digest output_digest
  source_digest=$(checkpoint_inventory "$branch/checkpoints/global_step_200")
  output_digest=$(checkpoint_inventory "$output/checkpoints/global_step_200")
  if [[ "$source_digest" != "$output_digest" ]]; then
    echo "step-200 copy inventory differs for $output" >&2
    return 1
  fi
  printf '%s\n' "$source_digest"
}

clone_branch_metadata() {
  local output=$1
  test ! -e "$output"
  mkdir -p "$output"
  (
    cd "$branch"
    tar --exclude=./checkpoints --exclude=./.picf-single-writer.lock -cf - .
  ) | (
    cd "$output"
    tar -xf -
  )
  mkdir -p "$output/checkpoints"
}

run_arm() {
  local mode=$1
  local output=$2
  local log=$3
  rm -f "$output/.picf-single-writer.lock" "$output/causal_arm_manifest.json"
  record_status "$mode" RUNNING
  if PICF_REPO="$REPO" bash "$REPO/adr146/run_recurrence_causal_arm.sh" \
    "$mode" "$output" >"$log" 2>&1; then
    record_status "$mode" COMPLETE
  else
    record_status "$mode" FAILED
    return 1
  fi
  test -f "$output/checkpoints/global_step_300/task_independent_checkpoint.json"
  rm -rf "$output/checkpoints/global_step_200"
}

stamp=$(date +%Y%m%dT%H%M%S%z)
zero_log="$ROOT/logs/zero-state-$stamp.log"
recurrent_log="$ROOT/logs/recurrent-state-$stamp.log"

zero_inventory=$(verify_step_200_copy "$zero")
printf 'zero_step_200_inventory_sha256=%s\n' "$zero_inventory" >>"$ENV_FILE"
run_arm zero_state "$zero" "$zero_log"

clone_branch_metadata "$recurrent"
cp -a "$branch/checkpoints/global_step_200" "$recurrent/checkpoints/"
recurrent_inventory=$(verify_step_200_copy "$recurrent")
printf 'recurrent_step_200_inventory_sha256=%s\n' "$recurrent_inventory" >>"$ENV_FILE"
run_arm recurrent_state "$recurrent" "$recurrent_log"

printf '%s\n' "$branch" >"$ROOT/LATEST_BRANCH_RUN"
printf '%s\n' "$zero" >"$ROOT/LATEST_ZERO_RUN"
printf '%s\n' "$recurrent" >"$ROOT/LATEST_RECURRENT_RUN"
printf 'branch=%s zero=%s recurrent=%s\n' "$branch" "$zero" "$recurrent"
