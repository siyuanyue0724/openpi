#!/usr/bin/env bash
set -euo pipefail

TRAINING_PREFIX_STEPS=${PICF_ADR209_TRAINING_PREFIX_STEPS:-250}
REPO=${PICF_ADR209_REPO:-/mnt/picf-next/adr209/worktree-local-sync}
LOG_ROOT=${PICF_ADR209_CACHE_LOG_ROOT:-/mnt/picf-next/adr209/cache-build-logs/prefix${TRAINING_PREFIX_STEPS}-v1}
[[ -d "$REPO" && ! -L "$REPO" ]] || {
  echo "ADR-209 repository snapshot is absent" >&2
  exit 1
}
[[ ! -e "$LOG_ROOT" && ! -L "$LOG_ROOT" ]] || {
  echo "ADR-209 cache-build log root already exists: $LOG_ROOT" >&2
  exit 2
}

mkdir -p "$LOG_ROOT"
cd "$REPO"

launch() {
  local name=$1
  local gpu=$2
  shift 2
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PICF_ADR209_TRAINING_PREFIX_STEPS="$TRAINING_PREFIX_STEPS" \
    "$@" >"$LOG_ROOT/$name.log" 2>&1 </dev/null &
  printf '%s\n' "$!" >"$LOG_ROOT/$name.pid"
}

launch anytouch 0 ./adr209/build_dense_cache_4gpu_contract.sh anytouch
launch sonata 1 ./adr209/build_dense_cache_4gpu_contract.sh sonata
launch vjepa 2 ./adr209/build_dense_cache_4gpu_contract.sh vjepa
launch flare 3 ./adr209/build_flare_cache_4gpu_contract.sh

for path in "$LOG_ROOT"/*.pid; do
  printf '%s %s\n' "$(basename "$path" .pid)" "$(cat "$path")"
done
