#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PICF_WORLD_SIZE=4

declare -a PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 "$SCRIPT_DIR/build_dense_cache_2gpu_contract.sh" vjepa &
PIDS+=("$!")
CUDA_VISIBLE_DEVICES=1 "$SCRIPT_DIR/build_dense_cache_2gpu_contract.sh" anytouch &
PIDS+=("$!")
CUDA_VISIBLE_DEVICES=2 "$SCRIPT_DIR/build_dense_cache_2gpu_contract.sh" sonata &
PIDS+=("$!")

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
trap - EXIT
echo "ADR-207 four-GPU-contract all-modal target caches are complete"
