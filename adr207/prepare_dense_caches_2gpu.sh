#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

CUDA_VISIBLE_DEVICES=0 "$SCRIPT_DIR/build_dense_cache_2gpu_contract.sh" vjepa &
VJEPA_PID=$!
cleanup() {
  if kill -0 "$VJEPA_PID" 2>/dev/null; then
    kill "$VJEPA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=1 "$SCRIPT_DIR/build_dense_cache_2gpu_contract.sh" anytouch
CUDA_VISIBLE_DEVICES=1 "$SCRIPT_DIR/build_dense_cache_2gpu_contract.sh" sonata
wait "$VJEPA_PID"
trap - EXIT

echo "ADR-207 all-modal target caches are complete"

