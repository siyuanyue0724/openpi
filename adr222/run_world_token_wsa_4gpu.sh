#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [LOAD_GLOBAL_STEP]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
export PICF_ADR222_REPO=${PICF_ADR222_REPO:-$REPO}
export PICF_WORLD_SIZE=4
export PICF_ADR221_CONTRACT_ROOT=${PICF_ADR221_CONTRACT_ROOT:-/mnt/picf-next/adr207/contracts/native-query-posterior-4gpu-30k-v1}
export PICF_ADR221_CACHE_ROOT=${PICF_ADR221_CACHE_ROOT:-/mnt/picf-next/adr207/caches/native-query-posterior-4gpu-30k-v1}
exec "$SCRIPT_DIR/run_world_token_wsa_2gpu.sh" "$@"
