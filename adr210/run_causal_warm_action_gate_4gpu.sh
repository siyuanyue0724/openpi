#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd "$SCRIPT_DIR/.." && pwd -P)
[[ -f "$REPO/source-freeze.receipt.json" && ! -L "$REPO" ]] || {
  echo "ADR-210 must execute from its immutable source freeze" >&2
  exit 1
}

export PICF_ADR207_REPO=$REPO
export PICF_WORLD_SIZE=4
export PICF_ADR207_CONTRACT_ROOT=${PICF_ADR207_CONTRACT_ROOT:-/mnt/picf-next/adr210/contracts/causal-warm-4gpu-30k-v2}
export PICF_ADR207_CACHE_ROOT=${PICF_ADR207_CACHE_ROOT:-/mnt/picf-next/adr207/caches/native-query-posterior-4gpu-30k-v1}
export PICF_ADR178_SUPPLEMENT_CACHE_ROOT=${PICF_ADR178_SUPPLEMENT_CACHE_ROOT:-/mnt/picf-next/adr210/caches/causal-warm-history-v2}
export PICF_ADR210_ENABLE_DENSE_SUPPLEMENT=1
export PICF_STOP_AFTER_STEP=30000
exec "$REPO/adr207/run_native_videomt_query_posterior_4gpu.sh" "$1"
