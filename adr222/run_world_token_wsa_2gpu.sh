#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [LOAD_GLOBAL_STEP]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
export PICF_ADR221_REPO=${PICF_ADR222_REPO:-$REPO}
export PICF_WSA_ARCHITECTURE_PROFILE=adr222_native_videomt_world_token_wsa_v1
exec "$REPO/adr221/run_full_source_wsa_2gpu.sh" "$@"
