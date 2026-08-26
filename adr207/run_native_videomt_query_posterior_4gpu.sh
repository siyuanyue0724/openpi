#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PICF_WORLD_SIZE=4
exec "$SCRIPT_DIR/run_native_videomt_query_posterior_2gpu.sh" "$@"
