#!/usr/bin/env bash
set -euo pipefail

repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PICF_LTOP_MODE=restart-smoke

exec "$repository_root/adr164/run_ltop_core_long_2gpu.sh" "$@"
