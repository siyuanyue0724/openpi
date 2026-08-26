#!/usr/bin/env bash
set -euo pipefail

repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PICF_LTOP_MODE=restart-smoke

exec "$repository_root/adr174/run_fixed_head_ltop_production_2gpu.sh" "$@"
