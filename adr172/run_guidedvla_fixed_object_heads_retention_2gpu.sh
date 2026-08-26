#!/usr/bin/env bash
set -euo pipefail

export PICF_ADR172_DIRECT_POSTERIOR_HEAD_SCOPE=guidedvla-fixed-object-heads-0-1
exec "$(dirname "${BASH_SOURCE[0]}")/run_direct_posterior_retention_2gpu.sh" "$@"
