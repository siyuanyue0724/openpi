#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PICF_ARCHITECTURE_PROFILE=${PICF_ARCHITECTURE_PROFILE:-adr193_implicit_multimodal_anchor_v1}
export PICF_STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-250}
export PICF_REPO=${PICF_REPO:-/mnt/picf-next/adr193/source-freezes/adr193-implicit-anchor-v2}

exec "$SCRIPT_DIR/../adr178/run_direct_action_posterior_full_modal.sh" "$1"
