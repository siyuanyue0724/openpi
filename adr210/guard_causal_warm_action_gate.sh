#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_DIR TRAIN_PID LINGBOT_STEP100_JSON" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PYTHON=/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12
exec "$PYTHON" "$SCRIPT_DIR/guard_causal_warm_action_gate_fast.py" "$@"
