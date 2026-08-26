#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SUITE_ROOT" >&2
  exit 2
fi

SUITE_ROOT=$1
[[ "$SUITE_ROOT" == /mnt/* && ! -e "$SUITE_ROOT" && ! -L "$SUITE_ROOT" ]] || {
  echo "acceptance suite root must be one absent persistent /mnt path" >&2
  exit 1
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
OUTPUT=${PICF_ADR150_FULL_MODAL_ACTION_ADOPTION:-/mnt/picf-next/adr150/acceptance/full_modal_action_adoption.json}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
  echo "full-modal adoption output already exists" >&2
  exit 1
}

mkdir -p "$SUITE_ROOT"
"$SCRIPT_DIR/run_full_modal_acceptance_4gpu.sh" \
  action-interventions "$SUITE_ROOT/action_interventions"
"$SCRIPT_DIR/run_full_modal_acceptance_4gpu.sh" action-presence "$SUITE_ROOT/action_presence"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src:$REPO" "$PYTHON" \
  "$REPO/tools/compose_adr150_action_adoption_core.py" \
  --presence "$SUITE_ROOT/action_presence/full_modal_action_adoption_presence.json" \
  --interventions \
    "$SUITE_ROOT/action_interventions/full_modal_action_adoption_interventions.json" \
  --output "$SUITE_ROOT/full_modal_action_adoption_core.json"

"$SCRIPT_DIR/run_full_modal_acceptance_4gpu.sh" dcp-uninterrupted "$SUITE_ROOT/dcp"
"$SCRIPT_DIR/run_full_modal_acceptance_4gpu.sh" dcp-restored "$SUITE_ROOT/dcp"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src:$REPO" "$PYTHON" \
  "$REPO/tools/compose_adr150_full_modal_action_adoption.py" \
  --core "$SUITE_ROOT/full_modal_action_adoption_core.json" \
  --dcp-uninterrupted "$SUITE_ROOT/dcp/acceptance/dcp_uninterrupted.json" \
  --dcp-restored "$SUITE_ROOT/dcp/acceptance/dcp_restored.json" \
  --output "$OUTPUT"
