#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 PICF_RUN_DIR LBOT_RUN_DIR [200|2000] [OUTPUT]" >&2
  exit 2
fi

PICF_RUN_DIR=$1
LBOT_RUN_DIR=$2
STOP_STEP=${3:-200}
case "$STOP_STEP" in
  200)
    STEPS=0,20,100,200
    ;;
  2000)
    STEPS=0,20,100,200,500,1000,1500,2000
    ;;
  *)
    echo "ADR-207 comparison supports only registered 200- or 2000-step curves" >&2
    exit 2
    ;;
esac

OUTPUT=${4:-$PICF_RUN_DIR/matched_lbot_curve_step_$STOP_STEP.json}
for run_dir in "$PICF_RUN_DIR" "$LBOT_RUN_DIR"; do
  [[ "$run_dir" == /mnt/* && -d "$run_dir" && ! -L "$run_dir" ]] || {
    echo "comparison run directories must be existing direct paths beneath /mnt: $run_dir" >&2
    exit 2
  }
done
[[ "$OUTPUT" == /mnt/* && ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
  echo "comparison output must be an absent direct path beneath /mnt" >&2
  exit 2
}

REPO=${PICF_ADR207_REPO:-/mnt/picf-next/adr207/source-freezes/native-query-posterior-v18}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
[[ -d "$REPO" && ! -L "$REPO" ]] || {
  echo "ADR-207 immutable source freeze is absent: $REPO" >&2
  exit 1
}
[[ -f "$REPO/source-freeze.receipt.json" && ! -L "$REPO/source-freeze.receipt.json" ]] || {
  echo "ADR-207 immutable source freeze receipt is absent" >&2
  exit 1
}
[[ -f "$PYTHON" && ! -L "$PYTHON" ]] || {
  echo "ADR-207 runtime Python is absent: $PYTHON" >&2
  exit 1
}
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-207 comparison requires the exact clean source freeze" >&2
  exit 1
}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"
cd "$REPO"

exec "$PYTHON" -m tools.compare_adr207_matched_action_curves \
  --picf-run-dir "$PICF_RUN_DIR" \
  --lbot-run-dir "$LBOT_RUN_DIR" \
  --steps "$STEPS" \
  --output "$OUTPUT"
