#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR MATCHED_LBOT_REPORT" >&2
  exit 2
fi

RUN_DIR=$1
MATCHED_LBOT_REPORT=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
CONTRACT_ROOT=${PICF_ADR150_CONTRACT_ROOT:-/mnt/picf-next/adr150/contracts/calvin-official-30k-v1}
HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr150/handoff_20260810}
RUNTIME_HANDOFF=${PICF_RUNTIME_HANDOFF_ROOT:-/mnt/picf-next/adr147/four_gpu_handoff_20260808}
FROZEN_INPUTS=${PICF_ADR150_FROZEN_INPUTS:-$HANDOFF/frozen_inputs.sha256}

[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" ]] || {
  echo "ADR-150 initial training requires one absent run directory under /mnt" >&2
  exit 1
}
[[ "$MATCHED_LBOT_REPORT" == /mnt/* && -f "$MATCHED_LBOT_REPORT" && ! -L "$MATCHED_LBOT_REPORT" ]] || {
  echo "ADR-150 requires the persistent matched four-GPU LBOT report" >&2
  exit 1
}
[[ -f "$FROZEN_INPUTS" && ! -L "$FROZEN_INPUTS" ]] || {
  echo "ADR-150 frozen input receipt is absent" >&2
  exit 1
}

MATCHED_LBOT_SHA=$(sha256sum "$MATCHED_LBOT_REPORT")
MATCHED_LBOT_SHA=${MATCHED_LBOT_SHA%% *}
grep -Fx "$MATCHED_LBOT_SHA  $MATCHED_LBOT_REPORT" "$FROZEN_INPUTS" >/dev/null || {
  echo "ADR-150 matched LBOT is not bound by the frozen input receipt" >&2
  exit 1
}

PICF_REPO=$REPO PICF_HANDOFF_ROOT=$RUNTIME_HANDOFF \
  "$REPO/adr147/restore_four_gpu_runtime.sh"

"$PYTHON" - "$MATCHED_LBOT_REPORT" "$HANDOFF/frozen_inputs.manifest.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
validation = manifest.get("matched_lbot_validation")
bound = manifest.get("matched_lbot_report")
if not isinstance(validation, dict) or validation.get("status") != "PASS":
    raise SystemExit("ADR-150 typed matched-LBOT validation is absent")
if not isinstance(bound, dict):
    raise SystemExit("ADR-150 frozen matched-LBOT identity is absent")
canonical = json.dumps(
    report,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii")
canonical_sha256 = hashlib.sha256(canonical).hexdigest()
if (
    validation.get("lbot_report_sha256") != canonical_sha256
    or bound.get("canonical_sha256") != canonical_sha256
):
    raise SystemExit("ADR-150 matched-LBOT report differs from its typed frozen validation")
print("ADR-150 typed matched-LBOT launch gate=PASS")
PY

export PICF_REPO=$REPO
export PICF_PYTHON=$PYTHON
exec "$REPO/adr150/run_full_picf.sh" fresh "$RUN_DIR" 2000 0
