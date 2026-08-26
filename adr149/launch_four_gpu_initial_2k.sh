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
HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr149/handoff_20260809}
RUNTIME_HANDOFF=${PICF_RUNTIME_HANDOFF_ROOT:-/mnt/picf-next/adr147/four_gpu_handoff_20260808}
FROZEN_INPUTS=${PICF_ADR149_FROZEN_INPUTS:-$HANDOFF/frozen_inputs.sha256}

[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" ]] || {
  echo "ADR-149 initial training requires one absent run directory under /mnt" >&2
  exit 1
}
[[ "$MATCHED_LBOT_REPORT" == /mnt/* && -f "$MATCHED_LBOT_REPORT" && ! -L "$MATCHED_LBOT_REPORT" ]] || {
  echo "ADR-149 requires the persistent matched four-GPU LBOT report" >&2
  exit 1
}
[[ -f "$FROZEN_INPUTS" && ! -L "$FROZEN_INPUTS" ]] || {
  echo "ADR-149 frozen input receipt is absent" >&2
  exit 1
}

MATCHED_LBOT_SHA=$(sha256sum "$MATCHED_LBOT_REPORT")
MATCHED_LBOT_SHA=${MATCHED_LBOT_SHA%% *}
grep -Fx "$MATCHED_LBOT_SHA  $MATCHED_LBOT_REPORT" "$FROZEN_INPUTS" >/dev/null || {
  echo "ADR-149 matched LBOT is not bound by the frozen input receipt" >&2
  exit 1
}

PICF_REPO=$REPO PICF_HANDOFF_ROOT=$RUNTIME_HANDOFF \
  "$REPO/adr147/restore_four_gpu_runtime.sh"

"$PYTHON" - \
  "$MATCHED_LBOT_REPORT" \
  "$HANDOFF/four-gpu-30k.physical.stream-plan.json" \
  "$HANDOFF/four-gpu-30k.physical.split.json" \
  "$HANDOFF/four-gpu-30k.physical.evaluation-plan.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
plan = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
split = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
evaluation = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
if report.get("schema") not in {
    "picf-next.lingbot-vla2-official-calvin-lbot.v1",
    "picf-next.lingbot-vla2-official-calvin-p0.v1",
}:
    raise SystemExit("matched LBOT report schema differs")
if report.get("status") != "PASS" or report.get("picf_graph_installed") is not False:
    raise SystemExit("matched LBOT is not a passing official no-PICF baseline")
if report.get("world_size") != 4 or report.get("steps") != 200:
    raise SystemExit("matched LBOT is not the required four-rank 200-step control")
if report.get("physical_event_stream") is not True:
    raise SystemExit("matched LBOT does not use ADR-149 physical events")
if report.get("maximum_control_tokens") != 64:
    raise SystemExit("matched LBOT control capacity differs")
if report.get("plan_sha256") != plan.get("plan_sha256"):
    raise SystemExit("matched LBOT belongs to another frozen stream")
if report.get("representation_split_sha256") != split.get("artifact_sha256"):
    raise SystemExit("matched LBOT belongs to another representation split")
if report.get("evaluation_plan_sha256") != evaluation.get("artifact_sha256"):
    raise SystemExit("matched LBOT belongs to another evaluation plan")
if report.get("registered_evaluation_steps") != [0, 20, 100, 200]:
    raise SystemExit("matched LBOT curve checkpoints differ")
if report.get("seed") != 20260721:
    raise SystemExit("matched LBOT seed differs")
optimizer = report.get("optimizer_contract")
if not isinstance(optimizer, dict) or float(
    optimizer.get("learning_rate", "nan")
).hex() != float(1e-4).hex():
    raise SystemExit("matched LBOT learning rate differs")
if float(report.get("max_grad_norm", "nan")).hex() != float(1.0).hex():
    raise SystemExit("matched LBOT gradient clipping differs")
print("ADR-149 matched LBOT launch gate=PASS")
PY

export PICF_REPO=$REPO
export PICF_PYTHON=$PYTHON
exec "$REPO/adr149/run_full_picf.sh" fresh "$RUN_DIR" 2000 0
