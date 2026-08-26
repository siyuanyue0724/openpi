#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR MATCHED_LBOT_REPORT" >&2
  exit 2
fi

RUN_DIR=$1
MATCHED_LBOT_REPORT=$2
HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr147/four_gpu_handoff_20260808}
REPO=${PICF_REPO:-/mnt/picf-next/worktrees/adr147-fourgpu-candidate-20260808}
PATCH=$HANDOFF/adr147-four-gpu-candidate.patch
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python}

[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" ]] || {
  echo "four-GPU training requires one absent run directory under /mnt" >&2
  exit 1
}
[[ -f "$PATCH" ]] || {
  echo "four-GPU source patch receipt is absent" >&2
  exit 1
}
[[ "$MATCHED_LBOT_REPORT" == /mnt/* && -f "$MATCHED_LBOT_REPORT" ]] || {
  echo "four-GPU training requires one matched LBOT report under /mnt" >&2
  exit 1
}

"$REPO/adr147/restore_four_gpu_runtime.sh"

"$PYTHON" - "$MATCHED_LBOT_REPORT" "$HANDOFF/four-gpu-30k.stream-plan.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
plan = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
if report.get("schema") not in {
    "picf-next.lingbot-vla2-official-calvin-lbot.v1",
    "picf-next.lingbot-vla2-official-calvin-p0.v1",
}:
    raise SystemExit("matched LBOT report schema differs")
if report.get("status") != "PASS" or report.get("picf_graph_installed") is not False:
    raise SystemExit("matched LBOT report did not pass as an official no-PICF baseline")
if report.get("world_size") != 4 or report.get("steps") != 200:
    raise SystemExit("matched LBOT report is not the required four-rank 200-step control")
if report.get("plan_sha256") != plan.get("plan_sha256"):
    raise SystemExit("matched LBOT report belongs to another frozen stream")
if report.get("seed") != 20260721:
    raise SystemExit("matched LBOT report seed differs")
optimizer = report.get("optimizer_contract")
if not isinstance(optimizer, dict) or float(
    optimizer.get("learning_rate", "nan")
).hex() != float(1e-4).hex():
    raise SystemExit("matched LBOT report learning rate differs")
if float(report.get("max_grad_norm", "nan")).hex() != float(1.0).hex():
    raise SystemExit("matched LBOT report gradient clipping differs")
print("matched LBOT launch gate=PASS")
PY

export PICF_WORLD_SIZE=4
export PICF_EXPECTED_GIT_DIFF_SHA256
PICF_EXPECTED_GIT_DIFF_SHA256=$(sha256sum "$PATCH")
PICF_EXPECTED_GIT_DIFF_SHA256=${PICF_EXPECTED_GIT_DIFF_SHA256%% *}
export PICF_REPO=$REPO
export PICF_PYTHON=$PYTHON
export PICF_LINGBOT_NATIVE_SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147}

exec "$REPO/adr147/run_layerwise_v2.sh" fresh "$RUN_DIR" 30000 0
