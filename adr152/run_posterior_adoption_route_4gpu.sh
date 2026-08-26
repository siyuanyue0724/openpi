#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR EXACT_LBOT_RUN_DIR" >&2
  exit 2
fi

RUN_DIR=$1
LBOT_RUN_DIR=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
CONTRACT_ROOT=${PICF_ADR150_CONTRACT_ROOT:-/mnt/picf-next/adr150/contracts/calvin-official-30k-v1}

[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
  echo "ADR-152 route output must be one absent direct path under /mnt" >&2
  exit 1
}
[[ "$LBOT_RUN_DIR" == /mnt/* && -d "$LBOT_RUN_DIR" && ! -L "$LBOT_RUN_DIR" ]] || {
  echo "ADR-152 requires one direct persistent exact-LBOT run directory" >&2
  exit 1
}
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-152 route acceptance requires an exact clean checkout" >&2
  exit 1
}

for step in 000000 000020 000100 000200; do
  [[ -f "$LBOT_RUN_DIR/action_evaluation_step_$step.json" && \
     ! -L "$LBOT_RUN_DIR/action_evaluation_step_$step.json" ]] || {
    echo "ADR-152 exact LBOT is missing action evaluation step $step" >&2
    exit 1
  }
done
[[ -f "$LBOT_RUN_DIR/official_lbot_steps_200.json" && \
   ! -L "$LBOT_RUN_DIR/official_lbot_steps_200.json" ]] || {
  echo "ADR-152 exact LBOT summary is absent" >&2
  exit 1
}

"$PYTHON" - \
  "$LBOT_RUN_DIR" \
  "$CONTRACT_ROOT/four-gpu-30k.physical.stream-plan.json" \
  "$CONTRACT_ROOT/four-gpu-30k.physical.split.json" \
  "$CONTRACT_ROOT/four-gpu-30k.physical.evaluation.json" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
stream_plan = Path(sys.argv[2])
representation_split = Path(sys.argv[3])
evaluation_plan = Path(sys.argv[4])
stream_contract = json.loads(stream_plan.read_text(encoding="utf-8"))
split_contract = json.loads(representation_split.read_text(encoding="utf-8"))
evaluation_contract = json.loads(evaluation_plan.read_text(encoding="utf-8"))
stream_digest = stream_contract.get("plan_sha256")
split_digest = split_contract.get("artifact_sha256")
evaluation_digest = evaluation_contract.get("artifact_sha256")
if not all(isinstance(value, str) and len(value) == 64 for value in (
    stream_digest,
    split_digest,
    evaluation_digest,
)):
    raise SystemExit("ADR-152 contract omitted a canonical artifact digest")
if split_contract.get("stream_plan_sha256") != stream_digest:
    raise SystemExit("ADR-152 representation split is not bound to the stream plan")
if evaluation_contract.get("representation_split_sha256") != split_digest:
    raise SystemExit("ADR-152 evaluation plan is not bound to the representation split")
summary = json.loads((run / "official_lbot_steps_200.json").read_text(encoding="utf-8"))
expected_summary = {
    "status": "PASS",
    "steps": 200,
    "seed": 20260721,
    "world_size": 4,
    "picf_graph_installed": False,
    "task_scorer_present": False,
    "registered_evaluation_steps": [0, 20, 100, 200],
}
for key, expected in expected_summary.items():
    if summary.get(key) != expected:
        raise SystemExit(f"ADR-152 exact LBOT summary mismatch: {key}")
if summary.get("representation_split_sha256") != split_digest:
    raise SystemExit("ADR-152 exact LBOT representation split differs")
if summary.get("evaluation_plan_sha256") != evaluation_digest:
    raise SystemExit("ADR-152 exact LBOT evaluation plan differs")

evaluation_input_sha256 = None
for step in (0, 20, 100, 200):
    snapshot = json.loads(
        (run / f"action_evaluation_step_{step:06d}.json").read_text(encoding="utf-8")
    )
    expected_snapshot = {
        "status": "PASS",
        "checkpoint_global_step": step,
        "picf_graph_installed": False,
        "posterior_present": False,
        "task_scorer_present": False,
        "physical_sidecar_read": False,
        "stream_plan_sha256": stream_digest,
        "representation_split_sha256": split_digest,
        "evaluation_plan_sha256": evaluation_digest,
    }
    for key, expected in expected_snapshot.items():
        if snapshot.get(key) != expected:
            raise SystemExit(f"ADR-152 exact LBOT snapshot mismatch at step {step}: {key}")
    current = snapshot.get("evaluation_input_sha256")
    if not isinstance(current, str) or len(current) != 64:
        raise SystemExit("ADR-152 exact LBOT omitted its evaluation-input digest")
    if evaluation_input_sha256 is None:
        evaluation_input_sha256 = current
    elif current != evaluation_input_sha256:
        raise SystemExit("ADR-152 exact LBOT evaluation inputs changed across checkpoints")

print("ADR-152 exact matched LBOT route gate=PASS")
PY

export PICF_REPO=$REPO
exec "$REPO/adr150/run_full_modal_acceptance_4gpu.sh" posterior-route "$RUN_DIR"
