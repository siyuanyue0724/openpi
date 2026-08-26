#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR LONG_AUTHORIZATION" >&2
  exit 2
fi

RUN_DIR=$1
AUTHORIZATION=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python}
FROZEN_INPUTS=${PICF_ADR148_FROZEN_INPUTS:-$REPO/adr148/frozen_inputs.sha256}

[[ "$RUN_DIR" == /mnt/* && -d "$RUN_DIR" ]] || {
  echo "ADR-148 long training requires its persistent initial run directory" >&2
  exit 1
}
[[ "$AUTHORIZATION" == /mnt/* && -f "$AUTHORIZATION" && ! -L "$AUTHORIZATION" ]] || {
  echo "ADR-148 long training requires one real persistent authorization" >&2
  exit 1
}
[[ -d "$RUN_DIR/checkpoints/global_step_2000" ]] || {
  echo "ADR-148 long training requires the accepted step-2000 checkpoint" >&2
  exit 1
}

PICF_REPO=$REPO "$REPO/adr147/restore_four_gpu_runtime.sh"

"$PYTHON" - "$AUTHORIZATION" "$RUN_DIR" "$FROZEN_INPUTS" <<'PY'
import hashlib
import json
import pathlib
import sys

authorization_path = pathlib.Path(sys.argv[1])
run_dir = pathlib.Path(sys.argv[2]).resolve()
frozen_inputs = pathlib.Path(sys.argv[3])
value = json.loads(authorization_path.read_text(encoding="utf-8"))
required = {
    "schema",
    "status",
    "input_global_step",
    "maximum_global_step",
    "run_dir",
    "checkpoint_report",
    "frozen_inputs_sha256",
    "subject",
    "evidence",
}
if set(value) != required:
    raise SystemExit("ADR-148 long authorization fields differ from schema")
if value["schema"] != "picf-next.adr148-long-authorization/v1" or value["status"] != "PASS":
    raise SystemExit("ADR-148 long authorization did not pass")
if value["input_global_step"] != 2000 or value["maximum_global_step"] != 30000:
    raise SystemExit("ADR-148 long authorization covers another step interval")
if pathlib.Path(value["run_dir"]).resolve() != run_dir:
    raise SystemExit("ADR-148 long authorization targets another run")
frozen_digest = hashlib.sha256(frozen_inputs.read_bytes()).hexdigest()
if value["frozen_inputs_sha256"] != frozen_digest:
    raise SystemExit("ADR-148 long authorization uses another frozen input receipt")

subject = value["subject"]
subject_fields = {
    "execution_contract_sha256",
    "implementation_sha256",
    "model_family_sha256",
}
if not isinstance(subject, dict) or set(subject) != subject_fields:
    raise SystemExit("ADR-148 long authorization subject is malformed")
if any(
    not isinstance(subject[name], str)
    or len(subject[name]) != 64
    or any(character not in "0123456789abcdef" for character in subject[name])
    for name in subject_fields
):
    raise SystemExit("ADR-148 long authorization subject has an invalid digest")

checkpoint = value["checkpoint_report"]
if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}:
    raise SystemExit("ADR-148 checkpoint receipt is malformed")
checkpoint_path = pathlib.Path(checkpoint["path"])
expected_checkpoint = run_dir / "checkpoints/global_step_2000/task_independent_checkpoint.json"
if checkpoint_path.resolve() != expected_checkpoint or checkpoint_path.is_symlink():
    raise SystemExit("ADR-148 authorization references another checkpoint")
checkpoint_payload = checkpoint_path.read_bytes()
if hashlib.sha256(checkpoint_payload).hexdigest() != checkpoint["sha256"]:
    raise SystemExit("ADR-148 accepted checkpoint report changed")
checkpoint_value = json.loads(checkpoint_payload.decode("utf-8"))
if checkpoint_value.get("status") != "PASS" or checkpoint_value.get("global_step") != 2000:
    raise SystemExit("ADR-148 checkpoint report did not pass at step 2000")
if any(checkpoint_value.get(name) != subject[name] for name in subject_fields):
    raise SystemExit("ADR-148 checkpoint provenance differs from authorization")

required_evidence = (
    "cold_resume_equivalence",
    "heldout_action",
    "calvin_rollout",
    "full_curve_comparison",
    "visual_review",
    "causal_interventions",
    "gradient_adoption",
    "filter_interaction",
)
evidence = value["evidence"]
if not isinstance(evidence, list) or tuple(item.get("name") for item in evidence) != required_evidence:
    raise SystemExit("ADR-148 long authorization evidence coverage or order differs")
for item in evidence:
    if not isinstance(item, dict) or set(item) != {"name", "path", "sha256"}:
        raise SystemExit("ADR-148 evidence receipt is malformed")
    path = pathlib.Path(item["path"])
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"ADR-148 evidence is absent: {item['name']}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise SystemExit(f"ADR-148 evidence changed: {item['name']}")
    report = json.loads(payload.decode("utf-8"))
    if report.get("status") != "PASS" or report.get("global_step") != 2000:
        raise SystemExit(f"ADR-148 evidence did not pass at step 2000: {item['name']}")
    if pathlib.Path(report.get("run_dir", "")).resolve() != run_dir:
        raise SystemExit(f"ADR-148 evidence targets another run: {item['name']}")
    if any(report.get(name) != subject[name] for name in subject_fields):
        raise SystemExit(f"ADR-148 evidence provenance differs: {item['name']}")
print("ADR-148 long-run authorization=PASS")
PY

export PICF_REPO=$REPO
export PICF_PYTHON=$PYTHON
exec "$REPO/adr148/run_full_picf.sh" resume "$RUN_DIR" 30000 2000
