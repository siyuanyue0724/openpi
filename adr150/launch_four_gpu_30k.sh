#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR LONG_AUTHORIZATION" >&2
  exit 2
fi

RUN_DIR=$1
AUTHORIZATION=$2

echo "ADR-150 2k->30k promotion is NO-GO until typed evidence validators are implemented" >&2
echo "The current thin authorization envelope is not scientific evidence and cannot launch training" >&2
exit 1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr150/handoff_20260810}
RUNTIME_HANDOFF=${PICF_RUNTIME_HANDOFF_ROOT:-/mnt/picf-next/adr147/four_gpu_handoff_20260808}
FROZEN_INPUTS=${PICF_ADR150_FROZEN_INPUTS:-$HANDOFF/frozen_inputs.sha256}

[[ "$RUN_DIR" == /mnt/* && -d "$RUN_DIR" ]] || {
  echo "ADR-150 long training requires its persistent initial run directory" >&2
  exit 1
}
[[ "$AUTHORIZATION" == /mnt/* && -f "$AUTHORIZATION" && ! -L "$AUTHORIZATION" ]] || {
  echo "ADR-150 long training requires one real persistent authorization" >&2
  exit 1
}
[[ -d "$RUN_DIR/checkpoints/global_step_2000" ]] || {
  echo "ADR-150 long training requires the accepted step-2000 checkpoint" >&2
  exit 1
}

LOAD_GLOBAL_STEP=2000
for path in "$RUN_DIR"/checkpoints/global_step_*; do
  [[ -d "$path" && ! -L "$path" ]] || continue
  step=${path##*/global_step_}
  [[ "$step" =~ ^[1-9][0-9]*$ ]] || continue
  if (( step % 2000 == 0 && step <= 30000 && step > LOAD_GLOBAL_STEP )); then
    LOAD_GLOBAL_STEP=$step
  fi
done

PICF_REPO=$REPO PICF_HANDOFF_ROOT=$RUNTIME_HANDOFF \
  "$REPO/adr147/restore_four_gpu_runtime.sh"

"$PYTHON" - "$AUTHORIZATION" "$RUN_DIR" "$FROZEN_INPUTS" "$LOAD_GLOBAL_STEP" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys


def write_exclusive_durable(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("ADR-150 durable write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

authorization_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2]).resolve()
frozen_inputs = Path(sys.argv[3])
load_global_step = int(sys.argv[4])
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
    raise SystemExit("ADR-150 long authorization fields differ from schema")
if value["schema"] != "picf-next.adr150-long-authorization/v1" or value["status"] != "PASS":
    raise SystemExit("ADR-150 long authorization did not pass")
if value["input_global_step"] != 2000 or value["maximum_global_step"] != 30000:
    raise SystemExit("ADR-150 long authorization covers another step interval")
if Path(value["run_dir"]).resolve() != run_dir:
    raise SystemExit("ADR-150 long authorization targets another run")
frozen_digest = hashlib.sha256(frozen_inputs.read_bytes()).hexdigest()
if value["frozen_inputs_sha256"] != frozen_digest:
    raise SystemExit("ADR-150 long authorization uses another frozen input receipt")

subject = value["subject"]
subject_fields = {
    "evidence_sha256",
    "execution_contract_sha256",
    "implementation_sha256",
    "model_family_sha256",
}
if not isinstance(subject, dict) or set(subject) != subject_fields:
    raise SystemExit("ADR-150 long authorization subject is malformed")
if any(
    not isinstance(subject[name], str)
    or len(subject[name]) != 64
    or any(character not in "0123456789abcdef" for character in subject[name])
    for name in subject_fields
):
    raise SystemExit("ADR-150 long authorization subject has an invalid digest")

checkpoint = value["checkpoint_report"]
if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}:
    raise SystemExit("ADR-150 checkpoint receipt is malformed")
checkpoint_path = Path(checkpoint["path"])
expected_checkpoint = run_dir / "checkpoints/global_step_2000/task_independent_checkpoint.json"
if checkpoint_path.resolve() != expected_checkpoint or checkpoint_path.is_symlink():
    raise SystemExit("ADR-150 authorization references another checkpoint")
checkpoint_payload = checkpoint_path.read_bytes()
if hashlib.sha256(checkpoint_payload).hexdigest() != checkpoint["sha256"]:
    raise SystemExit("ADR-150 accepted checkpoint report changed")
checkpoint_value = json.loads(checkpoint_payload.decode("utf-8"))
if checkpoint_value.get("status") != "PASS" or checkpoint_value.get("global_step") != 2000:
    raise SystemExit("ADR-150 checkpoint report did not pass at step 2000")
if any(checkpoint_value.get(name) != subject[name] for name in subject_fields):
    raise SystemExit("ADR-150 checkpoint provenance differs from authorization")

required_evidence = (
    "cold_resume_equivalence",
    "heldout_action",
    "calvin_rollout",
    "full_curve_comparison",
    "visual_review",
    "causal_interventions",
    "gradient_adoption",
    "filter_interaction",
    "full_modal_adoption",
)
evidence_schemas = {
    "cold_resume_equivalence": "picf-next.adr150-cold-resume-equivalence/v1",
    "heldout_action": "picf-next.adr150-heldout-action/v1",
    "calvin_rollout": "picf-next.adr150-calvin-rollout/v1",
    "full_curve_comparison": "picf-next.adr150-full-curve-comparison/v1",
    "visual_review": "picf-next.adr150-visual-review/v1",
    "causal_interventions": "picf-next.adr150-causal-interventions/v1",
    "gradient_adoption": "picf-next.adr150-gradient-adoption/v1",
    "filter_interaction": "picf-next.adr150-filter-interaction/v1",
    "full_modal_adoption": "picf-next.adr150-full-modal-adoption/v1",
}
evidence = value["evidence"]
if not isinstance(evidence, list) or tuple(item.get("name") for item in evidence) != required_evidence:
    raise SystemExit("ADR-150 long authorization evidence coverage or order differs")
observed_paths = {checkpoint_path.resolve(), authorization_path.resolve()}
for item in evidence:
    if not isinstance(item, dict) or set(item) != {"name", "path", "sha256"}:
        raise SystemExit("ADR-150 evidence receipt is malformed")
    path = Path(item["path"])
    resolved_path = path.resolve()
    if path.is_symlink() or not path.is_file() or resolved_path in observed_paths:
        raise SystemExit(f"ADR-150 evidence is absent: {item['name']}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise SystemExit(f"ADR-150 evidence changed: {item['name']}")
    report = json.loads(payload.decode("utf-8"))
    if (
        report.get("schema") != evidence_schemas[item["name"]]
        or report.get("status") != "PASS"
        or report.get("global_step") != 2000
    ):
        raise SystemExit(f"ADR-150 evidence did not pass at step 2000: {item['name']}")
    if Path(report.get("run_dir", "")).resolve() != run_dir:
        raise SystemExit(f"ADR-150 evidence targets another run: {item['name']}")
    if any(report.get(name) != subject[name] for name in subject_fields):
        raise SystemExit(f"ADR-150 evidence provenance differs: {item['name']}")
    observed_paths.add(resolved_path)

resume_checkpoint = run_dir / f"checkpoints/global_step_{load_global_step}"
resume_report = resume_checkpoint / "task_independent_checkpoint.json"
if (
    load_global_step < 2000
    or load_global_step > 30000
    or load_global_step % 2000
    or resume_checkpoint.is_symlink()
    or not resume_checkpoint.is_dir()
    or resume_report.is_symlink()
    or not resume_report.is_file()
):
    raise SystemExit("ADR-150 latest resume checkpoint is not a legal 2k boundary")
resume_value = json.loads(resume_report.read_text(encoding="utf-8"))
if (
    resume_value.get("status") != "PASS"
    or resume_value.get("global_step") != load_global_step
    or any(resume_value.get(name) != subject[name] for name in subject_fields)
):
    raise SystemExit("ADR-150 latest resume checkpoint differs from the accepted subject")

promotion_root = run_dir / "promotion"
promotion_root.mkdir(exist_ok=True)
if promotion_root.is_symlink() or not promotion_root.is_dir():
    raise SystemExit("ADR-150 promotion root must be one direct directory")
authorization_copy = promotion_root / "long_authorization.json"
authorization_payload = authorization_path.read_bytes()
if authorization_copy.exists():
    if authorization_copy.is_symlink() or authorization_copy.read_bytes() != authorization_payload:
        raise SystemExit("ADR-150 persisted long authorization differs")
else:
    write_exclusive_durable(authorization_copy, authorization_payload)
attempt = promotion_root / f"resume_from_{load_global_step:08d}.json"
attempt_payload = (
    json.dumps(
        {
            "schema": "picf-next.adr150-resume-attempt/v1",
            "authorization_sha256": hashlib.sha256(authorization_payload).hexdigest(),
            "checkpoint_report_sha256": hashlib.sha256(resume_report.read_bytes()).hexdigest(),
            "load_global_step": load_global_step,
            "subject": subject,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
).encode("ascii")
if attempt.exists():
    if attempt.is_symlink() or attempt.read_bytes() != attempt_payload:
        raise SystemExit("ADR-150 persisted resume attempt differs")
else:
    write_exclusive_durable(attempt, attempt_payload)
directory = os.open(promotion_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
print(f"ADR-150 long-run authorization=PASS resume_step={load_global_step}")
PY

export PICF_REPO=$REPO
export PICF_PYTHON=$PYTHON
exec "$REPO/adr150/run_full_picf.sh" resume "$RUN_DIR" 30000 "$LOAD_GLOBAL_STEP"
