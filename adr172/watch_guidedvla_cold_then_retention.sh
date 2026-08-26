#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 COLD_PID_FILE COLD_REPORT COLD_VALIDATION CHECKPOINT RETENTION_RUN RETENTION_LOG RETENTION_PID_FILE REPOSITORY_ROOT" >&2
  exit 2
fi

cold_pid_file=$1
cold_report=$2
cold_validation=$3
checkpoint=$4
retention_run=$5
retention_log=$6
retention_pid_file=$7
repository_root=$8
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
poll_seconds=${PICF_ADR172_WATCH_POLL_SECONDS:-30}

for path in "$repository_root"; do
  [[ "$path" == /mnt/* && -e "$path" ]] || {
    echo "required watcher input is absent or not persistent: $path" >&2
    exit 1
  }
done
for path in \
  "$cold_pid_file" \
  "$cold_report" \
  "$cold_validation" \
  "$checkpoint" \
  "$retention_run" \
  "$retention_log" \
  "$retention_pid_file"; do
  [[ "$path" == /mnt/* ]] || {
    echo "watcher output/input path is not persistent: $path" >&2
    exit 1
  }
done
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo "PICF_ADR172_WATCH_POLL_SECONDS must be a positive integer" >&2
  exit 1
}

while [[ ! -s "$cold_pid_file" ]]; do
  sleep "$poll_seconds"
done
cold_pid=$(<"$cold_pid_file")
[[ "$cold_pid" =~ ^[1-9][0-9]*$ ]] || {
  echo "cold PID file does not contain a positive integer" >&2
  exit 1
}

while kill -0 "$cold_pid" 2>/dev/null; do
  if [[ -r "/proc/$cold_pid/stat" ]] && [[ $(awk '{print $3}' "/proc/$cold_pid/stat") == Z ]]; then
    break
  fi
  sleep "$poll_seconds"
done

[[ -f "$cold_report" && -f "$cold_validation" ]] || {
  echo "cold gate exited without publishing both report and validation" >&2
  exit 1
}

"$python_bin" - "$cold_report" "$cold_validation" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
report = json.loads(report_path.read_text(encoding="ascii"))
validation = json.loads(validation_path.read_text(encoding="ascii"))
expected_scope = "guidedvla-fixed-object-heads-0-1"
expected_heads = [0, 1]
expected_weight = 0.001

if report.get("status") != "PASS" or report.get("failures") != []:
    raise SystemExit("cold report did not pass cleanly")
if report.get("phase") != "evaluation" or report.get("mode") != "gate":
    raise SystemExit("cold report has the wrong phase or mode")
if report.get("steps") != 128 or report.get("evaluation_scope") != "full":
    raise SystemExit("cold report is not the registered full gate")
contract = report.get("training_contract", {})
adoption = contract.get("direct_posterior_adoption", {})
weights = contract.get("loss_weights", {})
if adoption.get("head_scope") != expected_scope:
    raise SystemExit("cold report has the wrong fixed-head scope")
if adoption.get("head_indices") != expected_heads:
    raise SystemExit("cold report has the wrong fixed-head indices")
if not math.isclose(float(weights.get("direct_grounding", math.nan)), expected_weight):
    raise SystemExit("cold report has the wrong grounding weight")

if validation.get("status") != "PASS" or validation.get("failures") != []:
    raise SystemExit("independent cold validation did not pass cleanly")
report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
if validation.get("source_report_sha256") != report_sha256:
    raise SystemExit("independent cold validation does not bind the cold report bytes")
PY

[[ ! -e "$retention_run" && ! -L "$retention_run" ]] || {
  echo "retention output path already exists: $retention_run" >&2
  exit 1
}
mkdir -p "$(dirname "$retention_log")" "$(dirname "$retention_pid_file")"
printf '%s\n' "$$" >"$retention_pid_file"
exec env PICF_REPOSITORY_ROOT="$repository_root" \
  "$repository_root/adr172/run_guidedvla_fixed_object_heads_retention_2gpu.sh" \
  "$checkpoint" "$retention_run" >"$retention_log" 2>&1
