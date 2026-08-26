#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 TRIAL_PID_FILE TRIAL_REPORT CHECKPOINT COLD_RUN COLD_LOG COLD_PID_FILE REPOSITORY_ROOT" >&2
  exit 2
fi

trial_pid_file=$1
trial_report=$2
checkpoint=$3
cold_run=$4
cold_log=$5
cold_pid_file=$6
repository_root=$7
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
poll_seconds=${PICF_ADR172_WATCH_POLL_SECONDS:-30}

for path in "$trial_pid_file" "$repository_root"; do
  [[ "$path" == /mnt/* && -e "$path" ]] || {
    echo "required watcher input is absent or not persistent: $path" >&2
    exit 1
  }
done
for path in "$trial_report" "$checkpoint" "$cold_run" "$cold_log" "$cold_pid_file"; do
  [[ "$path" == /mnt/* ]] || {
    echo "watcher output/input path is not persistent: $path" >&2
    exit 1
  }
done
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo "PICF_ADR172_WATCH_POLL_SECONDS must be a positive integer" >&2
  exit 1
}

trial_pid=$(<"$trial_pid_file")
[[ "$trial_pid" =~ ^[1-9][0-9]*$ ]] || {
  echo "trial PID file does not contain a positive integer" >&2
  exit 1
}

while kill -0 "$trial_pid" 2>/dev/null; do
  if [[ -r "/proc/$trial_pid/stat" ]] && [[ $(awk '{print $3}' "/proc/$trial_pid/stat") == Z ]]; then
    break
  fi
  sleep "$poll_seconds"
done

[[ -f "$trial_report" ]] || {
  echo "trial exited without publishing its report: $trial_report" >&2
  exit 1
}
manifest=$checkpoint/ltop_g3_training_checkpoint.json
[[ -f "$manifest" && -f "$checkpoint/model/.metadata" ]] || {
  echo "trial PASS artifact omits the required checkpoint manifest or DCP metadata" >&2
  exit 1
}
compgen -G "$checkpoint/model/*.distcp" >/dev/null || {
  echo "trial PASS artifact omits DCP shards" >&2
  exit 1
}

"$python_bin" - "$trial_report" "$manifest" <<'PY'
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="ascii"))
expected_scope = "guidedvla-fixed-object-heads-0-1"
expected_heads = [0, 1]
expected_weight = 0.001

if report.get("status") != "PASS" or report.get("failures") != []:
    raise SystemExit("trial report did not pass cleanly")
if report.get("phase") != "training" or report.get("mode") != "direct-trial":
    raise SystemExit("trial report has the wrong phase or mode")
if report.get("steps") != 256:
    raise SystemExit("trial report did not bind the registered 256-step screen")
contract = report.get("training_contract", {})
adoption = contract.get("direct_posterior_adoption", {})
weights = contract.get("loss_weights", {})
if adoption.get("head_scope") != expected_scope:
    raise SystemExit("trial report has the wrong fixed-head scope")
if adoption.get("head_indices") != expected_heads:
    raise SystemExit("trial report has the wrong fixed-head indices")
if not math.isclose(float(weights.get("direct_grounding", math.nan)), expected_weight):
    raise SystemExit("trial report has the wrong grounding weight")

if manifest.get("status") != "PASS" or manifest.get("global_step") != 256:
    raise SystemExit("checkpoint manifest did not pass at global step 256")
if manifest.get("direct_posterior_head_scope") != expected_scope:
    raise SystemExit("checkpoint manifest has the wrong fixed-head scope")
if manifest.get("direct_posterior_head_indices") != expected_heads:
    raise SystemExit("checkpoint manifest has the wrong fixed-head indices")
if not math.isclose(float(manifest.get("direct_grounding_weight", math.nan)), expected_weight):
    raise SystemExit("checkpoint manifest has the wrong grounding weight")
PY

[[ ! -e "$cold_run" && ! -L "$cold_run" ]] || {
  echo "cold output path already exists: $cold_run" >&2
  exit 1
}
mkdir -p "$(dirname "$cold_log")" "$(dirname "$cold_pid_file")"
printf '%s\n' "$$" >"$cold_pid_file"
exec env PICF_REPOSITORY_ROOT="$repository_root" \
  "$repository_root/adr172/run_guidedvla_fixed_object_heads_cold_action_2gpu.sh" \
  "$checkpoint" "$cold_run" >"$cold_log" 2>&1
