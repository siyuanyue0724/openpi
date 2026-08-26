#!/usr/bin/env bash
set -euo pipefail

repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
g3_training_root=${PICF_G3_TRAINING_ROOT:-/mnt/picf-next/runs/adr160-g3-training-cb311b6-v1}
g3_training_report=$g3_training_root/ltop_g3_training_report.json
poll_seconds=${PICF_G3_POLL_SECONDS:-60}
log_root=${PICF_CORE_PILOT_LOG_ROOT:-/mnt/picf-next/logs}

if [[ "$repository_root" != /mnt/* ]]; then
  echo "post-G3 continuation must execute from an immutable /mnt source snapshot" >&2
  exit 1
fi
if [[ ! -x "$repository_root/adr161/run_ltop_core_pilot_2gpu.sh" ]]; then
  echo "LTOP core-pilot launcher is absent or not executable" >&2
  exit 1
fi
if [[ ! -x "$repository_root/adr161/run_ltop_g3_staged_evaluation_2gpu.sh" ]]; then
  echo "staged G3 evaluation launcher is absent or not executable" >&2
  exit 1
fi
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "PICF_G3_POLL_SECONDS must be a positive integer" >&2
  exit 1
fi

mkdir -p "$log_root"
export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
revision=$(git -C "$repository_root" rev-parse --verify HEAD)
short_revision=${revision:0:7}
smoke_run=/mnt/picf-next/runs/adr161-ltop-ec-factual-smoke-$short_revision-v1
pilot_run=/mnt/picf-next/runs/adr161-ltop-ec-factual-2k-$short_revision-v1
g3_evaluation_run=/mnt/picf-next/runs/adr160-g3-evaluation-$short_revision-v1
g3_composed_run=/mnt/picf-next/runs/adr160-g3-composed-$short_revision-v1
g3_report=$g3_composed_run/ltop_g3_action_mediation_report.json
smoke_log=$log_root/$(basename "$smoke_run").console.log
pilot_log=$log_root/$(basename "$pilot_run").console.log
g3_evaluation_log=$log_root/$(basename "$g3_evaluation_run").console.log
status_log=$log_root/adr161-post-g3-$short_revision.status.log

if [[ -e "$smoke_run" || -e "$pilot_run" || -e "$g3_evaluation_run" \
  || -e "$g3_composed_run" || -e "$smoke_log" || -e "$pilot_log" \
  || -e "$g3_evaluation_log" ]]; then
  echo "post-G3 continuation output already exists for revision $revision" >&2
  exit 1
fi

printf '%s waiting for G3 training report %s\n' "$(date --iso-8601=seconds)" \
  "$g3_training_report" >>"$status_log"
while [[ ! -f "$g3_training_report" ]]; do
  if ! pgrep -f 'run_lingbot_vla2_ltop_g3_action_mediation.py.*--phase training' >/dev/null; then
    printf '%s G3 training process exited without a final report\n' \
      "$(date --iso-8601=seconds)" \
      >>"$status_log"
    exit 1
  fi
  sleep "$poll_seconds"
done

# The report is published before every distributed worker necessarily releases
# its CUDA context.  Avoid overlapping two full-model process groups.
while pgrep -f 'run_lingbot_vla2_ltop_g3_action_mediation.py.*--phase training' >/dev/null; do
  sleep 5
done

printf '%s G3 training PASS; cold-starting isolated evaluation\n' \
  "$(date --iso-8601=seconds)" >>"$status_log"
PICF_REPOSITORY_ROOT="$repository_root" \
  "$repository_root/adr161/run_ltop_g3_staged_evaluation_2gpu.sh" \
  "$g3_training_root" "$g3_evaluation_run" "$g3_composed_run" \
  >"$g3_evaluation_log" 2>&1

printf '%s G3 PASS; starting disposable two-step smoke\n' "$(date --iso-8601=seconds)" \
  >>"$status_log"
PICF_REPOSITORY_ROOT="$repository_root" PICF_G3_RUN_ROOT="$g3_composed_run" \
  "$repository_root/adr161/run_ltop_core_pilot_2gpu.sh" \
  ltop-ec-factual "$smoke_run" smoke >"$smoke_log" 2>&1

"$python_bin" - "$smoke_run" <<'PY'
import json
import hashlib
import os
import shutil
import sys
from pathlib import Path

from picf_next.artifact_io import write_text_durable_exclusive

root = Path(sys.argv[1])
report_path = root / "ltop_core_pilot_report.json"
report = json.loads(report_path.read_text(encoding="ascii"))
if report.get("status") != "PASS" or report.get("failures") != []:
    raise SystemExit(f"engineering smoke failed: {report.get('failures')}")
if report.get("mode") != "smoke" or report.get("steps") != 2:
    raise SystemExit("engineering smoke report has the wrong registered cadence")
checkpoint = root / "checkpoints" / "global_step_2" / "ltop_core_pilot_checkpoint.json"
metrics = root / "metrics" / "steps_00000001_00000002.json"
diagnostics = sorted((root / "diagnostics" / "step_00000002").glob("rank_*.json"))
visuals = sorted((root / "entity_visuals" / "step_00000002").glob("rank_*/*.png"))
if not checkpoint.is_file() or not metrics.is_file():
    raise SystemExit("engineering smoke omitted its transactional checkpoint or metrics")
if len(diagnostics) != 2 or len(visuals) != 2:
    raise SystemExit("engineering smoke omitted a rank diagnostic or entity visual")

checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
checkpoint_directory = checkpoint.parent
shutil.rmtree(checkpoint_directory)
directory_descriptor = os.open(
    checkpoint_directory.parent,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
write_text_durable_exclusive(
    root / "discarded_smoke_checkpoint.json",
    json.dumps(
        {
            "schema": "picf-next.ltop-core-pilot-disposable-checkpoint.v1",
            "status": "DISCARDED_AFTER_PASS",
            "global_step": 2,
            "checkpoint_manifest_sha256": checkpoint_digest,
            "reason": "free persistent capacity before the formal 2k checkpoint",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
)
PY

printf '%s smoke PASS; cold-starting formal factual 2k pilot\n' \
  "$(date --iso-8601=seconds)" >>"$status_log"
exec env PICF_REPOSITORY_ROOT="$repository_root" \
  PICF_G3_RUN_ROOT="$g3_composed_run" \
  "$repository_root/adr161/run_ltop_core_pilot_2gpu.sh" \
  ltop-ec-factual "$pilot_run" pilot >"$pilot_log" 2>&1
