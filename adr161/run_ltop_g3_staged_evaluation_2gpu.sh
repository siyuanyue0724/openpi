#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TRAINING_RUN_ROOT EVALUATION_RUN_ROOT COMPOSED_RUN_ROOT" >&2
  exit 2
fi

training_root=$1
evaluation_root=$2
composed_root=$3
repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
source_checkout=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr162-muon-align-dbed72e-v1}
runtime_hotfix=${PICF_LINGBOT_RUNTIME_HOTFIX:-$repository_root/references/patches/lingbot_vla2_distributed_muon_collective_alignment.patch}
evaluation_timeout_seconds=${PICF_G3_EVALUATION_TIMEOUT_SECONDS:-3600}
training_report=$training_root/ltop_g3_training_report.json
evaluation_report=$evaluation_root/ltop_g3_evaluation_report.json
final_report=$composed_root/ltop_g3_action_mediation_report.json

if [[ "$repository_root" != /mnt/* ]]; then
  echo "staged G3 evaluation must execute from an immutable /mnt source snapshot" >&2
  exit 1
fi
for path in "$training_report" "$source_checkout" "$runtime_hotfix"; do
  if [[ ! -e "$path" ]]; then
    echo "required staged G3 artifact is absent: $path" >&2
    exit 1
  fi
done
for path in "$evaluation_root" "$composed_root"; do
  if [[ -e "$path" ]]; then
    echo "staged G3 output already exists: $path" >&2
    exit 1
  fi
  if [[ "$path" != /mnt/* ]]; then
    echo "staged G3 outputs must live under /mnt" >&2
    exit 1
  fi
done
if [[ ! "$evaluation_timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "PICF_G3_EVALUATION_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi

trained_checkpoint=$(
  PYTHONPATH="$repository_root:$repository_root/src" "$python_bin" - "$training_report" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="ascii"))
expected = {
    "schema": "picf-next.ltop-g3-training-phase.v1",
    "status": "PASS",
    "failures": [],
    "phase": "training",
    "mode": "gate",
    "steps": 128,
    "eval_every": 32,
    "world_size": 2,
}
for field, value in expected.items():
    if report.get(field) != value:
        raise SystemExit(f"G3 training report violates {field}: {report.get(field)!r}")
checkpoint = report.get("checkpoint")
if not isinstance(checkpoint, dict) or checkpoint.get("optimizer_saved") is not False:
    raise SystemExit("G3 training report omits its model-only checkpoint")
checkpoint_path = Path(checkpoint.get("path", ""))
if not checkpoint_path.is_dir() or checkpoint_path.is_symlink():
    raise SystemExit(f"G3 training checkpoint is absent: {checkpoint_path}")
print(checkpoint_path.resolve())
PY
)

mkdir -p "$evaluation_root" "$composed_root"
cd "$repository_root"
export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

set +e
timeout --signal=TERM --kill-after=60s "$evaluation_timeout_seconds" \
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/run_lingbot_vla2_ltop_g3_action_mediation.py \
  --source-checkout "$source_checkout" \
  --patch "$repository_root/references/patches/lingbot_vla2_picf_native.patch" \
  --runtime-hotfix "$runtime_hotfix" \
  --checkpoint-dir /mnt/picf-next/models/lingbot-vla-v2-6b \
  --processor-dir /mnt/picf-next/models/qwen3-vl-4b-instruct \
  --stage-checkpoint /mnt/picf-next/checkpoints/adr160-g2b-confirm-49eac80-v3 \
  --g2-report /mnt/picf-next/runs/adr160-g2b-confirm-49eac80-v3/ltop_g2_representation_report.json \
  --dataset-split /mnt/calvin_data/task_ABC_D/training \
  --dataset-manifest /mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json \
  --norm-stats /mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json \
  --physical-sidecar-root /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z \
  --physical-sidecar-manifest /mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json \
  --physical-sidecar-manifest-sha256 ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d \
  --execution-contract /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json \
  --offline-labels /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json \
  --output "$evaluation_report" \
  --trained-checkpoint "$trained_checkpoint" \
  --mode gate \
  --phase evaluation \
  --steps 128 \
  --eval-every 32 \
  --cuda-allocator expandable-segments
evaluation_status=$?
set -e
if [[ "$evaluation_status" -ne 0 ]]; then
  printf '{"schema":"picf-next.ltop-g3-evaluation-runtime-failure.v1","status":"FAIL","exit_code":%d,"timeout_seconds":%d}\n' \
    "$evaluation_status" "$evaluation_timeout_seconds" \
    >"$evaluation_root/ltop_g3_evaluation_runtime_failure.json"
  exit "$evaluation_status"
fi

"$python_bin" tools/compose_ltop_g3_staged.py \
  --training-report "$training_report" \
  --evaluation-report "$evaluation_report" \
  --output "$final_report"

"$python_bin" - "$final_report" <<'PY'
import sys
from pathlib import Path

from picf_next.lingbot_native.ltop_core_pilot import load_accepted_g3_gate

accepted = load_accepted_g3_gate(Path(sys.argv[1]))
print(accepted.file_sha256)
PY
