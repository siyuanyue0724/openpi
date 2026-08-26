#!/usr/bin/env bash
set -euo pipefail

WORLD_SIZE=4
TRAINING_PREFIX_STEPS=${PICF_ADR209_TRAINING_PREFIX_STEPS:-250}
[[ "$TRAINING_PREFIX_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "PICF_ADR209_TRAINING_PREFIX_STEPS must be positive" >&2
  exit 2
}
CONTRACT_ROOT=${PICF_ADR209_CONTRACT_ROOT:-/mnt/picf-next/adr209/contracts/flare-4gpu-prefix${TRAINING_PREFIX_STEPS}-v1}
[[ "$CONTRACT_ROOT" == /mnt/* && ! -e "$CONTRACT_ROOT" && ! -L "$CONTRACT_ROOT" ]] || {
  echo "ADR-209 contract root must be one absent persistent path beneath /mnt" >&2
  exit 2
}

REPO=${PICF_ADR209_REPO:-/mnt/picf-next/adr209/worktree-local-sync}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
REFERENCE_SPLIT=${PICF_ADR209_REFERENCE_SPLIT:-/mnt/picf-next/adr207/contracts/native-query-posterior-4gpu-30k-v1/representation-split.json}

for path in "$REPO" "$DATASET"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required direct directory is absent: $path" >&2
    exit 1
  }
done
for path in "$PYTHON" "$DATASET_MANIFEST" "$REFERENCE_SPLIT"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required direct file is absent: $path" >&2
    exit 1
  }
done

file_sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

PARENT=$(dirname "$CONTRACT_ROOT")
mkdir -p "$PARENT"
STAGING=$(mktemp -d "$PARENT/.flare-4gpu-prefix${TRAINING_PREFIX_STEPS}-v1.XXXXXX")
cleanup() {
  rm -rf "$STAGING"
}
trap cleanup EXIT

STREAM_PLAN=$STAGING/stream-plan.json
REPRESENTATION_SPLIT=$STAGING/representation-split.json
SPLIT_REPORT=$STAGING/representation-split.build-report.json
EVALUATION_PLAN=$STAGING/evaluation-plan.json
DENSE_COVERAGE=$STAGING/dense-evidence-coverage.json

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"

"$PYTHON" tools/build_lingbot_representation_split.py \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --comparison-id lingbot-vla2-native-picf-full \
  --plan-seed 20260721 \
  --global-batch-size "$WORLD_SIZE" \
  --total-steps 30000 \
  --lane-interleave-factor 1 \
  --physical-event-stream \
  --minimum-future-source-frames 16 \
  --evaluation-reference-split "$REFERENCE_SPLIT" \
  --evaluation-reference-split-sha256 "$(file_sha256 "$REFERENCE_SPLIT")" \
  --allow-reference-budget-change \
  --segments-per-task 2 \
  --stream-plan-output "$STREAM_PLAN" \
  --representation-split-output "$REPRESENTATION_SPLIT" \
  --build-report-output "$SPLIT_REPORT"

"$PYTHON" tools/build_lingbot_entity_evaluation_plan.py \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --representation-split "$REPRESENTATION_SPLIT" \
  --representation-split-sha256 "$(file_sha256 "$REPRESENTATION_SPLIT")" \
  --world-size "$WORLD_SIZE" \
  --output "$EVALUATION_PLAN"

"$PYTHON" tools/build_calvin_dense_evidence_coverage_plan.py \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --stream-plan "$STREAM_PLAN" \
  --stream-plan-sha256 "$(file_sha256 "$STREAM_PLAN")" \
  --representation-split "$REPRESENTATION_SPLIT" \
  --representation-split-sha256 "$(file_sha256 "$REPRESENTATION_SPLIT")" \
  --evaluation-plan "$EVALUATION_PLAN" \
  --evaluation-plan-sha256 "$(file_sha256 "$EVALUATION_PLAN")" \
  --action-horizon 1 \
  --minimum-future-source-frames 16 \
  --training-step-prefix "$TRAINING_PREFIX_STEPS" \
  --output "$DENSE_COVERAGE"

"$PYTHON" - "$STAGING" "$REFERENCE_SPLIT" "$TRAINING_PREFIX_STEPS" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
reference = Path(sys.argv[2]).resolve()
prefix = int(sys.argv[3])
implementation = (
    "tools/build_lingbot_representation_split.py",
    "tools/build_lingbot_entity_evaluation_plan.py",
    "tools/build_calvin_dense_evidence_coverage_plan.py",
)
repo = Path.cwd()
files = tuple(sorted(path for path in root.iterdir() if path.is_file()))
payload = {
    "schema": "picf-next.adr209-contract-freeze/v1",
    "comparison_id": "adr209-native-videomt-flare-generic-v1",
    "future_source_frame_count": 16,
    "global_batch_size": 4,
    "world_size": 4,
    "total_steps": 30000,
    "training_cache_prefix_steps": prefix,
    "reference_evaluation_split": str(reference),
    "implementation_sha256": {
        relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        for relative in implementation
    },
    "files": {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    },
}
destination = root / "receipt.json"
with destination.open("x", encoding="ascii") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY

chmod 0444 "$STAGING"/*
mv -T "$STAGING" "$CONTRACT_ROOT"
trap - EXIT
sha256sum "$CONTRACT_ROOT"/*
