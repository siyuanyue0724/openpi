#!/usr/bin/env bash
set -euo pipefail

[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || {
  echo "FLARE cache publication requires one explicit CUDA_VISIBLE_DEVICES index" >&2
  exit 2
}

TRAINING_PREFIX_STEPS=${PICF_ADR209_TRAINING_PREFIX_STEPS:-250}
REPO=${PICF_ADR209_REPO:-/mnt/picf-next/adr209/worktree-local-sync}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
CONTRACT_ROOT=${PICF_ADR209_CONTRACT_ROOT:-/mnt/picf-next/adr209/contracts/flare-4gpu-prefix${TRAINING_PREFIX_STEPS}-v1}
TEACHER_ROOT=${PICF_FLARE_TEACHER_ROOT:-/mnt/picf-next/adr209/models/siglip2-large-patch16-256-787800c}
TEACHER_MANIFEST=${PICF_FLARE_TEACHER_MANIFEST:-/mnt/picf-next/adr209/models/siglip2-large-patch16-256-787800c.runtime-files.sha256}
OUTPUT_PARENT=${PICF_ADR209_FLARE_CACHE_PARENT:-/mnt/picf-next/adr209/caches/flare-4gpu-prefix${TRAINING_PREFIX_STEPS}-v1}
OUTPUT_ROOT=$OUTPUT_PARENT/future-targets
BUILD_REPORT=$OUTPUT_PARENT/future-targets.build-report.json

for path in "$REPO" "$DATASET" "$CONTRACT_ROOT" "$TEACHER_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required direct directory is absent: $path" >&2
    exit 1
  }
done
for path in "$PYTHON" "$DATASET_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required direct file is absent: $path" >&2
    exit 1
  }
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
  echo "FLARE target cache already exists: $OUTPUT_ROOT" >&2
  exit 2
}
[[ ! -e "$BUILD_REPORT" && ! -L "$BUILD_REPORT" ]] || {
  echo "FLARE target build report already exists: $BUILD_REPORT" >&2
  exit 2
}

if [[ ! -e "$TEACHER_MANIFEST" && ! -L "$TEACHER_MANIFEST" ]]; then
  "$PYTHON" - "$TEACHER_ROOT" "$TEACHER_MANIFEST" <<'PY'
import hashlib
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
files = tuple(
    sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    )
)
if not files:
    raise RuntimeError("SigLIP2 teacher root is empty")
rows = []
for path in files:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append(f"{digest.hexdigest()}  {path.relative_to(root).as_posix()}\n")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
descriptor = os.open(output, flags, 0o444)
with os.fdopen(descriptor, "w", encoding="ascii") as stream:
    stream.writelines(rows)
    stream.flush()
    os.fsync(stream.fileno())
PY
fi
[[ -f "$TEACHER_MANIFEST" && ! -L "$TEACHER_MANIFEST" ]] || {
  echo "stable SigLIP2 teacher manifest is absent" >&2
  exit 1
}

file_sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

mkdir -p "$OUTPUT_PARENT"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"
exec "$PYTHON" tools/build_flare_future_target_cache.py \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --stream-plan "$CONTRACT_ROOT/stream-plan.json" \
  --stream-plan-sha256 "$(file_sha256 "$CONTRACT_ROOT/stream-plan.json")" \
  --representation-split "$CONTRACT_ROOT/representation-split.json" \
  --representation-split-sha256 "$(file_sha256 "$CONTRACT_ROOT/representation-split.json")" \
  --teacher-root "$TEACHER_ROOT" \
  --teacher-files-manifest "$TEACHER_MANIFEST" \
  --teacher-files-manifest-sha256 "$(file_sha256 "$TEACHER_MANIFEST")" \
  --training-prefix-steps "$TRAINING_PREFIX_STEPS" \
  --encoder-batch-size "${PICF_FLARE_ENCODER_BATCH_SIZE:-32}" \
  --records-per-shard 256 \
  --device cuda:0 \
  --compute-dtype bfloat16 \
  --output-root "$OUTPUT_ROOT" \
  --build-report "$BUILD_REPORT"
