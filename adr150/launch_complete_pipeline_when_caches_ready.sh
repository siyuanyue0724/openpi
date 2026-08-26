#!/usr/bin/env bash
set -euo pipefail

# Fail-closed ADR-150 promotion: final caches -> audits/acceptance -> matched LBOT -> 2k.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PRODUCER_PYTHON=${PICF_PRODUCER_PYTHON:-/root/.venvs/openpi/bin/python}
ACTOR_PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
DATASET_ROOT=${PICF_CALVIN_DATASET_ROOT:-/mnt/calvin_data/task_ABC_D}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
CALIBRATION=${PICF_TACTILE_CALIBRATION:-/mnt/picf-next-provenance/calvin-tactile-calibration-identity-a60b7934/tactile_backgrounds.npz}
CALIBRATION_RECEIPT=${PICF_TACTILE_CALIBRATION_RECEIPT:-/mnt/picf-next-provenance/calvin-tactile-calibration-identity-a60b7934/tactile_backgrounds.receipt.json}
CACHE_ROOT=${PICF_ADR150_CACHE_ROOT:-/mnt/picf-next/adr150/caches/calvin-official-30k-v1}
COVERAGE=${PICF_ADR150_COVERAGE:-/mnt/picf-next/adr150/contracts/calvin-official-30k-v1/four-gpu-30k.physical.dense-evidence-coverage.json}
ANYTOUCH=$CACHE_ROOT/anytouch-observed-pose
SONATA=$CACHE_ROOT/sonata-native
VJEPA=$CACHE_ROOT/vjepa-final
SOURCE_AUDIT=$CACHE_ROOT/full-dense-source-input-audit.json
SEMANTIC_AUDIT=$CACHE_ROOT/full-dense-semantic-audit.json
ADOPTION=${PICF_ADR150_FULL_MODAL_ACTION_ADOPTION:-/mnt/picf-next/adr150/acceptance/full_modal_action_adoption.json}
SUITE_ROOT=${PICF_ADR150_ACCEPTANCE_SUITE_ROOT:-/mnt/picf-next/adr150/acceptance/full-cache-r1}
LBOT_RUN=${PICF_ADR150_LBOT_RUN:-/mnt/picf-next/runs/adr150-matched-lbot-200-fullmodal-r1}
TRAIN_RUN=${PICF_ADR150_TRAIN_RUN:-/mnt/picf-next/runs/adr150-fullmodal-initial2k-r1}
STATE=${PICF_ADR150_PIPELINE_STATE:-/mnt/picf-next/adr150/acceptance/full-cache-pipeline-r1.state}
SOURCE_LOG=${PICF_ADR150_SOURCE_AUDIT_LOG:-/mnt/picf-next/adr150/logs/full-dense-source-input-audit-r1.log}

COVERAGE_FILE_SHA=8a0ef47470625cfdddfd3f7bf15020e693b8fa146d5d4307e35c157348448c6f
COVERAGE_ARTIFACT_SHA=5478693bc4796cb2857aba44dc66cc465e91fad569d67d3ec7bd047194e9469e
CALIBRATION_RECEIPT_SHA=3bb381922df52c2cd561a2acb1824bc17c78dcacc2ca971727736c62c4baca65

PARTITIONS=(
  "$CACHE_ROOT/vjepa-parts/p0-000000-008068"
  "$CACHE_ROOT/vjepa-parts/p1-008068-036068"
  "$CACHE_ROOT/vjepa-parts/p2-036068-064068"
  "$CACHE_ROOT/vjepa-parts/p3a-064068-078068"
  "$CACHE_ROOT/vjepa-parts/p3b-078068-092068"
  "$CACHE_ROOT/vjepa-parts/p4-092068-106068"
  "$CACHE_ROOT/vjepa-parts/p5-106068-113068"
  "$CACHE_ROOT/vjepa-parts/p6-113068-120068"
)

stage() {
  local value=$1
  local tmp
  tmp=$(mktemp "${STATE}.XXXXXX")
  printf '%s\n' "$value" >"$tmp"
  mv "$tmp" "$STATE"
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$value"
}

sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

required_final_artifacts() {
  printf '%s\n' \
    "$ANYTOUCH/manifest.json" "$ANYTOUCH.receipt.json" \
    "$SONATA/manifest.json" "$SONATA.receipt.json"
  local root
  for root in "${PARTITIONS[@]}"; do
    printf '%s\n' "$root/manifest.json" "$root.receipt.json"
  done
}

[[ "$REPO" == /mnt/* && -d "$REPO" && ! -L "$REPO" ]] || {
  echo "pipeline requires one direct clean repository under /mnt" >&2
  exit 1
}
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "pipeline requires an exact clean repository" >&2
  exit 1
}
for path in "$PRODUCER_PYTHON" "$ACTOR_PYTHON"; do
  [[ -x "$path" && -f "$(readlink -f "$path")" ]] || {
    echo "pipeline Python entry point is not executable: $path" >&2
    exit 1
  }
done
for path in "$DATASET_MANIFEST" "$CALIBRATION" "$CALIBRATION_RECEIPT" "$COVERAGE"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "pipeline input is absent or indirect: $path" >&2
    exit 1
  }
done
[[ "$(sha256 "$COVERAGE")" == "$COVERAGE_FILE_SHA" ]] || {
  echo "coverage plan identity changed" >&2
  exit 1
}
[[ "$(sha256 "$CALIBRATION_RECEIPT")" == "$CALIBRATION_RECEIPT_SHA" ]] || {
  echo "tactile calibration receipt identity changed" >&2
  exit 1
}
for path in "$SOURCE_AUDIT" "$SEMANTIC_AUDIT" "$ADOPTION" "$SUITE_ROOT" \
  "$LBOT_RUN" "$TRAIN_RUN"; do
  [[ ! -e "$path" && ! -L "$path" ]] || {
    echo "pipeline output already exists: $path" >&2
    exit 1
  }
done
if [[ -e "$VJEPA" || -e "$VJEPA.receipt.json" || -L "$VJEPA" || -L "$VJEPA.receipt.json" ]]; then
  [[ -d "$VJEPA" && ! -L "$VJEPA" && -f "$VJEPA/manifest.json" \
    && ! -L "$VJEPA/manifest.json" && -f "$VJEPA.receipt.json" \
    && ! -L "$VJEPA.receipt.json" ]] || {
    echo "V-JEPA merged cache and receipt must be jointly absent or jointly direct" >&2
    exit 1
  }
  REUSE_VJEPA=1
else
  REUSE_VJEPA=0
fi
mkdir -p "$(dirname "$STATE")" "$(dirname "$SOURCE_LOG")"

stage WAITING_FOR_FINAL_CACHES
while true; do
  missing=0
  while IFS= read -r path; do
    if [[ ! -f "$path" || -L "$path" ]]; then
      missing=1
      break
    fi
  done < <(required_final_artifacts)
  [[ "$missing" -eq 0 ]] && break
  sleep 60
done

# A final manifest precedes process exit by a small atomic-publication tail.
while pgrep -f 'tools/republish_calvin_frozen_evidence_cache.py' >/dev/null; do
  sleep 10
done

if [[ "$REUSE_VJEPA" -eq 1 ]]; then
  stage REUSING_VJEPA_INDEX
else
  stage MERGING_VJEPA
  PICF_REPO=$REPO PICF_PRODUCER_PYTHON=$PRODUCER_PYTHON \
    "$REPO/adr150/merge_dense_cache_partitions.sh" vjepa "$VJEPA" "${PARTITIONS[@]}"
fi

ANY_SHA=$(sha256 "$ANYTOUCH/manifest.json")
SONATA_SHA=$(sha256 "$SONATA/manifest.json")
VJEPA_SHA=$(sha256 "$VJEPA/manifest.json")
DATASET_TREE_SHA=$(
  "$PRODUCER_PYTHON" - "$DATASET_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tree = value.get("tree_sha256")
if not isinstance(tree, str) or len(tree) != 64:
    raise SystemExit("dataset tree identity is absent")
print(tree)
PY
)

stage RUNNING_SEMANTIC_AUDIT
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src:$REPO" "$PRODUCER_PYTHON" \
  "$REPO/tools/audit_calvin_dense_evidence_semantics.py" \
  --dataset-tree-sha256 "$DATASET_TREE_SHA" \
  --coverage-plan-sha256 "$COVERAGE_ARTIFACT_SHA" \
  --anytouch-cache "$ANYTOUCH" --anytouch-manifest-sha256 "$ANY_SHA" \
  --sonata-cache "$SONATA" --sonata-manifest-sha256 "$SONATA_SHA" \
  --vjepa-cache "$VJEPA" --vjepa-manifest-sha256 "$VJEPA_SHA" \
  --output "$SEMANTIC_AUDIT"

stage RUNNING_SOURCE_AUDIT_AND_FOUR_GPU_ACCEPTANCE
ionice -c 3 nice -n 10 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src:$REPO" \
  "$PRODUCER_PYTHON" "$REPO/tools/audit_calvin_dense_evidence_source_inputs.py" \
  --dataset-root "$DATASET_ROOT" --split training \
  --dataset-manifest "$DATASET_MANIFEST" \
  --coverage-plan-sha256 "$COVERAGE_ARTIFACT_SHA" \
  --anytouch-cache "$ANYTOUCH" --anytouch-manifest-sha256 "$ANY_SHA" \
  --sonata-cache "$SONATA" --sonata-manifest-sha256 "$SONATA_SHA" \
  --vjepa-cache "$VJEPA" --vjepa-manifest-sha256 "$VJEPA_SHA" \
  --tactile-calibration-archive "$CALIBRATION" \
  --tactile-calibration-receipt "$CALIBRATION_RECEIPT" \
  --tactile-calibration-receipt-sha256 "$CALIBRATION_RECEIPT_SHA" \
  --workers "${PICF_SOURCE_AUDIT_WORKERS:-32}" \
  --progress-every 1024 --output "$SOURCE_AUDIT" >"$SOURCE_LOG" 2>&1 &
SOURCE_PID=$!

PICF_REPO=$REPO PICF_PYTHON=$ACTOR_PYTHON \
  "$REPO/adr150/run_full_modal_acceptance_suite_4gpu.sh" "$SUITE_ROOT"

stage RUNNING_MATCHED_LBOT_200
PICF_REPO=$REPO PICF_PYTHON=$ACTOR_PYTHON \
  "$REPO/adr150/run_matched_lbot_4gpu.sh" "$LBOT_RUN" 200
LBOT_REPORT=$LBOT_RUN/official_lbot_steps_200.json
[[ -f "$LBOT_REPORT" && ! -L "$LBOT_REPORT" ]] || {
  echo "matched LBOT report was not published" >&2
  exit 1
}

stage WAITING_FOR_SOURCE_AUDIT
wait "$SOURCE_PID"
[[ -f "$SOURCE_AUDIT" && ! -L "$SOURCE_AUDIT" ]] || {
  echo "full source-input audit was not published" >&2
  exit 1
}

stage FREEZING_INPUTS
PICF_REPO=$REPO PICF_PYTHON=$ACTOR_PYTHON \
  "$REPO/adr150/freeze_inputs.sh" "$LBOT_REPORT"

stage STARTING_FULL_MODAL_2K
PICF_REPO=$REPO PICF_PYTHON=$ACTOR_PYTHON \
  exec "$REPO/adr150/launch_four_gpu_initial_2k.sh" "$TRAIN_RUN" "$LBOT_REPORT"
