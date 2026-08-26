#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 anytouch|sonata|vjepa OUTPUT_ROOT PARTITION_ROOT..." >&2
  exit 2
fi

MODALITY=$1
OUTPUT_ROOT=$2
shift 2
case "$MODALITY" in anytouch|sonata|vjepa) ;; *) echo "unknown modality" >&2; exit 2 ;; esac
[[ "$OUTPUT_ROOT" == /mnt/* && ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
  echo "merged dense-cache output must be one absent persistent path" >&2
  exit 1
}
[[ ! -e "${OUTPUT_ROOT}.receipt.json" && ! -L "${OUTPUT_ROOT}.receipt.json" ]] || {
  echo "merged dense-cache receipt already exists" >&2
  exit 1
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PRODUCER_PYTHON:-/root/.venvs/openpi/bin/python}
COVERAGE=/mnt/picf-next/adr150/contracts/calvin-official-30k-v1/four-gpu-30k.physical.dense-evidence-coverage.json
PARTITION_ARGS=()
for root in "$@"; do
  [[ "$root" == /mnt/* && -d "$root" && ! -L "$root" ]] || {
    echo "dense-cache partition is absent or indirect: $root" >&2
    exit 1
  }
  manifest=$root/manifest.json
  [[ -f "$manifest" && ! -L "$manifest" ]] || {
    echo "dense-cache partition manifest is absent or indirect: $manifest" >&2
    exit 1
  }
  digest=$(sha256sum "$manifest")
  PARTITION_ARGS+=(--partition-root "$root" --partition-manifest-sha256 "${digest%% *}")
done

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"
exec "$PYTHON" tools/merge_dense_evidence_cache_partitions.py \
  "${PARTITION_ARGS[@]}" \
  --coverage-plan "$COVERAGE" \
  --coverage-plan-sha256 8a0ef47470625cfdddfd3f7bf15020e693b8fa146d5d4307e35c157348448c6f \
  --output-root "$OUTPUT_ROOT" \
  --receipt-output "${OUTPUT_ROOT}.receipt.json" \
  --reference-partitions
