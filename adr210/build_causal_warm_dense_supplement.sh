#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 anytouch|sonata|vjepa" >&2
  exit 2
fi
MODALITY=$1
case "$MODALITY" in anytouch|sonata|vjepa) ;; *)
  echo "unknown dense modality: $MODALITY" >&2
  exit 2
esac
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || {
  echo "causal-warm supplement publication requires one explicit GPU" >&2
  exit 2
}

REPO=${PICF_ADR210_REPO:-/mnt/picf-next/adr210/staging-v22}
PYTHON=${PICF_PRODUCER_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
DATASET_ROOT=${PICF_CALVIN_ROOT:-/mnt/calvin_data/task_ABC_D}
TARGET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
SOURCE_MANIFEST=${PICF_CALVIN_SOURCE_DATASET_MANIFEST:-/mnt/picf-next/manifests/calvin-training-files.json}
SOURCE_RECEIPT=${PICF_CALVIN_SOURCE_RECEIPT:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/receipt.json}
ASSET_MANIFEST=${PICF_FULL_MODAL_ASSET_MANIFEST:-/mnt/picf-next/manifests/full_modal_assets.json}
TARGET_COVERAGE=${PICF_ADR210_DENSE_COVERAGE:-/mnt/picf-next/adr210/contracts/causal-warm-4gpu-30k-v2/dense-evidence-coverage.json}
DONOR_COVERAGE=${PICF_ADR210_DONOR_DENSE_COVERAGE:-/mnt/picf-next/adr207/contracts/native-query-posterior-4gpu-30k-v1/dense-evidence-coverage.json}
DONOR_ROOT=${PICF_ADR210_DONOR_DENSE_CACHE_ROOT:-/mnt/picf-next/adr207/caches/native-query-posterior-4gpu-30k-v1}
OUTPUT_ROOT=${PICF_ADR210_DENSE_SUPPLEMENT_ROOT:-/mnt/picf-next/adr210/caches/causal-warm-history-v2}
VJEPA_ENCODER_BATCH_SIZE=${PICF_VJEPA_ENCODER_BATCH_SIZE:-8}

file_sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

case "$MODALITY" in
  anytouch)
    OUTPUT=$OUTPUT_ROOT/anytouch
    DONOR=$DONOR_ROOT/anytouch-observed-pose
    MODALITY_ARGS=(
      --tactile-calibration-archive /mnt/picf-next-provenance/calvin-tactile-calibration-identity-a60b7934/tactile_backgrounds.npz
      --tactile-calibration-receipt /mnt/picf-next-provenance/calvin-tactile-calibration-identity-a60b7934/tactile_backgrounds.receipt.json
      --tactile-calibration-receipt-sha256 3bb381922df52c2cd561a2acb1824bc17c78dcacc2ca971727736c62c4baca65
      --encoder-batch-size 1
    )
    ;;
  sonata)
    OUTPUT=$OUTPUT_ROOT/sonata
    DONOR=$DONOR_ROOT/sonata-native
    MODALITY_ARGS=(
      --camera-calibration "$DATASET_ROOT/calib/cameras.json"
      --point-pixel-stride 2
      --point-budget 4096
      --encoder-batch-size 1
    )
    ;;
  vjepa)
    OUTPUT=$OUTPUT_ROOT/vjepa
    DONOR=$DONOR_ROOT/vjepa-final
    MODALITY_ARGS=(--encoder-batch-size "$VJEPA_ENCODER_BATCH_SIZE")
    ;;
esac

for path in "$REPO" "$DATASET_ROOT" "$DONOR"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required direct directory is absent: $path" >&2
    exit 1
  }
done
for path in \
  "$PYTHON" "$TARGET_MANIFEST" "$SOURCE_MANIFEST" "$SOURCE_RECEIPT" \
  "$ASSET_MANIFEST" "$TARGET_COVERAGE" "$DONOR_COVERAGE" "$DONOR/manifest.json"
do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required direct file is absent: $path" >&2
    exit 1
  }
done
if [[ -e "$OUTPUT.receipt.json" || -L "$OUTPUT.receipt.json" ]]; then
  echo "completed causal-warm supplement receipt already exists: $OUTPUT.receipt.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"
exec "$PYTHON" tools/republish_calvin_frozen_evidence_cache.py \
  --dataset-root "$DATASET_ROOT" \
  --split training \
  --target-dataset-manifest "$TARGET_MANIFEST" \
  --donor-dataset-manifest "$TARGET_MANIFEST" \
  --source-dataset-manifest "$SOURCE_MANIFEST" \
  --source-receipt "$SOURCE_RECEIPT" \
  --source-receipt-sha256 "$(file_sha256 "$SOURCE_RECEIPT")" \
  --asset-manifest "$ASSET_MANIFEST" \
  --coverage-plan "$TARGET_COVERAGE" \
  --coverage-plan-sha256 "$(file_sha256 "$TARGET_COVERAGE")" \
  --donor-coverage-plan "$DONOR_COVERAGE" \
  --donor-coverage-plan-sha256 "$(file_sha256 "$DONOR_COVERAGE")" \
  --donor-cache-root "$DONOR" \
  --donor-cache-manifest-sha256 "$(file_sha256 "$DONOR/manifest.json")" \
  --missing-from-donor-only \
  --modality "$MODALITY" \
  --output-root "$OUTPUT" \
  --receipt-output "$OUTPUT.receipt.json" \
  --device cuda:0 \
  --token-dtype float16 \
  --shard-rows 64 \
  --action-horizon 1 \
  "${MODALITY_ARGS[@]}"
