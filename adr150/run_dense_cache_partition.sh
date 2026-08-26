#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 anytouch|sonata|vjepa RECORD_START RECORD_STOP OUTPUT_ROOT" >&2
  exit 2
fi

MODALITY=$1
RECORD_START=$2
RECORD_STOP=$3
OUTPUT_ROOT=$4
[[ "$RECORD_START" =~ ^(0|[1-9][0-9]*)$ && "$RECORD_STOP" =~ ^[1-9][0-9]*$ ]] || {
  echo "dense-cache partition bounds must be canonical non-negative integers" >&2
  exit 2
}
(( RECORD_START < RECORD_STOP && RECORD_STOP <= 120068 )) || {
  echo "dense-cache partition bounds lie outside the official coverage" >&2
  exit 2
}
[[ "$OUTPUT_ROOT" == /mnt/* && ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
  echo "dense-cache partition output must be one absent persistent path" >&2
  exit 1
}
[[ ! -e "${OUTPUT_ROOT}.receipt.json" && ! -L "${OUTPUT_ROOT}.receipt.json" ]] || {
  echo "dense-cache partition receipt already exists" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || {
  echo "dense-cache partition requires exactly one explicit CUDA_VISIBLE_DEVICES index" >&2
  exit 1
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PRODUCER_PYTHON:-/root/.venvs/openpi/bin/python}
VJEPA_ENCODER_BATCH_SIZE=${PICF_VJEPA_ENCODER_BATCH_SIZE:-8}
[[ "$VJEPA_ENCODER_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "V-JEPA encoder batch size must be a canonical positive integer" >&2
  exit 2
}
DATASET_ROOT=/mnt/calvin_data/task_ABC_D
TARGET_MANIFEST=/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json
SOURCE_MANIFEST=/mnt/picf-next/manifests/calvin-training-files.json
SOURCE_RECEIPT=/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/receipt.json
ASSET_MANIFEST=/mnt/picf-next/manifests/full_modal_assets.json
CONTRACT_ROOT=/mnt/picf-next/adr150/contracts/calvin-official-30k-v1
COVERAGE=$CONTRACT_ROOT/four-gpu-30k.physical.dense-evidence-coverage.json
DONOR_COVERAGE=$CONTRACT_ROOT/four-gpu-30k-prefix-2k.physical.dense-evidence-coverage.json
DONOR_ROOT=/mnt/picf-next/adr150/caches/calvin-official-30k-prefix-2k-v1

for required_file in "$TARGET_MANIFEST" "$SOURCE_MANIFEST" "$SOURCE_RECEIPT" \
  "$ASSET_MANIFEST" "$COVERAGE" "$DONOR_COVERAGE"; do
  [[ -f "$required_file" && ! -L "$required_file" ]] || {
    echo "dense-cache partition input is not one regular non-symlink file: $required_file" >&2
    exit 1
  }
done
[[ "$(sha256sum "$COVERAGE" | awk '{print $1}')" == \
  8a0ef47470625cfdddfd3f7bf15020e693b8fa146d5d4307e35c157348448c6f ]] || {
  echo "official dense-evidence coverage identity mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$DONOR_COVERAGE" | awk '{print $1}')" == \
  836c60a04be2a1205b57550c1cb9a17170a3c1f0f02293a9aeee7e290e1b9e5f ]] || {
  echo "prefix dense-evidence coverage identity mismatch" >&2
  exit 1
}

COMMON=(
  --dataset-root "$DATASET_ROOT"
  --split training
  --target-dataset-manifest "$TARGET_MANIFEST"
  --donor-dataset-manifest "$TARGET_MANIFEST"
  --source-dataset-manifest "$SOURCE_MANIFEST"
  --source-receipt "$SOURCE_RECEIPT"
  --source-receipt-sha256 07d0d479315c07c45f868de58e232d6274063b5bdad801ebc3c436f9da05a75f
  --asset-manifest "$ASSET_MANIFEST"
  --coverage-plan "$COVERAGE"
  --coverage-plan-sha256 8a0ef47470625cfdddfd3f7bf15020e693b8fa146d5d4307e35c157348448c6f
  --donor-coverage-plan "$DONOR_COVERAGE"
  --donor-coverage-plan-sha256 836c60a04be2a1205b57550c1cb9a17170a3c1f0f02293a9aeee7e290e1b9e5f
  --token-dtype float16
  --shard-rows 64
  --action-horizon 1
  --modality "$MODALITY"
  --output-root "$OUTPUT_ROOT"
  --receipt-output "${OUTPUT_ROOT}.receipt.json"
  --record-start "$RECORD_START"
  --record-stop "$RECORD_STOP"
  --device cuda:0
)

case "$MODALITY" in
  anytouch)
    MODALITY_ARGS=(
      --donor-cache-root "$DONOR_ROOT/anytouch-observed-pose"
      --donor-cache-manifest-sha256 d340c1fbfc3834f04d583dca01f42faccaca3578f40d41ca1509eafb711709ae
      --encoder-batch-size 1
      --tactile-calibration-archive /mnt/picf-next-provenance/calvin-tactile-calibration-identity-a60b7934/tactile_backgrounds.npz
      --tactile-calibration-receipt /mnt/picf-next-provenance/calvin-tactile-calibration-identity-a60b7934/tactile_backgrounds.receipt.json
      --tactile-calibration-receipt-sha256 3bb381922df52c2cd561a2acb1824bc17c78dcacc2ca971727736c62c4baca65
    )
    ;;
  sonata)
    MODALITY_ARGS=(
      --donor-cache-root "$DONOR_ROOT/sonata-native"
      --donor-cache-manifest-sha256 18839d882da1cd9e814668a7dc3f27f44520aceb009c1da1d8a753a04d3c6d69
      --encoder-batch-size 1
      --camera-calibration /mnt/calvin_data/task_ABC_D/calib/cameras.json
      --point-pixel-stride 2
      --point-budget 4096
    )
    ;;
  vjepa)
    MODALITY_ARGS=(
      --donor-cache-root "$DONOR_ROOT/vjepa-final"
      --donor-cache-manifest-sha256 ae6055a906ccaacb133f3c9109ecffa558732309abb92dec18f39c9a39126f23
      --encoder-batch-size "$VJEPA_ENCODER_BATCH_SIZE"
    )
    ;;
  *)
    echo "unknown dense-cache modality: $MODALITY" >&2
    exit 2
    ;;
esac

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"
exec "$PYTHON" tools/republish_calvin_frozen_evidence_cache.py \
  "${COMMON[@]}" "${MODALITY_ARGS[@]}"
