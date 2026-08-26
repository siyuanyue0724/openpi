#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPECTED_REPO=/mnt/picf-next/adr204/source-freezes/full-source-row-refinement-v3
EXPECTED_SOURCE=/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr199-prepared-frozen-visual-root-v1

if [[ -n ${PICF_REPO:-} && "$PICF_REPO" != "$EXPECTED_REPO" ]]; then
  echo "ADR-204 forbids a non-frozen PICF_REPO override: $PICF_REPO" >&2
  exit 2
fi
if [[ -n ${PICF_LINGBOT_NATIVE_SOURCE:-} && "$PICF_LINGBOT_NATIVE_SOURCE" != "$EXPECTED_SOURCE" ]]; then
  echo "ADR-204 forbids a different prepared LingBot checkout" >&2
  exit 2
fi

export PICF_ARCHITECTURE_PROFILE=adr193_implicit_multimodal_anchor_v1
export PICF_STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-250}
export PICF_REPO=$EXPECTED_REPO
export PICF_LINGBOT_NATIVE_SOURCE=$EXPECTED_SOURCE
export PICF_VIDEOMT_STAGE_PQ_MODE=frozen-adapted-eval-causal-c5-pqrf
export PICF_VIDEOMT_IDLE_PLACEMENT=${PICF_VIDEOMT_IDLE_PLACEMENT:-cpu-between-forwards}
export PICF_FSDP2_PLACEMENT=selective-embedding-frozen-vision-offload
export PICF_RUNTIME_PYTHON_OVERLAY=${PICF_RUNTIME_PYTHON_OVERLAY:-/mnt/picf-next/adr199/python-overlays/videomt-runtime-c81d600f-v1}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PICF_REPO/src:$PICF_REPO:$PICF_RUNTIME_PYTHON_OVERLAY"

: "${PICF_VIDEOMT_CHECKPOINT:?PICF_VIDEOMT_CHECKPOINT is required}"
: "${PICF_VIDEOMT_DINOV3_BUNDLE:?PICF_VIDEOMT_DINOV3_BUNDLE is required}"
: "${PICF_VIDEOMT_ADAPTED_CHECKPOINT:?PICF_VIDEOMT_ADAPTED_CHECKPOINT is required}"
: "${PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256:?PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256 is required}"

[[ -d "$PICF_RUNTIME_PYTHON_OVERLAY" && ! -L "$PICF_RUNTIME_PYTHON_OVERLAY" ]] || {
  echo "required direct VidEoMT Python overlay is absent: $PICF_RUNTIME_PYTHON_OVERLAY" >&2
  exit 1
}

exec "$SCRIPT_DIR/../adr193/run_implicit_multimodal_anchor_2gpu.sh" "$1"
