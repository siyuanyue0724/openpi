#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR=$1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPECTED_REPO=/mnt/picf-next/adr200/source-freezes/full-pqm-v1
EXPECTED_SOURCE=/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr199-prepared-frozen-visual-root-v1
if [[ -n ${PICF_REPO:-} && "$PICF_REPO" != "$EXPECTED_REPO" ]]; then
  echo "ADR-200 forbids a non-frozen PICF_REPO override: $PICF_REPO" >&2
  exit 2
fi
if [[ -n ${PICF_LINGBOT_NATIVE_SOURCE:-} && "$PICF_LINGBOT_NATIVE_SOURCE" != "$EXPECTED_SOURCE" ]]; then
  echo "ADR-200 forbids a different prepared LingBot checkout" >&2
  exit 2
fi

export PICF_ARCHITECTURE_PROFILE=adr193_implicit_multimodal_anchor_v1
export PICF_STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-250}
export PICF_REPO=$EXPECTED_REPO
export PICF_VIDEOMT_STAGE_PQ_MODE=frozen-released-eval-causal-c5-pqm
export PICF_VIDEOMT_IDLE_PLACEMENT=${PICF_VIDEOMT_IDLE_PLACEMENT:-cpu-between-forwards}
export PICF_FSDP2_PLACEMENT=selective-embedding-frozen-vision-offload
export PYTHONDONTWRITEBYTECODE=1
export PICF_RUNTIME_PYTHON_OVERLAY=${PICF_RUNTIME_PYTHON_OVERLAY:-/mnt/picf-next/adr199/python-overlays/videomt-runtime-c81d600f-v1}
export PYTHONPATH="$PICF_REPO/src:$PICF_REPO:$PICF_RUNTIME_PYTHON_OVERLAY"
: "${PICF_VIDEOMT_CHECKPOINT:?PICF_VIDEOMT_CHECKPOINT is required}"
: "${PICF_VIDEOMT_DINOV3_BUNDLE:?PICF_VIDEOMT_DINOV3_BUNDLE is required}"

[[ -d "$PICF_RUNTIME_PYTHON_OVERLAY" && ! -L "$PICF_RUNTIME_PYTHON_OVERLAY" ]] || {
  echo "required direct VidEoMT Python overlay is absent: $PICF_RUNTIME_PYTHON_OVERLAY" >&2
  exit 1
}

PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=$EXPECTED_SOURCE
LINGBOT_REPOSITORY=${PICF_LINGBOT_REPOSITORY:-/mnt/picf-next/adr199/official-source-234f6f0a-v1/lingbot-vla-v2}
export PICF_LINGBOT_NATIVE_SOURCE=$SOURCE
export PICF_LINGBOT_REPOSITORY=$LINGBOT_REPOSITORY
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
AUDIT_OUTPUT=${PICF_FULL_TRANSPLANT_AUDIT_OUTPUT:-${RUN_DIR}.full-transplant-audit.json}

[[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" ]] || {
  echo "RUN_DIR must be an absent persistent path beneath /mnt" >&2
  exit 2
}
[[ "$AUDIT_OUTPUT" == /mnt/* && ! -e "$AUDIT_OUTPUT" ]] || {
  echo "full-transplant audit output must be an absent persistent path beneath /mnt" >&2
  exit 2
}

"$PYTHON" "$PICF_REPO/tools/audit_full_transplant_contract.py" \
  --lingbot-repository "$LINGBOT_REPOSITORY" \
  --prepared-lingbot-checkout "$SOURCE" \
  --lingbot-checkpoint-dir "$CHECKPOINT" \
  --processor-dir "$PROCESSOR" \
  --videomt-checkpoint "$PICF_VIDEOMT_CHECKPOINT" \
  --dinov3-bundle "$PICF_VIDEOMT_DINOV3_BUNDLE" \
  --strict-runtime \
  --json-out "$AUDIT_OUTPUT"

exec "$SCRIPT_DIR/../adr193/run_implicit_multimodal_anchor_2gpu.sh" "$RUN_DIR"
