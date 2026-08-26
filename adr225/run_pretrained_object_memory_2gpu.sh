#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [LOAD_GLOBAL_STEP]" >&2
  exit 2
fi

RUN_DIR=$1
LOAD_GLOBAL_STEP=${2:-0}
WORLD_SIZE=${PICF_WORLD_SIZE:-2}
case "$WORLD_SIZE" in
  2|4) ;;
  *)
    echo "PICF_WORLD_SIZE must be 2 or 4" >&2
    exit 2
    ;;
esac
if [[ "$LOAD_GLOBAL_STEP" == 0 ]]; then
  export PICF_PHASE=fresh
  export PICF_LOAD_GLOBAL_STEP=0
  [[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
    echo "fresh ADR-225 run directory must be absent beneath /mnt" >&2
    exit 2
  }
else
  [[ "$LOAD_GLOBAL_STEP" =~ ^[0-9]+$ ]] \
    && (( LOAD_GLOBAL_STEP > 0 && LOAD_GLOBAL_STEP % 2000 == 0 )) || {
    echo "ADR-225 resume requires a positive 2k checkpoint boundary" >&2
    exit 2
  }
  export PICF_PHASE=resume
  export PICF_LOAD_GLOBAL_STEP=$LOAD_GLOBAL_STEP
  [[ "$RUN_DIR" == /mnt/* && -d "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
    echo "resume ADR-225 run directory must be an existing direct path beneath /mnt" >&2
    exit 2
  }
fi

EXPECTED_REPO=${PICF_ADR225_REPO:-/mnt/picf-next/adr225/source-freezes/pretrained-object-memory-v2}
[[ -d "$EXPECTED_REPO" && ! -L "$EXPECTED_REPO" ]] || {
  echo "ADR-225 immutable source freeze is absent: $EXPECTED_REPO" >&2
  exit 1
}
[[ -f "$EXPECTED_REPO/source-freeze.receipt.json" \
  && ! -L "$EXPECTED_REPO/source-freeze.receipt.json" ]] || {
  echo "ADR-225 immutable source freeze receipt is absent" >&2
  exit 1
}
[[ -z "$(git -C "$EXPECTED_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-225 candidate requires the exact clean source freeze" >&2
  exit 1
}

export PICF_REPO=$EXPECTED_REPO
export PICF_WORLD_SIZE=$WORLD_SIZE
export PICF_LINGBOT_NATIVE_SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr224-wla-native-muon-vlm-class-offload-v1}

# Reuse ADR-207's exact stream/evaluation split and dense evidence so the action
# curves are paired. Only the declared host integration differs.
export PICF_CONTRACT_ROOT=${PICF_ADR225_CONTRACT_ROOT:-/mnt/picf-next/adr207/contracts/native-query-posterior-${WORLD_SIZE}gpu-30k-v1}
export PICF_ADR178_CACHE_ROOT=${PICF_ADR225_CACHE_ROOT:-/mnt/picf-next/adr207/caches/native-query-posterior-${WORLD_SIZE}gpu-30k-v1}
export PICF_USE_DENSE_SUPPLEMENT=0
export PICF_USE_CURRENT_GRID_CACHE=0

export PICF_ARCHITECTURE_PROFILE=adr225_pretrained_native_object_memory_v1
export PICF_VIDEOMT_STAGE_PQ_MODE=trainable-adapted-native-query-causal-c5
export PICF_VIDEOMT_IDLE_PLACEMENT=cuda-resident
export PICF_VIDEOMT_FSDP2_PLACEMENT=${PICF_VIDEOMT_FSDP2_PLACEMENT:-cpu-offload}
export PICF_VIDEOMT_CHECKPOINT=${PICF_VIDEOMT_CHECKPOINT:-/mnt/picf-next/adr199/assets/videomt/yt_2019_dinov3_68.9.pth}
export PICF_VIDEOMT_DINOV3_BUNDLE=${PICF_VIDEOMT_DINOV3_BUNDLE:-/mnt/picf-next/adr199/assets/dinov3-vitl16-from-videomt}
export PICF_VIDEOMT_ADAPTED_CHECKPOINT=${PICF_VIDEOMT_ADAPTED_CHECKPOINT:-/mnt/picf-next/adr202/assets/videomt-calvin-adapted-step250-v1.pt}
export PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256=4437d8632c4e3877adcf5cfec5bf6e673445ad9d3d2de3a3afdd924651b5bd5d
export PICF_RUNTIME_PYTHON_OVERLAY=${PICF_RUNTIME_PYTHON_OVERLAY:-/mnt/picf-next/adr207/python-overlays/videomt-torch280-functorch-v1}
RUNTIME_OVERLAY_RECEIPT=$PICF_RUNTIME_PYTHON_OVERLAY/overlay-receipt.json
RUNTIME_OVERLAY_RECEIPT_SHA256=857d364103403df8aafc97674e97e518acf781bc8fc080840ca11c99f25aacd0
[[ -f "$RUNTIME_OVERLAY_RECEIPT" && ! -L "$RUNTIME_OVERLAY_RECEIPT" ]] || {
  echo "ADR-225 runtime overlay receipt is absent" >&2
  exit 1
}
[[ "$(sha256sum "$RUNTIME_OVERLAY_RECEIPT" | cut -d' ' -f1)" \
  == "$RUNTIME_OVERLAY_RECEIPT_SHA256" ]] || {
  echo "ADR-225 runtime overlay receipt changed" >&2
  exit 1
}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PICF_RUNTIME_PYTHON_OVERLAY" "$PYTHON" -c \
  'import functorch, torch; assert torch.__version__ == functorch.__version__ == "2.8.0+cu128"'

STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-250}
LONG_RUN_AUTHORIZED=${PICF_ADR225_LONG_RUN_AUTHORIZED:-0}
[[ "$STOP_AFTER_STEP" =~ ^[0-9]+$ ]] && (( STOP_AFTER_STEP <= 30000 )) || {
  echo "PICF_STOP_AFTER_STEP must be an integer no greater than 30000" >&2
  exit 2
}
case "$LONG_RUN_AUTHORIZED" in 0|1) ;; *)
  echo "PICF_ADR225_LONG_RUN_AUTHORIZED must be 0 or 1" >&2
  exit 2
esac
if (( STOP_AFTER_STEP > 2000 )) && [[ "$LONG_RUN_AUTHORIZED" != 1 ]]; then
  echo "ADR-225 runs beyond 2000 steps require an explicit Gate-D authorization" >&2
  exit 2
fi
export PICF_STOP_AFTER_STEP=$STOP_AFTER_STEP
export PICF_ATTENTION_IMPLEMENTATION=flex_cached
# ADR-225 captures the exact native visual merger input through audited Python
# hooks; torch.compile is fail-closed until a hook-preservation probe passes.
export PICF_LINGBOT_COMPILE_MODE=disabled
export PICF_TRAINABLE_SCOPE=full-host
# Reuse ADR-224's exact eight-overlay FSDP2 execution path. It changes only
# parameter/gradient/optimizer storage placement; model topology and objectives
# remain byte-identical to the ADR-225 scientific arm.
export PICF_FSDP2_PLACEMENT=${PICF_FSDP2_PLACEMENT:-selective-embedding-trainable-vision-offload}
export PICF_FSDP2_BACKWARD_PREFETCH=${PICF_FSDP2_BACKWARD_PREFETCH:-disabled}
export PICF_SEQUENTIAL_FACTUAL_GRADIENT_STORAGE=gpu
export PICF_CUDA_ALLOCATOR=${PICF_CUDA_ALLOCATOR:-expandable-segments}

export PICF_DENSE_TOKEN_BRIDGE=exact_tokens_v1
export PICF_CAPACITY=200
export PICF_RELATION_SUPERVISION_LAYERS=
export PICF_PICF_LEARNING_RATE_MULTIPLIER=1.0
export PICF_MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER=1.0
export PICF_ENTITY_WEIGHT=0.0
export PICF_PREDICTIVE_WEIGHT=0.0
export PICF_LOCAL_BPTT_PROBABILITY=0.0
export PICF_OVERSHOOT_PROBABILITY=0.0
export PICF_SOURCE_MASK_PROBABILITY=0.0
export PICF_OMITTED_STATIC_REMATERIALIZATION=none

BASE=$EXPECTED_REPO/adr178/run_direct_action_posterior_full_modal.sh
[[ -f "$BASE" && ! -L "$BASE" ]] || {
  echo "ADR-225 base launcher is absent from its immutable source freeze" >&2
  exit 1
}
exec "$BASE" "$RUN_DIR"
