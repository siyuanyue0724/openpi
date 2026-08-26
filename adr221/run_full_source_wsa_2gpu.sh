#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [LOAD_GLOBAL_STEP]" >&2
  exit 2
fi

RUN_DIR=$1
LOAD_GLOBAL_STEP=${2:-0}
REPO=${PICF_ADR221_REPO:-/mnt/picf-next/adr221/source-staging/full-source-wsa-v1}

[[ -d "$REPO" && ! -L "$REPO" ]] || {
  echo "ADR-221 source staging is absent: $REPO" >&2
  exit 1
}
if [[ "$LOAD_GLOBAL_STEP" == 0 ]]; then
  export PICF_PHASE=fresh
  export PICF_LOAD_GLOBAL_STEP=0
  [[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
    echo "fresh ADR-221 run directory must be absent beneath /mnt" >&2
    exit 2
  }
else
  [[ "$LOAD_GLOBAL_STEP" =~ ^[0-9]+$ ]] \
    && (( LOAD_GLOBAL_STEP > 0 && LOAD_GLOBAL_STEP % 2000 == 0 )) || {
    echo "ADR-221 resume requires a positive 2k checkpoint boundary" >&2
    exit 2
  }
  export PICF_PHASE=resume
  export PICF_LOAD_GLOBAL_STEP=$LOAD_GLOBAL_STEP
fi

export PICF_REPO=$REPO
export PICF_WORLD_SIZE=${PICF_WORLD_SIZE:-2}
export PICF_LINGBOT_NATIVE_SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr221-combined-offload-v1}
export PICF_CONTRACT_ROOT=${PICF_ADR221_CONTRACT_ROOT:-/mnt/picf-next/adr207/contracts/native-query-posterior-2gpu-30k-v1}
export PICF_ADR178_CACHE_ROOT=${PICF_ADR221_CACHE_ROOT:-/mnt/picf-next/adr207/caches/native-query-posterior-2gpu-30k-v1}
export PICF_USE_DENSE_SUPPLEMENT=0
export PICF_USE_CURRENT_GRID_CACHE=0

if [[ -n "${PICF_WSA_ARCHITECTURE_PROFILE:-}" ]]; then
  export PICF_ARCHITECTURE_PROFILE=$PICF_WSA_ARCHITECTURE_PROFILE
else
  export PICF_ARCHITECTURE_PROFILE=adr221_native_videomt_wsa_full_modal_v1
fi
export PICF_VIDEOMT_STAGE_PQ_MODE=trainable-adapted-native-query-causal-c5
export PICF_VIDEOMT_IDLE_PLACEMENT=cuda-resident
export PICF_VIDEOMT_FSDP2_PLACEMENT=${PICF_VIDEOMT_FSDP2_PLACEMENT:-cpu-offload}
export PICF_VIDEOMT_CHECKPOINT=${PICF_VIDEOMT_CHECKPOINT:-/mnt/picf-next/adr199/assets/videomt/yt_2019_dinov3_68.9.pth}
export PICF_VIDEOMT_DINOV3_BUNDLE=${PICF_VIDEOMT_DINOV3_BUNDLE:-/mnt/picf-next/adr199/assets/dinov3-vitl16-from-videomt}
export PICF_VIDEOMT_ADAPTED_CHECKPOINT=${PICF_VIDEOMT_ADAPTED_CHECKPOINT:-/mnt/picf-next/adr202/assets/videomt-calvin-adapted-step250-v1.pt}
export PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256=4437d8632c4e3877adcf5cfec5bf6e673445ad9d3d2de3a3afdd924651b5bd5d
VIDEOMT_RUNTIME_OVERLAY=/mnt/picf-next/adr207/python-overlays/videomt-torch280-functorch-v1
DA3_RUNTIME_OVERLAY=/mnt/picf-next/adr221/python-overlays/da3-api-moviepy-v1
DA3_RUNTIME_MANIFEST=/mnt/picf-next/adr221/python-overlays/da3-api-moviepy-v1.files.sha256
DA3_RUNTIME_MANIFEST_SHA256=54af863ae5d7c2bc0085fc38d9c09da669e6672b652a0be7dbe94b7d8a57afc2
[[ -d "$VIDEOMT_RUNTIME_OVERLAY" && ! -L "$VIDEOMT_RUNTIME_OVERLAY" ]] || {
  echo "VidEoMT runtime overlay is absent" >&2
  exit 1
}
[[ -d "$DA3_RUNTIME_OVERLAY" && ! -L "$DA3_RUNTIME_OVERLAY" ]] || {
  echo "DA3 runtime overlay is absent" >&2
  exit 1
}
[[ -f "$DA3_RUNTIME_MANIFEST" && ! -L "$DA3_RUNTIME_MANIFEST" ]] || {
  echo "DA3 runtime overlay manifest is absent" >&2
  exit 1
}
[[ "$(sha256sum "$DA3_RUNTIME_MANIFEST" | cut -d' ' -f1)" == "$DA3_RUNTIME_MANIFEST_SHA256" ]] || {
  echo "DA3 runtime overlay manifest identity changed" >&2
  exit 1
}
(cd / && sha256sum -c "$DA3_RUNTIME_MANIFEST" >/dev/null) || {
  echo "DA3 runtime overlay contents changed" >&2
  exit 1
}
export PICF_RUNTIME_PYTHON_OVERLAY=$DA3_RUNTIME_OVERLAY:${PICF_RUNTIME_PYTHON_OVERLAY:-$VIDEOMT_RUNTIME_OVERLAY}

export PICF_WSA_SOURCE_ROOT=${PICF_WSA_SOURCE_ROOT:-/mnt/picf-next/adr218/source/wsa-bfee742c585d5ee85722e658978111934c926ca3}
export PICF_WSA_ADAPTED_CHECKPOINT=${PICF_WSA_ADAPTED_CHECKPOINT:-/mnt/picf-next/adr218/adapted/wsa-future3d-36l-32h-432s.safetensors}
export PICF_WSA_ADAPTED_CHECKPOINT_SHA256=29d789d9a97459e33ed95aa85fb6e0ec0661879789db090bb8cabc1edf6a9130
export PICF_DA3_SOURCE_ROOT=${PICF_DA3_SOURCE_ROOT:-/mnt/picf-next/adr218/source/depth-anything-3}
export PICF_DA3_MODEL_DIR=${PICF_DA3_MODEL_DIR:-/mnt/picf-next/adr218/assets/da3-large-1.1}
export PICF_DA3_MODEL_CHECKPOINT_SHA256=739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64
export PICF_DA3_MODEL_CONFIG_SHA256=744dcaf53859490ed92fc6cb98d68d3daf624b8c54533aaf604bdb53f06321f5

export PICF_STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-2}
export PICF_ATTENTION_IMPLEMENTATION=flex_cached
export PICF_LINGBOT_COMPILE_MODE=upstream-default
export PICF_TRAINABLE_SCOPE=full-host
# Reuse ADR-209's exact execution-only FSDP2 placement. Vision remains
# trainable; only its idle parameter/gradient storage moves to CPU.
export PICF_FSDP2_PLACEMENT=${PICF_FSDP2_PLACEMENT:-selective-embedding-trainable-vision-offload}
export PICF_FSDP2_BACKWARD_PREFETCH=disabled
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

BASE=$REPO/adr178/run_direct_action_posterior_full_modal.sh
[[ -f "$BASE" && ! -L "$BASE" ]] || {
  echo "ADR-221 base launcher is absent" >&2
  exit 1
}
exec "$BASE" "$RUN_DIR"
