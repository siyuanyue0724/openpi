#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [LOAD_GLOBAL_STEP]" >&2
  exit 2
fi

RUN_DIR=$1
LOAD_GLOBAL_STEP=${2:-0}
TRAINING_PREFIX_STEPS=${PICF_ADR209_TRAINING_PREFIX_STEPS:-250}
REPO=${PICF_ADR209_REPO:-/mnt/picf-next/adr209/worktree-local-sync}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr209-native-muon-trainable-vision-flare-v5}
CONTRACT_ROOT=${PICF_ADR209_CONTRACT_ROOT:-/mnt/picf-next/adr209/contracts/flare-4gpu-prefix${TRAINING_PREFIX_STEPS}-v1}
CACHE_ROOT=${PICF_ADR209_CACHE_ROOT:-/mnt/picf-next/adr209/caches/flare-4gpu-prefix${TRAINING_PREFIX_STEPS}-v1}
ENABLE_FLARE=${PICF_ADR209_ENABLE_FLARE:-1}

for path in "$REPO" "$SOURCE" "$CONTRACT_ROOT" "$CACHE_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required direct ADR-209 directory is absent: $path" >&2
    exit 1
  }
done

if [[ "$LOAD_GLOBAL_STEP" == 0 ]]; then
  export PICF_PHASE=fresh
  export PICF_LOAD_GLOBAL_STEP=0
else
  export PICF_PHASE=resume
  export PICF_LOAD_GLOBAL_STEP=$LOAD_GLOBAL_STEP
fi

export PICF_REPO=$REPO
export PICF_WORLD_SIZE=4
export PICF_LINGBOT_NATIVE_SOURCE=$SOURCE
export PICF_CONTRACT_ROOT=$CONTRACT_ROOT
export PICF_ADR178_CACHE_ROOT=$CACHE_ROOT
export PICF_MINIMUM_FUTURE_SOURCE_FRAMES=16
case "$ENABLE_FLARE" in
  1)
    export PICF_ARCHITECTURE_PROFILE=adr209_native_videomt_flare_v1
    export PICF_FUTURE_LATENT_OBJECTIVE_SCALE=1
    ;;
  0)
    export PICF_ARCHITECTURE_PROFILE=adr209_native_videomt_query_control_t16_v1
    export PICF_FUTURE_LATENT_OBJECTIVE_SCALE=0
    ;;
  *)
    echo "PICF_ADR209_ENABLE_FLARE must be 0 or 1" >&2
    exit 2
    ;;
esac
export PICF_FUTURE_LATENT_CACHE_ROOT=$CACHE_ROOT/future-targets
export PICF_FUTURE_LATENT_CACHE_BUILD_REPORT=$CACHE_ROOT/future-targets.build-report.json
export PICF_USE_DENSE_SUPPLEMENT=0
export PICF_USE_CURRENT_GRID_CACHE=0

export PICF_VIDEOMT_STAGE_PQ_MODE=trainable-adapted-native-query-causal-c5
export PICF_VIDEOMT_IDLE_PLACEMENT=cuda-resident
export PICF_VIDEOMT_FSDP2_PLACEMENT=${PICF_VIDEOMT_FSDP2_PLACEMENT:-cpu-offload}
export PICF_VIDEOMT_CHECKPOINT=${PICF_VIDEOMT_CHECKPOINT:-/mnt/picf-next/adr199/assets/videomt/yt_2019_dinov3_68.9.pth}
export PICF_VIDEOMT_DINOV3_BUNDLE=${PICF_VIDEOMT_DINOV3_BUNDLE:-/mnt/picf-next/adr199/assets/dinov3-vitl16-from-videomt}
export PICF_VIDEOMT_ADAPTED_CHECKPOINT=${PICF_VIDEOMT_ADAPTED_CHECKPOINT:-/mnt/picf-next/adr202/assets/videomt-calvin-adapted-step250-v1.pt}
export PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256=4437d8632c4e3877adcf5cfec5bf6e673445ad9d3d2de3a3afdd924651b5bd5d
export PICF_RUNTIME_PYTHON_OVERLAY=${PICF_RUNTIME_PYTHON_OVERLAY:-/mnt/picf-next/adr207/python-overlays/videomt-torch280-functorch-v1}

export PICF_STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-$TRAINING_PREFIX_STEPS}
export PICF_ATTENTION_IMPLEMENTATION=flex_cached
export PICF_LINGBOT_COMPILE_MODE=upstream-default
export PICF_TRAINABLE_SCOPE=full-host
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

BASE=$REPO/adr178/run_direct_action_posterior_full_modal.sh
[[ -f "$BASE" && ! -L "$BASE" ]] || {
  echo "ADR-209 base launcher is absent" >&2
  exit 1
}
exec "$BASE" "$RUN_DIR"
