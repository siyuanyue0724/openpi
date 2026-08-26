#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [LOAD_GLOBAL_STEP]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
RUN_DIR=$1
LOAD_GLOBAL_STEP=${2:-0}

export PICF_REPO=$REPO
export PICF_WORLD_SIZE=4
export PICF_PHASE=$([[ "$LOAD_GLOBAL_STEP" == 0 ]] && echo fresh || echo resume)
export PICF_LOAD_GLOBAL_STEP=$LOAD_GLOBAL_STEP
export PICF_LINGBOT_NATIVE_SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr224-wla-native-muon-vlm-class-offload-v1}
export PICF_CONTRACT_ROOT=${PICF_ADR224_CONTRACT_ROOT:-/mnt/picf-next/adr224/contracts/lingbot-wla-4gpu-30k-v2-source-h8}
export PICF_ADR178_CACHE_ROOT=${PICF_ADR224_CACHE_ROOT:-/mnt/picf-next/adr207/caches/native-query-posterior-4gpu-30k-v1}
export PICF_USE_DENSE_SUPPLEMENT=0
export PICF_DENSE_EVIDENCE_SUBSET_VIEW=1
export PICF_USE_CURRENT_GRID_CACHE=0

export PICF_ARCHITECTURE_PROFILE=adr207_native_videomt_query_posterior_v1
export PICF_VIDEOMT_STAGE_PQ_MODE=trainable-adapted-native-query-causal-c5
export PICF_VIDEOMT_IDLE_PLACEMENT=cuda-resident
export PICF_VIDEOMT_FSDP2_PLACEMENT=${PICF_VIDEOMT_FSDP2_PLACEMENT:-cpu-offload}
export PICF_VIDEOMT_CHECKPOINT=${PICF_VIDEOMT_CHECKPOINT:-/mnt/picf-next/adr199/assets/videomt/yt_2019_dinov3_68.9.pth}
export PICF_VIDEOMT_DINOV3_BUNDLE=${PICF_VIDEOMT_DINOV3_BUNDLE:-/mnt/picf-next/adr199/assets/dinov3-vitl16-from-videomt}
export PICF_VIDEOMT_ADAPTED_CHECKPOINT=${PICF_VIDEOMT_ADAPTED_CHECKPOINT:-/mnt/picf-next/adr202/assets/videomt-calvin-adapted-step250-v1.pt}
export PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256=4437d8632c4e3877adcf5cfec5bf6e673445ad9d3d2de3a3afdd924651b5bd5d

export PICF_RUNTIME_PYTHON_OVERLAY=/opt/picf-adr224-wla-venv/lib/python3.12/site-packages:/mnt/picf-next/adr207/python-overlays/videomt-torch280-functorch-v1
export PICF_ACTION_BACKEND=wla_complete
export PICF_WLA_HOST_EVIDENCE_ARM=${PICF_WLA_HOST_EVIDENCE_ARM:-picf_full}
export PICF_WLA_SOURCE_ROOT=/mnt/picf-next/adr224/upstreams/WLA-155ac94e-immutable
export PICF_WLA_PRETRAINED_ROOT=/mnt/picf-next/adr224/models/Sana_600M_512px_diffusers_64channels
export PICF_LEARNING_RATE=5e-5
export PICF_MINIMUM_FUTURE_SOURCE_FRAMES=8

export PICF_STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-30000}
export PICF_ATTENTION_IMPLEMENTATION=flex_cached
export PICF_LINGBOT_COMPILE_MODE=disabled
export PICF_TRAINABLE_SCOPE=full-host
export PICF_FSDP2_PLACEMENT=${PICF_FSDP2_PLACEMENT:-selective-embedding-offload}
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

exec "$REPO/adr178/run_direct_action_posterior_full_modal.sh" "$RUN_DIR"
