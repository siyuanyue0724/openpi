#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR=$1
WORLD_SIZE=${PICF_WORLD_SIZE:-2}
case "$WORLD_SIZE" in
  2)
    CUDA_VISIBLE_DEVICES_VALUE=0,1
    ;;
  4)
    CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3
    ;;
  *)
    echo "PICF_WORLD_SIZE must be 2 or 4" >&2
    exit 2
    ;;
esac
PHASE=${PICF_PHASE:-fresh}
LOAD_GLOBAL_STEP=${PICF_LOAD_GLOBAL_STEP:-0}
case "$PHASE" in
  fresh)
    [[ "$LOAD_GLOBAL_STEP" == 0 ]] || {
      echo "fresh training requires PICF_LOAD_GLOBAL_STEP=0" >&2
      exit 2
    }
    [[ "$RUN_DIR" == /mnt/* && ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
      echo "fresh RUN_DIR must be an absent persistent path beneath /mnt" >&2
      exit 2
    }
    ;;
  resume)
    [[ "$LOAD_GLOBAL_STEP" =~ ^[0-9]+$ ]] \
      && (( LOAD_GLOBAL_STEP > 0 && LOAD_GLOBAL_STEP % 2000 == 0 )) || {
      echo "resume requires a positive 2k PICF_LOAD_GLOBAL_STEP boundary" >&2
      exit 2
    }
    [[ "$RUN_DIR" == /mnt/* && -d "$RUN_DIR" && ! -L "$RUN_DIR" ]] || {
      echo "resume RUN_DIR must be an existing direct path beneath /mnt" >&2
      exit 2
    }
    [[ -d "$RUN_DIR/checkpoints/global_step_$LOAD_GLOBAL_STEP" \
      && ! -L "$RUN_DIR/checkpoints/global_step_$LOAD_GLOBAL_STEP" ]] || {
      echo "resume checkpoint boundary is absent" >&2
      exit 1
    }
    ;;
  *)
    echo "PICF_PHASE must be fresh or resume" >&2
    exit 2
    ;;
esac

REPO=${PICF_REPO:-/mnt/picf-next/adr178/source-freezes/adr178-direct-action-posterior-v1}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr176-native}
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json}
SIDECAR_ROOT=${PICF_CALVIN_PHYSICAL_SIDECAR:-/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z}
SIDECAR_MANIFEST=${PICF_CALVIN_PHYSICAL_SIDECAR_MANIFEST:-/mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json}
VISUAL_ACCEPTANCE=${PICF_CALVIN_VISUAL_ACCEPTANCE:-/mnt/picf-next-provenance/calvin-physical-visual-review-identity-a60b7934/calvin-physical-visual-acceptance.json}
CONTRACT_ROOT=${PICF_CONTRACT_ROOT:-${PICF_ADR178_CONTRACT_ROOT:-/mnt/picf-next/adr177/contracts/full-modal-2gpu-30k-v1}}
STREAM_PLAN=$CONTRACT_ROOT/stream-plan.json
REPRESENTATION_SPLIT=$CONTRACT_ROOT/representation-split.json
EVALUATION_PLAN=$CONTRACT_ROOT/evaluation-plan.json
DENSE_COVERAGE=$CONTRACT_ROOT/dense-evidence-coverage.json
DENSE_CACHE_ROOT=${PICF_ADR178_CACHE_ROOT:-/mnt/picf-next/adr150/caches/calvin-official-30k-v1}
DENSE_SUPPLEMENT_ROOT=${PICF_ADR178_SUPPLEMENT_CACHE_ROOT:-/mnt/picf-next/adr176/caches/full-modal-2gpu-prefix1500-v1}
CURRENT_CACHE_ROOT=${PICF_ADR178_CURRENT_CACHE_ROOT:-/mnt/picf-next/adr176/caches/full-modal-2gpu-prefix1500-v1}
CURRENT_CACHE=$CURRENT_CACHE_ROOT/current-grid
CURRENT_SHARD_ROOT=${PICF_CURRENT_GRID_SHARD_ROOT:-/mnt/picf-next/adr150/caches/calvin-official-30k-v1/current-filter-dino-physical-v1}
CURRENT_REPORT=${CURRENT_CACHE}.build_report.json
FUTURE_LATENT_CACHE_ROOT=${PICF_FUTURE_LATENT_CACHE_ROOT:-}
FUTURE_LATENT_CACHE_BUILD_REPORT=${PICF_FUTURE_LATENT_CACHE_BUILD_REPORT:-}
FUTURE_LATENT_OBJECTIVE_SCALE=${PICF_FUTURE_LATENT_OBJECTIVE_SCALE:-1}
MINIMUM_FUTURE_SOURCE_FRAMES=${PICF_MINIMUM_FUTURE_SOURCE_FRAMES:-}
ANYTOUCH_CACHE=$DENSE_CACHE_ROOT/anytouch-observed-pose
SONATA_CACHE=$DENSE_CACHE_ROOT/sonata-native
VJEPA_CACHE=$DENSE_CACHE_ROOT/vjepa-final
ANYTOUCH_SUPPLEMENT=$DENSE_SUPPLEMENT_ROOT/anytouch
SONATA_SUPPLEMENT=$DENSE_SUPPLEMENT_ROOT/sonata
VJEPA_SUPPLEMENT=$DENSE_SUPPLEMENT_ROOT/vjepa
USE_DENSE_SUPPLEMENT=${PICF_USE_DENSE_SUPPLEMENT:-1}
DENSE_EVIDENCE_SUBSET_VIEW=${PICF_DENSE_EVIDENCE_SUBSET_VIEW:-0}
USE_CURRENT_GRID_CACHE=${PICF_USE_CURRENT_GRID_CACHE:-1}
STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-250}
BACKWARD_PREFETCH=${PICF_FSDP2_BACKWARD_PREFETCH:-disabled}
FSDP2_PLACEMENT=${PICF_FSDP2_PLACEMENT:-selective-embedding-offload}
FACTUAL_GRADIENT_STORAGE=${PICF_SEQUENTIAL_FACTUAL_GRADIENT_STORAGE:-cpu}
TRAINABLE_SCOPE=${PICF_TRAINABLE_SCOPE:-frozen-vision-host}
CUDA_ALLOCATOR=${PICF_CUDA_ALLOCATOR:-expandable-segments}
ATTENTION_IMPLEMENTATION=${PICF_ATTENTION_IMPLEMENTATION:-eager}
LINGBOT_COMPILE_MODE=${PICF_LINGBOT_COMPILE_MODE:-disabled}
ARCHITECTURE_PROFILE=${PICF_ARCHITECTURE_PROFILE:-adr178_direct_action_posterior_full_modal_v1}
ADR221_WSA_EDGE_DIAGNOSTIC=${PICF_ADR221_WSA_EDGE_DIAGNOSTIC:-0}
VIDEOMT_STAGE_PQ_MODE=${PICF_VIDEOMT_STAGE_PQ_MODE:-disabled}
VIDEOMT_IDLE_PLACEMENT=${PICF_VIDEOMT_IDLE_PLACEMENT:-cuda-resident}
VIDEOMT_FSDP2_PLACEMENT=${PICF_VIDEOMT_FSDP2_PLACEMENT:-cuda-sharded}
RUNTIME_PYTHON_OVERLAY=${PICF_RUNTIME_PYTHON_OVERLAY:-}
WSA_SOURCE_ROOT=${PICF_WSA_SOURCE_ROOT:-}
WSA_ADAPTED_CHECKPOINT=${PICF_WSA_ADAPTED_CHECKPOINT:-}
WSA_ADAPTED_CHECKPOINT_SHA256=${PICF_WSA_ADAPTED_CHECKPOINT_SHA256:-}
DA3_SOURCE_ROOT=${PICF_DA3_SOURCE_ROOT:-}
DA3_MODEL_DIR=${PICF_DA3_MODEL_DIR:-}
DA3_MODEL_CHECKPOINT_SHA256=${PICF_DA3_MODEL_CHECKPOINT_SHA256:-}
DA3_MODEL_CONFIG_SHA256=${PICF_DA3_MODEL_CONFIG_SHA256:-}
ACTION_BACKEND=${PICF_ACTION_BACKEND:-lingbot_released}
WLA_HOST_EVIDENCE_ARM=${PICF_WLA_HOST_EVIDENCE_ARM:-picf_full}
WLA_SOURCE_ROOT=${PICF_WLA_SOURCE_ROOT:-}
WLA_PRETRAINED_ROOT=${PICF_WLA_PRETRAINED_ROOT:-}
VIDEOMT_SHARED_QUERY_GRADIENT_DIAGNOSTIC=${PICF_VIDEOMT_SHARED_QUERY_GRADIENT_DIAGNOSTIC:-0}
VIDEOMT_SOURCE_UPDATE_ARM=${PICF_VIDEOMT_SOURCE_UPDATE_ARM:-joint}
LEARNING_RATE=${PICF_LEARNING_RATE:-1e-4}
RELATION_SUPERVISION_LAYERS=${PICF_RELATION_SUPERVISION_LAYERS-8,17,26}
DENSE_TOKEN_BRIDGE=${PICF_DENSE_TOKEN_BRIDGE:-lingbot_task_token_resampler_v1}
CAPACITY=${PICF_CAPACITY:-16}
PICF_LEARNING_RATE_MULTIPLIER=${PICF_PICF_LEARNING_RATE_MULTIPLIER:-2.0}
MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER=${PICF_MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER:-0.5}
ENTITY_WEIGHT=${PICF_ENTITY_WEIGHT:-0.08}
PREDICTIVE_WEIGHT=${PICF_PREDICTIVE_WEIGHT:-0.004}
LOCAL_BPTT_PROBABILITY=${PICF_LOCAL_BPTT_PROBABILITY:-0.0}
OVERSHOOT_PROBABILITY=${PICF_OVERSHOOT_PROBABILITY:-0.0}
SOURCE_MASK_PROBABILITY=${PICF_SOURCE_MASK_PROBABILITY:-0.10}
OMITTED_STATIC_REMATERIALIZATION=${PICF_OMITTED_STATIC_REMATERIALIZATION:-sequential-backward}

case "$USE_DENSE_SUPPLEMENT" in 0|1) ;; *)
  echo "PICF_USE_DENSE_SUPPLEMENT must be 0 or 1" >&2
  exit 2
esac
case "$DENSE_EVIDENCE_SUBSET_VIEW" in 0|1) ;; *)
  echo "PICF_DENSE_EVIDENCE_SUBSET_VIEW must be 0 or 1" >&2
  exit 2
esac
case "$USE_CURRENT_GRID_CACHE" in 0|1) ;; *)
  echo "PICF_USE_CURRENT_GRID_CACHE must be 0 or 1" >&2
  exit 2
esac

case "$FSDP2_PLACEMENT" in
  cpu-offload|selective-embedding-offload|selective-embedding-frozen-vision-offload|selective-embedding-trainable-vision-offload) ;;
  *)
    echo "unsupported PICF_FSDP2_PLACEMENT for ADR-178/193: $FSDP2_PLACEMENT" >&2
    exit 2
    ;;
esac

case "$ATTENTION_IMPLEMENTATION" in
  eager|flex_cached) ;;
  *)
    echo "unsupported PICF_ATTENTION_IMPLEMENTATION: $ATTENTION_IMPLEMENTATION" >&2
    exit 2
    ;;
esac

case "$LINGBOT_COMPILE_MODE" in
  disabled|upstream-default) ;;
  *)
    echo "unsupported PICF_LINGBOT_COMPILE_MODE: $LINGBOT_COMPILE_MODE" >&2
    exit 2
    ;;
esac

case "$ADR221_WSA_EDGE_DIAGNOSTIC" in
  0)
    WSA_EDGE_DIAGNOSTIC_ARGS=()
    MINIMUM_STOP_AFTER_STEP=2
    ;;
  1)
    WSA_EDGE_DIAGNOSTIC_ARGS=(--adr221-wsa-edge-diagnostic)
    MINIMUM_STOP_AFTER_STEP=1
    ;;
  *)
    echo "PICF_ADR221_WSA_EDGE_DIAGNOSTIC must be 0 or 1" >&2
    exit 2
    ;;
esac

case "$FUTURE_LATENT_OBJECTIVE_SCALE" in
  0|1) ;;
  *)
    echo "PICF_FUTURE_LATENT_OBJECTIVE_SCALE must be 0 or 1" >&2
    exit 2
    ;;
esac

case "$VIDEOMT_IDLE_PLACEMENT" in
  cuda-resident|cpu-between-forwards) ;;
  *)
    echo "unknown PICF_VIDEOMT_IDLE_PLACEMENT: $VIDEOMT_IDLE_PLACEMENT" >&2
    exit 2
    ;;
esac

case "$VIDEOMT_FSDP2_PLACEMENT" in
  cuda-sharded|cpu-offload) ;;
  *)
    echo "unknown PICF_VIDEOMT_FSDP2_PLACEMENT: $VIDEOMT_FSDP2_PLACEMENT" >&2
    exit 2
    ;;
esac
if [[ "$VIDEOMT_STAGE_PQ_MODE" != trainable-adapted-native-query-causal-c5 \
      && "$VIDEOMT_FSDP2_PLACEMENT" != cuda-sharded ]]; then
  echo "only trainable native VidEoMT supports a nondefault FSDP2 placement" >&2
  exit 2
fi

case "$VIDEOMT_STAGE_PQ_MODE" in
  disabled)
    [[ "$VIDEOMT_IDLE_PLACEMENT" == "cuda-resident" ]] || {
      echo "disabled VidEoMT Stage-PQ requires cuda-resident idle placement" >&2
      exit 2
    }
    VIDEOMT_STAGE_PQ_ARGS=(
      --videomt-stage-pq-mode disabled
      --videomt-idle-placement cuda-resident
      --videomt-fsdp2-placement cuda-sharded
    )
    ;;
  frozen-released-eval-causal-c5|frozen-released-eval-causal-c5-pqm)
    : "${PICF_VIDEOMT_CHECKPOINT:?PICF_VIDEOMT_CHECKPOINT is required}"
    : "${PICF_VIDEOMT_DINOV3_BUNDLE:?PICF_VIDEOMT_DINOV3_BUNDLE is required}"
    [[ -f "$PICF_VIDEOMT_CHECKPOINT" && ! -L "$PICF_VIDEOMT_CHECKPOINT" ]] || {
      echo "required direct VidEoMT checkpoint is absent: $PICF_VIDEOMT_CHECKPOINT" >&2
      exit 1
    }
    [[ -d "$PICF_VIDEOMT_DINOV3_BUNDLE" && ! -L "$PICF_VIDEOMT_DINOV3_BUNDLE" ]] || {
      echo "required direct DINOv3 bundle is absent: $PICF_VIDEOMT_DINOV3_BUNDLE" >&2
      exit 1
    }
    VIDEOMT_STAGE_PQ_ARGS=(
      --videomt-stage-pq-mode "$VIDEOMT_STAGE_PQ_MODE"
      --videomt-checkpoint "$PICF_VIDEOMT_CHECKPOINT"
      --videomt-dinov3-bundle "$PICF_VIDEOMT_DINOV3_BUNDLE"
      --videomt-idle-placement "$VIDEOMT_IDLE_PLACEMENT"
      --videomt-fsdp2-placement "$VIDEOMT_FSDP2_PLACEMENT"
    )
    ;;
  frozen-adapted-eval-causal-c5-pqm|frozen-adapted-eval-causal-c5-pqmr|frozen-adapted-eval-causal-c5-pqrf|trainable-adapted-native-query-causal-c5)
    : "${PICF_VIDEOMT_CHECKPOINT:?PICF_VIDEOMT_CHECKPOINT is required}"
    : "${PICF_VIDEOMT_DINOV3_BUNDLE:?PICF_VIDEOMT_DINOV3_BUNDLE is required}"
    : "${PICF_VIDEOMT_ADAPTED_CHECKPOINT:?PICF_VIDEOMT_ADAPTED_CHECKPOINT is required}"
    : "${PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256:?PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256 is required}"
    [[ -f "$PICF_VIDEOMT_CHECKPOINT" && ! -L "$PICF_VIDEOMT_CHECKPOINT" ]] || {
      echo "required direct VidEoMT checkpoint is absent: $PICF_VIDEOMT_CHECKPOINT" >&2
      exit 1
    }
    [[ -d "$PICF_VIDEOMT_DINOV3_BUNDLE" && ! -L "$PICF_VIDEOMT_DINOV3_BUNDLE" ]] || {
      echo "required direct DINOv3 bundle is absent: $PICF_VIDEOMT_DINOV3_BUNDLE" >&2
      exit 1
    }
    [[ -f "$PICF_VIDEOMT_ADAPTED_CHECKPOINT" && ! -L "$PICF_VIDEOMT_ADAPTED_CHECKPOINT" ]] || {
      echo "required adapted VidEoMT checkpoint is absent: $PICF_VIDEOMT_ADAPTED_CHECKPOINT" >&2
      exit 1
    }
    [[ "$PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
      echo "PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256 must be a lowercase SHA-256" >&2
      exit 2
    }
    VIDEOMT_STAGE_PQ_ARGS=(
      --videomt-stage-pq-mode "$VIDEOMT_STAGE_PQ_MODE"
      --videomt-checkpoint "$PICF_VIDEOMT_CHECKPOINT"
      --videomt-dinov3-bundle "$PICF_VIDEOMT_DINOV3_BUNDLE"
      --videomt-adapted-checkpoint "$PICF_VIDEOMT_ADAPTED_CHECKPOINT"
      --videomt-adapted-checkpoint-sha256 "$PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256"
      --videomt-idle-placement "$VIDEOMT_IDLE_PLACEMENT"
      --videomt-fsdp2-placement "$VIDEOMT_FSDP2_PLACEMENT"
    )
    ;;
  *)
    echo "unknown PICF_VIDEOMT_STAGE_PQ_MODE: $VIDEOMT_STAGE_PQ_MODE" >&2
    exit 2
    ;;
esac

if [[ -n "$RUNTIME_PYTHON_OVERLAY" ]]; then
  case ":$RUNTIME_PYTHON_OVERLAY:" in
    *::*)
      echo "runtime Python overlay list contains an empty path" >&2
      exit 2
      ;;
  esac
  IFS=':' read -r -a RUNTIME_PYTHON_OVERLAYS <<< "$RUNTIME_PYTHON_OVERLAY"
  for overlay in "${RUNTIME_PYTHON_OVERLAYS[@]}"; do
    [[ -d "$overlay" && ! -L "$overlay" ]] || {
      echo "required direct runtime Python overlay is absent: $overlay" >&2
      exit 1
    }
  done
fi

[[ "$STOP_AFTER_STEP" =~ ^[0-9]+$ ]] \
  && (( STOP_AFTER_STEP >= MINIMUM_STOP_AFTER_STEP && STOP_AFTER_STEP <= 30000 )) || {
  echo "PICF_STOP_AFTER_STEP must be an integer in [$MINIMUM_STOP_AFTER_STEP, 30000]" >&2
  exit 2
}
(( STOP_AFTER_STEP > LOAD_GLOBAL_STEP )) || {
  echo "PICF_STOP_AFTER_STEP must exceed PICF_LOAD_GLOBAL_STEP" >&2
  exit 2
}
REQUIRED_DIRECTORIES=(
  "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" "$SIDECAR_ROOT" \
  "$ANYTOUCH_CACHE" "$SONATA_CACHE" "$VJEPA_CACHE"
)
REQUIRED_FILES=(
  "$PYTHON" "$DATASET_MANIFEST" "$NORM_STATS" "$SIDECAR_MANIFEST" \
  "$VISUAL_ACCEPTANCE" "$STREAM_PLAN" "$REPRESENTATION_SPLIT" \
  "$EVALUATION_PLAN" "$DENSE_COVERAGE" "$ANYTOUCH_CACHE/manifest.json" \
  "$SONATA_CACHE/manifest.json" "$VJEPA_CACHE/manifest.json"
)
if [[ "$USE_CURRENT_GRID_CACHE" == 1 ]]; then
  REQUIRED_DIRECTORIES+=("$CURRENT_CACHE" "$CURRENT_SHARD_ROOT")
  REQUIRED_FILES+=("$CURRENT_CACHE/manifest.json" "$CURRENT_REPORT")
fi
if [[ "$USE_DENSE_SUPPLEMENT" == 1 ]]; then
  REQUIRED_DIRECTORIES+=(
    "$ANYTOUCH_SUPPLEMENT" "$SONATA_SUPPLEMENT" "$VJEPA_SUPPLEMENT"
  )
  REQUIRED_FILES+=(
    "$ANYTOUCH_SUPPLEMENT/manifest.json" "$SONATA_SUPPLEMENT/manifest.json" \
    "$VJEPA_SUPPLEMENT/manifest.json"
  )
fi
if [[ -n "$FUTURE_LATENT_CACHE_ROOT" || -n "$FUTURE_LATENT_CACHE_BUILD_REPORT" ]]; then
  [[ -n "$FUTURE_LATENT_CACHE_ROOT" && -n "$FUTURE_LATENT_CACHE_BUILD_REPORT" ]] || {
    echo "future-latent cache root and build report must be supplied together" >&2
    exit 2
  }
  REQUIRED_DIRECTORIES+=("$FUTURE_LATENT_CACHE_ROOT")
  REQUIRED_FILES+=(
    "$FUTURE_LATENT_CACHE_ROOT/manifest.json"
    "$FUTURE_LATENT_CACHE_BUILD_REPORT"
  )
fi
for path in "${REQUIRED_DIRECTORIES[@]}"; do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required direct directory is absent: $path" >&2
    exit 1
  }
done
for path in "${REQUIRED_FILES[@]}"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required direct file is absent: $path" >&2
    exit 1
  }
done

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq "$WORLD_SIZE" ]] || {
  echo "ADR-178 requires exactly $WORLD_SIZE visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "ADR-178 requires $WORLD_SIZE A100 40GB GPUs; observed: $row" >&2
    exit 1
  }
done

file_sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

CURRENT_GRID_ARGS=()
if [[ "$USE_CURRENT_GRID_CACHE" == 1 ]]; then
  CURRENT_GRID_ARGS=(
    --current-grid-cache-root "$CURRENT_CACHE"
    --current-grid-cache-shard-root "$CURRENT_SHARD_ROOT"
    --current-grid-cache-build-report "$CURRENT_REPORT"
    --current-grid-cache-build-report-sha256 "$(file_sha256 "$CURRENT_REPORT")"
  )
fi
DENSE_SUPPLEMENT_ARGS=()
if [[ "$USE_DENSE_SUPPLEMENT" == 1 ]]; then
  DENSE_SUPPLEMENT_ARGS=(
    --dense-evidence-supplement-cache-root "$ANYTOUCH_SUPPLEMENT"
    --dense-evidence-supplement-cache-manifest-sha256 "$(file_sha256 "$ANYTOUCH_SUPPLEMENT/manifest.json")"
    --dense-evidence-supplement-cache-root "$SONATA_SUPPLEMENT"
    --dense-evidence-supplement-cache-manifest-sha256 "$(file_sha256 "$SONATA_SUPPLEMENT/manifest.json")"
    --dense-evidence-supplement-cache-root "$VJEPA_SUPPLEMENT"
    --dense-evidence-supplement-cache-manifest-sha256 "$(file_sha256 "$VJEPA_SUPPLEMENT/manifest.json")"
  )
fi
DENSE_SUBSET_VIEW_ARGS=()
if [[ "$DENSE_EVIDENCE_SUBSET_VIEW" == 1 ]]; then
  DENSE_SUBSET_VIEW_ARGS=(--dense-evidence-subset-view)
fi
FUTURE_LATENT_ARGS=()
if [[ -n "$FUTURE_LATENT_CACHE_ROOT" ]]; then
  FUTURE_LATENT_MANIFEST_SHA256=$("$PYTHON" - "$FUTURE_LATENT_CACHE_ROOT/manifest.json" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text("ascii"))
value = payload.get("manifest_sha256")
if not isinstance(value, str) or len(value) != 64:
    raise ValueError("future-latent manifest omits its canonical digest")
print(value)
PY
)
  FUTURE_LATENT_ARGS=(
    --future-latent-cache-root "$FUTURE_LATENT_CACHE_ROOT"
    --future-latent-cache-manifest-sha256 "$FUTURE_LATENT_MANIFEST_SHA256"
    --future-latent-cache-build-report "$FUTURE_LATENT_CACHE_BUILD_REPORT"
    --future-latent-cache-build-report-sha256 "$(file_sha256 "$FUTURE_LATENT_CACHE_BUILD_REPORT")"
  )
fi
FUTURE_SOURCE_DOMAIN_ARGS=()
if [[ -n "$MINIMUM_FUTURE_SOURCE_FRAMES" ]]; then
  case "$MINIMUM_FUTURE_SOURCE_FRAMES" in 4|8|16) ;; *)
    echo "PICF_MINIMUM_FUTURE_SOURCE_FRAMES must be empty, 4, 8, or 16" >&2
    exit 2
  esac
  FUTURE_SOURCE_DOMAIN_ARGS=(
    --minimum-future-source-frames "$MINIMUM_FUTURE_SOURCE_FRAMES"
  )
fi
WSA_ARGS=()
if [[ "$ARCHITECTURE_PROFILE" == adr221_native_videomt_wsa_full_modal_v1 \
  || "$ARCHITECTURE_PROFILE" == adr222_native_videomt_world_token_wsa_v1 ]]; then
  : "${WSA_SOURCE_ROOT:?PICF_WSA_SOURCE_ROOT is required for ADR-221}"
  : "${WSA_ADAPTED_CHECKPOINT:?PICF_WSA_ADAPTED_CHECKPOINT is required for ADR-221}"
  : "${WSA_ADAPTED_CHECKPOINT_SHA256:?PICF_WSA_ADAPTED_CHECKPOINT_SHA256 is required for ADR-221}"
  : "${DA3_SOURCE_ROOT:?PICF_DA3_SOURCE_ROOT is required for ADR-221}"
  : "${DA3_MODEL_DIR:?PICF_DA3_MODEL_DIR is required for ADR-221}"
  : "${DA3_MODEL_CHECKPOINT_SHA256:?PICF_DA3_MODEL_CHECKPOINT_SHA256 is required for ADR-221}"
  : "${DA3_MODEL_CONFIG_SHA256:?PICF_DA3_MODEL_CONFIG_SHA256 is required for ADR-221}"
  for path in "$WSA_SOURCE_ROOT" "$DA3_SOURCE_ROOT" "$DA3_MODEL_DIR"; do
    [[ -d "$path" && ! -L "$path" ]] || {
      echo "required direct ADR-221 directory is absent: $path" >&2
      exit 1
    }
  done
  [[ -f "$WSA_ADAPTED_CHECKPOINT" && ! -L "$WSA_ADAPTED_CHECKPOINT" ]] || {
    echo "required direct ADR-221 checkpoint is absent: $WSA_ADAPTED_CHECKPOINT" >&2
    exit 1
  }
  for digest in \
    "$WSA_ADAPTED_CHECKPOINT_SHA256" \
    "$DA3_MODEL_CHECKPOINT_SHA256" \
    "$DA3_MODEL_CONFIG_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
      echo "ADR-221 asset digests must be lowercase SHA-256 values" >&2
      exit 2
    }
  done
  WSA_ARGS=(
    --wsa-source-root "$WSA_SOURCE_ROOT"
    --wsa-adapted-checkpoint "$WSA_ADAPTED_CHECKPOINT"
    --wsa-adapted-checkpoint-sha256 "$WSA_ADAPTED_CHECKPOINT_SHA256"
    --da3-source-root "$DA3_SOURCE_ROOT"
    --da3-model-dir "$DA3_MODEL_DIR"
    --da3-model-checkpoint-sha256 "$DA3_MODEL_CHECKPOINT_SHA256"
    --da3-model-config-sha256 "$DA3_MODEL_CONFIG_SHA256"
  )
elif [[ -n "$WSA_SOURCE_ROOT$WSA_ADAPTED_CHECKPOINT$WSA_ADAPTED_CHECKPOINT_SHA256$DA3_SOURCE_ROOT$DA3_MODEL_DIR$DA3_MODEL_CHECKPOINT_SHA256$DA3_MODEL_CONFIG_SHA256" ]]; then
  echo "WSA/DA3 assets are exclusive to registered WSA profiles" >&2
  exit 2
fi

WLA_ARGS=()
case "$ACTION_BACKEND" in
  lingbot_released)
    [[ -z "$WLA_SOURCE_ROOT$WLA_PRETRAINED_ROOT" ]] || {
      echo "WLA assets require PICF_ACTION_BACKEND=wla_complete" >&2
      exit 2
    }
    ;;
  wla_complete)
    : "${WLA_SOURCE_ROOT:?PICF_WLA_SOURCE_ROOT is required for complete WLA}"
    : "${WLA_PRETRAINED_ROOT:?PICF_WLA_PRETRAINED_ROOT is required for complete WLA}"
    for path in "$WLA_SOURCE_ROOT" "$WLA_PRETRAINED_ROOT"; do
      [[ -d "$path" && ! -L "$path" ]] || {
        echo "required direct WLA directory is absent: $path" >&2
        exit 1
      }
    done
    WLA_ARGS=(
      --action-backend wla_complete
      --wla-source-root "$WLA_SOURCE_ROOT"
      --wla-pretrained-root "$WLA_PRETRAINED_ROOT"
    )
    ;;
  *)
    echo "unknown PICF_ACTION_BACKEND: $ACTION_BACKEND" >&2
    exit 2
    ;;
esac
case "$WLA_HOST_EVIDENCE_ARM" in
  picf_full|wla_lbot_masked) ;;
  *)
    echo "unknown PICF_WLA_HOST_EVIDENCE_ARM: $WLA_HOST_EVIDENCE_ARM" >&2
    exit 2
    ;;
esac
SHARED_QUERY_GRADIENT_ARGS=()
case "$VIDEOMT_SHARED_QUERY_GRADIENT_DIAGNOSTIC" in
  0) ;;
  1) SHARED_QUERY_GRADIENT_ARGS=(--videomt-shared-query-gradient-diagnostic) ;;
  *)
    echo "PICF_VIDEOMT_SHARED_QUERY_GRADIENT_DIAGNOSTIC must be 0 or 1" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO${RUNTIME_PYTHON_OVERLAY:+:$RUNTIME_PYTHON_OVERLAY}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ENABLE_TIMING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_TRACE_CPP_STACK=1
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_FR_CPP_STACK=1
cd "$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$WORLD_SIZE" \
  tools/run_lingbot_vla2_task_independent_full.py \
  --phase "$PHASE" \
  --source-checkout "$SOURCE" \
  --checkpoint-dir "$CHECKPOINT" \
  --processor-dir "$PROCESSOR" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --norm-stats "$NORM_STATS" \
  --stream-plan "$STREAM_PLAN" \
  --stream-plan-sha256 "$(file_sha256 "$STREAM_PLAN")" \
  "${FUTURE_SOURCE_DOMAIN_ARGS[@]}" \
  "${WSA_ARGS[@]}" \
  "${WLA_ARGS[@]}" \
  --wla-host-evidence-arm "$WLA_HOST_EVIDENCE_ARM" \
  --videomt-source-update-arm "$VIDEOMT_SOURCE_UPDATE_ARM" \
  "${SHARED_QUERY_GRADIENT_ARGS[@]}" \
  --representation-split "$REPRESENTATION_SPLIT" \
  --representation-split-sha256 "$(file_sha256 "$REPRESENTATION_SPLIT")" \
  --evaluation-plan "$EVALUATION_PLAN" \
  --evaluation-plan-sha256 "$(file_sha256 "$EVALUATION_PLAN")" \
  --physical-sidecar-root "$SIDECAR_ROOT" \
  --physical-sidecar-manifest "$SIDECAR_MANIFEST" \
  --physical-sidecar-manifest-sha256 ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d \
  --physical-visual-acceptance "$VISUAL_ACCEPTANCE" \
  --physical-visual-acceptance-sha256 6443c34b6e8180a8ec090d50ee14dbb2e9d0ad6c4a5e2fc0d9f03a1dbd156552 \
  "${CURRENT_GRID_ARGS[@]}" \
  "${FUTURE_LATENT_ARGS[@]}" \
  --future-latent-objective-scale "$FUTURE_LATENT_OBJECTIVE_SCALE" \
  --dense-evidence-mode calvin_full_v1 \
  --dense-token-bridge "$DENSE_TOKEN_BRIDGE" \
  --dense-evidence-cache-root "$ANYTOUCH_CACHE" \
  --dense-evidence-cache-manifest-sha256 "$(file_sha256 "$ANYTOUCH_CACHE/manifest.json")" \
  --dense-evidence-cache-root "$SONATA_CACHE" \
  --dense-evidence-cache-manifest-sha256 "$(file_sha256 "$SONATA_CACHE/manifest.json")" \
  --dense-evidence-cache-root "$VJEPA_CACHE" \
  --dense-evidence-cache-manifest-sha256 "$(file_sha256 "$VJEPA_CACHE/manifest.json")" \
  "${DENSE_SUPPLEMENT_ARGS[@]}" \
  "${DENSE_SUBSET_VIEW_ARGS[@]}" \
  --dense-evidence-coverage-plan "$DENSE_COVERAGE" \
  --dense-evidence-coverage-plan-sha256 "$(file_sha256 "$DENSE_COVERAGE")" \
  --run-dir "$RUN_DIR" \
  --load-global-step "$LOAD_GLOBAL_STEP" \
  --stop-after-step "$STOP_AFTER_STEP" \
  --seed 20260721 \
  --attention-implementation "$ATTENTION_IMPLEMENTATION" \
  --lingbot-compile-mode "$LINGBOT_COMPILE_MODE" \
  --trainable-scope "$TRAINABLE_SCOPE" \
  --capacity "$CAPACITY" \
  --maximum-control-tokens 64 \
  --prior-gradient-control-tokens 8 \
  --posterior-architecture two_pass_v3 \
  --picf-architecture-profile "$ARCHITECTURE_PROFILE" \
  --task-query-count 0 \
  --relation-supervision-layers "$RELATION_SUPERVISION_LAYERS" \
  --learning-rate "$LEARNING_RATE" \
  --picf-learning-rate-multiplier "$PICF_LEARNING_RATE_MULTIPLIER" \
  --modality-bridge-learning-rate-multiplier "$MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER" \
  --max-grad-norm 1.0 \
  --maximum-peak-reserved-gib 39.0 \
  --maximum-optimizer-lag 8 \
  --causal-ablation-mode none \
  --entity-weight "$ENTITY_WEIGHT" \
  --predictive-weight "$PREDICTIVE_WEIGHT" \
  --local-bptt-probability "$LOCAL_BPTT_PROBABILITY" \
  --overshoot-probability "$OVERSHOOT_PROBABILITY" \
  --source-mask-probability "$SOURCE_MASK_PROBABILITY" \
  --source-mask-token-fraction 0.0625 \
  --source-prediction-mode omitted_static \
  --omitted-static-rematerialization "$OMITTED_STATIC_REMATERIALIZATION" \
  --fsdp2-placement "$FSDP2_PLACEMENT" \
  --fsdp2-backward-prefetch "$BACKWARD_PREFETCH" \
  --sequential-factual-gradient-storage "$FACTUAL_GRADIENT_STORAGE" \
  --cuda-allocator "$CUDA_ALLOCATOR" \
  "${WSA_EDGE_DIAGNOSTIC_ARGS[@]}" \
  "${VIDEOMT_STAGE_PQ_ARGS[@]}"
