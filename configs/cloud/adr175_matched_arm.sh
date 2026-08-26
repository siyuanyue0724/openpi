#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  printf 'usage: %s [lbot|physical-set|native-attention] [1|250|500|1000|2000]\n' "$0" >&2
  exit 2
fi

ARM=$1
STEPS=$2

case "$ARM" in
  lbot|physical-set|native-attention) ;;
  *)
    printf 'usage: %s [lbot|physical-set|native-attention] [1|250|500|1000|2000]\n' "$0" >&2
    exit 2
    ;;
esac

case "$STEPS" in
  1) evaluation_steps=0,1 ;;
  250) evaluation_steps=0,250 ;;
  500) evaluation_steps=0,250,500 ;;
  1000) evaluation_steps=0,250,500,1000 ;;
  2000) evaluation_steps=0,250,500,1000,2000 ;;
  *)
    printf 'usage: %s [lbot|physical-set|native-attention] [1|250|500|1000|2000]\n' "$0" >&2
    exit 2
    ;;
esac

WORKTREE=${PICF_ADR175_WORKTREE:?set PICF_ADR175_WORKTREE to the immutable source freeze}
SOURCE=${PICF_LINGBOT_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr175-native-v1}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
RUN_ROOT=${PICF_ADR175_RUN_ROOT:-/mnt/picf-next/adr175/runs}
DATASET=${PICF_ADR175_DATASET:-/mnt/calvin_data/task_ABC_D/training}
PROVENANCE=/mnt/picf-next-provenance
CONTRACT_ROOT=${PICF_ADR175_CONTRACT_ROOT:?set PICF_ADR175_CONTRACT_ROOT}
IMPLEMENTATION_CLOSURE=${PICF_ADR175_IMPLEMENTATION_CLOSURE:-$WORKTREE/ADR175_IMPLEMENTATION_CLOSURE.json}
IMPLEMENTATION_CLOSURE_FILE_SHA256=${PICF_ADR175_IMPLEMENTATION_CLOSURE_FILE_SHA256:?set closure file SHA-256}
IMPLEMENTATION_CLOSURE_ARTIFACT_SHA256=${PICF_ADR175_IMPLEMENTATION_CLOSURE_ARTIFACT_SHA256:?set closure artifact SHA-256}
CONTRACT_FILE_SHA256=${PICF_ADR175_CONTRACT_FILE_SHA256:?set broad-support contract file SHA-256}
STREAM_PLAN_FILE_SHA256=${PICF_ADR175_STREAM_PLAN_FILE_SHA256:?set stream-plan file SHA-256}
REPRESENTATION_SPLIT_FILE_SHA256=${PICF_ADR175_REPRESENTATION_SPLIT_FILE_SHA256:?set representation-split file SHA-256}
ENTITY_EVALUATION_PLAN_FILE_SHA256=${PICF_ADR175_ENTITY_EVALUATION_PLAN_FILE_SHA256:?set entity-evaluation-plan file SHA-256}

if pgrep -f '[r]un_lingbot_vla2_task_independent_p1.py' >/dev/null; then
  echo "refusing to launch while another P1/ADR-175 runner is active" >&2
  exit 1
fi

stamp=$(date +%Y%m%dT%H%M%S%N%z)
run_dir="$RUN_ROOT/${ARM}-steps-${STEPS}-${stamp}"
mkdir "$run_dir"
log="$run_dir/bridge.log"

"$PYTHON" "$WORKTREE/tools/verify_adr175_implementation_closure.py" \
  --root "$WORKTREE" \
  --manifest "$IMPLEMENTATION_CLOSURE" \
  --manifest-file-sha256 "$IMPLEMENTATION_CLOSURE_FILE_SHA256" \
  --expected-artifact-sha256 "$IMPLEMENTATION_CLOSURE_ARTIFACT_SHA256" \
  --output "$run_dir/implementation-closure-verification.json"

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$WORKTREE/src:$WORKTREE:$SOURCE"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

nohup "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  "$WORKTREE/tools/run_lingbot_vla2_task_independent_p1.py" \
  --source-checkout "$SOURCE" \
  --patch "$WORKTREE/references/patches/lingbot_vla2_picf_native.patch" \
  --training-config "$SOURCE/configs/vla/robotwin/robotwin.yaml" \
  --robot-config "$WORKTREE/configs/lingbot/calvin_robot.yaml" \
  --data-config "$WORKTREE/configs/lingbot/calvin_data.json" \
  --checkpoint-dir /mnt/picf-next/models/lingbot-vla-v2-6b \
  --processor-dir /mnt/picf-next/models/qwen3-vl-4b-instruct \
  --dataset-split "$DATASET" \
  --dataset-manifest "$PROVENANCE/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json" \
  --norm-stats "$PROVENANCE/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json" \
  --physical-sidecar-root /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z \
  --physical-sidecar-manifest "$PROVENANCE/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json" \
  --physical-sidecar-manifest-sha256 ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d \
  --run-dir "$run_dir" \
  --adr175-arm "$ARM" \
  --adr175-contract "$CONTRACT_ROOT/broad-support-contract.json" \
  --adr175-contract-sha256 "$CONTRACT_FILE_SHA256" \
  --stream-plan "$CONTRACT_ROOT/stream-plan.json" \
  --stream-plan-sha256 "$STREAM_PLAN_FILE_SHA256" \
  --representation-split "$CONTRACT_ROOT/representation-split.json" \
  --representation-split-sha256 "$REPRESENTATION_SPLIT_FILE_SHA256" \
  --entity-evaluation-plan "$CONTRACT_ROOT/entity-evaluation-plan.json" \
  --entity-evaluation-plan-sha256 "$ENTITY_EVALUATION_PLAN_FILE_SHA256" \
  --evaluation-steps "$evaluation_steps" \
  --evaluation-visuals-per-partition 4 \
  --steps "$STEPS" \
  --seed 20260816 \
  --capacity 16 \
  --maximum-control-tokens 64 \
  --visual-lattice 8 \
  --current-frame-action-weight 1.0 \
  --current-frame-entity-weight 0.08 \
  --fsdp2-placement selective-embedding-offload \
  --maximum-peak-reserved-gib 39.0 \
  --cuda-allocator native \
  >"$log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" >"$run_dir/launcher.pid"
printf '%s\n' "$run_dir" >"$RUN_ROOT/ACTIVE_ADR175_RUN"
printf '%s\n' "$log" >"$RUN_ROOT/ACTIVE_ADR175_LOG"
printf 'pid=%s run_dir=%s log=%s\n' "$pid" "$run_dir" "$log"
