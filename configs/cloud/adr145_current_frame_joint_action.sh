#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}
WORKTREE=${PICF_ADR145_WORKTREE:-/mnt/picf-next/worktrees/adr145-current-frame-joint-v3-20260807}
SOURCE=${PICF_LINGBOT_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python}
RUN_ROOT=${PICF_ADR145_RUN_ROOT:-/mnt/picf-next/adr145/runs}
DATASET=${PICF_ADR145_DATASET:-/mnt/calvin_data/task_ABC_D/training}

case "$MODE" in
  smoke)
    steps=1
    evaluation_args=()
    ;;
  curve)
    steps=200
    evaluation_args=(
      --stream-plan /mnt/picf-next/adr135/contracts/k1-1000.stream-plan.json
      --stream-plan-sha256 ad83dd98553247bf782c54362b6893ff7904ea3f4cad3a10a54fb2f2808df1fd
      --representation-split /mnt/picf-next/adr135/contracts/k1-1000.split.json
      --representation-split-sha256 feff4db372ad3f2623c1e43acfe321a0682ddf35560a387dc3e935e907abb149
      --entity-evaluation-plan /mnt/picf-next/adr138/contracts/p1-entity-evaluation-plan.json
      --entity-evaluation-plan-sha256 7905e9cadd1f0d9f680bbe69a176c28c3453adfaea58c29e89739ca4c27f02e8
      --evaluation-steps 20,100,200
    )
    ;;
  *)
    echo "usage: $0 [smoke|curve]" >&2
    exit 2
    ;;
esac

if pgrep -f '[r]un_lingbot_vla2_task_independent_p1.py' >/dev/null; then
  echo "refusing to launch while another P1/bridge runner is active" >&2
  exit 1
fi

stamp=$(date +%Y%m%dT%H%M%S%z)
run_dir="$RUN_ROOT/current-frame-joint-action-${MODE}-${stamp}"
mkdir -p "$run_dir"
log="$run_dir/bridge.log"

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
  --dataset-manifest /mnt/picf-next/manifests/calvin-training-files.json \
  --norm-stats /mnt/picf-next/manifests/calvin-lingbot-norm-stats.json \
  --physical-sidecar-root /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z \
  --physical-sidecar-manifest /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z/manifest.json \
  --physical-sidecar-manifest-sha256 0198b9d184069f40f1804de411e25ffb3f3a446fcd61d5dd619e944488244ed4 \
  --run-dir "$run_dir" \
  --steps "$steps" \
  --seed 20260805 \
  --visual-lattice 8 \
  --current-frame-action-weight 1.0 \
  --current-frame-entity-weight 0.08 \
  --fsdp2-placement selective-embedding-offload \
  --maximum-peak-reserved-gib 39.0 \
  --cuda-allocator native \
  "${evaluation_args[@]}" \
  >"$log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" >"$run_dir/launcher.pid"
printf '%s\n' "$run_dir" >"$RUN_ROOT/ACTIVE_ADR145_RUN"
printf '%s\n' "$log" >"$RUN_ROOT/ACTIVE_ADR145_LOG"
printf 'pid=%s run_dir=%s log=%s\n' "$pid" "$run_dir" "$log"
