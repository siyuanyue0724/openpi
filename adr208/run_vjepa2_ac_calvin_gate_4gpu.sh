#!/usr/bin/env bash
set -euo pipefail

REPO=${PICF_ADR208_REPO:-/mnt/picf-next/adr208/source-freezes/vjepa2-ac-arm-b-v3}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
CHECKPOINT=${PICF_VJEPA2_AC_CHECKPOINT:-/mnt/picf-next/models/foundation/vjepa2_ac/vjepa2-ac-vitg.pt}
RUN_ROOT=${PICF_ADR208_RUN_ROOT:-/mnt/picf-next/adr208/runs}
GLOBAL_CLIP_COUNT=${PICF_VJEPA2_AC_CLIP_COUNT:-32}
CONTROL_SEED=${PICF_VJEPA2_AC_CONTROL_SEED:-239}
CAMERA=${PICF_VJEPA2_AC_CAMERA:-rgb_static}
SHARD_COUNT=4

fail() {
  printf 'ADR-208 four-GPU gate failed: %s\n' "$*" >&2
  exit 2
}

[[ -d "$REPO" ]] || fail "missing immutable source freeze: $REPO"
[[ -f "$REPO/SOURCE_FREEZE.sha256" ]] || fail "missing source-freeze inventory"
if [[ ! -x "$PYTHON" ]]; then
  PICF_RUNTIME_ROOT=$(dirname "$(dirname "$PYTHON")") \
    "$REPO/adr208/restore_arm_b_runtime.sh"
fi
[[ -x "$PYTHON" ]] || fail "missing runtime Python after restore: $PYTHON"
[[ "$GLOBAL_CLIP_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "invalid global clip count"
(( GLOBAL_CLIP_COUNT >= 16 && GLOBAL_CLIP_COUNT % SHARD_COUNT == 0 )) || fail \
  "global clip count must be at least 16 and divisible by four"
[[ "$CONTROL_SEED" =~ ^[0-9]+$ ]] || fail "invalid control seed"
[[ "$CAMERA" == rgb_static || "$CAMERA" == rgb_gripper ]] || fail "invalid camera"

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$gpu_count" -eq 4 ]] || fail "expected exactly four visible GPUs, found $gpu_count"
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[[ "$available_kib" -ge 125829120 ]] || fail \
  "four concurrent exact model loads require at least 120 GiB available host memory"

(
  cd "$REPO"
  sha256sum -c SOURCE_FREEZE.sha256 >/dev/null
)

export PICF_ADR208_REPO=$REPO
export PICF_VJEPA2_AC_CHECKPOINT=$CHECKPOINT
"$REPO/adr208/fetch_vjepa2_ac_checkpoint.sh"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REPO/src:$REPO
"$PYTHON" -m pytest -q \
  "$REPO/tests/test_calvin_vjepa2_ac.py" \
  "$REPO/tests/test_probe_calvin_vjepa2_ac.py" \
  "$REPO/tests/test_vjepa2_ac_source_fidelity.py"

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
batch=$RUN_ROOT/vjepa2-ac-arm-b-4gpu-$timestamp-$CAMERA-seed$CONTROL_SEED-n$GLOBAL_CLIP_COUNT
mkdir -p "$batch"
nvidia-smi -q >"$batch/nvidia-smi-before.txt"

pids=()
for shard_index in 0 1 2 3; do
  shard_root=$batch/shard-$shard_index
  mkdir -p "$shard_root"
  (
    export CUDA_VISIBLE_DEVICES=$shard_index
    export OMP_NUM_THREADS=4
    export PICF_ADR208_RUN_ROOT=$shard_root
    export PICF_VJEPA2_AC_CLIP_COUNT=$GLOBAL_CLIP_COUNT
    export PICF_VJEPA2_AC_CONTROL_SEED=$CONTROL_SEED
    export PICF_VJEPA2_AC_CAMERA=$CAMERA
    export PICF_VJEPA2_AC_SHARD_INDEX=$shard_index
    export PICF_VJEPA2_AC_SHARD_COUNT=$SHARD_COUNT
    export PICF_VJEPA2_AC_SKIP_LOCAL_TESTS=1
    "$REPO/adr208/run_vjepa2_ac_calvin_gate.sh"
  ) >"$shard_root/launch.log" 2>&1 &
  pids+=("$!")
done

failed=0
for shard_index in 0 1 2 3; do
  if ! wait "${pids[$shard_index]}"; then
    printf 'shard %s failed; inspect %s/shard-%s/launch.log\n' \
      "$shard_index" "$batch" "$shard_index" >&2
    failed=1
  fi
done
nvidia-smi -q >"$batch/nvidia-smi-after.txt"
[[ "$failed" == 0 ]] || exit 2

aggregate_args=()
for shard_index in 0 1 2 3; do
  mapfile -t reports < <(find "$batch/shard-$shard_index" -type f -name causal-gate.json)
  [[ ${#reports[@]} == 1 ]] || fail "shard $shard_index produced ${#reports[@]} reports"
  aggregate_args+=(--report "${reports[0]}")
done

"$PYTHON" "$REPO/tools/aggregate_vjepa2_ac_shards.py" \
  "${aggregate_args[@]}" \
  --output "$batch/four-gpu-verdict.json" \
  >"$batch/aggregate.stdout.json"

printf 'ADR-208 four-GPU gate passed: %s\n' "$batch/four-gpu-verdict.json"
