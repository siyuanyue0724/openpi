#!/usr/bin/env bash
set -euo pipefail

# A7 CALVIN action/video gate for the May-22 action-aware PICF checkpoints.
#
# Purpose:
#   Inspect actual CALVIN rollout actions for the 2000 and 3000 checkpoints.
#   This is a behavior/video probe, not a latency gate.  It intentionally runs
#   full PICF observe every control step by default so checkpoint quality is not
#   confounded with deployment-time belief-update amortization.
#
#   Anchor debug is disabled by default.  The compact CALVIN action video and
#   action_safety.jsonl are the correct artifacts for checking actual actions.
#   Anchor debug can produce very large JSONL payloads and should be enabled
#   only for a separate short anchor-diagnostic probe.
#
# GPU topology:
#   physical GPU 0: policy server
#   physical GPU 1: CALVIN EGL/evaluator

REPO_ROOT="${REPO_ROOT:-/root/openpi_slot_quality_ea2c5f2}"
PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
CALVIN_AGENT_ROOT="${CALVIN_AGENT_ROOT:-/mnt/calvin/calvin_models/calvin_agent}"
CALVIN_DATASET="${CALVIN_DATASET:-/mnt/calvin_data/task_ABC_D}"

RUN_NAME="${RUN_NAME:-picf_a7_actionaware_ckpt2000_3000_video_eval_20260523}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_actionaware_qgfloor003_from1500_long30k_20260522}"
CKPT_STEPS="${CKPT_STEPS:-2000 3000}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1}"
SERVER_GPU="${SERVER_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
BASE_PORT="${BASE_PORT:-8140}"
OUT_ROOT="${OUT_ROOT:-/mnt/checkpoints/picf_core/eval/${RUN_NAME}}"
TMP_ROOT="${TMP_ROOT:-/tmp/${RUN_NAME}}"
PICF_OBSERVE_INTERVAL="${PICF_OBSERVE_INTERVAL:-1}"
SAVE_ANCHORS="${SAVE_ANCHORS:-0}"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TORCHDYNAMO_DISABLE=1
export OPENPI_DISABLE_TORCH_COMPILE=1
export OPENPI_PICF_TIMING_BREAKDOWN=1
export OPENPI_PICF_EXPORT_ANCHOR_DENSE=0
export OPENPI_PICF_EXPORT_PREDICTIONS=0

mkdir -p "${OUT_ROOT}/logs" "${TMP_ROOT}"

cleanup_processes() {
  for port in $(seq "${BASE_PORT}" "$((BASE_PORT + 8))"); do
    pkill -f "scripts/serve_picf_policy.py.*--port ${port}" 2>/dev/null || true
  done
  pkill -f "scripts/calvin/evaluate_picf_policy.py.*${RUN_NAME}" 2>/dev/null || true
}

wait_health() {
  local port="$1"
  local deadline="$2"
  "${PYTHON_BIN}" - "$port" "$deadline" <<'PY'
import sys, time, urllib.request

port = int(sys.argv[1])
deadline = time.time() + float(sys.argv[2])
url = f"http://127.0.0.1:{port}/healthz"
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            if resp.status == 200:
                print(f"server_ready port={port}")
                raise SystemExit(0)
    except Exception as exc:
        last = exc
        time.sleep(2.0)
print(f"server_not_ready port={port} last={last}", file=sys.stderr)
raise SystemExit(1)
PY
}

mirror_outputs() {
  local tmp_dir="$1"
  local run_dir="$2"
  mkdir -p "${run_dir}/videos" "${run_dir}/logs" "${run_dir}/eval_logs" "${run_dir}/anchor_debug"
  cp -f "${tmp_dir}/logs/eval.log" "${run_dir}/logs/eval.log" 2>/dev/null || true
  cp -f "${tmp_dir}/logs/eval.log" "${run_dir}/eval.log" 2>/dev/null || true
  cp -f "${tmp_dir}/eval_logs"/* "${run_dir}/eval_logs"/ 2>/dev/null || true
  cp -f "${tmp_dir}/anchor_debug"/* "${run_dir}/anchor_debug"/ 2>/dev/null || true
  for src in "${tmp_dir}"/videos/*.mp4 "${tmp_dir}"/anchor_debug/*.mp4; do
    [ -f "${src}" ] || continue
    local subdir="videos"
    case "${src}" in
      *"/anchor_debug/"*) subdir="anchor_debug" ;;
    esac
    local dst="${run_dir}/${subdir}/$(basename "${src}")"
    local src_size
    src_size=$(stat -c %s "${src}")
    if [ "${src_size}" -gt 4096 ]; then
      cp -f "${src}" "${dst}.tmp" && mv -f "${dst}.tmp" "${dst}"
    fi
  done
}

summarize_actions() {
  local run_dir="$1"
  local safety_log="${run_dir}/action_safety.jsonl"
  "${PYTHON_BIN}" - "${safety_log}" "${run_dir}/logs/action_summary.txt" <<'PY'
import json, math, statistics, sys
from pathlib import Path

safety_log = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
if safety_log.is_file():
    for line in safety_log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
nonfinite = sum(1 for r in rows if not bool(r.get("finite_all", True)))
clipped = sum(1 for r in rows if bool(r.get("clip_changed", False)))
infer = [float(r["infer_ms"]) for r in rows if isinstance(r.get("infer_ms"), (int, float)) and math.isfinite(float(r["infer_ms"]))]
lines = [
    f"records={len(rows)}",
    f"nonfinite_actions={nonfinite}",
    f"clip_changed={clipped}",
]
if infer:
    lines.append(f"infer_ms_mean={statistics.fmean(infer):.2f}")
    lines.append(f"infer_ms_p50={statistics.median(sorted(infer)):.2f}")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

run_ckpt() {
  local step="$1"
  local port="$2"
  local checkpoint="${CHECKPOINT_ROOT}/${step}"
  local tag="ckpt${step}_observe${PICF_OBSERVE_INTERVAL}"
  local run_dir="${OUT_ROOT}/${tag}"
  local tmp_dir="${TMP_ROOT}/${tag}"
  mkdir -p "${run_dir}/logs" "${run_dir}/videos" "${run_dir}/eval_logs" "${run_dir}/anchor_debug"
  mkdir -p "${tmp_dir}/logs" "${tmp_dir}/videos" "${tmp_dir}/eval_logs" "${tmp_dir}/anchor_debug"

  if [ ! -d "${checkpoint}" ]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    return 1
  fi

  echo "===== ${tag}: starting server checkpoint=${checkpoint}"
  cleanup_processes
  cd "${REPO_ROOT}"
  (
    export CUDA_VISIBLE_DEVICES="${SERVER_GPU}"
    export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src"
    export OPENPI_PICF_EXPORT_ANCHORS="${SAVE_ANCHORS}"
    exec "${PYTHON_BIN}" scripts/serve_picf_policy.py \
      --checkpoint "${checkpoint}" \
      --device cuda:0 \
      --port "${port}" \
      --picf-mode enabled \
      --picf-observe-interval "${PICF_OBSERVE_INTERVAL}" \
      $(if [ "${SAVE_ANCHORS}" = "1" ]; then printf '%s' "--export-anchor-debug"; fi)
  ) > "${run_dir}/logs/server.log" 2>&1 &
  local server_pid=$!
  echo "${server_pid}" > "${run_dir}/server.pid"
  wait_health "${port}" 420

  echo "===== ${tag}: starting CALVIN evaluator num_sequences=${NUM_SEQUENCES}"
  (
    cd "${CALVIN_AGENT_ROOT}"
    set +u
    eval "$(/root/bin/micromamba shell hook -s bash)"
    micromamba activate calvin38
    set -u
    export PYTHONUNBUFFERED=1
    export PYOPENGL_PLATFORM=egl
    export CUDA_VISIBLE_DEVICES="${EVAL_GPU}"
    export EGL_VISIBLE_DEVICES="${EVAL_GPU}"
    export OPENPI_SERVER_HOST=127.0.0.1
    export OPENPI_SERVER_PORT="${port}"
    export OPENPI_EVAL_TAG="${tag}"
    export PYTHONPATH="/root/calvin_patch:/mnt/calvin/calvin_env:${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src:${CALVIN_AGENT_ROOT}"
    python "${REPO_ROOT}/scripts/calvin/evaluate_picf_policy.py" \
      --dataset_path "${CALVIN_DATASET}" \
      --eval_log_dir "${tmp_dir}/eval_logs" \
      --num_sequences "${NUM_SEQUENCES}" \
      --server_host 127.0.0.1 \
      --server_port "${port}" \
      --save_video \
      --video_dir "${tmp_dir}/videos" \
      $(if [ "${SAVE_ANCHORS}" = "1" ]; then printf '%s' "--save_anchor_debug --anchor_debug_dir ${tmp_dir}/anchor_debug"; fi) \
      --action_safety_log "${run_dir}/action_safety.jsonl" \
      --action_clip 1.0 \
      --calvin_agent_root "${CALVIN_AGENT_ROOT}"
  ) > "${tmp_dir}/logs/eval.log" 2>&1 || true

  mirror_outputs "${tmp_dir}" "${run_dir}"
  summarize_actions "${run_dir}" | tee "${run_dir}/logs/action_summary.stdout"
  echo "===== ${tag}: stopping server pid=${server_pid}"
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}

main() {
  cleanup_processes
  {
    echo "started_at=$(date -Is)"
    echo "repo=${REPO_ROOT}"
    echo "checkpoint_root=${CHECKPOINT_ROOT}"
    echo "ckpt_steps=${CKPT_STEPS}"
    echo "num_sequences=${NUM_SEQUENCES}"
    echo "picf_observe_interval=${PICF_OBSERVE_INTERVAL}"
    echo "save_anchors=${SAVE_ANCHORS}"
  } > "${OUT_ROOT}/run_config.txt"

  local idx=0
  for step in ${CKPT_STEPS}; do
    run_ckpt "${step}" "$((BASE_PORT + idx))"
    idx=$((idx + 1))
  done
  echo "finished_at=$(date -Is)" >> "${OUT_ROOT}/run_config.txt"
  cleanup_processes
}

main "$@"
