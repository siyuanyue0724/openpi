#!/usr/bin/env bash
set -euo pipefail

# A7 CALVIN inference latency gate.
#
# Purpose:
#   Compare PI0.5-only ablated serving against PICF-enabled serving under the
#   same CALVIN websocket/evaluator path, with timing breakdown enabled and all
#   heavy debug payloads disabled.  This measures deploy-time inference
#   overhead; it is not a behavior-acceptance run.
#
# GPU topology:
#   physical GPU 0: policy server
#   physical GPU 1: CALVIN EGL/evaluator
#
# Acceptance targets:
#   strong: PICF median policy latency <= 2.0x ablated PI0.5 median
#   weak:   PICF median policy latency <= 3.0x ablated PI0.5 median

REPO_ROOT="${REPO_ROOT:-/root/openpi_slot_quality_ea2c5f2}"
PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
CALVIN_AGENT_ROOT="${CALVIN_AGENT_ROOT:-/mnt/calvin/calvin_models/calvin_agent}"
CALVIN_DATASET="${CALVIN_DATASET:-/mnt/calvin_data/task_ABC_D}"

PI05_CHECKPOINT="${PI05_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_v22_ablated_pi05_30000_ckpt2500_print100_20260422_r2/20000}"
PICF_CHECKPOINT="${PICF_CHECKPOINT:-/mnt/checkpoints/picf_core/picf_core/picf_a7_actionaware_qgfloor003_from1500_long30k_20260522/3000}"

NUM_SEQUENCES="${NUM_SEQUENCES:-2}"
SERVER_GPU="${SERVER_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
BASE_PORT="${BASE_PORT:-8110}"
OUT_ROOT="${OUT_ROOT:-/mnt/checkpoints/picf_core/eval/picf_inference_latency_gate_20260523}"
TMP_ROOT="${TMP_ROOT:-/tmp/picf_inference_latency_gate_20260523}"
PICF_OBSERVE_INTERVAL="${PICF_OBSERVE_INTERVAL:-1}"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TORCHDYNAMO_DISABLE=1
export OPENPI_DISABLE_TORCH_COMPILE=1
export OPENPI_PICF_TIMING_BREAKDOWN=1
export OPENPI_PICF_EXPORT_ANCHORS=0
export OPENPI_PICF_EXPORT_ANCHOR_DENSE=0
export OPENPI_PICF_EXPORT_PREDICTIONS=0

mkdir -p "${OUT_ROOT}/logs" "${TMP_ROOT}"

cleanup_processes() {
  pkill -f "scripts/serve_picf_policy.py.*--port ${BASE_PORT}" 2>/dev/null || true
  pkill -f "scripts/calvin/evaluate_picf_policy.py.*${OUT_ROOT}" 2>/dev/null || true
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

summarize_timing() {
  local run_dir="$1"
  local mode="$2"
  local safety_log="${run_dir}/action_safety.jsonl"
  local out_json="${run_dir}/timing_summary.json"
  "${PYTHON_BIN}" - "$safety_log" "$out_json" "$mode" <<'PY'
import json, math, statistics, sys
from pathlib import Path

safety_log = Path(sys.argv[1])
out_json = Path(sys.argv[2])
mode = sys.argv[3]

def flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (int, float)) and math.isfinite(float(obj)):
        out[prefix] = float(obj)

rows = []
if safety_log.is_file():
    for line in safety_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        flat = {}
        flatten("", record.get("server_timing", {}), flat)
        rows.append(flat)

keys = sorted({key for row in rows for key in row})
summary = {"mode": mode, "records": len(rows), "metrics": {}}
for key in keys:
    values = [row[key] for row in rows if key in row and math.isfinite(row[key])]
    if not values:
        continue
    values_sorted = sorted(values)
    p50 = statistics.median(values_sorted)
    p95 = values_sorted[min(len(values_sorted) - 1, int(math.ceil(0.95 * len(values_sorted))) - 1)]
    summary["metrics"][key] = {
        "mean": statistics.fmean(values_sorted),
        "p50": p50,
        "p95": p95,
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }
out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

interesting = [
    "infer_ms",
    "prev_total_ms",
    "policy.checkpoint_policy_total_ms",
    "policy.policy_policy_act_total_ms",
    "policy.policy_semantic_encode_ms",
    "policy.policy_picf_observe_ms",
    "policy.policy_action_sample_ms",
    "policy.policy_picf_finalize_ms",
    "policy.policy_picf_observe_visual_maps_ms",
    "policy.policy_picf_observe_point_features_ms",
    "policy.policy_picf_observe_token_field_ms",
    "policy.policy_picf_observe_anchor_graph_ms",
    "policy.policy_picf_observe_object_explanation_ms",
    "policy.policy_picf_observe_posterior_update_ms",
]
print(f"TIMING_SUMMARY mode={mode} records={len(rows)}")
for key in interesting:
    item = summary["metrics"].get(key)
    if item:
        print(
            f"{key}: mean={item['mean']:.2f}ms p50={item['p50']:.2f}ms "
            f"p95={item['p95']:.2f}ms min={item['min']:.2f}ms max={item['max']:.2f}ms"
        )
PY
}

run_one() {
  local mode="$1"
  local checkpoint="$2"
  local port="$3"
  local tag="$4"
  local run_dir="${OUT_ROOT}/${tag}"
  local tmp_dir="${TMP_ROOT}/${tag}"
  mkdir -p "${run_dir}/logs" "${run_dir}/eval_logs" "${tmp_dir}/eval_logs"

  echo "===== ${tag}: starting server mode=${mode} checkpoint=${checkpoint}"
  cleanup_processes
  cd "${REPO_ROOT}"
  (
    export CUDA_VISIBLE_DEVICES="${SERVER_GPU}"
    export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src"
    exec "${PYTHON_BIN}" scripts/serve_picf_policy.py \
      --checkpoint "${checkpoint}" \
      --device cuda:0 \
      --port "${port}" \
      --picf-mode "${mode}" \
      --picf-observe-interval "${PICF_OBSERVE_INTERVAL}"
  ) > "${run_dir}/logs/server.log" 2>&1 &
  local server_pid=$!
  echo "${server_pid}" > "${run_dir}/server.pid"

  wait_health "${port}" 420

  echo "===== ${tag}: starting evaluator num_sequences=${NUM_SEQUENCES}"
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
      --action_safety_log "${run_dir}/action_safety.jsonl" \
      --action_clip 1.0 \
      --calvin_agent_root "${CALVIN_AGENT_ROOT}"
  ) > "${run_dir}/logs/eval.log" 2>&1 || true

  cp -f "${run_dir}/logs/eval.log" "${run_dir}/eval.log" 2>/dev/null || true
  cp -f "${tmp_dir}/eval_logs"/* "${run_dir}/eval_logs"/ 2>/dev/null || true
  summarize_timing "${run_dir}" "${mode}" | tee "${run_dir}/logs/timing_summary.txt"

  echo "===== ${tag}: stopping server pid=${server_pid}"
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}

compare_ratio() {
  "${PYTHON_BIN}" - "${OUT_ROOT}/pi05_ablated/timing_summary.json" "${OUT_ROOT}/picf_enabled/timing_summary.json" "${OUT_ROOT}/latency_ratio.json" <<'PY'
import json, math, sys
from pathlib import Path
base = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
picf = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
out = Path(sys.argv[3])
keys = ["infer_ms", "policy.checkpoint_policy_total_ms", "policy.policy_policy_act_total_ms"]
payload = {}
for key in keys:
    b = base.get("metrics", {}).get(key, {}).get("p50")
    p = picf.get("metrics", {}).get(key, {}).get("p50")
    if isinstance(b, (int, float)) and isinstance(p, (int, float)) and b > 0:
        payload[key] = {"pi05_p50_ms": b, "picf_p50_ms": p, "ratio": p / b}
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print("LATENCY_RATIO")
for key, item in payload.items():
    print(f"{key}: pi05={item['pi05_p50_ms']:.2f}ms picf={item['picf_p50_ms']:.2f}ms ratio={item['ratio']:.3f}x")
PY
}

main() {
  cleanup_processes
  mkdir -p "${OUT_ROOT}/logs"
  {
    echo "started_at=$(date -Is)"
    echo "repo=${REPO_ROOT}"
    echo "pi05_checkpoint=${PI05_CHECKPOINT}"
    echo "picf_checkpoint=${PICF_CHECKPOINT}"
    echo "num_sequences=${NUM_SEQUENCES}"
    echo "server_gpu=${SERVER_GPU}"
    echo "eval_gpu=${EVAL_GPU}"
    echo "picf_observe_interval=${PICF_OBSERVE_INTERVAL}"
  } > "${OUT_ROOT}/run_config.txt"

  run_one "ablated" "${PI05_CHECKPOINT}" "${BASE_PORT}" "pi05_ablated"
  run_one "enabled" "${PICF_CHECKPOINT}" "$((BASE_PORT + 1))" "picf_enabled"
  compare_ratio | tee "${OUT_ROOT}/logs/latency_ratio.txt"
  echo "finished_at=$(date -Is)" >> "${OUT_ROOT}/run_config.txt"
  cleanup_processes
}

main "$@"
