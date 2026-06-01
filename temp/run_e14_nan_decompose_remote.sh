#!/usr/bin/env bash
set -euo pipefail
cd /root/openpi_probe_current_20260529

OUT=/mnt/picf_exact_window_probes/e14_action_nan_decompose_20260531
mkdir -p "$OUT"

ARGS=/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531/args.json
CKPT=/mnt/checkpoints/picf_core/picf_core/picf_a7_stepindexed_from9100_lrcontinuity_prefixfusion_h30k_20260531/9200
WIN=/mnt/picf_exact_window_probes/e14a_actionreadout_trainmode4_20260531/windows4.jsonl

CUDA_VISIBLE_DEVICES=0 python3.12 scripts/picf_action_nan_decompose_probe.py \
  --args-json "$ARGS" \
  --checkpoint "$CKPT" \
  --window-jsonl "$WIN" \
  --output-json "$OUT/as_called.json" \
  --device cuda:0 \
  --mode train \
  --context-mode as_called \
  > "$OUT/as_called.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python3.12 scripts/picf_action_nan_decompose_probe.py \
  --args-json "$ARGS" \
  --checkpoint "$CKPT" \
  --window-jsonl "$WIN" \
  --output-json "$OUT/no_context.json" \
  --device cuda:0 \
  --mode train \
  --context-mode no_context \
  > "$OUT/no_context.log" 2>&1 &
PID1=$!

wait "$PID0"
echo as_called_done
wait "$PID1"
echo no_context_done

python3.12 - <<'PY'
import json
import pathlib

out = pathlib.Path("/mnt/picf_exact_window_probes/e14_action_nan_decompose_20260531")
for name in ["as_called", "no_context"]:
    p = out / f"{name}.json"
    print("---", name, "exists", p.exists())
    if not p.exists():
        log_path = out / f"{name}.log"
        print(log_path.read_text()[-2000:] if log_path.exists() else "missing log")
        continue
    d = json.loads(p.read_text())
    rec = d["debug_records"][0]
    for key in [
        "input_action_chunk_target",
        "input_extra_prefix_tokens",
        "input_extra_action_context_tokens",
        "target",
        "u_t",
        "prefix_embs_pre_dtype",
        "suffix_embs_pre_adapter",
        "suffix_embs_post_adapter",
        "att_2d_masks_4d",
        "suffix_out_all",
        "suffix_out_action_horizon",
        "v_t",
        "predicted_chunk",
    ]:
        stats = rec.get(key, {})
        print(
            key,
            "finite=", stats.get("finite_all"),
            "nan=", stats.get("nan_count"),
            "inf=", stats.get("inf_count"),
            "rms=", stats.get("rms"),
            "shape=", stats.get("shape"),
        )
    print(
        "loss_total", rec.get("loss_total"),
        "loss_pos", rec.get("loss_pos"),
        "loss_rot", rec.get("loss_rot"),
        "loss_grip", rec.get("loss_grip"),
    )
    print("adapter_metrics", {k: v.get("finite_all") for k, v in rec.get("adapter_metrics", {}).items()})
PY
