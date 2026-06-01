import json
import pathlib

out = pathlib.Path("/mnt/picf_exact_window_probes/e14_action_nan_decompose_20260531")
for name in ["as_called", "no_context"]:
    p = out / f"{name}.json"
    print("---", name, p.exists(), p.stat().st_size if p.exists() else None)
    if p.exists():
        d = json.loads(p.read_text())
        print("loaded", d.get("loaded_step"), "debug_records_len", len(d.get("debug_records", [])))
        os = d.get("output_stats", {})
        for key in [
            "loss_total",
            "loss_action",
            "loss_action_default_equiv",
            "loss_action_active7",
            "loss_total_minus_action",
            "loss_anchor_object_pull",
            "pi_prefix_nonfinite_count",
            "pi_prefix_scale_mean",
        ]:
            print(key, os.get(key))
    print("logtail")
    lp = out / f"{name}.log"
    print(lp.read_text()[-4000:] if lp.exists() else "missing")
