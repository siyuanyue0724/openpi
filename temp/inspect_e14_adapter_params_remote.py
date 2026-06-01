import json
import pathlib
import pprint

p = pathlib.Path("/mnt/picf_exact_window_probes/e14_action_nan_decompose_20260531/as_called.json")
d = json.loads(p.read_text())
rec = d["debug_records"][0]
pprint.pp(rec["adapter_params"])
print(
    "output nonfinite keys",
    [k for k, v in d["output_stats"].items() if isinstance(v, dict) and not v.get("finite_all", True)][:120],
)
