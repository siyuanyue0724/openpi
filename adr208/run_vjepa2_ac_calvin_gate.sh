#!/usr/bin/env bash
set -euo pipefail

# Frozen V-JEPA2-AC Arm B gate. This script evaluates a released world-model
# sidecar only; it never starts PICF policy training.

REPO=${PICF_ADR208_REPO:-/mnt/picf-next/adr208/source-freezes/vjepa2-ac-arm-b-v3}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
CHECKPOINT=${PICF_VJEPA2_AC_CHECKPOINT:-/mnt/picf-next/models/foundation/vjepa2_ac/vjepa2-ac-vitg.pt}
RUN_ROOT=${PICF_ADR208_RUN_ROOT:-/mnt/picf-next/adr208/runs}
CLIP_COUNT=${PICF_VJEPA2_AC_CLIP_COUNT:-16}
CAMERA=${PICF_VJEPA2_AC_CAMERA:-rgb_static}
CONTROL_SEED=${PICF_VJEPA2_AC_CONTROL_SEED:-239}
SHARD_INDEX=${PICF_VJEPA2_AC_SHARD_INDEX:-0}
SHARD_COUNT=${PICF_VJEPA2_AC_SHARD_COUNT:-1}
SKIP_LOCAL_TESTS=${PICF_VJEPA2_AC_SKIP_LOCAL_TESTS:-0}
EXPECTED_BYTES=11760743310
MANIFEST=$REPO/adr208/vjepa2_ac_exact_source_manifest.json

fail() {
  printf 'ADR-208 Arm B preflight failed: %s\n' "$*" >&2
  exit 2
}

[[ -x "$PYTHON" ]] || fail "missing runtime Python: $PYTHON"
[[ -d "$REPO" ]] || fail "missing immutable source freeze: $REPO"
[[ -d "$DATASET" ]] || fail "missing CALVIN split: $DATASET"
[[ -f "$DATASET_MANIFEST" ]] || fail "missing CALVIN content manifest: $DATASET_MANIFEST"
[[ -f "$CHECKPOINT" && ! -L "$CHECKPOINT" ]] || fail "missing regular checkpoint: $CHECKPOINT"
[[ -f "$MANIFEST" ]] || fail "missing ADR-208 source manifest: $MANIFEST"
[[ "$CLIP_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "clip count must be positive"
[[ "$CONTROL_SEED" =~ ^[0-9]+$ ]] || fail "control seed must be non-negative"
[[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || fail "shard index must be non-negative"
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "shard count must be positive"
(( SHARD_INDEX < SHARD_COUNT )) || fail "shard index must be below shard count"
[[ "$SKIP_LOCAL_TESTS" == 0 || "$SKIP_LOCAL_TESTS" == 1 ]] || fail "invalid test switch"
[[ "$CAMERA" == rgb_static || "$CAMERA" == rgb_gripper ]] || fail "unsupported camera"
[[ "${CUDA_VISIBLE_DEVICES:-0}" =~ ^[0-9]+$ ]] || fail "gate requires one explicit GPU index"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

actual_bytes=$(stat -c '%s' "$CHECKPOINT")
[[ "$actual_bytes" == "$EXPECTED_BYTES" ]] || fail \
  "checkpoint byte count $actual_bytes != $EXPECTED_BYTES"

checkpoint_sha=$(
  "$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
value = manifest["upstream"]["checkpoint_sha256"]
if not isinstance(value, str) or len(value) != 64:
    raise SystemExit("manifest has no frozen checkpoint SHA-256")
print(value)
PY
) || fail "checkpoint identity is not frozen in the manifest"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REPO/src:$REPO

"$PYTHON" - <<'PY'
import importlib
import json

import torch

required = ("torchvision", "timm", "einops")
versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
for name in required:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")
if not versions["torch"].startswith("2.8.0"):
    raise SystemExit(f"unvalidated torch ABI: {versions['torch']}")
if not versions["torchvision"].startswith("0.23.0"):
    raise SystemExit(f"unvalidated torchvision ABI: {versions['torchvision']}")
if versions["timm"] != "1.0.20" or versions["einops"] != "0.8.1":
    raise SystemExit(f"unvalidated donor dependencies: {versions}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ADR-208 Arm B requires exactly one visible CUDA device")
versions["gpu"] = torch.cuda.get_device_name(0)
versions["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
if not versions["bf16_supported"]:
    raise SystemExit("released bfloat16 evaluation requires a bf16-capable GPU")
print(json.dumps(versions, sort_keys=True))
PY

if [[ "$SKIP_LOCAL_TESTS" == 0 ]]; then
  "$PYTHON" -m pytest -q \
    "$REPO/tests/test_calvin_vjepa2_ac.py" \
    "$REPO/tests/test_probe_calvin_vjepa2_ac.py" \
    "$REPO/tests/test_vjepa2_ac_source_fidelity.py"
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir=$RUN_ROOT/vjepa2-ac-arm-b-$timestamp-$CAMERA-seed$CONTROL_SEED-shard$SHARD_INDEX-of-$SHARD_COUNT
mkdir -p "$run_dir"
report=$run_dir/causal-gate.json

cat >"$run_dir/input-receipt.json" <<EOF
{
  "camera": "$CAMERA",
  "checkpoint_bytes": $actual_bytes,
  "checkpoint_sha256": "$checkpoint_sha",
  "clip_count": $CLIP_COUNT,
  "control_seed": $CONTROL_SEED,
  "dataset_manifest": "$DATASET_MANIFEST",
  "dataset_split": "$DATASET",
  "repo": "$REPO",
  "shard_count": $SHARD_COUNT,
  "shard_index": $SHARD_INDEX
}
EOF

"$PYTHON" "$REPO/tools/probe_calvin_vjepa2_ac.py" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$checkpoint_sha" \
  --camera "$CAMERA" \
  --clip-count "$CLIP_COUNT" \
  --seed "$CONTROL_SEED" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --device cuda:0 \
  --output "$report" \
  >"$run_dir/probe.stdout.json"

"$PYTHON" - "$report" "$run_dir/verdict.json" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
verdict_path = Path(sys.argv[2])
report = json.loads(report_path.read_text())
shard_count = report["selection"]["shard_count"]
passed = report["aggregate"]["causal_signal_pass"] is True if shard_count == 1 else False
verdict = {
    "arm": "ADR-208-B",
    "authorizes_arm_c": passed,
    "authorizes_policy_training": False,
    "causal_signal_pass": passed,
    "shard_complete": True,
    "shard_count": shard_count,
    "shard_index": report["selection"]["shard_index"],
    "report": str(report_path),
}
verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, sort_keys=True))
raise SystemExit(0 if passed or shard_count > 1 else 3)
PY
