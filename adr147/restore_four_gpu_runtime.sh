#!/usr/bin/env bash
set -euo pipefail

HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr147/four_gpu_handoff_20260808}
REPO=${PICF_REPO:-/mnt/picf-next/worktrees/adr147-fourgpu-candidate-20260808}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147}
RUNTIME=${PICF_RUNTIME_ROOT:-/opt/picf-runtime-restore-probe-94305690cafb}
ARCHIVE=${PICF_RUNTIME_ARCHIVE:-/mnt/picf-next/runtime-archives/picf-runtime-restore-probe-94305690cafb-20260808.tar}

for path in "$HANDOFF" "$REPO" "$SOURCE"; do
  [[ -d "$path" ]] || {
    echo "required persistent directory is absent: $path" >&2
    exit 1
  }
done

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq 4 ]] || {
  echo "four-GPU handoff requires exactly four visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "four-GPU handoff requires A100 40GB; observed: $row" >&2
    exit 1
  }
done

[[ -f "$ARCHIVE" && -f "$ARCHIVE.sha256" ]] || {
  echo "audited runtime archive or receipt is absent" >&2
  exit 1
}
sha256sum -c "$ARCHIVE.sha256"
[[ "$RUNTIME" == /opt/picf-runtime-* && "$RUNTIME" != /opt/picf-runtime- ]] || {
  echo "runtime restore target must be one managed /opt/picf-runtime-* path" >&2
  exit 1
}
[[ ! -L "$RUNTIME" ]] || {
  echo "runtime restore target must not be a symlink" >&2
  exit 1
}

# Package metadata alone cannot detect modified site-packages.  Recreate the
# runtime from the pinned archive for every scientific launch so the bytes used
# by LBOT/training cannot inherit mutable state from an earlier cloud session.
RUNTIME_PARENT=$(dirname "$RUNTIME")
RUNTIME_STAGE=$(mktemp -d "$RUNTIME_PARENT/.picf-runtime-restore.XXXXXX")
cleanup() {
  rm -rf "$RUNTIME_STAGE"
}
trap cleanup EXIT
tar -C "$RUNTIME_STAGE" -xf "$ARCHIVE"
mapfile -t EXTRACTED_ROOTS < <(find "$RUNTIME_STAGE" -mindepth 1 -maxdepth 1 -type d)
[[ ${#EXTRACTED_ROOTS[@]} -eq 1 && -x "${EXTRACTED_ROOTS[0]}/bin/python" ]] || {
  echo "runtime archive does not contain one executable environment" >&2
  exit 1
}
rm -rf "$RUNTIME"
mv -T "${EXTRACTED_ROOTS[0]}" "$RUNTIME"
trap - EXIT
rm -rf "$RUNTIME_STAGE"

EXPECTED_FREEZE_SHA=$(awk '$2 ~ /pip-freeze.txt$/ {print $1}' "$HANDOFF/runtime_receipts.sha256")
OBSERVED_FREEZE_SHA=$("$RUNTIME/bin/python" -m pip freeze --all | sha256sum)
OBSERVED_FREEZE_SHA=${OBSERVED_FREEZE_SHA%% *}
[[ -n "$EXPECTED_FREEZE_SHA" && "$OBSERVED_FREEZE_SHA" == "$EXPECTED_FREEZE_SHA" ]] || {
  echo "restored runtime package set differs from the accepted environment" >&2
  exit 1
}

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src:$REPO" "$RUNTIME/bin/python" - <<'PY'
import torch
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy

if torch.__version__ != "2.8.0+cu128":
    raise RuntimeError(f"unexpected torch runtime: {torch.__version__}")
if not torch.distributed.is_available():
    raise RuntimeError("torch distributed runtime is unavailable")
if not torch.distributed.is_nccl_available():
    raise RuntimeError("NCCL distributed backend is unavailable")
print(f"runtime=PASS policy={LingbotVlaV2Policy.__name__} torch={torch.__version__}")
PY

echo "four-GPU runtime preflight=PASS"
