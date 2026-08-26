#!/usr/bin/env bash
set -euo pipefail

RUNTIME=${PICF_RUNTIME_ROOT:-/opt/picf-runtime-restore-probe-94305690cafb}
ARCHIVE=${PICF_RUNTIME_ARCHIVE:-/mnt/picf-next/runtime-archives/picf-runtime-restore-probe-94305690cafb-20260808.tar}
EXPECTED_BYTES=10136064000
EXPECTED_SHA256=46c480491e9b491c573fbe94eb0f4f3bcfdb7b2f9707658f4e1361152304ff82

fail() {
  printf 'ADR-208 runtime restore failed: %s\n' "$*" >&2
  exit 2
}

[[ -f "$ARCHIVE" && ! -L "$ARCHIVE" ]] || fail "missing regular runtime archive: $ARCHIVE"
[[ $(stat -c '%s' "$ARCHIVE") == "$EXPECTED_BYTES" ]] || fail "runtime archive size changed"
[[ $(sha256sum "$ARCHIVE" | awk '{print $1}') == "$EXPECTED_SHA256" ]] || fail \
  "runtime archive SHA-256 changed"
[[ "$RUNTIME" == /opt/picf-runtime-* && "$RUNTIME" != /opt/picf-runtime- ]] || fail \
  "runtime target must be a managed /opt/picf-runtime-* path"
[[ ! -L "$RUNTIME" ]] || fail "runtime target must not be a symlink"

parent=$(dirname "$RUNTIME")
stage=$(mktemp -d "$parent/.picf-adr208-runtime.XXXXXX")
cleanup() {
  rm -rf "$stage"
}
trap cleanup EXIT
tar -C "$stage" -xf "$ARCHIVE"
mapfile -t roots < <(find "$stage" -mindepth 1 -maxdepth 1 -type d)
[[ ${#roots[@]} == 1 && -x "${roots[0]}/bin/python3.12" ]] || fail \
  "runtime archive does not contain one executable Python environment"
rm -rf "$RUNTIME"
mv -T "${roots[0]}" "$RUNTIME"
trap - EXIT
rm -rf "$stage"

"$RUNTIME/bin/python3.12" - <<'PY'
import torch

if torch.__version__ != "2.8.0+cu128":
    raise SystemExit(f"unexpected restored torch build: {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("restored runtime cannot access CUDA")
print(f"restored_runtime=PASS torch={torch.__version__}")
PY
