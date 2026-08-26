#!/usr/bin/env bash
set -euo pipefail

REPO=${PICF_ADR208_REPO:-/mnt/picf-next/adr208/source-freezes/vjepa2-ac-arm-b-v3}
PYTHON=${PICF_SYSTEM_PYTHON:-python3}
MANIFEST=$REPO/adr208/vjepa2_ac_exact_source_manifest.json
DESTINATION=${PICF_VJEPA2_AC_CHECKPOINT:-/mnt/picf-next/models/foundation/vjepa2_ac/vjepa2-ac-vitg.pt}

fail() {
  printf 'V-JEPA2-AC checkpoint fetch failed: %s\n' "$*" >&2
  exit 2
}

[[ -f "$MANIFEST" ]] || fail "missing source manifest: $MANIFEST"
mapfile -t identity < <(
  "$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))["upstream"]
print(manifest["checkpoint_url"])
print(manifest["checkpoint_http_content_length"])
print(manifest["checkpoint_sha256"])
PY
)
[[ ${#identity[@]} == 3 ]] || fail "incomplete checkpoint identity"
url=${identity[0]}
expected_bytes=${identity[1]}
expected_sha=${identity[2]}
[[ "$expected_bytes" =~ ^[1-9][0-9]*$ ]] || fail "invalid byte count"
[[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA-256"

mkdir -p "$(dirname "$DESTINATION")"
lock=$DESTINATION.fetch-lock
mkdir "$lock" 2>/dev/null || fail "another checkpoint fetch owns $lock"
trap 'rmdir "$lock"' EXIT

verify() {
  local path=$1
  [[ -f "$path" && ! -L "$path" ]] || return 1
  [[ $(stat -c '%s' "$path") == "$expected_bytes" ]] || return 1
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected_sha" ]]
}

if [[ -e "$DESTINATION" ]]; then
  verify "$DESTINATION" || fail "existing final checkpoint has the wrong identity"
  printf 'verified existing checkpoint: %s\n' "$DESTINATION"
  exit 0
fi

partial=$DESTINATION.partial
if [[ -e "$partial" && ( ! -f "$partial" || -L "$partial" ) ]]; then
  fail "partial path is not a regular file"
fi
if [[ -f "$partial" && $(stat -c '%s' "$partial") -gt $expected_bytes ]]; then
  fail "partial checkpoint is larger than the published object"
fi

curl \
  --location \
  --fail \
  --silent \
  --show-error \
  --retry 100 \
  --retry-all-errors \
  --retry-delay 5 \
  --connect-timeout 30 \
  --continue-at - \
  --output "$partial" \
  "$url"

verify "$partial" || fail "completed download failed byte-count or SHA validation"
mv "$partial" "$DESTINATION"
printf '%s  %s\n' "$expected_sha" "$DESTINATION" >"$DESTINATION.sha256"
printf 'published checkpoint: %s\n' "$DESTINATION"
