#!/usr/bin/env bash
set -euo pipefail

DESTINATION=${PICF_ADR225_REPO:-/mnt/picf-next/adr225/source-freezes/pretrained-object-memory-v1}
[[ "$DESTINATION" == /mnt/* && ! -e "$DESTINATION" && ! -L "$DESTINATION" ]] || {
  echo "ADR-225 source freeze must be one absent persistent path beneath /mnt" >&2
  exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SOURCE=${PICF_SOURCE_REPO:-$(cd "$SCRIPT_DIR/.." && pwd -P)}
[[ -d "$SOURCE/.git" && ! -L "$SOURCE" ]] || {
  echo "ADR-225 source must be one direct Git working tree" >&2
  exit 1
}
[[ -z "$(git -C "$SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-225 source must be clean before an immutable freeze" >&2
  exit 1
}
command -v rsync >/dev/null || {
  echo "rsync is required to construct the immutable source freeze" >&2
  exit 1
}

PARENT=$(dirname "$DESTINATION")
mkdir -p "$PARENT"
STAGING=$(mktemp -d "$PARENT/.pretrained-object-memory-v1.XXXXXX")
cleanup() {
  chmod -R u+w "$STAGING" 2>/dev/null || true
  rm -rf "$STAGING"
}
trap cleanup EXIT

rsync -a --delete \
  --exclude='/.git' \
  --exclude='/.venv*/' \
  --exclude='/.pytest_cache/' \
  --exclude='/.ruff_cache/' \
  --exclude='/.audit-logs/' \
  --exclude='/.codex-transfer/' \
  --exclude='/.tmp*/' \
  --exclude='/source-freeze.receipt.json' \
  --exclude='/artifacts/' \
  --exclude='/evidence/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$SOURCE/" "$STAGING/"

python3 - "$SOURCE" "$STAGING" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

source = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
rows = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    relative = path.relative_to(root).as_posix()
    if relative == "source-freeze.receipt.json":
        continue
    rows.append(
        {
            "bytes": path.stat().st_size,
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema": "picf-next.adr225-source-freeze/v1",
    "source_branch": subprocess.check_output(
        ["git", "-C", str(source), "branch", "--show-current"], text=True
    ).strip(),
    "source_head": subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip(),
    "file_count": len(rows),
    "files_sha256": hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest(),
    "files": rows,
}
destination = root / "source-freeze.receipt.json"
with destination.open("x", encoding="ascii") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY

git -C "$STAGING" init -q
git -C "$STAGING" config user.name picf-freeze
git -C "$STAGING" config user.email picf-freeze@invalid
git -C "$STAGING" add -A
git -C "$STAGING" commit -q -m "Freeze ADR-225 pretrained object memory v1"
git -C "$STAGING" status --porcelain=v1 | grep -q . && {
  echo "ADR-225 source freeze is not clean after its atomic commit" >&2
  exit 1
}
chmod -R a-w "$STAGING"
mv -T "$STAGING" "$DESTINATION"
trap - EXIT

echo "ADR-225 source freeze commit=$(git -C "$DESTINATION" rev-parse HEAD)"
echo "ADR-225 source freeze path=$DESTINATION"
