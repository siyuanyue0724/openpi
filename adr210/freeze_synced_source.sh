#!/usr/bin/env bash
set -euo pipefail

SOURCE=${PICF_ADR210_SYNCED_SOURCE:-/mnt/picf-next/adr209/worktree-local-sync}
DESTINATION=${PICF_ADR210_REPO:-/mnt/picf-next/adr207/source-freezes/native-query-posterior-v20}
SOURCE_BRANCH=${PICF_ADR210_SOURCE_BRANCH:?PICF_ADR210_SOURCE_BRANCH is required}
SOURCE_HEAD=${PICF_ADR210_SOURCE_HEAD:?PICF_ADR210_SOURCE_HEAD is required}

[[ -d "$SOURCE" && ! -L "$SOURCE" ]] || {
  echo "ADR-210 synchronized source is absent or indirect" >&2
  exit 1
}
[[ "$DESTINATION" == /mnt/* && ! -e "$DESTINATION" && ! -L "$DESTINATION" ]] || {
  echo "ADR-210 destination must be one absent persistent path beneath /mnt" >&2
  exit 2
}
[[ "$SOURCE_HEAD" =~ ^[0-9a-f]{40}$ ]] || {
  echo "ADR-210 source HEAD must be one lowercase Git object ID" >&2
  exit 2
}

PARENT=$(dirname "$DESTINATION")
mkdir -p "$PARENT"
STAGING=$(mktemp -d "$PARENT/.native-query-posterior-v20.XXXXXX")
cleanup() {
  chmod -R u+w "$STAGING" 2>/dev/null || true
  rm -rf "$STAGING"
}
trap cleanup EXIT

cp -a "$SOURCE/." "$STAGING/"
rm -rf \
  "$STAGING/.git" \
  "$STAGING/.pytest_cache" \
  "$STAGING/.ruff_cache" \
  "$STAGING/.audit-logs" \
  "$STAGING/.codex-transfer" \
  "$STAGING/artifacts" \
  "$STAGING/evidence" \
  "$STAGING/source-freeze.receipt.json"
find "$STAGING" -depth \
  \( -name '.venv*' -o -name '.tmp*' -o -name '__pycache__' \) \
  -type d -exec rm -rf {} +
find "$STAGING" -type f -name '*.pyc' -delete

python3 - "$STAGING" "$SOURCE" "$SOURCE_BRANCH" "$SOURCE_HEAD" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
rows = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    relative = path.relative_to(root).as_posix()
    rows.append(
        {
            "bytes": path.stat().st_size,
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema": "picf-next.adr210-source-freeze/v1",
    "source_mode": "content-addressed-synchronized-non-git-tree",
    "source_path": str(source),
    "source_branch": sys.argv[3],
    "source_head": sys.argv[4],
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
git -C "$STAGING" commit -q -m "Freeze ADR-210 causal-warm action gate"
[[ -z "$(git -C "$STAGING" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-210 source freeze is not clean after its atomic commit" >&2
  exit 1
}
chmod -R a-w "$STAGING"
mv -T "$STAGING" "$DESTINATION"
trap - EXIT

echo "ADR-210 source freeze commit=$(git -C "$DESTINATION" rev-parse HEAD)"
echo "ADR-210 source freeze path=$DESTINATION"
