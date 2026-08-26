#!/usr/bin/env bash
set -euo pipefail

SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DESTINATION=${1:-}

if [[ -z "$DESTINATION" ]]; then
  printf 'usage: %s ABSOLUTE_DESTINATION\n' "$0" >&2
  exit 2
fi
if [[ "$DESTINATION" != /* ]]; then
  printf 'destination must be absolute: %s\n' "$DESTINATION" >&2
  exit 2
fi
if [[ -e "$DESTINATION" ]]; then
  printf 'refusing to replace an existing source freeze: %s\n' "$DESTINATION" >&2
  exit 2
fi

stage=$DESTINATION.partial.$$
cleanup() {
  rm -rf "$stage"
}
trap cleanup EXIT
mkdir -p "$stage/docs" "$stage/references/source_archives" "$stage/references/patches"

copy_tree() {
  local source_path=$1
  local target_path=$2
  mkdir -p "$target_path"
  rsync -a \
    --exclude '__pycache__/' \
    --exclude '*.py[co]' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    "$source_path/" "$target_path/"
}

copy_tree "$SOURCE/src" "$stage/src"
copy_tree "$SOURCE/tests" "$stage/tests"
copy_tree "$SOURCE/tools" "$stage/tools"
copy_tree "$SOURCE/adr208" "$stage/adr208"
cp "$SOURCE/pyproject.toml" "$stage/pyproject.toml"
cp "$SOURCE/README.md" "$stage/README.md"
cp \
  "$SOURCE/docs/208_ACTION_MEDIATED_OBJECT_POSTERIOR_REDIRECTION_20260823.md" \
  "$stage/docs/208_ACTION_MEDIATED_OBJECT_POSTERIOR_REDIRECTION_20260823.md"
cp \
  "$SOURCE/references/source_archives/vjepa2-204698b45b3712590f06245fbfba32d3be539812.tar.gz" \
  "$stage/references/source_archives/"
cp \
  "$SOURCE/references/patches/vjepa2_ac_official_download_url.patch" \
  "$stage/references/patches/"

(
  cd "$stage"
  find . -type f ! -name SOURCE_FREEZE.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum >SOURCE_FREEZE.sha256
  sha256sum -c SOURCE_FREEZE.sha256 >/dev/null
)

chmod -R a-w "$stage"
mv "$stage" "$DESTINATION"
trap - EXIT
printf '%s\n' "$DESTINATION"
