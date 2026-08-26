#!/usr/bin/env python3
"""Fail closed if the pinned PMT source snapshot changes in any way."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "references/upstream/pmt-442a8113.receipt.json"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snapshot_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def main() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    snapshot = ROOT / receipt["snapshot_path"]
    files = _snapshot_files(snapshot)
    if len(files) != receipt["tracked_file_count"]:
        raise RuntimeError(
            f"PMT snapshot file count changed: {len(files)} != "
            f"{receipt['tracked_file_count']}"
        )

    content_lines: list[bytes] = []
    mode_lines: list[bytes] = []
    for path in files:
        relative = path.relative_to(snapshot).as_posix()
        content_lines.append(f"{_sha256(path.read_bytes())}  {relative}\n".encode())
        mode = stat.S_IMODE(path.stat().st_mode)
        mode_lines.append(f"{mode:o} {relative}\n".encode())

    content_digest = _sha256(b"".join(content_lines))
    mode_digest = _sha256(b"".join(sorted(mode_lines)))
    expected_content = receipt["content_manifest_sha256"]
    expected_mode = receipt["mode_manifest_sha256"]
    if content_digest != expected_content:
        raise RuntimeError(
            f"PMT snapshot content changed: {content_digest} != {expected_content}"
        )
    if mode_digest != expected_mode:
        raise RuntimeError(f"PMT snapshot modes changed: {mode_digest} != {expected_mode}")

    print(
        json.dumps(
            {
                "status": "pass",
                "source_commit": receipt["source_commit"],
                "file_count": len(files),
                "content_manifest_sha256": content_digest,
                "mode_manifest_sha256": mode_digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
