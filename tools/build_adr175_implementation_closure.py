#!/usr/bin/env python3
"""Publish a content-addressed file closure for one frozen ADR-175 source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".artifacts",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _included_files(root: Path, output: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path == output:
            continue
        if path.is_symlink():
            raise ValueError(f"implementation source closure contains a symlink: {relative}")
        if path.is_file():
            result.append(relative)
    if not result:
        raise ValueError("implementation source closure is empty")
    return tuple(sorted(result, key=lambda value: value.as_posix()))


def main() -> None:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("implementation closure output must be inside its source root") from error
    files = []
    for relative in _included_files(root, output):
        path = root / relative
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": relative.as_posix(),
                "sha256": _file_sha256(path),
            }
        )
    semantic = {
        "files": files,
        "schema": "picf-next.adr175-implementation-closure.v1",
        "source_root": str(root),
    }
    payload = {"artifact_sha256": _canonical_sha256(semantic), **semantic}
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    print(
        json.dumps(
            {
                "artifact_sha256": payload["artifact_sha256"],
                "file_count": len(files),
                "file_sha256": hashlib.sha256(data).hexdigest(),
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
