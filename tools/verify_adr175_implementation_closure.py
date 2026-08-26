#!/usr/bin/env python3
"""Verify every file declared by one ADR-175 implementation closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("implementation closure path must be nonempty text")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"implementation closure path is unsafe: {value!r}")
    return Path(*pure.parts)


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _actual_source_files(root: Path, manifest: Path) -> set[Path]:
    result: set[Path] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path == manifest:
            continue
        if path.is_symlink():
            raise ValueError(f"implementation source tree contains a symlink: {relative}")
        if path.is_file():
            result.add(relative)
    return result


def main() -> None:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    manifest = args.manifest.resolve(strict=True)
    manifest_bytes = manifest.read_bytes()
    observed_manifest_file_sha256 = _sha256_bytes(manifest_bytes)
    if observed_manifest_file_sha256 != _require_sha256(
        args.manifest_file_sha256,
        "implementation closure manifest file",
    ):
        raise ValueError("implementation closure manifest file SHA-256 differs")
    try:
        payload = json.loads(manifest_bytes.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("implementation closure manifest is invalid ASCII JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_sha256",
        "files",
        "schema",
        "source_root",
    }:
        raise ValueError("implementation closure manifest fields changed")
    if payload["schema"] != "picf-next.adr175-implementation-closure.v1":
        raise ValueError("implementation closure schema changed")
    artifact_sha256 = _require_sha256(
        payload["artifact_sha256"],
        "implementation closure artifact",
    )
    semantic = dict(payload)
    semantic.pop("artifact_sha256")
    if _canonical_sha256(semantic) != artifact_sha256:
        raise ValueError("implementation closure semantic SHA-256 differs")
    if artifact_sha256 != _require_sha256(
        args.expected_artifact_sha256,
        "expected implementation closure artifact",
    ):
        raise ValueError("implementation closure artifact differs from its launch pin")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("implementation closure file inventory is empty")
    seen: set[Path] = set()
    total_bytes = 0
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"bytes", "path", "sha256"}:
            raise ValueError("implementation closure file entry fields changed")
        relative = _safe_relative_path(entry["path"])
        if relative in seen:
            raise ValueError(f"implementation closure repeats a path: {relative}")
        seen.add(relative)
        expected_bytes = entry["bytes"]
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ValueError("implementation closure byte size must be an integer")
        expected_sha256 = _require_sha256(entry["sha256"], f"{relative} SHA-256")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"implementation closure file is absent or not regular: {relative}")
        observed_bytes = path.stat().st_size
        if observed_bytes != expected_bytes:
            raise ValueError(f"implementation closure byte size differs: {relative}")
        if _file_sha256(path) != expected_sha256:
            raise ValueError(f"implementation closure file SHA-256 differs: {relative}")
        total_bytes += observed_bytes
    actual_files = _actual_source_files(root, manifest)
    if actual_files != seen:
        undeclared = sorted((actual_files - seen), key=lambda path: path.as_posix())
        absent = sorted((seen - actual_files), key=lambda path: path.as_posix())
        raise ValueError(
            "implementation closure inventory differs from the source tree: "
            f"undeclared={[path.as_posix() for path in undeclared]}, "
            f"absent={[path.as_posix() for path in absent]}"
        )
    receipt = {
        "closure_artifact_sha256": artifact_sha256,
        "closure_manifest_file_sha256": observed_manifest_file_sha256,
        "file_count": len(seen),
        "root": str(root),
        "schema": "picf-next.adr175-implementation-closure-verification.v1",
        "total_bytes": total_bytes,
    }
    receipt["artifact_sha256"] = _canonical_sha256(receipt)
    if args.output is not None:
        _write_exclusive(args.output, receipt)
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
