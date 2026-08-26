#!/usr/bin/env python3
"""Validate three complete ADR-175 arm reports and atomically publish PASS evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from picf_next.contracts import ContractError
from picf_next.lingbot_native.adr175_validation import (
    ADR175ArmReport,
    canonical_json_bytes,
    validate_adr175_matched_three_arm,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lbot-report", type=Path, required=True)
    parser.add_argument("--physical-set-report", type=Path, required=True)
    parser.add_argument("--native-attention-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _read_report(path: Path, *, expected_arm: str) -> ADR175ArmReport:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{expected_arm} report must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{expected_arm} report is not readable UTF-8 JSON: {path}") from error
    report = ADR175ArmReport.from_dict(value)
    if report.arm != expected_arm:
        raise ContractError(
            f"{expected_arm} CLI input contains the {report.arm!r} arm instead"
        )
    return report


def _atomic_write_json(path: Path, value: object) -> None:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ContractError(f"ADR-175 validation output must not be a symlink: {expanded}")
    expanded.parent.mkdir(parents=True, exist_ok=True)
    path = expanded.parent.resolve() / expanded.name
    encoded = canonical_json_bytes(value) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        reports = (
            _read_report(args.lbot_report, expected_arm="lbot"),
            _read_report(args.physical_set_report, expected_arm="physical-set"),
            _read_report(args.native_attention_report, expected_arm="native-attention"),
        )
        result = validate_adr175_matched_three_arm(reports)
        _atomic_write_json(args.output, result.to_dict())
    except (ContractError, OSError) as error:
        print(f"ADR-175 validation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_sha256": result.artifact_sha256,
                "output": str(args.output.expanduser().resolve()),
                "status": result.status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
