#!/usr/bin/env python3
"""Bind four immutable LingBot predictive fixed-batch arm reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import OrderedDict
from pathlib import Path

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.fixed_batch_probe import (
    PREDICTIVE_FIXED_BATCH_ARMS,
    assemble_predictive_fixed_batch_experiment,
    validate_predictive_fixed_batch_arm_report,
    validate_predictive_fixed_batch_experiment_report,
)

try:
    from tools.bootstrap_lingbot_vla2 import LINGBOT_CHECKPOINT_REVISION
    from tools.bootstrap_lingbot_vla2_native import LINGBOT_NATIVE_SOURCE_COMMIT
except ModuleNotFoundError:
    from bootstrap_lingbot_vla2 import LINGBOT_CHECKPOINT_REVISION  # type: ignore[no-redef]
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        LINGBOT_NATIVE_SOURCE_COMMIT,
    )


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _supported_regular_link_count(*, mode: int, link_count: int) -> bool:
    """Accept local POSIX files and the persistent FUSE zero-link convention."""

    return stat.S_ISREG(mode) and link_count in {0, 1}


def _read_stable_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("fixed-batch arm report is not a readable regular file") from error
    try:
        before = os.fstat(descriptor)
        if not _supported_regular_link_count(
            mode=before.st_mode,
            link_count=before.st_nlink,
        ):
            raise ValueError(
                "fixed-batch arm report is not a regular file with supported link semantics"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise ValueError("fixed-batch arm report changed while it was read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ValueError("fixed-batch arm report read was incomplete")
        return payload
    finally:
        os.close(descriptor)


def _required_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _load_arm(path: Path, *, expected_sha256: str) -> dict[str, object]:
    expected = _required_sha256(expected_sha256, name="arm report digest")
    payload = _read_stable_regular_bytes(path)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("fixed-batch arm report differs from its expected digest")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fixed-batch arm report is not valid ASCII JSON") from error
    report = validate_predictive_fixed_batch_arm_report(value)
    provenance = report["provenance"]
    if (
        provenance["source_commit"] != LINGBOT_NATIVE_SOURCE_COMMIT
        or provenance["checkpoint_revision"] != LINGBOT_CHECKPOINT_REVISION
    ):
        raise ValueError("fixed-batch arm belongs to another LingBot source or checkpoint")
    return report


def build_experiment(
    report_paths: OrderedDict[str, Path],
    *,
    report_sha256: OrderedDict[str, str],
) -> dict[str, object]:
    if tuple(report_paths) != PREDICTIVE_FIXED_BATCH_ARMS:
        raise ValueError("fixed-batch report paths must use frozen arm order")
    if tuple(report_sha256) != PREDICTIVE_FIXED_BATCH_ARMS:
        raise ValueError("fixed-batch report digests must use frozen arm order")
    reports: OrderedDict[str, dict[str, object]] = OrderedDict()
    for arm in PREDICTIVE_FIXED_BATCH_ARMS:
        reports[arm] = _load_arm(
            report_paths[arm],
            expected_sha256=report_sha256[arm],
        )
    return assemble_predictive_fixed_batch_experiment(
        reports,
        report_sha256=report_sha256,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in PREDICTIVE_FIXED_BATCH_ARMS:
        option = arm.replace("_", "-")
        parser.add_argument(f"--{option}-report", type=Path, required=True)
        parser.add_argument(f"--{option}-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_paths: OrderedDict[str, Path] = OrderedDict()
    report_digests: OrderedDict[str, str] = OrderedDict()
    for arm in PREDICTIVE_FIXED_BATCH_ARMS:
        report_paths[arm] = getattr(args, f"{arm}_report")
        report_digests[arm] = getattr(args, f"{arm}_report_sha256")
    result = build_experiment(
        report_paths,
        report_sha256=report_digests,
    )
    validate_predictive_fixed_batch_experiment_report(result)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    write_text_durable_exclusive(args.output, payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
