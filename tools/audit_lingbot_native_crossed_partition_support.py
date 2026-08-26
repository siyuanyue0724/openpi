#!/usr/bin/env python3
"""Audit observed P/X/N/C support in one native-VL training curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.repository_import import bind_entrypoint_to_own_repository
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot crossed grounding partition-support audit",
)

from picf_next.artifact_io import write_bytes_durable_exclusive  # noqa: E402
from picf_next.lingbot_native.crossed_causal_grounding import (  # noqa: E402
    build_crossed_partition_support_report,
    crossed_support_report_bytes,
)
from picf_next.lingbot_native.vl_curriculum import (  # noqa: E402
    NATIVE_VL_CURRICULUM_MAXIMUM_BYTES,
    NativeVLGroundingCurriculumPlan,
)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _strict_json_mapping(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite JSON constant {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _load_verified_json(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
    name: str,
) -> tuple[Mapping[str, Any], str]:
    expected = _sha256(expected_sha256, name=f"{name} expected SHA-256")
    source = path.expanduser().absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be one real file")
    size = source.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{name} file size is outside the supported contract")
    payload = source.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(f"{name} file SHA-256 changed")
    return _strict_json_mapping(payload, name=name), observed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", required=True, type=Path)
    parser.add_argument("--curriculum-sha256", required=True)
    parser.add_argument("--expected-curriculum-artifact-sha256", required=True)
    parser.add_argument("--scene-audit", required=True, type=Path)
    parser.add_argument("--scene-audit-sha256", required=True)
    parser.add_argument(
        "--expected-task-key",
        action="append",
        dest="expected_task_keys",
        required=True,
    )
    parser.add_argument(
        "--target-identity",
        action="append",
        dest="target_identities",
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    curriculum_json, curriculum_file_sha256 = _load_verified_json(
        args.curriculum,
        expected_sha256=args.curriculum_sha256,
        maximum_bytes=NATIVE_VL_CURRICULUM_MAXIMUM_BYTES,
        name="native VL curriculum",
    )
    curriculum = NativeVLGroundingCurriculumPlan.from_dict(curriculum_json)
    expected_artifact = _sha256(
        args.expected_curriculum_artifact_sha256,
        name="expected curriculum artifact SHA-256",
    )
    if curriculum.artifact_sha256 != expected_artifact:
        raise ValueError("native VL curriculum artifact SHA-256 changed")
    scene_audit, scene_file_sha256 = _load_verified_json(
        args.scene_audit,
        expected_sha256=args.scene_audit_sha256,
        maximum_bytes=64 * 1024 * 1024,
        name="scene audit",
    )
    report = build_crossed_partition_support_report(
        curriculum.groups,
        scene_audit,
        curriculum_artifact_sha256=curriculum.artifact_sha256,
        curriculum_file_sha256=curriculum_file_sha256,
        scene_audit_file_sha256=scene_file_sha256,
        expected_task_keys=args.expected_task_keys,
        expected_target_identity_keys=args.target_identities,
    )
    output = args.output.expanduser().absolute()
    write_bytes_durable_exclusive(output, crossed_support_report_bytes(report))
    print(
        json.dumps(
            {
                "artifact_sha256": report["artifact_sha256"],
                "output": str(output),
                "partition": report["partition"],
                "status": report["status"],
                "training_authorized": report["training_authorized"],
            },
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
