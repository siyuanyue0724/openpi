#!/usr/bin/env python3
"""Materialize every native-VL curriculum target before allocating a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from tools.repository_import import bind_entrypoint_to_own_repository
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot native VL curriculum materialization audit",
)

from picf_next.artifact_io import write_bytes_durable_exclusive  # noqa: E402
from picf_next.contracts import ContractError  # noqa: E402
from picf_next.data.calvin import CalvinDatasetIndex  # noqa: E402
from picf_next.data.calvin_physical_supervision_sidecar import (  # noqa: E402
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (  # noqa: E402
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.vl_cotraining import (  # noqa: E402
    materialize_fixed_observation_native_vl_records,
)
from picf_next.lingbot_native.vl_curriculum import (  # noqa: E402
    NativeVLGroundingCurriculumPlan,
)

SCHEMA = "picf-next.native-vl-curriculum-materialization-audit.v1"


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("native VL materialization audit is not canonical JSON") from error


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return value


def _verified_file_sha256(path: Path, expected: str, *, name: str) -> str:
    expected_sha256 = _sha256(expected, name=f"{name} expected SHA-256")
    source = path.expanduser().absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be one real file")
    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"{name} file SHA-256 changed")
    return observed


def _materialization_summary(
    records: Sequence[Mapping[str, object]],
    *,
    expected_record_count: int,
    expected_unique_variant_count: int,
) -> dict[str, object]:
    expected_records = _positive_int(expected_record_count, name="expected record count")
    expected_unique = _positive_int(
        expected_unique_variant_count,
        name="expected unique variant count",
    )
    if len(records) != expected_records:
        raise ValueError("native VL materialization record count changed")
    required = {
        "bbox_xyxy",
        "camera_name",
        "global_index",
        "instruction_sha256",
        "optimizer_step",
        "qwen_bbox_xyxy",
        "rank",
        "source_rgb_sha256",
        "target_identity_key",
        "task_key",
    }
    if any(set(record) != required for record in records):
        raise ValueError("native VL materialization record fields changed")
    keys = [
        (
            record["global_index"],
            record["task_key"],
            record["target_identity_key"],
            record["instruction_sha256"],
        )
        for record in records
    ]
    counts = Counter(keys)
    if len(counts) != expected_unique or any(value not in (1, 2) for value in counts.values()):
        raise ValueError("native VL materialization unique coverage changed")
    duplicate_counts = sorted(value for value in counts.values() if value > 1)
    if sum(value - 1 for value in counts.values()) != expected_records - expected_unique:
        raise ValueError("native VL materialization duplicate accounting changed")
    camera_histogram = Counter(record["camera_name"] for record in records)
    bbox_areas = []
    for record in records:
        bbox = record["bbox_xyxy"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
        ):
            raise ValueError("native VL materialization bbox is malformed")
        bbox_areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if any(area <= 0 for area in bbox_areas):
        raise ValueError("native VL materialization bbox has non-positive area")
    return {
        "bbox_area_maximum": max(bbox_areas),
        "bbox_area_minimum": min(bbox_areas),
        "camera_histogram": dict(sorted(camera_histogram.items())),
        "duplicate_multiplicities": duplicate_counts,
        "materialized_record_count": len(records),
        "unique_variant_count": len(counts),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-plan", type=Path, required=True)
    parser.add_argument("--curriculum-plan-sha256", required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--picf-code-revision", required=True)
    parser.add_argument("--expected-group-count", type=int, required=True)
    parser.add_argument("--expected-optimizer-step-count", type=int, required=True)
    parser.add_argument("--expected-record-count", type=int, required=True)
    parser.add_argument("--expected-unique-variant-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    revision = args.picf_code_revision
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("PICF code revision must be one lowercase Git commit")
    curriculum_file_sha256 = _verified_file_sha256(
        args.curriculum_plan,
        args.curriculum_plan_sha256,
        name="native VL curriculum",
    )
    plan = NativeVLGroundingCurriculumPlan.load(args.curriculum_plan)
    expected_groups = _positive_int(args.expected_group_count, name="expected group count")
    expected_steps = _positive_int(
        args.expected_optimizer_step_count,
        name="expected optimizer step count",
    )
    if len(plan.groups) != expected_groups or len(plan.steps) != expected_steps:
        raise ValueError("native VL materialization plan dimensions changed")

    manifest_file_sha256 = hashlib.sha256(args.dataset_manifest.read_bytes()).hexdigest()
    manifest = load_dataset_file_manifest(args.dataset_manifest)
    validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=args.dataset_split.name,
    )
    if (
        plan.dataset_id,
        plan.dataset_revision,
        plan.dataset_manifest_sha256,
    ) != (manifest.dataset_id, manifest.dataset_revision, manifest.tree_sha256):
        raise ContractError("native VL curriculum belongs to another dataset")
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(args.physical_sidecar_root, index)

    records: list[dict[str, object]] = []
    for step in plan.steps:
        group, batches = plan.resolve_step(step.optimizer_step)
        (
            (_low_lattice, low_camera, low_variants),
            (
                _high_lattice,
                high_camera,
                high_variants,
            ),
        ) = batches
        if high_variants != (low_variants[1], low_variants[0]) or high_camera != low_camera:
            raise ContractError("native VL materialization rank reversal changed")
        pair = materialize_fixed_observation_native_vl_records(
            index=index,
            sidecar=sidecar,
            group=group,
            variants=low_variants,
            expected_camera_name=low_camera,
        )
        for rank, (variant, record) in enumerate(zip(low_variants, pair, strict=True)):
            if record.instruction != variant.instruction or record.camera_name != low_camera:
                raise ContractError("native VL materialization semantics changed")
            records.append(
                {
                    "bbox_xyxy": list(record.bbox_xyxy),
                    "camera_name": record.camera_name,
                    "global_index": record.global_index,
                    "instruction_sha256": variant.instruction_sha256,
                    "optimizer_step": step.optimizer_step,
                    "qwen_bbox_xyxy": list(record.qwen_bbox_xyxy),
                    "rank": rank,
                    "source_rgb_sha256": record.source_rgb_sha256,
                    "target_identity_key": record.target_identity_key,
                    "task_key": record.task_key,
                }
            )

    summary = _materialization_summary(
        records,
        expected_record_count=args.expected_record_count,
        expected_unique_variant_count=args.expected_unique_variant_count,
    )
    planned_measurable_count = sum(len(group.variants) for group in plan.groups)
    if summary["unique_variant_count"] != planned_measurable_count:
        raise ContractError("native VL materialization differs from measurable plan coverage")
    content = {
        "curriculum_artifact_sha256": plan.artifact_sha256,
        "curriculum_file_sha256": curriculum_file_sha256,
        "dataset_manifest_file_sha256": manifest_file_sha256,
        "dataset_tree_sha256": manifest.tree_sha256,
        "physical_sidecar_manifest_sha256": sidecar.manifest_sha256,
        "picf_code_revision": revision,
        "records": records,
        "schema": SCHEMA,
        "status": "PASS",
        "summary": summary,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    artifact_sha256 = hashlib.sha256(_canonical_bytes(content)).hexdigest()
    payload = _canonical_bytes({**content, "artifact_sha256": artifact_sha256}) + b"\n"
    write_bytes_durable_exclusive(args.output, payload)
    file_sha256 = hashlib.sha256(payload).hexdigest()
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "file_sha256": file_sha256,
                "output": str(args.output.expanduser().absolute()),
                **summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
