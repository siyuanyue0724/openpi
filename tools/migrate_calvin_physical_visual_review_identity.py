#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Migrate an accepted CALVIN physical visual review to official identity."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="CALVIN physical visual-review identity migration",
)

from picf_next.artifact_io import (
    publish_prepared_directory_durable_exclusive,
    write_bytes_durable_exclusive,
)
from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_SOURCE_COMMIT,
    CALVIN_STATE_RESTORATION,
)
from picf_next.data.calvin_official_source import (
    validate_calvin_content_identity_migration,
    validate_calvin_official_source_receipt,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CAMERA_SPECS,
    CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS,
    CALVIN_DEPTH_CONSISTENT_OWNER_CONTRACT,
    CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION,
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA,
    CalvinPhysicalSupervisionShard,
    calvin_physical_calibration_summary_fields,
    validate_calvin_depth_consistent_diagnostics,
)
from picf_next.data.calvin_physical_visual_acceptance import (
    build_calvin_physical_visual_acceptance,
    load_calvin_physical_visual_acceptance,
    validate_calvin_physical_audit_manifest,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    file_sha256,
    load_dataset_file_manifest,
    read_sha256_verified_file_beneath,
)
from picf_next.data.lingbot_calvin_projection import (
    load_lingbot_calvin_projection_contract,
)

MIGRATION_RECEIPT_SCHEMA = "picf-next.calvin-physical-visual-review-identity-migration.v1"
OUTPUT_REVIEW_NAME = "calvin-physical-visual-review.json"
OUTPUT_ACCEPTANCE_NAME = "calvin-physical-visual-acceptance.json"
OUTPUT_RECEIPT_NAME = "migration-receipt.json"

_MAXIMUM_JSON_BYTES = 32 * 1024 * 1024
_MAXIMUM_PANEL_BYTES = 128 * 1024 * 1024
_SIDECAR_FIELDS = {
    "schema",
    "dataset_id",
    "dataset_revision",
    "split_name",
    "calvin_commit",
    "calvin_env_commit",
    "state_restoration",
    "geometry_contract",
    "geometry_contract_sha256",
    "owner_contract",
    "camera_specs",
    "calibration_summary",
    "runtime_input",
    "task_conditioned",
    "source_fields",
    "scene_info_sha256",
    "frame_count",
    "object_record_count",
    "global_indices_sha256",
    "shards",
    "coverage",
    "owner_supervision",
    "frame_diagnostics",
}
_EXPECTED_SOURCE_FIELDS = [
    "depth_gripper",
    "depth_static",
    "rgb_gripper",
    "rgb_static",
    "robot_obs",
    "scene_info",
    "scene_obs",
]
_AUDIT_IDENTITY_FIELDS = {
    "dataset_manifest_sha256",
    "sidecar_manifest_sha256",
    "training_projection_contract_sha256",
    "training_projection_payload_sha256",
    "training_projection",
}
_ACCEPTANCE_IDENTITY_FIELDS = {
    "audit_manifest_sha256",
    "review_sha256",
    "dataset_manifest_sha256",
    "sidecar_manifest_sha256",
    "training_projection_contract_sha256",
    "training_projection_payload_sha256",
    "training_projection",
}
_PROJECTION_IDENTITY_FIELDS = {
    "dataset_manifest_sha256",
    "dataset_tree_sha256",
}


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _read_pinned_mapping(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, object], str]:
    expected = _sha256(expected_sha256, label=f"{label} expected SHA-256")
    resolved = _absolute(path)
    actual = file_sha256(resolved)
    if actual != expected:
        raise ContractError(f"{label} differs from its pinned SHA-256")
    if resolved.stat().st_size > _MAXIMUM_JSON_BYTES:
        raise ContractError(f"{label} exceeds the maximum JSON size")
    try:
        payload = json.loads(resolved.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid ASCII JSON") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ContractError(f"{label} must be a string-keyed mapping")
    if file_sha256(resolved) != expected:
        raise ContractError(f"{label} changed while loading")
    return payload, expected


def _load_pinned_dataset_manifest(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[DatasetFileManifest, str, Path]:
    expected = _sha256(expected_sha256, label=f"{label} expected SHA-256")
    resolved = _absolute(path)
    if file_sha256(resolved) != expected:
        raise ContractError(f"{label} differs from its pinned SHA-256")
    manifest = load_dataset_file_manifest(resolved)
    if file_sha256(resolved) != expected:
        raise ContractError(f"{label} changed while loading")
    return manifest, expected, resolved


def _sidecar_shards(
    payload: Mapping[str, object],
    *,
    manifest: DatasetFileManifest,
    label: str,
) -> tuple[CalvinPhysicalSupervisionShard, ...]:
    if set(payload) != _SIDECAR_FIELDS:
        raise ContractError(f"{label} fields differ from the frozen all-source schema")
    try:
        scene_info_sha256 = manifest.record_for("scene_info.npy").sha256
    except ContractError as error:
        raise ContractError(f"{label} dataset does not inventory scene_info.npy") from error
    if (
        payload.get("schema") != CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA
        or payload.get("dataset_id") != manifest.dataset_id
        or payload.get("dataset_revision") != manifest.dataset_revision
        or payload.get("split_name") != manifest.split_name
        or payload.get("calvin_commit") != CALVIN_SOURCE_COMMIT
        or payload.get("calvin_env_commit") != CALVIN_ENV_SOURCE_COMMIT
        or payload.get("state_restoration") != CALVIN_STATE_RESTORATION
        or payload.get("geometry_contract") != CALVIN_OBJECT_GEOMETRY_CONTRACT.to_dict()
        or payload.get("geometry_contract_sha256") != CALVIN_OBJECT_GEOMETRY_CONTRACT.fingerprint
        or payload.get("owner_contract") != CALVIN_DEPTH_CONSISTENT_OWNER_CONTRACT
        or payload.get("camera_specs") != [dict(value) for value in CALVIN_CAMERA_SPECS]
        or payload.get("runtime_input") is not False
        or payload.get("task_conditioned") is not False
        or payload.get("source_fields") != _EXPECTED_SOURCE_FIELDS
        or payload.get("scene_info_sha256") != scene_info_sha256
        or payload.get("coverage") != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
        or payload.get("owner_supervision") != CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION
        or payload.get("frame_diagnostics") != CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS
    ):
        raise ContractError(f"{label} did not pass its all-source machine contract")
    _sha256(payload.get("global_indices_sha256"), label=f"{label} global-indices SHA-256")

    summary = payload.get("calibration_summary")
    expected_summary = calvin_physical_calibration_summary_fields(
        CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
    )
    if (
        not isinstance(summary, Mapping)
        or set(summary) != expected_summary
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
            for value in summary.values()
        )
    ):
        raise ContractError(f"{label} calibration summary is invalid")
    validate_calvin_depth_consistent_diagnostics(cast(Mapping[str, float], summary))

    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ContractError(f"{label} has no shards")
    shards = tuple(CalvinPhysicalSupervisionShard.from_dict(item) for item in raw_shards)
    frame_count = payload.get("frame_count")
    object_count = payload.get("object_record_count")
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count <= 0
        or not isinstance(object_count, int)
        or isinstance(object_count, bool)
        or object_count <= 0
        or sum(shard.frame_count for shard in shards) != frame_count
        or sum(shard.object_record_count for shard in shards) != object_count
    ):
        raise ContractError(f"{label} counts differ from its shards")
    next_global_index = 0
    for shard in shards:
        if (
            shard.first_global_index != next_global_index
            or shard.frame_count != shard.last_global_index - shard.first_global_index + 1
        ):
            raise ContractError(f"{label} shard ranges are not contiguous all-source coverage")
        next_global_index = shard.last_global_index + 1
    if next_global_index != frame_count:
        raise ContractError(f"{label} shard ranges differ from its frame count")
    return shards


def _without_fields(value: Mapping[str, object], fields: set[str]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key not in fields}


def _assert_projection_identity_migration(
    source: Mapping[str, object],
    target: Mapping[str, object],
) -> None:
    if _without_fields(source, _PROJECTION_IDENTITY_FIELDS) != _without_fields(
        target,
        _PROJECTION_IDENTITY_FIELDS,
    ):
        raise ContractError("CALVIN official projection changed beyond dataset identity")


def _assert_audit_identity_migration(
    source: Mapping[str, object],
    target: Mapping[str, object],
) -> None:
    if _without_fields(source, _AUDIT_IDENTITY_FIELDS) != _without_fields(
        target,
        _AUDIT_IDENTITY_FIELDS,
    ):
        raise ContractError("CALVIN official audit changed deterministic checks or panel records")
    _assert_projection_identity_migration(
        cast(Mapping[str, object], source["training_projection"]),
        cast(Mapping[str, object], target["training_projection"]),
    )


def _panel_bytes(
    audit_manifest_path: Path,
    record: Mapping[str, object],
) -> bytes:
    root = _absolute(audit_manifest_path).parent
    return read_sha256_verified_file_beneath(
        root,
        cast(str, record["panel"]),
        expected_sha256=cast(str, record["panel_sha256"]),
        maximum_bytes=_MAXIMUM_PANEL_BYTES,
    )


def _assert_panel_bytes_identical(
    *,
    source_audit_path: Path,
    source_records: Sequence[Mapping[str, object]],
    target_audit_path: Path,
    target_records: Sequence[Mapping[str, object]],
) -> None:
    if len(source_records) != len(target_records):
        raise ContractError("CALVIN official audit changed the panel count")
    for index, (source_record, target_record) in enumerate(
        zip(source_records, target_records, strict=True)
    ):
        source = _panel_bytes(source_audit_path, source_record)
        target = _panel_bytes(target_audit_path, target_record)
        if source != target:
            raise ContractError(f"CALVIN official audit panel bytes differ at row {index}")


def _assert_acceptance_identity_migration(
    source: Mapping[str, object],
    target: Mapping[str, object],
) -> None:
    if _without_fields(source, _ACCEPTANCE_IDENTITY_FIELDS) != _without_fields(
        target,
        _ACCEPTANCE_IDENTITY_FIELDS,
    ):
        raise ContractError("CALVIN migrated acceptance changed a reviewed decision")
    _assert_projection_identity_migration(
        cast(Mapping[str, object], source["training_projection"]),
        cast(Mapping[str, object], target["training_projection"]),
    )


def _assert_stable_inputs(inputs: Sequence[tuple[Path, str, str]]) -> None:
    for path, expected_sha256, label in inputs:
        if file_sha256(path) != expected_sha256:
            raise ContractError(f"{label} changed during visual-review migration")


def _publish(
    *,
    output_dir: Path,
    migrated_review: dict[str, object],
    target_audit_path: Path,
    target_audit_sha256: str,
    target_dataset_sha256: str,
    target_sidecar_sha256: str,
    source_acceptance: Mapping[str, object],
    receipt: dict[str, object],
    stable_inputs: Sequence[tuple[Path, str, str]],
    source_audit_path: Path,
    source_records: Sequence[Mapping[str, object]],
    target_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    destination = _absolute(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    partial.mkdir(exist_ok=False)
    try:
        review_bytes = _json_bytes(migrated_review)
        review_sha256 = hashlib.sha256(review_bytes).hexdigest()
        review_path = partial / OUTPUT_REVIEW_NAME
        write_bytes_durable_exclusive(review_path, review_bytes)

        acceptance = build_calvin_physical_visual_acceptance(
            audit_manifest_path=target_audit_path,
            audit_manifest_sha256=target_audit_sha256,
            dataset_manifest_sha256=target_dataset_sha256,
            sidecar_manifest_sha256=target_sidecar_sha256,
            review_path=review_path,
            review_sha256=review_sha256,
            require_pass=True,
        )
        _assert_acceptance_identity_migration(source_acceptance, acceptance)
        acceptance_bytes = _json_bytes(acceptance)
        acceptance_sha256 = hashlib.sha256(acceptance_bytes).hexdigest()
        write_bytes_durable_exclusive(
            partial / OUTPUT_ACCEPTANCE_NAME,
            acceptance_bytes,
        )

        receipt["target_review"] = {
            "file_name": OUTPUT_REVIEW_NAME,
            "file_sha256": review_sha256,
        }
        receipt["target_acceptance"] = {
            "file_name": OUTPUT_ACCEPTANCE_NAME,
            "file_sha256": acceptance_sha256,
        }
        write_bytes_durable_exclusive(
            partial / OUTPUT_RECEIPT_NAME,
            _json_bytes(receipt),
        )

        _assert_stable_inputs(stable_inputs)
        _assert_panel_bytes_identical(
            source_audit_path=source_audit_path,
            source_records=source_records,
            target_audit_path=target_audit_path,
            target_records=target_records,
        )
        publish_prepared_directory_durable_exclusive(partial, destination)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return receipt


def _run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = _absolute(args.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)

    source_dataset, source_dataset_sha256, source_dataset_path = _load_pinned_dataset_manifest(
        args.old_dataset_manifest,
        expected_sha256=args.expected_old_dataset_manifest_sha256,
        label="old CALVIN dataset manifest",
    )
    target_dataset, target_dataset_sha256, target_dataset_path = _load_pinned_dataset_manifest(
        args.new_dataset_manifest,
        expected_sha256=args.expected_new_dataset_manifest_sha256,
        label="official CALVIN dataset manifest",
    )
    validate_calvin_content_identity_migration(source_dataset, target_dataset)

    official_source_manifest_sha256 = _sha256(
        args.expected_official_source_manifest_sha256,
        label="official receipt source manifest expected SHA-256",
    )
    official_source_manifest_path = _absolute(args.official_source_manifest)
    if official_source_manifest_sha256 == source_dataset_sha256:
        if file_sha256(official_source_manifest_path) != official_source_manifest_sha256:
            raise ContractError("official receipt source manifest differs from its pinned SHA-256")
        official_source_manifest = source_dataset
    else:
        official_source_manifest, _, official_source_manifest_path = _load_pinned_dataset_manifest(
            args.official_source_manifest,
            expected_sha256=official_source_manifest_sha256,
            label="official receipt source manifest",
        )
    validate_calvin_content_identity_migration(official_source_manifest, target_dataset)

    source_receipt, source_receipt_sha256 = _read_pinned_mapping(
        args.official_source_receipt,
        expected_sha256=args.expected_official_source_receipt_sha256,
        label="CALVIN official source receipt",
    )
    source_receipt_path = _absolute(args.official_source_receipt)
    validate_calvin_official_source_receipt(
        source_receipt,
        source_manifest=official_source_manifest,
        source_manifest_sha256=official_source_manifest_sha256,
        target_manifest=target_dataset,
        target_manifest_sha256=target_dataset_sha256,
    )

    source_sidecar, source_sidecar_sha256 = _read_pinned_mapping(
        args.old_sidecar_manifest,
        expected_sha256=args.expected_old_sidecar_manifest_sha256,
        label="old CALVIN sidecar manifest",
    )
    target_sidecar, target_sidecar_sha256 = _read_pinned_mapping(
        args.new_sidecar_manifest,
        expected_sha256=args.expected_new_sidecar_manifest_sha256,
        label="official CALVIN sidecar manifest",
    )
    source_sidecar_path = _absolute(args.old_sidecar_manifest)
    target_sidecar_path = _absolute(args.new_sidecar_manifest)
    source_shards = _sidecar_shards(
        source_sidecar,
        manifest=source_dataset,
        label="old CALVIN sidecar manifest",
    )
    target_shards = _sidecar_shards(
        target_sidecar,
        manifest=target_dataset,
        label="official CALVIN sidecar manifest",
    )
    expected_target_sidecar = copy.deepcopy(source_sidecar)
    expected_target_sidecar["dataset_id"] = target_dataset.dataset_id
    expected_target_sidecar["dataset_revision"] = target_dataset.dataset_revision
    if target_sidecar != expected_target_sidecar or target_shards != source_shards:
        raise ContractError("CALVIN official sidecar changed beyond dataset identity")

    projection_sha256 = _sha256(
        args.expected_official_projection_sha256,
        label="official CALVIN projection expected SHA-256",
    )
    projection_path = _absolute(args.official_projection)
    projection = load_lingbot_calvin_projection_contract(
        projection_path,
        expected_sha256=projection_sha256,
        expected_dataset_manifest_sha256=target_dataset_sha256,
    )
    if projection["dataset_tree_sha256"] != target_dataset.tree_sha256:
        raise ContractError("official CALVIN projection belongs to another dataset tree")

    source_audit_sha256 = _sha256(
        args.expected_old_audit_manifest_sha256,
        label="old CALVIN audit expected SHA-256",
    )
    target_audit_sha256 = _sha256(
        args.expected_new_audit_manifest_sha256,
        label="official CALVIN audit expected SHA-256",
    )
    source_audit_path = _absolute(args.old_audit_manifest)
    target_audit_path = _absolute(args.new_audit_manifest)
    source_audit = validate_calvin_physical_audit_manifest(
        source_audit_path,
        expected_sha256=source_audit_sha256,
        expected_dataset_manifest_sha256=source_dataset_sha256,
        expected_sidecar_manifest_sha256=source_sidecar_sha256,
    )
    target_audit = validate_calvin_physical_audit_manifest(
        target_audit_path,
        expected_sha256=target_audit_sha256,
        expected_dataset_manifest_sha256=target_dataset_sha256,
        expected_sidecar_manifest_sha256=target_sidecar_sha256,
    )
    if (
        source_audit["frame_count"] != source_sidecar["frame_count"]
        or target_audit["frame_count"] != target_sidecar["frame_count"]
    ):
        raise ContractError("CALVIN visual audit frame count differs from its sidecar")
    if (
        cast(Mapping[str, object], source_audit["training_projection"])["dataset_tree_sha256"]
        != source_dataset.tree_sha256
    ):
        raise ContractError("old CALVIN audit projection belongs to another dataset tree")
    if (
        target_audit["training_projection_contract_sha256"] != projection_sha256
        or target_audit["training_projection"] != projection
    ):
        raise ContractError("official CALVIN audit does not bind the supplied projection")
    _assert_audit_identity_migration(source_audit, target_audit)

    source_records = cast(list[Mapping[str, object]], source_audit["records"])
    target_records = cast(list[Mapping[str, object]], target_audit["records"])
    _assert_panel_bytes_identical(
        source_audit_path=source_audit_path,
        source_records=source_records,
        target_audit_path=target_audit_path,
        target_records=target_records,
    )

    source_review, source_review_sha256 = _read_pinned_mapping(
        args.old_review,
        expected_sha256=args.expected_old_review_sha256,
        label="old CALVIN physical visual review",
    )
    source_review_path = _absolute(args.old_review)
    source_acceptance_sha256 = _sha256(
        args.expected_old_acceptance_sha256,
        label="old CALVIN physical visual acceptance expected SHA-256",
    )
    source_acceptance_path = _absolute(args.old_acceptance)
    rebuilt_source_acceptance = build_calvin_physical_visual_acceptance(
        audit_manifest_path=source_audit_path,
        audit_manifest_sha256=source_audit_sha256,
        dataset_manifest_sha256=source_dataset_sha256,
        sidecar_manifest_sha256=source_sidecar_sha256,
        review_path=source_review_path,
        review_sha256=source_review_sha256,
        require_pass=True,
    )
    source_acceptance = load_calvin_physical_visual_acceptance(
        source_acceptance_path,
        expected_sha256=source_acceptance_sha256,
        expected_dataset_manifest_sha256=source_dataset_sha256,
        expected_sidecar_manifest_sha256=source_sidecar_sha256,
    )
    if source_acceptance != rebuilt_source_acceptance:
        raise ContractError("old CALVIN acceptance is not reproduced by its audit and review")

    migrated_review = copy.deepcopy(source_review)
    migrated_review["audit_manifest_sha256"] = target_audit_sha256
    migrated_review["sidecar_manifest_sha256"] = target_sidecar_sha256
    expected_review = copy.deepcopy(migrated_review)
    expected_review["audit_manifest_sha256"] = source_audit_sha256
    expected_review["sidecar_manifest_sha256"] = source_sidecar_sha256
    if expected_review != source_review:
        raise ContractError("CALVIN review migration changed more than identity bindings")

    stable_inputs = (
        (source_dataset_path, source_dataset_sha256, "old CALVIN dataset manifest"),
        (target_dataset_path, target_dataset_sha256, "official CALVIN dataset manifest"),
        (
            official_source_manifest_path,
            official_source_manifest_sha256,
            "official receipt source manifest",
        ),
        (source_receipt_path, source_receipt_sha256, "CALVIN official source receipt"),
        (source_sidecar_path, source_sidecar_sha256, "old CALVIN sidecar manifest"),
        (target_sidecar_path, target_sidecar_sha256, "official CALVIN sidecar manifest"),
        (projection_path, projection_sha256, "official CALVIN projection"),
        (source_audit_path, source_audit_sha256, "old CALVIN audit"),
        (target_audit_path, target_audit_sha256, "official CALVIN audit"),
        (source_review_path, source_review_sha256, "old CALVIN visual review"),
        (
            source_acceptance_path,
            source_acceptance_sha256,
            "old CALVIN visual acceptance",
        ),
    )
    _assert_stable_inputs(stable_inputs)

    receipt: dict[str, object] = {
        "schema": MIGRATION_RECEIPT_SCHEMA,
        "status": "PASS",
        "source_audit_manifest_sha256": source_audit_sha256,
        "source_review_sha256": source_review_sha256,
        "source_acceptance_sha256": source_acceptance_sha256,
        "target_audit_manifest_sha256": target_audit_sha256,
        "source_dataset_manifest_sha256": source_dataset_sha256,
        "target_dataset_manifest_sha256": target_dataset_sha256,
        "source_sidecar_manifest_sha256": source_sidecar_sha256,
        "target_sidecar_manifest_sha256": target_sidecar_sha256,
        "official_source_receipt_sha256": source_receipt_sha256,
        "official_source_manifest_sha256": official_source_manifest_sha256,
        "official_projection_sha256": projection_sha256,
        "dataset_content_sha256": target_dataset.content_sha256,
        "panel_count": len(source_records),
        "panel_ids_unchanged": True,
        "panel_sha256_unchanged": True,
        "panel_bytes_content_identical": True,
        "deterministic_audit_checks_unchanged": True,
        "sidecar_manifest_unchanged_except_dataset_identity": True,
        "review_unchanged_except_audit_and_sidecar_bindings": True,
        "reviewed_verdicts_unchanged": True,
        "source_acceptance_reproduced_by_finalizer": True,
        "target_acceptance_rebuilt_by_finalizer": True,
        "official_source_content_identity_verified": True,
        "visual_inference_performed": False,
        "verdicts_hand_edited": False,
        "training_authorized": False,
        "training_authorization_reason": (
            "identity migration preserves an existing visual decision; the CALVIN "
            "visual finalizer does not explicitly own model-training authorization"
        ),
    }
    return _publish(
        output_dir=output_dir,
        migrated_review=migrated_review,
        target_audit_path=target_audit_path,
        target_audit_sha256=target_audit_sha256,
        target_dataset_sha256=target_dataset_sha256,
        target_sidecar_sha256=target_sidecar_sha256,
        source_acceptance=source_acceptance,
        receipt=receipt,
        stable_inputs=stable_inputs,
        source_audit_path=source_audit_path,
        source_records=source_records,
        target_records=target_records,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-audit-manifest", type=Path, required=True)
    parser.add_argument("--expected-old-audit-manifest-sha256", required=True)
    parser.add_argument("--old-review", type=Path, required=True)
    parser.add_argument("--expected-old-review-sha256", required=True)
    parser.add_argument("--old-acceptance", type=Path, required=True)
    parser.add_argument("--expected-old-acceptance-sha256", required=True)
    parser.add_argument("--new-audit-manifest", type=Path, required=True)
    parser.add_argument("--expected-new-audit-manifest-sha256", required=True)
    parser.add_argument("--old-dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-old-dataset-manifest-sha256", required=True)
    parser.add_argument("--new-dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-new-dataset-manifest-sha256", required=True)
    parser.add_argument("--old-sidecar-manifest", type=Path, required=True)
    parser.add_argument("--expected-old-sidecar-manifest-sha256", required=True)
    parser.add_argument("--new-sidecar-manifest", type=Path, required=True)
    parser.add_argument("--expected-new-sidecar-manifest-sha256", required=True)
    parser.add_argument("--official-source-receipt", type=Path, required=True)
    parser.add_argument("--expected-official-source-receipt-sha256", required=True)
    parser.add_argument("--official-source-manifest", type=Path, required=True)
    parser.add_argument("--expected-official-source-manifest-sha256", required=True)
    parser.add_argument("--official-projection", type=Path, required=True)
    parser.add_argument("--expected-official-projection-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    receipt = _run(_parse_args())
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
