from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data import calvin_official_source as official_source
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CalvinPhysicalSupervisionShard,
    calvin_physical_calibration_summary_fields,
    physical_supervision_manifest_payload,
)
from picf_next.data.calvin_physical_visual_acceptance import (
    CALVIN_PHYSICAL_AUDIT_SCHEMA,
    CALVIN_PHYSICAL_VISUAL_REVIEW_SCHEMA,
    build_calvin_physical_visual_acceptance,
)
from picf_next.data.dataset_manifest import (
    build_dataset_file_manifest,
    content_identified_dataset_manifest,
)
from picf_next.data.lingbot_calvin_projection import projection_payload_sha256
from picf_next.data.token_supervision_policy import (
    build_known_pixel_token_supervision_policy,
    token_supervision_policy_sha256,
)
from tools import migrate_calvin_physical_visual_review_identity as migrate


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


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(payload)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_source_claims(selected_file_count: int) -> dict[str, object]:
    return {
        "official_archive": {
            "url": official_source.CALVIN_OFFICIAL_ARCHIVE_URL,
            "transport": "http",
            "content_length": official_source.CALVIN_OFFICIAL_ARCHIVE_CONTENT_LENGTH,
            "last_modified": official_source.CALVIN_OFFICIAL_ARCHIVE_LAST_MODIFIED,
            "etag": official_source.CALVIN_OFFICIAL_ARCHIVE_ETAG,
            "tail_size_bytes": official_source.CALVIN_OFFICIAL_ARCHIVE_TAIL_SIZE_BYTES,
            "tail_sha256": official_source.CALVIN_OFFICIAL_ARCHIVE_TAIL_SHA256,
            "central_directory_offset": (official_source.CALVIN_OFFICIAL_CENTRAL_DIRECTORY_OFFSET),
            "central_directory_size": official_source.CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SIZE,
            "central_directory_sha256": (official_source.CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SHA256),
            "entry_count": official_source.CALVIN_OFFICIAL_ARCHIVE_ENTRY_COUNT,
            "zip64": True,
            "publisher_authenticity": official_source.CALVIN_OFFICIAL_PUBLISHER_AUTHENTICITY,
        },
        "official_training_inventory": {
            "archive_prefix": official_source.CALVIN_OFFICIAL_TRAINING_PREFIX,
            "archive_entry_count": official_source.CALVIN_OFFICIAL_ARCHIVE_ENTRY_COUNT,
            "file_count": selected_file_count
            + len(official_source.CALVIN_OFFICIAL_NON_RUNTIME_TRAINING_FILES),
            "excluded_non_runtime_files": list(
                official_source.CALVIN_OFFICIAL_NON_RUNTIME_TRAINING_FILES
            ),
        },
    }


def _projection(dataset_sha256: str, dataset_tree_sha256: str) -> dict[str, object]:
    def view(source_field: str, shape: list[int], digit: str) -> dict[str, object]:
        return {
            "source_field": source_field,
            "source_shape": shape,
            "image_grid_thw": [1, 16, 16],
            "merged_grid_hw": [8, 8],
            "raw_patch_count": 256,
            "merged_token_count": 64,
            "pixel_values_shape": [256, 1536],
            "source_rgb_sha256": [digit * 64],
        }

    return {
        "schema": "picf-next.lingbot-calvin-qwen-projection.v1",
        "status": "PASS",
        "runtime_input": False,
        "processor_id": "Qwen/Qwen3-VL-4B-Instruct",
        "processor_revision": "c" * 40,
        "processor_assets_sha256": "d" * 64,
        "processor_config_sha256": "e" * 64,
        "processor_preprocessor_config_sha256": "f" * 64,
        "dataset_manifest_sha256": dataset_sha256,
        "dataset_tree_sha256": dataset_tree_sha256,
        "source_frame_count": 1,
        "sample_global_indices": [0],
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "views": {
            "static": view("rgb_static", [200, 200, 3], "1"),
            "gripper": view("rgb_gripper", [84, 84, 3], "2"),
        },
        "transformers_version": "5.0.0",
    }


def _audit(
    *,
    dataset_sha256: str,
    sidecar_sha256: str,
    projection_sha256: str,
    projection: dict[str, object],
    panel_name: str,
    panel_sha256: str,
) -> dict[str, object]:
    supervision_policy = build_known_pixel_token_supervision_policy()
    return {
        "format": CALVIN_PHYSICAL_AUDIT_SCHEMA,
        "mode": "full_tail",
        "runtime_input": False,
        "task_used_for_owner_selection": False,
        "task_used_for_audit_selection": True,
        "selection_affects_training": False,
        "coverage": CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
        "dataset_manifest_sha256": dataset_sha256,
        "sidecar_manifest_sha256": sidecar_sha256,
        "training_projection_contract_sha256": projection_sha256,
        "training_projection_payload_sha256": projection_payload_sha256(projection),
        "training_projection": projection,
        "training_supervision_policy_sha256": token_supervision_policy_sha256(supervision_policy),
        "training_supervision_policy": supervision_policy,
        "frame_count": 1,
        "first_global_index": 0,
        "last_global_index": 0,
        "full_shard_schema_validation": True,
        "manifest_summary_match": True,
        "manifest_summary_absolute_error": 0.0,
        "distributions": {"static": {"rgb_mae": {"maximum": 0.25}}},
        "selection_contract": {
            "tail_per_metric": 4,
            "tail_directions": {
                "rgb_mae": ["high"],
                "depth_mae_m": ["high"],
                "depth_p95_m": ["high"],
                "known_pixel_fraction": ["low"],
                "raw_object_pixel_fraction": ["high"],
                "known_object_pixel_fraction": ["low", "high"],
                "known_owner_retention": ["low"],
            },
            "temporal_strata": 16,
            "one_median_occurrence_midpoint_per_task": True,
            "deduplicated": True,
        },
        "record_count": 1,
        "records": [
            {
                "global_index": 0,
                "selection_reasons": ["temporal:00"],
                "task_annotations": [],
                "identity_keys": ["block_red"],
                "visible_identity_keys": ["block_red"],
                "panel": panel_name,
                "panel_sha256": panel_sha256,
                "cameras": {"static": {"rgb_mae": 0.25}, "gripper": {"rgb_mae": 0.1}},
                "scanned_metrics": {
                    "static": {"rgb_mae": 0.25},
                    "gripper": {"rgb_mae": 0.1},
                },
            }
        ],
    }


def _review(audit_sha256: str, sidecar_sha256: str, panel_sha256: str) -> dict[str, object]:
    return {
        "schema": CALVIN_PHYSICAL_VISUAL_REVIEW_SCHEMA,
        "reviewer": "Pinned physical review",
        "reviewed_at_utc": "2026-08-10T00:00:00+00:00",
        "audit_manifest_sha256": audit_sha256,
        "sidecar_manifest_sha256": sidecar_sha256,
        "rows": [
            {
                "global_index": 0,
                "panel": "step0000000_task.png",
                "panel_sha256": panel_sha256,
                "verdict": "PASS",
                "observations": "Both source views and physical overlays agree exactly.",
                "context_expanded": False,
            }
        ],
        "checks": {
            "every_panel_opened_original_resolution": True,
            "both_camera_views_reviewed": True,
            "task_annotation_matches_scene": True,
            "visible_owner_assignment_is_correct": True,
            "unknown_regions_do_not_paint_hidden_objects": True,
            "training_token_overlay_is_consistent": True,
            "partially_observed_tokens_are_visually_distinct": True,
            "ambiguous_cases_expanded": True,
        },
        "status": "PASS",
        "findings": "The complete deterministic panel set passed every frozen visual check.",
    }


def _fixture(tmp_path: Path) -> argparse.Namespace:
    split = tmp_path / "training"
    split.mkdir()
    (split / "scene_info.npy").write_bytes(b"official scene bytes")
    source_dataset = build_dataset_file_manifest(
        split,
        dataset_id="old.audit/calvin",
        dataset_revision="old-audit-revision",
        split_name="training",
        relative_paths=("scene_info.npy",),
    )
    receipt_source_dataset = build_dataset_file_manifest(
        split,
        dataset_id="receipt.source/calvin",
        dataset_revision="receipt-source-revision",
        split_name="training",
        relative_paths=("scene_info.npy",),
    )
    target_dataset = content_identified_dataset_manifest(
        receipt_source_dataset,
        dataset_id=official_source.CALVIN_OFFICIAL_DATASET_ID,
    )
    source_dataset_path = tmp_path / "old-dataset.json"
    receipt_source_dataset_path = tmp_path / "receipt-source-dataset.json"
    target_dataset_path = tmp_path / official_source.CALVIN_OFFICIAL_MANIFEST_NAME
    source_dataset_sha256 = _write_json(source_dataset_path, source_dataset.to_dict())
    receipt_source_dataset_sha256 = _write_json(
        receipt_source_dataset_path,
        receipt_source_dataset.to_dict(),
    )
    target_dataset_sha256 = _write_json(target_dataset_path, target_dataset.to_dict())

    receipt = {
        **_official_source_claims(len(target_dataset.files)),
        "schema": official_source.CALVIN_OFFICIAL_SOURCE_RECEIPT_SCHEMA,
        "source_manifest": {
            "file_sha256": receipt_source_dataset_sha256,
            "tree_sha256": receipt_source_dataset.tree_sha256,
            "declared_dataset_id": receipt_source_dataset.dataset_id,
            "declared_dataset_revision": receipt_source_dataset.dataset_revision,
        },
        "migrated_manifest": {
            "file_name": official_source.CALVIN_OFFICIAL_MANIFEST_NAME,
            "file_sha256": target_dataset_sha256,
            "tree_sha256": target_dataset.tree_sha256,
        },
        "verified_content": {
            "dataset_id": target_dataset.dataset_id,
            "dataset_revision": target_dataset.dataset_revision,
            "content_sha256": target_dataset.content_sha256,
            "split_name": target_dataset.split_name,
            "file_count": len(target_dataset.files),
            "total_size_bytes": target_dataset.total_size_bytes,
            "verification_mode": official_source.CALVIN_OFFICIAL_SOURCE_VERIFICATION_MODE,
            "all_manifest_sha256_matches": True,
            "all_official_crc32_matches": True,
            "official_inventory_exact_after_declared_exclusions": True,
        },
        "training_authorized": False,
    }
    receipt_path = tmp_path / "official-source-receipt.json"
    receipt_sha256 = _write_json(receipt_path, receipt)

    shard = CalvinPhysicalSupervisionShard(
        path="shard00000.npz",
        sha256=hashlib.sha256(b"immutable shard bytes").hexdigest(),
        first_global_index=0,
        last_global_index=0,
        frame_count=1,
        object_record_count=1,
    )
    summary = {
        field: 0.5
        for field in calvin_physical_calibration_summary_fields(
            CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
        )
    }
    source_sidecar = physical_supervision_manifest_payload(
        dataset_id=source_dataset.dataset_id,
        dataset_revision=source_dataset.dataset_revision,
        split_name=source_dataset.split_name,
        scene_info_sha256=hashlib.sha256(b"official scene bytes").hexdigest(),
        global_indices=np.asarray([0], dtype=np.int64),
        shards=(shard,),
        calibration_summary=summary,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    target_sidecar = copy.deepcopy(source_sidecar)
    target_sidecar["dataset_id"] = target_dataset.dataset_id
    target_sidecar["dataset_revision"] = target_dataset.dataset_revision
    source_sidecar_path = tmp_path / "old-sidecar.json"
    target_sidecar_path = tmp_path / "official-sidecar.json"
    source_sidecar_sha256 = _write_json(source_sidecar_path, source_sidecar)
    target_sidecar_sha256 = _write_json(target_sidecar_path, target_sidecar)

    source_projection = _projection(source_dataset_sha256, source_dataset.tree_sha256)
    target_projection = _projection(target_dataset_sha256, target_dataset.tree_sha256)
    target_projection_path = tmp_path / "official-projection.json"
    target_projection_sha256 = _write_json(target_projection_path, target_projection)

    panel_bytes = b"\x89PNG\r\n\x1a\ncontent-identical-physical-panel"
    panel_name = "step0000000_task.png"
    source_audit_dir = tmp_path / "old-audit"
    target_audit_dir = tmp_path / "official-audit"
    source_audit_dir.mkdir()
    target_audit_dir.mkdir()
    (source_audit_dir / panel_name).write_bytes(panel_bytes)
    (target_audit_dir / panel_name).write_bytes(panel_bytes)
    panel_sha256 = hashlib.sha256(panel_bytes).hexdigest()
    source_audit = _audit(
        dataset_sha256=source_dataset_sha256,
        sidecar_sha256=source_sidecar_sha256,
        projection_sha256="9" * 64,
        projection=source_projection,
        panel_name=panel_name,
        panel_sha256=panel_sha256,
    )
    target_audit = _audit(
        dataset_sha256=target_dataset_sha256,
        sidecar_sha256=target_sidecar_sha256,
        projection_sha256=target_projection_sha256,
        projection=target_projection,
        panel_name=panel_name,
        panel_sha256=panel_sha256,
    )
    source_audit_path = source_audit_dir / "audit_manifest.json"
    target_audit_path = target_audit_dir / "audit_manifest.json"
    source_audit_sha256 = _write_json(source_audit_path, source_audit)
    target_audit_sha256 = _write_json(target_audit_path, target_audit)

    review_path = tmp_path / "old-review.json"
    review_sha256 = _write_json(
        review_path,
        _review(source_audit_sha256, source_sidecar_sha256, panel_sha256),
    )
    acceptance = build_calvin_physical_visual_acceptance(
        audit_manifest_path=source_audit_path,
        audit_manifest_sha256=source_audit_sha256,
        dataset_manifest_sha256=source_dataset_sha256,
        sidecar_manifest_sha256=source_sidecar_sha256,
        review_path=review_path,
        review_sha256=review_sha256,
    )
    acceptance_path = tmp_path / "old-acceptance.json"
    acceptance_sha256 = _write_json(acceptance_path, acceptance)

    return argparse.Namespace(
        old_audit_manifest=source_audit_path,
        expected_old_audit_manifest_sha256=source_audit_sha256,
        old_review=review_path,
        expected_old_review_sha256=review_sha256,
        old_acceptance=acceptance_path,
        expected_old_acceptance_sha256=acceptance_sha256,
        new_audit_manifest=target_audit_path,
        expected_new_audit_manifest_sha256=target_audit_sha256,
        old_dataset_manifest=source_dataset_path,
        expected_old_dataset_manifest_sha256=source_dataset_sha256,
        new_dataset_manifest=target_dataset_path,
        expected_new_dataset_manifest_sha256=target_dataset_sha256,
        old_sidecar_manifest=source_sidecar_path,
        expected_old_sidecar_manifest_sha256=source_sidecar_sha256,
        new_sidecar_manifest=target_sidecar_path,
        expected_new_sidecar_manifest_sha256=target_sidecar_sha256,
        official_source_receipt=receipt_path,
        expected_official_source_receipt_sha256=receipt_sha256,
        official_source_manifest=receipt_source_dataset_path,
        expected_official_source_manifest_sha256=receipt_source_dataset_sha256,
        official_projection=target_projection_path,
        expected_official_projection_sha256=target_projection_sha256,
        output_dir=tmp_path / "official-visual-review",
    )


def test_visual_review_identity_migration_republishes_exact_review_atomically(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    source_review = json.loads(args.old_review.read_text(encoding="ascii"))

    report = migrate._run(args)  # noqa: SLF001

    output_review_path = args.output_dir / migrate.OUTPUT_REVIEW_NAME
    output_acceptance_path = args.output_dir / migrate.OUTPUT_ACCEPTANCE_NAME
    output_receipt_path = args.output_dir / migrate.OUTPUT_RECEIPT_NAME
    output_review = json.loads(output_review_path.read_text(encoding="ascii"))
    output_acceptance = json.loads(output_acceptance_path.read_text(encoding="ascii"))
    output_receipt = json.loads(output_receipt_path.read_text(encoding="ascii"))
    target_audit = json.loads(args.new_audit_manifest.read_text(encoding="ascii"))

    assert report == output_receipt
    assert report["training_authorized"] is False
    assert report["visual_inference_performed"] is False
    assert report["reviewed_verdicts_unchanged"] is True
    assert output_review["rows"] == source_review["rows"]
    assert output_review["checks"] == source_review["checks"]
    assert output_review["reviewer"] == source_review["reviewer"]
    assert output_review["reviewed_at_utc"] == source_review["reviewed_at_utc"]
    assert output_review["audit_manifest_sha256"] == args.expected_new_audit_manifest_sha256
    assert output_review["sidecar_manifest_sha256"] == args.expected_new_sidecar_manifest_sha256
    assert output_acceptance["status"] == "PASS"
    assert output_acceptance["training_projection"] == target_audit["training_projection"]
    assert report["target_review"]["file_sha256"] == _sha256(output_review_path)
    assert report["target_acceptance"]["file_sha256"] == _sha256(output_acceptance_path)
    with pytest.raises(FileExistsError):
        migrate._run(args)  # noqa: SLF001


def test_visual_review_identity_migration_requires_reproducible_old_acceptance(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    acceptance = json.loads(args.old_acceptance.read_text(encoding="ascii"))
    acceptance["reviewer"] = "Substituted reviewer"
    args.expected_old_acceptance_sha256 = _write_json(args.old_acceptance, acceptance)

    with pytest.raises(ContractError, match="not reproduced"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


@pytest.mark.parametrize("field", ["distributions", "records"])
def test_visual_review_identity_migration_rejects_deterministic_audit_drift(
    tmp_path: Path,
    field: str,
) -> None:
    args = _fixture(tmp_path)
    audit = json.loads(args.new_audit_manifest.read_text(encoding="ascii"))
    if field == "distributions":
        audit[field]["static"]["rgb_mae"]["maximum"] = 0.3
    else:
        audit[field][0]["selection_reasons"] = ["rgb_mae:high"]
    args.expected_new_audit_manifest_sha256 = _write_json(args.new_audit_manifest, audit)

    with pytest.raises(ContractError, match="changed deterministic checks or panel records"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_visual_review_identity_migration_rejects_panel_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture(tmp_path)
    original = migrate.validate_calvin_physical_audit_manifest

    def validate_then_mutate(path: Path, **keyword: object) -> dict[str, object]:
        audit = original(path, **keyword)
        if Path(path) == args.new_audit_manifest:
            (args.new_audit_manifest.parent / "step0000000_task.png").write_bytes(b"changed")
        return audit

    monkeypatch.setattr(migrate, "validate_calvin_physical_audit_manifest", validate_then_mutate)

    with pytest.raises(ContractError, match="content hash mismatch"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_visual_review_identity_migration_rejects_sidecar_payload_drift(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    sidecar = json.loads(args.new_sidecar_manifest.read_text(encoding="ascii"))
    sidecar["calibration_summary"]["maximum_static_rgb_mae"] = 0.75
    args.expected_new_sidecar_manifest_sha256 = _write_json(args.new_sidecar_manifest, sidecar)
    audit = json.loads(args.new_audit_manifest.read_text(encoding="ascii"))
    audit["sidecar_manifest_sha256"] = args.expected_new_sidecar_manifest_sha256
    args.expected_new_audit_manifest_sha256 = _write_json(args.new_audit_manifest, audit)

    with pytest.raises(ContractError, match="changed beyond dataset identity"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_visual_review_identity_migration_requires_supplied_official_projection(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    projection = json.loads(args.official_projection.read_text(encoding="ascii"))
    projection["processor_assets_sha256"] = "a" * 64
    args.expected_official_projection_sha256 = _write_json(args.official_projection, projection)

    with pytest.raises(ContractError, match="does not bind the supplied projection"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_visual_review_identity_migration_rejects_authorizing_source_receipt(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    receipt = json.loads(args.official_source_receipt.read_text(encoding="ascii"))
    receipt["training_authorized"] = True
    args.expected_official_source_receipt_sha256 = _write_json(
        args.official_source_receipt,
        receipt,
    )

    with pytest.raises(ContractError, match="must not authorize model training"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_visual_review_identity_migration_cleans_staging_after_input_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture(tmp_path)
    original = migrate.build_calvin_physical_visual_acceptance
    calls = 0

    def finalize_then_mutate(**keyword: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        acceptance = original(**keyword)
        if calls == 2:
            args.official_projection.write_bytes(b"changed during finalization")
        return acceptance

    monkeypatch.setattr(migrate, "build_calvin_physical_visual_acceptance", finalize_then_mutate)

    with pytest.raises(ContractError, match="official CALVIN projection changed"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()
    assert not args.output_dir.with_name(f".{args.output_dir.name}.partial-{os.getpid()}").exists()
