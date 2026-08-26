from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data import calvin_official_source as official_source
from picf_next.data.calvin_geometry_schema import sha256_file
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CalvinPhysicalSupervisionShard,
    calvin_physical_calibration_summary_fields,
    physical_supervision_manifest_payload,
)
from picf_next.data.dataset_manifest import (
    build_dataset_file_manifest,
    content_identified_dataset_manifest,
)
from tools import audit_calvin_official_source as source_audit
from tools import migrate_calvin_physical_sidecar_identity as migrate


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _official_source_claims(selected_file_count: int) -> dict[str, object]:
    return {
        "official_archive": {
            "url": official_source.CALVIN_OFFICIAL_ARCHIVE_URL,
            "transport": "http",
            "content_length": official_source.CALVIN_OFFICIAL_ARCHIVE_CONTENT_LENGTH,
            "last_modified": official_source.CALVIN_OFFICIAL_ARCHIVE_LAST_MODIFIED.replace(",", ""),
            "etag": official_source.CALVIN_OFFICIAL_ARCHIVE_ETAG,
            "tail_size_bytes": official_source.CALVIN_OFFICIAL_ARCHIVE_TAIL_SIZE_BYTES,
            "tail_sha256": official_source.CALVIN_OFFICIAL_ARCHIVE_TAIL_SHA256,
            "central_directory_offset": (official_source.CALVIN_OFFICIAL_CENTRAL_DIRECTORY_OFFSET),
            "central_directory_size": official_source.CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SIZE,
            "central_directory_sha256": (official_source.CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SHA256),
            "entry_count": official_source.CALVIN_OFFICIAL_ARCHIVE_ENTRY_COUNT,
            "zip64": True,
            "publisher_authenticity": (official_source.CALVIN_OFFICIAL_PUBLISHER_AUTHENTICITY),
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


def _fixture(tmp_path: Path) -> argparse.Namespace:
    split = tmp_path / "training"
    split.mkdir()
    (split / "scene_info.npy").write_bytes(b"scene")
    source_manifest = build_dataset_file_manifest(
        split,
        dataset_id="mees/calvin-debug-dataset",
        dataset_revision="sha256:" + "1" * 64,
        split_name="training",
        relative_paths=("scene_info.npy",),
    )
    target_manifest = content_identified_dataset_manifest(
        source_manifest,
        dataset_id=source_audit.OFFICIAL_DATASET_ID,
    )
    source_manifest_path = tmp_path / "source-manifest.json"
    target_manifest_path = tmp_path / source_audit.MIGRATED_MANIFEST_NAME
    source_manifest_path.write_bytes(_json_bytes(source_manifest.to_dict()))
    target_manifest_path.write_bytes(_json_bytes(target_manifest.to_dict()))

    source_receipt = {
        **_official_source_claims(len(target_manifest.files)),
        "schema": source_audit.RECEIPT_SCHEMA,
        "source_manifest": {
            "file_sha256": sha256_file(source_manifest_path),
            "tree_sha256": source_manifest.tree_sha256,
            "declared_dataset_id": source_manifest.dataset_id,
            "declared_dataset_revision": source_manifest.dataset_revision,
        },
        "migrated_manifest": {
            "file_name": source_audit.MIGRATED_MANIFEST_NAME,
            "file_sha256": sha256_file(target_manifest_path),
            "tree_sha256": target_manifest.tree_sha256,
        },
        "verified_content": {
            "dataset_id": target_manifest.dataset_id,
            "dataset_revision": target_manifest.dataset_revision,
            "content_sha256": target_manifest.content_sha256,
            "split_name": target_manifest.split_name,
            "file_count": len(target_manifest.files),
            "total_size_bytes": target_manifest.total_size_bytes,
            "verification_mode": source_audit.VERIFICATION_MODE,
            "all_manifest_sha256_matches": True,
            "all_official_crc32_matches": True,
            "official_inventory_exact_after_declared_exclusions": True,
        },
        "training_authorized": False,
    }
    source_receipt_path = tmp_path / source_audit.RECEIPT_NAME
    source_receipt_path.write_bytes(_json_bytes(source_receipt))

    sidecar_root = tmp_path / "sidecar"
    sidecar_root.mkdir()
    shard_path = sidecar_root / "shard00000.npz"
    shard_path.write_bytes(b"immutable-sidecar-shard")
    shard = CalvinPhysicalSupervisionShard(
        path=shard_path.name,
        sha256=sha256_file(shard_path),
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
    sidecar = physical_supervision_manifest_payload(
        dataset_id=source_manifest.dataset_id,
        dataset_revision=source_manifest.dataset_revision,
        split_name=source_manifest.split_name,
        scene_info_sha256=hashlib.sha256(b"scene").hexdigest(),
        global_indices=np.asarray([0], dtype=np.int64),
        shards=(shard,),
        calibration_summary=summary,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    sidecar_manifest_path = sidecar_root / "manifest.json"
    sidecar_manifest_path.write_bytes(_json_bytes(sidecar))
    return argparse.Namespace(
        source_dataset_manifest=source_manifest_path,
        target_dataset_manifest=target_manifest_path,
        source_receipt=source_receipt_path,
        source_sidecar_root=sidecar_root,
        expected_source_sidecar_manifest_sha256=sha256_file(sidecar_manifest_path),
        output_dir=tmp_path / "identity-view",
        workers=2,
        progress_every=1,
    )


def test_sidecar_identity_migration_verifies_shards_without_copying_them(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    source_sidecar = json.loads((args.source_sidecar_root / "manifest.json").read_text())

    report = migrate._run(args)  # noqa: SLF001

    output = json.loads(
        (args.output_dir / migrate.OUTPUT_MANIFEST_NAME).read_text(encoding="ascii")
    )
    receipt = json.loads(
        (args.output_dir / migrate.OUTPUT_RECEIPT_NAME).read_text(encoding="ascii")
    )
    target_dataset = json.loads(args.target_dataset_manifest.read_text(encoding="ascii"))
    assert report == receipt
    assert receipt["all_sidecar_shard_sha256_matches"] is True
    assert receipt["training_authorized"] is False
    assert output["dataset_id"] == target_dataset["dataset_id"]
    assert output["dataset_revision"] == target_dataset["dataset_revision"]
    expected = copy.deepcopy(source_sidecar)
    expected["dataset_id"] = target_dataset["dataset_id"]
    expected["dataset_revision"] = target_dataset["dataset_revision"]
    assert output == expected
    assert not (args.output_dir / "shard00000.npz").exists()
    with pytest.raises(FileExistsError):
        migrate._run(args)  # noqa: SLF001


def test_sidecar_identity_migration_rejects_shard_tampering(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    (args.source_sidecar_root / "shard00000.npz").write_bytes(b"changed")

    with pytest.raises(ContractError, match="content hash mismatch"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_sidecar_identity_migration_requires_pinned_source_manifest(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.expected_source_sidecar_manifest_sha256 = "0" * 64

    with pytest.raises(ContractError, match="manifest digest differs"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_sidecar_identity_migration_rejects_content_or_receipt_drift(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    target = json.loads(args.target_dataset_manifest.read_text())
    target["dataset_id"] = "wrong"
    provisional = {
        key: value
        for key, value in target.items()
        if key not in {"tree_sha256", "file_count", "total_size_bytes"}
    }
    target["tree_sha256"] = hashlib.sha256(
        json.dumps(provisional, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    args.target_dataset_manifest.write_bytes(_json_bytes(target))

    with pytest.raises(ContractError, match="official source identity|bindings differ"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_sidecar_identity_migration_rejects_noncanonical_http_date(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    receipt = json.loads(args.source_receipt.read_text(encoding="ascii"))
    receipt["official_archive"]["last_modified"] = "Thu,, 15 Sep 2022 17:47:47 GMT"
    args.source_receipt.write_bytes(_json_bytes(receipt))

    with pytest.raises(ContractError, match="official archive binding differs"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_sidecar_identity_migration_rejects_target_manifest_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture(tmp_path)

    def mutate_target(**_: object) -> int:
        args.target_dataset_manifest.write_bytes(b"changed during verification")
        return len(b"immutable-sidecar-shard")

    monkeypatch.setattr(migrate, "_verify_shards", mutate_target)

    with pytest.raises(ContractError, match="target dataset manifest changed"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()
