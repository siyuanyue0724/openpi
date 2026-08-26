from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.data import calvin_official_source as official_source
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_normalization import (
    build_calvin_normalization_artifact,
    content_identified_calvin_normalization_artifact,
    validate_lingbot_calvin_norm_stats,
)
from picf_next.data.dataset_manifest import content_identified_dataset_manifest
from tests.test_calvin_data import _split_manifest, _write_split
from tools import audit_calvin_official_source as source_audit
from tools import migrate_calvin_normalization_identity as migrate


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"


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
    _write_split(split)
    source_manifest = _split_manifest(split)
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
            "file_sha256": _sha256(source_manifest_path),
            "tree_sha256": source_manifest.tree_sha256,
            "declared_dataset_id": source_manifest.dataset_id,
            "declared_dataset_revision": source_manifest.dataset_revision,
        },
        "migrated_manifest": {
            "file_name": source_audit.MIGRATED_MANIFEST_NAME,
            "file_sha256": _sha256(target_manifest_path),
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

    index = CalvinDatasetIndex.load(
        split,
        dataset_id=source_manifest.dataset_id,
        dataset_revision=source_manifest.dataset_revision,
        dataset_manifest=source_manifest,
    )
    source_normalization = build_calvin_normalization_artifact(index)
    source_normalization_path = tmp_path / "source-normalization.json"
    source_normalization_path.write_bytes(_json_bytes(source_normalization))
    return argparse.Namespace(
        source_dataset_manifest=source_manifest_path,
        target_dataset_manifest=target_manifest_path,
        source_receipt=source_receipt_path,
        source_normalization=source_normalization_path,
        expected_source_normalization_sha256=_sha256(source_normalization_path),
        output_dir=tmp_path / "identity-view",
    )


def test_normalization_identity_migration_preserves_every_statistic(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    source = json.loads(args.source_normalization.read_text(encoding="ascii"))

    report = migrate._run(args)  # noqa: SLF001

    target = json.loads((args.output_dir / migrate.OUTPUT_CALVIN_NAME).read_text(encoding="ascii"))
    lingbot = json.loads(
        (args.output_dir / migrate.OUTPUT_LINGBOT_NAME).read_text(encoding="ascii")
    )
    receipt = json.loads(
        (args.output_dir / migrate.OUTPUT_RECEIPT_NAME).read_text(encoding="ascii")
    )
    target_manifest = json.loads(args.target_dataset_manifest.read_text(encoding="ascii"))
    provenance = {"artifact_sha256", "dataset_id", "dataset_revision", "dataset_tree_sha256"}
    assert report == receipt
    assert report["statistics_unchanged"] is True
    assert {key: value for key, value in source.items() if key not in provenance} == {
        key: value for key, value in target.items() if key not in provenance
    }
    assert target["dataset_id"] == target_manifest["dataset_id"]
    assert target["dataset_revision"] == target_manifest["dataset_revision"]
    assert target["dataset_tree_sha256"] == target_manifest["tree_sha256"]
    assert target["artifact_sha256"] != source["artifact_sha256"]
    validate_lingbot_calvin_norm_stats(lingbot)
    assert lingbot["source"]["artifact_sha256"] == target["artifact_sha256"]
    assert lingbot["source"]["dataset_tree_sha256"] == target_manifest["tree_sha256"]
    with pytest.raises(FileExistsError):
        migrate._run(args)  # noqa: SLF001


def test_normalization_identity_migration_rejects_tampered_statistics(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    source = json.loads(args.source_normalization.read_text(encoding="ascii"))
    source["action"]["mean"][0] += 0.01
    args.source_normalization.write_bytes(_json_bytes(source))

    with pytest.raises(ContractError, match="digest differs"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_normalization_identity_migration_rejects_source_identity_drift(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    source = json.loads(args.source_normalization.read_text(encoding="ascii"))
    target_manifest = json.loads(args.target_dataset_manifest.read_text(encoding="ascii"))
    changed = content_identified_calvin_normalization_artifact(
        source,
        dataset_id=target_manifest["dataset_id"],
        dataset_revision=target_manifest["dataset_revision"],
        dataset_tree_sha256=target_manifest["tree_sha256"],
    )
    args.source_normalization.write_bytes(_json_bytes(changed))
    args.expected_source_normalization_sha256 = _sha256(args.source_normalization)

    with pytest.raises(ContractError, match="source normalization identity differs"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (("file_count", 0), ("verification_mode", "partial-scan")),
)
def test_normalization_identity_migration_rejects_incomplete_source_receipt(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    args = _fixture(tmp_path)
    receipt = json.loads(args.source_receipt.read_text(encoding="ascii"))
    verified = receipt["verified_content"]
    assert isinstance(verified, dict)
    verified[field] = value
    args.source_receipt.write_bytes(_json_bytes(receipt))

    with pytest.raises(ContractError, match="has not closed content verification"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_normalization_identity_migration_rejects_training_authorizing_source_receipt(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    receipt = json.loads(args.source_receipt.read_text(encoding="ascii"))
    receipt["training_authorized"] = True
    args.source_receipt.write_bytes(_json_bytes(receipt))

    with pytest.raises(ContractError, match="must not authorize model training"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_normalization_identity_migration_rejects_wrong_official_archive(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    receipt = json.loads(args.source_receipt.read_text(encoding="ascii"))
    archive = receipt["official_archive"]
    assert isinstance(archive, dict)
    archive["etag"] = '"wrong"'
    args.source_receipt.write_bytes(_json_bytes(receipt))

    with pytest.raises(ContractError, match="official archive binding differs"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_normalization_identity_migration_rejects_target_manifest_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture(tmp_path)
    original = migrate.official_lingbot_calvin_norm_stats

    def mutate_target(*positional: object, **keyword: object) -> dict[str, object]:
        result = original(*positional, **keyword)
        args.target_dataset_manifest.write_bytes(b"changed during migration")
        return result

    monkeypatch.setattr(migrate, "official_lingbot_calvin_norm_stats", mutate_target)

    with pytest.raises(ContractError, match="target dataset manifest changed"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()
