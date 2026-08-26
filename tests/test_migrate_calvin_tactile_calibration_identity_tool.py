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
from picf_next.data.calvin_tactile import (
    CALVIN_TACTILE_SOURCE_COMMIT,
    CALVIN_TACTILE_SOURCE_FILES_SHA256,
)
from picf_next.data.calvin_tactile_calibration import (
    CALVIN_TACTILE_BACKGROUND_ALGORITHM,
    CALVIN_TACTILE_BACKGROUND_ARCHIVE_SCHEMA,
    CALVIN_TACTILE_CALIBRATION_SCHEMA,
    canonical_calibration_receipt_sha256,
    load_calvin_tactile_backgrounds,
    tactile_background_sha256,
)
from picf_next.data.dataset_manifest import (
    build_dataset_file_manifest,
    content_identified_dataset_manifest,
    file_sha256,
)
from tools import audit_calvin_official_source as source_audit
from tools import migrate_calvin_tactile_calibration_identity as migrate


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode(
            "ascii"
        )
        + b"\n"
    )


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    split.mkdir()
    (split / "scene_info.npy").write_bytes(b"selected CALVIN bytes")
    source_manifest = build_dataset_file_manifest(
        split,
        dataset_id="alias.example/calvin-task-abc-d",
        dataset_revision="alias-revision",
        split_name="training",
        relative_paths=("scene_info.npy",),
    )
    official_source_manifest = build_dataset_file_manifest(
        split,
        dataset_id="original.example/calvin-task-abc-d",
        dataset_revision="original-revision",
        split_name="training",
        relative_paths=("scene_info.npy",),
    )
    target_manifest = content_identified_dataset_manifest(
        source_manifest,
        dataset_id=source_audit.OFFICIAL_DATASET_ID,
    )
    source_manifest_path = tmp_path / "alias-source-manifest.json"
    official_source_manifest_path = tmp_path / "original-official-source-manifest.json"
    target_manifest_path = tmp_path / source_audit.MIGRATED_MANIFEST_NAME
    source_manifest_path.write_bytes(_json_bytes(source_manifest.to_dict()))
    official_source_manifest_path.write_bytes(_json_bytes(official_source_manifest.to_dict()))
    target_manifest_path.write_bytes(_json_bytes(target_manifest.to_dict()))

    official_receipt = {
        **_official_source_claims(len(target_manifest.files)),
        "schema": source_audit.RECEIPT_SCHEMA,
        "source_manifest": {
            "file_sha256": file_sha256(official_source_manifest_path),
            "tree_sha256": official_source_manifest.tree_sha256,
            "declared_dataset_id": official_source_manifest.dataset_id,
            "declared_dataset_revision": official_source_manifest.dataset_revision,
        },
        "migrated_manifest": {
            "file_name": source_audit.MIGRATED_MANIFEST_NAME,
            "file_sha256": file_sha256(target_manifest_path),
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
    official_receipt_path = tmp_path / "official-source-receipt.json"
    official_receipt_path.write_bytes(_json_bytes(official_receipt))

    left_background = np.full((160, 120, 3), 20.0, dtype=np.float32)
    right_background = np.full((160, 120, 3), 40.0, dtype=np.float32)
    selected_steps = (1, 7)
    candidate_steps = (1, 4, 7)
    source_archive_path = tmp_path / "source-tactile-backgrounds.npz"
    np.savez_compressed(
        source_archive_path,
        schema=np.asarray(CALVIN_TACTILE_BACKGROUND_ARCHIVE_SCHEMA),
        left_digit=left_background,
        right_digit=right_background,
        left_digit_selected_steps=np.asarray(selected_steps, dtype=np.int64),
        right_digit_selected_steps=np.asarray(selected_steps, dtype=np.int64),
    )
    streams = {
        name: {
            "background_sha256": tactile_background_sha256(background),
            "candidate_count": len(candidate_steps),
            "candidate_steps_sha256": _digest(np.asarray(candidate_steps, dtype="<i8").tobytes()),
            "selected_count": len(selected_steps),
            "selected_source_sha256": _digest(f"{name}-sources".encode("ascii")),
            "selected_steps": list(selected_steps),
            "validity_threshold_m": 1e-4,
        }
        for name, background in (
            ("left_digit", left_background),
            ("right_digit", right_background),
        )
    }
    tactile_receipt = {
        "schema": CALVIN_TACTILE_CALIBRATION_SCHEMA,
        "dataset": {
            "dataset_id": source_manifest.dataset_id,
            "dataset_revision": source_manifest.dataset_revision,
            "file_count": len(source_manifest.files),
            "manifest_sha256": file_sha256(source_manifest_path),
            "split_name": source_manifest.split_name,
            "tree_sha256": source_manifest.tree_sha256,
        },
        "sampling": {
            "sample_count": len(candidate_steps),
            "sampled_steps_sha256": _digest(np.asarray(candidate_steps, dtype="<i8").tobytes()),
            "tactile_audit_sha256": _digest(b"tactile audit"),
            "visual_review_manifest_sha256": _digest(b"visual review"),
        },
        "official_calvin_source": {
            "commit": CALVIN_TACTILE_SOURCE_COMMIT,
            "files_sha256": CALVIN_TACTILE_SOURCE_FILES_SHA256,
        },
        "calibration": {
            "algorithm": CALVIN_TACTILE_BACKGROUND_ALGORITHM,
            "background_noise_ceiling_m": 1e-6,
            "streams": streams,
        },
        "archive": {
            "path": str(source_archive_path.resolve()),
            "sha256": file_sha256(source_archive_path),
        },
    }
    tactile_receipt["receipt_payload_sha256"] = canonical_calibration_receipt_sha256(
        tactile_receipt
    )
    source_tactile_receipt_path = tmp_path / "source-tactile-receipt.json"
    source_tactile_receipt_path.write_bytes(_json_bytes(tactile_receipt))
    return argparse.Namespace(
        source_dataset_manifest=source_manifest_path,
        official_source_dataset_manifest=official_source_manifest_path,
        target_dataset_manifest=target_manifest_path,
        official_source_receipt=official_receipt_path,
        expected_official_source_receipt_sha256=file_sha256(official_receipt_path),
        source_tactile_archive=source_archive_path,
        expected_source_tactile_archive_sha256=file_sha256(source_archive_path),
        source_tactile_receipt=source_tactile_receipt_path,
        expected_source_tactile_receipt_sha256=file_sha256(source_tactile_receipt_path),
        output_dir=tmp_path / "identity-view",
    )


def _reseal_tactile_receipt(args: argparse.Namespace, receipt: dict[str, object]) -> None:
    receipt.pop("receipt_payload_sha256", None)
    receipt["receipt_payload_sha256"] = canonical_calibration_receipt_sha256(receipt)
    args.source_tactile_receipt.write_bytes(_json_bytes(receipt))
    args.expected_source_tactile_receipt_sha256 = file_sha256(args.source_tactile_receipt)


def test_tactile_identity_migration_preserves_archive_and_all_semantics(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    source_archive_bytes = args.source_tactile_archive.read_bytes()
    source_receipt = json.loads(args.source_tactile_receipt.read_text(encoding="ascii"))
    target_manifest = json.loads(args.target_dataset_manifest.read_text(encoding="ascii"))

    report = migrate._run(args)  # noqa: SLF001

    target_archive = args.output_dir / migrate.OUTPUT_ARCHIVE_NAME
    target_receipt_path = args.output_dir / migrate.OUTPUT_CALIBRATION_RECEIPT_NAME
    target_receipt = json.loads(target_receipt_path.read_text(encoding="ascii"))
    written_report = json.loads(
        (args.output_dir / migrate.OUTPUT_RECEIPT_NAME).read_text(encoding="ascii")
    )
    expected_receipt = copy.deepcopy(source_receipt)
    expected_receipt["dataset"] = {
        "dataset_id": target_manifest["dataset_id"],
        "dataset_revision": target_manifest["dataset_revision"],
        "file_count": target_manifest["file_count"],
        "manifest_sha256": file_sha256(args.target_dataset_manifest),
        "split_name": target_manifest["split_name"],
        "tree_sha256": target_manifest["tree_sha256"],
    }
    expected_receipt["archive"]["path"] = str(target_archive.resolve())
    expected_receipt.pop("receipt_payload_sha256")
    expected_receipt["receipt_payload_sha256"] = canonical_calibration_receipt_sha256(
        expected_receipt
    )

    assert target_archive.read_bytes() == source_archive_bytes
    assert target_receipt == expected_receipt
    assert report == written_report
    assert report["archive_bytes_unchanged"] is True
    assert report["calibration_semantics_unchanged"] is True
    assert report["training_authorized"] is False
    assert set(path.name for path in args.output_dir.iterdir()) == {
        migrate.OUTPUT_ARCHIVE_NAME,
        migrate.OUTPUT_CALIBRATION_RECEIPT_NAME,
        migrate.OUTPUT_RECEIPT_NAME,
    }
    loaded = load_calvin_tactile_backgrounds(
        target_archive,
        target_receipt_path,
        receipt_sha256=file_sha256(target_receipt_path),
        dataset_tree_sha256=target_manifest["tree_sha256"],
    )
    assert loaded.archive_sha256 == args.expected_source_tactile_archive_sha256
    assert loaded.receipt_payload_sha256 == target_receipt["receipt_payload_sha256"]
    with pytest.raises(FileExistsError):
        migrate._run(args)  # noqa: SLF001


@pytest.mark.parametrize(
    "expected_hash_attribute",
    (
        "expected_official_source_receipt_sha256",
        "expected_source_tactile_archive_sha256",
        "expected_source_tactile_receipt_sha256",
    ),
)
def test_tactile_identity_migration_requires_every_pinned_hash(
    tmp_path: Path,
    expected_hash_attribute: str,
) -> None:
    args = _fixture(tmp_path)
    setattr(args, expected_hash_attribute, "0" * 64)

    with pytest.raises(ContractError, match="digest differs from the pinned input"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_tactile_identity_migration_uses_receipts_original_source_manifest(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    args.official_source_dataset_manifest = args.source_dataset_manifest

    with pytest.raises(ContractError, match="source receipt manifest bindings differ"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_tactile_identity_migration_rejects_selected_byte_drift(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    drift_split = tmp_path / "drift" / "training"
    drift_split.mkdir(parents=True)
    (drift_split / "scene_info.npy").write_bytes(b"different CALVIN bytes")
    drifted = build_dataset_file_manifest(
        drift_split,
        dataset_id="alias.example/calvin-task-abc-d",
        dataset_revision="alias-revision",
        split_name="training",
        relative_paths=("scene_info.npy",),
    )
    args.source_dataset_manifest.write_bytes(_json_bytes(drifted.to_dict()))

    with pytest.raises(ContractError, match="changed the selected source bytes"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_tactile_identity_migration_rejects_receipt_identity_drift(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    receipt = json.loads(args.source_tactile_receipt.read_text(encoding="ascii"))
    receipt["dataset"]["dataset_id"] = "wrong-alias"
    _reseal_tactile_receipt(args, receipt)

    with pytest.raises(ContractError, match="identity differs from its manifest"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_tactile_identity_migration_rejects_malformed_semantic_hash(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    receipt = json.loads(args.source_tactile_receipt.read_text(encoding="ascii"))
    receipt["calibration"]["streams"]["left_digit"]["candidate_steps_sha256"] = "bad"
    _reseal_tactile_receipt(args, receipt)

    with pytest.raises(ContractError, match="must be one lowercase SHA-256 digest"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


@pytest.mark.parametrize("changed_input", ("archive", "receipt"))
def test_tactile_identity_migration_rejects_tactile_input_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
) -> None:
    args = _fixture(tmp_path)
    original = migrate.load_calvin_tactile_backgrounds

    def mutate_after_load(*positional: object, **keyword: object) -> object:
        loaded = original(*positional, **keyword)
        path = (
            args.source_tactile_archive
            if changed_input == "archive"
            else args.source_tactile_receipt
        )
        with path.open("ab") as stream:
            stream.write(b"changed during migration")
        return loaded

    monkeypatch.setattr(migrate, "load_calvin_tactile_backgrounds", mutate_after_load)

    with pytest.raises(ContractError, match="changed during tactile calibration migration"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_tactile_identity_migration_cleans_failed_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture(tmp_path)

    def fail_publication(*_: object, **__: object) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(
        migrate,
        "publish_prepared_directory_durable_exclusive",
        fail_publication,
    )

    with pytest.raises(OSError, match="injected publication failure"):
        migrate._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()
    assert not tuple(tmp_path.glob(".identity-view.partial-*"))


def test_tactile_identity_migration_rejects_dangling_output_symlink(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    symlink_target = tmp_path / "unpublished-target"
    args.output_dir.symlink_to(symlink_target, target_is_directory=True)

    with pytest.raises(FileExistsError):
        migrate._run(args)  # noqa: SLF001

    assert args.output_dir.is_symlink()
    assert not symlink_target.exists()
