from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import build_dataset_file_manifest
from tools import audit_calvin_official_source as audit


def _archive_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, payload in files.items():
            archive.writestr(path, payload)
    return stream.getvalue()


def _archive_parts(payload: bytes) -> tuple[bytes, bytes, audit.ZipDirectoryMetadata]:
    tail = payload[-min(len(payload), 256 * 1024) :]
    metadata = audit.parse_zip_directory_metadata(tail, archive_size=len(payload))
    start = metadata.central_directory_offset
    stop = start + metadata.central_directory_size
    return payload[start:stop], tail, metadata


def _fixture(tmp_path: Path) -> argparse.Namespace:
    split = tmp_path / "training"
    (split / "nested").mkdir(parents=True)
    (split / "a.bin").write_bytes(b"alpha")
    (split / "nested" / "b.bin").write_bytes(b"beta")
    manifest = build_dataset_file_manifest(
        split,
        dataset_id="mees/calvin-debug-dataset",
        dataset_revision="sha256:" + "1" * 64,
        split_name="training",
        relative_paths=("a.bin", "nested/b.bin"),
    )
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), sort_keys=True))
    archive_files = {
        f"{audit.OFFICIAL_TRAINING_PREFIX}a.bin": b"alpha",
        f"{audit.OFFICIAL_TRAINING_PREFIX}nested/b.bin": b"beta",
        **{
            f"{audit.OFFICIAL_TRAINING_PREFIX}{path}": path.encode("ascii")
            for path in audit.OFFICIAL_NON_RUNTIME_TRAINING_FILES
        },
        "task_ABC_D/validation/ignored.bin": b"ignored",
    }
    payload = _archive_bytes(archive_files)
    central, tail, _metadata = _archive_parts(payload)
    central_path = tmp_path / "central.bin"
    tail_path = tmp_path / "tail.bin"
    central_path.write_bytes(central)
    tail_path.write_bytes(tail)
    return argparse.Namespace(
        split_root=split,
        source_manifest=manifest_path,
        central_directory=central_path,
        archive_tail=tail_path,
        archive_content_length=len(payload),
        archive_last_modified="Thu, 15 Sep 2022 17:47:47 GMT",
        archive_etag='"fixture"',
        expected_central_directory_sha256=hashlib.sha256(central).hexdigest(),
        expected_tail_sha256=hashlib.sha256(tail).hexdigest(),
        output_dir=tmp_path / "receipt",
        workers=2,
        progress_every=1,
    )


def test_official_receipt_full_scan_publishes_content_identity(tmp_path: Path) -> None:
    args = _fixture(tmp_path)

    report = audit._run(args)  # noqa: SLF001

    receipt = json.loads((args.output_dir / audit.RECEIPT_NAME).read_text())
    migrated = json.loads((args.output_dir / audit.MIGRATED_MANIFEST_NAME).read_text())
    assert receipt == report
    assert receipt["schema"] == audit.RECEIPT_SCHEMA
    assert receipt["verified_content"]["all_manifest_sha256_matches"] is True
    assert receipt["verified_content"]["all_official_crc32_matches"] is True
    assert receipt["training_authorized"] is False
    assert migrated["dataset_id"] == audit.OFFICIAL_DATASET_ID
    assert migrated["dataset_revision"] == (
        "sha256:" + receipt["verified_content"]["content_sha256"]
    )
    assert (
        receipt["migrated_manifest"]["file_sha256"]
        == hashlib.sha256((args.output_dir / audit.MIGRATED_MANIFEST_NAME).read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        audit._run(args)  # noqa: SLF001


def test_official_receipt_rejects_source_crc_or_sha_drift_without_publication(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    (args.split_root / "a.bin").write_bytes(b"ALPHA")

    with pytest.raises(ContractError, match="SHA-256|CRC32"):
        audit._run(args)  # noqa: SLF001

    assert not args.output_dir.exists()


def test_official_receipt_rejects_unclassified_archive_files(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    payload = _archive_bytes(
        {
            f"{audit.OFFICIAL_TRAINING_PREFIX}a.bin": b"alpha",
            f"{audit.OFFICIAL_TRAINING_PREFIX}nested/b.bin": b"beta",
            **{
                f"{audit.OFFICIAL_TRAINING_PREFIX}{path}": path.encode("ascii")
                for path in audit.OFFICIAL_NON_RUNTIME_TRAINING_FILES
            },
            f"{audit.OFFICIAL_TRAINING_PREFIX}unexpected.bin": b"unexpected",
        }
    )
    central, tail, _metadata = _archive_parts(payload)
    args.central_directory.write_bytes(central)
    args.archive_tail.write_bytes(tail)
    args.archive_content_length = len(payload)
    args.expected_central_directory_sha256 = hashlib.sha256(central).hexdigest()
    args.expected_tail_sha256 = hashlib.sha256(tail).hexdigest()

    with pytest.raises(ContractError, match="non-runtime inventory"):
        audit._run(args)  # noqa: SLF001


def test_zip_central_parser_supports_zip64_sizes_and_rejects_unsafe_paths() -> None:
    filename = b"task_ABC_D/training/a.bin"
    size = 2**32 + 17
    compressed_size = 2**32 + 9
    local_offset = 2**32 + 3
    zip64_payload = struct.pack("<QQQ", size, compressed_size, local_offset)
    extra = struct.pack("<HH", 0x0001, len(zip64_payload)) + zip64_payload
    header = audit._CENTRAL_HEADER.pack(  # noqa: SLF001
        audit._CENTRAL_SIGNATURE,  # noqa: SLF001
        45,
        45,
        0x800,
        zipfile.ZIP_DEFLATED,
        0,
        0,
        0x12345678,
        0xFFFFFFFF,
        0xFFFFFFFF,
        len(filename),
        len(extra),
        0,
        0,
        0,
        0,
        0xFFFFFFFF,
    )

    entries = tuple(
        audit.iter_zip_central_entries(
            header + filename + extra,
            expected_entry_count=1,
        )
    )

    assert entries[0].size_bytes == size
    assert entries[0].compressed_size_bytes == compressed_size
    assert entries[0].crc32 == 0x12345678

    unsafe_name = b"../escape.bin"
    unsafe_header = audit._CENTRAL_HEADER.pack(  # noqa: SLF001
        audit._CENTRAL_SIGNATURE,  # noqa: SLF001
        20,
        20,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        len(unsafe_name),
        0,
        0,
        0,
        0,
        0,
        0,
    )
    with pytest.raises(ContractError, match="normalized and relative"):
        tuple(
            audit.iter_zip_central_entries(
                unsafe_header + unsafe_name,
                expected_entry_count=1,
            )
        )


def test_zip_tail_parser_rejects_central_directory_overlap() -> None:
    payload = _archive_bytes({"a.bin": b"alpha"})
    central, tail, metadata = _archive_parts(payload)
    changed = bytearray(tail)
    overlap_absolute = max(
        metadata.central_directory_offset,
        metadata.archive_size - len(tail),
    )
    changed[overlap_absolute - (metadata.archive_size - len(tail))] ^= 1

    with pytest.raises(ContractError, match="disagree"):
        audit._validate_central_tail_overlap(  # noqa: SLF001
            metadata=metadata,
            central_directory=central,
            tail=bytes(changed),
        )
