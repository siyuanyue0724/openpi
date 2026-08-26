#!/usr/bin/env python3
"""Bind an extracted CALVIN training split to the official task_ABC_D ZIP inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import struct
import sys
import time
import zlib
from array import array
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from picf_next.artifact_io import (
    publish_prepared_directory_durable_exclusive,
    write_bytes_durable_exclusive,
)
from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import CALVIN_SOURCE_COMMIT
from picf_next.data.calvin_official_source import (
    CALVIN_OFFICIAL_ARCHIVE_URL as OFFICIAL_ARCHIVE_URL,
)
from picf_next.data.calvin_official_source import (
    CALVIN_OFFICIAL_DATASET_ID as OFFICIAL_DATASET_ID,
)
from picf_next.data.calvin_official_source import (
    CALVIN_OFFICIAL_MANIFEST_NAME as MIGRATED_MANIFEST_NAME,
)
from picf_next.data.calvin_official_source import (
    CALVIN_OFFICIAL_NON_RUNTIME_TRAINING_FILES as OFFICIAL_NON_RUNTIME_TRAINING_FILES,
)
from picf_next.data.calvin_official_source import (
    CALVIN_OFFICIAL_PUBLISHER_AUTHENTICITY as OFFICIAL_PUBLISHER_AUTHENTICITY,
)
from picf_next.data.calvin_official_source import (
    CALVIN_OFFICIAL_SOURCE_RECEIPT_SCHEMA as RECEIPT_SCHEMA,
)
from picf_next.data.calvin_official_source import (
    CALVIN_OFFICIAL_SOURCE_VERIFICATION_MODE as VERIFICATION_MODE,
)
from picf_next.data.calvin_official_source import (
    CALVIN_OFFICIAL_TRAINING_PREFIX as OFFICIAL_TRAINING_PREFIX,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    DatasetFileRecord,
    content_identified_dataset_manifest,
    file_sha256,
    load_dataset_file_manifest,
)

RECEIPT_NAME = "receipt.json"

_EOCD = struct.Struct("<4s4H2IH")
_ZIP64_LOCATOR = struct.Struct("<4sIQI")
_ZIP64_EOCD = struct.Struct("<4sQ2H2I4Q")
_CENTRAL_HEADER = struct.Struct("<4s6H3I5H2I")
_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_CENTRAL_SIGNATURE = b"PK\x01\x02"


@dataclass(frozen=True, slots=True)
class ZipDirectoryMetadata:
    archive_size: int
    central_directory_offset: int
    central_directory_size: int
    entry_count: int
    eocd_offset: int
    zip64: bool


@dataclass(frozen=True, slots=True)
class ZipCentralEntry:
    path: str
    size_bytes: int
    compressed_size_bytes: int
    crc32: int
    compression_method: int
    flags: int
    is_directory: bool


@dataclass(frozen=True, slots=True)
class OfficialTrainingInventory:
    crc32_by_manifest_index: array
    official_training_file_count: int
    official_training_directory_count: int
    archive_entry_count: int
    compression_method_counts: dict[str, int]


def _require_nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name} must be a nonnegative integer")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _canonical_archive_path(value: str) -> tuple[str, bool]:
    if not value or "\\" in value or "\0" in value:
        raise ContractError("official ZIP paths must use nonempty canonical POSIX syntax")
    is_directory = value.endswith("/")
    normalized = value[:-1] if is_directory else value
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or path.as_posix() != normalized
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ContractError("official ZIP paths must be normalized and relative")
    return value, is_directory


def _find_eocd(tail: bytes) -> int:
    search_start = max(0, len(tail) - (65_535 + _EOCD.size))
    position = len(tail)
    while True:
        position = tail.rfind(_EOCD_SIGNATURE, search_start, position)
        if position < 0:
            raise ContractError("official ZIP tail has no valid end-of-central-directory record")
        if position + _EOCD.size <= len(tail):
            comment_length = _EOCD.unpack_from(tail, position)[-1]
            if position + _EOCD.size + comment_length == len(tail):
                return position


def parse_zip_directory_metadata(tail: bytes, *, archive_size: int) -> ZipDirectoryMetadata:
    """Parse classic or ZIP64 directory bounds from bytes ending at archive EOF."""

    if not isinstance(tail, bytes) or not tail:
        raise ContractError("official ZIP tail must be nonempty immutable bytes")
    archive_size = _require_nonnegative_int(archive_size, "official ZIP archive size")
    if archive_size < len(tail):
        raise ContractError("official ZIP tail is larger than its archive")
    tail_start = archive_size - len(tail)
    eocd_relative = _find_eocd(tail)
    (
        signature,
        disk_number,
        directory_disk,
        entries_on_disk,
        total_entries,
        directory_size,
        directory_offset,
        _comment_length,
    ) = _EOCD.unpack_from(tail, eocd_relative)
    if signature != _EOCD_SIGNATURE or disk_number != 0 or directory_disk != 0:
        raise ContractError("multi-disk or malformed official ZIP archives are unsupported")

    saturated = (
        entries_on_disk == 0xFFFF
        or total_entries == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    )
    if saturated:
        locator_relative = eocd_relative - _ZIP64_LOCATOR.size
        if locator_relative < 0:
            raise ContractError("official ZIP64 archive has no locator")
        locator = _ZIP64_LOCATOR.unpack_from(tail, locator_relative)
        if locator[0] != _ZIP64_LOCATOR_SIGNATURE or locator[1] != 0 or locator[3] != 1:
            raise ContractError("official ZIP64 locator is malformed or multi-disk")
        zip64_eocd_offset = locator[2]
        zip64_relative = zip64_eocd_offset - tail_start
        if zip64_relative < 0 or zip64_relative + _ZIP64_EOCD.size > len(tail):
            raise ContractError("official ZIP64 end record is outside the supplied tail")
        zip64 = _ZIP64_EOCD.unpack_from(tail, zip64_relative)
        if zip64[0] != _ZIP64_EOCD_SIGNATURE or zip64[1] < 44:
            raise ContractError("official ZIP64 end record is malformed")
        if zip64_relative + 12 + zip64[1] != locator_relative:
            raise ContractError("official ZIP64 end record does not terminate at its locator")
        if zip64[4] != 0 or zip64[5] != 0 or zip64[6] != zip64[7]:
            raise ContractError("multi-disk official ZIP64 archives are unsupported")
        entry_count = zip64[7]
        directory_size = zip64[8]
        directory_offset = zip64[9]
        classic_pairs = (
            (entries_on_disk, entry_count, 0xFFFF),
            (total_entries, entry_count, 0xFFFF),
            (_EOCD.unpack_from(tail, eocd_relative)[5], directory_size, 0xFFFFFFFF),
            (_EOCD.unpack_from(tail, eocd_relative)[6], directory_offset, 0xFFFFFFFF),
        )
        if any(
            classic != sentinel and classic != extended
            for classic, extended, sentinel in classic_pairs
        ):
            raise ContractError("official ZIP and ZIP64 end records disagree")
        zip64_mode = True
    else:
        if entries_on_disk != total_entries:
            raise ContractError("multi-disk official ZIP archives are unsupported")
        entry_count = total_entries
        zip64_mode = False

    eocd_offset = tail_start + eocd_relative
    values = (entry_count, directory_size, directory_offset)
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ContractError("official ZIP directory bounds are invalid")
    if directory_offset + directory_size > eocd_offset:
        raise ContractError("official ZIP central directory overlaps its end record")
    return ZipDirectoryMetadata(
        archive_size=archive_size,
        central_directory_offset=directory_offset,
        central_directory_size=directory_size,
        entry_count=entry_count,
        eocd_offset=eocd_offset,
        zip64=zip64_mode,
    )


def _zip64_central_values(
    extra: bytes,
    *,
    size_32: int,
    compressed_size_32: int,
    local_offset_32: int,
    disk_start_16: int,
) -> tuple[int, int, int, int]:
    fields: list[tuple[int, bytes]] = []
    cursor = 0
    while cursor < len(extra):
        if cursor + 4 > len(extra):
            raise ContractError("official ZIP central extra field is truncated")
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        stop = cursor + field_size
        if stop > len(extra):
            raise ContractError("official ZIP central extra field exceeds its record")
        fields.append((field_id, extra[cursor:stop]))
        cursor = stop
    zip64_fields = [payload for field_id, payload in fields if field_id == 0x0001]
    needs_zip64 = (
        size_32 == 0xFFFFFFFF
        or compressed_size_32 == 0xFFFFFFFF
        or local_offset_32 == 0xFFFFFFFF
        or disk_start_16 == 0xFFFF
    )
    if not needs_zip64:
        return size_32, compressed_size_32, local_offset_32, disk_start_16
    if len(zip64_fields) != 1:
        raise ContractError("official ZIP64 central record lacks one ZIP64 extra field")
    payload = zip64_fields[0]
    cursor = 0

    def consume(width: int) -> int:
        nonlocal cursor
        if cursor + width > len(payload):
            raise ContractError("official ZIP64 central extra field is truncated")
        value = int.from_bytes(payload[cursor : cursor + width], "little")
        cursor += width
        return value

    size = consume(8) if size_32 == 0xFFFFFFFF else size_32
    compressed = consume(8) if compressed_size_32 == 0xFFFFFFFF else compressed_size_32
    local_offset = consume(8) if local_offset_32 == 0xFFFFFFFF else local_offset_32
    disk_start = consume(4) if disk_start_16 == 0xFFFF else disk_start_16
    return size, compressed, local_offset, disk_start


def iter_zip_central_entries(
    payload: bytes,
    *,
    expected_entry_count: int,
) -> Iterator[ZipCentralEntry]:
    """Yield validated central-directory entries without materializing ZipInfo objects."""

    if not isinstance(payload, bytes):
        raise ContractError("official ZIP central directory must be immutable bytes")
    expected_entry_count = _require_nonnegative_int(
        expected_entry_count,
        "official ZIP entry count",
    )
    cursor = 0
    count = 0
    while cursor < len(payload):
        if cursor + _CENTRAL_HEADER.size > len(payload):
            raise ContractError("official ZIP central directory is truncated")
        values = _CENTRAL_HEADER.unpack_from(payload, cursor)
        if values[0] != _CENTRAL_SIGNATURE:
            raise ContractError("official ZIP central directory has an invalid signature")
        flags = values[3]
        compression_method = values[4]
        crc32 = values[7]
        compressed_size_32 = values[8]
        size_32 = values[9]
        filename_length, extra_length, comment_length = values[10:13]
        disk_start_16 = values[13]
        local_offset_32 = values[16]
        variable_start = cursor + _CENTRAL_HEADER.size
        filename_stop = variable_start + filename_length
        extra_stop = filename_stop + extra_length
        record_stop = extra_stop + comment_length
        if filename_length == 0 or record_stop > len(payload):
            raise ContractError("official ZIP central record has invalid variable fields")
        filename_bytes = payload[variable_start:filename_stop]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        try:
            filename = filename_bytes.decode(encoding)
        except UnicodeDecodeError as error:
            raise ContractError("official ZIP central filename cannot be decoded") from error
        filename, is_directory = _canonical_archive_path(filename)
        size, compressed_size, _local_offset, disk_start = _zip64_central_values(
            payload[filename_stop:extra_stop],
            size_32=size_32,
            compressed_size_32=compressed_size_32,
            local_offset_32=local_offset_32,
            disk_start_16=disk_start_16,
        )
        if flags & 0x1 or disk_start != 0:
            raise ContractError("encrypted or multi-disk official ZIP entries are unsupported")
        yield ZipCentralEntry(
            path=filename,
            size_bytes=size,
            compressed_size_bytes=compressed_size,
            crc32=crc32,
            compression_method=compression_method,
            flags=flags,
            is_directory=is_directory,
        )
        count += 1
        cursor = record_stop
    if count != expected_entry_count:
        raise ContractError("official ZIP central entry count differs from its end record")


def _validate_central_tail_overlap(
    *,
    metadata: ZipDirectoryMetadata,
    central_directory: bytes,
    tail: bytes,
) -> None:
    if len(central_directory) != metadata.central_directory_size:
        raise ContractError("official ZIP central-directory byte count differs from its end record")
    tail_start = metadata.archive_size - len(tail)
    overlap_start = max(metadata.central_directory_offset, tail_start)
    overlap_stop = min(
        metadata.central_directory_offset + metadata.central_directory_size,
        metadata.archive_size,
    )
    if overlap_start >= overlap_stop:
        raise ContractError("official ZIP tail does not overlap its supplied central directory")
    central_slice = central_directory[
        overlap_start - metadata.central_directory_offset : overlap_stop
        - metadata.central_directory_offset
    ]
    tail_slice = tail[overlap_start - tail_start : overlap_stop - tail_start]
    if central_slice != tail_slice:
        raise ContractError("official ZIP tail and central-directory bytes disagree")


def _match_official_training_inventory(
    manifest: DatasetFileManifest,
    central_directory: bytes,
    metadata: ZipDirectoryMetadata,
) -> OfficialTrainingInventory:
    path_to_index = {record.path: index for index, record in enumerate(manifest.files)}
    found = bytearray(len(manifest.files))
    expected_crc32 = array("I", [0]) * len(manifest.files)
    ancillary: set[str] = set()
    training_files = 0
    training_directories = 0
    method_counts: Counter[int] = Counter()
    archive_entries = 0
    for entry in iter_zip_central_entries(
        central_directory,
        expected_entry_count=metadata.entry_count,
    ):
        archive_entries += 1
        if not entry.path.startswith(OFFICIAL_TRAINING_PREFIX):
            continue
        relative = entry.path[len(OFFICIAL_TRAINING_PREFIX) :]
        if entry.is_directory:
            training_directories += 1
            continue
        if not relative:
            raise ContractError("official CALVIN training archive contains an empty file path")
        training_files += 1
        method_counts[entry.compression_method] += 1
        index = path_to_index.get(relative)
        if index is None:
            if relative in ancillary:
                raise ContractError("official CALVIN training archive contains duplicate files")
            ancillary.add(relative)
            continue
        if found[index]:
            raise ContractError("official CALVIN training archive contains duplicate files")
        record = manifest.files[index]
        if record.size_bytes != entry.size_bytes:
            raise ContractError(f"official CALVIN file size differs from manifest: {relative}")
        found[index] = 1
        expected_crc32[index] = entry.crc32

    missing = [record.path for index, record in enumerate(manifest.files) if not found[index]]
    if missing:
        raise ContractError(
            "dataset manifest files are absent from the official CALVIN archive: "
            + ", ".join(missing[:8])
        )
    expected_ancillary = set(OFFICIAL_NON_RUNTIME_TRAINING_FILES)
    if ancillary != expected_ancillary:
        missing_ancillary = sorted(expected_ancillary - ancillary)
        unexpected = sorted(ancillary - expected_ancillary)
        raise ContractError(
            "official CALVIN non-runtime inventory differs; "
            f"missing={missing_ancillary}, unexpected={unexpected}"
        )
    if training_files != len(manifest.files) + len(expected_ancillary):
        raise ContractError("official CALVIN training file accounting is inconsistent")
    return OfficialTrainingInventory(
        crc32_by_manifest_index=expected_crc32,
        official_training_file_count=training_files,
        official_training_directory_count=training_directories,
        archive_entry_count=archive_entries,
        compression_method_counts={str(key): value for key, value in sorted(method_counts.items())},
    )


def _open_regular_file_beneath(root: Path, relative: str) -> int:
    parts = PurePosixPath(relative).parts
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(root, directory_flags)
    except OSError as error:
        raise ContractError(f"CALVIN split root cannot be opened safely: {root}") from error
    try:
        for part in parts[:-1]:
            try:
                child_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise ContractError(f"unsafe CALVIN source path: {relative}") from error
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor
        try:
            return os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as error:
            raise ContractError(f"unsafe CALVIN source path: {relative}") from error
    finally:
        os.close(directory_descriptor)


def _verify_one_local_file(
    root: Path,
    record: DatasetFileRecord,
    expected_crc32: int,
) -> int:
    descriptor = _open_regular_file_beneath(root, record.path)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"CALVIN source is not a regular file: {record.path}")
        digest = hashlib.sha256()
        crc32 = 0
        observed_size = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            observed_size += len(chunk)
            digest.update(chunk)
            crc32 = zlib.crc32(chunk, crc32)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_fingerprint = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_fingerprint = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_fingerprint != after_fingerprint:
        raise ContractError(f"CALVIN source changed while reading: {record.path}")
    if observed_size != record.size_bytes or digest.hexdigest() != record.sha256:
        raise ContractError(f"CALVIN source SHA-256 differs from manifest: {record.path}")
    if crc32 != expected_crc32:
        raise ContractError(f"CALVIN source CRC32 differs from official ZIP: {record.path}")
    return observed_size


def _verify_all_local_files(
    *,
    split_root: Path,
    manifest: DatasetFileManifest,
    expected_crc32: array,
    maximum_workers: int,
    progress_every: int,
) -> None:
    if (
        not isinstance(maximum_workers, int)
        or isinstance(maximum_workers, bool)
        or maximum_workers <= 0
    ):
        raise TypeError("maximum_workers must be a positive integer")
    if (
        not isinstance(progress_every, int)
        or isinstance(progress_every, bool)
        or progress_every <= 0
    ):
        raise TypeError("progress_every must be a positive integer")
    if len(expected_crc32) != len(manifest.files):
        raise ContractError("official CRC inventory differs from dataset manifest length")
    started = time.monotonic()
    completed = 0
    verified_bytes = 0
    next_report = progress_every
    window_size = max(1024, maximum_workers * 64)
    with ThreadPoolExecutor(
        max_workers=maximum_workers, thread_name_prefix="calvin-receipt"
    ) as pool:
        for start in range(0, len(manifest.files), window_size):
            stop = min(start + window_size, len(manifest.files))
            sizes = pool.map(
                lambda item: _verify_one_local_file(split_root, *item),
                zip(manifest.files[start:stop], expected_crc32[start:stop], strict=True),
            )
            for size_bytes in sizes:
                completed += 1
                verified_bytes += size_bytes
                if completed >= next_report or completed == len(manifest.files):
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        json.dumps(
                            {
                                "completed_files": completed,
                                "elapsed_seconds": round(elapsed, 3),
                                "event": "calvin_official_source_progress",
                                "mib_per_second": round(verified_bytes / elapsed / 2**20, 3),
                                "total_files": len(manifest.files),
                                "verified_bytes": verified_bytes,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    while next_report <= completed:
                        next_report += progress_every
    if verified_bytes != manifest.total_size_bytes:
        raise ContractError("verified CALVIN byte count differs from dataset manifest")


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode(
            "ascii"
        )
        + b"\n"
    )


def _publish_receipt(
    *,
    output_dir: Path,
    manifest: DatasetFileManifest,
    receipt: dict[str, object],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    partial.mkdir(parents=False, exist_ok=False)
    try:
        manifest_payload = _json_bytes(manifest.to_dict())
        write_bytes_durable_exclusive(partial / MIGRATED_MANIFEST_NAME, manifest_payload)
        receipt["migrated_manifest"] = {
            "file_name": MIGRATED_MANIFEST_NAME,
            "file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "tree_sha256": manifest.tree_sha256,
        }
        write_bytes_durable_exclusive(partial / RECEIPT_NAME, _json_bytes(receipt))
        publish_prepared_directory_durable_exclusive(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _run(args: argparse.Namespace) -> dict[str, object]:
    split_root = args.split_root.resolve()
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    manifest_path = args.source_manifest.resolve()
    source_manifest_sha256 = file_sha256(manifest_path)
    manifest = load_dataset_file_manifest(manifest_path)
    if file_sha256(manifest_path) != source_manifest_sha256:
        raise ContractError("CALVIN source manifest changed while loading")
    if manifest.split_name != split_root.name:
        raise ContractError("CALVIN source manifest split differs from source root")
    central_path = args.central_directory.resolve()
    tail_path = args.archive_tail.resolve()
    central_directory = central_path.read_bytes()
    tail = tail_path.read_bytes()
    central_sha256 = hashlib.sha256(central_directory).hexdigest()
    tail_sha256 = hashlib.sha256(tail).hexdigest()
    if central_sha256 != _require_sha256(
        args.expected_central_directory_sha256,
        "expected central-directory SHA-256",
    ):
        raise ContractError("official ZIP central-directory SHA-256 changed")
    if tail_sha256 != _require_sha256(args.expected_tail_sha256, "expected tail SHA-256"):
        raise ContractError("official ZIP tail SHA-256 changed")
    metadata = parse_zip_directory_metadata(
        tail,
        archive_size=args.archive_content_length,
    )
    _validate_central_tail_overlap(
        metadata=metadata,
        central_directory=central_directory,
        tail=tail,
    )
    inventory = _match_official_training_inventory(manifest, central_directory, metadata)
    _verify_all_local_files(
        split_root=split_root,
        manifest=manifest,
        expected_crc32=inventory.crc32_by_manifest_index,
        maximum_workers=args.workers,
        progress_every=args.progress_every,
    )
    if file_sha256(manifest_path) != source_manifest_sha256:
        raise ContractError("CALVIN source manifest changed during source verification")
    migrated = content_identified_dataset_manifest(
        manifest,
        dataset_id=OFFICIAL_DATASET_ID,
    )
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "calvin_source_commit": CALVIN_SOURCE_COMMIT,
        "official_archive": {
            "url": OFFICIAL_ARCHIVE_URL,
            "transport": "http",
            "content_length": metadata.archive_size,
            "last_modified": args.archive_last_modified,
            "etag": args.archive_etag,
            "tail_size_bytes": len(tail),
            "tail_sha256": tail_sha256,
            "central_directory_offset": metadata.central_directory_offset,
            "central_directory_size": metadata.central_directory_size,
            "central_directory_sha256": central_sha256,
            "entry_count": metadata.entry_count,
            "zip64": metadata.zip64,
            "publisher_authenticity": OFFICIAL_PUBLISHER_AUTHENTICITY,
        },
        "official_training_inventory": {
            "archive_prefix": OFFICIAL_TRAINING_PREFIX,
            "archive_entry_count": inventory.archive_entry_count,
            "file_count": inventory.official_training_file_count,
            "directory_count": inventory.official_training_directory_count,
            "compression_method_counts": inventory.compression_method_counts,
            "excluded_non_runtime_files": list(OFFICIAL_NON_RUNTIME_TRAINING_FILES),
        },
        "source_manifest": {
            "file_sha256": source_manifest_sha256,
            "tree_sha256": manifest.tree_sha256,
            "declared_dataset_id": manifest.dataset_id,
            "declared_dataset_revision": manifest.dataset_revision,
        },
        "verified_content": {
            "dataset_id": migrated.dataset_id,
            "dataset_revision": migrated.dataset_revision,
            "content_sha256": migrated.content_sha256,
            "split_name": migrated.split_name,
            "file_count": len(migrated.files),
            "total_size_bytes": migrated.total_size_bytes,
            "verification_mode": VERIFICATION_MODE,
            "all_manifest_sha256_matches": True,
            "all_official_crc32_matches": True,
            "official_inventory_exact_after_declared_exclusions": True,
        },
        "training_authorized": False,
        "training_authorization_reason": (
            "source receipt closes dataset provenance only; model gates remain independent"
        ),
    }
    output_dir = args.output_dir.resolve()
    _publish_receipt(output_dir=output_dir, manifest=migrated, receipt=receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--central-directory", type=Path, required=True)
    parser.add_argument("--archive-tail", type=Path, required=True)
    parser.add_argument("--archive-content-length", type=int, required=True)
    parser.add_argument("--archive-last-modified", required=True)
    parser.add_argument("--archive-etag", required=True)
    parser.add_argument("--expected-central-directory-sha256", required=True)
    parser.add_argument("--expected-tail-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()
    receipt = _run(args)
    verified_content = receipt.get("verified_content")
    if not isinstance(verified_content, dict):
        raise RuntimeError("CALVIN source receipt lacks verified content")
    print(
        json.dumps(
            {
                "dataset_revision": verified_content["dataset_revision"],
                "output_dir": str(args.output_dir.resolve()),
                "schema": receipt["schema"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
