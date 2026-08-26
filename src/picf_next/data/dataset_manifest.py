"""Content-addressed manifests for immutable training dataset trees."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from picf_next.contracts import ContractError

DATASET_FILE_MANIFEST_SCHEMA = "picf-next.dataset-file-manifest.v1"
DATASET_CONTENT_IDENTITY_SCHEMA = "picf-next.dataset-content-identity.v1"
DATASET_RUNTIME_VERIFICATION_MODE = (
    "picf-next.content-addressed-manifest-with-verified-runtime-reads.v1"
)
DATASET_RUNTIME_BINDING_FIELDS = frozenset(
    {
        "dataset_file_count",
        "dataset_total_size_bytes",
        "dataset_tree_sha256",
        "dataset_manifest_self_consistent",
        "dataset_full_tree_rescanned",
        "dataset_runtime_verified_read_required",
        "dataset_runtime_probe_file_count",
        "dataset_runtime_probe_sha256",
        "dataset_verification_mode",
    }
)
_MANIFEST_FIELDS = {
    "dataset_id",
    "dataset_revision",
    "file_count",
    "files",
    "schema",
    "split_name",
    "total_size_bytes",
    "tree_sha256",
}
_FILE_FIELDS = {"path", "sha256", "size_bytes"}


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name} must be a nonnegative integer")
    return value


def _relative_posix_path(value: object) -> str:
    text = _require_text(value, "dataset manifest path")
    if "\\" in text or "\0" in text:
        raise ContractError("dataset manifest paths must use canonical POSIX syntax")
    path = PurePosixPath(text)
    invalid_part = any(part in {"", ".", ".."} for part in path.parts)
    if path.is_absolute() or path.as_posix() != text or invalid_part:
        raise ContractError("dataset manifest paths must be normalized and relative")
    return text


def _consume_file_descriptor(
    descriptor: int,
    path_label: object,
    *,
    retain_bytes: bool,
    maximum_bytes: int | None = None,
) -> tuple[int, str, bytes | None]:
    if maximum_bytes is not None and (
        not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes <= 0
    ):
        raise TypeError("maximum_bytes must be a positive integer")
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ContractError(f"dataset source path is not a regular file: {path_label}")
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise ContractError(f"dataset file exceeds the verified-read byte limit: {path_label}")
    digest = hashlib.sha256()
    retained = bytearray() if retain_bytes else None
    while chunk := os.read(descriptor, 8 * 1024 * 1024):
        digest.update(chunk)
        if retained is not None:
            retained.extend(chunk)
            if maximum_bytes is not None and len(retained) > maximum_bytes:
                raise ContractError(
                    f"dataset file exceeds the verified-read byte limit: {path_label}"
                )
    after = os.fstat(descriptor)
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if fingerprint_before != fingerprint_after:
        raise ContractError(f"dataset source file changed while hashing: {path_label}")
    payload = bytes(retained) if retained is not None else None
    return before.st_size, digest.hexdigest(), payload


def _hash_file_descriptor(descriptor: int, path_label: object) -> tuple[int, str]:
    size_bytes, digest, _ = _consume_file_descriptor(
        descriptor,
        path_label,
        retain_bytes=False,
    )
    return size_bytes, digest


def _hash_regular_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError(f"dataset source file cannot be opened safely: {path}") from error
    try:
        return _hash_file_descriptor(descriptor, path)
    finally:
        os.close(descriptor)


def _open_regular_file_beneath(root: Path, relative: str) -> int:
    parts = PurePosixPath(relative).parts
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(root, directory_flags)
    except OSError as error:
        raise ContractError(f"dataset split root cannot be opened safely: {root}") from error
    try:
        for part in parts[:-1]:
            try:
                child_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise ContractError(
                    "dataset manifest source path must not use symlinks or unsafe "
                    f"components: {relative}"
                ) from error
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor
        try:
            return os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as error:
            raise ContractError(
                "dataset manifest source path must not use symlinks or unsafe "
                f"components: {relative}"
            ) from error
    finally:
        os.close(directory_descriptor)


def _hash_regular_file_beneath(root: Path, relative: str) -> tuple[int, str]:
    """Hash one file by descriptor-relative traversal beneath an opened root."""

    file_descriptor = _open_regular_file_beneath(root, relative)
    try:
        return _hash_file_descriptor(file_descriptor, relative)
    finally:
        os.close(file_descriptor)


def file_sha256(path: str | Path) -> str:
    """Return a race-checked digest for one regular non-symlink file."""

    return _hash_regular_file(Path(path))[1]


def _hash_windows_beneath(
    root: Path,
    relative_paths: tuple[str, ...],
    *,
    maximum_workers: int,
) -> Iterator[tuple[tuple[str, tuple[int, str]], ...]]:
    """Hash ordered paths with bounded concurrency and bounded Future count."""

    window_size = max(1024, maximum_workers * 64)
    if maximum_workers == 1:
        for start in range(0, len(relative_paths), window_size):
            window = relative_paths[start : start + window_size]
            yield tuple(
                (relative, _hash_regular_file_beneath(root, relative)) for relative in window
            )
        return

    with ThreadPoolExecutor(
        max_workers=maximum_workers,
        thread_name_prefix="dataset-sha256",
    ) as executor:
        for start in range(0, len(relative_paths), window_size):
            window = relative_paths[start : start + window_size]
            hashes = executor.map(
                lambda relative: _hash_regular_file_beneath(root, relative),
                window,
            )
            yield tuple(zip(window, hashes, strict=True))


@dataclass(frozen=True, slots=True)
class DatasetFileRecord:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_posix_path(self.path)
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ContractError("dataset manifest file size must be a nonnegative integer")
        _require_sha256(self.sha256, "dataset manifest file sha256")

    @classmethod
    def from_dict(cls, payload: object) -> DatasetFileRecord:
        if not isinstance(payload, Mapping) or set(payload) != _FILE_FIELDS:
            raise ContractError("dataset manifest file record fields differ from schema")
        return cls(
            path=_relative_posix_path(payload["path"]),
            size_bytes=_require_nonnegative_int(
                payload["size_bytes"], "dataset manifest file size"
            ),
            sha256=_require_sha256(payload["sha256"], "dataset manifest file sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class DatasetFileManifest:
    dataset_id: str
    dataset_revision: str
    split_name: str
    files: tuple[DatasetFileRecord, ...]
    tree_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.dataset_id, "dataset_id")
        _require_text(self.dataset_revision, "dataset_revision")
        _require_text(self.split_name, "split_name")
        if not self.files:
            raise ContractError("dataset file manifest cannot be empty")
        previous_path: str | None = None
        for record in self.files:
            if previous_path is not None and record.path <= previous_path:
                raise ContractError("dataset manifest file paths must be unique and sorted")
            previous_path = record.path
        _require_sha256(self.tree_sha256, "dataset manifest tree sha256")
        if self.tree_sha256 != self.computed_tree_sha256:
            raise ContractError("dataset manifest tree SHA-256 changed")

    @property
    def total_size_bytes(self) -> int:
        return sum(record.size_bytes for record in self.files)

    @property
    def computed_tree_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._tree_payload())).hexdigest()

    @property
    def content_sha256(self) -> str:
        """Hash only the split name and immutable file bytes, excluding labels."""

        payload = {
            "files": [record.to_dict() for record in self.files],
            "schema": DATASET_CONTENT_IDENTITY_SCHEMA,
            "split_name": self.split_name,
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    def record_for(self, relative_path: str) -> DatasetFileRecord:
        """Return one canonical record or fail closed when it was not inventoried."""

        relative = _relative_posix_path(relative_path)
        lower = 0
        upper = len(self.files)
        while lower < upper:
            middle = (lower + upper) // 2
            candidate = self.files[middle]
            if candidate.path < relative:
                lower = middle + 1
            else:
                upper = middle
        if lower < len(self.files) and self.files[lower].path == relative:
            return self.files[lower]
        raise ContractError(f"dataset file is absent from the frozen manifest: {relative}")

    def _tree_payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "files": [record.to_dict() for record in self.files],
            "schema": DATASET_FILE_MANIFEST_SCHEMA,
            "split_name": self.split_name,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._tree_payload(),
            "file_count": len(self.files),
            "total_size_bytes": self.total_size_bytes,
            "tree_sha256": self.tree_sha256,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DatasetFileManifest:
        if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_FIELDS:
            raise ContractError("dataset file manifest fields differ from schema")
        if payload["schema"] != DATASET_FILE_MANIFEST_SCHEMA:
            raise ContractError("dataset file manifest schema changed")
        raw_files = payload["files"]
        if not isinstance(raw_files, list):
            raise ContractError("dataset file manifest files must be a list")
        manifest = cls(
            dataset_id=_require_text(payload["dataset_id"], "dataset_id"),
            dataset_revision=_require_text(payload["dataset_revision"], "dataset_revision"),
            split_name=_require_text(payload["split_name"], "split_name"),
            files=tuple(DatasetFileRecord.from_dict(record) for record in raw_files),
            tree_sha256=_require_sha256(payload["tree_sha256"], "dataset manifest tree sha256"),
        )
        if (
            not isinstance(payload["file_count"], int)
            or isinstance(payload["file_count"], bool)
            or payload["file_count"] != len(manifest.files)
        ):
            raise ContractError("dataset file manifest count is inconsistent")
        if (
            not isinstance(payload["total_size_bytes"], int)
            or isinstance(payload["total_size_bytes"], bool)
            or payload["total_size_bytes"] != manifest.total_size_bytes
        ):
            raise ContractError("dataset file manifest byte count is inconsistent")
        return manifest


def build_dataset_file_manifest(
    split_root: str | Path,
    *,
    dataset_id: str,
    dataset_revision: str,
    split_name: str,
    relative_paths: Iterable[str | Path],
    maximum_workers: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DatasetFileManifest:
    """Hash an explicit immutable file set under one split root."""

    if (
        not isinstance(maximum_workers, int)
        or isinstance(maximum_workers, bool)
        or maximum_workers <= 0
    ):
        raise TypeError("maximum_workers must be a positive integer")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    root = Path(split_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    normalized = tuple(
        sorted(_relative_posix_path(Path(path).as_posix()) for path in relative_paths)
    )
    if not normalized or len(set(normalized)) != len(normalized):
        raise ContractError("dataset manifest inputs must be nonempty and unique")
    records: list[DatasetFileRecord] = []
    for window in _hash_windows_beneath(
        root,
        normalized,
        maximum_workers=maximum_workers,
    ):
        records.extend(
            DatasetFileRecord(relative, size_bytes, digest)
            for relative, (size_bytes, digest) in window
        )
        if progress_callback is not None:
            progress_callback(len(records), len(normalized))
    provisional = {
        "dataset_id": _require_text(dataset_id, "dataset_id"),
        "dataset_revision": _require_text(dataset_revision, "dataset_revision"),
        "files": [record.to_dict() for record in records],
        "schema": DATASET_FILE_MANIFEST_SCHEMA,
        "split_name": _require_text(split_name, "split_name"),
    }
    tree_sha256 = hashlib.sha256(_canonical_json(provisional)).hexdigest()
    return DatasetFileManifest(
        dataset_id,
        dataset_revision,
        split_name,
        tuple(records),
        tree_sha256,
    )


def load_dataset_file_manifest(path: str | Path) -> DatasetFileManifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(
            f"dataset file manifest is not valid ASCII JSON: {manifest_path}"
        ) from error
    return DatasetFileManifest.from_dict(payload)


def content_identified_dataset_manifest(
    manifest: DatasetFileManifest,
    *,
    dataset_id: str,
) -> DatasetFileManifest:
    """Return the same immutable file set under a content-derived identity."""

    if not isinstance(manifest, DatasetFileManifest):
        raise TypeError("content identification requires a DatasetFileManifest")
    identified_dataset_id = _require_text(dataset_id, "dataset_id")
    dataset_revision = f"sha256:{manifest.content_sha256}"
    provisional = {
        "dataset_id": identified_dataset_id,
        "dataset_revision": dataset_revision,
        "files": [record.to_dict() for record in manifest.files],
        "schema": DATASET_FILE_MANIFEST_SCHEMA,
        "split_name": manifest.split_name,
    }
    tree_sha256 = hashlib.sha256(_canonical_json(provisional)).hexdigest()
    return DatasetFileManifest(
        dataset_id=identified_dataset_id,
        dataset_revision=dataset_revision,
        split_name=manifest.split_name,
        files=manifest.files,
        tree_sha256=tree_sha256,
    )


def read_verified_dataset_file(
    manifest: DatasetFileManifest,
    split_root: str | Path,
    relative_path: str,
    *,
    maximum_bytes: int,
) -> bytes:
    """Read exact manifest-pinned bytes from one non-symlink source inode."""

    if not isinstance(manifest, DatasetFileManifest):
        raise TypeError("verified dataset reads require a DatasetFileManifest")
    relative = _relative_posix_path(relative_path)
    if not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes <= 0:
        raise TypeError("maximum_bytes must be a positive integer")
    expected = manifest.record_for(relative)
    if expected.size_bytes > maximum_bytes:
        raise ContractError(f"dataset file exceeds the verified-read byte limit: {relative}")
    root = Path(split_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    descriptor = _open_regular_file_beneath(root, relative)
    try:
        size_bytes, digest, payload = _consume_file_descriptor(
            descriptor,
            relative,
            retain_bytes=True,
            maximum_bytes=maximum_bytes,
        )
    finally:
        os.close(descriptor)
    if size_bytes != expected.size_bytes or digest != expected.sha256 or payload is None:
        raise ContractError(f"dataset source file differs from frozen manifest: {relative}")
    return payload


def read_sha256_verified_file_beneath(
    root: str | Path,
    relative_path: str,
    *,
    expected_sha256: str,
    maximum_bytes: int,
) -> bytes:
    """Read bounded bytes from one pinned regular inode beneath ``root``."""

    relative = _relative_posix_path(relative_path)
    expected = _require_sha256(expected_sha256, "expected file sha256")
    if not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes <= 0:
        raise TypeError("maximum_bytes must be a positive integer")
    resolved_root = Path(root).resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(resolved_root)
    descriptor = _open_regular_file_beneath(resolved_root, relative)
    try:
        _, digest, payload = _consume_file_descriptor(
            descriptor,
            relative,
            retain_bytes=True,
            maximum_bytes=maximum_bytes,
        )
    finally:
        os.close(descriptor)
    if digest != expected or payload is None:
        raise ContractError(f"content hash mismatch: {relative}")
    return payload


def validate_dataset_files(
    manifest: DatasetFileManifest,
    split_root: str | Path,
    *,
    dataset_id: str,
    dataset_revision: str,
    split_name: str,
    verify_hashes: bool = True,
    maximum_workers: int = 1,
) -> dict[str, object]:
    """Validate identity and, when requested, every source byte before decoding."""

    if not isinstance(manifest, DatasetFileManifest):
        raise TypeError("dataset validation requires a DatasetFileManifest")
    expected_identity = (dataset_id, dataset_revision, split_name)
    if (manifest.dataset_id, manifest.dataset_revision, manifest.split_name) != expected_identity:
        raise ContractError("dataset file manifest identity differs from the training recipe")
    if not isinstance(verify_hashes, bool):
        raise TypeError("verify_hashes must be boolean")
    if (
        not isinstance(maximum_workers, int)
        or isinstance(maximum_workers, bool)
        or maximum_workers <= 0
    ):
        raise TypeError("maximum_workers must be a positive integer")
    root = Path(split_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if verify_hashes:
        expected = iter(manifest.files)
        paths = tuple(record.path for record in manifest.files)
        for window in _hash_windows_beneath(
            root,
            paths,
            maximum_workers=maximum_workers,
        ):
            for relative, (size_bytes, digest) in window:
                record = next(expected)
                if (
                    relative != record.path
                    or size_bytes != record.size_bytes
                    or digest != record.sha256
                ):
                    raise ContractError(
                        f"dataset source file differs from frozen manifest: {relative}"
                    )
    return {
        "dataset_file_count": len(manifest.files),
        "dataset_total_size_bytes": manifest.total_size_bytes,
        "dataset_tree_sha256": manifest.tree_sha256,
        "dataset_files_verified": verify_hashes,
    }


def validate_dataset_runtime_binding(
    manifest: DatasetFileManifest,
    split_root: str | Path,
    *,
    dataset_id: str,
    dataset_revision: str,
    split_name: str,
) -> dict[str, object]:
    """Bind a runtime to one manifest without rescanning causally unused files.

    Consumers of this contract must route every source read through
    :func:`read_verified_dataset_file`. That function hashes and retains bytes
    from the same descriptor before decoding, so every byte that can influence
    an output is checked against the immutable manifest without an extra
    full-tree pass at every process launch.
    """

    identity = validate_dataset_files(
        manifest,
        split_root,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        split_name=split_name,
        verify_hashes=False,
    )
    probe_records = (manifest.files[0], manifest.files[-1])
    if probe_records[0].path == probe_records[-1].path:
        probe_records = probe_records[:1]
    probe_digest = hashlib.sha256(b"picf-next.dataset-runtime-probes.v1\0")
    for record in probe_records:
        read_verified_dataset_file(
            manifest,
            split_root,
            record.path,
            maximum_bytes=max(record.size_bytes, 1),
        )
        encoded_path = record.path.encode("utf-8")
        probe_digest.update(len(encoded_path).to_bytes(8, "big"))
        probe_digest.update(encoded_path)
        probe_digest.update(bytes.fromhex(record.sha256))
    report = {
        "dataset_file_count": identity["dataset_file_count"],
        "dataset_total_size_bytes": identity["dataset_total_size_bytes"],
        "dataset_tree_sha256": identity["dataset_tree_sha256"],
        "dataset_manifest_self_consistent": True,
        "dataset_full_tree_rescanned": False,
        "dataset_runtime_verified_read_required": True,
        "dataset_runtime_probe_file_count": len(probe_records),
        "dataset_runtime_probe_sha256": probe_digest.hexdigest(),
        "dataset_verification_mode": DATASET_RUNTIME_VERIFICATION_MODE,
    }
    validate_dataset_runtime_binding_report(report)
    return report


def validate_dataset_runtime_binding_report(payload: object) -> dict[str, object]:
    """Validate exact evidence emitted by :func:`validate_dataset_runtime_binding`."""

    if not isinstance(payload, Mapping) or set(payload) != DATASET_RUNTIME_BINDING_FIELDS:
        raise ContractError("dataset runtime binding fields differ from schema")
    file_count = payload["dataset_file_count"]
    total_size_bytes = payload["dataset_total_size_bytes"]
    probe_file_count = payload["dataset_runtime_probe_file_count"]
    if not isinstance(file_count, int) or isinstance(file_count, bool) or file_count <= 0:
        raise ContractError("dataset runtime binding file count must be positive")
    if (
        not isinstance(total_size_bytes, int)
        or isinstance(total_size_bytes, bool)
        or total_size_bytes <= 0
    ):
        raise ContractError("dataset runtime binding byte count must be positive")
    if (
        not isinstance(probe_file_count, int)
        or isinstance(probe_file_count, bool)
        or probe_file_count != min(2, file_count)
    ):
        raise ContractError("dataset runtime binding probe count is inconsistent")
    _require_sha256(payload["dataset_tree_sha256"], "dataset runtime binding tree")
    _require_sha256(payload["dataset_runtime_probe_sha256"], "dataset runtime binding probes")
    if (
        payload["dataset_manifest_self_consistent"] is not True
        or payload["dataset_full_tree_rescanned"] is not False
        or payload["dataset_runtime_verified_read_required"] is not True
        or payload["dataset_verification_mode"] != DATASET_RUNTIME_VERIFICATION_MODE
    ):
        raise ContractError("dataset runtime verified-read mode changed")
    return dict(payload)
