"""Deterministic identity checks for the pinned upstream WLA source tree.

This module deliberately does not import WLA.  It binds an immutable upstream
commit to the complete regular-file tree shipped by that commit, independent of
checkout metadata, mtimes, ownership, permissions, or archive compression.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

WLA_UPSTREAM_COMMIT = "155ac94eaca8b3d1ae0789ae298fc55e37936081"
WLA_SOURCE_TREE_SCHEMA = "picf-next.wla-source-tree/v1"
WLA_SOURCE_TREE_SELECTION = "all-regular-files-excluding-root-dot-git/v1"
WLA_TREE_DIGEST_ALGORITHM = "sha256-canonical-json-file-inventory/v1"
WLA_RECEIPT_SCHEMA = "picf-next.wla-source-tree-receipt/v1"

WLA_PINNED_FILE_COUNT = 72
WLA_PINNED_TOTAL_BYTES = 617_184
WLA_PINNED_TREE_SHA256 = "fa1c9a8857a2280b14eeb2a3864d55825ca7e2411548643ead89ebf8068abc3f"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_CHUNK_BYTES = 1024 * 1024


class WLASourceIdentityError(RuntimeError):
    """Raised when a WLA source carrier cannot prove the pinned identity."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise WLASourceIdentityError(
            f"invalid {label} keys: missing={missing!r}, extra={extra!r}"
        )


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise WLASourceIdentityError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise WLASourceIdentityError("upstream_commit must be a full lowercase Git commit id")
    return value


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise WLASourceIdentityError("receipt file path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise WLASourceIdentityError(f"unsafe receipt file path: {value!r}")
    if path.as_posix() != value:
        raise WLASourceIdentityError(f"non-canonical receipt file path: {value!r}")
    if path.parts[0] == ".git":
        raise WLASourceIdentityError("root .git metadata cannot appear in the source inventory")
    return value


@dataclass(frozen=True, slots=True)
class WLASourceFileIdentity:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        if not _is_plain_int(self.bytes) or self.bytes < 0:
            raise WLASourceIdentityError("receipt file bytes must be a non-negative integer")
        _require_sha256(self.sha256, "receipt file sha256")

    def to_mapping(self) -> dict[str, object]:
        return {"bytes": self.bytes, "path": self.path, "sha256": self.sha256}

    @classmethod
    def from_mapping(cls, value: object) -> WLASourceFileIdentity:
        if not isinstance(value, Mapping):
            raise WLASourceIdentityError("receipt file entry must be a mapping")
        _require_exact_keys(value, frozenset({"bytes", "path", "sha256"}), "file entry")
        return cls(
            path=_validate_relative_path(value["path"]),
            bytes=value["bytes"],  # type: ignore[arg-type]
            sha256=_require_sha256(value["sha256"], "receipt file sha256"),
        )


def _tree_payload(files: tuple[WLASourceFileIdentity, ...]) -> dict[str, object]:
    return {
        "files": [item.to_mapping() for item in files],
        "schema": WLA_SOURCE_TREE_SCHEMA,
        "selection": WLA_SOURCE_TREE_SELECTION,
    }


def _tree_sha256(files: tuple[WLASourceFileIdentity, ...]) -> str:
    return hashlib.sha256(_canonical_json_bytes(_tree_payload(files))).hexdigest()


@dataclass(frozen=True, slots=True)
class WLASourceTreeReceipt:
    upstream_commit: str
    file_count: int
    total_bytes: int
    tree_sha256: str
    files: tuple[WLASourceFileIdentity, ...]
    receipt_sha256: str
    schema: str = WLA_RECEIPT_SCHEMA
    selection: str = WLA_SOURCE_TREE_SELECTION
    tree_digest_algorithm: str = WLA_TREE_DIGEST_ALGORITHM

    def __post_init__(self) -> None:
        _require_commit(self.upstream_commit)
        if self.schema != WLA_RECEIPT_SCHEMA:
            raise WLASourceIdentityError(f"unsupported receipt schema: {self.schema!r}")
        if self.selection != WLA_SOURCE_TREE_SELECTION:
            raise WLASourceIdentityError(f"unsupported source selection: {self.selection!r}")
        if self.tree_digest_algorithm != WLA_TREE_DIGEST_ALGORITHM:
            raise WLASourceIdentityError(
                f"unsupported tree digest algorithm: {self.tree_digest_algorithm!r}"
            )
        if not _is_plain_int(self.file_count) or self.file_count < 0:
            raise WLASourceIdentityError("file_count must be a non-negative integer")
        if not _is_plain_int(self.total_bytes) or self.total_bytes < 0:
            raise WLASourceIdentityError("total_bytes must be a non-negative integer")
        _require_sha256(self.tree_sha256, "tree_sha256")
        _require_sha256(self.receipt_sha256, "receipt_sha256")

        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise WLASourceIdentityError("receipt file entries must have unique sorted paths")
        if self.file_count != len(self.files):
            raise WLASourceIdentityError("receipt file_count does not match its inventory")
        if self.total_bytes != sum(item.bytes for item in self.files):
            raise WLASourceIdentityError("receipt total_bytes does not match its inventory")
        if self.tree_sha256 != _tree_sha256(self.files):
            raise WLASourceIdentityError("receipt tree_sha256 does not match its inventory")
        expected_receipt_sha256 = hashlib.sha256(
            _canonical_json_bytes(self.unsigned_mapping())
        ).hexdigest()
        if self.receipt_sha256 != expected_receipt_sha256:
            raise WLASourceIdentityError("receipt_sha256 does not match the receipt content")

    def unsigned_mapping(self) -> dict[str, object]:
        return {
            "file_count": self.file_count,
            "files": [item.to_mapping() for item in self.files],
            "schema": self.schema,
            "selection": self.selection,
            "total_bytes": self.total_bytes,
            "tree_digest_algorithm": self.tree_digest_algorithm,
            "tree_sha256": self.tree_sha256,
            "upstream_commit": self.upstream_commit,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self.unsigned_mapping(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_mapping(cls, value: object) -> WLASourceTreeReceipt:
        if not isinstance(value, Mapping):
            raise WLASourceIdentityError("WLA source receipt must be a mapping")
        expected = frozenset(
            {
                "file_count",
                "files",
                "receipt_sha256",
                "schema",
                "selection",
                "total_bytes",
                "tree_digest_algorithm",
                "tree_sha256",
                "upstream_commit",
            }
        )
        _require_exact_keys(value, expected, "receipt")
        raw_files = value["files"]
        if not isinstance(raw_files, list):
            raise WLASourceIdentityError("receipt files must be a JSON list")
        return cls(
            upstream_commit=_require_commit(value["upstream_commit"]),
            file_count=value["file_count"],  # type: ignore[arg-type]
            total_bytes=value["total_bytes"],  # type: ignore[arg-type]
            tree_sha256=_require_sha256(value["tree_sha256"], "tree_sha256"),
            files=tuple(WLASourceFileIdentity.from_mapping(item) for item in raw_files),
            receipt_sha256=_require_sha256(value["receipt_sha256"], "receipt_sha256"),
            schema=value["schema"],  # type: ignore[arg-type]
            selection=value["selection"],  # type: ignore[arg-type]
            tree_digest_algorithm=value["tree_digest_algorithm"],  # type: ignore[arg-type]
        )


def _receipt_from_files(
    files: tuple[WLASourceFileIdentity, ...], upstream_commit: str
) -> WLASourceTreeReceipt:
    commit = _require_commit(upstream_commit)
    ordered = tuple(sorted(files, key=lambda item: item.path))
    if len({item.path for item in ordered}) != len(ordered):
        raise WLASourceIdentityError("source inventory contains duplicate paths")
    unsigned = {
        "file_count": len(ordered),
        "files": [item.to_mapping() for item in ordered],
        "schema": WLA_RECEIPT_SCHEMA,
        "selection": WLA_SOURCE_TREE_SELECTION,
        "total_bytes": sum(item.bytes for item in ordered),
        "tree_digest_algorithm": WLA_TREE_DIGEST_ALGORITHM,
        "tree_sha256": _tree_sha256(ordered),
        "upstream_commit": commit,
    }
    receipt_sha256 = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    return WLASourceTreeReceipt(
        upstream_commit=commit,
        file_count=len(ordered),
        total_bytes=sum(item.bytes for item in ordered),
        tree_sha256=str(unsigned["tree_sha256"]),
        files=ordered,
        receipt_sha256=receipt_sha256,
    )


def _hash_stream(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def _hash_regular_file(path: Path, relative: str) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WLASourceIdentityError(f"cannot open source file {relative!r}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise WLASourceIdentityError(f"source entry is not a regular file: {relative!r}")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            size, digest = _hash_stream(stream)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or size != before.st_size:
            raise WLASourceIdentityError(f"source file changed while hashing: {relative!r}")
        return size, digest
    finally:
        os.close(descriptor)


def _directory_files(root: Path) -> tuple[WLASourceFileIdentity, ...]:
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise WLASourceIdentityError(f"cannot inspect WLA source root {root}: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise WLASourceIdentityError("WLA source root must be a real directory, not a symlink")

    files: list[WLASourceFileIdentity] = []

    def visit(directory: Path, relative_parent: PurePosixPath) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise WLASourceIdentityError(
                f"cannot enumerate WLA source directory {directory}"
            ) from exc
        for entry in entries:
            if not relative_parent.parts and entry.name == ".git":
                continue
            relative_path = relative_parent / entry.name
            relative = relative_path.as_posix()
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise WLASourceIdentityError(f"cannot inspect source entry {relative!r}") from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise WLASourceIdentityError(
                    f"symlink is forbidden in WLA source tree: {relative!r}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                visit(Path(entry.path), relative_path)
            elif stat.S_ISREG(entry_stat.st_mode):
                size, digest = _hash_regular_file(Path(entry.path), relative)
                files.append(WLASourceFileIdentity(relative, size, digest))
            else:
                raise WLASourceIdentityError(
                    f"non-regular source entry is forbidden: {relative!r}"
                )

    visit(root, PurePosixPath())
    return tuple(sorted(files, key=lambda item: item.path))


def _git_head(root: Path, expected_commit: str, require_git_head: bool | None) -> None:
    marker = root / ".git"
    marker_exists = os.path.lexists(marker)
    if marker_exists and marker.is_symlink():
        raise WLASourceIdentityError("root .git metadata cannot be a symlink")
    should_check = marker_exists if require_git_head is None else require_git_head
    if not should_check:
        return
    if not marker_exists:
        raise WLASourceIdentityError("Git HEAD verification was required but root .git is absent")

    def run_git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", os.fspath(root), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise WLASourceIdentityError(f"cannot verify WLA Git identity: {exc}") from exc
        return completed.stdout.strip()

    top_level = Path(run_git("rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != root:
        raise WLASourceIdentityError(
            f"WLA source root is not the Git top-level directory: {top_level} != {root}"
        )
    actual_commit = run_git("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise WLASourceIdentityError(
            f"WLA Git HEAD mismatch: expected {expected_commit}, got {actual_commit}"
        )


def build_wla_source_tree_receipt(
    source_root: str | os.PathLike[str],
    *,
    upstream_commit: str = WLA_UPSTREAM_COMMIT,
    require_git_head: bool | None = None,
) -> WLASourceTreeReceipt:
    """Build a deterministic receipt from a checkout or extracted source archive.

    When ``require_git_head`` is ``None``, a checkout containing root ``.git``
    must have exactly ``upstream_commit`` at HEAD.  An extracted immutable
    archive has no Git metadata and is authenticated by its complete tree hash.
    """

    commit = _require_commit(upstream_commit)
    root = Path(source_root).expanduser().resolve(strict=True)
    _git_head(root, commit, require_git_head)
    return _receipt_from_files(_directory_files(root), commit)


def _safe_archive_parts(name: str) -> tuple[str, ...]:
    if not name or "\\" in name:
        raise WLASourceIdentityError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise WLASourceIdentityError(f"unsafe archive member path: {name!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        raise WLASourceIdentityError(f"empty archive member path: {name!r}")
    return parts


def _normalize_archive_paths(raw_paths: list[tuple[str, ...]]) -> list[str | None]:
    first_parts = {parts[0] for parts in raw_paths}
    strip_root = len(first_parts) == 1 and all(len(parts) > 1 for parts in raw_paths)
    normalized: list[str | None] = []
    seen: set[str] = set()
    for parts in raw_paths:
        selected = parts[1:] if strip_root else parts
        if selected[0] == ".git":
            normalized.append(None)
            continue
        relative = _validate_relative_path(PurePosixPath(*selected).as_posix())
        if relative in seen:
            raise WLASourceIdentityError(f"duplicate archive member path: {relative!r}")
        seen.add(relative)
        normalized.append(relative)
    return normalized


def _tar_files(path: Path) -> tuple[WLASourceFileIdentity, ...]:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = []
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    raise WLASourceIdentityError(
                        f"non-regular tar member is forbidden: {member.name!r}"
                    )
                members.append((member, _safe_archive_parts(member.name)))
            normalized = _normalize_archive_paths([parts for _, parts in members])
            files: list[WLASourceFileIdentity] = []
            for (member, _), relative in zip(members, normalized, strict=True):
                if relative is None:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise WLASourceIdentityError(f"cannot read tar member: {member.name!r}")
                with stream:
                    size, digest = _hash_stream(stream)
                if size != member.size:
                    raise WLASourceIdentityError(
                        f"tar member size changed while reading: {member.name!r}"
                    )
                files.append(WLASourceFileIdentity(relative, size, digest))
    except WLASourceIdentityError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise WLASourceIdentityError(f"cannot read WLA tar archive {path}: {exc}") from exc
    return tuple(sorted(files, key=lambda item: item.path))


def _zip_files(path: Path) -> tuple[WLASourceFileIdentity, ...]:
    try:
        archive = zipfile.ZipFile(path, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise WLASourceIdentityError(f"cannot open WLA zip archive {path}: {exc}") from exc
    with archive:
        members = []
        for member in archive.infolist():
            if member.is_dir():
                continue
            mode = (member.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if stat.S_ISLNK(mode) or file_type not in {0, stat.S_IFREG}:
                raise WLASourceIdentityError(
                    f"non-regular zip member is forbidden: {member.filename!r}"
                )
            members.append((member, _safe_archive_parts(member.filename)))
        normalized = _normalize_archive_paths([parts for _, parts in members])
        files: list[WLASourceFileIdentity] = []
        for (member, _), relative in zip(members, normalized, strict=True):
            if relative is None:
                continue
            try:
                with archive.open(member, mode="r") as stream:
                    size, digest = _hash_stream(stream)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise WLASourceIdentityError(
                    f"cannot read zip member {member.filename!r}: {exc}"
                ) from exc
            if size != member.file_size:
                raise WLASourceIdentityError(
                    f"zip member size changed while reading: {member.filename!r}"
                )
            files.append(WLASourceFileIdentity(relative, size, digest))
    return tuple(sorted(files, key=lambda item: item.path))


def build_wla_source_archive_receipt(
    archive_path: str | os.PathLike[str],
    *,
    upstream_commit: str = WLA_UPSTREAM_COMMIT,
) -> WLASourceTreeReceipt:
    """Build the same source receipt directly from a tar or zip archive."""

    commit = _require_commit(upstream_commit)
    path = Path(archive_path).expanduser().resolve(strict=True)
    if tarfile.is_tarfile(path):
        files = _tar_files(path)
    elif zipfile.is_zipfile(path):
        files = _zip_files(path)
    else:
        raise WLASourceIdentityError(f"unsupported WLA source archive: {path}")
    return _receipt_from_files(files, commit)


def _assert_pinned(receipt: WLASourceTreeReceipt) -> WLASourceTreeReceipt:
    actual = (
        receipt.upstream_commit,
        receipt.file_count,
        receipt.total_bytes,
        receipt.tree_sha256,
    )
    expected = (
        WLA_UPSTREAM_COMMIT,
        WLA_PINNED_FILE_COUNT,
        WLA_PINNED_TOTAL_BYTES,
        WLA_PINNED_TREE_SHA256,
    )
    if actual != expected:
        raise WLASourceIdentityError(
            "WLA source identity mismatch: "
            f"expected commit/files/bytes/tree={expected!r}, got {actual!r}"
        )
    return receipt


def verify_pinned_wla_source_tree(
    source_root: str | os.PathLike[str],
    *,
    require_git_head: bool | None = None,
) -> WLASourceTreeReceipt:
    """Verify an extracted tree and, when present, its real Git HEAD."""

    return _assert_pinned(
        build_wla_source_tree_receipt(
            source_root,
            upstream_commit=WLA_UPSTREAM_COMMIT,
            require_git_head=require_git_head,
        )
    )


def verify_pinned_wla_source_archive(
    archive_path: str | os.PathLike[str],
) -> WLASourceTreeReceipt:
    """Verify a pinned immutable tar/zip archive without extracting it."""

    return _assert_pinned(
        build_wla_source_archive_receipt(archive_path, upstream_commit=WLA_UPSTREAM_COMMIT)
    )


def verify_wla_source_tree_receipt(
    source_root: str | os.PathLike[str],
    receipt: WLASourceTreeReceipt,
    *,
    require_git_head: bool | None = None,
) -> WLASourceTreeReceipt:
    """Recompute a tree and require exact equality with a validated receipt."""

    actual = build_wla_source_tree_receipt(
        source_root,
        upstream_commit=receipt.upstream_commit,
        require_git_head=require_git_head,
    )
    if actual != receipt:
        raise WLASourceIdentityError("WLA source tree does not match the supplied receipt")
    return actual


def verify_wla_source_archive_receipt(
    archive_path: str | os.PathLike[str], receipt: WLASourceTreeReceipt
) -> WLASourceTreeReceipt:
    """Recompute an immutable archive and require exact receipt equality."""

    actual = build_wla_source_archive_receipt(
        archive_path, upstream_commit=receipt.upstream_commit
    )
    if actual != receipt:
        raise WLASourceIdentityError("WLA source archive does not match the supplied receipt")
    return actual


def load_wla_source_tree_receipt(path: str | os.PathLike[str]) -> WLASourceTreeReceipt:
    """Load and fully self-validate a receipt JSON document."""

    receipt_path = Path(path)
    try:
        value = json.loads(receipt_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WLASourceIdentityError(
            f"cannot load WLA source receipt {receipt_path}: {exc}"
        ) from exc
    return WLASourceTreeReceipt.from_mapping(value)


def wla_source_tree_receipt_bytes(receipt: WLASourceTreeReceipt) -> bytes:
    """Return the canonical, reproducible on-disk receipt representation."""

    return _canonical_json_bytes(receipt.to_mapping()) + b"\n"


def write_wla_source_tree_receipt(
    receipt: WLASourceTreeReceipt, destination: str | os.PathLike[str]
) -> Path:
    """Atomically create, but never replace, a read-only receipt file."""

    output = Path(destination).expanduser()
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    payload = wla_source_tree_receipt_bytes(receipt)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.link(temporary, output)
        directory_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_pinned_wla_source_tree_receipt(
    source_root: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    require_git_head: bool | None = None,
) -> WLASourceTreeReceipt:
    """Verify the pinned tree, then atomically emit its portable receipt."""

    root = Path(source_root).expanduser().resolve(strict=True)
    output_parent = Path(destination).expanduser().parent.resolve(strict=True)
    output = output_parent / Path(destination).name
    if output == root or root in output.parents:
        raise WLASourceIdentityError("receipt destination must be outside the hashed source tree")
    receipt = verify_pinned_wla_source_tree(root, require_git_head=require_git_head)
    write_wla_source_tree_receipt(receipt, output)
    return receipt


def write_pinned_wla_source_archive_receipt(
    archive_path: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> WLASourceTreeReceipt:
    """Verify an immutable pinned archive, then emit its portable receipt."""

    archive = Path(archive_path).expanduser().resolve(strict=True)
    output_parent = Path(destination).expanduser().parent.resolve(strict=True)
    output = output_parent / Path(destination).name
    if output == archive:
        raise WLASourceIdentityError("receipt destination cannot replace its source archive")
    receipt = verify_pinned_wla_source_archive(archive)
    write_wla_source_tree_receipt(receipt, output)
    return receipt
