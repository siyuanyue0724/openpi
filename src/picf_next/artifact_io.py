"""Fail-closed artifact publication on local filesystems and FUSE mounts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def directory_tree_sha256(
    root: str | Path,
    *,
    schema: str = "picf-next.artifact-directory-tree.v1",
) -> str:
    """Content-address one direct, symlink-free directory tree.

    Relative paths, file sizes, and file bytes are all part of the identity.
    Callers choose a domain-specific schema so unrelated artifact classes
    cannot accidentally share an identity namespace.
    """

    if not isinstance(schema, str) or not schema:
        raise ValueError("directory-tree digest schema must be non-empty")
    source = Path(root).expanduser()
    if source.is_symlink():
        raise ValueError("artifact must be one direct directory")
    directory = source.resolve()
    if not directory.is_dir():
        raise ValueError("artifact must be one direct directory")
    files: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact tree contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("artifact tree contains a non-regular entry")
        with path.open("rb") as stream:
            content_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": content_sha256,
            }
        )
    if not files:
        raise ValueError("artifact tree is empty")
    payload = json.dumps(
        {"schema": schema, "files": files},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publication_lock(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.publish-lock")


def publish_prepared_file_durable_exclusive(
    temporary: str | Path,
    destination: str | Path,
) -> Path:
    """Atomically consume one staged file without replacing an existing artifact.

    Some persistent cloud FUSE mounts reject both hard links and
    ``renameat2(RENAME_NOREPLACE)``. An exclusive lock directory serializes
    cooperating publishers, while same-directory ``os.replace`` retains atomic
    visibility. A stale lock or staging file is intentionally a hard failure.
    """

    source = _absolute_path(temporary)
    target = _absolute_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.parent != target.parent:
        raise ValueError("durable publication requires a same-directory staging file")
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    lock = _publication_lock(target)
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FileExistsError(lock) from error

    published = False
    try:
        _fsync_directory(target.parent)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        with source.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(source, target)
        published = True
        _fsync_directory(target.parent)
    except BaseException:
        if published:
            target.unlink(missing_ok=True)
            _fsync_directory(target.parent)
        source.unlink(missing_ok=True)
        raise
    finally:
        if lock.is_dir() and not lock.is_symlink():
            lock.rmdir()
            _fsync_directory(target.parent)
    return target


def publish_prepared_directory_durable_exclusive(
    temporary: str | Path,
    destination: str | Path,
) -> Path:
    """Atomically consume one prepared directory without replacing an artifact."""

    source = _absolute_path(temporary)
    target = _absolute_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.parent != target.parent:
        raise ValueError("durable publication requires a same-directory staging directory")
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    lock = _publication_lock(target)
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FileExistsError(lock) from error

    published = False
    try:
        _fsync_directory(source)
        _fsync_directory(target.parent)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.replace(source, target)
        published = True
        _fsync_directory(target.parent)
    except BaseException:
        cleanup = target if published else source
        if cleanup.is_dir() and not cleanup.is_symlink():
            shutil.rmtree(cleanup)
            _fsync_directory(target.parent)
        raise
    finally:
        if lock.is_dir() and not lock.is_symlink():
            lock.rmdir()
            _fsync_directory(target.parent)
    return target


def write_bytes_durable_exclusive(path: str | Path, payload: bytes) -> Path:
    """Durably publish immutable bytes through the shared FUSE-safe protocol."""

    if not isinstance(payload, bytes):
        raise TypeError("durable binary payload must be bytes")
    destination = _absolute_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return publish_prepared_file_durable_exclusive(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_text_durable_exclusive(
    path: str | Path,
    payload: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Encode and durably publish one immutable text artifact."""

    if not isinstance(payload, str):
        raise TypeError("durable text payload must be a string")
    return write_bytes_durable_exclusive(path, payload.encode(encoding))
