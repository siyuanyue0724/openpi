"""Process-scoped single-writer leases for persistent training runs."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import stat
from pathlib import Path
from typing import Any


class ExclusiveRunLease:
    """Hold one kernel-released advisory lock for a persistent run root.

    The lock file is intentionally retained after release. Unlinking a lock file
    can let a third process lock a new inode while a second process still owns the
    old one. The kernel releases the actual lease whenever the descriptor or
    process exits, including abrupt worker termination.
    """

    _FILENAME = ".picf-single-writer.lock"

    def __init__(self, *, path: Path, descriptor: int, owner: dict[str, Any]) -> None:
        self.path = path
        self._descriptor: int | None = descriptor
        self.owner = owner

    @classmethod
    def acquire(cls, run_root: str | Path) -> ExclusiveRunLease:
        root_input = Path(run_root).expanduser()
        if root_input.is_symlink():
            raise ValueError("run lease root cannot be a symbolic link")
        root = root_input.resolve()
        if not root.is_dir():
            raise ValueError("run lease root must be one existing real directory")
        lock_path = root / cls._FILENAME
        if lock_path.is_symlink():
            raise ValueError("run lease file cannot be a symbolic link")

        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise ValueError("run lease file cannot be opened safely") from error
        try:
            metadata = os.fstat(descriptor)
            # Object-backed FUSE mounts can report zero links for a live named
            # file and reject hard links entirely. Bind the locked descriptor
            # back to the path below instead of assuming local-POSIX nlink=1.
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink not in {0, 1}:
                raise ValueError("run lease path must be one regular file")
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                owner = os.pread(descriptor, 4096, 0).decode("ascii", errors="replace").strip()
                detail = owner if owner else "owner metadata unavailable"
                raise RuntimeError(f"run root already has an active writer: {detail}") from error

            try:
                path_metadata = os.stat(lock_path, follow_symlinks=False)
            except OSError as error:
                raise ValueError("run lease file disappeared after opening") from error
            if (
                not stat.S_ISREG(path_metadata.st_mode)
                or path_metadata.st_nlink not in {0, 1}
                or (path_metadata.st_dev, path_metadata.st_ino)
                != (metadata.st_dev, metadata.st_ino)
            ):
                raise ValueError("run lease path changed after opening")

            owner_payload: dict[str, Any] = {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "run_root": str(root),
                "schema": "picf-next.exclusive-run-lease.v1",
            }
            encoded = (
                json.dumps(
                    owner_payload,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("ascii")
            os.ftruncate(descriptor, 0)
            offset = 0
            while offset < len(encoded):
                written = os.pwrite(descriptor, encoded[offset:], offset)
                if written <= 0:  # pragma: no cover - regular-file kernel failure
                    raise OSError("run lease owner metadata write made no progress")
                offset += written
            os.fsync(descriptor)
            return cls(path=lock_path, descriptor=descriptor, owner=owner_payload)
        except BaseException:
            os.close(descriptor)
            raise

    @property
    def active(self) -> bool:
        return self._descriptor is not None

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> ExclusiveRunLease:
        if not self.active:
            raise RuntimeError("closed run lease cannot be re-entered")
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


def acquire_distributed_run_lease(
    run_root: str | Path,
    *,
    rank: int,
    distributed: Any,
) -> ExclusiveRunLease | None:
    """Acquire one rank-zero lease and fail every distributed rank together."""

    lease: ExclusiveRunLease | None = None
    result: list[str | None] = [None]
    if rank == 0:
        try:
            lease = ExclusiveRunLease.acquire(run_root)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            result[0] = f"{type(error).__name__}: {error}"
    try:
        distributed.broadcast_object_list(result, src=0)
    except BaseException:
        if lease is not None:
            lease.close()
        raise
    if result[0] is not None:
        if lease is not None:
            lease.close()
        raise RuntimeError(f"distributed run lease acquisition failed: {result[0]}")
    return lease
