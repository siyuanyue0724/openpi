"""Frozen host-capacity checks for LingBot-native deployment."""

from __future__ import annotations

import shutil
from pathlib import Path

MINIMUM_LINGBOT_HOST_MEMORY_BYTES = 128 * 2**30
MINIMUM_LINGBOT_FREE_STORAGE_BYTES = 250 * 2**30
MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES = 64 * 2**30
MINIMUM_LINGBOT_EVIDENCE_WRITE_FREE_BYTES = 64 * 2**20
PERSISTENT_CLOUD_ROOT = Path("/mnt")


def require_persistent_run_root(path: Path) -> Path:
    """Require run artifacts to remain below the cloud's persistent mount."""

    selected = path.expanduser()
    if selected.is_symlink():
        raise ValueError("native run root cannot be a symbolic link")
    persistent = PERSISTENT_CLOUD_ROOT.resolve(strict=True)
    resolved = selected.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("native run root must be one existing real directory")
    if persistent not in resolved.parents:
        raise RuntimeError("native run root must be a strict descendant of persistent /mnt")
    return resolved


def existing_filesystem_path(path: Path) -> Path:
    """Resolve the nearest existing directory that owns ``path``'s filesystem."""

    probe = path.expanduser()
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise FileNotFoundError("capacity path has no existing parent")
        probe = parent
    resolved = probe.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("capacity path parent is not a directory")
    return resolved


def checkpoint_write_free_bytes(path: Path) -> int:
    """Return live free bytes on the filesystem that will receive a checkpoint."""

    return shutil.disk_usage(existing_filesystem_path(path)).free


def require_checkpoint_write_capacity(path: Path) -> int:
    """Require one measured 48.16-GiB checkpoint plus a bounded write margin."""

    free_bytes = checkpoint_write_free_bytes(path)
    if free_bytes < MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES:
        raise RuntimeError(
            "native checkpoint filesystem has "
            f"{free_bytes / 2**30:.2f} GiB free; "
            f"{MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES / 2**30:.0f} GiB is required"
        )
    return free_bytes


def require_evidence_write_capacity(path: Path) -> int:
    """Require bounded room for an immutable JSON probe and its launcher log."""

    free_bytes = checkpoint_write_free_bytes(path)
    if free_bytes < MINIMUM_LINGBOT_EVIDENCE_WRITE_FREE_BYTES:
        raise RuntimeError(
            "native evidence filesystem has "
            f"{free_bytes / 2**20:.2f} MiB free; "
            f"{MINIMUM_LINGBOT_EVIDENCE_WRITE_FREE_BYTES / 2**20:.0f} MiB is required"
        )
    return free_bytes
