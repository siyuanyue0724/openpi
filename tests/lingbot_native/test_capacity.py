from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import picf_next.lingbot_native.capacity as capacity
from picf_next.lingbot_native.capacity import (
    MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES,
    MINIMUM_LINGBOT_EVIDENCE_WRITE_FREE_BYTES,
    checkpoint_write_free_bytes,
    existing_filesystem_path,
    require_checkpoint_write_capacity,
    require_evidence_write_capacity,
)


def test_checkpoint_capacity_resolves_a_future_run_to_its_existing_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future_run = tmp_path / "runs" / "new-run"
    observed: list[Path] = []

    def fake_disk_usage(path: Path) -> SimpleNamespace:
        observed.append(path)
        return SimpleNamespace(free=MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES)

    monkeypatch.setattr(capacity.shutil, "disk_usage", fake_disk_usage)
    assert existing_filesystem_path(future_run) == tmp_path
    assert checkpoint_write_free_bytes(future_run) == MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES
    assert require_checkpoint_write_capacity(future_run) == (
        MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES
    )
    assert observed == [tmp_path, tmp_path]


def test_checkpoint_capacity_fails_below_the_measured_write_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capacity.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES - 1),
    )
    with pytest.raises(RuntimeError, match="64 GiB is required"):
        require_checkpoint_write_capacity(tmp_path)


def test_evidence_capacity_uses_its_bounded_artifact_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capacity.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_LINGBOT_EVIDENCE_WRITE_FREE_BYTES),
    )
    assert require_evidence_write_capacity(tmp_path) == (MINIMUM_LINGBOT_EVIDENCE_WRITE_FREE_BYTES)
    monkeypatch.setattr(
        capacity.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_LINGBOT_EVIDENCE_WRITE_FREE_BYTES - 1),
    )
    with pytest.raises(RuntimeError, match="64 MiB is required"):
        require_evidence_write_capacity(tmp_path)


def test_checkpoint_capacity_rejects_a_file_as_the_storage_parent(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact"
    file_path.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        existing_filesystem_path(file_path)
