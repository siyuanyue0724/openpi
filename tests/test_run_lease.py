from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import picf_next.training.run_lease as run_lease_module
from picf_next.training.run_lease import (
    ExclusiveRunLease,
    acquire_distributed_run_lease,
)


class _Broadcast:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def broadcast_object_list(self, value: list[str | None], *, src: int) -> None:
        assert src == 0
        assert len(value) == 1
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected broadcast failure")


def test_exclusive_run_lease_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()

    first = ExclusiveRunLease.acquire(run_root)
    try:
        assert first.active
        assert first.owner["run_root"] == str(run_root.resolve())
        with pytest.raises(RuntimeError, match="active writer"):
            ExclusiveRunLease.acquire(run_root)
    finally:
        first.close()

    with ExclusiveRunLease.acquire(run_root) as second:
        assert second.active
        payload = json.loads(second.path.read_text(encoding="ascii"))
        assert payload == second.owner
        assert os.stat(second.path).st_mode & 0o777 == 0o600
    assert not second.active
    assert second.path.is_file()


def test_exclusive_run_lease_rejects_unsafe_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    alias = tmp_path / "run-alias"
    alias.symlink_to(run_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        ExclusiveRunLease.acquire(alias)

    lock_target = tmp_path / "lock-target"
    lock_target.write_text("not a lease\n", encoding="ascii")
    (run_root / ".picf-single-writer.lock").symlink_to(lock_target)
    with pytest.raises(ValueError, match="symbolic link"):
        ExclusiveRunLease.acquire(run_root)

    (run_root / ".picf-single-writer.lock").unlink()
    os.link(lock_target, run_root / ".picf-single-writer.lock")
    with pytest.raises(ValueError, match="regular file"):
        ExclusiveRunLease.acquire(run_root)


def test_exclusive_run_lease_closes_when_context_fails(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    with (
        pytest.raises(RuntimeError, match="injected"),
        ExclusiveRunLease.acquire(run_root),
    ):
        raise RuntimeError("injected")

    recovered = ExclusiveRunLease.acquire(run_root)
    recovered.close()
    recovered.close()
    with pytest.raises(RuntimeError, match="cannot be re-entered"):
        recovered.__enter__()


def test_exclusive_run_lease_accepts_zero_link_count_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    native_fstat = os.fstat

    def zero_link_fstat(descriptor: int) -> os.stat_result:
        fields = list(native_fstat(descriptor))
        fields[3] = 0
        return os.stat_result(fields)

    monkeypatch.setattr(run_lease_module.os, "fstat", zero_link_fstat)
    with ExclusiveRunLease.acquire(run_root) as lease:
        assert lease.active
        assert lease.path.is_file()


def test_exclusive_run_lease_rejects_path_replacement_after_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock_path = run_root / ".picf-single-writer.lock"
    native_flock = run_lease_module.fcntl.flock

    def replacing_flock(descriptor: int, operation: int) -> None:
        native_flock(descriptor, operation)
        lock_path.unlink()
        lock_path.write_text("replacement\n", encoding="ascii")

    monkeypatch.setattr(run_lease_module.fcntl, "flock", replacing_flock)
    with pytest.raises(ValueError, match="changed after opening"):
        ExclusiveRunLease.acquire(run_root)


def test_distributed_run_lease_is_owned_only_by_rank_zero(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    distributed = _Broadcast()

    lease = acquire_distributed_run_lease(
        run_root,
        rank=0,
        distributed=distributed,
    )
    assert lease is not None and lease.active
    lease.close()
    assert distributed.calls == 1

    assert (
        acquire_distributed_run_lease(
            run_root,
            rank=1,
            distributed=distributed,
        )
        is None
    )


def test_distributed_run_lease_closes_if_broadcast_fails(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    with pytest.raises(RuntimeError, match="broadcast failure"):
        acquire_distributed_run_lease(
            run_root,
            rank=0,
            distributed=_Broadcast(fail=True),
        )

    recovered = ExclusiveRunLease.acquire(run_root)
    recovered.close()
