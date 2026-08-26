from __future__ import annotations

import argparse
import signal
import subprocess
from pathlib import Path
from typing import BinaryIO, cast

import pytest

import tools.run_calvin_physical_supervision_parallel as parallel_tool


def _args(tmp_path: Path) -> argparse.Namespace:
    builder = tmp_path / "tools" / "build_calvin_physical_supervision.py"
    builder.parent.mkdir()
    builder.touch()
    return argparse.Namespace(
        python=tmp_path / "python",
        builder=builder,
        split_root=tmp_path / "training",
        calvin_env_root=tmp_path / "calvin_env",
        output_dir=tmp_path / "physical",
        dataset_id="calvin-test",
        dataset_revision="revision-test",
        dataset_manifest=tmp_path / "manifest.json",
        partition_count=8,
        shard_size=256,
        progress_every=100,
        coverage="all_source_frames",
        poll_seconds=0.5,
        termination_grace_seconds=10.0,
    )


def test_partition_and_finalize_commands_preserve_builder_contract(tmp_path: Path) -> None:
    args = _args(tmp_path)

    partition = parallel_tool._partition_command(args, 3)
    finalizer = parallel_tool._finalize_command(args)

    assert partition[-6:] == (
        "--calvin-env-root",
        str(args.calvin_env_root.resolve()),
        "--partition-index",
        "3",
        "--resume-completed-partition",
        "--defer-finalize",
    )
    assert finalizer[-1] == "--finalize-only"
    assert "--calvin-env-root" not in finalizer
    assert partition[partition.index("--partition-count") + 1] == "8"
    assert partition[partition.index("--coverage") + 1] == "all_source_frames"


class _FakeProcess:
    def __init__(self, pid: int, returncode: int | None) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        del timeout
        if self.returncode is None:
            raise subprocess.TimeoutExpired(("fake",), 1.0)
        return self.returncode


def _worker(name: str, process: _FakeProcess) -> parallel_tool._Worker:
    return parallel_tool._Worker(
        name=name,
        command=("python", "builder.py"),
        process=cast(subprocess.Popen[bytes], process),
        log_handle=cast(BinaryIO, None),
    )


def test_terminate_process_groups_signals_extant_group_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = _FakeProcess(101, 7)
    finished = _FakeProcess(202, 0)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        parallel_tool,
        "_process_group_exists",
        lambda process: process.pid == 101,
    )
    monkeypatch.setattr(
        parallel_tool,
        "_signal_process_group",
        lambda process, requested_signal: signals.append((process.pid, requested_signal)),
    )
    monkeypatch.setattr(parallel_tool.time, "monotonic", lambda: 1.0)

    parallel_tool._terminate_process_groups(
        (_worker("running", running), _worker("finished", finished)),
        grace_seconds=0.0,
    )

    assert signals == [(101, signal.SIGTERM), (101, signal.SIGKILL)]


def test_wait_for_workers_fails_on_first_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = _worker("partition-0", _FakeProcess(101, 7))
    pending = _worker("partition-1", _FakeProcess(202, None))
    monkeypatch.setattr(
        parallel_tool.time,
        "sleep",
        lambda seconds: pytest.fail(f"unexpected sleep: {seconds}"),
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        parallel_tool._wait_for_workers(
            (failed, pending),
            poll_seconds=0.5,
            stop_requested=lambda: None,
        )

    assert error.value.returncode == 7


def test_argument_validation_rejects_completed_output(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.output_dir.mkdir()
    (args.output_dir / "manifest.json").write_text("{}")

    with pytest.raises(FileExistsError, match="immutable"):
        parallel_tool._validate_args(args)
