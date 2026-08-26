#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build partitioned CALVIN physical supervision with fail-fast process groups."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="physical supervision orchestrator",
)

from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
)


@dataclass(slots=True)
class _Worker:
    name: str
    command: tuple[str, ...]
    process: subprocess.Popen[bytes]
    log_handle: BinaryIO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--builder",
        type=Path,
        default=Path(__file__).resolve().with_name("build_calvin_physical_supervision.py"),
    )
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--calvin-env-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--partition-count", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--coverage",
        choices=(
            CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
            CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
        ),
        default=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--termination-grace-seconds", type=float, default=10.0)
    return parser.parse_args()


def _common_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python.resolve()),
        str(args.builder.resolve()),
        "--split-root",
        str(args.split_root.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--dataset-id",
        args.dataset_id,
        "--dataset-revision",
        args.dataset_revision,
        "--dataset-manifest",
        str(args.dataset_manifest.resolve()),
        "--partition-count",
        str(args.partition_count),
        "--shard-size",
        str(args.shard_size),
        "--progress-every",
        str(args.progress_every),
        "--coverage",
        args.coverage,
    ]


def _partition_command(args: argparse.Namespace, partition_index: int) -> tuple[str, ...]:
    return tuple(
        [
            *_common_command(args),
            "--calvin-env-root",
            str(args.calvin_env_root.resolve()),
            "--partition-index",
            str(partition_index),
            "--resume-completed-partition",
            "--defer-finalize",
        ]
    )


def _finalize_command(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple([*_common_command(args), "--finalize-only"])


def _start_worker(
    *,
    name: str,
    command: tuple[str, ...],
    log_path: Path,
    working_directory: Path,
) -> _Worker:
    if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
        raise RuntimeError(f"worker log is not a regular file: {log_path}")
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(
        (
            json.dumps(
                {
                    "command": list(command),
                    "event": "worker_attempt",
                    "name": name,
                    "started_at_unix": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    print(
        json.dumps(
            {
                "event": "worker_started",
                "name": name,
                "pid": process.pid,
                "process_group": process.pid,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return _Worker(name=name, command=command, process=process, log_handle=log_handle)


def _wait_for_workers(
    workers: Sequence[_Worker],
    *,
    poll_seconds: float,
    stop_requested: Callable[[], int | None],
) -> None:
    pending = {worker.name: worker for worker in workers}
    while pending:
        requested_signal = stop_requested()
        if requested_signal is not None:
            raise InterruptedError(f"received signal {requested_signal}")
        for name, worker in tuple(pending.items()):
            returncode = worker.process.poll()
            if returncode is None:
                continue
            print(
                json.dumps(
                    {
                        "event": "worker_exited",
                        "name": name,
                        "pid": worker.process.pid,
                        "returncode": returncode,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del pending[name]
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, worker.command)
        if pending:
            time.sleep(poll_seconds)


def _signal_process_group(process: subprocess.Popen[bytes], requested_signal: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, requested_signal)


def _process_group_exists(process: subprocess.Popen[bytes]) -> bool:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_groups(workers: Sequence[_Worker], *, grace_seconds: float) -> None:
    running = [worker for worker in workers if _process_group_exists(worker.process)]
    for worker in running:
        _signal_process_group(worker.process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while running and time.monotonic() < deadline:
        running = [worker for worker in running if _process_group_exists(worker.process)]
        if running:
            time.sleep(min(0.1, grace_seconds))
    for worker in running:
        _signal_process_group(worker.process, signal.SIGKILL)
    for worker in workers:
        if worker.process.poll() is None:
            with suppress(subprocess.TimeoutExpired):
                worker.process.wait(timeout=1.0)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("partition_count", "shard_size", "progress_every"):
        value = getattr(args, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("poll_seconds", "termination_grace_seconds"):
        value = getattr(args, name)
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if not args.builder.resolve().is_file():
        raise FileNotFoundError(args.builder)
    output_dir = args.output_dir.resolve()
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise RuntimeError(f"output directory is invalid: {output_dir}")
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("completed CALVIN physical artifacts are immutable")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    working_directory = args.builder.resolve().parent.parent
    workers: list[_Worker] = []
    requested_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal requested_signal
        requested_signal = signum

    previous_handlers = {
        signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for partition_index in range(args.partition_count):
            workers.append(
                _start_worker(
                    name=f"partition-{partition_index}",
                    command=_partition_command(args, partition_index),
                    log_path=output_dir / f"partition_{partition_index}.stdout.log",
                    working_directory=working_directory,
                )
            )
        _wait_for_workers(
            workers,
            poll_seconds=float(args.poll_seconds),
            stop_requested=lambda: requested_signal,
        )
        finalizer = _start_worker(
            name="finalize",
            command=_finalize_command(args),
            log_path=output_dir / "finalize.stdout.log",
            working_directory=working_directory,
        )
        workers.append(finalizer)
        _wait_for_workers(
            (finalizer,),
            poll_seconds=float(args.poll_seconds),
            stop_requested=lambda: requested_signal,
        )
        if not (output_dir / "manifest.json").is_file():
            raise RuntimeError("CALVIN physical finalizer did not publish manifest.json")
        print(
            json.dumps(
                {
                    "event": "physical_supervision_complete",
                    "manifest": str(output_dir / "manifest.json"),
                    "partition_count": args.partition_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        _terminate_process_groups(
            workers,
            grace_seconds=float(args.termination_grace_seconds),
        )
        for worker in workers:
            worker.log_handle.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
