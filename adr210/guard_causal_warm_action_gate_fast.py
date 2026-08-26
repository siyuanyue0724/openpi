#!/usr/bin/env python3
"""Fail-close ADR-210 at step 100 without racing the in-process evaluator.

The training loop publishes the warm snapshot immediately before checking the
external STOP marker.  This guard preloads the comparator, busy-waits only
while the expensive step-100 evaluation is active, and briefly stops all four
torchrun workers while the registered comparison is computed.  It changes no
model, data, objective, optimizer, sample, or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
COMPARE = REPO / "tools" / "compare_adr210_causal_warm_action_gate.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("train_pid", type=int)
    parser.add_argument("lingbot_step100", type=Path)
    return parser.parse_args()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _append_durable(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="ascii") as stream:
        stream.write(message.rstrip("\n") + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _set_stop(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    os.close(descriptor)
    _fsync_directory(path.parent)


def _clear_stop(path: Path) -> None:
    path.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _direct_children(parent_pid: int) -> tuple[int, ...]:
    children: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="ascii").split()
            if int(fields[3]) == parent_pid:
                children.append(int(entry.name))
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
    return tuple(sorted(children))


def _local_rank(pid: int) -> int | None:
    try:
        environment = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError):
        return None
    for item in environment:
        if item.startswith(b"LOCAL_RANK="):
            return int(item.partition(b"=")[2])
    return None


def _workers(train_pid: int) -> tuple[int, ...]:
    ranked = sorted(
        (rank, pid)
        for pid in _direct_children(train_pid)
        if (rank := _local_rank(pid)) is not None
    )
    if [rank for rank, _pid in ranked] != [0, 1, 2, 3]:
        raise RuntimeError(f"torchrun worker set differs: {ranked}")
    return tuple(pid for _rank, pid in ranked)


def _pause(workers: tuple[int, ...]) -> None:
    for pid in workers:
        os.kill(pid, signal.SIGSTOP)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        states = []
        for pid in workers:
            try:
                status = (Path("/proc") / str(pid) / "status").read_text(encoding="ascii")
            except FileNotFoundError:
                states.append("exited")
                continue
            state_line = next(line for line in status.splitlines() if line.startswith("State:"))
            states.append(state_line.split()[1])
        if all(state in {"T", "t"} for state in states):
            return
        time.sleep(0.001)
    raise RuntimeError("torchrun workers did not enter a stopped state")


def _resume(workers: tuple[int, ...]) -> None:
    for pid in workers:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    if not str(run_dir).startswith("/mnt/") or not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError("run directory must be a direct directory beneath /mnt")
    if not _pid_alive(args.train_pid):
        raise ValueError("training process is absent")
    if not args.lingbot_step100.is_file() or args.lingbot_step100.is_symlink():
        raise ValueError("matched LingBot step-100 snapshot is absent")
    if not REPO.is_dir() or not COMPARE.is_file():
        raise ValueError("immutable source or ADR-210 comparator is absent")

    run_manifest_path = run_dir / "run_manifest.json"
    while _pid_alive(args.train_pid) and not run_manifest_path.is_file():
        time.sleep(0.05)
    if not run_manifest_path.is_file() or run_manifest_path.is_symlink():
        raise RuntimeError("training exited before publishing its immutable run manifest")

    sys.path.insert(0, str(REPO))
    specification = importlib.util.spec_from_file_location("adr210_compare_gate", COMPARE)
    if specification is None or specification.loader is None:
        raise RuntimeError("ADR-210 comparator cannot be imported by absolute path")
    comparator = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = comparator
    specification.loader.exec_module(comparator)
    cold_schema = comparator.COLD_SCHEMA
    lbot_schema = comparator.LBOT_SCHEMA
    warm_schema = comparator.WARM_SCHEMA
    load_snapshot = comparator._load
    compare_gate = comparator.compare_gate

    warm_path = run_dir / "causal_warm_action_evaluations/step_00000100/distributed.json"
    cold_path = run_dir / "action_evaluations/step_00000100/distributed.json"
    report_path = run_dir / "audits/adr210-causal-warm-action-gate-step100.json"
    log_path = run_dir / "audits/adr210-causal-warm-action-gate-guard.log"
    journal_path = run_dir / "metrics/rank_journal/rank_0.jsonl"
    stop_path = run_dir / "STOP"
    _append_durable(log_path, f"fast_guard_started pid={os.getpid()} train_pid={args.train_pid}")

    armed = False
    workers: tuple[int, ...] = ()
    while _pid_alive(args.train_pid):
        if not armed and journal_path.is_file():
            if '"global_step":100,' in journal_path.read_text(encoding="ascii"):
                _set_stop(stop_path)
                workers = _workers(args.train_pid)
                armed = True
                _append_durable(log_path, f"fail_closed_stop_armed workers={list(workers)}")
        if armed and warm_path.is_file() and cold_path.is_file():
            try:
                _pause(workers)
            except BaseException:
                _resume(workers)
                raise
            decision = "COMPARISON_FAILED"
            try:
                warm = load_snapshot(warm_path, schema=warm_schema)
                cold = load_snapshot(cold_path, schema=cold_schema)
                lbot = load_snapshot(args.lingbot_step100, schema=lbot_schema)
                report = compare_gate(
                    warm=warm,
                    cold=cold,
                    lbot=lbot,
                    cold_path=cold_path,
                    bootstrap_replicates=10_000,
                    minimum_relative_reduction=0.02,
                )
                if report.get("artifact_sha256") != _canonical_sha256(
                    {key: value for key, value in report.items() if key != "artifact_sha256"}
                ):
                    raise RuntimeError("comparison semantic digest differs")
                _write_exclusive(report_path, report)
                decision = str(report["decision"])
                if decision == "AUTHORIZE_30K":
                    _clear_stop(stop_path)
                else:
                    _set_stop(stop_path)
                _append_durable(log_path, f"decision={decision}")
            except BaseException as error:
                _set_stop(stop_path)
                _append_durable(
                    log_path,
                    f"comparison_failed={type(error).__name__}: {error}",
                )
                raise
            finally:
                _resume(workers)
                _append_durable(log_path, f"workers_resumed decision={decision}")
            return
        time.sleep(0.05 if not armed else 0.001)

    _set_stop(stop_path)
    _append_durable(log_path, "training_exited_before_gate")
    raise RuntimeError("training exited before ADR-210 gate completed")


if __name__ == "__main__":
    main()
