#!/usr/bin/env python3
"""Versioned post-acceptance orchestration and passive LTOP health supervision."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Final

INITIAL_GRACE_SECONDS: Final = 3_600.0
CHECKPOINT_BOUNDARY_GRACE_SECONDS: Final = 3_600.0
STALE_THRESHOLD_SECONDS: Final = 900.0
POLL_INTERVAL_SECONDS: Final = 30.0
STATUS_HEARTBEAT_SECONDS: Final = 300.0
TERMINATION_GRACE_SECONDS: Final = 120.0
RESTART_TIMEOUT_SECONDS: Final = 14_400.0
LONG_TOTAL_STEPS: Final = 30_000
LONG_CHECKPOINT_EVERY: Final = 2_000
LONG_PROGRESS_EVERY: Final = 8

ACCEPTANCE_SCHEMA: Final = "picf-next.ltop-g3-source-aligned-acceptance.v1"
ACTION_VALIDATION_SCHEMA: Final = "picf-next.ltop-g3-cold-action-evidence-validation.v1"
PROGRESS_SCHEMA: Final = "picf-next.ltop-core-pilot-progress.v1"
STATUS_SCHEMA: Final = "picf-next.ltop-process-supervisor-status.v1"
EXIT_SCHEMA: Final = "picf-next.ltop-process-supervisor-exit.v1"
CHAIN_STATUS_SCHEMA: Final = "picf-next.adr170-post-acceptance-status.v1"
CHAIN_EXIT_SCHEMA: Final = "picf-next.adr170-post-acceptance-exit.v1"
PROMOTION_SCHEMA: Final = "picf-next.adr170-post-acceptance-promotion.v1"

_MNT_ROOT = Path("/mnt").resolve()
_EXPECTED_EVIDENCE_KEYS: Final = (
    "training_report",
    "arm_validation",
    "cold_action_factual",
    "cold_action_mediator_required",
    "cold_retention",
)
_EVIDENCE_SNAPSHOT_FILENAMES: Final = {
    "training_report": "training-report.json",
    "arm_validation": "arm-validation.json",
    "cold_action_factual": "cold-action-factual.json",
    "cold_action_mediator_required": "cold-action-mediator-required.json",
    "cold_retention": "cold-retention.json",
}


class SupervisorInterrupted(RuntimeError):
    """Raised when the supervisor receives an interactive termination signal."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"supervisor received signal {signum}")
        self.signum = signum


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    boot_id: str


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    status: str
    reason: str
    returncode: int | None
    exit_receipt: Path
    last_completed_steps: int | None


@dataclass(frozen=True, slots=True)
class SupervisionSpec:
    kind: str
    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    run_root: Path
    log_output: Path
    status_output: Path
    exit_output: Path
    progress_path: Path | None = None
    timeout_seconds: float | None = None
    initial_grace_seconds: float = INITIAL_GRACE_SECONDS
    checkpoint_boundary_grace_seconds: float = CHECKPOINT_BOUNDARY_GRACE_SECONDS
    stale_threshold_seconds: float = STALE_THRESHOLD_SECONDS
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS
    status_heartbeat_seconds: float = STATUS_HEARTBEAT_SECONDS
    termination_grace_seconds: float = TERMINATION_GRACE_SECONDS
    expected_total_steps: int | None = None
    checkpoint_every: int = LONG_CHECKPOINT_EVERY
    progress_every: int = LONG_PROGRESS_EVERY


@dataclass(frozen=True, slots=True)
class ChainArtifacts:
    acceptance_snapshot: Path
    semantic_validation: Path
    action_validation: Path
    promotion_receipt: Path
    chain_status: Path
    chain_exit: Path
    restart_log: Path
    restart_status: Path
    restart_exit: Path
    long_log: Path
    long_status: Path
    long_exit: Path

    def outputs(self) -> tuple[Path, ...]:
        return (
            self.acceptance_snapshot,
            self.semantic_validation,
            self.action_validation,
            self.promotion_receipt,
            self.chain_status,
            self.chain_exit,
            self.restart_log,
            self.restart_status,
            self.restart_exit,
            self.long_log,
            self.long_status,
            self.long_exit,
        )


def derive_chain_artifacts(restart_root: Path, long_root: Path) -> ChainArtifacts:
    """Derive every immutable log/output sentinel from the two run roots."""

    def sibling(root: Path, suffix: str) -> Path:
        return root.with_name(f"{root.name}{suffix}")

    return ChainArtifacts(
        acceptance_snapshot=sibling(long_root, ".acceptance-snapshot"),
        semantic_validation=sibling(long_root, ".acceptance-semantic-validation.json"),
        action_validation=sibling(long_root, ".cold-action-evidence-validation.json"),
        promotion_receipt=sibling(long_root, ".post-acceptance-promotion.json"),
        chain_status=sibling(long_root, ".post-acceptance-status.json"),
        chain_exit=sibling(long_root, ".post-acceptance-exit.json"),
        restart_log=sibling(restart_root, ".launcher.log"),
        restart_status=sibling(restart_root, ".supervisor-status.json"),
        restart_exit=sibling(restart_root, ".supervisor-exit.json"),
        long_log=sibling(long_root, ".launcher.log"),
        long_status=sibling(long_root, ".supervisor-status.json"),
        long_exit=sibling(long_root, ".supervisor-exit.json"),
    )


def _signal_handler(signum: int, _frame: FrameType | None) -> None:
    raise SupervisorInterrupted(signum)


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _write_json_atomic_replace(path: Path, value: object) -> None:
    destination = _require_mnt_target(path, name="atomic status output", must_be_absent=False)
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise ValueError("atomic status output must be absent or a direct regular file")
    temporary = _temporary_path(destination)
    payload = (_canonical_json(value) + "\n").encode("ascii")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.is_symlink():
            raise ValueError("atomic status output became a symbolic link")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic_exclusive(path: Path, value: object) -> None:
    payload = (_canonical_json(value) + "\n").encode("ascii")
    _write_bytes_atomic_exclusive(path, payload, name="atomic receipt output")


def _write_bytes_atomic_exclusive(path: Path, payload: bytes, *, name: str) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("atomic byte payload must be bytes")
    destination = _require_mnt_target(path, name=name, must_be_absent=True)
    temporary = _temporary_path(destination)
    lock = destination.with_name(f".{destination.name}.publish.lock")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        lock.mkdir(mode=0o700)
        try:
            _fsync_directory(destination.parent)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            lock.rmdir()
            _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(destination.parent)


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"SHA-256 input is absent or not a direct file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _require_under_mnt(path: Path, *, name: str, must_exist: bool) -> Path:
    raw = str(path)
    if not path.is_absolute() or _contains_control_character(raw):
        raise ValueError(f"{name} must be one absolute control-character-free path")
    resolved = path.resolve(strict=must_exist)
    if resolved != path:
        raise ValueError(f"{name} must be one canonical direct path")
    if resolved == _MNT_ROOT or not resolved.is_relative_to(_MNT_ROOT):
        raise ValueError(f"{name} must live strictly below /mnt")
    return resolved


def _require_mnt_directory(path: Path, *, name: str) -> Path:
    resolved = _require_under_mnt(path, name=name, must_exist=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{name} must be one direct directory")
    return resolved


def _require_mnt_file(path: Path, *, name: str) -> Path:
    resolved = _require_under_mnt(path, name=name, must_exist=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} must be one direct regular file")
    return resolved


def _require_mnt_target(path: Path, *, name: str, must_be_absent: bool) -> Path:
    resolved = _require_under_mnt(path, name=name, must_exist=False)
    parent = _require_mnt_directory(resolved.parent, name=f"{name} parent")
    if parent != resolved.parent:
        raise ValueError(f"{name} parent changed during validation")
    if must_be_absent and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(f"{name} must be absent at startup: {resolved}")
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def preflight_chain_paths(
    *,
    acceptance_root: Path,
    restart_root: Path,
    long_root: Path,
    artifacts: ChainArtifacts,
) -> Path:
    """Fail before any write unless every run/log/output sentinel is fresh and persistent."""

    acceptance = _require_mnt_directory(acceptance_root, name="ADR170 acceptance root")
    restart = _require_mnt_target(
        restart_root,
        name="restart-smoke root",
        must_be_absent=True,
    )
    long = _require_mnt_target(long_root, name="long-run root", must_be_absent=True)
    if _paths_overlap(acceptance, restart) or _paths_overlap(acceptance, long):
        raise ValueError("restart and long roots must not overlap immutable acceptance evidence")
    if _paths_overlap(restart, long):
        raise ValueError("restart and long roots must be distinct non-nested paths")
    outputs = artifacts.outputs()
    if len(set(outputs)) != len(outputs):
        raise ValueError("post-acceptance derived output sentinels are not unique")
    for output in outputs:
        target = _require_mnt_target(
            output,
            name="post-acceptance log/output sentinel",
            must_be_absent=True,
        )
        if any(_paths_overlap(target, root) for root in (acceptance, restart, long)):
            raise ValueError("post-acceptance output sentinel overlaps a run root")
    report = acceptance / "ltop_g3_source_aligned_acceptance.json"
    _require_mnt_file(report, name="ADR170 acceptance report")
    return report


def _read_json_regular(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be one direct regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _read_regular_bytes_no_follow(path: Path, *, name: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{name} must be one canonical direct regular file")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{name} must be one canonical direct regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} must be one canonical direct regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before:
            raise ValueError(f"{name} changed while it was being frozen")
        payload = b"".join(chunks)
        if len(payload) != after.st_size:
            raise ValueError(f"{name} size changed while it was being frozen")
        return payload
    finally:
        os.close(descriptor)


def _json_object_from_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _acceptance_evidence_receipts(
    acceptance_payload: bytes,
) -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
    report = _json_object_from_bytes(
        acceptance_payload,
        name="ADR170 acceptance report",
    )
    if (
        report.get("schema") != ACCEPTANCE_SCHEMA
        or report.get("status") != "PASS"
        or report.get("failures") != []
    ):
        raise ValueError("ADR170 acceptance report is not one completed PASS")
    evidence = report.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(_EXPECTED_EVIDENCE_KEYS):
        raise ValueError("ADR170 acceptance evidence set differs")
    receipts: dict[str, tuple[Path, str]] = {}
    for key in _EXPECTED_EVIDENCE_KEYS:
        item = evidence[key]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"ADR170 acceptance evidence {key} is malformed")
        path_value = item["path"]
        digest = item["sha256"]
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise TypeError(f"ADR170 acceptance evidence {key} has untyped fields")
        path = _require_mnt_file(Path(path_value), name=f"ADR170 evidence {key}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"ADR170 acceptance evidence {key} has malformed SHA-256")
        receipts[key] = (path, digest)
    return report, receipts


def _freeze_acceptance_inputs(
    *,
    acceptance_report: Path,
    snapshot_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    snapshot = _require_mnt_target(
        snapshot_root,
        name="acceptance evidence snapshot root",
        must_be_absent=True,
    )
    acceptance_payload = _read_regular_bytes_no_follow(
        acceptance_report,
        name="ADR170 acceptance report",
    )
    original, source_receipts = _acceptance_evidence_receipts(acceptance_payload)
    snapshot.mkdir(mode=0o700)
    _fsync_directory(snapshot.parent)

    original_snapshot = snapshot / "original-acceptance.json"
    _write_bytes_atomic_exclusive(
        original_snapshot,
        acceptance_payload,
        name="original acceptance snapshot",
    )
    frozen_paths: dict[str, Path] = {}
    evidence_receipts: dict[str, dict[str, str]] = {}
    for key in _EXPECTED_EVIDENCE_KEYS:
        source_path, expected_digest = source_receipts[key]
        payload = _read_regular_bytes_no_follow(
            source_path,
            name=f"ADR170 evidence {key}",
        )
        observed_digest = _sha256_bytes(payload)
        if observed_digest != expected_digest:
            raise ValueError(f"ADR170 acceptance evidence {key} changed after acceptance")
        frozen_path = snapshot / _EVIDENCE_SNAPSHOT_FILENAMES[key]
        _write_bytes_atomic_exclusive(
            frozen_path,
            payload,
            name=f"frozen ADR170 evidence {key}",
        )
        frozen_paths[key] = frozen_path
        evidence_receipts[key] = {
            "source_path": str(source_path),
            "accepted_sha256": expected_digest,
            "snapshot_path": str(frozen_path),
            "snapshot_sha256": observed_digest,
        }
    return (
        original,
        {
            "root": str(snapshot),
            "original_acceptance": {
                "source_path": str(acceptance_report),
                "source_sha256": _sha256_bytes(acceptance_payload),
                "snapshot_path": str(original_snapshot),
                "snapshot_sha256": _sha256_bytes(acceptance_payload),
            },
            "evidence": evidence_receipts,
        },
        frozen_paths,
    )


def _require_repository_file(path: Path, *, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must be one direct regular file")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{name} must be one direct regular file")
    return resolved


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository_root), *arguments),
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed with exit {completed.returncode}: "
            f"{completed.stderr[-1_000:]!r}"
        )
    return completed.stdout.strip()


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository_root), *arguments),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed with exit {completed.returncode}: "
            f"{completed.stderr[-1_000:]!r}"
        )
    return completed.stdout


def _repository_source_receipt(
    *,
    repository_root: Path,
    expected_commit: str,
    source_files: Mapping[str, Path],
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("expected repository commit must be one lowercase 40-hex SHA")
    if _git_output(repository_root, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError("repository root is not inside one Git worktree")
    top_level = Path(_git_output(repository_root, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != repository_root:
        raise ValueError("repository root differs from the Git worktree top level")
    actual_commit = _git_output(repository_root, "rev-parse", "--verify", "HEAD^{commit}")
    if actual_commit != expected_commit:
        raise ValueError("repository HEAD differs from the operator-approved commit")
    status = _git_output(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("repository worktree is not exactly clean")
    tree_sha = _git_output(repository_root, "rev-parse", "--verify", "HEAD^{tree}")
    files: dict[str, dict[str, str]] = {}
    for name, raw_path in source_files.items():
        path = _require_repository_file(raw_path, name=name)
        try:
            relative = path.relative_to(repository_root)
        except ValueError as error:
            raise ValueError(f"{name} must live inside the exact repository checkout") from error
        committed_payload = _git_bytes(
            repository_root,
            "show",
            f"{expected_commit}:{relative.as_posix()}",
        )
        working_payload = _read_regular_bytes_no_follow(path, name=name)
        committed_sha256 = _sha256_bytes(committed_payload)
        working_sha256 = _sha256_bytes(working_payload)
        if working_sha256 != committed_sha256:
            raise ValueError(f"{name} bytes differ from the operator-approved commit")
        files[name] = {
            "path": str(path),
            "repository_relative_path": relative.as_posix(),
            "sha256": working_sha256,
            "committed_sha256": committed_sha256,
        }
    return {
        "repository_root": str(repository_root),
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "tree_sha": tree_sha,
        "worktree_clean": True,
        "files": files,
    }


def _run_checked(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str]) -> None:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode == 0:
        return
    stdout_tail = completed.stdout[-2_000:]
    stderr_tail = completed.stderr[-2_000:]
    raise RuntimeError(
        f"validator failed with exit {completed.returncode}; "
        f"stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}"
    )


def revalidate_acceptance(
    *,
    acceptance_report: Path,
    semantic_validator: Path,
    action_validator: Path,
    semantic_output: Path,
    action_output: Path,
    snapshot_root: Path,
    repository_root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Re-run both frozen semantic composition and independent raw action validation."""

    semantic_tool = _require_repository_file(
        semantic_validator,
        name="ADR170 acceptance semantic validator",
    )
    action_tool = _require_repository_file(
        action_validator,
        name="ADR170 cold-action evidence validator",
    )
    _require_mnt_target(
        semantic_output,
        name="acceptance semantic revalidation output",
        must_be_absent=True,
    )
    original, snapshot_receipt, evidence = _freeze_acceptance_inputs(
        acceptance_report=acceptance_report,
        snapshot_root=snapshot_root,
    )
    _require_mnt_target(
        action_output,
        name="cold-action revalidation output",
        must_be_absent=True,
    )
    _run_checked(
        (
            sys.executable,
            str(semantic_tool),
            "--training-report",
            str(evidence["training_report"]),
            "--arm-validation",
            str(evidence["arm_validation"]),
            "--factual-action-report",
            str(evidence["cold_action_factual"]),
            "--mediator-action-report",
            str(evidence["cold_action_mediator_required"]),
            "--retention-report",
            str(evidence["cold_retention"]),
            "--output",
            str(semantic_output),
        ),
        cwd=repository_root,
        environment=environment,
    )
    recomposed = _read_json_regular(
        semantic_output,
        name="ADR170 recomposed acceptance report",
    )
    expected_recomposed = _json_object_from_bytes(
        _canonical_json(original).encode("ascii"),
        name="normalized ADR170 acceptance report",
    )
    for key, frozen_path in evidence.items():
        expected_recomposed["evidence"][key]["path"] = str(frozen_path)
    if recomposed != expected_recomposed:
        raise ValueError("recomposed ADR170 acceptance differs beyond frozen evidence paths")
    _run_checked(
        (
            sys.executable,
            str(action_tool),
            "--factual-report",
            str(evidence["cold_action_factual"]),
            "--mediator-required-report",
            str(evidence["cold_action_mediator_required"]),
            "--output",
            str(action_output),
        ),
        cwd=repository_root,
        environment=environment,
    )
    action_report = _read_json_regular(action_output, name="cold-action validation report")
    if action_report.get("schema") != ACTION_VALIDATION_SCHEMA:
        raise ValueError("cold-action validation schema differs")
    if action_report.get("status") != "PASS" or action_report.get("failures") != []:
        raise ValueError("cold-action validation did not pass")
    inputs = action_report.get("inputs")
    expected_inputs = {
        "factual": evidence["cold_action_factual"],
        "mediator_required": evidence["cold_action_mediator_required"],
    }
    if not isinstance(inputs, dict) or set(inputs) != set(expected_inputs):
        raise ValueError("cold-action validation input set differs")
    for label, expected_path in expected_inputs.items():
        item = inputs[label]
        if not isinstance(item, dict):
            raise TypeError("cold-action validation input receipt is malformed")
        if item.get("path") != str(expected_path) or item.get("sha256") != _sha256(expected_path):
            raise ValueError("cold-action validation is not bound to accepted evidence")
    return {
        "original_acceptance_report": str(acceptance_report),
        "original_acceptance_report_sha256": snapshot_receipt["original_acceptance"][
            "source_sha256"
        ],
        "acceptance_snapshot": snapshot_receipt,
        "acceptance_report": str(semantic_output),
        "acceptance_report_sha256": _sha256(semantic_output),
        "semantic_validation": str(semantic_output),
        "semantic_validation_sha256": _sha256(semantic_output),
        "action_validation": str(action_output),
        "action_validation_sha256": _sha256(action_output),
    }


def linux_boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if not path.is_file():
        raise RuntimeError("Linux boot identity is unavailable")
    value = path.read_text(encoding="ascii").strip()
    if not value:
        raise RuntimeError("Linux boot identity is empty")
    return value


def process_start_ticks(pid: int) -> int:
    """Read Linux starttime field 22 without assuming the comm field has no spaces."""

    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = payload.rfind(")")
    if closing < 0:
        raise RuntimeError("Linux process stat has no closing comm delimiter")
    fields_after_comm = payload[closing + 1 :].split()
    if len(fields_after_comm) <= 19:
        raise RuntimeError("Linux process stat omits its start tick")
    value = int(fields_after_comm[19])
    if value <= 0:
        raise RuntimeError("Linux process start tick is invalid")
    return value


def _capture_process_identity(process: subprocess.Popen[bytes]) -> ProcessIdentity:
    boot_id = linux_boot_id()
    deadline = time.monotonic() + 5.0
    while True:
        try:
            return ProcessIdentity(
                pid=process.pid,
                start_ticks=process_start_ticks(process.pid),
                boot_id=boot_id,
            )
        except FileNotFoundError:
            returncode = process.poll()
            if returncode is not None or time.monotonic() >= deadline:
                raise RuntimeError(
                    f"child exited before process identity capture: {returncode}"
                ) from None
            time.sleep(0.01)


def _identity_failure(
    process: subprocess.Popen[bytes],
    identity: ProcessIdentity,
) -> str | None:
    if linux_boot_id() != identity.boot_id:
        return "boot-id-changed"
    try:
        observed_ticks = process_start_ticks(identity.pid)
    except FileNotFoundError:
        if process.poll() is not None:
            return None
        time.sleep(0.05)
        try:
            observed_ticks = process_start_ticks(identity.pid)
        except FileNotFoundError:
            return "process-identity-missing"
    if observed_ticks != identity.start_ticks:
        return "process-start-tick-changed"
    return None


def _read_progress(path: Path, *, expected_total_steps: int) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("LTOP progress must be one direct regular file")
    payload = _read_json_regular(path, name="LTOP progress")
    completed = payload.get("completed_steps")
    total = payload.get("total_steps")
    if payload.get("schema") != PROGRESS_SCHEMA:
        raise ValueError("LTOP progress schema differs")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise TypeError("LTOP progress completed_steps must be an integer")
    if completed < 0 or completed > expected_total_steps:
        raise ValueError("LTOP progress completed_steps is out of range")
    if total != expected_total_steps:
        raise ValueError("LTOP progress total_steps differs")
    return payload


def _checkpoint_boundary_pending(
    completed_steps: int,
    *,
    checkpoint_every: int,
    progress_every: int,
) -> bool:
    if completed_steps <= 0:
        return False
    if completed_steps % checkpoint_every == 0:
        return True
    next_boundary = ((completed_steps // checkpoint_every) + 1) * checkpoint_every
    return completed_steps < next_boundary <= completed_steps + progress_every


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
) -> dict[str, Any]:
    if process.poll() is not None:
        return {"signal": None, "forced": False}
    sent_signal = "SIGTERM"
    forced = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"signal": None, "forced": False}
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        forced = True
        sent_signal = "SIGKILL"
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    return {"signal": sent_signal, "forced": forced}


def _status_payload(
    *,
    spec: SupervisionSpec,
    phase: str,
    identity: ProcessIdentity,
    started_unix_s: float,
    started_monotonic: float,
    last_completed_steps: int | None,
    last_progress_change_monotonic: float,
    active_stale_threshold_seconds: float | None,
    progress_error: str | None,
) -> dict[str, Any]:
    now_monotonic = time.monotonic()
    return {
        "schema": STATUS_SCHEMA,
        "status": "RUNNING" if phase == "running" else phase.upper(),
        "kind": spec.kind,
        "phase": phase,
        "run_root": str(spec.run_root),
        "log_output": str(spec.log_output),
        "progress_path": None if spec.progress_path is None else str(spec.progress_path),
        "process_identity": {
            "pid": identity.pid,
            "start_ticks": identity.start_ticks,
            "boot_id": identity.boot_id,
        },
        "command": list(spec.command),
        "started_unix_s": started_unix_s,
        "updated_unix_s": time.time(),
        "elapsed_monotonic_s": now_monotonic - started_monotonic,
        "last_completed_steps": last_completed_steps,
        "seconds_since_progress_change": now_monotonic - last_progress_change_monotonic,
        "active_stale_threshold_seconds": active_stale_threshold_seconds,
        "progress_error": progress_error,
        "policy": {
            "passive_first": True,
            "automatic_recovery": False,
            "scientific_thresholds": False,
            "initial_grace_seconds": spec.initial_grace_seconds,
            "checkpoint_boundary_grace_seconds": (spec.checkpoint_boundary_grace_seconds),
            "stale_threshold_seconds": spec.stale_threshold_seconds,
        },
    }


def _validate_supervision_spec(spec: SupervisionSpec) -> None:
    if spec.kind not in {"restart-smoke", "long"}:
        raise ValueError("supervisor kind must be restart-smoke or long")
    if not spec.command or any(not value for value in spec.command):
        raise ValueError("supervised command must be non-empty")
    if spec.poll_interval_seconds <= 0 or spec.status_heartbeat_seconds <= 0:
        raise ValueError("supervisor polling and heartbeat intervals must be positive")
    if spec.termination_grace_seconds <= 0:
        raise ValueError("supervisor termination grace must be positive")
    if spec.kind == "long":
        if spec.progress_path is None or spec.expected_total_steps is None:
            raise ValueError("long supervision requires progress and expected total steps")
        if (
            min(
                spec.initial_grace_seconds,
                spec.checkpoint_boundary_grace_seconds,
                spec.stale_threshold_seconds,
            )
            <= 0
        ):
            raise ValueError("long supervision grace thresholds must be positive")
        if spec.checkpoint_every <= 0 or spec.progress_every <= 0:
            raise ValueError("long checkpoint/progress cadences must be positive")
    elif spec.timeout_seconds is None or spec.timeout_seconds <= 0:
        raise ValueError("restart-smoke supervision requires a positive timeout")
    _require_mnt_target(spec.run_root, name="supervised run root", must_be_absent=True)
    for name, path in (
        ("supervisor log", spec.log_output),
        ("supervisor status", spec.status_output),
        ("supervisor exit", spec.exit_output),
    ):
        _require_mnt_target(path, name=name, must_be_absent=True)
    if spec.progress_path is not None:
        progress = _require_under_mnt(
            spec.progress_path,
            name="supervised progress path",
            must_exist=False,
        )
        if not progress.is_relative_to(spec.run_root):
            raise ValueError("supervised progress must live below its run root")


def supervise_process(spec: SupervisionSpec) -> ProcessOutcome:
    """Launch one child and terminate only for explicit loss, failure, timeout, or stale."""

    _validate_supervision_spec(spec)
    started_unix_s = time.time()
    started_monotonic = time.monotonic()
    last_progress_change = started_monotonic
    last_status_write = started_monotonic
    last_completed_steps: int | None = None
    progress_error: str | None = None
    reason = "supervisor-error"
    status = "FAIL"
    returncode: int | None = None
    termination: dict[str, Any] = {"signal": None, "forced": False}
    identity: ProcessIdentity | None = None
    process: subprocess.Popen[bytes] | None = None

    try:
        with spec.log_output.open("xb") as log_stream:
            log_stream.flush()
            os.fsync(log_stream.fileno())
            _fsync_directory(spec.log_output.parent)
            process = subprocess.Popen(
                list(spec.command),
                cwd=spec.cwd,
                env=dict(spec.environment),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            identity = _capture_process_identity(process)
            _write_json_atomic_replace(
                spec.status_output,
                _status_payload(
                    spec=spec,
                    phase="running",
                    identity=identity,
                    started_unix_s=started_unix_s,
                    started_monotonic=started_monotonic,
                    last_completed_steps=None,
                    last_progress_change_monotonic=last_progress_change,
                    active_stale_threshold_seconds=(
                        spec.initial_grace_seconds if spec.kind == "long" else None
                    ),
                    progress_error=None,
                ),
            )
            while True:
                now = time.monotonic()
                returncode = process.poll()
                if returncode is not None:
                    if returncode != 0:
                        reason = "nonzero-exit"
                        break
                    if spec.kind == "long":
                        assert spec.progress_path is not None
                        assert spec.expected_total_steps is not None
                        final_progress = _read_progress(
                            spec.progress_path,
                            expected_total_steps=spec.expected_total_steps,
                        )
                        if (
                            final_progress is None
                            or final_progress.get("completed_steps") != spec.expected_total_steps
                        ):
                            reason = "incomplete-zero-exit"
                            break
                        last_completed_steps = spec.expected_total_steps
                    reason = "completed"
                    status = "PASS"
                    break

                identity_failure = _identity_failure(process, identity)
                if identity_failure is not None:
                    reason = identity_failure
                    termination = _terminate_process_group(
                        process,
                        grace_seconds=spec.termination_grace_seconds,
                    )
                    returncode = process.poll()
                    break

                active_threshold: float | None = None
                if spec.kind == "restart-smoke":
                    assert spec.timeout_seconds is not None
                    if now - started_monotonic > spec.timeout_seconds:
                        reason = "timeout"
                        termination = _terminate_process_group(
                            process,
                            grace_seconds=spec.termination_grace_seconds,
                        )
                        returncode = process.poll()
                        break
                else:
                    assert spec.progress_path is not None
                    assert spec.expected_total_steps is not None
                    try:
                        progress = _read_progress(
                            spec.progress_path,
                            expected_total_steps=spec.expected_total_steps,
                        )
                        progress_error = None
                    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                        progress = None
                        progress_error = f"{type(error).__name__}: {error}"
                    if progress is not None:
                        completed = int(progress["completed_steps"])
                        if last_completed_steps is None or completed > last_completed_steps:
                            last_completed_steps = completed
                            last_progress_change = now
                        elif completed < last_completed_steps:
                            progress_error = "completed_steps moved backwards"
                    if last_completed_steps is None:
                        active_threshold = spec.initial_grace_seconds
                        stale_reason = "initial-progress-stale"
                    elif _checkpoint_boundary_pending(
                        last_completed_steps,
                        checkpoint_every=spec.checkpoint_every,
                        progress_every=spec.progress_every,
                    ):
                        active_threshold = spec.checkpoint_boundary_grace_seconds
                        stale_reason = "checkpoint-progress-stale"
                    else:
                        active_threshold = spec.stale_threshold_seconds
                        stale_reason = "progress-stale"
                    if now - last_progress_change > active_threshold:
                        reason = stale_reason
                        termination = _terminate_process_group(
                            process,
                            grace_seconds=spec.termination_grace_seconds,
                        )
                        returncode = process.poll()
                        break

                if (
                    now - last_status_write >= spec.status_heartbeat_seconds
                    or now == last_progress_change
                ):
                    _write_json_atomic_replace(
                        spec.status_output,
                        _status_payload(
                            spec=spec,
                            phase="running",
                            identity=identity,
                            started_unix_s=started_unix_s,
                            started_monotonic=started_monotonic,
                            last_completed_steps=last_completed_steps,
                            last_progress_change_monotonic=last_progress_change,
                            active_stale_threshold_seconds=active_threshold,
                            progress_error=progress_error,
                        ),
                    )
                    last_status_write = now
                time.sleep(spec.poll_interval_seconds)
    except SupervisorInterrupted as error:
        reason = f"interrupted-signal-{error.signum}"
        if process is not None:
            termination = _terminate_process_group(
                process,
                grace_seconds=spec.termination_grace_seconds,
            )
            returncode = process.poll()
    except BaseException as error:
        reason = f"supervisor-error:{type(error).__name__}:{error}"
        if process is not None:
            termination = _terminate_process_group(
                process,
                grace_seconds=spec.termination_grace_seconds,
            )
            returncode = process.poll()

    finished_unix_s = time.time()
    exit_payload = {
        "schema": EXIT_SCHEMA,
        "status": status,
        "kind": spec.kind,
        "reason": reason,
        "returncode": returncode,
        "run_root": str(spec.run_root),
        "log_output": str(spec.log_output),
        "status_output": str(spec.status_output),
        "progress_path": None if spec.progress_path is None else str(spec.progress_path),
        "process_identity": (
            None
            if identity is None
            else {
                "pid": identity.pid,
                "start_ticks": identity.start_ticks,
                "boot_id": identity.boot_id,
            }
        ),
        "command": list(spec.command),
        "started_unix_s": started_unix_s,
        "finished_unix_s": finished_unix_s,
        "duration_s": time.monotonic() - started_monotonic,
        "last_completed_steps": last_completed_steps,
        "termination": termination,
        "automatic_recovery_attempted": False,
        "scientific_threshold_applied": False,
    }
    _write_json_atomic_exclusive(spec.exit_output, exit_payload)
    if identity is not None:
        _write_json_atomic_replace(
            spec.status_output,
            _status_payload(
                spec=spec,
                phase="passed" if status == "PASS" else "failed",
                identity=identity,
                started_unix_s=started_unix_s,
                started_monotonic=started_monotonic,
                last_completed_steps=last_completed_steps,
                last_progress_change_monotonic=last_progress_change,
                active_stale_threshold_seconds=None,
                progress_error=progress_error,
            ),
        )
    return ProcessOutcome(
        status=status,
        reason=reason,
        returncode=returncode,
        exit_receipt=spec.exit_output,
        last_completed_steps=last_completed_steps,
    )


def _validate_restart_report(restart_root: Path, acceptance_sha256: str) -> dict[str, Any]:
    report_path = _require_mnt_file(
        restart_root / "ltop_core_pilot_report.json",
        name="restart-smoke terminal report",
    )
    report = _read_json_regular(report_path, name="restart-smoke terminal report")
    cadence = report.get("cadence")
    if (
        report.get("status") != "PASS"
        or report.get("mode") != "restart-smoke"
        or report.get("stop_global_step") != 4
        or cadence
        != {
            "total_steps": 4,
            "metrics_every": 2,
            "diagnostics_every": 2,
            "checkpoint_every": 2,
        }
    ):
        raise ValueError("restart-smoke terminal report does not prove the frozen 4/2/2/2 path")
    if report.get("g3_report_sha256") != acceptance_sha256:
        raise ValueError("restart-smoke did not consume the revalidated acceptance report")
    return {"path": str(report_path), "sha256": _sha256(report_path)}


def _frozen_long_contract() -> dict[str, Any]:
    from picf_next.lingbot_native.ltop_core_pilot import (
        LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY,
        LTOP_CORE_PILOT_WORLD_SIZE,
        LTOPCoreLongCadence,
    )

    cadence = LTOPCoreLongCadence()
    observed = {
        "total_steps": cadence.total_steps,
        "metrics_every": cadence.metrics_every,
        "diagnostics_every": cadence.diagnostics_every,
        "checkpoint_every": cadence.checkpoint_every,
        "world_size": LTOP_CORE_PILOT_WORLD_SIZE,
        "action_information_set_policy": LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY,
    }
    expected = {
        "total_steps": 30_000,
        "metrics_every": 100,
        "diagnostics_every": 250,
        "checkpoint_every": 2_000,
        "world_size": 2,
        "action_information_set_policy": "rank-step-counterbalanced-50-50",
    }
    if observed != expected:
        raise ValueError("frozen LTOP long contract differs from 30k/100/250/2k/two-rank")
    return observed


def _chain_status(
    *,
    artifacts: ChainArtifacts,
    phase: str,
    acceptance_report: Path,
    restart_root: Path,
    long_root: Path,
    details: object | None = None,
) -> None:
    _write_json_atomic_replace(
        artifacts.chain_status,
        {
            "schema": CHAIN_STATUS_SCHEMA,
            "status": "RUNNING",
            "phase": phase,
            "acceptance_report": str(acceptance_report),
            "restart_root": str(restart_root),
            "long_root": str(long_root),
            "details": details,
            "updated_unix_s": time.time(),
        },
    )


def _run_chain(args: argparse.Namespace) -> int:
    repository_argument = Path(args.repository_root)
    if repository_argument.is_symlink():
        raise ValueError("repository root must be one direct directory")
    repository_root = repository_argument.resolve(strict=True)
    if repository_root != repository_argument or not repository_root.is_dir():
        raise ValueError("repository root must be one canonical direct directory")
    source_files = {
        "post_acceptance_supervisor": Path(__file__).resolve(),
        "post_acceptance_launcher": (
            repository_root / "adr170/run_ltop_g3_source_aligned_post_acceptance_2gpu.sh"
        ),
        "semantic_validator": Path(args.semantic_validator),
        "action_validator": Path(args.action_validator),
        "restart_launcher": Path(args.restart_launcher),
        "long_launcher": Path(args.long_launcher),
        "training_runner": repository_root / "tools/run_lingbot_vla2_ltop_core_pilot.py",
        "picf_native_patch": (
            repository_root / "references/patches/lingbot_vla2_picf_native.patch"
        ),
        "distributed_alignment_patch": (
            repository_root
            / "references/patches/lingbot_vla2_distributed_muon_collective_alignment.patch"
        ),
    }
    source_receipt = _repository_source_receipt(
        repository_root=repository_root,
        expected_commit=args.expected_repository_commit,
        source_files=source_files,
    )
    acceptance_root = Path(args.acceptance_root)
    restart_root = Path(args.restart_root)
    long_root = Path(args.long_root)
    artifacts = derive_chain_artifacts(restart_root, long_root)
    acceptance_report = preflight_chain_paths(
        acceptance_root=acceptance_root,
        restart_root=restart_root,
        long_root=long_root,
        artifacts=artifacts,
    )
    if args.initial_grace_seconds < INITIAL_GRACE_SECONDS:
        raise ValueError("production initial grace cannot be below 3600 seconds")
    if args.checkpoint_boundary_grace_seconds < CHECKPOINT_BOUNDARY_GRACE_SECONDS:
        raise ValueError("production checkpoint-boundary grace cannot be below 3600 seconds")
    if args.stale_threshold_seconds < STALE_THRESHOLD_SECONDS:
        raise ValueError("production stale threshold cannot be below 900 seconds")

    environment = dict(os.environ)
    pythonpath = f"{repository_root}:{repository_root / 'src'}"
    if environment.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}:{environment['PYTHONPATH']}"
    environment["PYTHONPATH"] = pythonpath
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PICF_REPOSITORY_ROOT"] = str(repository_root)

    started_unix_s = time.time()
    phase = "preflight"
    exit_status = "FAIL"
    exit_reason = "post-acceptance-chain-failed"
    details: object | None = None
    try:
        _chain_status(
            artifacts=artifacts,
            phase=phase,
            acceptance_report=acceptance_report,
            restart_root=restart_root,
            long_root=long_root,
        )
        long_contract = _frozen_long_contract()
        phase = "acceptance-revalidation"
        _chain_status(
            artifacts=artifacts,
            phase=phase,
            acceptance_report=acceptance_report,
            restart_root=restart_root,
            long_root=long_root,
            details=long_contract,
        )
        validation = revalidate_acceptance(
            acceptance_report=acceptance_report,
            semantic_validator=Path(args.semantic_validator),
            action_validator=Path(args.action_validator),
            semantic_output=artifacts.semantic_validation,
            action_output=artifacts.action_validation,
            snapshot_root=artifacts.acceptance_snapshot,
            repository_root=repository_root,
            environment=environment,
        )
        source_after_validation = _repository_source_receipt(
            repository_root=repository_root,
            expected_commit=args.expected_repository_commit,
            source_files=source_files,
        )
        if source_after_validation != source_receipt:
            raise ValueError("repository source changed during acceptance revalidation")
        acceptance_report = _require_mnt_file(
            Path(validation["acceptance_report"]),
            name="frozen revalidated ADR170 acceptance report",
        )
        environment["PICF_G3_ACCEPTANCE_REPORT"] = str(acceptance_report)
        for target in (
            restart_root,
            artifacts.restart_log,
            artifacts.restart_status,
            artifacts.restart_exit,
        ):
            _require_mnt_target(
                target,
                name="restart-smoke launch sentinel",
                must_be_absent=True,
            )
        phase = "restart-smoke"
        _chain_status(
            artifacts=artifacts,
            phase=phase,
            acceptance_report=acceptance_report,
            restart_root=restart_root,
            long_root=long_root,
            details=validation,
        )
        restart_launcher = _require_repository_file(
            Path(args.restart_launcher),
            name="restart-smoke launcher",
        )
        if not os.access(restart_launcher, os.X_OK):
            raise ValueError("restart-smoke launcher must be executable")
        restart_outcome = supervise_process(
            SupervisionSpec(
                kind="restart-smoke",
                command=(
                    str(restart_launcher),
                    str(restart_root),
                    "fresh",
                ),
                cwd=repository_root,
                environment=environment,
                run_root=restart_root,
                log_output=artifacts.restart_log,
                status_output=artifacts.restart_status,
                exit_output=artifacts.restart_exit,
                timeout_seconds=args.restart_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                status_heartbeat_seconds=args.status_heartbeat_seconds,
                termination_grace_seconds=args.termination_grace_seconds,
            )
        )
        if restart_outcome.status != "PASS":
            raise RuntimeError(f"restart-smoke failed: {restart_outcome.reason}")
        restart_report = _validate_restart_report(
            restart_root,
            validation["acceptance_report_sha256"],
        )
        source_before_long = _repository_source_receipt(
            repository_root=repository_root,
            expected_commit=args.expected_repository_commit,
            source_files=source_files,
        )
        if source_before_long != source_receipt:
            raise ValueError("repository source changed before long-run promotion")
        promotion = {
            "schema": PROMOTION_SCHEMA,
            "status": "PASS",
            "validation": validation,
            "source": source_receipt,
            "restart_smoke": {
                "root": str(restart_root),
                "report": restart_report,
                "exit_receipt": str(artifacts.restart_exit),
                "exit_receipt_sha256": _sha256(artifacts.restart_exit),
            },
            "long_contract": long_contract,
            "watchdog_policy": {
                "passive_first": True,
                "automatic_recovery": False,
                "scientific_thresholds": False,
                "initial_grace_seconds": args.initial_grace_seconds,
                "checkpoint_boundary_grace_seconds": (args.checkpoint_boundary_grace_seconds),
                "stale_threshold_seconds": args.stale_threshold_seconds,
            },
        }
        _write_json_atomic_exclusive(artifacts.promotion_receipt, promotion)
        for target in (
            long_root,
            artifacts.long_log,
            artifacts.long_status,
            artifacts.long_exit,
        ):
            _require_mnt_target(
                target,
                name="long-run launch sentinel",
                must_be_absent=True,
            )
        phase = "long"
        _chain_status(
            artifacts=artifacts,
            phase=phase,
            acceptance_report=acceptance_report,
            restart_root=restart_root,
            long_root=long_root,
            details={"promotion_receipt": str(artifacts.promotion_receipt)},
        )
        long_launcher = _require_repository_file(
            Path(args.long_launcher),
            name="long-run launcher",
        )
        if not os.access(long_launcher, os.X_OK):
            raise ValueError("long-run launcher must be executable")
        long_outcome = supervise_process(
            SupervisionSpec(
                kind="long",
                command=(
                    str(long_launcher),
                    str(long_root),
                    "fresh",
                ),
                cwd=repository_root,
                environment=environment,
                run_root=long_root,
                log_output=artifacts.long_log,
                status_output=artifacts.long_status,
                exit_output=artifacts.long_exit,
                progress_path=long_root / "progress.json",
                initial_grace_seconds=args.initial_grace_seconds,
                checkpoint_boundary_grace_seconds=(args.checkpoint_boundary_grace_seconds),
                stale_threshold_seconds=args.stale_threshold_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                status_heartbeat_seconds=args.status_heartbeat_seconds,
                termination_grace_seconds=args.termination_grace_seconds,
                expected_total_steps=LONG_TOTAL_STEPS,
                checkpoint_every=LONG_CHECKPOINT_EVERY,
                progress_every=LONG_PROGRESS_EVERY,
            )
        )
        if long_outcome.status != "PASS":
            raise RuntimeError(f"long run failed: {long_outcome.reason}")
        phase = "completed"
        exit_status = "PASS"
        exit_reason = "completed"
        details = {
            "promotion_receipt": str(artifacts.promotion_receipt),
            "long_exit_receipt": str(artifacts.long_exit),
        }
        _chain_status(
            artifacts=artifacts,
            phase=phase,
            acceptance_report=acceptance_report,
            restart_root=restart_root,
            long_root=long_root,
            details=details,
        )
    except SupervisorInterrupted as error:
        exit_reason = f"interrupted-signal-{error.signum}"
        details = {"error": str(error), "phase": phase}
    except BaseException as error:
        exit_reason = f"{type(error).__name__}: {error}"
        details = {"error": str(error), "phase": phase}

    _write_json_atomic_exclusive(
        artifacts.chain_exit,
        {
            "schema": CHAIN_EXIT_SCHEMA,
            "status": exit_status,
            "reason": exit_reason,
            "phase": phase,
            "acceptance_report": str(acceptance_report),
            "restart_root": str(restart_root),
            "long_root": str(long_root),
            "details": details,
            "started_unix_s": started_unix_s,
            "finished_unix_s": time.time(),
        },
    )
    if exit_status == "PASS":
        return 0
    sys.stderr.write(f"ADR170 post-acceptance chain failed in {phase}: {exit_reason}\n")
    return 1


def _run_watch(args: argparse.Namespace) -> int:
    command = tuple(args.child_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("watch requires one child command after --")
    run_root = Path(args.run_root)
    progress = None if args.kind == "restart-smoke" else Path(args.progress)
    outcome = supervise_process(
        SupervisionSpec(
            kind=args.kind,
            command=command,
            cwd=Path(args.cwd).resolve(strict=True),
            environment=dict(os.environ),
            run_root=run_root,
            log_output=Path(args.log_output),
            status_output=Path(args.status_output),
            exit_output=Path(args.exit_output),
            progress_path=progress,
            timeout_seconds=args.timeout_seconds,
            initial_grace_seconds=args.initial_grace_seconds,
            checkpoint_boundary_grace_seconds=args.checkpoint_boundary_grace_seconds,
            stale_threshold_seconds=args.stale_threshold_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            status_heartbeat_seconds=args.status_heartbeat_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
            expected_total_steps=(
                None if args.kind == "restart-smoke" else args.expected_total_steps
            ),
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
        )
    )
    return 0 if outcome.status == "PASS" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    chain = subparsers.add_parser("chain", help="run the complete ADR170 post-acceptance chain")
    chain.add_argument("--acceptance-root", required=True)
    chain.add_argument("--restart-root", required=True)
    chain.add_argument("--long-root", required=True)
    chain.add_argument("--repository-root", required=True)
    chain.add_argument("--expected-repository-commit", required=True)
    chain.add_argument("--semantic-validator", required=True)
    chain.add_argument("--action-validator", required=True)
    chain.add_argument("--restart-launcher", required=True)
    chain.add_argument("--long-launcher", required=True)
    chain.add_argument("--restart-timeout-seconds", type=float, default=RESTART_TIMEOUT_SECONDS)
    chain.add_argument("--initial-grace-seconds", type=float, default=INITIAL_GRACE_SECONDS)
    chain.add_argument(
        "--checkpoint-boundary-grace-seconds",
        type=float,
        default=CHECKPOINT_BOUNDARY_GRACE_SECONDS,
    )
    chain.add_argument("--stale-threshold-seconds", type=float, default=STALE_THRESHOLD_SECONDS)
    chain.add_argument("--poll-interval-seconds", type=float, default=POLL_INTERVAL_SECONDS)
    chain.add_argument(
        "--status-heartbeat-seconds",
        type=float,
        default=STATUS_HEARTBEAT_SECONDS,
    )
    chain.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=TERMINATION_GRACE_SECONDS,
    )

    watch = subparsers.add_parser("watch", help="launch and supervise one bounded or long child")
    watch.add_argument("--kind", choices=("restart-smoke", "long"), required=True)
    watch.add_argument("--run-root", required=True)
    watch.add_argument("--log-output", required=True)
    watch.add_argument("--status-output", required=True)
    watch.add_argument("--exit-output", required=True)
    watch.add_argument("--progress", default="")
    watch.add_argument("--cwd", default=str(Path.cwd()))
    watch.add_argument("--timeout-seconds", type=float, default=RESTART_TIMEOUT_SECONDS)
    watch.add_argument("--initial-grace-seconds", type=float, default=INITIAL_GRACE_SECONDS)
    watch.add_argument(
        "--checkpoint-boundary-grace-seconds",
        type=float,
        default=CHECKPOINT_BOUNDARY_GRACE_SECONDS,
    )
    watch.add_argument("--stale-threshold-seconds", type=float, default=STALE_THRESHOLD_SECONDS)
    watch.add_argument("--poll-interval-seconds", type=float, default=POLL_INTERVAL_SECONDS)
    watch.add_argument(
        "--status-heartbeat-seconds",
        type=float,
        default=STATUS_HEARTBEAT_SECONDS,
    )
    watch.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=TERMINATION_GRACE_SECONDS,
    )
    watch.add_argument("--expected-total-steps", type=int, default=LONG_TOTAL_STEPS)
    watch.add_argument("--checkpoint-every", type=int, default=LONG_CHECKPOINT_EVERY)
    watch.add_argument("--progress-every", type=int, default=LONG_PROGRESS_EVERY)
    watch.add_argument("child_command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _install_signal_handlers()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.subcommand == "chain":
            return _run_chain(args)
        return _run_watch(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(f"{type(error).__name__}: {error}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
