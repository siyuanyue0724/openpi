from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import watch_ltop_core_long_health as supervisor


def _persistent_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    mount = tmp_path / "mnt"
    workspace = mount / "picf-next" / "runs"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(supervisor, "_MNT_ROOT", mount.resolve())
    return workspace.resolve()


def _progress_child(run_root: Path, *, delay: float = 0.15) -> tuple[str, ...]:
    source = """
import json
import os
import pathlib
import sys
import time

root = pathlib.Path(sys.argv[1])
root.mkdir()
progress = root / "progress.json"

def publish(step):
    temporary = root / f".progress.{step}.tmp"
    temporary.write_text(json.dumps({
        "schema": "picf-next.ltop-core-pilot-progress.v1",
        "completed_steps": step,
        "total_steps": 4,
    }), encoding="ascii")
    os.replace(temporary, progress)

publish(2)
time.sleep(float(sys.argv[2]))
publish(4)
"""
    return (sys.executable, "-c", source, str(run_root), str(delay))


def test_versioned_launcher_exposes_only_three_roots_and_frozen_tools() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "adr170/run_ltop_g3_source_aligned_post_acceptance_2gpu.sh").read_text(
        encoding="utf-8"
    )
    watchdog = (root / "tools/watch_ltop_core_long_health.py").read_text(encoding="utf-8")

    assert "[[ $# -ne 3 ]]" in launcher
    assert "ACCEPTANCE_ROOT RESTART_ROOT LONG_ROOT" in launcher
    assert "compose_ltop_g3_source_aligned_acceptance.py" in launcher
    assert "validate_ltop_g3_cold_action_evidence.py" in launcher
    assert "run_ltop_core_restart_smoke_2gpu.sh" in launcher
    assert "run_ltop_core_long_2gpu.sh" in launcher
    assert "status --porcelain=v1 --untracked-files=all" in launcher
    assert "PICF_ADR170_EXPECTED_SOURCE_COMMIT" in launcher
    assert "--expected-repository-commit" in launcher
    assert launcher.index("--restart-root") < launcher.index("--long-root")
    assert "--restart-timeout-seconds" in launcher
    assert 'automatic_recovery_attempted": False' in watchdog
    assert 'scientific_threshold_applied": False' in watchdog
    assert watchdog.index('phase = "restart-smoke"') < watchdog.index('phase = "long"')


def test_production_watchdog_defaults_are_passive_and_conservative() -> None:
    assert supervisor.INITIAL_GRACE_SECONDS >= 3_600
    assert supervisor.CHECKPOINT_BOUNDARY_GRACE_SECONDS >= 3_600
    assert supervisor.STALE_THRESHOLD_SECONDS >= 900
    assert supervisor.LONG_TOTAL_STEPS == 30_000
    assert supervisor.LONG_CHECKPOINT_EVERY == 2_000
    assert supervisor.LONG_PROGRESS_EVERY == 8


def test_preflight_rejects_any_existing_run_log_or_output_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _persistent_workspace(tmp_path, monkeypatch)
    acceptance = workspace / "acceptance"
    acceptance.mkdir()
    (acceptance / "ltop_g3_source_aligned_acceptance.json").write_text(
        "{}\n",
        encoding="ascii",
    )
    restart = workspace / "restart"
    long = workspace / "long"
    artifacts = supervisor.derive_chain_artifacts(restart, long)

    report = supervisor.preflight_chain_paths(
        acceptance_root=acceptance,
        restart_root=restart,
        long_root=long,
        artifacts=artifacts,
    )
    assert report == acceptance / "ltop_g3_source_aligned_acceptance.json"

    artifacts.long_log.write_text("stale\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="absent at startup"):
        supervisor.preflight_chain_paths(
            acceptance_root=acceptance,
            restart_root=restart,
            long_root=long,
            artifacts=artifacts,
        )


def test_watchdog_preserves_checkpoint_boundary_grace_and_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _persistent_workspace(tmp_path, monkeypatch)
    run_root = workspace / "long"
    outcome = supervisor.supervise_process(
        supervisor.SupervisionSpec(
            kind="long",
            command=_progress_child(run_root),
            cwd=workspace,
            environment=dict(os.environ),
            run_root=run_root,
            log_output=workspace / "long.log",
            status_output=workspace / "long.status.json",
            exit_output=workspace / "long.exit.json",
            progress_path=run_root / "progress.json",
            initial_grace_seconds=0.5,
            checkpoint_boundary_grace_seconds=0.5,
            stale_threshold_seconds=0.05,
            poll_interval_seconds=0.01,
            status_heartbeat_seconds=0.02,
            termination_grace_seconds=0.1,
            expected_total_steps=4,
            checkpoint_every=2,
            progress_every=2,
        )
    )

    assert outcome.status == "PASS"
    assert outcome.reason == "completed"
    assert outcome.last_completed_steps == 4
    receipt = json.loads(outcome.exit_receipt.read_text(encoding="ascii"))
    assert receipt["status"] == "PASS"
    assert receipt["process_identity"]["pid"] > 0
    assert receipt["process_identity"]["start_ticks"] > 0
    assert receipt["process_identity"]["boot_id"]
    assert receipt["automatic_recovery_attempted"] is False
    assert receipt["scientific_threshold_applied"] is False


def test_watchdog_terminates_only_after_explicit_initial_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _persistent_workspace(tmp_path, monkeypatch)
    run_root = workspace / "stale"
    source = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).mkdir(); time.sleep(10)"
    outcome = supervisor.supervise_process(
        supervisor.SupervisionSpec(
            kind="long",
            command=(sys.executable, "-c", source, str(run_root)),
            cwd=workspace,
            environment=dict(os.environ),
            run_root=run_root,
            log_output=workspace / "stale.log",
            status_output=workspace / "stale.status.json",
            exit_output=workspace / "stale.exit.json",
            progress_path=run_root / "progress.json",
            initial_grace_seconds=0.05,
            checkpoint_boundary_grace_seconds=0.2,
            stale_threshold_seconds=0.05,
            poll_interval_seconds=0.01,
            status_heartbeat_seconds=0.02,
            termination_grace_seconds=0.1,
            expected_total_steps=4,
            checkpoint_every=2,
            progress_every=2,
        )
    )

    assert outcome.status == "FAIL"
    assert outcome.reason == "initial-progress-stale"
    receipt = json.loads(outcome.exit_receipt.read_text(encoding="ascii"))
    assert receipt["termination"]["signal"] in {"SIGTERM", "SIGKILL"}
    assert receipt["automatic_recovery_attempted"] is False


def test_restart_smoke_nonzero_exit_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _persistent_workspace(tmp_path, monkeypatch)
    run_root = workspace / "restart"
    source = (
        "import pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).mkdir(); "
        "time.sleep(0.05); "
        "raise SystemExit(7)"
    )
    outcome = supervisor.supervise_process(
        supervisor.SupervisionSpec(
            kind="restart-smoke",
            command=(sys.executable, "-c", source, str(run_root)),
            cwd=workspace,
            environment=dict(os.environ),
            run_root=run_root,
            log_output=workspace / "restart.log",
            status_output=workspace / "restart.status.json",
            exit_output=workspace / "restart.exit.json",
            timeout_seconds=1.0,
            poll_interval_seconds=0.01,
            status_heartbeat_seconds=0.02,
            termination_grace_seconds=0.1,
        )
    )

    assert outcome.status == "FAIL"
    assert outcome.reason == "nonzero-exit"
    assert outcome.returncode == 7


def test_atomic_receipts_leave_no_staging_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _persistent_workspace(tmp_path, monkeypatch)
    status = workspace / "status.json"
    receipt = workspace / "exit.json"

    supervisor._write_json_atomic_replace(status, {"value": 1})
    supervisor._write_json_atomic_replace(status, {"value": 2})
    supervisor._write_json_atomic_exclusive(receipt, {"status": "PASS"})

    assert json.loads(status.read_text(encoding="ascii")) == {"value": 2}
    assert json.loads(receipt.read_text(encoding="ascii")) == {"status": "PASS"}
    assert not [path for path in workspace.iterdir() if path.name.startswith(".")]


def test_atomic_receipt_never_replaces_existing_output_or_removes_foreign_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _persistent_workspace(tmp_path, monkeypatch)
    existing = workspace / "existing.json"
    existing.write_text("original\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="absent at startup"):
        supervisor._write_json_atomic_exclusive(existing, {"status": "PASS"})
    assert existing.read_text(encoding="ascii") == "original\n"

    blocked = workspace / "blocked.json"
    lock = workspace / f".{blocked.name}.publish.lock"
    lock.mkdir()
    with pytest.raises(FileExistsError):
        supervisor._write_json_atomic_exclusive(blocked, {"status": "PASS"})
    assert lock.is_dir()
    assert not blocked.exists()
    assert list(workspace.glob(f".{blocked.name}.*.tmp")) == []


def test_revalidation_freezes_exact_acceptance_bytes_before_running_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _persistent_workspace(tmp_path, monkeypatch)
    acceptance_root = workspace / "acceptance"
    acceptance_root.mkdir()
    source_payloads: dict[str, bytes] = {}
    source_paths: dict[str, Path] = {}
    for key in supervisor._EXPECTED_EVIDENCE_KEYS:
        payload = (json.dumps({"key": key}, sort_keys=True) + "\n").encode("ascii")
        path = acceptance_root / f"{key}.json"
        path.write_bytes(payload)
        source_payloads[key] = payload
        source_paths[key] = path
    original_report = {
        "schema": supervisor.ACCEPTANCE_SCHEMA,
        "status": "PASS",
        "failures": [],
        "evidence": {
            key: {
                "path": str(source_paths[key]),
                "sha256": hashlib.sha256(source_payloads[key]).hexdigest(),
            }
            for key in supervisor._EXPECTED_EVIDENCE_KEYS
        },
    }
    acceptance_report = acceptance_root / "ltop_g3_source_aligned_acceptance.json"
    original_acceptance_payload = (
        json.dumps(original_report, allow_nan=False, sort_keys=True) + "\n"
    ).encode("ascii")
    acceptance_report.write_bytes(original_acceptance_payload)

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    semantic_tool = repository_root / "semantic.py"
    action_tool = repository_root / "action.py"
    semantic_tool.write_text("pass\n", encoding="ascii")
    action_tool.write_text("pass\n", encoding="ascii")
    snapshot_root = workspace / "snapshot"
    semantic_output = workspace / "semantic.json"
    action_output = workspace / "action.json"
    mutated = False

    def fake_run_checked(
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
    ) -> None:
        nonlocal mutated
        assert cwd == repository_root
        assert environment == {}
        arguments = list(command)
        output = Path(arguments[arguments.index("--output") + 1])
        if "--training-report" in arguments:
            snapshot_paths = {
                "training_report": Path(arguments[arguments.index("--training-report") + 1]),
                "arm_validation": Path(arguments[arguments.index("--arm-validation") + 1]),
                "cold_action_factual": Path(
                    arguments[arguments.index("--factual-action-report") + 1]
                ),
                "cold_action_mediator_required": Path(
                    arguments[arguments.index("--mediator-action-report") + 1]
                ),
                "cold_retention": Path(arguments[arguments.index("--retention-report") + 1]),
            }
            assert all(path.parent == snapshot_root for path in snapshot_paths.values())
            acceptance_report.write_text('{"replaced":true}\n', encoding="ascii")
            source_paths["cold_action_factual"].write_text(
                '{"replaced":true}\n',
                encoding="ascii",
            )
            mutated = True
            recomposed = json.loads(json.dumps(original_report))
            for key, path in snapshot_paths.items():
                recomposed["evidence"][key]["path"] = str(path)
            supervisor._write_json_atomic_exclusive(output, recomposed)
            return
        factual = Path(arguments[arguments.index("--factual-report") + 1])
        mediator = Path(arguments[arguments.index("--mediator-required-report") + 1])
        supervisor._write_json_atomic_exclusive(
            output,
            {
                "schema": supervisor.ACTION_VALIDATION_SCHEMA,
                "status": "PASS",
                "failures": [],
                "inputs": {
                    "factual": {"path": str(factual), "sha256": supervisor._sha256(factual)},
                    "mediator_required": {
                        "path": str(mediator),
                        "sha256": supervisor._sha256(mediator),
                    },
                },
            },
        )

    monkeypatch.setattr(supervisor, "_run_checked", fake_run_checked)
    result = supervisor.revalidate_acceptance(
        acceptance_report=acceptance_report,
        semantic_validator=semantic_tool,
        action_validator=action_tool,
        semantic_output=semantic_output,
        action_output=action_output,
        snapshot_root=snapshot_root,
        repository_root=repository_root,
        environment={},
    )

    assert mutated is True
    assert (
        result["original_acceptance_report_sha256"]
        == hashlib.sha256(original_acceptance_payload).hexdigest()
    )
    assert result["acceptance_report"] == str(semantic_output)
    assert result["acceptance_report_sha256"] == supervisor._sha256(semantic_output)
    assert (snapshot_root / "original-acceptance.json").read_bytes() == original_acceptance_payload
    for key, payload in source_payloads.items():
        frozen_path = snapshot_root / supervisor._EVIDENCE_SNAPSHOT_FILENAMES[key]
        assert frozen_path.read_bytes() == payload


def test_repository_receipt_binds_exact_clean_commit_and_tool_hash(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "ADR170 Test"],
        check=True,
    )
    tool = repository / "tool.py"
    tool.write_text("pass\n", encoding="ascii")
    subprocess.run(["git", "-C", str(repository), "add", "tool.py"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    receipt = supervisor._repository_source_receipt(
        repository_root=repository.resolve(),
        expected_commit=commit,
        source_files={"tool": tool.resolve()},
    )
    assert receipt["actual_commit"] == commit
    assert receipt["expected_commit"] == commit
    assert receipt["worktree_clean"] is True
    assert receipt["files"]["tool"]["repository_relative_path"] == "tool.py"
    assert receipt["files"]["tool"]["sha256"] == supervisor._sha256(tool)

    with pytest.raises(ValueError, match="operator-approved commit"):
        supervisor._repository_source_receipt(
            repository_root=repository.resolve(),
            expected_commit="0" * 40,
            source_files={"tool": tool.resolve()},
        )
    tool.write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="worktree is not exactly clean"):
        supervisor._repository_source_receipt(
            repository_root=repository.resolve(),
            expected_commit=commit,
            source_files={"tool": tool.resolve()},
        )


def test_process_identity_reads_current_linux_start_tick() -> None:
    assert supervisor.process_start_ticks(os.getpid()) > 0
    assert supervisor.linux_boot_id()


def test_repository_file_rejects_symbolic_links(tmp_path: Path) -> None:
    direct = tmp_path / "direct.py"
    direct.write_text("pass\n", encoding="ascii")
    indirect = tmp_path / "indirect.py"
    indirect.symlink_to(direct)

    assert supervisor._require_repository_file(direct, name="direct") == direct
    with pytest.raises(ValueError, match="direct regular file"):
        supervisor._require_repository_file(indirect, name="indirect")


def test_launcher_is_valid_bash() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = root / "adr170/run_ltop_g3_source_aligned_post_acceptance_2gpu.sh"
    completed = subprocess.run(
        ["bash", "-n", str(launcher)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
