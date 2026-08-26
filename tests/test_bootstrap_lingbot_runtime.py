from __future__ import annotations

from pathlib import Path

from tools.bootstrap_lingbot_runtime import (
    RUNTIME_VERSIONS,
    _run_json,
    _write_text_durable,
    runtime_install_commands,
)


def test_runtime_install_order_is_pinned_and_keeps_lerobot_dependencies_isolated(
    tmp_path: Path,
) -> None:
    python = tmp_path / "python"
    source = tmp_path / "source"
    commands = runtime_install_commands(
        python=python,
        source_checkout=source,
        uv_command="uv",
    )
    assert commands[0][-2:] == ["-r", str(source / "requirements.txt")]
    assert commands[1][-1] == f"datasets=={RUNTIME_VERSIONS['datasets']}"
    assert "--no-deps" in commands[2]
    assert commands[2][-1] == f"lerobot=={RUNTIME_VERSIONS['lerobot']}"
    assert all(str(python) in command for command in commands)


def test_runtime_probe_preserves_virtual_environment_launcher(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.write_text('#!/bin/sh\nprintf \'{"launcher": "venv"}\\n\'\n')
    launcher.chmod(0o755)

    assert _run_json(launcher, source, "ignored") == {"launcher": "venv"}


def test_runtime_report_publication_is_atomic(tmp_path: Path) -> None:
    report = tmp_path / "nested" / "runtime.json"
    _write_text_durable(report, '{"status":"PASS"}\n')
    assert report.read_text() == '{"status":"PASS"}\n'
    assert not tuple(report.parent.glob("*.tmp"))
