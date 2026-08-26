from __future__ import annotations

import subprocess
from pathlib import Path

from tools.audit_legacy_workspace import audit_workspace


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_legacy_inventory_separates_readme_docs_temp_and_core(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    _write(tmp_path / "README.md", "# Root\nG12\n")
    _write(tmp_path / "docs" / "design.md", "# Design\nG99a\n")
    _write(tmp_path / "temp" / "run.md", "# Run\nG7\n")
    for relative in (
        "src/openpi/picf/core/config.py",
        "src/openpi/picf/core/contracts.py",
        "src/openpi/picf/core/pipeline.py",
        "src/openpi/picf/core/training.py",
        "src/openpi/picf/policy.py",
        "src/openpi/picf/paligemma/wrapper.py",
        "scripts/picf_core_train.py",
    ):
        _write(tmp_path / relative, "def entry():\n    return 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    report = audit_workspace(tmp_path)

    assert report["counts"] == {
        "readme_like": 1,
        "docs_markdown_present": 1,
        "temp_markdown_present": 1,
        "git_status_entries": 0,
        "tracked_deleted": 0,
    }
    assert report["readmes"][0]["experiment_ids"] == ["G12"]
    assert report["documents"][0]["experiment_ids"] == ["G99a"]
    assert len(report["core_python"]) == 7
    assert report["core_python"][0]["top_level_symbols"] == ["entry"]
