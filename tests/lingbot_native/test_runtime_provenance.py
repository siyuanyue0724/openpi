from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.lingbot_native.runtime_provenance import (
    python_source_tree_contract,
    revision_bound_python_source_tree_contract,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    source = repo / "src"
    source.mkdir(parents=True)
    (source / "runtime.py").write_text("VALUE = 1\n")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "src/runtime.py")
    _git(repo, "commit", "-m", "initial")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_revision_bound_tree_accepts_exact_commit_and_changes_with_bytes(tmp_path: Path) -> None:
    repo, revision = _repository(tmp_path)
    roots = {"src": repo / "src"}

    first = revision_bound_python_source_tree_contract(
        repo_root=repo,
        revision=revision,
        roots=roots,
    )
    assert first == python_source_tree_contract(roots)

    (repo / "src/runtime.py").write_text("VALUE = 2\n")
    with pytest.raises(ContractError, match="tracked runtime Python"):
        revision_bound_python_source_tree_contract(
            repo_root=repo,
            revision=revision,
            roots=roots,
        )


def test_revision_bound_tree_rejects_untracked_or_ignored_python(tmp_path: Path) -> None:
    repo, revision = _repository(tmp_path)
    roots = {"src": repo / "src"}
    (repo / "src/extra.py").write_text("VALUE = 2\n")
    with pytest.raises(ContractError, match="untracked runtime Python"):
        revision_bound_python_source_tree_contract(
            repo_root=repo,
            revision=revision,
            roots=roots,
        )

    (repo / "src/extra.py").unlink()
    (repo / ".gitignore").write_text("ignored.py\n")
    (repo / "src/ignored.py").write_text("VALUE = 3\n")
    with pytest.raises(ContractError, match="untracked runtime Python"):
        revision_bound_python_source_tree_contract(
            repo_root=repo,
            revision=revision,
            roots=roots,
        )
