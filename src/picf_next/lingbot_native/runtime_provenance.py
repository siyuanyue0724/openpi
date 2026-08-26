"""Fail-closed provenance contracts for PICF/LingBot Python runtime trees."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

from picf_next.contracts import ContractError

ADR127_LINGBOT_RUNTIME_PYTHON_TREE = {
    "file_count": 244,
    "tree_sha256": "8d21f35c01adf362b409c91f0126b41c783783f20aa0c0d96acf8c927a0c8dc8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_source_tree_contract(roots: Mapping[str, Path]) -> dict[str, object]:
    """Hash every runtime Python source file, including intentional overlays."""

    entries = []
    for namespace, root in sorted(roots.items()):
        if not namespace or not root.is_dir() or root.is_symlink():
            raise ContractError("runtime Python source root is invalid")
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if not path.is_file() or path.is_symlink():
                raise ContractError("runtime Python source file is invalid")
            entries.append(
                {
                    "path": f"{namespace}/{path.relative_to(root).as_posix()}",
                    "sha256": _sha256(path),
                }
            )
    if not entries or len({entry["path"] for entry in entries}) != len(entries):
        raise ContractError("runtime Python source tree is empty or ambiguous")
    canonical = json.dumps(entries, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return {
        "file_count": len(entries),
        "tree_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def revision_bound_python_source_tree_contract(
    *,
    repo_root: Path,
    revision: str,
    roots: Mapping[str, Path],
) -> dict[str, object]:
    """Require the runtime Python files to be exactly those of ``revision``."""

    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ContractError("runtime Python revision must be one lowercase Git commit")
    resolved_repo = repo_root.resolve()
    if not resolved_repo.is_dir() or resolved_repo.is_symlink():
        raise ContractError("runtime Python Git root is invalid")

    def run_git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(resolved_repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    top_level = Path(run_git("rev-parse", "--show-toplevel").strip()).resolve()
    if top_level != resolved_repo:
        raise ContractError("runtime Python Git root is not repository top level")
    if run_git("rev-parse", "HEAD").strip() != revision:
        raise ContractError("runtime Python checkout differs from its declared revision")

    pathspecs = []
    for root in roots.values():
        resolved = root.resolve()
        if not resolved.is_relative_to(resolved_repo):
            raise ContractError("runtime Python source root escapes its Git checkout")
        pathspecs.append(resolved.relative_to(resolved_repo).as_posix())

    changed = run_git("diff", "--name-only", revision, "--", *pathspecs).splitlines()
    if any(Path(value).suffix == ".py" for value in changed):
        raise ContractError("tracked runtime Python source differs from the declared revision")
    untracked = run_git("ls-files", "--others", "--exclude-standard", "--", *pathspecs).splitlines()
    ignored = run_git(
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        *pathspecs,
    ).splitlines()
    if any(Path(value).suffix == ".py" for value in (*untracked, *ignored)):
        raise ContractError("untracked runtime Python source is not revision-bound")
    return python_source_tree_contract(roots)


def validate_adr127_lingbot_runtime_tree(source_checkout: Path) -> dict[str, object]:
    contract = python_source_tree_contract({"lingbotvla": source_checkout / "lingbotvla"})
    if contract != ADR127_LINGBOT_RUNTIME_PYTHON_TREE:
        raise ContractError("ADR-127 LingBot runtime Python tree differs from preregistration")
    return contract


def adr127_runtime_python_trees_contract(
    *,
    repo_root: Path,
    revision: str,
    source_checkout: Path,
) -> dict[str, dict[str, object]]:
    """Bind both runtime source trees at one execution boundary."""

    return {
        "lingbot": validate_adr127_lingbot_runtime_tree(source_checkout),
        "picf": revision_bound_python_source_tree_contract(
            repo_root=repo_root,
            revision=revision,
            roots={"src": repo_root / "src", "tools": repo_root / "tools"},
        ),
    }
