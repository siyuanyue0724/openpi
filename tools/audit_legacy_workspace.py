#!/usr/bin/env python3
"""Build an immutable mechanical inventory of the dirty legacy workspace."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

_CORE_PATHS = (
    "src/openpi/picf/core/config.py",
    "src/openpi/picf/core/contracts.py",
    "src/openpi/picf/core/pipeline.py",
    "src/openpi/picf/core/training.py",
    "src/openpi/picf/policy.py",
    "src/openpi/picf/paligemma/wrapper.py",
    "scripts/picf_core_train.py",
)
_EXPERIMENT = re.compile(r"\bG[0-9]+[A-Za-z0-9_-]*\b")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_record(path: Path, root: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    headings = [line.strip() for line in text.splitlines() if line.startswith("#")]
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "lines": len(text.splitlines()),
        "headings": headings,
        "experiment_ids": sorted(set(_EXPERIMENT.findall(text))),
    }


def _python_record(path: Path, root: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    tree = ast.parse(text, filename=str(path))
    symbols = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "lines": len(text.splitlines()),
        "top_level_symbols": symbols,
    }


def _git_status(root: Path) -> dict[str, str]:
    output = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line[3:]: line[:2] for line in output.splitlines() if len(line) >= 4}


def _readme_paths(root: Path) -> list[Path]:
    excluded = {
        ".cache",
        ".git",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "data",
        "node_modules",
        "outputs",
        "temp",
    }
    paths = []
    for directory, names, files in os.walk(root):
        names[:] = [name for name in names if name not in excluded]
        paths.extend(Path(directory) / name for name in files if "readme" in name.lower())
    return sorted(paths)


def audit_workspace(root: Path) -> dict[str, object]:
    root = root.resolve()
    if not (root / ".git").exists():
        raise ValueError(f"legacy workspace is not a Git repository: {root}")
    status = _git_status(root)
    readmes = _readme_paths(root)
    documents = sorted((root / "docs").rglob("*.md"))
    temp_documents = sorted((root / "temp").glob("*.md")) if (root / "temp").is_dir() else []
    core = []
    for relative in _CORE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        record = _python_record(path, root)
        record["git_status"] = status.get(relative, "  ")
        core.append(record)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema": "picf-next.legacy-workspace-inventory.v1",
        "workspace": str(root),
        "git_head": head,
        "counts": {
            "readme_like": len(readmes),
            "docs_markdown_present": len(documents),
            "temp_markdown_present": len(temp_documents),
            "git_status_entries": len(status),
            "tracked_deleted": sum(value == " D" for value in status.values()),
        },
        "readmes": [_text_record(path, root) for path in readmes],
        "documents": [_text_record(path, root) for path in documents],
        "temp_documents": [_text_record(path, root) for path in temp_documents],
        "core_python": core,
    }


def main() -> None:
    args = _parse_args()
    report = audit_workspace(args.workspace)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
