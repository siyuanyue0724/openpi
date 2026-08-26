#!/usr/bin/env python3
"""Validate immutable upstream attribution without network access."""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import tokenize
import warnings
from pathlib import Path

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STATUSES = {
    "api-consumer",
    "audit-only",
    "principle-only",
    "semantic-adaptation",
    "substantial-adaptation",
}


def _without_pep695_definition_parameters(source: str) -> str:
    """Erase only PEP 695 class/function type parameters for older parsers.

    The replacement preserves lines and columns and leaves the full executable
    body to ``ast.parse``. It is deliberately narrower than a token-only symbol
    fallback, which could accidentally accept otherwise invalid Python.
    """

    try:
        tokens = tuple(tokenize.generate_tokens(io.StringIO(source).readline))
    except (IndentationError, tokenize.TokenError):
        return source
    ranges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    expect_definition_name = False
    definition_name_seen = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.NAME and token.string in {"class", "def"}:
            expect_definition_name = True
            definition_name_seen = False
            index += 1
            continue
        if expect_definition_name:
            if token.type == tokenize.NAME:
                expect_definition_name = False
                definition_name_seen = True
            elif token.type not in {tokenize.NL, tokenize.COMMENT}:
                expect_definition_name = False
            index += 1
            continue
        if definition_name_seen:
            if token.type in {tokenize.NL, tokenize.COMMENT}:
                index += 1
                continue
            definition_name_seen = False
            if token.type == tokenize.OP and token.string == "[":
                depth = 1
                closing = index + 1
                while closing < len(tokens) and depth:
                    candidate = tokens[closing]
                    if candidate.type == tokenize.OP:
                        if candidate.string == "[":
                            depth += 1
                        elif candidate.string == "]":
                            depth -= 1
                    closing += 1
                if depth:
                    return source
                ranges.append((token.start, tokens[closing - 1].end))
                index = closing
                continue
        index += 1
    if not ranges:
        return source

    line_offsets: list[int] = []
    cursor = 0
    for line in source.splitlines(keepends=True):
        line_offsets.append(cursor)
        cursor += len(line)
    characters = list(source)
    for start, stop in ranges:
        start_offset = line_offsets[start[0] - 1] + start[1]
        stop_offset = line_offsets[stop[0] - 1] + stop[1]
        for offset in range(start_offset, stop_offset):
            if characters[offset] not in {"\n", "\r"}:
                characters[offset] = " "
    return "".join(characters)


def _python_qualified_symbols(source_path: Path) -> set[str]:
    try:
        source = source_path.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(
                source,
                filename=str(source_path),
            )
    except UnicodeDecodeError as error:
        raise ValueError(f"cannot parse attributed Python source {source_path}") from error
    except SyntaxError as first_error:
        compatible_source = _without_pep695_definition_parameters(source)
        if compatible_source == source:
            raise ValueError(
                f"cannot parse attributed Python source {source_path}"
            ) from first_error
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(compatible_source, filename=str(source_path))
        except SyntaxError as error:
            raise ValueError(f"cannot parse attributed Python source {source_path}") from error

    class QualifiedSymbolCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []
            self.symbols: set[str] = set()

        def _visit_scope(self, node: ast.AST, name: str) -> None:
            self.scope.append(name)
            self.symbols.add(".".join(self.scope))
            self.generic_visit(node)
            self.scope.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self._visit_scope(node, node.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_scope(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self._visit_scope(node, node.name)

    collector = QualifiedSymbolCollector()
    collector.visit(tree)
    return collector.symbols


def validate_evidence_symbols(checkout: Path, symbols: list[str], *, source_name: str) -> None:
    """Require every attributed file and terminal symbol to exist at the pinned checkout."""
    for specification in symbols:
        if specification.count("::") != 1:
            raise ValueError(f"source {source_name} has malformed evidence {specification!r}")
        relative_path, qualified_symbol = specification.split("::")
        source_path = checkout / relative_path
        if not source_path.is_file():
            raise ValueError(
                f"source {source_name} references missing evidence file {relative_path}"
            )
        if source_path.suffix == ".py":
            symbol_exists = qualified_symbol in _python_qualified_symbols(source_path)
        else:
            terminal_symbol = qualified_symbol.rsplit(".", 1)[-1]
            symbol_exists = bool(
                terminal_symbol
                and re.search(
                    rf"\b{re.escape(terminal_symbol)}\b",
                    source_path.read_text(errors="replace"),
                )
            )
        if not symbol_exists:
            raise ValueError(
                f"source {source_name} evidence symbol {qualified_symbol} "
                f"is absent from {relative_path}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("references/upstream_sources.json"),
    )
    parser.add_argument("--check-checkouts", action="store_true")
    return parser.parse_args()


def audit_manifest(path: Path, *, root: Path, check_checkouts: bool) -> dict[str, int]:
    data = json.loads(path.read_text())
    if data.get("schema") != "picf-next.upstream-sources.v1":
        raise ValueError("unsupported upstream source manifest schema")
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("upstream source manifest must contain sources")
    names: set[str] = set()
    urls: set[str] = set()
    adapted = 0
    for source in sources:
        name = source.get("name")
        url = source.get("url")
        commit = source.get("commit")
        status = source.get("status")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"invalid or duplicate source name {name!r}")
        if not isinstance(url, str) or not url.startswith("https://github.com/") or url in urls:
            raise ValueError(f"invalid or duplicate official source URL for {name}")
        if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
            raise ValueError(f"source {name} must pin a full lowercase commit")
        if status not in _STATUSES:
            raise ValueError(f"source {name} has unsupported status {status!r}")
        if not isinstance(source.get("license"), str) or not source["license"]:
            raise ValueError(f"source {name} requires an explicit license status")
        symbols = source.get("symbols")
        if not isinstance(symbols, list) or not symbols or not all(symbols):
            raise ValueError(f"source {name} requires exact source symbols")
        local_paths = source.get("local_paths")
        if not isinstance(local_paths, list):
            raise ValueError(f"source {name} local_paths must be a list")
        if status in {"semantic-adaptation", "substantial-adaptation", "api-consumer"}:
            adapted += 1
            if not local_paths:
                raise ValueError(f"adapted source {name} requires a local implementation path")
        for local_path in local_paths:
            if not (root / local_path).is_file():
                raise ValueError(f"source {name} references missing local path {local_path}")
        if check_checkouts:
            checkout = root / source["checkout"]
            if not (checkout / ".git").exists():
                raise ValueError(f"source {name} checkout is absent: {checkout}")
            actual = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if actual != commit:
                raise ValueError(f"source {name} checkout {actual} differs from {commit}")
            validate_evidence_symbols(checkout, symbols, source_name=name)
        names.add(name)
        urls.add(url)
    return {"sources": len(sources), "adapted_sources": adapted}


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    result = audit_manifest(manifest, root=root, check_checkouts=args.check_checkouts)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
