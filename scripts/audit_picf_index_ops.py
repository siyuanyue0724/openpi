from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "scripts/picf_core_train.py",
    ROOT / "scripts/picf_replay_windows.py",
    ROOT / "src/openpi/picf/core/pipeline.py",
    ROOT / "src/openpi/models/sonata_encoder.py",
    ROOT / "src/openpi/picf/sonata/wrapper.py",
    ROOT / "src/openpi/picf/vjepa/vendor/masks_utils.py",
    ROOT / "src/openpi/picf/anytouch/wrapper.py",
    ROOT / "src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py",
    ROOT / "src/openpi/models_pytorch/transformers_replace/models/paligemma/safe_ops.py",
]
CALL_NAMES = {"gather", "index_select", "masked_scatter", "scatter_", "scatter", "take_along_dim"}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    col: int
    kind: str
    snippet: str


def _call_name(node: ast.Call) -> str | None:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def _iter_findings(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            snippet = lines[node.lineno - 1].strip()
            findings.append(Finding(path=path, line=node.lineno, col=node.col_offset, kind="subscript", snippet=snippet))
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in CALL_NAMES:
                snippet = lines[node.lineno - 1].strip()
                findings.append(Finding(path=path, line=node.lineno, col=node.col_offset, kind=f"call:{name}", snippet=snippet))
    return sorted(findings, key=lambda item: (str(item.path), item.line, item.col, item.kind))


def main() -> None:
    all_findings: list[Finding] = []
    for path in TARGETS:
        if path.is_file():
            all_findings.extend(_iter_findings(path))
    for finding in all_findings:
        rel = finding.path.relative_to(ROOT)
        print(f"{rel}:{finding.line}:{finding.col}: {finding.kind}: {finding.snippet}")
    print(f"\nTotal findings: {len(all_findings)}")


if __name__ == "__main__":
    main()
