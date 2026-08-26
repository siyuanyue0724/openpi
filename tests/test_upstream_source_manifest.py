from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audit_upstream_sources import audit_manifest, validate_evidence_symbols


def test_repository_upstream_manifest_is_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    result = audit_manifest(
        root / "references/upstream_sources.json",
        root=root,
        check_checkouts=False,
    )
    assert result["sources"] >= 23
    assert result["adapted_sources"] >= 7


def test_manifest_rejects_abbreviated_commit(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema": "picf-next.upstream-sources.v1",
        "sources": [
            {
                "name": "bad",
                "url": "https://github.com/example/bad.git",
                "commit": "1234",
                "license": "MIT",
                "checkout": "missing",
                "status": "audit-only",
                "symbols": ["module::symbol"],
                "local_paths": [],
            }
        ],
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="full lowercase commit"):
        audit_manifest(path, root=root, check_checkouts=False)


def test_python_evidence_requires_the_exact_qualified_symbol(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "class Actual:\n"
        "    def step(self):\n"
        "        def nested():\n"
        "            return None\n"
        "        return nested()\n"
        "\n"
        "def top_level():\n"
        "    return None\n"
    )

    validate_evidence_symbols(
        tmp_path,
        [
            "module.py::Actual.step",
            "module.py::Actual.step.nested",
            "module.py::top_level",
        ],
        source_name="fixture",
    )
    with pytest.raises(ValueError, match="Wrong.step"):
        validate_evidence_symbols(
            tmp_path,
            ["module.py::Wrong.step"],
            source_name="fixture",
        )


def test_python_evidence_supports_pep695_definitions_on_older_auditors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generic_module.py"
    source.write_text(
        "class GenericBox[T]:\n    def map[U](self, value: U) -> U:\n        return value\n"
    )

    validate_evidence_symbols(
        tmp_path,
        [
            "generic_module.py::GenericBox",
            "generic_module.py::GenericBox.map",
        ],
        source_name="pep695-fixture",
    )


def test_python_evidence_rejects_unparseable_source(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def broken(:\n")
    with pytest.raises(ValueError, match="cannot parse attributed Python source"):
        validate_evidence_symbols(
            tmp_path,
            ["broken.py::broken"],
            source_name="fixture",
        )
