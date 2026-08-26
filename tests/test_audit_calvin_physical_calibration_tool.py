from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import tools.audit_calvin_physical_calibration as audit_tool
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.dataset_manifest import DatasetFileManifest


def test_load_bound_index_uses_verified_reads_without_full_tree_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_root = tmp_path / "training"
    split_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    args = argparse.Namespace(
        split_root=split_root,
        dataset_manifest=manifest_path,
        dataset_id="calvin-test",
        dataset_revision="revision-test",
    )
    manifest = cast(DatasetFileManifest, SimpleNamespace())
    index = cast(CalvinDatasetIndex, SimpleNamespace())
    binding = {"dataset_runtime_verified_read_required": True}
    bindings: list[dict[str, object]] = []
    loads: list[dict[str, object]] = []

    monkeypatch.setattr(
        audit_tool,
        "load_dataset_file_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        audit_tool,
        "validate_dataset_runtime_binding",
        lambda *positional, **keywords: (
            bindings.append({"positional": positional, **keywords}) or binding
        ),
    )
    monkeypatch.setattr(
        audit_tool.CalvinDatasetIndex,
        "load",
        lambda *positional, **keywords: (
            loads.append({"positional": positional, **keywords}) or index
        ),
    )

    actual_root, actual_manifest, actual_index, actual_binding = audit_tool._load_bound_index(args)

    assert actual_root == split_root.resolve()
    assert actual_manifest is manifest
    assert actual_index is index
    assert actual_binding is binding
    assert bindings == [
        {
            "positional": (manifest, split_root.resolve()),
            "dataset_id": "calvin-test",
            "dataset_revision": "revision-test",
            "split_name": "training",
        }
    ]
    assert loads == [
        {
            "positional": (split_root.resolve(),),
            "dataset_id": "calvin-test",
            "dataset_revision": "revision-test",
            "dataset_manifest": manifest,
            "verify_files": False,
        }
    ]
