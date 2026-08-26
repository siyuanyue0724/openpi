from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.build_lingbot_calvin_norm_stats as norm_tool
from tools.build_lingbot_calvin_norm_stats import _write_atomic


def test_norm_stats_publication_is_durable_and_no_replace(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "norm.json"
    payload: dict[str, object] = {"schema": 1, "values": [1, 2, 3]}

    _write_atomic(payload, destination)

    assert json.loads(destination.read_text(encoding="ascii")) == payload
    assert list(destination.parent.glob(".*.incomplete-*")) == []
    with pytest.raises(FileExistsError):
        _write_atomic({"schema": 2}, destination)
    assert json.loads(destination.read_text(encoding="ascii")) == payload


def test_norm_stats_publication_rejects_symlink_destination(tmp_path: Path) -> None:
    protected = tmp_path / "protected.json"
    protected.write_text("protected\n", encoding="ascii")
    destination = tmp_path / "norm.json"
    destination.symlink_to(protected)

    with pytest.raises(FileExistsError):
        _write_atomic({"schema": 1}, destination)

    assert protected.read_text(encoding="ascii") == "protected\n"
    assert destination.is_symlink()


def test_norm_stats_publication_rejects_active_publisher_lock(tmp_path: Path) -> None:
    destination = tmp_path / "norm.json"
    lock = tmp_path / ".norm.json.publish-lock"
    lock.mkdir()

    with pytest.raises(FileExistsError):
        _write_atomic({"winner": False}, destination)

    assert not destination.exists()
    assert lock.is_dir()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_norm_stats_cli_binds_the_dataset_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    normalization = tmp_path / "normalization.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "lingbot.json"
    normalization.write_text("{}", encoding="ascii")
    manifest.write_text("{}", encoding="ascii")
    tree = "d" * 64
    source = {
        "dataset_id": "calvin",
        "dataset_revision": "revision",
        "dataset_tree_sha256": tree,
    }
    translated = {"schema": "translated", "dataset_tree_sha256": tree}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        norm_tool,
        "load_calvin_normalization_artifact",
        lambda path: source if path == normalization.resolve() else pytest.fail("wrong source"),
    )
    monkeypatch.setattr(
        norm_tool,
        "load_dataset_file_manifest",
        lambda path: (
            SimpleNamespace(
                dataset_id="calvin",
                dataset_revision="revision",
                tree_sha256=tree,
            )
            if path == manifest.resolve()
            else pytest.fail("wrong manifest")
        ),
    )

    def translate(payload: object, *, dataset_tree_sha256: str) -> dict[str, object]:
        assert payload is source
        assert dataset_tree_sha256 == tree
        return translated

    monkeypatch.setattr(norm_tool, "official_lingbot_calvin_norm_stats", translate)
    monkeypatch.setattr(
        norm_tool,
        "_write_atomic",
        lambda payload, destination: captured.update(
            payload=payload,
            destination=destination,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_lingbot_calvin_norm_stats.py",
            "--calvin-normalization",
            str(normalization),
            "--dataset-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    norm_tool.main()

    assert captured == {"payload": translated, "destination": output}
    assert json.loads(capsys.readouterr().out) == translated
