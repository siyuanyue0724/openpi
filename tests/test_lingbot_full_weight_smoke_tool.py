from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.smoke_lingbot_vla2_full_weight import (
    TARGET_ONLY_FIELDS,
    _asset_manifest,
    _merge_training_sections,
    _resolve_training_config,
)


def test_lingbot_full_weight_tool_delays_accelerator_imports() -> None:
    path = Path(__file__).resolve().parents[1] / "tools/smoke_lingbot_vla2_full_weight.py"
    tree = ast.parse(path.read_text())
    top_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in top_imports
    assert "transformers" not in top_imports
    assert "lingbotvla" not in top_imports


def test_training_config_resolution_is_non_mutating_and_pins_external_assets(
    tmp_path: Path,
) -> None:
    training = {
        "model": {"tokenizer_path": "/stale", "post_training": True},
        "data": {"joints": [{"arm.position": 14}]},
        "train": {
            "use_compile": True,
            "attention_implementation": "flex_cached",
            "align_params": {
                "depth": {"model_type": "MoRGBD", "morgbd_path": "/stale"},
                "video": {"ckpt_path": "/stale", "config_path": "/stale"},
            },
        },
    }
    before = repr(training)
    merged, data = _resolve_training_config(
        training,
        checkpoint_dir=tmp_path / "checkpoint",
        processor_dir=tmp_path / "processor",
        num_steps=3,
    )

    assert repr(training) == before
    assert merged["tokenizer_path"] == str((tmp_path / "processor").resolve())
    assert merged["use_compile"] is False
    assert merged["attention_implementation"] == "eager"
    assert merged["vit_attn_implementation"] == "eager"
    assert merged["num_steps"] == 3
    assert merged["align_params"]["depth"]["morgbd_path"].endswith("depth/model.pt")
    assert merged["align_params"]["video"]["ckpt_path"].endswith(
        "dino_video/teacher_step_10000.pth"
    )
    assert data == training["data"]


def test_training_config_resolution_rejects_partial_or_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model and train"):
        _merge_training_sections({"model": {}})
    with pytest.raises(ValueError, match="num-steps"):
        _resolve_training_config(
            {"model": {}, "train": {}, "data": {}},
            checkpoint_dir=tmp_path,
            processor_dir=tmp_path,
            num_steps=0,
        )


def test_asset_manifest_hashes_exact_required_files(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"alpha")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b").write_bytes(b"beta")
    manifest = _asset_manifest(tmp_path, ("nested/b", "a"))
    assert [entry["path"] for entry in manifest] == ["a", "nested/b"]
    assert all(len(entry["sha256"]) == 64 for entry in manifest)
    with pytest.raises(FileNotFoundError):
        _asset_manifest(tmp_path, ("missing",))


def test_g0_input_contract_explicitly_forbids_training_targets() -> None:
    assert {"actions", "mask", "object_id", "teacher", "targets"} <= TARGET_ONLY_FIELDS
