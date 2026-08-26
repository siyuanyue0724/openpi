from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import build_dataset_file_manifest
from picf_next.data.lingbot_calvin_projection import (
    LINGBOT_CALVIN_PROJECTION_SCHEMA,
    load_lingbot_calvin_projection_contract,
    processor_assets_sha256,
    projection_payload_sha256,
    validate_lingbot_calvin_projection_payload,
)
from tools import probe_lingbot_calvin_projection as projection_probe

_DATASET_MANIFEST_SHA256 = "a" * 64


def _payload() -> dict[str, object]:
    view = {
        "source_field": "rgb_static",
        "source_shape": [200, 200, 3],
        "image_grid_thw": [1, 16, 16],
        "merged_grid_hw": [8, 8],
        "raw_patch_count": 256,
        "merged_token_count": 64,
        "pixel_values_shape": [256, 1536],
        "source_rgb_sha256": ["1" * 64, "2" * 64, "3" * 64],
    }
    gripper = dict(view)
    gripper["source_field"] = "rgb_gripper"
    gripper["source_shape"] = [84, 84, 3]
    gripper["source_rgb_sha256"] = ["4" * 64, "5" * 64, "6" * 64]
    return {
        "schema": LINGBOT_CALVIN_PROJECTION_SCHEMA,
        "status": "PASS",
        "runtime_input": False,
        "processor_id": "Qwen/Qwen3-VL-4B-Instruct",
        "processor_revision": "b" * 40,
        "processor_assets_sha256": "c" * 64,
        "processor_config_sha256": "d" * 64,
        "processor_preprocessor_config_sha256": "e" * 64,
        "dataset_manifest_sha256": _DATASET_MANIFEST_SHA256,
        "dataset_tree_sha256": "f" * 64,
        "source_frame_count": 10,
        "sample_global_indices": [0, 5, 9],
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "views": {"static": view, "gripper": gripper},
        "transformers_version": "5.0.0",
    }


def test_projection_contract_validates_exact_qwen_geometry() -> None:
    payload = _payload()

    validated = validate_lingbot_calvin_projection_payload(
        payload,
        expected_dataset_manifest_sha256=_DATASET_MANIFEST_SHA256,
    )

    assert validated == payload
    assert validated["views"]["static"]["merged_token_count"] == 64
    assert len(projection_payload_sha256(validated)) == 64


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("views", "static", "merged_grid_hw"), [27, 27], "merged grid"),
        (("views", "static", "pixel_values_shape"), [256, 768], "pixel-values shape"),
        (("sample_global_indices",), [0, 5, 10], "in-range"),
        (("runtime_input",), True, "did not pass"),
    ),
)
def test_projection_contract_rejects_geometry_or_semantic_drift(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = _payload()
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ContractError, match=message):
        validate_lingbot_calvin_projection_payload(payload)


def test_projection_contract_rejects_another_dataset_manifest() -> None:
    with pytest.raises(ContractError, match="another dataset manifest"):
        validate_lingbot_calvin_projection_payload(
            _payload(),
            expected_dataset_manifest_sha256="0" * 64,
        )


def test_projection_contract_load_is_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(_payload(), sort_keys=True), encoding="ascii")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_lingbot_calvin_projection_contract(
        path,
        expected_sha256=digest,
        expected_dataset_manifest_sha256=_DATASET_MANIFEST_SHA256,
    )

    assert loaded == _payload()
    with pytest.raises(ContractError, match="expected SHA-256"):
        load_lingbot_calvin_projection_contract(
            path,
            expected_sha256="0" * 64,
            expected_dataset_manifest_sha256=_DATASET_MANIFEST_SHA256,
        )


def test_processor_asset_digest_is_ordered_and_exact() -> None:
    assets = [
        {"path": "config.json", "bytes": 3, "sha256": "1" * 64},
        {"path": "preprocessor_config.json", "bytes": 4, "sha256": "2" * 64},
    ]

    assert len(processor_assets_sha256(assets)) == 64
    with pytest.raises(ContractError, match="path-sorted"):
        processor_assets_sha256(list(reversed(assets)))
    with pytest.raises(ContractError, match="normalized and relative"):
        processor_assets_sha256([{"path": "../config.json", "bytes": 3, "sha256": "1" * 64}])


def test_projection_probe_rejects_dataset_manifest_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "training"
    split.mkdir()
    (split / "frame.bin").write_bytes(b"frame")
    manifest = build_dataset_file_manifest(
        split,
        dataset_id="dataset",
        dataset_revision="revision",
        split_name="training",
        relative_paths=("frame.bin",),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), sort_keys=True), encoding="ascii")
    original = projection_probe.load_dataset_file_manifest

    def mutate_manifest(selected: Path) -> object:
        loaded = original(selected)
        selected.write_bytes(b"changed during load")
        return loaded

    monkeypatch.setattr(projection_probe, "load_dataset_file_manifest", mutate_manifest)

    with pytest.raises(ContractError, match="changed while loading"):
        projection_probe._load_stable_dataset_manifest(path)  # noqa: SLF001
