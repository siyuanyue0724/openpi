from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinEpisode,
    CalvinLanguageSegment,
)
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_SOURCE_COMMIT,
    CALVIN_STATE_RESTORATION,
    CalvinGeometryShard,
    geometry_manifest_payload,
    sha256_file,
)
from picf_next.data.calvin_geometry_sidecar import CalvinPhysicalGeometrySidecar


def _index(tmp_path: Path, *, revision: str = "fixture-v1") -> CalvinDatasetIndex:
    split_root = tmp_path / "training"
    split_root.mkdir(parents=True, exist_ok=True)
    np.save(
        split_root / "scene_info.npy",
        {"calvin_scene_D": np.asarray([10, 12], dtype=np.int64)},
        allow_pickle=True,
    )
    return CalvinDatasetIndex(
        split_root=split_root,
        dataset_id="fixture/calvin",
        dataset_revision=revision,
        control_hz=30,
        episodes=(CalvinEpisode(index=0, start=10, end=12),),
        segments=(
            CalvinLanguageSegment(
                index=0,
                start=10,
                end=12,
                task_key="move_fixture",
                instruction="move the fixture",
                episode_index=0,
            ),
        ),
    )


def _write_sidecar(
    root: Path,
    index: CalvinDatasetIndex,
    *,
    duplicate_last_frame_keys: bool = False,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    shard_path = root / "part00000_shard000000.npz"
    keys = np.asarray(
        [
            "movable/block_red",
            "part/table/button_link",
            "movable/block_red",
            "part/table/button_link",
            "movable/block_red",
            "movable/block_red" if duplicate_last_frame_keys else "part/table/button_link",
        ],
        dtype=np.str_,
    )
    geometry = np.arange(18, dtype=np.float32).reshape(6, 3) / 10.0
    np.savez_compressed(
        shard_path,
        global_indices=np.asarray([10, 11, 12], dtype=np.int64),
        source_state_sha256=np.asarray(["a" * 64, "b" * 64, "c" * 64], dtype=np.str_),
        frame_offsets=np.asarray([0, 2, 4, 6], dtype=np.int64),
        identity_keys=keys,
        geometry=geometry,
        geometry_variance=np.zeros_like(geometry),
        geometry_supervised=np.ones_like(geometry, dtype=np.bool_),
    )
    shard = CalvinGeometryShard(
        path=shard_path.name,
        sha256=sha256_file(shard_path),
        first_global_index=10,
        last_global_index=12,
        frame_count=3,
        object_record_count=6,
    )
    manifest = geometry_manifest_payload(
        dataset_id=index.dataset_id,
        dataset_revision=index.dataset_revision,
        split_name=index.split_root.name,
        calvin_commit=CALVIN_SOURCE_COMMIT,
        calvin_env_commit=CALVIN_ENV_SOURCE_COMMIT,
        scene_info_sha256=sha256_file(index.split_root / "scene_info.npy"),
        global_indices=np.asarray([10, 11, 12], dtype=np.int64),
        shards=(shard,),
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, shard_path


def test_calvin_geometry_sidecar_returns_typed_loss_only_frame(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    _write_sidecar(root, index)
    provider = CalvinPhysicalGeometrySidecar(root, index)

    frame = provider(0, 11)

    assert frame.identity_keys == ("movable/block_red", "part/table/button_link")
    assert frame.geometry.shape == (2, 3)
    np.testing.assert_allclose(
        frame.geometry.numpy(),
        np.asarray([[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]], dtype=np.float32),
    )
    assert frame.geometry_variance.count_nonzero() == 0
    assert frame.geometry_supervised.all()
    assert frame.geometry_contract == CALVIN_OBJECT_GEOMETRY_CONTRACT
    assert provider.source_state_sha256(0, 11) == "b" * 64

    provider.clear_cache()
    assert not provider._cache


def test_calvin_geometry_sidecar_fails_closed_on_hash_contract_and_range_drift(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    manifest_path, shard_path = _write_sidecar(root, index)
    with shard_path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ContractError, match="hash mismatch"):
        CalvinPhysicalGeometrySidecar(root, index)

    _write_sidecar(root, index)
    manifest = json.loads(manifest_path.read_text())
    contract = dict(manifest["geometry_contract"])
    contract["quantity"] = "object_center_of_mass"
    manifest["geometry_contract"] = contract
    from picf_next.geometry import PhysicalGeometryContract

    manifest["geometry_contract_sha256"] = PhysicalGeometryContract.from_dict(contract).fingerprint
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="unexpected physical chart"):
        CalvinPhysicalGeometrySidecar(root, index)

    _write_sidecar(root, index)
    provider = CalvinPhysicalGeometrySidecar(root, index)
    with pytest.raises(ContractError, match="outside"):
        provider(0, 13)


def test_calvin_geometry_sidecar_rejects_duplicate_frame_identity(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    _write_sidecar(root, index, duplicate_last_frame_keys=True)
    provider = CalvinPhysicalGeometrySidecar(root, index)

    with pytest.raises(ContractError, match="unique"):
        provider(0, 12)


def test_calvin_geometry_sidecar_rejects_mutation_after_initial_validation(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    _, shard_path = _write_sidecar(root, index)
    provider = CalvinPhysicalGeometrySidecar(root, index)

    with shard_path.open("ab") as handle:
        handle.write(b"post-validation-mutation")

    with pytest.raises(ContractError, match="hash mismatch"):
        provider(0, 11)


def test_calvin_geometry_sidecar_rejects_dataset_revision_mismatch(tmp_path: Path) -> None:
    source_index = _index(tmp_path, revision="source-v1")
    root = tmp_path / "sidecar"
    _write_sidecar(root, source_index)
    target_index = _index(tmp_path, revision="target-v2")

    with pytest.raises(ContractError, match="dataset identity"):
        CalvinPhysicalGeometrySidecar(root, target_index)


def test_calvin_geometry_sidecar_rejects_old_restoration_semantics(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    manifest_path, _ = _write_sidecar(root, index)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["state_restoration"] == CALVIN_STATE_RESTORATION
    manifest["state_restoration"] = "environment-reset-with-step.v1"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContractError, match="restoration semantics"):
        CalvinPhysicalGeometrySidecar(root, index)


def test_calvin_geometry_sidecar_rejects_scene_assignment_drift(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    _write_sidecar(root, index)
    np.save(
        index.split_root / "scene_info.npy",
        {"calvin_scene_A": np.asarray([10, 12], dtype=np.int64)},
        allow_pickle=True,
    )

    with pytest.raises(ContractError, match="scene assignment"):
        CalvinPhysicalGeometrySidecar(root, index)
