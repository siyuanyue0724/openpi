from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinPICFEvidenceFrame, CalvinPICFSensorObservation
from picf_next.data.calvin_geometry_schema import CALVIN_OBJECT_GEOMETRY_CONTRACT
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CAMERA_SPECS,
    calvin_camera_name_from_host_image_key,
    source_array_sha256,
)
from picf_next.eval.calvin_same_renderer_removal import (
    CalvinSameRendererRemovalStore,
)


def test_calvin_host_image_keys_resolve_through_the_camera_contract() -> None:
    assert calvin_camera_name_from_host_image_key("observation.images.image") == "static"
    assert calvin_camera_name_from_host_image_key("observation.images.wrist_image") == "gripper"
    with pytest.raises(ContractError, match="unknown or ambiguous"):
        calvin_camera_name_from_host_image_key("image")


def _readonly(value: np.ndarray) -> np.ndarray:
    output = value.copy()
    output.setflags(write=False)
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, *, outside_change: bool = False) -> CalvinPICFEvidenceFrame:
    arrays: dict[str, np.ndarray] = {}
    observations = []
    camera_contracts = []
    for camera_index, spec in enumerate(CALVIN_CAMERA_SPECS):
        name = str(spec["camera_name"])
        height, width = int(spec["height"]), int(spec["width"])
        archived = np.full((height, width, 3), 20 + camera_index, dtype=np.uint8)
        factual = archived.copy()
        removed = factual.copy()
        removed[1, 1] = 80
        if outside_change and camera_index == 0:
            removed[2, 2] = 90
        factual_depth = np.ones((height, width), dtype=np.float32)
        removed_depth = factual_depth.copy()
        removed_depth[1, 1] = 1.1
        factual_owner = np.zeros((height, width), dtype=np.uint8)
        factual_owner[1, 1] = 1
        removed_owner = factual_owner.copy()
        removed_owner[1, 1] = 0
        arrays.update(
            {
                f"{name}_archived_rgb": archived,
                f"{name}_factual_depth_m": factual_depth,
                f"{name}_factual_owner": factual_owner,
                f"{name}_factual_rgb": factual,
                f"{name}_removed_depth_m": removed_depth,
                f"{name}_removed_owner": removed_owner,
                f"{name}_removed_rgb": removed,
            }
        )
        observations.extend(
            (
                CalvinPICFSensorObservation(
                    key=(
                        "observation.images.rgb_static"
                        if name == "static"
                        else "observation.images.rgb_gripper"
                    ),
                    value=_readonly(archived),
                    timestamp_s=1.0,
                    units="sRGB uint8",
                ),
                CalvinPICFSensorObservation(
                    key=(
                        "observation.depth.static"
                        if name == "static"
                        else "observation.depth.gripper"
                    ),
                    value=_readonly(factual_depth),
                    timestamp_s=1.0,
                    units="meters",
                ),
            )
        )
        changed = (
            np.any(factual != removed, axis=-1)
            | (factual_depth != removed_depth)
            | (factual_owner != removed_owner)
        )
        camera_contracts.append(
            {
                "camera_name": name,
                "changed_pixel_count": int(changed.sum()),
                "factual_rgb_sha256": source_array_sha256(f"{name}_factual_rgb", factual),
                "removed_rgb_sha256": source_array_sha256(f"{name}_removed_rgb", removed),
                "target_pixel_count": 1,
            }
        )
    archive = root / "frame0000007_object-target.npz"
    np.savez_compressed(archive, **arrays)
    pair = {
        "cameras": camera_contracts,
        "identity_keys": ["object/target"],
        "method": "same-restored-state.exact-link-alpha-removal.v1",
        "model_input_contains_identity_or_owner": False,
        "source_global_index": 7,
        "source_state_sha256": "a" * 64,
        "target_identity_key": "object/target",
        "target_owner_index": 1,
    }
    summary = {
        "dataset_id": "dataset",
        "dataset_revision": "revision",
        "probe_count": 1,
        "probes": [
            {
                "array_archive": archive.name,
                "array_archive_sha256": _sha256(archive),
                "pair": pair,
            }
        ],
        "schema": "picf-next.calvin-object-removal-probe.v1",
    }
    (root / "summary.json").write_text(json.dumps(summary))
    observations.append(
        CalvinPICFSensorObservation(
            key="observation.tactile.depth",
            value=_readonly(np.ones((2, 2), dtype=np.float32)),
            timestamp_s=1.0,
            units="meters",
        )
    )
    return CalvinPICFEvidenceFrame(
        sensor_observations=tuple(observations),
        timestamp_s=1.0,
        delta_t_s=1.0 / 30.0,
    )


def test_same_renderer_store_reconstructs_verified_target_free_pair(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    store = CalvinSameRendererRemovalStore(
        tmp_path,
        dataset_id="dataset",
        dataset_revision="revision",
    )

    pair = store(
        source,
        global_index=7,
        target_identity_keys=("object/target",),
    )

    assert pair is not None
    assert store.keys == ((7, "object/target"),)
    assert (
        store(
            source,
            global_index=7,
            target_identity_keys=("object/target", "object/control"),
        )
        is None
    )
    assert pair.target_identity_keys == ("object/target",)
    assert pair.contract_dict()["model_input_contains_identity_or_owner"] is False
    factual = {item.key: item.value for item in pair.factual_evidence_frame.sensor_observations}
    removed = {item.key: item.value for item in pair.evidence_frame.sensor_observations}
    assert factual["observation.images.rgb_static"][1, 1].tolist() == [20, 20, 20]
    assert removed["observation.images.rgb_static"][1, 1].tolist() == [80, 80, 80]
    assert np.array_equal(
        factual["observation.tactile.depth"],
        removed["observation.tactile.depth"],
    )
    assert all(not item.value.flags.writeable for item in pair.evidence_frame.sensor_observations)


def test_planned_removal_bank_exposes_strict_partitions(tmp_path: Path) -> None:
    _fixture(tmp_path)
    original = json.loads((tmp_path / "summary.json").read_text())
    source_record = original["probes"][0]
    source_archive = tmp_path / source_record["array_archive"]
    probes = []
    for offset, partition in enumerate(("train", "validation", "heldout")):
        global_index = 7 + offset
        identity = f"object/{partition}"
        archive = tmp_path / f"frame{global_index:07d}_{partition}.npz"
        shutil.copyfile(source_archive, archive)
        pair = dict(source_record["pair"])
        pair.update(
            {
                "identity_keys": [identity],
                "source_global_index": global_index,
                "target_identity_key": identity,
            }
        )
        probes.append(
            {
                "array_archive": archive.name,
                "array_archive_sha256": _sha256(archive),
                "calibration": {},
                "contact_sheet": f"{partition}.png",
                "contact_sheet_sha256": "c" * 64,
                "pair": pair,
                "plan_request": {
                    "partition": partition,
                    "global_index": global_index,
                    "source_segment_index": offset,
                    "scene": "calvin_scene_D",
                    "target_identity_key": identity,
                    "static_visible_pixels": 1,
                    "gripper_visible_pixels": 1,
                    "task_key": f"task_{partition}",
                    "instruction": f"instruction {partition}",
                },
                "tasks": [],
            }
        )
    summary = {
        "dataset_id": "dataset",
        "dataset_revision": "revision",
        "pair_plan": "/mnt/plan.json",
        "pair_plan_sha256": "a" * 64,
        "probe_count": len(probes),
        "probes": probes,
        "schema": "picf-next.calvin-object-removal-bank.v2",
        "source_sidecar_manifest_sha256": "b" * 64,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    store = CalvinSameRendererRemovalStore(
        tmp_path,
        dataset_id="dataset",
        dataset_revision="revision",
    )

    assert store.pair_plan_sha256 == "a" * 64
    assert store.keys_for_partition("train") == ((7, "object/train"),)
    assert store.keys_for_partition("validation") == ((8, "object/validation"),)
    assert store.keys_for_partition("heldout") == ((9, "object/heldout"),)


def test_same_renderer_store_builds_measurement_only_physical_pair(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionFrame,
        CalvinVisibleOwnerRaster,
    )

    source = _fixture(tmp_path)
    cameras = []
    for spec in CALVIN_CAMERA_SPECS:
        height, width = int(spec["height"]), int(spec["width"])
        owner = np.zeros((height, width), dtype=np.uint8)
        owner[1, 1] = 1
        owner.setflags(write=False)
        supervised = np.ones((height, width), dtype=np.bool_)
        supervised.setflags(write=False)
        cameras.append(
            CalvinVisibleOwnerRaster(
                camera_name=str(spec["camera_name"]),
                host_image_key=str(spec["host_image_key"]),
                owner_index=owner,
                owner_supervised=supervised,
                source_rgb_sha256="a" * 64,
                source_depth_sha256="b" * 64,
                rgb_mae=0.0,
                depth_mae_m=0.0,
                depth_p95_m=0.0,
                depth_consistent_fraction=1.0,
            )
        )
    physical = CalvinPhysicalSupervisionFrame(
        identity_keys=("object/target",),
        geometry=torch.tensor([[0.1, 0.2, 0.3]]),
        geometry_variance=torch.zeros(1, 3),
        geometry_supervised=torch.ones(1, 3, dtype=torch.bool),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=tuple(cameras),
    )
    store = CalvinSameRendererRemovalStore(
        tmp_path,
        dataset_id="dataset",
        dataset_revision="revision",
    )

    pair = store(
        source,
        global_index=7,
        target_identity_keys=("object/target",),
        physical_frame=physical,
    )

    assert pair is not None
    assert pair.factual_physical_frame is not None
    assert pair.removed_physical_frame is not None
    assert any(np.any(camera.owner_index == 1) for camera in pair.factual_physical_frame.cameras)
    assert all(
        not np.any(camera.owner_index == 1) for camera in pair.removed_physical_frame.cameras
    )
    for frame, branch in (
        (pair.factual_physical_frame, pair.factual_evidence_frame),
        (pair.removed_physical_frame, pair.evidence_frame),
    ):
        sensors = {item.key: item.value for item in branch.sensor_observations}
        for camera, spec in zip(frame.cameras, CALVIN_CAMERA_SPECS, strict=True):
            assert camera.source_rgb_sha256 == source_array_sha256(
                str(spec["source_rgb_field"]),
                sensors[
                    "observation.images.rgb_static"
                    if camera.camera_name == "static"
                    else "observation.images.rgb_gripper"
                ],
            )


def test_same_renderer_store_rejects_changes_outside_target_support(tmp_path: Path) -> None:
    source = _fixture(tmp_path, outside_change=True)
    store = CalvinSameRendererRemovalStore(
        tmp_path,
        dataset_id="dataset",
        dataset_revision="revision",
    )

    with pytest.raises(ContractError, match="outside exact target support"):
        store(
            source,
            global_index=7,
            target_identity_keys=("object/target",),
        )
