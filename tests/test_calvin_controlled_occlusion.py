from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from picf_next.data.calvin import (  # noqa: E402
    CalvinPICFEvidenceFrame,
    CalvinPICFSensorObservation,
)
from picf_next.data.calvin_geometry_schema import (  # noqa: E402
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
)
from picf_next.data.calvin_physical_supervision_schema import (  # noqa: E402
    source_array_sha256,
)
from picf_next.data.calvin_physical_supervision_sidecar import (  # noqa: E402
    CalvinPhysicalSupervisionFrame,
    CalvinVisibleOwnerRaster,
)
from picf_next.eval.calvin_controlled_occlusion import (  # noqa: E402
    build_calvin_controlled_rgb_occlusion,
)


def _readonly(value: np.ndarray) -> np.ndarray:
    output = value.copy()
    output.setflags(write=False)
    return output


def _camera(
    name: str,
    host_key: str,
    source_field: str,
    image: np.ndarray,
    owner: np.ndarray,
) -> CalvinVisibleOwnerRaster:
    return CalvinVisibleOwnerRaster(
        camera_name=name,
        host_image_key=host_key,
        owner_index=_readonly(owner.astype(np.uint8)),
        owner_supervised=_readonly(np.ones(owner.shape, dtype=np.bool_)),
        source_rgb_sha256=source_array_sha256(source_field, image),
        source_depth_sha256="a" * 64,
        rgb_mae=0.0,
        depth_mae_m=0.0,
        depth_p95_m=0.0,
        depth_consistent_fraction=1.0,
    )


def _fixture() -> tuple[CalvinPICFEvidenceFrame, CalvinPhysicalSupervisionFrame]:
    static = np.full((200, 200, 3), (10, 20, 30), dtype=np.uint8)
    static[50:60, 70:80] = (240, 10, 10)
    wrist = np.full((84, 84, 3), (40, 50, 60), dtype=np.uint8)
    static = _readonly(static)
    wrist = _readonly(wrist)
    depth = _readonly(np.ones((200, 200), dtype=np.float32))
    frame = CalvinPICFEvidenceFrame(
        sensor_observations=(
            CalvinPICFSensorObservation(
                key="observation.images.rgb_static",
                value=static,
                timestamp_s=1.0,
                units="sRGB uint8",
            ),
            CalvinPICFSensorObservation(
                key="observation.images.rgb_gripper",
                value=wrist,
                timestamp_s=1.0,
                units="sRGB uint8",
            ),
            CalvinPICFSensorObservation(
                key="observation.depth.static",
                value=depth,
                timestamp_s=1.0,
                units="meters",
            ),
        ),
        timestamp_s=1.0,
        delta_t_s=1.0 / 30.0,
    )
    static_owner = np.zeros((200, 200), dtype=np.uint8)
    static_owner[50:60, 70:80] = 1
    wrist_owner = np.zeros((84, 84), dtype=np.uint8)
    physical = CalvinPhysicalSupervisionFrame(
        identity_keys=("movable/block_red", "movable/block_blue"),
        geometry=torch.zeros(2, CALVIN_OBJECT_GEOMETRY_CONTRACT.dimension),
        geometry_variance=torch.ones(2, CALVIN_OBJECT_GEOMETRY_CONTRACT.dimension),
        geometry_supervised=torch.ones(
            2,
            CALVIN_OBJECT_GEOMETRY_CONTRACT.dimension,
            dtype=torch.bool,
        ),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=(
            _camera(
                "static",
                "observation.images.image",
                "rgb_static",
                static,
                static_owner,
            ),
            _camera(
                "gripper",
                "observation.images.wrist_image",
                "rgb_gripper",
                wrist,
                wrist_owner,
            ),
        ),
    )
    return frame, physical


def test_controlled_occlusion_changes_only_visible_rgb_with_audited_bbox() -> None:
    frame, physical = _fixture()
    source_values = {item.key: item.value for item in frame.sensor_observations}

    result = build_calvin_controlled_rgb_occlusion(
        frame,
        physical,
        target_identity_keys=("movable/block_red",),
    )

    changed = {item.key: item.value for item in result.evidence_frame.sensor_observations}
    static_report, wrist_report = result.cameras
    assert static_report.target_pixel_count == 100
    assert static_report.target_bbox_xyxy == (70, 50, 80, 60)
    assert static_report.occluder_bbox_xyxy == (67, 47, 83, 63)
    assert static_report.occluder_pixel_count == 256
    assert static_report.fill_rgb == (10, 20, 30)
    assert static_report.source_rgb_sha256 != static_report.occluded_rgb_sha256
    assert wrist_report.target_pixel_count == 0
    assert wrist_report.source_rgb_sha256 == wrist_report.occluded_rgb_sha256
    assert changed["observation.images.rgb_static"].flags.writeable is False
    assert np.all(changed["observation.images.rgb_static"][47:63, 67:83] == (10, 20, 30))
    assert (
        changed["observation.images.rgb_gripper"] is source_values["observation.images.rgb_gripper"]
    )
    assert changed["observation.depth.static"] is source_values["observation.depth.static"]
    assert np.all(source_values["observation.images.rgb_static"][50:60, 70:80] == (240, 10, 10))
    assert result.contract_dict()["model_input_contains_structural_target"] is False


def test_controlled_occlusion_fails_closed_on_absent_target_or_hash_drift() -> None:
    frame, physical = _fixture()
    with pytest.raises(ValueError, match="no visible pixel"):
        build_calvin_controlled_rgb_occlusion(
            frame,
            physical,
            target_identity_keys=("movable/block_blue",),
        )

    camera = physical.cameras[0]
    drifted = replace(camera, source_rgb_sha256="b" * 64)
    with pytest.raises(ValueError, match="differs from the physical sidecar"):
        build_calvin_controlled_rgb_occlusion(
            frame,
            replace(physical, cameras=(drifted, physical.cameras[1])),
            target_identity_keys=("movable/block_red",),
        )


def test_controlled_occlusion_rejects_unbounded_parameters() -> None:
    frame, physical = _fixture()
    with pytest.raises(ValueError, match="lie in"):
        build_calvin_controlled_rgb_occlusion(
            frame,
            physical,
            target_identity_keys=("movable/block_red",),
            bbox_expansion_fraction=1.1,
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        build_calvin_controlled_rgb_occlusion(
            frame,
            physical,
            target_identity_keys=("movable/block_red",),
            minimum_margin_pixels=-1,
        )
