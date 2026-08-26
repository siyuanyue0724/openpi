from __future__ import annotations

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinPICFEvidenceFrame, CalvinPICFSensorObservation
from picf_next.data.calvin_tactile import (
    CALVIN_TACTILE_STREAM_NAMES,
    build_calvin_tactile_encoder_clips,
    calvin_digit_sensor_poses_world,
    calvin_tactile_source_frames,
)


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _frame(
    timestamp_s: float,
    *,
    left_deformation: float = 0.0,
    right_deformation: float = 0.0,
    rgb_value: int = 0,
) -> CalvinPICFEvidenceFrame:
    values = {
        "observation.images.rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
        "observation.images.rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
        "observation.depth.static": np.zeros((200, 200), dtype=np.float32),
        "observation.depth.gripper": np.zeros((84, 84), dtype=np.float32),
        "observation.tactile.rgb": np.full((160, 120, 6), rgb_value, dtype=np.uint8),
        "observation.tactile.depth": np.zeros((160, 120, 2), dtype=np.float32),
    }
    values["observation.tactile.depth"][..., 0] = left_deformation
    values["observation.tactile.depth"][..., 1] = right_deformation
    observations = tuple(
        CalvinPICFSensorObservation(
            key=key,
            value=_readonly(value),
            timestamp_s=timestamp_s,
            units="test",
        )
        for key, value in values.items()
    )
    return CalvinPICFEvidenceFrame(
        sensor_observations=observations,
        timestamp_s=timestamp_s,
        delta_t_s=1.0 / 30.0,
    )


def test_calvin_tactile_validity_is_independent_per_digit_stream() -> None:
    prefix = tuple(
        _frame(index / 30.0, left_deformation=0.0, right_deformation=0.0, rgb_value=index)
        for index in range(3)
    ) + (_frame(0.1, left_deformation=0.007, right_deformation=4.3e-5, rgb_value=3),)

    clips = build_calvin_tactile_encoder_clips(
        prefix,
        validity_thresholds_m={name: 1e-4 for name in CALVIN_TACTILE_STREAM_NAMES},
    )

    assert tuple(clip.stream_name for clip in clips) == ("left_digit",)
    assert clips[0].hardware_type == "digit"
    assert clips[0].current_absolute_deformation_max_m == pytest.approx(0.007)
    assert clips[0].source_valid == (True, True, True, True)
    assert clips[0].as_array().shape == (4, 160, 120, 3)


def test_calvin_tactile_padding_is_explicit_and_hash_binds_deformation() -> None:
    first = _frame(0.0, left_deformation=0.001)
    clip = build_calvin_tactile_encoder_clips(
        (first,),
        validity_thresholds_m={name: 1e-4 for name in CALVIN_TACTILE_STREAM_NAMES},
    )[0]

    assert clip.padding_count == 3
    assert clip.source_valid == (False, False, False, True)
    assert clip.frames[0] is clip.frames[1] is clip.frames[2] is clip.frames[3]
    assert len(set(clip.source_frame_sha256)) == 1

    changed = _frame(0.0, left_deformation=0.002)
    assert (
        calvin_tactile_source_frames(first)[0].source_sha256
        != calvin_tactile_source_frames(changed)[0].source_sha256
    )


def test_calvin_tactile_two_valid_sensors_remain_two_groups() -> None:
    clips = build_calvin_tactile_encoder_clips(
        (_frame(0.0, left_deformation=0.002, right_deformation=-0.003),),
        validity_thresholds_m={name: 1e-4 for name in CALVIN_TACTILE_STREAM_NAMES},
    )

    assert tuple(clip.stream_name for clip in clips) == CALVIN_TACTILE_STREAM_NAMES
    assert {clip.hardware_type for clip in clips} == {"digit"}


def test_calvin_digit_poses_follow_official_urdf_and_tcp_state() -> None:
    state = np.zeros(15, dtype=np.float32)
    state[:3] = (0.1, -0.2, 0.3)
    state[6] = 0.04

    poses = calvin_digit_sensor_poses_world(state)

    np.testing.assert_allclose(poses["left_digit"][:3, 3], (0.1, -0.16, 0.2884))
    np.testing.assert_allclose(poses["right_digit"][:3, 3], (0.1, -0.24, 0.2884))
    assert np.linalg.det(poses["left_digit"][:3, :3]) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.det(poses["right_digit"][:3, :3]) == pytest.approx(1.0, abs=1e-6)
    assert all(not pose.flags.writeable for pose in poses.values())


@pytest.mark.parametrize(
    "observed",
    (
        -0.010713515300537114,
        0.08377021915967425,
    ),
)
def test_calvin_digit_poses_preserve_observed_contact_joint_excursions(observed: float) -> None:
    state = np.zeros(15, dtype=np.float64)
    state[6] = observed

    poses = calvin_digit_sensor_poses_world(state)
    half_separation = 0.020 + observed / 2.0

    assert poses["left_digit"][1, 3] == pytest.approx(half_separation)
    assert poses["right_digit"][1, 3] == pytest.approx(-half_separation)


def test_calvin_tactile_builder_rejects_incomplete_calibration_and_noncausal_input() -> None:
    with pytest.raises(ContractError, match="cover both"):
        build_calvin_tactile_encoder_clips(
            (_frame(0.0),),
            validity_thresholds_m={"left_digit": 1e-4},
        )
    with pytest.raises(ContractError, match="chronological"):
        build_calvin_tactile_encoder_clips(
            (_frame(0.1), _frame(0.0)),
            validity_thresholds_m={name: 1e-4 for name in CALVIN_TACTILE_STREAM_NAMES},
        )
