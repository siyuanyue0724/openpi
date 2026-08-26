from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.lingbot_libero import (
    LINGBOT_LIBERO_CAMERA_KEYS,
    LINGBOT_VLA2_FEATURE_SLICES,
    LINGBOT_VLA2_WIDTH,
    map_libero_record_to_lingbot,
)
from picf_next.data.robot_record import RobotTransitionRecord, decode_molmoact2_libero_row

ROOT = Path(__file__).resolve().parents[1]


def _record():
    row = {
        "observation.images.image": {"bytes": b"static", "path": "static.png"},
        "observation.images.wrist_image": {"bytes": b"wrist", "path": "wrist.png"},
        "observation.state": [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.04, -0.039],
        "action": [0.2, -0.3, 0.4, -0.1, 0.05, 0.2, -1.0],
        "timestamp": 0.3,
        "frame_index": 3,
        "episode_index": 2,
        "index": 17,
        "task_index": 5,
    }
    return decode_molmoact2_libero_row(row, task="open the drawer", episode_length=8)


def test_libero_maps_to_official_lingbot_feature_order_and_masks() -> None:
    record = _record()
    view = map_libero_record_to_lingbot(record)
    end = LINGBOT_VLA2_FEATURE_SLICES["end.position"]
    effector = LINGBOT_VLA2_FEATURE_SLICES["effector.position"]

    assert view.state.shape == (LINGBOT_VLA2_WIDTH,)
    assert view.action.shape == (LINGBOT_VLA2_WIDTH,)
    np.testing.assert_array_equal(view.state[end.start : end.start + 6], record.state[:6])
    np.testing.assert_array_equal(view.action[end.start : end.start + 6], record.action[:6])
    assert view.state[effector.start] == pytest.approx(0.079)
    assert view.action[effector.start] == -1.0
    assert view.state_valid.sum() == 7
    assert view.action_valid.sum() == 7
    assert not view.state_valid[LINGBOT_VLA2_FEATURE_SLICES["arm.position"]].any()
    assert not view.state_valid[LINGBOT_VLA2_FEATURE_SLICES["reserved"]].any()


def test_feature_order_matches_the_pinned_official_real_robot_config() -> None:
    config_path = (
        ROOT / "references/source_checkouts/lingbot-vla-v2/configs/vla/real_robot/real_robot.yaml"
    )
    if not config_path.is_file():
        pytest.skip("optional pinned LingBot source checkout is absent")
    text = config_path.read_text()
    joints_block = text.split("  joints:\n", 1)[1].split("  cameras:\n", 1)[0]
    cameras_block = text.split("  cameras:\n", 1)[1].split("  norm_type:", 1)[0]
    official_joints = [
        (name, int(width))
        for name, width in re.findall(r"^    - ([a-z.]+): ([0-9]+)$", joints_block, re.M)
    ]
    official_cameras = re.findall(r"^    - ([a-z_]+)$", cameras_block, re.M)

    expected_joints = []
    for name, feature_slice in LINGBOT_VLA2_FEATURE_SLICES.items():
        if name == "reserved":
            continue
        expected_joints.append((name, feature_slice.stop - feature_slice.start))
    assert official_joints == expected_joints
    assert official_cameras == [key.rsplit(".", 1)[-1] for key in LINGBOT_LIBERO_CAMERA_KEYS]
    assert sum(width for _, width in official_joints) == 51


def test_camera_mapping_retains_both_sources_and_marks_missing_slot() -> None:
    view = map_libero_record_to_lingbot(_record())
    assert view.camera_payloads == (b"static", b"wrist", None)
    np.testing.assert_array_equal(view.camera_valid, [True, True, False])
    assert view.task == "open the drawer"
    assert view.source_global_index == 17


def test_mapping_is_an_additional_view_and_does_not_mutate_raw_record() -> None:
    record = _record()
    state_before = record.state.copy()
    action_before = record.action.copy()
    view = map_libero_record_to_lingbot(record)

    np.testing.assert_array_equal(record.state, state_before)
    np.testing.assert_array_equal(record.action, action_before)
    assert not view.state.flags.writeable
    assert not view.action.flags.writeable
    assert not view.state_valid.flags.writeable
    assert not view.action_valid.flags.writeable
    assert not hasattr(view, "mask")
    assert not hasattr(view, "object_id")


def test_mapping_rejects_a_non_libero_contract() -> None:
    record = object.__new__(RobotTransitionRecord)
    object.__setattr__(record, "contract", "other/v1")
    with pytest.raises(ContractError, match="unsupported record"):
        map_libero_record_to_lingbot(record)
