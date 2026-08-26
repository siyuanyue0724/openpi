from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from picf_next.data.calvin_pointcloud import (
    CalvinCalibratedPointCloudBuilder,
    deterministic_farthest_point_indices,
)


def _write_calibration(tmp_path: Path) -> Path:
    payload = {
        "cameras": {
            "static": {
                "K": [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]],
                "W_T_C": np.eye(4).tolist(),
            },
            "gripper": {
                "K": [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]],
                "E_T_C": np.eye(4).tolist(),
            },
        }
    }
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _frame() -> dict[str, np.ndarray]:
    static = np.zeros((4, 4, 3), dtype=np.uint8)
    static[..., 0] = 255
    wrist = np.zeros((4, 4, 3), dtype=np.uint8)
    wrist[..., 1] = 255
    robot = np.zeros(15, dtype=np.float32)
    robot[:3] = [1.0, 0.0, 0.0]
    return {
        "rgb_static": static,
        "depth_static": np.ones((4, 4), dtype=np.float32),
        "rgb_gripper": wrist,
        "depth_gripper": np.ones((4, 4), dtype=np.float32),
        "robot_obs": robot,
    }


def test_calibrated_point_builder_merges_static_and_dynamic_wrist_views(
    tmp_path: Path,
) -> None:
    builder = CalvinCalibratedPointCloudBuilder(
        _write_calibration(tmp_path), pixel_stride=2, maximum_points=32
    )

    cloud = builder.build(_frame())

    assert cloud.xyz_world.shape == cloud.colors.shape == (8, 3)
    assert cloud.view_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert (cloud.xyz_world[4:, 0] > cloud.xyz_world[:4, 0]).all()
    np.testing.assert_array_equal(cloud.colors[:4, 0], np.ones(4))
    np.testing.assert_array_equal(cloud.colors[4:, 1], np.ones(4))


def test_calibrated_point_builder_is_invariant_to_task_focus_fields(tmp_path: Path) -> None:
    builder = CalvinCalibratedPointCloudBuilder(
        _write_calibration(tmp_path), pixel_stride=1, maximum_points=7
    )
    factual = _frame()
    task_conditioned = {**factual, "focus_center_world": np.asarray([99.0, 99.0, 99.0])}

    first = builder.build(factual)
    second = builder.build(task_conditioned)

    np.testing.assert_array_equal(first.xyz_world, second.xyz_world)
    np.testing.assert_array_equal(first.colors, second.colors)
    np.testing.assert_array_equal(first.view_ids, second.view_ids)


def test_deterministic_farthest_points_are_unique_and_repeatable() -> None:
    points = np.stack((np.arange(20, dtype=np.float32), np.zeros(20), np.zeros(20)), axis=1)

    first = deterministic_farthest_point_indices(points, 6)
    second = deterministic_farthest_point_indices(points, 6)

    np.testing.assert_array_equal(first, second)
    assert np.unique(first).shape[0] == 6
    assert not first.flags.writeable
