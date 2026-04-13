from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_picf_policy as sut


def test_build_policy_example_includes_depth_gripper_and_reset() -> None:
    obs = {
        "rgb_obs": {
            "rgb_static": np.zeros((2, 2, 3), dtype=np.uint8),
            "rgb_gripper": np.ones((2, 2, 3), dtype=np.uint8),
        },
        "depth_obs": {
            "depth_static": np.zeros((2, 2), dtype=np.float32),
            "depth_gripper": np.ones((2, 2), dtype=np.float32),
        },
        "robot_obs": np.arange(7, dtype=np.float32),
    }
    payload = sut._build_policy_example(obs, "open drawer", needs_reset=True)
    assert payload["openpi/reset"] is True
    assert payload["prompt"] == "open drawer"
    assert "observation/depth_gripper" in payload


def test_discretize_calvin_gripper_maps_last_dimension_to_binary() -> None:
    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.01], dtype=np.float32)
    discrete = sut._discretize_calvin_gripper(action)
    assert discrete.shape == (7,)
    assert discrete[-1] == -1.0
