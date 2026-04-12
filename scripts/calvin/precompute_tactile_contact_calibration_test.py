from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np

from openpi.picf.test_utils import build_mini_calvin_dataset


def test_precompute_tactile_contact_calibration_smoke(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)
    output_dir = tmp_path / "artifacts"
    script_path = Path(__file__).with_name("precompute_tactile_contact_calibration.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--calvin-root",
            calvin_root,
            "--split",
            "training",
            "--backend",
            "dir",
            "--sample-stride",
            "1",
            "--max-frames",
            "4",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["sampled_frames"] == 4
    assert 0.0 <= float(summary["active_rate_tau_on"]) <= 1.0

    backgrounds = np.load(output_dir / "tactile_backgrounds.npz")
    assert set(backgrounds.files) == {"digit", "gelsight_mini"}

    stats = json.loads((output_dir / "tactile_contact_stats.json").read_text())
    assert stats["score_mode"] == "rgb_only"
    assert stats["sampled_frames"] == 4
    assert stats["negative_pool_size"] >= 1
    assert stats["tau_on"] >= stats["tau_off"]

    calibration = json.loads((output_dir / "tactile_fingertip_calibration.json").read_text())
    assert len(calibration["u_open_local"]) == 3
    assert len(calibration["o_local"]) == 3
    assert 0.035 <= float(calibration["recommended_pt_bag_radius_m"]) <= 0.055


def test_calibrate_fingertips_precomputes_support_cloud_once_per_selected_frame(monkeypatch, tmp_path: Path) -> None:
    script_path = Path(__file__).with_name("precompute_tactile_contact_calibration.py")
    spec = importlib.util.spec_from_file_location("picf_tactile_calib_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "_search_grids",
        lambda: (
            [np.asarray([1.0, 0.0, 0.0], dtype=np.float32)],
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0, 0.005], dtype=np.float32),
        ),
    )

    call_count = {"value": 0}

    class _FakePointSet:
        def __init__(self) -> None:
            self.xyz_world = np.asarray(
                [
                    [0.02, 0.0, 0.0],
                    [0.03, 0.0, 0.0],
                    [0.025, 0.005, 0.0],
                ],
                dtype=np.float32,
            )

    class _FakeBuilder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, payload):
            call_count["value"] += 1
            return _FakePointSet()

    monkeypatch.setattr(module, "CalvinDepthToPicfPointCloud", _FakeBuilder)

    records = [
        module._FrameRecord(
            step_id=index,
            robot_obs=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04], dtype=np.float32),
            rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
            depth_static=np.ones((8, 8), dtype=np.float32),
            rgb_gripper=np.zeros((8, 8, 3), dtype=np.uint8),
            depth_gripper=np.ones((8, 8), dtype=np.float32),
            tactile_rgb_by_sensor={
                "digit": np.zeros((8, 8, 3), dtype=np.uint8),
                "gelsight_mini": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        )
        for index in range(2)
    ]

    calibration = module._calibrate_fingertips(
        calvin_root=str(tmp_path),
        records=records,
        combined_scores=np.asarray([0.1, 1.0], dtype=np.float32),
        top_fraction=0.5,
        max_top_frames=1,
        target_front_ratio=0.60,
        point_stride=4,
        point_max_points=1024,
        point_crop_radius_m=0.10,
        front_radius_m=0.05,
        front_slack_m=0.008,
    )

    assert call_count["value"] == 1
    assert calibration["evaluated_frames"] == 1
    assert calibration["d_nn_trimmed_mean"] >= 0.0


def test_calibrate_fingertips_caps_geometry_frames(monkeypatch, tmp_path: Path) -> None:
    script_path = Path(__file__).with_name("precompute_tactile_contact_calibration.py")
    spec = importlib.util.spec_from_file_location("picf_tactile_calib_script_cap", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "_search_grids",
        lambda: (
            [np.asarray([1.0, 0.0, 0.0], dtype=np.float32)],
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        ),
    )

    call_count = {"value": 0}

    class _FakePointSet:
        def __init__(self) -> None:
            self.xyz_world = np.asarray([[0.02, 0.0, 0.0]], dtype=np.float32)

    class _FakeBuilder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, payload):
            call_count["value"] += 1
            return _FakePointSet()

    monkeypatch.setattr(module, "CalvinDepthToPicfPointCloud", _FakeBuilder)

    records = [
        module._FrameRecord(
            step_id=index,
            robot_obs=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04], dtype=np.float32),
            rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
            depth_static=np.ones((8, 8), dtype=np.float32),
            rgb_gripper=np.zeros((8, 8, 3), dtype=np.uint8),
            depth_gripper=np.ones((8, 8), dtype=np.float32),
            tactile_rgb_by_sensor={
                "digit": np.zeros((8, 8, 3), dtype=np.uint8),
                "gelsight_mini": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        )
        for index in range(6)
    ]

    calibration = module._calibrate_fingertips(
        calvin_root=str(tmp_path),
        records=records,
        combined_scores=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        top_fraction=1.0,
        max_top_frames=2,
        target_front_ratio=0.60,
        point_stride=4,
        point_max_points=1024,
        point_crop_radius_m=0.10,
        front_radius_m=0.05,
        front_slack_m=0.008,
    )

    assert call_count["value"] == 2
    assert calibration["evaluated_frames"] == 2


def test_candidate_selection_prioritizes_front_ratio_threshold() -> None:
    from scripts.calvin.precompute_tactile_contact_calibration import _candidate_is_better

    incumbent = {
        "objective": 0.008,
        "d_nn_trimmed_mean": 0.006,
        "front_ratio": 0.55,
    }
    candidate = {
        "objective": 0.011,
        "d_nn_trimmed_mean": 0.011,
        "front_ratio": 0.61,
    }
    assert _candidate_is_better(candidate, incumbent, target_front_ratio=0.60) is True
    assert _candidate_is_better(incumbent, candidate, target_front_ratio=0.60) is False
