from __future__ import annotations

import numpy as np
import pytest

from picf_next.contracts import ContractError
from tools.audit_calvin_tactile_validity import (
    build_report,
    deterministic_sample_steps,
    summarize_tactile_frames,
)


def test_deterministic_sample_steps_spans_episode_support_and_adds_requested_step() -> None:
    ranges = np.asarray([[10, 12], [20, 23]], dtype=np.int64)

    steps = deterministic_sample_steps(ranges, sample_count=3, include_steps=(11,))

    assert steps == (10, 11, 20, 23)
    with pytest.raises(ContractError, match="outside episode support"):
        deterministic_sample_steps(ranges, sample_count=3, include_steps=(19,))


def test_tactile_summary_keeps_sensor_measurements_separate() -> None:
    first = np.zeros((2, 2, 2), dtype=np.float32)
    second = np.zeros((2, 2, 2), dtype=np.float32)
    second[0, 0, 0] = 0.2
    second[..., 1] = 0.01

    report = summarize_tactile_frames((first, second), frame_steps=(10, 20), thresholds=(0.0, 0.05))

    assert report["frame_count"] == 2
    assert report["depth_tactile_shape"] == [2, 2, 2]
    sensor_0, sensor_1 = report["sensors"]
    assert sensor_0["exact_zero_frames"] == 1
    assert sensor_0["strongest_absolute_deformation_step"] == 20
    assert sensor_0["frames_above_absolute_max_threshold"] == {"0": 1, "0.05": 1}
    assert sensor_0["representative_steps_by_absolute_max_band"] == {
        "exact_zero": [10],
        "(0,0.05]": [],
        ">0.05": [20],
    }
    assert sensor_1["exact_zero_frames"] == 1
    assert sensor_1["frames_above_absolute_max_threshold"] == {"0": 1, "0.05": 0}


def test_tactile_summary_accepts_signed_deformation_but_rejects_nonfinite_measurements() -> None:
    signed = summarize_tactile_frames(
        (np.full((2, 2, 1), -1.0, dtype=np.float32),), frame_steps=(7,)
    )
    assert signed["sensors"][0]["frame_absolute_max"]["q1"] == 1.0
    with pytest.raises(ContractError, match="finite"):
        summarize_tactile_frames((np.full((2, 2, 1), np.nan, dtype=np.float32),))


def test_build_report_reads_exact_calvin_frame_paths(tmp_path) -> None:
    split = tmp_path / "training"
    split.mkdir()
    np.save(split / "ep_start_end_ids.npy", np.asarray([[0, 1], [3, 3]], dtype=np.int64))
    for step in (0, 1, 3):
        tactile = np.zeros((2, 2, 2), dtype=np.float32)
        tactile[..., 0] = float(step)
        np.savez(split / f"episode_{step:07d}.npz", depth_tactile=tactile)

    report = build_report(split, sample_count=2, include_steps=(1,))

    assert report["episode_frame_count"] == 3
    assert report["sampled_steps_count"] == 3
    assert report["explicit_include_steps"] == [1]
    assert report["thresholds_are_diagnostics_not_contact_labels"] is True
