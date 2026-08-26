# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from picf_next.data.calvin import CalvinDatasetIndex, CalvinEpisode, CalvinLanguageSegment
from picf_next.data.calvin_rollout_targets import build_calvin_source_geometry_rollout_sample
from picf_next.data.rollout_targets import PhysicalObjectGeometryFrame
from picf_next.geometry import PhysicalGeometryContract

GEOMETRY = PhysicalGeometryContract(
    name="fixture.xyz",
    axes=("x", "y", "z"),
    units=("m", "m", "m"),
    reference_frame="fixture",
    quantity="point",
    normalization_offset=(0.0, 0.0, 0.0),
    normalization_scale=(1.0, 1.0, 1.0),
)


def _index(root: Path) -> CalvinDatasetIndex:
    root.mkdir()
    for global_index in range(10, 15):
        relative = np.asarray(
            [0.01 * (global_index - 9), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        absolute = np.zeros(7, dtype=np.float64)
        absolute[:3] = relative[:3] * 0.02
        absolute[3:6] = relative[3:6] * 0.05
        absolute[-1] = relative[-1]
        np.savez(
            root / f"episode_{global_index:07d}.npz",
            robot_obs=np.zeros(15, dtype=np.float64),
            actions=absolute,
            rel_actions=relative,
        )
    return CalvinDatasetIndex(
        split_root=root,
        dataset_id="fixture",
        dataset_revision="fixture",
        control_hz=30,
        episodes=(CalvinEpisode(0, 10, 14),),
        segments=(
            CalvinLanguageSegment(0, 10, 11, "first", "first task", 0),
            CalvinLanguageSegment(1, 12, 13, "second", "second task", 0),
        ),
    )


def _frame(global_index: int) -> PhysicalObjectGeometryFrame:
    geometry = torch.tensor([[float(global_index), 0.0, 0.0]])
    return PhysicalObjectGeometryFrame(
        identity_keys=("object:0",),
        geometry=geometry,
        geometry_variance=torch.zeros_like(geometry),
        geometry_supervised=torch.ones_like(geometry, dtype=torch.bool),
        geometry_contract=GEOMETRY,
    )


def test_source_rollout_crosses_language_boundary_but_stops_at_episode(tmp_path: Path) -> None:
    index = _index(tmp_path / "calvin")
    calls = []

    def provider(global_index: int) -> PhysicalObjectGeometryFrame:
        calls.append(global_index)
        return _frame(global_index)

    rollout = build_calvin_source_geometry_rollout_sample(
        index,
        global_index=11,
        maximum_horizon=3,
        supervised_horizons=(1, 2),
        geometry_contract=GEOMETRY,
        geometry_provider=provider,
    )
    assert rollout.executed_actions.shape == (3, 7)
    np.testing.assert_allclose(
        rollout.executed_actions.numpy(),
        np.stack((index.action(11), index.action(12), index.action(13))),
    )
    assert calls == [12, 13]
    assert rollout.geometry_frames[2].identity_keys == ()

    calls.clear()
    boundary = build_calvin_source_geometry_rollout_sample(
        index,
        global_index=13,
        maximum_horizon=3,
        supervised_horizons=(1, 2),
        geometry_contract=GEOMETRY,
        geometry_provider=provider,
    )
    assert boundary.executed_actions.shape == (1, 7)
    assert calls == [14]


def test_source_rollout_rejects_terminal_start_and_invalid_horizons(tmp_path: Path) -> None:
    index = _index(tmp_path / "calvin")
    with pytest.raises(ValueError, match="must have a next frame"):
        build_calvin_source_geometry_rollout_sample(
            index,
            global_index=14,
            maximum_horizon=2,
            supervised_horizons=(1,),
            geometry_contract=GEOMETRY,
            geometry_provider=_frame,
        )
    with pytest.raises(ValueError, match="include one"):
        build_calvin_source_geometry_rollout_sample(
            index,
            global_index=10,
            maximum_horizon=2,
            supervised_horizons=(2,),
            geometry_contract=GEOMETRY,
            geometry_provider=_frame,
        )
