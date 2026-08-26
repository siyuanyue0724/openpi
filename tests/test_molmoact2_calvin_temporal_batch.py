# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from picf_next.data.calvin import CalvinDatasetIndex, CalvinEpisode, CalvinLanguageSegment
from picf_next.data.rollout_targets import PhysicalObjectGeometryFrame
from picf_next.hosts.molmoact2_training import CalvinStatefulLossTargets
from picf_next.models.dynamics_loss import ObjectLifecycleInventoryTarget
from picf_next.models.set_loss import ObjectSetTarget
from picf_next.training.molmoact2_calvin_temporal import (
    CalvinStationaryTemporalBatchBuilder,
)
from picf_next.training.temporal_clips import StationaryTemporalClip
from tests.geometry_contract import synthetic_geometry_contract
from tests.test_molmoact2_source_cache import _load, _write_cache

GEOMETRY = synthetic_geometry_contract(3)


def _index(root: Path) -> CalvinDatasetIndex:
    root.mkdir()
    for global_index in range(100, 103):
        relative = np.asarray(
            [0.01 * (global_index - 99), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
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
        episodes=(CalvinEpisode(0, 100, 102),),
        segments=(CalvinLanguageSegment(0, 100, 101, "fixture", "fixture", 0),),
    )


def _geometry(global_index: int) -> PhysicalObjectGeometryFrame:
    value = torch.tensor([[float(global_index), 0.0, 0.0]])
    return PhysicalObjectGeometryFrame(
        identity_keys=("object:0",),
        geometry=value,
        geometry_variance=torch.zeros_like(value),
        geometry_supervised=torch.ones_like(value, dtype=torch.bool),
        geometry_contract=GEOMETRY,
    )


def test_calvin_stationary_batch_keeps_targets_post_forward_and_time_causal(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    payload = {
        "tokens": torch.stack(
            (
                torch.full((4, 3), 1.0, dtype=torch.bfloat16),
                torch.full((4, 3), 2.0, dtype=torch.bfloat16),
            )
        ),
        "valid": torch.ones(2, 4, dtype=torch.bool),
    }
    manifest_sha, _manifest = _write_cache(cache_root, shard_payloads=(payload,))
    cache = _load(cache_root, manifest_sha)
    index = _index(tmp_path / "calvin")
    target_calls = []

    def visible(requests, layout) -> CalvinStatefulLossTargets:
        target_calls.append(tuple(request.source_global_index for request in requests))
        ownership = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            device=layout.token_valid.device,
        )
        targets = tuple(
            ObjectSetTarget(
                ownership=ownership,
                token_valid=layout.token_valid[row],
                object_inventory_complete=True,
                temporal_identity_keys=("object:0",),
            )
            for row in range(layout.token_valid.shape[0])
        )
        lifecycle = tuple(
            ObjectLifecycleInventoryTarget(
                alive_identity_keys=("object:0",),
                inventory_complete=True,
                visibility=torch.ones(1, device=layout.token_valid.device),
                visibility_supervised=torch.ones(
                    1,
                    device=layout.token_valid.device,
                    dtype=torch.bool,
                ),
            )
            for _row in range(layout.token_valid.shape[0])
        )
        return CalvinStatefulLossTargets(
            set_targets=targets,
            lifecycle_targets=lifecycle,
        )

    builder = CalvinStationaryTemporalBatchBuilder(
        index,
        cache,
        visible_target_builder=visible,
        geometry_contract=GEOMETRY,
        geometry_provider=_geometry,
        maximum_horizon=1,
        supervised_horizons=(1,),
    )
    batch = builder.build(
        (StationaryTemporalClip(0, 0, 100, 1, 1),),
        device="cpu",
    )

    assert target_calls == []
    assert batch.source_indices_by_frame == ((100,), (101,))
    assert torch.count_nonzero(batch.observations[0].previous_executed_action) == 0
    torch.testing.assert_close(
        batch.observations[1].previous_executed_action.float()[0],
        torch.from_numpy(index.action(100).copy()),
    )
    assert batch.observations[0].native_banks[0].encoder_contract is not None
    assert torch.equal(batch.layouts[0].token_valid, torch.ones(1, 4, dtype=torch.bool))

    supervision = batch.build_supervision(0)
    assert target_calls == [(100,)]
    assert supervision.set_targets[0].temporal_identity_keys == ("object:0",)
    rollout = batch.build_geometry_rollout()
    assert rollout.executed_actions.shape == (1, 1, 7)
    assert rollout.identity_keys[0][0][0] == "object:0"
