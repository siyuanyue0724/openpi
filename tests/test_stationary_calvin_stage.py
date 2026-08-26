# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from picf_next.training.stationary_calvin_stage import (
    load_stationary_calvin_stage_definition,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/molmoact2_calvin_m3_stationary_temporal.json"


def test_stationary_calvin_definition_closes_stage_and_distributed_clip_plan() -> None:
    definition = load_stationary_calvin_stage_definition(
        CONFIG,
        repository_root=ROOT,
    )

    assert definition.structural_foundation.objective_config.action_weight == 0.0
    assert definition.historical_foundation.objective_config.action_weight == 1.0
    assert definition.maximum_horizon == 2
    assert definition.clip_plan.source_ranges == ((358482, 360282),)
    assert definition.clip_plan.world_size == 2
    assert definition.clip_plan.optimizer_steps == 200
    assert definition.clip_plan.required_future_horizon == 2
    for step, clips in enumerate(definition.clip_plan.clips_by_step):
        assert all(clip.optimizer_step == step for clip in clips)
        assert len({clip.prefix_length for clip in clips}) == 1
        assert len({clip.start_global_index for clip in clips}) == 2
        assert all(clip.stop_global_index + 2 <= 360282 for clip in clips)
