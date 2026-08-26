from __future__ import annotations

import pytest

from picf_next.training.temporal_clips import (
    STATIONARY_STATE_CONTRACT,
    build_distributed_stationary_temporal_clip_plan,
    build_stationary_temporal_clip_plan,
)


def test_stationary_clip_plan_is_deterministic_balanced_and_range_safe() -> None:
    kwargs = {
        "source_ranges": ((100, 500), (600, 900)),
        "prefix_lengths": (0, 8, 32, 128),
        "train_length": 4,
        "optimizer_steps": 12,
        "seed": 57,
    }

    first = build_stationary_temporal_clip_plan(**kwargs)
    second = build_stationary_temporal_clip_plan(**kwargs)

    assert first == second
    assert first.plan_sha256 == second.plan_sha256
    assert first.state_contract == STATIONARY_STATE_CONTRACT
    assert {clip.prefix_length for clip in first.clips} == {0, 8, 32, 128}
    for clip in first.clips:
        source_start, source_stop = first.source_ranges[clip.source_range_index]
        assert source_start <= clip.start_global_index
        assert clip.stop_global_index <= source_stop
        assert clip.prefix_indices + clip.train_indices == tuple(
            range(clip.start_global_index, clip.stop_global_index)
        )


def test_stationary_clip_plan_seed_changes_plan_without_changing_contract() -> None:
    common = {
        "source_ranges": ((100, 500),),
        "prefix_lengths": (0, 16, 64),
        "train_length": 8,
        "optimizer_steps": 9,
    }

    first = build_stationary_temporal_clip_plan(seed=1, **common)
    second = build_stationary_temporal_clip_plan(seed=2, **common)

    assert first.plan_sha256 != second.plan_sha256
    assert first.state_contract == second.state_contract


def test_stationary_clip_plan_rejects_crossing_and_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        build_stationary_temporal_clip_plan(
            source_ranges=((100, 200), (150, 250)),
            prefix_lengths=(0,),
            train_length=4,
            optimizer_steps=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="no source range"):
        build_stationary_temporal_clip_plan(
            source_ranges=((100, 120),),
            prefix_lengths=(32,),
            train_length=4,
            optimizer_steps=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="unique increasing"):
        build_stationary_temporal_clip_plan(
            source_ranges=((100, 200),),
            prefix_lengths=(8, 0),
            train_length=4,
            optimizer_steps=1,
            seed=1,
        )


def test_distributed_stationary_plan_aligns_collectives_and_separates_ranks() -> None:
    kwargs = {
        "source_ranges": ((100, 500), (600, 900)),
        "prefix_lengths": (0, 8, 32, 128),
        "train_length": 2,
        "required_future_horizon": 2,
        "optimizer_steps": 12,
        "world_size": 2,
        "seed": 57,
    }
    first = build_distributed_stationary_temporal_clip_plan(**kwargs)
    second = build_distributed_stationary_temporal_clip_plan(**kwargs)

    assert first == second
    assert first.plan_sha256 == second.plan_sha256
    assert first.state_contract == STATIONARY_STATE_CONTRACT
    assert {clips[0].prefix_length for clips in first.clips_by_step} == {0, 8, 32, 128}
    for step, rank_clips in enumerate(first.clips_by_step):
        assert len({clip.prefix_length for clip in rank_clips}) == 1
        assert len({(clip.source_range_index, clip.start_global_index) for clip in rank_clips}) == 2
        assert first.clip(step, 0) == rank_clips[0]
        assert first.clip(step, 1) == rank_clips[1]
        for clip in rank_clips:
            _start, stop = first.source_ranges[clip.source_range_index]
            assert clip.stop_global_index + first.required_future_horizon <= stop


def test_distributed_stationary_plan_requires_one_distinct_clip_per_rank() -> None:
    with pytest.raises(ValueError, match="fewer distinct clips"):
        build_distributed_stationary_temporal_clip_plan(
            source_ranges=((100, 103),),
            prefix_lengths=(0,),
            train_length=2,
            required_future_horizon=1,
            optimizer_steps=1,
            world_size=3,
            seed=1,
        )
