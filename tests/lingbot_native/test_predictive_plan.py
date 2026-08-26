from __future__ import annotations

import hashlib
import inspect

from picf_next.lingbot_native.predictive_plan import (
    build_native_current_grid_coverage_plan,
    build_native_predictive_coverage_plan,
)
from picf_next.lingbot_native.temporal import (
    FROZEN_OVERSHOOT_HORIZONS,
    TemporalEstimatorConfig,
    native_temporal_batch_seed,
    sample_temporal_batch_plan,
)
from picf_next.training.control import EpisodeSampleSequence, FrozenEpisodeStreamPlan


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _config(**overrides: object) -> TemporalEstimatorConfig:
    values = {
        "local_bptt_probability": 0.2,
        "overshoot_probability": 0.1,
        "source_mask_probability": 0.1,
        "maximum_optimizer_lag": 8,
    }
    values.update(overrides)
    return TemporalEstimatorConfig(**values)  # type: ignore[arg-type]


def _episode(name: str, length: int) -> EpisodeSampleSequence:
    return EpisodeSampleSequence(
        episode_key=name,
        sample_keys=tuple(f"{name}/frame-{index:04d}" for index in range(length)),
    )


def _plan(
    episodes: tuple[EpisodeSampleSequence, ...],
    *,
    total_steps: int,
    global_batch_size: int,
) -> FrozenEpisodeStreamPlan:
    return FrozenEpisodeStreamPlan(
        dataset_id="calvin",
        dataset_revision="fixture",
        dataset_manifest_sha256=_sha("dataset"),
        episodes=episodes,
        comparison_id="picf-full",
        seed=19,
        global_batch_size=global_batch_size,
        total_steps=total_steps,
    )


def _source_mapping(
    episodes: tuple[EpisodeSampleSequence, ...],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for episode_index, episode in enumerate(episodes):
        base = episode_index * 10_000
        result.update({key: base + index for index, key in enumerate(episode.sample_keys)})
    return result


def test_predictive_coverage_is_deterministic_sparse_and_segment_bounded() -> None:
    episodes = (_episode("segment-a", 80), _episode("segment-b", 80))
    plan = _plan(episodes, total_steps=12, global_batch_size=2)
    config = _config(
        local_bptt_probability=0.0,
        overshoot_probability=1.0,
        source_mask_probability=0.0,
    )
    sources = _source_mapping(episodes)

    first = build_native_predictive_coverage_plan(
        plan,
        config,
        source_global_index_for_sample=sources.__getitem__,
    )
    # Partitioning the same global plan cannot alter offline target coverage.
    plan.microbatch_for_rank(
        0,
        rank=1,
        world_size=2,
        gradient_accumulation_steps=1,
        accumulation_index=0,
    )
    second = build_native_predictive_coverage_plan(
        plan,
        config,
        source_global_index_for_sample=sources.__getitem__,
    )

    assert first == second
    assert first.stream_plan_sha256 == plan.plan_sha256
    assert first.temporal_estimator_sha256 == config.digest
    assert len(first.pairs) < sum(
        max(0, len(episode.sample_keys) - horizon)
        for episode in episodes
        for horizon in FROZEN_OVERSHOOT_HORIZONS
    )
    for source, horizon in first.pairs:
        base = source // 10_000 * 10_000
        assert source + horizon < base + 80


def test_predictive_coverage_contains_only_exact_controlled_overshoot_queries() -> None:
    episodes = (_episode("segment-a", 80),)
    plan = _plan(episodes, total_steps=1, global_batch_size=1)
    config = _config(
        local_bptt_probability=0.0,
        overshoot_probability=1.0,
        source_mask_probability=0.0,
    )
    sources = _source_mapping(episodes)
    transition = plan.global_batch(0).transitions[0]
    seed = native_temporal_batch_seed(
        parent_seed=plan.seed,
        comparison_id=plan.comparison_id,
        optimizer_step=0,
        sample_keys=(transition.sample.sample_key,),
    )
    sampled = sample_temporal_batch_plan(
        config,
        seed=seed,
        state_ages=(0,),
        available_future_steps=(79,),
        optimizer_lags=(0,),
    )
    assert sampled.local_bptt_steps is None
    assert sampled.overshoot_horizon is not None

    coverage = build_native_predictive_coverage_plan(
        plan,
        config,
        source_global_index_for_sample=sources.__getitem__,
    )
    expected = {(0, sampled.overshoot_horizon)}
    assert set(coverage.pairs) == expected


def test_predictive_coverage_adds_required_horizon_for_every_planned_sample() -> None:
    episodes = (_episode("segment-a", 20), _episode("segment-b", 20))
    plan = _plan(episodes, total_steps=5, global_batch_size=2)
    config = _config(overshoot_probability=0.0)
    sources = _source_mapping(episodes)

    coverage = build_native_predictive_coverage_plan(
        plan,
        config,
        source_global_index_for_sample=sources.__getitem__,
        required_horizons=(1,),
    )

    expected = {
        (sources[transition.sample.sample_key], 1)
        for step in range(plan.total_steps)
        for transition in plan.global_batch(step).transitions
    }
    assert set(coverage.pairs) == expected


def test_predictive_coverage_omits_unavailable_required_horizon() -> None:
    episodes = (_episode("segment-a", 2), _episode("segment-b", 2))
    plan = _plan(episodes, total_steps=2, global_batch_size=2)
    sources = _source_mapping(episodes)

    coverage = build_native_predictive_coverage_plan(
        plan,
        _config(overshoot_probability=0.0),
        source_global_index_for_sample=sources.__getitem__,
        required_horizons=(1,),
    )

    expected = {
        (sources[episode.sample_keys[0]], 1)
        for episode in episodes
    }
    assert set(coverage.pairs) == expected


def test_predictive_coverage_identity_changes_with_plan_or_estimator() -> None:
    episodes = (_episode("segment-a", 80),)
    sources = _source_mapping(episodes)
    first = build_native_predictive_coverage_plan(
        _plan(episodes, total_steps=2, global_batch_size=1),
        _config(
            local_bptt_probability=0.0,
            overshoot_probability=1.0,
            source_mask_probability=0.0,
        ),
        source_global_index_for_sample=sources.__getitem__,
    )
    longer = build_native_predictive_coverage_plan(
        _plan(episodes, total_steps=3, global_batch_size=1),
        _config(
            local_bptt_probability=0.0,
            overshoot_probability=1.0,
            source_mask_probability=0.0,
        ),
        source_global_index_for_sample=sources.__getitem__,
    )
    changed_estimator = build_native_predictive_coverage_plan(
        _plan(episodes, total_steps=2, global_batch_size=1),
        _config(
            local_bptt_probability=0.0,
            overshoot_probability=0.9,
            source_mask_probability=0.0,
        ),
        source_global_index_for_sample=sources.__getitem__,
    )

    assert first.coverage_sha256 != longer.coverage_sha256
    assert first.coverage_sha256 != changed_estimator.coverage_sha256
    assert set(inspect.signature(build_native_predictive_coverage_plan).parameters) == {
        "stream_plan",
        "temporal_config",
        "source_global_index_for_sample",
        "required_horizons",
    }


def test_current_grid_coverage_is_exact_for_primary_and_local_correction_frames() -> None:
    episodes = (_episode("segment-a", 20), _episode("segment-b", 20))
    plan = _plan(episodes, total_steps=5, global_batch_size=2)
    sources = _source_mapping(episodes)
    config = _config(
        local_bptt_probability=1.0,
        overshoot_probability=0.0,
        source_mask_probability=0.0,
    )

    coverage = build_native_current_grid_coverage_plan(
        plan,
        config,
        source_global_index_for_sample=sources.__getitem__,
    )
    expected: set[int] = set()
    for optimizer_step in range(plan.total_steps):
        transitions = plan.global_batch(optimizer_step).transitions
        availability = tuple(19 - transition.transition_index for transition in transitions)
        seed = native_temporal_batch_seed(
            parent_seed=plan.seed,
            comparison_id=plan.comparison_id,
            optimizer_step=optimizer_step,
            sample_keys=tuple(transition.sample.sample_key for transition in transitions),
        )
        sampled = sample_temporal_batch_plan(
            config,
            seed=seed,
            state_ages=tuple(transition.transition_index for transition in transitions),
            available_future_steps=availability,
            optimizer_lags=(0,) * len(transitions),
        )
        for transition in transitions:
            episode = next(
                value for value in episodes if value.episode_key == transition.episode_key
            )
            expected.add(sources[transition.sample.sample_key])
            for offset in range(1, sampled.local_bptt_steps or 1):
                expected.add(sources[episode.sample_keys[transition.transition_index + offset]])
    expected_sources = tuple(sorted(expected))
    assert coverage.source_global_indices == expected_sources
    assert coverage.stream_plan_sha256 == plan.plan_sha256
    assert coverage.temporal_estimator_sha256 == config.digest
    assert len(coverage.source_keys_sha256) == 64
    assert len(coverage.coverage_sha256) == 64

    correction_only = build_native_current_grid_coverage_plan(
        plan,
        _config(source_mask_probability=0.0),
        source_global_index_for_sample=sources.__getitem__,
    )
    assert correction_only.source_global_indices

    with_egress = build_native_current_grid_coverage_plan(
        plan,
        _config(source_mask_probability=0.0),
        source_global_index_for_sample=sources.__getitem__,
        required_future_offsets=(1,),
    )
    expected_egress = set(correction_only.source_global_indices)
    for optimizer_step in range(plan.total_steps):
        for transition in plan.global_batch(optimizer_step).transitions:
            episode = next(
                value for value in episodes if value.episode_key == transition.episode_key
            )
            if transition.transition_index + 1 < len(episode.sample_keys):
                expected_egress.add(
                    sources[episode.sample_keys[transition.transition_index + 1]]
                )
    assert set(with_egress.source_global_indices) == expected_egress
