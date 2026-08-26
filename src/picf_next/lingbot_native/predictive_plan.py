"""Content-addressed predictive-target coverage for one frozen training plan.

The offline teacher encodes only source/horizon pairs that the registered
training estimator can consume.  Coverage is derived from source identity,
episode availability and frozen RNG; labels, masks, task text, model outputs
and current losses are deliberately absent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from picf_next.contracts import ContractError
from picf_next.lingbot_native.current_grid_cache import (
    current_grid_coverage_digest,
    current_grid_source_keys_digest,
)
from picf_next.lingbot_native.predictive_cache import (
    native_predictive_coverage_digest,
    native_predictive_pair_keys_digest,
)
from picf_next.lingbot_native.temporal import (
    TemporalEstimatorConfig,
    native_temporal_batch_seed,
    sample_temporal_batch_plan,
)
from picf_next.training.control import (
    EpisodeStreamPlan,
    FrozenEpisodeStreamPlan,
    FrozenResetMixtureStreamPlan,
)


def _sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class NativePredictiveCoveragePlan:
    """Exact controlled-rollout target keys consumed by one training plan."""

    dataset_tree_sha256: str
    stream_plan_sha256: str
    temporal_estimator_sha256: str
    horizons: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _sha256(self.dataset_tree_sha256, "predictive dataset tree sha256")
        _sha256(self.stream_plan_sha256, "predictive stream plan sha256")
        _sha256(self.temporal_estimator_sha256, "predictive temporal estimator sha256")
        if (
            not self.horizons
            or tuple(sorted(set(self.horizons))) != self.horizons
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.horizons
            )
        ):
            raise ContractError("predictive plan horizons must be sorted unique positive integers")
        if not self.pairs:
            raise ContractError("predictive training plan requests no future targets")
        if tuple(sorted(set(self.pairs))) != self.pairs:
            raise ContractError("predictive plan pairs must be sorted and unique")
        if any(
            isinstance(source, bool)
            or not isinstance(source, int)
            or source < 0
            or isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon not in self.horizons
            for source, horizon in self.pairs
        ):
            raise ContractError("predictive plan pair lies outside its source/horizon contract")

    @property
    def pair_keys_sha256(self) -> str:
        return native_predictive_pair_keys_digest(self.pairs)

    @property
    def coverage_sha256(self) -> str:
        return native_predictive_coverage_digest(
            dataset_tree_sha256=self.dataset_tree_sha256,
            stream_plan_sha256=self.stream_plan_sha256,
            temporal_estimator_sha256=self.temporal_estimator_sha256,
            pair_keys_sha256=self.pair_keys_sha256,
            expected_record_count=len(self.pairs),
            horizons=self.horizons,
        )


@dataclass(frozen=True, slots=True)
class NativeCurrentGridCoveragePlan:
    """Exact current frames consumed by correction and sampled source objectives."""

    dataset_tree_sha256: str
    stream_plan_sha256: str
    temporal_estimator_sha256: str
    source_global_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _sha256(self.dataset_tree_sha256, "current-grid dataset tree sha256")
        _sha256(self.stream_plan_sha256, "current-grid stream plan sha256")
        _sha256(self.temporal_estimator_sha256, "current-grid temporal estimator sha256")
        if (
            not self.source_global_indices
            or tuple(sorted(set(self.source_global_indices))) != self.source_global_indices
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.source_global_indices
            )
        ):
            raise ContractError("current-grid sources must be sorted unique non-negative indices")

    @property
    def source_keys_sha256(self) -> str:
        return current_grid_source_keys_digest(self.source_global_indices)

    @property
    def coverage_sha256(self) -> str:
        return current_grid_coverage_digest(
            dataset_tree_sha256=self.dataset_tree_sha256,
            stream_plan_sha256=self.stream_plan_sha256,
            temporal_estimator_sha256=self.temporal_estimator_sha256,
            source_keys_sha256=self.source_keys_sha256,
            expected_record_count=len(self.source_global_indices),
        )


def build_native_predictive_coverage_plan(
    stream_plan: EpisodeStreamPlan,
    temporal_config: TemporalEstimatorConfig,
    *,
    source_global_index_for_sample: Callable[[str], int],
    required_horizons: tuple[int, ...] = (),
) -> NativePredictiveCoveragePlan:
    """Resolve sampled overshoot plus every available objective-mandatory key.

    A source at the end of a bounded episode has no same-episode future target.
    Such a source remains valid for current-frame objectives but contributes no
    predictive pair; predictive validity is part of the estimator rather than
    a reason to fabricate or cross a reset boundary.
    """

    if not isinstance(
        stream_plan,
        (FrozenEpisodeStreamPlan, FrozenResetMixtureStreamPlan),
    ):
        raise TypeError("predictive coverage requires a frozen episode stream plan")
    if not isinstance(temporal_config, TemporalEstimatorConfig):
        raise TypeError("predictive coverage requires a frozen temporal estimator config")
    if not callable(source_global_index_for_sample):
        raise TypeError("predictive coverage requires a source-index resolver")
    if (
        not isinstance(required_horizons, tuple)
        or tuple(sorted(set(required_horizons))) != required_horizons
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in temporal_config.overshoot_horizons
            for value in required_horizons
        )
    ):
        raise ContractError(
            "required predictive horizons must be a sorted unique subset of the estimator"
        )

    episodes = {episode.episode_key: episode for episode in stream_plan.episodes}
    pairs: set[tuple[int, int]] = set()
    for optimizer_step in range(stream_plan.total_steps):
        transitions = stream_plan.global_batch(optimizer_step).transitions
        availability: list[int] = []
        sources: list[int] = []
        for transition in transitions:
            episode = episodes.get(transition.episode_key)
            if episode is None:
                raise ContractError("stream transition references an absent episode")
            if episode.sample_keys[transition.transition_index] != transition.sample.sample_key:
                raise ContractError("stream transition and episode sample identity disagree")
            available = len(episode.sample_keys) - transition.transition_index - 1
            availability.append(available)
            source = source_global_index_for_sample(transition.sample.sample_key)
            if isinstance(source, bool) or not isinstance(source, int) or source < 0:
                raise ContractError("source-index resolver returned an invalid global index")
            sources.append(source)

        seed = native_temporal_batch_seed(
            parent_seed=stream_plan.seed,
            comparison_id=stream_plan.comparison_id,
            optimizer_step=optimizer_step,
            sample_keys=tuple(value.sample.sample_key for value in transitions),
        )
        temporal = sample_temporal_batch_plan(
            temporal_config,
            seed=seed,
            state_ages=tuple(value.transition_index for value in transitions),
            available_future_steps=tuple(availability),
            optimizer_lags=(0,) * len(transitions),
        )
        for source, available in zip(sources, availability, strict=True):
            if temporal.overshoot_horizon is not None:
                pairs.add((source, temporal.overshoot_horizon))
            for horizon in required_horizons:
                if available >= horizon:
                    pairs.add((source, horizon))

    return NativePredictiveCoveragePlan(
        dataset_tree_sha256=stream_plan.dataset_manifest_sha256,
        stream_plan_sha256=stream_plan.plan_sha256,
        temporal_estimator_sha256=temporal_config.digest,
        horizons=temporal_config.overshoot_horizons,
        pairs=tuple(sorted(pairs)),
    )


def build_native_current_grid_coverage_plan(
    stream_plan: EpisodeStreamPlan,
    temporal_config: TemporalEstimatorConfig,
    *,
    source_global_index_for_sample: Callable[[str], int],
    required_future_offsets: tuple[int, ...] = (),
) -> NativeCurrentGridCoveragePlan:
    """Resolve every primary, local, and mandatory egress target frame."""

    if not isinstance(
        stream_plan,
        (FrozenEpisodeStreamPlan, FrozenResetMixtureStreamPlan),
    ):
        raise TypeError("current-grid coverage requires a frozen episode stream plan")
    if not isinstance(temporal_config, TemporalEstimatorConfig):
        raise TypeError("current-grid coverage requires a frozen temporal estimator config")
    if not callable(source_global_index_for_sample):
        raise TypeError("current-grid coverage requires a source-index resolver")
    if (
        not isinstance(required_future_offsets, tuple)
        or tuple(sorted(set(required_future_offsets))) != required_future_offsets
        or any(
            isinstance(offset, bool) or not isinstance(offset, int) or offset <= 0
            for offset in required_future_offsets
        )
    ):
        raise ContractError(
            "required current-grid future offsets must be sorted unique positive integers"
        )

    episodes = {episode.episode_key: episode for episode in stream_plan.episodes}
    sources: set[int] = set()
    for optimizer_step in range(stream_plan.total_steps):
        transitions = stream_plan.global_batch(optimizer_step).transitions
        availability: list[int] = []
        for transition in transitions:
            episode = episodes.get(transition.episode_key)
            if episode is None:
                raise ContractError("stream transition references an absent episode")
            if episode.sample_keys[transition.transition_index] != transition.sample.sample_key:
                raise ContractError("stream transition and episode sample identity disagree")
            availability.append(len(episode.sample_keys) - transition.transition_index - 1)
            source = source_global_index_for_sample(transition.sample.sample_key)
            if isinstance(source, bool) or not isinstance(source, int) or source < 0:
                raise ContractError("source-index resolver returned an invalid current-grid index")
            sources.add(source)
            for offset in required_future_offsets:
                if availability[-1] < offset:
                    continue
                future_key = episode.sample_keys[transition.transition_index + offset]
                future_source = source_global_index_for_sample(future_key)
                if (
                    isinstance(future_source, bool)
                    or not isinstance(future_source, int)
                    or future_source < 0
                ):
                    raise ContractError(
                        "source-index resolver returned an invalid mandatory egress index"
                    )
                sources.add(future_source)
        seed = native_temporal_batch_seed(
            parent_seed=stream_plan.seed,
            comparison_id=stream_plan.comparison_id,
            optimizer_step=optimizer_step,
            sample_keys=tuple(value.sample.sample_key for value in transitions),
        )
        temporal = sample_temporal_batch_plan(
            temporal_config,
            seed=seed,
            state_ages=tuple(value.transition_index for value in transitions),
            available_future_steps=tuple(availability),
            optimizer_lags=(0,) * len(transitions),
        )
        if temporal.local_bptt_steps is not None:
            for transition in transitions:
                episode = episodes[transition.episode_key]
                for offset in range(1, temporal.local_bptt_steps):
                    local_key = episode.sample_keys[transition.transition_index + offset]
                    local_source = source_global_index_for_sample(local_key)
                    if (
                        isinstance(local_source, bool)
                        or not isinstance(local_source, int)
                        or local_source < 0
                    ):
                        raise ContractError(
                            "source-index resolver returned an invalid local-BPTT index"
                        )
                    sources.add(local_source)
    if not sources:
        raise ContractError("frozen training plan sampled no current-grid branch")
    return NativeCurrentGridCoveragePlan(
        dataset_tree_sha256=stream_plan.dataset_manifest_sha256,
        stream_plan_sha256=stream_plan.plan_sha256,
        temporal_estimator_sha256=temporal_config.digest,
        source_global_indices=tuple(sorted(sources)),
    )
