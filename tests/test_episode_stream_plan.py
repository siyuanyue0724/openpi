from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from picf_next.training.control import (
    EpisodeSampleSequence,
    FrozenEpisodeStreamPlan,
    FrozenResetMixtureStreamPlan,
)


def _episodes() -> tuple[EpisodeSampleSequence, ...]:
    return (
        EpisodeSampleSequence("episode-a", tuple(f"episode-a/frame-{i}" for i in range(5))),
        EpisodeSampleSequence("episode-b", tuple(f"episode-b/frame-{i}" for i in range(2))),
        EpisodeSampleSequence("episode-c", tuple(f"episode-c/frame-{i}" for i in range(4))),
        EpisodeSampleSequence("episode-d", tuple(f"episode-d/frame-{i}" for i in range(3))),
    )


def _plan(
    *,
    global_batch_size: int = 2,
    total_steps: int = 30,
    lane_interleave_factor: int = 1,
    episodes: tuple[EpisodeSampleSequence, ...] | None = None,
) -> FrozenEpisodeStreamPlan:
    return FrozenEpisodeStreamPlan(
        dataset_id="episode-stream-fixture",
        dataset_revision="v1",
        dataset_manifest_sha256="a" * 64,
        episodes=_episodes() if episodes is None else episodes,
        comparison_id="episode-stream-seed-43",
        seed=43,
        global_batch_size=global_batch_size,
        total_steps=total_steps,
        lane_interleave_factor=lane_interleave_factor,
    )


def _interleaved_episodes() -> tuple[EpisodeSampleSequence, ...]:
    return tuple(
        EpisodeSampleSequence(
            f"episode-{episode_index:02d}",
            tuple(
                f"episode-{episode_index:02d}/frame-{frame_index}"
                for frame_index in range(2 + episode_index % 4)
            ),
        )
        for episode_index in range(24)
    )


def _reset_mixture_plan(
    *,
    total_steps: int = 8,
    lane_interleave_factor: int = 2,
    episode_count: int = 32,
) -> FrozenResetMixtureStreamPlan:
    episodes = tuple(
        EpisodeSampleSequence(
            f"reset-episode-{episode_index:03d}",
            tuple(
                f"reset-episode-{episode_index:03d}/frame-{frame_index}" for frame_index in range(4)
            ),
        )
        for episode_index in range(episode_count)
    )
    causal = _plan(
        total_steps=total_steps // 2,
        lane_interleave_factor=lane_interleave_factor,
        episodes=episodes,
    )
    causal_keys = {
        transition.sample.sample_key
        for step in range(causal.total_steps)
        for transition in causal.global_batch(step).transitions
    }
    reset_keys = tuple(
        episode.sample_keys[0] for episode in episodes if episode.sample_keys[0] not in causal_keys
    )[:total_steps]
    return FrozenResetMixtureStreamPlan(
        causal_plan=causal,
        reset_sample_keys=reset_keys,
        reset_source_global_indices=tuple(range(10_000, 10_000 + len(reset_keys))),
        total_steps=total_steps,
    )


def test_reset_mixture_is_exact_stateless_first_and_topology_independent() -> None:
    plan = _reset_mixture_plan()
    assert [plan.component_for_step(step) for step in range(plan.total_steps)] == [
        "reset",
        "causal",
        "reset",
        "causal",
        "reset",
        "causal",
        "reset",
        "causal",
    ]
    assert plan.reset_step_count == plan.causal_step_count == 4
    assert plan.reset_sample_count == 8
    assert not plan.posterior_committed_for_step(0)
    assert plan.posterior_committed_for_step(1)
    assert plan.global_batch(1).transitions == plan.causal_plan.global_batch(0).transitions
    assert plan.global_batch(3).transitions == plan.causal_plan.global_batch(1).transitions

    reset_keys = []
    causal_keys = []
    for step in range(plan.total_steps):
        batch = plan.global_batch(step)
        assert batch.optimizer_step == step
        keys = [item.sample.sample_key for item in batch.transitions]
        if plan.component_for_step(step) == "reset":
            assert all(item.transition_index == 0 for item in batch.transitions)
            reset_keys.extend(keys)
        else:
            causal_keys.extend(keys)
        shards = tuple(
            plan.microbatch_for_rank(
                step,
                rank=rank,
                world_size=2,
                gradient_accumulation_steps=1,
                accumulation_index=0,
            )
            for rank in range(2)
        )
        assert tuple(item for shard in shards for item in shard.transitions) == batch.transitions
    assert tuple(reset_keys) == plan.reset_sample_keys
    assert len(reset_keys) == len(set(reset_keys))
    assert set(reset_keys).isdisjoint(causal_keys)


def test_reset_mixture_random_access_round_trip_and_hash_mutation(tmp_path: Path) -> None:
    plan = _reset_mixture_plan()
    ascending = tuple(plan.global_batch(step) for step in range(plan.total_steps))
    descending = {step: plan.global_batch(step) for step in reversed(range(plan.total_steps))}
    assert ascending == tuple(descending[step] for step in range(plan.total_steps))

    path = tmp_path / "reset-mixture.json"
    plan.write_metadata(path)
    restored = FrozenResetMixtureStreamPlan.from_metadata(
        path,
        episodes=plan.episodes,
    )
    assert restored == plan
    assert restored.plan_sha256 == plan.plan_sha256

    payload = json.loads(path.read_text())
    payload["metadata"]["reset_sample_keys"][0] = plan.episodes[0].sample_keys[1]
    payload["plan_sha256"] = hashlib.sha256(
        json.dumps(
            payload["metadata"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="non-transition-zero"):
        FrozenResetMixtureStreamPlan.from_metadata(path, episodes=plan.episodes)


def test_reset_mixture_recurrent_corrections_move_to_updates_18_and_34() -> None:
    plan = _reset_mixture_plan(
        total_steps=36,
        lane_interleave_factor=8,
        episode_count=96,
    )
    correction_updates = [
        step + 1
        for step in range(plan.total_steps)
        if plan.component_for_step(step) == "causal"
        and plan.component_index_for_step(step) in {8, 16}
    ]
    assert correction_updates == [18, 34]


def test_each_lane_reads_only_the_next_transition_and_resets_at_episode_boundary() -> None:
    plan = _plan()
    expected_samples = {}
    sample_index = 0
    for episode in plan.episodes:
        for transition_index, sample_key in enumerate(episode.sample_keys):
            expected_samples[sample_key] = (
                episode.episode_key,
                transition_index,
                sample_index,
            )
            sample_index += 1
    previous_by_lane = {}
    occurrence_lanes: dict[str, set[str]] = {}
    occurrence_indices: dict[str, list[int]] = {}

    for step in range(plan.total_steps):
        batch = plan.global_batch(step)
        assert len(batch.transitions) == plan.global_batch_size
        for transition in batch.transitions:
            previous = previous_by_lane.get(transition.lane_id)
            if (
                previous is not None
                and previous.episode_instance_id == transition.episode_instance_id
            ):
                assert transition.transition_index == previous.transition_index + 1
            elif previous is not None:
                assert transition.transition_index == 0
            assert expected_samples[transition.sample.sample_key] == (
                transition.episode_key,
                transition.transition_index,
                transition.sample.sample_index,
            )
            previous_by_lane[transition.lane_id] = transition
            occurrence_lanes.setdefault(transition.episode_instance_id, set()).add(
                transition.lane_id
            )
            occurrence_indices.setdefault(transition.episode_instance_id, []).append(
                transition.transition_index
            )

    assert all(len(lanes) == 1 for lanes in occurrence_lanes.values())
    assert all(indices == list(range(len(indices))) for indices in occurrence_indices.values())


def test_source_sample_and_randomness_are_stable_under_random_access() -> None:
    plan = _plan(total_steps=40)
    ascending = {step: plan.global_batch(step) for step in range(plan.total_steps)}
    descending = {step: plan.global_batch(step) for step in reversed(range(plan.total_steps))}
    assert ascending == descending

    seeds_by_source: dict[str, set[int]] = {}
    instances_by_source: dict[str, set[str]] = {}
    for batch in ascending.values():
        for transition in batch.transitions:
            seeds_by_source.setdefault(transition.sample.sample_key, set()).add(
                transition.sample.augmentation_seed
            )
            instances_by_source.setdefault(transition.sample.sample_key, set()).add(
                transition.episode_instance_id
            )
    repeated_sources = [key for key, instances in instances_by_source.items() if len(instances) > 1]
    assert repeated_sources
    assert all(
        len(seeds_by_source[key]) == len(instances_by_source[key]) for key in repeated_sources
    )


def test_interleaving_rotates_lanes_without_changing_active_global_batch_size() -> None:
    plan = _plan(
        total_steps=20,
        lane_interleave_factor=4,
        episodes=_interleaved_episodes(),
    )
    assert plan.lane_count == 8
    assert len(plan.lane_ids) == 8
    assert [
        tuple(transition.lane_id for transition in plan.global_batch(step).transitions)
        for step in range(5)
    ] == [
        ("global-lane-00000", "global-lane-00004"),
        ("global-lane-00001", "global-lane-00005"),
        ("global-lane-00002", "global-lane-00006"),
        ("global-lane-00003", "global-lane-00007"),
        ("global-lane-00000", "global-lane-00004"),
    ]
    assert all(
        len(plan.global_batch(step).transitions) == plan.global_batch_size
        for step in range(plan.total_steps)
    )


def test_interleaved_lanes_advance_one_real_frame_only_when_revisited() -> None:
    factor = 4
    plan = _plan(
        total_steps=80,
        lane_interleave_factor=factor,
        episodes=_interleaved_episodes(),
    )
    previous_by_lane = {}
    previous_step_by_lane = {}
    saw_continuation = False
    saw_reset = False
    for step in range(plan.total_steps):
        for transition in plan.global_batch(step).transitions:
            previous = previous_by_lane.get(transition.lane_id)
            if previous is not None:
                assert step - previous_step_by_lane[transition.lane_id] == factor
                if previous.episode_instance_id == transition.episode_instance_id:
                    assert transition.transition_index == previous.transition_index + 1
                    saw_continuation = True
                else:
                    assert transition.transition_index == 0
                    saw_reset = True
            previous_by_lane[transition.lane_id] = transition
            previous_step_by_lane[transition.lane_id] = step
    assert saw_continuation
    assert saw_reset


def test_preregistered_k8_preserves_sample_budget_and_exposes_sixteen_contexts() -> None:
    plan = _plan(
        total_steps=200,
        lane_interleave_factor=8,
        episodes=_interleaved_episodes(),
    )
    first_window = tuple(
        transition for step in range(8) for transition in plan.global_batch(step).transitions
    )
    assert len(first_window) == 16
    assert len({transition.lane_id for transition in first_window}) == 16
    assert len({transition.episode_instance_id for transition in first_window}) == 16
    assert all(transition.transition_index == 0 for transition in first_window)

    all_transitions = tuple(
        transition
        for step in range(plan.total_steps)
        for transition in plan.global_batch(step).transitions
    )
    assert len(all_transitions) == 400
    assert {
        lane_id: sum(transition.lane_id == lane_id for transition in all_transitions)
        for lane_id in plan.lane_ids
    } == dict.fromkeys(plan.lane_ids, 25)


def test_rank_and_accumulation_shards_reconstruct_the_global_lane_order() -> None:
    plan = _plan(global_batch_size=4)
    for step in (0, 1, 7, 19):
        reconstructed = []
        for accumulation_index in range(2):
            for rank in range(2):
                microbatch = plan.microbatch_for_rank(
                    step,
                    rank=rank,
                    world_size=2,
                    gradient_accumulation_steps=2,
                    accumulation_index=accumulation_index,
                )
                reconstructed.extend(microbatch.transitions)
        assert tuple(reconstructed) == plan.global_batch(step).transitions


def test_plan_metadata_roundtrip_fails_closed_on_episode_manifest_change(
    tmp_path: Path,
) -> None:
    plan = _plan()
    path = tmp_path / "stream-plan.json"
    plan.write_metadata(path)
    restored = FrozenEpisodeStreamPlan.from_metadata(path, episodes=_episodes())
    assert restored.plan_sha256 == plan.plan_sha256
    assert restored.global_batch(17) == plan.global_batch(17)

    changed = list(_episodes())
    changed[0] = EpisodeSampleSequence(
        "episode-a",
        tuple(reversed(changed[0].sample_keys)),
    )
    with pytest.raises(ValueError, match="episode manifest differs"):
        FrozenEpisodeStreamPlan.from_metadata(path, episodes=changed)

    payload = json.loads(path.read_text())
    payload["metadata"]["total_steps"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        FrozenEpisodeStreamPlan.from_metadata(path, episodes=_episodes())


def test_legacy_plan_identity_and_interleaved_metadata_roundtrip(tmp_path: Path) -> None:
    legacy = _plan()
    assert legacy.plan_sha256 == (
        "007ec374d6376a18899897e949da6e6ffb5a23e40a0eee769bb6617881e498cb"
    )
    assert legacy.metadata["schema"] == "picf-next.frozen-episode-stream-plan.v1"
    assert "lane_interleave_factor" not in legacy.metadata

    interleaved = _plan(
        total_steps=40,
        lane_interleave_factor=4,
        episodes=_interleaved_episodes(),
    )
    path = tmp_path / "interleaved-stream-plan.json"
    interleaved.write_metadata(path)
    restored = FrozenEpisodeStreamPlan.from_metadata(
        path,
        episodes=_interleaved_episodes(),
    )
    assert restored == interleaved
    assert restored.global_batch(39) == interleaved.global_batch(39)
    assert restored.metadata["lane_count"] == 8
    assert restored.metadata["lane_interleave_factor"] == 4

    payload = json.loads(path.read_text())
    payload["metadata"]["lane_count"] = 9
    payload["plan_sha256"] = hashlib.sha256(
        json.dumps(
            payload["metadata"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="lane count differs"):
        FrozenEpisodeStreamPlan.from_metadata(
            path,
            episodes=_interleaved_episodes(),
        )


def test_episode_stream_plan_rejects_ambiguous_or_underfilled_manifests() -> None:
    with pytest.raises(ValueError, match="initially fill"):
        FrozenEpisodeStreamPlan(
            dataset_id="fixture",
            dataset_revision="v1",
            dataset_manifest_sha256="a" * 64,
            episodes=_episodes()[:1],
            comparison_id="comparison",
            seed=1,
            global_batch_size=2,
            total_steps=2,
        )

    duplicate_sample = (
        EpisodeSampleSequence("episode-a", ("shared",)),
        EpisodeSampleSequence("episode-b", ("shared",)),
    )
    with pytest.raises(ValueError, match="globally unique"):
        FrozenEpisodeStreamPlan(
            dataset_id="fixture",
            dataset_revision="v1",
            dataset_manifest_sha256="a" * 64,
            episodes=duplicate_sample,
            comparison_id="comparison",
            seed=1,
            global_batch_size=2,
            total_steps=2,
        )

    with pytest.raises(ValueError, match="unique within"):
        EpisodeSampleSequence("episode-a", ("same", "same"))

    with pytest.raises(ValueError, match="cannot exceed"):
        _plan(
            total_steps=3,
            lane_interleave_factor=4,
            episodes=_interleaved_episodes(),
        )


@pytest.mark.parametrize(
    ("rank", "world_size", "accumulation_steps", "accumulation_index"),
    [(-1, 2, 1, 0), (2, 2, 1, 0), (0, 0, 1, 0), (0, 2, 0, 0), (0, 2, 1, 1)],
)
def test_stream_microbatch_rejects_invalid_partition_coordinates(
    rank: int,
    world_size: int,
    accumulation_steps: int,
    accumulation_index: int,
) -> None:
    with pytest.raises(ValueError):
        _plan().microbatch_for_rank(
            0,
            rank=rank,
            world_size=world_size,
            gradient_accumulation_steps=accumulation_steps,
            accumulation_index=accumulation_index,
        )
