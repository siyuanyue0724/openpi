from __future__ import annotations

from pathlib import Path

from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinEpisode,
    CalvinLanguageSegment,
)
from picf_next.videomt_exact.calvin_full_dataset import (
    build_calvin_videomt_episode_split_plan,
    stateless_calvin_videomt_window,
)


def _index(tmp_path: Path) -> CalvinDatasetIndex:
    episodes = tuple(
        CalvinEpisode(index=index, start=index * 100, end=index * 100 + 19)
        for index in range(6)
    )
    segments = tuple(
        CalvinLanguageSegment(
            index=index,
            start=episode.start + 2,
            end=episode.start + 14,
            task_key=f"task_{index}",
            instruction=f"instruction {index}",
            episode_index=episode.index,
        )
        for index, episode in enumerate(episodes)
    )
    return CalvinDatasetIndex(
        split_root=tmp_path,
        dataset_id="calvin-fixture",
        dataset_revision="fixture-v1",
        control_hz=30,
        episodes=episodes,
        segments=segments,
    )


def test_episode_split_is_raw_episode_disjoint_and_compact(tmp_path: Path) -> None:
    plan = build_calvin_videomt_episode_split_plan(
        _index(tmp_path),
        clip_length=5,
        heldout_modulus=3,
        heldout_remainder=2,
    )

    assert plan.episode_indices("train") == (0, 1, 3, 4)
    assert plan.episode_indices("heldout") == (2, 5)
    assert set(plan.episode_indices("train")).isdisjoint(plan.episode_indices("heldout"))
    assert plan.window_count("train") == 4 * 8
    assert plan.window_count("heldout") == 2 * 8
    assert len(plan.fingerprint) == 64

    train_frames = {
        value
        for position in range(plan.window_count("train"))
        for value in plan.window_at("train", position)
    }
    heldout_frames = {
        value
        for position in range(plan.window_count("heldout"))
        for value in plan.window_at("heldout", position)
    }
    assert train_frames.isdisjoint(heldout_frames)


def test_stateless_sampler_is_one_permutation_per_epoch(tmp_path: Path) -> None:
    plan = build_calvin_videomt_episode_split_plan(
        _index(tmp_path),
        clip_length=5,
        heldout_modulus=3,
        heldout_remainder=2,
    )
    count = plan.window_count("train")
    first_epoch = tuple(
        stateless_calvin_videomt_window(
            plan,
            split="train",
            visit_index=visit,
            seed=20260822,
        )
        for visit in range(count)
    )
    second_epoch = tuple(
        stateless_calvin_videomt_window(
            plan,
            split="train",
            visit_index=count + visit,
            seed=20260822,
        )
        for visit in range(count)
    )

    assert len(set(first_epoch)) == count
    assert len(set(second_epoch)) == count
    assert set(first_epoch) == set(second_epoch)
    assert first_epoch != second_epoch
