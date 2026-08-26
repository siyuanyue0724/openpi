from __future__ import annotations

import copy

import pytest

from tools.build_libero_episode_manifests import (
    attach_episode_task_identity,
    build_episode_splits,
    validate_episode_splits,
)


def _fixture():
    tasks = {0: "task zero", 1: "task one"}
    episodes = []
    cursor = 0
    for episode_index in range(12):
        task_index = episode_index % 2
        length = 3 + episode_index
        episodes.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "length": length,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + length,
                # These known-defective upstream fields must never affect split.
                "data/chunk_index": 999,
                "data/file_index": 999,
            }
        )
        cursor += length
    return episodes, tasks


def test_episode_split_is_deterministic_grouped_complete_and_locator_independent() -> None:
    episodes, tasks = _fixture()
    before = copy.deepcopy(episodes)
    train, validation = build_episode_splits(
        episodes,
        tasks,
        validation_fraction=0.25,
        min_validation_per_task=1,
    )
    changed_locators = copy.deepcopy(episodes)
    for episode in changed_locators:
        episode["data/file_index"] = -123
    train_again, validation_again = build_episode_splits(
        changed_locators,
        tasks,
        validation_fraction=0.25,
        min_validation_per_task=1,
    )

    assert episodes == before
    assert train == train_again
    assert validation == validation_again
    summary = validate_episode_splits(
        train,
        validation,
        expected_episodes=12,
        expected_tasks=2,
    )
    assert summary["train_episodes"] + summary["validation_episodes"] == 12
    assert summary["tasks_each_arm"] == 2
    train_ids = {row["episode_index"] for row in train}
    validation_ids = {row["episode_index"] for row in validation}
    assert not (train_ids & validation_ids)


def test_episode_split_fails_closed_on_duplicates_ranges_and_incomplete_coverage() -> None:
    episodes, tasks = _fixture()
    duplicate = copy.deepcopy(episodes)
    duplicate[-1]["episode_index"] = duplicate[0]["episode_index"]
    with pytest.raises(ValueError, match="duplicate"):
        build_episode_splits(duplicate, tasks, min_validation_per_task=1)

    bad_range = copy.deepcopy(episodes)
    bad_range[0]["dataset_to_index"] += 1
    with pytest.raises(ValueError, match="invalid row range"):
        build_episode_splits(bad_range, tasks, min_validation_per_task=1)

    train, validation = build_episode_splits(episodes, tasks, min_validation_per_task=1)
    with pytest.raises(ValueError, match="exactly cover"):
        validate_episode_splits(
            train[:-1],
            validation,
            expected_episodes=12,
            expected_tasks=2,
        )


def test_episode_task_identity_uses_unique_text_not_stale_summary_stats() -> None:
    episode = {
        "episode_index": 0,
        "tasks": ["task zero"],
        "stats/task_index/min": [0],
        "stats/task_index/max": [0],
    }
    resolved = attach_episode_task_identity([episode], {0: "task zero"})
    assert resolved[0]["task_index"] == 0
    assert "task_index" not in episode

    stale = dict(episode)
    stale["stats/task_index/min"] = [9]
    stale["stats/task_index/max"] = [9]
    assert attach_episode_task_identity([stale], {0: "task zero"})[0]["task_index"] == 0
