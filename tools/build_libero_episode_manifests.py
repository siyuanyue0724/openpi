#!/usr/bin/env python3
"""Build frozen task-stratified LIBERO episode manifests from row identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from picf_next.data.robot_record import (
    MOLMOACT2_LIBERO_DATASET_ID,
    MOLMOACT2_LIBERO_REVISION,
)

try:
    from tools.audit_molmoact2_libero_full import (
        EXPECTED_EPISODE_TASK_STAT_MISMATCHES,
        EXPECTED_EPISODES,
        EXPECTED_TASKS,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from audit_molmoact2_libero_full import (
        EXPECTED_EPISODE_TASK_STAT_MISMATCHES,
        EXPECTED_EPISODES,
        EXPECTED_TASKS,
    )

SPLIT_SEED = "picf-next-libero-v1"
VALIDATION_FRACTION = 0.1
MIN_VALIDATION_EPISODES_PER_TASK = 2


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _episode_rank(episode_index: int, task_index: int, *, seed: str) -> str:
    payload = (f"{MOLMOACT2_LIBERO_REVISION}:{seed}:{task_index}:{episode_index}").encode()
    return hashlib.sha256(payload).hexdigest()


def build_episode_splits(
    episodes: Sequence[Mapping[str, Any]],
    task_by_index: Mapping[int, str],
    *,
    validation_fraction: float = VALIDATION_FRACTION,
    min_validation_per_task: int = MIN_VALIDATION_EPISODES_PER_TASK,
    seed: str = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split complete trajectories while retaining every task in both arms."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if min_validation_per_task <= 0 or not seed:
        raise ValueError("validation split controls must be positive and non-empty")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[int] = set()
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        task_index = int(episode["task_index"])
        if episode_index in seen:
            raise ValueError(f"duplicate episode_index {episode_index}")
        if task_index not in task_by_index:
            raise ValueError(f"episode {episode_index} has unknown task {task_index}")
        seen.add(episode_index)
        grouped[task_index].append(episode)
    if set(grouped) != set(task_by_index):
        raise ValueError("some tasks have no episodes")

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for task_index in sorted(grouped):
        candidates = sorted(
            grouped[task_index],
            key=lambda episode: _episode_rank(int(episode["episode_index"]), task_index, seed=seed),
        )
        if len(candidates) <= min_validation_per_task:
            raise ValueError(f"task {task_index} has too few episodes for a grouped split")
        validation_count = max(
            min_validation_per_task,
            int(math.floor(len(candidates) * validation_fraction + 0.5)),
        )
        validation_count = min(validation_count, len(candidates) - 1)
        validation_ids = {
            int(episode["episode_index"]) for episode in candidates[:validation_count]
        }
        for episode in grouped[task_index]:
            episode_index = int(episode["episode_index"])
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            length = int(episode["length"])
            if start < 0 or end - start != length or length <= 0:
                raise ValueError(f"episode {episode_index} has an invalid row range")
            record = {
                "dataset_id": MOLMOACT2_LIBERO_DATASET_ID,
                "dataset_revision": MOLMOACT2_LIBERO_REVISION,
                "episode_index": episode_index,
                "task_index": task_index,
                "task": task_by_index[task_index],
                "length": length,
                "dataset_from_index": start,
                "dataset_to_index": end,
                "split_rank_sha256": _episode_rank(episode_index, task_index, seed=seed),
            }
            (validation if episode_index in validation_ids else train).append(record)
    train.sort(key=lambda item: item["episode_index"])
    validation.sort(key=lambda item: item["episode_index"])
    return train, validation


def validate_episode_splits(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    *,
    expected_episodes: int,
    expected_tasks: int,
) -> dict[str, Any]:
    train_ids = {int(item["episode_index"]) for item in train}
    validation_ids = {int(item["episode_index"]) for item in validation}
    if len(train_ids) != len(train) or len(validation_ids) != len(validation):
        raise ValueError("split contains duplicate episodes")
    if train_ids & validation_ids:
        raise ValueError("train and validation episodes overlap")
    if train_ids | validation_ids != set(range(expected_episodes)):
        raise ValueError("split does not exactly cover the expected episode identity range")
    train_tasks = {int(item["task_index"]) for item in train}
    validation_tasks = {int(item["task_index"]) for item in validation}
    expected_task_ids = set(range(expected_tasks))
    if train_tasks != expected_task_ids or validation_tasks != expected_task_ids:
        raise ValueError("both split arms must contain every task")
    train_frames = sum(int(item["length"]) for item in train)
    validation_frames = sum(int(item["length"]) for item in validation)
    return {
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "train_frames": train_frames,
        "validation_frames": validation_frames,
        "tasks_each_arm": expected_tasks,
        "train_manifest_sha256": _canonical_sha256(train),
        "validation_manifest_sha256": _canonical_sha256(validation),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )


def attach_episode_task_identity(
    episodes: Sequence[Mapping[str, Any]], task_by_index: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Resolve task identity from immutable episode text, never file locators."""

    task_index_by_text = {text: index for index, text in task_by_index.items()}
    if len(task_index_by_text) != len(task_by_index):
        raise ValueError("task texts are not unique")
    resolved = []
    for episode in episodes:
        task_texts = episode.get("tasks")
        if not isinstance(task_texts, list) or len(task_texts) != 1:
            raise ValueError("each episode must contain exactly one task text")
        task = str(task_texts[0])
        if task not in task_index_by_text:
            raise ValueError(f"episode task is absent from task metadata: {task!r}")
        task_index = task_index_by_text[task]
        item = dict(episode)
        item["task_index"] = task_index
        resolved.append(item)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    dataset_root = args.dataset_root.resolve()
    tasks = pq.read_table(dataset_root / "meta/tasks.parquet").to_pylist()
    task_by_index = {int(item["task_index"]): str(item["task"]) for item in tasks}
    raw_episodes = pq.read_table(
        dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    episodes = attach_episode_task_identity(raw_episodes, task_by_index)
    episode_task_stat_mismatches = sum(
        episode.get("stats/task_index/min") != [resolved["task_index"]]
        or episode.get("stats/task_index/max") != [resolved["task_index"]]
        for episode, resolved in zip(raw_episodes, episodes, strict=True)
    )
    if episode_task_stat_mismatches != EXPECTED_EPISODE_TASK_STAT_MISMATCHES:
        raise ValueError("episode-level task summary mismatch count changed")
    episodes.sort(key=lambda item: int(item["episode_index"]))
    if len(episodes) != EXPECTED_EPISODES or len(task_by_index) != EXPECTED_TASKS:
        raise ValueError("dataset episode/task cardinality differs from the pinned revision")

    train, validation = build_episode_splits(episodes, task_by_index)
    summary = validate_episode_splits(
        train,
        validation,
        expected_episodes=EXPECTED_EPISODES,
        expected_tasks=EXPECTED_TASKS,
    )
    summary.update(
        {
            "schema": "picf-next.libero-episode-split.v1",
            "dataset_id": MOLMOACT2_LIBERO_DATASET_ID,
            "dataset_revision": MOLMOACT2_LIBERO_REVISION,
            "split_seed": SPLIT_SEED,
            "validation_fraction": VALIDATION_FRACTION,
            "min_validation_episodes_per_task": MIN_VALIDATION_EPISODES_PER_TASK,
            "source_episode_metadata_sha256": _canonical_sha256(raw_episodes),
            "source_task_metadata_sha256": _canonical_sha256(tasks),
            "locator_fields_used": False,
            "episode_task_stats_used": False,
            "episode_task_stat_mismatches": episode_task_stat_mismatches,
            "grouping_unit": "episode_index",
        }
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "train.jsonl", train)
    _write_jsonl(output / "validation.jsonl", validation)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
