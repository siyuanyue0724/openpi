from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from picf_next.eval.m2_protocol import (  # noqa: E402
    group_by_target_count,
    language_samples,
    task_intervention_check,
    unique_source_keys,
)


def _cache_row(
    *,
    sample_key: str,
    segment_index: int,
    transition_index: int,
    global_index: int,
    token_value: float,
    sensor_digest: str = "a" * 64,
) -> tuple[Any, Any, dict[str, Any]]:
    return (
        torch.full((3, 2), token_value, dtype=torch.bfloat16),
        torch.ones(3, dtype=torch.bool),
        {
            "sample_key": sample_key,
            "segment_index": segment_index,
            "transition_index": transition_index,
            "global_index": global_index,
            "source_sensor_sha256": [["rgb", sensor_digest]],
        },
    )


def test_unique_source_keys_remove_overlapping_language_annotations_stably() -> None:
    cache = {
        "later-task": _cache_row(
            sample_key="later-task",
            segment_index=4,
            transition_index=1,
            global_index=101,
            token_value=2.0,
        ),
        "first-task": _cache_row(
            sample_key="first-task",
            segment_index=1,
            transition_index=0,
            global_index=101,
            token_value=2.0,
        ),
        "next-frame": _cache_row(
            sample_key="next-frame",
            segment_index=1,
            transition_index=1,
            global_index=102,
            token_value=3.0,
        ),
    }

    assert unique_source_keys(cache) == ["first-task", "next-frame"]


def test_language_samples_reuse_dataset_transition_boundaries() -> None:
    samples = [
        SimpleNamespace(
            sample_key="second",
            transition_index=1,
            record=SimpleNamespace(task_index=2),
        ),
        SimpleNamespace(
            sample_key="first",
            transition_index=0,
            record=SimpleNamespace(task_index=1),
        ),
    ]

    assert [sample.sample_key for sample in language_samples(samples)] == [
        "first",
        "second",
    ]


def test_task_intervention_requires_identical_dense_features_for_same_source() -> None:
    cache = {
        "task-a": _cache_row(
            sample_key="task-a",
            segment_index=0,
            transition_index=0,
            global_index=100,
            token_value=1.0,
        ),
        "task-b": _cache_row(
            sample_key="task-b",
            segment_index=1,
            transition_index=0,
            global_index=100,
            token_value=1.0,
        ),
    }

    report = task_intervention_check(cache)

    assert report["pair_count"] == 1
    assert report["all_dense_features_exact"] is True
    assert report["maximum_absolute_error"] == 0.0
    assert report["task_text_enters_trainable_m2_graph"] is False

    changed = dict(cache)
    changed["task-b"] = _cache_row(
        sample_key="task-b",
        segment_index=1,
        transition_index=0,
        global_index=100,
        token_value=1.5,
    )
    with pytest.raises(RuntimeError, match="non-exact"):
        task_intervention_check(changed)


def test_target_count_strata_preserve_hard_count_and_representation_metrics() -> None:
    rows = [
        {
            "target_object_count": 7,
            "predicted_object_count": 6,
            "exact_count": False,
            "mean_object_dice": 0.4,
            "ownership_accuracy": 0.6,
        },
        {
            "target_object_count": 7,
            "predicted_object_count": 7,
            "exact_count": True,
            "mean_object_dice": 0.8,
            "ownership_accuracy": 1.0,
        },
        {
            "target_object_count": 10,
            "predicted_object_count": 8,
            "exact_count": False,
            "mean_object_dice": 0.5,
            "ownership_accuracy": 0.7,
        },
    ]

    assert group_by_target_count(rows) == {
        "7": {
            "sample_count": 2,
            "predicted_count_mean": 6.5,
            "count_mae": 0.5,
            "exact_count_accuracy": 0.5,
            "mean_object_dice": pytest.approx(0.6),
            "ownership_accuracy": pytest.approx(0.8),
        },
        "10": {
            "sample_count": 1,
            "predicted_count_mean": 8.0,
            "count_mae": 2.0,
            "exact_count_accuracy": 0.0,
            "mean_object_dice": 0.5,
            "ownership_accuracy": 0.7,
        },
    }
