from __future__ import annotations

import pytest
import torch

from tools.audit_flare_target_identifiability import (
    _adjacent_pairs,
    _temporal_pairs,
    summarize_normalized_targets,
)


def test_sample_independent_template_is_exact_for_identical_targets() -> None:
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = target.unsqueeze(0).repeat(3, 1, 1)

    summary = summarize_normalized_targets(
        targets,
        current_future_pairs=((0, 1), (1, 2)),
    )

    fixed = summary["optimal_sample_independent_position_template"]
    assert fixed["mean_cosine"] == pytest.approx(1.0)
    assert fixed["raw_cosine_loss"] == pytest.approx(0.0)
    assert summary["cached_current_to_t_plus_h_cosine"]["mean"] == pytest.approx(1.0)


def test_sample_independent_template_exposes_opposite_target_ambiguity() -> None:
    targets = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )

    summary = summarize_normalized_targets(targets, current_future_pairs=())

    fixed = summary["optimal_sample_independent_position_template"]
    assert fixed["mean_cosine"] == pytest.approx(0.0)
    assert fixed["raw_cosine_loss"] == pytest.approx(1.0)


def test_temporal_pairs_use_global_manifest_order_not_shard_local_rows() -> None:
    records = [
        {
            "sample_key": "episode-a/frame-00000000",
            "source_global_index": 0,
            "future_global_index": 16,
            "shard": 0,
            "row": 0,
        },
        {
            "sample_key": "episode-a/frame-00000016",
            "source_global_index": 16,
            "future_global_index": 32,
            "shard": 0,
            "row": 1,
        },
        {
            "sample_key": "episode-a/frame-00000032",
            "source_global_index": 32,
            "future_global_index": 48,
            "shard": 1,
            "row": 0,
        },
    ]

    assert _temporal_pairs(records) == [(0, 1), (1, 2)]


def test_adjacent_pairs_are_same_episode_and_one_source_frame_apart() -> None:
    records = [
        {
            "sample_key": "episode-a/frame-00000010",
            "source_global_index": 10,
            "future_global_index": 26,
        },
        {
            "sample_key": "episode-a/frame-00000011",
            "source_global_index": 11,
            "future_global_index": 27,
        },
        {
            "sample_key": "episode-b/frame-00000012",
            "source_global_index": 12,
            "future_global_index": 28,
        },
    ]

    assert _adjacent_pairs(records) == [(0, 1)]
