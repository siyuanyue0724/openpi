from __future__ import annotations

import pytest

from picf_next.lingbot_native.representation_row_permutation import (
    audit_row_permutation_sample,
)


def _row(row: int, identity: str, prediction: list[float], target: list[float]) -> dict:
    intersection = sum(a * b for a, b in zip(prediction, target, strict=True))
    union = sum(a + b - a * b for a, b in zip(prediction, target, strict=True))
    return {
        "row_index": row,
        "identity_key": identity,
        "prediction": prediction,
        "target": target,
        "weight": [1.0, 1.0],
        "soft_iou": intersection / union,
    }


def test_row_permutation_oracle_recovers_swapped_identity_rows() -> None:
    sample = {
        "partition": "heldout",
        "task_key": "find-a",
        "sample_key": "sample-a",
        "factual_target_identity_keys": ["a"],
        "factual_ownership_rows": [
            _row(2, "a", [0.0, 1.0], [1.0, 0.0]),
            _row(5, "b", [1.0, 0.0], [0.0, 1.0]),
        ],
        "factual_task_row_diagnostic": {
            "exact_task": True,
            "task_logits": [-3.0, -3.0, -2.0, -3.0, -3.0, 4.0],
            "row_task_valid": [True] * 6,
            "target_vs_hardest_negative_logit_margin": -6.0,
            "all_targets_beat_known_negatives": False,
        },
    }

    result = audit_row_permutation_sample(sample)

    assert result is not None
    assert result["changed_identity_count"] == 2
    assert result["assigned_macro_soft_iou"] == 0.0
    assert result["oracle_macro_soft_iou"] == 1.0
    assert result["oracle_target_soft_iou"] == 1.0
    assert result["oracle_task_margin"] == pytest.approx(6.0)
    assert result["oracle_task_rank_one"] is True


def test_row_permutation_oracle_keeps_correct_identity_rows() -> None:
    sample = {
        "partition": "validation",
        "task_key": "find-a",
        "sample_key": "sample-b",
        "factual_target_identity_keys": ["a"],
        "factual_ownership_rows": [
            _row(0, "a", [1.0, 0.0], [1.0, 0.0]),
            _row(1, "b", [0.0, 1.0], [0.0, 1.0]),
        ],
        "factual_task_row_diagnostic": {
            "exact_task": True,
            "task_logits": [2.0, -1.0],
            "row_task_valid": [True, True],
            "target_vs_hardest_negative_logit_margin": 3.0,
            "all_targets_beat_known_negatives": True,
        },
    }

    result = audit_row_permutation_sample(sample)

    assert result is not None
    assert result["changed_identity_count"] == 0
    assert result["assigned_macro_soft_iou"] == 1.0
    assert result["oracle_macro_soft_iou"] == 1.0
    assert result["oracle_task_margin"] == 3.0
    assert result["oracle_task_rank_one"] is True
