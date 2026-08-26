from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tools.audit_lingbot_calvin_entity_grid_ceiling import (
    _SUPPORTED_VISUAL_LATTICES,
    GRID_CEILING_REPORT_SCHEMA,
    _dummy_relation,
    _evaluation_replay_seed,
    _summarize,
)


def test_grid_ceiling_tool_freezes_only_registered_native_lattices() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "tools/audit_lingbot_calvin_entity_grid_ceiling.py"
    ).read_text()

    assert GRID_CEILING_REPORT_SCHEMA.endswith(".v2")
    assert _SUPPORTED_VISUAL_LATTICES == (8, 12)
    assert "configure_native_processor_lattice(processor, args.visual_lattice)" in source
    assert "expected_visual_tokens = 2 * args.visual_lattice**2" in source


def test_grid_ceiling_dummy_relation_is_visual_only_and_task_free() -> None:
    relation = _dummy_relation(visual_tokens=128, capacity=16)

    assert relation.support_logits.shape == (1, 128, 16)
    assert relation.ownership.shape == (1, 128, 17)
    assert relation.existence_logits.shape == (1, 16)
    assert relation.structural_valid.shape == (1, 128)
    assert relation.structural_valid.all()
    torch.testing.assert_close(
        relation.ownership.sum(dim=-1),
        torch.ones(1, 128),
    )


def test_grid_ceiling_replay_seed_is_content_addressed() -> None:
    plan = "1" * 64
    first = _evaluation_replay_seed(plan, "sample-a")

    assert first == _evaluation_replay_seed(plan, "sample-a")
    assert first != _evaluation_replay_seed(plan, "sample-b")
    with pytest.raises(ValueError, match="SHA-256"):
        _evaluation_replay_seed("bad", "sample-a")


def test_grid_ceiling_summary_is_partition_and_stratum_exact() -> None:
    samples = [
        {
            "partition": "heldout",
            "task_key": "task-a",
            "rows": [
                {
                    "area_fraction": 0.01,
                    "area_stratum": "lt_2_percent",
                    "soft_iou_ceiling": 0.4,
                },
                {
                    "area_fraction": 0.06,
                    "area_stratum": "ge_5_percent",
                    "soft_iou_ceiling": 0.9,
                },
            ],
        },
        {
            "partition": "validation",
            "task_key": "task-a",
            "rows": [
                {
                    "area_fraction": 0.03,
                    "area_stratum": "2_to_5_percent",
                    "soft_iou_ceiling": 0.8,
                }
            ],
        },
    ]

    summary = _summarize(samples, partition="heldout")

    assert summary["sample_count"] == 1
    assert summary["task_count"] == 1
    assert summary["entity_count"] == 2
    assert math.isclose(summary["mean_soft_iou_ceiling"], 0.65)
    assert summary["area_strata"]["lt_2_percent"]["entity_count"] == 1
    assert summary["area_strata"]["2_to_5_percent"]["entity_count"] == 0
    assert summary["area_strata"]["ge_5_percent"]["minimum_soft_iou_ceiling"] == 0.9
