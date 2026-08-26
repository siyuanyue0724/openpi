from __future__ import annotations

from copy import deepcopy

import pytest

from tools.audit_lingbot_native_vl_curriculum_materialization import (
    _materialization_summary,
)


def _record(index: int) -> dict[str, object]:
    return {
        "bbox_xyxy": [1, 2, 11, 22],
        "camera_name": "static",
        "global_index": index,
        "instruction_sha256": f"{index:064x}",
        "optimizer_step": index,
        "qwen_bbox_xyxy": [10, 20, 110, 220],
        "rank": index % 2,
        "source_rgb_sha256": f"{index + 1:064x}",
        "target_identity_key": f"target-{index}",
        "task_key": f"task-{index}",
    }


def test_materialization_summary_requires_exact_coverage_and_declared_duplicate() -> None:
    records = [_record(0), _record(1)]
    records.append(deepcopy(records[0]))
    summary = _materialization_summary(
        records,
        expected_record_count=3,
        expected_unique_variant_count=2,
    )
    assert summary == {
        "bbox_area_maximum": 200,
        "bbox_area_minimum": 200,
        "camera_histogram": {"static": 3},
        "duplicate_multiplicities": [2],
        "materialized_record_count": 3,
        "unique_variant_count": 2,
    }

    with pytest.raises(ValueError, match="record count"):
        _materialization_summary(
            records,
            expected_record_count=4,
            expected_unique_variant_count=2,
        )
    records[-1]["bbox_xyxy"] = [1, 2, 1, 22]
    with pytest.raises(ValueError, match="non-positive"):
        _materialization_summary(
            records,
            expected_record_count=3,
            expected_unique_variant_count=2,
        )
