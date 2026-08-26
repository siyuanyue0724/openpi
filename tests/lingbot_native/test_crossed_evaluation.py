from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from picf_next.lingbot_native.crossed_bounded_plan import CrossedBoundedRecord
from picf_next.lingbot_native.crossed_evaluation import (
    CrossedEvaluationPair,
    CrossedEvaluationPlan,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _record(
    *,
    group_index: int,
    task_key: str,
    target_identity_key: str,
    camera_name: str,
    instruction: str,
    bbox: tuple[int, int, int, int],
) -> CrossedBoundedRecord:
    return CrossedBoundedRecord(
        group_index=group_index,
        variant_index=0,
        global_index=10_000 + group_index,
        source_episode_index=100 + group_index,
        source_state_sha256=_sha(f"state-{group_index}"),
        camera_name=camera_name,
        source_rgb_sha256=_sha(f"rgb-{camera_name}-{group_index}"),
        task_key=task_key,
        instruction_sha256=_sha(instruction),
        target_identity_key=target_identity_key,
        bbox_qwen_xyxy=bbox,
    )


def _pair(
    *,
    first_group: int,
    task_key: str,
    target_identity_key: str,
    camera_name: str,
    instruction: str,
) -> CrossedEvaluationPair:
    return CrossedEvaluationPair(
        first=_record(
            group_index=first_group,
            task_key=task_key,
            target_identity_key=target_identity_key,
            camera_name=camera_name,
            instruction=instruction,
            bbox=(10, 10, 110, 110),
        ),
        second=_record(
            group_index=first_group + 1,
            task_key=task_key,
            target_identity_key=target_identity_key,
            camera_name=camera_name,
            instruction=instruction,
            bbox=(800, 800, 900, 900),
        ),
    )


def _ordered_pair(
    first: CrossedBoundedRecord,
    second: CrossedBoundedRecord,
) -> CrossedEvaluationPair:
    records = sorted(
        (first, second),
        key=lambda record: (
            record.task_key,
            record.target_identity_key,
            record.camera_name,
            record.instruction_sha256,
            record.group_index,
            record.variant_index,
        ),
    )
    return CrossedEvaluationPair(first=records[0], second=records[1])


def _plan() -> CrossedEvaluationPlan:
    pairs = tuple(
        sorted(
            (
                _pair(
                    first_group=0,
                    task_key="task-a",
                    target_identity_key="object/a",
                    camera_name="static",
                    instruction="move object a",
                ),
                _pair(
                    first_group=2,
                    task_key="task-b",
                    target_identity_key="object/b",
                    camera_name="gripper",
                    instruction="move object b",
                ),
            ),
            key=lambda pair: pair.key,
        )
    )
    return CrossedEvaluationPlan(
        dataset_id="calvin",
        dataset_revision="synthetic",
        dataset_manifest_sha256=_sha("manifest"),
        curriculum_file_sha256=_sha("curriculum-file"),
        curriculum_artifact_sha256=_sha("curriculum-artifact"),
        scene_audit_file_sha256=_sha("scene-file"),
        scene_audit_artifact_sha256=_sha("scene-artifact"),
        episode_split_file_sha256=_sha("split-file"),
        episode_split_artifact_sha256=_sha("split-artifact"),
        episode_split_picf_code_revision="1" * 40,
        picf_code_revision="2" * 40,
        expected_task_keys=("task-a", "task-b"),
        expected_target_identity_keys=("object/a", "object/b"),
        heldout_source_episode_indices=(100, 101, 102, 103),
        expected_pair_count=2,
        pairs=pairs,
    )


def _resign(value: dict[str, object]) -> None:
    content = {key: child for key, child in value.items() if key != "artifact_sha256"}
    payload = json.dumps(
        content,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    value["artifact_sha256"] = hashlib.sha256(payload).hexdigest()


def test_crossed_evaluation_plan_round_trip_is_complete_and_nonselective(
    tmp_path: Path,
) -> None:
    plan = _plan()
    output = tmp_path / "evaluation.json"

    plan.write(output)

    assert CrossedEvaluationPlan.load(output) == plan
    assert plan.summary == {
        "camera_record_histogram": {"gripper": 2, "static": 2},
        "covered_instruction_count": 2,
        "covered_target_cameras": [
            {"camera_name": "static", "target_identity_key": "object/a"},
            {"camera_name": "gripper", "target_identity_key": "object/b"},
        ],
        "covered_target_identity_keys": ["object/a", "object/b"],
        "covered_task_keys": ["task-a", "task-b"],
        "pair_count": 2,
        "unique_record_count": 4,
    }
    assert plan.as_dict()["checkpoint_selection_authorized"] is False
    assert plan.as_dict()["training_authorized"] is False


def test_crossed_evaluation_rejects_content_and_semantic_tampering() -> None:
    unsigned = _plan().as_dict()
    unsigned["pairs"][0]["second"]["source_rgb_sha256"] = unsigned["pairs"][0]["first"][
        "source_rgb_sha256"
    ]
    with pytest.raises(ValueError, match="artifact SHA-256 changed"):
        CrossedEvaluationPlan.from_dict(unsigned)

    resigned = deepcopy(_plan().as_dict())
    resigned["pairs"][0]["second"]["source_rgb_sha256"] = resigned["pairs"][0]["first"][
        "source_rgb_sha256"
    ]
    _resign(resigned)
    with pytest.raises(ValueError, match="strict exact-X"):
        CrossedEvaluationPlan.from_dict(resigned)


def test_crossed_evaluation_pair_rejects_prompt_drift_and_spatial_overlap() -> None:
    first = _record(
        group_index=0,
        task_key="task-a",
        target_identity_key="object/a",
        camera_name="static",
        instruction="move object a",
        bbox=(10, 10, 110, 110),
    )
    with pytest.raises(ValueError, match="strict exact-X"):
        _ordered_pair(
            first,
            _record(
                group_index=1,
                task_key="task-a",
                target_identity_key="object/a",
                camera_name="static",
                instruction="different wording",
                bbox=(800, 800, 900, 900),
            ),
        )
    with pytest.raises(ValueError, match="strict exact-X"):
        _ordered_pair(
            first,
            _record(
                group_index=1,
                task_key="task-a",
                target_identity_key="object/a",
                camera_name="static",
                instruction="move object a",
                bbox=(50, 50, 120, 120),
            ),
        )
