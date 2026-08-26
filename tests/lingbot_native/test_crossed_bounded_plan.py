from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from picf_next.lingbot_native.crossed_bounded_plan import (
    CrossedBoundedPlan,
    CrossedBoundedRecord,
    CrossedBoundedStep,
    _Edge,
    _solve_selection,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


TASKS = tuple(f"task-{index:02d}" for index in range(17))
TARGETS = tuple(f"target-{index:02d}" for index in range(7))


def _record(
    *,
    group_index: int,
    variant_index: int,
    camera_name: str,
    task_index: int,
    instruction: str,
    bbox: tuple[int, int, int, int],
) -> CrossedBoundedRecord:
    return CrossedBoundedRecord(
        group_index=group_index,
        variant_index=variant_index,
        global_index=10_000 + group_index,
        source_episode_index=100 + group_index,
        source_state_sha256=_sha(f"state-{group_index}"),
        camera_name=camera_name,
        source_rgb_sha256=_sha(f"rgb-{camera_name}-{group_index}"),
        task_key=TASKS[task_index],
        instruction_sha256=_sha(instruction),
        target_identity_key=TARGETS[task_index % len(TARGETS)],
        bbox_qwen_xyxy=bbox,
    )


def _valid_plan() -> CrossedBoundedPlan:
    prompt_pairs = []
    for pair_index in range(32):
        camera = "static" if pair_index < 16 else "gripper"
        first_task = (2 * pair_index) % len(TASKS)
        second_task = (2 * pair_index + 1) % len(TASKS)
        prompt_pairs.append(
            (
                _record(
                    group_index=pair_index,
                    variant_index=0,
                    camera_name=camera,
                    task_index=first_task,
                    instruction=f"prompt-{pair_index}-0",
                    bbox=(10, 10, 110, 110),
                ),
                _record(
                    group_index=pair_index,
                    variant_index=1,
                    camera_name=camera,
                    task_index=second_task,
                    instruction=f"prompt-{pair_index}-1",
                    bbox=(800, 800, 900, 900),
                ),
            )
        )

    x_specs = [
        *((task_index, "static") for task_index in range(16)),
        *((task_index, "gripper") for task_index in range(4, 17)),
        *((task_index, "gripper") for task_index in range(4, 7)),
    ]
    assert len(x_specs) == 32
    steps = []
    for pair_index, (task_index, camera) in enumerate(x_specs):
        prompt = prompt_pairs[pair_index]
        steps.append(
            CrossedBoundedStep(
                optimizer_step=len(steps),
                cell="P",
                candidate_records=prompt,
                control_records=prompt,
            )
        )
        first_group = 32 + 2 * pair_index
        second_group = first_group + 1
        candidate = (
            _record(
                group_index=first_group,
                variant_index=0,
                camera_name=camera,
                task_index=task_index,
                instruction=f"candidate-{pair_index}",
                bbox=(10, 10, 110, 110),
            ),
            _record(
                group_index=second_group,
                variant_index=0,
                camera_name=camera,
                task_index=task_index,
                instruction=f"candidate-{pair_index}",
                bbox=(800, 800, 900, 900),
            ),
        )
        control = (
            _record(
                group_index=first_group,
                variant_index=1,
                camera_name=camera,
                task_index=task_index,
                instruction=f"control-{pair_index}-0",
                bbox=(10, 10, 110, 110),
            ),
            _record(
                group_index=second_group,
                variant_index=1,
                camera_name=camera,
                task_index=task_index,
                instruction=f"control-{pair_index}-1",
                bbox=(800, 800, 900, 900),
            ),
        )
        steps.append(
            CrossedBoundedStep(
                optimizer_step=len(steps),
                cell="X",
                candidate_records=candidate,
                control_records=control,
            )
        )
    return CrossedBoundedPlan(
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
        expected_task_keys=TASKS,
        expected_target_identity_keys=TARGETS,
        expected_x_task_camera_strata=tuple(
            sorted({(TASKS[task_index], camera) for task_index, camera in x_specs})
        ),
        heldout_source_episode_indices=(999,),
        maximum_control_x_source_group_overlap_count=64,
        steps=tuple(steps),
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


def test_crossed_bounded_plan_round_trip_and_matched_measure(tmp_path: Path) -> None:
    plan = _valid_plan()
    output = tmp_path / "plan.json"
    plan.write(output)
    assert CrossedBoundedPlan.load(output) == plan
    assert plan.summary["arm_calvin_record_count"] == 128
    assert plan.summary["arm_unique_source_group_count"] == 96
    assert plan.summary["maximum_control_x_source_group_overlap_count"] == 64
    assert plan.summary["candidate_exact_x_instruction_count"] == 32
    assert plan.summary["control_unique_x_instruction_count"] == 64


def test_crossed_bounded_plan_rejects_semantic_and_provenance_tampering() -> None:
    value = _valid_plan().as_dict()
    repeated = deepcopy(value)
    repeated["steps"][1]["control_records"][1]["instruction_sha256"] = repeated["steps"][1][
        "control_records"
    ][0]["instruction_sha256"]
    _resign(repeated)
    with pytest.raises(ValueError, match="semantically matched"):
        CrossedBoundedPlan.from_dict(repeated)

    wrong_overlap = deepcopy(value)
    wrong_overlap["maximum_control_x_source_group_overlap_count"] = 63
    wrong_overlap["summary"]["maximum_control_x_source_group_overlap_count"] = 63
    _resign(wrong_overlap)
    with pytest.raises(ValueError, match="maximum source overlap"):
        CrossedBoundedPlan.from_dict(wrong_overlap)

    leaked = deepcopy(value)
    leaked["heldout_source_episode_indices"] = [132]
    _resign(leaked)
    with pytest.raises(ValueError, match="heldout"):
        CrossedBoundedPlan.from_dict(leaked)


def test_binary_selection_treats_full_index_vector_as_indices() -> None:
    record = _record(
        group_index=0,
        variant_index=0,
        camera_name="static",
        task_index=0,
        instruction="solver-record",
        bbox=(10, 10, 110, 110),
    )
    edges = tuple(_Edge(records=(record, record)) for _index in range(3))
    selected = _solve_selection(
        edges,
        [(tuple(range(3)), 2, 2)],
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        name="unit",
    )
    assert len(selected) == 2
