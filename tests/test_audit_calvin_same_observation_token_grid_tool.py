from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_SOURCE_COMMIT,
)
from picf_next.data.calvin_task_applicability import (
    CALVIN_OFFICIAL_ANNOTATIONS_SHA256,
    CALVIN_OFFICIAL_TASKS_SHA256,
)
from picf_next.lingbot_native.representation_split import (
    RepresentationEvaluationSegment,
    RepresentationTrialSplit,
)
from tools.audit_calvin_same_observation_token_grid import (
    _EXPECTED_APPLICABILITY_SCOPE,
    _EXPECTED_LEAKAGE_CONTRACT,
    APPLICABILITY_AUDIT_SCHEMA,
    _bound_json,
    _canonical_json_bytes,
    _parse_groups,
    _representation_partition_coordinates,
    _validate_report_identity,
)


def _variant(task: str, instruction: str, target: str) -> dict[str, object]:
    return {
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "proof": f"proof:{task}",
        "target_identity_key": target,
        "task_key": task,
    }


def _split() -> RepresentationTrialSplit:
    return RepresentationTrialSplit(
        dataset_id="calvin",
        dataset_revision="revision",
        dataset_manifest_sha256="8" * 64,
        comparison_id="comparison",
        stream_plan_sha256="9" * 64,
        partition_seed=17,
        training_steps=1,
        training_sample_count=1,
        training_sample_keys_sha256="a" * 64,
        training_source_global_indices_sha256="b" * 64,
        training_segment_indices=(3,),
        training_source_episode_indices=(2,),
        segments_per_task=1,
        validation_segments=(
            RepresentationEvaluationSegment(
                task_key="task",
                segment_index=4,
                source_episode_index=4,
                source_start=40,
                source_end=41,
            ),
        ),
        heldout_segments=(
            RepresentationEvaluationSegment(
                task_key="task",
                segment_index=5,
                source_episode_index=5,
                source_start=50,
                source_end=51,
            ),
        ),
    )


def _report(
    representation_split: RepresentationTrialSplit | None = None,
    *,
    partition: str = "training",
) -> dict[str, object]:
    representation_split = representation_split or _split()
    segment_index, source_episode_index, source_global_index = {
        "training": (3, 2, 17),
        "validation": (4, 4, 40),
        "heldout": (5, 5, 50),
    }[partition]
    content = {
        "acceptance_scope": _EXPECTED_APPLICABILITY_SCOPE,
        "accepted_group_count": 1,
        "accepted_groups": [
            {
                "applicable_tasks": [],
                "model_input_contains_simulator_state_or_identity": False,
                "raw_visible_supervised_support": [],
                "scene": "calvin_scene_A",
                "schema": "picf-next.calvin-task-applicability.v1",
                "source_global_index": source_global_index,
                "source_sensor_sha256": {
                    "depth_gripper": "1" * 64,
                    "depth_static": "2" * 64,
                    "rgb_gripper": "3" * 64,
                    "rgb_static": "4" * 64,
                },
                "source_state_sha256": "5" * 64,
                "stateful_reset_binding": {
                    "language_segment_index": segment_index,
                    "source_episode_index": source_episode_index,
                    "source_instruction_sha256": "c" * 64,
                    "source_task_key": "turn_on_led",
                    "stateful_episode_key": (f"calvin-language-segment-{segment_index:08d}"),
                    "stateful_sample_key": (
                        f"calvin-language-segment-{segment_index:08d}/"
                        f"transition-00000000-frame-{source_global_index:08d}"
                    ),
                    "transition_index": 0,
                },
                "token_grid_measurability": "pending-host-native-projection",
                "variants": [
                    _variant(
                        "turn_on_led",
                        "turn on the led",
                        "part/table/button_link",
                    ),
                    _variant(
                        "lift_blue_block_table",
                        "lift the blue block",
                        "movable/block_blue",
                    ),
                ],
            }
        ],
        "calvin_env_source_commit": CALVIN_ENV_SOURCE_COMMIT,
        "calvin_source_commit": CALVIN_SOURCE_COMMIT,
        "dataset": {
            "dataset_manifest_file_sha256": "6" * 64,
            "split_name": "training",
        },
        "leakage_contract": _EXPECTED_LEAKAGE_CONTRACT,
        "official_annotations_sha256": CALVIN_OFFICIAL_ANNOTATIONS_SHA256,
        "official_task_config_sha256": CALVIN_OFFICIAL_TASKS_SHA256,
        "physical_sidecar_manifest_sha256": "7" * 64,
        "representation_split": {
            "artifact_sha256": representation_split.artifact_sha256,
            "comparison_id": representation_split.comparison_id,
            "file_sha256": "d" * 64,
            "partition": partition,
            "partition_segment_count": 1,
            "partition_source_episode_count": 1,
            "schema": representation_split.schema,
            "stream_plan_sha256": representation_split.stream_plan_sha256,
        },
        "rejected_frame_count": 0,
        "rejected_frames": [],
        "schema": APPLICABILITY_AUDIT_SCHEMA,
        "selection": {},
        "summary": {},
        "visual_artifacts": [],
    }
    return {
        **content,
        "artifact_sha256": hashlib.sha256(_canonical_json_bytes(content)).hexdigest(),
    }


def _write(path: Path, report: dict[str, object]) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True).encode("ascii") + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_token_grid_audit_reopens_content_bound_applicability_groups(
    tmp_path: Path,
) -> None:
    path = tmp_path / "applicability.json"
    representation_split = _split()
    expected_sha256 = _write(path, _report(representation_split))

    loaded = _bound_json(path, expected_sha256=expected_sha256)
    groups = _validate_report_identity(
        loaded,
        dataset_manifest_sha256="6" * 64,
        sidecar_manifest_sha256="7" * 64,
        representation_split=representation_split,
        representation_split_file_sha256="d" * 64,
        expected_representation_partition="training",
    )

    assert len(groups) == 1
    assert groups[0].group.source_global_index == 17
    assert [value.task_key for value in groups[0].group.variants] == [
        "turn_on_led",
        "lift_blue_block_table",
    ]


@pytest.mark.parametrize(
    ("partition", "expected_segment", "expected_source", "expected_global"),
    (
        ("training", 3, 2, 17),
        ("validation", 4, 4, 40),
        ("heldout", 5, 5, 50),
    ),
)
def test_token_grid_audit_accepts_each_frozen_representation_partition(
    partition: str,
    expected_segment: int,
    expected_source: int,
    expected_global: int,
) -> None:
    representation_split = _split()
    segments, sources = _representation_partition_coordinates(
        representation_split,
        partition,
    )
    groups = _validate_report_identity(
        _report(representation_split, partition=partition),
        dataset_manifest_sha256="6" * 64,
        sidecar_manifest_sha256="7" * 64,
        representation_split=representation_split,
        representation_split_file_sha256="d" * 64,
        expected_representation_partition=partition,
    )

    assert segments == (expected_segment,)
    assert sources == (expected_source,)
    assert groups[0].group.source_global_index == expected_global


def test_token_grid_audit_rejects_internal_artifact_tampering(tmp_path: Path) -> None:
    path = tmp_path / "applicability.json"
    report = _report()
    report["accepted_group_count"] = 2
    expected_sha256 = _write(path, report)

    with pytest.raises(ContractError, match="artifact digest"):
        _bound_json(path, expected_sha256=expected_sha256)


def test_token_grid_audit_rejects_duplicate_fixed_observation(tmp_path: Path) -> None:
    report = _report()
    first = report["accepted_groups"][0]  # type: ignore[index]
    report["accepted_groups"] = [first, first]
    report["accepted_group_count"] = 2
    content = {key: value for key, value in report.items() if key != "artifact_sha256"}
    report["artifact_sha256"] = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    path = tmp_path / "applicability.json"
    expected_sha256 = _write(path, report)

    loaded = _bound_json(path, expected_sha256=expected_sha256)
    with pytest.raises(ContractError, match="repeat"):
        _parse_groups(loaded)


def test_token_grid_audit_rejects_nonreset_stateful_address(tmp_path: Path) -> None:
    report = _report()
    group = report["accepted_groups"][0]  # type: ignore[index]
    group["stateful_reset_binding"]["transition_index"] = 1
    content = {key: value for key, value in report.items() if key != "artifact_sha256"}
    report["artifact_sha256"] = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    path = tmp_path / "applicability.json"
    expected_sha256 = _write(path, report)

    loaded = _bound_json(path, expected_sha256=expected_sha256)
    with pytest.raises(ContractError, match="exact stateful reset"):
        _parse_groups(loaded)


def test_token_grid_audit_rejects_representation_split_mismatch(
    tmp_path: Path,
) -> None:
    representation_split = _split()
    path = tmp_path / "applicability.json"
    expected_sha256 = _write(path, _report(representation_split))
    loaded = _bound_json(path, expected_sha256=expected_sha256)

    with pytest.raises(ContractError, match="another representation split"):
        _validate_report_identity(
            loaded,
            dataset_manifest_sha256="6" * 64,
            sidecar_manifest_sha256="7" * 64,
            representation_split=representation_split,
            representation_split_file_sha256="e" * 64,
            expected_representation_partition="training",
        )
