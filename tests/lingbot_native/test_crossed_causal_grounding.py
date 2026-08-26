from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tools.audit_lingbot_native_crossed_episode_split as split_audit_tool
import tools.audit_lingbot_native_crossed_grounding_support as audit_tool
import tools.audit_lingbot_native_crossed_partition_support as partition_audit_tool
import tools.render_lingbot_native_crossed_partition_review as review_tool
from picf_next.lingbot_native.crossed_causal_grounding import (
    CROSSED_EPISODE_SPLIT_SCHEMA,
    CROSSED_PARTITION_SUPPORT_SCHEMA,
    CROSSED_PHYSICAL_SUPPORT_SCHEMA,
    build_crossed_episode_split_report,
    build_crossed_partition_support_report,
    build_crossed_physical_support_report,
    crossed_support_report_bytes,
    materialize_crossed_variant_views,
    parse_scene_target_evidence,
)
from picf_next.lingbot_native.fixed_observation import (
    FixedObservationGroup,
    FixedObservationVariant,
)

_TARGET = "movable/block_blue"
_CURRICULUM_SHA256 = "c" * 64


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    content = dict(value)
    content.pop("artifact_sha256", None)
    return {
        **content,
        "artifact_sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest(),
    }


def _scene_audit(
    *,
    second_state: str = "supervised",
    fixed_boxes: bool = False,
    include_second_gripper: bool = True,
) -> dict[str, Any]:
    views: list[dict[str, Any]] = []
    for group_index, global_index in enumerate((101, 202)):
        for camera_name in ("static", "gripper"):
            if group_index == 1 and camera_name == "gripper" and not include_second_gripper:
                continue
            if camera_name == "static":
                first_box = [10, 10, 40, 40]
                second_box = first_box if fixed_boxes else [130, 130, 170, 170]
            else:
                first_box = [4, 4, 20, 20]
                second_box = first_box if fixed_boxes else [55, 55, 76, 76]
            state = "supervised" if group_index == 0 else second_state
            objects = (
                [
                    {
                        "bbox_xyxy": first_box if group_index == 0 else second_box,
                        "identity_key": _TARGET,
                    }
                ]
                if state == "supervised"
                else []
            )
            subpatch = (
                [{"bbox_xyxy": second_box, "identity_key": _TARGET}] if state == "subpatch" else []
            )
            absent = [_TARGET] if state == "absent" else []
            views.append(
                {
                    "absent_identity_keys": absent,
                    "camera_name": camera_name,
                    "global_index": global_index,
                    "group_index": group_index,
                    "objects": objects,
                    "source_rgb_sha256": (
                        f"{group_index + 1:x}" * 64
                        if camera_name == "static"
                        else f"{group_index + 5:x}" * 64
                    ),
                    "subpatch_objects": subpatch,
                }
            )
    return _signed(
        {
            "arm_steps": [
                {"camera_name": "static"},
                {"camera_name": "gripper"},
            ],
            "curriculum_artifact_sha256": _CURRICULUM_SHA256,
            "scene_views": views,
            "schema": "picf-next.native-vl-scene-curriculum-audit.v2",
            "status": "PASS",
        }
    )


def _report(scene: Mapping[str, Any]) -> dict[str, object]:
    return build_crossed_physical_support_report(
        scene,
        scene_audit_file_sha256="f" * 64,
        target_identity_keys=[_TARGET],
        expected_curriculum_artifact_sha256=_CURRICULUM_SHA256,
    )


def _crossed_partition(
    *,
    camera_transfer_gap: bool = False,
    group_count: int = 3,
    prompt_overlap: bool = False,
    prompt_drift: bool = False,
    shared_episode: bool = False,
) -> tuple[tuple[FixedObservationGroup, ...], dict[str, Any]]:
    targets = ("object/a", "object/b")
    groups = []
    views = []
    for group_index in range(group_count):
        global_index = (group_index + 1) * 100
        variants = []
        for variant_index, target in enumerate(targets):
            instruction = (
                f"canonical task {variant_index}"
                if not prompt_drift
                else f"task {variant_index} source {group_index}"
            )
            variants.append(
                FixedObservationVariant(
                    task_key=f"task_{variant_index}",
                    instruction=instruction,
                    instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                    target_identity_key=target,
                    target_mass=1.0,
                )
            )
        sensor_hashes = tuple(
            (name, _sha(f"{name}-{group_index}"))
            for name in ("depth_gripper", "depth_static", "rgb_gripper", "rgb_static")
        )
        groups.append(
            FixedObservationGroup(
                scene=f"scene-{group_index}",
                source_global_index=global_index,
                source_state_sha256=_sha(f"state-{group_index}"),
                source_sensor_sha256=sensor_hashes,
                source_episode_index=0 if shared_episode else group_index,
                source_task_key=f"source-task-{group_index}",
                source_instruction_sha256=_sha(f"source-prompt-{group_index}"),
                stateful_episode_key=f"episode-{group_index}",
                stateful_sample_key=f"sample-{group_index}",
                variants=tuple(variants),
            )
        )
        sensor_by_name = dict(sensor_hashes)
        for camera_name in ("static", "gripper"):
            extent = 200 if camera_name == "static" else 84
            position = group_index % 3
            low = 10 if position == 0 else 120 if position == 1 else 60
            if camera_name == "gripper":
                low = 4 if position == 0 else 54 if position == 1 else 30
            width = 25 if camera_name == "static" else 14
            object_rows = []
            absent = []
            for target_index, target in enumerate(targets):
                if (position == 2 and target_index == 0) or (
                    camera_transfer_gap and camera_name == "gripper" and target_index == 1
                ):
                    absent.append(target)
                    continue
                offset = 0 if target_index == 0 or prompt_overlap else extent // 4
                x1 = min(low + offset, extent - width - 1)
                object_rows.append(
                    {
                        "bbox_xyxy": [x1, x1, x1 + width, x1 + width],
                        "identity_key": target,
                    }
                )
            views.append(
                {
                    "absent_identity_keys": absent,
                    "camera_name": camera_name,
                    "global_index": global_index,
                    "group_index": group_index,
                    "objects": object_rows,
                    "source_rgb_sha256": sensor_by_name[f"rgb_{camera_name}"],
                    "subpatch_objects": [],
                }
            )
    scene = _signed(
        {
            "arm_steps": [
                {"camera_name": "static"},
                {"camera_name": "gripper"},
            ],
            "curriculum_artifact_sha256": "d" * 64,
            "scene_views": views,
            "schema": "picf-next.native-vl-scene-curriculum-audit.v2",
            "status": "PASS",
        }
    )
    return tuple(groups), scene


def _partition_report(
    groups: tuple[FixedObservationGroup, ...],
    scene: Mapping[str, Any],
    *,
    expected_task_keys: tuple[str, ...] = ("task_0", "task_1"),
) -> dict[str, object]:
    return build_crossed_partition_support_report(
        groups,
        scene,
        curriculum_artifact_sha256="d" * 64,
        curriculum_file_sha256="e" * 64,
        scene_audit_file_sha256="f" * 64,
        expected_task_keys=expected_task_keys,
        expected_target_identity_keys=("object/a", "object/b"),
    )


def _episode_split_report(
    groups: tuple[FixedObservationGroup, ...],
    scene: Mapping[str, Any],
    support: Mapping[str, Any],
    *,
    heldout_source_episode_indices: tuple[int, ...],
) -> dict[str, object]:
    return build_crossed_episode_split_report(
        groups,
        scene,
        support,
        curriculum_artifact_sha256="d" * 64,
        curriculum_file_sha256="e" * 64,
        scene_audit_file_sha256="f" * 64,
        crossed_support_report_file_sha256="a" * 64,
        picf_code_revision="b" * 40,
        expected_task_keys=("task_0", "task_1"),
        expected_target_identity_keys=("object/a", "object/b"),
        heldout_source_episode_indices=heldout_source_episode_indices,
    )


def test_physical_crossed_support_passes_only_as_a_nontraining_gate() -> None:
    report = _report(_scene_audit())

    assert report["schema"] == CROSSED_PHYSICAL_SUPPORT_SCHEMA
    assert report["status"] == "PASS"
    assert report["training_authorized"] is False
    support = report["identity_camera_support"]
    assert isinstance(support, Mapping)
    target = support[_TARGET]
    assert isinstance(target, Mapping)
    assert target["static"]["mutually_center_exclusive_source_pair_count"] == 1
    assert target["gripper"]["mutually_center_exclusive_source_pair_count"] == 1

    payload = crossed_support_report_bytes(report)
    assert json.loads(payload)["artifact_sha256"] == report["artifact_sha256"]


def test_identical_boxes_do_not_create_false_pixel_causal_support() -> None:
    report = _report(_scene_audit(fixed_boxes=True))

    assert report["status"] == "FAIL"
    assert report["failures"] == [f"target {_TARGET} has no physical pixel-causal pair"]


def test_subpatch_is_neither_training_box_nor_absence() -> None:
    report = _report(_scene_audit(second_state="subpatch"))
    support = report["identity_camera_support"]
    assert isinstance(support, Mapping)
    target = support[_TARGET]
    assert target["static"]["supervised_count"] == 1
    assert target["static"]["subpatch_count"] == 1
    assert target["static"]["absent_count"] == 0
    assert report["status"] == "FAIL"


def test_scene_audit_content_signature_is_recomputed() -> None:
    scene = _scene_audit()
    scene["status"] = "FAIL"

    with pytest.raises(ValueError, match="artifact SHA-256 changed"):
        parse_scene_target_evidence(scene, target_identity_keys=[_TARGET])


def test_scene_target_partition_must_be_unique() -> None:
    scene = _scene_audit()
    scene["scene_views"][0]["absent_identity_keys"] = [_TARGET]
    scene = _signed(scene)

    with pytest.raises(ValueError, match="visibility partitions overlap"):
        parse_scene_target_evidence(scene, target_identity_keys=[_TARGET])


def test_every_group_must_have_both_cameras() -> None:
    scene = _scene_audit(include_second_gripper=False)

    with pytest.raises(ValueError, match="both cameras for every group"):
        parse_scene_target_evidence(scene, target_identity_keys=[_TARGET])


def test_tool_binds_input_file_and_publishes_immutable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene_path = tmp_path / "scene.json"
    scene_payload = _canonical_bytes(_scene_audit()) + b"\n"
    scene_path.write_bytes(scene_payload)
    output = tmp_path / "report.json"
    args = argparse.Namespace(
        scene_audit=scene_path,
        scene_audit_sha256=hashlib.sha256(scene_payload).hexdigest(),
        expected_curriculum_artifact_sha256=_CURRICULUM_SHA256,
        target_identities=[_TARGET],
        output=output,
    )
    monkeypatch.setattr(audit_tool, "_parse_args", lambda: args)

    audit_tool.main()

    written = json.loads(output.read_text())
    assert written["status"] == "PASS"
    assert written["training_authorized"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    with pytest.raises(FileExistsError):
        audit_tool.main()


def test_tool_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "scene.json"
    path.write_text("{}")

    with pytest.raises(ValueError, match="file SHA-256 changed"):
        audit_tool._load_verified_json(path, expected_sha256="0" * 64)


def test_partition_gate_accepts_fully_crossed_exact_prompts() -> None:
    groups, scene = _crossed_partition()

    report = _partition_report(groups, scene)

    assert report["schema"] == CROSSED_PARTITION_SUPPORT_SCHEMA
    assert report["status"] == "PASS"
    assert report["training_authorized"] is False
    exact = report["pixel_causal_cells_exact_instruction"]
    assert exact["covered_task_keys"] == ["task_0", "task_1"]
    assert exact["pair_count"] > 0
    assert report["null_cells_exact_instruction"]["pair_count"] > 0
    assert report["camera_transfer_cells"]["covered_target_identity_keys"] == [
        "object/a",
        "object/b",
    ]
    crossed_support_report_bytes(report)


def test_partition_gate_rejects_same_size_wrong_task_inventory() -> None:
    groups, scene = _crossed_partition()

    report = _partition_report(
        groups,
        scene,
        expected_task_keys=("task_0", "task_2"),
    )

    assert report["status"] == "FAIL"
    assert report["expected_task_keys"] == ["task_0", "task_2"]
    assert report["task_keys"] == ["task_0", "task_1"]
    assert any("task inventory differs" in value for value in report["failures"])


def test_partition_gate_rejects_camera_transfer_gap() -> None:
    groups, scene = _crossed_partition(camera_transfer_gap=True)

    report = _partition_report(groups, scene)

    assert report["status"] == "FAIL"
    assert any(
        value == "camera-transfer C cells miss tasks: task_1" for value in report["failures"]
    )
    assert any(
        value == "camera-transfer C cells miss targets: object/b" for value in report["failures"]
    )


def test_partition_gate_rejects_spatially_ambiguous_prompt_pairs() -> None:
    groups, scene = _crossed_partition(prompt_overlap=True)

    report = _partition_report(groups, scene)

    assert report["status"] == "FAIL"
    assert report["prompt_causal_cells"]["pair_count"] == 0
    assert "prompt-causal P cells do not cover every task" in report["failures"]
    assert "prompt-causal P cells do not cover every target" in report["failures"]


def test_task_key_crossing_cannot_hide_exact_prompt_drift() -> None:
    groups, scene = _crossed_partition(prompt_drift=True)

    report = _partition_report(groups, scene)

    assert report["status"] == "FAIL"
    assert report["pixel_causal_cells_exact_instruction"]["pair_count"] == 0
    assert report["pixel_causal_cells_task_semantic_only"]["pair_count"] > 0
    assert any("exact-instruction X cells miss tasks" in value for value in report["failures"])


def test_same_episode_pairs_are_not_source_disjoint() -> None:
    groups, scene = _crossed_partition(shared_episode=True)

    report = _partition_report(groups, scene)

    assert report["status"] == "FAIL"
    assert report["pixel_causal_cells_exact_instruction"]["pair_count"] == 0


def test_episode_split_gate_preserves_training_support_and_heldout_tasks() -> None:
    groups, scene = _crossed_partition(group_count=6)
    support = _partition_report(groups, scene)

    report = _episode_split_report(
        groups,
        scene,
        support,
        heldout_source_episode_indices=(3, 4, 5),
    )

    assert report["schema"] == CROSSED_EPISODE_SPLIT_SCHEMA
    assert report["status"] == "PASS"
    assert report["training_authorized"] is False
    assert report["validation_claim"] == "no-complete-third-calvin-exact-x-partition"
    assert report["heldout_camera_transfer_missing_task_keys"] == []
    assert all(row["intersection_count"] == 0 for row in report["disjointness"].values())
    graph = report["exact_x_episode_graph"]
    assert min(row["maximum_source_disjoint_partition_count"] for row in graph["by_task"]) == 2
    crossed_support_report_bytes(report)


def test_episode_split_gate_rejects_heldout_without_exact_x_support() -> None:
    groups, scene = _crossed_partition(group_count=6)
    support = _partition_report(groups, scene)

    report = _episode_split_report(
        groups,
        scene,
        support,
        heldout_source_episode_indices=(5,),
    )

    assert report["status"] == "FAIL"
    assert any(
        value.startswith("heldout pixel_causal_cells_exact_instruction misses tasks")
        for value in report["failures"]
    )


def test_episode_split_gate_rejects_source_state_leakage() -> None:
    groups, scene = _crossed_partition(group_count=6)
    support = _partition_report(groups, scene)
    changed = (
        *groups[:3],
        replace(groups[3], source_state_sha256=groups[0].source_state_sha256),
        *groups[4:],
    )

    with pytest.raises(ValueError, match="leaks source_state_sha256s"):
        _episode_split_report(
            changed,
            scene,
            support,
            heldout_source_episode_indices=(3, 4, 5),
        )


def test_episode_split_tool_binds_inputs_and_publishes_immutable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    groups, scene = _crossed_partition(group_count=6)
    support = _partition_report(groups, scene)
    curriculum_path = tmp_path / "curriculum.json"
    curriculum_payload = b"{}\n"
    curriculum_path.write_bytes(curriculum_payload)
    scene_path = tmp_path / "scene.json"
    scene_payload = _canonical_bytes(scene) + b"\n"
    scene_path.write_bytes(scene_payload)
    support_path = tmp_path / "support.json"
    support_payload = crossed_support_report_bytes(support)
    support_path.write_bytes(support_payload)
    output = tmp_path / "split.json"
    args = argparse.Namespace(
        curriculum=curriculum_path,
        curriculum_sha256=hashlib.sha256(curriculum_payload).hexdigest(),
        expected_curriculum_artifact_sha256="d" * 64,
        scene_audit=scene_path,
        scene_audit_sha256=hashlib.sha256(scene_payload).hexdigest(),
        crossed_support_report=support_path,
        crossed_support_report_sha256=hashlib.sha256(support_payload).hexdigest(),
        picf_code_revision="b" * 40,
        expected_task_keys=["task_0", "task_1"],
        target_identities=["object/a", "object/b"],
        heldout_source_episode_indices=[3, 4, 5],
        output=output,
    )
    parsed_curriculum = SimpleNamespace(artifact_sha256="d" * 64, groups=groups)
    monkeypatch.setattr(split_audit_tool, "_parse_args", lambda: args)
    monkeypatch.setattr(
        split_audit_tool,
        "_validated_checkout_revision",
        lambda _repository: "b" * 40,
    )
    monkeypatch.setattr(
        split_audit_tool.NativeVLGroundingCurriculumPlan,
        "from_dict",
        lambda _value: parsed_curriculum,
    )

    split_audit_tool.main()

    written = json.loads(output.read_text())
    assert written["status"] == "PASS"
    assert written["heldout_source_episode_indices"] == [3, 4, 5]
    assert written["training_authorized"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    with pytest.raises(FileExistsError):
        split_audit_tool.main()


def test_visual_review_selection_covers_every_registered_causal_axis() -> None:
    groups, scene = _crossed_partition()
    for view in scene["scene_views"]:
        if view["group_index"] != 2:
            continue
        view["objects"] = [row for row in view["objects"] if row["identity_key"] != "object/b"]
        view["absent_identity_keys"].append("object/b")
    scene = _signed(scene)
    rows = materialize_crossed_variant_views(
        groups,
        scene,
        expected_curriculum_artifact_sha256="d" * 64,
    )

    cells = review_tool.select_crossed_review_cells(
        rows,
        expected_task_keys=("task_0", "task_1"),
        expected_target_identity_keys=("object/a", "object/b"),
    )

    assert {cell.kind for cell in cells} == {"P", "X", "N", "C"}
    for kind in ("P", "X", "N", "C"):
        kind_cells = tuple(cell for cell in cells if cell.kind == kind)
        assert {task for cell in kind_cells for task in cell.task_keys} == {
            "task_0",
            "task_1",
        }
        assert {target for cell in kind_cells for target in cell.target_identity_keys} == {
            "object/a",
            "object/b",
        }
    for kind in ("X", "N"):
        assert {
            (cell.first.target_identity_key, cell.first.camera_name)
            for cell in cells
            if cell.kind == kind
        } == {
            ("object/a", "static"),
            ("object/a", "gripper"),
            ("object/b", "static"),
            ("object/b", "gripper"),
        }


def test_visual_review_selection_rejects_prompt_drift() -> None:
    groups, scene = _crossed_partition(prompt_drift=True)
    rows = materialize_crossed_variant_views(
        groups,
        scene,
        expected_curriculum_artifact_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="X review cells do not cover task"):
        review_tool.select_crossed_review_cells(
            rows,
            expected_task_keys=("task_0", "task_1"),
            expected_target_identity_keys=("object/a", "object/b"),
        )


def test_visual_review_contact_sheets_are_paired_and_content_addressed(
    tmp_path: Path,
) -> None:
    (tmp_path / "contact_sheets").mkdir()
    panels = [
        review_tool.Image.new("RGB", review_tool._PANEL_SIZE, color) for color in ("red", "blue")
    ]

    rows = review_tool._write_contact_sheet_pages(
        tmp_path,
        kind="P",
        panels=panels,
        cell_ids=["a" * 64],
    )

    assert rows[0]["cell_ids"] == ["a" * 64]
    relative = rows[0]["path"]
    assert isinstance(relative, str)
    path = tmp_path / relative
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rows[0]["sha256"]
    with pytest.raises(FileExistsError):
        review_tool._write_contact_sheet_pages(
            tmp_path,
            kind="P",
            panels=panels,
            cell_ids=["a" * 64],
        )


def test_partition_join_rejects_rgb_drift() -> None:
    groups, scene = _crossed_partition()
    scene["scene_views"][0]["source_rgb_sha256"] = "a" * 64
    scene = _signed(scene)

    with pytest.raises(ValueError, match="RGB digest differs"):
        _partition_report(groups, scene)


def test_partition_tool_binds_inputs_and_publishes_immutable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    groups, scene = _crossed_partition()
    curriculum_path = tmp_path / "curriculum.json"
    curriculum_payload = b"{}\n"
    curriculum_path.write_bytes(curriculum_payload)
    scene_path = tmp_path / "scene.json"
    scene_payload = _canonical_bytes(scene) + b"\n"
    scene_path.write_bytes(scene_payload)
    output = tmp_path / "report.json"
    args = argparse.Namespace(
        curriculum=curriculum_path,
        curriculum_sha256=hashlib.sha256(curriculum_payload).hexdigest(),
        expected_curriculum_artifact_sha256="d" * 64,
        scene_audit=scene_path,
        scene_audit_sha256=hashlib.sha256(scene_payload).hexdigest(),
        expected_task_keys=["task_0", "task_1"],
        target_identities=["object/a", "object/b"],
        output=output,
    )

    parsed_curriculum = SimpleNamespace(artifact_sha256="d" * 64, groups=groups)

    monkeypatch.setattr(partition_audit_tool, "_parse_args", lambda: args)
    monkeypatch.setattr(
        partition_audit_tool.NativeVLGroundingCurriculumPlan,
        "from_dict",
        lambda _value: parsed_curriculum,
    )

    partition_audit_tool.main()

    written = json.loads(output.read_text())
    assert written["status"] == "PASS"
    assert written["partition"] == "training"
    assert written["training_authorized"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    with pytest.raises(FileExistsError):
        partition_audit_tool.main()
