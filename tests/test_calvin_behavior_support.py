from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinLanguageSegment
from picf_next.data.calvin_behavior_support import (
    CALVIN_ROBOT_BASE_POSITION_M,
    CALVIN_SCENE_CONFIG_SHA256,
    CALVIN_SCENE_D_ROBOT_BASE_POSITION_M,
    calvin_behavior_review_keyframes,
    calvin_scene_d_tcp_robot_base_position,
    calvin_tcp_robot_base_position,
    select_calvin_behavior_segments,
    summarize_calvin_behavior_support,
)


def _valid_inputs() -> dict[str, object]:
    geometry = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]],
            [[0.2, 0.0, 0.0], [0.0, 0.01, 0.1]],
            [[0.4, 0.0, 0.0], [0.0, 0.02, 0.1]],
        ],
        dtype=np.float64,
    )
    tcp_robot_base = np.asarray(
        [[0.5, 0.0, 0.0], [0.21, 0.0, 0.0], [0.45, 0.0, 0.0]],
        dtype=np.float64,
    )
    return {
        "task_key": "push_blue_block_right",
        "target_identity_key": "movable/block_blue",
        "global_indices": (10, 11, 12),
        "identity_keys": ("movable/block_blue", "movable/block_red"),
        "geometry_robot_base_m": geometry,
        "tcp_position_world_m": tcp_robot_base + np.asarray(CALVIN_SCENE_D_ROBOT_BASE_POSITION_M),
        "actions": np.ones((2, 7), dtype=np.float64),
        "visible_target_pixels": {
            "static": (0, 5, 10),
            "gripper": (1, 0, 0),
        },
    }


def test_scene_d_tcp_world_to_robot_base_transform() -> None:
    world = np.asarray([CALVIN_SCENE_D_ROBOT_BASE_POSITION_M, (0.0, 0.0, 0.0)])
    actual = calvin_scene_d_tcp_robot_base_position(world)
    np.testing.assert_allclose(actual[0], np.zeros(3))
    np.testing.assert_allclose(actual[1], (0.34, 0.46, -0.24))
    np.testing.assert_allclose(calvin_tcp_robot_base_position(world), actual)
    assert CALVIN_SCENE_D_ROBOT_BASE_POSITION_M == CALVIN_ROBOT_BASE_POSITION_M
    assert set(CALVIN_SCENE_CONFIG_SHA256) == {
        "calvin_scene_A",
        "calvin_scene_B",
        "calvin_scene_C",
        "calvin_scene_D",
    }


def test_behavior_support_summarizes_real_segment_without_authorizing_training() -> None:
    summary = summarize_calvin_behavior_support(**_valid_inputs())

    assert summary.target_motion_rank == 1
    assert summary.maximum_motion_identity_key == "movable/block_blue"
    assert summary.maximum_identity_displacement_m == pytest.approx(0.4)
    assert summary.target_motion_margin_to_best_other_m == pytest.approx(0.38)
    assert dict(summary.identity_max_displacements_m) == pytest.approx(
        {"movable/block_blue": 0.4, "movable/block_red": 0.02}
    )
    assert summary.target_net_displacement_m == pytest.approx(0.4)
    assert summary.target_max_displacement_m == pytest.approx(0.4)
    assert summary.initial_tcp_target_distance_m == pytest.approx(0.5)
    assert summary.minimum_tcp_target_distance_m == pytest.approx(0.01)
    assert summary.closest_global_index == 11
    assert summary.maximum_displacement_global_index == 12
    assert dict(summary.camera_visible_frame_counts) == {"gripper": 1, "static": 2}
    assert dict(summary.camera_max_visible_pixels) == {"gripper": 1, "static": 10}
    assert summary.geometry_observation_scope == "aabb-centre-translation-only"
    assert summary.task_success_certified is False
    assert summary.training_authorized is False
    assert calvin_behavior_review_keyframes(summary) == (10, 11, 12)
    nonchronological_roles = replace(
        summary,
        global_indices=(10, 11, 12, 13),
        closest_global_index=12,
        maximum_displacement_global_index=11,
    )
    assert calvin_behavior_review_keyframes(nonchronological_roles) == (10, 11, 12, 13)
    assert summary.to_dict()["training_authorized"] is False


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("target_identity_key", "movable/block_red", "reviewed CALVIN protocol"),
        ("global_indices", (10, 12, 13), "contiguous source frames"),
        ("actions", np.ones((4, 7)), "align to segment transitions or frames"),
        ("visible_target_pixels", {"static": (1, 1, 1)}, "two pinned CALVIN cameras"),
        (
            "visible_target_pixels",
            {"static": (1, -1, 1), "gripper": (1, 1, 1)},
            "nonnegative frame-aligned integers",
        ),
    ],
)
def test_behavior_support_fails_closed_on_invalid_evidence(
    field: str,
    replacement: object,
    message: str,
) -> None:
    values = _valid_inputs()
    values[field] = replacement
    with pytest.raises(ContractError, match=message):
        summarize_calvin_behavior_support(**values)


def test_behavior_support_rejects_ambiguous_task_semantics() -> None:
    values = _valid_inputs()
    values["task_key"] = "stack_block"
    with pytest.raises(ContractError, match="reviewed exact action target"):
        summarize_calvin_behavior_support(**values)


def test_behavior_segment_selection_is_task_scene_stratified_and_evidence_independent() -> None:
    segments = tuple(
        CalvinLanguageSegment(index, index * 10, index * 10 + 2, task, task, index)
        for index, task in enumerate(("b", "a", "b", "a", "b", "a", "b", "a"))
    )
    scenes = {
        0: "calvin_scene_A",
        1: "calvin_scene_A",
        2: "calvin_scene_A",
        3: "calvin_scene_A",
        4: "calvin_scene_B",
        5: "calvin_scene_B",
        6: "calvin_scene_B",
        7: "calvin_scene_B",
    }
    selected = select_calvin_behavior_segments(
        segments,
        samples_per_task_scene=2,
        scene_by_segment_index=scenes,
    )
    assert tuple((item.task_key, item.index) for item in selected) == (
        ("a", 1),
        ("a", 3),
        ("a", 5),
        ("a", 7),
        ("b", 0),
        ("b", 2),
        ("b", 4),
        ("b", 6),
    )


@pytest.mark.parametrize("samples_per_task_scene", (0, -1, True, 1.5))
def test_behavior_segment_selection_rejects_invalid_sample_count(
    samples_per_task_scene: object,
) -> None:
    with pytest.raises(ContractError, match="samples_per_task_scene"):
        select_calvin_behavior_segments(
            (),
            samples_per_task_scene=samples_per_task_scene,  # type: ignore[arg-type]
            scene_by_segment_index={},
        )


def test_behavior_segment_selection_rejects_incomplete_scene_assignment() -> None:
    segment = CalvinLanguageSegment(1, 10, 12, "task", "task", 0)
    with pytest.raises(ContractError, match="no scene assignment"):
        select_calvin_behavior_segments(
            (segment,),
            samples_per_task_scene=1,
            scene_by_segment_index={},
        )
