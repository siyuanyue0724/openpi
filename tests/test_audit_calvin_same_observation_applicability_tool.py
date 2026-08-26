from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
import torch
from PIL import Image

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinLanguageSegment
from picf_next.data.calvin_geometry_schema import CALVIN_OBJECT_GEOMETRY_CONTRACT
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinVisibleOwnerRaster,
)
from picf_next.data.calvin_simulator_geometry import CalvinSceneRange
from picf_next.data.calvin_task_applicability import (
    CalvinSameObservationGroup,
    CalvinSameObservationVariant,
)
from tools.audit_calvin_same_observation_applicability import (
    _AcceptedFrame,
    _StatefulResetBinding,
    _verify_source_binding,
    render_group_visual,
    stratified_partition_reset_candidates,
)


def _variant(task_key: str, instruction: str, target: str) -> CalvinSameObservationVariant:
    return CalvinSameObservationVariant(
        task_key=task_key,
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        target_identity_key=target,
        proof=f"proof:{task_key}",
    )


def _frame_fixture() -> tuple[dict[str, np.ndarray], CalvinPhysicalSupervisionFrame]:
    arrays = {
        "rgb_static": np.full((200, 200, 3), 180, dtype=np.uint8),
        "depth_static": np.ones((200, 200), dtype=np.float32),
        "rgb_gripper": np.full((84, 84, 3), 100, dtype=np.uint8),
        "depth_gripper": np.ones((84, 84), dtype=np.float32),
    }
    cameras = []
    for camera_name, rgb_field, depth_field in (
        ("static", "rgb_static", "depth_static"),
        ("gripper", "rgb_gripper", "depth_gripper"),
    ):
        height, width = arrays[depth_field].shape
        owner = np.zeros((height, width), dtype=np.uint8)
        owner[10:30, 10:30] = 1
        owner[40:60, 40:60] = 2
        supervised = np.ones_like(owner, dtype=np.bool_)
        cameras.append(
            CalvinVisibleOwnerRaster(
                camera_name=camera_name,
                host_image_key=(
                    "observation.images.image"
                    if camera_name == "static"
                    else "observation.images.wrist_image"
                ),
                owner_index=owner,
                owner_supervised=supervised,
                source_rgb_sha256=source_array_sha256(rgb_field, arrays[rgb_field]),
                source_depth_sha256=source_array_sha256(depth_field, arrays[depth_field]),
                rgb_mae=0.0,
                depth_mae_m=0.0,
                depth_p95_m=0.0,
                depth_consistent_fraction=1.0,
            )
        )
    dimension = CALVIN_OBJECT_GEOMETRY_CONTRACT.dimension
    return arrays, CalvinPhysicalSupervisionFrame(
        identity_keys=("part/table/button_link", "movable/block_blue"),
        geometry=torch.zeros(2, dimension),
        geometry_variance=torch.zeros(2, dimension),
        geometry_supervised=torch.ones(2, dimension, dtype=torch.bool),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=tuple(cameras),
    )


def _segments() -> tuple[CalvinLanguageSegment, ...]:
    return tuple(
        CalvinLanguageSegment(
            index=scene_index * 10 + offset,
            start=scene_index * 100 + offset * 2,
            end=scene_index * 100 + offset * 2 + 1,
            task_key=f"task_{offset}",
            instruction=f"instruction {offset}",
            episode_index=scene_index,
        )
        for scene_index in range(4)
        for offset in range(10)
    )


def test_stratified_reset_candidates_are_balanced_unique_and_deterministic() -> None:
    segments = _segments()
    scenes = tuple(
        CalvinSceneRange(f"calvin_scene_{name}", index * 100, index * 100 + 99)
        for index, name in enumerate("ABCD")
    )
    segment_indices = tuple(segment.index for segment in segments)
    source_episode_indices = tuple(range(4))

    first = stratified_partition_reset_candidates(
        segments,
        scenes,
        admitted_segment_indices=segment_indices,
        admitted_source_episode_indices=source_episode_indices,
        sample_count=12,
        seed=17,
    )
    second = stratified_partition_reset_candidates(
        segments,
        scenes,
        admitted_segment_indices=segment_indices,
        admitted_source_episode_indices=source_episode_indices,
        sample_count=12,
        seed=17,
    )

    assert first == second
    assert len(first) == len({item.source_global_index for item in first}) == 12
    assert [
        sum(scene.contains(item.source_global_index) for item in first) for scene in scenes
    ] == [3, 3, 3, 3]
    assert all(item.language_segment_index in segment_indices for item in first)
    assert all(item.source_episode_index in source_episode_indices for item in first)


def test_stratified_reset_candidates_reject_excess_request() -> None:
    segments = _segments()[:2]
    scenes = (CalvinSceneRange("calvin_scene_A", 0, 99),)

    with pytest.raises(ContractError, match="exceeds"):
        stratified_partition_reset_candidates(
            segments,
            scenes,
            admitted_segment_indices=tuple(segment.index for segment in segments),
            admitted_source_episode_indices=(0,),
            sample_count=3,
            seed=17,
        )


def test_stratified_reset_candidates_fail_closed_to_frozen_partition() -> None:
    segments = (
        CalvinLanguageSegment(5, 10, 11, "task_a", "instruction a", 1),
        CalvinLanguageSegment(3, 10, 11, "task_b", "instruction b", 1),
        CalvinLanguageSegment(7, 20, 21, "task_c", "instruction c", 2),
    )
    selected = stratified_partition_reset_candidates(
        segments,
        (CalvinSceneRange("calvin_scene_A", 0, 99),),
        admitted_segment_indices=(3, 5),
        admitted_source_episode_indices=(1,),
        sample_count=1,
        seed=17,
    )

    assert len(selected) == 1
    assert selected[0].source_global_index == 10
    assert selected[0].language_segment_index == 3
    assert selected[0].source_episode_index == 1


def test_stratified_reset_candidates_can_cover_an_uneven_partition() -> None:
    segments = (
        CalvinLanguageSegment(0, 10, 11, "task_a", "instruction a", 0),
        CalvinLanguageSegment(1, 100, 101, "task_b", "instruction b", 1),
        CalvinLanguageSegment(2, 110, 111, "task_c", "instruction c", 1),
        CalvinLanguageSegment(3, 120, 121, "task_d", "instruction d", 1),
    )
    selected = stratified_partition_reset_candidates(
        segments,
        (
            CalvinSceneRange("calvin_scene_A", 0, 99),
            CalvinSceneRange("calvin_scene_B", 100, 199),
        ),
        admitted_segment_indices=(0, 1, 2, 3),
        admitted_source_episode_indices=(0, 1),
        sample_count=4,
        seed=17,
    )

    assert {item.source_global_index for item in selected} == {10, 100, 110, 120}


def test_stratified_reset_candidates_reject_split_metadata_drift() -> None:
    segments = (CalvinLanguageSegment(5, 10, 11, "task_a", "instruction a", 1),)

    with pytest.raises(ContractError, match="another source episode"):
        stratified_partition_reset_candidates(
            segments,
            (CalvinSceneRange("calvin_scene_A", 0, 99),),
            admitted_segment_indices=(5,),
            admitted_source_episode_indices=(2,),
            sample_count=1,
            seed=17,
        )


def test_source_binding_and_task_labelled_visual_are_exact() -> None:
    arrays, physical = _frame_fixture()
    hashes = _verify_source_binding(arrays, physical)
    group = CalvinSameObservationGroup(
        source_global_index=19,
        source_state_sha256="a" * 64,
        variants=(
            _variant(
                "turn_on_led",
                "toggle the button to turn on the led",
                "part/table/button_link",
            ),
            _variant(
                "lift_blue_block_table",
                "lift the blue block",
                "movable/block_blue",
            ),
        ),
    )
    record = _AcceptedFrame(
        scene="calvin_scene_A",
        group=group,
        stateful_reset_binding=_StatefulResetBinding(
            language_segment_index=3,
            source_episode_index=2,
            source_instruction_sha256="b" * 64,
            source_task_key="turn_on_led",
            stateful_episode_key="calvin-language-segment-00000003",
            stateful_sample_key=(
                "calvin-language-segment-00000003/transition-00000000-frame-00000019"
            ),
            transition_index=0,
        ),
        visible_support=(
            {
                "camera_pixel_counts": {"gripper": 400, "static": 400},
                "identity_key": "part/table/button_link",
                "total_pixel_count": 800,
            },
            {
                "camera_pixel_counts": {"gripper": 400, "static": 400},
                "identity_key": "movable/block_blue",
                "total_pixel_count": 800,
            },
        ),
        applicable_tasks=(),
        source_sensor_sha256=hashes,
    )

    payload = render_group_visual(
        record=record,
        frame_arrays=arrays,
        physical=physical,
    )

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(payload)) as image:
        assert image.width > 700
        assert image.height > 450

    altered = {**arrays, "rgb_static": arrays["rgb_static"].copy()}
    altered["rgb_static"][0, 0, 0] = 0
    with pytest.raises(ContractError, match="another source frame"):
        _verify_source_binding(altered, physical)
