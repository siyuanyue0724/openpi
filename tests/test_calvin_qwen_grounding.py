from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import CALVIN_OBJECT_GEOMETRY_CONTRACT
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinVisibleOwnerRaster,
)
from picf_next.data.calvin_qwen_grounding import (
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    CalvinQwenGroundingRecord,
    build_calvin_qwen_grounding_distractors,
    build_calvin_qwen_grounding_records,
    build_calvin_qwen_scene_grounding_record,
    minimum_projected_target_mass_for_raw_patch,
    qwen3vl_normalized_bbox,
    qwen_grounding_label,
    tight_visible_owner_bbox,
)


@pytest.mark.parametrize(
    ("task_key", "target_identity_key", "expected_label"),
    (
        ("push_red_block_left", "movable/block_red", "red block"),
        ("push_blue_block_left", "movable/block_blue", "blue block"),
        ("push_pink_block_left", "movable/block_pink", "pink block"),
        ("open_drawer", "part/table/drawer_link", "drawer"),
        ("move_slider_left", "part/table/slide_link", "sliding door"),
        ("turn_on_led", "part/table/button_link", "push button"),
        ("turn_on_lightbulb", "part/table/switch_link", "light switch"),
    ),
)
def test_native_grounding_labels_match_physical_owner_granularity(
    task_key: str,
    target_identity_key: str,
    expected_label: str,
) -> None:
    image = _image("static")
    image.flags.writeable = False
    record = CalvinQwenGroundingRecord(
        global_index=0,
        task_key=task_key,
        instruction="complete the task",
        target_identity_key=target_identity_key,
        camera_name="static",
        host_image_key="observation.images.image",
        bbox_xyxy=(10, 20, 30, 40),
        image=image,
        source_rgb_sha256=source_array_sha256("rgb_static", image),
    )

    assert record.qwen_label == expected_label
    assert json.loads(record.assistant_text) == [
        {
            "label": expected_label,
            "bbox_2d": [50, 100, 150, 200],
        }
    ]
    assert record.assistant_text.index('"label"') < record.assistant_text.index('"bbox_2d"')


def _image(camera_name: str) -> np.ndarray:
    shape = (200, 200, 3) if camera_name == "static" else (84, 84, 3)
    return np.zeros(shape, dtype=np.uint8)


def _camera(camera_name: str, owner_index: np.ndarray, image: np.ndarray):
    source_name = "rgb_static" if camera_name == "static" else "rgb_gripper"
    return CalvinVisibleOwnerRaster(
        camera_name=camera_name,
        host_image_key=(
            "observation.images.image"
            if camera_name == "static"
            else "observation.images.wrist_image"
        ),
        owner_index=owner_index,
        owner_supervised=np.ones_like(owner_index, dtype=np.bool_),
        source_rgb_sha256=source_array_sha256(source_name, image),
        source_depth_sha256=("1" if camera_name == "static" else "2") * 64,
        rgb_mae=0.0,
        depth_mae_m=0.0,
        depth_p95_m=0.0,
        depth_consistent_fraction=1.0,
    )


def _frame(
    static_owner: np.ndarray,
    gripper_owner: np.ndarray,
    static_image: np.ndarray,
    gripper_image: np.ndarray,
) -> CalvinPhysicalSupervisionFrame:
    dimension = CALVIN_OBJECT_GEOMETRY_CONTRACT.dimension
    return CalvinPhysicalSupervisionFrame(
        identity_keys=("movable/block_red", "part/table/button_link"),
        geometry=torch.zeros(2, dimension),
        geometry_variance=torch.zeros(2, dimension),
        geometry_supervised=torch.ones(2, dimension, dtype=torch.bool),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=(
            _camera("static", static_owner, static_image),
            _camera("gripper", gripper_owner, gripper_image),
        ),
    )


def _scene_frame(
    static_owner: np.ndarray,
    gripper_owner: np.ndarray,
    static_image: np.ndarray,
    gripper_image: np.ndarray,
) -> CalvinPhysicalSupervisionFrame:
    dimension = CALVIN_OBJECT_GEOMETRY_CONTRACT.dimension
    identity_count = len(CALVIN_QWEN_SCENE_IDENTITY_ORDER)
    return CalvinPhysicalSupervisionFrame(
        identity_keys=CALVIN_QWEN_SCENE_IDENTITY_ORDER,
        geometry=torch.zeros(identity_count, dimension),
        geometry_variance=torch.zeros(identity_count, dimension),
        geometry_supervised=torch.ones(identity_count, dimension, dtype=torch.bool),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=(
            _camera("static", static_owner, static_image),
            _camera("gripper", gripper_owner, gripper_image),
        ),
    )


def _visible_scene_fixture() -> tuple[
    CalvinPhysicalSupervisionFrame,
    CalvinQwenGroundingRecord,
]:
    static_image = _image("static")
    gripper_image = _image("gripper")
    static_owner = np.zeros((200, 200), dtype=np.uint8)
    # Eight identities exceed one Qwen raw-patch-equivalent (157 pixels),
    # pink is subpatch-visible, and red is fully absent.
    visible_owner_ids = (1, 4, 5, 6, 7, 8, 9, 10)
    for slot, owner_id in enumerate(visible_owner_ids):
        row = (slot // 4) * 24
        column = (slot % 4) * 24
        static_owner[row : row + 13, column : column + 13] = owner_id
    static_owner[100, 100] = 2
    frame = _scene_frame(
        static_owner,
        np.zeros((84, 84), dtype=np.uint8),
        static_image,
        gripper_image,
    )
    target = build_calvin_qwen_grounding_records(
        global_index=42,
        task_key="turn_on_led",
        instruction="toggle the button to turn on the led",
        observation_images={
            "observation.images.image": static_image,
            "observation.images.wrist_image": gripper_image,
        },
        physical_frame=frame,
    )[0]
    return frame, target


def test_turn_on_led_targets_button_owner_not_first_block() -> None:
    static_image = _image("static")
    gripper_image = _image("gripper")
    static_owner = np.zeros((200, 200), dtype=np.uint8)
    static_owner[20:30, 150:160] = 2
    gripper_owner = np.zeros((84, 84), dtype=np.uint8)
    records = build_calvin_qwen_grounding_records(
        global_index=1_001_913,
        task_key="turn_on_led",
        instruction="toggle the button to turn on the led",
        observation_images={
            "observation.images.image": static_image,
            "observation.images.wrist_image": gripper_image,
        },
        physical_frame=_frame(
            static_owner,
            gripper_owner,
            static_image,
            gripper_image,
        ),
    )

    assert len(records) == 1
    record = records[0]
    assert record.target_identity_key == "part/table/button_link"
    assert record.camera_name == "static"
    assert record.bbox_xyxy == (150, 20, 160, 30)
    assert record.qwen_bbox_xyxy == (750, 100, 800, 150)
    assert record.qwen_label == "push button"
    assert record.assistant_text == ('[{"label":"push button","bbox_2d":[750,100,800,150]}]')
    assert not record.image.flags.writeable

    messages = record.qwen_messages(image_value="visible-image")
    assert messages[0]["content"][0] == {"type": "image", "image": "visible-image"}
    request = messages[0]["content"][1]["text"]
    assert record.instruction in request
    assert record.task_key not in request
    assert record.target_identity_key not in request
    assert messages[1]["content"][0]["text"] == record.assistant_text
    user_messages = record.qwen_user_messages(image_value="generation-image")
    assert len(user_messages) == 1
    assert user_messages[0]["content"][0] == {"type": "image", "image": "generation-image"}
    assert user_messages[0]["content"][1]["text"] == request


def test_invisible_exact_target_omits_grounding_factor() -> None:
    static_image = _image("static")
    gripper_image = _image("gripper")
    records = build_calvin_qwen_grounding_records(
        global_index=7,
        task_key="turn_on_led",
        instruction="press the button",
        observation_images={
            "observation.images.image": static_image,
            "observation.images.wrist_image": gripper_image,
        },
        physical_frame=_frame(
            np.zeros((200, 200), dtype=np.uint8),
            np.zeros((84, 84), dtype=np.uint8),
            static_image,
            gripper_image,
        ),
    )
    assert records == ()


def test_same_image_distractor_changes_only_candidate_box() -> None:
    static_image = _image("static")
    gripper_image = _image("gripper")
    static_owner = np.zeros((200, 200), dtype=np.uint8)
    static_owner[5:15, 10:20] = 1
    static_owner[20:30, 150:160] = 2
    frame = _frame(
        static_owner,
        np.zeros((84, 84), dtype=np.uint8),
        static_image,
        gripper_image,
    )
    record = build_calvin_qwen_grounding_records(
        global_index=10,
        task_key="turn_on_led",
        instruction="press the button",
        observation_images={
            "observation.images.image": static_image,
            "observation.images.wrist_image": gripper_image,
        },
        physical_frame=frame,
    )[0]
    distractors = build_calvin_qwen_grounding_distractors(record, frame)

    assert len(distractors) == 1
    assert distractors[0].distractor_identity_key == "movable/block_red"
    candidate = distractors[0].candidate_record
    assert candidate.target_identity_key == record.target_identity_key
    assert candidate.bbox_xyxy == (10, 5, 20, 15)
    assert candidate.image is record.image


def test_ambiguous_task_omits_grounding_factor() -> None:
    static_image = _image("static")
    gripper_image = _image("gripper")
    records = build_calvin_qwen_grounding_records(
        global_index=8,
        task_key="stack_block",
        instruction="stack one block on another",
        observation_images={
            "observation.images.image": static_image,
            "observation.images.wrist_image": gripper_image,
        },
        physical_frame=_frame(
            np.zeros((200, 200), dtype=np.uint8),
            np.zeros((84, 84), dtype=np.uint8),
            static_image,
            gripper_image,
        ),
    )
    assert records == ()


def test_source_image_drift_fails_closed() -> None:
    static_image = _image("static")
    gripper_image = _image("gripper")
    static_owner = np.zeros((200, 200), dtype=np.uint8)
    static_owner[1:3, 4:7] = 2
    frame = _frame(
        static_owner,
        np.zeros((84, 84), dtype=np.uint8),
        static_image,
        gripper_image,
    )
    changed = static_image.copy()
    changed[0, 0, 0] = 1
    with pytest.raises(ContractError, match="hash differs"):
        build_calvin_qwen_grounding_records(
            global_index=9,
            task_key="turn_on_led",
            instruction="press the button",
            observation_images={
                "observation.images.image": changed,
                "observation.images.wrist_image": gripper_image,
            },
            physical_frame=frame,
        )


def test_tight_visible_owner_bbox_rejects_empty_or_nonboolean_support() -> None:
    with pytest.raises(ContractError, match="cannot be empty"):
        tight_visible_owner_bbox(np.zeros((2, 2), dtype=np.bool_))
    with pytest.raises(ContractError, match="boolean"):
        tight_visible_owner_bbox(np.zeros((2, 2), dtype=np.uint8))


def test_qwen3vl_bbox_uses_relative_1000_grid_not_source_pixels() -> None:
    assert qwen3vl_normalized_bbox((0, 0, 200, 200), width=200, height=200) == (
        0,
        0,
        1000,
        1000,
    )
    assert qwen3vl_normalized_bbox((21, 42, 63, 84), width=84, height=84) == (
        250,
        500,
        750,
        1000,
    )
    with pytest.raises(ContractError, match="outside"):
        qwen3vl_normalized_bbox((0, 0, 201, 200), width=200, height=200)


def test_scene_observability_threshold_is_one_exact_raw_patch_mass() -> None:
    assert minimum_projected_target_mass_for_raw_patch(merge_size=2) == 0.25
    assert minimum_projected_target_mass_for_raw_patch(merge_size=4) == 0.0625
    with pytest.raises(ContractError, match="positive integer"):
        minimum_projected_target_mass_for_raw_patch(merge_size=0)


def test_scene_grounding_uses_fixed_inventory_and_unknown_visibility_partition() -> None:
    frame, target = _visible_scene_fixture()
    record = build_calvin_qwen_scene_grounding_record(
        global_index=target.global_index,
        camera_name=target.camera_name,
        image=target.image,
        physical_frame=frame,
        category_identity_order=CALVIN_QWEN_SCENE_IDENTITY_ORDER,
        visual_lattice=8,
    )

    assert np.array_equal(record.image, target.image)
    assert not record.image.flags.writeable
    assert record.source_rgb_sha256 == target.source_rgb_sha256
    assert record.minimum_projected_target_mass == 0.25
    assert record.image_grid_thw == (1, 16, 16)
    assert record.patch_size == 16
    assert record.merge_size == 2
    assert tuple(item.identity_key for item in record.objects) == (
        "movable/block_blue",
        "part/table/button_link",
        "part/table/drawer_link",
        "part/table/led_link",
        "part/table/light_link",
        "part/table/plank_link",
        "part/table/slide_link",
        "part/table/switch_link",
    )
    assert record.subpatch_visible_identity_keys == ("movable/block_pink",)
    assert record.absent_identity_keys == ("movable/block_red",)
    answer = json.loads(record.assistant_text)
    assert [item["label"] for item in answer] == [
        qwen_grounding_label(item.identity_key) for item in record.objects
    ]
    assert all(set(item) == {"label", "bbox_2d"} for item in answer)

    request = record.grounding_request
    assert all(
        qwen_grounding_label(identity_key) in request
        for identity_key in CALVIN_QWEN_SCENE_IDENTITY_ORDER
    )
    assert target.instruction not in request
    assert target.task_key not in request
    assert all(identity_key not in request for identity_key in CALVIN_QWEN_SCENE_IDENTITY_ORDER)
    assert "157" not in request
    assert "subpatch" not in request
    assert "absent" not in request


def test_scene_grounding_category_order_is_a_true_same_image_counterfactual() -> None:
    frame, target = _visible_scene_fixture()
    canonical = build_calvin_qwen_scene_grounding_record(
        global_index=target.global_index,
        camera_name=target.camera_name,
        image=target.image,
        physical_frame=frame,
        category_identity_order=CALVIN_QWEN_SCENE_IDENTITY_ORDER,
        visual_lattice=8,
    )
    reverse = build_calvin_qwen_scene_grounding_record(
        global_index=target.global_index,
        camera_name=target.camera_name,
        image=target.image,
        physical_frame=frame,
        category_identity_order=tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER)),
        visual_lattice=8,
    )

    assert np.array_equal(canonical.image, reverse.image)
    assert not canonical.image.flags.writeable
    assert not reverse.image.flags.writeable
    assert canonical.source_rgb_sha256 == reverse.source_rgb_sha256
    assert canonical.category_identity_order == tuple(reversed(reverse.category_identity_order))
    assert tuple(item.identity_key for item in canonical.objects) == tuple(
        reversed(tuple(item.identity_key for item in reverse.objects))
    )
    assert {
        item.identity_key: (
            item.bbox_xyxy,
            item.visible_owner_pixels,
            item.projected_target_mass,
            item.positive_visual_token_count,
        )
        for item in canonical.objects
    } == {
        item.identity_key: (
            item.bbox_xyxy,
            item.visible_owner_pixels,
            item.projected_target_mass,
            item.positive_visual_token_count,
        )
        for item in reverse.objects
    }


def test_scene_grounding_rejects_unreviewed_or_incomplete_inventory() -> None:
    static_image = _image("static")
    gripper_image = _image("gripper")
    static_owner = np.zeros((200, 200), dtype=np.uint8)
    static_owner[:13, :13] = 2
    incomplete = _frame(
        static_owner,
        np.zeros((84, 84), dtype=np.uint8),
        static_image,
        gripper_image,
    )
    target = build_calvin_qwen_grounding_records(
        global_index=43,
        task_key="turn_on_led",
        instruction="press the button",
        observation_images={
            "observation.images.image": static_image,
            "observation.images.wrist_image": gripper_image,
        },
        physical_frame=incomplete,
    )[0]
    with pytest.raises(ContractError, match="inventory"):
        build_calvin_qwen_scene_grounding_record(
            global_index=target.global_index,
            camera_name=target.camera_name,
            image=target.image,
            physical_frame=incomplete,
            category_identity_order=CALVIN_QWEN_SCENE_IDENTITY_ORDER,
            visual_lattice=8,
        )
