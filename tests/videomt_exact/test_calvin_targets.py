from __future__ import annotations

import numpy as np
import pytest
import torch

import picf_next.videomt_exact.calvin_targets as calvin_targets
from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import CALVIN_OBJECT_GEOMETRY_CONTRACT
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinVisibleOwnerRaster,
)
from picf_next.videomt_exact.calvin_targets import (
    prepare_calvin_videomt_clip,
    prepare_calvin_videomt_training_clip,
)


def _physical_frame(
    rgb: np.ndarray,
    owner: np.ndarray,
    *,
    supervised: np.ndarray | None = None,
    keys: tuple[str, ...] = ("blue_block", "pink_block"),
) -> CalvinPhysicalSupervisionFrame:
    if supervised is None:
        supervised = np.ones(owner.shape, dtype=np.bool_)
    camera = CalvinVisibleOwnerRaster(
        camera_name="static",
        host_image_key="observation.images.image",
        owner_index=np.asarray(owner, dtype=np.uint8),
        owner_supervised=np.asarray(supervised, dtype=np.bool_),
        source_rgb_sha256=source_array_sha256("rgb_static", rgb),
        source_depth_sha256="0" * 64,
        rgb_mae=0.0,
        depth_mae_m=0.0,
        depth_p95_m=0.0,
        depth_consistent_fraction=float(np.asarray(supervised).mean()),
    )
    count = len(keys)
    return CalvinPhysicalSupervisionFrame(
        identity_keys=keys,
        geometry=torch.zeros(count, 3),
        geometry_variance=torch.zeros(count, 3),
        geometry_supervised=torch.ones(count, 3, dtype=torch.bool),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=(camera,),
    )


def test_calvin_target_preserves_identity_visibility_context_and_geometry() -> None:
    rgb0 = np.zeros((200, 200, 3), dtype=np.uint8)
    rgb1 = np.ones((200, 200, 3), dtype=np.uint8)
    owner0 = np.zeros((200, 200), dtype=np.uint8)
    owner1 = np.zeros((200, 200), dtype=np.uint8)
    owner0[25:75, 50:100] = 1
    owner1[125:175, 100:150] = 2
    clip = prepare_calvin_videomt_clip(
        (rgb0, rgb1),
        (_physical_frame(rgb0, owner0), _physical_frame(rgb1, owner1)),
        short_edge=200,
        max_size=200,
        size_divisibility=8,
    )

    assert clip.identity_keys == ("blue_block", "pink_block")
    assert clip.frames.model_input.shape == (2, 3, 200, 200)
    assert clip.target["masks"].shape == (2, 2, 200, 200)
    assert clip.target["ids"].tolist() == [[0, -1], [-1, 1]]
    assert clip.target["masks"][0, 0].sum() == 2500
    assert clip.target["masks"][0, 1].sum() == 0
    assert clip.target["masks"][1, 0].sum() == 0
    assert clip.target["masks"][1, 1].sum() == 2500
    assert not (clip.target["masks"].sum(dim=0) > 1).any()


def test_calvin_target_rejects_rgb_supervision_mismatch() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    changed = rgb.copy()
    changed[0, 0] = 1
    frame = _physical_frame(rgb, np.zeros((200, 200), dtype=np.uint8))
    with pytest.raises(ContractError, match="RGB/source supervision mismatch"):
        prepare_calvin_videomt_clip((changed,), (frame,), short_edge=8, max_size=8)


def test_calvin_target_preserves_unknown_pixels_outside_masks_and_validity() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    owner = np.ones((200, 200), dtype=np.uint8)
    supervised = np.ones((200, 200), dtype=np.bool_)
    supervised[0, 0] = False
    frame = _physical_frame(rgb, owner, supervised=supervised)
    clip = prepare_calvin_videomt_clip((rgb,), (frame,), short_edge=200, max_size=200)

    assert not clip.target["valid_pixels"][0, 0, 0]
    assert not clip.target["masks"][0, 0, 0, 0]
    assert clip.target["valid_pixels"].sum() == 200 * 200 - 1


def test_calvin_target_drops_never_visible_inventory_rows() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    owner = np.ones((200, 200), dtype=np.uint8)
    clip = prepare_calvin_videomt_clip(
        (rgb,),
        (_physical_frame(rgb, owner, keys=("blue_block", "pink_block")),),
        short_edge=8,
        max_size=8,
    )
    assert clip.identity_keys == ("blue_block",)
    assert clip.target["ids"].tolist() == [[0]]


def test_physical_supervision_changes_only_loss_targets_not_model_inputs() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    left = np.zeros((200, 200), dtype=np.uint8)
    right = np.zeros((200, 200), dtype=np.uint8)
    left[50:100, 25:75] = 1
    right[50:100, 125:175] = 1

    left_clip = prepare_calvin_videomt_clip(
        (rgb,),
        (_physical_frame(rgb, left),),
        short_edge=200,
        max_size=200,
    )
    right_clip = prepare_calvin_videomt_clip(
        (rgb,),
        (_physical_frame(rgb, right),),
        short_edge=200,
        max_size=200,
    )

    torch.testing.assert_close(left_clip.frames.model_input, right_clip.frames.model_input)
    assert not torch.equal(left_clip.target["masks"], right_clip.target["masks"])


def test_released_training_augmentation_is_clip_consistent() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    rgb[20:80, 30:90] = (25, 100, 200)
    owner = np.zeros((200, 200), dtype=np.uint8)
    owner[20:80, 30:90] = 1
    frames = tuple(rgb.copy() for _ in range(5))
    supervision = tuple(_physical_frame(frame, owner) for frame in frames)
    np.random.seed(198)

    clip = prepare_calvin_videomt_training_clip(
        frames,
        supervision,
        short_edges=(224,),
        max_size=224,
    )

    assert clip.frames.model_input.shape == (5, 3, 224, 224)
    assert clip.target["masks"].shape == (1, 5, 224, 224)
    assert clip.target["valid_pixels"].shape == (5, 224, 224)
    for time_index in range(1, 5):
        torch.testing.assert_close(
            clip.target["masks"][0, 0],
            clip.target["masks"][0, time_index],
        )
        assert np.array_equal(clip.frames.resized_rgb[0], clip.frames.resized_rgb[time_index])


def test_released_training_augmentation_requires_official_five_frames() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    owner = np.ones((200, 200), dtype=np.uint8)
    frame = _physical_frame(rgb, owner)
    with pytest.raises(ContractError, match="five-frame"):
        prepare_calvin_videomt_training_clip((rgb, rgb), (frame, frame))


def test_released_crop_marks_removed_instances_absent_and_filters_empty_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    owner = np.zeros((200, 200), dtype=np.uint8)
    owner[:4, :4] = 1
    frames = tuple(rgb.copy() for _ in range(5))
    supervision = tuple(_physical_frame(frame, owner, keys=("edge_object",)) for frame in frames)
    monkeypatch.setattr(calvin_targets, "VIDEOMT_YTVIS19_CROP_PROBABILITY", 1.0)
    np.random.seed(0)

    clip = prepare_calvin_videomt_training_clip(
        frames,
        supervision,
        short_edges=(224,),
        max_size=224,
    )

    assert clip.identity_keys == ()
    assert clip.target["labels"].shape == (0,)
    assert clip.target["ids"].shape == (0, 5)
    assert clip.target["masks"].shape == (0, 5, 224, 224)


def test_post_crop_survivors_are_bijectively_reindexed_for_consistent_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    owner = np.zeros((200, 200), dtype=np.uint8)
    owner[:4, :4] = 1
    owner[90:130, 90:130] = 2
    frames = tuple(rgb.copy() for _ in range(5))
    supervision = tuple(
        _physical_frame(frame, owner, keys=("edge_object", "center_object")) for frame in frames
    )
    monkeypatch.setattr(calvin_targets, "VIDEOMT_YTVIS19_CROP_PROBABILITY", 1.0)
    np.random.seed(0)

    clip = prepare_calvin_videomt_training_clip(
        frames,
        supervision,
        short_edges=(224,),
        max_size=224,
    )

    assert clip.identity_keys == ("center_object",)
    assert clip.target["ids"].tolist() == [[0, 0, 0, 0, 0]]
    assert (clip.target["masks"].sum(dim=(2, 3)) > 0).all()
