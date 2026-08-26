"""Fail-closed CALVIN visible-owner targets for exact VidEoMT adaptation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data import transforms as T
from detectron2.structures import ImageList

from picf_next._vendor.videomt.data_video.augmentation import (
    RandomApplyClip,
    RandomCropClip,
    RandomFlip,
    ResizeShortestEdge,
)
from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CAMERA_SPECS,
    source_array_sha256,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinVisibleOwnerRaster,
)
from picf_next.videomt_exact.preprocessing import (
    VIDEOMT_SIZE_DIVISIBILITY,
    VIDEOMT_TEST_MAX_SIZE,
    VIDEOMT_TEST_SHORT_EDGE,
    PreparedVidEoMTFrames,
    prepare_rgb_frames,
)
from picf_next.videomt_exact.runtime import normalize_rgb_255

VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES = (
    320,
    352,
    392,
    416,
    448,
    480,
    512,
    544,
    576,
    608,
    640,
)
VIDEOMT_YTVIS19_TRAIN_MAX_SIZE = 768
VIDEOMT_YTVIS19_CROP_PRE_RESIZE_EDGES = (400, 500, 600)
VIDEOMT_YTVIS19_CROP_PRE_RESIZE_MAX_SIZE = 1333
VIDEOMT_YTVIS19_CROP_SIZE = (384, 600)
VIDEOMT_YTVIS19_CROP_TYPE = "absolute_range"
VIDEOMT_YTVIS19_CROP_PROBABILITY = 0.5
VIDEOMT_YTVIS19_CLIP_LENGTH = 5


@dataclass(frozen=True, slots=True)
class PreparedCalvinVidEoMTClip:
    """One hash-bound RGB clip and its class-agnostic VidEoMT target."""

    frames: PreparedVidEoMTFrames
    target: dict[str, torch.Tensor]
    identity_keys: tuple[str, ...]
    camera_name: str

    def __post_init__(self) -> None:
        labels = self.target.get("labels")
        ids = self.target.get("ids")
        masks = self.target.get("masks")
        valid_pixels = self.target.get("valid_pixels")
        object_count = len(self.identity_keys)
        time = self.frames.model_input.shape[0]
        if set(self.target) not in (
            {"labels", "ids", "masks"},
            {"labels", "ids", "masks", "valid_pixels"},
        ):
            raise ContractError("CALVIN VidEoMT target fields drifted")
        if not isinstance(labels, torch.Tensor) or labels.shape != (object_count,):
            raise ContractError("CALVIN VidEoMT labels must have shape [objects]")
        if labels.dtype != torch.long or labels.any():
            raise ContractError("CALVIN class-agnostic labels must all be zero")
        if not isinstance(ids, torch.Tensor) or ids.shape != (object_count, time):
            raise ContractError("CALVIN VidEoMT ids must have shape [objects, time]")
        if ids.dtype != torch.long:
            raise ContractError("CALVIN VidEoMT ids must be int64")
        if not isinstance(masks, torch.Tensor) or masks.shape != (
            object_count,
            time,
            *self.frames.padded_size,
        ):
            raise ContractError("CALVIN VidEoMT masks disagree with padded RGB geometry")
        if masks.dtype != torch.float32 or not torch.isfinite(masks).all():
            raise ContractError("CALVIN VidEoMT masks must be finite float32")
        if ((masks != 0) & (masks != 1)).any():
            raise ContractError("CALVIN VidEoMT masks must remain binary")
        if valid_pixels is not None and (
            not isinstance(valid_pixels, torch.Tensor)
            or valid_pixels.dtype != torch.bool
            or valid_pixels.shape != (time, *self.frames.padded_size)
            or not valid_pixels.flatten(1).any(dim=1).all()
        ):
            raise ContractError(
                "CALVIN VidEoMT valid_pixels must contain measured pixels in every frame"
            )
        if any(not key for key in self.identity_keys):
            raise ContractError("CALVIN VidEoMT identities must be nonempty strings")


def _camera_for(
    frame: CalvinPhysicalSupervisionFrame,
    camera_name: str,
) -> CalvinVisibleOwnerRaster:
    matches = tuple(camera for camera in frame.cameras if camera.camera_name == camera_name)
    if len(matches) != 1:
        raise ContractError(f"CALVIN frame does not contain exactly one {camera_name!r} camera")
    return matches[0]


def _camera_source_field(camera_name: str) -> str:
    matches = tuple(
        str(spec["source_rgb_field"])
        for spec in CALVIN_CAMERA_SPECS
        if spec["camera_name"] == camera_name
    )
    if len(matches) != 1:
        raise ContractError(f"unknown CALVIN physical camera {camera_name!r}")
    return matches[0]


def prepare_calvin_videomt_clip(
    frames_rgb: Sequence[np.ndarray],
    supervision: Sequence[CalvinPhysicalSupervisionFrame],
    *,
    camera_name: str = "static",
    minimum_visible_pixels: int = 1,
    short_edge: int = VIDEOMT_TEST_SHORT_EDGE,
    max_size: int = VIDEOMT_TEST_MAX_SIZE,
    size_divisibility: int = VIDEOMT_SIZE_DIVISIBILITY,
) -> PreparedCalvinVidEoMTClip:
    """Bind full-pixel CALVIN owner masks to the released deterministic geometry.

    Partially measured owner rasters retain an explicit ``valid_pixels`` field.
    The measured-pixel criterion restricts matching and mask losses to that
    field, so unknown depth-inconsistent pixels never become background labels.
    """

    if not frames_rgb or len(frames_rgb) != len(supervision):
        raise ContractError("CALVIN RGB and supervision clips must have equal positive length")
    if (
        isinstance(minimum_visible_pixels, bool)
        or not isinstance(minimum_visible_pixels, int)
        or minimum_visible_pixels <= 0
    ):
        raise ValueError("minimum_visible_pixels must be a positive integer")

    prepared = prepare_rgb_frames(
        frames_rgb,
        short_edge=short_edge,
        max_size=max_size,
        size_divisibility=size_divisibility,
    )
    source_field = _camera_source_field(camera_name)
    cameras: list[CalvinVisibleOwnerRaster] = []
    identity_order: list[str] = []
    seen: set[str] = set()

    for time_index, (rgb, frame) in enumerate(zip(frames_rgb, supervision, strict=True)):
        if not isinstance(frame, CalvinPhysicalSupervisionFrame):
            raise TypeError("CALVIN VidEoMT supervision has an invalid frame type")
        camera = _camera_for(frame, camera_name)
        array = np.asarray(rgb)
        if camera.owner_index.shape != array.shape[:2]:
            raise ContractError("CALVIN RGB and owner raster geometry disagree")
        if source_array_sha256(source_field, array) != camera.source_rgb_sha256:
            raise ContractError(f"CALVIN RGB/source supervision mismatch at time {time_index}")
        maximum_owner = int(camera.owner_index.max(initial=0))
        if maximum_owner > len(frame.identity_keys):
            raise ContractError("CALVIN owner raster references an unknown identity")
        if len(set(frame.identity_keys)) != len(frame.identity_keys):
            raise ContractError("CALVIN physical frame contains duplicate identity keys")
        cameras.append(camera)
        for key in frame.identity_keys:
            if key not in seen:
                seen.add(key)
                identity_order.append(key)

    object_masks: list[torch.Tensor] = []
    object_ids: list[torch.Tensor] = []
    visible_keys: list[str] = []
    padded_h, padded_w = prepared.padded_size
    valid_pixels = torch.zeros((len(cameras), padded_h, padded_w), dtype=torch.bool)
    for time_index, camera in enumerate(cameras):
        measured = torch.from_numpy(
            np.array(camera.owner_supervised, copy=True, order="C")
        )[None, None]
        resized_measured = F.interpolate(
            measured.to(torch.float32),
            size=prepared.resized_sizes[time_index],
            mode="nearest",
        )[0, 0].bool()
        resized_h, resized_w = prepared.resized_sizes[time_index]
        valid_pixels[time_index, :resized_h, :resized_w] = resized_measured

    for key in identity_order:
        masks_by_time: list[torch.Tensor] = []
        ids_by_time: list[int] = []
        visible_anywhere = False
        for time_index, (frame, camera) in enumerate(zip(supervision, cameras, strict=True)):
            if key not in frame.identity_keys:
                original = np.zeros(camera.owner_index.shape, dtype=np.bool_)
            else:
                owner = frame.identity_keys.index(key) + 1
                original = (camera.owner_index == owner) & camera.owner_supervised
            visible = int(np.count_nonzero(original)) >= minimum_visible_pixels
            visible_anywhere |= visible
            ids_by_time.append(len(visible_keys) if visible else -1)

            source = torch.from_numpy(np.ascontiguousarray(original)).to(torch.float32)[None, None]
            resized = F.interpolate(
                source,
                size=prepared.resized_sizes[time_index],
                mode="nearest",
            )[0, 0]
            padded = torch.zeros((padded_h, padded_w), dtype=torch.float32)
            resized_h, resized_w = prepared.resized_sizes[time_index]
            padded[:resized_h, :resized_w] = resized
            masks_by_time.append(padded)

        if visible_anywhere:
            visible_keys.append(key)
            canonical_id = len(visible_keys) - 1
            object_masks.append(torch.stack(masks_by_time, dim=0))
            object_ids.append(
                torch.tensor(
                    [canonical_id if value != -1 else -1 for value in ids_by_time],
                    dtype=torch.long,
                )
            )

    if not visible_keys:
        raise ContractError("CALVIN clip has no sufficiently visible supervised object")

    target = {
        "labels": torch.zeros(len(visible_keys), dtype=torch.long),
        "ids": torch.stack(object_ids, dim=0),
        "masks": torch.stack(object_masks, dim=0),
        "valid_pixels": valid_pixels,
    }
    return PreparedCalvinVidEoMTClip(
        frames=prepared,
        target=target,
        identity_keys=tuple(visible_keys),
        camera_name=camera_name,
    )


def prepare_calvin_videomt_training_clip(
    frames_rgb: Sequence[np.ndarray],
    supervision: Sequence[CalvinPhysicalSupervisionFrame],
    *,
    camera_name: str = "static",
    minimum_visible_pixels: int = 1,
    short_edges: tuple[int, ...] = VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES,
    max_size: int = VIDEOMT_YTVIS19_TRAIN_MAX_SIZE,
    size_divisibility: int = VIDEOMT_SIZE_DIVISIBILITY,
) -> PreparedCalvinVidEoMTClip:
    """Apply the complete released YTVIS-2019 online augmentation path.

    RGB/source hashes are checked before augmentation.  The upstream
    ``ResizeShortestEdge`` and ``RandomFlip`` classes are vendored byte-for-byte;
    only the CALVIN array-to-target adapter is local.
    """

    if len(frames_rgb) != VIDEOMT_YTVIS19_CLIP_LENGTH:
        raise ContractError("exact VidEoMT training requires the released five-frame clip")
    if (
        not short_edges
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in short_edges
        )
        or isinstance(max_size, bool)
        or not isinstance(max_size, int)
        or max_size <= 0
    ):
        raise ValueError("VidEoMT training resize values must be positive integers")

    # This identity transform first performs all source-hash, ownership, and
    # visibility checks in the single normative target implementation.
    source = prepare_calvin_videomt_clip(
        frames_rgb,
        supervision,
        camera_name=camera_name,
        minimum_visible_pixels=minimum_visible_pixels,
        short_edge=int(np.asarray(frames_rgb[0]).shape[0]),
        max_size=max(int(value) for value in np.asarray(frames_rgb[0]).shape[:2]),
        size_divisibility=1,
    )
    source_masks = source.target["masks"]
    augmentations = T.AugmentationList(
        [
            RandomApplyClip(
                T.AugmentationList(
                    [
                        ResizeShortestEdge(
                            VIDEOMT_YTVIS19_CROP_PRE_RESIZE_EDGES,
                            VIDEOMT_YTVIS19_CROP_PRE_RESIZE_MAX_SIZE,
                            "choice_by_clip",
                            clip_frame_cnt=VIDEOMT_YTVIS19_CLIP_LENGTH,
                        ),
                        RandomCropClip(
                            VIDEOMT_YTVIS19_CROP_TYPE,
                            VIDEOMT_YTVIS19_CROP_SIZE,
                            clip_length=VIDEOMT_YTVIS19_CLIP_LENGTH,
                        ),
                    ]
                ),
                prob=VIDEOMT_YTVIS19_CROP_PROBABILITY,
                clip_frame_cnt=VIDEOMT_YTVIS19_CLIP_LENGTH,
            ),
            ResizeShortestEdge(
                short_edges,
                max_size,
                "choice_by_clip",
                clip_frame_cnt=VIDEOMT_YTVIS19_CLIP_LENGTH,
            ),
            RandomFlip(
                horizontal=True,
                vertical=False,
                clip_frame_cnt=VIDEOMT_YTVIS19_CLIP_LENGTH,
            ),
        ]
    )

    transformed_rgb: list[np.ndarray] = []
    transformed_masks: list[torch.Tensor] = []
    transformed_validity: list[torch.Tensor] = []
    normalized_frames: list[torch.Tensor] = []
    resized_sizes: list[tuple[int, int]] = []
    original_sizes: list[tuple[int, int]] = []
    for time_index, rgb in enumerate(frames_rgb):
        original = np.asarray(rgb)
        aug_input = T.AugInput(original.copy())
        transforms = augmentations(aug_input)
        transformed = np.ascontiguousarray(aug_input.image).copy()
        if transformed.dtype != np.uint8 or transformed.ndim != 3 or transformed.shape[2] != 3:
            raise ContractError("released VidEoMT augmentation produced invalid RGB")
        frame_masks = []
        for mask in source_masks[:, time_index]:
            transformed_mask = transforms.apply_segmentation(mask.numpy().astype(np.uint8))
            if transformed_mask.shape != transformed.shape[:2]:
                raise ContractError("released VidEoMT RGB/mask augmentation geometry diverged")
            if not np.isin(transformed_mask, (0, 1)).all():
                raise ContractError("released VidEoMT segmentation transform lost binary labels")
            frame_masks.append(
                torch.from_numpy(np.ascontiguousarray(transformed_mask).copy()).float()
            )
        transformed_measured = transforms.apply_segmentation(
            source.target["valid_pixels"][time_index].numpy().astype(np.uint8)
        )
        if transformed_measured.shape != transformed.shape[:2] or not np.isin(
            transformed_measured, (0, 1)
        ).all():
            raise ContractError("released VidEoMT validity augmentation geometry diverged")
        transformed_rgb.append(transformed)
        transformed_masks.append(torch.stack(frame_masks, dim=0))
        transformed_validity.append(
            torch.from_numpy(np.ascontiguousarray(transformed_measured).copy()).bool()
        )
        rgb_chw = torch.from_numpy(transformed.transpose(2, 0, 1)).unsqueeze(0)
        normalized_frames.append(normalize_rgb_255(rgb_chw).squeeze(0))
        original_sizes.append((int(original.shape[0]), int(original.shape[1])))
        resized_sizes.append((int(transformed.shape[0]), int(transformed.shape[1])))

    if len(set(resized_sizes)) != 1:
        raise ContractError("released clip-consistent augmentation changed geometry within a clip")
    image_list = ImageList.from_tensors(
        normalized_frames,
        size_divisibility=size_divisibility,
        pad_value=0.0,
    )
    padded_h, padded_w = (int(value) for value in image_list.tensor.shape[-2:])
    object_count = len(source.identity_keys)
    padded_masks = torch.zeros(
        (object_count, VIDEOMT_YTVIS19_CLIP_LENGTH, padded_h, padded_w),
        dtype=torch.float32,
    )
    padded_validity = torch.zeros(
        (VIDEOMT_YTVIS19_CLIP_LENGTH, padded_h, padded_w),
        dtype=torch.bool,
    )
    for time_index, frame_masks in enumerate(transformed_masks):
        height, width = resized_sizes[time_index]
        padded_masks[:, time_index, :height, :width] = frame_masks
        padded_validity[time_index, :height, :width] = transformed_validity[time_index]

    # Upstream ``filter_empty_instances`` changes an instance ID to -1 when
    # crop/resize removes its complete mask in that frame.  Preserve that exact
    # post-augmentation visibility contract for CALVIN's typed target adapter.
    transformed_ids = source.target["ids"].clone()
    transformed_visible = padded_masks.flatten(2).any(dim=-1)
    transformed_ids[~transformed_visible] = -1
    valid_after_augmentation = transformed_ids.ne(-1).any(dim=-1)
    retained_indices = valid_after_augmentation.nonzero(as_tuple=False).flatten().tolist()
    retained_keys = tuple(source.identity_keys[index] for index in retained_indices)
    retained_ids = transformed_ids[valid_after_augmentation].clone()
    for new_index, old_index in enumerate(retained_indices):
        retained_ids[retained_ids == old_index] = new_index

    prepared = PreparedVidEoMTFrames(
        model_input=image_list.tensor,
        resized_rgb=tuple(transformed_rgb),
        original_sizes=tuple(original_sizes),
        resized_sizes=tuple(resized_sizes),
        padded_size=(padded_h, padded_w),
    )
    return PreparedCalvinVidEoMTClip(
        frames=prepared,
        target={
            "labels": source.target["labels"][valid_after_augmentation].clone(),
            "ids": retained_ids,
            "masks": padded_masks[valid_after_augmentation],
            "valid_pixels": padded_validity,
        },
        identity_keys=retained_keys,
        camera_name=camera_name,
    )
