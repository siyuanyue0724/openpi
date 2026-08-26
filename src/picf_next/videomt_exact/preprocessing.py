"""Released VidEoMT evaluation preprocessing and postprocessing primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data import transforms as T
from detectron2.structures import ImageList

from picf_next.videomt_exact.runtime import normalize_rgb_255

VIDEOMT_TEST_SHORT_EDGE = 480
VIDEOMT_TEST_MAX_SIZE = 1333
VIDEOMT_SIZE_DIVISIBILITY = 32


@dataclass(frozen=True, slots=True)
class PreparedVidEoMTFrames:
    """Frames transformed exactly as the released Detectron2 evaluation path."""

    model_input: torch.Tensor
    resized_rgb: tuple[np.ndarray, ...]
    original_sizes: tuple[tuple[int, int], ...]
    resized_sizes: tuple[tuple[int, int], ...]
    padded_size: tuple[int, int]

    def __post_init__(self) -> None:
        time = len(self.original_sizes)
        if self.model_input.ndim != 4 or self.model_input.shape[:2] != (time, 3):
            raise ValueError("prepared input must have shape [time, 3, padded_h, padded_w]")
        if len(self.resized_rgb) != time or len(self.resized_sizes) != time:
            raise ValueError("frame metadata lengths disagree")
        if tuple(self.model_input.shape[-2:]) != self.padded_size:
            raise ValueError("recorded padding size disagrees with model input")


def prepare_rgb_frames(
    frames_rgb: Sequence[np.ndarray],
    *,
    short_edge: int = VIDEOMT_TEST_SHORT_EDGE,
    max_size: int = VIDEOMT_TEST_MAX_SIZE,
    size_divisibility: int = VIDEOMT_SIZE_DIVISIBILITY,
) -> PreparedVidEoMTFrames:
    """Apply the release's ResizeShortestEdge, normalization, and ImageList pad."""

    if not frames_rgb:
        raise ValueError("at least one RGB frame is required")
    first = np.asarray(frames_rgb[0])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("frames must be uint8 HWC RGB arrays")
    source_shape = first.shape
    transform = T.ResizeShortestEdge(
        short_edge_length=(short_edge, short_edge),
        max_size=max_size,
        sample_style="choice",
    ).get_transform(first)

    resized_arrays: list[np.ndarray] = []
    normalized_tensors: list[torch.Tensor] = []
    original_sizes: list[tuple[int, int]] = []
    resized_sizes: list[tuple[int, int]] = []
    for frame in frames_rgb:
        array = np.asarray(frame)
        if array.shape != source_shape or array.dtype != np.uint8:
            raise ValueError("all temporal frames must share uint8 HWC RGB shape")
        resized = np.ascontiguousarray(transform.apply_image(array)).copy()
        rgb_chw = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0)
        normalized = normalize_rgb_255(rgb_chw).squeeze(0)
        resized_arrays.append(resized)
        normalized_tensors.append(normalized)
        original_sizes.append((int(array.shape[0]), int(array.shape[1])))
        resized_sizes.append((int(resized.shape[0]), int(resized.shape[1])))

    image_list = ImageList.from_tensors(
        normalized_tensors,
        size_divisibility=size_divisibility,
        pad_value=0.0,
    )
    return PreparedVidEoMTFrames(
        model_input=image_list.tensor,
        resized_rgb=tuple(resized_arrays),
        original_sizes=tuple(original_sizes),
        resized_sizes=tuple(resized_sizes),
        padded_size=tuple(int(value) for value in image_list.tensor.shape[-2:]),
    )


def official_track_scores(class_logits: torch.Tensor) -> torch.Tensor:
    """Match upstream post-processing: average frame logits, then softmax."""

    if class_logits.ndim != 3 or class_logits.shape[-1] < 2:
        raise ValueError("class logits must have shape [time, query, classes+1]")
    return class_logits.mean(dim=0).softmax(dim=-1)[:, :-1]


def official_topk_query_classes(
    class_logits: torch.Tensor,
    *,
    topk: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy the released VIS query-class top-k selection."""

    scores = official_track_scores(class_logits)
    count = min(topk, scores.numel())
    selected_scores, flat_indices = scores.flatten(0, 1).topk(count, sorted=False)
    class_count = scores.shape[1]
    query_indices = torch.div(flat_indices, class_count, rounding_mode="floor")
    class_indices = flat_indices.remainder(class_count)
    return selected_scores, query_indices, class_indices


def unique_query_topk(
    class_logits: torch.Tensor,
    *,
    topk: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Presentation-only ranking with at most one class label per object query."""

    scores = official_track_scores(class_logits)
    best_scores, best_classes = scores.max(dim=-1)
    count = min(topk, best_scores.numel())
    selected_scores, query_indices = best_scores.topk(count, sorted=True)
    return selected_scores, query_indices, best_classes[query_indices]


def resize_query_masks_to_original(
    mask_logits: torch.Tensor,
    *,
    padded_size: tuple[int, int],
    resized_size: tuple[int, int],
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Copy upstream two-stage bilinear mask resizing for one video."""

    if mask_logits.ndim != 4:
        raise ValueError("mask logits must have shape [query, time, height, width]")
    masks = F.interpolate(mask_logits, size=padded_size, mode="bilinear", align_corners=False)
    masks = masks[:, :, : resized_size[0], : resized_size[1]]
    return F.interpolate(masks, size=original_size, mode="bilinear", align_corners=False)
