"""Exact released VidEoMT matching, losses, and video-axis transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from picf_next._vendor.videomt.criterion_videomt import VideoSetCriterion
from picf_next._vendor.videomt.matcher import (
    VideoHungarianMatcher,
    VideoHungarianMatcher_Consistent,
)
from picf_next.videomt_exact.runtime import VIDEOMT_DINOV3_L_CLASSES, ExactVidEoMTOutput

VIDEOMT_CLASS_WEIGHT = 2.0
VIDEOMT_MASK_WEIGHT = 5.0
VIDEOMT_DICE_WEIGHT = 5.0
VIDEOMT_NO_OBJECT_WEIGHT = 0.1
VIDEOMT_TRAIN_NUM_POINTS = 12_544
VIDEOMT_OVERSAMPLE_RATIO = 3.0
VIDEOMT_IMPORTANCE_SAMPLE_RATIO = 0.75
VIDEOMT_DEEP_SUPERVISION_LAYERS = 4
VIDEOMT_ONLINE_REID_WEIGHT = 2.0


def released_weight_dict() -> dict[str, float]:
    """Reproduce the official base ``videomt`` loss weight construction."""

    base = {
        "loss_ce": VIDEOMT_CLASS_WEIGHT,
        "loss_mask": VIDEOMT_MASK_WEIGHT,
        "loss_dice": VIDEOMT_DICE_WEIGHT,
    }
    result = dict(base)
    for layer in range(VIDEOMT_DEEP_SUPERVISION_LAYERS - 1):
        result.update({f"{name}_{layer}": weight for name, weight in base.items()})
    return result


def released_online_weight_dict() -> dict[str, float]:
    """Reproduce ``videomt_online.from_config`` including its declared re-ID keys.

    The frozen upstream online backbone does not emit ``pred_reid_embed`` on
    this path, so the released criterion emits no re-ID value despite declaring
    these two weights. Keeping the keys preserves upstream behavior; inventing
    a local re-ID head would not.
    """

    result = released_weight_dict()
    result.update(
        {
            "loss_reid": VIDEOMT_ONLINE_REID_WEIGHT,
            "loss_reid_aux": VIDEOMT_ONLINE_REID_WEIGHT,
        }
    )
    return result


def build_released_criterion() -> VideoSetCriterion:
    """Instantiate the released base ``videomt`` frame-local criterion."""

    matcher = VideoHungarianMatcher(
        cost_class=VIDEOMT_CLASS_WEIGHT,
        cost_mask=VIDEOMT_MASK_WEIGHT,
        cost_dice=VIDEOMT_DICE_WEIGHT,
        num_points=VIDEOMT_TRAIN_NUM_POINTS,
    )
    return VideoSetCriterion(
        VIDEOMT_DINOV3_L_CLASSES,
        matcher=matcher,
        weight_dict=released_weight_dict(),
        eos_coef=VIDEOMT_NO_OBJECT_WEIGHT,
        losses=("labels", "masks"),
        num_points=VIDEOMT_TRAIN_NUM_POINTS,
        oversample_ratio=VIDEOMT_OVERSAMPLE_RATIO,
        importance_sample_ratio=VIDEOMT_IMPORTANCE_SAMPLE_RATIO,
    )


def build_released_online_criterion(*, num_frames: int = 5) -> VideoSetCriterion:
    """Instantiate the selected DINOv3 ``videomt_online`` criterion exactly."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("released online VidEoMT num_frames must be positive")
    matcher = VideoHungarianMatcher_Consistent(
        cost_class=VIDEOMT_CLASS_WEIGHT,
        cost_mask=VIDEOMT_MASK_WEIGHT,
        cost_dice=VIDEOMT_DICE_WEIGHT,
        num_points=VIDEOMT_TRAIN_NUM_POINTS,
        frames=num_frames,
    )
    return VideoSetCriterion(
        VIDEOMT_DINOV3_L_CLASSES,
        matcher=matcher,
        weight_dict=released_online_weight_dict(),
        eos_coef=VIDEOMT_NO_OBJECT_WEIGHT,
        losses=("labels", "masks"),
        num_points=VIDEOMT_TRAIN_NUM_POINTS,
        oversample_ratio=VIDEOMT_OVERSAMPLE_RATIO,
        importance_sample_ratio=VIDEOMT_IMPORTANCE_SAMPLE_RATIO,
    )


def flatten_video_outputs_for_released_criterion(
    output: ExactVidEoMTOutput,
) -> dict[str, object]:
    """Copy upstream ``frame_decoder_loss_reshape`` for typed runtime outputs."""

    batch, time, queries, classes = output.class_logits.shape

    def flatten_pair(
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected_class = (batch, time, queries, classes)
        if class_logits.shape != expected_class:
            raise ValueError(f"auxiliary class shape {class_logits.shape} != {expected_class}")
        if mask_logits.shape[:3] != (batch, queries, time):
            raise ValueError("auxiliary mask axes disagree with final output")
        flat_classes = class_logits.reshape(batch * time, queries, classes)
        flat_masks = (
            mask_logits.permute(0, 2, 1, 3, 4)
            .reshape(batch * time, queries, mask_logits.shape[-2], mask_logits.shape[-1])
            .unsqueeze(2)
        )
        return {"pred_logits": flat_classes, "pred_masks": flat_masks}

    result: dict[str, object] = flatten_pair(output.class_logits, output.mask_logits)
    result["aux_outputs"] = [
        flatten_pair(auxiliary["pred_logits"], auxiliary["pred_masks"])
        for auxiliary in output.auxiliary_outputs
    ]
    return result


def flatten_video_targets_for_released_criterion(
    clip_targets: Sequence[Mapping[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Copy upstream per-clip to per-frame target expansion."""

    frame_targets: list[dict[str, torch.Tensor]] = []
    for target in clip_targets:
        labels = target["labels"]
        ids = target["ids"]
        masks = target["masks"]
        if labels.ndim != 1 or ids.ndim != 2 or masks.ndim != 4:
            raise ValueError("clip target axes must be labels[N], ids[N,T], masks[N,T,H,W]")
        if ids.shape != masks.shape[:2] or labels.shape[0] != ids.shape[0]:
            raise ValueError("clip target object and time axes disagree")
        for frame in range(ids.shape[1]):
            frame_target = {
                "labels": labels,
                "ids": ids[:, [frame]],
                "masks": masks[:, [frame]],
            }
            valid_pixels = target.get("valid_pixels")
            if valid_pixels is not None:
                if (
                    valid_pixels.dtype != torch.bool
                    or valid_pixels.shape != (ids.shape[1], *masks.shape[-2:])
                ):
                    raise ValueError("clip valid_pixels axes disagree with masks")
                frame_target["valid_pixels"] = valid_pixels[[frame]]
            frame_targets.append(frame_target)
    return frame_targets


def apply_released_loss_weights(
    raw_losses: Mapping[str, torch.Tensor],
    criterion: VideoSetCriterion,
) -> dict[str, torch.Tensor]:
    """Copy the upstream meta-architecture's weighting and unknown-key filter."""

    return {
        name: value * criterion.weight_dict[name]
        for name, value in raw_losses.items()
        if name in criterion.weight_dict
    }
