"""Parameter-free class-agnostic adaptation of the released VidEoMT objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from picf_next._vendor.videomt.criterion_videomt import VideoSetCriterion
from picf_next._vendor.videomt.matcher import VideoHungarianMatcher
from picf_next.videomt_exact.runtime import VIDEOMT_DINOV3_L_CLASSES, ExactVidEoMTOutput
from picf_next.videomt_exact.partial_supervision import (
    MeasuredPixelVideoHungarianMatcherConsistent,
    MeasuredPixelVideoSetCriterion,
)
from picf_next.videomt_exact.training import (
    VIDEOMT_CLASS_WEIGHT,
    VIDEOMT_DICE_WEIGHT,
    VIDEOMT_IMPORTANCE_SAMPLE_RATIO,
    VIDEOMT_MASK_WEIGHT,
    VIDEOMT_NO_OBJECT_WEIGHT,
    VIDEOMT_OVERSAMPLE_RATIO,
    VIDEOMT_TRAIN_NUM_POINTS,
    flatten_video_outputs_for_released_criterion,
    flatten_video_targets_for_released_criterion,
    released_online_weight_dict,
    released_weight_dict,
)

VIDEOMT_ONLINE_CONSISTENT_MATCHER = "online-consistent"
VIDEOMT_FRAME_LOCAL_MATCHER_ABLATION = "frame-local-ablation"
VIDEOMT_MATCHER_IDENTITIES = (
    VIDEOMT_ONLINE_CONSISTENT_MATCHER,
    VIDEOMT_FRAME_LOCAL_MATCHER_ABLATION,
)


def marginalize_videomt_taxonomy(class_logits: torch.Tensor) -> torch.Tensor:
    """Collapse 40 foreground classes without changing foreground probability mass."""

    if (
        class_logits.ndim < 1
        or not class_logits.is_floating_point()
        or class_logits.shape[-1] != VIDEOMT_DINOV3_L_CLASSES + 1
    ):
        raise ValueError("released VidEoMT taxonomy logits must end in 41 floating channels")
    if not torch.isfinite(class_logits).all():
        raise ValueError("released VidEoMT taxonomy logits contain NaN or infinity")
    foreground = torch.logsumexp(
        class_logits[..., :VIDEOMT_DINOV3_L_CLASSES],
        dim=-1,
        keepdim=True,
    )
    no_object = class_logits[..., VIDEOMT_DINOV3_L_CLASSES:]
    return torch.cat((foreground, no_object), dim=-1)


def flatten_class_agnostic_outputs(
    output: ExactVidEoMTOutput,
) -> dict[str, object]:
    """Preserve official video-axis transforms and marginalize only taxonomy."""

    released = flatten_video_outputs_for_released_criterion(output)
    result: dict[str, object] = {
        "pred_logits": marginalize_videomt_taxonomy(released["pred_logits"]),
        "pred_masks": released["pred_masks"],
    }
    auxiliary = released["aux_outputs"]
    if not isinstance(auxiliary, list):
        raise RuntimeError("released VidEoMT auxiliary outputs changed container type")
    result["aux_outputs"] = [
        {
            "pred_logits": marginalize_videomt_taxonomy(value["pred_logits"]),
            "pred_masks": value["pred_masks"],
        }
        for value in auxiliary
    ]
    return result


def flatten_class_agnostic_targets(
    clip_targets: Sequence[Mapping[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Keep masks and identities while removing dataset-specific category names."""

    for target in clip_targets:
        ids = target["ids"]
        expected = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1).expand_as(ids)
        if ((ids != -1) & (ids != expected)).any():
            raise ValueError("consistent VidEoMT identities must equal their target row or -1")
    released = flatten_video_targets_for_released_criterion(clip_targets)
    result: list[dict[str, torch.Tensor]] = []
    for target in released:
        converted = {
            "labels": torch.zeros_like(target["labels"], dtype=torch.long),
            "ids": target["ids"],
            "masks": target["masks"],
        }
        valid_pixels = target.get("valid_pixels")
        if valid_pixels is not None:
            converted["valid_pixels"] = valid_pixels
        result.append(converted)
    return result


def build_class_agnostic_criterion(
    *,
    matcher_identity: str = VIDEOMT_ONLINE_CONSISTENT_MATCHER,
    num_frames: int = 5,
) -> VideoSetCriterion:
    """Adapt taxonomy only; default to the selected online donor's matcher."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("class-agnostic VidEoMT num_frames must be positive")
    if matcher_identity not in VIDEOMT_MATCHER_IDENTITIES:
        raise ValueError("class-agnostic VidEoMT matcher identity is unsupported")
    if matcher_identity == VIDEOMT_ONLINE_CONSISTENT_MATCHER:
        matcher = MeasuredPixelVideoHungarianMatcherConsistent(
            cost_class=VIDEOMT_CLASS_WEIGHT,
            cost_mask=VIDEOMT_MASK_WEIGHT,
            cost_dice=VIDEOMT_DICE_WEIGHT,
            num_points=VIDEOMT_TRAIN_NUM_POINTS,
            frames=num_frames,
        )
    else:
        matcher = VideoHungarianMatcher(
            cost_class=VIDEOMT_CLASS_WEIGHT,
            cost_mask=VIDEOMT_MASK_WEIGHT,
            cost_dice=VIDEOMT_DICE_WEIGHT,
            num_points=VIDEOMT_TRAIN_NUM_POINTS,
        )
    weights = (
        released_online_weight_dict()
        if matcher_identity == VIDEOMT_ONLINE_CONSISTENT_MATCHER
        else released_weight_dict()
    )
    criterion_type = (
        MeasuredPixelVideoSetCriterion
        if matcher_identity == VIDEOMT_ONLINE_CONSISTENT_MATCHER
        else VideoSetCriterion
    )
    return criterion_type(
        1,
        matcher=matcher,
        weight_dict=weights,
        eos_coef=VIDEOMT_NO_OBJECT_WEIGHT,
        losses=("labels", "masks"),
        num_points=VIDEOMT_TRAIN_NUM_POINTS,
        oversample_ratio=VIDEOMT_OVERSAMPLE_RATIO,
        importance_sample_ratio=VIDEOMT_IMPORTANCE_SAMPLE_RATIO,
    )
