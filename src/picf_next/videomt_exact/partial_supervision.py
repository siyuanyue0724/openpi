"""Official VidEoMT point losses restricted to measured CALVIN pixels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from scipy.optimize import linear_sum_assignment

from picf_next._vendor.videomt.criterion_videomt import (
    VideoSetCriterion,
    calculate_uncertainty,
)
from picf_next._vendor.videomt.matcher import VideoHungarianMatcher_Consistent


def _valid_pixels(target: Mapping[str, torch.Tensor]) -> torch.Tensor:
    masks = target["masks"]
    value = target.get("valid_pixels")
    if value is None:
        return torch.ones(masks.shape[1:], dtype=torch.bool, device=masks.device)
    if value.dtype != torch.bool or value.shape != masks.shape[1:]:
        raise ValueError("valid_pixels must be bool with the target mask time/spatial axes")
    if not value.any():
        raise ValueError("a VidEoMT target frame cannot have zero measured pixels")
    return value


def masked_batch_sigmoid_ce_cost(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Pairwise BCE cost over a shared measured-pixel domain."""

    if inputs.ndim != 2 or targets.ndim != 2 or validity.shape != (inputs.shape[1],):
        raise ValueError("masked pairwise BCE inputs have incompatible axes")
    weights = validity.to(dtype=inputs.dtype)
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("masked pairwise BCE has no measured sample")
    positive = F.binary_cross_entropy_with_logits(
        inputs,
        torch.ones_like(inputs),
        reduction="none",
    )
    negative = F.binary_cross_entropy_with_logits(
        inputs,
        torch.zeros_like(inputs),
        reduction="none",
    )
    weighted_targets = targets * weights
    return (
        torch.einsum("qp,np->qn", positive, weighted_targets)
        + torch.einsum("qp,np->qn", negative, (1.0 - targets) * weights)
    ) / denominator


def masked_batch_dice_cost(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Pairwise Dice cost over a shared measured-pixel domain."""

    if inputs.ndim != 2 or targets.ndim != 2 or validity.shape != (inputs.shape[1],):
        raise ValueError("masked pairwise Dice inputs have incompatible axes")
    weights = validity.to(dtype=inputs.dtype)
    if weights.sum() <= 0:
        raise ValueError("masked pairwise Dice has no measured sample")
    probabilities = inputs.sigmoid()
    weighted_targets = targets * weights
    numerator = 2.0 * torch.einsum("qp,np->qn", probabilities * weights, targets)
    denominator = (probabilities * weights).sum(-1)[:, None] + weighted_targets.sum(-1)[
        None, :
    ]
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def masked_sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
    num_masks: float,
) -> torch.Tensor:
    """Matched BCE with equal object weight and no unknown-pixel gradient."""

    if inputs.shape != targets.shape or validity.shape != inputs.shape:
        raise ValueError("masked BCE inputs, targets, and validity must have equal axes")
    weights = validity.to(dtype=inputs.dtype)
    denominator = weights.sum(dim=1)
    if (denominator <= 0).any():
        raise ValueError("masked BCE contains a target with no measured sample")
    values = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return ((values * weights).sum(dim=1) / denominator).sum() / num_masks


def masked_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
    num_masks: float,
) -> torch.Tensor:
    """Matched Dice with all sums restricted to measured pixels."""

    if inputs.shape != targets.shape or validity.shape != inputs.shape:
        raise ValueError("masked Dice inputs, targets, and validity must have equal axes")
    weights = validity.to(dtype=inputs.dtype)
    if (weights.sum(dim=1) <= 0).any():
        raise ValueError("masked Dice contains a target with no measured sample")
    probabilities = inputs.sigmoid()
    numerator = 2.0 * (probabilities * targets * weights).sum(dim=1)
    denominator = (probabilities * weights).sum(dim=1) + (targets * weights).sum(dim=1)
    return (1.0 - (numerator + 1.0) / (denominator + 1.0)).sum() / num_masks


class MeasuredPixelVideoHungarianMatcherConsistent(VideoHungarianMatcher_Consistent):
    """Released online matcher with costs integrated only where CALVIN is measured."""

    @torch.no_grad()
    def memory_efficient_forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        batch_frames, num_queries = outputs["pred_logits"].shape[:2]
        if batch_frames % self.frames:
            raise ValueError("online matcher batch is not divisible by its frame count")
        indices: list[list[list[int]]] = []
        for batch_index in range(batch_frames // self.frames):
            first_appearance: dict[int, int] = {}
            for frame in range(self.frames):
                overall = batch_index * self.frames + frame
                instance_ids = targets[overall]["ids"].squeeze(1)
                for identity in torch.nonzero(instance_ids != -1).flatten().tolist():
                    first_appearance.setdefault(int(identity), frame)

            by_frame: dict[int, list[int]] = {}
            for identity, frame in first_appearance.items():
                by_frame.setdefault(frame, []).append(identity)
            used_queries: list[int] = []
            matched: list[list[int]] = [[], []]
            for frame in sorted(by_frame):
                overall = batch_index * self.frames + frame
                used_targets = by_frame[frame]
                logits = outputs["pred_logits"][overall]
                probabilities = logits.softmax(-1)
                target_classes = targets[overall]["labels"][used_targets].to(torch.int64)
                if ((target_classes < 0) | (target_classes >= probabilities.shape[-1])).any():
                    raise ValueError("class-agnostic target class lies outside donor logits")
                class_cost = -probabilities[:, target_classes]

                predicted_masks = outputs["pred_masks"][overall]
                target_masks = targets[overall]["masks"][used_targets].to(predicted_masks)
                valid_pixels = _valid_pixels(targets[overall]).to(predicted_masks.device)
                point_coordinates = torch.rand(
                    1,
                    self.num_points,
                    2,
                    device=predicted_masks.device,
                )
                sampled_targets = point_sample(
                    target_masks,
                    point_coordinates.repeat(target_masks.shape[0], 1, 1).to(target_masks),
                    align_corners=False,
                ).flatten(1)
                sampled_predictions = point_sample(
                    predicted_masks,
                    point_coordinates.repeat(predicted_masks.shape[0], 1, 1).to(
                        predicted_masks
                    ),
                    align_corners=False,
                ).flatten(1)
                sampled_validity = point_sample(
                    valid_pixels[None].to(torch.float32),
                    point_coordinates,
                    align_corners=False,
                ).flatten()
                with torch.amp.autocast("cuda", enabled=False):
                    sampled_predictions = sampled_predictions.float()
                    sampled_targets = sampled_targets.float()
                    sampled_validity = sampled_validity.float()
                    mask_cost = masked_batch_sigmoid_ce_cost(
                        sampled_predictions,
                        sampled_targets,
                        sampled_validity,
                    )
                    dice_cost = masked_batch_dice_cost(
                        sampled_predictions,
                        sampled_targets,
                        sampled_validity,
                    )
                cost = (
                    self.cost_mask * mask_cost
                    + self.cost_class * class_cost
                    + self.cost_dice * dice_cost
                ).reshape(num_queries, -1)
                cost = cost.cpu()
                if used_queries:
                    cost[used_queries, :] = 1.0e6
                query_indices, target_indices = linear_sum_assignment(cost)
                used_queries.extend(int(value) for value in query_indices)
                selected_target_indices = np.asarray(used_targets)[target_indices]
                matched[0].extend(int(value) for value in query_indices)
                matched[1].extend(int(value) for value in selected_target_indices)
            indices.extend([matched] * self.frames)
        return [
            (
                torch.as_tensor(query, dtype=torch.int64),
                torch.as_tensor(target, dtype=torch.int64),
            )
            for query, target in indices
        ]


class MeasuredPixelVideoSetCriterion(VideoSetCriterion):
    """Released criterion whose mask terms ignore unmeasured CALVIN pixels."""

    def loss_masks(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
        indices: Sequence[tuple[torch.Tensor, torch.Tensor]],
        num_masks: float,
    ) -> dict[str, torch.Tensor]:
        source_indices = self._get_src_permutation_idx(indices)
        source_masks = outputs["pred_masks"][source_indices]
        target_masks = torch.cat(
            [target["masks"][selected] for target, (_, selected) in zip(targets, indices)]
        ).to(source_masks)
        target_validity = torch.cat(
            [
                _valid_pixels(target)
                .expand(target["masks"].shape[0], -1, -1)[selected]
                .to(source_masks.device)
                for target, (_, selected) in zip(targets, indices)
            ]
        )
        source_masks = source_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]
        target_validity = target_validity[:, None]

        with torch.no_grad():
            point_coordinates = get_uncertain_point_coords_with_randomness(
                source_masks.float(),
                calculate_uncertainty,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks,
                point_coordinates.to(target_masks),
                align_corners=False,
            ).squeeze(1)
            point_validity = point_sample(
                target_validity.float(),
                point_coordinates,
                align_corners=False,
            ).squeeze(1)
        point_logits = point_sample(
            source_masks,
            point_coordinates.to(source_masks),
            align_corners=False,
        ).squeeze(1)
        return {
            "loss_mask": masked_sigmoid_ce_loss(
                point_logits,
                point_labels,
                point_validity,
                num_masks,
            ),
            "loss_dice": masked_dice_loss(
                point_logits,
                point_labels,
                point_validity,
                num_masks,
            ),
        }
