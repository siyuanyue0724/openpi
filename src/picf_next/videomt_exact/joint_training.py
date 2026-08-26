"""Complete CALVIN VidEoMT source objective for joint PICF training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from picf_next.videomt_exact.class_agnostic import (
    VIDEOMT_ONLINE_CONSISTENT_MATCHER,
    build_class_agnostic_criterion,
    flatten_class_agnostic_outputs,
    flatten_class_agnostic_targets,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput
from picf_next.videomt_exact.training import (
    VIDEOMT_DEEP_SUPERVISION_LAYERS,
    apply_released_loss_weights,
    released_weight_dict,
)


def _expected_raw_loss_names() -> frozenset[str]:
    base = ("loss_ce", "loss_mask", "loss_dice")
    return frozenset(
        (*base, *(f"{name}_{layer}" for layer in range(VIDEOMT_DEEP_SUPERVISION_LAYERS) for name in base))
    )


COMPLETE_VIDEOMT_RAW_LOSS_NAMES = _expected_raw_loss_names()
COMPLETE_VIDEOMT_WEIGHTED_LOSS_NAMES = frozenset(released_weight_dict())


@dataclass(frozen=True, slots=True)
class CompleteVidEoMTSourceObjective:
    """Every final and auxiliary source loss, with released weighting intact."""

    total: torch.Tensor
    raw_losses: Mapping[str, torch.Tensor]
    weighted_losses: Mapping[str, torch.Tensor]
    target_count: int

    def __post_init__(self) -> None:
        if self.total.ndim != 0 or not self.total.is_floating_point():
            raise ValueError("complete VidEoMT objective total must be one floating scalar")
        if not torch.isfinite(self.total):
            raise ValueError("complete VidEoMT objective total is not finite")
        if set(self.raw_losses) != COMPLETE_VIDEOMT_RAW_LOSS_NAMES:
            raise ValueError("complete VidEoMT raw loss inventory drifted")
        if set(self.weighted_losses) != COMPLETE_VIDEOMT_WEIGHTED_LOSS_NAMES:
            raise ValueError("complete VidEoMT weighted loss inventory drifted")
        values = (*self.raw_losses.values(), *self.weighted_losses.values())
        if any(
            value.ndim != 0
            or not value.is_floating_point()
            or not torch.isfinite(value)
            or value.device != self.total.device
            for value in values
        ):
            raise ValueError("complete VidEoMT loss terms must be finite colocated scalars")
        if isinstance(self.target_count, bool) or self.target_count < 0:
            raise ValueError("complete VidEoMT target count must be non-negative")


class CompleteCalvinVidEoMTObjective(nn.Module):
    """Execute the selected complete source criterion without host-side replacement.

    CALVIN changes only the category coordinate system and measured-pixel domain.
    The 200 queries, final prediction, four auxiliary predictions, online-consistent
    matcher, point losses, and released loss weights remain present.
    """

    def __init__(self, *, num_frames: int = 5) -> None:
        super().__init__()
        if num_frames != 5:
            raise ValueError("complete CALVIN VidEoMT joint training requires five frames")
        self.num_frames = num_frames
        self.criterion = build_class_agnostic_criterion(
            matcher_identity=VIDEOMT_ONLINE_CONSISTENT_MATCHER,
            num_frames=num_frames,
        )

    def forward(
        self,
        output: ExactVidEoMTOutput,
        clip_targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> CompleteVidEoMTSourceObjective:
        if not isinstance(output, ExactVidEoMTOutput):
            raise TypeError("complete source objective requires an ExactVidEoMTOutput")
        if output.class_logits.shape[1] != self.num_frames:
            raise ValueError("complete source output does not contain five frames")
        if len(output.auxiliary_outputs) != VIDEOMT_DEEP_SUPERVISION_LAYERS:
            raise ValueError("complete source output must retain all four auxiliary reads")
        flat_outputs = flatten_class_agnostic_outputs(output)
        flat_targets = flatten_class_agnostic_targets(clip_targets)
        raw_losses = self.criterion(flat_outputs, flat_targets)
        weighted_losses = apply_released_loss_weights(raw_losses, self.criterion)
        if set(raw_losses) != COMPLETE_VIDEOMT_RAW_LOSS_NAMES:
            raise RuntimeError("complete source criterion omitted or added a loss term")
        if set(weighted_losses) != COMPLETE_VIDEOMT_WEIGHTED_LOSS_NAMES:
            raise RuntimeError("released source weighting omitted or added a loss term")
        total = torch.stack(tuple(weighted_losses.values())).sum()
        return CompleteVidEoMTSourceObjective(
            total=total,
            raw_losses=dict(raw_losses),
            weighted_losses=dict(weighted_losses),
            target_count=sum(int(target["labels"].numel()) for target in clip_targets),
        )


def complete_picf_joint_total(
    *,
    host_total: torch.Tensor,
    source: CompleteVidEoMTSourceObjective,
) -> torch.Tensor:
    """Sum the unchanged host objective and unchanged complete source objective."""

    if (
        host_total.ndim != 0
        or not host_total.is_floating_point()
        or not torch.isfinite(host_total)
        or host_total.device != source.total.device
    ):
        raise ValueError("host and complete source objectives must be finite colocated scalars")
    return host_total + source.total
