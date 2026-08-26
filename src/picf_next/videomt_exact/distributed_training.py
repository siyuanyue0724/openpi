"""Distributed accumulation contracts for complete VidEoMT training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

VIDEOMT_CLASS_LOSS_PREFIX = "loss_ce"
VIDEOMT_MASK_LOSS_PREFIXES = ("loss_mask", "loss_dice")


@dataclass(frozen=True, slots=True)
class VidEoMTEffectiveBatchReceipt:
    world_size: int
    accumulation_steps: int
    local_clips_per_microstep: int
    effective_batch_clips: int
    global_target_counts: tuple[int, ...]
    classification_scales: tuple[float, ...]
    mask_scales: tuple[float, ...]

    def __post_init__(self) -> None:
        integers = (
            self.world_size,
            self.accumulation_steps,
            self.local_clips_per_microstep,
            self.effective_batch_clips,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integers):
            raise ValueError("VidEoMT effective-batch dimensions must be positive integers")
        if self.effective_batch_clips != (
            self.world_size * self.accumulation_steps * self.local_clips_per_microstep
        ):
            raise ValueError("VidEoMT effective-batch product is inconsistent")
        if (
            len(self.global_target_counts) != self.accumulation_steps
            or len(self.classification_scales) != self.accumulation_steps
            or len(self.mask_scales) != self.accumulation_steps
            or any(value < 0 for value in self.global_target_counts)
            or sum(self.global_target_counts) <= 0
        ):
            raise ValueError("VidEoMT effective-batch target counts are invalid")
        tolerance = 1e-9
        if abs(sum(self.classification_scales) - 1.0) > tolerance:
            raise ValueError("VidEoMT classification scales do not sum to one")
        if abs(sum(self.mask_scales) - 1.0) > tolerance:
            raise ValueError("VidEoMT mask scales do not sum to one")


def make_effective_batch_receipt(
    global_target_counts: Sequence[int],
    *,
    world_size: int,
    local_clips_per_microstep: int = 1,
) -> VidEoMTEffectiveBatchReceipt:
    counts = tuple(int(value) for value in global_target_counts)
    if not counts or any(value < 0 for value in counts) or sum(counts) <= 0:
        raise ValueError("global target counts must be non-negative and not all zero")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or isinstance(local_clips_per_microstep, bool)
        or not isinstance(local_clips_per_microstep, int)
        or local_clips_per_microstep <= 0
    ):
        raise ValueError("effective-batch dimensions must be positive integers")
    accumulation_steps = len(counts)
    total_targets = sum(counts)
    return VidEoMTEffectiveBatchReceipt(
        world_size=world_size,
        accumulation_steps=accumulation_steps,
        local_clips_per_microstep=local_clips_per_microstep,
        effective_batch_clips=(world_size * accumulation_steps * local_clips_per_microstep),
        global_target_counts=counts,
        classification_scales=tuple(1.0 / accumulation_steps for _ in counts),
        mask_scales=tuple(value / total_targets for value in counts),
    )


def scale_videomt_microstep_losses(
    weighted_losses: Mapping[str, torch.Tensor],
    receipt: VidEoMTEffectiveBatchReceipt,
    *,
    microstep: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scale CE and mask terms exactly like one effective batch."""

    if not weighted_losses:
        raise ValueError("VidEoMT weighted losses cannot be empty")
    if (
        isinstance(microstep, bool)
        or not isinstance(microstep, int)
        or not 0 <= microstep < receipt.accumulation_steps
    ):
        raise IndexError("VidEoMT microstep is out of range")
    scaled: dict[str, torch.Tensor] = {}
    for name, value in weighted_losses.items():
        if not isinstance(name, str) or not name:
            raise TypeError("VidEoMT loss names must be nonempty strings")
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TypeError("VidEoMT losses must be scalar tensors")
        if name == VIDEOMT_CLASS_LOSS_PREFIX or name.startswith(
            VIDEOMT_CLASS_LOSS_PREFIX + "_"
        ):
            scale = receipt.classification_scales[microstep]
        elif any(name == prefix or name.startswith(prefix + "_") for prefix in VIDEOMT_MASK_LOSS_PREFIXES):
            scale = receipt.mask_scales[microstep]
        else:
            raise ValueError(f"unrecognized released VidEoMT loss {name!r}")
        scaled[name] = value * scale
    total = sum(scaled.values())
    if not torch.isfinite(total):
        raise RuntimeError("scaled VidEoMT effective-batch loss is non-finite")
    return total, scaled
