"""Released VidEoMT optimizer grouping and mathematically equivalent warmup freeze."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW

from picf_next._vendor.videomt.modeling.two_stage_warmup_poly_schedule import (
    TwoStageWarmupPolySchedule,
)

VIDEOMT_BASE_LR = 1.0e-4
VIDEOMT_LAYERWISE_LR_DECAY = 0.6
VIDEOMT_WEIGHT_DECAY = 0.05
VIDEOMT_NON_VIT_WARMUP_STEPS = 500
VIDEOMT_VIT_WARMUP_STEPS = 1000
VIDEOMT_POLY_POWER = 0.9
VIDEOMT_RELEASED_TOTAL_STEPS = 160_000
VIDEOMT_ADAPTATION_BUDGET_STEPS = 30_000
_BACKBONE_PREFIX = "encoder.backbone."


@dataclass(frozen=True, slots=True)
class VidEoMTOptimizerReceipt:
    parameter_group_count: int
    backbone_parameter_group_count: int
    parameter_tensor_count: int
    parameter_numel: int
    minimum_initial_lr: float
    maximum_initial_lr: float
    block_learning_rates: tuple[tuple[int, float], ...]


def build_exact_videomt_optimizer(
    model: nn.Module,
    *,
    base_lr: float = VIDEOMT_BASE_LR,
    layerwise_lr_decay: float = VIDEOMT_LAYERWISE_LR_DECAY,
    weight_decay: float = VIDEOMT_WEIGHT_DECAY,
) -> tuple[AdamW, VidEoMTOptimizerReceipt]:
    """Copy the released reversed-parameter LLRD construction for the bare backbone."""

    if not hasattr(model, "encoder") or not hasattr(model.encoder, "backbone"):
        raise TypeError("exact VidEoMT optimizer requires the released bare model")
    if base_lr <= 0 or not 0 < layerwise_lr_decay <= 1 or weight_decay < 0:
        raise ValueError("VidEoMT optimizer hyperparameters are invalid")
    encoder = model.encoder.backbone
    encoder_parameter_names = {name for name, _parameter in encoder.named_parameters()}
    backbone_blocks = len(encoder.blocks)
    block_index = backbone_blocks
    backbone_groups: list[dict[str, object]] = []
    other_groups: list[dict[str, object]] = []

    for name, parameter in reversed(list(model.named_parameters())):
        if not parameter.requires_grad:
            continue
        learning_rate = base_lr
        suffix = name.removeprefix(_BACKBONE_PREFIX)
        if name.startswith(_BACKBONE_PREFIX) and suffix in encoder_parameter_names:
            name_parts = name.split(".")
            is_block = False
            for position, key in enumerate(name_parts):
                if key == "blocks":
                    block_index = int(name_parts[position + 1])
                    is_block = True
            if is_block or block_index == 0:
                learning_rate *= layerwise_lr_decay ** (backbone_blocks - 1 - block_index)
            backbone_groups.append({"params": [parameter], "lr": learning_rate, "name": name})
        else:
            other_groups.append({"params": [parameter], "lr": base_lr, "name": name})
    parameter_groups = backbone_groups + other_groups
    if not parameter_groups:
        raise ValueError("exact VidEoMT optimizer found no trainable parameters")
    optimizer = AdamW(parameter_groups, weight_decay=weight_decay)

    block_lrs: dict[int, float] = {}
    for group in backbone_groups:
        name = str(group["name"])
        parts = name.split(".")
        if "blocks" in parts:
            index = int(parts[parts.index("blocks") + 1])
            block_lrs.setdefault(index, float(group["lr"]))
    receipt = VidEoMTOptimizerReceipt(
        parameter_group_count=len(parameter_groups),
        backbone_parameter_group_count=len(backbone_groups),
        parameter_tensor_count=sum(len(group["params"]) for group in parameter_groups),
        parameter_numel=sum(
            parameter.numel() for group in parameter_groups for parameter in group["params"]
        ),
        minimum_initial_lr=min(float(group["lr"]) for group in parameter_groups),
        maximum_initial_lr=max(float(group["lr"]) for group in parameter_groups),
        block_learning_rates=tuple(sorted(block_lrs.items())),
    )
    return optimizer, receipt


def build_exact_videomt_scheduler(
    optimizer: AdamW,
    receipt: VidEoMTOptimizerReceipt,
    *,
    total_steps: int = VIDEOMT_RELEASED_TOTAL_STEPS,
) -> TwoStageWarmupPolySchedule:
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("VidEoMT scheduler total_steps must be positive")
    if len(optimizer.param_groups) != receipt.parameter_group_count:
        raise ValueError("VidEoMT optimizer receipt no longer describes the optimizer")
    return TwoStageWarmupPolySchedule(
        optimizer,
        num_backbone_params=receipt.backbone_parameter_group_count,
        warmup_steps=(VIDEOMT_NON_VIT_WARMUP_STEPS, VIDEOMT_VIT_WARMUP_STEPS),
        total_steps=total_steps,
        poly_power=VIDEOMT_POLY_POWER,
    )


def optimizer_group_learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, ...]:
    return tuple(float(group["lr"]) for group in optimizer.param_groups)
