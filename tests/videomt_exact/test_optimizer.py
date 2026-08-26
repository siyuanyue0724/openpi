from __future__ import annotations

import math

import torch
from torch import nn

from picf_next.videomt_exact.optimizer import (
    VIDEOMT_BASE_LR,
    VIDEOMT_LAYERWISE_LR_DECAY,
    VIDEOMT_RELEASED_TOTAL_STEPS,
    build_exact_videomt_optimizer,
    build_exact_videomt_scheduler,
)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(24))
        self.norm = nn.LayerNorm(4)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.q = nn.Embedding(3, 4)
        self.class_head = nn.Linear(4, 2)


def test_optimizer_copies_released_layerwise_decay_and_group_order() -> None:
    model = _Model()
    optimizer, receipt = build_exact_videomt_optimizer(model)
    block_lrs = dict(receipt.block_learning_rates)

    assert receipt.backbone_parameter_group_count == len(tuple(model.encoder.backbone.parameters()))
    assert block_lrs[23] == VIDEOMT_BASE_LR
    assert block_lrs[22] == VIDEOMT_BASE_LR * VIDEOMT_LAYERWISE_LR_DECAY
    assert math.isclose(
        block_lrs[0],
        VIDEOMT_BASE_LR * VIDEOMT_LAYERWISE_LR_DECAY**23,
        rel_tol=1e-12,
    )
    assert all(
        str(group["name"]).startswith("encoder.backbone.")
        for group in optimizer.param_groups[: receipt.backbone_parameter_group_count]
    )


def test_scheduler_defaults_to_released_160k_and_zero_lr_keeps_adam_state() -> None:
    model = _Model()
    optimizer, receipt = build_exact_videomt_optimizer(model)
    scheduler = build_exact_videomt_scheduler(optimizer, receipt)

    assert scheduler.total_steps == VIDEOMT_RELEASED_TOTAL_STEPS
    assert scheduler.last_epoch == 0
    assert all(float(group["lr"]) == 0.0 for group in optimizer.param_groups)
    assert all(parameter.requires_grad for parameter in model.encoder.backbone.parameters())
    assert model.q.weight.requires_grad

    before = model.encoder.backbone.patch_embed.weight.detach().clone()
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    torch.testing.assert_close(model.encoder.backbone.patch_embed.weight, before)
    assert optimizer.state[model.encoder.backbone.patch_embed.weight]["step"] == 1
    scheduler.step()
    assert scheduler.last_epoch == 1
    assert all(
        float(group["lr"]) == 0.0
        for group in optimizer.param_groups[: receipt.backbone_parameter_group_count]
    )
    assert any(
        float(group["lr"]) > 0.0
        for group in optimizer.param_groups[receipt.backbone_parameter_group_count :]
    )
