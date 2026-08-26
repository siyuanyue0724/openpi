from __future__ import annotations

import pytest
import torch

from picf_next._vendor.videomt.criterion_videomt import dice_loss, sigmoid_ce_loss
from picf_next._vendor.videomt.matcher import batch_dice_loss, batch_sigmoid_ce_loss
from picf_next.videomt_exact.partial_supervision import (
    MeasuredPixelVideoHungarianMatcherConsistent,
    masked_batch_dice_cost,
    masked_batch_sigmoid_ce_cost,
    masked_dice_loss,
    masked_sigmoid_ce_loss,
)


def test_all_measured_costs_are_identical_to_released_costs() -> None:
    generator = torch.Generator().manual_seed(201)
    predictions = torch.randn(7, 29, generator=generator)
    targets = (torch.rand(4, 29, generator=generator) > 0.6).float()
    validity = torch.ones(29)

    torch.testing.assert_close(
        masked_batch_sigmoid_ce_cost(predictions, targets, validity),
        batch_sigmoid_ce_loss(predictions, targets),
    )
    torch.testing.assert_close(
        masked_batch_dice_cost(predictions, targets, validity),
        batch_dice_loss(predictions, targets),
    )


def test_all_measured_losses_are_identical_to_released_losses() -> None:
    generator = torch.Generator().manual_seed(202)
    predictions = torch.randn(5, 31, generator=generator)
    targets = (torch.rand(5, 31, generator=generator) > 0.5).float()
    validity = torch.ones_like(targets)

    torch.testing.assert_close(
        masked_sigmoid_ce_loss(predictions, targets, validity, 5.0),
        sigmoid_ce_loss(predictions, targets, 5.0),
    )
    torch.testing.assert_close(
        masked_dice_loss(predictions, targets, validity, 5.0),
        dice_loss(predictions, targets, 5.0),
    )


def test_unmeasured_points_have_exactly_zero_mask_loss_gradient() -> None:
    predictions = torch.tensor([[0.3, -0.8, 1.2, -0.4]], requires_grad=True)
    targets = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    validity = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    loss = masked_sigmoid_ce_loss(predictions, targets, validity, 1.0)
    loss = loss + masked_dice_loss(predictions, targets, validity, 1.0)
    loss.backward()

    assert predictions.grad is not None
    assert predictions.grad[0, 1] == 0
    assert predictions.grad[0, 3] == 0
    assert predictions.grad[0, 0] != 0
    assert predictions.grad[0, 2] != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 contract")
def test_matcher_casts_point_coordinates_to_bfloat16_mask_dtype() -> None:
    device = torch.device("cuda")
    matcher = MeasuredPixelVideoHungarianMatcherConsistent(
        cost_class=2.0,
        cost_mask=5.0,
        cost_dice=5.0,
        num_points=32,
        frames=1,
    )
    outputs = {
        "pred_logits": torch.randn(1, 3, 2, device=device, dtype=torch.bfloat16),
        "pred_masks": torch.randn(1, 3, 1, 8, 8, device=device, dtype=torch.bfloat16),
    }
    targets = [
        {
            "labels": torch.zeros(1, device=device, dtype=torch.long),
            "ids": torch.zeros(1, 1, device=device, dtype=torch.long),
            "masks": torch.ones(1, 1, 8, 8, device=device, dtype=torch.bfloat16),
            "valid_pixels": torch.ones(1, 8, 8, device=device, dtype=torch.bool),
        }
    ]

    indices = matcher(outputs, targets)

    assert len(indices) == 1
    assert indices[0][0].tolist() in ([0], [1], [2])
    assert indices[0][1].tolist() == [0]
