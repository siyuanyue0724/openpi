from __future__ import annotations

import pytest
import torch

from picf_next.videomt_exact.distributed_training import (
    make_effective_batch_receipt,
    scale_videomt_microstep_losses,
)


def test_effective_batch_eight_uses_distinct_ce_and_mask_scales() -> None:
    receipt = make_effective_batch_receipt((2, 4, 1, 3), world_size=2)

    assert receipt.effective_batch_clips == 8
    assert receipt.classification_scales == (0.25, 0.25, 0.25, 0.25)
    assert receipt.mask_scales == pytest.approx((0.2, 0.4, 0.1, 0.3))

    totals = []
    for microstep in range(4):
        total, scaled = scale_videomt_microstep_losses(
            {
                "loss_ce": torch.tensor(2.0),
                "loss_mask": torch.tensor(5.0),
                "loss_dice_0": torch.tensor(7.0),
            },
            receipt,
            microstep=microstep,
        )
        assert scaled["loss_ce"].item() == pytest.approx(0.5)
        totals.append(total)
    assert float(torch.stack(totals).sum()) == pytest.approx(14.0)


def test_effective_batch_rejects_unknown_upstream_loss() -> None:
    receipt = make_effective_batch_receipt((1, 1, 1, 1), world_size=2)
    with pytest.raises(ValueError, match="unrecognized"):
        scale_videomt_microstep_losses(
            {"loss_local_shortcut": torch.tensor(1.0)},
            receipt,
            microstep=0,
        )
