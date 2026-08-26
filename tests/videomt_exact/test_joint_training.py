from __future__ import annotations

import torch

from picf_next.videomt_exact.joint_training import (
    COMPLETE_VIDEOMT_RAW_LOSS_NAMES,
    COMPLETE_VIDEOMT_WEIGHTED_LOSS_NAMES,
    CompleteCalvinVidEoMTObjective,
    complete_picf_joint_total,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput


def _complete_output() -> ExactVidEoMTOutput:
    return ExactVidEoMTOutput(
        class_logits=torch.randn(1, 5, 200, 41, requires_grad=True),
        mask_logits=torch.randn(1, 200, 5, 8, 8, requires_grad=True),
        query_embeddings=torch.randn(1, 5, 200, 1024, requires_grad=True),
        propagated_queries=torch.randn(1, 200, 1024, requires_grad=True),
        auxiliary_outputs=tuple(
            {
                "pred_logits": torch.randn(1, 5, 200, 41, requires_grad=True),
                "pred_masks": torch.randn(1, 200, 5, 8, 8, requires_grad=True),
            }
            for _ in range(4)
        ),
    )


def _target() -> dict[str, torch.Tensor]:
    masks = torch.zeros(2, 5, 16, 16)
    masks[0, :, 1:7, 2:8] = 1
    masks[1, :, 9:15, 8:14] = 1
    valid_pixels = torch.ones(5, 16, 16, dtype=torch.bool)
    valid_pixels[:, :2, :2] = False
    return {
        "labels": torch.zeros(2, dtype=torch.long),
        "ids": torch.tensor([[0] * 5, [1] * 5]),
        "masks": masks,
        "valid_pixels": valid_pixels,
    }


def test_complete_joint_source_retains_every_released_read_and_measured_domain() -> None:
    torch.manual_seed(207)
    output = _complete_output()
    target = _target()
    objective = CompleteCalvinVidEoMTObjective()

    flattened = objective.criterion.matcher.frames
    assert flattened == 5
    result = objective(output, [target])

    assert set(result.raw_losses) == COMPLETE_VIDEOMT_RAW_LOSS_NAMES
    assert set(result.weighted_losses) == COMPLETE_VIDEOMT_WEIGHTED_LOSS_NAMES
    assert len(result.raw_losses) == 15
    assert len(result.weighted_losses) == 12
    assert result.target_count == 2
    result.total.backward()
    assert output.class_logits.grad is not None
    assert output.mask_logits.grad is not None
    for auxiliary in output.auxiliary_outputs[:3]:
        assert auxiliary["pred_logits"].grad is not None
        assert auxiliary["pred_masks"].grad is not None


def test_complete_joint_total_preserves_both_gradient_paths() -> None:
    torch.manual_seed(208)
    output = _complete_output()
    source = CompleteCalvinVidEoMTObjective()(output, [_target()])
    host_parameter = torch.tensor(2.0, requires_grad=True)
    joint = complete_picf_joint_total(host_total=host_parameter.square(), source=source)
    joint.backward()

    torch.testing.assert_close(host_parameter.grad, torch.tensor(4.0))
    assert output.class_logits.grad is not None
    assert output.mask_logits.grad is not None
