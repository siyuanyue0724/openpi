from __future__ import annotations

import torch

from picf_next.videomt_exact.runtime import ExactVidEoMTOutput
from picf_next.videomt_exact.training import (
    apply_released_loss_weights,
    build_released_criterion,
    build_released_online_criterion,
    flatten_video_outputs_for_released_criterion,
    flatten_video_targets_for_released_criterion,
    released_online_weight_dict,
    released_weight_dict,
)


def _typed_output(*, auxiliary_count: int = 4) -> ExactVidEoMTOutput:
    classes = torch.randn(1, 2, 200, 41, requires_grad=True)
    masks = torch.randn(1, 200, 2, 8, 8, requires_grad=True)
    auxiliary = tuple(
        {
            "pred_logits": torch.randn(1, 2, 200, 41, requires_grad=True),
            "pred_masks": torch.randn(1, 200, 2, 8, 8, requires_grad=True),
        }
        for _ in range(auxiliary_count)
    )
    return ExactVidEoMTOutput(
        class_logits=classes,
        mask_logits=masks,
        query_embeddings=torch.randn(1, 2, 200, 1024),
        propagated_queries=torch.randn(1, 200, 1024),
        auxiliary_outputs=auxiliary,
    )


def test_video_axis_transforms_match_released_contract() -> None:
    output = _typed_output()
    flat = flatten_video_outputs_for_released_criterion(output)
    assert flat["pred_logits"].shape == (2, 200, 41)
    assert flat["pred_masks"].shape == (2, 200, 1, 8, 8)
    assert len(flat["aux_outputs"]) == 4
    target = {
        "labels": torch.tensor([0, 3]),
        "ids": torch.tensor([[11, 11], [12, -1]]),
        "masks": torch.zeros(2, 2, 16, 16),
    }
    frame_targets = flatten_video_targets_for_released_criterion([target])
    assert len(frame_targets) == 2
    assert frame_targets[0]["ids"].shape == (2, 1)
    assert frame_targets[1]["masks"].shape == (2, 1, 16, 16)


def test_video_axis_transforms_are_value_identical_to_released_einops_equations() -> None:
    batch, time, queries, classes, height, width = 2, 3, 200, 41, 2, 3
    class_logits = torch.arange(batch * time * queries * classes, dtype=torch.float32).reshape(
        batch, time, queries, classes
    )
    mask_logits = torch.arange(
        batch * queries * time * height * width,
        dtype=torch.float32,
    ).reshape(batch, queries, time, height, width)
    output = ExactVidEoMTOutput(
        class_logits=class_logits,
        mask_logits=mask_logits,
        query_embeddings=torch.zeros(batch, time, queries, 1024),
        propagated_queries=torch.zeros(batch, queries, 1024),
        auxiliary_outputs=(),
    )

    flat = flatten_video_outputs_for_released_criterion(output)
    expected_classes = class_logits.reshape(batch * time, queries, classes)
    expected_masks = mask_logits.permute(0, 2, 1, 3, 4).reshape(
        batch * time, queries, 1, height, width
    )
    torch.testing.assert_close(flat["pred_logits"], expected_classes)
    torch.testing.assert_close(flat["pred_masks"], expected_masks)

    labels = torch.tensor([2, 4])
    ids = torch.tensor([[0, 0, -1], [-1, 1, 1]])
    masks = torch.arange(2 * time * height * width, dtype=torch.float32).reshape(
        2, time, height, width
    )
    frame_targets = flatten_video_targets_for_released_criterion(
        [{"labels": labels, "ids": ids, "masks": masks}]
    )
    for frame, target in enumerate(frame_targets):
        assert target["labels"] is labels
        torch.testing.assert_close(target["ids"], ids[:, [frame]])
        torch.testing.assert_close(target["masks"], masks[:, [frame]])


def test_released_criterion_is_finite_and_reaches_logits() -> None:
    torch.manual_seed(198)
    output = _typed_output(auxiliary_count=0)
    flat = flatten_video_outputs_for_released_criterion(output)
    targets = []
    for frame in range(2):
        masks = torch.zeros(2, 1, 16, 16)
        masks[0, :, 1 + frame : 7 + frame, 2:8] = 1
        masks[1, :, 9:15, 8 - frame : 14 - frame] = 1
        targets.append(
            {
                "labels": torch.tensor([0, 3]),
                "ids": torch.tensor([[11], [12]]),
                "masks": masks,
            }
        )
    criterion = build_released_criterion()
    raw = criterion(flat, targets)
    weighted = apply_released_loss_weights(raw, criterion)
    assert set(weighted) == {"loss_ce", "loss_mask", "loss_dice"}
    loss = sum(weighted.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert output.class_logits.grad is not None
    assert output.mask_logits.grad is not None
    assert torch.isfinite(output.class_logits.grad).all()
    assert torch.isfinite(output.mask_logits.grad).all()


def test_release_intentionally_weights_only_three_of_four_auxiliary_reads() -> None:
    weights = released_weight_dict()
    assert len(weights) == 12
    assert "loss_ce_2" in weights
    assert "loss_ce_3" not in weights


def test_selected_online_criterion_preserves_consistent_matcher_and_inactive_reid_keys() -> None:
    criterion = build_released_online_criterion(num_frames=5)
    assert criterion.matcher.__class__.__name__ == "VideoHungarianMatcher_Consistent"
    assert criterion.matcher.frames == 5
    assert released_online_weight_dict() == {
        **released_weight_dict(),
        "loss_reid": 2.0,
        "loss_reid_aux": 2.0,
    }
