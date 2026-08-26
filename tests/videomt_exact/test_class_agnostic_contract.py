from __future__ import annotations

import pytest
import torch

from picf_next._vendor.videomt.matcher import (
    VideoHungarianMatcher,
    VideoHungarianMatcher_Consistent,
)
from picf_next.videomt_exact.class_agnostic import (
    VIDEOMT_FRAME_LOCAL_MATCHER_ABLATION,
    VIDEOMT_ONLINE_CONSISTENT_MATCHER,
    build_class_agnostic_criterion,
    flatten_class_agnostic_outputs,
    flatten_class_agnostic_targets,
    marginalize_videomt_taxonomy,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput
from picf_next.videomt_exact.training import apply_released_loss_weights


def _output(*, time: int = 2) -> ExactVidEoMTOutput:
    return ExactVidEoMTOutput(
        class_logits=torch.randn(1, time, 200, 41, requires_grad=True),
        mask_logits=torch.randn(1, 200, time, 8, 8, requires_grad=True),
        query_embeddings=torch.randn(1, time, 200, 1024),
        propagated_queries=torch.randn(1, 200, 1024),
        auxiliary_outputs=(),
    )


def _targets() -> list[dict[str, torch.Tensor]]:
    masks = torch.zeros(2, 2, 16, 16)
    masks[0, :, 1:7, 2:8] = 1
    masks[1, :, 9:15, 8:14] = 1
    return [
        {
            "labels": torch.tensor([17, 39]),
            "ids": torch.tensor([[0, 0], [1, 1]]),
            "masks": masks,
        }
    ]


def test_taxonomy_marginalization_exactly_preserves_foreground_mass() -> None:
    torch.manual_seed(198)
    logits = torch.randn(3, 7, 41, requires_grad=True)
    binary = marginalize_videomt_taxonomy(logits)
    expected = logits.softmax(dim=-1)[..., :40].sum(dim=-1)
    actual = binary.softmax(dim=-1)[..., 0]
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)

    actual.sum().backward()
    assert logits.grad is not None
    assert (logits.grad.abs().sum(dim=(0, 1)) > 0).all()


def test_class_agnostic_objective_reuses_released_losses_and_reaches_heads() -> None:
    torch.manual_seed(199)
    output = _output()
    flattened = flatten_class_agnostic_outputs(output)
    targets = flatten_class_agnostic_targets(_targets())
    criterion = build_class_agnostic_criterion(num_frames=2)

    assert flattened["pred_logits"].shape == (2, 200, 2)
    assert all((target["labels"] == 0).all() for target in targets)
    assert isinstance(criterion.matcher, VideoHungarianMatcher_Consistent)
    raw = criterion(flattened, targets)
    weighted = apply_released_loss_weights(raw, criterion)
    loss = sum(weighted.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert output.class_logits.grad is not None
    assert output.mask_logits.grad is not None
    assert (output.class_logits.grad.abs().sum(dim=(0, 1, 2)) > 0).all()
    assert output.mask_logits.grad.abs().sum() > 0


def test_default_identity_arm_is_the_selected_unmodified_online_matcher() -> None:
    criterion = build_class_agnostic_criterion(
        matcher_identity=VIDEOMT_ONLINE_CONSISTENT_MATCHER,
        num_frames=2,
    )
    assert isinstance(criterion.matcher, VideoHungarianMatcher_Consistent)
    assert criterion.matcher.frames == 2
    assert criterion.weight_dict["loss_reid"] == 2.0
    assert criterion.weight_dict["loss_reid_aux"] == 2.0


def test_frame_local_matcher_is_only_an_explicit_ablation() -> None:
    criterion = build_class_agnostic_criterion(
        matcher_identity=VIDEOMT_FRAME_LOCAL_MATCHER_ABLATION,
        num_frames=2,
    )
    assert type(criterion.matcher) is VideoHungarianMatcher
    assert "loss_reid" not in criterion.weight_dict


def test_consistent_target_adapter_rejects_noncanonical_identity_values() -> None:
    targets = _targets()
    targets[0]["ids"][0] = 17
    with pytest.raises(ValueError, match="target row or -1"):
        flatten_class_agnostic_targets(targets)


def test_class_agnostic_target_adapter_preserves_measured_pixel_domain() -> None:
    targets = _targets()
    valid_pixels = torch.ones(2, 16, 16, dtype=torch.bool)
    valid_pixels[:, :3, :5] = False
    targets[0]["valid_pixels"] = valid_pixels

    flattened = flatten_class_agnostic_targets(targets)

    assert len(flattened) == 2
    for frame, target in enumerate(flattened):
        torch.testing.assert_close(target["valid_pixels"], valid_pixels[[frame]])


def test_selected_online_objective_accepts_a_fully_cropped_empty_clip() -> None:
    torch.manual_seed(200)
    output = _output(time=5)
    flattened = flatten_class_agnostic_outputs(output)
    targets = flatten_class_agnostic_targets(
        [
            {
                "labels": torch.empty(0, dtype=torch.long),
                "ids": torch.empty(0, 5, dtype=torch.long),
                "masks": torch.empty(0, 5, 16, 16),
            }
        ]
    )
    criterion = build_class_agnostic_criterion(num_frames=5)

    raw = criterion(flattened, targets)
    weighted = apply_released_loss_weights(raw, criterion)
    assert set(weighted) == {"loss_ce", "loss_mask", "loss_dice"}
    total = sum(weighted.values())
    assert torch.isfinite(total)
    total.backward()
    assert output.class_logits.grad is not None
    assert output.mask_logits.grad is not None
