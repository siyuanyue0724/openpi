from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from picf_next.lingbot_native.wsa_da3_loss import (  # noqa: E402
    WSADA3TeacherTargets,
    build_wsa_da3_token_mask,
    compute_official_wsa_da3_loss,
)


def _layers(*, requires_grad: bool) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(218)
    return tuple(
        torch.randn(1, 2592, 2048, generator=generator, requires_grad=requires_grad)
        for _ in range(4)
    )


def test_view_mask_retains_all_tokens_from_each_valid_view() -> None:
    mask = build_wsa_da3_token_mask(
        torch.tensor([[True, False], [False, True]]),
        target_length=2592,
    )
    assert mask.shape == (2, 2592)
    assert mask[0, :1296].all() and not mask[0, 1296:].any()
    assert not mask[1, :1296].any() and mask[1, 1296:].all()


def test_identical_predictions_have_zero_official_loss() -> None:
    targets = _layers(requires_grad=False)
    loss, logs = compute_official_wsa_da3_loss(
        tuple(layer.clone().requires_grad_() for layer in targets),
        WSADA3TeacherTargets(
            layers=targets,
            view_valid=torch.ones(1, 2, dtype=torch.bool),
        ),
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=2e-6, rtol=0)
    assert set(logs) == {
        "loss_3d_q17_t11",
        "loss_3d_q23_t15",
        "loss_3d_q29_t19",
        "loss_3d_q35_t23",
    }


def test_official_loss_backpropagates_through_all_four_predictions() -> None:
    predictions = _layers(requires_grad=True)
    targets = _layers(requires_grad=False)
    loss, _ = compute_official_wsa_da3_loss(
        predictions,
        WSADA3TeacherTargets(
            layers=targets,
            view_valid=torch.tensor([[True, False]]),
        ),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(layer.grad is not None and torch.isfinite(layer.grad).all() for layer in predictions)


def test_missing_teacher_layer_is_rejected() -> None:
    with pytest.raises(ValueError, match="all four"):
        WSADA3TeacherTargets(
            layers=_layers(requires_grad=False)[:3],
            view_valid=torch.ones(1, 2, dtype=torch.bool),
        ).validate()
