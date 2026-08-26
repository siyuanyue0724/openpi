from __future__ import annotations

import pytest
import torch
from torch import nn

from picf_next.videomt_exact.posterior_refiner import (
    PREPEND_PROJECTED_ROWS,
    REPLACE_WITH_RELEASED_PROPAGATION,
    FrozenVidEoMTPosteriorRowRefiner,
)


class _IdentityAttention(nn.Module):
    def forward(self, hidden_states, **_unused):
        return (hidden_states,)


class _ReleasedBlockProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.Identity()
        self.attention = _IdentityAttention()
        self.drop_path = nn.Identity()
        self.layer_scale1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.mlp = nn.Identity()
        self.layer_scale2 = nn.Identity()


def _refiner_probe() -> FrozenVidEoMTPosteriorRowRefiner:
    refiner = FrozenVidEoMTPosteriorRowRefiner.__new__(FrozenVidEoMTPosteriorRowRefiner)
    nn.Module.__init__(refiner)
    refiner.blocks = nn.ModuleList(_ReleasedBlockProbe() for _ in range(4))
    refiner.norm = nn.Identity()
    refiner.class_head = nn.Linear(1024, 41)
    refiner.mask_head = nn.Identity()
    refiner.upscale = nn.Identity()
    refiner.query_prior = nn.Embedding(3, 1024)
    refiner.query_updater = nn.Identity()
    refiner.source_query_count = 3
    refiner.prefix_token_count = 5
    refiner.norm_queries = True
    refiner.query_integration = PREPEND_PROJECTED_ROWS
    return refiner.requires_grad_(False).eval()


def test_checkpointed_refinement_preserves_row_and_projection_gradients() -> None:
    refiner = _refiner_probe()
    rows = torch.randn(1, 2, 4, requires_grad=True)
    projection = torch.randn(4, 1024, requires_grad=True)
    source_tokens = torch.randn(1, 3 + 5 + 4, 1024)

    output = refiner(
        posterior_rows=rows,
        semantic_projection_weight=projection,
        segmenter_input_tokens=source_tokens,
        position_cos=torch.randn(4, 64),
        position_sin=torch.randn(4, 64),
        patch_grid_height=2,
        patch_grid_width=2,
    )
    assert output.support_logits.shape == (1, 4, 2)
    assert output.class_logits.shape == (1, 2, 41)
    output.support_logits.square().mean().backward()

    for gradient in (rows.grad, projection.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in refiner.parameters())


def test_block_checkpointing_matches_direct_values_and_input_gradients() -> None:
    refiner = _refiner_probe()
    source_tokens = torch.randn(1, 3 + 5 + 4, 1024)
    cos = torch.randn(4, 64)
    sin = torch.randn(4, 64)
    direct_rows = torch.randn(1, 2, 4, requires_grad=True)
    direct_projection = torch.randn(4, 1024, requires_grad=True)
    checkpointed_rows = direct_rows.detach().clone().requires_grad_(True)
    checkpointed_projection = direct_projection.detach().clone().requires_grad_(True)

    direct = refiner._refine_and_predict(
        direct_rows,
        direct_projection,
        source_tokens,
        cos,
        sin,
        patch_grid_height=2,
        patch_grid_width=2,
        checkpoint_blocks=False,
    )
    checkpointed = refiner._refine_and_predict(
        checkpointed_rows,
        checkpointed_projection,
        source_tokens,
        cos,
        sin,
        patch_grid_height=2,
        patch_grid_width=2,
        checkpoint_blocks=True,
    )
    for candidate, reference in zip(checkpointed, direct, strict=True):
        torch.testing.assert_close(candidate, reference)

    direct_loss = direct[0].square().mean() + direct[1].square().mean()
    checkpointed_loss = checkpointed[0].square().mean() + checkpointed[1].square().mean()
    direct_loss.backward()
    checkpointed_loss.backward()
    torch.testing.assert_close(checkpointed_rows.grad, direct_rows.grad)
    torch.testing.assert_close(checkpointed_projection.grad, direct_projection.grad)
    assert all(parameter.grad is None for parameter in refiner.parameters())


def test_eval_refinement_without_grad_uses_the_same_forward_function() -> None:
    refiner = _refiner_probe()
    rows = torch.randn(1, 2, 4)
    projection = torch.randn(4, 1024)
    source_tokens = torch.randn(1, 3 + 5 + 4, 1024)
    cos = torch.randn(4, 64)
    sin = torch.randn(4, 64)

    with torch.no_grad():
        output = refiner(
            posterior_rows=rows,
            semantic_projection_weight=projection,
            segmenter_input_tokens=source_tokens,
            position_cos=cos,
            position_sin=sin,
            patch_grid_height=2,
            patch_grid_width=2,
        )
        expected = refiner._refine_and_predict(
            rows,
            projection,
            source_tokens,
            cos,
            sin,
            patch_grid_height=2,
            patch_grid_width=2,
        )

    torch.testing.assert_close(output.support_logits, expected[0])
    torch.testing.assert_close(output.class_logits, expected[1])


def test_released_propagation_replaces_rows_without_changing_source_length() -> None:
    refiner = _refiner_probe()
    refiner.query_integration = REPLACE_WITH_RELEASED_PROPAGATION
    decoded = torch.randn(1, 2, 1024)
    source = torch.randn(1, 3 + 5 + 4, 1024)
    with torch.no_grad():
        refiner.query_prior.weight[0].fill_(1.0)
        refiner.query_prior.weight[1].fill_(2.0)

    composed = refiner._compose_source_input(decoded, source)

    assert composed.shape == source.shape
    torch.testing.assert_close(composed[:, 0], decoded[:, 0] + 1.0)
    torch.testing.assert_close(composed[:, 1], decoded[:, 1] + 2.0)
    torch.testing.assert_close(composed[:, 2:], source[:, 2:])


def test_released_propagation_rejects_more_rows_than_source_queries() -> None:
    refiner = _refiner_probe()
    refiner.query_integration = REPLACE_WITH_RELEASED_PROPAGATION

    with (
        torch.no_grad(),
        pytest.raises(
            ValueError,
            match="exceed the released source query bank",
        ),
    ):
        refiner._compose_source_input(
            torch.randn(1, 4, 1024),
            torch.randn(1, 3 + 5 + 4, 1024),
        )
