from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.action_posterior_learning import (
    action_posterior_target_mass_loss,
    aggregate_action_posterior_distribution,
    conditional_action_posterior_distribution,
)
from picf_next.lingbot_native.action_posterior_receipt import (
    LingBotActionAttentionLayout,
    action_posterior_attention_receipt,
    build_lingbot_action_attention_layout,
    build_lingbot_joint_action_attention_layout,
)


def _layout(*, posterior_valid: torch.Tensor | None = None) -> LingBotActionAttentionLayout:
    if posterior_valid is None:
        posterior_valid = torch.tensor([[True, True]])
    return LingBotActionAttentionLayout(
        batch_size=1,
        query_count=3,
        key_count=5,
        native_prefix_count=2,
        compact_prefix_count=4,
        state_query_slice=slice(0, 1),
        action_query_slice=slice(1, 3),
        posterior_key_indices=torch.tensor([2, 3]),
        posterior_key_valid=posterior_valid,
        expanded_posterior_indices=torch.tensor([7, 8]),
        selected_inserted_indices=torch.tensor([7, 8]),
    )


def test_receipt_replays_gqa_attention_and_excludes_state_query() -> None:
    query = torch.zeros(1, 3, 2, 2)
    key = torch.zeros(1, 5, 1, 2)
    mask = torch.ones(1, 3, 5, dtype=torch.bool)
    query[:, 1] = torch.tensor([1.0, 0.0])
    query[:, 2] = torch.tensor([0.0, 1.0])
    key[:, 2, 0] = torch.tensor([1.0, 0.0])
    key[:, 3, 0] = torch.tensor([0.0, 1.0])

    receipt = action_posterior_attention_receipt(
        query_states=query,
        key_states=key,
        attention_mask=mask,
        layout=_layout(),
        layer_index=4,
        layer_count=6,
    )

    repeated = key.float().repeat_interleave(2, dim=2)
    logits = torch.einsum("bahd,bkhd->bhak", query[:, 1:].float(), repeated)
    reference = torch.softmax(logits * (2**-0.5), dim=-1)[..., (2, 3)]
    assert receipt.posterior_attention.shape == (1, 2, 2, 2)
    assert receipt.layer_index == 4
    assert receipt.layer_count == 6
    assert torch.allclose(receipt.posterior_attention, reference)
    assert torch.allclose(receipt.total_posterior_mass, reference.sum(dim=-1))


def test_receipt_assigns_exact_zero_mass_to_blocked_posterior_key() -> None:
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 5, 1, 4)
    mask = torch.ones(1, 3, 5, dtype=torch.bool)
    mask[:, 1:, 2] = False
    receipt = action_posterior_attention_receipt(
        query_states=query,
        key_states=key,
        attention_mask=mask,
        layout=_layout(),
        layer_index=4,
        layer_count=6,
    )
    assert torch.equal(receipt.posterior_attention[..., 0], torch.zeros(1, 2, 2))


def test_receipt_rejects_visible_invalid_posterior_row() -> None:
    with pytest.raises(ValueError, match="invalid posterior"):
        action_posterior_attention_receipt(
            query_states=torch.zeros(1, 3, 2, 2),
            key_states=torch.zeros(1, 5, 1, 2),
            attention_mask=torch.ones(1, 3, 5, dtype=torch.bool),
            layout=_layout(posterior_valid=torch.tensor([[True, False]])),
            layer_index=4,
            layer_count=6,
        )


def test_layout_rejects_posterior_alias_of_native_prefix() -> None:
    with pytest.raises(ValueError, match="native prefix"):
        LingBotActionAttentionLayout(
            batch_size=1,
            query_count=3,
            key_count=5,
            native_prefix_count=2,
            compact_prefix_count=4,
            state_query_slice=slice(0, 1),
            action_query_slice=slice(1, 3),
            posterior_key_indices=torch.tensor([1, 3]),
            posterior_key_valid=torch.ones(1, 2, dtype=torch.bool),
            expanded_posterior_indices=torch.tensor([7, 8]),
            selected_inserted_indices=torch.tensor([7, 8]),
        )


def test_layout_rejects_incorrect_expanded_to_compact_mapping() -> None:
    with pytest.raises(ValueError, match="do not match"):
        LingBotActionAttentionLayout(
            batch_size=1,
            query_count=3,
            key_count=6,
            native_prefix_count=2,
            compact_prefix_count=5,
            state_query_slice=slice(0, 1),
            action_query_slice=slice(1, 3),
            posterior_key_indices=torch.tensor([2, 3]),
            posterior_key_valid=torch.ones(1, 2, dtype=torch.bool),
            expanded_posterior_indices=torch.tensor([8, 9]),
            selected_inserted_indices=torch.tensor([7, 8, 9]),
        )


def test_layout_builder_maps_expanded_posterior_rows_into_compact_cache() -> None:
    layout = build_lingbot_action_attention_layout(
        batch_size=2,
        native_prefix_count=4,
        suffix_count=6,
        selected_inserted_indices=torch.tensor([5, 7, 8]),
        expanded_posterior_indices=torch.tensor([7, 8]),
        posterior_key_valid=torch.tensor([[True, True], [True, False]]),
    )
    assert layout.compact_prefix_count == 7
    assert layout.key_count == 13
    assert layout.action_query_slice == slice(1, 6)
    assert torch.equal(layout.posterior_key_indices, torch.tensor([5, 6]))


def test_joint_layout_replays_actual_noncompact_action_posterior_surface() -> None:
    query = torch.zeros(1, 9, 2, 2)
    key = torch.zeros(1, 12, 2, 2)
    mask = torch.ones(1, 9, 12, dtype=torch.bool)
    query[:, 7] = torch.tensor([1.0, 0.0])
    query[:, 8] = torch.tensor([0.0, 1.0])
    key[:, 4] = torch.tensor([1.0, 0.0])
    key[:, 6] = torch.tensor([0.0, 1.0])
    layout = build_lingbot_joint_action_attention_layout(
        batch_size=1,
        query_count=9,
        key_count=12,
        action_query_slice=slice(7, 9),
        posterior_key_indices=torch.tensor([4, 6]),
        posterior_key_valid=torch.tensor([[True, True]]),
    )

    receipt = action_posterior_attention_receipt(
        query_states=query,
        key_states=key,
        attention_mask=mask,
        layout=layout,
        layer_index=3,
        layer_count=6,
    )

    logits = torch.einsum("bahd,bkhd->bhak", query[:, 7:].float(), key.float())
    reference = torch.softmax(logits * (2**-0.5), dim=-1)[..., (4, 6)]
    torch.testing.assert_close(receipt.posterior_attention, reference)
    torch.testing.assert_close(receipt.total_posterior_mass, reference.sum(dim=-1))


def test_joint_layout_rejects_out_of_axis_posterior_key() -> None:
    with pytest.raises(ValueError, match="outside the key axis"):
        build_lingbot_joint_action_attention_layout(
            batch_size=1,
            query_count=9,
            key_count=12,
            action_query_slice=slice(7, 9),
            posterior_key_indices=torch.tensor([4, 12]),
            posterior_key_valid=torch.tensor([[True, True]]),
        )


def test_target_mass_uses_full_key_mass_without_posterior_renormalization() -> None:
    attention = torch.tensor(
        [[[[0.10, 0.20], [0.15, 0.05]], [[0.20, 0.10], [0.05, 0.15]]]],
        requires_grad=True,
    )
    result = action_posterior_target_mass_loss(
        attention,
        target_row_weights=torch.tensor([[1.0, 0.0]]),
        target_valid=torch.tensor([True]),
    )
    expected = torch.tensor([[0.15, 0.10]])
    assert torch.allclose(result.target_mass, expected)
    assert torch.allclose(result.total_posterior_mass, torch.tensor([[0.30, 0.20]]))
    assert torch.allclose(result.loss, -torch.log(expected).mean())
    result.loss.backward()
    assert attention.grad is not None and torch.isfinite(attention.grad).all()


def test_target_mass_fixed_head_scope_leaves_unselected_heads_without_loss_gradient() -> None:
    attention = torch.tensor(
        [
            [
                [[0.20, 0.10], [0.10, 0.20]],
                [[0.80, 0.10], [0.70, 0.20]],
                [[0.40, 0.10], [0.30, 0.20]],
                [[0.60, 0.10], [0.50, 0.20]],
            ]
        ],
        requires_grad=True,
    )
    head_indices = torch.tensor([0, 2], dtype=torch.long)

    result = action_posterior_target_mass_loss(
        attention,
        target_row_weights=torch.tensor([[1.0, 0.0]]),
        target_valid=torch.tensor([True]),
        head_indices=head_indices,
    )

    assert torch.allclose(result.target_mass, torch.tensor([[0.30, 0.20]]))
    result.loss.backward()
    assert attention.grad is not None
    assert attention.grad[:, head_indices].abs().sum() > 0
    assert torch.equal(attention.grad[:, [1, 3]], torch.zeros_like(attention.grad[:, [1, 3]]))


def test_target_mass_all_invalid_has_connected_exact_zero_gradient() -> None:
    attention = torch.rand(2, 3, 4, 5, requires_grad=True)
    result = action_posterior_target_mass_loss(
        attention,
        target_row_weights=torch.zeros(2, 5),
        target_valid=torch.zeros(2, dtype=torch.bool),
    )
    assert result.loss.item() == 0.0
    assert not result.valid_entries.any()
    result.loss.backward()
    assert attention.grad is not None
    assert torch.equal(attention.grad, torch.zeros_like(attention))


def test_target_mass_gradient_reaches_action_queries_and_posterior_keys() -> None:
    query = torch.randn(1, 3, 2, 4, requires_grad=True)
    key = torch.randn(1, 5, 1, 4, requires_grad=True)
    receipt = action_posterior_attention_receipt(
        query_states=query,
        key_states=key,
        attention_mask=torch.ones(1, 3, 5, dtype=torch.bool),
        layout=_layout(),
        layer_index=4,
        layer_count=6,
    )
    result = action_posterior_target_mass_loss(
        receipt.posterior_attention,
        target_row_weights=torch.tensor([[1.0, 0.0]]),
        target_valid=torch.tensor([True]),
    )
    result.loss.backward()
    assert query.grad is not None and query.grad[:, 1:].abs().sum() > 0
    assert key.grad is not None and key.grad[:, 2:4].abs().sum() > 0


def test_conditional_action_posterior_distribution_averages_heads_then_rows() -> None:
    attention = torch.tensor(
        [[[[0.10, 0.30], [0.20, 0.20]], [[0.30, 0.10], [0.10, 0.30]]]]
    )

    distribution = conditional_action_posterior_distribution(attention)

    assert distribution.shape == (1, 2, 2)
    assert torch.allclose(distribution.sum(dim=-1), torch.ones(1, 2))
    assert torch.allclose(distribution[0, 0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(distribution[0, 1], torch.tensor([0.375, 0.625]))


def test_conditional_action_posterior_distribution_rejects_zero_mass() -> None:
    with pytest.raises(ValueError, match="no posterior-row mass"):
        conditional_action_posterior_distribution(torch.zeros(1, 2, 3, 4))


def test_aggregate_action_posterior_distribution_weights_actual_adoption_mass() -> None:
    attention = torch.tensor([[[[0.90, 0.00]], [[0.00, 0.10]]]])

    distribution = aggregate_action_posterior_distribution(attention)

    assert distribution.shape == (1, 1, 2)
    assert torch.allclose(distribution[0, 0], torch.tensor([0.9, 0.1]))


def test_aggregate_action_posterior_distribution_is_action_count_invariant() -> None:
    one_action = torch.tensor([[[[0.2, 0.8]]]])
    repeated = one_action.expand(-1, -1, 7, -1).clone()

    assert torch.allclose(
        aggregate_action_posterior_distribution(one_action),
        aggregate_action_posterior_distribution(repeated),
    )
