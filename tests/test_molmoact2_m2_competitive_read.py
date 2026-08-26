from __future__ import annotations

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")

from picf_next.models.discovery import (  # noqa: E402
    ObjectDiscoveryConfig,
    TaskIndependentObjectDiscovery,
    _normalized_competitive_ownership,
    _SetDecoderLayer,
)


def _config() -> ObjectDiscoveryConfig:
    return ObjectDiscoveryConfig(
        input_dim=7,
        hidden_dim=8,
        num_queries=3,
        num_layers=2,
        num_heads=2,
        address_dim=4,
        content_dim=5,
        geometry_dim=3,
        geometry_contract=synthetic_geometry_contract(3),
        initial_variance=0.2,
        dropout=0.0,
    )


def test_competitive_ownership_normalizes_each_query_over_valid_tokens() -> None:
    ownership = torch.tensor(
        [
            [
                [0.20, 0.30, 0.10, 0.40],
                [0.40, 0.10, 0.20, 0.30],
                [0.90, 0.02, 0.03, 0.05],
            ]
        ]
    )
    valid = torch.tensor([[True, True, False]])

    normalized = _normalized_competitive_ownership(ownership, valid)

    torch.testing.assert_close(normalized.sum(dim=1), torch.ones(1, 3))
    assert torch.equal(normalized[:, 2], torch.zeros_like(normalized[:, 2]))


def test_competitive_cross_read_ignores_invalid_token_values() -> None:
    torch.manual_seed(11)
    layer = _SetDecoderLayer(_config()).eval()
    memory = torch.randn(2, 4, 8)
    valid = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    ownership = torch.softmax(torch.randn(2, 4, 4), dim=-1)
    expected = layer.cross_read(memory, valid, ownership)
    changed = memory.clone()
    changed[~valid] = 1e5
    changed_ownership = ownership.clone()
    changed_ownership[~valid] = torch.tensor([0.99, 0.003, 0.003, 0.004])

    actual = layer.cross_read(changed, valid, changed_ownership)

    torch.testing.assert_close(actual, expected)


def test_competitive_cross_read_implements_normalized_ownership_equation() -> None:
    layer = _SetDecoderLayer(_config()).eval()
    with torch.no_grad():
        identity = torch.eye(8)
        layer.cross_read.value_projection.weight.copy_(identity)
        layer.cross_read.value_projection.bias.zero_()
        layer.cross_read.output_projection.weight.copy_(identity)
        layer.cross_read.output_projection.bias.zero_()
    memory = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    valid = torch.tensor([[True, True, False]])
    ownership = torch.tensor(
        [
            [
                [0.60, 0.20, 0.10, 0.10],
                [0.20, 0.60, 0.10, 0.10],
                [0.99, 0.003, 0.003, 0.004],
            ]
        ]
    )
    weights = _normalized_competitive_ownership(ownership, valid)

    actual = layer.cross_read(memory, valid, ownership)
    expected = torch.einsum("bnk,bnh->bkh", weights, memory)

    torch.testing.assert_close(actual, expected)


def test_competitive_layer_is_query_permutation_equivariant() -> None:
    torch.manual_seed(13)
    layer = _SetDecoderLayer(_config()).eval()
    queries = torch.randn(2, 3, 8)
    memory = torch.randn(2, 5, 8)
    valid = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )
    ownership = torch.softmax(torch.randn(2, 5, 4), dim=-1)
    permutation = torch.tensor([2, 0, 1])
    expected = layer(queries, memory, valid, ownership)
    permuted_ownership = torch.cat(
        (ownership[..., :-1][..., permutation], ownership[..., -1:]),
        dim=-1,
    )

    actual = layer(queries[:, permutation], memory, valid, permuted_ownership)

    torch.testing.assert_close(actual, expected[:, permutation])


def test_competitive_read_backpropagates_through_slot_competition() -> None:
    torch.manual_seed(17)
    layer = _SetDecoderLayer(_config()).eval()
    memory = torch.randn(1, 4, 8)
    valid = torch.tensor([[True, True, True, True]])
    logits = torch.randn(1, 4, 4, requires_grad=True)
    ownership = torch.softmax(logits, dim=-1)

    update = layer.cross_read(memory, valid, ownership)
    update.square().sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[..., :-1].abs().sum() > 0.0
    assert logits.grad[..., -1].abs().sum() > 0.0


def test_production_discovery_uses_only_value_and_output_cross_read_parameters() -> None:
    torch.manual_seed(19)
    discovery = TaskIndependentObjectDiscovery(_config()).eval()
    parameter_names = tuple(name for name, _ in discovery.named_parameters())
    features = torch.randn(2, 6, 7)
    valid = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
        ]
    )
    features = features * valid.unsqueeze(-1)

    output = discovery(features, valid)

    assert any("cross_read.value_projection" in name for name in parameter_names)
    assert any("cross_read.output_projection" in name for name in parameter_names)
    assert not any("cross_attention" in name or "cross_norm" in name for name in parameter_names)
    assert output.ownership.shape == (2, 6, 4)
    assert output.existence_logits.shape == (2, 3)
    assert len(output.auxiliary_outputs) == discovery.config.num_layers
    assert torch.isfinite(output.observation_mean).all()
    torch.testing.assert_close(
        output.ownership.sum(dim=-1),
        torch.ones(2, 6),
    )
