from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
pytest.importorskip("olmo.hf_model.modeling_molmoact2")

from picf_next.models.discovery import (  # noqa: E402
    ObjectDiscoveryConfig,
    TaskIndependentObjectDiscovery,
)
from tools.audit_molmoact2_m2_read_normalization import (  # noqa: E402
    _ALL_STAGES,
    _cardinality_response,
    _constant_scaled_competitive_ownership,
    _enable_constant_scaled_read,
    _enable_source_recurrent_update,
)

_SOURCE_RECURRENT_UPSTREAM = (
    Path(__file__).resolve().parents[1]
    / "references/source_checkouts/slot-attention-normalization"
    / "sa_generalization/slot_attention/slot_attention.py"
)
_REQUIRES_SOURCE_RECURRENT_UPSTREAM = pytest.mark.skipif(
    not _SOURCE_RECURRENT_UPSTREAM.is_file(),
    reason="optional pinned Slot Attention source checkout is absent",
)


def _discovery() -> TaskIndependentObjectDiscovery:
    return TaskIndependentObjectDiscovery(
        ObjectDiscoveryConfig(
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
    )


def test_constant_scaled_ownership_preserves_assignment_mass() -> None:
    ownership = torch.tensor(
        [
            [
                [0.60, 0.20, 0.10, 0.10],
                [0.20, 0.60, 0.10, 0.10],
                [0.99, 0.003, 0.003, 0.004],
            ]
        ]
    )
    valid = torch.tensor([[True, True, False]])

    weights = _constant_scaled_competitive_ownership(ownership, valid)

    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.tensor([[0.40, 0.40, 0.10]]),
    )
    assert torch.equal(weights[:, 2], torch.zeros_like(weights[:, 2]))


def test_constant_scaled_ownership_keeps_low_support_smaller() -> None:
    ownership = torch.tensor(
        [
            [
                [0.90, 0.01, 0.09],
                [0.80, 0.01, 0.19],
                [0.70, 0.01, 0.29],
            ]
        ]
    )
    valid = torch.ones(1, 3, dtype=torch.bool)

    weights = _constant_scaled_competitive_ownership(ownership, valid)

    torch.testing.assert_close(weights.sum(dim=1), torch.tensor([[0.80, 0.01]]))


def test_constant_scaled_ownership_handles_no_valid_tokens() -> None:
    ownership = torch.full((2, 3, 4), 0.25)
    valid = torch.zeros(2, 3, dtype=torch.bool)

    weights = _constant_scaled_competitive_ownership(ownership, valid)

    assert torch.equal(weights, torch.zeros_like(weights))


def test_constant_scaled_ownership_backpropagates_through_context_competition() -> None:
    logits = torch.randn(1, 4, 4, requires_grad=True)
    ownership = torch.softmax(logits, dim=-1)
    valid = torch.ones(1, 4, dtype=torch.bool)

    weights = _constant_scaled_competitive_ownership(ownership, valid)
    weights[..., 0].sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[..., :-1].abs().sum() > 0.0
    assert logits.grad[..., -1].abs().sum() > 0.0


@_REQUIRES_SOURCE_RECURRENT_UPSTREAM
def test_source_recurrent_update_removes_query_mixing_and_adds_shared_gru() -> None:
    current_frame = SimpleNamespace(discovery=_discovery())

    _enable_source_recurrent_update(current_frame)

    parameter_names = tuple(name for name, _ in current_frame.discovery.named_parameters())
    assert all(hasattr(layer, "gru") for layer in current_frame.discovery.layers)
    assert current_frame.discovery.layers[0] is current_frame.discovery.layers[1]
    assert not any("self_attention" in name for name in parameter_names)
    assert not any("cross_read.output_projection" in name for name in parameter_names)
    assert any("gru.kernel" in name for name in parameter_names)


@_REQUIRES_SOURCE_RECURRENT_UPSTREAM
def test_source_recurrent_layer_is_query_permutation_equivariant() -> None:
    torch.manual_seed(23)
    current_frame = SimpleNamespace(discovery=_discovery().eval())
    _enable_source_recurrent_update(current_frame)
    layer = current_frame.discovery.layers[0]
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


@_REQUIRES_SOURCE_RECURRENT_UPSTREAM
def test_source_recurrent_layer_preserves_queries_without_evidence() -> None:
    torch.manual_seed(29)
    current_frame = SimpleNamespace(discovery=_discovery().eval())
    _enable_source_recurrent_update(current_frame)
    layer = current_frame.discovery.layers[0]
    queries = torch.randn(2, 3, 8)
    memory = torch.randn(2, 4, 8)
    valid = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    ownership = torch.softmax(torch.randn(2, 4, 4), dim=-1)

    actual = layer(queries, memory, valid, ownership)

    torch.testing.assert_close(actual[1], queries[1])
    assert not torch.equal(actual[0], queries[0])


@_REQUIRES_SOURCE_RECURRENT_UPSTREAM
def test_source_recurrent_supervises_only_post_evidence_predictions() -> None:
    torch.manual_seed(30)
    discovery = _discovery().eval()
    current_frame = SimpleNamespace(discovery=discovery)
    _enable_source_recurrent_update(current_frame)
    features = torch.randn(2, 6, 7)
    valid = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
        ]
    )
    features = features * valid.unsqueeze(-1)

    output = discovery(features, valid)

    assert len(output.auxiliary_outputs) == discovery.config.num_layers - 1


@_REQUIRES_SOURCE_RECURRENT_UPSTREAM
def test_source_recurrent_can_reproduce_legacy_no_evidence_supervision() -> None:
    torch.manual_seed(31)
    discovery = _discovery().eval()
    current_frame = SimpleNamespace(discovery=discovery)
    _enable_source_recurrent_update(
        current_frame,
        supervision_stages=_ALL_STAGES,
    )
    features = torch.randn(2, 6, 7)
    valid = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
        ]
    )
    features = features * valid.unsqueeze(-1)

    output = discovery(features, valid)

    assert len(output.auxiliary_outputs) == discovery.config.num_layers


@_REQUIRES_SOURCE_RECURRENT_UPSTREAM
def test_constant_treatment_changes_no_recurrent_parameters() -> None:
    current_frame = SimpleNamespace(discovery=_discovery())
    _enable_source_recurrent_update(current_frame)
    before = tuple(
        (name, tuple(parameter.shape))
        for name, parameter in current_frame.discovery.named_parameters()
    )

    _enable_constant_scaled_read(current_frame)

    after = tuple(
        (name, tuple(parameter.shape))
        for name, parameter in current_frame.discovery.named_parameters()
    )
    assert after == before


def test_cardinality_response_rejects_fixed_count_shortcut() -> None:
    grouped = {
        str(target): {
            "predicted_count_mean": 9.0,
            "exact_count_accuracy": 1.0 if target == 9 else 0.0,
        }
        for target in (7, 8, 9, 10)
    }

    response = _cardinality_response(grouped)

    assert response["least_squares_slope"] == pytest.approx(0.0)
    assert response["predicted_mean_range"] == pytest.approx(0.0)
    assert response["maximum_absolute_group_bias"] == pytest.approx(2.0)
    assert response["minimum_group_exact_count_accuracy"] == pytest.approx(0.0)


def test_cardinality_response_accepts_identity_count_relation() -> None:
    grouped = {
        str(target): {
            "predicted_count_mean": float(target),
            "exact_count_accuracy": 1.0,
        }
        for target in (7, 8, 9, 10)
    }

    response = _cardinality_response(grouped)

    assert response["least_squares_slope"] == pytest.approx(1.0)
    assert response["predicted_mean_range"] == pytest.approx(3.0)
    assert response["maximum_absolute_group_bias"] == pytest.approx(0.0)
    assert response["minimum_group_exact_count_accuracy"] == pytest.approx(1.0)


@_REQUIRES_SOURCE_RECURRENT_UPSTREAM
def test_source_gaussian_queries_are_random_in_train_and_fixed_in_eval() -> None:
    torch.manual_seed(32)
    discovery = _discovery()
    current_frame = SimpleNamespace(discovery=discovery)
    _enable_source_recurrent_update(
        current_frame,
        query_initialization="source_gaussian_train_fixed_eval",
    )
    features = torch.randn(2, 6, 7)
    valid = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
        ]
    )
    features = features * valid.unsqueeze(-1)

    discovery.eval()
    first_eval = discovery(features, valid)
    second_eval = discovery(features, valid)
    discovery.train()
    first_train = discovery(features, valid)
    second_train = discovery(features, valid)

    assert discovery.query_embeddings is None
    assert discovery.slot_mu.shape == (1, 1, discovery.config.hidden_dim)
    assert discovery.slot_logsigma.shape == (1, 1, discovery.config.hidden_dim)
    assert discovery.slot_eval_noise.shape == (
        discovery.config.num_queries,
        discovery.config.hidden_dim,
    )
    torch.testing.assert_close(first_eval.query_features, second_eval.query_features)
    assert not torch.equal(first_train.query_features, second_train.query_features)
    parameter_names = tuple(name for name, _ in discovery.named_parameters())
    assert "slot_mu" in parameter_names
    assert "slot_logsigma" in parameter_names
    assert "query_embeddings" not in parameter_names
