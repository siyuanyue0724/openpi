from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
discovery = pytest.importorskip("picf_next.models.discovery")

ObjectDiscoveryConfig = discovery.ObjectDiscoveryConfig
TaskIndependentObjectDiscovery = discovery.TaskIndependentObjectDiscovery
GEOMETRY = synthetic_geometry_contract(3)


def _model() -> TaskIndependentObjectDiscovery:
    torch.manual_seed(101)
    return TaskIndependentObjectDiscovery(
        ObjectDiscoveryConfig(
            input_dim=7,
            hidden_dim=16,
            num_queries=4,
            num_layers=2,
            num_heads=4,
            address_dim=5,
            content_dim=6,
            geometry_dim=3,
            geometry_contract=GEOMETRY,
            initial_variance=0.1,
            dropout=0.0,
        )
    ).eval()


def _evidence():
    torch.manual_seed(103)
    valid = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
        ]
    )
    features = torch.randn(2, 6, 7) * valid.unsqueeze(-1)
    return features, valid


def test_output_contract_and_ownership_simplex() -> None:
    model = _model()
    features, valid = _evidence()
    output = model(features, valid)

    assert output.query_features.shape == (2, 4, 16)
    assert output.address_mean.shape == (2, 4, 5)
    assert output.content_mean.shape == (2, 4, 6)
    assert output.geometry_mean.shape == (2, 4, 3)
    assert torch.equal(output.geometry_mean, torch.zeros_like(output.geometry_mean))
    assert torch.equal(model.geometry_head.weight, torch.zeros_like(model.geometry_head.weight))
    assert torch.equal(model.geometry_head.bias, torch.zeros_like(model.geometry_head.bias))
    assert output.observation_mean.shape == (2, 4, 14)
    assert output.geometry_variance.shape == (2, 4, 3)
    torch.testing.assert_close(
        torch.linalg.vector_norm(output.address_mean, dim=-1),
        torch.ones(2, 4),
    )
    assert output.existence.shape == (2, 4)
    torch.testing.assert_close(output.existence, torch.full_like(output.existence, 0.5))
    torch.testing.assert_close(
        output.localization_confidence,
        torch.full_like(output.localization_confidence, 0.5),
    )
    torch.testing.assert_close(
        output.measurement_probability,
        torch.full_like(output.measurement_probability, 0.25),
    )
    assert output.mask_quality.shape == output.existence.shape
    assert ((output.mask_quality >= 0.0) & (output.mask_quality <= 1.0)).all()
    assert (output.object_confidence <= output.existence.float()).all()
    assert torch.equal(model.existence_head.weight, torch.zeros_like(model.existence_head.weight))
    assert model.existence_head.bias.item() == pytest.approx(
        model.config.existence_calibration.training_logit_at_half_posterior
    )
    assert output.ownership.shape == (2, 6, 5)
    torch.testing.assert_close(
        output.ownership.sum(dim=-1),
        torch.ones(2, 6),
    )
    assert (output.geometry_variance >= model.config.minimum_variance).all()
    torch.testing.assert_close(
        output.geometry_variance,
        torch.full_like(output.geometry_variance, model.config.initial_variance),
    )
    assert torch.isfinite(output.observation_mean).all()
    assert len(output.auxiliary_outputs) == model.config.num_layers
    assert all(auxiliary.auxiliary_outputs == () for auxiliary in output.auxiliary_outputs)


def test_autocast_restores_one_posterior_compute_dtype() -> None:
    model = _model().to(dtype=torch.bfloat16)
    features, valid = _evidence()
    features = features.to(dtype=torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(features, valid)

    floating = (
        output.query_features,
        output.address_mean,
        output.content_mean,
        output.geometry_mean,
        output.geometry_variance,
        output.existence_logits,
        output.localization_confidence_logits,
        output.ownership_logits,
        output.ownership,
    )
    assert all(value.dtype == torch.bfloat16 for value in floating)


def test_input_token_permutation_is_equivariant() -> None:
    model = _model()
    features, valid = _evidence()
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    expected = model(features, valid)
    actual = model(features[:, permutation], valid[:, permutation])

    torch.testing.assert_close(actual.query_features, expected.query_features)
    torch.testing.assert_close(actual.observation_mean, expected.observation_mean)
    torch.testing.assert_close(actual.ownership, expected.ownership[:, permutation])


def test_padding_is_invariant_and_invalid_tokens_are_context_only() -> None:
    model = _model()
    features, valid = _evidence()
    expected = model(features, valid)
    padded_features = torch.cat((features, torch.zeros(2, 4, 7)), dim=1)
    padded_valid = torch.cat((valid, torch.zeros(2, 4, dtype=torch.bool)), dim=1)
    actual = model(padded_features, padded_valid)

    torch.testing.assert_close(actual.observation_mean, expected.observation_mean)
    torch.testing.assert_close(actual.existence, expected.existence)
    torch.testing.assert_close(actual.ownership[:, :6], expected.ownership)
    assert torch.equal(actual.ownership[:, 6:, :-1], torch.zeros(2, 4, 4))
    assert torch.equal(actual.ownership[:, 6:, -1], torch.ones(2, 4))


def test_query_parameter_permutation_only_permutes_object_axis() -> None:
    model = _model()
    permuted = copy.deepcopy(model)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        permuted.query_embeddings.copy_(model.query_embeddings[permutation])
    features, valid = _evidence()
    expected = model(features, valid)
    actual = permuted(features, valid)

    torch.testing.assert_close(actual.observation_mean, expected.observation_mean[:, permutation])
    torch.testing.assert_close(actual.existence, expected.existence[:, permutation])
    torch.testing.assert_close(
        actual.localization_confidence,
        expected.localization_confidence[:, permutation],
    )
    torch.testing.assert_close(actual.ownership[..., :-1], expected.ownership[..., permutation])
    torch.testing.assert_close(actual.context_ownership, expected.context_ownership)
    for actual_auxiliary, expected_auxiliary in zip(
        actual.auxiliary_outputs,
        expected.auxiliary_outputs,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_auxiliary.ownership[..., :-1],
            expected_auxiliary.ownership[..., permutation],
        )


def test_mask_coherence_score_requires_positive_object_vs_context_support() -> None:
    model = _model()
    features, valid = _evidence()
    output = model(features[:1], valid[:1])
    logits = torch.full_like(output.ownership_logits, -2.0)
    logits[..., -1] = 0.0
    logits[0, 0, 0] = 2.0
    controlled = replace(output, ownership_logits=logits)

    expected_quality = torch.zeros_like(controlled.existence)
    expected_quality[0, 0] = torch.sigmoid(torch.tensor(2.0))
    torch.testing.assert_close(controlled.mask_quality, expected_quality)
    torch.testing.assert_close(
        controlled.mask_coherence_score,
        controlled.existence.float() * expected_quality,
    )
    torch.testing.assert_close(
        controlled.object_confidence,
        controlled.measurement_probability,
    )


def test_empty_evidence_cannot_emit_birth_and_has_context_only_padding() -> None:
    model = _model()
    features = torch.zeros(2, 3, 7)
    valid = torch.zeros(2, 3, dtype=torch.bool)
    output = model(features, valid)

    assert torch.equal(output.existence, torch.zeros(2, 4))
    assert torch.equal(output.object_confidence, torch.zeros(2, 4))
    assert torch.equal(output.ownership[..., :-1], torch.zeros(2, 3, 4))
    assert torch.equal(output.context_ownership, torch.ones(2, 3))
    assert not output.evidence_available.any()


def test_active_contact_group_keeps_all_tokens_and_shares_one_assignment() -> None:
    model = _model()
    features, valid = _evidence()
    groups = torch.full_like(valid, -1, dtype=torch.long)
    groups[0, 2:5] = 7
    output = model(features, valid, groups)

    assert output.ownership.shape[1] == features.shape[1]
    expected = output.ownership[0, 2].expand(3, -1)
    torch.testing.assert_close(output.ownership[0, 2:5], expected, atol=0.0, rtol=0.0)
    assert not torch.equal(output.ownership[0, 0], output.ownership[0, 2])


def test_all_learned_prediction_heads_and_decoder_receive_gradient() -> None:
    model = _model().train()
    features, valid = _evidence()
    features.requires_grad_(True)
    output = model(features, valid)
    loss = (
        output.observation_mean.square().mean()
        + output.geometry_variance.mean()
        + output.existence.mean()
        + output.localization_confidence.mean()
        + output.ownership[..., :-1].square().mean()
    )
    loss.backward()

    assert features.grad is not None
    assert model.query_embeddings.grad is not None
    for head in (
        model.address_head,
        model.content_head,
        model.geometry_head,
        model.existence_head,
        model.localization_confidence_head,
        model.ownership_query,
        model.ownership_token,
        model.context_head,
    ):
        assert head.weight.grad is not None
    assert not model.variance_head.weight.requires_grad
    assert model.variance_head.weight.grad is None
    assert model.variance_head.bias.grad is not None
    assert all(layer.cross_read.value_projection.weight.grad is not None for layer in model.layers)
    assert all(layer.cross_read.output_projection.weight.grad is not None for layer in model.layers)


def test_variance_head_has_gradient_above_the_removed_unit_cap() -> None:
    model = _model().train()
    features, valid = _evidence()
    with torch.no_grad():
        model.variance_head.weight.zero_()
        model.variance_head.bias.fill_(2.0)

    output = model(features, valid)
    assert (output.geometry_variance > 1.0).all()
    output.geometry_variance.mean().backward()

    assert model.variance_head.bias.grad is not None
    assert (model.variance_head.bias.grad > 0.0).all()


def test_geometry_variance_is_axis_constant_and_ignores_legacy_weight() -> None:
    model = _model()
    features, valid = _evidence()
    expected = model(features, valid).geometry_variance

    changed_features = torch.randn_like(features) * valid.unsqueeze(-1)
    with torch.no_grad():
        model.variance_head.weight.normal_(mean=100.0, std=20.0)
    actual = model(changed_features, valid).geometry_variance

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, actual[:, :1].expand_as(actual))

    with torch.no_grad():
        model.variance_head.bias.add_(0.25)
    recalibrated = model(changed_features, valid).geometry_variance
    assert (recalibrated > actual).all()


def test_rejects_nonzero_invalid_padding() -> None:
    model = _model()
    with pytest.raises(ValueError, match="padding"):
        model(
            torch.ones(1, 2, 7),
            torch.zeros(1, 2, dtype=torch.bool),
        )


def test_rejects_group_on_invalid_token() -> None:
    model = _model()
    with pytest.raises(ValueError, match="invalid tokens"):
        model(
            torch.zeros(1, 2, 7),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.tensor([[3, -1]], dtype=torch.long),
        )
