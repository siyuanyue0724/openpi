from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
binding = pytest.importorskip("picf_next.models.binding_loss")
evidence = pytest.importorskip("picf_next.models.evidence")
set_loss = pytest.importorskip("picf_next.models.set_loss")

BindingLossConfig = binding.BindingLossConfig
MultimodalBindingCriterion = binding.MultimodalBindingCriterion
pool_object_modality_views = binding._pool_object_modality_views
sigmoid_density_ratio_loss = binding.sigmoid_density_ratio_loss
BindingProjectionOutput = evidence.BindingProjectionOutput
ModalityTokenSpan = evidence.ModalityTokenSpan
NativeTokenBank = evidence.NativeTokenBank
ObjectSetTarget = set_loss.ObjectSetTarget


def _projection(features: torch.Tensor) -> BindingProjectionOutput:
    valid = torch.ones(1, features.shape[1], dtype=torch.bool)
    modality = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    return BindingProjectionOutput(
        native_banks=(
            NativeTokenBank("vision", features[:, :2], valid[:, :2]),
            NativeTokenBank("touch", features[:, 2:], valid[:, 2:]),
        ),
        binding_features=features,
        token_valid=valid,
        current_measurement_valid=valid,
        token_group_id=torch.full_like(valid, -1, dtype=torch.long),
        modality_index=modality,
        spans=(ModalityTokenSpan("vision", 0, 2), ModalityTokenSpan("touch", 2, 4)),
    )


def _target(permutation: torch.Tensor | None = None) -> ObjectSetTarget:
    ownership = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    if permutation is not None:
        ownership = torch.cat((ownership[:, :-1][:, permutation], ownership[:, -1:]), dim=-1)
    return ObjectSetTarget(ownership, torch.ones(4, dtype=torch.bool))


@pytest.mark.parametrize("objective", ["sigmoid", "multi_positive_infonce"])
def test_aligned_cross_modal_objects_are_better_than_swapped(objective: str) -> None:
    aligned = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
        requires_grad=True,
    )
    swapped = aligned.detach().clone()
    swapped[:, 2:] = swapped[:, 2:].flip(1)
    criterion = MultimodalBindingCriterion(BindingLossConfig(objective=objective, temperature=0.1))

    good = criterion(_projection(aligned), (_target(),))
    bad = criterion(_projection(swapped), (_target(),))

    assert good.loss < bad.loss
    assert good.object_modality_views == 4
    assert good.positive_pairs == 4
    assert good.negative_pairs == 4
    good.loss.backward()
    assert aligned.grad is not None and aligned.grad.abs().sum() > 0.0


def test_object_target_permutation_and_duplicate_token_density_do_not_change_loss() -> None:
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    criterion = MultimodalBindingCriterion(BindingLossConfig(temperature=0.2))
    expected = criterion(_projection(features), (_target(),)).loss
    permuted = criterion(_projection(features), (_target(torch.tensor([1, 0])),)).loss
    torch.testing.assert_close(permuted, expected)

    duplicated_features = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
    )
    valid = torch.ones(1, 5, dtype=torch.bool)
    projection = BindingProjectionOutput(
        native_banks=(),
        binding_features=duplicated_features,
        token_valid=valid,
        current_measurement_valid=valid,
        token_group_id=torch.full_like(valid, -1, dtype=torch.long),
        modality_index=torch.tensor([[0, 0, 0, 1, 1]]),
        spans=(),
    )
    duplicated_target = ObjectSetTarget(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        torch.ones(5, dtype=torch.bool),
    )
    actual = criterion(projection, (duplicated_target,)).loss
    torch.testing.assert_close(actual, expected)


def test_context_unsupervised_or_missing_modality_produces_no_false_pair() -> None:
    features = torch.randn(1, 4, 3, requires_grad=True)
    projection = _projection(features)
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        token_valid=torch.ones(4, dtype=torch.bool),
        token_supervised=torch.tensor([True, True, False, False]),
    )
    output = MultimodalBindingCriterion()(projection, (target,))

    assert output.object_modality_views == 1
    assert output.positive_pairs == 0
    assert output.negative_pairs == 0
    assert output.loss == 0.0
    output.loss.backward()
    assert torch.equal(features.grad, torch.zeros_like(features))


@pytest.mark.parametrize("objective", ["sigmoid", "multi_positive_infonce"])
def test_relation_graph_with_only_negatives_is_inactive(objective: str) -> None:
    features = torch.randn(1, 4, 3, requires_grad=True)
    projection = _projection(features)
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        token_valid=torch.ones(4, dtype=torch.bool),
        token_supervised=torch.tensor([True, False, True, False]),
    )
    output = MultimodalBindingCriterion(BindingLossConfig(objective=objective))(
        projection,
        (target,),
    )

    assert output.positive_pairs == 0
    assert output.negative_pairs == 2
    assert output.loss == 0.0
    output.loss.backward()
    assert torch.equal(features.grad, torch.zeros_like(features))


def test_rejects_misaligned_or_non_simplex_training_targets() -> None:
    projection = _projection(torch.randn(1, 4, 3))
    bad_validity = ObjectSetTarget(
        _target().ownership,
        torch.tensor([True, True, True, False]),
    )
    with pytest.raises(ValueError, match="validity must match"):
        MultimodalBindingCriterion()(projection, (bad_validity,))

    ownership = _target().ownership.clone()
    ownership[0, 0] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        MultimodalBindingCriterion()(
            projection,
            (ObjectSetTarget(ownership, torch.ones(4, dtype=torch.bool)),),
        )

    negative = _target().ownership.clone()
    negative[0] = torch.tensor([-0.5, 1.5, 0.0])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        MultimodalBindingCriterion()(
            projection,
            (ObjectSetTarget(negative, torch.ones(4, dtype=torch.bool)),),
        )


def test_rejects_differentiable_loss_only_ownership() -> None:
    projection = _projection(torch.randn(1, 4, 3))
    ownership = _target().ownership.clone().requires_grad_(True)

    with pytest.raises(ValueError, match="loss-only binding ownership"):
        MultimodalBindingCriterion()(
            projection,
            (ObjectSetTarget(ownership, torch.ones(4, dtype=torch.bool)),),
        )


def test_infonce_rejects_an_unidentifiable_logit_bias() -> None:
    with pytest.raises(ValueError, match="identifiable only"):
        BindingLossConfig(objective="multi_positive_infonce", logit_bias=1.0)


def test_density_ratio_default_retains_conservative_siglip_initialization() -> None:
    sigmoid = BindingLossConfig()
    infonce = BindingLossConfig(objective="multi_positive_infonce")

    assert sigmoid.temperature == 0.1
    assert sigmoid.effective_logit_bias == -2.71
    assert infonce.effective_logit_bias == 0.0


@pytest.mark.parametrize("negative_count", [1, 3, 15])
def test_sigmoid_sampling_prior_correction_leaves_neutral_llr_stationary(
    negative_count: int,
) -> None:
    bias = torch.tensor(0.0, requires_grad=True)
    logits = bias.expand(negative_count + 1)
    positive = torch.zeros_like(logits, dtype=torch.bool)
    positive[0] = True
    negative = ~positive

    loss = sigmoid_density_ratio_loss(logits, positive, negative)
    loss.backward()

    torch.testing.assert_close(bias.grad, torch.tensor(0.0), atol=1e-7, rtol=0.0)


def test_sigmoid_density_ratio_requires_both_distributions() -> None:
    logits = torch.tensor([0.2, -0.3], requires_grad=True)
    positive = torch.tensor([True, True])
    negative = torch.tensor([False, False])

    loss = sigmoid_density_ratio_loss(logits, positive, negative)
    loss.backward()

    assert loss == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_sigmoid_relation_scale_and_bias_are_learned_like_official_siglip() -> None:
    features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]]],
        requires_grad=True,
    )
    criterion = MultimodalBindingCriterion(BindingLossConfig(temperature=0.2))

    output = criterion(_projection(features), (_target(),))
    output.loss.backward()

    parameters = tuple(criterion.parameters())
    assert len(parameters) == 2
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0.0 for parameter in parameters
    )


def test_vectorized_object_modality_pool_matches_the_original_definition_and_gradient() -> None:
    features = torch.tensor(
        [
            [
                [1.0, 0.0, 0.5],
                [0.2, 0.8, 0.1],
                [0.0, 1.0, 0.3],
                [0.4, 0.1, 0.7],
                [0.7, 0.2, 0.0],
                [0.3, 0.6, 0.9],
            ]
        ],
        requires_grad=True,
    )
    valid = torch.ones(1, 6, dtype=torch.bool)
    projection = BindingProjectionOutput(
        native_banks=(),
        binding_features=features,
        token_valid=valid,
        current_measurement_valid=valid,
        token_group_id=torch.full_like(valid, -1, dtype=torch.long),
        modality_index=torch.tensor([[0, 0, 1, 1, 2, 2]]),
        spans=(),
    )
    ownership = torch.tensor(
        [
            [0.8, 0.2, 0.0],
            [0.3, 0.7, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
            [0.6, 0.4, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    supervised = torch.tensor([True, True, True, True, True, False])
    target = ObjectSetTarget(ownership, valid[0], supervised)
    minimum_mass = 0.5

    actual = pool_object_modality_views(
        projection,
        target,
        0,
        minimum_mass=minimum_mass,
    )

    expected_embeddings = []
    expected_objects = []
    expected_modalities = []
    for object_index in range(target.num_objects):
        for modality_index in torch.unique(projection.modality_index[0, supervised]).tolist():
            selected = supervised & (projection.modality_index[0] == modality_index)
            weight = ownership[:, object_index] * selected
            mass = weight.sum()
            if mass.detach().item() < minimum_mass:
                continue
            expected_embeddings.append((features[0] * weight.unsqueeze(-1)).sum(dim=0) / mass)
            expected_objects.append(object_index)
            expected_modalities.append(modality_index)

    expected = torch.stack(expected_embeddings)
    torch.testing.assert_close(actual.embeddings, expected)
    assert actual.object_index.tolist() == expected_objects
    assert actual.modality_index.tolist() == expected_modalities
    assert int(actual.count) == len(expected_embeddings)

    actual_objective = actual.embeddings.square().sum()
    expected_objective = expected.square().sum()
    actual_gradient = torch.autograd.grad(actual_objective, features, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected_objective, features)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
