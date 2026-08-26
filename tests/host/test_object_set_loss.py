from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
discovery = pytest.importorskip("picf_next.models.discovery")
set_loss = pytest.importorskip("picf_next.models.set_loss")

ObjectDiscoveryConfig = discovery.ObjectDiscoveryConfig
ObjectDiscoveryOutput = discovery.ObjectDiscoveryOutput
ObjectExistenceCalibration = discovery.ObjectExistenceCalibration
TaskIndependentObjectDiscovery = discovery.TaskIndependentObjectDiscovery
ObjectSetCriterion = set_loss.ObjectSetCriterion
ObjectSetHungarianMatcher = set_loss.ObjectSetHungarianMatcher
ObjectSetLossConfig = set_loss.ObjectSetLossConfig
ObjectSetMatcherConfig = set_loss.ObjectSetMatcherConfig
ObjectSetTarget = set_loss.ObjectSetTarget
GEOMETRY = synthetic_geometry_contract(2)


def _target(permutation: torch.Tensor | None = None) -> ObjectSetTarget:
    ownership = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    address = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    content = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    geometry = torch.tensor([[0.2, 0.3], [0.7, 0.8]])
    if permutation is not None:
        ownership = torch.cat((ownership[:, :-1][:, permutation], ownership[:, -1:]), dim=1)
        address = address[permutation]
        content = content[permutation]
        geometry = geometry[permutation]
    return ObjectSetTarget(
        ownership=ownership,
        token_valid=torch.tensor([True, True, True, True, True, False]),
        object_inventory_complete=True,
        address=address,
        content=content,
        geometry=geometry,
        geometry_contract=GEOMETRY,
    )


def _output(ownership_logits: torch.Tensor | None = None) -> ObjectDiscoveryOutput:
    if ownership_logits is None:
        ownership_logits = torch.tensor(
            [
                [9.0, -9.0, -9.0, -9.0],
                [9.0, -9.0, -9.0, -9.0],
                [-9.0, 9.0, -9.0, -9.0],
                [-9.0, 9.0, -9.0, -9.0],
                [-9.0, -9.0, -9.0, 9.0],
                [-torch.inf, -torch.inf, -torch.inf, 0.0],
            ]
        ).unsqueeze(0)
    token_valid = torch.tensor([[True, True, True, True, True, False]])
    address = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]]])
    content = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [3.0, 3.0]]])
    geometry = torch.tensor([[[0.2, 0.3], [0.7, 0.8], [3.0, 3.0]]])
    variance = torch.full((1, 3, 2), 0.1)
    return ObjectDiscoveryOutput(
        query_features=torch.zeros(1, 3, 4),
        address_mean=address,
        content_mean=content,
        geometry_mean=geometry,
        geometry_variance=variance,
        geometry_contract=GEOMETRY,
        existence_logits=torch.tensor([[9.0, 9.0, -9.0]]),
        localization_confidence_logits=torch.tensor([[9.0, 9.0, -9.0]]),
        ownership_logits=ownership_logits,
        ownership=torch.softmax(ownership_logits, dim=-1),
        token_valid=token_valid,
        token_group_id=torch.full((1, 6), -1, dtype=torch.long),
        evidence_available=torch.tensor([True]),
        existence_calibration=ObjectExistenceCalibration(),
    )


def _criterion() -> ObjectSetCriterion:
    return ObjectSetCriterion(
        ObjectSetHungarianMatcher(
            ObjectSetMatcherConfig(
                existence_cost=1.0,
                ownership_ce_cost=1.0,
                ownership_dice_cost=1.0,
                address_cost=0.2,
                content_cost=0.2,
                geometry_cost=0.2,
            )
        ),
        ObjectSetLossConfig(
            address_cosine_weight=0.1,
            content_cosine_weight=0.1,
            geometry_weight=0.1,
        ),
    )


def _permute_queries(output: ObjectDiscoveryOutput, permutation: torch.Tensor):
    object_logits = output.ownership_logits[..., :-1][..., permutation]
    logits = torch.cat((object_logits, output.ownership_logits[..., -1:]), dim=-1)
    return ObjectDiscoveryOutput(
        query_features=output.query_features[:, permutation],
        address_mean=output.address_mean[:, permutation],
        content_mean=output.content_mean[:, permutation],
        geometry_mean=output.geometry_mean[:, permutation],
        geometry_variance=output.geometry_variance[:, permutation],
        geometry_contract=output.geometry_contract,
        existence_logits=output.existence_logits[:, permutation],
        localization_confidence_logits=output.localization_confidence_logits[:, permutation],
        ownership_logits=logits,
        ownership=torch.softmax(logits, dim=-1),
        token_valid=output.token_valid,
        token_group_id=output.token_group_id,
        evidence_available=output.evidence_available,
        existence_calibration=output.existence_calibration,
    )


def test_perfect_prediction_matches_objects_and_has_small_loss() -> None:
    result = _criterion()(_output(), (_target(),))

    assert result.matches[0].prediction_indices.tolist() == [0, 1]
    assert result.matches[0].target_indices.tolist() == [0, 1]
    assert result.losses["loss_existence"] < 2e-4
    assert result.losses["loss_localization_confidence"] < 2e-4
    assert result.losses["loss_ownership_ce"] < 2e-4
    assert result.losses["loss_ownership_dice"] < 2e-4
    assert torch.isfinite(result.total)


def test_bfloat16_matching_and_losses_use_float32_control_arithmetic() -> None:
    output = _output()
    target = _target()
    floating_output_fields = (
        "query_features",
        "address_mean",
        "content_mean",
        "geometry_mean",
        "geometry_variance",
        "existence_logits",
        "localization_confidence_logits",
        "ownership_logits",
        "ownership",
    )
    floating_target_fields = ("ownership", "address", "content", "geometry")

    quantized_output = replace(
        output,
        **{name: getattr(output, name).to(torch.bfloat16) for name in floating_output_fields},
    )
    quantized_target = replace(
        target,
        **{name: getattr(target, name).to(torch.bfloat16) for name in floating_target_fields},
    )
    float32_reference = replace(
        quantized_output,
        **{name: getattr(quantized_output, name).float() for name in floating_output_fields},
    )
    float32_target = replace(
        quantized_target,
        **{name: getattr(quantized_target, name).float() for name in floating_target_fields},
    )

    expected = _criterion()(float32_reference, (float32_target,))
    actual = _criterion()(quantized_output, (quantized_target,))

    assert torch.equal(
        actual.matches[0].prediction_indices,
        expected.matches[0].prediction_indices,
    )
    assert torch.equal(actual.matches[0].target_indices, expected.matches[0].target_indices)
    for name, expected_loss in expected.losses.items():
        assert actual.losses[name].dtype == torch.float32
        torch.testing.assert_close(actual.losses[name], expected_loss, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("field", ["ownership", "address", "content", "geometry"])
def test_set_criterion_rejects_differentiable_loss_targets(field: str) -> None:
    target = _target()
    values = {
        "ownership": target.ownership,
        "address": target.address,
        "content": target.content,
        "geometry": target.geometry,
    }
    values[field] = values[field].clone().requires_grad_(True)
    differentiable = ObjectSetTarget(
        ownership=values["ownership"],
        token_valid=target.token_valid,
        object_inventory_complete=target.object_inventory_complete,
        address=values["address"],
        content=values["content"],
        geometry=values["geometry"],
        geometry_contract=target.geometry_contract,
    )

    with pytest.raises(ValueError, match="loss-only target"):
        _criterion()(_output(), (differentiable,))


def test_target_object_permutation_does_not_change_loss() -> None:
    criterion = _criterion()
    expected = criterion(_output(), (_target(),)).losses
    actual = criterion(_output(), (_target(torch.tensor([1, 0])),)).losses

    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_query_permutation_does_not_change_loss() -> None:
    criterion = _criterion()
    output = _output()
    permutation = torch.tensor([2, 0, 1])
    expected = criterion(output, (_target(),)).losses
    actual = criterion(_permute_queries(output, permutation), (_target(),)).losses

    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_localization_confidence_learns_detached_matched_soft_iou() -> None:
    base = _output()
    ownership_logits = base.ownership_logits.detach().clone().requires_grad_(True)
    confidence_logits = torch.zeros_like(
        base.localization_confidence_logits,
        requires_grad=True,
    )
    output = replace(
        base,
        localization_confidence_logits=confidence_logits,
        ownership_logits=ownership_logits,
        ownership=torch.softmax(ownership_logits, dim=-1),
    )
    criterion = ObjectSetCriterion(
        config=ObjectSetLossConfig(
            existence_weight=0.0,
            localization_confidence_weight=1.0,
            ownership_ce_weight=0.0,
            ownership_dice_weight=0.0,
            geometry_weight=0.0,
        )
    )

    result = criterion(output, (_target(),))
    result.total.backward()

    # The perfect masks have a soft-IoU target near one, so a neutral quality
    # logit receives a negative gradient. Ownership has its own objective and
    # cannot improve the detached label used by this confidence loss.
    matched = result.matches[0].prediction_indices
    assert (confidence_logits.grad[0, matched] < 0.0).all()
    assert ownership_logits.grad is not None
    torch.testing.assert_close(ownership_logits.grad, torch.zeros_like(ownership_logits.grad))


def test_localization_confidence_cannot_change_hungarian_matching() -> None:
    output = _output()
    expected = ObjectSetHungarianMatcher()(output, (_target(),))
    changed = replace(
        output,
        localization_confidence_logits=torch.tensor([[100.0, -100.0, 50.0]]),
    )
    actual = ObjectSetHungarianMatcher()(changed, (_target(),))

    assert torch.equal(actual[0].prediction_indices, expected[0].prediction_indices)
    assert torch.equal(actual[0].target_indices, expected[0].target_indices)


def test_ownership_ce_is_the_balanced_simplex_marginal_composite_score() -> None:
    output = _output()
    target = _target()
    result = ObjectSetCriterion()(output, (target,))
    match = result.matches[0]
    valid = target.supervision_valid
    remapped = torch.zeros(int(valid.sum()), output.ownership.shape[-1])
    remapped[:, -1] = target.ownership[valid, -1]
    remapped[:, match.prediction_indices] = target.ownership[valid][:, match.target_indices]
    probability = output.ownership[0, valid].clamp(min=1e-6, max=1.0 - 1e-6)
    positive_mass = remapped.sum(dim=0)
    negative_target = 1.0 - remapped
    positive = -(remapped * probability.log()).sum(dim=0) / positive_mass.clamp_min(1e-6)
    negative = -(negative_target * torch.log1p(-probability)).sum(dim=0) / negative_target.sum(
        dim=0
    ).clamp_min(1e-6)
    active = positive_mass > 0.0
    expected = (0.5 * (positive + negative))[active].mean()

    torch.testing.assert_close(result.losses["loss_ownership_ce"], expected)


def test_ownership_composite_score_keeps_one_simplex_per_token() -> None:
    output = _output()
    target = _target()
    result = ObjectSetCriterion()(output, (target,))

    torch.testing.assert_close(
        output.ownership.sum(dim=-1),
        torch.ones_like(output.ownership[..., 0]),
    )
    assert torch.isfinite(result.losses["loss_ownership_ce"])


def test_geometry_loss_and_matching_ignore_unsupervised_coordinates() -> None:
    base_target = _target()
    target = replace(
        base_target,
        geometry=torch.tensor([[0.2, 0.0], [0.0, 0.8]]),
        geometry_supervised=torch.tensor([[True, False], [False, True]]),
    )
    output = _output()
    perturbed_geometry = output.geometry_mean.clone()
    perturbed_geometry[0, 0, 1] = 1000.0
    perturbed_geometry[0, 1, 0] = -1000.0
    perturbed = replace(output, geometry_mean=perturbed_geometry)

    expected = _criterion()(output, (target,))
    actual = _criterion()(perturbed, (target,))

    assert torch.equal(
        actual.matches[0].prediction_indices,
        expected.matches[0].prediction_indices,
    )
    assert torch.equal(
        actual.matches[0].target_indices,
        expected.matches[0].target_indices,
    )
    torch.testing.assert_close(
        actual.losses["loss_geometry"],
        expected.losses["loss_geometry"],
    )
    torch.testing.assert_close(actual.total, expected.total)


def test_unsupervised_geometry_coordinates_must_be_zero() -> None:
    target = replace(
        _target(),
        geometry_supervised=torch.tensor([[True, False], [True, True]]),
    )

    with pytest.raises(ValueError, match="exactly zero"):
        _criterion()(_output(), (target,))


def test_geometry_loss_adds_calibrated_target_measurement_variance() -> None:
    output = _output()
    target = replace(
        _target(),
        geometry=torch.tensor([[1.2, 1.3], [1.7, 1.8]]),
        geometry_variance=torch.full((2, 2), 0.4),
    )
    result = _criterion()(output, (target,))
    robust_mean = torch.nn.functional.smooth_l1_loss(
        output.geometry_mean[0, :2],
        target.geometry,
        beta=1.0,
        reduction="none",
    )
    squared_residual = (output.geometry_mean[0, :2] - target.geometry).square()
    variance = output.geometry_variance[0, :2] + target.geometry_variance
    expected = (robust_mean + 0.5 * (squared_residual / variance + variance.log())).mean()

    torch.testing.assert_close(result.losses["loss_geometry"], expected)


@pytest.mark.parametrize(
    "variance",
    [
        torch.tensor([[-0.1, 0.0], [0.0, 0.0]]),
        torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]),
    ],
)
def test_geometry_target_variance_must_be_finite_and_nonnegative(
    variance: torch.Tensor,
) -> None:
    target = replace(_target(), geometry_variance=variance)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        _criterion()(_output(), (target,))


def test_unsupervised_geometry_variance_must_be_zero() -> None:
    target = replace(
        _target(),
        geometry=torch.tensor([[0.2, 0.0], [0.7, 0.8]]),
        geometry_variance=torch.tensor([[0.1, 0.1], [0.1, 0.1]]),
        geometry_supervised=torch.tensor([[True, False], [True, True]]),
    )

    with pytest.raises(ValueError, match="variance must be exactly zero"):
        _criterion()(_output(), (target,))


def test_set_loss_rejects_same_width_but_different_geometry_contract() -> None:
    target = replace(
        _target(),
        geometry_contract=synthetic_geometry_contract(
            2,
            name="picf.same-width-other-position.v1",
        ),
    )

    with pytest.raises(ValueError, match="contracts differ"):
        _criterion()(_output(), (target,))


def test_duplicate_wrong_ownership_is_worse_than_separated_objects() -> None:
    bad_logits = torch.tensor(
        [
            [9.0, -9.0, -9.0, -9.0],
            [9.0, -9.0, -9.0, -9.0],
            [9.0, -9.0, -9.0, -9.0],
            [9.0, -9.0, -9.0, -9.0],
            [-9.0, -9.0, -9.0, 9.0],
            [-torch.inf, -torch.inf, -torch.inf, 0.0],
        ]
    ).unsqueeze(0)
    criterion = _criterion()
    good = criterion(_output(), (_target(),)).total
    bad = criterion(_output(bad_logits), (_target(),)).total

    assert bad > good + 1.0


def test_empty_object_set_trains_every_query_as_non_object() -> None:
    output = _output()
    target = ObjectSetTarget(
        ownership=torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0], [0.0]]),
        token_valid=output.token_valid[0],
        object_inventory_complete=True,
    )
    result = ObjectSetCriterion()(output, (target,))

    assert result.matches[0].prediction_indices.numel() == 0
    assert result.losses["loss_existence"] > 1.0
    assert torch.isfinite(result.total)


def test_partial_object_inventory_does_not_create_unmatched_negative_labels() -> None:
    base = _output()
    logits = base.existence_logits.detach().clone().requires_grad_(True)
    output = ObjectDiscoveryOutput(
        query_features=base.query_features,
        address_mean=base.address_mean,
        content_mean=base.content_mean,
        geometry_mean=base.geometry_mean,
        geometry_variance=base.geometry_variance,
        geometry_contract=base.geometry_contract,
        existence_logits=logits,
        localization_confidence_logits=base.localization_confidence_logits,
        ownership_logits=base.ownership_logits,
        ownership=base.ownership,
        token_valid=base.token_valid,
        token_group_id=base.token_group_id,
        evidence_available=base.evidence_available,
        existence_calibration=base.existence_calibration,
    )
    complete = _target()
    partial = ObjectSetTarget(
        ownership=complete.ownership,
        token_valid=complete.token_valid,
        object_inventory_complete=False,
        address=complete.address,
        content=complete.content,
        geometry=complete.geometry,
        geometry_contract=complete.geometry_contract,
    )
    criterion = ObjectSetCriterion(
        ObjectSetHungarianMatcher(
            ObjectSetMatcherConfig(
                existence_cost=0.0,
                ownership_ce_cost=1.0,
                ownership_dice_cost=1.0,
                geometry_cost=0.0,
            )
        )
    )

    result = criterion(output, (partial,))
    result.total.backward()
    matched = set(result.matches[0].prediction_indices.tolist())
    unmatched = ({0, 1, 2} - matched).pop()

    assert logits.grad is not None
    assert logits.grad[0, unmatched] == 0.0
    assert (logits.grad[0, list(matched)] < 0.0).all()


def test_unknown_empty_inventory_does_not_claim_an_empty_scene() -> None:
    base = _output()
    logits = base.existence_logits.detach().clone().requires_grad_(True)
    output = ObjectDiscoveryOutput(
        query_features=base.query_features,
        address_mean=base.address_mean,
        content_mean=base.content_mean,
        geometry_mean=base.geometry_mean,
        geometry_variance=base.geometry_variance,
        geometry_contract=base.geometry_contract,
        existence_logits=logits,
        localization_confidence_logits=base.localization_confidence_logits,
        ownership_logits=base.ownership_logits,
        ownership=base.ownership,
        token_valid=base.token_valid,
        token_group_id=base.token_group_id,
        evidence_available=base.evidence_available,
        existence_calibration=base.existence_calibration,
    )
    target = ObjectSetTarget(
        ownership=torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0], [0.0]]),
        token_valid=base.token_valid[0],
        object_inventory_complete=False,
    )

    result = ObjectSetCriterion()(output, (target,))
    result.total.backward()

    assert result.losses["loss_existence"] == 0.0
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_soft_overlap_and_context_target_is_accepted() -> None:
    target = _target()
    ownership = target.ownership.clone()
    ownership[1] = torch.tensor([0.45, 0.35, 0.20])
    soft = ObjectSetTarget(
        ownership=ownership,
        token_valid=target.token_valid,
        geometry=target.geometry,
        geometry_contract=target.geometry_contract,
    )

    result = ObjectSetCriterion()(_output(), (soft,))
    assert torch.isfinite(result.total)


def test_optional_state_targets_may_be_absent() -> None:
    target = _target()
    ownership_only = ObjectSetTarget(target.ownership, target.token_valid)
    result = ObjectSetCriterion()(_output(), (ownership_only,))

    assert result.losses["loss_address_cosine"] == 0.0
    assert result.losses["loss_content_cosine"] == 0.0
    assert result.losses["loss_geometry"] == 0.0


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        (("only-one",), "align"),
        (("same", "same"), "unique"),
        (("", "second"), "nonempty"),
    ],
)
def test_temporal_identity_keys_are_loss_only_well_formed_metadata(
    keys: tuple[str, ...],
    message: str,
) -> None:
    target = _target()
    malformed = ObjectSetTarget(
        ownership=target.ownership,
        token_valid=target.token_valid,
        geometry=target.geometry,
        geometry_contract=target.geometry_contract,
        temporal_identity_keys=keys,
    )

    with pytest.raises(ValueError, match=message):
        _criterion()(_output(), (malformed,))


def test_unsupervised_valid_tokens_do_not_contribute_to_ownership_loss() -> None:
    output = _output()
    target = _target()
    supervised = target.token_valid.clone()
    supervised[4] = False
    ownership = target.ownership.clone()
    ownership[4] = 0.0
    selective = ObjectSetTarget(
        ownership=ownership,
        token_valid=target.token_valid,
        token_supervised=supervised,
        address=target.address,
        content=target.content,
        geometry=target.geometry,
        geometry_contract=target.geometry_contract,
    )
    changed_logits = output.ownership_logits.clone()
    changed_logits[:, 4] = torch.tensor([50.0, -50.0, -50.0, -50.0])

    expected = _criterion()(output, (selective,)).losses
    actual = _criterion()(_output(changed_logits), (selective,)).losses

    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_targets_are_not_discovery_inputs_and_gradients_reach_discovery() -> None:
    parameters = inspect.signature(TaskIndependentObjectDiscovery.forward).parameters
    assert tuple(parameters) == ("self", "binding_features", "token_valid", "token_group_id")

    torch.manual_seed(401)
    model = TaskIndependentObjectDiscovery(
        ObjectDiscoveryConfig(
            input_dim=5,
            hidden_dim=12,
            num_queries=3,
            num_layers=2,
            num_heads=3,
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            initial_variance=0.1,
        )
    ).train()
    valid = _target().token_valid.unsqueeze(0)
    features = (torch.randn(1, 6, 5) * valid.unsqueeze(-1)).requires_grad_(True)
    output = model(features, valid)
    result = _criterion()(output, (_target(),))
    result.total.backward()

    assert features.grad is not None
    assert model.query_embeddings.grad is not None
    assert model.existence_head.weight.grad is not None
    assert model.ownership_query.weight.grad is not None
    assert model.geometry_head.weight.grad is not None


def test_rejects_target_token_misalignment_and_non_simplex_rows() -> None:
    output = _output()
    target = _target()
    with pytest.raises(ValueError, match="validity must match"):
        ObjectSetCriterion()(
            output,
            (
                ObjectSetTarget(
                    target.ownership,
                    torch.tensor([True, True, True, True, False, False]),
                ),
            ),
        )

    bad_ownership = target.ownership.clone()
    bad_ownership[0, 0] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        ObjectSetCriterion()(output, (ObjectSetTarget(bad_ownership, target.token_valid),))

    with pytest.raises(ValueError, match="object_inventory_complete"):
        ObjectSetCriterion()(
            output,
            (
                ObjectSetTarget(
                    target.ownership,
                    target.token_valid,
                    object_inventory_complete=1,  # type: ignore[arg-type]
                ),
            ),
        )


def test_unmatched_query_receives_no_object_gradient() -> None:
    logits = torch.tensor([[8.0, 8.0, 8.0]], requires_grad=True)
    output = _output()
    output = ObjectDiscoveryOutput(
        query_features=output.query_features,
        address_mean=output.address_mean,
        content_mean=output.content_mean,
        geometry_mean=output.geometry_mean,
        geometry_variance=output.geometry_variance,
        geometry_contract=output.geometry_contract,
        existence_logits=logits,
        localization_confidence_logits=output.localization_confidence_logits,
        ownership_logits=output.ownership_logits,
        ownership=output.ownership,
        token_valid=output.token_valid,
        token_group_id=output.token_group_id,
        evidence_available=output.evidence_available,
        existence_calibration=output.existence_calibration,
    )
    result = ObjectSetCriterion()(output, (_target(),))
    result.total.backward()
    unmatched = ({0, 1, 2} - set(result.matches[0].prediction_indices.tolist())).pop()

    assert logits.grad[0, unmatched] > 0.0


def test_auxiliary_decoder_predictions_receive_deep_supervision() -> None:
    base = _output()
    auxiliary_logits = base.ownership_logits.detach().clone().requires_grad_(True)
    auxiliary_existence = base.existence_logits.detach().clone().requires_grad_(True)
    auxiliary_confidence = base.localization_confidence_logits.detach().clone().requires_grad_(True)
    auxiliary = replace(
        base,
        existence_logits=auxiliary_existence,
        localization_confidence_logits=auxiliary_confidence,
        ownership_logits=auxiliary_logits,
        ownership=torch.softmax(auxiliary_logits, dim=-1),
        auxiliary_outputs=(),
    )
    output = replace(base, auxiliary_outputs=(auxiliary,))
    criterion = ObjectSetCriterion(
        config=ObjectSetLossConfig(
            address_cosine_weight=0.0,
            content_cosine_weight=0.0,
            geometry_weight=0.0,
        )
    )

    criterion(output, (_target(),)).total.backward()

    assert auxiliary_logits.grad is not None
    assert auxiliary_logits.grad.abs().sum() > 0.0
    assert auxiliary_existence.grad is not None
    assert auxiliary_existence.grad.abs().sum() > 0.0
    assert auxiliary_confidence.grad is not None
    assert auxiliary_confidence.grad.abs().sum() > 0.0


def test_geometry_scale_calibration_does_not_rescale_mean_gradient() -> None:
    base = _output()
    geometry = base.geometry_mean.detach().clone()
    geometry[0, :2] += 3.0
    geometry.requires_grad_(True)
    output = replace(base, geometry_mean=geometry)
    criterion = ObjectSetCriterion(
        config=ObjectSetLossConfig(
            existence_weight=0.0,
            ownership_ce_weight=0.0,
            ownership_dice_weight=0.0,
            geometry_weight=1.0,
        )
    )

    criterion(output, (_target(),)).total.backward()

    assert geometry.grad is not None
    assert geometry.grad.abs().max() == pytest.approx(0.25)
