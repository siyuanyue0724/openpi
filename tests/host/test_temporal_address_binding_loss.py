from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from picf_next.models.binding_loss import (  # noqa: E402
    BindingLossConfig,
    TemporalAddressBindingCriterion,
)
from picf_next.models.discovery import (  # noqa: E402
    ObjectDiscoveryOutput,
    ObjectExistenceCalibration,
)
from picf_next.models.set_loss import ObjectSetTarget, SetMatch  # noqa: E402
from tests.geometry_contract import synthetic_geometry_contract  # noqa: E402

GEOMETRY = synthetic_geometry_contract(1)


def _discovery(address: torch.Tensor) -> ObjectDiscoveryOutput:
    batch_size, queries, _ = address.shape
    token_valid = torch.ones(batch_size, 1, dtype=torch.bool)
    ownership_logits = torch.zeros(batch_size, 1, queries + 1)
    return ObjectDiscoveryOutput(
        query_features=torch.zeros(batch_size, queries, 4),
        address_mean=address,
        content_mean=torch.zeros(batch_size, queries, 1),
        geometry_mean=torch.zeros(batch_size, queries, 1),
        geometry_variance=torch.ones(batch_size, queries, 1),
        geometry_contract=GEOMETRY,
        existence_logits=torch.ones(batch_size, queries),
        localization_confidence_logits=torch.zeros(batch_size, queries),
        ownership_logits=ownership_logits,
        ownership=torch.softmax(ownership_logits, dim=-1),
        token_valid=token_valid,
        token_group_id=torch.full_like(token_valid, -1, dtype=torch.long),
        evidence_available=torch.ones(batch_size, dtype=torch.bool),
        existence_calibration=ObjectExistenceCalibration(),
    )


def _target(
    keys: tuple[str, ...] | None,
    *,
    complete_inventory: bool = False,
) -> ObjectSetTarget:
    return ObjectSetTarget(
        ownership=torch.tensor([[0.5, 0.5, 0.0]]),
        token_valid=torch.ones(1, dtype=torch.bool),
        object_inventory_complete=complete_inventory,
        temporal_identity_keys=keys,
    )


def _match(prediction_indices: tuple[int, ...] = (0, 1)) -> SetMatch:
    return SetMatch(
        prediction_indices=torch.tensor(prediction_indices),
        target_indices=torch.tensor((0, 1)),
    )


def test_same_physical_address_alignment_beats_a_temporal_identity_swap() -> None:
    criterion = TemporalAddressBindingCriterion(BindingLossConfig(temperature=0.5))
    first = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    aligned = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    swapped = _discovery(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    targets = ((_target(("track:a", "track:b")),),) * 2
    matches = ((_match(),),) * 2

    good = criterion((first, aligned), targets, matches)
    bad = criterion((first, swapped), targets, matches)

    assert good.positive_pairs == 4
    assert good.negative_pairs == 4
    assert good.loss < bad.loss


def test_temporal_binding_is_invariant_to_transient_query_permutation() -> None:
    criterion = TemporalAddressBindingCriterion(BindingLossConfig(temperature=0.5))
    first = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    second = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    permuted = _discovery(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    targets = ((_target(("track:a", "track:b")),),) * 2

    expected = criterion((first, second), targets, ((_match(),), (_match(),)))
    actual = criterion((first, permuted), targets, ((_match(),), (_match((1, 0)),)))

    torch.testing.assert_close(actual.loss, expected.loss)
    assert actual.positive_pairs == expected.positive_pairs
    assert actual.negative_pairs == expected.negative_pairs


def test_complete_inventory_null_relation_is_query_permutation_invariant() -> None:
    criterion = TemporalAddressBindingCriterion(BindingLossConfig(temperature=0.5))
    first = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]))
    second = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]))
    permuted = _discovery(torch.tensor([[[0.0, -1.0], [0.0, 1.0], [1.0, 0.0]]]))
    complete_target = _target(("track:a", "track:b"), complete_inventory=True)

    expected = criterion(
        (first, second),
        ((complete_target,), (complete_target,)),
        ((_match(),), (_match(),)),
    )
    actual = criterion(
        (first, permuted),
        ((complete_target,), (complete_target,)),
        ((_match(),), (_match((2, 1)),)),
    )

    torch.testing.assert_close(actual.loss, expected.loss)
    assert actual.null_address_views == expected.null_address_views
    assert actual.positive_pairs == expected.positive_pairs
    assert actual.negative_pairs == expected.negative_pairs
    assert actual.null_negative_pairs == expected.null_negative_pairs


def test_temporal_binding_skips_unknown_correspondence_and_backpropagates_known_pairs() -> None:
    criterion = TemporalAddressBindingCriterion()
    first_address = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    second_address = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]], requires_grad=True)
    first = _discovery(first_address)
    second = _discovery(second_address)

    unknown = criterion(
        (first, second),
        ((_target(None),), (_target(None),)),
        ((_match(),), (_match(),)),
    )
    assert unknown.positive_pairs == 0
    assert unknown.negative_pairs == 0
    assert unknown.loss == 0.0

    known = criterion(
        (first, second),
        ((_target(("track:a", "track:b")),),) * 2,
        ((_match(),), (_match(),)),
    )
    known.loss.backward()
    assert first_address.grad is not None and first_address.grad.abs().sum() > 0.0
    assert second_address.grad is not None and second_address.grad.abs().sum() > 0.0


def test_positive_only_temporal_graph_is_reported_but_not_claimed_as_covered() -> None:
    criterion = TemporalAddressBindingCriterion()
    first = _discovery(torch.tensor([[[1.0, 0.0]]], requires_grad=True))
    second = _discovery(torch.tensor([[[0.9, 0.1]]], requires_grad=True))
    single_target = ObjectSetTarget(
        ownership=torch.tensor([[1.0, 0.0]]),
        token_valid=torch.ones(1, dtype=torch.bool),
        temporal_identity_keys=("track:a",),
    )
    single_match = SetMatch(
        prediction_indices=torch.tensor([0]),
        target_indices=torch.tensor([0]),
    )

    result = criterion(
        (first, second),
        ((single_target,), (single_target,)),
        ((single_match,), (single_match,)),
    )

    assert result.eligible_samples == 1
    assert result.covered_eligible_samples == 0
    assert result.positive_pairs == 2
    assert result.negative_pairs == 0
    assert result.loss == 0.0


def test_checkpointed_prior_track_enables_one_transition_credit_without_history_gradient() -> None:
    criterion = TemporalAddressBindingCriterion(BindingLossConfig(temperature=0.5))
    prior_address = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
        requires_grad=True,
    )
    current_address = torch.tensor(
        [[[0.9, 0.1], [0.1, 0.9]]],
        requires_grad=True,
    )

    result = criterion(
        (_discovery(current_address),),
        ((_target(("track:a", "track:b")),),),
        ((_match(),),),
        initial_address=prior_address,
        initial_valid=torch.tensor([[True, True, True]]),
        initial_identity_keys_by_row=(("track:a", "track:b", None),),
    )

    assert result.address_views == 4
    assert result.eligible_samples == 1
    assert result.covered_eligible_samples == 1
    assert result.positive_pairs == 4
    assert result.negative_pairs == 4
    result.loss.backward()
    assert prior_address.grad is None
    assert current_address.grad is not None and current_address.grad.abs().sum() > 0.0


def test_complete_inventory_adds_unmatched_queries_as_temporal_null_negatives() -> None:
    criterion = TemporalAddressBindingCriterion(BindingLossConfig(temperature=0.5))
    prior_address = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    current_address = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    discovery = _discovery(current_address)
    orthogonal_null = _discovery(
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    )
    initial = {
        "initial_address": prior_address,
        "initial_valid": torch.tensor([[True, True]]),
        "initial_identity_keys_by_row": (("track:a", "track:b"),),
    }

    partial = criterion(
        (discovery,),
        ((_target(("track:a", "track:b")),),),
        ((_match(),),),
        **initial,
    )
    complete = criterion(
        (discovery,),
        ((_target(("track:a", "track:b"), complete_inventory=True),),),
        ((_match(),),),
        **initial,
    )
    calibrated_null = criterion(
        (orthogonal_null,),
        ((_target(("track:a", "track:b"), complete_inventory=True),),),
        ((_match(),),),
        **initial,
    )

    assert partial.address_views == 4
    assert partial.null_address_views == 0
    assert partial.positive_pairs == 4
    assert partial.negative_pairs == 4
    assert partial.null_negative_pairs == 0
    assert complete.address_views == 5
    assert complete.null_address_views == 1
    assert complete.positive_pairs == 4
    assert complete.negative_pairs == 8
    assert complete.null_negative_pairs == 4
    assert complete.loss > calibrated_null.loss
    complete.loss.backward()
    assert current_address.grad is not None
    assert current_address.grad[0, 2].abs().sum() > 0.0


def test_temporal_relation_uses_runtime_calibration_and_trains_both_parameters() -> None:
    criterion = TemporalAddressBindingCriterion(BindingLossConfig(temperature=0.5))
    prior_address = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    current_address = torch.tensor(
        [[[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]],
        requires_grad=True,
    )
    logit_scale = torch.tensor(2.0, requires_grad=True)
    logit_bias = torch.tensor(-1.0, requires_grad=True)

    result = criterion(
        (_discovery(current_address),),
        ((_target(("track:a", "track:b"), complete_inventory=True),),),
        ((_match(),),),
        initial_address=prior_address,
        initial_valid=torch.tensor([[True, True]]),
        initial_identity_keys_by_row=(("track:a", "track:b"),),
        relation_logit_scale=logit_scale,
        relation_logit_bias=logit_bias,
    )
    result.loss.backward()

    assert logit_scale.grad is not None and logit_scale.grad.abs() > 0.0
    assert logit_bias.grad is not None and logit_bias.grad.abs() > 0.0
    assert current_address.grad is not None and current_address.grad.abs().sum() > 0.0


def test_temporal_binding_excludes_null_null_relations() -> None:
    criterion = TemporalAddressBindingCriterion(BindingLossConfig(temperature=0.5))
    first = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]))
    second = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]))
    complete_target = _target(("track:a", "track:b"), complete_inventory=True)

    result = criterion(
        (first, second),
        ((complete_target,), (complete_target,)),
        ((_match(),), (_match(),)),
    )

    assert result.address_views == 6
    assert result.null_address_views == 2
    assert result.positive_pairs == 4
    assert result.negative_pairs == 12
    assert result.null_negative_pairs == 8


def test_temporal_relation_eligibility_is_independent_of_prediction_matching() -> None:
    criterion = TemporalAddressBindingCriterion()
    discovery = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    empty_match = SetMatch(
        prediction_indices=torch.empty(0, dtype=torch.long),
        target_indices=torch.empty(0, dtype=torch.long),
    )

    result = criterion(
        (discovery,),
        ((_target(("track:a", "track:b")),),),
        ((empty_match,),),
        initial_address=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        initial_valid=torch.tensor([[True, True]]),
        initial_identity_keys_by_row=(("track:a", "track:b"),),
    )

    assert result.eligible_samples == 1
    assert result.covered_eligible_samples == 0
    assert result.positive_pairs == 0


def test_checkpointed_prior_track_inputs_fail_closed_as_one_atomic_contract() -> None:
    criterion = TemporalAddressBindingCriterion()
    discovery = _discovery(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    targets = ((_target(("track:a", "track:b")),),)
    matches = ((_match(),),)

    with pytest.raises(ValueError, match="are atomic"):
        criterion(
            (discovery,),
            targets,
            matches,
            initial_address=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        )
    with pytest.raises(ValueError, match="cannot name unused posterior rows"):
        criterion(
            (discovery,),
            targets,
            matches,
            initial_address=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            initial_valid=torch.tensor([[True, False]]),
            initial_identity_keys_by_row=(("track:a", "track:b"),),
        )
