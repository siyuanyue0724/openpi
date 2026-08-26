from __future__ import annotations

import random

import pytest
import torch

from picf_next.lingbot_native.objective import (
    NativeObjectiveConfig,
    build_native_predictive_normalization_ledger,
    combine_native_objective,
    combine_native_sequential_branch,
    merge_repeated_objective_terms,
)
from picf_next.objective import ObjectiveTerm


def _term(name: str, values: torch.Tensor, valid: torch.Tensor, weight: float) -> ObjectiveTerm:
    return ObjectiveTerm(name=name, values=values, valid=valid, weight=weight)


def test_objective_term_normalizes_by_observed_mass_and_preserves_gradient_scale() -> None:
    values = torch.tensor([1.0, 3.0, 99.0], requires_grad=True)
    term = ObjectiveTerm(
        name="set/ownership",
        values=values,
        valid=torch.tensor([True, True, False]),
        weight=1.0,
        sample_weight=torch.tensor([0.25, 0.75, 0.0]),
    )

    normalized = term.normalized()

    torch.testing.assert_close(normalized, torch.tensor(2.5))
    normalized.backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.25, 0.75, 0.0]))


def test_objective_term_rejects_trainable_or_invalid_sample_weights() -> None:
    common = {
        "name": "set/ownership",
        "values": torch.ones(2),
        "valid": torch.tensor([True, False]),
        "weight": 1.0,
    }
    with pytest.raises(ValueError, match="detached"):
        ObjectiveTerm(
            **common,
            sample_weight=torch.tensor([1.0, 0.0], requires_grad=True),
        )
    with pytest.raises(ValueError, match="zero when invalid"):
        ObjectiveTerm(**common, sample_weight=torch.ones(2))


def test_sequential_two_branch_objective_matches_joint_value_and_gradient() -> None:
    factual_action = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
    omitted_action = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
    factual_values = torch.tensor([1.0, 9.0], dtype=torch.float64, requires_grad=True)
    omitted_values = torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64, requires_grad=True)
    correction_values = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)
    structural_value = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    factual_terms = (
        _term(
            "xmod/vision/binding",
            factual_values,
            torch.tensor([True, False]),
            2.0,
        ),
        _term(
            "correction/vision/binding",
            correction_values,
            torch.tensor([True, True]),
            1.0,
        ),
    )
    omitted_terms = (
        _term(
            "xmod/vision/binding",
            omitted_values,
            torch.tensor([True, True, True]),
            2.0,
        ),
    )
    structural_terms = (
        _term("set/frame_000/entities", structural_value, torch.tensor([True]), 1.0),
    )
    config = NativeObjectiveConfig(
        action_weight=1.3,
        predictive_weight=0.7,
        structural_weight=0.4,
    )
    joint = combine_native_objective(
        official_policy_loss=torch.stack((factual_action, omitted_action)).mean(),
        predictive_terms=merge_repeated_objective_terms(
            (*factual_terms, *omitted_terms)
        ),
        structural_terms=structural_terms,
        config=config,
    )
    ledger = build_native_predictive_normalization_ledger(
        tuple(term.support() for term in (*factual_terms, *omitted_terms))
    )
    factual = combine_native_sequential_branch(
        official_policy_loss=factual_action,
        action_scale=0.5,
        predictive_terms=factual_terms,
        structural_terms=structural_terms,
        predictive_ledger=ledger,
        config=config,
    )
    omitted = combine_native_sequential_branch(
        official_policy_loss=omitted_action,
        action_scale=0.5,
        predictive_terms=omitted_terms,
        structural_terms=(),
        predictive_ledger=ledger,
        config=config,
    )
    sequential_total = factual.total + omitted.total
    leaves = (
        factual_action,
        omitted_action,
        factual_values,
        omitted_values,
        correction_values,
        structural_value,
    )
    joint_gradients = torch.autograd.grad(joint.total, leaves, retain_graph=True)
    sequential_gradients = torch.autograd.grad(sequential_total, leaves)

    torch.testing.assert_close(sequential_total, joint.total)
    for sequential_gradient, joint_gradient in zip(
        sequential_gradients,
        joint_gradients,
        strict=True,
    ):
        torch.testing.assert_close(sequential_gradient, joint_gradient)


def test_sequential_zero_support_route_preserves_zero_gradient_connectivity() -> None:
    action = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    active_values = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    inactive_values = torch.tensor([7.0], dtype=torch.float64, requires_grad=True)
    terms = (
        _term(
            "xmod/vision/binding",
            active_values,
            torch.tensor([True]),
            1.0,
        ),
        _term(
            "xmod/touch/binding",
            inactive_values,
            torch.tensor([False]),
            1.0,
        ),
    )
    ledger = build_native_predictive_normalization_ledger(
        tuple(term.support() for term in terms)
    )

    branch = combine_native_sequential_branch(
        official_policy_loss=action,
        action_scale=0.5,
        predictive_terms=terms,
        structural_terms=(),
        predictive_ledger=ledger,
        config=NativeObjectiveConfig(
            action_weight=1.0,
            predictive_weight=1.0,
            structural_weight=0.0,
        ),
    )
    branch.total.backward()

    torch.testing.assert_close(inactive_values.grad, torch.zeros_like(inactive_values))
    torch.testing.assert_close(active_values.grad, torch.ones_like(active_values))


def test_sequential_all_zero_support_preserves_every_declared_route_graph() -> None:
    action = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    first = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([7.0], dtype=torch.float64, requires_grad=True)
    terms = (
        _term("xmod/vision/binding", first, torch.tensor([False]), 1.0),
        _term("xmod/touch/binding", second, torch.tensor([False]), 1.0),
    )
    ledger = build_native_predictive_normalization_ledger(
        tuple(term.support() for term in terms)
    )

    branch = combine_native_sequential_branch(
        official_policy_loss=action,
        action_scale=0.5,
        predictive_terms=terms,
        structural_terms=(),
        predictive_ledger=ledger,
        config=NativeObjectiveConfig(
            action_weight=1.0,
            predictive_weight=1.0,
            structural_weight=0.0,
        ),
    )
    branch.total.backward()

    torch.testing.assert_close(first.grad, torch.zeros_like(first))
    torch.testing.assert_close(second.grad, torch.zeros_like(second))


def test_zero_observed_mass_is_a_finite_zero_objective() -> None:
    values = torch.tensor([4.0], requires_grad=True)
    term = ObjectiveTerm(
        name="set/ownership",
        values=values,
        valid=torch.tensor([True]),
        weight=1.0,
        sample_weight=torch.zeros(1),
    )

    normalized = term.normalized()

    torch.testing.assert_close(normalized, torch.tensor(0.0))
    normalized.backward()
    torch.testing.assert_close(values.grad, torch.zeros(1))


def test_zero_observed_mass_does_not_dilute_an_active_family() -> None:
    result = combine_native_objective(
        official_policy_loss=torch.tensor(0.0),
        predictive_terms=(),
        structural_terms=(
            ObjectiveTerm(
                name="set/active",
                values=torch.tensor([4.0]),
                valid=torch.tensor([True]),
                weight=1.0,
            ),
            ObjectiveTerm(
                name="set/unobserved",
                values=torch.tensor([99.0]),
                valid=torch.tensor([True]),
                weight=1.0,
                sample_weight=torch.tensor([0.0]),
            ),
        ),
        config=NativeObjectiveConfig(predictive_weight=0.0, structural_weight=1.0),
    )

    torch.testing.assert_close(result.family_terms["structural"], torch.tensor(4.0))


def test_zero_weight_diagnostic_does_not_change_structural_objective() -> None:
    result = combine_native_objective(
        official_policy_loss=torch.tensor(0.5),
        predictive_terms=(),
        structural_terms=(
            ObjectiveTerm(
                name="set/ownership",
                values=torch.tensor([2.0]),
                valid=torch.tensor([True]),
                weight=1.0,
            ),
            ObjectiveTerm(
                name="set/ownership_nll",
                values=torch.tensor([999.0]),
                valid=torch.tensor([True]),
                weight=0.0,
            ),
        ),
        config=NativeObjectiveConfig(predictive_weight=0.0, structural_weight=0.25),
    )

    torch.testing.assert_close(result.family_terms["structural"], torch.tensor(0.5))
    torch.testing.assert_close(result.total, torch.tensor(1.0))


def test_native_objective_contains_exactly_three_declared_families() -> None:
    action = torch.tensor(2.0, requires_grad=True)
    predictive_values = torch.tensor([1.0, 99.0], requires_grad=True)
    structural_values = torch.tensor([3.0, 5.0], requires_grad=True)
    result = combine_native_objective(
        official_policy_loss=action,
        predictive_terms=(
            _term(
                "rollout/rgb/binding",
                predictive_values,
                torch.tensor([True, False]),
                0.5,
            ),
        ),
        structural_terms=(
            _term(
                "set/existence",
                structural_values,
                torch.tensor([True, True]),
                0.25,
            ),
        ),
        config=NativeObjectiveConfig(predictive_weight=2.0, structural_weight=4.0),
    )
    assert set(result.normalized_terms) == {
        "action",
        "rollout/rgb/binding",
        "set/existence",
    }
    assert set(result.family_terms) == {"action", "predictive", "structural"}
    torch.testing.assert_close(sum(result.family_terms.values()), result.total)
    torch.testing.assert_close(result.total, torch.tensor(20.0))
    result.total.backward()
    torch.testing.assert_close(action.grad, torch.tensor(1.0))
    torch.testing.assert_close(predictive_values.grad, torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(structural_values.grad, torch.tensor([2.0, 2.0]))


def test_representation_objective_requires_absent_action_and_preserves_gradients() -> None:
    predictive = torch.tensor([2.0], requires_grad=True)
    structural = torch.tensor([4.0], requires_grad=True)
    result = combine_native_objective(
        official_policy_loss=None,
        predictive_terms=(
            _term(
                "correction/vision/binding",
                predictive,
                torch.tensor([True]),
                1.0,
            ),
        ),
        structural_terms=(
            _term(
                "set/ownership",
                structural,
                torch.tensor([True]),
                1.0,
            ),
        ),
        config=NativeObjectiveConfig(
            action_weight=0.0,
            predictive_weight=0.5,
            structural_weight=0.25,
        ),
    )

    assert "action" not in result.normalized_terms
    torch.testing.assert_close(result.family_terms["action"], torch.tensor(0.0))
    torch.testing.assert_close(result.total, torch.tensor(2.0))
    result.total.backward()
    torch.testing.assert_close(predictive.grad, torch.tensor([0.5]))
    torch.testing.assert_close(structural.grad, torch.tensor([0.25]))


def test_action_presence_must_match_its_family_weight() -> None:
    structural = (
        _term(
            "set/ownership",
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            1.0,
        ),
    )
    with pytest.raises(ValueError, match="inactive action family"):
        combine_native_objective(
            official_policy_loss=torch.tensor(1.0),
            predictive_terms=(),
            structural_terms=structural,
            config=NativeObjectiveConfig(
                action_weight=0.0,
                predictive_weight=0.0,
                structural_weight=1.0,
            ),
        )
    with pytest.raises(ValueError, match="active action family"):
        combine_native_objective(
            official_policy_loss=None,
            predictive_terms=(),
            structural_terms=structural,
            config=NativeObjectiveConfig(
                action_weight=1.0,
                predictive_weight=0.0,
                structural_weight=1.0,
            ),
        )


def test_native_objective_rejects_an_empty_representation_objective() -> None:
    with pytest.raises(ValueError, match="at least one active loss term"):
        combine_native_objective(
            official_policy_loss=None,
            predictive_terms=(),
            structural_terms=(),
            config=NativeObjectiveConfig(
                action_weight=0.0,
                predictive_weight=1.0,
                structural_weight=1.0,
            ),
        )


def test_family_scale_is_invariant_to_duplicate_active_routes() -> None:
    action = torch.tensor(0.5)
    single = combine_native_objective(
        official_policy_loss=action,
        predictive_terms=(
            _term("correction/vision/binding", torch.tensor([2.0]), torch.tensor([True]), 1.0),
        ),
        structural_terms=(),
        config=NativeObjectiveConfig(predictive_weight=0.25, structural_weight=0.0),
    )
    duplicated = combine_native_objective(
        official_policy_loss=action,
        predictive_terms=(
            _term("correction/vision/binding", torch.tensor([2.0]), torch.tensor([True]), 1.0),
            _term("rollout/touch/binding", torch.tensor([2.0]), torch.tensor([True]), 1.0),
        ),
        structural_terms=(),
        config=NativeObjectiveConfig(predictive_weight=0.25, structural_weight=0.0),
    )
    torch.testing.assert_close(single.total, torch.tensor(1.0))
    torch.testing.assert_close(duplicated.total, single.total)


def test_zero_valid_component_does_not_dilute_active_family() -> None:
    action = torch.tensor(0.5)
    result = combine_native_objective(
        official_policy_loss=action,
        predictive_terms=(
            _term("correction/vision/binding", torch.tensor([2.0]), torch.tensor([True]), 1.0),
            _term("xmod/touch/binding", torch.tensor([99.0]), torch.tensor([False]), 3.0),
        ),
        structural_terms=(),
        config=NativeObjectiveConfig(predictive_weight=0.25, structural_weight=0.0),
    )
    torch.testing.assert_close(result.total, torch.tensor(1.0))


def test_component_weights_are_relative_inside_each_family() -> None:
    result = combine_native_objective(
        official_policy_loss=torch.tensor(0.0),
        predictive_terms=(
            _term("correction/vision/binding", torch.tensor([1.0]), torch.tensor([True]), 1.0),
            _term("rollout/touch/binding", torch.tensor([3.0]), torch.tensor([True]), 3.0),
        ),
        structural_terms=(),
        config=NativeObjectiveConfig(predictive_weight=2.0, structural_weight=0.0),
    )
    torch.testing.assert_close(result.total, torch.tensor(5.0))


def test_active_weighted_family_mean_matches_reference_under_random_missingness() -> None:
    """Property probe for route count, component weights and missing factors."""

    generator = random.Random(20260723)
    for case in range(128):
        terms = []
        numerator = 0.0
        denominator = 0.0
        for route in range(generator.randint(1, 6)):
            values = torch.tensor(
                [generator.uniform(0.0, 5.0) for _ in range(5)],
                dtype=torch.float64,
            )
            valid = torch.tensor(
                [generator.random() < 0.6 for _ in range(5)],
                dtype=torch.bool,
            )
            weight = generator.uniform(0.0, 3.0)
            terms.append(_term(f"xmod/modality_{case}_{route}/binding", values, valid, weight))
            if valid.any():
                numerator += float(values[valid].mean()) * weight
                denominator += weight
        action = torch.tensor(generator.uniform(0.0, 1.0), dtype=torch.float64)
        family_weight = generator.uniform(0.0, 2.0)
        result = combine_native_objective(
            official_policy_loss=action,
            predictive_terms=tuple(terms),
            structural_terms=(),
            config=NativeObjectiveConfig(
                predictive_weight=family_weight,
                structural_weight=0.0,
            ),
        )
        expected_family = 0.0 if denominator == 0.0 else numerator / denominator
        expected = action + family_weight * expected_family
        torch.testing.assert_close(result.total, expected)


def test_missing_optional_target_has_zero_count_and_no_denominator() -> None:
    action = torch.tensor(0.75, requires_grad=True)
    absent = torch.tensor([0.0], requires_grad=True)
    result = combine_native_objective(
        official_policy_loss=action,
        predictive_terms=(
            _term(
                "xmod/touch/binding",
                absent,
                torch.tensor([False]),
                1.0,
            ),
        ),
        structural_terms=(),
        config=NativeObjectiveConfig(predictive_weight=1.0, structural_weight=1.0),
    )
    assert result.valid_counts["xmod/touch/binding"] == 0
    torch.testing.assert_close(result.total, action)
    result.total.backward()
    torch.testing.assert_close(absent.grad, torch.tensor([0.0]))


@pytest.mark.parametrize("name", ["host/router", "over/horizon", "lifecycle/survival"])
def test_native_objective_rejects_undeclared_predictive_families(name: str) -> None:
    term = _term(name, torch.ones(1), torch.ones(1, dtype=torch.bool), 1.0)
    with pytest.raises(
        ValueError,
        match="xmod/, correction/, filter_prior/, filter_posterior/, or rollout/",
    ):
        combine_native_objective(
            official_policy_loss=torch.tensor(1.0),
            predictive_terms=(term,),
            structural_terms=(),
            config=NativeObjectiveConfig(1.0, 1.0),
        )


def test_native_objective_requires_the_official_scalar_policy_loss() -> None:
    with pytest.raises(ValueError, match="policy loss"):
        combine_native_objective(
            official_policy_loss=torch.ones(2),
            predictive_terms=(),
            structural_terms=(),
            config=NativeObjectiveConfig(1.0, 1.0),
        )
