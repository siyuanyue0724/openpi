from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from picf_next.models.marginal import (  # noqa: E402
    equal_weight_covariance_intersection,
    marginal_one_to_one_association,
    probabilistic_data_association_moments,
)


def _exact_partial_matching_marginals(edge_odds: np.ndarray) -> np.ndarray:
    prior_count, observation_count = edge_odds.shape
    assignments: list[tuple[tuple[int, ...], float]] = []

    def visit(row: int, used: frozenset[int], selected: tuple[int, ...], weight: float) -> None:
        if row == prior_count:
            assignments.append((selected, weight))
            return
        visit(row + 1, used, (*selected, -1), weight)
        for observation in range(observation_count):
            if observation not in used:
                visit(
                    row + 1,
                    used | {observation},
                    (*selected, observation),
                    weight * float(edge_odds[row, observation]),
                )

    visit(0, frozenset(), (), 1.0)
    normalizer = sum(weight for _selected, weight in assignments)
    output = np.zeros((prior_count, observation_count + 1), dtype=np.float64)
    for selected, weight in assignments:
        for row, observation in enumerate(selected):
            output[row, 0 if observation < 0 else observation + 1] += weight / normalizer
    return output


def test_variational_partial_matching_is_exact_for_one_bernoulli_edge() -> None:
    log_edge_odds = torch.tensor([[[-4.0]], [[0.0]], [[4.0]]])
    result = marginal_one_to_one_association(
        log_edge_odds,
        torch.ones(3, 1, dtype=torch.bool),
    )
    expected = torch.sigmoid(log_edge_odds[:, 0, 0])

    torch.testing.assert_close(
        result.match_probability[:, 0, 0],
        expected,
        atol=3e-5,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        result.null_probability[:, 0],
        1.0 - expected,
        atol=3e-5,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        result.unexplained_observation_probability[:, 0],
        1.0 - expected,
        atol=3e-5,
        rtol=3e-5,
    )


def test_variational_partial_matching_preserves_ambiguity_against_exact_oracle() -> None:
    log_edge_odds = np.asarray(
        [
            [9.5, 8.0, 8.2],
            [8.1, 9.2, 8.4],
            [8.3, 8.5, 9.0],
        ],
        dtype=np.float32,
    )
    result = marginal_one_to_one_association(
        torch.from_numpy(log_edge_odds).unsqueeze(0),
        torch.ones(1, 3, dtype=torch.bool),
    )
    exact = _exact_partial_matching_marginals(np.exp(log_edge_odds))[:, 1:]
    actual = result.match_probability[0].numpy()

    # The former Bethe/LBP approximation saturated the diagonal above 0.997
    # even though the exact posterior retains substantial alternate mass.
    assert np.diag(actual).max() < 0.9
    assert np.abs(actual - exact).mean() < 0.04


def test_variational_normalization_capacity_permutation_and_gradients() -> None:
    torch.manual_seed(1901)
    log_odds = torch.randn(2, 4, 3, requires_grad=True)
    valid = torch.tensor([[True, True, False, True], [True, True, True, True]])
    result = marginal_one_to_one_association(log_odds, valid)

    torch.testing.assert_close(
        result.null_probability + result.match_probability.sum(dim=-1),
        torch.ones_like(result.null_probability),
        atol=2e-6,
        rtol=2e-6,
    )
    assert (result.match_probability.sum(dim=1) <= 1.0 + 2e-6).all()
    assert (result.match_probability[0, 2] == 0.0).all()
    assert result.null_probability[0, 2] == 1.0
    assert result.convergence_residual < 1e-3

    row_permutation = torch.tensor([3, 0, 2, 1])
    observation_permutation = torch.tensor([2, 0, 1])
    permuted = marginal_one_to_one_association(
        log_odds[:, row_permutation][:, :, observation_permutation],
        valid[:, row_permutation],
    )
    torch.testing.assert_close(
        permuted.match_probability,
        result.match_probability[:, row_permutation][:, :, observation_permutation],
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        permuted.null_probability,
        result.null_probability[:, row_permutation],
        atol=2e-6,
        rtol=2e-6,
    )

    loss = result.match_probability.square().sum() + result.null_probability.square().sum()
    loss.backward()
    assert log_odds.grad is not None and torch.isfinite(log_odds.grad).all()


def test_equal_weight_ci_matches_stonesoup_equations_and_is_idempotent() -> None:
    prior_mean = torch.tensor([[[-1.0, 3.0]]])
    prior_covariance = torch.tensor([[[0.2, 2.0]]])
    observation_mean = torch.tensor([[[2.0, -1.0]]])
    observation_covariance = torch.tensor([[[0.8, 1.0]]])

    result = equal_weight_covariance_intersection(
        prior_mean,
        prior_covariance,
        observation_mean,
        observation_covariance,
        minimum_variance=1e-6,
    )
    expected_covariance = 1.0 / (0.5 / prior_covariance + 0.5 / observation_covariance)
    expected_mean = expected_covariance * (
        0.5 * prior_mean / prior_covariance + 0.5 * observation_mean / observation_covariance
    )
    torch.testing.assert_close(result.mean, expected_mean)
    torch.testing.assert_close(result.covariance_diag, expected_covariance)

    duplicate = equal_weight_covariance_intersection(
        prior_mean,
        prior_covariance,
        prior_mean,
        prior_covariance,
        minimum_variance=1e-6,
    )
    torch.testing.assert_close(duplicate.mean, prior_mean)
    torch.testing.assert_close(duplicate.covariance_diag, prior_covariance)

    repeated_mean = prior_mean
    repeated_covariance = prior_covariance
    for _ in range(100):
        repeated = equal_weight_covariance_intersection(
            repeated_mean,
            repeated_covariance,
            prior_mean,
            prior_covariance,
            minimum_variance=1e-6,
        )
        repeated_mean = repeated.mean
        repeated_covariance = repeated.covariance_diag
    torch.testing.assert_close(repeated_covariance, prior_covariance, atol=0.0, rtol=0.0)


def test_pda_moments_include_measurement_origin_uncertainty() -> None:
    prior_mean = torch.tensor([[[0.0]]])
    prior_covariance = torch.tensor([[[0.1]]])
    match_mean = torch.tensor([[[[-2.0], [2.0]]]])
    match_covariance = torch.tensor([[[[0.2], [0.2]]]])
    null_weight = torch.tensor([[0.2]])
    match_weight = torch.tensor([[[0.4, 0.4]]])

    mean, covariance = probabilistic_data_association_moments(
        prior_mean,
        prior_covariance,
        match_mean,
        match_covariance,
        null_weight,
        match_weight,
        minimum_variance=1e-6,
    )

    torch.testing.assert_close(mean, torch.zeros_like(mean))
    assert covariance.item() == pytest.approx(3.38)


@pytest.mark.parametrize("bad_iterations", [0, -1, True])
def test_variational_association_rejects_invalid_iteration_count(bad_iterations: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        marginal_one_to_one_association(
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            iterations=bad_iterations,  # type: ignore[arg-type]
        )
