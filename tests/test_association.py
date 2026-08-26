from __future__ import annotations

import itertools

import numpy as np
import pytest

from picf_next.association import associate, associate_lifecycle


def _exhaustive_cost(real: np.ndarray, miss: np.ndarray, birth: np.ndarray) -> float:
    """Enumerate every injective partial prior-to-observation mapping."""

    prior_count, observation_count = real.shape
    best = np.inf

    def visit(prior_index: int, used: set[int], cost: float) -> None:
        nonlocal best
        if prior_index == prior_count:
            cost += float(birth[[j for j in range(observation_count) if j not in used]].sum())
            best = min(best, cost)
            return

        visit(prior_index + 1, used, cost + float(miss[prior_index]))
        for observation_index in range(observation_count):
            if observation_index not in used and np.isfinite(real[prior_index, observation_index]):
                visit(
                    prior_index + 1,
                    used | {observation_index},
                    cost + float(real[prior_index, observation_index]),
                )

    visit(0, set(), 0.0)
    return float(best)


def _exhaustive_lifecycle_cost(
    real: np.ndarray,
    survive: np.ndarray,
    death: np.ndarray,
    birth: np.ndarray,
    clutter: np.ndarray,
    capacity: int,
) -> float:
    prior_count, observation_count = real.shape
    best = np.inf

    def finish_observations(
        observation_index: int,
        used: set[int],
        occupied: int,
        cost: float,
    ) -> None:
        nonlocal best
        if observation_index == observation_count:
            best = min(best, cost)
            return
        if observation_index in used:
            finish_observations(observation_index + 1, used, occupied, cost)
            return
        finish_observations(
            observation_index + 1,
            used,
            occupied,
            cost + float(clutter[observation_index]),
        )
        if occupied < capacity:
            finish_observations(
                observation_index + 1,
                used,
                occupied + 1,
                cost + float(birth[observation_index]),
            )

    def visit_prior(
        prior_index: int,
        used: set[int],
        occupied: int,
        cost: float,
    ) -> None:
        if prior_index == prior_count:
            finish_observations(0, used, occupied, cost)
            return
        visit_prior(prior_index + 1, used, occupied, cost + float(death[prior_index]))
        if occupied >= capacity:
            return
        visit_prior(
            prior_index + 1,
            used,
            occupied + 1,
            cost + float(survive[prior_index]),
        )
        for observation_index in range(observation_count):
            if observation_index not in used and np.isfinite(real[prior_index, observation_index]):
                visit_prior(
                    prior_index + 1,
                    used | {observation_index},
                    occupied + 1,
                    cost + float(real[prior_index, observation_index]),
                )

    visit_prior(0, set(), 0, 0.0)
    return float(best)


@pytest.mark.parametrize(
    "prior_count,observation_count", list(itertools.product(range(4), range(4)))
)
def test_matches_exhaustive_oracle(prior_count: int, observation_count: int) -> None:
    rng = np.random.default_rng(19 + 7 * prior_count + observation_count)
    real = rng.uniform(0.0, 4.0, size=(prior_count, observation_count))
    miss = rng.uniform(0.5, 2.5, size=prior_count)
    birth = rng.uniform(0.5, 2.5, size=observation_count)

    result = associate(real, miss, birth)

    assert result.total_cost == pytest.approx(_exhaustive_cost(real, miss, birth))


@pytest.mark.parametrize(
    "prior_count,observation_count,capacity",
    [
        (prior_count, observation_count, capacity)
        for prior_count in range(4)
        for observation_count in range(4)
        for capacity in range(prior_count, 4)
    ],
)
def test_lifecycle_map_matches_capacity_constrained_exhaustive_oracle(
    prior_count: int,
    observation_count: int,
    capacity: int,
) -> None:
    rng = np.random.default_rng(101 + 19 * prior_count + 7 * observation_count + capacity)
    real = rng.uniform(0.0, 4.0, size=(prior_count, observation_count))
    survive = rng.uniform(0.1, 3.0, size=prior_count)
    death = rng.uniform(0.1, 3.0, size=prior_count)
    birth = rng.uniform(0.1, 3.0, size=observation_count)
    clutter = rng.uniform(0.1, 3.0, size=observation_count)

    result = associate_lifecycle(
        real,
        survive,
        death,
        birth,
        clutter,
        capacity=capacity,
    )

    assert result.total_cost == pytest.approx(
        _exhaustive_lifecycle_cost(
            real,
            survive,
            death,
            birth,
            clutter,
            capacity,
        )
    )
    assert result.occupied_count <= capacity


def test_lifecycle_capacity_lets_stronger_birth_replace_weaker_survivor() -> None:
    result = associate_lifecycle(
        np.array([[100.0]]),
        [1.0],
        [1.1],
        [0.01],
        [4.0],
        capacity=1,
    )

    np.testing.assert_array_equal(result.death_prior_indices, [0])
    np.testing.assert_array_equal(result.birth_observation_indices, [0])
    assert result.occupied_count == 1


def test_randomized_lifecycle_map_matches_exhaustive_oracle_with_forbidden_edges() -> None:
    rng = np.random.default_rng(20260715)
    for _ in range(128):
        prior_count = int(rng.integers(0, 4))
        observation_count = int(rng.integers(0, 4))
        capacity = int(rng.integers(prior_count, 5))
        real = rng.uniform(0.0, 5.0, size=(prior_count, observation_count))
        if real.size:
            real[rng.random(real.shape) < 0.2] = np.inf
        survive = rng.uniform(0.01, 4.0, size=prior_count)
        death = rng.uniform(0.01, 4.0, size=prior_count)
        birth = rng.uniform(0.01, 4.0, size=observation_count)
        clutter = rng.uniform(0.01, 4.0, size=observation_count)

        result = associate_lifecycle(
            real,
            survive,
            death,
            birth,
            clutter,
            capacity=capacity,
        )

        assert result.total_cost == pytest.approx(
            _exhaustive_lifecycle_cost(
                real,
                survive,
                death,
                birth,
                clutter,
                capacity,
            )
        )
        assert result.occupied_count <= capacity


def test_prefers_birth_and_miss_over_bad_real_match() -> None:
    result = associate([[100.0]], [1.0], [2.0])

    np.testing.assert_array_equal(result.prior_to_observation, [-1])
    np.testing.assert_array_equal(result.observation_to_prior, [-1])
    assert result.total_cost == 3.0


def test_forbidden_edge_is_never_selected() -> None:
    result = associate([[np.inf, 0.2], [0.1, np.inf]], [5.0, 5.0], [5.0, 5.0])

    np.testing.assert_array_equal(result.prior_to_observation, [1, 0])
    assert result.total_cost == pytest.approx(0.3)


def test_empty_sides_are_explicit_null_events() -> None:
    only_births = associate(np.empty((0, 2)), [], [0.4, 0.7])
    np.testing.assert_array_equal(only_births.birth_observation_indices, [0, 1])
    assert only_births.total_cost == pytest.approx(1.1)

    only_misses = associate(np.empty((2, 0)), [0.2, 0.8], [])
    np.testing.assert_array_equal(only_misses.unmatched_prior_indices, [0, 1])
    assert only_misses.total_cost == pytest.approx(1.0)


def test_unique_solution_is_permutation_equivariant() -> None:
    real = np.array([[0.1, 8.0, 9.0], [7.0, 0.2, 6.0], [5.0, 4.0, 0.3]])
    miss = np.full(3, 3.0)
    birth = np.full(3, 3.0)
    base = associate(real, miss, birth)

    prior_permutation = np.array([2, 0, 1])
    observation_permutation = np.array([1, 2, 0])
    permuted = associate(
        real[prior_permutation][:, observation_permutation],
        miss[prior_permutation],
        birth[observation_permutation],
    )

    mapped = np.full(3, -1, dtype=np.int64)
    for permuted_prior, permuted_observation in enumerate(permuted.prior_to_observation):
        if permuted_observation >= 0:
            mapped[prior_permutation[permuted_prior]] = observation_permutation[
                permuted_observation
            ]

    np.testing.assert_array_equal(mapped, base.prior_to_observation)
    assert permuted.total_cost == pytest.approx(base.total_cost)


def test_ties_are_repeatable_with_pinned_scipy_solver() -> None:
    outputs = [associate(np.zeros((3, 3)), np.full(3, 5.0), np.full(3, 5.0)) for _ in range(20)]
    for output in outputs[1:]:
        np.testing.assert_array_equal(output.prior_to_observation, outputs[0].prior_to_observation)


@pytest.mark.parametrize(
    "real,miss,birth",
    [
        ([[np.nan]], [1.0], [1.0]),
        ([[0.0]], [np.inf], [1.0]),
        ([[0.0]], [1.0], [np.nan]),
    ],
)
def test_rejects_invalid_costs(
    real: list[list[float]], miss: list[float], birth: list[float]
) -> None:
    with pytest.raises(ValueError):
        associate(real, miss, birth)
