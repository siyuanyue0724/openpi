"""Stateless null-augmented energy-MAP association for unordered object sets.

The linear-assignment subproblem follows the semantics of the official
ICML-2026 Rethinking OCL Hungarian predictor and Mask2Former matcher. The
rectangular birth/miss augmentation is PICF-Next-specific and is verified
against an exhaustive oracle in the local test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """One-to-one matches plus explicit unmatched-prior and birth decisions."""

    prior_to_observation: IntArray
    observation_to_prior: IntArray
    matched_prior_indices: IntArray
    matched_observation_indices: IntArray
    unmatched_prior_indices: IntArray
    birth_observation_indices: IntArray
    total_cost: float


@dataclass(frozen=True, slots=True)
class LifecycleAssociationResult:
    """Minimum-energy lifecycle hypothesis under fixed posterior capacity."""

    prior_to_observation: IntArray
    observation_to_prior: IntArray
    retained_unmatched_prior_indices: IntArray
    death_prior_indices: IntArray
    birth_observation_indices: IntArray
    clutter_observation_indices: IntArray
    occupied_count: int
    total_cost: float


def _as_cost_vector(name: str, value: NDArray | list[float], length: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _as_real_cost(value: NDArray | list[list[float]]) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"real_cost must be rank two, got shape {result.shape}")
    if np.isnan(result).any() or np.isneginf(result).any():
        raise ValueError("real_cost may contain +inf forbidden edges, but not NaN or -inf")
    return result


def associate(
    real_cost: NDArray | list[list[float]],
    unmatched_prior_cost: NDArray | list[float],
    birth_observation_cost: NDArray | list[float],
) -> AssociationResult:
    """Solve one-to-one prior/observation association with null decisions.

    The objective is

    `sum(real matches) + sum(unmatched priors) + sum(birth observations)`.

    Each prior and observation participates in exactly one real or null event.
    `+inf` in `real_cost` forbids a real edge. The returned `-1` sentinel means
    unmatched or birth and is never a persistent object identifier.
    """

    real = _as_real_cost(real_cost)
    prior_count, observation_count = real.shape
    miss = _as_cost_vector("unmatched_prior_cost", unmatched_prior_cost, prior_count)
    birth = _as_cost_vector("birth_observation_cost", birth_observation_cost, observation_count)

    if prior_count == 0 and observation_count == 0:
        empty = np.empty(0, dtype=np.int64)
        return AssociationResult(empty, empty, empty, empty, empty, empty, 0.0)

    # Rows are [prior rows, birth rows]. Columns are [observation columns,
    # unmatched-prior columns]. The zero bottom-right block consumes the dummy
    # row/column capacity left after real/null decisions.
    size = prior_count + observation_count
    augmented = np.full((size, size), np.inf, dtype=np.float64)
    augmented[:prior_count, :observation_count] = real
    prior_indices = np.arange(prior_count)
    observation_indices = np.arange(observation_count)
    augmented[prior_indices, observation_count + prior_indices] = miss
    augmented[prior_count + observation_indices, observation_indices] = birth
    augmented[prior_count:, observation_count:] = 0.0

    row_indices, column_indices = linear_sum_assignment(augmented)
    chosen_cost = augmented[row_indices, column_indices]
    if not np.isfinite(chosen_cost).all():
        raise RuntimeError("null-augmented association unexpectedly has no feasible assignment")

    prior_to_observation = np.full(prior_count, -1, dtype=np.int64)
    observation_to_prior = np.full(observation_count, -1, dtype=np.int64)

    for row, column in zip(row_indices, column_indices, strict=True):
        if row < prior_count and column < observation_count:
            prior_to_observation[row] = column
            observation_to_prior[column] = row

    matched_prior = np.flatnonzero(prior_to_observation >= 0).astype(np.int64)
    matched_observation = prior_to_observation[matched_prior]
    unmatched_prior = np.flatnonzero(prior_to_observation < 0).astype(np.int64)
    birth_observation = np.flatnonzero(observation_to_prior < 0).astype(np.int64)

    original_total = float(real[matched_prior, matched_observation].sum())
    original_total += float(miss[unmatched_prior].sum())
    original_total += float(birth[birth_observation].sum())

    return AssociationResult(
        prior_to_observation=prior_to_observation,
        observation_to_prior=observation_to_prior,
        matched_prior_indices=matched_prior,
        matched_observation_indices=matched_observation,
        unmatched_prior_indices=unmatched_prior,
        birth_observation_indices=birth_observation,
        total_cost=original_total,
    )


def associate_lifecycle(
    real_cost: NDArray | list[list[float]],
    missed_survival_cost: NDArray | list[float],
    death_cost: NDArray | list[float],
    birth_cost: NDArray | list[float],
    clutter_cost: NDArray | list[float],
    *,
    capacity: int,
) -> LifecycleAssociationResult:
    """Solve the exact finite-capacity lifecycle energy in one assignment.

    Start from the valid configuration in which every prior dies and every
    observation is clutter. Selecting a real match, a missed survivor or a
    birth replaces the corresponding baseline event and occupies exactly one
    posterior row. These optional events form a bipartite matching:

    * prior -> observation: real match;
    * prior -> private null: missed but alive;
    * private birth -> observation: new object.

    The helper below adds zero-cost dummy edges and requests exactly
    ``capacity`` edges, so unused rows are represented by dummy choices. This
    is an exact reduction of the at-most-capacity additive objective to one
    linear assignment, not a post-hoc top-k truncation. When every supplied
    cost is a calibrated negative log factor, this is also the corresponding
    hard MAP. The solver itself deliberately does not claim that calibration.
    """

    real = _as_real_cost(real_cost)
    prior_count, observation_count = real.shape
    survive = _as_cost_vector("missed_survival_cost", missed_survival_cost, prior_count)
    death = _as_cost_vector("death_cost", death_cost, prior_count)
    birth = _as_cost_vector("birth_cost", birth_cost, observation_count)
    clutter = _as_cost_vector("clutter_cost", clutter_cost, observation_count)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("capacity must be a nonnegative integer")
    if prior_count > capacity:
        raise ValueError("valid prior count cannot exceed posterior capacity")

    prior_to_observation = np.full(prior_count, -1, dtype=np.int64)
    observation_to_prior = np.full(observation_count, -1, dtype=np.int64)
    if capacity == 0:
        return LifecycleAssociationResult(
            prior_to_observation=prior_to_observation,
            observation_to_prior=observation_to_prior,
            retained_unmatched_prior_indices=np.empty(0, dtype=np.int64),
            death_prior_indices=np.arange(prior_count, dtype=np.int64),
            birth_observation_indices=np.empty(0, dtype=np.int64),
            clutter_observation_indices=np.arange(observation_count, dtype=np.int64),
            occupied_count=0,
            total_cost=float(death.sum() + clutter.sum()),
        )

    actual_left = prior_count + observation_count
    actual_right = observation_count + prior_count
    edge_count = actual_left + capacity
    edge_cost = np.full((edge_count, edge_count), np.inf, dtype=np.float64)

    if prior_count and observation_count:
        edge_cost[:prior_count, :observation_count] = real - death[:, None] - clutter[None, :]
    prior_indices = np.arange(prior_count)
    edge_cost[prior_indices, observation_count + prior_indices] = survive - death
    observation_indices = np.arange(observation_count)
    edge_cost[prior_count + observation_indices, observation_indices] = birth - clutter
    dummy_indices = np.arange(capacity)
    edge_cost[
        actual_left + dummy_indices,
        actual_right + dummy_indices,
    ] = 0.0

    selected_rows, selected_columns = _minimum_exact_cardinality_matching(
        edge_cost,
        cardinality=capacity,
    )
    retained = []
    births = []
    selected_delta = 0.0
    occupied_count = 0
    for row, column in zip(selected_rows, selected_columns, strict=True):
        if row >= actual_left or column >= actual_right:
            continue
        selected_delta += float(edge_cost[row, column])
        occupied_count += 1
        if row < prior_count and column < observation_count:
            prior_to_observation[row] = column
            observation_to_prior[column] = row
        elif row < prior_count and column == observation_count + row:
            retained.append(row)
        elif prior_count <= row < actual_left and column == row - prior_count:
            births.append(column)
        else:  # pragma: no cover - forbidden edges cannot be selected
            raise RuntimeError("lifecycle assignment selected an invalid event edge")

    retained_array = np.asarray(retained, dtype=np.int64)
    birth_array = np.asarray(births, dtype=np.int64)
    matched_prior = np.flatnonzero(prior_to_observation >= 0)
    alive_prior = np.zeros(prior_count, dtype=np.bool_)
    alive_prior[matched_prior] = True
    alive_prior[retained_array] = True
    death_array = np.flatnonzero(~alive_prior).astype(np.int64)
    observed_object = observation_to_prior >= 0
    observed_object[birth_array] = True
    clutter_array = np.flatnonzero(~observed_object).astype(np.int64)
    baseline = float(death.sum() + clutter.sum())
    return LifecycleAssociationResult(
        prior_to_observation=prior_to_observation,
        observation_to_prior=observation_to_prior,
        retained_unmatched_prior_indices=retained_array,
        death_prior_indices=death_array,
        birth_observation_indices=birth_array,
        clutter_observation_indices=clutter_array,
        occupied_count=occupied_count,
        total_cost=baseline + selected_delta,
    )


def _minimum_exact_cardinality_matching(
    edge_cost: FloatArray,
    *,
    cardinality: int,
) -> tuple[IntArray, IntArray]:
    """Return a minimum-cost matching with exactly ``cardinality`` edges."""

    if edge_cost.ndim != 2:
        raise ValueError("edge_cost must be rank two")
    left_count, right_count = edge_cost.shape
    if cardinality < 0 or cardinality > min(left_count, right_count):
        raise ValueError("matching cardinality is infeasible")
    if cardinality == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    # A perfect assignment over this augmented matrix must use exactly k edges
    # from the original top-left block. Bottom rows consume right vertices not
    # selected by the matching; top rows not selected consume dummy columns.
    size = left_count + right_count - cardinality
    augmented = np.full((size, size), np.inf, dtype=np.float64)
    augmented[:left_count, :right_count] = edge_cost
    augmented[:left_count, right_count:] = 0.0
    augmented[left_count:, :right_count] = 0.0
    rows, columns = linear_sum_assignment(augmented)
    chosen = augmented[rows, columns]
    if not np.isfinite(chosen).all():
        raise RuntimeError("cardinality-constrained assignment is infeasible")
    selected = (rows < left_count) & (columns < right_count)
    selected_rows = rows[selected].astype(np.int64)
    selected_columns = columns[selected].astype(np.int64)
    if selected_rows.shape != (cardinality,):
        raise RuntimeError("cardinality reduction returned the wrong number of edges")
    return selected_rows, selected_columns
