from __future__ import annotations

import numpy as np
import pytest

from picf_next.association import associate
from picf_next.contracts import ObjectBeliefSet, ObjectObservationSet
from picf_next.posterior import (
    BIRTH_EVENT,
    DEATH_EVENT,
    MATCH_EVENT,
    MISS_EVENT,
    PosteriorCapacityError,
    build_object_association_problem,
    diagonal_gaussian_correct,
    diagonal_gaussian_predict,
    pack_dynamic,
    pack_state,
    pairwise_cosine_distance,
    pairwise_diagonal_gaussian_nll,
    pairwise_feature_cosine_distance,
    predict_object_belief,
    update_object_posterior,
)


def _belief() -> ObjectBeliefSet:
    return ObjectBeliefSet(
        address=np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        content=np.array([[0.0], [0.0], [1.0]], dtype=np.float64),
        geometry=np.array([[0.0], [0.0], [1.0]], dtype=np.float64),
        geometry_covariance_diag=np.array(
            [[0.2], [0.0], [0.2]],
            dtype=np.float64,
        ),
        existence=np.array([0.9, 0.0, 0.8], dtype=np.float64),
        visibility=np.array([0.8, 0.0, 0.7], dtype=np.float64),
        measurement_age_s=np.array([0.3, 0.0, 0.2], dtype=np.float64),
        valid=np.array([True, False, True], dtype=np.bool_),
        age=np.array([3, 0, 2], dtype=np.int64),
    )


def _observations() -> ObjectObservationSet:
    return ObjectObservationSet(
        address=np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        content=np.array([[1.0], [0.0], [0.0]], dtype=np.float64),
        geometry=np.array([[1.1], [0.0], [0.1]], dtype=np.float64),
        geometry_covariance_diag=np.array(
            [[0.1], [0.0], [0.1]],
            dtype=np.float64,
        ),
        existence=np.array([0.95, 0.0, 0.95], dtype=np.float64),
        valid=np.array([True, False, True], dtype=np.bool_),
    )


def test_diagonal_correction_matches_analytic_solution() -> None:
    prior_mean = np.array([[0.0, 4.0]])
    prior_variance = np.array([[1.0, 3.0]])
    observation_mean = np.array([[2.0, 0.0]])
    observation_variance = np.array([[1.0, 1.0]])

    result = diagonal_gaussian_correct(
        prior_mean,
        prior_variance,
        observation_mean,
        observation_variance,
    )

    np.testing.assert_allclose(result.gain, [[0.5, 0.75]])
    np.testing.assert_allclose(result.innovation, [[2.0, -4.0]])
    np.testing.assert_allclose(result.mean, [[1.0, 1.0]])
    np.testing.assert_allclose(result.covariance_diag, [[0.5, 0.75]])


def test_diagonal_correction_uses_joseph_form_at_variance_floor() -> None:
    result = diagonal_gaussian_correct(
        np.array([[0.0]]),
        np.array([[1e-12]]),
        np.array([[1.0]]),
        np.array([[0.0]]),
        variance_floor=1e-8,
    )

    np.testing.assert_allclose(result.gain, [[1e-4]])
    np.testing.assert_allclose(result.mean, [[1e-4]])
    np.testing.assert_allclose(result.covariance_diag, [[1e-8]])


def test_prediction_increases_uncertainty_without_mutating_inputs() -> None:
    mean = np.array([[1.0, 2.0]])
    covariance = np.array([[0.1, 0.2]])
    original_mean = mean.copy()
    original_covariance = covariance.copy()

    predicted_mean, predicted_covariance = diagonal_gaussian_predict(
        mean,
        covariance,
        residual_delta=np.array([[0.5, -0.5]]),
        process_variance=np.array([[0.01, 0.03]]),
    )

    np.testing.assert_allclose(predicted_mean, [[1.5, 1.5]])
    np.testing.assert_allclose(predicted_covariance, [[0.11, 0.23]])
    np.testing.assert_array_equal(mean, original_mean)
    np.testing.assert_array_equal(covariance, original_covariance)


def test_informative_correction_reduces_prediction_uncertainty() -> None:
    predicted_mean, predicted_covariance = diagonal_gaussian_predict(
        np.zeros((1, 3)),
        np.full((1, 3), 0.2),
        np.zeros((1, 3)),
        np.full((1, 3), 0.1),
    )
    corrected = diagonal_gaussian_correct(
        predicted_mean,
        predicted_covariance,
        np.ones((1, 3)),
        np.full((1, 3), 0.05),
    )

    assert np.all(corrected.covariance_diag < predicted_covariance)


def test_pairwise_cost_penalizes_variance_inflation() -> None:
    prior_mean = np.zeros((1, 2))
    observation_mean = np.zeros((2, 2))
    prior_variance = np.full((1, 2), 0.1)
    observation_variance = np.array([[0.1, 0.1], [10.0, 10.0]])

    cost = pairwise_diagonal_gaussian_nll(
        prior_mean,
        prior_variance,
        observation_mean,
        observation_variance,
    )

    assert cost[0, 0] < cost[0, 1]


def test_pairwise_cost_retains_normalizer_for_real_vs_null_energy() -> None:
    cost = pairwise_diagonal_gaussian_nll(
        np.zeros((1, 2)),
        np.full((1, 2), 0.5),
        np.zeros((1, 2)),
        np.full((1, 2), 0.5),
    )

    np.testing.assert_allclose(cost, [[0.5 * np.log(2.0 * np.pi)]])


def test_pairwise_cost_is_permutation_equivariant() -> None:
    beliefs = _belief()
    observations = _observations()
    prior = beliefs.geometry[beliefs.valid]
    prior_covariance = beliefs.geometry_covariance_diag[beliefs.valid]
    observed = observations.geometry[observations.valid]
    observed_covariance = observations.geometry_covariance_diag[observations.valid]
    base = pairwise_diagonal_gaussian_nll(
        prior,
        prior_covariance,
        observed,
        observed_covariance,
    )

    prior_permutation = np.array([1, 0])
    observation_permutation = np.array([1, 0])
    permuted = pairwise_diagonal_gaussian_nll(
        prior[prior_permutation],
        prior_covariance[prior_permutation],
        observed[observation_permutation],
        observed_covariance[observation_permutation],
    )

    np.testing.assert_allclose(
        permuted,
        base[prior_permutation][:, observation_permutation],
    )


def test_cosine_address_cost_is_scale_free_and_permutation_equivariant() -> None:
    prior = np.array([[1.0, 0.0], [0.0, 1.0]])
    observed = np.array([[0.0, 1.0], [1.0, 0.0]])
    base = pairwise_cosine_distance(prior, observed)

    np.testing.assert_allclose(base, [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="unit norm"):
        pairwise_cosine_distance(2.0 * prior, observed)


def test_content_cosine_cost_is_scale_free_without_requiring_unit_inputs() -> None:
    prior = np.array([[2.0, 0.0], [0.0, 3.0]])
    observed = np.array([[0.0, 5.0], [7.0, 0.0]])

    base = pairwise_feature_cosine_distance(prior, observed)
    scaled = pairwise_feature_cosine_distance(11.0 * prior, 0.25 * observed)

    np.testing.assert_allclose(base, [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(scaled, base)


def test_object_problem_compacts_valid_rows_and_recovers_global_maps() -> None:
    beliefs = _belief()
    observations = _observations()
    problem = build_object_association_problem(
        beliefs,
        observations,
        unmatched_prior_cost=np.full(3, 5.0),
        birth_observation_cost=np.full(3, 5.0),
    )
    result = associate(
        problem.real_cost,
        problem.unmatched_prior_cost,
        problem.birth_observation_cost,
    )

    np.testing.assert_array_equal(problem.prior_indices, [0, 2])
    np.testing.assert_array_equal(problem.observation_indices, [0, 2])
    local_matches = result.matched_observation_indices
    global_matches = problem.observation_indices[local_matches]
    np.testing.assert_array_equal(global_matches, [2, 0])


def test_low_existence_observation_is_more_expensive() -> None:
    beliefs = _belief()
    observations = _observations()
    altered = ObjectObservationSet(
        address=np.repeat(observations.address[[0]], 2, axis=0),
        content=np.repeat(observations.content[[0]], 2, axis=0),
        geometry=np.repeat(observations.geometry[[0]], 2, axis=0),
        geometry_covariance_diag=np.repeat(observations.geometry_covariance_diag[[0]], 2, axis=0),
        existence=np.array([0.9, 0.1]),
        valid=np.array([True, True]),
    )
    problem = build_object_association_problem(
        beliefs,
        altered,
        unmatched_prior_cost=np.full(3, 5.0),
        birth_observation_cost=np.full(2, 5.0),
    )

    assert np.all(problem.real_cost[:, 0] < problem.real_cost[:, 1])


@pytest.mark.parametrize("bad_variance", [-1.0, np.nan])
def test_prediction_rejects_invalid_variance(bad_variance: float) -> None:
    with pytest.raises(ValueError):
        diagonal_gaussian_predict(
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.array([[bad_variance]]),
        )


def test_object_prediction_keeps_address_mean_and_updates_dynamic_state() -> None:
    prior = _belief()
    dynamic_delta = np.array([[0.5, -0.5], [7.0, 7.0], [0.2, -0.2]])
    process_variance = np.full((3, 1), 0.05)
    predicted = predict_object_belief(
        prior,
        dynamic_delta,
        process_variance,
        survival_probability=np.array([0.9, 0.9, 0.8]),
        conditional_visibility_probability=np.array([0.5, 0.5, 0.25]),
        delta_t_s=0.1,
    )

    np.testing.assert_allclose(predicted.address[predicted.valid], prior.address[prior.valid])
    np.testing.assert_array_equal(predicted.address[~predicted.valid], 0.0)
    np.testing.assert_allclose(predicted.content[[0, 2]], [[0.5], [1.2]])
    np.testing.assert_allclose(predicted.geometry[[0, 2]], [[-0.5], [0.8]])
    np.testing.assert_allclose(predicted.geometry_covariance_diag[[0, 2]], 0.25)
    np.testing.assert_allclose(predicted.existence, [0.81, 0.0, 0.64])
    np.testing.assert_allclose(predicted.visibility, [0.405, 0.0, 0.16])
    np.testing.assert_allclose(predicted.measurement_age_s, [0.4, 0.0, 0.3])
    np.testing.assert_array_equal(predicted.age, [4, 0, 3])


def test_matched_update_preserves_persistent_row_and_exposes_innovation() -> None:
    predicted = _belief()
    observations = _observations()
    problem = build_object_association_problem(
        predicted,
        observations,
        unmatched_prior_cost=np.full(3, 5.0),
        birth_observation_cost=np.full(3, 5.0),
    )
    assignment = associate(
        problem.real_cost,
        problem.unmatched_prior_cost,
        problem.birth_observation_cost,
    )
    update = update_object_posterior(
        predicted,
        observations,
        problem,
        assignment,
        posterior_prior_existence=np.array([0.95, 0.0, 0.9]),
        birth_existence=np.array([0.9, 0.0, 0.9]),
        retain_unmatched_prior=np.array([True, False, True]),
    )

    np.testing.assert_array_equal(update.event_type, [MATCH_EVENT, 0, MATCH_EVENT])
    np.testing.assert_array_equal(update.posterior_to_observation, [2, -1, 0])
    np.testing.assert_allclose(
        update.innovation[0],
        pack_dynamic(observations)[2] - pack_dynamic(predicted)[0],
    )
    np.testing.assert_allclose(
        update.innovation[2],
        pack_dynamic(observations)[0] - pack_dynamic(predicted)[2],
    )
    np.testing.assert_array_equal(update.belief.address, predicted.address)
    np.testing.assert_allclose(update.belief.content[0], observations.content[2])
    np.testing.assert_allclose(update.belief.content[2], observations.content[0])
    assert np.all(update.belief.geometry_covariance_diag[update.belief.valid] < 0.2)


def test_occluded_object_survives_without_visibility_or_fake_correction() -> None:
    prior = ObjectBeliefSet(
        address=np.array([[0.5547002, 0.83205029], [0.0, 0.0]]),
        content=np.array([[0.4], [0.0]]),
        geometry=np.array([[0.5], [0.0]]),
        geometry_covariance_diag=np.array([[0.1], [0.0]]),
        existence=np.array([0.9, 0.0]),
        visibility=np.array([0.8, 0.0]),
        measurement_age_s=np.array([0.0, 0.0]),
        valid=np.array([True, False]),
        age=np.array([2, 0]),
    )
    predicted = predict_object_belief(
        prior,
        dynamic_residual_delta=np.zeros((2, 2)),
        process_variance=np.array([[0.05], [0.0]]),
        survival_probability=np.array([0.95, 0.0]),
        conditional_visibility_probability=np.array([0.2, 0.0]),
        delta_t_s=0.1,
    )
    empty = ObjectObservationSet(
        address=np.empty((0, 2)),
        content=np.empty((0, 1)),
        geometry=np.empty((0, 1)),
        geometry_covariance_diag=np.empty((0, 1)),
        existence=np.empty(0),
        valid=np.empty(0, dtype=np.bool_),
    )
    problem = build_object_association_problem(
        predicted,
        empty,
        unmatched_prior_cost=np.array([0.2, 0.0]),
        birth_observation_cost=np.empty(0),
    )
    assignment = associate(
        problem.real_cost,
        problem.unmatched_prior_cost,
        problem.birth_observation_cost,
    )
    update = update_object_posterior(
        predicted,
        empty,
        problem,
        assignment,
        posterior_prior_existence=np.array([0.82, 0.0]),
        birth_existence=np.empty(0),
        retain_unmatched_prior=np.array([True, False]),
    )

    assert update.event_type[0] == MISS_EVENT
    assert update.belief.valid[0]
    assert update.belief.visibility[0] == 0.0
    np.testing.assert_allclose(update.belief.geometry_covariance_diag[0], [0.15])
    np.testing.assert_array_equal(update.innovation[0], np.zeros(2))
    np.testing.assert_allclose(update.belief.measurement_age_s, [0.1, 0.0])


def test_death_releases_capacity_for_a_calibrated_birth() -> None:
    predicted = _belief()
    observation = ObjectObservationSet(
        address=np.array([[np.sqrt(0.5), np.sqrt(0.5)]]),
        content=np.array([[4.0]]),
        geometry=np.array([[4.0]]),
        geometry_covariance_diag=np.full((1, 1), 0.1),
        existence=np.array([0.9]),
        valid=np.array([True]),
    )
    problem = build_object_association_problem(
        predicted,
        observation,
        unmatched_prior_cost=np.array([0.1, 0.0, 0.1]),
        birth_observation_cost=np.array([0.1]),
    )
    assignment = associate(
        problem.real_cost,
        problem.unmatched_prior_cost,
        problem.birth_observation_cost,
    )
    update = update_object_posterior(
        predicted,
        observation,
        problem,
        assignment,
        posterior_prior_existence=np.array([0.0, 0.0, 0.0]),
        birth_existence=np.array([0.85]),
        retain_unmatched_prior=np.array([False, False, False]),
    )

    np.testing.assert_array_equal(update.event_type, [BIRTH_EVENT, 0, DEATH_EVENT])
    np.testing.assert_allclose(pack_state(update.belief)[0], pack_state(observation)[0])
    assert update.belief.age[0] == 0


def test_capacity_overflow_is_explicit_not_hidden_eviction() -> None:
    predicted = ObjectBeliefSet(
        address=np.array([[1.0], [-1.0]]),
        content=np.array([[0.0], [10.0]]),
        geometry=np.array([[0.0], [10.0]]),
        geometry_covariance_diag=np.full((2, 1), 0.1),
        existence=np.array([0.9, 0.9]),
        visibility=np.array([0.8, 0.8]),
        measurement_age_s=np.zeros(2),
        valid=np.array([True, True]),
        age=np.array([1, 1]),
    )
    observation = ObjectObservationSet(
        address=np.array([[1.0]]),
        content=np.array([[100.0]]),
        geometry=np.array([[100.0]]),
        geometry_covariance_diag=np.full((1, 1), 0.1),
        existence=np.array([0.9]),
        valid=np.array([True]),
    )
    problem = build_object_association_problem(
        predicted,
        observation,
        unmatched_prior_cost=np.array([0.1, 0.1]),
        birth_observation_cost=np.array([0.1]),
    )
    assignment = associate(
        problem.real_cost,
        problem.unmatched_prior_cost,
        problem.birth_observation_cost,
    )

    with pytest.raises(PosteriorCapacityError):
        update_object_posterior(
            predicted,
            observation,
            problem,
            assignment,
            posterior_prior_existence=np.array([0.8, 0.8]),
            birth_existence=np.array([0.8]),
            retain_unmatched_prior=np.array([True, True]),
        )


def test_reappearance_corrects_the_same_persistent_row_after_occlusion() -> None:
    prior = ObjectBeliefSet(
        address=np.array([[0.5547002, 0.83205029], [0.0, 0.0]]),
        content=np.array([[0.4], [0.0]]),
        geometry=np.array([[0.5], [0.0]]),
        geometry_covariance_diag=np.array([[0.1], [0.0]]),
        existence=np.array([0.9, 0.0]),
        visibility=np.array([0.8, 0.0]),
        measurement_age_s=np.array([0.0, 0.0]),
        valid=np.array([True, False]),
        age=np.array([2, 0]),
    )
    occluded_prior = predict_object_belief(
        prior,
        dynamic_residual_delta=np.zeros((2, 2)),
        process_variance=np.array([[0.05], [0.0]]),
        survival_probability=np.array([0.95, 0.0]),
        conditional_visibility_probability=np.array([0.0, 0.0]),
        delta_t_s=0.1,
    )
    empty = ObjectObservationSet(
        address=np.empty((0, 2)),
        content=np.empty((0, 1)),
        geometry=np.empty((0, 1)),
        geometry_covariance_diag=np.empty((0, 1)),
        existence=np.empty(0),
        valid=np.empty(0, dtype=np.bool_),
    )
    empty_problem = build_object_association_problem(
        occluded_prior,
        empty,
        unmatched_prior_cost=np.array([0.2, 0.0]),
        birth_observation_cost=np.empty(0),
    )
    missed = update_object_posterior(
        occluded_prior,
        empty,
        empty_problem,
        associate(
            empty_problem.real_cost,
            empty_problem.unmatched_prior_cost,
            empty_problem.birth_observation_cost,
        ),
        posterior_prior_existence=np.array([0.82, 0.0]),
        birth_existence=np.empty(0),
        retain_unmatched_prior=np.array([True, False]),
    )

    predicted_again = predict_object_belief(
        missed.belief,
        dynamic_residual_delta=np.zeros((2, 2)),
        process_variance=np.array([[0.05], [0.0]]),
        survival_probability=np.array([0.95, 0.0]),
        conditional_visibility_probability=np.array([0.8, 0.0]),
        delta_t_s=0.1,
    )
    reappeared = ObjectObservationSet(
        address=np.array([[0.58650981, 0.80994213]]),
        content=np.array([[0.39]]),
        geometry=np.array([[0.51]]),
        geometry_covariance_diag=np.full((1, 1), 0.05),
        existence=np.array([0.95]),
        valid=np.array([True]),
    )
    problem = build_object_association_problem(
        predicted_again,
        reappeared,
        unmatched_prior_cost=np.array([2.0, 0.0]),
        birth_observation_cost=np.array([2.0]),
    )
    assignment = associate(
        problem.real_cost,
        problem.unmatched_prior_cost,
        problem.birth_observation_cost,
    )
    corrected = update_object_posterior(
        predicted_again,
        reappeared,
        problem,
        assignment,
        posterior_prior_existence=np.array([0.9, 0.0]),
        birth_existence=np.array([0.9]),
        retain_unmatched_prior=np.array([True, False]),
    )

    assert corrected.event_type[0] == MATCH_EVENT
    assert corrected.observation_to_posterior[0] == 0
    assert corrected.belief.age[0] == 4
    assert corrected.belief.visibility[0] > 0.0
    assert corrected.belief.measurement_age_s[0] == 0.0


def test_address_evidence_preserves_identity_when_geometry_crosses() -> None:
    predicted = ObjectBeliefSet(
        address=np.array([[0.0, 1.0], [1.0, 0.0]]),
        content=np.array([[0.0], [1.0]]),
        geometry=np.array([[-1.0], [1.0]]),
        geometry_covariance_diag=np.full((2, 1), 1.0),
        existence=np.array([0.95, 0.95]),
        visibility=np.array([0.9, 0.9]),
        measurement_age_s=np.zeros(2),
        valid=np.array([True, True]),
        age=np.array([3, 3]),
    )
    crossed = ObjectObservationSet(
        address=np.array([[1.0, 0.0], [0.0, 1.0]]),
        content=np.array([[1.0], [0.0]]),
        geometry=np.array([[-1.0], [1.0]]),
        geometry_covariance_diag=np.full((2, 1), 1.0),
        existence=np.array([0.95, 0.95]),
        valid=np.array([True, True]),
    )
    problem = build_object_association_problem(
        predicted,
        crossed,
        unmatched_prior_cost=np.full(2, 5.0),
        birth_observation_cost=np.full(2, 5.0),
        address_weight=4.0,
    )
    assignment = associate(
        problem.real_cost,
        problem.unmatched_prior_cost,
        problem.birth_observation_cost,
    )

    np.testing.assert_array_equal(assignment.prior_to_observation, [1, 0])
