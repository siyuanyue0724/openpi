"""Auditable probabilistic primitives for the PICF object posterior.

This module contains no learned network and no benchmark policy. A learned
transition, discovery model, and lifecycle calibrator must emit the explicit
quantities consumed here. Keeping these operations separate prevents a hidden
recurrent state from silently becoming the persistent identity mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from picf_next.association import AssociationResult
from picf_next.contracts import ObjectBeliefSet, ObjectObservationSet

FloatArray = NDArray[np.floating]
IndexArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GaussianCorrection:
    """Result of an exact diagonal linear-Gaussian correction."""

    mean: FloatArray
    covariance_diag: FloatArray
    gain: FloatArray
    innovation: FloatArray


@dataclass(frozen=True, slots=True)
class ObjectAssociationProblem:
    """Compact valid-row association problem with global row maps."""

    prior_indices: IndexArray
    observation_indices: IndexArray
    real_cost: FloatArray
    unmatched_prior_cost: FloatArray
    birth_observation_cost: FloatArray


@dataclass(frozen=True, slots=True)
class PosteriorUpdate:
    """Corrected belief plus explicit event and correspondence diagnostics."""

    belief: ObjectBeliefSet
    innovation: FloatArray
    event_type: IndexArray
    posterior_to_observation: IndexArray
    observation_to_posterior: IndexArray


class PosteriorCapacityError(RuntimeError):
    """Raised when calibrated births exceed available posterior capacity."""


UNUSED_EVENT = 0
MATCH_EVENT = 1
MISS_EVENT = 2
BIRTH_EVENT = 3
DEATH_EVENT = 4


def _require_same_shape(name: str, *arrays: FloatArray) -> tuple[int, ...]:
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        shapes = [array.shape for array in arrays]
        raise ValueError(f"{name} arrays must have the same shape, got {shapes}")
    return shape


def _require_finite(name: str, value: FloatArray) -> None:
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must be floating point")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def pack_state(objects: ObjectBeliefSet | ObjectObservationSet) -> FloatArray:
    """Concatenate address, content, and geometry without changing row order."""

    return np.concatenate((objects.address, objects.content, objects.geometry), axis=1)


def pack_dynamic(objects: ObjectBeliefSet | ObjectObservationSet) -> FloatArray:
    """Concatenate only the Euclidean dynamic state coordinates."""

    return np.concatenate((objects.content, objects.geometry), axis=1)


def diagonal_gaussian_predict(
    mean: FloatArray,
    covariance_diag: FloatArray,
    residual_delta: FloatArray,
    process_variance: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply an additive residual transition with explicit process variance."""

    _require_same_shape("prediction", mean, covariance_diag, residual_delta, process_variance)
    for name, value in {
        "mean": mean,
        "covariance_diag": covariance_diag,
        "residual_delta": residual_delta,
        "process_variance": process_variance,
    }.items():
        _require_finite(name, value)
    if (covariance_diag < 0.0).any() or (process_variance < 0.0).any():
        raise ValueError("prediction variances must be non-negative")

    return mean + residual_delta, covariance_diag + process_variance


def diagonal_gaussian_correct(
    prior_mean: FloatArray,
    prior_covariance_diag: FloatArray,
    observation_mean: FloatArray,
    observation_covariance_diag: FloatArray,
    *,
    variance_floor: float = 1e-8,
) -> GaussianCorrection:
    """Fuse equal-coordinate diagonal Gaussian states with the Kalman update."""

    _require_same_shape(
        "correction",
        prior_mean,
        prior_covariance_diag,
        observation_mean,
        observation_covariance_diag,
    )
    for name, value in {
        "prior_mean": prior_mean,
        "prior_covariance_diag": prior_covariance_diag,
        "observation_mean": observation_mean,
        "observation_covariance_diag": observation_covariance_diag,
    }.items():
        _require_finite(name, value)
    if variance_floor <= 0.0 or not np.isfinite(variance_floor):
        raise ValueError("variance_floor must be finite and positive")
    if (prior_covariance_diag < 0.0).any() or (observation_covariance_diag < 0.0).any():
        raise ValueError("correction variances must be non-negative")

    total_variance = np.maximum(
        prior_covariance_diag + observation_covariance_diag,
        variance_floor,
    )
    gain = prior_covariance_diag / total_variance
    innovation = observation_mean - prior_mean
    corrected_mean = prior_mean + gain * innovation
    # Use the same Joseph form as the production Torch corrector.  The simpler
    # ``(1 - K) P`` identity is exact only before the denominator floor and can
    # fall below the declared variance floor in the regime where that floor is
    # active.
    corrected_covariance = (
        1.0 - gain
    ) ** 2 * prior_covariance_diag + gain**2 * observation_covariance_diag
    corrected_covariance = np.maximum(corrected_covariance, variance_floor)
    return GaussianCorrection(corrected_mean, corrected_covariance, gain, innovation)


def pairwise_diagonal_gaussian_nll(
    prior_mean: FloatArray,
    prior_covariance_diag: FloatArray,
    observation_mean: FloatArray,
    observation_covariance_diag: FloatArray,
    *,
    dimension_weights: FloatArray | None = None,
    variance_floor: float = 1e-8,
) -> FloatArray:
    """Return a dimension-normalized diagonal-Gaussian matching energy.

    The log-variance term is retained so an uncertain candidate does not become
    artificially attractive merely by inflating its covariance.  The
    ``log(2*pi)`` normalizer is also retained because it is constant across real
    matches but not across real and null lifecycle hypotheses. Dimension weights
    sum to one, keeping this energy scale stable as geometry contracts expand;
    consequently this is a calibrated composite-energy term rather than the sum
    of a fully normalized multivariate log density.
    """

    if prior_mean.ndim != 2 or observation_mean.ndim != 2:
        raise ValueError("prior_mean and observation_mean must be rank two")
    if prior_mean.shape[1] != observation_mean.shape[1]:
        raise ValueError("prior and observation state widths must match")
    if prior_covariance_diag.shape != prior_mean.shape:
        raise ValueError("prior covariance shape must match prior mean")
    if observation_covariance_diag.shape != observation_mean.shape:
        raise ValueError("observation covariance shape must match observation mean")
    for name, value in {
        "prior_mean": prior_mean,
        "prior_covariance_diag": prior_covariance_diag,
        "observation_mean": observation_mean,
        "observation_covariance_diag": observation_covariance_diag,
    }.items():
        _require_finite(name, value)
    if variance_floor <= 0.0 or not np.isfinite(variance_floor):
        raise ValueError("variance_floor must be finite and positive")
    if (prior_covariance_diag < 0.0).any() or (observation_covariance_diag < 0.0).any():
        raise ValueError("pairwise variances must be non-negative")

    width = prior_mean.shape[1]
    if dimension_weights is None:
        weights = np.full(width, 1.0 / max(width, 1), dtype=np.float64)
    else:
        weights = np.asarray(dimension_weights, dtype=np.float64)
        if weights.shape != (width,) or not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("dimension_weights must be a finite non-negative state-width vector")
        if weights.sum() <= 0.0:
            raise ValueError("dimension_weights must have positive total mass")
        weights = weights / weights.sum()

    residual = observation_mean[None, :, :] - prior_mean[:, None, :]
    variance = np.maximum(
        prior_covariance_diag[:, None, :] + observation_covariance_diag[None, :, :],
        variance_floor,
    )
    terms = 0.5 * (residual**2 / variance + np.log(variance) + np.log(2.0 * np.pi))
    return np.sum(terms * weights[None, None, :], axis=-1)


def pairwise_cosine_distance(
    prior_address: FloatArray, observation_address: FloatArray
) -> FloatArray:
    """Return the geodesically monotone energy ``1 - cosine`` for unit keys."""

    if prior_address.ndim != 2 or observation_address.ndim != 2:
        raise ValueError("address arrays must be rank two")
    if prior_address.shape[1] != observation_address.shape[1]:
        raise ValueError("prior and observation address widths must match")
    for name, value in {
        "prior_address": prior_address,
        "observation_address": observation_address,
    }.items():
        _require_finite(name, value)
        if value.shape[0] and not np.allclose(
            np.linalg.norm(value, axis=1),
            1.0,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(f"{name} rows must have unit norm")
    return 1.0 - np.clip(prior_address @ observation_address.T, -1.0, 1.0)


def pairwise_feature_cosine_distance(
    prior_feature: FloatArray,
    observation_feature: FloatArray,
) -> FloatArray:
    """Return bounded cosine distance for deterministic latent descriptors."""

    if prior_feature.ndim != 2 or observation_feature.ndim != 2:
        raise ValueError("feature arrays must be rank two")
    if prior_feature.shape[1] != observation_feature.shape[1]:
        raise ValueError("prior and observation feature widths must match")
    _require_finite("prior_feature", prior_feature)
    _require_finite("observation_feature", observation_feature)
    epsilon = np.finfo(np.result_type(prior_feature.dtype, observation_feature.dtype)).eps
    prior_norm = prior_feature / np.maximum(
        np.linalg.norm(prior_feature, axis=1, keepdims=True),
        epsilon,
    )
    observation_norm = observation_feature / np.maximum(
        np.linalg.norm(observation_feature, axis=1, keepdims=True),
        epsilon,
    )
    return 1.0 - np.clip(prior_norm @ observation_norm.T, -1.0, 1.0)


def build_object_association_problem(
    predicted: ObjectBeliefSet,
    observations: ObjectObservationSet,
    unmatched_prior_cost: FloatArray,
    birth_observation_cost: FloatArray,
    *,
    address_weight: float = 1.0,
    content_weight: float = 0.0,
    geometry_weight: float = 1.0,
    existence_weight: float = 1.0,
    probability_floor: float = 1e-8,
) -> ObjectAssociationProblem:
    """Build a valid-row association energy without inventing calibration.

    Miss and birth costs are required inputs from a separately tested
    calibrator. This function deliberately does not hide a fixed deletion or
    birth threshold inside the association solver.
    """

    if predicted.address.shape[1] != observations.address.shape[1]:
        raise ValueError("predicted and observation address widths must match")
    if predicted.dynamic_width != observations.dynamic_width:
        raise ValueError("predicted and observation dynamic widths must match")
    if existence_weight < 0.0 or not np.isfinite(existence_weight):
        raise ValueError("existence_weight must be finite and non-negative")
    if probability_floor <= 0.0 or probability_floor >= 1.0:
        raise ValueError("probability_floor must lie strictly between zero and one")
    block_weights = np.asarray(
        (address_weight, content_weight, geometry_weight),
        dtype=np.float64,
    )
    if not np.isfinite(block_weights).all() or (block_weights < 0.0).any():
        raise ValueError("association block weights must be finite and non-negative")
    if block_weights.sum() <= 0.0:
        raise ValueError("association block weights must have positive total mass")

    miss = np.asarray(unmatched_prior_cost, dtype=np.float64)
    birth = np.asarray(birth_observation_cost, dtype=np.float64)
    if miss.shape != (predicted.capacity,) or not np.isfinite(miss).all():
        raise ValueError("unmatched_prior_cost must be finite and cover posterior capacity")
    if birth.shape != (observations.address.shape[0],) or not np.isfinite(birth).all():
        raise ValueError("birth_observation_cost must be finite and cover observation capacity")

    prior_indices = np.flatnonzero(predicted.valid).astype(np.int64)
    observation_indices = np.flatnonzero(observations.valid).astype(np.int64)
    address_cost = pairwise_cosine_distance(
        predicted.address[prior_indices],
        observations.address[observation_indices],
    )
    content_cost = pairwise_feature_cosine_distance(
        predicted.content[prior_indices],
        observations.content[observation_indices],
    )
    geometry_cost = pairwise_diagonal_gaussian_nll(
        predicted.geometry[prior_indices],
        predicted.geometry_covariance_diag[prior_indices],
        observations.geometry[observation_indices],
        observations.geometry_covariance_diag[observation_indices],
    )
    real = (
        address_weight * address_cost
        + content_weight * content_cost
        + geometry_weight * geometry_cost
    ) / block_weights.sum()
    observation_existence = observations.existence[observation_indices]
    real = (
        real
        - existence_weight * np.log(np.maximum(observation_existence, probability_floor))[None, :]
    )

    return ObjectAssociationProblem(
        prior_indices=prior_indices,
        observation_indices=observation_indices,
        real_cost=real,
        unmatched_prior_cost=miss[prior_indices],
        birth_observation_cost=birth[observation_indices],
    )


def predict_object_belief(
    prior: ObjectBeliefSet,
    dynamic_residual_delta: FloatArray,
    process_variance: FloatArray,
    survival_probability: FloatArray,
    conditional_visibility_probability: FloatArray,
    *,
    delta_t_s: float,
) -> ObjectBeliefSet:
    """Advance one posterior set without observations.

    Address means are deterministic identity keys on the unit sphere and are
    intentionally not action-drifted. Content is a deterministic descriptor;
    geometry alone carries Gaussian covariance.
    """

    dynamic_width = prior.content.shape[1] + prior.geometry.shape[1]
    if dynamic_residual_delta.shape != (prior.capacity, dynamic_width):
        raise ValueError("dynamic_residual_delta must cover content + geometry")
    if process_variance.shape != prior.geometry_covariance_diag.shape:
        raise ValueError("process_variance must cover geometry")
    survival = np.asarray(survival_probability)
    conditional_visibility = np.asarray(conditional_visibility_probability)
    for name, value in {
        "dynamic_residual_delta": dynamic_residual_delta,
        "process_variance": process_variance,
        "survival_probability": survival,
        "conditional_visibility_probability": conditional_visibility,
    }.items():
        _require_finite(name, value)
    if survival.shape != (prior.capacity,) or conditional_visibility.shape != (prior.capacity,):
        raise ValueError("survival and visibility probabilities must cover posterior capacity")
    if ((survival < 0.0) | (survival > 1.0)).any() or (
        (conditional_visibility < 0.0) | (conditional_visibility > 1.0)
    ).any():
        raise ValueError("survival and visibility probabilities must lie in [0, 1]")
    if isinstance(delta_t_s, bool) or not np.isfinite(delta_t_s) or delta_t_s <= 0.0:
        raise ValueError("delta_t_s must be finite and positive")

    content_width = prior.content.shape[1]
    predicted_content = prior.content + dynamic_residual_delta[:, :content_width]
    predicted_geometry, predicted_covariance = diagonal_gaussian_predict(
        prior.geometry,
        prior.geometry_covariance_diag,
        dynamic_residual_delta[:, content_width:],
        process_variance,
    )

    valid = prior.valid
    predicted_content = np.where(valid[:, None], predicted_content, 0.0)
    predicted_geometry = np.where(valid[:, None], predicted_geometry, 0.0)
    predicted_covariance = np.where(valid[:, None], predicted_covariance, 0.0)
    existence = np.where(valid, prior.existence * survival, 0.0)
    visibility = np.where(valid, existence * conditional_visibility, 0.0)
    measurement_age_s = np.where(valid, prior.measurement_age_s + delta_t_s, 0.0)
    age = np.where(valid, prior.age + 1, 0)
    return ObjectBeliefSet(
        address=prior.address.copy(),
        content=predicted_content,
        geometry=predicted_geometry,
        geometry_covariance_diag=predicted_covariance,
        existence=existence,
        visibility=visibility,
        measurement_age_s=measurement_age_s,
        valid=valid.copy(),
        age=age,
    )


def _require_probability_vector(name: str, value: FloatArray, length: int) -> FloatArray:
    result = np.asarray(value)
    _require_finite(name, result)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}")
    if ((result < 0.0) | (result > 1.0)).any():
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def update_object_posterior(
    predicted: ObjectBeliefSet,
    observations: ObjectObservationSet,
    problem: ObjectAssociationProblem,
    association: AssociationResult,
    posterior_prior_existence: FloatArray,
    birth_existence: FloatArray,
    retain_unmatched_prior: NDArray[np.bool_],
) -> PosteriorUpdate:
    """Apply matched correction and explicit calibrated lifecycle decisions.

    The caller owns survival/death calibration. This function never converts a
    cosine threshold into a lifetime policy and never evicts a live object to
    hide capacity overflow.
    """

    prior_existence = _require_probability_vector(
        "posterior_prior_existence",
        posterior_prior_existence,
        predicted.capacity,
    )
    observation_capacity = observations.address.shape[0]
    new_existence = _require_probability_vector(
        "birth_existence",
        birth_existence,
        observation_capacity,
    )
    retain = np.asarray(retain_unmatched_prior)
    if retain.dtype != np.bool_ or retain.shape != (predicted.capacity,):
        raise ValueError("retain_unmatched_prior must be a posterior-capacity bool vector")
    if retain[~predicted.valid].any():
        raise ValueError("unused prior rows cannot be retained")

    compact_prior_count = problem.prior_indices.shape[0]
    compact_observation_count = problem.observation_indices.shape[0]
    if association.prior_to_observation.shape != (compact_prior_count,):
        raise ValueError("association prior map does not match the compact problem")
    if association.observation_to_prior.shape != (compact_observation_count,):
        raise ValueError("association observation map does not match the compact problem")
    if not np.array_equal(problem.prior_indices, np.flatnonzero(predicted.valid)):
        raise ValueError("association problem prior rows do not match the predicted belief")
    if not np.array_equal(problem.observation_indices, np.flatnonzero(observations.valid)):
        raise ValueError("association problem observation rows do not match observations")

    dtype = np.result_type(predicted.address.dtype, observations.address.dtype)
    state = np.zeros((predicted.capacity, predicted.state_width), dtype=dtype)
    geometry_covariance = np.zeros(
        (predicted.capacity, predicted.geometry.shape[1]),
        dtype=dtype,
    )
    existence = np.zeros(predicted.capacity, dtype=dtype)
    visibility = np.zeros(predicted.capacity, dtype=dtype)
    measurement_age_s = np.zeros(predicted.capacity, dtype=dtype)
    valid = np.zeros(predicted.capacity, dtype=np.bool_)
    age = np.zeros(predicted.capacity, dtype=np.int64)
    innovation = np.zeros((predicted.capacity, predicted.dynamic_width), dtype=dtype)
    event_type = np.full(predicted.capacity, UNUSED_EVENT, dtype=np.int64)
    posterior_to_observation = np.full(predicted.capacity, -1, dtype=np.int64)
    observation_to_posterior = np.full(observation_capacity, -1, dtype=np.int64)

    predicted_state = pack_state(predicted)
    observation_state = pack_state(observations)
    predicted_dynamic = pack_dynamic(predicted)
    observation_dynamic = pack_dynamic(observations)
    address_width = predicted.address.shape[1]
    for local_prior, posterior_row in enumerate(problem.prior_indices):
        local_observation = association.prior_to_observation[local_prior]
        if local_observation >= 0:
            observation_row = problem.observation_indices[local_observation]
            corrected = diagonal_gaussian_correct(
                predicted.geometry[[posterior_row]],
                predicted.geometry_covariance_diag[[posterior_row]],
                observations.geometry[[observation_row]],
                observations.geometry_covariance_diag[[observation_row]],
            )
            state[posterior_row, :address_width] = predicted.address[posterior_row]
            content_end = address_width + predicted.content.shape[1]
            state[posterior_row, address_width:content_end] = observations.content[observation_row]
            state[posterior_row, content_end:] = corrected.mean[0]
            geometry_covariance[posterior_row] = corrected.covariance_diag[0]
            innovation[posterior_row] = (
                observation_dynamic[observation_row] - predicted_dynamic[posterior_row]
            )
            existence[posterior_row] = prior_existence[posterior_row]
            if existence[posterior_row] <= 0.0:
                raise ValueError("a matched posterior row must have positive existence")
            visibility[posterior_row] = min(
                existence[posterior_row], observations.existence[observation_row]
            )
            measurement_age_s[posterior_row] = 0.0
            valid[posterior_row] = True
            age[posterior_row] = predicted.age[posterior_row]
            event_type[posterior_row] = MATCH_EVENT
            posterior_to_observation[posterior_row] = observation_row
            observation_to_posterior[observation_row] = posterior_row
        elif retain[posterior_row]:
            state[posterior_row] = predicted_state[posterior_row]
            geometry_covariance[posterior_row] = predicted.geometry_covariance_diag[posterior_row]
            existence[posterior_row] = prior_existence[posterior_row]
            if existence[posterior_row] <= 0.0:
                raise ValueError("a retained posterior row must have positive existence")
            visibility[posterior_row] = 0.0
            measurement_age_s[posterior_row] = predicted.measurement_age_s[posterior_row]
            valid[posterior_row] = True
            age[posterior_row] = predicted.age[posterior_row]
            event_type[posterior_row] = MISS_EVENT
        else:
            event_type[posterior_row] = DEATH_EVENT

    birth_rows = problem.observation_indices[association.birth_observation_indices]
    free_rows = list(np.flatnonzero(~valid))
    if birth_rows.shape[0] > len(free_rows):
        raise PosteriorCapacityError(
            f"{birth_rows.shape[0]} births exceed {len(free_rows)} free posterior rows"
        )
    allocated_rows = free_rows[: birth_rows.shape[0]]
    for observation_row, posterior_row in zip(birth_rows, allocated_rows, strict=True):
        if new_existence[observation_row] <= 0.0:
            raise ValueError("a birth observation must have positive posterior existence")
        state[posterior_row] = observation_state[observation_row]
        geometry_covariance[posterior_row] = observations.geometry_covariance_diag[observation_row]
        existence[posterior_row] = new_existence[observation_row]
        visibility[posterior_row] = min(
            new_existence[observation_row], observations.existence[observation_row]
        )
        measurement_age_s[posterior_row] = 0.0
        valid[posterior_row] = True
        age[posterior_row] = 0
        event_type[posterior_row] = BIRTH_EVENT
        posterior_to_observation[posterior_row] = observation_row
        observation_to_posterior[observation_row] = posterior_row

    address_end = predicted.address.shape[1]
    content_end = address_end + predicted.content.shape[1]
    belief = ObjectBeliefSet(
        address=state[:, :address_end],
        content=state[:, address_end:content_end],
        geometry=state[:, content_end:],
        geometry_covariance_diag=geometry_covariance,
        existence=existence,
        visibility=visibility,
        measurement_age_s=measurement_age_s,
        valid=valid,
        age=age,
    )
    return PosteriorUpdate(
        belief=belief,
        innovation=innovation,
        event_type=event_type,
        posterior_to_observation=posterior_to_observation,
        observation_to_posterior=observation_to_posterior,
    )
