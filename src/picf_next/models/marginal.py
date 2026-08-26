"""Batched probabilistic primitives for one marginal object-belief filter.

These operations contain no learned parameters, task labels or benchmark
thresholds. Association is one null-augmented Sinkhorn variational posterior
over partial matchings. Gaussian fusion uses Stone Soup's equal-weight
Chernoff/covariance-intersection rule because adjacent neural observations
have unknown cross-covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_SINKHORN_ITERATIONS = 100
# A partial matching between one prior and one observation has two outcomes.
# Its dustbin-augmented 2x2 transport plan repeats both probabilities twice,
# so component-wise transport entropy is twice the Bernoulli entropy. One half
# is therefore the unique structural correction that recovers sigmoid(log_odds)
# exactly in the atomic case; it is not a dataset-tuned temperature.
PARTIAL_MATCHING_ENTROPY_WEIGHT = 0.5
EQUAL_CHERNOFF_WEIGHT = 0.5
# Under equal false-presence and false-absence cost, a Bernoulli component is
# part of the MAP finite set iff its posterior existence exceeds one half.
# This is a decision-theoretic boundary, not a dataset-tuned detection cutoff.
MAP_PRESENCE_PROBABILITY = 0.5


@dataclass(frozen=True, slots=True)
class MarginalAssociation:
    """Null-augmented one-to-one association marginals."""

    match_probability: torch.Tensor
    null_probability: torch.Tensor
    unexplained_observation_probability: torch.Tensor
    convergence_residual: torch.Tensor


@dataclass(frozen=True, slots=True)
class DiagonalGaussianFusion:
    """One diagonal equal-source covariance-intersection result."""

    mean: torch.Tensor
    covariance_diag: torch.Tensor
    observation_gain: torch.Tensor
    innovation: torch.Tensor


def _log_sinkhorn_plan(
    log_scores: torch.Tensor,
    log_row_mass: torch.Tensor,
    log_column_mass: torch.Tensor,
    *,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a scaled transport plan and one-step fixed-point residual.

    The alternating log normalization follows SuperGlue's released optimal
    transport implementation. A final extra step measures convergence in plan
    space, avoiding the arbitrary additive gauge of the dual variables.
    """

    log_v = torch.zeros_like(log_column_mass)
    log_u = torch.zeros_like(log_row_mass)
    for _ in range(iterations):
        log_u = log_row_mass - torch.logsumexp(
            log_scores + log_v.unsqueeze(1),
            dim=2,
        )
        log_v = log_column_mass - torch.logsumexp(
            log_scores + log_u.unsqueeze(2),
            dim=1,
        )

    log_plan = log_scores + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    next_log_u = log_row_mass - torch.logsumexp(
        log_scores + log_v.unsqueeze(1),
        dim=2,
    )
    next_log_v = log_column_mass - torch.logsumexp(
        log_scores + next_log_u.unsqueeze(2),
        dim=1,
    )
    next_log_plan = log_scores + next_log_u.unsqueeze(2) + next_log_v.unsqueeze(1)
    plan = next_log_plan.exp()
    residual = (plan - log_plan.exp()).abs().amax()
    return plan, residual


def marginal_one_to_one_association(
    log_edge_odds: torch.Tensor,
    valid_prior: torch.Tensor,
    *,
    iterations: int = DEFAULT_SINKHORN_ITERATIONS,
) -> MarginalAssociation:
    """Infer scalable variational marginals over null-augmented matchings.

    ``log_edge_odds[b, i, j]`` is the real-edge mass divided by both the
    existing-track null mass and observation-origin null mass. One shared
    dustbin row and column represent unexplained observations and unmatched
    priors without assuming the number of real matches. Invalid prior rows are
    exact null hypotheses. The returned real matrix lies in the partial
    matching polytope; row and column complements are the two null marginals.
    """

    if log_edge_odds.ndim != 3:
        raise ValueError("log_edge_odds must be batch-by-prior-by-observation")
    if valid_prior.shape != log_edge_odds.shape[:2] or valid_prior.dtype != torch.bool:
        raise ValueError("valid_prior must be bool batch-by-prior")
    if valid_prior.device != log_edge_odds.device:
        raise ValueError("association tensors must share one device")
    if not torch.is_floating_point(log_edge_odds):
        raise ValueError("log_edge_odds must be floating point")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if log_edge_odds.shape[1] <= 0 or log_edge_odds.shape[2] <= 0:
        raise ValueError("association dimensions must be nonempty")

    work = (log_edge_odds.float() / PARTIAL_MATCHING_ENTROPY_WEIGHT).masked_fill(
        ~valid_prior.unsqueeze(-1), -torch.inf
    )
    batch_size, prior_count, observation_count = work.shape
    zero = work.new_zeros(())
    dustbin_column = zero.expand(batch_size, prior_count, 1)
    dustbin_row = zero.expand(batch_size, 1, observation_count + 1)
    augmented_scores = torch.cat(
        (
            torch.cat((work, dustbin_column), dim=2),
            dustbin_row,
        ),
        dim=1,
    )

    total_mass = float(prior_count + observation_count)
    log_normalized_unit = -work.new_tensor(total_mass).log()
    log_row_mass = torch.cat(
        (
            log_normalized_unit.expand(batch_size, prior_count),
            (work.new_tensor(float(observation_count)).log() + log_normalized_unit).expand(
                batch_size, 1
            ),
        ),
        dim=1,
    )
    log_column_mass = torch.cat(
        (
            log_normalized_unit.expand(batch_size, observation_count),
            (work.new_tensor(float(prior_count)).log() + log_normalized_unit).expand(batch_size, 1),
        ),
        dim=1,
    )
    normalized_plan, residual = _log_sinkhorn_plan(
        augmented_scores,
        log_row_mass,
        log_column_mass,
        iterations=iterations,
    )
    transport = normalized_plan * total_mass
    residual = residual * total_mass
    match_probability = transport[:, :prior_count, :observation_count]
    match_probability = match_probability * valid_prior.unsqueeze(-1)

    # Finite IPF is already close to the transport polytope. These two
    # monotone capacity projections make the real partial matching exactly
    # feasible without inventing matches or changing their relative ordering.
    row_scale = match_probability.sum(dim=2, keepdim=True).clamp_min(1.0)
    match_probability = match_probability / row_scale
    column_scale = match_probability.sum(dim=1, keepdim=True).clamp_min(1.0)
    match_probability = match_probability / column_scale
    null_probability = (1.0 - match_probability.sum(dim=2)).clamp(min=0.0, max=1.0)
    null_probability = torch.where(valid_prior, null_probability, torch.ones_like(null_probability))
    unexplained = (1.0 - match_probability.sum(dim=1)).clamp(min=0.0, max=1.0)
    return MarginalAssociation(
        match_probability=match_probability,
        null_probability=null_probability,
        unexplained_observation_probability=unexplained,
        convergence_residual=residual,
    )


def equal_weight_covariance_intersection(
    prior_mean: torch.Tensor,
    prior_covariance_diag: torch.Tensor,
    observation_mean: torch.Tensor,
    observation_covariance_diag: torch.Tensor,
    *,
    minimum_variance: float,
) -> DiagonalGaussianFusion:
    """Fuse Gaussian states without assuming their errors are independent."""

    shapes = {
        prior_mean.shape,
        prior_covariance_diag.shape,
        observation_mean.shape,
        observation_covariance_diag.shape,
    }
    if len(shapes) != 1:
        raise ValueError("covariance-intersection tensors must share one shape")
    tensors = (
        prior_mean,
        prior_covariance_diag,
        observation_mean,
        observation_covariance_diag,
    )
    if any(
        not torch.is_floating_point(value)
        or value.device != prior_mean.device
        or value.dtype != prior_mean.dtype
        for value in tensors
    ):
        raise ValueError("covariance-intersection tensors must share floating dtype and device")
    if not isinstance(minimum_variance, float) or minimum_variance <= 0.0:
        raise ValueError("minimum_variance must be a positive float")

    prior_variance = prior_covariance_diag.float().clamp_min(minimum_variance)
    observation_variance = observation_covariance_diag.float().clamp_min(minimum_variance)
    prior_precision = EQUAL_CHERNOFF_WEIGHT / prior_variance
    observation_precision = EQUAL_CHERNOFF_WEIGHT / observation_variance
    total_precision = prior_precision + observation_precision
    covariance = total_precision.reciprocal().clamp_min(minimum_variance)
    observation_gain = observation_precision / total_precision
    innovation = observation_mean.float() - prior_mean.float()
    mean = prior_mean.float() + observation_gain * innovation
    return DiagonalGaussianFusion(
        mean=mean.to(prior_mean.dtype),
        covariance_diag=covariance.to(prior_covariance_diag.dtype),
        observation_gain=observation_gain.to(prior_mean.dtype),
        innovation=innovation.to(prior_mean.dtype),
    )


def probabilistic_data_association_moments(
    prior_mean: torch.Tensor,
    prior_covariance_diag: torch.Tensor,
    match_mean: torch.Tensor,
    match_covariance_diag: torch.Tensor,
    null_weight: torch.Tensor,
    match_weight: torch.Tensor,
    *,
    minimum_variance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a null-plus-match Gaussian mixture to its first two moments."""

    if prior_mean.shape != prior_covariance_diag.shape:
        raise ValueError("prior mean and covariance must share one shape")
    expected_component_shape = (
        *prior_mean.shape[:-1],
        match_weight.shape[-1],
        prior_mean.shape[-1],
    )
    if (
        match_mean.shape != expected_component_shape
        or match_covariance_diag.shape != expected_component_shape
    ):
        raise ValueError("match Gaussian components do not align with prior and weights")
    if null_weight.shape != prior_mean.shape[:-1]:
        raise ValueError("null_weight must align with prior components")
    if match_weight.shape[:-1] != prior_mean.shape[:-1]:
        raise ValueError("match_weight must align with prior components")
    total_weight = null_weight + match_weight.sum(dim=-1)
    tolerance = 32.0 * torch.finfo(match_weight.dtype).eps
    if not torch.allclose(
        total_weight.float(),
        torch.ones_like(total_weight, dtype=torch.float32),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("conditional mixture weights must sum to one")

    null = null_weight.unsqueeze(-1)
    matches = match_weight.unsqueeze(-1)
    mean = null * prior_mean + (matches * match_mean).sum(dim=-2)
    null_second = prior_covariance_diag + (prior_mean - mean).square()
    match_second = match_covariance_diag + (match_mean - mean.unsqueeze(-2)).square()
    covariance = null * null_second + (matches * match_second).sum(dim=-2)
    return mean, covariance.clamp_min(minimum_variance)
