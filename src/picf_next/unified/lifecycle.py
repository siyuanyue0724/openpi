"""Threshold-free lifecycle and correlation-safe evidence operators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from picf_next.unified.state import BIRTH, CONTINUE, EMPTY, LIFECYCLE_MODES


@dataclass(frozen=True, slots=True)
class FootprintEvidence:
    responsibilities: torch.Tensor
    context_probability: torch.Tensor
    support: torch.Tensor
    robust_log_likelihood_ratio: torch.Tensor
    valid_footprint_mass: torch.Tensor


@dataclass(frozen=True, slots=True)
class GaussianCIFusion:
    mean: torch.Tensor
    information: torch.Tensor
    information_vector: torch.Tensor


def deterministic_logdet_ci_weights(
    prior_information: torch.Tensor,
    information_increments: torch.Tensor,
    available: torch.Tensor,
    *,
    iterations: int = 8,
    step_size: float = 0.25,
    information_floor: float = 1e-5,
) -> torch.Tensor:
    """Solve the small covariance-intersection simplex without a neural scorer.

    The first returned coordinate is the unspent/prior weight.  Remaining
    coordinates weight modality information increments.  Fixed-step analytic
    exponentiated-gradient ascent maximizes posterior log-determinant.  The
    weights are deliberately detached: they implement a deterministic fusion
    rule, while gradients still reach every increment selected by that rule.
    """

    if prior_information.ndim < 2 or prior_information.shape[-1] != prior_information.shape[-2]:
        raise ValueError("prior_information must end in a square matrix")
    if information_increments.shape[:-3] != prior_information.shape[:-2]:
        raise ValueError("information increment batch/row axes do not match prior")
    if information_increments.shape[-2:] != prior_information.shape[-2:]:
        raise ValueError("information increments and prior geometry widths do not match")
    modalities = information_increments.shape[-3]
    if available.shape != (*prior_information.shape[:-2], modalities):
        raise ValueError("available must have one value per modality opinion")
    if available.dtype != torch.bool:
        raise TypeError("available must be boolean")
    if type(iterations) is not int:
        raise TypeError("iterations must be a Python int")
    controls = (step_size, information_floor)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in controls):
        raise TypeError("CI solver step_size and information_floor must be real-valued")
    if iterations <= 0 or any(not math.isfinite(value) or value <= 0 for value in controls):
        raise ValueError("CI solver controls must be positive")
    tensors = (prior_information, information_increments)
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("CI information tensors must be floating point")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("CI information tensors must be finite")
    if any(value.device != prior_information.device for value in (*tensors, available)):
        raise ValueError("CI solver tensors must share one device")
    if not torch.allclose(prior_information, prior_information.transpose(-1, -2), atol=1e-5):
        raise ValueError("prior information must be symmetric")
    if not torch.allclose(
        information_increments,
        information_increments.transpose(-1, -2),
        atol=1e-5,
    ):
        raise ValueError("information increments must be symmetric")

    with torch.no_grad():
        prior = prior_information.float()
        increments = information_increments.float()
        valid_opinions = torch.cat((torch.ones_like(available[..., :1]), available), dim=-1)
        logits = torch.zeros_like(valid_opinions, dtype=prior.dtype)
        identity = torch.eye(prior.shape[-1], device=prior.device, dtype=prior.dtype)
        for _ in range(iterations):
            masked_logits = logits.masked_fill(~valid_opinions, -torch.inf)
            weights = torch.softmax(masked_logits, dim=-1)
            information = prior + torch.einsum("...m,...mij->...ij", weights[..., 1:], increments)
            inverse = torch.linalg.inv(information + information_floor * identity)
            modality_gain = torch.einsum("...ij,...mji->...m", inverse, increments)
            gain = torch.cat((torch.zeros_like(modality_gain[..., :1]), modality_gain), dim=-1)
            centered = gain - (weights * gain).sum(dim=-1, keepdim=True)
            logits = logits + step_size * weights * centered
        return torch.softmax(logits.masked_fill(~valid_opinions, -torch.inf), dim=-1).to(
            prior_information
        )


def categorical_lifecycle_prior(
    continuation_probability: torch.Tensor,
    birth_hazard: torch.Tensor,
) -> torch.Tensor:
    """Return normalized log ``continue/birth/empty`` predictive mass."""

    if continuation_probability.shape != birth_hazard.shape:
        raise ValueError("continuation_probability and birth_hazard must match")
    if not continuation_probability.is_floating_point() or not birth_hazard.is_floating_point():
        raise TypeError("lifecycle probabilities must be floating point")
    if continuation_probability.device != birth_hazard.device:
        raise ValueError("lifecycle probabilities must share one device")
    if not torch.isfinite(continuation_probability).all() or not torch.isfinite(birth_hazard).all():
        raise ValueError("lifecycle probabilities must be finite")
    if ((continuation_probability < 0) | (continuation_probability > 1)).any():
        raise ValueError("continuation_probability must lie in [0, 1]")
    if ((birth_hazard < 0) | (birth_hazard > 1)).any():
        raise ValueError("birth_hazard must lie in [0, 1]")
    continuation = continuation_probability
    vacant = 1.0 - continuation
    probabilities = torch.stack(
        (continuation, vacant * birth_hazard, vacant * (1.0 - birth_hazard)),
        dim=-1,
    )
    tiny = torch.finfo(probabilities.dtype).tiny
    logits = probabilities.clamp_min(tiny).log()
    return torch.log_softmax(logits, dim=-1)


def reliability_simplex(
    prior_reliability: torch.Tensor,
    modality_reliability: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    """Normalize fixed reliabilities over the prior and modalities present."""

    if modality_reliability.ndim != 1:
        raise ValueError("modality_reliability must have shape [modalities]")
    if available.shape[-1] != modality_reliability.numel() or available.dtype != torch.bool:
        raise ValueError("available must be boolean with one entry per modality")
    if prior_reliability.numel() != 1:
        raise ValueError("prior_reliability must be scalar")
    if not prior_reliability.is_floating_point() or not modality_reliability.is_floating_point():
        raise TypeError("reliabilities must be floating point")
    if (
        not torch.isfinite(prior_reliability).all()
        or not torch.isfinite(modality_reliability).all()
    ):
        raise ValueError("reliabilities must be finite")
    if prior_reliability.item() <= 0 or (modality_reliability < 0).any():
        raise ValueError("reliabilities must be non-negative and prior must be positive")
    dtype = modality_reliability.dtype
    prior = prior_reliability.to(device=available.device, dtype=dtype).expand(
        *available.shape[:-1], 1
    )
    modalities = modality_reliability.to(available.device) * available.to(dtype)
    raw = torch.cat((prior, modalities), dim=-1)
    return raw / raw.sum(dim=-1, keepdim=True)


def logarithmic_lifecycle_pool(
    prior_log_probs: torch.Tensor,
    incremental_log_bayes_factors: torch.Tensor,
    opinion_weights: torch.Tensor,
    *,
    available: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse categorical opinions while counting the common prior exactly once.

    ``opinion_weights`` contains modality weights only.  The corresponding prior
    weight is implicit; mathematically the common prior remains once and each
    modality contributes its tempered incremental log Bayes factor.
    """

    if prior_log_probs.shape[-1] != LIFECYCLE_MODES:
        raise ValueError("prior_log_probs must end in three lifecycle modes")
    if incremental_log_bayes_factors.shape[:-2] != prior_log_probs.shape[:-1]:
        raise ValueError("lifecycle opinion batch/row shapes do not match")
    if incremental_log_bayes_factors.shape[-1] != LIFECYCLE_MODES:
        raise ValueError("incremental log Bayes factors must have three modes")
    modalities = incremental_log_bayes_factors.shape[-2]
    target_shape = (*prior_log_probs.shape[:-1], modalities)
    tensors = (prior_log_probs, incremental_log_bayes_factors, opinion_weights)
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("lifecycle pooling tensors must be floating point")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("lifecycle pooling tensors must be finite")
    if any(value.device != prior_log_probs.device for value in tensors):
        raise ValueError("lifecycle pooling tensors must share one device")
    normalization = torch.logsumexp(prior_log_probs.float(), dim=-1)
    if not torch.allclose(normalization, torch.zeros_like(normalization), atol=1e-5):
        raise ValueError("prior_log_probs must be normalized")
    weights = torch.broadcast_to(opinion_weights, target_shape).to(prior_log_probs)
    if (weights < 0).any() or (weights.sum(dim=-1) > 1.0 + 1e-6).any():
        raise ValueError("modality opinion weights must be non-negative and sum to at most one")
    if available is not None:
        available_weights = torch.broadcast_to(available, target_shape)
        if available_weights.dtype != torch.bool:
            raise TypeError("available must be boolean")
        if available_weights.device != prior_log_probs.device:
            raise ValueError("available and lifecycle opinions must share one device")
        weights = weights * available_weights.to(weights.dtype)
    # Fix the additive gauge at empty, so a constant/uninformative opinion is zero.
    factors = incremental_log_bayes_factors - incremental_log_bayes_factors[..., EMPTY:]
    fused_logits = prior_log_probs + (weights.unsqueeze(-1) * factors).sum(dim=-2)
    return torch.log_softmax(fused_logits, dim=-1)


def nonempty_probability(lifecycle_log_probs: torch.Tensor) -> torch.Tensor:
    if lifecycle_log_probs.shape[-1] != LIFECYCLE_MODES:
        raise ValueError("lifecycle_log_probs must end in three modes")
    if not lifecycle_log_probs.is_floating_point():
        raise TypeError("lifecycle_log_probs must be floating point")
    if not torch.isfinite(lifecycle_log_probs).all():
        raise ValueError("lifecycle_log_probs must be finite")
    normalization = torch.logsumexp(lifecycle_log_probs.float(), dim=-1)
    if not torch.allclose(normalization, torch.zeros_like(normalization), atol=1e-5):
        raise ValueError("lifecycle_log_probs must be normalized")
    probabilities = lifecycle_log_probs.exp()
    return probabilities[..., CONTINUE] + probabilities[..., BIRTH]


def posterior_expected_age(
    prior_expected_age: torch.Tensor,
    lifecycle_log_probs: torch.Tensor,
    elapsed_time: torch.Tensor | float,
) -> torch.Tensor:
    """Update expected age by marginalizing lifecycle instead of selecting a mode."""

    if lifecycle_log_probs.shape != (*prior_expected_age.shape, LIFECYCLE_MODES):
        raise ValueError("lifecycle_log_probs must match prior age and end in three modes")
    if not prior_expected_age.is_floating_point() or not lifecycle_log_probs.is_floating_point():
        raise TypeError("posterior age tensors must be floating point")
    if prior_expected_age.device != lifecycle_log_probs.device:
        raise ValueError("posterior age tensors must share one device")
    if (
        not torch.isfinite(prior_expected_age).all()
        or not torch.isfinite(lifecycle_log_probs).all()
    ):
        raise ValueError("posterior age inputs must be finite")
    if (prior_expected_age < 0).any():
        raise ValueError("prior expected age must be non-negative")
    normalization = torch.logsumexp(lifecycle_log_probs.float(), dim=-1)
    if not torch.allclose(normalization, torch.zeros_like(normalization), atol=1e-5):
        raise ValueError("lifecycle_log_probs must be normalized")
    if isinstance(elapsed_time, bool):
        raise TypeError("elapsed_time must be real-valued")
    if isinstance(elapsed_time, torch.Tensor):
        if not elapsed_time.is_floating_point():
            raise TypeError("elapsed_time tensor must be floating point")
        if elapsed_time.device != prior_expected_age.device:
            raise ValueError("elapsed_time and posterior age must share one device")
        elapsed = elapsed_time.to(prior_expected_age)
    else:
        if not isinstance(elapsed_time, (int, float)):
            raise TypeError("elapsed_time must be real-valued")
        elapsed = torch.as_tensor(elapsed_time, device=prior_expected_age.device).to(
            prior_expected_age
        )
    if not torch.isfinite(elapsed).all() or (elapsed < 0).any():
        raise ValueError("elapsed_time must be finite and non-negative")
    try:
        broadcast_shape = torch.broadcast_shapes(prior_expected_age.shape, elapsed.shape)
    except RuntimeError as error:
        raise ValueError("elapsed_time is not broadcastable to prior expected age") from error
    if broadcast_shape != prior_expected_age.shape:
        raise ValueError("elapsed_time cannot expand the prior expected-age shape")
    probability = lifecycle_log_probs.exp()
    nonempty = probability[..., CONTINUE] + probability[..., BIRTH]
    continuation_given_nonempty = probability[..., CONTINUE] / nonempty.clamp_min(
        torch.finfo(probability.dtype).tiny
    )
    return continuation_given_nonempty * (prior_expected_age + elapsed)


def footprint_evidence(
    assignment_logits: torch.Tensor,
    footprint: torch.Tensor,
    valid: torch.Tensor,
    *,
    robust_clip: float = 8.0,
) -> FootprintEvidence:
    """Compute context-normalized ownership and tokenizer-invariant support.

    The final assignment logit is context/null.  Footprints are measures, not
    token counts, so splitting a token into identical sub-tokens conserves both
    normalized support and the robust likelihood statistic.
    """

    if assignment_logits.ndim != 3 or assignment_logits.shape[-1] < 2:
        raise ValueError("assignment_logits must have shape [batch, tokens, beliefs+1]")
    if footprint.shape != assignment_logits.shape[:2] or valid.shape != footprint.shape:
        raise ValueError("footprint and valid must match assignment token axes")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be boolean")
    if not assignment_logits.is_floating_point() or not footprint.is_floating_point():
        raise TypeError("assignment logits and footprint must be floating point")
    if assignment_logits.device != footprint.device or valid.device != footprint.device:
        raise ValueError("assignment evidence tensors must share one device")
    if not torch.isfinite(assignment_logits).all():
        raise ValueError("assignment_logits must be finite")
    if (footprint < 0).any() or not torch.isfinite(footprint).all():
        raise ValueError("footprint must be finite and non-negative")
    if isinstance(robust_clip, bool) or not isinstance(robust_clip, (int, float)):
        raise TypeError("robust_clip must be real-valued")
    if not math.isfinite(robust_clip) or robust_clip <= 0:
        raise ValueError("robust_clip must be positive")
    probabilities = torch.softmax(assignment_logits.float(), dim=-1).to(assignment_logits)
    probabilities = probabilities * valid.unsqueeze(-1)
    responsibilities = probabilities[..., :-1]
    context_probability = probabilities[..., -1]
    weighted_footprint = footprint.to(assignment_logits) * valid.to(assignment_logits.dtype)
    valid_mass = weighted_footprint.sum(dim=-1, keepdim=True)
    normalized_footprint = weighted_footprint / valid_mass.clamp_min(
        torch.finfo(weighted_footprint.dtype).tiny
    )
    support = torch.einsum("bn,bnk->bk", normalized_footprint, responsibilities)

    object_probability = torch.softmax(assignment_logits.float(), dim=-1)[..., :-1]
    tiny = torch.finfo(object_probability.dtype).eps
    local_log_odds = torch.logit(object_probability.clamp(tiny, 1.0 - tiny))
    clipped = local_log_odds.clamp(-robust_clip, robust_clip)
    log_weight = torch.where(
        normalized_footprint > 0,
        normalized_footprint.log(),
        torch.full_like(normalized_footprint, -torch.inf),
    )
    robust = torch.logsumexp(log_weight.unsqueeze(-1) + clipped, dim=1)
    robust = torch.where(valid_mass > 0, robust, torch.zeros_like(robust))
    return FootprintEvidence(
        responsibilities=responsibilities,
        context_probability=context_probability,
        support=support,
        robust_log_likelihood_ratio=robust.to(assignment_logits.dtype),
        valid_footprint_mass=valid_mass.squeeze(-1),
    )


def generalized_covariance_intersection(
    prior_mean: torch.Tensor,
    prior_information: torch.Tensor,
    observation_means: torch.Tensor,
    information_increments: torch.Tensor,
    simplex_weights: torch.Tensor,
    *,
    available: torch.Tensor | None = None,
) -> GaussianCIFusion:
    """Fuse correlated Gaussian opinions with the common prior exactly once."""

    if prior_information.shape != (*prior_mean.shape, prior_mean.shape[-1]):
        raise ValueError("prior_information must have shape [..., geometry, geometry]")
    if observation_means.shape[:-2] != prior_mean.shape[:-1]:
        raise ValueError("observation mean batch/row dimensions do not match prior")
    if observation_means.shape[-1] != prior_mean.shape[-1]:
        raise ValueError("observation and prior geometry widths do not match")
    if information_increments.shape != (*observation_means.shape, prior_mean.shape[-1]):
        raise ValueError("information increments have the wrong shape")
    modalities = observation_means.shape[-2]
    target_weight_shape = (*prior_mean.shape[:-1], modalities + 1)
    tensors = (
        prior_mean,
        prior_information,
        observation_means,
        information_increments,
        simplex_weights,
    )
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("Gaussian fusion tensors must be floating point")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("Gaussian fusion tensors must be finite")
    if any(value.device != prior_mean.device for value in tensors):
        raise ValueError("Gaussian fusion tensors must share one device")
    weights = torch.broadcast_to(simplex_weights, target_weight_shape).to(prior_mean)
    if available is not None:
        target_available_shape = (*prior_mean.shape[:-1], modalities)
        available_mask = torch.broadcast_to(available, target_available_shape)
        if available_mask.dtype != torch.bool:
            raise TypeError("available must be boolean")
        if available_mask.device != prior_mean.device:
            raise ValueError("available and Gaussian opinions must share one device")
        weights = torch.cat(
            (weights[..., :1], weights[..., 1:] * available_mask.to(weights.dtype)), dim=-1
        )
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
    if (weights < 0).any() or not torch.allclose(
        weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-5
    ):
        raise ValueError("simplex_weights must be non-negative and sum to one")
    if not torch.allclose(prior_information, prior_information.transpose(-1, -2), atol=1e-5):
        raise ValueError("prior information must be symmetric")
    if not torch.allclose(
        information_increments,
        information_increments.transpose(-1, -2),
        atol=1e-5,
    ):
        raise ValueError("information increments must be symmetric")

    output_dtype = prior_mean.dtype
    prior_mean = prior_mean.float()
    prior_information = prior_information.float()
    observation_means = observation_means.float()
    information_increments = information_increments.float()
    modality_weights = weights[..., 1:].float()
    weighted_increment = torch.einsum(
        "...m,...mij->...ij", modality_weights, information_increments
    )
    information = prior_information + weighted_increment
    prior_information_vector = torch.einsum("...ij,...j->...i", prior_information, prior_mean)
    observation_vectors = torch.einsum(
        "...mij,...mj->...mi", information_increments, observation_means
    )
    information_vector = prior_information_vector + torch.einsum(
        "...m,...mi->...i", modality_weights, observation_vectors
    )
    # Solve in innovation form. It is exactly equivalent to the information-form
    # mean on a nonsingular observed subspace, while the Moore-Penrose solution
    # leaves every unobserved null-space coordinate at its prior mean instead of
    # silently replacing it with zero.
    innovation = observation_means - prior_mean.unsqueeze(-2)
    innovation_vectors = torch.einsum("...mij,...mj->...mi", information_increments, innovation)
    weighted_innovation = torch.einsum("...m,...mi->...i", modality_weights, innovation_vectors)
    correction = torch.einsum(
        "...ij,...j->...i",
        torch.linalg.pinv(information),
        weighted_innovation,
    )
    mean = prior_mean + correction
    return GaussianCIFusion(
        mean=mean.to(output_dtype),
        information=information.to(output_dtype),
        information_vector=information_vector.to(output_dtype),
    )
