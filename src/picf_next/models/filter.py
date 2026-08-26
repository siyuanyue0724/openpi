"""End-to-end task-independent persistent object filter orchestration.

Training-time target matching is deliberately absent. Runtime correspondence is
one differentiable null-augmented variational partial-matching posterior.
Unknown cross-covariance is handled by equal-source covariance intersection,
and only the final finite-capacity/MAP-presence projection is discrete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.posterior import (
    BIRTH_EVENT,
    DEATH_EVENT,
    MATCH_EVENT,
    MISS_EVENT,
    UNUSED_EVENT,
)

from .binding_loss import (
    BindingLossConfig,
    SphericalRelationCalibration,
    spherical_relation_logits,
)
from .discovery import ObjectDiscoveryOutput
from .marginal import (
    MAP_PRESENCE_PROBABILITY,
    MarginalAssociation,
    equal_weight_covariance_intersection,
    marginal_one_to_one_association,
    probabilistic_data_association_moments,
)
from .temporal import (
    ActionConditionedObjectTransition,
    ObjectBeliefBatch,
    ObjectPredictionOutput,
    TemporalFilterConfig,
)


@dataclass(frozen=True, slots=True)
class PersistentFilterOutput:
    prior_prediction: ObjectPredictionOutput
    belief: ObjectBeliefBatch
    innovation: torch.Tensor
    ownership: torch.Tensor
    match_probability: torch.Tensor
    null_probability: torch.Tensor
    birth_probability: torch.Tensor
    born: torch.Tensor
    map_present: torch.Tensor
    address_relation_logit_scale: torch.Tensor
    address_relation_logit_bias: torch.Tensor
    association_convergence_residual: torch.Tensor
    observation_to_posterior: torch.Tensor
    event_type: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObjectActionBank:
    """Address-only keys and dynamic probabilistic values for an action host."""

    address: torch.Tensor
    value: torch.Tensor
    valid: torch.Tensor
    log_prior: torch.Tensor


@dataclass(frozen=True, slots=True)
class _MarginalLifecycle:
    belief: ObjectBeliefBatch
    innovation: torch.Tensor
    ownership: torch.Tensor
    birth_probability: torch.Tensor
    born: torch.Tensor
    map_present: torch.Tensor
    observation_to_posterior: torch.Tensor
    event_type: torch.Tensor


# The pinned official PMBM implementation prunes Bernoulli components at
# 1e-5, separately from its 0.4/0.5 state-extraction threshold. Capacity top-k
# remains the dominant bounded-memory reduction here; this floor only removes
# numerically negligible support.
BERNOULLI_PRUNING_PROBABILITY = 1e-5


def build_object_action_bank(output: PersistentFilterOutput) -> ObjectActionBank:
    """Expose prior/posterior/innovation without promoting dynamics to identity."""

    predicted = output.prior_prediction.belief
    posterior = output.belief
    valid = output.map_present
    predicted_dynamic = torch.cat((predicted.content_mean, predicted.geometry_mean), dim=-1)
    posterior_dynamic = torch.cat((posterior.content_mean, posterior.geometry_mean), dim=-1)
    # A row recycled by a birth has no prior prediction for the new physical
    # object. Reusing the dead row's predicted dynamics would mix two identities
    # in one action token. Initialize the birth's prior view from its own first
    # posterior observation instead.
    predicted_dynamic = torch.where(
        output.born.unsqueeze(-1),
        posterior_dynamic,
        predicted_dynamic,
    )
    # CUDA autocast evaluates log in float32.  Keep that numerically stable
    # calculation, then restore the posterior compute dtype before assembling
    # the typed object bank consumed by the action expert.
    log_covariance = torch.log(
        posterior.geometry_covariance_diag.float().clamp_min(torch.finfo(torch.float32).tiny)
    ).to(posterior_dynamic.dtype)
    predicted_lifecycle = torch.stack(
        (
            predicted.existence,
            predicted.visibility,
        ),
        dim=-1,
    )
    posterior_lifecycle = torch.stack(
        (
            posterior.existence,
            posterior.visibility,
        ),
        dim=-1,
    )
    predicted_lifecycle = torch.where(
        output.born.unsqueeze(-1),
        posterior_lifecycle,
        predicted_lifecycle,
    )
    predicted_lifecycle = predicted_lifecycle.to(predicted_dynamic.dtype)
    posterior_lifecycle = posterior_lifecycle.to(posterior_dynamic.dtype)
    value = torch.cat(
        (
            predicted_dynamic,
            posterior_dynamic,
            log_covariance,
            output.innovation,
            predicted_lifecycle,
            posterior_lifecycle,
        ),
        dim=-1,
    )
    value = value * valid.unsqueeze(-1)
    address = posterior.address_mean * valid.unsqueeze(-1)
    probability_floor = torch.finfo(torch.float32).eps
    log_prior = torch.where(
        valid,
        posterior.existence.float().clamp_min(probability_floor).log(),
        torch.zeros_like(posterior.existence.float()),
    ).to(posterior.address_mean.dtype)
    return ObjectActionBank(
        address=address,
        value=value,
        valid=valid,
        log_prior=log_prior,
    )


class PersistentObjectFilter(nn.Module):
    """Predict and update one finite, marginal persistent-object belief."""

    def __init__(
        self,
        config: TemporalFilterConfig,
        *,
        validate_tensor_values: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(validate_tensor_values, bool):
            raise ValueError("validate_tensor_values must be boolean")
        self.config = config
        self.validate_tensor_values = validate_tensor_values
        self.transition = ActionConditionedObjectTransition(
            config,
            validate_tensor_values=validate_tensor_values,
        )
        self.address_relation = SphericalRelationCalibration(
            BindingLossConfig(
                objective="sigmoid",
                temperature=config.association_address_temperature,
                logit_bias=config.association_address_logit_bias,
            )
        )

    def forward(
        self,
        prior: ObjectBeliefBatch,
        discovery: ObjectDiscoveryOutput,
        previous_executed_action: torch.Tensor,
        delta_t_s: torch.Tensor,
    ) -> PersistentFilterOutput:
        _validate_discovery(
            discovery,
            prior,
            self.config,
            validate_values=self.validate_tensor_values,
        )
        prediction = self.transition(prior, previous_executed_action, delta_t_s)
        association, observation_probability, birth_odds = _marginal_association(
            prediction.belief,
            discovery,
            self.config,
            relation_logit_scale=self.address_relation.logit_scale,
            relation_logit_bias=self.address_relation.logit_bias,
        )
        lifecycle = _assemble_marginal_lifecycle(
            prediction.belief,
            discovery,
            association,
            observation_probability,
            birth_odds,
            self.config,
        )
        return PersistentFilterOutput(
            prior_prediction=prediction,
            belief=lifecycle.belief,
            innovation=lifecycle.innovation,
            ownership=lifecycle.ownership,
            match_probability=association.match_probability,
            null_probability=association.null_probability,
            birth_probability=lifecycle.birth_probability,
            born=lifecycle.born,
            map_present=lifecycle.map_present,
            address_relation_logit_scale=self.address_relation.logit_scale,
            address_relation_logit_bias=self.address_relation.logit_bias,
            association_convergence_residual=association.convergence_residual,
            observation_to_posterior=lifecycle.observation_to_posterior,
            event_type=lifecycle.event_type,
        )


def _marginal_association(
    predicted: ObjectBeliefBatch,
    discovery: ObjectDiscoveryOutput,
    config: TemporalFilterConfig,
    *,
    relation_logit_scale: torch.Tensor | None = None,
    relation_logit_bias: torch.Tensor | None = None,
) -> tuple[MarginalAssociation, torch.Tensor, torch.Tensor]:
    """Return one-to-one association marginals and observation-origin priors.

    The real-edge odds divide the joint match mass by both relevant null
    masses: an existing row's no-detection mass and an observation's
    birth-or-clutter mass.  Geometry uses a conservative innovation covariance
    because consecutive neural observations have unknown, strongly correlated
    error rather than independent sensor noise.
    """

    prior_address = F.normalize(predicted.address_mean.float(), dim=-1)
    observation_address = F.normalize(discovery.address_mean.float(), dim=-1)
    address_cosine = torch.einsum(
        "bkd,bqd->bkq",
        prior_address,
        observation_address,
    ).clamp(min=-1.0, max=1.0)
    relation_supplied = relation_logit_scale is not None, relation_logit_bias is not None
    if any(relation_supplied) and not all(relation_supplied):
        raise ValueError("association relation scale and bias are atomic")
    if relation_logit_scale is not None and relation_logit_bias is not None:
        address_log_likelihood_ratio = spherical_relation_logits(
            address_cosine,
            logit_scale=relation_logit_scale,
            logit_bias=relation_logit_bias,
        )
    else:
        address_log_likelihood_ratio = (
            address_cosine / config.association_address_temperature
            + config.association_address_logit_bias
        )

    geometry_residual = discovery.geometry_mean.float().unsqueeze(
        1
    ) - predicted.geometry_mean.float().unsqueeze(2)
    # For unknown cross-covariance C, 2(P + R) is the conservative quadratic
    # bound induced by ||a-b||^2 <= 2||a||^2 + 2||b||^2.  It deliberately does
    # not pretend that two encodings of adjacent frames are independent.
    innovation_variance = (
        2.0
        * (
            predicted.geometry_covariance_diag.float().unsqueeze(2)
            + discovery.geometry_variance.float().unsqueeze(1)
        )
    ).clamp_min(config.minimum_variance)
    geometry_negative_log_likelihood = 0.5 * (
        geometry_residual.square() / innovation_variance
        + innovation_variance.log()
        + math.log(2.0 * math.pi)
    ).sum(dim=-1)

    probability_floor = torch.finfo(torch.float32).eps
    existence = predicted.existence.float()
    detection = torch.sigmoid(predicted.visibility_given_existence_logits.float())
    # A current query is a usable measurement only when it denotes a physical
    # object *and* its spatial support is reliable. Persistent existence and
    # detectability remain row-side variables; pairwise identity remains in the
    # calibrated address likelihood ratio.
    observation_probability = discovery.measurement_probability.float()
    birth_odds = _birth_to_clutter_odds(
        predicted.valid,
        observation_probability,
        config,
    ).float()
    row_null_mass = (1.0 - existence * detection).clamp_min(probability_floor)
    observation_null_mass = (
        1.0 - observation_probability + birth_odds * observation_probability
    ).clamp_min(probability_floor)
    log_edge_odds = (
        existence.clamp_min(probability_floor).log().unsqueeze(2)
        + detection.clamp_min(probability_floor).log().unsqueeze(2)
        + observation_probability.clamp_min(probability_floor).log().unsqueeze(1)
        + address_log_likelihood_ratio
        - geometry_negative_log_likelihood
        - row_null_mass.log().unsqueeze(2)
        - observation_null_mass.log().unsqueeze(1)
    )
    log_edge_odds = log_edge_odds.masked_fill(~predicted.valid.unsqueeze(2), -torch.inf)
    association = marginal_one_to_one_association(log_edge_odds, predicted.valid)
    return association, observation_probability, birth_odds


def _assemble_marginal_lifecycle(
    predicted: ObjectBeliefBatch,
    discovery: ObjectDiscoveryOutput,
    association: MarginalAssociation,
    observation_probability: torch.Tensor,
    birth_odds: torch.Tensor,
    config: TemporalFilterConfig,
) -> _MarginalLifecycle:
    """Moment-match the marginal RFS posterior and reduce it to finite capacity."""

    batch_size, capacity = predicted.valid.shape
    observation_count = observation_probability.shape[1]
    probability_floor = torch.finfo(torch.float32).eps
    match_probability = association.match_probability.float()
    null_probability = association.null_probability.float()
    existence = predicted.existence.float()
    detection = torch.sigmoid(predicted.visibility_given_existence_logits.float())

    no_detection_mass = (1.0 - existence * detection).clamp_min(probability_floor)
    existence_given_no_detection = existence * (1.0 - detection) / no_detection_mass
    detected_mass = match_probability.sum(dim=-1)
    missed_alive_mass = null_probability * existence_given_no_detection
    posterior_existence = (detected_mass + missed_alive_mass).clamp(min=0.0, max=1.0)

    conditional_denominator = posterior_existence.clamp_min(probability_floor)
    active_existing = predicted.valid & (posterior_existence > 0.0)
    null_weight = torch.where(
        active_existing,
        missed_alive_mass / conditional_denominator,
        torch.ones_like(posterior_existence),
    )
    match_weight = (
        match_probability / conditional_denominator.unsqueeze(-1)
    ) * active_existing.unsqueeze(-1)

    # Keep all probability-weighted first and second moments in float32.  In
    # particular, casting the association weights or CI components to bf16
    # before the mixture can erase small but causally important alternatives.
    prior_geometry = (
        predicted.geometry_mean.float()
        .unsqueeze(2)
        .expand(
            -1,
            -1,
            observation_count,
            -1,
        )
    )
    prior_covariance = (
        predicted.geometry_covariance_diag.float().unsqueeze(2).expand_as(prior_geometry)
    )
    observation_geometry = discovery.geometry_mean.float().unsqueeze(1).expand_as(prior_geometry)
    observation_covariance = (
        discovery.geometry_variance.float().unsqueeze(1).expand_as(prior_geometry)
    )
    pairwise_geometry = equal_weight_covariance_intersection(
        prior_geometry,
        prior_covariance,
        observation_geometry,
        observation_covariance,
        minimum_variance=float(config.minimum_variance),
    )
    existing_geometry, existing_covariance = probabilistic_data_association_moments(
        predicted.geometry_mean.float(),
        predicted.geometry_covariance_diag.float(),
        pairwise_geometry.mean,
        pairwise_geometry.covariance_diag,
        null_weight,
        match_weight,
        minimum_variance=float(config.minimum_variance),
    )
    existing_geometry = existing_geometry.to(predicted.geometry_mean.dtype)
    existing_covariance = existing_covariance.to(predicted.geometry_covariance_diag.dtype)

    observation_content = discovery.content_mean.unsqueeze(1)
    existing_content = (
        null_weight.unsqueeze(-1) * predicted.content_mean.float()
        + (match_weight.unsqueeze(-1) * observation_content.float()).sum(dim=2)
    ).to(predicted.content_mean.dtype)

    observation_null_mass = (
        1.0 - observation_probability + birth_odds * observation_probability
    ).clamp_min(probability_floor)
    birth_given_unexplained = birth_odds * observation_probability / observation_null_mass
    birth_probability = (
        association.unexplained_observation_probability.float() * birth_given_unexplained
    ).clamp(min=0.0, max=1.0)

    retained_existing = predicted.valid & (posterior_existence > BERNOULLI_PRUNING_PROBABILITY)
    # One PMB component set cannot encode the mutually exclusive hypotheses
    # "this observation matched an existing row" and "this observation is a
    # new object" as two simultaneously existing identities. During the sole
    # hard finite-capacity projection, compare the birth event with the full
    # mutually exclusive existing-origin class, not only its strongest edge.
    # The retained birth may still be tentative (well below MAP presence).
    existing_origin_probability = match_probability.sum(dim=1)
    retained_birth = (birth_probability > BERNOULLI_PRUNING_PROBABILITY) & (
        birth_probability > existing_origin_probability
    )
    candidate_score = torch.cat(
        (
            posterior_existence.masked_fill(~retained_existing, -torch.inf),
            birth_probability.masked_fill(~retained_birth, -torch.inf),
        ),
        dim=1,
    )
    # Capacity reduction is the sole hard operation.  Its indices are detached;
    # selected component values retain their gradients.
    selected_index = candidate_score.detach().topk(capacity, dim=1, sorted=True).indices
    selected_candidate = torch.zeros_like(candidate_score, dtype=torch.bool).scatter(
        1,
        selected_index,
        True,
    )
    selected_existing = selected_candidate[:, :capacity] & retained_existing
    selected_birth = selected_candidate[:, capacity:] & retained_birth
    birth_assignment = _assign_selected_births_to_free_rows(
        selected_index,
        selected_existing,
        selected_birth,
        capacity=capacity,
        observation_count=observation_count,
    )
    birth_by_row = birth_assignment.any(dim=-1)
    valid = selected_existing | birth_by_row

    assignment_float = birth_assignment.to(discovery.address_mean.dtype)
    birth_address = torch.einsum("bkq,bqd->bkd", assignment_float, discovery.address_mean)
    birth_content = torch.einsum("bkq,bqd->bkd", assignment_float, discovery.content_mean)
    birth_geometry = torch.einsum("bkq,bqd->bkd", assignment_float, discovery.geometry_mean)
    birth_covariance = torch.einsum(
        "bkq,bqd->bkd",
        assignment_float,
        discovery.geometry_variance,
    )
    birth_existence_by_row = torch.einsum(
        "bkq,bq->bk",
        birth_assignment.float(),
        birth_probability,
    )

    select_birth = birth_by_row.unsqueeze(-1)
    select_valid = valid.unsqueeze(-1)
    address = torch.where(select_birth, birth_address, predicted.address_mean) * select_valid
    content = torch.where(select_birth, birth_content, existing_content) * select_valid
    geometry = torch.where(select_birth, birth_geometry, existing_geometry) * select_valid
    covariance = (
        torch.where(
            select_birth,
            birth_covariance,
            existing_covariance,
        )
        * select_valid
    )

    final_existence = (
        torch.where(
            birth_by_row,
            birth_existence_by_row,
            posterior_existence,
        )
        * valid
    )
    map_present = valid & (final_existence > MAP_PRESENCE_PROBABILITY)
    existing_conditional_visibility = detected_mass / conditional_denominator
    conditional_visibility = (
        torch.where(
            birth_by_row,
            torch.ones_like(final_existence),
            existing_conditional_visibility,
        )
        * valid
    )
    existence_logits = _probability_logits(
        final_existence,
        valid,
        predicted.existence_logits.dtype,
    )
    visibility_logits = _probability_logits(
        conditional_visibility,
        valid,
        predicted.visibility_given_existence_logits.dtype,
    )
    existing_measurement_age_s = (null_weight * predicted.measurement_age_s.float()).to(
        predicted.measurement_age_s.dtype
    )
    measurement_age_s = (
        torch.where(
            birth_by_row,
            torch.zeros_like(existing_measurement_age_s),
            existing_measurement_age_s,
        )
        * valid
    )
    age = torch.where(birth_by_row, torch.zeros_like(predicted.age), predicted.age) * valid
    belief = ObjectBeliefBatch(
        address_mean=address,
        content_mean=content,
        geometry_mean=geometry,
        geometry_covariance_diag=covariance,
        existence_logits=existence_logits,
        visibility_given_existence_logits=visibility_logits,
        measurement_age_s=measurement_age_s,
        valid=valid,
        age=age,
    )

    existing_dynamic = torch.cat((existing_content, existing_geometry), dim=-1)
    innovation = (existing_dynamic - predicted.dynamic_mean) * selected_existing.unsqueeze(-1)
    ownership = _transport_marginal_ownership(
        discovery.ownership,
        match_probability,
        birth_probability,
        selected_existing,
        birth_assignment,
        map_present,
    )
    observation_to_posterior, event_type = _diagnostic_lifecycle_projection(
        predicted.valid,
        selected_existing,
        birth_assignment,
        match_probability,
        null_probability,
    )
    return _MarginalLifecycle(
        belief=belief,
        innovation=innovation,
        ownership=ownership,
        birth_probability=birth_probability.to(discovery.existence_logits.dtype),
        born=birth_by_row,
        map_present=map_present,
        observation_to_posterior=observation_to_posterior,
        event_type=event_type,
    )


def _assign_selected_births_to_free_rows(
    selected_index: torch.Tensor,
    selected_existing: torch.Tensor,
    selected_birth: torch.Tensor,
    *,
    capacity: int,
    observation_count: int,
) -> torch.Tensor:
    """Map selected birth components to rows not retained by old identities."""

    batch_size = selected_index.shape[0]
    rows = torch.arange(capacity, device=selected_index.device).expand(batch_size, -1)
    free_rows = rows.masked_fill(selected_existing, capacity).sort(dim=1).values
    top_is_birth = selected_index >= capacity
    birth_rank = top_is_birth.long().cumsum(dim=1) - 1
    safe_rank = birth_rank.clamp(min=0, max=capacity - 1)
    destination = free_rows.gather(1, safe_rank)
    query = (selected_index - capacity).clamp(min=0, max=observation_count - 1)
    active = top_is_birth & selected_birth.gather(1, query) & (destination < capacity)
    row_indicator = F.one_hot(destination.clamp(max=capacity - 1), capacity).bool()
    query_indicator = F.one_hot(query, observation_count).bool()
    return (
        row_indicator.unsqueeze(-1) & query_indicator.unsqueeze(-2) & active[:, :, None, None]
    ).any(dim=1)


def _transport_marginal_ownership(
    discovery_ownership: torch.Tensor,
    match_probability: torch.Tensor,
    birth_probability: torch.Tensor,
    selected_existing: torch.Tensor,
    birth_assignment: torch.Tensor,
    map_present: torch.Tensor,
) -> torch.Tensor:
    """Move current ownership only through the extracted MAP object set."""

    extracted_existing = selected_existing & map_present
    existing_transport = match_probability.permute(0, 2, 1) * extracted_existing.unsqueeze(1)
    birth_transport = (
        birth_assignment.permute(0, 2, 1).float()
        * birth_probability.unsqueeze(-1)
        * map_present.unsqueeze(1)
    )
    object_transport = existing_transport + birth_transport
    context_transport = (1.0 - object_transport.sum(dim=-1)).clamp(min=0.0, max=1.0)
    transport = torch.cat((object_transport, context_transport.unsqueeze(-1)), dim=-1)
    output = torch.bmm(discovery_ownership[..., :-1].float(), transport)
    output[..., -1] = output[..., -1] + discovery_ownership[..., -1].float()
    return output.to(discovery_ownership.dtype)


def _diagnostic_lifecycle_projection(
    prior_valid: torch.Tensor,
    selected_existing: torch.Tensor,
    birth_assignment: torch.Tensor,
    match_probability: torch.Tensor,
    null_probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project marginals to one-to-one telemetry without driving inference."""

    batch_size, capacity, observation_count = match_probability.shape
    birth_by_row = birth_assignment.any(dim=-1)
    birth_by_observation = birth_assignment.any(dim=1)
    rows = torch.arange(capacity, device=prior_valid.device).view(1, capacity)
    row_match_probability, row_observation = match_probability.max(dim=-1)
    column_row = match_probability.argmax(dim=1)
    mutual = column_row.gather(1, row_observation) == rows
    observation_is_birth = birth_by_observation.gather(1, row_observation)
    diagnostic_match = (
        selected_existing
        & mutual
        & ~observation_is_birth
        & (row_match_probability > null_probability)
    )
    match_assignment = F.one_hot(
        row_observation, observation_count
    ).bool() & diagnostic_match.unsqueeze(-1)
    combined_assignment = birth_assignment | match_assignment
    mapped = combined_assignment.permute(0, 2, 1)
    mapped_row = (
        mapped.long() * torch.arange(capacity, device=prior_valid.device).view(1, 1, capacity)
    ).sum(dim=-1)
    observation_to_posterior = torch.where(
        mapped.any(dim=-1),
        mapped_row,
        torch.full_like(mapped_row, -1),
    )

    event_type = torch.full(
        (batch_size, capacity),
        UNUSED_EVENT,
        dtype=torch.long,
        device=prior_valid.device,
    )
    event_type = torch.where(
        prior_valid,
        torch.full_like(event_type, DEATH_EVENT),
        event_type,
    )
    event_type = torch.where(
        selected_existing,
        torch.full_like(event_type, MISS_EVENT),
        event_type,
    )
    event_type = torch.where(
        diagnostic_match,
        torch.full_like(event_type, MATCH_EVENT),
        event_type,
    )
    event_type = torch.where(
        birth_by_row,
        torch.full_like(event_type, BIRTH_EVENT),
        event_type,
    )
    return observation_to_posterior, event_type


def _probability_logits(
    probability: torch.Tensor,
    valid: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    epsilon = torch.finfo(torch.float32).eps
    logits = torch.logit(probability.float().clamp(min=epsilon, max=1.0 - epsilon))
    return torch.where(valid, logits, torch.zeros_like(logits)).to(dtype)


def _birth_to_clutter_odds(
    prior_valid: torch.Tensor,
    observation_probability: torch.Tensor,
    config: TemporalFilterConfig,
) -> torch.Tensor:
    """Return the initial or recurrent birth intensity for each observation."""

    bank_is_empty = ~prior_valid.any(dim=1, keepdim=True)
    empty = observation_probability.new_full(
        observation_probability.shape,
        config.empty_bank_birth_to_clutter_prior_odds,
    )
    recurrent = observation_probability.new_full(
        observation_probability.shape,
        config.recurrent_birth_to_clutter_prior_odds,
    )
    return torch.where(bank_is_empty, empty, recurrent)


def _validate_discovery(
    discovery: ObjectDiscoveryOutput,
    prior: ObjectBeliefBatch,
    config: TemporalFilterConfig,
    *,
    validate_values: bool = True,
) -> None:
    batch_size = prior.valid.shape[0]
    if discovery.existence_logits.ndim != 2 or discovery.existence_logits.shape[0] != batch_size:
        raise ValueError("discovery existence logits must be batch-by-observation")
    observation_count = discovery.existence_logits.shape[1]
    if observation_count <= 0:
        raise ValueError("discovery must expose at least one observation query")
    if discovery.token_valid.ndim != 2 or discovery.token_valid.shape[0] != batch_size:
        raise ValueError("discovery token validity must be batch-by-token")
    token_count = discovery.token_valid.shape[1]
    expected = {
        "address_mean": (batch_size, observation_count, config.address_dim),
        "content_mean": (batch_size, observation_count, config.content_dim),
        "geometry_mean": (batch_size, observation_count, config.geometry_dim),
        "geometry_variance": (
            batch_size,
            observation_count,
            config.geometry_dim,
        ),
        "localization_confidence_logits": (batch_size, observation_count),
        "ownership_logits": (batch_size, token_count, observation_count + 1),
        "ownership": (batch_size, token_count, observation_count + 1),
        "token_group_id": (batch_size, token_count),
        "evidence_available": (batch_size,),
    }
    for name, shape in expected.items():
        if getattr(discovery, name).shape != shape:
            raise ValueError(f"discovery {name} must have shape {shape}")
    if discovery.query_features.ndim != 3 or discovery.query_features.shape[:2] != (
        batch_size,
        observation_count,
    ):
        raise ValueError("discovery query features must align batch-by-observation")
    if discovery.query_features.shape[-1] <= 0:
        raise ValueError("discovery query features must have positive width")

    device = prior.address_mean.device
    dtype = prior.address_mean.dtype
    floating = (
        discovery.query_features,
        discovery.address_mean,
        discovery.content_mean,
        discovery.geometry_mean,
        discovery.geometry_variance,
        discovery.existence_logits,
        discovery.localization_confidence_logits,
        discovery.ownership_logits,
        discovery.ownership,
    )
    floating_by_name = dict(
        zip(
            (
                "query_features",
                "address_mean",
                "content_mean",
                "geometry_mean",
                "geometry_variance",
                "existence_logits",
                "localization_confidence_logits",
                "ownership_logits",
                "ownership",
            ),
            floating,
            strict=True,
        )
    )
    mismatched = {
        name: f"dtype={value.dtype},device={value.device},floating={value.is_floating_point()}"
        for name, value in floating_by_name.items()
        if value.device != device or value.dtype != dtype or not torch.is_floating_point(value)
    }
    if mismatched:
        details = "; ".join(f"{name}({description})" for name, description in mismatched.items())
        raise ValueError(
            "discovery floating tensors must match prior dtype and device "
            f"(prior dtype={dtype},device={device}; mismatched: {details})"
        )
    if validate_values:
        finite_required = (*floating[:7], discovery.ownership)
        if any(not torch.isfinite(value).all() for value in finite_required):
            raise ValueError("discovery contains NaN or infinity")
        if (
            torch.isnan(discovery.ownership_logits).any()
            or torch.isposinf(discovery.ownership_logits).any()
        ):
            raise ValueError("discovery ownership logits contain NaN or positive infinity")
        if (discovery.geometry_variance < config.minimum_variance).any():
            raise ValueError("discovery observation variance is below the configured minimum")
        address_norm = torch.linalg.vector_norm(discovery.address_mean.float(), dim=-1)
        tolerance = max(1e-5, torch.finfo(discovery.address_mean.dtype).eps)
        if not torch.allclose(
            address_norm,
            torch.ones_like(address_norm),
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError("discovery addresses must have unit norm")

    if discovery.token_valid.dtype != torch.bool or discovery.token_valid.device != device:
        raise ValueError("discovery token validity must be a colocated bool tensor")
    if discovery.token_group_id.dtype != torch.long or discovery.token_group_id.device != device:
        raise ValueError("discovery token group IDs must be a colocated long tensor")
    if (
        discovery.evidence_available.dtype != torch.bool
        or discovery.evidence_available.device != device
    ):
        raise ValueError("discovery evidence availability must be a colocated bool tensor")
    if validate_values:
        if (discovery.token_group_id < -1).any() or (
            discovery.token_group_id[~discovery.token_valid] != -1
        ).any():
            raise ValueError("discovery token group IDs are invalid")
        if not torch.equal(
            discovery.evidence_available,
            discovery.token_valid.any(dim=1),
        ):
            raise ValueError("discovery evidence availability must equal token validity")
        if (discovery.ownership < 0.0).any():
            raise ValueError("discovery ownership cannot be negative")
        probability_tolerance = max(1e-5, torch.finfo(discovery.ownership.dtype).eps)
        if not torch.allclose(
            discovery.ownership.float().sum(dim=-1),
            torch.ones_like(discovery.ownership[..., 0], dtype=torch.float32),
            atol=probability_tolerance,
            rtol=probability_tolerance,
        ):
            raise ValueError("discovery ownership must form a per-token simplex")
        recomputed = torch.softmax(discovery.ownership_logits.float(), dim=-1)
        if not torch.allclose(
            discovery.ownership.float(),
            recomputed,
            atol=probability_tolerance,
            rtol=probability_tolerance,
        ):
            raise ValueError("discovery ownership must equal softmax of its logits")
        if (~discovery.token_valid).any():
            invalid = discovery.ownership[~discovery.token_valid]
            if (invalid[..., :-1] != 0.0).any() or (invalid[..., -1] != 1.0).any():
                raise ValueError("invalid discovery tokens must belong exactly to context")
