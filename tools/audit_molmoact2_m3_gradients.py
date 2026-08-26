#!/usr/bin/env python3
"""Audit per-objective gradients at an exact completed M3 stream checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from tools.audit_molmoact2_m3_temporal import (  # noqa: E402
    _checkpoint_contract,
    _git_revision,
    _sha256,
    _validate_extended_plan_prefix,
)

_BELIEF_FIELDS = (
    "address_mean",
    "content_mean",
    "geometry_mean",
    "geometry_covariance_diag",
    "existence_logits",
    "visibility_given_existence_logits",
    "measurement_age_s",
    "valid",
    "age",
)
_COMPONENTS = (
    "weighted_action",
    "weighted_set",
    "weighted_dynamics_one_step",
    "weighted_dynamics_overshooting",
    "weighted_dynamics",
    "weighted_binding",
    "total",
)


@dataclass(frozen=True, slots=True)
class _AuditLifecyclePlan:
    """Rejected M3 hard lifecycle, retained only to reproduce causal evidence."""

    observation_to_posterior: Any
    matched_observation_for_row: Any
    birth_observation_for_row: Any
    event_type: Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m3_probe.json",
    )
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa2-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--extended-plan-steps",
        type=int,
        default=None,
        help=(
            "explicit read-only plan length when the checkpoint plan has no next "
            "transition; the complete checkpoint plan must remain an exact prefix"
        ),
    )
    return parser.parse_args()


def _audit_plan_steps(
    *,
    checkpoint_steps: int,
    checkpoint_plan_steps: int,
    extended_plan_steps: int | None,
) -> int:
    """Select a plan containing one post-checkpoint transition without silent extension."""

    for value, name in (
        (checkpoint_steps, "checkpoint steps"),
        (checkpoint_plan_steps, "checkpoint plan steps"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if checkpoint_steps > checkpoint_plan_steps:
        raise ValueError("checkpoint progress exceeds its frozen plan")
    if extended_plan_steps is None:
        if checkpoint_steps >= checkpoint_plan_steps:
            raise ValueError(
                "checkpoint has no next frozen transition to audit; pass an explicit "
                "--extended-plan-steps and validate the complete frozen prefix"
            )
        return checkpoint_plan_steps
    if (
        not isinstance(extended_plan_steps, int)
        or isinstance(extended_plan_steps, bool)
        or extended_plan_steps <= checkpoint_plan_steps
    ):
        raise ValueError("extended audit plan must be longer than the checkpoint plan")
    if extended_plan_steps <= checkpoint_steps:
        raise ValueError("extended audit plan has no post-checkpoint transition")
    return extended_plan_steps


def _parameter_family(name: str) -> str:
    if name.startswith("joint_bridge.objective.binding_criterion.relation."):
        return "multimodal_relation_calibration"
    if name.startswith("joint_bridge.sequence_bridge.core.posterior_filter.address_relation."):
        return "temporal_relation_calibration"
    if name.startswith("joint_bridge.sequence_bridge.policy.action_layer_adapter."):
        return "action_adapter"
    # Retain the direct prefix for synthetic and alternate bridge owners.
    if name.startswith("joint_bridge.sequence_bridge.action_adapter."):
        return "action_adapter"
    if name.startswith("joint_bridge.sequence_bridge.policy."):
        return "host_policy"
    if name.startswith("joint_bridge.sequence_bridge.core."):
        return "picf_core"
    return "other"


def _gradient_statistics(
    named_parameters: tuple[tuple[str, Any], ...],
    *,
    clip_norm: float,
) -> dict[str, Any]:
    group_squared: dict[str, float] = {}
    group_elements: dict[str, int] = {}
    total_squared = 0.0
    maximum_absolute = 0.0
    nonfinite = 0
    top: list[tuple[str, float]] = []
    tensors = 0
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        tensors += 1
        detached = gradient.detach().float()
        finite = detached.isfinite()
        nonfinite += int((~finite).sum().item())
        if not bool(finite.all()):
            detached = detached.where(finite, detached.new_zeros(()))
        norm = float(detached.norm().item())
        maximum_absolute = max(maximum_absolute, float(detached.abs().max().item()))
        squared = norm * norm
        family = _parameter_family(name)
        group_squared[family] = group_squared.get(family, 0.0) + squared
        group_elements[family] = group_elements.get(family, 0) + parameter.numel()
        total_squared += squared
        top.append((name, norm))
    total_norm = math.sqrt(total_squared)
    top.sort(key=lambda item: item[1], reverse=True)
    return {
        "clip_multiplier": min(1.0, clip_norm / max(total_norm, 1e-12)),
        "global_l2_norm": total_norm,
        "gradient_tensors": tensors,
        "group_l2_norm": {name: math.sqrt(value) for name, value in sorted(group_squared.items())},
        "group_parameter_elements_with_gradient": dict(sorted(group_elements.items())),
        "maximum_absolute_gradient": maximum_absolute,
        "nonfinite_gradient_elements": nonfinite,
        "top_parameter_l2_norm": [{"name": name, "l2_norm": norm} for name, norm in top[:16]],
    }


def _copy_family_gradients(
    named_parameters: tuple[tuple[str, Any], ...],
    family: str,
) -> dict[str, Any]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in named_parameters
        if _parameter_family(name) == family and parameter.grad is not None
    }


def _cosine_with_reference(
    named_parameters: tuple[tuple[str, Any], ...],
    reference: dict[str, Any],
    family: str,
) -> float | None:
    dot = 0.0
    reference_squared = 0.0
    current_squared = 0.0
    used = 0
    for name, parameter in named_parameters:
        if _parameter_family(name) != family or parameter.grad is None or name not in reference:
            continue
        current = parameter.grad.detach().float().cpu()
        baseline = reference[name]
        dot += float((current * baseline).sum().item())
        reference_squared += float(baseline.square().sum().item())
        current_squared += float(current.square().sum().item())
        used += 1
    denominator = math.sqrt(reference_squared * current_squared)
    return None if used == 0 or denominator == 0.0 else dot / denominator


def _geometry_rollout_diagnostics(
    transition: Any,
    start: Any,
    identity_keys_by_row: Any,
    target: Any,
) -> dict[str, Any]:
    """Expose the exact per-object terms hidden by the scalar overshoot loss."""

    import torch
    from torch.nn import functional as F

    from picf_next.models.dynamics_loss import (
        _detach_rollout_lifecycle,
        _detached_rollout_belief,
        _validate_geometry_rollout_target,
    )

    horizon, capacity = _validate_geometry_rollout_target(target, start, transition)
    frozen_keys = tuple(tuple(keys) for keys in identity_keys_by_row)
    if len(frozen_keys) != start.valid.shape[0] or any(
        len(keys) != capacity for keys in frozen_keys
    ):
        raise ValueError("geometry diagnostic keys must be batch-by-posterior-row")
    row_by_key = tuple(
        {key: row for row, key in enumerate(keys) if key is not None} for keys in frozen_keys
    )

    def values(tensor: Any) -> list[Any]:
        return tensor.detach().float().cpu().tolist()

    start_rows: list[dict[str, Any]] = []
    for batch_index, keys in enumerate(frozen_keys):
        for row, key in enumerate(keys):
            if key is None and not bool(start.valid[batch_index, row]):
                continue
            start_rows.append(
                {
                    "age": int(start.age[batch_index, row].detach().cpu().item()),
                    "measurement_age_s": float(
                        start.measurement_age_s[batch_index, row].detach().cpu().item()
                    ),
                    "batch_index": batch_index,
                    "existence_probability": float(
                        torch.sigmoid(start.existence_logits[batch_index, row].float())
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "geometry_mean": values(start.geometry_mean[batch_index, row]),
                    "geometry_variance": values(start.geometry_covariance_diag[batch_index, row]),
                    "identity_key": key,
                    "row": row,
                    "valid": bool(start.valid[batch_index, row].detach().cpu().item()),
                    "visibility_given_existence_probability": float(
                        torch.sigmoid(
                            start.visibility_given_existence_logits[batch_index, row].float()
                        )
                        .detach()
                        .cpu()
                        .item()
                    ),
                }
            )

    belief = _detached_rollout_belief(start)
    horizon_reports: list[dict[str, Any]] = []
    horizon_mean_losses: list[float] = []
    with torch.no_grad():
        for horizon_index in range(horizon):
            safe_delta_t = torch.where(
                target.step_valid[:, horizon_index],
                target.delta_t_s[:, horizon_index],
                torch.full_like(
                    target.delta_t_s[:, horizon_index],
                    transition.config.reference_delta_t_s,
                ),
            )
            prediction = transition(
                belief,
                target.executed_actions[:, horizon_index],
                safe_delta_t,
            )
            belief = _detach_rollout_lifecycle(prediction.belief)
            objects: list[dict[str, Any]] = []
            object_losses: list[float] = []
            unaligned_keys: list[str] = []
            for batch_index, frame_keys in enumerate(target.identity_keys):
                for target_index, key in enumerate(frame_keys[horizon_index]):
                    if key is None:
                        continue
                    row = row_by_key[batch_index].get(key)
                    if row is None:
                        unaligned_keys.append(key)
                        continue
                    supervised = target.geometry_supervised[
                        batch_index, horizon_index, target_index
                    ]
                    predicted = belief.geometry_mean[batch_index, row].float()
                    expected = target.geometry[batch_index, horizon_index, target_index].float()
                    predicted_variance = belief.geometry_covariance_diag[batch_index, row].float()
                    expected_variance = target.geometry_variance[
                        batch_index, horizon_index, target_index
                    ].float()
                    total_variance = predicted_variance + expected_variance
                    residual = predicted - expected
                    terms = F.gaussian_nll_loss(
                        predicted,
                        expected,
                        total_variance,
                        full=False,
                        reduction="none",
                    )
                    active_terms = terms[supervised]
                    standardized = residual / total_variance.clamp_min(1e-6).sqrt()
                    object_loss = float(active_terms.mean().cpu().item())
                    object_losses.append(object_loss)
                    objects.append(
                        {
                            "batch_index": batch_index,
                            "dynamic_delta_geometry": values(
                                prediction.dynamic_delta[
                                    batch_index,
                                    row,
                                    transition.config.content_dim :,
                                ]
                            ),
                            "identity_key": key,
                            "mean_gaussian_nll": object_loss,
                            "predicted_geometry": values(predicted),
                            "predicted_variance": values(predicted_variance),
                            "process_variance": values(
                                prediction.process_variance[batch_index, row]
                            ),
                            "residual_predicted_minus_target": values(residual),
                            "row": row,
                            "standardized_residual": values(standardized),
                            "supervised": supervised.detach().cpu().tolist(),
                            "target_geometry": values(expected),
                            "target_index": target_index,
                            "target_variance": values(expected_variance),
                            "total_variance": values(total_variance),
                        }
                    )
            horizon_mean = sum(object_losses) / len(object_losses) if object_losses else None
            if horizon_mean is not None:
                horizon_mean_losses.append(horizon_mean)
            horizon_reports.append(
                {
                    "horizon": horizon_index + 1,
                    "mean_gaussian_nll": horizon_mean,
                    "objects": objects,
                    "unaligned_identity_keys": unaligned_keys,
                }
            )
    return {
        "criterion_loss_reconstructed": (
            sum(horizon_mean_losses) / len(horizon_mean_losses) if horizon_mean_losses else None
        ),
        "horizons": horizon_reports,
        "start_rows": start_rows,
    }


def _loss_only_oracle_plan(
    predicted_valid: Any,
    initial_identity_keys_by_row: Any,
    targets: Any,
    matches: Any,
    *,
    observation_count: int,
) -> tuple[Any, tuple[tuple[str | None, ...], ...], dict[str, int]]:
    """Build a read-only physical-identity plan for one causal counterfactual.

    Existing rows are retained on a miss, visible physical identities are mapped
    to their independently set-matched discovery query, and genuinely new
    identities use an unused row. This is an audit oracle only: no target value
    is exposed to the production forward or committed stream state.
    """

    import torch

    from picf_next.posterior import BIRTH_EVENT, MATCH_EVENT, MISS_EVENT, UNUSED_EVENT

    if predicted_valid.ndim != 2 or predicted_valid.dtype != torch.bool:
        raise ValueError("oracle validity must be bool batch-by-posterior-row")
    if not isinstance(observation_count, int) or observation_count <= 0:
        raise ValueError("oracle observation count must be a positive integer")
    batch_size, capacity = predicted_valid.shape
    frozen_keys = tuple(tuple(keys) for keys in initial_identity_keys_by_row)
    frozen_targets = tuple(targets)
    frozen_matches = tuple(matches)
    if (
        len(frozen_keys) != batch_size
        or len(frozen_targets) != batch_size
        or len(frozen_matches) != batch_size
        or any(len(keys) != capacity for keys in frozen_keys)
    ):
        raise ValueError("oracle keys, targets and matches must align with the posterior batch")

    device = predicted_valid.device
    observation_to_posterior = torch.full(
        (batch_size, observation_count), -1, dtype=torch.long, device=device
    )
    matched_observation_for_row = torch.full(
        (batch_size, capacity), -1, dtype=torch.long, device=device
    )
    birth_observation_for_row = torch.full(
        (batch_size, capacity), -1, dtype=torch.long, device=device
    )
    event_type = torch.full((batch_size, capacity), UNUSED_EVENT, dtype=torch.long, device=device)
    next_keys: list[tuple[str | None, ...]] = []
    matched_count = 0
    birth_count = 0
    unallocated_birth_count = 0
    valid_cpu = predicted_valid.detach().cpu()

    for batch_index, (keys, target, match) in enumerate(
        zip(frozen_keys, frozen_targets, frozen_matches, strict=True)
    ):
        present = [key for key in keys if key is not None]
        if any(not isinstance(key, str) or not key for key in present):
            raise ValueError("oracle initial keys must be nonempty strings or None")
        if len(set(present)) != len(present):
            raise ValueError("oracle initial keys must be unique")
        if any(
            key is not None and not bool(valid_cpu[batch_index, row])
            for row, key in enumerate(keys)
        ):
            raise ValueError("oracle initial keys cannot name unused rows")
        event_type[batch_index, predicted_valid[batch_index]] = MISS_EVENT
        row_keys = list(keys)
        row_by_key = {key: row for row, key in enumerate(keys) if key is not None}

        identities = target.temporal_identity_keys
        if identities is None or len(identities) != target.num_objects:
            raise ValueError("oracle target requires one physical key per target object")
        if len(set(identities)) != len(identities):
            raise ValueError("oracle target physical keys must be unique")
        predictions = match.prediction_indices.detach().cpu().tolist()
        target_indices = match.target_indices.detach().cpu().tolist()
        if len(predictions) != len(target_indices):
            raise ValueError("oracle set match is malformed")

        pending_births: list[tuple[int, str]] = []
        for query, target_index in zip(predictions, target_indices, strict=True):
            if not 0 <= query < observation_count or not 0 <= target_index < len(identities):
                raise ValueError("oracle set match contains an out-of-range index")
            key = identities[target_index]
            row = row_by_key.get(key)
            if row is None:
                pending_births.append((query, key))
                continue
            if int(event_type[batch_index, row]) == MATCH_EVENT:
                raise ValueError("oracle physical row was matched more than once")
            observation_to_posterior[batch_index, query] = row
            matched_observation_for_row[batch_index, row] = query
            event_type[batch_index, row] = MATCH_EVENT
            matched_count += 1

        free_rows = (
            torch.nonzero(
                event_type[batch_index] == UNUSED_EVENT,
                as_tuple=False,
            )
            .flatten()
            .cpu()
            .tolist()
        )
        for birth_index, (query, key) in enumerate(pending_births):
            if birth_index >= len(free_rows):
                unallocated_birth_count += 1
                continue
            row = free_rows[birth_index]
            observation_to_posterior[batch_index, query] = row
            birth_observation_for_row[batch_index, row] = query
            event_type[batch_index, row] = BIRTH_EVENT
            row_keys[row] = key
            birth_count += 1
        next_keys.append(tuple(row_keys))

    return (
        _AuditLifecyclePlan(
            observation_to_posterior=observation_to_posterior,
            matched_observation_for_row=matched_observation_for_row,
            birth_observation_for_row=birth_observation_for_row,
            event_type=event_type,
        ),
        tuple(next_keys),
        {
            "births": birth_count,
            "matches": matched_count,
            "unallocated_births": unallocated_birth_count,
        },
    )


def _loss_only_oracle_posterior(
    posterior_filter: Any,
    core_output: Any,
    targets: Any,
    matches: Any,
    initial_identity_keys_by_row: Any,
) -> tuple[Any, tuple[tuple[str | None, ...], ...], dict[str, int], Any]:
    """Reproduce the rejected M3 hard correction under a physical oracle plan.

    This historical operator deliberately lives outside ``picf_next``.  It is
    needed to reproduce the causal evidence that rejected hard association and
    independent-observation Kalman correction; it is never called by training,
    evaluation or deployment.
    """

    import torch

    from picf_next.models.temporal import ObjectBeliefBatch
    from picf_next.posterior import BIRTH_EVENT, MATCH_EVENT, MISS_EVENT

    def gather(observations: Any, observation_for_row: Any, selected: Any) -> Any:
        safe_index = observation_for_row.clamp_min(0)
        if observations.ndim == 2:
            gathered = observations.gather(1, safe_index)
        else:
            index = safe_index.unsqueeze(-1).expand(-1, -1, observations.shape[-1])
            gathered = observations.gather(1, index)
        feature_axes = (1,) * (gathered.ndim - 2)
        return gathered * selected.reshape(*selected.shape, *feature_axes)

    prediction = core_output.posterior.prior_prediction
    discovery = core_output.discovery
    with torch.no_grad():
        plan, next_keys, diagnostics = _loss_only_oracle_plan(
            prediction.belief.valid,
            initial_identity_keys_by_row,
            targets,
            matches,
            observation_count=discovery.existence_logits.shape[1],
        )
        matched = plan.event_type == MATCH_EVENT
        missed = plan.event_type == MISS_EVENT
        retained = matched | missed
        births = plan.event_type == BIRTH_EVENT
        valid = retained | births
        prior = prediction.belief
        matched_observation = plan.matched_observation_for_row
        observed_content = gather(discovery.content_mean, matched_observation, matched)
        observed_geometry = gather(discovery.geometry_mean, matched_observation, matched)
        observed_covariance = gather(
            discovery.geometry_variance,
            matched_observation,
            matched,
        )

        prior_variance = prior.geometry_covariance_diag.float()
        observation_variance = observed_covariance.float()
        gain = (
            prior_variance
            / (prior_variance + observation_variance).clamp_min(
                posterior_filter.config.minimum_variance
            )
        ) * matched.unsqueeze(-1)
        corrected_content = torch.where(
            matched.unsqueeze(-1),
            observed_content,
            prior.content_mean,
        )
        corrected_geometry = prior.geometry_mean.float() + gain * (
            observed_geometry.float() - prior.geometry_mean.float()
        )
        corrected_covariance = (
            (1.0 - gain).square() * prior_variance + gain.square() * observation_variance
        ).clamp_min(posterior_filter.config.minimum_variance)

        birth_observation = plan.birth_observation_for_row
        birth_address = gather(discovery.address_mean, birth_observation, births)
        birth_content = gather(discovery.content_mean, birth_observation, births)
        birth_geometry = gather(discovery.geometry_mean, birth_observation, births)
        birth_covariance = gather(discovery.geometry_variance, birth_observation, births)
        birth_object_probability = gather(discovery.existence, birth_observation, births).float()

        predicted_existence = prior.existence.float()
        missed_alive = (predicted_existence - prior.visibility.float()).clamp_min(0.0)
        missed_absent = (1.0 - predicted_existence).clamp_min(0.0)
        missed_existence = missed_alive / (missed_alive + missed_absent).clamp_min(
            posterior_filter.config.minimum_variance
        )
        bank_is_empty = ~prior.valid.any(dim=1, keepdim=True)
        empty_birth_odds = torch.full_like(
            discovery.existence.float(),
            posterior_filter.config.empty_bank_birth_to_clutter_prior_odds,
        )
        recurrent_birth_odds = torch.full_like(
            discovery.existence.float(),
            posterior_filter.config.recurrent_birth_to_clutter_prior_odds,
        )
        birth_odds = torch.where(bank_is_empty, empty_birth_odds, recurrent_birth_odds)
        selected_birth_odds = gather(birth_odds, birth_observation, births)
        birth_mass = selected_birth_odds * birth_object_probability
        birth_existence = birth_mass / (birth_mass + 1.0 - birth_object_probability).clamp_min(
            posterior_filter.config.minimum_variance
        )

        epsilon = 1e-6
        posterior_existence = torch.where(
            matched,
            torch.full_like(predicted_existence, 1.0 - epsilon),
            torch.where(missed, missed_existence, birth_existence),
        )
        conditional_visibility = torch.where(
            matched | births,
            torch.full_like(posterior_existence, 1.0 - epsilon),
            torch.full_like(posterior_existence, epsilon),
        )
        existence_logits = torch.where(
            valid,
            torch.logit(posterior_existence.clamp(min=epsilon, max=1.0 - epsilon)),
            torch.zeros_like(posterior_existence),
        ).to(prior.existence_logits.dtype)
        visibility_logits = torch.where(
            valid,
            torch.logit(conditional_visibility.clamp(min=epsilon, max=1.0 - epsilon)),
            torch.zeros_like(conditional_visibility),
        ).to(prior.visibility_given_existence_logits.dtype)
        select_birth = births.unsqueeze(-1)
        select_valid = valid.unsqueeze(-1)
        belief = ObjectBeliefBatch(
            address_mean=(
                torch.where(select_birth, birth_address, prior.address_mean) * select_valid
            ),
            content_mean=(
                torch.where(select_birth, birth_content, corrected_content) * select_valid
            ),
            geometry_mean=(
                torch.where(
                    select_birth,
                    birth_geometry,
                    corrected_geometry.to(prior.geometry_mean.dtype),
                )
                * select_valid
            ),
            geometry_covariance_diag=(
                torch.where(
                    select_birth,
                    birth_covariance,
                    corrected_covariance.to(prior.geometry_covariance_diag.dtype),
                )
                * select_valid
            ),
            existence_logits=existence_logits,
            visibility_given_existence_logits=visibility_logits,
            measurement_age_s=torch.where(
                matched | births,
                torch.zeros_like(prior.measurement_age_s),
                prior.measurement_age_s,
            )
            * valid,
            valid=valid,
            age=torch.where(births, torch.zeros_like(prior.age), prior.age) * valid,
        )
    return belief, next_keys, diagnostics, plan


def _replace_oracle_geometry_with_current_target(
    belief: Any,
    identity_keys_by_row: Any,
    targets: Any,
) -> tuple[Any, int]:
    """Replace only supervised geometry coordinates for a stronger audit bound."""

    import torch

    from picf_next.models.temporal import ObjectBeliefBatch

    frozen_keys = tuple(tuple(keys) for keys in identity_keys_by_row)
    frozen_targets = tuple(targets)
    if len(frozen_keys) != belief.valid.shape[0] or len(frozen_targets) != belief.valid.shape[0]:
        raise ValueError("oracle geometry targets must align with the belief batch")
    geometry = belief.geometry_mean.detach().clone()
    replaced_coordinates = 0
    with torch.no_grad():
        for batch_index, (keys, target) in enumerate(zip(frozen_keys, frozen_targets, strict=True)):
            identities = target.temporal_identity_keys
            if identities is None or target.geometry is None:
                continue
            supervised = target.geometry_supervised
            if supervised is None:
                supervised = torch.ones_like(target.geometry, dtype=torch.bool)
            target_by_key = {key: index for index, key in enumerate(identities)}
            for row, key in enumerate(keys):
                target_index = target_by_key.get(key)
                if target_index is None:
                    continue
                selected = supervised[target_index]
                geometry[batch_index, row] = torch.where(
                    selected,
                    target.geometry[target_index].to(geometry.dtype),
                    geometry[batch_index, row],
                )
                replaced_coordinates += int(selected.sum().item())
    return (
        ObjectBeliefBatch(
            address_mean=belief.address_mean.detach(),
            content_mean=belief.content_mean.detach(),
            geometry_mean=geometry,
            geometry_covariance_diag=belief.geometry_covariance_diag.detach(),
            existence_logits=belief.existence_logits.detach(),
            visibility_given_existence_logits=(belief.visibility_given_existence_logits.detach()),
            measurement_age_s=belief.measurement_age_s.detach(),
            valid=belief.valid.detach(),
            age=belief.age.detach(),
        ),
        replaced_coordinates,
    )


def _replace_oracle_geometry_with_current_discovery(
    belief: Any,
    discovery: Any,
    plan: Any,
) -> tuple[Any, int]:
    """Overwrite mapped rows with the deploy-visible current measurement mean.

    This is a read-only counterfactual for separating association error from
    covariance-lock-in. It does not alter covariance or lifecycle state and is
    never used by the production filter.
    """

    import torch

    from picf_next.models.temporal import ObjectBeliefBatch

    mapping = plan.observation_to_posterior
    if mapping.ndim != 2 or mapping.shape != discovery.geometry_mean.shape[:2]:
        raise ValueError("oracle observation mapping must align with discovery geometry")
    if mapping.shape[0] != belief.valid.shape[0]:
        raise ValueError("oracle observation mapping must align with the belief batch")
    geometry = belief.geometry_mean.detach().clone()
    replaced_coordinates = 0
    with torch.no_grad():
        for batch_index in range(mapping.shape[0]):
            for observation_index in range(mapping.shape[1]):
                row = int(mapping[batch_index, observation_index])
                if row < 0:
                    continue
                if not 0 <= row < geometry.shape[1] or not bool(belief.valid[batch_index, row]):
                    raise ValueError("oracle observation mapping names an invalid posterior row")
                geometry[batch_index, row] = discovery.geometry_mean[
                    batch_index, observation_index
                ].detach()
                replaced_coordinates += geometry.shape[-1]
    return (
        ObjectBeliefBatch(
            address_mean=belief.address_mean.detach(),
            content_mean=belief.content_mean.detach(),
            geometry_mean=geometry,
            geometry_covariance_diag=belief.geometry_covariance_diag.detach(),
            existence_logits=belief.existence_logits.detach(),
            visibility_given_existence_logits=(belief.visibility_given_existence_logits.detach()),
            measurement_age_s=belief.measurement_age_s.detach(),
            valid=belief.valid.detach(),
            age=belief.age.detach(),
        ),
        replaced_coordinates,
    )


def _current_geometry_measurement_diagnostics(
    predicted: Any,
    discovery: Any,
    plan: Any,
    targets: Any,
    matches: Any,
    identity_keys_by_row: Any,
) -> dict[str, Any]:
    """Report target-relative prior, measurement and Kalman-corrected errors."""

    import torch

    frozen_targets = tuple(targets)
    frozen_matches = tuple(matches)
    frozen_keys = tuple(tuple(keys) for keys in identity_keys_by_row)
    batch_size, capacity = predicted.valid.shape
    if (
        len(frozen_targets) != batch_size
        or len(frozen_matches) != batch_size
        or len(frozen_keys) != batch_size
        or any(len(keys) != capacity for keys in frozen_keys)
    ):
        raise ValueError("geometry diagnostic inputs must align with the predicted belief")

    objects: list[dict[str, Any]] = []
    measurement_absolute_errors: list[torch.Tensor] = []
    prior_absolute_errors: list[torch.Tensor] = []
    corrected_absolute_errors: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_index, (target, match, keys) in enumerate(
            zip(frozen_targets, frozen_matches, frozen_keys, strict=True)
        ):
            identities = target.temporal_identity_keys
            if identities is None or target.geometry is None:
                continue
            supervised = target.geometry_supervised
            if supervised is None:
                supervised = torch.ones_like(target.geometry, dtype=torch.bool)
            row_by_key = {key: row for row, key in enumerate(keys) if key is not None}
            predictions = match.prediction_indices.detach().cpu().tolist()
            target_indices = match.target_indices.detach().cpu().tolist()
            for observation_index, target_index in zip(predictions, target_indices, strict=True):
                selected = supervised[target_index]
                if not bool(selected.any()):
                    continue
                key = identities[target_index]
                expected = target.geometry[target_index].float()
                measurement = discovery.geometry_mean[batch_index, observation_index].float()
                measurement_variance = discovery.geometry_variance[
                    batch_index, observation_index
                ].float()
                measurement_error = (measurement - expected).abs()[selected]
                measurement_absolute_errors.append(measurement_error)
                prior_row = row_by_key.get(key)
                item: dict[str, Any] = {
                    "batch_index": batch_index,
                    "identity_key": key,
                    "observation_index": observation_index,
                    "oracle_posterior_row": int(
                        plan.observation_to_posterior[batch_index, observation_index]
                    ),
                    "measurement_absolute_error": measurement_error.tolist(),
                    "measurement_geometry": measurement.tolist(),
                    "measurement_variance": measurement_variance.tolist(),
                    "supervised": selected.tolist(),
                    "target_geometry": expected.tolist(),
                }
                if prior_row is not None:
                    prior = predicted.geometry_mean[batch_index, prior_row].float()
                    prior_variance = predicted.geometry_covariance_diag[
                        batch_index, prior_row
                    ].float()
                    gain = prior_variance / (prior_variance + measurement_variance).clamp_min(
                        torch.finfo(torch.float32).tiny
                    )
                    corrected = prior + gain * (measurement - prior)
                    prior_error = (prior - expected).abs()[selected]
                    corrected_error = (corrected - expected).abs()[selected]
                    prior_absolute_errors.append(prior_error)
                    corrected_absolute_errors.append(corrected_error)
                    item.update(
                        {
                            "kalman_corrected_absolute_error": corrected_error.tolist(),
                            "kalman_corrected_geometry": corrected.tolist(),
                            "kalman_gain": gain.tolist(),
                            "prior_absolute_error": prior_error.tolist(),
                            "prior_geometry": prior.tolist(),
                            "prior_row": prior_row,
                            "prior_variance": prior_variance.tolist(),
                        }
                    )
                objects.append(item)

    def mean_or_none(values: list[torch.Tensor]) -> float | None:
        if not values:
            return None
        return float(torch.cat(values).mean().item())

    return {
        "kalman_corrected_mean_absolute_error_existing": mean_or_none(corrected_absolute_errors),
        "measurement_mean_absolute_error": mean_or_none(measurement_absolute_errors),
        "objects": objects,
        "prior_mean_absolute_error_existing": mean_or_none(prior_absolute_errors),
    }


def _validated_rank_state(
    checkpoint: Path,
    control: dict[str, Any],
    *,
    rank: int,
    device: Any,
) -> tuple[Any, tuple[tuple[str | None, ...], ...], dict[str, Any]]:
    import torch

    from picf_next.models.temporal import ObjectBeliefBatch

    path = checkpoint / f"picf_rank_state_{rank:05d}.pt"
    state_files = control.get("state_files")
    expected = state_files.get(path.name) if isinstance(state_files, dict) else None
    if (
        not isinstance(expected, dict)
        or expected.get("size_bytes") != path.stat().st_size
        or expected.get("sha256") != _sha256(path)
    ):
        raise ValueError("rank-local posterior state differs from checkpoint control")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "picf-next.posterior-stream-state-group.v3":
        raise ValueError("unsupported rank-local posterior state group schema")
    streams = payload.get("streams")
    if not isinstance(streams, dict) or tuple(streams) != ("accumulation-00000",):
        raise ValueError("gradient audit requires one accumulation stream")
    stream = streams["accumulation-00000"]
    if stream.get("schema") != "picf-next.posterior-stream-state.v5":
        raise ValueError("unsupported posterior stream state schema")
    belief_payload = stream.get("belief")
    if not isinstance(belief_payload, dict) or set(belief_payload) != set(_BELIEF_FIELDS):
        raise ValueError("rank-local posterior belief fields are malformed")
    belief = ObjectBeliefBatch(**{name: belief_payload[name].to(device) for name in _BELIEF_FIELDS})
    raw_keys = stream.get("loss_track_keys_by_row")
    if not isinstance(raw_keys, list) or len(raw_keys) != belief.valid.shape[0]:
        raise ValueError("rank-local loss-track keys are malformed")
    keys = tuple(tuple(value for value in row) for row in raw_keys)
    if any(len(row) != belief.valid.shape[1] for row in keys):
        raise ValueError("rank-local loss-track width differs from posterior capacity")
    return belief, keys, stream


def _component_loss(output: Any, objective: Any, name: str) -> Any:
    losses = output.losses
    if name == "total":
        return output.loss
    if name == "weighted_dynamics_one_step":
        return objective.config.dynamics_weight * losses["loss_dynamics_one_step"]
    if name == "weighted_dynamics_overshooting":
        return (
            objective.config.dynamics_weight
            * losses["loss_dynamics_geometry_overshooting_weighted"]
        )
    key = {
        "weighted_action": "loss_weighted_action",
        "weighted_set": "loss_weighted_set",
        "weighted_dynamics": "loss_weighted_dynamics",
        "weighted_binding": "loss_weighted_binding",
    }[name]
    return losses[key]


def main() -> None:
    args = _parse_args()
    import torch
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from safetensors.torch import load_model

    from picf_next.data.vjepa2_cache import Vjepa2FeatureCache
    from picf_next.hosts.vjepa2_context import CalvinVjepa2CachedContextBuilder
    from picf_next.training.molmoact2_calvin import (
        build_calvin_episode_stream_plan,
        build_molmoact2_calvin_training_stack,
        build_molmoact2_policy_config,
        load_calvin_training_assets,
    )
    from picf_next.training.recipe import load_training_recipe

    checkpoint = args.training_checkpoint.expanduser().resolve()
    control, model_path = _checkpoint_contract(checkpoint)
    contract = control.get("contract")
    progress = control.get("progress")
    if not isinstance(contract, dict) or not isinstance(progress, dict):
        raise ValueError("training checkpoint contract/progress is malformed")
    arm = contract.get("arm_config", {}).get("causal_factorization")
    if not isinstance(arm, dict) or arm.get("id") != "D":
        raise ValueError("M3 gradient audit requires a full Arm D checkpoint")
    checkpoint_steps = progress.get("attempted_optimizer_steps")
    if (
        not isinstance(checkpoint_steps, int)
        or isinstance(checkpoint_steps, bool)
        or checkpoint_steps <= 0
        or progress.get("successful_optimizer_steps") != checkpoint_steps
    ):
        raise ValueError("training checkpoint is not a completed successful prefix")
    world_size = int(contract["world_size"])
    if not 0 <= args.rank < world_size:
        raise ValueError("audit rank is outside the checkpoint world")
    if int(contract["gradient_accumulation_steps"]) != 1:
        raise ValueError("gradient audit currently requires one accumulation step")

    recipe = load_training_recipe(args.recipe.resolve())
    if recipe.recipe_sha256 != contract["common_config"]["recipe_sha256"]:
        raise ValueError("gradient-audit recipe differs from checkpoint")
    checkpoint_plan_steps = int(control["plan"]["total_steps"])
    audit_plan_steps = _audit_plan_steps(
        checkpoint_steps=checkpoint_steps,
        checkpoint_plan_steps=checkpoint_plan_steps,
        extended_plan_steps=args.extended_plan_steps,
    )
    assets = load_calvin_training_assets(
        recipe,
        repository_root=_ROOT,
        split_root=args.dataset_split_root,
    )
    checkpoint_plan = build_calvin_episode_stream_plan(
        recipe,
        assets.dataset,
        comparison_id=str(contract["comparison_id"]),
        seed=int(control["plan"]["seed"]),
        global_batch_size=int(contract["optimizer_global_batch_size"]),
        total_steps=checkpoint_plan_steps,
    )
    if checkpoint_plan.plan_sha256 != control["plan_sha256"]:
        raise ValueError("reconstructed gradient-audit plan differs from checkpoint")
    plan = checkpoint_plan
    validated_prefix_steps = checkpoint_plan_steps
    if audit_plan_steps > checkpoint_plan_steps:
        recipe.assert_optimizer_steps_authorized(audit_plan_steps)
        plan = build_calvin_episode_stream_plan(
            recipe,
            assets.dataset,
            comparison_id=str(contract["comparison_id"]),
            seed=int(control["plan"]["seed"]),
            global_batch_size=int(contract["optimizer_global_batch_size"]),
            total_steps=audit_plan_steps,
        )
        validated_prefix_steps = _validate_extended_plan_prefix(checkpoint_plan, plan)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M3 gradient audit requires CUDA")
    belief, loss_track_keys, stream = _validated_rank_state(
        checkpoint,
        control,
        rank=args.rank,
        device=device,
    )
    if stream.get("next_transition_indices") != [checkpoint_steps]:
        raise ValueError("rank-local state is not positioned at the next checkpoint transition")
    microbatch = plan.microbatch_for_rank(
        checkpoint_steps,
        rank=args.rank,
        world_size=world_size,
        gradient_accumulation_steps=1,
        accumulation_index=0,
    )
    if len(microbatch.transitions) != 1:
        raise ValueError("gradient audit requires one transition on the selected rank")
    if stream.get("episode_keys") != [microbatch.transitions[0].episode_instance_id]:
        raise ValueError("rank-local state episode differs from the next frozen transition")

    vjepa_contract = contract["arm_config"].get("vjepa2_cache")
    if not isinstance(vjepa_contract, dict):
        raise ValueError("Arm D checkpoint has no V-JEPA cache binding")
    cache = Vjepa2FeatureCache.load(
        args.vjepa2_cache_root.expanduser().resolve(),
        manifest_sha256=str(vjepa_contract["manifest_sha256"]),
        dataset_tree_sha256=assets.dataset_manifest.tree_sha256,
        memory_capacity=64,
    )
    policy_config = build_molmoact2_policy_config(
        recipe,
        checkpoint_path=args.foundation_checkpoint_dir.expanduser().resolve(),
    )
    policy = MolmoAct2Policy(policy_config).to(device)
    parameter = next(policy.parameters())
    native_builder = CalvinVjepa2CachedContextBuilder(
        cache,
        device=device,
        dtype=parameter.dtype,
    )
    stack = build_molmoact2_calvin_training_stack(
        recipe,
        policy=policy,
        assets=assets,
        build_native_banks=native_builder,
        native_evidence_history_frames=native_builder.maximum_source_frames,
        action_context_token_dims=native_builder.token_dims,
        include_posterior_action_context=True,
    )
    module = stack.module.to(device)
    missing, unexpected = load_model(module, model_path, strict=True, device=str(device))
    if missing or unexpected:
        raise RuntimeError("strict model load unexpectedly reported key drift")
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    named_parameters = tuple(
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    unknown = [name for name, _parameter in named_parameters if _parameter_family(name) == "other"]
    if unknown:
        raise RuntimeError(f"unclassified trainable parameters: {unknown[:8]}")
    clip_norm = float(recipe.optimizer.gradient_clip_norm)
    objective_module = module.joint_bridge.objective
    captured: list[Any] = []
    captured_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    geometry_rollout_diagnostics: dict[str, Any] | None = None

    def capture_objective(
        _module: Any,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        captured.append(output)
        captured_calls.append((inputs, kwargs))

    def capture_geometry_rollout(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
        nonlocal geometry_rollout_diagnostics
        if geometry_rollout_diagnostics is None:
            geometry_rollout_diagnostics = _geometry_rollout_diagnostics(*inputs)

    handle = objective_module.register_forward_hook(capture_objective, with_kwargs=True)
    geometry_handle = objective_module.geometry_overshooting_criterion.register_forward_hook(
        capture_geometry_rollout
    )
    module.train()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    action_core_gradients: dict[str, Any] | None = None
    component_reports: dict[str, Any] = {}
    objective_values: dict[str, float] | None = None
    try:
        for component in _COMPONENTS:
            module.zero_grad(set_to_none=True)
            captured.clear()
            captured_calls.clear()
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                stateful = module(microbatch, belief, loss_track_keys)
                if len(captured) != 1 or len(captured_calls) != 1:
                    raise RuntimeError("gradient audit did not capture exactly one objective")
                objective_output = captured[0]
                loss = _component_loss(objective_output, objective_module, component)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError(f"gradient component {component} is not one finite scalar")
            if not loss.requires_grad:
                raise ValueError(f"gradient component {component} has no trainable path")
            loss.backward()
            torch.cuda.synchronize(device)
            statistics = _gradient_statistics(named_parameters, clip_norm=clip_norm)
            statistics.update(
                {
                    "elapsed_seconds": time.perf_counter() - started,
                    "loss": float(loss.detach().float().item()),
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
            )
            if component == "weighted_action":
                action_core_gradients = _copy_family_gradients(named_parameters, "picf_core")
            elif action_core_gradients is not None:
                statistics["picf_core_cosine_with_weighted_action"] = _cosine_with_reference(
                    named_parameters,
                    action_core_gradients,
                    "picf_core",
                )
            component_reports[component] = statistics
            if objective_values is None:
                objective_values = {
                    name: float(value.detach().float().item())
                    for name, value in objective_output.losses.items()
                }
            print(
                json.dumps(
                    {
                        "clip_multiplier": statistics["clip_multiplier"],
                        "component": component,
                        "global_l2_norm": statistics["global_l2_norm"],
                        "loss": statistics["loss"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del stateful, objective_output, loss
    finally:
        handle.remove()
        geometry_handle.remove()
        module.zero_grad(set_to_none=True)

    if not captured_calls:
        raise RuntimeError("gradient audit lost the final captured objective call")
    objective_inputs, objective_kwargs = captured_calls[-1]
    core_outputs = tuple(objective_inputs[0])
    set_target_frames = objective_kwargs.get("set_targets")
    rollout_target = objective_kwargs.get("geometry_rollout_target")
    if not core_outputs or set_target_frames is None or rollout_target is None:
        raise RuntimeError("oracle counterfactual requires core, set and rollout targets")
    frame_targets = tuple(set_target_frames[-1])
    final_core_output = core_outputs[-1]
    set_matches = objective_module.set_criterion.matcher(
        final_core_output.discovery,
        frame_targets,
    )
    posterior_filter = module.joint_bridge.sequence_bridge.core.posterior_filter
    oracle_belief, oracle_keys, oracle_plan_diagnostics, oracle_plan = _loss_only_oracle_posterior(
        posterior_filter,
        final_core_output,
        frame_targets,
        set_matches,
        loss_track_keys,
    )
    discovery_geometry_belief, discovery_geometry_coordinates = (
        _replace_oracle_geometry_with_current_discovery(
            oracle_belief,
            final_core_output.discovery,
            oracle_plan,
        )
    )
    target_geometry_belief, replaced_coordinates = _replace_oracle_geometry_with_current_target(
        oracle_belief,
        oracle_keys,
        frame_targets,
    )
    counterfactual_reports: dict[str, Any] = {}
    for counterfactual_name, counterfactual_belief in (
        ("loss_only_identity_association", oracle_belief),
        (
            "loss_only_identity_and_current_discovery_mean",
            discovery_geometry_belief,
        ),
        ("loss_only_identity_and_current_target_geometry", target_geometry_belief),
    ):
        module.zero_grad(set_to_none=True)
        raw_loss = objective_module.geometry_overshooting_criterion(
            posterior_filter.transition,
            counterfactual_belief,
            oracle_keys,
            rollout_target,
        ).loss
        weighted_loss = (
            objective_module.config.dynamics_weight
            * objective_module.geometry_overshooting_criterion.config.weight
            * raw_loss
        )
        weighted_loss.backward()
        statistics = _gradient_statistics(named_parameters, clip_norm=clip_norm)
        if action_core_gradients is not None:
            statistics["picf_core_cosine_with_weighted_action"] = _cosine_with_reference(
                named_parameters,
                action_core_gradients,
                "picf_core",
            )
        counterfactual_reports[counterfactual_name] = {
            "geometry_rollout_diagnostics": _geometry_rollout_diagnostics(
                posterior_filter.transition,
                counterfactual_belief,
                oracle_keys,
                rollout_target,
            ),
            "gradient_statistics": statistics,
            "raw_overshooting_loss": float(raw_loss.detach().float().item()),
            "weighted_overshooting_loss": float(weighted_loss.detach().float().item()),
        }
    module.zero_grad(set_to_none=True)

    sample = assets.dataset.by_key(microbatch.transitions[0].sample.sample_key)
    report = {
        "audit_code_revision": _git_revision(_ROOT),
        "audit_plan_steps": audit_plan_steps,
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "audited_transition_inside_checkpoint_plan": checkpoint_steps < checkpoint_plan_steps,
        "checkpoint_code_revision": contract["code_revision"],
        "checkpoint_model_sha256": _sha256(model_path),
        "checkpoint_optimizer_steps": checkpoint_steps,
        "checkpoint_plan_steps": checkpoint_plan_steps,
        "component_gradients": component_reports,
        "counterfactual_oracle_association": {
            "current_geometry_measurement_diagnostics": (
                _current_geometry_measurement_diagnostics(
                    final_core_output.posterior.prior_prediction.belief,
                    final_core_output.discovery,
                    oracle_plan,
                    frame_targets,
                    set_matches,
                    loss_track_keys,
                )
            ),
            "replaced_current_discovery_geometry_coordinates": (discovery_geometry_coordinates),
            "plan": oracle_plan_diagnostics,
            "replaced_current_target_geometry_coordinates": replaced_coordinates,
            "reports": counterfactual_reports,
            "warning": (
                "Loss-only physical identities are used only after the deploy-visible forward "
                "to construct a read-only causal counterfactual. This path is not deployable."
            ),
        },
        "global_clip_norm": clip_norm,
        "geometry_rollout_diagnostics": geometry_rollout_diagnostics,
        "objective_losses": objective_values,
        "rank": args.rank,
        "sample": {
            "episode_key": sample.episode_key,
            "frame": sample.record.global_index,
            "instruction": sample.record.task,
            "sample_key": sample.sample_key,
            "task_key": sample.host_sample.task_key,
        },
        "schema": "picf-next.molmoact2-m3-gradient-audit.v4",
        "stream_state_parameter_versions": stream["state_parameter_versions"],
        "validated_checkpoint_plan_prefix_steps": validated_prefix_steps,
        "warning": (
            "Read-only exact-continuation diagnostic. No optimizer state was loaded and no "
            "parameter or posterior update was committed."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"output": str(output), "schema": report["schema"]}, sort_keys=True))


if __name__ == "__main__":
    main()
