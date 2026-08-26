"""Predictive losses for the explicit object posterior.

The current-frame discovery observation is a stop-gradient target for the
action-conditioned prior prediction. Trusted loss-side physical keys supply an
independent permutation-invariant correspondence when available; otherwise the
detached runtime association marginals provide an EM/PDA expected loss. Survival
and visibility are trained only from an independent loss-side target, because
using the filter's own projected event as a label would create a self-confirming
hard-EM loop. Masks, task labels and simulator IDs are never inputs to the
deployable forward path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.data.rollout_targets import ObjectGeometryRolloutTarget
from picf_next.geometry import PhysicalGeometryContract

from .core import PICFCoreOutput
from .temporal import (
    ActionConditionedObjectTransition,
    ObjectBeliefBatch,
    ObjectPredictionOutput,
)


@dataclass(frozen=True, slots=True)
class ObjectDynamicsLossConfig:
    content_cosine_weight: float = 1.0
    geometry_nll_weight: float = 1.0
    survival_weight: float = 0.0
    visibility_weight: float = 0.0
    probability_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        weights = (
            self.content_cosine_weight,
            self.geometry_nll_weight,
            self.survival_weight,
            self.visibility_weight,
        )
        if any(
            isinstance(weight, bool) or not math.isfinite(weight) or weight < 0.0
            for weight in weights
        ) or not any(weight > 0.0 for weight in weights):
            raise ValueError("dynamics weights must be nonnegative and not all zero")
        if (
            isinstance(self.probability_epsilon, bool)
            or not math.isfinite(self.probability_epsilon)
            or not 0.0 < self.probability_epsilon < 0.5
        ):
            raise ValueError("probability_epsilon must lie in (0, 0.5)")


@dataclass(frozen=True, slots=True)
class ObjectDynamicsLossOutput:
    losses: dict[str, torch.Tensor]
    matched_predictions: int
    independently_aligned_predictions: int
    lifecycle_predictions: int
    survival_positive_target_mass: torch.Tensor
    survival_negative_target_mass: torch.Tensor
    visibility_positive_target_mass: torch.Tensor
    visibility_negative_target_mass: torch.Tensor
    visibility_loss_sum: torch.Tensor
    visibility_detected_loss_sum: torch.Tensor
    visibility_missed_loss_sum: torch.Tensor
    visibility_prediction_count: torch.Tensor
    visibility_previous_detected_mass: torch.Tensor
    visibility_previous_missed_mass: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.losses["loss_dynamics_total"]


@dataclass(frozen=True, slots=True)
class ObjectGeometryOvershootingConfig:
    """Weight of bounded autoregressive geometry prediction in dynamics loss."""

    weight: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("geometry overshooting weight must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ObjectGeometryOvershootingOutput:
    loss: torch.Tensor
    active_horizons: int
    matched_predictions: int
    unaligned_target_objects: int
    maximum_horizon: int


@dataclass(frozen=True, slots=True)
class AlignedObjectLifecycleTarget:
    """Independent loss-only lifecycle evidence aligned to predicted rows.

    Alignment is performed after the deploy-visible forward using a trusted
    physical-track contract. Survival is an existence target; visibility is the
    conditional probability that a physically alive object yields a trustworthy
    current measurement. It is not raw geometric visibility. Values may be soft
    calibrated probabilities. Unsupervised rows are exactly zero and contribute
    no loss. This object must never be passed to discovery, association,
    correction or the action host.
    """

    survival: torch.Tensor
    survival_supervised: torch.Tensor
    visibility: torch.Tensor
    visibility_supervised: torch.Tensor
    previous_visibility: torch.Tensor
    previous_visibility_supervised: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObjectDetectabilityTransitionLossOutput:
    """Sufficient statistics for one directly identified observation kernel."""

    loss_sum: torch.Tensor
    detected_loss_sum: torch.Tensor
    missed_loss_sum: torch.Tensor
    supervised_count: torch.Tensor
    positive_target_mass: torch.Tensor
    negative_target_mass: torch.Tensor
    previous_detected_mass: torch.Tensor
    previous_missed_mass: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        return balanced_conditional_detectability_loss(
            self.detected_loss_sum,
            self.previous_detected_mass,
            self.missed_loss_sum,
            self.previous_missed_mass,
        )


def balanced_conditional_detectability_loss(
    detected_loss_sum: torch.Tensor,
    detected_mass: torch.Tensor,
    missed_loss_sum: torch.Tensor,
    missed_mass: torch.Tensor,
) -> torch.Tensor:
    """Average proper Bernoulli scores over active previous-state strata."""

    detected_active = detected_mass > 0.0
    missed_active = missed_mass > 0.0
    zero = detected_loss_sum * 0.0 + missed_loss_sum * 0.0
    detected_mean = torch.where(
        detected_active,
        detected_loss_sum / detected_mass.clamp_min(1.0),
        zero,
    )
    missed_mean = torch.where(
        missed_active,
        missed_loss_sum / missed_mass.clamp_min(1.0),
        zero,
    )
    active_count = detected_active.to(torch.long) + missed_active.to(torch.long)
    return (detected_mean + missed_mean) / active_count.clamp_min(1)


@dataclass(frozen=True, slots=True)
class AlignedObjectDynamicsTarget:
    """Loss-only physical-key alignment from prior rows to current queries.

    ``observation_index_by_row`` is ``-1`` when no trustworthy current
    observation exists. It is built only after set matching against external
    physical identity keys, so an early runtime MAP row swap cannot train the
    transition to reproduce that same swap.
    """

    observation_index_by_row: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObjectLifecycleInventoryTarget:
    """Loss-only current-time inventory of physically alive objects.

    This target is intentionally separate from :class:`ObjectSetTarget`: an
    alive object may be fully occluded and therefore absent from the current
    measurement set while still supervising posterior survival.  Listed keys
    are positive survival evidence.  An absent key is negative evidence only
    when ``inventory_complete`` is true. Visibility labels are optional and
    selective; for listed alive objects they describe trustworthy current
    measurement availability conditioned on physical existence, not raw
    geometric visibility.
    """

    alive_identity_keys: tuple[str, ...]
    inventory_complete: bool = False
    visibility: torch.Tensor | None = None
    visibility_supervised: torch.Tensor | None = None


def align_object_lifecycle_inventory(
    targets: Sequence[ObjectLifecycleInventoryTarget | None],
    identity_keys_by_row: Sequence[Sequence[str | None]],
    predicted_valid: torch.Tensor,
    *,
    dtype: torch.dtype,
    previous_targets: Sequence[ObjectLifecycleInventoryTarget | None] | None = None,
    supervise_survival: bool = True,
    supervise_visibility: bool = True,
) -> AlignedObjectLifecycleTarget:
    """Align independent physical inventories to prior posterior rows.

    The alignment consumes only checkpointed loss metadata and target-side
    physical keys.  It does not inspect discovery queries, runtime MAP events or
    posterior predictions, preventing a self-confirming lifecycle label path.
    """

    if predicted_valid.ndim != 2 or predicted_valid.dtype != torch.bool:
        raise ValueError("predicted lifecycle validity must be bool batch-by-row")
    if not dtype.is_floating_point:
        raise ValueError("aligned lifecycle target dtype must be floating point")
    if not isinstance(supervise_survival, bool) or not isinstance(supervise_visibility, bool):
        raise ValueError("lifecycle supervision switches must be boolean")
    if not (supervise_survival or supervise_visibility):
        raise ValueError("lifecycle alignment requires at least one enabled target family")
    batch_size, capacity = predicted_valid.shape
    frozen_targets = tuple(targets)
    frozen_keys = tuple(tuple(keys) for keys in identity_keys_by_row)
    if len(frozen_targets) != batch_size:
        raise ValueError("lifecycle inventory targets must match posterior batch size")
    if len(frozen_keys) != batch_size or any(len(keys) != capacity for keys in frozen_keys):
        raise ValueError("lifecycle identity keys must be batch-by-posterior-row")
    previous_aligned = (
        None
        if previous_targets is None
        else align_object_lifecycle_inventory(
            previous_targets,
            frozen_keys,
            predicted_valid,
            dtype=dtype,
            supervise_survival=False,
            supervise_visibility=True,
        )
    )

    survival = torch.zeros(predicted_valid.shape, device=predicted_valid.device, dtype=dtype)
    survival_supervised = torch.zeros_like(predicted_valid)
    visibility = torch.zeros_like(survival)
    visibility_supervised = torch.zeros_like(predicted_valid)
    valid_cpu = predicted_valid.detach().cpu().tolist()

    for batch_index, (target, row_keys) in enumerate(zip(frozen_targets, frozen_keys, strict=True)):
        present_row_keys = [key for key in row_keys if key is not None]
        if any(not isinstance(key, str) or not key for key in present_row_keys):
            raise ValueError("lifecycle row keys must be nonempty strings or None")
        if len(set(present_row_keys)) != len(present_row_keys):
            raise ValueError("lifecycle row keys must be unique within each sample")
        if any(
            key is not None and not valid_cpu[batch_index][row] for row, key in enumerate(row_keys)
        ):
            raise ValueError("lifecycle row keys cannot name unused posterior rows")
        if target is None:
            continue
        alive_keys, visibility_values, visibility_mask = _validate_inventory_target(
            target,
            device=predicted_valid.device,
            dtype=dtype,
        )
        alive_index = {key: index for index, key in enumerate(alive_keys)}
        alive_destinations: list[int] = []
        alive_sources: list[int] = []
        dead_destinations: list[int] = []
        for row, key in enumerate(row_keys):
            if key is None:
                continue
            alive_row = alive_index.get(key)
            if alive_row is None:
                if target.inventory_complete:
                    dead_destinations.append(row)
                continue
            alive_destinations.append(row)
            alive_sources.append(alive_row)

        alive_destination_index = torch.tensor(
            alive_destinations,
            device=predicted_valid.device,
            dtype=torch.long,
        )
        if supervise_survival and alive_destinations:
            survival[batch_index, alive_destination_index] = 1.0
            survival_supervised[batch_index, alive_destination_index] = True
        if supervise_survival and dead_destinations:
            survival_supervised[
                batch_index,
                torch.tensor(dead_destinations, device=predicted_valid.device, dtype=torch.long),
            ] = True

        if supervise_visibility and visibility_values is not None and alive_destinations:
            if visibility_mask is None:
                raise RuntimeError("validated lifecycle visibility lost its supervision mask")
            visibility_mask_cpu = visibility_mask.detach().cpu().tolist()
            selected_pairs = [
                (destination, source)
                for destination, source in zip(
                    alive_destinations,
                    alive_sources,
                    strict=True,
                )
                if visibility_mask_cpu[source]
            ]
            if selected_pairs:
                destination_index = torch.tensor(
                    [pair[0] for pair in selected_pairs],
                    device=predicted_valid.device,
                    dtype=torch.long,
                )
                source_index = torch.tensor(
                    [pair[1] for pair in selected_pairs],
                    device=predicted_valid.device,
                    dtype=torch.long,
                )
                visibility[batch_index, destination_index] = visibility_values[source_index]
                visibility_supervised[batch_index, destination_index] = True
    return AlignedObjectLifecycleTarget(
        survival=survival,
        survival_supervised=survival_supervised,
        visibility=visibility,
        visibility_supervised=visibility_supervised,
        previous_visibility=(
            torch.zeros_like(visibility)
            if previous_aligned is None
            else previous_aligned.visibility
        ),
        previous_visibility_supervised=(
            torch.zeros_like(visibility_supervised)
            if previous_aligned is None
            else previous_aligned.visibility_supervised
        ),
    )


def _validate_inventory_target(
    target: ObjectLifecycleInventoryTarget,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[str, ...], torch.Tensor | None, torch.Tensor | None]:
    if not isinstance(target, ObjectLifecycleInventoryTarget):
        raise TypeError("lifecycle inventory entries must use ObjectLifecycleInventoryTarget")
    keys = target.alive_identity_keys
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("alive lifecycle identity keys must be nonempty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("alive lifecycle identity keys must be unique")
    if not isinstance(target.inventory_complete, bool):
        raise ValueError("lifecycle inventory_complete must be boolean")
    supplied = target.visibility is not None, target.visibility_supervised is not None
    if any(supplied) and not all(supplied):
        raise ValueError("lifecycle visibility values and supervision are atomic")
    if not all(supplied):
        return keys, None, None
    if target.visibility is None or target.visibility_supervised is None:
        raise RuntimeError("validated lifecycle visibility fields lost atomicity")
    values = target.visibility
    supervised = target.visibility_supervised
    if values.shape != (len(keys),) or not torch.is_floating_point(values):
        raise ValueError("lifecycle visibility must be one floating value per alive object")
    if supervised.shape != (len(keys),) or supervised.dtype != torch.bool:
        raise ValueError("lifecycle visibility supervision must be one bool per alive object")
    if values.device != device or supervised.device != device or values.dtype != dtype:
        raise ValueError("lifecycle visibility labels must share prediction dtype and device")
    if values.requires_grad or not torch.isfinite(values).all():
        raise ValueError("loss-only lifecycle visibility must be finite and detached")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("lifecycle visibility must lie in [0, 1]")
    if (values[~supervised] != 0.0).any():
        raise ValueError("unsupervised lifecycle visibility must be exactly zero")
    return keys, values, supervised


def _detached_rollout_belief(belief: ObjectBeliefBatch) -> ObjectBeliefBatch:
    return ObjectBeliefBatch(
        address_mean=belief.address_mean.detach(),
        content_mean=belief.content_mean.detach(),
        geometry_mean=belief.geometry_mean.detach(),
        geometry_covariance_diag=belief.geometry_covariance_diag.detach(),
        existence_logits=belief.existence_logits.detach(),
        visibility_given_existence_logits=(belief.visibility_given_existence_logits.detach()),
        measurement_age_s=belief.measurement_age_s.detach(),
        valid=belief.valid.detach(),
        age=belief.age.detach(),
    )


def _detach_rollout_lifecycle(belief: ObjectBeliefBatch) -> ObjectBeliefBatch:
    """Prevent geometry rollout from becoming lifecycle pseudo-supervision."""

    return ObjectBeliefBatch(
        address_mean=belief.address_mean,
        content_mean=belief.content_mean,
        geometry_mean=belief.geometry_mean,
        geometry_covariance_diag=belief.geometry_covariance_diag,
        existence_logits=belief.existence_logits.detach(),
        visibility_given_existence_logits=(belief.visibility_given_existence_logits.detach()),
        measurement_age_s=belief.measurement_age_s,
        valid=belief.valid,
        age=belief.age,
    )


def _validate_geometry_rollout_target(
    target: ObjectGeometryRolloutTarget,
    start: ObjectBeliefBatch,
    transition: ActionConditionedObjectTransition,
) -> tuple[int, int]:
    if not isinstance(target, ObjectGeometryRolloutTarget):
        raise TypeError("geometry rollout target must use ObjectGeometryRolloutTarget")
    batch_size, capacity = start.valid.shape
    action_dim = transition.config.action_dim
    geometry_dim = transition.config.geometry_dim
    if not isinstance(target.geometry_contract, PhysicalGeometryContract):
        raise ValueError("rollout geometry requires a physical geometry contract")
    if target.geometry_contract != transition.config.geometry_contract:
        raise ValueError("rollout and transition geometry contracts differ")
    actions = target.executed_actions
    if (
        actions.ndim != 3
        or actions.shape[0] != batch_size
        or actions.shape[2] != action_dim
        or not torch.is_floating_point(actions)
    ):
        raise ValueError("rollout actions must be floating batch-by-horizon-by-action")
    horizon = actions.shape[1]
    if horizon <= 0:
        raise ValueError("geometry rollout horizon must be positive")
    if (
        actions.device != start.address_mean.device
        or actions.dtype != start.address_mean.dtype
        or actions.requires_grad
        or not torch.isfinite(actions).all()
    ):
        raise ValueError("loss-only rollout actions must be finite, detached and colocated")
    delta_t = target.delta_t_s
    if (
        delta_t.shape != (batch_size, horizon)
        or not torch.is_floating_point(delta_t)
        or delta_t.device != actions.device
        or delta_t.dtype != actions.dtype
        or delta_t.requires_grad
        or not torch.isfinite(delta_t).all()
    ):
        raise ValueError("rollout delta_t must be finite detached batch-by-horizon")
    step_valid = target.step_valid
    if (
        step_valid.shape != (batch_size, horizon)
        or step_valid.dtype != torch.bool
        or step_valid.device != actions.device
        or step_valid.requires_grad
    ):
        raise ValueError("rollout step validity must be detached bool batch-by-horizon")
    if not step_valid.any() or not step_valid[:, -1].any():
        raise ValueError("rollout horizon must be minimal and contain at least one valid step")
    if horizon > 1 and ((~step_valid[:, :-1]) & step_valid[:, 1:]).any():
        raise ValueError("rollout step validity must be one contiguous prefix per sample")
    if (delta_t[step_valid] <= 0.0).any():
        raise ValueError("valid rollout delta_t must be positive")
    if (delta_t[~step_valid] != 0.0).any() or (actions[~step_valid] != 0.0).any():
        raise ValueError("invalid rollout action and delta_t padding must be exactly zero")

    geometry = target.geometry
    if geometry.ndim != 4 or geometry.shape[:2] != (batch_size, horizon):
        raise ValueError("rollout geometry must be batch-by-horizon-by-object-by-feature")
    target_capacity = geometry.shape[2]
    if target_capacity <= 0 or geometry.shape[3] != geometry_dim:
        raise ValueError("rollout geometry object/feature dimensions are invalid")
    expected_geometry_shape = (batch_size, horizon, target_capacity, geometry_dim)
    variance = target.geometry_variance
    supervised = target.geometry_supervised
    if variance.shape != expected_geometry_shape or supervised.shape != expected_geometry_shape:
        raise ValueError("rollout geometry tensors must share one exact shape")
    if supervised.dtype != torch.bool or supervised.requires_grad:
        raise ValueError("rollout geometry supervision must be detached boolean")
    if geometry.dtype != variance.dtype:
        raise ValueError("rollout geometry and variance must share one supervision dtype")
    for name, value in (("geometry", geometry), ("geometry variance", variance)):
        if (
            not torch.is_floating_point(value)
            or value.device != actions.device
            or value.requires_grad
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"loss-only rollout {name} must be finite, detached and colocated")
    if supervised.device != actions.device:
        raise ValueError("rollout geometry supervision must be colocated")
    if (variance < 0.0).any():
        raise ValueError("rollout target geometry variance cannot be negative")
    if (geometry[~supervised] != 0.0).any() or (variance[~supervised] != 0.0).any():
        raise ValueError("unsupervised rollout geometry and variance must be exactly zero")

    keys = target.identity_keys
    if len(keys) != batch_size or any(len(sample) != horizon for sample in keys):
        raise ValueError("rollout identity keys must be batch-by-horizon-by-object")
    supervised_object = supervised.any(dim=-1).detach().cpu()
    for batch_index, sample in enumerate(keys):
        for horizon_index, frame in enumerate(sample):
            if len(frame) != target_capacity:
                raise ValueError("rollout identity object axis differs from geometry")
            present = [key for key in frame if key is not None]
            if any(not isinstance(key, str) or not key for key in present):
                raise ValueError("rollout identity keys must be nonempty strings or None")
            if len(set(present)) != len(present):
                raise ValueError("rollout identity keys must be unique within each horizon")
            for object_index, key in enumerate(frame):
                has_supervision = bool(supervised_object[batch_index, horizon_index, object_index])
                if has_supervision and not bool(step_valid[batch_index, horizon_index]):
                    raise ValueError("invalid rollout steps cannot contain geometry supervision")
                if has_supervision != (key is not None):
                    raise ValueError(
                        "each rollout object key must have supervised geometry and vice versa"
                    )
    if not supervised.any():
        raise ValueError("geometry rollout target contains no supervised coordinate")

    start_keys_capacity = capacity
    return horizon, start_keys_capacity


class ObjectGeometryOvershootingCriterion(nn.Module):
    """Bounded autoregressive geometry loss on the production transition.

    The current posterior is detached once. Dynamic state and covariance remain
    differentiable across rollout steps, while lifecycle logits are detached
    between steps so geometry cannot act as uncalibrated survival/detection
    pseudo-supervision. The chronology is ``u[t+h-1] -> g[t+h]``. Its
    prefix/action/future-suffix alignment is a semantic adaptation of the
    corrected sequential rollout in
    ``facebookresearch/jepa-wms@13cf1d9c...::VideoWM.rollout``; no upstream code
    is copied.
    """

    def __init__(self, config: ObjectGeometryOvershootingConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObjectGeometryOvershootingConfig()

    def forward(
        self,
        transition: ActionConditionedObjectTransition,
        start: ObjectBeliefBatch,
        identity_keys_by_row: Sequence[Sequence[str | None]],
        target: ObjectGeometryRolloutTarget,
    ) -> ObjectGeometryOvershootingOutput:
        if not isinstance(transition, ActionConditionedObjectTransition):
            raise TypeError("geometry overshooting requires the production object transition")
        horizon, capacity = _validate_geometry_rollout_target(target, start, transition)
        frozen_row_keys = tuple(tuple(keys) for keys in identity_keys_by_row)
        batch_size = start.valid.shape[0]
        if len(frozen_row_keys) != batch_size or any(
            len(keys) != capacity for keys in frozen_row_keys
        ):
            raise ValueError("rollout start identity keys must be batch-by-posterior-row")
        start_valid_cpu = start.valid.detach().cpu()
        row_by_key: list[dict[str, int]] = []
        for batch_index, keys in enumerate(frozen_row_keys):
            present = [key for key in keys if key is not None]
            if any(not isinstance(key, str) or not key for key in present):
                raise ValueError("rollout start keys must be nonempty strings or None")
            if len(set(present)) != len(present):
                raise ValueError("rollout start keys must be unique within each sample")
            if any(
                key is not None and not bool(start_valid_cpu[batch_index, row])
                for row, key in enumerate(keys)
            ):
                raise ValueError("rollout start keys cannot name unused posterior rows")
            row_by_key.append({key: row for row, key in enumerate(keys) if key is not None})

        belief = _detached_rollout_belief(start)
        horizon_losses: list[torch.Tensor] = []
        matched_predictions = 0
        unaligned_target_objects = 0
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
            predicted_rows: list[torch.Tensor] = []
            predicted_variances: list[torch.Tensor] = []
            target_rows: list[torch.Tensor] = []
            target_variances: list[torch.Tensor] = []
            target_masks: list[torch.Tensor] = []
            for batch_index, frame_keys in enumerate(target.identity_keys):
                for object_index, key in enumerate(frame_keys[horizon_index]):
                    if key is None:
                        continue
                    posterior_row = row_by_key[batch_index].get(key)
                    if posterior_row is None:
                        unaligned_target_objects += 1
                        continue
                    predicted_rows.append(belief.geometry_mean[batch_index, posterior_row])
                    predicted_variances.append(
                        belief.geometry_covariance_diag[batch_index, posterior_row]
                    )
                    target_rows.append(target.geometry[batch_index, horizon_index, object_index])
                    target_variances.append(
                        target.geometry_variance[batch_index, horizon_index, object_index]
                    )
                    target_masks.append(
                        target.geometry_supervised[batch_index, horizon_index, object_index]
                    )
            if not predicted_rows:
                continue
            predicted_geometry = torch.stack(predicted_rows).float()
            expected_geometry = torch.stack(target_rows).float()
            expected_variance = torch.stack(target_variances).float()
            supervised = torch.stack(target_masks)
            predicted_variance = torch.stack(predicted_variances).float()
            terms = F.gaussian_nll_loss(
                predicted_geometry,
                expected_geometry,
                predicted_variance + expected_variance,
                full=False,
                reduction="none",
            )
            coordinate_count = supervised.sum(dim=-1)
            active = coordinate_count > 0
            if active.any():
                object_losses = (terms * supervised).sum(dim=-1) / coordinate_count.clamp_min(1)
                horizon_losses.append(object_losses[active].mean())
                matched_predictions += int(active.sum().detach().cpu().item())

        if not horizon_losses:
            raise ValueError("no rollout geometry target aligns with a current posterior track")
        return ObjectGeometryOvershootingOutput(
            loss=torch.stack(horizon_losses).mean(),
            active_horizons=len(horizon_losses),
            matched_predictions=matched_predictions,
            unaligned_target_objects=unaligned_target_objects,
            maximum_horizon=horizon,
        )


def object_detectability_transition_loss(
    prediction: ObjectPredictionOutput,
    target: AlignedObjectLifecycleTarget,
    *,
    probability_epsilon: float,
) -> ObjectDetectabilityTransitionLossOutput:
    """Directly supervise both branches of the conditional-detectability HMM.

    The previous loss-side label selects the corresponding transition branch.
    Soft previous labels produce the expected complete-data Bernoulli NLL. The
    runtime mixed detection probability is deliberately not supervised here.
    """

    if not isinstance(prediction, ObjectPredictionOutput):
        raise TypeError("detectability transition loss requires an object prediction")
    if (
        isinstance(probability_epsilon, bool)
        or not math.isfinite(probability_epsilon)
        or not 0.0 < probability_epsilon < 0.5
    ):
        raise ValueError("detectability probability epsilon must lie in (0, 0.5)")
    predicted_valid = prediction.belief.valid
    _validate_lifecycle_target(target, predicted_valid)
    supervised = target.visibility_supervised & target.previous_visibility_supervised
    current = target.visibility.float()
    previous = target.previous_visibility.float()
    logit_limit = math.log((1.0 - probability_epsilon) / probability_epsilon)
    detected_terms = F.binary_cross_entropy_with_logits(
        prediction.detectability_if_detected_logits.float().clamp(
            min=-logit_limit,
            max=logit_limit,
        ),
        current,
        reduction="none",
    )
    missed_terms = F.binary_cross_entropy_with_logits(
        prediction.detectability_if_missed_logits.float().clamp(
            min=-logit_limit,
            max=logit_limit,
        ),
        current,
        reduction="none",
    )
    mask = supervised.float()
    detected_weight = previous * mask
    missed_weight = (1.0 - previous) * mask
    detected_loss_sum = (detected_terms * detected_weight).sum()
    missed_loss_sum = (missed_terms * missed_weight).sum()
    previous_detected_mass = detected_weight.sum().detach()
    previous_missed_mass = missed_weight.sum().detach()
    return ObjectDetectabilityTransitionLossOutput(
        loss_sum=detected_loss_sum + missed_loss_sum,
        detected_loss_sum=detected_loss_sum,
        missed_loss_sum=missed_loss_sum,
        supervised_count=supervised.sum().detach(),
        positive_target_mass=(current * mask).sum().detach(),
        negative_target_mass=((1.0 - current) * mask).sum().detach(),
        previous_detected_mass=previous_detected_mass,
        previous_missed_mass=previous_missed_mass,
    )


class ObjectDynamicsCriterion(nn.Module):
    """Calibrate one-step dynamic state, survival and visibility predictions.

    Address is an identity coordinate copied unchanged by the transition.  It
    is therefore excluded from this marginal-EM self-prediction objective:
    under detached one-step streaming it has no trainable prediction path, and
    under a differentiable window a mistaken runtime association could make it
    self-confirming.  Trusted cross-time identity keys supervise address in the
    separate temporal binding criterion.
    """

    def __init__(self, config: ObjectDynamicsLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObjectDynamicsLossConfig()

    def forward(
        self,
        output: PICFCoreOutput,
        lifecycle_target: AlignedObjectLifecycleTarget | None = None,
        dynamics_target: AlignedObjectDynamicsTarget | None = None,
    ) -> ObjectDynamicsLossOutput:
        prediction = output.posterior.prior_prediction
        predicted = prediction.belief
        discovery = output.discovery
        zero = predicted.state_mean.float().sum() * 0.0

        observation_dynamic = (
            torch.cat(
                (discovery.content_mean, discovery.geometry_mean),
                dim=-1,
            )
            .detach()
            .float()
        )
        observation_variance = discovery.geometry_variance.detach().float()
        independently_aligned_predictions = 0
        if dynamics_target is None:
            association_weight = output.posterior.match_probability.detach().float()
            expected_shape = (
                predicted.valid.shape[0],
                predicted.valid.shape[1],
                observation_dynamic.shape[1],
            )
            if association_weight.shape != expected_shape:
                raise ValueError(
                    "posterior match marginals must align prior rows and discovery observations"
                )
            association_weight = association_weight * predicted.valid.unsqueeze(-1)
            predicted_content = predicted.content_mean.float().unsqueeze(2)
            target_content = observation_dynamic[..., : predicted.content_mean.shape[-1]].unsqueeze(
                1
            )
            observation_count = observation_dynamic.shape[1]
            posterior_capacity = predicted.valid.shape[1]
            content_cosine = 1.0 - F.cosine_similarity(
                predicted_content.expand(-1, -1, observation_count, -1),
                target_content.expand(-1, posterior_capacity, -1, -1),
                dim=-1,
            )
            predicted_geometry = predicted.geometry_mean.float().unsqueeze(2)
            target_geometry = observation_dynamic[
                ..., predicted.content_mean.shape[-1] :
            ].unsqueeze(1)
            total_variance = predicted.geometry_covariance_diag.float().unsqueeze(
                2
            ) + observation_variance.unsqueeze(1)
            geometry_nll = F.gaussian_nll_loss(
                predicted_geometry.expand(-1, -1, observation_count, -1),
                target_geometry.expand(-1, posterior_capacity, -1, -1),
                total_variance,
                full=False,
                reduction="none",
            ).mean(dim=-1)
            active_association = association_weight > 0.0
            content_cosine = torch.where(
                active_association,
                content_cosine,
                torch.zeros_like(content_cosine),
            )
            geometry_nll = torch.where(
                active_association,
                geometry_nll,
                torch.zeros_like(geometry_nll),
            )
            association_mass = association_weight.sum()
            normalizer = association_mass.clamp_min(self.config.probability_epsilon)
            content_loss = (association_weight * content_cosine).sum() / normalizer
            geometry_loss = (association_weight * geometry_nll).sum() / normalizer
            matched_count = (association_weight.sum(dim=-1) > self.config.probability_epsilon).sum()
        else:
            row_to_observation = _validate_dynamics_target(
                dynamics_target,
                predicted.valid,
                observation_dynamic.shape[1],
            )
            independently_aligned = row_to_observation >= 0
            safe_observation = row_to_observation.clamp_min(0)
            dynamic_index = safe_observation.unsqueeze(-1).expand(
                -1,
                -1,
                observation_dynamic.shape[-1],
            )
            geometry_index = safe_observation.unsqueeze(-1).expand(
                -1,
                -1,
                observation_variance.shape[-1],
            )
            predicted_dynamic = predicted.dynamic_mean[independently_aligned].float()
            predicted_variance = predicted.geometry_covariance_diag[independently_aligned].float()
            target_dynamic = observation_dynamic.gather(1, dynamic_index)[independently_aligned]
            target_variance = observation_variance.gather(1, geometry_index)[independently_aligned]
            matched_count = independently_aligned.sum()
            independently_aligned_predictions = int(matched_count.detach().cpu().item())
            content_end = predicted.content_mean.shape[-1]
            content_cosine = 1.0 - F.cosine_similarity(
                predicted_dynamic[..., :content_end],
                target_dynamic[..., :content_end],
                dim=-1,
            )
            geometry_nll = F.gaussian_nll_loss(
                predicted_dynamic[..., content_end:],
                target_dynamic[..., content_end:],
                predicted_variance + target_variance,
                full=False,
                reduction="none",
            ).mean(dim=-1)
            content_loss = content_cosine.sum() / matched_count.clamp_min(1)
            geometry_loss = geometry_nll.sum() / matched_count.clamp_min(1)

        survival_loss = zero
        visibility_loss = zero
        lifecycle_predictions = 0
        target_mass_zero = torch.zeros((), device=predicted.valid.device, dtype=torch.float32)
        survival_positive_target_mass = target_mass_zero
        survival_negative_target_mass = target_mass_zero
        visibility_positive_target_mass = target_mass_zero
        visibility_negative_target_mass = target_mass_zero
        visibility_loss_sum = zero
        visibility_detected_loss_sum = zero
        visibility_missed_loss_sum = zero
        visibility_prediction_count = torch.zeros(
            (), dtype=torch.long, device=predicted.valid.device
        )
        visibility_previous_detected_mass = target_mass_zero
        visibility_previous_missed_mass = target_mass_zero
        lifecycle_enabled = self.config.survival_weight > 0.0 or self.config.visibility_weight > 0.0
        if lifecycle_enabled:
            if lifecycle_target is None:
                raise ValueError(
                    "positive lifecycle weights require independent loss-side lifecycle targets"
                )
            _validate_lifecycle_target(lifecycle_target, predicted.valid)
            epsilon = self.config.probability_epsilon
            logit_limit = math.log((1.0 - epsilon) / epsilon)
            lifecycle_count = torch.zeros((), dtype=torch.long, device=predicted.valid.device)
            if self.config.survival_weight > 0.0:
                supervised = lifecycle_target.survival_supervised
                survival_target = lifecycle_target.survival.float()
                survival_logits = prediction.survival_logits.float().clamp(
                    min=-logit_limit,
                    max=logit_limit,
                )
                survival_terms = F.binary_cross_entropy_with_logits(
                    survival_logits,
                    survival_target,
                    reduction="none",
                )
                survival_count = supervised.sum()
                survival_loss = (survival_terms * supervised).sum() / survival_count.clamp_min(1)
                survival_positive_target_mass = (survival_target * supervised).sum().detach()
                survival_negative_target_mass = (
                    ((1.0 - survival_target) * supervised).sum().detach()
                )
                lifecycle_count = lifecycle_count + survival_count
            elif lifecycle_target.survival_supervised.any():
                raise ValueError("survival targets were supplied while survival weight is zero")

            if self.config.visibility_weight > 0.0:
                # Factor lifecycle likelihood as p(E_t) p(V_t | E_t=1).
                # Dead rows train survival only; treating them as visibility
                # negatives duplicates existence and collapses detectability.
                # Adjacent labels identify p(V_t | V_{t-1}, E_t=1)
                # directly; the runtime Chapman-Kolmogorov mixture is not a
                # second, underdetermined training target.
                visibility_output = object_detectability_transition_loss(
                    prediction,
                    lifecycle_target,
                    probability_epsilon=self.config.probability_epsilon,
                )
                visibility_loss = visibility_output.loss
                visibility_loss_sum = visibility_output.loss_sum
                visibility_detected_loss_sum = visibility_output.detected_loss_sum
                visibility_missed_loss_sum = visibility_output.missed_loss_sum
                visibility_prediction_count = visibility_output.supervised_count
                visibility_positive_target_mass = visibility_output.positive_target_mass
                visibility_negative_target_mass = visibility_output.negative_target_mass
                visibility_previous_detected_mass = visibility_output.previous_detected_mass
                visibility_previous_missed_mass = visibility_output.previous_missed_mass
                lifecycle_count = lifecycle_count + visibility_prediction_count
            elif lifecycle_target.visibility_supervised.any():
                raise ValueError("visibility targets were supplied while visibility weight is zero")
            lifecycle_predictions = int(lifecycle_count.detach().cpu().item())
        elif lifecycle_target is not None:
            raise ValueError("lifecycle targets were supplied while lifecycle weights are zero")
        losses = {
            "loss_dynamics_content_cosine": content_loss,
            "loss_dynamics_geometry_nll": geometry_loss,
            "loss_dynamics_survival": survival_loss,
            "loss_dynamics_visibility": visibility_loss,
        }
        losses["loss_dynamics_total"] = (
            self.config.content_cosine_weight * content_loss
            + self.config.geometry_nll_weight * geometry_loss
            + self.config.survival_weight * survival_loss
            + self.config.visibility_weight * visibility_loss
        )
        matched_predictions = int(matched_count.detach().cpu().item())
        return ObjectDynamicsLossOutput(
            losses=losses,
            matched_predictions=matched_predictions,
            independently_aligned_predictions=independently_aligned_predictions,
            lifecycle_predictions=lifecycle_predictions,
            survival_positive_target_mass=survival_positive_target_mass,
            survival_negative_target_mass=survival_negative_target_mass,
            visibility_positive_target_mass=visibility_positive_target_mass,
            visibility_negative_target_mass=visibility_negative_target_mass,
            visibility_loss_sum=visibility_loss_sum,
            visibility_detected_loss_sum=visibility_detected_loss_sum,
            visibility_missed_loss_sum=visibility_missed_loss_sum,
            visibility_prediction_count=visibility_prediction_count,
            visibility_previous_detected_mass=visibility_previous_detected_mass,
            visibility_previous_missed_mass=visibility_previous_missed_mass,
        )


def _validate_dynamics_target(
    target: AlignedObjectDynamicsTarget,
    predicted_valid: torch.Tensor,
    observation_count: int,
) -> torch.Tensor:
    if not isinstance(target, AlignedObjectDynamicsTarget):
        raise TypeError("dynamics alignment must use AlignedObjectDynamicsTarget")
    mapping = target.observation_index_by_row
    if (
        mapping.shape != predicted_valid.shape
        or mapping.dtype != torch.long
        or mapping.device != predicted_valid.device
        or mapping.requires_grad
    ):
        raise ValueError("dynamics alignment must be detached long batch-by-posterior-row")
    if ((mapping < -1) | (mapping >= observation_count)).any():
        raise ValueError("dynamics alignment contains an out-of-range observation")
    if (mapping[~predicted_valid] != -1).any():
        raise ValueError("dynamics alignment cannot name an unused prior row")
    for sample in mapping.detach().cpu().tolist():
        selected = [index for index in sample if index >= 0]
        if len(set(selected)) != len(selected):
            raise ValueError("dynamics alignment must be one-to-one within each sample")
    return mapping


def _validate_lifecycle_target(
    target: AlignedObjectLifecycleTarget,
    predicted_valid: torch.Tensor,
) -> None:
    expected = predicted_valid.shape
    for name in ("survival", "visibility", "previous_visibility"):
        value = getattr(target, name)
        if value.shape != expected or not torch.is_floating_point(value):
            raise ValueError(f"lifecycle {name} must be one floating batch-by-object tensor")
        if value.requires_grad:
            raise ValueError(f"loss-only lifecycle {name} must not require gradients")
        if value.device != predicted_valid.device or not torch.isfinite(value).all():
            raise ValueError(f"lifecycle {name} must be finite and colocated with the prediction")
        if ((value < 0.0) | (value > 1.0)).any():
            raise ValueError(f"lifecycle {name} must lie in [0, 1]")
        supervised = getattr(target, f"{name}_supervised")
        if supervised.shape != expected or supervised.dtype != torch.bool:
            raise ValueError(f"lifecycle {name}_supervised must be a bool batch-by-object tensor")
        if supervised.device != predicted_valid.device:
            raise ValueError(f"lifecycle {name} supervision must be colocated with the prediction")
        if (supervised & ~predicted_valid).any():
            raise ValueError(f"lifecycle {name} cannot supervise an unused posterior row")
        if (value[~supervised] != 0.0).any():
            raise ValueError(f"unsupervised lifecycle {name} targets must be exactly zero")
