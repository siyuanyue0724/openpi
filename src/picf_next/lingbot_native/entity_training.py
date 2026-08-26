"""Clean action-plus-entity objective for the task-independent PICF graph."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from picf_next.lingbot_native.calvin_entity_set import (
    physical_frame_predictions_from_relation,
)
from picf_next.lingbot_native.entity_set_objective import (
    PhysicalFrameAssignment,
    PhysicalFrameTargets,
    PhysicalSetLoss,
    physical_frame_set_loss,
)
from picf_next.lingbot_native.objective import (
    NativeObjectiveConfig,
    combine_native_objective,
)
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
from picf_next.lingbot_native.physical_sequence import (
    PhysicalSequenceAssignment,
    extend_physical_sequence_row_bindings,
    match_physical_sequence_entities,
    physical_frame_assignment_at_time,
)
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveLossInput,
    materialize_native_predictive_terms,
)
from picf_next.lingbot_native.row_binding import RowBindings
from picf_next.objective import ObjectiveTerm, UnifiedObjective, normalized_scalar_term


@dataclass(frozen=True, slots=True)
class TaskIndependentEntityObjectiveConfig:
    """Weights for one host-native entity/action transaction."""

    action_weight: float = 1.0
    entity_weight: float = 1.0
    predictive_weight: float = 0.0
    mask_focal_weight: float = 1.0
    mask_dice_weight: float = 1.0
    existence_weight: float = 1.0
    ownership_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    def __post_init__(self) -> None:
        nonnegative = (
            ("action_weight", self.action_weight),
            ("entity_weight", self.entity_weight),
            ("predictive_weight", self.predictive_weight),
            ("mask_focal_weight", self.mask_focal_weight),
            ("mask_dice_weight", self.mask_dice_weight),
            ("existence_weight", self.existence_weight),
            ("ownership_weight", self.ownership_weight),
            ("focal_gamma", self.focal_gamma),
        )
        for name, value in nonnegative:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(self.focal_alpha, bool)
            or not isinstance(self.focal_alpha, (int, float))
            or not math.isfinite(self.focal_alpha)
            or not 0 <= self.focal_alpha <= 1
        ):
            raise ValueError("focal_alpha must be finite and lie in [0,1]")
        if self.action_weight == 0 and self.entity_weight == 0 and self.predictive_weight == 0:
            raise ValueError("at least one objective family must be active")
        entity_components = (
            self.mask_focal_weight,
            self.mask_dice_weight,
            self.existence_weight,
            self.ownership_weight,
        )
        if self.entity_weight > 0 and not any(value > 0 for value in entity_components):
            raise ValueError("an active entity family requires one positive component weight")


@dataclass(frozen=True, slots=True)
class TaskIndependentEntityObjectiveResult:
    """Auditable losses without a task-row or scorer surface."""

    objective: UnifiedObjective
    frame_losses: tuple[PhysicalSetLoss, ...]


@dataclass(frozen=True, slots=True)
class TaskIndependentPersistentEntityObjectiveResult:
    """Persistent physical losses and the next loss-side episode gauge."""

    objective: UnifiedObjective
    frame_losses: tuple[PhysicalSetLoss, ...]
    assignment: PhysicalSequenceAssignment
    row_bindings_by_batch: tuple[RowBindings, ...]
    predictive_terms: tuple[ObjectiveTerm, ...] = ()
    structural_terms: tuple[ObjectiveTerm, ...] = ()


def _scalar_metric(name: str, value: torch.Tensor) -> ObjectiveTerm:
    return normalized_scalar_term(name, value, weight=0.0)


def _validate_policy_loss(
    official_policy_loss: torch.Tensor | None,
    *,
    config: TaskIndependentEntityObjectiveConfig,
) -> None:
    if config.action_weight > 0:
        if (
            not isinstance(official_policy_loss, torch.Tensor)
            or official_policy_loss.ndim != 0
            or not official_policy_loss.is_floating_point()
            or not torch.isfinite(official_policy_loss)
            or not official_policy_loss.requires_grad
        ):
            raise ValueError("active action training requires one finite attached policy loss")
    elif official_policy_loss is not None:
        raise ValueError("inactive action training requires an absent policy loss")


def materialize_task_independent_structural_terms(
    frame_losses: Sequence[PhysicalSetLoss],
) -> tuple[ObjectiveTerm, ...]:
    """Expose the exact structural family for sequential objective execution."""

    structural_terms: list[ObjectiveTerm] = []
    for time_index, loss in enumerate(frame_losses):
        prefix = f"set/frame_{time_index:03d}"
        structural_terms.extend(
            (
                normalized_scalar_term(f"{prefix}/entities", loss.total),
                _scalar_metric(f"{prefix}/mask_focal", loss.mask_focal),
                _scalar_metric(f"{prefix}/mask_dice", loss.mask_dice),
                _scalar_metric(f"{prefix}/existence_focal", loss.existence_focal),
                _scalar_metric(f"{prefix}/ownership_nll", loss.ownership_nll),
            )
        )
    return tuple(structural_terms)


def _combine_entity_frame_losses(
    *,
    official_policy_loss: torch.Tensor | None,
    frame_losses: Sequence[PhysicalSetLoss],
    predictive_terms: tuple[ObjectiveTerm, ...],
    config: TaskIndependentEntityObjectiveConfig,
) -> UnifiedObjective:
    structural_terms = materialize_task_independent_structural_terms(frame_losses)
    return combine_native_objective(
        official_policy_loss=official_policy_loss,
        predictive_terms=predictive_terms,
        structural_terms=structural_terms,
        config=NativeObjectiveConfig(
            action_weight=config.action_weight,
            predictive_weight=config.predictive_weight,
            structural_weight=config.entity_weight,
        ),
    )


def compose_task_independent_entity_objective(
    *,
    official_policy_loss: torch.Tensor | None,
    relations: Sequence[PhysicalRelationOutput],
    targets: Sequence[PhysicalFrameTargets],
    config: TaskIndependentEntityObjectiveConfig,
) -> TaskIndependentEntityObjectiveResult:
    """Combine official LingBot action risk with prompt-free entity-set risk."""

    if not isinstance(config, TaskIndependentEntityObjectiveConfig):
        raise TypeError("entity training requires its frozen typed config")
    if not relations or len(relations) != len(targets):
        raise ValueError("entity relations and targets require one equal non-empty time axis")
    if any(not isinstance(value, PhysicalRelationOutput) for value in relations):
        raise TypeError("entity training accepts only task-independent physical relations")
    if any(not isinstance(value, PhysicalFrameTargets) for value in targets):
        raise TypeError("entity training accepts only task-free physical targets")
    _validate_policy_loss(official_policy_loss, config=config)
    if config.predictive_weight > 0:
        raise ValueError("current-frame P1 cannot activate the persistent predictive family")

    frame_losses = tuple(
        physical_frame_set_loss(
            physical_frame_predictions_from_relation(relation),
            target,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            mask_focal_weight=config.mask_focal_weight,
            mask_dice_weight=config.mask_dice_weight,
            existence_weight=config.existence_weight,
            ownership_weight=config.ownership_weight,
        )
        for relation, target in zip(relations, targets, strict=True)
    )
    objective = _combine_entity_frame_losses(
        official_policy_loss=official_policy_loss,
        frame_losses=frame_losses,
        predictive_terms=(),
        config=config,
    )
    return TaskIndependentEntityObjectiveResult(
        objective=objective,
        frame_losses=frame_losses,
    )


def compose_task_independent_persistent_entity_objective(
    *,
    official_policy_loss: torch.Tensor | None,
    relations: Sequence[PhysicalRelationOutput],
    targets: Sequence[PhysicalFrameTargets],
    identity_keys_by_batch: tuple[tuple[str, ...], ...],
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
    config: TaskIndependentEntityObjectiveConfig,
    predictive_inputs: Sequence[NativePredictiveLossInput] = (),
) -> TaskIndependentPersistentEntityObjectiveResult:
    """Compose a causal physical sequence without task-row supervision."""

    if not isinstance(config, TaskIndependentEntityObjectiveConfig):
        raise TypeError("persistent entity training requires its frozen typed config")
    if not relations or len(relations) != len(targets):
        raise ValueError("persistent entity relations and targets require one equal time axis")
    if any(not isinstance(value, PhysicalRelationOutput) for value in relations):
        raise TypeError("persistent entity training accepts only physical relations")
    if any(not isinstance(value, PhysicalFrameTargets) for value in targets):
        raise TypeError("persistent entity training accepts only physical targets")
    _validate_policy_loss(official_policy_loss, config=config)
    if bool(predictive_inputs) != (config.predictive_weight > 0):
        raise ValueError(
            "persistent predictive inputs and the predictive family must be active together"
        )

    predictions = tuple(physical_frame_predictions_from_relation(value) for value in relations)
    assignment = match_physical_sequence_entities(
        predictions,
        targets,
        identity_keys_by_batch=identity_keys_by_batch,
        prior_bindings_by_batch=prior_row_bindings_by_batch,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
    )
    frame_assignments: tuple[PhysicalFrameAssignment, ...] = tuple(
        physical_frame_assignment_at_time(assignment, time_index=time_index)
        for time_index in range(len(relations))
    )
    frame_losses = tuple(
        physical_frame_set_loss(
            prediction,
            target,
            assignment=frame_assignment,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            mask_focal_weight=config.mask_focal_weight,
            mask_dice_weight=config.mask_dice_weight,
            existence_weight=config.existence_weight,
            ownership_weight=config.ownership_weight,
        )
        for prediction, target, frame_assignment in zip(
            predictions,
            targets,
            frame_assignments,
            strict=True,
        )
    )
    row_bindings = extend_physical_sequence_row_bindings(
        assignment,
        identity_keys_by_batch=identity_keys_by_batch,
        prior_bindings_by_batch=prior_row_bindings_by_batch,
    )
    predictive_terms = materialize_native_predictive_terms(
        predictive_inputs,
        assignment=assignment,
        expected_track_identity_keys=identity_keys_by_batch,
        sequence_time_count=len(relations),
    )
    structural_terms = materialize_task_independent_structural_terms(frame_losses)
    return TaskIndependentPersistentEntityObjectiveResult(
        objective=combine_native_objective(
            official_policy_loss=official_policy_loss,
            predictive_terms=predictive_terms,
            structural_terms=structural_terms,
            config=NativeObjectiveConfig(
                action_weight=config.action_weight,
                predictive_weight=config.predictive_weight,
                structural_weight=config.entity_weight,
            ),
        ),
        frame_losses=frame_losses,
        assignment=assignment,
        row_bindings_by_batch=row_bindings,
        predictive_terms=predictive_terms,
        structural_terms=structural_terms,
    )
