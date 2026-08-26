"""Fixed-parameter reset/replay training for the PICF temporal core."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from picf_next.models.core import PICFCore, PICFCoreOutput
from picf_next.models.dynamics_loss import (
    AlignedObjectLifecycleTarget,
    ObjectDetectabilityTransitionLossOutput,
    ObjectGeometryRolloutTarget,
    ObjectLifecycleInventoryTarget,
    align_object_lifecycle_inventory,
    object_detectability_transition_loss,
)
from picf_next.models.evidence import NativeTokenBank
from picf_next.models.objective import PICFObjective, PICFObjectiveOutput
from picf_next.models.set_loss import ObjectSetTarget
from picf_next.models.temporal import ObjectBeliefBatch, empty_object_belief

STATIONARY_TEMPORAL_EXECUTION_CONTRACT: Final = "reset-replay-one-parameter-version.v3"

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


def _detach_belief(belief: ObjectBeliefBatch) -> ObjectBeliefBatch:
    return ObjectBeliefBatch(**{name: getattr(belief, name).detach() for name in _BELIEF_FIELDS})


def _concatenate_beliefs(beliefs: list[ObjectBeliefBatch]) -> ObjectBeliefBatch:
    if not beliefs:
        raise ValueError("cannot concatenate an empty posterior replay")
    return ObjectBeliefBatch(
        **{
            name: torch.cat([getattr(belief, name) for belief in beliefs], dim=0)
            for name in _BELIEF_FIELDS
        }
    )


def _concatenate_lifecycle_targets(
    targets: list[AlignedObjectLifecycleTarget],
) -> AlignedObjectLifecycleTarget:
    if not targets:
        raise ValueError("cannot concatenate empty lifecycle replay targets")
    return AlignedObjectLifecycleTarget(
        **{
            name: torch.cat([getattr(target, name) for target in targets], dim=0)
            for name in (
                "survival",
                "survival_supervised",
                "visibility",
                "visibility_supervised",
                "previous_visibility",
                "previous_visibility_supervised",
            )
        }
    )


def _replay_autocast_context(reference: torch.Tensor) -> AbstractContextManager[object]:
    """Keep no-grad replay out of an enclosing autocast weight cache.

    Autocast caches low-precision parameter copies for the lifetime of its
    outer context. If a module is first evaluated under ``no_grad``, those
    cached copies can remain detached when the same module is evaluated by the
    trainable suffix. Replay must use the same compute dtype without publishing
    its parameter casts into that cache.
    """

    device_type = reference.device.type
    if not torch.is_autocast_enabled(device_type):
        return nullcontext()
    return torch.autocast(
        device_type=device_type,
        dtype=torch.get_autocast_dtype(device_type),
        enabled=True,
        cache_enabled=False,
    )


@dataclass(frozen=True, slots=True)
class StationaryTemporalObservation:
    """One deploy-visible observation in a reset/replay clip."""

    native_banks: tuple[NativeTokenBank, ...]
    previous_executed_action: torch.Tensor
    delta_t_s: torch.Tensor

    def __post_init__(self) -> None:
        if not self.native_banks or any(
            not isinstance(bank, NativeTokenBank) for bank in self.native_banks
        ):
            raise TypeError("temporal observation requires nonempty native token banks")
        batch_size = self.native_banks[0].tokens.shape[0]
        if any(bank.tokens.shape[0] != batch_size for bank in self.native_banks):
            raise ValueError("temporal observation native-bank batches differ")
        action = self.previous_executed_action
        delta_t = self.delta_t_s
        if action.ndim != 2 or action.shape[0] != batch_size or not torch.is_floating_point(action):
            raise ValueError("previous action must be floating batch-by-action")
        if (
            delta_t.shape != (batch_size,)
            or not torch.is_floating_point(delta_t)
            or delta_t.device != action.device
            or delta_t.dtype != action.dtype
        ):
            raise ValueError("delta_t must be floating batch-aligned with previous action")
        if any(bank.tokens.device != action.device for bank in self.native_banks):
            raise ValueError("temporal observation tensors must share one device")

    @property
    def batch_size(self) -> int:
        return self.previous_executed_action.shape[0]


@dataclass(frozen=True, slots=True)
class StationaryTemporalSupervision:
    """Loss-only labels paired with one observation after its core forward."""

    set_targets: tuple[ObjectSetTarget, ...]
    lifecycle_targets: tuple[ObjectLifecycleInventoryTarget | None, ...]

    def __post_init__(self) -> None:
        if not self.set_targets or any(
            not isinstance(target, ObjectSetTarget) for target in self.set_targets
        ):
            raise TypeError("temporal supervision requires nonempty object-set targets")
        if len(self.lifecycle_targets) != len(self.set_targets) or any(
            target is not None and not isinstance(target, ObjectLifecycleInventoryTarget)
            for target in self.lifecycle_targets
        ):
            raise TypeError("temporal lifecycle targets must align with set targets")


StationaryTemporalSupervisionBuilder = Callable[[int], StationaryTemporalSupervision]
StationaryTemporalGeometryBuilder = Callable[[], ObjectGeometryRolloutTarget | None]


@dataclass(frozen=True, slots=True)
class StationaryTemporalTrainingOutput:
    """One suffix objective; no recurrent state is accepted by a later call."""

    objective: PICFObjectiveOutput
    train_outputs: tuple[PICFCoreOutput, ...]
    final_belief: ObjectBeliefBatch
    prefix_length: int
    train_length: int
    prefix_assignment_conflicts: int
    execution_contract: str = STATIONARY_TEMPORAL_EXECUTION_CONTRACT


class StationaryTemporalCoreTrainer(nn.Module):
    """Replay a mature state and train a bounded suffix under one parameter version."""

    def __init__(self, core: PICFCore, objective: PICFObjective, *, capacity: int) -> None:
        super().__init__()
        if not isinstance(core, PICFCore) or not isinstance(objective, PICFObjective):
            raise TypeError("stationary temporal training requires PICFCore and PICFObjective")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("stationary temporal capacity must be a positive integer")
        if objective.config.action_weight != 0.0:
            raise ValueError("temporal-core training cannot include an action objective")
        if not any(
            weight > 0.0
            for weight in (
                objective.config.set_weight,
                objective.config.dynamics_weight,
                objective.config.binding_weight,
            )
        ):
            raise ValueError("temporal-core training requires a structural objective")
        self.core = core
        self.objective = objective
        self.capacity = capacity

    def forward(
        self,
        observations: tuple[StationaryTemporalObservation, ...],
        *,
        prefix_length: int,
        supervision_builder: StationaryTemporalSupervisionBuilder,
        geometry_builder: StationaryTemporalGeometryBuilder | None = None,
    ) -> StationaryTemporalTrainingOutput:
        if not observations:
            raise ValueError("temporal observations must be nonempty")
        if not callable(supervision_builder):
            raise TypeError("temporal supervision builder must be callable")
        if geometry_builder is not None and not callable(geometry_builder):
            raise TypeError("temporal geometry builder must be callable")
        if (
            not isinstance(prefix_length, int)
            or isinstance(prefix_length, bool)
            or prefix_length < 0
            or prefix_length >= len(observations)
        ):
            raise ValueError("prefix_length must leave at least one train transition")
        batch_size = observations[0].batch_size
        if any(observation.batch_size != batch_size for observation in observations):
            raise ValueError("temporal clip batch size changed across time")

        def build_supervision(frame_index: int) -> StationaryTemporalSupervision:
            targets = supervision_builder(frame_index)
            if not isinstance(targets, StationaryTemporalSupervision):
                raise TypeError("temporal supervision builder returned an invalid value")
            if len(targets.set_targets) != batch_size:
                raise ValueError("temporal supervision batch differs from observations")
            return targets

        reference = observations[0].native_banks[0].tokens
        dynamics_config = self.objective.dynamics_criterion.config
        lifecycle_active = self.objective.config.dynamics_weight > 0.0 and (
            dynamics_config.survival_weight > 0.0 or dynamics_config.visibility_weight > 0.0
        )
        detectability_active = lifecycle_active and dynamics_config.visibility_weight > 0.0
        belief = empty_object_belief(
            self.core.posterior_filter.config,
            batch_size=batch_size,
            capacity=self.capacity,
            device=reference.device,
            dtype=reference.dtype,
        )
        loss_track_keys: tuple[tuple[str | None, ...], ...] | None = None
        previous_supervision: StationaryTemporalSupervision | None = None
        replay_beliefs: list[ObjectBeliefBatch] = []
        replay_actions: list[torch.Tensor] = []
        replay_delta_t: list[torch.Tensor] = []
        replay_targets: list[AlignedObjectLifecycleTarget] = []
        prefix_conflicts = 0
        with torch.no_grad(), _replay_autocast_context(reference):
            for frame_index, observation in enumerate(observations[:prefix_length]):
                prior_belief = _detach_belief(belief)
                prior_loss_track_keys = loss_track_keys
                output = self.core(
                    observation.native_banks,
                    belief,
                    observation.previous_executed_action,
                    observation.delta_t_s,
                )
                targets = build_supervision(frame_index)
                if (
                    detectability_active
                    and prior_loss_track_keys is not None
                    and previous_supervision is not None
                ):
                    replay_beliefs.append(prior_belief)
                    replay_actions.append(observation.previous_executed_action.detach())
                    replay_delta_t.append(observation.delta_t_s.detach())
                    replay_targets.append(
                        align_object_lifecycle_inventory(
                            targets.lifecycle_targets,
                            prior_loss_track_keys,
                            output.posterior.prior_prediction.belief.valid,
                            dtype=torch.float32,
                            previous_targets=previous_supervision.lifecycle_targets,
                            supervise_survival=False,
                            supervise_visibility=True,
                        )
                    )
                alignment = self.objective.advance_loss_tracks(
                    (output,),
                    (targets.set_targets,),
                    initial_loss_track_keys_by_row=loss_track_keys,
                )
                belief = _detach_belief(output.posterior.belief)
                loss_track_keys = alignment.loss_track_keys_by_row
                previous_supervision = targets
                prefix_conflicts += alignment.assignment_conflicts

        if loss_track_keys is None:
            loss_track_keys = tuple((None,) * self.capacity for _ in range(batch_size))

        detectability_replay: ObjectDetectabilityTransitionLossOutput | None = None
        if replay_beliefs:
            replay_prediction = self.core.posterior_filter.transition(
                _concatenate_beliefs(replay_beliefs),
                torch.cat(replay_actions, dim=0),
                torch.cat(replay_delta_t, dim=0),
            )
            detectability_replay = object_detectability_transition_loss(
                replay_prediction,
                _concatenate_lifecycle_targets(replay_targets),
                probability_epsilon=dynamics_config.probability_epsilon,
            )

        train_outputs = []
        train_supervision = []
        for frame_index, observation in enumerate(
            observations[prefix_length:],
            start=prefix_length,
        ):
            output = self.core(
                observation.native_banks,
                belief,
                observation.previous_executed_action,
                observation.delta_t_s,
            )
            train_outputs.append(output)
            train_supervision.append(build_supervision(frame_index))
            belief = output.posterior.belief

        targets_active = (
            self.objective.config.set_weight > 0.0
            or self.objective.config.binding_weight > 0.0
            or lifecycle_active
            or self.objective.geometry_overshooting_criterion.config.weight > 0.0
        )
        result = self.objective(
            tuple(train_outputs),
            action_loss=None,
            set_targets=(
                tuple(frame.set_targets for frame in train_supervision) if targets_active else None
            ),
            lifecycle_targets=(
                tuple(frame.lifecycle_targets for frame in train_supervision)
                if lifecycle_active
                else None
            ),
            initial_lifecycle_targets=(
                previous_supervision.lifecycle_targets
                if lifecycle_active and previous_supervision is not None
                else None
            ),
            detectability_replay=detectability_replay,
            initial_loss_track_keys_by_row=loss_track_keys,
            geometry_rollout_target=(geometry_builder() if geometry_builder is not None else None),
            transition=self.core.posterior_filter.transition,
        )
        return StationaryTemporalTrainingOutput(
            objective=result,
            train_outputs=tuple(train_outputs),
            final_belief=_detach_belief(belief),
            prefix_length=prefix_length,
            train_length=len(train_outputs),
            prefix_assignment_conflicts=prefix_conflicts,
        )
