"""One explicit multi-objective training contract for PICF-Next."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from picf_next.posterior import BIRTH_EVENT, MATCH_EVENT, MISS_EVENT

from .binding_loss import MultimodalBindingCriterion, TemporalAddressBindingCriterion
from .core import PICFCoreOutput
from .dynamics_loss import (
    AlignedObjectDynamicsTarget,
    AlignedObjectLifecycleTarget,
    ObjectDetectabilityTransitionLossOutput,
    ObjectDynamicsCriterion,
    ObjectGeometryOvershootingCriterion,
    ObjectGeometryRolloutTarget,
    ObjectLifecycleInventoryTarget,
    align_object_lifecycle_inventory,
    balanced_conditional_detectability_loss,
)
from .set_loss import ObjectSetCriterion, ObjectSetTarget, SetMatch
from .temporal import ActionConditionedObjectTransition


@dataclass(frozen=True, slots=True)
class PICFObjectiveConfig:
    action_weight: float
    set_weight: float
    dynamics_weight: float
    binding_weight: float
    require_temporal_positive_pairs: bool = False

    def __post_init__(self) -> None:
        values = (
            self.action_weight,
            self.set_weight,
            self.dynamics_weight,
            self.binding_weight,
        )
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0.0 for value in values
        ) or not any(value > 0.0 for value in values):
            raise ValueError("objective weights must be nonnegative and not all zero")
        if not isinstance(self.require_temporal_positive_pairs, bool):
            raise ValueError("require_temporal_positive_pairs must be boolean")
        if self.require_temporal_positive_pairs and self.binding_weight <= 0.0:
            raise ValueError("temporal positive pairs can be required only when binding is active")


@dataclass(frozen=True, slots=True)
class PICFObjectiveOutput:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    diagnostics: dict[str, int | float]
    loss_track_keys_by_row: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True, slots=True)
class LossTrackAdvanceOutput:
    """Loss-only physical identity alignment after one or more observations."""

    loss_track_keys_by_row: tuple[tuple[str | None, ...], ...]
    assignment_conflicts: int


class PICFObjective(nn.Module):
    """Combine the four documented loss families without hidden schedules."""

    def __init__(
        self,
        config: PICFObjectiveConfig,
        *,
        set_criterion: ObjectSetCriterion | None = None,
        dynamics_criterion: ObjectDynamicsCriterion | None = None,
        geometry_overshooting_criterion: ObjectGeometryOvershootingCriterion | None = None,
        binding_criterion: MultimodalBindingCriterion | None = None,
        temporal_binding_criterion: TemporalAddressBindingCriterion | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.set_criterion = set_criterion or ObjectSetCriterion()
        self.dynamics_criterion = dynamics_criterion or ObjectDynamicsCriterion()
        self.geometry_overshooting_criterion = (
            geometry_overshooting_criterion or ObjectGeometryOvershootingCriterion()
        )
        if (
            self.geometry_overshooting_criterion.config.weight > 0.0
            and config.dynamics_weight <= 0.0
        ):
            raise ValueError("geometry overshooting requires an active dynamics objective")
        self.binding_criterion = binding_criterion or MultimodalBindingCriterion()
        self.temporal_binding_criterion = (
            temporal_binding_criterion
            or TemporalAddressBindingCriterion(self.binding_criterion.config)
        )

    @torch.no_grad()
    def advance_loss_tracks(
        self,
        core_outputs: Sequence[PICFCoreOutput],
        set_targets: Sequence[Sequence[ObjectSetTarget]],
        *,
        initial_loss_track_keys_by_row: Sequence[Sequence[str | None]] | None = None,
    ) -> LossTrackAdvanceOutput:
        """Advance target-side row identities without creating a training loss.

        Fixed-parameter prefix replay needs the mature posterior row identities
        at the suffix boundary, but it must not retain a prefix computation
        graph or optimize prefix losses. Matching remains loss-only and cannot
        influence the deploy-visible core forward.
        """

        outputs = tuple(core_outputs)
        targets = tuple(tuple(frame) for frame in set_targets)
        if not outputs or len(outputs) != len(targets):
            raise ValueError("loss-track advance requires aligned nonempty frames")
        matches = tuple(
            self.set_criterion(output.discovery, frame_targets).matches
            for output, frame_targets in zip(outputs, targets, strict=True)
        )
        keys, conflicts, _alignment = _advance_loss_track_keys(
            outputs,
            targets,
            matches,
            initial_loss_track_keys_by_row,
        )
        return LossTrackAdvanceOutput(
            loss_track_keys_by_row=keys,
            assignment_conflicts=conflicts,
        )

    def forward(
        self,
        core_outputs: Sequence[PICFCoreOutput],
        *,
        action_loss: torch.Tensor | None,
        set_targets: Sequence[Sequence[ObjectSetTarget]] | None,
        lifecycle_targets: Sequence[Sequence[ObjectLifecycleInventoryTarget | None]] | None = None,
        initial_lifecycle_targets: Sequence[ObjectLifecycleInventoryTarget | None] | None = None,
        detectability_replay: ObjectDetectabilityTransitionLossOutput | None = None,
        initial_loss_track_keys_by_row: Sequence[Sequence[str | None]] | None = None,
        geometry_rollout_target: ObjectGeometryRolloutTarget | None = None,
        transition: ActionConditionedObjectTransition | None = None,
    ) -> PICFObjectiveOutput:
        outputs: tuple[PICFCoreOutput, ...] = tuple(core_outputs)
        if not outputs:
            raise ValueError("PICF objective requires at least one core transition")
        reference_output = next(iter(outputs))
        final_output = next(reversed(outputs))
        reference = reference_output.discovery.existence_logits
        zero = reference.sum() * 0.0
        if self.config.action_weight > 0.0:
            if (
                not isinstance(action_loss, torch.Tensor)
                or action_loss.ndim != 0
                or not torch.is_floating_point(action_loss)
                or action_loss.device != reference.device
                or not torch.isfinite(action_loss)
            ):
                raise ValueError("positive action weight requires one finite scalar action loss")
        elif action_loss is not None:
            raise ValueError("action loss was supplied while its objective weight is zero")

        dynamics_config = self.dynamics_criterion.config
        geometry_overshooting_active = self.geometry_overshooting_criterion.config.weight > 0.0
        if geometry_overshooting_active:
            if geometry_rollout_target is None:
                raise ValueError(
                    "active geometry overshooting requires one loss-only future rollout target"
                )
            if transition is None:
                raise ValueError(
                    "active geometry overshooting requires the production object transition"
                )
        elif geometry_rollout_target is not None:
            raise ValueError("geometry rollout target was supplied while overshooting is inactive")
        lifecycle_supervision_active = self.config.dynamics_weight > 0.0 and (
            dynamics_config.survival_weight > 0.0 or dynamics_config.visibility_weight > 0.0
        )
        requires_targets = (
            self.config.set_weight > 0.0
            or self.config.binding_weight > 0.0
            or lifecycle_supervision_active
            or geometry_overshooting_active
        )
        # Set matching is needed by the set objective and by cross-time address
        # binding. A normal one-transition multimodal-binding update consumes
        # token ownership directly and must not pay for a loss-side Hungarian
        # solve whose result cannot affect the objective.
        requires_set_matches = (
            self.config.set_weight > 0.0
            or geometry_overshooting_active
            or (
                self.config.binding_weight > 0.0
                and (len(outputs) > 1 or initial_loss_track_keys_by_row is not None)
            )
            or lifecycle_supervision_active
        )
        if requires_targets:
            if set_targets is None or len(set_targets) != len(outputs):
                raise ValueError("set/binding objectives require targets for every transition")
            targets = tuple(tuple(items) for items in set_targets)
        else:
            if set_targets is not None:
                raise ValueError("targets were supplied while set and binding weights are zero")
            targets = ()

        if self.config.dynamics_weight > 0.0:
            if lifecycle_supervision_active:
                if lifecycle_targets is None or len(lifecycle_targets) != len(outputs):
                    raise ValueError(
                        "active lifecycle supervision requires an inventory target batch "
                        "for every core transition"
                    )
                if initial_loss_track_keys_by_row is None:
                    raise ValueError(
                        "active lifecycle supervision requires checkpointed loss track keys"
                    )
                lifecycle_inventory = tuple(tuple(frame) for frame in lifecycle_targets)
                initial_lifecycle_inventory = (
                    None if initial_lifecycle_targets is None else tuple(initial_lifecycle_targets)
                )
                if detectability_replay is not None and not isinstance(
                    detectability_replay,
                    ObjectDetectabilityTransitionLossOutput,
                ):
                    raise TypeError(
                        "detectability replay must use ObjectDetectabilityTransitionLossOutput"
                    )
                if dynamics_config.visibility_weight <= 0.0 and (
                    initial_lifecycle_inventory is not None or detectability_replay is not None
                ):
                    raise ValueError(
                        "detectability history was supplied while visibility weight is zero"
                    )
            else:
                if lifecycle_targets is not None:
                    raise ValueError(
                        "lifecycle targets were supplied while lifecycle weights are zero"
                    )
                lifecycle_inventory = ()
                initial_lifecycle_inventory = None
                if initial_lifecycle_targets is not None or detectability_replay is not None:
                    raise ValueError(
                        "detectability history was supplied while lifecycle supervision is inactive"
                    )
        else:
            if lifecycle_targets is not None:
                raise ValueError("lifecycle targets were supplied while dynamics weight is zero")
            lifecycle_inventory = ()
            initial_lifecycle_inventory = None
            if initial_lifecycle_targets is not None or detectability_replay is not None:
                raise ValueError("detectability history was supplied while dynamics weight is zero")

        set_losses = []
        binding_outputs = []
        set_outputs = []
        for index, output in enumerate(outputs):
            if requires_set_matches:
                set_output = self.set_criterion(output.discovery, targets[index])
                set_outputs.append(set_output)
                if self.config.set_weight > 0.0:
                    set_losses.append(set_output.total)
            if self.config.binding_weight > 0.0:
                binding_outputs.append(self.binding_criterion(output.projection, targets[index]))

        frozen_matches = tuple(result.matches for result in set_outputs)
        if lifecycle_supervision_active:
            if initial_loss_track_keys_by_row is None:
                raise RuntimeError("active lifecycle supervision lost its validated track keys")
            (
                lifecycle,
                dynamics_alignment,
                loss_track_keys,
                track_conflicts,
            ) = _align_lifecycle_and_advance_loss_tracks(
                outputs,
                targets,
                frozen_matches,
                lifecycle_inventory,
                initial_loss_track_keys_by_row,
                initial_lifecycle_inventory=initial_lifecycle_inventory,
                supervise_survival=dynamics_config.survival_weight > 0.0,
                supervise_visibility=dynamics_config.visibility_weight > 0.0,
            )
        else:
            lifecycle = (None,) * len(outputs)
            loss_track_keys, track_conflicts, dynamics_alignment = _advance_loss_track_keys(
                outputs,
                targets if frozen_matches else (),
                frozen_matches,
                initial_loss_track_keys_by_row,
            )
        dynamics_outputs = [
            self.dynamics_criterion(
                output,
                lifecycle[index],
                dynamics_alignment[index],
            )
            for index, output in enumerate(outputs)
            if self.config.dynamics_weight > 0.0
        ]

        action = action_loss if action_loss is not None else zero
        set_loss = torch.stack(set_losses).mean() if set_losses else zero
        set_components = {
            name: (
                torch.stack([result.losses[name] for result in set_outputs]).mean()
                if set_outputs
                else zero
            )
            for name in (
                "loss_existence",
                "loss_localization_confidence",
                "loss_ownership_ce",
                "loss_ownership_dice",
                "loss_address_cosine",
                "loss_content_cosine",
                "loss_geometry_mean",
                "loss_geometry_calibration",
                "loss_geometry",
            )
        }
        dynamics_content_cosine = (
            torch.stack(
                [result.losses["loss_dynamics_content_cosine"] for result in dynamics_outputs]
            ).mean()
            if dynamics_outputs
            else zero
        )
        dynamics_geometry_nll = (
            torch.stack(
                [result.losses["loss_dynamics_geometry_nll"] for result in dynamics_outputs]
            ).mean()
            if dynamics_outputs
            else zero
        )
        dynamics_survival = (
            torch.stack(
                [result.losses["loss_dynamics_survival"] for result in dynamics_outputs]
            ).mean()
            if dynamics_outputs
            else zero
        )
        suffix_detected_loss_sum = sum(
            (result.visibility_detected_loss_sum for result in dynamics_outputs),
            start=zero,
        )
        suffix_missed_loss_sum = sum(
            (result.visibility_missed_loss_sum for result in dynamics_outputs),
            start=zero,
        )
        suffix_previous_detected_mass = sum(
            (result.visibility_previous_detected_mass for result in dynamics_outputs),
            start=zero.detach(),
        )
        suffix_previous_missed_mass = sum(
            (result.visibility_previous_missed_mass for result in dynamics_outputs),
            start=zero.detach(),
        )
        if detectability_replay is not None:
            _validate_detectability_replay(detectability_replay, reference)
            detected_loss_sum = suffix_detected_loss_sum + detectability_replay.detected_loss_sum
            missed_loss_sum = suffix_missed_loss_sum + detectability_replay.missed_loss_sum
            previous_detected_mass = (
                suffix_previous_detected_mass + detectability_replay.previous_detected_mass
            )
            previous_missed_mass = (
                suffix_previous_missed_mass + detectability_replay.previous_missed_mass
            )
        else:
            detected_loss_sum = suffix_detected_loss_sum
            missed_loss_sum = suffix_missed_loss_sum
            previous_detected_mass = suffix_previous_detected_mass
            previous_missed_mass = suffix_previous_missed_mass
        dynamics_visibility = balanced_conditional_detectability_loss(
            detected_loss_sum,
            previous_detected_mass,
            missed_loss_sum,
            previous_missed_mass,
        )
        dynamics_one_step = (
            dynamics_config.content_cosine_weight * dynamics_content_cosine
            + dynamics_config.geometry_nll_weight * dynamics_geometry_nll
            + dynamics_config.survival_weight * dynamics_survival
            + dynamics_config.visibility_weight * dynamics_visibility
        )
        dynamics_loss = dynamics_one_step
        geometry_overshooting_output = None
        geometry_overshooting_loss = zero
        weighted_geometry_overshooting_loss = zero
        if geometry_overshooting_active:
            if transition is None or geometry_rollout_target is None:
                raise RuntimeError("active geometry overshooting lost its validated inputs")
            geometry_overshooting_output = self.geometry_overshooting_criterion(
                transition,
                final_output.posterior.belief,
                loss_track_keys,
                geometry_rollout_target,
            )
            geometry_overshooting_loss = geometry_overshooting_output.loss
            weighted_geometry_overshooting_loss = (
                self.geometry_overshooting_criterion.config.weight * geometry_overshooting_loss
            )
            dynamics_loss = dynamics_one_step + weighted_geometry_overshooting_loss
        active_multimodal_losses = [
            result.loss
            for result in binding_outputs
            if result.positive_pairs > 0 and result.negative_pairs > 0
        ]
        multimodal_binding_loss = (
            torch.stack(active_multimodal_losses).mean() if active_multimodal_losses else zero
        )
        temporal_binding_loss = zero
        temporal_binding_output = None
        temporal_credit_available = len(outputs) > 1 or initial_loss_track_keys_by_row is not None
        if self.config.binding_weight > 0.0 and temporal_credit_available:
            initial_prediction = reference_output.posterior.prior_prediction.belief
            temporal_binding_output = self.temporal_binding_criterion(
                tuple(output.discovery for output in outputs),
                targets,
                frozen_matches,
                initial_address=(
                    initial_prediction.address_mean
                    if initial_loss_track_keys_by_row is not None
                    else None
                ),
                initial_valid=(
                    initial_prediction.valid if initial_loss_track_keys_by_row is not None else None
                ),
                initial_identity_keys_by_row=initial_loss_track_keys_by_row,
                relation_logit_scale=(reference_output.posterior.address_relation_logit_scale),
                relation_logit_bias=(reference_output.posterior.address_relation_logit_bias),
            )
            temporal_binding_loss = temporal_binding_output.loss
            if (
                self.config.require_temporal_positive_pairs
                and temporal_binding_output.covered_eligible_samples
                != temporal_binding_output.eligible_samples
            ):
                raise ValueError(
                    "this temporal-credit update failed to realize every supervised "
                    "cross-time identity relation"
                )
        elif self.config.require_temporal_positive_pairs:
            raise ValueError(
                "this temporal-credit update has only one transition and therefore no "
                "supervised cross-time address positive pairs"
            )
        active_binding_losses = []
        if active_multimodal_losses:
            active_binding_losses.append(multimodal_binding_loss)
        if (
            temporal_binding_output is not None
            and temporal_binding_output.positive_pairs > 0
            and temporal_binding_output.negative_pairs > 0
        ):
            active_binding_losses.append(temporal_binding_loss)
        binding_loss = torch.stack(active_binding_losses).mean() if active_binding_losses else zero
        # Missing modalities must not create evidence or dilute active relation
        # graphs, but DDP still requires every trainable relation scalar to
        # participate in every iteration. This exact-zero dependency preserves
        # both contracts without fabricating a cross-modal pair.
        for parameter in self.binding_criterion.parameters():
            if parameter.requires_grad:
                binding_loss = binding_loss + parameter.float().sum() * 0.0
        weighted_action = self.config.action_weight * action
        weighted_set = self.config.set_weight * set_loss
        weighted_dynamics = self.config.dynamics_weight * dynamics_loss
        weighted_binding = self.config.binding_weight * binding_loss
        total = weighted_action + weighted_set + weighted_dynamics + weighted_binding
        lifecycle_target_masses = (
            torch.stack(
                [
                    torch.stack(
                        (
                            result.survival_positive_target_mass,
                            result.survival_negative_target_mass,
                            result.visibility_positive_target_mass,
                            result.visibility_negative_target_mass,
                        )
                    )
                    for result in dynamics_outputs
                ]
            )
            .sum(dim=0)
            .detach()
            .float()
            .cpu()
            .tolist()
            if dynamics_outputs
            else [0.0, 0.0, 0.0, 0.0]
        )
        visibility_previous_masses = (
            torch.stack(
                [
                    torch.stack(
                        (
                            result.visibility_previous_detected_mass,
                            result.visibility_previous_missed_mass,
                        )
                    )
                    for result in dynamics_outputs
                ]
            ).sum(dim=0)
            if dynamics_outputs
            else torch.zeros(2, device=reference.device, dtype=torch.float32)
        )
        replay_predictions = 0
        if detectability_replay is not None:
            lifecycle_target_masses[2] += float(
                detectability_replay.positive_target_mass.detach().cpu().item()
            )
            lifecycle_target_masses[3] += float(
                detectability_replay.negative_target_mass.detach().cpu().item()
            )
            visibility_previous_masses = visibility_previous_masses + torch.stack(
                (
                    detectability_replay.previous_detected_mass,
                    detectability_replay.previous_missed_mass,
                )
            )
            replay_predictions = int(detectability_replay.supervised_count.detach().cpu().item())
        visibility_previous_mass_values = visibility_previous_masses.detach().float().cpu().tolist()
        diagnostics = {
            "transitions": len(outputs),
            "target_samples": sum(len(frame) for frame in targets),
            "target_objects": sum(target.num_objects for frame in targets for target in frame),
            "target_supervised_tokens": sum(
                int(target.supervision_valid.sum().item()) for frame in targets for target in frame
            ),
            "complete_inventory_samples": sum(
                int(target.object_inventory_complete) for frame in targets for target in frame
            ),
            "set_matches": sum(
                int(match.prediction_indices.numel())
                for result in set_outputs
                for match in result.matches
            ),
            "dynamics_matched_predictions": sum(
                result.matched_predictions for result in dynamics_outputs
            ),
            "dynamics_independently_aligned_predictions": sum(
                result.independently_aligned_predictions for result in dynamics_outputs
            ),
            "geometry_overshooting_active_horizons": (
                geometry_overshooting_output.active_horizons
                if geometry_overshooting_output is not None
                else 0
            ),
            "geometry_overshooting_matched_predictions": (
                geometry_overshooting_output.matched_predictions
                if geometry_overshooting_output is not None
                else 0
            ),
            "geometry_overshooting_unaligned_target_objects": (
                geometry_overshooting_output.unaligned_target_objects
                if geometry_overshooting_output is not None
                else 0
            ),
            "geometry_overshooting_maximum_horizon": (
                geometry_overshooting_output.maximum_horizon
                if geometry_overshooting_output is not None
                else 0
            ),
            "lifecycle_predictions": sum(
                result.lifecycle_predictions for result in dynamics_outputs
            )
            + replay_predictions,
            "lifecycle_survival_positive_target_mass": lifecycle_target_masses[0],
            "lifecycle_survival_negative_target_mass": lifecycle_target_masses[1],
            "lifecycle_detection_positive_target_mass": lifecycle_target_masses[2],
            "lifecycle_detection_negative_target_mass": lifecycle_target_masses[3],
            "lifecycle_detection_previous_detected_mass": visibility_previous_mass_values[0],
            "lifecycle_detection_previous_missed_mass": visibility_previous_mass_values[1],
            "lifecycle_detection_prefix_replay_predictions": replay_predictions,
            "multimodal_active_transitions": sum(
                int(result.positive_pairs > 0) for result in binding_outputs
            ),
            "multimodal_object_views": sum(
                result.object_modality_views for result in binding_outputs
            ),
            "multimodal_positive_pairs": sum(result.positive_pairs for result in binding_outputs),
            "multimodal_negative_pairs": sum(result.negative_pairs for result in binding_outputs),
            "temporal_address_views": (
                temporal_binding_output.address_views if temporal_binding_output is not None else 0
            ),
            "temporal_null_address_views": (
                temporal_binding_output.null_address_views
                if temporal_binding_output is not None
                else 0
            ),
            "temporal_positive_pairs": (
                temporal_binding_output.positive_pairs if temporal_binding_output is not None else 0
            ),
            "temporal_negative_pairs": (
                temporal_binding_output.negative_pairs if temporal_binding_output is not None else 0
            ),
            "temporal_null_negative_pairs": (
                temporal_binding_output.null_negative_pairs
                if temporal_binding_output is not None
                else 0
            ),
            "temporal_eligible_samples": (
                temporal_binding_output.eligible_samples
                if temporal_binding_output is not None
                else 0
            ),
            "temporal_covered_eligible_samples": (
                temporal_binding_output.covered_eligible_samples
                if temporal_binding_output is not None
                else 0
            ),
            "active_binding_families": len(active_binding_losses),
            "loss_track_assignment_conflicts": track_conflicts,
            "loss_track_rows": sum(
                int(key is not None) for sample in loss_track_keys for key in sample
            ),
        }
        return PICFObjectiveOutput(
            loss=total,
            losses={
                "loss_action": action,
                "loss_set": set_loss,
                "loss_set_existence": set_components["loss_existence"],
                "loss_set_localization_confidence": set_components["loss_localization_confidence"],
                "loss_set_ownership_ce": set_components["loss_ownership_ce"],
                "loss_set_ownership_dice": set_components["loss_ownership_dice"],
                "loss_set_address_cosine": set_components["loss_address_cosine"],
                "loss_set_content_cosine": set_components["loss_content_cosine"],
                "loss_set_geometry_mean": set_components["loss_geometry_mean"],
                "loss_set_geometry_calibration": set_components["loss_geometry_calibration"],
                "loss_set_geometry": set_components["loss_geometry"],
                "loss_dynamics": dynamics_loss,
                "loss_dynamics_one_step": dynamics_one_step,
                "loss_dynamics_geometry_overshooting": geometry_overshooting_loss,
                "loss_dynamics_geometry_overshooting_weighted": (
                    weighted_geometry_overshooting_loss
                ),
                "loss_dynamics_content_cosine": dynamics_content_cosine,
                "loss_dynamics_geometry_nll": dynamics_geometry_nll,
                "loss_dynamics_survival": dynamics_survival,
                "loss_dynamics_visibility": dynamics_visibility,
                "loss_binding": binding_loss,
                "loss_binding_multimodal": multimodal_binding_loss,
                "loss_binding_temporal_address": temporal_binding_loss,
                "loss_weighted_action": weighted_action,
                "loss_weighted_set": weighted_set,
                "loss_weighted_dynamics": weighted_dynamics,
                "loss_weighted_binding": weighted_binding,
                "loss_total": total,
            },
            diagnostics=diagnostics,
            loss_track_keys_by_row=loss_track_keys,
        )


def _validate_detectability_replay(
    replay: ObjectDetectabilityTransitionLossOutput,
    reference: torch.Tensor,
) -> None:
    if not isinstance(replay, ObjectDetectabilityTransitionLossOutput):
        raise TypeError("detectability replay uses an invalid output type")
    if (
        replay.supervised_count.ndim != 0
        or replay.supervised_count.dtype != torch.long
        or replay.supervised_count.device != reference.device
        or replay.supervised_count.requires_grad
        or replay.supervised_count < 0
    ):
        raise ValueError("detectability replay count must be one nonnegative colocated integer")
    for name in (
        "loss_sum",
        "detected_loss_sum",
        "missed_loss_sum",
        "positive_target_mass",
        "negative_target_mass",
        "previous_detected_mass",
        "previous_missed_mass",
    ):
        value = getattr(replay, name)
        if (
            value.ndim != 0
            or not torch.is_floating_point(value)
            or value.device != reference.device
            or not torch.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"detectability replay {name} must be one finite nonnegative scalar")
        if name not in {"loss_sum", "detected_loss_sum", "missed_loss_sum"} and value.requires_grad:
            raise ValueError(f"detectability replay {name} must be detached")
    if not torch.isclose(
        replay.loss_sum,
        replay.detected_loss_sum + replay.missed_loss_sum,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("detectability replay branch losses differ from their total")
    count = replay.supervised_count.float()
    for name, total_mass in (
        (
            "current target mass",
            replay.positive_target_mass + replay.negative_target_mass,
        ),
        (
            "previous target mass",
            replay.previous_detected_mass + replay.previous_missed_mass,
        ),
    ):
        if not torch.isclose(total_mass, count, atol=1e-4, rtol=1e-5):
            raise ValueError(f"detectability replay {name} differs from its count")


def _advance_loss_track_keys(
    outputs: tuple[PICFCoreOutput, ...],
    targets: tuple[tuple[ObjectSetTarget, ...], ...],
    matches: tuple[tuple[SetMatch, ...], ...],
    initial_keys: Sequence[Sequence[str | None]] | None,
) -> tuple[
    tuple[tuple[str | None, ...], ...],
    int,
    tuple[AlignedObjectDynamicsTarget | None, ...],
]:
    """Advance trusted physical-key/row alignment without endorsing MAP swaps."""

    current = _validated_initial_loss_track_keys(outputs, initial_keys)
    if bool(targets) != bool(matches):
        raise ValueError("loss track targets and set matches must be supplied together")
    if targets and (len(targets) != len(outputs) or len(matches) != len(outputs)):
        raise ValueError("loss track targets and matches must align with every transition")
    conflicts = 0
    dynamics_alignment: list[AlignedObjectDynamicsTarget | None] = []
    for frame_index, output in enumerate(outputs):
        frame_targets = targets[frame_index] if targets else ()
        frame_matches = matches[frame_index] if matches else ()
        trusted_tracks_available = initial_keys is not None or frame_index > 0
        dynamics_alignment.append(
            _align_one_dynamics_frame(output, frame_targets, frame_matches, current)
            if frame_targets and trusted_tracks_available
            else None
        )
        current, frame_conflicts = _advance_one_loss_track_frame(
            output,
            frame_targets,
            frame_matches,
            current,
        )
        conflicts += frame_conflicts
    return current, conflicts, tuple(dynamics_alignment)


def _align_lifecycle_and_advance_loss_tracks(
    outputs: tuple[PICFCoreOutput, ...],
    targets: tuple[tuple[ObjectSetTarget, ...], ...],
    matches: tuple[tuple[SetMatch, ...], ...],
    lifecycle_inventory: tuple[tuple[ObjectLifecycleInventoryTarget | None, ...], ...],
    initial_keys: Sequence[Sequence[str | None]],
    *,
    initial_lifecycle_inventory: tuple[ObjectLifecycleInventoryTarget | None, ...] | None,
    supervise_survival: bool,
    supervise_visibility: bool,
) -> tuple[
    tuple[AlignedObjectLifecycleTarget, ...],
    tuple[AlignedObjectDynamicsTarget, ...],
    tuple[tuple[str | None, ...], ...],
    int,
]:
    """Align lifecycle labels before advancing each frame's loss-only row map."""

    if (
        len(targets) != len(outputs)
        or len(matches) != len(outputs)
        or len(lifecycle_inventory) != len(outputs)
    ):
        raise ValueError("lifecycle, set targets and matches must align with every transition")
    current = _validated_initial_loss_track_keys(outputs, initial_keys)
    aligned = []
    dynamics_alignment = []
    conflicts = 0
    previous_inventory = initial_lifecycle_inventory
    for frame_index, output in enumerate(outputs):
        prior = output.posterior.prior_prediction.belief
        aligned.append(
            align_object_lifecycle_inventory(
                lifecycle_inventory[frame_index],
                current,
                prior.valid,
                # Lifecycle labels are loss-only Bernoulli targets. The
                # criterion explicitly evaluates both sides in float32, so
                # label precision must not depend on the AMP prediction dtype.
                dtype=torch.float32,
                previous_targets=previous_inventory,
                supervise_survival=supervise_survival,
                supervise_visibility=supervise_visibility,
            )
        )
        dynamics_alignment.append(
            _align_one_dynamics_frame(
                output,
                targets[frame_index],
                matches[frame_index],
                current,
            )
        )
        current, frame_conflicts = _advance_one_loss_track_frame(
            output,
            targets[frame_index],
            matches[frame_index],
            current,
        )
        previous_inventory = lifecycle_inventory[frame_index]
        conflicts += frame_conflicts
    return tuple(aligned), tuple(dynamics_alignment), current, conflicts


def _align_one_dynamics_frame(
    output: PICFCoreOutput,
    targets: Sequence[ObjectSetTarget],
    matches: Sequence[SetMatch],
    identity_keys_by_row: tuple[tuple[str | None, ...], ...],
) -> AlignedObjectDynamicsTarget:
    """Align prior rows to current discovery queries through physical keys."""

    predicted_valid = output.posterior.prior_prediction.belief.valid
    batch_size, capacity = predicted_valid.shape
    observation_count = output.discovery.existence_logits.shape[1]
    if len(targets) != batch_size or len(matches) != batch_size:
        raise ValueError("dynamics targets and matches must agree with the posterior batch")
    if len(identity_keys_by_row) != batch_size or any(
        len(keys) != capacity for keys in identity_keys_by_row
    ):
        raise ValueError("dynamics physical keys must be batch-by-posterior-row")

    mapping = torch.full(
        (batch_size, capacity),
        -1,
        dtype=torch.long,
        device=predicted_valid.device,
    )
    valid_cpu = predicted_valid.detach().cpu()
    for batch_index, (target, match, row_keys) in enumerate(
        zip(targets, matches, identity_keys_by_row, strict=True)
    ):
        identities = target.temporal_identity_keys
        if identities is None:
            continue
        if len(identities) != target.num_objects:
            raise ValueError("dynamics target identity keys must align with target objects")
        prediction_indices = match.prediction_indices.detach().cpu().tolist()
        target_indices = match.target_indices.detach().cpu().tolist()
        if len(prediction_indices) != len(target_indices):
            raise ValueError("dynamics set match is malformed")
        query_by_identity: dict[str, int] = {}
        for query, target_index in zip(prediction_indices, target_indices, strict=True):
            if not 0 <= query < observation_count or not 0 <= target_index < len(identities):
                raise ValueError("dynamics set match index is out of range")
            identity = identities[target_index]
            if identity in query_by_identity:
                raise ValueError("dynamics target identity was matched more than once")
            query_by_identity[identity] = query
        for row, identity in enumerate(row_keys):
            if identity is None:
                continue
            if not bool(valid_cpu[batch_index, row]):
                raise ValueError("dynamics physical key cannot name an unused prior row")
            query = query_by_identity.get(identity)
            if query is not None:
                mapping[batch_index, row] = query
    return AlignedObjectDynamicsTarget(mapping)


def _validated_initial_loss_track_keys(
    outputs: tuple[PICFCoreOutput, ...],
    initial_keys: Sequence[Sequence[str | None]] | None,
) -> tuple[tuple[str | None, ...], ...]:
    batch_size, capacity = outputs[0].posterior.prior_prediction.belief.valid.shape
    if initial_keys is None:
        return tuple((None,) * capacity for _batch in range(batch_size))
    current = tuple(tuple(keys) for keys in initial_keys)
    if len(current) != batch_size or any(len(keys) != capacity for keys in current):
        raise ValueError("initial loss track keys must be batch-by-posterior-row")
    initial_valid = outputs[0].posterior.prior_prediction.belief.valid.detach().cpu()
    for batch_index, keys in enumerate(current):
        present = [key for key in keys if key is not None]
        if any(not isinstance(key, str) or not key for key in present):
            raise ValueError("initial loss track keys must be nonempty strings or None")
        if len(set(present)) != len(present):
            raise ValueError("initial loss track keys must be unique within each sample")
        if any(
            key is not None and not bool(initial_valid[batch_index, row])
            for row, key in enumerate(keys)
        ):
            raise ValueError("initial loss track keys cannot name unused posterior rows")
    return current


def _advance_one_loss_track_frame(
    output: PICFCoreOutput,
    targets: Sequence[ObjectSetTarget],
    matches: Sequence[SetMatch],
    previous_keys: tuple[tuple[str | None, ...], ...],
) -> tuple[tuple[tuple[str | None, ...], ...], int]:
    batch_size, capacity = output.posterior.belief.valid.shape
    if len(previous_keys) != batch_size:
        raise ValueError("loss track rows differ from the posterior batch")
    if bool(targets) != bool(matches):
        raise ValueError("loss track frame targets and matches must be supplied together")
    if targets and (len(targets) != batch_size or len(matches) != batch_size):
        raise ValueError("loss track frame targets and matches must match posterior batch size")
    final_valid = output.posterior.belief.valid.detach().cpu()
    event_type = output.posterior.event_type.detach().cpu()
    if event_type.shape != final_valid.shape:
        raise ValueError("loss track lifecycle events must align with posterior rows")
    retained_identity = (event_type == MATCH_EVENT) | (event_type == MISS_EVENT)
    occupied = retained_identity | (event_type == BIRTH_EVENT)
    if not torch.equal(occupied, final_valid):
        raise ValueError("loss track lifecycle events disagree with posterior validity")
    observation_to_row = output.posterior.observation_to_posterior.detach().cpu()
    next_keys: list[tuple[str | None, ...]] = []
    conflicts = 0
    for batch_index in range(batch_size):
        previous = previous_keys[batch_index]
        row_keys: list[str | None] = [
            key if bool(retained_identity[batch_index, row]) else None
            for row, key in enumerate(previous)
        ]
        if not targets:
            next_keys.append(tuple(row_keys))
            continue
        identity = targets[batch_index].temporal_identity_keys
        if identity is None:
            next_keys.append(tuple(row_keys))
            continue
        if any(not isinstance(key, str) or not key for key in identity) or len(
            set(identity)
        ) != len(identity):
            raise ValueError("temporal identity keys must be nonempty and unique")
        match = matches[batch_index]
        if match.prediction_indices.numel() != match.target_indices.numel():
            raise ValueError("loss track set match is malformed")
        prediction_indices = match.prediction_indices.detach().cpu().tolist()
        target_indices = match.target_indices.detach().cpu().tolist()
        previous_row = {key: row for row, key in enumerate(previous) if key is not None}
        for query, target_index in zip(prediction_indices, target_indices, strict=True):
            if not 0 <= query < observation_to_row.shape[1] or not 0 <= target_index < len(
                identity
            ):
                raise ValueError("loss track set match index is out of range")
            key = identity[target_index]
            old_row = previous_row.get(key)
            runtime_row = int(observation_to_row[batch_index, query])
            if old_row is not None and bool(retained_identity[batch_index, old_row]):
                row_keys[old_row] = key
                if runtime_row >= 0 and runtime_row != old_row:
                    conflicts += 1
                continue
            if not 0 <= runtime_row < capacity or not bool(final_valid[batch_index, runtime_row]):
                conflicts += 1
                continue
            occupant = row_keys[runtime_row]
            if occupant is not None and occupant != key:
                conflicts += 1
                continue
            row_keys[runtime_row] = key
        next_keys.append(tuple(row_keys))
    return tuple(next_keys), conflicts
