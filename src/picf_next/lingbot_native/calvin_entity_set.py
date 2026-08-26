"""Loss-side CALVIN adapter for the task-independent entity-set objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_target_request import NativeCALVINStructuralTargetRequest
from picf_next.lingbot_native.calvin_supervision import (
    NativeCALVINSequenceTargetBundle,
    build_native_calvin_sequence_target_bundle,
)
from picf_next.lingbot_native.entity_set_objective import (
    PhysicalFrameAssignment,
    PhysicalFramePredictions,
    PhysicalFrameTargets,
)
from picf_next.lingbot_native.modalities import NO_RELATION_TARGET
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
from picf_next.lingbot_native.row_binding import RowBindings, normalize_row_bindings


@dataclass(frozen=True, slots=True)
class PhysicalCALVINFrameTargetBundle:
    """Task-free frame targets and their loss-side simulator identities."""

    targets: PhysicalFrameTargets
    identity_keys_by_batch: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.targets, PhysicalFrameTargets):
            raise TypeError("CALVIN physical frame bundle requires physical targets")
        if len(self.identity_keys_by_batch) != self.targets.masks.shape[0]:
            raise ValueError("CALVIN physical identities differ from the frame batch")


def physical_frame_row_bindings(
    bundle: PhysicalCALVINFrameTargetBundle,
    assignment: PhysicalFrameAssignment,
    *,
    capacity: int,
) -> tuple[RowBindings, ...]:
    """Materialize loss-side physical identities for detached lane auditing.

    The returned bindings are metadata attached after the deploy-visible host
    forward. They never enter the model and therefore cannot leak simulator
    identities into entity discovery.
    """

    if not isinstance(bundle, PhysicalCALVINFrameTargetBundle):
        raise TypeError("physical row bindings require one CALVIN target bundle")
    if not isinstance(assignment, PhysicalFrameAssignment):
        raise TypeError("physical row bindings require one set assignment")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("physical row-binding capacity must be positive")
    row_to_track = assignment.row_to_track
    if row_to_track.shape != (len(bundle.identity_keys_by_batch), capacity):
        raise ValueError("physical assignment differs from CALVIN batch and capacity")

    resolved: list[RowBindings] = []
    for batch_index, identity_keys in enumerate(bundle.identity_keys_by_batch):
        bindings: list[tuple[str, int]] = []
        for row_index, track_index in enumerate(row_to_track[batch_index].tolist()):
            if track_index < 0:
                continue
            if track_index >= len(identity_keys):
                raise ValueError("physical assignment references an absent CALVIN identity")
            bindings.append((identity_keys[track_index], row_index))
        resolved.append(normalize_row_bindings(bindings, capacity=capacity))
    return tuple(resolved)


def physical_frame_predictions_from_relation(
    relation: PhysicalRelationOutput,
) -> PhysicalFramePredictions:
    """Expose Qwen plus typed supervised native surfaces under one row gauge."""

    if not isinstance(relation, PhysicalRelationOutput):
        raise TypeError("physical frame predictions require a task-independent relation")
    surfaces = tuple(
        surface
        for surface in relation.relation_surfaces
        if surface.target_kind != NO_RELATION_TARGET
    )
    return PhysicalFramePredictions(
        support_logits=torch.cat(
            (relation.support_logits, *(surface.support_logits for surface in surfaces)),
            dim=1,
        ),
        ownership_log_probability=torch.cat(
            (
                relation.ownership_log_probability,
                *(surface.ownership_log_probability for surface in surfaces),
            ),
            dim=1,
        ),
        existence_logits=relation.existence_logits,
        sensor_valid=torch.cat(
            (relation.structural_valid, *(surface.sensor_valid for surface in surfaces)),
            dim=1,
        ),
    )


def physical_calvin_frame_targets(
    bundle: NativeCALVINSequenceTargetBundle,
    *,
    time_index: int,
) -> PhysicalCALVINFrameTargetBundle:
    """Drop every task field while preserving audited physical supervision."""

    if not isinstance(bundle, NativeCALVINSequenceTargetBundle):
        raise TypeError("CALVIN physical conversion requires a native sequence bundle")
    targets = bundle.targets
    time = targets.masks.shape[1]
    if (
        isinstance(time_index, bool)
        or not isinstance(time_index, int)
        or not 0 <= time_index < time
    ):
        raise IndexError("CALVIN physical frame index is outside the local sequence")
    masks = targets.masks[:, time_index]
    mask_valid = targets.mask_valid[:, time_index]
    observed = targets.token_observed_fraction[:, time_index]
    visible = (masks * mask_valid.to(masks.dtype) * observed.unsqueeze(1).to(masks.dtype)).sum(
        dim=-1
    ) > 0
    visible &= targets.track_valid & ~targets.capacity_censored
    physical = PhysicalFrameTargets(
        masks=masks,
        mask_valid=mask_valid,
        existence=visible.to(masks.dtype),
        existence_valid=visible,
        track_valid=targets.track_valid,
        capacity_censored=targets.capacity_censored,
        token_observed_fraction=observed,
        inventory_exhaustive=targets.inventory_exhaustive[:, time_index],
        token_measure_weight=targets.token_measure[:, time_index],
        exclusive_ownership=targets.exclusive_ownership,
    )
    if physical.masks.shape[-1] != targets.masks.shape[-1]:
        raise RuntimeError("CALVIN physical conversion changed the sensor token axis")
    return PhysicalCALVINFrameTargetBundle(
        targets=physical,
        identity_keys_by_batch=bundle.identity_keys_by_batch,
    )


def build_task_independent_calvin_targets(
    *,
    requests_by_time: Sequence[Sequence[NativeCALVINStructuralTargetRequest]],
    model_inputs_by_time: Sequence[Mapping[str, Any]],
    relations: Sequence[PhysicalRelationOutput],
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    patch_size: int,
    merge_size: int,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
) -> tuple[PhysicalCALVINFrameTargetBundle, ...]:
    """Project audited CALVIN instances without resolving a task identity."""

    bundle = build_native_calvin_sequence_target_bundle(
        requests_by_time=requests_by_time,
        model_inputs_by_time=model_inputs_by_time,
        relations=relations,
        physical_sidecar=physical_sidecar,
        capacity=capacity,
        task_identity_resolver=None,
        patch_size=patch_size,
        merge_size=merge_size,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
    )
    if bundle.targets.task_valid.any() or bundle.targets.task_relevance.any():
        raise RuntimeError("task-independent CALVIN projection produced a task target")
    return tuple(
        physical_calvin_frame_targets(bundle, time_index=time_index)
        for time_index in range(len(relations))
    )
