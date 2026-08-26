"""Decode offline object evidence into a selective unordered set target.

This module is intentionally dataset-neutral. A dataset adapter may project a
visible mask, point track, contact event or other calibrated annotation onto a
native token bank, but only the resulting probabilities enter here. Labels are
returned to the training criterion and are never accepted by the PICF forward
path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from picf_next.geometry import PhysicalGeometryContract
from picf_next.models.evidence import BindingProjectionOutput
from picf_next.models.set_loss import ObjectSetTarget


@dataclass(frozen=True, slots=True)
class ModalityObjectMembership:
    """Per-token offline claims for one modality and one sample.

    ``object_ids`` are sample-local correspondence keys, not semantic classes
    or persistent runtime IDs. Equal keys across modalities say that two labels
    describe the same physical object. ``probability`` is token-by-object and
    may contain overlapping soft claims. ``supervised`` distinguishes unknown
    labels from known context.
    """

    modality: str
    object_ids: tuple[str, ...]
    probability: torch.Tensor
    token_valid: torch.Tensor
    supervised: torch.Tensor
    context_probability: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class ObjectStateTable:
    """Optional complete object-state supervision keyed by sample-local IDs."""

    object_ids: tuple[str, ...]
    address: torch.Tensor | None = None
    content: torch.Tensor | None = None
    geometry: torch.Tensor | None = None
    geometry_variance: torch.Tensor | None = None
    geometry_supervised: torch.Tensor | None = None
    geometry_contract: PhysicalGeometryContract | None = None


def _validate_membership(
    item: ModalityObjectMembership,
    *,
    expected_tokens: int,
    expected_valid: torch.Tensor,
) -> None:
    if not item.modality:
        raise ValueError("target modality must be nonempty")
    if len(set(item.object_ids)) != len(item.object_ids):
        raise ValueError(f"{item.modality} object IDs must be unique")
    if any(not object_id for object_id in item.object_ids):
        raise ValueError(f"{item.modality} object IDs must be nonempty")
    expected_shape = (expected_tokens, len(item.object_ids))
    if item.probability.shape != expected_shape:
        raise ValueError(
            f"{item.modality} membership must have shape {expected_shape}, "
            f"got {tuple(item.probability.shape)}"
        )
    if not torch.is_floating_point(item.probability):
        raise ValueError(f"{item.modality} membership must be floating point")
    if not torch.isfinite(item.probability).all():
        raise ValueError(f"{item.modality} membership contains NaN or infinity")
    if ((item.probability < 0.0) | (item.probability > 1.0)).any():
        raise ValueError(f"{item.modality} membership must lie in [0, 1]")
    if item.token_valid.dtype != torch.bool or item.token_valid.shape != (expected_tokens,):
        raise ValueError(f"{item.modality} target validity must be a bool token vector")
    if item.supervised.dtype != torch.bool or item.supervised.shape != (expected_tokens,):
        raise ValueError(f"{item.modality} supervision must be a bool token vector")
    if (
        item.probability.device != expected_valid.device
        or item.token_valid.device != expected_valid.device
        or item.supervised.device != expected_valid.device
    ):
        raise ValueError(f"{item.modality} targets must share the projection device")
    if not torch.equal(item.token_valid, expected_valid):
        raise ValueError(f"{item.modality} target validity differs from projected evidence")
    if (item.supervised & ~item.token_valid).any():
        raise ValueError(f"{item.modality} cannot supervise invalid evidence")
    if (item.probability[~item.supervised] != 0.0).any():
        raise ValueError(f"{item.modality} unsupervised membership must be exactly zero")
    if item.context_probability is not None:
        context = item.context_probability
        if context.shape != (expected_tokens,) or not torch.is_floating_point(context):
            raise ValueError(f"{item.modality} context probability must be a floating token vector")
        if context.device != expected_valid.device or not torch.isfinite(context).all():
            raise ValueError(
                f"{item.modality} context probability must be finite and share the device"
            )
        if ((context < 0.0) | (context > 1.0)).any():
            raise ValueError(f"{item.modality} context probability must lie in [0, 1]")
        if (context[~item.supervised] != 0.0).any():
            raise ValueError(
                f"{item.modality} unsupervised context probability must be exactly zero"
            )
        total = item.probability.sum(dim=-1) + context
        if not torch.allclose(
            total[item.supervised],
            torch.ones_like(total[item.supervised]),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(
                f"{item.modality} categorical object-plus-context target must sum to one"
            )
    if item.object_ids and (item.probability[item.supervised].sum(dim=0) <= 0.0).any():
        raise ValueError(f"{item.modality} contains an object with no supervised support")


def _exclusive_simplex(probability: torch.Tensor) -> torch.Tensor:
    """Project binary object-vs-context claims onto one categorical simplex.

    Each calibrated claim is converted to odds ``p_j / (1 - p_j)`` relative to
    unit context mass, then all alternatives are normalized. A single claim
    therefore preserves its original probability. Exact hard overlaps divide
    mass only among the hard claims and assign no mass to context.
    """

    hard = probability == 1.0
    has_hard = hard.any(dim=-1, keepdim=True)
    odds = probability / (1.0 - probability).clamp_min(torch.finfo(probability.dtype).eps)
    context = torch.ones(
        (*probability.shape[:-1], 1), dtype=probability.dtype, device=probability.device
    )
    mass = torch.cat((odds, context), dim=-1)
    soft_result = mass / mass.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(mass.dtype).tiny)
    hard_mass = torch.cat((hard.to(probability.dtype), torch.zeros_like(context)), dim=-1)
    hard_result = hard_mass / hard_mass.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.where(has_hard, hard_result, soft_result)


def _validate_group_consistency(
    ownership: torch.Tensor,
    supervised: torch.Tensor,
    group_ids: torch.Tensor,
    modality: str,
) -> None:
    for group_id in torch.unique(group_ids[group_ids >= 0]):
        members = (group_ids == group_id) & supervised
        if members.any() and not torch.all(supervised[group_ids == group_id]):
            raise ValueError(f"{modality} group {int(group_id)} is only partly supervised")
        if members.sum() > 1:
            rows = ownership[members]
            if not torch.allclose(rows, rows[:1].expand_as(rows), atol=1e-6, rtol=1e-6):
                raise ValueError(
                    f"{modality} group {int(group_id)} must share one object assignment"
                )


def _reorder_state(
    value: torch.Tensor | None,
    table_ids: tuple[str, ...],
    object_ids: tuple[str, ...],
    name: str,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim != 2 or value.shape[0] != len(table_ids):
        raise ValueError(f"state {name} must be object-by-feature")
    if value.device != device or not torch.is_floating_point(value):
        raise ValueError(f"state {name} must be floating and share the projection device")
    if not torch.isfinite(value).all():
        raise ValueError(f"state {name} contains NaN or infinity")
    index = {object_id: row for row, object_id in enumerate(table_ids)}
    return torch.stack([value[index[object_id]] for object_id in object_ids], dim=0)


def _reorder_geometry_supervision(
    value: torch.Tensor | None,
    table_ids: tuple[str, ...],
    object_ids: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim != 2 or value.shape[0] != len(table_ids) or value.dtype != torch.bool:
        raise ValueError("state geometry_supervised must be a bool object-by-feature tensor")
    if value.device != device or value.requires_grad:
        raise ValueError("state geometry_supervised must be detached and share the device")
    index = {object_id: row for row, object_id in enumerate(table_ids)}
    return torch.stack([value[index[object_id]] for object_id in object_ids], dim=0)


def build_object_set_target(
    projection: BindingProjectionOutput,
    *,
    batch_index: int,
    memberships: tuple[ModalityObjectMembership, ...],
    state: ObjectStateTable | None = None,
    temporal_identity_by_object: dict[str, str] | None = None,
    object_inventory_complete: bool = False,
) -> ObjectSetTarget:
    """Build one loss-only all-object target aligned to projected token spans.

    ``temporal_identity_by_object`` must come from an explicit physical track
    contract, such as simulator body/link identity or a verified trajectory.
    It is not inferred from a transient query and is never returned to runtime.
    ``object_inventory_complete`` defaults to false and may be set true only
    when the producer proves that the provided object IDs exhaust the current
    observable target set under its annotation ontology. It does not assert
    that occluded physical tracks are dead; lifecycle inventory is a separate
    target. Otherwise unmatched discovery queries are unknown rather than
    non-objects. Unknown token regions must additionally be excluded with each
    membership's ``supervised`` mask.
    """

    if not isinstance(object_inventory_complete, bool):
        raise ValueError("object_inventory_complete must be a bool")
    if not 0 <= batch_index < projection.binding_features.shape[0]:
        raise IndexError("batch_index is outside the projection batch")
    by_modality = {item.modality: item for item in memberships}
    if len(by_modality) != len(memberships):
        raise ValueError("a target modality may appear at most once")
    known_modalities = {span.modality for span in projection.spans}
    unknown = set(by_modality) - known_modalities
    if unknown:
        raise ValueError(f"targets contain unknown modalities: {sorted(unknown)}")

    object_ids = tuple(sorted({key for item in memberships for key in item.object_ids}))
    object_index = {object_id: index for index, object_id in enumerate(object_ids)}
    temporal_identity_keys = None
    if temporal_identity_by_object is not None:
        if set(temporal_identity_by_object) != set(object_ids):
            raise ValueError("temporal identity map must exactly cover target objects")
        temporal_identity_keys = tuple(
            temporal_identity_by_object[object_id] for object_id in object_ids
        )
        if any(not key for key in temporal_identity_keys) or len(
            set(temporal_identity_keys)
        ) != len(temporal_identity_keys):
            raise ValueError("temporal identity keys must be nonempty and unique within one frame")
    total_tokens = projection.total_tokens
    device = projection.binding_features.device
    dtype = projection.binding_features.dtype
    ownership = torch.zeros(total_tokens, len(object_ids) + 1, device=device, dtype=dtype)
    supervised = torch.zeros(total_tokens, device=device, dtype=torch.bool)

    for span in projection.spans:
        item = by_modality.get(span.modality)
        if item is None:
            continue
        expected_valid = projection.current_measurement_valid[batch_index, span.start : span.stop]
        _validate_membership(
            item,
            expected_tokens=span.stop - span.start,
            expected_valid=expected_valid,
        )
        local_probability = torch.zeros(
            item.probability.shape[0], len(object_ids), device=device, dtype=dtype
        )
        for local_index, object_id in enumerate(item.object_ids):
            local_probability[:, object_index[object_id]] = item.probability[:, local_index]
        if item.context_probability is None:
            local_ownership = _exclusive_simplex(local_probability)
        else:
            local_ownership = torch.cat(
                (
                    local_probability,
                    item.context_probability.to(dtype=dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
        local_ownership[~item.supervised] = 0.0
        group_ids = projection.token_group_id[batch_index, span.start : span.stop]
        _validate_group_consistency(local_ownership, item.supervised, group_ids, item.modality)
        ownership[span.start : span.stop] = local_ownership
        supervised[span.start : span.stop] = item.supervised

    if object_ids and not supervised.any() and state is None:
        raise ValueError("object targets require membership or state supervision")

    state_values: dict[str, torch.Tensor | None] = {
        "address": None,
        "content": None,
        "geometry": None,
        "geometry_variance": None,
        "geometry_supervised": None,
    }
    if state is not None:
        if len(set(state.object_ids)) != len(state.object_ids):
            raise ValueError("state object IDs must be unique")
        if set(state.object_ids) != set(object_ids):
            raise ValueError("state object IDs must exactly match membership object IDs")
        for name in ("address", "content", "geometry", "geometry_variance"):
            state_values[name] = _reorder_state(
                getattr(state, name), state.object_ids, object_ids, name, device
            )
        state_values["geometry_supervised"] = _reorder_geometry_supervision(
            state.geometry_supervised,
            state.object_ids,
            object_ids,
            device,
        )
        if state.geometry_supervised is not None and state.geometry is None:
            raise ValueError("state geometry supervision requires geometry values")
        if state.geometry_variance is not None and state.geometry is None:
            raise ValueError("state geometry variance requires geometry values")
        if state.geometry is None:
            if state.geometry_contract is not None:
                raise ValueError("state geometry contract cannot be supplied without geometry")
        elif not isinstance(state.geometry_contract, PhysicalGeometryContract):
            raise ValueError("state geometry requires a physical geometry contract")

    return ObjectSetTarget(
        ownership=ownership,
        token_valid=projection.current_measurement_valid[batch_index],
        token_supervised=supervised,
        object_inventory_complete=object_inventory_complete,
        address=state_values["address"],
        content=state_values["content"],
        geometry=state_values["geometry"],
        geometry_variance=state_values["geometry_variance"],
        geometry_supervised=state_values["geometry_supervised"],
        geometry_contract=state.geometry_contract if state is not None else None,
        temporal_identity_keys=temporal_identity_keys,
    )
