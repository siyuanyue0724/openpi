"""Physical co-reference from the host attention projections."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from picf_next.unified.lifecycle import FootprintEvidence, footprint_evidence


@dataclass(frozen=True, slots=True)
class CoreferenceOutput:
    logits: torch.Tensor
    evidence: FootprintEvidence


@dataclass(frozen=True, slots=True)
class GroupedRelationEvidence:
    message: torch.Tensor
    support: torch.Tensor
    robust_log_likelihood_ratio: torch.Tensor
    available: torch.Tensor
    valid_footprint_mass: torch.Tensor


def tie_assignment_logits_by_group(
    assignment_logits: torch.Tensor,
    footprint: torch.Tensor,
    valid: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    """Give one source-known physical token group a shared assignment law.

    A group identifies an indivisible observation such as all dense tokens from
    one tactile contact; it does not identify the destination object. Weighted
    logit pooling is invariant to splitting a token into identical sub-tokens
    and retains gradients to every grouped token.
    """

    if assignment_logits.ndim != 3:
        raise ValueError("assignment_logits must have shape [batch, tokens, destinations]")
    if min(assignment_logits.shape) <= 0:
        raise ValueError("grouped assignment axes must be non-empty")
    if not assignment_logits.is_floating_point():
        raise TypeError("assignment_logits must be floating point")
    if not torch.isfinite(assignment_logits).all():
        raise ValueError("assignment_logits must be finite")
    if footprint.shape != assignment_logits.shape[:2] or valid.shape != footprint.shape:
        raise ValueError("footprint and valid must match assignment token axes")
    if not footprint.is_floating_point():
        raise TypeError("footprint must be floating point")
    if group_ids.shape != footprint.shape or group_ids.dtype != torch.long:
        raise ValueError("group_ids must be long and match assignment token axes")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be boolean")
    if (
        assignment_logits.device != group_ids.device
        or footprint.device != group_ids.device
        or valid.device != group_ids.device
    ):
        raise ValueError("grouped assignment tensors must share one device")
    if (group_ids < -1).any():
        raise ValueError("group_ids must use -1 for independent tokens")
    if ((group_ids >= 0) & ~valid).any():
        raise ValueError("invalid tokens cannot belong to a physical token group")
    if (footprint < 0).any() or not torch.isfinite(footprint).all():
        raise ValueError("grouped assignment footprint must be finite and non-negative")

    active = group_ids >= 0
    member_indices = active.nonzero(as_tuple=False)
    if member_indices.numel() == 0:
        return assignment_logits

    # Include the batch index in each group key so equal source-local IDs in
    # different samples never interact.  This vectorized reduction scales with
    # active grouped tokens instead of issuing one Python/GPU synchronization
    # per contact group.
    group_keys = torch.stack(
        (member_indices[:, 0], group_ids[active]),
        dim=-1,
    )
    unique_groups, inverse = torch.unique(group_keys, dim=0, return_inverse=True)
    group_count = unique_groups.shape[0]
    accumulation_dtype = (
        torch.float32
        if assignment_logits.dtype in (torch.float16, torch.bfloat16)
        else assignment_logits.dtype
    )
    weights = footprint[active].to(dtype=accumulation_dtype)
    mass = torch.zeros(group_count, dtype=weights.dtype, device=weights.device)
    mass.index_add_(0, inverse, weights)
    if (mass <= 0).any():
        raise ValueError("every physical token group requires positive footprint mass")

    pooled = torch.zeros(
        group_count,
        assignment_logits.shape[-1],
        dtype=accumulation_dtype,
        device=assignment_logits.device,
    )
    pooled.index_add_(
        0,
        inverse,
        assignment_logits[active].to(accumulation_dtype) * weights.unsqueeze(-1),
    )
    pooled = pooled / mass.unsqueeze(-1)
    tied = assignment_logits.clone()
    tied[active] = pooled[inverse].to(assignment_logits.dtype)
    return tied


def repeat_kv_heads(value: torch.Tensor, query_heads: int) -> torch.Tensor:
    if value.ndim < 2 or value.shape[-2] <= 0:
        raise ValueError("value must contain a non-empty key/value-head axis")
    if type(query_heads) is not int:
        raise TypeError("query_heads must be a Python int")
    if query_heads <= 0:
        raise ValueError("query_heads must be positive")
    key_heads = value.shape[-2]
    if key_heads == query_heads:
        return value
    if query_heads % key_heads:
        raise ValueError("query head count must be divisible by key/value head count")
    return value.repeat_interleave(query_heads // key_heads, dim=-2)


def shared_qk_coreference(
    belief_and_context_queries: torch.Tensor,
    sensor_keys: torch.Tensor,
    footprint: torch.Tensor,
    valid: torch.Tensor,
    *,
    geometry_bias: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    robust_clip: float = 8.0,
) -> CoreferenceOutput:
    """Normalize shared attention Q/K scores over beliefs plus context/null.

    The last query is the context/null destination.  No independent identity
    projection or semantic scorer is introduced.
    """

    if belief_and_context_queries.ndim != 4 or sensor_keys.ndim != 4:
        raise ValueError("queries and keys must have shape [batch, tokens, heads, head_dim]")
    if min((*belief_and_context_queries.shape, *sensor_keys.shape[1:])) <= 0:
        raise ValueError("co-reference Q/K axes must be non-empty")
    if not belief_and_context_queries.is_floating_point() or not sensor_keys.is_floating_point():
        raise TypeError("co-reference queries and keys must be floating point")
    if (
        not torch.isfinite(belief_and_context_queries).all()
        or not torch.isfinite(sensor_keys).all()
    ):
        raise ValueError("co-reference queries and keys must be finite")
    if belief_and_context_queries.device != sensor_keys.device:
        raise ValueError("co-reference queries and keys must share one device")
    if belief_and_context_queries.dtype != sensor_keys.dtype:
        raise ValueError("co-reference queries and keys must share one dtype")
    if belief_and_context_queries.shape[0] != sensor_keys.shape[0]:
        raise ValueError("query and sensor batches must match")
    if belief_and_context_queries.shape[-1] != sensor_keys.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if belief_and_context_queries.shape[1] < 2:
        raise ValueError("co-reference requires at least one belief and one context query")
    keys = repeat_kv_heads(sensor_keys, belief_and_context_queries.shape[-2])
    scale = belief_and_context_queries.shape[-1] ** -0.5
    per_head = torch.einsum("bihd,bnhd->bnih", belief_and_context_queries, keys)
    logits = per_head.mean(dim=-1) * scale
    if geometry_bias is not None:
        if geometry_bias.shape != logits.shape:
            raise ValueError("geometry_bias must match [batch, sensor_tokens, beliefs+1]")
        if not geometry_bias.is_floating_point() or not torch.isfinite(geometry_bias).all():
            raise ValueError("geometry_bias must be finite floating point")
        if geometry_bias.device != logits.device:
            raise ValueError("geometry_bias and co-reference logits must share one device")
        logits = logits + geometry_bias.to(logits)
    if group_ids is not None:
        logits = tie_assignment_logits_by_group(logits, footprint, valid, group_ids)
    evidence = footprint_evidence(
        logits,
        footprint,
        valid,
        robust_clip=robust_clip,
    )
    return CoreferenceOutput(logits=logits, evidence=evidence)


def responsibility_weighted_message(
    responsibilities: torch.Tensor,
    sensor_values: torch.Tensor,
    footprint: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Aggregate a relation message in the host value-head space."""

    if responsibilities.ndim != 3 or sensor_values.ndim != 4:
        raise ValueError("responsibilities and values have invalid ranks")
    if responsibilities.shape[:2] != sensor_values.shape[:2]:
        raise ValueError("responsibility and value token axes must match")
    if min((*responsibilities.shape, *sensor_values.shape[2:])) <= 0:
        raise ValueError("responsibility message axes must be non-empty")
    if footprint.shape != responsibilities.shape[:2] or valid.shape != footprint.shape:
        raise ValueError("footprint and valid must match sensor token axes")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be boolean")
    tensors = (responsibilities, sensor_values, footprint)
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("responsibility message tensors must be floating point")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("responsibility message tensors must be finite")
    if any(value.device != responsibilities.device for value in (*tensors, valid)):
        raise ValueError("responsibility message tensors must share one device")
    if (footprint < 0).any():
        raise ValueError("footprint must be non-negative")
    if (responsibilities < 0).any() or (responsibilities.float().sum(dim=-1) > 1.0 + 1e-5).any():
        raise ValueError("responsibilities must form a sub-probability simplex")
    weighted = (
        responsibilities
        * footprint.to(responsibilities).unsqueeze(-1)
        * valid.to(responsibilities.dtype).unsqueeze(-1)
    )
    normalizer = weighted.sum(dim=1).clamp_min(torch.finfo(weighted.dtype).tiny)
    message = torch.einsum("bnk,bnhd->bkhd", weighted, sensor_values)
    return message / normalizer.unsqueeze(-1).unsqueeze(-1)


def grouped_relation_evidence(
    assignment_logits: torch.Tensor,
    responsibilities: torch.Tensor,
    sensor_values: torch.Tensor,
    footprint: torch.Tensor,
    valid: torch.Tensor,
    modality_ids: torch.Tensor,
    *,
    modality_count: int,
    robust_clip: float = 8.0,
) -> GroupedRelationEvidence:
    """Aggregate one shared-Q/K/V observation opinion per typed modality."""

    if type(modality_count) is not int:
        raise TypeError("modality_count must be a Python int")
    if modality_count <= 0:
        raise ValueError("modality_count must be positive")
    if assignment_logits.ndim != 3 or assignment_logits.shape[-1] < 2:
        raise ValueError("assignment logits must have shape [batch, tokens, beliefs+1]")
    if assignment_logits.shape[:-1] != footprint.shape:
        raise ValueError("assignment logits must match the token footprint axes")
    if responsibilities.shape != (*footprint.shape, assignment_logits.shape[-1] - 1):
        raise ValueError("responsibilities must omit exactly the context assignment")
    if sensor_values.ndim != 4 or sensor_values.shape[:2] != footprint.shape:
        raise ValueError("sensor_values must match the token axes")
    if valid.shape != footprint.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean and match token axes")
    if modality_ids.shape != footprint.shape or modality_ids.dtype != torch.long:
        raise ValueError("modality_ids must be long and match token axes")
    if ((modality_ids < -1) | (modality_ids >= modality_count)).any():
        raise ValueError("modality_ids contain an undeclared modality")
    if ((modality_ids < 0) & valid).any():
        raise ValueError("valid physical tokens require a declared modality")
    tensors = (assignment_logits, responsibilities, sensor_values, footprint)
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("grouped relation tensors must be floating point")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("grouped relation tensors must be finite")
    if any(value.device != assignment_logits.device for value in (*tensors, valid, modality_ids)):
        raise ValueError("grouped relation tensors must share one device")
    if (footprint < 0).any():
        raise ValueError("grouped relation footprint must be non-negative")
    if (responsibilities < 0).any() or (responsibilities.float().sum(dim=-1) > 1.0 + 1e-5).any():
        raise ValueError("responsibilities must form a sub-probability simplex")
    if isinstance(robust_clip, bool) or not isinstance(robust_clip, (int, float)):
        raise TypeError("robust_clip must be real-valued")
    if not math.isfinite(robust_clip) or robust_clip <= 0:
        raise ValueError("robust_clip must be positive")

    one_hot = torch.nn.functional.one_hot(modality_ids.clamp_min(0), modality_count).to(footprint)
    one_hot = one_hot * (modality_ids >= 0).unsqueeze(-1)
    group_footprint = (footprint.to(responsibilities) * valid.to(responsibilities.dtype)).unsqueeze(
        -1
    ) * one_hot.to(responsibilities)
    mass = group_footprint.sum(dim=1)
    normalized = group_footprint / mass.unsqueeze(1).clamp_min(
        torch.finfo(group_footprint.dtype).tiny
    )
    support = torch.einsum("bnm,bnk->bmk", normalized, responsibilities)

    weighted = torch.einsum("bnm,bnk->bnmk", group_footprint, responsibilities)
    normalizer = weighted.sum(dim=1).clamp_min(torch.finfo(weighted.dtype).tiny)
    message = torch.einsum("bnmk,bnhd->bmkhd", weighted, sensor_values)
    message = message / normalizer.unsqueeze(-1).unsqueeze(-1)

    object_probability = torch.softmax(assignment_logits.float(), dim=-1)[..., :-1]
    tiny = torch.finfo(object_probability.dtype).eps
    local_log_odds = torch.logit(object_probability.clamp(tiny, 1.0 - tiny)).clamp(
        -robust_clip, robust_clip
    )
    log_weight = torch.where(
        normalized > 0,
        normalized.log(),
        torch.full_like(normalized, -torch.inf),
    )
    robust = torch.logsumexp(
        log_weight.unsqueeze(-1) + local_log_odds.unsqueeze(-2),
        dim=1,
    )
    robust = torch.where(mass.unsqueeze(-1) > 0, robust, torch.zeros_like(robust))
    return GroupedRelationEvidence(
        message=message,
        support=support,
        robust_log_likelihood_ratio=robust.to(responsibilities),
        available=mass > 0,
        valid_footprint_mass=mass,
    )
