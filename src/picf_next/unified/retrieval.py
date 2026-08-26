"""Task-conditioned reads that cannot mutate the physical posterior."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RetrievalOutput:
    readout: torch.Tensor
    belief_weights: torch.Tensor
    context_weights: torch.Tensor


def retrieve_task_context(
    task_queries: torch.Tensor,
    belief_keys: torch.Tensor,
    belief_values: torch.Tensor,
    belief_nonempty: torch.Tensor,
    context_keys: torch.Tensor,
    context_values: torch.Tensor,
    context_valid: torch.Tensor,
) -> RetrievalOutput:
    """Read beliefs and dense context using host-space task queries.

    All arguments are read-only tensors.  Physical lifecycle is an input prior to
    retrieval and no updated belief state is returned.
    """

    if task_queries.ndim != 3:
        raise ValueError("task_queries must have shape [batch, queries, width]")
    if belief_keys.shape != belief_values.shape or context_keys.shape != context_values.shape:
        raise ValueError("retrieval key/value shapes must match")
    if belief_keys.ndim != 3 or context_keys.ndim != 3:
        raise ValueError("retrieval banks must have shape [batch, tokens, width]")
    if (
        belief_keys.shape[0] != task_queries.shape[0]
        or context_keys.shape[0] != task_queries.shape[0]
    ):
        raise ValueError("retrieval batches must match")
    if (
        belief_keys.shape[-1] != task_queries.shape[-1]
        or context_keys.shape[-1] != task_queries.shape[-1]
    ):
        raise ValueError("retrieval widths must match")
    if belief_nonempty.shape != belief_keys.shape[:2]:
        raise ValueError("belief_nonempty must have one mass per belief")
    if context_valid.shape != context_keys.shape[:2] or context_valid.dtype != torch.bool:
        raise ValueError("context_valid must be boolean with one value per context token")
    if ((belief_nonempty < 0) | (belief_nonempty > 1)).any():
        raise ValueError("belief_nonempty must lie in [0, 1]")
    tensors = (
        task_queries,
        belief_keys,
        belief_values,
        belief_nonempty,
        context_keys,
        context_values,
    )
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("retrieval tensors must be finite floating point")
    if belief_keys.shape[1] <= 0 or task_queries.shape[-1] <= 0:
        raise ValueError("retrieval requires at least one belief and one feature")
    device = task_queries.device
    if any(value.device != device for value in (*tensors, context_valid)):
        raise ValueError("retrieval tensors must share one device")

    scale = task_queries.shape[-1] ** -0.5
    belief_logits = torch.einsum("bqd,bkd->bqk", task_queries, belief_keys) * scale
    belief_logits = belief_logits + belief_nonempty.clamp_min(
        torch.finfo(belief_logits.dtype).tiny
    ).log().unsqueeze(1)
    context_logits = torch.einsum("bqd,bnd->bqn", task_queries, context_keys) * scale
    context_logits = context_logits.masked_fill(~context_valid.unsqueeze(1), -torch.inf)
    logits = torch.cat((belief_logits, context_logits), dim=-1)
    weights = torch.softmax(logits.float(), dim=-1).to(logits)
    belief_count = belief_keys.shape[1]
    belief_weights = weights[..., :belief_count]
    context_weights = weights[..., belief_count:]
    readout = torch.einsum("bqk,bkd->bqd", belief_weights, belief_values)
    readout = readout + torch.einsum("bqn,bnd->bqd", context_weights, context_values)
    return RetrievalOutput(
        readout=readout,
        belief_weights=belief_weights,
        context_weights=context_weights,
    )
