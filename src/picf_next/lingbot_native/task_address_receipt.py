"""Faithful attention receipts for task-conditioned physical-row reads.

The receipt reuses the exact post-MRoPE query/key tensors and boolean mask
consumed by LingBot's eager attention kernel.  It adds no learned parameter and
does not participate in the model output; it only aggregates probability mass
from each OBJECT_READ query onto the three legal carriers of one physical row:
layerwise memory, current PRIOR, and current POSTERIOR.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TaskAddressAttentionLayout:
    """Exact executed token layout required by a task-address receipt."""

    batch_size: int
    query_count: int
    capacity: int
    object_read_slice: slice
    prior_slice: slice
    posterior_slice: slice

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("query_count", self.query_count),
            ("capacity", self.capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"task-address {name} must be a positive integer")
        read_count = self.object_read_slice.stop - self.object_read_slice.start
        if read_count <= 0:
            raise ValueError("task-address layout requires at least one OBJECT_READ")
        _validate_slice(
            "object_read_slice",
            self.object_read_slice,
            length=self.query_count,
            count=read_count,
        )
        _validate_slice(
            "prior_slice",
            self.prior_slice,
            length=self.query_count,
            count=self.capacity,
        )
        _validate_slice(
            "posterior_slice",
            self.posterior_slice,
            length=self.query_count,
            count=self.capacity,
        )


@dataclass(frozen=True, slots=True)
class TaskAddressAttentionReceipt:
    """Compact per-read, per-row attention evidence from one host layer."""

    row_mass: torch.Tensor
    carrier_mass: torch.Tensor
    visible_mass: torch.Tensor

    def __post_init__(self) -> None:
        if self.row_mass.ndim != 3:
            raise ValueError("task-address row mass must be [batch, read, row]")
        if self.carrier_mass.shape != (*self.row_mass.shape, 3):
            raise ValueError(
                "task-address carrier mass must be [batch, read, row, carrier]"
            )
        if self.visible_mass.shape != self.row_mass.shape[:2]:
            raise ValueError("task-address visible mass must be [batch, read]")
        tensors = (self.row_mass, self.carrier_mass, self.visible_mass)
        if any(not value.is_floating_point() for value in tensors):
            raise TypeError("task-address receipt tensors must be floating point")
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("task-address receipt contains a non-finite value")


def _validate_slice(name: str, value: slice, *, length: int, count: int) -> None:
    if value.step not in (None, 1) or value.start is None or value.stop is None:
        raise ValueError(f"{name} must be one contiguous concrete slice")
    if value.start < 0 or value.stop > length or value.stop - value.start != count:
        raise ValueError(f"{name} does not contain exactly {count} tokens")


def task_address_attention_receipt(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor,
    object_read_slice: slice,
    prior_slice: slice,
    posterior_slice: slice,
    capacity: int,
) -> TaskAddressAttentionReceipt:
    """Aggregate the real eager-attention distribution onto physical rows.

    ``key_states`` may contain one prepended layerwise-memory bank.  The
    correction graph requires that bank to contain exactly ``capacity`` rows;
    each row's mass is then the sum of memory, PRIOR, and POSTERIOR carriers.
    The softmax is recomputed in float32 with the same GQA expansion and mask as
    LingBot's eager kernel, so this remains a diagnostic of the executed Q/K
    path rather than a surrogate scorer.
    """

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("task-address capacity must be a positive integer")
    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("task-address Q/K tensors must be [batch, token, head, dim]")
    batch, query_count, query_heads, head_dim = query_states.shape
    if (
        key_states.shape[0] != batch
        or key_states.shape[3] != head_dim
        or attention_mask.shape != (batch, query_count, key_states.shape[1])
        or attention_mask.dtype is not torch.bool
        or query_states.device != key_states.device
        or attention_mask.device != query_states.device
    ):
        raise ValueError("task-address Q/K/mask surfaces are incompatible")
    key_heads = key_states.shape[2]
    if key_heads <= 0 or query_heads % key_heads:
        raise ValueError("task-address query heads must evenly group key/value heads")
    read_count = object_read_slice.stop - object_read_slice.start
    if read_count <= 0:
        raise ValueError("task-address receipt requires at least one OBJECT_READ")
    _validate_slice("object_read_slice", object_read_slice, length=query_count, count=read_count)
    _validate_slice("prior_slice", prior_slice, length=query_count, count=capacity)
    _validate_slice("posterior_slice", posterior_slice, length=query_count, count=capacity)

    memory_count = key_states.shape[1] - query_count
    if memory_count != capacity:
        raise ValueError(
            "task-address correction receipt requires one capacity-sized memory bank"
        )
    repeated_keys = key_states.float().repeat_interleave(query_heads // key_heads, dim=2)
    reads = query_states[:, object_read_slice].float()
    logits = torch.einsum("bjhd,bkhd->bhjk", reads, repeated_keys)
    logits.mul_(head_dim**-0.5)
    read_mask = attention_mask[:, object_read_slice]
    logits = torch.where(
        read_mask[:, None],
        logits,
        torch.tensor(-2.3819763e38, device=logits.device, dtype=logits.dtype),
    )
    probabilities = torch.softmax(logits, dim=-1)

    rows = torch.arange(capacity, device=query_states.device)
    carrier_indices = torch.stack(
        (
            rows,
            memory_count + prior_slice.start + rows,
            memory_count + posterior_slice.start + rows,
        ),
        dim=-1,
    )
    expanded_indices = carrier_indices.view(1, 1, 1, capacity, 3).expand(
        batch,
        query_heads,
        read_count,
        -1,
        -1,
    )
    carrier_mass = probabilities.unsqueeze(-2).expand(-1, -1, -1, capacity, -1).gather(
        -1,
        expanded_indices,
    )
    carrier_mass = carrier_mass.mean(dim=1)
    row_mass = carrier_mass.sum(dim=-1)
    visible_mass = row_mass.sum(dim=-1)
    return TaskAddressAttentionReceipt(
        row_mass=row_mass,
        carrier_mass=carrier_mass,
        visible_mass=visible_mass,
    )
