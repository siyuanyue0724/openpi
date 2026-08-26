"""Loss-side objectives for the existing LTOP task-address attention path.

The functions in this module add no deploy-time parameter or inference branch.
They score the real OBJECT_READ attention mass already emitted by LingBot's
shared host.  Multiple read queries are treated as a set: a task target is
covered when at least one read assigns probability to its physical row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TASK_ADDRESS_SUPERVISION_DEPTH_SCHEMA = (
    "picf-next.action-consumable-task-address-depth.v1"
)


@dataclass(frozen=True, slots=True)
class TaskAddressCoverage:
    """Differentiable target coverage for one batch of OBJECT_READ queries."""

    loss: torch.Tensor
    conditional_distribution: torch.Tensor
    target_probability_per_read: torch.Tensor
    target_coverage: torch.Tensor

    def __post_init__(self) -> None:
        if self.loss.ndim != 0:
            raise ValueError("task-address loss must be scalar")
        if self.conditional_distribution.ndim != 3:
            raise ValueError("task-address distribution must be [batch,read,row]")
        batch, reads, _rows = self.conditional_distribution.shape
        if self.target_probability_per_read.shape != (batch, reads):
            raise ValueError("task-address target probability must be [batch,read]")
        if self.target_coverage.shape != (batch,):
            raise ValueError("task-address coverage must be [batch]")


@dataclass(frozen=True, slots=True)
class ActionConsumableTaskAddress:
    """Latest address receipt whose output reaches a later ACTION layer."""

    row_mass: torch.Tensor
    producer_layer_index: int
    consumer_layer_index: int
    layer_count: int

    def __post_init__(self) -> None:
        if self.row_mass.ndim != 3 or not self.row_mass.is_floating_point():
            raise ValueError("action-consumable address mass must be float [batch,read,row]")
        if self.layer_count < 2:
            raise ValueError("action-consumable addressing requires at least two host layers")
        if self.producer_layer_index != self.layer_count - 2:
            raise ValueError("action-consumable address must use the penultimate host layer")
        if self.consumer_layer_index != self.producer_layer_index + 1:
            raise ValueError("action-consumable address must name its immediate consumer layer")


def action_consumable_task_address(
    row_mass: torch.Tensor,
    *,
    layer_count: int,
) -> ActionConsumableTaskAddress:
    """Bind the penultimate address output to its final ACTION consumer.

    Transformer tokens form Q/K/V from one layer input before any token's new
    attention output exists. Therefore OBJECT_READ addressing produced at layer
    ``l`` can first affect ACTION attention at layer ``l + 1``. The final-layer
    receipt has no such consumer and is intentionally excluded.
    """

    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 2:
        raise ValueError("action-consumable addressing requires at least two host layers")
    return ActionConsumableTaskAddress(
        row_mass=row_mass,
        producer_layer_index=layer_count - 2,
        consumer_layer_index=layer_count - 1,
        layer_count=layer_count,
    )


def action_consumable_task_address_depth_contract(layer_count: int) -> dict[str, object]:
    """Serialize the action-consumable supervision depth without a tensor."""

    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 2:
        raise ValueError("action-consumable addressing requires at least two host layers")
    return {
        "schema": TASK_ADDRESS_SUPERVISION_DEPTH_SCHEMA,
        "producer_layer_index": layer_count - 2,
        "consumer_layer_index": layer_count - 1,
        "layer_count": layer_count,
        "final_layer_excluded": True,
        "reason": "address-output-must-precede-a-later-action-attention-layer",
    }


def conditional_task_address_distribution(row_mass: torch.Tensor) -> torch.Tensor:
    """Normalize real physical-carrier attention mass over object rows."""

    if row_mass.ndim != 3:
        raise ValueError("task-address row mass must be [batch,read,row]")
    if not row_mass.is_floating_point():
        raise TypeError("task-address row mass must be floating point")
    if not torch.isfinite(row_mass).all():
        raise ValueError("task-address row mass contains NaN or infinity")
    if (row_mass < 0).any():
        raise ValueError("task-address row mass cannot be negative")
    mass = row_mass.float()
    denominator = mass.sum(dim=-1, keepdim=True)
    if (denominator <= 0).any():
        raise ValueError("task-address reads expose no physical carrier mass")
    return mass / denominator


def task_address_target_coverage(
    row_mass: torch.Tensor,
    target_rows: torch.Tensor,
) -> TaskAddressCoverage:
    """Require at least one OBJECT_READ to cover each target physical row.

    The noisy-OR aggregation preserves the intended multi-query semantics: it
    does not force every read to collapse onto the primary task object, while a
    uniformly addressed or entirely wrong read set remains penalized.
    """

    distribution = conditional_task_address_distribution(row_mass)
    batch, reads, rows = distribution.shape
    if target_rows.shape != (batch,) or target_rows.dtype != torch.long:
        raise ValueError("task-address target rows must be long [batch]")
    if target_rows.device != distribution.device:
        raise ValueError("task-address target rows must share the attention device")
    if (target_rows < 0).any() or (target_rows >= rows).any():
        raise ValueError("task-address target row is outside the physical capacity")
    indices = target_rows.view(batch, 1, 1).expand(-1, reads, 1)
    per_read = distribution.gather(-1, indices).squeeze(-1)
    coverage = 1.0 - torch.prod(1.0 - per_read, dim=-1)
    tiny = torch.finfo(distribution.dtype).tiny
    loss = -torch.log(coverage.clamp_min(tiny)).mean()
    return TaskAddressCoverage(
        loss=loss,
        conditional_distribution=distribution,
        target_probability_per_read=per_read,
        target_coverage=coverage,
    )


def task_address_row_coverage(
    conditional_distribution: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Return noisy-OR coverage for arbitrary rows without recomputing softmax."""

    if conditional_distribution.ndim != 3:
        raise ValueError("task-address distribution must be [batch,read,row]")
    batch, reads, capacity = conditional_distribution.shape
    if rows.shape != (batch,) or rows.dtype != torch.long:
        raise ValueError("task-address rows must be long [batch]")
    if rows.device != conditional_distribution.device:
        raise ValueError("task-address rows must share the distribution device")
    if (rows < 0).any() or (rows >= capacity).any():
        raise ValueError("task-address row is outside the physical capacity")
    indices = rows.view(batch, 1, 1).expand(-1, reads, 1)
    per_read = conditional_distribution.gather(-1, indices).squeeze(-1)
    return 1.0 - torch.prod(1.0 - per_read, dim=-1)
