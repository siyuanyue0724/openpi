"""GuidedVLA-style target-mass supervision on native posterior keys."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ActionPosteriorTargetMass:
    """Differentiable action-to-target posterior mass and its validity mask."""

    loss: torch.Tensor
    target_mass: torch.Tensor
    total_posterior_mass: torch.Tensor
    valid_entries: torch.Tensor

    def __post_init__(self) -> None:
        if self.loss.ndim != 0:
            raise ValueError("action-posterior target-mass loss must be scalar")
        if self.target_mass.ndim != 2:
            raise ValueError("action-posterior target mass must be [batch,action]")
        if self.total_posterior_mass.shape != self.target_mass.shape:
            raise ValueError("total posterior mass must match target mass")
        if (
            self.valid_entries.shape != self.target_mass.shape
            or self.valid_entries.dtype != torch.bool
        ):
            raise ValueError("action-posterior validity must be boolean [batch,action]")


def conditional_action_posterior_distribution(
    posterior_attention: torch.Tensor,
) -> torch.Tensor:
    """Normalize mean-head action attention over the physical posterior rows.

    The training loss intentionally preserves mass on ordinary LingBot keys.
    Retention instead asks which posterior row each native action query selects
    conditional on consulting the posterior, so it renormalizes only for that
    diagnostic metric and introduces no training or deployment operation.
    """

    if posterior_attention.ndim != 4 or not posterior_attention.is_floating_point():
        raise ValueError("posterior attention must be float [batch,head,action,row]")
    if not torch.isfinite(posterior_attention).all() or (posterior_attention < 0).any():
        raise ValueError("posterior attention contains invalid probability mass")
    row_mass = posterior_attention.float().mean(dim=1)
    denominator = row_mass.sum(dim=-1, keepdim=True)
    if (denominator <= 0).any():
        raise ValueError("action queries expose no posterior-row mass")
    return row_mass / denominator


def aggregate_action_posterior_distribution(
    posterior_attention: torch.Tensor,
) -> torch.Tensor:
    """Return one action-count-invariant posterior-row diagnostic distribution.

    Action tokens are weighted by their actual posterior adoption mass rather
    than treated as an arbitrary number of independent object-read queries.
    The singleton middle axis preserves the generic ``[batch,query,row]``
    metric ABI without claiming equivalence to the retired OBJECT_READ set.
    """

    if posterior_attention.ndim != 4 or not posterior_attention.is_floating_point():
        raise ValueError("posterior attention must be float [batch,head,action,row]")
    if not torch.isfinite(posterior_attention).all() or (posterior_attention < 0).any():
        raise ValueError("posterior attention contains invalid probability mass")
    row_mass = posterior_attention.float().mean(dim=1).sum(dim=1, keepdim=True)
    denominator = row_mass.sum(dim=-1, keepdim=True)
    if (denominator <= 0).any():
        raise ValueError("action queries expose no posterior-row mass")
    return row_mass / denominator


def action_posterior_target_mass_loss(
    posterior_attention: torch.Tensor,
    *,
    target_row_weights: torch.Tensor,
    target_valid: torch.Tensor,
    head_indices: torch.Tensor | None = None,
) -> ActionPosteriorTargetMass:
    """Port GuidedVLA mean-head full-key mass semantics to posterior rows.

    ``posterior_attention`` is not normalized over rows. Its missing probability
    mass remains on ordinary LingBot keys, so this objective measures both
    posterior adoption and target-row selection.
    """

    if posterior_attention.ndim != 4 or not posterior_attention.is_floating_point():
        raise ValueError("posterior attention must be float [batch,head,action,row]")
    batch, heads, actions, rows = posterior_attention.shape
    if (
        target_row_weights.shape != (batch, rows)
        or not target_row_weights.is_floating_point()
        or target_valid.shape != (batch,)
        or target_valid.dtype != torch.bool
    ):
        raise ValueError("target rows must be float [batch,row] with boolean [batch] validity")
    if (
        target_row_weights.device != posterior_attention.device
        or target_valid.device != posterior_attention.device
    ):
        raise ValueError("action-posterior attention and targets must share one device")
    tensors = (posterior_attention, target_row_weights)
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("action-posterior target-mass inputs contain NaN or infinity")
    if (posterior_attention < 0).any() or (target_row_weights < 0).any():
        raise ValueError("action-posterior masses and target weights cannot be negative")
    if (target_row_weights > 1).any():
        raise ValueError("posterior target weights cannot exceed one")

    if head_indices is None:
        selected = posterior_attention
    else:
        if (
            head_indices.ndim != 1
            or head_indices.dtype != torch.long
            or not head_indices.numel()
            or head_indices.device != posterior_attention.device
            or (head_indices < 0).any()
            or (head_indices >= heads).any()
            or torch.unique(head_indices).numel() != head_indices.numel()
        ):
            raise ValueError("registered action head indices are invalid")
        selected = posterior_attention.index_select(1, head_indices)

    averaged = selected.float().mean(dim=1)
    weights = target_row_weights.float()
    target_mass = (averaged * weights[:, None]).sum(dim=-1)
    total_posterior_mass = averaged.sum(dim=-1)
    per_batch_valid = target_valid & (weights.amax(dim=-1) > 0)
    valid_entries = per_batch_valid[:, None].expand(batch, actions)
    loss_per_entry = -torch.log(target_mass.clamp_min(1e-6))
    valid_count = valid_entries.sum()
    if not bool(valid_count):
        loss = posterior_attention.sum() * 0.0
    else:
        loss = (loss_per_entry * valid_entries).sum() / valid_count.to(loss_per_entry.dtype)
    return ActionPosteriorTargetMass(
        loss=loss,
        target_mass=target_mass,
        total_posterior_mass=total_posterior_mass,
        valid_entries=valid_entries,
    )
