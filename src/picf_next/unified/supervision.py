"""Permutation-invariant, loss-side-only supervision for unified beliefs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment

from picf_next.unified.objective import ObjectiveTerm
from picf_next.unified.state import EMPTY, UnifiedBeliefState


@dataclass(frozen=True, slots=True)
class BeliefSetTarget:
    """Optional set labels that never enter the deploy-visible token graph.

    ``token_owner`` is a soft distribution over target objects plus context in
    its final column. It therefore represents overlap and ambiguous pixels
    without forcing a hard owner.
    """

    sample_valid: torch.Tensor
    exhaustive: torch.Tensor
    object_valid: torch.Tensor
    geometry: torch.Tensor
    geometry_valid: torch.Tensor
    token_owner: torch.Tensor
    token_valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.sample_valid.ndim != 1 or self.sample_valid.dtype != torch.bool:
            raise ValueError("sample_valid must be boolean with shape [batch]")
        if self.exhaustive.shape != self.sample_valid.shape or self.exhaustive.dtype != torch.bool:
            raise ValueError("exhaustive must be boolean with shape [batch]")
        if (self.exhaustive & ~self.sample_valid).any():
            raise ValueError("an invalid sample cannot declare exhaustive labels")
        if self.object_valid.ndim != 2 or self.object_valid.dtype != torch.bool:
            raise ValueError("object_valid must be boolean with shape [batch, targets]")
        batch, targets = self.object_valid.shape
        if self.sample_valid.shape != (batch,):
            raise ValueError("sample and object validity batches must match")
        if self.geometry.ndim != 3 or self.geometry.shape[:2] != (batch, targets):
            raise ValueError("geometry must have shape [batch, targets, geometry]")
        if self.geometry_valid.shape != self.geometry.shape or (
            self.geometry_valid.dtype != torch.bool
        ):
            raise ValueError("geometry_valid must be boolean and match geometry")
        if (
            self.token_owner.ndim != 3
            or self.token_owner.shape[0] != batch
            or (self.token_owner.shape[-1] != targets + 1)
        ):
            raise ValueError("token_owner must address every target plus context")
        if self.token_valid.shape != self.token_owner.shape[:2] or (
            self.token_valid.dtype != torch.bool
        ):
            raise ValueError("token_valid must be boolean and match token_owner")
        floating = (self.geometry, self.token_owner)
        if any(not value.is_floating_point() for value in floating):
            raise TypeError("set target values must be floating point")
        if any(not torch.isfinite(value).all() for value in floating):
            raise ValueError("set target values must be finite")
        if any(value.requires_grad or value.grad_fn is not None for value in floating):
            raise ValueError("set targets must be stop-gradient tensors")
        device = self.sample_valid.device
        tensors = (
            self.exhaustive,
            self.object_valid,
            self.geometry,
            self.geometry_valid,
            self.token_owner,
            self.token_valid,
        )
        if any(value.device != device for value in tensors):
            raise ValueError("set target tensors must share one device")
        if (self.token_owner < 0).any():
            raise ValueError("token ownership probabilities must be non-negative")
        owner_sum = self.token_owner.sum(dim=-1)
        active_tokens = self.token_valid & self.sample_valid[:, None]
        if not torch.allclose(
            owner_sum.masked_select(active_tokens),
            torch.ones_like(owner_sum.masked_select(active_tokens)),
            atol=1e-5,
        ):
            raise ValueError("valid token ownership must sum to one")
        if (self.token_owner.masked_select(~active_tokens.unsqueeze(-1)) != 0).any():
            raise ValueError("invalid samples/tokens must carry zero ownership")
        invalid_object = ~self.object_valid
        if (
            self.token_owner[..., :-1]
            .masked_select(invalid_object[:, None, :].expand(-1, self.token_owner.shape[1], -1))
            .ne(0)
            .any()
        ):
            raise ValueError("invalid target objects cannot own tokens")
        if (self.geometry_valid & ~self.object_valid.unsqueeze(-1)).any():
            raise ValueError("invalid target objects cannot carry valid geometry")
        if (self.object_valid & ~self.sample_valid.unsqueeze(-1)).any():
            raise ValueError("invalid samples cannot carry valid target objects")


@dataclass(frozen=True, slots=True)
class BeliefSetLossConfig:
    lifecycle_weight: float = 1.0
    geometry_weight: float = 1.0
    assignment_weight: float = 1.0
    matching_lifecycle_weight: float = 1.0
    matching_geometry_weight: float = 1.0
    matching_assignment_weight: float = 1.0
    information_floor: float = 1e-5

    def __post_init__(self) -> None:
        values = (
            self.lifecycle_weight,
            self.geometry_weight,
            self.assignment_weight,
            self.matching_lifecycle_weight,
            self.matching_geometry_weight,
            self.matching_assignment_weight,
        )
        controls = (*values, self.information_floor)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in controls
        ):
            raise TypeError("set loss weights and information_floor must be real-valued")
        if any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError("set loss weights must be finite and non-negative")
        if self.information_floor <= 0 or not math.isfinite(self.information_floor):
            raise ValueError("information_floor must be finite and positive")


def _detached_matching_cost(
    state: UnifiedBeliefState,
    assignment_log_probs: torch.Tensor,
    target: BeliefSetTarget,
    *,
    batch_index: int,
    target_indices: torch.Tensor,
    config: BeliefSetLossConfig,
) -> torch.Tensor:
    capacity = state.capacity
    nonempty = state.nonempty_probability[batch_index].clamp_min(torch.finfo(torch.float32).tiny)
    columns = []
    for target_index in target_indices.tolist():
        cost = -nonempty.log() * config.matching_lifecycle_weight
        valid_geometry = target.geometry_valid[batch_index, target_index]
        if valid_geometry.any():
            difference = (
                state.geometry_mean[batch_index, :, valid_geometry]
                - target.geometry[batch_index, target_index, valid_geometry]
            )
            cost = cost + difference.float().square().mean(dim=-1) * (
                config.matching_geometry_weight
            )
        owner = target.token_owner[batch_index, :, target_index] * target.token_valid[
            batch_index
        ].to(target.token_owner)
        owner_mass = owner.sum()
        if owner_mass > 0:
            token_cost = (
                -torch.einsum(
                    "n,nk->k",
                    owner,
                    assignment_log_probs[batch_index, :, :capacity],
                )
                / owner_mass
            )
            cost = cost + token_cost * config.matching_assignment_weight
        columns.append(cost)
    return torch.stack(columns, dim=-1).detach().cpu()


def belief_set_supervision_terms(
    state: UnifiedBeliefState,
    assignment_logits: torch.Tensor,
    target: BeliefSetTarget,
    *,
    config: BeliefSetLossConfig | None = None,
) -> tuple[ObjectiveTerm, ObjectiveTerm, ObjectiveTerm]:
    """Build proper set/lifecycle/geometry/assignment terms after matching."""

    if config is None:
        config = BeliefSetLossConfig()
    batch, tokens, assignments = assignment_logits.shape
    if batch != state.batch_size or assignments != state.capacity + 1:
        raise ValueError("assignment logits must address every belief plus context")
    if target.sample_valid.shape[0] != batch or target.token_owner.shape[:2] != (batch, tokens):
        raise ValueError("set target and prediction batch/token axes differ")
    if target.geometry.shape[-1] != state.geometry_dim:
        raise ValueError("set target and belief geometry widths differ")
    if target.object_valid.shape[1] > state.capacity:
        raise ValueError("set target capacity exceeds the persistent belief capacity")
    if assignment_logits.device != state.content.device or target.sample_valid.device != (
        state.content.device
    ):
        raise ValueError("set predictions and targets must share one device")
    if not assignment_logits.is_floating_point() or not torch.isfinite(assignment_logits).all():
        raise ValueError("assignment logits must be finite floating point")

    assignment_log_probs = torch.log_softmax(assignment_logits.float(), dim=-1)
    matched_target_for_row = torch.full(
        (batch, state.capacity),
        -1,
        dtype=torch.long,
        device=state.content.device,
    )
    matched_row_for_target = torch.full(
        target.object_valid.shape,
        -1,
        dtype=torch.long,
        device=state.content.device,
    )
    for batch_index in range(batch):
        if not target.sample_valid[batch_index]:
            continue
        target_indices = torch.nonzero(target.object_valid[batch_index], as_tuple=False).flatten()
        if target_indices.numel() == 0:
            continue
        cost = _detached_matching_cost(
            state,
            assignment_log_probs,
            target,
            batch_index=batch_index,
            target_indices=target_indices,
            config=config,
        )
        rows, columns = linear_sum_assignment(cost.numpy())
        rows_tensor = torch.as_tensor(rows, dtype=torch.long, device=state.content.device)
        target_tensor = target_indices.index_select(
            0,
            torch.as_tensor(columns, dtype=torch.long, device=state.content.device),
        )
        matched_target_for_row[batch_index, rows_tensor] = target_tensor
        matched_row_for_target[batch_index, target_tensor] = rows_tensor

    lifecycle_probability = state.lifecycle_probs[..., EMPTY]
    matched = matched_target_for_row >= 0
    lifecycle_probability = torch.where(
        matched,
        1.0 - lifecycle_probability,
        lifecycle_probability,
    )
    lifecycle_values = -lifecycle_probability.clamp_min(
        torch.finfo(lifecycle_probability.dtype).tiny
    ).log()
    # Partial object annotations provide positive evidence only.  They must not
    # turn every unlabeled physical object into a false EMPTY target.  Empty-row
    # supervision is legal only when the dataset explicitly certifies that the
    # target set is exhaustive for the frame.
    lifecycle_valid = matched | (
        target.exhaustive[:, None] & target.sample_valid[:, None]
    ).expand_as(matched)

    geometry_values = state.geometry_mean.new_zeros(target.object_valid.shape)
    geometry_loss_valid = torch.zeros_like(target.object_valid)
    remapped_owner = assignment_logits.new_zeros((batch, tokens, state.capacity + 1))
    remapped_owner[..., -1] = target.token_owner[..., -1]
    for batch_index in range(batch):
        for target_index in (
            torch.nonzero(target.object_valid[batch_index], as_tuple=False).flatten().tolist()
        ):
            row = int(matched_row_for_target[batch_index, target_index].item())
            if row < 0:
                continue
            remapped_owner[batch_index, :, row] = target.token_owner[batch_index, :, target_index]
            valid_geometry = (
                target.geometry_valid[batch_index, target_index]
                & (state.geometry_valid[batch_index, row])
            )
            if not valid_geometry.any():
                continue
            difference = (
                state.geometry_mean[batch_index, row, valid_geometry]
                - target.geometry[batch_index, target_index, valid_geometry]
            )
            information = state.geometry_information[batch_index, row][valid_geometry][
                :, valid_geometry
            ].float()
            identity = torch.eye(
                information.shape[-1],
                device=information.device,
                dtype=information.dtype,
            )
            information = information + config.information_floor * identity
            quadratic = torch.einsum(
                "i,ij,j->",
                difference.float(),
                information,
                difference.float(),
            )
            _, logdet = torch.linalg.slogdet(information)
            geometry_values[batch_index, target_index] = 0.5 * (
                quadratic - logdet + information.shape[-1] * math.log(2 * math.pi)
            )
            geometry_loss_valid[batch_index, target_index] = True

    labelled_mass = remapped_owner[..., :-1].sum(dim=-1)
    conditional_positive_owner = remapped_owner.clone()
    conditional_positive_owner[..., :-1] = remapped_owner[..., :-1] / labelled_mass.clamp_min(
        torch.finfo(remapped_owner.dtype).tiny
    ).unsqueeze(-1)
    conditional_positive_owner[..., -1] = 0
    exhaustive = target.exhaustive[:, None]
    effective_owner = torch.where(
        exhaustive.unsqueeze(-1),
        remapped_owner,
        conditional_positive_owner,
    )
    assignment_values = -(effective_owner * assignment_log_probs).sum(dim=-1)
    # A partial set is positive-only supervision. Unknown regions cannot be
    # declared context merely because that dataset omitted their object labels.
    assignment_values = torch.where(
        exhaustive,
        assignment_values,
        assignment_values * labelled_mass,
    )
    assignment_valid = (
        target.token_valid & target.sample_valid[:, None] & (exhaustive | (labelled_mass > 0))
    )
    return (
        ObjectiveTerm(
            "set/lifecycle",
            lifecycle_values,
            lifecycle_valid,
            weight=config.lifecycle_weight,
        ),
        ObjectiveTerm(
            "set/geometry",
            geometry_values,
            geometry_loss_valid,
            weight=config.geometry_weight,
        ),
        ObjectiveTerm(
            "set/assignment",
            assignment_values,
            assignment_valid,
            weight=config.assignment_weight,
        ),
    )
