"""Causal loss-side identity gauge for task-independent physical entity rows.

The gauge removes the permutation ambiguity of set supervision inside one
episode. Dataset identities and Hungarian assignments are never model inputs,
recurrent tensors, or deployment state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment

from picf_next.lingbot_native.entity_set_objective import (
    PhysicalFrameAssignment,
    PhysicalFramePredictions,
    PhysicalFrameTargets,
    physical_pairwise_assignment_cost,
)
from picf_next.lingbot_native.row_binding import (
    RowBindings,
    normalize_row_bindings,
    row_binding_map,
)


@dataclass(frozen=True, slots=True)
class PhysicalSequenceAssignment:
    """One causal row gauge over a local sequence.

    Phase ``2*t`` is the prior before observation ``t`` and ``2*t+1`` is the
    posterior after that observation. ``reserved_rows`` holds prior episode
    identities absent from the local target axis, so exhaustive inventory
    supervision cannot turn an occluded carried row into a false no-object row.
    """

    row_to_track: torch.Tensor
    binding_start_phase: torch.Tensor
    reserved_rows: torch.Tensor
    time_count: int

    def __post_init__(self) -> None:
        if self.row_to_track.ndim != 2 or self.row_to_track.dtype != torch.long:
            raise ValueError("physical sequence assignment must be long [batch,rows]")
        if (self.row_to_track < -1).any():
            raise ValueError("unmatched physical sequence rows must use -1")
        expected = self.row_to_track.shape
        if (
            self.binding_start_phase.shape != expected
            or self.binding_start_phase.dtype != torch.long
            or self.binding_start_phase.device != self.row_to_track.device
            or (self.binding_start_phase < 0).any()
        ):
            raise ValueError("physical binding phases must be non-negative long [batch,rows]")
        if (
            self.reserved_rows.shape != expected
            or self.reserved_rows.dtype != torch.bool
            or self.reserved_rows.device != self.row_to_track.device
        ):
            raise ValueError("physical reserved rows must be boolean [batch,rows]")
        if isinstance(self.time_count, bool) or not isinstance(self.time_count, int):
            raise TypeError("physical sequence time count must be an integer")
        if self.time_count <= 0:
            raise ValueError("physical sequence time count must be positive")
        terminal_phase = 2 * self.time_count
        if (self.binding_start_phase > terminal_phase).any():
            raise ValueError("physical binding phase lies outside the local sequence")
        unmatched = self.row_to_track < 0
        if (self.binding_start_phase[unmatched] != terminal_phase).any():
            raise ValueError("unmatched physical rows must remain unbound for the sequence")
        if (self.reserved_rows & ~unmatched).any():
            raise ValueError("a physical row cannot be matched and anonymously reserved")


def _validate_sequence(
    predictions: Sequence[PhysicalFramePredictions],
    targets: Sequence[PhysicalFrameTargets],
    *,
    identity_keys_by_batch: tuple[tuple[str, ...], ...],
    prior_bindings_by_batch: tuple[RowBindings, ...],
) -> tuple[int, int, int]:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("physical predictions and targets require one equal non-empty time axis")
    if any(not isinstance(value, PhysicalFramePredictions) for value in predictions):
        raise TypeError("physical sequence predictions require physical frame values")
    if any(not isinstance(value, PhysicalFrameTargets) for value in targets):
        raise TypeError("physical sequence targets require physical frame values")
    batch, _tokens, rows = predictions[0].support_logits.shape
    tracks = targets[0].masks.shape[1]
    if len(identity_keys_by_batch) != batch or len(prior_bindings_by_batch) != batch:
        raise ValueError("physical sequence identity metadata differs from its batch")
    reference_track_valid = targets[0].track_valid
    reference_censored = targets[0].capacity_censored
    for time_index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        current_batch, tokens, current_rows = prediction.support_logits.shape
        if (current_batch, current_rows) != (batch, rows):
            raise ValueError("physical prediction row axes differ across time")
        if target.masks.shape != (batch, tracks, tokens):
            raise ValueError("physical prediction and target axes differ at one time")
        if not torch.equal(target.track_valid, reference_track_valid) or not torch.equal(
            target.capacity_censored,
            reference_censored,
        ):
            raise ValueError("physical target identity axes must remain fixed across time")
        if prediction.support_logits.device != predictions[0].support_logits.device:
            raise ValueError("physical sequence predictions must share one device")
        if target.masks.device != predictions[0].support_logits.device:
            raise ValueError("physical sequence targets and predictions must share one device")
        del time_index
    for batch_index, identity_keys in enumerate(identity_keys_by_batch):
        if (
            not identity_keys
            or len(identity_keys) > tracks
            or len(set(identity_keys)) != len(identity_keys)
            or any(not isinstance(key, str) or not key for key in identity_keys)
        ):
            raise ValueError("physical sequence identities must be non-empty and unique")
        expected_valid = torch.arange(
            tracks,
            device=reference_track_valid.device,
        ) < len(identity_keys)
        if not torch.equal(reference_track_valid[batch_index], expected_valid):
            raise ValueError("physical sequence identities differ from track validity")
        row_binding_map(prior_bindings_by_batch[batch_index], capacity=rows)
    return batch, rows, tracks


def _identity_evidence_by_time(
    targets: Sequence[PhysicalFrameTargets],
    *,
    batch_index: int,
) -> torch.Tensor:
    evidence: list[torch.Tensor] = []
    for target in targets:
        observed = target.token_observed_fraction[batch_index] > 0
        visible = (
            target.mask_valid[batch_index] & (target.masks[batch_index] > 0) & observed.unsqueeze(0)
        ).any(dim=-1)
        existing = target.existence_valid[batch_index] & (target.existence[batch_index] > 0)
        evidence.append(visible | existing)
    return torch.stack(evidence, dim=0)


@torch.no_grad()
def match_physical_sequence_entities(
    predictions: Sequence[PhysicalFramePredictions],
    targets: Sequence[PhysicalFrameTargets],
    *,
    identity_keys_by_batch: tuple[tuple[str, ...], ...],
    prior_bindings_by_batch: tuple[RowBindings, ...],
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> PhysicalSequenceAssignment:
    """Assign births at first evidence while preserving the episode row gauge."""

    batch, rows, _tracks = _validate_sequence(
        predictions,
        targets,
        identity_keys_by_batch=identity_keys_by_batch,
        prior_bindings_by_batch=prior_bindings_by_batch,
    )
    time_count = len(predictions)
    device = predictions[0].support_logits.device
    terminal_phase = 2 * time_count
    assignment = torch.full((batch, rows), -1, dtype=torch.long, device=device)
    binding_start_phase = torch.full_like(assignment, terminal_phase)
    reserved_rows = torch.zeros_like(assignment, dtype=torch.bool)

    for batch_index, (identity_keys, prior_bindings) in enumerate(
        zip(identity_keys_by_batch, prior_bindings_by_batch, strict=True)
    ):
        identity_to_track = {identity: index for index, identity in enumerate(identity_keys)}
        prior = row_binding_map(prior_bindings, capacity=rows)
        occupied_rows = set(prior.values())
        censored = targets[0].capacity_censored[batch_index]
        for identity, row in prior.items():
            track = identity_to_track.get(identity)
            if track is None or bool(censored[track].item()):
                reserved_rows[batch_index, row] = True
                continue
            assignment[batch_index, row] = track
            binding_start_phase[batch_index, row] = 0

        evidence = _identity_evidence_by_time(targets, batch_index=batch_index)
        eligible = targets[0].track_valid[batch_index] & ~censored & evidence.any(dim=0)
        for identity in prior:
            track = identity_to_track.get(identity)
            if track is not None:
                eligible[track] = False
        first_evidence = torch.arange(time_count, device=device).unsqueeze(1)
        first_evidence = (
            first_evidence.expand(time_count, evidence.shape[1])
            .masked_fill(
                ~evidence,
                time_count,
            )
            .amin(dim=0)
        )
        birth_times = sorted(set(first_evidence[eligible].detach().cpu().tolist()))
        free_rows = sorted(set(range(rows)) - occupied_rows)
        for birth_time in birth_times:
            born = (eligible & (first_evidence == birth_time)).nonzero().flatten()
            if born.numel() > len(free_rows):
                raise ValueError("physical births exceed the remaining row capacity")
            costs = physical_pairwise_assignment_cost(
                predictions[birth_time],
                targets[birth_time],
                batch_index=batch_index,
                track_indices=born,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )
            free_tensor = torch.tensor(free_rows, dtype=torch.long, device=device)
            relative_rows, relative_tracks = linear_sum_assignment(
                costs.index_select(0, free_tensor).cpu().numpy()
            )
            selected_rows = free_tensor[
                torch.as_tensor(relative_rows, dtype=torch.long, device=device)
            ]
            selected_tracks = born[
                torch.as_tensor(relative_tracks, dtype=torch.long, device=device)
            ]
            assignment[batch_index, selected_rows] = selected_tracks
            binding_start_phase[batch_index, selected_rows] = 2 * birth_time + 1
            selected = set(selected_rows.detach().cpu().tolist())
            free_rows = [row for row in free_rows if row not in selected]

    return PhysicalSequenceAssignment(
        row_to_track=assignment,
        binding_start_phase=binding_start_phase,
        reserved_rows=reserved_rows,
        time_count=time_count,
    )


def physical_frame_assignment_at_time(
    assignment: PhysicalSequenceAssignment,
    *,
    time_index: int,
) -> PhysicalFrameAssignment:
    """Expose only bindings available after one causal observation."""

    if not isinstance(assignment, PhysicalSequenceAssignment):
        raise TypeError("physical frame assignment requires a sequence gauge")
    if (
        isinstance(time_index, bool)
        or not isinstance(time_index, int)
        or not 0 <= time_index < assignment.time_count
    ):
        raise IndexError("physical frame time lies outside the sequence gauge")
    observation_phase = 2 * time_index + 1
    valid = (assignment.row_to_track >= 0) & (assignment.binding_start_phase <= observation_phase)
    carried = valid & (assignment.binding_start_phase < observation_phase)
    return PhysicalFrameAssignment(
        row_to_track=assignment.row_to_track.masked_fill(~valid, -1),
        reserved_rows=assignment.reserved_rows,
        carried_rows=carried,
    )


def extend_physical_sequence_row_bindings(
    assignment: PhysicalSequenceAssignment,
    *,
    identity_keys_by_batch: tuple[tuple[str, ...], ...],
    prior_bindings_by_batch: tuple[RowBindings, ...],
    commit_time_index: int = 0,
) -> tuple[RowBindings, ...]:
    """Commit only births represented by the posterior that advances a lane."""

    if not isinstance(assignment, PhysicalSequenceAssignment):
        raise TypeError("physical row-binding extension requires a sequence gauge")
    if (
        isinstance(commit_time_index, bool)
        or not isinstance(commit_time_index, int)
        or not 0 <= commit_time_index < assignment.time_count
    ):
        raise IndexError("physical binding commit time lies outside the sequence")
    batch, rows = assignment.row_to_track.shape
    if len(identity_keys_by_batch) != batch or len(prior_bindings_by_batch) != batch:
        raise ValueError("physical binding extension metadata differs from its batch")
    commit_phase = 2 * commit_time_index + 1
    resolved: list[RowBindings] = []
    for batch_index, (identity_keys, prior_bindings) in enumerate(
        zip(identity_keys_by_batch, prior_bindings_by_batch, strict=True)
    ):
        current = row_binding_map(prior_bindings, capacity=rows)
        occupied = {row: identity for identity, row in current.items()}
        for row, track in enumerate(assignment.row_to_track[batch_index].tolist()):
            if (
                track < 0
                or int(assignment.binding_start_phase[batch_index, row].item()) > commit_phase
            ):
                continue
            if track >= len(identity_keys):
                raise ValueError("physical assignment references an absent identity")
            identity = identity_keys[track]
            old_row = current.get(identity)
            if old_row is not None and old_row != row:
                raise ValueError("an established physical identity changed rows")
            old_identity = occupied.get(row)
            if old_identity is not None and old_identity != identity:
                raise ValueError("a physical birth replaced an established row identity")
            current[identity] = row
            occupied[row] = identity
        resolved.append(normalize_row_bindings(current, capacity=rows))
    return tuple(resolved)
