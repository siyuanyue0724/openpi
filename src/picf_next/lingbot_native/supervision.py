"""Loss-only sequence matching and supervision for unordered object rows.

The no-gradient rectangular assignment and BCE-plus-Dice mask costs follow the
set-prediction semantics used by Mask2Former.  PICF adapts those semantics to
sequence tracks, partial inventories, overlapping support, and finite-capacity
censoring.  Targets and assignments never enter the LingBot forward graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from picf_next.lingbot_native.row_binding import (
    RowBindings,
    normalize_row_bindings,
    row_binding_map,
)
from picf_next.objective import ObjectiveTerm

TOKEN_MICRO_OWNERSHIP = "token_micro_categorical"
TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP = "token_micro_entity_conditional_equal"
OWNERSHIP_ESTIMATORS = (
    TOKEN_MICRO_OWNERSHIP,
    TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
)
ENTITY_CONDITIONAL_OWNERSHIP_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class NativeSequencePredictions:
    """Read-only final-host predictions over a local sequence branch."""

    support_logits: torch.Tensor
    ownership: torch.Tensor
    existence_logits: torch.Tensor
    task_relevance_logits: torch.Tensor
    dense_task_grounding_logits: torch.Tensor
    task_object_log_probability: torch.Tensor | None = None
    task_object_probability: torch.Tensor | None = None
    task_event_distribution: torch.Tensor | None = None
    task_row_probability: torch.Tensor | None = None
    task_row_probability_by_time: torch.Tensor | None = None
    ownership_log_probability: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.support_logits.ndim != 4:
            raise ValueError("support logits must have shape [batch,time,tokens,rows]")
        batch, time, tokens, rows = self.support_logits.shape
        if self.ownership.shape != (batch, time, tokens, rows + 1):
            raise ValueError("ownership must append exactly one no-object class")
        if self.existence_logits.shape != (batch, time, rows):
            raise ValueError("existence logits must have shape [batch,time,rows]")
        if self.task_relevance_logits.shape != (batch, rows):
            raise ValueError("task relevance logits must have shape [batch,rows]")
        if self.dense_task_grounding_logits.shape != (batch, time, tokens):
            raise ValueError("dense task grounding logits must have shape [batch,time,tokens]")
        base_tensors = (
            self.support_logits,
            self.ownership,
            self.existence_logits,
            self.task_relevance_logits,
            self.dense_task_grounding_logits,
        )
        ownership_log_probability = self.ownership_log_probability
        if ownership_log_probability is not None and ownership_log_probability.shape != (
            batch,
            time,
            tokens,
            rows + 1,
        ):
            raise ValueError("ownership log probabilities have inconsistent axes")
        factorized_values = (
            self.task_object_log_probability,
            self.task_object_probability,
            self.task_event_distribution,
            self.task_row_probability,
            self.task_row_probability_by_time,
        )
        if any(value is not None for value in factorized_values):
            if any(value is None for value in factorized_values):
                raise ValueError(
                    "factorized sequence predictions must be provided as one complete set"
                )
            task_object_log_probability = cast(
                torch.Tensor,
                self.task_object_log_probability,
            )
            task_object_probability = cast(torch.Tensor, self.task_object_probability)
            task_event_distribution = cast(torch.Tensor, self.task_event_distribution)
            task_row_probability = cast(torch.Tensor, self.task_row_probability)
            task_row_probability_by_time = cast(
                torch.Tensor,
                self.task_row_probability_by_time,
            )
            expected_factorized = (batch, time, tokens, rows)
            if (
                task_object_log_probability.shape != expected_factorized
                or task_object_probability.shape != expected_factorized
                or task_event_distribution.shape != (batch, time, tokens, rows + 1)
                or task_row_probability.shape != (batch, rows)
                or task_row_probability_by_time.shape != (batch, time, rows)
            ):
                raise ValueError("factorized sequence predictions have inconsistent axes")
            factorized_tensors = (
                task_object_log_probability,
                task_object_probability,
                task_event_distribution,
                task_row_probability,
                task_row_probability_by_time,
            )
        else:
            factorized_tensors = ()
        tensors = (
            *base_tensors,
            *((ownership_log_probability,) if ownership_log_probability is not None else ()),
            *factorized_tensors,
        )
        if any(
            not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors
        ):
            raise ValueError("sequence predictions must be finite floating point")
        if any(value.device != self.support_logits.device for value in tensors):
            raise ValueError("sequence predictions must share one device")
        if ((self.ownership < 0) | (self.ownership > 1)).any():
            raise ValueError("ownership must contain probabilities in [0,1]")
        if ownership_log_probability is not None:
            active = self.ownership.float().sum(dim=-1) > 0
            tolerance = max(1e-5, 2 * torch.finfo(self.ownership.dtype).eps)
            if (
                (ownership_log_probability > tolerance).any()
                or not torch.allclose(
                    torch.logsumexp(ownership_log_probability.float(), dim=-1)[active],
                    torch.zeros_like(self.ownership[..., 0].float()[active]),
                    atol=tolerance,
                    rtol=0.0,
                )
                or not torch.allclose(
                    ownership_log_probability.float().exp()[active],
                    self.ownership.float()[active],
                    atol=tolerance,
                    rtol=0.0,
                )
            ):
                raise ValueError("ownership log probabilities differ from the categorical output")
        for value in factorized_tensors[1:]:
            if ((value < 0) | (value > 1)).any():
                raise ValueError("factorized relation probabilities must lie in [0,1]")
        if factorized_tensors:
            task_object_log_probability = cast(
                torch.Tensor,
                self.task_object_log_probability,
            )
            task_object_probability = cast(torch.Tensor, self.task_object_probability)
            task_event_distribution = cast(torch.Tensor, self.task_event_distribution)
            task_row_probability = cast(torch.Tensor, self.task_row_probability)
            task_row_probability_by_time = cast(
                torch.Tensor,
                self.task_row_probability_by_time,
            )
            tolerance = max(
                1e-5,
                2 * torch.finfo(task_object_probability.dtype).eps,
            )
            expected_row_probability = self.task_relevance_logits.float().sigmoid()
            if not torch.allclose(
                task_row_probability.float(),
                expected_row_probability,
                atol=tolerance,
                rtol=0.0,
            ):
                raise ValueError("factorized task-row probability differs from its logits")
            if not torch.allclose(
                task_row_probability.float(),
                task_row_probability_by_time[:, -1].float(),
                atol=tolerance,
                rtol=0.0,
            ):
                raise ValueError("factorized final task-row probability differs from its sequence")
            active = self.ownership.float().sum(dim=-1) > 0
            expected_object_probability = (
                task_row_probability_by_time[:, :, None, :].float()
                * self.ownership[..., :-1].float()
            ).masked_fill(
                ~active.unsqueeze(-1),
                0,
            )
            if not torch.allclose(
                task_object_probability.float(),
                expected_object_probability,
                atol=tolerance,
                rtol=0.0,
            ):
                raise ValueError(
                    "factorized task-object probability breaks its all-time product identity"
                )
            expected_event = torch.cat(
                (
                    task_object_probability.float(),
                    (1 - task_object_probability.float().sum(dim=-1, keepdim=True)).clamp(
                        min=0,
                        max=1,
                    ),
                ),
                dim=-1,
            ).masked_fill(~active.unsqueeze(-1), 0)
            if not torch.allclose(
                task_event_distribution.float(),
                expected_event,
                atol=tolerance,
                rtol=0.0,
            ):
                raise ValueError("factorized task-event distribution is not the closed event")
            if not torch.allclose(
                task_object_log_probability.float().exp(),
                task_object_probability.float(),
                atol=tolerance,
                rtol=0.0,
            ):
                raise ValueError("factorized task-object log probability is inconsistent")


@dataclass(frozen=True, slots=True)
class NativeSequenceTargets:
    """Loss metadata with independent pixel evidence and inventory completeness.

    ``token_observed_fraction`` is the spatial mass with a known, exhaustive
    one-of-K physical owner inside each visual token. It is independent from
    track-local ``mask_valid`` on non-exclusive datasets. ``inventory_exhaustive``
    separately means all eligible persistent entities at a time are known.
    Conflating these contracts creates false background or non-existence targets.
    """

    masks: torch.Tensor
    mask_valid: torch.Tensor
    existence: torch.Tensor
    existence_valid: torch.Tensor
    task_relevance: torch.Tensor
    task_valid: torch.Tensor
    track_valid: torch.Tensor
    capacity_censored: torch.Tensor
    token_observed_fraction: torch.Tensor
    inventory_exhaustive: torch.Tensor
    token_measure_weight: torch.Tensor | None = None
    exclusive_ownership: bool = False

    def __post_init__(self) -> None:
        if self.masks.ndim != 4:
            raise ValueError("target masks must have shape [batch,time,tracks,tokens]")
        batch, time, tracks, tokens = self.masks.shape
        if not self.masks.is_floating_point() or not torch.isfinite(self.masks).all():
            raise ValueError("target masks must be finite floating point")
        if ((self.masks < 0) | (self.masks > 1)).any():
            raise ValueError("target masks must lie in [0,1]")
        expected = {
            "mask_valid": (self.mask_valid, (batch, time, tracks, tokens)),
            "existence_valid": (self.existence_valid, (batch, time, tracks)),
            "task_valid": (self.task_valid, (batch, tracks)),
            "track_valid": (self.track_valid, (batch, tracks)),
            "capacity_censored": (self.capacity_censored, (batch, tracks)),
            "inventory_exhaustive": (self.inventory_exhaustive, (batch, time)),
        }
        for name, (value, shape) in expected.items():
            if value.shape != shape or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape {shape}")
        if (
            self.token_observed_fraction.shape != (batch, time, tokens)
            or not self.token_observed_fraction.is_floating_point()
            or not torch.isfinite(self.token_observed_fraction).all()
            or ((self.token_observed_fraction < 0) | (self.token_observed_fraction > 1)).any()
            or self.token_observed_fraction.requires_grad
        ):
            raise ValueError("token_observed_fraction must be detached floating point in [0,1]")
        if self.token_measure_weight is not None and (
            self.token_measure_weight.shape != (batch, time, tokens)
            or not self.token_measure_weight.is_floating_point()
            or not torch.isfinite(self.token_measure_weight).all()
            or (self.token_measure_weight < 0).any()
            or self.token_measure_weight.requires_grad
        ):
            raise ValueError("token_measure_weight must be detached finite and non-negative")
        if self.existence.shape != (batch, time, tracks):
            raise ValueError("existence targets must have shape [batch,time,tracks]")
        if self.task_relevance.shape != (batch, tracks):
            raise ValueError("task relevance targets must have shape [batch,tracks]")
        for value in (self.existence, self.task_relevance):
            if (
                not value.is_floating_point()
                or not torch.isfinite(value).all()
                or ((value < 0) | (value > 1)).any()
            ):
                raise ValueError("existence and task targets must be finite in [0,1]")
        if any(value.requires_grad for value in (self.masks, self.existence, self.task_relevance)):
            raise ValueError("loss-side targets must not require gradients")
        tensors = (
            self.mask_valid,
            self.existence,
            self.existence_valid,
            self.task_relevance,
            self.task_valid,
            self.track_valid,
            self.capacity_censored,
            self.token_observed_fraction,
            self.inventory_exhaustive,
            *((self.token_measure_weight,) if self.token_measure_weight is not None else ()),
        )
        if any(value.device != self.masks.device for value in tensors):
            raise ValueError("sequence targets must share one device")
        if (self.capacity_censored & ~self.track_valid).any():
            raise ValueError("only valid tracks may be capacity-censored")
        invalid = ~self.track_valid
        if (self.mask_valid & invalid[:, None, :, None]).any():
            raise ValueError("invalid tracks cannot carry mask supervision")
        if (self.existence_valid & invalid[:, None, :]).any():
            raise ValueError("invalid tracks cannot carry existence supervision")
        if (self.task_valid & invalid).any():
            raise ValueError("invalid tracks cannot carry task supervision")
        if self.exclusive_ownership:
            expected_mask_valid = (self.token_observed_fraction > 0).unsqueeze(
                2
            ) & self.track_valid[:, None, :, None]
            if not torch.equal(self.mask_valid, expected_mask_valid):
                raise ValueError(
                    "exclusive mask validity must match observed token mass for every valid track"
                )
            visible_mass = (self.masks * self.mask_valid).sum(dim=2)
            if (visible_mass > 1 + 1e-5).any():
                raise ValueError("exclusive ownership targets cannot overlap")

    @property
    def token_measure(self) -> torch.Tensor:
        """Resolution-invariant integration measure over each token surface."""

        if self.token_measure_weight is None:
            return torch.ones_like(self.token_observed_fraction)
        return self.token_measure_weight


@dataclass(frozen=True, slots=True)
class SequenceAssignment:
    """Loss-side row identity and its causal activation phase; -1 is unmatched.

    Phase ``2*t`` is the prior before observation ``t`` and phase ``2*t+1`` is
    the posterior after that observation.  This distinguishes an identity
    carried into a window from a fresh identity first observed at time zero.
    """

    row_to_track: torch.Tensor
    binding_start_phase: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.row_to_track.ndim != 2 or self.row_to_track.dtype != torch.long:
            raise ValueError("row_to_track must be long [batch,rows]")
        if (self.row_to_track < -1).any():
            raise ValueError("unmatched rows must use -1")
        if self.binding_start_phase is not None and (
            self.binding_start_phase.shape != self.row_to_track.shape
            or self.binding_start_phase.dtype != torch.long
            or self.binding_start_phase.device != self.row_to_track.device
            or (self.binding_start_phase < 0).any()
        ):
            raise ValueError("assignment binding phase must be non-negative long [batch,rows]")


@dataclass(frozen=True, slots=True)
class RowTaskSupervision:
    """Exact loss-side task targets over the assigned object-row axis."""

    target: torch.Tensor
    valid: torch.Tensor
    exact_task: bool

    def __post_init__(self) -> None:
        if self.target.ndim != 1 or self.valid.shape != self.target.shape:
            raise ValueError("row task targets and validity must share one row axis")
        if (
            not self.target.is_floating_point()
            or not torch.isfinite(self.target).all()
            or ((self.target < 0) | (self.target > 1)).any()
        ):
            raise ValueError("row task targets must be finite floating point in [0,1]")
        if self.valid.dtype != torch.bool or self.valid.device != self.target.device:
            raise ValueError("row task validity must be boolean and share the target device")
        if not isinstance(self.exact_task, bool):
            raise TypeError("row task exactness must be boolean")


def materialize_row_task_supervision(
    targets: NativeSequenceTargets,
    assignment: SequenceAssignment,
    *,
    batch_index: int,
    dtype: torch.dtype,
    binding_valid: torch.Tensor | None = None,
) -> RowTaskSupervision:
    """Map track-level task labels to rows without exposing them to the forward."""

    if not isinstance(targets, NativeSequenceTargets) or not isinstance(
        assignment, SequenceAssignment
    ):
        raise TypeError("row task supervision requires typed targets and assignment")
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or not 0 <= batch_index < targets.masks.shape[0]
    ):
        raise IndexError("row task batch index is outside the target batch")
    if not dtype.is_floating_point:
        raise TypeError("row task supervision requires a floating output dtype")
    if assignment.row_to_track.shape[0] != targets.masks.shape[0]:
        raise ValueError("row task assignment and targets have different batches")

    row_to_track = assignment.row_to_track[batch_index]
    rows = row_to_track.numel()
    if binding_valid is None:
        binding_valid = torch.ones(rows, dtype=torch.bool, device=targets.masks.device)
    elif (
        binding_valid.shape != (rows,)
        or binding_valid.dtype != torch.bool
        or binding_valid.device != targets.masks.device
    ):
        raise ValueError("row task binding validity must be boolean [rows]")
    target = torch.zeros(rows, dtype=dtype, device=targets.masks.device)
    valid = torch.zeros(rows, dtype=torch.bool, device=targets.masks.device)
    for row_index, track_index in enumerate(row_to_track.tolist()):
        if track_index < 0 or not bool(binding_valid[row_index]):
            continue
        if track_index >= targets.masks.shape[2]:
            raise ValueError("row task assignment references an absent target track")
        if targets.task_valid[batch_index, track_index]:
            valid[row_index] = True
            target[row_index] = targets.task_relevance[batch_index, track_index].to(dtype)

    matched_rows = (row_to_track >= 0) & binding_valid
    unmatched_rows = row_to_track < 0
    censored_tracks = targets.capacity_censored[batch_index]
    exact_task = bool(
        torch.equal(
            targets.task_valid[batch_index],
            targets.track_valid[batch_index],
        )
        and (targets.task_relevance[batch_index] > 0).any()
    )
    relevant_target_is_censored = bool(
        (censored_tracks & (targets.task_relevance[batch_index] > 0)).any()
    )
    if (
        exact_task
        and bool(targets.inventory_exhaustive[batch_index].all())
        and not relevant_target_is_censored
    ):
        matched_tracks = torch.zeros_like(targets.track_valid[batch_index])
        matched_tracks[row_to_track[matched_rows]] = True
        unbound_relevant = (
            targets.track_valid[batch_index]
            & targets.task_valid[batch_index]
            & (targets.task_relevance[batch_index] > 0)
            & ~matched_tracks
        )
        if not unbound_relevant.any():
            valid[unmatched_rows] = True
    return RowTaskSupervision(target=target, valid=valid, exact_task=exact_task)


def _support_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    observed_fraction: torch.Tensor,
) -> torch.Tensor | None:
    """Return observed-mass-weighted BCE-plus-Dice for one row/track span."""

    effective_weight = observed_fraction[valid].float()
    if not valid.any() or not (effective_weight > 0).any():
        return None
    predicted = logits[valid].float()
    expected = target[valid].float()
    normalizer = effective_weight.sum()
    binary_ce = (
        F.binary_cross_entropy_with_logits(predicted, expected, reduction="none") * effective_weight
    ).sum() / normalizer
    probability = predicted.sigmoid()
    dice = 1 - (2 * (effective_weight * probability * expected).sum() + 1) / (
        (effective_weight * probability).sum() + (effective_weight * expected).sum() + 1
    )
    return binary_ce + dice


def _dense_task_score(
    predictions: NativeSequencePredictions,
    targets: NativeSequenceTargets,
    *,
    batch_index: int,
) -> torch.Tensor | None:
    """Ground one exact task directly in observed sensor tokens."""

    valid_tracks = targets.track_valid[batch_index]
    if not torch.equal(targets.task_valid[batch_index], valid_tracks):
        return None
    relevance = targets.task_relevance[batch_index] * valid_tracks
    relevant_tracks = relevance > 0
    if not relevant_tracks.any():
        return None

    weighted_masks = targets.masks[batch_index, :, relevant_tracks] * relevance[
        relevant_tracks
    ].view(1, -1, 1)
    target_union = 1 - (1 - weighted_masks).prod(dim=1)
    valid = targets.mask_valid[batch_index, :, relevant_tracks].all(dim=1)
    observed_fraction = (
        targets.token_observed_fraction[batch_index]
        if targets.exclusive_ownership
        else torch.ones_like(targets.token_observed_fraction[batch_index])
    )
    return _support_score(
        predictions.dense_task_grounding_logits[batch_index],
        target_union,
        valid,
        observed_fraction,
    )


def _balanced_task_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor | None:
    """Return capacity-invariant positive/negative conditional logistic risk."""

    if logits.ndim != 1 or targets.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError("balanced task inputs must share one row axis")
    if valid.dtype != torch.bool:
        raise TypeError("balanced task validity must be boolean")
    if not valid.any():
        return None
    selected_logits = logits[valid].float()
    selected_targets = targets[valid].to(dtype=selected_logits.dtype)
    positive_mass = selected_targets.sum()
    negative_mass = (1 - selected_targets).sum()
    components: list[torch.Tensor] = []
    if bool(positive_mass > 0):
        components.append((selected_targets * F.softplus(-selected_logits)).sum() / positive_mass)
    if bool(negative_mass > 0):
        components.append(
            ((1 - selected_targets) * F.softplus(selected_logits)).sum() / negative_mass
        )
    if not components:
        raise ValueError("balanced task targets contain no positive or negative mass")
    return torch.stack(components).mean()


def _validate_exclusive_ownership_inputs(
    log_probability: torch.Tensor,
    expected: torch.Tensor,
    supervised: torch.Tensor,
    observed_fraction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate one exclusive ownership simplex and return FP32 tensors."""

    if log_probability.ndim != 3 or expected.shape != log_probability.shape:
        raise ValueError("exclusive ownership tensors must share [time,tokens,categories]")
    time, tokens, categories = log_probability.shape
    if categories < 2:
        raise ValueError("exclusive ownership requires an object row and context")
    if supervised.shape != (time, tokens) or supervised.dtype != torch.bool:
        raise ValueError("exclusive ownership validity must be boolean [time,tokens]")
    if observed_fraction.shape != (time, tokens):
        raise ValueError("exclusive ownership observed mass must match time and tokens")
    tensors = (expected, observed_fraction)
    if (
        not log_probability.is_floating_point()
        or any(not value.is_floating_point() for value in tensors)
        or any(value.device != log_probability.device for value in (*tensors, supervised))
    ):
        raise ValueError("exclusive ownership inputs must be floating and share one device")
    if (
        not torch.isfinite(log_probability).all()
        or not torch.isfinite(expected).all()
        or not torch.isfinite(observed_fraction).all()
        or (log_probability > 1e-5).any()
        or ((expected < 0) | (expected > 1)).any()
        or ((observed_fraction < 0) | (observed_fraction > 1)).any()
    ):
        raise ValueError("exclusive ownership inputs must be finite probabilities")
    if expected.requires_grad or observed_fraction.requires_grad:
        raise ValueError("exclusive ownership targets and observed mass must be detached")
    if (observed_fraction.masked_select(supervised) <= 0).any():
        raise ValueError("supervised ownership tokens require positive observed mass")

    log_probability32 = log_probability.float()
    expected32 = expected.float()
    observed32 = observed_fraction.float()
    if not supervised.any():
        return log_probability32, expected32, observed32
    tolerance = max(1e-5, torch.finfo(log_probability.dtype).eps * categories)
    if not torch.allclose(
        torch.logsumexp(log_probability32[supervised], dim=-1),
        torch.zeros_like(observed32[supervised]),
        atol=tolerance,
        rtol=0.0,
    ):
        raise ValueError("supervised ownership log probabilities must form one simplex")
    if not torch.allclose(
        expected32[supervised].sum(dim=-1),
        torch.ones_like(observed32[supervised]),
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError("supervised ownership targets must form one simplex")

    return log_probability32, expected32, observed32


def _exclusive_ownership_nll_samples(
    log_probability: torch.Tensor,
    expected: torch.Tensor,
    supervised: torch.Tensor,
    observed_fraction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return observed-mass-weighted proper categorical NLL samples."""

    log_probability32, expected32, observed32 = _validate_exclusive_ownership_inputs(
        log_probability,
        expected,
        supervised,
        observed_fraction,
    )
    if not supervised.any():
        empty = log_probability32.reshape(-1)[:0]
        return empty, observed32.reshape(-1)[:0]
    micro_nll = -(expected32[supervised] * log_probability32[supervised]).sum(dim=-1)
    return micro_nll, observed32[supervised]


def _entity_conditional_ownership_nll_samples(
    log_probability: torch.Tensor,
    expected: torch.Tensor,
    supervised: torch.Tensor,
    observed_fraction: torch.Tensor,
    row_binding_valid: torch.Tensor,
) -> torch.Tensor:
    """Return one proper spatial conditional log score per visible entity.

    The forward posterior remains categorical over rows plus context at every
    token.  This loss-only view conditions each bound object row on its observed
    target mass and normalizes that row across supervised sensor tokens.  It
    therefore gives each visible entity one outer risk unit without changing
    the calibrated token posterior or creating another prediction surface.
    """

    log_probability32, expected32, observed32 = _validate_exclusive_ownership_inputs(
        log_probability,
        expected,
        supervised,
        observed_fraction,
    )
    time, _, categories = log_probability32.shape
    rows = categories - 1
    if row_binding_valid.shape != (time, rows) or row_binding_valid.dtype != torch.bool:
        raise ValueError("entity-conditional binding validity must be boolean [time,rows]")
    if row_binding_valid.device != log_probability.device:
        raise ValueError("entity-conditional binding validity must share the ownership device")

    tiny = torch.finfo(log_probability32.dtype).tiny
    valid = supervised.unsqueeze(-1) & row_binding_valid.unsqueeze(1)
    target_mass = observed32.unsqueeze(-1) * expected32[..., :rows]
    target_mass = target_mass * valid
    target_normalizer = target_mass.sum(dim=(0, 1))
    entity_valid = target_normalizer > 0

    predicted_log_measure = log_probability32[..., :rows] + observed32.clamp_min(
        tiny
    ).log().unsqueeze(-1)
    masked_log_measure = predicted_log_measure.masked_fill(
        ~valid,
        torch.finfo(predicted_log_measure.dtype).min,
    )
    predicted_log_normalizer = torch.logsumexp(masked_log_measure, dim=(0, 1))
    target_distribution = target_mass / target_normalizer.clamp_min(tiny)
    predicted_log_distribution = predicted_log_measure - predicted_log_normalizer.view(1, 1, rows)
    values = -(target_distribution * predicted_log_distribution * valid).sum(dim=(0, 1))
    return values[entity_valid]


def _track_identity_evidence_by_time(
    targets: NativeSequenceTargets,
    *,
    batch_index: int,
) -> torch.Tensor:
    """Return causal positive identity evidence as ``[time,tracks]``.

    A simulator may know that an object exists while every camera is occluded.
    Such privileged existence is not enough to assign a previously unseen
    identity to an arbitrary model row. Established episode bindings remain
    valid through occlusion; only new births are deferred.
    """

    observed = (
        targets.token_observed_fraction[batch_index] > 0
        if targets.exclusive_ownership
        else torch.ones_like(
            targets.token_observed_fraction[batch_index],
            dtype=torch.bool,
        )
    )
    positive_support = (
        targets.mask_valid[batch_index] & (targets.masks[batch_index] > 0) & observed.unsqueeze(1)
    )
    return positive_support.any(dim=2)


def _tracks_with_identity_evidence(
    targets: NativeSequenceTargets,
    *,
    batch_index: int,
) -> torch.Tensor:
    return _track_identity_evidence_by_time(
        targets,
        batch_index=batch_index,
    ).any(dim=0)


def _track_first_identity_evidence_time(
    targets: NativeSequenceTargets,
    *,
    batch_index: int,
) -> torch.Tensor:
    evidence = _track_identity_evidence_by_time(targets, batch_index=batch_index)
    time, tracks = evidence.shape
    indices = torch.arange(time, device=evidence.device).unsqueeze(1).expand(time, tracks)
    return indices.masked_fill(~evidence, time).amin(dim=0)


def assignment_binding_start_phase(
    assignment: SequenceAssignment,
    targets: NativeSequenceTargets,
) -> torch.Tensor:
    """Resolve the first causal phase at which each row identity may supervise.

    Matchers populate this field explicitly so an established prior binding can
    remain active at phase zero. Manually constructed assignments default to a
    fresh birth and activate after the track's first observed identity evidence.
    """

    batch, rows = assignment.row_to_track.shape
    time = targets.masks.shape[1]
    terminal_phase = 2 * time
    if assignment.binding_start_phase is not None:
        resolved = assignment.binding_start_phase
    else:
        resolved = torch.full(
            (batch, rows),
            terminal_phase,
            dtype=torch.long,
            device=assignment.row_to_track.device,
        )
        for batch_index in range(batch):
            matched = assignment.row_to_track[batch_index] >= 0
            if matched.any():
                first = _track_first_identity_evidence_time(
                    targets,
                    batch_index=batch_index,
                )
                resolved[batch_index, matched] = (
                    2 * first[assignment.row_to_track[batch_index, matched]] + 1
                )
    if resolved.shape != (batch, rows) or (resolved > terminal_phase).any():
        raise ValueError("assignment binding phase lies outside the target sequence")
    unmatched = assignment.row_to_track < 0
    if (resolved[unmatched] != terminal_phase).any():
        raise ValueError("unmatched rows must remain unbound for the complete sequence")
    return resolved


def assignment_binding_valid_at_phase(
    assignment: SequenceAssignment,
    targets: NativeSequenceTargets,
    *,
    source_phase: int,
) -> torch.Tensor:
    """Return rows whose physical identity is available at one causal cut."""

    terminal_phase = 2 * targets.masks.shape[1]
    if (
        isinstance(source_phase, bool)
        or not isinstance(source_phase, int)
        or not 0 <= source_phase < terminal_phase
    ):
        raise ValueError("assignment source phase lies outside the target sequence")
    binding_start_phase = assignment_binding_start_phase(assignment, targets)
    return (assignment.row_to_track >= 0) & (binding_start_phase <= source_phase)


def assignment_row_to_track_at_phase(
    assignment: SequenceAssignment,
    targets: NativeSequenceTargets,
    *,
    source_phase: int,
) -> torch.Tensor:
    """Materialize the loss-side row gauge visible at one source phase."""

    valid = assignment_binding_valid_at_phase(
        assignment,
        targets,
        source_phase=source_phase,
    )
    return assignment.row_to_track.masked_fill(~valid, -1)


def _pairwise_assignment_costs(
    predictions: NativeSequencePredictions,
    targets: NativeSequenceTargets,
    *,
    batch_index: int,
    eligible: torch.Tensor,
    causal_cut: int,
) -> torch.Tensor:
    """Return row-to-track costs using evidence no later than one causal cut."""

    support_logits = predictions.support_logits[batch_index].float()
    if (
        isinstance(causal_cut, bool)
        or not isinstance(causal_cut, int)
        or not 0 <= causal_cut < support_logits.shape[0]
    ):
        raise ValueError("assignment causal cut lies outside the sequence")
    _, _, rows = support_logits.shape
    tracks = eligible.numel()
    component_sum = support_logits.new_zeros(rows, tracks)
    component_count = support_logits.new_zeros(rows, tracks)
    first_evidence = _track_first_identity_evidence_time(
        targets,
        batch_index=batch_index,
    ).index_select(0, eligible)
    if (first_evidence > causal_cut).any():
        raise ValueError("assignment causal cut precedes identity evidence")
    time_index = torch.arange(support_logits.shape[0], device=support_logits.device).unsqueeze(1)
    causal_track_valid = (time_index >= first_evidence.unsqueeze(0)) & (time_index <= causal_cut)

    support_target = targets.masks[batch_index, :, eligible].float().permute(0, 2, 1)
    support_valid = targets.mask_valid[batch_index, :, eligible].permute(0, 2, 1)
    support_valid = support_valid & causal_track_valid.unsqueeze(1)
    support_available = support_valid.any(dim=(0, 1))
    expanded_target = support_target.unsqueeze(2).expand(-1, -1, rows, -1)
    expanded_valid = support_valid.unsqueeze(2).expand(-1, -1, rows, -1)
    observed_fraction = (
        targets.token_observed_fraction[batch_index].float()
        if targets.exclusive_ownership
        else torch.ones_like(targets.token_observed_fraction[batch_index])
    )
    expanded_weight = (
        observed_fraction.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, rows, tracks) * expanded_valid
    )
    valid_weight = expanded_weight.sum(dim=(0, 1)).clamp_min(
        torch.finfo(expanded_weight.dtype).tiny
    )
    if targets.exclusive_ownership:
        probability = (
            predictions.ownership[batch_index, :, :, :-1]
            .float()
            .unsqueeze(-1)
            .expand(-1, -1, -1, tracks)
        )
        binary_ce = (
            F.binary_cross_entropy(
                probability.clamp(min=1e-6, max=1 - 1e-6),
                expanded_target,
                reduction="none",
            )
            * expanded_weight
        ).sum(dim=(0, 1)) / valid_weight
    else:
        expanded_logits = support_logits.unsqueeze(-1).expand(-1, -1, -1, tracks)
        binary_ce = (
            F.binary_cross_entropy_with_logits(
                expanded_logits,
                expanded_target,
                reduction="none",
            )
            * expanded_weight
        ).sum(dim=(0, 1)) / valid_weight
        probability = expanded_logits.sigmoid()
    intersection = (probability * expanded_target * expanded_weight).sum(dim=(0, 1))
    probability_mass = (probability * expanded_weight).sum(dim=(0, 1))
    target_mass = (expanded_target * expanded_weight).sum(dim=(0, 1))
    dice = 1 - (2 * intersection + 1) / (probability_mass + target_mass + 1)
    component_sum += (binary_ce + dice) * support_available.unsqueeze(0)
    component_count += support_available.unsqueeze(0)

    existence_target = targets.existence[batch_index, :, eligible].float()
    existence_valid = targets.existence_valid[batch_index, :, eligible] & causal_track_valid
    existence_available = existence_valid.any(dim=0)
    expanded_existence_logits = predictions.existence_logits[batch_index].float().unsqueeze(-1)
    expanded_existence_target = existence_target.unsqueeze(1).expand(-1, rows, -1)
    expanded_existence_valid = existence_valid.unsqueeze(1).expand(-1, rows, -1)
    existence_count = expanded_existence_valid.sum(dim=0).clamp_min(1)
    existence_cost = (
        F.binary_cross_entropy_with_logits(
            expanded_existence_logits.expand(-1, -1, tracks),
            expanded_existence_target,
            reduction="none",
        )
        * expanded_existence_valid
    ).sum(dim=0) / existence_count
    component_sum += existence_cost * existence_available.unsqueeze(0)
    component_count += existence_available.unsqueeze(0)

    if (component_count == 0).any():
        raise ValueError("an eligible track needs task-independent physical supervision")
    return component_sum / component_count


def _assign_birth_tracks_causally(
    predictions: NativeSequencePredictions,
    targets: NativeSequenceTargets,
    *,
    batch_index: int,
    eligible: torch.Tensor,
    free_rows: list[int],
    assignment: torch.Tensor,
    binding_start_phase: torch.Tensor,
) -> None:
    """Allocate each birth at its first evidence cut, then freeze its row."""

    if eligible.numel() == 0:
        return
    first_evidence = _track_first_identity_evidence_time(
        targets,
        batch_index=batch_index,
    ).index_select(0, eligible)
    birth_times = sorted(set(first_evidence.detach().cpu().tolist()))
    for birth_time in birth_times:
        born = eligible[first_evidence == birth_time]
        if born.numel() > len(free_rows):
            raise ValueError("causal births exceed the remaining row capacity")
        costs = _pairwise_assignment_costs(
            predictions,
            targets,
            batch_index=batch_index,
            eligible=born,
            causal_cut=birth_time,
        )
        free_row_tensor = torch.tensor(
            free_rows,
            dtype=torch.long,
            device=targets.masks.device,
        )
        selected_costs = costs.index_select(0, free_row_tensor)
        relative_rows, target_columns = linear_sum_assignment(selected_costs.cpu().numpy())
        selected_rows = free_row_tensor[
            torch.as_tensor(relative_rows, device=free_row_tensor.device)
        ]
        selected_tracks = born[torch.as_tensor(target_columns, device=born.device)]
        assignment[batch_index, selected_rows] = selected_tracks
        binding_start_phase[batch_index, selected_rows] = 2 * birth_time + 1
        selected_row_set = set(selected_rows.detach().cpu().tolist())
        free_rows[:] = [row for row in free_rows if row not in selected_row_set]


@torch.no_grad()
def match_sequence_rows(
    predictions: NativeSequencePredictions,
    targets: NativeSequenceTargets,
) -> SequenceAssignment:
    """Perform label-only Hungarian births at each causal first-evidence cut."""

    batch, time, tokens, rows = predictions.support_logits.shape
    if targets.masks.shape[:2] != (batch, time) or targets.masks.shape[3] != tokens:
        raise ValueError("prediction and target sequence axes differ")
    assignment = torch.full((batch, rows), -1, dtype=torch.long, device=targets.masks.device)
    binding_start_phase = torch.full_like(assignment, 2 * time)
    for batch_index in range(batch):
        uncensored = targets.track_valid[batch_index] & ~targets.capacity_censored[batch_index]
        if int(uncensored.sum().item()) > rows:
            raise ValueError("target producer must mark tracks beyond row capacity as censored")
        eligible = (
            (uncensored & _tracks_with_identity_evidence(targets, batch_index=batch_index))
            .nonzero()
            .flatten()
        )
        if eligible.numel() == 0:
            continue
        _assign_birth_tracks_causally(
            predictions,
            targets,
            batch_index=batch_index,
            eligible=eligible,
            free_rows=list(range(rows)),
            assignment=assignment,
            binding_start_phase=binding_start_phase,
        )
    return SequenceAssignment(assignment, binding_start_phase)


def _validate_binding_identity_axis(
    targets: NativeSequenceTargets,
    *,
    batch_index: int,
    identity_keys: tuple[str, ...],
) -> None:
    valid_count = int(targets.track_valid[batch_index].sum().item())
    expected_valid = torch.arange(
        targets.track_valid.shape[1],
        device=targets.track_valid.device,
    ) < len(identity_keys)
    if (
        len(identity_keys) != valid_count
        or not torch.equal(targets.track_valid[batch_index], expected_valid)
        or len(set(identity_keys)) != len(identity_keys)
        or any(not isinstance(key, str) or not key for key in identity_keys)
    ):
        raise ValueError("row-binding identities differ from valid target tracks")


@torch.no_grad()
def match_sequence_rows_with_bindings(
    predictions: NativeSequencePredictions,
    targets: NativeSequenceTargets,
    *,
    identity_keys_by_batch: tuple[tuple[str, ...], ...],
    prior_bindings_by_batch: tuple[RowBindings, ...],
) -> SequenceAssignment:
    """Match births only while preserving every established episode binding."""

    batch, time, tokens, rows = predictions.support_logits.shape
    if targets.masks.shape[:2] != (batch, time) or targets.masks.shape[3] != tokens:
        raise ValueError("prediction and target sequence axes differ")
    if len(identity_keys_by_batch) != batch or len(prior_bindings_by_batch) != batch:
        raise ValueError("row-binding metadata differs from the prediction batch")

    assignment = torch.full(
        (batch, rows),
        -1,
        dtype=torch.long,
        device=targets.masks.device,
    )
    binding_start_phase = torch.full_like(assignment, 2 * time)
    for batch_index, (identity_keys, prior_bindings) in enumerate(
        zip(identity_keys_by_batch, prior_bindings_by_batch, strict=True)
    ):
        _validate_binding_identity_axis(
            targets,
            batch_index=batch_index,
            identity_keys=identity_keys,
        )
        prior = row_binding_map(prior_bindings, capacity=rows)
        occupied_rows = set(prior.values())
        eligible = (
            (targets.track_valid[batch_index] & ~targets.capacity_censored[batch_index])
            .nonzero()
            .flatten()
        )
        identity_evidence = _tracks_with_identity_evidence(
            targets,
            batch_index=batch_index,
        )
        new_tracks: list[int] = []
        for track_index in eligible.tolist():
            identity = identity_keys[track_index]
            row = prior.get(identity)
            if row is None:
                if identity_evidence[track_index]:
                    new_tracks.append(track_index)
                continue
            assignment[batch_index, row] = track_index
            binding_start_phase[batch_index, row] = 0

        if not new_tracks:
            continue
        free_rows = sorted(set(range(rows)) - occupied_rows)
        if len(new_tracks) > len(free_rows):
            raise ValueError("episode bindings reserve too few free rows for uncensored births")
        new_track_tensor = torch.tensor(
            new_tracks,
            dtype=torch.long,
            device=targets.masks.device,
        )
        _assign_birth_tracks_causally(
            predictions,
            targets,
            batch_index=batch_index,
            eligible=new_track_tensor,
            free_rows=free_rows,
            assignment=assignment,
            binding_start_phase=binding_start_phase,
        )
    return SequenceAssignment(assignment, binding_start_phase)


def extend_sequence_row_bindings(
    assignment: SequenceAssignment,
    targets: NativeSequenceTargets,
    *,
    identity_keys_by_batch: tuple[tuple[str, ...], ...],
    prior_bindings_by_batch: tuple[RowBindings, ...],
) -> tuple[RowBindings, ...]:
    """Commit births known by the primary posterior without changing identities.

    Later frames in a local BPTT window are loss-only.  Their labels may train
    those later posterior surfaces, but they cannot be committed beside the
    primary time-zero posterior that advances the online lane.
    """

    batch, rows = assignment.row_to_track.shape
    if (
        targets.masks.shape[0] != batch
        or len(identity_keys_by_batch) != batch
        or len(prior_bindings_by_batch) != batch
    ):
        raise ValueError("row-binding extension inputs differ by batch")
    binding_start_phase = assignment_binding_start_phase(assignment, targets)
    resolved: list[RowBindings] = []
    for batch_index, (identity_keys, prior_bindings) in enumerate(
        zip(identity_keys_by_batch, prior_bindings_by_batch, strict=True)
    ):
        _validate_binding_identity_axis(
            targets,
            batch_index=batch_index,
            identity_keys=identity_keys,
        )
        current = row_binding_map(prior_bindings, capacity=rows)
        for row, track_index in enumerate(
            assignment.row_to_track[batch_index].detach().cpu().tolist()
        ):
            if track_index < 0 or int(binding_start_phase[batch_index, row].item()) > 1:
                continue
            if track_index >= len(identity_keys):
                raise ValueError("row assignment references an absent identity")
            identity = identity_keys[track_index]
            old_row = current.get(identity)
            if old_row is not None and old_row != row:
                raise ValueError("an established identity changed rows")
            current[identity] = row
        resolved.append(normalize_row_bindings(current, capacity=rows))
    return tuple(resolved)


def _term(
    name: str,
    values: list[torch.Tensor],
    *,
    reference: torch.Tensor,
    weight: float,
    sample_weights: list[torch.Tensor] | None = None,
) -> ObjectiveTerm:
    if values:
        joined = torch.cat([value.reshape(-1) for value in values])
        valid = torch.ones_like(joined, dtype=torch.bool)
        joined_weights = (
            None
            if sample_weights is None
            else torch.cat([value.reshape(-1) for value in sample_weights]).to(
                device=joined.device,
                dtype=torch.float32,
            )
        )
        if joined_weights is not None and joined_weights.shape != joined.shape:
            raise ValueError("objective values and sample weights differ")
    else:
        # PyTorch 2.8 FSDP2 requires every rank to retain the same backward
        # participation even when local labels provide no evidence.
        joined = reference.reshape(-1)[:1] * 0
        valid = torch.zeros(1, dtype=torch.bool, device=reference.device)
        joined_weights = None if sample_weights is None else reference.new_zeros(1)
    return ObjectiveTerm(
        name=name,
        values=joined,
        valid=valid,
        weight=weight,
        sample_weight=joined_weights,
    )


def _validate_assignment(
    assignment: SequenceAssignment,
    targets: NativeSequenceTargets,
    *,
    batch: int,
    rows: int,
) -> None:
    if assignment.row_to_track.shape != (batch, rows):
        raise ValueError("sequence assignment does not match prediction rows")
    if assignment.row_to_track.device != targets.masks.device:
        raise ValueError("sequence assignment and targets must share one device")
    tracks = targets.masks.shape[2]
    if (assignment.row_to_track >= tracks).any():
        raise ValueError("sequence assignment references an absent target track")
    for batch_index in range(batch):
        selected = assignment.row_to_track[batch_index]
        matched = selected[selected >= 0]
        if matched.unique().numel() != matched.numel():
            raise ValueError("sequence assignment maps two rows to one track")
        if (
            matched.numel()
            and (
                ~targets.track_valid[batch_index, matched]
                | targets.capacity_censored[batch_index, matched]
            ).any()
        ):
            raise ValueError("sequence assignment selected an invalid or censored track")
    assignment_binding_start_phase(assignment, targets)


def sequence_set_terms(
    predictions: NativeSequencePredictions,
    targets: NativeSequenceTargets,
    assignment: SequenceAssignment,
    *,
    support_weight: float,
    existence_weight: float,
    task_weight: float,
    dense_task_weight: float,
    ownership_weight: float = 0.0,
    ownership_estimator: str = TOKEN_MICRO_OWNERSHIP,
    include_task_term: bool = True,
) -> tuple[ObjectiveTerm, ...]:
    """Build proper structural terms without turning unknowns into negatives."""

    if not isinstance(include_task_term, bool):
        raise TypeError("task-term inclusion marker must be boolean")
    if ownership_estimator not in OWNERSHIP_ESTIMATORS:
        raise ValueError("unknown ownership estimator")
    batch, time, tokens, rows = predictions.support_logits.shape
    _validate_assignment(assignment, targets, batch=batch, rows=rows)
    binding_start_phase = assignment_binding_start_phase(assignment, targets)
    time_indices = torch.arange(time, device=targets.masks.device)
    posterior_phases = 2 * time_indices + 1
    support_values: list[torch.Tensor] = []
    existence_values: list[torch.Tensor] = []
    task_values: list[torch.Tensor] = []
    dense_task_values: list[torch.Tensor] = []
    ownership_values: list[torch.Tensor] = []
    ownership_sample_weights: list[torch.Tensor] = []
    ownership_nll_values: list[torch.Tensor] = []
    ownership_nll_sample_weights: list[torch.Tensor] = []
    ownership_entity_values: list[torch.Tensor] = []
    for batch_index in range(batch):
        row_binding_valid = posterior_phases.unsqueeze(1) >= binding_start_phase[
            batch_index
        ].unsqueeze(0)
        dense_task = _dense_task_score(
            predictions,
            targets,
            batch_index=batch_index,
        )
        if dense_task is not None:
            dense_task_values.append(dense_task.reshape(1))
        for row_index, track_index in enumerate(assignment.row_to_track[batch_index].tolist()):
            if track_index < 0:
                continue
            if not targets.exclusive_ownership:
                support = _support_score(
                    predictions.support_logits[batch_index, :, :, row_index],
                    targets.masks[batch_index, :, track_index],
                    targets.mask_valid[batch_index, :, track_index]
                    & row_binding_valid[:, row_index].unsqueeze(-1),
                    torch.ones_like(targets.token_observed_fraction[batch_index]),
                )
                if support is not None:
                    support_values.append(support.reshape(1))
            existence_valid = (
                targets.existence_valid[batch_index, :, track_index]
                & row_binding_valid[:, row_index]
            )
            existence_loss = F.binary_cross_entropy_with_logits(
                predictions.existence_logits[batch_index, :, row_index],
                targets.existence[batch_index, :, track_index].to(
                    predictions.existence_logits.dtype
                ),
                reduction="none",
            )
            existence_values.append(existence_loss[existence_valid])

        matched_rows = assignment.row_to_track[batch_index] >= 0
        censored_tracks = targets.capacity_censored[batch_index]
        row_to_track = assignment.row_to_track[batch_index]
        track_binding_start_phase = torch.full(
            (targets.masks.shape[2],),
            2 * time,
            dtype=torch.long,
            device=targets.masks.device,
        )
        track_binding_start_phase[row_to_track[matched_rows]] = binding_start_phase[
            batch_index,
            matched_rows,
        ]
        unbound_track_by_time = (
            targets.track_valid[batch_index].unsqueeze(0)
            & ~censored_tracks.unsqueeze(0)
            & (posterior_phases.unsqueeze(1) < track_binding_start_phase.unsqueeze(0))
        )
        unbound_present = (
            targets.existence_valid[batch_index]
            & (targets.existence[batch_index] > 0)
            & unbound_track_by_time
        ).any(dim=1)
        unbound_visible = (
            targets.mask_valid[batch_index]
            & (targets.masks[batch_index] > 0)
            & unbound_track_by_time.unsqueeze(-1)
        ).any(dim=1)
        if include_task_term:
            row_task = materialize_row_task_supervision(
                targets,
                assignment,
                batch_index=batch_index,
                dtype=predictions.task_relevance_logits.dtype,
                binding_valid=row_binding_valid[-1],
            )
            task_score = _balanced_task_score(
                predictions.task_relevance_logits[batch_index],
                row_task.target,
                row_task.valid,
            )
            if task_score is not None:
                task_values.append(task_score.reshape(1))
        censored_visible = (
            (
                targets.mask_valid[batch_index, :, censored_tracks]
                & (targets.masks[batch_index, :, censored_tracks] > 0)
            ).any(dim=1)
            if censored_tracks.any()
            else torch.zeros(time, tokens, dtype=torch.bool, device=targets.masks.device)
        )
        censored_present = (
            (
                targets.existence_valid[batch_index, :, censored_tracks]
                & (targets.existence[batch_index, :, censored_tracks] > 0)
            ).any(dim=1)
            if censored_tracks.any()
            else torch.zeros(time, dtype=torch.bool, device=targets.masks.device)
        )

        if targets.exclusive_ownership:
            if predictions.ownership_log_probability is None:
                raise ValueError("exclusive ownership requires attached log probabilities")
            safe_track = row_to_track.clamp_min(0)
            gathered_masks = targets.masks[batch_index, :, safe_track]
            gathered_valid = targets.mask_valid[batch_index, :, safe_track]
            expected_rows = (
                gathered_masks
                * gathered_valid
                * matched_rows.view(1, rows, 1)
                * row_binding_valid.unsqueeze(-1)
            ).permute(0, 2, 1)
            # Keep fractional loss targets in FP32; BF16 complement rounding can
            # otherwise destroy the object-plus-context simplex.
            expected_rows = expected_rows.float()

            unassigned_positive = (
                targets.mask_valid[batch_index]
                & (targets.masks[batch_index] > 0)
                & unbound_track_by_time.unsqueeze(-1)
            ).any(dim=1)
            supervised = (
                (targets.token_observed_fraction[batch_index] > 0)
                & ~censored_visible
                & ~unassigned_positive
            )
            object_mass = expected_rows.sum(dim=-1, keepdim=True)
            context_mass = (1 - object_mass).clamp_min(0)
            expected = torch.cat((expected_rows, context_mass), dim=-1)
            # Projection arithmetic may exceed unit mass by the small tolerance
            # accepted by NativeSequenceTargets.  Renormalize that numerical
            # residue before evaluating a proper categorical likelihood.
            expected = expected / expected.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(expected.dtype).tiny
            )
            micro_nll, micro_weight = _exclusive_ownership_nll_samples(
                predictions.ownership_log_probability[batch_index],
                expected,
                supervised,
                targets.token_observed_fraction[batch_index],
            )
            if micro_nll.numel():
                ownership_values.append(micro_nll)
                ownership_sample_weights.append(micro_weight)
                # Retain a zero-weight audit alias so historical report readers
                # can compare exact NLL without changing the optimized schema.
                ownership_nll_values.append(micro_nll)
                ownership_nll_sample_weights.append(micro_weight)
            if ownership_estimator == TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP:
                entity_nll = _entity_conditional_ownership_nll_samples(
                    predictions.ownership_log_probability[batch_index],
                    expected,
                    supervised,
                    targets.token_observed_fraction[batch_index],
                    row_binding_valid,
                )
                if entity_nll.numel():
                    ownership_entity_values.append(entity_nll)
        else:
            negative_support_valid = (
                (targets.token_observed_fraction[batch_index] > 0)
                & ~censored_visible
                & ~unbound_visible
            )
            for row_index in (~matched_rows).nonzero().flatten().tolist():
                support = _support_score(
                    predictions.support_logits[batch_index, :, :, row_index],
                    torch.zeros_like(predictions.support_logits[batch_index, :, :, row_index]),
                    negative_support_valid,
                    targets.token_observed_fraction[batch_index],
                )
                if support is not None:
                    support_values.append(support.reshape(1))

        negative_existence_valid = (
            targets.inventory_exhaustive[batch_index] & ~censored_present & ~unbound_present
        )
        for row_index in (~matched_rows).nonzero().flatten().tolist():
            existence_values.append(
                F.softplus(predictions.existence_logits[batch_index, :, row_index])[
                    negative_existence_valid
                ]
            )

    terms = (
        _term(
            "set/support",
            support_values,
            reference=predictions.support_logits,
            weight=support_weight,
        ),
        _term(
            "set/existence",
            existence_values,
            reference=predictions.existence_logits,
            weight=existence_weight,
        ),
    )
    if include_task_term:
        terms += (
            _term(
                "set/task",
                task_values,
                reference=predictions.task_relevance_logits,
                weight=task_weight,
            ),
        )
    terms += (
        _term(
            "set/task_dense",
            dense_task_values,
            reference=predictions.dense_task_grounding_logits,
            weight=dense_task_weight,
        ),
    )
    if targets.exclusive_ownership:
        ownership_fraction = (
            1.0
            if ownership_estimator == TOKEN_MICRO_OWNERSHIP
            else 1.0 - ENTITY_CONDITIONAL_OWNERSHIP_FRACTION
        )
        ownership_terms = (
            _term(
                "set/ownership",
                ownership_values,
                reference=predictions.ownership,
                weight=ownership_weight * ownership_fraction,
                sample_weights=ownership_sample_weights,
            ),
            _term(
                "set/ownership_nll",
                ownership_nll_values,
                reference=predictions.ownership,
                weight=0.0,
                sample_weights=ownership_nll_sample_weights,
            ),
        )
        if ownership_estimator == TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP:
            ownership_terms += (
                _term(
                    "set/ownership_entity",
                    ownership_entity_values,
                    reference=predictions.ownership,
                    weight=ownership_weight * ENTITY_CONDITIONAL_OWNERSHIP_FRACTION,
                ),
            )
        return terms + ownership_terms
    return terms
