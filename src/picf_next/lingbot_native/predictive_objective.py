"""V-JEPA-style stop-gradient targets for native LingBot object rows.

The production CALVIN route uses separately frozen LingBot DINO-video targets,
not EMA/deep/dense V-JEPA 2.1 pretraining. PICF's narrow adaptation is loss-side
row/track gathering after the shared LingBot forward pass. No target metadata
enters a query or recurrent state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.objective import ObjectiveSupport, ObjectiveTerm


@runtime_checkable
class PredictiveRowAssignment(Protocol):
    """Loss-side row gauge required by an object-indexed predictive target."""

    row_to_track: torch.Tensor


@runtime_checkable
class CausalPredictiveRowAssignment(PredictiveRowAssignment, Protocol):
    """A row gauge whose identities become available at explicit causal phases."""

    binding_start_phase: torch.Tensor | None


class TargetEncoderMode(str, Enum):
    FROZEN = "frozen"


def _sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class NativePredictiveTarget:
    """Independent target features indexed by source track, never model rows."""

    modality: str
    features: torch.Tensor
    valid: torch.Tensor
    importance: torch.Tensor
    route_ids: torch.Tensor
    horizons: torch.Tensor
    source: PredictionSource
    evidence: PredictionEvidence
    encoder_mode: TargetEncoderMode
    source_batch_digest: str
    target_data_digest: str
    encoder_digest: str
    query_schema_digest: str
    validity_semantics: str
    track_identity_keys: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 4:
            raise ValueError("predictive target features must be [batch,tracks,queries,width]")
        batch, tracks, queries, width = self.features.shape
        if tracks < 1:
            raise ValueError("predictive targets require at least one source track")
        if width < 2:
            raise ValueError("predictive target width must be at least two")
        if self.valid.shape != self.features.shape[:-1] or self.valid.dtype != torch.bool:
            raise ValueError("predictive target validity must match non-feature axes")
        if self.importance.shape != self.valid.shape or not self.importance.is_floating_point():
            raise ValueError("predictive importance must be floating point and match validity")
        if self.route_ids.shape != (batch, queries) or self.route_ids.dtype != torch.long:
            raise ValueError("predictive target route IDs must be long [batch,queries]")
        if self.horizons.shape != (batch, queries) or self.horizons.dtype != torch.long:
            raise ValueError("predictive target horizons must be long [batch,queries]")
        tensors = (
            self.features,
            self.valid,
            self.importance,
            self.route_ids,
            self.horizons,
        )
        if any(value.device != self.features.device for value in tensors):
            raise ValueError("predictive target tensors must share one device")
        if not self.features.is_floating_point() or not torch.isfinite(self.features).all():
            raise ValueError("predictive target features must be finite floating point")
        if (
            not torch.isfinite(self.importance).all()
            or ((self.importance < 0) | (self.importance > 1)).any()
        ):
            raise ValueError("predictive target importance must lie in [0,1]")
        if not torch.equal(self.valid, self.importance > 0):
            raise ValueError("target validity must exactly identify positive importance")
        if any(value.requires_grad or value.grad_fn is not None for value in tensors):
            raise ValueError("predictive target tensors must be stop-gradient")
        if (self.route_ids < 0).any() or (self.horizons < 0).any():
            raise ValueError("target routes and horizons must be non-negative")
        if not isinstance(self.source, PredictionSource) or not isinstance(
            self.evidence, PredictionEvidence
        ):
            raise TypeError("predictive target source and evidence must use frozen enums")
        if not isinstance(self.encoder_mode, TargetEncoderMode):
            raise TypeError("target encoder mode must use the frozen enum")
        if not isinstance(self.modality, str) or not self.modality:
            raise ValueError("predictive target modality must be non-empty")
        if not isinstance(self.validity_semantics, str) or not self.validity_semantics:
            raise ValueError("predictive target validity semantics must be non-empty")
        if (
            not isinstance(self.track_identity_keys, tuple)
            or len(self.track_identity_keys) != batch
        ):
            raise ValueError("predictive target identities must provide one tuple per batch")
        for batch_index, identity_keys in enumerate(self.track_identity_keys):
            if (
                not isinstance(identity_keys, tuple)
                or not identity_keys
                or len(identity_keys) > tracks
                or len(set(identity_keys)) != len(identity_keys)
                or any(not isinstance(key, str) or not key for key in identity_keys)
            ):
                raise ValueError("predictive target track identities must be non-empty and unique")
            if (
                self.valid[batch_index, len(identity_keys) :].any()
                or self.importance[batch_index, len(identity_keys) :].any()
            ):
                raise ValueError("padded predictive tracks cannot carry target mass")
        for name in (
            "source_batch_digest",
            "target_data_digest",
            "encoder_digest",
            "query_schema_digest",
        ):
            _sha256(getattr(self, name), name)

    @property
    def supports_object_binding_claim(self) -> bool:
        return self.evidence in (
            PredictionEvidence.CURRENT_CORRECTION,
            PredictionEvidence.CURRENT_PRIOR,
            PredictionEvidence.CURRENT_POSTERIOR,
            PredictionEvidence.PRIOR_ONLY,
            PredictionEvidence.FUTURE,
            PredictionEvidence.OMITTED_MODALITY,
        )


def make_native_predictive_target(
    *,
    modality: str,
    features: torch.Tensor,
    valid: torch.Tensor,
    importance: torch.Tensor | None,
    route_ids: torch.Tensor,
    horizons: torch.Tensor,
    source: PredictionSource,
    evidence: PredictionEvidence,
    encoder_mode: TargetEncoderMode,
    source_batch_digest: str,
    target_data_digest: str,
    encoder_digest: str,
    query_schema_digest: str,
    validity_semantics: str,
    track_identity_keys: tuple[tuple[str, ...], ...],
) -> NativePredictiveTarget:
    detached_valid = valid.detach().clone()
    detached_importance = (
        detached_valid.to(torch.float32)
        if importance is None
        else importance.detach().float().clone()
    )
    return NativePredictiveTarget(
        modality=modality,
        features=features.detach().clone(),
        valid=detached_valid,
        importance=detached_importance,
        route_ids=route_ids.detach().clone(),
        horizons=horizons.detach().clone(),
        source=source,
        evidence=evidence,
        encoder_mode=encoder_mode,
        source_batch_digest=source_batch_digest,
        target_data_digest=target_data_digest,
        encoder_digest=encoder_digest,
        query_schema_digest=query_schema_digest,
        validity_semantics=validity_semantics,
        track_identity_keys=track_identity_keys,
    )


def make_object_summary_target(
    *,
    modality: str,
    token_features: torch.Tensor,
    track_support: torch.Tensor,
    token_valid: torch.Tensor,
    token_footprint: torch.Tensor,
    route_ids: torch.Tensor,
    horizons: torch.Tensor,
    source: PredictionSource,
    evidence: PredictionEvidence,
    encoder_mode: TargetEncoderMode,
    source_batch_digest: str,
    target_data_digest: str,
    encoder_digest: str,
    query_schema_digest: str,
    validity_semantics: str,
    track_identity_keys: tuple[tuple[str, ...], ...],
    minimum_support: float = 0.0,
) -> NativePredictiveTarget:
    """Pool normalized detached target tokens with independent track support."""

    if token_features.ndim != 3:
        raise ValueError("token features must have shape [batch,tokens,width]")
    if track_support.ndim != 3 or track_support.shape[:2] != token_features.shape[:2]:
        raise ValueError("track support must have shape [batch,tokens,tracks]")
    if token_valid.shape != token_features.shape[:2] or token_valid.dtype != torch.bool:
        raise ValueError("token validity must be boolean [batch,tokens]")
    if token_footprint.shape != token_valid.shape or not token_footprint.is_floating_point():
        raise ValueError("token footprint must be floating point [batch,tokens]")
    tensors = (track_support, token_valid, token_footprint)
    if any(value.device != token_features.device for value in tensors):
        raise ValueError("object-summary tensors must share one device")
    if not token_features.is_floating_point() or not track_support.is_floating_point():
        raise TypeError("object-summary features and support must be floating point")
    if any(not torch.isfinite(value).all() for value in (token_features, track_support)):
        raise ValueError("object-summary tensors must be finite")
    if (track_support < 0).any() or (track_support.sum(dim=-1) > 1 + 1e-5).any():
        raise ValueError("track support must be a non-negative sub-probability simplex")
    if not torch.isfinite(token_footprint).all() or (token_footprint < 0).any():
        raise ValueError("token footprint must be finite and non-negative")
    if (
        isinstance(minimum_support, bool)
        or not isinstance(minimum_support, (int, float))
        or not math.isfinite(minimum_support)
        or minimum_support < 0
    ):
        raise ValueError("minimum support must be finite and non-negative")
    if route_ids.shape[1] != 1 or horizons.shape != route_ids.shape:
        raise ValueError("an object-summary route uses exactly one non-spatial query")

    normalized = F.layer_norm(
        token_features.detach().float(),
        (token_features.shape[-1],),
    )
    weights = (
        track_support.detach().float()
        * token_valid.detach().unsqueeze(-1)
        * token_footprint.detach().float().unsqueeze(-1)
    )
    support = weights.sum(dim=1)
    summaries = torch.einsum("bnj,bnd->bjd", weights, normalized)
    summaries = summaries / support.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    valid = support > minimum_support
    summaries = summaries.masked_fill(~valid.unsqueeze(-1), 0).unsqueeze(2)
    importance = support.masked_fill(~valid, 0).clamp_max(1).unsqueeze(2)
    return make_native_predictive_target(
        modality=modality,
        features=summaries,
        valid=valid.unsqueeze(2),
        importance=importance,
        route_ids=route_ids,
        horizons=horizons,
        source=source,
        evidence=evidence,
        encoder_mode=encoder_mode,
        source_batch_digest=source_batch_digest,
        target_data_digest=target_data_digest,
        encoder_digest=encoder_digest,
        query_schema_digest=query_schema_digest,
        validity_semantics=validity_semantics,
        track_identity_keys=track_identity_keys,
    )


class NativePredictiveReadout(nn.Module):
    """Route-indexed linear width adapter; semantics remain in LingBot."""

    def __init__(
        self,
        host_width: int,
        target_width: int,
        route_count: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dimensions = (host_width, target_width, route_count)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
            raise TypeError("predictive readout dimensions must be integers")
        if min(dimensions) <= 0:
            raise ValueError("predictive readout dimensions must be positive")
        self.host_width = host_width
        self.target_width = target_width
        self.route_count = route_count
        self.weight = nn.Parameter(
            torch.empty(route_count, target_width, host_width, device=device, dtype=dtype)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, hidden: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[-1] != self.host_width:
            raise ValueError("prediction hidden must be [batch,rows,queries,host_width]")
        if route_ids.shape != (hidden.shape[0], hidden.shape[2]) or route_ids.dtype != torch.long:
            raise ValueError("route IDs must be long [batch,queries]")
        if route_ids.device != hidden.device or hidden.device != self.weight.device:
            raise ValueError("prediction hidden, routes and readout must share one device")
        if hidden.dtype != self.weight.dtype:
            raise ValueError("prediction hidden and readout must share one dtype")
        if (route_ids < 0).any() or (route_ids >= self.route_count).any():
            raise ValueError("prediction route is outside the readout table")
        route_weight = self.weight[route_ids]
        return torch.einsum("bkqd,bqod->bkqo", hidden, route_weight)


def _validate_target_request(
    target: NativePredictiveTarget,
    request: NativePredictionRequest,
) -> None:
    if target.source != request.source or target.evidence != request.evidence:
        raise ValueError("prediction target and source request differ")
    if not torch.equal(target.route_ids, request.route_ids):
        raise ValueError("prediction target and request routes differ")
    if not torch.equal(target.horizons, request.horizons):
        raise ValueError("prediction target and request horizons differ")
    if target.features.shape[0] != request.batch_size:
        raise ValueError("prediction target and request batches differ")
    if target.features.shape[2] != request.query_count:
        raise ValueError("prediction target and request query counts differ")


@dataclass(frozen=True, slots=True)
class NativePredictiveLossInput:
    """Unmatched loss input produced after a shared host forward.

    Source-track assignment is deliberately absent.  The CALVIN objective owns
    matching and materializes these inputs only after every host forward and
    every independent target have been validated.
    """

    prediction: torch.Tensor
    request: NativePredictionRequest
    target: NativePredictiveTarget
    weight: float
    identity_source_phase: int
    loss_power: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, torch.Tensor):
            raise TypeError("predictive loss input prediction must be a tensor")
        if not isinstance(self.request, NativePredictionRequest) or not isinstance(
            self.target, NativePredictiveTarget
        ):
            raise TypeError("predictive loss input requires typed request and target values")
        _validate_target_request(self.target, self.request)
        for name, value, minimum in (
            ("weight", self.weight, 0.0),
            ("loss_power", self.loss_power, 1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < minimum
            ):
                raise ValueError(f"predictive {name} must be finite and at least {minimum}")
        if (
            isinstance(self.identity_source_phase, bool)
            or not isinstance(self.identity_source_phase, int)
            or self.identity_source_phase < 0
        ):
            raise ValueError("predictive identity source phase must be a non-negative integer")


def _native_predictive_name(
    request: NativePredictionRequest,
    target: NativePredictiveTarget,
) -> str:
    family = {
        PredictionEvidence.CURRENT_CORRECTION: "correction",
        PredictionEvidence.CURRENT_PRIOR: "filter_prior",
        PredictionEvidence.CURRENT_POSTERIOR: "filter_posterior",
        PredictionEvidence.FUTURE: "rollout",
    }.get(request.evidence, "xmod")
    grade = "binding" if request.supports_object_binding_claim else "representation"
    return f"{family}/{target.modality}/{grade}"


def _native_predictive_alignment(
    *,
    request: NativePredictionRequest,
    target: NativePredictiveTarget,
    assignment: PredictiveRowAssignment,
    row_binding_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_target_request(target, request)
    if not isinstance(assignment, PredictiveRowAssignment):
        raise TypeError("predictive loss requires a loss-side row assignment")
    batch = request.batch_size
    if assignment.row_to_track.ndim != 2 or assignment.row_to_track.shape[0] != batch:
        raise ValueError("predictive assignment differs from its request batch axis")
    rows = assignment.row_to_track.shape[1]
    device = target.features.device
    if (
        assignment.row_to_track.dtype != torch.long
        or assignment.row_to_track.device != device
        or (assignment.row_to_track < -1).any()
    ):
        raise ValueError("predictive assignment must be long [batch,rows] on the loss device")
    if (
        row_binding_valid.shape != (batch, rows)
        or row_binding_valid.dtype != torch.bool
        or row_binding_valid.device != device
    ):
        raise ValueError("predictive row-binding validity must be boolean [batch,rows]")
    tracks = target.features.shape[1]
    if (assignment.row_to_track >= tracks).any():
        raise ValueError("predictive assignment references an absent target track")

    matched = (assignment.row_to_track >= 0) & row_binding_valid
    gather_track = assignment.row_to_track.clamp_min(0)
    batch_index = torch.arange(batch, device=device).unsqueeze(1)
    expected = target.features[batch_index, gather_track]
    valid = target.valid[batch_index, gather_track] & matched.unsqueeze(-1)
    valid = valid & request.valid.unsqueeze(1)
    importance = target.importance[batch_index, gather_track] * valid
    return expected, valid, importance


def native_predictive_term(
    *,
    prediction: torch.Tensor,
    request: NativePredictionRequest,
    target: NativePredictiveTarget,
    assignment: PredictiveRowAssignment,
    row_binding_valid: torch.Tensor,
    weight: float,
    loss_power: float = 1.0,
) -> ObjectiveTerm:
    """Apply assignment after forward and score the official normalized power loss."""

    _validate_target_request(target, request)
    if (
        isinstance(loss_power, bool)
        or not isinstance(loss_power, (int, float))
        or not math.isfinite(loss_power)
        or loss_power < 1
    ):
        raise ValueError("predictive loss power must be finite and at least one")
    if prediction.ndim != 4 or not prediction.is_floating_point():
        raise ValueError("prediction must be floating [batch,rows,queries,width]")
    if prediction.shape[0] != request.batch_size or prediction.shape[2] != request.query_count:
        raise ValueError("prediction differs from its request batch/query axes")
    if prediction.device != target.features.device:
        raise ValueError("prediction and target features must share one device")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction contains NaN or infinity")
    batch, rows, queries, width = prediction.shape
    if target.features.shape[-1] != width:
        raise ValueError("predictive readout and target widths differ")
    if assignment.row_to_track.shape != (batch, rows):
        raise ValueError("predictive assignment differs from row predictions")
    expected, valid, importance = _native_predictive_alignment(
        request=request,
        target=target,
        assignment=assignment,
        row_binding_valid=row_binding_valid,
    )
    normalized_prediction = F.layer_norm(prediction.float(), (width,))
    normalized_target = F.layer_norm(expected.float(), (width,)).detach()
    values = (normalized_prediction - normalized_target).abs().pow(loss_power).mean(
        dim=-1
    ) / loss_power
    # Preserve absolute evidence mass. Renormalizing a barely visible sliver to
    # one full sample would erase the uncertainty represented by its support.
    values = values * importance.to(values)
    return ObjectiveTerm(
        name=_native_predictive_name(request, target),
        values=values,
        valid=valid,
        weight=weight,
    )


def materialize_native_predictive_support(
    *,
    request: NativePredictionRequest,
    target: NativePredictiveTarget,
    weight: float,
    identity_source_phase: int,
    assignment: CausalPredictiveRowAssignment,
    expected_track_identity_keys: tuple[tuple[str, ...], ...],
    sequence_time_count: int,
) -> ObjectiveSupport:
    """Materialize exact detached normalization support before a branch forward."""

    if not isinstance(assignment, CausalPredictiveRowAssignment):
        raise TypeError("predictive support requires a causal row assignment")
    if target.track_identity_keys != expected_track_identity_keys:
        raise ValueError("predictive support track identities differ from structural targets")
    if (
        isinstance(sequence_time_count, bool)
        or not isinstance(sequence_time_count, int)
        or sequence_time_count <= 0
    ):
        raise ValueError("predictive support sequence time count must be positive")
    terminal_phase = 2 * sequence_time_count
    binding_start_phase = assignment.binding_start_phase
    if binding_start_phase is None:
        raise ValueError("predictive support requires explicit causal binding phases")
    if (
        isinstance(identity_source_phase, bool)
        or not isinstance(identity_source_phase, int)
        or not 0 <= identity_source_phase < terminal_phase
    ):
        raise ValueError("predictive support source phase lies outside the sequence")
    row_binding_valid = binding_start_phase <= identity_source_phase
    _expected, valid, _importance = _native_predictive_alignment(
        request=request,
        target=target,
        assignment=assignment,
        row_binding_valid=row_binding_valid,
    )
    return ObjectiveSupport(
        name=_native_predictive_name(request, target),
        valid=valid.detach(),
        weight=weight,
    )


def materialize_native_predictive_terms(
    inputs: Sequence[NativePredictiveLossInput],
    *,
    assignment: CausalPredictiveRowAssignment,
    expected_track_identity_keys: tuple[tuple[str, ...], ...],
    sequence_time_count: int,
) -> tuple[ObjectiveTerm, ...]:
    """Apply one post-forward assignment and merge repeated route families."""

    if not isinstance(assignment, CausalPredictiveRowAssignment):
        raise TypeError("predictive term materialization requires a causal row assignment")
    if not isinstance(expected_track_identity_keys, tuple):
        raise TypeError("expected predictive track identities must be a tuple")
    if (
        isinstance(sequence_time_count, bool)
        or not isinstance(sequence_time_count, int)
        or sequence_time_count <= 0
    ):
        raise ValueError("predictive sequence time count must be a positive integer")
    binding_start_phase = assignment.binding_start_phase
    if binding_start_phase is None:
        raise ValueError("predictive materialization requires explicit causal binding phases")
    if (
        binding_start_phase.shape != assignment.row_to_track.shape
        or binding_start_phase.dtype != torch.long
        or binding_start_phase.device != assignment.row_to_track.device
        or (binding_start_phase < 0).any()
    ):
        raise ValueError("predictive binding phases must be non-negative long [batch,rows]")
    terminal_phase = 2 * sequence_time_count
    matched = assignment.row_to_track >= 0
    if (binding_start_phase[matched] >= terminal_phase).any() or (
        binding_start_phase[~matched] != terminal_phase
    ).any():
        raise ValueError("predictive assignment binding phases differ from the sequence")
    grouped: dict[str, list[ObjectiveTerm]] = {}
    for value in inputs:
        if not isinstance(value, NativePredictiveLossInput):
            raise TypeError("predictive inputs must use NativePredictiveLossInput")
        if value.target.track_identity_keys != expected_track_identity_keys:
            raise ValueError("predictive target track identities differ from structural targets")
        if value.identity_source_phase >= terminal_phase:
            raise ValueError("predictive identity source phase lies outside the sequence")
        row_binding_valid = binding_start_phase <= value.identity_source_phase
        term = native_predictive_term(
            prediction=value.prediction,
            request=value.request,
            target=value.target,
            assignment=assignment,
            row_binding_valid=row_binding_valid,
            weight=value.weight,
            loss_power=value.loss_power,
        )
        grouped.setdefault(term.name, []).append(term)

    merged: list[ObjectiveTerm] = []
    for name, terms in grouped.items():
        reference = terms[0]
        if any(term.weight != reference.weight for term in terms[1:]):
            raise ValueError(f"repeated predictive term {name!r} has inconsistent weights")
        if any(
            term.values.device != reference.values.device
            or term.values.dtype != reference.values.dtype
            for term in terms[1:]
        ):
            raise ValueError(f"repeated predictive term {name!r} has incompatible tensors")
        merged.append(
            ObjectiveTerm(
                name=name,
                values=torch.cat(tuple(term.values.reshape(-1) for term in terms)),
                valid=torch.cat(tuple(term.valid.reshape(-1) for term in terms)),
                weight=reference.weight,
            )
        )
    return tuple(merged)
