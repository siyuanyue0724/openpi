"""JEPA-style target routing without deploy-time or self-copy leakage."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from picf_next.unified.objective import ObjectiveTerm
from picf_next.unified.state import UnifiedBeliefState

ROW_SUMMARY_TARGET = "row_summary"
DENSE_LATTICE_TARGET = "dense_lattice"
_TARGET_KINDS = frozenset((ROW_SUMMARY_TARGET, DENSE_LATTICE_TARGET))
PREDICTIVE_TARGET_CHECKPOINT_SCHEMA = "picf-next.predictive-target-checkpoint.v1"
PREDICTIVE_TARGET_PROVENANCE_SCHEMA = "picf-next.predictive-target-provenance.v1"


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def predictive_source_batch_digest(
    episode_keys: Sequence[str],
    frame_indices: Sequence[int],
) -> str:
    """Bind one predictive request to an exact ordered source batch.

    The digest is loss-side provenance only.  Episode IDs and frame indices are
    never embedded into the model, so they cannot become an identity shortcut.
    """

    if not episode_keys or len(episode_keys) != len(frame_indices):
        raise ValueError("predictive source batch metadata must be nonempty and aligned")
    if any(not isinstance(key, str) or not key for key in episode_keys):
        raise ValueError("predictive source episode keys must be nonempty strings")
    if any(
        isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
        for frame in frame_indices
    ):
        raise ValueError("predictive source frame indices must be non-negative integers")
    return _canonical_json_sha256(
        {
            "episode_keys": list(episode_keys),
            "frame_indices": list(frame_indices),
            "schema": "picf-next.predictive-source-batch.v1",
        }
    )


def _state_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError("predictive target state must contain dense tensors only")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Hash an exact tensor-state schema and payload independent of device."""

    return _state_mapping_sha256(module.state_dict())


@dataclass(frozen=True, slots=True)
class PredictionQueryRequest:
    """Only source-known metadata allowed to parameterize a prediction query."""

    modality: str
    target_kind: str
    horizon: int
    query_schema_digest: str
    source_batch_digest: str
    source_batch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.modality, str) or not self.modality:
            raise ValueError("prediction query modality must be non-empty")
        _sha256(self.query_schema_digest, "query_schema_digest")
        _sha256(self.source_batch_digest, "source_batch_digest")
        if self.target_kind not in _TARGET_KINDS:
            raise ValueError("prediction query target kind is unsupported")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("prediction query horizon must be an integer")
        if self.horizon < 0:
            raise ValueError("prediction query horizon must be non-negative")
        if (
            isinstance(self.source_batch_size, bool)
            or not isinstance(self.source_batch_size, int)
            or self.source_batch_size <= 0
        ):
            raise ValueError("prediction query source_batch_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class PredictiveTargetProvenance:
    """Immutable run/checkpoint identity for an EMA or cached target source."""

    modality: str
    target_kind: str
    target_data_digest: str
    target_model_digest: str
    assignment_schema_digest: str
    query_schema_digest: str
    validity_semantics: str
    optimizer_step: int
    schema: str = PREDICTIVE_TARGET_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PREDICTIVE_TARGET_PROVENANCE_SCHEMA:
            raise ValueError("predictive target provenance schema changed")
        if (
            not isinstance(self.modality, str)
            or not isinstance(self.validity_semantics, str)
            or not self.modality
            or not self.validity_semantics
        ):
            raise ValueError("predictive target provenance identifiers must be non-empty")
        if self.target_kind not in _TARGET_KINDS:
            raise ValueError("predictive target provenance kind is unsupported")
        for name in (
            "target_data_digest",
            "target_model_digest",
            "assignment_schema_digest",
            "query_schema_digest",
        ):
            _sha256(getattr(self, name), name)
        if (
            isinstance(self.optimizer_step, bool)
            or not isinstance(self.optimizer_step, int)
            or self.optimizer_step < 0
        ):
            raise ValueError("predictive target optimizer_step must be a non-negative integer")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema": self.schema,
            "modality": self.modality,
            "target_kind": self.target_kind,
            "target_data_digest": self.target_data_digest,
            "target_model_digest": self.target_model_digest,
            "assignment_schema_digest": self.assignment_schema_digest,
            "query_schema_digest": self.query_schema_digest,
            "validity_semantics": self.validity_semantics,
            "optimizer_step": self.optimizer_step,
        }

    @property
    def digest(self) -> str:
        return _canonical_json_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> PredictiveTargetProvenance:
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise ValueError("predictive target provenance must be a string-keyed mapping")
        fields = {
            "schema",
            "modality",
            "target_kind",
            "target_data_digest",
            "target_model_digest",
            "assignment_schema_digest",
            "query_schema_digest",
            "validity_semantics",
            "optimizer_step",
        }
        if set(value) != fields:
            raise ValueError("predictive target provenance fields differ from its frozen schema")
        return cls(
            schema=cast(str, value["schema"]),
            modality=cast(str, value["modality"]),
            target_kind=cast(str, value["target_kind"]),
            target_data_digest=cast(str, value["target_data_digest"]),
            target_model_digest=cast(str, value["target_model_digest"]),
            assignment_schema_digest=cast(str, value["assignment_schema_digest"]),
            query_schema_digest=cast(str, value["query_schema_digest"]),
            validity_semantics=cast(str, value["validity_semantics"]),
            optimizer_step=cast(int, value["optimizer_step"]),
        )

    def validate_target(self, target: PredictiveTarget) -> None:
        if target.provenance_digest != self.digest:
            raise ValueError("predictive target provenance digest differs")
        if target.modality != self.modality or target.target_kind != self.target_kind:
            raise ValueError("predictive target differs from checkpoint modality or kind")
        if target.encoder_digest != self.target_model_digest:
            raise ValueError("predictive target encoder differs from checkpoint target model")
        if target.target_data_digest != self.target_data_digest:
            raise ValueError("predictive target data manifest differs from checkpoint provenance")
        if target.assignment_digest != self.assignment_schema_digest:
            raise ValueError("predictive target assignment schema differs from provenance")
        if target.query_schema_digest != self.query_schema_digest:
            raise ValueError("predictive target query schema differs from checkpoint provenance")
        if target.validity_semantics != self.validity_semantics:
            raise ValueError(
                "predictive target validity semantics differ from checkpoint provenance"
            )


@dataclass(frozen=True, slots=True)
class PredictiveTarget:
    modality: str
    features: torch.Tensor
    valid: torch.Tensor
    importance: torch.Tensor
    horizon: int
    source_batch_digest: str
    target_data_digest: str
    encoder_digest: str
    target_kind: str
    assignment_digest: str
    query_schema_digest: str
    validity_semantics: str
    provenance_digest: str

    def __post_init__(self) -> None:
        identifiers = (self.modality, self.validity_semantics)
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("predictive target identifiers must be non-empty")
        for name in (
            "encoder_digest",
            "assignment_digest",
            "query_schema_digest",
            "provenance_digest",
            "source_batch_digest",
            "target_data_digest",
        ):
            _sha256(getattr(self, name), name)
        if self.target_kind not in _TARGET_KINDS:
            raise ValueError("predictive target kind is unsupported")
        if (
            self.features.ndim < 2
            or self.features.shape[-1] == 0
            or self.valid.shape != self.features.shape[:-1]
        ):
            raise ValueError("target validity must match every non-feature axis")
        if self.valid.dtype != torch.bool:
            raise TypeError("target validity must be boolean")
        if self.importance.shape != self.valid.shape or not self.importance.is_floating_point():
            raise ValueError("target importance must be floating point and match validity")
        if not self.features.is_floating_point() or not torch.isfinite(self.features).all():
            raise ValueError("predictive target features must be finite floating point")
        if (
            not torch.isfinite(self.importance).all()
            or ((self.importance < 0) | (self.importance > 1)).any()
        ):
            raise ValueError("target importance must be finite and lie in [0, 1]")
        if self.features.device != self.valid.device or self.importance.device != self.valid.device:
            raise ValueError("predictive target tensors must share one device")
        if any(
            value.requires_grad or value.grad_fn is not None
            for value in (self.features, self.importance)
        ):
            raise ValueError("predictive targets must be stop-gradient tensors")
        if not torch.equal(self.valid, self.importance > 0):
            raise ValueError("target validity must exactly identify positive importance")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("predictive target horizon must be an integer")
        if self.horizon < 0:
            raise ValueError("predictive target horizon must be non-negative")


@dataclass(frozen=True, slots=True)
class LeaveOneModalityOutRoute:
    context: dict[str, torch.Tensor]
    target: PredictiveTarget


@dataclass(frozen=True, slots=True)
class LatentCollapseDiagnostics:
    valid_count: int
    mean_variance: float
    effective_rank: float


@dataclass(frozen=True, slots=True)
class BeliefStateTarget:
    state: UnifiedBeliefState
    source_frame: int
    target_frame: int
    schema_digest: str
    model_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, UnifiedBeliefState):
            raise TypeError("future belief target state must be UnifiedBeliefState")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.source_frame, self.target_frame)
        ):
            raise TypeError("future belief target frames must be integers")
        if self.source_frame < 0 or self.target_frame <= self.source_frame:
            raise ValueError("future belief target frames are invalid")
        _sha256(self.schema_digest, "schema_digest")
        _sha256(self.model_digest, "model_digest")
        for field in self.state.__dataclass_fields__:
            value = getattr(self.state, field)
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError("future belief target state must be stop-gradient")


@torch.no_grad()
def initialize_ema_target_(target: nn.Module, online: nn.Module) -> None:
    """Initialize and freeze a small target stem from its online counterpart."""

    target_state = target.state_dict()
    online_state = online.state_dict()
    if target_state.keys() != online_state.keys():
        raise ValueError("EMA target and online stem state schemas differ")
    for name in target_state:
        if target_state[name].shape != online_state[name].shape:
            raise ValueError(f"EMA state shape differs: {name}")
        if target_state[name].dtype != online_state[name].dtype:
            raise ValueError(f"EMA state dtype differs: {name}")
    target.load_state_dict(online_state, strict=True)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()


@torch.no_grad()
def update_ema_target_(target: nn.Module, online: nn.Module, *, momentum: float) -> None:
    """Apply V-JEPA-style momentum update to an initialized target stem."""

    if isinstance(momentum, bool) or not isinstance(momentum, (int, float)):
        raise TypeError("EMA momentum must be real-valued")
    if not 0 <= momentum < 1 or not math.isfinite(momentum):
        raise ValueError("EMA momentum must be finite and lie in [0, 1)")
    if target.training:
        raise ValueError("EMA target stem must remain in evaluation mode")
    target_parameters = dict(target.named_parameters())
    online_parameters = dict(online.named_parameters())
    if target_parameters.keys() != online_parameters.keys():
        raise ValueError("EMA target and online parameter schemas differ")
    if any(parameter.requires_grad for parameter in target_parameters.values()):
        raise ValueError("EMA target parameters must remain frozen")
    for name, target_parameter in target_parameters.items():
        online_parameter = online_parameters[name]
        if target_parameter.shape != online_parameter.shape:
            raise ValueError(f"EMA parameter shape differs: {name}")
        if target_parameter.dtype != online_parameter.dtype:
            raise ValueError(f"EMA parameter dtype differs: {name}")
        target_parameter.mul_(momentum).add_(online_parameter, alpha=1.0 - momentum)

    target_buffers = dict(target.named_buffers())
    online_buffers = dict(online.named_buffers())
    if target_buffers.keys() != online_buffers.keys():
        raise ValueError("EMA target and online buffer schemas differ")
    for name, target_buffer in target_buffers.items():
        online_buffer = online_buffers[name]
        if target_buffer.shape != online_buffer.shape:
            raise ValueError(f"EMA buffer shape differs: {name}")
        if target_buffer.dtype != online_buffer.dtype:
            raise ValueError(f"EMA buffer dtype differs: {name}")
        target_buffer.copy_(online_buffer)


def predictive_target_checkpoint_payload(
    target: nn.Module,
    provenance: PredictiveTargetProvenance,
) -> dict[str, object]:
    """Create the exact loss-side target payload embedded in a training checkpoint."""

    if target.training or any(parameter.requires_grad for parameter in target.parameters()):
        raise ValueError("predictive target checkpoint requires a frozen evaluation-mode module")
    actual_digest = module_state_sha256(target)
    if actual_digest != provenance.target_model_digest:
        raise ValueError("predictive target module differs from its provenance digest")
    state = {name: value.detach().cpu().clone() for name, value in target.state_dict().items()}
    return {
        "schema": PREDICTIVE_TARGET_CHECKPOINT_SCHEMA,
        "provenance": provenance.to_dict(),
        "target": state,
    }


def restore_predictive_target_checkpoint_(
    target: nn.Module,
    payload: object,
    *,
    expected_provenance_digest: str,
) -> PredictiveTargetProvenance:
    """Restore a target module only when state and immutable provenance both match."""

    _sha256(expected_provenance_digest, "expected_provenance_digest")
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "provenance", "target"}:
        raise ValueError("predictive target checkpoint fields differ from its frozen schema")
    if payload["schema"] != PREDICTIVE_TARGET_CHECKPOINT_SCHEMA:
        raise ValueError("predictive target checkpoint schema changed")
    provenance = PredictiveTargetProvenance.from_dict(payload["provenance"])
    if provenance.digest != expected_provenance_digest:
        raise ValueError("predictive target checkpoint provenance digest differs")
    state = payload["target"]
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("predictive target checkpoint state is not a tensor mapping")
    target_state = target.state_dict()
    if set(state) != set(target_state):
        raise ValueError("predictive target checkpoint state keys differ")
    for name, reference in target_state.items():
        value = state[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"predictive target checkpoint tensor schema differs: {name}")
    state = cast(Mapping[str, torch.Tensor], state)
    if _state_mapping_sha256(state) != provenance.target_model_digest:
        raise ValueError("predictive target checkpoint tensor digest differs")
    target.load_state_dict(state, strict=True)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()
    if module_state_sha256(target) != provenance.target_model_digest:
        raise RuntimeError("restored predictive target module digest differs")
    return provenance


def make_predictive_target(
    modality: str,
    features: torch.Tensor,
    valid: torch.Tensor,
    *,
    importance: torch.Tensor | None = None,
    horizon: int,
    source_batch_digest: str,
    target_data_digest: str,
    encoder_digest: str,
    target_kind: str,
    assignment_digest: str,
    query_schema_digest: str,
    validity_semantics: str,
    provenance_digest: str,
) -> PredictiveTarget:
    detached_valid = valid.detach().clone()
    detached_importance = (
        detached_valid.to(dtype=torch.float32)
        if importance is None
        else importance.detach().float().clone()
    )
    return PredictiveTarget(
        modality=modality,
        features=features.detach().clone(),
        valid=detached_valid,
        importance=detached_importance,
        horizon=horizon,
        source_batch_digest=source_batch_digest,
        target_data_digest=target_data_digest,
        encoder_digest=encoder_digest,
        target_kind=target_kind,
        assignment_digest=assignment_digest,
        query_schema_digest=query_schema_digest,
        validity_semantics=validity_semantics,
        provenance_digest=provenance_digest,
    )


def make_row_predictive_target(
    modality: str,
    token_features: torch.Tensor,
    responsibilities: torch.Tensor,
    token_valid: torch.Tensor,
    token_footprint: torch.Tensor,
    *,
    horizon: int,
    source_batch_digest: str,
    target_data_digest: str,
    encoder_digest: str,
    assignment_digest: str,
    query_schema_digest: str,
    validity_semantics: str,
    provenance_digest: str,
    minimum_support: float = 0.0,
) -> PredictiveTarget:
    """Aggregate detached target-token latents by detached physical-row responsibility."""

    if token_features.ndim != 3:
        raise ValueError("token_features must have shape [batch, tokens, width]")
    if responsibilities.ndim != 3 or responsibilities.shape[:2] != token_features.shape[:2]:
        raise ValueError("responsibilities must have shape [batch, tokens, capacity]")
    if token_valid.shape != token_features.shape[:2] or token_valid.dtype != torch.bool:
        raise ValueError("token_valid must be boolean [batch, tokens]")
    if token_footprint.shape != token_valid.shape or not token_footprint.is_floating_point():
        raise ValueError("token_footprint must be floating point [batch, tokens]")
    if any(
        value.device != token_features.device
        for value in (responsibilities, token_valid, token_footprint)
    ):
        raise ValueError("row target tensors must share one device")
    if not token_features.is_floating_point() or not responsibilities.is_floating_point():
        raise TypeError("row target features and responsibilities must be floating point")
    if not torch.isfinite(token_features).all() or not torch.isfinite(responsibilities).all():
        raise ValueError("row target tensors must be finite")
    if not torch.isfinite(token_footprint).all() or (token_footprint < 0).any():
        raise ValueError("token_footprint must be finite and non-negative")
    if (responsibilities < 0).any():
        raise ValueError("row target responsibilities must be non-negative")
    if (responsibilities.float().sum(dim=-1) > 1.0 + 1e-5).any():
        raise ValueError("row target responsibilities must be a sub-probability simplex")
    if isinstance(minimum_support, bool) or not isinstance(minimum_support, (int, float)):
        raise TypeError("minimum_support must be real-valued")
    if not math.isfinite(minimum_support) or minimum_support < 0:
        raise ValueError("minimum_support must be finite and non-negative")

    detached_features = token_features.detach().float()
    valid_footprint = token_footprint.detach().float() * token_valid.detach()
    normalized_footprint = valid_footprint / valid_footprint.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    weights = responsibilities.detach().float() * normalized_footprint.unsqueeze(-1)
    support = weights.sum(dim=1)
    summaries = torch.einsum("bnk,bnd->bkd", weights, detached_features)
    summaries = summaries / support.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    valid = support > minimum_support
    summaries = summaries.masked_fill(~valid.unsqueeze(-1), 0)
    importance = support.masked_fill(~valid, 0)
    return make_predictive_target(
        modality,
        summaries,
        valid,
        importance=importance,
        horizon=horizon,
        source_batch_digest=source_batch_digest,
        target_data_digest=target_data_digest,
        encoder_digest=encoder_digest,
        target_kind=ROW_SUMMARY_TARGET,
        assignment_digest=assignment_digest,
        query_schema_digest=query_schema_digest,
        validity_semantics=validity_semantics,
        provenance_digest=provenance_digest,
    )


def make_belief_state_target(
    state: UnifiedBeliefState,
    *,
    source_frame: int,
    target_frame: int,
    schema_digest: str,
    model_digest: str,
) -> BeliefStateTarget:
    return BeliefStateTarget(
        state=state.detached(),
        source_frame=source_frame,
        target_frame=target_frame,
        schema_digest=schema_digest,
        model_digest=model_digest,
    )


def leave_one_modality_out(
    online_modalities: Mapping[str, torch.Tensor],
    target: PredictiveTarget,
) -> LeaveOneModalityOutRoute:
    if target.modality not in online_modalities:
        raise ValueError("withheld target modality is absent from the online modality set")
    context = {name: value for name, value in online_modalities.items() if name != target.modality}
    if not context:
        raise ValueError("leave-one-modality-out requires at least one context modality")
    return LeaveOneModalityOutRoute(context=context, target=target)


def predictive_target_loss(
    prediction: torch.Tensor,
    target: PredictiveTarget,
) -> torch.Tensor:
    return predictive_target_term(
        prediction,
        target,
        name=f"future/{target.modality}",
        weight=1.0,
    ).normalized()


def predictive_target_term(
    prediction: torch.Tensor,
    target: PredictiveTarget,
    *,
    name: str,
    weight: float,
) -> ObjectiveTerm:
    """Create a valid-count-normalized latent target term for the joint law."""

    if prediction.shape != target.features.shape:
        raise ValueError("prediction and predictive target shapes must match")
    if prediction.device != target.features.device:
        raise ValueError("prediction and predictive target must share one device")
    if not prediction.is_floating_point() or not torch.isfinite(prediction).all():
        raise ValueError("prediction must be finite floating point")
    per_token = (prediction.float() - target.features.float()).square().mean(dim=-1)
    valid_count = target.valid.sum().to(dtype=per_token.dtype)
    importance_mass = target.importance.masked_select(target.valid).sum().to(per_token)
    scale = valid_count / importance_mass.clamp_min(torch.finfo(per_token.dtype).tiny)
    per_token = per_token * target.importance.to(per_token) * scale
    return ObjectiveTerm(
        name=name,
        values=per_token,
        valid=target.valid,
        weight=weight,
    )


def row_conditioned_predictive_term(
    prediction: torch.Tensor,
    target: PredictiveTarget,
    request: PredictionQueryRequest,
    *,
    weight: float,
) -> ObjectiveTerm:
    """Compare one exchangeable prediction per belief row in normalized latent space."""

    if target.target_kind != ROW_SUMMARY_TARGET or request.target_kind != ROW_SUMMARY_TARGET:
        raise ValueError("row-conditioned prediction requires row-summary contracts")
    if target.modality != request.modality:
        raise ValueError("prediction request and target modalities differ")
    if target.query_schema_digest != request.query_schema_digest:
        raise ValueError("prediction request and target query schemas differ")
    if target.horizon != request.horizon:
        raise ValueError("prediction request and target horizons differ")
    if target.source_batch_digest != request.source_batch_digest:
        raise ValueError("prediction request and target source batches differ")
    if prediction.shape[0] != request.source_batch_size:
        raise ValueError("prediction request source batch size differs from row predictions")
    if prediction.shape != target.features.shape:
        raise ValueError("row prediction and target shapes must match")
    if prediction.shape[-1] < 2:
        raise ValueError("normalized row prediction requires at least two feature coordinates")
    normalized_prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    normalized_target = F.layer_norm(target.features.float(), (target.features.shape[-1],))
    family = "xmod" if request.horizon == 0 else "future"
    normalized = PredictiveTarget(
        modality=target.modality,
        features=normalized_target.detach(),
        valid=target.valid,
        importance=target.importance,
        horizon=target.horizon,
        source_batch_digest=target.source_batch_digest,
        target_data_digest=target.target_data_digest,
        encoder_digest=target.encoder_digest,
        target_kind=target.target_kind,
        assignment_digest=target.assignment_digest,
        query_schema_digest=target.query_schema_digest,
        validity_semantics=target.validity_semantics,
        provenance_digest=target.provenance_digest,
    )
    return predictive_target_term(
        normalized_prediction,
        normalized,
        name=f"{family}/{target.modality}",
        weight=weight,
    )


@torch.no_grad()
def latent_collapse_diagnostics(
    features: torch.Tensor,
    valid: torch.Tensor,
) -> LatentCollapseDiagnostics:
    """Measure variance and covariance-spectrum rank without adding a training loss."""

    if features.ndim < 2 or valid.shape != features.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("latent diagnostics require features [..., width] and matching validity")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("latent diagnostic features must be finite floating point")
    if features.device != valid.device:
        raise ValueError("latent diagnostic features and validity must share one device")
    selected = features.detach().float()[valid]
    count = int(selected.shape[0])
    if count < 2:
        return LatentCollapseDiagnostics(count, 0.0, 0.0)
    centered = selected - selected.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0).mean()
    singular_values = torch.linalg.svdvals(centered)
    spectrum = singular_values.square()
    probabilities = spectrum / spectrum.sum().clamp_min(torch.finfo(torch.float32).tiny)
    entropy = -(
        probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum()
    return LatentCollapseDiagnostics(
        valid_count=count,
        mean_variance=float(variance.item()),
        effective_rank=float(entropy.exp().item()),
    )


def belief_overshooting_term(
    prediction: UnifiedBeliefState,
    target: BeliefStateTarget,
    *,
    weight: float,
    lifecycle_scale: float = 1.0,
    content_scale: float = 1.0,
    geometry_scale: float = 1.0,
) -> ObjectiveTerm:
    """Compare future belief distributions without assuming fixed row order."""

    expected_schema = (
        target.state.batch_size,
        target.state.capacity,
        target.state.content_dim,
        target.state.geometry_dim,
        target.state.uncertainty_dim,
    )
    actual_schema = (
        prediction.batch_size,
        prediction.capacity,
        prediction.content_dim,
        prediction.geometry_dim,
        prediction.uncertainty_dim,
    )
    if actual_schema != expected_schema:
        raise ValueError("predicted and target belief schemas differ")
    if prediction.content.device != target.state.content.device:
        raise ValueError("predicted and target beliefs must share one device")
    scales = (lifecycle_scale, content_scale, geometry_scale)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in scales):
        raise TypeError("overshooting component scales must be real-valued")
    if any(value < 0 or not math.isfinite(value) for value in scales):
        raise ValueError("overshooting component scales must be finite and non-negative")

    losses = []
    for batch_index in range(prediction.batch_size):
        predicted_log_lifecycle = prediction.lifecycle_log_probs[batch_index]
        target_lifecycle = target.state.lifecycle_probs[batch_index]
        lifecycle_cost = torch.einsum(
            "tm,km->kt",
            target_lifecycle,
            -predicted_log_lifecycle,
        )
        content_cost = (
            (
                prediction.content[batch_index, :, None, :]
                - target.state.content[batch_index, None, :, :]
            )
            .float()
            .square()
            .mean(dim=-1)
        )
        geometry_cost = prediction.content.new_zeros((prediction.capacity, prediction.capacity))
        geometry_available = torch.zeros_like(geometry_cost, dtype=torch.bool)
        for predicted_row in range(prediction.capacity):
            for target_row in range(target.state.capacity):
                valid = (
                    prediction.geometry_valid[batch_index, predicted_row]
                    & (target.state.geometry_valid[batch_index, target_row])
                )
                if valid.any():
                    difference = (
                        prediction.geometry_mean[batch_index, predicted_row, valid]
                        - (target.state.geometry_mean[batch_index, target_row, valid])
                    )
                    geometry_cost[predicted_row, target_row] = difference.float().square().mean()
                    geometry_available[predicted_row, target_row] = True
        target_nonempty = target.state.nonempty_probability[batch_index]
        detached_cost = lifecycle_scale * lifecycle_cost + target_nonempty.unsqueeze(0) * (
            content_scale * content_cost + geometry_scale * geometry_cost
        )
        rows, columns = linear_sum_assignment(detached_cost.detach().cpu().numpy())
        row_index = torch.as_tensor(rows, device=prediction.content.device, dtype=torch.long)
        column_index = torch.as_tensor(columns, device=prediction.content.device, dtype=torch.long)
        matched_lifecycle = lifecycle_cost[row_index, column_index].mean()
        matched_weight = target_nonempty[column_index]
        nonempty_mass = matched_weight.sum().clamp_min(torch.finfo(torch.float32).tiny)
        matched_content = (
            matched_weight * content_cost[row_index, column_index]
        ).sum() / nonempty_mass
        matched_geometry_valid = geometry_available[row_index, column_index]
        geometry_weight = matched_weight * matched_geometry_valid.to(matched_weight.dtype)
        geometry_mass = geometry_weight.sum()
        matched_geometry = torch.where(
            geometry_mass > 0,
            (geometry_weight * geometry_cost[row_index, column_index]).sum()
            / geometry_mass.clamp_min(torch.finfo(torch.float32).tiny),
            geometry_cost.sum() * 0,
        )
        losses.append(
            lifecycle_scale * matched_lifecycle
            + content_scale * matched_content
            + geometry_scale * matched_geometry
        )
    value = torch.stack(losses).mean()
    return ObjectiveTerm(
        name="over/belief_set",
        values=value.reshape(1),
        valid=torch.ones(1, dtype=torch.bool, device=value.device),
        weight=weight,
    )
