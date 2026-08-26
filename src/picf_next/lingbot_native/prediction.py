"""Leak-closed training-query and tokenizer dependency contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

import torch


class PredictionSource(str, Enum):
    PRIOR = "prior"
    POSTERIOR = "posterior"


class PredictionEvidence(str, Enum):
    CURRENT_RANDOM_GRID = "current_random_grid"
    CURRENT_CORRECTION = "current_correction"
    CURRENT_PRIOR = "current_prior"
    CURRENT_POSTERIOR = "current_posterior"
    PRIOR_ONLY = "prior_only"
    FUTURE = "future"
    OMITTED_MODALITY = "omitted_modality"


@dataclass(frozen=True, slots=True)
class NativePredictionRequest:
    """Source-known metadata replicated for every candidate row before matching."""

    source: PredictionSource
    evidence: PredictionEvidence
    route_ids: torch.Tensor
    horizons: torch.Tensor
    addresses: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.source, PredictionSource) or not isinstance(
            self.evidence, PredictionEvidence
        ):
            raise TypeError("prediction source and evidence must use frozen enums")
        if self.route_ids.ndim != 2 or self.route_ids.dtype != torch.long:
            raise ValueError("prediction route_ids must be long [batch, queries]")
        if self.horizons.shape != self.route_ids.shape or self.horizons.dtype != torch.long:
            raise ValueError("prediction horizons must be long and match route_ids")
        if self.addresses.ndim != 3 or self.addresses.shape[:2] != self.route_ids.shape:
            raise ValueError("prediction addresses must have shape [batch, queries, width]")
        if not self.addresses.is_floating_point() or not torch.isfinite(self.addresses).all():
            raise ValueError("prediction addresses must be finite floating point")
        if self.valid.shape != self.route_ids.shape or self.valid.dtype != torch.bool:
            raise ValueError("prediction validity must be boolean and match route_ids")
        tensors = (self.horizons, self.addresses, self.valid)
        if any(value.device != self.route_ids.device for value in tensors):
            raise ValueError("prediction request tensors must share one device")
        if (self.route_ids < 0).any() or (self.horizons < 0).any():
            raise ValueError("prediction route IDs and horizons must be non-negative")
        if self.evidence == PredictionEvidence.PRIOR_ONLY and self.source != PredictionSource.PRIOR:
            raise ValueError("prior-only evidence must query prior rows")
        if (
            self.evidence == PredictionEvidence.CURRENT_CORRECTION
            and self.source != PredictionSource.PRIOR
        ):
            raise ValueError("current-correction evidence must query prior rows")
        if (
            self.evidence == PredictionEvidence.CURRENT_PRIOR
            and self.source != PredictionSource.PRIOR
        ):
            raise ValueError("current-prior evidence must query prior rows")
        if (
            self.evidence == PredictionEvidence.CURRENT_POSTERIOR
            and self.source != PredictionSource.POSTERIOR
        ):
            raise ValueError("current-posterior evidence must query posterior rows")
        if self.evidence == PredictionEvidence.FUTURE and not (self.horizons > 0).all():
            raise ValueError("future evidence requires positive horizons")
        if self.evidence != PredictionEvidence.FUTURE and (self.horizons != 0).any():
            raise ValueError("non-future evidence requires zero horizon")
        if (
            self.evidence
            in {
                PredictionEvidence.CURRENT_CORRECTION,
                PredictionEvidence.CURRENT_PRIOR,
                PredictionEvidence.CURRENT_POSTERIOR,
                PredictionEvidence.OMITTED_MODALITY,
            }
            and torch.count_nonzero(self.addresses).item()
        ):
            raise ValueError("nonspatial correction and omission evidence require zero address")

    @property
    def batch_size(self) -> int:
        return self.route_ids.shape[0]

    @property
    def query_count(self) -> int:
        return self.route_ids.shape[1]

    @property
    def address_width(self) -> int:
        return self.addresses.shape[2]

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


def make_native_future_request(
    *,
    source: PredictionSource,
    batch_size: int,
    horizon: int,
    valid: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    route_id: int = 0,
    address_width: int = 0,
) -> NativePredictionRequest:
    """Create a source-known row-replicated future query."""

    integers = (batch_size, horizon, route_id, address_width)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("native future request dimensions must be integers")
    if batch_size <= 0 or horizon <= 0 or route_id < 0 or address_width < 0:
        raise ValueError("native future request dimensions are outside their valid range")
    if source not in (PredictionSource.POSTERIOR, PredictionSource.PRIOR):
        raise ValueError("native future request source must be posterior or prior")
    target_device = torch.device(device)
    if valid.shape != (batch_size,) or valid.dtype != torch.bool:
        raise ValueError("native future validity must be boolean [batch]")
    if valid.device != target_device:
        raise ValueError("native future validity and request must share one device")
    return NativePredictionRequest(
        source=source,
        evidence=PredictionEvidence.FUTURE,
        route_ids=torch.full(
            (batch_size, 1),
            route_id,
            dtype=torch.long,
            device=target_device,
        ),
        horizons=torch.full(
            (batch_size, 1),
            horizon,
            dtype=torch.long,
            device=target_device,
        ),
        addresses=torch.zeros(
            batch_size,
            1,
            address_width,
            dtype=dtype,
            device=target_device,
        ),
        valid=valid[:, None],
    )


@dataclass(frozen=True, slots=True)
class TokenizerDependencyMap:
    """Exact output-to-raw receptive-field incidence for one tokenizer."""

    output_depends_on_raw: torch.Tensor

    def __post_init__(self) -> None:
        value = self.output_depends_on_raw
        if value.ndim != 2 or value.dtype != torch.bool or min(value.shape) <= 0:
            raise ValueError("tokenizer dependency map must be boolean [outputs, raw_inputs]")
        if not value.any(dim=1).all():
            raise ValueError("every tokenizer output must declare at least one raw dependency")

    def source_output_valid(self, raw_target_mask: torch.Tensor) -> torch.Tensor:
        if (
            raw_target_mask.ndim != 2
            or raw_target_mask.shape[1] != self.output_depends_on_raw.shape[1]
        ):
            raise ValueError("raw target mask does not match tokenizer dependency width")
        if raw_target_mask.dtype != torch.bool:
            raise TypeError("raw target mask must be boolean")
        if raw_target_mask.device != self.output_depends_on_raw.device:
            raise ValueError("raw target mask and dependency map must share one device")
        overlap = torch.einsum(
            "br,or->bo",
            raw_target_mask.to(torch.int32),
            self.output_depends_on_raw.to(torch.int32),
        )
        return overlap == 0

    @property
    def digest(self) -> str:
        tensor = self.output_depends_on_raw.detach().cpu().contiguous()
        metadata = json.dumps(list(tensor.shape), separators=(",", ":")).encode()
        return hashlib.sha256(metadata + tensor.to(torch.uint8).numpy().tobytes()).hexdigest()
