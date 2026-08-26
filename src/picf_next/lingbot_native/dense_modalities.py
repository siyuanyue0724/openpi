"""Lossless audited bridge from encoder evidence to shared-host token streams."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.lingbot_native.modalities import (
    NativeModalityBatch,
    NativeModalitySpec,
    NativeModalityStream,
)


@dataclass(frozen=True, slots=True)
class NativeDenseModalityBinding:
    """Frozen upstream boundary and token budget for one optional modality.

    The bridge owns no semantic decision. It retains every selected upstream
    token and exposes only typed geometry, relative age, encoder confidence,
    and current-measurement role as metadata for a linear shared-host adapter.
    """

    name: str
    encoder_contract: str
    token_width: int
    maximum_tokens: int
    geometry_width: int = 0

    def __post_init__(self) -> None:
        NativeModalitySpec(
            name=self.name,
            input_width=self.token_width,
            maximum_tokens=self.maximum_tokens,
            metadata_width=self.metadata_width,
        )
        if not isinstance(self.encoder_contract, str) or not self.encoder_contract.strip():
            raise ValueError("dense modality encoder contract must be nonempty")
        if (
            isinstance(self.geometry_width, bool)
            or not isinstance(self.geometry_width, int)
            or self.geometry_width < 0
        ):
            raise ValueError("dense modality geometry width must be nonnegative")

    @property
    def metadata_width(self) -> int:
        # Geometry plus log relative age, source confidence and current role.
        return self.geometry_width + 3

    @property
    def native_spec(self) -> NativeModalitySpec:
        return NativeModalitySpec(
            name=self.name,
            input_width=self.token_width,
            maximum_tokens=self.maximum_tokens,
            metadata_width=self.metadata_width,
        )

def dense_modality_bindings_sha256(
    bindings: tuple[NativeDenseModalityBinding, ...],
) -> str:
    """Hash the complete static evidence-to-host ABI."""

    _validate_bindings(bindings)
    payload = [
        {
            "encoder_contract": binding.encoder_contract,
            "geometry_width": binding.geometry_width,
            "maximum_tokens": binding.maximum_tokens,
            "name": binding.name,
            "token_width": binding.token_width,
        }
        for binding in bindings
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_bindings(bindings: tuple[NativeDenseModalityBinding, ...]) -> None:
    if not isinstance(bindings, tuple) or not bindings or any(
        not isinstance(binding, NativeDenseModalityBinding) for binding in bindings
    ):
        raise TypeError("dense modality bindings must be one nonempty typed tuple")
    names = tuple(binding.name for binding in bindings)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("dense modality bindings must be sorted with unique names")


def native_modalities_from_dense_evidence(
    samples: Sequence[tuple[DenseEvidence, ...]],
    bindings: tuple[NativeDenseModalityBinding, ...],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> NativeModalityBatch:
    """Collate complete dense evidence without token selection or fake inputs.

    Metadata is not a side prediction. For every valid token it contains
    ``[geometry..., log1p(newest_timestamp - timestamp), confidence,
    current_measurement_valid]``. Missing and padded rows remain invalid and
    all-zero. The shared LingBot host alone learns how this evidence relates to
    objects, language, posterior state and action.
    """

    _validate_bindings(bindings)
    if not samples:
        raise ValueError("dense modality collation requires at least one sample")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("native dense modality tokens require a floating dtype")
    allowed = {binding.name for binding in bindings}
    indexed_samples: list[dict[str, DenseEvidence]] = []
    for sample in samples:
        mapping = {evidence.modality: evidence for evidence in sample}
        if len(mapping) != len(sample):
            raise ContractError("a sample contains duplicate dense modality evidence")
        unknown = sorted(set(mapping).difference(allowed))
        if unknown:
            raise ContractError(f"sample contains unconfigured modalities: {unknown}")
        indexed_samples.append(mapping)

    target_device = torch.device(device)
    streams: list[NativeModalityStream] = []
    for binding in bindings:
        evidence_rows = [sample.get(binding.name) for sample in indexed_samples]
        maximum = max(
            (evidence.token_count for evidence in evidence_rows if evidence is not None),
            default=0,
        )
        if maximum > binding.maximum_tokens:
            raise ValueError(
                f"modality {binding.name!r} exceeds its frozen token budget"
            )
        tokens = torch.zeros(
            len(samples),
            maximum,
            binding.token_width,
            dtype=dtype,
            device=target_device,
        )
        valid = torch.zeros(
            len(samples), maximum, dtype=torch.bool, device=target_device
        )
        canonical_token_ids = torch.full(
            (len(samples), maximum),
            -1,
            dtype=torch.long,
            device=target_device,
        )
        metadata = torch.zeros(
            len(samples),
            maximum,
            binding.metadata_width,
            dtype=dtype,
            device=target_device,
        )
        for batch_index, evidence in enumerate(evidence_rows):
            if evidence is None:
                continue
            if evidence.encoder_contract != binding.encoder_contract:
                raise ContractError(
                    f"{binding.name} encoder contract differs from its frozen binding"
                )
            if evidence.tokens.shape[1] != binding.token_width:
                raise ContractError(
                    f"{binding.name} token width differs from its frozen binding"
                )
            if not evidence.available and evidence.token_count:
                raise ContractError(f"missing {binding.name} evidence emitted tokens")
            if binding.geometry_width:
                if (
                    evidence.geometry is None
                    or evidence.geometry.shape[1] != binding.geometry_width
                ):
                    raise ContractError(
                        f"{binding.name} geometry differs from its frozen binding"
                    )
            elif evidence.geometry is not None:
                raise ContractError(f"{binding.name} supplied unconfigured geometry")
            count = evidence.token_count
            if count == 0:
                continue
            tokens[batch_index, :count] = torch.tensor(
                evidence.tokens,
                dtype=dtype,
                device=target_device,
            )
            valid[batch_index, :count] = True
            canonical_token_ids[batch_index, :count] = torch.arange(
                count,
                dtype=torch.long,
                device=target_device,
            )
            columns: list[torch.Tensor] = []
            if evidence.geometry is not None:
                columns.append(
                    torch.tensor(evidence.geometry, dtype=dtype, device=target_device)
                )
            timestamps = torch.tensor(
                evidence.timestamps,
                dtype=torch.float32,
                device=target_device,
            )
            columns.extend(
                (
                    torch.log1p((timestamps.max() - timestamps).clamp_min(0))
                    .to(dtype)
                    .unsqueeze(-1),
                    torch.tensor(
                        evidence.confidence,
                        dtype=dtype,
                        device=target_device,
                    ).unsqueeze(-1),
                    torch.tensor(
                        evidence.effective_current_measurement_valid,
                        dtype=dtype,
                        device=target_device,
                    ).unsqueeze(-1),
                )
            )
            metadata[batch_index, :count] = torch.cat(columns, dim=-1)
        streams.append(
            NativeModalityStream(
                name=binding.name,
                tokens=tokens,
                valid=valid,
                metadata=metadata,
                canonical_token_ids=canonical_token_ids,
            )
        )
    batch = NativeModalityBatch(tuple(streams))
    batch.validate_against(tuple(binding.native_spec for binding in bindings))
    return batch
