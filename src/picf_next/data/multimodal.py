"""Collate versioned heterogeneous encoder outputs without token deletion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.models.evidence import NativeTokenBank


@dataclass(frozen=True, slots=True)
class ModalityBatchSpec:
    name: str
    encoder_contract: str
    token_dim: int
    geometry_dim: int = 0
    require_single_active_group: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.encoder_contract:
            raise ContractError("modality name and encoder contract must be explicit")
        if (
            not isinstance(self.token_dim, int)
            or isinstance(self.token_dim, bool)
            or self.token_dim <= 0
            or not isinstance(self.geometry_dim, int)
            or isinstance(self.geometry_dim, bool)
            or self.geometry_dim < 0
        ):
            raise ContractError("modality token dimensions are invalid")
        if not isinstance(self.require_single_active_group, bool):
            raise ContractError("require_single_active_group must be boolean")


def collate_dense_evidence(
    samples: Sequence[tuple[DenseEvidence, ...]],
    specs: tuple[ModalityBatchSpec, ...],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[NativeTokenBank, ...]:
    """Pad complete per-sample token banks while retaining every valid token.

    A modality absent from one sample is represented only by invalid zero
    padding. The function never fabricates a learned missing-modality token and
    never truncates a longer sample to match a shorter one.
    """

    if not samples or not specs:
        raise ValueError("multimodal collation requires samples and modality specs")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("native token banks require a floating dtype")
    if len({spec.name for spec in specs}) != len(specs):
        raise ContractError("modality batch specs must be unique")

    indexed_samples: list[dict[str, DenseEvidence]] = []
    allowed = {spec.name for spec in specs}
    for sample in samples:
        mapping = {evidence.modality: evidence for evidence in sample}
        if len(mapping) != len(sample):
            raise ContractError("a sample contains duplicate modality evidence")
        unknown = sorted(set(mapping).difference(allowed))
        if unknown:
            raise ContractError(f"sample contains unconfigured modalities: {unknown}")
        indexed_samples.append(mapping)

    banks: list[NativeTokenBank] = []
    batch_size = len(samples)
    for spec in specs:
        evidence_rows = [sample.get(spec.name) for sample in indexed_samples]
        max_tokens = max((row.token_count for row in evidence_rows if row is not None), default=0)
        tokens = torch.zeros((batch_size, max_tokens, spec.token_dim), dtype=dtype, device=device)
        valid = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)
        current_measurement_valid = torch.zeros(
            (batch_size, max_tokens), dtype=torch.bool, device=device
        )
        timestamps = torch.zeros((batch_size, max_tokens), dtype=torch.float32, device=device)
        confidence = torch.zeros((batch_size, max_tokens), dtype=dtype, device=device)
        geometry = (
            torch.zeros((batch_size, max_tokens, spec.geometry_dim), dtype=dtype, device=device)
            if spec.geometry_dim
            else None
        )
        group_id = (
            torch.full((batch_size, max_tokens), -1, dtype=torch.long, device=device)
            if spec.require_single_active_group
            else None
        )

        for batch_index, evidence in enumerate(evidence_rows):
            if evidence is None:
                continue
            if evidence.encoder_contract != spec.encoder_contract:
                raise ContractError(
                    f"{spec.name} encoder contract differs from the frozen batch spec"
                )
            if evidence.tokens.shape[1] != spec.token_dim:
                raise ContractError(f"{spec.name} token width differs from the batch spec")
            if not evidence.available and evidence.token_count:
                raise ContractError(f"missing {spec.name} evidence emitted tokens")
            if spec.geometry_dim:
                if evidence.geometry is None or evidence.geometry.shape[1] != spec.geometry_dim:
                    raise ContractError(f"{spec.name} geometry differs from the batch spec")
            elif evidence.geometry is not None:
                raise ContractError(f"{spec.name} supplied unconfigured geometry")
            if spec.require_single_active_group:
                if evidence.token_count and evidence.group_ids is None:
                    raise ContractError(f"active {spec.name} evidence requires one group")
                if evidence.group_ids is not None and evidence.token_count:
                    groups = np.unique(evidence.group_ids)
                    if groups.size != 1 or int(groups[0]) < 0:
                        raise ContractError(f"all active {spec.name} tokens must share one group")
            elif evidence.group_ids is not None:
                raise ContractError(f"{spec.name} does not permit runtime groups")

            count = evidence.token_count
            if not count:
                continue
            # Encoder contracts expose immutable NumPy arrays. ``as_tensor``
            # would alias that read-only storage and PyTorch explicitly marks
            # later writes as undefined, even though this assignment only
            # reads it. ``tensor`` makes the ownership transfer explicit.
            tokens[batch_index, :count] = torch.tensor(
                np.asarray(evidence.tokens), dtype=dtype, device=device
            )
            valid[batch_index, :count] = True
            current_measurement_valid[batch_index, :count] = torch.tensor(
                np.asarray(evidence.effective_current_measurement_valid),
                dtype=torch.bool,
                device=device,
            )
            timestamps[batch_index, :count] = torch.tensor(
                np.asarray(evidence.timestamps), dtype=torch.float32, device=device
            )
            confidence[batch_index, :count] = torch.tensor(
                np.asarray(evidence.confidence), dtype=dtype, device=device
            )
            if geometry is not None and evidence.geometry is not None:
                geometry[batch_index, :count] = torch.tensor(
                    np.asarray(evidence.geometry), dtype=dtype, device=device
                )
            if group_id is not None and evidence.group_ids is not None:
                group_id[batch_index, :count] = torch.tensor(
                    np.asarray(evidence.group_ids), dtype=torch.long, device=device
                )

        banks.append(
            NativeTokenBank(
                modality=spec.name,
                tokens=tokens,
                valid=valid,
                geometry=geometry,
                group_id=group_id,
                timestamps=timestamps,
                confidence=confidence,
                encoder_contract=spec.encoder_contract,
                current_measurement_valid=current_measurement_valid,
            )
        )
    return tuple(banks)
