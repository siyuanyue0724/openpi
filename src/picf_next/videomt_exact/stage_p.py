"""Stage-PQ/PQM execution with the complete released VidEoMT donor graph.

Stage-PQ is the historical query-only arm. Stage-PQM preserves all queries and
the complete class/mask relation, then lets the full LingBot host assign donor
queries to persistent posterior rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from picf_next.lingbot_native.modalities import (
    NativeModalityBatch,
    NativeModalitySpec,
    NativeObjectQuerySpatialSpec,
    merge_native_modality_batches,
    validate_modality_specs,
)
from picf_next.videomt_exact.observations import (
    VIDEOMT_QUERY_MODALITY,
    VIDEOMT_STAGE_PQ_INTERFACE,
    VIDEOMT_STAGE_PQM_INTERFACE,
    VIDEOMT_STAGE_PQMR_INTERFACE,
    VidEoMTQueryObservation,
    videomt_calvin_object_query_spatial_spec,
    videomt_query_modality_spec,
    videomt_row_mask_query_modality_spec,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput, ExactVidEoMTRuntime


def with_videomt_query_modality_spec(
    specs: tuple[NativeModalitySpec, ...],
) -> tuple[NativeModalitySpec, ...]:
    """Extend one frozen LingBot modality ABI without changing existing streams."""

    validate_modality_specs(specs)
    if VIDEOMT_QUERY_MODALITY in {spec.name for spec in specs}:
        raise ValueError("LingBot modality ABI already declares VidEoMT queries")
    result = tuple(sorted((*specs, videomt_query_modality_spec()), key=lambda spec: spec.name))
    validate_modality_specs(result)
    return result


def with_videomt_row_mask_query_modality_spec(
    specs: tuple[NativeModalitySpec, ...],
) -> tuple[NativeModalitySpec, ...]:
    """Extend LingBot with the complete query plus mask-embedding ABI."""

    validate_modality_specs(specs)
    if VIDEOMT_QUERY_MODALITY in {spec.name for spec in specs}:
        raise ValueError("LingBot modality ABI already declares VidEoMT queries")
    result = tuple(
        sorted((*specs, videomt_row_mask_query_modality_spec()), key=lambda spec: spec.name)
    )
    validate_modality_specs(result)
    return result


@dataclass(frozen=True, slots=True)
class VidEoMTStagePResult:
    upstream: ExactVidEoMTOutput
    observation: VidEoMTQueryObservation
    modalities: NativeModalityBatch

    @property
    def interface_identity(self) -> str:
        return VIDEOMT_STAGE_PQ_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.upstream, ExactVidEoMTOutput):
            raise TypeError("Stage-P result requires exact upstream outputs")
        if not isinstance(self.observation, VidEoMTQueryObservation):
            raise TypeError("Stage-P result requires typed query observations")
        if not isinstance(self.modalities, NativeModalityBatch):
            raise TypeError("Stage-P result requires one typed LingBot modality batch")
        if VIDEOMT_QUERY_MODALITY not in {stream.name for stream in self.modalities.streams}:
            raise ValueError("Stage-P result omitted the VidEoMT query stream")


class VidEoMTStageP(nn.Module):
    """One-way query adapter with no task, mask, or lifecycle subnetwork."""

    def __init__(self, runtime: ExactVidEoMTRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    def forward(
        self,
        normalized_padded_rgb: torch.Tensor,
        *,
        existing_modalities: NativeModalityBatch | None = None,
        host_dtype: torch.dtype | None = None,
        resume: bool = False,
    ) -> VidEoMTStagePResult:
        upstream = self.runtime(normalized_padded_rgb, resume=resume)
        observation = VidEoMTQueryObservation.from_exact_output(upstream)
        query_batch = observation.as_native_modality_batch(dtype=host_dtype)
        modalities = (
            query_batch
            if existing_modalities is None
            else merge_native_modality_batches((existing_modalities, query_batch))
        )
        return VidEoMTStagePResult(
            upstream=upstream,
            observation=observation,
            modalities=modalities,
        )


@dataclass(frozen=True, slots=True)
class VidEoMTStagePQMResult:
    """Complete mature donor outputs bound to the shared-host P-QM ABI."""

    upstream: ExactVidEoMTOutput
    observation: VidEoMTQueryObservation
    modalities: NativeModalityBatch

    @property
    def interface_identity(self) -> str:
        return VIDEOMT_STAGE_PQM_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.upstream, ExactVidEoMTOutput):
            raise TypeError("Stage-PQM result requires exact upstream outputs")
        if not isinstance(self.observation, VidEoMTQueryObservation):
            raise TypeError("Stage-PQM result requires typed query-mask observations")
        if not isinstance(self.modalities, NativeModalityBatch):
            raise TypeError("Stage-PQM result requires one typed LingBot modality batch")
        if len(self.modalities.object_query_spatial_relations) != 1:
            raise ValueError("Stage-PQM result must retain one complete query-mask relation")


class VidEoMTStagePQM(nn.Module):
    """Complete VidEoMT query-mask adapter; no donor component is replaced."""

    def __init__(
        self,
        runtime: ExactVidEoMTRuntime,
        relation_spec: NativeObjectQuerySpatialSpec | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.relation_spec = (
            videomt_calvin_object_query_spatial_spec()
            if relation_spec is None
            else relation_spec
        )

    def forward(
        self,
        normalized_padded_rgb: torch.Tensor,
        *,
        existing_modalities: NativeModalityBatch | None = None,
        host_dtype: torch.dtype | None = None,
        resume: bool = False,
    ) -> VidEoMTStagePQMResult:
        upstream = self.runtime(normalized_padded_rgb, resume=resume)
        observation = VidEoMTQueryObservation.from_exact_output(upstream)
        query_mask_batch = observation.as_native_pqm_batch(
            relation_spec=self.relation_spec,
            dtype=host_dtype,
        )
        modalities = (
            query_mask_batch
            if existing_modalities is None
            else merge_native_modality_batches((existing_modalities, query_mask_batch))
        )
        return VidEoMTStagePQMResult(
            upstream=upstream,
            observation=observation,
            modalities=modalities,
        )


@dataclass(frozen=True, slots=True)
class VidEoMTStagePQMRResult:
    """Complete donor outputs with a source-faithful posterior-row mask basis."""

    upstream: ExactVidEoMTOutput
    observation: VidEoMTQueryObservation
    modalities: NativeModalityBatch

    @property
    def interface_identity(self) -> str:
        return VIDEOMT_STAGE_PQMR_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.upstream, ExactVidEoMTOutput):
            raise TypeError("Stage-PQMR result requires exact upstream outputs")
        if not isinstance(self.observation, VidEoMTQueryObservation):
            raise TypeError("Stage-PQMR result requires typed decoder observations")
        if not isinstance(self.modalities, NativeModalityBatch):
            raise TypeError("Stage-PQMR result requires one typed LingBot modality batch")
        if len(self.modalities.object_query_spatial_relations) != 1:
            raise ValueError("Stage-PQMR result must retain one complete spatial relation")
        relation = self.modalities.object_query_spatial_relations[0]
        if relation.dense_mask_features is None:
            raise ValueError("Stage-PQMR result omitted the released dense mask feature")


class VidEoMTStagePQMR(nn.Module):
    """Source-faithful donor basis for direct contextual posterior-row masks."""

    def __init__(
        self,
        runtime: ExactVidEoMTRuntime,
        relation_spec: NativeObjectQuerySpatialSpec | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.relation_spec = (
            videomt_calvin_object_query_spatial_spec()
            if relation_spec is None
            else relation_spec
        )

    def forward(
        self,
        normalized_padded_rgb: torch.Tensor,
        *,
        existing_modalities: NativeModalityBatch | None = None,
        host_dtype: torch.dtype | None = None,
        resume: bool = False,
    ) -> VidEoMTStagePQMRResult:
        upstream = self.runtime(normalized_padded_rgb, resume=resume)
        observation = VidEoMTQueryObservation.from_exact_output(upstream)
        row_mask_batch = observation.as_native_row_mask_batch(
            relation_spec=self.relation_spec,
            dtype=host_dtype,
        )
        modalities = (
            row_mask_batch
            if existing_modalities is None
            else merge_native_modality_batches((existing_modalities, row_mask_batch))
        )
        return VidEoMTStagePQMRResult(
            upstream=upstream,
            observation=observation,
            modalities=modalities,
        )
