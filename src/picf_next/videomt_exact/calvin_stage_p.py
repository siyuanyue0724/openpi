"""Strict causal CALVIN adapter for the complete VidEoMT Stage-PQ donor."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinPICFEvidenceFrame
from picf_next.lingbot_native.modalities import (
    NativeModalityBatch,
    NativeModalityStream,
)
from picf_next.videomt_exact.observations import VIDEOMT_QUERY_MODALITY
from picf_next.videomt_exact.preprocessing import PreparedVidEoMTFrames, prepare_rgb_frames
from picf_next.videomt_exact.runtime import (
    VIDEOMT_DINOV3_L_QUERIES,
    VIDEOMT_DINOV3_L_WIDTH,
    ExactVidEoMTOutput,
)

VIDEOMT_STAGE_PQ_C5_INTERFACE = "videomt-stage-pq-causal-five-real-frames/v1"
VIDEOMT_STAGE_PQ_QUERY_HOST_OUTPUT = "query_embeddings.latest_all_200"
VIDEOMT_STAGE_PQM_HOST_OUTPUT = "query_embeddings.latest_all_200+class_mask_relation.latest_full"
VIDEOMT_STAGE_PQMR_HOST_OUTPUT = (
    "query_embeddings+mask_embeddings.latest_all_200+dense_mask_features.latest_full"
)
VIDEOMT_STAGE_PQRF_HOST_OUTPUT = (
    "query_embeddings+complete_segmenter_boundary.latest_all_200+prefixes+patches"
)
VIDEOMT_CAUSAL_FRAME_COUNT = 5
CALVIN_STATIC_RGB_KEY = "observation.images.rgb_static"


class CalvinRawEvidenceIndex(Protocol):
    def source_picf_evidence_prefix(
        self,
        global_index: int,
        *,
        maximum_source_frames: int,
    ) -> tuple[CalvinPICFEvidenceFrame, ...]: ...


class InsufficientCausalPrefixError(ContractError):
    """The current frame is valid but has fewer than four real predecessors."""


@dataclass(frozen=True, slots=True)
class PreparedCalvinStagePQInput:
    """Five real causal source frames after released evaluation preprocessing."""

    current_source_global_index: int
    source_global_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    source_rgb_sha256s: tuple[str, ...]
    frames: PreparedVidEoMTFrames
    interface_identity: str = VIDEOMT_STAGE_PQ_C5_INTERFACE

    def __post_init__(self) -> None:
        expected_indices = tuple(
            range(
                self.current_source_global_index - VIDEOMT_CAUSAL_FRAME_COUNT + 1,
                self.current_source_global_index + 1,
            )
        )
        if self.source_global_indices != expected_indices:
            raise ContractError("Stage P-Q-C5 source indices are not the exact causal suffix")
        if len(self.timestamps_s) != VIDEOMT_CAUSAL_FRAME_COUNT:
            raise ContractError("Stage P-Q-C5 timestamp count changed")
        if (
            len(self.source_rgb_sha256s) != VIDEOMT_CAUSAL_FRAME_COUNT
            or any(
                len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.source_rgb_sha256s
            )
        ):
            raise ContractError("Stage P-Q-C5 source RGB hashes are invalid")
        if self.frames.model_input.shape[0] != VIDEOMT_CAUSAL_FRAME_COUNT:
            raise ContractError("Stage P-Q-C5 preprocessing changed the frame count")


@dataclass(frozen=True, slots=True)
class VidEoMTQueryPrecisionReceipt:
    """Measured finite-precision error for one complete query bank cast."""

    source_dtype: str
    target_dtype: str
    shape: tuple[int, int, int]
    maximum_absolute_error: float
    relative_l2_error: float
    minimum_query_cosine: float
    exact_value_fraction: float
    induced_row_collision_count: int


@dataclass(frozen=True, slots=True)
class VidEoMTStagePQExecutionReceipt:
    """One real execution receipt covering every released output surface."""

    interface_identity: str
    current_source_global_index: int
    source_global_indices: tuple[int, ...]
    source_rgb_sha256s: tuple[str, ...]
    model_input_shape: tuple[int, int, int, int]
    class_logits_shape: tuple[int, int, int, int]
    mask_logits_shape: tuple[int, int, int, int, int]
    query_embeddings_shape: tuple[int, int, int, int]
    propagated_queries_shape: tuple[int, int, int]
    auxiliary_output_count: int
    host_injected_output: str
    precision_cast: VidEoMTQueryPrecisionReceipt

    def __post_init__(self) -> None:
        if self.interface_identity != VIDEOMT_STAGE_PQ_C5_INTERFACE:
            raise ContractError("VidEoMT execution receipt names another interface")
        if self.model_input_shape[0] != VIDEOMT_CAUSAL_FRAME_COUNT:
            raise ContractError("VidEoMT execution receipt changed the causal frame count")
        if len(self.source_rgb_sha256s) != VIDEOMT_CAUSAL_FRAME_COUNT:
            raise ContractError("VidEoMT execution receipt lost source RGB hashes")
        if self.class_logits_shape[:3] != (
            1,
            VIDEOMT_CAUSAL_FRAME_COUNT,
            VIDEOMT_DINOV3_L_QUERIES,
        ):
            raise ContractError("VidEoMT execution receipt changed class-output axes")
        if self.mask_logits_shape[:3] != (
            1,
            VIDEOMT_DINOV3_L_QUERIES,
            VIDEOMT_CAUSAL_FRAME_COUNT,
        ):
            raise ContractError("VidEoMT execution receipt changed mask-output axes")
        if self.query_embeddings_shape != (
            1,
            VIDEOMT_CAUSAL_FRAME_COUNT,
            VIDEOMT_DINOV3_L_QUERIES,
            VIDEOMT_DINOV3_L_WIDTH,
        ):
            raise ContractError("VidEoMT execution receipt changed query-output axes")
        if self.propagated_queries_shape != (
            1,
            VIDEOMT_DINOV3_L_QUERIES,
            VIDEOMT_DINOV3_L_WIDTH,
        ):
            raise ContractError("VidEoMT execution receipt changed propagated-state axes")
        if self.host_injected_output not in {
            VIDEOMT_STAGE_PQ_QUERY_HOST_OUTPUT,
            VIDEOMT_STAGE_PQM_HOST_OUTPUT,
            VIDEOMT_STAGE_PQMR_HOST_OUTPUT,
            VIDEOMT_STAGE_PQRF_HOST_OUTPUT,
        }:
            raise ContractError("VidEoMT execution receipt changed the Stage-PQ host boundary")


def empty_videomt_query_modality_batch(
    *,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    include_mask_metadata: bool = False,
) -> NativeModalityBatch:
    """Represent an ineligible short prefix as absence, never padded evidence."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("empty VidEoMT modality batch size must be positive")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("empty VidEoMT modality requires a supported host dtype")
    target = torch.device(device)
    return NativeModalityBatch(
        (
            NativeModalityStream(
                name=VIDEOMT_QUERY_MODALITY,
                tokens=torch.empty(
                    batch_size,
                    0,
                    VIDEOMT_DINOV3_L_WIDTH,
                    device=target,
                    dtype=dtype,
                ),
                metadata=(
                    torch.empty(
                        batch_size,
                        0,
                        VIDEOMT_DINOV3_L_WIDTH,
                        device=target,
                        dtype=dtype,
                    )
                    if include_mask_metadata
                    else None
                ),
                valid=torch.empty(batch_size, 0, device=target, dtype=torch.bool),
                canonical_token_ids=torch.empty(
                    batch_size,
                    0,
                    device=target,
                    dtype=torch.long,
                ),
            ),
        )
    )


def make_videomt_stage_pq_execution_receipt(
    prepared: PreparedCalvinStagePQInput,
    output: ExactVidEoMTOutput,
    *,
    host_dtype: torch.dtype,
    host_injected_output: str = VIDEOMT_STAGE_PQ_QUERY_HOST_OUTPUT,
) -> VidEoMTStagePQExecutionReceipt:
    """Bind one causal input to full donor outputs and the host precision cast."""

    if not isinstance(prepared, PreparedCalvinStagePQInput):
        raise TypeError("VidEoMT execution receipt requires one prepared CALVIN clip")
    if not isinstance(output, ExactVidEoMTOutput):
        raise TypeError("VidEoMT execution receipt requires exact typed donor outputs")
    precision = measure_videomt_query_precision_cast(
        output.query_embeddings[:, -1],
        host_dtype,
    )
    return VidEoMTStagePQExecutionReceipt(
        interface_identity=prepared.interface_identity,
        current_source_global_index=prepared.current_source_global_index,
        source_global_indices=prepared.source_global_indices,
        source_rgb_sha256s=prepared.source_rgb_sha256s,
        model_input_shape=tuple(int(value) for value in prepared.frames.model_input.shape),
        class_logits_shape=tuple(int(value) for value in output.class_logits.shape),
        mask_logits_shape=tuple(int(value) for value in output.mask_logits.shape),
        query_embeddings_shape=tuple(int(value) for value in output.query_embeddings.shape),
        propagated_queries_shape=tuple(int(value) for value in output.propagated_queries.shape),
        auxiliary_output_count=len(output.auxiliary_outputs),
        host_injected_output=host_injected_output,
        precision_cast=precision,
    )


def _static_rgb(frame: CalvinPICFEvidenceFrame) -> np.ndarray:
    matches = tuple(
        observation.value
        for observation in frame.sensor_observations
        if observation.key == CALVIN_STATIC_RGB_KEY
    )
    if len(matches) != 1:
        raise ContractError("CALVIN evidence frame must contain exactly one static RGB sensor")
    value = matches[0]
    if not isinstance(value, np.ndarray) or value.shape != (200, 200, 3):
        raise ContractError("CALVIN static RGB geometry changed")
    if value.dtype != np.uint8:
        raise ContractError("CALVIN static RGB dtype changed")
    return value


def _validate_real_contiguous_prefix(
    frames: Sequence[CalvinPICFEvidenceFrame],
) -> tuple[float, ...]:
    if len(frames) != VIDEOMT_CAUSAL_FRAME_COUNT:
        raise InsufficientCausalPrefixError(
            "Stage P-Q-C5 requires five real predecessor/current frames; padding is forbidden"
        )
    timestamps = tuple(float(frame.timestamp_s) for frame in frames)
    for index in range(1, len(frames)):
        previous = frames[index - 1]
        current = frames[index]
        if not math.isclose(
            current.timestamp_s - previous.timestamp_s,
            current.delta_t_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ContractError("Stage P-Q-C5 frames are not temporally contiguous")
        if not math.isclose(
            previous.delta_t_s,
            current.delta_t_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ContractError("Stage P-Q-C5 control frequency changed inside the clip")
    return timestamps


def prepare_calvin_stage_pq_c5(
    index: CalvinRawEvidenceIndex,
    current_source_global_index: int,
) -> PreparedCalvinStagePQInput:
    """Map one current source index to five causal frames without labels or padding."""

    if isinstance(current_source_global_index, bool) or not isinstance(
        current_source_global_index, int
    ):
        raise TypeError("current CALVIN source index must be an integer")
    resolve = getattr(index, "source_picf_evidence_prefix", None)
    if not callable(resolve):
        raise TypeError("Stage P-Q-C5 requires the raw CALVIN evidence-index contract")
    prefix = tuple(
        resolve(
            current_source_global_index,
            maximum_source_frames=VIDEOMT_CAUSAL_FRAME_COUNT,
        )
    )
    if any(not isinstance(frame, CalvinPICFEvidenceFrame) for frame in prefix):
        raise TypeError("Stage P-Q-C5 received an untyped CALVIN evidence frame")
    timestamps = _validate_real_contiguous_prefix(prefix)
    source_rgb = tuple(_static_rgb(frame) for frame in prefix)
    source_rgb_sha256s = tuple(
        hashlib.sha256(value.tobytes(order="C")).hexdigest() for value in source_rgb
    )
    prepared = prepare_rgb_frames(source_rgb)
    indices = tuple(
        range(
            current_source_global_index - VIDEOMT_CAUSAL_FRAME_COUNT + 1,
            current_source_global_index + 1,
        )
    )
    return PreparedCalvinStagePQInput(
        current_source_global_index=current_source_global_index,
        source_global_indices=indices,
        timestamps_s=timestamps,
        source_rgb_sha256s=source_rgb_sha256s,
        frames=prepared,
    )


def measure_videomt_query_precision_cast(
    source: torch.Tensor,
    target_dtype: torch.dtype,
) -> VidEoMTQueryPrecisionReceipt:
    """Measure, rather than assume, the error induced by the host dtype cast."""

    if source.ndim != 3 or source.shape[1:] != (200, 1024):
        raise ValueError("precision receipt requires a complete [batch,200,1024] query bank")
    if not source.is_floating_point() or not torch.isfinite(source).all():
        raise ValueError("precision receipt source must be finite floating point")
    if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported VidEoMT host precision")
    target = source.to(dtype=target_dtype)
    reconstructed = target.float()
    reference = source.float()
    difference = reconstructed - reference
    source_norm = torch.linalg.vector_norm(reference)
    relative_l2 = torch.linalg.vector_norm(difference) / source_norm.clamp_min(
        torch.finfo(reference.dtype).tiny
    )
    cosine = torch.nn.functional.cosine_similarity(reference, reconstructed, dim=-1, eps=1e-12)

    source_rows = reference.detach().cpu().contiguous()
    target_rows = target.detach().cpu().contiguous()
    target_to_sources: dict[bytes, set[bytes]] = {}
    for source_row, target_row in zip(
        source_rows.reshape(-1, source_rows.shape[-1]),
        target_rows.reshape(-1, target_rows.shape[-1]),
        strict=True,
    ):
        source_key = source_row.view(torch.uint8).numpy().tobytes()
        target_key = target_row.view(torch.uint8).numpy().tobytes()
        target_to_sources.setdefault(target_key, set()).add(source_key)
    induced_collisions = sum(max(0, len(values) - 1) for values in target_to_sources.values())

    return VidEoMTQueryPrecisionReceipt(
        source_dtype=str(source.dtype),
        target_dtype=str(target_dtype),
        shape=tuple(int(value) for value in source.shape),
        maximum_absolute_error=float(difference.abs().max()),
        relative_l2_error=float(relative_l2),
        minimum_query_cosine=float(cosine.min()),
        exact_value_fraction=float((reconstructed == reference).float().mean()),
        induced_row_collision_count=induced_collisions,
    )
