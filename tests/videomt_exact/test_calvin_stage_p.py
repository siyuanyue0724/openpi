from __future__ import annotations

import numpy as np
import pytest
import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinPICFEvidenceFrame, CalvinPICFSensorObservation
from picf_next.videomt_exact.calvin_stage_p import (
    VIDEOMT_STAGE_PQ_C5_INTERFACE,
    InsufficientCausalPrefixError,
    empty_videomt_query_modality_batch,
    make_videomt_stage_pq_execution_receipt,
    measure_videomt_query_precision_cast,
    prepare_calvin_stage_pq_c5,
)
from picf_next.videomt_exact.observations import VIDEOMT_QUERY_MODALITY
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput


def _frame(index: int) -> CalvinPICFEvidenceFrame:
    timestamp = index / 30.0
    rgb = np.full((200, 200, 3), index, dtype=np.uint8)
    rgb.setflags(write=False)
    return CalvinPICFEvidenceFrame(
        sensor_observations=(
            CalvinPICFSensorObservation(
                key="observation.images.rgb_static",
                value=rgb,
                timestamp_s=timestamp,
                units="sRGB uint8",
            ),
        ),
        timestamp_s=timestamp,
        delta_t_s=1.0 / 30.0,
    )


class _Index:
    def __init__(self, frames: tuple[CalvinPICFEvidenceFrame, ...]) -> None:
        self.frames = frames
        self.request: tuple[int, int] | None = None

    def source_picf_evidence_prefix(
        self,
        global_index: int,
        *,
        maximum_source_frames: int,
    ) -> tuple[CalvinPICFEvidenceFrame, ...]:
        self.request = (global_index, maximum_source_frames)
        return self.frames


def test_stage_pq_c5_uses_exactly_five_real_causal_frames() -> None:
    index = _Index(tuple(_frame(value) for value in range(7, 12)))
    prepared = prepare_calvin_stage_pq_c5(index, 11)

    assert index.request == (11, 5)
    assert prepared.interface_identity == VIDEOMT_STAGE_PQ_C5_INTERFACE
    assert prepared.source_global_indices == (7, 8, 9, 10, 11)
    assert prepared.frames.model_input.shape == (5, 3, 480, 480)
    assert prepared.timestamps_s[-1] == 11 / 30.0


def test_stage_pq_c5_rejects_short_prefix_instead_of_padding() -> None:
    index = _Index(tuple(_frame(value) for value in range(4)))
    with pytest.raises(InsufficientCausalPrefixError, match="five real"):
        prepare_calvin_stage_pq_c5(index, 3)


def test_stage_pq_c5_rejects_temporal_gap() -> None:
    frames = list(_frame(value) for value in range(7, 12))
    frames[3] = _frame(20)
    with pytest.raises(ContractError, match="temporally contiguous"):
        prepare_calvin_stage_pq_c5(_Index(tuple(frames)), 11)


def test_query_precision_receipt_measures_bfloat16_cast_without_row_collision() -> None:
    torch.manual_seed(199)
    source = torch.randn(1, 200, 1024)
    receipt = measure_videomt_query_precision_cast(source, torch.bfloat16)

    assert receipt.source_dtype == "torch.float32"
    assert receipt.target_dtype == "torch.bfloat16"
    assert receipt.shape == (1, 200, 1024)
    assert receipt.maximum_absolute_error > 0
    assert 0 < receipt.relative_l2_error < 0.01
    assert receipt.minimum_query_cosine > 0.999
    assert 0 < receipt.exact_value_fraction < 1
    assert receipt.induced_row_collision_count == 0


def test_short_prefix_absence_has_no_fabricated_query_token() -> None:
    batch = empty_videomt_query_modality_batch(
        batch_size=1,
        device="cpu",
        dtype=torch.bfloat16,
    )

    stream = batch.streams[0]
    assert stream.name == VIDEOMT_QUERY_MODALITY
    assert stream.tokens.shape == (1, 0, 1024)
    assert stream.valid.shape == (1, 0)
    assert stream.canonical_token_ids.shape == (1, 0)


def test_execution_receipt_covers_full_donor_outputs_and_latest_query_boundary() -> None:
    prepared = prepare_calvin_stage_pq_c5(
        _Index(tuple(_frame(value) for value in range(7, 12))),
        11,
    )
    output = ExactVidEoMTOutput(
        class_logits=torch.randn(1, 5, 200, 41),
        mask_logits=torch.randn(1, 200, 5, 30, 30),
        query_embeddings=torch.randn(1, 5, 200, 1024),
        propagated_queries=torch.randn(1, 200, 1024),
        auxiliary_outputs=(),
    )

    receipt = make_videomt_stage_pq_execution_receipt(
        prepared,
        output,
        host_dtype=torch.bfloat16,
    )

    assert receipt.class_logits_shape == (1, 5, 200, 41)
    assert receipt.mask_logits_shape == (1, 200, 5, 30, 30)
    assert receipt.query_embeddings_shape == (1, 5, 200, 1024)
    assert receipt.propagated_queries_shape == (1, 200, 1024)
    assert receipt.host_injected_output == "query_embeddings.latest_all_200"
    assert receipt.precision_cast.target_dtype == "torch.bfloat16"
