from __future__ import annotations

import torch
from torch import nn

from picf_next.lingbot_native.host import LingBotNativeGraph, LingBotNativeGraphConfig
from picf_next.lingbot_native.modalities import (
    TOKEN_IDENTITY,
    NativeModalityBatch,
    NativeModalitySpec,
    NativeModalityStream,
    NativeObjectQuerySpatialSpec,
)
from picf_next.videomt_exact.observations import (
    VIDEOMT_MASK_RELATION,
    VIDEOMT_QUERY_MODALITY,
    VIDEOMT_STAGE_PQ_INTERFACE,
    VIDEOMT_STAGE_PQM_INTERFACE,
    VIDEOMT_STAGE_PQMR_INTERFACE,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput
from picf_next.videomt_exact.stage_p import (
    VidEoMTStageP,
    VidEoMTStagePQM,
    VidEoMTStagePQMR,
    with_videomt_query_modality_spec,
    with_videomt_row_mask_query_modality_spec,
)


class _Runtime(nn.Module):
    def forward(self, value: torch.Tensor, *, resume: bool = False) -> ExactVidEoMTOutput:
        assert value.shape == (2, 3, 16, 16)
        assert not resume
        return ExactVidEoMTOutput(
            class_logits=torch.randn(1, 2, 200, 41),
            mask_logits=torch.randn(1, 200, 2, 4, 4),
            query_embeddings=torch.randn(1, 2, 200, 1024),
            propagated_queries=torch.randn(1, 200, 1024),
            auxiliary_outputs=(),
        )


class _RowMaskRuntime(nn.Module):
    def forward(self, value: torch.Tensor, *, resume: bool = False) -> ExactVidEoMTOutput:
        assert value.shape == (2, 3, 16, 16)
        assert not resume
        return ExactVidEoMTOutput(
            class_logits=torch.randn(1, 2, 200, 41),
            mask_logits=torch.randn(1, 200, 2, 4, 4),
            query_embeddings=torch.randn(1, 2, 200, 1024),
            propagated_queries=torch.randn(1, 200, 1024),
            auxiliary_outputs=(),
            latest_mask_embeddings=torch.randn(1, 200, 1024),
            latest_mask_features=torch.randn(1, 1024, 4, 4),
        )


def test_stage_p_augments_existing_modalities_without_replacing_them() -> None:
    touch = NativeModalityBatch(
        (
            NativeModalityStream(
                name="anytouch",
                tokens=torch.randn(1, 3, 8),
                valid=torch.ones(1, 3, dtype=torch.bool),
            ),
        )
    )
    result = VidEoMTStageP(_Runtime())(
        torch.randn(2, 3, 16, 16),
        existing_modalities=touch,
        host_dtype=torch.float32,
    )
    assert tuple(stream.name for stream in result.modalities.streams) == (
        "anytouch",
        "videomt_queries",
    )
    assert result.modalities.token_count == 203
    assert result.interface_identity == VIDEOMT_STAGE_PQ_INTERFACE

    specs = with_videomt_query_modality_spec((NativeModalitySpec("anytouch", 8, 3),))
    result.modalities.validate_against(specs)
    assert specs[-1].token_normalization == TOKEN_IDENTITY


def test_stage_p_host_precision_adapter_retains_all_queries() -> None:
    result = VidEoMTStageP(_Runtime())(
        torch.randn(2, 3, 16, 16),
        host_dtype=torch.bfloat16,
    )

    stream = result.modalities.streams[0]
    assert stream.tokens.dtype == torch.bfloat16
    assert stream.tokens.shape == (1, 200, 1024)
    assert stream.valid.all()
    assert torch.equal(stream.canonical_token_ids, torch.arange(200).unsqueeze(0))


def test_all_200_queries_reach_the_shared_host_projection_without_donor_gradient() -> None:
    runtime = _Runtime().requires_grad_(False).eval()
    with torch.no_grad():
        result = VidEoMTStageP(runtime).eval()(
            torch.randn(2, 3, 16, 16),
            host_dtype=torch.float32,
            resume=False,
        )
    specs = with_videomt_query_modality_spec(())
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=1024,
            executed_action_dim=2,
            num_layers=2,
            maximum_control_tokens=1,
            modality_specs=specs,
        )
    ).train()
    context = type("Context", (), {"modalities": result.modalities})()

    projected, valid, direct_action_visible, relation_surfaces = graph._project_modalities(
        context,
        prefix=torch.zeros(1, 1, 1024),
    )

    assert projected.shape == (1, 200, 1024)
    assert valid.shape == (1, 200)
    assert valid.all()
    assert not direct_action_visible.any()
    assert relation_surfaces == ()
    projected.square().mean().backward()
    host_projection = graph.modality_projections["videomt_queries"]
    assert host_projection.weight.grad is not None
    assert host_projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in runtime.parameters())


def test_stage_pqm_augments_existing_modalities_with_the_complete_dense_codec() -> None:
    touch = NativeModalityBatch(
        (
            NativeModalityStream(
                name="anytouch",
                tokens=torch.randn(1, 3, 8),
                valid=torch.ones(1, 3, dtype=torch.bool),
            ),
        )
    )
    spec = NativeObjectQuerySpatialSpec(
        name=VIDEOMT_MASK_RELATION,
        query_modality=VIDEOMT_QUERY_MODALITY,
        geometry_kind="image_grid",
        layout="videomt.calvin.static.4x4.v1",
    )
    result = VidEoMTStagePQM(_Runtime(), relation_spec=spec)(
        torch.randn(2, 3, 16, 16),
        existing_modalities=touch,
        host_dtype=torch.float32,
    )

    assert result.interface_identity == VIDEOMT_STAGE_PQM_INTERFACE
    assert tuple(stream.name for stream in result.modalities.streams) == (
        "anytouch",
        "videomt_queries",
    )
    assert len(result.modalities.object_query_spatial_relations) == 1
    relation = result.modalities.object_query_spatial_relations[0]
    assert relation.mask_logits.shape == (1, 200, 16)


def test_stage_pqmr_preserves_complete_source_decoder_basis() -> None:
    spec = NativeObjectQuerySpatialSpec(
        name=VIDEOMT_MASK_RELATION,
        query_modality=VIDEOMT_QUERY_MODALITY,
        geometry_kind="image_grid",
        layout="videomt.calvin.static.4x4.v1",
    )
    result = VidEoMTStagePQMR(_RowMaskRuntime(), relation_spec=spec)(
        torch.randn(2, 3, 16, 16),
        host_dtype=torch.bfloat16,
    )

    assert result.interface_identity == VIDEOMT_STAGE_PQMR_INTERFACE
    stream = result.modalities.streams[0]
    relation = result.modalities.object_query_spatial_relations[0]
    assert stream.tokens.shape == (1, 200, 1024)
    assert stream.metadata.shape == (1, 200, 1024)
    assert relation.dense_mask_features.shape == (1, 16, 1024)
    assert stream.tokens.dtype == stream.metadata.dtype == torch.bfloat16
    assert relation.dense_mask_features.dtype == torch.bfloat16
    specs = with_videomt_row_mask_query_modality_spec(())
    result.modalities.validate_against(specs)
    result.modalities.validate_object_query_spatial_relations((spec,))
