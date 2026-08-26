from __future__ import annotations

import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)
from picf_next.lingbot_native.modalities import NativeObjectQuerySpatialSpec
from picf_next.videomt_exact.observations import (
    VIDEOMT_CALVIN_MASK_LAYOUT,
    VIDEOMT_MASK_RELATION,
    VIDEOMT_QUERY_MODALITY,
    VidEoMTQueryObservation,
    videomt_query_modality_spec,
    videomt_row_mask_query_modality_spec,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput


def _output(
    *,
    with_mask_decoder: bool = False,
    mask_shape: tuple[int, int] = (14, 14),
) -> ExactVidEoMTOutput:
    height, width = mask_shape
    mask_embeddings = torch.randn(1, 200, 1024) if with_mask_decoder else None
    mask_features = torch.randn(1, 1024, height, width) if with_mask_decoder else None
    return ExactVidEoMTOutput(
        class_logits=torch.randn(1, 2, 200, 41),
        mask_logits=torch.randn(1, 200, 2, height, width),
        query_embeddings=torch.randn(1, 2, 200, 1024, requires_grad=True),
        propagated_queries=torch.randn(1, 200, 1024),
        auxiliary_outputs=(),
        latest_mask_embeddings=mask_embeddings,
        latest_mask_features=mask_features,
    )


def _controls() -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.zeros(1, 1, 2),
        field_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        delta_time=torch.ones(1, 1),
        reset=torch.zeros(1, 1, dtype=torch.bool),
        acknowledged=torch.ones(1, 1, dtype=torch.bool),
    )


def test_stage_p_retains_every_query_without_objectness_threshold() -> None:
    output = _output()
    with torch.no_grad():
        output.class_logits[..., -1] = 100.0
    observation = VidEoMTQueryObservation.from_exact_output(output)
    batch = observation.as_native_modality_batch()
    stream = batch.streams[0]

    assert stream.name == VIDEOMT_QUERY_MODALITY
    assert stream.tokens.shape == (1, 200, 1024)
    assert stream.valid.all()
    assert observation.object_probability.max() < 1e-6
    torch.testing.assert_close(stream.tokens, output.query_embeddings[:, -1])
    torch.testing.assert_close(
        stream.canonical_token_ids,
        torch.arange(200).unsqueeze(0),
    )
    batch.validate_against((videomt_query_modality_spec(),))


def test_stage_p_exact_bridge_preserves_query_values_and_reaches_upstream_queries() -> None:
    output = _output()
    observation = VidEoMTQueryObservation.from_exact_output(output)
    batch = observation.as_native_modality_batch()
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=1024,
            executed_action_dim=2,
            num_layers=2,
            modality_specs=(videomt_query_modality_spec(),),
        )
    )
    context = LingBotNativeContext(controls=_controls(), modalities=batch)
    projected, valid, _direct_action_visible, relation_surfaces = graph._project_modalities(
        context,
        prefix=torch.zeros(1, 1, 1024),
    )

    assert projected.shape == (1, 200, 1024)
    assert valid.all()
    assert relation_surfaces == ()
    projection = graph.modality_projections[VIDEOMT_QUERY_MODALITY]
    gram = projection.weight.T @ projection.weight
    torch.testing.assert_close(gram, torch.eye(1024), atol=0, rtol=0)
    expected = projection(observation.query_tokens) + graph.modality_embeddings[0]
    torch.testing.assert_close(projected, expected, atol=0, rtol=0)

    projected.square().mean().backward()
    assert output.query_embeddings.grad is not None
    assert output.query_embeddings.grad[:, -1].abs().sum() > 0
    assert output.query_embeddings.grad[:, 0].abs().sum() == 0


def test_stage_pqm_preserves_complete_class_mask_relation_without_selection() -> None:
    output = _output()
    observation = VidEoMTQueryObservation.from_exact_output(output)
    spec = NativeObjectQuerySpatialSpec(
        name=VIDEOMT_MASK_RELATION,
        query_modality=VIDEOMT_QUERY_MODALITY,
        geometry_kind="image_grid",
        layout="videomt.calvin.static.14x14.v1",
    )
    batch = observation.as_native_pqm_batch(relation_spec=spec)
    relation = batch.object_query_spatial_relations[0]

    assert relation.object_logits.shape == (1, 200)
    assert relation.mask_logits.shape == (1, 200, 14 * 14)
    assert relation.grid_shape == (14, 14)
    assert relation.query_valid.all() and relation.pixel_valid.all()
    torch.testing.assert_close(
        relation.object_logits,
        torch.logsumexp(output.class_logits[:, -1, :, :-1], dim=-1)
        - output.class_logits[:, -1, :, -1],
    )
    torch.testing.assert_close(
        relation.mask_logits,
        output.mask_logits[:, :, -1].flatten(2),
    )
    batch.validate_object_query_spatial_relations((spec,))


def test_row_mask_boundary_preserves_mask_embeddings_and_dense_features() -> None:
    output = _output(with_mask_decoder=True)
    observation = VidEoMTQueryObservation.from_exact_output(output)
    spec = NativeObjectQuerySpatialSpec(
        name=VIDEOMT_MASK_RELATION,
        query_modality=VIDEOMT_QUERY_MODALITY,
        geometry_kind="image_grid",
        layout="videomt.calvin.static.14x14.v1",
    )
    batch = observation.as_native_row_mask_batch(relation_spec=spec)
    stream = batch.streams[0]
    relation = batch.object_query_spatial_relations[0]

    batch.validate_against((videomt_row_mask_query_modality_spec(),))
    batch.validate_object_query_spatial_relations((spec,))
    torch.testing.assert_close(stream.tokens, output.query_embeddings[:, -1])
    torch.testing.assert_close(stream.metadata, output.latest_mask_embeddings)
    torch.testing.assert_close(
        relation.dense_mask_features,
        output.latest_mask_features.flatten(2).transpose(1, 2),
    )
    assert relation.mask_logits.shape == (1, 200, 14 * 14)
    assert relation.grid_shape == (14, 14)


def test_one_static_layout_accepts_dynamic_training_mask_grids() -> None:
    spec = NativeObjectQuerySpatialSpec(
        name=VIDEOMT_MASK_RELATION,
        query_modality=VIDEOMT_QUERY_MODALITY,
        geometry_kind="image_grid",
        layout=VIDEOMT_CALVIN_MASK_LAYOUT,
    )

    for grid_shape in ((104, 104), (128, 128)):
        observation = VidEoMTQueryObservation.from_exact_output(
            _output(mask_shape=grid_shape)
        )
        relation = observation.as_native_pqm_batch(
            relation_spec=spec
        ).object_query_spatial_relations[0]

        assert relation.layout == VIDEOMT_CALVIN_MASK_LAYOUT
        assert relation.grid_shape == grid_shape
        assert relation.pixel_count == grid_shape[0] * grid_shape[1]
