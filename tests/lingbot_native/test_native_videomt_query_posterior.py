from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import (
    NATIVE_VIDEOMT_QUERY_COUNT,
    NATIVE_VIDEOMT_QUERY_POSTERIOR,
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)
from picf_next.lingbot_native.modalities import (
    CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
    NativeModalityBatch,
    NativeModalitySpec,
    NativeModalityStream,
    NativeObjectQuerySpatialRelation,
    NativeObjectQuerySpatialSpec,
    modality_bridge_input,
)
from picf_next.lingbot_native.physical_relations import (
    NativeObjectQueryPosteriorOutput,
)
from picf_next.lingbot_native.state import NativeLayerwisePriorTrace
from picf_next.lingbot_native.training import _validate_finalized_native_context


def _controls() -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.tensor([[[0.25, -0.5]]]),
        field_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        delta_time=torch.full((1, 1), 0.1),
        reset=torch.zeros(1, 1, dtype=torch.bool),
        acknowledged=torch.ones(1, 1, dtype=torch.bool),
    )


def _spatial_spec() -> NativeObjectQuerySpatialSpec:
    return NativeObjectQuerySpatialSpec(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout="videomt.calvin.static.2x2.v1",
    )


def _graph() -> LingBotNativeGraph:
    torch.manual_seed(207)
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=NATIVE_VIDEOMT_QUERY_COUNT,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(
                NativeModalitySpec(
                    "videomt_queries",
                    input_width=4,
                    maximum_tokens=NATIVE_VIDEOMT_QUERY_COUNT,
                ),
            ),
            object_query_spatial_specs=(_spatial_spec(),),
            architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
        )
    )


def _modalities() -> NativeModalityBatch:
    query_count = NATIVE_VIDEOMT_QUERY_COUNT
    valid = torch.ones(1, query_count, dtype=torch.bool)
    canonical = torch.arange(query_count).unsqueeze(0)
    tokens = torch.arange(query_count * 4, dtype=torch.float32).reshape(1, query_count, 4)
    class_logits = torch.randn(1, query_count, 3)
    relation = NativeObjectQuerySpatialRelation(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout="videomt.calvin.static.2x2.v1",
        object_logits=(
            torch.logsumexp(class_logits[..., :-1], dim=-1) - class_logits[..., -1]
        ),
        mask_logits=torch.randn(1, query_count, 4),
        query_valid=valid,
        pixel_valid=torch.ones(1, 4, dtype=torch.bool),
        canonical_query_ids=canonical,
        grid_shape=(2, 2),
        class_logits=class_logits,
    )
    return NativeModalityBatch(
        (
            NativeModalityStream(
                "videomt_queries",
                tokens,
                valid,
                canonical_token_ids=canonical,
            ),
        ),
        (relation,),
    )


def _context(modalities: NativeModalityBatch) -> LingBotNativeContext:
    return LingBotNativeContext(
        controls=_controls(),
        modalities=modalities,
        prior_trace=NativeLayerwisePriorTrace(torch.zeros(1, 3, 200, 8)),
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ),
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2]),
    )


def test_native_query_profile_rejects_any_query_reduction() -> None:
    with pytest.raises(ValueError, match="all 200"):
        LingBotNativeGraphConfig(
            capacity=199,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            modality_specs=(NativeModalitySpec("videomt_queries", 4, 200),),
            object_query_spatial_specs=(_spatial_spec(),),
            architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
        )


def test_native_queries_enter_once_as_same_index_posterior_rows() -> None:
    graph = _graph()
    modalities = _modalities()
    context = _context(modalities)
    prefix = torch.randn(1, 3, 8)
    action = torch.randn(1, 2, 4)
    inputs, mask, positions, visual, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=torch.ones(1, 5, 5, dtype=torch.bool),
        position_ids=torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5).clone(),
        visual_pos_masks=torch.tensor([[True, True, False]]),
        context=context,
    )
    assert runtime.modality_slice.start == runtime.modality_slice.stop
    assert runtime.posterior_slice.stop - runtime.posterior_slice.start == 200
    assert mask.shape[-1] == inputs[0].shape[1] + action.shape[1]
    assert positions.shape[-1] == mask.shape[-1]
    assert visual is not None and visual.shape[1] == inputs[0].shape[1]

    stream = modalities.streams[0]
    spec = graph.config.modality_specs[0]
    projected = graph.modality_projections[stream.name](modality_bridge_input(stream, spec))
    projected = projected + graph.modality_embeddings[0] + graph.role_embeddings[1]
    torch.testing.assert_close(inputs[0][:, runtime.posterior_slice], projected)


def test_native_output_preserves_complete_source_masks_classes_and_indices() -> None:
    graph = _graph()
    modalities = _modalities()
    context = _context(modalities)
    prefix = torch.randn(1, 3, 8)
    action = torch.randn(1, 2, 4)
    inputs, _mask, _positions, _visual, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=torch.ones(1, 5, 5, dtype=torch.bool),
        position_ids=torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5).clone(),
        visual_pos_masks=torch.tensor([[True, True, False]]),
        context=context,
    )
    for layer_index in range(graph.config.num_layers):
        graph.record_layerwise_posterior(
            prefix_hidden=inputs[0],
            runtime=runtime,
            layer_index=layer_index,
        )
    graph.finalize_joint_outputs(outputs_embeds=inputs, runtime=runtime)

    output = context.relation_output
    assert isinstance(output, NativeObjectQueryPosteriorOutput)
    source = modalities.object_query_spatial_relations[0]
    assert output.relation is source
    torch.testing.assert_close(output.support_logits, source.mask_logits.transpose(1, 2))
    torch.testing.assert_close(output.relation.class_logits, source.class_logits)
    torch.testing.assert_close(output.posterior_rows, context.posterior_state.rows)
    assert len(context.root_output_tensors()) >= 7

    checks = _validate_finalized_native_context(
        context=context,
        graph=graph,
        advertised_native_outputs=context.root_output_tensors(),
        root_output_dtype=None,
        require_prediction_grad=False,
        required_relation_grad_fields=(),
    )
    assert checks
    assert all(bool(predicate.item()) for _message, predicate in checks)

    with pytest.raises(RuntimeError, match="forbids legacy relation-gradient"):
        _validate_finalized_native_context(
            context=context,
            graph=graph,
            advertised_native_outputs=context.root_output_tensors(),
            root_output_dtype=None,
            require_prediction_grad=False,
            required_relation_grad_fields=("ownership",),
        )
