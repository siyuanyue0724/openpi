from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import (
    CONTENT_ADDRESSED_SET_TRANSITION,
    LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
    LINGBOT_TASK_TOKEN_RESAMPLER_BRIDGE,
    NATIVE_VIDEOMT_QUERY_COUNT,
    NATIVE_VIDEOMT_QUERY_POSTERIOR,
    UNIFIED_LAYERWISE_PREDICT_CORRECT,
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
    LingBotNativePriorStepper,
    LingBotPriorRolloutContext,
    install_lingbot_native_graph,
    native_context_from_prior_trace,
)
from picf_next.lingbot_native.modalities import (
    CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
    CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
    TOKEN_IDENTITY,
    NativeModalityBatch,
    NativeModalitySpec,
    NativeModalityStream,
    NativeObjectQuerySpatialRelation,
    NativeObjectQuerySpatialSpec,
    NativeRelationSurfaceSpec,
)
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.state import (
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
    NativePosteriorState,
    clone_persistent_state,
)


def _controls(
    batch: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
) -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.tensor([[[0.25, -0.5]]], dtype=dtype).expand(batch, -1, -1).clone(),
        field_valid=torch.ones(batch, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(batch, 1, dtype=torch.bool),
        delta_time=torch.full((batch, 1), 0.1, dtype=dtype),
        reset=torch.zeros(batch, 1, dtype=torch.bool),
        acknowledged=torch.ones(batch, 1, dtype=torch.bool),
    )


def _context(previous: NativePosteriorState | None = None) -> LingBotNativeContext:
    return LingBotNativeContext(
        controls=_controls(),
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ),
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2]),
        previous_state=previous,
    )


def _inputs() -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(5)
    prefix = torch.randn(1, 3, 8)
    action = torch.randn(1, 2, 4)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    positions = torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5).clone()
    visual = torch.tensor([[True, True, False]])
    return [prefix, action], mask, positions, visual


class _FakeTaskTokenResampler(torch.nn.Module):
    """Shape-faithful test double for LingBot's released one-layer resampler."""

    def __init__(self, host_width: int = 8, output_width: int = 4) -> None:
        super().__init__()
        self.num_queries = 4
        self.proj_in1 = torch.nn.Linear(host_width, host_width)
        self.proj_in2 = torch.nn.Linear(host_width, host_width)
        self.proj_out = torch.nn.Linear(host_width, output_width)
        self.norm_out = torch.nn.LayerNorm(output_width)
        self.layers = torch.nn.ModuleList(
            [torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])]
        )

    def forward(self, tokens: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        query = self.proj_in1(queries)
        evidence = torch.tanh(self.proj_in2(tokens)).mean(dim=1, keepdim=True)
        return self.norm_out(self.proj_out(query + evidence))


def _resampled_graph(
    *,
    direct_proprioception: bool = False,
    relation_surface: bool = False,
) -> LingBotNativeGraph:
    torch.manual_seed(97)
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(
                NativeModalitySpec("anytouch", 3, 2, metadata_width=3),
                NativeModalitySpec("proprioception", 2, 1),
                NativeModalitySpec("sonata", 4, 2, metadata_width=2),
                NativeModalitySpec("vjepa", 5, 2, metadata_width=3),
            ),
            modality_bridge_identity=LINGBOT_TASK_TOKEN_RESAMPLER_BRIDGE,
            modality_bridge_query_count=4,
            resampled_modality_names=("anytouch", "sonata", "vjepa"),
            direct_action_modality_names=(("proprioception",) if direct_proprioception else ()),
            relation_surface_specs=(
                (
                    NativeRelationSurfaceSpec(
                        name="vjepa",
                        geometry_kind="image_grid",
                        layout="vjepa21.calvin.static-gripper.24x24.v1",
                        target_kind=CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
                    ),
                )
                if relation_surface
                else ()
            ),
            relation_supervision_layers=((0,) if relation_surface else ()),
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
        ),
        modality_bridge_projector=_FakeTaskTokenResampler(),
        modality_bridge_queries=torch.randn(4, 8),
    )


def _resampled_modalities(*, dense_valid: bool = True) -> NativeModalityBatch:
    valid = torch.full((1, 2), dense_valid, dtype=torch.bool)
    canonical_token_ids = (
        torch.arange(2).unsqueeze(0) if dense_valid else torch.full((1, 2), -1, dtype=torch.long)
    )
    return NativeModalityBatch(
        (
            NativeModalityStream(
                "anytouch",
                torch.tensor([[[0.1, 0.2, 0.3], [1.0, -0.5, 0.7]]]),
                valid.clone(),
                metadata=torch.tensor([[[0.4, -0.3, 0.8], [-0.2, 0.9, 0.5]]]),
                canonical_token_ids=canonical_token_ids.clone(),
            ),
            NativeModalityStream(
                "proprioception",
                torch.tensor([[[0.25, -0.75]]]),
                torch.ones(1, 1, dtype=torch.bool),
            ),
            NativeModalityStream(
                "sonata",
                torch.tensor([[[0.6, 0.2, -0.8, 0.4], [-0.1, 0.9, 0.5, -0.7]]]),
                valid.clone(),
                metadata=torch.tensor([[[0.7, -0.4], [0.2, 0.8]]]),
                canonical_token_ids=canonical_token_ids.clone(),
            ),
            NativeModalityStream(
                "vjepa",
                torch.tensor([[[0.3, -0.2, 0.5, 0.8, -0.6], [0.9, 0.1, -0.4, 0.2, 0.7]]]),
                valid.clone(),
                metadata=torch.tensor([[[0.2, 0.6, -0.9], [0.8, -0.1, 0.4]]]),
                canonical_token_ids=canonical_token_ids.clone(),
            ),
        )
    )


def _object_query_graph() -> LingBotNativeGraph:
    torch.manual_seed(703)
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(NativeModalitySpec("videomt_queries", 4, 2),),
            object_query_spatial_specs=(
                NativeObjectQuerySpatialSpec(
                    name="videomt_masks",
                    query_modality="videomt_queries",
                    geometry_kind="image_grid",
                    target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
                    layout="videomt.calvin.static.2x2.v1",
                ),
            ),
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
        )
    )


def _object_query_modalities() -> NativeModalityBatch:
    query_valid = torch.ones(1, 2, dtype=torch.bool)
    canonical_ids = torch.arange(2).unsqueeze(0)
    relation = NativeObjectQuerySpatialRelation(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout="videomt.calvin.static.2x2.v1",
        object_logits=torch.tensor([[8.0, 8.0]]),
        mask_logits=torch.tensor([[[8.0, 8.0, -8.0, -8.0], [-8.0, -8.0, 8.0, 8.0]]]),
        query_valid=query_valid,
        pixel_valid=torch.ones(1, 4, dtype=torch.bool),
        canonical_query_ids=canonical_ids,
        grid_shape=(2, 2),
    )
    return NativeModalityBatch(
        (
            NativeModalityStream(
                "videomt_queries",
                torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
                query_valid,
                canonical_token_ids=canonical_ids,
            ),
        ),
        (relation,),
    )


def _native_videomt_multimodal_graph() -> LingBotNativeGraph:
    torch.manual_seed(207)
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=200,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(
                NativeModalitySpec("anytouch", 3, 1),
                NativeModalitySpec("sonata", 4, 1),
                NativeModalitySpec("videomt_queries", 4, 200),
                NativeModalitySpec("vjepa", 5, 1),
            ),
            object_query_spatial_specs=(
                NativeObjectQuerySpatialSpec(
                    name="videomt_masks",
                    query_modality="videomt_queries",
                    geometry_kind="image_grid",
                    target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
                    layout="videomt.calvin.static.2x2.v1",
                ),
            ),
            architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
        )
    )


def _native_videomt_multimodal_batch(
) -> tuple[NativeModalityBatch, dict[str, torch.Tensor]]:
    tokens = {
        "anytouch": torch.randn(1, 1, 3, requires_grad=True),
        "sonata": torch.randn(1, 1, 4, requires_grad=True),
        "videomt_queries": torch.randn(1, 200, 4, requires_grad=True),
        "vjepa": torch.randn(1, 1, 5, requires_grad=True),
    }
    query_valid = torch.ones(1, 200, dtype=torch.bool)
    canonical_query_ids = torch.arange(200).unsqueeze(0)
    streams = (
        NativeModalityStream(
            "anytouch",
            tokens["anytouch"],
            torch.ones(1, 1, dtype=torch.bool),
        ),
        NativeModalityStream(
            "sonata",
            tokens["sonata"],
            torch.ones(1, 1, dtype=torch.bool),
        ),
        NativeModalityStream(
            "videomt_queries",
            tokens["videomt_queries"],
            query_valid,
            canonical_token_ids=canonical_query_ids,
        ),
        NativeModalityStream(
            "vjepa",
            tokens["vjepa"],
            torch.ones(1, 1, dtype=torch.bool),
        ),
    )
    relation = NativeObjectQuerySpatialRelation(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout="videomt.calvin.static.2x2.v1",
        object_logits=torch.zeros(1, 200),
        mask_logits=torch.zeros(1, 200, 4),
        query_valid=query_valid,
        pixel_valid=torch.ones(1, 4, dtype=torch.bool),
        canonical_query_ids=canonical_query_ids,
        grid_shape=(2, 2),
    )
    return NativeModalityBatch(streams, (relation,)), tokens


def _direct_row_mask_graph() -> LingBotNativeGraph:
    torch.manual_seed(704)
    source_mask_head = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        source_mask_head.weight.copy_(torch.eye(4))
    source_mask_head.requires_grad_(False).eval()
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(
                NativeModalitySpec(
                    "videomt_queries",
                    4,
                    2,
                    metadata_width=4,
                    token_normalization=TOKEN_IDENTITY,
                    metadata_normalization=TOKEN_IDENTITY,
                ),
            ),
            object_query_spatial_specs=(
                NativeObjectQuerySpatialSpec(
                    name="videomt_masks",
                    query_modality="videomt_queries",
                    geometry_kind="image_grid",
                    target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
                    layout="videomt.calvin.static.2x2.v1",
                ),
            ),
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
        ),
        source_mask_head=source_mask_head,
    )


def _direct_row_mask_modalities() -> NativeModalityBatch:
    query_valid = torch.ones(1, 2, dtype=torch.bool)
    canonical_ids = torch.arange(2).unsqueeze(0)
    relation = NativeObjectQuerySpatialRelation(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout="videomt.calvin.static.2x2.v1",
        object_logits=torch.zeros(1, 2),
        mask_logits=torch.zeros(1, 2, 4),
        query_valid=query_valid,
        pixel_valid=torch.ones(1, 4, dtype=torch.bool),
        canonical_query_ids=canonical_ids,
        grid_shape=(2, 2),
        dense_mask_features=torch.eye(4).unsqueeze(0),
    )
    return NativeModalityBatch(
        (
            NativeModalityStream(
                "videomt_queries",
                torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
                query_valid,
                metadata=torch.tensor(
                    [[[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]]]
                ),
                canonical_token_ids=canonical_ids,
            ),
        ),
        (relation,),
    )


def _graph() -> LingBotNativeGraph:
    torch.manual_seed(2)
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            predictive_target_widths=(("dino_video", 4),),
        )
    )


def _unified_graph() -> LingBotNativeGraph:
    torch.manual_seed(2)
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
        )
    )


def _toy_unified_prior_policy(
    graph: LingBotNativeGraph,
    *,
    seen_validity: list[torch.Tensor] | None = None,
) -> SimpleNamespace:
    def prior_forward(**kwargs: object):
        context = kwargs["picf_native_context"]
        assert isinstance(context, LingBotPriorRolloutContext)
        assert context.previous_memory_valid is not None
        if seen_validity is not None:
            seen_validity.append(context.previous_memory_valid.clone())
        prepared, _mask, _positions, _visual, runtime = graph.prepare_joint_inputs(
            inputs_embeds=kwargs["inputs_embeds"],  # type: ignore[arg-type]
            attention_mask=kwargs["attention_mask"],  # type: ignore[arg-type]
            position_ids=kwargs["position_ids"],  # type: ignore[arg-type]
            visual_pos_masks=kwargs["visual_pos_masks"],  # type: ignore[arg-type]
            context=context,
        )
        assert prepared[0] is not None
        hidden = prepared[0]
        for layer_index in range(graph.config.num_layers):
            memory_inputs = graph.layerwise_memory_inputs(
                layer_index=layer_index,
                runtime=runtime,
            )
            memory_update = torch.zeros_like(hidden)
            if memory_inputs is not None:
                memory_hidden, _address, visibility = memory_inputs
                memory_update = visibility.to(hidden.dtype) @ memory_hidden
            hidden = hidden + memory_update + float(layer_index + 1)
            graph.record_layerwise_posterior(
                prefix_hidden=hidden,
                runtime=runtime,
                layer_index=layer_index,
            )
        outputs = [hidden, None]
        graph.finalize_joint_outputs(outputs_embeds=outputs, runtime=runtime)
        return outputs, None, []

    return SimpleNamespace(
        model=SimpleNamespace(
            qwenvl_with_expert=SimpleNamespace(picf_native_graph=graph),
        ),
        picf_native_prior_forward=prior_forward,
    )


def test_unified_v3_has_an_explicit_identity_and_exactly_the_v2_parameter_surface() -> None:
    torch.manual_seed(2)
    v2 = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            architecture_identity=LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
        )
    )
    v3 = _unified_graph()

    assert v3.config.architecture_identity == UNIFIED_LAYERWISE_PREDICT_CORRECT
    assert v3.unified_predict_correct
    assert v3.layerwise_recurrence
    assert v3.task_independent
    assert [(name, type(module)) for name, module in v3.named_modules()] == [
        (name, type(module)) for name, module in v2.named_modules()
    ]
    assert [(name, parameter.shape) for name, parameter in v3.named_parameters()] == [
        (name, parameter.shape) for name, parameter in v2.named_parameters()
    ]
    v3.load_state_dict(v2.state_dict(), strict=True)
    forbidden = ("scorer", "confidence", "lifecycle")
    assert not any(
        fragment in name for name, _module in v3.named_modules() for fragment in forbidden
    )


def test_unified_prior_and_correction_traces_remain_attached_across_host_passes() -> None:
    graph = _unified_graph().train()
    history_rows = torch.randn(1, 3, 2, 8, requires_grad=True)
    previous = NativeLayerwisePosteriorState(history_rows)
    prior_context = LingBotPriorRolloutContext(
        controls=_controls(),
        previous_memory=previous,
        previous_memory_valid=torch.tensor([True]),
    )
    empty = torch.empty(1, 0, 8)
    prepared, _mask, _positions, _visual, prior_runtime = graph.prepare_joint_inputs(
        inputs_embeds=[empty, None],
        attention_mask=torch.empty(1, 0, 0, dtype=torch.bool),
        position_ids=torch.empty(3, 1, 0, dtype=torch.long),
        visual_pos_masks=torch.empty(1, 0, dtype=torch.bool),
        context=prior_context,
    )
    assert prepared[0] is not None
    prior_hidden = prepared[0]
    for layer_index in range(graph.config.num_layers):
        address_bias = graph.layerwise_qk_address_bias(
            prefix_hidden=prior_hidden,
            runtime=prior_runtime,
        )
        assert address_bias is not None
        assert not address_bias[:, : prior_runtime.prior_slice.start].any()
        torch.testing.assert_close(
            address_bias[:, prior_runtime.prior_slice],
            graph.object_addresses.unsqueeze(0),
        )
        memory_inputs = graph.layerwise_memory_inputs(
            layer_index=layer_index,
            runtime=prior_runtime,
        )
        assert memory_inputs is not None
        memory_hidden, _memory_address, visibility = memory_inputs
        torch.testing.assert_close(memory_hidden, history_rows[:, layer_index])
        assert visibility[:, prior_runtime.prior_slice].sum().item() == graph.config.capacity
        assert not visibility[:, : prior_runtime.prior_slice.start].any()

        control_summary = prior_hidden[:, : prior_runtime.prior_slice.start].mean(
            dim=1,
            keepdim=True,
        )
        next_hidden = prior_hidden.clone()
        next_hidden[:, prior_runtime.prior_slice] = (
            prior_hidden[:, prior_runtime.prior_slice] + memory_hidden + control_summary
        )
        graph.record_layerwise_posterior(
            prefix_hidden=next_hidden,
            runtime=prior_runtime,
            layer_index=layer_index,
        )
        prior_hidden = next_hidden
    graph.finalize_joint_outputs(
        outputs_embeds=[prior_hidden, None],
        runtime=prior_runtime,
    )
    prior_trace = prior_context.prior_trace
    assert isinstance(prior_trace, NativeLayerwisePriorTrace)
    assert prior_trace.layer_rows.requires_grad
    assert prior_trace.layer_rows.grad_fn is not None

    correction_context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=prior_trace,
    )
    correction_context.bind_native_prefix(
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        visual_sensor_mask=torch.tensor([[True, True, False]]),
        language_start=2,
        language_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    inputs, host_mask, position_ids, visual = _inputs()
    corrected, _mask, _positions, _visual, correction_runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=position_ids,
        visual_pos_masks=visual,
        context=correction_context,
    )
    assert corrected[0] is not None and corrected[1] is not None
    correction_hidden = corrected[0]
    action_start = correction_hidden.shape[1]
    for layer_index in range(graph.config.num_layers):
        memory_inputs = graph.layerwise_memory_inputs(
            layer_index=layer_index,
            runtime=correction_runtime,
        )
        assert memory_inputs is not None
        memory_hidden, _memory_address, visibility = memory_inputs
        torch.testing.assert_close(memory_hidden, prior_trace.layer(layer_index))
        assert visibility[:, action_start:].all()
        assert visibility[:, correction_runtime.posterior_slice].sum().item() == (
            graph.config.capacity
        )
        assert not visibility[:, : correction_runtime.control_slice.stop].any()

        sensor_summary = correction_hidden[:, : correction_runtime.control_slice.start].mean(
            dim=1,
            keepdim=True,
        )
        next_hidden = correction_hidden.clone()
        next_hidden[:, correction_runtime.posterior_slice] = (
            correction_hidden[:, correction_runtime.posterior_slice]
            + memory_hidden
            + sensor_summary
        )
        graph.record_layerwise_posterior(
            prefix_hidden=next_hidden,
            runtime=correction_runtime,
            layer_index=layer_index,
        )
        correction_hidden = next_hidden
    graph.finalize_joint_outputs(
        outputs_embeds=[correction_hidden, corrected[1]],
        runtime=correction_runtime,
    )

    posterior_trace = correction_context.posterior_memory
    assert posterior_trace is not None
    assert posterior_trace.layer_rows.requires_grad
    assert posterior_trace.layer_rows.grad_fn is not None
    posterior_trace.layer_rows.square().mean().backward()
    assert history_rows.grad is not None and history_rows.grad.abs().sum() > 0
    assert graph.control_projection.weight.grad is not None
    assert graph.control_projection.weight.grad.abs().sum() > 0
    committed = clone_persistent_state(posterior_trace)
    assert isinstance(committed, NativeLayerwisePosteriorState)
    assert not committed.layer_rows.requires_grad
    assert committed.layer_rows.data_ptr() != posterior_trace.layer_rows.data_ptr()


def test_unified_correction_rejects_every_raw_previous_posterior_surface() -> None:
    graph = _unified_graph()
    previous = NativeLayerwisePosteriorState(torch.randn(1, 3, 2, 8))
    direct = LingBotNativeContext(
        controls=_controls(),
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ),
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2]),
        previous_memory=previous,
    )
    inputs, mask, positions, visual = _inputs()
    with pytest.raises(ValueError, match="cannot read a previous posterior directly"):
        graph.prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=direct,
        )

    missing = _context()
    with pytest.raises(ValueError, match="requires a completed layerwise prior trace"):
        graph.prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=missing,
        )

    with pytest.raises(TypeError, match="transient v3 prior schema"):
        LingBotNativeContext(
            controls=_controls(),
            prior_trace=previous,  # type: ignore[arg-type]
        )


def test_unified_prior_stepper_supports_none_and_explicitly_invalid_reset_memory() -> None:
    graph = _unified_graph()
    seen_validity: list[torch.Tensor] = []
    policy = _toy_unified_prior_policy(graph, seen_validity=seen_validity)
    stepper = LingBotNativePriorStepper(policy, graph)  # type: ignore[arg-type]
    controls = _controls()

    none_reset = stepper(None, controls)
    assert isinstance(none_reset, NativeLayerwisePriorTrace)
    placeholder_rows = torch.full((1, 3, 2, 8), 17.0, requires_grad=True)
    placeholder = NativeLayerwisePosteriorState(placeholder_rows)
    masked_reset = stepper(
        placeholder,
        controls,
        previous_memory_valid=torch.tensor([False]),
    )
    assert isinstance(masked_reset, NativeLayerwisePriorTrace)
    torch.testing.assert_close(masked_reset.layer_rows, none_reset.layer_rows)
    masked_reset.layer_rows.sum().backward()
    assert placeholder_rows.grad is not None
    torch.testing.assert_close(placeholder_rows.grad, torch.zeros_like(placeholder_rows))

    continuation = stepper(
        placeholder,
        controls,
        previous_memory_valid=torch.tensor([True]),
    )
    assert isinstance(continuation, NativeLayerwisePriorTrace)
    assert not torch.equal(continuation.layer_rows, none_reset.layer_rows)
    assert [valid.tolist() for valid in seen_validity] == [[False], [False], [True]]


def test_unified_cold_prior_uses_runtime_control_dtype_before_fsdp_compute_cast() -> None:
    graph = _unified_graph().float()
    base_policy = _toy_unified_prior_policy(graph)
    observed_prefix_dtypes: list[torch.dtype] = []

    def mixed_precision_prior_forward(**kwargs: object):
        inputs = kwargs["inputs_embeds"]
        assert isinstance(inputs, list) and isinstance(inputs[0], torch.Tensor)
        observed_prefix_dtypes.append(inputs[0].dtype)
        graph.to(dtype=torch.bfloat16)
        return base_policy.picf_native_prior_forward(**kwargs)

    policy = SimpleNamespace(
        model=base_policy.model,
        picf_native_prior_forward=mixed_precision_prior_forward,
    )
    trace = LingBotNativePriorStepper(policy, graph)(  # type: ignore[arg-type]
        None,
        _controls(dtype=torch.bfloat16),
    )

    assert isinstance(trace, NativeLayerwisePriorTrace)
    assert trace.layer_rows.dtype == torch.bfloat16
    assert observed_prefix_dtypes == [torch.bfloat16]


def test_unified_step_with_prediction_accepts_current_prior_and_future_evidence() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            predictive_target_widths=(("probe", 4),),
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
        )
    ).train()
    policy = _toy_unified_prior_policy(graph)
    stepper = LingBotNativePriorStepper(policy, graph)  # type: ignore[arg-type]

    for evidence, horizon in (
        (PredictionEvidence.CURRENT_PRIOR, 0),
        (PredictionEvidence.FUTURE, 2),
    ):
        request = NativePredictionRequest(
            source=PredictionSource.PRIOR,
            evidence=evidence,
            route_ids=torch.zeros(1, 1, dtype=torch.long),
            horizons=torch.full((1, 1), horizon, dtype=torch.long),
            addresses=torch.empty(1, 1, 0),
            valid=torch.ones(1, 1, dtype=torch.bool),
        )
        trace, prediction = stepper.step_with_prediction(
            None,
            _controls(),
            request,
            target_name="probe",
        )
        assert isinstance(trace, NativeLayerwisePriorTrace)
        assert trace.layer_rows.requires_grad
        assert prediction.shape == (1, 2, 1, 4)


def test_prediction_evidence_is_fail_closed_by_architecture_and_host_pass() -> None:
    current_prior = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.CURRENT_PRIOR,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    legacy_prior_context = LingBotPriorRolloutContext(
        controls=_controls(),
        previous_state=NativePosteriorState(torch.randn(1, 2, 8)),
        prediction_request=current_prior,
    )
    empty = torch.empty(1, 0, 8)
    with pytest.raises(ValueError, match="legacy prior-only prediction requires FUTURE"):
        _graph().train().prepare_joint_inputs(
            inputs_embeds=[empty, None],
            attention_mask=torch.empty(1, 0, 0, dtype=torch.bool),
            position_ids=torch.empty(3, 1, 0, dtype=torch.long),
            visual_pos_masks=torch.empty(1, 0, dtype=torch.bool),
            context=legacy_prior_context,
        )

    trace = NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8, requires_grad=True))
    current_posterior = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.CURRENT_POSTERIOR,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=trace,
        prediction_request=current_posterior,
    )
    context.bind_native_prefix(
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        visual_sensor_mask=torch.tensor([[True, True, False]]),
        language_start=2,
        language_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _positions, _visual, runtime = (
        _unified_graph()
        .train()
        .prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=host_mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=context,
        )
    )
    assert prepared[0] is not None
    assert runtime.prediction_slice is not None
    for row_index in range(2):
        query = runtime.prediction_slice.start + row_index
        visible = set(mask[0, query].nonzero().flatten().tolist())
        assert runtime.posterior_slice.start + row_index in visible
        assert runtime.prior_slice.start + row_index not in visible

    current_correction = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.CURRENT_CORRECTION,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    rejected = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=trace,
        prediction_request=current_correction,
    )
    rejected.bind_native_prefix(
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        visual_sensor_mask=torch.tensor([[True, True, False]]),
        language_start=2,
        language_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="CURRENT_CORRECTION evidence is reserved for v2"):
        _unified_graph().train().prepare_joint_inputs(
            inputs_embeds=_inputs()[0],
            attention_mask=_inputs()[1],
            position_ids=_inputs()[2],
            visual_pos_masks=_inputs()[3],
            context=rejected,
        )

    current_prior_full = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=trace,
        prediction_request=current_prior,
    )
    current_prior_full.bind_native_prefix(
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        visual_sensor_mask=torch.tensor([[True, True, False]]),
        language_start=2,
        language_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="CURRENT_PRIOR evidence requires the v3 prior-only"):
        _unified_graph().train().prepare_joint_inputs(
            inputs_embeds=_inputs()[0],
            attention_mask=_inputs()[1],
            position_ids=_inputs()[2],
            visual_pos_masks=_inputs()[3],
            context=current_prior_full,
        )

    v2 = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            architecture_identity=LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
        )
    ).train()
    v2_replay = LingBotNativeContext(
        controls=_controls(),
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ),
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2]),
        prediction_request=current_correction,
    )
    _prepared, _mask, _positions, _visual, v2_runtime = v2.prepare_joint_inputs(
        inputs_embeds=_inputs()[0],
        attention_mask=_inputs()[1],
        position_ids=_inputs()[2],
        visual_pos_masks=_inputs()[3],
        context=v2_replay,
    )
    assert v2_runtime.prediction_slice is not None

    v2_new_evidence = LingBotNativeContext(
        controls=_controls(),
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ),
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2]),
        prediction_request=current_posterior,
    )
    with pytest.raises(ValueError, match="requires the v3 architecture"):
        v2.prepare_joint_inputs(
            inputs_embeds=_inputs()[0],
            attention_mask=_inputs()[1],
            position_ids=_inputs()[2],
            visual_pos_masks=_inputs()[3],
            context=v2_new_evidence,
        )


def test_native_graph_has_no_fsdp2_incompatible_scalar_trainables() -> None:
    graph = _graph()
    scalar_parameters = [
        name for name, parameter in graph.named_parameters() if parameter.ndim == 0
    ]
    assert scalar_parameters == []


def _official_policy_contract(
    *,
    host_width: int = 8,
    action_dim: int = 2,
    num_layers: int = 3,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    language_model = SimpleNamespace(
        layers=tuple(SimpleNamespace(hidden_size=host_width) for _ in range(num_layers)),
        config=SimpleNamespace(initializer_range=0.02),
    )
    action_model = SimpleNamespace(layers=tuple(object() for _ in range(num_layers)))
    host = SimpleNamespace(
        qwenvl=SimpleNamespace(model=SimpleNamespace(language_model=language_model)),
        qwen_expert=SimpleNamespace(model=action_model),
    )

    def install(graph: LingBotNativeGraph) -> None:
        host.picf_native_graph = graph

    host.set_picf_native_graph = install
    policy = SimpleNamespace(
        model=SimpleNamespace(
            qwenvl_with_expert=host,
            config=SimpleNamespace(max_action_dim=action_dim),
        )
    )
    return policy, host


def test_graph_config_and_installation_match_the_official_host_contract() -> None:
    policy, host = _official_policy_contract()
    graph = _graph()

    config = LingBotNativeGraphConfig.from_policy(
        policy,
        capacity=graph.config.capacity,
        maximum_control_tokens=graph.config.maximum_control_tokens,
    )
    assert (config.host_width, config.executed_action_dim, config.num_layers) == (8, 2, 3)
    assert config.object_transition == CONTENT_ADDRESSED_SET_TRANSITION
    install_lingbot_native_graph(policy, graph)
    assert host.picf_native_graph is graph

    with pytest.raises(ValueError, match="object transition is unsupported"):
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            object_transition="fixed_row_pointer_v1",
        )


def test_native_videomt_graph_installation_preserves_complete_spatial_relation() -> None:
    policy, host = _official_policy_contract()
    spatial = NativeObjectQuerySpatialSpec(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout="videomt.calvin.static.2x2.v1",
    )
    graph = LingBotNativeGraph(
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
            object_query_spatial_specs=(spatial,),
            architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
        )
    )

    install_lingbot_native_graph(policy, graph)

    assert host.picf_native_graph is graph
    assert graph.config.object_query_spatial_specs == (spatial,)


def test_relation_supervision_layers_are_sorted_unique_nonfinal_host_depths() -> None:
    common = {
        "capacity": 2,
        "host_width": 8,
        "executed_action_dim": 2,
        "num_layers": 4,
    }
    config = LingBotNativeGraphConfig(
        **common,
        relation_supervision_layers=(0, 2),
    )
    assert config.relation_supervision_layers == (0, 2)
    with pytest.raises(ValueError, match="sorted and unique"):
        LingBotNativeGraphConfig(
            **common,
            relation_supervision_layers=(2, 0),
        )
    with pytest.raises(ValueError, match="non-final"):
        LingBotNativeGraphConfig(
            **common,
            relation_supervision_layers=(3,),
        )
    with pytest.raises(TypeError, match="integers"):
        LingBotNativeGraphConfig(
            **common,
            relation_supervision_layers=(True,),
        )


def test_graph_installation_rejects_an_incompatible_or_unpatched_host() -> None:
    policy, _host = _official_policy_contract(host_width=16)
    with pytest.raises(ValueError, match="host_width"):
        install_lingbot_native_graph(policy, _graph())

    unpatched, host = _official_policy_contract()
    del host.set_picf_native_graph
    with pytest.raises(TypeError, match="set_picf_native_graph"):
        install_lingbot_native_graph(unpatched, _graph())


def _shared_host_layers(
    hidden: torch.Tensor, mask: torch.Tensor, *, layers: int = 3
) -> torch.Tensor:
    """Small deterministic proxy for the official shared Transformer stack."""

    width = hidden.shape[-1]
    output = hidden
    for layer in range(layers):
        generator = torch.Generator().manual_seed(100 + layer)
        weight = torch.randn(width, width, generator=generator) / width**0.5
        query = output @ weight
        score = query @ query.transpose(-1, -2) / width**0.5
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        attended = torch.softmax(score, dim=-1) @ output
        output = torch.nn.functional.layer_norm(output + attended, (width,))
        output = torch.nn.functional.layer_norm(output + torch.tanh(output @ weight), (width,))
    return output


def test_none_context_is_exact_host_identity() -> None:
    graph = _graph()
    inputs, mask, positions, visual = _inputs()
    returned = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=None,
    )
    assert returned == (inputs, mask, positions, visual, None)


def test_official_prefix_roles_are_bound_once_from_exact_host_layout() -> None:
    context = LingBotNativeContext(controls=_controls())
    graph = _graph()
    inputs, mask, positions, visual = _inputs()
    with pytest.raises(ValueError, match="before prefix binding"):
        graph.prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=context,
        )

    native_valid = torch.tensor([[True, True, True, True, True, False]])
    visual_sensor = torch.tensor([[False, True, False, False, False, False]])
    language_valid = torch.tensor([[True, True]])
    context.bind_native_prefix(
        native_valid=native_valid,
        visual_sensor_mask=visual_sensor,
        language_start=2,
        language_valid=language_valid,
    )
    assert context.native_roles is not None
    assert context.native_roles.tolist() == [
        [
            int(NativeRole.HOST_AUX),
            int(NativeRole.SENSOR),
            int(NativeRole.LANGUAGE),
            int(NativeRole.LANGUAGE),
            int(NativeRole.HOST_AUX),
            int(NativeRole.HOST_AUX),
        ]
    ]
    assert context.instruction_last_index is not None
    assert context.instruction_last_index.tolist() == [3]
    with pytest.raises(RuntimeError, match="only once"):
        context.bind_native_prefix(
            native_valid=native_valid,
            visual_sensor_mask=visual_sensor,
            language_start=2,
            language_valid=language_valid,
        )


def test_prepare_inserts_control_rows_and_ephemeral_match_tokens_with_identity_mrope() -> None:
    graph = _graph()
    previous = NativePosteriorState(torch.randn(1, 2, 8))
    context = _context(previous)
    inputs, mask, positions, visual = _inputs()
    prepared, unified_mask, expanded_positions, expanded_visual, runtime = (
        graph.prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=context,
        )
    )
    assert runtime is not None
    assert prepared[0].shape == (1, 10, 8)
    assert prepared[1].shape == inputs[1].shape
    assert unified_mask.shape == (1, 12, 12)
    assert not expanded_positions[:, :, 3:10].any()
    assert expanded_visual is not None and expanded_visual.shape == (1, 10)
    assert not expanded_visual[:, 3:].any()
    assert set(runtime.layout.roles[0, 3:10].tolist()) == {
        int(NativeRole.CONTROL),
        int(NativeRole.PRIOR),
        int(NativeRole.POSTERIOR),
        int(NativeRole.MATCH),
    }
    assert runtime.match_slice == slice(8, 10)
    expected_match_sources = [
        0,
        1,
        2,
        *range(runtime.posterior_slice.start, runtime.posterior_slice.stop),
        *range(runtime.match_slice.start, runtime.match_slice.stop),
    ]
    for query in range(runtime.match_slice.start, runtime.match_slice.stop):
        assert unified_mask[0, query].nonzero().flatten().tolist() == expected_match_sources
    match_keys = unified_mask[0, :, runtime.match_slice]
    expected_match_keys = torch.zeros_like(match_keys)
    expected_match_keys[runtime.match_slice] = True
    expected_match_keys[prepared[0].shape[1] :, :] = True
    assert torch.equal(match_keys, expected_match_keys)
    prior = prepared[0][:, runtime.prior_slice]
    context_without_state = _context()
    prepared_empty, *_ = graph.prepare_joint_inputs(
        inputs_embeds=_inputs()[0],
        attention_mask=_inputs()[1],
        position_ids=_inputs()[2],
        visual_pos_masks=_inputs()[3],
        context=context_without_state,
    )
    empty_prior = prepared_empty[0][:, runtime.prior_slice]
    assert not torch.equal(prior, empty_prior)


def test_content_seed_selects_history_for_continuation_and_queries_for_reset() -> None:
    graph = _graph()
    previous = NativePosteriorState(torch.randn(2, 2, 8))
    context = LingBotNativeContext(
        controls=_controls(batch=2),
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ).expand(2, -1),
        native_valid=torch.ones(2, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2, 2]),
        previous_state=previous,
        previous_state_valid=torch.tensor([True, False]),
    )
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[inputs[0].expand(2, -1, -1).clone(), inputs[1].expand(2, -1, -1).clone()],
        attention_mask=host_mask.expand(2, -1, -1).clone(),
        position_ids=positions.expand(-1, 2, -1).clone(),
        visual_pos_masks=visual.expand(2, -1).clone(),
        context=context,
    )
    assert prepared[0].shape[0] == 2
    normalized = torch.nn.functional.layer_norm(previous.rows, (graph.config.host_width,))
    expected_content = torch.stack((normalized[0], graph.object_queries), dim=0)
    for target_slice, role_index in (
        (runtime.prior_slice, 0),
        (runtime.posterior_slice, 1),
        (runtime.match_slice, 3),
    ):
        torch.testing.assert_close(
            prepared[0][:, target_slice],
            expected_content + graph.role_embeddings[role_index],
        )
    assert mask[:, runtime.prior_slice, runtime.prior_slice].all()
    assert mask[:, runtime.posterior_slice, runtime.prior_slice].all()
    assert mask[:, runtime.posterior_slice, runtime.posterior_slice].all()


def test_reset_discovery_rows_are_distinct_and_receive_gradient() -> None:
    graph = _graph()
    inputs, host_mask, positions, visual = _inputs()
    prepared, _, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=_context(),
    )
    reset_content = prepared[0][:, runtime.posterior_slice] - graph.role_embeddings[1]
    torch.testing.assert_close(reset_content, graph.object_queries.unsqueeze(0))
    assert not torch.equal(reset_content[:, 0], reset_content[:, 1])
    reset_content.square().sum().backward()
    assert graph.object_queries.grad is not None
    assert (graph.object_queries.grad.abs().sum(dim=-1) > 0).all()


def test_continuation_rows_use_complete_object_sets_without_fixed_row_pointers() -> None:
    graph = _graph()
    context = _context(NativePosteriorState(torch.randn(1, 2, 8)))
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    prefix_mask = mask[:, : prepared[0].shape[1], : prepared[0].shape[1]]
    baseline = _shared_host_layers(prepared[0], prefix_mask, layers=1)
    delta = torch.linspace(-3.0, 3.0, 8)

    off_row_input = prepared[0].clone()
    off_row_input[:, runtime.prior_slice.start + 1] += delta
    off_row_one = _shared_host_layers(off_row_input, prefix_mask, layers=1)
    assert not torch.allclose(
        baseline[:, runtime.posterior_slice.start],
        off_row_one[:, runtime.posterior_slice.start],
    )
    assert not torch.allclose(
        baseline[:, runtime.prior_slice.start],
        off_row_one[:, runtime.prior_slice.start],
    )

    sensor_input = prepared[0].clone()
    sensor_input[:, 0] += delta
    sensor_output = _shared_host_layers(sensor_input, prefix_mask, layers=1)
    assert not torch.allclose(
        baseline[:, runtime.posterior_slice.start],
        sensor_output[:, runtime.posterior_slice.start],
    )


def test_match_reads_complete_posterior_set_in_one_layer() -> None:
    graph = _graph()
    context = _context(NativePosteriorState(torch.randn(1, 2, 8)))
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    prefix_mask = mask[:, : prepared[0].shape[1], : prepared[0].shape[1]]
    baseline_one = _shared_host_layers(prepared[0], prefix_mask, layers=1)
    delta = torch.linspace(-3.0, 3.0, 8)
    intervened_input = prepared[0].clone()
    intervened_input[:, runtime.posterior_slice.start] += delta
    intervened_one = _shared_host_layers(intervened_input, prefix_mask, layers=1)

    assert not torch.allclose(
        baseline_one[:, runtime.match_slice.start],
        intervened_one[:, runtime.match_slice.start],
    )
    assert not torch.allclose(
        baseline_one[:, runtime.match_slice.start + 1],
        intervened_one[:, runtime.match_slice.start + 1],
    )


def test_continuation_is_equivariant_when_only_previous_state_is_permuted() -> None:
    graph = _graph()
    permutation = torch.tensor([1, 0])

    previous_a = NativePosteriorState(torch.randn(1, 2, 8))
    previous_b = previous_a.permute_rows(permutation)
    inputs, host_mask, positions, visual = _inputs()
    context_a = _context(previous_a)
    context_b = _context(previous_b)
    prepared_a, mask_a, _, _, runtime_a = graph.prepare_joint_inputs(
        inputs_embeds=[value.clone() for value in inputs],
        attention_mask=host_mask.clone(),
        position_ids=positions.clone(),
        visual_pos_masks=visual.clone(),
        context=context_a,
    )
    prepared_b, mask_b, _, _, runtime_b = graph.prepare_joint_inputs(
        inputs_embeds=[value.clone() for value in inputs],
        attention_mask=host_mask.clone(),
        position_ids=positions.clone(),
        visual_pos_masks=visual.clone(),
        context=context_b,
    )
    prefix_count = prepared_a[0].shape[1]
    output_a = _shared_host_layers(
        prepared_a[0],
        mask_a[:, :prefix_count, :prefix_count],
    )
    output_b = _shared_host_layers(
        prepared_b[0],
        mask_b[:, :prefix_count, :prefix_count],
    )

    for slice_a, slice_b in (
        (runtime_a.prior_slice, runtime_b.prior_slice),
        (runtime_a.posterior_slice, runtime_b.posterior_slice),
        (runtime_a.match_slice, runtime_b.match_slice),
    ):
        torch.testing.assert_close(
            output_b[:, slice_b],
            output_a[:, slice_a].index_select(1, permutation),
            rtol=1e-5,
            atol=1e-6,
        )

    graph.finalize_joint_outputs(
        outputs_embeds=[output_a, prepared_a[1]],
        runtime=runtime_a,
    )
    graph.finalize_joint_outputs(
        outputs_embeds=[output_b, prepared_b[1]],
        runtime=runtime_b,
    )
    assert context_a.relation_output is not None
    assert context_b.relation_output is not None
    relation_a = context_a.relation_output
    relation_b = context_b.relation_output
    torch.testing.assert_close(
        relation_b.ownership[..., :-1],
        relation_a.ownership[..., :-1].index_select(-1, permutation),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        relation_b.ownership[..., -1],
        relation_a.ownership[..., -1],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        relation_b.task_row_probability,
        relation_a.task_row_probability.index_select(-1, permutation),
        rtol=1e-5,
        atol=1e-6,
    )


def test_reset_output_is_a_set_under_discovery_query_reordering() -> None:
    graph_a = _graph()
    graph_b = _graph()
    graph_b.load_state_dict(graph_a.state_dict(), strict=True)
    permutation = torch.tensor([1, 0])
    with torch.no_grad():
        graph_b.object_queries.copy_(graph_a.object_queries.index_select(0, permutation))

    inputs, host_mask, positions, visual = _inputs()
    outputs = []
    runtimes = []
    for graph in (graph_a, graph_b):
        prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
            inputs_embeds=[value.clone() for value in inputs],
            attention_mask=host_mask.clone(),
            position_ids=positions.clone(),
            visual_pos_masks=visual.clone(),
            context=_context(),
        )
        outputs.append(
            _shared_host_layers(
                prepared[0],
                mask[:, : prepared[0].shape[1], : prepared[0].shape[1]],
            )
        )
        runtimes.append(runtime)
    for target_slice_a, target_slice_b in (
        (runtimes[0].prior_slice, runtimes[1].prior_slice),
        (runtimes[0].posterior_slice, runtimes[1].posterior_slice),
        (runtimes[0].match_slice, runtimes[1].match_slice),
    ):
        torch.testing.assert_close(
            outputs[1][:, target_slice_b],
            outputs[0][:, target_slice_a].index_select(1, permutation),
            rtol=1e-5,
            atol=1e-6,
        )


def test_match_reads_every_valid_language_token_but_no_padding_or_host_aux() -> None:
    graph = _graph()
    prefix = torch.randn(1, 5, 8)
    action = torch.randn(1, 2, 4)
    host_mask = torch.ones(1, 7, 7, dtype=torch.bool)
    positions = torch.arange(7).reshape(1, 1, 7).expand(3, 1, 7).clone()
    visual = torch.tensor([[True, False, False, False, False]])
    context = LingBotNativeContext(
        controls=_controls(),
        native_roles=torch.tensor(
            [
                [
                    int(NativeRole.SENSOR),
                    int(NativeRole.LANGUAGE),
                    int(NativeRole.LANGUAGE),
                    int(NativeRole.LANGUAGE),
                    int(NativeRole.HOST_AUX),
                ]
            ]
        ),
        native_valid=torch.tensor([[True, True, True, False, True]]),
        instruction_last_index=torch.tensor([2]),
    )

    _, unified_mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )

    for row_index in range(graph.config.capacity):
        query = runtime.match_slice.start + row_index
        assert unified_mask[0, query].nonzero().flatten().tolist() == [
            0,
            1,
            2,
            *range(runtime.posterior_slice.start, runtime.posterior_slice.stop),
            *range(runtime.match_slice.start, runtime.match_slice.stop),
        ]


def test_complete_prompt_changes_match_but_cannot_write_persistent_posterior() -> None:
    graph = _graph()
    inputs_a, host_mask, positions, visual = _inputs()
    inputs_b = [inputs_a[0].clone(), inputs_a[1].clone()]
    inputs_b[0][:, 2] += 7.0
    context_a = _context()
    context_b = _context()
    prepared_a, mask_a, _, _, runtime_a = graph.prepare_joint_inputs(
        inputs_embeds=inputs_a,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context_a,
    )
    prepared_b, mask_b, _, _, runtime_b = graph.prepare_joint_inputs(
        inputs_embeds=inputs_b,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context_b,
    )
    output_a = _shared_host_layers(
        prepared_a[0],
        mask_a[:, : prepared_a[0].shape[1], : prepared_a[0].shape[1]],
    )
    output_b = _shared_host_layers(
        prepared_b[0],
        mask_b[:, : prepared_b[0].shape[1], : prepared_b[0].shape[1]],
    )

    assert torch.equal(
        output_a[:, runtime_a.posterior_slice],
        output_b[:, runtime_b.posterior_slice],
    )
    assert not torch.equal(
        output_a[:, runtime_a.match_slice],
        output_b[:, runtime_b.match_slice],
    )
    graph.finalize_joint_outputs(
        outputs_embeds=[output_a, prepared_a[1]],
        runtime=runtime_a,
    )
    graph.finalize_joint_outputs(
        outputs_embeds=[output_b, prepared_b[1]],
        runtime=runtime_b,
    )
    assert context_a.relation_output is not None
    assert context_b.relation_output is not None
    assert torch.equal(
        context_a.relation_output.ownership,
        context_b.relation_output.ownership,
    )
    assert not torch.equal(
        context_a.relation_output.task_relevance_logits,
        context_b.relation_output.task_relevance_logits,
    )


def test_sensor_intervention_updates_posterior_and_prompt_object_match() -> None:
    graph = _graph()
    inputs_a, host_mask, positions, visual = _inputs()
    inputs_b = [inputs_a[0].clone(), inputs_a[1].clone()]
    inputs_b[0][:, 0] -= 5.0
    prepared_a, mask_a, _, _, runtime_a = graph.prepare_joint_inputs(
        inputs_embeds=inputs_a,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=_context(),
    )
    prepared_b, mask_b, _, _, runtime_b = graph.prepare_joint_inputs(
        inputs_embeds=inputs_b,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=_context(),
    )
    output_a = _shared_host_layers(
        prepared_a[0],
        mask_a[:, : prepared_a[0].shape[1], : prepared_a[0].shape[1]],
    )
    output_b = _shared_host_layers(
        prepared_b[0],
        mask_b[:, : prepared_b[0].shape[1], : prepared_b[0].shape[1]],
    )

    assert not torch.equal(
        output_a[:, runtime_a.posterior_slice],
        output_b[:, runtime_b.posterior_slice],
    )
    assert not torch.equal(
        output_a[:, runtime_a.match_slice],
        output_b[:, runtime_b.match_slice],
    )


def test_executed_control_intervention_updates_physical_posterior() -> None:
    graph = _graph()
    inputs, host_mask, positions, visual = _inputs()
    previous = NativePosteriorState(torch.randn(1, 2, 8))
    controls_a = _controls()
    controls_b = _controls()
    controls_b.values.add_(3.0)
    contexts = (
        LingBotNativeContext(
            controls=controls_a,
            native_roles=_context().native_roles,
            native_valid=_context().native_valid,
            instruction_last_index=_context().instruction_last_index,
            previous_state=previous,
        ),
        LingBotNativeContext(
            controls=controls_b,
            native_roles=_context().native_roles,
            native_valid=_context().native_valid,
            instruction_last_index=_context().instruction_last_index,
            previous_state=previous,
        ),
    )
    outputs = []
    runtimes = []
    for context in contexts:
        prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
            inputs_embeds=[value.clone() for value in inputs],
            attention_mask=host_mask.clone(),
            position_ids=positions.clone(),
            visual_pos_masks=visual.clone(),
            context=context,
        )
        outputs.append(
            _shared_host_layers(
                prepared[0],
                mask[:, : prepared[0].shape[1], : prepared[0].shape[1]],
            )
        )
        runtimes.append(runtime)
    assert not torch.equal(
        outputs[0][:, runtimes[0].posterior_slice],
        outputs[1][:, runtimes[1].posterior_slice],
    )


def test_finalize_serializes_final_rows_and_relation_head_cannot_write_them() -> None:
    graph = _graph()
    context = _context()
    with pytest.raises(RuntimeError, match="incomplete or not finalized"):
        context.root_output_tensors()
    inputs, mask, positions, visual = _inputs()
    prepared, _, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    outputs = [prepared[0] + 0.5, prepared[1]]
    expected = outputs[0][:, runtime.posterior_slice].clone()
    graph.finalize_joint_outputs(outputs_embeds=outputs, runtime=runtime)
    assert context.prior_state is not None
    assert context.posterior_state is not None
    assert torch.equal(context.posterior_state.rows, expected)
    assert context.relation_output is not None
    relation_snapshot = context.relation_output.existence.detach().clone()
    assert torch.equal(context.posterior_state.rows, expected)
    assert torch.equal(relation_snapshot, context.relation_output.existence)
    root_outputs = context.root_output_tensors()
    assert len(root_outputs) == 19
    assert root_outputs[0] is context.prior_state.rows
    assert root_outputs[1] is context.posterior_state.rows
    assert root_outputs[2] is context.relation_output.support_logits
    assert context.relation_output.task_embedding is None
    assert root_outputs[7] is context.relation_output.match_embeddings
    assert root_outputs[8] is context.relation_output.row_embeddings
    assert root_outputs[9] is context.relation_output.relation_temperature
    assert root_outputs[13] is context.relation_output.existence_logits
    assert root_outputs[14] is context.relation_output.task_object_log_probability
    assert root_outputs[15] is context.relation_output.task_object_probability
    assert root_outputs[16] is context.relation_output.task_event_distribution
    assert root_outputs[17] is context.relation_output.task_row_probability
    assert root_outputs[18] is context.relation_output.ownership_log_probability
    assert all(value.is_floating_point() for value in root_outputs)
    with pytest.raises(RuntimeError, match="only once"):
        graph.finalize_joint_outputs(outputs_embeds=outputs, runtime=runtime)


def test_intermediate_relations_share_one_readout_and_cross_the_root_boundary() -> None:
    torch.manual_seed(2)
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            relation_supervision_layers=(0, 1),
        )
    )
    baseline = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
        )
    )
    baseline_parameter_count = sum(parameter.numel() for parameter in baseline.parameters())
    assert sum(parameter.numel() for parameter in graph.parameters()) == baseline_parameter_count
    context = _context()
    context.supervise_intermediate_relations = True
    inputs, mask, positions, visual = _inputs()
    prepared, _, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    assert graph.requires_intermediate_relation(layer_index=0, runtime=runtime)
    assert graph.requires_intermediate_relation(layer_index=1, runtime=runtime)
    assert not graph.requires_intermediate_relation(layer_index=2, runtime=runtime)
    graph.record_intermediate_relation(
        normalized_prefix=prepared[0] + 0.1,
        runtime=runtime,
        layer_index=0,
    )
    graph.record_intermediate_relation(
        normalized_prefix=prepared[0] + 0.2,
        runtime=runtime,
        layer_index=1,
    )
    with pytest.raises(RuntimeError, match="only once"):
        graph.record_intermediate_relation(
            normalized_prefix=prepared[0] + 0.3,
            runtime=runtime,
            layer_index=1,
        )
    graph.finalize_joint_outputs(
        outputs_embeds=[prepared[0] + 0.5, prepared[1]],
        runtime=runtime,
    )
    assert tuple(context.intermediate_relation_outputs) == (0, 1)
    root_outputs = context.root_output_tensors()
    assert len(root_outputs) == 23
    assert root_outputs[-4] is context.intermediate_relation_outputs[0].ownership
    assert root_outputs[-3] is context.intermediate_relation_outputs[0].ownership_log_probability
    assert root_outputs[-2] is context.intermediate_relation_outputs[1].ownership
    assert root_outputs[-1] is context.intermediate_relation_outputs[1].ownership_log_probability
    assert all(
        relation.ownership.requires_grad
        for relation in context.intermediate_relation_outputs.values()
    )


def test_action_can_consume_match_tokens_without_match_writing_posterior() -> None:
    torch.manual_seed(37)
    graph = _graph()
    context = _context()
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    action_hidden = torch.nn.functional.pad(prepared[1], (0, 4))
    baseline_input = torch.cat((prepared[0], action_hidden), dim=1)
    intervened_input = baseline_input.clone()
    intervened_input[:, runtime.match_slice] += torch.linspace(-3.0, 3.0, 8)

    baseline = _shared_host_layers(baseline_input, mask, layers=1)
    intervened = _shared_host_layers(intervened_input, mask, layers=1)

    torch.testing.assert_close(
        baseline[:, runtime.posterior_slice],
        intervened[:, runtime.posterior_slice],
    )
    action_slice = slice(prepared[0].shape[1], baseline.shape[1])
    assert not torch.allclose(baseline[:, action_slice], intervened[:, action_slice])


def test_intermediate_relation_supervision_fails_on_missing_or_unexpected_depths() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            relation_supervision_layers=(0,),
        )
    )
    context = _context()
    context.supervise_intermediate_relations = True
    inputs, mask, positions, visual = _inputs()
    prepared, _, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    with pytest.raises(RuntimeError, match="differ from the configured"):
        graph.finalize_joint_outputs(
            outputs_embeds=[prepared[0], prepared[1]],
            runtime=runtime,
        )

    disabled = _context()
    prepared, _, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=_inputs()[0],
        attention_mask=_inputs()[1],
        position_ids=_inputs()[2],
        visual_pos_masks=_inputs()[3],
        context=disabled,
    )
    with pytest.raises(RuntimeError, match="unexpected intermediate"):
        graph.record_intermediate_relation(
            normalized_prefix=prepared[0],
            runtime=runtime,
            layer_index=0,
        )


def test_disabled_intermediate_supervision_is_bit_exact_to_the_original_graph() -> None:
    common = {
        "capacity": 2,
        "host_width": 8,
        "executed_action_dim": 2,
        "num_layers": 3,
        "maximum_control_tokens": 2,
    }
    baseline = LingBotNativeGraph(LingBotNativeGraphConfig(**common))
    candidate = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            **common,
            relation_supervision_layers=(0, 1),
        )
    )
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline_context = _context()
    candidate_context = _context()
    inputs, mask, positions, visual = _inputs()
    baseline_prepared, _, _, _, baseline_runtime = baseline.prepare_joint_inputs(
        inputs_embeds=[value.clone() for value in inputs],
        attention_mask=mask.clone(),
        position_ids=positions.clone(),
        visual_pos_masks=visual.clone(),
        context=baseline_context,
    )
    candidate_prepared, _, _, _, candidate_runtime = candidate.prepare_joint_inputs(
        inputs_embeds=[value.clone() for value in inputs],
        attention_mask=mask.clone(),
        position_ids=positions.clone(),
        visual_pos_masks=visual.clone(),
        context=candidate_context,
    )
    torch.testing.assert_close(
        baseline_prepared[0],
        candidate_prepared[0],
        rtol=0,
        atol=0,
    )
    baseline.finalize_joint_outputs(
        outputs_embeds=[baseline_prepared[0] + 0.5, baseline_prepared[1]],
        runtime=baseline_runtime,
    )
    candidate.finalize_joint_outputs(
        outputs_embeds=[candidate_prepared[0] + 0.5, candidate_prepared[1]],
        runtime=candidate_runtime,
    )
    assert candidate_context.intermediate_relation_outputs == {}
    assert baseline_context.relation_output is not None
    assert candidate_context.relation_output is not None
    for field in (
        "support_logits",
        "ownership",
        "task_relevance_logits",
        "dense_task_grounding_logits",
        "existence_logits",
    ):
        torch.testing.assert_close(
            getattr(baseline_context.relation_output, field),
            getattr(candidate_context.relation_output, field),
            rtol=0,
            atol=0,
        )


def test_enabled_intermediate_reads_do_not_mutate_the_final_posterior_or_relation() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            relation_supervision_layers=(0, 1),
        )
    )
    disabled_context = _context()
    enabled_context = _context()
    enabled_context.supervise_intermediate_relations = True
    inputs, mask, positions, visual = _inputs()
    disabled_prepared, _, _, _, disabled_runtime = graph.prepare_joint_inputs(
        inputs_embeds=[value.clone() for value in inputs],
        attention_mask=mask.clone(),
        position_ids=positions.clone(),
        visual_pos_masks=visual.clone(),
        context=disabled_context,
    )
    enabled_prepared, _, _, _, enabled_runtime = graph.prepare_joint_inputs(
        inputs_embeds=[value.clone() for value in inputs],
        attention_mask=mask.clone(),
        position_ids=positions.clone(),
        visual_pos_masks=visual.clone(),
        context=enabled_context,
    )
    graph.record_intermediate_relation(
        normalized_prefix=enabled_prepared[0] + 0.1,
        runtime=enabled_runtime,
        layer_index=0,
    )
    graph.record_intermediate_relation(
        normalized_prefix=enabled_prepared[0] + 0.2,
        runtime=enabled_runtime,
        layer_index=1,
    )
    disabled_outputs = [disabled_prepared[0] + 0.5, disabled_prepared[1]]
    enabled_outputs = [enabled_prepared[0] + 0.5, enabled_prepared[1]]
    graph.finalize_joint_outputs(
        outputs_embeds=disabled_outputs,
        runtime=disabled_runtime,
    )
    graph.finalize_joint_outputs(
        outputs_embeds=enabled_outputs,
        runtime=enabled_runtime,
    )
    assert disabled_context.posterior_state is not None
    assert enabled_context.posterior_state is not None
    torch.testing.assert_close(
        disabled_context.posterior_state.rows,
        enabled_context.posterior_state.rows,
        rtol=0,
        atol=0,
    )
    assert disabled_context.relation_output is not None
    assert enabled_context.relation_output is not None
    for field in (
        "support_logits",
        "visible_support",
        "ownership",
        "task_relevance",
        "task_relevance_logits",
        "dense_task_grounding",
        "dense_task_grounding_logits",
        "existence",
        "existence_logits",
        "sensor_valid",
    ):
        torch.testing.assert_close(
            getattr(disabled_context.relation_output, field),
            getattr(enabled_context.relation_output, field),
            rtol=0,
            atol=0,
        )


def test_intermediate_ownership_gradients_reach_early_middle_and_late_host_layers() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=4,
            maximum_control_tokens=2,
            relation_supervision_layers=(0, 1, 2),
        )
    )
    context = _context()
    context.supervise_intermediate_relations = True
    inputs, mask, positions, visual = _inputs()
    prepared, _, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    layers = torch.nn.ModuleList(torch.nn.Linear(8, 8, bias=False) for _ in range(4))
    with torch.no_grad():
        for layer in layers:
            layer.weight.copy_(torch.eye(8))
    hidden = prepared[0]
    for layer_index, layer in enumerate(layers):
        hidden = checkpoint(
            lambda value, host_layer=layer: torch.tanh(host_layer(value)),
            hidden,
            use_reentrant=False,
        )
        if graph.requires_intermediate_relation(
            layer_index=layer_index,
            runtime=runtime,
        ):
            graph.record_intermediate_relation(
                normalized_prefix=torch.nn.functional.layer_norm(hidden, (8,)),
                runtime=runtime,
                layer_index=layer_index,
            )
    graph.finalize_joint_outputs(
        outputs_embeds=[
            torch.nn.functional.layer_norm(hidden, (8,)),
            prepared[1],
        ],
        runtime=runtime,
    )
    relations = tuple(context.intermediate_relation_outputs.values())
    loss = sum(relation.ownership[:, :, 0].square().mean() for relation in relations)
    loss.backward()
    assert all(
        layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0 for layer in layers[:3]
    )
    assert layers[3].weight.grad is None


def test_prompt_content_cannot_enter_prior_or_posterior_input_rows() -> None:
    graph = _graph()
    context_a = _context()
    context_b = _context()
    inputs_a, mask, positions, visual = _inputs()
    inputs_b = [inputs_a[0].clone(), inputs_a[1].clone()]
    inputs_b[0][:, 2] += 1000
    prepared_a, _, _, _, runtime_a = graph.prepare_joint_inputs(
        inputs_embeds=inputs_a,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context_a,
    )
    prepared_b, _, _, _, runtime_b = graph.prepare_joint_inputs(
        inputs_embeds=inputs_b,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context_b,
    )
    assert torch.equal(
        prepared_a[0][:, runtime_a.prior_slice],
        prepared_b[0][:, runtime_b.prior_slice],
    )
    assert torch.equal(
        prepared_a[0][:, runtime_a.posterior_slice],
        prepared_b[0][:, runtime_b.posterior_slice],
    )


def test_current_action_target_cannot_enter_prior_or_posterior_input_rows() -> None:
    graph = _graph()
    context_a = _context()
    context_b = _context()
    inputs_a, mask, positions, visual = _inputs()
    inputs_b = [inputs_a[0].clone(), inputs_a[1].clone() + 1000]
    prepared_a, _, _, _, runtime_a = graph.prepare_joint_inputs(
        inputs_embeds=inputs_a,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context_a,
    )
    prepared_b, _, _, _, runtime_b = graph.prepare_joint_inputs(
        inputs_embeds=inputs_b,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context_b,
    )
    assert torch.equal(
        prepared_a[0][:, runtime_a.prior_slice],
        prepared_b[0][:, runtime_b.prior_slice],
    )
    assert torch.equal(
        prepared_a[0][:, runtime_a.posterior_slice],
        prepared_b[0][:, runtime_b.posterior_slice],
    )


def test_optional_modalities_use_only_linear_bridges_into_the_shared_host() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(
                NativeModalitySpec("geometry", 4, 1, metadata_width=3),
                NativeModalitySpec("touch", 3, 2),
            ),
        )
    )
    modalities = NativeModalityBatch(
        (
            NativeModalityStream(
                "geometry",
                torch.randn(1, 1, 4),
                torch.ones(1, 1, dtype=torch.bool),
                metadata=torch.randn(1, 1, 3),
            ),
            NativeModalityStream(
                "touch",
                torch.randn(1, 2, 3),
                torch.tensor([[True, False]]),
            ),
        )
    )
    context = LingBotNativeContext(
        controls=_controls(),
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ),
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2]),
        modalities=modalities,
    )
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    assert runtime is not None
    assert runtime.modality_slice == slice(3, 6)
    assert prepared[0].shape == (1, 13, 8)
    assert mask.shape == (1, 15, 15)
    assert runtime.layout.roles[0, runtime.modality_slice].tolist() == [
        int(NativeRole.SENSOR),
        int(NativeRole.SENSOR),
        int(NativeRole.SENSOR),
    ]
    assert runtime.layout.valid[0, runtime.modality_slice].tolist() == [True, True, False]
    assert all(
        isinstance(module, torch.nn.Linear) for module in graph.modality_projections.values()
    )
    assert set(graph.modality_metadata_projections) == {"geometry"}
    assert isinstance(graph.modality_metadata_projections["geometry"], torch.nn.Linear)

    prefix_count = prepared[0].shape[1]
    shared = _shared_host_layers(prepared[0], mask[:, :prefix_count, :prefix_count])
    graph.finalize_joint_outputs(outputs_embeds=[shared, prepared[1]], runtime=runtime)
    assert context.relation_output is not None
    assert context.relation_output.sensor_valid.shape == (1, 6)
    assert context.relation_output.sensor_valid.sum().item() == 4


def test_action_gradient_reaches_every_dense_modality_through_the_shared_host() -> None:
    """The action path must consume modalities through LingBot, not an auxiliary head."""

    torch.manual_seed(73)
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(
                NativeModalitySpec("anytouch", 3, 2, metadata_width=3),
                NativeModalitySpec("sonata", 4, 2, metadata_width=6),
                NativeModalitySpec("vjepa", 5, 2, metadata_width=3),
            ),
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
        )
    ).train()
    modalities = NativeModalityBatch(
        (
            NativeModalityStream(
                "anytouch",
                torch.randn(1, 2, 3),
                torch.ones(1, 2, dtype=torch.bool),
                metadata=torch.randn(1, 2, 3),
            ),
            NativeModalityStream(
                "sonata",
                torch.randn(1, 2, 4),
                torch.ones(1, 2, dtype=torch.bool),
                metadata=torch.randn(1, 2, 6),
            ),
            NativeModalityStream(
                "vjepa",
                torch.randn(1, 2, 5),
                torch.ones(1, 2, dtype=torch.bool),
                metadata=torch.randn(1, 2, 3),
            ),
        )
    )
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8)),
        modalities=modalities,
    )
    context.native_roles = torch.tensor(
        [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
    )
    context.native_valid = torch.ones(1, 3, dtype=torch.bool)
    context.instruction_last_index = torch.tensor([2])
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    assert runtime is not None and prepared[1] is not None
    action_hidden = torch.nn.functional.pad(prepared[1], (0, 4))
    joint = torch.cat((prepared[0], action_hidden), dim=1)
    action_start = prepared[0].shape[1]
    action_loss = _shared_host_layers(joint, mask, layers=3)[:, action_start:].square().mean()

    action_loss.backward()

    for name in ("anytouch", "sonata", "vjepa"):
        gradient = graph.modality_projections[name].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        metadata_gradient = graph.modality_metadata_projections[name].weight.grad
        assert metadata_gradient is not None and torch.isfinite(metadata_gradient).all()
        assert metadata_gradient.abs().sum() > 0
    assert graph.role_embeddings.grad is not None
    assert graph.role_embeddings.grad.abs().sum() > 0


def test_released_resampler_bounds_dense_rows_and_keeps_proprioception_exact() -> None:
    graph = _resampled_graph()
    context = _context()
    context.modalities = _resampled_modalities()
    projected, valid, direct, relation_surfaces = graph._project_modalities(
        context,
        prefix=torch.zeros(1, 3, 8),
    )

    assert projected.shape == (1, 5, 8)
    assert valid.tolist() == [[True, True, True, True, True]]
    assert not direct.any()
    assert relation_surfaces == ()
    assert graph.modality_bridge is not None
    assert graph.config.resampled_modality_names == ("anytouch", "sonata", "vjepa")


def test_released_resampler_is_set_invariant_but_preserves_value_metadata_pairing() -> None:
    graph = _resampled_graph().eval()
    baseline_context = _context()
    baseline_context.modalities = _resampled_modalities()
    baseline, baseline_valid, baseline_direct, baseline_surfaces = graph._project_modalities(
        baseline_context,
        prefix=torch.zeros(1, 3, 8),
    )

    jointly_permuted = []
    value_only_permuted = []
    for stream in baseline_context.modalities.streams:
        if stream.name == "proprioception":
            jointly_permuted.append(stream)
            value_only_permuted.append(stream)
            continue
        permutation = torch.tensor([1, 0])
        jointly_permuted.append(
            NativeModalityStream(
                stream.name,
                stream.tokens[:, permutation],
                stream.valid[:, permutation],
                metadata=(None if stream.metadata is None else stream.metadata[:, permutation]),
                canonical_token_ids=stream.canonical_token_ids[:, permutation],
            )
        )
        value_only_permuted.append(
            NativeModalityStream(
                stream.name,
                stream.tokens[:, permutation],
                stream.valid,
                metadata=stream.metadata,
                canonical_token_ids=stream.canonical_token_ids,
            )
        )

    joint_context = _context()
    joint_context.modalities = NativeModalityBatch(tuple(jointly_permuted))
    joint, joint_valid, joint_direct, joint_surfaces = graph._project_modalities(
        joint_context,
        prefix=torch.zeros(1, 3, 8),
    )
    value_context = _context()
    value_context.modalities = NativeModalityBatch(tuple(value_only_permuted))
    value_only, value_valid, value_direct, value_surfaces = graph._project_modalities(
        value_context,
        prefix=torch.zeros(1, 3, 8),
    )

    torch.testing.assert_close(joint, baseline, rtol=0, atol=1e-6)
    assert torch.equal(joint_valid, baseline_valid)
    assert torch.equal(value_valid, baseline_valid)
    assert torch.equal(joint_direct, baseline_direct)
    assert torch.equal(value_direct, baseline_direct)
    assert baseline_surfaces == joint_surfaces == value_surfaces == ()
    assert not torch.allclose(value_only[:, :4], baseline[:, :4], rtol=0, atol=1e-6)


def test_released_resampler_masks_fully_absent_dense_evidence_and_padding_values() -> None:
    graph = _resampled_graph().eval()

    def project(
        fill: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[object, ...],
    ]:
        streams = []
        for stream in _resampled_modalities(dense_valid=False).streams:
            if stream.name == "proprioception":
                streams.append(stream)
            else:
                streams.append(
                    NativeModalityStream(
                        stream.name,
                        torch.full_like(stream.tokens, fill),
                        stream.valid,
                        metadata=(
                            None
                            if stream.metadata is None
                            else torch.full_like(stream.metadata, fill)
                        ),
                        canonical_token_ids=stream.canonical_token_ids,
                    )
                )
        context = _context()
        context.modalities = NativeModalityBatch(tuple(streams))
        return graph._project_modalities(context, prefix=torch.zeros(1, 3, 8))

    baseline, baseline_valid, baseline_direct, baseline_surfaces = project(0.0)
    padded, padded_valid, padded_direct, padded_surfaces = project(1_000_000.0)
    torch.testing.assert_close(padded, baseline, rtol=0, atol=0)
    assert torch.equal(padded_valid, baseline_valid)
    assert torch.equal(padded_direct, baseline_direct)
    assert baseline_surfaces == padded_surfaces == ()
    assert baseline_valid.tolist() == [[False, False, False, False, True]]
    assert not baseline[:, :4].any()


def test_released_resampler_keeps_native_surface_outside_the_query_bottleneck() -> None:
    graph = _resampled_graph(relation_surface=True)
    context = _context()
    context.modalities = _resampled_modalities()

    projected, valid, direct, surfaces = graph._project_modalities(
        context,
        prefix=torch.zeros(1, 3, 8),
    )

    assert projected.shape == (1, 5, 8)
    assert valid.tolist() == [[True, True, True, True, True]]
    assert not direct.any()
    assert len(surfaces) == 1
    surface = surfaces[0]
    assert surface.name == "vjepa"
    assert surface.sensor_hidden.shape == (1, 2, 8)
    assert surface.sensor_valid.tolist() == [[True, True]]
    assert surface.canonical_token_ids is not None
    assert surface.canonical_token_ids.tolist() == [[0, 1]]


def test_native_surface_crosses_policy_root_and_reaches_shared_host_rows() -> None:
    graph = _resampled_graph(relation_surface=True)
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8)),
        modalities=_resampled_modalities(),
    )
    context.native_roles = _context().native_roles
    context.native_valid = _context().native_valid
    context.instruction_last_index = _context().instruction_last_index
    context.supervise_intermediate_relations = True
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    assert runtime is not None
    output = prepared[0]
    prefix_mask = mask[:, : output.shape[1], : output.shape[1]]
    for layer_index in range(graph.config.num_layers):
        output = _shared_host_layers(output, prefix_mask, layers=1)
        graph.record_layerwise_posterior(
            prefix_hidden=output,
            runtime=runtime,
            layer_index=layer_index,
        )
        if graph.requires_intermediate_relation(layer_index=layer_index, runtime=runtime):
            graph.record_intermediate_relation(
                normalized_prefix=output,
                runtime=runtime,
                layer_index=layer_index,
            )
    graph.finalize_joint_outputs(outputs_embeds=[output, prepared[1]], runtime=runtime)

    relation = context.relation_output
    assert relation is not None
    surface = relation.surface("vjepa")
    root_outputs = context.root_output_tensors()
    assert any(value is surface.support_logits for value in root_outputs)
    assert any(value is surface.ownership for value in root_outputs)
    assert any(value is surface.ownership_log_probability for value in root_outputs)
    intermediate = context.intermediate_relation_outputs[0]
    intermediate_surface = intermediate.surface("vjepa")
    assert any(value is intermediate_surface.support_logits for value in root_outputs)
    assert any(value is intermediate_surface.ownership for value in root_outputs)
    assert any(value is intermediate_surface.ownership_log_probability for value in root_outputs)
    loss = -surface.ownership_log_probability[..., 0].mean()
    loss.backward()
    gradient = graph.modality_projections["vjepa"].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    readout_gradient = graph.relation_readout.projection.weight.grad
    assert readout_gradient is not None and torch.isfinite(readout_gradient).all()
    assert readout_gradient.abs().sum() > 0


def test_object_query_masks_are_assigned_by_full_host_contextual_queries() -> None:
    graph = _object_query_graph().eval()
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8)),
        modalities=_object_query_modalities(),
    )
    context.native_roles = _context().native_roles
    context.native_valid = _context().native_valid
    context.instruction_last_index = _context().instruction_last_index
    inputs, host_mask, positions, visual = _inputs()
    prepared, _mask, _positions, _visual, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    assert runtime is not None
    assert prepared[0] is not None
    hidden = prepared[0].clone()
    hidden[:, runtime.posterior_slice] = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    local_query_slice = dict(runtime.modality_slices)["videomt_queries"]
    query_slice = slice(
        runtime.modality_slice.start + local_query_slice.start,
        runtime.modality_slice.start + local_query_slice.stop,
    )
    aligned = hidden.clone()
    aligned[:, query_slice] = hidden[:, runtime.posterior_slice]
    swapped = hidden.clone()
    swapped[:, query_slice] = hidden[:, runtime.posterior_slice].flip(dims=(1,))
    with torch.no_grad():
        graph.relation_readout.no_object.zero_()
        aligned_surface = graph._read_relation(
            prefix=aligned,
            runtime=runtime,
        ).surface("videomt_masks")
        swapped_surface = graph._read_relation(
            prefix=swapped,
            runtime=runtime,
        ).surface("videomt_masks")

    assert torch.all(aligned_surface.object_probability[0, :2, 0] > 0.99)
    assert torch.all(aligned_surface.object_probability[0, 2:, 1] > 0.99)
    assert torch.all(swapped_surface.object_probability[0, :2, 1] > 0.99)
    assert torch.all(swapped_surface.object_probability[0, 2:, 0] > 0.99)


def test_direct_row_masks_use_the_tied_semantic_projection_and_frozen_source_head() -> None:
    graph = _direct_row_mask_graph().train()
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8)),
        modalities=_direct_row_mask_modalities(),
    )
    context.native_roles = _context().native_roles
    context.native_valid = _context().native_valid
    context.instruction_last_index = _context().instruction_last_index
    inputs, host_mask, positions, visual = _inputs()
    prepared, _mask, _positions, _visual, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    assert runtime is not None and prepared[0] is not None
    hidden = prepared[0].clone()
    hidden[:, runtime.posterior_slice] = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    surface = graph._read_relation(prefix=hidden, runtime=runtime).surface("videomt_masks")
    weight = graph.modality_projections["videomt_queries"].weight
    expected = torch.matmul(hidden[:, runtime.posterior_slice], weight)
    torch.testing.assert_close(surface.support_logits, expected.transpose(1, 2))
    assert surface.donor_query_probability is None

    loss = -surface.ownership_log_probability[..., 0].mean()
    loss.backward()
    gradient = graph.modality_projections["videomt_queries"].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert graph.relation_readout.projection.weight.grad is None
    assert all(
        parameter.grad is None
        for parameter in graph.relation_readout.source_mask_head.parameters()
    )


def test_posterior_adoption_route_keeps_only_proprioception_direct_to_action() -> None:
    graph = _resampled_graph(direct_proprioception=True)
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8)),
        modalities=_resampled_modalities(),
        posterior_adoption_route=torch.ones(1, dtype=torch.bool),
    )
    context.native_roles = torch.tensor(
        [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
    )
    context.native_valid = torch.ones(1, 3, dtype=torch.bool)
    context.instruction_last_index = torch.tensor([2])
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )

    assert runtime is not None and prepared[1] is not None
    action_index = prepared[0].shape[1]
    modality_slice = runtime.modality_slice
    assert modality_slice.stop - modality_slice.start == 5
    assert not mask[0, action_index, 0]
    assert not mask[0, action_index, 1]
    assert mask[0, action_index, 2]
    assert not mask[0, action_index, modality_slice.start : modality_slice.stop - 1].any()
    assert mask[0, action_index, modality_slice.stop - 1]
    assert not context.expanded_action_cache_visible[
        0,
        modality_slice.start : modality_slice.stop - 1,
    ].any()
    assert context.expanded_action_cache_visible[0, modality_slice.stop - 1]
    assert torch.equal(
        context.expanded_posterior_indices,
        torch.arange(runtime.posterior_slice.start, runtime.posterior_slice.stop),
    )
    assert torch.equal(
        context.expanded_posterior_valid,
        runtime.layout.valid[:, runtime.posterior_slice],
    )


def test_direct_posterior_row_intervention_removes_only_selected_cache_rows() -> None:
    graph = _resampled_graph(direct_proprioception=True)
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8)),
        modalities=_resampled_modalities(),
        posterior_adoption_route=torch.ones(1, dtype=torch.bool),
        posterior_action_row_visible=torch.tensor([[True, False]]),
    )
    context.native_roles = torch.tensor(
        [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
    )
    context.native_valid = torch.ones(1, 3, dtype=torch.bool)
    context.instruction_last_index = torch.tensor([2])
    inputs, host_mask, positions, visual = _inputs()
    _prepared, _mask, _positions, _visual, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )

    assert runtime is not None
    assert torch.equal(
        context.expanded_action_cache_visible[:, runtime.posterior_slice],
        torch.tensor([[True, False]]),
    )
    assert torch.equal(context.expanded_posterior_valid, torch.tensor([[True, True]]))


def test_direct_posterior_row_intervention_requires_the_native_direct_route() -> None:
    with pytest.raises(ValueError, match="requires every sample on the direct route"):
        native_context_from_prior_trace(
            controls=_controls(),
            prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8)),
            posterior_action_row_visible=torch.ones(1, 2, dtype=torch.bool),
        )


def test_unselected_posterior_adoption_route_is_bit_identical_to_default() -> None:
    graph = _resampled_graph(direct_proprioception=True)
    prior = NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8))
    default = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=prior,
        modalities=_resampled_modalities(),
    )
    unselected = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=prior,
        modalities=_resampled_modalities(),
        posterior_adoption_route=torch.zeros(1, dtype=torch.bool),
    )
    for context in (default, unselected):
        context.native_roles = torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        )
        context.native_valid = torch.ones(1, 3, dtype=torch.bool)
        context.instruction_last_index = torch.tensor([2])
    inputs, host_mask, positions, visual = _inputs()
    default_prepared = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=default,
    )
    unselected_prepared = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=unselected,
    )
    torch.testing.assert_close(default_prepared[0][0], unselected_prepared[0][0])
    assert torch.equal(default_prepared[1], unselected_prepared[1])
    assert torch.equal(
        default.expanded_action_cache_visible,
        unselected.expanded_action_cache_visible,
    )


def test_action_gradient_crosses_released_resampler_and_every_dense_adapter() -> None:
    graph = _resampled_graph().train()
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8)),
        modalities=_resampled_modalities(),
    )
    context.native_roles = torch.tensor(
        [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
    )
    context.native_valid = torch.ones(1, 3, dtype=torch.bool)
    context.instruction_last_index = torch.tensor([2])
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    assert runtime is not None and prepared[1] is not None
    action_hidden = torch.nn.functional.pad(prepared[1], (0, 4))
    joint = torch.cat((prepared[0], action_hidden), dim=1)
    action_start = prepared[0].shape[1]
    action_loss = _shared_host_layers(joint, mask, layers=3)[:, action_start:].square().mean()
    action_loss.backward()

    for name in ("anytouch", "sonata", "vjepa"):
        assert graph.modality_projections[name].weight.grad is not None
        assert graph.modality_projections[name].weight.grad.abs().sum() > 0
        assert graph.modality_metadata_projections[name].weight.grad is not None
        assert graph.modality_metadata_projections[name].weight.grad.abs().sum() > 0
    assert graph.modality_bridge is not None
    assert graph.modality_bridge.queries.grad is not None
    assert graph.modality_bridge.queries.grad.abs().sum() > 0
    assert graph.modality_bridge.host_projection.weight.grad is not None
    assert graph.modality_bridge.host_projection.weight.grad.abs().sum() > 0
    for parameter in graph.modality_bridge.projector.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_native_videomt_action_gradient_uses_all_200_queries_and_full_modal_context() -> None:
    """ADR-207 association lives in shared host layers, not a selector head."""

    graph = _native_videomt_multimodal_graph().train()
    modalities, source_tokens = _native_videomt_multimodal_batch()
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 200, 8)),
        modalities=modalities,
    )
    context.native_roles = torch.tensor(
        [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
    )
    context.native_valid = torch.ones(1, 3, dtype=torch.bool)
    context.instruction_last_index = torch.tensor([2])
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    if runtime is None or prepared[1] is None:
        raise RuntimeError("native VidEoMT test did not create a joint runtime")
    assert runtime.posterior_slice.stop - runtime.posterior_slice.start == 200
    assert runtime.modality_slice.stop - runtime.modality_slice.start == 3

    action_hidden = torch.nn.functional.pad(prepared[1], (0, 4))
    joint = torch.cat((prepared[0], action_hidden), dim=1)
    action_start = prepared[0].shape[1]
    shared = _shared_host_layers(joint, mask, layers=3)
    action_loss = shared[:, action_start:].square().mean()
    action_loss.backward()

    for name, tokens in source_tokens.items():
        if tokens.grad is None:
            raise RuntimeError(f"native VidEoMT action path omitted {name}")
        assert torch.isfinite(tokens.grad).all()
        assert tokens.grad.abs().sum() > 0
    source_projection = graph.modality_projections["videomt_queries"].weight.grad
    assert source_projection is not None and torch.isfinite(source_projection).all()
    assert source_projection.abs().sum() > 0
    assert graph.modality_bridge is None


def test_absent_dense_evidence_retains_exact_zero_action_gradient_connectivity() -> None:
    graph = _resampled_graph().train()
    context = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 2, 8)),
        modalities=_resampled_modalities(dense_valid=False),
    )
    context.native_roles = torch.tensor(
        [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
    )
    context.native_valid = torch.ones(1, 3, dtype=torch.bool)
    context.instruction_last_index = torch.tensor([2])
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    assert runtime is not None and prepared[1] is not None
    action_hidden = torch.nn.functional.pad(prepared[1], (0, 4))
    joint = torch.cat((prepared[0], action_hidden), dim=1)
    action_start = prepared[0].shape[1]
    action_loss = _shared_host_layers(joint, mask, layers=3)[:, action_start:].square().mean()
    action_loss.backward()

    for name in ("anytouch", "sonata", "vjepa"):
        gradient = graph.modality_projections[name].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert not gradient.any()
        metadata_gradient = graph.modality_metadata_projections[name].weight.grad
        assert metadata_gradient is not None and torch.isfinite(metadata_gradient).all()
        assert not metadata_gradient.any()
    assert graph.modality_bridge is not None
    bridge_parameters = (
        graph.modality_bridge.queries,
        graph.modality_bridge.host_projection.weight,
        *tuple(graph.modality_bridge.projector.parameters()),
    )
    for parameter in bridge_parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert not parameter.grad.any()


def test_invalid_dense_modality_padding_cannot_change_shared_host_action() -> None:
    torch.manual_seed(79)
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(NativeModalitySpec("vjepa", 5, 2, metadata_width=3),),
        )
    )

    def action_output(padded_value: float) -> torch.Tensor:
        tokens = torch.randn(1, 2, 5)
        tokens[:, 1] = padded_value
        metadata = torch.randn(1, 2, 3)
        metadata[:, 1] = padded_value
        context = _context()
        context.modalities = NativeModalityBatch(
            (
                NativeModalityStream(
                    "vjepa",
                    tokens,
                    torch.tensor([[True, False]]),
                    metadata=metadata,
                ),
            )
        )
        inputs, host_mask, positions, visual = _inputs()
        prepared, mask, _, _, _runtime = graph.prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=host_mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=context,
        )
        assert prepared[1] is not None
        action_hidden = torch.nn.functional.pad(prepared[1], (0, 4))
        joint = torch.cat((prepared[0], action_hidden), dim=1)
        return _shared_host_layers(joint, mask, layers=2)[:, prepared[0].shape[1] :]

    torch.manual_seed(83)
    baseline = action_output(0.0)
    torch.manual_seed(83)
    intervened = action_output(1_000_000.0)
    torch.testing.assert_close(baseline, intervened, rtol=0, atol=0)


def test_optional_modality_bridge_rejects_pre_host_compression() -> None:
    with pytest.raises(ValueError, match="would compress"):
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=4,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(NativeModalitySpec("vision", 5, 1),),
        )


def test_modality_declarations_and_runtime_inputs_are_fail_closed() -> None:
    modalities = NativeModalityBatch(
        (
            NativeModalityStream(
                "touch",
                torch.randn(1, 1, 3),
                torch.ones(1, 1, dtype=torch.bool),
            ),
        )
    )
    inputs, mask, positions, visual = _inputs()
    undeclared = _context()
    undeclared.modalities = modalities
    with pytest.raises(ValueError, match="undeclared graph"):
        _graph().prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=undeclared,
        )

    declared = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            modality_specs=(NativeModalitySpec("touch", 3, 1),),
        )
    )
    with pytest.raises(ValueError, match="require one typed runtime batch"):
        declared.prepare_joint_inputs(
            inputs_embeds=_inputs()[0],
            attention_mask=_inputs()[1],
            position_ids=_inputs()[2],
            visual_pos_masks=_inputs()[3],
            context=_context(),
        )


def test_row_only_prior_is_the_same_shared_transition_as_the_full_graph() -> None:
    graph = _graph()
    previous = NativePosteriorState(torch.randn(1, 2, 8))
    full_context = _context(previous)
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=full_context,
    )
    assert runtime is not None and prepared[1] is not None
    full_prefix_count = prepared[0].shape[1]
    full_output = _shared_host_layers(
        prepared[0],
        mask[:, :full_prefix_count, :full_prefix_count],
    )
    graph.finalize_joint_outputs(
        outputs_embeds=[
            full_output,
            prepared[1],
        ],
        runtime=runtime,
    )

    row_context = LingBotPriorRolloutContext(
        controls=_controls(),
        previous_state=previous,
    )
    empty = torch.empty(1, 0, 8)
    row_prepared, row_mask, row_positions, row_visual, row_runtime = graph.prepare_joint_inputs(
        inputs_embeds=[empty, None],
        attention_mask=torch.empty(1, 0, 0, dtype=torch.bool),
        position_ids=torch.empty(3, 1, 0, dtype=torch.long),
        visual_pos_masks=torch.empty(1, 0, dtype=torch.bool),
        context=row_context,
    )
    assert row_prepared[1] is None
    assert not row_positions.any()
    assert row_visual is not None and not row_visual.any()
    row_output = _shared_host_layers(row_prepared[0], row_mask)
    graph.finalize_joint_outputs(
        outputs_embeds=[row_output, None],
        runtime=row_runtime,
    )
    assert full_context.prior_state is not None
    assert row_context.prior_state is not None
    torch.testing.assert_close(row_context.prior_state.rows, full_context.prior_state.rows)


def test_row_only_rollout_rejects_any_raw_observation_or_action_stream() -> None:
    graph = _graph()
    context = LingBotPriorRolloutContext(
        controls=_controls(),
        previous_state=NativePosteriorState(torch.randn(1, 2, 8)),
    )
    with pytest.raises(ValueError, match="prefix must be empty"):
        graph.prepare_joint_inputs(
            inputs_embeds=[torch.randn(1, 1, 8), None],
            attention_mask=torch.ones(1, 1, 1, dtype=torch.bool),
            position_ids=torch.zeros(3, 1, 1, dtype=torch.long),
            visual_pos_masks=None,
            context=context,
        )


def test_row_only_rollout_prediction_queries_read_only_their_paired_prior() -> None:
    graph = _graph().train()
    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.FUTURE,
        route_ids=torch.zeros(1, 2, dtype=torch.long),
        horizons=torch.tensor([[2, 8]], dtype=torch.long),
        addresses=torch.empty(1, 2, 0),
        valid=torch.ones(1, 2, dtype=torch.bool),
    )
    context = LingBotPriorRolloutContext(
        controls=_controls(),
        previous_state=NativePosteriorState(torch.randn(1, 2, 8)),
        prediction_request=request,
    )
    empty = torch.empty(1, 0, 8)
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[empty, None],
        attention_mask=torch.empty(1, 0, 0, dtype=torch.bool),
        position_ids=torch.empty(3, 1, 0, dtype=torch.long),
        visual_pos_masks=torch.empty(1, 0, dtype=torch.bool),
        context=context,
    )
    assert runtime.prediction_slice is not None
    prior_indices = tuple(range(runtime.prior_slice.start, runtime.prior_slice.stop))
    query_count = request.query_count
    for row_index, prior_index in enumerate(prior_indices):
        for query_offset in range(query_count):
            query_index = runtime.prediction_slice.start + row_index * query_count + query_offset
            visible = set(mask[0, query_index].nonzero().flatten().tolist())
            assert visible == {prior_index, query_index}

    output = _shared_host_layers(prepared[0], mask)
    graph.finalize_joint_outputs(outputs_embeds=[output, None], runtime=runtime)
    assert context.prior_state is not None
    assert context.prediction_hidden is not None
    assert context.prediction_hidden.shape == (1, 2, 2, 8)
    assert set(context.prediction_outputs) == {"dino_video"}
    assert context.prediction_outputs["dino_video"].shape == (1, 2, 2, 4)


def test_prediction_value_depends_on_paired_source_not_wrong_source() -> None:
    graph = _graph().train()
    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.FUTURE,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.ones(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    context = LingBotPriorRolloutContext(
        controls=_controls(),
        previous_state=NativePosteriorState(torch.randn(1, 2, 8)),
        prediction_request=request,
    )
    empty = torch.empty(1, 0, 8)
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[empty, None],
        attention_mask=torch.empty(1, 0, 0, dtype=torch.bool),
        position_ids=torch.empty(3, 1, 0, dtype=torch.long),
        visual_pos_masks=torch.empty(1, 0, dtype=torch.bool),
        context=context,
    )
    assert runtime.prediction_slice is not None

    baseline = _shared_host_layers(prepared[0], mask, layers=1)
    intervened_input = prepared[0].clone()
    intervened_input[:, runtime.prior_slice.start] += torch.linspace(-2.0, 2.0, 8)
    intervened = _shared_host_layers(intervened_input, mask, layers=1)
    baseline_prediction = baseline[:, runtime.prediction_slice].reshape(1, 2, 1, 8)
    intervened_prediction = intervened[:, runtime.prediction_slice].reshape(1, 2, 1, 8)

    assert not torch.allclose(
        baseline_prediction[:, 0],
        intervened_prediction[:, 0],
    )
    torch.testing.assert_close(
        baseline_prediction[:, 1],
        intervened_prediction[:, 1],
    )


def test_prediction_gradient_reaches_only_its_paired_source_row() -> None:
    graph = _graph().train()
    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.FUTURE,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.ones(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    context = LingBotPriorRolloutContext(
        controls=_controls(),
        previous_state=NativePosteriorState(torch.randn(1, 2, 8)),
        prediction_request=request,
    )
    empty = torch.empty(1, 0, 8)
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[empty, None],
        attention_mask=torch.empty(1, 0, 0, dtype=torch.bool),
        position_ids=torch.empty(3, 1, 0, dtype=torch.long),
        visual_pos_masks=torch.empty(1, 0, dtype=torch.bool),
        context=context,
    )
    assert runtime.prediction_slice is not None

    host_input = prepared[0].detach().requires_grad_(True)
    output = _shared_host_layers(host_input, mask, layers=1)
    first_query = output[:, runtime.prediction_slice.start]
    gradient = torch.autograd.grad(first_query.square().sum(), host_input)[0]

    paired_prior = gradient[:, runtime.prior_slice.start]
    wrong_prior = gradient[:, runtime.prior_slice.start + 1]
    assert paired_prior.abs().sum() > 0
    torch.testing.assert_close(wrong_prior, torch.zeros_like(wrong_prior))


def test_current_grid_prediction_receives_current_sensor_with_one_layer_delay() -> None:
    graph = _graph().train()
    request = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.CURRENT_RANDOM_GRID,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    context = _context()
    context.prediction_request = request
    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    assert runtime.prediction_slice is not None
    prefix_count = prepared[0].shape[1]
    prefix_mask = mask[:, :prefix_count, :prefix_count]

    sensor_intervention = prepared[0].clone()
    sensor_intervention[:, 0] += torch.linspace(-4.0, 4.0, 8)
    baseline_one = _shared_host_layers(prepared[0], prefix_mask, layers=1)
    changed_one = _shared_host_layers(sensor_intervention, prefix_mask, layers=1)
    torch.testing.assert_close(
        baseline_one[:, runtime.prediction_slice],
        changed_one[:, runtime.prediction_slice],
    )

    baseline_two = _shared_host_layers(prepared[0], prefix_mask, layers=2)
    changed_two = _shared_host_layers(sensor_intervention, prefix_mask, layers=2)
    assert not torch.allclose(
        baseline_two[:, runtime.prediction_slice],
        changed_two[:, runtime.prediction_slice],
    )

    language_intervention = prepared[0].clone()
    language_intervention[:, 2] += torch.linspace(-8.0, 8.0, 8)
    changed_language = _shared_host_layers(language_intervention, prefix_mask, layers=3)
    baseline_three = _shared_host_layers(prepared[0], prefix_mask, layers=3)
    torch.testing.assert_close(
        baseline_three[:, runtime.prediction_slice],
        changed_language[:, runtime.prediction_slice],
    )


def test_observation_context_rejects_future_prediction_requests() -> None:
    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.FUTURE,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.ones(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="row-only prior context"):
        LingBotNativeContext(controls=_controls(), prediction_request=request)


def test_prior_only_context_rejects_unrecognized_prediction_evidence() -> None:
    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.PRIOR_ONLY,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="FUTURE or CURRENT_PRIOR"):
        LingBotPriorRolloutContext(
            controls=_controls(),
            previous_state=NativePosteriorState(torch.randn(1, 2, 8)),
            prediction_request=request,
        )


def test_production_prediction_adapter_is_not_a_tiny_terminal_head() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=16,
            host_width=2560,
            executed_action_dim=7,
            num_layers=36,
            maximum_control_tokens=8,
            prediction_route_count=1,
            prediction_address_width=2,
            predictive_target_widths=(("dino_video", 1024),),
        ),
        device="meta",
    )

    adapter = graph.predictive_readout("dino_video")
    assert adapter.weight.numel() == 2_621_440
    address_projection = graph.prediction_address_projection
    assert address_projection is not None
    predictive_interface_parameters = (
        graph.prediction_role.numel()
        + graph.prediction_route_embeddings.numel()
        + graph.prediction_horizon_projection.weight.numel()
        + address_projection.weight.numel()
        + adapter.weight.numel()
    )
    assert predictive_interface_parameters == 2_636_800
