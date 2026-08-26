from __future__ import annotations

import pytest
import torch
from torch import nn

from picf_next.lingbot_native.addresses import EpisodeAddressState
from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import (
    LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
    LEGACY_TASK_MATCH_ARCHITECTURE,
    LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
    TASK_INDEPENDENT_ENTITY_POSTERIOR,
    UNIFIED_LAYERWISE_PREDICT_CORRECT,
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
    LingBotNativePriorStepper,
    LingBotPriorRolloutContext,
    ObjectReadActionIntervention,
    compact_lingbot_action_cache,
    native_context_from_persistent_state,
    native_context_from_prior_trace,
)
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    AddressedLayerwisePriorTrace,
    NativeLayerwisePriorTrace,
)
from picf_next.lingbot_native.task_address_graph import (
    TaskAddressActionInformationSet,
    TaskAddressRole,
    TaskAddressTokenLayout,
    task_address_attention_mask,
)


def _controls(*, reset: bool = False) -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.tensor([[[0.25, -0.5]]]),
        field_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        delta_time=torch.tensor([[0.1]]),
        reset=torch.tensor([[reset]], dtype=torch.bool),
        acknowledged=torch.ones(1, 1, dtype=torch.bool),
    )


def _graph() -> LingBotNativeGraph:
    torch.manual_seed(159)
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            task_query_count=2,
            architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        )
    )
    graph.eval()
    return graph


def _address_state(graph: LingBotNativeGraph, episode_id: int = 159) -> EpisodeAddressState:
    codebook = graph.episode_address_codebook
    assert isinstance(codebook, torch.Tensor)
    return EpisodeAddressState.from_episode_ids(
        codebook=codebook,
        episode_ids=torch.tensor([episode_id], dtype=torch.long),
    )


def _addressed_trace(
    graph: LingBotNativeGraph,
    *,
    state: EpisodeAddressState | None = None,
    rows: torch.Tensor | None = None,
) -> AddressedLayerwisePriorTrace:
    return AddressedLayerwisePriorTrace(
        layer_rows=torch.randn(1, 3, 2, 8) if rows is None else rows,
        episode_address_state=_address_state(graph) if state is None else state,
        architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
    )


def _context(
    graph: LingBotNativeGraph,
    *,
    classify_aux: bool = True,
    prior_trace: AddressedLayerwisePriorTrace | None = None,
    object_read_action_intervention: ObjectReadActionIntervention = (
        ObjectReadActionIntervention.FACTUAL
    ),
    object_read_source_row_visible: torch.Tensor | None = None,
    action_information_sets: tuple[TaskAddressActionInformationSet, ...] = (),
) -> LingBotNativeContext:
    roles = torch.tensor(
        [
            [
                int(NativeRole.SENSOR),
                int(NativeRole.SENSOR),
                int(NativeRole.LANGUAGE),
                int(NativeRole.LANGUAGE),
                int(NativeRole.HOST_AUX),
                int(NativeRole.HOST_AUX),
            ]
        ]
    )
    current = torch.tensor([[False, False, False, False, True, False]])
    future = torch.tensor([[False, False, False, False, False, True]])
    return LingBotNativeContext(
        controls=_controls(),
        native_roles=roles,
        native_valid=torch.tensor([[True, True, True, False, True, True]]),
        native_host_current=current if classify_aux else None,
        native_host_future=future if classify_aux else None,
        instruction_last_index=torch.tensor([2]),
        prior_trace=_addressed_trace(graph) if prior_trace is None else prior_trace,
        episode_address_state=(
            _address_state(graph) if prior_trace is None else prior_trace.episode_address_state
        ),
        object_read_action_intervention=object_read_action_intervention,
        action_information_sets=action_information_sets,
        object_read_source_row_visible=object_read_source_row_visible,
    )


def _inputs() -> tuple[
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(160)
    prefix = torch.randn(1, 6, 8)
    action = torch.randn(1, 2, 4)
    mask = torch.ones(1, 8, 8, dtype=torch.bool)
    positions = torch.arange(24).reshape(3, 1, 8)
    visual = torch.tensor([[True, True, False, False, False, False]])
    return [prefix, action], mask, positions, visual


def _prior_inputs() -> tuple[list[torch.Tensor | None], torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        [torch.empty(1, 0, 8), None],
        torch.empty(1, 0, 0, dtype=torch.bool),
        torch.empty(3, 1, 0, dtype=torch.long),
        torch.empty(1, 0, dtype=torch.bool),
    )


def _bind_correction_context(context: LingBotNativeContext) -> None:
    context.bind_native_prefix(
        native_valid=torch.tensor([[True, True, True, False, True, True]]),
        visual_sensor_mask=torch.tensor([[True, True, False, False, False, False]]),
        language_start=2,
        language_valid=torch.tensor([[True, False]]),
        host_current_mask=torch.tensor([[False, False, False, False, True, False]]),
        host_future_mask=torch.tensor([[False, False, False, False, False, True]]),
    )


def test_visual_boundaries_have_their_own_nonphysical_native_role() -> None:
    context = LingBotNativeContext(controls=_controls())
    native_valid = torch.ones(1, 4, dtype=torch.bool)
    context.bind_native_prefix(
        native_valid=native_valid,
        visual_sensor_mask=torch.tensor([[False, True, False, False]]),
        visual_boundary_mask=torch.tensor([[True, False, True, False]]),
        language_start=3,
        language_valid=torch.tensor([[True]]),
        host_current_mask=torch.zeros_like(native_valid),
        host_future_mask=torch.zeros_like(native_valid),
    )
    roles = LingBotNativeGraph._task_address_native_roles(context)
    assert roles.tolist() == [[
        int(TaskAddressRole.SENSOR_BOUNDARY),
        int(TaskAddressRole.SENSOR),
        int(TaskAddressRole.SENSOR_BOUNDARY),
        int(TaskAddressRole.LANGUAGE),
    ]]
    layout = TaskAddressTokenLayout(roles=roles, valid=native_valid)
    host_mask = torch.ones(1, 4, 4, dtype=torch.bool).tril()
    assert torch.equal(task_address_attention_mask(layout, host_mask=host_mask), host_mask)


def _prepared():
    graph = _graph()
    context = _context(graph)
    inputs, mask, positions, visual = _inputs()
    result = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    return graph, context, inputs, positions, result


def test_task_text_copy_preserves_embeddings_padding_and_three_axis_positions() -> None:
    _graph_value, context, inputs, positions, result = _prepared()
    prepared, _mask, expanded_positions, visual, runtime = result
    assert prepared[0] is not None
    assert torch.equal(prepared[0][:, :6], inputs[0])
    assert torch.equal(
        prepared[0][:, runtime.task_text_slice],
        inputs[0][:, runtime.language_slice],
    )
    assert torch.equal(
        expanded_positions[:, :, runtime.task_text_slice],
        positions[:, :, runtime.language_slice],
    )
    assert context.expanded_cache_valid is not None
    assert torch.equal(
        context.expanded_cache_valid[:, runtime.task_text_slice],
        torch.tensor([[True, False]]),
    )
    assert visual is not None
    assert not visual[:, 6:].any()


def test_task_address_layout_uses_zero_reads_and_native_queries() -> None:
    graph, context, _inputs_value, _positions, result = _prepared()
    prepared, _mask, _expanded_positions, _visual, runtime = result
    assert prepared[0] is not None
    hidden = prepared[0]
    assert runtime.task_text_slice == slice(6, 8)
    assert runtime.control_slice == slice(8, 9)
    assert runtime.prior_slice == slice(9, 11)
    assert runtime.posterior_slice == slice(11, 13)
    assert runtime.task_query_slice == slice(13, 15)
    assert runtime.object_read_slice == slice(15, 17)
    assert torch.count_nonzero(hidden[:, runtime.object_read_slice]) == 0
    assert graph.task_query_embeddings is not None
    assert torch.equal(
        hidden[:, runtime.task_query_slice],
        graph.task_query_embeddings.unsqueeze(0),
    )
    assert not (runtime.layout.roles == int(TaskAddressRole.OBJECT_MEMORY)).any()
    receipt_layout = context.task_address_attention_layout
    assert receipt_layout is not None
    assert receipt_layout.batch_size == runtime.layout.batch_size
    assert receipt_layout.query_count == runtime.layout.token_count
    assert receipt_layout.capacity == graph.config.capacity
    assert receipt_layout.object_read_slice == runtime.object_read_slice
    assert receipt_layout.prior_slice == runtime.prior_slice
    assert receipt_layout.posterior_slice == runtime.posterior_slice


def test_value_mask_and_qk_bias_are_separate_and_close_action_bypasses() -> None:
    graph, context, _inputs_value, _positions, result = _prepared()
    prepared, mask, _expanded_positions, _visual, runtime = result
    assert prepared[0] is not None
    value_mask = mask[0]
    task_query = runtime.task_query_slice.start
    object_read = runtime.object_read_slice.start
    action = runtime.layout.token_count - 2

    assert value_mask[task_query, runtime.task_text_slice.start]
    assert value_mask[task_query, task_query]
    assert not value_mask[object_read, task_query]
    assert value_mask[object_read, runtime.prior_slice.start]
    assert value_mask[object_read, runtime.posterior_slice.start]
    assert not value_mask[object_read, object_read]

    assert value_mask[action, 0]
    assert value_mask[action, 2]
    assert value_mask[action, 4]
    assert not value_mask[action, 5]
    assert not value_mask[action, runtime.control_slice.start]
    assert value_mask[action, runtime.task_text_slice.start]
    assert not value_mask[action, runtime.task_query_slice.start]
    assert not value_mask[action, runtime.prior_slice.start]
    assert not value_mask[action, runtime.posterior_slice.start]
    assert value_mask[action, runtime.object_read_slice.start]
    assert context.expanded_action_cache_visible is not None
    assert not context.expanded_action_cache_visible[0, runtime.task_query_slice].any()
    assert not context.expanded_action_cache_visible[0, runtime.control_slice].any()
    assert not context.expanded_action_cache_visible[0, runtime.prior_slice].any()
    assert context.expanded_action_cache_visible[0, runtime.object_read_slice].all()

    bias = graph.layerwise_qk_address_bias(
        prefix_hidden=prepared[0],
        runtime=runtime,
    )
    assert bias is not None
    expected = torch.zeros_like(prepared[0])
    expected[:, runtime.prior_slice] = runtime.episode_addresses
    expected[:, runtime.posterior_slice] = runtime.episode_addresses
    expected[:, runtime.object_read_slice] = prepared[0][:, runtime.task_query_slice]
    assert torch.equal(bias, expected)


def test_direct_posterior_route_replaces_task_query_object_read_action_path() -> None:
    graph = _graph()
    context = _context(graph)
    context.posterior_adoption_route = torch.ones(1, dtype=torch.bool)
    inputs, mask, positions, visual = _inputs()

    prepared, direct_mask, _positions, _visual, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )

    assert prepared[0] is not None
    assert runtime is not None
    assert runtime.match_slice is None
    assert runtime.episode_addresses is not None
    assert runtime.layout.token_count == 6 + 1 + 2 + 2 + 2
    assert context.expanded_action_cache_visible is not None
    assert not context.expanded_action_cache_visible[0, runtime.prior_slice].any()
    assert context.expanded_action_cache_visible[0, runtime.posterior_slice].all()
    action = runtime.layout.token_count - 1
    assert not direct_mask[0, action, runtime.prior_slice].any()
    assert direct_mask[0, action, runtime.posterior_slice].all()

    bias = graph.layerwise_qk_address_bias(
        prefix_hidden=prepared[0],
        runtime=runtime,
    )
    assert bias is not None
    expected = torch.zeros_like(prepared[0])
    expected[:, runtime.prior_slice] = runtime.episode_addresses
    expected[:, runtime.posterior_slice] = runtime.episode_addresses
    assert torch.equal(bias, expected)

    hidden, address, visibility = graph.layerwise_memory_inputs(
        layer_index=0,
        runtime=runtime,
    )
    assert torch.equal(hidden, context.prior_trace.layer(0))
    assert torch.equal(address, runtime.episode_addresses)
    assert visibility.shape == (1, runtime.layout.token_count, graph.config.capacity)


def test_factual_none_is_bit_identical_to_all_source_rows_visible() -> None:
    graph = _graph()
    prior_trace = _addressed_trace(graph)
    factual = _context(graph, prior_trace=prior_trace)
    all_visible = _context(
        graph,
        prior_trace=prior_trace,
        object_read_source_row_visible=torch.ones(1, 2, dtype=torch.bool),
    )
    inputs, mask, positions, visual = _inputs()
    factual_result = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=factual,
    )
    visible_result = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=all_visible,
    )
    factual_inputs, factual_mask, factual_positions, factual_visual, factual_runtime = (
        factual_result
    )
    visible_inputs, visible_mask, visible_positions, visible_visual, visible_runtime = (
        visible_result
    )

    assert all(
        torch.equal(factual_value, visible_value)
        for factual_value, visible_value in zip(
            factual_inputs,
            visible_inputs,
            strict=True,
        )
        if factual_value is not None and visible_value is not None
    )
    assert torch.equal(factual_mask, visible_mask)
    assert torch.equal(factual_positions, visible_positions)
    assert torch.equal(factual_visual, visible_visual)
    assert torch.equal(factual_runtime.layout.roles, visible_runtime.layout.roles)
    assert torch.equal(factual_runtime.layout.valid, visible_runtime.layout.valid)
    assert torch.equal(
        factual.expanded_action_cache_visible,
        all_visible.expanded_action_cache_visible,
    )


def test_source_row_visibility_changes_only_object_read_access_to_paired_rows() -> None:
    graph = _graph()
    prior_trace = _addressed_trace(graph)
    factual = _context(graph, prior_trace=prior_trace)
    row_visible = torch.tensor([[False, True]])
    intervened = _context(
        graph,
        prior_trace=prior_trace,
        object_read_source_row_visible=row_visible,
    )
    inputs, mask, positions, visual = _inputs()
    factual_result = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=factual,
    )
    intervened_result = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=intervened,
    )
    factual_inputs, factual_mask, factual_positions, factual_visual, runtime = factual_result
    intervened_inputs, intervened_mask, intervened_positions, intervened_visual, _runtime = (
        intervened_result
    )
    object_read_queries = runtime.layout.roles == int(TaskAddressRole.OBJECT_READ)
    expected = factual_mask.clone()
    object_read_source_visible = ~object_read_queries.unsqueeze(-1) | row_visible.unsqueeze(1)
    for source_slice in (runtime.prior_slice, runtime.posterior_slice):
        expected[:, :, source_slice] &= object_read_source_visible

    assert torch.equal(intervened_mask, expected)
    assert torch.equal(intervened_mask[~object_read_queries], factual_mask[~object_read_queries])
    writer_queries = (runtime.layout.roles == int(TaskAddressRole.PRIOR)) | (
        runtime.layout.roles == int(TaskAddressRole.POSTERIOR)
    )
    assert torch.equal(intervened_mask[writer_queries], factual_mask[writer_queries])
    assert all(
        torch.equal(factual_value, intervened_value)
        for factual_value, intervened_value in zip(
            factual_inputs,
            intervened_inputs,
            strict=True,
        )
        if factual_value is not None and intervened_value is not None
    )
    assert torch.equal(factual_positions, intervened_positions)
    assert torch.equal(factual_visual, intervened_visual)
    assert torch.equal(
        factual.expanded_action_cache_visible,
        intervened.expanded_action_cache_visible,
    )
    assert torch.equal(
        graph.layerwise_qk_address_bias(
            prefix_hidden=factual_inputs[0],
            runtime=runtime,
        ),
        graph.layerwise_qk_address_bias(
            prefix_hidden=intervened_inputs[0],
            runtime=_runtime,
        ),
    )


def test_blocked_object_read_intervention_changes_only_the_mediator_action_edge() -> None:
    graph = _graph()
    prior_trace = _addressed_trace(graph)
    factual = _context(graph, prior_trace=prior_trace)
    blocked = _context(
        graph,
        prior_trace=prior_trace,
        object_read_action_intervention=ObjectReadActionIntervention.BLOCKED,
    )
    inputs, mask, positions, visual = _inputs()
    factual_result = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=factual,
    )
    blocked_result = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=blocked,
    )
    factual_inputs, factual_mask, factual_positions, factual_visual, runtime = factual_result
    blocked_inputs, blocked_mask, blocked_positions, blocked_visual, blocked_runtime = (
        blocked_result
    )
    assert runtime.object_read_slice == blocked_runtime.object_read_slice
    assert torch.equal(factual_inputs[0], blocked_inputs[0])
    assert torch.equal(factual_inputs[1], blocked_inputs[1])
    assert torch.equal(factual_positions, blocked_positions)
    assert torch.equal(factual_visual, blocked_visual)

    action_queries = runtime.layout.roles == int(TaskAddressRole.ACTION)
    expected_mask = factual_mask.clone()
    expected_mask[:, :, runtime.object_read_slice] &= ~action_queries.unsqueeze(-1)
    assert torch.equal(blocked_mask, expected_mask)
    assert factual.expanded_action_cache_visible is not None
    assert blocked.expanded_action_cache_visible is not None
    expected_cache = factual.expanded_action_cache_visible.clone()
    expected_cache[:, runtime.object_read_slice] = False
    assert torch.equal(blocked.expanded_action_cache_visible, expected_cache)


def test_source_row_visibility_composes_with_whole_object_read_edge_blocking() -> None:
    graph = _graph()
    prior_trace = _addressed_trace(graph)
    row_visible = torch.tensor([[True, False]])
    factual = _context(
        graph,
        prior_trace=prior_trace,
        object_read_source_row_visible=row_visible,
    )
    blocked = _context(
        graph,
        prior_trace=prior_trace,
        object_read_action_intervention=ObjectReadActionIntervention.BLOCKED,
        object_read_source_row_visible=row_visible,
    )
    inputs, mask, positions, visual = _inputs()
    factual_result = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=factual,
    )
    blocked_result = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=blocked,
    )
    factual_inputs, factual_mask, factual_positions, factual_visual, runtime = factual_result
    blocked_inputs, blocked_mask, blocked_positions, blocked_visual, _runtime = blocked_result
    action_queries = runtime.layout.roles == int(TaskAddressRole.ACTION)
    expected = factual_mask.clone()
    expected[:, :, runtime.object_read_slice] &= ~action_queries.unsqueeze(-1)

    assert torch.equal(blocked_mask, expected)
    assert all(
        torch.equal(factual_value, blocked_value)
        for factual_value, blocked_value in zip(
            factual_inputs,
            blocked_inputs,
            strict=True,
        )
        if factual_value is not None and blocked_value is not None
    )
    assert torch.equal(factual_positions, blocked_positions)
    assert torch.equal(factual_visual, blocked_visual)
    object_read_queries = runtime.layout.roles == int(TaskAddressRole.OBJECT_READ)
    for source_slice in (runtime.prior_slice, runtime.posterior_slice):
        assert not blocked_mask[object_read_queries, source_slice.stop - 1].any()
    assert factual.expanded_action_cache_visible is not None
    assert blocked.expanded_action_cache_visible is not None
    expected_cache = factual.expanded_action_cache_visible.clone()
    expected_cache[:, runtime.object_read_slice] = False
    assert torch.equal(blocked.expanded_action_cache_visible, expected_cache)


def test_object_read_action_intervention_rejects_untyped_values() -> None:
    with pytest.raises(TypeError, match="typed enum"):
        LingBotNativeContext(
            controls=_controls(),
            object_read_action_intervention="blocked",  # type: ignore[arg-type]
        )


def test_object_read_source_row_visibility_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="must be a tensor"):
        LingBotNativeContext(
            controls=_controls(),
            object_read_source_row_visible=[[True, False]],  # type: ignore[arg-type]
        )


def test_object_read_source_row_visibility_rejects_non_boolean() -> None:
    with pytest.raises(TypeError, match="must be boolean"):
        LingBotNativeContext(
            controls=_controls(),
            object_read_source_row_visible=torch.ones(1, 2),
        )


@pytest.mark.parametrize("shape", [(2,), (2, 2)])
def test_object_read_source_row_visibility_rejects_invalid_batch_shape(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match=r"shape \[batch, capacity\]"):
        LingBotNativeContext(
            controls=_controls(),
            object_read_source_row_visible=torch.ones(shape, dtype=torch.bool),
        )


def test_object_read_source_row_visibility_rejects_wrong_capacity() -> None:
    graph = _graph()
    context = _context(
        graph,
        object_read_source_row_visible=torch.ones(1, 3, dtype=torch.bool),
    )
    inputs, mask, positions, visual = _inputs()
    with pytest.raises(ValueError, match=r"shape \[batch, capacity\]"):
        graph.prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=context,
        )


def test_object_read_source_row_visibility_rejects_device_mismatch() -> None:
    with pytest.raises(ValueError, match="controls must share one device"):
        LingBotNativeContext(
            controls=_controls(),
            object_read_source_row_visible=torch.ones(
                1,
                2,
                dtype=torch.bool,
                device="meta",
            ),
        )


def test_context_helpers_thread_object_read_source_row_visibility() -> None:
    graph = _graph()
    row_visible = torch.tensor([[True, False]])

    persistent = native_context_from_persistent_state(
        controls=_controls(),
        persistent_state=None,
        object_read_source_row_visible=row_visible,
    )
    correction = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=_addressed_trace(graph),
        object_read_source_row_visible=row_visible,
    )

    assert persistent.object_read_source_row_visible is row_visible
    assert correction.object_read_source_row_visible is row_visible


def test_context_helpers_default_source_row_visibility_to_none() -> None:
    graph = _graph()

    persistent = native_context_from_persistent_state(
        controls=_controls(),
        persistent_state=None,
    )
    correction = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=_addressed_trace(graph),
    )

    assert persistent.object_read_source_row_visible is None
    assert correction.object_read_source_row_visible is None


def _action_cache_fixture(
    *,
    intervention: ObjectReadActionIntervention,
    inserted_visibility: tuple[bool, ...],
) -> tuple[
    LingBotNativeContext,
    dict[int, dict[str, torch.Tensor]],
    dict[int, dict[str, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
]:
    graph = _graph()
    context = _context(graph, object_read_action_intervention=intervention)
    native_valid = context.native_valid
    assert native_valid is not None
    native_positions = torch.arange(3 * native_valid.shape[1]).reshape(
        3,
        1,
        native_valid.shape[1],
    )
    inserted_count = len(inserted_visibility)
    expanded_valid = torch.cat(
        (native_valid, torch.ones(1, inserted_count, dtype=torch.bool)),
        dim=1,
    )
    expanded_positions = torch.cat(
        (
            native_positions,
            torch.arange(
                3 * native_valid.shape[1],
                3 * (native_valid.shape[1] + inserted_count),
            ).reshape(3, 1, inserted_count),
        ),
        dim=2,
    )
    context.expanded_cache_valid = expanded_valid
    context.expanded_cache_position_ids = expanded_positions
    context.expanded_action_cache_visible = torch.cat(
        (
            native_valid,
            torch.tensor([inserted_visibility], dtype=torch.bool),
        ),
        dim=1,
    )
    native_cache: dict[int, dict[str, torch.Tensor]] = {}
    expanded_cache: dict[int, dict[str, torch.Tensor]] = {}
    for layer in range(2):
        native_key = torch.arange(
            native_valid.shape[1] * 6,
            dtype=torch.float32,
        ).reshape(1, native_valid.shape[1], 2, 3) + 1000 * layer
        expanded_key = torch.arange(
            expanded_valid.shape[1] * 6,
            dtype=torch.float32,
        ).reshape(1, expanded_valid.shape[1], 2, 3) + 10000 + 1000 * layer
        native_cache[layer] = {
            "key_states": native_key,
            "value_states": native_key + 100,
        }
        expanded_cache[layer] = {
            "key_states": expanded_key,
            "value_states": expanded_key + 100,
        }
    return context, native_cache, expanded_cache, native_valid, native_positions


def test_action_cache_keeps_released_native_kv_and_appends_only_visible_rows() -> None:
    context, native_cache, expanded_cache, native_valid, native_positions = (
        _action_cache_fixture(
            intervention=ObjectReadActionIntervention.FACTUAL,
            inserted_visibility=(False, True, False, True),
        )
    )
    compact = compact_lingbot_action_cache(
        native_past_key_values=native_cache,
        expanded_past_key_values=expanded_cache,
        native_valid=native_valid,
        native_position_ids=native_positions,
        context=context,
    )

    expected_indices = torch.tensor([7, 9])
    assert torch.equal(compact.selected_inserted_indices, expected_indices)
    assert torch.equal(
        compact.valid,
        torch.cat((native_valid, torch.ones(1, 2, dtype=torch.bool)), dim=1),
    )
    assert torch.equal(
        compact.position_valid,
        torch.cat((native_valid, torch.zeros(1, 2, dtype=torch.bool)), dim=1),
    )
    assert torch.equal(compact.position_ids[:, :, :6], native_positions)
    assert torch.equal(
        compact.position_ids[:, :, 6:],
        context.expanded_cache_position_ids.index_select(2, expected_indices),
    )
    for layer, values in compact.past_key_values.items():
        for name in ("key_states", "value_states"):
            assert torch.equal(values[name][:, :6], native_cache[layer][name])
            assert torch.equal(
                values[name][:, 6:],
                expanded_cache[layer][name].index_select(1, expected_indices),
            )


def test_action_cache_publishes_exact_posterior_key_layout_for_suffix() -> None:
    context, native_cache, expanded_cache, native_valid, native_positions = (
        _action_cache_fixture(
            intervention=ObjectReadActionIntervention.FACTUAL,
            inserted_visibility=(False, True, False, True),
        )
    )
    context.expanded_posterior_indices = torch.tensor([7, 9])
    context.expanded_posterior_valid = torch.tensor([[True, False]])
    compact = compact_lingbot_action_cache(
        native_past_key_values=native_cache,
        expanded_past_key_values=expanded_cache,
        native_valid=native_valid,
        native_position_ids=native_positions,
        context=context,
        suffix_count=3,
    )

    layout = compact.action_attention_layout
    assert layout is not None
    assert layout.native_prefix_count == 6
    assert layout.compact_prefix_count == 8
    assert layout.key_count == 11
    assert layout.action_query_slice == slice(1, 3)
    assert torch.equal(layout.posterior_key_indices, torch.tensor([6, 7]))
    assert torch.equal(layout.posterior_key_valid, torch.tensor([[True, False]]))


def test_blocked_action_cache_is_the_exact_released_cache() -> None:
    context, native_cache, expanded_cache, native_valid, native_positions = (
        _action_cache_fixture(
            intervention=ObjectReadActionIntervention.BLOCKED,
            inserted_visibility=(False, False, False),
        )
    )
    compact = compact_lingbot_action_cache(
        native_past_key_values=native_cache,
        expanded_past_key_values=expanded_cache,
        native_valid=native_valid,
        native_position_ids=native_positions,
        context=context,
    )

    assert compact.past_key_values is native_cache
    assert compact.valid is native_valid
    assert compact.position_ids is native_positions
    assert compact.position_valid is native_valid
    assert compact.selected_inserted_indices.numel() == 0


def test_serialized_rows_and_external_prior_trace_are_strictly_same_row() -> None:
    graph, _context_value, _inputs_value, _positions, result = _prepared()
    prepared, mask, _expanded_positions, _visual, runtime = result
    assert prepared[0] is not None
    value_mask = mask[0]
    for row in range(2):
        other = 1 - row
        prior = runtime.prior_slice.start + row
        posterior = runtime.posterior_slice.start + row
        assert value_mask[prior, prior]
        assert not value_mask[prior, runtime.prior_slice.start + other]
        assert value_mask[posterior, prior]
        assert value_mask[posterior, posterior]
        assert not value_mask[posterior, runtime.prior_slice.start + other]
        assert not value_mask[posterior, runtime.posterior_slice.start + other]

    memory = graph.layerwise_memory_inputs(layer_index=1, runtime=runtime)
    assert memory is not None
    hidden, address, visibility = memory
    assert hidden.shape == (1, 2, 8)
    assert torch.equal(address, runtime.episode_addresses)
    assert visibility[0, runtime.prior_slice.start, 0]
    assert not visibility[0, runtime.prior_slice.start, 1]
    assert visibility[0, runtime.posterior_slice.start + 1, 1]
    assert not visibility[0, runtime.posterior_slice.start + 1, 0]
    action = runtime.layout.token_count - 2
    assert not visibility[0, action].any()


def test_object_read_increment_is_zero_without_physical_values() -> None:
    _graph_value, _context_value, _inputs_value, _positions, result = _prepared()
    prepared, mask, _expanded_positions, _visual, runtime = result
    assert prepared[0] is not None
    prefix_count = prepared[0].shape[1]
    read_mask = mask[:, runtime.object_read_slice, :prefix_count]
    logits = torch.randn_like(read_mask, dtype=prepared[0].dtype)
    weights = torch.softmax(logits.masked_fill(~read_mask, float("-inf")), dim=-1)

    no_physical = torch.zeros_like(prepared[0])
    no_physical[:, runtime.task_query_slice] = 100.0
    increment = weights @ no_physical
    assert torch.count_nonzero(increment) == 0

    physical = torch.zeros_like(prepared[0])
    physical[:, runtime.prior_slice] = 1.0
    physical[:, runtime.posterior_slice] = 1.0
    assert torch.count_nonzero(weights @ physical) > 0


def test_addressed_prior_rollout_reuses_one_receipt_in_correction() -> None:
    graph = _graph().train()
    prior_context = LingBotPriorRolloutContext(
        controls=_controls(reset=True),
        episode_ids=torch.tensor([4101], dtype=torch.long),
    )
    prior_inputs, prior_mask, prior_positions, prior_visual = _prior_inputs()
    prepared, value_mask, _positions, _visual, prior_runtime = graph.prepare_joint_inputs(
        inputs_embeds=prior_inputs,
        attention_mask=prior_mask,
        position_ids=prior_positions,
        visual_pos_masks=prior_visual,
        context=prior_context,
    )
    assert prepared[0] is not None
    assert prior_context.episode_address_state is not None
    assert prior_runtime.episode_addresses is not None
    hidden = prepared[0]
    qk_bias = graph.layerwise_qk_address_bias(
        prefix_hidden=hidden,
        runtime=prior_runtime,
    )
    assert qk_bias is not None
    assert torch.equal(
        qk_bias[:, prior_runtime.prior_slice],
        prior_runtime.episode_addresses,
    )
    for row in range(graph.config.capacity):
        query = prior_runtime.prior_slice.start + row
        assert value_mask[0, query, query]
        assert not value_mask[0, query, prior_runtime.prior_slice.start + (1 - row)]
    assert graph.layerwise_memory_inputs(layer_index=0, runtime=prior_runtime) is None

    for layer_index in range(graph.config.num_layers):
        next_hidden = hidden.clone()
        next_hidden[:, prior_runtime.prior_slice] += float(layer_index + 1)
        graph.record_layerwise_posterior(
            prefix_hidden=next_hidden,
            runtime=prior_runtime,
            layer_index=layer_index,
        )
        hidden = next_hidden
    graph.finalize_joint_outputs(outputs_embeds=[hidden, None], runtime=prior_runtime)
    trace = prior_context.prior_trace
    assert isinstance(trace, AddressedLayerwisePriorTrace)
    assert trace.address_receipt == prior_context.episode_address_state.receipt

    correction = native_context_from_prior_trace(
        controls=_controls(),
        prior_trace=trace,
    )
    assert correction.episode_address_state is trace.episode_address_state
    _bind_correction_context(correction)
    inputs, host_mask, positions, visual = _inputs()
    corrected, correction_mask, _positions, _visual, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=correction,
    )
    assert corrected[0] is not None
    correction_bias = graph.layerwise_qk_address_bias(
        prefix_hidden=corrected[0],
        runtime=runtime,
    )
    assert correction_bias is not None
    assert torch.equal(correction_bias[:, runtime.prior_slice], prior_runtime.episode_addresses)
    assert torch.equal(
        correction_bias[:, runtime.posterior_slice],
        prior_runtime.episode_addresses,
    )
    assert torch.equal(
        correction_bias[:, runtime.object_read_slice],
        corrected[0][:, runtime.task_query_slice],
    )
    external = graph.layerwise_memory_inputs(layer_index=1, runtime=runtime)
    assert external is not None
    external_hidden, external_address, external_visibility = external
    assert torch.equal(external_hidden, trace.layer(1))
    assert torch.equal(external_address, prior_runtime.episode_addresses)
    action_start = runtime.layout.token_count - inputs[1].shape[1]
    assert not external_visibility[:, action_start:].any()
    assert not correction_mask[:, action_start:, runtime.prior_slice].any()
    assert not correction_mask[:, action_start:, runtime.posterior_slice].any()
    assert not correction_mask[:, action_start:, runtime.task_query_slice].any()


def test_addressed_trace_row_swap_is_jointly_equivariant() -> None:
    graph = _graph()
    rows = torch.arange(48, dtype=torch.float32).reshape(1, 3, 2, 8)
    trace = _addressed_trace(graph, rows=rows)
    row_swap = torch.tensor([1, 0], dtype=torch.long)
    swapped = trace.permute_rows(row_swap)
    codebook = graph.episode_address_codebook
    assert isinstance(codebook, torch.Tensor)
    torch.testing.assert_close(swapped.layer_rows, rows[:, :, row_swap])
    torch.testing.assert_close(
        swapped.episode_address_state.materialize(codebook),
        trace.episode_address_state.materialize(codebook)[:, row_swap],
    )
    assert swapped.address_receipt != trace.address_receipt


def test_fp32_address_receipt_materializes_bfloat16_qk_bias_without_rehashing() -> None:
    graph = _graph()
    state = _address_state(graph)
    codebook = graph.episode_address_codebook
    assert isinstance(codebook, torch.Tensor)
    assert codebook.dtype == torch.float32
    receipt = state.receipt

    addresses = graph._materialize_episode_addresses(
        state,
        batch=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert addresses.dtype == torch.bfloat16
    torch.testing.assert_close(addresses.float(), state.materialize(codebook), atol=0.004, rtol=0)
    assert state.receipt == receipt


def test_wrong_receipt_and_unaddressed_trace_fail_closed() -> None:
    graph = _graph()
    trace = _addressed_trace(graph)
    wrong = trace.episode_address_state.permute_rows(torch.tensor([1, 0]))
    with pytest.raises(ValueError, match="differs from the prior trace receipt"):
        native_context_from_prior_trace(
            controls=_controls(),
            prior_trace=trace,
            episode_address_state=wrong,
        )

    raw_trace = NativeLayerwisePriorTrace(trace.layer_rows)
    raw_context = LingBotNativeContext(
        controls=_controls(),
        prior_trace=raw_trace,
        episode_address_state=trace.episode_address_state,
    )
    _bind_correction_context(raw_context)
    inputs, mask, positions, visual = _inputs()
    with pytest.raises(ValueError, match="addressed prior trace"):
        graph.prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=raw_context,
        )


def test_reset_and_invalid_lanes_require_explicit_new_address_assignment() -> None:
    graph = _graph()
    source = _addressed_trace(graph)
    prior_inputs, prior_mask, prior_positions, prior_visual = _prior_inputs()
    reset_context = LingBotPriorRolloutContext(
        controls=_controls(reset=True),
        source_prior_trace=source,
        source_prior_trace_valid=torch.tensor([True]),
    )
    with pytest.raises(ValueError, match="reset or invalid prior lanes"):
        graph.prepare_joint_inputs(
            inputs_embeds=prior_inputs,
            attention_mask=prior_mask,
            position_ids=prior_positions,
            visual_pos_masks=prior_visual,
            context=reset_context,
        )

    cold_context = LingBotPriorRolloutContext(controls=_controls())
    with pytest.raises(ValueError, match="explicit addresses or deterministic episode IDs"):
        graph.prepare_joint_inputs(
            inputs_embeds=prior_inputs,
            attention_mask=prior_mask,
            position_ids=prior_positions,
            visual_pos_masks=prior_visual,
            context=cold_context,
        )

    resumed_context = LingBotPriorRolloutContext(
        controls=_controls(),
        source_prior_trace=source,
        source_prior_trace_valid=torch.tensor([True]),
        episode_address_state=source.episode_address_state.permute_rows(torch.tensor([1, 0])),
    )
    with pytest.raises(ValueError, match="continuing prior lanes changed"):
        graph.prepare_joint_inputs(
            inputs_embeds=prior_inputs,
            attention_mask=prior_mask,
            position_ids=prior_positions,
            visual_pos_masks=prior_visual,
            context=resumed_context,
        )

    reset_with_id = LingBotPriorRolloutContext(
        controls=_controls(reset=True),
        source_prior_trace=source,
        source_prior_trace_valid=torch.tensor([True]),
        episode_ids=torch.tensor([4102], dtype=torch.long),
    )
    _prepared, _mask, _positions, _visual, runtime = graph.prepare_joint_inputs(
        inputs_embeds=prior_inputs,
        attention_mask=prior_mask,
        position_ids=prior_positions,
        visual_pos_masks=prior_visual,
        context=reset_with_id,
    )
    assert reset_with_id.episode_address_state is not None
    assert runtime.episode_addresses is not None
    assert reset_with_id.source_prior_trace_valid is not None
    assert not reset_with_id.source_prior_trace_valid.any()


def test_prior_stepper_does_not_silently_reuse_old_addresses_on_reset() -> None:
    graph = _graph()

    class _Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.qwenvl_with_expert = nn.Module()
            self.model.qwenvl_with_expert.picf_native_graph = graph

        def picf_native_prior_forward(self, **_kwargs):
            return ()

    stepper = LingBotNativePriorStepper(_Policy(), graph)
    state = AddressedLayerwisePosteriorState(
        layer_rows=torch.randn(1, 3, 2, 8),
        episode_address_state=_address_state(graph, episode_id=4101),
        architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
    )
    context, _reference = stepper._build_context(
        state,
        _controls(reset=True),
        previous_memory_valid=torch.tensor([True]),
    )
    assert context.episode_address_state is None
    prior_inputs, prior_mask, prior_positions, prior_visual = _prior_inputs()
    with pytest.raises(ValueError, match="reset or invalid prior lanes"):
        graph.prepare_joint_inputs(
            inputs_embeds=prior_inputs,
            attention_mask=prior_mask,
            position_ids=prior_positions,
            visual_pos_masks=prior_visual,
            context=context,
        )


def test_task_address_identity_opens_training_but_rejects_unclassified_aux() -> None:
    graph = _graph()
    inputs, mask, positions, visual = _inputs()
    graph.train()
    prepared, _value_mask, _positions, _visual, _runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=_context(graph),
    )
    assert prepared[0] is not None
    graph.eval()
    with pytest.raises(RuntimeError, match="current/future"):
        graph.prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            visual_pos_masks=visual,
            context=_context(graph, classify_aux=False),
        )


def test_historical_identity_parameter_surfaces_are_unchanged() -> None:
    common = {
        "role_embeddings",
        "prediction_role",
        "prediction_route_embeddings",
        "prediction_horizon_projection.weight",
        "control_projection.weight",
        "relation_readout.no_object",
        "relation_readout.temperature_parameter",
        "relation_readout.projection.weight",
        "relation_readout.existence_projection.weight",
        "relation_readout.existence_projection.bias",
    }
    expected = {
        LEGACY_TASK_MATCH_ARCHITECTURE: common
        | {
            "object_queries",
            "relation_readout.match_projection.weight",
            "relation_readout.match_projection.bias",
        },
        TASK_INDEPENDENT_ENTITY_POSTERIOR: common | {"object_queries"},
        LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR: common | {"object_addresses"},
        UNIFIED_LAYERWISE_PREDICT_CORRECT: common | {"object_addresses"},
    }
    for identity, names in expected.items():
        graph = LingBotNativeGraph(
            LingBotNativeGraphConfig(
                capacity=2,
                host_width=8,
                executed_action_dim=2,
                num_layers=3,
                architecture_identity=identity,
            )
        )
        assert set(graph.state_dict()) == names
        assert graph.task_query_embeddings is None

    candidate = _graph()
    assert candidate.object_addresses is None
    assert candidate.object_queries is None
    assert "episode_address_codebook" in candidate.state_dict()
    assert "task_query_embeddings" in candidate.state_dict()
    assert "episode_address_codebook" in dict(candidate.named_buffers())
    assert "episode_address_codebook" not in dict(candidate.named_parameters())
    assert not candidate.episode_address_codebook.requires_grad
