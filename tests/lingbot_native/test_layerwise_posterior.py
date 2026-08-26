from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import (
    LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)
from picf_next.lingbot_native.state import (
    NativeLayerwisePosteriorState,
    NativePosteriorState,
)
from picf_next.lingbot_native.training import _audit_root_forward_input_contract


def _controls(batch: int = 1) -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.zeros(batch, 1, 2),
        field_valid=torch.ones(batch, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(batch, 1, dtype=torch.bool),
        delta_time=torch.full((batch, 1), 0.1),
        reset=torch.zeros(batch, 1, dtype=torch.bool),
        acknowledged=torch.ones(batch, 1, dtype=torch.bool),
    )


def _graph() -> LingBotNativeGraph:
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=1,
            architecture_identity=LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
        )
    )


def _context(
    memory: NativeLayerwisePosteriorState | None = None,
) -> LingBotNativeContext:
    return LingBotNativeContext(
        controls=_controls(),
        native_roles=torch.tensor([[int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]),
        native_valid=torch.ones(1, 2, dtype=torch.bool),
        instruction_last_index=torch.tensor([1]),
        previous_memory=memory,
    )


def _prepare(graph: LingBotNativeGraph, context: LingBotNativeContext):
    prefix = torch.randn(1, 2, 8)
    action = torch.randn(1, 1, 4)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    positions = torch.zeros(3, 1, 3, dtype=torch.long)
    visual = torch.tensor([[True, False]])
    return graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )


def test_layerwise_rows_have_neutral_content_and_qk_only_stable_addresses() -> None:
    graph = _graph()
    prepared, _mask, _positions, _visual, runtime = _prepare(graph, _context())
    assert runtime is not None
    prefix = prepared[0]
    assert prefix is not None
    prior = prefix[:, runtime.prior_slice]
    posterior = prefix[:, runtime.posterior_slice]
    torch.testing.assert_close(
        prior,
        graph.role_embeddings[0].reshape(1, 1, -1).expand_as(prior),
    )
    torch.testing.assert_close(
        posterior,
        graph.role_embeddings[1].reshape(1, 1, -1).expand_as(posterior),
    )
    bias = graph.layerwise_qk_address_bias(prefix_hidden=prefix, runtime=runtime)
    assert bias is not None
    assert not bias[:, : runtime.prior_slice.start].any()
    torch.testing.assert_close(
        bias[:, runtime.prior_slice],
        graph.object_addresses.unsqueeze(0),
    )
    torch.testing.assert_close(
        bias[:, runtime.posterior_slice],
        graph.object_addresses.unsqueeze(0),
    )
    bias.sum().backward()
    assert graph.object_addresses.grad is not None
    assert graph.object_addresses.grad.abs().sum() > 0


def test_layerwise_memory_is_same_depth_paired_and_current_output_derived() -> None:
    graph = _graph()
    rows = torch.arange(1 * 3 * 2 * 8, dtype=torch.float32).reshape(1, 3, 2, 8)
    previous = NativeLayerwisePosteriorState(rows)
    context = _context(previous)
    prepared, _mask, _positions, _visual, runtime = _prepare(graph, context)
    assert runtime is not None
    prefix = prepared[0]
    assert prefix is not None
    for layer_index in range(graph.config.num_layers):
        memory = graph.layerwise_memory_inputs(
            layer_index=layer_index,
            runtime=runtime,
        )
        assert memory is not None
        hidden, address, visibility = memory
        assert torch.equal(hidden, rows[:, layer_index])
        torch.testing.assert_close(address, graph.object_addresses.unsqueeze(0))
        expected = torch.zeros_like(visibility)
        expected[0, runtime.prior_slice.start, 0] = True
        expected[0, runtime.prior_slice.start + 1, 1] = True
        expected[0, runtime.posterior_slice.start, 0] = True
        expected[0, runtime.posterior_slice.start + 1, 1] = True
        assert torch.equal(visibility, expected)

        layer_output = (prefix + float(layer_index + 1)).requires_grad_()
        graph.record_layerwise_posterior(
            prefix_hidden=layer_output,
            runtime=runtime,
            layer_index=layer_index,
        )
        captured = runtime.layerwise_outputs[-1]
        assert not captured.requires_grad
        assert captured.untyped_storage().data_ptr() != layer_output.untyped_storage().data_ptr()
        unchanged = graph.layerwise_memory_inputs(
            layer_index=layer_index,
            runtime=runtime,
        )
        assert unchanged is not None
        assert torch.equal(unchanged[0], rows[:, layer_index])

    final_prefix = prefix + 9.0
    graph.finalize_joint_outputs(
        outputs_embeds=[final_prefix, prepared[1]],
        runtime=runtime,
    )
    assert context.posterior_memory is not None
    expected_outputs = torch.stack(
        tuple(
            (prefix + float(layer + 1))[:, runtime.posterior_slice]
            for layer in range(graph.config.num_layers)
        ),
        dim=1,
    )
    assert torch.equal(context.posterior_memory.layer_rows, expected_outputs)
    assert context.posterior_state is not None
    assert torch.equal(
        context.posterior_state.rows,
        final_prefix[:, runtime.posterior_slice],
    )


def test_layerwise_graph_rejects_legacy_final_row_recurrence() -> None:
    graph = _graph()
    context = _context()
    context.previous_state = NativePosteriorState(torch.zeros(1, 2, 8))
    context.previous_state_valid = torch.ones(1, dtype=torch.bool)
    with pytest.raises(ValueError, match="retired final-row"):
        _prepare(graph, context)


def test_layerwise_address_and_memory_permutation_is_one_reparameterization() -> None:
    first = _graph()
    second = _graph()
    second.load_state_dict(first.state_dict(), strict=True)
    permutation = torch.tensor([1, 0])
    memory = NativeLayerwisePosteriorState(torch.randn(1, 3, 2, 8))
    with torch.no_grad():
        second.object_addresses.copy_(first.object_addresses.index_select(0, permutation))
    prepared_a, _mask_a, _pos_a, _visual_a, runtime_a = _prepare(first, _context(memory))
    prepared_b, _mask_b, _pos_b, _visual_b, runtime_b = _prepare(
        second,
        _context(memory.permute_rows(permutation)),
    )
    assert runtime_a is not None and runtime_b is not None
    assert prepared_a[0] is not None and prepared_b[0] is not None
    bias_a = first.layerwise_qk_address_bias(
        prefix_hidden=prepared_a[0],
        runtime=runtime_a,
    )
    bias_b = second.layerwise_qk_address_bias(
        prefix_hidden=prepared_b[0],
        runtime=runtime_b,
    )
    assert bias_a is not None and bias_b is not None
    for row_slice in (runtime_a.prior_slice, runtime_a.posterior_slice):
        torch.testing.assert_close(
            bias_b[:, row_slice],
            bias_a[:, row_slice].index_select(1, permutation),
        )
    memory_a = first.layerwise_memory_inputs(layer_index=1, runtime=runtime_a)
    memory_b = second.layerwise_memory_inputs(layer_index=1, runtime=runtime_b)
    assert memory_a is not None and memory_b is not None
    torch.testing.assert_close(memory_b[0], memory_a[0].index_select(1, permutation))
    torch.testing.assert_close(memory_b[1], memory_a[1].index_select(1, permutation))


def test_layerwise_memory_must_match_the_root_compute_dtype() -> None:
    context = _context(NativeLayerwisePosteriorState(torch.zeros(1, 3, 2, 8, dtype=torch.bfloat16)))
    with pytest.raises(TypeError, match="previous layerwise memory"):
        _audit_root_forward_input_contract({}, context)
