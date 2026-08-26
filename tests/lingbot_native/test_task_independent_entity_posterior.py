from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import (
    TASK_INDEPENDENT_ENTITY_POSTERIOR,
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)
from picf_next.lingbot_native.physical_relations import (
    TASK_INDEPENDENT_PHYSICAL_INTERFACE,
    PhysicalEntityReadout,
    PhysicalRelationOutput,
)


def _controls() -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.tensor([[[0.25, -0.5]]]),
        field_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        delta_time=torch.full((1, 1), 0.1),
        reset=torch.zeros(1, 1, dtype=torch.bool),
        acknowledged=torch.ones(1, 1, dtype=torch.bool),
    )


def _context() -> LingBotNativeContext:
    return LingBotNativeContext(
        controls=_controls(),
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ),
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2]),
    )


def _inputs() -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    prefix = torch.randn(1, 3, 8, generator=generator)
    action = torch.randn(1, 2, 4, generator=generator)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    positions = torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5).clone()
    visual = torch.tensor([[True, True, False]])
    return [prefix, action], mask, positions, visual


def _graph(*, task_independent: bool = True) -> LingBotNativeGraph:
    torch.manual_seed(2)
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            architecture_identity=(
                TASK_INDEPENDENT_ENTITY_POSTERIOR
                if task_independent
                else "content_addressed_task_match_v1"
            ),
        )
    )


def _shared_host(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = hidden
    width = hidden.shape[-1]
    for layer in range(3):
        generator = torch.Generator().manual_seed(100 + layer)
        weight = torch.randn(width, width, generator=generator) / width**0.5
        query = output @ weight
        score = query @ query.transpose(-1, -2) / width**0.5
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        attended = torch.softmax(score, dim=-1) @ output
        output = torch.nn.functional.layer_norm(output + attended, (width,))
        output = torch.nn.functional.layer_norm(output + torch.tanh(output @ weight), (width,))
    return output


def test_task_independent_graph_has_no_task_match_parameters_or_tokens() -> None:
    graph = _graph()
    assert graph.task_independent is True
    assert isinstance(graph.relation_readout, PhysicalEntityReadout)
    assert graph.role_embeddings.shape == (3, 8)
    assert all("match" not in name for name, _parameter in graph.named_parameters())

    inputs, host_mask, positions, visual = _inputs()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=inputs,
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=_context(),
    )
    assert runtime is not None
    assert runtime.match_slice is None
    assert prepared[0].shape == (1, 8, 8)
    assert mask.shape == (1, 10, 10)
    assert not (runtime.layout.roles == int(NativeRole.MATCH)).any()

    action_indices = (runtime.layout.roles[0] == int(NativeRole.ACTION)).nonzero().flatten()
    language_indices = (runtime.layout.roles[0] == int(NativeRole.LANGUAGE)).nonzero().flatten()
    posterior_indices = torch.arange(runtime.posterior_slice.start, runtime.posterior_slice.stop)
    assert mask[0, action_indices[:, None], language_indices].all()
    assert mask[0, action_indices[:, None], posterior_indices].all()


def test_physical_posterior_and_readout_are_prompt_invariant() -> None:
    graph = _graph()
    inputs_a, host_mask, positions, visual = _inputs()
    inputs_b = [inputs_a[0].clone(), inputs_a[1].clone()]
    inputs_b[0][:, 2] += 7.0
    prepared_outputs = []
    contexts = []
    runtimes = []
    for inputs in (inputs_a, inputs_b):
        context = _context()
        prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
            inputs_embeds=inputs,
            attention_mask=host_mask.clone(),
            position_ids=positions.clone(),
            visual_pos_masks=visual.clone(),
            context=context,
        )
        assert runtime is not None
        output = _shared_host(prepared[0], mask[:, : prepared[0].shape[1], : prepared[0].shape[1]])
        prepared_outputs.append((prepared, output))
        contexts.append(context)
        runtimes.append(runtime)

    torch.testing.assert_close(
        prepared_outputs[0][1][:, runtimes[0].posterior_slice],
        prepared_outputs[1][1][:, runtimes[1].posterior_slice],
        rtol=0,
        atol=0,
    )
    for (prepared, output), context, runtime in zip(
        prepared_outputs,
        contexts,
        runtimes,
        strict=True,
    ):
        graph.finalize_joint_outputs(
            outputs_embeds=[output, prepared[1]],
            runtime=runtime,
        )
        assert isinstance(context.relation_output, PhysicalRelationOutput)
        assert context.relation_output.interface == TASK_INDEPENDENT_PHYSICAL_INTERFACE
        assert not hasattr(context.relation_output, "task_relevance")
        context.root_output_tensors()
    first = contexts[0].relation_output
    second = contexts[1].relation_output
    assert isinstance(first, PhysicalRelationOutput)
    assert isinstance(second, PhysicalRelationOutput)
    torch.testing.assert_close(first.ownership, second.ownership, rtol=0, atol=0)


def test_legacy_match_checkpoint_cannot_load_into_task_independent_graph() -> None:
    legacy = _graph(task_independent=False)
    physical = _graph(task_independent=True)
    with pytest.raises(RuntimeError):
        physical.load_state_dict(legacy.state_dict(), strict=True)


def test_task_independent_host_is_equivariant_to_discovery_row_permutation() -> None:
    first = _graph()
    second = _graph()
    second.load_state_dict(first.state_dict(), strict=True)
    permutation = torch.tensor([1, 0])
    with torch.no_grad():
        second.object_queries.copy_(first.object_queries.index_select(0, permutation))

    inputs, host_mask, positions, visual = _inputs()
    outputs = []
    contexts = []
    runtimes = []
    for graph in (first, second):
        context = _context()
        prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
            inputs_embeds=[value.clone() for value in inputs],
            attention_mask=host_mask.clone(),
            position_ids=positions.clone(),
            visual_pos_masks=visual.clone(),
            context=context,
        )
        assert runtime is not None
        output = _shared_host(
            prepared[0],
            mask[:, : prepared[0].shape[1], : prepared[0].shape[1]],
        )
        graph.finalize_joint_outputs(outputs_embeds=[output, prepared[1]], runtime=runtime)
        outputs.append(output)
        contexts.append(context)
        runtimes.append(runtime)

    for first_slice, second_slice in (
        (runtimes[0].prior_slice, runtimes[1].prior_slice),
        (runtimes[0].posterior_slice, runtimes[1].posterior_slice),
    ):
        torch.testing.assert_close(
            outputs[1][:, second_slice],
            outputs[0][:, first_slice].index_select(1, permutation),
            rtol=1e-5,
            atol=1e-6,
        )
    first_relation = contexts[0].relation_output
    second_relation = contexts[1].relation_output
    assert isinstance(first_relation, PhysicalRelationOutput)
    assert isinstance(second_relation, PhysicalRelationOutput)
    torch.testing.assert_close(
        second_relation.ownership[..., :-1],
        first_relation.ownership[..., :-1].index_select(-1, permutation),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        second_relation.context_probability,
        first_relation.context_probability,
        rtol=1e-5,
        atol=1e-6,
    )


def test_physical_readout_forms_row_plus_context_simplex() -> None:
    readout = PhysicalEntityReadout(8)
    rows = torch.randn(2, 3, 8)
    sensors = torch.randn(2, 4, 8)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    output = readout(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=valid,
    )
    torch.testing.assert_close(
        output.ownership.sum(dim=-1)[valid],
        torch.ones_like(output.ownership[..., 0][valid]),
    )
    assert not output.ownership[~valid].any()
    assert torch.equal(output.object_probability, output.ownership[..., :-1])
    assert torch.equal(output.context_probability, output.ownership[..., -1])
