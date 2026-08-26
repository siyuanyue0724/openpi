from __future__ import annotations

import io

import pytest
import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
    LingBotPriorRolloutContext,
)
from picf_next.lingbot_native.objective import (
    NativeObjectiveConfig,
    combine_native_objective,
)
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveReadout,
    TargetEncoderMode,
    make_native_predictive_target,
    native_predictive_term,
)
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
    SequenceAssignment,
    sequence_set_terms,
)
from picf_next.lingbot_native.training import audit_native_optimizer_coverage

_DIGESTS = tuple(character * 64 for character in "abcd")


def _components() -> tuple[LingBotNativeGraph, NativePredictiveReadout]:
    torch.manual_seed(55)
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            prediction_route_count=2,
            prediction_address_width=2,
            predictive_target_widths=(("dino_video", 4),),
        )
    )
    return graph, graph.predictive_readout("dino_video")


def _modules(
    graph: LingBotNativeGraph,
    readout: NativePredictiveReadout,
) -> dict[str, torch.nn.Module]:
    assert readout is graph.predictive_readout("dino_video")
    return {"policy.picf_native_graph": graph}


def _shared_host(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = hidden
    weights = mask.to(hidden.dtype) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
    for _ in range(3):
        output = torch.nn.functional.layer_norm(output + weights @ output, (8,))
        output = torch.nn.functional.layer_norm(output + torch.tanh(output), (8,))
    return output


def _objective(
    graph: LingBotNativeGraph,
    readout: NativePredictiveReadout,
) -> torch.Tensor:
    controls = ExecutedControlBatch(
        values=torch.tensor([[[0.25, -0.5]]]),
        field_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        delta_time=torch.full((1, 1), 0.1),
        reset=torch.zeros(1, 1, dtype=torch.bool),
        acknowledged=torch.ones(1, 1, dtype=torch.bool),
    )
    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.FUTURE,
        route_ids=torch.tensor([[0, 1]]),
        horizons=torch.tensor([[1, 2]]),
        addresses=torch.tensor([[[0.25, -0.5], [0.75, 0.5]]]),
        valid=torch.ones(1, 2, dtype=torch.bool),
    )
    context = LingBotNativeContext(
        controls=controls,
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ),
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2]),
    )
    generator = torch.Generator().manual_seed(77)
    prefix = torch.randn(1, 3, 8, generator=generator)
    action = torch.randn(1, 2, 8, generator=generator)
    prepared, attention, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=torch.ones(1, 5, 5, dtype=torch.bool),
        position_ids=torch.zeros(3, 1, 5, dtype=torch.long),
        visual_pos_masks=torch.tensor([[True, True, False]]),
        context=context,
    )
    prefix_count = prepared[0].shape[1]
    hidden = _shared_host(torch.cat((prepared[0], prepared[1]), dim=1), attention)
    prefix_output = hidden[:, :prefix_count]
    action_output = hidden[:, prefix_count:]
    graph.finalize_joint_outputs(
        outputs_embeds=[prefix_output, action_output],
        runtime=runtime,
    )
    assert context.relation_output is not None
    assert context.posterior_state is not None
    rollout = LingBotPriorRolloutContext(
        controls=controls,
        previous_state=context.posterior_state,
        prediction_request=request,
    )
    empty = prefix.new_empty(1, 0, 8)
    rollout_prepared, rollout_attention, _, _, rollout_runtime = graph.prepare_joint_inputs(
        inputs_embeds=[empty, None],
        attention_mask=torch.empty(1, 0, 0, dtype=torch.bool),
        position_ids=torch.empty(3, 1, 0, dtype=torch.long),
        visual_pos_masks=torch.empty(1, 0, dtype=torch.bool),
        context=rollout,
    )
    rollout_hidden = _shared_host(rollout_prepared[0], rollout_attention)
    graph.finalize_joint_outputs(
        outputs_embeds=[rollout_hidden, None],
        runtime=rollout_runtime,
    )
    assert rollout.prediction_hidden is not None
    assert set(rollout.prediction_outputs) == {"dino_video"}
    relation = context.relation_output
    assert relation.ownership_log_probability is not None
    predictions = NativeSequencePredictions(
        support_logits=relation.support_logits.unsqueeze(1),
        ownership=relation.ownership.unsqueeze(1),
        ownership_log_probability=relation.ownership_log_probability.unsqueeze(1),
        existence_logits=relation.existence_logits.unsqueeze(1),
        task_relevance_logits=relation.task_relevance_logits,
        dense_task_grounding_logits=relation.dense_task_grounding_logits.unsqueeze(1),
    )
    targets = NativeSequenceTargets(
        masks=torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]),
        mask_valid=torch.tensor([[[[True, True, False], [True, True, False]]]]),
        existence=torch.ones(1, 1, 2),
        existence_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        task_relevance=torch.tensor([[1.0, 0.0]]),
        task_valid=torch.ones(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.tensor([[[1.0, 1.0, 0.0]]]),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    assignment = SequenceAssignment(torch.tensor([[0, 1]]))
    structural = sequence_set_terms(
        predictions,
        targets,
        assignment,
        support_weight=1.0,
        existence_weight=1.0,
        task_weight=1.0,
        dense_task_weight=1.0,
        ownership_weight=1.0,
    )
    predictive_target = make_native_predictive_target(
        modality="vision",
        features=torch.randn(1, 2, 2, 4, generator=generator),
        valid=torch.ones(1, 2, 2, dtype=torch.bool),
        importance=None,
        route_ids=request.route_ids,
        horizons=request.horizons,
        source=request.source,
        evidence=request.evidence,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=_DIGESTS[0],
        target_data_digest=_DIGESTS[1],
        encoder_digest=_DIGESTS[2],
        query_schema_digest=_DIGESTS[3],
        validity_semantics="complete-toy-track",
        track_identity_keys=(("object/a", "object/b"),),
    )
    predictive = native_predictive_term(
        prediction=rollout.prediction_outputs["dino_video"],
        request=request,
        target=predictive_target,
        assignment=assignment,
        row_binding_valid=assignment.row_to_track >= 0,
        weight=1.0,
    )
    objective = combine_native_objective(
        official_policy_loss=action_output.square().mean(),
        predictive_terms=(predictive,),
        structural_terms=structural,
        config=NativeObjectiveConfig(predictive_weight=1.0, structural_weight=1.0),
    )
    assert objective.valid_counts == {
        "action": 1,
        "rollout/vision/binding": 4,
        "set/support": 0,
        "set/existence": 2,
        "set/task": 1,
        "set/task_dense": 1,
        "set/ownership": 2,
        "set/ownership_nll": 2,
    }
    return objective.total


def test_complete_native_objective_reaches_every_declared_trainable_and_checkpoints() -> None:
    graph, readout = _components()
    modules = _modules(graph, readout)
    optimizer = torch.optim.AdamW(
        [parameter for module in modules.values() for parameter in module.parameters()],
        lr=1e-3,
    )
    manifest = audit_native_optimizer_coverage(modules=modules, optimizer=optimizer)
    assert manifest.parameter_count == 15
    loss = _objective(graph, readout)
    loss.backward()
    for module in modules.values():
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.detach().ne(0).any()
    optimizer.step()

    buffer = io.BytesIO()
    torch.save(
        {
            "graph": graph.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schema_sha256": manifest.schema_sha256,
        },
        buffer,
    )
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored_graph, restored_readout = _components()
    restored_graph.load_state_dict(checkpoint["graph"], strict=True)
    restored_modules = _modules(restored_graph, restored_readout)
    restored_optimizer = torch.optim.AdamW(
        [parameter for module in restored_modules.values() for parameter in module.parameters()],
        lr=1e-3,
    )
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_manifest = audit_native_optimizer_coverage(
        modules=restored_modules,
        optimizer=restored_optimizer,
    )
    assert restored_manifest.schema_sha256 == checkpoint["schema_sha256"]
    for expected, actual in zip(
        graph.state_dict().values(),
        restored_graph.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_optimizer_audit_rejects_an_unowned_predictive_projection() -> None:
    graph, readout = _components()
    optimizer = torch.optim.AdamW(
        [
            parameter
            for name, parameter in graph.named_parameters()
            if not name.startswith("predictive_readouts.")
        ],
        lr=1e-3,
    )
    with pytest.raises(
        ValueError,
        match=r"coverage mismatch.*predictive_readouts\.dino_video\.weight",
    ):
        audit_native_optimizer_coverage(
            modules=_modules(graph, readout),
            optimizer=optimizer,
        )
