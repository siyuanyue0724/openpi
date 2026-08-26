from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

import picf_next.videomt_exact.lingbot_joint as joint_module
from picf_next.lingbot_native.host import (
    NATIVE_VIDEOMT_QUERY_POSTERIOR,
    LingBotNativeContext,
    LingBotNativeGraph,
)
from picf_next.lingbot_native.physical_relations import NativeObjectQueryPosteriorOutput
from picf_next.lingbot_native.state import (
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
    NativeVidEoMTPairedPosteriorState,
)
from picf_next.lingbot_native.training import (
    NativePolicyForwardResult,
    NativeV3TwoPassForwardResult,
)
from picf_next.lingbot_native.wsa_lingbot_training_runtime import (
    WSALingBotAttentionIntervention,
)
from picf_next.videomt_exact.joint_training import CompleteCalvinVidEoMTObjective
from picf_next.videomt_exact.lingbot_joint import (
    run_causal_warm_native_videomt_lingbot_evaluation,
    run_cold_native_videomt_lingbot_evaluation,
    run_complete_native_videomt_lingbot_step,
)
from picf_next.videomt_exact.observations import (
    VIDEOMT_MASK_RELATION,
    VIDEOMT_QUERY_MODALITY,
)
from picf_next.videomt_exact.paired_training import (
    run_complete_causal_videomt_training_transaction,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTRuntime
from tests.lingbot_native.test_calvin_entity_training import _batch
from tests.videomt_exact.test_paired_training import (
    _CompleteObjectiveStub,
    _targets,
)
from tests.videomt_exact.test_runtime_contract import _HookCompatibleVidEoMT


def test_complete_joint_step_preserves_paired_boundaries(monkeypatch) -> None:
    runtime = ExactVidEoMTRuntime(object(), _HookCompatibleVidEoMT().train())
    frames = torch.randn(1, 5, 3, 16, 16)
    source_result = run_complete_causal_videomt_training_transaction(
        runtime,
        _CompleteObjectiveStub(),
        normalized_padded_rgb=frames,
        clip_targets=_targets(1),
        previous_queries=None,
        reset=torch.ones(1, dtype=torch.bool),
    )
    previous = NativeVidEoMTPairedPosteriorState(
        layer_rows=torch.randn(1, 3, 200, 8),
        source_queries=torch.randn(1, 200, 1024),
        architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
    )
    captured: dict[str, object] = {}

    def source_forward(*args, **kwargs):
        captured["source_previous"] = kwargs["previous_queries"]
        captured["reset"] = kwargs["reset"]
        return source_result

    def policy_forward(
        policy,
        *,
        model_inputs,
        controls,
        previous_memory,
        previous_memory_valid,
        modalities,
        **kwargs,
    ):
        del policy, model_inputs
        captured["host_previous"] = previous_memory
        captured["previous_valid"] = previous_memory_valid
        captured["action_attention_callback"] = kwargs["action_attention_callback"]
        captured["wsa_da3_teacher_targets"] = kwargs["wsa_da3_teacher_targets"]
        context = LingBotNativeContext(controls=controls, modalities=modalities)
        context.posterior_memory = NativeLayerwisePosteriorState(
            torch.randn(1, 3, 200, 8, requires_grad=True)
        )
        action = torch.tensor(0.25, requires_grad=True)
        return NativeV3TwoPassForwardResult(
            prior_trace=NativeLayerwisePriorTrace(torch.randn(1, 3, 200, 8)),
            policy_forward=NativePolicyForwardResult(
                official_outputs=(action,) * 11,
                official_total_loss=action,
                official_action_loss=action,
                official_moe_regularizer=action * 0,
                context=context,
            ),
        )

    monkeypatch.setattr(
        joint_module,
        "run_complete_causal_videomt_training_transaction",
        source_forward,
    )
    monkeypatch.setattr(
        joint_module,
        "run_native_v3_two_pass_policy_training_forward",
        policy_forward,
    )

    batch = _batch(frame_index=1)
    attention_callback = object()
    teacher_targets = object()
    result = run_complete_native_videomt_lingbot_step(
        nn.Linear(1, 1),
        runtime,
        CompleteCalvinVidEoMTObjective(),
        batch=batch,
        normalized_padded_rgb=frames,
        clip_targets=_targets(1),
        relation_spec=joint_module.NativeObjectQuerySpatialSpec(
            name=VIDEOMT_MASK_RELATION,
            query_modality=VIDEOMT_QUERY_MODALITY,
            geometry_kind="image_grid",
            layout="videomt.calvin.static.2x2.v1",
        ),
        previous_state=previous,
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        host_dtype=torch.float32,
        action_attention_callback=attention_callback,
        wsa_da3_teacher_targets=teacher_targets,
    )

    assert captured["source_previous"] is previous.source_queries
    assert isinstance(captured["host_previous"], NativeLayerwisePosteriorState)
    assert not isinstance(captured["host_previous"], NativeVidEoMTPairedPosteriorState)
    assert captured["host_previous"].layer_rows is previous.layer_rows
    assert captured["action_attention_callback"] is attention_callback
    assert captured["wsa_da3_teacher_targets"] is teacher_targets
    assert result.next_state.source_queries is source_result.current_propagated_queries
    assert result.next_state.layer_rows is result.policy.context.posterior_memory.layer_rows
    source_stream = next(
        stream
        for stream in result.host_batch.modalities.streams
        if stream.name == "videomt_queries"
    )
    assert source_stream.tokens.shape == (1, 200, 1024)
    assert source_stream.valid.all()
    relation = result.host_batch.modalities.object_query_spatial_relations[0]
    assert relation.mask_logits.shape == (1, 200, 4)
    torch.testing.assert_close(
        result.total,
        result.policy.official_total_loss + source_result.source_objective.total,
    )

    masked = run_complete_native_videomt_lingbot_step(
        nn.Linear(1, 1),
        runtime,
        CompleteCalvinVidEoMTObjective(),
        batch=batch,
        normalized_padded_rgb=frames,
        clip_targets=_targets(1),
        relation_spec=joint_module.NativeObjectQuerySpatialSpec(
            name=VIDEOMT_MASK_RELATION,
            query_modality=VIDEOMT_QUERY_MODALITY,
            geometry_kind="image_grid",
            layout="videomt.calvin.static.2x2.v1",
        ),
        previous_state=previous,
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        host_dtype=torch.float32,
        wla_host_evidence_arm="wla_lbot_masked",
    )
    assert captured["source_previous"] is previous.source_queries
    assert isinstance(captured["host_previous"], NativeLayerwisePosteriorState)
    torch.testing.assert_close(
        captured["host_previous"].layer_rows,
        torch.zeros_like(previous.layer_rows),
    )
    assert masked.host_batch.model_inputs is batch.model_inputs
    stream_by_name = {
        stream.name: stream for stream in masked.host_batch.modalities.streams
    }
    source_stream = stream_by_name[VIDEOMT_QUERY_MODALITY]
    assert source_stream.valid.all()
    assert not source_stream.tokens.count_nonzero()
    for name, stream in stream_by_name.items():
        assert not stream.tokens.count_nonzero()
        if name != VIDEOMT_QUERY_MODALITY:
            assert not stream.valid.any()
            if stream.canonical_token_ids is not None:
                assert (stream.canonical_token_ids == -1).all()
    masked_relation = masked.host_batch.modalities.object_query_spatial_relations[0]
    factual_relation = result.host_batch.modalities.object_query_spatial_relations[0]
    torch.testing.assert_close(masked_relation.object_logits, factual_relation.object_logits)
    torch.testing.assert_close(masked_relation.mask_logits, factual_relation.mask_logits)


def test_complete_joint_step_rejects_unknown_wla_evidence_arm() -> None:
    runtime = ExactVidEoMTRuntime(object(), _HookCompatibleVidEoMT().train())
    try:
        run_complete_native_videomt_lingbot_step(
            nn.Linear(1, 1),
            runtime,
            CompleteCalvinVidEoMTObjective(),
            batch=_batch(frame_index=0),
            normalized_padded_rgb=torch.randn(1, 5, 3, 16, 16),
            clip_targets=_targets(1),
            relation_spec=joint_module.NativeObjectQuerySpatialSpec(
                name=VIDEOMT_MASK_RELATION,
                query_modality=VIDEOMT_QUERY_MODALITY,
                geometry_kind="image_grid",
                layout="videomt.calvin.static.2x2.v1",
            ),
            previous_state=None,
            previous_state_valid=torch.zeros(1, dtype=torch.bool),
            host_dtype=torch.float32,
            wla_host_evidence_arm="unknown",
        )
    except ValueError as error:
        assert "unknown WLA host-evidence arm" in str(error)
    else:
        raise AssertionError("unknown WLA host-evidence arm was accepted")


def test_cold_joint_evaluation_is_current_only_target_free_and_graphless(
    monkeypatch,
) -> None:
    runtime = ExactVidEoMTRuntime(object(), _HookCompatibleVidEoMT().train())
    graph = LingBotNativeGraph.__new__(LingBotNativeGraph)
    nn.Module.__init__(graph)
    relation_spec = joint_module.NativeObjectQuerySpatialSpec(
        name=VIDEOMT_MASK_RELATION,
        query_modality=VIDEOMT_QUERY_MODALITY,
        geometry_kind="image_grid",
        layout="videomt.calvin.static.2x2.v1",
    )
    graph.config = SimpleNamespace(
        architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
        object_query_spatial_specs=(relation_spec,),
    )
    captured: dict[str, object] = {}

    def prior_forward(policy, *, graph, control_chunks, require_grad, **kwargs):
        del policy, kwargs
        captured["graph"] = graph
        captured["control_chunks"] = control_chunks
        captured["require_grad"] = require_grad
        return NativeLayerwisePriorTrace(torch.randn(1, 3, 200, 8)), None

    def diagnostic_forward(
        policy,
        *,
        model_inputs,
        context,
        wsa_attention_intervention=None,
    ):
        del policy, model_inputs
        captured["wsa_attention_intervention"] = wsa_attention_intervention
        relation = context.modalities.object_query_spatial_relations[0]
        context.relation_output = NativeObjectQueryPosteriorOutput(
            posterior_rows=torch.randn(1, 200, 8),
            relation=relation,
        )
        context.posterior_memory = NativeLayerwisePosteriorState(
            torch.randn(1, 3, 200, 8)
        )
        action = torch.tensor(0.25)
        return NativePolicyForwardResult(
            official_outputs=(action,) * 11,
            official_total_loss=action,
            official_action_loss=action,
            official_moe_regularizer=action * 0,
            context=context,
        )

    monkeypatch.setattr(joint_module, "run_native_v3_prior_chain", prior_forward)
    monkeypatch.setattr(
        joint_module,
        "run_native_policy_diagnostic_forward",
        diagnostic_forward,
    )
    batch = _batch(frame_index=0)
    result = run_cold_native_videomt_lingbot_evaluation(
        nn.Linear(1, 1),
        runtime,
        graph=graph,
        batch=batch,
        normalized_current_rgb=torch.randn(1, 3, 16, 16),
        relation_spec=relation_spec,
        host_dtype=torch.float32,
        prior_host_steps=1,
        wsa_attention_intervention=(
            WSALingBotAttentionIntervention.BLOCK_FUTURE_TO_ACTION
        ),
    )

    assert result.source_output.class_logits.shape[:3] == (1, 1, 200)
    assert result.source_output.propagated_queries is not runtime.propagated_queries
    assert runtime.propagated_queries is None
    assert captured == {
        "graph": graph,
        "control_chunks": result.host_batch.effective_prior_control_chunks,
        "require_grad": False,
        "wsa_attention_intervention": (
            WSALingBotAttentionIntervention.BLOCK_FUTURE_TO_ACTION
        ),
    }
    assert not result.policy.official_total_loss.requires_grad

    masked = run_cold_native_videomt_lingbot_evaluation(
        nn.Linear(1, 1),
        runtime,
        graph=graph,
        batch=batch,
        normalized_current_rgb=torch.randn(1, 3, 16, 16),
        relation_spec=relation_spec,
        host_dtype=torch.float32,
        prior_host_steps=1,
        wla_host_evidence_arm="wla_lbot_masked",
    )
    stream_by_name = {
        stream.name: stream for stream in masked.host_batch.modalities.streams
    }
    assert stream_by_name[VIDEOMT_QUERY_MODALITY].valid.all()
    assert not stream_by_name[VIDEOMT_QUERY_MODALITY].tokens.count_nonzero()
    for name, stream in stream_by_name.items():
        assert not stream.tokens.count_nonzero()
        if name != VIDEOMT_QUERY_MODALITY:
            assert not stream.valid.any()


def test_causal_warm_joint_evaluation_replays_past_only_and_commits_current(
    monkeypatch,
) -> None:
    runtime = ExactVidEoMTRuntime(object(), _HookCompatibleVidEoMT().train())
    graph = LingBotNativeGraph.__new__(LingBotNativeGraph)
    nn.Module.__init__(graph)
    relation_spec = joint_module.NativeObjectQuerySpatialSpec(
        name=VIDEOMT_MASK_RELATION,
        query_modality=VIDEOMT_QUERY_MODALITY,
        geometry_kind="image_grid",
        layout="videomt.calvin.static.2x2.v1",
    )
    graph.config = SimpleNamespace(
        architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
        object_query_spatial_specs=(relation_spec,),
    )
    prior_previous: list[NativeLayerwisePosteriorState | None] = []
    prior_valid: list[bool] = []
    observation_calls = 0

    def prior_forward(
        policy,
        *,
        previous_memory,
        previous_memory_valid,
        **kwargs,
    ):
        del policy, kwargs
        prior_previous.append(previous_memory)
        prior_valid.append(bool(previous_memory_valid.item()))
        return NativeLayerwisePriorTrace(torch.randn(1, 3, 200, 8)), None

    def observation_forward(policy, *, model_inputs, context):
        nonlocal observation_calls
        del policy, model_inputs
        observation_calls += 1
        relation = context.modalities.object_query_spatial_relations[0]
        context.relation_output = NativeObjectQueryPosteriorOutput(
            posterior_rows=torch.full((1, 200, 8), float(observation_calls)),
            relation=relation,
        )
        context.posterior_memory = NativeLayerwisePosteriorState(
            torch.full((1, 3, 200, 8), float(observation_calls))
        )
        return context

    def action_forward(policy, *, model_inputs, context):
        del policy, model_inputs
        relation = context.modalities.object_query_spatial_relations[0]
        context.relation_output = NativeObjectQueryPosteriorOutput(
            posterior_rows=torch.full((1, 200, 8), 5.0),
            relation=relation,
        )
        context.posterior_memory = NativeLayerwisePosteriorState(
            torch.full((1, 3, 200, 8), 5.0)
        )
        action = torch.tensor(0.125)
        return NativePolicyForwardResult(
            official_outputs=(action,) * 11,
            official_total_loss=action,
            official_action_loss=action,
            official_moe_regularizer=action * 0,
            context=context,
        )

    monkeypatch.setattr(joint_module, "run_native_v3_prior_chain", prior_forward)
    monkeypatch.setattr(
        joint_module,
        "run_native_policy_observation_diagnostic_forward",
        observation_forward,
    )
    monkeypatch.setattr(
        joint_module,
        "run_native_policy_diagnostic_forward",
        action_forward,
    )
    batches = tuple(
        _batch(frame_index=frame_index, source_index=10 + frame_index)
        for frame_index in range(3, 8)
    )
    result = run_causal_warm_native_videomt_lingbot_evaluation(
        nn.Linear(1, 1),
        runtime,
        graph=graph,
        history_batches=batches[:-1],
        current_batch=batches[-1],
        normalized_rgb_sequence=torch.randn(5, 3, 16, 16),
        relation_spec=relation_spec,
        host_dtype=torch.float32,
        prior_host_steps=(1, 1, 1, 1, 1),
    )

    assert observation_calls == 4
    assert prior_valid == [False, True, True, True, True]
    assert prior_previous[0] is None
    for age, previous in enumerate(prior_previous[1:], start=1):
        assert isinstance(previous, NativeLayerwisePosteriorState)
        torch.testing.assert_close(
            previous.layer_rows,
            torch.full_like(previous.layer_rows, float(age)),
        )
    assert len(result.history) == 4
    assert result.current.host_batch.routing.frame_indices == (7,)
    assert result.next_state.layer_rows is result.current.policy.context.posterior_memory.layer_rows
    assert result.next_state.source_queries is result.source_sequence.propagated_queries_by_frame[-1]
    assert runtime.propagated_queries is None
    assert not result.current.policy.official_action_loss.requires_grad

    prior_previous.clear()
    prior_valid.clear()
    observation_calls = 0
    masked = run_causal_warm_native_videomt_lingbot_evaluation(
        nn.Linear(1, 1),
        runtime,
        graph=graph,
        history_batches=batches[:-1],
        current_batch=batches[-1],
        normalized_rgb_sequence=torch.randn(5, 3, 16, 16),
        relation_spec=relation_spec,
        host_dtype=torch.float32,
        prior_host_steps=(1, 1, 1, 1, 1),
        wla_host_evidence_arm="wla_lbot_masked",
    )
    assert prior_valid == [False, True, True, True, True]
    for previous in prior_previous[1:]:
        assert isinstance(previous, NativeLayerwisePosteriorState)
        assert not previous.layer_rows.count_nonzero()
    for diagnostic in (*masked.history, masked.current):
        streams = {stream.name: stream for stream in diagnostic.host_batch.modalities.streams}
        assert streams[VIDEOMT_QUERY_MODALITY].valid.all()
        assert not streams[VIDEOMT_QUERY_MODALITY].tokens.count_nonzero()
        assert all(
            not stream.valid.any()
            for name, stream in streams.items()
            if name != VIDEOMT_QUERY_MODALITY
        )


def test_causal_warm_joint_evaluation_rejects_nonconsecutive_history() -> None:
    runtime = ExactVidEoMTRuntime(object(), _HookCompatibleVidEoMT().train())
    graph = LingBotNativeGraph.__new__(LingBotNativeGraph)
    nn.Module.__init__(graph)
    relation_spec = joint_module.NativeObjectQuerySpatialSpec(
        name=VIDEOMT_MASK_RELATION,
        query_modality=VIDEOMT_QUERY_MODALITY,
        geometry_kind="image_grid",
        layout="videomt.calvin.static.2x2.v1",
    )
    graph.config = SimpleNamespace(
        architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
        object_query_spatial_specs=(relation_spec,),
    )
    batches = (
        _batch(frame_index=1, source_index=11),
        _batch(frame_index=3, source_index=13),
    )

    try:
        run_causal_warm_native_videomt_lingbot_evaluation(
            nn.Linear(1, 1),
            runtime,
            graph=graph,
            history_batches=batches[:-1],
            current_batch=batches[-1],
            normalized_rgb_sequence=torch.randn(2, 3, 16, 16),
            relation_spec=relation_spec,
            host_dtype=torch.float32,
            prior_host_steps=(1, 1),
        )
    except ValueError as error:
        assert "not consecutive" in str(error)
    else:
        raise AssertionError("nonconsecutive causal-warm history was accepted")
