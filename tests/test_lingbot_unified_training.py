from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from picf_next.hosts.lingbot_unified import (
    LingBotUnifiedBeliefGraph,
    LingBotUnifiedGraphConfig,
)
from picf_next.hosts.lingbot_unified_training import (
    LingBotUnifiedForwardResult,
    LingBotUnifiedLaneSession,
    LingBotUnifiedSessionConfig,
    LingBotUnifiedStepBatch,
    PreparedLingBotUnifiedSequence,
    combine_lingbot_policy_objective,
    lingbot_optimizer_source_digest,
    run_lingbot_unified_optimizer_attempt,
    run_lingbot_unified_sequence,
)
from picf_next.unified.codec import BeliefCodecConfig
from picf_next.unified.graph import TokenRole
from picf_next.unified.objective import ObjectiveTerm
from picf_next.unified.predictive import (
    ROW_SUMMARY_TARGET,
    PredictionQueryRequest,
    predictive_source_batch_digest,
)
from picf_next.unified.state import GeometrySchema, empty_belief_state
from picf_next.unified.temporal import (
    EpisodeLaneBank,
    LaneStateError,
    SparseBPTTPlan,
    StateStamp,
)


def _graph() -> LingBotUnifiedBeliefGraph:
    return LingBotUnifiedBeliefGraph(
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(3, 2, 1, 32),
            geometry_schema=GeometrySchema(
                names=("x", "y"),
                units=("normalized", "normalized"),
                frame="camera",
            ),
            attention_value_width=32,
            num_layers=3,
            executed_action_dim=2,
        )
    )


def _batch(
    *,
    frame: int,
    reset: bool,
    optimizer_step: int,
    episode: str = "episode-a",
) -> LingBotUnifiedStepBatch:
    return LingBotUnifiedStepBatch(
        lane_ids=(0,),
        episode_keys=(episode,),
        frame_indices=(frame,),
        reset=(reset,),
        optimizer_step=optimizer_step,
        elapsed_time=torch.tensor([0.1]),
        previous_executed_action=torch.tensor([[0.25, -0.5]]),
        previous_action_valid=torch.tensor([frame > 0]),
        modality_geometry_valid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
    )


def _run_toy_host(
    graph: LingBotUnifiedBeliefGraph,
    context,
    *,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    prefix = torch.randn(1, 3, 32)
    action = torch.randn(1, 2, 16)
    attention_mask = torch.ones(1, 5, 5, dtype=torch.bool)
    position_ids = torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5).clone()
    inputs, _, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=torch.tensor([[True, True, False]]),
        context=context,
    )
    assert runtime is not None
    total = inputs[0].shape[1] + inputs[1].shape[1]
    runtime = graph.observe_joint_qkv(
        layer_index=graph.config.penultimate_layer,
        query_states=torch.randn(1, total, 4, 8),
        key_states=torch.randn(1, total, 2, 8),
        value_states=torch.randn(1, total, 2, 8),
        runtime=runtime,
    )
    graph.after_layer(
        layer_index=graph.config.penultimate_layer,
        outputs_embeds=inputs,
        runtime=runtime,
    )


def test_session_and_step_metadata_reject_ambiguous_runtime_types() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        LingBotUnifiedSessionConfig(model_family_digest=7, capacity=2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed must be integers"):
        LingBotUnifiedSessionConfig(
            model_family_digest="fixed-model",
            capacity=2,
            birth_noise_seed=1.5,
        )
    with pytest.raises(ValueError, match="finite"):
        LingBotUnifiedSessionConfig(
            model_family_digest="fixed-model",
            capacity=2,
            birth_hazard=float("nan"),
        )
    with pytest.raises(ValueError, match="frame indices"):
        replace(_batch(frame=0, reset=True, optimizer_step=0), frame_indices=(0.5,))
    with pytest.raises(TypeError, match="reset flags"):
        replace(_batch(frame=0, reset=True, optimizer_step=0), reset=(1,))
    with pytest.raises(ValueError, match="optimizer_step"):
        replace(_batch(frame=0, reset=True, optimizer_step=0), optimizer_step=True)
    with pytest.raises(TypeError, match="burn-in steps"):
        PreparedLingBotUnifiedSequence(
            prepared_steps=(object(),),  # type: ignore[arg-type]
            objectives=(object(),),  # type: ignore[arg-type]
            burn_in_steps=True,
        )


def test_lane_session_closes_a_two_frame_host_loop_and_detaches_state() -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(
            model_family_digest="fixed-model",
            capacity=2,
            birth_noise_seed=7,
        ),
    )
    first = session.prepare(_batch(frame=0, reset=True, optimizer_step=0))
    _run_toy_host(graph, first.context, seed=10)
    assert first.context.posterior is not None
    first_posterior = first.context.posterior.detached()
    session.commit(first)

    second = session.prepare(_batch(frame=1, reset=False, optimizer_step=1))
    for field in first_posterior.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(second.context.previous_posterior, field),
            getattr(first_posterior, field),
        )
    assert all(
        not getattr(second.context.previous_posterior, field).requires_grad
        for field in second.context.previous_posterior.__dataclass_fields__
    )
    _run_toy_host(graph, second.context, seed=11)
    session.commit(second)
    assert len(session.lane_bank) == 1
    with pytest.raises(RuntimeError, match="only be committed once"):
        session.commit(second)


def test_lane_session_restart_snapshot_preserves_the_next_frame_contract() -> None:
    graph = _graph()
    config = LingBotUnifiedSessionConfig(
        model_family_digest="fixed-model",
        capacity=2,
        birth_noise_seed=7,
    )
    session = LingBotUnifiedLaneSession(graph, config)
    first = session.prepare(_batch(frame=0, reset=True, optimizer_step=3))
    _run_toy_host(graph, first.context, seed=20)
    session.commit(first)

    restored = LingBotUnifiedLaneSession(
        graph,
        config,
        lane_bank=EpisodeLaneBank.from_snapshot(session.lane_bank.snapshot()),
    )
    original_next = session.prepare(_batch(frame=1, reset=False, optimizer_step=4))
    restored_next = restored.prepare(_batch(frame=1, reset=False, optimizer_step=4))
    assert restored.schema_digest == session.schema_digest
    assert torch.equal(
        restored_next.context.birth_proposal_noise,
        original_next.context.birth_proposal_noise,
    )
    assert restored_next.context.previous_posterior.serialize() == (
        original_next.context.previous_posterior.serialize()
    )


def test_lane_schema_digest_binds_geometry_names_units_and_frame() -> None:
    graph = _graph()
    baseline = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    baseline.prepare(_batch(frame=0, reset=True, optimizer_step=0))

    changed_graph = LingBotUnifiedBeliefGraph(
        replace(
            graph.config,
            geometry_schema=GeometrySchema(
                names=("u", "v"),
                units=("pixel", "pixel"),
                frame="camera_top",
            ),
        )
    )
    changed = LingBotUnifiedLaneSession(
        changed_graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    changed.prepare(_batch(frame=0, reset=True, optimizer_step=0))
    assert changed.schema_digest != baseline.schema_digest

    changed_fusion = LingBotUnifiedLaneSession(
        LingBotUnifiedBeliefGraph(replace(graph.config, robust_clip=3.0)),
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    assert changed_fusion.schema_digest != baseline.schema_digest


def test_lane_session_requires_explicit_reset_and_successful_forward() -> None:
    session = LingBotUnifiedLaneSession(
        _graph(),
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    with pytest.raises(LaneStateError, match="no cached posterior"):
        session.prepare(_batch(frame=0, reset=False, optimizer_step=0))

    prepared = session.prepare(_batch(frame=0, reset=True, optimizer_step=0))
    with pytest.raises(RuntimeError, match="did not publish"):
        session.commit(prepared)
    assert len(session.lane_bank) == 0
    assert session.last_published_optimizer_step is None
    with pytest.raises(RuntimeError, match="before its first optimizer"):
        session.snapshot()


def test_session_capacity_is_static_and_never_inferred_from_a_failed_batch() -> None:
    session = LingBotUnifiedLaneSession(
        _graph(),
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    wrong_capacity = replace(
        _batch(frame=0, reset=True, optimizer_step=0),
        modality_geometry_valid=torch.ones(1, 1, 3, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="immutable session schema"):
        session.prepare(wrong_capacity)
    assert session.last_published_optimizer_step is None
    prepared = session.prepare(_batch(frame=0, reset=True, optimizer_step=0))
    assert prepared.context.previous_posterior.capacity == 2


def test_lane_batch_write_is_atomic_when_one_record_is_invalid() -> None:
    state = empty_belief_state(
        batch_size=1,
        capacity=2,
        content_dim=3,
        geometry_dim=2,
        uncertainty_dim=1,
    )
    bank = EpisodeLaneBank()
    initial = StateStamp("a", 0, "schema", "model", 0)
    bank.write(0, state, initial)
    bank.write(1, state, initial)
    digest_before = bank.digest
    valid = StateStamp("a", 1, "schema", "model", 1)
    invalid = StateStamp("b", 1, "schema", "model", 1)
    with pytest.raises(LaneStateError, match="episode changed"):
        bank.write_batch(
            (
                (0, state, valid, False),
                (1, state, invalid, False),
            )
        )
    assert bank.digest == digest_before


def test_session_commit_many_is_atomic_across_accumulation_shards() -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    first = session.prepare(_batch(frame=0, reset=True, optimizer_step=0))
    second_batch = replace(
        _batch(frame=0, reset=True, optimizer_step=0, episode="episode-b"),
        lane_ids=(1,),
    )
    second = session.prepare(second_batch)
    first.context.posterior = first.context.previous_posterior
    second.context.posterior = empty_belief_state(
        batch_size=2,
        capacity=2,
        content_dim=3,
        geometry_dim=2,
        uncertainty_dim=1,
    )
    with pytest.raises(RuntimeError, match="posterior batch"):
        session.commit_many((first, second))
    assert len(session.lane_bank) == 0
    assert not first.committed and not second.committed

    second.context.posterior = second.context.previous_posterior
    session.commit_many((first, second))
    assert len(session.lane_bank) == 2
    assert first.committed and second.committed
    with pytest.raises(RuntimeError, match="only be committed once"):
        session.commit_many((first,))


def test_reset_replaces_an_existing_lane_without_identity_carryover() -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    first = session.prepare(_batch(frame=10, reset=True, optimizer_step=0))
    _run_toy_host(graph, first.context, seed=30)
    session.commit(first)

    reset = session.prepare(_batch(frame=4, reset=True, optimizer_step=1, episode="episode-b"))
    expected_empty = empty_belief_state(
        batch_size=1,
        capacity=2,
        content_dim=3,
        geometry_dim=2,
        uncertainty_dim=1,
    )
    assert reset.context.previous_posterior.serialize() == expected_empty.serialize()
    reset.context.posterior = replace(
        expected_empty, content=torch.ones_like(expected_empty.content)
    )
    session.commit(reset)
    next_step = session.prepare(_batch(frame=5, reset=False, optimizer_step=2, episode="episode-b"))
    assert torch.equal(next_step.context.previous_posterior.content, torch.ones(1, 2, 3))


def test_session_passes_complete_multimodal_metadata_without_reinterpretation() -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    batch = _batch(frame=0, reset=True, optimizer_step=0)
    roles = torch.tensor(
        [[int(TokenRole.SENSOR), int(TokenRole.LANGUAGE), int(TokenRole.SENSOR)]],
        dtype=torch.long,
    )
    valid = torch.tensor([[True, True, True]])
    footprint = torch.tensor([[0.25, 0.0, 0.75]])
    modality_ids = torch.tensor([[0, -1, 0]], dtype=torch.long)
    group_ids = torch.tensor([[12, -1, 12]], dtype=torch.long)
    prediction_request = PredictionQueryRequest(
        modality="vision",
        target_kind=ROW_SUMMARY_TARGET,
        horizon=1,
        query_schema_digest="f" * 64,
        source_batch_digest=predictive_source_batch_digest(("episode-a",), (0,)),
        source_batch_size=1,
    )
    enriched = replace(
        batch,
        native_roles=roles,
        native_valid=valid,
        native_footprint=footprint,
        native_modality_ids=modality_ids,
        native_group_ids=group_ids,
        prediction_request=prediction_request,
    )
    prepared = session.prepare(enriched)
    assert prepared.context.native_roles is roles
    assert prepared.context.native_valid is valid
    assert prepared.context.native_footprint is footprint
    with pytest.raises(ValueError, match="different source batch"):
        session.prepare(
            replace(
                enriched,
                prediction_request=replace(
                    prediction_request,
                    source_batch_digest="e" * 64,
                ),
            )
        )
    assert prepared.context.native_modality_ids is modality_ids
    assert prepared.context.native_group_ids is group_ids
    assert prepared.context.prediction_request is prediction_request
    prepared.context.posterior = prepared.context.previous_posterior
    session.commit(prepared)
    assert prediction_request.query_schema_digest.encode() not in session.snapshot()


def test_session_rejects_partial_native_metadata() -> None:
    batch = _batch(frame=0, reset=True, optimizer_step=0)
    with pytest.raises(ValueError, match="supplied completely"):
        replace(
            batch,
            native_roles=torch.tensor([[int(TokenRole.SENSOR)]], dtype=torch.long),
        )


def test_optimizer_transaction_and_atomic_snapshot_close_restart_loop(tmp_path) -> None:
    graph = _graph()
    config = LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2)
    session = LingBotUnifiedLaneSession(graph, config)
    prepared = session.prepare(_batch(frame=0, reset=True, optimizer_step=0))
    prepared.context.posterior = prepared.context.previous_posterior
    checkpoint = tmp_path / "rank-00000.picf-session"

    assert not session.publish_after_optimizer_step(
        (prepared,),
        optimizer_step_succeeded=False,
        checkpoint_path=checkpoint,
    )
    assert len(session.lane_bank) == 0
    assert not prepared.committed
    assert not checkpoint.exists()

    assert session.publish_after_optimizer_step(
        (prepared,),
        optimizer_step_succeeded=True,
        checkpoint_path=checkpoint,
    )
    assert session.last_published_optimizer_step == 0
    assert checkpoint.is_file()
    restored = LingBotUnifiedLaneSession.load_snapshot(
        graph,
        config,
        checkpoint,
        expected_optimizer_step=0,
    )
    original_next = session.prepare(_batch(frame=1, reset=False, optimizer_step=1))
    restored_next = restored.prepare(_batch(frame=1, reset=False, optimizer_step=1))
    assert restored.snapshot() == session.snapshot()
    assert restored_next.context.previous_posterior.serialize() == (
        original_next.context.previous_posterior.serialize()
    )

    corrupted = bytearray(checkpoint.read_bytes())
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="digest differs"):
        LingBotUnifiedLaneSession.from_snapshot(
            graph,
            config,
            bytes(corrupted),
            expected_optimizer_step=0,
        )
    with pytest.raises(ValueError, match="model-family"):
        LingBotUnifiedLaneSession.from_snapshot(
            graph,
            LingBotUnifiedSessionConfig(model_family_digest="different-model", capacity=2),
            checkpoint.read_bytes(),
            expected_optimizer_step=0,
        )
    with pytest.raises(ValueError, match="optimizer checkpoint differs"):
        LingBotUnifiedLaneSession.from_snapshot(
            graph,
            config,
            checkpoint.read_bytes(),
            expected_optimizer_step=1,
        )
    with pytest.raises(ValueError, match="capacity differs"):
        LingBotUnifiedLaneSession.from_snapshot(
            graph,
            replace(config, capacity=3),
            checkpoint.read_bytes(),
            expected_optimizer_step=0,
        )
    with pytest.raises(ValueError, match="process contract differs"):
        LingBotUnifiedLaneSession.from_snapshot(
            graph,
            replace(config, birth_hazard=0.02),
            checkpoint.read_bytes(),
            expected_optimizer_step=0,
        )


def test_checkpoint_write_failure_cannot_publish_in_memory_state(tmp_path) -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    prepared = session.prepare(_batch(frame=0, reset=True, optimizer_step=0))
    prepared.context.posterior = prepared.context.previous_posterior
    invalid_destination = tmp_path / "directory"
    invalid_destination.mkdir()
    with pytest.raises(OSError):
        session.publish_after_optimizer_step(
            (prepared,),
            optimizer_step_succeeded=True,
            checkpoint_path=invalid_destination,
        )
    assert len(session.lane_bank) == 0
    assert not prepared.committed
    assert session.last_published_optimizer_step is None
    assert session.poisoned
    with pytest.raises(RuntimeError, match="restore the last coordinated checkpoint"):
        session.prepare(_batch(frame=0, reset=True, optimizer_step=0))


def test_optimizer_transaction_poisoning_is_fail_stop() -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )

    def forward(index, context, differentiable):
        _run_toy_host(graph, context, seed=70 + index)
        assert context.posterior is not None
        action = context.posterior.content.square().mean()
        return LingBotUnifiedForwardResult((action, action, 0, 0, 0, 0, {}, None, None, None, None))

    sequence = run_lingbot_unified_sequence(
        session,
        (
            _batch(frame=0, reset=True, optimizer_step=0),
            _batch(frame=1, reset=False, optimizer_step=0),
        ),
        SparseBPTTPlan(burn_in_steps=0, differentiable_steps=2, state_age=0),
        forward,
    )

    def broken_optimizer_attempt() -> bool:
        raise OSError("simulated optimizer failure after entering the update boundary")

    with pytest.raises(OSError, match="simulated optimizer failure"):
        session.complete_sequences_optimizer_transaction(
            (sequence,),
            optimizer_attempt=broken_optimizer_attempt,
        )
    assert session.poisoned
    assert len(session.lane_bank) == 0
    with pytest.raises(RuntimeError, match="restore the last coordinated checkpoint"):
        session.snapshot()


def test_official_host_optimizer_bridge_replays_a_skipped_source_exactly(tmp_path) -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    optimizer = torch.optim.AdamW(graph.parameters(), lr=1e-4)
    batch = _batch(frame=0, reset=True, optimizer_step=0)

    def forward(model_inputs, context):
        _run_toy_host(graph, context, seed=int(model_inputs["seed"]))
        assert context.posterior is not None
        action = context.posterior.content.square().mean()
        return LingBotUnifiedForwardResult((action, action, 0, 0, 0, 0, {}, None, None, None, None))

    cleared = 0

    def clear() -> None:
        nonlocal cleared
        cleared += 1
        optimizer.zero_grad(set_to_none=True)

    skipped = run_lingbot_unified_optimizer_attempt(
        session,
        ((batch, {"seed": 101}, "a" * 64),),
        forward_step=forward,
        backward_step=lambda loss: loss.backward(),
        optimizer_attempt=lambda: False,
        clear_gradients_after_skip=clear,
    )
    assert not skipped.published
    assert cleared == 1
    assert len(session.lane_bank) == 0
    assert session.last_published_optimizer_step is None
    assert all(parameter.grad is None for parameter in graph.parameters())

    def update() -> bool:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return True

    checkpoint = tmp_path / "official-bridge.picf-session"
    replay = run_lingbot_unified_optimizer_attempt(
        session,
        ((batch, {"seed": 101}, "a" * 64),),
        forward_step=forward,
        backward_step=lambda loss: loss.backward(),
        optimizer_attempt=update,
        clear_gradients_after_skip=clear,
        checkpoint_path=checkpoint,
    )
    assert replay.published
    assert replay.source_digest == skipped.source_digest
    assert replay.normalized_loss == skipped.normalized_loss
    assert session.last_published_optimizer_step == 0
    assert checkpoint.is_file()
    session.prepare(_batch(frame=1, reset=False, optimizer_step=1))


def test_optimizer_bridge_rejects_duplicate_accumulation_lanes_before_forward() -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    calls = 0

    def forward(model_inputs, context):
        nonlocal calls
        calls += 1
        raise AssertionError("duplicate lanes must fail before forward")

    with pytest.raises(ValueError, match="cannot repeat a lane"):
        run_lingbot_unified_optimizer_attempt(
            session,
            ((_batch(frame=0, reset=True, optimizer_step=0), {}, "a" * 64),) * 2,
            forward_step=forward,
            backward_step=lambda loss: loss.backward(),
            optimizer_attempt=lambda: True,
            clear_gradients_after_skip=lambda: None,
        )
    assert calls == 0


def test_optimizer_bridge_clears_partial_gradients_after_preupdate_failure() -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )
    optimizer = torch.optim.AdamW(graph.parameters(), lr=1e-4)
    first = _batch(frame=0, reset=True, optimizer_step=0)
    second = replace(
        _batch(frame=0, reset=True, optimizer_step=0, episode="episode-b"),
        lane_ids=(1,),
    )
    cleared = 0

    def forward(model_inputs, context):
        if model_inputs["fail"]:
            raise RuntimeError("second microbatch failed")
        _run_toy_host(graph, context, seed=404)
        assert context.posterior is not None
        action = context.posterior.content.square().mean()
        return LingBotUnifiedForwardResult((action, action, 0, 0, 0, 0, {}, None, None, None, None))

    def clear() -> None:
        nonlocal cleared
        cleared += 1
        optimizer.zero_grad(set_to_none=True)

    transaction = (
        (first, {"fail": False}, "a" * 64),
        (second, {"fail": True}, "b" * 64),
    )
    digest = lingbot_optimizer_source_digest(transaction)
    assert digest == lingbot_optimizer_source_digest(transaction)
    with pytest.raises(RuntimeError, match="second microbatch"):
        run_lingbot_unified_optimizer_attempt(
            session,
            transaction,
            forward_step=forward,
            backward_step=lambda loss: loss.backward(),
            optimizer_attempt=lambda: True,
            clear_gradients_after_skip=clear,
        )
    assert cleared == 1
    assert all(parameter.grad is None for parameter in graph.parameters())
    assert len(session.lane_bank) == 0
    assert session.last_published_optimizer_step is None


def test_optimizer_source_digest_rejects_unbound_or_reordered_sources() -> None:
    first = (_batch(frame=0, reset=True, optimizer_step=0), {}, "a" * 64)
    second = (
        replace(
            _batch(frame=0, reset=True, optimizer_step=0, episode="episode-b"),
            lane_ids=(1,),
        ),
        {},
        "b" * 64,
    )
    assert lingbot_optimizer_source_digest((first, second)) != lingbot_optimizer_source_digest(
        (second, first)
    )
    with pytest.raises(ValueError, match="SHA-256"):
        lingbot_optimizer_source_digest(((first[0], {}, "not-a-digest"),))


def test_policy_objective_reconstructs_official_total_before_picf_auxiliaries() -> None:
    action_parameter = torch.tensor(2.0, requires_grad=True)
    host_parameter = torch.tensor(3.0, requires_grad=True)
    picf_parameter = torch.tensor([4.0, 10.0], requires_grad=True)
    action = action_parameter.square()
    current_depth = host_parameter * 0.1
    future_depth = host_parameter * 0.2
    future_video = 0
    sequence = host_parameter * 0.3
    router = host_parameter * 0.4
    total = action + current_depth + future_depth + sequence + router
    model_outputs = (
        total,
        action,
        current_depth,
        future_depth,
        future_video,
        sequence,
        {"router_z_loss": router.detach()},
        None,
        None,
        None,
        None,
    )
    baseline = combine_lingbot_policy_objective(model_outputs)
    torch.testing.assert_close(baseline.total, total)
    assert tuple(baseline.normalized_terms) == (
        "action",
        "host/current_depth",
        "host/sequence",
        "host/remainder",
        "future/host_depth",
    )

    augmented = combine_lingbot_policy_objective(
        model_outputs,
        cross_modal_prediction=(
            ObjectiveTerm(
                "xmod/touch",
                picf_parameter,
                torch.tensor([True, False]),
                weight=0.5,
            ),
        ),
    )
    torch.testing.assert_close(augmented.total, total + 2.0)
    augmented.total.backward()
    torch.testing.assert_close(action_parameter.grad, torch.tensor(4.0))
    torch.testing.assert_close(host_parameter.grad, torch.tensor(1.0))
    torch.testing.assert_close(picf_parameter.grad, torch.tensor([0.5, 0.0]))

    with pytest.raises(ValueError, match="11-item"):
        combine_lingbot_policy_objective(model_outputs[:-1])
    with pytest.raises(TypeError, match="cannot be boolean"):
        combine_lingbot_policy_objective((*model_outputs[:4], False, *model_outputs[5:]))


def test_sparse_short_bptt_keeps_local_credit_and_publishes_atomically(tmp_path) -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(
            model_family_digest="fixed-model",
            capacity=2,
            birth_noise_seed=19,
        ),
    )
    batches = tuple(_batch(frame=frame, reset=frame == 0, optimizer_step=0) for frame in range(3))
    seen_contexts = []

    def forward(index, context, differentiable):
        seen_contexts.append(context)
        assert differentiable == (index >= 1)
        _run_toy_host(graph, context, seed=100 + index)
        assert context.posterior is not None
        action = context.posterior.content.float().square().mean()
        return LingBotUnifiedForwardResult(
            model_outputs=(
                action,
                action,
                0,
                0,
                0,
                0,
                {},
                None,
                None,
                None,
                None,
            )
        )

    sequence = run_lingbot_unified_sequence(
        session,
        batches,
        SparseBPTTPlan(burn_in_steps=1, differentiable_steps=2, state_age=17),
        forward,
    )
    assert len(sequence.objectives) == 2
    assert len(session.lane_bank) == 0
    assert all(
        not getattr(seen_contexts[1].previous_posterior, field).requires_grad
        for field in seen_contexts[1].previous_posterior.__dataclass_fields__
    )
    assert seen_contexts[2].previous_posterior.content.requires_grad

    optimizer = torch.optim.AdamW(graph.parameters(), lr=1e-4)
    optimizer.zero_grad()
    sequence.loss.backward()
    finite_gradients = [
        parameter.grad
        for parameter in graph.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert finite_gradients
    assert all(torch.isfinite(gradient).all() for gradient in finite_gradients)

    checkpoint = tmp_path / "short-bptt.picf-session"
    assert not session.publish_sequences_after_optimizer_step(
        (sequence,),
        optimizer_step_succeeded=False,
        checkpoint_path=checkpoint,
    )
    assert len(session.lane_bank) == 0
    assert not checkpoint.exists()

    def optimizer_attempt() -> bool:
        optimizer.step()
        return True

    assert session.complete_sequences_optimizer_transaction(
        (sequence,),
        optimizer_attempt=optimizer_attempt,
        checkpoint_path=checkpoint,
    )
    assert all(step.committed for step in sequence.prepared_steps)
    restored = LingBotUnifiedLaneSession.load_snapshot(
        graph,
        session.config,
        checkpoint,
        expected_optimizer_step=0,
    )
    continuation = restored.prepare(_batch(frame=3, reset=False, optimizer_step=1))
    assert continuation.context.previous_posterior.serialize() == (
        sequence.prepared_steps[-1].context.posterior.detached().serialize()
    )


def test_short_bptt_rejects_discontinuity_without_mutating_lanes() -> None:
    graph = _graph()
    session = LingBotUnifiedLaneSession(
        graph,
        LingBotUnifiedSessionConfig(model_family_digest="fixed-model", capacity=2),
    )

    def forward(index, context, differentiable):
        _run_toy_host(graph, context, seed=200 + index)
        assert context.posterior is not None
        action = context.posterior.content.square().mean()
        return LingBotUnifiedForwardResult((action, action, 0, 0, 0, 0, {}, None, None, None, None))

    batches = (
        _batch(frame=0, reset=True, optimizer_step=0),
        _batch(frame=2, reset=False, optimizer_step=0),
    )
    with pytest.raises(LaneStateError, match="advance by exactly one"):
        run_lingbot_unified_sequence(
            session,
            batches,
            SparseBPTTPlan(burn_in_steps=0, differentiable_steps=2, state_age=2),
            forward,
        )
    assert len(session.lane_bank) == 0
    assert session.last_published_optimizer_step is None
