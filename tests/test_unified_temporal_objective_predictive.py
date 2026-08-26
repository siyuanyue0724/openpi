from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

torch = pytest.importorskip("torch")

from picf_next.unified.objective import (  # noqa: E402
    DeclaredUnifiedObjective,
    ObjectiveTerm,
    combine_declared_objective,
    combine_objective,
    normalized_scalar_term,
)
from picf_next.unified.predictive import (  # noqa: E402
    ROW_SUMMARY_TARGET,
    PredictionQueryRequest,
    PredictiveTargetProvenance,
    belief_overshooting_term,
    initialize_ema_target_,
    latent_collapse_diagnostics,
    leave_one_modality_out,
    make_belief_state_target,
    make_predictive_target,
    make_row_predictive_target,
    module_state_sha256,
    predictive_source_batch_digest,
    predictive_target_checkpoint_payload,
    predictive_target_loss,
    predictive_target_term,
    restore_predictive_target_checkpoint_,
    row_conditioned_predictive_term,
    update_ema_target_,
)
from picf_next.unified.state import UnifiedBeliefState  # noqa: E402
from picf_next.unified.temporal import (  # noqa: E402
    EpisodeLaneBank,
    LaneStateError,
    PackedHorizonPlan,
    StateStamp,
    assert_deploy_payload_is_causal,
    logarithmic_horizons,
    semigroup_consistency_error,
    sparse_bptt_plan,
)

TARGET_PROVENANCE_DIGEST = "e" * 64
TARGET_MODEL_DIGEST = "d" * 64
TARGET_DATA_DIGEST = "a" * 64
ASSIGNMENT_SCHEMA_DIGEST = "b" * 64
QUERY_SCHEMA_DIGEST = "c" * 64
SOURCE_BATCH_DIGEST = predictive_source_batch_digest(("episode-a",), (3,))


def _state(*, requires_grad: bool = False, offset: float = 0.0) -> UnifiedBeliefState:
    content = (torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) + offset).requires_grad_(
        requires_grad
    )
    return UnifiedBeliefState(
        content=content,
        lifecycle_log_probs=torch.log_softmax(torch.ones(1, 2, 3), dim=-1),
        geometry_mean=torch.zeros(1, 2, 2),
        geometry_information=torch.eye(2).expand(1, 2, 2, 2).clone(),
        geometry_valid=torch.ones(1, 2, 2, dtype=torch.bool),
        content_log_variance=torch.zeros(1, 2, 1),
        expected_age=torch.ones(1, 2),
        evidence_age=torch.ones(1, 2),
    )


def _stamp(frame: int, optimizer_step: int = 10, episode: str = "episode-a") -> StateStamp:
    return StateStamp(
        episode_key=episode,
        frame_index=frame,
        schema_digest="schema-v1",
        model_family_digest="lingbot-v2-picf-v1",
        optimizer_step=optimizer_step,
    )


def test_episode_lane_carries_detached_state_across_long_age() -> None:
    bank = EpisodeLaneBank()
    state = _state(requires_grad=True)
    bank.write(3, state, _stamp(0))
    carried = bank.read_for_next_frame(
        3,
        episode_key="episode-a",
        frame_index=1,
        schema_digest="schema-v1",
        model_family_digest="lingbot-v2-picf-v1",
        optimizer_step=15,
        max_optimizer_lag=10,
    )
    assert carried is not None
    assert not carried.content.requires_grad
    original_value = carried.content.clone()
    state.content.detach().add_(1000)
    torch.testing.assert_close(carried.content, original_value)
    carried.content.add_(500)
    reread = bank.read_for_next_frame(
        3,
        episode_key="episode-a",
        frame_index=1,
        schema_digest="schema-v1",
        model_family_digest="lingbot-v2-picf-v1",
        optimizer_step=15,
        max_optimizer_lag=10,
    )
    assert reread is not None
    torch.testing.assert_close(reread.content, original_value)
    for frame in range(1, 101):
        bank.write(3, _state(offset=float(frame)), _stamp(frame, optimizer_step=15 + frame))
    assert len(bank) == 1


def test_episode_lane_rejects_discontinuity_staleness_and_implicit_reset() -> None:
    bank = EpisodeLaneBank()
    bank.write(0, _state(), _stamp(4))
    common = {
        "lane_id": 0,
        "episode_key": "episode-a",
        "schema_digest": "schema-v1",
        "model_family_digest": "lingbot-v2-picf-v1",
        "optimizer_step": 20,
        "max_optimizer_lag": 5,
    }
    with pytest.raises(LaneStateError, match="contiguous"):
        bank.read_for_next_frame(frame_index=8, **common)
    with pytest.raises(LaneStateError, match="staleness"):
        bank.read_for_next_frame(frame_index=5, **common)
    with pytest.raises(LaneStateError, match="explicit"):
        bank.write(0, _state(), _stamp(0, episode="episode-b"))
    with pytest.raises(LaneStateError, match="schema"):
        bank.write(
            0,
            _state(),
            StateStamp(
                episode_key="episode-a",
                frame_index=5,
                schema_digest="schema-v2",
                model_family_digest="lingbot-v2-picf-v1",
                optimizer_step=11,
            ),
        )
    with pytest.raises(LaneStateError, match="model family"):
        bank.write(
            0,
            _state(),
            StateStamp(
                episode_key="episode-a",
                frame_index=5,
                schema_digest="schema-v1",
                model_family_digest="other-model",
                optimizer_step=11,
            ),
        )
    bank.write(0, _state(), _stamp(0, episode="episode-b"), allow_episode_reset=True)


def test_lane_snapshot_is_deterministic_and_restart_exact() -> None:
    bank = EpisodeLaneBank()
    bank.write(7, _state(offset=2), _stamp(3))
    bank.write(2, _state(offset=1), _stamp(5, episode="episode-b"))
    payload = bank.snapshot()
    assert payload == bank.snapshot()
    restored = EpisodeLaneBank.from_snapshot(payload)
    assert restored.snapshot() == payload
    assert restored.digest == bank.digest
    with pytest.raises(ValueError, match="truncated"):
        EpisodeLaneBank.from_snapshot(payload[:-3])


def test_sparse_bptt_is_low_frequency_and_bounded_to_two_to_four_steps() -> None:
    assert (
        sparse_bptt_plan(
            state_age=100,
            draw=0.9,
            differentiable_probability=0.1,
        )
        is None
    )
    plan = sparse_bptt_plan(
        state_age=100,
        draw=0.05,
        differentiable_probability=0.2,
    )
    assert plan is not None
    assert plan.burn_in_steps == 2
    assert 2 <= plan.differentiable_steps <= 4
    assert plan.loaded_steps <= 6

    with pytest.raises(TypeError, match="must be integers"):
        sparse_bptt_plan(
            state_age=1.5,
            draw=0.05,
            differentiable_probability=0.2,
        )


def test_logarithmic_packed_horizons_cover_long_age_without_linear_unroll() -> None:
    assert logarithmic_horizons(100) == (1, 2, 4, 8, 16, 32, 64, 100)
    plan = PackedHorizonPlan(
        horizons=logarithmic_horizons(100),
        source_frame=40,
        target_data_digest="a" * 64,
        target_model_digest="b" * 64,
    )
    assert len(plan.horizons) == 8
    with pytest.raises(ValueError, match="training-only"):
        PackedHorizonPlan(
            horizons=(1, 2),
            source_frame=0,
            target_data_digest="a" * 64,
            target_model_digest="b" * 64,
            training_only=False,
        )
    with pytest.raises(TypeError, match="horizons must be integers"):
        PackedHorizonPlan(
            horizons=(1, 2.5),
            source_frame=0,
            target_data_digest="a" * 64,
            target_model_digest="b" * 64,
        )
    with pytest.raises(ValueError, match="target_data_digest.*SHA-256"):
        PackedHorizonPlan(
            horizons=(1, 2),
            source_frame=0,
            target_data_digest="data",
            target_model_digest="b" * 64,
        )


def test_deploy_payload_rejects_nested_future_label_and_action_targets() -> None:
    assert_deploy_payload_is_causal({"image": torch.zeros(1), "state": {"proprio": 0}})
    with pytest.raises(ValueError, match="action_target.*future_features"):
        assert_deploy_payload_is_causal(
            {"image": 0, "loss_side": {"future_features": 1, "action_target": 2}}
        )
    with pytest.raises(ValueError, match="target_actions"):
        assert_deploy_payload_is_causal({"Target-Actions": torch.zeros(1)})
    with pytest.raises(ValueError, match="action_targets"):
        assert_deploy_payload_is_causal({"actionTargets": torch.zeros(1)})
    with pytest.raises(ValueError, match="action"):
        assert_deploy_payload_is_causal({"action": torch.zeros(1)})

    @dataclass(frozen=True)
    class HiddenDataclassLeak:
        action_targets: torch.Tensor

    with pytest.raises(ValueError, match="action_targets"):
        assert_deploy_payload_is_causal(
            {"metadata": HiddenDataclassLeak(action_targets=torch.zeros(1))}
        )
    training_plan = PackedHorizonPlan(
        horizons=(1, 2),
        source_frame=0,
        target_data_digest="a" * 64,
        target_model_digest="b" * 64,
    )
    with pytest.raises(ValueError, match="PackedHorizonPlan"):
        assert_deploy_payload_is_causal({"metadata": training_plan})
    request = PredictionQueryRequest(
        modality="touch",
        target_kind=ROW_SUMMARY_TARGET,
        horizon=0,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        source_batch_size=1,
    )
    assert set(request.__dataclass_fields__) == {
        "modality",
        "target_kind",
        "horizon",
        "query_schema_digest",
        "source_batch_digest",
        "source_batch_size",
    }
    with pytest.raises(ValueError, match="PredictionQueryRequest"):
        assert_deploy_payload_is_causal({"metadata": request})


def test_predictive_contract_rejects_symbolic_or_malformed_digests() -> None:
    with pytest.raises(ValueError, match="query_schema_digest"):
        PredictionQueryRequest(
            modality="touch",
            target_kind=ROW_SUMMARY_TARGET,
            horizon=0,
            query_schema_digest="row-query-v1",
            source_batch_digest=SOURCE_BATCH_DIGEST,
            source_batch_size=1,
        )
    with pytest.raises(ValueError, match="source_batch_digest"):
        PredictionQueryRequest(
            modality="touch",
            target_kind=ROW_SUMMARY_TARGET,
            horizon=0,
            query_schema_digest=QUERY_SCHEMA_DIGEST,
            source_batch_digest="episode-a:3",
            source_batch_size=1,
        )
    with pytest.raises(ValueError, match="assignment_digest"):
        make_predictive_target(
            "touch",
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, dtype=torch.bool),
            horizon=0,
            source_batch_digest=SOURCE_BATCH_DIGEST,
            target_data_digest=TARGET_DATA_DIGEST,
            encoder_digest=TARGET_MODEL_DIGEST,
            target_kind=ROW_SUMMARY_TARGET,
            assignment_digest="assignment-v1",
            query_schema_digest=QUERY_SCHEMA_DIGEST,
            validity_semantics="positive physical support",
            provenance_digest=TARGET_PROVENANCE_DIGEST,
        )


def test_semigroup_diagnostic_uses_only_valid_horizons() -> None:
    direct = torch.tensor([[[1.0], [4.0]]])
    recursive = torch.tensor([[[2.0], [100.0]]])
    valid = torch.tensor([[True, False]])
    torch.testing.assert_close(
        semigroup_consistency_error(direct, recursive, valid), torch.tensor(1.0)
    )


def test_objective_normalizes_each_term_by_its_own_valid_count() -> None:
    action_values = torch.tensor([1.0, 3.0], requires_grad=True)
    absent_values = torch.tensor([1000.0], requires_grad=True)
    objective = combine_objective(
        (
            ObjectiveTerm(
                "action",
                action_values,
                torch.tensor([True, True]),
                weight=1.0,
            ),
            ObjectiveTerm(
                "touch",
                absent_values,
                torch.tensor([False]),
                weight=4.0,
            ),
        )
    )
    torch.testing.assert_close(objective.total, torch.tensor(2.0))
    assert objective.valid_counts == {"action": 2, "touch": 0}
    objective.total.backward()
    torch.testing.assert_close(action_values.grad, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(absent_values.grad, torch.tensor([0.0]))
    with pytest.raises(ValueError, match="finite"):
        ObjectiveTerm(
            "invalid",
            torch.tensor([float("nan")]),
            torch.tensor([True]),
            weight=1.0,
        )


def test_leave_one_modality_out_target_is_detached_and_cannot_self_copy() -> None:
    rgb = torch.randn(1, 3, 4, requires_grad=True)
    touch = torch.randn(1, 2, 4, requires_grad=True)
    target = make_predictive_target(
        "touch",
        touch,
        torch.ones(1, 2, dtype=torch.bool),
        horizon=0,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=TARGET_DATA_DIGEST,
        encoder_digest=TARGET_MODEL_DIGEST,
        target_kind=ROW_SUMMARY_TARGET,
        assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        validity_semantics="positive physical support",
        provenance_digest=TARGET_PROVENANCE_DIGEST,
    )
    route = leave_one_modality_out({"rgb": rgb, "touch": touch}, target)
    assert set(route.context) == {"rgb"}
    assert not route.target.features.requires_grad
    prediction = target.features + 1
    loss = predictive_target_loss(prediction, target)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    assert touch.grad is None


def test_declared_joint_law_keeps_action_primary_and_absent_targets_out() -> None:
    action_parameter = torch.tensor(2.0, requires_grad=True)
    predictor_parameter = torch.tensor([1.0, 3.0], requires_grad=True)
    target = make_predictive_target(
        "touch",
        torch.zeros(2, 1),
        torch.tensor([True, False]),
        horizon=0,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=TARGET_DATA_DIGEST,
        encoder_digest=TARGET_MODEL_DIGEST,
        target_kind=ROW_SUMMARY_TARGET,
        assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        validity_semantics="positive physical support",
        provenance_digest=TARGET_PROVENANCE_DIGEST,
    )
    declaration = DeclaredUnifiedObjective(
        action=normalized_scalar_term("action", action_parameter.square()),
        host_regularization=(
            normalized_scalar_term("host/router", action_parameter * 0.0, weight=0.01),
        ),
        cross_modal_prediction=(
            predictive_target_term(
                predictor_parameter[:, None],
                target,
                name="xmod/touch",
                weight=0.5,
            ),
        ),
        future_prediction=(
            ObjectiveTerm(
                "future/vision",
                torch.tensor([100.0], requires_grad=True),
                torch.tensor([False]),
                weight=2.0,
            ),
        ),
    )
    objective = combine_declared_objective(declaration)
    torch.testing.assert_close(objective.total, torch.tensor(4.5))
    assert objective.valid_counts == {
        "action": 1,
        "host/router": 1,
        "xmod/touch": 1,
        "future/vision": 0,
    }
    objective.total.backward()
    torch.testing.assert_close(action_parameter.grad, torch.tensor(4.0))
    torch.testing.assert_close(predictor_parameter.grad, torch.tensor([1.0, 0.0]))

    with pytest.raises(ValueError, match="mandatory action"):
        DeclaredUnifiedObjective(
            action=normalized_scalar_term("host/action", torch.tensor(1.0)),
        )
    with pytest.raises(ValueError, match="must start with xmod"):
        DeclaredUnifiedObjective(
            action=normalized_scalar_term("action", torch.tensor(1.0)),
            cross_modal_prediction=(normalized_scalar_term("future/touch", torch.tensor(1.0)),),
        )


def test_future_target_is_loss_side_only_and_requires_monotonic_time() -> None:
    feature = torch.randn(1, 2, 3)
    target = make_predictive_target(
        "video",
        feature,
        torch.ones(1, 2, dtype=torch.bool),
        horizon=32,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=TARGET_DATA_DIGEST,
        encoder_digest=TARGET_MODEL_DIGEST,
        target_kind="dense_lattice",
        assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        validity_semantics="cached tokenizer lattice validity",
        provenance_digest=TARGET_PROVENANCE_DIGEST,
    )
    assert target.horizon == 32
    with pytest.raises(ValueError, match="horizon"):
        make_predictive_target(
            "video",
            feature,
            torch.ones(1, 2, dtype=torch.bool),
            horizon=-1,
            source_batch_digest=SOURCE_BATCH_DIGEST,
            target_data_digest=TARGET_DATA_DIGEST,
            encoder_digest=TARGET_MODEL_DIGEST,
            target_kind="dense_lattice",
            assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
            query_schema_digest=QUERY_SCHEMA_DIGEST,
            validity_semantics="cached tokenizer lattice validity",
            provenance_digest=TARGET_PROVENANCE_DIGEST,
        )
    with pytest.raises(ValueError, match="finite"):
        make_predictive_target(
            "video",
            torch.full((1, 2, 3), float("nan")),
            torch.ones(1, 2, dtype=torch.bool),
            horizon=1,
            source_batch_digest=SOURCE_BATCH_DIGEST,
            target_data_digest=TARGET_DATA_DIGEST,
            encoder_digest=TARGET_MODEL_DIGEST,
            target_kind="dense_lattice",
            assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
            query_schema_digest=QUERY_SCHEMA_DIGEST,
            validity_semantics="cached tokenizer lattice validity",
            provenance_digest=TARGET_PROVENANCE_DIGEST,
        )
    with pytest.raises(TypeError, match="horizon must be an integer"):
        make_predictive_target(
            "video",
            feature,
            torch.ones(1, 2, dtype=torch.bool),
            horizon=False,
            source_batch_digest=SOURCE_BATCH_DIGEST,
            target_data_digest=TARGET_DATA_DIGEST,
            encoder_digest=TARGET_MODEL_DIGEST,
            target_kind="dense_lattice",
            assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
            query_schema_digest=QUERY_SCHEMA_DIGEST,
            validity_semantics="cached tokenizer lattice validity",
            provenance_digest=TARGET_PROVENANCE_DIGEST,
        )


def test_row_target_aggregation_is_detached_and_support_normalized() -> None:
    features = torch.tensor([[[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]]], requires_grad=True)
    responsibilities = torch.tensor(
        [[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]],
        requires_grad=True,
    )
    target = make_row_predictive_target(
        "touch",
        features,
        responsibilities,
        torch.tensor([[True, True, False]]),
        torch.tensor([[0.5, 0.5, 0.0]]),
        horizon=0,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=TARGET_DATA_DIGEST,
        encoder_digest=TARGET_MODEL_DIGEST,
        assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        validity_semantics="detached non-context support",
        provenance_digest=TARGET_PROVENANCE_DIGEST,
    )
    torch.testing.assert_close(
        target.features,
        torch.tensor([[[7.0 / 3.0, 10.0 / 3.0], [5.0, 6.0]]]),
    )
    assert target.valid.tolist() == [[True, True]]
    assert not target.features.requires_grad
    assert features.grad is None and responsibilities.grad is None
    with pytest.raises(ValueError, match="sub-probability simplex"):
        make_row_predictive_target(
            "touch",
            features,
            responsibilities * 2,
            torch.tensor([[True, True, False]]),
            torch.tensor([[0.5, 0.5, 0.0]]),
            horizon=0,
            source_batch_digest=SOURCE_BATCH_DIGEST,
            target_data_digest=TARGET_DATA_DIGEST,
            encoder_digest=TARGET_MODEL_DIGEST,
            assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
            query_schema_digest=QUERY_SCHEMA_DIGEST,
            validity_semantics="detached non-context support",
            provenance_digest=TARGET_PROVENANCE_DIGEST,
        )


def test_row_target_is_footprint_refinement_invariant_and_support_weighted() -> None:
    common = {
        "horizon": 0,
        "source_batch_digest": SOURCE_BATCH_DIGEST,
        "target_data_digest": TARGET_DATA_DIGEST,
        "encoder_digest": TARGET_MODEL_DIGEST,
        "assignment_digest": ASSIGNMENT_SCHEMA_DIGEST,
        "query_schema_digest": QUERY_SCHEMA_DIGEST,
        "validity_semantics": "continuous normalized physical support",
        "provenance_digest": TARGET_PROVENANCE_DIGEST,
    }
    base = make_row_predictive_target(
        "touch",
        torch.tensor([[[2.0, 0.0], [8.0, 4.0]]]),
        torch.tensor([[[0.8], [0.2]]]),
        torch.tensor([[True, True]]),
        torch.tensor([[0.25, 0.75]]),
        **common,
    )
    refined = make_row_predictive_target(
        "touch",
        torch.tensor([[[2.0, 0.0], [2.0, 0.0], [8.0, 4.0]]]),
        torch.tensor([[[0.8], [0.8], [0.2]]]),
        torch.tensor([[True, True, True]]),
        torch.tensor([[0.125, 0.125, 0.75]]),
        **common,
    )
    torch.testing.assert_close(refined.features, base.features)
    torch.testing.assert_close(refined.importance, base.importance)
    assert base.valid.tolist() == [[True]]

    weighted = make_predictive_target(
        "touch",
        torch.zeros(1, 2, 2),
        torch.ones(1, 2, dtype=torch.bool),
        importance=torch.tensor([[1.0, 0.25]]),
        target_kind=ROW_SUMMARY_TARGET,
        **common,
    )
    prediction = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]], requires_grad=True)
    term = predictive_target_term(prediction, weighted, name="xmod/touch", weight=1.0)
    torch.testing.assert_close(term.normalized(), torch.tensor(1.6))
    term.normalized().backward()
    assert prediction.grad is not None
    torch.testing.assert_close(
        prediction.grad[0, 0].abs().sum() / prediction.grad[0, 1].abs().sum(),
        torch.tensor(2.0),
    )


def test_row_conditioned_term_binds_query_schema_and_horizon() -> None:
    request = PredictionQueryRequest(
        modality="touch",
        target_kind=ROW_SUMMARY_TARGET,
        horizon=0,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        source_batch_size=1,
    )
    target = make_predictive_target(
        "touch",
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        torch.ones(1, 2, dtype=torch.bool),
        horizon=0,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=TARGET_DATA_DIGEST,
        encoder_digest=TARGET_MODEL_DIGEST,
        target_kind=ROW_SUMMARY_TARGET,
        assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        validity_semantics="positive support",
        provenance_digest=TARGET_PROVENANCE_DIGEST,
    )
    prediction = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], requires_grad=True)
    term = row_conditioned_predictive_term(prediction, target, request, weight=0.5)
    assert term.name == "xmod/touch"
    assert term.valid.sum() == 2
    term.normalized().backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()

    with pytest.raises(ValueError, match="horizons"):
        row_conditioned_predictive_term(
            prediction.detach(),
            target,
            PredictionQueryRequest(
                "touch",
                ROW_SUMMARY_TARGET,
                1,
                QUERY_SCHEMA_DIGEST,
                SOURCE_BATCH_DIGEST,
                1,
            ),
            weight=1.0,
        )
    with pytest.raises(ValueError, match="source batches"):
        row_conditioned_predictive_term(
            prediction.detach(),
            target,
            replace(request, source_batch_digest="f" * 64),
            weight=1.0,
        )
    with pytest.raises(ValueError, match="batch size"):
        row_conditioned_predictive_term(
            prediction.detach(),
            target,
            replace(request, source_batch_size=2),
            weight=1.0,
        )
    scalar_target = make_predictive_target(
        "touch",
        torch.ones(1, 2, 1),
        torch.ones(1, 2, dtype=torch.bool),
        horizon=0,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=TARGET_DATA_DIGEST,
        encoder_digest=TARGET_MODEL_DIGEST,
        target_kind=ROW_SUMMARY_TARGET,
        assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        validity_semantics="positive support",
        provenance_digest=TARGET_PROVENANCE_DIGEST,
    )
    with pytest.raises(ValueError, match="at least two feature coordinates"):
        row_conditioned_predictive_term(
            torch.ones(1, 2, 1),
            scalar_target,
            request,
            weight=1.0,
        )


def test_row_conditioned_term_requires_exact_row_pairing_but_is_jointly_equivariant() -> None:
    prediction = torch.tensor([[[2.0, 0.0, -1.0], [-1.0, 3.0, 1.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    request = PredictionQueryRequest(
        modality="touch",
        target_kind=ROW_SUMMARY_TARGET,
        horizon=0,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        source_batch_size=1,
    )
    base_target = make_predictive_target(
        "touch",
        prediction,
        valid,
        horizon=0,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=TARGET_DATA_DIGEST,
        encoder_digest=TARGET_MODEL_DIGEST,
        target_kind=ROW_SUMMARY_TARGET,
        assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        validity_semantics="positive physical support",
        provenance_digest=TARGET_PROVENANCE_DIGEST,
    )
    permutation = torch.tensor([1, 0])
    permuted_features = prediction.index_select(1, permutation)
    permuted_target = make_predictive_target(
        "touch",
        permuted_features,
        valid,
        horizon=0,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=TARGET_DATA_DIGEST,
        encoder_digest=TARGET_MODEL_DIGEST,
        target_kind=ROW_SUMMARY_TARGET,
        assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
        query_schema_digest=QUERY_SCHEMA_DIGEST,
        validity_semantics="positive physical support",
        provenance_digest=TARGET_PROVENANCE_DIGEST,
    )
    base = row_conditioned_predictive_term(
        prediction,
        base_target,
        request,
        weight=1.0,
    ).normalized()
    target_only = row_conditioned_predictive_term(
        prediction,
        permuted_target,
        request,
        weight=1.0,
    ).normalized()
    simultaneous = row_conditioned_predictive_term(
        permuted_features,
        permuted_target,
        request,
        weight=1.0,
    ).normalized()
    torch.testing.assert_close(base, torch.tensor(0.0))
    assert target_only > 0
    torch.testing.assert_close(simultaneous, torch.tensor(0.0))


def test_latent_collapse_diagnostics_distinguish_rank_from_constant_output() -> None:
    diverse = torch.eye(4).reshape(1, 4, 4)
    collapsed = torch.ones_like(diverse)
    valid = torch.ones(1, 4, dtype=torch.bool)
    diverse_stats = latent_collapse_diagnostics(diverse, valid)
    collapsed_stats = latent_collapse_diagnostics(collapsed, valid)
    assert diverse_stats.valid_count == 4
    assert diverse_stats.mean_variance > 0
    assert diverse_stats.effective_rank > 1
    assert collapsed_stats.mean_variance == 0
    assert collapsed_stats.effective_rank == 1.0


def test_ema_target_stem_is_frozen_and_uses_source_derived_momentum_update() -> None:
    online = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.BatchNorm1d(3),
    )
    target = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.BatchNorm1d(3),
    )
    with torch.no_grad():
        online[0].weight.fill_(2.0)
        online[0].bias.fill_(1.0)
        online[1].running_mean.fill_(4.0)
    initialize_ema_target_(target, online)
    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert not target.training
    torch.testing.assert_close(target[0].weight, online[0].weight)

    with torch.no_grad():
        online[0].weight.fill_(6.0)
        online[1].running_mean.fill_(8.0)
    update_ema_target_(target, online, momentum=0.75)
    torch.testing.assert_close(target[0].weight, torch.full_like(target[0].weight, 3.0))
    torch.testing.assert_close(target[1].running_mean, online[1].running_mean)

    target[0].weight.requires_grad_(True)
    with pytest.raises(ValueError, match="remain frozen"):
        update_ema_target_(target, online, momentum=0.9)
    target[0].weight.requires_grad_(False)
    target.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        update_ema_target_(target, online, momentum=0.9)
    with pytest.raises(ValueError, match="momentum"):
        update_ema_target_(target, online, momentum=1.0)
    with pytest.raises(TypeError, match="real-valued"):
        update_ema_target_(target, online, momentum=False)
    with pytest.raises(ValueError, match="dtype"):
        initialize_ema_target_(
            torch.nn.Linear(2, 3, dtype=torch.float64),
            torch.nn.Linear(2, 3, dtype=torch.float32),
        )


def test_predictive_target_checkpoint_binds_state_and_immutable_provenance() -> None:
    torch.manual_seed(43)
    online = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    initialize_ema_target_(target, online)
    target_digest = module_state_sha256(target)
    provenance = PredictiveTargetProvenance(
        modality="touch",
        target_kind=ROW_SUMMARY_TARGET,
        target_data_digest="a" * 64,
        target_model_digest=target_digest,
        assignment_schema_digest="b" * 64,
        query_schema_digest="c" * 64,
        validity_semantics="positive detached target support",
        optimizer_step=19,
    )
    payload = predictive_target_checkpoint_payload(target, provenance)

    restored = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    restored_provenance = restore_predictive_target_checkpoint_(
        restored,
        payload,
        expected_provenance_digest=provenance.digest,
    )
    assert restored_provenance == provenance
    assert module_state_sha256(restored) == target_digest
    assert not restored.training
    assert all(not parameter.requires_grad for parameter in restored.parameters())

    record = make_predictive_target(
        "touch",
        torch.randn(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
        horizon=0,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=provenance.target_data_digest,
        encoder_digest=target_digest,
        target_kind=ROW_SUMMARY_TARGET,
        assignment_digest=provenance.assignment_schema_digest,
        query_schema_digest="c" * 64,
        validity_semantics="positive detached target support",
        provenance_digest=provenance.digest,
    )
    provenance.validate_target(record)
    with pytest.raises(ValueError, match="data manifest"):
        provenance.validate_target(replace(record, target_data_digest="f" * 64))

    raw_state = payload["target"]
    assert isinstance(raw_state, dict)
    tampered_state = {name: value.clone() for name, value in raw_state.items()}
    first_name = next(iter(tampered_state))
    tampered_state[first_name].reshape(-1)[0] += 1
    tampered_payload = {**payload, "target": tampered_state}
    with pytest.raises(ValueError, match="tensor digest"):
        restore_predictive_target_checkpoint_(
            torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4)),
            tampered_payload,
            expected_provenance_digest=provenance.digest,
        )
    with pytest.raises(ValueError, match="provenance digest"):
        restore_predictive_target_checkpoint_(
            torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4)),
            payload,
            expected_provenance_digest="d" * 64,
        )


def test_future_belief_overshooting_is_stop_gradient_and_permutation_invariant() -> None:
    prediction = _state(requires_grad=True, offset=0.5)
    future = _state(offset=2.0)
    target = make_belief_state_target(
        future,
        source_frame=5,
        target_frame=21,
        schema_digest="e" * 64,
        model_digest="f" * 64,
    )
    assert all(
        not getattr(target.state, field).requires_grad
        for field in target.state.__dataclass_fields__
    )
    baseline = belief_overshooting_term(prediction, target, weight=0.25)
    permutation = torch.tensor([1, 0])
    permuted = belief_overshooting_term(
        prediction.permute_rows(permutation),
        target,
        weight=0.25,
    )
    torch.testing.assert_close(permuted.normalized(), baseline.normalized())
    baseline.normalized().backward()
    assert prediction.content.grad is not None
    assert future.content.grad is None
    with pytest.raises(ValueError, match="BeliefStateTarget"):
        assert_deploy_payload_is_causal({"target": target})


def test_objective_and_predictive_controls_reject_boolean_numeric_aliases() -> None:
    with pytest.raises(TypeError, match="objective weight"):
        ObjectiveTerm(
            "bad-weight",
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            weight=True,
        )
    with pytest.raises(TypeError, match="minimum_support"):
        make_row_predictive_target(
            "touch",
            torch.ones(1, 1, 2),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1),
            horizon=0,
            source_batch_digest=SOURCE_BATCH_DIGEST,
            target_data_digest=TARGET_DATA_DIGEST,
            encoder_digest=TARGET_MODEL_DIGEST,
            assignment_digest=ASSIGNMENT_SCHEMA_DIGEST,
            query_schema_digest=QUERY_SCHEMA_DIGEST,
            validity_semantics="positive physical support",
            provenance_digest=TARGET_PROVENANCE_DIGEST,
            minimum_support=False,
        )
