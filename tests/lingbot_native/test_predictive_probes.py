from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.full_training import (
    make_native_current_correction_request,
    make_native_future_request,
)
from picf_next.lingbot_native.host import (
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
    LingBotNativePriorStepper,
)
from picf_next.lingbot_native.prediction import PredictionEvidence, PredictionSource
from picf_next.lingbot_native.predictive_objective import (
    TargetEncoderMode,
    make_native_predictive_target,
)
from picf_next.lingbot_native.predictive_probes import (
    ABSENT_SOURCE,
    BATCH_SHIFT_CONTROL,
    BATCH_SHIFT_SOURCE,
    BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS,
    BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_SCHEMA,
    ROW_SHIFT_SOURCE,
    WRONG_TIME_SOURCE,
    ZERO_CONTROL,
    ZERO_CURRENT_OBSERVATION,
    ZERO_SOURCE,
    NativeBehaviorPosteriorControlCell,
    NativeBehaviorPosteriorControlPredictions,
    batch_shift_executed_control,
    behavior_posterior_control_diagnostics,
    predictive_correction_counterfactual_diagnostics,
    predictive_correction_counterfactual_from_mapping,
    predictive_fixed_batch_fit_diagnostics,
    predictive_fixed_batch_fit_from_mapping,
    predictive_future_counterfactual_diagnostics,
    run_native_behavior_causal_probe,
    run_native_behavior_posterior_control_forwards,
    run_native_correction_counterfactual_forwards,
    run_native_future_counterfactual_forwards,
    zero_current_observation,
    zero_executed_control,
)
from picf_next.lingbot_native.state import NativePosteriorState
from picf_next.lingbot_native.supervision import SequenceAssignment
from tests.lingbot_native.test_training_runtime import (
    _FakeOfficialTrainingPolicy,
    _model_inputs,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _controls() -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.tensor([[[0.2, -0.1]], [[-0.4, 0.3]]]),
        field_valid=torch.ones(2, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(2, 1, dtype=torch.bool),
        delta_time=torch.tensor([[0.1], [0.2]]),
        reset=torch.zeros(2, 1, dtype=torch.bool),
        acknowledged=torch.ones(2, 1, dtype=torch.bool),
    )


def _policy(*, stochastic_observation: bool = False) -> _FakeOfficialTrainingPolicy:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            predictive_target_widths=(("dino_video", 4),),
        )
    )
    if not stochastic_observation:
        return _FakeOfficialTrainingPolicy(graph).train()

    class _StochasticObservationPolicy(_FakeOfficialTrainingPolicy):
        def picf_native_observation_forward(
            self,
            **kwargs: object,
        ) -> tuple[torch.Tensor, ...]:
            images = kwargs.get("images")
            if not isinstance(images, torch.Tensor):
                raise TypeError("stochastic fixture requires tensor images")
            changed = dict(kwargs)
            changed["images"] = images + torch.rand_like(images)
            return super().picf_native_observation_forward(**changed)  # type: ignore[arg-type]

    return _StochasticObservationPolicy(graph).train()


def test_control_counterfactuals_preserve_typed_contract() -> None:
    controls = _controls()
    zero = zero_executed_control(controls)
    shifted = batch_shift_executed_control(controls)

    assert torch.count_nonzero(zero.values) == 0
    torch.testing.assert_close(zero.delta_time, controls.delta_time)
    torch.testing.assert_close(shifted.values[0], controls.values[1])
    torch.testing.assert_close(shifted.delta_time[0], controls.delta_time[1])
    with pytest.raises(ValueError, match="at least two"):
        batch_shift_executed_control(
            ExecutedControlBatch(
                values=controls.values[:1],
                field_valid=controls.field_valid[:1],
                token_valid=controls.token_valid[:1],
                delta_time=controls.delta_time[:1],
                reset=controls.reset[:1],
                acknowledged=controls.acknowledged[:1],
            )
        )


def test_exact_host_counterfactual_probe_freezes_rng_and_isolates_current_observation() -> None:
    policy = _policy()
    state = NativePosteriorState(
        torch.tensor(
            [
                [[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [2.0] * 8],
                [[0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0], [-2.0] * 8],
            ]
        )
    )
    valid = torch.ones(2, dtype=torch.bool)
    request = make_native_current_correction_request(
        batch_size=2,
        valid=valid,
        device="cpu",
        dtype=torch.float32,
    )
    before_rng = torch.random.get_rng_state().clone()
    model_inputs = _model_inputs(2)
    outputs = run_native_correction_counterfactual_forwards(
        policy,
        model_inputs=model_inputs,
        controls=_controls(),
        previous_state=state,
        previous_state_valid=valid,
        request=request,
    )

    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert policy.forward_grad_enabled == []
    assert policy.observation_forward_grad_enabled
    assert not any(policy.observation_forward_grad_enabled)
    assert set(dict(outputs.interventions)) == {
        BATCH_SHIFT_CONTROL,
        BATCH_SHIFT_SOURCE,
        ROW_SHIFT_SOURCE,
        ZERO_CONTROL,
        ZERO_CURRENT_OBSERVATION,
        ZERO_SOURCE,
    }
    assert all(not value.requires_grad for value in outputs.as_mapping().values())
    torch.testing.assert_close(
        dict(outputs.interventions)[ZERO_CURRENT_OBSERVATION],
        outputs.factual,
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(dict(outputs.interventions)[ZERO_SOURCE], outputs.factual)
    assert not torch.allclose(dict(outputs.interventions)[ZERO_CONTROL], outputs.factual)

    # The exact equality above covers both visual and proprioceptive current
    # evidence.  Verify the intervention constructor does not silently omit
    # either physical stream while retaining all public geometry/validity.
    changed_inputs, changed_modalities = zero_current_observation(model_inputs, None)
    assert torch.count_nonzero(changed_inputs["images"]) == 0
    assert torch.count_nonzero(changed_inputs["state"]) == 0
    assert changed_modalities is None
    torch.testing.assert_close(changed_inputs["image_grid_thw"], model_inputs["image_grid_thw"])
    torch.testing.assert_close(changed_inputs["img_masks"], model_inputs["img_masks"])

    target = make_native_predictive_target(
        modality="vision",
        features=outputs.factual.detach().clone(),
        valid=torch.ones(2, 2, 1, dtype=torch.bool),
        importance=torch.ones(2, 2, 1),
        route_ids=request.route_ids,
        horizons=request.horizons,
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.CURRENT_CORRECTION,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=_sha("source"),
        target_data_digest=_sha("target"),
        encoder_digest=_sha("encoder"),
        query_schema_digest=_sha("query"),
        validity_semantics="fixture full support",
        track_identity_keys=(("a", "b"), ("c", "d")),
    )
    diagnostics = predictive_correction_counterfactual_diagnostics(
        outputs,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0, 1], [0, 1]])),
        row_binding_valid=torch.ones(2, 2, dtype=torch.bool),
    )
    assert diagnostics.valid_target_count == 4
    assert diagnostics.factual_loss == pytest.approx(0.0, abs=1e-7)
    by_name = {value.name: value for value in diagnostics.interventions}
    assert by_name[ZERO_CURRENT_OBSERVATION].normalized_prediction_l1 == 0.0
    assert by_name[ZERO_SOURCE].normalized_prediction_l1 > 0
    assert by_name[ZERO_SOURCE].loss_margin_over_factual > 0
    assert predictive_correction_counterfactual_from_mapping(diagnostics.as_dict()) == diagnostics

    source_cut = predictive_correction_counterfactual_diagnostics(
        outputs,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0, 1], [0, 1]])),
        row_binding_valid=torch.tensor([[True, False], [True, False]]),
    )
    assert source_cut.valid_target_count == 2

    edited = diagnostics.as_dict()
    interventions = edited["interventions"]
    assert isinstance(interventions, list)
    interventions[0]["loss_margin_over_factual"] = 123.0
    with pytest.raises(ValueError, match="margin"):
        predictive_correction_counterfactual_from_mapping(edited)


def test_wrong_time_probe_requires_a_complete_typed_pair() -> None:
    state = NativePosteriorState(torch.randn(2, 2, 8))
    valid = torch.ones(2, dtype=torch.bool)
    request = make_native_current_correction_request(
        batch_size=2,
        valid=valid,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="supplied together"):
        run_native_correction_counterfactual_forwards(
            _policy(),
            model_inputs=_model_inputs(2),
            controls=_controls(),
            previous_state=state,
            previous_state_valid=valid,
            request=request,
            wrong_time_state=state,
        )


def test_exact_future_probe_reruns_correction_and_prior_with_one_frozen_target() -> None:
    policy = _policy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    stepper = LingBotNativePriorStepper(policy, graph)
    state = NativePosteriorState(torch.randn(2, 2, 8))
    valid = torch.ones(2, dtype=torch.bool)
    request = make_native_future_request(
        source=PredictionSource.PRIOR,
        batch_size=2,
        horizon=1,
        valid=valid,
        device="cpu",
        dtype=torch.float32,
    )
    before_rng = torch.random.get_rng_state().clone()
    outputs = run_native_future_counterfactual_forwards(
        policy,
        stepper=stepper,
        model_inputs=_model_inputs(2),
        controls=_controls(),
        future_controls=(_controls(),),
        previous_state=state,
        previous_state_valid=valid,
        request=request,
        wrong_time_state=NativePosteriorState(state.rows.roll(1, dims=1)),
        wrong_time_state_valid=valid,
    )

    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert policy.observation_forward_grad_enabled
    assert not any(policy.observation_forward_grad_enabled)
    assert policy.prior_forward_grad_enabled
    assert not any(policy.prior_forward_grad_enabled)
    assert set(dict(outputs.interventions)) == {
        ABSENT_SOURCE,
        BATCH_SHIFT_CONTROL,
        BATCH_SHIFT_SOURCE,
        ROW_SHIFT_SOURCE,
        WRONG_TIME_SOURCE,
        ZERO_CONTROL,
        ZERO_CURRENT_OBSERVATION,
        ZERO_SOURCE,
    }
    assert all(not value.requires_grad for value in outputs.as_mapping().values())

    target = make_native_predictive_target(
        modality="vision",
        features=outputs.factual.detach().clone(),
        valid=torch.ones(2, 2, 1, dtype=torch.bool),
        importance=torch.ones(2, 2, 1),
        route_ids=request.route_ids,
        horizons=request.horizons,
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.FUTURE,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=_sha("future-source"),
        target_data_digest=_sha("future-target"),
        encoder_digest=_sha("future-encoder"),
        query_schema_digest=_sha("future-query"),
        validity_semantics="fixture full future support",
        track_identity_keys=(("a", "b"), ("c", "d")),
    )
    diagnostics = predictive_future_counterfactual_diagnostics(
        outputs,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0, 1], [0, 1]])),
        row_binding_valid=torch.ones(2, 2, dtype=torch.bool),
    )
    assert diagnostics.valid_target_count == 4
    assert diagnostics.factual_loss == pytest.approx(0.0, abs=1e-7)
    assert diagnostics.as_dict()["schema"].endswith("/v1")
    assert {value.name for value in diagnostics.interventions} == set(
        dict(outputs.interventions)
    )


def test_behavior_causal_probe_changes_prediction_but_not_deploy_tensors() -> None:
    policy = _policy()
    controls = _controls()
    valid = torch.ones(2, dtype=torch.bool)
    request = make_native_future_request(
        source=PredictionSource.PRIOR,
        batch_size=2,
        horizon=1,
        valid=valid,
        device="cpu",
        dtype=torch.float32,
    )
    previous_state = NativePosteriorState(torch.randn(2, 2, 8))
    model_inputs = _model_inputs(2)
    before_rng = torch.random.get_rng_state().clone()

    probe = run_native_behavior_causal_probe(
        policy,
        graph=policy.model.qwenvl_with_expert.picf_native_graph,
        model_inputs=model_inputs,
        controls=controls,
        previous_state=previous_state,
        previous_state_valid=valid,
        request=request,
        prediction_controls=controls,
        peer_prediction_controls=batch_shift_executed_control(controls),
    )

    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert probe.horizon == 1
    assert probe.deploy_tensor_count > 0
    assert probe.as_dict()["deploy_bit_identical"] is True
    assert probe.as_dict()["fresh_primary_rerun_bit_identical"] is True
    assert probe.as_dict()["deploy_isolation"] == "separate_same_weight_auxiliary_forward"
    assert any(dict(probe.intervention_prediction_changed).values())


def test_behavior_posterior_control_factorial_is_complete_rng_fixed_and_detached() -> None:
    policy = _policy(stochastic_observation=True)
    controls = _controls()
    request = make_native_future_request(
        source=PredictionSource.PRIOR,
        batch_size=2,
        horizon=1,
        valid=torch.ones(2, dtype=torch.bool),
        device="cpu",
        dtype=torch.float32,
    )
    factual_state = NativePosteriorState(
        torch.tensor(
            [
                [[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [2.0] * 8],
                [[0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0], [-2.0] * 8],
            ]
        )
    )
    peer_state = NativePosteriorState(factual_state.rows.roll(1, dims=0))
    factual_rows_before = factual_state.rows.clone()
    peer_rows_before = peer_state.rows.clone()
    before_rng = torch.random.get_rng_state().clone()
    result = run_native_behavior_posterior_control_forwards(
        policy,
        graph=policy.model.qwenvl_with_expert.picf_native_graph,
        factual_state=factual_state,
        peer_state=peer_state,
        request=request,
        prediction_controls=controls,
        peer_prediction_controls=batch_shift_executed_control(controls),
    )

    assert isinstance(result, NativeBehaviorPosteriorControlPredictions)
    assert result.schema == BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_SCHEMA
    assert tuple(result.as_mapping()) == BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS
    assert len(result.cells) == 9
    assert policy.observation_forward_grad_enabled == []
    assert len(policy.prior_forward_grad_enabled) == 10
    assert not any(policy.prior_forward_grad_enabled)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert torch.equal(factual_state.rows, factual_rows_before)
    assert torch.equal(peer_state.rows, peer_rows_before)
    assert torch.equal(
        result.factual_repeat,
        result.prediction_for("factual", "factual"),
    )
    assert all(
        not prediction.requires_grad
        and prediction.grad_fn is None
        and torch.isfinite(prediction).all()
        for prediction in (*result.as_mapping().values(), result.factual_repeat)
    )
    assert not torch.equal(
        result.prediction_for("factual", "factual"),
        result.prediction_for("zero", "factual"),
    )
    assert not torch.equal(
        result.prediction_for("factual", "factual"),
        result.prediction_for("factual", "batch_shift"),
    )
    with pytest.raises(KeyError):
        result.prediction_for("unsupported", "factual")

    target = make_native_predictive_target(
        modality="vision",
        features=result.prediction_for("factual", "factual").detach().clone(),
        valid=torch.ones(2, 2, 1, dtype=torch.bool),
        importance=torch.ones(2, 2, 1),
        route_ids=request.route_ids,
        horizons=request.horizons,
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.FUTURE,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=_sha("factorial-source"),
        target_data_digest=_sha("factorial-target"),
        encoder_digest=_sha("factorial-encoder"),
        query_schema_digest=_sha("factorial-query"),
        validity_semantics="fixture full support",
        track_identity_keys=(("a", "b"), ("c", "d")),
    )
    diagnostics = behavior_posterior_control_diagnostics(
        result,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0, 1], [0, 1]])),
        row_binding_valid=torch.ones(2, 2, dtype=torch.bool),
    )
    assert diagnostics.valid_target_count == 4
    assert diagnostics.factual_loss == pytest.approx(0.0, abs=1e-7)
    assert tuple(cell.key for cell in diagnostics.cells) == (
        BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS
    )
    assert diagnostics.posterior_margins_at_factual_control["zero"] > 0
    assert diagnostics.control_margins_at_factual_posterior["batch_shift"] > 0
    assert set(diagnostics.interactions) == {
        "zero__zero",
        "zero__batch_shift",
        "batch_shift__zero",
        "batch_shift__batch_shift",
    }
    source_cut = behavior_posterior_control_diagnostics(
        result,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0, 1], [0, 1]])),
        row_binding_valid=torch.tensor([[True, False], [True, False]]),
    )
    assert source_cut.valid_target_count == 2


def test_behavior_posterior_control_result_rejects_incomplete_or_tampered_cells() -> None:
    prediction = torch.zeros(2, 2, 1, 4)
    request = make_native_future_request(
        source=PredictionSource.PRIOR,
        batch_size=2,
        horizon=1,
        valid=torch.ones(2, dtype=torch.bool),
        device="cpu",
        dtype=torch.float32,
    )
    cells = tuple(
        NativeBehaviorPosteriorControlCell(
            posterior_level=posterior_level,
            control_level=control_level,
            prediction=prediction.clone(),
        )
        for posterior_level, control_level in BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS
    )
    result = NativeBehaviorPosteriorControlPredictions(
        request=request,
        cells=cells,
        factual_repeat=prediction.clone(),
    )

    with pytest.raises(ValueError, match="missing, duplicated or out of order"):
        replace(result, cells=result.cells[:-1])
    with pytest.raises(ValueError, match="missing, duplicated or out of order"):
        replace(result, cells=(result.cells[1], result.cells[0], *result.cells[2:]))
    with pytest.raises(ValueError, match="not bit-identical"):
        replace(result, factual_repeat=result.factual_repeat + 1)
    with pytest.raises(ValueError, match="must be detached"):
        replace(
            result.cells[0],
            prediction=result.cells[0].prediction.clone().requires_grad_(),
        )
    nonfinite = result.cells[0].prediction.clone()
    nonfinite[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="must be finite"):
        replace(result.cells[0], prediction=nonfinite)
    with pytest.raises(ValueError, match="unsupported level pair"):
        NativeBehaviorPosteriorControlCell(
            posterior_level="peer",
            control_level="factual",
            prediction=prediction,
        )


def test_fixed_batch_fit_diagnostics_use_equal_budget_and_no_hidden_threshold() -> None:
    result = predictive_fixed_batch_fit_diagnostics(
        {
            "full_host": (1.0, 0.5, 0.1),
            "native_graph_only": (1.0, 0.7, 0.4),
            "readout_only": (1.0, 0.8, 0.6),
            "shuffled_target": torch.tensor([1.0, 0.95, 0.9]),
        }
    )

    assert result.full_host_final_advantage_over_native_graph == pytest.approx(0.3)
    assert result.full_host_final_advantage_over_readout == pytest.approx(0.5)
    assert result.full_host_final_advantage_over_shuffled_target == pytest.approx(0.8)
    assert result.arms[0].absolute_reduction == pytest.approx(0.9)
    assert result.arms[0].normalized_auc == pytest.approx(0.525)
    assert predictive_fixed_batch_fit_from_mapping(result.as_dict()) == result

    edited = result.as_dict()
    arms = edited["arms"]
    assert isinstance(arms, list)
    arms[0]["absolute_reduction"] = 123.0
    with pytest.raises(ValueError, match="summaries"):
        predictive_fixed_batch_fit_from_mapping(edited)

    edited = result.as_dict()
    arms = edited["arms"]
    assert isinstance(arms, list)
    curve = arms[0]["loss_curve"]
    assert isinstance(curve, list)
    curve[1] = 0.9
    with pytest.raises(ValueError, match="summaries"):
        predictive_fixed_batch_fit_from_mapping(edited)
    with pytest.raises(ValueError, match="equal curve-point"):
        predictive_fixed_batch_fit_diagnostics(
            {
                "full_host": (1.0, 0.5),
                "native_graph_only": (1.0, 0.7),
                "readout_only": (1.0, 0.8, 0.6),
                "shuffled_target": (1.0, 0.9),
            }
        )
