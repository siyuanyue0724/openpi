from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.lingbot_native.calvin import (
    CollatedNativeCALVINBatch,
    NativeCALVINRouting,
)
from picf_next.lingbot_native.calvin_entity_set import physical_frame_row_bindings
from picf_next.lingbot_native.calvin_entity_training import (
    OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT,
    OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU,
    OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD,
    finalize_task_independent_calvin_sequential_omitted_result,
    run_task_independent_calvin_current_frame_diagnostic,
    run_task_independent_calvin_current_frame_objective,
    run_task_independent_calvin_frame_objective,
    run_task_independent_calvin_joint_sequence_objective,
    run_task_independent_calvin_recurrent_frame_diagnostic,
    run_task_independent_calvin_sequence_objective,
    run_task_independent_calvin_sequential_omitted_static_objective,
)
from picf_next.lingbot_native.entity_training import (
    TaskIndependentEntityObjectiveConfig,
)
from picf_next.lingbot_native.host import (
    LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
    TASK_INDEPENDENT_ENTITY_POSTERIOR,
    UNIFIED_LAYERWISE_PREDICT_CORRECT,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
    LingBotNativePriorStepper,
)
from picf_next.lingbot_native.prediction import (
    PredictionEvidence,
    PredictionSource,
    make_native_future_request,
)
from picf_next.lingbot_native.predictive_cache import LingBotPredictiveTargetCache
from picf_next.lingbot_native.predictive_objective import (
    TargetEncoderMode,
    make_native_predictive_target,
)
from picf_next.lingbot_native.source_mask import QwenWholeViewOmission
from picf_next.lingbot_native.state import (
    NativeLayerwisePosteriorState,
    NativePosteriorState,
)
from picf_next.lingbot_native.temporal import rollout_native_prior_prediction
from picf_next.lingbot_native.training import run_native_v3_prior_chain
from tests.lingbot_native.test_calvin_supervision import (
    _frame,
    _official_calvin_model_inputs,
    _request,
    _Sidecar,
)
from tests.lingbot_native.test_training_runtime import (
    _components,
    _controls,
    _FakeOfficialTrainingPolicy,
    _StatefulStochasticTrainingPolicy,
)


def _batch(
    *,
    task_key: str = "push_a",
    source_index: int = 10,
    frame_index: int = 0,
    optimizer_step: int = 0,
) -> CollatedNativeCALVINBatch:
    request = _request(source_index)
    if task_key != request.task_key:
        request = replace(request, task_key=task_key)
    return CollatedNativeCALVINBatch(
        model_inputs=_official_calvin_model_inputs(),
        controls=_controls(1, reset=frame_index == 0),
        routing=NativeCALVINRouting(
            lane_ids=(0,),
            episode_keys=("episode/3",),
            frame_indices=(frame_index,),
            reset=(frame_index == 0,),
            sample_keys=(request.sample_key,),
            optimizer_step=optimizer_step,
        ),
        source_digest="f" * 64,
        structural_target_requests=(request,),
    )


def _sidecar() -> CalvinPhysicalSupervisionSidecar:
    return _Sidecar({10: _frame(("object/a", "object/b"))})


class _ForwardOrderedSidecar(_Sidecar):
    def __init__(self, policy: _FakeOfficialTrainingPolicy) -> None:
        super().__init__({10: _frame(("object/a", "object/b"))})
        self.policy = policy

    def __call__(
        self,
        segment_index: int,
        global_index: int,
    ) -> CalvinPhysicalSupervisionFrame:
        observation_calls = self.policy.observation_forward_grad_enabled
        policy_calls = self.policy.forward_grad_enabled
        assert observation_calls or policy_calls
        return super().__call__(segment_index, global_index)


class _ForwardOrderedPredictiveCache(LingBotPredictiveTargetCache):
    def __init__(self, policy: _FakeOfficialTrainingPolicy) -> None:
        self.policy = policy
        self.calls = 0

    def target_for(self, **arguments: object):  # type: ignore[no-untyped-def]
        assert len(self.policy.observation_forward_grad_enabled) == 2
        assert len(self.policy.prior_forward_grad_enabled) == 1
        self.calls += 1
        request = arguments["request"]
        identities = arguments["track_identity_keys"]
        device = torch.device(arguments["device"])
        assert hasattr(request, "query_count")
        tracks = max(len(value) for value in identities)
        features = torch.zeros(1, tracks, request.query_count, 4, device=device)
        for track_index in range(tracks):
            features[:, track_index, :, track_index % 4] = 1.0
        valid = torch.ones(1, tracks, request.query_count, dtype=torch.bool, device=device)
        return make_native_predictive_target(
            modality="vision",
            features=features,
            valid=valid,
            importance=torch.ones_like(valid, dtype=torch.float32),
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest="1" * 64,
            target_data_digest="2" * 64,
            encoder_digest="3" * 64,
            query_schema_digest="4" * 64,
            validity_semantics="positive synthetic visible-owner support",
            track_identity_keys=identities,
        )


class _CurrentCorrectionCache:
    contract = SimpleNamespace(route_id=0)

    @staticmethod
    def _target(**arguments: object):  # type: ignore[no-untyped-def]
        request = arguments["request"]
        identities = arguments["track_identity_keys"]
        device = torch.device(arguments["device"])
        tracks = max(len(value) for value in identities)
        features = torch.zeros(1, tracks, request.query_count, 4, device=device)
        for track_index in range(tracks):
            features[:, track_index, :, track_index % 4] = 1.0
        valid = torch.ones(1, tracks, request.query_count, dtype=torch.bool, device=device)
        return make_native_predictive_target(
            modality="vision",
            features=features,
            valid=valid,
            importance=torch.ones_like(valid, dtype=torch.float32),
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest="5" * 64,
            target_data_digest="6" * 64,
            encoder_digest="7" * 64,
            query_schema_digest="8" * 64,
            validity_semantics="synthetic current-correction support",
            track_identity_keys=identities,
        )

    def current_correction_summary_target_for(self, **arguments: object):  # type: ignore[no-untyped-def]
        return self._target(**arguments)

    def omitted_static_summary_target_for(self, **arguments: object):  # type: ignore[no-untyped-def]
        return self._target(**arguments)


class _OrderedCurrentFilterCache(_CurrentCorrectionCache):
    def __init__(self, policy: _FakeOfficialTrainingPolicy) -> None:
        self.policy = policy
        self.calls: list[tuple[tuple[int, ...], PredictionEvidence, list[list[bool]]]] = []

    def current_correction_summary_target_for(self, **arguments: object):  # type: ignore[no-untyped-def]
        assert self.policy.call_order == ["prior", "policy", "prior"]
        request = arguments["request"]
        assert hasattr(request, "evidence") and hasattr(request, "valid")
        source_indices = tuple(arguments["source_global_indices"])  # type: ignore[arg-type]
        self.calls.append((source_indices, request.evidence, request.valid.tolist()))
        return self._target(**arguments)


def _predictive_policy() -> _FakeOfficialTrainingPolicy:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
            predictive_target_widths=(("dino_video", 4),),
        )
    )
    return _FakeOfficialTrainingPolicy(graph).train()


def _layerwise_predictive_policy() -> _FakeOfficialTrainingPolicy:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            architecture_identity=LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
            prediction_address_width=0,
            predictive_target_widths=(("dino_video", 4),),
        )
    )
    return _FakeOfficialTrainingPolicy(graph).train()


def _v3_predictive_policy() -> _FakeOfficialTrainingPolicy:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
            prediction_address_width=0,
            predictive_target_widths=(("dino_video", 4),),
        )
    )
    return _FakeOfficialTrainingPolicy(graph).train()


def test_frame_transaction_has_no_task_or_winner_control_surface() -> None:
    parameters = inspect.signature(run_task_independent_calvin_frame_objective).parameters
    assert not any("task" in name or "winner" in name or "match" in name for name in parameters)
    sequence_parameters = inspect.signature(
        run_task_independent_calvin_sequence_objective
    ).parameters
    assert not any(
        "task" in name or "winner" in name or "match" in name for name in sequence_parameters
    )
    joint_parameters = inspect.signature(
        run_task_independent_calvin_joint_sequence_objective
    ).parameters
    assert not any(
        "task" in name or "winner" in name or "match" in name for name in joint_parameters
    )


def test_entity_only_frame_transaction_uses_observation_root_and_backpropagates() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )

    result = run_task_independent_calvin_frame_objective(
        policy,
        batch=_batch(),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=0.0,
            entity_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
    )

    assert result.policy_forward is None
    assert policy.observation_forward_grad_enabled == [True]
    assert not policy.forward_grad_enabled
    assert (
        result.targets.targets.masks.shape == result.relation.support_logits.transpose(1, 2).shape
    )
    result.objective.objective.total.backward()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in graph.parameters()
    )


def test_current_frame_transaction_uses_discovery_without_recurrent_state() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    result = run_task_independent_calvin_current_frame_objective(
        policy,
        batch=_batch(frame_index=1),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
        patch_size=1,
        merge_size=2,
    )

    assert result.context.previous_state is None
    assert result.context.previous_state_valid.tolist() == [False]
    assert result.context.posterior_state is not None


def test_current_frame_diagnostic_uses_complete_official_no_grad_root() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    result = run_task_independent_calvin_current_frame_diagnostic(
        policy,
        batch=_batch(frame_index=1),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
        patch_size=1,
        merge_size=2,
    )

    assert policy.training
    assert policy.forward_grad_enabled == [False]
    assert not policy.observation_forward_grad_enabled
    assert not result.objective.objective.total.requires_grad
    assert result.context.previous_state is None
    assert result.context.previous_state_valid.tolist() == [False]
    assert result.policy_forward is not None
    assert not result.policy_forward.official_action_loss.requires_grad


def test_complete_diagnostic_relation_is_invariant_to_action_suffix_targets() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    batch = _batch(frame_index=1)
    changed_inputs = dict(batch.model_inputs)
    for name in ("actions", "noise", "time", "joint_mask", "action_is_pad"):
        changed_inputs[name] = torch.zeros_like(changed_inputs[name])

    torch.manual_seed(17)
    factual = run_task_independent_calvin_current_frame_diagnostic(
        policy,
        batch=batch,
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
        patch_size=1,
        merge_size=2,
    )
    torch.manual_seed(17)
    changed = run_task_independent_calvin_current_frame_diagnostic(
        policy,
        batch=replace(batch, model_inputs=changed_inputs),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
        patch_size=1,
        merge_size=2,
    )

    for actual, expected in zip(
        factual.context.root_output_tensors(),
        changed.context.root_output_tensors(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_current_frame_diagnostic_rejects_action_suffix() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    with pytest.raises(ValueError, match="cannot score the action suffix"):
        run_task_independent_calvin_current_frame_diagnostic(
            policy,
            batch=_batch(),
            physical_sidecar=_sidecar(),
            objective_config=TaskIndependentEntityObjectiveConfig(action_weight=1.0),
            patch_size=1,
            merge_size=2,
        )


def test_recurrent_frame_diagnostic_holds_the_loss_side_gauge_without_gradients() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    previous_state = torch.zeros(1, 2, 8)
    previous_state[:, 0, 0] = 1
    result = run_task_independent_calvin_recurrent_frame_diagnostic(
        policy,
        batch=_batch(frame_index=1),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
        patch_size=1,
        merge_size=2,
        previous_state=NativePosteriorState(previous_state),
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((("object/a", 0),),),
    )

    assert policy.forward_grad_enabled == [False]
    assert not policy.observation_forward_grad_enabled
    assert not result.objective.objective.total.requires_grad
    assert result.diagnostic_action_loss is not None
    assert result.diagnostic_action_loss.ndim == 0
    assert torch.isfinite(result.diagnostic_action_loss)
    assert not result.diagnostic_action_loss.requires_grad
    assert result.contexts[0].previous_state is not None
    assert result.contexts[0].previous_state_valid.tolist() == [True]
    assert dict(result.objective.row_bindings_by_batch[0])["object/a"] == 0


@pytest.mark.parametrize(
    "config",
    (
        TaskIndependentEntityObjectiveConfig(action_weight=1.0),
        TaskIndependentEntityObjectiveConfig(action_weight=0.0, predictive_weight=1.0),
    ),
)
def test_recurrent_frame_diagnostic_rejects_action_and_prediction(
    config: TaskIndependentEntityObjectiveConfig,
) -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    with pytest.raises(ValueError, match="cannot execute"):
        run_task_independent_calvin_recurrent_frame_diagnostic(
            policy,
            batch=_batch(frame_index=1),
            physical_sidecar=_sidecar(),
            objective_config=config,
            patch_size=1,
            merge_size=2,
            previous_state=NativePosteriorState(torch.zeros(1, 2, 8)),
            previous_state_valid=torch.ones(1, dtype=torch.bool),
            prior_row_bindings_by_batch=((),),
        )


def test_physical_assignment_becomes_loss_side_lane_metadata() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    result = run_task_independent_calvin_frame_objective(
        policy,
        batch=_batch(),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
        patch_size=1,
        merge_size=2,
    )

    bindings = physical_frame_row_bindings(
        result.targets,
        result.objective.frame_losses[0].assignment,
        capacity=result.relation.support_logits.shape[-1],
    )

    assert len(bindings) == 1
    assert {identity for identity, _row in bindings[0]} == {"object/a", "object/b"}
    assert len({row for _identity, row in bindings[0]}) == 2


def test_sidecar_is_not_read_until_after_the_deploy_visible_host_forward() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )

    run_task_independent_calvin_frame_objective(
        policy,
        batch=_batch(),
        physical_sidecar=_ForwardOrderedSidecar(policy),
        objective_config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
        patch_size=1,
        merge_size=2,
    )


def test_joint_frame_transaction_uses_complete_official_policy_loss() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )

    result = run_task_independent_calvin_frame_objective(
        policy,
        batch=_batch(),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=0.25,
        ),
        patch_size=1,
        merge_size=2,
    )

    assert result.policy_forward is not None
    torch.testing.assert_close(
        result.objective.objective.normalized_terms["action"],
        result.policy_forward.official_total_loss,
    )
    result.objective.objective.total.backward()
    assert policy.forward_grad_enabled == [True]


def test_loss_side_targets_are_task_key_invariant() -> None:
    first_policy, _ = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    second_policy, _ = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    second_policy.load_state_dict(first_policy.state_dict())
    config = TaskIndependentEntityObjectiveConfig(action_weight=0.0)

    first = run_task_independent_calvin_frame_objective(
        first_policy,
        batch=_batch(task_key="push_a"),
        physical_sidecar=_sidecar(),
        objective_config=config,
        patch_size=1,
        merge_size=2,
    )
    second = run_task_independent_calvin_frame_objective(
        second_policy,
        batch=_batch(task_key="unseen_prompt"),
        physical_sidecar=_sidecar(),
        objective_config=config,
        patch_size=1,
        merge_size=2,
    )

    torch.testing.assert_close(first.targets.targets.masks, second.targets.targets.masks)
    torch.testing.assert_close(
        first.targets.targets.existence,
        second.targets.targets.existence,
    )


def test_recurrent_sequence_uses_one_physical_gauge_and_commits_primary_births() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    sidecar = _Sidecar(
        {
            10: _frame(("object/a",)),
            11: _frame(("object/a", "object/b")),
        }
    )
    first = _batch(source_index=10, frame_index=0)
    second = _batch(source_index=11, frame_index=1)

    result = run_task_independent_calvin_sequence_objective(
        policy,
        batches=(first, second),
        physical_sidecar=sidecar,
        objective_config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
        patch_size=1,
        merge_size=2,
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
    )

    assert len(result.contexts) == len(result.relations) == len(result.targets) == 2
    assert result.committable_context is result.contexts[0]
    assert result.contexts[0].posterior_state is not None
    assert result.contexts[1].posterior_state is not None
    assert result.contexts[1].posterior_state.rows.requires_grad
    assert len(result.objective.row_bindings_by_batch[0]) == 1
    assert result.objective.row_bindings_by_batch[0][0][0] == "object/a"
    result.objective.objective.total.backward()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in graph.parameters()
    )


def test_joint_sequence_runs_action_once_and_local_credit_through_shared_host() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    sidecar = _Sidecar(
        {
            10: _frame(("object/a",)),
            11: _frame(("object/a", "object/b")),
        }
    )
    batches = (
        _batch(source_index=10, frame_index=0),
        _batch(source_index=11, frame_index=1),
    )

    result = run_task_independent_calvin_joint_sequence_objective(
        policy,
        batches=batches,
        physical_sidecar=sidecar,
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
    )

    assert policy.forward_grad_enabled == [True]
    assert policy.observation_forward_grad_enabled == [True]
    assert len(result.auxiliary) == 1
    assert len(result.relations) == len(result.targets) == 2
    assert result.committable_context is result.primary.context
    assert result.committable_context.posterior_state is not None
    torch.testing.assert_close(
        result.objective.objective.normalized_terms["action"],
        result.primary.official_total_loss,
    )
    result.objective.objective.total.backward()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in graph.parameters()
    )


def test_joint_single_frame_preserves_persistent_loss_side_gauge() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )

    result = run_task_independent_calvin_joint_sequence_objective(
        policy,
        batches=(_batch(),),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(action_weight=1.0),
        patch_size=1,
        merge_size=2,
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
    )

    assert result.auxiliary == ()
    assert len(result.objective.row_bindings_by_batch) == 1
    assert result.committable_context.posterior_state is not None
    result.objective.objective.total.backward()
    assert policy.forward_grad_enabled == [True]


def test_joint_sequence_attaches_mature_current_correction_to_the_same_host() -> None:
    policy = _predictive_policy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    sidecar = _Sidecar(
        {
            10: _frame(("object/a",)),
            11: _frame(("object/a", "object/b")),
        }
    )
    result = run_task_independent_calvin_joint_sequence_objective(
        policy,
        batches=(
            _batch(source_index=10, frame_index=0),
            _batch(source_index=11, frame_index=1),
        ),
        physical_sidecar=sidecar,
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
        graph=graph,
        current_grid_cache=_CurrentCorrectionCache(),  # type: ignore[arg-type]
    )

    assert len(result.correction_branches) == 2
    assert result.current_grid_branch is None
    assert "correction/vision/binding" in result.objective.objective.normalized_terms
    result.objective.objective.total.backward()
    assert policy.forward_grad_enabled == [True]
    assert policy.observation_forward_grad_enabled == [True]


def test_layerwise_joint_step_executes_prior_current_correction_in_shared_host() -> None:
    policy = _layerwise_predictive_policy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    previous_memory = NativeLayerwisePosteriorState(torch.randn(1, 3, 2, 8))
    result = run_task_independent_calvin_joint_sequence_objective(
        policy,
        batches=(_batch(source_index=10, frame_index=1),),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=previous_memory,
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((("object/a", 0), ("object/b", 1)),),
        graph=graph,
        current_grid_cache=_CurrentCorrectionCache(),  # type: ignore[arg-type]
    )

    assert len(result.correction_branches) == 1
    request = result.correction_branches[0].request
    assert request.source is PredictionSource.PRIOR
    assert request.address_width == 0
    assert request.valid.tolist() == [[True]]
    assert result.committable_context.posterior_memory is not None
    assert "correction/vision/binding" in result.objective.objective.normalized_terms
    result.objective.objective.total.backward()
    readout = graph.predictive_readout("dino_video")
    assert readout.weight.grad is not None
    assert torch.isfinite(readout.weight.grad).all()
    assert bool(torch.count_nonzero(readout.weight.grad))


def test_v3_joint_step_composes_explicit_prior_and_posterior_filter_phases() -> None:
    policy = _v3_predictive_policy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    previous_memory = NativeLayerwisePosteriorState(torch.randn(1, 3, 2, 8))
    result = run_task_independent_calvin_joint_sequence_objective(
        policy,
        batches=(_batch(source_index=10, frame_index=1),),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=previous_memory,
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((("object/a", 0), ("object/b", 1)),),
        graph=graph,
        current_grid_cache=_CurrentCorrectionCache(),  # type: ignore[arg-type]
    )

    assert policy.call_order == ["prior", "policy"]
    assert result.correction_branches == ()
    assert len(result.filter_phase_branches) == 2
    assert [branch.request.evidence for branch in result.filter_phase_branches] == [
        PredictionEvidence.CURRENT_PRIOR,
        PredictionEvidence.CURRENT_POSTERIOR,
    ]
    assert [branch.identity_source_phase for branch in result.filter_phase_branches] == [0, 1]
    assert result.primary.context.previous_memory is None
    assert result.primary.context.prior_trace is result.v3_prior_traces[0]
    assert "filter_prior/vision/binding" in result.objective.objective.normalized_terms
    assert "filter_posterior/vision/binding" in result.objective.objective.normalized_terms
    result.objective.objective.total.backward()
    assert graph.predictive_readout("dino_video").weight.grad is not None


def test_v3_joint_step_exposes_primary_action_attention_callback() -> None:
    policy = _v3_predictive_policy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph

    def callback(**_surface: object) -> None:
        return None

    result = run_task_independent_calvin_joint_sequence_objective(
        policy,
        batches=(_batch(source_index=10, frame_index=0),),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
        graph=graph,
        current_grid_cache=_CurrentCorrectionCache(),  # type: ignore[arg-type]
        factual_action_attention_callback=callback,
    )

    assert result.primary.context.posterior_memory is not None
    assert policy.action_attention_callbacks == [callback]


def test_v3_reset_keeps_current_posterior_supervision_active() -> None:
    policy = _v3_predictive_policy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    result = run_task_independent_calvin_joint_sequence_objective(
        policy,
        batches=(_batch(source_index=10, frame_index=0),),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
        graph=graph,
        current_grid_cache=_CurrentCorrectionCache(),  # type: ignore[arg-type]
    )

    prior, posterior = result.filter_phase_branches
    assert prior.request.valid.tolist() == [[False]]
    assert posterior.request.valid.tolist() == [[True]]
    assert result.objective.objective.valid_counts["filter_prior/vision/binding"] == 0
    assert result.objective.objective.valid_counts["filter_posterior/vision/binding"] > 0


def test_v3_production_egress_adds_only_one_attached_low_token_prior() -> None:
    policy = _v3_predictive_policy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    sidecar = _Sidecar(
        {
            10: _frame(("object/a", "object/b")),
            11: _frame(("object/a", "object/b")),
        }
    )
    cache = _OrderedCurrentFilterCache(policy)
    result = run_task_independent_calvin_joint_sequence_objective(
        policy,
        batches=(_batch(source_index=10, frame_index=1),),
        egress_batch=_batch(source_index=11, frame_index=2),
        physical_sidecar=sidecar,
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=NativeLayerwisePosteriorState(torch.randn(1, 3, 2, 8)),
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((("object/a", 0), ("object/b", 1)),),
        graph=graph,
        current_grid_cache=cache,  # type: ignore[arg-type]
    )

    assert policy.call_order == ["prior", "policy", "prior"]
    assert policy.forward_grad_enabled == [True]
    assert policy.observation_forward_grad_enabled == []
    assert result.attached_egress is not None
    assert not hasattr(result.attached_egress, "context")
    assert [branch.identity_source_phase for branch in result.filter_phase_branches] == [
        0,
        1,
        1,
    ]
    assert [branch.request.evidence for branch in result.filter_phase_branches] == [
        PredictionEvidence.CURRENT_PRIOR,
        PredictionEvidence.CURRENT_POSTERIOR,
        PredictionEvidence.CURRENT_PRIOR,
    ]
    assert cache.calls == [
        ((10,), PredictionEvidence.CURRENT_PRIOR, [[True]]),
        ((10,), PredictionEvidence.CURRENT_POSTERIOR, [[True]]),
        ((11,), PredictionEvidence.CURRENT_PRIOR, [[True]]),
    ]
    result.objective.objective.total.backward()
    assert result.primary.context.posterior_memory is not None
    assert result.primary.context.posterior_memory.layer_rows.grad_fn is not None


def test_v3_rejects_an_egress_schedule_without_an_egress_batch() -> None:
    policy = _v3_predictive_policy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph

    with pytest.raises(ValueError, match="requires an egress batch"):
        run_task_independent_calvin_joint_sequence_objective(
            policy,
            batches=(_batch(source_index=10, frame_index=0),),
            physical_sidecar=_sidecar(),
            objective_config=TaskIndependentEntityObjectiveConfig(action_weight=1.0),
            patch_size=1,
            merge_size=2,
            previous_state=None,
            previous_state_valid=torch.zeros(1, dtype=torch.bool),
            prior_row_bindings_by_batch=((),),
            graph=graph,
            current_grid_cache=_CurrentCorrectionCache(),  # type: ignore[arg-type]
            egress_prior_host_steps=1,
        )


def test_v3_omitted_static_branch_reuses_prior_and_runs_complete_action_root() -> None:
    policy = _v3_predictive_policy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    batch = _batch(source_index=10, frame_index=1)
    omission = QwenWholeViewOmission(
        omitted_view_index=0,
        image_grid_thw=batch.model_inputs["image_grid_thw"],
        image_valid=batch.model_inputs["img_masks"],
        seed=73,
    )
    result = run_task_independent_calvin_joint_sequence_objective(
        policy,
        batches=(batch,),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=NativeLayerwisePosteriorState(torch.randn(1, 3, 2, 8)),
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((("object/a", 0), ("object/b", 1)),),
        graph=graph,
        current_grid_cache=_CurrentCorrectionCache(),  # type: ignore[arg-type]
        omitted_static_view=omission,
    )

    assert policy.call_order == ["prior", "policy", "policy"]
    assert result.omitted_static_policy is not None
    assert result.omitted_static_branch is not None
    assert result.omitted_static_policy.official_action_loss.requires_grad
    assert not hasattr(result.omitted_static_policy, "context")
    assert not hasattr(result.omitted_static_policy, "committable_context")
    assert policy.forward_contexts[1].prior_trace is result.v3_prior_traces[0]
    assert all(
        factual is omitted
        for factual, omitted in zip(
            policy.action_randomness[0],
            policy.action_randomness[1],
            strict=True,
        )
    )
    torch.testing.assert_close(
        result.objective.objective.normalized_terms["action"],
        torch.stack(
            (
                result.primary.official_total_loss,
                result.omitted_static_policy.official_total_loss,
            )
        ).mean(),
    )
    assert len(result.relations) == 1


def test_v3_omitted_static_complete_checkpoint_preserves_gradient_and_runtime_state() -> None:
    def graph() -> LingBotNativeGraph:
        return LingBotNativeGraph(
            LingBotNativeGraphConfig(
                capacity=2,
                host_width=8,
                executed_action_dim=2,
                num_layers=3,
                architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
                prediction_address_width=0,
                predictive_target_widths=(("dino_video", 4),),
            )
        )

    batch = _batch(source_index=10, frame_index=1)
    omission = QwenWholeViewOmission(
        omitted_view_index=0,
        image_grid_thw=batch.model_inputs["image_grid_thw"],
        image_valid=batch.model_inputs["img_masks"],
        seed=73,
    )
    previous_rows = torch.linspace(-1, 1, steps=48).reshape(1, 3, 2, 8)

    torch.manual_seed(101)
    baseline_policy = _StatefulStochasticTrainingPolicy(graph()).train()
    torch.manual_seed(202)
    baseline = run_task_independent_calvin_joint_sequence_objective(
        baseline_policy,
        batches=(batch,),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=NativeLayerwisePosteriorState(previous_rows.clone()),
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((('object/a', 0), ('object/b', 1)),),
        graph=baseline_policy.model.qwenvl_with_expert.picf_native_graph,
        current_grid_cache=_CurrentCorrectionCache(),  # type: ignore[arg-type]
        omitted_static_view=omission,
    )
    baseline_forward_buffers = baseline_policy.tokens_per_expert.detach().clone()
    baseline.objective.objective.total.backward()
    baseline_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in baseline_policy.named_parameters()
        if parameter.grad is not None
    }
    torch.testing.assert_close(baseline_policy.tokens_per_expert, baseline_forward_buffers)

    torch.manual_seed(101)
    checkpoint_policy = _StatefulStochasticTrainingPolicy(graph()).train()

    @contextmanager
    def preserve_runtime_buffers():
        saved = checkpoint_policy.tokens_per_expert.detach().clone()
        try:
            yield
        finally:
            checkpoint_policy.tokens_per_expert.copy_(saved)

    def checkpoint_contexts():
        return nullcontext(), preserve_runtime_buffers()

    torch.manual_seed(202)
    checkpointed = run_task_independent_calvin_joint_sequence_objective(
        checkpoint_policy,
        batches=(batch,),
        physical_sidecar=_sidecar(),
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=1.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=NativeLayerwisePosteriorState(previous_rows.clone()),
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((('object/a', 0), ('object/b', 1)),),
        graph=checkpoint_policy.model.qwenvl_with_expert.picf_native_graph,
        current_grid_cache=_CurrentCorrectionCache(),  # type: ignore[arg-type]
        omitted_static_view=omission,
        omitted_static_rematerialization=(
            OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT
        ),
        omitted_static_checkpoint_context_fn=checkpoint_contexts,
    )
    checkpoint_forward_buffers = checkpoint_policy.tokens_per_expert.detach().clone()
    checkpointed.objective.objective.total.backward()
    checkpoint_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in checkpoint_policy.named_parameters()
        if parameter.grad is not None
    }

    torch.testing.assert_close(
        checkpointed.objective.objective.total,
        baseline.objective.objective.total,
    )
    assert checkpoint_gradients.keys() == baseline_gradients.keys()
    for name in baseline_gradients:
        torch.testing.assert_close(checkpoint_gradients[name], baseline_gradients[name])
    torch.testing.assert_close(checkpoint_policy.tokens_per_expert, checkpoint_forward_buffers)
    assert checkpointed.omitted_static_rematerialization == (
        OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT
    )
    assert checkpoint_policy.call_order == ["prior", "policy", "policy", "policy"]


def test_v3_omitted_static_save_on_cpu_preserves_loss_and_gradient() -> None:
    def graph() -> LingBotNativeGraph:
        return LingBotNativeGraph(
            LingBotNativeGraphConfig(
                capacity=2,
                host_width=8,
                executed_action_dim=2,
                num_layers=3,
                architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
                prediction_address_width=0,
                predictive_target_widths=(("dino_video", 4),),
            )
        )

    batch = _batch(source_index=10, frame_index=1)
    omission = QwenWholeViewOmission(
        omitted_view_index=0,
        image_grid_thw=batch.model_inputs["image_grid_thw"],
        image_valid=batch.model_inputs["img_masks"],
        seed=73,
    )
    previous_rows = torch.linspace(-1, 1, steps=48).reshape(1, 3, 2, 8)
    objective_config = TaskIndependentEntityObjectiveConfig(
        action_weight=1.0,
        entity_weight=1.0,
        predictive_weight=1.0,
    )
    common = {
        "batches": (batch,),
        "physical_sidecar": _sidecar(),
        "objective_config": objective_config,
        "patch_size": 1,
        "merge_size": 2,
        "previous_state": NativeLayerwisePosteriorState(previous_rows.clone()),
        "previous_state_valid": torch.ones(1, dtype=torch.bool),
        "prior_row_bindings_by_batch": ((('object/a', 0), ('object/b', 1)),),
        "current_grid_cache": _CurrentCorrectionCache(),
        "omitted_static_view": omission,
    }

    torch.manual_seed(101)
    baseline_policy = _StatefulStochasticTrainingPolicy(graph()).train()
    torch.manual_seed(202)
    baseline = run_task_independent_calvin_joint_sequence_objective(
        baseline_policy,
        graph=baseline_policy.model.qwenvl_with_expert.picf_native_graph,
        **common,
    )
    baseline.objective.objective.total.backward()
    baseline_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in baseline_policy.named_parameters()
        if parameter.grad is not None
    }

    torch.manual_seed(101)
    offload_policy = _StatefulStochasticTrainingPolicy(graph()).train()
    torch.manual_seed(202)
    forward_context_entries = 0

    @contextmanager
    def offload_forward_context():
        nonlocal forward_context_entries
        forward_context_entries += 1
        yield

    offloaded = run_task_independent_calvin_joint_sequence_objective(
        offload_policy,
        graph=offload_policy.model.qwenvl_with_expert.picf_native_graph,
        omitted_static_rematerialization=OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU,
        omitted_static_forward_context_fn=offload_forward_context,
        **common,
    )
    offloaded.objective.objective.total.backward()
    offload_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in offload_policy.named_parameters()
        if parameter.grad is not None
    }

    torch.testing.assert_close(
        offloaded.objective.objective.total,
        baseline.objective.objective.total,
    )
    assert offload_gradients.keys() == baseline_gradients.keys()
    for name in baseline_gradients:
        torch.testing.assert_close(offload_gradients[name], baseline_gradients[name])
    assert offloaded.omitted_static_rematerialization == (
        OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU
    )
    assert offload_policy.call_order == ["prior", "policy", "policy"]
    assert forward_context_entries == 1


def test_v3_omitted_static_sequential_backward_matches_joint_gradient_and_state() -> None:
    graph_config = LingBotNativeGraphConfig(
        capacity=2,
        host_width=8,
        executed_action_dim=2,
        num_layers=3,
        architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
        prediction_address_width=0,
        predictive_target_widths=(("dino_video", 4),),
    )
    batch = _batch(source_index=10, frame_index=1)
    omission = QwenWholeViewOmission(
        omitted_view_index=0,
        image_grid_thw=batch.model_inputs["image_grid_thw"],
        image_valid=batch.model_inputs["img_masks"],
        seed=73,
    )
    previous_rows = torch.linspace(-1, 1, steps=48).reshape(1, 3, 2, 8)
    objective_config = TaskIndependentEntityObjectiveConfig(
        action_weight=1.0,
        entity_weight=1.0,
        predictive_weight=1.0,
    )

    def run_arguments() -> dict[str, object]:
        return {
            "batches": (batch,),
            "physical_sidecar": _sidecar(),
            "objective_config": objective_config,
            "patch_size": 1,
            "merge_size": 2,
            "previous_state": NativeLayerwisePosteriorState(previous_rows.clone()),
            "previous_state_valid": torch.ones(1, dtype=torch.bool),
            "prior_row_bindings_by_batch": ((('object/a', 0), ('object/b', 1)),),
            "current_grid_cache": _CurrentCorrectionCache(),
            "omitted_static_view": omission,
        }

    torch.manual_seed(101)
    baseline_policy = _StatefulStochasticTrainingPolicy(
        LingBotNativeGraph(graph_config)
    ).train()
    torch.manual_seed(202)
    baseline = run_task_independent_calvin_joint_sequence_objective(
        baseline_policy,
        graph=baseline_policy.model.qwenvl_with_expert.picf_native_graph,
        **run_arguments(),
    )
    baseline_forward_buffers = baseline_policy.tokens_per_expert.detach().clone()
    baseline.objective.objective.total.backward()
    baseline_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in baseline_policy.named_parameters()
        if parameter.grad is not None
    }

    torch.manual_seed(101)
    sequential_policy = _StatefulStochasticTrainingPolicy(
        LingBotNativeGraph(graph_config)
    ).train()
    torch.manual_seed(202)
    prior_rng = torch.get_rng_state()
    prior_buffers = sequential_policy.tokens_per_expert.detach().clone()
    sequential = run_task_independent_calvin_joint_sequence_objective(
        sequential_policy,
        graph=sequential_policy.model.qwenvl_with_expert.picf_native_graph,
        omitted_static_rematerialization=(
            OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD
        ),
        **run_arguments(),
    )
    omitted_rng = torch.get_rng_state()
    omitted_buffers = sequential_policy.tokens_per_expert.detach().clone()
    assert sequential.sequential_omitted_static is not None
    assert not sequential.objective.objective.total.requires_grad
    assert sequential.sequential_omitted_static.factual_backward_loss.requires_grad
    sequential.sequential_omitted_static.factual_backward_loss.backward()

    torch.set_rng_state(prior_rng)
    sequential_policy.tokens_per_expert.copy_(prior_buffers)
    replayed_prior, _prediction = run_native_v3_prior_chain(
        sequential_policy,
        graph=sequential_policy.model.qwenvl_with_expert.picf_native_graph,
        previous_memory=NativeLayerwisePosteriorState(previous_rows.clone()),
        previous_memory_valid=torch.ones(1, dtype=torch.bool),
        control_chunks=batch.effective_prior_control_chunks,
        filter_prediction=sequential.v3_filter_specs[0],
        require_attached_memory=False,
    )
    torch.set_rng_state(omitted_rng)
    sequential_policy.tokens_per_expert.copy_(omitted_buffers)
    omitted = run_task_independent_calvin_sequential_omitted_static_objective(
        sequential_policy,
        plan=sequential.sequential_omitted_static,
        prior_trace=replayed_prior,
    )
    omitted.backward_loss.backward()
    finalized = finalize_task_independent_calvin_sequential_omitted_result(
        sequential,
        omitted,
    )
    sequential_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in sequential_policy.named_parameters()
        if parameter.grad is not None
    }

    torch.testing.assert_close(
        finalized.objective.objective.total,
        baseline.objective.objective.total,
    )
    assert sequential_gradients.keys() == baseline_gradients.keys()
    for name in baseline_gradients:
        torch.testing.assert_close(sequential_gradients[name], baseline_gradients[name])
    torch.testing.assert_close(sequential_policy.tokens_per_expert, baseline_forward_buffers)
    assert sequential_policy.call_order == ["prior", "policy", "prior", "policy"]
    assert finalized.omitted_static_policy is not None
    assert finalized.omitted_static_branch is not None


def test_recurrent_sequence_reads_future_targets_only_after_all_host_forwards() -> None:
    policy = _predictive_policy()
    sidecar = _ForwardOrderedSidecar(policy)
    sidecar.frames = {
        10: _frame(("object/a", "object/b")),
        11: _frame(("object/a", "object/b")),
    }
    cache = _ForwardOrderedPredictiveCache(policy)
    batches = (
        _batch(source_index=10, frame_index=0),
        _batch(source_index=11, frame_index=1),
    )
    request = make_native_future_request(
        source=PredictionSource.PRIOR,
        batch_size=1,
        horizon=1,
        valid=torch.ones(1, dtype=torch.bool),
        device="cpu",
        dtype=torch.float32,
    )
    graph = policy.model.qwenvl_with_expert.picf_native_graph

    def predictive_rollout(state):  # type: ignore[no-untyped-def]
        return rollout_native_prior_prediction(
            LingBotNativePriorStepper(policy, graph),
            state,
            (batches[1].controls,),
            request=request,
            target_name="dino_video",
        )

    result = run_task_independent_calvin_sequence_objective(
        policy,
        batches=batches,
        physical_sidecar=sidecar,
        objective_config=TaskIndependentEntityObjectiveConfig(
            action_weight=0.0,
            predictive_weight=1.0,
        ),
        patch_size=1,
        merge_size=2,
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
        predictive_rollout_factory=predictive_rollout,
        predictive_cache=cache,
    )

    assert cache.calls == 1
    assert result.objective.objective.valid_counts["rollout/vision/binding"] == 2
    result.objective.objective.total.backward()
    assert graph.predictive_readout("dino_video").weight.grad is not None
    assert graph.predictive_readout("dino_video").weight.grad.abs().sum() > 0


def test_recurrent_sequence_rejects_prior_gauge_at_episode_reset() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )

    with pytest.raises(ValueError, match="reset must clear"):
        run_task_independent_calvin_sequence_objective(
            policy,
            batches=(_batch(),),
            physical_sidecar=_sidecar(),
            objective_config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
            patch_size=1,
            merge_size=2,
            previous_state=None,
            previous_state_valid=torch.zeros(1, dtype=torch.bool),
            prior_row_bindings_by_batch=((("object/a", 0),),),
        )


def test_real_age_lane_state_crosses_optimizer_steps_without_long_bptt() -> None:
    policy, coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    sidecar = _Sidecar(
        {
            10: _frame(("object/a", "object/b")),
            11: _frame(("object/a", "object/b")),
        }
    )
    config = TaskIndependentEntityObjectiveConfig(action_weight=0.0)

    first_attempt = coordinator.begin(optimizer_step=0, source_weight_version=0)
    first_batch = _batch()
    first_prepared = first_attempt.prepare(first_batch.routing)
    first = run_task_independent_calvin_frame_objective(
        policy,
        batch=first_batch,
        physical_sidecar=sidecar,
        objective_config=config,
        patch_size=1,
        merge_size=2,
        previous_state=first_prepared.previous_state,
        previous_state_valid=first_prepared.previous_state_valid,
    )
    first.objective.objective.total.backward()
    assert first.context.posterior_state is not None
    first_attempt.stage(
        first_prepared,
        first.context.posterior_state,
        row_bindings_by_batch=first_prepared.previous_row_bindings,
    )
    assert first_attempt.finish(lambda: 1)

    second_attempt = coordinator.begin(optimizer_step=1, source_weight_version=0)
    second_batch = _batch(source_index=11, frame_index=1, optimizer_step=1)
    second_prepared = second_attempt.prepare(second_batch.routing)
    assert second_prepared.previous_state_valid.tolist() == [True]
    assert not second_prepared.previous_state.rows.requires_grad
    second = run_task_independent_calvin_frame_objective(
        policy,
        batch=second_batch,
        physical_sidecar=sidecar,
        objective_config=config,
        patch_size=1,
        merge_size=2,
        previous_state=second_prepared.previous_state,
        previous_state_valid=second_prepared.previous_state_valid,
    )
    assert second.context.posterior_state is not None
    second_attempt.stage(
        second_prepared,
        second.context.posterior_state,
        row_bindings_by_batch=second_prepared.previous_row_bindings,
    )
    second_attempt.abort()
