from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from picf_next.data.calvin_target_request import NativeCALVINStructuralTargetRequest
from picf_next.lingbot_native import full_training
from picf_next.lingbot_native.calvin import CollatedNativeCALVINBatch
from picf_next.lingbot_native.calvin_objective import NativeStructuralLossConfig
from picf_next.lingbot_native.full_training import (
    NativeFilterPhaseBranch,
    make_native_current_correction_request,
    make_native_current_filter_request,
    make_native_current_grid_request,
    make_native_future_request,
    make_native_omitted_static_request,
    run_native_calvin_full_objective,
    run_native_calvin_relation_probe_objective,
    run_native_calvin_representation_objective,
)
from picf_next.lingbot_native.host import LingBotNativeGraph, LingBotNativeGraphConfig
from picf_next.lingbot_native.modalities import (
    NativeModalityBatch,
    NativeModalitySpec,
    NativeModalityStream,
)
from picf_next.lingbot_native.objective import NativeObjectiveConfig
from picf_next.lingbot_native.prediction import PredictionEvidence, PredictionSource
from picf_next.lingbot_native.predictive_objective import (
    TargetEncoderMode,
    make_native_predictive_target,
)
from picf_next.lingbot_native.source_mask import (
    QwenWholeViewOmission,
    sample_qwen_packed_patch_mask,
)
from picf_next.lingbot_native.state import NativePosteriorState
from picf_next.lingbot_native.temporal import NativePriorPredictiveRollout
from tests.lingbot_native.test_calvin_supervision import (
    _frame,
    _official_calvin_model_inputs,
    _request,
    _Sidecar,
)
from tests.lingbot_native.test_training_runtime import (
    _controls,
    _FakeOfficialTrainingPolicy,
    _routing,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _PredictiveCache:
    def __init__(self) -> None:
        self.contract = SimpleNamespace(route_id=0, minimum_visible_fraction=0.0)
        self.calls: list[tuple[int, ...]] = []

    def target_for(
        self,
        *,
        source_global_indices,
        source_rgb_sha256,
        track_identity_keys,
        request,
        device,
    ):
        del source_rgb_sha256
        self.calls.append(tuple(source_global_indices))
        tracks = max(len(value) for value in track_identity_keys)
        features = torch.arange(
            request.batch_size * tracks * request.query_count * 4,
            dtype=torch.float32,
            device=device,
        ).reshape(request.batch_size, tracks, request.query_count, 4)
        valid = torch.zeros(
            request.batch_size,
            tracks,
            request.query_count,
            dtype=torch.bool,
            device=device,
        )
        for batch_index, keys in enumerate(track_identity_keys):
            valid[batch_index, : len(keys)] = request.valid[batch_index]
        return make_native_predictive_target(
            modality="vision",
            features=features,
            valid=valid,
            importance=None,
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest=_sha(f"source-{source_global_indices}"),
            target_data_digest=_sha("target-data"),
            encoder_digest=_sha("encoder"),
            query_schema_digest=_sha("query"),
            validity_semantics="fixture independent visible support",
            track_identity_keys=track_identity_keys,
        )


class _CurrentGridCache:
    def __init__(self) -> None:
        self.contract = SimpleNamespace(route_id=0)
        self.calls = 0
        self.correction_calls: list[tuple[int, ...]] = []
        self.omitted_calls = 0

    def current_correction_summary_target_for(
        self,
        *,
        source_global_indices,
        source_static_rgb_sha256,
        track_identity_keys,
        request,
        physical_sidecar,
        minimum_visible_fraction,
        device,
    ):
        del source_static_rgb_sha256, physical_sidecar
        if minimum_visible_fraction != 0.0:
            raise AssertionError("fixture expected the frozen zero support threshold")
        self.correction_calls.append(tuple(source_global_indices))
        tracks = max(len(value) for value in track_identity_keys)
        features = torch.arange(
            request.batch_size * tracks * request.query_count * 4,
            dtype=torch.float32,
            device=device,
        ).reshape(request.batch_size, tracks, request.query_count, 4)
        valid = torch.zeros(
            request.batch_size,
            tracks,
            request.query_count,
            dtype=torch.bool,
            device=device,
        )
        for batch_index, keys in enumerate(track_identity_keys):
            valid[batch_index, : len(keys)] = request.valid[batch_index]
        return make_native_predictive_target(
            modality="vision",
            features=features,
            valid=valid,
            importance=None,
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest=_sha(f"correction-source-{source_global_indices}"),
            target_data_digest=_sha("correction-target-data"),
            encoder_digest=_sha("correction-encoder"),
            query_schema_digest=_sha("correction-query"),
            validity_semantics="fixture current correction support",
            track_identity_keys=track_identity_keys,
        )

    def target_for(
        self,
        *,
        source_global_indices,
        source_rgb_sha256,
        track_identity_keys,
        selected_token_indices,
        merged_grid_hw,
        request,
        physical_sidecar,
        device,
    ):
        del source_rgb_sha256, selected_token_indices, merged_grid_hw, physical_sidecar
        self.calls += 1
        tracks = max(len(value) for value in track_identity_keys)
        features = torch.arange(
            request.batch_size * tracks * request.query_count * 4,
            dtype=torch.float32,
            device=device,
        ).reshape(request.batch_size, tracks, request.query_count, 4)
        valid = torch.zeros(
            request.batch_size,
            tracks,
            request.query_count,
            dtype=torch.bool,
            device=device,
        )
        for batch_index, keys in enumerate(track_identity_keys):
            valid[batch_index, : len(keys)] = request.valid[batch_index]
        return make_native_predictive_target(
            modality="vision",
            features=features,
            valid=valid,
            importance=None,
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest=_sha(f"current-source-{source_global_indices}"),
            target_data_digest=_sha("current-target-data"),
            encoder_digest=_sha("current-encoder"),
            query_schema_digest=_sha("current-query"),
            validity_semantics="fixture independent current visible support",
            track_identity_keys=track_identity_keys,
        )

    def omitted_static_summary_target_for(
        self,
        *,
        source_global_indices,
        source_static_rgb_sha256,
        source_gripper_rgb_sha256,
        track_identity_keys,
        request,
        omission,
        physical_sidecar,
        device,
    ):
        del (
            source_static_rgb_sha256,
            source_gripper_rgb_sha256,
            omission,
            physical_sidecar,
        )
        self.omitted_calls += 1
        tracks = max(len(value) for value in track_identity_keys)
        features = torch.arange(
            request.batch_size * tracks * request.query_count * 4,
            dtype=torch.float32,
            device=device,
        ).reshape(request.batch_size, tracks, request.query_count, 4)
        valid = torch.zeros(
            request.batch_size,
            tracks,
            request.query_count,
            dtype=torch.bool,
            device=device,
        )
        for batch_index, keys in enumerate(track_identity_keys):
            valid[batch_index, : len(keys)] = request.valid[batch_index]
        return make_native_predictive_target(
            modality="vision",
            features=features,
            valid=valid,
            importance=None,
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest=_sha(f"omitted-source-{source_global_indices}"),
            target_data_digest=_sha("omitted-target-data"),
            encoder_digest=_sha("omitted-encoder"),
            query_schema_digest=_sha("omitted-query"),
            validity_semantics="fixture independent cross-view visible support",
            track_identity_keys=track_identity_keys,
        )


def _batch(
    frame_index: int,
    source_index: int,
    *,
    modalities: NativeModalityBatch | None = None,
) -> CollatedNativeCALVINBatch:
    routing = _routing(
        0,
        optimizer_step=0,
        frame_index=frame_index,
        episode_key="episode/3",
    )
    base = _request(source_index)
    request = NativeCALVINStructuralTargetRequest(
        sample_key=routing.sample_keys[0],
        episode_key=routing.episode_keys[0],
        task_key=base.task_key,
        segment_index=base.segment_index,
        source_global_index=base.source_global_index,
        source_sensor_sha256=base.source_sensor_sha256,
    )
    return CollatedNativeCALVINBatch(
        model_inputs=_official_calvin_model_inputs(),
        controls=_controls(1, reset=frame_index == 0),
        routing=routing,
        source_digest=_sha(f"batch-{frame_index}"),
        structural_target_requests=(request,),
        modalities=modalities,
    )


def _components(
    *,
    modality_specs: tuple[NativeModalitySpec, ...] = (),
    relation_supervision_layers: tuple[int, ...] = (),
):
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            predictive_target_widths=(("dino_video", 4),),
            modality_specs=modality_specs,
            relation_supervision_layers=relation_supervision_layers,
        )
    )
    return _FakeOfficialTrainingPolicy(graph).train(), graph


def test_production_full_objective_exposes_no_diagnostic_gradient_switch() -> None:
    production_parameters = inspect.signature(run_native_calvin_full_objective).parameters
    relation_parameters = inspect.signature(run_native_calvin_relation_probe_objective).parameters

    assert "optimize_official_policy_loss" not in production_parameters
    assert "predictive_cache" not in relation_parameters
    assert "current_grid_cache" not in relation_parameters
    assert "overshoot_factory" not in relation_parameters


def test_current_correction_request_is_prior_only_and_zero_horizon() -> None:
    request = make_native_current_correction_request(
        batch_size=2,
        valid=torch.tensor([True, False]),
        device="cpu",
        dtype=torch.float32,
        route_id=3,
        address_width=2,
    )

    assert request.source is PredictionSource.PRIOR
    assert request.evidence is PredictionEvidence.CURRENT_CORRECTION
    assert request.route_ids.tolist() == [[3], [3]]
    assert request.horizons.tolist() == [[0], [0]]
    assert request.valid.tolist() == [[True], [False]]
    assert not request.addresses.any()


@pytest.mark.parametrize(
    ("source", "evidence", "phase"),
    (
        (PredictionSource.PRIOR, PredictionEvidence.CURRENT_PRIOR, 0),
        (PredictionSource.POSTERIOR, PredictionEvidence.CURRENT_POSTERIOR, 1),
    ),
)
def test_current_filter_request_and_branch_make_latent_phase_explicit(
    source: PredictionSource,
    evidence: PredictionEvidence,
    phase: int,
) -> None:
    request = make_native_current_filter_request(
        source=source,
        batch_size=1,
        valid=torch.ones(1, dtype=torch.bool),
        device="cpu",
        dtype=torch.float32,
        address_width=2,
    )
    prediction = torch.zeros(1, 2, 1, 4, requires_grad=True)
    branch = NativeFilterPhaseBranch(
        batch=_batch(0, 10),
        request=request,
        prediction=prediction,
        identity_source_phase=phase,
    )

    assert branch.request.evidence is evidence
    assert branch.request.source is source
    assert not branch.request.addresses.any()


@pytest.mark.parametrize(
    ("frame_index", "previous_valid", "message"),
    (
        (0, True, "reset sample"),
        (1, False, "continuation sample"),
    ),
)
def test_direct_objective_source_contract_rejects_inconsistent_recurrent_validity(
    frame_index: int,
    previous_valid: bool,
    message: str,
) -> None:
    batch = _batch(frame_index, 10)
    previous_state = NativePosteriorState(torch.zeros(1, 2, 8))

    with pytest.raises(ValueError, match=message):
        full_training._correction_valid_by_time(
            batches=(batch,),
            previous_state=previous_state,
            previous_state_valid=torch.tensor([previous_valid]),
        )


def test_current_correction_finds_a_reset_in_the_complete_prior_control_chain() -> None:
    batch = _batch(0, 10)
    reset = batch.controls
    final_controls = _controls(1, reset=False)
    batch = replace(
        batch,
        controls=final_controls,
        prior_control_chunks=(reset, final_controls),
    )

    assert not final_controls.reset.any()
    assert batch.prior_control_reset.tolist() == [True]
    valid = full_training._correction_valid_by_time(
        batches=(batch,),
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
    )
    assert len(valid) == 1
    assert valid[0].tolist() == [False]


def test_full_objective_uses_runtime_dtype_instead_of_fp32_master_dtype(
    monkeypatch,
) -> None:
    class CapturedRuntimeDtype(RuntimeError):
        pass

    policy, graph = _components()
    batch = _batch(1, 10)
    model_inputs = {
        name: value.to(torch.bfloat16) if value.is_floating_point() else value
        for name, value in batch.model_inputs.items()
    }
    controls = batch.controls
    batch = replace(
        batch,
        model_inputs=model_inputs,
        controls=replace(
            controls,
            values=controls.values.to(torch.bfloat16),
            delta_time=controls.delta_time.to(torch.bfloat16),
        ),
    )

    def capture_forward(_policy, *, model_inputs, context):
        del model_inputs
        assert graph.object_queries.dtype == torch.float32
        assert context.prediction_request is not None
        assert context.prediction_request.addresses.dtype == torch.bfloat16
        raise CapturedRuntimeDtype

    monkeypatch.setattr(full_training, "run_native_policy_training_forward", capture_forward)
    with pytest.raises(CapturedRuntimeDtype):
        run_native_calvin_full_objective(
            policy,
            graph=graph,
            batches=(batch,),
            previous_state=NativePosteriorState(
                torch.zeros(1, 2, graph.config.host_width, dtype=torch.bfloat16)
            ),
            previous_state_valid=torch.ones(1, dtype=torch.bool),
            prior_row_bindings_by_batch=((),),
            predictive_cache=_PredictiveCache(),
            current_grid_cache=_CurrentGridCache(),  # type: ignore[arg-type]
            physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
            capacity=2,
            task_identity_resolver=lambda _task: ("object/a",),
            patch_size=1,
            merge_size=2,
            objective_config=NativeObjectiveConfig(
                predictive_weight=1.0,
                structural_weight=1.0,
            ),
            structural_config=NativeStructuralLossConfig(
                support_weight=1.0,
                existence_weight=1.0,
                task_weight=1.0,
                dense_task_weight=1.0,
                ownership_weight=1.0,
            ),
        )


def _run(
    *,
    local_steps: int,
    modality_specs: tuple[NativeModalitySpec, ...] = (),
    modalities: NativeModalityBatch | None = None,
    relation_supervision_layers: tuple[int, ...] = (),
    prior_bindings: tuple[tuple[str, int], ...] = (
        ("object/a", 0),
        ("object/b", 1),
    ),
):
    policy, graph = _components(
        modality_specs=modality_specs,
        relation_supervision_layers=relation_supervision_layers,
    )
    future_cache = _PredictiveCache()
    current_cache = _CurrentGridCache()
    batches = tuple(
        _batch(index + 1, 10 + index, modalities=modalities) for index in range(local_steps)
    )
    previous_state = NativePosteriorState(torch.randn(1, 2, graph.config.host_width))
    result = run_native_calvin_full_objective(
        policy,
        graph=graph,
        batches=batches,
        previous_state=previous_state,
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=(prior_bindings,),
        predictive_cache=future_cache,
        current_grid_cache=current_cache,  # type: ignore[arg-type]
        physical_sidecar=_Sidecar(
            {10 + index: _frame(("object/a", "object/b")) for index in range(local_steps)}
        ),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    )
    return policy, graph, future_cache, current_cache, result


def _run_representation(*, local_steps: int):
    policy, graph = _components()
    future_cache = _PredictiveCache()
    current_cache = _CurrentGridCache()
    batches = tuple(_batch(index + 1, 10 + index) for index in range(local_steps))
    previous_state = NativePosteriorState(torch.randn(1, 2, graph.config.host_width))
    result = run_native_calvin_representation_objective(
        policy,
        graph=graph,
        batches=batches,
        previous_state=previous_state,
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((("object/a", 0), ("object/b", 1)),),
        predictive_cache=future_cache,
        current_grid_cache=current_cache,  # type: ignore[arg-type]
        physical_sidecar=_Sidecar(
            {10 + index: _frame(("object/a", "object/b")) for index in range(local_steps)}
        ),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            action_weight=0.0,
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    )
    return policy, graph, future_cache, current_cache, result


def test_relation_probe_objective_has_no_predictive_queries_or_action_gradient() -> None:
    policy, graph = _components()
    selected = {
        id(graph.relation_readout.projection.weight),
        id(graph.relation_readout.no_object),
        id(graph.relation_readout.temperature_parameter),
    }
    for parameter in policy.parameters():
        parameter.requires_grad_(id(parameter) in selected)
    batches = (_batch(0, 10), _batch(1, 11))

    result = run_native_calvin_relation_probe_objective(
        policy,
        graph=graph,
        batches=batches,
        previous_state=None,
        previous_state_valid=None,
        prior_row_bindings_by_batch=((),),
        physical_sidecar=_Sidecar(
            {
                10: _frame(("object/a", "object/b")),
                11: _frame(("object/a", "object/b")),
            }
        ),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=1.0,
        ),
    )

    assert result.objective.predictive_terms == ()
    assert result.primary.context.prediction_hidden is None
    assert result.primary.context.prediction_outputs == {}
    assert not result.primary.official_total_loss.requires_grad
    ownership = result.objective.objective.normalized_terms["set/ownership"]
    assert ownership.requires_grad
    ownership.backward()
    gradient = graph.relation_readout.projection.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_relation_probe_objective_rejects_predictive_family_weight() -> None:
    policy, graph = _components()
    batch = _batch(0, 10)

    with pytest.raises(ValueError, match="zero predictive family weight"):
        run_native_calvin_relation_probe_objective(
            policy,
            graph=graph,
            batches=(batch,),
            previous_state=None,
            previous_state_valid=None,
            prior_row_bindings_by_batch=((),),
            physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
            capacity=2,
            task_identity_resolver=lambda _task: ("object/a",),
            patch_size=1,
            merge_size=2,
            objective_config=NativeObjectiveConfig(
                predictive_weight=1.0,
                structural_weight=1.0,
            ),
            structural_config=NativeStructuralLossConfig(
                support_weight=0.0,
                existence_weight=0.0,
                task_weight=0.0,
                dense_task_weight=0.0,
                ownership_weight=1.0,
            ),
        )


def test_full_objective_primary_has_action_correction_and_structural_gradients() -> None:
    _policy, graph, future_cache, current_cache, result = _run(local_steps=1)

    assert future_cache.calls == []
    assert current_cache.correction_calls == [(10,)]
    assert result.objective.objective.valid_counts["action"] == 1
    assert result.objective.objective.valid_counts["correction/vision/binding"] == 2
    result.objective.objective.total.backward()
    assert graph.object_queries.grad is None or not graph.object_queries.grad.count_nonzero()
    assert graph.role_embeddings.grad is not None
    assert graph.role_embeddings.grad.abs().sum() > 0
    assert graph.relation_readout.projection.weight.grad is not None
    assert graph.relation_readout.projection.weight.grad.abs().sum() > 0
    readout = graph.predictive_readout("dino_video")
    assert readout.weight.grad is not None
    assert readout.weight.grad.abs().sum() > 0


def test_fresh_birth_cannot_retroactively_supervise_prior_correction() -> None:
    _policy, _graph, _future_cache, _current_cache, first = _run(
        local_steps=1,
        prior_bindings=(),
    )
    assert [branch.identity_source_phase for branch in first.correction_branches] == [0]
    assert first.objective.objective.valid_counts["correction/vision/binding"] == 0

    _policy, _graph, _future_cache, _current_cache, second = _run(
        local_steps=2,
        prior_bindings=(),
    )
    assert [branch.identity_source_phase for branch in second.correction_branches] == [0, 2]
    assert second.objective.objective.valid_counts["correction/vision/binding"] == 2


def test_representation_objective_uses_no_action_forward_and_trains_both_families() -> None:
    policy, graph, future_cache, current_cache, result = _run_representation(local_steps=1)

    assert policy.forward_grad_enabled == []
    assert policy.observation_forward_grad_enabled == [True]
    assert future_cache.calls == []
    assert current_cache.correction_calls == [(10,)]
    assert "action" not in result.objective.objective.normalized_terms
    torch.testing.assert_close(
        result.objective.objective.family_terms["action"],
        torch.tensor(0.0),
    )
    assert result.objective.objective.valid_counts["correction/vision/binding"] == 2
    assert result.objective.objective.valid_counts["set/ownership"] == 2
    result.objective.objective.total.backward()
    assert graph.object_queries.grad is None or not graph.object_queries.grad.count_nonzero()
    assert graph.role_embeddings.grad is not None
    assert graph.role_embeddings.grad.abs().sum() > 0
    assert graph.relation_readout.projection.weight.grad is not None
    assert graph.relation_readout.projection.weight.grad.abs().sum() > 0
    readout = graph.predictive_readout("dino_video")
    assert readout.weight.grad is not None
    assert readout.weight.grad.abs().sum() > 0


def test_representation_objective_keeps_local_window_in_observation_host() -> None:
    policy, _graph, _future_cache, current_cache, result = _run_representation(local_steps=2)

    assert policy.forward_grad_enabled == []
    assert policy.observation_forward_grad_enabled == [True, True]
    assert current_cache.correction_calls == [(10,), (11,)]
    assert len(result.correction_branches) == 2
    assert result.objective.objective.valid_counts["correction/vision/binding"] == 4
    assert result.objective.objective.valid_counts["set/existence"] == 4
    assert result.primary.posterior_state is not None


def test_representation_objective_rejects_an_active_action_family() -> None:
    policy, graph = _components()
    with pytest.raises(ValueError, match="zero action-family weight"):
        run_native_calvin_representation_objective(
            policy,
            graph=graph,
            batches=(_batch(1, 10),),
            previous_state=NativePosteriorState(torch.randn(1, 2, graph.config.host_width)),
            previous_state_valid=torch.ones(1, dtype=torch.bool),
            prior_row_bindings_by_batch=((),),
            predictive_cache=_PredictiveCache(),
            current_grid_cache=_CurrentGridCache(),  # type: ignore[arg-type]
            physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
            capacity=2,
            task_identity_resolver=lambda _task: ("object/a",),
            patch_size=1,
            merge_size=2,
            objective_config=NativeObjectiveConfig(
                predictive_weight=1.0,
                structural_weight=1.0,
            ),
            structural_config=NativeStructuralLossConfig(
                support_weight=0.0,
                existence_weight=1.0,
                task_weight=1.0,
                dense_task_weight=1.0,
                ownership_weight=1.0,
            ),
        )


def test_current_teacher_target_changes_only_post_forward_loss() -> None:
    class ShiftedTargetCache(_CurrentGridCache):
        def current_correction_summary_target_for(self, **kwargs):
            target = super().current_correction_summary_target_for(**kwargs)
            return replace(
                target,
                features=target.features.flip(dims=(-1,)),
                target_data_digest=_sha("shifted-current-target-data"),
            )

    policy, graph = _components()
    batch = _batch(1, 10)
    previous_state = NativePosteriorState(torch.arange(16, dtype=torch.float32).reshape(1, 2, 8))
    common = {
        "policy": policy,
        "graph": graph,
        "batches": (batch,),
        "previous_state": previous_state,
        "previous_state_valid": torch.ones(1, dtype=torch.bool),
        "prior_row_bindings_by_batch": ((("object/a", 0), ("object/b", 1)),),
        "predictive_cache": _PredictiveCache(),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
        "objective_config": NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        "structural_config": NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    }
    factual = run_native_calvin_full_objective(
        current_grid_cache=_CurrentGridCache(),  # type: ignore[arg-type]
        **common,
    )
    shifted = run_native_calvin_full_objective(
        current_grid_cache=ShiftedTargetCache(),  # type: ignore[arg-type]
        **common,
    )

    torch.testing.assert_close(
        factual.primary.context.posterior_state.rows,
        shifted.primary.context.posterior_state.rows,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        factual.correction_branches[0].prediction,
        shifted.correction_branches[0].prediction,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        factual.primary.official_action_loss,
        shifted.primary.official_action_loss,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        factual.objective.objective.normalized_terms["correction/vision/binding"],
        shifted.objective.objective.normalized_terms["correction/vision/binding"],
    )


def test_future_teacher_target_changes_only_post_forward_loss() -> None:
    class ShiftedTargetCache(_PredictiveCache):
        def target_for(self, **kwargs):
            target = super().target_for(**kwargs)
            return replace(
                target,
                features=target.features.flip(dims=(-1,)),
                target_data_digest=_sha("shifted-future-target-data"),
            )

    policy, graph = _components()
    batches = (_batch(1, 10),)
    previous_state = NativePosteriorState(torch.arange(16, dtype=torch.float32).reshape(1, 2, 8))

    def overshoot_factory(state: NativePosteriorState) -> NativePriorPredictiveRollout:
        request = make_native_future_request(
            source=PredictionSource.PRIOR,
            batch_size=state.batch_size,
            horizon=2,
            valid=torch.ones(state.batch_size, dtype=torch.bool),
            device=state.rows.device,
            dtype=state.rows.dtype,
        )
        return NativePriorPredictiveRollout(
            horizon=2,
            state=NativePosteriorState(state.rows * 1.0),
            target_name="dino_video",
            prediction=state.rows[:, :, None, :4] * 1.0,
            request=request,
        )

    common = {
        "policy": policy,
        "graph": graph,
        "batches": batches,
        "previous_state": previous_state,
        "previous_state_valid": torch.ones(1, dtype=torch.bool),
        "prior_row_bindings_by_batch": ((),),
        "current_grid_cache": _CurrentGridCache(),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
        "objective_config": NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        "structural_config": NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
        "overshoot_factory": overshoot_factory,
    }
    factual = run_native_calvin_full_objective(
        predictive_cache=_PredictiveCache(),
        **common,
    )
    shifted = run_native_calvin_full_objective(
        predictive_cache=ShiftedTargetCache(),
        **common,
    )

    assert len(factual.future_branches) == len(shifted.future_branches) == 1
    torch.testing.assert_close(
        factual.primary.context.posterior_state.rows,
        shifted.primary.context.posterior_state.rows,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        factual.primary.official_action_loss,
        shifted.primary.official_action_loss,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        factual.future_branches[0].prediction,
        shifted.future_branches[0].prediction,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        factual.objective.objective.normalized_terms["rollout/vision/binding"],
        shifted.objective.objective.normalized_terms["rollout/vision/binding"],
    )


def test_behavior_conditioned_future_uses_isolated_shared_weight_forward() -> None:
    policy, graph = _components()
    batch = _batch(1, 10)
    previous_state = NativePosteriorState(torch.arange(16, dtype=torch.float32).reshape(1, 2, 8))
    factual_controls = _controls(1, reset=False)
    counterfactual_controls = replace(
        factual_controls,
        values=-factual_controls.values,
    )
    common = {
        "policy": policy,
        "graph": graph,
        "batches": (batch,),
        "previous_state": previous_state,
        "previous_state_valid": torch.ones(1, dtype=torch.bool),
        "prior_row_bindings_by_batch": ((),),
        "predictive_cache": _PredictiveCache(),
        "current_grid_cache": _CurrentGridCache(),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
        "objective_config": NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        "structural_config": NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
        "behavior_prediction_horizon": 1,
    }

    factual = run_native_calvin_full_objective(
        behavior_prediction_controls=factual_controls,
        **common,
    )
    counterfactual = run_native_calvin_full_objective(
        behavior_prediction_controls=counterfactual_controls,
        **common,
    )

    assert factual.correction_branches == ()
    assert len(factual.future_branches) == 1
    assert factual.future_branches[0].request.source is PredictionSource.PRIOR
    assert factual.future_branches[0].request.evidence is PredictionEvidence.FUTURE
    assert factual.future_branches[0].request.horizons.tolist() == [[1]]
    assert factual.primary.context.prediction_request is None
    current_state = factual.primary.context.posterior_state
    assert current_state is not None
    future_term = factual.objective.objective.normalized_terms["rollout/vision/binding"]
    state_gradient = torch.autograd.grad(
        future_term,
        current_state.rows,
        retain_graph=True,
    )[0]
    assert torch.isfinite(state_gradient).all()
    assert state_gradient.abs().sum() > 0
    torch.testing.assert_close(
        factual.primary.context.posterior_state.rows,
        counterfactual.primary.context.posterior_state.rows,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        factual.primary.official_action_loss,
        counterfactual.primary.official_action_loss,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        factual.future_branches[0].prediction,
        counterfactual.future_branches[0].prediction,
    )


def test_behavior_conditioned_future_rejects_extra_controls_and_recursive_overshoot() -> None:
    policy, graph = _components()
    batch = _batch(1, 10)
    controls = _controls(1, reset=False)
    extra_controls = replace(
        controls,
        values=torch.cat((controls.values, controls.values), dim=1),
        field_valid=torch.cat((controls.field_valid, controls.field_valid), dim=1),
        token_valid=torch.cat((controls.token_valid, controls.token_valid), dim=1),
        delta_time=torch.cat((controls.delta_time, controls.delta_time), dim=1),
        reset=torch.cat((controls.reset, controls.reset), dim=1),
        acknowledged=torch.cat((controls.acknowledged, controls.acknowledged), dim=1),
    )
    common = {
        "policy": policy,
        "graph": graph,
        "batches": (batch,),
        "previous_state": NativePosteriorState(torch.zeros(1, 2, graph.config.host_width)),
        "previous_state_valid": torch.ones(1, dtype=torch.bool),
        "prior_row_bindings_by_batch": ((),),
        "predictive_cache": _PredictiveCache(),
        "current_grid_cache": _CurrentGridCache(),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
        "objective_config": NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        "structural_config": NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
        "behavior_prediction_controls": extra_controls,
        "behavior_prediction_horizon": 1,
    }
    with pytest.raises(ValueError, match="exactly equal"):
        run_native_calvin_full_objective(**common)
    with pytest.raises(ValueError, match="exclusive"):
        run_native_calvin_full_objective(
            **common,
            overshoot_factory=lambda _state: None,
        )


def test_owner_raster_changes_only_post_forward_supervision() -> None:
    policy, graph = _components()
    batch = _batch(1, 10)
    previous_state = NativePosteriorState(torch.arange(16, dtype=torch.float32).reshape(1, 2, 8))
    common = {
        "policy": policy,
        "graph": graph,
        "batches": (batch,),
        "previous_state": previous_state,
        "previous_state_valid": torch.ones(1, dtype=torch.bool),
        "prior_row_bindings_by_batch": ((),),
        "predictive_cache": _PredictiveCache(),
        "current_grid_cache": _CurrentGridCache(),
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
        "objective_config": NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        "structural_config": NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    }
    owner_a = run_native_calvin_full_objective(
        physical_sidecar=_Sidecar(
            {10: _frame(("object/a", "object/b"), static_owner=1, gripper_owner=1)}
        ),
        **common,
    )
    owner_b = run_native_calvin_full_objective(
        physical_sidecar=_Sidecar(
            {10: _frame(("object/a", "object/b"), static_owner=2, gripper_owner=2)}
        ),
        **common,
    )

    torch.testing.assert_close(
        owner_a.primary.context.prior_state.rows,
        owner_b.primary.context.prior_state.rows,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        owner_a.primary.context.posterior_state.rows,
        owner_b.primary.context.posterior_state.rows,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        owner_a.correction_branches[0].prediction,
        owner_b.correction_branches[0].prediction,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        owner_a.primary.official_action_loss,
        owner_b.primary.official_action_loss,
        rtol=0,
        atol=0,
    )


def test_row_binding_labels_change_only_post_forward_supervision() -> None:
    policy, graph = _components()
    batch = _batch(1, 10)
    previous_state = NativePosteriorState(torch.arange(16, dtype=torch.float32).reshape(1, 2, 8))
    common = {
        "policy": policy,
        "graph": graph,
        "batches": (batch,),
        "previous_state": previous_state,
        "previous_state_valid": torch.ones(1, dtype=torch.bool),
        "predictive_cache": _PredictiveCache(),
        "current_grid_cache": _CurrentGridCache(),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
        "objective_config": NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        "structural_config": NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    }
    first = run_native_calvin_full_objective(
        prior_row_bindings_by_batch=((("object/a", 0), ("object/b", 1)),),
        **common,
    )
    swapped = run_native_calvin_full_objective(
        prior_row_bindings_by_batch=((("object/a", 1), ("object/b", 0)),),
        **common,
    )

    torch.testing.assert_close(
        first.primary.context.posterior_state.rows,
        swapped.primary.context.posterior_state.rows,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.primary.official_action_loss,
        swapped.primary.official_action_loss,
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        first.objective.assignment.row_to_track,
        torch.tensor([[0, 1]]),
    )
    assert torch.equal(
        swapped.objective.assignment.row_to_track,
        torch.tensor([[1, 0]]),
    )


def test_full_objective_local_window_keeps_only_primary_state_committable() -> None:
    _policy, _graph, future_cache, current_cache, result = _run(local_steps=2)

    assert future_cache.calls == []
    assert current_cache.correction_calls == [(10,), (11,)]
    assert len(result.correction_branches) == 2
    assert result.future_branches == ()
    assert result.primary.context.posterior_state is not None
    assert result.objective.objective.valid_counts["correction/vision/binding"] == 4
    assert result.objective.objective.valid_counts["set/existence"] == 4
    torch.testing.assert_close(
        result.objective.objective.normalized_terms["action"],
        result.primary.official_total_loss,
    )


def test_full_objective_routes_optional_modalities_through_shared_lingbot_only() -> None:
    modalities = NativeModalityBatch(
        (
            NativeModalityStream(
                "touch",
                torch.randn(1, 2, 3),
                torch.tensor([[True, False]]),
            ),
        )
    )
    _policy, graph, _future_cache, _current_cache, result = _run(
        local_steps=1,
        modality_specs=(NativeModalitySpec("touch", 3, 2),),
        modalities=modalities,
    )
    relation = result.primary.context.relation_output
    assert relation is not None
    assert relation.sensor_valid.sum().item() == relation.structural_valid.sum().item() + 1
    result.objective.objective.total.backward()
    projection = graph.modality_projections["touch"]
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0
    assert result.primary.official_moe_regularizer.detach().gt(0)


def test_full_objective_trains_shared_intermediate_relations_across_local_time() -> None:
    _policy, graph, _future_cache, _current_cache, result = _run(
        local_steps=2,
        relation_supervision_layers=(0, 1),
    )
    assert tuple(result.primary.context.intermediate_relation_outputs) == (0, 1)
    terms = {term.name: term for term in result.objective.structural_terms}
    ownership_names = (
        "set/ownership",
        "set/ownership_q1",
        "set/ownership_q2",
    )
    assert all(terms[name].weight == pytest.approx(1 / 3) for name in ownership_names)
    assert sum(terms[name].weight for name in ownership_names) == pytest.approx(1.0)
    assert all(
        result.objective.objective.normalized_terms[name].requires_grad for name in ownership_names
    )
    result.objective.objective.total.backward()
    assert graph.relation_readout.projection.weight.grad is not None
    assert graph.relation_readout.projection.weight.grad.abs().sum() > 0


def test_full_objective_reset_has_no_correction_target_denominator() -> None:
    policy, graph = _components()
    future_cache = _PredictiveCache()
    current_cache = _CurrentGridCache()
    batch = _batch(0, 10)
    result = run_native_calvin_full_objective(
        policy,
        graph=graph,
        batches=(batch,),
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
        predictive_cache=future_cache,
        current_grid_cache=current_cache,  # type: ignore[arg-type]
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    )

    assert future_cache.calls == []
    assert current_cache.correction_calls == [(10,)]
    assert result.objective.objective.valid_counts["correction/vision/binding"] == 0


def test_full_objective_rejects_prior_identity_without_a_valid_recurrent_source() -> None:
    policy, graph = _components()
    batch = _batch(0, 10)
    with pytest.raises(ValueError, match="valid non-reset recurrent source"):
        run_native_calvin_full_objective(
            policy,
            graph=graph,
            batches=(batch,),
            previous_state=None,
            previous_state_valid=torch.zeros(1, dtype=torch.bool),
            prior_row_bindings_by_batch=((("object/a", 0),),),
            predictive_cache=_PredictiveCache(),
            current_grid_cache=_CurrentGridCache(),  # type: ignore[arg-type]
            physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
            capacity=2,
            task_identity_resolver=lambda _task: ("object/a",),
            patch_size=1,
            merge_size=2,
            objective_config=NativeObjectiveConfig(
                predictive_weight=1.0,
                structural_weight=1.0,
            ),
            structural_config=NativeStructuralLossConfig(
                support_weight=0.0,
                existence_weight=1.0,
                task_weight=1.0,
                dense_task_weight=1.0,
                ownership_weight=1.0,
            ),
        )


def test_full_objective_builds_overshoot_only_from_primary_posterior() -> None:
    policy, graph = _components()
    future_cache = _PredictiveCache()
    current_cache = _CurrentGridCache()
    batch = _batch(1, 10)
    previous_state = NativePosteriorState(torch.randn(1, 2, graph.config.host_width))
    observed: list[NativePosteriorState] = []

    def overshoot_factory(state: NativePosteriorState) -> NativePriorPredictiveRollout:
        observed.append(state)
        request = make_native_future_request(
            source=PredictionSource.PRIOR,
            batch_size=state.batch_size,
            horizon=2,
            valid=torch.ones(state.batch_size, dtype=torch.bool),
            device=state.rows.device,
            dtype=state.rows.dtype,
        )
        return NativePriorPredictiveRollout(
            horizon=2,
            state=NativePosteriorState(state.rows * 1.0),
            target_name="dino_video",
            prediction=state.rows[:, :, None, :4] * 1.0,
            request=request,
        )

    result = run_native_calvin_full_objective(
        policy,
        graph=graph,
        batches=(batch,),
        previous_state=previous_state,
        previous_state_valid=torch.ones(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
        predictive_cache=future_cache,
        current_grid_cache=current_cache,  # type: ignore[arg-type]
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
        overshoot_factory=overshoot_factory,
    )

    assert observed == [result.primary.context.posterior_state]
    assert result.overshoot is not None
    assert len(result.correction_branches) == 1
    assert len(result.future_branches) == 1
    assert current_cache.correction_calls == [(10,)]
    assert future_cache.calls == [(10,)]


def test_full_objective_never_calls_a_readout_after_the_root_forward() -> None:
    source = inspect.getsource(full_training)
    assert ".predictive_readout(" not in source


@pytest.mark.parametrize(("with_prior_binding", "expected_valid"), [(False, 0), (True, 2)])
def test_full_objective_current_grid_branch_is_weight_shared_and_uncommittable(
    with_prior_binding: bool,
    expected_valid: int,
) -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            prediction_address_width=2,
            predictive_target_widths=(("dino_video", 4),),
        )
    )
    policy = _FakeOfficialTrainingPolicy(graph).train()
    future_cache = _PredictiveCache()
    current_cache = _CurrentGridCache()
    batch = _batch(int(with_prior_binding), 10)
    source_mask = sample_qwen_packed_patch_mask(
        images=batch.model_inputs["images"],
        image_valid=batch.model_inputs["img_masks"],
        image_grid_thw=batch.model_inputs["image_grid_thw"],
        spatial_merge_size=2,
        probability=1.0,
        seed=7,
        eligible_view_indices=(0,),
    )
    request = make_native_current_grid_request(
        source_mask=source_mask,
        route_id=0,
        dtype=graph.object_queries.dtype,
    )
    assert request.addresses.shape == (1, 1, 2)

    result = run_native_calvin_full_objective(
        policy,
        graph=graph,
        batches=(batch,),
        previous_state=(
            NativePosteriorState(torch.randn(1, 2, graph.config.host_width))
            if with_prior_binding
            else None
        ),
        previous_state_valid=torch.tensor([with_prior_binding]),
        prior_row_bindings_by_batch=(
            (("object/a", 0), ("object/b", 1)) if with_prior_binding else (),
        ),
        predictive_cache=future_cache,
        current_grid_cache=current_cache,  # type: ignore[arg-type]
        source_mask=source_mask,
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    )

    assert result.current_grid_branch is not None
    assert result.current_grid_branch.source_mask.digest == source_mask.digest
    assert result.current_grid_branch.identity_source_phase == 0
    assert current_cache.calls == 1
    assert current_cache.correction_calls == [(10,)]
    assert result.objective.objective.valid_counts["xmod/vision/representation"] == expected_valid
    assert result.primary.context.posterior_state is not None


def test_local_window_never_constructs_auxiliary_action_tails_before_source_branch(
    monkeypatch,
) -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            prediction_address_width=2,
            predictive_target_widths=(("dino_video", 4),),
        )
    )
    policy = _FakeOfficialTrainingPolicy(graph).train()
    batches = (_batch(0, 10), _batch(1, 11))
    source_mask = sample_qwen_packed_patch_mask(
        images=batches[0].model_inputs["images"],
        image_valid=batches[0].model_inputs["img_masks"],
        image_grid_thw=batches[0].model_inputs["image_grid_thw"],
        spatial_merge_size=2,
        probability=1.0,
        seed=7,
        eligible_view_indices=(0,),
    )
    original_local_bptt = full_training.run_native_local_bptt
    original_source_forward = full_training.run_native_source_masked_training_forward

    def capture_local_bptt(*args, **kwargs):
        result = original_local_bptt(*args, **kwargs)
        assert all(not hasattr(auxiliary, "official_total_loss") for auxiliary in result.auxiliary)
        return result

    def assert_released_before_source_forward(*args, **kwargs):
        assert policy.forward_grad_enabled == [True]
        assert policy.observation_forward_grad_enabled == [True]
        return original_source_forward(*args, **kwargs)

    monkeypatch.setattr(full_training, "run_native_local_bptt", capture_local_bptt)
    monkeypatch.setattr(
        full_training,
        "run_native_source_masked_training_forward",
        assert_released_before_source_forward,
    )
    result = run_native_calvin_full_objective(
        policy,
        graph=graph,
        batches=batches,
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prior_row_bindings_by_batch=((),),
        predictive_cache=_PredictiveCache(),  # type: ignore[arg-type]
        current_grid_cache=_CurrentGridCache(),  # type: ignore[arg-type]
        source_mask=source_mask,
        physical_sidecar=_Sidecar(
            {
                10: _frame(("object/a", "object/b")),
                11: _frame(("object/a", "object/b")),
            }
        ),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    )

    assert result.objective.objective.valid_counts["correction/vision/binding"] == 2
    assert policy.forward_grad_enabled == [True]
    assert policy.observation_forward_grad_enabled == [True, True]
    result.objective.objective.total.backward()
    assert graph.object_queries.grad is not None
    assert graph.object_queries.grad.abs().sum() > 0


@pytest.mark.parametrize(("with_prior_binding", "expected_valid"), [(False, 0), (True, 2)])
def test_full_objective_omitted_static_branch_is_cross_view_binding_and_uncommittable(
    with_prior_binding: bool,
    expected_valid: int,
) -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            prediction_address_width=2,
            predictive_target_widths=(("dino_video", 4),),
        )
    )
    policy = _FakeOfficialTrainingPolicy(graph).train()
    future_cache = _PredictiveCache()
    current_cache = _CurrentGridCache()
    batch = _batch(int(with_prior_binding), 10)
    omission = QwenWholeViewOmission(
        omitted_view_index=0,
        image_grid_thw=batch.model_inputs["image_grid_thw"],
        image_valid=batch.model_inputs["img_masks"],
        seed=11,
    )
    request = make_native_omitted_static_request(
        omission=omission,
        route_id=0,
        address_width=2,
        dtype=graph.object_queries.dtype,
    )
    assert request.addresses.shape == (1, 1, 2)
    assert not request.addresses.any()

    result = run_native_calvin_full_objective(
        policy,
        graph=graph,
        batches=(batch,),
        previous_state=(
            NativePosteriorState(torch.randn(1, 2, graph.config.host_width))
            if with_prior_binding
            else None
        ),
        previous_state_valid=torch.tensor([with_prior_binding]),
        prior_row_bindings_by_batch=(
            (("object/a", 0), ("object/b", 1)) if with_prior_binding else (),
        ),
        predictive_cache=future_cache,
        current_grid_cache=current_cache,  # type: ignore[arg-type]
        omitted_static_view=omission,
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=1.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    )

    assert result.current_grid_branch is None
    assert result.omitted_static_branch is not None
    assert result.omitted_static_branch.omission.digest == omission.digest
    assert result.omitted_static_branch.identity_source_phase == 0
    assert current_cache.calls == 0
    assert current_cache.correction_calls == [(10,)]
    assert current_cache.omitted_calls == 1
    assert result.objective.objective.valid_counts["correction/vision/binding"] == expected_valid
    assert result.objective.objective.valid_counts["xmod/vision/binding"] == expected_valid
    assert result.primary.context.posterior_state is not None
    assert not hasattr(result.omitted_static_branch, "posterior_state")
    result.objective.objective.total.backward()
    assert graph.object_queries.grad is not None
