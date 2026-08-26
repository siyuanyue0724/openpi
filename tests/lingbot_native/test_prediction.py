from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
    TokenizerDependencyMap,
)


def _request(
    *,
    source: PredictionSource = PredictionSource.POSTERIOR,
    evidence: PredictionEvidence = PredictionEvidence.OMITTED_MODALITY,
    horizon: int = 0,
) -> NativePredictionRequest:
    return NativePredictionRequest(
        source=source,
        evidence=evidence,
        route_ids=torch.tensor([[0, 1]]),
        horizons=torch.full((1, 2), horizon, dtype=torch.long),
        addresses=torch.empty(1, 2, 0),
        valid=torch.tensor([[True, True]]),
    )


def test_prediction_evidence_grade_distinguishes_representation_from_binding() -> None:
    random_grid = _request(evidence=PredictionEvidence.CURRENT_RANDOM_GRID)
    omitted = _request(evidence=PredictionEvidence.OMITTED_MODALITY)
    prior = _request(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.PRIOR_ONLY,
    )
    future = _request(evidence=PredictionEvidence.FUTURE, horizon=8)
    assert not random_grid.supports_object_binding_claim
    assert omitted.supports_object_binding_claim
    assert prior.supports_object_binding_claim
    assert future.supports_object_binding_claim
    with pytest.raises(ValueError, match="prior-only"):
        _request(evidence=PredictionEvidence.PRIOR_ONLY)


def test_nonspatial_prediction_evidence_cannot_encode_a_hidden_address() -> None:
    for evidence, source in (
        (PredictionEvidence.CURRENT_CORRECTION, PredictionSource.PRIOR),
        (PredictionEvidence.CURRENT_PRIOR, PredictionSource.PRIOR),
        (PredictionEvidence.CURRENT_POSTERIOR, PredictionSource.POSTERIOR),
        (PredictionEvidence.OMITTED_MODALITY, PredictionSource.POSTERIOR),
    ):
        with pytest.raises(ValueError, match="zero address"):
            NativePredictionRequest(
                source=source,
                evidence=evidence,
                route_ids=torch.zeros(1, 1, dtype=torch.long),
                horizons=torch.zeros(1, 1, dtype=torch.long),
                addresses=torch.ones(1, 1, 2),
                valid=torch.ones(1, 1, dtype=torch.bool),
            )


def test_dependency_map_removes_every_output_whose_receptive_field_overlaps_target() -> None:
    dependency = TokenizerDependencyMap(
        torch.tensor(
            [
                [True, True, False, False],
                [False, True, True, False],
                [False, False, False, True],
            ]
        )
    )
    raw_target = torch.tensor([[False, True, False, False]])
    valid = dependency.source_output_valid(raw_target)
    assert torch.equal(valid, torch.tensor([[False, False, True]]))
    assert len(dependency.digest) == 64


def test_prediction_queries_are_weight_shared_and_read_exactly_their_paired_row() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=3,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            prediction_route_count=2,
        )
    )
    graph.train()
    request = _request()
    context = LingBotNativeContext(
        controls=ExecutedControlBatch.reset_only(
            batch_size=1,
            action_dim=2,
            device="cpu",
            dtype=torch.float32,
        ),
        native_roles=torch.tensor([[int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]),
        native_valid=torch.ones(1, 2, dtype=torch.bool),
        instruction_last_index=torch.tensor([1]),
        prediction_request=request,
    )
    prefix = torch.randn(1, 2, 8)
    action = torch.randn(1, 2, 4)
    host_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    positions = torch.arange(4).reshape(1, 1, 4).expand(3, 1, 4).clone()
    prepared, mask, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=host_mask,
        position_ids=positions,
        visual_pos_masks=torch.tensor([[True, False]]),
        context=context,
    )
    assert runtime is not None and runtime.prediction_slice is not None
    prediction = prepared[0][:, runtime.prediction_slice].reshape(1, 3, 2, 8)
    torch.testing.assert_close(prediction[:, 0], prediction[:, 1])
    torch.testing.assert_close(prediction[:, 1], prediction[:, 2])
    for row in range(3):
        source_index = runtime.posterior_slice.start + row
        for query in range(2):
            index = runtime.prediction_slice.start + row * 2 + query
            readable = mask[0, index].nonzero().flatten().tolist()
            assert readable == sorted((source_index, index))
    action_index = mask.shape[1] - 1
    assert not mask[0, action_index, runtime.prediction_slice].any()

    graph.finalize_joint_outputs(outputs_embeds=prepared, runtime=runtime)
    assert context.prediction_hidden is not None
    assert context.prediction_hidden.shape == (1, 3, 2, 8)


def test_prediction_queries_are_rejected_at_deployment() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            prediction_route_count=2,
        )
    ).eval()
    context = LingBotNativeContext(
        controls=ExecutedControlBatch.reset_only(
            batch_size=1, action_dim=2, device="cpu", dtype=torch.float32
        ),
        native_roles=torch.tensor([[int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]),
        native_valid=torch.ones(1, 2, dtype=torch.bool),
        instruction_last_index=torch.tensor([1]),
        prediction_request=_request(),
    )
    with pytest.raises(ValueError, match="training-only"):
        graph.prepare_joint_inputs(
            inputs_embeds=[torch.randn(1, 2, 8), None],
            attention_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            position_ids=torch.zeros(3, 1, 2, dtype=torch.long),
            visual_pos_masks=torch.tensor([[True, False]]),
            context=context,
        )
