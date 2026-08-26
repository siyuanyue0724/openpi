from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.physical_sequence import PhysicalSequenceAssignment
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveLossInput,
    NativePredictiveReadout,
    TargetEncoderMode,
    make_native_predictive_target,
    make_object_summary_target,
    materialize_native_predictive_terms,
    native_predictive_term,
)
from picf_next.lingbot_native.supervision import SequenceAssignment

DIGEST = "a" * 64


def _request(
    *,
    source: PredictionSource = PredictionSource.POSTERIOR,
    evidence: PredictionEvidence = PredictionEvidence.OMITTED_MODALITY,
    horizon: int = 0,
) -> NativePredictionRequest:
    return NativePredictionRequest(
        source=source,
        evidence=evidence,
        route_ids=torch.tensor([[0]]),
        horizons=torch.tensor([[horizon]]),
        addresses=torch.empty(1, 1, 0),
        valid=torch.tensor([[True]]),
    )


def _target(
    features: torch.Tensor,
    request: NativePredictionRequest,
    *,
    valid: torch.Tensor | None = None,
    importance: torch.Tensor | None = None,
) -> object:
    valid = torch.ones(features.shape[:-1], dtype=torch.bool) if valid is None else valid
    return make_native_predictive_target(
        modality="vision",
        features=features,
        valid=valid,
        importance=importance,
        route_ids=request.route_ids,
        horizons=request.horizons,
        source=request.source,
        evidence=request.evidence,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=DIGEST,
        target_data_digest=DIGEST,
        encoder_digest=DIGEST,
        query_schema_digest=DIGEST,
        validity_semantics="independent-track-mask",
        track_identity_keys=tuple(
            tuple(f"object/{track_index}" for track_index in range(features.shape[1]))
            for _ in range(features.shape[0])
        ),
    )


def test_object_summary_uses_normalized_detached_target_tokens() -> None:
    token_features = torch.tensor(
        [[[1.0, 2.0, 4.0, 8.0], [8.0, 4.0, 2.0, 1.0], [2.0, 3.0, 5.0, 7.0]]],
        requires_grad=True,
    )
    target = make_object_summary_target(
        modality="touch",
        token_features=token_features,
        track_support=torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]]),
        token_valid=torch.tensor([[True, True, False]]),
        token_footprint=torch.ones(1, 3),
        route_ids=torch.tensor([[0]]),
        horizons=torch.tensor([[0]]),
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.OMITTED_MODALITY,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=DIGEST,
        target_data_digest=DIGEST,
        encoder_digest=DIGEST,
        query_schema_digest=DIGEST,
        validity_semantics="valid-contact-group",
        track_identity_keys=(("object/0", "object/1"),),
    )
    assert target.features.shape == (1, 2, 1, 4)
    assert target.valid.all()
    assert not target.features.requires_grad
    expected = torch.nn.functional.layer_norm(token_features.detach().float(), (4,))
    torch.testing.assert_close(target.features[0, 0, 0], expected[0, 0])
    torch.testing.assert_close(target.features[0, 1, 0], expected[0, 1])


def test_route_readout_is_linear_and_gradients_reach_host_and_projection() -> None:
    readout = NativePredictiveReadout(4, 3, 2)
    hidden = torch.randn(2, 3, 2, 4, requires_grad=True)
    routes = torch.tensor([[0, 1], [1, 0]])
    output = readout(hidden, routes)
    assert output.shape == (2, 3, 2, 3)
    output.square().mean().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert readout.weight.grad is not None and readout.weight.grad.abs().sum() > 0
    assert tuple(name for name, _ in readout.named_parameters()) == ("weight",)


def test_matching_is_applied_after_forward_and_row_permutation_has_zero_cost() -> None:
    request = _request()
    hidden = torch.tensor(
        [[[[2.0, -1.0, 0.5, 3.0]], [[-2.0, 4.0, 1.0, 0.0]]]],
        requires_grad=True,
    )
    readout = NativePredictiveReadout(4, 4, 1)
    with torch.no_grad():
        readout.weight[0].copy_(torch.eye(4))
    target = _target(
        torch.stack((hidden.detach()[0, 1], hidden.detach()[0, 0])).unsqueeze(0),
        request,
    )
    prediction = readout(hidden, request.route_ids)
    correct = native_predictive_term(
        prediction=prediction,
        request=request,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[1, 0]])),
        row_binding_valid=torch.ones(1, 2, dtype=torch.bool),
        weight=1.0,
    )
    wrong = native_predictive_term(
        prediction=prediction,
        request=request,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0, 1]])),
        row_binding_valid=torch.ones(1, 2, dtype=torch.bool),
        weight=1.0,
    )
    torch.testing.assert_close(correct.normalized(), torch.tensor(0.0))
    assert wrong.normalized() > 0.5
    correct.normalized().backward()
    assert hidden.grad is not None
    assert readout.weight.grad is not None


def test_unmatched_inputs_materialize_once_and_merge_repeated_route_families() -> None:
    request = _request()
    readout = NativePredictiveReadout(4, 4, 1)
    with torch.no_grad():
        readout.weight[0].copy_(torch.eye(4))
    first_hidden = torch.randn(1, 2, 1, 4, requires_grad=True)
    second_hidden = torch.randn(1, 2, 1, 4, requires_grad=True)
    target = _target(torch.randn(1, 2, 1, 4), request)
    inputs = tuple(
        NativePredictiveLossInput(
            prediction=readout(hidden, request.route_ids),
            request=request,
            target=target,
            weight=0.25,
            identity_source_phase=0,
        )
        for hidden in (first_hidden, second_hidden)
    )

    terms = materialize_native_predictive_terms(
        inputs,
        assignment=SequenceAssignment(
            torch.tensor([[1, 0]]),
            binding_start_phase=torch.zeros(1, 2, dtype=torch.long),
        ),
        expected_track_identity_keys=target.track_identity_keys,
        sequence_time_count=1,
    )

    assert len(terms) == 1
    assert terms[0].name == "xmod/vision/binding"
    assert int(terms[0].valid.sum()) == 4
    terms[0].normalized().backward()
    assert first_hidden.grad is not None and first_hidden.grad.abs().sum() > 0
    assert second_hidden.grad is not None and second_hidden.grad.abs().sum() > 0
    assert readout.weight.grad is not None and readout.weight.grad.abs().sum() > 0


def test_predictive_materialization_rejects_structural_track_order_drift() -> None:
    request = _request()
    target = _target(torch.randn(1, 2, 1, 4), request)
    value = NativePredictiveLossInput(
        prediction=torch.randn(1, 2, 1, 4),
        request=request,
        target=target,
        weight=1.0,
        identity_source_phase=0,
    )
    with pytest.raises(ValueError, match="track identities differ"):
        materialize_native_predictive_terms(
            (value,),
            assignment=SequenceAssignment(
                torch.tensor([[0, 1]]),
                binding_start_phase=torch.zeros(1, 2, dtype=torch.long),
            ),
            expected_track_identity_keys=(("object/1", "object/0"),),
            sequence_time_count=1,
        )


def test_predictive_materialization_cannot_use_future_birth_at_past_source_phase() -> None:
    request = _request()
    target = _target(torch.zeros(1, 1, 1, 4), request)
    assignment = SequenceAssignment(
        row_to_track=torch.tensor([[0, -1]]),
        # Track zero is first identified by the posterior after observation t=1.
        binding_start_phase=torch.tensor([[3, 4]]),
    )
    past_prediction = torch.ones(1, 2, 1, 4, requires_grad=True)
    past = materialize_native_predictive_terms(
        (
            NativePredictiveLossInput(
                prediction=past_prediction,
                request=request,
                target=target,
                weight=1.0,
                identity_source_phase=1,
            ),
        ),
        assignment=assignment,
        expected_track_identity_keys=target.track_identity_keys,
        sequence_time_count=2,
    )[0]
    assert not past.valid.any()
    past.normalized().backward()
    torch.testing.assert_close(past_prediction.grad, torch.zeros_like(past_prediction))

    current_prediction = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0]], [[0.0, 0.0, 0.0, 0.0]]]],
        requires_grad=True,
    )
    current = materialize_native_predictive_terms(
        (
            NativePredictiveLossInput(
                prediction=current_prediction,
                request=request,
                target=target,
                weight=1.0,
                identity_source_phase=3,
            ),
        ),
        assignment=assignment,
        expected_track_identity_keys=target.track_identity_keys,
        sequence_time_count=2,
    )[0]
    assert int(current.valid.sum()) == 1
    current.normalized().backward()
    assert current_prediction.grad is not None
    assert current_prediction.grad[0, 0].abs().sum() > 0
    torch.testing.assert_close(
        current_prediction.grad[0, 1],
        torch.zeros_like(current_prediction.grad[0, 1]),
    )


def test_predictive_materialization_accepts_task_independent_physical_gauge() -> None:
    request = _request(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.FUTURE,
        horizon=2,
    )
    target = _target(torch.zeros(1, 1, 1, 4), request)
    prediction = torch.tensor(
        [[[[1.0, 2.0, 4.0, 8.0]], [[8.0, 4.0, 2.0, 1.0]]]],
        requires_grad=True,
    )
    terms = materialize_native_predictive_terms(
        (
            NativePredictiveLossInput(
                prediction=prediction,
                request=request,
                target=target,
                weight=1.0,
                identity_source_phase=1,
            ),
        ),
        assignment=PhysicalSequenceAssignment(
            row_to_track=torch.tensor([[0, -1]]),
            binding_start_phase=torch.tensor([[1, 2]]),
            reserved_rows=torch.tensor([[False, True]]),
            time_count=1,
        ),
        expected_track_identity_keys=target.track_identity_keys,
        sequence_time_count=1,
    )
    assert len(terms) == 1
    assert terms[0].name == "rollout/vision/binding"
    assert int(terms[0].valid.sum()) == 1
    terms[0].normalized().backward()
    assert prediction.grad is not None and prediction.grad[0, 0].abs().sum() > 0
    torch.testing.assert_close(prediction.grad[0, 1], torch.zeros_like(prediction.grad[0, 1]))


def test_unmatched_and_invalid_targets_have_no_predictive_denominator() -> None:
    request = _request()
    hidden = torch.randn(1, 2, 1, 4, requires_grad=True)
    readout = NativePredictiveReadout(4, 4, 1)
    target = _target(
        torch.randn(1, 1, 1, 4),
        request,
        valid=torch.zeros(1, 1, 1, dtype=torch.bool),
    )
    term = native_predictive_term(
        prediction=readout(hidden, request.route_ids),
        request=request,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0, -1]])),
        row_binding_valid=torch.ones(1, 2, dtype=torch.bool),
        weight=1.0,
    )
    assert int(term.valid.sum()) == 0
    torch.testing.assert_close(term.normalized(), torch.tensor(0.0))


def test_predictive_loss_preserves_absolute_visible_evidence_mass() -> None:
    request = _request(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.CURRENT_CORRECTION,
    )
    prediction = torch.tensor([[[[2.0, -1.0, 0.5, 3.0]]]])
    features = torch.tensor([[[[-2.0, 4.0, 1.0, 0.0]]]])
    assignment = SequenceAssignment(torch.tensor([[0]]))
    full = native_predictive_term(
        prediction=prediction,
        request=request,
        target=_target(features, request, importance=torch.ones(1, 1, 1)),
        assignment=assignment,
        row_binding_valid=torch.ones(1, 1, dtype=torch.bool),
        weight=1.0,
    )
    quarter = native_predictive_term(
        prediction=prediction,
        request=request,
        target=_target(features, request, importance=torch.full((1, 1, 1), 0.25)),
        assignment=assignment,
        row_binding_valid=torch.ones(1, 1, dtype=torch.bool),
        weight=1.0,
    )

    assert full.name == "correction/vision/binding"
    torch.testing.assert_close(quarter.normalized(), full.normalized() * 0.25)


def test_request_target_provenance_mismatch_fails_closed() -> None:
    request = _request()
    target = _target(torch.randn(1, 1, 1, 4), request)
    different = _request(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.PRIOR_ONLY,
    )
    with pytest.raises(ValueError, match="source request differ"):
        native_predictive_term(
            prediction=torch.randn(1, 1, 1, 4),
            request=different,
            target=target,
            assignment=SequenceAssignment(torch.tensor([[0]])),
            row_binding_valid=torch.ones(1, 1, dtype=torch.bool),
            weight=1.0,
        )


def test_current_random_grid_is_representation_only() -> None:
    request = _request(evidence=PredictionEvidence.CURRENT_RANDOM_GRID)
    target = _target(torch.randn(1, 1, 1, 4), request)
    assert not target.supports_object_binding_claim
    term = native_predictive_term(
        prediction=torch.randn(1, 1, 1, 4),
        request=request,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0]])),
        row_binding_valid=torch.ones(1, 1, dtype=torch.bool),
        weight=1.0,
    )
    assert term.name == "xmod/vision/representation"
