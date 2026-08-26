from __future__ import annotations

import inspect

import pytest
import torch

from picf_next.lingbot_native.entity_set_objective import PhysicalFrameTargets
from picf_next.lingbot_native.entity_training import (
    TaskIndependentEntityObjectiveConfig,
    compose_task_independent_entity_objective,
    compose_task_independent_persistent_entity_objective,
)
from picf_next.lingbot_native.physical_relations import PhysicalEntityReadout
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveLossInput,
    TargetEncoderMode,
    make_native_predictive_target,
)

DIGEST = "a" * 64


def _physical_inputs() -> tuple[
    PhysicalEntityReadout,
    object,
    PhysicalFrameTargets,
]:
    readout = PhysicalEntityReadout(4)
    rows = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    sensors = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    relation = readout(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=torch.ones(1, 3, dtype=torch.bool),
    )
    targets = PhysicalFrameTargets(
        masks=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        mask_valid=torch.ones(1, 2, 3, dtype=torch.bool),
        existence=torch.ones(1, 2),
        existence_valid=torch.ones(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 3),
        inventory_exhaustive=torch.ones(1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    return readout, relation, targets


def test_entity_objective_has_no_task_or_winner_input_surface() -> None:
    parameters = inspect.signature(compose_task_independent_entity_objective).parameters
    assert tuple(parameters) == ("official_policy_loss", "relations", "targets", "config")
    assert not any("task" in name or "winner" in name or "match" in name for name in parameters)

    persistent_parameters = inspect.signature(
        compose_task_independent_persistent_entity_objective
    ).parameters
    assert not any(
        "task" in name or "winner" in name or "match" in name for name in persistent_parameters
    )


def test_entity_only_objective_backpropagates_through_host_native_readout() -> None:
    readout, relation, targets = _physical_inputs()

    result = compose_task_independent_entity_objective(
        official_policy_loss=None,
        relations=(relation,),
        targets=(targets,),
        config=TaskIndependentEntityObjectiveConfig(action_weight=0.0, entity_weight=0.4),
    )

    torch.testing.assert_close(
        result.objective.total,
        0.4 * result.frame_losses[0].total,
    )
    result.objective.total.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in readout.parameters()
    )


def test_joint_objective_preserves_official_action_loss_as_its_own_family() -> None:
    _readout, relation, targets = _physical_inputs()
    action = torch.tensor(0.6, requires_grad=True)

    result = compose_task_independent_entity_objective(
        official_policy_loss=action,
        relations=(relation, relation),
        targets=(targets, targets),
        config=TaskIndependentEntityObjectiveConfig(action_weight=1.5, entity_weight=0.2),
    )

    entity_mean = torch.stack(tuple(loss.total for loss in result.frame_losses)).mean()
    torch.testing.assert_close(result.objective.total, 1.5 * action + 0.2 * entity_mean)
    result.objective.total.backward()
    torch.testing.assert_close(action.grad, torch.tensor(1.5))


def test_persistent_objective_keeps_prior_identity_and_commits_only_primary_births() -> None:
    readout, relation, targets = _physical_inputs()

    result = compose_task_independent_persistent_entity_objective(
        official_policy_loss=None,
        relations=(relation, relation),
        targets=(targets, targets),
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_row_bindings_by_batch=((("object/a", 0),),),
        config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
    )

    assert result.assignment.row_to_track[0, 0].item() == 0
    assert result.assignment.binding_start_phase[0, 0].item() == 0
    assert result.row_bindings_by_batch == ((("object/a", 0), ("object/b", 1)),)
    assert len(result.frame_losses) == 2
    result.objective.total.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in readout.parameters()
    )


def test_persistent_objective_uses_same_physical_gauge_for_predictive_identity() -> None:
    _readout, relation, targets = _physical_inputs()
    request = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.FUTURE,
        route_ids=torch.tensor([[0]]),
        horizons=torch.tensor([[2]]),
        addresses=torch.empty(1, 1, 0),
        valid=torch.tensor([[True]]),
    )
    prediction = torch.tensor(
        [[[[1.0, 2.0, 4.0, 8.0]], [[8.0, 4.0, 2.0, 1.0]]]],
        requires_grad=True,
    )
    target = make_native_predictive_target(
        modality="vision",
        features=torch.zeros(1, 2, 1, 4),
        valid=torch.ones(1, 2, 1, dtype=torch.bool),
        importance=None,
        route_ids=request.route_ids,
        horizons=request.horizons,
        source=request.source,
        evidence=request.evidence,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=DIGEST,
        target_data_digest=DIGEST,
        encoder_digest=DIGEST,
        query_schema_digest=DIGEST,
        validity_semantics="physical-track-only",
        track_identity_keys=(("object/a", "object/b"),),
    )
    result = compose_task_independent_persistent_entity_objective(
        official_policy_loss=None,
        relations=(relation,),
        targets=(targets,),
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_row_bindings_by_batch=((),),
        config=TaskIndependentEntityObjectiveConfig(
            action_weight=0.0,
            predictive_weight=0.5,
        ),
        predictive_inputs=(
            NativePredictiveLossInput(
                prediction=prediction,
                request=request,
                target=target,
                weight=1.0,
                identity_source_phase=1,
            ),
        ),
    )

    assert result.objective.valid_counts["rollout/vision/binding"] == 2
    assert result.objective.family_terms["predictive"] > 0
    result.objective.total.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0


def test_entity_objective_rejects_cross_architecture_relations() -> None:
    _readout, _relation, targets = _physical_inputs()

    with pytest.raises(TypeError, match="task-independent physical relations"):
        compose_task_independent_entity_objective(
            official_policy_loss=None,
            relations=(object(),),  # type: ignore[arg-type]
            targets=(targets,),
            config=TaskIndependentEntityObjectiveConfig(action_weight=0.0),
        )


def test_active_action_family_rejects_detached_compatibility_numbers() -> None:
    _readout, relation, targets = _physical_inputs()

    with pytest.raises(ValueError, match="attached policy loss"):
        compose_task_independent_entity_objective(
            official_policy_loss=torch.tensor(0.6),
            relations=(relation,),
            targets=(targets,),
            config=TaskIndependentEntityObjectiveConfig(),
        )
