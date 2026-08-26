from __future__ import annotations

import copy

import pytest
import torch

from picf_next.lingbot_native.calvin_objective import NativeCALVINObjectiveResult
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
    SequenceAssignment,
)
from picf_next.lingbot_native.task_diagnostics import (
    TASK_ROW_DIAGNOSTIC_SCHEMA,
    build_task_row_diagnostics,
    validate_task_row_diagnostic,
    validate_task_row_diagnostics,
)
from picf_next.objective import UnifiedObjective


def _objective(*, exact: bool = True) -> NativeCALVINObjectiveResult:
    logits = torch.tensor([[-1.0, 0.5, 0.1, -0.2]])
    support = torch.zeros(1, 1, 2, 4)
    ownership = torch.softmax(
        torch.cat((support, torch.zeros(1, 1, 2, 1)), dim=-1),
        dim=-1,
    )
    predictions = NativeSequencePredictions(
        support_logits=support,
        ownership=ownership,
        existence_logits=torch.zeros(1, 1, 4),
        task_relevance_logits=logits,
        dense_task_grounding_logits=torch.zeros(1, 1, 2),
    )
    targets = NativeSequenceTargets(
        masks=torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        mask_valid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        existence=torch.ones(1, 1, 2),
        existence_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        task_relevance=torch.tensor([[1.0, 0.0]]),
        task_valid=(torch.ones(1, 2, dtype=torch.bool) if exact else torch.tensor([[True, False]])),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 1, 2),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    return NativeCALVINObjectiveResult(
        objective=UnifiedObjective(
            total=torch.zeros(()),
            normalized_terms={},
            valid_counts={},
        ),
        predictions=predictions,
        targets=targets,
        assignment=SequenceAssignment(torch.tensor([[1, 0, -1, -1]])),
        track_identity_keys_by_batch=(("pink_block", "drawer"),),
        row_bindings_by_batch=((("drawer", 0), ("pink_block", 1)),),
        predictive_terms=(),
        structural_terms=(),
    )


def test_task_row_diagnostic_recomputes_strict_winner_and_assignment() -> None:
    (report,) = build_task_row_diagnostics(_objective())

    assert report["schema"] == TASK_ROW_DIAGNOSTIC_SCHEMA
    assert report["sequence_time_count"] == 1
    assert report["source_time"] == 0
    assert report["source_side"] == "posterior"
    assert report["source_phase"] == 1
    assert report["binding_start_phase"] == [1, 1, 2, 2]
    assert report["source_binding_valid"] == [True, True, False, False]
    assert report["exact_task"] is True
    assert report["target_rows"] == [1]
    assert report["target_identity_keys"] == ["pink_block"]
    assert report["materialized_target_identity_keys"] == ["pink_block"]
    assert report["unmaterialized_target_identity_keys"] == []
    assert report["known_negative_rows"] == [0, 2, 3]
    assert report["worst_target_rank"] == 1
    assert report["all_targets_beat_known_negatives"] is True
    assert report["target_vs_hardest_negative_logit_margin"] == pytest.approx(0.4)
    assert validate_task_row_diagnostic(report) == report
    assert validate_task_row_diagnostics([report], expected_batch_size=1) == [report]


def test_partial_task_keeps_unmatched_rows_unknown() -> None:
    (report,) = build_task_row_diagnostics(_objective(exact=False))

    assert report["exact_task"] is False
    assert report["row_task_valid"] == [False, True, False, False]
    assert report["known_negative_rows"] == []
    assert report["worst_target_rank"] is None
    assert report["all_targets_beat_known_negatives"] is None


def test_censored_exact_target_is_reported_without_inventing_a_row() -> None:
    objective = _objective()
    targets = NativeSequenceTargets(
        masks=objective.targets.masks,
        mask_valid=objective.targets.mask_valid,
        existence=objective.targets.existence,
        existence_valid=objective.targets.existence_valid,
        task_relevance=objective.targets.task_relevance,
        task_valid=objective.targets.task_valid,
        track_valid=objective.targets.track_valid,
        capacity_censored=torch.tensor([[True, False]]),
        token_observed_fraction=objective.targets.token_observed_fraction,
        inventory_exhaustive=objective.targets.inventory_exhaustive,
        exclusive_ownership=True,
    )
    censored = NativeCALVINObjectiveResult(
        objective=objective.objective,
        predictions=objective.predictions,
        targets=targets,
        assignment=SequenceAssignment(torch.tensor([[1, -1, -1, -1]])),
        track_identity_keys_by_batch=objective.track_identity_keys_by_batch,
        row_bindings_by_batch=((("drawer", 0),),),
        predictive_terms=(),
        structural_terms=(),
    )

    (report,) = build_task_row_diagnostics(censored)

    assert report["exact_task"] is True
    assert report["target_identity_keys"] == ["pink_block"]
    assert report["materialized_target_identity_keys"] == []
    assert report["unmaterialized_target_identity_keys"] == ["pink_block"]
    assert report["target_rows"] == []
    assert report["all_targets_beat_known_negatives"] is None
    assert validate_task_row_diagnostic(report) == report


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("assignment_sha256", "0" * 64, "assignment digest"),
        ("source_phase", 0, "source cut"),
        ("binding_start_phase", [3, 1, 2, 2], "binding phases"),
        ("source_binding_valid", [False, True, False, False], "binding validity"),
        ("task_probabilities", [0.5, 0.5, 0.5, 0.5], "probabilities"),
        ("target_rows", [0], "target_rows"),
        ("worst_target_rank", 2, "rank"),
        ("all_targets_beat_known_negatives", False, "winner"),
    ),
)
def test_task_row_diagnostic_fails_closed_on_tampering(
    field: str,
    replacement: object,
    message: str,
) -> None:
    (report,) = build_task_row_diagnostics(_objective())
    edited = copy.deepcopy(report)
    edited[field] = replacement

    with pytest.raises(ValueError, match=message):
        validate_task_row_diagnostic(edited)
