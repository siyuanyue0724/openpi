from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from picf_next.lingbot_native.supervision import (
    TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
    TOKEN_MICRO_OWNERSHIP,
    NativeSequencePredictions,
    NativeSequenceTargets,
    SequenceAssignment,
    _balanced_task_score,
    _entity_conditional_ownership_nll_samples,
    _exclusive_ownership_nll_samples,
    _pairwise_assignment_costs,
    _support_score,
    _term,
    extend_sequence_row_bindings,
    match_sequence_rows,
    match_sequence_rows_with_bindings,
    materialize_row_task_supervision,
    sequence_set_terms,
)


def _predictions(
    support: torch.Tensor,
    *,
    existence: torch.Tensor | None = None,
    task: torch.Tensor | None = None,
    dense_task: torch.Tensor | None = None,
) -> NativeSequencePredictions:
    batch, time, tokens, rows = support.shape
    support = support.clone().requires_grad_()
    existence = (
        torch.zeros(batch, time, rows) if existence is None else existence.clone()
    ).requires_grad_()
    task = (torch.zeros(batch, rows) if task is None else task.clone()).requires_grad_()
    dense_task = (
        torch.zeros(batch, time, tokens) if dense_task is None else dense_task.clone()
    ).requires_grad_()
    ownership_logits = torch.cat(
        (
            support,
            torch.zeros(
                batch,
                time,
                tokens,
                1,
                dtype=support.dtype,
                device=support.device,
            ),
        ),
        dim=-1,
    )
    ownership = torch.softmax(ownership_logits, dim=-1)
    ownership_log_probability = F.log_softmax(ownership_logits.float(), dim=-1)
    return NativeSequencePredictions(
        support_logits=support,
        ownership=ownership,
        ownership_log_probability=ownership_log_probability,
        existence_logits=existence,
        task_relevance_logits=task,
        dense_task_grounding_logits=dense_task,
    )


def _targets(
    masks: torch.Tensor,
    *,
    mask_valid: torch.Tensor | None = None,
    existence: torch.Tensor | None = None,
    existence_valid: torch.Tensor | None = None,
    task: torch.Tensor | None = None,
    task_valid: torch.Tensor | None = None,
    track_valid: torch.Tensor | None = None,
    capacity_censored: torch.Tensor | None = None,
    token_observed_fraction: torch.Tensor | None = None,
    inventory_exhaustive: torch.Tensor | None = None,
    exclusive: bool = False,
) -> NativeSequenceTargets:
    batch, time, tracks, tokens = masks.shape
    resolved_mask_valid = (
        torch.ones_like(masks, dtype=torch.bool) if mask_valid is None else mask_valid
    )
    if token_observed_fraction is None:
        token_observed_fraction = (
            resolved_mask_valid.any(dim=2).float()
            if exclusive
            else torch.zeros(batch, time, tokens)
        )
    return NativeSequenceTargets(
        masks=masks.float(),
        mask_valid=resolved_mask_valid,
        existence=torch.ones(batch, time, tracks) if existence is None else existence,
        existence_valid=torch.ones(batch, time, tracks, dtype=torch.bool)
        if existence_valid is None
        else existence_valid,
        task_relevance=torch.zeros(batch, tracks) if task is None else task,
        task_valid=torch.zeros(batch, tracks, dtype=torch.bool)
        if task_valid is None
        else task_valid,
        track_valid=torch.ones(batch, tracks, dtype=torch.bool)
        if track_valid is None
        else track_valid,
        capacity_censored=torch.zeros(batch, tracks, dtype=torch.bool)
        if capacity_censored is None
        else capacity_censored,
        token_observed_fraction=token_observed_fraction,
        inventory_exhaustive=torch.zeros(batch, time, dtype=torch.bool)
        if inventory_exhaustive is None
        else inventory_exhaustive,
        exclusive_ownership=exclusive,
    )


def _normalized(terms: tuple) -> dict[str, torch.Tensor]:
    return {term.name: term.normalized() for term in terms}


def test_empty_term_is_exact_zero_with_backward_connectivity() -> None:
    reference = torch.randn(2, 3, requires_grad=True)
    term = _term(
        "set/empty",
        [],
        reference=reference,
        weight=1.0,
    )

    assert term.values.shape == term.valid.shape == (1,)
    assert not term.valid.any()
    torch.testing.assert_close(term.values, torch.zeros(1))
    torch.testing.assert_close(term.normalized(), torch.zeros(()))

    term.normalized().backward()
    assert reference.grad is not None
    torch.testing.assert_close(reference.grad, torch.zeros_like(reference))


def test_empty_structural_families_keep_zero_gradient_connections() -> None:
    predictions = _predictions(torch.zeros(1, 1, 1, 2))
    targets = _targets(
        torch.ones(1, 1, 1, 1),
        mask_valid=torch.zeros(1, 1, 1, 1, dtype=torch.bool),
        existence_valid=torch.zeros(1, 1, 1, dtype=torch.bool),
        task_valid=torch.zeros(1, 1, dtype=torch.bool),
    )
    terms = sequence_set_terms(
        predictions,
        targets,
        SequenceAssignment(torch.tensor([[-1, -1]])),
        support_weight=1.0,
        existence_weight=1.0,
        task_weight=1.0,
        dense_task_weight=1.0,
    )

    sum(term.normalized() for term in terms).backward()
    for prediction in (
        predictions.support_logits,
        predictions.existence_logits,
        predictions.task_relevance_logits,
        predictions.dense_task_grounding_logits,
    ):
        assert prediction.grad is not None
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_sequence_matcher_recovers_row_permutation_and_gradients_stay_loss_side() -> None:
    masks = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]])
    support = torch.tensor(
        [[[[-6.0, 6.0], [-6.0, -6.0], [6.0, -6.0]], [[-6.0, 6.0], [-6.0, -6.0], [6.0, -6.0]]]]
    )
    predictions = _predictions(
        support,
        existence=torch.full((1, 2, 2), 4.0),
        task=torch.tensor([[-4.0, 4.0]]),
    )
    targets = _targets(
        masks,
        task=torch.tensor([[1.0, 0.0]]),
        task_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    assignment = match_sequence_rows(predictions, targets)
    assert torch.equal(assignment.row_to_track, torch.tensor([[1, 0]]))

    terms = sequence_set_terms(
        predictions,
        targets,
        assignment,
        support_weight=1.0,
        existence_weight=1.0,
        task_weight=1.0,
        dense_task_weight=0.0,
    )
    sum(term.normalized() for term in terms).backward()
    assert predictions.support_logits.grad is not None
    assert predictions.support_logits.grad.abs().sum() > 0
    assert predictions.existence_logits.grad is not None
    assert predictions.existence_logits.grad.abs().sum() > 0
    assert predictions.task_relevance_logits.grad is not None
    assert predictions.task_relevance_logits.grad.abs().sum() > 0


def test_sequence_matcher_is_invariant_to_task_relevance() -> None:
    masks = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]])
    support = torch.tensor(
        [[[[-6.0, 6.0], [-6.0, -6.0], [6.0, -6.0]], [[-6.0, 6.0], [-6.0, -6.0], [6.0, -6.0]]]]
    )
    targets = _targets(
        masks,
        task=torch.tensor([[1.0, 0.0]]),
        task_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    first = _predictions(
        support,
        existence=torch.full((1, 2, 2), 4.0),
        task=torch.tensor([[-20.0, 20.0]]),
    )
    second = _predictions(
        support,
        existence=torch.full((1, 2, 2), 4.0),
        task=torch.tensor([[20.0, -20.0]]),
    )

    first_assignment = match_sequence_rows(first, targets)
    second_assignment = match_sequence_rows(second, targets)

    assert torch.equal(first_assignment.row_to_track, torch.tensor([[1, 0]]))
    assert torch.equal(second_assignment.row_to_track, first_assignment.row_to_track)


def test_bound_sequence_matcher_preserves_identity_and_allocates_only_births() -> None:
    masks = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    targets = _targets(masks)
    first = _predictions(
        torch.tensor([[[[-8.0, 8.0], [8.0, -8.0]]]]),
        existence=torch.full((1, 1, 2), 4.0),
    )
    first_assignment = match_sequence_rows_with_bindings(
        first,
        targets,
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=((),),
    )
    first_bindings = extend_sequence_row_bindings(
        first_assignment,
        targets,
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=((),),
    )
    assert torch.equal(first_assignment.row_to_track, torch.tensor([[1, 0]]))
    assert first_bindings == ((("object/a", 1), ("object/b", 0)),)

    swapped_preference = _predictions(
        torch.tensor([[[[8.0, -8.0], [-8.0, 8.0]]]]),
        existence=torch.full((1, 1, 2), 4.0),
    )
    continuation = match_sequence_rows_with_bindings(
        swapped_preference,
        targets,
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=first_bindings,
    )
    assert torch.equal(continuation.row_to_track, first_assignment.row_to_track)


def test_bound_sequence_matcher_reserves_absent_rows_and_rejects_rebinding() -> None:
    targets = _targets(
        torch.tensor([[[[1.0, 0.0]]]]),
        track_valid=torch.tensor([[True]]),
    )
    predictions = _predictions(
        torch.tensor([[[[8.0, -8.0, 7.0], [-8.0, 8.0, -7.0]]]]),
        existence=torch.full((1, 1, 3), 4.0),
    )
    assignment = match_sequence_rows_with_bindings(
        predictions,
        targets,
        identity_keys_by_batch=(("object/new",),),
        prior_bindings_by_batch=((("object/occluded", 0),),),
    )
    assert assignment.row_to_track[0, 0] == -1
    assert assignment.row_to_track[0, 2] == 0
    bindings = extend_sequence_row_bindings(
        assignment,
        targets,
        identity_keys_by_batch=(("object/new",),),
        prior_bindings_by_batch=((("object/occluded", 0),),),
    )
    assert bindings == ((("object/new", 2), ("object/occluded", 0)),)

    with pytest.raises(ValueError, match="changed rows"):
        extend_sequence_row_bindings(
            SequenceAssignment(torch.tensor([[0, -1, -1]])),
            targets,
            identity_keys_by_batch=(("object/new",),),
            prior_bindings_by_batch=((("object/new", 1),),),
        )


def test_unseen_fully_occluded_track_is_deferred_without_false_negative_existence() -> None:
    targets = _targets(
        torch.zeros(1, 1, 1, 2),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
    )
    predictions = _predictions(
        torch.tensor([[[[9.0, -9.0], [-9.0, 9.0]]]]),
        existence=torch.full((1, 1, 2), 4.0),
    )

    assignment = match_sequence_rows(predictions, targets)
    assert torch.equal(assignment.row_to_track, torch.tensor([[-1, -1]]))
    terms = {
        term.name: term
        for term in sequence_set_terms(
            predictions,
            targets,
            assignment,
            support_weight=1.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=0.0,
        )
    }
    assert not terms["set/existence"].valid.any()


def test_unseen_fully_occluded_relevant_track_does_not_create_task_negatives() -> None:
    targets = _targets(
        torch.zeros(1, 1, 1, 2),
        task=torch.ones(1, 1),
        task_valid=torch.ones(1, 1, dtype=torch.bool),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
    )
    predictions = _predictions(
        torch.zeros(1, 1, 2, 2),
        task=torch.tensor([[4.0, -4.0]]),
    )

    assignment = match_sequence_rows(predictions, targets)
    assert torch.equal(assignment.row_to_track, torch.tensor([[-1, -1]]))
    row_task = materialize_row_task_supervision(
        targets,
        assignment,
        batch_index=0,
        dtype=predictions.task_relevance_logits.dtype,
    )
    assert row_task.exact_task
    assert not row_task.valid.any()

    task_term = {
        term.name: term
        for term in sequence_set_terms(
            predictions,
            targets,
            assignment,
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=1.0,
            dense_task_weight=0.0,
        )
    }["set/task"]
    assert not task_term.valid.any()
    task_term.normalized().backward()
    assert predictions.task_relevance_logits.grad is not None
    torch.testing.assert_close(
        predictions.task_relevance_logits.grad,
        torch.zeros_like(predictions.task_relevance_logits),
    )


def test_established_binding_survives_complete_visual_occlusion() -> None:
    targets = _targets(
        torch.zeros(1, 1, 1, 2),
        task=torch.ones(1, 1),
        task_valid=torch.ones(1, 1, dtype=torch.bool),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
    )
    predictions = _predictions(
        torch.zeros(1, 1, 2, 2),
        existence=torch.full((1, 1, 2), 4.0),
    )

    assignment = match_sequence_rows_with_bindings(
        predictions,
        targets,
        identity_keys_by_batch=(("object/hidden",),),
        prior_bindings_by_batch=((("object/hidden", 1),),),
    )
    assert torch.equal(assignment.row_to_track, torch.tensor([[-1, 0]]))
    assert torch.equal(assignment.binding_start_phase, torch.tensor([[2, 0]]))
    row_task = materialize_row_task_supervision(
        targets,
        assignment,
        batch_index=0,
        dtype=predictions.task_relevance_logits.dtype,
    )
    assert row_task.exact_task
    assert torch.equal(row_task.valid, torch.tensor([True, True]))
    torch.testing.assert_close(row_task.target, torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize("future_row", [0, 1])
def test_future_birth_assignment_cannot_change_past_existence_gradient(
    future_row: int,
) -> None:
    masks = torch.tensor([[[[0.0]], [[1.0]]]])
    targets = _targets(
        masks,
        existence=torch.ones(1, 2, 1),
        existence_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        task=torch.ones(1, 1),
        task_valid=torch.ones(1, 1, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 2, 1),
        inventory_exhaustive=torch.ones(1, 2, dtype=torch.bool),
        exclusive=True,
    )
    support = torch.tensor(
        [[[[100.0, -100.0]], [[-10.0, 10.0]]]]
        if future_row == 1
        else [[[[-100.0, 100.0]], [[10.0, -10.0]]]]
    )
    predictions = _predictions(
        support,
        existence=torch.tensor([[[50.0, -50.0], [0.0, 0.0]]]),
    )

    assignment = match_sequence_rows(predictions, targets)
    expected_row_to_track = torch.full((1, 2), -1, dtype=torch.long)
    expected_row_to_track[0, future_row] = 0
    expected_start = torch.full((1, 2), 4, dtype=torch.long)
    expected_start[0, future_row] = 3
    assert torch.equal(assignment.row_to_track, expected_row_to_track)
    assert torch.equal(assignment.binding_start_phase, expected_start)

    existence = {
        term.name: term
        for term in sequence_set_terms(
            predictions,
            targets,
            assignment,
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=0.0,
        )
    }["set/existence"]
    assert int(existence.valid.sum()) == 2
    existence.normalized().backward()
    assert predictions.existence_logits.grad is not None
    torch.testing.assert_close(
        predictions.existence_logits.grad[:, 0],
        torch.zeros_like(predictions.existence_logits.grad[:, 0]),
    )
    assert predictions.existence_logits.grad[:, 1].abs().sum() > 0
    assert extend_sequence_row_bindings(
        assignment,
        targets,
        identity_keys_by_batch=(("object/future",),),
        prior_bindings_by_batch=((),),
    ) == ((),)


@pytest.mark.parametrize("future_prefers_row", [0, 1])
def test_future_logits_cannot_change_a_row_frozen_at_first_evidence(
    future_prefers_row: int,
) -> None:
    targets = _targets(
        torch.ones(1, 2, 1, 1),
        existence=torch.ones(1, 2, 1),
        existence_valid=torch.ones(1, 2, 1, dtype=torch.bool),
    )
    future = [12.0, -12.0] if future_prefers_row == 0 else [-100.0, 100.0]
    predictions = _predictions(
        torch.tensor([[[[12.0, -12.0]], [future]]]),
        existence=torch.zeros(1, 2, 2),
    )

    assignment = match_sequence_rows(predictions, targets)

    assert torch.equal(assignment.row_to_track, torch.tensor([[0, -1]]))
    assert torch.equal(assignment.binding_start_phase, torch.tensor([[1, 4]]))


def test_bound_sequence_contract_rejects_noncanonical_or_duplicate_identities() -> None:
    targets = _targets(torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]))
    predictions = _predictions(
        torch.tensor([[[[8.0, -8.0], [-8.0, 8.0]]]]),
        existence=torch.full((1, 1, 2), 4.0),
    )
    with pytest.raises(ValueError, match="canonical"):
        match_sequence_rows_with_bindings(
            predictions,
            targets,
            identity_keys_by_batch=(("object/a", "object/b"),),
            prior_bindings_by_batch=((("object/b", 1), ("object/a", 0)),),
        )
    with pytest.raises(ValueError, match="identities differ"):
        extend_sequence_row_bindings(
            SequenceAssignment(torch.tensor([[0, 1]])),
            targets,
            identity_keys_by_batch=(("object/a", "object/a"),),
            prior_bindings_by_batch=((),),
        )


def test_exclusive_sequence_matcher_uses_categorical_ownership() -> None:
    masks = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]])
    targets = _targets(
        masks,
        token_observed_fraction=torch.ones(1, 1, 3),
        exclusive=True,
    )
    raw_support = torch.tensor([[[[12.0, -12.0], [-12.0, -12.0], [-12.0, 12.0]]]])
    ownership = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]])
    predictions = NativeSequencePredictions(
        support_logits=raw_support,
        ownership=ownership,
        existence_logits=torch.full((1, 1, 2), 4.0),
        task_relevance_logits=torch.zeros(1, 2),
        dense_task_grounding_logits=torch.zeros(1, 1, 3),
    )

    assignment = match_sequence_rows(predictions, targets)

    assert torch.equal(assignment.row_to_track, torch.tensor([[1, 0]]))


def test_factorized_prediction_contract_rejects_broken_product_or_event() -> None:
    ownership = torch.tensor([[[[0.2, 0.3, 0.5]], [[0.4, 0.1, 0.5]]]])
    task_logits = torch.zeros(1, 2)
    task_rows = task_logits.sigmoid()
    task_rows_by_time = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])
    task_object = task_rows_by_time[:, :, None] * ownership[..., :-1]
    task_event = torch.cat(
        (task_object, 1 - task_object.sum(dim=-1, keepdim=True)),
        dim=-1,
    )
    common = {
        "support_logits": torch.zeros(1, 2, 1, 2),
        "ownership": ownership,
        "ownership_log_probability": ownership.log(),
        "existence_logits": torch.zeros(1, 2, 2),
        "task_relevance_logits": task_logits,
        "dense_task_grounding_logits": torch.zeros(1, 2, 1),
        "task_object_log_probability": task_object.log(),
        "task_object_probability": task_object,
        "task_event_distribution": task_event,
        "task_row_probability": task_rows,
        "task_row_probability_by_time": task_rows_by_time,
    }
    NativeSequencePredictions(**common)

    broken_product = task_object.clone()
    broken_product[0, 0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="all-time product identity"):
        NativeSequencePredictions(**{**common, "task_object_probability": broken_product})

    broken_event = task_event.clone()
    broken_event[0, 0, 0, -1] -= 0.1
    with pytest.raises(ValueError, match="closed event"):
        NativeSequencePredictions(**{**common, "task_event_distribution": broken_event})


def test_partial_labels_never_turn_an_unmatched_row_into_background() -> None:
    masks = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    targets = _targets(masks)
    assignment = SequenceAssignment(torch.tensor([[0, -1]]))
    first = _predictions(torch.tensor([[[[5.0, -9.0], [-5.0, 9.0]], [[5.0, -9.0], [-5.0, 9.0]]]]))
    second = _predictions(
        torch.tensor([[[[5.0, 9.0], [-5.0, -9.0]], [[5.0, 9.0], [-5.0, -9.0]]]]),
        existence=torch.tensor([[[0.0, 20.0], [0.0, -20.0]]]),
        task=torch.tensor([[0.0, 20.0]]),
    )
    kwargs = dict(
        support_weight=1.0,
        existence_weight=1.0,
        task_weight=1.0,
        dense_task_weight=0.0,
    )
    first_terms = _normalized(sequence_set_terms(first, targets, assignment, **kwargs))
    second_terms = _normalized(sequence_set_terms(second, targets, assignment, **kwargs))
    for name in first_terms:
        torch.testing.assert_close(first_terms[name], second_terms[name])


def test_exhaustive_regions_create_declared_unmatched_negatives() -> None:
    masks = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    targets = _targets(
        masks,
        token_observed_fraction=torch.ones(1, 2, 2),
        inventory_exhaustive=torch.ones(1, 2, dtype=torch.bool),
    )
    assignment = SequenceAssignment(torch.tensor([[0, -1]]))
    low = _predictions(
        torch.tensor([[[[5.0, -8.0], [-5.0, -8.0]], [[5.0, -8.0], [-5.0, -8.0]]]]),
        existence=torch.tensor([[[4.0, -8.0], [4.0, -8.0]]]),
    )
    high = _predictions(
        torch.tensor([[[[5.0, 8.0], [-5.0, 8.0]], [[5.0, 8.0], [-5.0, 8.0]]]]),
        existence=torch.tensor([[[4.0, 8.0], [4.0, 8.0]]]),
    )
    kwargs = dict(
        support_weight=1.0,
        existence_weight=1.0,
        task_weight=0.0,
        dense_task_weight=0.0,
    )
    low_terms = _normalized(sequence_set_terms(low, targets, assignment, **kwargs))
    high_terms = _normalized(sequence_set_terms(high, targets, assignment, **kwargs))
    assert high_terms["set/support"] > low_terms["set/support"]
    assert high_terms["set/existence"] > low_terms["set/existence"]


def test_exact_exhaustive_task_creates_unmatched_row_negatives() -> None:
    targets = _targets(
        torch.tensor([[[[1.0, 0.0]]]]),
        task=torch.tensor([[1.0]]),
        task_valid=torch.ones(1, 1, dtype=torch.bool),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
    )
    assignment = SequenceAssignment(torch.tensor([[0, -1]]))
    low = _predictions(
        torch.zeros(1, 1, 2, 2),
        task=torch.tensor([[6.0, -8.0]]),
    )
    high = _predictions(
        torch.zeros(1, 1, 2, 2),
        task=torch.tensor([[6.0, 8.0]]),
    )
    kwargs = dict(
        support_weight=0.0,
        existence_weight=0.0,
        task_weight=1.0,
        dense_task_weight=0.0,
    )

    low_term = {term.name: term for term in sequence_set_terms(low, targets, assignment, **kwargs)}[
        "set/task"
    ]
    high_term = {
        term.name: term for term in sequence_set_terms(high, targets, assignment, **kwargs)
    }["set/task"]

    assert low_term.valid.sum() == high_term.valid.sum() == 1
    assert high_term.normalized() > low_term.normalized()


def test_exact_task_risk_is_invariant_to_unmatched_row_capacity() -> None:
    targets = _targets(
        torch.tensor([[[[1.0, 0.0]]]]),
        task=torch.tensor([[1.0]]),
        task_valid=torch.ones(1, 1, dtype=torch.bool),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
    )
    kwargs = dict(
        support_weight=0.0,
        existence_weight=0.0,
        task_weight=1.0,
        dense_task_weight=0.0,
    )

    small = _predictions(
        torch.zeros(1, 1, 2, 2),
        task=torch.full((1, 2), -2.0),
    )
    large = _predictions(
        torch.zeros(1, 1, 2, 16),
        task=torch.full((1, 16), -2.0),
    )
    small_term = {
        term.name: term
        for term in sequence_set_terms(
            small,
            targets,
            SequenceAssignment(torch.tensor([[0, -1]])),
            **kwargs,
        )
    }["set/task"]
    large_term = {
        term.name: term
        for term in sequence_set_terms(
            large,
            targets,
            SequenceAssignment(torch.tensor([[0, *([-1] * 15)]])),
            **kwargs,
        )
    }["set/task"]

    torch.testing.assert_close(small_term.normalized(), large_term.normalized())


def test_exact_task_risk_rejects_the_unbalanced_capacity_base_rate() -> None:
    targets = _targets(
        torch.tensor([[[[1.0, 0.0]]]]),
        task=torch.tensor([[1.0]]),
        task_valid=torch.ones(1, 1, dtype=torch.bool),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
    )
    base_rate = torch.logit(torch.tensor(1.0 / 16.0))
    predictions = _predictions(
        torch.zeros(1, 1, 2, 16),
        task=torch.full((1, 16), base_rate),
    )
    term = {
        item.name: item
        for item in sequence_set_terms(
            predictions,
            targets,
            SequenceAssignment(torch.tensor([[0, *([-1] * 15)]])),
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=1.0,
            dense_task_weight=0.0,
        )
    }["set/task"]

    term.normalized().backward()

    assert predictions.task_relevance_logits.grad is not None
    assert predictions.task_relevance_logits.grad[0, 0] < 0
    assert predictions.task_relevance_logits.grad[0, 1:].sum() > 0
    assert predictions.task_relevance_logits.grad.sum() < 0


def test_balanced_task_risk_preserves_soft_confidence_targets() -> None:
    logits = torch.tensor([-0.7, 0.4, 9.0], requires_grad=True)
    targets = torch.tensor([0.8, 0.2, 1.0])
    valid = torch.tensor([True, True, False])

    actual = _balanced_task_score(logits, targets, valid)
    assert actual is not None
    positive = (targets[:2] * F.softplus(-logits[:2])).sum() / targets[:2].sum()
    negative = ((1 - targets[:2]) * F.softplus(logits[:2])).sum() / (1 - targets[:2]).sum()

    torch.testing.assert_close(actual, (positive + negative) / 2)
    actual.backward()
    assert logits.grad is not None
    assert logits.grad[2] == 0


def test_censored_exact_target_keeps_unmatched_task_rows_unknown() -> None:
    targets = _targets(
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        task=torch.tensor([[1.0, 0.0]]),
        task_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.tensor([[True, False]]),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
    )
    assignment = SequenceAssignment(torch.tensor([[1, -1]]))
    first = _predictions(
        torch.zeros(1, 1, 2, 2),
        task=torch.tensor([[-6.0, -20.0]]),
    )
    second = _predictions(
        torch.zeros(1, 1, 2, 2),
        task=torch.tensor([[-6.0, 20.0]]),
    )
    kwargs = dict(
        support_weight=0.0,
        existence_weight=0.0,
        task_weight=1.0,
        dense_task_weight=0.0,
    )

    first_term = {
        term.name: term for term in sequence_set_terms(first, targets, assignment, **kwargs)
    }["set/task"]
    second_term = {
        term.name: term for term in sequence_set_terms(second, targets, assignment, **kwargs)
    }["set/task"]

    assert first_term.valid.sum() == second_term.valid.sum() == 1
    torch.testing.assert_close(first_term.normalized(), second_term.normalized())


def test_capacity_censoring_contributes_neither_positive_nor_negative_evidence() -> None:
    masks = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]])
    targets = _targets(
        masks,
        capacity_censored=torch.tensor([[False, True]]),
        token_observed_fraction=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        inventory_exhaustive=torch.ones(1, 2, dtype=torch.bool),
    )
    assignment = SequenceAssignment(torch.tensor([[0, -1]]))
    first = _predictions(
        torch.tensor([[[[5.0, -20.0], [-5.0, -20.0]], [[5.0, -20.0], [-5.0, -20.0]]]]),
        existence=torch.tensor([[[4.0, -20.0], [4.0, -20.0]]]),
    )
    second = _predictions(
        torch.tensor([[[[5.0, 20.0], [-5.0, 20.0]], [[5.0, 20.0], [-5.0, 20.0]]]]),
        existence=torch.tensor([[[4.0, 20.0], [4.0, 20.0]]]),
    )
    kwargs = dict(
        support_weight=1.0,
        existence_weight=1.0,
        task_weight=0.0,
        dense_task_weight=0.0,
    )
    first_terms = _normalized(sequence_set_terms(first, targets, assignment, **kwargs))
    second_terms = _normalized(sequence_set_terms(second, targets, assignment, **kwargs))
    torch.testing.assert_close(first_terms["set/support"], second_terms["set/support"])
    torch.testing.assert_close(first_terms["set/existence"], second_terms["set/existence"])


def test_overlap_is_multilabel_only_and_exclusive_loss_does_not_double_count_support() -> None:
    overlap = torch.tensor([[[[0.8], [0.7]]]])
    _targets(overlap, exclusive=False)
    with pytest.raises(ValueError, match="cannot overlap"):
        _targets(overlap, exclusive=True)

    exclusive_masks = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    targets = _targets(
        exclusive_masks,
        exclusive=True,
        token_observed_fraction=torch.ones(1, 1, 2),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
    )
    predictions = _predictions(torch.tensor([[[[6.0, -6.0], [-6.0, 6.0]]]]))
    terms = sequence_set_terms(
        predictions,
        targets,
        SequenceAssignment(torch.tensor([[0, 1]])),
        support_weight=1.0,
        existence_weight=1.0,
        task_weight=1.0,
        dense_task_weight=0.0,
        ownership_weight=1.0,
    )
    by_name = {term.name: term for term in terms}
    assert int(by_name["set/support"].valid.sum()) == 0
    assert int(by_name["set/ownership"].valid.sum()) == 2
    assert int(by_name["set/ownership_nll"].valid.sum()) == 2


def test_exclusive_ownership_uses_observed_mass_and_unknown_tokens_have_zero_gradient() -> None:
    masks = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    targets = _targets(
        masks,
        mask_valid=torch.tensor([[[[True, True, False]]]]),
        token_observed_fraction=torch.tensor([[[0.25, 0.75, 0.0]]]),
        exclusive=True,
    )
    predictions = _predictions(torch.tensor([[[[2.0], [-2.0], [9.0]]]]))
    terms = {
        item.name: item
        for item in sequence_set_terms(
            predictions,
            targets,
            SequenceAssignment(torch.tensor([[0]])),
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=1.0,
        )
    }
    term = terms["set/ownership"]
    nll = terms["set/ownership_nll"]
    assert predictions.ownership_log_probability is not None
    expected_values = -torch.stack(
        (
            predictions.ownership_log_probability[0, 0, 0, 0],
            predictions.ownership_log_probability[0, 0, 1, 1],
        )
    )

    torch.testing.assert_close(term.values, expected_values)
    torch.testing.assert_close(term.sample_weight, torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(nll.values, expected_values)
    torch.testing.assert_close(nll.sample_weight, torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(
        nll.normalized(),
        (0.25 * expected_values[0] + 0.75 * expected_values[1]),
    )
    term.normalized().backward()
    assert predictions.support_logits.grad is not None
    torch.testing.assert_close(predictions.support_logits.grad[0, 0, 2], torch.zeros(1))


def test_exclusive_ownership_nll_keeps_gradient_below_old_probability_floor() -> None:
    predictions = _predictions(torch.tensor([[[[-20.0]]]]))
    targets = _targets(
        torch.ones(1, 1, 1, 1),
        token_observed_fraction=torch.ones(1, 1, 1),
        exclusive=True,
    )
    assert predictions.ownership[0, 0, 0, 0] < 1e-8

    ownership_term = {
        term.name: term
        for term in sequence_set_terms(
            predictions,
            targets,
            SequenceAssignment(torch.tensor([[0]])),
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=1.0,
        )
    }["set/ownership"]
    loss = ownership_term.normalized()

    torch.testing.assert_close(loss, torch.tensor(20.0), atol=1e-6, rtol=0.0)
    loss.backward()
    assert predictions.support_logits.grad is not None
    torch.testing.assert_close(
        predictions.support_logits.grad,
        torch.full_like(predictions.support_logits, -1.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_exclusive_ownership_keeps_observed_mass_and_loss_in_float32() -> None:
    targets = _targets(
        torch.tensor([[[[1.0]]]]),
        mask_valid=torch.tensor([[[[True]]]]),
        token_observed_fraction=torch.tensor([[[1.0 / 65536.0]]]),
        exclusive=True,
    )
    predictions = _predictions(torch.tensor([[[[0.25]]]], dtype=torch.bfloat16))
    terms = {
        item.name: item
        for item in sequence_set_terms(
            predictions,
            targets,
            SequenceAssignment(torch.tensor([[0]])),
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=1.0,
        )
    }
    term = terms["set/ownership"]
    nll = terms["set/ownership_nll"]

    assert term.values.dtype == torch.float32
    assert term.sample_weight is not None
    assert term.sample_weight.dtype == torch.float32
    assert nll.values.dtype == torch.float32
    assert nll.sample_weight is not None
    assert nll.sample_weight.dtype == torch.float32
    torch.testing.assert_close(nll.sample_weight, torch.tensor([1.0 / 65536.0]))
    assert torch.isfinite(term.normalized())
    assert torch.isfinite(nll.normalized())


def test_exclusive_ownership_builds_fractional_targets_before_bfloat16_compute() -> None:
    object_fraction = 0.2793108820915222
    targets = _targets(
        torch.tensor([[[[object_fraction]]]]),
        token_observed_fraction=torch.ones(1, 1, 1),
        exclusive=True,
    )
    predictions = _predictions(torch.tensor([[[[0.25]]]], dtype=torch.bfloat16))
    terms = {
        item.name: item
        for item in sequence_set_terms(
            predictions,
            targets,
            SequenceAssignment(torch.tensor([[0]])),
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=1.0,
        )
    }
    expected_target = torch.tensor([object_fraction, 1 - object_fraction])
    assert predictions.ownership_log_probability is not None
    expected_nll = -(expected_target * predictions.ownership_log_probability[0, 0, 0].float()).sum()

    assert terms["set/ownership"].values.dtype == torch.float32
    torch.testing.assert_close(terms["set/ownership"].values, expected_nll.reshape(1))
    torch.testing.assert_close(terms["set/ownership_nll"].values, expected_nll.reshape(1))


def test_exclusive_ownership_nll_matches_fractional_scalar_reference() -> None:
    logits = torch.tensor(
        [
            [[1.0, -0.5, 0.25], [-1.0, 2.0, 0.5], [5.0, -3.0, 0.0]],
            [[0.4, 0.8, -0.2], [-2.0, -1.0, 2.0], [1.5, 0.0, -0.5]],
        ],
        requires_grad=True,
    )
    log_probability = F.log_softmax(logits, dim=-1)
    expected = torch.tensor(
        [
            [[0.75, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.25, 0.5, 0.25], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ]
    )
    supervised = torch.tensor([[True, True, False], [True, True, True]])
    observed = torch.tensor([[0.25, 0.75, 0.0], [1.0, 0.5, 0.125]])
    nll, nll_weight = _exclusive_ownership_nll_samples(
        log_probability,
        expected,
        supervised,
        observed,
    )
    torch.testing.assert_close(
        nll,
        -(expected[supervised] * log_probability[supervised]).sum(dim=-1),
    )
    torch.testing.assert_close(nll_weight, observed[supervised])


def test_exclusive_ownership_nll_is_proper_for_fractional_targets() -> None:
    expected = torch.tensor([[[0.1, 0.9]]])
    supervised = torch.ones(1, 1, dtype=torch.bool)
    observed = torch.ones(1, 1)
    calibrated, _ = _exclusive_ownership_nll_samples(
        expected.log(),
        expected,
        supervised,
        observed,
    )
    uncalibrated, _ = _exclusive_ownership_nll_samples(
        torch.tensor([[[0.5, 0.5]]]).log(),
        expected,
        supervised,
        observed,
    )

    assert calibrated.item() < uncalibrated.item()


def test_exclusive_ownership_nll_is_object_row_permutation_invariant() -> None:
    generator = torch.Generator().manual_seed(415)
    logits = torch.randn(2, 5, 4, generator=generator)
    log_probability = F.log_softmax(logits, dim=-1)
    owners = torch.tensor([[0, 1, 2, 0, 3], [2, 1, 0, 3, 1]])
    expected = F.one_hot(owners, num_classes=4).float()
    supervised = torch.ones(2, 5, dtype=torch.bool)
    observed = torch.rand(2, 5, generator=generator).clamp_min(0.1)
    permutation = torch.tensor([2, 0, 1])
    ownership_permutation = torch.cat((permutation, torch.tensor([3])))

    actual = _exclusive_ownership_nll_samples(
        log_probability,
        expected,
        supervised,
        observed,
    )
    permuted = _exclusive_ownership_nll_samples(
        log_probability[..., ownership_permutation],
        expected[..., ownership_permutation],
        supervised,
        observed,
    )

    torch.testing.assert_close(actual[0], permuted[0])
    torch.testing.assert_close(actual[1], permuted[1])


def test_entity_conditional_ownership_is_proper_and_row_permutation_invariant() -> None:
    expected = torch.tensor(
        [
            [
                [0.45, 0.05, 0.50],
                [0.05, 0.30, 0.65],
                [0.05, 0.30, 0.65],
                [0.05, 0.30, 0.65],
            ]
        ]
    )
    supervised = torch.ones(1, 4, dtype=torch.bool)
    observed = torch.ones(1, 4)
    binding_valid = torch.ones(1, 2, dtype=torch.bool)
    calibrated = _entity_conditional_ownership_nll_samples(
        expected.log(),
        expected,
        supervised,
        observed,
        binding_valid,
    )
    spatially_inverted = _entity_conditional_ownership_nll_samples(
        expected.flip(1).log(),
        expected,
        supervised,
        observed,
        binding_valid,
    )

    assert calibrated.shape == (2,)
    assert calibrated.mean() < spatially_inverted.mean()

    permutation = torch.tensor([1, 0, 2])
    permuted = _entity_conditional_ownership_nll_samples(
        expected[..., permutation].log(),
        expected[..., permutation],
        supervised,
        observed,
        binding_valid[..., [1, 0]],
    )
    torch.testing.assert_close(permuted, calibrated[[1, 0]])


def test_entity_conditional_ownership_has_stationary_calibrated_optimum() -> None:
    expected = torch.tensor(
        [
            [
                [0.45, 0.05, 0.50],
                [0.05, 0.30, 0.65],
                [0.05, 0.30, 0.65],
                [0.05, 0.30, 0.65],
            ]
        ]
    )
    supervised = torch.ones(1, 4, dtype=torch.bool)
    observed = torch.ones(1, 4)
    binding_valid = torch.ones(1, 2, dtype=torch.bool)
    logits = expected.log().clone().requires_grad_()

    def score(value: torch.Tensor) -> torch.Tensor:
        return _entity_conditional_ownership_nll_samples(
            F.log_softmax(value, dim=-1),
            expected,
            supervised,
            observed,
            binding_valid,
        ).mean()

    calibrated = score(logits)
    calibrated.backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad), atol=1e-6, rtol=0.0)

    direction = torch.tensor(
        [
            [
                [1.0, -0.5, -0.5],
                [-0.5, 1.0, -0.5],
                [0.5, -1.0, 0.5],
                [-1.0, 0.5, 0.5],
            ]
        ]
    )
    epsilon = 0.05
    with torch.no_grad():
        assert score(logits + epsilon * direction) > calibrated
        assert score(logits - epsilon * direction) > calibrated


def test_entity_conditional_ownership_gradient_is_invariant_to_token_replication() -> None:
    def score_and_aggregate_gradient(repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
        expected = torch.zeros(1, 16 * repeats, 3)
        expected[:, :repeats, 0] = 1
        expected[:, repeats : 9 * repeats, 1] = 1
        expected[:, 9 * repeats :, 2] = 1
        logits = torch.zeros_like(expected, requires_grad=True)
        score = _entity_conditional_ownership_nll_samples(
            F.log_softmax(logits, dim=-1),
            expected,
            torch.ones(1, 16 * repeats, dtype=torch.bool),
            torch.ones(1, 16 * repeats),
            torch.ones(1, 2, dtype=torch.bool),
        )
        score.mean().backward()
        assert logits.grad is not None
        aggregate = logits.grad.reshape(1, 16, repeats, 3).sum(dim=2)
        return score.detach(), aggregate

    base_score, base_gradient = score_and_aggregate_gradient(1)
    repeated_score, repeated_gradient = score_and_aggregate_gradient(4)

    torch.testing.assert_close(
        repeated_score - base_score,
        torch.full_like(base_score, torch.log(torch.tensor(4.0))),
    )
    torch.testing.assert_close(repeated_gradient, base_gradient, atol=0.0, rtol=0.0)


def test_entity_conditional_ownership_excludes_prebinding_and_unknown_tokens() -> None:
    logits = torch.tensor(
        [
            [[0.2, 1.0, -0.5], [0.3, -0.2, 0.1], [2.0, -1.0, 0.0]],
            [[-0.5, 1.2, 0.0], [0.1, 0.7, -0.2], [9.0, -9.0, 0.0]],
        ],
        requires_grad=True,
    )
    expected = torch.tensor(
        [
            [[0.0, 0.6, 0.4], [0.0, 0.4, 0.6], [0.0, 0.5, 0.5]],
            [[0.0, 0.7, 0.3], [0.0, 0.3, 0.7], [0.0, 0.5, 0.5]],
        ]
    )
    supervised = torch.tensor([[True, True, True], [True, True, False]])
    binding_valid = torch.tensor([[True, False], [True, True]])
    values = _entity_conditional_ownership_nll_samples(
        F.log_softmax(logits, dim=-1),
        expected,
        supervised,
        torch.ones(2, 3),
        binding_valid,
    )

    assert values.shape == (1,)
    values.mean().backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[0], torch.zeros_like(logits.grad[0]))
    torch.testing.assert_close(logits.grad[1, 2], torch.zeros_like(logits.grad[1, 2]))
    assert logits.grad[1, :2].abs().sum() > 0


def test_entity_conditional_ownership_is_invariant_to_missing_modality_tokens() -> None:
    expected = torch.tensor([[[0.6, 0.1, 0.3], [0.1, 0.7, 0.2]]])
    base_logits = expected.log()
    base = _entity_conditional_ownership_nll_samples(
        base_logits,
        expected,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    extended_logits = torch.cat(
        (base_logits, torch.tensor([[[9.0, -9.0, 0.0]]])),
        dim=1,
    ).requires_grad_()
    extended_expected = torch.cat((expected, torch.zeros(1, 1, 3)), dim=1)
    extended = _entity_conditional_ownership_nll_samples(
        F.log_softmax(extended_logits, dim=-1),
        extended_expected,
        torch.tensor([[True, True, False]]),
        torch.tensor([[1.0, 1.0, 0.0]]),
        torch.ones(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(extended, base)
    extended.mean().backward()
    assert extended_logits.grad is not None
    torch.testing.assert_close(
        extended_logits.grad[:, -1],
        torch.zeros_like(extended_logits.grad[:, -1]),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_entity_conditional_ownership_is_finite_in_supported_compute_dtypes(
    dtype: torch.dtype,
) -> None:
    logits = torch.tensor(
        [[[0.5, -0.5, 0.0], [-0.25, 0.75, 0.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    expected = torch.tensor([[[0.6, 0.1, 0.3], [0.1, 0.7, 0.2]]])
    values = _entity_conditional_ownership_nll_samples(
        F.log_softmax(logits, dim=-1),
        expected,
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0.25, 1.0]]),
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert values.dtype == torch.float32
    assert values.shape == (2,)
    assert torch.isfinite(values).all()
    values.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_entity_conditional_ownership_has_no_fake_context_entity() -> None:
    values = _entity_conditional_ownership_nll_samples(
        F.log_softmax(torch.zeros(1, 3, 3), dim=-1),
        torch.tensor([[[0.0, 0.0, 1.0]] * 3]),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 3),
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert values.numel() == 0


def test_entity_conditional_estimator_conserves_family_weight_and_equalizes_entities() -> None:
    expected = torch.tensor(
        [
            [
                [0.45, 0.05, 0.50],
                [0.05, 0.30, 0.65],
                [0.05, 0.30, 0.65],
                [0.05, 0.30, 0.65],
            ]
        ]
    )
    masks = expected[0, :, :2].transpose(0, 1).reshape(1, 1, 2, 4)
    targets = _targets(
        masks,
        token_observed_fraction=torch.ones(1, 1, 4),
        exclusive=True,
    )
    support = (expected[..., :2] / expected[..., 2:]).log().unsqueeze(0)
    predictions = _predictions(support)
    terms = {
        term.name: term
        for term in sequence_set_terms(
            predictions,
            targets,
            SequenceAssignment(torch.tensor([[0, 1]])),
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=1.0,
            ownership_estimator=TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
        )
    }

    assert terms["set/ownership"].weight == pytest.approx(0.5)
    assert terms["set/ownership_entity"].weight == pytest.approx(0.5)
    assert terms["set/ownership_nll"].weight == 0.0
    assert int(terms["set/ownership_entity"].valid.sum()) == 2
    torch.testing.assert_close(
        terms["set/ownership_entity"].normalized(),
        terms["set/ownership_entity"].values.mean(),
    )
    assert sum(term.weight for term in terms.values()) == pytest.approx(1.0)


def test_entity_conditional_zero_visible_entity_retains_zero_gradient_graph() -> None:
    predictions = _predictions(torch.tensor([[[[0.5], [-0.5]]]]))
    targets = _targets(
        torch.zeros(1, 1, 1, 2),
        token_observed_fraction=torch.ones(1, 1, 2),
        exclusive=True,
    )
    terms = {
        term.name: term
        for term in sequence_set_terms(
            predictions,
            targets,
            SequenceAssignment(
                torch.tensor([[0]]),
                binding_start_phase=torch.tensor([[0]]),
            ),
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=1.0,
            ownership_estimator=TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
        )
    }
    entity = terms["set/ownership_entity"]

    assert not entity.valid.any()
    torch.testing.assert_close(entity.normalized(), torch.zeros(()))
    entity.normalized().backward()
    assert predictions.support_logits.grad is not None
    torch.testing.assert_close(
        predictions.support_logits.grad,
        torch.zeros_like(predictions.support_logits.grad),
    )


def test_explicit_micro_ownership_estimator_is_exact_legacy_parity() -> None:
    targets = _targets(
        torch.tensor([[[[0.75, 0.0, 0.25], [0.0, 1.0, 0.5]]]]),
        token_observed_fraction=torch.tensor([[[0.25, 1.0, 0.75]]]),
        exclusive=True,
    )
    assignment = SequenceAssignment(torch.tensor([[0, 1]]))
    first = _predictions(torch.tensor([[[[0.5, -0.5], [-1.0, 1.0], [0.2, 0.4]]]]))
    second = _predictions(torch.tensor([[[[0.5, -0.5], [-1.0, 1.0], [0.2, 0.4]]]]))
    kwargs = {
        "support_weight": 0.0,
        "existence_weight": 0.0,
        "task_weight": 0.0,
        "dense_task_weight": 0.0,
        "ownership_weight": 1.0,
    }
    legacy = sequence_set_terms(first, targets, assignment, **kwargs)
    explicit = sequence_set_terms(
        second,
        targets,
        assignment,
        ownership_estimator=TOKEN_MICRO_OWNERSHIP,
        **kwargs,
    )

    assert tuple(term.name for term in legacy) == tuple(term.name for term in explicit)
    for left, right in zip(legacy, explicit, strict=True):
        assert left.weight == right.weight
        assert torch.equal(left.values, right.values)
        assert torch.equal(left.valid, right.valid)
        if left.sample_weight is None:
            assert right.sample_weight is None
        else:
            assert right.sample_weight is not None
            assert torch.equal(left.sample_weight, right.sample_weight)
    legacy[-2].normalized().backward()
    explicit[-2].normalized().backward()
    assert first.support_logits.grad is not None and second.support_logits.grad is not None
    assert torch.equal(first.support_logits.grad, second.support_logits.grad)


def test_weighted_support_score_prioritizes_more_observed_evidence() -> None:
    logits = torch.tensor([6.0, -6.0])
    target = torch.ones(2)
    valid = torch.ones(2, dtype=torch.bool)

    mostly_correct = _support_score(logits, target, valid, torch.tensor([0.99, 0.01]))
    mostly_wrong = _support_score(logits, target, valid, torch.tensor([0.01, 0.99]))

    assert mostly_correct is not None
    assert mostly_wrong is not None
    assert mostly_correct < mostly_wrong


def test_target_tensors_cannot_be_trainable() -> None:
    with pytest.raises(ValueError, match="must not require gradients"):
        _targets(torch.ones(1, 1, 1, 1, requires_grad=True))


def test_vectorized_assignment_costs_match_scalar_reference() -> None:
    generator = torch.Generator().manual_seed(8301)
    batch, time, tokens, rows, tracks = 2, 3, 5, 4, 3
    predictions = _predictions(
        torch.randn(batch, time, tokens, rows, generator=generator),
        existence=torch.randn(batch, time, rows, generator=generator),
        task=torch.randn(batch, rows, generator=generator),
    )
    masks = torch.rand(batch, time, tracks, tokens, generator=generator)
    mask_valid = torch.rand(batch, time, tracks, tokens, generator=generator) > 0.4
    mask_valid[:, 0, :, 0] = True
    existence_valid = torch.rand(batch, time, tracks, generator=generator) > 0.45
    task_valid = torch.rand(batch, tracks, generator=generator) > 0.5
    targets = _targets(
        masks,
        mask_valid=mask_valid,
        existence=torch.rand(batch, time, tracks, generator=generator),
        existence_valid=existence_valid,
        task=torch.rand(batch, tracks, generator=generator),
        task_valid=task_valid,
    )

    for batch_index in range(batch):
        eligible = torch.arange(tracks)
        actual = _pairwise_assignment_costs(
            predictions,
            targets,
            batch_index=batch_index,
            eligible=eligible,
            causal_cut=time - 1,
        )
        expected = torch.empty_like(actual)
        for row_index in range(rows):
            for track_index in range(tracks):
                components = []
                support = _support_score(
                    predictions.support_logits[batch_index, :, :, row_index],
                    targets.masks[batch_index, :, track_index],
                    targets.mask_valid[batch_index, :, track_index],
                    torch.ones_like(targets.token_observed_fraction[batch_index]),
                )
                if support is not None:
                    components.append(support)
                valid = targets.existence_valid[batch_index, :, track_index]
                if valid.any():
                    components.append(
                        F.binary_cross_entropy_with_logits(
                            predictions.existence_logits[batch_index, :, row_index].float()[valid],
                            targets.existence[batch_index, :, track_index].float()[valid],
                        )
                    )
                expected[row_index, track_index] = torch.stack(components).mean()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_vectorized_exclusive_ownership_matches_scalar_reference() -> None:
    masks = torch.zeros(1, 3, 4, 5)
    owners = torch.tensor([[0, 1, 2, 3, -1], [2, 1, 0, -1, 3], [1, -1, 2, 0, 3]])
    for time_index in range(owners.shape[0]):
        for token_index in range(owners.shape[1]):
            owner = int(owners[time_index, token_index])
            if owner >= 0:
                masks[0, time_index, owner, token_index] = 1
    targets = _targets(
        masks,
        capacity_censored=torch.tensor([[False, False, False, True]]),
        token_observed_fraction=torch.ones(1, 3, 5),
        inventory_exhaustive=torch.ones(1, 3, dtype=torch.bool),
        exclusive=True,
    )
    predictions = _predictions(
        torch.randn(1, 3, 5, 3, generator=torch.Generator().manual_seed(992))
    )
    assignment = SequenceAssignment(torch.tensor([[2, -1, 0]]))
    terms = sequence_set_terms(
        predictions,
        targets,
        assignment,
        support_weight=0.0,
        existence_weight=0.0,
        task_weight=0.0,
        dense_task_weight=0.0,
        ownership_weight=1.0,
    )
    by_name = {term.name: term for term in terms}
    actual = by_name["set/ownership_nll"].values

    expected = []
    target_to_row = {2: 0, 0: 2}
    for time_index in range(3):
        for token_index in range(5):
            if owners[time_index, token_index] == 3:
                continue
            expected_distribution = predictions.ownership.new_zeros(4)
            for track_index, row_index in target_to_row.items():
                expected_distribution[row_index] = masks[0, time_index, track_index, token_index]
            if owners[time_index, token_index] == 1:
                continue
            expected_distribution[-1] = 1 - expected_distribution[:-1].sum()
            assert predictions.ownership_log_probability is not None
            log_probability = predictions.ownership_log_probability[0, time_index, token_index]
            expected.append(-(expected_distribution * log_probability).sum().reshape(1))
    expected_tensor = torch.cat(expected)
    torch.testing.assert_close(actual, expected_tensor)
    by_name["set/ownership"].normalized().backward()
    assert predictions.support_logits.grad is not None
    assert predictions.support_logits.grad.abs().sum() > 0


def test_dense_task_grounding_closes_exact_task_sensor_relation() -> None:
    masks = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]])
    targets = _targets(
        masks,
        task=torch.tensor([[1.0, 0.0]]),
        task_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    assignment = SequenceAssignment(torch.tensor([[0, 1]]))
    correct = _predictions(
        torch.zeros(1, 1, 4, 2),
        dense_task=torch.tensor([[[6.0, -6.0, -6.0, -6.0]]]),
    )
    inverted = _predictions(
        torch.zeros(1, 1, 4, 2),
        dense_task=torch.tensor([[[-6.0, 6.0, 6.0, 6.0]]]),
    )
    kwargs = dict(
        support_weight=0.0,
        existence_weight=0.0,
        task_weight=0.0,
        dense_task_weight=1.0,
    )

    correct_term = {
        term.name: term for term in sequence_set_terms(correct, targets, assignment, **kwargs)
    }["set/task_dense"]
    inverted_term = {
        term.name: term for term in sequence_set_terms(inverted, targets, assignment, **kwargs)
    }["set/task_dense"]

    assert correct_term.valid.sum() == 1
    assert correct_term.normalized() < inverted_term.normalized()
    correct_term.normalized().backward()
    assert correct.dense_task_grounding_logits.grad is not None
    assert correct.dense_task_grounding_logits.grad.abs().sum() > 0


def test_dense_task_grounding_ignores_inexact_task_and_unobserved_tokens() -> None:
    masks = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    inexact = _targets(
        masks,
        task=torch.tensor([[1.0]]),
        task_valid=torch.zeros(1, 1, dtype=torch.bool),
    )
    prediction = _predictions(
        torch.zeros(1, 1, 3, 1),
        dense_task=torch.tensor([[[3.0, -3.0, 50.0]]]),
    )
    kwargs = dict(
        support_weight=0.0,
        existence_weight=0.0,
        task_weight=0.0,
        dense_task_weight=1.0,
    )
    inexact_term = {
        term.name: term
        for term in sequence_set_terms(
            prediction,
            inexact,
            SequenceAssignment(torch.tensor([[0]])),
            **kwargs,
        )
    }["set/task_dense"]
    assert not inexact_term.valid.any()

    exact_exclusive = _targets(
        masks,
        mask_valid=torch.tensor([[[[True, True, False]]]]),
        task=torch.tensor([[1.0]]),
        task_valid=torch.ones(1, 1, dtype=torch.bool),
        token_observed_fraction=torch.tensor([[[0.25, 0.75, 0.0]]]),
        exclusive=True,
    )
    first = _predictions(
        torch.zeros(1, 1, 3, 1),
        dense_task=torch.tensor([[[3.0, -3.0, 50.0]]]),
    )
    second = _predictions(
        torch.zeros(1, 1, 3, 1),
        dense_task=torch.tensor([[[3.0, -3.0, -50.0]]]),
    )
    first_term = {
        term.name: term
        for term in sequence_set_terms(
            first,
            exact_exclusive,
            SequenceAssignment(torch.tensor([[0]])),
            **kwargs,
        )
    }["set/task_dense"]
    second_term = {
        term.name: term
        for term in sequence_set_terms(
            second,
            exact_exclusive,
            SequenceAssignment(torch.tensor([[0]])),
            **kwargs,
        )
    }["set/task_dense"]

    torch.testing.assert_close(first_term.normalized(), second_term.normalized())
    first_term.normalized().backward()
    assert first.dense_task_grounding_logits.grad is not None
    torch.testing.assert_close(
        first.dense_task_grounding_logits.grad[0, 0, 2],
        torch.zeros(()),
    )
