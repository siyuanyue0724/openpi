from __future__ import annotations

import torch

from picf_next.lingbot_native.entity_set_evaluation import evaluate_physical_entity_frame
from picf_next.lingbot_native.entity_set_objective import (
    PhysicalFramePredictions,
    PhysicalFrameTargets,
    physical_frame_set_loss,
)
from picf_next.lingbot_native.physical_sequence import (
    extend_physical_sequence_row_bindings,
    match_physical_sequence_entities,
    physical_frame_assignment_at_time,
)


def _prediction(
    logits: torch.Tensor,
    *,
    existence: torch.Tensor | None = None,
) -> PhysicalFramePredictions:
    batch, tokens, rows = logits.shape
    if existence is None:
        existence = torch.zeros(batch, rows, dtype=logits.dtype, device=logits.device)
    context = torch.zeros(batch, tokens, 1, dtype=logits.dtype, device=logits.device)
    return PhysicalFramePredictions(
        support_logits=logits,
        ownership_log_probability=torch.log_softmax(torch.cat((logits, context), dim=-1), dim=-1),
        existence_logits=existence,
        sensor_valid=torch.ones(batch, tokens, dtype=torch.bool, device=logits.device),
    )


def _target(masks: torch.Tensor) -> PhysicalFrameTargets:
    batch, tracks, tokens = masks.shape
    visible = masks.sum(dim=-1) > 0
    return PhysicalFrameTargets(
        masks=masks,
        mask_valid=torch.ones_like(masks, dtype=torch.bool),
        existence=visible.to(masks.dtype),
        existence_valid=visible,
        track_valid=torch.ones(batch, tracks, dtype=torch.bool),
        capacity_censored=torch.zeros(batch, tracks, dtype=torch.bool),
        token_observed_fraction=torch.ones(batch, tokens),
        inventory_exhaustive=torch.ones(batch, dtype=torch.bool),
        exclusive_ownership=True,
    )


def test_physical_births_use_first_evidence_then_keep_one_episode_row() -> None:
    targets = (
        _target(torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])),
        _target(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
    )
    predictions = (
        _prediction(torch.tensor([[[6.0, -6.0], [-6.0, -6.0]]])),
        _prediction(torch.tensor([[[6.0, -6.0], [-6.0, 6.0]]])),
    )

    assignment = match_physical_sequence_entities(
        predictions,
        targets,
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=((),),
    )

    assert assignment.row_to_track.tolist() == [[0, 1]]
    assert assignment.binding_start_phase.tolist() == [[1, 3]]
    assert physical_frame_assignment_at_time(assignment, time_index=0).row_to_track.tolist() == [
        [0, -1]
    ]
    assert physical_frame_assignment_at_time(assignment, time_index=1).row_to_track.tolist() == [
        [0, 1]
    ]
    assert extend_physical_sequence_row_bindings(
        assignment,
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=((),),
    ) == ((("object/a", 0),),)
    assert extend_physical_sequence_row_bindings(
        assignment,
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=((),),
        commit_time_index=1,
    ) == ((("object/a", 0), ("object/b", 1)),)


def test_future_preference_cannot_rebind_an_identity_born_in_the_past() -> None:
    targets = (
        _target(torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])),
        _target(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
    )
    first = _prediction(torch.tensor([[[7.0, -7.0], [-7.0, -7.0]]]))
    later_prefers_swap = _prediction(torch.tensor([[[-9.0, 9.0], [9.0, -9.0]]]))
    later_prefers_stay = _prediction(torch.tensor([[[9.0, -9.0], [-9.0, 9.0]]]))

    swapped = match_physical_sequence_entities(
        (first, later_prefers_swap),
        targets,
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=((),),
    )
    stayed = match_physical_sequence_entities(
        (first, later_prefers_stay),
        targets,
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=((),),
    )

    assert swapped.row_to_track[0, 0].item() == 0
    assert stayed.row_to_track[0, 0].item() == 0
    assert swapped.binding_start_phase[0, 0].item() == 1
    assert stayed.binding_start_phase[0, 0].item() == 1


def test_prior_identity_survives_complete_occlusion_without_false_death() -> None:
    existence = torch.tensor([[0.4, -0.2]], requires_grad=True)
    prediction = _prediction(
        torch.tensor([[[-1.0, -1.0], [-1.0, -1.0]]], requires_grad=True),
        existence=existence,
    )
    target = _target(torch.zeros(1, 1, 2))
    assignment = match_physical_sequence_entities(
        (prediction,),
        (target,),
        identity_keys_by_batch=(("object/a",),),
        prior_bindings_by_batch=((("object/a", 1),),),
    )
    frame_assignment = physical_frame_assignment_at_time(assignment, time_index=0)

    assert frame_assignment.row_to_track.tolist() == [[-1, 0]]
    assert frame_assignment.carried.tolist() == [[False, True]]
    evidence = evaluate_physical_entity_frame(
        prediction,
        target,
        frame_assignment,
        identity_keys=("object/a",),
    )
    assert evidence["target_evidence_count"] == 0
    assert evidence["matched_evidence_count"] == 0
    assert evidence["matched_assignment_count"] == 1
    assert evidence["carried_unknown_count"] == 1
    assert evidence["rows"] == []
    result = physical_frame_set_loss(prediction, target, assignment=frame_assignment)
    result.total.backward()
    assert existence.grad is not None
    assert existence.grad[0, 0].item() != 0
    assert existence.grad[0, 1].item() == 0


def test_prior_identity_absent_from_local_axis_reserves_its_row() -> None:
    existence = torch.tensor([[0.1, 0.2]], requires_grad=True)
    prediction = _prediction(
        torch.tensor([[[5.0, -5.0], [5.0, -5.0]]], requires_grad=True),
        existence=existence,
    )
    target = _target(torch.tensor([[[1.0, 1.0]]]))
    assignment = match_physical_sequence_entities(
        (prediction,),
        (target,),
        identity_keys_by_batch=(("object/b",),),
        prior_bindings_by_batch=((("object/a", 1),),),
    )
    frame_assignment = physical_frame_assignment_at_time(assignment, time_index=0)

    assert frame_assignment.row_to_track.tolist() == [[0, -1]]
    assert frame_assignment.reserved.tolist() == [[False, True]]
    assert frame_assignment.carried.tolist() == [[False, False]]
    result = physical_frame_set_loss(prediction, target, assignment=frame_assignment)
    result.total.backward()
    assert existence.grad is not None
    assert existence.grad[0, 0].item() != 0
    assert existence.grad[0, 1].item() == 0
    assert extend_physical_sequence_row_bindings(
        assignment,
        identity_keys_by_batch=(("object/b",),),
        prior_bindings_by_batch=((("object/a", 1),),),
    ) == ((("object/a", 1), ("object/b", 0)),)


def test_state_and_gauge_row_permutation_is_equivariant() -> None:
    target = _target(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    logits = torch.tensor([[[6.0, -6.0], [-6.0, 6.0]]])
    original_prediction = _prediction(logits)
    original = match_physical_sequence_entities(
        (original_prediction,),
        (target,),
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=((("object/a", 0),),),
    )

    permutation = torch.tensor([1, 0])
    permuted_prediction = _prediction(logits.index_select(-1, permutation))
    permuted = match_physical_sequence_entities(
        (permuted_prediction,),
        (target,),
        identity_keys_by_batch=(("object/a", "object/b"),),
        prior_bindings_by_batch=((("object/a", 1),),),
    )

    torch.testing.assert_close(
        permuted.row_to_track,
        original.row_to_track.index_select(-1, permutation),
    )
    torch.testing.assert_close(
        permuted.binding_start_phase,
        original.binding_start_phase.index_select(-1, permutation),
    )
    original_loss = physical_frame_set_loss(
        original_prediction,
        target,
        assignment=physical_frame_assignment_at_time(original, time_index=0),
    )
    permuted_loss = physical_frame_set_loss(
        permuted_prediction,
        target,
        assignment=physical_frame_assignment_at_time(permuted, time_index=0),
    )
    torch.testing.assert_close(permuted_loss.total, original_loss.total)
