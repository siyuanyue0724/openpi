from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from picf_next.lingbot_native.calvin_entity_set import physical_calvin_frame_targets
from picf_next.lingbot_native.calvin_supervision import NativeCALVINSequenceTargetBundle
from picf_next.lingbot_native.entity_set_objective import (
    PhysicalFramePredictions,
    PhysicalFrameTargets,
    match_physical_frame_entities,
    physical_frame_set_loss,
    sam31_dice_loss,
    sam31_sigmoid_focal_loss,
)
from picf_next.lingbot_native.supervision import NativeSequenceTargets


def _predictions(
    support_logits: torch.Tensor,
    *,
    existence_logits: torch.Tensor | None = None,
) -> PhysicalFramePredictions:
    batch, tokens, rows = support_logits.shape
    if existence_logits is None:
        existence_logits = torch.full(
            (batch, rows),
            2.0,
            dtype=support_logits.dtype,
            device=support_logits.device,
        )
    context = torch.zeros(
        batch,
        tokens,
        1,
        dtype=support_logits.dtype,
        device=support_logits.device,
    )
    ownership_log_probability = torch.log_softmax(
        torch.cat((support_logits, context), dim=-1),
        dim=-1,
    )
    return PhysicalFramePredictions(
        support_logits=support_logits,
        ownership_log_probability=ownership_log_probability,
        existence_logits=existence_logits,
        sensor_valid=torch.ones(batch, tokens, dtype=torch.bool),
    )


def _targets() -> PhysicalFrameTargets:
    masks = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    return PhysicalFrameTargets(
        masks=masks,
        mask_valid=torch.ones_like(masks, dtype=torch.bool),
        existence=torch.ones(1, 2),
        existence_valid=torch.ones(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 4),
        inventory_exhaustive=torch.ones(1, dtype=torch.bool),
        exclusive_ownership=True,
    )


def test_sam31_focal_formula_matches_released_nontriton_path() -> None:
    inputs = torch.tensor([[-2.0, -0.25, 0.5, 3.0]], dtype=torch.float64)
    targets = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float64)
    probability = inputs.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    target_probability = probability * targets + (1 - probability) * (1 - targets)
    expected = cross_entropy * ((1 - target_probability) ** 2)
    expected *= 0.25 * targets + 0.75 * (1 - targets)
    torch.testing.assert_close(sam31_sigmoid_focal_loss(inputs, targets), expected)


def test_sam31_dice_formula_matches_released_source() -> None:
    inputs = torch.tensor([[0.2, -1.0, 2.0, 0.0]], dtype=torch.float64)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    probability = inputs.sigmoid()
    expected = 1 - (2 * (probability * targets).sum(-1) + 1) / (
        probability.sum(-1) + targets.sum(-1) + 1
    )
    torch.testing.assert_close(sam31_dice_loss(inputs, targets), expected)


def test_hungarian_assignment_is_class_free_and_row_permutation_equivariant() -> None:
    logits = torch.tensor([[[-6.0, 6.0], [-6.0, 6.0], [6.0, -6.0], [6.0, -6.0]]])
    assignment = match_physical_frame_entities(_predictions(logits), _targets())
    assert assignment.row_to_track.tolist() == [[1, 0]]

    permutation = torch.tensor([1, 0])
    permuted = _predictions(logits.index_select(-1, permutation))
    permuted_assignment = match_physical_frame_entities(permuted, _targets())
    torch.testing.assert_close(
        permuted_assignment.row_to_track,
        assignment.row_to_track.index_select(-1, permutation),
    )


def test_hungarian_assignment_ignores_unknown_existence_evidence() -> None:
    predictions = _predictions(
        torch.tensor([[[5.0, 5.0], [5.0, 5.0]]]),
        existence_logits=torch.tensor([[-10.0, 10.0]]),
    )
    targets = PhysicalFrameTargets(
        masks=torch.ones(1, 1, 2),
        mask_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        existence=torch.ones(1, 1),
        existence_valid=torch.zeros(1, 1, dtype=torch.bool),
        track_valid=torch.ones(1, 1, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 1, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 2),
        inventory_exhaustive=torch.zeros(1, dtype=torch.bool),
        exclusive_ownership=True,
    )

    assignment = match_physical_frame_entities(predictions, targets)

    assert assignment.row_to_track.tolist() == [[0, -1]]


def test_absent_sequence_track_does_not_consume_a_current_frame_row() -> None:
    targets = PhysicalFrameTargets(
        masks=torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
        mask_valid=torch.ones(1, 2, 2, dtype=torch.bool),
        existence=torch.tensor([[1.0, 0.0]]),
        existence_valid=torch.ones(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 2),
        inventory_exhaustive=torch.ones(1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    predictions = _predictions(torch.tensor([[[5.0, -5.0], [5.0, -5.0]]]))

    assignment = match_physical_frame_entities(predictions, targets)

    assert sorted(assignment.row_to_track[0].tolist()) == [-1, 0]


def test_physical_set_loss_has_no_task_surface_and_backpropagates() -> None:
    logits = torch.tensor(
        [[[-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]],
        requires_grad=True,
    )
    existence = torch.zeros(1, 2, requires_grad=True)
    predictions = _predictions(logits, existence_logits=existence)
    loss = physical_frame_set_loss(predictions, _targets())
    assert torch.isfinite(loss.total)
    assert loss.assignment.row_to_track.tolist() == [[1, 0]]
    assert loss.total.item() > 0
    loss.total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert existence.grad is not None and (existence.grad != 0).all()
    assert not hasattr(loss, "task")


def test_unknown_inventory_does_not_create_false_negative_existence() -> None:
    logits = torch.tensor([[[4.0, -4.0], [4.0, -4.0]]], requires_grad=True)
    existence = torch.tensor([[2.0, 3.0]], requires_grad=True)
    predictions = _predictions(logits, existence_logits=existence)
    targets = PhysicalFrameTargets(
        masks=torch.tensor([[[1.0, 1.0]]]),
        mask_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        existence=torch.ones(1, 1),
        existence_valid=torch.ones(1, 1, dtype=torch.bool),
        track_valid=torch.ones(1, 1, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 1, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 2),
        inventory_exhaustive=torch.zeros(1, dtype=torch.bool),
        exclusive_ownership=False,
    )
    result = physical_frame_set_loss(
        predictions,
        targets,
        ownership_weight=0,
    )
    result.total.backward()
    unmatched = result.assignment.row_to_track[0] < 0
    assert unmatched.sum() == 1
    assert existence.grad is not None
    assert existence.grad[0, unmatched].item() == pytest.approx(0.0)


def test_ownership_nll_uses_the_same_observed_fraction_measure_as_masks() -> None:
    predictions = _predictions(torch.tensor([[[-2.0], [2.0]]]))
    targets = PhysicalFrameTargets(
        masks=torch.ones(1, 1, 2),
        mask_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        existence=torch.ones(1, 1),
        existence_valid=torch.ones(1, 1, dtype=torch.bool),
        track_valid=torch.ones(1, 1, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 1, dtype=torch.bool),
        token_observed_fraction=torch.tensor([[0.1, 1.0]]),
        inventory_exhaustive=torch.ones(1, dtype=torch.bool),
        exclusive_ownership=True,
    )

    result = physical_frame_set_loss(
        predictions,
        targets,
        mask_focal_weight=0,
        mask_dice_weight=0,
        existence_weight=0,
    )

    object_log_probability = predictions.ownership_log_probability[0, :, 0]
    expected = -(0.1 * object_log_probability[0] + object_log_probability[1]) / 1.1
    torch.testing.assert_close(result.ownership_nll, expected)


def test_physical_objective_is_invariant_to_surface_sampling_density() -> None:
    logits = torch.tensor([[[-2.0], [2.0]]])
    masks = torch.tensor([[[0.0, 1.0]]])

    def targets(values: torch.Tensor, measure: torch.Tensor | None) -> PhysicalFrameTargets:
        return PhysicalFrameTargets(
            masks=values,
            mask_valid=torch.ones_like(values, dtype=torch.bool),
            existence=torch.ones(1, 1),
            existence_valid=torch.ones(1, 1, dtype=torch.bool),
            track_valid=torch.ones(1, 1, dtype=torch.bool),
            capacity_censored=torch.zeros(1, 1, dtype=torch.bool),
            token_observed_fraction=torch.ones(1, values.shape[-1]),
            inventory_exhaustive=torch.ones(1, dtype=torch.bool),
            token_measure_weight=measure,
            exclusive_ownership=True,
        )

    base = physical_frame_set_loss(_predictions(logits), targets(masks, None))
    density = 7
    dense = physical_frame_set_loss(
        _predictions(logits.repeat_interleave(density, dim=1)),
        targets(
            masks.repeat_interleave(density, dim=-1),
            torch.full((1, 2 * density), 1.0 / density),
        ),
    )

    assert dense.assignment.row_to_track.tolist() == base.assignment.row_to_track.tolist()
    for field in ("mask_focal", "mask_dice", "existence_focal", "ownership_nll", "total"):
        torch.testing.assert_close(getattr(dense, field), getattr(base, field), rtol=0, atol=1e-6)


def test_unknown_inventory_never_labels_unannotated_tokens_as_context() -> None:
    predictions = _predictions(torch.tensor([[[2.0], [-2.0]]]))
    targets = PhysicalFrameTargets(
        masks=torch.tensor([[[1.0, 0.0]]]),
        mask_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        existence=torch.ones(1, 1),
        existence_valid=torch.ones(1, 1, dtype=torch.bool),
        track_valid=torch.ones(1, 1, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 1, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 2),
        inventory_exhaustive=torch.zeros(1, dtype=torch.bool),
        exclusive_ownership=True,
    )

    result = physical_frame_set_loss(
        predictions,
        targets,
        mask_focal_weight=0,
        mask_dice_weight=0,
        existence_weight=0,
    )

    expected = -predictions.ownership_log_probability[0, 0, 0]
    torch.testing.assert_close(result.ownership_nll, expected)


def test_calvin_physical_adapter_is_invariant_to_task_labels() -> None:
    masks = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]])

    def bundle(task_relevance: torch.Tensor) -> NativeCALVINSequenceTargetBundle:
        targets = NativeSequenceTargets(
            masks=masks,
            mask_valid=torch.ones_like(masks, dtype=torch.bool),
            existence=torch.ones(1, 2, 2),
            existence_valid=torch.ones(1, 2, 2, dtype=torch.bool),
            task_relevance=task_relevance,
            task_valid=torch.ones(1, 2, dtype=torch.bool),
            track_valid=torch.ones(1, 2, dtype=torch.bool),
            capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
            token_observed_fraction=torch.ones(1, 2, 3),
            inventory_exhaustive=torch.ones(1, 2, dtype=torch.bool),
            exclusive_ownership=True,
        )
        return NativeCALVINSequenceTargetBundle(
            targets=targets,
            identity_keys_by_batch=(("object-a", "object-b"),),
        )

    first = physical_calvin_frame_targets(
        bundle(torch.tensor([[1.0, 0.0]])),
        time_index=1,
    )
    second = physical_calvin_frame_targets(
        bundle(torch.tensor([[0.0, 1.0]])),
        time_index=1,
    )
    for name in (
        "masks",
        "mask_valid",
        "existence",
        "existence_valid",
        "track_valid",
        "capacity_censored",
        "token_observed_fraction",
        "inventory_exhaustive",
    ):
        assert torch.equal(getattr(first.targets, name), getattr(second.targets, name))
    assert first.targets.exclusive_ownership == second.targets.exclusive_ownership


def test_calvin_physical_adapter_does_not_match_unobserved_inventory() -> None:
    masks = torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])
    native = NativeSequenceTargets(
        masks=masks,
        mask_valid=torch.ones_like(masks, dtype=torch.bool),
        existence=torch.ones(1, 1, 2),
        existence_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        task_relevance=torch.zeros(1, 2),
        task_valid=torch.zeros(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 1, 3),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    converted = physical_calvin_frame_targets(
        NativeCALVINSequenceTargetBundle(
            targets=native,
            identity_keys_by_batch=(("visible", "off-camera"),),
        ),
        time_index=0,
    )

    assert converted.targets.existence.tolist() == [[1.0, 0.0]]
    assert converted.targets.existence_valid.tolist() == [[True, False]]
    assignment = match_physical_frame_entities(
        _predictions(torch.tensor([[[3.0, -3.0], [3.0, -3.0], [-3.0, -3.0]]])),
        converted.targets,
    )
    assert (assignment.row_to_track >= 0).sum().item() == 1
    assert 1 not in assignment.row_to_track[0].tolist()


def test_visible_ownership_rejects_double_owned_pixels_but_partial_masks_remain_legal() -> None:
    common = {
        "masks": torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]),
        "mask_valid": torch.ones(1, 2, 2, dtype=torch.bool),
        "existence": torch.ones(1, 2),
        "existence_valid": torch.ones(1, 2, dtype=torch.bool),
        "track_valid": torch.ones(1, 2, dtype=torch.bool),
        "capacity_censored": torch.zeros(1, 2, dtype=torch.bool),
        "token_observed_fraction": torch.ones(1, 2),
        "inventory_exhaustive": torch.ones(1, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="cannot overlap"):
        PhysicalFrameTargets(**common, exclusive_ownership=True)
    partial = PhysicalFrameTargets(**common, exclusive_ownership=False)
    assert partial.exclusive_ownership is False
