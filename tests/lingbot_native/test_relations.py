from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from picf_next.lingbot_native.relations import (
    HOST_NATIVE_MATCH_INTERFACE,
    SharedRelationReadout,
)


def test_shared_relation_head_has_only_declared_linear_output_parameters() -> None:
    head = SharedRelationReadout(8)
    assert tuple(name for name, _ in head.named_modules()) == (
        "",
        "projection",
        "match_projection",
        "existence_projection",
    )
    assert head.projection.bias is None
    assert head.projection.in_features == head.projection.out_features == 8
    assert head.match_projection.in_features == 8
    assert head.match_projection.out_features == 1
    assert head.temperature_parameter.shape == (1,)
    assert all(parameter.ndim > 0 for parameter in head.parameters())
    torch.testing.assert_close(head.temperature, torch.tensor([0.07]))


def test_relation_views_are_consistent_and_invalid_sensor_tokens_are_zero() -> None:
    torch.manual_seed(4)
    head = SharedRelationReadout(8)
    rows = torch.randn(2, 3, 8, requires_grad=True)
    sensors = torch.randn(2, 5, 8, requires_grad=True)
    valid = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])
    match = torch.randn(2, 3, 8, requires_grad=True)
    result = head(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=valid,
        match_hidden=match,
    )
    assert result.visible_support.shape == (2, 5, 3)
    assert result.ownership.shape == (2, 5, 4)
    assert result.task_relevance.shape == (2, 3)
    assert result.dense_task_grounding_logits.shape == (2, 5)
    assert result.existence.shape == (2, 3)
    assert result.task_object_log_probability is not None
    assert result.task_object_probability is not None
    assert result.task_event_distribution is not None
    assert result.task_row_probability is not None
    assert result.task_object_probability.shape == (2, 5, 3)
    assert result.task_event_distribution.shape == (2, 5, 4)
    assert result.task_row_probability.shape == (2, 3)
    assert not result.visible_support[~valid].any()
    assert not result.ownership[~valid].any()
    assert not result.dense_task_grounding[~valid].any()
    assert not result.task_object_probability[~valid].any()
    assert not result.task_event_distribution[~valid].any()
    valid_ownership = result.ownership[valid]
    torch.testing.assert_close(
        valid_ownership.sum(dim=-1),
        torch.ones(
            valid_ownership.shape[0],
            device=valid_ownership.device,
            dtype=valid_ownership.dtype,
        ),
    )
    projected_rows = head._project(rows)
    projected_sensors = head._project(sensors)
    temperature = head.temperature.to(rows.dtype)
    torch.testing.assert_close(
        result.support_logits,
        torch.einsum("bnd,bkd->bnk", projected_sensors, projected_rows) / temperature,
    )
    torch.testing.assert_close(
        result.task_relevance_logits,
        head.match_projection(match).squeeze(-1),
    )
    expected_row_probability = result.task_relevance_logits.float().sigmoid()
    expected_task_object = (
        expected_row_probability.unsqueeze(1) * result.ownership[..., :-1].float()
    )
    expected_log_probability = (
        expected_task_object.log()
        .masked_fill(
            ~valid.unsqueeze(-1),
            torch.finfo(result.task_object_log_probability.dtype).min,
        )
        .to(result.task_object_log_probability.dtype)
    )
    torch.testing.assert_close(
        result.task_object_log_probability,
        expected_log_probability,
    )
    torch.testing.assert_close(
        result.task_row_probability.float(),
        expected_row_probability,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        result.task_object_probability.float(),
        expected_task_object,
        atol=1e-6,
        rtol=1e-5,
    )
    conditioned_valid = result.task_event_distribution[valid]
    torch.testing.assert_close(
        conditioned_valid.sum(dim=-1),
        torch.ones(conditioned_valid.shape[0], dtype=conditioned_valid.dtype),
    )
    expected_dense = result.task_object_probability.sum(dim=-1)
    torch.testing.assert_close(
        result.dense_task_grounding,
        expected_dense * valid,
    )
    torch.testing.assert_close(
        result.persistent_anchor,
        result.task_object_probability,
    )
    torch.testing.assert_close(result.display_union, result.dense_task_grounding)
    torch.testing.assert_close(
        result.dense_task_grounding[valid],
        result.dense_task_grounding_logits[valid].sigmoid(),
        atol=1e-6,
        rtol=1e-5,
    )
    assert result.task_interface == HOST_NATIVE_MATCH_INTERFACE
    assert result.task_embedding is None
    assert result.match_embeddings is match
    loss = -result.task_object_probability[:, 0, 0].float().log().sum() + result.existence.sum()
    loss.backward()
    assert rows.grad is not None and rows.grad.abs().sum() > 0
    assert sensors.grad is not None and sensors.grad.abs().sum() > 0
    assert match.grad is not None and match.grad.abs().sum() > 0


def test_host_native_relation_is_row_permutation_equivariant() -> None:
    torch.manual_seed(17)
    head = SharedRelationReadout(8)
    rows = torch.randn(2, 3, 8)
    sensors = torch.randn(2, 5, 8)
    match = torch.randn(2, 3, 8)
    valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    permutation = torch.tensor([2, 0, 1])

    reference = head(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=valid,
        match_hidden=match,
    )
    permuted = head(
        posterior_rows=rows[:, permutation],
        sensor_hidden=sensors,
        sensor_valid=valid,
        match_hidden=match[:, permutation],
    )

    torch.testing.assert_close(
        permuted.support_logits,
        reference.support_logits[:, :, permutation],
    )
    torch.testing.assert_close(
        permuted.ownership[..., :-1],
        reference.ownership[..., :-1][:, :, permutation],
    )
    torch.testing.assert_close(
        permuted.ownership[..., -1],
        reference.ownership[..., -1],
    )
    torch.testing.assert_close(
        permuted.task_relevance_logits,
        reference.task_relevance_logits[:, permutation],
    )
    assert reference.task_object_log_probability is not None
    assert reference.task_object_probability is not None
    assert reference.task_event_distribution is not None
    assert reference.task_row_probability is not None
    assert permuted.task_object_log_probability is not None
    assert permuted.task_object_probability is not None
    assert permuted.task_event_distribution is not None
    assert permuted.task_row_probability is not None
    torch.testing.assert_close(
        permuted.task_object_log_probability,
        reference.task_object_log_probability[:, :, permutation],
    )
    torch.testing.assert_close(
        permuted.task_object_probability,
        reference.task_object_probability[:, :, permutation],
    )
    torch.testing.assert_close(
        permuted.task_event_distribution,
        torch.cat(
            (
                reference.task_event_distribution[..., :-1][:, :, permutation],
                reference.task_event_distribution[..., -1:],
            ),
            dim=-1,
        ),
    )
    torch.testing.assert_close(
        permuted.task_row_probability,
        reference.task_row_probability[:, permutation],
    )
    torch.testing.assert_close(
        permuted.existence_logits,
        reference.existence_logits[:, permutation],
    )
    torch.testing.assert_close(
        permuted.dense_task_grounding,
        reference.dense_task_grounding,
    )


def test_host_native_task_score_persists_exact_fp32_recomputation() -> None:
    torch.manual_seed(23)
    head = SharedRelationReadout(8).to(dtype=torch.bfloat16)
    rows = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    sensors = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    match = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    valid = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])

    result = head(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=valid,
        match_hidden=match,
    )
    expected = F.linear(
        match.float(),
        head.match_projection.weight.float(),
        head.match_projection.bias.float(),
    ).squeeze(-1)

    assert result.task_relevance_logits_fp32 is not None
    assert result.task_relevance_logits_fp32.dtype == torch.float32
    assert result.task_relevance_logits_fp32.requires_grad is False
    assert result.task_object_log_probability is not None
    assert result.task_object_probability is not None
    assert result.task_event_distribution is not None
    assert torch.isfinite(result.task_object_log_probability).all()
    assert not result.task_object_probability[~valid].any()
    torch.testing.assert_close(
        result.task_event_distribution[valid].float().sum(dim=-1),
        torch.ones(int(valid.sum())),
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        result.task_relevance_logits_fp32,
        expected,
        atol=0,
        rtol=0,
    )


def test_task_intervention_changes_task_object_relation_but_not_physical_ownership() -> None:
    torch.manual_seed(29)
    head = SharedRelationReadout(4)
    with torch.no_grad():
        head.match_projection.weight.zero_()
        head.match_projection.weight[0, 0] = 1.0
        head.match_projection.bias.zero_()
    rows = torch.randn(1, 2, 4)
    sensors = torch.randn(1, 3, 4)
    valid = torch.ones(1, 3, dtype=torch.bool)
    first_match = torch.zeros(1, 2, 4)
    second_match = first_match.clone()
    second_match[0, 0, 0] = 5.0
    second_match[0, 1, 0] = -5.0

    first = head(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=valid,
        match_hidden=first_match,
    )
    second = head(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=valid,
        match_hidden=second_match,
    )

    torch.testing.assert_close(first.ownership, second.ownership)
    torch.testing.assert_close(first.support_logits, second.support_logits)
    assert first.task_object_probability is not None
    assert second.task_object_probability is not None
    assert first.task_row_probability is not None
    assert second.task_row_probability is not None
    assert not torch.allclose(first.task_object_probability, second.task_object_probability)
    assert second.task_row_probability[0, 0] > first.task_row_probability[0, 0]
    assert second.task_row_probability[0, 1] < first.task_row_probability[0, 1]


def test_independent_task_rows_do_not_steal_probability_mass() -> None:
    torch.manual_seed(30)
    head = SharedRelationReadout(4)
    with torch.no_grad():
        head.match_projection.weight.zero_()
        head.match_projection.weight[0, 0] = 1.0
        head.match_projection.bias.zero_()
    rows = torch.randn(1, 2, 4)
    sensors = torch.randn(1, 3, 4)
    first_match = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [-8.0, 0.0, 0.0, 0.0]]])
    second_match = first_match.clone()
    second_match[0, 1, 0] = 8.0

    first = head(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=torch.ones(1, 3, dtype=torch.bool),
        match_hidden=first_match,
    )
    second = head(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=torch.ones(1, 3, dtype=torch.bool),
        match_hidden=second_match,
    )

    assert first.task_row_probability is not None
    assert second.task_row_probability is not None
    assert first.task_object_probability is not None
    assert second.task_object_probability is not None
    torch.testing.assert_close(first.task_row_probability[:, 0], second.task_row_probability[:, 0])
    torch.testing.assert_close(
        first.task_object_probability[..., 0],
        second.task_object_probability[..., 0],
    )
    assert second.task_row_probability[0, 1] > first.task_row_probability[0, 1]


def test_unlabelled_or_new_modality_tokens_cannot_renormalize_existing_anchors() -> None:
    torch.manual_seed(31)
    head = SharedRelationReadout(4)
    rows = torch.randn(1, 2, 4)
    sensors = torch.randn(1, 2, 4)
    match = torch.randn(1, 2, 4)
    reference = head(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=torch.ones(1, 2, dtype=torch.bool),
        match_hidden=match,
    )
    extended = head(
        posterior_rows=rows,
        sensor_hidden=torch.cat((sensors, torch.full((1, 1, 4), 100.0)), dim=1),
        sensor_valid=torch.ones(1, 3, dtype=torch.bool),
        structural_sensor_valid=torch.tensor([[True, True, False]]),
        match_hidden=match,
    )

    assert reference.task_object_probability is not None
    assert extended.task_object_probability is not None
    torch.testing.assert_close(
        extended.task_object_probability[:, :2],
        reference.task_object_probability,
    )
    torch.testing.assert_close(extended.ownership[:, :2], reference.ownership)


def test_relation_rejects_sample_without_any_valid_sensor_token() -> None:
    head = SharedRelationReadout(4)
    with torch.no_grad(), pytest.raises(ValueError, match="at least one valid sensor token"):
        head(
            posterior_rows=torch.randn(1, 2, 4),
            sensor_hidden=torch.randn(1, 3, 4),
            sensor_valid=torch.zeros(1, 3, dtype=torch.bool),
            match_hidden=torch.randn(1, 2, 4),
        )
