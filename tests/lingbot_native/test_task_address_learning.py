from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.task_address_learning import (
    TASK_ADDRESS_SUPERVISION_DEPTH_SCHEMA,
    action_consumable_task_address,
    action_consumable_task_address_depth_contract,
    conditional_task_address_distribution,
    task_address_row_coverage,
    task_address_target_coverage,
)


def test_target_coverage_rewards_one_correct_read_without_collapsing_all_reads() -> None:
    mass = torch.tensor(
        [[[9.0, 1.0, 0.0], [1.0, 8.0, 1.0], [1.0, 1.0, 8.0]]],
        requires_grad=True,
    )
    result = task_address_target_coverage(mass, torch.tensor([1]))
    assert result.conditional_distribution.shape == (1, 3, 3)
    assert result.target_probability_per_read.shape == (1, 3)
    assert result.target_coverage.item() > 0.8
    result.loss.backward()
    assert mass.grad is not None
    assert torch.isfinite(mass.grad).all()
    assert mass.grad.abs().sum() > 0


def test_uniform_coverage_matches_the_noisy_or_probability() -> None:
    mass = torch.ones(1, 4, 16)
    result = task_address_target_coverage(mass, torch.tensor([7]))
    expected = 1.0 - (15.0 / 16.0) ** 4
    assert result.target_coverage.item() == pytest.approx(expected)
    assert result.loss.item() == pytest.approx(-torch.log(torch.tensor(expected)).item())


def test_row_coverage_scores_crossed_targets_from_one_distribution() -> None:
    distribution = conditional_task_address_distribution(
        torch.tensor([[[8.0, 1.0], [7.0, 1.0]]])
    )
    target = task_address_row_coverage(distribution, torch.tensor([0]))
    crossed = task_address_row_coverage(distribution, torch.tensor([1]))
    assert target.item() > crossed.item()


def test_distribution_rejects_missing_physical_mass() -> None:
    with pytest.raises(ValueError, match="no physical carrier mass"):
        conditional_task_address_distribution(torch.zeros(1, 2, 3))


def test_action_consumable_address_binds_penultimate_to_final_layer() -> None:
    row_mass = torch.full((1, 2, 3), 2.0)
    selected = action_consumable_task_address(row_mass, layer_count=4)
    assert selected.row_mass is row_mass
    assert selected.producer_layer_index == 2
    assert selected.consumer_layer_index == 3
    assert selected.layer_count == 4


def test_latest_action_consumable_address_requires_a_later_action_layer() -> None:
    with pytest.raises(ValueError, match="at least two host layers"):
        action_consumable_task_address(torch.ones(1, 2, 3), layer_count=1)


def test_action_consumable_address_rejects_invalid_receipt() -> None:
    with pytest.raises(ValueError, match="float"):
        action_consumable_task_address(
            torch.ones(1, 2, 3, dtype=torch.int64),
            layer_count=2,
        )


def test_action_consumable_depth_contract_seals_penultimate_to_final_order() -> None:
    assert action_consumable_task_address_depth_contract(36) == {
        "schema": TASK_ADDRESS_SUPERVISION_DEPTH_SCHEMA,
        "producer_layer_index": 34,
        "consumer_layer_index": 35,
        "layer_count": 36,
        "final_layer_excluded": True,
        "reason": "address-output-must-precede-a-later-action-attention-layer",
    }
