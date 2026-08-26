from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.task_address_receipt import (
    TaskAddressAttentionLayout,
    task_address_attention_receipt,
)


def _surface() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.zeros(1, 8, 2, 2)
    key = torch.zeros(1, 10, 1, 2)
    mask = torch.ones(1, 8, 10, dtype=torch.bool)
    query[:, 6] = torch.tensor([1.0, 0.0])
    query[:, 7] = torch.tensor([0.0, 1.0])
    # Memory rows 0/1, PRIOR rows 4/5 after the two prepended memory keys,
    # and POSTERIOR rows 6/7 after the memory offset all carry the same row key.
    key[:, (0, 4, 6), 0] = torch.tensor([1.0, 0.0])
    key[:, (1, 5, 7), 0] = torch.tensor([0.0, 1.0])
    return query, key, mask


def test_receipt_aggregates_all_three_physical_carriers() -> None:
    query, key, mask = _surface()
    receipt = task_address_attention_receipt(
        query_states=query,
        key_states=key,
        attention_mask=mask,
        object_read_slice=slice(6, 8),
        prior_slice=slice(2, 4),
        posterior_slice=slice(4, 6),
        capacity=2,
    )
    assert receipt.row_mass.shape == (1, 2, 2)
    assert receipt.carrier_mass.shape == (1, 2, 2, 3)
    assert receipt.row_mass[0, 0, 0] > receipt.row_mass[0, 0, 1]
    assert receipt.row_mass[0, 1, 1] > receipt.row_mass[0, 1, 0]
    assert torch.allclose(receipt.row_mass, receipt.carrier_mass.sum(dim=-1))
    assert torch.allclose(receipt.visible_mass, receipt.row_mass.sum(dim=-1))


def test_receipt_respects_the_executed_boolean_mask() -> None:
    query, key, mask = _surface()
    mask[:, 6, (0, 4, 6)] = False
    receipt = task_address_attention_receipt(
        query_states=query,
        key_states=key,
        attention_mask=mask,
        object_read_slice=slice(6, 8),
        prior_slice=slice(2, 4),
        posterior_slice=slice(4, 6),
        capacity=2,
    )
    assert torch.equal(receipt.carrier_mass[0, 0, 0], torch.zeros(3))


def test_receipt_rejects_an_unpaired_memory_bank() -> None:
    query, key, mask = _surface()
    with pytest.raises(ValueError, match="memory bank"):
        task_address_attention_receipt(
            query_states=query,
            key_states=key[:, 1:],
            attention_mask=mask[:, :, 1:],
            object_read_slice=slice(6, 8),
            prior_slice=slice(2, 4),
            posterior_slice=slice(4, 6),
            capacity=2,
        )


def test_attention_layout_rejects_a_slice_outside_the_executed_query() -> None:
    with pytest.raises(ValueError, match="object_read_slice"):
        TaskAddressAttentionLayout(
            batch_size=1,
            query_count=8,
            capacity=2,
            object_read_slice=slice(7, 9),
            prior_slice=slice(2, 4),
            posterior_slice=slice(4, 6),
        )
