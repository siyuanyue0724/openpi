from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

from tools.smoke_molmoact2_m1_ddp import (  # noqa: E402
    _assert_nested_equal,
    _validate_processed_pair,
)


def _processed_pair_fixture(batch_size: int = 2) -> tuple[dict, dict]:
    observations = {
        "input_ids": torch.zeros((batch_size, 4), dtype=torch.int64),
        "attention_mask": torch.ones((batch_size, 4), dtype=torch.int64),
        "pixel_values": torch.zeros((batch_size, 3, 2, 2)),
        "image_grids": torch.zeros((batch_size, 2), dtype=torch.int64),
        "image_num_crops": torch.ones((batch_size,), dtype=torch.int64),
        "image_token_pooling": torch.zeros((batch_size, 2), dtype=torch.int64),
        "token_type_ids": torch.zeros((batch_size, 4), dtype=torch.int64),
        "observation.state": torch.zeros((batch_size, 8)),
    }
    targetful = {
        **observations,
        "action": torch.zeros((batch_size, 10, 32)),
        "action_dim_is_pad": torch.zeros((batch_size, 32), dtype=torch.bool),
        "action_horizon_is_pad": torch.zeros((batch_size, 10), dtype=torch.bool),
    }
    target_free = {
        **observations,
        # LeRobot canonicalizes the absent action into a present slot with a None value.
        "action": None,
        "action_dim_is_pad": torch.zeros((batch_size, 32), dtype=torch.bool),
        "action_horizon_is_pad": None,
    }
    return targetful, target_free


def test_m1_target_free_canonical_none_action_is_not_leakage() -> None:
    targetful, target_free = _processed_pair_fixture()
    _validate_processed_pair(
        targetful=targetful,
        target_free=target_free,
        batch_size=2,
    )


def test_m1_target_free_tensor_action_still_fails_closed() -> None:
    targetful, target_free = _processed_pair_fixture()
    target_free["action"] = torch.zeros((2, 10, 32))
    with pytest.raises(ValueError, match="crossed the target-free"):
        _validate_processed_pair(
            targetful=targetful,
            target_free=target_free,
            batch_size=2,
        )


def test_m1_checkpoint_state_comparison_is_exact_but_device_independent() -> None:
    expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
    _assert_nested_equal(expected, expected.clone())
    if torch.cuda.is_available():
        _assert_nested_equal(expected.cuda(), expected)

    with pytest.raises(AssertionError, match="tensor state differs"):
        _assert_nested_equal(expected, expected.to(dtype=torch.float64))
    with pytest.raises(AssertionError, match="tensor state differs"):
        _assert_nested_equal(expected, torch.tensor([1.0, 3.0]))
