from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.controls import (
    ExecutedControlBatch,
    concatenate_executed_controls,
    executed_control_chain_reset,
)


def _controls() -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.tensor([[[1.0, 0.0], [2.0, 3.0]]]),
        field_valid=torch.tensor([[[True, False], [True, True]]]),
        token_valid=torch.tensor([[True, True]]),
        delta_time=torch.tensor([[0.1, 0.2]]),
        reset=torch.tensor([[False, False]]),
        acknowledged=torch.tensor([[True, True]]),
    )


def test_control_features_preserve_values_validity_time_and_reset() -> None:
    controls = _controls()
    features = controls.canonical_features()
    assert features.shape == (1, 2, 7)
    torch.testing.assert_close(features[..., :2], controls.values)
    torch.testing.assert_close(features[..., 2:4], controls.field_valid.float())
    torch.testing.assert_close(features[..., 4], controls.delta_time)
    torch.testing.assert_close(features[..., 5], torch.log1p(controls.delta_time))
    assert torch.equal(features[..., 6].bool(), controls.reset)


def test_control_bound_fails_closed_without_silent_truncation() -> None:
    controls = _controls()
    controls.validate_bound(2)
    with pytest.raises(ValueError, match="increase U_max"):
        controls.validate_bound(1)


def test_reset_is_an_explicit_valid_no_measurement_control_token() -> None:
    reset = ExecutedControlBatch.reset_only(
        batch_size=2,
        action_dim=3,
        device="cpu",
        dtype=torch.float32,
    )
    assert reset.token_valid.all()
    assert reset.reset.all()
    assert not reset.field_valid.any()
    assert not reset.values.any()


def test_control_chain_reset_reduces_every_chunk_without_losing_batch_identity() -> None:
    reset = ExecutedControlBatch.reset_only(
        batch_size=1,
        action_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    controls = _controls()

    assert executed_control_chain_reset((reset, controls)).tolist() == [True]
    assert executed_control_chain_reset((controls, controls)).tolist() == [False]

    with pytest.raises(ValueError, match="share batch"):
        executed_control_chain_reset(
            (
                controls,
                ExecutedControlBatch.reset_only(
                    batch_size=2,
                    action_dim=2,
                    device="cpu",
                    dtype=torch.float32,
                ),
            )
        )


def test_invalid_control_fields_cannot_hide_nonzero_data() -> None:
    controls = _controls()
    with pytest.raises(ValueError, match="exactly zero"):
        ExecutedControlBatch(
            values=controls.values + torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
            field_valid=controls.field_valid,
            token_valid=controls.token_valid,
            delta_time=controls.delta_time,
            reset=controls.reset,
            acknowledged=controls.acknowledged,
        )


def test_unacknowledged_controls_are_rejected_before_model_input() -> None:
    controls = _controls()
    with pytest.raises(ValueError, match="execution-acknowledged"):
        ExecutedControlBatch(
            values=controls.values,
            field_valid=controls.field_valid,
            token_valid=controls.token_valid,
            delta_time=controls.delta_time,
            reset=controls.reset,
            acknowledged=torch.tensor([[True, False]]),
        )


def test_concatenate_controls_preserves_every_typed_event_field() -> None:
    controls = _controls()
    first = ExecutedControlBatch(
        values=controls.values[:, :1],
        field_valid=controls.field_valid[:, :1],
        token_valid=controls.token_valid[:, :1],
        delta_time=controls.delta_time[:, :1],
        reset=controls.reset[:, :1],
        acknowledged=controls.acknowledged[:, :1],
    )
    second = ExecutedControlBatch(
        values=controls.values[:, 1:],
        field_valid=controls.field_valid[:, 1:],
        token_valid=controls.token_valid[:, 1:],
        delta_time=controls.delta_time[:, 1:],
        reset=controls.reset[:, 1:],
        acknowledged=controls.acknowledged[:, 1:],
    )

    joined = concatenate_executed_controls((first, second))

    for field in (
        "values",
        "field_valid",
        "token_valid",
        "delta_time",
        "reset",
        "acknowledged",
    ):
        assert torch.equal(getattr(joined, field), getattr(controls, field))


def test_concatenate_future_controls_rejects_reset_crossing() -> None:
    controls = _controls()
    reset = ExecutedControlBatch.reset_only(
        batch_size=1,
        action_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="cannot cross"):
        concatenate_executed_controls((controls, reset))
