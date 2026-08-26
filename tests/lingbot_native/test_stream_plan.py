from __future__ import annotations

import argparse

import pytest

from picf_next.lingbot_native.stream_plan import (
    add_reset_mixture_arguments,
    adr121_recurrent_audit_updates,
    adr121_required_optimizer_lag,
    reset_mixture_values,
    validate_stream_optimizer_lag,
)


def _parse(*values: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_reset_mixture_arguments(parser)
    return parser.parse_args(values)


def test_reset_mixture_cli_is_paired_and_restricted_to_preregistered_half() -> None:
    assert reset_mixture_values(_parse()) is None
    assert reset_mixture_values(
        _parse("--reset-mixture-numerator", "1", "--reset-mixture-denominator", "2")
    ) == (1, 2)
    with pytest.raises(ValueError, match="provided together"):
        reset_mixture_values(_parse("--reset-mixture-numerator", "1"))
    with pytest.raises(ValueError, match="ADR-121"):
        reset_mixture_values(
            _parse("--reset-mixture-numerator", "1", "--reset-mixture-denominator", "3")
        )


def test_reset_first_alternation_moves_k8_corrections_and_lag_exactly() -> None:
    assert adr121_required_optimizer_lag(8) == 16
    assert adr121_recurrent_audit_updates(8) == (18, 34)
    with pytest.raises(ValueError, match="positive"):
        adr121_required_optimizer_lag(0)


def test_stream_optimizer_lag_matches_the_selected_estimator_schedule() -> None:
    validate_stream_optimizer_lag(
        reset_mixture=None,
        lane_interleave_factor=8,
        maximum_optimizer_lag=8,
    )
    validate_stream_optimizer_lag(
        reset_mixture=(1, 2),
        lane_interleave_factor=8,
        maximum_optimizer_lag=16,
    )
    with pytest.raises(ValueError, match="exceeds maximum optimizer lag"):
        validate_stream_optimizer_lag(
            reset_mixture=None,
            lane_interleave_factor=8,
            maximum_optimizer_lag=7,
        )
    with pytest.raises(ValueError, match="twice the lane interleave factor"):
        validate_stream_optimizer_lag(
            reset_mixture=(1, 2),
            lane_interleave_factor=8,
            maximum_optimizer_lag=8,
        )
