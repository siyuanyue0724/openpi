"""Shared CLI and audit contract for LingBot native training streams."""

from __future__ import annotations

import argparse
from typing import Any

ADR121_RESET_NUMERATOR = 1
ADR121_RESET_DENOMINATOR = 2


def add_reset_mixture_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the paired estimator arguments on one command-line tool."""

    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError("reset mixture arguments require an ArgumentParser")
    parser.add_argument("--reset-mixture-numerator", type=int)
    parser.add_argument("--reset-mixture-denominator", type=int)


def reset_mixture_values(args: Any) -> tuple[int, int] | None:
    """Validate the sole preregistered Arm-NR mixture or the causal baseline."""

    numerator = getattr(args, "reset_mixture_numerator", None)
    denominator = getattr(args, "reset_mixture_denominator", None)
    if (numerator is None) != (denominator is None):
        raise ValueError("reset mixture numerator and denominator must be provided together")
    if numerator is None or denominator is None:
        return None
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or (numerator, denominator) != (ADR121_RESET_NUMERATOR, ADR121_RESET_DENOMINATOR)
    ):
        raise ValueError("only the preregistered ADR-121 reset mixture 1/2 is supported")
    return numerator, denominator


def adr121_required_optimizer_lag(lane_interleave_factor: int) -> int:
    """Return the exact global update lag between K=8 causal lane visits."""

    if (
        isinstance(lane_interleave_factor, bool)
        or not isinstance(lane_interleave_factor, int)
        or lane_interleave_factor <= 0
    ):
        raise ValueError("lane interleave factor must be positive")
    return 2 * lane_interleave_factor


def adr121_recurrent_audit_updates(
    lane_interleave_factor: int,
) -> tuple[int, int]:
    """Return one-based first and second correction updates under reset-first alternation."""

    lag = adr121_required_optimizer_lag(lane_interleave_factor)
    return lag + 2, 2 * lag + 2


def validate_stream_optimizer_lag(
    *,
    reset_mixture: tuple[int, int] | None,
    lane_interleave_factor: int,
    maximum_optimizer_lag: int,
) -> None:
    """Fail closed when a cache or runner lag differs from its stream schedule."""

    required = adr121_required_optimizer_lag(lane_interleave_factor)
    if (
        isinstance(maximum_optimizer_lag, bool)
        or not isinstance(maximum_optimizer_lag, int)
        or maximum_optimizer_lag < 0
    ):
        raise ValueError("maximum optimizer lag must be a nonnegative integer")
    if reset_mixture is None:
        if lane_interleave_factor > maximum_optimizer_lag:
            raise ValueError("lane interleave factor exceeds maximum optimizer lag")
        return
    if reset_mixture != (ADR121_RESET_NUMERATOR, ADR121_RESET_DENOMINATOR):
        raise ValueError("only the preregistered ADR-121 reset mixture 1/2 is supported")
    if maximum_optimizer_lag != required:
        raise ValueError(
            "ADR-121 maximum optimizer lag must equal twice the lane interleave factor"
        )
