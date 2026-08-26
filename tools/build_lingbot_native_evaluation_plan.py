#!/usr/bin/env python3
"""Freeze one preregistered G2-G6 paired hierarchical evaluation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from picf_next.lingbot_native.empirical_statistics import (
    EMPIRICAL_COMPARISON_RULES,
    EMPIRICAL_EVALUATION_PLAN_SCHEMA,
    EMPIRICAL_REQUIRED_ARMS,
    EMPIRICAL_REQUIRED_CHECKS,
    validate_empirical_metric_config,
)

try:
    from tools.run_lingbot_vla2_native_g0 import _write_text_durable
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from run_lingbot_vla2_native_g0 import _write_text_durable  # type: ignore[no-redef]


def _acceptance_bounds(values: list[str], *, gate: str) -> dict[str, float]:
    bounds: dict[str, float] = {}
    for value in values:
        name, separator, raw_bound = value.partition("=")
        if not separator or name in bounds or name not in EMPIRICAL_COMPARISON_RULES[gate]:
            raise ValueError("acceptance bounds must be unique registered NAME=VALUE entries")
        try:
            bound = float(raw_bound)
        except ValueError as error:
            raise ValueError(f"acceptance bound {name} is not numeric") from error
        if not math.isfinite(bound):
            raise ValueError(f"acceptance bound {name} must be finite")
        bounds[name] = bound
    if set(bounds) != set(EMPIRICAL_COMPARISON_RULES[gate]):
        missing = sorted(set(EMPIRICAL_COMPARISON_RULES[gate]) - set(bounds))
        raise ValueError(f"acceptance-bound coverage is incomplete: {missing}")
    return bounds


def _metric_config(values: list[str], *, gate: str) -> dict[str, object]:
    config: dict[str, object] = {}
    for value in values:
        name, separator, raw_value = value.partition("=")
        if not separator or name in config:
            raise ValueError("metric config must use unique NAME=VALUE entries")
        try:
            config[name] = float(raw_value)
        except ValueError as error:
            raise ValueError(f"metric config {name} is not numeric") from error
    return validate_empirical_metric_config(config, gate=gate)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=tuple(EMPIRICAL_COMPARISON_RULES), required=True)
    parser.add_argument(
        "--acceptance-bound",
        action="append",
        default=[],
        metavar="NAME=VALUE",
    )
    parser.add_argument(
        "--metric-config",
        action="append",
        default=[],
        metavar="NAME=VALUE",
    )
    parser.add_argument("--paired-seed-count", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.paired_seed_count < 5:
        raise ValueError("empirical gates require at least five paired seeds")
    if args.bootstrap_replicates < 1_000:
        raise ValueError("empirical gates require at least 1000 bootstrap replicates")
    if args.bootstrap_seed < 0:
        raise ValueError("bootstrap seed must be non-negative")
    design = {
        "arms": list(EMPIRICAL_REQUIRED_ARMS[args.gate]),
        "paired_seed_count": args.paired_seed_count,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "confidence_level": 0.95,
        "top_level_unit": "seed",
        "nested_units": ["task", "episode"],
        "aggregation": "equal_seed_task_episode_mean",
        "frames_treated_as_independent": False,
    }
    value = {
        "schema": EMPIRICAL_EVALUATION_PLAN_SCHEMA,
        "gate": args.gate,
        "design": design,
        "metric_config": _metric_config(args.metric_config, gate=args.gate),
        "acceptance_bounds": _acceptance_bounds(args.acceptance_bound, gate=args.gate),
        "required_checks": sorted(EMPIRICAL_REQUIRED_CHECKS[args.gate]),
    }
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    _write_text_durable(args.output, payload)
    print(
        json.dumps(
            {
                "gate": args.gate,
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
