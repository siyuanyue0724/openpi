#!/usr/bin/env python3
# ruff: noqa: E402
"""Build an owner-reviewed LingBot predictive-objective decision artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from tools.repository_import import bind_entrypoint_to_own_repository
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot predictive objective decision builder",
)

from picf_next.lingbot_native.predictive_decision import (
    PREDICTIVE_OBJECTIVE_CLAIMS,
    PREDICTIVE_OBJECTIVE_DECISION_SCHEMA,
    PREDICTIVE_VISIBLE_SUPPORT_WEIGHTINGS,
    validate_predictive_objective_decision,
)

try:
    from tools.run_lingbot_vla2_native_g0 import _write_text_durable
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from run_lingbot_vla2_native_g0 import _write_text_durable  # type: ignore[no-redef]


def build_predictive_objective_decision(
    *,
    reviewer: str,
    temporal_objective: str,
    visible_support_weighting: str,
    minimum_visible_fraction: float,
    decision_record: Path,
) -> dict[str, Any]:
    """Bind explicit semantics to the exact owner-reviewed ADR content."""

    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("predictive objective decision requires an explicit reviewer")
    if temporal_objective not in PREDICTIVE_OBJECTIVE_CLAIMS:
        raise ValueError("predictive temporal objective is outside the reviewed alternatives")
    if visible_support_weighting not in PREDICTIVE_VISIBLE_SUPPORT_WEIGHTINGS:
        raise ValueError("predictive visible-support weighting is outside reviewed alternatives")
    if (
        isinstance(minimum_visible_fraction, bool)
        or not isinstance(minimum_visible_fraction, (int, float))
        or not math.isfinite(minimum_visible_fraction)
        or not 0 <= minimum_visible_fraction < 1
    ):
        raise ValueError("minimum visible fraction must lie in [0,1)")
    if decision_record.is_symlink() or not decision_record.is_file():
        raise ValueError("predictive decision record must be one real file")
    payload = decision_record.read_bytes()
    if not payload:
        raise ValueError("predictive decision record cannot be empty")
    value = {
        "schema": PREDICTIVE_OBJECTIVE_DECISION_SCHEMA,
        "status": "PASS",
        "reviewer": reviewer.strip(),
        "temporal_objective": temporal_objective,
        "claim_scope": PREDICTIVE_OBJECTIVE_CLAIMS[temporal_objective],
        "visible_support": {
            "weighting": visible_support_weighting,
            "minimum_visible_fraction_hex": float(minimum_visible_fraction).hex(),
        },
        "decision_record": {
            "path": str(decision_record.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    validate_predictive_objective_decision(value)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--temporal-objective",
        choices=tuple(PREDICTIVE_OBJECTIVE_CLAIMS),
        required=True,
    )
    parser.add_argument(
        "--visible-support-weighting",
        choices=tuple(sorted(PREDICTIVE_VISIBLE_SUPPORT_WEIGHTINGS)),
        required=True,
    )
    parser.add_argument("--minimum-visible-fraction", type=float, required=True)
    parser.add_argument("--decision-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    value = build_predictive_objective_decision(
        reviewer=args.reviewer,
        temporal_objective=args.temporal_objective,
        visible_support_weighting=args.visible_support_weighting,
        minimum_visible_fraction=args.minimum_visible_fraction,
        decision_record=args.decision_record,
    )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    _write_text_durable(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
                "temporal_objective": value["temporal_objective"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
