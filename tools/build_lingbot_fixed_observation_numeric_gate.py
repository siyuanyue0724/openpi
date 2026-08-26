#!/usr/bin/env python3
"""Build the preregistered update-0/update-200 fixed-X numeric decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.repository_import import bind_entrypoint_to_own_repository
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="fixed-observation numeric gate builder",
)

from picf_next.lingbot_native.fixed_observation_evaluation import (  # noqa: E402
    FixedObservationEvaluationPlan,
)
from picf_next.lingbot_native.fixed_observation_gate import (  # noqa: E402
    build_fixed_observation_numeric_gate,
    write_fixed_observation_numeric_gate,
)


def _json_mapping(path: Path, *, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be one real file")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not canonical JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--baseline-snapshot", type=Path, required=True)
    parser.add_argument("--decision-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = FixedObservationEvaluationPlan.load(args.evaluation_plan)
    gate = build_fixed_observation_numeric_gate(
        _json_mapping(args.baseline_snapshot, name="baseline snapshot"),
        _json_mapping(args.decision_snapshot, name="decision snapshot"),
        plan=plan,
    )
    write_fixed_observation_numeric_gate(args.output, gate)
    payload = args.output.read_bytes()
    print(
        json.dumps(
            {
                "artifact_sha256": gate["artifact_sha256"],
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "output": str(args.output.resolve()),
                "status": gate["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
