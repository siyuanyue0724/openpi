#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build the preregistered numeric gate for representation steps 0 and 200."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="representation numeric gate builder",
)

from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationPlan,
)
from picf_next.lingbot_native.representation_gate import (
    build_representation_numeric_gate,
    write_representation_numeric_gate,
)


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline-snapshot", type=Path, required=True)
    parser.add_argument("--decision-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = RepresentationEvaluationPlan.load(args.plan)
    gate = build_representation_numeric_gate(
        _load_json(args.baseline_snapshot),
        _load_json(args.decision_snapshot),
        plan=plan,
    )
    write_representation_numeric_gate(args.output, gate)
    print(
        json.dumps(
            {
                "artifact_sha256": gate["artifact_sha256"],
                "authorizes_joint_adoption": gate["authorizes_joint_adoption"],
                "output": str(args.output.resolve()),
                "status": gate["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
