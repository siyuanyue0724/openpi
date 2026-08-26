#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Audit learned task/ownership factors from an immutable representation snapshot."""

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
    entrypoint_name="representation factor-oracle auditor",
)

from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationPlan,
)
from picf_next.lingbot_native.representation_factor_oracle import (
    build_representation_factor_oracle,
    write_representation_factor_oracle,
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
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("validation", "heldout"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = RepresentationEvaluationPlan.load(args.plan)
    artifact = build_representation_factor_oracle(
        _load_json(args.snapshot),
        plan=plan,
        partition=args.partition,
    )
    write_representation_factor_oracle(args.output, artifact)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "checkpoint_global_step": artifact["checkpoint_global_step"],
                "output": str(args.output.resolve()),
                "partition": artifact["partition"],
                "scope": artifact["scope"],
                "status": artifact["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
