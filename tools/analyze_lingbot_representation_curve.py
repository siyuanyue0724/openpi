#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Freeze historical thresholds or analyze one matched representation curve."""

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
    entrypoint_name="representation clustered-curve analyzer",
)

from picf_next.lingbot_native.representation_cluster_statistics import (
    build_representation_cluster_curve,
    build_representation_cluster_thresholds,
    validate_representation_cluster_thresholds,
    write_representation_cluster_artifact,
)
from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationPlan,
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze-thresholds",
        help="derive evaluation-bank resolution from historical artifacts",
    )
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument(
        "--source",
        nargs=3,
        action="append",
        metavar=("NAME", "SNAPSHOT", "FACTOR_ORACLE"),
        required=True,
    )
    freeze.add_argument("--replicates", type=int, default=20_000)
    freeze.add_argument("--bootstrap-seed", type=int, default=135_202_608)
    freeze.add_argument("--confidence-level", type=float, default=0.95)
    freeze.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser(
        "compare",
        help="compare matched M/E boundaries with frozen thresholds",
    )
    compare.add_argument("--plan", type=Path, required=True)
    compare.add_argument("--thresholds", type=Path, required=True)
    compare.add_argument(
        "--boundary",
        nargs=4,
        action="append",
        metavar=("ARM", "STEP", "SNAPSHOT", "FACTOR_ORACLE"),
        required=True,
    )
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _freeze(args: argparse.Namespace) -> dict[str, object]:
    plan = RepresentationEvaluationPlan.load(args.plan)
    sources = {}
    for name, snapshot, factor in args.source:
        if name in sources:
            raise ValueError(f"duplicate historical source name: {name}")
        sources[name] = (_load_json(Path(snapshot)), _load_json(Path(factor)))
    artifact = build_representation_cluster_thresholds(
        sources,
        plan=plan,
        replicates=args.replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    write_representation_cluster_artifact(args.output, artifact)
    return artifact


def _compare(args: argparse.Namespace) -> dict[str, object]:
    plan = RepresentationEvaluationPlan.load(args.plan)
    thresholds = validate_representation_cluster_thresholds(_load_json(args.thresholds))
    arms: dict[str, dict[int, tuple[dict[str, object], dict[str, object]]]] = {}
    for arm, raw_step, snapshot, factor in args.boundary:
        try:
            step = int(raw_step)
        except ValueError as error:
            raise ValueError(f"invalid boundary step: {raw_step}") from error
        if step < 0 or step in arms.setdefault(arm, {}):
            raise ValueError(f"duplicate or negative boundary: {arm}/{raw_step}")
        arms[arm][step] = (_load_json(Path(snapshot)), _load_json(Path(factor)))
    artifact = build_representation_cluster_curve(
        arms,
        thresholds,
        plan=plan,
    )
    write_representation_cluster_artifact(args.output, artifact)
    return artifact


def main() -> None:
    args = _parse_args()
    artifact = _freeze(args) if args.command == "freeze-thresholds" else _compare(args)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "command": args.command,
                "output": str(args.output.resolve()),
                "status": artifact["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
