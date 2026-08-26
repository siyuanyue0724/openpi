#!/usr/bin/env python3
# ruff: noqa: E402
"""Build an immutable step-zero representation replay baseline."""

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
    entrypoint_name="representation evaluation baseline builder",
)

from picf_next.lingbot_native.representation_baseline import (
    build_representation_evaluation_baseline,
    file_sha256,
    validate_representation_evaluation_baseline,
    write_representation_evaluation_baseline,
)
from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationPlan,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--source-evaluation-plan", type=Path, required=True)
    parser.add_argument("--source-evaluation-plan-sha256", required=True)
    parser.add_argument("--source-visual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if file_sha256(args.source_snapshot) != args.source_snapshot_sha256:
        raise ValueError("source snapshot file SHA-256 differs")
    if file_sha256(args.source_evaluation_plan) != args.source_evaluation_plan_sha256:
        raise ValueError("source evaluation plan file SHA-256 differs")
    try:
        snapshot = json.loads(args.source_snapshot.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("source representation snapshot is invalid") from error
    if not isinstance(snapshot, dict):
        raise ValueError("source representation snapshot must be a mapping")
    plan = RepresentationEvaluationPlan.load(args.source_evaluation_plan)
    baseline = build_representation_evaluation_baseline(
        source_snapshot=snapshot,
        source_snapshot_file_sha256=args.source_snapshot_sha256,
        source_evaluation_plan=plan,
        source_evaluation_plan_file_sha256=args.source_evaluation_plan_sha256,
        source_visual_root=args.source_visual_root,
    )
    write_representation_evaluation_baseline(args.output, baseline)
    loaded = json.loads(args.output.read_text(encoding="ascii"))
    if validate_representation_evaluation_baseline(loaded) != baseline:
        raise RuntimeError("representation evaluation baseline changed after publication")
    print(
        json.dumps(
            {
                "artifact_sha256": baseline["artifact_sha256"],
                "file_sha256": file_sha256(args.output),
                "output": str(args.output.resolve()),
                "sample_count": baseline["sample_count"],
                "source_evaluation_plan_artifact_sha256": (
                    baseline["source_evaluation_plan_artifact_sha256"]
                ),
                "source_replay_seed_sha256": baseline["source_replay_seed_sha256"],
                "source_snapshot_artifact_sha256": (baseline["source_snapshot_artifact_sha256"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
