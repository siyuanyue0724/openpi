#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build immutable source-disjoint fixed-X validation and held-out banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="fixed-observation evaluation-plan builder",
)

from picf_next.lingbot_native.fixed_observation import (
    FixedObservationPairPlan,
    load_fixed_observation_audit,
)
from picf_next.lingbot_native.fixed_observation_evaluation import (
    FIXED_OBSERVATION_EVALUATION_PARTITIONS,
    FixedObservationEvaluationPlan,
    build_fixed_observation_evaluation_plan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pair-plan", type=Path, required=True)
    parser.add_argument("--training-pair-plan-sha256", required=True)
    for partition in ("training", *FIXED_OBSERVATION_EVALUATION_PARTITIONS):
        parser.add_argument(
            f"--{partition}-token-grid-audit",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{partition}-token-grid-audit-sha256",
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    pair_plan_file_sha256 = _sha256(args.training_pair_plan)
    if pair_plan_file_sha256 != args.training_pair_plan_sha256:
        raise ValueError("training pair-plan file SHA-256 differs")
    pair_plan = FixedObservationPairPlan.load(args.training_pair_plan)

    audits = {}
    for partition in ("training", *FIXED_OBSERVATION_EVALUATION_PARTITIONS):
        path = getattr(args, f"{partition}_token_grid_audit")
        expected_file_sha256 = getattr(
            args,
            f"{partition}_token_grid_audit_sha256",
        )
        audits[partition] = load_fixed_observation_audit(
            path,
            expected_file_sha256=expected_file_sha256,
            expected_partition=partition,
        )

    plan = build_fixed_observation_evaluation_plan(
        training_audit=audits["training"],
        validation_audit=audits["validation"],
        heldout_audit=audits["heldout"],
        training_pair_plan=pair_plan,
    )
    plan.write(args.output)
    if FixedObservationEvaluationPlan.load(args.output) != plan:
        raise RuntimeError("fixed-X evaluation plan changed after publication")
    print(
        json.dumps(
            {
                "artifact_sha256": plan.artifact_sha256,
                "file_sha256": _sha256(args.output),
                "item_count": len(plan.items),
                "output": str(args.output.resolve()),
                "partition_counts": {
                    partition: sum(item.partition == partition for item in plan.items)
                    for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
                },
                "rank_counts": {
                    partition: [
                        len(plan.items_for(partition, rank)) for rank in range(plan.world_size)
                    ]
                    for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
                },
                "target_histogram": plan.target_histogram,
                "task_histogram": plan.task_histogram,
                "training_pair_plan_artifact_sha256": (plan.training_pair_plan_sha256),
                "training_pair_plan_file_sha256": pair_plan_file_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
