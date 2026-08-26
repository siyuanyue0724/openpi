#!/usr/bin/env python3
"""Build the exhaustive two-lattice shared-Qwen grounding curriculum."""

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
    entrypoint_name="LingBot native VL curriculum builder",
)

from picf_next.lingbot_native.fixed_observation import (  # noqa: E402
    FixedObservationPairPlan,
    load_native_vl_grounding_audit,
)
from picf_next.lingbot_native.vl_curriculum import (  # noqa: E402
    NativeVLGroundingCurriculumPlan,
    build_native_vl_grounding_curriculum,
)


def _sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _verified_file_sha256(path: Path, expected: str, *, name: str) -> str:
    expected_sha256 = _sha256(expected, name=f"{name} expected SHA-256")
    source = path.expanduser().absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be one real file")
    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"{name} file SHA-256 changed")
    return observed


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-plan", type=Path, required=True)
    parser.add_argument("--pair-plan-sha256", required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--training-audit-sha256", required=True)
    parser.add_argument("--expected-group-count", type=int, required=True)
    parser.add_argument("--expected-source-variant-count", type=int, required=True)
    parser.add_argument("--expected-measurable-variant-count", type=int, required=True)
    parser.add_argument(
        "--expected-object-row-addressable-variant-count",
        type=int,
        required=True,
    )
    parser.add_argument("--expected-optimizer-step-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pair_plan_file_sha256 = _verified_file_sha256(
        args.pair_plan,
        args.pair_plan_sha256,
        name="fixed-X pair plan",
    )
    pair_plan = FixedObservationPairPlan.load(args.pair_plan)
    audit = load_native_vl_grounding_audit(
        args.training_audit,
        expected_file_sha256=_sha256(
            args.training_audit_sha256,
            name="training audit expected SHA-256",
        ),
        expected_partition="training",
    )
    plan = build_native_vl_grounding_curriculum(
        pair_plan,
        audit,
        pair_plan_file_sha256=pair_plan_file_sha256,
    )
    observed_counts = (
        len(plan.groups),
        plan.source_variant_count,
        sum(len(group.variants) for group in plan.groups),
        plan.object_row_addressable_variant_count,
        len(plan.steps),
    )
    expected_counts = tuple(
        _positive_int(value, name=name)
        for name, value in (
            ("expected group count", args.expected_group_count),
            ("expected source variant count", args.expected_source_variant_count),
            ("expected measurable variant count", args.expected_measurable_variant_count),
            (
                "expected object-row-addressable variant count",
                args.expected_object_row_addressable_variant_count,
            ),
            ("expected optimizer step count", args.expected_optimizer_step_count),
        )
    )
    if observed_counts != expected_counts:
        raise ValueError(
            f"native VL curriculum counts {observed_counts} differ from {expected_counts}"
        )
    plan.write(args.output)
    reloaded = NativeVLGroundingCurriculumPlan.load(args.output)
    if reloaded != plan:
        raise RuntimeError("native VL curriculum changed during durable round trip")
    output_payload = args.output.read_bytes()
    print(
        json.dumps(
            {
                "artifact_sha256": plan.artifact_sha256,
                "file_sha256": hashlib.sha256(output_payload).hexdigest(),
                "forward_microbatch_count": sum(len(step.batches) for step in plan.steps),
                "group_count": len(plan.groups),
                "optimizer_step_count": len(plan.steps),
                "output": str(args.output.resolve()),
                "rank_target_histograms": plan.rank_target_histograms,
                "rank_task_histograms": plan.rank_task_histograms,
                "measurable_variant_count": sum(len(group.variants) for group in plan.groups),
                "object_row_addressable_variant_count": (plan.object_row_addressable_variant_count),
                "source_variant_count": plan.source_variant_count,
                "visual_lattices": list(plan.visual_lattices),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
