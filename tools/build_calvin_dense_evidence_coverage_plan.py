#!/usr/bin/env python3
# ruff: noqa: E402
"""Freeze the exact CALVIN event set needed by one train/evaluation protocol."""

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
    entrypoint_name="dense evidence coverage plan builder",
)

from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.data.dense_evidence_coverage import (
    build_calvin_dense_evidence_coverage_plan,
)
from picf_next.lingbot_native.calvin import build_native_calvin_physical_episode_domain
from picf_next.lingbot_native.entity_evaluation_plan import EntityEvaluationPlan
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit
from picf_next.training.control import load_frozen_episode_stream_plan


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--stream-plan", type=Path, required=True)
    parser.add_argument("--stream-plan-sha256", required=True)
    parser.add_argument("--representation-split", type=Path, required=True)
    parser.add_argument("--representation-split-sha256", required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan-sha256", required=True)
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument("--minimum-future-source-frames", type=int, default=0)
    parser.add_argument(
        "--evaluation-history-transitions",
        type=int,
        default=0,
        help="Include this many real predecessor frames for eligible evaluation items.",
    )
    parser.add_argument(
        "--training-step-prefix",
        type=int,
        help="Cache only this prefix of the complete frozen stream.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.action_horizon <= 0:
        raise ValueError("action horizon must be positive")
    if args.minimum_future_source_frames < 0:
        raise ValueError("minimum future source frames must be non-negative")
    if args.evaluation_history_transitions < 0:
        raise ValueError("evaluation history transitions must be non-negative")
    for path, expected, name in (
        (args.stream_plan, args.stream_plan_sha256, "stream plan"),
        (
            args.representation_split,
            args.representation_split_sha256,
            "representation split",
        ),
        (args.evaluation_plan, args.evaluation_plan_sha256, "evaluation plan"),
    ):
        if _file_sha256(path) != expected:
            raise ValueError(f"{name} file SHA-256 differs")

    manifest = load_dataset_file_manifest(args.dataset_manifest)
    validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=args.dataset_split.resolve().name,
    )
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    physical_dataset = CalvinPhysicalTransitionDataset(
        index,
        action_horizon=args.action_horizon,
    )
    evaluation_dataset = CalvinStatefulTransitionDataset(
        index,
        action_horizon=args.action_horizon,
    )
    split = RepresentationTrialSplit.load(args.representation_split)
    stream = load_frozen_episode_stream_plan(
        args.stream_plan,
        episodes=build_native_calvin_physical_episode_domain(
            physical_dataset,
            excluded_source_episode_indices=(split.stream_domain_excluded_source_episode_indices),
            minimum_future_source_frames=args.minimum_future_source_frames,
        ),
    )
    evaluation = EntityEvaluationPlan.load(args.evaluation_plan)
    coverage = build_calvin_dense_evidence_coverage_plan(
        stream_plan=stream,
        representation_split=split,
        evaluation_plan=evaluation,
        physical_dataset=physical_dataset,
        evaluation_dataset=evaluation_dataset,
        training_step_prefix=args.training_step_prefix,
        evaluation_history_transitions=args.evaluation_history_transitions,
    )
    coverage.write(args.output)
    payload = {
        "artifact_sha256": coverage.artifact_sha256,
        "evaluation_record_count": sum(
            record.partition == "evaluation" for record in coverage.records
        ),
        "evaluation_history_transition_count": (
            coverage.evaluation_history_transition_count
        ),
        "evaluation_history_visit_count": coverage.evaluation_history_visit_count,
        "file_sha256": _file_sha256(args.output),
        "output": str(args.output.resolve()),
        "record_count": len(coverage.records),
        "records_sha256": coverage.records_sha256,
        "training_record_count": sum(record.partition == "training" for record in coverage.records),
        "training_visit_count": coverage.training_visit_count,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
