#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build the frozen causal task-intervention plan for LingBot representation training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="representation task intervention plan builder",
)

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.eval.calvin_task_relevance import calvin_exact_task_loss_identities
from picf_next.lingbot_native.calvin import build_native_calvin_stream_plan
from picf_next.lingbot_native.representation_intervention import (
    RepresentationTaskInterventionPlan,
    build_representation_task_intervention_plan,
)
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--representation-split", type=Path, required=True)
    parser.add_argument("--representation-split-sha256", required=True)
    parser.add_argument(
        "--comparison-id",
        default="lingbot-vla2-native-picf-full",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--lane-interleave-factor", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _summary(plan: RepresentationTaskInterventionPlan) -> dict[str, object]:
    exact = tuple(item for item in plan.slots if item.intervened)
    donor_targets_by_episode: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    ordered_donors_by_episode: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for item in exact:
        if item.donor_target_identity_keys is None:
            raise RuntimeError("published exact task slot lost its donor target")
        donor_targets_by_episode[item.episode_instance_id].add(item.donor_target_identity_keys)
        ordered_donors_by_episode[item.episode_instance_id].append(item.donor_target_identity_keys)

    maximum_run = 0
    for targets in ordered_donors_by_episode.values():
        run = 0
        previous: tuple[str, ...] | None = None
        for target in targets:
            run = run + 1 if target == previous else 1
            previous = target
            maximum_run = max(maximum_run, run)

    def histogram(values) -> dict[str, int]:
        return {
            "|".join(key) if isinstance(key, tuple) else str(key): count
            for key, count in sorted(Counter(values).items())
        }

    return {
        "exact_slot_count": plan.exact_slot_count,
        "inexact_slot_count": plan.inexact_slot_count,
        "matching_attempt": plan.matching_attempt,
        "maximum_same_donor_target_run": maximum_run,
        "minimum_distinct_donor_targets_per_exact_episode": min(
            map(len, donor_targets_by_episode.values())
        ),
        "natural_target_histogram": histogram(item.target_identity_keys for item in exact),
        "donor_target_histogram": histogram(item.donor_target_identity_keys for item in exact),
        "natural_task_histogram": histogram(item.task_key for item in exact),
        "donor_task_histogram": histogram(item.donor_task_key for item in exact),
    }


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if _sha256(args.representation_split) != args.representation_split_sha256:
        raise ValueError("representation split file SHA-256 differs")

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
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    split = RepresentationTrialSplit.load(args.representation_split)
    stream_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id=args.comparison_id,
        seed=args.seed,
        global_batch_size=args.global_batch_size,
        total_steps=args.total_steps,
        lane_interleave_factor=args.lane_interleave_factor,
        excluded_source_episode_indices=split.evaluation_source_episode_indices,
    )
    split_identity = (
        split.dataset_id,
        split.dataset_revision,
        split.dataset_manifest_sha256,
        split.comparison_id,
        split.stream_plan_sha256,
        split.training_steps,
    )
    stream_identity = (
        stream_plan.dataset_id,
        stream_plan.dataset_revision,
        stream_plan.dataset_manifest_sha256,
        stream_plan.comparison_id,
        stream_plan.plan_sha256,
        stream_plan.total_steps,
    )
    if split_identity != stream_identity:
        raise ValueError("representation source split differs from the rebuilt stream")

    plan = build_representation_task_intervention_plan(
        stream_plan,
        dataset,
        task_identity_resolver=calvin_exact_task_loss_identities,
    )
    plan.write(args.output)
    if RepresentationTaskInterventionPlan.load(args.output) != plan:
        raise RuntimeError("task intervention plan changed after publication")
    print(
        json.dumps(
            {
                "artifact_sha256": plan.artifact_sha256,
                "file_sha256": _sha256(args.output),
                "output": str(args.output.resolve()),
                "representation_split_artifact_sha256": split.artifact_sha256,
                "stream_plan_sha256": stream_plan.plan_sha256,
                "summary": _summary(plan),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
