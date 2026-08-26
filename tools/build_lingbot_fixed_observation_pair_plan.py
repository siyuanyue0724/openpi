#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build the immutable truth-audited fixed-X reset-pair training plan."""

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
    entrypoint_name="fixed-observation pair plan builder",
)

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.fixed_observation import (
    FixedObservationPairPlan,
    build_fixed_observation_pair_plan,
    load_fixed_observation_audit,
)
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit
from picf_next.training.control import (
    EpisodeSampleSequence,
    FrozenResetMixtureStreamPlan,
    load_frozen_episode_stream_plan,
)


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
    parser.add_argument("--stream-plan", type=Path, required=True)
    parser.add_argument("--stream-plan-sha256", required=True)
    parser.add_argument("--token-grid-audit", type=Path, required=True)
    parser.add_argument("--token-grid-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    for path, expected, name in (
        (
            args.representation_split,
            args.representation_split_sha256,
            "representation split",
        ),
        (args.stream_plan, args.stream_plan_sha256, "stream plan"),
        (args.token_grid_audit, args.token_grid_audit_sha256, "token-grid audit"),
    ):
        if _sha256(path) != expected:
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
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    split = RepresentationTrialSplit.load(args.representation_split)
    excluded = frozenset(split.evaluation_source_episode_indices)
    episodes = tuple(
        EpisodeSampleSequence(
            episode_key=episode.episode_key,
            sample_keys=episode.sample_keys,
        )
        for episode, segment in zip(
            dataset.episode_manifest,
            dataset.index.segments,
            strict=True,
        )
        if int(segment.episode_index) not in excluded
    )
    stream = load_frozen_episode_stream_plan(
        args.stream_plan,
        episodes=episodes,
    )
    if not isinstance(stream, FrozenResetMixtureStreamPlan):
        raise ValueError("fixed-X training requires a frozen reset-mixture stream")
    audit = load_fixed_observation_audit(
        args.token_grid_audit,
        expected_file_sha256=args.token_grid_audit_sha256,
        expected_partition="training",
    )
    if _sha256(args.dataset_manifest) != audit.dataset_manifest_file_sha256:
        raise ValueError("fixed-X audit belongs to another dataset manifest file")
    split_identity = (
        split.artifact_sha256,
        args.representation_split_sha256,
        split.comparison_id,
        split.stream_plan_sha256,
    )
    audit_identity = (
        audit.representation_split_artifact_sha256,
        audit.representation_split_file_sha256,
        audit.comparison_id,
        audit.stream_plan_sha256,
    )
    if split_identity != audit_identity:
        raise ValueError("fixed-X audit belongs to another representation split")

    plan = build_fixed_observation_pair_plan(stream, dataset, audit)
    plan.write(args.output)
    if FixedObservationPairPlan.load(args.output) != plan:
        raise RuntimeError("fixed-X pair plan changed after publication")
    print(
        json.dumps(
            {
                "artifact_sha256": plan.artifact_sha256,
                "candidate_group_count": plan.candidate_group_count,
                "file_sha256": _sha256(args.output),
                "output": str(args.output.resolve()),
                "pair_count": len(plan.pairs),
                "stream_plan_sha256": plan.stream_plan_sha256,
                "target_histogram": plan.target_histogram,
                "task_histogram": plan.task_histogram,
                "unique_source_count": len({item.group.source_global_index for item in plan.pairs}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
