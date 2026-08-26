#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build one immutable ADR-175 broad-support strata companion artifact."""

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
    entrypoint_name="ADR-175 broad-support contract builder",
)

from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.dataset_manifest import file_sha256, load_dataset_file_manifest
from picf_next.lingbot_native.adr175_contract import (
    Adr175BroadSupportContract,
    build_adr175_broad_support_contract,
)
from picf_next.lingbot_native.calvin import (
    build_native_calvin_physical_episode_domain,
    build_native_calvin_physical_sample_domain,
)
from picf_next.lingbot_native.entity_evaluation_plan import EntityEvaluationPlan
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit
from picf_next.training.control import (
    FrozenEpisodeStreamPlan,
    FrozenSamplePlan,
    load_frozen_episode_stream_plan,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-tree-sha256", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-file-sha256", required=True)
    parser.add_argument("--stream-plan", type=Path, required=True)
    parser.add_argument("--stream-plan-file-sha256", required=True)
    parser.add_argument("--stream-plan-sha256", required=True)
    parser.add_argument("--representation-split", type=Path, required=True)
    parser.add_argument("--representation-split-file-sha256", required=True)
    parser.add_argument("--representation-split-artifact-sha256", required=True)
    parser.add_argument("--entity-evaluation-plan", type=Path, required=True)
    parser.add_argument("--entity-evaluation-plan-file-sha256", required=True)
    parser.add_argument("--entity-evaluation-plan-artifact-sha256", required=True)
    parser.add_argument("--training-prefix-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_file_sha256(path: Path, expected: str, name: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{name} file SHA-256 differs: expected {expected}, observed {observed}")


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    _require_file_sha256(
        args.dataset_manifest,
        args.dataset_manifest_file_sha256,
        "dataset manifest",
    )
    manifest = load_dataset_file_manifest(args.dataset_manifest)
    expected_dataset_identity = (
        args.dataset_id,
        args.dataset_revision,
        args.dataset_split.resolve().name,
        args.dataset_tree_sha256,
    )
    observed_dataset_identity = (
        manifest.dataset_id,
        manifest.dataset_revision,
        manifest.split_name,
        manifest.tree_sha256,
    )
    if observed_dataset_identity != expected_dataset_identity:
        raise ValueError("dataset identity or content tree differs from its CLI pins")

    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=1)

    _require_file_sha256(
        args.representation_split,
        args.representation_split_file_sha256,
        "representation split",
    )
    split = RepresentationTrialSplit.load(args.representation_split)
    if split.artifact_sha256 != args.representation_split_artifact_sha256:
        raise ValueError("representation split semantic SHA-256 differs from its CLI pin")

    _require_file_sha256(
        args.entity_evaluation_plan,
        args.entity_evaluation_plan_file_sha256,
        "entity evaluation plan",
    )
    evaluation_plan = EntityEvaluationPlan.load(args.entity_evaluation_plan)
    if evaluation_plan.artifact_sha256 != args.entity_evaluation_plan_artifact_sha256:
        raise ValueError("entity evaluation plan semantic SHA-256 differs from its CLI pin")

    _require_file_sha256(
        args.stream_plan,
        args.stream_plan_file_sha256,
        "stream plan",
    )
    stream_plan_payload = json.loads(args.stream_plan.read_text())
    stream_plan_metadata = stream_plan_payload.get("metadata")
    if not isinstance(stream_plan_metadata, dict):
        raise ValueError("ADR-175 stream plan omits metadata")
    if stream_plan_metadata.get("schema") == "picf-next.frozen-sample-plan.v1":
        stream_plan = FrozenSamplePlan.from_metadata(
            args.stream_plan,
            sample_keys=build_native_calvin_physical_sample_domain(
                dataset,
                excluded_source_episode_indices=(
                    split.stream_domain_excluded_source_episode_indices
                ),
            ),
        )
    else:
        episodes = build_native_calvin_physical_episode_domain(
            dataset,
            excluded_source_episode_indices=(
                split.stream_domain_excluded_source_episode_indices
            ),
        )
        stream_plan = load_frozen_episode_stream_plan(args.stream_plan, episodes=episodes)
        if not isinstance(stream_plan, FrozenEpisodeStreamPlan):
            raise TypeError("ADR-175 broad support requires a physical training plan")
    if stream_plan.plan_sha256 != args.stream_plan_sha256:
        raise ValueError("stream plan semantic SHA-256 differs from its CLI pin")

    contract = build_adr175_broad_support_contract(
        dataset=dataset,
        stream_plan=stream_plan,
        representation_split=split,
        entity_evaluation_plan=evaluation_plan,
        training_prefix_steps=args.training_prefix_steps,
    )
    contract.write(args.output)
    restored = Adr175BroadSupportContract.load(args.output)
    if restored != contract:
        raise RuntimeError("ADR-175 broad-support contract changed after publication")
    print(
        json.dumps(
            {
                "ambiguous_task_count": contract.ambiguous_task_count,
                "artifact_sha256": contract.artifact_sha256,
                "exact_task_count": contract.exact_task_count,
                "matched_arm_input_sha256": contract.matched_arm_input_sha256,
                "output": str(args.output.resolve()),
                "training_prefix_sample_count": contract.training_prefix_sample_count,
                "training_prefix_steps": contract.training_prefix_steps,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
