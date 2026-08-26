#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build the immutable source-only LingBot representation evaluation plan."""

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
    entrypoint_name="representation evaluation plan builder",
)

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.eval.calvin_task_relevance import calvin_exact_task_loss_identities
from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationPlan,
    build_representation_evaluation_plan,
    build_representation_warm_evaluation_plan,
)
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
    RepresentationTrialSplit,
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
    parser.add_argument("--evaluation-reference-split", type=Path)
    parser.add_argument("--evaluation-reference-split-sha256")
    parser.add_argument("--warm-history-transitions", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
    split = RepresentationTrialSplit.load(args.representation_split)
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    reference: RepresentationTrialSplit | None = None
    reference_plan: RepresentationEvaluationPlan | None = None
    if args.warm_history_transitions is not None:
        if args.warm_history_transitions != 8:
            raise ValueError("ADR-121 warm evaluation requires exactly eight transitions")
        if (
            args.evaluation_reference_split is not None
            or args.evaluation_reference_split_sha256 is not None
        ):
            raise ValueError("warm evaluation cannot consume a reset evaluation reference")
    elif split.schema == REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA:
        if (
            args.evaluation_reference_split is None
            or args.evaluation_reference_split_sha256 is None
        ):
            raise ValueError(
                "reference-derived representation split requires its exact source split"
            )
        if _sha256(args.evaluation_reference_split) != args.evaluation_reference_split_sha256:
            raise ValueError("evaluation reference split file SHA-256 differs")
        reference = RepresentationTrialSplit.load(args.evaluation_reference_split)
        if split.evaluation_reference_split_artifact_sha256 != reference.artifact_sha256:
            raise ValueError("representation split names another evaluation reference")
        reference_plan = build_representation_evaluation_plan(
            reference,
            dataset,
            task_identity_resolver=calvin_exact_task_loss_identities,
        )
    elif (
        args.evaluation_reference_split is not None
        or args.evaluation_reference_split_sha256 is not None
    ):
        raise ValueError("v1 representation split cannot consume an evaluation reference")
    if args.warm_history_transitions is None:
        plan = build_representation_evaluation_plan(
            split,
            dataset,
            task_identity_resolver=calvin_exact_task_loss_identities,
            evaluation_reference_plan_sha256=(
                None if reference_plan is None else reference_plan.artifact_sha256
            ),
        )
    else:
        plan = build_representation_warm_evaluation_plan(
            split,
            dataset,
            task_identity_resolver=calvin_exact_task_loss_identities,
            history_transitions=args.warm_history_transitions,
        )
    if reference_plan is not None and (
        plan.world_size != reference_plan.world_size or plan.items != reference_plan.items
    ):
        raise RuntimeError("referenced representation evaluation bank changed")
    plan.write(args.output)
    loaded = RepresentationEvaluationPlan.load(args.output)
    if loaded != plan:
        raise RuntimeError("representation evaluation plan changed after publication")
    print(
        json.dumps(
            {
                "artifact_sha256": plan.artifact_sha256,
                "file_sha256": _sha256(args.output),
                "item_count": len(plan.items),
                "history_transitions": plan.history_transitions,
                "output": str(args.output.resolve()),
                "representation_split_sha256": split.artifact_sha256,
                "evaluation_reference_preserved": reference is not None,
                "evaluation_reference_plan_artifact_sha256": (
                    None if reference_plan is None else reference_plan.artifact_sha256
                ),
                "evaluation_reference_split_artifact_sha256": (
                    None if reference is None else reference.artifact_sha256
                ),
                "evaluation_reference_split_file_sha256": (
                    None
                    if args.evaluation_reference_split is None
                    else _sha256(args.evaluation_reference_split)
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
