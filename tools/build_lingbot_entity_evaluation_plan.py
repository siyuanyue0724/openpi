#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build a task-independent, source-disjoint entity evaluation plan."""

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
    entrypoint_name="entity evaluation plan builder",
)

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.entity_evaluation_plan import build_entity_evaluation_plan
from picf_next.lingbot_native.entity_evaluation_plan import ENTITY_EVALUATION_WORLD_SIZES
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
        "--world-size",
        type=int,
        choices=ENTITY_EVALUATION_WORLD_SIZES,
        default=2,
    )
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
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    split = RepresentationTrialSplit.load(args.representation_split)
    plan = build_entity_evaluation_plan(split, dataset, world_size=args.world_size)
    plan.write(args.output)
    print(
        json.dumps(
            {
                "artifact_sha256": plan.artifact_sha256,
                "item_count": len(plan.items),
                "output": str(args.output.expanduser().absolute()),
                "representation_split_sha256": split.artifact_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
