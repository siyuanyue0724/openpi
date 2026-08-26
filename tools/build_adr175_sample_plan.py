#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build ADR-175's reset-only, source-disjoint physical sample plan."""

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
    entrypoint_name="ADR-175 physical sample-plan builder",
)

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.calvin import (
    build_native_calvin_physical_sample_domain,
    build_native_calvin_physical_sample_plan,
    native_calvin_sample_plan_instance_id,
    select_native_calvin_physical_prompt_segment,
)
from picf_next.lingbot_native.representation_split import (
    RepresentationTrialSplit,
    build_representation_trial_split_with_reference_evaluation,
)
from picf_next.training.control import FrozenSamplePlan


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--plan-seed", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--evaluation-reference-split", type=Path, required=True)
    parser.add_argument("--evaluation-reference-split-sha256", required=True)
    parser.add_argument("--stream-plan-output", type=Path, required=True)
    parser.add_argument("--representation-split-output", type=Path, required=True)
    parser.add_argument("--build-report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = (
        args.stream_plan_output,
        args.representation_split_output,
        args.build_report_output,
    )
    if len(set(outputs)) != len(outputs):
        raise ValueError("ADR-175 sample-plan outputs must differ")
    conflicts = tuple(path for path in outputs if path.exists() or path.is_symlink())
    if conflicts:
        raise FileExistsError(conflicts)
    if _file_sha256(args.evaluation_reference_split) != (
        args.evaluation_reference_split_sha256
    ):
        raise ValueError("ADR-175 evaluation reference file SHA-256 differs")

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
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=1)
    reference = RepresentationTrialSplit.load(args.evaluation_reference_split)
    if reference.comparison_id != args.comparison_id:
        raise ValueError("ADR-175 comparison identity differs from its evaluation reference")

    excluded_sources = reference.evaluation_source_episode_indices
    plan = build_native_calvin_physical_sample_plan(
        dataset,
        comparison_id=args.comparison_id,
        seed=args.plan_seed,
        global_batch_size=args.global_batch_size,
        total_steps=args.total_steps,
        excluded_source_episode_indices=excluded_sources,
    )
    split = build_representation_trial_split_with_reference_evaluation(
        plan,
        dataset,
        training_steps=args.total_steps,
        evaluation_reference=reference,
        require_equal_training_budget=False,
    )

    task_visits: dict[str, int] = {}
    sample_receipts: list[dict[str, object]] = []
    for optimizer_step in range(plan.total_steps):
        for global_slot, sample in enumerate(plan.global_batch(optimizer_step).samples):
            instance_id = native_calvin_sample_plan_instance_id(
                optimizer_step=optimizer_step,
                sample=sample,
            )
            segment_index, prompt_receipt = select_native_calvin_physical_prompt_segment(
                dataset,
                sample_key=sample.sample_key,
                plan_sha256=plan.plan_sha256,
                episode_instance_id=instance_id,
            )
            task_key = dataset.index.segments[segment_index].task_key
            task_visits[task_key] = task_visits.get(task_key, 0) + 1
            sample_receipts.append(
                {
                    "global_slot": global_slot,
                    "optimizer_step": optimizer_step,
                    "prompt_selection_receipt_sha256": prompt_receipt,
                    "sample_key": sample.sample_key,
                    "selected_segment_index": segment_index,
                    "task_key": task_key,
                }
            )

    plan.write_metadata(args.stream_plan_output)
    split.write(args.representation_split_output)
    restored_plan = FrozenSamplePlan.from_metadata(
        args.stream_plan_output,
        sample_keys=build_native_calvin_physical_sample_domain(
            dataset,
            excluded_source_episode_indices=excluded_sources,
        ),
    )
    restored_split = RepresentationTrialSplit.load(args.representation_split_output)
    if restored_plan != plan or restored_split != split:
        raise RuntimeError("ADR-175 sample plan or representation split changed after publication")

    report = {
        "comparison_id": args.comparison_id,
        "dataset_manifest_sha256": manifest.tree_sha256,
        "evaluation_reference_split_artifact_sha256": reference.artifact_sha256,
        "evaluation_reference_split_file_sha256": args.evaluation_reference_split_sha256,
        "excluded_source_episode_indices": list(excluded_sources),
        "plan_file_sha256": _file_sha256(args.stream_plan_output),
        "plan_sha256": plan.plan_sha256,
        "representation_split_artifact_sha256": split.artifact_sha256,
        "representation_split_file_sha256": _file_sha256(
            args.representation_split_output
        ),
        "sample_count": len(sample_receipts),
        "sample_receipts_sha256": hashlib.sha256(
            _canonical_bytes(sample_receipts).rstrip(b"\n")
        ).hexdigest(),
        "schema": "picf-next.adr175-sample-plan-build-report.v1",
        "task_count": len(task_visits),
        "task_visits": dict(sorted(task_visits.items())),
    }
    write_bytes_durable_exclusive(args.build_report_output, _canonical_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
