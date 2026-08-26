#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build one immutable interleaved stream and source-disjoint representation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="representation split builder",
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
from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native.calvin import (
    build_native_calvin_physical_stream_plan,
    build_native_calvin_stream_plan,
    build_native_calvin_training_stream_plan,
    select_native_calvin_physical_prompt_segment,
)
from picf_next.lingbot_native.representation_split import (
    RepresentationTrialSplit,
    build_representation_trial_split,
    build_representation_trial_split_with_reference_evaluation,
)
from picf_next.lingbot_native.stream_plan import (
    add_reset_mixture_arguments,
    reset_mixture_values,
)
from picf_next.training.control import (
    EpisodeStreamPlan,
    FrozenEpisodeStreamPlan,
    FrozenResetMixtureStreamPlan,
    load_frozen_episode_stream_plan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _maximum_run(values: list[str]) -> int:
    maximum = 0
    previous: str | None = None
    current = 0
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        maximum = max(maximum, current)
    return maximum


def _stream_summary(
    plan: EpisodeStreamPlan,
    dataset: CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset,
) -> tuple[dict[str, object], tuple[str, ...]]:
    lane_visits: Counter[str] = Counter()
    previous_lane_step: dict[str, int] = {}
    revisit_lags: list[int] = []
    sample_keys: list[str] = []
    source_episodes: set[int] = set()
    episode_instances: set[str] = set()
    tasks: set[str] = set()
    task_sequences = [[] for _ in range(plan.global_batch_size)]
    instance_sequences = [[] for _ in range(plan.global_batch_size)]
    reset_count = 0
    component_counts: Counter[str] = Counter()

    for optimizer_step in range(plan.total_steps):
        component = (
            plan.component_for_step(optimizer_step)
            if isinstance(plan, FrozenResetMixtureStreamPlan)
            else "causal"
        )
        component_counts[component] += plan.global_batch_size
        for global_slot, transition in enumerate(plan.global_batch(optimizer_step).transitions):
            if isinstance(dataset, CalvinPhysicalTransitionDataset):
                selected_segment_index, _receipt = select_native_calvin_physical_prompt_segment(
                    dataset,
                    sample_key=transition.sample.sample_key,
                    plan_sha256=plan.plan_sha256,
                    episode_instance_id=transition.episode_instance_id,
                )
                segment = dataset.index.segments[selected_segment_index]
            else:
                locator = dataset.locator_by_key(transition.sample.sample_key)
                segment = dataset.index.segments[locator.segment_index]
            task_key = segment.task_key
            sample_keys.append(transition.sample.sample_key)
            source_episodes.add(int(segment.episode_index))
            episode_instances.add(transition.episode_instance_id)
            tasks.add(task_key)
            task_sequences[global_slot].append(task_key)
            instance_sequences[global_slot].append(transition.episode_instance_id)
            if component == "causal":
                lane_visits[transition.lane_id] += 1
                previous_step = previous_lane_step.get(transition.lane_id)
                if previous_step is not None:
                    revisit_lags.append(optimizer_step - previous_step)
                previous_lane_step[transition.lane_id] = optimizer_step
            reset_count += int(transition.transition_index == 0)

    visit_counts = tuple(lane_visits.values())
    return (
        {
            "episode_instance_count": len(episode_instances),
            "estimator_component_sample_counts": dict(sorted(component_counts.items())),
            "lane_count": plan.lane_count,
            "lane_interleave_factor": plan.lane_interleave_factor,
            "lane_revisit_lag_maximum": max(revisit_lags) if revisit_lags else None,
            "lane_revisit_lag_minimum": min(revisit_lags) if revisit_lags else None,
            "lane_visit_count_maximum": max(visit_counts),
            "lane_visit_count_minimum": min(visit_counts),
            "maximum_consecutive_episode_instance_steps": max(
                _maximum_run(sequence) for sequence in instance_sequences
            ),
            "maximum_consecutive_task_steps": max(
                _maximum_run(sequence) for sequence in task_sequences
            ),
            "reset_sample_count": reset_count,
            "sample_count": len(sample_keys),
            "sample_keys_sha256": _canonical_sha256(sample_keys),
            "source_episode_count": len(source_episodes),
            "task_count": len(tasks),
        },
        tuple(sample_keys),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--plan-seed", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--lane-interleave-factor", type=int, default=1)
    parser.add_argument("--physical-event-stream", action="store_true")
    parser.add_argument("--minimum-future-source-frames", type=int, default=0)
    add_reset_mixture_arguments(parser)
    evaluation = parser.add_mutually_exclusive_group(required=True)
    evaluation.add_argument("--partition-seed", type=int)
    evaluation.add_argument("--evaluation-reference-split", type=Path)
    parser.add_argument("--evaluation-reference-split-sha256")
    parser.add_argument("--allow-reference-budget-change", action="store_true")
    parser.add_argument("--segments-per-task", type=int, default=2)
    parser.add_argument("--stream-plan-output", type=Path, required=True)
    parser.add_argument("--representation-split-output", type=Path, required=True)
    parser.add_argument("--build-report-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reset_mixture = reset_mixture_values(args)
    if args.evaluation_reference_split is None:
        if args.evaluation_reference_split_sha256 is not None:
            raise ValueError("evaluation reference split digest requires its reference split")
        if args.allow_reference_budget_change:
            raise ValueError("reference budget change requires an evaluation reference split")
    elif args.evaluation_reference_split_sha256 is None or args.build_report_output is None:
        raise ValueError(
            "referenced evaluation requires its file digest and a durable build report"
        )
    outputs = (
        args.stream_plan_output,
        args.representation_split_output,
        *((args.build_report_output,) if args.build_report_output is not None else ()),
    )
    if len(set(outputs)) != len(outputs):
        raise ValueError("representation build outputs must differ")
    conflicts = tuple(path for path in outputs if path.exists() or path.is_symlink())
    if conflicts:
        raise FileExistsError(conflicts)

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
    dataset = (
        CalvinPhysicalTransitionDataset(index, action_horizon=1)
        if args.physical_event_stream
        else CalvinStatefulTransitionDataset(index, action_horizon=1)
    )
    if args.minimum_future_source_frames < 0:
        raise ValueError("minimum future source frames must be non-negative")
    if args.minimum_future_source_frames and not args.physical_event_stream:
        raise ValueError(
            "minimum future source frames require the physical-event stream"
        )
    reference: RepresentationTrialSplit | None = None
    excluded_source_episode_indices: tuple[int, ...] = ()
    if args.evaluation_reference_split is not None:
        if _sha256(args.evaluation_reference_split) != (args.evaluation_reference_split_sha256):
            raise ValueError("evaluation reference split file SHA-256 differs")
        if args.lane_interleave_factor <= 1 and not args.allow_reference_budget_change:
            raise ValueError("referenced evaluation mode requires an interleaved candidate")
        reference = RepresentationTrialSplit.load(args.evaluation_reference_split)
        if args.segments_per_task != reference.segments_per_task:
            raise ValueError("evaluation reference segments-per-task differs")
        excluded_source_episode_indices = reference.evaluation_source_episode_indices
    if args.physical_event_stream and reset_mixture is not None:
        raise ValueError("the ADR149 physical stream does not use a synthetic reset mixture")
    plan = (
        build_native_calvin_physical_stream_plan(
            dataset,
            comparison_id=args.comparison_id,
            seed=args.plan_seed,
            global_batch_size=args.global_batch_size,
            total_steps=args.total_steps,
            lane_interleave_factor=args.lane_interleave_factor,
            excluded_source_episode_indices=excluded_source_episode_indices,
            minimum_future_source_frames=args.minimum_future_source_frames,
        )
        if args.physical_event_stream
        else build_native_calvin_training_stream_plan(
            dataset,
            comparison_id=args.comparison_id,
            seed=args.plan_seed,
            global_batch_size=args.global_batch_size,
            total_steps=args.total_steps,
            lane_interleave_factor=args.lane_interleave_factor,
            excluded_source_episode_indices=excluded_source_episode_indices,
            reset_numerator=(None if reset_mixture is None else reset_mixture[0]),
            reset_denominator=(None if reset_mixture is None else reset_mixture[1]),
        )
    )
    baseline_plan: FrozenEpisodeStreamPlan | None = None
    if reference is None:
        split = build_representation_trial_split(
            plan,
            dataset,
            training_steps=args.total_steps,
            partition_seed=args.partition_seed,
            segments_per_task=args.segments_per_task,
        )
    else:
        if not args.allow_reference_budget_change:
            baseline_plan = (
                build_native_calvin_physical_stream_plan(
                    dataset,
                    comparison_id=args.comparison_id,
                    seed=args.plan_seed,
                    global_batch_size=args.global_batch_size,
                    total_steps=args.total_steps,
                    lane_interleave_factor=1,
                    minimum_future_source_frames=args.minimum_future_source_frames,
                )
                if args.physical_event_stream
                else build_native_calvin_stream_plan(
                    dataset,
                    comparison_id=args.comparison_id,
                    seed=args.plan_seed,
                    global_batch_size=args.global_batch_size,
                    total_steps=args.total_steps,
                    lane_interleave_factor=1,
                )
            )
            if reference.stream_plan_sha256 != baseline_plan.plan_sha256:
                raise ValueError("evaluation reference is not the exact non-interleaved baseline")
        split = build_representation_trial_split_with_reference_evaluation(
            plan,
            dataset,
            training_steps=args.total_steps,
            evaluation_reference=reference,
            require_equal_training_budget=not args.allow_reference_budget_change,
        )

    plan.write_metadata(args.stream_plan_output)
    split.write(args.representation_split_output)
    restored_plan = load_frozen_episode_stream_plan(
        args.stream_plan_output,
        episodes=plan.episodes,
    )
    restored_split = RepresentationTrialSplit.load(args.representation_split_output)
    if restored_plan != plan or restored_split != split:
        raise RuntimeError("representation plan or split changed after publication")

    candidate_summary, candidate_keys = _stream_summary(plan, dataset)
    baseline_summary: dict[str, object] | None = None
    baseline_overlap: dict[str, object] | None = None
    if baseline_plan is not None and reference is not None:
        baseline_summary, baseline_keys = _stream_summary(baseline_plan, dataset)
        candidate_key_set = set(candidate_keys)
        baseline_key_set = set(baseline_keys)
        overlap = candidate_key_set & baseline_key_set
        multiset_overlap = Counter(candidate_keys) & Counter(baseline_keys)
        source_overlap = set(split.training_source_episode_indices) & set(
            reference.training_source_episode_indices
        )
        baseline_overlap = {
            "positionally_equal_sample_count": sum(
                candidate == baseline
                for candidate, baseline in zip(candidate_keys, baseline_keys, strict=True)
            ),
            "sample_key_sequence_equal": candidate_keys == baseline_keys,
            "sample_key_multiset_intersection_count": sum(multiset_overlap.values()),
            "training_source_episode_intersection_count": len(source_overlap),
            "unique_sample_key_intersection_count": len(overlap),
            "unique_sample_key_union_count": len(candidate_key_set | baseline_key_set),
        }
    evaluation_overlap: dict[str, object] | None = None
    if reference is not None:
        training_sources = set(split.training_source_episode_indices)
        validation_sources = {item.source_episode_index for item in reference.validation_segments}
        heldout_sources = {item.source_episode_index for item in reference.heldout_segments}
        evaluation_overlap = {
            "candidate_training_heldout_source_episode_intersection_count": len(
                training_sources & heldout_sources
            ),
            "candidate_training_validation_source_episode_intersection_count": len(
                training_sources & validation_sources
            ),
            "heldout_source_episode_count": len(heldout_sources),
            "validation_heldout_source_episode_intersection_count": len(
                validation_sources & heldout_sources
            ),
            "validation_source_episode_count": len(validation_sources),
        }
    report: dict[str, object] = {
        "schema": "picf-next.lingbot-representation-split-build-report.v2",
        "baseline_noninterleaved_stream": baseline_summary,
        "baseline_overlap": baseline_overlap,
        "candidate_stream": candidate_summary,
        "evaluation_source_overlap": evaluation_overlap,
        "evaluation_reference_preserved": reference is not None,
        "evaluation_reference_budget_change": args.allow_reference_budget_change,
        "evaluation_reference_split_artifact_sha256": (
            None if reference is None else reference.artifact_sha256
        ),
        "evaluation_reference_split_file_sha256": (
            None
            if args.evaluation_reference_split is None
            else _sha256(args.evaluation_reference_split)
        ),
        "excluded_evaluation_source_episode_count": len(excluded_source_episode_indices),
        "excluded_evaluation_source_episode_indices_sha256": _canonical_sha256(
            excluded_source_episode_indices
        ),
        "lane_count": plan.lane_count,
        "lane_interleave_factor": plan.lane_interleave_factor,
        "minimum_future_source_frames": args.minimum_future_source_frames,
        "partition_seed": split.partition_seed,
        "physical_event_stream": args.physical_event_stream,
        "representation_split_artifact_sha256": split.artifact_sha256,
        "representation_split_file_sha256": _sha256(args.representation_split_output),
        "stream_plan_file_sha256": _sha256(args.stream_plan_output),
        "stream_plan_sha256": plan.plan_sha256,
        "training_sample_count": split.training_sample_count,
        "training_source_episode_count": len(split.training_source_episode_indices),
    }
    if args.build_report_output is not None:
        write_bytes_durable_exclusive(
            args.build_report_output,
            json.dumps(report, indent=2, sort_keys=True).encode("ascii") + b"\n",
        )
        report["build_report_file_sha256"] = _sha256(args.build_report_output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
