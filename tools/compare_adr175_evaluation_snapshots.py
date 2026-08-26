#!/usr/bin/env python3
"""Compare two immutable ADR-175 fixed-evaluation snapshots."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from picf_next.lingbot_native.adr175_validation import (
    ADR175_EXACT_TASK_TARGETS,
    canonical_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-baseline-step", type=int, default=0)
    parser.add_argument("--expected-candidate-step", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_snapshot(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise TypeError(f"snapshot is not an object: {path}")
    artifact_sha256 = value.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
        raise ValueError(f"snapshot omits artifact identity: {path}")
    semantic = dict(value)
    semantic.pop("artifact_sha256")
    if canonical_sha256(semantic) != artifact_sha256:
        raise ValueError(f"snapshot semantic SHA-256 differs: {path}")
    if value.get("status") != "PASS":
        raise ValueError(f"snapshot did not pass: {path}")
    return value


def sample_identity(sample: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        sample.get(field)
        for field in (
            "ordinal",
            "partition",
            "rank",
            "sample_key",
            "segment_index",
            "source_digest",
            "source_episode_index",
            "source_global_index",
            "task_key",
            "transition_index",
        )
    )


def entity_score(sample: dict[str, Any]) -> float:
    rows = sample["entity_evidence"]["rows"]
    if not rows:
        raise ValueError("entity evidence has no rows")
    values = [float(row["support_soft_iou_efficiency"]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("entity evidence is non-finite")
    return sum(values) / len(values)


def target_observable(sample: dict[str, Any], target_identity_keys: tuple[str, ...]) -> bool:
    evidence = sample.get("entity_evidence")
    if not isinstance(evidence, dict):
        raise TypeError("exact sample has no entity evidence")
    rows = evidence.get("rows")
    if not isinstance(rows, list):
        raise TypeError("exact entity evidence rows are not a list")
    observed_identity_keys: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("identity_key"), str):
            raise TypeError("exact entity row omits its identity key")
        observed_identity_keys.append(row["identity_key"])
    if len(observed_identity_keys) != len(set(observed_identity_keys)):
        raise ValueError("exact entity identities are not unique")
    target_visible_count = evidence.get("target_visible_count")
    if (
        isinstance(target_visible_count, bool)
        or not isinstance(target_visible_count, int)
        or target_visible_count != len(observed_identity_keys)
    ):
        raise ValueError("exact target-visible count differs from entity evidence")
    observable = set(target_identity_keys).issubset(observed_identity_keys)
    if sample.get("target_valid") is True and not observable:
        raise ValueError("exact sample resolved an unobservable target")
    return observable


def resolved_attention_value(
    sample: dict[str, Any], *, field: str, observable: bool
) -> float | None:
    resolved = sample.get("target_valid")
    if not isinstance(resolved, bool):
        raise TypeError("exact target resolution is not boolean")
    value = sample.get(field)
    if not observable:
        if resolved:
            raise ValueError("exact sample resolved an unobservable target")
        return None
    if not resolved:
        if value is not None:
            raise ValueError(f"unresolved exact target published {field}")
        return 0.0
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"exact {field} lies outside [0,1]")
    return result


def resolved_attention_mean(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    side: int,
    field: str,
    observable: list[bool],
) -> float | None:
    values = [
        resolved_attention_value(pair[side], field=field, observable=is_observable)
        for pair, is_observable in zip(pairs, observable, strict=True)
    ]
    observed_values = [float(value) for value in values if value is not None]
    return sum(observed_values) / len(observed_values) if observed_values else None


def paired_field_mean(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    side: int,
    field: str,
    context: str,
) -> float:
    values = [float(pair[side][field]) for pair in pairs]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite {field}: {context}")
    return sum(values) / len(values)


def summary(snapshot: dict[str, Any], partition: str) -> dict[str, Any]:
    source = snapshot["partition_summaries"][partition]
    entity = source["entity_set_summary"]
    return {
        "action_loss": float(source["action_loss"]),
        "area_efficiency": {
            key: float(value["mean_support_soft_iou_efficiency"])
            for key, value in sorted(entity["area_strata"].items())
        },
        "cardinality_abs_error": float(entity["mean_cardinality_absolute_error_at_0_5"]),
        "conditional_selectivity": float(source["conditional_selectivity"]),
        "context_probability": float(entity["mean_context_region_probability"]),
        "entity_set_score": float(source["entity_set_score"]),
        "existence_probability": float(entity["mean_existence_probability"]),
        "object_ownership_target_recall": float(entity["mean_object_ownership_target_recall"]),
        "ownership_soft_iou": float(entity["mean_ownership_soft_iou"]),
        "ownership_target_recall": float(entity["mean_ownership_target_recall"]),
        "pairwise_support_overlap": float(entity["mean_mean_pairwise_support_overlap"]),
        "posterior_adoption": float(source["posterior_adoption"]),
        "support_iou": float(entity["mean_support_soft_iou"]),
    }


def delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = {key: candidate[key] - baseline[key] for key in candidate if key != "area_efficiency"}
    result["action_relative"] = result["action_loss"] / baseline["action_loss"]
    result["entity_set_relative"] = result["entity_set_score"] / baseline["entity_set_score"]
    result["area_efficiency"] = {
        key: candidate["area_efficiency"][key] - baseline["area_efficiency"][key]
        for key in candidate["area_efficiency"]
    }
    return result


def by_identity(snapshot: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for sample in snapshot["samples"]:
        identity = sample_identity(sample)
        if identity in result:
            raise ValueError(f"duplicate sample identity: {identity}")
        result[identity] = sample
    return result


def exact_attention(
    baseline: dict[tuple[Any, ...], dict[str, Any]],
    candidate: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    exact_targets = dict(ADR175_EXACT_TASK_TARGETS)
    exact_tasks = set(exact_targets)
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for identity in sorted(baseline, key=lambda item: tuple(str(value) for value in item)):
        first = baseline[identity]
        second = candidate[identity]
        task = str(first["task_key"])
        if task in exact_tasks:
            grouped[(task, str(first["partition"]))].append((first, second))

    changes: list[list[Any]] = []
    censored_partitions: list[dict[str, Any]] = []
    positives_by_task = {task: 0 for task in exact_tasks}
    sample_positive_count = 0
    task_details: dict[str, dict[str, Any]] = {}
    valid_sample_count = 0
    for task in sorted(exact_tasks):
        task_details[task] = {}
        for partition in ("validation", "heldout"):
            pairs = grouped.get((task, partition), [])
            expected_count = 1 if partition == "validation" else 2
            if len(pairs) != expected_count:
                raise ValueError(
                    f"exact task requires {expected_count} {partition} samples: {task}"
                )
            source_episodes = [first.get("source_episode_index") for first, _ in pairs]
            if len(set(source_episodes)) != expected_count:
                raise ValueError(f"exact {task}/{partition} source episodes are not distinct")
            baseline_observable = [
                target_observable(first, exact_targets[task]) for first, _ in pairs
            ]
            candidate_observable = [
                target_observable(second, exact_targets[task]) for _, second in pairs
            ]
            if baseline_observable != candidate_observable:
                raise ValueError(f"exact target observability changed: {task}/{partition}")
            partition_valid = all(baseline_observable)

            old: float | None = None
            new: float | None = None
            change: float | None = None
            if partition_valid:
                old_values = [
                    resolved_attention_value(
                        first, field="conditional_selectivity", observable=True
                    )
                    for first, _ in pairs
                ]
                new_values = [
                    resolved_attention_value(
                        second, field="conditional_selectivity", observable=True
                    )
                    for _, second in pairs
                ]
                if any(value is None for value in (*old_values, *new_values)):
                    raise RuntimeError("observable exact target omitted a selectivity score")
                old = sum(float(value) for value in old_values) / len(old_values)
                new = sum(float(value) for value in new_values) / len(new_values)
                change = new - old
                changes.append([change, partition, task, old, new])
                positives_by_task[task] += int(change > 0.0)
                valid_sample_count += len(pairs)
                sample_positive_count += sum(
                    float(new_value) > float(old_value)
                    for old_value, new_value in zip(old_values, new_values, strict=True)
                )
            else:
                censored_partitions.append(
                    {
                        "partition": partition,
                        "reason": "target_not_visible_in_all_fixed_samples",
                        "sample_count": len(pairs),
                        "target_observable_count": sum(baseline_observable),
                        "task_key": task,
                    }
                )

            task_details[task][partition] = {
                "action_loss": {
                    "baseline": paired_field_mean(
                        pairs,
                        side=0,
                        field="official_action_loss",
                        context=f"{task}/{partition}",
                    ),
                    "candidate": paired_field_mean(
                        pairs,
                        side=1,
                        field="official_action_loss",
                        context=f"{task}/{partition}",
                    ),
                },
                "censored": not partition_valid,
                "conditional_selectivity": {
                    "baseline": old,
                    "candidate": new,
                    "change": change,
                },
                "entity_set_score": {
                    "baseline": sum(entity_score(first) for first, _ in pairs) / len(pairs),
                    "candidate": sum(entity_score(second) for _, second in pairs) / len(pairs),
                },
                "posterior_adoption": {
                    "baseline": resolved_attention_mean(
                        pairs,
                        side=0,
                        field="posterior_adoption",
                        observable=baseline_observable,
                    ),
                    "candidate": resolved_attention_mean(
                        pairs,
                        side=1,
                        field="posterior_adoption",
                        observable=candidate_observable,
                    ),
                },
                "sample_count": len(pairs),
                "baseline_target_resolved_count": sum(
                    first.get("target_valid") is True for first, _ in pairs
                ),
                "candidate_target_resolved_count": sum(
                    second.get("target_valid") is True for _, second in pairs
                ),
                "target_observable_count": sum(baseline_observable),
            }

    neither = sorted(task for task, count in positives_by_task.items() if count == 0)
    one = sum(count == 1 for count in positives_by_task.values())
    both = sum(count == 2 for count in positives_by_task.values())
    return {
        "best_changes": sorted(changes, reverse=True)[:8],
        "censored_partition_count": len(censored_partitions),
        "censored_partitions": censored_partitions,
        "change_unit": "task_partition_macro",
        "partition_count": len(exact_tasks) * 2,
        "partition_positive_count": sum(change[0] > 0.0 for change in changes),
        "sample_count": sum(len(pairs) for pairs in grouped.values()),
        "sample_positive_count": sample_positive_count,
        "task_count": len(exact_tasks),
        "task_details": task_details,
        "tasks_positive_both_partitions": both,
        "tasks_positive_neither_keys": neither,
        "tasks_positive_neither_partition": len(neither),
        "tasks_positive_one_partition": one,
        "valid_partition_count": len(changes),
        "valid_sample_count": valid_sample_count,
        "worst_changes": sorted(changes)[:8],
    }


def sample_improvements(
    baseline: dict[tuple[Any, ...], dict[str, Any]],
    candidate: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    action_samples = 0
    entity_samples = 0
    for identity in baseline:
        first = baseline[identity]
        second = candidate[identity]
        task = str(first["task_key"])
        partition = str(first["partition"])
        grouped[(task, partition)].append((first, second))
        action_improved = float(second["official_action_loss"]) < float(
            first["official_action_loss"]
        )
        entity_improved = entity_score(second) > entity_score(first)
        action_samples += int(action_improved)
        entity_samples += int(entity_improved)

    tasks = {task for task, _partition in grouped}
    action_positive = {task: 0 for task in tasks}
    entity_positive = {task: 0 for task in tasks}
    action_partition_positive = 0
    entity_partition_positive = 0
    for task in sorted(tasks):
        for partition in ("validation", "heldout"):
            pairs = grouped.get((task, partition), [])
            if not pairs:
                raise ValueError(f"task lacks {partition} coverage: {task}")
            baseline_action = sum(float(first["official_action_loss"]) for first, _ in pairs) / len(
                pairs
            )
            candidate_action = sum(
                float(second["official_action_loss"]) for _, second in pairs
            ) / len(pairs)
            baseline_entity = sum(entity_score(first) for first, _ in pairs) / len(pairs)
            candidate_entity = sum(entity_score(second) for _, second in pairs) / len(pairs)
            action_improved = candidate_action < baseline_action
            entity_improved = candidate_entity > baseline_entity
            action_positive[task] += int(action_improved)
            entity_positive[task] += int(entity_improved)
            action_partition_positive += int(action_improved)
            entity_partition_positive += int(entity_improved)

    return {
        "action_partition_positive_count": action_partition_positive,
        "action_sample_positive_count": action_samples,
        "action_task_both_partitions_count": sum(action_positive[task] == 2 for task in tasks),
        "entity_partition_positive_count": entity_partition_positive,
        "entity_sample_positive_count": entity_samples,
        "entity_task_both_partitions_count": sum(entity_positive[task] == 2 for task in tasks),
        "partition_count": len(grouped),
        "sample_count": len(baseline),
        "task_count": len(tasks),
    }


def operational_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
    return {
        "action_within_5pct": candidate["action_loss"] <= baseline["action_loss"] * 1.05,
        "cardinality_not_worse": (
            candidate["cardinality_abs_error"] <= baseline["cardinality_abs_error"]
        ),
        "entity_improved": candidate["entity_set_score"] > baseline["entity_set_score"],
        "entity_relative_10pct": (
            candidate["entity_set_score"] >= baseline["entity_set_score"] * 1.10
        ),
        "overlap_lt_0_98": candidate["pairwise_support_overlap"] < 0.98,
        "selectivity_gt_uniform_plus_0_002": (
            candidate["conditional_selectivity"] > 1.0 / 16.0 + 0.002
        ),
    }


def main() -> None:
    args = parse_args()
    baseline_snapshot = load_snapshot(args.baseline)
    candidate_snapshot = load_snapshot(args.candidate)
    identity_fields = (
        "arm",
        "entity_evaluation_plan_sha256",
        "evaluation_input_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "representation_split_sha256",
        "stream_plan_sha256",
    )
    for field in identity_fields:
        if baseline_snapshot.get(field) != candidate_snapshot.get(field):
            raise ValueError(f"snapshot identity changed: {field}")
    baseline_samples = by_identity(baseline_snapshot)
    candidate_samples = by_identity(candidate_snapshot)
    if baseline_samples.keys() != candidate_samples.keys():
        raise ValueError("fixed-evaluation sample identities changed")
    baseline_step = int(baseline_snapshot["checkpoint_global_step"])
    candidate_step = int(candidate_snapshot["checkpoint_global_step"])
    if candidate_step <= baseline_step:
        raise ValueError("candidate milestone must follow baseline milestone")
    if baseline_step != args.expected_baseline_step:
        raise ValueError("baseline milestone differs from the registered comparison")
    if args.expected_candidate_step is not None and candidate_step != args.expected_candidate_step:
        raise ValueError("candidate milestone differs from the registered comparison")
    steps: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for partition in ("validation", "heldout"):
        first = summary(baseline_snapshot, partition)
        second = summary(candidate_snapshot, partition)
        steps.setdefault(str(baseline_step), {})[partition] = first
        steps.setdefault(str(candidate_step), {})[partition] = second
        deltas[partition] = delta(first, second)
        gates[partition] = operational_gate(first, second)
    joint_gate = {
        "action_within_5pct_both": all(
            gates[partition]["action_within_5pct"] for partition in ("validation", "heldout")
        ),
        "cardinality_not_worse_both": all(
            gates[partition]["cardinality_not_worse"] for partition in ("validation", "heldout")
        ),
        "entity_improved_both": all(
            gates[partition]["entity_improved"] for partition in ("validation", "heldout")
        ),
        "entity_relative_10pct_any": any(
            gates[partition]["entity_relative_10pct"] for partition in ("validation", "heldout")
        ),
        "overlap_lt_0_98_both": all(
            gates[partition]["overlap_lt_0_98"] for partition in ("validation", "heldout")
        ),
        "selectivity_gt_uniform_plus_0_002_both": all(
            gates[partition]["selectivity_gt_uniform_plus_0_002"]
            for partition in ("validation", "heldout")
        ),
    }
    result = {
        "baseline_artifact_sha256": baseline_snapshot["artifact_sha256"],
        "candidate_artifact_sha256": candidate_snapshot["artifact_sha256"],
        "delta": deltas,
        "evaluation_input_sha256": baseline_snapshot["evaluation_input_sha256"],
        "exact_attention": exact_attention(baseline_samples, candidate_samples),
        "operational_gate": {"joint": joint_gate, "partitions": gates},
        "sample_improvements": sample_improvements(baseline_samples, candidate_samples),
        "steps": steps,
    }
    rendered = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="ascii")


if __name__ == "__main__":
    main()
