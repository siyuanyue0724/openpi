"""Host-neutral lifecycle-transition coverage and calibration diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from picf_next.eval.cardinality import binary_calibration_metrics, probability_distribution

_TEMPORAL_AUDIT_SCHEMAS = {
    "picf-next.molmoact2-m3-temporal-audit.v8",
    "picf-next.molmoact2-m3-temporal-audit.v9",
    "picf-next.molmoact2-m3-temporal-audit.v10",
}
_TRANSITION_KERNEL_SCHEMA = "picf-next.molmoact2-m3-temporal-audit.v9"
_HIDDEN_DURATION_BINS = (
    (1, 1),
    (2, 2),
    (3, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, None),
)


def _hidden_duration_bin(duration: int) -> str:
    if duration <= 0:
        raise ValueError("hidden duration must be positive")
    for lower, upper in _HIDDEN_DURATION_BINS:
        if upper is None or duration <= upper:
            return (
                f"{lower}+"
                if upper is None
                else (str(lower) if lower == upper else f"{lower}-{upper}")
            )
    raise RuntimeError("hidden-duration bins do not cover positive integers")


def _duration_hazard_summary(
    at_risk_by_duration: Mapping[int, int],
    event_by_duration: Mapping[int, int],
) -> dict[str, Any]:
    exact = {
        str(duration): {
            "at_risk_count": at_risk_by_duration[duration],
            "reappearance_count": event_by_duration.get(duration, 0),
            "remained_hidden_count": (
                at_risk_by_duration[duration] - event_by_duration.get(duration, 0)
            ),
            "reappearance_hazard": (
                event_by_duration.get(duration, 0) / at_risk_by_duration[duration]
            ),
        }
        for duration in sorted(at_risk_by_duration)
    }
    binned_counts: dict[str, dict[str, int]] = {
        _hidden_duration_bin(lower): {"at_risk_count": 0, "reappearance_count": 0}
        for lower, _upper in _HIDDEN_DURATION_BINS
    }
    for duration, at_risk_count in at_risk_by_duration.items():
        name = _hidden_duration_bin(duration)
        binned_counts[name]["at_risk_count"] += at_risk_count
        binned_counts[name]["reappearance_count"] += event_by_duration.get(duration, 0)
    binned = {
        name: {
            **counts,
            "remained_hidden_count": (counts["at_risk_count"] - counts["reappearance_count"]),
            "reappearance_hazard": (
                counts["reappearance_count"] / counts["at_risk_count"]
                if counts["at_risk_count"]
                else None
            ),
        }
        for name, counts in binned_counts.items()
    }
    return {
        "exact_by_elapsed_hidden_frames": exact,
        "binned_by_elapsed_hidden_frames": binned,
    }


def _require_probability(value: Any, name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return probability


def _calibration_or_none(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> dict[str, Any] | None:
    if not probabilities:
        return None
    return binary_calibration_metrics(probabilities, labels)


def partition_contiguous_visibility_targets(
    global_indices: Sequence[int],
    visibility_by_global_index: Mapping[int, Mapping[str, int | None]],
) -> tuple[tuple[Mapping[str, int | None], ...], ...]:
    """Partition source targets without treating missing frames as transitions."""

    indices = tuple(global_indices)
    if not indices or any(
        not isinstance(index, int) or isinstance(index, bool) for index in indices
    ):
        raise ValueError("visibility target indices must be non-empty integers")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("visibility target indices must be unique and increasing")
    missing = tuple(index for index in indices if index not in visibility_by_global_index)
    if missing:
        raise ValueError(f"visibility targets are missing source indices: {missing[:3]}")

    sequences: list[list[Mapping[str, int | None]]] = []
    current: list[Mapping[str, int | None]] = []
    previous_global_index: int | None = None
    for global_index in indices:
        if previous_global_index is None or global_index == previous_global_index + 1:
            current.append(visibility_by_global_index[global_index])
        else:
            sequences.append(current)
            current = [visibility_by_global_index[global_index]]
        previous_global_index = global_index
    if current:
        sequences.append(current)
    return tuple(tuple(sequence) for sequence in sequences)


def audit_visibility_target_sequences(
    sequences: Sequence[Sequence[Mapping[str, int | None]]],
) -> dict[str, Any]:
    """Measure identifiability of a loss-only visibility transition target.

    Each inner sequence is one uninterrupted episode. Mapping values are binary
    visibility labels, or ``None`` when the object is deliberately
    unsupervised. Direct transitions require adjacent supervised frames. The
    next-supervised census additionally bridges unknown intervals without
    inventing labels inside them. This is a target-side audit: model
    predictions and runtime association never enter the calculation.
    """

    frozen_sequences = tuple(tuple(frame for frame in sequence) for sequence in sequences)
    if not frozen_sequences or any(not sequence for sequence in frozen_sequences):
        raise ValueError("visibility target audit requires non-empty sequences")

    transition_names = ("0->0", "0->1", "1->0", "1->1")
    transition_count: defaultdict[str, int] = defaultdict(int)
    transition_by_identity: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    next_supervised_transition_count: defaultdict[str, int] = defaultdict(int)
    next_supervised_transition_by_identity: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    bridged_transition_count: defaultdict[str, int] = defaultdict(int)
    bridged_transition_by_identity: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    elapsed_steps_by_transition: defaultdict[str, list[float]] = defaultdict(list)
    bridged_unknown_lengths: list[float] = []
    label_count: defaultdict[str, int] = defaultdict(int)
    hidden_run_lengths: list[float] = []
    reacquired_hidden_run_lengths: list[float] = []
    right_censored_hidden_run_lengths: list[float] = []
    unknown_censored_hidden_run_lengths: list[float] = []
    death_terminated_hidden_run_lengths: list[float] = []
    hidden_at_risk_by_duration: defaultdict[int, int] = defaultdict(int)
    hidden_reappearance_by_duration: defaultdict[int, int] = defaultdict(int)
    seen_reacquired_hidden_run_lengths: list[float] = []
    seen_right_censored_hidden_run_lengths: list[float] = []
    seen_unknown_censored_hidden_run_lengths: list[float] = []
    seen_death_terminated_hidden_run_lengths: list[float] = []
    seen_hidden_at_risk_by_duration: defaultdict[int, int] = defaultdict(int)
    seen_hidden_reappearance_by_duration: defaultdict[int, int] = defaultdict(int)
    supervised_rows = 0
    unsupervised_rows = 0
    adjacent_identity_pairs = 0
    supervised_transition_pairs = 0

    for sequence in frozen_sequences:
        previous: Mapping[str, int | None] | None = None
        hidden_run_by_identity: defaultdict[str, int] = defaultdict(int)
        hidden_run_was_seen_by_identity: dict[str, bool] = {}
        previously_seen_identities: set[str] = set()
        last_supervised_by_identity: dict[str, tuple[int, int]] = {}
        for frame_index, frame in enumerate(sequence):
            if not isinstance(frame, Mapping) or not frame:
                raise ValueError("visibility target frames must be non-empty mappings")
            normalized: dict[str, int | None] = {}
            for identity, label in frame.items():
                if not isinstance(identity, str) or not identity:
                    raise ValueError("visibility target identities must be non-empty strings")
                if label is not None and (isinstance(label, bool) or label not in (0, 1)):
                    raise ValueError("visibility target labels must be 0, 1 or None")
                normalized[identity] = label
                if label is None:
                    unsupervised_rows += 1
                else:
                    supervised_rows += 1
                    label_count[str(label)] += 1

            for identity in tuple(last_supervised_by_identity):
                if identity not in normalized:
                    del last_supervised_by_identity[identity]
            for identity, label in normalized.items():
                if label is None:
                    continue
                last_supervised = last_supervised_by_identity.get(identity)
                if last_supervised is not None:
                    previous_label, previous_frame_index = last_supervised
                    elapsed_steps = frame_index - previous_frame_index
                    transition = f"{previous_label}->{label}"
                    next_supervised_transition_count[transition] += 1
                    next_supervised_transition_by_identity[identity][transition] += 1
                    elapsed_steps_by_transition[transition].append(float(elapsed_steps))
                    if elapsed_steps > 1:
                        bridged_transition_count[transition] += 1
                        bridged_transition_by_identity[identity][transition] += 1
                        bridged_unknown_lengths.append(float(elapsed_steps - 1))
                last_supervised_by_identity[identity] = (label, frame_index)

            if previous is not None:
                for identity, duration in tuple(hidden_run_by_identity.items()):
                    was_seen = hidden_run_was_seen_by_identity[identity]
                    if identity not in normalized:
                        death_terminated_hidden_run_lengths.append(float(duration))
                        if was_seen:
                            seen_death_terminated_hidden_run_lengths.append(float(duration))
                        continue
                    current_label = normalized[identity]
                    if current_label is None:
                        unknown_censored_hidden_run_lengths.append(float(duration))
                        if was_seen:
                            seen_unknown_censored_hidden_run_lengths.append(float(duration))
                        continue
                    hidden_at_risk_by_duration[duration] += 1
                    if was_seen:
                        seen_hidden_at_risk_by_duration[duration] += 1
                    if current_label == 1:
                        hidden_reappearance_by_duration[duration] += 1
                        reacquired_hidden_run_lengths.append(float(duration))
                        if was_seen:
                            seen_hidden_reappearance_by_duration[duration] += 1
                            seen_reacquired_hidden_run_lengths.append(float(duration))

            previously_seen_identities.intersection_update(normalized)

            for identity in hidden_run_by_identity.keys() | normalized.keys():
                label = normalized.get(identity)
                if label == 0:
                    if identity not in hidden_run_by_identity:
                        hidden_run_was_seen_by_identity[identity] = (
                            identity in previously_seen_identities
                        )
                    hidden_run_by_identity[identity] = hidden_run_by_identity.get(identity, 0) + 1
                else:
                    completed_run = hidden_run_by_identity.pop(identity, 0)
                    hidden_run_was_seen_by_identity.pop(identity, None)
                    if completed_run:
                        hidden_run_lengths.append(float(completed_run))
                if label == 1:
                    previously_seen_identities.add(identity)
                elif identity not in normalized:
                    previously_seen_identities.discard(identity)

            if previous is not None:
                shared = previous.keys() & normalized.keys()
                adjacent_identity_pairs += len(shared)
                for identity in shared:
                    prior_label = previous[identity]
                    current_label = normalized[identity]
                    if prior_label is None or current_label is None:
                        continue
                    transition = f"{prior_label}->{current_label}"
                    transition_count[transition] += 1
                    transition_by_identity[identity][transition] += 1
                    supervised_transition_pairs += 1
            previous = normalized

        hidden_run_lengths.extend(
            float(value) for value in hidden_run_by_identity.values() if value
        )
        right_censored_hidden_run_lengths.extend(
            float(value) for value in hidden_run_by_identity.values() if value
        )
        seen_right_censored_hidden_run_lengths.extend(
            float(value)
            for identity, value in hidden_run_by_identity.items()
            if value and hidden_run_was_seen_by_identity[identity]
        )

    all_hidden_hazard = _duration_hazard_summary(
        hidden_at_risk_by_duration,
        hidden_reappearance_by_duration,
    )
    seen_hidden_hazard = _duration_hazard_summary(
        seen_hidden_at_risk_by_duration,
        seen_hidden_reappearance_by_duration,
    )

    return {
        "schema": "picf-next.visibility-target-transition-census.v2",
        "sequence_count": len(frozen_sequences),
        "frame_occurrence_count": sum(len(sequence) for sequence in frozen_sequences),
        "supervised_row_count": supervised_rows,
        "unsupervised_row_count": unsupervised_rows,
        "adjacent_identity_pair_count": adjacent_identity_pairs,
        "supervised_transition_pair_count": supervised_transition_pairs,
        "label_count": {name: label_count[name] for name in ("0", "1")},
        "transition_count": {name: transition_count[name] for name in transition_names},
        "transition_count_by_identity": {
            identity: {name: transition_by_identity[identity][name] for name in transition_names}
            for identity in sorted(transition_by_identity)
        },
        "next_supervised_transition_count": {
            name: next_supervised_transition_count[name] for name in transition_names
        },
        "next_supervised_transition_count_by_identity": {
            identity: {
                name: next_supervised_transition_by_identity[identity][name]
                for name in transition_names
            }
            for identity in sorted(next_supervised_transition_by_identity)
        },
        "next_supervised_elapsed_steps_by_transition": {
            name: probability_distribution(elapsed_steps_by_transition[name])
            for name in transition_names
        },
        "bridged_transition_count": {
            name: bridged_transition_count[name] for name in transition_names
        },
        "bridged_transition_count_by_identity": {
            identity: {
                name: bridged_transition_by_identity[identity][name] for name in transition_names
            }
            for identity in sorted(bridged_transition_by_identity)
        },
        "bridged_unknown_run_length": probability_distribution(bridged_unknown_lengths),
        "hidden_run_length": probability_distribution(hidden_run_lengths),
        "hidden_reappearance_hazard": {
            "definition": (
                "P(visible at next supervised alive frame | hidden for exactly d "
                "consecutive supervised frames)"
            ),
            **all_hidden_hazard,
            "reacquired_run_length": probability_distribution(reacquired_hidden_run_lengths),
            "right_censored_run_length": probability_distribution(
                right_censored_hidden_run_lengths
            ),
            "unknown_censored_run_length": probability_distribution(
                unknown_censored_hidden_run_lengths
            ),
            "death_terminated_run_length": probability_distribution(
                death_terminated_hidden_run_lengths
            ),
            "seen_then_hidden": {
                "definition": (
                    "same hazard restricted to runs whose identity had a trustworthy "
                    "visible measurement earlier in the contiguous source sequence"
                ),
                **seen_hidden_hazard,
                "reacquired_run_length": probability_distribution(
                    seen_reacquired_hidden_run_lengths
                ),
                "right_censored_run_length": probability_distribution(
                    seen_right_censored_hidden_run_lengths
                ),
                "unknown_censored_run_length": probability_distribution(
                    seen_unknown_censored_hidden_run_lengths
                ),
                "death_terminated_run_length": probability_distribution(
                    seen_death_terminated_hidden_run_lengths
                ),
            },
        },
    }


def audit_partitioned_visibility_target_sequences(
    sequences_by_partition: Mapping[
        str,
        Sequence[Sequence[Mapping[str, int | None]]],
    ],
) -> dict[str, dict[str, Any]]:
    """Audit frozen data partitions independently instead of pooling labels."""

    if not isinstance(sequences_by_partition, Mapping) or not sequences_by_partition:
        raise ValueError("partitioned visibility audit requires non-empty partitions")
    output: dict[str, dict[str, Any]] = {}
    for partition in sorted(sequences_by_partition):
        if not isinstance(partition, str) or not partition:
            raise ValueError("visibility partition names must be non-empty strings")
        sequences = sequences_by_partition[partition]
        if not sequences:
            raise ValueError(f"visibility partition {partition!r} has no sequences")
        output[partition] = audit_visibility_target_sequences(sequences)
    return output


def audit_lifecycle_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure deployed conditional-detection coverage from temporal replays.

    Only rows whose loss-side physical key is unchanged across the transition
    are used. Audits v8 and v10 record the identifiable direct-detection model;
    legacy v9 additionally records the rejected persistence/reappearance
    parameterization so historical evidence remains readable. Excluding
    identity changes prevents a diagnostic row swap from being mistaken for
    sensor-model supervision.
    """

    if not reports:
        raise ValueError("lifecycle audit requires at least one temporal report")

    frames: list[Mapping[str, Any]] = []
    seen_rank_steps: set[tuple[int, int]] = set()
    checkpoint_revisions: set[str] = set()
    checkpoint_hashes: set[str] = set()
    report_schemas: set[str] = set()
    for report in reports:
        report_schema = report.get("schema")
        if report_schema not in _TEMPORAL_AUDIT_SCHEMAS:
            raise ValueError("unsupported temporal audit schema")
        report_schemas.add(str(report_schema))
        checkpoint_revision = report.get("checkpoint_code_revision")
        checkpoint_hash = report.get("checkpoint_model_sha256")
        if not isinstance(checkpoint_revision, str) or not checkpoint_revision:
            raise ValueError("temporal report has no checkpoint code revision")
        if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
            raise ValueError("temporal report has no checkpoint model hash")
        checkpoint_revisions.add(checkpoint_revision)
        checkpoint_hashes.add(checkpoint_hash)
        rows = report.get("rows")
        if not isinstance(rows, list):
            raise ValueError("temporal report rows must be a list")
        for frame in rows:
            if not isinstance(frame, Mapping):
                raise ValueError("temporal report frame must be a mapping")
            rank = frame.get("rank")
            step = frame.get("step")
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank < 0
                or not isinstance(step, int)
                or isinstance(step, bool)
                or step <= 0
            ):
                raise ValueError("temporal report rank/step is malformed")
            rank_step = rank, step
            if rank_step in seen_rank_steps:
                raise ValueError("temporal reports contain duplicate rank/step rows")
            seen_rank_steps.add(rank_step)
            frames.append(frame)
    if len(checkpoint_revisions) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("lifecycle reports must describe one exact checkpoint")
    if len(report_schemas) != 1:
        raise ValueError("lifecycle reports must share one temporal audit schema")
    temporal_audit_schema = next(iter(report_schemas))
    transition_kernel_available = temporal_audit_schema == _TRANSITION_KERNEL_SCHEMA

    frames.sort(key=lambda frame: (int(frame["rank"]), int(frame["step"])))
    previous_frame_by_rank: dict[int, tuple[int, str, int]] = {}
    episode_ordinal_by_rank: defaultdict[int, int] = defaultdict(lambda: -1)
    observations: list[dict[str, Any]] = []
    identity_changed_rows = 0
    rows_without_prior_identity = 0
    rows_with_negligible_prior_existence = 0
    unsupervised_rows = 0

    for frame in frames:
        rank = int(frame["rank"])
        step = int(frame["step"])
        episode_key = frame.get("episode_key")
        episode_reset = frame.get("episode_reset")
        if not isinstance(episode_key, str) or not episode_key:
            raise ValueError("temporal report episode key is malformed")
        if not isinstance(episode_reset, bool):
            raise ValueError("temporal report episode_reset must be boolean")
        previous = previous_frame_by_rank.get(rank)
        discontinuous = (
            previous is None
            or step != previous[0] + 1
            or episode_key != previous[1]
            or episode_reset
        )
        if discontinuous:
            episode_ordinal_by_rank[rank] += 1
        previous_frame_by_rank[rank] = step, episode_key, episode_ordinal_by_rank[rank]

        row_traces = frame.get("row_traces")
        if not isinstance(row_traces, list):
            raise ValueError("temporal report row_traces must be a list")
        seen_prior_keys: set[str] = set()
        for trace in row_traces:
            if not isinstance(trace, Mapping):
                raise ValueError("temporal row trace must be a mapping")
            prior_key = trace.get("prior_key")
            identity_key = trace.get("identity_key")
            if prior_key is None:
                rows_without_prior_identity += 1
                continue
            if not isinstance(prior_key, str) or not prior_key:
                raise ValueError("prior identity key must be a nonempty string or null")
            if prior_key in seen_prior_keys:
                raise ValueError("a physical prior key appears in multiple rows of one frame")
            seen_prior_keys.add(prior_key)
            if identity_key != prior_key:
                identity_changed_rows += 1
                continue

            existence = _require_probability(
                trace.get("prior_existence_probability"),
                "prior existence probability",
            )
            visibility = _require_probability(
                trace.get("prior_visibility_probability"),
                "prior visibility probability",
            )
            if visibility > existence + 1e-5:
                raise ValueError("joint visibility exceeds prior existence")
            if existence <= 1e-7:
                rows_with_negligible_prior_existence += 1
                continue
            detection = min(max(visibility / existence, 0.0), 1.0)
            transition_kernel: dict[str, float] = {}
            if transition_kernel_available:
                reported_detection = _require_probability(
                    trace.get("prior_conditional_detection_probability"),
                    "reported conditional detection probability",
                )
                if abs(reported_detection - detection) > 2e-5:
                    raise ValueError("reported conditional detection disagrees with belief ratio")
                previous_visibility = _require_probability(
                    trace.get("previous_conditional_visibility_probability"),
                    "previous conditional visibility probability",
                )
                persistence = _require_probability(
                    trace.get("visibility_persistence_probability"),
                    "visibility persistence probability",
                )
                reappearance = _require_probability(
                    trace.get("visibility_reappearance_probability"),
                    "visibility reappearance probability",
                )
                reconstructed = (
                    previous_visibility * persistence + (1.0 - previous_visibility) * reappearance
                )
                transition_kernel = {
                    "previous_visibility_probability": previous_visibility,
                    "visibility_persistence_probability": persistence,
                    "visibility_reappearance_probability": reappearance,
                    "mixture_residual": abs(reconstructed - reported_detection),
                }

            supervised = trace.get("target_visibility_supervised")
            if not isinstance(supervised, bool):
                raise ValueError("target visibility supervision must be boolean")
            target = trace.get("target_visibility")
            if not supervised:
                unsupervised_rows += 1
                continue
            target_probability = _require_probability(target, "target visibility")
            if target_probability not in (0.0, 1.0):
                raise ValueError("lifecycle visibility target must be binary")
            observations.append(
                {
                    "rank": rank,
                    "episode_ordinal": episode_ordinal_by_rank[rank],
                    "episode_key": episode_key,
                    "step": step,
                    "identity_key": prior_key,
                    "label": int(target_probability),
                    "detection_probability": detection,
                    **transition_kernel,
                }
            )

    probabilities = [row["detection_probability"] for row in observations]
    labels = [row["label"] for row in observations]
    grouped: defaultdict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(row["rank"], row["episode_ordinal"], row["identity_key"])].append(row)

    transition_probabilities: defaultdict[str, list[float]] = defaultdict(list)
    transition_kernel_probabilities: defaultdict[str, list[float]] = defaultdict(list)
    transition_kernel_labels: defaultdict[str, list[int]] = defaultdict(list)
    hidden_run_lengths: list[float] = []
    for sequence in grouped.values():
        sequence.sort(key=lambda row: row["step"])
        hidden_run = 0
        previous: dict[str, Any] | None = None
        for row in sequence:
            if row["label"] == 0:
                if previous is not None and row["step"] == previous["step"] + 1:
                    hidden_run = hidden_run + 1 if previous["label"] == 0 else 1
                else:
                    if hidden_run:
                        hidden_run_lengths.append(float(hidden_run))
                    hidden_run = 1
            elif hidden_run:
                hidden_run_lengths.append(float(hidden_run))
                hidden_run = 0

            if previous is not None and row["step"] == previous["step"] + 1:
                transition = f"{previous['label']}->{row['label']}"
                transition_probabilities[transition].append(row["detection_probability"])
                if transition_kernel_available:
                    origin = "visible_origin" if previous["label"] == 1 else "hidden_origin"
                    probability_name = (
                        "visibility_persistence_probability"
                        if previous["label"] == 1
                        else "visibility_reappearance_probability"
                    )
                    transition_kernel_probabilities[origin].append(row[probability_name])
                    transition_kernel_labels[origin].append(row["label"])
            elif hidden_run > 1:
                raise RuntimeError("hidden-run accounting crossed a discontinuity")
            previous = row
        if hidden_run:
            hidden_run_lengths.append(float(hidden_run))

    transition_names = ("0->0", "0->1", "1->0", "1->1")
    return {
        "schema": "picf-next.m3-lifecycle-calibration-audit.v2",
        "temporal_audit_schema": temporal_audit_schema,
        "checkpoint_code_revision": next(iter(checkpoint_revisions)),
        "checkpoint_model_sha256": next(iter(checkpoint_hashes)),
        "report_count": len(reports),
        "frame_count": len(frames),
        "supervised_row_count": len(observations),
        "unsupervised_row_count": unsupervised_rows,
        "identity_changed_row_count": identity_changed_rows,
        "row_without_prior_identity_count": rows_without_prior_identity,
        "negligible_prior_existence_row_count": rows_with_negligible_prior_existence,
        "conditional_detection_calibration": _calibration_or_none(probabilities, labels),
        "transition_count": {
            name: len(transition_probabilities[name]) for name in transition_names
        },
        "transition_detection_probability": {
            name: probability_distribution(transition_probabilities[name])
            for name in transition_names
        },
        "hidden_run_length": probability_distribution(hidden_run_lengths),
        "visibility_transition_kernel": (
            {
                "previous_visibility_probability": probability_distribution(
                    [row["previous_visibility_probability"] for row in observations]
                ),
                "visibility_persistence_probability": probability_distribution(
                    [row["visibility_persistence_probability"] for row in observations]
                ),
                "visibility_reappearance_probability": probability_distribution(
                    [row["visibility_reappearance_probability"] for row in observations]
                ),
                "maximum_mixture_residual": max(
                    (row["mixture_residual"] for row in observations),
                    default=0.0,
                ),
                "visible_origin_calibration": _calibration_or_none(
                    transition_kernel_probabilities["visible_origin"],
                    transition_kernel_labels["visible_origin"],
                ),
                "hidden_origin_calibration": _calibration_or_none(
                    transition_kernel_probabilities["hidden_origin"],
                    transition_kernel_labels["hidden_origin"],
                ),
            }
            if transition_kernel_available
            else None
        ),
    }
