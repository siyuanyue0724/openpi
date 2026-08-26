"""Loss-side diagnostics for task-to-object-row binding.

These reports observe the same post-forward Hungarian assignment used by the
structural objective. They never alter model inputs, recurrent state, attention,
loss targets, or optimizer gradients.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from picf_next.lingbot_native.calvin_objective import NativeCALVINObjectiveResult
from picf_next.lingbot_native.supervision import (
    assignment_binding_start_phase,
    assignment_binding_valid_at_phase,
    assignment_row_to_track_at_phase,
    materialize_row_task_supervision,
)

TASK_ROW_DIAGNOSTIC_SCHEMA = "picf-next.lingbot-native-task-row-diagnostic.v2"

_FIELDS = frozenset(
    {
        "schema",
        "exact_task",
        "identity_keys",
        "track_task_targets",
        "track_task_valid",
        "capacity_censored",
        "sequence_time_count",
        "source_time",
        "source_side",
        "source_phase",
        "binding_start_phase",
        "source_binding_valid",
        "row_to_track",
        "assignment_sha256",
        "row_task_targets",
        "row_task_valid",
        "task_logits",
        "task_probabilities",
        "target_rows",
        "target_identity_keys",
        "materialized_target_identity_keys",
        "unmaterialized_target_identity_keys",
        "known_negative_rows",
        "minimum_target_logit",
        "maximum_known_negative_logit",
        "target_vs_hardest_negative_logit_margin",
        "minimum_target_probability",
        "maximum_known_negative_probability",
        "target_vs_hardest_negative_probability_margin",
        "worst_target_rank",
        "all_targets_beat_known_negatives",
    }
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1 / (1 + inverse)
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite_float(value: object, *, name: str) -> float | None:
    return None if value is None else _finite_float(value, name=name)


def _strict_target_rank(logits: list[float], valid: list[bool], row: int) -> int:
    """Return a pessimistic rank: tied alternatives count ahead of the target."""

    target = logits[row]
    return 1 + sum(
        int(index != row and is_valid and value >= target)
        for index, (value, is_valid) in enumerate(zip(logits, valid, strict=True))
    )


def _derived_statistics(
    *,
    logits: list[float],
    probabilities: list[float],
    targets: list[float],
    valid: list[bool],
) -> dict[str, object]:
    target_rows = [
        index
        for index, (target, is_valid) in enumerate(zip(targets, valid, strict=True))
        if is_valid and target > 0
    ]
    negative_rows = [
        index
        for index, (target, is_valid) in enumerate(zip(targets, valid, strict=True))
        if is_valid and target == 0
    ]
    if not target_rows or not negative_rows:
        return {
            "target_rows": target_rows,
            "known_negative_rows": negative_rows,
            "minimum_target_logit": None,
            "maximum_known_negative_logit": None,
            "target_vs_hardest_negative_logit_margin": None,
            "minimum_target_probability": None,
            "maximum_known_negative_probability": None,
            "target_vs_hardest_negative_probability_margin": None,
            "worst_target_rank": None,
            "all_targets_beat_known_negatives": None,
        }

    minimum_target_logit = min(logits[index] for index in target_rows)
    maximum_negative_logit = max(logits[index] for index in negative_rows)
    minimum_target_probability = min(probabilities[index] for index in target_rows)
    maximum_negative_probability = max(probabilities[index] for index in negative_rows)
    return {
        "target_rows": target_rows,
        "known_negative_rows": negative_rows,
        "minimum_target_logit": minimum_target_logit,
        "maximum_known_negative_logit": maximum_negative_logit,
        "target_vs_hardest_negative_logit_margin": (minimum_target_logit - maximum_negative_logit),
        "minimum_target_probability": minimum_target_probability,
        "maximum_known_negative_probability": maximum_negative_probability,
        "target_vs_hardest_negative_probability_margin": (
            minimum_target_probability - maximum_negative_probability
        ),
        "worst_target_rank": max(_strict_target_rank(logits, valid, row) for row in target_rows),
        "all_targets_beat_known_negatives": minimum_target_logit > maximum_negative_logit,
    }


def build_task_row_diagnostics(
    objective: NativeCALVINObjectiveResult,
) -> tuple[dict[str, object], ...]:
    """Materialize one recomputable diagnostic per batch item."""

    if not isinstance(objective, NativeCALVINObjectiveResult):
        raise TypeError("task-row diagnostics require one native CALVIN objective")
    predictions = objective.predictions
    targets = objective.targets
    assignment = objective.assignment
    batch, rows = predictions.task_relevance_logits.shape
    sequence_time_count = targets.masks.shape[1]
    source_time = sequence_time_count - 1
    source_phase = 2 * source_time + 1
    binding_start_phase = assignment_binding_start_phase(assignment, targets)
    source_binding_valid = assignment_binding_valid_at_phase(
        assignment,
        targets,
        source_phase=source_phase,
    )
    source_row_to_track = assignment_row_to_track_at_phase(
        assignment,
        targets,
        source_phase=source_phase,
    )
    if len(objective.track_identity_keys_by_batch) != batch:
        raise ValueError("task-row identities and predictions have different batches")

    reports: list[dict[str, object]] = []
    for batch_index in range(batch):
        row_task = materialize_row_task_supervision(
            targets,
            assignment,
            batch_index=batch_index,
            dtype=predictions.task_relevance_logits.dtype,
            binding_valid=source_binding_valid[batch_index],
        )
        logits = predictions.task_relevance_logits[batch_index].detach().float().cpu().tolist()
        probabilities = [_sigmoid(float(value)) for value in logits]
        row_to_track = source_row_to_track[batch_index].detach().cpu().tolist()
        row_binding_start_phase = binding_start_phase[batch_index].detach().cpu().tolist()
        row_source_binding_valid = source_binding_valid[batch_index].detach().cpu().tolist()
        task_targets = row_task.target.detach().float().cpu().tolist()
        task_valid = row_task.valid.detach().cpu().tolist()
        identity_keys = list(objective.track_identity_keys_by_batch[batch_index])
        track_limit = len(identity_keys)
        track_task_targets = (
            targets.task_relevance[batch_index, :track_limit].detach().float().cpu().tolist()
        )
        track_task_valid = targets.task_valid[batch_index, :track_limit].detach().cpu().tolist()
        capacity_censored = (
            targets.capacity_censored[batch_index, :track_limit].detach().cpu().tolist()
        )
        if not (
            len(logits)
            == len(probabilities)
            == len(row_to_track)
            == len(task_targets)
            == len(task_valid)
            == rows
        ):
            raise RuntimeError("task-row diagnostic axes changed during materialization")
        statistics = _derived_statistics(
            logits=[float(value) for value in logits],
            probabilities=probabilities,
            targets=[float(value) for value in task_targets],
            valid=[bool(value) for value in task_valid],
        )
        target_rows = statistics["target_rows"]
        if not isinstance(target_rows, list):
            raise RuntimeError("derived task rows are malformed")
        target_identity_keys = [
            identity_keys[index]
            for index, (target, is_valid) in enumerate(
                zip(track_task_targets, track_task_valid, strict=True)
            )
            if is_valid and target > 0
        ]
        materialized_target_identity_keys = []
        for row in target_rows:
            track = row_to_track[row]
            if track < 0 or track >= len(identity_keys):
                raise RuntimeError("positive task row lacks one assigned physical identity")
            materialized_target_identity_keys.append(identity_keys[track])
        materialized_target_identity_set = set(materialized_target_identity_keys)
        unmaterialized_target_identity_keys = [
            key for key in target_identity_keys if key not in materialized_target_identity_set
        ]
        report: dict[str, object] = {
            "schema": TASK_ROW_DIAGNOSTIC_SCHEMA,
            "exact_task": row_task.exact_task,
            "identity_keys": identity_keys,
            "track_task_targets": track_task_targets,
            "track_task_valid": track_task_valid,
            "capacity_censored": capacity_censored,
            "sequence_time_count": sequence_time_count,
            "source_time": source_time,
            "source_side": "posterior",
            "source_phase": source_phase,
            "binding_start_phase": row_binding_start_phase,
            "source_binding_valid": row_source_binding_valid,
            "row_to_track": row_to_track,
            "assignment_sha256": _canonical_sha256(
                {
                    "identity_keys": identity_keys,
                    "sequence_time_count": sequence_time_count,
                    "source_phase": source_phase,
                    "binding_start_phase": row_binding_start_phase,
                    "source_binding_valid": row_source_binding_valid,
                    "row_to_track": row_to_track,
                }
            ),
            "row_task_targets": task_targets,
            "row_task_valid": task_valid,
            "task_logits": logits,
            "task_probabilities": probabilities,
            "target_identity_keys": target_identity_keys,
            "materialized_target_identity_keys": materialized_target_identity_keys,
            "unmaterialized_target_identity_keys": unmaterialized_target_identity_keys,
            **statistics,
        }
        validate_task_row_diagnostic(report)
        reports.append(report)
    return tuple(reports)


def validate_task_row_diagnostic(value: object) -> dict[str, Any]:
    """Recompute all task-row diagnostic fields from persisted primitive arrays."""

    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("task-row diagnostic fields differ from schema")
    if value["schema"] != TASK_ROW_DIAGNOSTIC_SCHEMA:
        raise ValueError("task-row diagnostic schema changed")
    if not isinstance(value["exact_task"], bool):
        raise TypeError("task-row exactness must be boolean")

    identity_keys = value["identity_keys"]
    if (
        not isinstance(identity_keys, list)
        or not identity_keys
        or len(set(identity_keys)) != len(identity_keys)
        or any(not isinstance(item, str) or not item for item in identity_keys)
    ):
        raise ValueError("task-row identity keys must be unique non-empty strings")
    row_arrays = (
        value["binding_start_phase"],
        value["source_binding_valid"],
        value["row_to_track"],
        value["row_task_targets"],
        value["row_task_valid"],
        value["task_logits"],
        value["task_probabilities"],
    )
    if any(not isinstance(item, list) or not item for item in row_arrays):
        raise ValueError("task-row diagnostic arrays must be non-empty lists")
    lengths = {len(item) for item in row_arrays}
    if len(lengths) != 1:
        raise ValueError("task-row diagnostic arrays have different row counts")
    sequence_time_count = value["sequence_time_count"]
    source_time = value["source_time"]
    source_phase = value["source_phase"]
    if (
        isinstance(sequence_time_count, bool)
        or not isinstance(sequence_time_count, int)
        or sequence_time_count <= 0
        or isinstance(source_time, bool)
        or not isinstance(source_time, int)
        or source_time != sequence_time_count - 1
        or value["source_side"] != "posterior"
        or isinstance(source_phase, bool)
        or not isinstance(source_phase, int)
        or source_phase != 2 * source_time + 1
    ):
        raise ValueError("task-row source cut is not the final posterior")
    binding_start_phase = value["binding_start_phase"]
    source_binding_valid = value["source_binding_valid"]
    terminal_phase = 2 * sequence_time_count
    if any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= terminal_phase
        for item in binding_start_phase
    ):
        raise ValueError("task-row binding phases lie outside the sequence")
    if any(not isinstance(item, bool) for item in source_binding_valid):
        raise TypeError("task-row source binding validity must be boolean")
    track_arrays = (
        value["track_task_targets"],
        value["track_task_valid"],
        value["capacity_censored"],
    )
    if any(not isinstance(item, list) or len(item) != len(identity_keys) for item in track_arrays):
        raise ValueError("task-row track arrays must match the physical identity axis")

    row_to_track = value["row_to_track"]
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item < -1
        or item >= len(identity_keys)
        for item in row_to_track
    ):
        raise ValueError("task-row assignment references an invalid physical track")
    assigned = [item for item in row_to_track if item >= 0]
    if len(set(assigned)) != len(assigned):
        raise ValueError("task-row assignment maps two rows to one physical track")
    expected_source_binding_valid = [
        track >= 0 and phase <= source_phase
        for track, phase in zip(row_to_track, binding_start_phase, strict=True)
    ]
    if source_binding_valid != expected_source_binding_valid:
        raise ValueError("task-row source binding validity was not recomputed")
    if any(
        (not valid) and track >= 0
        for valid, track in zip(source_binding_valid, row_to_track, strict=True)
    ):
        raise ValueError("task-row source assignment contains a future identity")
    expected_assignment_sha256 = _canonical_sha256(
        {
            "identity_keys": identity_keys,
            "sequence_time_count": sequence_time_count,
            "source_phase": source_phase,
            "binding_start_phase": binding_start_phase,
            "source_binding_valid": source_binding_valid,
            "row_to_track": row_to_track,
        }
    )
    if value["assignment_sha256"] != expected_assignment_sha256:
        raise ValueError("task-row assignment digest was not recomputed")

    track_targets = [
        _finite_float(item, name="task-row track target") for item in value["track_task_targets"]
    ]
    if any(not 0 <= item <= 1 for item in track_targets):
        raise ValueError("task-row track targets must lie in [0,1]")
    track_valid = value["track_task_valid"]
    censored = value["capacity_censored"]
    if any(not isinstance(item, bool) for item in (*track_valid, *censored)):
        raise TypeError("task-row track validity and censoring must be boolean")

    targets = [_finite_float(item, name="task-row target") for item in value["row_task_targets"]]
    if any(not 0 <= item <= 1 for item in targets):
        raise ValueError("task-row targets must lie in [0,1]")
    valid = value["row_task_valid"]
    if any(not isinstance(item, bool) for item in valid):
        raise TypeError("task-row validity must be boolean")
    logits = [_finite_float(item, name="task-row logit") for item in value["task_logits"]]
    probabilities = [
        _finite_float(item, name="task-row probability") for item in value["task_probabilities"]
    ]
    expected_probabilities = [_sigmoid(item) for item in logits]
    if any(
        not 0 <= measured <= 1 or not math.isclose(measured, expected, rel_tol=1e-7, abs_tol=1e-7)
        for measured, expected in zip(probabilities, expected_probabilities, strict=True)
    ):
        raise ValueError("task-row probabilities were not recomputed from logits")

    derived = _derived_statistics(
        logits=logits,
        probabilities=probabilities,
        targets=targets,
        valid=valid,
    )
    for name in ("target_rows", "known_negative_rows"):
        if value[name] != derived[name]:
            raise ValueError(f"task-row {name} were not recomputed")
    target_rows = derived["target_rows"]
    if not isinstance(target_rows, list):
        raise RuntimeError("validated target rows are malformed")
    expected_target_identities = [
        identity_keys[index]
        for index, (target, is_valid) in enumerate(zip(track_targets, track_valid, strict=True))
        if is_valid and target > 0
    ]
    expected_materialized_target_identities = []
    for row in target_rows:
        track = row_to_track[row]
        if track < 0:
            raise ValueError("a positive task row is not assigned to a physical track")
        expected_materialized_target_identities.append(identity_keys[track])
    if value["target_identity_keys"] != expected_target_identities:
        raise ValueError("task-row target identities were not recomputed")
    if value["materialized_target_identity_keys"] != expected_materialized_target_identities:
        raise ValueError("task-row materialized target identities were not recomputed")
    materialized_target_identity_set = set(expected_materialized_target_identities)
    expected_unmaterialized_target_identities = [
        key for key in expected_target_identities if key not in materialized_target_identity_set
    ]
    if value["unmaterialized_target_identity_keys"] != expected_unmaterialized_target_identities:
        raise ValueError("task-row unmaterialized target identities were not recomputed")
    if value["exact_task"] and not expected_target_identities:
        raise ValueError("an exact task diagnostic has no positive physical target")

    optional_float_fields = (
        "minimum_target_logit",
        "maximum_known_negative_logit",
        "target_vs_hardest_negative_logit_margin",
        "minimum_target_probability",
        "maximum_known_negative_probability",
        "target_vs_hardest_negative_probability_margin",
    )
    for name in optional_float_fields:
        measured = _optional_finite_float(value[name], name=f"task-row {name}")
        expected = _optional_finite_float(
            derived[name],
            name=f"derived task-row {name}",
        )
        if (measured is None) is not (expected is None) or (
            measured is not None
            and expected is not None
            and not math.isclose(measured, expected, rel_tol=1e-7, abs_tol=1e-7)
        ):
            raise ValueError(f"task-row {name} was not recomputed")
    expected_rank = derived["worst_target_rank"]
    if value["worst_target_rank"] != expected_rank or (
        value["worst_target_rank"] is not None
        and (
            isinstance(value["worst_target_rank"], bool)
            or not isinstance(value["worst_target_rank"], int)
            or value["worst_target_rank"] <= 0
        )
    ):
        raise ValueError("task-row rank was not recomputed")
    expected_winner = derived["all_targets_beat_known_negatives"]
    if value["all_targets_beat_known_negatives"] != expected_winner or (
        value["all_targets_beat_known_negatives"] is not None
        and not isinstance(value["all_targets_beat_known_negatives"], bool)
    ):
        raise ValueError("task-row winner decision was not recomputed")
    return value


def validate_task_row_diagnostics(
    value: object,
    *,
    expected_batch_size: int,
) -> list[dict[str, Any]]:
    """Validate one ordered batch of task-row diagnostics."""

    if (
        isinstance(expected_batch_size, bool)
        or not isinstance(expected_batch_size, int)
        or expected_batch_size <= 0
    ):
        raise ValueError("task-row diagnostics require a positive expected batch size")
    if not isinstance(value, list) or len(value) != expected_batch_size:
        raise ValueError("task-row diagnostic batch size differs from the model batch")
    return [validate_task_row_diagnostic(item) for item in value]
