#!/usr/bin/env python3
"""Validate G3 mediator-trial arms from immutable rank journals.

The validator is deliberately independent from the training runner.  It reads
the append-only rank journals, recomputes every arm statistic, and optionally
cross-checks the final runner report.  It never imports or mutates the runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

JOURNAL_SCHEMA = "picf-next.ltop-g3-action-information-set-step.v1"
FINAL_REPORT_SCHEMA = "picf-next.ltop-g3-training-phase.v1"
OUTPUT_SCHEMA = "picf-next.ltop-g3-mediator-trial-arm-validation.v1"
ARM_VALUES = ("factual", "mediator-required")
ARM_LABELS = ("FACTUAL", "MEDIATOR_REQUIRED")
ARM_LABEL_BY_VALUE = dict(zip(ARM_VALUES, ARM_LABELS, strict=True))
LOSS_FIELDS = ("action_loss", "total_loss", "physical_set_loss", "task_address_loss")
DEFAULT_WINDOW_SIZE = 16
DEFAULT_EXPECTED_COUNT_PER_ARM_PER_RANK = 128
DEFAULT_EXPECTED_WORLD_SIZE = 2
DEFAULT_MAXIMUM_LAST_TO_FIRST_RATIO = 0.95
_RANK_FILE_PATTERN = re.compile(r"rank_(0|[1-9][0-9]*)\.jsonl\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class ValidationInputError(ValueError):
    """Raised when an input artifact cannot be interpreted safely."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--expected-count-per-arm-per-rank",
        type=int,
        default=DEFAULT_EXPECTED_COUNT_PER_ARM_PER_RANK,
    )
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=DEFAULT_EXPECTED_WORLD_SIZE,
    )
    parser.add_argument(
        "--maximum-last-to-first-ratio",
        type=float,
        default=DEFAULT_MAXIMUM_LAST_TO_FIRST_RATIO,
    )
    return parser.parse_args(argv)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationInputError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationInputError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationInputError(f"{name} must be at least {minimum}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationInputError(f"{name} must be numeric")
    return float(value)


def _regular_file_bytes(path: Path, *, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValidationInputError(f"{name} must be a regular non-symlink file: {path}")
    return path.read_bytes()


def _load_json(payload: bytes, *, name: str) -> object:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationInputError(f"{name} is not valid UTF-8 JSON") from error


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values or any(not math.isfinite(value) for value in values):
        return None
    return math.fsum(values) / len(values)


def _window_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "finite": all(math.isfinite(value) for value in values),
        "mean_action_loss": _mean_or_none(values),
    }


def _arm_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    first_values: Sequence[float],
    last_values: Sequence[float],
    expected_count: int,
    maximum_ratio: float,
) -> dict[str, Any]:
    action_values = [float(record["action_loss"]) for record in records]
    all_losses_finite = all(
        math.isfinite(float(record[field])) for record in records for field in LOSS_FIELDS
    )
    first_mean = _mean_or_none(first_values)
    last_mean = _mean_or_none(last_values)
    ratio = None
    relative_improvement = None
    if first_mean is not None and last_mean is not None and first_mean > 0.0:
        ratio = last_mean / first_mean
        relative_improvement = (first_mean - last_mean) / first_mean
    return {
        "action_loss_finite": all(math.isfinite(value) for value in action_values),
        "all_reported_losses_finite": all_losses_finite,
        "balanced_count_pass": len(records) == expected_count,
        "count": len(records),
        "expected_count": expected_count,
        "first_window": _window_summary(first_values),
        "last_to_first_ratio": ratio,
        "last_window": _window_summary(last_values),
        "maximum_last_to_first_ratio": maximum_ratio,
        "relative_improvement": relative_improvement,
        "window_gate_pass": ratio is not None and ratio <= maximum_ratio,
    }


def _rank_journal_paths(journal_dir: Path) -> list[tuple[int, Path]]:
    if journal_dir.is_symlink() or not journal_dir.is_dir():
        raise ValidationInputError(
            f"journal_dir must be a regular non-symlink directory: {journal_dir}"
        )
    matches: list[tuple[int, Path]] = []
    for path in journal_dir.iterdir():
        match = _RANK_FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            matches.append((int(match.group(1)), path))
    if not matches:
        raise ValidationInputError(f"journal_dir contains no rank_<n>.jsonl files: {journal_dir}")
    matches.sort(key=lambda value: value[0])
    ranks = [rank for rank, _ in matches]
    if ranks != list(range(len(ranks))):
        raise ValidationInputError(f"rank journal files are not contiguous from rank 0: {ranks}")
    return matches


def _load_rank_journal(rank: int, path: Path) -> tuple[list[Mapping[str, Any]], str]:
    payload = _regular_file_bytes(path, name=f"rank {rank} journal")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValidationInputError(f"rank {rank} journal is not ASCII") from error
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValidationInputError(f"rank {rank} journal line {line_number} is blank")
        record = _mapping(
            _load_json(line.encode("ascii"), name=f"rank {rank} journal line {line_number}"),
            name=f"rank {rank} journal line {line_number}",
        )
        if record.get("schema") != JOURNAL_SCHEMA:
            raise ValidationInputError(
                f"rank {rank} journal line {line_number} has the wrong schema"
            )
        if _integer(record.get("rank"), name=f"rank {rank} record rank", minimum=0) != rank:
            raise ValidationInputError(f"rank {rank} journal line {line_number} rank differs")
        _integer(record.get("global_step"), name=f"rank {rank} global_step", minimum=1)
        arm = record.get("arm")
        if arm not in ARM_VALUES:
            raise ValidationInputError(
                f"rank {rank} journal line {line_number} has unknown arm {arm!r}"
            )
        for field in LOSS_FIELDS:
            _number(record.get(field), name=f"rank {rank} line {line_number} {field}")
        for field in ("cycle_index", "scene_index", "prompt_index"):
            _integer(
                record.get(field),
                name=f"rank {rank} line {line_number} {field}",
                minimum=0,
            )
        for field in ("scene_key", "prompt_key"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValidationInputError(
                    f"rank {rank} journal line {line_number} {field} must be non-empty"
                )
        sample_keys = record.get("sample_keys")
        if not isinstance(sample_keys, list) or any(
            not isinstance(value, str) or not value for value in sample_keys
        ):
            raise ValidationInputError(
                f"rank {rank} journal line {line_number} sample_keys must be strings"
            )
        digest = record.get("schedule_sha256")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValidationInputError(
                f"rank {rank} journal line {line_number} has invalid schedule_sha256"
            )
        records.append(record)
    steps = [_integer(record["global_step"], name="global_step", minimum=1) for record in records]
    if steps != list(range(1, len(records) + 1)):
        raise ValidationInputError(
            f"rank {rank} global_step sequence is not contiguous from 1: {steps[:4]}..."
        )
    return records, _sha256_bytes(payload)


def _summarize_rank(
    rank: int,
    path: Path,
    records: Sequence[Mapping[str, Any]],
    journal_sha256: str,
    *,
    window_size: int,
    expected_count_per_arm: int,
    maximum_ratio: float,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    digests = sorted({str(record["schedule_sha256"]) for record in records})
    schedule_consistent = len(digests) == 1
    if not schedule_consistent:
        failures.append(f"rank {rank}: schedule digest is not constant across its journal")
    arms: dict[str, Any] = {}
    for arm_value in ARM_VALUES:
        arm_label = ARM_LABEL_BY_VALUE[arm_value]
        arm_records = [record for record in records if record["arm"] == arm_value]
        action_values = [float(record["action_loss"]) for record in arm_records]
        first_values = action_values[:window_size]
        last_values = action_values[-window_size:] if len(action_values) >= window_size else []
        summary = _arm_summary(
            arm_records,
            first_values=first_values,
            last_values=last_values,
            expected_count=expected_count_per_arm,
            maximum_ratio=maximum_ratio,
        )
        arms[arm_label] = summary
        if not summary["balanced_count_pass"]:
            failures.append(
                f"rank {rank} {arm_label}: count {len(arm_records)} != {expected_count_per_arm}"
            )
        if not summary["all_reported_losses_finite"]:
            failures.append(f"rank {rank} {arm_label}: one or more reported losses are non-finite")
        if len(first_values) != window_size or len(last_values) != window_size:
            failures.append(
                f"rank {rank} {arm_label}: cannot form complete {window_size}-step windows"
            )
        elif not summary["window_gate_pass"]:
            failures.append(
                f"rank {rank} {arm_label}: last-{window_size} action loss is not at most "
                f"{maximum_ratio} times first-{window_size}"
            )
    return (
        {
            "arms": arms,
            "balanced_arms_pass": all(
                arms[arm_label]["balanced_count_pass"] for arm_label in ARM_LABELS
            ),
            "finite_pass": all(
                arms[arm_label]["all_reported_losses_finite"] for arm_label in ARM_LABELS
            ),
            "journal": {
                "file_sha256": journal_sha256,
                "path": str(path.resolve()),
                "record_count": len(records),
            },
            "rank": rank,
            "schedule": {
                "consistent": schedule_consistent,
                "digests": digests,
                "sha256": digests[0] if schedule_consistent else None,
            },
            "window_gates_pass": all(
                arms[arm_label]["window_gate_pass"] for arm_label in ARM_LABELS
            ),
        },
        failures,
    )


def _validate_final_report(
    path: Path,
    *,
    records_by_rank: Mapping[int, Sequence[Mapping[str, Any]]],
    journal_sha256_by_rank: Mapping[int, str],
    schedule_sha256: str | None,
    expected_count_per_arm: int,
) -> tuple[dict[str, Any], list[str]]:
    payload = _regular_file_bytes(path, name="final report")
    report = _mapping(_load_json(payload, name="final report"), name="final report")
    failures: list[str] = []
    if report.get("schema") != FINAL_REPORT_SCHEMA:
        failures.append("final report schema is not the G3 training-phase schema")
    if report.get("mode") != "mediator-trial" or report.get("phase") != "training":
        failures.append("final report is not a mediator-trial training report")
    if report.get("status") != "PASS":
        failures.append("final report status is not PASS")
    if report.get("world_size") != len(records_by_rank):
        failures.append("final report world_size differs from rank journals")
    expected_steps = 2 * expected_count_per_arm
    if report.get("steps") != expected_steps:
        failures.append(f"final report steps differs from expected {expected_steps}")

    rank_reports_value = report.get("rank_reports")
    if not isinstance(rank_reports_value, list) or any(
        not isinstance(value, Mapping) for value in rank_reports_value
    ):
        failures.append("final report rank_reports is not a list of objects")
        rank_reports: list[Mapping[str, Any]] = []
    else:
        rank_reports = cast(list[Mapping[str, Any]], rank_reports_value)
    report_ranks = [value.get("rank") for value in rank_reports]
    if report_ranks != list(records_by_rank):
        failures.append("final report rank order differs from rank journals")

    for rank_report in rank_reports:
        rank_value = rank_report.get("rank")
        if isinstance(rank_value, bool) or not isinstance(rank_value, int):
            continue
        if rank_value not in records_by_rank:
            continue
        records = records_by_rank[rank_value]
        counts = {
            arm_value: sum(record["arm"] == arm_value for record in records)
            for arm_value in ARM_VALUES
        }
        if rank_report.get("action_information_set_counts") != counts:
            failures.append(f"final report rank {rank_value} arm counts differ from journal")
        if rank_report.get("action_information_set_schedule_sha256") != schedule_sha256:
            failures.append(f"final report rank {rank_value} schedule digest differs")
        if rank_report.get("all_gradients_finite") is not True:
            failures.append(f"final report rank {rank_value} gradients are not all finite")
        action_losses = rank_report.get("action_losses")
        journal_action_losses = [float(record["action_loss"]) for record in records]
        if action_losses != journal_action_losses:
            failures.append(f"final report rank {rank_value} action losses differ from journal")
        history = rank_report.get("action_information_set_history")
        expected_history = [
            {
                key: record[key]
                for key in (
                    "global_step",
                    "cycle_index",
                    "scene_index",
                    "scene_key",
                    "prompt_index",
                    "prompt_key",
                    "arm",
                )
            }
            for record in records
        ]
        if history != expected_history:
            failures.append(f"final report rank {rank_value} arm history differs from journal")
        receipt = rank_report.get("arm_journal")
        if not isinstance(receipt, Mapping):
            failures.append(f"final report rank {rank_value} journal receipt is absent")
        elif (
            receipt.get("rank") != rank_value
            or receipt.get("record_count") != len(records)
            or receipt.get("file_sha256") != journal_sha256_by_rank[rank_value]
        ):
            failures.append(f"final report rank {rank_value} journal receipt differs")

    schedule = report.get("training_contract")
    if isinstance(schedule, Mapping):
        schedule = schedule.get("action_information_set_trial")
    if isinstance(schedule, Mapping):
        schedule = schedule.get("schedule")
    report_schedule_sha256 = schedule.get("sha256") if isinstance(schedule, Mapping) else None
    if report_schedule_sha256 != schedule_sha256:
        failures.append("final report sealed schedule digest differs from journals")

    return (
        {
            "consistent": not failures,
            "file_sha256": _sha256_bytes(payload),
            "path": str(path.resolve()),
            "runner_failures": report.get("failures"),
            "runner_status": report.get("status"),
            "schema": report.get("schema"),
        },
        failures,
    )


def validate_ltop_g3_mediator_trial(
    *,
    journal_dir: Path,
    report_path: Path | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    expected_count_per_arm_per_rank: int = DEFAULT_EXPECTED_COUNT_PER_ARM_PER_RANK,
    expected_world_size: int = DEFAULT_EXPECTED_WORLD_SIZE,
    maximum_last_to_first_ratio: float = DEFAULT_MAXIMUM_LAST_TO_FIRST_RATIO,
) -> dict[str, Any]:
    """Return a deterministic validation report without mutating trial artifacts."""

    if window_size <= 0:
        raise ValidationInputError("window_size must be positive")
    if expected_count_per_arm_per_rank < window_size:
        raise ValidationInputError("expected_count_per_arm_per_rank must cover one window")
    if expected_world_size <= 0:
        raise ValidationInputError("expected_world_size must be positive")
    if not math.isfinite(maximum_last_to_first_ratio) or not (
        0.0 < maximum_last_to_first_ratio <= 1.0
    ):
        raise ValidationInputError("maximum_last_to_first_ratio must lie in (0, 1]")

    journal_paths = _rank_journal_paths(journal_dir)
    failures: list[str] = []
    if len(journal_paths) != expected_world_size:
        failures.append(
            f"world size {len(journal_paths)} differs from expected {expected_world_size}"
        )

    records_by_rank: dict[int, list[Mapping[str, Any]]] = {}
    journal_sha256_by_rank: dict[int, str] = {}
    rank_summaries = []
    for rank, path in journal_paths:
        records, journal_sha256 = _load_rank_journal(rank, path)
        records_by_rank[rank] = records
        journal_sha256_by_rank[rank] = journal_sha256
        rank_summary, rank_failures = _summarize_rank(
            rank,
            path,
            records,
            journal_sha256,
            window_size=window_size,
            expected_count_per_arm=expected_count_per_arm_per_rank,
            maximum_ratio=maximum_last_to_first_ratio,
        )
        rank_summaries.append(rank_summary)
        failures.extend(rank_failures)

    rank_digests = [summary["schedule"]["sha256"] for summary in rank_summaries]
    global_schedule_consistent = (
        bool(rank_digests)
        and all(digest is not None for digest in rank_digests)
        and len(set(rank_digests)) == 1
    )
    schedule_sha256 = rank_digests[0] if global_schedule_consistent else None
    if not global_schedule_consistent:
        failures.append("rank journals do not share one schedule digest")
    reference_schedule = [
        (
            record["global_step"],
            record["cycle_index"],
            record["scene_index"],
            record["scene_key"],
            record["prompt_index"],
            record["prompt_key"],
            record["arm"],
        )
        for record in records_by_rank[min(records_by_rank)]
    ]
    global_schedule_entries_consistent = all(
        [
            (
                record["global_step"],
                record["cycle_index"],
                record["scene_index"],
                record["scene_key"],
                record["prompt_index"],
                record["prompt_key"],
                record["arm"],
            )
            for record in records_by_rank[rank]
        ]
        == reference_schedule
        for rank in records_by_rank
    )
    if not global_schedule_entries_consistent:
        failures.append("rank journals do not contain the same counterbalanced schedule entries")

    global_arms: dict[str, Any] = {}
    for arm_value in ARM_VALUES:
        arm_label = ARM_LABEL_BY_VALUE[arm_value]
        rank_action_values = {
            rank: [
                float(record["action_loss"])
                for record in records_by_rank[rank]
                if record["arm"] == arm_value
            ]
            for rank in records_by_rank
        }
        records = [
            record
            for rank in records_by_rank
            for record in records_by_rank[rank]
            if record["arm"] == arm_value
        ]
        first_values = [
            value for rank in rank_action_values for value in rank_action_values[rank][:window_size]
        ]
        last_values = [
            value
            for rank in rank_action_values
            for value in (
                rank_action_values[rank][-window_size:]
                if len(rank_action_values[rank]) >= window_size
                else []
            )
        ]
        expected_global_count = expected_count_per_arm_per_rank * expected_world_size
        global_arms[arm_label] = _arm_summary(
            records,
            first_values=first_values,
            last_values=last_values,
            expected_count=expected_global_count,
            maximum_ratio=maximum_last_to_first_ratio,
        )
        if not global_arms[arm_label]["balanced_count_pass"]:
            failures.append(f"global {arm_label}: count {len(records)} != {expected_global_count}")
        if not global_arms[arm_label]["all_reported_losses_finite"]:
            failures.append(f"global {arm_label}: one or more reported losses are non-finite")
        optimizer_steps = sorted({int(record["global_step"]) for record in records})
        global_arms[arm_label]["balanced_optimizer_steps_pass"] = (
            len(optimizer_steps) == expected_count_per_arm_per_rank
        )
        global_arms[arm_label]["expected_optimizer_step_count"] = expected_count_per_arm_per_rank
        global_arms[arm_label]["optimizer_step_count"] = len(optimizer_steps)
        if not global_arms[arm_label]["balanced_optimizer_steps_pass"]:
            failures.append(
                f"global {arm_label}: optimizer-step count {len(optimizer_steps)} != "
                f"{expected_count_per_arm_per_rank}"
            )
        expected_global_window = window_size * expected_world_size
        if (
            len(first_values) != expected_global_window
            or len(last_values) != expected_global_window
        ):
            failures.append(f"global {arm_label}: pooled rank-local windows are incomplete")
        elif not global_arms[arm_label]["window_gate_pass"]:
            failures.append(
                f"global {arm_label}: pooled last-{window_size} action loss is not at most "
                f"{maximum_last_to_first_ratio} times pooled first-{window_size}"
            )

    final_report_summary = None
    if report_path is not None:
        final_report_summary, report_failures = _validate_final_report(
            report_path,
            records_by_rank=records_by_rank,
            journal_sha256_by_rank=journal_sha256_by_rank,
            schedule_sha256=schedule_sha256,
            expected_count_per_arm=expected_count_per_arm_per_rank,
        )
        failures.extend(report_failures)

    result = {
        "failures": failures,
        "final_report": final_report_summary,
        "global": {
            "aggregation": "pooled-rank-local-arm-windows",
            "arms": global_arms,
            "balanced_arms_pass": all(
                global_arms[arm_label]["balanced_count_pass"] for arm_label in ARM_LABELS
            ),
            "finite_pass": all(
                global_arms[arm_label]["all_reported_losses_finite"] for arm_label in ARM_LABELS
            ),
            "record_count": sum(len(records) for records in records_by_rank.values()),
            "schedule": {
                "consistent": global_schedule_consistent,
                "entries_consistent_across_ranks": global_schedule_entries_consistent,
                "rank_digests": rank_digests,
                "sha256": schedule_sha256,
            },
            "window_gates_pass": all(
                global_arms[arm_label]["window_gate_pass"] for arm_label in ARM_LABELS
            ),
        },
        "inputs": {
            "journal_dir": str(journal_dir.resolve()),
            "report": None if report_path is None else str(report_path.resolve()),
        },
        "ranks": rank_summaries,
        "schema": OUTPUT_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "thresholds": {
            "expected_count_per_arm_per_rank": expected_count_per_arm_per_rank,
            "expected_world_size": expected_world_size,
            "maximum_last_to_first_ratio": maximum_last_to_first_ratio,
            "window_size": window_size,
        },
    }
    return result


def _failure_report(message: str) -> dict[str, Any]:
    return {
        "failures": [message],
        "final_report": None,
        "global": None,
        "inputs": None,
        "ranks": [],
        "schema": OUTPUT_SCHEMA,
        "status": "FAIL",
        "thresholds": None,
    }


def _write_atomic(path: Path, payload: str) -> None:
    if path.is_symlink():
        raise ValidationInputError(f"output cannot be a symbolic link: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise ValidationInputError(f"temporary output path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="ascii") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = validate_ltop_g3_mediator_trial(
            journal_dir=args.journal_dir,
            report_path=args.report,
            window_size=args.window_size,
            expected_count_per_arm_per_rank=args.expected_count_per_arm_per_rank,
            expected_world_size=args.expected_world_size,
            maximum_last_to_first_ratio=args.maximum_last_to_first_ratio,
        )
    except (OSError, ValidationInputError) as error:
        result = _failure_report(f"{type(error).__name__}: {error}")
    payload = _canonical_json(result) + "\n"
    if args.output is not None:
        try:
            _write_atomic(args.output, payload)
        except (OSError, ValidationInputError) as error:
            result = _failure_report(f"{type(error).__name__}: {error}")
            payload = _canonical_json(result) + "\n"
    sys.stdout.write(payload)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
