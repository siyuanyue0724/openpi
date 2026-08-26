"""Audit durable task-independent training windows without touching the trainer.

The report is an engineering integrity and stratification artifact. It deliberately does
not turn short, task-confounded loss windows into a scientific convergence verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

METRIC_SCHEMA = "picf-next.task-independent-full-metrics/v1"
REPORT_SCHEMA = "picf-next.task-independent-full-window-audit/v1"
LANGUAGE_SEGMENT_SAMPLE_KEY = re.compile(
    r"^calvin-language-segment-(?P<segment>\d{8})/"
    r"transition-(?P<transition>\d{8})-frame-(?P<frame>\d{8})$"
)
SOURCE_EPISODE_SAMPLE_KEY = re.compile(
    r"^calvin-source-episode-(?P<segment>\d{8})/frame-(?P<frame>\d{8})$"
)
SAMPLE_KEYS = (LANGUAGE_SEGMENT_SAMPLE_KEY, SOURCE_EPISODE_SAMPLE_KEY)
REQUIRED_GRADIENTS = (
    "action_output_norm",
    "native_graph_norm",
    "relation_projection_norm",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return math.fsum(values) / len(values)


def _linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != METRIC_SCHEMA:
            raise ValueError(f"unexpected metric schema in {path}")
        reports = payload.get("rank_reports")
        if not isinstance(reports, list) or not reports:
            raise ValueError(f"metric file has no rank reports: {path}")
        for report in reports:
            rank = report.get("rank")
            steps = report.get("steps")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise ValueError(f"rank must be an integer in {path}")
            if not isinstance(steps, list) or not steps:
                raise ValueError(f"rank report has no steps in {path}")
            for step in steps:
                if not isinstance(step, dict):
                    raise ValueError(f"step record must be an object in {path}")
                records.append({"rank": rank, **step})
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    return records, sources


def _segment_id(record: dict[str, Any]) -> int:
    keys = record.get("sample_keys")
    if not isinstance(keys, list) or len(keys) != 1 or not isinstance(keys[0], str):
        raise ValueError("the released batch-one run must have one sample key per rank")
    for pattern in SAMPLE_KEYS:
        match = pattern.fullmatch(keys[0])
        if match is not None:
            return int(match.group("segment"))
    raise ValueError(f"malformed CALVIN sample key: {keys[0]}")


def _segments(
    records: list[dict[str, Any]],
    annotations: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for rank in sorted({int(record["rank"]) for record in records}):
        ranked = sorted(
            (record for record in records if int(record["rank"]) == rank),
            key=lambda record: int(record["global_step"]),
        )
        current: list[dict[str, Any]] = []
        for record in ranked:
            if current and _segment_id(record) != _segment_id(current[-1]):
                result.append(_summarize_segment(rank, current, annotations))
                current = []
            current.append(record)
        if current:
            result.append(_summarize_segment(rank, current, annotations))
    return result


def _summarize_segment(
    rank: int,
    records: list[dict[str, Any]],
    annotations: dict[int, dict[str, object]],
) -> dict[str, object]:
    ages = [int(record["state_ages"][0]) for record in records]
    segment = _segment_id(records[0])
    bindings = [_binding_map(record) for record in records]
    binding_comparisons = 0
    binding_switches = 0
    for previous, current in zip(bindings, bindings[1:], strict=False):
        shared = previous.keys() & current.keys()
        binding_comparisons += len(shared)
        binding_switches += sum(previous[key] != current[key] for key in shared)
    result: dict[str, object] = {
        "rank": rank,
        "segment_index": segment,
        "first_global_step": int(records[0]["global_step"]),
        "last_global_step": int(records[-1]["global_step"]),
        "first_state_age": ages[0],
        "last_state_age": ages[-1],
        "maximum_state_age": max(ages),
        "record_count": len(records),
        "row_binding_comparisons": binding_comparisons,
        "row_binding_switches": binding_switches,
        "row_binding_switch_fraction": (
            None if binding_comparisons == 0 else binding_switches / binding_comparisons
        ),
        "state_age_windows": _state_age_windows(records),
        "mean_action_loss": _mean(
            [_finite_number(record["official_action_loss"], label="action") for record in records]
        ),
    }
    if segment in annotations:
        result.update(annotations[segment])
    return result


def _state_age_windows(
    records: list[dict[str, Any]], size: int = 16
) -> list[dict[str, int | float]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        age = int(record["state_ages"][0])
        grouped.setdefault(age // size, []).append(record)
    result: list[dict[str, int | float]] = []
    for bucket in sorted(grouped):
        selected = grouped[bucket]
        losses = [loss for record in selected for loss in record["frame_losses"]]
        ages = [int(record["state_ages"][0]) for record in selected]
        result.append(
            {
                "minimum_state_age": min(ages),
                "maximum_state_age": max(ages),
                "record_count": len(selected),
                "mean_action_loss": _mean(
                    [
                        _finite_number(record["official_action_loss"], label="action")
                        for record in selected
                    ]
                ),
                "mean_entity_total": _mean(
                    [_finite_number(loss["total"], label="entity total") for loss in losses]
                ),
                "mean_mask_dice": _mean(
                    [_finite_number(loss["mask_dice"], label="mask dice") for loss in losses]
                ),
                "mean_ownership_nll": _mean(
                    [
                        _finite_number(loss["ownership_nll"], label="ownership NLL")
                        for loss in losses
                    ]
                ),
            }
        )
    return result


def _binding_map(record: dict[str, Any]) -> dict[str, int]:
    batches = record.get("row_bindings")
    if not isinstance(batches, list) or len(batches) != 1:
        raise ValueError("the released batch-one run must have one row-binding set")
    result: dict[str, int] = {}
    used_rows: set[int] = set()
    for pair in batches[0]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("row binding must be one identity/row pair")
        identity, row = pair
        if not isinstance(identity, str) or not identity:
            raise ValueError("row-binding identity must be a nonempty string")
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            raise ValueError("row-binding row must be a non-negative integer")
        if identity in result or row in used_rows:
            raise ValueError("row bindings must be one-to-one")
        result[identity] = row
        used_rows.add(row)
    return result


def load_calvin_annotations(path: Path) -> dict[int, dict[str, object]]:
    import numpy as np

    payload = np.load(path, allow_pickle=True).item()
    language = payload.get("language")
    info = payload.get("info")
    if not isinstance(language, dict) or not isinstance(info, dict):
        raise ValueError("CALVIN annotation payload is malformed")
    annotations = language.get("ann")
    task_keys = language.get("task")
    intervals = info.get("indx")
    if not (len(annotations) == len(task_keys) == len(intervals)):
        raise ValueError("CALVIN annotation arrays have different lengths")
    return {
        index: {
            "task_annotation": str(annotation),
            "task_key": str(task_key),
            "dataset_first_frame": int(interval[0]),
            "dataset_stop_frame_exclusive": int(interval[1]),
        }
        for index, (annotation, task_key, interval) in enumerate(
            zip(annotations, task_keys, intervals, strict=True)
        )
    }


def _window_summary(records: list[dict[str, Any]], size: int) -> list[dict[str, float | int]]:
    first = min(int(record["global_step"]) for record in records)
    last = max(int(record["global_step"]) for record in records)
    result: list[dict[str, float | int]] = []
    for start in range(first, last + 1, size):
        end = min(start + size - 1, last)
        selected = [record for record in records if start <= int(record["global_step"]) <= end]
        frame_losses = [loss for record in selected for loss in record.get("frame_losses", [])]
        result.append(
            {
                "start_global_step": start,
                "end_global_step": end,
                "record_count": len(selected),
                "mean_action_loss": _mean(
                    [
                        _finite_number(record["official_action_loss"], label="action")
                        for record in selected
                    ]
                ),
                "mean_entity_total": _mean(
                    [_finite_number(loss["total"], label="entity total") for loss in frame_losses]
                ),
                "mean_mask_focal": _mean(
                    [
                        _finite_number(loss["mask_focal"], label="mask focal")
                        for loss in frame_losses
                    ]
                ),
                "mean_mask_dice": _mean(
                    [_finite_number(loss["mask_dice"], label="mask dice") for loss in frame_losses]
                ),
                "mean_ownership_nll": _mean(
                    [
                        _finite_number(loss["ownership_nll"], label="ownership NLL")
                        for loss in frame_losses
                    ]
                ),
                "mean_predictive_family_loss": _mean(
                    [
                        _finite_number(record["family_terms"]["predictive"], label="predictive")
                        for record in selected
                    ]
                ),
            }
        )
    return result


def audit(
    paths: list[Path],
    *,
    window_size: int = 10,
    annotation_path: Path | None = None,
) -> dict[str, object]:
    records, sources = _load(paths)
    annotations = load_calvin_annotations(annotation_path) if annotation_path is not None else {}
    ranks = sorted({int(record["rank"]) for record in records})
    if ranks != list(range(len(ranks))):
        raise ValueError("rank ids must be contiguous from zero")
    start = min(int(record["global_step"]) for record in records)
    end = max(int(record["global_step"]) for record in records)
    expected_steps = list(range(start, end + 1))

    rank_checks = []
    integrity_errors: list[str] = []
    for rank in ranks:
        ranked = sorted(
            (record for record in records if int(record["rank"]) == rank),
            key=lambda record: int(record["global_step"]),
        )
        steps = [int(record["global_step"]) for record in ranked]
        exact = steps == expected_steps
        if not exact:
            integrity_errors.append(f"rank {rank} does not contain the exact step interval")
        rank_checks.append(
            {
                "rank": rank,
                "record_count": len(ranked),
                "exact_steps": exact,
                "unique_posterior_digest_count": len(
                    {str(record["posterior_bank_sha256"]) for record in ranked}
                ),
                "minimum_state_age": min(int(record["state_ages"][0]) for record in ranked),
                "maximum_state_age": max(int(record["state_ages"][0]) for record in ranked),
            }
        )

    by_step: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_step.setdefault(int(record["global_step"]), []).append(record)
    for step, grouped in by_step.items():
        signatures = {
            (
                int(record["local_bptt_steps"]),
                int(record["overshoot_horizon"]),
                bool(record["source_masked_branch"]),
                bool(record["omitted_static_branch"]),
            )
            for record in grouped
        }
        if len(grouped) != len(ranks) or len(signatures) != 1:
            integrity_errors.append(f"optimizer event {step} differs across ranks")

    gradient_values: dict[str, list[float]] = {
        name: [] for name in (*REQUIRED_GRADIENTS, "predictive_readout_norm")
    }
    all_finite = True
    for record in records:
        metrics = record.get("gradient_metrics")
        if not isinstance(metrics, dict) or metrics.get("all_finite") is not True:
            all_finite = False
            continue
        for name in gradient_values:
            gradient_values[name].append(_finite_number(metrics[name], label=name))
    if not all_finite:
        integrity_errors.append("one or more gradient reports are absent or non-finite")
    for name in REQUIRED_GRADIENTS:
        if any(value <= 0 for value in gradient_values[name]):
            integrity_errors.append(f"{name} is not positive on every record")

    unique_events = [by_step[step][0] for step in sorted(by_step)]
    overshoots = Counter(
        int(record["overshoot_horizon"])
        for record in unique_events
        if int(record["overshoot_horizon"]) > 0
    )
    step_times = [_finite_number(record["step_time_s"], label="step time") for record in records]
    gradient_summary = {
        name: {
            "minimum": min(values),
            "maximum": max(values),
            "zero_count": sum(value == 0 for value in values),
        }
        for name, values in gradient_values.items()
    }
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "source_metrics": sources,
        "bounds": {"start_global_step": start, "end_global_step": end},
        "status": "PASS" if not integrity_errors else "FAIL",
        "engineering_integrity_errors": integrity_errors,
        "scientific_acceptance": False,
        "scientific_acceptance_reason": (
            "short task-confounded training windows and no held-out rollout cannot establish "
            "action convergence or anchor quality"
        ),
        "rank_checks": rank_checks,
        "all_gradients_finite": all_finite,
        "gradient_norms": gradient_summary,
        "optimizer_overshoot_horizon_counts": dict(sorted(overshoots.items())),
        "optimizer_source_masked_count": sum(
            bool(record["source_masked_branch"]) for record in unique_events
        ),
        "optimizer_omitted_static_count": sum(
            bool(record["omitted_static_branch"]) for record in unique_events
        ),
        "local_bptt_step_counts": dict(
            sorted(Counter(int(record["local_bptt_steps"]) for record in unique_events).items())
        ),
        "maximum_cuda_allocated_gib": max(
            int(record["peak_cuda_allocated_bytes"]) for record in records
        )
        / 2**30,
        "maximum_cuda_reserved_gib": max(
            int(record["peak_cuda_reserved_bytes"]) for record in records
        )
        / 2**30,
        "step_time_seconds": {
            "mean": _mean(step_times),
            "median": statistics.median(step_times),
            "p95": _linear_quantile(step_times, 0.95),
            "maximum": max(step_times),
        },
        "annotation_source": (
            None
            if annotation_path is None
            else {"path": str(annotation_path), "sha256": sha256_file(annotation_path)}
        ),
        "segments": _segments(records, annotations),
        "loss_windows": _window_summary(records, window_size),
    }
    report["artifact_sha256"] = canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--calvin-annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.window_size <= 0:
        raise ValueError("window size must be positive")
    report = audit(
        arguments.metrics,
        window_size=arguments.window_size,
        annotation_path=arguments.calvin_annotations,
    )
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(arguments.output)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "artifact_sha256": report["artifact_sha256"],
                "bounds": report["bounds"],
                "status": report["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
