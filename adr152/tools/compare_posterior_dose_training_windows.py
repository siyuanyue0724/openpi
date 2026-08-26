#!/usr/bin/env python3
"""Compare regular and high-dose posterior training on the same stochastic stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "picf-next.adr152-posterior-dose-paired-training/v1"

PAIR_FIELDS = (
    "sample_keys",
    "frame_indices",
    "lane_ids",
    "augmentation_seeds",
    "flow_noise_seeds",
    "flow_timestep_seeds",
    "optimizer_lags",
    "local_bptt_steps",
    "overshoot_horizon",
    "reset",
    "source_digest",
    "causal_ablation_mode",
    "posterior_input_mode",
)

LOSS_FIELDS = {
    "factual_action": ("official_action_loss",),
    "entity_total": ("normalized_terms", "set/frame_000/entities"),
    "existence_focal": ("normalized_terms", "set/frame_000/existence_focal"),
    "mask_dice": ("normalized_terms", "set/frame_000/mask_dice"),
    "mask_focal": ("normalized_terms", "set/frame_000/mask_focal"),
    "ownership_nll": ("normalized_terms", "set/frame_000/ownership_nll"),
    "predictive": ("family_terms", "predictive"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = payload.get("rank_reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError(f"rank_reports are absent: {path}")
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for rank_report in reports:
        rank = int(rank_report["rank"])
        for step in rank_report["steps"]:
            key = (rank, int(step["global_step"]))
            if key in records:
                raise ValueError(f"duplicate rank/step record {key}: {path}")
            records[key] = step
    return records


def _nested(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value at {'/'.join(path)}")
    return number


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty window")
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _paired_interval(differences: list[float]) -> list[float]:
    mean = statistics.fmean(differences)
    if len(differences) == 1:
        return [mean, mean]
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    return [mean - 1.96 * standard_error, mean + 1.96 * standard_error]


def _losses(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, path in LOSS_FIELDS.items():
        base_values = [_nested(record, path) for record in baseline]
        cand_values = [_nested(record, path) for record in candidate]
        differences = [
            candidate_value - baseline_value
            for baseline_value, candidate_value in zip(
                base_values, cand_values, strict=True
            )
        ]
        base_mean = statistics.fmean(base_values)
        candidate_mean = statistics.fmean(cand_values)
        output[name] = {
            "baseline": _summary(base_values),
            "candidate": _summary(cand_values),
            "candidate_minus_baseline_mean": statistics.fmean(differences),
            "normal_approximation_95_percent_interval": _paired_interval(differences),
            "relative_change_percent": (
                None
                if base_mean == 0
                else 100.0 * (candidate_mean - base_mean) / abs(base_mean)
            ),
            "candidate_lower_fraction": sum(value < 0 for value in differences)
            / len(differences),
        }
    return output


def _route_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    routed = [record.get("source_masked_branch") is True for record in records]
    omitted = [record.get("omitted_static_branch") is True for record in records]
    action_present = [record.get("omitted_static_action_loss") is not None for record in records]
    return {
        "record_count": len(records),
        "source_masked_fraction": sum(routed) / len(routed),
        "omitted_static_fraction": sum(omitted) / len(omitted),
        "routed_action_present_fraction": sum(action_present) / len(action_present),
    }


def _candidate_route_gap(records: list[dict[str, Any]]) -> dict[str, Any]:
    differences = []
    for record in records:
        if record.get("source_masked_branch") is not True:
            raise ValueError("candidate is not routed on every rank-step record")
        routed = record.get("omitted_static_action_loss")
        if routed is None:
            raise ValueError("candidate routed action is absent")
        differences.append(float(routed) - float(record["official_action_loss"]))
    return {
        **_summary(differences),
        "routed_lower_fraction": sum(value < 0 for value in differences)
        / len(differences),
    }


def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    boundaries: tuple[int, ...],
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate rank/step keys differ")
    keys = sorted(baseline)
    mismatches = [
        {"rank": rank, "global_step": step, "field": field}
        for rank, step in keys
        for field in PAIR_FIELDS
        if baseline[(rank, step)].get(field) != candidate[(rank, step)].get(field)
    ]
    if mismatches:
        raise ValueError(f"paired stochastic stream mismatch: {mismatches[:5]}")
    steps = sorted({step for _, step in keys})
    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("optimizer steps are not contiguous")
    if not boundaries:
        boundaries = (steps[-1],)
    if tuple(sorted(set(boundaries))) != boundaries or boundaries[-1] != steps[-1]:
        raise ValueError("window boundaries must increase and end at the final step")

    baseline_records = [baseline[key] for key in keys]
    candidate_records = [candidate[key] for key in keys]
    windows = []
    start = steps[0]
    for stop in boundaries:
        window_keys = [key for key in keys if start <= key[1] <= stop]
        if not window_keys:
            raise ValueError(f"empty window {start}--{stop}")
        windows.append(
            {
                "start_global_step": start,
                "end_global_step": stop,
                "rank_step_record_count": len(window_keys),
                "losses": _losses(
                    [baseline[key] for key in window_keys],
                    [candidate[key] for key in window_keys],
                ),
            }
        )
        start = stop + 1

    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
        "candidate": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
        "pairing": {
            "rank_step_record_count": len(keys),
            "optimizer_step_count": len(steps),
            "start_global_step": steps[0],
            "end_global_step": steps[-1],
            "exact_fields": list(PAIR_FIELDS),
            "intentional_treatment_fields": [
                "source_masked_branch",
                "omitted_static_branch",
                "omitted_static_action_branch",
                "temporal_plan_sha256",
            ],
            "mismatch_count": 0,
        },
        "route_dose": {
            "baseline": _route_summary(baseline_records),
            "candidate": _route_summary(candidate_records),
            "candidate_routed_minus_factual_action": _candidate_route_gap(
                candidate_records
            ),
        },
        "overall_losses": _losses(baseline_records, candidate_records),
        "windows": windows,
        "scientific_scope": (
            "Exact stochastic-stream comparison of a route-dose treatment. It tests "
            "optimization and factual-action trade-offs; it does not establish "
            "object-row mediation, rollout success, or long-horizon convergence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--window-boundary", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = compare(
        args.baseline,
        args.candidate,
        boundaries=tuple(args.window_boundary),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "PASS", "output": str(args.output), "sha256": _sha256(args.output)}
        )
    )


if __name__ == "__main__":
    main()
