#!/usr/bin/env python3
"""Compare two durable training windows on their exact paired stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

PAIR_FIELDS = (
    "sample_keys",
    "frame_indices",
    "lane_ids",
    "local_bptt_steps",
    "overshoot_horizon",
    "source_masked_branch",
    "omitted_static_branch",
    "temporal_plan_sha256",
)

LOSS_FIELDS = {
    "action": ("official_action_loss",),
    "entity_total": ("normalized_terms", "set/frame_000/entities"),
    "existence_focal": ("normalized_terms", "set/frame_000/existence_focal"),
    "mask_dice": ("normalized_terms", "set/frame_000/mask_dice"),
    "mask_focal": ("normalized_terms", "set/frame_000/mask_focal"),
    "ownership_nll": ("normalized_terms", "set/frame_000/ownership_nll"),
    "predictive": ("family_terms", "predictive"),
}

GRADIENT_FIELDS = (
    "action_output_norm",
    "native_graph_norm",
    "predictive_readout_norm",
    "relation_projection_norm",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = json.loads(path.read_text())
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for rank_report in payload["rank_reports"]:
        rank = int(rank_report["rank"])
        for step in rank_report["steps"]:
            key = (rank, int(step["global_step"]))
            if key in records:
                raise ValueError(f"duplicate record {key} in {path}")
            records[key] = step
    return records


def _nested(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def _percent(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return 100.0 * (candidate - baseline) / abs(baseline)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _loss_comparison(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, path in LOSS_FIELDS.items():
        base_values = [_nested(record, path) for record in baseline]
        cand_values = [_nested(record, path) for record in candidate]
        base_mean = statistics.fmean(base_values)
        cand_mean = statistics.fmean(cand_values)
        differences = [cand - base for base, cand in zip(base_values, cand_values, strict=True)]
        output[name] = {
            "baseline_mean": base_mean,
            "baseline_median": statistics.median(base_values),
            "candidate_mean": cand_mean,
            "candidate_median": statistics.median(cand_values),
            "candidate_minus_baseline": statistics.fmean(differences),
            "relative_change_percent": _percent(cand_mean, base_mean),
            "candidate_lower_fraction": sum(diff < 0 for diff in differences) / len(differences),
        }
    return output


def _gradient_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in GRADIENT_FIELDS:
        result[name] = _summary([float(record["gradient_metrics"][name]) for record in records])
    preclip_by_step: dict[int, list[float]] = {}
    for record in records:
        preclip_by_step.setdefault(int(record["global_step"]), []).append(
            float(record["gradient_metrics"]["preclip_global_norm"])
        )
    preclip = [statistics.fmean(values) for _, values in sorted(preclip_by_step.items())]
    result["preclip_global_norm_per_optimizer_step"] = {
        **_summary(preclip),
        "steps_over_10": sum(value > 10 for value in preclip),
        "step_count": len(preclip),
    }
    action_mean = result["action_output_norm"]["mean"]
    result["mean_group_to_action_ratios"] = {
        name.removesuffix("_norm"): result[name]["mean"] / action_mean
        for name in GRADIENT_FIELDS
        if name != "action_output_norm"
    }
    result["all_finite"] = all(bool(record["gradient_metrics"]["all_finite"]) for record in records)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate rank/step keys differ")
    keys = sorted(baseline)
    mismatches = [
        {"rank": rank, "global_step": step, "field": field}
        for rank, step in keys
        for field in PAIR_FIELDS
        if baseline[(rank, step)][field] != candidate[(rank, step)][field]
    ]
    if mismatches:
        raise ValueError(f"paired stream mismatch: {mismatches[:5]}")

    base_records = [baseline[key] for key in keys]
    cand_records = [candidate[key] for key in keys]
    steps = sorted({step for _, step in keys})
    state_age_keys: dict[str, list[tuple[int, int]]] = {
        "reset": [],
        "continuation": [],
        "mixed": [],
    }
    for key in keys:
        ages = candidate[key]["state_ages"]
        if all(int(age) == 0 for age in ages):
            state_age_keys["reset"].append(key)
        elif all(int(age) > 0 for age in ages):
            state_age_keys["continuation"].append(key)
        else:
            state_age_keys["mixed"].append(key)
    windows = []
    for start in range(steps[0], steps[-1] + 1, args.window_size):
        stop = min(start + args.window_size - 1, steps[-1])
        window_keys = [key for key in keys if start <= key[1] <= stop]
        windows.append(
            {
                "start_global_step": start,
                "end_global_step": stop,
                "record_count": len(window_keys),
                "losses": _loss_comparison(
                    [baseline[key] for key in window_keys],
                    [candidate[key] for key in window_keys],
                ),
            }
        )

    report = {
        "schema": "picf-next.paired-training-window-comparison/v1",
        "status": "PASS",
        "baseline": {
            "label": args.baseline_label,
            "path": str(args.baseline),
            "sha256": _sha256(args.baseline),
        },
        "candidate": {
            "label": args.candidate_label,
            "path": str(args.candidate),
            "sha256": _sha256(args.candidate),
        },
        "pairing": {
            "record_count": len(keys),
            "optimizer_step_count": len(steps),
            "start_global_step": steps[0],
            "end_global_step": steps[-1],
            "exact_fields": list(PAIR_FIELDS),
            "mismatch_count": 0,
        },
        "overall_losses": _loss_comparison(base_records, cand_records),
        "state_age_strata": {
            name: {
                "record_count": len(stratum_keys),
                "losses": _loss_comparison(
                    [baseline[key] for key in stratum_keys],
                    [candidate[key] for key in stratum_keys],
                ),
            }
            for name, stratum_keys in state_age_keys.items()
            if stratum_keys
        },
        "windows": windows,
        "gradients": {
            args.baseline_label: _gradient_summary(base_records),
            args.candidate_label: _gradient_summary(cand_records),
        },
        "scientific_scope": (
            "Exact-stream short-window engineering comparison only; it does not establish "
            "held-out spatial grounding, rollout success, or long-horizon convergence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"status": "PASS", "output": str(args.output), "sha256": _sha256(args.output)})
    )


if __name__ == "__main__":
    main()
