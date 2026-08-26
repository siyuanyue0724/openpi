#!/usr/bin/env python3
"""Compare the exact-stream action prefix of ADR-147 against matched official LBOT."""

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
    "reset",
    "source_digest",
    "augmentation_seeds",
    "flow_noise_seeds",
    "flow_timestep_seeds",
)
LBOT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-lbot.v1"
LEGACY_LBOT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-p0.v1"
ACCEPTED_LBOT_SCHEMAS = frozenset({LBOT_SCHEMA, LEGACY_LBOT_SCHEMA})
PICF_METRICS_SCHEMA = "picf-next.task-independent-full-metrics/v1"
REPORT_SCHEMA = "picf-next.adr147-matched-lbot-action-prefix/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _rank_steps(
    rank_reports: Any,
    *,
    loss_field: str,
    limit: int,
    source: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    if not isinstance(rank_reports, list):
        raise ValueError(f"{source} rank reports must be a list")
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for rank_report in rank_reports:
        rank = int(rank_report["rank"])
        for step in rank_report["steps"]:
            global_step = int(step["global_step"])
            if global_step > limit:
                continue
            key = (rank, global_step)
            if key in records:
                raise ValueError(f"duplicate {source} record {key}")
            for field in PAIR_FIELDS:
                if field not in step:
                    raise ValueError(f"{source} record {key} omits {field}")
            _finite(step[loss_field], name=f"{source} action loss {key}")
            records[key] = step
    return records


def _load_candidate(
    run_dir: Path, *, steps: int
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"candidate run manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted((run_dir / "metrics").glob("steps_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != PICF_METRICS_SCHEMA:
            raise ValueError(f"candidate metric schema changed: {path}")
        window = _rank_steps(
            payload.get("rank_reports"),
            loss_field="official_action_loss",
            limit=steps,
            source=str(path),
        )
        duplicate = records.keys() & window.keys()
        if duplicate:
            raise ValueError(f"candidate metric windows overlap: {sorted(duplicate)[:3]}")
        records.update(window)
    return manifest, records


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty action window")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _comparison(
    keys: list[tuple[int, int]],
    baseline: dict[tuple[int, int], dict[str, Any]],
    candidate: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    baseline_values = [float(baseline[key]["action_loss"]) for key in keys]
    candidate_values = [float(candidate[key]["official_action_loss"]) for key in keys]
    differences = [
        right - left for left, right in zip(baseline_values, candidate_values, strict=True)
    ]
    baseline_mean = statistics.fmean(baseline_values)
    candidate_mean = statistics.fmean(candidate_values)
    return {
        "baseline": _summary(baseline_values),
        "candidate": _summary(candidate_values),
        "candidate_minus_baseline_mean": statistics.fmean(differences),
        "candidate_lower_fraction": sum(value < 0 for value in differences) / len(differences),
        "relative_change_percent": (
            None
            if baseline_mean == 0
            else 100.0 * (candidate_mean - baseline_mean) / abs(baseline_mean)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.window_size <= 0 or args.steps % args.window_size:
        raise ValueError("steps must be positive and divisible by window size")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    baseline_payload = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if (
        baseline_payload.get("schema") not in ACCEPTED_LBOT_SCHEMAS
        or baseline_payload.get("status") != "PASS"
    ):
        raise ValueError("baseline is not one accepted official LBOT report")
    if baseline_payload.get("picf_graph_installed") is not False:
        raise ValueError("baseline unexpectedly contains PICF")
    candidate_manifest, candidate = _load_candidate(args.candidate_run_dir, steps=args.steps)
    baseline = _rank_steps(
        baseline_payload.get("rank_reports"),
        loss_field="action_loss",
        limit=args.steps,
        source=str(args.baseline_report),
    )

    world_size = int(baseline_payload.get("world_size", -1))
    if world_size != 4 or int(candidate_manifest.get("world_size", -1)) != world_size:
        raise ValueError("matched ADR-147 comparison requires two four-rank runs")
    if baseline_payload.get("plan_sha256") != candidate_manifest.get("stream_plan_sha256"):
        raise ValueError("baseline and candidate frozen stream plans differ")
    if baseline_payload.get("seed") != candidate_manifest.get("execution_contract", {}).get("seed"):
        raise ValueError("baseline and candidate seeds differ")
    optimizer_contract = baseline_payload.get("optimizer_contract")
    execution_contract = candidate_manifest.get("execution_contract")
    if not isinstance(optimizer_contract, dict) or not isinstance(execution_contract, dict):
        raise ValueError("baseline or candidate optimizer contract is absent")
    if float(optimizer_contract.get("learning_rate", math.nan)).hex() != execution_contract.get(
        "learning_rate"
    ):
        raise ValueError("baseline and candidate learning rates differ")
    if float(baseline_payload.get("max_grad_norm", math.nan)).hex() != execution_contract.get(
        "max_grad_norm"
    ):
        raise ValueError("baseline and candidate gradient clipping differs")
    expected_keys = {
        (rank, step) for rank in range(world_size) for step in range(1, args.steps + 1)
    }
    if baseline.keys() != expected_keys or candidate.keys() != expected_keys:
        raise ValueError("baseline or candidate does not cover the complete matched prefix")

    mismatches = [
        {"rank": rank, "global_step": step, "field": field}
        for rank, step in sorted(expected_keys)
        for field in PAIR_FIELDS
        if baseline[(rank, step)][field] != candidate[(rank, step)][field]
    ]
    if mismatches:
        raise ValueError(f"paired action stream mismatch: {mismatches[:5]}")

    ordered = sorted(expected_keys)
    windows = []
    for start in range(1, args.steps + 1, args.window_size):
        stop = start + args.window_size - 1
        keys = [key for key in ordered if start <= key[1] <= stop]
        windows.append(
            {
                "start_global_step": start,
                "end_global_step": stop,
                "record_count": len(keys),
                "action": _comparison(keys, baseline, candidate),
            }
        )

    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "baseline": {
            "path": str(args.baseline_report),
            "sha256": _sha256(args.baseline_report),
        },
        "candidate": {
            "run_dir": str(args.candidate_run_dir),
            "run_manifest_sha256": _sha256(args.candidate_run_dir / "run_manifest.json"),
        },
        "contract": {
            "world_size": world_size,
            "optimizer_step_count": args.steps,
            "record_count": len(ordered),
            "frozen_stream_plan_sha256": baseline_payload["plan_sha256"],
            "seed": baseline_payload["seed"],
            "learning_rate": optimizer_contract["learning_rate"],
            "max_grad_norm": baseline_payload["max_grad_norm"],
            "exact_pair_fields": list(PAIR_FIELDS),
            "pair_mismatch_count": 0,
        },
        "overall_action": _comparison(ordered, baseline, candidate),
        "windows": windows,
        "scientific_scope": (
            "Exact-stream early action-loss comparison only. It does not establish held-out "
            "rollout success, long-horizon convergence, or causal benefit from the posterior."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": "PASS", "output": str(args.output), "sha256": _sha256(args.output)})
    )


if __name__ == "__main__":
    main()
