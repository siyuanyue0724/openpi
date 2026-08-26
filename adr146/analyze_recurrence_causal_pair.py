#!/usr/bin/env python3
"""Audit the exact ADR-146 zero-state/recurrent-state paired experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

BRANCH_STEP = 200
ARM_STOP_STEP = 300
DIAGNOSTIC_STEPS = (250, 300)
RANKS = (0, 1)
PAIR_FIELDS = (
    "sample_keys",
    "lane_ids",
    "frame_indices",
    "temporal_plan_sha256",
    "local_bptt_steps",
    "overshoot_horizon",
    "source_masked_branch",
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rank_records(root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for rank in RANKS:
        path = root / "metrics" / "rank_journal" / f"rank_{rank}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = (rank, int(record["global_step"]))
            if key in records:
                raise ValueError(f"duplicate rank-step record {key} in {path}")
            records[key] = record
    return records


def _require_steps(
    records: dict[tuple[int, int], dict[str, Any]],
    *,
    start: int,
    stop: int,
    label: str,
) -> None:
    expected = {(rank, step) for rank in RANKS for step in range(start, stop + 1)}
    actual = {key for key in records if start <= key[1] <= stop}
    if actual != expected:
        missing = sorted(expected - actual)[:8]
        extra = sorted(actual - expected)[:8]
        raise ValueError(f"{label} rank-step coverage differs: missing={missing}, extra={extra}")


def _latest_summary(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("run_summary_step_*.json"))
    if not paths:
        raise FileNotFoundError(f"no run summary under {root}")
    return _json(paths[-1])


def _mean_by_step(
    records: dict[tuple[int, int], dict[str, Any]],
    *,
    start: int,
    stop: int,
    value,
) -> list[float]:
    return [
        statistics.fmean(value(records[(rank, step)]) for rank in RANKS)
        for step in range(start, stop + 1)
    ]


def _moving_block_interval(
    values: Sequence[float],
    *,
    seed: int = 146,
    block_size: int = 10,
    draws: int = 10_000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty paired series")
    rng = random.Random(seed)
    n = len(values)
    estimates: list[float] = []
    for _ in range(draws):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(values[(start + offset) % n] for offset in range(block_size))
        estimates.append(statistics.fmean(sample[:n]))
    estimates.sort()
    return estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]


def _paired_summary(zero: Sequence[float], recurrent: Sequence[float]) -> dict[str, Any]:
    if len(zero) != len(recurrent) or not zero:
        raise ValueError("paired summaries require equal non-empty series")
    delta = [right - left for left, right in zip(zero, recurrent, strict=True)]
    low, high = _moving_block_interval(delta)
    return {
        "zero_mean": statistics.fmean(zero),
        "recurrent_mean": statistics.fmean(recurrent),
        "recurrent_minus_zero_mean": statistics.fmean(delta),
        "recurrent_minus_zero_median": statistics.median(delta),
        "recurrent_better_fraction": statistics.fmean(value < 0 for value in delta),
        "moving_block_95pct_interval": [low, high],
        "paired_global_steps": len(delta),
    }


def _diagnostics(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for step in DIAGNOSTIC_STEPS:
        report = _json(
            root / "causal_diagnostics" / f"step_{step:08d}" / "distributed.json"
        )
        rank_reports = report["rank_reports"]
        margins: dict[str, list[float]] = {
            "zero": [],
            "wrong_time": [],
            "cross_batch": [],
        }
        eligible = True
        for rank_report in rank_reports:
            eligible = eligible and bool(rank_report["eligible"])
            variants = rank_report["variants"]
            if not variants:
                continue
            factual = float(variants["factual"]["entity_loss"])
            for control in margins:
                margins[control].append(float(variants[control]["entity_loss"]) - factual)
        output[str(step)] = {
            "eligible": eligible,
            "control_minus_factual_entity_loss": {
                control: {
                    "rank_values": values,
                    "mean": statistics.fmean(values) if values else None,
                    "all_positive": bool(values) and all(value > 0 for value in values),
                }
                for control, values in margins.items()
            },
            "report_sha256": _canonical_sha256(report),
        }
    return output


def _all(records: Iterable[dict[str, Any]], predicate) -> bool:
    return all(predicate(record) for record in records)


def analyze(branch: Path, zero: Path, recurrent: Path) -> dict[str, Any]:
    branch_records = _rank_records(branch)
    zero_records = _rank_records(zero)
    recurrent_records = _rank_records(recurrent)
    _require_steps(branch_records, start=1, stop=BRANCH_STEP, label="branch")
    _require_steps(zero_records, start=1, stop=ARM_STOP_STEP, label="zero")
    _require_steps(recurrent_records, start=1, stop=ARM_STOP_STEP, label="recurrent")

    prefix = {(rank, step) for rank in RANKS for step in range(1, BRANCH_STEP + 1)}
    arm_keys = {
        (rank, step) for rank in RANKS for step in range(BRANCH_STEP + 1, ARM_STOP_STEP + 1)
    }
    branch_prefix_sha = _canonical_sha256(
        [branch_records[key] for key in sorted(prefix)]
    )
    zero_prefix_sha = _canonical_sha256([zero_records[key] for key in sorted(prefix)])
    recurrent_prefix_sha = _canonical_sha256(
        [recurrent_records[key] for key in sorted(prefix)]
    )
    routing_mismatches = [
        [rank, step, field]
        for rank, step in sorted(arm_keys)
        for field in PAIR_FIELDS
        if zero_records[(rank, step)][field] != recurrent_records[(rank, step)][field]
    ]

    zero_arm = [zero_records[key] for key in sorted(arm_keys)]
    recurrent_arm = [recurrent_records[key] for key in sorted(arm_keys)]
    zero_summary = _latest_summary(zero)
    recurrent_summary = _latest_summary(recurrent)
    zero_manifest = _json(zero / "causal_arm_manifest.json")
    recurrent_manifest = _json(recurrent / "causal_arm_manifest.json")

    integrity = {
        "shared_branch_prefix": branch_prefix_sha == zero_prefix_sha == recurrent_prefix_sha,
        "paired_routing": not routing_mismatches,
        "same_resume_boundary": (
            zero_summary.get("loaded_boundary_sha256")
            == recurrent_summary.get("loaded_boundary_sha256")
            and zero_summary.get("loaded_boundary_sha256") is not None
        ),
        "same_execution_contract": (
            zero_manifest.get("execution_contract_sha256")
            == recurrent_manifest.get("execution_contract_sha256")
        ),
        "zero_withholds_all_state": _all(
            zero_arm,
            lambda record: record["causal_ablation_mode"] == "zero_state"
            and record["posterior_input_mode"] == "withheld"
            and record["consumed_previous_state_count"] == 0
            and record["staged_row_bindings"] == [[] for _ in record["staged_row_bindings"]],
        ),
        "recurrent_consumes_exact_available_state": _all(
            recurrent_arm,
            lambda record: record["causal_ablation_mode"] == "recurrent_state"
            and record["posterior_input_mode"] == "causal_lane"
            and record["consumed_previous_state_count"]
            == record["available_previous_state_count"],
        ),
        "all_gradients_finite": _all(
            (*zero_arm, *recurrent_arm),
            lambda record: bool(record["gradient_metrics"]["all_finite"]),
        ),
        "all_auxiliary_branches_absent": _all(
            (*zero_arm, *recurrent_arm),
            lambda record: record["correction_branch_count"] == 0
            and not record["current_grid_branch"]
            and not record["omitted_static_branch"]
            and record["gradient_metrics"]["predictive_readout_elements"] == 0,
        ),
    }

    zero_action = _mean_by_step(
        zero_records,
        start=BRANCH_STEP + 1,
        stop=ARM_STOP_STEP,
        value=lambda record: float(record["official_action_loss"]),
    )
    recurrent_action = _mean_by_step(
        recurrent_records,
        start=BRANCH_STEP + 1,
        stop=ARM_STOP_STEP,
        value=lambda record: float(record["official_action_loss"]),
    )
    zero_entity = _mean_by_step(
        zero_records,
        start=BRANCH_STEP + 1,
        stop=ARM_STOP_STEP,
        value=lambda record: float(record["frame_losses"][0]["total"]),
    )
    recurrent_entity = _mean_by_step(
        recurrent_records,
        start=BRANCH_STEP + 1,
        stop=ARM_STOP_STEP,
        value=lambda record: float(record["frame_losses"][0]["total"]),
    )
    diagnostics = {
        "zero_state": _diagnostics(zero),
        "recurrent_state": _diagnostics(recurrent),
    }
    finite_diagnostics = all(
        math.isfinite(value)
        for arm in diagnostics.values()
        for step in arm.values()
        for control in step["control_minus_factual_entity_loss"].values()
        for value in control["rank_values"]
    )
    integrity["diagnostics_complete_and_finite"] = finite_diagnostics and all(
        step["eligible"] for arm in diagnostics.values() for step in arm.values()
    )

    return {
        "schema": "picf-next.adr146-recurrence-causal-analysis/v1",
        "status": "PASS_INTEGRITY" if all(integrity.values()) else "FAIL_INTEGRITY",
        "runs": {
            "branch": str(branch),
            "zero_state": str(zero),
            "recurrent_state": str(recurrent),
        },
        "integrity": integrity,
        "routing_mismatches": routing_mismatches[:32],
        "branch_prefix_sha256": branch_prefix_sha,
        "curves": {
            "action": _paired_summary(zero_action, recurrent_action),
            "entity": _paired_summary(zero_entity, recurrent_entity),
        },
        "diagnostics": diagnostics,
        "scientific_decision": "MANUAL_PENDING_CAUSAL_AND_VISUAL_REVIEW",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.branch, args.zero, args.recurrent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
