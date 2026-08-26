#!/usr/bin/env python3
"""Audit the complete factual/routed ledger of a posterior-adoption dose run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "picf-next.adr152-posterior-dose-ledger/v1"


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


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty ledger")
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _load_records(run_dir: Path, *, stop_step: int) -> tuple[list[dict[str, Any]], list[Path]]:
    journal_root = run_dir / "metrics" / "rank_journal"
    paths = sorted(journal_root.glob("rank_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"rank journals are absent: {journal_root}")
    records: list[dict[str, Any]] = []
    keys: set[tuple[int, int]] = set()
    for path in paths:
        rank = int(path.stem.removeprefix("rank_"))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record["global_step"])
            if step > stop_step:
                continue
            key = (rank, step)
            if key in keys:
                raise ValueError(f"duplicate rank/step record {key}")
            keys.add(key)
            record["_rank"] = rank
            record["_line_number"] = line_number
            records.append(record)
    return records, paths


def _action_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    factual = [
        _finite(record["official_action_loss"], name="factual action") for record in records
    ]
    routed = [
        _finite(record["omitted_static_action_loss"], name="routed action")
        for record in records
    ]
    effective = [
        _finite(record["effective_training_action_loss"], name="effective action")
        for record in records
    ]
    gaps = [right - left for left, right in zip(factual, routed, strict=True)]
    return {
        "factual": _summary(factual),
        "routed_omitted_static": _summary(routed),
        "effective_training": _summary(effective),
        "routed_minus_factual": _summary(gaps),
        "routed_lower_fraction": sum(value < 0 for value in gaps) / len(gaps),
    }


def summarize(run_dir: Path, *, stop_step: int, boundaries: tuple[int, ...]) -> dict[str, Any]:
    if stop_step <= 0:
        raise ValueError("stop step must be positive")
    if not boundaries or tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError("window boundaries must be unique and strictly increasing")
    if boundaries[-1] != stop_step:
        raise ValueError("final window boundary must equal stop step")
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records, journal_paths = _load_records(run_dir, stop_step=stop_step)
    world_size = len(journal_paths)
    expected = {
        (rank, step) for rank in range(world_size) for step in range(1, stop_step + 1)
    }
    measured = {(int(record["_rank"]), int(record["global_step"])) for record in records}
    if measured != expected:
        missing = sorted(expected - measured)[:5]
        unexpected = sorted(measured - expected)[:5]
        raise ValueError(f"dose ledger is incomplete: missing={missing}, unexpected={unexpected}")
    ordered = sorted(records, key=lambda record: (int(record["global_step"]), record["_rank"]))
    missing_branch = [
        (record["_rank"], record["global_step"])
        for record in ordered
        if record.get("source_masked_branch") is not True
        or record.get("omitted_static_branch") is not True
        or record.get("omitted_static_action_loss") is None
    ]
    nonfinite_gradient = [
        (record["_rank"], record["global_step"])
        for record in ordered
        if record.get("gradient_metrics", {}).get("all_finite") is not True
    ]
    if missing_branch or nonfinite_gradient:
        raise ValueError(
            "dose run violates its all-step branch/finite contract: "
            f"branch={missing_branch[:5]}, gradient={nonfinite_gradient[:5]}"
        )

    windows = []
    start = 1
    for stop in boundaries:
        window = [record for record in ordered if start <= int(record["global_step"]) <= stop]
        windows.append(
            {
                "start_global_step": start,
                "end_global_step": stop,
                "rank_step_record_count": len(window),
                "action": _action_summary(window),
            }
        )
        start = stop + 1

    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "run": {
            "path": str(run_dir),
            "manifest_sha256": _sha256(manifest_path),
            "acceptance_mode": manifest.get("execution_contract", {}).get(
                "acceptance_mode"
            ),
        },
        "contract": {
            "world_size": world_size,
            "optimizer_step_count": stop_step,
            "rank_step_record_count": len(ordered),
            "all_records_routed": True,
            "all_gradients_finite": True,
            "routed_full_step_equivalents": stop_step / 2,
            "branch_weight_contract": "0.5 factual + 0.5 routed omitted-static",
        },
        "overall_action": _action_summary(ordered),
        "windows": windows,
        "runtime": {
            "step_time_seconds": _summary(
                [_finite(record["step_time_s"], name="step time") for record in ordered]
            ),
            "maximum_peak_reserved_gib": max(
                _finite(record["peak_cuda_reserved_bytes"], name="peak reserved bytes")
                for record in ordered
            )
            / 2**30,
            "maximum_preclip_global_norm": max(
                _finite(
                    record["gradient_metrics"]["preclip_global_norm"],
                    name="preclip global norm",
                )
                for record in ordered
            ),
        },
        "journal_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in journal_paths
        ],
        "scientific_scope": (
            "Complete execution and optimization ledger for the registered route-dose "
            "treatment. It does not establish object identity, rollout success, or "
            "same-object cross-modal binding."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stop-step", type=int, required=True)
    parser.add_argument("--window-boundary", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = summarize(
        args.run_dir,
        stop_step=args.stop_step,
        boundaries=tuple(args.window_boundary),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "sha256": _sha256(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
