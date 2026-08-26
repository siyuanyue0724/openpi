#!/usr/bin/env python3
"""Compare two exact-input ADR-149 cold action evaluations sample by sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ACTION_SCHEMAS = (
    "picf-next.adr149-cold-action-snapshot/v1",
    "picf-next.adr149-cold-action-snapshot/v2",
)
REPORT_SCHEMA = "picf-next.adr152-paired-fixed-action-comparison/v1"
PAIR_FIELDS = (
    "ordinal",
    "partition",
    "sample_key",
    "task_key",
    "segment_index",
    "source_episode_index",
    "source_global_index",
    "transition_index",
    "source_digest",
    "model_inputs_sha256",
    "prior_control_chunk_count",
)
CONTRACT_FIELDS = (
    "evaluation_input_sha256",
    "evaluation_plan_sha256",
    "representation_split_sha256",
    "stream_plan_sha256",
    "state_mode",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") not in ACTION_SCHEMAS or payload.get("status") != "PASS":
        raise ValueError(f"not an accepted ADR-149 cold action evaluation: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"action evaluation has no samples: {path}")
    return payload


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _paired_summary(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired action samples must be non-empty and equally sized")
    differences = [right - left for left, right in zip(reference, candidate, strict=True)]
    mean_difference = statistics.fmean(differences)
    if len(differences) == 1:
        standard_error = 0.0
    else:
        standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    reference_mean = statistics.fmean(reference)
    return {
        "sample_count": len(differences),
        "reference": _summary(reference),
        "candidate": _summary(candidate),
        "candidate_minus_reference_mean": mean_difference,
        "candidate_minus_reference_median": statistics.median(differences),
        "normal_approximation_95_percent_interval": [
            mean_difference - 1.96 * standard_error,
            mean_difference + 1.96 * standard_error,
        ],
        "candidate_lower_fraction": sum(value < 0 for value in differences)
        / len(differences),
        "relative_change_percent": (
            None
            if reference_mean == 0
            else 100.0 * mean_difference / abs(reference_mean)
        ),
    }


def compare(
    *,
    reference_path: Path,
    candidate_path: Path,
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    reference = _load(reference_path)
    candidate = _load(candidate_path)
    contract_mismatches = [
        field for field in CONTRACT_FIELDS if reference.get(field) != candidate.get(field)
    ]
    if contract_mismatches:
        raise ValueError(f"action evaluation contract mismatch: {contract_mismatches}")

    reference_samples = reference["samples"]
    candidate_samples = candidate["samples"]
    if len(reference_samples) != len(candidate_samples):
        raise ValueError("action evaluation sample counts differ")
    pair_mismatches = [
        {"sample_index": index, "field": field}
        for index, (left, right) in enumerate(
            zip(reference_samples, candidate_samples, strict=True)
        )
        for field in PAIR_FIELDS
        if left.get(field) != right.get(field)
    ]
    if pair_mismatches:
        raise ValueError(f"action evaluation sample pairing changed: {pair_mismatches[:5]}")

    partitions = sorted({str(sample["partition"]) for sample in reference_samples})
    comparisons: dict[str, Any] = {}
    for partition in [*partitions, "all"]:
        paired = [
            (left, right)
            for left, right in zip(reference_samples, candidate_samples, strict=True)
            if partition == "all" or left["partition"] == partition
        ]
        comparisons[partition] = _paired_summary(
            [
                _finite(left["action_loss"], name=f"reference action loss {index}")
                for index, (left, _right) in enumerate(paired)
            ],
            [
                _finite(right["action_loss"], name=f"candidate action loss {index}")
                for index, (_left, right) in enumerate(paired)
            ],
        )

    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "reference": {
            "label": reference_label,
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
            "checkpoint_global_step": int(reference["checkpoint_global_step"]),
        },
        "candidate": {
            "label": candidate_label,
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
            "checkpoint_global_step": int(candidate["checkpoint_global_step"]),
        },
        "pairing": {
            "sample_count": len(reference_samples),
            "exact_fields": list(PAIR_FIELDS),
            "contract_fields": list(CONTRACT_FIELDS),
            "mismatch_count": 0,
        },
        "comparisons": comparisons,
        "scientific_scope": (
            "Exact-input paired cold action comparison. It tests factual action "
            "non-inferiority or improvement at one checkpoint; it does not by itself "
            "establish object-row mediation, rollout success, or long-horizon convergence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = compare(
        reference_path=args.reference,
        candidate_path=args.candidate,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
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
