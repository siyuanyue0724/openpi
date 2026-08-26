#!/usr/bin/env python3
"""Summarize identity-selective mediation in one ADR-149 two-pass diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

DIAGNOSTIC_SCHEMA = "picf-next.adr149-two-pass-filter-diagnostic/v1"
REPORT_SCHEMA = "picf-next.adr152-two-pass-binding-summary/v1"
STATE_ARMS = ("zero", "wrong_time", "cross_batch", "wrong_row")
DIRECT_PRIOR_ARMS = ("zero", "cross_batch", "wrong_row")
VARIANT_METRICS = (
    "official_action_loss",
    "factual_official_action_loss",
    "entity_loss",
    "predictive_family_loss",
)


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
        raise ValueError("cannot summarize an empty diagnostic sample")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "nonzero_fraction": sum(value != 0 for value in values) / len(values),
    }


def _paired_delta(
    reports: list[dict[str, Any]], *, arm: str, metric: str
) -> dict[str, Any]:
    factual = [
        _finite(report["variants"]["factual"][metric], name=f"factual {metric}")
        for report in reports
    ]
    candidate = [
        _finite(report["variants"][arm][metric], name=f"{arm} {metric}")
        for report in reports
    ]
    differences = [right - left for left, right in zip(factual, candidate, strict=True)]
    return {
        "factual": _summary(factual),
        "candidate": _summary(candidate),
        "candidate_minus_factual": _summary(differences),
        "absolute_candidate_minus_factual": _summary([abs(value) for value in differences]),
        "per_rank_candidate_minus_factual": differences,
    }


def analyze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != DIAGNOSTIC_SCHEMA:
        raise ValueError(f"not an ADR-149 two-pass diagnostic: {path}")
    rank_reports = payload.get("rank_reports")
    if not isinstance(rank_reports, list) or not rank_reports:
        raise ValueError("two-pass diagnostic contains no rank reports")
    eligible = [report for report in rank_reports if report.get("eligible") is True]
    if not eligible:
        raise ValueError("two-pass diagnostic has no eligible rank")
    for report in eligible:
        variants = report.get("variants")
        if not isinstance(variants, dict) or not {"factual", *STATE_ARMS} <= variants.keys():
            raise ValueError("two-pass diagnostic omits a registered state arm")
        intervention = report.get("control_intervention", {}).get(
            "direct_prior_intervention", {}
        )
        direct_arms = intervention.get("arms")
        if not isinstance(direct_arms, dict) or not {
            "factual",
            *DIRECT_PRIOR_ARMS,
        } <= direct_arms.keys():
            raise ValueError("two-pass diagnostic omits a direct-prior arm")

    state_interventions: dict[str, Any] = {}
    for arm in STATE_ARMS:
        state_interventions[arm] = {
            "relative_state_manipulation": _summary(
                [
                    _finite(
                        report["variants"][arm]["relative_state_manipulation"],
                        name=f"{arm} state manipulation",
                    )
                    for report in eligible
                ]
            ),
            "metrics": {
                metric: _paired_delta(eligible, arm=arm, metric=metric)
                for metric in VARIANT_METRICS
            },
        }

    direct_prior_interventions: dict[str, Any] = {}
    for arm in DIRECT_PRIOR_ARMS:
        arm_reports = [
            report["control_intervention"]["direct_prior_intervention"]["arms"][arm]
            for report in eligible
        ]
        direct_prior_interventions[arm] = {
            "prior_relative_l2": _summary(
                [
                    _finite(item["prior_relative_l2"], name=f"{arm} prior relative L2")
                    for item in arm_reports
                ]
            ),
            "official_action_loss_delta": _summary(
                [
                    _finite(
                        item["official_action_loss_delta"],
                        name=f"{arm} direct-prior action delta",
                    )
                    for item in arm_reports
                ]
            ),
            "absolute_official_action_loss_delta": _summary(
                [abs(float(item["official_action_loss_delta"])) for item in arm_reports]
            ),
        }

    wrong_row_state = state_interventions["wrong_row"]["relative_state_manipulation"]
    wrong_row_route_action = state_interventions["wrong_row"]["metrics"][
        "official_action_loss"
    ]["candidate_minus_factual"]
    wrong_row_direct_action = direct_prior_interventions["wrong_row"][
        "official_action_loss_delta"
    ]
    zero_direct_action = direct_prior_interventions["zero"]["official_action_loss_delta"]
    exact_flags = {
        "all_ranks_eligible": len(eligible) == len(rank_reports),
        "wrong_row_changes_state_on_every_rank": wrong_row_state["minimum"] > 0,
        "routed_action_exactly_invariant_to_wrong_row_on_every_rank": (
            wrong_row_route_action["nonzero_fraction"] == 0
        ),
        "direct_action_exactly_invariant_to_wrong_row_on_every_rank": (
            wrong_row_direct_action["nonzero_fraction"] == 0
        ),
        "direct_action_changes_under_zero_prior_on_any_rank": (
            zero_direct_action["nonzero_fraction"] > 0
        ),
    }

    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "source": {"path": str(path), "sha256": _sha256(path)},
        "global_step": int(payload["global_step"]),
        "rank_count": len(rank_reports),
        "eligible_rank_count": len(eligible),
        "state_interventions": state_interventions,
        "direct_prior_interventions": direct_prior_interventions,
        "exact_flags": exact_flags,
        "interpretation_contract": (
            "A nonzero state perturbation with exact wrong-row action invariance, while a "
            "zero-prior intervention changes action, demonstrates scene-level prior use "
            "without established row-identity-selective mediation. Nonzero wrong-row deltas "
            "must still be interpreted with their magnitude, consistency, factual action, "
            "and visual/entity evidence; this tool does not auto-authorize training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = analyze(args.diagnostic)
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
