#!/usr/bin/env python3
"""Compare immutable representation snapshots by target-area stratum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError

SCHEMA = "picf-next.lingbot-representation-area-strata-audit.v1"
AREA_STRATA = (
    ("lt_2_percent", 0.0, 0.02),
    ("2_to_5_percent", 0.02, 0.05),
    ("ge_5_percent", 0.05, None),
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ContractError(f"representation sample lacks {'.'.join(keys)}")
        value = value[key]
    return value


def _load_snapshot(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load representation snapshot {path}") from error
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise ContractError("representation snapshot must be a passing object")
    if not isinstance(value.get("samples"), list) or not value["samples"]:
        raise ContractError("representation snapshot must contain samples")
    return value


def _sample_rows(
    snapshot: Mapping[str, Any],
    *,
    partition: str,
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for sample in snapshot["samples"]:
        if not isinstance(sample, Mapping):
            raise ContractError("representation samples must be objects")
        if sample.get("partition") != partition:
            continue
        key = sample.get("sample_key")
        if not isinstance(key, str) or not key or key in rows:
            raise ContractError("representation sample keys must be unique nonempty strings")
        eligible = _nested(sample, "factual_token_evidence", "metrics", "eligible")
        if not isinstance(eligible, bool):
            raise ContractError("target-area eligibility must be boolean")
        task_rank_one = _nested(
            sample,
            "factual_task_row_diagnostic",
            "all_targets_beat_known_negatives",
        )
        margin_logit = _nested(
            sample,
            "factual_task_row_diagnostic",
            "target_vs_hardest_negative_logit_margin",
        )
        margin_probability = _nested(
            sample,
            "factual_task_row_diagnostic",
            "target_vs_hardest_negative_probability_margin",
        )
        ownership = _nested(sample, "factual_ownership_summary", "target_soft_iou")
        if eligible and not isinstance(task_rank_one, bool):
            raise ContractError("eligible task-row rank-one diagnostic must be boolean")
        rows[key] = {
            "all_targets_beat_known_negatives": task_rank_one,
            "eligible": eligible,
            "margin_logit": (
                _finite_float(margin_logit, name="hardest-negative logit margin")
                if eligible
                else None
            ),
            "margin_probability": (
                _finite_float(
                    margin_probability,
                    name="hardest-negative probability margin",
                )
                if eligible
                else None
            ),
            "ownership": (
                _finite_float(ownership, name="target ownership soft-IoU") if eligible else None
            ),
            "target_area_fraction": _finite_float(
                _nested(
                    sample,
                    "factual_token_evidence",
                    "metrics",
                    "target_area_fraction",
                ),
                name="target area fraction",
            ),
            "target_identity_keys": sample.get("factual_target_identity_keys"),
            "task_instruction_sha256": sample.get("factual_task_instruction_sha256"),
            "task_key": sample.get("task_key"),
        }
    if not rows:
        raise ContractError(f"representation snapshot has no {partition!r} samples")
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        raise ContractError("cannot summarize an empty area stratum")
    return math.fsum(values) / len(values)


def _stratum_name(area: float) -> str:
    for name, lower, upper in AREA_STRATA:
        if area >= lower and (upper is None or area < upper):
            return name
    raise ContractError("target area fraction lies outside [0,1]")


def build_area_strata_audit(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    partition: str,
) -> dict[str, object]:
    """Return paired scale diagnostics without defining an acceptance gate."""

    baseline_rows = _sample_rows(baseline, partition=partition)
    candidate_rows = _sample_rows(candidate, partition=partition)
    if baseline_rows.keys() != candidate_rows.keys():
        raise ContractError("baseline and candidate sample-key sets differ")

    paired: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {
        name: [] for name, _, _ in AREA_STRATA
    }
    for key in sorted(baseline_rows):
        before = baseline_rows[key]
        after = candidate_rows[key]
        for field in (
            "eligible",
            "target_identity_keys",
            "task_instruction_sha256",
            "task_key",
        ):
            if before[field] != after[field]:
                raise ContractError(f"paired representation field {field} differs for {key}")
        before_area = float(before["target_area_fraction"])
        after_area = float(after["target_area_fraction"])
        if not 0 <= before_area <= 1 or not math.isclose(
            before_area,
            after_area,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ContractError(f"paired target area differs for {key}")
        if bool(before["eligible"]):
            paired[_stratum_name(before_area)].append((before, after))

    summaries: dict[str, object] = {}
    for name, _, _ in AREA_STRATA:
        pairs = paired[name]
        if not pairs:
            raise ContractError(f"area stratum {name} has no eligible paired samples")
        metrics: dict[str, object] = {}
        for field in ("ownership", "margin_logit", "margin_probability"):
            before_values = [float(before[field]) for before, _ in pairs]
            after_values = [float(after[field]) for _, after in pairs]
            deltas = [
                after - before for before, after in zip(before_values, after_values, strict=True)
            ]
            metrics[field] = {
                "baseline_mean": _mean(before_values),
                "candidate_mean": _mean(after_values),
                "mean_delta": _mean(deltas),
                "positive_delta_count": sum(delta > 0 for delta in deltas),
            }
        before_rank = [
            float(bool(before["all_targets_beat_known_negatives"])) for before, _ in pairs
        ]
        after_rank = [float(bool(after["all_targets_beat_known_negatives"])) for _, after in pairs]
        metrics["rank_one_rate"] = {
            "baseline_mean": _mean(before_rank),
            "candidate_mean": _mean(after_rank),
            "mean_delta": _mean(
                [after - before for before, after in zip(before_rank, after_rank, strict=True)]
            ),
        }
        summaries[name] = {"count": len(pairs), "metrics": metrics}

    return {
        "candidate_artifact_sha256": candidate.get("artifact_sha256"),
        "baseline_artifact_sha256": baseline.get("artifact_sha256"),
        "diagnostic_only": True,
        "eligible_pair_count": sum(len(pairs) for pairs in paired.values()),
        "partition": partition,
        "schema": SCHEMA,
        "status": "COMPLETE",
        "strata": summaries,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--partition", default="heldout")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    baseline = _load_snapshot(args.baseline)
    candidate = _load_snapshot(args.candidate)
    content = {
        **build_area_strata_audit(
            baseline,
            candidate,
            partition=args.partition,
        ),
        "baseline_file_sha256": _sha256(args.baseline),
        "candidate_file_sha256": _sha256(args.candidate),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    artifact_sha256 = hashlib.sha256(_canonical_bytes(content)).hexdigest()
    payload = _canonical_bytes({**content, "artifact_sha256": artifact_sha256}) + b"\n"
    write_bytes_durable_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "eligible_pair_count": content["eligible_pair_count"],
                "output": str(args.output.expanduser().absolute()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
