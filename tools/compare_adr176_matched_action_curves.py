#!/usr/bin/env python3
"""Compare strict ADR-176 PICF and released-LingBot fixed action curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

PICF_SCHEMAS = (
    "picf-next.adr149-cold-action-snapshot/v1",
    "picf-next.adr149-cold-action-snapshot/v2",
)
LBOT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-lbot-snapshot.v1"
REPORT_SCHEMA = "picf-next.adr176-matched-action-curve-comparison/v1"
IDENTITY_FIELDS = (
    "ordinal",
    "partition",
    "rank",
    "sample_key",
    "segment_index",
    "source_digest",
    "source_episode_index",
    "source_global_index",
    "task_key",
    "transition_index",
    "model_inputs_sha256",
)
PARTITIONS = ("validation", "heldout")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _parse_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from error
    if not steps or steps[0] != 0 or tuple(sorted(set(steps))) != steps:
        raise argparse.ArgumentTypeError("steps must be sorted, unique, and start at zero")
    return steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picf-run-dir", type=Path, required=True)
    parser.add_argument("--lbot-run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=_parse_steps, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def _load_snapshot(
    path: Path,
    *,
    expected_schema: str | tuple[str, ...],
    expected_step: int,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    accepted = (expected_schema,) if isinstance(expected_schema, str) else expected_schema
    if not isinstance(value, dict) or value.get("schema") not in accepted:
        raise ValueError(f"unexpected snapshot schema: {path}")
    if value.get("status") != "PASS" or value.get("checkpoint_global_step") != expected_step:
        raise ValueError(f"snapshot is not a passing step-{expected_step} result: {path}")
    artifact_sha256 = value.get("artifact_sha256")
    semantic = dict(value)
    semantic.pop("artifact_sha256", None)
    if artifact_sha256 != _canonical_sha256(semantic):
        raise ValueError(f"snapshot semantic SHA-256 differs: {path}")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"snapshot has no samples: {path}")
    return value


def _snapshot_path(run_dir: Path, *, treatment: str, step: int) -> Path:
    if treatment == "picf":
        return run_dir / "action_evaluations" / f"step_{step:08d}" / "distributed.json"
    if treatment == "lbot":
        return run_dir / f"action_evaluation_step_{step:06d}.json"
    raise ValueError(f"unknown treatment: {treatment}")


def _identity(sample: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(sample.get(field) for field in IDENTITY_FIELDS)


def _finite_action(sample: dict[str, Any]) -> float:
    value = float(sample["action_loss"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("action loss must be finite and nonnegative")
    return value


def _matched_samples(
    picf: dict[str, Any],
    lbot: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    picf_samples = sorted(picf["samples"], key=_identity)
    lbot_samples = sorted(lbot["samples"], key=_identity)
    if [_identity(sample) for sample in picf_samples] != [
        _identity(sample) for sample in lbot_samples
    ]:
        raise ValueError("PICF and LingBot fixed sample identities differ")
    return list(zip(picf_samples, lbot_samples, strict=True))


def _trapezoid(values: list[float], steps: tuple[int, ...]) -> float:
    if len(values) != len(steps):
        raise ValueError("curve values and steps differ")
    horizon = steps[-1] - steps[0]
    if horizon <= 0:
        raise ValueError("curve requires a positive horizon")
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for left_step, right_step, left_value, right_value in zip(
            steps[:-1],
            steps[1:],
            values[:-1],
            values[1:],
            strict=True,
        )
    )
    return area / horizon


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_interval(
    values: list[float],
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    generator = random.Random(seed)
    means = [
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    ]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def compare_curves(
    *,
    picf_snapshots: list[dict[str, Any]],
    lbot_snapshots: list[dict[str, Any]],
    steps: tuple[int, ...],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if len(picf_snapshots) != len(steps) or len(lbot_snapshots) != len(steps):
        raise ValueError("snapshot curves and registered steps differ")
    contract_fields = (
        "stream_plan_sha256",
        "representation_split_sha256",
        "evaluation_plan_sha256",
    )
    for field in contract_fields:
        values = {snapshot.get(field) for snapshot in [*picf_snapshots, *lbot_snapshots]}
        if len(values) != 1 or None in values:
            raise ValueError(f"matched action contract differs for {field}")
    if any(snapshot.get("picf_graph_installed") is not True for snapshot in picf_snapshots):
        raise ValueError("PICF curve does not install the PICF graph")
    if any(snapshot.get("picf_graph_installed") is not False for snapshot in lbot_snapshots):
        raise ValueError("LingBot baseline unexpectedly installs the PICF graph")

    pairs_by_step = [
        _matched_samples(picf, lbot)
        for picf, lbot in zip(picf_snapshots, lbot_snapshots, strict=True)
    ]
    identities = [_identity(pair[0]) for pair in pairs_by_step[0]]
    for pairs in pairs_by_step[1:]:
        if [_identity(pair[0]) for pair in pairs] != identities:
            raise ValueError("fixed sample identities changed across the curve")

    partitions: dict[str, Any] = {}
    for partition_index, partition in enumerate(PARTITIONS):
        indices = [
            index
            for index, pair in enumerate(pairs_by_step[0])
            if pair[0]["partition"] == partition
        ]
        if not indices:
            raise ValueError(f"matched curve has no {partition} samples")
        picf_means: list[float] = []
        lbot_means: list[float] = []
        per_sample_picf: dict[int, list[float]] = defaultdict(list)
        per_sample_lbot: dict[int, list[float]] = defaultdict(list)
        for pairs in pairs_by_step:
            picf_values = [_finite_action(pairs[index][0]) for index in indices]
            lbot_values = [_finite_action(pairs[index][1]) for index in indices]
            picf_means.append(sum(picf_values) / len(picf_values))
            lbot_means.append(sum(lbot_values) / len(lbot_values))
            for local_index, (picf_value, lbot_value) in enumerate(
                zip(picf_values, lbot_values, strict=True)
            ):
                per_sample_picf[local_index].append(picf_value)
                per_sample_lbot[local_index].append(lbot_value)
        endpoint_deltas = [
            per_sample_picf[index][-1] - per_sample_lbot[index][-1] for index in per_sample_picf
        ]
        auc_deltas = [
            _trapezoid(per_sample_picf[index], steps) - _trapezoid(per_sample_lbot[index], steps)
            for index in per_sample_picf
        ]
        picf_auc = _trapezoid(picf_means, steps)
        lbot_auc = _trapezoid(lbot_means, steps)
        partitions[partition] = {
            "sample_count": len(indices),
            "picf_curve": picf_means,
            "lbot_curve": lbot_means,
            "endpoint": {
                "picf": picf_means[-1],
                "lbot": lbot_means[-1],
                "picf_over_lbot": picf_means[-1] / lbot_means[-1],
                "paired_delta": sum(endpoint_deltas) / len(endpoint_deltas),
                "paired_delta_bootstrap_95": _bootstrap_interval(
                    endpoint_deltas,
                    replicates=bootstrap_replicates,
                    seed=20260818 + partition_index,
                ),
                "picf_sample_wins": sum(value < 0 for value in endpoint_deltas),
                "ties": sum(value == 0 for value in endpoint_deltas),
                "lbot_sample_wins": sum(value > 0 for value in endpoint_deltas),
            },
            "normalized_auc": {
                "picf": picf_auc,
                "lbot": lbot_auc,
                "picf_over_lbot": picf_auc / lbot_auc,
                "paired_delta": sum(auc_deltas) / len(auc_deltas),
                "paired_delta_bootstrap_95": _bootstrap_interval(
                    auc_deltas,
                    replicates=bootstrap_replicates,
                    seed=20260820 + partition_index,
                ),
            },
        }

    ratios = [partitions[partition]["normalized_auc"]["picf_over_lbot"] for partition in PARTITIONS]
    endpoint_upper = [
        partitions[partition]["endpoint"]["paired_delta_bootstrap_95"][1]
        for partition in PARTITIONS
    ]
    if any(ratio > 1.02 for ratio in ratios):
        decision = "PICF_ACTION_GATE_FAIL"
    elif all(ratio < 1.0 for ratio in ratios) and all(value < 0 for value in endpoint_upper):
        decision = "PICF_ACTION_ADVANTAGE"
    else:
        decision = "PICF_ACTION_TOLERANCE_ONLY"
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "decision": decision,
        "steps": list(steps),
        "contracts": {field: picf_snapshots[0][field] for field in contract_fields},
        "partitions": partitions,
    }
    return {**payload, "artifact_sha256": _canonical_sha256(payload)}


def main() -> None:
    args = _parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    picf_snapshots = [
        _load_snapshot(
            _snapshot_path(args.picf_run_dir, treatment="picf", step=step),
            expected_schema=PICF_SCHEMAS,
            expected_step=step,
        )
        for step in args.steps
    ]
    lbot_snapshots = [
        _load_snapshot(
            _snapshot_path(args.lbot_run_dir, treatment="lbot", step=step),
            expected_schema=LBOT_SCHEMA,
            expected_step=step,
        )
        for step in args.steps
    ]
    report = compare_curves(
        picf_snapshots=picf_snapshots,
        lbot_snapshots=lbot_snapshots,
        steps=args.steps,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
