#!/usr/bin/env python3
"""Compare the ADR-210 warm posterior against cold PICF and released LingBot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.compare_adr176_matched_action_curves import (
    _bootstrap_interval,
    _canonical_sha256,
)

WARM_SCHEMA = "picf-next.adr210-causal-warm-native-query-action-snapshot/v1"
COLD_SCHEMA = "picf-next.adr207-cold-native-query-action-snapshot/v1"
LBOT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-lbot-snapshot.v1"
REPORT_SCHEMA = "picf-next.adr210-causal-warm-action-gate-comparison/v1"
PARTITIONS = ("validation", "heldout")
CONTRACT_FIELDS = (
    "stream_plan_sha256",
    "representation_split_sha256",
    "evaluation_plan_sha256",
    "lingbot_base_family_sha256",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--lbot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--minimum-relative-reduction", type=float, default=0.02)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, *, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required direct snapshot is absent: {path}")
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"snapshot schema differs: {path}")
    if payload.get("status") != "PASS":
        raise ValueError(f"snapshot did not pass: {path}")
    semantic = dict(payload)
    observed = semantic.pop("artifact_sha256", None)
    if observed != _canonical_sha256(semantic):
        raise ValueError(f"snapshot semantic digest differs: {path}")
    return payload


def _finite_loss(sample: Mapping[str, Any]) -> float:
    value = float(sample.get("action_loss", math.nan))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("action loss must be finite and non-negative")
    return value


def _sample_map(snapshot: Mapping[str, Any], *, name: str) -> dict[str, Mapping[str, Any]]:
    samples = snapshot.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{name} snapshot has no samples")
    result: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError(f"{name} sample is not an object")
        key = sample.get("sample_key")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"{name} sample keys are invalid")
        result[key] = sample
    return result


def _paired_summary(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("paired action summary cannot be empty")
    warm = [_finite_loss(row[0]) for row in rows]
    cold = [_finite_loss(row[1]) for row in rows]
    lbot = [_finite_loss(row[2]) for row in rows]
    warm_cold = [left - right for left, right in zip(warm, cold, strict=True)]
    warm_lbot = [left - right for left, right in zip(warm, lbot, strict=True)]
    cold_lbot = [left - right for left, right in zip(cold, lbot, strict=True)]

    def comparison(
        deltas: list[float],
        *,
        reference_mean: float,
        comparison_seed: int,
    ) -> dict[str, Any]:
        mean_delta = sum(deltas) / len(deltas)
        return {
            "mean_delta": mean_delta,
            "relative_loss_reduction": (
                -mean_delta / reference_mean if reference_mean != 0.0 else None
            ),
            "paired_delta_bootstrap_95": _bootstrap_interval(
                deltas,
                replicates=bootstrap_replicates,
                seed=comparison_seed,
            ),
            "left_wins": sum(delta < 0.0 for delta in deltas),
            "ties": sum(delta == 0.0 for delta in deltas),
            "right_wins": sum(delta > 0.0 for delta in deltas),
        }

    cold_mean = sum(cold) / len(cold)
    lbot_mean = sum(lbot) / len(lbot)
    return {
        "sample_count": len(rows),
        "means": {
            "causal_warm_picf": sum(warm) / len(warm),
            "cold_picf": cold_mean,
            "released_current_frame_lingbot": lbot_mean,
        },
        "warm_minus_cold": comparison(
            warm_cold,
            reference_mean=cold_mean,
            comparison_seed=seed,
        ),
        "warm_minus_lingbot": comparison(
            warm_lbot,
            reference_mean=lbot_mean,
            comparison_seed=seed + 1,
        ),
        "cold_minus_lingbot": comparison(
            cold_lbot,
            reference_mean=lbot_mean,
            comparison_seed=seed + 2,
        ),
    }


def compare_gate(
    *,
    warm: Mapping[str, Any],
    cold: Mapping[str, Any],
    lbot: Mapping[str, Any],
    cold_path: Path | None,
    bootstrap_replicates: int,
    minimum_relative_reduction: float,
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 <= minimum_relative_reduction < 1.0:
        raise ValueError("minimum relative reduction must be in [0,1)")
    expected = (
        (warm, WARM_SCHEMA, "causal_warm_four_past_frames"),
        (cold, COLD_SCHEMA, "cold_reset"),
    )
    for snapshot, schema, state_mode in expected:
        if snapshot.get("schema") != schema or snapshot.get("status") != "PASS":
            raise ValueError("PICF snapshot schema or status differs")
        if snapshot.get("state_mode") != state_mode:
            raise ValueError("PICF state mode differs")
        if snapshot.get("picf_graph_installed") is not True:
            raise ValueError("PICF graph is absent")
    if lbot.get("schema") != LBOT_SCHEMA or lbot.get("status") != "PASS":
        raise ValueError("LingBot snapshot schema or status differs")
    if lbot.get("picf_graph_installed") is not False:
        raise ValueError("released LingBot unexpectedly installs PICF")
    steps = {int(snapshot.get("checkpoint_global_step", -1)) for snapshot in (warm, cold, lbot)}
    if steps != {100}:
        raise ValueError("ADR-210 gate requires three step-100 snapshots")
    if warm.get("history_transitions") != 4:
        raise ValueError("ADR-210 warm history length differs")
    if warm.get("eligible_sample_count") != 94:
        raise ValueError("ADR-210 warm eligible sample count differs")
    excluded = warm.get("excluded_samples")
    if not isinstance(excluded, list) or len(excluded) != 8:
        raise ValueError("ADR-210 excluded cold-start partition differs")
    for field in CONTRACT_FIELDS:
        values = {snapshot.get(field) for snapshot in (warm, cold, lbot)}
        if len(values) != 1 or None in values:
            raise ValueError(f"matched contract differs for {field}")

    if cold_path is not None:
        receipt = warm.get("cold_action_evaluation")
        if not isinstance(receipt, Mapping):
            raise ValueError("warm snapshot lacks its cold receipt")
        embedded = receipt.get("path")
        if not isinstance(embedded, str) or Path(embedded).resolve() != cold_path.resolve():
            raise ValueError("warm snapshot points to another cold artifact")
        if receipt.get("artifact_sha256") != cold.get("artifact_sha256"):
            raise ValueError("warm snapshot references another cold semantic identity")
        if receipt.get("file_sha256") != _file_sha256(cold_path):
            raise ValueError("warm snapshot references another cold file identity")

    warm_by_key = _sample_map(warm, name="warm")
    cold_by_key = _sample_map(cold, name="cold")
    lbot_by_key = _sample_map(lbot, name="LingBot")
    if len(warm_by_key) != 94:
        raise ValueError("warm sample set differs from its eligible count")
    rows: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    for key, warm_sample in warm_by_key.items():
        cold_sample = cold_by_key.get(key)
        lbot_sample = lbot_by_key.get(key)
        if cold_sample is None or lbot_sample is None:
            raise ValueError("warm sample is absent from cold or LingBot")
        for field in (
            "partition",
            "task_key",
            "segment_index",
            "source_episode_index",
            "source_global_index",
            "transition_index",
            "source_digest",
            "model_inputs_sha256",
        ):
            if not (warm_sample.get(field) == cold_sample.get(field) == lbot_sample.get(field)):
                raise ValueError(f"matched sample differs for {field}")
        if warm_sample.get("native_source_rgb_sha256") != cold_sample.get(
            "native_source_rgb_sha256"
        ):
            raise ValueError("warm and cold PICF current RGB differ")
        rows.append((warm_sample, cold_sample, lbot_sample))

    partition_reports: dict[str, Any] = {}
    task_reports: dict[str, Any] = {}
    for partition_index, partition in enumerate(PARTITIONS):
        partition_rows = [row for row in rows if row[0]["partition"] == partition]
        partition_reports[partition] = _paired_summary(
            partition_rows,
            bootstrap_replicates=bootstrap_replicates,
            seed=20260824 + partition_index * 100,
        )
        by_task: dict[str, list[Any]] = defaultdict(list)
        for row in partition_rows:
            by_task[str(row[0]["task_key"])].append(row)
        task_reports[partition] = {
            task: _paired_summary(
                task_rows,
                bootstrap_replicates=bootstrap_replicates,
                seed=20260824 + partition_index * 100 + task_index + 10,
            )
            for task_index, (task, task_rows) in enumerate(sorted(by_task.items()))
        }

    def is_advantage(report: Mapping[str, Any], comparison: str) -> bool:
        value = report[comparison]
        return bool(
            value["relative_loss_reduction"] >= minimum_relative_reduction
            and value["paired_delta_bootstrap_95"][1] < 0.0
        )

    if all(
        is_advantage(partition_reports[partition], comparison)
        for partition in PARTITIONS
        for comparison in ("warm_minus_cold", "warm_minus_lingbot")
    ):
        decision = "AUTHORIZE_30K"
    elif any(
        partition_reports[partition][comparison]["mean_delta"] >= 0.0
        for partition in PARTITIONS
        for comparison in ("warm_minus_cold", "warm_minus_lingbot")
    ):
        decision = "REJECT_30K"
    else:
        decision = "INCONCLUSIVE_NO_30K"

    payload = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "decision": decision,
        "checkpoint_global_step": 100,
        "minimum_relative_reduction": minimum_relative_reduction,
        "bootstrap_replicates": bootstrap_replicates,
        "causal_estimand": "same_checkpoint_warm_picf_minus_cold_picf",
        "system_estimand": (
            "warm_picf_with_four_past_frames_minus_released_lingbot_current_frame"
        ),
        "system_estimand_information_sets_equal": False,
        "contracts": {field: warm[field] for field in CONTRACT_FIELDS},
        "partitions": partition_reports,
        "tasks": task_reports,
    }
    return {**payload, "artifact_sha256": _canonical_sha256(payload)}


def main() -> None:
    args = _parse_args()
    warm = _load(args.warm, schema=WARM_SCHEMA)
    cold = _load(args.cold, schema=COLD_SCHEMA)
    lbot = _load(args.lbot, schema=LBOT_SCHEMA)
    report = compare_gate(
        warm=warm,
        cold=cold,
        lbot=lbot,
        cold_path=args.cold,
        bootstrap_replicates=args.bootstrap_replicates,
        minimum_relative_reduction=args.minimum_relative_reduction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"decision": report["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
