#!/usr/bin/env python3
"""Add an episode-clustered robustness check to the registered ADR-210 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.compare_adr210_causal_warm_action_gate import (
    COLD_SCHEMA,
    LBOT_SCHEMA,
    PARTITIONS,
    REPORT_SCHEMA,
    WARM_SCHEMA,
    _load,
    compare_gate,
)

SCHEMA = "picf-next.adr210-episode-cluster-robustness/v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--lbot", type=Path, required=True)
    parser.add_argument("--formal-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--minimum-relative-reduction", type=float, default=0.02)
    return parser.parse_args()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _episode_summary(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not rows or replicates <= 0:
        raise ValueError("episode-cluster bootstrap inputs are empty")
    grouped_delta: dict[int, list[float]] = defaultdict(list)
    grouped_reference: dict[int, list[float]] = defaultdict(list)
    for left, right in rows:
        episode = int(left["source_episode_index"])
        left_loss = float(left["action_loss"])
        right_loss = float(right["action_loss"])
        if not all(math.isfinite(value) and value >= 0.0 for value in (left_loss, right_loss)):
            raise ValueError("clustered action losses must be finite and non-negative")
        grouped_delta[episode].append(left_loss - right_loss)
        grouped_reference[episode].append(right_loss)
    episodes = sorted(grouped_delta)
    cluster_deltas = [math.fsum(grouped_delta[key]) / len(grouped_delta[key]) for key in episodes]
    cluster_references = [
        math.fsum(grouped_reference[key]) / len(grouped_reference[key]) for key in episodes
    ]
    estimate = math.fsum(cluster_deltas) / len(cluster_deltas)
    reference = math.fsum(cluster_references) / len(cluster_references)
    generator = random.Random(seed)
    bootstrap = []
    for _ in range(replicates):
        selected = generator.choices(range(len(episodes)), k=len(episodes))
        bootstrap.append(math.fsum(cluster_deltas[index] for index in selected) / len(selected))
    return {
        "cluster_unit": "source_episode_index",
        "cluster_count": len(episodes),
        "sample_count": len(rows),
        "mean_cluster_delta": estimate,
        "mean_cluster_reference_loss": reference,
        "relative_loss_reduction": -estimate / reference if reference else None,
        "paired_cluster_bootstrap_95": [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "method": "paired source-episode nonparametric percentile bootstrap",
    }


def compare_cluster_robustness(
    *,
    warm: Mapping[str, Any],
    cold: Mapping[str, Any],
    lbot: Mapping[str, Any],
    formal_gate: Mapping[str, Any],
    bootstrap_replicates: int,
    minimum_relative_reduction: float,
) -> dict[str, Any]:
    if formal_gate.get("schema") != REPORT_SCHEMA or formal_gate.get("status") != "PASS":
        raise ValueError("registered ADR-210 gate report is invalid")
    semantic = dict(formal_gate)
    observed = semantic.pop("artifact_sha256", None)
    if observed != _canonical_sha256(semantic):
        raise ValueError("registered ADR-210 gate digest differs")
    # Reuse the registered validator; the lower replicate count is sufficient
    # because its statistics are not republished by this robustness report.
    validated = compare_gate(
        warm=warm,
        cold=cold,
        lbot=lbot,
        cold_path=None,
        bootstrap_replicates=100,
        minimum_relative_reduction=minimum_relative_reduction,
    )
    if validated["decision"] != formal_gate["decision"]:
        raise ValueError("registered gate decision cannot be reproduced")

    warm_by_key = {sample["sample_key"]: sample for sample in warm["samples"]}
    cold_by_key = {sample["sample_key"]: sample for sample in cold["samples"]}
    lbot_by_key = {sample["sample_key"]: sample for sample in lbot["samples"]}
    reports: dict[str, Any] = {}
    for partition_index, partition in enumerate(PARTITIONS):
        keys = sorted(
            key for key, sample in warm_by_key.items() if sample["partition"] == partition
        )
        warm_cold = [(warm_by_key[key], cold_by_key[key]) for key in keys]
        warm_lbot = [(warm_by_key[key], lbot_by_key[key]) for key in keys]
        reports[partition] = {
            "warm_minus_cold": _episode_summary(
                warm_cold,
                replicates=bootstrap_replicates,
                seed=20260824 + partition_index * 100,
            ),
            "warm_minus_lingbot": _episode_summary(
                warm_lbot,
                replicates=bootstrap_replicates,
                seed=20260825 + partition_index * 100,
            ),
        }

    def passes(value: Mapping[str, Any]) -> bool:
        return bool(
            value["relative_loss_reduction"] >= minimum_relative_reduction
            and value["paired_cluster_bootstrap_95"][1] < 0.0
        )

    cluster_pass = all(
        passes(reports[partition][comparison])
        for partition in PARTITIONS
        for comparison in ("warm_minus_cold", "warm_minus_lingbot")
    )
    if formal_gate["decision"] != "AUTHORIZE_30K":
        decision = "FORMAL_NO_GO"
    elif cluster_pass:
        decision = "ROBUST_AUTHORIZE_30K"
    else:
        decision = "FORMAL_PASS_CLUSTER_FAIL"
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "decision": decision,
        "formal_gate_decision": formal_gate["decision"],
        "formal_gate_artifact_sha256": formal_gate["artifact_sha256"],
        "minimum_relative_reduction": minimum_relative_reduction,
        "partitions": reports,
        "claim_scope": (
            "secondary pre-result robustness check; preserves the registered sample-level "
            "gate and prevents correlated frames from supporting a strong success claim"
        ),
    }
    return {**payload, "artifact_sha256": _canonical_sha256(payload)}


def main() -> None:
    args = _parse_args()
    warm = _load(args.warm, schema=WARM_SCHEMA)
    cold = _load(args.cold, schema=COLD_SCHEMA)
    lbot = _load(args.lbot, schema=LBOT_SCHEMA)
    formal_gate = json.loads(args.formal_gate.read_text(encoding="ascii"))
    report = compare_cluster_robustness(
        warm=warm,
        cold=cold,
        lbot=lbot,
        formal_gate=formal_gate,
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
