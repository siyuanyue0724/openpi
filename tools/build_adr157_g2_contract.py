#!/usr/bin/env python3
"""Build label-separated ADR-157 fixed-observation G2 contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _build_pair(
    *,
    plan_path: Path,
    rebind_path: Path,
    coverage_path: Path,
    output_dir: Path,
    name: str,
    per_partition: int,
    world_size: int,
) -> None:
    plan = _load(plan_path)
    rebind = _load(rebind_path)
    coverage = _load(coverage_path)
    if plan.get("schema") != "picf-next.lingbot-fixed-observation-evaluation-plan.v2":
        raise ValueError("fixed-observation source plan schema differs")
    if rebind.get("schema") != "picf-next.adr154-current-dataset-fixed-x-source-rebind/v1":
        raise ValueError("current-source rebind schema differs")
    if rebind.get("status") != "PASS" or rebind.get("validated_item_count") != len(
        plan.get("items", ())
    ):
        raise ValueError("current-source rebind is incomplete")
    if rebind.get("old_plan_file_sha256") != _sha256(plan_path):
        raise ValueError("current-source rebind names a different source plan")
    if coverage.get("schema") != "picf-next.dense-evidence-coverage-plan/v1":
        raise ValueError("dense-evidence coverage plan schema differs")
    if coverage.get("dataset_tree_sha256") != rebind.get("current_dataset_tree_sha256"):
        raise ValueError("dense-evidence coverage belongs to another current dataset")
    coverage_records = coverage.get("records")
    if not isinstance(coverage_records, list) or not coverage_records:
        raise ValueError("dense-evidence coverage plan has no records")
    covered_source_indices = {
        record["source_global_index"]
        for record in coverage_records
        if isinstance(record, dict)
        and isinstance(record.get("source_global_index"), int)
        and not isinstance(record.get("source_global_index"), bool)
    }
    if len(covered_source_indices) != len(coverage_records):
        raise ValueError("dense-evidence coverage contains invalid or duplicate source indices")

    rebound = {(row["partition"], row["ordinal"]): row for row in rebind.get("rows", ())}
    selected_by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition in ("validation", "heldout"):
        eligible = []
        for item in plan["items"]:
            if item["partition"] != partition:
                continue
            source_global_index = item["group"].get("source_global_index")
            if source_global_index not in covered_source_indices:
                continue
            variants = item.get("variants")
            if not isinstance(variants, list) or len(variants) != 2:
                raise ValueError("G2 requires exactly two prompt variants")
            masses = tuple(float(variant["target_mass"]) for variant in variants)
            if any(not math.isfinite(mass) or mass <= 0.0 for mass in masses):
                raise ValueError("G2 target masses must be finite and positive")
            mass_log_gap = abs(math.log(masses[0] / masses[1]))
            eligible.append((mass_log_gap, item["ordinal"], item))
        items = [item for _gap, _ordinal, item in sorted(eligible)[:per_partition]]
        if len(items) != per_partition:
            raise ValueError(
                f"partition {partition!r} has too few full-modal fixed-observation items"
            )
        selected_by_partition[partition] = items

    selected: list[dict[str, Any]] = []
    for offset in range(per_partition):
        selected.extend(
            selected_by_partition[partition][offset] for partition in ("validation", "heldout")
        )

    execution_items = []
    label_items = []
    selected_source_indices: set[int] = set()
    selected_sample_keys: set[str] = set()
    for index, item in enumerate(selected):
        partition = item["partition"]
        ordinal = item["ordinal"]
        group = item["group"]
        row = rebound[(partition, ordinal)]
        sample_key = group["stateful_sample_key"]
        source_global_index = group["source_global_index"]
        if source_global_index not in covered_source_indices:
            raise ValueError("selected G2 source lacks complete dense-evidence coverage")
        if source_global_index in selected_source_indices or sample_key in selected_sample_keys:
            raise ValueError("G2 selected duplicate fixed observations")
        selected_source_indices.add(source_global_index)
        selected_sample_keys.add(sample_key)
        if row["sample_key"] != sample_key:
            raise ValueError("rebound sample key differs from the fixed-X source")
        source_sensors = dict(row["source_sensor_sha256"])
        if source_sensors != group["source_sensor_sha256"]:
            raise ValueError("rebound source sensors differ from the fixed-X source")
        if row["source_state_sha256"] != group["source_state_sha256"]:
            raise ValueError("rebound source state differs from the fixed-X source")
        variants = item["variants"]
        if len(variants) != 2:
            raise ValueError("G2 requires exactly two prompt variants")
        if variants[0]["target_identity_key"] == variants[1]["target_identity_key"]:
            raise ValueError("G2 prompt variants retained one target identity")
        item_id = f"{partition}-{ordinal:04d}"
        execution_items.append(
            {
                "execution_rank": index % world_size,
                "item_id": item_id,
                "ordinal": ordinal,
                "partition": partition,
                "prompts": [
                    {
                        "instruction": variant["instruction"],
                        "instruction_sha256": variant["instruction_sha256"],
                        "name": f"{item_id}/prompt-{prompt_index}",
                        "task_key": variant["task_key"],
                    }
                    for prompt_index, variant in enumerate(variants)
                ],
                "replay_seed": item["replay_seed"],
                "sample_key": sample_key,
                "source_global_index": source_global_index,
                "source_sensor_sha256": source_sensors,
                "source_state_sha256": row["source_state_sha256"],
            }
        )
        label_items.append(
            {
                "item_id": item_id,
                "prompts": [
                    {
                        "name": f"{item_id}/prompt-{prompt_index}",
                        "target_identity_key": variant["target_identity_key"],
                        "target_mass": variant["target_mass"],
                    }
                    for prompt_index, variant in enumerate(variants)
                ],
            }
        )

    provenance = {
        "current_dataset_manifest_file_sha256": rebind["current_dataset_manifest_file_sha256"],
        "current_dataset_tree_sha256": rebind["current_dataset_tree_sha256"],
        "dense_evidence_coverage_artifact_sha256": coverage["artifact_sha256"],
        "dense_evidence_coverage_file_sha256": _sha256(coverage_path),
        "dense_evidence_coverage_records_sha256": coverage["records_sha256"],
        "selection_policy": "full_modal_mass_balanced_v1",
        "source_plan_file_sha256": _sha256(plan_path),
        "source_rebind_file_sha256": _sha256(rebind_path),
    }
    execution = {
        "item_count": len(execution_items),
        "items": execution_items,
        "name": name,
        "provenance": provenance,
        "schema": "picf-next.adr157-g2-label-free-execution/v1",
        "world_size": world_size,
    }
    labels = {
        "item_count": len(label_items),
        "items": label_items,
        "name": name,
        "schema": "picf-next.adr157-g2-offline-labels/v1",
        "source_execution_sha256": hashlib.sha256(_canonical_bytes(execution)).hexdigest(),
    }
    forbidden = {"target_identity_key", "target_mass", "target_row", "sidecar", "labels"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    retained = sorted(forbidden & keys(execution))
    if retained:
        raise RuntimeError(f"label-free G2 execution retained target fields: {retained}")
    _write(output_dir / f"{name}.execution.json", execution)
    _write(output_dir / f"{name}.labels.json", labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--dense-evidence-coverage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    if args.world_size != 4:
        raise ValueError("ADR-157 G2 is registered for exactly four ranks")
    _build_pair(
        plan_path=args.evaluation_plan,
        rebind_path=args.source_rebind,
        coverage_path=args.dense_evidence_coverage,
        output_dir=args.output_dir,
        name="g2-smoke-4",
        per_partition=2,
        world_size=args.world_size,
    )
    _build_pair(
        plan_path=args.evaluation_plan,
        rebind_path=args.source_rebind,
        coverage_path=args.dense_evidence_coverage,
        output_dir=args.output_dir,
        name="g2-pilot-8",
        per_partition=4,
        world_size=args.world_size,
    )
    _build_pair(
        plan_path=args.evaluation_plan,
        rebind_path=args.source_rebind,
        coverage_path=args.dense_evidence_coverage,
        output_dir=args.output_dir,
        name="g2-compact-16",
        per_partition=8,
        world_size=args.world_size,
    )


if __name__ == "__main__":
    main()
