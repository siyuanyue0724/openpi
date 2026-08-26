#!/usr/bin/env python3
"""Strict paired comparison of accepted G2b and post-G3 retention reports.

This tool is intentionally report-only.  It validates that both reports refer
to the same frozen examples and physical identities, aligns their arbitrary
row gauges through those identities, and reports paired continuous changes.
It does not turn the G3 runner's absolute-threshold PASS into a scientific
retention conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError

G2_REPRESENTATION_SCHEMA = "picf-next.ltop-g2-shared-representation.v3"
G3_RETENTION_SCHEMA = "picf-next.ltop-g3-representation-retention.v1"
OUTPUT_SCHEMA = "picf-next.ltop-g2-g3-retention-paired-comparison.v1"
PARTITIONS = ("validation", "heldout")
PROMPTS_PER_SCENE = 2
EXPECTED_WORLD_SIZE = 2
EXPECTED_CAPACITY = 16
EXPECTED_TASK_QUERY_COUNT = 4
EXPECTED_SCENES_PER_RANK_PER_PARTITION = 4
DEFAULT_BOOTSTRAP_SAMPLES = 100_000
DEFAULT_BOOTSTRAP_SEED = 20260813
_ABSOLUTE_TOLERANCE = 1.0e-9
_RELATIVE_TOLERANCE = 1.0e-7


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2-report", type=Path, required=True)
    parser.add_argument("--g3-retention-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--maximum-positive-prompt-count-regression-per-partition",
        type=int,
        default=0,
        help=(
            "Reported count gate relative to each partition's observed G2b count; "
            "does not authorize a scientific PASS."
        ),
    )
    return parser.parse_args()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"retention comparison {name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _mapping_list(value: object, *, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ContractError(f"retention comparison {name} must be a list of mappings")
    return cast(list[Mapping[str, Any]], value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"retention comparison {name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"retention comparison {name} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"retention comparison {name} must be at least {minimum}")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"retention comparison {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"retention comparison {name} must be finite")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ContractError(f"retention comparison {name} must lie in [0, 1]")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_RELATIVE_TOLERANCE,
        abs_tol=_ABSOLUTE_TOLERANCE,
    )


def _require_close(left: float, right: float, *, name: str) -> None:
    if not _close(left, right):
        raise ContractError(
            f"retention comparison {name} is inconsistent: reported={left!r}, recomputed={right!r}"
        )


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise ContractError(f"retention comparison {name} is empty")
    return sum(values) / len(values)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_report(path: Path, *, name: str) -> tuple[Mapping[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"retention comparison {name} must be a regular non-symlink file")
    payload = path.read_bytes()
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ContractError(f"retention comparison {name} is not valid JSON") from error
    return _mapping(decoded, name=name), _sha256_bytes(payload)


def _binding_map(value: object, *, name: str, capacity: int) -> dict[str, int]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"retention comparison {name} must be a non-empty binding list")
    result: dict[str, int] = {}
    rows: set[int] = set()
    for index, pair in enumerate(value):
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise ContractError(f"retention comparison {name}[{index}] must be [identity, row]")
        identity = _string(pair[0], name=f"{name}[{index}].identity")
        row = _integer(pair[1], name=f"{name}[{index}].row", minimum=0)
        if row >= capacity:
            raise ContractError(f"retention comparison {name}[{index}].row exceeds capacity")
        if identity in result or row in rows:
            raise ContractError(f"retention comparison {name} is not one-to-one")
        result[identity] = row
        rows.add(row)
    return result


def _prompt_metrics(
    value: Mapping[str, Any],
    *,
    name: str,
    expected_target_row: int,
    expected_alternate_row: int,
    capacity: int,
) -> dict[str, Any]:
    target_row = _integer(value.get("target_row"), name=f"{name}.target_row", minimum=0)
    alternate_row = _integer(
        value.get("alternate_row"),
        name=f"{name}.alternate_row",
        minimum=0,
    )
    if target_row != expected_target_row or alternate_row != expected_alternate_row:
        raise ContractError(f"retention comparison {name} target/alternate row binding changed")
    target_coverage = _probability(value.get("target_coverage"), name=f"{name}.target_coverage")
    alternate_coverage = _probability(
        value.get("alternate_coverage"),
        name=f"{name}.alternate_coverage",
    )
    margin = _finite(value.get("margin"), name=f"{name}.margin")
    _require_close(
        margin,
        target_coverage - alternate_coverage,
        name=f"{name}.margin",
    )
    top_row = _integer(value.get("top_row"), name=f"{name}.top_row", minimum=0)
    if top_row >= capacity:
        raise ContractError(f"retention comparison {name}.top_row exceeds capacity")
    distribution_value = value.get("mean_row_distribution")
    if not isinstance(distribution_value, list) or len(distribution_value) != capacity:
        raise ContractError(
            f"retention comparison {name}.mean_row_distribution must match capacity"
        )
    distribution = [
        _probability(entry, name=f"{name}.mean_row_distribution[{index}]")
        for index, entry in enumerate(distribution_value)
    ]
    if top_row != max(range(capacity), key=distribution.__getitem__):
        raise ContractError(f"retention comparison {name}.top_row is inconsistent")
    return {
        "alternate_coverage": alternate_coverage,
        "alternate_row": alternate_row,
        "margin": margin,
        "target_coverage": target_coverage,
        "target_nll": -math.log(max(target_coverage, 1.0e-30)),
        "target_row": target_row,
        "top_row": top_row,
    }


def _scene_metrics(
    scene: Mapping[str, Any],
    *,
    name: str,
    capacity: int,
) -> dict[str, Any]:
    item_id = _string(scene.get("item_id"), name=f"{name}.item_id")
    sample_key = _string(scene.get("sample_key"), name=f"{name}.sample_key")
    identities_value = scene.get("target_identities")
    if not isinstance(identities_value, list) or len(identities_value) != PROMPTS_PER_SCENE:
        raise ContractError(
            f"retention comparison {name}.target_identities must contain two prompts"
        )
    target_identities = [
        _string(identity, name=f"{name}.target_identities[{index}]")
        for index, identity in enumerate(identities_value)
    ]
    if len(set(target_identities)) != PROMPTS_PER_SCENE:
        raise ContractError(f"retention comparison {name} target identities must be distinct")

    rows_value = scene.get("target_rows")
    if not isinstance(rows_value, list) or len(rows_value) != PROMPTS_PER_SCENE:
        raise ContractError(f"retention comparison {name}.target_rows must contain two rows")
    target_rows = [
        _integer(row, name=f"{name}.target_rows[{index}]", minimum=0)
        for index, row in enumerate(rows_value)
    ]
    if any(row >= capacity for row in target_rows) or len(set(target_rows)) != PROMPTS_PER_SCENE:
        raise ContractError(f"retention comparison {name} target rows are invalid")

    bindings_by_prompt = scene.get("bindings_by_prompt")
    independent_by_prompt = scene.get("independent_bindings_by_prompt")
    if (
        not isinstance(bindings_by_prompt, list)
        or not isinstance(independent_by_prompt, list)
        or len(bindings_by_prompt) != PROMPTS_PER_SCENE
        or len(independent_by_prompt) != PROMPTS_PER_SCENE
    ):
        raise ContractError(f"retention comparison {name} binding prompt axes are incomplete")
    applied = [
        _binding_map(value, name=f"{name}.bindings_by_prompt[{index}]", capacity=capacity)
        for index, value in enumerate(bindings_by_prompt)
    ]
    independent = [
        _binding_map(
            value,
            name=f"{name}.independent_bindings_by_prompt[{index}]",
            capacity=capacity,
        )
        for index, value in enumerate(independent_by_prompt)
    ]
    canonical = applied[0]
    if any(value != canonical for value in (*applied, *independent)):
        raise ContractError(f"retention comparison {name} does not preserve one shared row gauge")
    if scene.get("shared_row_gauge") is not True:
        raise ContractError(f"retention comparison {name}.shared_row_gauge is not true")
    for index, identity in enumerate(target_identities):
        if canonical.get(identity) != target_rows[index]:
            raise ContractError(
                f"retention comparison {name} target identity is not bound to its target row"
            )

    prompts = _mapping_list(scene.get("prompts"), name=f"{name}.prompts")
    if len(prompts) != PROMPTS_PER_SCENE:
        raise ContractError(f"retention comparison {name}.prompts must contain two prompts")
    prompt_metrics = [
        _prompt_metrics(
            prompt,
            name=f"{name}.prompts[{index}]",
            expected_target_row=target_rows[index],
            expected_alternate_row=target_rows[1 - index],
            capacity=capacity,
        )
        for index, prompt in enumerate(prompts)
    ]
    margins = [float(prompt["margin"]) for prompt in prompt_metrics]
    nlls = [float(prompt["target_nll"]) for prompt in prompt_metrics]
    mean_margin = _finite(scene.get("mean_margin"), name=f"{name}.mean_margin")
    positive_count = _integer(
        scene.get("positive_margin_count"),
        name=f"{name}.positive_margin_count",
        minimum=0,
    )
    mean_target_nll = _finite(
        scene.get("mean_target_nll"),
        name=f"{name}.mean_target_nll",
    )
    _require_close(mean_margin, _mean(margins, name=f"{name}.margins"), name=f"{name}.mean_margin")
    if positive_count != sum(margin > 0.0 for margin in margins):
        raise ContractError(f"retention comparison {name}.positive_margin_count is inconsistent")
    _require_close(
        mean_target_nll,
        _mean(nlls, name=f"{name}.target_nlls"),
        name=f"{name}.mean_target_nll",
    )
    physical_set_loss = _finite(
        scene.get("mean_physical_set_loss"),
        name=f"{name}.mean_physical_set_loss",
    )
    if physical_set_loss < 0.0:
        raise ContractError(f"retention comparison {name}.mean_physical_set_loss is negative")
    physical_drift = _finite(
        scene.get("physical_prompt_drift_max_abs"),
        name=f"{name}.physical_prompt_drift_max_abs",
    )
    if physical_drift < 0.0:
        raise ContractError(
            f"retention comparison {name}.physical_prompt_drift_max_abs is negative"
        )
    self_checks = _mapping(scene.get("metric_self_checks"), name=f"{name}.metric_self_checks")
    permutation_error = _finite(
        self_checks.get("matched_row_permutation_max_abs_error"),
        name=f"{name}.matched_row_permutation_max_abs_error",
    )
    if permutation_error < 0.0:
        raise ContractError(f"retention comparison {name} permutation error is negative")
    return {
        "binding_by_identity": canonical,
        "item_id": item_id,
        "mean_margin": mean_margin,
        "mean_physical_set_loss": physical_set_loss,
        "mean_target_nll": mean_target_nll,
        "physical_prompt_drift_max_abs": physical_drift,
        "positive_margin_count": positive_count,
        "prompts": prompt_metrics,
        "sample_key": sample_key,
        "target_identities": target_identities,
        "target_rows": target_rows,
    }


def _partition_metrics(
    partition: Mapping[str, Any],
    *,
    name: str,
    capacity: int,
) -> dict[str, Any]:
    scenes_value = _mapping_list(partition.get("scenes"), name=f"{name}.scenes")
    scenes: dict[str, dict[str, Any]] = {}
    for index, scene in enumerate(scenes_value):
        parsed = _scene_metrics(scene, name=f"{name}.scenes[{index}]", capacity=capacity)
        item_id = str(parsed["item_id"])
        if item_id in scenes:
            raise ContractError(f"retention comparison {name} duplicates item_id {item_id!r}")
        scenes[item_id] = parsed
    if not scenes:
        raise ContractError(f"retention comparison {name} has no scenes")

    prompt_rows = [prompt for scene in scenes.values() for prompt in scene["prompts"]]
    flattened_report_prompts = _mapping_list(
        partition.get("prompts"),
        name=f"{name}.prompts",
    )
    if flattened_report_prompts != [
        prompt
        for scene in scenes_value
        for prompt in _mapping_list(scene.get("prompts"), name=name)
    ]:
        raise ContractError(f"retention comparison {name}.prompts does not flatten scene prompts")
    scene_count = _integer(partition.get("scene_count"), name=f"{name}.scene_count", minimum=1)
    prompt_count = _integer(partition.get("prompt_count"), name=f"{name}.prompt_count", minimum=1)
    if scene_count != len(scenes) or prompt_count != len(prompt_rows):
        raise ContractError(f"retention comparison {name} count fields are inconsistent")
    if prompt_count != PROMPTS_PER_SCENE * scene_count:
        raise ContractError(f"retention comparison {name} prompt/scene ratio changed")

    margins = [float(prompt["margin"]) for prompt in prompt_rows]
    mean_margin = _finite(partition.get("mean_margin"), name=f"{name}.mean_margin")
    positive_count = _integer(
        partition.get("positive_margin_count"),
        name=f"{name}.positive_margin_count",
        minimum=0,
    )
    mean_target_nll = _finite(
        partition.get("mean_target_nll"),
        name=f"{name}.mean_target_nll",
    )
    mean_physical_set_loss = _finite(
        partition.get("mean_physical_set_loss"),
        name=f"{name}.mean_physical_set_loss",
    )
    _require_close(mean_margin, _mean(margins, name=f"{name}.margins"), name=f"{name}.mean_margin")
    if positive_count != sum(margin > 0.0 for margin in margins):
        raise ContractError(f"retention comparison {name}.positive_margin_count is inconsistent")
    _require_close(
        mean_target_nll,
        _mean(
            [float(scene["mean_target_nll"]) for scene in scenes.values()],
            name=f"{name}.scene_target_nlls",
        ),
        name=f"{name}.mean_target_nll",
    )
    _require_close(
        mean_physical_set_loss,
        _mean(
            [float(scene["mean_physical_set_loss"]) for scene in scenes.values()],
            name=f"{name}.scene_physical_losses",
        ),
        name=f"{name}.mean_physical_set_loss",
    )
    physical_drift = _finite(
        partition.get("physical_prompt_drift_max_abs"),
        name=f"{name}.physical_prompt_drift_max_abs",
    )
    _require_close(
        physical_drift,
        max(float(scene["physical_prompt_drift_max_abs"]) for scene in scenes.values()),
        name=f"{name}.physical_prompt_drift_max_abs",
    )
    self_checks = _mapping(partition.get("metric_self_checks"), name=f"{name}.metric_self_checks")
    permutation_error = _finite(
        self_checks.get("matched_row_permutation_max_abs_error"),
        name=f"{name}.matched_row_permutation_max_abs_error",
    )
    _require_close(
        permutation_error,
        max(
            _finite(
                _mapping(
                    scene.get("metric_self_checks"),
                    name=f"{name}.scene.metric_self_checks",
                ).get("matched_row_permutation_max_abs_error"),
                name=f"{name}.scene.matched_row_permutation_max_abs_error",
            )
            for scene in scenes_value
        ),
        name=f"{name}.matched_row_permutation_max_abs_error",
    )
    if partition.get("shared_row_gauge") is not True:
        raise ContractError(f"retention comparison {name}.shared_row_gauge is not true")
    return {
        "mean_margin": mean_margin,
        "mean_physical_set_loss": mean_physical_set_loss,
        "mean_target_nll": mean_target_nll,
        "positive_margin_count": positive_count,
        "prompt_count": prompt_count,
        "scene_count": scene_count,
        "scenes": scenes,
    }


def _rank_partitions(
    report: Mapping[str, Any],
    *,
    report_name: str,
    is_g2: bool,
    capacity: int,
) -> dict[int, dict[str, dict[str, Any]]]:
    ranks = _mapping_list(report.get("rank_reports"), name=f"{report_name}.rank_reports")
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for index, rank_report in enumerate(ranks):
        rank = _integer(
            rank_report.get("rank"), name=f"{report_name}.rank_reports[{index}].rank", minimum=0
        )
        if rank in result:
            raise ContractError(f"retention comparison {report_name} duplicates rank {rank}")
        history = _mapping_list(
            rank_report.get("history"),
            name=f"{report_name}.rank_reports[{index}].history",
        )
        if not history:
            raise ContractError(f"retention comparison {report_name} rank {rank} has no history")
        if not is_g2 and len(history) != 1:
            raise ContractError(
                f"retention comparison G3 retention rank {rank} must have one cold read"
            )
        final = history[-1]
        expected_step = _integer(report.get("steps"), name=f"{report_name}.steps", minimum=0)
        if (
            _integer(final.get("step"), name=f"{report_name}.rank[{rank}].final.step", minimum=0)
            != expected_step
        ):
            raise ContractError(
                f"retention comparison {report_name} rank {rank} final step changed"
            )
        partitions = {
            partition: _partition_metrics(
                _mapping(final.get(partition), name=f"{report_name}.rank[{rank}].{partition}"),
                name=f"{report_name}.rank[{rank}].{partition}",
                capacity=capacity,
            )
            for partition in PARTITIONS
        }
        for partition, parsed in partitions.items():
            if parsed["scene_count"] != EXPECTED_SCENES_PER_RANK_PER_PARTITION:
                raise ContractError(
                    f"retention comparison {report_name} rank {rank} {partition} "
                    "does not contain four full scenes"
                )
        if is_g2:
            local_items = _mapping(
                rank_report.get("local_items"),
                name=f"{report_name}.rank[{rank}].local_items",
            )
            for partition in PARTITIONS:
                items = _mapping_list(
                    local_items.get(partition),
                    name=f"{report_name}.rank[{rank}].local_items.{partition}",
                )
                if len(items) != len({str(item.get("item_id")) for item in items}):
                    raise ContractError(
                        f"retention comparison G2 rank {rank} {partition} local_items duplicate IDs"
                    )
                expected = {
                    str(item.get("item_id")): (
                        item.get("sample_key"),
                        item.get("target_identities"),
                    )
                    for item in items
                }
                observed = {
                    item_id: (scene["sample_key"], scene["target_identities"])
                    for item_id, scene in partitions[partition]["scenes"].items()
                }
                if expected != observed:
                    raise ContractError(
                        f"retention comparison G2 rank {rank} {partition} local_items changed"
                    )
        result[rank] = partitions
    declared_world_size = _integer(
        report.get("world_size"), name=f"{report_name}.world_size", minimum=1
    )
    if declared_world_size != EXPECTED_WORLD_SIZE:
        raise ContractError(
            f"retention comparison {report_name} must use the two-rank G2b/G3 contract"
        )
    if set(result) != set(range(declared_world_size)):
        raise ContractError(f"retention comparison {report_name} rank axis is incomplete")
    return result


def _validate_report_pair(
    g2: Mapping[str, Any],
    g3: Mapping[str, Any],
    *,
    g2_report_sha256: str,
) -> tuple[int, dict[int, dict[str, dict[str, Any]]], dict[int, dict[str, dict[str, Any]]]]:
    if g2.get("schema") != G2_REPRESENTATION_SCHEMA:
        raise ContractError("retention comparison G2 report schema changed")
    if g3.get("schema") != G3_RETENTION_SCHEMA:
        raise ContractError("retention comparison G3 retention report schema changed")
    if g2.get("status") != "PASS" or g2.get("failures") != []:
        raise ContractError("retention comparison requires the accepted passing G2b report")
    if g2.get("training_scope") != "representation":
        raise ContractError("retention comparison G2 report is not representation-only")
    if g3.get("phase") != "retention":
        raise ContractError("retention comparison G3 report is not a cold retention read")
    if g3.get("g2_report_sha256") != g2_report_sha256:
        raise ContractError("retention comparison G3 report is not bound to the exact G2 report")
    if g3.get("status") not in {"PASS", "FAIL"}:
        raise ContractError("retention comparison G3 runner status is invalid")
    if not isinstance(g3.get("failures"), list):
        raise ContractError("retention comparison G3 failures field is invalid")

    for field in ("architecture_identity", "world_size", "capacity", "task_query_count"):
        if g2.get(field) != g3.get(field):
            raise ContractError(f"retention comparison common {field} changed")
    if g2.get("capacity") != EXPECTED_CAPACITY:
        raise ContractError("retention comparison capacity differs from the accepted G2b contract")
    if g2.get("task_query_count") != EXPECTED_TASK_QUERY_COUNT:
        raise ContractError(
            "retention comparison task-query count differs from the accepted G2b contract"
        )
    if g2.get("dataset_contract") != g3.get("dataset_contract"):
        raise ContractError("retention comparison dataset contract changed")
    g2_inputs = _mapping(g2.get("input_sha256"), name="G2.input_sha256")
    digest_pairs = {
        "execution_contract": "execution_contract_sha256",
        "offline_labels": "offline_labels_sha256",
        "physical_sidecar_manifest": "physical_sidecar_manifest_sha256",
    }
    for g2_field, g3_field in digest_pairs.items():
        if g2_inputs.get(g2_field) != g3.get(g3_field):
            raise ContractError(f"retention comparison frozen {g2_field} changed")
    retention_contract = _mapping(
        g3.get("representation_retention_contract"),
        name="G3.representation_retention_contract",
    )
    if retention_contract.get("optimizer_updates") != 0:
        raise ContractError("retention comparison G3 report includes optimizer updates")
    if retention_contract.get("scientific_action_evidence") is not False:
        raise ContractError("retention comparison G3 retention contract changed purpose")
    if (
        retention_contract.get("scenes_per_rank_per_partition")
        != EXPECTED_SCENES_PER_RANK_PER_PARTITION
        or retention_contract.get("crossed_prompts_per_scene") != PROMPTS_PER_SCENE
    ):
        raise ContractError("retention comparison G3 retention scene contract changed")

    capacity = _integer(g2.get("capacity"), name="capacity", minimum=2)
    g2_ranks = _rank_partitions(g2, report_name="G2", is_g2=True, capacity=capacity)
    g3_ranks = _rank_partitions(g3, report_name="G3", is_g2=False, capacity=capacity)
    if set(g2_ranks) != set(g3_ranks):
        raise ContractError("retention comparison rank identities changed")
    return capacity, g2_ranks, g3_ranks


def _derived_seed(base_seed: int, *, partition: str, metric: str) -> int:
    payload = f"{base_seed}:{partition}:{metric}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bootstrap_ci(
    values: Sequence[float],
    *,
    base_seed: int,
    bootstrap_samples: int,
    partition: str,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ContractError("retention comparison bootstrap_samples must be at least 100")
    if not values:
        raise ContractError(f"retention comparison {partition} {metric} has no scenes")
    seed = _derived_seed(base_seed, partition=partition, metric=metric)
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(bootstrap_samples)
    )
    lower = estimates[int(0.025 * bootstrap_samples)]
    upper = estimates[int(0.975 * bootstrap_samples) - 1]
    return {
        "bootstrap_95_percent_ci": [lower, upper],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "direction": direction,
        "mean_delta": _mean(values, name=f"{partition} {metric} deltas"),
        "method": "paired-scene nonparametric percentile bootstrap",
        "scene_count": count,
    }


def compare_lingbot_vla2_ltop_g3_retention_reports(
    g2: Mapping[str, Any],
    g3: Mapping[str, Any],
    *,
    g2_report_sha256: str,
    g3_retention_report_sha256: str,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    maximum_positive_prompt_count_regression_per_partition: int = 0,
) -> dict[str, Any]:
    """Return a strict paired report without declaring a scientific PASS."""

    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ContractError("retention comparison bootstrap_seed must be an integer")
    maximum_regression = _integer(
        maximum_positive_prompt_count_regression_per_partition,
        name="maximum_positive_prompt_count_regression_per_partition",
        minimum=0,
    )
    _integer(bootstrap_samples, name="bootstrap_samples", minimum=100)
    capacity, g2_ranks, g3_ranks = _validate_report_pair(
        g2,
        g3,
        g2_report_sha256=g2_report_sha256,
    )

    partition_reports: dict[str, Any] = {}
    all_alignment_checks: list[bool] = []
    all_count_gates: list[bool] = []
    for partition in PARTITIONS:
        scene_rows = []
        prompt_rows = []
        seen_item_ids: set[str] = set()
        for rank in sorted(g2_ranks):
            g2_partition = g2_ranks[rank][partition]
            g3_partition = g3_ranks[rank][partition]
            if set(g2_partition["scenes"]) != set(g3_partition["scenes"]):
                raise ContractError(
                    f"retention comparison rank {rank} {partition} item_id axis changed"
                )
            for item_id in sorted(g2_partition["scenes"]):
                if item_id in seen_item_ids:
                    raise ContractError(
                        f"retention comparison {partition} duplicates item_id {item_id!r} "
                        "across ranks"
                    )
                seen_item_ids.add(item_id)
                baseline = g2_partition["scenes"][item_id]
                candidate = g3_partition["scenes"][item_id]
                if baseline["sample_key"] != candidate["sample_key"]:
                    raise ContractError(
                        f"retention comparison rank {rank} {partition} {item_id} sample_key changed"
                    )
                if baseline["target_identities"] != candidate["target_identities"]:
                    raise ContractError(
                        f"retention comparison rank {rank} {partition} {item_id} "
                        "prompt identity/order changed"
                    )
                baseline_bindings = baseline["binding_by_identity"]
                candidate_bindings = candidate["binding_by_identity"]
                if set(baseline_bindings) != set(candidate_bindings):
                    raise ContractError(
                        f"retention comparison rank {rank} {partition} {item_id} "
                        "physical identity set changed"
                    )
                row_permutation = {
                    int(baseline_bindings[identity]): int(candidate_bindings[identity])
                    for identity in sorted(baseline_bindings)
                }
                if len(row_permutation) != len(set(row_permutation.values())):
                    raise ContractError(
                        f"retention comparison rank {rank} {partition} {item_id} "
                        "gauge map is not bijective"
                    )
                for prompt_index, target_identity in enumerate(baseline["target_identities"]):
                    baseline_prompt = baseline["prompts"][prompt_index]
                    candidate_prompt = candidate["prompts"][prompt_index]
                    if row_permutation[int(baseline_prompt["target_row"])] != int(
                        candidate_prompt["target_row"]
                    ):
                        raise ContractError(
                            f"retention comparison rank {rank} {partition} {item_id} target row "
                            "does not follow the identity gauge permutation"
                        )
                    if row_permutation[int(baseline_prompt["alternate_row"])] != int(
                        candidate_prompt["alternate_row"]
                    ):
                        raise ContractError(
                            f"retention comparison rank {rank} {partition} {item_id} alternate row "
                            "does not follow the identity gauge permutation"
                        )
                    prompt_rows.append(
                        {
                            "alternate_identity": baseline["target_identities"][1 - prompt_index],
                            "g2": {
                                "margin": baseline_prompt["margin"],
                                "target_nll": baseline_prompt["target_nll"],
                                "target_row": baseline_prompt["target_row"],
                            },
                            "g3_retention": {
                                "margin": candidate_prompt["margin"],
                                "target_nll": candidate_prompt["target_nll"],
                                "target_row": candidate_prompt["target_row"],
                            },
                            "item_id": item_id,
                            "margin_delta_g3_minus_g2": (
                                candidate_prompt["margin"] - baseline_prompt["margin"]
                            ),
                            "partition": partition,
                            "physical_set_loss_note": (
                                "source schemas expose only the scene mean; see "
                                "scene_mean_physical_set_loss_delta_g3_minus_g2"
                            ),
                            "prompt_index": prompt_index,
                            "prompt_key": f"{item_id}/prompt-{prompt_index}:{target_identity}",
                            "rank": rank,
                            "sample_key": baseline["sample_key"],
                            "scene_mean_physical_set_loss_delta_g3_minus_g2": (
                                candidate["mean_physical_set_loss"]
                                - baseline["mean_physical_set_loss"]
                            ),
                            "target_identity": target_identity,
                            "target_nll_delta_g3_minus_g2": (
                                candidate_prompt["target_nll"] - baseline_prompt["target_nll"]
                            ),
                        }
                    )
                scene_rows.append(
                    {
                        "g2": {
                            "mean_margin": baseline["mean_margin"],
                            "mean_physical_set_loss": baseline["mean_physical_set_loss"],
                            "mean_target_nll": baseline["mean_target_nll"],
                            "positive_margin_count": baseline["positive_margin_count"],
                            "target_rows": baseline["target_rows"],
                        },
                        "g3_retention": {
                            "mean_margin": candidate["mean_margin"],
                            "mean_physical_set_loss": candidate["mean_physical_set_loss"],
                            "mean_target_nll": candidate["mean_target_nll"],
                            "positive_margin_count": candidate["positive_margin_count"],
                            "target_rows": candidate["target_rows"],
                        },
                        "gauge_permutation_g2_row_to_g3_row": {
                            str(row): row_permutation[row] for row in sorted(row_permutation)
                        },
                        "item_id": item_id,
                        "mean_margin_delta_g3_minus_g2": (
                            candidate["mean_margin"] - baseline["mean_margin"]
                        ),
                        "mean_physical_set_loss_delta_g3_minus_g2": (
                            candidate["mean_physical_set_loss"] - baseline["mean_physical_set_loss"]
                        ),
                        "mean_target_nll_delta_g3_minus_g2": (
                            candidate["mean_target_nll"] - baseline["mean_target_nll"]
                        ),
                        "rank": rank,
                        "raw_target_rows_equal": baseline["target_rows"]
                        == candidate["target_rows"],
                        "sample_key": baseline["sample_key"],
                        "target_identities": baseline["target_identities"],
                    }
                )

        g2_positive = sum(
            int(g2_ranks[rank][partition]["positive_margin_count"]) for rank in sorted(g2_ranks)
        )
        g3_positive = sum(
            int(g3_ranks[rank][partition]["positive_margin_count"]) for rank in sorted(g3_ranks)
        )
        required_minimum = g2_positive - maximum_regression
        count_gate = g3_positive >= required_minimum
        all_count_gates.append(count_gate)
        all_alignment_checks.append(True)
        partition_reports[partition] = {
            "aggregate": {
                "g2": {
                    "mean_margin": _mean(
                        [float(scene["g2"]["mean_margin"]) for scene in scene_rows],
                        name=f"{partition} G2 scene margins",
                    ),
                    "mean_physical_set_loss": _mean(
                        [float(scene["g2"]["mean_physical_set_loss"]) for scene in scene_rows],
                        name=f"{partition} G2 physical losses",
                    ),
                    "mean_target_nll": _mean(
                        [float(scene["g2"]["mean_target_nll"]) for scene in scene_rows],
                        name=f"{partition} G2 target NLLs",
                    ),
                },
                "g3_retention": {
                    "mean_margin": _mean(
                        [float(scene["g3_retention"]["mean_margin"]) for scene in scene_rows],
                        name=f"{partition} G3 scene margins",
                    ),
                    "mean_physical_set_loss": _mean(
                        [
                            float(scene["g3_retention"]["mean_physical_set_loss"])
                            for scene in scene_rows
                        ],
                        name=f"{partition} G3 physical losses",
                    ),
                    "mean_target_nll": _mean(
                        [float(scene["g3_retention"]["mean_target_nll"]) for scene in scene_rows],
                        name=f"{partition} G3 target NLLs",
                    ),
                },
                "mean_margin_delta_g3_minus_g2": _mean(
                    [float(scene["mean_margin_delta_g3_minus_g2"]) for scene in scene_rows],
                    name=f"{partition} scene margin deltas",
                ),
                "mean_physical_set_loss_delta_g3_minus_g2": _mean(
                    [
                        float(scene["mean_physical_set_loss_delta_g3_minus_g2"])
                        for scene in scene_rows
                    ],
                    name=f"{partition} scene physical loss deltas",
                ),
                "mean_target_nll_delta_g3_minus_g2": _mean(
                    [float(scene["mean_target_nll_delta_g3_minus_g2"]) for scene in scene_rows],
                    name=f"{partition} scene target NLL deltas",
                ),
            },
            "bootstrap": {
                "mean_margin_delta_g3_minus_g2": _bootstrap_ci(
                    [float(scene["mean_margin_delta_g3_minus_g2"]) for scene in scene_rows],
                    base_seed=bootstrap_seed,
                    bootstrap_samples=bootstrap_samples,
                    partition=partition,
                    metric="mean_margin_delta_g3_minus_g2",
                    direction="higher_is_better",
                ),
                "mean_physical_set_loss_delta_g3_minus_g2": _bootstrap_ci(
                    [
                        float(scene["mean_physical_set_loss_delta_g3_minus_g2"])
                        for scene in scene_rows
                    ],
                    base_seed=bootstrap_seed,
                    bootstrap_samples=bootstrap_samples,
                    partition=partition,
                    metric="mean_physical_set_loss_delta_g3_minus_g2",
                    direction="lower_is_better",
                ),
                "mean_target_nll_delta_g3_minus_g2": _bootstrap_ci(
                    [float(scene["mean_target_nll_delta_g3_minus_g2"]) for scene in scene_rows],
                    base_seed=bootstrap_seed,
                    bootstrap_samples=bootstrap_samples,
                    partition=partition,
                    metric="mean_target_nll_delta_g3_minus_g2",
                    direction="lower_is_better",
                ),
            },
            "positive_prompt_count_nonregression": {
                "g2_observed_positive_prompt_count": g2_positive,
                "g3_retention_observed_positive_prompt_count": g3_positive,
                "maximum_allowed_regression": maximum_regression,
                "required_g3_minimum": required_minimum,
                "satisfied": count_gate,
            },
            "prompt_count": len(prompt_rows),
            "prompts": prompt_rows,
            "scene_count": len(scene_rows),
            "scenes": scene_rows,
        }

    return {
        "alignment": {
            "all_checks_satisfied": all(all_alignment_checks),
            "gauge_policy": (
                "occupied rows are permutation gauges; exact identity bindings must induce "
                "a unique G2-to-G3 occupied-row bijection for every paired scene"
            ),
            "prompt_alignment": (
                "exact execution-contract digest plus rank/partition/item_id/sample_key/"
                "prompt-index/target-identity order"
            ),
            "prompt_text_in_scene_report": False,
        },
        "capacity": capacity,
        "comparison_status": "COMPLETE",
        "decision": {
            "default_positive_prompt_count_nonregression_satisfied": all(all_count_gates),
            "runner_status_is_not_scientific_conclusion": True,
            "scientific_conclusion": "NOT_AUTHORIZED_BY_COMPARISON_ALONE",
            "threshold_policy": (
                "use observed G2b positive prompt counts without fallback to the looser "
                "G3 runner thresholds; preserve all continuous paired quantities"
            ),
        },
        "input_reports": {
            "g2": {
                "schema": g2.get("schema"),
                "sha256": g2_report_sha256,
                "status": g2.get("status"),
            },
            "g3_retention": {
                "failures": g3.get("failures"),
                "runner_status": g3.get("status"),
                "schema": g3.get("schema"),
                "sha256": g3_retention_report_sha256,
            },
        },
        "parameters": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "maximum_positive_prompt_count_regression_per_partition": maximum_regression,
        },
        "partitions": partition_reports,
        "schema": OUTPUT_SCHEMA,
    }


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    g2, g2_digest = _load_report(args.g2_report, name="G2 report")
    g3, g3_digest = _load_report(args.g3_retention_report, name="G3 retention report")
    report = compare_lingbot_vla2_ltop_g3_retention_reports(
        g2,
        g3,
        g2_report_sha256=g2_digest,
        g3_retention_report_sha256=g3_digest,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        maximum_positive_prompt_count_regression_per_partition=(
            args.maximum_positive_prompt_count_regression_per_partition
        ),
    )
    write_text_durable_exclusive(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
