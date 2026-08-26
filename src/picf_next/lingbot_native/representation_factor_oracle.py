"""Offline factor attribution for immutable representation snapshots.

This module never runs the model.  It substitutes learned and loss-side
oracle values only for rows that the published snapshot proves were bound and
visible.  The resulting attribution is therefore conditional on row
materialization and must not be reported as an unconditional scene oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native.representation_evaluation import (
    REPRESENTATION_EVALUATION_PARTITIONS,
    RepresentationEvaluationPlan,
    validate_representation_evaluation_snapshot,
)

REPRESENTATION_FACTOR_ORACLE_SCHEMA = "picf-next.lingbot-representation-factor-oracle.v1"
REPRESENTATION_FACTOR_ORACLE_SCOPE = "visible_materialized_rows_only"

LEARNED_S_LEARNED_PI = "learned_s_learned_pi"
ORACLE_S_LEARNED_PI = "oracle_s_learned_pi"
LEARNED_S_ORACLE_PI = "learned_s_oracle_pi"
ORACLE_S_ORACLE_PI = "oracle_s_oracle_pi"
FACTOR_ORACLE_CORNERS = (
    LEARNED_S_LEARNED_PI,
    ORACLE_S_LEARNED_PI,
    LEARNED_S_ORACLE_PI,
    ORACLE_S_ORACLE_PI,
)
FACTOR_ORACLE_COHORTS = (
    "all_rows",
    "target_rows",
    "known_negative_rows",
    "area_lt_0p02",
    "area_0p02_to_lt_0p05",
    "area_ge_0p05",
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "status",
        "scope",
        "supports_unconditional_scene_attribution",
        "checkpoint_global_step",
        "partition",
        "snapshot_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "representation_split_sha256",
        "representation_evaluation_plan_sha256",
        "samples",
        "summary",
        "artifact_sha256",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _weighted_brier(
    prediction: Sequence[float],
    target: Sequence[float],
    weight: Sequence[float],
) -> float:
    if not prediction or not (len(prediction) == len(target) == len(weight)):
        raise ValueError("factor-oracle row vectors must be equal and nonempty")
    predictions = tuple(_finite(value, name="factor-oracle prediction") for value in prediction)
    targets = tuple(_finite(value, name="factor-oracle target") for value in target)
    weights = tuple(_finite(value, name="factor-oracle weight") for value in weight)
    if any(not 0.0 <= value <= 1.0 for value in (*predictions, *targets)):
        raise ValueError("factor-oracle probabilities and targets must lie in [0,1]")
    if any(value <= 0.0 for value in weights):
        raise ValueError("factor-oracle observed weights must be positive")
    normalizer = math.fsum(weights)
    return (
        math.fsum(
            observed_weight * (measured - expected) ** 2
            for measured, expected, observed_weight in zip(
                predictions,
                targets,
                weights,
                strict=True,
            )
        )
        / normalizer
    )


def _area_stratum(area_fraction: float) -> str:
    if area_fraction < 0.02:
        return "area_lt_0p02"
    if area_fraction < 0.05:
        return "area_0p02_to_lt_0p05"
    return "area_ge_0p05"


def _validate_row_binding(
    row: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> tuple[int, int]:
    row_index = int(row["row_index"])
    track_index = int(row["track_index"])
    row_to_track = diagnostic["row_to_track"]
    source_binding_valid = diagnostic["source_binding_valid"]
    identity_keys = diagnostic["identity_keys"]
    if not 0 <= row_index < len(row_to_track):
        raise ValueError("factor-oracle ownership row is outside the diagnostic capacity")
    if row_to_track[row_index] != track_index:
        raise ValueError("factor-oracle row/track binding differs from its diagnostic")
    if not source_binding_valid[row_index]:
        raise ValueError("factor-oracle ownership row lacks valid source binding")
    if identity_keys[track_index] != row["identity_key"]:
        raise ValueError("factor-oracle row identity differs from its assigned track")
    return row_index, track_index


def _row_value(
    row: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> dict[str, object]:
    row_index, track_index = _validate_row_binding(row, diagnostic)
    row_task_targets = diagnostic["row_task_targets"]
    row_task_valid = diagnostic["row_task_valid"]
    task_probabilities = diagnostic["task_probabilities"]
    if not row_task_valid[row_index]:
        raise ValueError("factor-oracle row lacks valid task supervision")

    learned_s = _finite(task_probabilities[row_index], name="factor-oracle learned s")
    oracle_s = _finite(row_task_targets[row_index], name="factor-oracle oracle s")
    if not 0.0 <= learned_s <= 1.0 or not 0.0 <= oracle_s <= 1.0:
        raise ValueError("factor-oracle task factors must lie in [0,1]")
    is_task_target = bool(row["is_task_target"])
    if is_task_target != (oracle_s > 0.0):
        raise ValueError("factor-oracle row target flag differs from task supervision")

    learned_pi = tuple(
        _finite(value, name="factor-oracle learned pi") for value in row["prediction"]
    )
    oracle_pi = tuple(_finite(value, name="factor-oracle oracle pi") for value in row["target"])
    weight = tuple(_finite(value, name="factor-oracle observed weight") for value in row["weight"])
    target = tuple(oracle_s * value for value in oracle_pi)
    predictions = {
        LEARNED_S_LEARNED_PI: tuple(learned_s * value for value in learned_pi),
        ORACLE_S_LEARNED_PI: tuple(oracle_s * value for value in learned_pi),
        LEARNED_S_ORACLE_PI: tuple(learned_s * value for value in oracle_pi),
        ORACLE_S_ORACLE_PI: target,
    }
    brier = {
        name: _weighted_brier(prediction, target, weight)
        for name, prediction in predictions.items()
    }
    if brier[ORACLE_S_ORACLE_PI] != 0.0:
        raise RuntimeError("factor-oracle double-oracle Brier must be exactly zero")
    semantic_contribution = 0.5 * (
        brier[LEARNED_S_LEARNED_PI]
        - brier[ORACLE_S_LEARNED_PI]
        + brier[LEARNED_S_ORACLE_PI]
        - brier[ORACLE_S_ORACLE_PI]
    )
    ownership_contribution = 0.5 * (
        brier[LEARNED_S_LEARNED_PI]
        - brier[LEARNED_S_ORACLE_PI]
        + brier[ORACLE_S_LEARNED_PI]
        - brier[ORACLE_S_ORACLE_PI]
    )
    total_excess = brier[LEARNED_S_LEARNED_PI] - brier[ORACLE_S_ORACLE_PI]
    if not math.isclose(
        semantic_contribution + ownership_contribution,
        total_excess,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("factor-oracle Shapley contributions do not close")

    weight_total = math.fsum(weight)
    area_fraction = (
        math.fsum(
            observed_weight * target_mass
            for observed_weight, target_mass in zip(weight, oracle_pi, strict=True)
        )
        / weight_total
    )
    return {
        "row_index": row_index,
        "track_index": track_index,
        "identity_key": row["identity_key"],
        "is_task_target": is_task_target,
        "learned_s": learned_s,
        "oracle_s": oracle_s,
        "target_area_fraction": area_fraction,
        "area_stratum": _area_stratum(area_fraction),
        "brier": brier,
        "semantic_shapley": semantic_contribution,
        "ownership_shapley": ownership_contribution,
        "total_excess_brier": total_excess,
    }


def _cohort_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    return {
        "row_count": len(rows),
        "mean_brier": {
            corner: _mean([float(row["brier"][corner]) for row in rows])
            for corner in FACTOR_ORACLE_CORNERS
        },
        "mean_semantic_shapley": _mean([float(row["semantic_shapley"]) for row in rows]),
        "mean_ownership_shapley": _mean([float(row["ownership_shapley"]) for row in rows]),
        "mean_total_excess_brier": _mean([float(row["total_excess_brier"]) for row in rows]),
    }


def _sample_value(sample: Mapping[str, Any]) -> dict[str, object]:
    diagnostic = sample["factual_task_row_diagnostic"]
    if set(diagnostic["target_identity_keys"]) != set(sample["factual_target_identity_keys"]):
        raise ValueError("factor-oracle diagnostic targets differ from the frozen evaluation item")
    visible_rows = tuple(sample["factual_ownership_rows"])
    eligible_rows = []
    task_invalid_visible_row_count = 0
    for row in visible_rows:
        row_index, _ = _validate_row_binding(row, diagnostic)
        if not diagnostic["row_task_valid"][row_index]:
            if row["is_task_target"]:
                raise ValueError("factor-oracle target row lacks valid task supervision")
            task_invalid_visible_row_count += 1
            continue
        eligible_rows.append(_row_value(row, diagnostic))
    rows = tuple(eligible_rows)
    if len({int(row["row_index"]) for row in rows}) != len(rows):
        raise ValueError("factor-oracle sample repeats an ownership row")

    target_identity_keys = tuple(diagnostic["target_identity_keys"])
    materialized_target_identity_keys = tuple(diagnostic["materialized_target_identity_keys"])
    unmaterialized_target_identity_keys = tuple(diagnostic["unmaterialized_target_identity_keys"])
    visible_target_identity_keys = tuple(
        sorted(str(row["identity_key"]) for row in rows if row["is_task_target"])
    )
    materialized_physical_row_count = sum(
        bool(value) for value in diagnostic["source_binding_valid"]
    )
    if materialized_physical_row_count <= 0 or len(visible_rows) > materialized_physical_row_count:
        raise ValueError("factor-oracle visible/materialized row coverage is inconsistent")
    visible_materialized_target_count = len(
        set(visible_target_identity_keys) & set(materialized_target_identity_keys)
    )

    cohorts = {
        "all_rows": rows,
        "target_rows": tuple(row for row in rows if row["is_task_target"]),
        "known_negative_rows": tuple(row for row in rows if not row["is_task_target"]),
        "area_lt_0p02": tuple(row for row in rows if row["area_stratum"] == "area_lt_0p02"),
        "area_0p02_to_lt_0p05": tuple(
            row for row in rows if row["area_stratum"] == "area_0p02_to_lt_0p05"
        ),
        "area_ge_0p05": tuple(row for row in rows if row["area_stratum"] == "area_ge_0p05"),
    }
    return {
        "sample_key": sample["sample_key"],
        "partition": sample["partition"],
        "task_key": sample["task_key"],
        "source_episode_index": int(sample["source_episode_index"]),
        "source_global_index": int(sample["source_global_index"]),
        "factor_eligible": bool(rows),
        "rows": list(rows),
        "cohorts": {
            cohort: _cohort_metrics(cohort_rows) for cohort, cohort_rows in cohorts.items()
        },
        "coverage": {
            "target_identity_count": len(target_identity_keys),
            "materialized_target_identity_count": len(materialized_target_identity_keys),
            "visible_materialized_target_identity_count": (visible_materialized_target_count),
            "unmaterialized_target_identity_count": len(unmaterialized_target_identity_keys),
            "capacity_censored_track_count": sum(
                bool(value) for value in diagnostic["capacity_censored"]
            ),
            "materialized_physical_row_count": materialized_physical_row_count,
            "visible_physical_row_count": len(visible_rows),
            "task_valid_visible_row_count": len(rows),
            "task_invalid_visible_row_count": task_invalid_visible_row_count,
            "materialized_target_coverage": (
                None
                if not target_identity_keys
                else len(materialized_target_identity_keys) / len(target_identity_keys)
            ),
            "visible_materialized_target_coverage": (
                None
                if not materialized_target_identity_keys
                else visible_materialized_target_count / len(materialized_target_identity_keys)
            ),
            "visible_physical_row_coverage": (len(visible_rows) / materialized_physical_row_count),
        },
    }


def _cluster_macro(
    samples: Sequence[Mapping[str, Any]],
    *,
    cohort: str,
    metric_path: tuple[str, ...],
) -> tuple[float | None, dict[str, object]]:
    by_task_episode: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        value: object = sample["cohorts"][cohort]
        for name in metric_path:
            if value is None or not isinstance(value, Mapping):
                value = None
                break
            value = value[name]
        if value is not None:
            by_task_episode[str(sample["task_key"])][int(sample["source_episode_index"])].append(
                _finite(value, name="factor-oracle cluster metric")
            )
    task_values: dict[str, object] = {}
    values: list[float] = []
    for task_key in sorted(by_task_episode):
        episode_values = {
            str(episode): math.fsum(measured) / len(measured)
            for episode, measured in sorted(by_task_episode[task_key].items())
        }
        task_mean = _mean(list(episode_values.values()))
        if task_mean is None:
            continue
        task_values[task_key] = {
            "episode_count": len(episode_values),
            "episode_values": episode_values,
            "mean": task_mean,
        }
        values.append(task_mean)
    return _mean(values), task_values


def _summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    task_macro: dict[str, object] = {}
    task_details: dict[str, object] = {}
    for cohort in FACTOR_ORACLE_COHORTS:
        cohort_metrics: dict[str, object] = {}
        cohort_details: dict[str, object] = {}
        for corner in FACTOR_ORACLE_CORNERS:
            mean_value, details = _cluster_macro(
                samples,
                cohort=cohort,
                metric_path=("mean_brier", corner),
            )
            cohort_metrics[corner] = mean_value
            cohort_details[corner] = details
        for output_name, source_name in (
            ("semantic_shapley", "mean_semantic_shapley"),
            ("ownership_shapley", "mean_ownership_shapley"),
            ("total_excess_brier", "mean_total_excess_brier"),
        ):
            mean_value, details = _cluster_macro(
                samples,
                cohort=cohort,
                metric_path=(source_name,),
            )
            cohort_metrics[output_name] = mean_value
            cohort_details[output_name] = details
        task_macro[cohort] = cohort_metrics
        task_details[cohort] = cohort_details

    target_total = sum(int(sample["coverage"]["target_identity_count"]) for sample in samples)
    materialized_target_total = sum(
        int(sample["coverage"]["materialized_target_identity_count"]) for sample in samples
    )
    visible_materialized_target_total = sum(
        int(sample["coverage"]["visible_materialized_target_identity_count"]) for sample in samples
    )
    materialized_physical_total = sum(
        int(sample["coverage"]["materialized_physical_row_count"]) for sample in samples
    )
    visible_physical_total = sum(
        int(sample["coverage"]["visible_physical_row_count"]) for sample in samples
    )
    return {
        "sample_count": len(samples),
        "factor_eligible_sample_count": sum(bool(sample["factor_eligible"]) for sample in samples),
        "factor_ineligible_sample_count": sum(
            not bool(sample["factor_eligible"]) for sample in samples
        ),
        "task_count": len({str(sample["task_key"]) for sample in samples}),
        "episode_cluster_count": len(
            {(str(sample["task_key"]), int(sample["source_episode_index"])) for sample in samples}
        ),
        "factor_eligible_row_count": sum(len(sample["rows"]) for sample in samples),
        "coverage": {
            "target_identity_count": target_total,
            "materialized_target_identity_count": materialized_target_total,
            "visible_materialized_target_identity_count": (visible_materialized_target_total),
            "unmaterialized_target_identity_count": sum(
                int(sample["coverage"]["unmaterialized_target_identity_count"])
                for sample in samples
            ),
            "capacity_censored_track_count": sum(
                int(sample["coverage"]["capacity_censored_track_count"]) for sample in samples
            ),
            "task_valid_visible_row_count": sum(
                int(sample["coverage"]["task_valid_visible_row_count"]) for sample in samples
            ),
            "task_invalid_visible_row_count": sum(
                int(sample["coverage"]["task_invalid_visible_row_count"]) for sample in samples
            ),
            "materialized_target_coverage": (
                None if target_total == 0 else materialized_target_total / target_total
            ),
            "visible_materialized_target_coverage": (
                None
                if materialized_target_total == 0
                else visible_materialized_target_total / materialized_target_total
            ),
            "visible_physical_row_coverage": (visible_physical_total / materialized_physical_total),
        },
        "task_episode_macro": task_macro,
        "task_episode_details": task_details,
    }


def _factor_oracle_value(
    snapshot: Mapping[str, Any],
    *,
    plan: RepresentationEvaluationPlan,
    partition: str,
) -> dict[str, object]:
    validated = validate_representation_evaluation_snapshot(dict(snapshot), plan=plan)
    if partition not in REPRESENTATION_EVALUATION_PARTITIONS:
        raise ValueError("factor-oracle partition is unsupported")
    selected = tuple(
        _sample_value(sample) for sample in validated["samples"] if sample["partition"] == partition
    )
    if not selected:
        raise ValueError("factor-oracle partition has no samples")
    value: dict[str, object] = {
        "schema": REPRESENTATION_FACTOR_ORACLE_SCHEMA,
        "status": "PASS",
        "scope": REPRESENTATION_FACTOR_ORACLE_SCOPE,
        "supports_unconditional_scene_attribution": False,
        "checkpoint_global_step": int(validated["checkpoint_global_step"]),
        "partition": partition,
        "snapshot_sha256": validated["artifact_sha256"],
        "implementation_sha256": validated["implementation_sha256"],
        "model_family_sha256": validated["model_family_sha256"],
        "representation_split_sha256": validated["representation_split_sha256"],
        "representation_evaluation_plan_sha256": plan.artifact_sha256,
        "samples": list(selected),
        "summary": _summary(selected),
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return value


def build_representation_factor_oracle(
    snapshot: Mapping[str, Any],
    *,
    plan: RepresentationEvaluationPlan,
    partition: str,
) -> dict[str, object]:
    """Build a conditional factor attribution from one published snapshot."""

    return validate_representation_factor_oracle(
        _factor_oracle_value(snapshot, plan=plan, partition=partition),
        snapshot=snapshot,
        plan=plan,
    )


def validate_representation_factor_oracle(
    value: object,
    *,
    snapshot: Mapping[str, Any],
    plan: RepresentationEvaluationPlan,
) -> dict[str, Any]:
    """Recompute every field and reject widened attribution claims."""

    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("representation factor-oracle fields differ from schema")
    if value["schema"] != REPRESENTATION_FACTOR_ORACLE_SCHEMA:
        raise ValueError("representation factor-oracle schema changed")
    if value["scope"] != REPRESENTATION_FACTOR_ORACLE_SCOPE:
        raise ValueError("representation factor-oracle scope changed")
    if value["supports_unconditional_scene_attribution"] is not False:
        raise ValueError("representation factor-oracle widened its attribution scope")
    expected = _factor_oracle_value(
        snapshot,
        plan=plan,
        partition=str(value["partition"]),
    )
    if value != expected:
        raise ValueError("representation factor-oracle was not recomputed")
    return value


def write_representation_factor_oracle(
    path: str | Path,
    value: Mapping[str, object],
) -> Path:
    """Publish one immutable factor-oracle artifact without replacement."""

    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    return write_bytes_durable_exclusive(path, payload.encode("ascii"))
