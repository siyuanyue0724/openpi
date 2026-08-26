"""Cluster-aware statistics for matched representation learning curves."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationPlan,
    validate_representation_evaluation_snapshot,
)
from picf_next.lingbot_native.representation_factor_oracle import (
    LEARNED_S_LEARNED_PI,
    validate_representation_factor_oracle,
)

REPRESENTATION_CLUSTER_THRESHOLDS_SCHEMA = "picf-next.lingbot-representation-cluster-thresholds.v1"
REPRESENTATION_CLUSTER_CURVE_SCHEMA = "picf-next.lingbot-representation-cluster-curve.v1"

HIGHER_IS_BETTER = "higher"
LOWER_IS_BETTER = "lower"
REPRESENTATION_CLUSTER_METRICS = {
    "token_auc": HIGHER_IS_BETTER,
    "token_target_background_margin": HIGHER_IS_BETTER,
    "row_rank_one": HIGHER_IS_BETTER,
    "row_hardest_negative_margin": HIGHER_IS_BETTER,
    "target_ownership_soft_iou": HIGHER_IS_BETTER,
    "official_action_loss": LOWER_IS_BETTER,
    "factor_target_joint_brier": LOWER_IS_BETTER,
    "factor_target_semantic_shapley": LOWER_IS_BETTER,
    "factor_target_ownership_shapley": LOWER_IS_BETTER,
    "factor_area_lt_0p02_joint_brier": LOWER_IS_BETTER,
    "factor_area_0p02_to_lt_0p05_joint_brier": LOWER_IS_BETTER,
    "factor_area_ge_0p05_joint_brier": LOWER_IS_BETTER,
}


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


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("cluster percentile inputs are invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cluster statistic cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _optional_metric(value: object, *, name: str) -> float | None:
    return None if value is None else _finite(value, name=name)


def _factor_samples_by_key(
    factor_oracle: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    samples = factor_oracle["samples"]
    if not isinstance(samples, list):
        raise ValueError("factor-oracle samples are malformed")
    result = {str(sample["sample_key"]): sample for sample in samples}
    if len(result) != len(samples):
        raise ValueError("factor-oracle repeats a sample key")
    return result


def extract_representation_cluster_records(
    snapshot: Mapping[str, Any],
    factor_oracle: Mapping[str, Any],
    *,
    plan: RepresentationEvaluationPlan,
    partition: str = "heldout",
) -> dict[str, dict[str, dict[str, object]]]:
    """Extract per-sample metrics without treating repeated frames as IID."""

    validated_snapshot = validate_representation_evaluation_snapshot(
        dict(snapshot),
        plan=plan,
    )
    validated_factor = validate_representation_factor_oracle(
        dict(factor_oracle),
        snapshot=validated_snapshot,
        plan=plan,
    )
    if validated_factor["partition"] != partition:
        raise ValueError("cluster records require the requested factor partition")
    factors = _factor_samples_by_key(validated_factor)
    snapshot_samples = tuple(
        sample for sample in validated_snapshot["samples"] if sample["partition"] == partition
    )
    if {str(sample["sample_key"]) for sample in snapshot_samples} != set(factors):
        raise ValueError("factor-oracle and snapshot sample coverage differ")

    records: dict[str, dict[str, dict[str, object]]] = {
        metric: {} for metric in REPRESENTATION_CLUSTER_METRICS
    }

    def add(metric: str, sample: Mapping[str, Any], value: float | None) -> None:
        if value is None:
            return
        sample_key = str(sample["sample_key"])
        records[metric][sample_key] = {
            "task_key": str(sample["task_key"]),
            "source_episode_index": int(sample["source_episode_index"]),
            "value": value,
        }

    for sample in snapshot_samples:
        token_metrics = sample["factual_token_evidence"]["metrics"]
        add(
            "token_auc",
            sample,
            _optional_metric(
                token_metrics["fractional_weighted_auc"],
                name="cluster token AUC",
            ),
        )
        add(
            "token_target_background_margin",
            sample,
            _optional_metric(
                token_metrics["target_background_logit_margin"],
                name="cluster token margin",
            ),
        )
        diagnostic = sample["factual_task_row_diagnostic"]
        worst_rank = diagnostic["worst_target_rank"]
        add(
            "row_rank_one",
            sample,
            None if worst_rank is None else float(int(worst_rank) == 1),
        )
        add(
            "row_hardest_negative_margin",
            sample,
            _optional_metric(
                diagnostic["target_vs_hardest_negative_logit_margin"],
                name="cluster row margin",
            ),
        )
        add(
            "target_ownership_soft_iou",
            sample,
            _optional_metric(
                sample["factual_ownership_summary"]["target_soft_iou"],
                name="cluster ownership soft-IoU",
            ),
        )
        add(
            "official_action_loss",
            sample,
            _finite(sample["official_action_loss"], name="cluster official action loss"),
        )

        factor = factors[str(sample["sample_key"])]
        target = factor["cohorts"]["target_rows"]
        add(
            "factor_target_joint_brier",
            sample,
            _optional_metric(
                target["mean_brier"][LEARNED_S_LEARNED_PI],
                name="cluster target joint Brier",
            ),
        )
        add(
            "factor_target_semantic_shapley",
            sample,
            _optional_metric(
                target["mean_semantic_shapley"],
                name="cluster target semantic Shapley",
            ),
        )
        add(
            "factor_target_ownership_shapley",
            sample,
            _optional_metric(
                target["mean_ownership_shapley"],
                name="cluster target ownership Shapley",
            ),
        )
        for cohort, metric in (
            ("area_lt_0p02", "factor_area_lt_0p02_joint_brier"),
            (
                "area_0p02_to_lt_0p05",
                "factor_area_0p02_to_lt_0p05_joint_brier",
            ),
            ("area_ge_0p05", "factor_area_ge_0p05_joint_brier"),
        ):
            add(
                metric,
                sample,
                _optional_metric(
                    factor["cohorts"][cohort]["mean_brier"][LEARNED_S_LEARNED_PI],
                    name=f"cluster {cohort} joint Brier",
                ),
            )
    return records


def _episode_values(
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records.values():
        grouped[str(record["task_key"])][int(record["source_episode_index"])].append(
            _finite(record["value"], name="cluster record value")
        )
    if not grouped:
        raise ValueError("cluster metric has no eligible records")
    return {
        task_key: {episode: _mean(values) for episode, values in sorted(episodes.items())}
        for task_key, episodes in sorted(grouped.items())
    }


def _hierarchical_mean(grouped: Mapping[str, Mapping[int, float]]) -> float:
    return _mean(
        [_mean([float(value) for value in episodes.values()]) for episodes in grouped.values()]
    )


def _hierarchical_interval(
    grouped: Mapping[str, Mapping[int, float]],
    *,
    replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    _positive_int(replicates, name="cluster bootstrap replicates")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("cluster confidence level must lie in (0,1)")
    tasks = sorted(grouped)
    rng = random.Random(bootstrap_seed)
    bootstrap: list[float] = []
    for _ in range(replicates):
        task_means = []
        for task_key in rng.choices(tasks, k=len(tasks)):
            episodes = grouped[task_key]
            episode_keys = sorted(episodes)
            task_means.append(
                _mean(
                    [
                        float(episodes[episode])
                        for episode in rng.choices(
                            episode_keys,
                            k=len(episode_keys),
                        )
                    ]
                )
            )
        bootstrap.append(_mean(task_means))
    tail = (1.0 - confidence_level) / 2.0
    return {
        "estimate": _hierarchical_mean(grouped),
        "lower": _percentile(bootstrap, tail),
        "upper": _percentile(bootstrap, 1.0 - tail),
    }


def _bindings(
    snapshot: Mapping[str, Any],
    factor: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "checkpoint_global_step": int(snapshot["checkpoint_global_step"]),
        "snapshot_sha256": snapshot["artifact_sha256"],
        "factor_oracle_sha256": factor["artifact_sha256"],
    }


def _validate_calibration_sources(
    sources: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    plan: RepresentationEvaluationPlan,
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, object]]]],
    dict[str, object],
]:
    if not sources:
        raise ValueError("cluster threshold calibration requires historical sources")
    records: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    bindings: dict[str, object] = {}
    for name in sorted(sources):
        if not isinstance(name, str) or not name:
            raise ValueError("cluster calibration source name is invalid")
        snapshot, factor = sources[name]
        records[name] = extract_representation_cluster_records(
            snapshot,
            factor,
            plan=plan,
        )
        bindings[name] = _bindings(snapshot, factor)
    return records, bindings


def _thresholds_value(
    sources: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    plan: RepresentationEvaluationPlan,
    replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, object]:
    records, bindings = _validate_calibration_sources(sources, plan=plan)
    thresholds: dict[str, float] = {}
    intervals: dict[str, object] = {}
    for metric in REPRESENTATION_CLUSTER_METRICS:
        metric_intervals: dict[str, object] = {}
        half_widths = []
        for source_name in sorted(records):
            metric_records = records[source_name][metric]
            if not metric_records:
                continue
            interval = _hierarchical_interval(
                _episode_values(metric_records),
                replicates=replicates,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
            metric_intervals[source_name] = interval
            half_widths.append(
                max(
                    interval["estimate"] - interval["lower"],
                    interval["upper"] - interval["estimate"],
                )
            )
        if not half_widths:
            raise ValueError(f"cluster calibration metric {metric} has no eligible source")
        thresholds[metric] = max(half_widths)
        intervals[metric] = metric_intervals
    value: dict[str, object] = {
        "schema": REPRESENTATION_CLUSTER_THRESHOLDS_SCHEMA,
        "status": "PASS",
        "calibration_scope": "historical_evaluation_bank_sampling_resolution_only",
        "is_run_to_run_variance_estimate": False,
        "representation_evaluation_plan_sha256": plan.artifact_sha256,
        "bootstrap": {
            "replicates": replicates,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "cluster_order": ["task_key", "source_episode_index", "sample_key"],
            "frames_treated_as_independent": False,
        },
        "metric_directions": dict(REPRESENTATION_CLUSTER_METRICS),
        "source_bindings": bindings,
        "calibration_intervals": intervals,
        "materiality_thresholds": thresholds,
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return value


def build_representation_cluster_thresholds(
    sources: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    plan: RepresentationEvaluationPlan,
    replicates: int = 20_000,
    bootstrap_seed: int = 135_202_608,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Freeze bank-resolution thresholds from historical evidence only."""

    return _thresholds_value(
        sources,
        plan=plan,
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


def validate_representation_cluster_thresholds(
    value: object,
    *,
    sources: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]] | None = None,
    plan: RepresentationEvaluationPlan | None = None,
) -> dict[str, Any]:
    """Validate content addressing, optionally by full historical recomputation."""

    if not isinstance(value, dict):
        raise ValueError("representation cluster thresholds must be a mapping")
    if value.get("schema") != REPRESENTATION_CLUSTER_THRESHOLDS_SCHEMA:
        raise ValueError("representation cluster threshold schema changed")
    artifact = value.get("artifact_sha256")
    if not isinstance(artifact, str) or len(artifact) != 64:
        raise ValueError("representation cluster threshold artifact digest is invalid")
    payload = {name: item for name, item in value.items() if name != "artifact_sha256"}
    if _canonical_sha256(payload) != artifact:
        raise ValueError("representation cluster threshold artifact changed")
    if value.get("is_run_to_run_variance_estimate") is not False:
        raise ValueError("cluster thresholds were mislabeled as run variance")
    if sources is not None or plan is not None:
        if sources is None or plan is None:
            raise ValueError("full threshold validation requires sources and plan")
        bootstrap = value["bootstrap"]
        expected = _thresholds_value(
            sources,
            plan=plan,
            replicates=int(bootstrap["replicates"]),
            bootstrap_seed=int(bootstrap["seed"]),
            confidence_level=float(bootstrap["confidence_level"]),
        )
        if value != expected:
            raise ValueError("representation cluster thresholds were not recomputed")
    return value


def _delta_records(
    current: Mapping[str, Mapping[str, object]],
    baseline: Mapping[str, Mapping[str, object]],
    *,
    direction: str,
) -> dict[str, dict[str, object]]:
    if set(current) != set(baseline):
        raise ValueError("matched curve metric eligibility changed across boundaries")
    sign = 1.0 if direction == HIGHER_IS_BETTER else -1.0
    result = {}
    for sample_key in sorted(current):
        left = current[sample_key]
        right = baseline[sample_key]
        if (
            left["task_key"] != right["task_key"]
            or left["source_episode_index"] != right["source_episode_index"]
        ):
            raise ValueError("matched curve sample cluster identity changed")
        result[sample_key] = {
            "task_key": left["task_key"],
            "source_episode_index": left["source_episode_index"],
            "value": sign
            * (
                _finite(left["value"], name="curve current metric")
                - _finite(right["value"], name="curve baseline metric")
            ),
        }
    return result


def _subtract_records(
    left: Mapping[str, Mapping[str, object]],
    right: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if set(left) != set(right):
        raise ValueError("matched curve arms have different metric eligibility")
    result = {}
    for sample_key in sorted(left):
        first = left[sample_key]
        second = right[sample_key]
        if (
            first["task_key"] != second["task_key"]
            or first["source_episode_index"] != second["source_episode_index"]
        ):
            raise ValueError("matched curve arm cluster identity changed")
        result[sample_key] = {
            "task_key": first["task_key"],
            "source_episode_index": first["source_episode_index"],
            "value": _finite(first["value"], name="curve left progress")
            - _finite(second["value"], name="curve right progress"),
        }
    return result


def _curve_value(
    arms: Mapping[
        str,
        Mapping[int, tuple[Mapping[str, Any], Mapping[str, Any]]],
    ],
    thresholds: Mapping[str, Any],
    *,
    plan: RepresentationEvaluationPlan,
) -> dict[str, object]:
    validated_thresholds = validate_representation_cluster_thresholds(dict(thresholds))
    if validated_thresholds["representation_evaluation_plan_sha256"] != plan.artifact_sha256:
        raise ValueError("matched curve thresholds target another evaluation plan")
    if set(arms) != {"M", "E"}:
        raise ValueError("matched curve requires exactly arms M and E")
    step_sets = {tuple(sorted(values)) for values in arms.values()}
    if len(step_sets) != 1:
        raise ValueError("matched curve arms have different checkpoints")
    steps = next(iter(step_sets))
    if not steps or steps[0] != 0 or len(steps) < 2:
        raise ValueError("matched curve requires baseline and learned checkpoints")
    bootstrap = validated_thresholds["bootstrap"]
    replicates = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])
    confidence_level = float(bootstrap["confidence_level"])

    records: dict[str, dict[int, dict[str, dict[str, dict[str, object]]]]] = {}
    bindings: dict[str, object] = {}
    for arm in ("M", "E"):
        records[arm] = {}
        bindings[arm] = {}
        for step in steps:
            snapshot, factor = arms[arm][step]
            if int(snapshot["checkpoint_global_step"]) != step:
                raise ValueError("matched curve checkpoint label differs from snapshot")
            records[arm][step] = extract_representation_cluster_records(
                snapshot,
                factor,
                plan=plan,
            )
            bindings[arm][str(step)] = _bindings(snapshot, factor)
    for arm in ("M", "E"):
        split_bindings = {arms[arm][step][0]["representation_split_sha256"] for step in steps}
        if len(split_bindings) != 1:
            raise ValueError("matched curve changed split within an arm")
    if (
        arms["M"][0][0]["representation_split_sha256"]
        != arms["E"][0][0]["representation_split_sha256"]
    ):
        raise ValueError("matched curve arms use different source splits")
    for sample_m, sample_e in zip(
        arms["M"][0][0]["samples"],
        arms["E"][0][0]["samples"],
        strict=True,
    ):
        if sample_m["sample_key"] != sample_e["sample_key"]:
            raise ValueError("matched curve step-zero sample order changed")
        if sample_m["tensor_sha256"] != sample_e["tensor_sha256"]:
            raise ValueError("matched curve step-zero tensors differ between arms")

    metric_results: dict[str, object] = {}
    thresholds_by_metric = validated_thresholds["materiality_thresholds"]
    for metric, direction in REPRESENTATION_CLUSTER_METRICS.items():
        threshold = _finite(
            thresholds_by_metric[metric],
            name="curve materiality threshold",
        )
        progress: dict[str, dict[int, dict[str, dict[str, object]]]] = {
            "M": {},
            "E": {},
        }
        arm_intervals: dict[str, object] = {"M": {}, "E": {}}
        did_intervals: dict[str, object] = {}
        for arm in ("M", "E"):
            baseline = records[arm][0][metric]
            for step in steps[1:]:
                delta = _delta_records(
                    records[arm][step][metric],
                    baseline,
                    direction=direction,
                )
                progress[arm][step] = delta
                interval = _hierarchical_interval(
                    _episode_values(delta),
                    replicates=replicates,
                    bootstrap_seed=seed,
                    confidence_level=confidence_level,
                )
                arm_intervals[arm][str(step)] = {
                    **interval,
                    "material_progress": (
                        interval["estimate"] > threshold and interval["lower"] > 0.0
                    ),
                }
        for step in steps[1:]:
            did = _subtract_records(progress["E"][step], progress["M"][step])
            interval = _hierarchical_interval(
                _episode_values(did),
                replicates=replicates,
                bootstrap_seed=seed,
                confidence_level=confidence_level,
            )
            did_intervals[str(step)] = {
                **interval,
                "material_E_advantage": (
                    interval["estimate"] > threshold and interval["lower"] > 0.0
                ),
            }
        metric_results[metric] = {
            "direction": direction,
            "materiality_threshold": threshold,
            "arm_progress": arm_intervals,
            "E_minus_M_progress": did_intervals,
        }
    value: dict[str, object] = {
        "schema": REPRESENTATION_CLUSTER_CURVE_SCHEMA,
        "status": "SINGLE_STREAM_DIAGNOSTIC",
        "authorizes_model_adoption": False,
        "representation_evaluation_plan_sha256": plan.artifact_sha256,
        "thresholds_sha256": validated_thresholds["artifact_sha256"],
        "steps": list(steps),
        "arm_bindings": bindings,
        "metrics": metric_results,
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return value


def build_representation_cluster_curve(
    arms: Mapping[
        str,
        Mapping[int, tuple[Mapping[str, Any], Mapping[str, Any]]],
    ],
    thresholds: Mapping[str, Any],
    *,
    plan: RepresentationEvaluationPlan,
) -> dict[str, object]:
    """Build one matched single-stream diagnostic learning curve."""

    return _curve_value(arms, thresholds, plan=plan)


def validate_representation_cluster_curve(
    value: object,
    *,
    arms: Mapping[
        str,
        Mapping[int, tuple[Mapping[str, Any], Mapping[str, Any]]],
    ],
    thresholds: Mapping[str, Any],
    plan: RepresentationEvaluationPlan,
) -> dict[str, Any]:
    """Recompute a complete matched curve from immutable boundary artifacts."""

    if not isinstance(value, dict) or value.get("schema") != REPRESENTATION_CLUSTER_CURVE_SCHEMA:
        raise ValueError("representation cluster curve schema changed")
    expected = _curve_value(arms, thresholds, plan=plan)
    if value != expected:
        raise ValueError("representation cluster curve was not recomputed")
    return value


def write_representation_cluster_artifact(
    path: str | Path,
    value: Mapping[str, object],
) -> Path:
    """Publish one immutable clustered-statistics artifact."""

    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    return write_bytes_durable_exclusive(path, payload.encode("ascii"))
