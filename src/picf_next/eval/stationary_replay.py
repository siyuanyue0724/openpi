"""Frozen decision contract for stationary-posterior checkpoint replay."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

STATIONARY_FIXED_REPLAY_SCHEMA: Final = "picf-next.stationary-fixed-checkpoint-replay.v2"
STATIONARY_FIXED_REPLAY_PASS: Final = "PASS"
STATIONARY_FIXED_REPLAY_FAIL: Final = "FAIL"

_MODEL_NAMES: Final = ("fresh_m2", "candidate")
_SPLIT_NAMES: Final = ("validation", "heldout")
_LOWER_IS_BETTER: Final = (
    "loss_total",
    "loss_set",
    "loss_dynamics",
    "loss_dynamics_survival",
    "loss_dynamics_visibility",
    "loss_binding",
    "assignment_conflicts_per_clip",
)
_HIGHER_IS_BETTER: Final = (
    "discovery_soft_iou",
    "posterior_soft_iou",
    "posterior_identity_coverage",
)
STATIONARY_FIXED_REPLAY_METRICS: Final = _LOWER_IS_BETTER + _HIGHER_IS_BETTER
_BINDING_SHA256_FIELDS: Final = (
    "candidate_checkpoint_sha256",
    "candidate_report_sha256",
    "dataset_manifest_sha256",
    "feature_cache_manifest_sha256",
    "foundation_recipe_sha256",
    "m2_checkpoint_sha256",
    "m2_report_sha256",
    "physical_sidecar_manifest_sha256",
    "source_coverage_recipe_sha256",
    "stage_recipe_sha256",
)
_BINDING_FIELDS: Final = set(_BINDING_SHA256_FIELDS) | {
    "audit_code_revision",
    "candidate_code_revision",
}
_CLIP_FIELDS: Final = {
    "optimizer_step",
    "source_range_index",
    "start_global_index",
    "prefix_length",
    "train_length",
    "train_start_global_index",
    "stop_global_index",
}


def _exact_mapping(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields differ from its frozen schema")
    return cast(dict[str, Any], value)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a finite number")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    return output


def _hex_digest(value: object, name: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase hexadecimal digest")
    return value


def validate_replay_metric_summary(value: object, name: str) -> dict[str, float]:
    """Validate one exact aggregate used by the preregistered comparison."""

    payload = _exact_mapping(value, name, set(STATIONARY_FIXED_REPLAY_METRICS))
    result = {
        metric: _finite_number(payload[metric], f"{name}.{metric}")
        for metric in STATIONARY_FIXED_REPLAY_METRICS
    }
    for metric in (
        "assignment_conflicts_per_clip",
        "loss_dynamics_survival",
        "loss_dynamics_visibility",
        "discovery_soft_iou",
        "posterior_soft_iou",
        "posterior_identity_coverage",
    ):
        if result[metric] < 0.0:
            raise ValueError(f"{name}.{metric} cannot be negative")
    for metric in (
        "discovery_soft_iou",
        "posterior_soft_iou",
        "posterior_identity_coverage",
    ):
        if result[metric] > 1.0 + 1e-6:
            raise ValueError(f"{name}.{metric} must lie in [0, 1]")
    return result


def compare_stationary_replay_summaries(
    *,
    fresh_m2: Mapping[str, float],
    candidate: Mapping[str, float],
    absolute_tolerance: float,
) -> dict[str, bool]:
    """Apply the frozen non-inferiority direction to one fixed split.

    All structural losses and assignment conflicts must not increase. Matched
    current-object overlap, persistent-object overlap and identity coverage
    must not decrease. No metric is selected after observing the candidate.
    """

    baseline = validate_replay_metric_summary(dict(fresh_m2), "fresh_m2 summary")
    treatment = validate_replay_metric_summary(dict(candidate), "candidate summary")
    tolerance = _finite_number(absolute_tolerance, "absolute_tolerance")
    if tolerance < 0.0:
        raise ValueError("absolute_tolerance cannot be negative")
    checks = {
        f"candidate_{metric}_not_worse": treatment[metric] <= baseline[metric] + tolerance
        for metric in _LOWER_IS_BETTER
    }
    checks.update(
        {
            f"candidate_{metric}_not_worse": treatment[metric] >= baseline[metric] - tolerance
            for metric in _HIGHER_IS_BETTER
        }
    )
    return checks


def validate_stationary_fixed_replay(payload: object) -> dict[str, Any]:
    """Validate machine replay evidence before it can enter acceptance."""

    report = _exact_mapping(
        payload,
        "stationary fixed replay",
        {
            "schema",
            "status",
            "protocol",
            "bindings",
            "plans",
            "thresholds",
            "splits",
            "checks",
            "failed_checks",
            "measurements",
            "long_training_authorized",
        },
    )
    if report["schema"] != STATIONARY_FIXED_REPLAY_SCHEMA:
        raise ValueError("stationary fixed replay schema changed")
    if report["status"] not in {STATIONARY_FIXED_REPLAY_PASS, STATIONARY_FIXED_REPLAY_FAIL}:
        raise ValueError("stationary fixed replay status is invalid")
    if report["long_training_authorized"] is not False:
        raise ValueError("fixed replay cannot authorize long training")

    protocol = _exact_mapping(
        report["protocol"],
        "stationary fixed replay protocol",
        {
            "comparison",
            "observation_inputs",
            "target_use",
            "split_names",
            "prefix_lengths",
            "train_length",
            "world_size",
            "optimizer_steps_per_split",
            "seed",
        },
    )
    if (
        protocol["comparison"] != "same-frozen-clips-fresh-m2-vs-stage-b-candidate.v1"
        or protocol["observation_inputs"] != "task-independent-cached-native-token-bank"
        or protocol["target_use"] != "post-forward-loss-and-evaluation-only"
        or protocol["split_names"] != list(_SPLIT_NAMES)
        or protocol["prefix_lengths"] != [0, 8, 32, 128]
        or protocol["train_length"] != 2
        or protocol["world_size"] != 2
        or not isinstance(protocol["optimizer_steps_per_split"], int)
        or isinstance(protocol["optimizer_steps_per_split"], bool)
        or protocol["optimizer_steps_per_split"] < 4
        or protocol["optimizer_steps_per_split"] % 4
        or not isinstance(protocol["seed"], int)
        or isinstance(protocol["seed"], bool)
        or protocol["seed"] < 0
    ):
        raise ValueError("stationary fixed replay protocol changed")

    thresholds = _exact_mapping(
        report["thresholds"],
        "stationary fixed replay thresholds",
        {"absolute_tolerance", "lower_is_better", "higher_is_better"},
    )
    tolerance = _finite_number(thresholds["absolute_tolerance"], "absolute_tolerance")
    if (
        tolerance < 0.0
        or thresholds["lower_is_better"] != list(_LOWER_IS_BETTER)
        or thresholds["higher_is_better"] != list(_HIGHER_IS_BETTER)
    ):
        raise ValueError("stationary fixed replay thresholds changed")

    splits = _exact_mapping(report["splits"], "stationary fixed replay splits", set(_SPLIT_NAMES))
    expected_checks: dict[str, bool] = {}
    for split_name in _SPLIT_NAMES:
        split = _exact_mapping(
            splits[split_name],
            f"stationary fixed replay {split_name}",
            {"clip_count", "models", "comparisons"},
        )
        expected_clip_count = protocol["optimizer_steps_per_split"] * protocol["world_size"]
        if split["clip_count"] != expected_clip_count:
            raise ValueError(f"{split_name} clip count changed")
        models = _exact_mapping(
            split["models"],
            f"stationary fixed replay {split_name} models",
            set(_MODEL_NAMES),
        )
        summaries = {
            name: validate_replay_metric_summary(
                models[name], f"stationary fixed replay {split_name}.{name}"
            )
            for name in _MODEL_NAMES
        }
        comparisons = compare_stationary_replay_summaries(
            fresh_m2=summaries["fresh_m2"],
            candidate=summaries["candidate"],
            absolute_tolerance=tolerance,
        )
        if split["comparisons"] != comparisons:
            raise ValueError(f"{split_name} replay comparisons were not recomputed exactly")
        expected_checks.update(
            {f"{split_name}_{name}": passed for name, passed in comparisons.items()}
        )

    plans = _exact_mapping(report["plans"], "stationary fixed replay plans", set(_SPLIT_NAMES))
    for split_name, plan in plans.items():
        plan_payload = _exact_mapping(
            plan,
            f"stationary fixed replay {split_name} plan",
            {"plan_sha256", "source_ranges"},
        )
        digest = plan_payload["plan_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(plan_payload["source_ranges"], list)
            or not plan_payload["source_ranges"]
        ):
            raise ValueError(f"stationary fixed replay {split_name} plan is malformed")

    bindings = _exact_mapping(
        report["bindings"],
        "stationary fixed replay bindings",
        _BINDING_FIELDS,
    )
    for name in _BINDING_SHA256_FIELDS:
        _hex_digest(bindings[name], f"stationary fixed replay binding {name}")
    _hex_digest(bindings["audit_code_revision"], "audit code revision", length=40)
    _hex_digest(bindings["candidate_code_revision"], "candidate code revision", length=40)

    measurements = report["measurements"]
    expected_measurement_count = (
        len(_SPLIT_NAMES)
        * len(_MODEL_NAMES)
        * protocol["optimizer_steps_per_split"]
        * protocol["world_size"]
    )
    if not isinstance(measurements, list) or len(measurements) != expected_measurement_count:
        raise ValueError("stationary fixed replay measurement coverage changed")
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {
        (split_name, model_name): [] for split_name in _SPLIT_NAMES for model_name in _MODEL_NAMES
    }
    clip_by_coordinate: dict[tuple[str, int, int], dict[str, Any]] = {}
    observed_coordinates: set[tuple[str, str, int, int]] = set()
    prefix_counts = {prefix: 0 for prefix in protocol["prefix_lengths"]}
    for index, raw_measurement in enumerate(measurements):
        measurement = _exact_mapping(
            raw_measurement,
            f"stationary fixed replay measurement {index}",
            {"clip", "metrics", "model", "optimizer_step", "rank", "split"},
        )
        split_name = measurement["split"]
        model_name = measurement["model"]
        optimizer_step = measurement["optimizer_step"]
        rank = measurement["rank"]
        if split_name not in _SPLIT_NAMES or model_name not in _MODEL_NAMES:
            raise ValueError("stationary fixed replay measurement identity changed")
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or not 0 <= optimizer_step < protocol["optimizer_steps_per_split"]
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < protocol["world_size"]
        ):
            raise ValueError("stationary fixed replay measurement coordinate is invalid")
        coordinate = (split_name, model_name, optimizer_step, rank)
        if coordinate in observed_coordinates:
            raise ValueError("stationary fixed replay measurement coordinate is duplicated")
        observed_coordinates.add(coordinate)
        clip = _exact_mapping(
            measurement["clip"],
            f"stationary fixed replay measurement {index} clip",
            _CLIP_FIELDS,
        )
        integer_fields = tuple(clip.values())
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_fields):
            raise ValueError("stationary fixed replay clip contains non-integer coordinates")
        if (
            clip["optimizer_step"] != optimizer_step
            or clip["prefix_length"] not in protocol["prefix_lengths"]
            or clip["train_length"] != protocol["train_length"]
            or clip["train_start_global_index"]
            != clip["start_global_index"] + clip["prefix_length"]
            or clip["stop_global_index"] != clip["train_start_global_index"] + clip["train_length"]
        ):
            raise ValueError("stationary fixed replay clip geometry changed")
        clip_coordinate = (split_name, optimizer_step, rank)
        frozen_clip = clip_by_coordinate.setdefault(clip_coordinate, clip)
        if frozen_clip != clip:
            raise ValueError("fresh and candidate replay used different clips")
        prefix_counts[clip["prefix_length"]] += 1
        grouped[(split_name, model_name)].append(
            validate_replay_metric_summary(
                measurement["metrics"],
                f"stationary fixed replay measurement {index} metrics",
            )
        )
    expected_prefix_count = (
        len(_SPLIT_NAMES)
        * len(_MODEL_NAMES)
        * protocol["world_size"]
        * protocol["optimizer_steps_per_split"]
        // len(protocol["prefix_lengths"])
    )
    if any(count != expected_prefix_count for count in prefix_counts.values()):
        raise ValueError("stationary fixed replay prefix coverage changed")
    for split_name in _SPLIT_NAMES:
        for model_name in _MODEL_NAMES:
            aggregate = aggregate_replay_measurements(grouped[(split_name, model_name)])
            published = splits[split_name]["models"][model_name]
            if any(
                not math.isclose(
                    aggregate[metric],
                    published[metric],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for metric in STATIONARY_FIXED_REPLAY_METRICS
            ):
                raise ValueError("stationary fixed replay aggregate differs from its rows")
    checks = _exact_mapping(
        report["checks"],
        "stationary fixed replay checks",
        set(expected_checks),
    )
    if checks != expected_checks:
        raise ValueError("stationary fixed replay top-level checks changed")
    failed = sorted(name for name, passed in expected_checks.items() if not passed)
    if report["failed_checks"] != failed:
        raise ValueError("stationary fixed replay failed checks changed")
    expected_status = STATIONARY_FIXED_REPLAY_PASS if not failed else STATIONARY_FIXED_REPLAY_FAIL
    if report["status"] != expected_status:
        raise ValueError("stationary fixed replay status differs from its measurements")
    return report


def aggregate_replay_measurements(
    measurements: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate exact per-clip replay rows without hidden weighting."""

    rows = tuple(measurements)
    if not rows:
        raise ValueError("stationary replay cannot aggregate an empty measurement set")
    validated = [validate_replay_metric_summary(dict(row), "stationary replay row") for row in rows]
    return {
        metric: sum(row[metric] for row in validated) / len(validated)
        for metric in STATIONARY_FIXED_REPLAY_METRICS
    }
