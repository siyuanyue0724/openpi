"""Preregistered numeric gate for the fixed-observation identifiability trial.

This is an offline evaluator, not a model component. It recomputes paired
prompt-switch evidence from immutable update-zero and update-200 snapshots.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native.fixed_observation_evaluation import (
    FIXED_OBSERVATION_EVALUATION_PARTITIONS,
    FIXED_OBSERVATION_MASS_STRATA,
    FixedObservationEvaluationItem,
    FixedObservationEvaluationPlan,
    validate_fixed_observation_evaluation_snapshot,
)

FIXED_OBSERVATION_NUMERIC_GATE_SCHEMA = "picf-next.lingbot-fixed-observation-numeric-gate.v2"
FIXED_OBSERVATION_BASELINE_STEP = 0
FIXED_OBSERVATION_DECISION_STEP = 200
FIXED_OBSERVATION_MAXIMUM_SIGN_TEST_PVALUE = 0.05
FIXED_OBSERVATION_MINIMUM_BREADTH_FRACTION = 2.0 / 3.0
FIXED_OBSERVATION_MINIMUM_BIDIRECTIONAL_FRACTION = 0.5

_ADVANTAGE_METRICS = {
    "dense": "dense_mean_diagonal_advantage",
    "fractional_auc": "fractional_auc_mean_diagonal_advantage",
    "row": "row_mean_diagonal_advantage",
}
_BIDIRECTIONAL_METRICS = {
    "fractional_auc": "fractional_auc_bidirectional_positive",
    "row": "row_bidirectional_positive",
}
_GATE_FIELDS = frozenset(
    {
        "artifact_sha256",
        "authorizes_action_or_long_training",
        "baseline_snapshot_sha256",
        "decision_snapshot_sha256",
        "fixed_observation_evaluation_plan_sha256",
        "numeric_gate_implementation_sha256",
        "invariant_checks",
        "partitions",
        "schema",
        "status",
        "thresholds",
        "visual_review_required",
    }
)


class _SignTestResult(TypedDict):
    negative_count: int
    nonzero_count: int
    one_sided_pvalue: float
    positive_count: int
    zero_count: int


class _PositiveBreadth(TypedDict):
    positive_task_count: int
    positive_target_count: int
    task_count: int
    task_mean_row_advantages: dict[str, float]
    target_count: int
    target_mean_row_advantages: dict[str, float]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("fixed-X gate value is not canonical finite JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _numeric_gate_implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _check(
    *,
    passed: bool,
    actual: object,
    relation: str,
    threshold: object,
) -> dict[str, object]:
    return {
        "actual": actual,
        "passed": bool(passed),
        "relation": relation,
        "threshold": threshold,
    }


def _exact_one_sided_sign_test_pvalue(values: Sequence[float]) -> _SignTestResult:
    """Return P[Binomial(n, 0.5) >= positive] after excluding exact ties."""

    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    zero = len(values) - positive - negative
    trials = positive + negative
    pvalue = (
        1.0
        if trials == 0
        else math.fsum(math.comb(trials, count) for count in range(positive, trials + 1))
        / (2**trials)
    )
    return {
        "negative_count": negative,
        "nonzero_count": trials,
        "one_sided_pvalue": pvalue,
        "positive_count": positive,
        "zero_count": zero,
    }


def _partition_samples(
    snapshot: Mapping[str, Any],
    *,
    partition: str,
) -> tuple[dict[str, Any], ...]:
    samples = snapshot["samples"]
    if not isinstance(samples, list):
        raise ValueError("fixed-X gate snapshot samples are malformed")
    selected = tuple(
        sample
        for sample in samples
        if isinstance(sample, dict) and sample["item"]["partition"] == partition
    )
    if not selected:
        raise ValueError("fixed-X gate partition has no samples")
    return selected


def _advantage_values(
    samples: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> tuple[float, ...]:
    field = _ADVANTAGE_METRICS[metric]
    values = []
    for sample in samples:
        pair_metrics = sample["pair_metrics"]
        if not isinstance(pair_metrics, Mapping):
            raise ValueError("fixed-X gate pair metrics are malformed")
        value = pair_metrics[field]
        if value is None:
            raise ValueError(f"fixed-X gate {metric} advantage lost eligibility")
        values.append(_finite(value, name=f"fixed-X {metric} advantage"))
    return tuple(values)


def _source_episode_index(sample: Mapping[str, Any]) -> int:
    item = FixedObservationEvaluationItem.from_dict(sample["item"])
    return item.group.source_episode_index


def _episode_clustered_values(
    samples: Sequence[Mapping[str, Any]],
    values: Sequence[float],
) -> tuple[float, ...]:
    """Average frame-level values before inference to avoid pseudoreplication."""

    if len(samples) != len(values):
        raise ValueError("fixed-X episode clustering lost sample alignment")
    grouped: dict[int, list[float]] = defaultdict(list)
    for sample, value in zip(samples, values, strict=True):
        grouped[_source_episode_index(sample)].append(value)
    if not grouped:
        raise ValueError("fixed-X episode clustering has no source episodes")
    return tuple(_mean(grouped[index]) for index in sorted(grouped))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("fixed-X gate cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _bidirectional_fraction(
    samples: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> float:
    field = _BIDIRECTIONAL_METRICS[metric]
    values = []
    for sample in samples:
        value = sample["pair_metrics"][field]
        if not isinstance(value, bool):
            raise ValueError(f"fixed-X gate {metric} bidirectional result lost eligibility")
        values.append(value)
    return _mean(
        _episode_clustered_values(
            samples,
            tuple(float(value) for value in values),
        )
    )


def _positive_breadth(
    samples: Sequence[Mapping[str, Any]],
) -> _PositiveBreadth:
    by_task: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_target: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        item = FixedObservationEvaluationItem.from_dict(sample["item"])
        episode = item.group.source_episode_index
        raw_values = sample["pair_metrics"]["row_variant_diagonal_advantages"]
        if not isinstance(raw_values, list) or len(raw_values) != len(item.variants):
            raise ValueError("fixed-X row breadth lost per-variant advantages")
        row_values = tuple(
            _finite(value, name="fixed-X per-variant row breadth advantage") for value in raw_values
        )
        for variant, row_value in zip(item.variants, row_values, strict=True):
            by_task[variant.task_key][episode].append(row_value)
            by_target[variant.target_identity_key][episode].append(row_value)

    def equal_episode_mean(values: Mapping[int, Sequence[float]]) -> float:
        return _mean(tuple(_mean(values[index]) for index in sorted(values)))

    task_means = {name: equal_episode_mean(values) for name, values in sorted(by_task.items())}
    target_means = {name: equal_episode_mean(values) for name, values in sorted(by_target.items())}
    return {
        "positive_task_count": sum(value > 0.0 for value in task_means.values()),
        "positive_target_count": sum(value > 0.0 for value in target_means.values()),
        "task_count": len(task_means),
        "task_mean_row_advantages": task_means,
        "target_count": len(target_means),
        "target_mean_row_advantages": target_means,
    }


def _mass_stratum_means(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["mass_stratum"])].append(sample)
    if set(grouped) != set(FIXED_OBSERVATION_MASS_STRATA):
        raise ValueError("fixed-X gate lost one preregistered target-mass stratum")
    return {
        stratum: {
            metric: _mean(
                _episode_clustered_values(
                    grouped[stratum],
                    _advantage_values(grouped[stratum], metric=metric),
                )
            )
            for metric in ("fractional_auc", "row")
        }
        for stratum in FIXED_OBSERVATION_MASS_STRATA
    }


def _partition_gate(
    baseline_samples: Sequence[Mapping[str, Any]],
    decision_samples: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    if len(baseline_samples) != len(decision_samples):
        raise ValueError("fixed-X gate sample count changed between checkpoints")
    if tuple(sample["item"] for sample in baseline_samples) != tuple(
        sample["item"] for sample in decision_samples
    ):
        raise ValueError("fixed-X gate sample identities changed between checkpoints")
    if any(
        baseline["model_input_sha256"] != decision["model_input_sha256"]
        for baseline, decision in zip(
            baseline_samples,
            decision_samples,
            strict=True,
        )
    ):
        raise ValueError("fixed-X gate model inputs changed between checkpoints")
    if any(
        tuple(result["target_sha256"] for result in baseline["variant_results"])
        != tuple(result["target_sha256"] for result in decision["variant_results"])
        for baseline, decision in zip(
            baseline_samples,
            decision_samples,
            strict=True,
        )
    ):
        raise ValueError("fixed-X gate supervision targets changed between checkpoints")

    baseline_values = {
        metric: _advantage_values(baseline_samples, metric=metric) for metric in _ADVANTAGE_METRICS
    }
    decision_values = {
        metric: _advantage_values(decision_samples, metric=metric) for metric in _ADVANTAGE_METRICS
    }
    baseline_episode_values = {
        metric: _episode_clustered_values(baseline_samples, values)
        for metric, values in baseline_values.items()
    }
    decision_episode_values = {
        metric: _episode_clustered_values(decision_samples, values)
        for metric, values in decision_values.items()
    }
    baseline_means = {metric: _mean(values) for metric, values in baseline_episode_values.items()}
    decision_means = {metric: _mean(values) for metric, values in decision_episode_values.items()}
    improvement_episode_values = {
        metric: _episode_clustered_values(
            decision_samples,
            tuple(
                decision - baseline
                for decision, baseline in zip(
                    decision_values[metric],
                    baseline_values[metric],
                    strict=True,
                )
            ),
        )
        for metric in _ADVANTAGE_METRICS
    }
    decision_sign_tests = {
        metric: _exact_one_sided_sign_test_pvalue(decision_episode_values[metric])
        for metric in ("fractional_auc", "row")
    }
    improvement_sign_tests = {
        metric: _exact_one_sided_sign_test_pvalue(improvement_episode_values[metric])
        for metric in ("fractional_auc", "row")
    }
    bidirectional = {
        metric: _bidirectional_fraction(decision_samples, metric=metric)
        for metric in _BIDIRECTIONAL_METRICS
    }
    breadth = _positive_breadth(decision_samples)
    mass_strata = _mass_stratum_means(decision_samples)
    required_task_count = math.ceil(
        FIXED_OBSERVATION_MINIMUM_BREADTH_FRACTION * int(breadth["task_count"])
    )
    required_target_count = math.ceil(
        FIXED_OBSERVATION_MINIMUM_BREADTH_FRACTION * int(breadth["target_count"])
    )
    relation_changed_fraction = sum(
        sample["pair_metrics"]["relation_output_changed"] is True for sample in decision_samples
    ) / len(decision_samples)
    checks: dict[str, dict[str, object]] = {}
    for metric in _ADVANTAGE_METRICS:
        checks[f"{metric}_mean_positive"] = _check(
            passed=decision_means[metric] > 0.0,
            actual=decision_means[metric],
            relation=">",
            threshold=0.0,
        )
        checks[f"{metric}_improves_over_update_zero"] = _check(
            passed=decision_means[metric] > baseline_means[metric],
            actual=decision_means[metric] - baseline_means[metric],
            relation=">",
            threshold=0.0,
        )
    for metric, result in decision_sign_tests.items():
        checks[f"{metric}_source_episode_exact_sign_test"] = _check(
            passed=(
                result["zero_count"] == 0
                and float(result["one_sided_pvalue"]) <= FIXED_OBSERVATION_MAXIMUM_SIGN_TEST_PVALUE
            ),
            actual=result,
            relation="zero_count == 0 and one_sided_pvalue <=",
            threshold=FIXED_OBSERVATION_MAXIMUM_SIGN_TEST_PVALUE,
        )
    for metric, result in improvement_sign_tests.items():
        checks[f"{metric}_source_episode_improvement_exact_sign_test"] = _check(
            passed=(
                result["zero_count"] == 0
                and float(result["one_sided_pvalue"]) <= FIXED_OBSERVATION_MAXIMUM_SIGN_TEST_PVALUE
            ),
            actual=result,
            relation="zero_count == 0 and one_sided_pvalue <=",
            threshold=FIXED_OBSERVATION_MAXIMUM_SIGN_TEST_PVALUE,
        )
    for metric, fraction in bidirectional.items():
        checks[f"{metric}_bidirectional_fraction"] = _check(
            passed=fraction > FIXED_OBSERVATION_MINIMUM_BIDIRECTIONAL_FRACTION,
            actual=fraction,
            relation=">",
            threshold=FIXED_OBSERVATION_MINIMUM_BIDIRECTIONAL_FRACTION,
        )
    checks["task_breadth"] = _check(
        passed=int(breadth["positive_task_count"]) >= required_task_count,
        actual={
            "positive": breadth["positive_task_count"],
            "total": breadth["task_count"],
        },
        relation="positive >= ceil(2/3 * total)",
        threshold=required_task_count,
    )
    checks["target_breadth"] = _check(
        passed=int(breadth["positive_target_count"]) >= required_target_count,
        actual={
            "positive": breadth["positive_target_count"],
            "total": breadth["target_count"],
        },
        relation="positive >= ceil(2/3 * total)",
        threshold=required_target_count,
    )
    checks["all_mass_strata_positive"] = _check(
        passed=all(
            metrics["fractional_auc"] > 0.0 and metrics["row"] > 0.0
            for metrics in mass_strata.values()
        ),
        actual=mass_strata,
        relation="row > 0 and fractional_auc > 0 for every frozen stratum",
        threshold=True,
    )
    checks["relation_output_changes_for_every_pair"] = _check(
        passed=relation_changed_fraction == 1.0,
        actual=relation_changed_fraction,
        relation="==",
        threshold=1.0,
    )
    return {
        "baseline_mean_advantages": baseline_means,
        "breadth": breadth,
        "checks": checks,
        "decision_mean_advantages": decision_means,
        "source_episode_count": len(decision_episode_values["row"]),
        "source_episode_exact_sign_tests": decision_sign_tests,
        "source_episode_improvement_exact_sign_tests": improvement_sign_tests,
        "mass_strata": mass_strata,
        "sample_count": len(decision_samples),
        "status": ("PASS" if all(bool(check["passed"]) for check in checks.values()) else "FAIL"),
    }


def _fixed_observation_numeric_gate_value(
    baseline_snapshot: Mapping[str, Any],
    decision_snapshot: Mapping[str, Any],
    *,
    plan: FixedObservationEvaluationPlan,
) -> dict[str, object]:
    baseline = validate_fixed_observation_evaluation_snapshot(
        dict(baseline_snapshot),
        plan=plan,
    )
    decision = validate_fixed_observation_evaluation_snapshot(
        dict(decision_snapshot),
        plan=plan,
    )
    if (
        baseline["checkpoint_global_step"] != FIXED_OBSERVATION_BASELINE_STEP
        or decision["checkpoint_global_step"] != FIXED_OBSERVATION_DECISION_STEP
    ):
        raise ValueError("fixed-X gate requires exactly checkpoints 0 and 200")
    binding_fields = (
        "implementation_sha256",
        "model_family_sha256",
        "representation_split_sha256",
        "fixed_observation_evaluation_plan_sha256",
    )
    invariant_checks = {
        "evidence_bindings_unchanged": all(
            baseline[name] == decision[name] for name in binding_fields
        ),
        "frozen_action_state_unchanged": (
            baseline["representation_frozen_action_state_sha256"]
            == decision["representation_frozen_action_state_sha256"]
        ),
    }
    partitions = {
        partition: _partition_gate(
            _partition_samples(baseline, partition=partition),
            _partition_samples(decision, partition=partition),
        )
        for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
    }
    passed = all(invariant_checks.values()) and all(
        partition["status"] == "PASS" for partition in partitions.values()
    )
    value: dict[str, object] = {
        "schema": FIXED_OBSERVATION_NUMERIC_GATE_SCHEMA,
        "status": "PASS_PENDING_STANDARD_GATES_AND_VISUAL_REVIEW" if passed else "FAIL",
        "baseline_snapshot_sha256": baseline["artifact_sha256"],
        "decision_snapshot_sha256": decision["artifact_sha256"],
        "fixed_observation_evaluation_plan_sha256": plan.artifact_sha256,
        "numeric_gate_implementation_sha256": _numeric_gate_implementation_sha256(),
        "thresholds": {
            "baseline_step": FIXED_OBSERVATION_BASELINE_STEP,
            "decision_step": FIXED_OBSERVATION_DECISION_STEP,
            "maximum_one_sided_sign_test_pvalue": (FIXED_OBSERVATION_MAXIMUM_SIGN_TEST_PVALUE),
            "sign_test_unit": "source_episode_mean",
            "update_200_advantage_and_paired_improvement_both_require_significance": True,
            "minimum_bidirectional_fraction_exclusive": (
                FIXED_OBSERVATION_MINIMUM_BIDIRECTIONAL_FRACTION
            ),
            "minimum_breadth_fraction": FIXED_OBSERVATION_MINIMUM_BREADTH_FRACTION,
            "mass_strata_must_each_have_positive_row_and_fractional_auc": True,
        },
        "invariant_checks": invariant_checks,
        "partitions": partitions,
        "visual_review_required": True,
        "authorizes_action_or_long_training": False,
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return value


def build_fixed_observation_numeric_gate(
    baseline_snapshot: Mapping[str, Any],
    decision_snapshot: Mapping[str, Any],
    *,
    plan: FixedObservationEvaluationPlan,
) -> dict[str, object]:
    """Build the immutable fixed-X numeric decision."""

    return validate_fixed_observation_numeric_gate(
        _fixed_observation_numeric_gate_value(
            baseline_snapshot,
            decision_snapshot,
            plan=plan,
        ),
        baseline_snapshot=baseline_snapshot,
        decision_snapshot=decision_snapshot,
        plan=plan,
    )


def validate_fixed_observation_numeric_gate(
    value: object,
    *,
    baseline_snapshot: Mapping[str, Any],
    decision_snapshot: Mapping[str, Any],
    plan: FixedObservationEvaluationPlan,
) -> dict[str, Any]:
    """Recompute the complete fixed-X gate and reject edited conclusions."""

    if not isinstance(value, dict) or set(value) != _GATE_FIELDS:
        raise ValueError("fixed-X numeric gate fields differ from schema")
    if value["schema"] != FIXED_OBSERVATION_NUMERIC_GATE_SCHEMA:
        raise ValueError("fixed-X numeric gate schema changed")
    expected = _fixed_observation_numeric_gate_value(
        baseline_snapshot,
        decision_snapshot,
        plan=plan,
    )
    if value != expected:
        raise ValueError("fixed-X numeric gate was not recomputed from evidence")
    return value


def write_fixed_observation_numeric_gate(
    path: str | Path,
    value: Mapping[str, object],
) -> None:
    """Write one prevalidated gate atomically without replacing evidence."""

    write_bytes_durable_exclusive(
        path,
        _canonical_bytes(dict(value)) + b"\n",
    )
