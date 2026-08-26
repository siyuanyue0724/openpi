"""Host-neutral object-existence calibration and cardinality diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def probability_distribution(
    values: Sequence[float],
) -> dict[str, float | int | None]:
    """Summarize one finite scalar distribution without external dependencies."""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "maximum": None,
        }
    finite = [float(value) for value in values]
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("distribution values must be finite")
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "minimum": min(finite),
        "q05": _quantile(finite, 0.05),
        "q25": _quantile(finite, 0.25),
        "median": _quantile(finite, 0.50),
        "q75": _quantile(finite, 0.75),
        "q95": _quantile(finite, 0.95),
        "maximum": max(finite),
    }


def query_usage_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    query_count: int,
) -> dict[str, Any]:
    """Summarize Hungarian query use without treating query indices as identities."""

    if not isinstance(query_count, int) or isinstance(query_count, bool) or query_count <= 0:
        raise ValueError("query_count must be positive")
    sample_count = len(rows)
    matched_counts = [0] * query_count
    for row in rows:
        matched = row["matched_query_indices"]
        if len(set(matched)) != len(matched):
            raise ValueError("matched query indices must be unique within one sample")
        for query_index in matched:
            index = int(query_index)
            if not 0 <= index < query_count:
                raise ValueError("matched query index exceeds query capacity")
            matched_counts[index] += 1
    return {
        "sample_count": sample_count,
        "target_count": probability_distribution([float(row["target_count"]) for row in rows]),
        "query_match_frequency": [
            {
                "query_index": query_index,
                "matched_sample_count": matched_count,
                "matched_sample_fraction": (matched_count / sample_count if sample_count else 0.0),
            }
            for query_index, matched_count in enumerate(matched_counts)
        ],
    }


def task_usage_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    query_count: int,
) -> dict[str, Any]:
    """Group query-use diagnostics by task without assigning semantic slot names."""

    task_keys = sorted({str(row["task_key"]) for row in rows})
    return {
        task_key: query_usage_summary(
            [row for row in rows if row["task_key"] == task_key],
            query_count=query_count,
        )
        for task_key in task_keys
    }


def binary_calibration_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Return proper-score and reliability diagnostics for binary predictions."""

    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("binary calibration inputs must be nonempty and aligned")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError("calibration bins must be positive")
    rows = []
    epsilon = 1e-7
    for probability, label in zip(probabilities, labels, strict=True):
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("calibration probabilities must lie in [0, 1]")
        if label not in (0, 1):
            raise ValueError("calibration labels must be binary")
        rows.append((min(max(value, epsilon), 1.0 - epsilon), int(label)))

    brier = sum((probability - label) ** 2 for probability, label in rows) / len(rows)
    nll = -sum(
        label * math.log(probability) + (1 - label) * math.log1p(-probability)
        for probability, label in rows
    ) / len(rows)
    bin_rows = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            row
            for row in rows
            if lower <= row[0] <= upper and (index == bins - 1 or row[0] < upper)
        ]
        if not selected:
            continue
        confidence = sum(row[0] for row in selected) / len(selected)
        frequency = sum(row[1] for row in selected) / len(selected)
        error = abs(confidence - frequency)
        expected_calibration_error += len(selected) / len(rows) * error
        maximum_calibration_error = max(maximum_calibration_error, error)
        bin_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(selected),
                "mean_probability": confidence,
                "positive_frequency": frequency,
                "absolute_calibration_error": error,
            }
        )

    positives = [probability for probability, label in rows if label == 1]
    negatives = [probability for probability, label in rows if label == 0]
    return {
        "sample_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "brier": brier,
        "negative_log_likelihood": nll,
        "expected_calibration_error": expected_calibration_error,
        "maximum_calibration_error": maximum_calibration_error,
        "positive_probability": probability_distribution(positives),
        "negative_probability": probability_distribution(negatives),
        "bins": bin_rows,
    }


def continuous_calibration_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Measure calibration against a continuous correctness target in ``[0, 1]``."""

    if len(predictions) != len(targets) or not predictions:
        raise ValueError("continuous calibration inputs must be nonempty and aligned")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError("calibration bins must be positive")
    rows = []
    for prediction, target in zip(predictions, targets, strict=True):
        predicted = float(prediction)
        expected = float(target)
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in (predicted, expected)
        ):
            raise ValueError("continuous calibration values must lie in [0, 1]")
        rows.append((predicted, expected))

    absolute_errors = [abs(prediction - target) for prediction, target in rows]
    squared_errors = [(prediction - target) ** 2 for prediction, target in rows]
    bin_rows = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            row
            for row in rows
            if lower <= row[0] <= upper and (index == bins - 1 or row[0] < upper)
        ]
        if not selected:
            continue
        mean_prediction = sum(row[0] for row in selected) / len(selected)
        mean_target = sum(row[1] for row in selected) / len(selected)
        error = abs(mean_prediction - mean_target)
        expected_calibration_error += len(selected) / len(rows) * error
        maximum_calibration_error = max(maximum_calibration_error, error)
        bin_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(selected),
                "mean_prediction": mean_prediction,
                "mean_target": mean_target,
                "absolute_calibration_error": error,
            }
        )

    return {
        "sample_count": len(rows),
        "mean_absolute_error": sum(absolute_errors) / len(rows),
        "mean_squared_error": sum(squared_errors) / len(rows),
        "expected_calibration_error": expected_calibration_error,
        "maximum_calibration_error": maximum_calibration_error,
        "prediction": probability_distribution([row[0] for row in rows]),
        "target": probability_distribution([row[1] for row in rows]),
        "bins": bin_rows,
    }


def poisson_binomial_distribution(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Return the exact count posterior for independent Bernoulli queries."""

    distribution = [1.0]
    for probability in probabilities:
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("count probabilities must lie in [0, 1]")
        updated = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            updated[count] += mass * (1.0 - value)
            updated[count + 1] += mass * value
        distribution = updated
    total = sum(distribution)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("count posterior has invalid probability mass")
    return tuple(value / total for value in distribution)


def poisson_binomial_mode(probabilities: Sequence[float]) -> int:
    """Return the smallest Bayes-optimal count under exact-count zero-one loss."""

    distribution = poisson_binomial_distribution(probabilities)
    return max(range(len(distribution)), key=distribution.__getitem__)


def count_metrics(
    per_sample_probabilities: Sequence[Sequence[float]],
    target_counts: Sequence[int],
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate hard and posterior-mean object counts at one fixed threshold."""

    if (
        len(per_sample_probabilities) != len(target_counts)
        or not per_sample_probabilities
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("count inputs or threshold are invalid")
    hard_errors = []
    expected_errors = []
    posterior_mode_errors = []
    predicted_counts = []
    expected_counts = []
    posterior_mode_counts = []
    for probabilities, target in zip(per_sample_probabilities, target_counts, strict=True):
        if target < 0:
            raise ValueError("target counts must be nonnegative")
        values = [float(value) for value in probabilities]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("count probabilities must lie in [0, 1]")
        # Exactly neutral posterior odds are not positive object evidence.
        predicted = sum(value > threshold for value in values)
        expected = sum(values)
        posterior_mode = poisson_binomial_mode(values)
        predicted_counts.append(predicted)
        expected_counts.append(expected)
        posterior_mode_counts.append(posterior_mode)
        hard_errors.append(abs(predicted - target))
        expected_errors.append(abs(expected - target))
        posterior_mode_errors.append(abs(posterior_mode - target))
    return {
        "threshold": threshold,
        "sample_count": len(target_counts),
        "hard_count_mean": sum(predicted_counts) / len(predicted_counts),
        "target_count_mean": sum(target_counts) / len(target_counts),
        "hard_count_mae": sum(hard_errors) / len(hard_errors),
        "hard_exact_count_accuracy": sum(error == 0 for error in hard_errors) / len(hard_errors),
        "posterior_expected_count_mean": sum(expected_counts) / len(expected_counts),
        "posterior_expected_count_mae": sum(expected_errors) / len(expected_errors),
        "posterior_mode_count_mean": sum(posterior_mode_counts) / len(posterior_mode_counts),
        "posterior_mode_count_mae": sum(posterior_mode_errors) / len(posterior_mode_errors),
        "posterior_mode_exact_count_accuracy": sum(error == 0 for error in posterior_mode_errors)
        / len(posterior_mode_errors),
    }


def threshold_sweep(
    per_sample_probabilities: Sequence[Sequence[float]],
    target_counts: Sequence[int],
) -> list[dict[str, float | int]]:
    """Evaluate a fixed preregistered threshold grid."""

    thresholds = [index / 100.0 for index in range(1, 100)]
    return [
        count_metrics(
            per_sample_probabilities,
            target_counts,
            threshold=threshold,
        )
        for threshold in thresholds
    ]


def select_count_threshold(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int]:
    """Select on validation only, preferring calibration-preserving ties."""

    if not rows:
        raise ValueError("threshold selection requires candidate rows")
    return dict(
        min(
            rows,
            key=lambda row: (
                float(row["hard_count_mae"]),
                -float(row["hard_exact_count_accuracy"]),
                abs(float(row["threshold"]) - 0.5),
                float(row["threshold"]),
            ),
        )
    )
