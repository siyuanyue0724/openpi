from __future__ import annotations

import math

import pytest

from tools.audit_molmoact2_m2_uncertainty import (
    _MODEL_SOURCE_PATHS,
    _ROOT,
    _group_decomposition,
    _metric_summary,
    _reliability_bins,
)


def _row(group: str, variance: float, error: float, index: int) -> dict:
    return {
        "group": group,
        "predicted_variance": variance,
        "squared_error": error,
        "sample_key": f"sample-{index}",
        "identity_key": group,
        "axis": "x",
    }


def test_bound_model_source_paths_exist() -> None:
    assert all((_ROOT / relative).is_file() for relative in _MODEL_SOURCE_PATHS)


def test_uncertainty_metric_summary_reports_calibration_and_coverage() -> None:
    rows = [
        _row("a", 1.0, 1.0, 0),
        _row("a", 2.0, 2.0, 1),
        _row("a", 3.0, 3.0, 2),
    ]

    summary = _metric_summary(
        rows,
        variance_field="predicted_variance",
        error_field="squared_error",
    )

    assert summary["variance_error_spearman"] == 1.0
    assert summary["mean_error_to_variance_ratio"] == 1.0
    assert summary["aggregate_error_to_variance_ratio"] == 1.0
    assert summary["standardized_squared_error_coverage"]["within_1_sigma"] == 1.0
    assert math.isfinite(summary["gaussian_nll_without_constant"])


def test_group_decomposition_exposes_simpson_sign_reversal() -> None:
    rows = [
        _row("high-variance-low-error", 10.0, 1.0, 0),
        _row("high-variance-low-error", 11.0, 2.0, 1),
        _row("high-variance-low-error", 12.0, 3.0, 2),
        _row("low-variance-high-error", 1.0, 10.0, 3),
        _row("low-variance-high-error", 2.0, 11.0, 4),
        _row("low-variance-high-error", 3.0, 12.0, 5),
    ]

    overall = _metric_summary(
        rows,
        variance_field="predicted_variance",
        error_field="squared_error",
    )
    decomposition = _group_decomposition(
        rows,
        group_fields=("group",),
        variance_field="predicted_variance",
        error_field="squared_error",
    )

    assert overall["variance_error_spearman"] < 0.0
    assert decomposition["between_group_mean_spearman"] == pytest.approx(-1.0)
    assert decomposition["within_group_centered_rank_correlation"] == pytest.approx(1.0)
    assert decomposition["median_group_spearman"] == pytest.approx(1.0)


def test_reliability_bins_are_deterministic_and_exhaustive() -> None:
    rows = [_row("a", float(index + 1), float(index + 1), index) for index in range(23)]

    result = _reliability_bins(
        rows,
        variance_field="predicted_variance",
        error_field="squared_error",
        bin_count=10,
    )

    assert result["all_rows_accounted_for"] is True
    assert len(result["bins"]) == 10
    assert sum(row["row_count"] for row in result["bins"]) == len(rows)
    assert result["binned_variance_error_spearman"] == pytest.approx(1.0)
