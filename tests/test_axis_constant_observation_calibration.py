from __future__ import annotations

import numpy as np
import pytest

from picf_next.models.observation_calibration import (
    fit_axis_constant_observation_variance,
    gaussian_axis_nll_without_constant,
    variance_to_softplus_bias,
)


def test_homoscedastic_target_has_exact_axiswise_gaussian_solution() -> None:
    residual = np.asarray(
        [
            [1.0, 2.0],
            [-1.0, 4.0],
            [2.0, -2.0],
        ],
        dtype=np.float64,
    )
    target_variance = np.asarray([[0.5, 1.0]] * 3, dtype=np.float64)

    result = fit_axis_constant_observation_variance(
        residual,
        target_variance,
        minimum_variance=1e-4,
    )

    expected = np.square(residual).mean(axis=0) - target_variance[0]
    np.testing.assert_allclose(result.observation_variance, expected, rtol=1e-12, atol=0.0)
    assert result.fit_method == (
        "analytic-homoscedastic-target",
        "analytic-homoscedastic-target",
    )
    assert result.supervised_count == (3, 3)


def test_heteroscedastic_fit_selects_global_boundary_over_interior_local_minimum() -> None:
    squared_residual = np.asarray([7.23159695e2, 1.85645575e-4], dtype=np.float64)
    residual = np.sqrt(squared_residual)[:, None]
    target_variance = np.asarray([6.87453886e1, 2.54240635e-3], dtype=np.float64)[:, None]

    result = fit_axis_constant_observation_variance(
        residual,
        target_variance,
        minimum_variance=1e-4,
    )

    assert result.fit_method == ("shgo-heteroscedastic-target",)
    assert result.observation_variance[0] < 1.001e-4
    fitted_nll = result.train_nll_without_constant
    interior_nll, _ = gaussian_axis_nll_without_constant(
        residual,
        target_variance,
        np.asarray([247.95], dtype=np.float64),
    )
    assert fitted_nll < interior_nll


def test_selective_supervision_and_softplus_bias_round_trip() -> None:
    residual = np.asarray([[1.0, 2.0], [3.0, 100.0]], dtype=np.float64)
    target_variance = np.zeros_like(residual)
    supervised = np.asarray([[True, True], [True, False]])

    result = fit_axis_constant_observation_variance(
        residual,
        target_variance,
        train_supervised=supervised,
        minimum_variance=1e-4,
    )
    variance = np.asarray(result.observation_variance)
    raw = variance_to_softplus_bias(variance, minimum_variance=1e-4)
    reconstructed = np.logaddexp(0.0, raw) + 1e-4

    np.testing.assert_allclose(variance, [5.0, 4.0])
    np.testing.assert_allclose(reconstructed, variance, rtol=1e-12, atol=1e-12)
    assert result.supervised_count == (2, 1)


@pytest.mark.parametrize(
    ("residual", "target", "supervised", "message"),
    [
        (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            None,
            "at least one row",
        ),
        (
            np.ones((2, 2), dtype=np.float64),
            -np.ones((2, 2), dtype=np.float64),
            None,
            "finite aligned float matrices",
        ),
        (
            np.ones((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.asarray([[True, False], [True, False]]),
            "every physical axis",
        ),
    ],
)
def test_calibration_rejects_invalid_or_unidentified_inputs(
    residual: np.ndarray,
    target: np.ndarray,
    supervised: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_axis_constant_observation_variance(
            residual,
            target,
            train_supervised=supervised,
            minimum_variance=1e-4,
        )
