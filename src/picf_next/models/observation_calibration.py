"""Train-only calibration for axis-constant Gaussian observation covariance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import shgo


@dataclass(frozen=True, slots=True)
class AxisConstantVarianceCalibration:
    """Immutable result of fitting one observation variance per physical axis."""

    observation_variance: tuple[float, ...]
    raw_softplus_bias: tuple[float, ...]
    supervised_count: tuple[int, ...]
    axis_nll_without_constant: tuple[float, ...]
    train_nll_without_constant: float
    minimum_variance: float
    fit_method: tuple[str, ...]


def _validated_arrays(
    residual: np.ndarray,
    target_variance: np.ndarray,
    supervised: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual_array = np.asarray(residual)
    target_array = np.asarray(target_variance)
    if (
        residual_array.ndim != 2
        or target_array.shape != residual_array.shape
        or not np.issubdtype(residual_array.dtype, np.floating)
        or not np.issubdtype(target_array.dtype, np.floating)
        or not np.isfinite(residual_array).all()
        or not np.isfinite(target_array).all()
        or (target_array < 0.0).any()
    ):
        raise ValueError("residual and target variance must be finite aligned float matrices")
    if residual_array.shape[0] == 0 or residual_array.shape[1] == 0:
        raise ValueError("observation calibration requires at least one row and axis")
    if supervised is None:
        supervised_array = np.ones(residual_array.shape, dtype=np.bool_)
    else:
        supervised_array = np.asarray(supervised)
        if supervised_array.dtype != np.bool_ or supervised_array.shape != residual_array.shape:
            raise ValueError("supervised must be an aligned boolean matrix")
    if not supervised_array.any(axis=0).all():
        raise ValueError("every physical axis requires at least one supervised train residual")
    return (
        residual_array.astype(np.float64, copy=False),
        target_array.astype(np.float64, copy=False),
        supervised_array,
    )


def _positive_minimum(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("minimum_variance must be a finite positive number")
    minimum = float(value)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum_variance must be a finite positive number")
    return minimum


def _representable_lower_bound(minimum_variance: float) -> float:
    return float(
        np.nextafter(
            np.float32(minimum_variance),
            np.float32(np.inf),
        )
    )


def variance_to_softplus_bias(variance: np.ndarray, *, minimum_variance: float) -> np.ndarray:
    """Convert positive variance to the model's finite raw-bias parameterization."""

    minimum = _positive_minimum(minimum_variance)
    value = np.asarray(variance)
    if (
        value.ndim != 1
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or (value <= minimum).any()
    ):
        raise ValueError("variance must be a finite float vector strictly above the minimum")
    delta = value.astype(np.float64, copy=False) - minimum
    raw = delta + np.log(-np.expm1(-delta))
    raw.setflags(write=False)
    return raw


def gaussian_axis_nll_without_constant(
    residual: np.ndarray,
    target_variance: np.ndarray,
    observation_variance: np.ndarray,
    *,
    supervised: np.ndarray | None = None,
) -> tuple[float, tuple[float, ...]]:
    """Evaluate the same diagonal Gaussian proper score used by discovery."""

    residual_array, target_array, supervised_array = _validated_arrays(
        residual,
        target_variance,
        supervised,
    )
    observation = np.asarray(observation_variance)
    if (
        observation.shape != (residual_array.shape[1],)
        or not np.issubdtype(observation.dtype, np.floating)
        or not np.isfinite(observation).all()
        or (observation <= 0.0).any()
    ):
        raise ValueError("observation_variance must be one finite positive value per axis")
    combined = target_array + observation.astype(np.float64, copy=False)[None, :]
    score = 0.5 * (np.square(residual_array) / combined + np.log(combined))
    axis_scores = tuple(
        float(score[supervised_array[:, axis], axis].mean())
        for axis in range(residual_array.shape[1])
    )
    return float(score[supervised_array].mean()), axis_scores


def _fit_one_axis(
    squared_residual: np.ndarray,
    target_variance: np.ndarray,
    *,
    minimum_variance: float,
) -> tuple[float, str]:
    lower = _representable_lower_bound(minimum_variance)
    if np.ptp(target_variance) == 0.0:
        optimum = float(squared_residual.mean() - target_variance[0])
        return max(optimum, lower), "analytic-homoscedastic-target"

    upper = max(float(squared_residual.max()), lower)
    if upper == lower:
        return lower, "boundary-heteroscedastic-target"

    def objective(value: np.ndarray) -> float:
        variance = float(value[0])
        combined = variance + target_variance
        return float(0.5 * np.mean(squared_residual / combined + np.log(combined)))

    result = shgo(
        objective,
        bounds=((lower, upper),),
        n=128,
        iters=3,
        sampling_method="simplicial",
        options={"f_tol": 1e-12},
    )
    if not result.success or result.x is None or not np.isfinite(result.fun):
        raise RuntimeError(f"global observation-variance calibration failed: {result.message}")
    candidates = [lower, upper, float(result.x[0])]
    local_minima = getattr(result, "xl", None)
    if local_minima is not None:
        candidates.extend(float(row[0]) for row in np.asarray(local_minima))
    optimum = min(candidates, key=lambda value: (objective(np.asarray([value])), value))
    return optimum, "shgo-heteroscedastic-target"


def fit_axis_constant_observation_variance(
    train_residual: np.ndarray,
    train_target_variance: np.ndarray,
    *,
    train_supervised: np.ndarray | None = None,
    minimum_variance: float,
) -> AxisConstantVarianceCalibration:
    """Fit one task/identity-independent variance per axis from train rows only."""

    minimum = _positive_minimum(minimum_variance)
    residual, target_variance, supervised = _validated_arrays(
        train_residual,
        train_target_variance,
        train_supervised,
    )
    fitted = []
    methods = []
    counts = []
    for axis in range(residual.shape[1]):
        selected = supervised[:, axis]
        variance, method = _fit_one_axis(
            np.square(residual[selected, axis]),
            target_variance[selected, axis],
            minimum_variance=minimum,
        )
        fitted.append(variance)
        methods.append(method)
        counts.append(int(selected.sum()))
    fitted_array = np.asarray(fitted, dtype=np.float64)
    raw_bias = variance_to_softplus_bias(fitted_array, minimum_variance=minimum)
    train_nll, axis_nll = gaussian_axis_nll_without_constant(
        residual,
        target_variance,
        fitted_array,
        supervised=supervised,
    )
    return AxisConstantVarianceCalibration(
        observation_variance=tuple(float(value) for value in fitted_array),
        raw_softplus_bias=tuple(float(value) for value in raw_bias),
        supervised_count=tuple(counts),
        axis_nll_without_constant=axis_nll,
        train_nll_without_constant=train_nll,
        minimum_variance=minimum,
        fit_method=tuple(methods),
    )
