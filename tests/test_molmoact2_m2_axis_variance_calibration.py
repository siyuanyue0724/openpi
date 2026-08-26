from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.calibrate_molmoact2_m2_axis_variance import (
    _calibration_metrics,
    _decision,
    _examples_to_arrays,
    _load_config,
)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs/training/molmoact2_calvin_m2_axis_variance_calibration.json"


def _example(squared: tuple[float, float], supervised: tuple[bool, bool] = (True, True)):
    return {
        "squared_residual": np.asarray(squared, dtype=np.float32),
        "measurement_variance": np.zeros(2, dtype=np.float32),
        "supervised": np.asarray(supervised, dtype=np.bool_),
    }


def test_frozen_axis_variance_config_is_strict() -> None:
    config = _load_config(_CONFIG)
    assert config["protocol"]["fit_data"] == "train-fixed-match-residuals-only"

    changed = json.loads(_CONFIG.read_text(encoding="ascii"))
    changed["protocol"]["validation_and_heldout"] = "fit"
    temporary = _CONFIG.with_name("test-axis-calibration-invalid.json")
    try:
        temporary.write_text(json.dumps(changed), encoding="ascii")
        try:
            _load_config(temporary)
        except ValueError as error:
            assert "protocol changed" in str(error)
        else:
            raise AssertionError("changed protocol was accepted")
    finally:
        temporary.unlink(missing_ok=True)


def test_examples_and_metrics_respect_selective_axes() -> None:
    examples = [_example((1.0, 4.0)), _example((9.0, 400.0), (True, False))]
    residual, target, supervised = _examples_to_arrays(examples)
    np.testing.assert_allclose(residual, [[1.0, 2.0], [3.0, 20.0]])
    assert not target.any()
    assert supervised.tolist() == [[True, True], [True, False]]

    metrics = _calibration_metrics(examples, [5.0, 4.0])
    assert metrics["coordinate_count"] == 3
    assert metrics["axis_error_to_variance_ratio"] == [1.0, 1.0]
    assert metrics["aggregate_error_to_variance_ratio"] == 1.0


def test_decision_requires_generalization_and_state_isolation() -> None:
    reset = {
        split: {
            "gaussian_nll_without_constant": 1.0,
            "aggregate_error_to_variance_ratio": 1.0,
        }
        for split in ("train", "validation", "heldout")
    }
    fitted = {
        split: {
            "gaussian_nll_without_constant": 0.5,
            "aggregate_error_to_variance_ratio": 1.1,
        }
        for split in ("train", "validation", "heldout")
    }
    acceptance = _load_config(_CONFIG)["acceptance"]
    passed = _decision(
        reset_metrics=reset,
        fitted_metrics=fitted,
        nonvariance_state_exact=True,
        variance_weight_zero=True,
        softplus_roundtrip_error=0.0,
        acceptance=acceptance,
    )
    assert passed["status"] == "PASS"
    assert passed["later_gates_authorized"] == ["M3_bounded_mechanism_smoke"]
    assert passed["long_training_authorized"] is False

    fitted["heldout"]["aggregate_error_to_variance_ratio"] = 3.0
    failed = _decision(
        reset_metrics=reset,
        fitted_metrics=fitted,
        nonvariance_state_exact=True,
        variance_weight_zero=True,
        softplus_roundtrip_error=0.0,
        acceptance=acceptance,
    )
    assert failed["status"] == "FAIL"
    assert failed["later_gates_authorized"] == []
