from __future__ import annotations

from tools.audit_molmoact2_m2_variance_dead_zone import summarize_dead_zone


def _row(raw: float, gradient: float, counterfactual: float) -> dict[str, float]:
    return {
        "variance_raw": raw,
        "raw_gradient": gradient,
        "counterfactual_softplus_local_gradient": counterfactual,
    }


def test_dead_zone_requires_saturated_zero_gradients_and_interior_signal() -> None:
    report = summarize_dead_zone(
        (
            _row(0.5, 0.0, 0.2),
            _row(2.0, 0.0, -0.3),
            _row(-1.0, 0.1, 0.1),
            _row(-2.0, -0.2, -0.2),
        ),
        minimum_variance=1e-4,
    )

    assert report["upper_saturated"]["row_count"] == 2
    assert report["upper_saturated"]["exact_zero_gradient_fraction"] == 1.0
    assert report["interior"]["maximum_absolute_gradient"] == 0.2
    assert report["upper_dead_zone_established"]


def test_dead_zone_is_not_declared_when_saturated_gradient_is_recoverable() -> None:
    report = summarize_dead_zone(
        (
            _row(0.5, 0.1, 0.2),
            _row(-1.0, 0.1, 0.1),
        ),
        minimum_variance=1e-4,
    )

    assert not report["upper_dead_zone_established"]
