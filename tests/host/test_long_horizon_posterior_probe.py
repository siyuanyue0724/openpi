from __future__ import annotations

# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from picf_next.eval.posterior_probe import run_long_horizon_posterior_probe


def test_long_horizon_probe_closes_identity_occlusion_and_uncertainty_invariants() -> None:
    report = run_long_horizon_posterior_probe(steps=160, seed=20260715)

    assert report["passed"], report
    assert report["query_order_count"] > 10
    assert len(report["occlusions"]) == 3
    assert report["max_posterior_support"] > 3
    assert report["max_tentative_ownership_leak"] == 0.0
    assert report["support_births_after_initialization"] > 0
    assert report["unexpected_births"] == 0


def test_bfloat16_long_horizon_probe_preserves_the_same_invariants() -> None:
    report = run_long_horizon_posterior_probe(
        steps=160,
        seed=20260715,
        dtype=torch.bfloat16,
    )

    assert report["passed"], report
    assert report["dtype"] == "bfloat16"


@pytest.mark.parametrize("steps", [0, 99, True])
def test_long_horizon_probe_rejects_invalid_length(steps: int) -> None:
    with pytest.raises(ValueError, match="at least 100"):
        run_long_horizon_posterior_probe(steps=steps)
