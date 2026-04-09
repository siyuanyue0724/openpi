import numpy as np

from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PointExpertState
from openpi.picf.posterior.contracts import PosteriorDebugMetrics
from openpi.picf.posterior.contracts import PosteriorState
from openpi.picf.posterior.prior import build_current_prior


def _dummy_previous(config: PosteriorConfig) -> PosteriorState:
    mu = np.arange(2 * config.dim_total, dtype=np.float32).reshape(2, config.dim_total)
    var_block = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float32)
    point = PointExpertState(
        mu=np.zeros_like(mu),
        var_block=np.ones((2, 3), dtype=np.float32),
        block_valid=np.zeros((2, 3), dtype=bool),
        gate=np.zeros((2,), dtype=bool),
        anchor_count=np.zeros((2,), dtype=np.int32),
        gamma_n=np.zeros((2,), dtype=np.float32),
        gamma_pc=np.zeros((2,), dtype=np.float32),
        delta_pc=np.zeros((2,), dtype=np.float32),
        delta2x=np.zeros((2, 3), dtype=np.float32),
    )
    debug = PosteriorDebugMetrics(
        point_gate_ratio=0.0,
        stale_prior_match_error=0.0,
        posterior_prior_equal_on_stale=True,
        fresh_scaffold=True,
        matched_prior_count=0,
        reset_prior_count=0,
        precision_gain_count=0,
        nan_count=0,
        max_abs_mu=0.0,
        min_var_block=0.0,
        max_var_block=0.0,
    )
    return PosteriorState(mu, var_block, mu, var_block, point, 0, 0, debug)


def test_build_current_prior_propagates_and_resets() -> None:
    config = PosteriorConfig()
    previous = _dummy_previous(config)
    matched_mask = np.array([True, False, True], dtype=bool)
    pred_idx = np.array([1, -1, 0], dtype=np.int32)

    mu_prop, var_prop, matched_count, reset_count = build_current_prior(
        config=config,
        matched_mask=matched_mask,
        pred_idx=pred_idx,
        previous=previous,
    )

    assert matched_count == 2
    assert reset_count == 1
    assert np.allclose(mu_prop[0], previous.mu[1])
    assert np.allclose(mu_prop[2], previous.mu[0])
    assert np.allclose(mu_prop[1], 0.0)
    assert np.allclose(var_prop[1], config.sigma_reset**2)
