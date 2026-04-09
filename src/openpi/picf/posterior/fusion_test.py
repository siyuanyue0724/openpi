import numpy as np

from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PointExpertState
from openpi.picf.posterior.fusion import fuse_point_only


def test_fusion_equals_prior_when_gate_off() -> None:
    config = PosteriorConfig()
    mu_prop = np.ones((2, config.dim_total), dtype=np.float32)
    var_prop = np.full((2, 3), 0.5, dtype=np.float32)
    point = PointExpertState(
        mu=np.zeros_like(mu_prop),
        var_block=np.ones((2, 3), dtype=np.float32),
        block_valid=np.zeros((2, 3), dtype=bool),
        gate=np.zeros((2,), dtype=bool),
        anchor_count=np.zeros((2,), dtype=np.int32),
        gamma_n=np.zeros((2,), dtype=np.float32),
        gamma_pc=np.zeros((2,), dtype=np.float32),
        delta_pc=np.zeros((2,), dtype=np.float32),
        delta2x=np.zeros((2, 3), dtype=np.float32),
    )
    mu, var_block, precision_gain_count = fuse_point_only(
        config=config,
        mu_prop=mu_prop,
        var_prop_block=var_prop,
        point=point,
    )
    assert precision_gain_count == 0
    assert np.allclose(mu, mu_prop)
    assert np.allclose(var_block, var_prop)


def test_fusion_reduces_variance_when_gate_on() -> None:
    config = PosteriorConfig()
    mu_prop = np.zeros((1, config.dim_total), dtype=np.float32)
    var_prop = np.full((1, 3), 1.0, dtype=np.float32)
    point_mu = np.zeros_like(mu_prop)
    point_mu[0, config.dim_h : config.dim_h + config.dim_g] = 2.0
    point = PointExpertState(
        mu=point_mu,
        var_block=np.full((1, 3), 0.5, dtype=np.float32),
        block_valid=np.array([[False, True, False]]),
        gate=np.array([True]),
        anchor_count=np.array([8], dtype=np.int32),
        gamma_n=np.array([1.0], dtype=np.float32),
        gamma_pc=np.array([1.0], dtype=np.float32),
        delta_pc=np.array([0.001], dtype=np.float32),
        delta2x=np.zeros((1, 3), dtype=np.float32),
    )
    mu, var_block, precision_gain_count = fuse_point_only(
        config=config,
        mu_prop=mu_prop,
        var_prop_block=var_prop,
        point=point,
    )
    assert precision_gain_count == 1
    assert var_block[0, 1] < var_prop[0, 1]
    assert np.isclose(var_block[0, 0], var_prop[0, 0])
    assert np.isclose(var_block[0, 2], var_prop[0, 2])
    assert np.max(mu[0, config.dim_h : config.dim_h + config.dim_g]) > 0
    assert np.allclose(mu[0, : config.dim_h], 0.0)
    assert np.allclose(mu[0, config.dim_h + config.dim_g :], 0.0)
