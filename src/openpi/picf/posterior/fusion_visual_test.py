import numpy as np

from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PointExpertState
from openpi.picf.posterior.contracts import VisualExpertState
from openpi.picf.posterior.fusion_visual import fuse_point_visual


def test_point_visual_fusion_combines_both_experts() -> None:
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
    visual_mu = np.zeros_like(mu_prop)
    visual_mu[0, : config.dim_h] = 3.0
    visual = VisualExpertState(
        mu=visual_mu,
        var_block=np.full((1, 3), 0.25, dtype=np.float32),
        block_valid=np.array([[True, True, False]]),
        gate=np.array([True]),
        in_view=np.array([True]),
        visibility=np.array([1.0], dtype=np.float32),
        depth_residual=np.array([0.0], dtype=np.float32),
        depth_available=np.array([True]),
    )

    mu, var_block, precision_gain_count, point_gain_count, visual_gain_count = fuse_point_visual(
        config=config,
        mu_prop=mu_prop,
        var_prop_block=var_prop,
        point=point,
        visual=visual,
    )

    assert precision_gain_count == 2
    assert point_gain_count == 1
    assert visual_gain_count == 1
    assert var_block[0, 0] < var_prop[0, 0]
    assert var_block[0, 1] < var_prop[0, 1]
    assert np.isclose(var_block[0, 2], var_prop[0, 2])
    assert np.max(mu[0, : config.dim_h]) > 0.0
    assert np.max(mu[0, config.dim_h : config.dim_h + config.dim_g]) > 0.0
    assert np.allclose(mu[0, config.dim_h + config.dim_g :], 0.0)
