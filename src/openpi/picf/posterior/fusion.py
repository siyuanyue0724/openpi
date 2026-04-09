from __future__ import annotations

import numpy as np

from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PointExpertState


def _clip_var(var_block: np.ndarray, config: PosteriorConfig) -> np.ndarray:
    return np.clip(np.asarray(var_block, dtype=np.float32), config.sigma_min2, config.sigma_max2)


def fuse_point_only(
    *,
    config: PosteriorConfig,
    mu_prop: np.ndarray,
    var_prop_block: np.ndarray,
    point: PointExpertState,
) -> tuple[np.ndarray, np.ndarray, int]:
    mu = np.asarray(mu_prop, dtype=np.float32).copy()
    var_block = _clip_var(var_prop_block, config)
    precision_gain_count = 0

    prior_slices = (
        slice(0, config.dim_h),
        slice(config.dim_h, config.dim_h + config.dim_g),
        slice(config.dim_h + config.dim_g, config.dim_total),
    )

    for slot in range(mu.shape[0]):
        if not bool(point.gate[slot]):
            continue
        precision_gain_count += 1
        for block_index, block_slice in enumerate(prior_slices):
            if not bool(point.block_valid[slot, block_index]):
                continue
            prior_var = float(var_block[slot, block_index])
            meas_var = float(point.var_block[slot, block_index])
            lambda_total = 1.0 / prior_var + 1.0 / meas_var
            eta_total = mu_prop[slot, block_slice] / prior_var + point.mu[slot, block_slice] / meas_var
            fused_var = 1.0 / lambda_total
            mu[slot, block_slice] = fused_var * eta_total
            var_block[slot, block_index] = fused_var

    return mu, _clip_var(var_block, config), precision_gain_count
