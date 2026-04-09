from __future__ import annotations

import numpy as np

from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PointExpertState
from openpi.picf.posterior.contracts import VisualExpertState


def _clip_var(var_block: np.ndarray, config: PosteriorConfig) -> np.ndarray:
    return np.clip(np.asarray(var_block, dtype=np.float32), config.sigma_min2, config.sigma_max2)


def fuse_point_visual(
    *,
    config: PosteriorConfig,
    mu_prop: np.ndarray,
    var_prop_block: np.ndarray,
    point: PointExpertState,
    visual: VisualExpertState,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    mu = np.asarray(mu_prop, dtype=np.float32).copy()
    var_block = _clip_var(var_prop_block, config)
    point_precision_gain_count = int(np.sum(point.gate))
    visual_precision_gain_count = int(np.sum(visual.gate))

    prior_slices = (
        slice(0, config.dim_h),
        slice(config.dim_h, config.dim_h + config.dim_g),
        slice(config.dim_h + config.dim_g, config.dim_total),
    )

    for slot in range(mu.shape[0]):
        for block_index, block_slice in enumerate(prior_slices):
            precisions = [1.0 / float(var_block[slot, block_index])]
            natural_params = [mu_prop[slot, block_slice] / float(var_block[slot, block_index])]
            if bool(point.gate[slot]) and bool(point.block_valid[slot, block_index]):
                meas_var = float(point.var_block[slot, block_index])
                precisions.append(1.0 / meas_var)
                natural_params.append(point.mu[slot, block_slice] / meas_var)
            if bool(visual.gate[slot]) and bool(visual.block_valid[slot, block_index]):
                meas_var = float(visual.var_block[slot, block_index])
                precisions.append(1.0 / meas_var)
                natural_params.append(visual.mu[slot, block_slice] / meas_var)
            lambda_total = float(np.sum(precisions))
            eta_total = np.sum(np.stack(natural_params, axis=0), axis=0)
            fused_var = 1.0 / lambda_total
            mu[slot, block_slice] = fused_var * eta_total
            var_block[slot, block_index] = fused_var

    total_precision_gain_count = point_precision_gain_count + visual_precision_gain_count
    return mu, _clip_var(var_block, config), total_precision_gain_count, point_precision_gain_count, visual_precision_gain_count
