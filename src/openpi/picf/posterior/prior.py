from __future__ import annotations

import numpy as np

from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PosteriorState


def build_current_prior(
    *,
    config: PosteriorConfig,
    matched_mask: np.ndarray,
    pred_idx: np.ndarray,
    previous: PosteriorState | None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    k_support = int(matched_mask.shape[0])
    dim_total = config.dim_total
    mu_prop = np.zeros((k_support, dim_total), dtype=np.float32)
    var_prop_block = np.full((k_support, 3), float(config.sigma_reset**2), dtype=np.float32)
    matched_prior_count = 0

    if previous is None:
        return mu_prop, var_prop_block, matched_prior_count, k_support

    motion = np.asarray(config.q_motion_block, dtype=np.float32).reshape(1, 3)
    for slot in np.flatnonzero(np.asarray(matched_mask, dtype=bool)):
        predecessor = int(pred_idx[slot])
        if predecessor < 0:
            continue
        mu_prop[slot] = previous.mu[predecessor]
        propagated = previous.var_block[predecessor] + motion[0]
        var_prop_block[slot] = propagated.astype(np.float32)
        matched_prior_count += 1

    reset_prior_count = int(k_support - matched_prior_count)
    return mu_prop, var_prop_block, matched_prior_count, reset_prior_count
