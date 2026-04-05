from __future__ import annotations

import numpy as np

from openpi.picf.contracts import SupportScaffoldState
from openpi.picf.frame_context import PointFrameContext
from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PointExpertState
from openpi.picf.scaffold.pipeline import DeterministicScaffoldConfig


def _pad_geometry_block(features: np.ndarray, dim_g: int) -> np.ndarray:
    if features.shape[1] >= dim_g:
        return features[:, :dim_g].astype(np.float32)
    pad = np.zeros((features.shape[0], dim_g - features.shape[1]), dtype=np.float32)
    return np.concatenate([features.astype(np.float32), pad], axis=1)


def build_point_expert(
    *,
    posterior_config: PosteriorConfig,
    scaffold_config: DeterministicScaffoldConfig,
    scaffold_state: SupportScaffoldState,
    frame_context: PointFrameContext,
) -> PointExpertState:
    k_support = scaffold_state.x.shape[0]
    dim_total = posterior_config.dim_total
    mu = np.zeros((k_support, dim_total), dtype=np.float32)
    var_block = np.tile(
        np.array(
            [posterior_config.point_var_h, posterior_config.point_var_g, posterior_config.point_var_c],
            dtype=np.float32,
        )[None, :],
        (k_support, 1),
    )
    gate = np.zeros((k_support,), dtype=bool)
    anchor_count = np.zeros((k_support,), dtype=np.int32)
    gamma_n = np.zeros((k_support,), dtype=np.float32)
    gamma_pc = np.zeros((k_support,), dtype=np.float32)
    delta_pc = np.full((k_support,), np.inf, dtype=np.float32)
    delta2x = np.zeros((k_support, 3), dtype=np.float32)

    if not scaffold_state.debug.fresh_scaffold or frame_context.points_local.shape[0] == 0 or scaffold_state.pi_geom.shape[1] == 0:
        return PointExpertState(mu, var_block, gate, anchor_count, gamma_n, gamma_pc, delta_pc, delta2x)

    for slot in range(k_support):
        if posterior_config.force_active_gate and not bool(scaffold_state.active_mask[slot]):
            continue
        dists = np.linalg.norm(frame_context.points_local - scaffold_state.x[slot : slot + 1], axis=1)
        neighborhood_radius = max(
            float(scaffold_state.r[slot]),
            float(scaffold_config.seed_init_radius_m),
            float(posterior_config.point_radius_min_m),
        )
        support_nonzero = dists <= neighborhood_radius
        if not np.any(support_nonzero):
            continue
        points = frame_context.points_local[support_nonzero]
        normals = frame_context.normals_local[support_nonzero]
        local_weights = np.exp(-(dists[support_nonzero] ** 2) / max(2.0 * neighborhood_radius * neighborhood_radius, 1e-8))
        local_weights = local_weights.astype(np.float32)
        local_weights /= max(float(local_weights.sum()), 1e-8)
        anchor_count[slot] = int(points.shape[0])
        centered = points - scaffold_state.x[slot : slot + 1]
        delta2x[slot] = (local_weights[:, None] * (centered**2)).sum(axis=0)
        gamma_n[slot] = float((local_weights * (normals @ scaffold_state.n[slot])).sum())

        if points.shape[0] >= 2:
            pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
            np.fill_diagonal(pairwise, np.inf)
            delta_pc[slot] = float(np.median(pairwise.min(axis=1)))
        elif points.shape[0] == 1:
            delta_pc[slot] = float("inf")

        gamma_pc_tilde = posterior_config.delta_ref_m / (delta_pc[slot] + posterior_config.epsilon_delta)
        if gamma_pc_tilde < posterior_config.gamma_min_pc or anchor_count[slot] < posterior_config.n_min_anchors:
            continue
        gamma_pc[slot] = float(np.clip(gamma_pc_tilde, posterior_config.gamma_min_pc, 1.0))
        var_block[slot, 1] = posterior_config.point_var_g / gamma_pc[slot]
        anchor_norm = min(anchor_count[slot] / max(posterior_config.anchor_count_norm, 1.0), 1.0)
        geom_features = np.concatenate(
            [
                scaffold_state.x[slot],
                scaffold_state.n[slot],
                np.array([scaffold_state.r[slot]], dtype=np.float32),
                delta2x[slot],
                np.array([gamma_n[slot], gamma_pc[slot], anchor_norm], dtype=np.float32),
            ],
            axis=0,
        )[None, :]
        geom_block = _pad_geometry_block(geom_features, posterior_config.dim_g)
        mu[slot, posterior_config.dim_h : posterior_config.dim_h + posterior_config.dim_g] = geom_block[0]
        gate[slot] = True

    return PointExpertState(mu, var_block, gate, anchor_count, gamma_n, gamma_pc, delta_pc, delta2x)
