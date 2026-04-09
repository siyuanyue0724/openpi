from __future__ import annotations

import numpy as np

from openpi.picf.contracts import SupportScaffoldState
from openpi.picf.frame_context import PointFrameContext
from openpi.picf.geometry import normalize_vectors
from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import PointExpertState


def _pad_block(features: np.ndarray, dim: int) -> np.ndarray:
    if features.shape[1] >= dim:
        return features[:, :dim].astype(np.float32)
    pad = np.zeros((features.shape[0], dim - features.shape[1]), dtype=np.float32)
    return np.concatenate([features.astype(np.float32), pad], axis=1)


def empty_point_expert(*, posterior_config: PosteriorConfig, k_support: int) -> PointExpertState:
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
    block_valid = np.zeros((k_support, 3), dtype=bool)
    anchor_count = np.zeros((k_support,), dtype=np.int32)
    gamma_n = np.zeros((k_support,), dtype=np.float32)
    gamma_pc = np.zeros((k_support,), dtype=np.float32)
    delta_pc = np.full((k_support,), np.inf, dtype=np.float32)
    delta2x = np.zeros((k_support, 3), dtype=np.float32)
    return PointExpertState(mu, var_block, block_valid, gate, anchor_count, gamma_n, gamma_pc, delta_pc, delta2x)


def build_point_expert(
    *,
    posterior_config: PosteriorConfig,
    scaffold_state: SupportScaffoldState,
    frame_context: PointFrameContext,
    point_features: np.ndarray | None = None,
) -> PointExpertState:
    k_support = scaffold_state.x.shape[0]
    state = empty_point_expert(posterior_config=posterior_config, k_support=k_support)

    if (
        not scaffold_state.debug.fresh_scaffold
        or frame_context.points_local.shape[0] == 0
        or scaffold_state.pi_geom.shape[1] == 0
    ):
        return state

    if point_features is not None and point_features.shape[0] != frame_context.points_local.shape[0]:
        raise ValueError(
            "Point feature / local-point mismatch: "
            f"{point_features.shape[0]} vs {frame_context.points_local.shape[0]}"
        )

    mu = state.mu
    var_block = state.var_block
    block_valid = state.block_valid
    gate = state.gate
    anchor_count = state.anchor_count
    gamma_n = state.gamma_n
    gamma_pc = state.gamma_pc
    delta_pc = state.delta_pc
    delta2x = state.delta2x

    for slot in range(k_support):
        if posterior_config.force_active_gate and not bool(scaffold_state.active_mask[slot]):
            continue
        weights = np.asarray(scaffold_state.pi_geom[slot], dtype=np.float32)
        weights_sum = float(weights.sum())
        if weights_sum <= 0.0:
            continue
        weights = weights / weights_sum
        centered_all = frame_context.points_local - scaffold_state.x[slot : slot + 1]
        delta2x[slot] = (weights[:, None] * (centered_all**2)).sum(axis=0)
        gamma_n[slot] = float((weights * (frame_context.normals_local @ scaffold_state.n[slot])).sum())

        dists = np.linalg.norm(centered_all, axis=1)
        neighborhood_radius = max(float(scaffold_state.r[slot]), float(posterior_config.point_radius_min_m))
        anchor_mask = dists <= neighborhood_radius
        anchor_count[slot] = int(np.count_nonzero(anchor_mask))
        if anchor_count[slot] == 0:
            continue
        points = frame_context.points_local[anchor_mask]

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
        geom_block = _pad_block(geom_features, posterior_config.dim_g)
        mu[slot, posterior_config.dim_h : posterior_config.dim_h + posterior_config.dim_g] = geom_block[0]
        if point_features is not None and point_features.shape[1] > 0:
            pooled = (weights[:, None] * point_features).sum(axis=0, keepdims=True)
            pooled = normalize_vectors(pooled, eps=posterior_config.epsilon_delta)
            mu[slot, : posterior_config.dim_h] = _pad_block(pooled, posterior_config.dim_h)[0]
            block_valid[slot, 0] = True
        block_valid[slot, 1] = True
        gate[slot] = True

    return PointExpertState(mu, var_block, block_valid, gate, anchor_count, gamma_n, gamma_pc, delta_pc, delta2x)
