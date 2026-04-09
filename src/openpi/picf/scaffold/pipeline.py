from __future__ import annotations

import dataclasses

import numpy as np

from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import RuntimeMeta
from openpi.picf.contracts import ScaffoldDebugMetrics
from openpi.picf.contracts import SupportScaffoldState
from openpi.picf.frame_context import PointFrameContext
from openpi.picf.frame_context import build_point_frame_context
from openpi.picf.geometry import invert_transform
from openpi.picf.geometry import normalize_vectors
from openpi.picf.geometry import transform_normals
from openpi.picf.geometry import transform_points
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.scaffold.birth import coverage_weights
from openpi.picf.scaffold.birth import weighted_fps
from openpi.picf.scaffold.local_frame import EndEffectorLocalFrame
from openpi.picf.scaffold.matching import match_supports
from openpi.picf.sonata.wrapper import SonataPointFeatureExtractor


@dataclasses.dataclass(frozen=True)
class DeterministicScaffoldConfig:
    k_support: int = 96
    k_birth: int = 12
    k_active: int = 96
    crop_radius_m: float = 0.08
    r_min_m: float = 0.003
    r_max_m: float = 0.02
    tau_sup: float = 0.16
    target_points_per_support: int = 24
    seed_init_radius_m: float = 0.01
    grouping_neighbors: int = 32
    grouping_radius_factor: float = 3.25
    tau_p_m: float = 0.034
    tau_n: float = 0.8
    epsilon_n: float = 1e-6
    epsilon_app: float = 1e-6
    lambda_app_match_m: float = 1e-3
    query_rounds: int = 2
    query_geom_weight: float = 1.0
    query_normal_weight: float = 0.5
    query_app_weight: float = 0.5
    delta_sync_s: float = 0.02
    tau_rgb_proj_m: float = 0.005
    delta_pc_scaf_s: float = 0.15
    n_hold_scaf: int = 2
    v_rgb_identity: bool = True
    query_cache_dim: int = 128
    n_min_anchors: int = 8
    birth_weight_threshold: float = 0.2
    seed_radius_knn: int = 8
    matched_omega_momentum: float = 0.9
    matched_cache_momentum: float = 0.9
    birth_omega_scale: float = 0.35
    unmatched_carry_omega_scale: float = 0.9


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.size == 0:
        return logits
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    denom = float(np.sum(exp_logits))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.full_like(logits, 1.0 / max(logits.size, 1), dtype=np.float32)
    return (exp_logits / denom).astype(np.float32)


def _cosine_similarity(x: np.ndarray, y: np.ndarray, *, eps: float) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= eps:
        return 0.0
    return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))


class DeterministicScaffoldPipeline:
    def __init__(
        self,
        pointcloud_builder: CalvinDepthToPicfPointCloud,
        *,
        config: DeterministicScaffoldConfig | None = None,
        local_frame: EndEffectorLocalFrame | None = None,
        point_feature_extractor: SonataPointFeatureExtractor | None = None,
    ):
        self.pointcloud_builder = pointcloud_builder
        self.config = config or DeterministicScaffoldConfig()
        self.local_frame = local_frame or EndEffectorLocalFrame()
        self.point_feature_extractor = point_feature_extractor

    def _build_runtime_meta(self, observation: PicfObservation, previous: RuntimeMeta | None) -> RuntimeMeta:
        if observation.runtime_meta is not None:
            return dataclasses.replace(observation.runtime_meta)
        meta = dataclasses.replace(previous) if previous is not None else RuntimeMeta()
        rgb_static = np.asarray(observation.rgb_static)
        visual_valid = rgb_static.size > 0 and bool(np.isfinite(rgb_static).all())
        if observation.reset_scaffold:
            meta.n_vis_upd = 0
        if visual_valid:
            meta.t_v_last = float(observation.timestamp_s)
            meta.n_vis_upd = 1 if observation.reset_scaffold else (meta.n_vis_upd + 1)
        point_frame = observation.point_set
        if point_frame is not None and point_frame.frame_valid:
            meta.t_p_last = float(observation.timestamp_s)
            meta.t_rgb_last = float(observation.timestamp_s)
            meta.b_rgb_avail = True
            meta.rgb_proj_residual = 0.0
        else:
            meta.b_rgb_avail = False
            meta.rgb_proj_residual = float("inf")
        meta.v_pc_scaf = bool((observation.timestamp_s - meta.t_p_last) <= self.config.delta_pc_scaf_s)
        meta.v_rgb_p = bool(
            meta.v_pc_scaf
            and meta.b_rgb_avail
            and abs(meta.t_rgb_last - meta.t_p_last) <= self.config.delta_sync_s
            and meta.rgb_proj_residual <= self.config.tau_rgb_proj_m
        )
        meta.stale_scaffold_steps = 0 if meta.v_pc_scaf else (0 if previous is None else previous.stale_scaffold_steps + 1)
        return meta

    def _build_frame_context(self, observation: PicfObservation) -> PointFrameContext:
        return build_point_frame_context(
            observation,
            crop_radius_m=self.config.crop_radius_m,
            local_frame=self.local_frame,
        )

    @staticmethod
    def _descriptor_dim(
        previous: SupportScaffoldState | None,
        descriptors: np.ndarray | None,
    ) -> int:
        if previous is not None:
            return int(previous.e_id.shape[1])
        if descriptors is not None:
            return int(descriptors.shape[1])
        return 0

    def _seed_from_points(
        self,
        points_local: np.ndarray,
        normals_local: np.ndarray,
        descriptors: np.ndarray,
        seed_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed_x = np.zeros((self.config.k_support, 3), dtype=np.float32)
        seed_n = np.zeros((self.config.k_support, 3), dtype=np.float32)
        seed_r = np.full((self.config.k_support,), self.config.seed_init_radius_m, dtype=np.float32)
        seed_eid = np.zeros((self.config.k_support, descriptors.shape[1]), dtype=np.float32)
        for slot, point_idx in enumerate(seed_indices.tolist()):
            seed_x[slot] = points_local[point_idx]
            seed_n[slot] = normals_local[point_idx]
            seed_eid[slot] = descriptors[point_idx]
            if points_local.shape[0] >= 2:
                dists = np.linalg.norm(points_local - points_local[point_idx : point_idx + 1], axis=1)
                positive = np.sort(dists[dists > 0.0])
                if positive.size > 0:
                    k = min(int(self.config.seed_radius_knn), int(positive.size))
                    seed_r[slot] = float(
                        np.clip(
                            np.median(positive[:k]),
                            self.config.r_min_m,
                            self.config.r_max_m,
                        )
                    )
        return seed_x, seed_n, seed_r, seed_eid

    def _prepare_seeds(
        self,
        observation: PicfObservation,
        points_local: np.ndarray,
        normals_local: np.ndarray,
        descriptors: np.ndarray,
        previous: SupportScaffoldState | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed_x = np.zeros((self.config.k_support, 3), dtype=np.float32)
        seed_n = np.zeros((self.config.k_support, 3), dtype=np.float32)
        seed_r = np.full((self.config.k_support,), self.config.seed_init_radius_m, dtype=np.float32)
        desc_dim = self._descriptor_dim(previous, descriptors)
        seed_eid = np.zeros((self.config.k_support, desc_dim), dtype=np.float32)
        seed_qry = np.zeros((self.config.k_support, self.config.query_cache_dim), dtype=np.float32)
        seed_from_birth = np.zeros((self.config.k_support,), dtype=bool)
        seed_valid = np.zeros((self.config.k_support,), dtype=bool)
        seed_prev_idx = np.full((self.config.k_support,), -1, dtype=np.int32)
        next_slot = 0
        carried_centers = np.zeros((0, 3), dtype=np.float32)
        carried_radii = np.zeros((0,), dtype=np.float32)

        if previous is not None and not observation.reset_scaffold:
            carry_slots = np.flatnonzero(previous.active_mask)[: self.config.k_support]
            if carry_slots.size > 0:
                prev_global = transform_points(previous.x[carry_slots], previous.G_t)
                prev_normals_global = transform_normals(previous.n[carry_slots], previous.G_t)
                world_to_local = invert_transform(observation.G_t)
                carried_centers = transform_points(prev_global, world_to_local)
                carried_normals = transform_normals(prev_normals_global, world_to_local)
                carried_keep = np.linalg.norm(carried_centers, axis=1) <= (self.config.crop_radius_m + self.config.r_max_m)
                carry_slots = carry_slots[carried_keep]
                carried_centers = carried_centers[carried_keep]
                carried_normals = carried_normals[carried_keep]
                if carry_slots.size == 0:
                    carried_centers = np.zeros((0, 3), dtype=np.float32)
                    carried_normals = np.zeros((0, 3), dtype=np.float32)
                    carried_radii = np.zeros((0,), dtype=np.float32)
                else:
                    carried_radii = previous.r[carry_slots]
                carried_count = min(int(carry_slots.size), self.config.k_support)
                seed_x[:carried_count] = carried_centers[:carried_count]
                seed_n[:carried_count] = carried_normals[:carried_count]
                seed_r[:carried_count] = previous.r[carry_slots[:carried_count]]
                if previous.e_id.shape[1] == seed_eid.shape[1]:
                    seed_eid[:carried_count] = previous.e_id[carry_slots[:carried_count]]
                seed_qry[:carried_count] = previous.s_qry[carry_slots[:carried_count]]
                seed_valid[:carried_count] = True
                seed_prev_idx[:carried_count] = carry_slots[:carried_count]
                next_slot = carried_count

        if points_local.shape[0] == 0 or next_slot >= self.config.k_support:
            return seed_x, seed_n, seed_r, seed_eid, seed_qry, seed_from_birth, seed_valid, seed_prev_idx

        remaining = self.config.k_support if previous is None or observation.reset_scaffold else min(
            self.config.k_birth, self.config.k_support - next_slot
        )
        if remaining <= 0:
            return seed_x, seed_n, seed_r, seed_eid, seed_qry, seed_from_birth, seed_valid, seed_prev_idx

        coverage_sigmas = None
        if carried_centers.shape[0] > 0:
            coverage_sigmas = np.clip(carried_radii, self.config.r_min_m, self.config.r_max_m)
        weights = coverage_weights(
            points_local,
            carried_centers,
            sigma=max(self.config.seed_init_radius_m, self.config.r_min_m) if coverage_sigmas is None else None,
            sigmas=coverage_sigmas,
        )
        if previous is not None and not observation.reset_scaffold:
            candidate_indices = np.flatnonzero(weights > self.config.birth_weight_threshold)
            if candidate_indices.size == 0:
                return seed_x, seed_n, seed_r, seed_eid, seed_qry, seed_from_birth, seed_valid, seed_prev_idx
            birth_count = min(int(candidate_indices.size), int(remaining))
            filtered = weighted_fps(points_local[candidate_indices], weights[candidate_indices], birth_count)
            birth_indices = candidate_indices[filtered]
        else:
            birth_indices = weighted_fps(points_local, weights, remaining)
        bx, bn, br, be = self._seed_from_points(points_local, normals_local, descriptors, birth_indices)
        end = next_slot + birth_indices.shape[0]
        seed_x[next_slot:end] = bx[: birth_indices.shape[0]]
        seed_n[next_slot:end] = bn[: birth_indices.shape[0]]
        seed_r[next_slot:end] = br[: birth_indices.shape[0]]
        seed_eid[next_slot:end] = be[: birth_indices.shape[0]]
        for slot in range(next_slot, end):
            seed_qry[slot] = self._build_birth_query_code(slot)
        seed_from_birth[next_slot:end] = True
        seed_valid[next_slot:end] = True
        return seed_x, seed_n, seed_r, seed_eid, seed_qry, seed_from_birth, seed_valid, seed_prev_idx

    def _group_points(
        self,
        points_local: np.ndarray,
        normals_local: np.ndarray,
        point_features: np.ndarray,
        seed_x: np.ndarray,
        seed_n: np.ndarray,
        seed_r: np.ndarray,
        seed_eid: np.ndarray,
        seed_qry: np.ndarray,
        seed_valid: np.ndarray,
        meta: RuntimeMeta,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_points = points_local.shape[0]
        pi = np.zeros((self.config.k_support, n_points), dtype=np.float32)
        x = np.zeros((self.config.k_support, 3), dtype=np.float32)
        n = np.zeros((self.config.k_support, 3), dtype=np.float32)
        r = np.zeros((self.config.k_support,), dtype=np.float32)
        omega = np.zeros((self.config.k_support,), dtype=np.float32)
        e_id = np.zeros((self.config.k_support, point_features.shape[1]), dtype=np.float32)
        s_qry = np.zeros((self.config.k_support, self.config.query_cache_dim), dtype=np.float32)
        normal_fallbacks = 0

        for slot in range(self.config.k_support):
            if not bool(seed_valid[slot]):
                continue
            center = seed_x[slot].astype(np.float32, copy=True)
            normal = normalize_vectors(seed_n[slot : slot + 1], eps=self.config.epsilon_n)[0]
            radius = float(np.clip(seed_r[slot], self.config.r_min_m, self.config.r_max_m))
            desc = seed_eid[slot].astype(np.float32, copy=True)
            query_cache = seed_qry[slot].astype(np.float32, copy=True)
            final_weights = None

            for _ in range(max(self.config.query_rounds, 1)):
                dists = np.linalg.norm(points_local - center[None, :], axis=1)
                radius_limit = max(float(self.config.grouping_radius_factor * radius), self.config.r_min_m)
                neighbor_mask = dists <= radius_limit
                neighbors = np.flatnonzero(neighbor_mask)
                if neighbors.size > self.config.grouping_neighbors:
                    ranked = np.argsort(dists[neighbors])[: self.config.grouping_neighbors]
                    neighbors = neighbors[ranked]
                if neighbors.size == 0:
                    seed_valid[slot] = False
                    final_weights = None
                    break
                sigma = max(radius, self.config.r_min_m)
                geom_logits = -self.config.query_geom_weight * (dists[neighbors] ** 2) / max(2.0 * sigma * sigma, 1e-8)
                normal_logits = self.config.query_normal_weight * np.clip(normals_local[neighbors] @ normal, -1.0, 1.0)
                logits = geom_logits + normal_logits

                if (
                    meta.v_rgb_p
                    and self.config.v_rgb_identity
                    and point_features.shape[1] > 0
                    and desc.shape[0] == point_features.shape[1]
                    and np.linalg.norm(desc) > self.config.epsilon_app
                ):
                    point_desc = normalize_vectors(point_features[neighbors], eps=self.config.epsilon_app)
                    seed_desc = normalize_vectors(desc[None, :], eps=self.config.epsilon_app)[0]
                    logits = logits + self.config.query_app_weight * np.sum(point_desc * seed_desc[None, :], axis=1)

                local_weights = _stable_softmax(logits)
                if not np.any(np.isfinite(local_weights)) or float(local_weights.sum()) <= 0.0:
                    local_weights = np.full((neighbors.size,), 1.0 / max(neighbors.size, 1), dtype=np.float32)
                weights = np.zeros((n_points,), dtype=np.float32)
                weights[neighbors] = local_weights
                final_weights = weights

                new_center = (weights[:, None] * points_local).sum(axis=0)
                pooled_norm = (weights[:, None] * normals_local).sum(axis=0)
                norm = float(np.linalg.norm(pooled_norm))
                if norm < self.config.epsilon_n:
                    new_normal = normal
                    normal_fallbacks += 1
                else:
                    new_normal = pooled_norm / norm
                sqdist = np.sum((points_local - new_center[None, :]) ** 2, axis=1)
                new_r2 = float(np.clip((weights * sqdist).sum(), self.config.r_min_m**2, self.config.r_max_m**2))
                new_radius = float(np.sqrt(new_r2))
                if point_features.shape[1] > 0:
                    pooled_desc = (weights[:, None] * point_features).sum(axis=0)
                    new_desc = normalize_vectors(pooled_desc[None, :], eps=self.config.epsilon_app)[0]
                else:
                    new_desc = desc

                candidate_cache = self._build_query_cache(
                    new_center[None, :],
                    new_normal[None, :],
                    np.array([new_radius], dtype=np.float32),
                    np.array([1.0], dtype=np.float32),
                    new_desc[None, :],
                    base_cache=query_cache[None, :],
                )[0]
                keep_ratio = 0.5 + 0.25 * _cosine_similarity(query_cache, candidate_cache, eps=self.config.epsilon_app)
                update_ratio = float(np.clip(1.0 - keep_ratio, 0.25, 0.75))
                center = (1.0 - update_ratio) * center + update_ratio * new_center
                normal = normalize_vectors(
                    ((1.0 - update_ratio) * normal + update_ratio * new_normal)[None, :],
                    eps=self.config.epsilon_n,
                )[0]
                radius = float((1.0 - update_ratio) * radius + update_ratio * new_radius)
                if desc.shape[0] > 0:
                    desc = normalize_vectors(
                        ((1.0 - update_ratio) * desc + update_ratio * new_desc)[None, :],
                        eps=self.config.epsilon_app,
                    )[0]
                query_cache = ((1.0 - update_ratio) * query_cache + update_ratio * candidate_cache).astype(np.float32)

            if final_weights is None:
                seed_valid[slot] = False
                continue

            pi[slot] = final_weights
            x[slot] = center
            n[slot] = normal
            r[slot] = float(np.clip(radius, self.config.r_min_m, self.config.r_max_m))
            s_qry[slot] = query_cache
            neighborhood = np.flatnonzero(np.linalg.norm(points_local - center[None, :], axis=1) <= r[slot])
            near_ratio = float(np.exp(-np.sum(final_weights * np.linalg.norm(points_local - center[None, :], axis=1)) / max(r[slot], self.config.r_min_m)))
            density_ratio = float(np.clip(neighborhood.size / max(self.config.target_points_per_support, 1), 0.0, 1.0))
            if neighborhood.size < self.config.n_min_anchors:
                density_ratio *= neighborhood.size / max(self.config.n_min_anchors, 1)
            omega[slot] = float(np.clip(near_ratio * density_ratio, 0.0, 1.0))
            if meta.v_rgb_p and point_features.shape[1] > 0:
                e_id[slot] = desc

        empty_ratio = float(np.mean(np.sum(pi, axis=1) <= 0))
        fallback_ratio = float(normal_fallbacks / max(np.count_nonzero(np.sum(pi, axis=1) > 0), 1))
        return pi, x, n, r, omega, e_id, s_qry, fallback_ratio, empty_ratio

    def _build_birth_query_code(self, slot: int) -> np.ndarray:
        idx = float(slot + 1)
        phases = idx * np.arange(1, self.config.query_cache_dim + 1, dtype=np.float32)
        code = np.sin(phases / max(self.config.k_support, 1))
        return normalize_vectors(code[None, :], eps=self.config.epsilon_app)[0]

    def _build_query_cache(
        self,
        x: np.ndarray,
        n: np.ndarray,
        r: np.ndarray,
        omega: np.ndarray,
        e_id: np.ndarray,
        base_cache: np.ndarray | None = None,
    ) -> np.ndarray:
        summary = np.concatenate([x, n, r[:, None], omega[:, None], e_id], axis=1)
        if base_cache is not None:
            summary = np.concatenate([base_cache.astype(np.float32), summary.astype(np.float32)], axis=1)
        if summary.shape[1] >= self.config.query_cache_dim:
            return summary[:, : self.config.query_cache_dim].astype(np.float32)
        pad = np.zeros((summary.shape[0], self.config.query_cache_dim - summary.shape[1]), dtype=np.float32)
        return np.concatenate([summary.astype(np.float32), pad], axis=1)

    def _active_mask(self, omega: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        slots = np.arange(self.config.k_support, dtype=np.int32)
        valid_slots = slots[np.asarray(valid_mask, dtype=bool)]
        if valid_slots.size == 0:
            return np.zeros((self.config.k_support,), dtype=bool)
        above = valid_slots[omega[valid_slots] > self.config.tau_sup]
        if 0 < above.size <= self.config.k_active:
            mask = np.zeros((self.config.k_support,), dtype=bool)
            mask[above] = True
            return mask
        if above.size > self.config.k_active:
            ranked = np.argsort(-omega[above])[: self.config.k_active]
            mask = np.zeros((self.config.k_support,), dtype=bool)
            mask[above[ranked]] = True
            return mask
        chosen = valid_slots if valid_slots.size <= self.config.k_active else valid_slots[np.argsort(-omega[valid_slots])[: self.config.k_active]]
        mask = np.zeros((self.config.k_support,), dtype=bool)
        mask[chosen] = True
        return mask

    def _match_previous(
        self,
        *,
        previous: SupportScaffoldState | None,
        observation: PicfObservation,
        x: np.ndarray,
        n: np.ndarray,
        active_mask: np.ndarray,
        e_id: np.ndarray,
        meta: RuntimeMeta,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        pred_idx = np.full((self.config.k_support,), -1, dtype=np.int32)
        matched_mask = np.zeros((self.config.k_support,), dtype=bool)
        if previous is None:
            return pred_idx, matched_mask, 0.0, 0.0, 0.0
        prev_slots = np.arange(self.config.k_support, dtype=np.int32)
        prev_active_mask = previous.active_mask.copy()
        prev_global = transform_points(previous.x[prev_active_mask], previous.G_t)
        prev_normals_global = transform_normals(previous.n[prev_active_mask], previous.G_t)
        world_to_local = invert_transform(observation.G_t)
        prev_x_transport = transform_points(prev_global, world_to_local)
        prev_n_transport = transform_normals(prev_normals_global, world_to_local)
        result = match_supports(
            x,
            n,
            active_mask,
            e_id,
            prev_x_transport,
            prev_n_transport,
            prev_slots[prev_active_mask],
            previous.e_id[prev_active_mask],
            tau_p=self.config.tau_p_m,
            tau_n=self.config.tau_n,
            rgb_enabled=self.config.v_rgb_identity and meta.v_rgb_p and previous.runtime_meta.v_rgb_p,
            lambda_app_match=self.config.lambda_app_match_m,
            epsilon_app=self.config.epsilon_app,
        )
        return (
            result.pred_idx,
            result.matched_mask,
            result.match_ratio,
            result.reindex_failure_rate,
            result.normal_flip_ratio,
        )

    def _apply_continuity_refinement(
        self,
        *,
        previous: SupportScaffoldState | None,
        omega: np.ndarray,
        s_qry: np.ndarray,
        seed_from_birth: np.ndarray,
        valid_support: np.ndarray,
        seed_prev_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        refined_omega = omega.astype(np.float32, copy=True)
        if previous is None:
            return refined_omega, s_qry.astype(np.float32, copy=True)

        carry_slots = np.flatnonzero((seed_prev_idx >= 0) & ~seed_from_birth & valid_support)
        if carry_slots.size > 0:
            prev_omega = previous.omega[seed_prev_idx[carry_slots]]
            refined_omega[carry_slots] = np.clip(
                self.config.matched_omega_momentum * prev_omega
                + (1.0 - self.config.matched_omega_momentum) * refined_omega[carry_slots],
                0.0,
                1.0,
            )

        birth_slots = np.flatnonzero(seed_from_birth & valid_support)
        if birth_slots.size > 0:
            refined_omega[birth_slots] *= self.config.birth_omega_scale

        refined_cache = s_qry.astype(np.float32, copy=True)
        if carry_slots.size > 0:
            prev_cache = previous.s_qry[seed_prev_idx[carry_slots]]
            refined_cache[carry_slots] = (
                self.config.matched_cache_momentum * prev_cache
                + (1.0 - self.config.matched_cache_momentum) * refined_cache[carry_slots]
            ).astype(np.float32)
        return refined_omega, refined_cache

    def _transport_only(
        self,
        observation: PicfObservation,
        previous: SupportScaffoldState,
        meta: RuntimeMeta,
    ) -> SupportScaffoldState:
        meta.v_rgb_p = False
        prev_global = transform_points(previous.x, previous.G_t)
        prev_normals_global = transform_normals(previous.n, previous.G_t)
        world_to_local = invert_transform(observation.G_t)
        x = transform_points(prev_global, world_to_local)
        n = transform_normals(prev_normals_global, world_to_local)
        pred_idx = np.where(previous.active_mask, np.arange(self.config.k_support, dtype=np.int32), -1)
        hold_triggered = meta.stale_scaffold_steps >= self.config.n_hold_scaf
        debug = ScaffoldDebugMetrics(
            num_points_local=0 if observation.point_set is None else int(observation.point_set.num_points),
            num_active=int(previous.active_mask.sum()),
            num_birth=0,
            match_ratio=1.0,
            mean_radius=float(previous.r[previous.active_mask].mean()) if np.any(previous.active_mask) else 0.0,
            normal_fallback_ratio=0.0,
            empty_support_ratio=1.0,
            hold_triggered=hold_triggered,
            hold_reason="scaffold_stale_timeout" if hold_triggered else None,
            reindex_failure_rate=0.0,
            normal_flip_ratio=0.0,
            fresh_scaffold=False,
        )
        return SupportScaffoldState(
            pi_geom=np.zeros((self.config.k_support, 0), dtype=np.float32),
            x=x,
            n=n,
            r=previous.r.copy(),
            omega=previous.omega.copy(),
            active_mask=previous.active_mask.copy(),
            pred_idx=pred_idx,
            matched_mask=previous.active_mask.copy(),
            birth_mask=np.zeros((self.config.k_support,), dtype=bool),
            e_id=previous.e_id.copy(),
            s_qry=previous.s_qry.copy(),
            G_t=np.asarray(observation.G_t, dtype=np.float32),
            step_id=int(observation.step_id),
            segment_id=int(observation.segment_id),
            runtime_meta=meta,
            debug=debug,
        )

    def step(self, observation: PicfObservation, previous: SupportScaffoldState | None = None) -> SupportScaffoldState:
        if observation.G_t is None:
            observation.G_t = self.local_frame.make_transform(observation.robot_obs)
        if observation.point_set is None:
            observation.point_set = self.pointcloud_builder(
                {
                    "rgb_static": observation.rgb_static,
                    "depth_static": observation.depth_static,
                    "focus_center_world": observation.G_t[:3, 3],
                    "focus_radius_m": self.config.crop_radius_m + self.config.r_max_m,
                }
            )
        meta = self._build_runtime_meta(observation, previous.runtime_meta if previous is not None else None)
        if observation.reset_scaffold:
            previous = None
        if previous is None and not meta.v_pc_scaf:
            raise RuntimeError("Scaffold rollout must start from a fresh point cloud.")
        if previous is not None and not meta.v_pc_scaf:
            return self._transport_only(observation, previous, meta)

        frame_context = self._build_frame_context(observation)
        points_local = frame_context.points_local
        normals_local = frame_context.normals_local
        if previous is None and points_local.shape[0] == 0:
            raise RuntimeError("Fresh scaffold step has no local points inside crop.")
        if previous is not None and points_local.shape[0] == 0:
            meta.v_pc_scaf = False
            meta.v_rgb_p = False
            meta.stale_scaffold_steps = previous.runtime_meta.stale_scaffold_steps + 1
            return self._transport_only(observation, previous, meta)

        feature_context = frame_context
        if not meta.v_rgb_p:
            feature_context = dataclasses.replace(frame_context, colors=np.zeros_like(frame_context.colors, dtype=np.float32))
        point_features = feature_context.colors.astype(np.float32, copy=False)
        if self.point_feature_extractor is not None and points_local.shape[0] > 0:
            point_features = self.point_feature_extractor.encode_local_context(feature_context).features

        seed_x, seed_n, seed_r, seed_eid, seed_qry, seed_from_birth, seed_valid, seed_prev_idx = self._prepare_seeds(
            observation, points_local, normals_local, point_features, previous
        )
        pi, x, n, r, omega, e_id, s_qry, fallback_ratio, empty_ratio = self._group_points(
            points_local,
            normals_local,
            point_features,
            seed_x,
            seed_n,
            seed_r,
            seed_eid,
            seed_qry,
            seed_valid,
            meta,
        )
        valid_support = np.sum(pi, axis=1) > 0
        provisional_active = self._active_mask(omega, valid_support)
        _, _, _, _, _ = self._match_previous(
            previous=previous,
            observation=observation,
            x=x,
            n=n,
            active_mask=provisional_active,
            e_id=e_id,
            meta=meta,
        )
        omega, s_qry = self._apply_continuity_refinement(
            previous=previous,
            omega=omega,
            s_qry=s_qry,
            seed_from_birth=seed_from_birth,
            valid_support=valid_support,
            seed_prev_idx=seed_prev_idx,
        )
        active_mask = self._active_mask(omega, valid_support)
        pred_idx, matched_mask, match_ratio, reindex_failure_rate, normal_flip_ratio = self._match_previous(
            previous=previous,
            observation=observation,
            x=x,
            n=n,
            active_mask=active_mask,
            e_id=e_id,
            meta=meta,
        )
        if previous is not None and np.any(matched_mask):
            matched_slots = np.flatnonzero(matched_mask)
            prev_cache = previous.s_qry[pred_idx[matched_slots]]
            s_qry[matched_slots] = (
                self.config.matched_cache_momentum * prev_cache
                + (1.0 - self.config.matched_cache_momentum) * s_qry[matched_slots]
            ).astype(np.float32)
        birth_mask = active_mask & ~matched_mask & seed_from_birth
        hold_triggered = meta.stale_scaffold_steps >= self.config.n_hold_scaf
        debug = ScaffoldDebugMetrics(
            num_points_local=int(points_local.shape[0]),
            num_active=int(active_mask.sum()),
            num_birth=int(birth_mask.sum()),
            match_ratio=float(match_ratio),
            mean_radius=float(r[active_mask].mean()) if np.any(active_mask) else 0.0,
            normal_fallback_ratio=float(fallback_ratio),
            empty_support_ratio=float(empty_ratio),
            hold_triggered=hold_triggered,
            hold_reason="scaffold_stale_timeout" if hold_triggered else None,
            reindex_failure_rate=float(reindex_failure_rate),
            normal_flip_ratio=float(normal_flip_ratio),
            fresh_scaffold=True,
        )
        return SupportScaffoldState(
            pi_geom=pi,
            x=x,
            n=n,
            r=r,
            omega=omega,
            active_mask=active_mask,
            pred_idx=pred_idx,
            matched_mask=matched_mask,
            birth_mask=birth_mask,
            e_id=e_id,
            s_qry=s_qry,
            G_t=np.asarray(observation.G_t, dtype=np.float32),
            step_id=int(observation.step_id),
            segment_id=int(observation.segment_id),
            runtime_meta=meta,
            debug=debug,
        )
