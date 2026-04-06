from __future__ import annotations

import dataclasses

import numpy as np

from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import RuntimeMeta
from openpi.picf.contracts import ScaffoldDebugMetrics
from openpi.picf.contracts import SupportScaffoldState
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


@dataclasses.dataclass(frozen=True)
class DeterministicScaffoldConfig:
    k_support: int = 96
    k_birth: int = 12
    k_active: int = 96
    crop_radius_m: float = 0.08
    r_min_m: float = 0.003
    r_max_m: float = 0.02
    tau_sup: float = 0.25
    target_points_per_support: int = 24
    seed_init_radius_m: float = 0.01
    grouping_neighbors: int = 32
    tau_p_m: float = 0.005
    tau_n: float = 0.8
    epsilon_n: float = 1e-6
    epsilon_app: float = 1e-6
    lambda_app_match_m: float = 1e-3
    delta_sync_s: float = 0.02
    tau_rgb_proj_m: float = 0.005
    delta_pc_scaf_s: float = 0.15
    n_hold_scaf: int = 2
    v_rgb_identity: bool = False
    query_cache_dim: int = 16
    n_min_anchors: int = 8
    birth_weight_threshold: float = 0.2


class DeterministicScaffoldPipeline:
    def __init__(
        self,
        pointcloud_builder: CalvinDepthToPicfPointCloud,
        *,
        config: DeterministicScaffoldConfig | None = None,
        local_frame: EndEffectorLocalFrame | None = None,
    ):
        self.pointcloud_builder = pointcloud_builder
        self.config = config or DeterministicScaffoldConfig()
        self.local_frame = local_frame or EndEffectorLocalFrame()

    def _build_runtime_meta(self, observation: PicfObservation, previous: RuntimeMeta | None) -> RuntimeMeta:
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

    def _select_local_points(self, observation: PicfObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        context = build_point_frame_context(
            observation,
            crop_radius_m=self.config.crop_radius_m,
            local_frame=self.local_frame,
        )
        return context.points_local, context.normals_local, context.colors

    def _seed_from_points(
        self,
        points_local: np.ndarray,
        normals_local: np.ndarray,
        colors: np.ndarray,
        seed_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed_x = np.zeros((self.config.k_support, 3), dtype=np.float32)
        seed_n = np.zeros((self.config.k_support, 3), dtype=np.float32)
        seed_r = np.full((self.config.k_support,), self.config.seed_init_radius_m, dtype=np.float32)
        seed_eid = np.zeros((self.config.k_support, 3), dtype=np.float32)
        for slot, point_idx in enumerate(seed_indices.tolist()):
            seed_x[slot] = points_local[point_idx]
            seed_n[slot] = normals_local[point_idx]
            seed_eid[slot] = colors[point_idx]
        return seed_x, seed_n, seed_r, seed_eid

    def _prepare_seeds(
        self,
        observation: PicfObservation,
        points_local: np.ndarray,
        normals_local: np.ndarray,
        colors: np.ndarray,
        previous: SupportScaffoldState | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed_x = np.zeros((self.config.k_support, 3), dtype=np.float32)
        seed_n = np.zeros((self.config.k_support, 3), dtype=np.float32)
        seed_r = np.full((self.config.k_support,), self.config.seed_init_radius_m, dtype=np.float32)
        seed_eid = np.zeros((self.config.k_support, 3), dtype=np.float32)
        seed_from_birth = np.zeros((self.config.k_support,), dtype=bool)
        seed_valid = np.zeros((self.config.k_support,), dtype=bool)
        next_slot = 0
        carried_centers = np.zeros((0, 3), dtype=np.float32)

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
                carried_count = min(int(carry_slots.size), self.config.k_support)
                seed_x[:carried_count] = carried_centers[:carried_count]
                seed_n[:carried_count] = carried_normals[:carried_count]
                seed_r[:carried_count] = previous.r[carry_slots[:carried_count]]
                seed_eid[:carried_count] = previous.e_id[carry_slots[:carried_count], :3]
                seed_valid[:carried_count] = True
                next_slot = carried_count

        if points_local.shape[0] == 0 or next_slot >= self.config.k_support:
            return seed_x, seed_n, seed_r, seed_eid, seed_from_birth, seed_valid

        remaining = self.config.k_support if previous is None or observation.reset_scaffold else min(
            self.config.k_birth, self.config.k_support - next_slot
        )
        if remaining <= 0:
            return seed_x, seed_n, seed_r, seed_eid, seed_from_birth, seed_valid

        weights = coverage_weights(points_local, carried_centers, sigma=max(self.config.seed_init_radius_m, self.config.r_min_m))
        if previous is not None and not observation.reset_scaffold:
            candidate_indices = np.flatnonzero(weights > self.config.birth_weight_threshold)
            if candidate_indices.size == 0:
                return seed_x, seed_n, seed_r, seed_eid, seed_from_birth, seed_valid
            birth_count = min(int(candidate_indices.size), int(remaining))
            filtered = weighted_fps(points_local[candidate_indices], weights[candidate_indices], birth_count)
            birth_indices = candidate_indices[filtered]
        else:
            birth_indices = weighted_fps(points_local, weights, remaining)
        bx, bn, br, be = self._seed_from_points(points_local, normals_local, colors, birth_indices)
        end = next_slot + birth_indices.shape[0]
        seed_x[next_slot:end] = bx[: birth_indices.shape[0]]
        seed_n[next_slot:end] = bn[: birth_indices.shape[0]]
        seed_r[next_slot:end] = br[: birth_indices.shape[0]]
        seed_eid[next_slot:end] = be[: birth_indices.shape[0]]
        seed_from_birth[next_slot:end] = True
        seed_valid[next_slot:end] = True
        return seed_x, seed_n, seed_r, seed_eid, seed_from_birth, seed_valid

    def _group_points(
        self,
        points_local: np.ndarray,
        normals_local: np.ndarray,
        colors: np.ndarray,
        seed_x: np.ndarray,
        seed_n: np.ndarray,
        seed_r: np.ndarray,
        seed_valid: np.ndarray,
        meta: RuntimeMeta,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_points = points_local.shape[0]
        pi = np.zeros((self.config.k_support, n_points), dtype=np.float32)
        x = np.zeros((self.config.k_support, 3), dtype=np.float32)
        n = np.zeros((self.config.k_support, 3), dtype=np.float32)
        r = np.zeros((self.config.k_support,), dtype=np.float32)
        omega = np.zeros((self.config.k_support,), dtype=np.float32)
        e_id = np.zeros((self.config.k_support, 3), dtype=np.float32)
        normal_fallbacks = 0

        for slot in range(self.config.k_support):
            if not bool(seed_valid[slot]):
                continue
            center = seed_x[slot]
            dists = np.linalg.norm(points_local - center[None, :], axis=1)
            radius_limit = max(2.0 * float(np.clip(seed_r[slot], self.config.r_min_m, self.config.r_max_m)), self.config.r_min_m)
            neighbor_mask = dists <= radius_limit
            neighbors = np.flatnonzero(neighbor_mask)
            if neighbors.size > self.config.grouping_neighbors:
                ranked = np.argsort(dists[neighbors])[: self.config.grouping_neighbors]
                neighbors = neighbors[ranked]
            if neighbors.size == 0:
                seed_valid[slot] = False
                continue
            sigma = float(np.clip(seed_r[slot], self.config.r_min_m, self.config.r_max_m))
            weights = np.exp(-(dists[neighbors] ** 2) / max(2.0 * sigma * sigma, 1e-8))
            if np.linalg.norm(seed_n[slot]) > 0:
                align = np.clip((normals_local[neighbors] @ normalize_vectors(seed_n[slot : slot + 1])[0] + 1.0) / 2.0, 0.05, 1.0)
                weights *= align
            if float(weights.sum()) <= 0:
                weights = np.ones((neighbors.size,), dtype=np.float32)
            weights = weights.astype(np.float32)
            weights /= weights.sum()
            pi[slot, neighbors] = weights
            x[slot] = (weights[:, None] * points_local[neighbors]).sum(axis=0)
            pooled_norm = (weights[:, None] * normals_local[neighbors]).sum(axis=0)
            norm = float(np.linalg.norm(pooled_norm))
            if norm < self.config.epsilon_n:
                fallback = seed_n[slot]
                if np.linalg.norm(fallback) < self.config.epsilon_n:
                    fallback = normals_local[neighbors[0]]
                n[slot] = normalize_vectors(fallback[None, :])[0]
                normal_fallbacks += 1
            else:
                n[slot] = pooled_norm / norm
            variance = (weights[:, None] * ((points_local[neighbors] - x[slot]) ** 2)).sum(axis=0)
            r[slot] = float(np.clip(np.sqrt(float(np.mean(variance))), self.config.r_min_m, self.config.r_max_m))
            near_ratio = float(np.exp(-dists[neighbors].mean() / max(sigma, self.config.r_min_m)))
            density_ratio = float(np.clip(neighbors.size / max(self.config.target_points_per_support, 1), 0.0, 1.0))
            if neighbors.size < self.config.n_min_anchors:
                density_ratio *= neighbors.size / max(self.config.n_min_anchors, 1)
            omega[slot] = float(np.clip(near_ratio * density_ratio, 0.0, 1.0))
            if meta.v_rgb_p:
                desc = (weights[:, None] * colors[neighbors]).sum(axis=0)
                e_id[slot] = normalize_vectors(desc[None, :], eps=self.config.epsilon_app)[0]

        empty_ratio = float(np.mean(np.sum(pi, axis=1) <= 0))
        fallback_ratio = float(normal_fallbacks / max(np.count_nonzero(np.sum(pi, axis=1) > 0), 1))
        return pi, x, n, r, omega, e_id, fallback_ratio, empty_ratio

    def _build_query_cache(self, x: np.ndarray, n: np.ndarray, r: np.ndarray, omega: np.ndarray, e_id: np.ndarray) -> np.ndarray:
        summary = np.concatenate([x, n, r[:, None], omega[:, None], e_id], axis=1)
        if summary.shape[1] >= self.config.query_cache_dim:
            return summary[:, : self.config.query_cache_dim]
        pad = np.zeros((summary.shape[0], self.config.query_cache_dim - summary.shape[1]), dtype=np.float32)
        return np.concatenate([summary, pad], axis=1)

    def _active_mask(self, omega: np.ndarray) -> np.ndarray:
        slots = np.arange(self.config.k_support, dtype=np.int32)
        above = slots[omega > self.config.tau_sup]
        if 0 < above.size <= self.config.k_active:
            mask = np.zeros((self.config.k_support,), dtype=bool)
            mask[above] = True
            return mask
        if above.size > self.config.k_active:
            ranked = np.argsort(-omega[above])[: self.config.k_active]
            mask = np.zeros((self.config.k_support,), dtype=bool)
            mask[above[ranked]] = True
            return mask
        positive = slots[omega > 0]
        if positive.size > 0:
            chosen = positive if positive.size <= self.config.k_active else positive[np.argsort(-omega[positive])[: self.config.k_active]]
            mask = np.zeros((self.config.k_support,), dtype=bool)
            mask[chosen] = True
            return mask
        return np.zeros((self.config.k_support,), dtype=bool)

    def _transport_only(
        self,
        observation: PicfObservation,
        previous: SupportScaffoldState,
        meta: RuntimeMeta,
    ) -> SupportScaffoldState:
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
        if observation.point_set is None:
            observation.point_set = self.pointcloud_builder(
                {
                    "rgb_static": observation.rgb_static,
                    "depth_static": observation.depth_static,
                }
            )
        if observation.G_t is None:
            observation.G_t = self.local_frame.make_transform(observation.robot_obs)
        meta = self._build_runtime_meta(observation, previous.runtime_meta if previous is not None else None)
        if observation.reset_scaffold:
            previous = None
        if previous is None and not meta.v_pc_scaf:
            raise RuntimeError("Scaffold rollout must start from a fresh point cloud.")
        if previous is not None and not meta.v_pc_scaf:
            return self._transport_only(observation, previous, meta)

        points_local, normals_local, colors = self._select_local_points(observation)
        if previous is None and points_local.shape[0] == 0:
            raise RuntimeError("Fresh scaffold step has no local points inside crop.")
        if previous is not None and points_local.shape[0] == 0:
            meta.v_pc_scaf = False
            meta.stale_scaffold_steps = previous.runtime_meta.stale_scaffold_steps + 1
            return self._transport_only(observation, previous, meta)

        seed_x, seed_n, seed_r, _, seed_from_birth, seed_valid = self._prepare_seeds(
            observation, points_local, normals_local, colors, previous
        )
        pi, x, n, r, omega, e_id, fallback_ratio, empty_ratio = self._group_points(
            points_local, normals_local, colors, seed_x, seed_n, seed_r, seed_valid, meta
        )
        active_mask = self._active_mask(omega)
        pred_idx = np.full((self.config.k_support,), -1, dtype=np.int32)
        matched_mask = np.zeros((self.config.k_support,), dtype=bool)
        match_ratio = 0.0
        reindex_failure_rate = 0.0
        normal_flip_ratio = 0.0
        if previous is not None:
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
            pred_idx = result.pred_idx
            matched_mask = result.matched_mask
            match_ratio = result.match_ratio
            reindex_failure_rate = result.reindex_failure_rate
            normal_flip_ratio = result.normal_flip_ratio
        birth_mask = active_mask & ~matched_mask & seed_from_birth
        s_qry = self._build_query_cache(x, n, r, omega, e_id)
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
