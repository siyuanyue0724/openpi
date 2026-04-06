from __future__ import annotations

import dataclasses

import numpy as np

from openpi.picf.camera_io import as_3x3
from openpi.picf.camera_io import as_4x4
from openpi.picf.camera_io import load_json
from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import SupportScaffoldState
from openpi.picf.frame_context import PointFrameContext
from openpi.picf.geometry import invert_transform
from openpi.picf.geometry import normalize_vectors
from openpi.picf.geometry import transform_points
from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.contracts import VisualExpertState
from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa.wrapper import VjepaFeatureMap


@dataclasses.dataclass(frozen=True)
class CameraModel:
    K: np.ndarray
    W_T_C: np.ndarray
    C_T_W: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float


def load_camera_model(camera_json_path: str, *, camera_name: str = "static") -> CameraModel:
    cameras = load_json(camera_json_path)
    camera_table = cameras.get("cameras", cameras)
    if camera_name not in camera_table:
        raise KeyError(f"Camera '{camera_name}' not found. Available: {list(camera_table.keys())}")
    camera = camera_table[camera_name]
    if "K" in camera:
        intrinsics = as_3x3(camera["K"])
    elif "intrinsics" in camera:
        intrinsics = as_3x3(camera["intrinsics"])
    else:
        raise KeyError(f"Camera '{camera_name}' missing intrinsics. Keys={list(camera.keys())}")
    if "W_T_C" in camera:
        world_from_camera = as_4x4(camera["W_T_C"])
    elif "viewMatrix" in camera:
        world_from_camera = np.linalg.inv(as_4x4(camera["viewMatrix"])).astype(np.float32)
    else:
        raise KeyError(f"Camera '{camera_name}' missing extrinsics. Keys={list(camera.keys())}")
    return CameraModel(
        K=intrinsics,
        W_T_C=world_from_camera,
        C_T_W=invert_transform(world_from_camera),
        fx=float(intrinsics[0, 0]),
        fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]),
        cy=float(intrinsics[1, 2]),
    )


def _pad_block(features: np.ndarray, dim: int) -> np.ndarray:
    if features.shape[0] >= dim:
        return features[:dim].astype(np.float32)
    out = np.zeros((dim,), dtype=np.float32)
    out[: features.shape[0]] = features.astype(np.float32)
    return out


def empty_visual_expert(*, posterior_config: PosteriorConfig, k_support: int) -> VisualExpertState:
    dim_total = posterior_config.dim_total
    mu = np.zeros((k_support, dim_total), dtype=np.float32)
    var_block = np.tile(
        np.array(
            [posterior_config.visual_var_h, posterior_config.visual_var_g, posterior_config.visual_var_c],
            dtype=np.float32,
        )[None, :],
        (k_support, 1),
    )
    gate = np.zeros((k_support,), dtype=bool)
    in_view = np.zeros((k_support,), dtype=bool)
    visibility = np.zeros((k_support,), dtype=np.float32)
    depth_residual = np.zeros((k_support,), dtype=np.float32)
    depth_available = np.zeros((k_support,), dtype=bool)
    return VisualExpertState(mu, var_block, gate, in_view, visibility, depth_residual, depth_available)


def _prepare_depth_image(depth_static: np.ndarray) -> np.ndarray | None:
    depth = np.asarray(depth_static, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.size == 0:
        return None
    return depth


def _project_world_points(
    world_points: np.ndarray,
    *,
    camera_model: CameraModel,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if world_points.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    points_cam = transform_points(world_points, camera_model.C_T_W)
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((world_points.shape[0], 2), dtype=np.float32)
    uv[:, 0] = camera_model.fx * points_cam[:, 0] / np.maximum(z, 1e-6) + camera_model.cx
    uv[:, 1] = camera_model.fy * points_cam[:, 1] / np.maximum(z, 1e-6) + camera_model.cy
    valid &= uv[:, 0] >= 0.0
    valid &= uv[:, 0] <= float(image_width - 1)
    valid &= uv[:, 1] >= 0.0
    valid &= uv[:, 1] <= float(image_height - 1)
    return uv, z.astype(np.float32), valid


def _scale_to_grid(uv: np.ndarray, *, source_hw: tuple[int, int], grid_hw: tuple[int, int]) -> np.ndarray:
    source_h, source_w = source_hw
    grid_h, grid_w = grid_hw
    scaled = np.zeros_like(uv, dtype=np.float32)
    scaled[:, 0] = uv[:, 0] * (grid_w - 1) / max(source_w - 1, 1)
    scaled[:, 1] = uv[:, 1] * (grid_h - 1) / max(source_h - 1, 1)
    return scaled


def _bilinear_sample_map(feature_map: np.ndarray, uv: np.ndarray, *, source_hw: tuple[int, int]) -> np.ndarray:
    if uv.shape[0] == 0:
        return np.zeros((0, feature_map.shape[-1]), dtype=np.float32)
    grid_uv = _scale_to_grid(uv, source_hw=source_hw, grid_hw=feature_map.shape[:2])
    x = grid_uv[:, 0]
    y = grid_uv[:, 1]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, feature_map.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, feature_map.shape[0] - 1)
    x0 = np.clip(x0, 0, feature_map.shape[1] - 1)
    y0 = np.clip(y0, 0, feature_map.shape[0] - 1)
    dx = x - x0
    dy = y - y0
    wa = (1.0 - dx) * (1.0 - dy)
    wb = (1.0 - dx) * dy
    wc = dx * (1.0 - dy)
    wd = dx * dy
    return (
        feature_map[y0, x0] * wa[:, None]
        + feature_map[y1, x0] * wb[:, None]
        + feature_map[y0, x1] * wc[:, None]
        + feature_map[y1, x1] * wd[:, None]
    ).astype(np.float32)


def _bilinear_sample_scalar(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    if uv.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    x = uv[:, 0]
    y = uv[:, 1]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, image.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, image.shape[0] - 1)
    x0 = np.clip(x0, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    dx = x - x0
    dy = y - y0
    wa = (1.0 - dx) * (1.0 - dy)
    wb = (1.0 - dx) * dy
    wc = dx * (1.0 - dy)
    wd = dx * dy
    return (
        image[y0, x0] * wa
        + image[y1, x0] * wb
        + image[y0, x1] * wc
        + image[y1, x1] * wd
    ).astype(np.float32)


def _patch_pool(
    feature_map: np.ndarray,
    uv: np.ndarray,
    *,
    source_hw: tuple[int, int],
    radius: int,
) -> np.ndarray:
    scaled = _scale_to_grid(uv[None, :], source_hw=source_hw, grid_hw=feature_map.shape[:2])[0]
    x = int(np.rint(scaled[0]))
    y = int(np.rint(scaled[1]))
    x0 = max(x - radius, 0)
    x1 = min(x + radius + 1, feature_map.shape[1])
    y0 = max(y - radius, 0)
    y1 = min(y + radius + 1, feature_map.shape[0])
    patch = feature_map[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((feature_map.shape[-1],), dtype=np.float32)
    return patch.mean(axis=(0, 1), dtype=np.float32)


def _normalize_feature(feature: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    return normalize_vectors(feature[None, :], eps=eps)[0]


def build_visual_expert(
    *,
    posterior_config: PosteriorConfig,
    visual_config: VjepaVisualConfig,
    observation: PicfObservation,
    scaffold_state: SupportScaffoldState,
    visual_features: VjepaFeatureMap,
    camera_model: CameraModel,
    frame_context: PointFrameContext | None,
) -> VisualExpertState:
    k_support = scaffold_state.x.shape[0]
    state = empty_visual_expert(posterior_config=posterior_config, k_support=k_support)
    runtime_meta = scaffold_state.runtime_meta
    if (observation.timestamp_s - runtime_meta.t_v_last) > visual_config.delta_v_max_s:
        return state

    current_map = visual_features.current_map(use_last_two_mean=visual_config.use_last_two_mean)
    source_hw = visual_features.source_hw
    depth_image = _prepare_depth_image(observation.depth_static)
    fresh_scaffold = bool(scaffold_state.debug.fresh_scaffold)

    if fresh_scaffold:
        if frame_context is None:
            raise ValueError("Fresh visual expert requires a point frame context.")
        if scaffold_state.pi_geom.shape[1] != frame_context.points_local.shape[0]:
            raise ValueError(
                "Support pi_geom / local-point mismatch: "
                f"{scaffold_state.pi_geom.shape[1]} vs {frame_context.points_local.shape[0]}"
            )
        world_points = transform_points(frame_context.points_local, scaffold_state.G_t)
        point_uv, point_depth, point_valid = _project_world_points(
            world_points,
            camera_model=camera_model,
            image_height=source_hw[0],
            image_width=source_hw[1],
        )
        point_features = np.zeros((world_points.shape[0], current_map.shape[-1]), dtype=np.float32)
        if np.any(point_valid):
            point_features[point_valid] = _bilinear_sample_map(
                current_map,
                point_uv[point_valid],
                source_hw=source_hw,
            )
        point_depth_samples = None
        point_depth_valid = np.zeros_like(point_valid, dtype=bool)
        if depth_image is not None and point_uv.shape[0] > 0:
            point_depth_samples = _bilinear_sample_scalar(depth_image, point_uv)
            point_depth_valid = point_valid & np.isfinite(point_depth_samples)
    else:
        point_valid = np.zeros((0,), dtype=bool)
        point_features = np.zeros((0, current_map.shape[-1]), dtype=np.float32)
        point_depth = np.zeros((0,), dtype=np.float32)
        point_depth_samples = None
        point_depth_valid = np.zeros((0,), dtype=bool)

    for slot in range(k_support):
        support_world = transform_points(scaffold_state.x[slot : slot + 1], scaffold_state.G_t)
        center_uv, _, center_valid = _project_world_points(
            support_world,
            camera_model=camera_model,
            image_height=source_hw[0],
            image_width=source_hw[1],
        )
        center_in_view = bool(center_valid[0]) if center_valid.size > 0 else False
        state.in_view[slot] = center_in_view
        patch_feature = (
            _patch_pool(
                current_map,
                center_uv[0],
                source_hw=source_hw,
                radius=visual_config.patch_pool_radius,
            )
            if center_in_view
            else np.zeros((current_map.shape[-1],), dtype=np.float32)
        )

        if fresh_scaffold and scaffold_state.pi_geom.shape[1] > 0:
            weights = scaffold_state.pi_geom[slot]
            visibility = float(np.sum(weights * point_valid.astype(np.float32)))
            state.visibility[slot] = visibility
            visible_mass = max(visibility, visual_config.epsilon_vis)
            if visibility > visual_config.epsilon_vis:
                pooled_feature = np.sum(weights[:, None] * point_features, axis=0) / visible_mass
            else:
                pooled_feature = np.zeros((current_map.shape[-1],), dtype=np.float32)
            depth_mass = float(np.sum(weights * point_depth_valid.astype(np.float32)))
            if depth_image is not None and point_depth_samples is not None and depth_mass > visual_config.epsilon_vis:
                residual = np.sum(weights * point_depth_valid.astype(np.float32) * np.abs(point_depth - point_depth_samples))
                state.depth_residual[slot] = float(residual / max(depth_mass, visual_config.epsilon_vis))
                state.depth_available[slot] = True
            appearance_terms = []
            if visibility > visual_config.epsilon_vis:
                appearance_terms.append(_normalize_feature(pooled_feature))
            if center_in_view:
                appearance_terms.append(_normalize_feature(patch_feature))
            appearance = (
                np.mean(np.stack(appearance_terms, axis=0), axis=0)
                if appearance_terms
                else np.zeros((current_map.shape[-1],), dtype=np.float32)
            )
            if state.depth_available[slot]:
                gate = center_in_view and visibility > visual_config.epsilon_vis and (
                    state.depth_residual[slot] < visual_config.tau_z_m
                )
            else:
                gate = center_in_view and visibility > visual_config.tau_vis
        else:
            state.visibility[slot] = 1.0 if center_in_view else 0.0
            appearance = _normalize_feature(patch_feature) if center_in_view else patch_feature
            gate = center_in_view

        geom_summary = np.concatenate(
            [
                scaffold_state.x[slot],
                scaffold_state.n[slot],
                np.array(
                    [
                        scaffold_state.r[slot],
                        state.visibility[slot],
                        state.depth_residual[slot],
                        float(state.depth_available[slot]),
                        float(center_in_view),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        state.mu[slot, : posterior_config.dim_h] = _pad_block(appearance, posterior_config.dim_h)
        state.mu[slot, posterior_config.dim_h : posterior_config.dim_h + posterior_config.dim_g] = _pad_block(
            geom_summary,
            posterior_config.dim_g,
        )
        state.gate[slot] = bool(gate)
        if not fresh_scaffold:
            state.var_block[slot, 0] *= 2.0
            state.var_block[slot, 1] *= 2.0

    return state
