from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from openpi.picf.contracts import PicfPointCloudFrame
from openpi.picf.geometry import normalize_vectors
from openpi.picf.geometry import transform_normals
from openpi.picf.geometry import transform_points
from openpi.transforms.calvin_depth_to_sonata_pointcloud import _as_3x3
from openpi.transforms.calvin_depth_to_sonata_pointcloud import _as_4x4
from openpi.transforms.calvin_depth_to_sonata_pointcloud import _load_json


def _finite_difference_normals(points_cam: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros_like(points_cam, dtype=np.float32)
    usable = np.zeros(valid_mask.shape, dtype=bool)
    if points_cam.ndim != 3 or points_cam.shape[-1] != 3:
        raise ValueError(f"points_cam must be [H,W,3], got {points_cam.shape}")
    if points_cam.shape[0] < 3 or points_cam.shape[1] < 3:
        return normals, usable

    dx = points_cam[1:-1, 2:, :] - points_cam[1:-1, :-2, :]
    dy = points_cam[2:, 1:-1, :] - points_cam[:-2, 1:-1, :]
    central_valid = (
        valid_mask[1:-1, 1:-1]
        & valid_mask[1:-1, 2:]
        & valid_mask[1:-1, :-2]
        & valid_mask[2:, 1:-1]
        & valid_mask[:-2, 1:-1]
    )
    raw = np.cross(dx, dy)
    raw = normalize_vectors(raw)
    normals[1:-1, 1:-1, :] = raw
    usable[1:-1, 1:-1] = central_valid
    return normals, usable


def _deterministic_fps(points: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if points.shape[0] <= count:
        return np.arange(points.shape[0], dtype=np.int64)
    centroid = points.mean(axis=0, dtype=np.float32)
    dists = np.linalg.norm(points - centroid[None, :], axis=1)
    first = int(np.argmax(dists))
    chosen = [first]
    min_dist = np.linalg.norm(points - points[first : first + 1], axis=1)
    while len(chosen) < count:
        next_idx = int(np.argmax(min_dist))
        chosen.append(next_idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[next_idx : next_idx + 1], axis=1))
    return np.asarray(chosen, dtype=np.int64)


class CalvinDepthToPicfPointCloud:
    """Deterministic CALVIN depth->pointcloud builder with normals for scaffold replay."""

    def __init__(
        self,
        cameras_json_path: str,
        *,
        cam_name: str = "static",
        stride: int = 2,
        max_points: int = 4096,
        voxel_size: float = 0.005,
        z_min: float = 0.1,
        z_max: float = 10.0,
        use_world: bool = True,
        depth_scale: float = 1.0,
        selection_mode: str = "fps",
    ):
        cams = _load_json(cameras_json_path)
        cam_table = cams.get("cameras", cams)
        if cam_name not in cam_table:
            raise KeyError(f"Camera '{cam_name}' not found. Available: {list(cam_table.keys())}")
        cam = cam_table[cam_name]
        if "K" in cam:
            self.K = _as_3x3(cam["K"])
        elif "intrinsics" in cam:
            self.K = _as_3x3(cam["intrinsics"])
        else:
            raise KeyError(f"Camera '{cam_name}' missing intrinsics. Keys={list(cam.keys())}")
        if "W_T_C" in cam:
            self.W_T_C = _as_4x4(cam["W_T_C"])
        elif "viewMatrix" in cam:
            self.W_T_C = np.linalg.inv(_as_4x4(cam["viewMatrix"])).astype(np.float32)
        else:
            raise KeyError(f"Camera '{cam_name}' missing extrinsics. Keys={list(cam.keys())}")
        self._fx = float(self.K[0, 0])
        self._fy = float(self.K[1, 1])
        self._cx = float(self.K[0, 2])
        self._cy = float(self.K[1, 2])
        self.stride = int(stride)
        self.max_points = int(max_points)
        self.voxel_size = float(voxel_size)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.use_world = bool(use_world)
        self.depth_scale = float(depth_scale)
        self.selection_mode = str(selection_mode)
        if self.selection_mode not in {"fps", "linspace"}:
            raise ValueError(f"selection_mode must be 'fps' or 'linspace', got {self.selection_mode!r}")

    def _select_indices(self, xyz: np.ndarray) -> np.ndarray:
        if xyz.shape[0] <= self.max_points:
            return np.arange(xyz.shape[0], dtype=np.int64)
        if self.selection_mode == "linspace":
            return np.linspace(0, xyz.shape[0] - 1, self.max_points, dtype=np.int64)
        return _deterministic_fps(xyz, self.max_points)

    def __call__(self, sample: Mapping[str, np.ndarray]) -> PicfPointCloudFrame:
        rgb = np.asarray(sample["rgb_static"])
        depth = np.asarray(sample["depth_static"], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"depth_static must be 2D, got {depth.shape}")
        depth = depth * self.depth_scale
        height, width = depth.shape
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        valid = np.isfinite(depth) & (depth > self.z_min) & (depth < self.z_max)
        if not np.any(valid):
            return PicfPointCloudFrame(
                grid_coord=np.zeros((0, 3), dtype=np.int32),
                xyz_world=np.zeros((0, 3), dtype=np.float32),
                rgb=np.zeros((0, 3), dtype=np.float32),
                normal_world=np.zeros((0, 3), dtype=np.float32),
                valid_point_mask=np.zeros((0,), dtype=bool),
                frame_valid=False,
            )

        x = (uu - self._cx) / self._fx * depth
        y = (vv - self._cy) / self._fy * depth
        points_cam = np.stack([x, y, depth], axis=-1).astype(np.float32)
        normals_cam, normal_valid = _finite_difference_normals(points_cam, valid)

        step_mask = np.zeros_like(valid, dtype=bool)
        step_mask[:: self.stride, :: self.stride] = True
        sampled_mask = valid & step_mask
        sampled_indices = np.argwhere(sampled_mask)
        if sampled_indices.size == 0:
            return PicfPointCloudFrame(
                grid_coord=np.zeros((0, 3), dtype=np.int32),
                xyz_world=np.zeros((0, 3), dtype=np.float32),
                rgb=np.zeros((0, 3), dtype=np.float32),
                normal_world=np.zeros((0, 3), dtype=np.float32),
                valid_point_mask=np.zeros((0,), dtype=bool),
                frame_valid=False,
            )

        xyz_cam = points_cam[sampled_mask]
        rgb_sel = rgb[sampled_mask].astype(np.float32) / 255.0
        normals_sel = normals_cam[sampled_mask]
        missing_normals = ~normal_valid[sampled_mask]
        if np.any(missing_normals):
            fallback = normalize_vectors(-xyz_cam[missing_normals])
            normals_sel = normals_sel.copy()
            normals_sel[missing_normals] = fallback

        if self.use_world:
            xyz = transform_points(xyz_cam, self.W_T_C)
            normals = transform_normals(normals_sel, self.W_T_C)
        else:
            xyz = xyz_cam.astype(np.float32)
            normals = normalize_vectors(normals_sel)

        choose = self._select_indices(xyz)
        xyz = xyz[choose]
        rgb_sel = rgb_sel[choose]
        normals = normals[choose]
        offset = xyz.min(axis=0, keepdims=True)
        grid = np.floor((xyz - offset) / self.voxel_size).astype(np.int32)
        grid -= grid.min(axis=0, keepdims=True)

        return PicfPointCloudFrame(
            grid_coord=grid,
            xyz_world=xyz,
            rgb=rgb_sel,
            normal_world=normals,
            valid_point_mask=np.ones((xyz.shape[0],), dtype=bool),
            frame_valid=True,
        )
