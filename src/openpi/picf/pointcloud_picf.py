from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from openpi.picf.camera_io import as_3x3
from openpi.picf.camera_io import as_4x4
from openpi.picf.camera_io import load_json
from openpi.picf.contracts import PicfPointCloudFrame
from openpi.picf.geometry import normalize_vectors
from openpi.picf.geometry import transform_normals
from openpi.picf.geometry import transform_points
from openpi.picf.scaffold.local_frame import EndEffectorLocalFrame


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


def _deterministic_weighted_fps(points: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if points.shape[0] <= count:
        return np.arange(points.shape[0], dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    weights = np.maximum(weights, 0.0)
    chosen: list[int] = []
    start = int(np.argmax(weights))
    chosen.append(start)
    min_dist = np.linalg.norm(points - points[start : start + 1], axis=1)
    while len(chosen) < count:
        score = min_dist * (0.5 + weights)
        score[np.asarray(chosen, dtype=np.int64)] = -1.0
        next_idx = int(np.argmax(score))
        if score[next_idx] < 0:
            break
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
        gripper_cam_name: str = "gripper",
        stride: int = 2,
        max_points: int = 4096,
        voxel_size: float = 0.005,
        z_min: float = 0.1,
        z_max: float = 10.0,
        use_world: bool = True,
        use_gripper_depth: bool = True,
        depth_scale: float = 1.0,
        selection_mode: str = "fps",
        min_peripheral_points: int = 128,
        focus_boost: float = 8.0,
    ):
        cams = load_json(cameras_json_path)
        cam_table = cams.get("cameras", cams)
        if cam_name not in cam_table:
            raise KeyError(f"Camera '{cam_name}' not found. Available: {list(cam_table.keys())}")
        self.static_cam_name = str(cam_name)
        self.gripper_cam_name = str(gripper_cam_name)
        self.use_gripper_depth = bool(use_gripper_depth)
        self._local_frame = EndEffectorLocalFrame()
        self._camera_specs = {
            self.static_cam_name: self._load_camera_spec(cam_table, self.static_cam_name),
        }
        if self.gripper_cam_name in cam_table:
            self._camera_specs[self.gripper_cam_name] = self._load_camera_spec(cam_table, self.gripper_cam_name)
        static_spec = self._camera_specs[self.static_cam_name]
        self.K = static_spec["K"]
        self.W_T_C = static_spec["W_T_C"]
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
        self.min_peripheral_points = max(int(min_peripheral_points), 0)
        self.focus_boost = float(focus_boost)
        if self.selection_mode not in {"fps", "linspace"}:
            raise ValueError(f"selection_mode must be 'fps' or 'linspace', got {self.selection_mode!r}")

    @staticmethod
    def _load_camera_spec(cam_table: Mapping[str, object], cam_name: str) -> dict[str, np.ndarray]:
        cam = cam_table[cam_name]
        if "K" in cam:
            K = as_3x3(cam["K"])
        elif "intrinsics" in cam:
            K = as_3x3(cam["intrinsics"])
        else:
            raise KeyError(f"Camera '{cam_name}' missing intrinsics. Keys={list(cam.keys())}")
        if "W_T_C" in cam:
            W_T_C = as_4x4(cam["W_T_C"])
            E_T_C = None
        elif "E_T_C" in cam:
            W_T_C = None
            E_T_C = as_4x4(cam["E_T_C"])
        elif "viewMatrix" in cam:
            W_T_C = np.linalg.inv(as_4x4(cam["viewMatrix"])).astype(np.float32)
            E_T_C = None
        else:
            raise KeyError(f"Camera '{cam_name}' missing extrinsics. Keys={list(cam.keys())}")
        return {
            "K": K,
            "W_T_C": W_T_C,
            "E_T_C": E_T_C,
            "fx": np.float32(K[0, 0]),
            "fy": np.float32(K[1, 1]),
            "cx": np.float32(K[0, 2]),
            "cy": np.float32(K[1, 2]),
        }

    def _select_subset_indices(self, xyz: np.ndarray, count: int) -> np.ndarray:
        if xyz.shape[0] <= count:
            return np.arange(xyz.shape[0], dtype=np.int64)
        if self.selection_mode == "linspace":
            return np.linspace(0, xyz.shape[0] - 1, count, dtype=np.int64)
        return _deterministic_fps(xyz, count)

    def _select_indices(
        self,
        xyz: np.ndarray,
        focus_mask: np.ndarray | None = None,
        focus_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        if xyz.shape[0] <= self.max_points:
            return np.arange(xyz.shape[0], dtype=np.int64)
        if focus_mask is None or focus_mask.shape[0] != xyz.shape[0] or not np.any(focus_mask):
            return self._select_subset_indices(xyz, self.max_points)
        if self.selection_mode == "fps" and focus_weights is not None and focus_weights.shape[0] == xyz.shape[0]:
            return _deterministic_weighted_fps(xyz, focus_weights, self.max_points)

        focus_idx = np.flatnonzero(focus_mask)
        peripheral_idx = np.flatnonzero(~focus_mask)
        peripheral_budget = 0
        if peripheral_idx.size > 0 and self.min_peripheral_points > 0 and self.max_points > 1:
            peripheral_budget = min(self.min_peripheral_points, int(peripheral_idx.size), self.max_points - 1)
        focus_budget = max(self.max_points - peripheral_budget, 1)

        if focus_idx.size <= focus_budget:
            chosen_focus = focus_idx
            remaining = self.max_points - int(chosen_focus.size)
        else:
            chosen_focus = focus_idx[self._select_subset_indices(xyz[focus_idx], focus_budget)]
            remaining = self.max_points - int(chosen_focus.size)

        if remaining <= 0 or peripheral_idx.size == 0:
            return chosen_focus

        chosen_peripheral = peripheral_idx[self._select_subset_indices(xyz[peripheral_idx], remaining)]
        return np.concatenate([chosen_focus, chosen_peripheral], axis=0)

    def _sample_camera(
        self,
        *,
        rgb: np.ndarray | None,
        depth: np.ndarray | None,
        camera_name: str,
        robot_obs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if depth is None:
            return None
        depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
        if depth_np.ndim != 2:
            raise ValueError(f"{camera_name} depth must be 2D, got {depth_np.shape}")
        if rgb is None:
            rgb_np = np.zeros((*depth_np.shape, 3), dtype=np.uint8)
        else:
            rgb_np = np.asarray(rgb)
        if rgb_np.ndim != 3 or rgb_np.shape[:2] != depth_np.shape or rgb_np.shape[-1] != 3:
            raise ValueError(
                f"{camera_name} rgb/depth resolution mismatch: rgb={rgb_np.shape} depth={depth_np.shape}"
            )
        depth_np = depth_np * self.depth_scale
        spec = self._camera_specs[camera_name]
        camera_pose_world = spec["W_T_C"]
        if camera_pose_world is None:
            if robot_obs is None:
                raise ValueError(
                    f"{camera_name} camera requires robot_obs to resolve E_T_C extrinsics into world coordinates."
                )
            camera_pose_world = (self._local_frame.make_transform(np.asarray(robot_obs, dtype=np.float32)) @ spec["E_T_C"]).astype(
                np.float32
            )
        height, width = depth_np.shape
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        valid = np.isfinite(depth_np) & (depth_np > self.z_min) & (depth_np < self.z_max)
        if not np.any(valid):
            return None
        x = (uu - float(spec["cx"])) / float(spec["fx"]) * depth_np
        y = (vv - float(spec["cy"])) / float(spec["fy"]) * depth_np
        points_cam = np.stack([x, y, depth_np], axis=-1).astype(np.float32)
        normals_cam, normal_valid = _finite_difference_normals(points_cam, valid)
        step_mask = np.zeros_like(valid, dtype=bool)
        step_mask[:: self.stride, :: self.stride] = True
        sampled_mask = valid & step_mask
        if not np.any(sampled_mask):
            return None
        xyz_cam = points_cam[sampled_mask]
        rgb_sel = rgb_np[sampled_mask].astype(np.float32) / 255.0
        normals_sel = normals_cam[sampled_mask]
        missing_normals = ~normal_valid[sampled_mask]
        if np.any(missing_normals):
            fallback = normalize_vectors(-xyz_cam[missing_normals])
            normals_sel = normals_sel.copy()
            normals_sel[missing_normals] = fallback
        if self.use_world:
            xyz = transform_points(xyz_cam, camera_pose_world)
            normals = transform_normals(normals_sel, camera_pose_world)
        else:
            xyz = xyz_cam.astype(np.float32)
            normals = normalize_vectors(normals_sel)
        return xyz, rgb_sel, normals

    def __call__(self, sample: Mapping[str, np.ndarray]) -> PicfPointCloudFrame:
        clouds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        static_cloud = self._sample_camera(
            rgb=np.asarray(sample["rgb_static"]),
            depth=np.asarray(sample["depth_static"], dtype=np.float32),
            camera_name=self.static_cam_name,
            robot_obs=None if sample.get("robot_obs") is None else np.asarray(sample["robot_obs"], dtype=np.float32),
        )
        if static_cloud is not None:
            clouds.append(static_cloud)
        if self.use_gripper_depth and self.gripper_cam_name in self._camera_specs:
            gripper_cloud = self._sample_camera(
                rgb=None if sample.get("rgb_gripper") is None else np.asarray(sample["rgb_gripper"]),
                depth=None if sample.get("depth_gripper") is None else np.asarray(sample["depth_gripper"], dtype=np.float32),
                camera_name=self.gripper_cam_name,
                robot_obs=None if sample.get("robot_obs") is None else np.asarray(sample["robot_obs"], dtype=np.float32),
            )
            if gripper_cloud is not None:
                clouds.append(gripper_cloud)
        if not clouds:
            return PicfPointCloudFrame(
                grid_coord=np.zeros((0, 3), dtype=np.int32),
                xyz_world=np.zeros((0, 3), dtype=np.float32),
                rgb=np.zeros((0, 3), dtype=np.float32),
                normal_world=np.zeros((0, 3), dtype=np.float32),
                valid_point_mask=np.zeros((0,), dtype=bool),
                frame_valid=False,
            )
        xyz = np.concatenate([xyz_part for xyz_part, _, _ in clouds], axis=0)
        rgb_sel = np.concatenate([rgb_part for _, rgb_part, _ in clouds], axis=0)
        normals = np.concatenate([normal_part for _, _, normal_part in clouds], axis=0)

        focus_mask = None
        focus_weights = None
        focus_centers_world = sample.get("focus_centers_world")
        focus_center_world = sample.get("focus_center_world")
        focus_radius_m = sample.get("focus_radius_m")
        if focus_centers_world is None and focus_center_world is not None:
            focus_centers_world = np.asarray(focus_center_world, dtype=np.float32).reshape(1, 3)
        if focus_centers_world is not None:
            if not self.use_world:
                raise ValueError("focus_center_world requires use_world=True so point selection stays in a single frame.")
            if focus_radius_m is None:
                raise ValueError("focus_radius_m is required when focus_centers_world is provided.")
            focus_centers = np.asarray(focus_centers_world, dtype=np.float32).reshape(-1, 3)
            focus_radius = float(focus_radius_m)
            focus_dist = np.linalg.norm(xyz[:, None, :] - focus_centers[None, :, :], axis=-1).min(axis=1)
            focus_mask = focus_dist <= focus_radius
            focus_weights = 1.0 + self.focus_boost * np.exp(-(focus_dist**2) / max(2.0 * focus_radius * focus_radius, 1e-8))

        choose = self._select_indices(xyz, focus_mask=focus_mask, focus_weights=focus_weights)
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
