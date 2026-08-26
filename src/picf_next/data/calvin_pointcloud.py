"""Task-independent calibrated CALVIN RGB-D point evidence."""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from picf_next.content_addressing import canonical_mapping_sha256
from picf_next.contracts import ContractError


def _as_3x3(value: object) -> NDArray[np.float32]:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.size != 9:
        raise ContractError("CALVIN camera intrinsics must contain nine values")
    matrix = matrix.reshape(3, 3)
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ContractError("CALVIN camera intrinsics are invalid")
    return matrix


def _as_4x4(value: object) -> NDArray[np.float32]:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape == (3, 4):
        matrix = np.concatenate(
            (matrix, np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)), axis=0
        )
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ContractError("CALVIN camera transform must be finite 4-by-4")
    return matrix


def _load_camera_json(path: str | Path) -> Mapping[str, object]:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "calib" / "cameras.json"
    try:
        if source.suffix == ".zip":
            with zipfile.ZipFile(source) as archive:
                candidates = sorted(
                    name
                    for name in archive.namelist()
                    if name == "calib/cameras.json" or name.endswith("/calib/cameras.json")
                )
                if not candidates:
                    raise ContractError("CALVIN archive contains no camera calibration")
                with archive.open(candidates[0]) as handle:
                    payload = json.load(io.TextIOWrapper(handle, encoding="utf-8"))
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("CALVIN camera calibration is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ContractError("CALVIN camera calibration must be a mapping")
    return payload


def _rpy_zyx_to_matrix(rpy: NDArray[np.generic]) -> NDArray[np.float32]:
    roll, pitch, yaw = [float(value) for value in np.asarray(rpy).reshape(3)]
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _end_effector_transform(robot_obs: NDArray[np.generic]) -> NDArray[np.float32]:
    state = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
    if state.shape[0] < 6 or not np.isfinite(state[:6]).all():
        raise ContractError("CALVIN robot state lacks a finite end-effector pose")
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = _rpy_zyx_to_matrix(state[3:6])
    transform[:3, 3] = state[:3]
    return transform


def _transform_points(
    points: NDArray[np.float32], transform: NDArray[np.float32]
) -> NDArray[np.float32]:
    homogeneous = np.concatenate((points, np.ones((points.shape[0], 1), dtype=np.float32)), axis=1)
    return ((transform @ homogeneous.T).T[:, :3]).astype(np.float32, copy=False)


def deterministic_farthest_point_indices(
    points: NDArray[np.generic], count: int
) -> NDArray[np.int64]:
    """Geometry-only bounded sampling; no task, label or object scorer enters."""

    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ContractError("farthest-point input must be finite N-by-3")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ContractError("farthest-point count must be positive")
    if xyz.shape[0] <= count:
        result = np.arange(xyz.shape[0], dtype=np.int64)
        result.setflags(write=False)
        return result
    centroid = xyz.mean(axis=0, dtype=np.float32)
    first = int(np.argmax(np.linalg.norm(xyz - centroid, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_distance = np.linalg.norm(xyz - xyz[first], axis=1)
    minimum_distance[first] = -1.0
    for index in range(1, count):
        next_index = int(np.argmax(minimum_distance))
        selected[index] = next_index
        candidate = np.linalg.norm(xyz - xyz[next_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, candidate)
        minimum_distance[selected[: index + 1]] = -1.0
    selected.setflags(write=False)
    return selected


@dataclass(frozen=True, slots=True)
class CalibratedPointCloud:
    xyz_world: NDArray[np.float32]
    colors: NDArray[np.float32]
    view_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        count = self.xyz_world.shape[0]
        if (
            self.xyz_world.shape != (count, 3)
            or self.colors.shape != (count, 3)
            or self.view_ids.shape != (count,)
            or not np.issubdtype(self.xyz_world.dtype, np.floating)
            or not np.issubdtype(self.colors.dtype, np.floating)
            or not np.issubdtype(self.view_ids.dtype, np.integer)
            or not np.isfinite(self.xyz_world).all()
            or not np.isfinite(self.colors).all()
            or ((self.colors < 0.0) | (self.colors > 1.0)).any()
            or ((self.view_ids < 0) | (self.view_ids > 1)).any()
        ):
            raise ContractError("calibrated point-cloud arrays violate their physical contract")


@dataclass(frozen=True, slots=True)
class _CameraSpec:
    intrinsics: NDArray[np.float32]
    world_from_camera: NDArray[np.float32] | None
    end_effector_from_camera: NDArray[np.float32] | None


class CalvinCalibratedPointCloudBuilder:
    """Dual-view world-frame reconstruction with no task-conditioned selection."""

    def __init__(
        self,
        cameras_json_path: str | Path,
        *,
        static_camera: str = "static",
        wrist_camera: str = "gripper",
        pixel_stride: int = 2,
        maximum_points: int = 4096,
        minimum_depth_m: float = 0.1,
        maximum_depth_m: float = 10.0,
    ) -> None:
        if (
            isinstance(pixel_stride, bool)
            or not isinstance(pixel_stride, int)
            or pixel_stride <= 0
            or isinstance(maximum_points, bool)
            or not isinstance(maximum_points, int)
            or maximum_points <= 0
        ):
            raise ValueError("CALVIN point-cloud stride and budget must be positive integers")
        if not (0.0 < minimum_depth_m < maximum_depth_m):
            raise ValueError("CALVIN point-cloud depth range is invalid")
        payload = _load_camera_json(cameras_json_path)
        cameras = payload.get("cameras", payload)
        if not isinstance(cameras, Mapping):
            raise ContractError("CALVIN camera table must be a mapping")
        self._static = self._camera_spec(cameras, static_camera)
        self._wrist = self._camera_spec(cameras, wrist_camera)
        self.pixel_stride = pixel_stride
        self.maximum_points = maximum_points
        self.minimum_depth_m = float(minimum_depth_m)
        self.maximum_depth_m = float(maximum_depth_m)
        calibration_sha256 = canonical_mapping_sha256(
            "picf-next.calvin-camera-calibration/v1", payload
        )
        self.encoder_input_contract = (
            f"calvin-dual-rgbd@{calibration_sha256}/"
            f"views-{static_camera}+{wrist_camera}-stride{pixel_stride}-"
            f"points{maximum_points}-depth{minimum_depth_m.hex()}:{maximum_depth_m.hex()}/v1"
        )

    @staticmethod
    def _camera_spec(cameras: Mapping[str, object], name: str) -> _CameraSpec:
        raw = cameras.get(name)
        if not isinstance(raw, Mapping):
            raise ContractError(f"CALVIN camera calibration lacks {name!r}")
        if "K" in raw:
            intrinsics = _as_3x3(raw["K"])
        elif "intrinsics" in raw:
            intrinsics = _as_3x3(raw["intrinsics"])
        else:
            raise ContractError(f"CALVIN camera {name!r} lacks intrinsics")
        if "W_T_C" in raw:
            world = _as_4x4(raw["W_T_C"])
            end_effector = None
        elif "E_T_C" in raw:
            world = None
            end_effector = _as_4x4(raw["E_T_C"])
        elif "viewMatrix" in raw:
            world = np.linalg.inv(_as_4x4(raw["viewMatrix"])).astype(np.float32)
            end_effector = None
        else:
            raise ContractError(f"CALVIN camera {name!r} lacks extrinsics")
        return _CameraSpec(intrinsics, world, end_effector)

    def _sample_view(
        self,
        *,
        rgb: NDArray[np.generic],
        depth: NDArray[np.generic],
        spec: _CameraSpec,
        robot_obs: NDArray[np.generic],
        view_id: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
        image = np.asarray(rgb)
        depth_m = np.asarray(depth, dtype=np.float32)
        if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
            depth_m = depth_m[..., 0]
        if (
            image.ndim != 3
            or image.shape[-1] != 3
            or image.shape[:2] != depth_m.shape
            or image.dtype != np.uint8
        ):
            raise ContractError("CALVIN RGB and depth view geometry is invalid")
        valid = (
            np.isfinite(depth_m)
            & (depth_m > self.minimum_depth_m)
            & (depth_m < self.maximum_depth_m)
        )
        sampled = np.zeros_like(valid, dtype=np.bool_)
        sampled[:: self.pixel_stride, :: self.pixel_stride] = True
        valid &= sampled
        if not valid.any():
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )
        height, width = depth_m.shape
        u, v = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        fx, fy = float(spec.intrinsics[0, 0]), float(spec.intrinsics[1, 1])
        cx, cy = float(spec.intrinsics[0, 2]), float(spec.intrinsics[1, 2])
        camera_points = np.stack(
            (
                (u - cx) / fx * depth_m,
                (v - cy) / fy * depth_m,
                depth_m,
            ),
            axis=-1,
        )[valid].astype(np.float32, copy=False)
        world_from_camera = spec.world_from_camera
        if world_from_camera is None:
            if spec.end_effector_from_camera is None:
                raise RuntimeError("CALVIN dynamic camera transform is incomplete")
            world_from_camera = _end_effector_transform(robot_obs) @ spec.end_effector_from_camera
        xyz = _transform_points(camera_points, world_from_camera)
        colors = image[valid].astype(np.float32) / 255.0
        views = np.full(xyz.shape[0], view_id, dtype=np.int64)
        return xyz, colors, views

    def build(self, frame: Mapping[str, Any]) -> CalibratedPointCloud:
        required = {
            "depth_gripper",
            "depth_static",
            "rgb_gripper",
            "rgb_static",
            "robot_obs",
        }
        if not required <= set(frame):
            raise ContractError("CALVIN full-modal frame lacks an RGB-D field")
        static = self._sample_view(
            rgb=np.asarray(frame["rgb_static"]),
            depth=np.asarray(frame["depth_static"]),
            spec=self._static,
            robot_obs=np.asarray(frame["robot_obs"]),
            view_id=0,
        )
        wrist = self._sample_view(
            rgb=np.asarray(frame["rgb_gripper"]),
            depth=np.asarray(frame["depth_gripper"]),
            spec=self._wrist,
            robot_obs=np.asarray(frame["robot_obs"]),
            view_id=1,
        )
        xyz = np.concatenate((static[0], wrist[0]), axis=0)
        colors = np.concatenate((static[1], wrist[1]), axis=0)
        views = np.concatenate((static[2], wrist[2]), axis=0)
        if xyz.shape[0] > self.maximum_points:
            chosen = deterministic_farthest_point_indices(xyz, self.maximum_points)
            xyz, colors, views = xyz[chosen], colors[chosen], views[chosen]
        for value in (xyz, colors, views):
            value.setflags(write=False)
        return CalibratedPointCloud(xyz, colors, views)
