# src/openpi/transforms/calvin_depth_to_sonata_pointcloud.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

class CalvinDepthToSonataPointCloud:
    """
    Build a fixed-size point cloud for Sonata:
      point_clouds["pointcloud"] : (max_points, 9) float32
        [:,0:3] grid_coord (stored as float32 but integer values)
        [:,3:6] xyz (meters)
        [:,6:9] rgb (0..1)
      point_cloud_masks["pointcloud"] : bool
    """

    def __init__(
        self,
        cameras_json_path: str,
        stride: int = 2,
        max_points: int = 8192,
        voxel_size: float = 0.01,
        z_min: float = 0.1,
        z_max: float = 10.0,
        use_world: bool = True,
        out_key: str = "pointcloud",
    ):
        if not os.path.exists(cameras_json_path):
            raise FileNotFoundError(cameras_json_path)

        with open(cameras_json_path, "r", encoding="utf-8") as f:
            cams = json.load(f)

        static = cams["static"]
        self.K = np.asarray(static["K"], dtype=np.float32)  # (3,3)

        # Prefer W_T_C if present; otherwise try derive from viewMatrix
        if "W_T_C" in static:
            self.W_T_C = np.asarray(static["W_T_C"], dtype=np.float32)  # (4,4)
        else:
            # viewMatrix is typically world->camera; so invert to get camera->world
            vm = np.asarray(static["viewMatrix"], dtype=np.float32)
            self.W_T_C = np.linalg.inv(vm)

        self.stride = int(stride)
        self.max_points = int(max_points)
        self.voxel_size = float(voxel_size)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.use_world = bool(use_world)
        self.out_key = out_key

        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        self._fx, self._fy, self._cx, self._cy = float(fx), float(fy), float(cx), float(cy)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rgb = data["rgb_static"]          # (H,W,3) uint8
        depth = data["depth_static"]      # (H,W) float32
        assert rgb.ndim == 3 and rgb.shape[-1] == 3
        assert depth.ndim == 2

        H, W = depth.shape
        s = self.stride

        us = np.arange(0, W, s, dtype=np.int32)
        vs = np.arange(0, H, s, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)  # (h',w')
        z = depth[vv, uu].astype(np.float32)

        # Valid depth mask
        m = np.isfinite(z) & (z > self.z_min) & (z < self.z_max)
        if not np.any(m):
            # Provide dummy fixed shape; mask False
            pc = np.zeros((self.max_points, 9), dtype=np.float32)
            data["point_clouds"] = {self.out_key: pc}
            data["point_cloud_masks"] = {self.out_key: False}
            return data

        uu = uu[m].astype(np.float32)
        vv = vv[m].astype(np.float32)
        z = z[m]

        x = (uu - self._cx) / self._fx * z
        y = (vv - self._cy) / self._fy * z
        pts_c = np.stack([x, y, z], axis=1)  # (N,3) camera frame

        # Color
        cols = rgb[vv.astype(np.int32), uu.astype(np.int32)].astype(np.float32) / 255.0  # (N,3)

        if self.use_world:
            ones = np.ones((pts_c.shape[0], 1), dtype=np.float32)
            pts_h = np.concatenate([pts_c, ones], axis=1)          # (N,4)
            pts_w = (self.W_T_C @ pts_h.T).T[:, :3]                # (N,3)
            xyz = pts_w
        else:
            xyz = pts_c

        # Downsample / upsample to fixed max_points
        N = xyz.shape[0]
        if N >= self.max_points:
            idx = np.random.choice(N, self.max_points, replace=False)
        else:
            idx = np.random.choice(N, self.max_points, replace=True)

        xyz = xyz[idx].astype(np.float32)
        cols = cols[idx].astype(np.float32)

        # grid_coord (non-negative)
        g = np.floor((xyz - xyz.min(axis=0, keepdims=True)) / self.voxel_size).astype(np.int32)
        g = g - g.min(axis=0, keepdims=True)  # still non-negative

        # Pack: [grid(3), feat(6)] where feat is [xyz(3), rgb(3)]
        feat = np.concatenate([xyz, cols], axis=1)  # (P,6)
        pc = np.concatenate([g.astype(np.float32), feat], axis=1).astype(np.float32)  # (P,9)

        data["point_clouds"] = {self.out_key: pc}
        data["point_cloud_masks"] = {self.out_key: True}
        return data
