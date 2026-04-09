# src/openpi/transforms/calvin_depth_to_sonata_pointcloud.py
from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Any, Dict, Literal

import numpy as np

def _infer_task_prefix_from_zip(zf: zipfile.ZipFile) -> str | None:
    """Infer task prefix such as task_ABCD_D from a CALVIN zip."""
    names = zf.namelist()
    suffixes = (
        "/training/lang_annotations/auto_lang_ann.npy",
        "/validation/lang_annotations/auto_lang_ann.npy",
        "/calib/cameras.json",
    )
    cands: set[str] = set()
    for suf in suffixes:
        for n in names:
            if n.endswith(suf):
                cands.add(n[:-len(suf)].rstrip("/"))
    cands = {c for c in cands if c}
    if len(cands) == 1:
        return next(iter(cands))
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    if len(tops) == 1:
        return next(iter(tops))
    return None

def _load_json(path: str) -> Dict[str, Any]:
    """Load JSON either from:

    - A normal filesystem path: /abs/or/rel/path.json
    - A zip reference: /path/data.zip::inner/path.json
    - A zip-only path: /path/data.zip (auto-detect inner calib/cameras.json)

    This keeps the CALVIN pipeline working for both extracted directories and zip backend.
    """
    # Allow passing a directory (e.g., /.../task_ABCD_D) - resolve to calib/cameras.json
    if os.path.isdir(path):
        candidate = os.path.join(path, "calib", "cameras.json")
        if os.path.exists(candidate):
            path = candidate

    if "::" in path:
        zip_path, inner_path = path.split("::", 1)
        zip_path = zip_path.strip()
        inner_path = inner_path.lstrip("/").strip()
        if not os.path.exists(zip_path):
            raise FileNotFoundError(zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            try:
                with zf.open(inner_path, "r") as f:
                    return json.load(io.TextIOWrapper(f, encoding="utf-8"))
            except KeyError:
                cands = [n for n in zf.namelist() if n.endswith("/calib/cameras.json") or n == "calib/cameras.json"]
                if cands:
                    with zf.open(cands[0], "r") as f:
                        return json.load(io.TextIOWrapper(f, encoding="utf-8"))

                sibling = zip_path[:-4] if zip_path.endswith(".zip") else zip_path
                alt = os.path.join(sibling, "calib", os.path.basename(inner_path))
                if os.path.exists(alt):
                    with open(alt, "r", encoding="utf-8") as f:
                        return json.load(f)

                raise FileNotFoundError(
                    f"JSON '{inner_path}' not found in zip '{zip_path}', and sibling fallback '{alt}' also missing. "
                    f"Zip cameras candidates={cands[:20]}"
                )

    if path.endswith(".zip") and os.path.exists(path):
        stem = os.path.splitext(os.path.basename(path))[0]  # task_ABCD_D

        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            cands: list[str] = []

            default_inner = f"{stem}/calib/cameras.json"
            if default_inner in names:
                cands.append(default_inner)

            inferred = _infer_task_prefix_from_zip(zf)
            if inferred:
                inferred_inner = f"{inferred}/calib/cameras.json"
                if inferred_inner in names and inferred_inner not in cands:
                    cands.append(inferred_inner)

            for n in names:
                if (n.endswith("/calib/cameras.json") or n == "calib/cameras.json") and n not in cands:
                    cands.append(n)

            if cands:
                with zf.open(cands[0], "r") as f:
                    return json.load(io.TextIOWrapper(f, encoding="utf-8"))

        # Fallback: common on some CALVIN releases where calib/ is outside the zip.
        stem_dir = path[:-4]
        alt = os.path.join(stem_dir, "calib", "cameras.json")
        if os.path.exists(alt):
            with open(alt, "r", encoding="utf-8") as f:
                return json.load(f)

        raise FileNotFoundError(
            f"cameras.json not found for zip '{path}'. "
            f"Tried inner '{stem}/calib/cameras.json' and sibling '{alt}'. "
            "If your calib lives elsewhere, pass cameras_json_path explicitly."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _as_4x4(mat: Any) -> np.ndarray:
    """Convert (4,4) or (3,4) list/array into float32 (4,4)."""
    m = np.asarray(mat, dtype=np.float32)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.concatenate([m, bottom], axis=0)
    raise ValueError(f"Expected extrinsics shape (4,4) or (3,4), got {m.shape}")


def _as_3x3(mat: Any) -> np.ndarray:
    """Convert (3,3) or flat 9 list into float32 (3,3)."""
    m = np.asarray(mat, dtype=np.float32)
    if m.shape == (3, 3):
        return m
    if m.size == 9:
        return m.reshape(3, 3)
    raise ValueError(f"Expected intrinsics shape (3,3) or flat 9, got {m.shape}")


class CalvinDepthToSonataPointCloud:
    """Build a fixed-size point cloud for Sonata.

    point_clouds[out_key] : (max_points, 9) float32
      [:,0:3] grid_coord (stored as float32 but integer values)
      [:,3:6] xyz (meters)
      [:,6:9] rgb (0..1)

    point_cloud_masks[out_key] : bool (sample-level availability)
    """

    def __init__(
        self,
        cameras_json_path: str,
        *,
        cam_name: str = "static",
        rgb_key: str = "rgb_static",
        depth_key: str = "depth_static",
        stride: int = 2,
        max_points: int = 8192,
        voxel_size: float = 0.01,
        z_min: float = 0.1,
        z_max: float = 10.0,
        use_world: bool = True,
        out_key: str = "pointcloud",
        extrinsics_convention: Literal["W_T_C", "C_T_W"] = "W_T_C",
        depth_scale: float = 1.0,
    ):
        cams = _load_json(cameras_json_path)

        # Some schemas nest cameras under "cameras".
        cam_table = cams.get("cameras", cams)
        if not isinstance(cam_table, dict):
            raise TypeError(f"Invalid cameras.json: expected dict, got {type(cam_table)}")

        if cam_name not in cam_table:
            raise KeyError(
                f"Camera '{cam_name}' not found in cameras.json. Available: {list(cam_table.keys())}"
            )
        cam = cam_table[cam_name]

        # --- intrinsics ---
        if "K" in cam:
            K = cam["K"]
        elif "intrinsics" in cam:
            K = cam["intrinsics"]
        else:
            raise KeyError(
                f"Camera '{cam_name}' missing intrinsics (expected 'K' or 'intrinsics'). Keys={list(cam.keys())}"
            )
        self.K = _as_3x3(K)

        # --- extrinsics ---
        if "W_T_C" in cam:
            W_T_C = _as_4x4(cam["W_T_C"])
        elif "viewMatrix" in cam:
            # viewMatrix is typically world->camera; invert to get camera->world.
            vm = _as_4x4(cam["viewMatrix"])
            W_T_C = np.linalg.inv(vm)
        elif "extrinsics" in cam:
            ext = _as_4x4(cam["extrinsics"])
            W_T_C = ext if extrinsics_convention == "W_T_C" else np.linalg.inv(ext)
        else:
            raise KeyError(
                f"Camera '{cam_name}' missing extrinsics (expected 'W_T_C' or 'viewMatrix' or 'extrinsics'). Keys={list(cam.keys())}"
            )

        self.W_T_C = W_T_C.astype(np.float32)

        self.rgb_key = str(rgb_key)
        self.depth_key = str(depth_key)

        self.stride = int(stride)
        self.max_points = int(max_points)
        self.voxel_size = float(voxel_size)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.use_world = bool(use_world)
        self.out_key = str(out_key)
        self.depth_scale = float(depth_scale)

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        self._fx, self._fy, self._cx, self._cy = float(fx), float(fy), float(cx), float(cy)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rgb = data[self.rgb_key]  # (H,W,3) uint8
        depth = data[self.depth_key]  # (H,W) float32 (meters) or scaled

        assert rgb.ndim == 3 and rgb.shape[-1] == 3
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        assert depth.ndim == 2

        depth = depth.astype(np.float32) * self.depth_scale

        H, W = depth.shape
        s = self.stride

        us = np.arange(0, W, s, dtype=np.int32)
        vs = np.arange(0, H, s, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)  # (h',w')
        z = depth[vv, uu].astype(np.float32)

        # Valid depth mask
        m = np.isfinite(z) & (z > self.z_min) & (z < self.z_max)
        if not np.any(m):
            pc = np.zeros((self.max_points, 9), dtype=np.float32)
            pcs = dict(data.get("point_clouds", {}))
            pms = dict(data.get("point_cloud_masks", {}))
            pcs[self.out_key] = pc
            pms[self.out_key] = False
            data["point_clouds"] = pcs
            data["point_cloud_masks"] = pms
            return data

        uu_i = uu[m].astype(np.int32)
        vv_i = vv[m].astype(np.int32)
        z = z[m]

        uu_f = uu_i.astype(np.float32)
        vv_f = vv_i.astype(np.float32)

        x = (uu_f - self._cx) / self._fx * z
        y = (vv_f - self._cy) / self._fy * z
        pts_c = np.stack([x, y, z], axis=1)  # (N,3) camera frame

        # Color
        cols = rgb[vv_i, uu_i].astype(np.float32) / 255.0  # (N,3)

        if self.use_world:
            ones = np.ones((pts_c.shape[0], 1), dtype=np.float32)
            pts_h = np.concatenate([pts_c, ones], axis=1)  # (N,4)
            xyz = (self.W_T_C @ pts_h.T).T[:, :3]  # (N,3)
        else:
            xyz = pts_c

        N = int(xyz.shape[0])
        if N <= 0:
            pc = np.zeros((self.max_points, 9), dtype=np.float32)
            pcs = dict(data.get("point_clouds", {}))
            pms = dict(data.get("point_cloud_masks", {}))
            pcs[self.out_key] = pc
            pms[self.out_key] = False
            data["point_clouds"] = pcs
            data["point_cloud_masks"] = pms
            return data

        # Deterministic fixed-size output (no RNG -> reproducible across workers).
        #
        # IMPORTANT:
        # - We *zero-pad* when N < max_points instead of repeating points.
        #   Repeating makes every sample look like it has max_points valid points,
        #   defeating downstream padding-aware filtering and wasting Sonata compute.
        if N >= self.max_points:
            idx = np.linspace(0, N - 1, self.max_points, dtype=np.int64)
            xyz_sel = xyz[idx].astype(np.float32)
            cols_sel = cols[idx].astype(np.float32)

            # grid_coord (non-negative) computed on selected points
            g = np.floor((xyz_sel - xyz_sel.min(axis=0, keepdims=True)) / self.voxel_size).astype(np.int32)
            g = g - g.min(axis=0, keepdims=True)

            # Pack: [grid(3), feat(6)] where feat is [xyz(3), rgb(3)]
            pc = np.concatenate([g.astype(np.float32), xyz_sel, cols_sel], axis=1).astype(np.float32)
        else:
            xyz_sel = xyz.astype(np.float32)
            cols_sel = cols.astype(np.float32)

            g = np.floor((xyz_sel - xyz_sel.min(axis=0, keepdims=True)) / self.voxel_size).astype(np.int32)
            g = g - g.min(axis=0, keepdims=True)
            pc_valid = np.concatenate([g.astype(np.float32), xyz_sel, cols_sel], axis=1).astype(np.float32)

            pc = np.zeros((self.max_points, 9), dtype=np.float32)
            pc[:N] = pc_valid

        pcs = dict(data.get("point_clouds", {}))
        pms = dict(data.get("point_cloud_masks", {}))
        pcs[self.out_key] = pc
        pms[self.out_key] = True
        data["point_clouds"] = pcs
        data["point_cloud_masks"] = pms
        return data
