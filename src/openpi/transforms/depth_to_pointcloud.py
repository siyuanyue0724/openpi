# openpi/transforms/depth_to_pointcloud.py
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import numpy as np

def _to_numpy(x: Any) -> np.ndarray:
    try:
        return np.asarray(x)
    except Exception:
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.array(x)

def _as_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape={arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):  # CHW
        return np.transpose(arr, (1, 2, 0))
    return arr  # already HWC

@dataclass
class DepthToPointCloud:
    """
    把单通道深度图（H×W）反投影成点云；可选从对应 RGB 取颜色。
    - depth_map: {cam: depth_key}，其中 sample[depth_key] 必须是 H×W float32
    - color_map: {cam: rgb_key}（可选），支持 CHW/HWC；自动归一化到 0..1
    - intrinsics: {cam: (fx,fy,cx,cy)}（可选），有则得到米制点云；无则相对尺度
    - stride: 下采样步长（控制点数）；max_points: 最多保留点
    - 输出：sample["point_clouds"][out_key] = [max_points, 9]（3×grid_coord + 3×xyz + 3×rgb；不足补0）

    过滤规则（发生在 max_points 抽样之前）：
      - 仅保留有效深度：np.isfinite(Z) 且 Z > min_depth（把 0/非正深度视为无效）
      - 如设置 max_depth，则 Z < max_depth
    """
    depth_map: Dict[str, str]
    # If False (default): missing configured depth keys will raise (fail-fast; avoids silent sensor drop).
    # If True: missing keys are skipped, but we will still emit an empty point cloud if none found
    #          (prevents dataloader collation from breaking due to inconsistent dict keys).
    allow_missing_depth: bool = False
    color_map: Optional[Dict[str, str]] = None
    intrinsics: Optional[Dict[str, Tuple[float, float, float, float]]] = None
    stride: int = 4
    max_points: int = 65536
    voxel_size: float = 0.01
    out_key: str = "pointcloud"
    min_depth: float = 1e-6
    max_depth: Optional[float] = None

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError(f"[DepthToPointCloud] stride must be > 0, got {self.stride}")
        if self.max_points <= 0:
            raise ValueError(f"[DepthToPointCloud] max_points must be > 0, got {self.max_points}")
        if self.voxel_size <= 0:
            raise ValueError(f"[DepthToPointCloud] voxel_size must be > 0, got {self.voxel_size}")

    def __call__(self, sample: dict) -> dict:
        pts_all, rgb_all = [], []
        for cam, dkey in self.depth_map.items():
            if dkey not in sample:
                if not self.allow_missing_depth:
                    raise KeyError(
                        f"[DepthToPointCloud] missing depth key '{dkey}' for cam='{cam}'. "
                        "If this is expected (optional sensor), set allow_missing_depth=True."
                    )
                continue
            depth = _to_numpy(sample[dkey])  # 期望 H×W
            if depth.ndim != 2:
                raise ValueError(f"[DepthToPointCloud] depth must be HxW float32, got {depth.shape} for '{dkey}'")

            H, W = depth.shape
            yy, xx = np.mgrid[0:H:self.stride, 0:W:self.stride]
            z = depth[::self.stride, ::self.stride].astype(np.float32, copy=False)

            if self.intrinsics and cam in self.intrinsics:
                fx, fy, cx, cy = self.intrinsics[cam]
                x = (xx - cx) / fx * z
                y = (yy - cy) / fy * z
            else:
                # 无内参：相对尺度（方便快速打通链路）
                x = (xx - (W - 1) * 0.5) / ((W - 1) * 0.5) * z
                y = (yy - (H - 1) * 0.5) / ((H - 1) * 0.5) * z

            # 先生成所有候选点，再过滤无效深度（NaN/Inf/非正/超上限）
            P_all = np.stack([x, y, z], axis=-1)  # [H_s, W_s, 3]
            zflat = z.reshape(-1)
            m = np.isfinite(P_all).all(axis=2).reshape(-1) & (zflat > self.min_depth)
            if self.max_depth is not None:
                m &= (zflat < self.max_depth)
            P = P_all.reshape(-1, 3)[m]

            # 颜色（若提供）
            C = np.zeros((P.shape[0], 3), dtype=np.float32)
            if self.color_map and cam in self.color_map and self.color_map[cam] in sample:
                rgb_raw = _to_numpy(sample[self.color_map[cam]])
                rgb_hwc = _as_hwc(rgb_raw)
                rgb_sub = rgb_hwc[::self.stride, ::self.stride]
                if rgb_sub.ndim == 3 and rgb_sub.shape[-1] >= 3:
                    C = rgb_sub.reshape(-1, rgb_sub.shape[-1])[m, :3].astype(np.float32, copy=False)
                    # 空数组时避免对 max() 归约
                    if C.size and C.max() > 1.0 + 1e-6:
                        C = C / 255.0  # 若是 0..255，统一到 0..1

            pts_all.append(P)
            rgb_all.append(C)

        if not pts_all:
            # Always emit a consistent key so dataloader collation does not break.
            pc = np.zeros((self.max_points, 9), dtype=np.float32)
            sample.setdefault("point_clouds", {})[self.out_key] = pc
            sample.setdefault("point_cloud_masks", {})[self.out_key] = np.bool_(False)
            return sample

        P = np.concatenate(pts_all, axis=0)
        C = np.concatenate(rgb_all, axis=0)
        n = int(P.shape[0])
        if n == 0:
            pc = np.zeros((self.max_points, 9), dtype=np.float32)
            sample.setdefault("point_clouds", {})[self.out_key] = pc
            sample.setdefault("point_cloud_masks", {})[self.out_key] = np.bool_(False)
            return sample

        if n > self.max_points:
            idx = np.random.choice(n, self.max_points, replace=False)
            P, C = P[idx], C[idx]
            n = self.max_points

        # Compute Sonata grid_coord by voxelizing xyz (shifted to keep non-negative)
        min_xyz = P.min(axis=0, keepdims=True)
        g = np.floor((P - min_xyz) / float(self.voxel_size)).astype(np.int64)
        g = g - g.min(axis=0, keepdims=True)
        g = g.astype(np.float32)

        pc = np.concatenate([g, P.astype(np.float32), C.astype(np.float32)], axis=1)  # [n, 9]
        if n < self.max_points:
            pc = np.concatenate([pc, np.zeros((self.max_points - n, 9), dtype=np.float32)], axis=0)
 

        sample.setdefault("point_clouds", {})[self.out_key] = pc
        sample.setdefault("point_cloud_masks", {})[self.out_key] = np.bool_(True)
        return sample
