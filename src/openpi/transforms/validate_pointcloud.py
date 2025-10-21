from __future__ import annotations
import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True)
class ValidatePointCloud:
    """
    校验单路点云：
      observation.point_clouds[key]        : [N, 3 + feat_dim] float32
      observation.point_cloud_masks[key]   : bool 或 [1]
    校验不过直接抛错（不做任何自动修复/退化）。
    """
    key: str = "pointcloud"
    feat_dim: int = 6
    min_points: int = 1
    allow_mask_all_false: bool = False

    def __call__(self, batch: dict) -> dict:
        pcs = pms = None
        if "point_clouds" in batch:
            pcs = batch["point_clouds"]; pms = batch.get("point_cloud_masks", None)
        elif "observation" in batch and isinstance(batch["observation"], dict):
            pcs = batch["observation"].get("point_clouds", None)
            pms = batch["observation"].get("point_cloud_masks", None)
        if pcs is None or self.key not in pcs:
            raise KeyError(f"missing point_clouds['{self.key}']")
        x = np.asarray(pcs[self.key])
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 2 or x.shape[1] != 3 + self.feat_dim:
            raise ValueError(f"pointcloud shape must be [N,{3 + self.feat_dim}], got {tuple(x.shape)}")
        if x.dtype != np.float32:
            raise TypeError(f"pointcloud dtype must be float32, got {x.dtype}")
        if not np.isfinite(x).all():
            raise ValueError("pointcloud contains NaN/Inf")
        if x.shape[0] < self.min_points:
            raise ValueError(f"pointcloud must have at least {self.min_points} points, got {x.shape[0]}")
        if pms is None or self.key not in pms:
            raise KeyError(f"missing point_cloud_masks['{self.key}']")
        m = np.asarray(pms[self.key])
        if m.shape == () or m.shape == (1,):
            valid = bool(m.reshape(-1)[0])
        elif m.ndim == 2 and m.shape[0] == 1:
            valid = bool(m[0])
        else:
            raise ValueError(f"mask must be a frame-level bool (scalar or [1]), got shape {tuple(m.shape)}")
        if not self.allow_mask_all_false and not valid:
            raise ValueError("frame-level mask is False")
        return batch
