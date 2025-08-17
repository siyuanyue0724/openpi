#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# === 配置本地根（可留 None 走 HF 缓存） ===
REPO_ID = "binhng/libero_10_lerobot_mask_depth"
ROOT = Path("~").expanduser() / "Documents/openpi/src/dataset/libero_10_lerobot_mask_depth"
root_arg = str(ROOT) if ROOT.exists() else None

print(f"[info] REPO_ID={REPO_ID}")
print(f"[info] ROOT={root_arg if root_arg else '(not set / will use HF cache)'}")

ds = LeRobotDataset(REPO_ID, root=root_arg, delta_timestamps={})
s = ds[0]

def to_np(x):
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

def describe(name, x):
    arr = to_np(x)
    print(f"[field] {name:35s} shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min() if arr.size else 'n/a'} max={arr.max() if arr.size else 'n/a'}")
    return arr

rgb_front   = describe("observation.images.image", s["observation.images.image"])
rgb_wrist   = describe("observation.images.wrist_image", s["observation.images.wrist_image"])
depth_front = describe("observation.images.image_depth", s["observation.images.image_depth"])
depth_wrist = describe("observation.images.wrist_depth", s["observation.images.wrist_depth"])
state       = describe("observation.state", s["observation.state"])
action      = describe("action", s["action"])

# ---- 新的健壮解码（兼容 CHW / HWC） ----
def as_hwc(arr):
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):  # CHW
        return np.transpose(arr, (1, 2, 0))
    return arr

def float_to_u8(img):
    if img.dtype.kind == "f":
        maxv = float(np.nanmax(img)) if img.size else 1.0
        if maxv <= 1.0 + 1e-6:
            return np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            return np.round(np.clip(img, 0.0, 255.0)).astype(np.uint8)
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)

def decode_depth_any(arr, scale=None, identical_eps=1.5/255.0):
    """返回 H×W float32 深度；先转 HWC 再解码。"""
    a = to_np(arr)
    if a.ndim == 2:
        depth = a.astype(np.float32)
    else:
        a = as_hwc(a)
        H, W, C = a.shape
        if C == 1:
            depth = a[..., 0].astype(np.float32)
        elif C == 3:
            c0, c1, c2 = a[..., 0], a[..., 1], a[..., 2]
            if np.allclose(c0, c1, atol=identical_eps) and np.allclose(c0, c2, atol=identical_eps):
                depth = c0.astype(np.float32)
            else:
                au = float_to_u8(a)
                r, g, b = au[..., 0].astype(np.uint32), au[..., 1].astype(np.uint32), au[..., 2].astype(np.uint32)
                z_rgb = (r + (g << 8) + (b << 16)).astype(np.float32)
                z_bgr = (b + (g << 8) + (r << 16)).astype(np.float32)
                def score(z):
                    s = 0.0
                    if z.shape[0] > 1: s += float(np.mean(np.abs(np.diff(z, axis=0))))
                    if z.shape[1] > 1: s += float(np.mean(np.abs(np.diff(z, axis=1))))
                    return s
                depth = z_rgb if score(z_rgb) <= score(z_bgr) else z_bgr
        else:
            raise ValueError(f"Unsupported channels: {C}")
    depth = np.clip(depth, 0.0, None)
    if scale is not None:
        depth = depth * float(scale)
    return depth.astype(np.float32)

def depth_to_xyz(depth, fx=None, fy=None, cx=None, cy=None, stride=8):
    H, W = depth.shape
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    if fx and fy and cx is not None and cy is not None:
        x = (xx - cx) / fx * z
        y = (yy - cy) / fy * z
    else:
        x = (xx - (W - 1) * 0.5) / ((W - 1) * 0.5) * z
        y = (yy - (H - 1) * 0.5) / ((H - 1) * 0.5) * z
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    m = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-6)
    return xyz[m]

for name, arr in [("front", depth_front), ("wrist", depth_wrist)]:
    try:
        d = decode_depth_any(arr, scale=None)  # 先相对尺度
        print(f"[ok] {name} depth: shape={d.shape} min={np.nanmin(d):.4g} max={np.nanmax(d):.4g}")
        P = depth_to_xyz(d, stride=8)
        print(f"[pc] {name} point cloud ~ {P.shape[0]} points (relative scale)")
    except Exception as e:
        print(f"[fail] {name} decode/pc error: {e}")

print("[done] verification complete.")
