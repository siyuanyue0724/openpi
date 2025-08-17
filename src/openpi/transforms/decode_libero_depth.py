# openpi/transforms/decode_libero_depth.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
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

def _as_hwc(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """把输入统一成 H×W×C；返回 (arr_hwc, was_chw)"""
    if arr.ndim == 2:
        return arr, False
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape={arr.shape}")
    h, w, c = arr.shape
    # 如果是 CHW：C,H,W；则 arr.shape[0] in {1,3} 且最后一维不是通道
    # 可靠判据：若 arr.shape[0] in {1,3} 且 arr.shape[-1] not in {1,3}，多半是 CHW
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        # CHW -> HWC
        return np.transpose(arr, (1, 2, 0)), True
    return arr, False  # 已是 HWC

def _float_to_u8(img: np.ndarray) -> np.ndarray:
    """把 0..1 或 0..255 的 float 转成 0..255 的 uint8（保留 8bit）"""
    if img.dtype.kind == "f":
        # 先判断范围
        maxv = float(np.nanmax(img)) if img.size else 1.0
        if maxv <= 1.0 + 1e-6:
            u8 = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            u8 = np.round(np.clip(img, 0.0, 255.0)).astype(np.uint8)
        return u8
    if img.dtype == np.uint8:
        return img
    # 其他整型统一映射到 u8
    return np.clip(img, 0, 255).astype(np.uint8)

@dataclass
class DecodeLiberoDepth:
    """
    把 LeRobot 数据集里的“深度图像”（可能是 CHW/HWC、三通道灰度或 24bit 打包）
    解码成 float32 单通道深度（H×W）。

    - 若三通道几乎相同：直接取任一通道（从 0..1 或 0..255 反量化）。
    - 否则尝试 24-bit 打包，两种位序（rgb / bgr）各生成一个候选，
      用“平滑度评分”择优（通常深度图在空间上更平滑）。
    - scale：单位缩放（比如毫米→米用 0.001）。不确定就先 None。
    - clip_{min,max}：可选裁剪，默认只裁掉负值。
    """
    src_keys: List[str]
    dst_keys: List[str]
    scale: Optional[float] = None
    clip_min: float = 0.0
    clip_max: Optional[float] = None
    identical_eps: float = 1.5 / 255.0  # 判定三通道相同的容差（归一化误差）

    def __call__(self, sample: dict) -> dict:
        for s, d in zip(self.src_keys, self.dst_keys):
            raw = _to_numpy(sample[s])
            arr = raw  # 可能是 CHW / HWC / 2D
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
                # 已是 HWC
                pass
            elif arr.ndim == 2:
                # 已是单通道
                depth = arr.astype(np.float32)
                depth = np.clip(depth, self.clip_min, self.clip_max) if self.clip_max is not None else np.clip(depth, self.clip_min, None)
                if self.scale is not None:
                    depth = depth * float(self.scale)
                sample[d] = depth.astype(np.float32)
                continue
            else:
                raise ValueError(f"[DecodeLiberoDepth] Unsupported depth shape for key '{s}': {arr.shape}")

            H, W, C = arr.shape
            if C == 1:
                depth = arr[..., 0].astype(np.float32)
                if self.scale is not None:
                    depth = depth * float(self.scale)
                depth = np.clip(depth, self.clip_min, self.clip_max) if self.clip_max is not None else np.clip(depth, self.clip_min, None)
                sample[d] = depth.astype(np.float32)
                continue

            # C==3：三通道灰度 或 24bit 打包
            # 先在浮点域做“相同通道”判定
            ch0, ch1, ch2 = arr[..., 0], arr[..., 1], arr[..., 2]
            if np.allclose(ch0, ch1, atol=self.identical_eps) and np.allclose(ch0, ch2, atol=self.identical_eps):
                # 三通道一致：当作灰度
                depth = ch0.astype(np.float32)
            else:
                # 可能是 24-bit 打包，需把每个通道恢复到 0..255 的整值后再组合
                a_u8 = _float_to_u8(arr)
                r, g, b = a_u8[..., 0].astype(np.uint32), a_u8[..., 1].astype(np.uint32), a_u8[..., 2].astype(np.uint32)
                z_rgb = (r + (g << 8) + (b << 16)).astype(np.float32)
                z_bgr = (b + (g << 8) + (r << 16)).astype(np.float32)
                # 用“平滑度”评分择优：相邻差分的平均绝对值更小者
                def _score(z: np.ndarray) -> float:
                    s = 0.0
                    if z.shape[0] > 1:
                        s += float(np.mean(np.abs(np.diff(z, axis=0))))
                    if z.shape[1] > 1:
                        s += float(np.mean(np.abs(np.diff(z, axis=1))))
                    return s
                depth = z_rgb if _score(z_rgb) <= _score(z_bgr) else z_bgr

            # 裁剪 + 单位缩放
            if self.clip_max is not None:
                depth = np.clip(depth, self.clip_min, self.clip_max)
            else:
                depth = np.clip(depth, self.clip_min, None)
            if self.scale is not None:
                depth = depth * float(self.scale)

            sample[d] = depth.astype(np.float32)

        return sample
