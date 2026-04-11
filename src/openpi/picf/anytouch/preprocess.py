from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as fn

from openpi.picf.anytouch.config import AnyTouchConfig


def _to_float01(image: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(np.array(image, copy=True), dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected tactile RGB image [H,W,3], got {tuple(tensor.shape)}")
    if float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def preprocess_tactile_clip(clip: np.ndarray, background_rgb: np.ndarray | None, config: AnyTouchConfig) -> torch.Tensor:
    frames = np.asarray(clip)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected tactile clip [T,H,W,3], got {frames.shape}")
    clip_tensor = torch.stack([_to_float01(frame) for frame in frames], dim=0)
    if background_rgb is None:
        if config.require_background:
            raise ValueError("Tactile preprocessing requires a calibrated background frame.")
        bg = clip_tensor[0]
    else:
        bg = _to_float01(background_rgb)
    clip_tensor = clip_tensor - bg.unsqueeze(0) + float(config.offset)
    clip_tensor = clip_tensor.clamp(0.0, 1.0)
    clip_tensor = clip_tensor.permute(0, 3, 1, 2).contiguous()
    clip_tensor = fn.interpolate(
        clip_tensor,
        size=(config.image_size, config.image_size),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.as_tensor(config.mean, dtype=clip_tensor.dtype, device=clip_tensor.device)[None, :, None, None]
    std = torch.as_tensor(config.std, dtype=clip_tensor.dtype, device=clip_tensor.device)[None, :, None, None]
    return (clip_tensor - mean) / std
