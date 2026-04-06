from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.picf.vjepa.config import VjepaVisualConfig


def preprocess_video_clip(clip: np.ndarray, config: VjepaVisualConfig) -> torch.Tensor:
    """Resize and normalize a THWC uint8/float clip into [1,T,3,H,W]."""

    clip = np.asarray(clip)
    if clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError(f"Expected clip shape [T,H,W,3], got {clip.shape}")
    tensor = torch.from_numpy(clip)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    else:
        tensor = tensor.to(torch.float32)
        if float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
    tensor = tensor.permute(0, 3, 1, 2)
    tensor = F.interpolate(
        tensor,
        size=(config.img_size, config.img_size),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(config.normalize_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(config.normalize_std, dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)
