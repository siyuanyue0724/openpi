from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from openpi.shared import normalize as _normalize


ActionNormalizationMode = Literal["none", "zscore", "quantile"]


def default_calvin_action_norm_stats_path() -> Path:
    return Path(__file__).resolve().parents[3] / "assets" / "pi05_calvin_sonata" / "calvin" / "norm_stats.json"


@dataclasses.dataclass(frozen=True)
class PicfActionNormalizer:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None
    q99: np.ndarray | None
    mode: ActionNormalizationMode

    @classmethod
    def from_path(cls, path: str | Path, *, mode: ActionNormalizationMode) -> "PicfActionNormalizer":
        resolved = Path(path).expanduser()
        stats_root = resolved.parent if resolved.is_file() else resolved
        stats = _normalize.load(stats_root)
        if "actions" not in stats:
            raise KeyError(f"norm_stats at {path!s} does not contain an 'actions' entry.")
        action_stats = stats["actions"]
        return cls(
            mean=np.asarray(action_stats.mean, dtype=np.float32),
            std=np.asarray(action_stats.std, dtype=np.float32),
            q01=None if action_stats.q01 is None else np.asarray(action_stats.q01, dtype=np.float32),
            q99=None if action_stats.q99 is None else np.asarray(action_stats.q99, dtype=np.float32),
            mode=mode,
        )

    def _normalize_np(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return x.astype(np.float32, copy=False)
        dims = min(int(x.shape[-1]), int(self.mean.shape[-1]))
        out = np.array(x, dtype=np.float32, copy=True)
        if self.mode == "zscore":
            out[..., :dims] = (out[..., :dims] - self.mean[:dims]) / (self.std[:dims] + 1e-6)
            return out
        if self.q01 is None or self.q99 is None:
            raise ValueError("Quantile action normalization requested, but q01/q99 are unavailable.")
        out[..., :dims] = ((out[..., :dims] - self.q01[:dims]) / (self.q99[:dims] - self.q01[:dims] + 1e-6) * 2.0) - 1.0
        return out

    def _unnormalize_np(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return x.astype(np.float32, copy=False)
        dims = min(int(x.shape[-1]), int(self.mean.shape[-1]))
        out = np.array(x, dtype=np.float32, copy=True)
        if self.mode == "zscore":
            out[..., :dims] = out[..., :dims] * (self.std[:dims] + 1e-6) + self.mean[:dims]
            return out
        if self.q01 is None or self.q99 is None:
            raise ValueError("Quantile action normalization requested, but q01/q99 are unavailable.")
        out[..., :dims] = ((out[..., :dims] + 1.0) * 0.5 * (self.q99[:dims] - self.q01[:dims] + 1e-6)) + self.q01[:dims]
        return out

    def normalize_np(self, x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        return self._normalize_np(np.asarray(x, dtype=np.float32))

    def unnormalize_np(self, x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        return self._unnormalize_np(np.asarray(x, dtype=np.float32))

    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        dims = min(int(x.shape[-1]), int(self.mean.shape[-1]))
        out = x.clone()
        mean = torch.as_tensor(self.mean[:dims], device=x.device, dtype=x.dtype)
        if self.mode == "zscore":
            std = torch.as_tensor(self.std[:dims], device=x.device, dtype=x.dtype)
            out[..., :dims] = (out[..., :dims] - mean) / (std + 1e-6)
            return out
        if self.q01 is None or self.q99 is None:
            raise ValueError("Quantile action normalization requested, but q01/q99 are unavailable.")
        q01 = torch.as_tensor(self.q01[:dims], device=x.device, dtype=x.dtype)
        q99 = torch.as_tensor(self.q99[:dims], device=x.device, dtype=x.dtype)
        out[..., :dims] = ((out[..., :dims] - q01) / (q99 - q01 + 1e-6) * 2.0) - 1.0
        return out

    def unnormalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        dims = min(int(x.shape[-1]), int(self.mean.shape[-1]))
        out = x.clone()
        mean = torch.as_tensor(self.mean[:dims], device=x.device, dtype=x.dtype)
        if self.mode == "zscore":
            std = torch.as_tensor(self.std[:dims], device=x.device, dtype=x.dtype)
            out[..., :dims] = out[..., :dims] * (std + 1e-6) + mean
            return out
        if self.q01 is None or self.q99 is None:
            raise ValueError("Quantile action normalization requested, but q01/q99 are unavailable.")
        q01 = torch.as_tensor(self.q01[:dims], device=x.device, dtype=x.dtype)
        q99 = torch.as_tensor(self.q99[:dims], device=x.device, dtype=x.dtype)
        out[..., :dims] = ((out[..., :dims] + 1.0) * 0.5 * (q99 - q01 + 1e-6)) + q01
        return out
