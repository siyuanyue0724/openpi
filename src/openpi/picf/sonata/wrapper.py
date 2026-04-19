from __future__ import annotations

import dataclasses
import importlib
import os
import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from openpi.picf.frame_context import PointFrameContext
from openpi.picf.sonata.config import SonataPointConfig


def _resolve_checkpoint_path(config: SonataPointConfig) -> Path | None:
    if config.checkpoint_path is not None:
        path = Path(config.checkpoint_path).expanduser().resolve()
        return path if path.is_file() else None
    env = os.environ.get("OPENPI_SONATA_CKPT")
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_file():
            return path
    for candidate in config.default_checkpoint_candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _unwrap_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if raw and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in raw.items()):
            return raw
        for key in ("state_dict", "model", "sonata", "encoder"):
            value = raw.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in value.items()
            ):
                return value
    raise RuntimeError("Unable to unwrap Sonata checkpoint into a plain state_dict.")


def _infer_in_channels(state_dict: dict[str, torch.Tensor]) -> int:
    key = "embedding.stem.linear.weight"
    if key not in state_dict:
        raise RuntimeError(f"Sonata checkpoint missing required key {key!r}.")
    weight = state_dict[key]
    if weight.ndim != 2:
        raise RuntimeError(f"Unexpected Sonata embedding weight shape: {tuple(weight.shape)}")
    return int(weight.shape[1])


def _infer_fourier(state_dict: dict[str, torch.Tensor]) -> bool:
    return "input_proj.weight" in state_dict


def _select_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_output_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _resolve_stage_name(model: Any, stage_name: str) -> str:
    stage_map = dict(model.enc.named_children())
    if stage_name == "embedding":
        return stage_name
    if stage_name in stage_map:
        return stage_name
    if stage_name.isdigit():
        candidate = f"enc{stage_name}"
        if candidate in stage_map:
            return candidate
    raise KeyError(f"Requested Sonata stage {stage_name!r} not found.")


def _infer_stage_dim(model: Any, stage_name: str) -> int:
    if stage_name == "embedding":
        return int(model.embedding.embed_channels)
    stage = dict(model.enc.named_children())[stage_name]
    for module in reversed(tuple(stage.children())):
        channels = getattr(module, "channels", None)
        if channels is not None:
            return int(channels)
        out_channels = getattr(module, "out_channels", None)
        if out_channels is not None:
            return int(out_channels)
    raise RuntimeError(f"Unable to infer feature dim for Sonata stage {stage_name!r}.")


def _restore_full_resolution_features(point: Any) -> torch.Tensor:
    feat = point.feat
    cursor = point
    while True:
        if hasattr(cursor, "keys"):
            has_parent = "pooling_parent" in cursor
            has_inverse = "pooling_inverse" in cursor
            if not (has_parent and has_inverse):
                break
            parent = cursor["pooling_parent"]
            inverse = cursor["pooling_inverse"]
        else:
            if not (hasattr(cursor, "pooling_parent") and hasattr(cursor, "pooling_inverse")):
                break
            parent = cursor.pooling_parent
            inverse = cursor.pooling_inverse
        if inverse.numel() > 0:
            inverse_long = inverse.long()
            min_idx = int(inverse_long.min().item())
            max_idx = int(inverse_long.max().item())
            if min_idx < 0 or max_idx >= int(feat.shape[0]):
                raise RuntimeError(
                    "Sonata full-resolution inverse out of bounds: "
                    f"min={min_idx} max={max_idx} size={int(feat.shape[0])} shape={tuple(inverse.shape)}"
                )
        feat = feat[inverse]
        cursor = parent
    return feat


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size == 0:
        return colors
    if float(np.nanmax(colors)) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0).astype(np.float32)


def _normalize_local_grid_coords(grid_coord: np.ndarray) -> np.ndarray:
    """Rebase a local point subset onto a zero-based voxel grid.

    PICF crops a local subset out of a larger per-frame point cloud before sending
    it into Sonata. The subset inherits the original frame-level grid coordinates,
    so its minimum voxel index is often far from zero even though the subset is
    only meant to represent a small local patch. Sonata's sparse path operates on
    local voxel neighborhoods, so rebasing to a zero-origin grid preserves all
    relative geometry while removing a large irrelevant offset.
    """
    grid = np.asarray(grid_coord, dtype=np.int32)
    if grid.size == 0:
        return grid.reshape((-1, 3))
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(f"grid_coord must have shape [N,3], got {grid.shape}")
    grid = grid - grid.min(axis=0, keepdims=True)
    return grid.astype(np.int32, copy=False)


@dataclasses.dataclass(frozen=True)
class SonataPointFeatures:
    features: np.ndarray | torch.Tensor
    checkpoint_loaded: bool
    checkpoint_path: str | None
    feature_dim: int
    stage_name: str
    cpu_fallback_used: bool


def sonata_runtime_available() -> bool:
    try:
        importlib.import_module("openpi.models.sonata_encoder")
    except ModuleNotFoundError:
        return False
    return True


def _load_sonata_runtime() -> tuple[type[Any], type[Any], Any]:
    try:
        module = importlib.import_module("openpi.models.sonata_encoder")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Sonata runtime dependencies are unavailable. "
            "Install the Sonata stack, including torch_scatter, on the target GPU server."
        ) from exc
    return module.Point, module.Sonata, getattr(module, "flash_attn", None)


class SonataPointFeatureExtractor(nn.Module):
    def __init__(self, config: SonataPointConfig | None = None):
        super().__init__()
        self.config = config or SonataPointConfig()
        self.device = _select_device(self.config.device)
        if self.device.type != "cuda":
            raise RuntimeError(
                "SonataPointFeatureExtractor requires CUDA. "
                "CPU fallback has been removed; run this path on a GPU-equipped server."
            )
        self.output_dtype = _select_output_dtype(self.config.dtype)
        self.trainable = bool(self.config.trainable)
        self.checkpoint_path = _resolve_checkpoint_path(self.config)
        self.checkpoint_loaded = False
        point_cls, sonata_cls, flash_attn = _load_sonata_runtime()
        self._point_cls = point_cls

        state_dict: dict[str, torch.Tensor] | None = None
        in_channels = 6
        enable_fourier = False
        if self.checkpoint_path is not None:
            raw = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = _unwrap_state_dict(raw)
            in_channels = _infer_in_channels(state_dict)
            enable_fourier = _infer_fourier(state_dict)
        if in_channels != 6:
            raise RuntimeError(
                f"PICF Sonata wrapper expects SpatialLM/Sonata xyz+rgb input (6 channels), got in_channels={in_channels}."
            )

        env_disable_flash = os.environ.get("OPENPI_SONATA_DISABLE_FLASH", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.flash_enabled = bool(
            self.config.enable_flash and not env_disable_flash and self.device.type == "cuda" and flash_attn is not None
        )
        self.model = sonata_cls(
            in_channels=in_channels,
            order=("z", "z-trans"),
            shuffle_orders=self.config.shuffle_orders,
            enable_flash=self.flash_enabled,
            enable_fourier_encode=enable_fourier,
        )
        if state_dict is not None:
            info = self.model.load_state_dict(state_dict, strict=False)
            missing = [key for key in info.missing_keys if "mask_token" not in key]
            if missing or info.unexpected_keys:
                raise RuntimeError(
                    "Sonata checkpoint did not load cleanly. "
                    f"missing={missing[:20]} unexpected={info.unexpected_keys[:20]}"
                )
            self.checkpoint_loaded = True
        elif not self.config.allow_random_init:
            raise RuntimeError("No Sonata checkpoint found and allow_random_init=False.")

        self.model.to(device=self.device, dtype=torch.float32)
        if not self.trainable:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.stage_name = _resolve_stage_name(self.model, self.config.stage_name)
        self.feature_dim = _infer_stage_dim(self.model, self.stage_name)
        self.cpu_fallback = False

    def _apply_train_checkpoint(self, func, *args):
        if bool(self.trainable and self.training):
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, preserve_rng_state=False)
        return func(*args)

    def _encode_stage(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        point = self._point_cls(sample)
        point = self.model.embedding(point)
        if self.stage_name == "embedding":
            feat = point.feat
            return feat if not self.config.return_full_resolution else _restore_full_resolution_features(point)
        point.serialization(order=self.model.order, shuffle_orders=self.config.shuffle_orders)
        point.sparsify()
        for name, module in self.model.enc.named_children():
            point = module(point)
            if name == self.stage_name:
                feat = point.feat
                return feat if not self.config.return_full_resolution else _restore_full_resolution_features(point)
        raise KeyError(f"Requested Sonata stage {self.stage_name!r} not found.")

    def _build_sample(self, frame_context: PointFrameContext) -> dict[str, torch.Tensor]:
        coord = np.asarray(frame_context.points_local, dtype=np.float32)
        grid_coord = _normalize_local_grid_coords(frame_context.grid_coord)
        color = _normalize_colors(frame_context.colors)
        normal = np.asarray(frame_context.normals_local, dtype=np.float32)
        # SpatialLM/Sonata point features are strictly xyz+rgb.
        # PICF still keeps normals on the side for scaffold/geometry, but does not
        # concatenate them into the Sonata backbone input.
        if coord.shape != grid_coord.shape:
            raise RuntimeError(
                "PICF Sonata sample contract violated: "
                f"coord.shape={coord.shape} != grid_coord.shape={grid_coord.shape}"
            )
        if normal.shape != coord.shape or color.shape != coord.shape:
            raise RuntimeError(
                "PICF Sonata sample contract violated: "
                f"coord.shape={coord.shape}, normal.shape={normal.shape}, color.shape={color.shape}"
            )
        feat = np.concatenate([coord, color], axis=1).astype(np.float32)
        in_channels = int(self.model.embedding.in_channels)
        if in_channels != 6:
            raise RuntimeError(
                f"PICF Sonata sample builder expects model.embedding.in_channels == 6, got {in_channels}."
            )
        if feat.shape[1] < in_channels:
            pad = np.zeros((feat.shape[0], in_channels - feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=1)
        elif feat.shape[1] > in_channels:
            feat = feat[:, :in_channels]
        n_points = int(coord.shape[0])
        return {
            "coord": torch.from_numpy(coord).to(device=self.device, dtype=torch.float32),
            "grid_coord": torch.from_numpy(grid_coord).to(device=self.device, dtype=torch.int32),
            "color": torch.from_numpy(color).to(device=self.device, dtype=torch.float32),
            "normal": torch.from_numpy(normal).to(device=self.device, dtype=torch.float32),
            "feat": torch.from_numpy(feat).to(device=self.device, dtype=torch.float32),
            "grid_size": float(self.config.voxel_size_m),
            "batch": torch.zeros((n_points,), device=self.device, dtype=torch.int64),
            "offset": torch.tensor([n_points], device=self.device, dtype=torch.int64),
        }

    def _encode_stage_checkpointed(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        def _forward(
            coord: torch.Tensor,
            grid_coord: torch.Tensor,
            color: torch.Tensor,
            normal: torch.Tensor,
            feat: torch.Tensor,
            batch: torch.Tensor,
            offset: torch.Tensor,
        ) -> torch.Tensor:
            return self._encode_stage(
                {
                    "coord": coord,
                    "grid_coord": grid_coord,
                    "color": color,
                    "normal": normal,
                    "feat": feat,
                    "grid_size": float(self.config.voxel_size_m),
                    "batch": batch,
                    "offset": offset,
                }
            )

        return self._apply_train_checkpoint(
            _forward,
            sample["coord"],
            sample["grid_coord"],
            sample["color"],
            sample["normal"],
            sample["feat"],
            sample["batch"],
            sample["offset"],
        )

    def encode_local_context(self, frame_context: PointFrameContext) -> SonataPointFeatures:
        n_points = int(frame_context.points_local.shape[0])
        if n_points == 0:
            return SonataPointFeatures(
                features=np.zeros((0, self.feature_dim), dtype=np.float32),
                checkpoint_loaded=self.checkpoint_loaded,
                checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path is not None else None,
                feature_dim=self.feature_dim,
                stage_name=self.stage_name,
                cpu_fallback_used=self.cpu_fallback,
            )
        sample = self._build_sample(frame_context)
        use_grad = bool(self.trainable and self.training)
        context = contextlib.nullcontext() if use_grad else torch.inference_mode()
        with context:
            feat = self._encode_stage_checkpointed(sample)
        feat_out = feat.to(dtype=self.output_dtype) if use_grad else feat.detach().to(dtype=self.output_dtype)
        return SonataPointFeatures(
            features=feat_out,
            checkpoint_loaded=self.checkpoint_loaded,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path is not None else None,
            feature_dim=int(feat_out.shape[1]),
            stage_name=self.stage_name,
            cpu_fallback_used=self.cpu_fallback,
        )

    def forward(self, frame_context: PointFrameContext) -> SonataPointFeatures:
        return self.encode_local_context(frame_context)
