from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa.preprocess import preprocess_video_clip
from openpi.picf.vjepa.vendor import vision_transformer as vendor_vit

_MODEL_ARCH_MAP = {
    "vjepa2_1_vit_base_384": "vit_base",
    "vjepa2_1_vit_large_384": "vit_large",
    "vjepa2_1_vit_giant_384": "vit_giant_xformers",
    "vjepa2_1_vit_gigantic_384": "vit_gigantic_xformers",
}

_STATE_PREFIXES = (
    "module.",
    "backbone.",
    "target_encoder.",
    "encoder.",
    "ema_encoder.",
    "model.",
)

_MODEL_CHECKPOINT_KEY_PREFERENCE = {
    "vjepa2_1_vit_base_384": ("ema_encoder", "target_encoder", "encoder", "state_dict"),
    "vjepa2_1_vit_large_384": ("ema_encoder", "target_encoder", "encoder", "state_dict"),
    "vjepa2_1_vit_giant_384": ("target_encoder", "ema_encoder", "encoder", "state_dict"),
    "vjepa2_1_vit_gigantic_384": ("target_encoder", "ema_encoder", "encoder", "state_dict"),
}


@dataclasses.dataclass(frozen=True)
class VjepaFeatureMap:
    tokens_thwc: np.ndarray
    source_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    checkpoint_loaded: bool
    model_name: str

    def current_map(self, *, use_last_two_mean: bool = False) -> np.ndarray:
        tokens = np.asarray(self.tokens_thwc, dtype=np.float32)
        if tokens.shape[0] == 0:
            raise RuntimeError("V-JEPA feature map has no temporal slices.")
        if use_last_two_mean and tokens.shape[0] >= 2:
            return tokens[-2:].mean(axis=0, dtype=np.float32)
        return tokens[-1]


def _resolve_device(config: VjepaVisualConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(config: VjepaVisualConfig, device: torch.device) -> torch.dtype:
    if config.dtype == "float16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if config.dtype == "bfloat16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return torch.float32


def _clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key_clean = key
        changed = True
        while changed:
            changed = False
            for prefix in _STATE_PREFIXES:
                if key_clean.startswith(prefix):
                    key_clean = key_clean[len(prefix) :]
                    changed = True
        cleaned[key_clean] = value
    return cleaned


def _extract_encoder_state_dict(payload: object, checkpoint_key: str | None) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if checkpoint_key is not None and checkpoint_key in payload and isinstance(payload[checkpoint_key], dict):
            return _clean_backbone_key(payload[checkpoint_key])
        for key in ("target_encoder", "ema_encoder", "encoder", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return _clean_backbone_key(payload[key])
        if all(isinstance(value, torch.Tensor) for value in payload.values()):
            return _clean_backbone_key(payload)
    raise ValueError("Unsupported V-JEPA checkpoint format.")


def _resolve_checkpoint_key(config: VjepaVisualConfig, payload: object) -> str | None:
    if config.checkpoint_key is not None:
        return config.checkpoint_key
    if not isinstance(payload, dict):
        return None
    for key in _MODEL_CHECKPOINT_KEY_PREFERENCE.get(config.model_name, ()):
        if key in payload and isinstance(payload[key], dict):
            return key
    return None


class Vjepa2VisualEncoder:
    """Frozen V-JEPA 2.1 encoder wrapper returning dense temporal feature maps."""

    def __init__(self, config: VjepaVisualConfig):
        self.config = config
        self.device = _resolve_device(config)
        self.dtype = _resolve_dtype(config, self.device)
        arch_name = config.arch_name_override or _MODEL_ARCH_MAP.get(config.model_name)
        if arch_name is None:
            raise KeyError(f"Unsupported V-JEPA model '{config.model_name}'.")
        encoder_builder = getattr(vendor_vit, arch_name)
        self.encoder = encoder_builder(
            patch_size=config.patch_size,
            img_size=(config.img_size, config.img_size),
            num_frames=config.num_frames,
            tubelet_size=config.tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        self.encoder.eval()
        self.encoder.to(device=self.device, dtype=self.dtype)
        self.checkpoint_loaded = False
        if config.checkpoint_path is not None:
            payload = torch.load(Path(config.checkpoint_path), map_location="cpu", weights_only=False)
            checkpoint_key = _resolve_checkpoint_key(config, payload)
            state_dict = _extract_encoder_state_dict(payload, checkpoint_key)
            self.encoder.load_state_dict(state_dict, strict=True)
            self.checkpoint_loaded = True

    @torch.inference_mode()
    def encode_clip(self, clip: np.ndarray) -> VjepaFeatureMap:
        clip = np.asarray(clip)
        if clip.ndim != 4 or clip.shape[0] != self.config.num_frames:
            raise ValueError(
                f"Expected clip shape [T,H,W,3] with T={self.config.num_frames}, got {clip.shape}"
            )
        source_hw = (int(clip.shape[1]), int(clip.shape[2]))
        pixel_values = preprocess_video_clip(clip, self.config)
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        video = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        tokens = self.encoder(video, training=False)
        token_grid = tokens.reshape(
            1,
            self.config.temporal_tokens,
            self.config.spatial_tokens,
            self.config.spatial_tokens,
            -1,
        )
        return VjepaFeatureMap(
            tokens_thwc=token_grid[0].to(dtype=torch.float32, device="cpu").numpy(),
            source_hw=source_hw,
            resized_hw=(self.config.img_size, self.config.img_size),
            checkpoint_loaded=self.checkpoint_loaded,
            model_name=self.config.model_name,
        )
