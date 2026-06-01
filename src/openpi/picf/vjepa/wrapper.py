from __future__ import annotations

import dataclasses
import importlib
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa.preprocess import preprocess_video_clip

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

_FEATURE_CACHE_VERSION = 1
_FEATURE_CACHE_MODES = {"off", "read", "read_or_encode"}
_FEATURE_CACHE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclasses.dataclass(frozen=True)
class VjepaFeatureMap:
    tokens_thwc: torch.Tensor | np.ndarray
    source_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    checkpoint_loaded: bool
    model_name: str

    def current_map(self, *, use_last_two_mean: bool = False) -> torch.Tensor | np.ndarray:
        tokens = self.tokens_thwc
        if isinstance(tokens, torch.Tensor):
            if tokens.shape[0] == 0:
                raise RuntimeError("V-JEPA feature map has no temporal slices.")
            if use_last_two_mean and tokens.shape[0] >= 2:
                return tokens[-2:].mean(dim=0)
            return tokens[-1]
        tokens_np = np.asarray(tokens, dtype=np.float32)
        if tokens_np.shape[0] == 0:
            raise RuntimeError("V-JEPA feature map has no temporal slices.")
        if use_last_two_mean and tokens_np.shape[0] >= 2:
            return tokens_np[-2:].mean(axis=0, dtype=np.float32)
        return tokens_np[-1]

    def recent_maps(self, n: int = 2) -> torch.Tensor | np.ndarray:
        """Return the most recent temporal latent maps without averaging time."""
        count = max(int(n), 1)
        tokens = self.tokens_thwc
        if isinstance(tokens, torch.Tensor):
            if tokens.shape[0] == 0:
                raise RuntimeError("V-JEPA feature map has no temporal slices.")
            return tokens[-min(count, int(tokens.shape[0])) :]
        tokens_np = np.asarray(tokens, dtype=np.float32)
        if tokens_np.shape[0] == 0:
            raise RuntimeError("V-JEPA feature map has no temporal slices.")
        return tokens_np[-min(count, int(tokens_np.shape[0])) :]


def vjepa_runtime_available() -> bool:
    try:
        importlib.import_module("openpi.picf.vjepa.vendor.vision_transformer")
    except ModuleNotFoundError:
        return False
    return True


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


def _vjepa_uses_autocast(*, device: torch.device, dtype: torch.dtype) -> bool:
    return bool(device.type == "cuda" and dtype in (torch.float16, torch.bfloat16))


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_clip(clip: np.ndarray) -> str:
    array = np.ascontiguousarray(clip)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(tuple(int(dim) for dim in array.shape)).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _normalize_feature_cache_mode(mode: str | None) -> str:
    value = "off" if mode is None else str(mode).strip().lower()
    if value not in _FEATURE_CACHE_MODES:
        raise ValueError(f"V-JEPA feature_cache_mode must be one of {sorted(_FEATURE_CACHE_MODES)}, got {mode!r}.")
    return value


def _normalize_feature_cache_dtype(dtype: str | None) -> str:
    value = "bfloat16" if dtype is None else str(dtype).strip().lower()
    if value not in _FEATURE_CACHE_DTYPES:
        raise ValueError(
            f"V-JEPA feature_cache_storage_dtype must be one of {sorted(_FEATURE_CACHE_DTYPES)}, got {dtype!r}."
        )
    return value


class Vjepa2VisualEncoder(nn.Module):
    """V-JEPA 2.1 encoder wrapper returning dense temporal feature maps."""

    def __init__(self, config: VjepaVisualConfig):
        super().__init__()
        self.config = config
        self.device = _resolve_device(config)
        self.dtype = _resolve_dtype(config, self.device)
        self.trainable = bool(config.trainable)
        arch_name = config.arch_name_override or _MODEL_ARCH_MAP.get(config.model_name)
        if arch_name is None:
            raise KeyError(f"Unsupported V-JEPA model '{config.model_name}'.")
        self._cache_mode = _normalize_feature_cache_mode(
            os.getenv("OPENPI_VJEPA_FEATURE_CACHE_MODE", config.feature_cache_mode)
        )
        self._cache_storage_dtype_name = _normalize_feature_cache_dtype(
            os.getenv("OPENPI_VJEPA_FEATURE_CACHE_STORAGE_DTYPE", config.feature_cache_storage_dtype)
        )
        self._cache_temporal_slices = max(
            int(os.getenv("OPENPI_VJEPA_FEATURE_CACHE_TEMPORAL_SLICES", config.feature_cache_temporal_slices)),
            1,
        )
        cache_root = os.getenv("OPENPI_VJEPA_FEATURE_CACHE_ROOT", config.feature_cache_root or "")
        self._cache_root = Path(cache_root).expanduser().resolve() if cache_root else None
        if self.trainable and self._cache_mode != "off":
            raise ValueError("V-JEPA feature cache is only valid for frozen encoders; disable it when trainable=True.")
        try:
            from openpi.picf.vjepa.vendor import vision_transformer as vendor_vit
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "V-JEPA runtime dependencies are unavailable. "
                "Install the V-JEPA vendor stack, including timm, on the target server."
            ) from exc

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
            use_activation_checkpointing=bool(config.trainable and config.use_activation_checkpointing),
        )
        # Keep V-JEPA weights in native fp32 whenever mixed precision is requested
        # on CUDA and rely on autocast for the forward path. The vendor stack is
        # not uniformly safe to hard-cast ahead of time, regardless of whether
        # the encoder is frozen or trainable.
        if _vjepa_uses_autocast(device=self.device, dtype=self.dtype):
            self.encoder.to(device=self.device)
        else:
            self.encoder.to(device=self.device, dtype=self.dtype)
        self.checkpoint_loaded = False
        checkpoint_hash = None
        if config.checkpoint_path is not None:
            checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
            checkpoint_hash = _sha256_file(checkpoint_path)
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_key = _resolve_checkpoint_key(config, payload)
            state_dict = _extract_encoder_state_dict(payload, checkpoint_key)
            self.encoder.load_state_dict(state_dict, strict=True)
            self.checkpoint_loaded = True
        if not self.trainable:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        feature_mode = str(getattr(self.config, "feature_mode", "auto")).lower().replace("-", "_")
        self._cache_contract = {
            "version": _FEATURE_CACHE_VERSION,
            "model_name": str(config.model_name),
            "arch_name": str(arch_name),
            "checkpoint_path": str(Path(config.checkpoint_path).expanduser().resolve()) if config.checkpoint_path else None,
            "checkpoint_key": str(config.checkpoint_key) if config.checkpoint_key is not None else None,
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_loaded": bool(self.checkpoint_loaded),
            "img_size": int(config.img_size),
            "num_frames": int(config.num_frames),
            "patch_size": int(config.patch_size),
            "tubelet_size": int(config.tubelet_size),
            "temporal_tokens": int(config.temporal_tokens),
            "spatial_tokens": int(config.spatial_tokens),
            "feature_mode": feature_mode,
            "cache_temporal_slices": int(self._cache_temporal_slices),
            "cache_storage_dtype": str(self._cache_storage_dtype_name),
            "normalize_mean": [float(v) for v in config.normalize_mean],
            "normalize_std": [float(v) for v in config.normalize_std],
        }
        self._cache_contract_hash = _sha256_json(self._cache_contract)
        if self._cache_mode != "off":
            if self._cache_root is None:
                raise ValueError("V-JEPA feature_cache_mode is enabled but no feature_cache_root was provided.")
            self._cache_root.mkdir(parents=True, exist_ok=True)

    def _feature_cache_path(self, clip_hash: str) -> Path:
        assert self._cache_root is not None
        return self._cache_root / self._cache_contract_hash[:16] / f"{clip_hash}.pt"

    def _read_feature_cache(self, path: Path, *, clip_hash: str, source_hw: tuple[int, int]) -> VjepaFeatureMap | None:
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        manifest = payload.get("manifest") if isinstance(payload, dict) else None
        expected_manifest = {
            "contract": self._cache_contract,
            "contract_hash": self._cache_contract_hash,
            "clip_hash": str(clip_hash),
            "source_hw": [int(source_hw[0]), int(source_hw[1])],
            "resized_hw": [int(self.config.img_size), int(self.config.img_size)],
            "stored_temporal_slices": int(self._cache_temporal_slices),
            "storage_dtype": str(self._cache_storage_dtype_name),
        }
        if manifest != expected_manifest:
            raise RuntimeError(f"Stale or invalid V-JEPA feature cache entry: {path}")
        tokens = payload.get("tokens_thwc")
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError(f"Invalid V-JEPA feature cache payload: {path}")
        expected_temporal = min(int(self._cache_temporal_slices), int(self.config.temporal_tokens))
        expected_prefix = (expected_temporal, int(self.config.spatial_tokens), int(self.config.spatial_tokens))
        if tuple(int(dim) for dim in tokens.shape[:3]) != expected_prefix:
            raise RuntimeError(f"V-JEPA feature cache shape mismatch for {path}: {tuple(tokens.shape)}")
        return VjepaFeatureMap(
            tokens_thwc=tokens.detach().to(dtype=torch.float32, device="cpu"),
            source_hw=source_hw,
            resized_hw=(int(self.config.img_size), int(self.config.img_size)),
            checkpoint_loaded=self.checkpoint_loaded,
            model_name=self.config.model_name,
        )

    def _write_feature_cache(self, path: Path, *, clip_hash: str, source_hw: tuple[int, int], tokens_thwc: torch.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "contract": self._cache_contract,
            "contract_hash": self._cache_contract_hash,
            "clip_hash": str(clip_hash),
            "source_hw": [int(source_hw[0]), int(source_hw[1])],
            "resized_hw": [int(self.config.img_size), int(self.config.img_size)],
            "stored_temporal_slices": int(self._cache_temporal_slices),
            "storage_dtype": str(self._cache_storage_dtype_name),
        }
        stored = tokens_thwc.detach()
        stored_slices = min(int(self._cache_temporal_slices), int(stored.shape[0]))
        stored = stored[-stored_slices:].to(
            dtype=_FEATURE_CACHE_DTYPES[str(self._cache_storage_dtype_name)],
            device="cpu",
        )
        payload = {
            "manifest": manifest,
            "tokens_thwc": stored,
        }
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def encode_clip(self, clip: np.ndarray) -> VjepaFeatureMap:
        clip = np.asarray(clip)
        if clip.ndim != 4 or clip.shape[0] != self.config.num_frames:
            raise ValueError(
                f"Expected clip shape [T,H,W,3] with T={self.config.num_frames}, got {clip.shape}"
            )
        source_hw = (int(clip.shape[1]), int(clip.shape[2]))
        cache_mode = getattr(self, "_cache_mode", "off")
        clip_hash = _sha256_clip(clip) if cache_mode != "off" else ""
        cache_path = self._feature_cache_path(clip_hash) if cache_mode != "off" else None
        if cache_path is not None:
            cached = self._read_feature_cache(cache_path, clip_hash=clip_hash, source_hw=source_hw)
            if cached is not None:
                return cached
            if cache_mode == "read":
                raise RuntimeError(f"Missing V-JEPA feature cache entry in read mode: {cache_path}")
        pixel_values = preprocess_video_clip(clip, self.config)
        use_autocast = _vjepa_uses_autocast(device=self.device, dtype=self.dtype)
        if use_autocast:
            pixel_values = pixel_values.to(device=self.device)
        else:
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        video = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        use_grad = bool(self.trainable and self.training)
        feature_mode = str(getattr(self.config, "feature_mode", "auto")).lower().replace("-", "_")
        if feature_mode == "auto":
            use_hierarchical = bool(self.trainable)
        elif feature_mode == "hierarchical":
            use_hierarchical = True
        elif feature_mode == "final":
            use_hierarchical = False
        else:
            raise ValueError(
                "VjepaVisualConfig.feature_mode must be one of {'auto', 'hierarchical', 'final'}, "
                f"got {getattr(self.config, 'feature_mode', None)!r}."
            )
        context = contextlib.nullcontext() if use_grad else torch.inference_mode()
        with context:
            previous_return_hierarchical = getattr(self.encoder, "return_hierarchical", None)
            if previous_return_hierarchical is not None:
                self.encoder.return_hierarchical = bool(use_hierarchical)
            try:
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        tokens = self.encoder(video, training=use_grad)
                else:
                    tokens = self.encoder(video, training=use_grad)
            finally:
                if previous_return_hierarchical is not None:
                    self.encoder.return_hierarchical = previous_return_hierarchical
            token_grid = tokens.reshape(
                1,
                self.config.temporal_tokens,
                self.config.spatial_tokens,
                self.config.spatial_tokens,
                -1,
            )
            if use_grad:
                token_payload = token_grid[0].to(dtype=torch.float32)
            else:
                token_payload = token_grid[0].detach().to(dtype=torch.float32, device="cpu")
        feature_map = VjepaFeatureMap(
            tokens_thwc=token_payload,
            source_hw=source_hw,
            resized_hw=(self.config.img_size, self.config.img_size),
            checkpoint_loaded=self.checkpoint_loaded,
            model_name=self.config.model_name,
        )
        if cache_path is not None and cache_mode == "read_or_encode" and not use_grad:
            self._write_feature_cache(
                cache_path,
                clip_hash=clip_hash,
                source_hw=source_hw,
                tokens_thwc=torch.as_tensor(feature_map.tokens_thwc),
            )
        return feature_map

    def forward(self, clip: np.ndarray) -> VjepaFeatureMap:
        return self.encode_clip(clip)
