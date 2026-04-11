from __future__ import annotations

from pathlib import Path
import contextlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as fn
from torch import nn
from transformers import AutoConfig

from openpi.picf.anytouch.config import AnyTouchConfig
from openpi.picf.anytouch.contracts import AnyTouchFeatureBundle
from openpi.picf.anytouch.contracts import AnyTouchSensorFeatures
from openpi.picf.anytouch.preprocess import preprocess_tactile_clip
from openpi.picf.anytouch.sensor_registry import resolve_sensor_id


def anytouch_runtime_available() -> bool:
    try:
        from openpi.picf.anytouch.vendor.tactile_mae import TactileVideoMAE  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _resolve_device(config: AnyTouchConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(config: AnyTouchConfig) -> torch.dtype:
    if config.dtype == "float16":
        return torch.float16
    if config.dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _unwrap_checkpoint(raw: object) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in raw.items()):
            return raw
        if any(isinstance(k, str) and "touch_mae_model" in k and isinstance(v, torch.Tensor) for k, v in raw.items()):
            return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}
        for key in ("state_dict", "model"):
            value = raw.get(key)
            if isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in value.items()):
                return value
    raise RuntimeError("Unsupported AnyTouch2 checkpoint format.")


def _touch_mae_state_dict(raw_state: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    new_state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if "touch_mae_model" in key and "decoder" not in key and "mask_token" not in key:
            new_state[key.replace("touch_mae_model.", "")] = value
    for key, value in model.state_dict().items():
        new_state.setdefault(key, value)
    return new_state


def _mean_abs_rgb_delta(current: np.ndarray, reference: np.ndarray) -> float:
    current_t = torch.as_tensor(np.array(current, copy=True), dtype=torch.float32)
    reference_t = torch.as_tensor(np.array(reference, copy=True), dtype=torch.float32)
    if current_t.ndim != 3 or current_t.shape[-1] != 3:
        raise ValueError(f"Expected tactile RGB image [H,W,3], got {tuple(current_t.shape)}")
    if reference_t.shape != current_t.shape:
        raise ValueError(
            f"Tactile delta reference must match current frame shape, got current={tuple(current_t.shape)} ref={tuple(reference_t.shape)}"
        )
    if float(current_t.max().item()) > 1.0:
        current_t = current_t / 255.0
    if float(reference_t.max().item()) > 1.0:
        reference_t = reference_t / 255.0
    return float(torch.mean(torch.abs(current_t - reference_t)).item())


def _robust_zscore(value: float, stats: dict[str, Any] | None) -> float:
    if not stats:
        return float(value)
    median = float(stats.get("median", 0.0))
    iqr = max(float(stats.get("iqr", 1.0)), 1e-8)
    return float((float(value) - median) / ((iqr / 1.349) + 1e-8))


def _calibrated_contact_score(
    *,
    sensor_name: str,
    rgb_residual_score: float,
    latent_residual_score: float,
    stats_payload: dict[str, Any] | None,
    use_latent: bool,
) -> float:
    if not stats_payload:
        return (0.5 * (rgb_residual_score + latent_residual_score)) if use_latent else float(rgb_residual_score)
    rgb_stats = stats_payload.get("per_sensor_rgb_stats", {}).get(sensor_name)
    latent_stats = stats_payload.get("per_sensor_latent_stats", {}).get(sensor_name)
    rgb_score = _robust_zscore(rgb_residual_score, rgb_stats)
    if not use_latent:
        return rgb_score
    latent_score = _robust_zscore(latent_residual_score, latent_stats)
    return 0.5 * (rgb_score + latent_score)


def _pooled_from_sensor_tokens(sensor_tokens: torch.Tensor) -> torch.Tensor:
    cls_token = sensor_tokens[0]
    sensor_token = sensor_tokens[1:6].mean(dim=0)
    patch_tokens = sensor_tokens[6:]
    patch_avg = patch_tokens.mean(dim=0)
    patch_max = patch_tokens.max(dim=0).values
    pooled = torch.cat([cls_token, sensor_token, patch_avg, patch_max], dim=-1)
    return fn.layer_norm(pooled, normalized_shape=(pooled.shape[0],))


class AnyTouch2TactileEncoder(nn.Module):
    def __init__(self, config: AnyTouchConfig | None = None):
        super().__init__()
        self.config = config or AnyTouchConfig()
        self.device = _resolve_device(self.config)
        self.dtype = _resolve_dtype(self.config)
        self.trainable = bool(self.config.trainable)
        if self.config.model_size != "base":
            raise NotImplementedError(f"Unsupported AnyTouch model size '{self.config.model_size}'.")
        from openpi.picf.anytouch.vendor.tactile_mae import TactileVideoMAE

        hf_config = AutoConfig.from_pretrained(str(self.config.clip_config_path))
        self.model = TactileVideoMAE(
            hf_config,
            num_frames=self.config.num_frames,
            stride=self.config.stride,
            mask_ratio=self.config.mask_ratio,
        )
        self.checkpoint_loaded = False
        if self.config.checkpoint_path is not None:
            raw = torch.load(Path(self.config.checkpoint_path), map_location="cpu", weights_only=False)
            raw_state = _unwrap_checkpoint(raw)
            self.model.load_state_dict(_touch_mae_state_dict(raw_state, self.model), strict=True)
            self.checkpoint_loaded = True
        elif not self.config.allow_random_init:
            raise RuntimeError("No AnyTouch2 checkpoint found and allow_random_init=False.")
        self.model.to(device=self.device, dtype=self.dtype)
        if not self.trainable:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def encode_sensor_clips(
        self,
        *,
        clips_by_sensor: dict[str, torch.Tensor | object],
        backgrounds_by_sensor: dict[str, object],
        poses_by_sensor: dict[str, torch.Tensor | object],
    ) -> AnyTouchFeatureBundle | None:
        sensor_names = sorted(clips_by_sensor)
        if not sensor_names:
            return None
        batch = []
        sensor_ids = []
        pose_tensors: dict[str, torch.Tensor] = {}
        background_batch = []
        background_sensor_names: list[str] = []
        background_sensor_ids: list[int] = []
        for sensor_name in sensor_names:
            clip_raw = np.asarray(clips_by_sensor[sensor_name])
            background_rgb = None if backgrounds_by_sensor.get(sensor_name) is None else np.asarray(backgrounds_by_sensor[sensor_name])
            clip = preprocess_tactile_clip(clip_raw, background_rgb, self.config)
            batch.append(clip)
            sensor_id = resolve_sensor_id(sensor_name, allow_universal=self.config.allow_universal_sensor_token)
            sensor_ids.append(sensor_id)
            pose_tensors[sensor_name] = torch.as_tensor(poses_by_sensor[sensor_name], device=self.device, dtype=self.dtype)
            if background_rgb is not None:
                background_clip = np.repeat(background_rgb[None, ...], clip_raw.shape[0], axis=0)
                background_batch.append(preprocess_tactile_clip(background_clip, background_rgb, self.config))
                background_sensor_names.append(sensor_name)
                background_sensor_ids.append(sensor_id)
        inputs = torch.stack(batch, dim=0).to(device=self.device, dtype=self.dtype)
        sensor_id_tensor = torch.as_tensor(sensor_ids, device=self.device, dtype=torch.long)
        use_grad = bool(self.trainable and self.training)
        context = contextlib.nullcontext() if use_grad else torch.inference_mode()
        with context:
            tokens = self.model(inputs, sensor_id_tensor, probe=True)
            background_pooled_by_sensor: dict[str, torch.Tensor] = {}
            if background_batch:
                background_inputs = torch.stack(background_batch, dim=0).to(device=self.device, dtype=self.dtype)
                background_ids = torch.as_tensor(background_sensor_ids, device=self.device, dtype=torch.long)
                background_tokens = self.model(background_inputs, background_ids, probe=True)
                for index, sensor_name in enumerate(background_sensor_names):
                    background_pooled_by_sensor[sensor_name] = _pooled_from_sensor_tokens(background_tokens[index])
        hidden_dim = int(tokens.shape[-1])
        if int(tokens.shape[0]) != len(sensor_names):
            raise RuntimeError(
                "AnyTouch sensor-token count mismatch: "
                f"tokens.shape[0]={int(tokens.shape[0])} sensor_names={len(sensor_names)}"
            )
        sensors: dict[str, AnyTouchSensorFeatures] = {}
        pooled_list = []
        for index, sensor_name in enumerate(sensor_names):
            sensor_tokens = tokens[index]
            clip_raw = np.asarray(clips_by_sensor[sensor_name])
            current_rgb = clip_raw[-1]
            temporal_ref = clip_raw[0]
            pooled = _pooled_from_sensor_tokens(sensor_tokens)
            pooled_list.append(pooled)
            background_pooled = background_pooled_by_sensor.get(sensor_name)
            if backgrounds_by_sensor.get(sensor_name) is not None:
                rgb_residual_score = _mean_abs_rgb_delta(current_rgb, np.asarray(backgrounds_by_sensor[sensor_name]))
                latent_residual_score = (
                    float(
                        1.0
                        - fn.cosine_similarity(
                            pooled,
                            background_pooled,
                            dim=0,
                            eps=1e-6,
                        ).item()
                    )
                    if background_pooled is not None
                    else 0.0
                )
                contact_score = _calibrated_contact_score(
                    sensor_name=sensor_name,
                    rgb_residual_score=rgb_residual_score,
                    latent_residual_score=latent_residual_score,
                    stats_payload=self.config.contact_stats_payload,
                    use_latent=background_pooled is not None,
                )
            else:
                rgb_residual_score = _mean_abs_rgb_delta(current_rgb, temporal_ref)
                latent_residual_score = 0.0
                contact_score = rgb_residual_score
            sensors[sensor_name] = AnyTouchSensorFeatures(
                sensor_name=sensor_name,
                sensor_id=int(sensor_id_tensor[index].item()),
                tokens=sensor_tokens,
                pooled_feature=pooled,
                T_sens_to_wrist=pose_tensors[sensor_name],
                pseudo_contact_score=float(contact_score),
                background_pooled_feature=background_pooled,
                rgb_residual_score=float(rgb_residual_score),
                latent_residual_score=float(latent_residual_score),
                contact_score=float(contact_score),
            )
        pooled_stack = torch.stack(pooled_list, dim=0)
        global_feature = pooled_stack.mean(dim=0)
        return AnyTouchFeatureBundle(
            global_feature=global_feature,
            sensors=sensors,
            checkpoint_loaded=self.checkpoint_loaded,
            hidden_dim=hidden_dim,
            pooled_dim=int(global_feature.shape[0]),
        )
