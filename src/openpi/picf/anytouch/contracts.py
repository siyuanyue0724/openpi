from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class AnyTouchSensorFeatures:
    sensor_name: str
    sensor_id: int
    tokens: torch.Tensor
    pooled_feature: torch.Tensor
    T_sens_to_wrist: torch.Tensor
    pseudo_contact_score: float = 0.0
    background_pooled_feature: torch.Tensor | None = None
    rgb_residual_score: float = 0.0
    latent_residual_score: float = 0.0
    contact_score: float = 0.0


@dataclasses.dataclass(frozen=True)
class AnyTouchFeatureBundle:
    global_feature: torch.Tensor
    sensors: dict[str, AnyTouchSensorFeatures]
    checkpoint_loaded: bool
    hidden_dim: int
    pooled_dim: int
