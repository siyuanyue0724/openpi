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


@dataclasses.dataclass(frozen=True)
class AnyTouchFeatureBundle:
    global_feature: torch.Tensor
    sensors: dict[str, AnyTouchSensorFeatures]
    checkpoint_loaded: bool
    hidden_dim: int
    pooled_dim: int
