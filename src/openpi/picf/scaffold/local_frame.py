from __future__ import annotations

import dataclasses

import numpy as np

from openpi.picf.geometry import make_transform
from openpi.picf.geometry import rpy_zyx_to_matrix


@dataclasses.dataclass(frozen=True)
class EndEffectorLocalFrame:
    """Default local frame anchored on CALVIN robot_obs[0:6]."""

    fallback_identity: bool = True

    def make_transform(self, robot_obs: np.ndarray) -> np.ndarray:
        robot_obs = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
        if robot_obs.shape[0] >= 6:
            translation = robot_obs[0:3]
            rotation = rpy_zyx_to_matrix(robot_obs[3:6])
            return make_transform(rotation, translation)
        if self.fallback_identity:
            return np.eye(4, dtype=np.float32)
        raise ValueError(f"robot_obs must expose at least 6 dims for local frame, got {robot_obs.shape}")
