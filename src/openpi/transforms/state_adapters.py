# openpi/transforms/state_adapters.py
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

@dataclass(frozen=True)
class SplitStateToDroid:
    """
    将单列状态 `observation/state` 拆成：
      - `observation/joint_position` (前 joint_dims 列，默认 6)
      - `observation/gripper_position` (后 1 列)
    形状兼容：[D] 或 [T, D]，只检查最后一维。
    """
    src_key: str = "observation/state"
    joint_key: str = "observation/joint_position"
    gripper_key: str = "observation/gripper_position"
    joint_dims: int = 6
    keep_src: bool = True  # True: 保留原始 src_key 以兼容已有归一化/统计

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.src_key not in batch:
            raise KeyError(f"SplitStateToDroid: missing '{self.src_key}' in batch")

        s = batch[self.src_key]
        arr = np.asarray(s)
        if arr.shape[-1] < self.joint_dims + 1:
            raise ValueError(
                f"SplitStateToDroid: '{self.src_key}' last dim={arr.shape[-1]} "
                f"< required {self.joint_dims + 1} (joint {self.joint_dims} + gripper 1)"
            )

        joint = arr[..., : self.joint_dims]
        grip  = arr[..., self.joint_dims : self.joint_dims + 1]

        batch[self.joint_key] = joint
        batch[self.gripper_key] = grip

        if not self.keep_src:
            del batch[self.src_key]

        return batch
