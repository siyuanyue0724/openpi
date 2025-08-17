# openpi/transforms/pointcloud_attach.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def _nested_get(d: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = d
    for p in path.split("/"):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def _ensure_obs_point_maps(obs: Any) -> None:
    # 观测对象里需要有两个可写字典：point_clouds / point_cloud_masks
    # 常见是 Python 对象（dataclass/nnx）+ 普通 dict 字段，直接 set 就行
    if getattr(obs, "point_clouds", None) is None:
        setattr(obs, "point_clouds", {})
    if getattr(obs, "point_cloud_masks", None) is None:
        setattr(obs, "point_cloud_masks", {})


@dataclass(frozen=True)
class AttachPointCloudToObservation:
    """
    将顶层生成的点云（来自 DepthToPointCloud）复制到**字典版 observation**里，供后续
    LiberoInputs 等变换使用。此时 observation 仍是 dict，而非对象。
    """
    key: str = "pointcloud"

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pcs = batch.get("point_clouds", None)
        pms = batch.get("point_cloud_masks", None)

        if pcs is None or self.key not in pcs:
            raise KeyError(
                f"AttachPointCloudToObservation: source point_clouds['{self.key}'] not found; "
                f"available: {list(pcs.keys()) if isinstance(pcs, dict) else 'N/A'}"
            )

        # 确保 observation 是 dict（此阶段尚未构造成对象）
        obs = batch.setdefault("observation", {})
        if not isinstance(obs, dict):
            # 如果已经是对象，先不强行写，交由下一步 AttachPointCloudToLiberoObservation 处理
            return batch

        obs_pcs = obs.setdefault("point_clouds", {})
        obs_pms = obs.setdefault("point_cloud_masks", {})

        obs_pcs[self.key] = pcs[self.key]
        if isinstance(pms, dict) and self.key in pms:
            obs_pms[self.key] = pms[self.key]
        else:
            # 没有显式 mask，就默认 True
            obs_pms[self.key] = np.array(True, dtype=np.bool_)

        return batch


@dataclass(frozen=True)
class AttachPointCloudToLiberoObservation:
    """
    在 LiberoInputs 之后，把点云写回**对象版 Observation**。
    兼容以下位置：
      - batch["observation"]
      - batch["inputs"]["observation"]
      - batch["policy_inputs"]["observation"]
    点云来源优先级：
      1) 顶层 batch["point_clouds"] / ["point_cloud_masks"]
      2) 字典版 batch["observation"]["point_clouds"] / ["point_cloud_masks"]
    """
    key: str = "pointcloud"

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # 1) 找 Observation 对象载体
        obs_obj = batch.get("observation", None)
        if obs_obj is None:
            obs_obj = _nested_get(batch, "inputs/observation") or _nested_get(batch, "policy_inputs/observation")

        if obs_obj is None:
            # 提供更多上下文，便于定位
            available_top = list(batch.keys())
            raise KeyError(
                "AttachPointCloudToLiberoObservation: 'observation' missing in batch; "
                f"checked ['observation', 'inputs/observation', 'policy_inputs/observation']; "
                f"top-level keys: {available_top}"
            )

        # 2) 找点云来源（优先顶层）
        pcs = batch.get("point_clouds", None)
        pms = batch.get("point_cloud_masks", None)

        if (pcs is None) or (self.key not in pcs):
            # 回退到字典版 observation（如果存在）
            obs_dict = batch.get("observation", None)
            if isinstance(obs_dict, dict):
                pcs = obs_dict.get("point_clouds", None)
                pms = obs_dict.get("point_cloud_masks", None)

        if (pcs is None) or (self.key not in pcs):
            raise KeyError(
                f"AttachPointCloudToLiberoObservation: source point_clouds['{self.key}'] not found at "
                f"top-level nor dict-observation; "
                f"top-level has keys: {list(batch.get('point_clouds', {}).keys()) if isinstance(batch.get('point_clouds'), dict) else 'N/A'}"
            )

        # 3) 真正写回对象
        try:
            _ensure_obs_point_maps(obs_obj)
            obs_obj.point_clouds[self.key] = pcs[self.key]
            if isinstance(pms, dict) and self.key in pms:
                obs_obj.point_cloud_masks[self.key] = pms[self.key]
            else:
                obs_obj.point_cloud_masks[self.key] = np.array(True, dtype=np.bool_)
        except Exception as e:
            raise TypeError(
                f"AttachPointCloudToLiberoObservation: cannot attach to observation of type {type(obs_obj)}"
            ) from e

        return batch
