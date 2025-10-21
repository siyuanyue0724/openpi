# -*- coding: utf-8 -*-
# A drop-in wrapper around LiberoInputs that preserves and re-attaches point clouds.
# - Lazy import of libero_policy.LiberoInputs inside __call__ to avoid circular import
# - No dependency on transforms.base / DataTransformFn; just a plain callable dataclass.

from __future__ import annotations
import dataclasses
from typing import Any, Mapping
import numpy as np


@dataclasses.dataclass(frozen=True)
class LiberoInputsKeepExtras:
    """
    Wraps `openpi.policies.libero_policy.LiberoInputs` but keeps extra fields
    (point_clouds / point_cloud_masks) and re-attaches them so they survive the
    LiberoInputs flattening/packing and the subsequent collate stage.

    Upstream assumptions (already satisfied by your pipeline):
    - DepthToPointCloud + AttachPointCloudToObservation + ValidatePointCloud ran
      and created either TOP-LEVEL keys:
          batch["point_clouds"]["pointcloud"]              : np.float32 [N, 3 + C]
          batch["point_cloud_masks"]["pointcloud"]         : bool (scalar or [1])
      or the same keys under a DICT-STYLE observation:
          batch["observation"]["point_clouds"]["pointcloud"]
          batch["observation"]["point_cloud_masks"]["pointcloud"]

    After calling LiberoInputs, this wrapper writes the saved point cloud to:
      (A) TOP-LEVEL of the returned dict (critical for collate to carry it into
          the final object-style Observation used by the trainer/model), and
      (B) ALSO under out["observation"] if present (object-style or dict-style),
          for robustness with pipelines that read from that branch.

    Notes:
    - We default the frame-level mask to True if missing.
    - We intentionally do *not* reshape the point cloud. ValidatePointCloud
      upstream guarantees shape/dtype (2D [N,F] or 3D [B,N,F] with B==1). The
      collate function will batch it as needed (→ [B, N, F]).
    """

    action_dim: int
    model_type: Any = None  # keep type-agnostic to avoid importing ModelType here

    def __call__(self, batch: dict) -> dict:
        # ------------------------------------------------------------------ #
        # 1) Stash point cloud(s) and mask(s) from top-level or dict-style obs
        # ------------------------------------------------------------------ #
        pcs = None
        pms = None

        if "point_clouds" in batch:
            pcs = batch.get("point_clouds")
            pms = batch.get("point_cloud_masks")
        elif isinstance(batch.get("observation"), dict):
            obs_dict = batch["observation"]
            pcs = obs_dict.get("point_clouds")
            pms = obs_dict.get("point_cloud_masks")

        # Normalize a frame-level mask if missing (scalar bool or [1] is fine)
        if pcs is not None:
            if pms is None:
                pms = {}
            # if missing, treat as valid frame
            if "pointcloud" not in pms or pms["pointcloud"] is None:
                pms["pointcloud"] = np.array(True, dtype=bool)

        # ------------------------------------------------------------------ #
        # 2) Call the original LiberoInputs (LAZY import to avoid cycles)
        # ------------------------------------------------------------------ #
        from openpi.policies.libero_policy import LiberoInputs  # type: ignore
        wrapped = LiberoInputs(action_dim=self.action_dim, model_type=self.model_type)
        out = wrapped(batch)

        # ------------------------------------------------------------------ #
        # 3A) TOP-LEVEL passthrough: ensure collate sees point clouds
        # ------------------------------------------------------------------ #
        if pcs is not None and isinstance(pcs, Mapping) and "pointcloud" in pcs:
            try:
                if isinstance(out, dict):
                    # ensure dict containers exist
                    if not isinstance(out.get("point_clouds"), dict):
                        out["point_clouds"] = {}
                    if not isinstance(out.get("point_cloud_masks"), dict):
                        out["point_cloud_masks"] = {}

                    out["point_clouds"]["pointcloud"] = pcs["pointcloud"]
                    mask_val = pms.get("pointcloud", True) if pms is not None else True
                    out["point_cloud_masks"]["pointcloud"] = np.array(mask_val, dtype=bool)
            except Exception:
                # If for some reason 'out' isn't dict (unexpected), skip gracefully;
                # 3B below may still succeed for the object-style branch.
                pass

        # ------------------------------------------------------------------ #
        # 3B) ALSO attach onto OBJECT-style Observation if LiberoInputs returned
        #     one under out['observation'] (object or dict). This is a no-op if
        #     that branch is not used downstream, but keeps the wrapper generic.
        # ------------------------------------------------------------------ #
        obs_obj = out.get("observation") if isinstance(out, dict) else None
        if pcs is not None and isinstance(pcs, Mapping) and "pointcloud" in pcs and obs_obj is not None:
            try:
                # Prefer object-style attributes
                if not hasattr(obs_obj, "point_clouds") or getattr(obs_obj, "point_clouds") is None:
                    setattr(obs_obj, "point_clouds", {})
                if not hasattr(obs_obj, "point_cloud_masks") or getattr(obs_obj, "point_cloud_masks") is None:
                    setattr(obs_obj, "point_cloud_masks", {})

                obs_obj.point_clouds["pointcloud"] = pcs["pointcloud"]
                mask_val = pms.get("pointcloud", True) if pms is not None else True
                obs_obj.point_cloud_masks["pointcloud"] = np.array(mask_val, dtype=bool)
            except Exception:
                # Fallback: if observation is dict-like, write back as dict
                if isinstance(obs_obj, dict):
                    obs_obj.setdefault("point_clouds", {})["pointcloud"] = pcs["pointcloud"]
                    mask_val = pms.get("pointcloud", True) if pms is not None else True
                    obs_obj.setdefault("point_cloud_masks", {})["pointcloud"] = np.array(mask_val, dtype=bool)
                else:
                    raise

        return out
