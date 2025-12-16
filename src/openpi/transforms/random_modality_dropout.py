# src/openpi/transforms/random_modality_dropout.py
from __future__ import annotations
from typing import Any, Dict
import numpy as np

class RandomModalityDropout:
    """
    Drop whole modalities via masks (best tradeoff: simple + consistent with model).
    - image_mask: {"image": bool, "wrist_image": bool}
    - point_cloud_masks: {"pointcloud": bool}

    You can also zero-out the tensor when masked to reduce "leak".
    """
    def __init__(
        self,
        p_drop_pointcloud: float = 0.2,
        p_drop_wrist_rgb: float = 0.1,
        p_drop_main_rgb: float = 0.0,
        zero_out_when_dropped: bool = True,
        seed: int = 0,
        point_key: str = "pointcloud",
    ):
        self.p_drop_pointcloud = float(p_drop_pointcloud)
        self.p_drop_wrist_rgb = float(p_drop_wrist_rgb)
        self.p_drop_main_rgb = float(p_drop_main_rgb)
        self.zero_out = bool(zero_out_when_dropped)
        self.rng = np.random.default_rng(seed)
        self.point_key = point_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # --- images ---
        img = data.get("image", None)
        img_mask = data.get("image_mask", None)

        if img is not None:
            if img_mask is None:
                img_mask = {k: True for k in img.keys()}
                data["image_mask"] = img_mask

            # drop main
            if "image" in img_mask and self.rng.random() < self.p_drop_main_rgb:
                img_mask["image"] = False
                if self.zero_out:
                    data["image"]["image"] = data["image"]["image"] * 0

            # drop wrist
            if "wrist_image" in img_mask and self.rng.random() < self.p_drop_wrist_rgb:
                img_mask["wrist_image"] = False
                if self.zero_out:
                    data["image"]["wrist_image"] = data["image"]["wrist_image"] * 0

            # avoid dropping all vision (optional but recommended)
            if ("image" in img_mask or "wrist_image" in img_mask):
                all_off = True
                for k,v in img_mask.items():
                    all_off = all_off and (not bool(v))
                if all_off:
                    # force keep main image
                    if "image" in img_mask:
                        img_mask["image"] = True

        # --- point cloud ---
        pcm = data.get("point_cloud_masks", None)
        pcs = data.get("point_clouds", None)
        if pcs is not None:
            if pcm is None:
                pcm = {k: True for k in pcs.keys()}
                data["point_cloud_masks"] = pcm

            if self.point_key in pcm and self.rng.random() < self.p_drop_pointcloud:
                pcm[self.point_key] = False
                if self.zero_out and self.point_key in pcs:
                    pcs[self.point_key] = pcs[self.point_key] * 0

        return data
