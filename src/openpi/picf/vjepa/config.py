from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class VjepaVisualConfig:
    model_name: str = "vjepa2_1_vit_base_384"
    arch_name_override: str | None = None
    checkpoint_path: str | None = None
    checkpoint_key: str | None = None
    camera_json_path: str | None = None
    camera_name: str = "static"
    img_size: int = 384
    num_frames: int = 64
    patch_size: int = 16
    tubelet_size: int = 2
    use_last_two_mean: bool = False
    device: str | None = None
    dtype: str = "bfloat16"
    trainable: bool = False
    feature_mode: str = "auto"
    use_activation_checkpointing: bool = False
    feature_cache_root: str | None = None
    feature_cache_mode: str = "off"
    # Frozen-feature cache is a PICF runtime cache, not an archival dump of the
    # full V-JEPA temporal volume.  PICF consumes current_map() and recent_maps(),
    # so retaining a bounded suffix preserves the training objective while
    # avoiding hundreds of MB of I/O per clip.
    feature_cache_temporal_slices: int = 4
    feature_cache_storage_dtype: str = "bfloat16"
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    delta_v_max_s: float = 0.15
    tau_z_m: float = 5e-3
    tau_vis: float = 0.5
    epsilon_vis: float = 1e-6
    patch_pool_radius: int = 1

    @property
    def temporal_tokens(self) -> int:
        return max(self.num_frames // self.tubelet_size, 1)

    @property
    def spatial_tokens(self) -> int:
        return max(self.img_size // self.patch_size, 1)
