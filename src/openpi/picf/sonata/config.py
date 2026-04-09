from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class SonataPointConfig:
    checkpoint_path: str | None = None
    voxel_size_m: float = 0.01
    stage_name: str = "enc4"
    return_full_resolution: bool = True
    device: str | None = None
    dtype: str = "float32"
    trainable: bool = False
    shuffle_orders: bool = False
    allow_random_init: bool = True

    @property
    def default_checkpoint_candidates(self) -> tuple[Path, ...]:
        repo_root = Path(__file__).resolve().parents[4]
        return (
            repo_root / "src" / "pretrain" / "SpatialLM_Sonata_encoder.pth",
            Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "SpatialLM_Sonata_encoder.pth",
        )
