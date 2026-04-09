from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class AnyTouchConfig:
    checkpoint_path: str | None = None
    device: str | None = None
    dtype: str = "float32"
    model_size: str = "base"
    num_frames: int = 4
    stride: int = 2
    image_size: int = 224
    mask_ratio: float = 0.0
    offset: float = 130.0 / 255.0
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    allow_random_init: bool = False
    allow_universal_sensor_token: bool = False

    @property
    def clip_config_path(self) -> Path:
        return Path(__file__).resolve().parent / "vendor" / "CLIP-B-16"
