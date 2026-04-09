from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PaliGemmaSemanticConfig:
    model_name: str = "google/paligemma2-3b-pt-224"
    checkpoint_path: str | None = None
    revision: str | None = None
    device: str | None = None
    dtype: str = "bfloat16"
    trainable: bool = False
    gradient_checkpointing: bool = True
    include_gripper_image: bool = True
    max_length: int = 256
