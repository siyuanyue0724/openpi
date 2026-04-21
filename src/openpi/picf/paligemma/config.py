from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PaliGemmaSemanticConfig:
    source: str = "auto"
    model_name: str = "google/paligemma2-3b-pt-224"
    checkpoint_path: str | None = None
    checkpoint_config_path: str | None = None
    revision: str | None = None
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    device: str | None = None
    dtype: str = "bfloat16"
    trainable: bool = False
    gradient_checkpointing: bool = True
    include_gripper_image: bool = True
    max_length: int = 256
    pi05: bool = True
    action_dim: int = 32
    action_horizon: int = 16
    denoise_steps: int = 10
    inject_state_into_prompt: bool = True
    tokenwise_chunk_size: int = 0
    projection_chunk_size: int | None = None
    mlp_chunk_size: int | None = None
