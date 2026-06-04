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
    trainable_scope: str = "backbone_only"
    gradient_checkpointing: bool = True
    include_gripper_image: bool = True
    max_length: int = 256
    pi05: bool = True
    action_dim: int = 32
    action_horizon: int = 16
    denoise_steps: int = 10
    inject_state_into_prompt: bool = True
    prompt_state_normalization: str = "none"
    prompt_state_norm_stats_path: str | None = None
    tokenwise_chunk_size: int = 0
    projection_chunk_size: int | None = None
    mlp_chunk_size: int | None = None
    action_context_adapter_gate_init: float = -2.0
    action_context_adapter_rms_cap: bool = True
    action_flow_loss: str = "mse"
    action_flow_huber_delta: float = 1.0
    action_flow_time_alpha: float = 1.5
    action_flow_time_beta: float = 1.0
    action_context_readout_aux_weight: float = 0.0
    action_context_readout_aux_loss: str = "smooth_l1"
    action_context_readout_aux_huber_delta: float = 1.0
    action_context_flow_residual_enabled: bool = False
    action_context_flow_residual_gate_init: float = -2.0
    action_context_flow_residual_time_floor: float = 0.05
    action_context_flow_residual_rms_cap: bool = True
    action_expert_router_enabled: bool = False
    action_expert_router_experts: int = 4
    action_expert_router_rank: int = 64
    action_expert_router_gate_init: float = -2.5
    action_expert_router_temperature: float = 1.0
    action_expert_router_rms_cap: bool = True
