from __future__ import annotations

import dataclasses

import torch

from openpi.picf.contracts import RuntimeMeta


@dataclasses.dataclass
class PicfProjectiveGeometryState:
    point_proj_grid_norm: torch.Tensor
    point_proj_grid_index: torch.Tensor
    point_visibility: torch.Tensor
    point_depth: torch.Tensor
    point_depth_sample: torch.Tensor
    point_depth_valid: torch.Tensor
    visual_grid_norm: torch.Tensor
    visual_grid_index: torch.Tensor
    visual_pixel_centers: torch.Tensor
    visual_ray_world: torch.Tensor
    camera_origin_world: torch.Tensor
    projective_compatibility: torch.Tensor
    projective_candidate_mask: torch.Tensor
    projective_attention_bias: torch.Tensor


@dataclasses.dataclass
class PicfTokenFieldState:
    point_tokens: torch.Tensor
    visual_tokens: torch.Tensor
    tactile_tokens: torch.Tensor
    context_tokens: torch.Tensor
    fused_tokens: torch.Tensor
    point_positions: torch.Tensor
    modality_ids: torch.Tensor
    point_align_embeddings: torch.Tensor
    visual_align_embeddings: torch.Tensor
    tactile_align_embeddings: torch.Tensor
    tactile_positions_world: torch.Tensor
    tactile_contact_gate: torch.Tensor
    fusion_attention_mean: torch.Tensor | None
    projective_geometry: PicfProjectiveGeometryState | None


@dataclasses.dataclass
class PicfObservationAnchorState:
    seed_indices: torch.Tensor
    tokens: torch.Tensor
    point_weights: torch.Tensor
    routing_mass_point: torch.Tensor
    routing_mass_visual: torch.Tensor
    routing_support_point: torch.Tensor
    routing_support_visual: torch.Tensor
    routing_gate_point: torch.Tensor
    routing_gate_visual: torch.Tensor
    x: torch.Tensor
    S: torch.Tensor
    a: torch.Tensor


@dataclasses.dataclass
class PicfPredictionCache:
    visual_latent: torch.Tensor | None
    visual_real: torch.Tensor | None
    tactile_real: torch.Tensor | None
    point_real: torch.Tensor | None
    availability: torch.Tensor


@dataclasses.dataclass
class PicfPosteriorAnchorState:
    h: torch.Tensor
    c: torch.Tensor
    mu: torch.Tensor
    Sigma: torch.Tensor
    x: torch.Tensor
    S: torch.Tensor
    a: torch.Tensor
    alpha: torch.Tensor
    contact_prob: torch.Tensor
    support_mass: torch.Tensor
    recycle_gate: torch.Tensor
    binding: torch.Tensor
    evidence_tokens: torch.Tensor
    tokens: torch.Tensor
    global_post: torch.Tensor


@dataclasses.dataclass
class PicfPredictiveState:
    semantic_summary: torch.Tensor
    innovation_token: torch.Tensor
    innovation_norm: torch.Tensor
    availability: torch.Tensor
    control_tokens: torch.Tensor
    pooled_state: torch.Tensor
    action: torch.Tensor
    executed_action: torch.Tensor
    global_pred: torch.Tensor
    prediction_cache: PicfPredictionCache


@dataclasses.dataclass
class PicfControlState:
    hold_reason: str | None


@dataclasses.dataclass
class PicfCoreState:
    runtime_meta: RuntimeMeta
    G_t: torch.Tensor
    token_field: PicfTokenFieldState
    observation_anchors: PicfObservationAnchorState
    posterior: PicfPosteriorAnchorState
    predictive: PicfPredictiveState
    control: PicfControlState
    last_prompt: str | None = None


@dataclasses.dataclass
class PicfCoreOutput:
    state: PicfCoreState
    debug: dict[str, float]
