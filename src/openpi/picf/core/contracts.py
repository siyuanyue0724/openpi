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
class PicfVLGroundingState:
    task_heatmap_logits: torch.Tensor
    effector_heatmap_logits: torch.Tensor
    interaction_heatmap_logits: torch.Tensor
    task_heatmap: torch.Tensor
    effector_heatmap: torch.Tensor
    interaction_heatmap: torch.Tensor
    task_point_prior: torch.Tensor
    effector_point_prior: torch.Tensor
    interaction_point_prior: torch.Tensor
    anchor_point_priors: torch.Tensor
    anchor_x: torch.Tensor
    anchor_S: torch.Tensor
    anchor_tokens: torch.Tensor
    anchor_roles: torch.Tensor
    anchor_scores: torch.Tensor
    visual_pixel_centers: torch.Tensor | None
    valid: torch.Tensor
    confidence: torch.Tensor
    task_pg_heatmap: torch.Tensor | None = None
    effector_pg_heatmap: torch.Tensor | None = None
    interaction_pg_heatmap: torch.Tensor | None = None


@dataclasses.dataclass
class PicfAnchorPriorGraphState:
    pg_priors: torch.Tensor | None
    visual_priors: torch.Tensor
    point_priors: torch.Tensor | None
    tactile_priors: torch.Tensor | None
    posterior_priors: torch.Tensor | None
    anchor_tokens: torch.Tensor
    anchor_roles: torch.Tensor
    anchor_scores: torch.Tensor
    anchor_confidence: torch.Tensor
    anchor_x: torch.Tensor | None
    anchor_S: torch.Tensor | None
    geometry_valid: torch.Tensor
    obs_slot_assignment: torch.Tensor | None
    task_assignment: torch.Tensor | None
    modality_confidence: torch.Tensor
    valid: torch.Tensor
    vjepa_temporal_priors: torch.Tensor | None = None
    cache_priors: torch.Tensor | None = None
    tracklet_priors: torch.Tensor | None = None
    proposal_priors: torch.Tensor | None = None
    local_priors: torch.Tensor | None = None
    slot_address: torch.Tensor | None = None
    slot_content: torch.Tensor | None = None
    support_uncertainty: torch.Tensor | None = None
    support_signature: torch.Tensor | None = None
    binding_support_score: torch.Tensor | None = None
    binding_address_score: torch.Tensor | None = None


@dataclasses.dataclass
class PicfTemporalVisualSupportState:
    tokens: torch.Tensor
    time_ids: torch.Tensor
    view_ids: torch.Tensor
    grid_index: torch.Tensor
    grid_hw: torch.Tensor
    current_token_count: torch.Tensor
    valid: torch.Tensor
    view_names: tuple[str, ...] = ()
    grid_hw_by_view: torch.Tensor | None = None
    source_hw_by_view: torch.Tensor | None = None


@dataclasses.dataclass
class PicfTrackletSupportState:
    tokens: torch.Tensor
    xy_norm: torch.Tensor
    velocity_norm: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor
    track_ids: torch.Tensor
    view_ids: torch.Tensor
    age: torch.Tensor
    valid: torch.Tensor


@dataclasses.dataclass
class PicfPseudoProposalState:
    tokens: torch.Tensor
    centers_xy: torch.Tensor
    boxes_xyxy: torch.Tensor
    objectness: torch.Tensor
    view_ids: torch.Tensor
    source_ids: torch.Tensor
    valid: torch.Tensor


@dataclasses.dataclass
class PicfEvidenceCacheState:
    tokens: torch.Tensor
    slot_address: torch.Tensor
    role_ids: torch.Tensor
    source_ids: torch.Tensor
    age: torch.Tensor
    uncertainty: torch.Tensor
    innovation_at_write: torch.Tensor
    modality_validity: torch.Tensor
    valid: torch.Tensor


@dataclasses.dataclass
class PicfCacheReadState:
    tokens: torch.Tensor
    slot_address: torch.Tensor
    role_ids: torch.Tensor
    source_ids: torch.Tensor
    age: torch.Tensor
    uncertainty: torch.Tensor
    innovation: torch.Tensor
    modality_validity: torch.Tensor
    valid: torch.Tensor


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
    tactile_tokens_all: torch.Tensor | None = None
    tactile_tokens_active: torch.Tensor | None = None
    tactile_group_ids: torch.Tensor | None = None
    tactile_contact_prob: torch.Tensor | None = None
    tactile_anchor_mask: torch.Tensor | None = None
    tactile_normals_world: torch.Tensor | None = None
    tactile_contact_score: torch.Tensor | None = None
    tactile_contact_score_ema: torch.Tensor | None = None
    fusion_attention_mean: torch.Tensor | None = None
    projective_geometry: PicfProjectiveGeometryState | None = None
    point_pool_ids: torch.Tensor | None = None
    point_positions_world: torch.Tensor | None = None
    point_projectable_mask: torch.Tensor | None = None
    temporal_visual: PicfTemporalVisualSupportState | None = None
    tracklet: PicfTrackletSupportState | None = None
    proposal: PicfPseudoProposalState | None = None


@dataclasses.dataclass
class PicfRecurrentTokenFieldState:
    tactile_contact_gate: torch.Tensor | None = None
    tactile_anchor_mask: torch.Tensor | None = None
    tactile_contact_score_ema: torch.Tensor | None = None


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
    routing_mass_tactile: torch.Tensor | None = None
    routing_support_tactile: torch.Tensor | None = None
    routing_gate_tactile: torch.Tensor | None = None
    role_ids: torch.Tensor | None = None
    graph_assignment: torch.Tensor | None = None
    graph_point_weights: torch.Tensor | None = None
    graph_visual_weights: torch.Tensor | None = None
    graph_pg_weights: torch.Tensor | None = None
    graph_temporal_weights: torch.Tensor | None = None
    graph_tactile_weights: torch.Tensor | None = None
    graph_tracklet_weights: torch.Tensor | None = None
    graph_proposal_weights: torch.Tensor | None = None
    anchor_address: torch.Tensor | None = None
    support_signature: torch.Tensor | None = None


@dataclasses.dataclass
class PicfPredictionCache:
    visual_latent: torch.Tensor | None
    visual_real: torch.Tensor | None
    tactile_real: torch.Tensor | None
    point_real: torch.Tensor | None
    availability: torch.Tensor


@dataclasses.dataclass
class PicfTaskReadoutState:
    conditioned_queries: torch.Tensor
    local_tokens: torch.Tensor
    global_token: torch.Tensor
    instruction_tokens: torch.Tensor
    point_weights: torch.Tensor
    x: torch.Tensor
    S: torch.Tensor
    a: torch.Tensor
    semantic_attention: torch.Tensor | None = None
    public_attention: torch.Tensor | None = None
    visual_public_attention: torch.Tensor | None = None
    point_public_attention: torch.Tensor | None = None
    tactile_public_attention: torch.Tensor | None = None
    visual_private_attention: torch.Tensor | None = None
    tactile_private_attention: torch.Tensor | None = None
    point_private_attention: torch.Tensor | None = None
    local_role_ids: torch.Tensor | None = None
    graph_assignment: torch.Tensor | None = None
    visual_weights: torch.Tensor | None = None
    tactile_weights: torch.Tensor | None = None
    geometry_valid: torch.Tensor | None = None
    graph_visual_weights: torch.Tensor | None = None
    graph_tactile_weights: torch.Tensor | None = None
    ordinal_active: torch.Tensor | None = None
    ordinal_scores: torch.Tensor | None = None
    ordinal_ranks: torch.Tensor | None = None
    ordinal_axis: torch.Tensor | None = None
    ordinal_target_rank: torch.Tensor | None = None
    ordinal_selected_slot: torch.Tensor | None = None
    ordinal_confidence: torch.Tensor | None = None


@dataclasses.dataclass
class PicfConditionedControlState:
    base_tokens: torch.Tensor
    task_tokens: torch.Tensor
    tokens: torch.Tensor
    query_state: torch.Tensor
    pi_prefix_tokens: torch.Tensor
    future_condition_tokens: torch.Tensor
    graph_tokens: torch.Tensor | None = None


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
    role_ids: torch.Tensor | None = None
    slot_address: torch.Tensor | None = None
    slot_content: torch.Tensor | None = None
    visual_signature: torch.Tensor | None = None
    temporal_signature: torch.Tensor | None = None
    point_signature: torch.Tensor | None = None
    pg_signature: torch.Tensor | None = None
    tactile_signature: torch.Tensor | None = None
    tracklet_signature: torch.Tensor | None = None
    proposal_signature: torch.Tensor | None = None
    support_signature: torch.Tensor | None = None


@dataclasses.dataclass
class PicfPredictiveState:
    semantic_tokens: torch.Tensor
    innovation_token: torch.Tensor
    innovation_norm: torch.Tensor
    availability: torch.Tensor
    physical_pred_tokens: torch.Tensor
    control_tokens: torch.Tensor
    action_condition_tokens: torch.Tensor | None
    control_query_state: torch.Tensor
    pooled_state: torch.Tensor
    action: torch.Tensor
    action_chunk: torch.Tensor | None
    executed_action: torch.Tensor
    physical_global_pred: torch.Tensor
    physical_prediction_cache: PicfPredictionCache
    predictive_query_state: torch.Tensor
    global_pred: torch.Tensor
    prediction_cache: PicfPredictionCache
    slot_prediction_tokens: torch.Tensor | None = None
    slot_prediction_supports: torch.Tensor | None = None
    evidence_cache: PicfEvidenceCacheState | None = None


@dataclasses.dataclass
class PicfRecurrentPredictiveState:
    executed_action: torch.Tensor
    physical_prediction_cache: PicfPredictionCache
    evidence_cache: PicfEvidenceCacheState | None = None


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
    task_readout: PicfTaskReadoutState
    conditioned_control: PicfConditionedControlState
    predictive: PicfPredictiveState
    control: PicfControlState
    last_prompt: str | None = None
    vl_grounding: PicfVLGroundingState | None = None
    anchor_prior_graph: PicfAnchorPriorGraphState | None = None


@dataclasses.dataclass
class PicfRecurrentCarryState:
    runtime_meta: RuntimeMeta
    token_field: PicfRecurrentTokenFieldState
    posterior: PicfPosteriorAnchorState
    predictive: PicfRecurrentPredictiveState


@dataclasses.dataclass
class PicfCoreOutput:
    state: PicfCoreState
    debug: dict[str, float]


PicfPreviousState = PicfCoreState | PicfRecurrentCarryState
