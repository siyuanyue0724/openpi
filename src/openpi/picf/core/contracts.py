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
    anchor_active: torch.Tensor | None = None
    anchor_downstream_weight: torch.Tensor | None = None
    vjepa_temporal_priors: torch.Tensor | None = None
    cache_priors: torch.Tensor | None = None
    tracklet_priors: torch.Tensor | None = None
    proposal_priors: torch.Tensor | None = None
    proposal_point_priors: torch.Tensor | None = None
    task_owner_point_priors: torch.Tensor | None = None
    proposal_anchor_seed_priors: torch.Tensor | None = None
    proposal_anchor_seed_assignment: torch.Tensor | None = None
    task_owner_visual_prior: torch.Tensor | None = None
    task_owner_proposal_score: torch.Tensor | None = None
    task_owner_anchor_score: torch.Tensor | None = None
    local_priors: torch.Tensor | None = None
    local_token_indices: torch.Tensor | None = None
    local_source_ids: torch.Tensor | None = None
    slot_address: torch.Tensor | None = None
    slot_content: torch.Tensor | None = None
    support_uncertainty: torch.Tensor | None = None
    support_signature: torch.Tensor | None = None
    binding_signature: torch.Tensor | None = None
    binding_support_score: torch.Tensor | None = None
    binding_address_score: torch.Tensor | None = None
    active_proposals: "PicfActiveProposalState | None" = None
    proposal_to_graph_assignment: torch.Tensor | None = None
    proposal_unexplained_evidence: torch.Tensor | None = None
    proposal_duplicate_cost: torch.Tensor | None = None
    proposal_count: torch.Tensor | None = None
    object_candidate_assignment: torch.Tensor | None = None
    object_candidate_owner_assignment: torch.Tensor | None = None
    object_candidate_owner_point_priors: torch.Tensor | None = None
    object_candidate_coverage: torch.Tensor | None = None
    object_candidate_background: torch.Tensor | None = None
    object_candidate_duplicate_overlap: torch.Tensor | None = None
    object_explanation_quality: torch.Tensor | None = None
    object_explanation_duplicate_overlap: torch.Tensor | None = None
    slot_quality: "PicfSlotQualityState | None" = None


@dataclasses.dataclass
class PicfActiveProposalState:
    """Variable-cardinality active measurement proposals.

    These tensors describe proposal/query initializers only. They are not
    posterior truth and must be matched by the existing posterior file/birth
    competition before they can affect persistent object state.
    """

    tokens: torch.Tensor
    stop_logits: torch.Tensor
    active_prob: torch.Tensor
    role_logits: torch.Tensor
    address_seed: torch.Tensor
    geometry_seed: torch.Tensor
    support_signature_seed: torch.Tensor
    coverage_score: torch.Tensor
    duplicate_score: torch.Tensor
    valid: torch.Tensor
    unexplained_evidence: torch.Tensor | None = None
    count_cost: torch.Tensor | None = None
    continuity_cost: torch.Tensor | None = None


@dataclasses.dataclass
class PicfSlotQualityState:
    """Adaptive object-file quality state for fixed-capacity PICF slots.

    This is the PICF analogue of adaptive-slot quality/no-object selection:
    fixed query capacity is retained for compatibility, but every row carries a
    differentiable belief that it is a real object owner, duplicate capacity, or
    no-object/background reserve. These values are measurement gates, not hard
    ground-truth labels.
    """

    object_quality: torch.Tensor
    no_object_prob: torch.Tensor
    duplicate_prob: torch.Tensor
    active_weight: torch.Tensor
    context_weight: torch.Tensor
    deterministic_object_quality: torch.Tensor
    target_object_quality: torch.Tensor
    target_no_object_prob: torch.Tensor
    target_duplicate_prob: torch.Tensor
    logits: torch.Tensor
    valid: torch.Tensor


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
    age: torch.Tensor
    valid: torch.Tensor
    mask_xy: torch.Tensor | None = None
    mask_weights: torch.Tensor | None = None
    mask_offsets: torch.Tensor | None = None


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
    slot_content: torch.Tensor
    role_ids: torch.Tensor
    source_ids: torch.Tensor
    score: torch.Tensor
    age: torch.Tensor
    uncertainty: torch.Tensor
    innovation: torch.Tensor
    modality_validity: torch.Tensor
    valid: torch.Tensor


@dataclasses.dataclass
class PicfObjectExplanationState:
    """Object/background explanation over typed evidence.

    The masks are column-normalized over object slots plus a background
    residual. They are measurements used to judge whether AQR anchors actually
    explain dense evidence; they are not segmentation labels and never replace
    the posterior belief state.
    """

    object_mask_visual: torch.Tensor | None
    background_mask_visual: torch.Tensor | None
    object_mask_temporal: torch.Tensor | None
    background_mask_temporal: torch.Tensor | None
    object_mask_point: torch.Tensor | None
    background_mask_point: torch.Tensor | None
    object_mask_tactile: torch.Tensor | None
    background_mask_tactile: torch.Tensor | None
    object_mask_tracklet: torch.Tensor | None
    background_mask_tracklet: torch.Tensor | None
    object_mask_proposal: torch.Tensor | None
    background_mask_proposal: torch.Tensor | None
    anchor_quality: torch.Tensor
    anchor_duplicate_overlap: torch.Tensor
    anchor_feature_variance: torch.Tensor
    point_spatial_variance: torch.Tensor
    contact_explanation_score: torch.Tensor
    valid: torch.Tensor
    candidate_coverage: torch.Tensor | None = None
    candidate_background: torch.Tensor | None = None
    candidate_duplicate_overlap: torch.Tensor | None = None


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
    tactile_evidence_mask: torch.Tensor | None = None
    tactile_evidence_weight: torch.Tensor | None = None
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
    owner_active: torch.Tensor | None = None
    support_signature: torch.Tensor | None = None
    binding_signature: torch.Tensor | None = None


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
    binding_signature: torch.Tensor | None = None
    binding_signature_linear_score_mean: torch.Tensor | None = None
    binding_signature_linear_score_abs_mean: torch.Tensor | None = None
    binding_signature_quadratic_score_mean: torch.Tensor | None = None
    binding_signature_quadratic_score_abs_mean: torch.Tensor | None = None
    binding_signature_low_rank_score_mean: torch.Tensor | None = None
    binding_signature_low_rank_score_abs_mean: torch.Tensor | None = None
    binding_signature_combined_score_mean: torch.Tensor | None = None
    binding_signature_combined_score_abs_mean: torch.Tensor | None = None
    binding_signature_calibrated_score_mean: torch.Tensor | None = None
    binding_signature_calibrated_score_abs_mean: torch.Tensor | None = None
    binding_signature_calibrated_score_std: torch.Tensor | None = None
    binding_signature_calibrated_top1_margin_mean: torch.Tensor | None = None
    binding_signature_gate_mean: torch.Tensor | None = None
    binding_signature_update_rate: torch.Tensor | None = None
    binding_signature_measurement_trust: torch.Tensor | None = None
    binding_signature_memory_keep_rate: torch.Tensor | None = None
    binding_signature_measurement_score_std: torch.Tensor | None = None
    binding_signature_measurement_margin: torch.Tensor | None = None
    binding_signature_measurement_dispersion_gate: torch.Tensor | None = None
    recycle_logits: torch.Tensor | None = None
    recycle_support_mass_raw: torch.Tensor | None = None
    recycle_prior_var_mean: torch.Tensor | None = None
    recycle_prior_alpha: torch.Tensor | None = None
    recycle_residual_summary_norm: torch.Tensor | None = None
    recycle_dustbin_raw_mass: torch.Tensor | None = None
    recycle_dustbin_final_mass: torch.Tensor | None = None
    lifecycle_assignment_confidence: torch.Tensor | None = None
    lifecycle_support_entropy: torch.Tensor | None = None
    lifecycle_support_margin: torch.Tensor | None = None
    lifecycle_owner_reliability: torch.Tensor | None = None
    lifecycle_survival_prob: torch.Tensor | None = None
    lifecycle_reset_allowance: torch.Tensor | None = None
    lifecycle_recycle_raw: torch.Tensor | None = None
    lifecycle_inactive_dustbin_mass: torch.Tensor | None = None
    lifecycle_unexplained_dustbin_mass: torch.Tensor | None = None
    file_competition_active: torch.Tensor | None = None
    file_competition_demoted_mass: torch.Tensor | None = None
    file_competition_duplicate_overlap_max: torch.Tensor | None = None
    file_competition_active_duplicate_overlap_max: torch.Tensor | None = None
    file_competition_birth_active: torch.Tensor | None = None
    file_competition_birth_share: torch.Tensor | None = None
    identity_innovation_risk: torch.Tensor | None = None
    address_update_rate: torch.Tensor | None = None
    owner_transport_mass: torch.Tensor | None = None
    owner_transport_confidence: torch.Tensor | None = None
    owner_transport_applied_fraction: torch.Tensor | None = None
    owner_transport_dist_to_standard: torch.Tensor | None = None


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
    object_explanation: PicfObjectExplanationState | None = None


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
