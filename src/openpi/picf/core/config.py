from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class PicfCoreConfig:
    device: str | None = None
    dtype: str = "float32"
    persistent_anchors: int = 8
    observation_anchors: int = 16
    effector_persistent_anchors: int = 1
    effector_observation_anchors: int = 1
    # Persistent object files need a small identity seed.  Zero-initialized
    # same-role slots are exactly permutation-symmetric and can coalesce under
    # the shared posterior residual/recycle path before action training has a
    # chance to assign stable identities.
    posterior_slot_identity_std: float = 0.02
    task_slot_identity_std: float = 0.02
    # Bootstrap the first posterior geometry from current observation anchors
    # using per-role farthest-point selection. This is not a label; it is the
    # initial object-file birth prior needed before recurrent identity exists.
    posterior_bootstrap_from_observation: bool = True
    # Posterior data association needs a same-role occupancy prior, otherwise
    # the doubly-normalized measurement routing can give every same-role object
    # file the same broad observation mixture and collapse them to one centroid.
    # This prior is label-free: it uses per-role farthest-point observation
    # hypotheses only as a measurement coverage prior before correction.
    posterior_occupancy_prior_enabled: bool = True
    posterior_occupancy_prior_weight: float = 1.0
    posterior_occupancy_prior_sigma_m: float = 0.04
    posterior_occupancy_prior_clip: float = 4.0
    # Observation anchors are measurement hypotheses, not free latent tokens.
    # Retaining a seed-point coverage component prevents every same-role
    # observation row from rereading the same broad point-cloud centroid before
    # posterior association has a chance to assign distinct object files.
    observation_anchor_seed_point_mix: float = 0.35
    hidden_dim: int = 512
    posterior_hidden_dim: int = 512
    latent_dim: int = 112
    innovation_dim: int = 512
    control_dim: int = 512
    semantic_dim: int = 2048
    semantic_cross_dim: int = 2048
    future_hidden_dim: int = 512
    future_vote_heads: int = 4
    fusion_layers: int = 4
    posterior_layers: int = 2
    predictive_layers: int = 2
    control_layers: int = 2
    control_query_tokens: int = 1
    predictive_query_tokens: int = 1
    task_local_queries: int = 8
    task_effector_queries: int = 1
    task_global_queries: int = 1
    task_instruction_queries: int = 2
    # Reserved compatibility field for an unimplemented iterative task-readout variant.
    # The live v2.2 path does not currently consume this knob.
    task_query_rounds: int = 2
    task_self_layers: int = 1
    conditioned_control_queries: int = 4
    pi_prefix_queries: int = 4
    conditioned_future_queries: int = 2
    predictive_semantic_reads: int = 2
    control_semantic_reads: int = 2
    predictive_semantic_dropout_prob: float = 0.1
    semantic_prefix_dropout_prob: float = 0.0
    attention_heads: int = 8
    query_rounds: int = 2
    crop_radius_m: float = 0.10
    global_scene_point_cap: int = 1024
    scene_anchor_border_patches: float = 1.0
    point_focus_sigma_m: float = 0.03
    workspace_radius_m: float = 0.5
    epsilon_s: float = 1e-6
    epsilon_a: float = 1e-6
    epsilon_ext_m2: float = 1e-8
    epsilon_residual: float = 1e-6
    z_min_m: float = 1e-3
    z_max_m: float = 10.0
    sigma_min2: float = 1e-4
    sigma_max2: float = 10.0
    sigma_reset: float = 1.0
    bind_sigma_m: float = 5e-3
    lambda_bind_hidden: float = 1.0
    lambda_bind_geom: float = 1.0
    lambda_bind_prior: float = 0.5
    alpha_init: float = 0.05
    sync_tolerance_s: float = 0.02
    visual_stale_s: float = 0.15
    tactile_stale_s: float = 0.05
    visual_real_grid: int = 64
    visual_latent_tokens: int = 8
    tactile_real_grid: int = 4
    tactile_latent_tokens: int = 4
    point_real_grid: int = 4
    point_latent_tokens: int = 8
    tactile_aux_dim: int = 8
    tau_force_n: float = 1.0
    tau_indent_m: float = 5e-4
    tau_tactile_pressure: float = 0.1
    tau_tactile_pseudo_contact: float = 0.04
    tactile_contact_tau_on: float = 0.08
    tactile_contact_tau_off: float = 0.04
    tactile_contact_temperature: float = 0.02
    tactile_contact_ema_beta: float = 0.8
    # Two-level tactile evidence contract:
    # - soft evidence enters the typed memory once calibrated contact is above
    #   the weak point/tactile alignment floor;
    # - dense tactile patch rereads are reserved for confident contact.
    # This prevents calibrated tactile geometry from becoming an all-or-nothing
    # branch while still avoiding noisy dense tactile activation.
    tactile_evidence_prob_floor: float = 0.35
    tactile_anchor_prob_on: float = 0.55
    # Contact is evidence about the contacted object, not a separate object
    # owner. When enabled, tactile tokens attach to role-1 object owners and
    # are blocked from non-object roles during assignment reads.
    tactile_attach_to_object_owner: bool = True
    tactile_group_proposals: int = 2
    max_action_pos_m: float = 0.025
    max_action_rot_rad: float = math.pi / 18.0
    max_action_gripper: float = 1.0
    action_output_clip: float | None = None
    pointcloud_requires_rgb: bool = True
    visual_real_enabled: bool = True
    language_enabled: bool = True
    hold_uncertainty_threshold: float = 1.5
    hold_innovation_threshold: float = 4.0
    hold_activity_threshold: float = 0.5
    sigma_proj_patches: float = 1.5
    tau_proj: float = 0.25
    tau_proj_depth_m: float = 0.01
    projective_bias_scale: float = 0.25
    tau_route_p: float = 0.1
    tau_route_v: float = 0.1
    vl_anchor_router_enabled: bool = False
    vl_grounding_view: str = "static"
    vl_heatmap_hidden_dim: int = 512
    vl_anchor_modes: int = 6
    vl_anchor_nms_radius_m: float = 0.04
    vl_anchor_local_sigma_m: float = 0.04
    vl_min_visible_mass: float = 1e-4
    vl_heatmap_temperature: float = 1.0
    vl_obs_anchor_gate_init: float = -4.0
    vl_task_point_gate_init: float = -4.0
    vl_posterior_bind_gate_init: float = -6.0
    vl_prior_bias_clip: float = 4.0
    lambda_vl_heatmap_task: float = 0.0
    lambda_vl_heatmap_effector: float = 0.0
    lambda_vl_heatmap_interaction: float = 0.0
    lambda_vl_point_consistency: float = 0.0
    lambda_vl_anchor_diversity: float = 0.0
    mapg_enabled: bool = False
    mapg_anchor_count: int = 8
    mapg_message_rounds: int = 1
    mapg_visual_sigma_patches: float = 2.0
    mapg_tactile_sigma_m: float = 0.08
    mapg_posterior_sigma_m: float = 0.08
    mapg_confidence_floor: float = 0.05
    mapg_assignment_sinkhorn_iters: int = 6
    mapg_assignment_temperature: float = 1.0
    mapg_assignment_quality_uniform_mix: float = 0.25
    mapg_mode_confidence_threshold: float = 0.10
    mapg_obs_gate_init: float = -2.0
    mapg_task_gate_init: float = -2.0
    mapg_posterior_gate_init: float = -4.0
    mapg_control_gate_init: float = -2.0
    mapg_obs_point_mix_floor: float = 0.25
    mapg_prior_bias_clip: float = 4.0
    # Direct-final AQR-MAPG path. This is the production PICF-AQR-OWM default:
    # learned physical/task anchor queries read typed support memory, while
    # posterior correction remains the authoritative belief update. Disable
    # only for explicit ablations or legacy compatibility tests.
    aqr_mapg_enabled: bool = True
    aqr_query_count_physical: int = 16
    aqr_query_count_task: int = 8
    # Role layout for AQR graph queries. The default "structured" keeps the
    # historical effector/object/contact/context split. The object-only probe
    # disables effector/contact graph roles so mask/contact evidence can test
    # object-owner binding without a blue effector row competing for the same
    # object. This is a graph role contract, not a visualization flag.
    aqr_role_layout: str = "structured"
    aqr_query_rounds: int = 2
    aqr_sinkhorn_iters: int = 6
    aqr_sinkhorn_temperature: float = 0.2
    # PaliGemma language remains active through task-query semantic conditioning.
    # The heatmap/grounding head is off by default for AQR because where should
    # be learned by query-to-support attention rather than inherited from weak
    # VLM heatmaps. Enable this only for explicit diagnostics or ablations.
    aqr_pg_grounding_enabled: bool = False
    # PaliGemma image tokens can still assist localization as typed
    # visual-semantic support. This is not the heatmap head: task queries read
    # PaliGemma image tokens and project that support onto the V-JEPA grid.
    aqr_pg_image_support_enabled: bool = True
    aqr_pg_image_support_weight: float = 0.35
    aqr_pg_entropy_threshold: float = 0.90
    aqr_pg_peak_threshold: float = 1.50
    aqr_pg_bias_weight: float = 0.0
    aqr_support_bias_clip: float = 4.0
    # Low-amplitude ownership prior that breaks same-role assignment symmetry
    # before AQR reads visual/temporal memory. This is an assignment prior, not
    # an auxiliary loss: if raw attention rows are identical, Sinkhorn cannot
    # create object ownership by itself.
    aqr_ownership_prior_enabled: bool = True
    aqr_ownership_prior_weight: float = 0.35
    aqr_ownership_point_prior_weight: float = 0.35
    aqr_ownership_point_prior_sigma_m: float = 0.04
    aqr_ownership_temporal_prior_weight: float = 0.20
    aqr_ownership_prior_uniform_mix: float = 0.05
    aqr_same_role_support_competition_enabled: bool = True
    aqr_same_role_support_competition_weight: float = 0.35
    aqr_same_role_support_competition_iters: int = 2
    aqr_same_role_support_competition_physical_only: bool = True
    # Capacity-aware active-slot selection. AQR keeps the fixed query set for
    # compatibility, but only distinct high-confidence same-role anchors are
    # eligible for downstream obs/task assignment; redundant anchors become
    # inactive/dustbin candidates instead of duplicating the same object.
    aqr_active_slot_filter_enabled: bool = True
    aqr_active_slot_min_per_role: int = 1
    aqr_active_slot_max_per_role: int = 4
    aqr_active_slot_min_confidence: float = 0.05
    aqr_active_slot_overlap_threshold: float = 0.75
    # Object-conditional dustbin router. This keeps the fixed AQR query set
    # but lets weak or duplicate same-role anchors become inactive candidates.
    aqr_active_slot_relative_score_threshold: float = 0.0
    aqr_active_slot_geometry_duplicate_enabled: bool = True
    aqr_active_slot_geometry_duplicate_sigma_m: float = 0.04
    aqr_active_slot_geometry_duplicate_threshold: float = 0.70
    # VCAP is the audited vNext variable-cardinality active-proposal layer.
    # It is disabled by default: enabling it changes only the active query
    # allocator, not dense typed memory, posterior correction, cache, or PI0.5.
    vcap_enabled: bool = False
    vcap_max_active: int = 12
    vcap_min_active: int = 1
    vcap_stop_threshold: float = 0.5
    vcap_action_grad_scale: float = 0.0
    # Three-state object routing. Active anchors are action-relevant object
    # owners, context anchors are real but lower-priority scene objects, and
    # reserve/dustbin anchors are duplicate/no-object capacity. Context anchors
    # keep background objects visible without forcing them to update posterior
    # object files as if they were the current target.
    aqr_context_slot_enabled: bool = True
    aqr_context_slot_weight: float = 0.15
    aqr_context_slot_min_confidence: float = 0.05
    aqr_context_slot_min_score: float = 0.01
    aqr_context_slot_duplicate_overlap_threshold: float = 0.75
    # Adaptive slot-quality selector.  This is the PICF-native analogue of
    # adaptive slot-count / slot-quality methods: fixed query capacity remains
    # available, but every row receives differentiable object/no-object/
    # duplicate scores that gate downstream evidence.  The learned head is
    # zero-initialized around deterministic sidecar/tracklet/contact evidence,
    # so enabling it is behavior-preserving before training.
    aqr_slot_quality_enabled: bool = True
    aqr_slot_quality_learned_enabled: bool = True
    aqr_slot_quality_learned_scale: float = 0.25
    aqr_slot_quality_floor: float = 0.05
    aqr_slot_quality_context_scale: float = 0.25
    aqr_slot_quality_duplicate_threshold: float = 0.50
    aqr_slot_quality_target_smoothing: float = 0.02
    # Posterior object-file ownership gate. The active-slot filter above
    # selects object owners from the fixed AQR query bank; this gate carries
    # that owner/reserve decision into posterior binding so inactive reserve
    # anchors go to dustbin instead of updating persistent object files.
    posterior_owner_active_gate_enabled: bool = True
    posterior_owner_active_min: float = 0.25
    posterior_owner_active_bias: float = -1.0e4
    aqr_vjepa_temporal_mode: str = "last_two_tokens"
    aqr_vjepa_temporal_tokens: int = 2
    aqr_vjepa_temporal_include_delta: bool = True
    vjepa_multiview_enabled: bool = True
    vjepa_views: tuple[str, ...] = ("static", "gripper")
    vjepa_max_views: int = 4
    aqr_obs_gate_init: float = 0.0
    aqr_task_gate_init: float = 0.0
    aqr_posterior_gate_init: float = -2.0
    aqr_control_gate_init: float = 0.0
    evidence_cache_enabled: bool = True
    evidence_cache_len: int = 4
    evidence_cache_read_weight: float = 0.05
    evidence_cache_innovation_downweight: float = 1.0
    evidence_cache_address_weight: float = 0.25
    evidence_cache_content_weight: float = 0.10
    evidence_cache_role_weight: float = 0.25
    tracklet_memory_enabled: bool = True
    tracklet_max_tokens: int = 256
    tracklet_confidence_floor: float = 0.05
    tracklet_read_weight: float = 0.25
    # Blind automatic SAM proposals are rejected for current training. Keep the
    # generic proposal sidecar/runtime path as explicit opt-in so inspected
    # contact/task/tracklet-aware proposals can be tested without letting blind
    # wall/robot/drawer fragments perturb production posterior training.
    proposal_memory_enabled: bool = False
    proposal_max_tokens: int = 128
    proposal_confidence_floor: float = 0.05
    proposal_read_weight: float = 0.0
    proposal_age_decay_steps: float = 8.0
    # When proposal memory is explicitly enabled, runtime routing applies a
    # soft geometry-quality prior before a proposal may influence task-owner or
    # projected-point support.
    proposal_shape_quality_enabled: bool = True
    proposal_shape_area_min: float = 0.002
    proposal_shape_area_max: float = 0.35
    proposal_shape_aspect_min: float = 0.20
    proposal_context_quality_power: float = 0.50
    proposal_point_bridge_weight: float = 0.0
    proposal_point_bridge_edge_tau: float = 0.02
    proposal_mask_point_tau: float = 0.025
    # Proposal-anchor reference seeding is the Deformable-DETR/DINO-style
    # transport leg for inspected task/contact proposal sidecars: top task-owner
    # proposals become bounded physical-row point references before posterior
    # correction. Dense typed memory is still read normally; this does not prune
    # V-JEPA/point/proposal tokens or turn sidecars into hard labels.
    proposal_anchor_seed_enabled: bool = False
    # When sidecar/contact proposals are explicitly enabled, their inspected
    # task-owner masks should initialize the object query before point reread,
    # not only correct geometry after AQR has already read the scene. This is
    # the PICF-native equivalent of SAVi mask/box conditioning and
    # Deformable-DETR/DINO reference-query initialization: proposal evidence is
    # still soft, typed, and posterior-gated, but the query is given the right
    # object-conditioned coordinate system before it competes for dense tokens.
    proposal_anchor_seed_pre_reader_enabled: bool = True
    proposal_anchor_seed_rows: int = 2
    proposal_anchor_seed_weight: float = 0.85
    proposal_anchor_seed_token_weight: float = 0.35
    proposal_anchor_seed_score_floor: float = 0.05
    proposal_anchor_seed_point_topk: int = 128
    proposal_anchor_seed_point_power: float = 1.5
    # Object-candidate-to-slot assignment is the full proposal-mask binding
    # leg: inspected sidecar masks become object measurement candidates, then
    # compete for physical scene slots with a background/no-object residual.
    # The result is a soft measurement prior, not a posterior label.
    object_candidate_assignment_enabled: bool = True
    object_candidate_assignment_temperature: float = 0.35
    object_candidate_background_prior: float = 0.25
    object_candidate_background_quality_weight: float = 2.0
    object_candidate_row_support_floor: float = 0.01
    # A sidecar/contact-motion candidate may need both an object file and a
    # contact/interaction bridge. Role 0 effector rows are deliberately excluded:
    # they carry robot/proprio/tactile context but must not become object owners.
    object_candidate_eligible_roles: tuple[int, ...] = (1, 2)
    object_candidate_max_rows_per_candidate: int = 2
    object_candidate_row_capacity: float = 1.25
    object_candidate_row_capacity_iters: int = 10
    object_candidate_point_weight: float = 1.0
    object_candidate_proposal_weight: float = 0.75
    object_candidate_seed_weight: float = 1.25
    object_candidate_task_owner_weight: float = 0.50
    object_candidate_anchor_score_weight: float = 1.0
    object_candidate_point_mix: float = 0.80
    object_candidate_proposal_mix: float = 0.35
    object_candidate_min_shape_quality: float = 0.01
    # Owner transport is the missing object-file leg: after a sidecar candidate
    # is accepted by object/contact assignment, the role-1 object file receives
    # a bounded point/mask geometry prior. Role-2 may bridge contact, but it
    # must not replace the task-object file as the spatial owner.
    object_candidate_owner_transport_enabled: bool = True
    object_candidate_owner_roles: tuple[int, ...] = (1,)
    object_candidate_owner_min_share: float = 0.65
    object_candidate_owner_point_mix: float = 1.0
    # Object Explanation Measurement Layer (OEML). This is the PICF-native
    # slot/OCL invariant: every dense typed evidence token is explained by
    # competing object anchors or a background/no-object residual before the
    # measurement is allowed to strengthen posterior assignment. This is not
    # a SAM label path and it does not prune dense V-JEPA/point/tactile memory.
    object_explanation_enabled: bool = True
    object_explanation_feed_quality_to_assignment: bool = True
    object_explanation_background_prior: float = 0.25
    object_explanation_min_slot_quality: float = 0.05
    object_explanation_duplicate_margin: float = 0.30
    object_explanation_point_sigma_m: float = 0.06
    object_explanation_contact_weight_floor: float = 0.05
    object_explanation_feature_eps: float = 1e-6
    task_owner_bias_enabled: bool = True
    task_owner_visual_bias_weight: float = 0.20
    task_owner_proposal_bias_weight: float = 0.0
    task_owner_proposal_point_bias_weight: float = 0.0
    task_owner_proposal_point_bridge_weight: float = 0.0
    task_owner_proposal_objectness_power: float = 0.50
    task_owner_proposal_static_only: bool = True
    task_owner_proposal_topk: int = 4
    task_owner_proposal_score_floor: float = 0.05
    bind_support_signature_weight: float = 0.50
    bind_embedding_signature_weight: float = 0.25
    bind_quadratic_signature_weight: float = 0.10
    bind_low_rank_signature_weight: float = 0.05
    binding_signature_dim: int = 128
    binding_low_rank_signature_rank: int = 16
    binding_signature_score_calibration_enabled: bool = True
    binding_signature_score_calibration_mode: str = "double_center_zscore"
    binding_signature_score_min_std: float = 0.05
    binding_signature_score_clip: float = 4.0
    # Object-binding papers indicate that same-object information is a
    # pairwise/projected subspace, not the raw common component of ViT tokens.
    # Center projected keys within each typed memory before support pooling so
    # a global scene/modality component cannot make all slot signatures nearly
    # collinear.
    binding_signature_centering_enabled: bool = True
    binding_signature_centering_min_tokens: int = 4
    # Posterior object-file identity is a latent belief-state descriptor. It
    # should not be overwritten by every instantaneous observation signature:
    # low-support/common-mode measurements keep the previous descriptor, while
    # trusted support, birth, or recycle events update it explicitly.
    posterior_binding_signature_memory_enabled: bool = True
    posterior_binding_signature_update_rate: float = 0.20
    posterior_binding_signature_update_max_rate: float = 0.50
    posterior_binding_signature_min_support: float = 0.02
    posterior_binding_signature_owner_weight: float = 0.50
    posterior_binding_signature_dispersion_gate_enabled: bool = True
    posterior_binding_signature_measurement_min_std: float = 0.05
    posterior_binding_signature_measurement_margin_min: float = 0.25
    posterior_binding_signature_measurement_margin_temperature: float = 0.10
    bind_address_weight: float = 0.25
    bind_address_innovation_downweight: float = 1.0
    address_update_rate: float = 0.05
    address_update_max_rate: float = 0.20
    recycle_normalize_residual_summary: bool = True
    recycle_residual_norm_mode: str = "layernorm"
    recycle_logit_clamp: float = 0.0
    # Recycle/reset is an object-file trust decision. A global dustbin residual
    # would reset multiple same-role scene slots into the same latent state.
    # Use each slot's own raw measurement mixture for recycle/reset and fall
    # back to the dustbin residual only when that slot has no support.
    posterior_slotwise_recycle_residual: bool = True
    # Lifecycle calibration factors object-file survival/reset/birth apart from
    # raw Sinkhorn dustbin mass. A stable slot with high support, high
    # assignment margin, active-owner reliability, and low innovation should not
    # be reset just because the learned recycle head is temporarily high.
    posterior_lifecycle_calibration_enabled: bool = True
    posterior_lifecycle_support_min: float = 0.05
    posterior_lifecycle_support_temperature: float = 0.05
    posterior_lifecycle_margin_min: float = 0.02
    posterior_lifecycle_margin_temperature: float = 0.05
    posterior_lifecycle_entropy_weight: float = 0.50
    posterior_lifecycle_owner_weight: float = 0.50
    posterior_lifecycle_innovation_downweight: float = 1.0
    # Persistent object files are fixed-capacity, but real scenes often contain
    # fewer active objects than slots. After posterior binding, duplicate
    # same-role files that explain the same measurement support are demoted to
    # no-object/dustbin instead of all updating to the same physical anchor.
    posterior_file_competition_enabled: bool = True
    posterior_file_competition_min_per_role: int = 1
    posterior_file_competition_max_per_role: int = 0
    posterior_file_competition_min_support: float = 0.02
    posterior_file_competition_relative_score_threshold: float = 0.0
    posterior_file_competition_support_overlap_threshold: float = 0.80
    posterior_file_competition_geometry_duplicate_enabled: bool = True
    posterior_file_competition_geometry_sigma_m: float = 0.04
    posterior_file_competition_geometry_threshold: float = 0.70
    # Owner-responsibility posterior closure.  Recent object-centric/tactile
    # slot systems use the same object responsibility that selected evidence to
    # update the persistent object state.  PICF keeps this as a soft
    # belief-filter measurement: accepted object/contact graph rows are
    # transported through obs->posterior assignments and fused into posterior
    # geometry only when reliability gates pass.
    posterior_owner_transport_enabled: bool = True
    posterior_owner_transport_roles: tuple[int, ...] = (1,)
    posterior_owner_transport_max_per_role: int = 1
    posterior_owner_transport_max_rate: float = 0.85
    posterior_owner_transport_precision_gain: float = 8.0
    posterior_owner_transport_min_mass: float = 0.01
    posterior_owner_transport_assignment_floor: float = 0.50
    posterior_owner_transport_reliability_floor: float = 0.50
    posterior_owner_transport_covariance_scale: float = 0.50
    posterior_owner_transport_inactive_prior: float = 0.35
    posterior_owner_transport_activates_file: bool = True
    posterior_owner_transport_active_threshold: float = 0.05
    # Birth/no-object competition after duplicate file demotion. File
    # competition decides which existing persistent files own current evidence;
    # this second transport decides which reserve files, if any, may consume the
    # remaining dustbin evidence as a new object birth. Without this layer, all
    # inactive same-role files can reset from the same dustbin residual and
    # become duplicate candidates again.
    posterior_birth_competition_enabled: bool = True
    posterior_birth_competition_max_per_role: int = 1
    posterior_birth_competition_min_score: float = 0.05
    posterior_birth_competition_inactive_only: bool = True
    posterior_birth_alpha_suppression_power: float = 0.5
    # Legacy archived local top-k reread. The 2026-05-13 diagnostics showed the
    # normalized recycle fix is the root repair, while this residual adds
    # recycle/gradient pressure. Keep the implementation for reproducible
    # ablations only; production config requires an explicit legacy opt-in.
    legacy_local_refinement_opt_in: bool = False
    local_refinement_enabled: bool = False
    local_refinement_topk: int = 0
    local_refinement_weight: float = 0.0
    # Optional same-object subspace reranking for the archived local refiner.
    # This must not be used as a production ownership rule without a fresh
    # explicit ablation and README update.
    local_refinement_binding_weight: float = 0.0
    slot_jepa_enabled: bool = True
    support_prediction_enabled: bool = True
    ordinal_relation_enabled: bool = True
    ordinal_weak_target_enabled: bool = True
    lambda_aqr_denoising: float = 0.0
    visual_reread_topk: int = 32
    tactile_reread_groups: int = 2
    task_visual_reread_topk: int = 32
    task_tactile_reread_groups: int = 2
    task_point_reread_topk: int = 32
    tokenwise_ff_chunk_size: int = 0
    require_pi0_action_generator: bool = True
    action_prefix_stopgrad: bool = False

    @property
    def point_occ_dim(self) -> int:
        return self.point_real_grid**3

    @property
    def point_latent_dim(self) -> int:
        return self.point_latent_tokens * self.hidden_dim

    @property
    def point_real_dim(self) -> int:
        return self.point_latent_dim + self.point_occ_dim

    @property
    def visual_latent_dim(self) -> int:
        return self.visual_latent_tokens * self.hidden_dim

    @property
    def visual_real_dim(self) -> int:
        return 3 * (self.visual_real_grid**2)

    @property
    def tactile_map_dim(self) -> int:
        return self.tactile_real_grid**2

    @property
    def tactile_latent_dim(self) -> int:
        return self.tactile_latent_tokens * self.hidden_dim

    @property
    def tactile_real_dim(self) -> int:
        return self.tactile_latent_dim + self.tactile_map_dim + self.tactile_aux_dim

    @property
    def a_min_m(self) -> tuple[float, float, float]:
        return (0.005, 0.005, 0.005)

    @property
    def a_max_m(self) -> tuple[float, float, float]:
        value = 2.0 * self.workspace_radius_m
        return (value, value, value)
