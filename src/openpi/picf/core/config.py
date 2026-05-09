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
    posterior_slot_identity_std: float = 0.0
    task_slot_identity_std: float = 0.0
    posterior_bootstrap_from_observation: bool = False
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
    tactile_anchor_prob_on: float = 0.8
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
    # Direct-final AQR-MAPG path. This replaces MAPG-v0 candidate priors with
    # task/role-conditioned anchor queries over typed support memory while
    # reusing the existing PICF graph-consumer interfaces.
    aqr_mapg_enabled: bool = False
    aqr_query_count_physical: int = 16
    aqr_query_count_task: int = 8
    aqr_query_rounds: int = 2
    aqr_sinkhorn_iters: int = 6
    aqr_sinkhorn_temperature: float = 0.2
    aqr_pg_entropy_threshold: float = 0.90
    aqr_pg_peak_threshold: float = 1.50
    aqr_pg_bias_weight: float = 1.0
    aqr_support_bias_clip: float = 4.0
    aqr_temporal_memory_tokens: int = 32
    aqr_obs_gate_init: float = 0.0
    aqr_task_gate_init: float = 0.0
    aqr_posterior_gate_init: float = -2.0
    aqr_control_gate_init: float = 0.0
    visual_reread_topk: int = 32
    tactile_reread_groups: int = 2
    task_visual_reread_topk: int = 32
    task_tactile_reread_groups: int = 2
    task_point_reread_topk: int = 32
    tokenwise_ff_chunk_size: int = 0
    require_pi0_action_generator: bool = True

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
