#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def _contains(source: str, *needles: str) -> bool:
    return all(needle in source for needle in needles)


def _contains_normalized(source: str, *needles: str) -> bool:
    normalized = " ".join(source.split())
    return all(needle in normalized for needle in needles)


def run_checks() -> list[Check]:
    picf_contracts = _read("src/openpi/picf/contracts.py")
    contracts = _read("src/openpi/picf/core/contracts.py")
    config = _read("src/openpi/picf/core/config.py")
    pipeline = _read("src/openpi/picf/core/pipeline.py")
    policy = _read("src/openpi/picf/policy.py")
    training = _read("src/openpi/picf/core/training.py")
    trainer = _read("scripts/picf_core_train.py")
    wrapper = _read("src/openpi/picf/vjepa/wrapper.py")
    evidence = _read("scripts/picf_owm_evidence_bundle.py")
    same_object_probe = _read("scripts/picf_owm_same_object_probe.py")
    nontruncated_audit = _read("scripts/picf_owm_nontruncated_paper_audit.py")
    binding_logit_audit = _read("scripts/picf_binding_logit_calibration_audit.py")
    binding_dataflow_audit = _read("scripts/picf_binding_dataflow_math_audit.py")
    posterior_signature_memory_audit = _read("scripts/picf_posterior_binding_signature_memory_audit.py")
    posterior_file_competition_audit = _read("scripts/picf_posterior_file_competition_audit.py")
    posterior_birth_transport_audit = _read("scripts/picf_posterior_birth_transport_audit.py")
    action_visible_reserve_audit = _read("scripts/picf_action_visible_reserve_gate_audit.py")
    object_candidate_audit = _read("scripts/picf_object_candidate_slot_binding_audit.py")
    anchor_run_report = _read("scripts/picf_anchor_run_diagnostic_report.py")
    professor_audit = _read("scripts/picf_owm_professor_grade_audit.py")
    full_binding_audit_doc = _read("docs/PICF_AQR_OWM_FULL_BINDING_MATH_DATAFLOW_AUDIT_20260516_TEMP.md")
    readme = _read("docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md")
    readme_v22 = _read("src/openpi/picf/README_v2.2.md")
    mvtrack_readme = _read("docs/PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md")
    vcap_readme = _read("docs/PICF_AQR_OWM_AR_ANCHOR_PROPOSAL_AUDIT_20260518_TEMP.md")
    object_candidate_doc = _read("docs/PICF_AQR_OWM_OBJECT_CANDIDATE_SLOT_BINDING_20260519_TEMP.md")
    professor_audit_doc = _read("docs/PICF_AQR_OWM_PROFESSOR_GRADE_INTERACTION_AUDIT_TEMP.md")
    burnin_body = pipeline.split("def recurrent_burnin_step", 1)[1].split("def _predictive_state", 1)[0]

    checks = [
        Check(
            "readme_definition_of_done_present",
            _contains(readme, "Final Definition Of Done", "posterior is the authoritative current belief state"),
            "README must define the final OWM acceptance contract and posterior authority.",
        ),
        Check(
            "contracts_expose_temporal_visual_support",
            _contains(contracts, "class PicfTemporalVisualSupportState", "temporal_visual: PicfTemporalVisualSupportState | None"),
            "Token field must expose typed recent V-JEPA support.",
        ),
        Check(
            "contracts_expose_fixed_evidence_cache",
            _contains(contracts, "class PicfEvidenceCacheState", "tokens: torch.Tensor", "slot_address: torch.Tensor", "age: torch.Tensor", "innovation_at_write: torch.Tensor"),
            "Evidence cache must be a fixed tensor state with address/age/innovation metadata.",
        ),
        Check(
            "contracts_expose_graph_owm_fields",
            _contains(
                contracts,
                "vjepa_temporal_priors",
                "cache_priors",
                "tracklet_priors",
                "local_priors",
                "slot_address",
                "slot_content",
                "support_uncertainty",
                "support_signature",
                "binding_signature",
                "binding_signature_quadratic_score_abs_mean",
                "binding_signature_low_rank_score_abs_mean",
                "tactile_evidence_mask",
                "tactile_evidence_weight",
            ),
            "Anchor graph must expose temporal/cache/address/content/uncertainty fields.",
        ),
        Check(
            "contracts_expose_mvtrack_states",
            _contains(
                contracts,
                "class PicfTrackletSupportState",
                "class PicfPseudoProposalState",
                "class PicfCacheReadState",
                "visual_signature",
                "tracklet_signature",
                "proposal_signature",
                "ordinal_target_rank",
            ),
            "MVTrack contracts must expose typed tracklets, cache read metadata, support signatures, and weak ordinal targets.",
        ),
        Check(
            "contracts_expose_object_explanation_state",
            _contains(
                contracts,
                "class PicfObjectExplanationState",
                "object_mask_visual",
                "background_mask_visual",
                "anchor_duplicate_overlap",
                "contact_explanation_score",
                "object_explanation: PicfObjectExplanationState | None",
            ),
            "OEML must expose object/background explanation state without replacing posterior authority.",
        ),
        Check(
            "vjepa_recent_maps_preserves_time",
            _contains(
                wrapper,
                "def recent_maps",
                "without averaging time",
                "return tokens[-min(count, int(tokens.shape[0])) :]",
                "return tokens_np[-min(count, int(tokens_np.shape[0])) :]",
            ),
            "V-JEPA wrapper must expose recent maps without averaging.",
        ),
        Check(
            "pipeline_builds_object_explanation_before_assignment",
            _contains(
                pipeline,
                "def _build_object_explanation_measurements",
                "def _object_explanation_masks",
                "graph.object_explanation_quality = explanation_quality",
                "object_explanation_feed_quality_to_assignment",
                "object_explanation=object_explanation",
                "oeml_anchor_quality_mean",
            ),
            "AQR runtime must build OEML measurements and feed explanation quality into assignment/metrics.",
        ),
        Check(
            "training_exposes_guarded_object_explanation_losses",
            _contains(
                training,
                "lambda_object_explanation_feature: float = 0.0",
                "def _object_explanation_loss",
                "object_explanation_feature",
                "object_explanation_duplicate",
            )
            and _contains(
                trainer,
                "--lambda-object-explanation-feature",
                "loss_object_explanation_feature",
                "oeml_anchor_quality_mean",
            ),
            "OEML training pressure must be explicit, guarded, and visible in train logs.",
        ),
        Check(
            "config_defaults_owm_graph_enabled",
            _contains(
                config,
                "aqr_mapg_enabled: bool = True",
                "mapg_enabled: bool = False",
                "vl_anchor_router_enabled: bool = False",
                "aqr_pg_grounding_enabled: bool = False",
                "aqr_pg_image_support_enabled: bool = True",
                'aqr_vjepa_temporal_mode: str = "last_two_tokens"',
                "evidence_cache_enabled: bool = True",
                "evidence_cache_read_weight: float = 0.05",
                "posterior_slot_identity_std: float = 0.02",
                "task_slot_identity_std: float = 0.02",
                "posterior_bootstrap_from_observation: bool = True",
                "posterior_occupancy_prior_enabled: bool = True",
                "posterior_occupancy_prior_weight: float = 1.0",
                "posterior_occupancy_prior_sigma_m: float = 0.04",
                "posterior_occupancy_prior_clip: float = 4.0",
                "observation_anchor_seed_point_mix: float = 0.35",
                "slot_jepa_enabled: bool = True",
                "support_prediction_enabled: bool = True",
                "ordinal_relation_enabled: bool = True",
                "vjepa_multiview_enabled: bool = True",
                "tracklet_memory_enabled: bool = True",
                "proposal_memory_enabled: bool = False",
                "proposal_read_weight: float = 0.0",
                "proposal_point_bridge_weight: float = 0.0",
                "task_owner_proposal_bias_weight: float = 0.0",
                "bind_support_signature_weight",
                "bind_embedding_signature_weight",
                "bind_quadratic_signature_weight",
                "bind_low_rank_signature_weight",
                "binding_signature_dim",
                "binding_low_rank_signature_rank",
                "binding_signature_score_calibration_enabled",
                "binding_signature_score_calibration_mode",
                "binding_signature_score_min_std",
                "binding_signature_score_clip",
                "binding_signature_centering_enabled",
                "binding_signature_centering_min_tokens",
                "posterior_binding_signature_memory_enabled: bool = True",
                "posterior_binding_signature_update_rate",
                "posterior_binding_signature_update_max_rate",
                "posterior_binding_signature_min_support",
                "posterior_binding_signature_owner_weight",
                "posterior_binding_signature_dispersion_gate_enabled: bool = True",
                "posterior_binding_signature_measurement_min_std",
                "posterior_binding_signature_measurement_margin_min",
                "posterior_binding_signature_measurement_margin_temperature",
                "aqr_ownership_prior_enabled",
                "aqr_ownership_prior_weight",
                "aqr_ownership_point_prior_weight",
                "aqr_ownership_point_prior_sigma_m",
                "aqr_ownership_temporal_prior_weight",
                "aqr_same_role_support_competition_enabled",
                "aqr_same_role_support_competition_weight",
                "aqr_active_slot_filter_enabled",
                "aqr_active_slot_overlap_threshold",
                "aqr_active_slot_relative_score_threshold",
                "aqr_active_slot_geometry_duplicate_enabled",
                "local_refinement_binding_weight",
                "recycle_normalize_residual_summary",
                "recycle_residual_norm_mode",
                "posterior_slotwise_recycle_residual: bool = True",
                "posterior_lifecycle_calibration_enabled: bool = True",
                "posterior_lifecycle_support_min",
                "posterior_lifecycle_margin_min",
                "posterior_birth_competition_enabled: bool = True",
                "posterior_birth_competition_max_per_role",
                "posterior_birth_competition_min_score",
                "action_prefix_stopgrad",
                "evidence_cache_address_weight",
                "object_explanation_enabled: bool = True",
                "object_explanation_feed_quality_to_assignment: bool = True",
                "object_explanation_background_prior",
            )
            and _contains(
                trainer,
                "def _looks_like_legacy_blind_sam_sidecar",
                "--allow-legacy-blind-sam-sidecar",
                "Blind automatic SAM proposals are rejected",
            ),
            "Config must default to the direct-final OWM graph profile, with legacy routers off, typed evidence on, and blind-SAM roots rejected by default.",
        ),
        Check(
            "vcap_contract_is_disabled_guarded_and_audited",
            _contains(
                config,
                "vcap_enabled: bool = False",
                "vcap_max_active: int = 12",
                "vcap_min_active: int = 1",
                "vcap_stop_threshold: float = 0.5",
                "vcap_action_grad_scale: float = 0.0",
            )
            and "vcap_count_prior_weight" not in config
            and "vcap_unexplained_weight" not in config
            and "vcap_duplicate_weight" not in config
            and "vcap_continuity_weight" not in config
            and "vcap_teacher_forcing_steps" not in config
            and "vcap_free_run_warmup_steps" not in config
            and _contains(
                contracts,
                "class PicfActiveProposalState",
                "active_proposals",
                "proposal_to_graph_assignment",
                "proposal_unexplained_evidence",
                "proposal_duplicate_cost",
                "proposal_count",
            )
            and _contains(
                pipeline,
                "def _vcap_active_proposal_queries",
                "def _finalize_vcap_proposal_state",
                "VCAP is not allowed to prune or replace dense typed memory",
                "active_forward = active_hard + (active_t - active_t.detach())",
                "vcap_matched_old_file_fraction",
                "vcap_birth_fraction",
                "vcap_noobject_fraction",
            )
            and _contains(
                training,
                "lambda_vcap_unexplained",
                "lambda_vcap_duplicate",
                "lambda_vcap_count",
                "lambda_vcap_continuity",
                "def _vcap_auxiliary_loss",
                "guarded_owm_aux",
            )
            and _contains(
                trainer,
                "--vcap-enabled",
                "--vcap-max-active",
                "--vcap-stop-threshold",
                "--vcap-action-grad-scale",
                "--lambda-vcap-unexplained",
                "loss_vcap",
                "vcap_proposal_count",
                "vcap_noobject_fraction",
            )
            and _contains(
                readme_v22,
                "disabled-by-default runtime prototype",
                "keeps dense typed memory intact",
                "VCAP is not the production default",
                "stop-token-only or count-loss-only",
            )
            and _contains(
                vcap_readme,
                "disabled-by-default runtime prototype is present",
                "Runtime config must contain only knobs consumed by the proposal allocator",
                "Training-pressure weights live in `PicfTransitionLossConfig`",
                "vcap_action_grad_guard",
                "vcap_no_dense_memory_prune",
            ),
            "VCAP must be disabled, step through AQR/posterior, expose health metrics, and avoid dead runtime knobs.",
        ),
        Check(
            "posterior_birth_transport_prevents_dustbin_broadcast",
            _contains(
                contracts,
                "file_competition_birth_active",
                "file_competition_birth_share",
            )
            and _contains(
                pipeline,
                "def _posterior_birth_competition",
                "high-reset, low-alpha files can become birth candidates",
                "birth_share = birth_recycle /",
                "binding_support = support_raw + (birth_share[:, None] * dustbin_raw[None, :])",
                "recycle_update = recycle * recycle_update_mask",
                "file_competition_birth_active=birth_active",
                "posterior_file_competition_birth_count",
            )
            and _contains(
                posterior_birth_transport_audit,
                "many_inactive_same_role_only_one_birth",
                "per_role_births_not_global_broadcast",
                "all_scores_below_threshold_keeps_dustbin",
                "mass conservation",
            ),
            "Posterior lifecycle must not broadcast one dustbin residual into every inactive duplicate file.",
        ),
        Check(
            "trainer_defaults_match_owm_profile",
            _contains(
                trainer,
                "_LOSS_DEFAULTS = PicfTransitionLossConfig()",
                'default="paligemma"',
                "default=_SPEC_DEFAULTS.aqr_mapg_enabled",
                "default=_SPEC_DEFAULTS.binding_signature_centering_enabled",
                "default=_SPEC_DEFAULTS.binding_signature_score_calibration_enabled",
                "default=_SPEC_DEFAULTS.posterior_binding_signature_memory_enabled",
                "default=_SPEC_DEFAULTS.posterior_binding_signature_dispersion_gate_enabled",
                "--bind-quadratic-signature-weight",
                "--bind-low-rank-signature-weight",
                "--binding-low-rank-signature-rank",
                "--binding-signature-score-calibration-mode",
                "--posterior-binding-signature-memory-enabled",
                "--posterior-binding-signature-update-rate",
                "--posterior-binding-signature-dispersion-gate-enabled",
                "binding_signature_centering_enabled=bool",
                "binding_signature_score_calibration_enabled=bool",
                "posterior_binding_signature_memory_enabled=bool",
                "posterior_binding_signature_dispersion_gate_enabled=bool",
                "posterior_lifecycle_calibration_enabled=bool",
                "--posterior-lifecycle-calibration-enabled",
                "posterior_birth_competition_enabled=bool",
                "--posterior-birth-competition-enabled",
                "--posterior-birth-competition-max-per-role",
                "Posterior birth transport contract",
                "default=_LOSS_DEFAULTS.lambda_mapg_cycle",
                "default=_LOSS_DEFAULTS.lambda_mapg_support_diversity",
                "default=_LOSS_DEFAULTS.lambda_slot_jepa",
            ),
            "Trainer CLI and fallback loss construction must inherit the same production defaults as the dataclass contracts.",
        ),
        Check(
            "trainer_anchor_overlay_diagnostic_is_side_effect_safe",
            _contains(
                trainer,
                "--anchor-overlay-interval",
                "--anchor-overlay-dump-signatures",
                "_save_anchor_overlay_diagnostic",
                "_anchor_overlay_snapshot_from_output",
                "capture_anchor_overlay_signatures",
                "binding_signature",
                "with_gray",
                "active_only",
                "sidecar_proposals",
                "proposal_age",
                "image_variants",
                "capture_anchor_overlay_step",
                "anchor_overlays",
                "This is a training diagnostic only and does not change the loss or forward path.",
                "and not capture_anchor_overlay_step",
            ),
            "Anchor position overlays must reuse the real training forward rather than run an extra buffer-mutating diagnostic forward.",
        ),
        Check(
            "sparse_proposal_sidecars_are_age_aware_not_silent_missing",
            _contains(
                trainer,
                "--mvtrack-sidecar-proposal-nearest-max-gap",
                "proposal_nearest_max_gap",
                "proposal_age_decay_steps",
                "Borrowed proposal objectness is exponentially decayed",
            )
            and _contains(picf_contracts, "proposal_age")
            and _contains(contracts, "class PicfPseudoProposalState", "age")
            and _contains(pipeline, "torch.exp(-proposal_age / age_decay)"),
            "Sparse proposal sidecars must be explicit age-aware weak evidence rather than silently disappearing from overlay/train frames.",
        ),
        Check(
            "same_object_probe_reads_eval_and_training_artifacts",
            _contains(
                same_object_probe,
                "--anchor-debug",
                "--anchor-overlays",
                "--quadratic-probe",
                "diag_quadratic",
                "low_rank_quadratic",
                "full_quadratic",
                "anchor_overlay_json",
                "binding_signature_cos_auc",
                "trained_quadratic_probes",
                "duplicate_candidate_fraction_within_frame",
                "binding_subspace_decodable",
            ),
            "IsSameObject-style audits must run on both evaluation anchor_debug and training anchor_overlays.",
        ),
        Check(
            "nontruncated_paper_audit_checks_provenance_and_no_weak_loss_leak",
            _contains(
                nontruncated_audit,
                "vit-object-binding",
                "slotcontrast",
                "014c66b45ea262f9b6eec83ff388a1e1c10dfcaa",
                "external_vit_binding_code_is_inspected_and_not_copied",
                "picf_probe_implements_full_quadratic_family",
                "training_overlays_export_signatures_without_extra_forward",
                "weak_same_object_probe_is_not_online_training_loss",
                "binding_signature_is_runtime_binding_evidence_not_json_only",
                "runtime_quadratic_binding_is_native_not_paper_code_copy",
            ),
            "Non-truncated audit must verify paper-code provenance, native quadratic probes, training overlay signature dataflow, runtime binding use, and no weak-label online loss leak.",
        ),
        Check(
            "binding_logit_calibration_audit_checks_common_mode_and_relative_pairs",
            _contains(
                binding_logit_audit,
                "math_common_mode_maps_to_zero",
                "math_relative_pairs_survive_calibration",
                "binding_signature_score_calibration_enabled",
                "posterior_binding_signature_calibrated_score_std",
            ),
            "Binding-logit calibration must be guarded by an executable script that rejects common-mode identity evidence and preserves relative pair structure.",
        ),
        Check(
            "binding_dataflow_math_audit_checks_paper_code_runtime_and_matrix_math",
            _contains(
                binding_dataflow_audit,
                "paper_quadratic_probe_family_present",
                "paper_pairwise_bce_calibration_present",
                "picf_observation_to_posterior_signature_dataflow",
                "math_row_column_bias_zero",
                "math_low_dispersion_rejected",
            )
            and _contains(
                readme_v22,
                "PICF_AQR_OWM_BINDING_DATAFLOW_MATH_FOLLOWTHROUGH_20260515_TEMP.md",
            ),
            "Binding dataflow/math audit must connect paper code, PICF runtime, matrix calibration, and README routing.",
        ),
        Check(
            "posterior_binding_signature_memory_audit_checks_state_update",
            _contains(
                posterior_signature_memory_audit,
                "math_low_trust_keeps_previous_signature",
                "math_birth_or_recycle_resets_to_instant_signature",
                "math_trusted_measurement_moves_but_does_not_jump",
                "math_common_mode_measurement_is_rejected_by_dispersion_gate",
                "math_relative_measurement_can_update_when_dispersed",
                "posterior_binding_signature_memory_enabled",
                "posterior_binding_signature_dispersion_gate_enabled",
                "posterior_binding_signature_update_rate_mean",
            )
            and _contains(readme_v22, "posterior_file_continuity_metric_followthrough.md"),
            "Posterior file signatures must be audited as trusted latent state, not per-frame overwrite.",
        ),
        Check(
            "anchor_run_diagnostic_separates_raw_and_active_overlap",
            _contains(
                anchor_run_report,
                "raw same-role support overlap is high while active-owner overlap is low",
                "identity switch remains high despite healthy active-owner overlap",
                "posterior_active_file_potential_swap_rate",
                "posterior_active_file_calibrated_potential_swap_rate",
                "posterior active-file continuity is healthy",
                "raw common-mode",
                "posterior_binding_signature_calibrated_score_std",
                "active-owner/posterior-object-file continuity",
            )
            and _contains(readme_v22, "picf_anchor_run_diagnostic_report.py", "active-owner/posterior-object-file continuity"),
            "Run diagnostics must separate reserve/raw overlap from active-owner overlap before proposing new model changes.",
        ),
        Check(
            "pipeline_builds_temporal_tokens_and_priors",
            _contains(pipeline, "def _visual_maps", "PicfTemporalVisualSupportState", "vjepa_temporal_priors", "temporal_visual_reader", "view_ids"),
            "Pipeline must construct temporal support and route AQR over it.",
        ),
        Check(
            "pipeline_routes_mvtrack_branches",
            _contains(
                pipeline,
                "self.clip_buffers",
                "PicfTrackletSupportState",
                "aqr_tracklet_reader",
                "aqr_proposal_reader",
                "tracklet_priors",
                "proposal_priors",
                "task_owner_visual_prior",
                "task_owner_proposal_score",
                "graph_tracklet_weights",
                "graph_proposal_weights",
                "bind_support_signature_weight",
                "bind_address_weight",
                "legacy_local_refinement_opt_in",
                "local_priors",
                "ordinal_target_rank",
                "binding_signature_proj",
                "bind_embedding_signature_weight",
                "binding_signature_centering_enabled",
                "projected - projected.mean",
                "posterior_slotwise_recycle_residual",
                "_posterior_occupancy_binding_bias",
                "posterior_occupancy_prior_weight",
                "posterior_occupancy_prior_clip",
                "observation_anchor_seed_point_mix",
                "seed_point_priors",
                "slot_residual_summary",
                "raw_cond = support_raw / raw_denom",
                "measurement_summary = binding_cond @ obs_anchors.tokens",
                "self.residual_mu_head(measurement_summary)",
                "_aqr_visual_ownership_bias",
                "_task_owner_visual_prior",
                "_task_owner_visual_bias",
                "_proposal_scores_from_visual_prior",
                "_task_owner_proposal_bias",
                "_aqr_temporal_ownership_bias",
                "_aqr_active_slot_mask",
                "aqr_ownership_prior_weight",
                "recycle_normalize_residual_summary",
            ),
            "Pipeline must route multiview temporal, tracklet, support-signature binding, ownership priors, archived local refinement, and weak ordinal states.",
        ),
        Check(
            "pipeline_ownership_prior_breaks_same_role_symmetry_before_sinkhorn",
            _contains(
                pipeline,
                "def _aqr_ownership_priors_from_coords",
                "def _aqr_visual_ownership_bias",
                "def _aqr_point_ownership_bias",
                "def _aqr_temporal_ownership_bias",
                "visual_bias = ownership_bias if visual_bias is None else (visual_bias + ownership_bias)",
                "point_ownership_bias = self._aqr_point_ownership_bias(token_field, roles)",
                "attn_bias=temporal_bias",
                "def _object_core_overlap_matrix",
            )
            and _contains(
                trainer,
                "--aqr-ownership-prior-enabled",
                "--aqr-ownership-prior-weight",
                "--aqr-ownership-point-prior-weight",
                "--aqr-ownership-point-prior-sigma-m",
                "--aqr-ownership-temporal-prior-weight",
                "--aqr-active-slot-filter-enabled",
                "--aqr-active-slot-overlap-threshold",
                "ownership_prior_enabled",
            ),
            "AQR must include a low-amplitude assignment prior that breaks identical same-role support rows before Sinkhorn/diversity losses.",
        ),
        Check(
            "pipeline_active_slot_filter_adds_capacity_aware_dustbin_path",
            _contains(
                contracts,
                "anchor_active",
            )
            and _contains(
                pipeline,
                "def _aqr_active_slot_mask",
                "def _observation_owner_active_from_graph",
                "def _posterior_owner_active_binding_bias",
                "active_slot_filter_enabled",
                "posterior_owner_active_gate_enabled",
                "aqr_active_slot_relative_score_threshold",
                "aqr_active_slot_geometry_duplicate_threshold",
                "aqr_active_same_role_support_overlap_max",
                "posterior_owner_active_eligible_fraction",
                "aqr_inactive_anchor_fraction",
                "anchor_downstream_weight",
                "def _aqr_downstream_slot_weights",
                "graph_tokens = graph_tokens * graph_weight",
            )
            and _contains(
                trainer,
                "--aqr-active-slot-filter-enabled",
                "--aqr-active-slot-min-per-role",
                "--aqr-active-slot-max-per-role",
                "--aqr-active-slot-relative-score-threshold",
                "--aqr-active-slot-geometry-duplicate-enabled",
                "--posterior-owner-active-gate-enabled",
                "--posterior-slotwise-recycle-residual",
            ),
            "AQR must distinguish active object slots from inactive/dustbin anchors instead of forcing all fixed queries to bind objects.",
        ),
        Check(
            "task_owner_bias_routes_semantics_to_physical_object_proposals",
            _contains(
                config,
                "task_owner_bias_enabled",
                "task_owner_visual_bias_weight",
                "task_owner_proposal_bias_weight",
                "task_owner_proposal_point_bias_weight",
                "task_owner_proposal_objectness_power",
                "proposal_shape_quality_enabled",
                "proposal_context_quality_power",
                "task_owner_proposal_topk",
                "task_owner_proposal_score_floor",
                "proposal_point_bridge_weight",
                "proposal_point_bridge_edge_tau",
                "proposal_mask_point_tau",
                "proposal_anchor_seed_enabled",
                "proposal_anchor_seed_weight",
                "proposal_anchor_seed_point_topk",
                "task_owner_proposal_point_bridge_weight",
            )
            and _contains(
                contracts,
                "task_owner_visual_prior",
                "task_owner_proposal_score",
                "proposal_point_priors",
                "task_owner_point_priors",
                "proposal_anchor_seed_priors",
                "proposal_anchor_seed_assignment",
                "task_owner_anchor_score",
            )
            and _contains(
                pipeline,
                "def _task_owner_query_rows",
                "def _task_owner_physical_rows",
                "def _proposal_shape_quality",
                "def _postprocess_task_owner_proposal_score",
                "def _proposal_scores_from_visual_prior",
                "def _proposal_to_point_matrix",
                "def _proposal_priors_to_point_priors",
                "proposal.mask_xy",
                "proposal.mask_weights",
                "proposal.mask_offsets",
                "def _proposal_anchor_seed_transport",
                "def _task_owner_proposal_point_bias",
                "def _task_owner_proposal_to_point_priors",
                "def _task_owner_anchor_score",
                "owner_proposal_bias",
                "owner_point_bias",
                "proposal_point_bridge_weight",
                "task_owner_proposal_point_bias_weight",
                "task_owner_proposal_point_bridge_weight",
                "aqr_proposal_anchor_seed_point_max",
                "aqr_proposal_point_bridge_max",
                "aqr_task_owner_point_bridge_max",
                "aqr_task_owner_anchor_score_max",
                "aqr_task_owner_proposal_score_max",
                "aqr_proposal_shape_quality_mean",
                "aqr_task_owner_proposal_selected_count",
                "proposal_priors=proposal_priors",
            )
            and _contains(
                trainer,
                "--task-owner-bias-enabled",
                "--task-owner-visual-bias-weight",
                "--task-owner-proposal-bias-weight",
                "--task-owner-proposal-point-bias-weight",
                "--task-owner-proposal-objectness-power",
                "--proposal-shape-quality-enabled",
                "--proposal-context-quality-power",
                "--task-owner-proposal-topk",
                "--task-owner-proposal-score-floor",
                "--proposal-point-bridge-weight",
                "--proposal-point-bridge-edge-tau",
                "--proposal-mask-point-tau",
                "--proposal-anchor-seed-enabled",
                "--proposal-anchor-seed-weight",
                "--proposal-anchor-seed-point-topk",
                "--task-owner-proposal-point-bridge-weight",
                "aqr_proposal_anchor_seed_point_max",
                "aqr_proposal_point_bridge_max",
                "aqr_task_owner_point_bridge_max",
                "aqr_task_owner_anchor_score_max",
                "aqr_task_owner_proposal_score_max",
            ),
            "Task-conditioned ownership must softly connect task visual support to proposal and 3D point evidence without hard labels.",
        ),
        Check(
            "sidecar_masks_route_through_object_candidate_slot_assignment",
            _contains(
                contracts,
                "object_candidate_assignment",
                "object_candidate_coverage",
                "object_candidate_background",
                "object_candidate_duplicate_overlap",
                "candidate_coverage",
                "candidate_background",
                "candidate_duplicate_overlap",
            )
            and _contains(
                config,
                "object_candidate_assignment_enabled: bool = True",
                "object_candidate_assignment_temperature",
                "object_candidate_background_prior",
                "object_candidate_background_quality_weight",
                "object_candidate_row_support_floor",
                "object_candidate_max_rows_per_candidate",
                "object_candidate_row_capacity",
                "object_candidate_row_capacity_iters",
                "object_candidate_point_weight",
                "object_candidate_proposal_weight",
                "object_candidate_seed_weight",
                "object_candidate_task_owner_weight",
                "object_candidate_anchor_score_weight",
                "object_candidate_point_mix",
                "object_candidate_proposal_mix",
                "object_candidate_min_shape_quality",
            )
            and _contains(
                pipeline,
                "def _proposal_object_candidate_assignment",
                "object candidates and lets physical scene slots compete",
                "background residual absorbing",
                "row_specific",
                "Do not create arbitrary row symmetry from a task-level proposal alone.",
                "candidate_point_priors",
                "candidate_proposal_priors",
                "object_candidate_point_mix",
                "object_candidate_proposal_mix",
                "object_candidate_anchor_score_weight",
                "object_candidate_max_rows_per_candidate",
                "object_candidate_row_capacity",
                "object_candidate_row_capacity_iters",
                "object_candidate_assignment=object_candidate_assignment",
                "candidate_coverage=graph.object_candidate_coverage",
                "aqr_object_candidate_coverage_mean",
                "oeml_candidate_coverage_mean",
            )
            and _contains(
                trainer,
                "--object-candidate-assignment-enabled",
                "--object-candidate-assignment-temperature",
                "--object-candidate-background-prior",
                "--object-candidate-background-quality-weight",
                "--object-candidate-row-support-floor",
                "--object-candidate-max-rows-per-candidate",
                "--object-candidate-row-capacity",
                "--object-candidate-row-capacity-iters",
                "--object-candidate-point-weight",
                "--object-candidate-proposal-weight",
                "--object-candidate-seed-weight",
                "--object-candidate-task-owner-weight",
                "--object-candidate-anchor-score-weight",
                "--object-candidate-point-mix",
                "--object-candidate-proposal-mix",
                "aqr_object_candidate_assignment_max",
                "aqr_object_candidate_duplicate_overlap_max",
            )
            and _contains(
                training,
                "anchor_pv_active_object_gate_only: bool = True",
                "anchor_pv_object_normalize_by_object_mass: bool = True",
                "anchor_pv_object_distribution_loss: bool = True",
                "def _object_projective_distribution_loss",
                "aqr_denoising_active_object_only: bool = True",
                "aqr_denoising_confirmed_object_only: bool = True",
                "object_explanation_active_object_only: bool = True",
                "def _active_object_row_weight",
                "def _confirmed_object_row_weight",
                "Fixed-capacity slot systems deliberately keep reserve/no-object capacity",
            )
            and _contains(
                trainer,
                "--anchor-pv-active-object-gate-only",
                "--anchor-pv-object-normalize-by-object-mass",
                "--anchor-pv-object-distribution-loss",
                "--aqr-denoising-active-object-only",
                "--aqr-denoising-confirmed-object-only",
                "--object-explanation-active-object-only",
            )
            and _contains_normalized(
                readme_v22,
                "object-candidate slot-binding update",
                "candidate object masks must be explained by physical scene slots or by an explicit background/no-object residual",
                "PICF_AQR_OWM_OBJECT_CANDIDATE_SLOT_BINDING_20260519_TEMP.md",
            )
            and _contains(
                object_candidate_doc,
                "object candidate p must be explained by object slot j or by background/no-object",
                "SlotAttention.step",
                "sidecar object candidates compete for physical scene slots",
                "explicit background residual absorbs invalid/noisy candidates",
                "top-k candidate ownership",
                "row capacity",
                "A_{j,p}",
                "B_p",
                "PicfObservation.proposal_mask_*",
                "_proposal_object_candidate_assignment(...)",
                "aqr_object_candidate_coverage_mean",
                "object_candidate_duplicate_overlap",
            )
            and _contains(
                object_candidate_audit,
                "task_owner_only_cannot_clone_to_all_slots",
                "candidate_columns_conserve_slot_background_mass",
                "row_specific_support_assigns_distinct_candidates",
                "duplicate_candidate_explanations_are_visible",
                "candidate_top1_suppresses_raw_same_candidate_clones",
                "row_capacity_limits_one_slot_from_eating_all_candidates",
                "active_object_scope_ignores_reserve_duplicates",
                "downstream_weight_fallback_excludes_context_rows",
                "denoising_active_object_scope_excludes_no_object_peaks",
                "denoising_confirmed_object_scope_excludes_unconfirmed_active_rows",
                "object_pv_normalizes_by_confirmed_object_mass_not_dense_floor",
                "candidate_mask_transport_preserves_object_support",
                "runtime_scale_candidate_not_absorbed_by_background",
                "task_quality_still_needs_row_specific_support",
            ),
            "Sidecar masks must become soft object candidates that compete for physical slots with a background residual, not remain weak proposal-only hints.",
        ),
        Check(
            "action_visible_reserve_gate_has_executable_audit",
            _contains(
                action_visible_reserve_audit,
                "math_tristate_gate_routes_active_context_reserve",
                "math_background_context_survives_reserve_gate",
                "pipeline_routes_graph_downstream_weight",
                "overlay_exports_dual_views",
            )
            and _contains(
                readme_v22,
                "PICF_AQR_OWM_PROFESSOR_GRADE_BINDING_FOLLOWTHROUGH_20260516_TEMP.md",
            ),
            "Reserve/dustbin files must have an executable audit proving inactive rows are not action-visible while background context is retained.",
        ),
        Check(
            "pipeline_posterior_file_competition_demotes_duplicate_files",
            _contains(
                contracts,
                "file_competition_active",
                "file_competition_demoted_mass",
                "file_competition_duplicate_overlap_max",
                "file_competition_active_duplicate_overlap_max",
            )
            and _contains(
                config,
                "posterior_file_competition_enabled",
                "posterior_file_competition_support_overlap_threshold",
                "posterior_file_competition_geometry_duplicate_enabled",
            )
            and _contains(
                pipeline,
                "def _posterior_file_competition",
                "same-role",
                "Move duplicate persistent file assignments into no-object dustbin",
                "dustbin_active = dustbin + demoted.sum(dim=0)",
                "file_competition = self._posterior_file_competition",
                "posterior_file_competition_active_count",
                "posterior_file_competition_active_duplicate_overlap_max",
                "_posterior_file_active_gate",
                "posterior_active_file_recycle_rate",
            )
            and _contains(
                trainer,
                "--posterior-file-competition-enabled",
                "posterior_file_competition_active_count",
                "posterior_file_competition_demoted_mass_mean",
                "posterior_file_competition_active_duplicate_overlap_max",
                "file_competition_active",
                "file_competition_demoted_mass",
            )
            and _contains(
                readme_v22,
                "PICF_AQR_OWM_POSTERIOR_FILE_COMPETITION_FOLLOWTHROUGH_20260516_TEMP.md",
                "duplicate persistent files demoted to no-object/dustbin",
            ),
            "Posterior update must include a no-object/file-competition step so multiple same-role persistent files cannot all update from one duplicated observation owner.",
        ),
        Check(
            "tactile_dense_patch_tokens_project_to_hidden_memory",
            _contains(
                pipeline,
                "self.tactile_patch_token_proj = nn.LazyLinear(hidden_dim)",
                "self.tactile_patch_token_proj(dense_tokens)",
                "self.tactile_patch_token_proj(sensor.tokens.to",
                "tactile_evidence_mask",
                "tactile_evidence_weight",
                "tactile_active_rows",
                "tactile_weights[tactile_active_rows]",
            )
            and _contains(
                trainer,
                "core.tactile_patch_token_proj.*",
                "tactile_patch_token_proj.*",
                "--tactile-evidence-prob-floor",
            ),
            "Dense AnyTouch patch tokens must be projected into PICF hidden memory before route/target rereads; soft tactile evidence must be explicit and configurable.",
        ),
        Check(
            "posterior_file_competition_has_executable_math_audit",
            _contains(
                posterior_file_competition_audit,
                "same_support_duplicate_demotes_one_file",
                "measurement_mass_is_conserved",
                "distinct_support_keeps_capacity",
                "geometry_duplicate_demotes_even_with_distinct_support",
            ),
            "Posterior file competition must have a standalone math audit for duplicate demotion, mass conservation, distinct-owner preservation, and geometry duplicate handling.",
        ),
        Check(
            "full_binding_math_dataflow_audit_links_paper_code_and_runtime_repair",
            _contains(
                readme_v22,
                "PICF_AQR_OWM_FULL_BINDING_MATH_DATAFLOW_AUDIT_20260516_TEMP.md",
            )
            and _contains(
                full_binding_audit_doc,
                "/tmp/vit-object-binding",
                "DiagonalQuadraticProbe",
                "QuadraticProbe",
                "QuadraticFixedRankProbe",
                "online weak-label BCE",
                "posterior file competition",
                "mass-conserving",
                "orange role-1 posterior files",
            ),
            "Full binding audit must connect pulled paper code, PICF runtime pairwise binding, no-label limits, and posterior file competition repair.",
        ),
        Check(
            "pipeline_uses_pairwise_binding_subspace",
            _contains(
                pipeline,
                "def _binding_keys",
                "center: bool = False",
                "projected - projected.mean",
                "binding_signature_proj",
                "binding_quadratic_diag",
                "binding_low_rank_left",
                "_binding_signature_quadratic_scores",
                "_calibrate_pairwise_binding_score",
                "binding_signature_calibrated_score_std",
                "_support_binding_signature",
                "token_field.visual_tokens",
                "prev.binding_signature",
                "obs.binding_signature",
                "bind_embedding_signature_weight",
                "bind_quadratic_signature_weight",
                "bind_low_rank_signature_weight",
                "local_refinement_binding_weight",
                "binding_logits = query_binding @ token_binding.T",
            ),
            "Binding and the archived local-refinement ablation path must include an explicit projected same-object subspace instead of relying only on hidden cosine/support mass.",
        ),
        Check(
            "policy_supports_action_prefix_stop_gradient",
            _contains(
                policy,
                "def _action_prefix_tokens",
                "action_prefix_stopgrad",
                "tokens.detach()",
                "extra_prefix_tokens=self._action_prefix_tokens",
            )
            and _contains(trainer, "--picf-action-prefix-stopgrad", "action_prefix_stopgrad"),
            "Cotrain must support stopping action-flow gradients at PICF pi-prefix tokens without detaching the action loss itself.",
        ),
        Check(
            "pipeline_preserves_pg_priors",
            _contains(pipeline, "def _aqr_pg_image_support_read", "pg_priors", "image_token_ranges", "pg_visual_bias"),
            "AQR PG image support must survive as graph.pg_priors, not only visual bias.",
        ),
        Check(
            "pipeline_cache_order_is_causal",
            _contains(pipeline, "def _previous_evidence_cache_tokens", "def _write_evidence_cache", "previous.predictive", "evidence_cache=evidence_cache"),
            "Cache must be read from previous carry and written after posterior correction.",
        ),
        Check(
            "pipeline_cache_address_prefers_posterior_identity",
            _contains(
                pipeline,
                "def _physical_query_addresses",
                "previous.posterior.slot_address",
                "def _aqr_cache_query_addresses",
                "query_address = self._aqr_cache_query_addresses(previous, physical_count, task_count)",
            ),
            "Cache address retrieval must prefer live posterior slot addresses over learned query carriers, with query-token fallback.",
        ),
        Check(
            "pipeline_cache_read_weight_scales_residual",
            _contains(pipeline, "q_before_cache", "cache_read - q_before_cache", "evidence_cache_read_weight")
            and "cache_bias = cache_bias + math.log(max(float(self.config.evidence_cache_read_weight)" not in pipeline,
            "Cache read weight must scale the cache residual, not disappear as a constant softmax bias.",
        ),
        Check(
            "pipeline_cache_skips_immediate_previous_posterior_duplicate",
            _contains(pipeline, "immediate_posterior", "valid = valid & ~immediate_posterior", "aqr_posterior_reader", "cache_roles", "role_mask"),
            "Cache must skip the newest posterior cache row and apply role-aware filtering because previous posterior has a dedicated AQR reader.",
        ),
        Check(
            "pipeline_recurrent_burnin_uses_aqr_graph",
            _contains(burnin_body, "if bool(self.config.aqr_mapg_enabled):", "anchor_prior_graph = self._build_aqr_anchor_graph("),
            "State-only burn-in must use the same AQR measurement graph as the suffix path when AQR is enabled.",
        ),
        Check(
            "pipeline_outputs_required_owm_debug_keys",
            _contains(
                pipeline,
                "aqr_temporal_support_entropy_mean",
                "aqr_pg_support_entropy_mean",
                "aqr_proposal_support_entropy_mean",
                "posterior_identity_switch_rate",
                "posterior_identity_switch_rate_stable",
                "posterior_binding_top1_margin_mean",
                "posterior_active_file_self_signature_sim_mean",
                "posterior_active_file_potential_swap_rate",
                "posterior_active_file_calibrated_potential_swap_rate",
                "posterior_file_calibrated_signature_score_std",
                "posterior_binding_signature_quadratic_score_abs_mean",
                "posterior_binding_signature_low_rank_score_abs_mean",
                "posterior_binding_signature_combined_score_abs_mean",
                "posterior_binding_signature_calibrated_score_abs_mean",
                "posterior_binding_signature_calibrated_score_std",
                "posterior_binding_signature_calibrated_top1_margin_mean",
                "posterior_binding_signature_gate_mean",
                "posterior_binding_signature_update_rate_mean",
                "posterior_binding_signature_measurement_trust_mean",
                "posterior_binding_signature_memory_keep_rate_mean",
                "posterior_binding_signature_measurement_score_std",
                "posterior_binding_signature_measurement_margin_mean",
                "posterior_binding_signature_measurement_dispersion_gate_mean",
                "aqr_same_role_anchor_binding_signature_overlap_max",
                "aqr_same_role_obs_binding_signature_overlap_max",
                "aqr_ownership_prior_enabled",
                "aqr_ownership_prior_weight",
                "aqr_ownership_point_prior_weight",
                "aqr_ownership_point_prior_sigma_m",
                "aqr_same_role_object_core_overlap_max",
                "aqr_active_same_role_object_core_overlap_max",
                "aqr_same_role_support_competition_enabled",
                "aqr_same_role_support_competition_weight",
                "aqr_active_anchor_count",
                "aqr_active_same_role_support_overlap_max",
                "evidence_cache_trust_mean",
                "innovation_norm_visual",
                "owm_ordinal_active",
            ),
            "Pipeline debug output must expose every OWM branch required for diagnosis.",
        ),
        Check(
            "training_uses_next_posterior_teacher",
            _contains(training, "posterior_tokens", "posterior_support_summary", "future.posterior_tokens", "future.posterior_support_summary", "_matched_prediction_loss"),
            "Slot-JEPA/support prediction must prefer detached next posterior targets.",
        ),
        Check(
            "training_binding_consistency_is_permutation_tolerant",
            _contains(
                training.split("def _binding_consistency_loss", 1)[1].split("def _matched_prediction_loss", 1)[0],
                "assign_row = torch.softmax",
                "assign_col = torch.softmax",
                "matched_target",
                "matched_current",
            ),
            "Binding consistency must avoid an index-label temporal identity assumption before the loss is enabled.",
        ),
        Check(
            "training_loss_family_exposed",
            _contains(
                training,
                "lambda_slot_jepa",
                "lambda_support_pred",
                "lambda_binding_consistency",
                "lambda_aqr_denoising",
                "_aqr_support_denoising_loss",
            )
            and not _contains(
                training,
                "lambda_cross_modal_align",
                "lambda_ordinal_relation",
                "lambda_innovation_calib",
                "ordinal_confidence_threshold",
            ),
            "Only mathematically grounded OWM loss knobs should be available; weak placeholder losses must stay removed.",
        ),
        Check(
            "trainer_threads_next_posterior_teacher",
            _contains(trainer, "future_targets_from_current_targets(current_targets, availability, posterior=posterior)"),
            "Window trainer must pass the next observed posterior as detached teacher target.",
        ),
        Check(
            "trainer_materializes_optional_mvtrack_adapters",
            _contains(
                trainer,
                "tracklet_token_proj.weight",
                "tracklet_in = torch.zeros",
                "(1, 23)",
                "proposal_token_proj.weight",
                "proposal_in = torch.zeros",
                "(1, 26)",
            ),
            "Trainer warmup must materialize optional tracklet/proposal adapters even when a dataset lacks those modalities.",
        ),
        Check(
            "trainer_threads_optional_mvtrack_episode_fields",
            _contains(
                trainer,
                "_MVTRACK_TRACKLET_KEYS",
                "_MVTRACK_PROPOSAL_KEYS",
                "_read_npz_required_optional",
                "load_tracklet_fields",
                "tracklet_xy=frame.get(\"tracklet_xy\")",
                "proposal_centers_xy=frame.get(\"proposal_centers_xy\")",
            ),
            "Trainer data source must feed optional tracklet/proposal episode arrays when present and no-op when absent.",
        ),
        Check(
            "trainer_anchor_only_fsdp_ignores_frozen_root_states",
            _contains(
                trainer,
                "def _fsdp_frozen_states_excluding_modules",
                "picf_trainable_scope",
                "anchor_only",
                "ignored_module_states",
                "ignored_states_by_id",
                'root_wrap_kwargs["ignored_states"] = ignored_states',
                "elif ignored_modules",
                "uniform `requires_grad`",
            ),
            "Anchor-only FSDP must ignore frozen root-managed params so flat handles stay uniformly trainable.",
        ),
        Check(
            "trainer_checkpoint_retention_prunes_numeric_steps_only",
            _contains(
                trainer,
                "def _prune_old_checkpoints",
                "--keep-last-checkpoints",
                "keep_last_checkpoints",
                "shutil.rmtree",
                "path.name.startswith(\"tmp_\")",
            ),
            "Trainer must support bounded checkpoint retention without deleting non-step diagnostics.",
        ),
        Check(
            "trainer_logs_required_owm_metrics",
            _contains(
                trainer,
                "OWM_DEBUG_METRIC_KEYS",
                "aqr_temporal_support_entropy_mean",
                "evidence_cache_trust_mean",
                "posterior_recycle_logit_mean",
                "posterior_dustbin_mass_raw",
                "posterior_address_update_rate_mean",
                "posterior_lifecycle_assignment_confidence_mean",
                "posterior_lifecycle_survival_prob_mean",
                "posterior_lifecycle_reset_allowance_mean",
                "posterior_lifecycle_inactive_dustbin_mass",
                "posterior_lifecycle_unexplained_dustbin_mass",
                "posterior_identity_switch_rate_stable",
                "posterior_binding_top1_margin_mean",
                "posterior_binding_signature_quadratic_score_abs_mean",
                "posterior_binding_signature_low_rank_score_abs_mean",
                "posterior_binding_signature_gate_mean",
                "owm_posterior_binding_signature_norm_mean",
                "aqr_same_role_local_true_overlap_max",
                "aqr_same_role_anchor_binding_signature_overlap_max",
                "aqr_same_role_obs_binding_signature_overlap_max",
                "aqr_ownership_prior_enabled",
                "aqr_ownership_prior_weight",
                "aqr_ownership_point_prior_weight",
                "aqr_ownership_point_prior_sigma_m",
                "aqr_same_role_object_core_overlap_max",
                "aqr_active_same_role_object_core_overlap_max",
                "aqr_same_role_support_competition_enabled",
                "aqr_same_role_support_competition_weight",
                "posterior_owner_active_eligible_fraction",
                "posterior_owner_active_score_mean",
                "posterior_file_competition_birth_active_mean",
                "posterior_file_competition_birth_count",
                "posterior_file_competition_birth_share_mean",
                "_owm_debug_metrics_from_output",
            ),
            "Training metrics must carry OWM debug keys into metrics.jsonl.",
        ),
        Check(
            "evidence_bundle_reads_required_keys",
            _contains(
                evidence,
                "OWM_KEYS",
                "loss_slot_jepa",
                "loss_aqr_denoising",
                "aqr_temporal_support_entropy_mean",
                "aqr_proposal_support_entropy_mean",
                "aqr_same_role_local_true_overlap_max",
                "owm_posterior_binding_signature_norm_mean",
                "posterior_owner_active_eligible_fraction",
                "posterior_identity_switch_rate",
                "posterior_identity_switch_rate_stable",
                "posterior_binding_top1_margin_mean",
                "posterior_active_file_self_signature_sim_mean",
                "posterior_active_file_potential_swap_rate",
                "evidence_cache_trust_mean",
            ),
            "Evidence bundle must include OWM loss/debug metrics for reviewer handoff.",
        ),
        Check(
            "mvtrack_next_contract_is_linked_and_guarded",
            _contains_normalized(
                readme_v22,
                "PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md",
                "not a replacement",
                "maintained v26 baseline",
                "code-level runtime completion",
            )
            and _contains(
                mvtrack_readme,
                "PICF-AQR-OWM-MVTrack",
                "static+wrist V-JEPA typed memory",
                "support-signature identity binding",
                "address-aware cache retrieval",
                "tracklet typed memory",
                "optional proposal memory",
                "training-only support denoising",
                "matched slot-JEPA/support prediction",
                "MVTrack should be considered code-level runtime-complete after Section 16",
            ),
            "README_v2.2 must route reviewers to the guarded MVTrack next-version contract without claiming runtime completion.",
        ),
        Check(
            "professor_grade_interaction_audit_is_linked_and_cross_module",
            _contains(
                readme_v22,
                "PICF_AQR_OWM_PROFESSOR_GRADE_INTERACTION_AUDIT_TEMP.md",
            )
            and _contains(
                professor_audit,
                "multiview_temporal_uses_wrist_without_static_geometry_leak",
                "cache_is_auxiliary_addressed_residual_not_truth_or_duplicate",
                "active_owner_state_reaches_posterior_measurement_eligibility",
                "binding_logit_combines_hidden_geometry_support_address_with_trust_gates",
                "posterior_recycle_is_slotwise_normalized_and_not_global_dustbin_reset",
                "future_teachers_are_detached_and_losses_guarded",
                "diagnostic_surface_covers_interaction_failures",
            )
            and _contains(
                professor_audit_doc,
                "This audit is stricter than a presence verifier",
                "cross-module interaction",
                "cache_is_auxiliary_addressed_residual_not_truth_or_duplicate",
            ),
            "Professor-grade audit must be linked from README_v2.2 and verify cross-module interactions, not isolated fields only.",
        ),
    ]
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the PICF-AQR-OWM README-to-code deployment contract.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    checks = run_checks()
    failed = [check for check in checks if not check.ok]
    if args.json:
        print(json.dumps({"ok": not failed, "checks": [check.__dict__ for check in checks]}, indent=2, sort_keys=True))
    else:
        for check in checks:
            status = "PASS" if check.ok else "FAIL"
            print(f"{status} {check.name}: {check.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
