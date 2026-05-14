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
    contracts = _read("src/openpi/picf/core/contracts.py")
    config = _read("src/openpi/picf/core/config.py")
    pipeline = _read("src/openpi/picf/core/pipeline.py")
    policy = _read("src/openpi/picf/policy.py")
    training = _read("src/openpi/picf/core/training.py")
    trainer = _read("scripts/picf_core_train.py")
    wrapper = _read("src/openpi/picf/vjepa/wrapper.py")
    evidence = _read("scripts/picf_owm_evidence_bundle.py")
    readme = _read("docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md")
    readme_v22 = _read("src/openpi/picf/README_v2.2.md")
    mvtrack_readme = _read("docs/PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md")
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
                "proposal_memory_enabled: bool = True",
                "bind_support_signature_weight",
                "bind_embedding_signature_weight",
                "binding_signature_dim",
                "aqr_ownership_prior_enabled",
                "aqr_ownership_prior_weight",
                "aqr_ownership_temporal_prior_weight",
                "aqr_same_role_support_competition_enabled",
                "aqr_same_role_support_competition_weight",
                "aqr_active_slot_filter_enabled",
                "aqr_active_slot_overlap_threshold",
                "local_refinement_binding_weight",
                "recycle_normalize_residual_summary",
                "recycle_residual_norm_mode",
                "posterior_slotwise_recycle_residual: bool = True",
                "action_prefix_stopgrad",
                "evidence_cache_address_weight",
            ),
            "Config must default to the direct-final OWM graph profile, with legacy routers off and typed evidence on.",
        ),
        Check(
            "trainer_defaults_match_owm_profile",
            _contains(
                trainer,
                "_LOSS_DEFAULTS = PicfTransitionLossConfig()",
                'default="paligemma"',
                "default=_SPEC_DEFAULTS.aqr_mapg_enabled",
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
                "_save_anchor_overlay_diagnostic",
                "_anchor_overlay_snapshot_from_output(output, current)",
                "capture_anchor_overlay_step",
                "anchor_overlays",
                "This is a training diagnostic only and does not change the loss or forward path.",
                "and not capture_anchor_overlay_step",
            ),
            "Anchor position overlays must reuse the real training forward rather than run an extra buffer-mutating diagnostic forward.",
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
                "graph_tracklet_weights",
                "graph_proposal_weights",
                "bind_support_signature_weight",
                "bind_address_weight",
                "legacy_local_refinement_opt_in",
                "local_priors",
                "ordinal_target_rank",
                "binding_signature_proj",
                "bind_embedding_signature_weight",
                "posterior_slotwise_recycle_residual",
                "_posterior_occupancy_binding_bias",
                "posterior_occupancy_prior_weight",
                "posterior_occupancy_prior_clip",
                "observation_anchor_seed_point_mix",
                "seed_point_priors",
                "slot_residual_summary",
                "raw_cond = support_raw / raw_denom",
                "self.residual_mu_head(slot_residual_summary)",
                "_aqr_visual_ownership_bias",
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
                "def _aqr_temporal_ownership_bias",
                "visual_bias = ownership_bias if visual_bias is None else (visual_bias + ownership_bias)",
                "attn_bias=temporal_bias",
            )
            and _contains(
                trainer,
                "--aqr-ownership-prior-enabled",
                "--aqr-ownership-prior-weight",
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
                "active_slot_filter_enabled",
                "aqr_active_same_role_support_overlap_max",
                "aqr_inactive_anchor_fraction",
            )
            and _contains(
                trainer,
                "--aqr-active-slot-filter-enabled",
                "--aqr-active-slot-min-per-role",
                "--aqr-active-slot-max-per-role",
                "--posterior-slotwise-recycle-residual",
            ),
            "AQR must distinguish active object slots from inactive/dustbin anchors instead of forcing all fixed queries to bind objects.",
        ),
        Check(
            "pipeline_uses_pairwise_binding_subspace",
            _contains(
                pipeline,
                "def _binding_keys",
                "binding_signature_proj",
                "_support_binding_signature",
                "token_field.visual_tokens",
                "prev.binding_signature",
                "obs.binding_signature",
                "bind_embedding_signature_weight",
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
                "aqr_same_role_anchor_binding_signature_overlap_max",
                "aqr_same_role_obs_binding_signature_overlap_max",
                "aqr_ownership_prior_enabled",
                "aqr_ownership_prior_weight",
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
                "posterior_identity_switch_rate_stable",
                "posterior_binding_top1_margin_mean",
                "owm_posterior_binding_signature_norm_mean",
                "aqr_same_role_local_true_overlap_max",
                "aqr_same_role_anchor_binding_signature_overlap_max",
                "aqr_same_role_obs_binding_signature_overlap_max",
                "aqr_ownership_prior_enabled",
                "aqr_ownership_prior_weight",
                "aqr_same_role_support_competition_enabled",
                "aqr_same_role_support_competition_weight",
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
                "posterior_identity_switch_rate",
                "posterior_identity_switch_rate_stable",
                "posterior_binding_top1_margin_mean",
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
