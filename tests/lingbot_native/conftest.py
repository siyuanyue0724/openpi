from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_SOURCE_COMMIT,
)
from picf_next.data.dataset_manifest import DATASET_RUNTIME_VERIFICATION_MODE
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    SELECTIVE_EMBEDDING_PARAMETER,
)
from picf_next.lingbot_native.task_diagnostics import TASK_ROW_DIAGNOSTIC_SCHEMA
from tools.bootstrap_lingbot_vla2 import (
    CHECKPOINT_ASSET_CONTRACT,
    LINGBOT_CHECKPOINT_ID,
    LINGBOT_CHECKPOINT_REVISION,
    PROCESSOR_ASSET_CONTRACT,
    QWEN_PROCESSOR_ID,
    QWEN_PROCESSOR_REVISION,
    asset_contract_manifest,
)
from tools.bootstrap_lingbot_vla2_native import LINGBOT_NATIVE_SOURCE_COMMIT
from tools.lingbot_vla2_runtime_helpers import resolve_lingbot_optimizer_contract
from tools.preflight_lingbot_native import PREFLIGHT_REPORT_SCHEMA
from tools.run_lingbot_vla2_native_full import (
    FULL_REPORT_SCHEMA,
    FULL_WORLD_SIZE,
    GRADIENT_AUDIT_TEMPORAL_SCOPE,
    REPRESENTATION_REPORT_SCHEMA,
)
from tools.run_lingbot_vla2_native_g0 import G0_REPORT_SCHEMA, _implementation_digest

ROOT = Path(__file__).resolve().parents[2]
CURRENT_G0_IMPLEMENTATION_SHA256 = _implementation_digest(ROOT)


def _selective_embedding_storage_report() -> dict[str, object]:
    return {
        "parameter_tensors": 2,
        "local_elements": 10,
        "master_dtype": "float32",
        "placement": FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
        "cpu_parameter_tensors": 1,
        "cpu_local_elements": 4,
        "cuda_parameter_tensors": 1,
        "cuda_local_elements": 6,
        "selective_cpu_parameter_names": [SELECTIVE_EMBEDDING_PARAMETER],
    }


def _released_optimizer_metadata() -> dict[str, object]:
    return resolve_lingbot_optimizer_contract(
        {
            "train": {
                "optimizer": "muon",
                "lr": 1e-4,
                "weight_decay": 0.0,
                "lr_decay_style": "constant",
                "lr_warmup_ratio": 0.0,
                "lr_start": 0.0,
                "use_moe": True,
                "use_moe_expert_lr": True,
                "token_moe_layers": list(range(36)),
                "token_num_experts": 32,
                "token_top_k": 4,
                "bias_update_speed": 0.0,
                "sequence_wise_loss_coeff": 1e-3,
                "sequence_wise_mode": "per_sequence",
                "router_z_loss_coeff": 1e-4,
                "router_activation": "sigmoid",
                "routed_scaling_factor": 4.0,
                "use_shared_expert_gate": False,
                "enable_fp32": True,
            }
        },
        requested_learning_rate=1e-4,
    ).metadata


@pytest.fixture
def full_objective_report_factory() -> Callable[..., Path]:
    """Write a minimal but semantically complete fresh full-objective report."""

    def write(
        path: Path,
        *,
        digest: str,
        input_step: int = 0,
        training_authorization: dict[str, object] | None = None,
        long_training_authorized: bool = False,
    ) -> Path:
        saved_step = input_step + 1
        checkpoint_dir = (
            path.parent / f"{path.stem}.run" / "checkpoints" / f"global_step_{saved_step}"
        ).resolve()
        checkpoint_dir.mkdir(parents=True)
        boundary = {
            "lane_snapshot_sha256": digest,
            "model_local_state_sha256": digest,
            "optimizer_local_state_sha256": digest,
            "rank_rng_state_sha256": digest,
        }
        family_diagnostics = {
            "all_finite": True,
            "cosines": {
                "action__predictive": 0.25,
                "action__structural": 0.5,
                "predictive__structural": 0.75,
            },
            "dot_products": {
                "action__predictive": 0.25,
                "action__structural": 0.5,
                "predictive__structural": 0.75,
            },
            "gradient_norms": {"action": 1.0, "predictive": 2.0, "structural": 3.0},
            "probe": "picf_native_graph.object_queries",
            "world_size": FULL_WORLD_SIZE,
        }
        relation_surface_diagnostics = {
            "all_finite": True,
            "cosines": {
                "task__task_dense@match_embeddings": 0.25,
                "task_dense__ownership@row_embeddings": -0.5,
            },
            "dot_products": {
                "task__task_dense@match_embeddings": 0.5,
                "task_dense__ownership@row_embeddings": -1.5,
            },
            "gradient_elements": {
                "task@match_embeddings": 81920,
                "task_dense@match_embeddings": 81920,
                "task_dense@row_embeddings": 81920,
                "ownership@row_embeddings": 81920,
            },
            "gradient_norms": {
                "task@match_embeddings": 1.0,
                "task_dense@match_embeddings": 2.0,
                "task_dense@row_embeddings": 2.0,
                "ownership@row_embeddings": 1.5,
            },
            "probe": "final_relation.match_embeddings+row_embeddings",
            "world_size": FULL_WORLD_SIZE,
        }
        task_assignment = {
            "identity_keys": ["pink_block", "drawer"],
            "sequence_time_count": 1,
            "source_phase": 1,
            "binding_start_phase": [1, 1],
            "source_binding_valid": [True, True],
            "row_to_track": [0, 1],
        }
        task_row_diagnostic = {
            "schema": TASK_ROW_DIAGNOSTIC_SCHEMA,
            "exact_task": True,
            **task_assignment,
            "source_time": 0,
            "source_side": "posterior",
            "track_task_targets": [1.0, 0.0],
            "track_task_valid": [True, True],
            "capacity_censored": [False, False],
            "assignment_sha256": hashlib.sha256(
                json.dumps(
                    task_assignment,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "row_task_targets": [1.0, 0.0],
            "row_task_valid": [True, True],
            "task_logits": [1.0, -1.0],
            "task_probabilities": [0.7310585786300049, 0.2689414213699951],
            "target_rows": [0],
            "target_identity_keys": ["pink_block"],
            "materialized_target_identity_keys": ["pink_block"],
            "unmaterialized_target_identity_keys": [],
            "known_negative_rows": [1],
            "minimum_target_logit": 1.0,
            "maximum_known_negative_logit": -1.0,
            "target_vs_hardest_negative_logit_margin": 2.0,
            "minimum_target_probability": 0.7310585786300049,
            "maximum_known_negative_probability": 0.2689414213699951,
            "target_vs_hardest_negative_probability_margin": 0.4621171572600098,
            "worst_target_rank": 1,
            "all_targets_beat_known_negatives": True,
        }
        step = {
            "global_step": saved_step,
            "estimator_component": "causal",
            "posterior_committed": True,
            "posterior_bank_sha256_before": digest,
            "posterior_bank_sha256_after": hashlib.sha256(
                f"{digest}:posterior-after".encode("ascii")
            ).hexdigest(),
            "sample_keys": [f"sample-{rank}" for rank in range(1)],
            "lane_ids": [0],
            "frame_indices": [saved_step - 1],
            "state_ages": [input_step],
            "temporal_plan_sha256": digest,
            "local_bptt_steps": 1,
            "overshoot_horizon": 0,
            "objective_total": 0.21072,
            "official_action_loss": 0.2,
            "official_moe_regularizer": 0.01,
            "official_policy_loss": 0.21,
            "normalized_terms": {
                "action": 0.21,
                "correction/dino_video": 0.1,
                "set/support": 0.0,
                "set/existence": 0.0,
                "set/task": 0.0,
                "set/task_dense": 0.0,
                "set/ownership": 0.09,
                "set/ownership_nll": 0.09,
                "xmod/vision/representation": 0.08,
            },
            "valid_counts": {
                "action": 1,
                "correction/dino_video": 1,
                "set/support": 0,
                "set/existence": 0,
                "set/task": 0,
                "set/task_dense": 0,
                "set/ownership": 1,
                "set/ownership_nll": 1,
                "xmod/vision/representation": 1,
            },
            "task_row_diagnostics": [task_row_diagnostic],
            "prior_row_bindings": (
                [[]] if input_step == 0 else [[["drawer", 1], ["pink_block", 0]]]
            ),
            "row_bindings": [[["drawer", 1], ["pink_block", 0]]],
            "row_binding_birth_count": 2 if input_step == 0 else 0,
            "gradient_metrics": {
                "all_finite": True,
                "native_graph_norm": 1.0,
                "native_graph_elements": 10,
                "relation_projection_norm": 1.0,
                "relation_projection_elements": 10,
                "match_projection_norm": 1.0,
                "match_projection_elements": 10,
                "action_output_norm": 1.0,
                "action_output_elements": 10,
                "predictive_readout_norm": 1.0,
                "predictive_readout_elements": 10,
                "preclip_global_norm": 1.0,
                "behavior_posterior_gradient_norm": 0.0,
                "behavior_posterior_gradient_elements": 0,
            },
            "family_gradient_diagnostics": family_diagnostics,
            "relation_surface_gradient_diagnostics": relation_surface_diagnostics,
            "predictive_host_gradient_diagnostics": {
                "all_finite": True,
                "decomposition": None,
                "gradient_elements": {"early": 2560, "middle": 2560, "late": 2560},
                "gradient_norms": {"early": 1.0, "middle": 1.0, "late": 1.0},
                "parameter_paths": {
                    "early": "model.language_model.layers.0.input_layernorm.weight",
                    "middle": "model.language_model.layers.18.input_layernorm.weight",
                    "late": "model.language_model.layers.35.input_layernorm.weight",
                },
                "probe": "lingbot.language_model.input_layernorm",
                "world_size": FULL_WORLD_SIZE,
            },
            "predictive_counterfactual_diagnostics": {
                "schema": "picf-next.lingbot-predictive-correction-counterfactual/v1",
                "valid_target_count": 1,
                "factual_loss": 0.1,
                "interventions": [
                    {
                        "name": "batch_shift_control",
                        "loss": 0.12,
                        "loss_margin_over_factual": 0.02,
                        "normalized_prediction_l1": 0.2,
                    },
                    {
                        "name": "batch_shift_source",
                        "loss": 0.13,
                        "loss_margin_over_factual": 0.03,
                        "normalized_prediction_l1": 0.3,
                    },
                    {
                        "name": "row_shift_source",
                        "loss": 0.14,
                        "loss_margin_over_factual": 0.04,
                        "normalized_prediction_l1": 0.4,
                    },
                    {
                        "name": "zero_control",
                        "loss": 0.11,
                        "loss_margin_over_factual": 0.01,
                        "normalized_prediction_l1": 0.1,
                    },
                    {
                        "name": "zero_current_observation",
                        "loss": 0.1,
                        "loss_margin_over_factual": 0.0,
                        "normalized_prediction_l1": 0.0,
                    },
                    {
                        "name": "zero_source",
                        "loss": 0.15,
                        "loss_margin_over_factual": 0.05,
                        "normalized_prediction_l1": 0.5,
                    },
                ],
            },
            "predictive_counterfactual_weight_boundary": "pre_update_post_backward",
            "source_masked_branch": True,
            "source_mask_digest": None,
            "source_mask_query_count": 0,
            "source_prediction_mode": "omitted_static",
            "omitted_static_digest": digest,
            "visual_artifacts": [],
            "visual_audit_seconds": 0.0,
            "step_time_s": 1.0,
            "peak_cuda_allocated_bytes": 100,
            "peak_cuda_reserved_bytes": 200,
        }
        if input_step == 0:
            step["objective_total"] = 0.21068
            step["normalized_terms"]["correction/dino_video"] = 0.0
            step["valid_counts"]["correction/dino_video"] = 0
            step["family_gradient_diagnostics"] = None
            step["relation_surface_gradient_diagnostics"] = None
            step["predictive_host_gradient_diagnostics"] = None
            step["predictive_counterfactual_diagnostics"] = None
            step["predictive_counterfactual_weight_boundary"] = None
        elif input_step >= 2:
            step["predictive_counterfactual_diagnostics"]["interventions"].append(
                {
                    "name": "wrong_time_source",
                    "loss": 0.16,
                    "loss_margin_over_factual": 0.06,
                    "normalized_prediction_l1": 0.6,
                }
            )
            step["predictive_counterfactual_diagnostics"]["interventions"].sort(
                key=lambda item: item["name"]
            )
        report = {
            "schema": FULL_REPORT_SCHEMA,
            "status": "PASS",
            "phase": "fresh" if input_step == 0 else "resume",
            "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
            "patch_sha256": digest,
            "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
            "input_global_step": input_step,
            "saved_global_step": saved_step,
            "execution_contract_sha256": digest,
            "implementation_sha256": digest,
            "model_family_sha256": digest,
            "plan_sha256": digest,
            "temporal_estimator_sha256": digest,
            "dataset_contract": {
                "status": "PASS",
                "manifest_sha256": digest,
                "normalization_sha256": digest,
                "validation": {
                    "dataset_file_count": 1,
                    "dataset_total_size_bytes": 1,
                    "dataset_tree_sha256": digest,
                    "dataset_manifest_self_consistent": True,
                    "dataset_full_tree_rescanned": False,
                    "dataset_runtime_verified_read_required": True,
                    "dataset_runtime_probe_file_count": 1,
                    "dataset_runtime_probe_sha256": digest,
                    "dataset_verification_mode": DATASET_RUNTIME_VERIFICATION_MODE,
                },
            },
            "physical_sidecar_manifest_sha256": digest,
            "predictive_cache_manifest_sha256": digest,
            "predictive_teacher_causality_audit_sha256": digest,
            "predictive_target_audit_sha256": digest,
            "predictive_temporal_audit_sha256": digest,
            "current_grid_cache_manifest_sha256": digest,
            "checkpoint_dir": str(checkpoint_dir),
            "full_shard": True,
            "fsdp2_placement": FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
            "cuda_allocator": "native",
            "gradient_checkpointing": True,
            "action_loss_enabled": True,
            "predictive_correction_loss_enabled": True,
            "behavior_future_loss_enabled": False,
            "behavior_conditioning": None,
            "structural_set_loss_enabled": True,
            "current_source_mask_enabled": False,
            "omitted_static_binding_enabled": True,
            "source_prediction_mode": "omitted_static",
            "objective_contract": {
                "predictive_family_weight": 0.004,
                "structural_family_weight": 0.004,
                "predictive_term_weight": 1.0,
                "current_grid_term_weight": 1.0,
                "omitted_static_term_weight": 1.0,
                "support_weight": 0.0,
                "existence_weight": 1.0,
                "task_weight": 1.0,
                "dense_task_weight": 1.0,
                "ownership_weight": 1.0,
                "family_reduction": "active_weighted_mean",
            },
            "evidence_profile": "acceptance",
            "gradient_audit_steps": [2, 3] if input_step == 0 else [saved_step],
            "gradient_audit_temporal_scope": GRADIENT_AUDIT_TEMPORAL_SCOPE,
            "complete_adr74_objective": False,
            "training_authorization": training_authorization,
            "long_training_authorized": long_training_authorized,
            "parameter_storage": _selective_embedding_storage_report(),
            "alignment_teacher_prune": {
                "schema": "picf-next.targetless-alignment-teacher-prune.v1",
                "removed": [
                    {
                        "name": "depth_align_head",
                        "parameter_count": 1,
                        "numel": 10,
                        "storage_bytes": 40,
                    }
                ],
                "removed_numel": 10,
                "removed_storage_bytes": 40,
                "retained_query_components": ["depth_align_embs"],
            },
            "action_fsdp2_topology": {
                "schema": "picf-next.lingbot-action-block-fsdp2.v1",
                "block_count": 1,
                "block_paths": ["model.qwenvl_with_expert.qwen_expert.model.layers.0"],
                "maximum_block_bf16_bytes_upper_bound": 1_024,
            },
            "vlm_fsdp2_topology": {
                "text_block_count": 1,
                "text_block_paths": ["model.vlm.text.layers.0"],
                "vision_block_count": 1,
                "vision_block_paths": ["model.vlm.vision.blocks.0"],
            },
            "maximum_peak_reserved_bytes": 1_000,
            "parameter_manifest": {
                "parameter_count": 2,
                "trainable_numel": 20,
                "schema_sha256": digest,
            },
            "rank_reports": [
                {
                    "rank": rank,
                    "steps": [
                        {
                            **step,
                            "sample_keys": [f"sample-{rank}"],
                            "lane_ids": [rank],
                        }
                    ],
                    "saved_boundary_sha256": boundary,
                    "loaded_boundary_sha256": None if input_step == 0 else boundary,
                }
                for rank in range(FULL_WORLD_SIZE)
            ],
        }
        payload = json.dumps(report, sort_keys=True)
        path.write_text(payload)
        (checkpoint_dir / "native_full_report.json").write_text(payload)
        return path

    return write


@pytest.fixture
def representation_objective_report_factory(
    full_objective_report_factory: Callable[..., Path],
) -> Callable[..., Path]:
    """Write an action-isolated representation report from the full-report fixture."""

    def write(
        path: Path,
        *,
        digest: str,
        input_step: int = 0,
        behavior: bool = False,
    ) -> Path:
        full_path = full_objective_report_factory(
            path.with_name(f"{path.stem}.full.json"),
            digest=digest,
            input_step=input_step,
        )
        report = json.loads(full_path.read_text())
        report["schema"] = REPRESENTATION_REPORT_SCHEMA
        report["action_loss_enabled"] = False
        report["training_authorization"] = None
        report["long_training_authorized"] = False
        report["checkpoint_publication"] = "always"
        report["training_stage"] = "representation"
        report["representation_split_sha256"] = digest
        report["representation_split_file_sha256"] = digest
        report["representation_frozen_action_state_sha256"] = digest
        report["visual_audit_every"] = 0
        report["objective_contract"]["action_family_weight"] = 0.0
        scope = {
            "schema": "picf-next.lingbot-representation-scope.v1",
            "production_trainable_sha256": digest,
            "production_frozen_sha256": digest,
            "representation_trainable_sha256": digest,
            "action_frozen_sha256": digest,
            "production_trainable_numel": 20,
            "production_frozen_numel": 10,
            "representation_trainable_numel": 15,
            "action_frozen_numel": 5,
        }
        report["representation_parameter_scope"] = scope
        report["representation_parameter_scope_sha256"] = hashlib.sha256(
            json.dumps(
                scope,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        report["parameter_manifest"]["trainable_numel"] = 15
        for rank_report in report["rank_reports"]:
            step = rank_report["steps"][0]
            step["fixed_observation_pair_sha256"] = None
            step["fixed_observation_fingerprint"] = None
            step["official_action_loss"] = None
            step["official_moe_regularizer"] = None
            step["official_policy_loss"] = None
            step["normalized_terms"].pop("action")
            step["valid_counts"].pop("action")
            predictive = 0.08 if input_step == 0 else 0.09
            step["objective_total"] = 0.004 * predictive + 0.004 * 0.09
            step["gradient_metrics"]["action_output_norm"] = 0.0
            step["gradient_metrics"]["action_output_elements"] = 0
            diagnostic = step["family_gradient_diagnostics"]
            if diagnostic is not None:
                diagnostic["probe"] = (
                    "lingbot.language_model.input_layernorm.early_middle_late.shared_objective"
                )
                diagnostic["gradient_norms"]["action"] = 0.0
                for pair in ("action__predictive", "action__structural"):
                    diagnostic["dot_products"][pair] = 0.0
                    diagnostic["cosines"][pair] = None

            if behavior:
                if input_step != 1:
                    raise ValueError("behavior report fixture requires the cold-resume interval")
                correction = step["normalized_terms"].pop("correction/dino_video")
                count = step["valid_counts"].pop("correction/dino_video")
                step["normalized_terms"]["rollout/vision/binding"] = correction
                step["valid_counts"]["rollout/vision/binding"] = count
                step["gradient_metrics"]["behavior_posterior_gradient_norm"] = 0.5
                step["gradient_metrics"]["behavior_posterior_gradient_elements"] = 16
                host = step["predictive_host_gradient_diagnostics"]
                host["probe"] = "lingbot.language_model.input_layernorm.via_primary_posterior_vjp"
                host["decomposition"] = {
                    "components": ["total", "via_posterior", "direct"],
                    "depths": {
                        depth: {
                            "total_norm": 1.0,
                            "via_posterior_norm": 0.5,
                            "direct_norm": 0.5,
                            "via_to_total_norm_ratio": 0.5,
                            "total_via_cosine": 1.0,
                            "total_direct_cosine": 1.0,
                            "closure_error_norm": 0.0,
                        }
                        for depth in ("early", "middle", "late")
                    },
                    "identity": "weighted_behavior_total=direct+via_primary_posterior",
                }
                step["predictive_counterfactual_diagnostics"] = None
                step["predictive_counterfactual_weight_boundary"] = None

        if behavior:
            report["predictive_correction_loss_enabled"] = False
            report["behavior_future_loss_enabled"] = True
            report["behavior_conditioning"] = {
                "schema": "picf-next.lingbot-behavior-conditioning.v2",
                "protocol": "g1_first_update_and_cold_resume",
                "horizon": 1,
                "isolation": "separate_same_weight_auxiliary_forward",
                "behavior_graph_sha256": digest,
                "g0_evidence_sha256": digest,
            }
            report["gradient_audit_steps"] = [1, 2]

        checkpoint_dir = Path(report["checkpoint_dir"])
        (checkpoint_dir / "native_full_report.json").unlink()
        payload = json.dumps(report, sort_keys=True)
        path.write_text(payload)
        (checkpoint_dir / "native_representation_report.json").write_text(payload)
        return path

    return write


@pytest.fixture
def g0_report_factory() -> Callable[..., Path]:
    """Write a complete hash-bound G0 fresh or cold-resume report."""

    def write(path: Path, *, phase: str, digest: str = "a" * 64) -> Path:
        input_step = 0 if phase == "fresh" else 1
        checkpoint_dir = (path.parent / f"{path.stem}.checkpoint").resolve()
        checkpoint_dir.mkdir()
        boundary = {
            "lane_snapshot_sha256": digest,
            "model_local_state_sha256": digest,
            "optimizer_local_state_sha256": digest,
            "rank_rng_state_sha256": digest,
        }
        rank_reports = []
        for rank in range(FULL_WORLD_SIZE):
            rank_reports.append(
                {
                    "rank": rank,
                    "sample_keys": [f"sample-{phase}-{rank}"],
                    "lane_ids": [rank],
                    "episode_keys": [f"episode-{rank}"],
                    "frame_indices": [input_step],
                    "official_action_loss": 0.2,
                    "official_moe_regularizer": 0.01,
                    "official_policy_loss": 0.21,
                    "gradient_metrics": {
                        "all_finite": True,
                        "native_graph_norm": 1.0,
                        "native_graph_elements": 10,
                        "action_output_norm": 1.0,
                        "action_output_elements": 10,
                        "preclip_global_norm": 1.0,
                    },
                    "optimizer_state": {
                        "optimizer_state_entries": 2,
                        "optimizer_local_moment_elements": 20,
                    },
                    "step_time_s": 1.0,
                    "peak_cuda_allocated_bytes": 100,
                    "peak_cuda_reserved_bytes": 200,
                    "saved_boundary_sha256": boundary,
                    "loaded_boundary_sha256": boundary if phase == "resume" else None,
                    "resume_boundary_verified": phase == "resume",
                    "resume_runtime_rng_verified": phase == "resume",
                }
            )
        report = {
            "schema": G0_REPORT_SCHEMA,
            "phase": phase,
            "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
            "patch_sha256": digest,
            "patched_source_sha256": {"lingbotvla/native.py": digest},
            "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
            "execution_contract_sha256": digest,
            "implementation_sha256": digest,
            "model_family_sha256": digest,
            "plan_sha256": digest,
            "dataset_contract": {
                "status": "PASS",
                "manifest_sha256": digest,
                "normalization_sha256": digest,
                "validation": {
                    "dataset_file_count": 1,
                    "dataset_total_size_bytes": 1,
                    "dataset_tree_sha256": digest,
                    "dataset_manifest_self_consistent": True,
                    "dataset_full_tree_rescanned": False,
                    "dataset_runtime_verified_read_required": True,
                    "dataset_runtime_probe_file_count": 1,
                    "dataset_runtime_probe_sha256": digest,
                    "dataset_verification_mode": DATASET_RUNTIME_VERIFICATION_MODE,
                },
            },
            "input_global_step": input_step,
            "saved_global_step": input_step + 1,
            "checkpoint_dir": str(checkpoint_dir),
            "status": "PASS",
            "full_shard": True,
            "fsdp2_placement": FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
            "cuda_allocator": "native",
            "gradient_checkpointing": True,
            "auxiliary_target_losses_enabled": False,
            "parameter_storage": _selective_embedding_storage_report(),
            "maximum_peak_reserved_bytes": 1_000,
            "parameter_manifest": {
                "parameter_count": 2,
                "trainable_numel": 20,
                "schema_sha256": digest,
            },
            "rank_reports": rank_reports,
        }
        payload = json.dumps(report, sort_keys=True)
        path.write_text(payload)
        (checkpoint_dir / "native_g0_report.json").write_text(payload)
        return path

    return write


@pytest.fixture
def preflight_report_factory() -> Callable[..., Path]:
    """Write the complete cloud-ready preflight report schema."""

    def write(path: Path, *, digest: str = "a" * 64) -> Path:
        root = (path.parent / f"{path.stem}.root").resolve()
        source = root / "third_party" / "lingbot-vla"
        calvin_environment = root / "third_party" / "calvin" / "calvin_env"
        checkpoint = root / "checkpoint"
        processor = root / "processor"
        dataset = root / "dataset" / "training"
        persistent_storage = root / "runs"
        import_origin = root / "src" / "picf_next" / "__init__.py"
        python = root / "venv" / "bin" / "python"
        for directory in (
            source,
            calvin_environment,
            checkpoint,
            processor,
            dataset,
            persistent_storage,
            import_origin.parent,
            python.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        import_origin.write_text("", encoding="ascii")
        python.write_text("", encoding="ascii")
        command = {
            "command": [str(python), "-c", "pass"],
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "passed": True,
        }
        report = {
            "schema": PREFLIGHT_REPORT_SCHEMA,
            "status": "PASS",
            "static_contract_pass": True,
            "local_tests_executed": True,
            "local_deployment_pass": True,
            "g0_action_only_static_ready": True,
            "future_structural_runner_static_ready": True,
            "complete_adr74_static_ready": True,
            "complete_adr74_missing_capabilities": [],
            "released_weight_omitted_static_binding_validated": False,
            "full_objective_static_ready": True,
            "full_objective_missing_files": [],
            "cloud_runtime_ready": True,
            "cloud_hardware_ready": True,
            "cloud_model_assets_ready": True,
            "cloud_data_ready": True,
            "cloud_assets_ready": True,
            "cloud_g0_ready": True,
            "authorized_gates": [
                "G0_full_weight_neutral_parity",
                "G0_two_rank_full_update_and_cold_resume",
            ],
            "long_training_authorized": False,
            "scientific_acceptance": "PENDING_G1_G8",
            "root": str(root),
            "python": str(python),
            "source_checkout": str(source),
            "static_checks": {
                "required_files": 1,
                "implementation_files": ["src/picf_next/__init__.py"],
                "implementation_sha256": digest,
                "full_implementation_files": ["src/picf_next/__init__.py"],
                "full_implementation_sha256": digest,
                "patch_replay": {
                    "apply_checked": True,
                    "commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                    "patch": "references/patches/lingbot_vla2_picf_native.patch",
                    "patch_sha256": digest,
                    "patched_sources": ["lingbotvla/native.py"],
                    "patched_source_sha256": {"lingbotvla/native.py": digest},
                    "verification_source": "immutable_commit_archive",
                },
                "lingbot_requirements_sha256": digest,
                "lingbot_depth_requirements_sha256": digest,
                "released_training_config_sha256": digest,
                "released_optimizer_contract": _released_optimizer_metadata(),
                "calvin_environment": {
                    "status": "PASS",
                    "calvin_env_root": str(calvin_environment),
                    "calvin_commit": CALVIN_SOURCE_COMMIT,
                    "calvin_env_commit": CALVIN_ENV_SOURCE_COMMIT,
                    "calvin_requirements_sha256": digest,
                    "calvin_setup_sha256": digest,
                },
            },
            "commands": [dict(command) for _ in range(6)],
            "import_origin": str(import_origin),
            "import_origin_valid": True,
            "python_version": "3.12.0",
            "python_major_minor": [3, 12],
            "package_versions": {"torch": "2.8.0"},
            "expected_cloud_runtime": {"torch": "2.8.0"},
            "host_import_probe": dict(command),
            "gpu_inventory": [
                {
                    "index": rank,
                    "name": "NVIDIA A100-SXM4-40GB",
                    "memory_mib": 40_960,
                }
                for rank in range(2)
            ],
            "hardware_capacity": {
                "host_memory_bytes": 128 * 2**30,
                "minimum_host_memory_bytes": 128 * 2**30,
                "persistent_storage_root": str(persistent_storage),
                "free_storage_bytes": 250 * 2**30,
                "minimum_free_storage_bytes": 250 * 2**30,
            },
            "checkpoint": {
                "checkpoint_id": LINGBOT_CHECKPOINT_ID,
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "checkpoint_dir": str(checkpoint),
                "required_files": len(CHECKPOINT_ASSET_CONTRACT),
                "checkpoint_assets": asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            },
            "processor": {
                "processor_id": QWEN_PROCESSOR_ID,
                "processor_revision": QWEN_PROCESSOR_REVISION,
                "processor_dir": str(processor),
                "required_processor_files": len(PROCESSOR_ASSET_CONTRACT),
                "processor_assets": asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            },
            "g0_data": {
                "ready": True,
                "status": "PASS",
                "dataset_split": str(dataset),
                "dataset_manifest_sha256": digest,
                "norm_stats_sha256": digest,
                "validation": {
                    "dataset_file_count": 1,
                    "dataset_total_size_bytes": 1,
                    "dataset_tree_sha256": digest,
                    "dataset_manifest_self_consistent": True,
                    "dataset_full_tree_rescanned": False,
                    "dataset_runtime_verified_read_required": True,
                    "dataset_runtime_probe_file_count": 1,
                    "dataset_runtime_probe_sha256": digest,
                    "dataset_verification_mode": DATASET_RUNTIME_VERIFICATION_MODE,
                },
            },
        }
        path.write_text(json.dumps(report, sort_keys=True))
        return path

    return write


@pytest.fixture
def smoke_report_factory() -> Callable[..., Path]:
    """Write the complete released-weight neutral/isolation smoke schema."""

    def write(path: Path, *, digest: str = "a" * 64) -> Path:
        config = path.parent / f"{path.stem}.yaml"
        image = path.parent / f"{path.stem}.png"
        config.write_bytes(b"config")
        image.write_bytes(b"image")
        config_digest = hashlib.sha256(config.read_bytes()).hexdigest()
        image_digest = hashlib.sha256(image.read_bytes()).hexdigest()
        report = {
            "schema": "picf-next.lingbot-vla2-native-full-weight-smoke.v4",
            "implementation_sha256": CURRENT_G0_IMPLEMENTATION_SHA256,
            "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
            "source_patch_sha256": digest,
            "patched_source_sha256": {"lingbotvla/native.py": digest},
            "source_diff_sha256": digest,
            "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
            "checkpoint_assets": asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            "processor_revision": QWEN_PROCESSOR_REVISION,
            "processor_assets": asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            "config": str(config.resolve()),
            "config_sha256": config_digest,
            "image": str(image.resolve()),
            "image_sha256": image_digest,
            "task": "pick up the red block",
            "alternate_task": "move the slider to the left",
            "target_only_fields_present": [],
            "device": "cuda:0",
            "device_name": "NVIDIA A100-SXM4-40GB",
            "dtype": "torch.bfloat16",
            "num_steps": 2,
            "native_graph": {
                "capacity": 16,
                "host_width": 256,
                "executed_action_dim": 7,
                "num_layers": 2,
                "maximum_control_tokens": 8,
            },
            "input_shapes": {"input_ids": [1, 8]},
            "moe_inference_backend": {
                "schema": "picf-next.lingbot-moe-inference-backend.v1",
                "selected": "fused_moe_forward",
                "fused_fallback_available": True,
                "robby_available_before_selection": True,
                "robby_disabled": True,
            },
            "official_action_sha256": digest,
            "official_repeat_action_sha256": digest,
            "targetless_action_sha256": digest,
            "installed_neutral_action_sha256": digest,
            "alignment_teacher_prune": {
                "schema": "picf-next.targetless-alignment-teacher-prune.v1",
                "removed": [
                    {
                        "name": "depth_align_head",
                        "parameter_count": 1,
                        "numel": 2,
                        "storage_bytes": 8,
                    }
                ],
                "removed_numel": 2,
                "removed_storage_bytes": 8,
                "retained_query_components": ["depth_align_embs"],
            },
            "official_repeat_action_bitwise_equal": True,
            "official_repeat_action_max_abs_error": 0.0,
            "targetless_action_bitwise_equal": True,
            "targetless_action_max_abs_error": 0.0,
            "neutral_action_bitwise_equal": True,
            "neutral_action_max_abs_error": 0.0,
            "official_routes": {
                "sha256": digest,
                "calls": 1,
                "tokens": 1,
                "layers": 1,
            },
            "official_repeat_routes": {
                "sha256": digest,
                "calls": 1,
                "tokens": 1,
                "layers": 1,
            },
            "targetless_routes": {
                "sha256": digest,
                "calls": 1,
                "tokens": 1,
                "layers": 1,
            },
            "installed_neutral_routes": {
                "sha256": digest,
                "calls": 1,
                "tokens": 1,
                "layers": 1,
            },
            "official_repeat_route_bitwise_equal": True,
            "targetless_route_bitwise_equal": True,
            "neutral_route_bitwise_equal": True,
            "first_action_sha256": digest,
            "second_action_sha256": digest,
            "first_prior_sha256": digest,
            "first_posterior_sha256": digest,
            "second_prior_sha256": digest,
            "second_posterior_sha256": digest,
            "native_actions_finite": True,
            "native_relations_finite": True,
            "session_snapshot_bytes": 1,
            "session_snapshot_sha256": digest,
            "session_snapshot_roundtrip_exact": True,
            "prompt_invariant_physical_posterior_bitwise_equal": True,
            "prompt_invariant_physical_posterior_max_abs_error": 0.0,
            "alternate_action_sha256": digest,
            "timings": {"official_action_s": 1.0},
            "cuda_memory_bytes": {
                "allocated": 1,
                "reserved": 2,
                "peak_allocated": 2,
                "peak_reserved": 2,
            },
            "pid": 1,
            "status": "PASS",
            "failures": [],
        }
        path.write_text(json.dumps(report, sort_keys=True))
        return path

    return write
