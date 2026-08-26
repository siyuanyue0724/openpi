#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Run the two-rank LingBot-native action/predictive/structural transaction.

This is the production training entrypoint above the released LingBot-VLA2
host. Accelerator imports stay inside :func:`main`, keeping source, argument,
checkpoint and provenance contracts testable on a CPU-only workstation.

Dominant prior-to-current correction and sampled source-omitted objectives read
an immutable current-frame cache. Only executed-control-conditioned rollouts
read the future cache. Object labels remain loss-side only and never enter a
sampler, LingBot inputs, recurrent state, or action path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import stat
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _repository_import_path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _repository_import_text = str(_repository_import_path)
    while _repository_import_text in sys.path:
        sys.path.remove(_repository_import_text)
    sys.path.insert(0, _repository_import_text)

from tools.cuda_allocator_bootstrap import (
    CUDA_ALLOCATOR_MODES,
    bootstrap_cuda_allocator,
    configure_cuda_allocator as _configure_cuda_allocator,
)

_BOOTSTRAPPED_CUDA_ALLOCATOR = (
    bootstrap_cuda_allocator(sys.argv[1:]) if __name__ == "__main__" else None
)

import picf_next as _picf_next_package

if (
    _picf_next_package.__file__ is None
    or Path(_picf_next_package.__file__).resolve().parent
    != (_REPOSITORY_ROOT / "src/picf_next").resolve()
):
    raise RuntimeError("native full runner did not import picf_next from its own checkout")

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_visual_acceptance import (
    load_calvin_physical_visual_acceptance,
)
from picf_next.data.lingbot_calvin_projection import (
    LINGBOT_CALVIN_CAMERA_SLOTS,
    processor_assets_sha256,
    validate_lingbot_calvin_projection_payload,
)
from picf_next.data.token_supervision_policy import (
    build_known_pixel_token_supervision_policy,
    validate_known_pixel_token_supervision_policy,
)
from picf_next.data.dataset_manifest import (
    DATASET_RUNTIME_BINDING_FIELDS,
    validate_dataset_runtime_binding_report,
)
from picf_next.lingbot_native.fixed_batch_probe import (
    PREDICTIVE_FIXED_BATCH_ARM_REPORT_SCHEMA,
    PREDICTIVE_FIXED_BATCH_ARMS,
    FixedBatchTrainableScope,
    ShuffledCurrentGridTargetCache,
    configure_fixed_batch_trainable_scope,
    fixed_batch_probe_subject,
    validate_predictive_fixed_batch_arm_report,
    verify_fixed_batch_trainable_scope,
)
from picf_next.lingbot_native.relation_geometry_probe import (
    RELATION_GEOMETRY_ARM_REPORT_SCHEMA,
    RELATION_GEOMETRY_FIXED_BATCH_ARMS,
    RelationProbeSampleMetadata,
    RelationProbeSampleSelection,
    RelationGeometryTrainableScope,
    configure_relation_geometry_trainable_scope,
    relation_geometry_probe_subject,
    select_relation_geometry_probe_sample,
    validate_relation_geometry_arm_report,
    verify_relation_geometry_trainable_scope,
)
from picf_next.lingbot_native.relation_gradient_diagnostics import (
    relation_surface_gradient_contract as _relation_surface_gradient_contract,
    relation_surface_component_gradients as _relation_surface_component_gradients,
)
from picf_next.lingbot_native.relation_bilinear_probe import (
    RELATION_BILINEAR_PROBE_ARM,
    RELATION_BILINEAR_PROBE_CURVE_NAMES,
    RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT,
    RELATION_BILINEAR_PROBE_GLOBAL_REFERENCES,
    RELATION_BILINEAR_PROBE_LEARNING_RATES,
    RELATION_BILINEAR_PROBE_SCHEMA,
    RELATION_BILINEAR_PROBE_UPDATE_COUNT,
    RELATION_BILINEAR_PROBE_VISUAL_POINTS,
    RELATION_BILINEAR_PROBE_WEIGHT_DECAY,
    RelationBilinearCandidate,
    build_relation_bilinear_probe_bank,
    relation_bilinear_decisions,
    relation_bilinear_probe_subject,
    validate_relation_bilinear_probe_report,
)
from picf_next.lingbot_native.relation_depth_probe import (
    RELATION_DEPTH_PROBE_ARM,
    RELATION_DEPTH_PROBE_CURVE_NAMES,
    RELATION_DEPTH_PROBE_CURVE_POINT_COUNT,
    RELATION_DEPTH_PROBE_GLOBAL_REFERENCES,
    RELATION_DEPTH_PROBE_LEARNING_RATES,
    RELATION_DEPTH_PROBE_SCHEMA,
    RELATION_DEPTH_PROBE_UPDATE_COUNT,
    RELATION_DEPTH_PROBE_VISUAL_POINTS,
    RELATION_DEPTH_PROBE_WEIGHT_DECAY,
    LingBotRelationDepthCapture,
    RelationDepthCandidate,
    build_relation_depth_probe_bank,
    relation_depth_decisions,
    relation_depth_inputs,
    relation_depth_probe_subject,
    relation_depth_recovery_summary,
    relation_depth_trainable_parameters,
    validate_relation_depth_probe_report,
)
from picf_next.lingbot_native.relations import SharedRelationReadout
from picf_next.lingbot_native.capacity import (
    require_checkpoint_write_capacity,
    require_evidence_write_capacity,
    require_persistent_run_root,
)
from picf_next.training.run_lease import acquire_distributed_run_lease
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    validate_fsdp2_placement,
    validate_fsdp2_storage_report,
)
from picf_next.lingbot_native.task_diagnostics import (
    build_task_row_diagnostics,
    validate_task_row_diagnostics,
)
from picf_next.lingbot_native.task_relation import (
    GLOBAL_MULTIPOSITIVE_TASK_RELATION,
    HOST_NATIVE_FACTORIZED_TASK_RELATION,
    LOCAL_BALANCED_TASK_RELATION,
    TASK_RELATION_ESTIMATORS,
)
from picf_next.lingbot_native.supervision import (
    OWNERSHIP_ESTIMATORS,
    TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
    TOKEN_MICRO_OWNERSHIP,
)
from picf_next.lingbot_native.gate_evidence import (
    EMPIRICAL_REPORT_SCHEMAS,
    validate_empirical_gate_report,
    validate_full_weight_smoke_report,
    validate_g0_report,
    validate_g2_visual_review,
    validate_g7_protocol,
    validate_gate_subject,
    validate_preflight_report,
)
from picf_next.lingbot_native.official_config import official_lingbot_data_config
from picf_next.lingbot_native.prompt_tokenization import (
    CompletePromptTokenizationAudit,
    audit_complete_prompt_tokenization,
    validate_distinct_prompt_tokenizations,
)
from picf_next.lingbot_native.predictive_decision import (
    IMPLEMENTED_PREDICTIVE_OBJECTIVE,
    IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
    PREDICTIVE_OBJECTIVE_DECISION_SCHEMA,
    PredictiveObjectiveDecision,
    load_predictive_objective_decision,
    validate_predictive_objective_decision,
)
from picf_next.lingbot_native.representation_baseline import (
    build_representation_baseline_replay_report,
    load_representation_baseline_replay_report,
    load_representation_evaluation_baseline,
    validate_representation_baseline_plan,
    write_representation_baseline_replay_report,
)
from picf_next.lingbot_native.representation_evaluation import (
    REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA,
    RepresentationEvaluationPlan,
)
from picf_next.lingbot_native.fixed_observation_training_contract import (
    validate_fixed_observation_training_rank_metadata,
)
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
    RepresentationTrialSplit,
)
from picf_next.lingbot_native.stage_control import (
    NATIVE_REPRESENTATION_STAGE,
)
from picf_next.lingbot_native.stream_plan import (
    add_reset_mixture_arguments,
    adr121_recurrent_audit_updates,
    adr121_required_optimizer_lag,
    reset_mixture_values,
)

RELATION_FIXED_BATCH_ARMS = (
    *RELATION_GEOMETRY_FIXED_BATCH_ARMS,
    RELATION_BILINEAR_PROBE_ARM,
    RELATION_DEPTH_PROBE_ARM,
)

try:
    from tools.bootstrap_lingbot_vla2 import (
        CHECKPOINT_ASSET_CONTRACT,
        LINGBOT_CHECKPOINT_ID,
        LINGBOT_CHECKPOINT_REVISION,
        PROCESSOR_ASSET_CONTRACT,
        QWEN_PROCESSOR_ID,
        QWEN_PROCESSOR_REVISION,
        asset_contract_manifest,
        validate_checkpoint,
        validate_processor,
    )
    from tools.bootstrap_lingbot_vla2_native import (
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        PATCH_RELATIVE_PATH,
        validate_prepared_native_source,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        _merge_qwen_config,
        _resolve_training_config,
        build_lingbot_fixed_batch_probe_optimizer,
        build_lingbot_official_optimizer,
        build_lingbot_representation_optimizer,
        clip_lingbot_distributed_l2_grad_norm_,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        require_lingbot_exact_resume_contract,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from tools.run_lingbot_vla2_native_g0 import (
        G0_REPORT_SCHEMA,
        _capture_rank_rng,
        _checkpoint_boundary,
        _distributed_gradient_metrics,
        _fsync_tree,
        _implementation_digest,
        _local_import_modules,
        _move_model_inputs,
        _rank_rng_digest,
        _resolve_local_module,
        _restore_rank_rng,
        _validate_fsdp2_parameter_storage,
        _validate_optimizer_state,
        _write_text_durable,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        CHECKPOINT_ASSET_CONTRACT,
        LINGBOT_CHECKPOINT_ID,
        LINGBOT_CHECKPOINT_REVISION,
        PROCESSOR_ASSET_CONTRACT,
        QWEN_PROCESSOR_ID,
        QWEN_PROCESSOR_REVISION,
        asset_contract_manifest,
        validate_checkpoint,
        validate_processor,
    )
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        PATCH_RELATIVE_PATH,
        validate_prepared_native_source,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        _merge_qwen_config,
        _resolve_training_config,
        build_lingbot_fixed_batch_probe_optimizer,
        build_lingbot_official_optimizer,
        build_lingbot_representation_optimizer,
        clip_lingbot_distributed_l2_grad_norm_,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        require_lingbot_exact_resume_contract,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        G0_REPORT_SCHEMA,
        _capture_rank_rng,
        _checkpoint_boundary,
        _distributed_gradient_metrics,
        _fsync_tree,
        _implementation_digest,
        _local_import_modules,
        _move_model_inputs,
        _rank_rng_digest,
        _resolve_local_module,
        _restore_rank_rng,
        _validate_fsdp2_parameter_storage,
        _validate_optimizer_state,
        _write_text_durable,
    )


FULL_WORLD_SIZE = 2
PREDICTIVE_SHARED_HOST_LAYERS = 36
PREDICTIVE_SHARED_HOST_WIDTH = 2560
FULL_PREDICTION_ADDRESS_WIDTH = 2
FULL_RELATION_SUPERVISION_LAYERS = (8, 17, 26)
FULL_COMPARISON_ID = "lingbot-vla2-native-picf-full"
COMPLETE_PROMPT_TOKENIZATION_FILENAME = "complete_prompt_tokenization_audit.json"
BEHAVIOR_CAUSAL_PROBE_DISTRIBUTED_SCHEMA = "picf-next.lingbot-behavior-causal-probe-distributed.v2"
BEHAVIOR_CAUSAL_PROBE_SCHEMA = "picf-next.lingbot-behavior-causal-probe/v2"
BEHAVIOR_POSTERIOR_CONTROL_PROBE_DISTRIBUTED_SCHEMA = (
    "picf-next.lingbot-behavior-posterior-control-probe-distributed.v2"
)
BEHAVIOR_CAUSAL_PROBE_FIELDS = frozenset(
    {
        "checkpoint_revision",
        "behavior_graph_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "optimizer_updates",
        "patch_sha256",
        "plan_sha256",
        "rank_reports",
        "sample_keys_by_rank",
        "schema",
        "source_commit",
        "status",
        "weight_boundary",
    }
)
BEHAVIOR_CAUSAL_PROBE_RANK_FIELDS = frozenset(
    {
        "cold_deploy_omits_future_controls",
        "deploy_bit_identical",
        "deploy_isolation",
        "deploy_tensor_count",
        "elapsed_s",
        "fresh_primary_rerun_bit_identical",
        "horizon",
        "intervention_prediction_changed",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "rank",
        "schema",
        "status",
    }
)
BEHAVIOR_POSTERIOR_CONTROL_PROBE_FIELDS = frozenset(
    {
        "aggregate_control_margins_at_factual_posterior",
        "aggregate_posterior_margins_at_factual_control",
        "behavior_graph_sha256",
        "checkpoint_publication",
        "checkpoint_revision",
        "current_execution_contract_sha256",
        "current_implementation_sha256",
        "g0_evidence_sha256",
        "g1_predecessor_execution_contract_sha256",
        "g1_predecessor_implementation_sha256",
        "g1_predecessor_report_sha256",
        "input_global_step",
        "loaded_boundary_sha256_by_rank",
        "model_family_sha256",
        "optimizer_updates",
        "patch_sha256",
        "plan_sha256",
        "rank_reports",
        "schema",
        "scientific_rule",
        "scientific_status",
        "source_commit",
        "status",
        "weight_boundary",
    }
)
BEHAVIOR_POSTERIOR_CONTROL_RANK_FIELDS = frozenset(
    {
        "assignment",
        "diagnostics",
        "elapsed_s",
        "factual_prediction_sha256",
        "factual_repeat_bit_identical",
        "loaded_boundary_sha256",
        "loss_only_labels_visible_to_model",
        "moe_routing_bias_unchanged",
        "optimizer_state_unchanged",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "peer_sample_keys",
        "posterior_bank_unchanged",
        "rank",
        "request_sha256",
        "rng_sha256",
        "rng_unchanged",
        "sample_keys",
        "target",
        "tasks",
        "training_prediction_bit_identical",
        "training_prediction_sha256",
    }
)
BEHAVIOR_CONDITIONING_SCHEMA = "picf-next.lingbot-behavior-conditioning.v2"
BEHAVIOR_JOINT_CONDITIONING_SCHEMA = "picf-next.lingbot-behavior-conditioning.v3"
BEHAVIOR_CONDITIONING_FIELDS = frozenset(
    {
        "schema",
        "protocol",
        "horizon",
        "isolation",
        "behavior_graph_sha256",
        "g0_evidence_sha256",
    }
)
BEHAVIOR_JOINT_CONDITIONING_FIELDS = frozenset(
    {*BEHAVIOR_CONDITIONING_FIELDS, "g2_evidence_sha256"}
)
BEHAVIOR_GRAPH_SCHEMA = "picf-next.lingbot-behavior-graph.v1"
BEHAVIOR_ACTION_EVIDENCE_MAXIMUM_STEPS = 20
BEHAVIOR_ACTION_DISCRIMINATION_STEPS = 60
BEHAVIOR_ACTION_DISCRIMINATION_AUDIT_STEPS = (2, 10, 20, 40, 60)
MATCHED_MEDIUM_HORIZON_PROFILE = "matched_medium_horizon"
MATCHED_MEDIUM_HORIZON_TOTAL_STEPS = 1000
MATCHED_MEDIUM_HORIZON_AUDIT_STEPS = (18, 34, 50, 100, 200, 500, 1000)
MATCHED_MEDIUM_HORIZON_VISUAL_CADENCE = 200
MATCHED_MEDIUM_HORIZON_SEGMENTS = (
    ("fresh", 0, 1, (0,)),
    ("resume", 1, 199, (200,)),
    ("resume", 200, 300, (500,)),
    ("resume", 500, 500, (1000,)),
)
CONTENT_ADDRESSED_SET_MEDIUM_HORIZON_SEGMENTS = (
    ("fresh", 0, 1, (0,)),
    ("resume", 1, 49, (50,)),
    ("resume", 50, 150, (200,)),
)
REGISTERED_MATCHED_MEDIUM_HORIZON_SEGMENTS = frozenset(
    (*MATCHED_MEDIUM_HORIZON_SEGMENTS, *CONTENT_ADDRESSED_SET_MEDIUM_HORIZON_SEGMENTS)
)
GRADIENT_AUDIT_TEMPORAL_SCOPE = "primary-transition-without-sparse-temporal-auxiliary/v1"
FULL_ACTION_STAGE = "action"
FULL_TRAINING_STAGES = (FULL_ACTION_STAGE, NATIVE_REPRESENTATION_STAGE)
REPRESENTATION_FAMILY_GRADIENT_PROBE = (
    "lingbot.language_model.input_layernorm.early_middle_late.shared_objective"
)
CHECKPOINT_PUBLICATION_MODES = ("always", "never")
FULL_EXTRA_STATE_SCHEMA = "picf-next.lingbot-vla2-native-full-extra-state.v5"
REPRESENTATION_EXTRA_STATE_SCHEMA = "picf-next.lingbot-vla2-native-representation-extra-state.v3"
FULL_REPORT_SCHEMA = "picf-next.lingbot-vla2-native-full.v23"
REPRESENTATION_REPORT_SCHEMA = "picf-next.lingbot-vla2-native-representation.v9"
TRAINING_AUTHORIZATION_SCHEMA = "picf-next.lingbot-native-training-authorization.v5"
TRAINING_GATE_DECISION_SCHEMA = "picf-next.lingbot-native-gate-decision.v3"
TRAINING_AUTHORIZATION_GATES = {
    "pilot": ("G0", "G1", "G2_PROTOCOL"),
    "long": (
        "G0",
        "G1",
        "G2_PROTOCOL",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7_PROTOCOL",
    ),
}
TRAINING_GATE_DECISION_KINDS = {
    "G0": "empirical",
    "G1": "empirical",
    "G2_PROTOCOL": "protocol",
    "G2": "empirical",
    "G3": "empirical",
    "G4": "empirical",
    "G5": "empirical",
    "G6": "empirical",
    "G7_PROTOCOL": "protocol",
}
TRAINING_GATE_EVIDENCE_SCHEMAS = {
    "G0": (
        ("preflight", "picf-next.lingbot-native-preflight.v4"),
        ("neutral", "picf-next.lingbot-vla2-native-full-weight-smoke.v4"),
        ("fresh_update", G0_REPORT_SCHEMA),
        ("cold_resume", G0_REPORT_SCHEMA),
    ),
    "G1": (
        ("static_causality", "picf-next.lingbot-native-preflight.v4"),
        ("released_isolation", "picf-next.lingbot-vla2-native-full-weight-smoke.v4"),
    ),
    "G2_PROTOCOL": (
        ("frozen_local_contract", "picf-next.lingbot-native-preflight.v4"),
        ("predictive_objective_decision", PREDICTIVE_OBJECTIVE_DECISION_SCHEMA),
    ),
    "G2": (
        ("pilot_train", FULL_REPORT_SCHEMA),
        ("object_metrics", EMPIRICAL_REPORT_SCHEMAS["G2"]),
        ("visual_review", "picf-next.lingbot-native-g2-visual-review.v1"),
    ),
    "G3": (("temporal_evaluation", EMPIRICAL_REPORT_SCHEMAS["G3"]),),
    "G4": (("cross_modal_evaluation", EMPIRICAL_REPORT_SCHEMAS["G4"]),),
    "G5": (("matched_action_curves", EMPIRICAL_REPORT_SCHEMAS["G5"]),),
    "G6": (("closed_loop_evaluation", EMPIRICAL_REPORT_SCHEMAS["G6"]),),
    "G7_PROTOCOL": (("second_dataset_protocol", "picf-next.lingbot-native-g7-protocol.v1"),),
}
LOCALLY_IMPLEMENTED_TRAINING_GATE_VALIDATORS = frozenset(TRAINING_GATE_DECISION_KINDS)
PREDICTIVE_BUILD_REPORT_FIELDS = frozenset(
    {
        "cache_manifest_sha256",
        "coverage_sha256",
        "expected_record_count",
        "output_root",
        "pair_keys_sha256",
        "patch_sha256",
        "physical_visual_acceptance_sha256",
        "stream_plan_sha256",
        "teacher_encoder_digest",
        "temporal_estimator_sha256",
    }
)
PREDICTIVE_TARGET_AUDIT_FIELDS = frozenset(
    {
        "cache_contract",
        "cache_manifest_sha256",
        "diagnostics",
        "encoder_digest",
        "horizon_record_counts",
        "identity_count",
        "interpretation",
        "maximum_samples",
        "sample_selection",
        "sample_selection_sha256",
        "sampled_target_count",
        "scanned_object_target_count",
        "scanned_record_count",
        "schema",
        "supported_object_target_count",
        "visible_support_diagnostics",
        "zero_support_object_target_count",
    }
)
PREDICTIVE_TARGET_AUDIT_INTERPRETATION_FIELDS = frozenset(
    {
        "numerical_status",
        "pretraining_readiness",
        "pretraining_readiness_failures",
        "retrieval_is_computable",
        "scientific_acceptance",
        "scientific_acceptance_reason",
    }
)
PREDICTIVE_TARGET_AUDIT_CONTRACT_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_revision",
        "split_name",
        "dataset_tree_sha256",
        "physical_sidecar_manifest_sha256",
        "lingbot_source_commit",
        "lingbot_checkpoint_revision",
        "teacher_config_sha256",
        "teacher_checkpoint_sha256",
        "query_schema_sha256",
        "horizons",
        "stream_plan_sha256",
        "temporal_estimator_sha256",
        "pair_keys_sha256",
        "coverage_sha256",
        "expected_record_count",
        "hidden_size",
        "input_size",
        "patch_tokens",
        "route_id",
        "camera_name",
        "attention_mode",
        "use_warmup_frame",
        "source_fps",
        "effective_fps_semantics",
        "minimum_visible_fraction",
    }
)
PREDICTIVE_TEMPORAL_AUDIT_FIELDS = frozenset(
    {
        "current_cache_manifest_sha256",
        "current_correction_diagnostics",
        "current_correction_identity_count",
        "current_correction_sample_selection_sha256",
        "current_correction_sampled_target_count",
        "current_correction_scanned_object_target_count",
        "current_correction_supported_object_target_count",
        "current_correction_visible_support_diagnostics",
        "current_correction_zero_support_object_target_count",
        "current_encoder_digest",
        "diagnostics",
        "feature_pairing",
        "future_cache_manifest_sha256",
        "future_encoder_digest",
        "horizon_supported_pair_counts",
        "interpretation",
        "matched_future_record_count",
        "maximum_samples",
        "physical_sidecar_manifest_sha256",
        "sample_selection",
        "sample_selection_sha256",
        "sampled_pair_count",
        "scanned_current_record_count",
        "schema",
        "supported_aligned_pair_count",
    }
)
PREDICTIVE_TEMPORAL_AUDIT_INTERPRETATION_FIELDS = frozenset(
    {
        "controlled_future_temporal_pretraining_readiness",
        "controlled_future_temporal_pretraining_readiness_failures",
        "current_correction_pretraining_readiness",
        "current_correction_pretraining_readiness_failures",
        "pretraining_readiness",
        "pretraining_readiness_failures",
        "scientific_acceptance",
        "scientific_acceptance_reason",
    }
)
TEACHER_CAUSALITY_AUDIT_FIELDS = frozenset(
    {
        "current_cache_manifest_sha256",
        "current_encoder_digest",
        "dataset_tree_sha256",
        "diagnostics",
        "patch_sha256",
        "physical_sidecar_manifest_sha256",
        "predictive_cache_manifest_sha256",
        "predictive_encoder_digest",
        "scanned_record_count",
        "schema",
    }
)
TEACHER_CAUSALITY_DIAGNOSTIC_FIELDS = frozenset(
    {
        "current_cache_patch_elements",
        "current_cache_patch_mismatch_count",
        "current_patch_elements",
        "current_patch_mismatch_count",
        "future_feature_elements",
        "future_feature_mismatch_count",
        "future_importance_elements",
        "future_importance_mismatch_count",
        "maximum_current_cache_patch_absolute_error",
        "maximum_current_patch_absolute_error",
        "maximum_future_feature_absolute_error",
        "maximum_future_importance_absolute_error",
        "sample_selection_sha256",
        "sampled_horizon_record_counts",
        "sampled_record_count",
        "same_call_supported_pair_count",
        "same_call_temporal_diagnostics",
        "same_call_temporal_pretraining_readiness",
        "same_call_temporal_pretraining_readiness_failures",
        "status",
    }
)
CURRENT_GRID_BUILD_REPORT_FIELDS = frozenset(
    {
        "cache_manifest_sha256",
        "coverage_sha256",
        "expected_record_count",
        "output_root",
        "patch_sha256",
        "physical_visual_acceptance_sha256",
        "source_keys_sha256",
        "stream_plan_sha256",
        "teacher_encoder_digest",
        "temporal_estimator_sha256",
    }
)
CURRENT_GRID_CONTENT_IDENTICAL_DONOR_FIELDS = frozenset(
    {
        "donor_cache_manifest_sha256",
        "donor_content_manifest_sha256",
        "official_source_receipt_sha256",
        "reused_record_count",
        "target_dataset_manifest_sha256",
    }
)
LEGACY_FULL_OBJECTIVE_CONTRACT_FIELDS = frozenset(
    {
        "predictive_family_weight",
        "structural_family_weight",
        "predictive_term_weight",
        "current_grid_term_weight",
        "omitted_static_term_weight",
        "support_weight",
        "existence_weight",
        "task_weight",
        "dense_task_weight",
        "ownership_weight",
        "family_reduction",
    }
)
PRE_ENTITY_FULL_OBJECTIVE_CONTRACT_FIELDS = LEGACY_FULL_OBJECTIVE_CONTRACT_FIELDS | {
    "task_relation_estimator"
}
FULL_OBJECTIVE_CONTRACT_FIELDS = PRE_ENTITY_FULL_OBJECTIVE_CONTRACT_FIELDS | {"ownership_estimator"}
REPRESENTATION_OBJECTIVE_CONTRACT_FIELDS = FULL_OBJECTIVE_CONTRACT_FIELDS | {"action_family_weight"}
PRE_ENTITY_REPRESENTATION_OBJECTIVE_CONTRACT_FIELDS = PRE_ENTITY_FULL_OBJECTIVE_CONTRACT_FIELDS | {
    "action_family_weight"
}
LEGACY_REPRESENTATION_OBJECTIVE_CONTRACT_FIELDS = LEGACY_FULL_OBJECTIVE_CONTRACT_FIELDS | {
    "action_family_weight"
}


def _ownership_estimator_from_args(args: argparse.Namespace) -> str:
    """Read the opt-in estimator while preserving pre-ADR-134 callers."""

    return (
        args.ownership_estimator if hasattr(args, "ownership_estimator") else TOKEN_MICRO_OWNERSHIP
    )


CHECKPOINT_BOUNDARY_FIELDS = frozenset(
    {
        "lane_snapshot_sha256",
        "model_local_state_sha256",
        "optimizer_local_state_sha256",
        "rank_rng_state_sha256",
    }
)
FULL_REPORT_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "status",
        "source_commit",
        "patch_sha256",
        "checkpoint_revision",
        "execution_contract_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "plan_sha256",
        "temporal_estimator_sha256",
        "dataset_contract",
        "physical_sidecar_manifest_sha256",
        "predictive_cache_manifest_sha256",
        "predictive_teacher_causality_audit_sha256",
        "predictive_target_audit_sha256",
        "predictive_temporal_audit_sha256",
        "current_grid_cache_manifest_sha256",
        "input_global_step",
        "saved_global_step",
        "checkpoint_dir",
        "full_shard",
        "fsdp2_placement",
        "cuda_allocator",
        "gradient_checkpointing",
        "action_loss_enabled",
        "predictive_correction_loss_enabled",
        "behavior_future_loss_enabled",
        "behavior_conditioning",
        "structural_set_loss_enabled",
        "current_source_mask_enabled",
        "omitted_static_binding_enabled",
        "source_prediction_mode",
        "objective_contract",
        "evidence_profile",
        "gradient_audit_steps",
        "gradient_audit_temporal_scope",
        "complete_adr74_objective",
        "training_authorization",
        "long_training_authorized",
        "parameter_storage",
        "alignment_teacher_prune",
        "action_fsdp2_topology",
        "vlm_fsdp2_topology",
        "maximum_peak_reserved_bytes",
        "parameter_manifest",
        "rank_reports",
    }
)


def _validate_task_relation_dense_weight(
    *,
    estimator: str,
    dense_task_weight: float,
    context: str,
) -> None:
    """Keep factorized task grounding free of a competing dense-task loss."""

    if estimator == HOST_NATIVE_FACTORIZED_TASK_RELATION:
        if dense_task_weight != 0:
            raise ValueError(f"{context} factorized task relation requires dense_task_weight=0")
    elif dense_task_weight <= 0:
        raise ValueError(f"{context} non-factorized task relation requires dense_task_weight>0")


REPRESENTATION_REPORT_FIELDS = FULL_REPORT_FIELDS | {
    "checkpoint_publication",
    "training_stage",
    "representation_split_sha256",
    "representation_split_file_sha256",
    "representation_parameter_scope",
    "representation_parameter_scope_sha256",
    "representation_frozen_action_state_sha256",
    "visual_audit_every",
}
REPRESENTATION_PARAMETER_SCOPE_FIELDS = frozenset(
    {
        "schema",
        "production_trainable_sha256",
        "production_frozen_sha256",
        "representation_trainable_sha256",
        "action_frozen_sha256",
        "production_trainable_numel",
        "production_frozen_numel",
        "representation_trainable_numel",
        "action_frozen_numel",
    }
)
FULL_RANK_REPORT_FIELDS = frozenset(
    {"rank", "steps", "saved_boundary_sha256", "loaded_boundary_sha256"}
)
FULL_STEP_REPORT_FIELDS = frozenset(
    {
        "global_step",
        "estimator_component",
        "posterior_committed",
        "posterior_bank_sha256_before",
        "posterior_bank_sha256_after",
        "sample_keys",
        "lane_ids",
        "frame_indices",
        "state_ages",
        "temporal_plan_sha256",
        "local_bptt_steps",
        "overshoot_horizon",
        "source_masked_branch",
        "source_mask_digest",
        "source_mask_query_count",
        "source_prediction_mode",
        "omitted_static_digest",
        "objective_total",
        "official_action_loss",
        "official_moe_regularizer",
        "official_policy_loss",
        "normalized_terms",
        "valid_counts",
        "task_row_diagnostics",
        "prior_row_bindings",
        "row_bindings",
        "row_binding_birth_count",
        "visual_artifacts",
        "visual_audit_seconds",
        "gradient_metrics",
        "family_gradient_diagnostics",
        "relation_surface_gradient_diagnostics",
        "predictive_host_gradient_diagnostics",
        "predictive_counterfactual_diagnostics",
        "predictive_counterfactual_weight_boundary",
        "step_time_s",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
    }
)
REPRESENTATION_STEP_REPORT_FIELDS = FULL_STEP_REPORT_FIELDS | {
    "fixed_observation_pair_sha256",
    "fixed_observation_fingerprint",
}
FULL_GRADIENT_REPORT_FIELDS = frozenset(
    {
        "all_finite",
        "native_graph_norm",
        "native_graph_elements",
        "relation_projection_norm",
        "relation_projection_elements",
        "match_projection_norm",
        "match_projection_elements",
        "action_output_norm",
        "action_output_elements",
        "predictive_readout_norm",
        "predictive_readout_elements",
        "preclip_global_norm",
        "behavior_posterior_gradient_norm",
        "behavior_posterior_gradient_elements",
    }
)
FULL_PARAMETER_MANIFEST_FIELDS = frozenset({"parameter_count", "trainable_numel", "schema_sha256"})
FULL_DATASET_CONTRACT_FIELDS = frozenset(
    {"status", "manifest_sha256", "normalization_sha256", "validation"}
)
FULL_DATASET_VALIDATION_FIELDS = DATASET_RUNTIME_BINDING_FIELDS
FULL_ALIGNMENT_PRUNE_FIELDS = frozenset(
    {"schema", "removed", "removed_numel", "removed_storage_bytes", "retained_query_components"}
)
FULL_ALIGNMENT_REMOVED_FIELDS = frozenset({"name", "parameter_count", "numel", "storage_bytes"})
FULL_VLM_TOPOLOGY_FIELDS = frozenset(
    {"text_block_count", "text_block_paths", "vision_block_count", "vision_block_paths"}
)
FULL_ACTION_TOPOLOGY_FIELDS = frozenset(
    {
        "schema",
        "block_count",
        "block_paths",
        "maximum_block_bf16_bytes_upper_bound",
    }
)
FULL_VISUAL_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "path",
        "sha256",
        "bytes",
        "global_step",
        "input_weight_global_step",
        "weight_boundary",
        "rank",
        "batch_index",
        "sample_key",
        "task",
        "identity_keys",
        "source_time",
        "source_side",
        "source_phase",
        "binding_start_phase",
        "source_binding_valid",
        "row_to_track",
        "sequence_row_to_track",
        "row_existence",
        "row_task_relevance",
        "row_matched_soft_iou",
        "anchor_surface",
        "views",
        "loss_only_labels_visible_to_model",
    }
)
FULL_VISUAL_VIEW_FIELDS = frozenset({"name", "merged_grid", "source_shape", "token_count"})


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def _require_runtime_storage_capacity(
    run_dir: Path,
    *,
    checkpoint_required: bool = True,
) -> int:
    """Fail before model loading unless the declared artifact can fit."""

    if checkpoint_required:
        return require_checkpoint_write_capacity(run_dir)
    return require_evidence_write_capacity(run_dir)


def validate_behavior_causal_probe_evidence(value: object) -> dict[str, Any]:
    """Recompute the complete two-rank released-weight G0 receipt contract."""

    if (
        not isinstance(value, dict)
        or set(value) != BEHAVIOR_CAUSAL_PROBE_FIELDS
        or value.get("schema") != BEHAVIOR_CAUSAL_PROBE_DISTRIBUTED_SCHEMA
        or value.get("status") != "PASS"
        or value.get("optimizer_updates") != 0
        or value.get("weight_boundary") != "released_pre_optimizer"
    ):
        raise ValueError("behavior causal-probe evidence is not a passed G0 receipt")
    for field in (
        "behavior_graph_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "patch_sha256",
        "plan_sha256",
    ):
        _require_sha256(f"behavior causal-probe {field}", value[field])
    if (
        value["source_commit"] != LINGBOT_NATIVE_SOURCE_COMMIT
        or value["checkpoint_revision"] != LINGBOT_CHECKPOINT_REVISION
    ):
        raise ValueError("behavior causal-probe evidence belongs to another LingBot release")
    rank_reports = value["rank_reports"]
    sample_keys = value["sample_keys_by_rank"]
    if (
        not isinstance(rank_reports, list)
        or len(rank_reports) != FULL_WORLD_SIZE
        or not isinstance(sample_keys, list)
        or len(sample_keys) != FULL_WORLD_SIZE
    ):
        raise ValueError("behavior causal-probe evidence lacks two-rank coverage")
    observed_ranks: set[int] = set()
    for rank_report in rank_reports:
        if (
            not isinstance(rank_report, dict)
            or set(rank_report) != BEHAVIOR_CAUSAL_PROBE_RANK_FIELDS
            or rank_report.get("schema") != BEHAVIOR_CAUSAL_PROBE_SCHEMA
            or rank_report.get("status") != "PASS"
            or rank_report.get("cold_deploy_omits_future_controls") is not True
            or rank_report.get("deploy_bit_identical") is not True
            or rank_report.get("fresh_primary_rerun_bit_identical") is not True
            or rank_report.get("deploy_isolation") != "separate_same_weight_auxiliary_forward"
        ):
            raise ValueError("behavior causal-probe rank evidence failed deploy isolation")
        rank = rank_report["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < FULL_WORLD_SIZE:
            raise ValueError("behavior causal-probe rank is invalid")
        observed_ranks.add(rank)
        _positive_report_integer(
            f"behavior causal-probe rank {rank} deploy tensor count",
            rank_report["deploy_tensor_count"],
        )
        horizon = _positive_report_integer(
            f"behavior causal-probe rank {rank} horizon",
            rank_report["horizon"],
        )
        if horizon != 1:
            raise ValueError("behavior causal-probe horizon differs from the frozen graph")
        _finite_report_number(
            f"behavior causal-probe rank {rank} elapsed seconds",
            rank_report["elapsed_s"],
            positive=True,
        )
        allocated = _positive_report_integer(
            f"behavior causal-probe rank {rank} allocated bytes",
            rank_report["peak_cuda_allocated_bytes"],
        )
        reserved = _positive_report_integer(
            f"behavior causal-probe rank {rank} reserved bytes",
            rank_report["peak_cuda_reserved_bytes"],
        )
        if allocated > reserved:
            raise ValueError("behavior causal-probe CUDA memory accounting is invalid")
        interventions = rank_report["intervention_prediction_changed"]
        if (
            not isinstance(interventions, dict)
            or set(interventions) != {"peer_replace_shuffle", "reverse", "zero"}
            or any(not isinstance(changed, bool) for changed in interventions.values())
            or interventions["peer_replace_shuffle"] is not True
            or interventions["zero"] is not True
        ):
            raise ValueError("behavior causal-probe prediction intervention evidence failed")
    if observed_ranks != set(range(FULL_WORLD_SIZE)):
        raise ValueError("behavior causal-probe evidence duplicated a distributed rank")
    observed_sample_keys: set[str] = set()
    for per_rank in sample_keys:
        if (
            not isinstance(per_rank, list)
            or not per_rank
            or any(not isinstance(key, str) or not key for key in per_rank)
        ):
            raise ValueError("behavior causal-probe sample provenance is malformed")
        if observed_sample_keys.intersection(per_rank):
            raise ValueError("behavior causal-probe ranks consumed overlapping samples")
        observed_sample_keys.update(per_rank)
    return value


def load_behavior_causal_probe_evidence(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load one exact released-weight G0 receipt before a bounded G1 update."""

    expected = _require_sha256("behavior causal-probe evidence sha256", expected_sha256)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("behavior causal-probe evidence is not a regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("behavior causal-probe evidence is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("behavior causal-probe evidence differs from its expected digest")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("behavior causal-probe evidence is not valid JSON") from error
    return validate_behavior_causal_probe_evidence(value)


def load_behavior_g1_predecessor_report(
    run_dir: Path,
    *,
    load_global_step: int,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Load the exact immutable G1 report that defines the G2 weight boundary."""

    expected = _require_sha256("behavior G1 predecessor report sha256", expected_sha256)
    path = (
        run_dir
        / "checkpoints"
        / f"global_step_{load_global_step}"
        / "native_representation_report.json"
    )
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("behavior G1 predecessor report is not a regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("behavior G1 predecessor report is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError("behavior G1 predecessor report differs from its expected digest")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("behavior G1 predecessor report is not valid JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != REPRESENTATION_REPORT_SCHEMA
        or value.get("status") != "PASS"
        or value.get("checkpoint_publication") != "always"
        or value.get("saved_global_step") != load_global_step
        or value.get("input_global_step") != load_global_step - 1
        or value.get("behavior_conditioning") is None
    ):
        raise ValueError("behavior G1 predecessor is not the passed cold-resume boundary")
    for field in (
        "execution_contract_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "plan_sha256",
    ):
        _require_sha256(f"behavior G1 predecessor {field}", value.get(field))
    return value, actual


def validate_behavior_posterior_control_probe_evidence(
    value: object,
) -> dict[str, Any]:
    """Recompute the zero-update G2 receipt before joint action adoption."""

    if (
        not isinstance(value, dict)
        or set(value) != BEHAVIOR_POSTERIOR_CONTROL_PROBE_FIELDS
        or value.get("schema") != BEHAVIOR_POSTERIOR_CONTROL_PROBE_DISTRIBUTED_SCHEMA
        or value.get("status") != "PASS"
        or value.get("scientific_status") != "PASS"
        or value.get("optimizer_updates") != 0
        or value.get("checkpoint_publication") != "never"
        or value.get("input_global_step") != 2
        or value.get("weight_boundary") != "loaded_g1_step2_pre_optimizer"
    ):
        raise ValueError("behavior posterior/control evidence is not a passed G2 receipt")
    if (
        value["source_commit"] != LINGBOT_NATIVE_SOURCE_COMMIT
        or value["checkpoint_revision"] != LINGBOT_CHECKPOINT_REVISION
    ):
        raise ValueError("behavior G2 evidence belongs to another LingBot release")
    for field in (
        "behavior_graph_sha256",
        "current_execution_contract_sha256",
        "current_implementation_sha256",
        "g0_evidence_sha256",
        "g1_predecessor_execution_contract_sha256",
        "g1_predecessor_implementation_sha256",
        "g1_predecessor_report_sha256",
        "model_family_sha256",
        "patch_sha256",
        "plan_sha256",
    ):
        _require_sha256(f"behavior G2 {field}", value[field])
    if (
        value["current_implementation_sha256"] != value["g1_predecessor_implementation_sha256"]
        or value["current_execution_contract_sha256"]
        != value["g1_predecessor_execution_contract_sha256"]
    ):
        raise ValueError("behavior G2 current code differs from its G1 predecessor")

    posterior_names = ("zero", "batch_shift")
    control_names = ("zero", "batch_shift")
    aggregate_posterior = value["aggregate_posterior_margins_at_factual_control"]
    aggregate_control = value["aggregate_control_margins_at_factual_posterior"]
    if (
        not isinstance(aggregate_posterior, dict)
        or set(aggregate_posterior) != set(posterior_names)
        or not isinstance(aggregate_control, dict)
        or set(aggregate_control) != set(control_names)
    ):
        raise ValueError("behavior G2 aggregate factorial margins are malformed")

    rank_reports = value["rank_reports"]
    boundaries = value["loaded_boundary_sha256_by_rank"]
    if (
        not isinstance(rank_reports, list)
        or len(rank_reports) != FULL_WORLD_SIZE
        or not isinstance(boundaries, list)
        or len(boundaries) != FULL_WORLD_SIZE
    ):
        raise ValueError("behavior G2 evidence lacks two-rank coverage")
    observed_ranks: set[int] = set()
    posterior_values = {name: [] for name in posterior_names}
    control_values = {name: [] for name in control_names}
    for index, report in enumerate(rank_reports):
        if not isinstance(report, dict) or set(report) != (BEHAVIOR_POSTERIOR_CONTROL_RANK_FIELDS):
            raise ValueError("behavior G2 rank report is malformed")
        rank = report["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < FULL_WORLD_SIZE:
            raise ValueError("behavior G2 rank is invalid")
        observed_ranks.add(rank)
        for field in (
            "factual_repeat_bit_identical",
            "moe_routing_bias_unchanged",
            "optimizer_state_unchanged",
            "posterior_bank_unchanged",
            "rng_unchanged",
            "training_prediction_bit_identical",
        ):
            if report[field] is not True:
                raise ValueError(f"behavior G2 rank {rank} failed {field}")
        if report["loss_only_labels_visible_to_model"] is not False:
            raise ValueError("behavior G2 exposed loss-only labels to the model")
        for field in (
            "factual_prediction_sha256",
            "request_sha256",
            "rng_sha256",
            "training_prediction_sha256",
        ):
            _require_sha256(f"behavior G2 rank {rank} {field}", report[field])
        if report["training_prediction_sha256"] != report["factual_prediction_sha256"]:
            raise ValueError("behavior G2 factual probe differs from its training prediction")
        target = report["target"]
        if (
            not isinstance(target, dict)
            or set(target)
            != {
                "encoder_digest",
                "importance_max",
                "importance_min",
                "importance_sum",
                "modality",
                "query_schema_digest",
                "source_batch_digest",
                "target_data_digest",
                "track_identity_keys",
                "valid_count",
                "validity_semantics",
            }
            or not isinstance(target["modality"], str)
            or not target["modality"]
            or not isinstance(target["validity_semantics"], str)
            or not target["validity_semantics"]
            or not isinstance(target["track_identity_keys"], list)
            or not target["track_identity_keys"]
        ):
            raise ValueError("behavior G2 target evidence is malformed")
        for field in (
            "encoder_digest",
            "query_schema_digest",
            "source_batch_digest",
            "target_data_digest",
        ):
            _require_sha256(f"behavior G2 rank {rank} target {field}", target[field])
        _positive_report_integer(
            f"behavior G2 rank {rank} target valid count",
            target["valid_count"],
        )
        importance_min = _finite_report_number(
            f"behavior G2 rank {rank} target importance minimum",
            target["importance_min"],
            positive=True,
        )
        importance_max = _finite_report_number(
            f"behavior G2 rank {rank} target importance maximum",
            target["importance_max"],
            positive=True,
        )
        importance_sum = _finite_report_number(
            f"behavior G2 rank {rank} target importance sum",
            target["importance_sum"],
            positive=True,
        )
        if importance_min > importance_max or importance_sum < importance_max:
            raise ValueError("behavior G2 target importance evidence is inconsistent")
        assignment = report["assignment"]
        if not isinstance(assignment, dict) or set(assignment) != {
            "binding_start_phase",
            "identity_source_phase",
            "row_binding_valid",
            "row_to_track",
            "sha256",
        }:
            raise ValueError("behavior G2 assignment evidence is malformed")
        row_to_track = assignment["row_to_track"]
        binding_start_phase = assignment["binding_start_phase"]
        row_binding_valid = assignment["row_binding_valid"]
        source_phase = assignment["identity_source_phase"]
        if (
            not isinstance(row_to_track, list)
            or not row_to_track
            or not isinstance(binding_start_phase, list)
            or len(binding_start_phase) != len(row_to_track)
            or not isinstance(row_binding_valid, list)
            or len(row_binding_valid) != len(row_to_track)
            or isinstance(source_phase, bool)
            or not isinstance(source_phase, int)
            or source_phase < 0
        ):
            raise ValueError("behavior G2 assignment source cut is malformed")
        expected_valid: list[list[bool]] = []
        for tracks, phases, validity in zip(
            row_to_track,
            binding_start_phase,
            row_binding_valid,
            strict=True,
        ):
            if (
                not isinstance(tracks, list)
                or not tracks
                or not isinstance(phases, list)
                or len(phases) != len(tracks)
                or not isinstance(validity, list)
                or len(validity) != len(tracks)
                or any(isinstance(item, bool) or not isinstance(item, int) for item in tracks)
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in phases
                )
                or any(not isinstance(item, bool) for item in validity)
            ):
                raise ValueError("behavior G2 assignment rows are malformed")
            expected_valid.append(
                [
                    track >= 0 and phase <= source_phase
                    for track, phase in zip(tracks, phases, strict=True)
                ]
            )
        if row_binding_valid != expected_valid:
            raise ValueError("behavior G2 row validity differs from its causal source cut")
        assignment_payload = {
            "row_to_track": row_to_track,
            "binding_start_phase": binding_start_phase,
            "identity_source_phase": source_phase,
            "row_binding_valid": row_binding_valid,
        }
        if assignment["sha256"] != _canonical_digest(assignment_payload):
            raise ValueError("behavior G2 assignment digest is inconsistent")
        _validate_report_boundary(
            f"behavior G2 rank {rank} loaded",
            report["loaded_boundary_sha256"],
        )
        if boundaries[index] != report["loaded_boundary_sha256"]:
            raise ValueError("behavior G2 loaded-boundary ordering differs")
        elapsed = _finite_report_number(
            f"behavior G2 rank {rank} elapsed seconds", report["elapsed_s"], positive=True
        )
        del elapsed
        allocated = _positive_report_integer(
            f"behavior G2 rank {rank} allocated bytes", report["peak_cuda_allocated_bytes"]
        )
        reserved = _positive_report_integer(
            f"behavior G2 rank {rank} reserved bytes", report["peak_cuda_reserved_bytes"]
        )
        if allocated > reserved:
            raise ValueError("behavior G2 CUDA memory accounting is invalid")
        diagnostics = report["diagnostics"]
        if not isinstance(diagnostics, dict):
            raise ValueError("behavior G2 rank diagnostics are malformed")
        posterior = diagnostics.get("posterior_margins_at_factual_control")
        control = diagnostics.get("control_margins_at_factual_posterior")
        if (
            not isinstance(posterior, dict)
            or set(posterior) != set(posterior_names)
            or not isinstance(control, dict)
            or set(control) != set(control_names)
        ):
            raise ValueError("behavior G2 rank margins are malformed")
        for name in posterior_names:
            posterior_values[name].append(
                _finite_report_number(f"behavior G2 posterior margin {name}", posterior[name])
            )
        for name in control_names:
            control_values[name].append(
                _finite_report_number(f"behavior G2 control margin {name}", control[name])
            )
    if observed_ranks != set(range(FULL_WORLD_SIZE)):
        raise ValueError("behavior G2 evidence duplicated a distributed rank")

    recomputed_posterior = {
        name: sum(posterior_values[name]) / FULL_WORLD_SIZE for name in posterior_names
    }
    recomputed_control = {
        name: sum(control_values[name]) / FULL_WORLD_SIZE for name in control_names
    }
    if recomputed_posterior != aggregate_posterior or recomputed_control != (aggregate_control):
        raise ValueError("behavior G2 aggregate margins differ from rank evidence")
    if not all(
        margin > 0 for margin in (*recomputed_posterior.values(), *recomputed_control.values())
    ):
        raise ValueError("behavior G2 posterior/control necessity did not pass")
    return value


def load_behavior_posterior_control_probe_evidence(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Load one exact G2 receipt without following a mutable link."""

    expected = _require_sha256("behavior G2 evidence sha256", expected_sha256)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("behavior G2 evidence is not a regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("behavior G2 evidence is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError("behavior G2 evidence differs from its expected digest")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("behavior G2 evidence is not valid JSON") from error
    return validate_behavior_posterior_control_probe_evidence(value), actual


def require_behavior_causal_probe_context(
    evidence: Mapping[str, Any],
    *,
    patch_sha256: str,
    implementation_sha256: str,
    model_family_sha256: str,
    plan_sha256: str,
    behavior_graph_sha256: str,
) -> None:
    """Bind a loaded G0 receipt to the exact G1 scientific context."""

    expected = {
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "patch_sha256": _require_sha256("behavior G1 patch", patch_sha256),
        "implementation_sha256": _require_sha256(
            "behavior G1 implementation",
            implementation_sha256,
        ),
        "model_family_sha256": _require_sha256(
            "behavior G1 model family",
            model_family_sha256,
        ),
        "plan_sha256": _require_sha256("behavior G1 plan", plan_sha256),
        "behavior_graph_sha256": _require_sha256(
            "behavior G1 graph",
            behavior_graph_sha256,
        ),
    }
    if any(evidence.get(field) != value for field, value in expected.items()):
        raise RuntimeError("behavior G0 evidence belongs to another scientific context")


def _behavior_probe_sample_keys_by_rank(
    gathered_metadata: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    """Normalize distributed tuple provenance to the persisted JSON contract."""

    return [list(item["sample_keys"]) for item in gathered_metadata]


def _behavior_graph_contract(args: argparse.Namespace) -> dict[str, object] | None:
    """Build the stage-independent graph identity shared by G0 and G1."""

    if not bool(getattr(args, "behavior_conditioned_prediction", False)):
        return None
    representation_split = getattr(args, "representation_split", None)
    if not isinstance(representation_split, Path):
        raise ValueError("behavior graph requires one representation split")
    return {
        "schema": BEHAVIOR_GRAPH_SCHEMA,
        "release": {
            "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
            "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
            "patch_sha256": _sha256(args.patch),
        },
        "data": {
            "training_config_sha256": _sha256(args.training_config),
            "robot_config_sha256": _sha256(args.robot_config),
            "data_config_sha256": _sha256(args.data_config),
            "dataset_manifest_sha256": _sha256(args.dataset_manifest),
            "normalization_sha256": _sha256(args.norm_stats),
            "representation_split_sha256": _sha256(representation_split),
        },
        "targets": {
            "physical_sidecar_manifest_sha256": _require_sha256(
                "behavior physical-sidecar manifest",
                args.physical_sidecar_manifest_sha256,
            ),
            "physical_visual_acceptance_sha256": _require_sha256(
                "behavior physical visual acceptance",
                args.physical_visual_acceptance_sha256,
            ),
            "predictive_build_report_sha256": _require_sha256(
                "behavior predictive build report",
                args.predictive_cache_build_report_sha256,
            ),
            "predictive_teacher_causality_sha256": _require_sha256(
                "behavior teacher-causality audit",
                args.predictive_teacher_causality_audit_sha256,
            ),
            "predictive_target_audit_sha256": _require_sha256(
                "behavior predictive target audit",
                args.predictive_target_audit_sha256,
            ),
            "predictive_temporal_audit_sha256": _require_sha256(
                "behavior predictive temporal audit",
                args.predictive_temporal_audit_sha256,
            ),
            "current_grid_build_report_sha256": _require_sha256(
                "behavior current-grid build report",
                args.current_grid_cache_build_report_sha256,
            ),
        },
        "objective": {
            "predictive_family_weight": float(args.predictive_weight).hex(),
            "structural_family_weight": float(args.structural_weight).hex(),
            "support_weight": float(args.support_weight).hex(),
            "existence_weight": float(args.existence_weight).hex(),
            "task_weight": float(args.task_weight).hex(),
            "dense_task_weight": float(args.dense_task_weight).hex(),
            "ownership_weight": float(args.ownership_weight).hex(),
            "ownership_estimator": _ownership_estimator_from_args(args),
            "predictive_term_weight": float(args.predictive_term_weight).hex(),
            "current_grid_term_weight": float(args.current_grid_term_weight).hex(),
            "omitted_static_term_weight": float(args.omitted_static_term_weight).hex(),
            "predictive_loss_power": float(args.predictive_loss_power).hex(),
            "minimum_supervised_fraction": float(args.minimum_supervised_fraction).hex(),
            "task_relation_estimator": args.task_relation_estimator,
        },
        "sampling": {
            "comparison_id": FULL_COMPARISON_ID,
            "seed": args.seed,
            "total_planned_steps": args.total_planned_steps,
            "lane_interleave_factor": args.lane_interleave_factor,
            "local_bptt_probability": float(args.local_bptt_probability).hex(),
            "overshoot_probability": float(args.overshoot_probability).hex(),
            "source_mask_probability": float(args.source_mask_probability).hex(),
            "source_prediction_mode": args.source_prediction_mode,
            "source_mask_token_fraction": float(args.source_mask_token_fraction).hex(),
        },
        "topology": {
            "world_size": FULL_WORLD_SIZE,
            "training_stage": NATIVE_REPRESENTATION_STAGE,
            "capacity": args.capacity,
            "maximum_control_tokens": args.maximum_control_tokens,
            "maximum_optimizer_lag": args.maximum_optimizer_lag,
            "relation_supervision_layers": list(args.relation_supervision_layers),
            "fsdp2_placement": args.fsdp2_placement,
            "cuda_allocator": args.cuda_allocator,
        },
        "behavior": {
            "horizon": 1,
            "isolation": "separate_same_weight_auxiliary_forward",
            "target": "rollout/vision/binding",
        },
    }


def _behavior_graph_digest(args: argparse.Namespace) -> str | None:
    contract = _behavior_graph_contract(args)
    return None if contract is None else _canonical_digest(contract)


def _require_unchanged_behavior_graph(
    args: argparse.Namespace,
    *,
    expected_sha256: str | None,
) -> None:
    """Fail closed if any scientific graph input drifted during an invocation."""

    actual = _behavior_graph_digest(args)
    if actual != expected_sha256:
        raise RuntimeError("behavior graph inputs changed during the invocation")


def _behavior_conditioning_contract(
    args: argparse.Namespace,
    *,
    behavior_graph_sha256: str | None,
) -> dict[str, object] | None:
    if not bool(getattr(args, "behavior_conditioned_prediction", False)):
        if behavior_graph_sha256 is not None:
            raise ValueError("ordinary training received a behavior graph identity")
        return None
    frozen_graph = _require_sha256("behavior graph", behavior_graph_sha256)
    g0_sha256 = getattr(args, "behavior_causal_probe_evidence_sha256", None)
    if g0_sha256 is not None:
        g0_sha256 = _require_sha256("behavior causal-probe evidence sha256", g0_sha256)
    g2_sha256 = getattr(
        args,
        "behavior_posterior_control_probe_evidence_sha256",
        None,
    )
    if g2_sha256 is not None:
        g2_sha256 = _require_sha256("behavior G2 evidence sha256", g2_sha256)
    if g2_sha256 is not None:
        if getattr(args, "training_stage", NATIVE_REPRESENTATION_STAGE) != FULL_ACTION_STAGE:
            raise ValueError("behavior G2 evidence can authorize only joint action training")
        return {
            "schema": BEHAVIOR_JOINT_CONDITIONING_SCHEMA,
            "protocol": "g2_approved_joint_action",
            "horizon": 1,
            "isolation": "separate_same_weight_auxiliary_forward",
            "behavior_graph_sha256": frozen_graph,
            "g0_evidence_sha256": g0_sha256,
            "g2_evidence_sha256": g2_sha256,
        }
    return {
        "schema": BEHAVIOR_CONDITIONING_SCHEMA,
        "protocol": (
            "g0_released_weight_causal_probe"
            if getattr(args, "behavior_causal_probe_output", None) is not None
            else "g1_first_update_and_cold_resume"
        ),
        "horizon": 1,
        "isolation": "separate_same_weight_auxiliary_forward",
        "behavior_graph_sha256": frozen_graph,
        "g0_evidence_sha256": g0_sha256,
    }


def _validate_behavior_conditioning_contract(
    value: object,
    *,
    allow_g0: bool = False,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("behavior-conditioning contract is malformed")
    schema = value.get("schema")
    expected_fields = (
        BEHAVIOR_JOINT_CONDITIONING_FIELDS
        if schema == BEHAVIOR_JOINT_CONDITIONING_SCHEMA
        else BEHAVIOR_CONDITIONING_FIELDS
    )
    if set(value) != expected_fields:
        raise ValueError("behavior-conditioning contract is malformed")
    if (
        schema not in {BEHAVIOR_CONDITIONING_SCHEMA, BEHAVIOR_JOINT_CONDITIONING_SCHEMA}
        or value["horizon"] != 1
        or value["isolation"] != "separate_same_weight_auxiliary_forward"
    ):
        raise ValueError("behavior-conditioning contract changed its frozen graph")
    _require_sha256("behavior-conditioning graph", value["behavior_graph_sha256"])
    if schema == BEHAVIOR_JOINT_CONDITIONING_SCHEMA:
        if value["protocol"] != "g2_approved_joint_action":
            raise ValueError("joint behavior-conditioning protocol is unsupported")
        _require_sha256("behavior-conditioning G0 evidence", value["g0_evidence_sha256"])
        _require_sha256("behavior-conditioning G2 evidence", value["g2_evidence_sha256"])
        return value
    protocols = {"g1_first_update_and_cold_resume"}
    if allow_g0:
        protocols.add("g0_released_weight_causal_probe")
    if value["protocol"] not in protocols:
        raise ValueError("behavior-conditioning protocol is unsupported")
    g0_sha256 = value["g0_evidence_sha256"]
    if value["protocol"] == "g1_first_update_and_cold_resume":
        _require_sha256("behavior-conditioning G0 evidence", g0_sha256)
    elif g0_sha256 is not None:
        raise ValueError("behavior G0 fabricated a prior G0 evidence receipt")
    return value


def _behavior_conditioning_digest(
    args: argparse.Namespace,
    *,
    behavior_graph_sha256: str | None,
) -> str | None:
    contract = _behavior_conditioning_contract(
        args,
        behavior_graph_sha256=behavior_graph_sha256,
    )
    return None if contract is None else _canonical_digest(contract)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _posterior_bank_digest(bank: Any) -> str:
    """Read the serialized lane-bank digest without assuming a callable API."""

    return _require_sha256("posterior bank digest", getattr(bank, "digest", None))


def _objective_result_posterior_state(result: Any) -> Any:
    """Return the deploy posterior from either native objective result shape."""

    primary = result.primary
    primary_context = getattr(primary, "context", primary)
    posterior = getattr(primary_context, "posterior_state", None)
    if posterior is None:
        raise RuntimeError("native objective result omitted the deploy posterior state")
    return posterior


def _parse_positive_step_set(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "gradient audit steps must be comma-separated integers"
        ) from error
    if not steps or any(step <= 0 for step in steps) or tuple(sorted(set(steps))) != steps:
        raise argparse.ArgumentTypeError(
            "gradient audit steps must be sorted unique positive integers"
        )
    return steps


def _parse_nonnegative_step_set(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "representation evaluation steps must be comma-separated integers"
        ) from error
    if not steps or any(step < 0 for step in steps) or tuple(sorted(set(steps))) != steps:
        raise argparse.ArgumentTypeError(
            "representation evaluation steps must be sorted unique non-negative integers"
        )
    return steps


def _parse_nonnegative_layer_set(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "relation supervision layers must be comma-separated integers"
        ) from error
    if not layers or any(layer < 0 for layer in layers) or tuple(sorted(set(layers))) != layers:
        raise argparse.ArgumentTypeError(
            "relation supervision layers must be sorted unique non-negative integers"
        )
    return layers


def validate_training_gate_evidence(
    *,
    gate: str,
    name: str,
    value: object,
) -> dict[str, Any]:
    """Validate one gate-specific machine or review report before it can authorize work."""

    expected = dict(TRAINING_GATE_EVIDENCE_SCHEMAS.get(gate, ())).get(name)
    if expected is None:
        raise ValueError("training gate evidence name is outside the frozen schema")
    if (
        not isinstance(value, dict)
        or value.get("status") != "PASS"
        or value.get("schema") != expected
    ):
        raise ValueError(f"{name} evidence does not match its passed report schema")
    if gate not in LOCALLY_IMPLEMENTED_TRAINING_GATE_VALIDATORS:
        raise ValueError(
            f"{gate} evidence validator is not implemented; long training remains forbidden"
        )
    if name in {
        "preflight",
        "static_causality",
        "frozen_local_contract",
    }:
        validate_preflight_report(
            value,
            schema=expected,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=FULL_WORLD_SIZE,
        )
    if name == "predictive_objective_decision":
        validate_predictive_objective_decision(value)
    if name in {"neutral", "released_isolation"}:
        validate_full_weight_smoke_report(
            value,
            schema=expected,
            implementation_sha256=_implementation_digest(Path(__file__).resolve().parents[1]),
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        )
    if gate == "G0" and name in {"fresh_update", "cold_resume"}:
        expected_fsdp2_placement = validate_fsdp2_placement(value.get("fsdp2_placement"))
        expected_cuda_allocator = value.get("cuda_allocator")
        if expected_cuda_allocator not in CUDA_ALLOCATOR_MODES:
            raise ValueError("G0 evidence has no explicit CUDA allocator")
        validate_g0_report(
            value,
            schema=expected,
            phase="fresh" if name == "fresh_update" else "resume",
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            world_size=FULL_WORLD_SIZE,
            expected_fsdp2_placement=expected_fsdp2_placement,
            expected_cuda_allocator=expected_cuda_allocator,
        )
    if gate == "G2" and name == "pilot_train":
        expected_fsdp2_placement = validate_fsdp2_placement(value.get("fsdp2_placement"))
        expected_cuda_allocator = value.get("cuda_allocator")
        if expected_cuda_allocator not in CUDA_ALLOCATOR_MODES:
            raise ValueError("G2 evidence has no explicit CUDA allocator")
        validate_full_objective_report(
            value,
            expected_saved_global_step=value.get("saved_global_step"),
            require_initial_probe=False,
            require_mature_wrong_time=True,
            require_source_evidence=True,
            expected_fsdp2_placement=expected_fsdp2_placement,
            expected_cuda_allocator=expected_cuda_allocator,
        )
        if value.get("input_global_step", 0) <= 0 or value.get("saved_global_step", 201) > 200:
            raise ValueError(
                "G2 pilot evidence must be a resumed checkpoint no later than step 200"
            )
    elif gate == "G2" and name == "object_metrics":
        validate_empirical_gate_report(value, gate=gate, schema=expected)
    elif gate == "G2" and name == "visual_review":
        validate_g2_visual_review(value, schema=expected)
    elif gate in {"G3", "G4", "G5", "G6"}:
        validate_empirical_gate_report(value, gate=gate, schema=expected)
    elif gate == "G7_PROTOCOL":
        validate_g7_protocol(value, schema=expected)
    return value


def predictive_objective_decision_from_gate_decision(
    value: Mapping[str, Any],
    *,
    expected_temporal_objective: str | None = None,
    expected_visible_support_weighting: str | None = None,
    expected_minimum_visible_fraction: float | None = None,
) -> PredictiveObjectiveDecision:
    """Reopen the owner decision bound inside one validated G2 protocol gate."""

    if value.get("gate") != "G2_PROTOCOL" or value.get("status") != "PASS":
        raise ValueError("predictive owner decision requires one passed G2 protocol gate")
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("G2 protocol evidence is malformed")
    matches = [
        item
        for item in evidence
        if isinstance(item, Mapping) and item.get("name") == "predictive_objective_decision"
    ]
    if len(matches) != 1:
        raise ValueError("G2 protocol does not bind exactly one predictive owner decision")
    reference = matches[0]
    if set(reference) != {"name", "path", "sha256"}:
        raise ValueError("predictive owner-decision reference is malformed")
    path_value = reference["path"]
    path = Path(path_value) if isinstance(path_value, str) else None
    if path is None:
        raise ValueError("predictive owner-decision path is malformed")
    return load_predictive_objective_decision(
        path,
        expected_sha256=reference["sha256"],
        expected_temporal_objective=expected_temporal_objective,
        expected_visible_support_weighting=expected_visible_support_weighting,
        expected_minimum_visible_fraction=expected_minimum_visible_fraction,
    )


def training_gate_decision_subject(
    *,
    gate: str,
    criteria_sha256: str,
    evidence: tuple[tuple[str, bytes, dict[str, Any]], ...],
) -> dict[str, Any]:
    """Derive the immutable subject shared by one gate's evidence reports."""

    criteria_digest = _require_sha256("training gate criteria sha256", criteria_sha256)
    expected_names = tuple(name for name, _schema in TRAINING_GATE_EVIDENCE_SCHEMAS[gate])
    if tuple(name for name, _payload, _value in evidence) != expected_names:
        raise ValueError("training gate subject evidence order differs from the frozen schema")
    if gate == "G0":
        fresh = evidence[2][2]
        resumed = evidence[3][2]
        shared_fields = (
            "source_commit",
            "patch_sha256",
            "patched_source_sha256",
            "checkpoint_revision",
            "execution_contract_sha256",
            "implementation_sha256",
            "model_family_sha256",
            "plan_sha256",
            "dataset_contract",
            "parameter_storage",
            "parameter_manifest",
            "maximum_peak_reserved_bytes",
            "fsdp2_placement",
            "cuda_allocator",
        )
        if any(fresh[field] != resumed[field] for field in shared_fields):
            raise ValueError("G0 fresh and cold-resume reports use different frozen contracts")
        fresh_ranks = {item["rank"]: item for item in fresh["rank_reports"]}
        resumed_ranks = {item["rank"]: item for item in resumed["rank_reports"]}
        if any(
            resumed_ranks[rank]["loaded_boundary_sha256"]
            != fresh_ranks[rank]["saved_boundary_sha256"]
            for rank in range(FULL_WORLD_SIZE)
        ):
            raise ValueError("G0 cold resume did not load the fresh rank checkpoint boundaries")
        return {
            "criteria_sha256": criteria_digest,
            "fresh_report_sha256": hashlib.sha256(evidence[2][1]).hexdigest(),
            "cold_resume_report_sha256": hashlib.sha256(evidence[3][1]).hexdigest(),
            "execution_contract_sha256": fresh["execution_contract_sha256"],
            "implementation_sha256": fresh["implementation_sha256"],
            "model_family_sha256": fresh["model_family_sha256"],
            "plan_sha256": fresh["plan_sha256"],
            "fsdp2_placement": fresh["fsdp2_placement"],
            "cuda_allocator": fresh["cuda_allocator"],
        }
    if gate not in {"G2", "G3", "G4", "G5", "G6", "G7_PROTOCOL"}:
        return {"criteria_sha256": criteria_digest}

    if gate == "G2":
        pilot_payload = evidence[0][1]
        pilot = evidence[0][2]
        subject = {
            "input_full_report_sha256": hashlib.sha256(pilot_payload).hexdigest(),
            "saved_global_step": pilot["saved_global_step"],
            "execution_contract_sha256": pilot["execution_contract_sha256"],
            "implementation_sha256": pilot["implementation_sha256"],
            "model_family_sha256": pilot["model_family_sha256"],
        }
        candidate_reports = (evidence[1][2], evidence[2][2])
        if evidence[1][2]["protocol"]["criteria_sha256"] != criteria_digest:
            raise ValueError("G2 object metrics use another frozen criteria document")
        if evidence[2][2]["criteria_sha256"] != criteria_digest:
            raise ValueError("G2 visual review uses another frozen criteria document")
    else:
        report = evidence[0][2]
        subject = report["subject"]
        candidate_reports = (report,)
        reported_criteria = (
            report["criteria_sha256"]
            if gate == "G7_PROTOCOL"
            else report["protocol"]["criteria_sha256"]
        )
        if reported_criteria != criteria_digest:
            raise ValueError(f"{gate} evidence uses another frozen criteria document")

    validate_gate_subject(subject)
    for report in candidate_reports:
        if report["subject"] != subject:
            raise ValueError(f"{gate} evidence reports target different checkpoints")
    return {"criteria_sha256": criteria_digest, **subject}


def training_authorization_acceptance_subject(
    *,
    stage: str,
    input_report: Mapping[str, Any],
    input_report_sha256: str,
) -> dict[str, Any] | None:
    """Return the immutable evaluated checkpoint inherited by a training segment."""

    if stage not in TRAINING_AUTHORIZATION_GATES:
        raise ValueError("training authorization stage is unsupported")
    digest = _require_sha256("training input report sha256", input_report_sha256)
    if stage == "pilot":
        return None
    if input_report.get("long_training_authorized") is True:
        prior = input_report.get("training_authorization")
        if not isinstance(prior, Mapping):
            raise ValueError("long continuation report lacks its prior authorization")
        subject = prior.get("acceptance_subject")
        return validate_gate_subject(subject)
    subject = {
        "input_full_report_sha256": digest,
        "saved_global_step": input_report.get("saved_global_step"),
        "execution_contract_sha256": input_report.get("execution_contract_sha256"),
        "implementation_sha256": input_report.get("implementation_sha256"),
        "model_family_sha256": input_report.get("model_family_sha256"),
    }
    return validate_gate_subject(subject)


def pilot_authorization_requires_initial_probe(
    *,
    stage: str,
    input_report: Mapping[str, Any],
    maximum_global_step: int,
    visual_audit_every: int,
) -> bool:
    """Validate a bounded pilot continuation and identify a fresh pilot input."""

    if stage != "pilot":
        return False
    input_interval_start = input_report.get("input_global_step")
    if (
        isinstance(input_interval_start, bool)
        or not isinstance(input_interval_start, int)
        or input_interval_start < 0
    ):
        raise ValueError("pilot authorization input interval is malformed")
    if input_interval_start == 0:
        return True
    prior = input_report.get("training_authorization")
    if (
        not isinstance(prior, Mapping)
        or prior.get("stage") != "pilot"
        or prior.get("visual_audit_every") != visual_audit_every
    ):
        raise ValueError("pilot continuation requires the same prior pilot and visual cadence")
    prior_maximum = prior.get("maximum_global_step")
    if (
        isinstance(prior_maximum, bool)
        or not isinstance(prior_maximum, int)
        or maximum_global_step > prior_maximum
    ):
        raise ValueError("pilot continuation cannot widen its prior authorization")
    return False


def _finite_report_number(name: str, value: object, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return float(value)


def _validate_step_row_bindings(
    *,
    prior_value: object,
    current_value: object,
    reported_birth_count: object,
    task_row_diagnostics: object,
    expected_batch_size: int,
) -> tuple[tuple[dict[str, int], dict[str, int]], ...]:
    """Validate persisted sequence gauges against exact loss-side diagnostics."""

    if (
        not isinstance(prior_value, list)
        or not isinstance(current_value, list)
        or len(prior_value) != expected_batch_size
        or len(current_value) != expected_batch_size
        or not isinstance(task_row_diagnostics, list)
        or len(task_row_diagnostics) != expected_batch_size
    ):
        raise ValueError("full-objective row-binding batches are malformed")

    births = 0
    resolved: list[tuple[dict[str, int], dict[str, int]]] = []
    for batch_index, (prior_pairs, current_pairs, diagnostic) in enumerate(
        zip(prior_value, current_value, task_row_diagnostics, strict=True)
    ):
        if not isinstance(diagnostic, dict):
            raise ValueError("full-objective row-binding diagnostic is malformed")
        task_logits = diagnostic.get("task_logits")
        identity_keys = diagnostic.get("identity_keys")
        row_to_track = diagnostic.get("row_to_track")
        binding_start_phase = diagnostic.get("binding_start_phase")
        if (
            not isinstance(task_logits, list)
            or not isinstance(identity_keys, list)
            or not isinstance(row_to_track, list)
            or len(row_to_track) != len(task_logits)
            or not isinstance(binding_start_phase, list)
            or len(binding_start_phase) != len(task_logits)
            or any(
                isinstance(phase, bool) or not isinstance(phase, int) or phase < 0
                for phase in binding_start_phase
            )
        ):
            raise ValueError("full-objective row-binding axes are malformed")
        capacity = len(task_logits)

        def parse(name: str, value: object, *, row_capacity: int) -> dict[str, int]:
            if not isinstance(value, list):
                raise ValueError(f"full-objective {name} must be a list")
            parsed: dict[str, int] = {}
            used_rows: set[int] = set()
            previous_identity: str | None = None
            for pair in value:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                    or not pair[0]
                    or isinstance(pair[1], bool)
                    or not isinstance(pair[1], int)
                    or not 0 <= pair[1] < row_capacity
                    or pair[0] in parsed
                    or pair[1] in used_rows
                    or (previous_identity is not None and pair[0] <= previous_identity)
                ):
                    raise ValueError(f"full-objective {name} is not canonical")
                parsed[pair[0]] = pair[1]
                used_rows.add(pair[1])
                previous_identity = pair[0]
            return parsed

        prior = parse(
            f"rank step batch {batch_index} prior row bindings",
            prior_pairs,
            row_capacity=capacity,
        )
        current = parse(
            f"rank step batch {batch_index} row bindings",
            current_pairs,
            row_capacity=capacity,
        )
        if any(current.get(identity) != row for identity, row in prior.items()):
            raise ValueError("full-objective row binding removed or rebound an identity")
        births += len(current) - len(prior)
        for row, (track_index, start_phase) in enumerate(
            zip(row_to_track, binding_start_phase, strict=True)
        ):
            if track_index < 0:
                continue
            if track_index >= len(identity_keys):
                raise ValueError("full-objective row assignment references an absent identity")
            identity = identity_keys[track_index]
            persisted_row = current.get(identity)
            if start_phase <= 1 and persisted_row != row:
                raise ValueError("full-objective assignment and persisted row binding disagree")
            if start_phase > 1 and persisted_row is not None:
                raise ValueError("full-objective persisted a loss-only future row binding")
        resolved.append((prior, current))

    if (
        isinstance(reported_birth_count, bool)
        or not isinstance(reported_birth_count, int)
        or reported_birth_count != births
    ):
        raise ValueError("full-objective row-binding birth count differs")
    return tuple(resolved)


def _advance_report_row_binding_continuity(
    lane_bindings: dict[int, dict[str, int]],
    *,
    lane_ids: list[int],
    frame_indices: list[int],
    state_ages: list[int],
    step_bindings: tuple[tuple[dict[str, int], dict[str, int]], ...],
) -> None:
    """Link every reported prior gauge to the preceding report for that lane."""

    if not (len(lane_ids) == len(frame_indices) == len(state_ages) == len(step_bindings)):
        raise ValueError("full-objective row-binding continuity axes differ")
    if len(set(lane_ids)) != len(lane_ids):
        raise ValueError("full-objective step repeats a temporal lane")
    for lane_id, frame_index, state_age, (prior, current) in zip(
        lane_ids,
        frame_indices,
        state_ages,
        step_bindings,
        strict=True,
    ):
        if frame_index != state_age:
            raise ValueError("full-objective lane frame and state age differ")
        if state_age == 0:
            if prior:
                raise ValueError("full-objective reset lane retained prior row bindings")
        else:
            previously_reported = lane_bindings.get(lane_id)
            if previously_reported is not None and prior != previously_reported:
                raise ValueError("full-objective row bindings break cross-step lane continuity")
        lane_bindings[lane_id] = current


def _validate_step_estimator_transaction(item: Mapping[str, Any]) -> bool:
    """Validate persisted reset/causal state-transaction evidence."""

    component = item.get("estimator_component")
    committed = item.get("posterior_committed")
    before = item.get("posterior_bank_sha256_before")
    after = item.get("posterior_bank_sha256_after")
    if component not in {"causal", "reset"} or not isinstance(committed, bool):
        raise ValueError("full-objective estimator transaction provenance is malformed")
    _require_sha256("full-objective posterior bank before", before)
    _require_sha256("full-objective posterior bank after", after)
    if component == "reset":
        if committed or before != after:
            raise ValueError("full-objective reset step changed or committed posterior state")
        if any(item["frame_indices"]) or any(item["state_ages"]):
            raise ValueError("full-objective reset estimator consumed a warm state")
        if any(item["prior_row_bindings"]):
            raise ValueError("full-objective reset estimator retained prior row bindings")
        return False
    if not committed or before == after:
        raise ValueError("full-objective causal step did not publish posterior state")
    return True


def _validate_report_boundary(name: str, value: object) -> None:
    if not isinstance(value, dict) or set(value) != CHECKPOINT_BOUNDARY_FIELDS:
        raise ValueError(f"{name} checkpoint boundary is incomplete")
    for field, digest in value.items():
        _require_sha256(f"{name} {field}", digest)


def _validate_family_gradient_report(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "all_finite",
        "cosines",
        "dot_products",
        "gradient_norms",
        "probe",
        "world_size",
    }:
        raise ValueError("full-objective family-gradient diagnostic is malformed")
    if (
        value["all_finite"] is not True
        or value["probe"] != "picf_native_graph.object_queries"
        or value["world_size"] != FULL_WORLD_SIZE
    ):
        raise ValueError("full-objective family-gradient diagnostic failed its probe contract")
    norms = value["gradient_norms"]
    if not isinstance(norms, dict) or set(norms) != {"action", "predictive", "structural"}:
        raise ValueError("full-objective family-gradient norms are incomplete")
    for family, norm in norms.items():
        _finite_report_number(f"{family} family-gradient norm", norm, positive=True)
    pair_names = {"action__predictive", "action__structural", "predictive__structural"}
    for field in ("cosines", "dot_products"):
        values = value[field]
        if not isinstance(values, dict) or set(values) != pair_names:
            raise ValueError(f"full-objective family-gradient {field} are incomplete")
        for pair, measured in values.items():
            finite = _finite_report_number(f"{pair} family-gradient {field}", measured)
            if field == "cosines" and not -1.000_001 <= finite <= 1.000_001:
                raise ValueError(f"{pair} family-gradient cosine lies outside [-1, 1]")


def _validate_representation_family_gradient_report(value: object) -> None:
    """Require predictive/structural host learning with an exactly inactive action family."""

    if not isinstance(value, dict) or set(value) != {
        "all_finite",
        "cosines",
        "dot_products",
        "gradient_norms",
        "probe",
        "world_size",
    }:
        raise ValueError("representation family-gradient diagnostic is malformed")
    if (
        value["all_finite"] is not True
        or value["probe"] != REPRESENTATION_FAMILY_GRADIENT_PROBE
        or value["world_size"] != FULL_WORLD_SIZE
    ):
        raise ValueError("representation family-gradient diagnostic failed its probe contract")
    norms = value["gradient_norms"]
    if not isinstance(norms, dict) or set(norms) != {"action", "predictive", "structural"}:
        raise ValueError("representation family-gradient norms are incomplete")
    if _finite_report_number("representation action family-gradient norm", norms["action"]) != 0:
        raise ValueError("representation action family reached the native graph")
    for family in ("predictive", "structural"):
        _finite_report_number(
            f"representation {family} family-gradient norm",
            norms[family],
            positive=True,
        )

    pair_names = {"action__predictive", "action__structural", "predictive__structural"}
    dot_products = value["dot_products"]
    cosines = value["cosines"]
    if (
        not isinstance(dot_products, dict)
        or set(dot_products) != pair_names
        or not isinstance(cosines, dict)
        or set(cosines) != pair_names
    ):
        raise ValueError("representation family-gradient pair coverage is incomplete")
    for pair in ("action__predictive", "action__structural"):
        if (
            _finite_report_number(
                f"representation {pair} family-gradient dot product",
                dot_products[pair],
            )
            != 0
            or cosines[pair] is not None
        ):
            raise ValueError("representation action-family conflict gauge is not exactly inactive")
    _finite_report_number(
        "representation predictive__structural family-gradient dot product",
        dot_products["predictive__structural"],
    )
    predictive_structural_cosine = _finite_report_number(
        "representation predictive__structural family-gradient cosine",
        cosines["predictive__structural"],
    )
    if not -1.000_001 <= predictive_structural_cosine <= 1.000_001:
        raise ValueError("representation predictive/structural cosine lies outside [-1, 1]")


def _validate_relation_surface_gradient_report(
    value: object,
    *,
    estimator: str,
) -> None:
    """Validate loss-component geometry without requiring a favorable outcome."""

    if estimator not in TASK_RELATION_ESTIMATORS:
        raise ValueError("relation-surface gradient estimator is invalid")
    fields = {
        "all_finite",
        "cosines",
        "dot_products",
        "gradient_elements",
        "gradient_norms",
        "probe",
        "world_size",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("relation-surface gradient diagnostic is malformed")
    if (
        value["all_finite"] is not True
        or value["probe"] != "final_relation.match_embeddings+row_embeddings"
        or value["world_size"] != FULL_WORLD_SIZE
    ):
        raise ValueError("relation-surface gradient diagnostic failed its probe contract")

    norms = value["gradient_norms"]
    elements = value["gradient_elements"]
    if not isinstance(norms, dict) or not isinstance(elements, dict) or set(norms) != set(elements):
        raise ValueError("relation-surface gradient coverage is incomplete")
    try:
        gradient_names, norm_pairs = _relation_surface_gradient_contract(norms)
    except RuntimeError as error:
        raise ValueError("relation-surface gradient coverage is incomplete") from error
    factorized = "task_row@match_embeddings" in gradient_names
    if factorized != (estimator == HOST_NATIVE_FACTORIZED_TASK_RELATION):
        raise ValueError("relation-surface gradient schema differs from task relation estimator")
    for name in gradient_names:
        norm = _finite_report_number(f"{name} relation-surface gradient norm", norms[name])
        if norm < 0 or (factorized and norm <= 0):
            qualifier = "non-positive" if factorized else "negative"
            raise ValueError(f"{name} relation-surface gradient norm is {qualifier}")
        _positive_report_integer(
            f"{name} relation-surface gradient elements",
            elements[name],
        )

    expected_pairs = {pair for pair, _left, _right in norm_pairs}
    cosines = value["cosines"]
    dot_products = value["dot_products"]
    if (
        not isinstance(cosines, dict)
        or set(cosines) != expected_pairs
        or not isinstance(dot_products, dict)
        or set(dot_products) != expected_pairs
    ):
        raise ValueError("relation-surface gradient pair coverage is incomplete")
    for pair, left, right in norm_pairs:
        _finite_report_number(
            f"{pair} relation-surface gradient dot product",
            dot_products[pair],
        )
        cosine = cosines[pair]
        denominator_is_zero = norms[left] == 0 or norms[right] == 0
        if denominator_is_zero:
            if cosine is not None:
                raise ValueError(f"{pair} zero-norm relation-surface cosine is defined")
            continue
        measured = _finite_report_number(
            f"{pair} relation-surface gradient cosine",
            cosine,
        )
        if not -1.000_001 <= measured <= 1.000_001:
            raise ValueError(f"{pair} relation-surface cosine lies outside [-1, 1]")


def _validate_predictive_host_gradient_report(
    value: object,
    *,
    expected_probe: str = "lingbot.language_model.input_layernorm",
) -> None:
    fields = {
        "all_finite",
        "decomposition",
        "gradient_elements",
        "gradient_norms",
        "parameter_paths",
        "probe",
        "world_size",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("full-objective predictive-host gradient diagnostic is malformed")
    if (
        value["all_finite"] is not True
        or value["probe"] != expected_probe
        or value["world_size"] != FULL_WORLD_SIZE
    ):
        raise ValueError("full-objective predictive-host gradient diagnostic failed")
    expected = {"early", "middle", "late"}
    norms = value["gradient_norms"]
    elements = value["gradient_elements"]
    paths = value["parameter_paths"]
    if any(
        not isinstance(item, dict) or set(item) != expected for item in (norms, elements, paths)
    ):
        raise ValueError("full-objective predictive-host layer coverage is incomplete")
    if len(set(paths.values())) != len(expected) or any(
        not isinstance(path, str) or not path.endswith(".input_layernorm.weight")
        for path in paths.values()
    ):
        raise ValueError("full-objective predictive-host parameter paths are malformed")
    for depth in sorted(expected):
        _finite_report_number(
            f"predictive-host {depth} gradient norm",
            norms[depth],
            positive=True,
        )
        element_count = _positive_report_integer(
            f"predictive-host {depth} gradient elements",
            elements[depth],
        )
        if element_count != PREDICTIVE_SHARED_HOST_WIDTH:
            raise ValueError(
                f"predictive-host {depth} gradient must cover exactly "
                f"{PREDICTIVE_SHARED_HOST_WIDTH} elements"
            )
    decomposition = value["decomposition"]
    behavior_probe = expected_probe.endswith(".via_primary_posterior_vjp")
    if not behavior_probe:
        if decomposition is not None:
            raise ValueError("ordinary predictive-host audit fabricated behavior decomposition")
        return
    if (
        not isinstance(decomposition, dict)
        or set(decomposition) != {"components", "depths", "identity"}
        or decomposition["components"] != ["total", "via_posterior", "direct"]
        or decomposition["identity"] != "weighted_behavior_total=direct+via_primary_posterior"
        or not isinstance(decomposition["depths"], dict)
        or set(decomposition["depths"]) != expected
    ):
        raise ValueError("behavior-host gradient decomposition is malformed")
    decomposition_fields = {
        "total_norm",
        "via_posterior_norm",
        "direct_norm",
        "via_to_total_norm_ratio",
        "total_via_cosine",
        "total_direct_cosine",
        "closure_error_norm",
    }
    for depth, measured in decomposition["depths"].items():
        if not isinstance(measured, dict) or set(measured) != decomposition_fields:
            raise ValueError("behavior-host gradient decomposition fields differ")
        total_norm = _finite_report_number(
            f"behavior-host {depth} total norm",
            measured["total_norm"],
            positive=True,
        )
        _finite_report_number(
            f"behavior-host {depth} posterior norm",
            measured["via_posterior_norm"],
            positive=True,
        )
        direct_norm = _finite_report_number(
            f"behavior-host {depth} direct norm",
            measured["direct_norm"],
        )
        ratio = _finite_report_number(
            f"behavior-host {depth} posterior/total ratio",
            measured["via_to_total_norm_ratio"],
            positive=True,
        )
        if direct_norm < 0 or ratio <= 0:
            raise ValueError("behavior-host gradient decomposition has an invalid norm")
        for name in ("total_via_cosine", "total_direct_cosine"):
            cosine = _finite_report_number(
                f"behavior-host {depth} {name}",
                measured[name],
            )
            if not -1.000_001 <= cosine <= 1.000_001:
                raise ValueError("behavior-host gradient cosine lies outside [-1, 1]")
        closure = _finite_report_number(
            f"behavior-host {depth} closure error",
            measured["closure_error_norm"],
        )
        if closure < 0 or closure > 1e-5 * max(total_norm, 1.0):
            raise ValueError("behavior-host gradient decomposition does not close")


def _positive_report_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _temporal_execution_counts(
    *,
    local_bptt_steps: object,
    overshoot_horizon: object,
) -> tuple[int, int]:
    """Translate optional sampled auxiliaries into actual executed work counts."""

    if local_bptt_steps is None:
        local_count = 1
    elif (
        isinstance(local_bptt_steps, bool)
        or not isinstance(local_bptt_steps, int)
        or not 2 <= local_bptt_steps <= 4
    ):
        raise ValueError("sampled local BPTT must be absent or contain 2..4 steps")
    else:
        local_count = local_bptt_steps
    if overshoot_horizon is None:
        overshoot_count = 0
    elif (
        isinstance(overshoot_horizon, bool)
        or not isinstance(overshoot_horizon, int)
        or not 1 <= overshoot_horizon <= 64
    ):
        raise ValueError("sampled overshoot must be absent or contain 1..64 steps")
    else:
        overshoot_count = overshoot_horizon
    return local_count, overshoot_count


def _fixed_observation_primary_temporal_plan(
    temporal: Any,
    *,
    pair_plan_sha256: str | None,
) -> Any:
    """Restrict truth-audited fixed-X resets to their audited primary frame."""

    if pair_plan_sha256 is None:
        return temporal
    _require_sha256("fixed-observation pair plan", pair_plan_sha256)
    from picf_next.lingbot_native.temporal import TemporalBatchPlan

    if not isinstance(temporal, TemporalBatchPlan):
        raise TypeError("fixed-observation temporal restriction requires a typed plan")
    return TemporalBatchPlan(
        seed=temporal.seed,
        state_ages=temporal.state_ages,
        local_bptt_steps=None,
        overshoot_horizon=None,
        source_masked_branch=temporal.source_masked_branch,
    )


def _validate_gradient_audit_target_coverage(
    *,
    stream_plan: Any,
    audit_steps: tuple[int, ...],
    source_global_index_for_sample: Callable[[str], int],
    target_has_support: Callable[..., bool],
) -> None:
    """Require every preregistered family-gradient probe to observe supported targets.

    The check depends only on the immutable sample plan and target cache, never on
    model outputs.  It runs before weight materialization so an invalid audit
    schedule cannot waste an accelerator launch or turn missing/occluded labels
    into a false optimization failure.
    """

    if not callable(source_global_index_for_sample) or not callable(target_has_support):
        raise TypeError("gradient target audit requires callable target resolvers")
    if 1 in audit_steps:
        raise ValueError("family-gradient audit cannot run on the fresh state-bootstrap step")

    def ineligible_at(saved_step: int) -> tuple[str, ...]:
        batch = stream_plan.global_batch(saved_step - 1)
        failures: list[str] = []
        for transition in batch.transitions:
            sample_key = transition.sample.sample_key
            if transition.transition_index <= 0:
                failures.append(f"{sample_key}:no_prior")
                continue
            source = source_global_index_for_sample(sample_key)
            if isinstance(source, bool) or not isinstance(source, int) or source < 0:
                raise ValueError("gradient target audit resolved an invalid source index")
            if not target_has_support(source_global_index=source):
                failures.append(f"{sample_key}:zero_supported_target_mass")
        return tuple(failures)

    failures: list[str] = []
    for saved_step in audit_steps:
        ineligible = ineligible_at(saved_step)
        if not ineligible:
            continue
        replacement = next(
            (
                candidate
                for candidate in range(saved_step + 1, stream_plan.total_steps + 1)
                if not ineligible_at(candidate)
            ),
            None,
        )
        failures.append(
            f"step={saved_step} ineligible={list(ineligible)!r} next_eligible={replacement!r}"
        )
    if failures:
        raise ValueError(
            "family-gradient audit schedule contains target-ineligible frozen samples: "
            + "; ".join(failures)
        )


def _validate_full_report_static_surfaces(
    value: Mapping[str, Any],
    *,
    expected_fsdp2_placement: str,
    expected_cuda_allocator: str,
) -> None:
    if set(value) != FULL_REPORT_FIELDS:
        raise ValueError("input full-objective report fields differ from schema")
    if value["source_commit"] != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ValueError("input full-objective report uses another LingBot source commit")
    if value["checkpoint_revision"] != LINGBOT_CHECKPOINT_REVISION:
        raise ValueError("input full-objective report uses another released checkpoint")
    _require_sha256("full-objective patch", value["patch_sha256"])
    behavior = _validate_behavior_conditioning_contract(value["behavior_conditioning"])
    if (
        not isinstance(value["predictive_correction_loss_enabled"], bool)
        or not isinstance(value["behavior_future_loss_enabled"], bool)
        or value["behavior_future_loss_enabled"] is (behavior is None)
        or value["predictive_correction_loss_enabled"] is (behavior is not None)
    ):
        raise ValueError("full-objective behavior/correction graph flags are inconsistent")
    if value["gradient_audit_temporal_scope"] != GRADIENT_AUDIT_TEMPORAL_SCOPE:
        raise ValueError("full-objective gradient-audit temporal scope changed")

    dataset = value["dataset_contract"]
    if (
        not isinstance(dataset, dict)
        or set(dataset) != FULL_DATASET_CONTRACT_FIELDS
        or dataset["status"] != "PASS"
    ):
        raise ValueError("full-objective dataset contract is malformed")
    _require_sha256("full-objective dataset manifest", dataset["manifest_sha256"])
    _require_sha256("full-objective normalization", dataset["normalization_sha256"])
    validation = dataset["validation"]
    if not isinstance(validation, dict) or set(validation) != FULL_DATASET_VALIDATION_FIELDS:
        raise ValueError("full-objective dataset validation is malformed")
    try:
        validate_dataset_runtime_binding_report(validation)
    except ContractError as error:
        raise ValueError("full-objective dataset verified-read contract changed") from error

    placement = validate_fsdp2_placement(expected_fsdp2_placement)
    if value["fsdp2_placement"] != placement:
        raise ValueError(
            "input full-objective report differs from the expected FSDP2 execution contract"
        )
    if value["cuda_allocator"] != expected_cuda_allocator:
        raise ValueError(
            "input full-objective report differs from the expected CUDA allocator contract"
        )
    validate_fsdp2_storage_report(
        value["parameter_storage"],
        expected_placement=placement,
    )
    manifest = value["parameter_manifest"]
    if not isinstance(manifest, dict) or set(manifest) != FULL_PARAMETER_MANIFEST_FIELDS:
        raise ValueError("full-objective parameter manifest is malformed")
    _positive_report_integer("full-objective parameter count", manifest["parameter_count"])
    _positive_report_integer("full-objective trainable elements", manifest["trainable_numel"])
    _require_sha256("full-objective parameter schema", manifest["schema_sha256"])
    _positive_report_integer(
        "full-objective maximum reserved bytes",
        value["maximum_peak_reserved_bytes"],
    )

    prune = value["alignment_teacher_prune"]
    if not isinstance(prune, dict) or set(prune) != FULL_ALIGNMENT_PRUNE_FIELDS:
        raise ValueError("full-objective alignment teacher-prune report is malformed")
    removed = prune["removed"]
    retained = prune["retained_query_components"]
    if (
        prune["schema"] != "picf-next.targetless-alignment-teacher-prune.v1"
        or not isinstance(removed, list)
        or not removed
        or not isinstance(retained, list)
        or not retained
        or len(set(retained)) != len(retained)
        or any(not isinstance(name, str) or not name for name in retained)
    ):
        raise ValueError("full-objective alignment teacher-prune coverage is malformed")
    removed_numel = 0
    removed_storage = 0
    removed_names: set[str] = set()
    for item in removed:
        if not isinstance(item, dict) or set(item) != FULL_ALIGNMENT_REMOVED_FIELDS:
            raise ValueError("full-objective removed teacher head is malformed")
        name = item["name"]
        if not isinstance(name, str) or not name or name in removed_names:
            raise ValueError("full-objective removed teacher-head name is invalid")
        _positive_report_integer("removed teacher parameters", item["parameter_count"])
        removed_numel += _positive_report_integer("removed teacher elements", item["numel"])
        removed_storage += _positive_report_integer(
            "removed teacher storage bytes", item["storage_bytes"]
        )
        removed_names.add(name)
    if prune["removed_numel"] != removed_numel or prune["removed_storage_bytes"] != removed_storage:
        raise ValueError("full-objective teacher-prune totals were not recomputed")

    action_topology = value["action_fsdp2_topology"]
    if (
        not isinstance(action_topology, dict)
        or set(action_topology) != FULL_ACTION_TOPOLOGY_FIELDS
        or action_topology["schema"] != "picf-next.lingbot-action-block-fsdp2.v1"
    ):
        raise ValueError("full-objective action FSDP2 topology is malformed")
    action_paths = action_topology["block_paths"]
    if (
        not isinstance(action_paths, list)
        or not action_paths
        or len(set(action_paths)) != len(action_paths)
        or any(not isinstance(path, str) or not path for path in action_paths)
        or action_topology["block_count"] != len(action_paths)
    ):
        raise ValueError("full-objective action FSDP2 topology is inconsistent")
    _positive_report_integer(
        "full-objective maximum action block BF16 bytes",
        action_topology["maximum_block_bf16_bytes_upper_bound"],
    )

    topology = value["vlm_fsdp2_topology"]
    if not isinstance(topology, dict) or set(topology) != FULL_VLM_TOPOLOGY_FIELDS:
        raise ValueError("full-objective VLM FSDP2 topology is malformed")
    for kind in ("text", "vision"):
        paths = topology[f"{kind}_block_paths"]
        count = topology[f"{kind}_block_count"]
        if (
            not isinstance(paths, list)
            or not paths
            or len(set(paths)) != len(paths)
            or any(not isinstance(path, str) or not path for path in paths)
            or count != len(paths)
        ):
            raise ValueError(f"full-objective VLM {kind} FSDP2 topology is inconsistent")


def _ownership_depth_indices(
    normalized_terms: Mapping[str, Any],
) -> tuple[int, ...]:
    """Validate and return the final-plus-intermediate ownership depth gauge."""

    indices: set[int] = set()
    diagnostic_indices: set[int] = set()
    entity_indices: set[int] = set()
    for name in normalized_terms:
        if name == "set/ownership":
            indices.add(0)
        elif name.startswith("set/ownership_q"):
            suffix = name.removeprefix("set/ownership_q")
            if not suffix.isdigit() or int(suffix) <= 0:
                raise ValueError("full-objective ownership depth term is malformed")
            indices.add(int(suffix))
        elif name == "set/ownership_nll":
            diagnostic_indices.add(0)
        elif name.startswith("set/ownership_nll_q"):
            suffix = name.removeprefix("set/ownership_nll_q")
            if not suffix.isdigit() or int(suffix) <= 0:
                raise ValueError("full-objective ownership NLL depth term is malformed")
            diagnostic_indices.add(int(suffix))
        elif name == "set/ownership_entity":
            entity_indices.add(0)
        elif name.startswith("set/ownership_entity_q"):
            suffix = name.removeprefix("set/ownership_entity_q")
            if not suffix.isdigit() or int(suffix) <= 0:
                raise ValueError("full-objective entity ownership depth term is malformed")
            entity_indices.add(int(suffix))
    ordered = tuple(sorted(indices))
    if ordered != tuple(range(len(ordered))):
        raise ValueError("full-objective ownership depth terms must be contiguous from final")
    if diagnostic_indices and diagnostic_indices != indices:
        raise ValueError("full-objective ownership NLL depths differ from ownership supervision")
    if entity_indices and entity_indices != indices:
        raise ValueError("full-objective entity ownership depths differ from ownership supervision")
    return ordered


def _validate_task_relation_term_schema(
    normalized_terms: Mapping[str, Any],
    *,
    estimator: str,
    ownership_estimator: str,
) -> int:
    """Bind each estimator to one exact structural objective schema."""

    indices = _ownership_depth_indices(normalized_terms)
    task_term = "set/task_row" if estimator == HOST_NATIVE_FACTORIZED_TASK_RELATION else "set/task"
    expected = {
        "set/support",
        "set/existence",
        task_term,
        "set/task_dense",
    }
    for index in indices:
        suffix = "" if index == 0 else f"_q{index}"
        expected.update({f"set/ownership{suffix}", f"set/ownership_nll{suffix}"})
        if ownership_estimator == TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP:
            expected.add(f"set/ownership_entity{suffix}")
    observed = {name for name in normalized_terms if name.startswith("set/")}
    if observed != expected:
        raise ValueError(
            f"{estimator} structural term schema differs: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return len(indices)


def _objective_term_weight(
    name: str,
    *,
    mode: str,
    contract: Mapping[str, Any],
    ownership_estimator: str,
    ownership_depth_count: int = 1,
) -> float:
    if name.startswith(("correction/", "rollout/")):
        return float(contract["predictive_term_weight"])
    if name.startswith("xmod/"):
        field = {
            "current_grid": "current_grid_term_weight",
            "omitted_static": "omitted_static_term_weight",
        }[mode]
        return float(contract[field])
    if name == "set/ownership_nll" or name.startswith("set/ownership_nll_q"):
        return 0.0
    if name == "set/ownership" or name.startswith("set/ownership_q"):
        if ownership_depth_count <= 0:
            raise ValueError("full-objective ownership depth count must be positive")
        fraction = 0.5 if ownership_estimator == TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP else 1.0
        return float(contract["ownership_weight"]) * fraction / ownership_depth_count
    if name == "set/ownership_entity" or name.startswith("set/ownership_entity_q"):
        if ownership_estimator != TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP:
            raise ValueError("entity ownership term requires the entity-conditional estimator")
        if ownership_depth_count <= 0:
            raise ValueError("full-objective ownership depth count must be positive")
        return float(contract["ownership_weight"]) * 0.5 / ownership_depth_count
    structural = {
        "set/support": "support_weight",
        "set/existence": "existence_weight",
        "set/task": "task_weight",
        "set/task_row": "task_weight",
        "set/task_dense": "dense_task_weight",
    }
    try:
        return float(contract[structural[name]])
    except KeyError as error:
        raise ValueError(f"full-objective term {name} has no registered weight") from error


def _recompute_report_objective(
    *,
    item: Mapping[str, Any],
    mode: str,
    contract: Mapping[str, Any],
) -> None:
    normalized = item["normalized_terms"]
    valid_counts = item["valid_counts"]
    policy = float(item["official_policy_loss"])
    action = float(item["official_action_loss"])
    moe = float(item["official_moe_regularizer"])
    if not math.isclose(policy, action + moe, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("full-objective official policy loss is not action plus MoE")
    if not math.isclose(float(normalized["action"]), policy, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("full-objective normalized action differs from official policy loss")

    ownership_estimator = str(contract.get("ownership_estimator", TOKEN_MICRO_OWNERSHIP))
    ownership_depth_count = _validate_task_relation_term_schema(
        normalized,
        estimator=str(contract.get("task_relation_estimator", LOCAL_BALANCED_TASK_RELATION)),
        ownership_estimator=ownership_estimator,
    )
    family_values: dict[str, float] = {}
    for family, prefix in (
        ("predictive", ("correction/", "rollout/", "xmod/")),
        ("structural", ("set/",)),
    ):
        numerator = 0.0
        active_weight = 0.0
        for name, measured in normalized.items():
            if name == "action" or not name.startswith(prefix):
                continue
            weight = _objective_term_weight(
                name,
                mode=mode,
                contract=contract,
                ownership_estimator=ownership_estimator,
                ownership_depth_count=ownership_depth_count,
            )
            if valid_counts[name] > 0:
                numerator += float(measured) * weight
                active_weight += weight
        family_values[family] = 0.0 if active_weight <= 0 else numerator / active_weight
    expected = (
        policy
        + float(contract["predictive_family_weight"]) * family_values["predictive"]
        + float(contract["structural_family_weight"]) * family_values["structural"]
    )
    if not math.isclose(
        float(item["objective_total"]),
        expected,
        rel_tol=2e-4,
        abs_tol=2e-5,
    ):
        raise ValueError("full-objective total was not recomputed from its registered families")


def _recompute_representation_report_objective(
    *,
    item: Mapping[str, Any],
    mode: str,
    contract: Mapping[str, Any],
) -> None:
    """Recompute a representation-only objective without accepting action placeholders."""

    normalized = item["normalized_terms"]
    valid_counts = item["valid_counts"]
    if "action" in normalized or "action" in valid_counts:
        raise ValueError("representation objective contains an action term")
    if any(
        item[field] is not None
        for field in (
            "official_action_loss",
            "official_moe_regularizer",
            "official_policy_loss",
        )
    ):
        raise ValueError("representation objective contains an official action loss")
    if float(contract["action_family_weight"]) != 0:
        raise ValueError("representation objective has a nonzero action-family weight")

    ownership_estimator = str(contract.get("ownership_estimator", TOKEN_MICRO_OWNERSHIP))
    ownership_depth_count = _validate_task_relation_term_schema(
        normalized,
        estimator=str(contract.get("task_relation_estimator", LOCAL_BALANCED_TASK_RELATION)),
        ownership_estimator=ownership_estimator,
    )
    family_values: dict[str, float] = {}
    for family, prefix in (
        ("predictive", ("correction/", "rollout/", "xmod/")),
        ("structural", ("set/",)),
    ):
        numerator = 0.0
        active_weight = 0.0
        for name, measured in normalized.items():
            if not name.startswith(prefix):
                continue
            weight = _objective_term_weight(
                name,
                mode=mode,
                contract=contract,
                ownership_estimator=ownership_estimator,
                ownership_depth_count=ownership_depth_count,
            )
            if valid_counts[name] > 0:
                numerator += float(measured) * weight
                active_weight += weight
        family_values[family] = 0.0 if active_weight <= 0 else numerator / active_weight
    expected = (
        float(contract["predictive_family_weight"]) * family_values["predictive"]
        + float(contract["structural_family_weight"]) * family_values["structural"]
    )
    if not math.isclose(
        float(item["objective_total"]),
        expected,
        rel_tol=2e-4,
        abs_tol=2e-5,
    ):
        raise ValueError(
            "representation objective total was not recomputed from predictive and structural "
            "families"
        )


def _validate_full_visual_artifact(
    value: object,
    *,
    run_root: Path,
    expected_step: int,
    expected_rank: int,
) -> None:
    if not isinstance(value, dict) or set(value) != FULL_VISUAL_ARTIFACT_FIELDS:
        raise ValueError("full-objective visual artifact fields differ from schema")
    if (
        value["schema"] != "picf-next.lingbot-native-relation-visual.v5"
        or value["global_step"] != expected_step
        or value["input_weight_global_step"] != expected_step - 1
        or value["weight_boundary"] != "pre_update_forward"
        or value["rank"] != expected_rank
        or value["loss_only_labels_visible_to_model"] is not False
        or value["anchor_surface"]
        not in {
            "task_object_probability.max(row)",
            "ownership_or_support_times_task_relevance.max(row)",
        }
    ):
        raise ValueError("full-objective visual artifact provenance is inconsistent")
    relative_value = value["path"]
    relative = PurePosixPath(relative_value) if isinstance(relative_value, str) else None
    if (
        relative is None
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("full-objective visual artifact path is invalid")
    path = run_root.joinpath(*relative.parts)
    expected_digest = _require_sha256("full-objective visual artifact", value["sha256"])
    byte_count = _positive_report_integer(
        "full-objective visual artifact bytes",
        value["bytes"],
    )
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(run_root.resolve())
        or path.stat().st_size != byte_count
        or _sha256(path) != expected_digest
    ):
        raise ValueError("full-objective visual artifact differs from its PNG")
    if (
        isinstance(value["batch_index"], bool)
        or not isinstance(value["batch_index"], int)
        or value["batch_index"] < 0
        or not isinstance(value["sample_key"], str)
        or not value["sample_key"]
        or not isinstance(value["task"], str)
        or not value["task"].strip()
    ):
        raise ValueError("full-objective visual artifact sample provenance is malformed")
    identity_keys = value["identity_keys"]
    row_to_track = value["row_to_track"]
    sequence_row_to_track = value["sequence_row_to_track"]
    binding_start_phase = value["binding_start_phase"]
    source_binding_valid = value["source_binding_valid"]
    existence = value["row_existence"]
    relevance = value["row_task_relevance"]
    matched_soft_iou = value["row_matched_soft_iou"]
    if (
        not isinstance(identity_keys, list)
        or len(set(identity_keys)) != len(identity_keys)
        or any(not isinstance(item, str) or not item for item in identity_keys)
        or not isinstance(row_to_track, list)
        or not row_to_track
        or not isinstance(sequence_row_to_track, list)
        or not isinstance(binding_start_phase, list)
        or not isinstance(source_binding_valid, list)
        or not isinstance(existence, list)
        or not isinstance(relevance, list)
        or not isinstance(matched_soft_iou, list)
        or len(row_to_track) != len(sequence_row_to_track)
        or len(row_to_track) != len(binding_start_phase)
        or len(row_to_track) != len(source_binding_valid)
        or len(row_to_track) != len(existence)
        or len(row_to_track) != len(relevance)
        or len(row_to_track) != len(matched_soft_iou)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < -1
            or index >= len(identity_keys)
            for index in row_to_track
        )
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < -1
            or index >= len(identity_keys)
            for index in sequence_row_to_track
        )
        or any(
            isinstance(phase, bool) or not isinstance(phase, int) or phase < 0
            for phase in binding_start_phase
        )
        or any(not isinstance(valid, bool) for valid in source_binding_valid)
        or any(
            isinstance(measured, bool)
            or not isinstance(measured, (int, float))
            or not math.isfinite(measured)
            or not 0 <= measured <= 1
            for measured in (*existence, *relevance)
        )
        or any(
            measured is not None
            and (
                isinstance(measured, bool)
                or not isinstance(measured, (int, float))
                or not math.isfinite(measured)
                or not 0 <= measured <= 1
            )
            for measured in matched_soft_iou
        )
    ):
        raise ValueError("full-objective visual row metadata is malformed")
    if (
        value["source_time"] != 0
        or value["source_side"] != "posterior"
        or value["source_phase"] != 1
        or source_binding_valid
        != [
            track >= 0 and phase <= 1
            for track, phase in zip(sequence_row_to_track, binding_start_phase, strict=True)
        ]
        or row_to_track
        != [
            track if valid else -1
            for track, valid in zip(sequence_row_to_track, source_binding_valid, strict=True)
        ]
    ):
        raise ValueError("full-objective visual source-cut assignment is inconsistent")
    views = value["views"]
    if not isinstance(views, list) or not views:
        raise ValueError("full-objective visual artifact has no camera view")
    observed_views: set[str] = set()
    for view in views:
        if not isinstance(view, dict) or set(view) != FULL_VISUAL_VIEW_FIELDS:
            raise ValueError("full-objective visual view fields differ from schema")
        merged_grid = view["merged_grid"]
        source_shape = view["source_shape"]
        if (
            not isinstance(view["name"], str)
            or view["name"] not in {"static", "gripper"}
            or view["name"] in observed_views
            or not isinstance(merged_grid, list)
            or len(merged_grid) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in merged_grid
            )
            or not isinstance(source_shape, list)
            or len(source_shape) != 3
            or source_shape[2] != 3
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in source_shape
            )
            or isinstance(view["token_count"], bool)
            or not isinstance(view["token_count"], int)
            or view["token_count"] != merged_grid[0] * merged_grid[1]
        ):
            raise ValueError("full-objective visual view geometry is inconsistent")
        observed_views.add(view["name"])


def validate_full_objective_report(
    value: object,
    *,
    expected_saved_global_step: int | None = None,
    expected_digests: Mapping[str, str] | None = None,
    require_initial_probe: bool,
    require_mature_wrong_time: bool = False,
    require_source_evidence: bool = False,
    require_checkpoint_copy: bool = True,
    expected_fsdp2_placement: str = FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    expected_cuda_allocator: str = "native",
) -> dict[str, Any]:
    """Validate that a PASS report records a real three-family optimizer transaction."""

    from picf_next.lingbot_native.predictive_probes import (
        BATCH_SHIFT_CONTROL,
        BATCH_SHIFT_SOURCE,
        ROW_SHIFT_SOURCE,
        WRONG_TIME_SOURCE,
        ZERO_CONTROL,
        ZERO_CURRENT_OBSERVATION,
        ZERO_SOURCE,
        predictive_correction_counterfactual_from_mapping,
    )

    if (
        not isinstance(require_initial_probe, bool)
        or not isinstance(require_mature_wrong_time, bool)
        or not isinstance(require_source_evidence, bool)
        or not isinstance(require_checkpoint_copy, bool)
    ):
        raise TypeError("full-objective evidence requirements must be boolean")
    placement = validate_fsdp2_placement(expected_fsdp2_placement)
    if expected_cuda_allocator not in CUDA_ALLOCATOR_MODES:
        raise ValueError("full-objective CUDA allocator expectation is unsupported")
    if expected_saved_global_step is not None and (
        isinstance(expected_saved_global_step, bool)
        or not isinstance(expected_saved_global_step, int)
        or expected_saved_global_step <= 0
    ):
        raise ValueError("expected full-objective saved step must be positive")
    if (
        not isinstance(value, dict)
        or value.get("schema") != FULL_REPORT_SCHEMA
        or (value.get("status") != "PASS")
    ):
        raise ValueError("input full-objective report is not a passed recognized report")
    _validate_full_report_static_surfaces(
        value,
        expected_fsdp2_placement=placement,
        expected_cuda_allocator=expected_cuda_allocator,
    )
    behavior_enabled = value["behavior_conditioning"] is not None

    input_step = value.get("input_global_step")
    saved_step = value.get("saved_global_step")
    if (
        isinstance(input_step, bool)
        or not isinstance(input_step, int)
        or input_step < 0
        or isinstance(saved_step, bool)
        or not isinstance(saved_step, int)
        or saved_step <= input_step
    ):
        raise ValueError("input full-objective report has an invalid optimizer interval")
    if expected_saved_global_step is not None and saved_step != expected_saved_global_step:
        raise ValueError("input full-objective report targets another checkpoint")
    expected_phase = "fresh" if input_step == 0 else "resume"
    if value.get("phase") != expected_phase:
        raise ValueError("input full-objective report phase differs from its optimizer interval")

    digest_fields = (
        "execution_contract_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "plan_sha256",
        "temporal_estimator_sha256",
        "physical_sidecar_manifest_sha256",
        "predictive_cache_manifest_sha256",
        "predictive_teacher_causality_audit_sha256",
        "predictive_target_audit_sha256",
        "predictive_temporal_audit_sha256",
    )
    for field in digest_fields:
        _require_sha256(f"full-objective {field}", value.get(field))
    if expected_digests is not None:
        for field, digest in expected_digests.items():
            _require_sha256(field, digest)
            if value.get(field) != digest:
                raise ValueError(
                    "input full-objective report targets another implementation or model"
                )

    required_true = (
        "full_shard",
        "gradient_checkpointing",
        "action_loss_enabled",
        "structural_set_loss_enabled",
    )
    if any(value.get(field) is not True for field in required_true):
        raise ValueError("input full-objective report did not execute the required training graph")
    if value.get("complete_adr74_objective") is not False:
        raise ValueError("input full-objective report made an unsupported complete-objective claim")
    if not isinstance(value.get("long_training_authorized"), bool):
        raise ValueError("input full-objective long-authorization state is malformed")
    evidence_profile = value.get("evidence_profile")
    if evidence_profile not in {
        "acceptance",
        "loss_visual_trial",
        "behavior_discrimination_trial",
        MATCHED_MEDIUM_HORIZON_PROFILE,
    }:
        raise ValueError("input full-objective evidence profile is unsupported")
    if require_source_evidence and evidence_profile != "acceptance":
        raise ValueError("long-training evidence cannot originate from a loss/visual trial")

    mode = value.get("source_prediction_mode")
    source_flags = {
        "current_grid": value.get("current_source_mask_enabled"),
        "omitted_static": value.get("omitted_static_binding_enabled"),
    }
    if (
        mode not in source_flags
        or source_flags[mode] is not True
        or sum(flag is True for flag in source_flags.values()) != 1
    ):
        raise ValueError("input full-objective report did not enable exactly one source branch")
    _require_sha256(
        "full-objective current-grid cache manifest",
        value.get("current_grid_cache_manifest_sha256"),
    )

    contract = value.get("objective_contract")
    if not isinstance(contract, dict) or set(contract) not in {
        LEGACY_FULL_OBJECTIVE_CONTRACT_FIELDS,
        PRE_ENTITY_FULL_OBJECTIVE_CONTRACT_FIELDS,
        FULL_OBJECTIVE_CONTRACT_FIELDS,
    }:
        raise ValueError("input full-objective report objective contract is incomplete")
    if contract["family_reduction"] != "active_weighted_mean":
        raise ValueError("input full-objective report uses a route-count-dependent reduction")
    task_relation_estimator = contract.get(
        "task_relation_estimator",
        LOCAL_BALANCED_TASK_RELATION,
    )
    if task_relation_estimator not in TASK_RELATION_ESTIMATORS:
        raise ValueError("input full-objective report task relation estimator is invalid")
    ownership_estimator = contract.get("ownership_estimator", TOKEN_MICRO_OWNERSHIP)
    if ownership_estimator not in OWNERSHIP_ESTIMATORS:
        raise ValueError("input full-objective report ownership estimator is invalid")
    positive_weights = (
        "predictive_family_weight",
        "structural_family_weight",
        "predictive_term_weight",
        "existence_weight",
        "task_weight",
        "ownership_weight",
    )
    for field in LEGACY_FULL_OBJECTIVE_CONTRACT_FIELDS - {"family_reduction"}:
        measured = _finite_report_number(f"full-objective {field}", contract[field])
        if field in positive_weights and measured <= 0:
            raise ValueError(f"full-objective {field} must be positive")
    _validate_task_relation_dense_weight(
        estimator=task_relation_estimator,
        dense_task_weight=float(contract["dense_task_weight"]),
        context="full-objective report",
    )
    if float(contract["support_weight"]) != 0:
        raise ValueError(
            "full-objective CALVIN exclusive ownership requires zero independent support weight"
        )
    selected_weight = {
        "current_grid": "current_grid_term_weight",
        "omitted_static": "omitted_static_term_weight",
    }[mode]
    if float(contract[selected_weight]) <= 0:
        raise ValueError("the enabled source-prediction branch has zero objective weight")

    audit_steps = value.get("gradient_audit_steps")
    if (
        not isinstance(audit_steps, list)
        or not audit_steps
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in audit_steps
        )
        or audit_steps != sorted(set(audit_steps))
    ):
        raise ValueError("input full-objective report gradient audit schedule is invalid")

    rank_reports = value.get("rank_reports")
    if (
        not isinstance(rank_reports, list)
        or len(rank_reports) != FULL_WORLD_SIZE
        or {report.get("rank") for report in rank_reports if isinstance(report, dict)}
        != set(range(FULL_WORLD_SIZE))
    ):
        raise ValueError("input full-objective report lacks one exact report per rank")
    expected_steps = list(range(input_step + 1, saved_step + 1))
    checkpoint_hint_value = value["checkpoint_dir"]
    if not isinstance(checkpoint_hint_value, str):
        raise ValueError("full-objective checkpoint path is malformed")
    checkpoint_hint = Path(checkpoint_hint_value)
    run_root = checkpoint_hint.parent.parent
    embedded_audit = value["training_authorization"]
    visual_cadence = (
        embedded_audit.get("visual_audit_every") if isinstance(embedded_audit, dict) else None
    )
    observed_step_samples = {step: set() for step in expected_steps}
    observed_step_predictive_ranks = {step: set() for step in expected_steps}
    observed_step_predictive_gradients: dict[int, tuple[float, int]] = {}
    observed_mature_wrong_time = False
    for rank_report in rank_reports:
        if not isinstance(rank_report, dict) or set(rank_report) != FULL_RANK_REPORT_FIELDS:
            raise ValueError("input full-objective rank report is malformed")
        rank = rank_report["rank"]
        _validate_report_boundary(f"rank {rank} saved", rank_report.get("saved_boundary_sha256"))
        loaded_boundary = rank_report.get("loaded_boundary_sha256")
        if input_step == 0:
            if loaded_boundary is not None:
                raise ValueError("fresh full-objective report unexpectedly loaded a checkpoint")
        else:
            _validate_report_boundary(f"rank {rank} loaded", loaded_boundary)
        steps = rank_report.get("steps")
        if (
            not isinstance(steps, list)
            or any(not isinstance(item, dict) for item in steps)
            or [item.get("global_step") for item in steps] != expected_steps
        ):
            raise ValueError("input full-objective rank steps are incomplete or non-contiguous")
        rank_has_source_evidence = False
        reported_lane_bindings: dict[int, dict[str, int]] = {}
        for item in steps:
            if set(item) != FULL_STEP_REPORT_FIELDS:
                raise ValueError("input full-objective step fields differ from schema")
            routing = (
                item["sample_keys"],
                item["lane_ids"],
                item["frame_indices"],
                item["state_ages"],
            )
            if (
                any(not isinstance(sequence, list) or not sequence for sequence in routing)
                or len({len(sequence) for sequence in routing}) != 1
                or any(not isinstance(entry, str) or not entry for entry in item["sample_keys"])
                or any(
                    isinstance(entry, bool) or not isinstance(entry, int) or entry < 0
                    for sequence in routing[1:]
                    for entry in sequence
                )
                or len(set(item["sample_keys"])) != len(item["sample_keys"])
            ):
                raise ValueError("full-objective step routing provenance is malformed")
            distributed_overlap = observed_step_samples[item["global_step"]].intersection(
                item["sample_keys"]
            )
            if distributed_overlap:
                raise ValueError("full-objective ranks consumed overlapping samples")
            observed_step_samples[item["global_step"]].update(item["sample_keys"])
            _require_sha256("full-objective temporal plan", item["temporal_plan_sha256"])
            local_bptt_steps = item["local_bptt_steps"]
            overshoot_horizon = item["overshoot_horizon"]
            if (
                isinstance(local_bptt_steps, bool)
                or not isinstance(local_bptt_steps, int)
                or not 1 <= local_bptt_steps <= 4
                or isinstance(overshoot_horizon, bool)
                or not isinstance(overshoot_horizon, int)
                or not 0 <= overshoot_horizon <= 64
            ):
                raise ValueError("full-objective temporal sample is malformed")
            if _finite_report_number(
                "full-objective visual audit time", item["visual_audit_seconds"]
            ) < 0 or not isinstance(item["visual_artifacts"], list):
                raise ValueError("full-objective visual audit report is malformed")
            artifacts = item["visual_artifacts"]
            if visual_cadence is not None and (
                isinstance(visual_cadence, bool)
                or not isinstance(visual_cadence, int)
                or visual_cadence <= 0
                or bool(artifacts) is not (item["global_step"] % visual_cadence == 0)
            ):
                raise ValueError("full-objective visual cadence differs from authorization")
            for artifact in artifacts:
                _validate_full_visual_artifact(
                    artifact,
                    run_root=run_root,
                    expected_step=item["global_step"],
                    expected_rank=rank,
                )
                if (
                    artifact["batch_index"] >= len(item["sample_keys"])
                    or artifact["sample_key"] not in item["sample_keys"]
                ):
                    raise ValueError("full-objective visual artifact is outside its batch")
            for field in (
                "objective_total",
                "official_action_loss",
                "official_moe_regularizer",
                "official_policy_loss",
            ):
                if _finite_report_number(f"rank {rank} {field}", item.get(field)) < 0:
                    raise ValueError(f"rank {rank} {field} must be non-negative")
            _finite_report_number(f"rank {rank} step time", item.get("step_time_s"), positive=True)
            normalized = item.get("normalized_terms")
            valid_counts = item.get("valid_counts")
            required_predictive_prefix = "rollout/" if behavior_enabled else "correction/"
            if (
                not isinstance(normalized, dict)
                or not isinstance(valid_counts, dict)
                or set(normalized) != set(valid_counts)
                or "action" not in normalized
                or not any(name.startswith(required_predictive_prefix) for name in normalized)
                or (behavior_enabled and any(name.startswith("correction/") for name in normalized))
                or not any(name.startswith("set/") for name in normalized)
            ):
                raise ValueError("full-objective step omitted an action, correction, or set term")
            if behavior_enabled:
                rollout_terms = {name for name in normalized if name.startswith("rollout/")}
                if rollout_terms != {"rollout/vision/binding"}:
                    raise ValueError("joint action step changed its sole causal rollout target")
                if valid_counts["rollout/vision/binding"] <= 0:
                    raise ValueError("joint action step has no valid causal rollout target")
            for name, measured in normalized.items():
                _finite_report_number(f"rank {rank} normalized term {name}", measured)
                count = valid_counts[name]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("full-objective valid counts must be non-negative integers")
            if "set/ownership" not in valid_counts or valid_counts["set/ownership"] <= 0:
                raise ValueError(
                    "full-objective CALVIN step omitted active exclusive ownership supervision"
                )
            if valid_counts.get("set/support", 0) != 0:
                raise ValueError(
                    "full-objective CALVIN step activated forbidden independent support supervision"
                )
            predictive_active = any(
                count > 0
                for name, count in valid_counts.items()
                if name.startswith(("correction/", "rollout/", "xmod/"))
            )
            structural_active = any(
                count > 0 for name, count in valid_counts.items() if name.startswith("set/")
            )
            if valid_counts["action"] <= 0 or not structural_active:
                raise ValueError(
                    "full-objective step did not activate action and structural families"
                )
            validate_task_row_diagnostics(
                item.get("task_row_diagnostics"),
                expected_batch_size=len(item["sample_keys"]),
            )
            step_bindings = _validate_step_row_bindings(
                prior_value=item.get("prior_row_bindings"),
                current_value=item.get("row_bindings"),
                reported_birth_count=item.get("row_binding_birth_count"),
                task_row_diagnostics=item.get("task_row_diagnostics"),
                expected_batch_size=len(item["sample_keys"]),
            )
            if _validate_step_estimator_transaction(item):
                _advance_report_row_binding_continuity(
                    reported_lane_bindings,
                    lane_ids=item["lane_ids"],
                    frame_indices=item["frame_indices"],
                    state_ages=item["state_ages"],
                    step_bindings=step_bindings,
                )
            if predictive_active:
                observed_step_predictive_ranks[item["global_step"]].add(rank)
            _recompute_report_objective(item=item, mode=mode, contract=contract)

            source_selected = item.get("source_masked_branch")
            if not isinstance(source_selected, bool):
                raise ValueError("full-objective step source-branch decision is malformed")
            reported_source_mode = item.get("source_prediction_mode")
            source_mask_digest = item.get("source_mask_digest")
            source_mask_query_count = item.get("source_mask_query_count")
            omitted_static_digest = item.get("omitted_static_digest")
            if (
                isinstance(source_mask_query_count, bool)
                or not isinstance(source_mask_query_count, int)
                or source_mask_query_count < 0
            ):
                raise ValueError("full-objective source-mask query count is malformed")
            if source_selected:
                if reported_source_mode != mode:
                    raise ValueError("full-objective source branch executed another mode")
                if mode == "current_grid":
                    _require_sha256("full-objective source-mask digest", source_mask_digest)
                    if source_mask_query_count <= 0 or omitted_static_digest is not None:
                        raise ValueError("full-objective current-grid evidence is inconsistent")
                else:
                    _require_sha256("full-objective omitted-static digest", omitted_static_digest)
                    if source_mask_digest is not None or source_mask_query_count != 0:
                        raise ValueError("full-objective omitted-static evidence is inconsistent")
                source_active = any(
                    count > 0 for name, count in valid_counts.items() if name.startswith("xmod/")
                )
                # Source omission is sampled without consulting loss-only row bindings.
                # A reset sample can therefore execute the branch while having no
                # causally established identity to supervise.  Keep that term
                # zero-valid, as required by missing-modality semantics, but do not
                # count it as source evidence for a long-training authorization.
                rank_has_source_evidence = rank_has_source_evidence or source_active
            elif (
                reported_source_mode is not None
                or source_mask_digest is not None
                or source_mask_query_count != 0
                or omitted_static_digest is not None
            ):
                raise ValueError("full-objective inactive source evidence is inconsistent")

            metrics = item.get("gradient_metrics")
            if (
                not isinstance(metrics, dict)
                or set(metrics) != FULL_GRADIENT_REPORT_FIELDS
                or metrics.get("all_finite") is not True
            ):
                raise ValueError("full-objective step gradients are absent or non-finite")
            for prefix in (
                "native_graph",
                "relation_projection",
                "match_projection",
                "action_output",
            ):
                _finite_report_number(
                    f"rank {rank} {prefix} gradient norm",
                    metrics.get(f"{prefix}_norm"),
                    positive=True,
                )
                elements = metrics.get(f"{prefix}_elements")
                if isinstance(elements, bool) or not isinstance(elements, int) or elements <= 0:
                    raise ValueError(f"rank {rank} {prefix} gradient coverage is empty")
            predictive_gradient = _finite_report_number(
                f"rank {rank} predictive_readout gradient norm",
                metrics.get("predictive_readout_norm"),
            )
            predictive_elements = metrics.get("predictive_readout_elements")
            if (
                predictive_gradient < 0
                or isinstance(predictive_elements, bool)
                or not isinstance(predictive_elements, int)
                or predictive_elements < 0
            ):
                raise ValueError("full-objective predictive gradient coverage is malformed")
            predictive_summary = (predictive_gradient, predictive_elements)
            prior_predictive_summary = observed_step_predictive_gradients.setdefault(
                item["global_step"], predictive_summary
            )
            if prior_predictive_summary != predictive_summary:
                raise ValueError("full-objective distributed predictive gradients differ by rank")
            behavior_posterior_norm = _finite_report_number(
                f"rank {rank} behavior-posterior gradient norm",
                metrics.get("behavior_posterior_gradient_norm"),
            )
            behavior_posterior_elements = metrics.get("behavior_posterior_gradient_elements")
            if (
                isinstance(behavior_posterior_elements, bool)
                or not isinstance(behavior_posterior_elements, int)
                or behavior_posterior_elements < 0
            ):
                raise ValueError("action behavior-posterior gradient size is invalid")
            if behavior_enabled:
                if behavior_posterior_norm <= 0 or behavior_posterior_elements <= 0:
                    raise ValueError(
                        "joint action objective did not reach the deploy posterior activation"
                    )
            elif behavior_posterior_norm != 0 or behavior_posterior_elements != 0:
                raise ValueError("ordinary action training fabricated behavior-posterior credit")
            _finite_report_number(
                f"rank {rank} preclip global norm",
                metrics.get("preclip_global_norm"),
                positive=True,
            )
            diagnostic = item.get("family_gradient_diagnostics")
            relation_diagnostic = item.get("relation_surface_gradient_diagnostics")
            host_diagnostic = item.get("predictive_host_gradient_diagnostics")
            counterfactual_diagnostic = item.get("predictive_counterfactual_diagnostics")
            counterfactual_weight_boundary = item.get("predictive_counterfactual_weight_boundary")
            if item["global_step"] in audit_steps:
                _validate_family_gradient_report(diagnostic)
                _validate_relation_surface_gradient_report(
                    relation_diagnostic,
                    estimator=task_relation_estimator,
                )
                _validate_predictive_host_gradient_report(
                    host_diagnostic,
                    expected_probe=(
                        "lingbot.language_model.input_layernorm.via_primary_posterior_vjp"
                        if behavior_enabled
                        else "lingbot.language_model.input_layernorm"
                    ),
                )
                if behavior_enabled:
                    if (
                        counterfactual_diagnostic is not None
                        or counterfactual_weight_boundary is not None
                    ):
                        raise ValueError(
                            "joint behavior/action training fabricated a correction counterfactual"
                        )
                else:
                    if any(frame <= 0 for frame in item["frame_indices"]) or any(
                        age <= 0 for age in item["state_ages"]
                    ):
                        raise ValueError(
                            "full-objective prior-correction audit ran before recurrent bootstrap"
                        )
                    parsed_counterfactual = predictive_correction_counterfactual_from_mapping(
                        counterfactual_diagnostic
                    )
                    if counterfactual_weight_boundary != "pre_update_post_backward":
                        raise ValueError(
                            "full-objective counterfactual used another weight boundary"
                        )
                    intervention_by_name = {
                        value.name: value for value in parsed_counterfactual.interventions
                    }
                    required_interventions = {
                        ZERO_SOURCE,
                        ROW_SHIFT_SOURCE,
                        BATCH_SHIFT_SOURCE,
                        ZERO_CONTROL,
                        BATCH_SHIFT_CONTROL,
                        ZERO_CURRENT_OBSERVATION,
                    }
                    if not intervention_by_name.keys() >= required_interventions:
                        raise ValueError(
                            "full-objective predictive counterfactual coverage is incomplete"
                        )
                    if (
                        intervention_by_name[ZERO_CURRENT_OBSERVATION].normalized_prediction_l1
                        > 1e-6
                    ):
                        raise ValueError(
                            "full-objective prior correction read the current observation"
                        )
                    mature = all(frame >= 2 for frame in item["frame_indices"])
                    has_wrong_time = WRONG_TIME_SOURCE in intervention_by_name
                    if mature:
                        if not has_wrong_time:
                            raise ValueError(
                                "full-objective mature correction audit omitted wrong-time state"
                            )
                        observed_mature_wrong_time = True
                    elif has_wrong_time:
                        raise ValueError(
                            "full-objective first continuation fabricated wrong-time state"
                        )
            elif (
                diagnostic is not None
                or relation_diagnostic is not None
                or host_diagnostic is not None
                or counterfactual_diagnostic is not None
                or counterfactual_weight_boundary is not None
            ):
                raise ValueError("full-objective report contains an unscheduled diagnostic")
            allocated = _positive_report_integer(
                f"rank {rank} peak CUDA allocated bytes",
                item["peak_cuda_allocated_bytes"],
            )
            reserved = _positive_report_integer(
                f"rank {rank} peak CUDA reserved bytes",
                item["peak_cuda_reserved_bytes"],
            )
            if allocated > reserved or reserved > value["maximum_peak_reserved_bytes"]:
                raise ValueError("full-objective CUDA memory exceeded its registered budget")
        if require_source_evidence and not rank_has_source_evidence:
            raise ValueError(
                "full-objective report lacks executed source-branch evidence on every rank"
            )

    bootstrap_only = input_step == 0 and saved_step == 1 and not behavior_enabled
    if not bootstrap_only and not any(observed_step_predictive_ranks.values()):
        raise ValueError("full-objective report contains no active predictive supervision")
    for step in expected_steps:
        predictive_ranks = observed_step_predictive_ranks[step]
        predictive_norm, predictive_elements = observed_step_predictive_gradients[step]
        if predictive_ranks:
            if predictive_norm <= 0 or predictive_elements <= 0:
                raise ValueError(
                    "full-objective active predictive family produced no readout gradient"
                )
        elif predictive_norm != 0:
            raise ValueError(
                "full-objective inactive predictive family produced a readout gradient"
            )
        if step in audit_steps and predictive_ranks != set(range(FULL_WORLD_SIZE)):
            raise ValueError(
                "full-objective gradient audit lacked predictive targets on every rank"
            )

    if require_initial_probe:
        if (input_step, saved_step) != (0, 1) or 1 in audit_steps:
            raise ValueError(
                "pilot authorization requires one unaudited fresh state-bootstrap step"
            )
        if value.get("training_authorization") is not None:
            raise ValueError("the initial full-objective probe cannot consume prior authorization")
        for rank_report in rank_reports:
            item = rank_report["steps"][0]
            if item["frame_indices"] != [0] or item["state_ages"] != [0]:
                raise ValueError("fresh state-bootstrap report did not consume exact reset frames")
            if item["local_bptt_steps"] == 1 and any(
                count > 0
                for name, count in item["valid_counts"].items()
                if name.startswith("correction/")
            ):
                raise ValueError("fresh state bootstrap fabricated prior-correction support")
    if require_mature_wrong_time and not observed_mature_wrong_time:
        raise ValueError("full-objective evidence lacks a mature wrong-time counterfactual")

    checkpoint_dir_value = value.get("checkpoint_dir")
    checkpoint_dir = Path(checkpoint_dir_value) if isinstance(checkpoint_dir_value, str) else None
    if checkpoint_dir is None or not checkpoint_dir.is_absolute() or checkpoint_dir.is_symlink():
        raise ValueError("input full-objective report has no absolute checkpoint directory")
    if checkpoint_dir.name != f"global_step_{saved_step}":
        raise ValueError("input full-objective checkpoint path differs from its saved step")
    if require_checkpoint_copy:
        if not checkpoint_dir.is_dir():
            raise ValueError("input full-objective report has no real checkpoint directory")
        checkpoint_report = checkpoint_dir / "native_full_report.json"
        if checkpoint_report.is_symlink() or not checkpoint_report.is_file():
            raise ValueError("input full-objective checkpoint lacks its immutable report copy")
        try:
            checkpoint_value = json.loads(checkpoint_report.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("input full-objective checkpoint report is not valid JSON") from error
        if checkpoint_value != value:
            raise ValueError(
                "input full-objective checkpoint report differs from the reviewed report"
            )

    embedded = value.get("training_authorization")
    if input_step == 0:
        if embedded is not None or value["long_training_authorized"] is not False:
            raise ValueError("fresh full-objective probe cannot contain prior authorization")
    else:
        if not isinstance(embedded, dict):
            raise ValueError("resumed full-objective report lacks its training authorization")
        manifest_path_value = embedded.get("manifest_path")
        manifest_digest = _require_sha256(
            "embedded training authorization manifest",
            embedded.get("manifest_sha256"),
        )
        manifest_path = Path(manifest_path_value) if isinstance(manifest_path_value, str) else None
        manifest_value = {
            name: measured
            for name, measured in embedded.items()
            if name not in {"manifest_path", "manifest_sha256"}
        }
        if (
            manifest_path is None
            or not manifest_path.is_absolute()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
            or _sha256(manifest_path) != manifest_digest
        ):
            raise ValueError("embedded training authorization manifest differs")
        try:
            published_manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("embedded training authorization is not valid JSON") from error
        if not isinstance(published_manifest, dict):
            raise ValueError("embedded training authorization manifest is malformed")
        if published_manifest != manifest_value:
            raise ValueError("embedded training authorization differs from its manifest")
        maximum_global_step = published_manifest.get("maximum_global_step")
        authorized_visual_cadence = published_manifest.get("visual_audit_every")
        if (
            isinstance(maximum_global_step, bool)
            or not isinstance(maximum_global_step, int)
            or maximum_global_step <= 0
            or isinstance(authorized_visual_cadence, bool)
            or not isinstance(authorized_visual_cadence, int)
            or authorized_visual_cadence <= 0
        ):
            raise ValueError("embedded training authorization interval is malformed")
        validated_manifest = validate_training_authorization(
            manifest_path,
            expected_sha256=manifest_digest,
            input_global_step=input_step,
            requested_global_step=saved_step,
            total_planned_steps=maximum_global_step,
            visual_audit_every=authorized_visual_cadence,
            execution_contract_sha256=value["execution_contract_sha256"],
            implementation_sha256=value["implementation_sha256"],
            model_family_sha256=value["model_family_sha256"],
            expected_fsdp2_placement=placement,
            expected_cuda_allocator=expected_cuda_allocator,
        )
        expected_long = validated_manifest["stage"] == "long"
        if value["long_training_authorized"] is not expected_long:
            raise ValueError("full-objective long-authorization state differs from its manifest")
    return value


def validate_representation_objective_report(
    value: object,
    *,
    expected_saved_global_step: int | None = None,
    expected_digests: Mapping[str, str] | None = None,
    expected_behavior_conditioning_sha256: str | None = None,
    require_initial_probe: bool,
    require_checkpoint_copy: bool = True,
    expected_checkpoint_publication: str = "always",
    expected_fsdp2_placement: str = FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    expected_cuda_allocator: str = "native",
) -> dict[str, Any]:
    """Validate one action-isolated shared-host representation transaction."""

    from picf_next.lingbot_native.predictive_probes import (
        BATCH_SHIFT_CONTROL,
        BATCH_SHIFT_SOURCE,
        ROW_SHIFT_SOURCE,
        WRONG_TIME_SOURCE,
        ZERO_CONTROL,
        ZERO_CURRENT_OBSERVATION,
        ZERO_SOURCE,
        predictive_correction_counterfactual_from_mapping,
    )

    if not isinstance(require_initial_probe, bool) or not isinstance(require_checkpoint_copy, bool):
        raise TypeError("representation evidence requirements must be boolean")
    if expected_checkpoint_publication not in CHECKPOINT_PUBLICATION_MODES:
        raise ValueError("representation checkpoint-publication expectation is unsupported")
    placement = validate_fsdp2_placement(expected_fsdp2_placement)
    if expected_cuda_allocator not in CUDA_ALLOCATOR_MODES:
        raise ValueError("representation CUDA allocator expectation is unsupported")
    if expected_saved_global_step is not None and (
        isinstance(expected_saved_global_step, bool)
        or not isinstance(expected_saved_global_step, int)
        or expected_saved_global_step <= 0
    ):
        raise ValueError("expected representation saved step must be positive")
    if (
        not isinstance(value, dict)
        or value.get("schema") != REPRESENTATION_REPORT_SCHEMA
        or value.get("status") != "PASS"
        or set(value) != REPRESENTATION_REPORT_FIELDS
    ):
        raise ValueError("input representation report is not a passed recognized report")
    if value["training_stage"] != NATIVE_REPRESENTATION_STAGE:
        raise ValueError("input representation report records another training stage")
    checkpoint_publication = value["checkpoint_publication"]
    if (
        checkpoint_publication not in CHECKPOINT_PUBLICATION_MODES
        or checkpoint_publication != expected_checkpoint_publication
    ):
        raise ValueError("input representation checkpoint-publication mode differs")
    _require_sha256(
        "representation split artifact",
        value["representation_split_sha256"],
    )
    _require_sha256(
        "representation split file",
        value["representation_split_file_sha256"],
    )
    _require_sha256(
        "representation parameter scope",
        value["representation_parameter_scope_sha256"],
    )
    _require_sha256(
        "representation frozen action state",
        value["representation_frozen_action_state_sha256"],
    )

    static_value = {
        name: measured for name, measured in value.items() if name in FULL_REPORT_FIELDS
    }
    static_value["schema"] = FULL_REPORT_SCHEMA
    _validate_full_report_static_surfaces(
        static_value,
        expected_fsdp2_placement=placement,
        expected_cuda_allocator=expected_cuda_allocator,
    )
    behavior_conditioning = value["behavior_conditioning"]
    behavior_enabled = behavior_conditioning is not None
    if expected_behavior_conditioning_sha256 is not None:
        expected_behavior = _require_sha256(
            "expected behavior-conditioning contract",
            expected_behavior_conditioning_sha256,
        )
        if behavior_conditioning is None or _canonical_digest(behavior_conditioning) != (
            expected_behavior
        ):
            raise ValueError("representation behavior-conditioning contract differs")

    parameter_scope = value["representation_parameter_scope"]
    if (
        not isinstance(parameter_scope, dict)
        or set(parameter_scope) != REPRESENTATION_PARAMETER_SCOPE_FIELDS
        or parameter_scope["schema"] != "picf-next.lingbot-representation-scope.v1"
        or _canonical_digest(parameter_scope) != value["representation_parameter_scope_sha256"]
    ):
        raise ValueError("representation parameter scope is malformed")
    for field in (
        "production_trainable_sha256",
        "production_frozen_sha256",
        "representation_trainable_sha256",
        "action_frozen_sha256",
    ):
        _require_sha256(f"representation parameter scope {field}", parameter_scope[field])
    scope_numel: dict[str, int] = {}
    for field in (
        "production_trainable_numel",
        "production_frozen_numel",
        "representation_trainable_numel",
        "action_frozen_numel",
    ):
        measured = parameter_scope[field]
        if isinstance(measured, bool) or not isinstance(measured, int) or measured < 0:
            raise ValueError(f"representation parameter scope {field} is invalid")
        scope_numel[field] = measured
    if (
        scope_numel["production_trainable_numel"] <= 0
        or scope_numel["representation_trainable_numel"] <= 0
        or scope_numel["action_frozen_numel"] <= 0
        or scope_numel["production_trainable_numel"]
        != scope_numel["representation_trainable_numel"] + scope_numel["action_frozen_numel"]
        or value["parameter_manifest"]["trainable_numel"]
        != scope_numel["representation_trainable_numel"]
    ):
        raise ValueError("representation parameter partition does not cover production training")

    input_step = value["input_global_step"]
    saved_step = value["saved_global_step"]
    if (
        isinstance(input_step, bool)
        or not isinstance(input_step, int)
        or input_step < 0
        or isinstance(saved_step, bool)
        or not isinstance(saved_step, int)
        or saved_step <= input_step
    ):
        raise ValueError("input representation report has an invalid optimizer interval")
    if expected_saved_global_step is not None and saved_step != expected_saved_global_step:
        raise ValueError("input representation report targets another checkpoint")
    expected_phase = "fresh" if input_step == 0 else "resume"
    if value["phase"] != expected_phase:
        raise ValueError("input representation report phase differs from its optimizer interval")
    if behavior_enabled and (
        checkpoint_publication != "always" or (input_step, saved_step) not in {(0, 1), (1, 2)}
    ):
        raise ValueError("behavior representation report is outside its bounded G1 interval")

    digest_fields = (
        "execution_contract_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "plan_sha256",
        "temporal_estimator_sha256",
        "physical_sidecar_manifest_sha256",
        "predictive_cache_manifest_sha256",
        "predictive_teacher_causality_audit_sha256",
        "predictive_target_audit_sha256",
        "predictive_temporal_audit_sha256",
    )
    for field in digest_fields:
        _require_sha256(f"representation {field}", value[field])
    if expected_digests is not None:
        for field, digest in expected_digests.items():
            _require_sha256(field, digest)
            if value.get(field) != digest:
                raise ValueError(
                    f"input representation report {field} targets another implementation or model"
                )

    required_true = ("gradient_checkpointing", "structural_set_loss_enabled")
    if (
        any(value[field] is not True for field in required_true)
        or value["action_loss_enabled"] is not False
        or value["full_shard"] is not (checkpoint_publication == "always")
    ):
        raise ValueError("input representation report executed another training graph")
    if (
        value["complete_adr74_objective"] is not False
        or value["training_authorization"] is not None
        or value["long_training_authorized"] is not False
    ):
        raise ValueError("representation report made an unsupported action-training claim")
    if value["evidence_profile"] not in {
        "acceptance",
        "loss_visual_trial",
        MATCHED_MEDIUM_HORIZON_PROFILE,
    }:
        raise ValueError("input representation evidence profile is unsupported")

    visual_cadence = value["visual_audit_every"]
    if (
        isinstance(visual_cadence, bool)
        or not isinstance(visual_cadence, int)
        or visual_cadence < 0
        or (saved_step - input_step > 1 and visual_cadence <= 0)
    ):
        raise ValueError("input representation visual cadence is invalid")

    mode = value["source_prediction_mode"]
    source_flags = {
        "current_grid": value["current_source_mask_enabled"],
        "omitted_static": value["omitted_static_binding_enabled"],
    }
    if (
        mode not in source_flags
        or source_flags[mode] is not True
        or sum(flag is True for flag in source_flags.values()) != 1
    ):
        raise ValueError("input representation report did not enable exactly one source branch")
    _require_sha256(
        "representation current-grid cache manifest",
        value["current_grid_cache_manifest_sha256"],
    )

    contract = value["objective_contract"]
    if (
        not isinstance(contract, dict)
        or set(contract)
        not in {
            LEGACY_REPRESENTATION_OBJECTIVE_CONTRACT_FIELDS,
            PRE_ENTITY_REPRESENTATION_OBJECTIVE_CONTRACT_FIELDS,
            REPRESENTATION_OBJECTIVE_CONTRACT_FIELDS,
        }
        or contract["family_reduction"] != "active_weighted_mean"
    ):
        raise ValueError("input representation objective contract is incomplete")
    task_relation_estimator = contract.get(
        "task_relation_estimator",
        LOCAL_BALANCED_TASK_RELATION,
    )
    if task_relation_estimator not in TASK_RELATION_ESTIMATORS:
        raise ValueError("input representation task relation estimator is invalid")
    ownership_estimator = contract.get("ownership_estimator", TOKEN_MICRO_OWNERSHIP)
    if ownership_estimator not in OWNERSHIP_ESTIMATORS:
        raise ValueError("input representation ownership estimator is invalid")
    positive_weights = (
        "predictive_family_weight",
        "structural_family_weight",
        "predictive_term_weight",
        "existence_weight",
        "task_weight",
        "ownership_weight",
    )
    for field in LEGACY_REPRESENTATION_OBJECTIVE_CONTRACT_FIELDS - {"family_reduction"}:
        measured = _finite_report_number(f"representation {field}", contract[field])
        if field in positive_weights and measured <= 0:
            raise ValueError(f"representation {field} must be positive")
    _validate_task_relation_dense_weight(
        estimator=task_relation_estimator,
        dense_task_weight=float(contract["dense_task_weight"]),
        context="representation report",
    )
    if float(contract["support_weight"]) != 0 or float(contract["action_family_weight"]) != 0:
        raise ValueError("representation objective activated a forbidden family weight")
    selected_weight = {
        "current_grid": "current_grid_term_weight",
        "omitted_static": "omitted_static_term_weight",
    }[mode]
    if float(contract[selected_weight]) <= 0:
        raise ValueError("the enabled representation source branch has zero objective weight")

    audit_steps = value["gradient_audit_steps"]
    if (
        not isinstance(audit_steps, list)
        or not audit_steps
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in audit_steps
        )
        or audit_steps != sorted(set(audit_steps))
    ):
        raise ValueError("input representation gradient audit schedule is invalid")
    if behavior_enabled and audit_steps != [1, 2]:
        raise ValueError("behavior representation report changed its two-step gradient audit")
    if value["evidence_profile"] == MATCHED_MEDIUM_HORIZON_PROFILE and (
        (value["phase"], input_step, saved_step - input_step)
        not in {
            (phase, load_step, invocation_steps)
            for phase, load_step, invocation_steps, _ in REGISTERED_MATCHED_MEDIUM_HORIZON_SEGMENTS
        }
        or checkpoint_publication != "always"
        or visual_cadence != MATCHED_MEDIUM_HORIZON_VISUAL_CADENCE
        or audit_steps != list(MATCHED_MEDIUM_HORIZON_AUDIT_STEPS)
        or behavior_enabled
        or task_relation_estimator != HOST_NATIVE_FACTORIZED_TASK_RELATION
        or ownership_estimator
        not in {TOKEN_MICRO_OWNERSHIP, TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP}
    ):
        raise ValueError("matched medium-horizon representation report changed its contract")

    rank_reports = value["rank_reports"]
    if (
        not isinstance(rank_reports, list)
        or len(rank_reports) != FULL_WORLD_SIZE
        or {report.get("rank") for report in rank_reports if isinstance(report, dict)}
        != set(range(FULL_WORLD_SIZE))
    ):
        raise ValueError("input representation report lacks one exact report per rank")
    expected_steps = list(range(input_step + 1, saved_step + 1))
    checkpoint_hint_value = value["checkpoint_dir"]
    if not isinstance(checkpoint_hint_value, str):
        raise ValueError("representation checkpoint path is malformed")
    checkpoint_hint = Path(checkpoint_hint_value)
    run_root = checkpoint_hint.parent.parent
    observed_step_routes: dict[
        int,
        list[tuple[int, frozenset[str], dict[str, object]]],
    ] = {step: [] for step in expected_steps}
    observed_step_predictive_ranks = {step: set() for step in expected_steps}
    observed_step_predictive_gradients: dict[int, tuple[float, int]] = {}

    for rank_report in rank_reports:
        if not isinstance(rank_report, dict) or set(rank_report) != FULL_RANK_REPORT_FIELDS:
            raise ValueError("input representation rank report is malformed")
        rank = rank_report["rank"]
        saved_boundary = rank_report["saved_boundary_sha256"]
        if checkpoint_publication == "always":
            _validate_report_boundary(
                f"representation rank {rank} saved",
                saved_boundary,
            )
        elif saved_boundary is not None:
            raise ValueError("non-published representation report fabricated a saved boundary")
        loaded_boundary = rank_report["loaded_boundary_sha256"]
        if input_step == 0:
            if loaded_boundary is not None:
                raise ValueError("fresh representation report unexpectedly loaded a checkpoint")
        else:
            _validate_report_boundary(
                f"representation rank {rank} loaded",
                loaded_boundary,
            )
        steps = rank_report["steps"]
        if (
            not isinstance(steps, list)
            or any(not isinstance(item, dict) for item in steps)
            or [item.get("global_step") for item in steps] != expected_steps
        ):
            raise ValueError("input representation rank steps are incomplete or non-contiguous")
        reported_lane_bindings: dict[int, dict[str, int]] = {}
        for item in steps:
            if set(item) != REPRESENTATION_STEP_REPORT_FIELDS:
                raise ValueError("input representation step fields differ from schema")
            routing = (
                item["sample_keys"],
                item["lane_ids"],
                item["frame_indices"],
                item["state_ages"],
            )
            if (
                any(not isinstance(sequence, list) or not sequence for sequence in routing)
                or len({len(sequence) for sequence in routing}) != 1
                or any(not isinstance(entry, str) or not entry for entry in item["sample_keys"])
                or any(
                    isinstance(entry, bool) or not isinstance(entry, int) or entry < 0
                    for sequence in routing[1:]
                    for entry in sequence
                )
                or len(set(item["sample_keys"])) != len(item["sample_keys"])
            ):
                raise ValueError("representation step routing provenance is malformed")
            observed_step_routes[item["global_step"]].append(
                (
                    rank,
                    frozenset(item["sample_keys"]),
                    {
                        "fixed_observation_pair_sha256": item["fixed_observation_pair_sha256"],
                        "fixed_observation_fingerprint": item["fixed_observation_fingerprint"],
                    },
                )
            )
            _require_sha256("representation temporal plan", item["temporal_plan_sha256"])
            local_bptt_steps = item["local_bptt_steps"]
            overshoot_horizon = item["overshoot_horizon"]
            if (
                isinstance(local_bptt_steps, bool)
                or not isinstance(local_bptt_steps, int)
                or not 1 <= local_bptt_steps <= 4
                or isinstance(overshoot_horizon, bool)
                or not isinstance(overshoot_horizon, int)
                or not 0 <= overshoot_horizon <= 64
            ):
                raise ValueError("representation temporal sample is malformed")
            if _finite_report_number(
                "representation visual audit time",
                item["visual_audit_seconds"],
            ) < 0 or not isinstance(item["visual_artifacts"], list):
                raise ValueError("representation visual audit report is malformed")
            artifacts = item["visual_artifacts"]
            expected_visual = visual_cadence > 0 and item["global_step"] % visual_cadence == 0
            if bool(artifacts) is not expected_visual:
                raise ValueError("representation visual cadence differs from its report")
            for artifact in artifacts:
                _validate_full_visual_artifact(
                    artifact,
                    run_root=run_root,
                    expected_step=item["global_step"],
                    expected_rank=rank,
                )
                if (
                    artifact["batch_index"] >= len(item["sample_keys"])
                    or artifact["sample_key"] not in item["sample_keys"]
                ):
                    raise ValueError("representation visual artifact is outside its batch")

            if (
                _finite_report_number(
                    f"representation rank {rank} objective total",
                    item["objective_total"],
                )
                < 0
            ):
                raise ValueError("representation objective total must be non-negative")
            if any(
                item[field] is not None
                for field in (
                    "official_action_loss",
                    "official_moe_regularizer",
                    "official_policy_loss",
                )
            ):
                raise ValueError("representation step reported an official action loss")
            _finite_report_number(
                f"representation rank {rank} step time",
                item["step_time_s"],
                positive=True,
            )
            normalized = item["normalized_terms"]
            valid_counts = item["valid_counts"]
            required_predictive_prefix = "rollout/" if behavior_enabled else "correction/"
            if (
                not isinstance(normalized, dict)
                or not isinstance(valid_counts, dict)
                or set(normalized) != set(valid_counts)
                or "action" in normalized
                or not any(name.startswith(required_predictive_prefix) for name in normalized)
                or (behavior_enabled and any(name.startswith("correction/") for name in normalized))
                or not any(name.startswith("set/") for name in normalized)
            ):
                raise ValueError(
                    "representation step omitted its predictive/set terms or contained a "
                    "forbidden correction/action term"
                )
            if behavior_enabled:
                rollout_terms = {name for name in normalized if name.startswith("rollout/")}
                if rollout_terms != {"rollout/vision/binding"}:
                    raise ValueError(
                        "behavior representation step changed its sole causal rollout target"
                    )
                if valid_counts["rollout/vision/binding"] <= 0:
                    raise ValueError(
                        "behavior representation step has no valid causal rollout target"
                    )
            for name, measured in normalized.items():
                _finite_report_number(
                    f"representation rank {rank} normalized term {name}",
                    measured,
                )
                count = valid_counts[name]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("representation valid counts must be non-negative integers")
            if valid_counts.get("set/ownership", 0) <= 0:
                raise ValueError(
                    "representation CALVIN step omitted exclusive ownership supervision"
                )
            if valid_counts.get("set/support", 0) != 0:
                raise ValueError(
                    "representation CALVIN step activated independent support supervision"
                )
            predictive_active = any(
                count > 0
                for name, count in valid_counts.items()
                if name.startswith(("correction/", "rollout/", "xmod/"))
            )
            structural_active = any(
                count > 0 for name, count in valid_counts.items() if name.startswith("set/")
            )
            if not structural_active:
                raise ValueError("representation step did not activate structural supervision")
            validate_task_row_diagnostics(
                item["task_row_diagnostics"],
                expected_batch_size=len(item["sample_keys"]),
            )
            step_bindings = _validate_step_row_bindings(
                prior_value=item["prior_row_bindings"],
                current_value=item["row_bindings"],
                reported_birth_count=item["row_binding_birth_count"],
                task_row_diagnostics=item["task_row_diagnostics"],
                expected_batch_size=len(item["sample_keys"]),
            )
            if _validate_step_estimator_transaction(item):
                _advance_report_row_binding_continuity(
                    reported_lane_bindings,
                    lane_ids=item["lane_ids"],
                    frame_indices=item["frame_indices"],
                    state_ages=item["state_ages"],
                    step_bindings=step_bindings,
                )
            if predictive_active:
                observed_step_predictive_ranks[item["global_step"]].add(rank)
            _recompute_representation_report_objective(
                item=item,
                mode=mode,
                contract=contract,
            )

            source_selected = item["source_masked_branch"]
            source_mask_digest = item["source_mask_digest"]
            source_mask_query_count = item["source_mask_query_count"]
            reported_source_mode = item["source_prediction_mode"]
            omitted_static_digest = item["omitted_static_digest"]
            if not isinstance(source_selected, bool) or (
                isinstance(source_mask_query_count, bool)
                or not isinstance(source_mask_query_count, int)
                or source_mask_query_count < 0
            ):
                raise ValueError("representation source-branch evidence is malformed")
            if source_selected:
                if reported_source_mode != mode:
                    raise ValueError("representation source branch executed another mode")
                if mode == "current_grid":
                    _require_sha256(
                        "representation source-mask digest",
                        source_mask_digest,
                    )
                    if source_mask_query_count <= 0 or omitted_static_digest is not None:
                        raise ValueError("representation current-grid evidence is inconsistent")
                else:
                    _require_sha256(
                        "representation omitted-static digest",
                        omitted_static_digest,
                    )
                    if source_mask_digest is not None or source_mask_query_count != 0:
                        raise ValueError("representation omitted-static evidence is inconsistent")
            elif (
                reported_source_mode is not None
                or source_mask_digest is not None
                or source_mask_query_count != 0
                or omitted_static_digest is not None
            ):
                raise ValueError("representation inactive source evidence is inconsistent")

            metrics = item["gradient_metrics"]
            if (
                not isinstance(metrics, dict)
                or set(metrics) != FULL_GRADIENT_REPORT_FIELDS
                or metrics["all_finite"] is not True
            ):
                raise ValueError("representation gradients are absent or non-finite")
            _finite_report_number(
                f"representation rank {rank} native-graph gradient norm",
                metrics["native_graph_norm"],
                positive=True,
            )
            _positive_report_integer(
                f"representation rank {rank} native-graph gradient elements",
                metrics["native_graph_elements"],
            )
            for prefix in ("relation_projection", "match_projection"):
                _finite_report_number(
                    f"representation rank {rank} {prefix} gradient norm",
                    metrics[f"{prefix}_norm"],
                    positive=True,
                )
                _positive_report_integer(
                    f"representation rank {rank} {prefix} gradient elements",
                    metrics[f"{prefix}_elements"],
                )
            if (
                _finite_report_number(
                    f"representation rank {rank} action-output gradient norm",
                    metrics["action_output_norm"],
                )
                != 0
                or metrics["action_output_elements"] != 0
            ):
                raise ValueError("representation objective reached the frozen action output")
            predictive_gradient = _finite_report_number(
                f"representation rank {rank} predictive-readout gradient norm",
                metrics["predictive_readout_norm"],
            )
            predictive_elements = metrics["predictive_readout_elements"]
            if (
                predictive_gradient < 0
                or isinstance(predictive_elements, bool)
                or not isinstance(predictive_elements, int)
                or predictive_elements < 0
            ):
                raise ValueError("representation predictive gradient coverage is malformed")
            predictive_summary = (predictive_gradient, predictive_elements)
            previous_summary = observed_step_predictive_gradients.setdefault(
                item["global_step"],
                predictive_summary,
            )
            if previous_summary != predictive_summary:
                raise ValueError("representation distributed predictive gradients differ by rank")
            behavior_posterior_norm = _finite_report_number(
                f"representation rank {rank} behavior-posterior gradient norm",
                metrics["behavior_posterior_gradient_norm"],
            )
            behavior_posterior_elements = metrics["behavior_posterior_gradient_elements"]
            if (
                isinstance(behavior_posterior_elements, bool)
                or not isinstance(behavior_posterior_elements, int)
                or behavior_posterior_elements < 0
            ):
                raise ValueError("representation behavior-posterior gradient size is invalid")
            if behavior_enabled:
                if behavior_posterior_norm <= 0 or behavior_posterior_elements <= 0:
                    raise ValueError(
                        "behavior objective did not reach the deploy posterior activation"
                    )
            elif behavior_posterior_norm != 0 or behavior_posterior_elements != 0:
                raise ValueError("ordinary representation training fabricated behavior credit")
            _finite_report_number(
                f"representation rank {rank} preclip global norm",
                metrics["preclip_global_norm"],
                positive=True,
            )

            diagnostic = item["family_gradient_diagnostics"]
            relation_diagnostic = item["relation_surface_gradient_diagnostics"]
            host_diagnostic = item["predictive_host_gradient_diagnostics"]
            counterfactual_diagnostic = item["predictive_counterfactual_diagnostics"]
            counterfactual_boundary = item["predictive_counterfactual_weight_boundary"]
            if item["global_step"] in audit_steps:
                _validate_representation_family_gradient_report(diagnostic)
                _validate_relation_surface_gradient_report(
                    relation_diagnostic,
                    estimator=task_relation_estimator,
                )
                _validate_predictive_host_gradient_report(
                    host_diagnostic,
                    expected_probe=(
                        "lingbot.language_model.input_layernorm.via_primary_posterior_vjp"
                        if behavior_enabled
                        else "lingbot.language_model.input_layernorm"
                    ),
                )
                if behavior_enabled:
                    if counterfactual_diagnostic is not None or counterfactual_boundary is not None:
                        raise ValueError("behavior G1 fabricated a prior-correction counterfactual")
                else:
                    if any(frame <= 0 for frame in item["frame_indices"]) or any(
                        age <= 0 for age in item["state_ages"]
                    ):
                        raise ValueError(
                            "representation prior-correction audit ran before recurrent bootstrap"
                        )
                    parsed_counterfactual = predictive_correction_counterfactual_from_mapping(
                        counterfactual_diagnostic
                    )
                    if counterfactual_boundary != "pre_update_post_backward":
                        raise ValueError(
                            "representation counterfactual used another weight boundary"
                        )
                    intervention_by_name = {
                        intervention.name: intervention
                        for intervention in parsed_counterfactual.interventions
                    }
                    required_interventions = {
                        ZERO_SOURCE,
                        ROW_SHIFT_SOURCE,
                        BATCH_SHIFT_SOURCE,
                        ZERO_CONTROL,
                        BATCH_SHIFT_CONTROL,
                        ZERO_CURRENT_OBSERVATION,
                    }
                    if not intervention_by_name.keys() >= required_interventions:
                        raise ValueError(
                            "representation predictive counterfactual coverage is incomplete"
                        )
                    if (
                        intervention_by_name[ZERO_CURRENT_OBSERVATION].normalized_prediction_l1
                        > 1e-6
                    ):
                        raise ValueError(
                            "representation prior correction read the current observation"
                        )
                    mature = all(frame >= 2 for frame in item["frame_indices"])
                    has_wrong_time = WRONG_TIME_SOURCE in intervention_by_name
                    if mature is not has_wrong_time:
                        raise ValueError(
                            "representation wrong-time intervention differs from state maturity"
                        )
            elif any(
                measured is not None
                for measured in (
                    diagnostic,
                    relation_diagnostic,
                    host_diagnostic,
                    counterfactual_diagnostic,
                    counterfactual_boundary,
                )
            ):
                raise ValueError("representation report contains an unscheduled diagnostic")

            allocated = _positive_report_integer(
                f"representation rank {rank} peak CUDA allocated bytes",
                item["peak_cuda_allocated_bytes"],
            )
            reserved = _positive_report_integer(
                f"representation rank {rank} peak CUDA reserved bytes",
                item["peak_cuda_reserved_bytes"],
            )
            if allocated > reserved or reserved > value["maximum_peak_reserved_bytes"]:
                raise ValueError("representation CUDA memory exceeded its registered budget")

    for step in expected_steps:
        routes = sorted(observed_step_routes[step], key=lambda item: item[0])
        if len(routes) != FULL_WORLD_SIZE or {rank for rank, _samples, _metadata in routes} != set(
            range(FULL_WORLD_SIZE)
        ):
            raise ValueError("representation step routing omitted a distributed rank")
        fixed_observation_pair = validate_fixed_observation_training_rank_metadata(
            [metadata for _rank, _samples, metadata in routes],
            expected_world_size=FULL_WORLD_SIZE,
        )
        sample_sets = [samples for _rank, samples, _metadata in routes]
        if fixed_observation_pair:
            if any(samples != sample_sets[0] for samples in sample_sets[1:]):
                raise ValueError(
                    "fixed-X representation ranks did not consume the same source samples"
                )
        else:
            observed: set[str] = set()
            for samples in sample_sets:
                if observed.intersection(samples):
                    raise ValueError("representation ranks consumed overlapping samples")
                observed.update(samples)

    bootstrap_only = input_step == 0 and saved_step == 1
    if not bootstrap_only and not any(observed_step_predictive_ranks.values()):
        raise ValueError("representation report contains no active predictive supervision")
    for step in expected_steps:
        predictive_ranks = observed_step_predictive_ranks[step]
        predictive_norm, predictive_elements = observed_step_predictive_gradients[step]
        if predictive_ranks:
            if predictive_norm <= 0 or predictive_elements <= 0:
                raise ValueError(
                    "representation active predictive family produced no readout gradient"
                )
        # An inactive differentiable family may allocate exact-zero gradient tensors.
        # Activity is defined by a nonzero gradient, not by tensor allocation.
        elif predictive_norm != 0:
            raise ValueError(
                "representation inactive predictive family produced a readout gradient"
            )
        if step in audit_steps and predictive_ranks != set(range(FULL_WORLD_SIZE)):
            raise ValueError(
                "representation gradient audit lacked predictive targets on every rank"
            )

    if require_initial_probe:
        if (input_step, saved_step) != (0, 1) or 1 in audit_steps:
            raise ValueError(
                "representation initial probe requires one unaudited fresh bootstrap step"
            )
        for rank_report in rank_reports:
            item = rank_report["steps"][0]
            if item["frame_indices"] != [0] or item["state_ages"] != [0]:
                raise ValueError("fresh representation report did not consume exact reset frames")
            if item["local_bptt_steps"] == 1 and any(
                count > 0
                for name, count in item["valid_counts"].items()
                if name.startswith("correction/")
            ):
                raise ValueError(
                    "fresh representation bootstrap fabricated prior-correction support"
                )

    checkpoint_dir = Path(checkpoint_hint_value)
    if (
        not checkpoint_dir.is_absolute()
        or checkpoint_dir.is_symlink()
        or checkpoint_dir.name != f"global_step_{saved_step}"
    ):
        raise ValueError("input representation checkpoint path is malformed")
    if checkpoint_publication == "never":
        if require_checkpoint_copy:
            raise ValueError("non-published representation evidence is not a checkpoint")
        if checkpoint_dir.exists() or checkpoint_dir.is_symlink():
            raise ValueError("non-published representation evidence has a checkpoint artifact")
    elif require_checkpoint_copy:
        if not checkpoint_dir.is_dir():
            raise ValueError("input representation report has no real checkpoint directory")
        checkpoint_report = checkpoint_dir / "native_representation_report.json"
        if checkpoint_report.is_symlink() or not checkpoint_report.is_file():
            raise ValueError("input representation checkpoint lacks its immutable report copy")
        try:
            checkpoint_value = json.loads(checkpoint_report.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("input representation checkpoint report is invalid JSON") from error
        if checkpoint_value != value:
            raise ValueError(
                "input representation checkpoint report differs from the reviewed report"
            )
    return value


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = _environment_path("PICF_LINGBOT_NATIVE_SOURCE") or (
        root / CHECKOUT_RELATIVE_PATH
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fresh", "resume"), required=True)
    parser.add_argument(
        "--training-stage",
        choices=FULL_TRAINING_STAGES,
        default=FULL_ACTION_STAGE,
    )
    parser.add_argument(
        "--representation-split",
        type=Path,
        default=_environment_path("PICF_LINGBOT_REPRESENTATION_SPLIT"),
    )
    parser.add_argument(
        "--representation-split-sha256",
        default=os.environ.get("PICF_LINGBOT_REPRESENTATION_SPLIT_SHA256"),
    )
    parser.add_argument(
        "--representation-task-intervention-plan",
        type=Path,
        default=_environment_path("PICF_LINGBOT_REPRESENTATION_TASK_INTERVENTION"),
    )
    parser.add_argument(
        "--representation-task-intervention-plan-sha256",
        default=os.environ.get("PICF_LINGBOT_REPRESENTATION_TASK_INTERVENTION_SHA256"),
    )
    parser.add_argument(
        "--fixed-observation-pair-plan",
        type=Path,
        default=_environment_path("PICF_LINGBOT_FIXED_OBSERVATION_PAIR_PLAN"),
    )
    parser.add_argument(
        "--fixed-observation-pair-plan-sha256",
        default=os.environ.get("PICF_LINGBOT_FIXED_OBSERVATION_PAIR_PLAN_SHA256"),
    )
    parser.add_argument(
        "--fixed-observation-training-audit",
        type=Path,
        default=_environment_path("PICF_LINGBOT_FIXED_OBSERVATION_TRAINING_AUDIT"),
    )
    parser.add_argument(
        "--fixed-observation-training-audit-sha256",
        default=os.environ.get("PICF_LINGBOT_FIXED_OBSERVATION_TRAINING_AUDIT_SHA256"),
    )
    parser.add_argument(
        "--fixed-observation-evaluation-plan",
        type=Path,
        default=_environment_path("PICF_LINGBOT_FIXED_OBSERVATION_EVALUATION_PLAN"),
    )
    parser.add_argument(
        "--fixed-observation-evaluation-plan-sha256",
        default=os.environ.get("PICF_LINGBOT_FIXED_OBSERVATION_EVALUATION_PLAN_SHA256"),
    )
    parser.add_argument(
        "--fixed-observation-validation-audit",
        type=Path,
        default=_environment_path("PICF_LINGBOT_FIXED_OBSERVATION_VALIDATION_AUDIT"),
    )
    parser.add_argument(
        "--fixed-observation-validation-audit-sha256",
        default=os.environ.get("PICF_LINGBOT_FIXED_OBSERVATION_VALIDATION_AUDIT_SHA256"),
    )
    parser.add_argument(
        "--fixed-observation-heldout-audit",
        type=Path,
        default=_environment_path("PICF_LINGBOT_FIXED_OBSERVATION_HELDOUT_AUDIT"),
    )
    parser.add_argument(
        "--fixed-observation-heldout-audit-sha256",
        default=os.environ.get("PICF_LINGBOT_FIXED_OBSERVATION_HELDOUT_AUDIT_SHA256"),
    )
    parser.add_argument(
        "--representation-evaluation-plan",
        type=Path,
        default=_environment_path("PICF_LINGBOT_REPRESENTATION_EVALUATION_PLAN"),
    )
    parser.add_argument(
        "--representation-evaluation-plan-sha256",
        default=os.environ.get("PICF_LINGBOT_REPRESENTATION_EVALUATION_PLAN_SHA256"),
    )
    parser.add_argument(
        "--representation-evaluation-baseline",
        type=Path,
        default=_environment_path("PICF_LINGBOT_REPRESENTATION_EVALUATION_BASELINE"),
    )
    parser.add_argument(
        "--representation-evaluation-baseline-sha256",
        default=os.environ.get("PICF_LINGBOT_REPRESENTATION_EVALUATION_BASELINE_SHA256"),
    )
    parser.add_argument(
        "--representation-warm-evaluation-plan",
        type=Path,
        default=_environment_path("PICF_LINGBOT_REPRESENTATION_WARM_EVALUATION_PLAN"),
    )
    parser.add_argument(
        "--representation-warm-evaluation-plan-sha256",
        default=os.environ.get("PICF_LINGBOT_REPRESENTATION_WARM_EVALUATION_PLAN_SHA256"),
    )
    parser.add_argument(
        "--representation-evaluation-steps",
        type=_parse_nonnegative_step_set,
        default=(),
        help=(
            "checkpoint boundaries to evaluate in this invocation; requires the immutable "
            "representation evaluation plan"
        ),
    )
    parser.add_argument("--source-checkout", type=Path, default=source_default)
    parser.add_argument("--patch", type=Path, default=root / PATCH_RELATIVE_PATH)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=root / "configs/lingbot/calvin_robot.yaml",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=root / "configs/lingbot/calvin_data.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_environment_path("PICF_CHECKPOINT_DIR"),
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        default=_environment_path("PICF_PROCESSOR_DIR"),
    )
    parser.add_argument("--dataset-split", type=Path, default=_environment_path("PICF_DATASET_DIR"))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=_environment_path("PICF_DATASET_MANIFEST"),
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=_environment_path("PICF_LINGBOT_NORM_STATS"),
    )
    parser.add_argument(
        "--physical-sidecar-root",
        type=Path,
        default=_environment_path("PICF_CALVIN_PHYSICAL_SIDECAR"),
    )
    parser.add_argument(
        "--physical-sidecar-manifest",
        type=Path,
        default=_environment_path("PICF_CALVIN_PHYSICAL_SIDECAR_MANIFEST"),
    )
    parser.add_argument(
        "--physical-sidecar-manifest-sha256",
        default=os.environ.get("PICF_CALVIN_PHYSICAL_SIDECAR_SHA256"),
    )
    parser.add_argument(
        "--physical-visual-acceptance",
        type=Path,
        default=_environment_path("PICF_CALVIN_PHYSICAL_VISUAL_ACCEPTANCE"),
    )
    parser.add_argument(
        "--physical-visual-acceptance-sha256",
        default=os.environ.get("PICF_CALVIN_PHYSICAL_VISUAL_ACCEPTANCE_SHA256"),
    )
    parser.add_argument(
        "--predictive-cache-root",
        type=Path,
        default=_environment_path("PICF_LINGBOT_PREDICTIVE_CACHE"),
    )
    parser.add_argument(
        "--predictive-cache-build-report",
        type=Path,
        default=_environment_path("PICF_LINGBOT_PREDICTIVE_CACHE_REPORT"),
    )
    parser.add_argument(
        "--predictive-cache-build-report-sha256",
        default=os.environ.get("PICF_LINGBOT_PREDICTIVE_CACHE_REPORT_SHA256"),
    )
    parser.add_argument(
        "--predictive-target-audit",
        type=Path,
        default=_environment_path("PICF_LINGBOT_PREDICTIVE_TARGET_AUDIT"),
    )
    parser.add_argument(
        "--predictive-target-audit-sha256",
        default=os.environ.get("PICF_LINGBOT_PREDICTIVE_TARGET_AUDIT_SHA256"),
    )
    parser.add_argument(
        "--predictive-teacher-causality-audit",
        type=Path,
        default=_environment_path("PICF_LINGBOT_PREDICTIVE_TEACHER_CAUSALITY_AUDIT"),
    )
    parser.add_argument(
        "--predictive-teacher-causality-audit-sha256",
        default=os.environ.get("PICF_LINGBOT_PREDICTIVE_TEACHER_CAUSALITY_AUDIT_SHA256"),
    )
    parser.add_argument(
        "--predictive-temporal-audit",
        type=Path,
        default=_environment_path("PICF_LINGBOT_PREDICTIVE_TEMPORAL_AUDIT"),
    )
    parser.add_argument(
        "--predictive-temporal-audit-sha256",
        default=os.environ.get("PICF_LINGBOT_PREDICTIVE_TEMPORAL_AUDIT_SHA256"),
    )
    parser.add_argument(
        "--current-grid-cache-root",
        type=Path,
        default=_environment_path("PICF_LINGBOT_CURRENT_GRID_CACHE"),
    )
    parser.add_argument(
        "--current-grid-cache-build-report",
        type=Path,
        default=_environment_path("PICF_LINGBOT_CURRENT_GRID_CACHE_REPORT"),
    )
    parser.add_argument(
        "--current-grid-cache-build-report-sha256",
        default=os.environ.get("PICF_LINGBOT_CURRENT_GRID_CACHE_REPORT_SHA256"),
    )
    parser.add_argument("--run-dir", type=Path, default=_environment_path("PICF_RUN_DIR"))
    parser.add_argument(
        "--authorization-manifest",
        type=Path,
        default=_environment_path("PICF_LINGBOT_TRAINING_AUTHORIZATION"),
    )
    parser.add_argument(
        "--authorization-manifest-sha256",
        default=os.environ.get("PICF_LINGBOT_TRAINING_AUTHORIZATION_SHA256"),
    )
    parser.add_argument("--load-global-step", type=int, default=0)
    parser.add_argument("--invocation-steps", type=int, default=1)
    parser.add_argument("--total-planned-steps", type=int, default=30_000)
    parser.add_argument(
        "--checkpoint-publication",
        choices=CHECKPOINT_PUBLICATION_MODES,
        default="always",
        help=(
            "always publishes one resumable final checkpoint; never is restricted to a fresh "
            "bounded representation diagnostic with immutable final evaluation"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--maximum-peak-reserved-gib", type=float, default=39.0)
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
        help=(
            "Explicit parameter-shard placement; selective embedding offload is the "
            "released two-A100 acceptance topology."
        ),
    )
    parser.add_argument(
        "--cuda-allocator",
        choices=CUDA_ALLOCATOR_MODES,
        default="native",
        help="Explicit pinned-runtime CUDA allocator mode; inherited allocator settings fail.",
    )
    parser.add_argument("--local-bptt-probability", type=float, default=0.10)
    parser.add_argument("--overshoot-probability", type=float, default=0.05)
    parser.add_argument("--source-mask-probability", type=float, default=0.10)
    parser.add_argument(
        "--behavior-conditioned-prediction",
        action="store_true",
        help=(
            "run the bounded posterior-and-future-action prediction view; until the released-"
            "weight causal gate passes this is restricted to a zero-update probe"
        ),
    )
    parser.add_argument(
        "--behavior-causal-probe-output",
        type=Path,
        default=None,
        help="new immutable JSON output for the released-weight behavior-causal gate",
    )
    parser.add_argument(
        "--behavior-causal-probe-evidence",
        type=Path,
        default=None,
        help="immutable passed G0 report authorizing the bounded G1 update",
    )
    parser.add_argument(
        "--behavior-causal-probe-evidence-sha256",
        default=None,
        help="SHA-256 of the immutable G0 report authorizing the bounded G1 update",
    )
    parser.add_argument(
        "--behavior-posterior-control-probe-output",
        type=Path,
        default=None,
        help="new immutable JSON output for the loaded-G1 posterior/control factorial",
    )
    parser.add_argument(
        "--behavior-posterior-control-probe-evidence",
        type=Path,
        default=None,
        help="immutable passed G2 receipt authorizing joint action adoption",
    )
    parser.add_argument(
        "--behavior-posterior-control-probe-evidence-sha256",
        default=None,
        help="SHA-256 of the immutable G2 receipt authorizing joint action adoption",
    )
    parser.add_argument(
        "--behavior-g1-predecessor-report-sha256",
        default=None,
        help="SHA-256 of the exact step-2 G1 report defining the zero-update G2 boundary",
    )
    parser.add_argument(
        "--source-prediction-mode",
        choices=("omitted_static", "current_grid"),
        default="omitted_static",
    )
    parser.add_argument("--source-mask-token-fraction", type=float, default=0.0625)
    parser.add_argument("--maximum-optimizer-lag", type=int, default=8)
    parser.add_argument(
        "--lane-interleave-factor",
        type=int,
        default=1,
        help=(
            "Number of detached causal lanes rotated through each active global "
            "sample slot; one preserves the released stream plan."
        ),
    )
    add_reset_mixture_arguments(parser)
    parser.add_argument("--predictive-weight", type=float, default=None)
    parser.add_argument("--structural-weight", type=float, default=None)
    parser.add_argument(
        "--evidence-profile",
        choices=(
            "acceptance",
            "loss_visual_trial",
            "behavior_discrimination_trial",
            MATCHED_MEDIUM_HORIZON_PROFILE,
        ),
        default="acceptance",
        help=(
            "acceptance executes preregistered family audits; loss_visual_trial is a "
            "fail-closed <=20-step loss/visual run whose family audits must lie later; "
            "behavior_discrimination_trial is the exact fresh 60-step joint "
            "behavior/action discrimination contract; matched_medium_horizon is the "
            "frozen ADR-135 0/200/500/1000 representation curve"
        ),
    )
    parser.add_argument("--gradient-audit-steps", type=_parse_positive_step_set, default=None)
    parser.add_argument("--support-weight", type=float, default=0.0)
    parser.add_argument("--existence-weight", type=float, default=1.0)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument(
        "--task-relation-estimator",
        choices=TASK_RELATION_ESTIMATORS,
        default=LOCAL_BALANCED_TASK_RELATION,
    )
    parser.add_argument("--dense-task-weight", type=float, default=1.0)
    parser.add_argument("--ownership-weight", type=float, default=1.0)
    parser.add_argument(
        "--ownership-estimator",
        choices=OWNERSHIP_ESTIMATORS,
        default=TOKEN_MICRO_OWNERSHIP,
    )
    parser.add_argument(
        "--relation-supervision-layers",
        type=_parse_nonnegative_layer_set,
        default=(),
        help=(
            "training-only zero-based host depths for the shared relation readout; "
            "the final normalized surface remains canonical"
        ),
    )
    parser.add_argument("--predictive-term-weight", type=float, default=1.0)
    parser.add_argument("--current-grid-term-weight", type=float, default=1.0)
    parser.add_argument("--omitted-static-term-weight", type=float, default=1.0)
    parser.add_argument("--predictive-loss-power", type=float, default=1.0)
    parser.add_argument("--minimum-supervised-fraction", type=float, default=0.0)
    parser.add_argument("--predictive-cache-memory-shards", type=int, default=2)
    parser.add_argument("--current-grid-cache-memory-shards", type=int, default=2)
    parser.add_argument(
        "--visual-audit-every",
        type=int,
        default=0,
        help="render current LingBot relation evidence every N saved steps; zero disables it",
    )
    parser.add_argument(
        "--predictive-fixed-batch-arm",
        choices=PREDICTIVE_FIXED_BATCH_ARMS,
        default=None,
        help="run one evidence-only predictive capacity arm instead of checkpoint training",
    )
    parser.add_argument(
        "--predictive-fixed-batch-curve-points",
        type=int,
        default=0,
        help=(
            "equal loss-curve points for one fixed-batch arm, including the common "
            "initial point; optimizer updates equal points minus one; zero disables the probe"
        ),
    )
    parser.add_argument(
        "--predictive-fixed-batch-output",
        type=Path,
        default=None,
        help="new immutable JSON report path for the selected fixed-batch arm",
    )
    parser.add_argument(
        "--relation-geometry-fixed-batch-arm",
        choices=RELATION_FIXED_BATCH_ARMS,
        default=None,
        help="run one disposable relation-recoverability arm instead of checkpoint training",
    )
    parser.add_argument(
        "--relation-geometry-fixed-batch-curve-points",
        type=int,
        default=0,
        help="ownership curve points including point zero; updates equal points minus one",
    )
    parser.add_argument(
        "--relation-geometry-fixed-batch-sample-step",
        type=int,
        default=0,
        help=(
            "zero-based scan start; the probe selects the earliest source-only "
            "two-rank exact-task observation at or after this stream-plan step"
        ),
    )
    parser.add_argument(
        "--relation-geometry-fixed-batch-output",
        type=Path,
        default=None,
        help="new immutable JSON report path for the ownership-recoverability arm",
    )
    parser.add_argument(
        "--relation-geometry-fixed-batch-visual-root",
        type=Path,
        default=None,
        help="new directory that receives task-labelled visuals at every curve point",
    )
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _validate_paths_and_args(args: argparse.Namespace) -> None:
    training_stage = getattr(args, "training_stage", FULL_ACTION_STAGE)
    checkpoint_publication = getattr(args, "checkpoint_publication", "always")
    behavior_conditioned = bool(getattr(args, "behavior_conditioned_prediction", False))
    behavior_probe_output = getattr(args, "behavior_causal_probe_output", None)
    behavior_probe_evidence = getattr(args, "behavior_causal_probe_evidence", None)
    behavior_probe_evidence_sha256 = getattr(
        args,
        "behavior_causal_probe_evidence_sha256",
        None,
    )
    behavior_factorial_output = getattr(
        args,
        "behavior_posterior_control_probe_output",
        None,
    )
    behavior_factorial_evidence = getattr(
        args,
        "behavior_posterior_control_probe_evidence",
        None,
    )
    behavior_factorial_evidence_sha256 = getattr(
        args,
        "behavior_posterior_control_probe_evidence_sha256",
        None,
    )
    behavior_predecessor_sha256 = getattr(
        args,
        "behavior_g1_predecessor_report_sha256",
        None,
    )
    if checkpoint_publication not in CHECKPOINT_PUBLICATION_MODES:
        raise ValueError("native checkpoint-publication mode is unsupported")
    representation_split_path = getattr(args, "representation_split", None)
    if representation_split_path is not None and not isinstance(
        representation_split_path,
        Path,
    ):
        raise TypeError("representation split path must be a pathlib.Path")
    representation_split_file: Path | None = (
        representation_split_path if isinstance(representation_split_path, Path) else None
    )
    representation_split_sha256 = getattr(
        args,
        "representation_split_sha256",
        None,
    )
    representation_task_intervention = getattr(
        args,
        "representation_task_intervention_plan",
        None,
    )
    if representation_task_intervention is not None and not isinstance(
        representation_task_intervention,
        Path,
    ):
        raise TypeError("representation task-intervention path must be a pathlib.Path")
    representation_task_intervention_file: Path | None = (
        representation_task_intervention
        if isinstance(representation_task_intervention, Path)
        else None
    )
    representation_task_intervention_sha256 = getattr(
        args,
        "representation_task_intervention_plan_sha256",
        None,
    )
    fixed_observation_pair_plan = getattr(
        args,
        "fixed_observation_pair_plan",
        None,
    )
    if fixed_observation_pair_plan is not None and not isinstance(
        fixed_observation_pair_plan,
        Path,
    ):
        raise TypeError("fixed-observation pair-plan path must be a pathlib.Path")
    fixed_observation_pair_plan_file: Path | None = (
        fixed_observation_pair_plan if isinstance(fixed_observation_pair_plan, Path) else None
    )
    fixed_observation_pair_plan_sha256 = getattr(
        args,
        "fixed_observation_pair_plan_sha256",
        None,
    )
    fixed_observation_training_audit = getattr(
        args,
        "fixed_observation_training_audit",
        None,
    )
    if fixed_observation_training_audit is not None and not isinstance(
        fixed_observation_training_audit,
        Path,
    ):
        raise TypeError("fixed-observation training-audit path must be a pathlib.Path")
    fixed_observation_training_audit_file: Path | None = (
        fixed_observation_training_audit
        if isinstance(fixed_observation_training_audit, Path)
        else None
    )
    fixed_observation_training_audit_sha256 = getattr(
        args,
        "fixed_observation_training_audit_sha256",
        None,
    )
    fixed_observation_evaluation_plan = getattr(
        args,
        "fixed_observation_evaluation_plan",
        None,
    )
    if fixed_observation_evaluation_plan is not None and not isinstance(
        fixed_observation_evaluation_plan,
        Path,
    ):
        raise TypeError("fixed-observation evaluation-plan path must be a pathlib.Path")
    fixed_observation_evaluation_plan_file: Path | None = (
        fixed_observation_evaluation_plan
        if isinstance(fixed_observation_evaluation_plan, Path)
        else None
    )
    fixed_observation_evaluation_plan_sha256 = getattr(
        args,
        "fixed_observation_evaluation_plan_sha256",
        None,
    )
    fixed_observation_validation_audit = getattr(
        args,
        "fixed_observation_validation_audit",
        None,
    )
    if fixed_observation_validation_audit is not None and not isinstance(
        fixed_observation_validation_audit,
        Path,
    ):
        raise TypeError("fixed-observation validation-audit path must be a pathlib.Path")
    fixed_observation_validation_audit_file: Path | None = (
        fixed_observation_validation_audit
        if isinstance(fixed_observation_validation_audit, Path)
        else None
    )
    fixed_observation_validation_audit_sha256 = getattr(
        args,
        "fixed_observation_validation_audit_sha256",
        None,
    )
    fixed_observation_heldout_audit = getattr(
        args,
        "fixed_observation_heldout_audit",
        None,
    )
    if fixed_observation_heldout_audit is not None and not isinstance(
        fixed_observation_heldout_audit,
        Path,
    ):
        raise TypeError("fixed-observation heldout-audit path must be a pathlib.Path")
    fixed_observation_heldout_audit_file: Path | None = (
        fixed_observation_heldout_audit
        if isinstance(fixed_observation_heldout_audit, Path)
        else None
    )
    fixed_observation_heldout_audit_sha256 = getattr(
        args,
        "fixed_observation_heldout_audit_sha256",
        None,
    )
    representation_evaluation_plan = getattr(args, "representation_evaluation_plan", None)
    if representation_evaluation_plan is not None and not isinstance(
        representation_evaluation_plan,
        Path,
    ):
        raise TypeError("representation evaluation plan path must be a pathlib.Path")
    representation_evaluation_plan_file: Path | None = (
        representation_evaluation_plan if isinstance(representation_evaluation_plan, Path) else None
    )
    representation_evaluation_plan_sha256 = getattr(
        args,
        "representation_evaluation_plan_sha256",
        None,
    )
    representation_evaluation_baseline = getattr(
        args,
        "representation_evaluation_baseline",
        None,
    )
    if representation_evaluation_baseline is not None and not isinstance(
        representation_evaluation_baseline,
        Path,
    ):
        raise TypeError("representation evaluation baseline path must be a pathlib.Path")
    representation_evaluation_baseline_file: Path | None = (
        representation_evaluation_baseline
        if isinstance(representation_evaluation_baseline, Path)
        else None
    )
    representation_evaluation_baseline_sha256 = getattr(
        args,
        "representation_evaluation_baseline_sha256",
        None,
    )
    representation_warm_evaluation_plan = args.representation_warm_evaluation_plan
    if representation_warm_evaluation_plan is not None and not isinstance(
        representation_warm_evaluation_plan,
        Path,
    ):
        raise TypeError("representation warm-evaluation plan path must be a pathlib.Path")
    representation_warm_evaluation_plan_file: Path | None = (
        representation_warm_evaluation_plan
        if isinstance(representation_warm_evaluation_plan, Path)
        else None
    )
    representation_warm_evaluation_plan_sha256 = args.representation_warm_evaluation_plan_sha256
    warm_evaluation_values = (
        representation_warm_evaluation_plan_file,
        representation_warm_evaluation_plan_sha256,
    )
    warm_evaluation_requested = any(value is not None for value in warm_evaluation_values)
    if warm_evaluation_requested and (
        representation_warm_evaluation_plan_file is None
        or representation_warm_evaluation_plan_sha256 is None
    ):
        raise ValueError("partial representation warm-evaluation arguments are forbidden")
    representation_evaluation_steps = getattr(args, "representation_evaluation_steps", ())
    if training_stage not in FULL_TRAINING_STAGES:
        raise ValueError("native full-objective training stage is unsupported")
    representation_stage = training_stage == NATIVE_REPRESENTATION_STAGE
    behavior_action_training = behavior_conditioned and not representation_stage
    reset_mixture = reset_mixture_values(args)
    if (reset_mixture is not None) is not warm_evaluation_requested:
        raise ValueError("ADR-121 reset mixture and warm-evaluation plan must be provided together")
    split: RepresentationTrialSplit | None = None
    if representation_stage or behavior_action_training:
        if representation_split_file is None:
            raise ValueError("behavior/representation training requires an explicit source split")
        _require_sha256(
            "representation split sha256",
            representation_split_sha256,
        )
        if representation_stage and (
            args.authorization_manifest is not None
            or (args.authorization_manifest_sha256 is not None)
        ):
            raise ValueError("bounded representation training cannot consume action authorization")
    elif representation_split_file is not None or representation_split_sha256 is not None:
        raise ValueError("action training cannot consume a representation source split")
    intervention_values = (
        representation_task_intervention_file,
        representation_task_intervention_sha256,
    )
    intervention_requested = any(value is not None for value in intervention_values)
    if intervention_requested:
        if (
            not representation_stage
            or representation_task_intervention_file is None
            or representation_task_intervention_sha256 is None
        ):
            raise ValueError("representation task intervention requires its plan, digest and stage")
        _require_sha256(
            "representation task-intervention plan sha256",
            representation_task_intervention_sha256,
        )
    fixed_observation_values = (
        fixed_observation_pair_plan_file,
        fixed_observation_pair_plan_sha256,
        fixed_observation_training_audit_file,
        fixed_observation_training_audit_sha256,
    )
    fixed_observation_requested = any(value is not None for value in fixed_observation_values)
    if fixed_observation_requested:
        if (
            not representation_stage
            or reset_mixture is None
            or any(value is None for value in fixed_observation_values)
        ):
            raise ValueError(
                "fixed-observation pairing requires its plan, audit, digests, "
                "representation stage and reset mixture"
            )
        if intervention_requested:
            raise ValueError(
                "fixed-observation pairing and legacy task intervention are mutually exclusive"
            )
        _require_sha256(
            "fixed-observation pair-plan sha256",
            fixed_observation_pair_plan_sha256,
        )
        _require_sha256(
            "fixed-observation training-audit sha256",
            fixed_observation_training_audit_sha256,
        )
    fixed_observation_evaluation_values = (
        fixed_observation_evaluation_plan_file,
        fixed_observation_evaluation_plan_sha256,
        fixed_observation_validation_audit_file,
        fixed_observation_validation_audit_sha256,
        fixed_observation_heldout_audit_file,
        fixed_observation_heldout_audit_sha256,
    )
    fixed_observation_evaluation_requested = any(
        value is not None for value in fixed_observation_evaluation_values
    )
    if fixed_observation_evaluation_requested:
        if any(value is None for value in fixed_observation_evaluation_values):
            raise ValueError(
                "fixed-observation evaluation requires its plan, validation/heldout "
                "audits and all digests"
            )
        _require_sha256(
            "fixed-observation evaluation-plan sha256",
            fixed_observation_evaluation_plan_sha256,
        )
        _require_sha256(
            "fixed-observation validation-audit sha256",
            fixed_observation_validation_audit_sha256,
        )
        _require_sha256(
            "fixed-observation heldout-audit sha256",
            fixed_observation_heldout_audit_sha256,
        )
    evaluation_values = (
        representation_evaluation_plan_file,
        representation_evaluation_plan_sha256,
        representation_evaluation_steps,
    )
    evaluation_requested = any(bool(value) for value in evaluation_values)
    if evaluation_requested:
        if (
            not representation_stage
            or representation_evaluation_plan_file is None
            or representation_evaluation_plan_sha256 is None
            or not representation_evaluation_steps
        ):
            raise ValueError("representation evaluation requires its plan, digest, steps and stage")
    elif any(value not in (None, ()) for value in evaluation_values):
        raise ValueError("partial representation evaluation arguments are forbidden")
    if fixed_observation_requested != fixed_observation_evaluation_requested:
        raise ValueError(
            "fixed-observation training and source-disjoint evaluation must be bound together"
        )
    if fixed_observation_requested and not evaluation_requested:
        raise ValueError(
            "fixed-observation training requires checkpoint-boundary representation evaluation"
        )
    baseline_values = (
        representation_evaluation_baseline_file,
        representation_evaluation_baseline_sha256,
    )
    baseline_requested = any(value is not None for value in baseline_values)
    if baseline_requested and (
        representation_evaluation_baseline_file is None
        or representation_evaluation_baseline_sha256 is None
    ):
        raise ValueError("partial representation evaluation baseline arguments are forbidden")
    fixed_batch_arm = getattr(args, "predictive_fixed_batch_arm", None)
    fixed_batch_curve_points = getattr(args, "predictive_fixed_batch_curve_points", 0)
    fixed_batch_output = getattr(args, "predictive_fixed_batch_output", None)
    relation_arm = getattr(args, "relation_geometry_fixed_batch_arm", None)
    relation_curve_points = getattr(args, "relation_geometry_fixed_batch_curve_points", 0)
    relation_sample_step = getattr(args, "relation_geometry_fixed_batch_sample_step", 0)
    relation_output = getattr(args, "relation_geometry_fixed_batch_output", None)
    relation_visual_root = getattr(args, "relation_geometry_fixed_batch_visual_root", None)
    common_behavior_conflict = False
    if behavior_conditioned:
        behavior_factorial_evidence_values = (
            behavior_factorial_evidence,
            behavior_factorial_evidence_sha256,
        )
        behavior_factorial_evidence_requested = any(
            value is not None for value in behavior_factorial_evidence_values
        )
        if behavior_factorial_evidence_requested and any(
            value is None for value in behavior_factorial_evidence_values
        ):
            raise ValueError("partial behavior G2 evidence arguments are forbidden")
        if behavior_factorial_evidence is not None and not isinstance(
            behavior_factorial_evidence,
            Path,
        ):
            raise TypeError("behavior G2 evidence must be a pathlib.Path")
        if behavior_probe_output is not None and behavior_factorial_output is not None:
            raise ValueError("behavior G0 and G2 outputs are mutually exclusive")

        if behavior_action_training:
            bounded_loss_visual_trial = (
                args.evidence_profile == "loss_visual_trial"
                and tuple(args.gradient_audit_steps) == (2, 10, 20)
                and args.phase == "fresh"
                and args.load_global_step == 0
                and args.invocation_steps <= BEHAVIOR_ACTION_EVIDENCE_MAXIMUM_STEPS
                and args.visual_audit_every == 1
            )
            discrimination_trial = (
                args.evidence_profile == "behavior_discrimination_trial"
                and tuple(args.gradient_audit_steps) == BEHAVIOR_ACTION_DISCRIMINATION_AUDIT_STEPS
                and args.phase == "fresh"
                and args.load_global_step == 0
                and args.invocation_steps == BEHAVIOR_ACTION_DISCRIMINATION_STEPS
                and args.visual_audit_every == 1
            )
            if (
                behavior_probe_output is not None
                or behavior_factorial_output is not None
                or behavior_predecessor_sha256 is not None
                or not isinstance(behavior_probe_evidence, Path)
                or behavior_probe_evidence_sha256 is None
                or not behavior_factorial_evidence_requested
                or args.visual_audit_every <= 0
                or checkpoint_publication != "always"
                or fixed_batch_arm is not None
                or relation_arm is not None
                or fixed_observation_requested
                or intervention_requested
                or evaluation_requested
                or args.lane_interleave_factor != 8
                or not (bounded_loss_visual_trial or discrimination_trial)
            ):
                raise ValueError(
                    "joint behavior/action training requires exact G0/G2 receipts, periodic "
                    "learned-anchor visuals, the frozen gradient schedule and bounded "
                    "checkpoint publication"
                )
            _require_sha256(
                "behavior causal-probe evidence sha256",
                behavior_probe_evidence_sha256,
            )
            _require_sha256(
                "behavior G2 evidence sha256",
                behavior_factorial_evidence_sha256,
            )
        else:
            if behavior_factorial_evidence_requested:
                raise ValueError(
                    "representation behavior probes cannot consume G2 adoption evidence"
                )
            common_behavior_conflict = (
                args.invocation_steps != 1
                or args.evidence_profile != "loss_visual_trial"
                or args.visual_audit_every != 0
                or fixed_batch_arm is not None
                or relation_arm is not None
                or fixed_observation_requested
                or intervention_requested
                or evaluation_requested
            )
        if not behavior_action_training and behavior_probe_output is not None:
            if not isinstance(behavior_probe_output, Path):
                raise TypeError("behavior causal-probe output must be a pathlib.Path")
            if (
                behavior_probe_evidence is not None
                or behavior_probe_evidence_sha256 is not None
                or behavior_predecessor_sha256 is not None
            ):
                raise ValueError("behavior G0 output and G1 evidence are mutually exclusive")
            if (
                common_behavior_conflict
                or args.phase != "fresh"
                or args.load_global_step != 0
                or checkpoint_publication != "never"
            ):
                raise ValueError(
                    "behavior causal G0 requires one fresh representation-only loss/visual "
                    "invocation without another probe, overlay, evaluation, or visual output"
                )
            if behavior_probe_output.exists() or behavior_probe_output.is_symlink():
                raise FileExistsError(
                    f"behavior causal probe output already exists: {behavior_probe_output}"
                )
            try:
                behavior_probe_output.resolve().relative_to(args.run_dir.resolve())
            except ValueError as error:
                raise ValueError(
                    "behavior causal probe output must live below the run directory"
                ) from error
        elif not behavior_action_training and behavior_factorial_output is not None:
            if not isinstance(behavior_factorial_output, Path):
                raise TypeError("behavior G2 output must be a pathlib.Path")
            if (
                not isinstance(behavior_probe_evidence, Path)
                or behavior_probe_evidence_sha256 is None
                or behavior_predecessor_sha256 is None
            ):
                raise ValueError("behavior G2 requires immutable G0 and G1 receipts")
            _require_sha256(
                "behavior causal-probe evidence sha256",
                behavior_probe_evidence_sha256,
            )
            _require_sha256(
                "behavior G1 predecessor report sha256",
                behavior_predecessor_sha256,
            )
            if (
                common_behavior_conflict
                or args.phase != "resume"
                or args.load_global_step != 2
                or checkpoint_publication != "never"
                or tuple(args.gradient_audit_steps) != (1, 2)
                or args.lane_interleave_factor != 8
            ):
                raise ValueError(
                    "behavior G2 requires the exact loaded step-2 G1 boundary and one "
                    "zero-update posterior/control invocation"
                )
            if behavior_factorial_output.exists() or behavior_factorial_output.is_symlink():
                raise FileExistsError(
                    f"behavior G2 output already exists: {behavior_factorial_output}"
                )
            try:
                behavior_factorial_output.resolve().relative_to(args.run_dir.resolve())
            except ValueError as error:
                raise ValueError(
                    "behavior G2 output must live below the G1 run directory"
                ) from error
        elif not behavior_action_training:
            if (
                not isinstance(behavior_probe_evidence, Path)
                or behavior_probe_evidence_sha256 is None
                or behavior_predecessor_sha256 is not None
            ):
                raise ValueError("behavior G1 requires one immutable passed G0 evidence receipt")
            _require_sha256(
                "behavior causal-probe evidence sha256",
                behavior_probe_evidence_sha256,
            )
            expected_input_step = 0 if args.phase == "fresh" else 1
            if (
                common_behavior_conflict
                or args.phase not in {"fresh", "resume"}
                or args.load_global_step != expected_input_step
                or checkpoint_publication != "always"
                or tuple(args.gradient_audit_steps) != (1, 2)
                or args.lane_interleave_factor != 8
            ):
                raise ValueError(
                    "behavior G1 is restricted to one fresh step and one exact cold-resume "
                    "step with the frozen two-step gradient audit"
                )
    elif any(
        value is not None
        for value in (
            behavior_probe_output,
            behavior_probe_evidence,
            behavior_probe_evidence_sha256,
            behavior_factorial_output,
            behavior_factorial_evidence,
            behavior_factorial_evidence_sha256,
            behavior_predecessor_sha256,
        )
    ):
        raise ValueError("behavior causal evidence requires behavior-conditioned prediction")
    if representation_stage and (fixed_batch_arm is not None or relation_arm is not None):
        raise ValueError("representation training cannot execute disposable fixed-batch arms")
    if evaluation_requested and (fixed_batch_arm is not None or relation_arm is not None):
        raise ValueError("representation evaluation cannot execute disposable probe arms")
    validate_fsdp2_placement(args.fsdp2_placement)
    if (
        getattr(args, "task_relation_estimator", LOCAL_BALANCED_TASK_RELATION)
        == GLOBAL_MULTIPOSITIVE_TASK_RELATION
    ):
        raise ValueError(
            "host-native MATCH rejects global embedding retrieval as a rejected legacy interface"
        )
    if _ownership_estimator_from_args(args) not in OWNERSHIP_ESTIMATORS:
        raise ValueError("native full-objective ownership estimator is unsupported")
    if args.cuda_allocator not in CUDA_ALLOCATOR_MODES:
        raise ValueError("native full-objective CUDA allocator mode is unsupported")
    if fixed_batch_arm is not None and relation_arm is not None:
        raise ValueError("predictive and relation fixed-batch probes are mutually exclusive")
    if args.relation_supervision_layers and (
        args.relation_supervision_layers != FULL_RELATION_SUPERVISION_LAYERS
        or fixed_batch_arm is not None
        or relation_arm is not None
    ):
        raise ValueError(
            "shared relation depth supervision requires the preregistered "
            "8,17,26 production-training surfaces"
        )
    if fixed_batch_arm is None:
        if fixed_batch_curve_points != 0 or fixed_batch_output is not None:
            raise ValueError("fixed-batch curve points/output require one selected arm")
    else:
        if (
            fixed_batch_arm not in PREDICTIVE_FIXED_BATCH_ARMS
            or isinstance(fixed_batch_curve_points, bool)
            or not isinstance(fixed_batch_curve_points, int)
            or fixed_batch_curve_points < 2
        ):
            raise ValueError("fixed-batch arm requires at least two curve points")
        if fixed_batch_output is None:
            raise ValueError("fixed-batch arm requires an explicit output report")
        if args.phase != "fresh" or args.load_global_step != 0 or args.invocation_steps != 1:
            raise ValueError("fixed-batch arms must start from one fresh released checkpoint")
        if (
            args.authorization_manifest is not None
            or args.authorization_manifest_sha256 is not None
        ):
            raise ValueError("fixed-batch evidence cannot consume a training authorization")
        if args.visual_audit_every != 0:
            raise ValueError("fixed-batch evidence does not render or consume visual audits")
        if args.evidence_profile != "acceptance":
            raise ValueError("fixed-batch evidence cannot use the loss/visual trial profile")
        output = Path(fixed_batch_output)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"fixed-batch output already exists: {output}")
    if relation_arm is None:
        if (
            relation_curve_points != 0
            or relation_output is not None
            or relation_visual_root is not None
        ):
            raise ValueError(
                "relation fixed-batch curve points/output/visual root require one selected arm"
            )
        if relation_sample_step != 0:
            raise ValueError("relation fixed-batch sample step requires one selected arm")
    else:
        if (
            relation_arm not in RELATION_FIXED_BATCH_ARMS
            or isinstance(relation_curve_points, bool)
            or not isinstance(relation_curve_points, int)
            or relation_curve_points < 2
        ):
            raise ValueError("relation fixed-batch arm requires at least two curve points")
        if (
            relation_arm
            in {
                RELATION_BILINEAR_PROBE_ARM,
                RELATION_DEPTH_PROBE_ARM,
            }
            and relation_curve_points != RELATION_DEPTH_PROBE_CURVE_POINT_COUNT
        ):
            raise ValueError(
                "external relation probe requires exactly 41 preregistered curve points"
            )
        if (
            isinstance(relation_sample_step, bool)
            or not isinstance(relation_sample_step, int)
            or relation_sample_step < 0
            or relation_sample_step >= args.total_planned_steps
        ):
            raise ValueError("relation fixed-batch sample step is outside the frozen plan")
        if relation_output is None or relation_visual_root is None:
            raise ValueError("relation fixed-batch arm requires output and visual root paths")
        if args.phase != "fresh" or args.load_global_step != 0 or args.invocation_steps != 1:
            raise ValueError(
                "relation fixed-batch arms must start from one fresh released checkpoint"
            )
        if (
            args.authorization_manifest is not None
            or args.authorization_manifest_sha256 is not None
        ):
            raise ValueError("relation fixed-batch evidence cannot consume an authorization")
        if args.visual_audit_every != 0:
            raise ValueError("relation fixed-batch visuals use their dedicated immutable root")
        if args.evidence_profile != "acceptance":
            raise ValueError("relation fixed-batch evidence requires the acceptance profile")
        for path in (Path(relation_output), Path(relation_visual_root)):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"relation fixed-batch output already exists: {path}")
    integer_names = (
        "seed",
        "capacity",
        "maximum_control_tokens",
        "load_global_step",
        "invocation_steps",
        "total_planned_steps",
        "maximum_optimizer_lag",
        "lane_interleave_factor",
        "predictive_cache_memory_shards",
        "current_grid_cache_memory_shards",
        "visual_audit_every",
    )
    for name in integer_names:
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"native full-objective {name} must be a non-negative integer")
    for name in (
        "capacity",
        "maximum_control_tokens",
        "invocation_steps",
        "total_planned_steps",
        "lane_interleave_factor",
        "predictive_cache_memory_shards",
        "current_grid_cache_memory_shards",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"native full-objective {name} must be positive")
    if reset_mixture is None:
        if args.lane_interleave_factor > args.maximum_optimizer_lag:
            raise ValueError("lane interleave factor exceeds the frozen maximum optimizer lag")
    else:
        required_lag = adr121_required_optimizer_lag(args.lane_interleave_factor)
        if args.maximum_optimizer_lag != required_lag:
            raise ValueError(
                "reset-mixture maximum optimizer lag must equal twice the lane interleave factor"
            )
        legacy_adr121_contract = (
            args.evidence_profile == "acceptance"
            and representation_stage
            and reset_mixture == (1, 2)
            and args.lane_interleave_factor == 8
            and args.total_planned_steps == 200
            and args.phase == "fresh"
            and args.load_global_step == 0
            and args.invocation_steps == 200
            and checkpoint_publication == "never"
            and args.visual_audit_every == 1
        )
        medium_segment = (
            args.phase,
            args.load_global_step,
            args.invocation_steps,
            tuple(representation_evaluation_steps),
        )
        matched_medium_horizon_contract = (
            args.evidence_profile == MATCHED_MEDIUM_HORIZON_PROFILE
            and representation_stage
            and reset_mixture == (1, 2)
            and args.lane_interleave_factor == 8
            and args.total_planned_steps == MATCHED_MEDIUM_HORIZON_TOTAL_STEPS
            and checkpoint_publication == "always"
            and args.visual_audit_every == MATCHED_MEDIUM_HORIZON_VISUAL_CADENCE
            and medium_segment in REGISTERED_MATCHED_MEDIUM_HORIZON_SEGMENTS
            and warm_evaluation_requested
            and fixed_observation_requested
            and fixed_observation_evaluation_requested
            and evaluation_requested
            and baseline_requested
            and not intervention_requested
            and not behavior_conditioned
            and fixed_batch_arm is None
            and relation_arm is None
            and args.task_relation_estimator == HOST_NATIVE_FACTORIZED_TASK_RELATION
            and _ownership_estimator_from_args(args)
            in {TOKEN_MICRO_OWNERSHIP, TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP}
        )
        if not (legacy_adr121_contract or matched_medium_horizon_contract):
            raise ValueError(
                "reset mixture requires either the frozen ADR-121 200-update contract "
                "or an explicitly registered matched medium-horizon transaction"
            )
    if args.source_prediction_mode not in {"omitted_static", "current_grid"}:
        raise ValueError("native source prediction mode is unsupported")
    if args.predictive_weight is None or args.structural_weight is None:
        raise ValueError(
            "native predictive and structural family weights must be explicitly provided"
        )
    if args.gradient_audit_steps is None:
        raise ValueError("native family-gradient audit steps must be explicitly provided")
    if 1 in args.gradient_audit_steps and not behavior_conditioned:
        raise ValueError("family-gradient audit cannot run on the fresh state-bootstrap step")
    requested_global_step = args.load_global_step + args.invocation_steps
    if args.evidence_profile == "acceptance":
        if reset_mixture is None:
            first_correction_step = args.lane_interleave_factor + 1
            second_correction_step = 2 * args.lane_interleave_factor + 1
        else:
            first_correction_step, second_correction_step = adr121_recurrent_audit_updates(
                args.lane_interleave_factor
            )
        if (
            first_correction_step not in args.gradient_audit_steps
            or second_correction_step not in args.gradient_audit_steps
        ):
            raise ValueError(
                "native gradient audits require first/second recurrent correction steps "
                f"{first_correction_step}/{second_correction_step}"
            )
        if reset_mixture is not None and args.gradient_audit_steps != (
            18,
            34,
            50,
            100,
            200,
        ):
            raise ValueError("ADR-121 requires the exact 18,34,50,100,200 audit schedule")
    elif args.evidence_profile == MATCHED_MEDIUM_HORIZON_PROFILE:
        if tuple(args.gradient_audit_steps) != MATCHED_MEDIUM_HORIZON_AUDIT_STEPS:
            raise ValueError("ADR-135 requires the exact 18,34,50,100,200,500,1000 audit schedule")
    elif args.evidence_profile == "loss_visual_trial":
        if requested_global_step > 20:
            raise ValueError("loss/visual trial evidence cannot exceed global step 20")
        if (
            any(step <= requested_global_step for step in args.gradient_audit_steps)
            and not behavior_conditioned
        ):
            raise ValueError(
                "loss/visual trial family-gradient audits must lie after its saved step"
            )
    elif args.evidence_profile == "behavior_discrimination_trial":
        if (
            not behavior_action_training
            or args.phase != "fresh"
            or args.load_global_step != 0
            or requested_global_step != BEHAVIOR_ACTION_DISCRIMINATION_STEPS
            or tuple(args.gradient_audit_steps) != BEHAVIOR_ACTION_DISCRIMINATION_AUDIT_STEPS
            or args.visual_audit_every != 1
        ):
            raise ValueError(
                "behavior discrimination evidence requires the exact fresh 0-to-60 "
                "joint-action contract"
            )
    else:
        raise ValueError("native full-objective evidence profile is unsupported")
    physical_sidecar_manifest = (
        args.physical_sidecar_manifest
        if args.physical_sidecar_manifest is not None
        else (
            None
            if args.physical_sidecar_root is None
            else args.physical_sidecar_root / "manifest.json"
        )
    )
    required = {
        "checkpoint-dir": args.checkpoint_dir,
        "current-grid-cache-root": args.current_grid_cache_root,
        "current-grid-cache-build-report": args.current_grid_cache_build_report,
        "processor-dir": args.processor_dir,
        "dataset-split": args.dataset_split,
        "dataset-manifest": args.dataset_manifest,
        "norm-stats": args.norm_stats,
        "physical-sidecar-root": args.physical_sidecar_root,
        "physical-sidecar-manifest": physical_sidecar_manifest,
        "physical-visual-acceptance": args.physical_visual_acceptance,
        "predictive-cache-root": args.predictive_cache_root,
        "predictive-cache-build-report": args.predictive_cache_build_report,
        "predictive-target-audit": args.predictive_target_audit,
        "predictive-teacher-causality-audit": args.predictive_teacher_causality_audit,
        "predictive-temporal-audit": args.predictive_temporal_audit,
        "run-dir": args.run_dir,
    }
    if args.phase == "resume" and not representation_stage:
        required["authorization-manifest"] = args.authorization_manifest
    if representation_stage or behavior_action_training:
        required["representation-split"] = representation_split_file
    if behavior_probe_evidence is not None:
        required["behavior-causal-probe-evidence"] = behavior_probe_evidence
    if behavior_factorial_evidence is not None:
        required["behavior-posterior-control-probe-evidence"] = behavior_factorial_evidence
    if intervention_requested:
        required["representation-task-intervention-plan"] = representation_task_intervention_file
    if fixed_observation_requested:
        required["fixed-observation-pair-plan"] = fixed_observation_pair_plan_file
        required["fixed-observation-training-audit"] = fixed_observation_training_audit_file
        required["fixed-observation-evaluation-plan"] = fixed_observation_evaluation_plan_file
        required["fixed-observation-validation-audit"] = fixed_observation_validation_audit_file
        required["fixed-observation-heldout-audit"] = fixed_observation_heldout_audit_file
    if evaluation_requested:
        required["representation-evaluation-plan"] = representation_evaluation_plan_file
    if baseline_requested:
        required["representation-evaluation-baseline"] = representation_evaluation_baseline_file
    if warm_evaluation_requested:
        required["representation-warm-evaluation-plan"] = representation_warm_evaluation_plan_file
    absent = sorted(name for name, value in required.items() if value is None)
    if absent:
        raise ValueError(f"native full-objective paths are absent: {absent}")
    files = (
        args.patch,
        args.training_config,
        args.robot_config,
        args.data_config,
        args.dataset_manifest,
        args.norm_stats,
        physical_sidecar_manifest,
        args.physical_visual_acceptance,
        args.predictive_cache_build_report,
        args.predictive_target_audit,
        args.predictive_teacher_causality_audit,
        args.predictive_temporal_audit,
        args.current_grid_cache_build_report,
    )
    if args.authorization_manifest is not None:
        files += (args.authorization_manifest,)
    if representation_stage or behavior_action_training:
        files += (representation_split_file,)
    if behavior_probe_evidence is not None:
        files += (behavior_probe_evidence,)
    if behavior_factorial_evidence is not None:
        files += (behavior_factorial_evidence,)
    if intervention_requested:
        files += (representation_task_intervention_file,)
    if fixed_observation_requested:
        files += (
            fixed_observation_pair_plan_file,
            fixed_observation_training_audit_file,
            fixed_observation_evaluation_plan_file,
            fixed_observation_validation_audit_file,
            fixed_observation_heldout_audit_file,
        )
    if evaluation_requested:
        files += (representation_evaluation_plan_file,)
    if baseline_requested:
        files += (representation_evaluation_baseline_file,)
    if warm_evaluation_requested:
        files += (representation_warm_evaluation_plan_file,)
    if any(path is None or not Path(path).is_file() for path in files):
        raise FileNotFoundError("one or more full-objective source/config/data files are absent")
    directories = (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
        args.physical_sidecar_root,
        args.predictive_cache_root,
        args.current_grid_cache_root,
    )
    if any(path is None or not Path(path).is_dir() for path in directories):
        raise FileNotFoundError(
            "one or more full-objective source/model/data directories are absent"
        )
    _require_sha256(
        "physical sidecar manifest sha256",
        args.physical_sidecar_manifest_sha256,
    )
    _require_sha256(
        "physical visual acceptance sha256",
        args.physical_visual_acceptance_sha256,
    )
    _require_sha256(
        "predictive cache build-report sha256",
        args.predictive_cache_build_report_sha256,
    )
    _require_sha256(
        "predictive target audit sha256",
        args.predictive_target_audit_sha256,
    )
    _require_sha256(
        "predictive teacher-causality audit sha256",
        args.predictive_teacher_causality_audit_sha256,
    )
    _require_sha256(
        "predictive temporal audit sha256",
        args.predictive_temporal_audit_sha256,
    )
    _require_sha256(
        "current-grid cache build-report sha256",
        args.current_grid_cache_build_report_sha256,
    )
    if args.phase == "resume" and not representation_stage:
        _require_sha256(
            "training authorization manifest sha256",
            args.authorization_manifest_sha256,
        )
    elif not representation_stage and (
        args.authorization_manifest is not None or args.authorization_manifest_sha256 is not None
    ):
        raise ValueError("the fresh acceptance probe cannot consume a training authorization")
    if args.phase == "fresh" and args.load_global_step != 0:
        raise ValueError("fresh full-objective training must begin at global step zero")
    if args.phase == "resume" and args.load_global_step <= 0:
        raise ValueError("resumed full-objective training requires a positive load step")
    if checkpoint_publication == "never":
        requested_global_step = args.load_global_step + args.invocation_steps
        behavior_g2 = behavior_factorial_output is not None
        if not representation_stage or (
            not behavior_g2 and (args.phase != "fresh" or args.load_global_step != 0)
        ):
            raise ValueError(
                "disabled checkpoint publication is restricted to fresh representation "
                "evidence or the loaded-G1 zero-update G2 probe"
            )
        if (
            not behavior_conditioned
            and not behavior_g2
            and (
                not evaluation_requested
                or requested_global_step not in representation_evaluation_steps
            )
        ):
            raise ValueError(
                "disabled checkpoint publication requires immutable final representation evaluation"
            )
    bounded_behavior_action_trial = behavior_action_training and (
        (
            args.evidence_profile == "loss_visual_trial"
            and args.phase == "fresh"
            and args.load_global_step == 0
            and args.invocation_steps <= BEHAVIOR_ACTION_EVIDENCE_MAXIMUM_STEPS
            and args.visual_audit_every == 1
            and tuple(args.gradient_audit_steps) == (2, 10, 20)
        )
        or (
            args.evidence_profile == "behavior_discrimination_trial"
            and args.phase == "fresh"
            and args.load_global_step == 0
            and args.invocation_steps == BEHAVIOR_ACTION_DISCRIMINATION_STEPS
            and args.visual_audit_every == 1
            and tuple(args.gradient_audit_steps) == BEHAVIOR_ACTION_DISCRIMINATION_AUDIT_STEPS
        )
    )
    if (
        args.invocation_steps > 1
        and args.phase != "resume"
        and checkpoint_publication == "always"
        and not bounded_behavior_action_trial
    ):
        raise ValueError("multi-step training must resume from a passed one-step checkpoint")
    if args.load_global_step + args.invocation_steps > args.total_planned_steps:
        raise ValueError("full-objective invocation exceeds the frozen stream plan")
    if args.gradient_audit_steps[-1] > args.total_planned_steps:
        raise ValueError("family-gradient audit schedule exceeds the frozen stream plan")
    if args.invocation_steps > 1 and args.visual_audit_every <= 0:
        raise ValueError("multi-step training requires periodic current-model visual audit")
    if representation_stage or behavior_action_training:
        if representation_split_file is None:
            raise RuntimeError("behavior/representation split path vanished after validation")
        split = RepresentationTrialSplit.load(representation_split_file)
        if _sha256(representation_split_file) != representation_split_sha256:
            raise ValueError("behavior/representation split SHA-256 differs from its argument")
        if split.training_steps != args.total_planned_steps:
            raise ValueError("representation total planned steps differ from the source split")
        if split.comparison_id != FULL_COMPARISON_ID:
            raise ValueError("representation split uses another comparison identity")
        if (
            args.lane_interleave_factor > 1
            and split.schema != REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
        ):
            raise ValueError(
                "interleaved representation training requires a fixed reference evaluation bank"
            )
        if (
            args.lane_interleave_factor > 1
            and not evaluation_requested
            and not behavior_conditioned
        ):
            raise ValueError(
                "interleaved representation training requires checkpoint-boundary evaluation"
            )
        if args.lane_interleave_factor > 1 and not baseline_requested and not behavior_conditioned:
            raise ValueError(
                "interleaved representation training requires exact K1 step-zero replay"
            )
        if args.lane_interleave_factor == 1 and baseline_requested:
            raise ValueError("single-lane representation training cannot consume a K1 baseline")
    if intervention_requested:
        if representation_task_intervention_file is None:
            raise RuntimeError("representation task-intervention path vanished after validation")
        if (
            _sha256(representation_task_intervention_file)
            != representation_task_intervention_sha256
        ):
            raise ValueError("representation task-intervention file SHA-256 differs")
    if fixed_observation_requested:
        if (
            fixed_observation_pair_plan_file is None
            or fixed_observation_training_audit_file is None
            or fixed_observation_evaluation_plan_file is None
            or fixed_observation_validation_audit_file is None
            or fixed_observation_heldout_audit_file is None
        ):
            raise RuntimeError("fixed-observation paths vanished after validation")
        if _sha256(fixed_observation_pair_plan_file) != fixed_observation_pair_plan_sha256:
            raise ValueError("fixed-observation pair-plan file SHA-256 differs")
        if (
            _sha256(fixed_observation_training_audit_file)
            != fixed_observation_training_audit_sha256
        ):
            raise ValueError("fixed-observation training-audit file SHA-256 differs")
        if (
            _sha256(fixed_observation_evaluation_plan_file)
            != fixed_observation_evaluation_plan_sha256
        ):
            raise ValueError("fixed-observation evaluation-plan file SHA-256 differs")
        if (
            _sha256(fixed_observation_validation_audit_file)
            != fixed_observation_validation_audit_sha256
        ):
            raise ValueError("fixed-observation validation-audit file SHA-256 differs")
        if _sha256(fixed_observation_heldout_audit_file) != fixed_observation_heldout_audit_sha256:
            raise ValueError("fixed-observation heldout-audit file SHA-256 differs")
    if evaluation_requested:
        if representation_evaluation_plan_file is None:
            raise RuntimeError("representation evaluation plan vanished after validation")
        _require_sha256(
            "representation evaluation plan sha256",
            args.representation_evaluation_plan_sha256,
        )
        if (
            _sha256(representation_evaluation_plan_file)
            != args.representation_evaluation_plan_sha256
        ):
            raise ValueError("representation evaluation plan file digest differs")
        evaluation_boundaries = {
            args.load_global_step,
            args.load_global_step + args.invocation_steps,
        }
        if not set(representation_evaluation_steps) <= evaluation_boundaries:
            raise ValueError(
                "representation evaluation may run only at invocation checkpoint boundaries"
            )
        if args.lane_interleave_factor > 1:
            if args.phase == "fresh" and 0 not in representation_evaluation_steps:
                raise ValueError("fresh interleaved representation training must replay step zero")
            if representation_evaluation_baseline_file is None:
                raise RuntimeError("representation evaluation baseline path vanished")
            plan = RepresentationEvaluationPlan.load(representation_evaluation_plan_file)
            _require_sha256(
                "representation evaluation baseline sha256",
                args.representation_evaluation_baseline_sha256,
            )
            if (
                _sha256(representation_evaluation_baseline_file)
                != args.representation_evaluation_baseline_sha256
            ):
                raise ValueError("representation evaluation baseline file digest differs")
            baseline = load_representation_evaluation_baseline(
                representation_evaluation_baseline_file
            )
            validate_representation_baseline_plan(baseline, candidate_plan=plan)
            if args.phase == "resume":
                step_zero_root = args.run_dir / "representation_evaluations" / "global_step_0"
                snapshot_path = step_zero_root / "representation_evaluation_snapshot.json"
                replay_report_path = step_zero_root / "representation_baseline_replay_report.json"
                if not snapshot_path.is_file() or not replay_report_path.is_file():
                    raise FileNotFoundError(
                        "resumed interleaved representation training has no step-zero "
                        "baseline evidence"
                    )
                try:
                    snapshot = json.loads(snapshot_path.read_text(encoding="ascii"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "step-zero representation evaluation snapshot is invalid"
                    ) from error
                if not isinstance(snapshot, dict):
                    raise ValueError(
                        "step-zero representation evaluation snapshot must be a mapping"
                    )
                expected_report = build_representation_baseline_replay_report(
                    baseline=baseline,
                    candidate_snapshot=snapshot,
                    candidate_plan=plan,
                    candidate_visual_root=step_zero_root,
                )
                observed_report = load_representation_baseline_replay_report(replay_report_path)
                if observed_report != expected_report:
                    raise ValueError("step-zero representation baseline replay report changed")
    elif baseline_requested:
        raise ValueError("representation baseline requires checkpoint-boundary evaluation")
    if warm_evaluation_requested:
        if representation_warm_evaluation_plan_file is None:
            raise RuntimeError("representation warm-evaluation plan vanished after validation")
        _require_sha256(
            "representation warm-evaluation plan sha256",
            representation_warm_evaluation_plan_sha256,
        )
        if (
            _sha256(representation_warm_evaluation_plan_file)
            != representation_warm_evaluation_plan_sha256
        ):
            raise ValueError("representation warm-evaluation plan file digest differs")
        warm_plan = RepresentationEvaluationPlan.load(representation_warm_evaluation_plan_file)
        if (
            split is None
            or warm_plan.schema != REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA
            or warm_plan.history_transitions != 8
            or warm_plan.representation_split_sha256 != split.artifact_sha256
        ):
            raise ValueError("ADR-121 warm-evaluation plan differs from the frozen split")
    for name in (
        "learning_rate",
        "max_grad_norm",
        "maximum_peak_reserved_gib",
        "predictive_loss_power",
    ):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"native full-objective {name} must be finite")
        if value <= 0:
            raise ValueError(f"native full-objective {name} must be positive")
    for name in (
        "local_bptt_probability",
        "overshoot_probability",
        "source_mask_probability",
        "source_mask_token_fraction",
        "minimum_supervised_fraction",
    ):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"native full-objective {name} must lie in [0,1]")
    if args.source_mask_probability <= 0:
        raise ValueError(
            "native full-objective source_mask_probability must be positive; "
            "use the G0 runner for a targetless action-only probe"
        )
    for name in (
        "predictive_weight",
        "structural_weight",
        "support_weight",
        "existence_weight",
        "task_weight",
        "dense_task_weight",
        "ownership_weight",
        "predictive_term_weight",
        "current_grid_term_weight",
        "omitted_static_term_weight",
    ):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"native full-objective {name} must be finite and non-negative")
    for name in (
        "predictive_weight",
        "structural_weight",
        "existence_weight",
        "task_weight",
        "ownership_weight",
        "predictive_term_weight",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"native full-objective {name} must be positive")
    _validate_task_relation_dense_weight(
        estimator=args.task_relation_estimator,
        dense_task_weight=args.dense_task_weight,
        context="native full-objective",
    )
    if args.support_weight != 0:
        raise ValueError(
            "native full-objective CALVIN exclusive ownership requires support_weight=0"
        )
    if (
        not isinstance(args.relation_supervision_layers, tuple)
        or any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in args.relation_supervision_layers
        )
        or tuple(sorted(set(args.relation_supervision_layers))) != args.relation_supervision_layers
    ):
        raise ValueError("native full-objective relation supervision layers are malformed")
    selected_source_weight = (
        args.current_grid_term_weight
        if args.source_prediction_mode == "current_grid"
        else args.omitted_static_term_weight
    )
    if args.source_mask_probability > 0 and selected_source_weight <= 0:
        raise ValueError(
            "native full-objective training requires a positive-weight enabled source branch"
        )
    if (
        args.source_mask_probability > 0
        and args.source_prediction_mode == "current_grid"
        and args.source_mask_token_fraction <= 0
    ):
        raise ValueError("enabled source masking requires a positive token fraction")
    if fixed_batch_arm is None and relation_arm is None and behavior_probe_output is None:
        requested_global_step = args.load_global_step + args.invocation_steps
        report_path = args.run_dir / (
            f"native_representation_step_{requested_global_step}.json"
            if representation_stage
            else f"native_full_step_{requested_global_step}.json"
        )
        output_paths = [report_path]
        if checkpoint_publication == "always":
            output_paths.extend(
                (
                    args.run_dir / "checkpoints" / f"global_step_{requested_global_step}",
                    args.run_dir
                    / "checkpoints"
                    / f".global_step_{requested_global_step}.incomplete",
                )
            )
        if args.visual_audit_every > 0:
            output_paths.extend(
                args.run_dir / "visuals" / f"step_{step:08d}"
                for step in range(args.load_global_step + 1, requested_global_step + 1)
                if step % args.visual_audit_every == 0
            )
        if evaluation_requested:
            output_paths.extend(
                args.run_dir / "representation_evaluations" / f"global_step_{step}"
                for step in representation_evaluation_steps
            )
        if fixed_observation_evaluation_requested:
            output_paths.extend(
                args.run_dir / "fixed_observation_evaluations" / f"global_step_{step}"
                for step in representation_evaluation_steps
            )
        conflicts = tuple(path for path in output_paths if path.exists() or path.is_symlink())
        if conflicts:
            rendered = ", ".join(str(path) for path in conflicts)
            raise FileExistsError(
                f"native full invocation has pre-existing output artifacts: {rendered}"
            )


def load_predictive_build_report(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load one immutable cache-build report used as the training target recipe."""

    expected = _require_sha256("predictive build-report sha256", expected_sha256)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError("predictive cache build report differs from its expected digest")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("predictive cache build report is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != PREDICTIVE_BUILD_REPORT_FIELDS:
        raise ValueError("predictive cache build report fields differ from schema")
    for name in (
        "cache_manifest_sha256",
        "coverage_sha256",
        "pair_keys_sha256",
        "patch_sha256",
        "physical_visual_acceptance_sha256",
        "stream_plan_sha256",
        "teacher_encoder_digest",
        "temporal_estimator_sha256",
    ):
        _require_sha256(name, value[name])
    if (
        isinstance(value["expected_record_count"], bool)
        or not isinstance(value["expected_record_count"], int)
        or value["expected_record_count"] <= 0
    ):
        raise ValueError("predictive build report expected count must be positive")
    if not isinstance(value["output_root"], str) or not value["output_root"]:
        raise ValueError("predictive build report output root must be non-empty")
    return value


def load_predictive_target_audit(
    path: Path,
    *,
    expected_sha256: str,
    predictive_report: Mapping[str, Any],
    dataset_tree_sha256: str,
    physical_sidecar_manifest_sha256: str,
    query_schema_sha256: str,
    stream_plan_sha256: str,
    temporal_estimator_sha256: str,
) -> dict[str, Any]:
    """Load and recompute the fail-closed pretraining target-quality gate."""

    from picf_next.lingbot_native.predictive_cache import (
        LINGBOT_PREDICTIVE_EFFECTIVE_FPS,
    )
    from picf_next.lingbot_native.predictive_diagnostics import (
        PREDICTIVE_TARGET_AUDIT_SCHEMA,
        predictive_latent_diagnostics_from_mapping,
        predictive_target_pretraining_readiness,
        predictive_visible_support_diagnostics_from_mapping,
    )

    expected = _require_sha256("predictive target-audit sha256", expected_sha256)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError("predictive target audit differs from its expected digest")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("predictive target audit is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != PREDICTIVE_TARGET_AUDIT_FIELDS:
        raise ValueError("predictive target audit fields differ from schema")
    if value["schema"] != PREDICTIVE_TARGET_AUDIT_SCHEMA:
        raise ValueError("predictive target audit schema changed")
    for name in ("cache_manifest_sha256", "encoder_digest", "sample_selection_sha256"):
        _require_sha256(f"predictive target audit {name}", value[name])
    if (
        value["cache_manifest_sha256"] != predictive_report["cache_manifest_sha256"]
        or value["encoder_digest"] != predictive_report["teacher_encoder_digest"]
    ):
        raise ValueError("predictive target audit belongs to another cache or teacher")

    contract = value["cache_contract"]
    if not isinstance(contract, dict) or set(contract) != PREDICTIVE_TARGET_AUDIT_CONTRACT_FIELDS:
        raise ValueError("predictive target audit cache contract is malformed")
    expected_contract = {
        "coverage_sha256": predictive_report["coverage_sha256"],
        "dataset_tree_sha256": dataset_tree_sha256,
        "expected_record_count": predictive_report["expected_record_count"],
        "lingbot_checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "lingbot_source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "physical_sidecar_manifest_sha256": physical_sidecar_manifest_sha256,
        "query_schema_sha256": query_schema_sha256,
        "stream_plan_sha256": stream_plan_sha256,
        "temporal_estimator_sha256": temporal_estimator_sha256,
    }
    if any(
        contract.get(name) != expected_value for name, expected_value in expected_contract.items()
    ):
        raise ValueError("predictive target audit cache contract differs from training")
    if (
        contract.get("pair_keys_sha256") != predictive_report["pair_keys_sha256"]
        or contract.get("hidden_size") != 1024
        or contract.get("input_size") != 256
        or contract.get("patch_tokens") != 256
        or contract.get("route_id") != 0
        or contract.get("camera_name") != "static"
        or contract.get("attention_mode") != "flex_block_causal"
        or contract.get("use_warmup_frame") is not True
        or contract.get("source_fps") != 30.0
        or contract.get("effective_fps_semantics") != LINGBOT_PREDICTIVE_EFFECTIVE_FPS
    ):
        raise ValueError("predictive target audit teacher/cache geometry changed")

    count_names = (
        "maximum_samples",
        "sampled_target_count",
        "scanned_object_target_count",
        "scanned_record_count",
        "supported_object_target_count",
        "zero_support_object_target_count",
        "identity_count",
    )
    counts = {
        name: _positive_report_integer(f"predictive target audit {name}", value[name])
        for name in count_names
        if name != "zero_support_object_target_count"
    }
    zero_support = value["zero_support_object_target_count"]
    if isinstance(zero_support, bool) or not isinstance(zero_support, int) or zero_support < 0:
        raise ValueError("predictive target audit zero-support count is malformed")
    if (
        counts["scanned_record_count"] != predictive_report["expected_record_count"]
        or counts["sampled_target_count"] > counts["maximum_samples"]
        or counts["sampled_target_count"] > counts["supported_object_target_count"]
        or counts["identity_count"] > counts["supported_object_target_count"]
        or counts["scanned_object_target_count"]
        != counts["supported_object_target_count"] + zero_support
        or value["sample_selection"] != "lowest-sha256-priority-without-replacement/v1"
    ):
        raise ValueError("predictive target audit coverage or sample accounting differs")
    horizon_counts = value["horizon_record_counts"]
    horizons = contract["horizons"]
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
            for horizon in horizons
        )
        or horizons != sorted(set(horizons))
        or not isinstance(horizon_counts, dict)
        or set(horizon_counts) != {str(horizon) for horizon in horizons}
        or any(
            not isinstance(key, str)
            or not key.isdigit()
            or isinstance(amount, bool)
            or not isinstance(amount, int)
            or amount < 0
            for key, amount in horizon_counts.items()
        )
        or sum(horizon_counts.values()) != counts["scanned_record_count"]
    ):
        raise ValueError("predictive target audit horizon accounting differs")

    diagnostics = predictive_latent_diagnostics_from_mapping(value["diagnostics"])
    if diagnostics.valid_count != counts["sampled_target_count"]:
        raise ValueError("predictive target audit sample and diagnostics differ")
    visible_support = predictive_visible_support_diagnostics_from_mapping(
        value["visible_support_diagnostics"]
    )
    if (
        visible_support.supported_count != counts["supported_object_target_count"]
        or visible_support.sampled_count != counts["sampled_target_count"]
    ):
        raise ValueError("predictive visible-support diagnostics differ from audit coverage")
    ready, failures = predictive_target_pretraining_readiness(diagnostics)
    interpretation = value["interpretation"]
    if (
        not isinstance(interpretation, dict)
        or set(interpretation) != PREDICTIVE_TARGET_AUDIT_INTERPRETATION_FIELDS
        or interpretation["numerical_status"]
        != (
            "obvious_target_collapse"
            if diagnostics.obvious_numerical_collapse
            else "no_obvious_numerical_collapse"
        )
        or interpretation["pretraining_readiness"] != ("PASS" if ready else "FAIL")
        or interpretation["pretraining_readiness_failures"] != list(failures)
        or interpretation["retrieval_is_computable"] is not (diagnostics.retrieval_query_count > 0)
        or interpretation["scientific_acceptance"] is not False
        or interpretation["scientific_acceptance_reason"]
        != (
            "target statistics cannot establish source-conditioned learnability, "
            "shared-host gradient reach, object semantics or action benefit"
        )
    ):
        raise ValueError("predictive target audit interpretation is inconsistent")
    if not ready:
        raise ValueError(f"predictive target audit failed pretraining readiness: {failures}")
    return value


def load_predictive_temporal_audit(
    path: Path,
    *,
    expected_sha256: str,
    predictive_report: Mapping[str, Any],
    current_grid_report: Mapping[str, Any],
    physical_sidecar_manifest_sha256: str,
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Load and recompute the fail-closed current-to-future content gate."""

    from picf_next.lingbot_native.predictive_diagnostics import (
        PREDICTIVE_TEMPORAL_AUDIT_SCHEMA,
        PREDICTIVE_TEMPORAL_FEATURE_PAIRING,
        predictive_latent_diagnostics_from_mapping,
        predictive_target_pretraining_readiness,
        predictive_temporal_diagnostics_from_mapping,
        predictive_temporal_pretraining_readiness,
        predictive_visible_support_diagnostics_from_mapping,
    )

    expected = _require_sha256("predictive temporal-audit sha256", expected_sha256)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError("predictive temporal audit differs from its expected digest")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("predictive temporal audit is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != PREDICTIVE_TEMPORAL_AUDIT_FIELDS:
        raise ValueError("predictive temporal audit fields differ from schema")
    if value["schema"] != PREDICTIVE_TEMPORAL_AUDIT_SCHEMA:
        raise ValueError("predictive temporal audit schema changed")
    if value["feature_pairing"] != PREDICTIVE_TEMPORAL_FEATURE_PAIRING:
        raise ValueError("predictive temporal audit feature-pairing semantics changed")
    for name in (
        "current_cache_manifest_sha256",
        "current_encoder_digest",
        "future_cache_manifest_sha256",
        "future_encoder_digest",
        "physical_sidecar_manifest_sha256",
        "sample_selection_sha256",
        "current_correction_sample_selection_sha256",
    ):
        _require_sha256(f"predictive temporal audit {name}", value[name])
    if (
        value["current_cache_manifest_sha256"] != current_grid_report["cache_manifest_sha256"]
        or value["current_encoder_digest"] != current_grid_report["teacher_encoder_digest"]
        or value["future_cache_manifest_sha256"] != predictive_report["cache_manifest_sha256"]
        or value["future_encoder_digest"] != predictive_report["teacher_encoder_digest"]
        or value["physical_sidecar_manifest_sha256"] != physical_sidecar_manifest_sha256
    ):
        raise ValueError("predictive temporal audit belongs to another target cache")

    expected_horizons = tuple(horizons)
    if (
        not expected_horizons
        or expected_horizons != tuple(sorted(set(expected_horizons)))
        or any(
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
            for horizon in expected_horizons
        )
    ):
        raise ValueError("predictive temporal audit expected horizons are malformed")
    count_names = (
        "matched_future_record_count",
        "maximum_samples",
        "sampled_pair_count",
        "scanned_current_record_count",
        "supported_aligned_pair_count",
    )
    counts = {
        name: _positive_report_integer(f"predictive temporal audit {name}", value[name])
        for name in count_names
    }
    if (
        counts["maximum_samples"] < 2
        or counts["sampled_pair_count"] < 2
        or counts["sampled_pair_count"] > counts["maximum_samples"]
        or counts["sampled_pair_count"] > counts["supported_aligned_pair_count"]
        or counts["matched_future_record_count"] != predictive_report["expected_record_count"]
        or counts["matched_future_record_count"]
        > current_grid_report["expected_record_count"] * len(expected_horizons)
        or counts["scanned_current_record_count"] != current_grid_report["expected_record_count"]
        or value["sample_selection"] != "lowest-sha256-priority-without-replacement/v1"
    ):
        raise ValueError("predictive temporal audit coverage or sample accounting differs")

    current_count_names = (
        "current_correction_identity_count",
        "current_correction_sampled_target_count",
        "current_correction_scanned_object_target_count",
        "current_correction_supported_object_target_count",
    )
    current_counts = {
        name: _positive_report_integer(f"predictive temporal audit {name}", value[name])
        for name in current_count_names
    }
    current_zero = value["current_correction_zero_support_object_target_count"]
    if isinstance(current_zero, bool) or not isinstance(current_zero, int) or current_zero < 0:
        raise ValueError("predictive temporal audit current zero-support count is malformed")
    if (
        current_counts["current_correction_sampled_target_count"] < 2
        or current_counts["current_correction_sampled_target_count"] > counts["maximum_samples"]
        or current_counts["current_correction_sampled_target_count"]
        > current_counts["current_correction_supported_object_target_count"]
        or current_counts["current_correction_identity_count"]
        > current_counts["current_correction_supported_object_target_count"]
        or current_counts["current_correction_supported_object_target_count"] + current_zero
        != current_counts["current_correction_scanned_object_target_count"]
    ):
        raise ValueError("predictive temporal audit current-target accounting differs")
    current_diagnostics = predictive_latent_diagnostics_from_mapping(
        value["current_correction_diagnostics"]
    )
    current_support = predictive_visible_support_diagnostics_from_mapping(
        value["current_correction_visible_support_diagnostics"]
    )
    if (
        current_diagnostics.valid_count != current_counts["current_correction_sampled_target_count"]
        or current_diagnostics.identity_count > current_counts["current_correction_identity_count"]
        or current_support.sampled_count
        != current_counts["current_correction_sampled_target_count"]
        or current_support.supported_count
        != current_counts["current_correction_supported_object_target_count"]
    ):
        raise ValueError("predictive temporal audit current diagnostics differ from coverage")
    current_ready, current_failures = predictive_target_pretraining_readiness(current_diagnostics)
    horizon_counts = value["horizon_supported_pair_counts"]
    if (
        not isinstance(horizon_counts, dict)
        or set(horizon_counts) != {str(horizon) for horizon in expected_horizons}
        or any(
            isinstance(amount, bool) or not isinstance(amount, int) or amount < 0
            for amount in horizon_counts.values()
        )
        or sum(horizon_counts.values()) != counts["supported_aligned_pair_count"]
    ):
        raise ValueError("predictive temporal audit horizon accounting differs")

    diagnostics = predictive_temporal_diagnostics_from_mapping(value["diagnostics"])
    supported_horizon_count = sum(amount > 0 for amount in horizon_counts.values())
    if (
        diagnostics.pair_count != counts["sampled_pair_count"]
        or diagnostics.horizon_count > supported_horizon_count
    ):
        raise ValueError("predictive temporal audit sample and diagnostics differ")
    ready, failures = predictive_temporal_pretraining_readiness(diagnostics)
    aggregate_ready = current_ready and ready
    aggregate_failures = [
        *(f"current_correction:{failure}" for failure in current_failures),
        *(f"controlled_future:{failure}" for failure in failures),
    ]
    interpretation = value["interpretation"]
    if (
        not isinstance(interpretation, dict)
        or set(interpretation) != PREDICTIVE_TEMPORAL_AUDIT_INTERPRETATION_FIELDS
        or interpretation["controlled_future_temporal_pretraining_readiness"]
        != ("PASS" if ready else "FAIL")
        or interpretation["controlled_future_temporal_pretraining_readiness_failures"]
        != list(failures)
        or interpretation["current_correction_pretraining_readiness"]
        != ("PASS" if current_ready else "FAIL")
        or interpretation["current_correction_pretraining_readiness_failures"]
        != list(current_failures)
        or interpretation["pretraining_readiness"] != ("PASS" if aggregate_ready else "FAIL")
        or interpretation["pretraining_readiness_failures"] != aggregate_failures
        or interpretation["scientific_acceptance"] is not False
        or interpretation["scientific_acceptance_reason"]
        != (
            "target-bank statistics do not establish source-conditioned prediction, "
            "action conditioning or action benefit"
        )
    ):
        raise ValueError("predictive temporal audit interpretation is inconsistent")
    if not aggregate_ready:
        raise ValueError(
            f"predictive temporal audit failed pretraining readiness: {aggregate_failures}"
        )
    return value


def load_predictive_teacher_causality_audit(
    path: Path,
    *,
    expected_sha256: str,
    predictive_report: Mapping[str, Any],
    current_grid_report: Mapping[str, Any],
    dataset_tree_sha256: str,
    physical_sidecar_manifest_sha256: str,
    patch_sha256: str,
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Load the released-teacher causal-isolation and cache-replay gate."""

    from picf_next.lingbot_native.predictive_diagnostics import (
        TEACHER_CAUSALITY_AUDIT_SCHEMA,
        predictive_temporal_diagnostics_from_mapping,
        predictive_temporal_pretraining_readiness,
    )

    expected = _require_sha256("predictive teacher-causality audit sha256", expected_sha256)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError("predictive teacher-causality audit differs from its expected digest")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("predictive teacher-causality audit is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != TEACHER_CAUSALITY_AUDIT_FIELDS:
        raise ValueError("predictive teacher-causality audit fields differ from schema")
    if value["schema"] != TEACHER_CAUSALITY_AUDIT_SCHEMA:
        raise ValueError("predictive teacher-causality audit schema changed")
    for name in (
        "current_cache_manifest_sha256",
        "current_encoder_digest",
        "dataset_tree_sha256",
        "patch_sha256",
        "physical_sidecar_manifest_sha256",
        "predictive_cache_manifest_sha256",
        "predictive_encoder_digest",
    ):
        _require_sha256(f"predictive teacher-causality audit {name}", value[name])
    if (
        value["current_cache_manifest_sha256"] != current_grid_report["cache_manifest_sha256"]
        or value["current_encoder_digest"] != current_grid_report["teacher_encoder_digest"]
        or value["dataset_tree_sha256"] != dataset_tree_sha256
        or value["patch_sha256"] != patch_sha256
        or value["physical_sidecar_manifest_sha256"] != physical_sidecar_manifest_sha256
        or value["predictive_cache_manifest_sha256"] != predictive_report["cache_manifest_sha256"]
        or value["predictive_encoder_digest"] != predictive_report["teacher_encoder_digest"]
    ):
        raise ValueError("predictive teacher-causality audit belongs to another artifact set")
    scanned = _positive_report_integer(
        "predictive teacher-causality scanned record count",
        value["scanned_record_count"],
    )
    if scanned != predictive_report["expected_record_count"]:
        raise ValueError("predictive teacher-causality audit did not scan complete coverage")

    diagnostics = value["diagnostics"]
    if not isinstance(diagnostics, dict) or set(diagnostics) != TEACHER_CAUSALITY_DIAGNOSTIC_FIELDS:
        raise ValueError("predictive teacher-causality diagnostics differ from schema")
    _require_sha256(
        "predictive teacher-causality sample selection",
        diagnostics["sample_selection_sha256"],
    )
    sampled = _positive_report_integer(
        "predictive teacher-causality sampled record count",
        diagnostics["sampled_record_count"],
    )
    if sampled < 2 or sampled > scanned:
        raise ValueError("predictive teacher-causality sample coverage is malformed")
    expected_horizons = tuple(horizons)
    horizon_counts = diagnostics["sampled_horizon_record_counts"]
    if (
        not expected_horizons
        or expected_horizons != tuple(sorted(set(expected_horizons)))
        or any(
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
            for horizon in expected_horizons
        )
        or not isinstance(horizon_counts, dict)
        or set(horizon_counts) != {str(horizon) for horizon in expected_horizons}
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in horizon_counts.values()
        )
        or sum(horizon_counts.values()) != sampled
    ):
        raise ValueError("predictive teacher-causality horizon coverage is malformed")
    temporal = predictive_temporal_diagnostics_from_mapping(
        diagnostics["same_call_temporal_diagnostics"]
    )
    supported = _positive_report_integer(
        "predictive teacher-causality same-call supported pair count",
        diagnostics["same_call_supported_pair_count"],
    )
    if supported != temporal.pair_count:
        raise ValueError("predictive teacher-causality temporal support count differs")
    temporal_ready, temporal_failures = predictive_temporal_pretraining_readiness(temporal)
    if (
        diagnostics["same_call_temporal_pretraining_readiness"]
        != ("PASS" if temporal_ready else "FAIL")
        or diagnostics["same_call_temporal_pretraining_readiness_failures"]
        != list(temporal_failures)
        or not temporal_ready
    ):
        raise ValueError("predictive teacher-causality temporal readiness did not pass")
    for name in (
        "current_cache_patch_elements",
        "current_patch_elements",
        "future_feature_elements",
        "future_importance_elements",
    ):
        _positive_report_integer(f"predictive teacher-causality {name}", diagnostics[name])
    for name in (
        "current_cache_patch_mismatch_count",
        "current_patch_mismatch_count",
        "future_feature_mismatch_count",
        "future_importance_mismatch_count",
    ):
        if isinstance(diagnostics[name], bool) or not isinstance(diagnostics[name], int):
            raise ValueError(f"predictive teacher-causality {name} is malformed")
        if diagnostics[name] != 0:
            raise ValueError(f"predictive teacher-causality {name} is nonzero")
    for name in (
        "maximum_current_cache_patch_absolute_error",
        "maximum_current_patch_absolute_error",
        "maximum_future_feature_absolute_error",
        "maximum_future_importance_absolute_error",
    ):
        error = diagnostics[name]
        if (
            isinstance(error, bool)
            or not isinstance(error, (int, float))
            or not math.isfinite(error)
            or error != 0.0
        ):
            raise ValueError(f"predictive teacher-causality {name} is nonzero or malformed")
    if diagnostics["status"] != "PASS":
        raise ValueError("predictive teacher-causality audit did not pass")
    return value


def load_current_grid_build_report(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load one immutable current-grid cache recipe."""

    expected = _require_sha256("current-grid build-report sha256", expected_sha256)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError("current-grid cache build report differs from its expected digest")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("current-grid cache build report is not valid JSON") from error
    if not isinstance(value, dict) or set(value) not in {
        CURRENT_GRID_BUILD_REPORT_FIELDS,
        CURRENT_GRID_BUILD_REPORT_FIELDS | {"content_identical_donor"},
    }:
        raise ValueError("current-grid cache build report fields differ from schema")
    for name in (
        "cache_manifest_sha256",
        "coverage_sha256",
        "patch_sha256",
        "physical_visual_acceptance_sha256",
        "source_keys_sha256",
        "stream_plan_sha256",
        "teacher_encoder_digest",
        "temporal_estimator_sha256",
    ):
        _require_sha256(name, value[name])
    if (
        isinstance(value["expected_record_count"], bool)
        or not isinstance(value["expected_record_count"], int)
        or value["expected_record_count"] <= 0
    ):
        raise ValueError("current-grid build report expected count must be positive")
    if not isinstance(value["output_root"], str) or not value["output_root"]:
        raise ValueError("current-grid build report output root must be non-empty")
    donor = value.get("content_identical_donor")
    if donor is not None:
        if not isinstance(donor, dict) or set(donor) != CURRENT_GRID_CONTENT_IDENTICAL_DONOR_FIELDS:
            raise ValueError("current-grid content-identical donor fields differ from schema")
        for name in CURRENT_GRID_CONTENT_IDENTICAL_DONOR_FIELDS - {"reused_record_count"}:
            _require_sha256(name, donor[name])
        reused = donor["reused_record_count"]
        if (
            isinstance(reused, bool)
            or not isinstance(reused, int)
            or not 0 < reused <= value["expected_record_count"]
        ):
            raise ValueError("current-grid donor reused record count is outside coverage")
    return value


def _cache_producer_patch_sha256(
    predictive_report: Mapping[str, Any],
    current_grid_report: Mapping[str, Any],
) -> str:
    """Bind both target banks to one producer without coupling them to the consumer."""

    predictive_patch = _require_sha256(
        "predictive cache producer patch",
        predictive_report.get("patch_sha256"),
    )
    current_grid_patch = _require_sha256(
        "current-grid cache producer patch",
        current_grid_report.get("patch_sha256"),
    )
    if predictive_patch != current_grid_patch:
        raise ValueError("current and future target caches use different producer patches")
    return predictive_patch


def load_training_gate_decision(
    path: Path,
    *,
    expected_gate: str,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Load one immutable, evidence-bound empirical or protocol decision."""

    expected_kind = TRAINING_GATE_DECISION_KINDS.get(expected_gate)
    if expected_kind is None:
        raise ValueError("training gate is not part of the frozen acceptance ladder")
    if path.is_symlink() or not path.is_file():
        raise ValueError("training gate decision must be one real file")
    payload = path.read_bytes()
    if expected_sha256 is not None:
        digest = _require_sha256("training gate decision sha256", expected_sha256)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("training gate decision differs from its expected digest")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training gate decision is not valid ASCII JSON") from error
    fields = {
        "schema",
        "status",
        "gate",
        "decision_kind",
        "subject",
        "reviewer",
        "criteria",
        "evidence",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("training gate decision fields differ from schema")
    if (
        value["schema"] != TRAINING_GATE_DECISION_SCHEMA
        or value["status"] != "PASS"
        or value["gate"] != expected_gate
        or value["decision_kind"] != expected_kind
    ):
        raise ValueError("training gate decision does not pass the expected gate")
    reviewer = value["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("training gate decision requires an explicit reviewer")

    criteria = value["criteria"]
    if not isinstance(criteria, dict) or set(criteria) != {"path", "sha256"}:
        raise ValueError("training gate criteria reference is malformed")
    criteria_path = Path(criteria["path"]) if isinstance(criteria["path"], str) else None
    criteria_digest = _require_sha256("training gate criteria sha256", criteria["sha256"])
    if (
        criteria_path is None
        or not criteria_path.is_absolute()
        or criteria_path.is_symlink()
        or not criteria_path.is_file()
        or _sha256(criteria_path) != criteria_digest
    ):
        raise ValueError("training gate criteria differ from the reviewed contract")

    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("training gate decision requires at least one evidence report")
    expected_evidence = TRAINING_GATE_EVIDENCE_SCHEMAS[expected_gate]
    if tuple(report.get("name") for report in evidence if isinstance(report, dict)) != tuple(
        name for name, _schema in expected_evidence
    ):
        raise ValueError("training gate evidence coverage or order differs from schema")
    observed_names: set[str] = set()
    observed_paths = {criteria_path.resolve()}
    validated_evidence: list[tuple[str, bytes, dict[str, Any]]] = []
    for report in evidence:
        if not isinstance(report, dict) or set(report) != {"name", "path", "sha256"}:
            raise ValueError("training gate evidence reference is malformed")
        name = report["name"]
        report_path = Path(report["path"]) if isinstance(report["path"], str) else None
        report_digest = _require_sha256("training gate evidence sha256", report["sha256"])
        if not isinstance(name, str) or not name.strip() or name in observed_names:
            raise ValueError("training gate evidence names must be nonempty and distinct")
        if report_path is None or not report_path.is_absolute():
            raise ValueError("training gate evidence paths must be absolute")
        resolved = report_path.resolve()
        if (
            report_path.is_symlink()
            or not report_path.is_file()
            or resolved in observed_paths
            or _sha256(report_path) != report_digest
        ):
            raise ValueError("training gate evidence differs from the reviewed report")
        report_payload = report_path.read_bytes()
        try:
            report_value = json.loads(report_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("training gate evidence is not valid JSON") from error
        validate_training_gate_evidence(
            gate=expected_gate,
            name=name,
            value=report_value,
        )
        validated_evidence.append((name, report_payload, report_value))
        observed_names.add(name)
        observed_paths.add(resolved)
    expected_subject = training_gate_decision_subject(
        gate=expected_gate,
        criteria_sha256=criteria_digest,
        evidence=tuple(validated_evidence),
    )
    if value["subject"] != expected_subject:
        raise ValueError("training gate decision subject differs from its evidence")
    return payload, value


def validate_training_authorization(
    path: Path,
    *,
    expected_sha256: str,
    input_global_step: int,
    requested_global_step: int,
    total_planned_steps: int,
    visual_audit_every: int,
    execution_contract_sha256: str,
    implementation_sha256: str,
    model_family_sha256: str,
    expected_fsdp2_placement: str = FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    expected_cuda_allocator: str = "native",
    expected_predictive_objective: str | None = None,
    expected_predictive_visible_support_weighting: str | None = None,
    expected_predictive_minimum_visible_fraction: float | None = None,
) -> dict[str, Any]:
    """Validate a human-reviewed, hash-bound pilot or long-run gate."""

    placement = validate_fsdp2_placement(expected_fsdp2_placement)
    if expected_cuda_allocator not in CUDA_ALLOCATOR_MODES:
        raise ValueError("training authorization CUDA allocator expectation is unsupported")
    expected = _require_sha256("training authorization sha256", expected_sha256)
    if path.is_symlink() or not path.is_file():
        raise ValueError("training authorization differs from its expected digest")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("training authorization differs from its expected digest")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training authorization is not valid ASCII JSON") from error
    required = {
        "schema",
        "status",
        "stage",
        "input_global_step",
        "maximum_global_step",
        "visual_audit_every",
        "execution_contract_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "predictive_objective",
        "predictive_claim_scope",
        "predictive_visible_support_weighting",
        "predictive_minimum_visible_fraction_hex",
        "acceptance_subject",
        "input_full_report_sha256",
        "input_full_report",
        "prerequisite_reports",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("training authorization fields differ from schema")
    stage = value["stage"]
    if (
        value["schema"] != TRAINING_AUTHORIZATION_SCHEMA
        or value["status"] != "PASS"
        or stage not in TRAINING_AUTHORIZATION_GATES
    ):
        raise ValueError("training authorization decision or stage is not accepted")
    if value["input_global_step"] != input_global_step:
        raise ValueError("training authorization targets another input checkpoint")
    maximum = value["maximum_global_step"]
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < requested_global_step
        or maximum > total_planned_steps
    ):
        raise ValueError("training authorization does not cover the requested step range")
    if stage == "pilot" and maximum > 200:
        raise ValueError("pilot authorization cannot exceed 200 optimizer steps")
    if value["visual_audit_every"] != visual_audit_every or visual_audit_every <= 0:
        raise ValueError("training authorization and visual audit cadence differ")
    expected_digests = {
        "execution_contract_sha256": execution_contract_sha256,
        "implementation_sha256": implementation_sha256,
        "model_family_sha256": model_family_sha256,
    }
    for name, digest in expected_digests.items():
        _require_sha256(name, digest)
        if value[name] != digest:
            raise ValueError("training authorization targets another implementation or model")

    input_report = value["input_full_report"]
    if not isinstance(input_report, dict) or set(input_report) != {"path", "sha256"}:
        raise ValueError("training authorization input report is malformed")
    input_path = Path(input_report["path"]) if isinstance(input_report["path"], str) else None
    input_digest = _require_sha256("training input report sha256", input_report["sha256"])
    if value["input_full_report_sha256"] != input_digest:
        raise ValueError("training authorization subject digest differs from its input report")
    if input_path is None or input_path.is_symlink() or not input_path.is_file():
        raise ValueError("training authorization input report differs")
    input_payload = input_path.read_bytes()
    if hashlib.sha256(input_payload).hexdigest() != input_digest:
        raise ValueError("training authorization input report differs")
    try:
        input_decision = json.loads(input_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training authorization input report is not valid JSON") from error
    require_initial_probe = pilot_authorization_requires_initial_probe(
        stage=stage,
        input_report=input_decision,
        maximum_global_step=maximum,
        visual_audit_every=visual_audit_every,
    )
    validate_full_objective_report(
        input_decision,
        expected_saved_global_step=input_global_step,
        expected_digests=expected_digests,
        require_initial_probe=require_initial_probe,
        require_mature_wrong_time=stage == "long",
        require_source_evidence=stage == "long",
        expected_fsdp2_placement=placement,
        expected_cuda_allocator=expected_cuda_allocator,
    )
    acceptance_subject = training_authorization_acceptance_subject(
        stage=stage,
        input_report=input_decision,
        input_report_sha256=input_digest,
    )
    if value["acceptance_subject"] != acceptance_subject:
        raise ValueError("training authorization uses another accepted checkpoint subject")
    if stage == "long" and acceptance_subject is None:
        raise ValueError("long training authorization lacks an evaluated checkpoint subject")

    reports = value["prerequisite_reports"]
    required_gates = TRAINING_AUTHORIZATION_GATES[stage]
    if not isinstance(reports, list) or len(reports) != len(required_gates):
        raise ValueError("training authorization prerequisite coverage differs")
    observed: list[str] = []
    observed_paths = {input_path.expanduser().resolve()}
    owner_decision: PredictiveObjectiveDecision | None = None
    for report in reports:
        if not isinstance(report, dict) or set(report) != {"gate", "path", "sha256"}:
            raise ValueError("training authorization prerequisite entry is malformed")
        gate = report["gate"]
        report_path = Path(report["path"]) if isinstance(report["path"], str) else None
        digest = _require_sha256("training prerequisite sha256", report["sha256"])
        if gate not in required_gates or gate in observed or report_path is None:
            raise ValueError("training authorization prerequisite gate is invalid")
        resolved_report = report_path.expanduser().resolve()
        if (
            report_path.is_symlink()
            or not report_path.is_file()
            or resolved_report in observed_paths
        ):
            raise ValueError("training authorization prerequisite report differs")
        _gate_payload, gate_decision = load_training_gate_decision(
            report_path,
            expected_gate=gate,
            expected_sha256=digest,
        )
        if gate == "G2_PROTOCOL":
            owner_decision = predictive_objective_decision_from_gate_decision(
                gate_decision,
                expected_temporal_objective=expected_predictive_objective,
                expected_visible_support_weighting=(expected_predictive_visible_support_weighting),
                expected_minimum_visible_fraction=(expected_predictive_minimum_visible_fraction),
            )
        if stage == "long" and gate in {"G2", "G3", "G4", "G5", "G6", "G7_PROTOCOL"}:
            subject = gate_decision["subject"]
            if acceptance_subject is None:
                raise RuntimeError("long training acceptance subject was unexpectedly absent")
            if any(subject.get(name) != expected for name, expected in acceptance_subject.items()):
                raise ValueError(f"{gate} decision targets another evaluated checkpoint")
        observed.append(gate)
        observed_paths.add(resolved_report)
    if tuple(observed) != required_gates:
        raise ValueError("training authorization prerequisite order differs from the frozen ladder")
    if owner_decision is None:
        raise ValueError("training authorization lacks a predictive owner decision")
    expected_owner_fields = {
        "predictive_objective": owner_decision.temporal_objective,
        "predictive_claim_scope": owner_decision.claim_scope,
        "predictive_visible_support_weighting": owner_decision.visible_support_weighting,
        "predictive_minimum_visible_fraction_hex": (owner_decision.minimum_visible_fraction.hex()),
    }
    if any(value.get(name) != expected for name, expected in expected_owner_fields.items()):
        raise ValueError("training authorization predictive semantics differ from owner decision")
    return value


def _full_implementation_paths(root: Path) -> tuple[Path, ...]:
    """Resolve the transitive local Python closure imported by this entrypoint."""

    root = root.resolve()
    pending = [
        root / "tools/audit_lingbot_dino_teacher_causality.py",
        root / "tools/audit_lingbot_predictive_targets.py",
        root / "tools/audit_lingbot_predictive_temporal_targets.py",
        root / "tools/build_lingbot_calvin_current_grid_cache.py",
        root / "tools/build_lingbot_calvin_predictive_cache.py",
        root / "tools/build_lingbot_fixed_observation_evaluation_plan.py",
        root / "tools/build_lingbot_fixed_observation_pair_plan.py",
        root / "tools/build_lingbot_representation_split.py",
        root / "tools/build_lingbot_representation_task_intervention.py",
        root / "tools/run_lingbot_vla2_native_full.py",
    ]
    resolved: set[Path] = {root / "references/patches/lingbot_vla2_picf_native.patch"}
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"native full implementation file is absent: {path}")
        resolved.add(path)
        for module in _local_import_modules(root, path):
            for imported in _resolve_local_module(root, module):
                if imported not in resolved:
                    pending.append(imported)
    return tuple(sorted(resolved))


def _full_implementation_digest(root: Path) -> str:
    payload = {
        str(path.relative_to(root)): _sha256(path) for path in _full_implementation_paths(root)
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def _bind_complete_prompt_tokenization_audit(
    run_dir: Path,
    audit: CompletePromptTokenizationAudit,
) -> Path:
    """Publish once, or prove an existing run-local prompt contract is identical."""

    if not isinstance(audit, CompletePromptTokenizationAudit):
        raise TypeError("prompt tokenization audit has the wrong type")
    run_dir = Path(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("prompt tokenization audit requires one real run directory")
    output = run_dir / COMPLETE_PROMPT_TOKENIZATION_FILENAME
    if output.is_symlink():
        raise ValueError("prompt tokenization audit cannot be a symlink")
    if output.exists():
        if not output.is_file():
            raise ValueError("prompt tokenization audit path is not a regular file")
        if CompletePromptTokenizationAudit.load(output) != audit:
            raise RuntimeError("run-local complete prompt tokenization contract changed")
        return output
    _write_text_durable(
        output,
        json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n",
    )
    return output


def _validate_vlm_fsdp2_topology(policy: Any) -> dict[str, object]:
    topology = getattr(policy, "_lingbot_vlm_fsdp2_topology", None)
    if not isinstance(topology, dict) or set(topology) != {"text", "vision"}:
        raise RuntimeError("LingBot FSDP2 topology report is absent or malformed")
    expected_classes = {
        "text": {"Qwen2_5_VLDecoderLayer", "Qwen3VLTextDecoderLayer"},
        "vision": {"Qwen2_5_VLVisionBlock", "Qwen3VLVisionBlock"},
    }
    normalized: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for kind in ("text", "vision"):
        paths = topology[kind]
        if not isinstance(paths, tuple) or not paths:
            raise RuntimeError(f"LingBot FSDP2 topology has no {kind} blocks")
        if any(not isinstance(path, str) or not path for path in paths):
            raise RuntimeError(f"LingBot FSDP2 topology has malformed {kind} paths")
        if len(set(paths)) != len(paths) or seen.intersection(paths):
            raise RuntimeError("LingBot FSDP2 topology contains duplicate block paths")
        normalized[kind] = paths
        seen.update(paths)
    for kind in ("text", "vision"):
        for path in normalized[kind]:
            module = policy.get_submodule(path)
            base_class_names = {base.__name__ for base in type(module).__mro__}
            if not base_class_names.intersection(expected_classes[kind]):
                raise RuntimeError(
                    f"LingBot FSDP2 {kind} path {path} resolves to an unexpected class"
                )
            if not hasattr(module, "reshard") or not hasattr(module, "unshard"):
                raise RuntimeError(f"LingBot FSDP2 {kind} block {path} is not sharded")
            if kind == "text":
                nested = [
                    f"{path}.{relative_path}"
                    for relative_path, child in module.named_modules()
                    if relative_path and hasattr(child, "reshard") and hasattr(child, "unshard")
                ]
                if nested:
                    raise RuntimeError(
                        f"LingBot text decoder block contains nested FSDP2 modules: {nested}"
                    )
    return {
        "text_block_count": len(normalized["text"]),
        "text_block_paths": list(normalized["text"]),
        "vision_block_count": len(normalized["vision"]),
        "vision_block_paths": list(normalized["vision"]),
    }


def _validate_action_fsdp2_topology(policy: Any) -> dict[str, object]:
    try:
        layers = policy.model.qwenvl_with_expert.qwen_expert.model.layers
    except AttributeError as error:
        raise RuntimeError("LingBot action decoder layers are absent") from error
    try:
        blocks = tuple(layers)
    except TypeError as error:
        raise RuntimeError("LingBot action decoder layers are not iterable") from error
    if not blocks:
        raise RuntimeError("LingBot action decoder has no blocks")

    paths: list[str] = []
    block_bf16_bytes: list[int] = []
    for index, block in enumerate(blocks):
        path = f"model.qwenvl_with_expert.qwen_expert.model.layers.{index}"
        base_class_names = {base.__name__ for base in type(block).__mro__}
        if "Qwen2DecoderLayer" not in base_class_names:
            raise RuntimeError(f"LingBot action FSDP2 path {path} has an unexpected class")
        if not hasattr(block, "reshard") or not hasattr(block, "unshard"):
            raise RuntimeError(f"LingBot action FSDP2 block {path} is not sharded")
        nested = [
            f"{path}.{relative_path}"
            for relative_path, child in block.named_modules()
            if relative_path and hasattr(child, "reshard") and hasattr(child, "unshard")
        ]
        if nested:
            raise RuntimeError(
                f"LingBot action decoder block contains nested FSDP2 modules: {nested}"
            )
        numel = sum(int(parameter.numel()) for parameter in block.parameters())
        if numel <= 0:
            raise RuntimeError(f"LingBot action FSDP2 block {path} has no parameters")
        paths.append(path)
        block_bf16_bytes.append(numel * 2)
    return {
        "schema": "picf-next.lingbot-action-block-fsdp2.v1",
        "block_count": len(paths),
        "block_paths": paths,
        "maximum_block_bf16_bytes_upper_bound": max(block_bf16_bytes),
    }


def _execution_contract_digest(
    *,
    root: Path,
    args: argparse.Namespace,
    patched_source_sha256: Mapping[str, str],
    predictive_report: Mapping[str, Any],
    current_grid_report: Mapping[str, Any],
    query_schema_sha256: Mapping[str, str],
    temporal_metadata: Mapping[str, object],
    optimizer_contract: Mapping[str, Any],
    behavior_graph_sha256: str | None = None,
) -> tuple[str, str]:
    required_query_schemas = {
        "controlled_future_rollout",
        "current_correction",
        "current_random_grid",
        "omitted_static",
    }
    if set(query_schema_sha256) != required_query_schemas:
        raise RuntimeError("full execution query schemas differ from the frozen contract")
    frozen_query_schemas = {
        name: _require_sha256(f"{name} query schema", value)
        for name, value in sorted(query_schema_sha256.items())
    }
    implementation_sha256 = _full_implementation_digest(root)
    training_stage = getattr(args, "training_stage", FULL_ACTION_STAGE)
    representation_split_path = getattr(args, "representation_split", None)
    representation_task_intervention_path = getattr(
        args,
        "representation_task_intervention_plan",
        None,
    )
    fixed_observation_pair_plan_path = getattr(
        args,
        "fixed_observation_pair_plan",
        None,
    )
    fixed_observation_training_audit_path = getattr(
        args,
        "fixed_observation_training_audit",
        None,
    )
    fixed_observation_evaluation_plan_path = getattr(
        args,
        "fixed_observation_evaluation_plan",
        None,
    )
    fixed_observation_validation_audit_path = getattr(
        args,
        "fixed_observation_validation_audit",
        None,
    )
    fixed_observation_heldout_audit_path = getattr(
        args,
        "fixed_observation_heldout_audit",
        None,
    )
    representation_warm_evaluation_path = getattr(
        args,
        "representation_warm_evaluation_plan",
        None,
    )
    behavior_probe_evidence_path = getattr(args, "behavior_causal_probe_evidence", None)
    behavior_probe_evidence_sha256 = getattr(
        args,
        "behavior_causal_probe_evidence_sha256",
        None,
    )
    behavior_factorial_evidence_path = getattr(
        args,
        "behavior_posterior_control_probe_evidence",
        None,
    )
    behavior_factorial_evidence_sha256 = getattr(
        args,
        "behavior_posterior_control_probe_evidence_sha256",
        None,
    )
    reset_mixture = reset_mixture_values(args)
    payload = {
        "schema": "picf-next.lingbot-vla2-native-full-execution.v8",
        "implementation_sha256": implementation_sha256,
        "training_stage": training_stage,
        "input_file_sha256": {
            "data_config": _sha256(args.data_config),
            "dataset_manifest": _sha256(args.dataset_manifest),
            "norm_stats": _sha256(args.norm_stats),
            "physical_sidecar_manifest": args.physical_sidecar_manifest_sha256,
            "physical_visual_acceptance": args.physical_visual_acceptance_sha256,
            "predictive_build_report": args.predictive_cache_build_report_sha256,
            "predictive_teacher_causality_audit": (args.predictive_teacher_causality_audit_sha256),
            "predictive_target_audit": args.predictive_target_audit_sha256,
            "predictive_temporal_audit": args.predictive_temporal_audit_sha256,
            "current_grid_build_report": args.current_grid_cache_build_report_sha256,
            "robot_config": _sha256(args.robot_config),
            "training_config": _sha256(args.training_config),
            "representation_split": (
                None if representation_split_path is None else _sha256(representation_split_path)
            ),
            "representation_task_intervention": (
                None
                if representation_task_intervention_path is None
                else _sha256(representation_task_intervention_path)
            ),
            "fixed_observation_pair_plan": (
                None
                if fixed_observation_pair_plan_path is None
                else _sha256(fixed_observation_pair_plan_path)
            ),
            "fixed_observation_training_audit": (
                None
                if fixed_observation_training_audit_path is None
                else _sha256(fixed_observation_training_audit_path)
            ),
            "fixed_observation_evaluation_plan": (
                None
                if fixed_observation_evaluation_plan_path is None
                else _sha256(fixed_observation_evaluation_plan_path)
            ),
            "fixed_observation_validation_audit": (
                None
                if fixed_observation_validation_audit_path is None
                else _sha256(fixed_observation_validation_audit_path)
            ),
            "fixed_observation_heldout_audit": (
                None
                if fixed_observation_heldout_audit_path is None
                else _sha256(fixed_observation_heldout_audit_path)
            ),
            "representation_warm_evaluation": (
                None
                if representation_warm_evaluation_path is None
                else _sha256(representation_warm_evaluation_path)
            ),
            "behavior_causal_probe_evidence": (
                None
                if behavior_probe_evidence_path is None
                else _require_sha256(
                    "behavior causal-probe evidence sha256",
                    behavior_probe_evidence_sha256,
                )
            ),
            "behavior_posterior_control_probe_evidence": (
                None
                if behavior_factorial_evidence_path is None
                else _require_sha256(
                    "behavior G2 evidence sha256",
                    behavior_factorial_evidence_sha256,
                )
            ),
        },
        "objective": {
            "action_weight": (
                (0.0).hex() if training_stage == NATIVE_REPRESENTATION_STAGE else (1.0).hex()
            ),
            "existence_weight": float(args.existence_weight).hex(),
            "dense_task_weight": float(args.dense_task_weight).hex(),
            "minimum_supervised_fraction": float(args.minimum_supervised_fraction).hex(),
            "ownership_weight": float(args.ownership_weight).hex(),
            "ownership_estimator": _ownership_estimator_from_args(args),
            "predictive_loss_power": float(args.predictive_loss_power).hex(),
            "predictive_term_weight": float(args.predictive_term_weight).hex(),
            "current_grid_term_weight": float(args.current_grid_term_weight).hex(),
            "omitted_static_term_weight": float(args.omitted_static_term_weight).hex(),
            "predictive_weight": float(args.predictive_weight).hex(),
            "structural_weight": float(args.structural_weight).hex(),
            "support_weight": float(args.support_weight).hex(),
            "task_weight": float(args.task_weight).hex(),
            "task_relation_estimator": args.task_relation_estimator,
            "predictive_objective_semantics": IMPLEMENTED_PREDICTIVE_OBJECTIVE,
            "predictive_visible_support_weighting": (
                IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING
            ),
            "behavior_conditioning": _behavior_conditioning_contract(
                args,
                behavior_graph_sha256=behavior_graph_sha256,
            ),
        },
        "evidence_profile": args.evidence_profile,
        "gradient_audit_steps": args.gradient_audit_steps,
        "optimizer": {
            "lingbot_release_contract": dict(optimizer_contract),
            "max_grad_norm": float(args.max_grad_norm).hex(),
        },
        "native": {
            "capacity": args.capacity,
            "maximum_control_tokens": args.maximum_control_tokens,
            "maximum_peak_reserved_gib": float(args.maximum_peak_reserved_gib).hex(),
            "relation_supervision_layers": list(args.relation_supervision_layers),
        },
        "patched_source_sha256": dict(sorted(patched_source_sha256.items())),
        "predictive_cache": dict(sorted(predictive_report.items())),
        "current_grid_cache": dict(sorted(current_grid_report.items())),
        "query_schema_sha256": frozen_query_schemas,
        "sampling": {
            "comparison_id": FULL_COMPARISON_ID,
            "global_batch_size": FULL_WORLD_SIZE,
            "lane_interleave_factor": args.lane_interleave_factor,
            "reset_mixture": (
                None
                if reset_mixture is None
                else {
                    "denominator": reset_mixture[1],
                    "numerator": reset_mixture[0],
                }
            ),
            "seed": args.seed,
            "source_prediction_mode": args.source_prediction_mode,
            "source_mask_token_fraction": float(args.source_mask_token_fraction).hex(),
            "total_steps": args.total_planned_steps,
        },
        "temporal": dict(temporal_metadata),
        "topology": {
            "cuda_allocator": args.cuda_allocator,
            "fsdp2_placement": args.fsdp2_placement,
            "data_parallel_mode": "fsdp2",
            "full_shard": True,
            "gradient_accumulation_steps": 1,
            "world_size": FULL_WORLD_SIZE,
        },
    }
    return _canonical_digest(payload), implementation_sha256


def _validate_resume_extra(
    value: object,
    *,
    expected_global_step: int,
    expected_implementation_sha256: str,
    expected_model_family_sha256: str,
    expected_execution_sha256: str,
    expected_plan_sha256: str,
    expected_temporal_sha256: str,
    expected_source_digest: str,
    expected_behavior_conditioning_sha256: str | None,
    rank: int,
) -> dict[str, Any]:
    required = {
        "boundary_sha256",
        "behavior_conditioning_sha256",
        "execution_contract_sha256",
        "global_step",
        "implementation_sha256",
        "lane_snapshot",
        "model_family_sha256",
        "next_optimizer_step",
        "optimizer_local_moment_elements",
        "optimizer_state_entries",
        "plan_sha256",
        "rank",
        "rank_rng_state",
        "schema",
        "source_digest",
        "temporal_estimator_sha256",
        "world_size",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("native full checkpoint extra state is incomplete")
    if value["schema"] != FULL_EXTRA_STATE_SCHEMA:
        raise ValueError("native full checkpoint schema differs")
    if value["global_step"] != expected_global_step or (
        value["next_optimizer_step"] != expected_global_step
    ):
        raise ValueError("native full checkpoint optimizer boundary differs")
    expected = {
        "execution_contract_sha256": expected_execution_sha256,
        "implementation_sha256": expected_implementation_sha256,
        "model_family_sha256": expected_model_family_sha256,
        "plan_sha256": expected_plan_sha256,
        "source_digest": expected_source_digest,
        "temporal_estimator_sha256": expected_temporal_sha256,
    }
    if any(value[name] != digest for name, digest in expected.items()):
        raise ValueError("native full checkpoint provenance differs")
    if value["behavior_conditioning_sha256"] != expected_behavior_conditioning_sha256:
        raise ValueError("native full checkpoint behavior-conditioning contract differs")
    if value["behavior_conditioning_sha256"] is not None:
        _require_sha256(
            "native full checkpoint behavior conditioning",
            value["behavior_conditioning_sha256"],
        )
    if value["rank"] != rank or value["world_size"] != FULL_WORLD_SIZE:
        raise ValueError("native full checkpoint topology differs")
    if not isinstance(value["lane_snapshot"], bytes) or not value["lane_snapshot"]:
        raise ValueError("native full checkpoint lane snapshot is absent")
    if not isinstance(value["rank_rng_state"], dict):
        raise ValueError("native full checkpoint RNG state is absent")
    for name in ("optimizer_state_entries", "optimizer_local_moment_elements"):
        if isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 0:
            raise ValueError("native full checkpoint optimizer summary is invalid")
    boundary = value["boundary_sha256"]
    if not isinstance(boundary, dict) or set(boundary) != {
        "lane_snapshot_sha256",
        "model_local_state_sha256",
        "optimizer_local_state_sha256",
        "rank_rng_state_sha256",
    }:
        raise ValueError("native full checkpoint boundary hashes are incomplete")
    for name, digest in boundary.items():
        _require_sha256(f"native full checkpoint {name}", digest)
    return value


def _validate_representation_resume_extra(
    value: object,
    *,
    expected_global_step: int,
    expected_implementation_sha256: str,
    expected_model_family_sha256: str,
    expected_execution_sha256: str,
    expected_plan_sha256: str,
    expected_temporal_sha256: str,
    expected_source_digest: str,
    expected_representation_split_sha256: str,
    expected_parameter_scope_sha256: str,
    expected_behavior_conditioning_sha256: str | None,
    rank: int,
) -> dict[str, Any]:
    """Validate an exact representation resume without weakening action checkpoints."""

    if not isinstance(value, dict):
        raise ValueError("native representation checkpoint extra state is incomplete")
    staged_fields = {
        "training_stage",
        "representation_split_sha256",
        "representation_parameter_scope_sha256",
        "representation_frozen_action_state_sha256",
    }
    if not staged_fields.issubset(value):
        raise ValueError("native representation checkpoint stage fields are incomplete")
    common = {name: item for name, item in value.items() if name not in staged_fields}
    common["schema"] = FULL_EXTRA_STATE_SCHEMA
    validated = _validate_resume_extra(
        common,
        expected_global_step=expected_global_step,
        expected_implementation_sha256=expected_implementation_sha256,
        expected_model_family_sha256=expected_model_family_sha256,
        expected_execution_sha256=expected_execution_sha256,
        expected_plan_sha256=expected_plan_sha256,
        expected_temporal_sha256=expected_temporal_sha256,
        expected_source_digest=expected_source_digest,
        expected_behavior_conditioning_sha256=expected_behavior_conditioning_sha256,
        rank=rank,
    )
    if value["schema"] != REPRESENTATION_EXTRA_STATE_SCHEMA:
        raise ValueError("native representation checkpoint schema differs")
    if value["training_stage"] != NATIVE_REPRESENTATION_STAGE:
        raise ValueError("native representation checkpoint stage differs")
    expected = {
        "representation_split_sha256": expected_representation_split_sha256,
        "representation_parameter_scope_sha256": expected_parameter_scope_sha256,
    }
    if any(value[name] != digest for name, digest in expected.items()):
        raise ValueError("native representation checkpoint provenance differs")
    _require_sha256(
        "native representation frozen action state",
        value["representation_frozen_action_state_sha256"],
    )
    return {**validated, **{name: value[name] for name in staged_fields}}


def _collated_to_device(batch: Any, *, device: Any, torch_module: Any) -> Any:
    from picf_next.lingbot_native.calvin import CollatedNativeCALVINBatch

    return CollatedNativeCALVINBatch(
        model_inputs=_move_model_inputs(
            batch.model_inputs,
            device=device,
            dtype=torch_module.bfloat16,
            torch_module=torch_module,
        ),
        controls=batch.controls,
        routing=batch.routing,
        source_digest=batch.source_digest,
        structural_target_requests=batch.structural_target_requests,
        modalities=(
            None
            if batch.modalities is None
            else batch.modalities.to(device=device, dtype=torch_module.bfloat16)
        ),
        prior_control_chunks=batch.prior_control_chunks,
        wla_world_target=(
            None
            if batch.wla_world_target is None
            else batch.wla_world_target.to(device=device, dtype=torch_module.bfloat16)
        ),
    )


def _validate_lingbot_projection_processor(
    *,
    physical_visual_acceptance: Mapping[str, object],
    processor_report: Mapping[str, object],
    vision_config: object,
    dataset_tree_sha256: str,
    transformers_version: str,
) -> dict[str, Any]:
    """Bind D1's measured token geometry to this exact training processor."""

    projection = validate_lingbot_calvin_projection_payload(
        physical_visual_acceptance.get("training_projection"),
        expected_dataset_manifest_sha256=cast(
            str,
            physical_visual_acceptance.get("dataset_manifest_sha256"),
        ),
    )
    if (
        projection["processor_id"] != QWEN_PROCESSOR_ID
        or projection["processor_revision"] != QWEN_PROCESSOR_REVISION
        or projection["dataset_tree_sha256"]
        != _require_sha256("dataset tree sha256", dataset_tree_sha256)
        or projection["transformers_version"] != transformers_version
    ):
        raise RuntimeError("accepted CALVIN projection and training runtime identity differ")
    if (
        processor_report.get("processor_id") != QWEN_PROCESSOR_ID
        or processor_report.get("processor_revision") != QWEN_PROCESSOR_REVISION
    ):
        raise RuntimeError("validated Qwen processor identity differs from training")
    assets = processor_report.get("processor_assets")
    if not isinstance(assets, list):
        raise RuntimeError("validated Qwen processor returned no exact asset manifest")
    if processor_assets_sha256(assets) != projection["processor_assets_sha256"]:
        raise RuntimeError("accepted CALVIN projection used different processor assets")
    assets_by_path = {
        str(item["path"]): str(item["sha256"])
        for item in assets
        if isinstance(item, Mapping) and set(item) == {"path", "bytes", "sha256"}
    }
    if (
        assets_by_path.get("config.json") != projection["processor_config_sha256"]
        or assets_by_path.get("preprocessor_config.json")
        != projection["processor_preprocessor_config_sha256"]
    ):
        raise RuntimeError("accepted CALVIN projection configuration assets differ")
    measured_geometry = (
        projection["patch_size"],
        projection["merge_size"],
        projection["temporal_patch_size"],
    )
    runtime_geometry = (
        getattr(vision_config, "patch_size", None),
        getattr(vision_config, "spatial_merge_size", None),
        getattr(vision_config, "temporal_patch_size", None),
    )
    if runtime_geometry != measured_geometry:
        raise RuntimeError("accepted CALVIN projection and Qwen vision geometry differ")
    return projection


def _validate_training_supervision_policy(
    *,
    physical_visual_acceptance: Mapping[str, object],
    minimum_supervised_fraction: float,
) -> dict[str, Any]:
    """Require D0/D1 and runtime to use one known-pixel loss measure."""

    accepted = validate_known_pixel_token_supervision_policy(
        physical_visual_acceptance.get("training_supervision_policy")
    )
    expected = build_known_pixel_token_supervision_policy(
        minimum_observed_fraction=minimum_supervised_fraction
    )
    if accepted != expected:
        raise RuntimeError("accepted CALVIN token supervision policy differs from training")
    return accepted


def _validate_lingbot_calvin_projection_batch(
    model_inputs: Mapping[str, object],
    *,
    projection: Mapping[str, object],
) -> None:
    """Require the released CALVIN transform to preserve measured view slots."""

    image_grid = model_inputs.get("image_grid_thw")
    image_valid = model_inputs.get("img_masks")
    grid_shape = getattr(image_grid, "shape", None)
    valid_shape = getattr(image_valid, "shape", None)
    if (
        grid_shape is None
        or valid_shape is None
        or len(grid_shape) != 3
        or tuple(grid_shape[1:]) != (3, 3)
        or tuple(valid_shape) != tuple(grid_shape[:2])
    ):
        raise RuntimeError("LingBot CALVIN runtime camera-slot geometry changed")
    grid_detach = getattr(image_grid, "detach", None)
    valid_detach = getattr(image_valid, "detach", None)
    if not callable(grid_detach) or not callable(valid_detach):
        raise TypeError("LingBot CALVIN camera slots must be tensors")
    grids = cast(Any, grid_detach()).cpu().tolist()
    valid = cast(Any, valid_detach()).cpu().tolist()
    views = projection.get("views")
    if not isinstance(views, Mapping):
        raise RuntimeError("accepted CALVIN projection has no views")
    static = views["static"]
    gripper = views["gripper"]
    if not isinstance(static, Mapping) or not isinstance(gripper, Mapping):
        raise RuntimeError("accepted CALVIN projection view geometry changed")
    projection_views = {"static": static, "gripper": gripper}
    expected_grids = [
        projection_views[slot.projection_camera_name]["image_grid_thw"]
        for slot in LINGBOT_CALVIN_CAMERA_SLOTS
    ]
    expected_valid = [slot.valid for slot in LINGBOT_CALVIN_CAMERA_SLOTS]
    if any(row != expected_grids for row in grids) or any(row != expected_valid for row in valid):
        raise RuntimeError("LingBot CALVIN batch differs from accepted camera projection")


def _optimizer_attempt(
    *,
    policy: Any,
    optimizer: Any,
    global_step: int,
    max_grad_norm: float,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> tuple[int | None, dict[str, float | int | bool]]:
    metrics = _distributed_gradient_metrics(
        policy,
        (
            ("native_graph", "picf_native_graph"),
            ("relation_projection", "picf_native_graph.relation_readout.projection"),
            ("match_projection", "picf_native_graph.relation_readout.match_projection"),
            ("action_output", "action_out_proj"),
            ("predictive_readout", "predictive_readouts.dino_video"),
        ),
        device=device,
        dist=dist,
        torch_module=torch_module,
    )
    if not bool(metrics["all_finite"]):
        optimizer.zero_grad(set_to_none=True)
        return None, metrics
    clipped_value = clip_lingbot_distributed_l2_grad_norm_(
        policy.parameters(),
        max_grad_norm,
        device=device,
        dist_module=dist,
        torch_module=torch_module,
        error_if_nonfinite=True,
    )
    metrics["preclip_global_norm"] = float(clipped_value)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return global_step + 1, metrics


def _emit_step_progress(
    event: str,
    *,
    rank: int,
    global_step: int,
    details: Mapping[str, object] | None = None,
) -> None:
    allowed = {
        "step_started",
        "step_batch_ready",
        "gradient_audit_replay_started",
        "objective_started",
        "objective_ready",
        "backward_started",
        "backward_completed",
        "step_completed",
    }
    if event not in allowed:
        raise ValueError(f"unsupported native full progress event: {event}")
    payload: dict[str, object] = {
        "event": f"native_full_{event}",
        "global_step": global_step,
        "rank": rank,
    }
    if details is not None:
        reserved = set(payload).intersection(details)
        if reserved:
            raise ValueError(
                f"native full progress details override reserved fields: {sorted(reserved)}"
            )
        payload.update(details)
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _trim_cuda_allocator_after_gradient_audit(
    *,
    gradient_audit: bool,
    torch_module: Any,
) -> None:
    """Return transient audit segments without perturbing ordinary-step caching."""

    if not isinstance(gradient_audit, bool):
        raise TypeError("gradient-audit allocator trim requires a boolean decision")
    if gradient_audit:
        torch_module.cuda.empty_cache()


def _distributed_ring_exchange_tensor(
    value: Any,
    *,
    dist: Any,
    torch_module: Any,
) -> Any:
    """Return the same-shaped tensor from the next rank for a negative control."""

    if not torch_module.is_tensor(value) or value.ndim < 1:
        raise TypeError("distributed counterfactual exchange requires a non-scalar tensor")
    world_size = int(dist.get_world_size())
    rank = int(dist.get_rank())
    if world_size != FULL_WORLD_SIZE or not 0 <= rank < world_size:
        raise RuntimeError("counterfactual exchange requires the frozen two-rank topology")
    gathered = [torch_module.empty_like(value) for _ in range(world_size)]
    dist.all_gather(gathered, value.detach())
    peer = gathered[(rank + 1) % world_size]
    if peer.shape != value.shape or peer.dtype != value.dtype or peer.device != value.device:
        raise RuntimeError("distributed counterfactual exchange changed tensor metadata")
    return peer


def _distributed_uniform_boolean(
    value: bool,
    *,
    name: str,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> bool:
    """Require every rank to take the same optional-forward branch."""

    if not isinstance(value, bool) or not isinstance(name, str) or not name:
        raise TypeError("distributed boolean consensus requires a named boolean")
    minimum = torch_module.tensor(int(value), dtype=torch_module.int32, device=device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if int(minimum.item()) != int(maximum.item()):
        raise RuntimeError(f"distributed ranks disagree on {name}")
    return bool(minimum.item())


def _distributed_any_boolean(
    value: bool,
    *,
    name: str,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> bool:
    """Return whether any rank needs one optional collective host forward."""

    if not isinstance(value, bool) or not isinstance(name, str) or not name:
        raise TypeError("distributed boolean union requires a named boolean")
    maximum = torch_module.tensor(int(value), dtype=torch_module.int32, device=device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return bool(maximum.item())


def _distributed_pre_backward_failures(
    error: BaseException | None,
    *,
    rank: int,
    expected_world_size: int,
    dist: Any,
) -> tuple[dict[str, Any], ...]:
    """Gather Python objective failures on the independent CPU control plane."""

    world_size = int(dist.get_world_size())
    if (
        isinstance(expected_world_size, bool)
        or expected_world_size not in (2, 4)
        or world_size != expected_world_size
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank != int(dist.get_rank())
        or not 0 <= rank < world_size
    ):
        raise RuntimeError("pre-backward failure exchange requires the registered topology")
    local = (
        None
        if error is None
        else {
            "rank": rank,
            "type": type(error).__name__,
            "message": str(error),
        }
    )
    if error is not None:
        print(
            json.dumps(
                {
                    "event": "local_distributed_failure_before_exchange",
                    "rank": rank,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(
                        traceback.format_exception(type(error), error, error.__traceback__)
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local)
    failures: list[dict[str, Any]] = []
    for expected_rank, item in enumerate(gathered):
        if item is None:
            continue
        if (
            not isinstance(item, dict)
            or set(item) != {"rank", "type", "message"}
            or item["rank"] != expected_rank
            or not isinstance(item["type"], str)
            or not item["type"]
            or not isinstance(item["message"], str)
        ):
            raise RuntimeError("pre-backward failure exchange returned malformed rank evidence")
        failures.append(item)
    return tuple(failures)


def _distributed_family_gradient_diagnostics(
    *,
    family_gradients: Mapping[str, Any],
    probe: str = "picf_native_graph.object_queries",
    device: Any,
    dist: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Measure objective-family conflict from isolated ordinary backward passes."""

    names = ("action", "predictive", "structural")
    if set(family_gradients) != set(names):
        raise RuntimeError("native objective omitted an exact family gradient surface")
    if probe not in {
        "picf_native_graph.object_queries",
        REPRESENTATION_FAMILY_GRADIENT_PROBE,
    }:
        raise ValueError("native family-gradient probe identity is unsupported")
    local_gradients: list[Any] = []
    for name in names:
        gradient = family_gradients[name]
        if not torch_module.is_tensor(gradient) or gradient.numel() <= 0:
            raise RuntimeError(f"native {name} family gradient snapshot is malformed")
        local_gradients.append(gradient.detach().float().reshape(-1))
    if len({int(gradient.numel()) for gradient in local_gradients}) != 1:
        raise RuntimeError("native family gradient snapshots have inconsistent local shapes")

    statistics = []
    for gradient in local_gradients:
        statistics.append(gradient.square().sum())
    for left, right in ((0, 1), (0, 2), (1, 2)):
        statistics.append((local_gradients[left] * local_gradients[right]).sum())
    finite = torch_module.stack(
        tuple(torch_module.isfinite(gradient).all() for gradient in local_gradients)
    ).all()
    packed = torch_module.stack((*statistics, finite.to(statistics[0].dtype))).to(device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = packed.detach().cpu().tolist()
    world_size = int(dist.get_world_size())
    if int(round(values[-1])) != world_size:
        raise FloatingPointError("a native objective-family gradient is non-finite")
    squared_norms = values[:3]
    dots = values[3:6]
    norms = [math.sqrt(max(value, 0.0)) for value in squared_norms]
    pair_names = (("action", "predictive"), ("action", "structural"), ("predictive", "structural"))
    cosines: dict[str, float | None] = {}
    dot_products: dict[str, float] = {}
    for (left_name, right_name), left, right, dot in zip(
        pair_names,
        (0, 0, 1),
        (1, 2, 2),
        dots,
        strict=True,
    ):
        key = f"{left_name}__{right_name}"
        denominator = norms[left] * norms[right]
        dot_products[key] = float(dot)
        cosines[key] = None if denominator == 0 else float(dot / denominator)
    return {
        "all_finite": True,
        "cosines": cosines,
        "dot_products": dot_products,
        "gradient_norms": dict(zip(names, (float(value) for value in norms), strict=True)),
        "probe": probe,
        "world_size": world_size,
    }


def _distributed_relation_surface_gradient_diagnostics(
    *,
    component_gradients: Mapping[str, Any],
    device: Any,
    dist: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Measure component conflict on the concatenated two-rank relation surfaces."""

    names, pair_inputs = _relation_surface_gradient_contract(component_gradients)
    local_gradients: dict[str, Any] = {}
    for name in names:
        gradient = component_gradients[name]
        if not torch_module.is_tensor(gradient) or gradient.numel() <= 0:
            raise RuntimeError(f"{name} relation-surface gradient is malformed")
        local_gradients[name] = gradient.detach().float().reshape(-1)

    if any(
        local_gradients[left].shape != local_gradients[right].shape
        for _pair, left, right in pair_inputs
    ):
        raise RuntimeError("relation-surface gradient pairs have inconsistent local shapes")
    squared_norms = tuple(local_gradients[name].square().sum() for name in names)
    dots = tuple(
        (local_gradients[left] * local_gradients[right]).sum() for _pair, left, right in pair_inputs
    )
    finite = torch_module.stack(
        tuple(torch_module.isfinite(gradient).all() for gradient in local_gradients.values())
    ).all()
    packed = torch_module.stack((*squared_norms, *dots, finite.to(squared_norms[0].dtype))).to(
        device=device
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    element_counts = torch_module.tensor(
        [gradient.numel() for gradient in local_gradients.values()],
        dtype=torch_module.int64,
        device=device,
    )
    dist.all_reduce(element_counts, op=dist.ReduceOp.SUM)
    world_size = int(dist.get_world_size())
    values = packed.detach().cpu().tolist()
    if int(round(values[-1])) != world_size:
        raise FloatingPointError("a relation-surface component gradient is non-finite")
    norms = [math.sqrt(max(value, 0.0)) for value in values[: len(names)]]
    dot_values = values[len(names) : len(names) + len(pair_inputs)]
    norm_by_name: dict[str, float] = dict(zip(names, norms, strict=True))
    dot_products: dict[str, float] = {}
    cosines: dict[str, float | None] = {}
    for (pair, left, right), dot in zip(pair_inputs, dot_values, strict=True):
        denominator = norm_by_name[left] * norm_by_name[right]
        dot_products[pair] = float(dot)
        cosines[pair] = None if denominator == 0 else float(dot / denominator)
    return {
        "all_finite": True,
        "cosines": cosines,
        "dot_products": dot_products,
        "gradient_elements": dict(
            zip(
                names,
                (int(value) for value in element_counts.detach().cpu().tolist()),
                strict=True,
            )
        ),
        "gradient_norms": {name: float(norm_by_name[name]) for name in names},
        "probe": "final_relation.match_embeddings+row_embeddings",
        "world_size": world_size,
    }


def _parameter_gradient_snapshot(
    parameter: Any,
    *,
    name: str,
    torch_module: Any,
) -> Any:
    """Clone one local post-backward gradient before FSDP2 zeroing or resharding."""

    if not isinstance(name, str) or not name:
        raise TypeError("gradient snapshot requires a non-empty name")
    if not getattr(parameter, "requires_grad", False):
        raise RuntimeError(f"{name} gradient snapshot selected a frozen parameter")
    gradient = getattr(parameter, "grad", None)
    if gradient is None:
        raise RuntimeError(f"{name} loss did not reach its audited parameter")
    to_local = getattr(gradient, "to_local", None)
    if callable(to_local):
        gradient = to_local()
    if not torch_module.is_tensor(gradient):
        raise TypeError(f"{name} gradient is not a non-empty tensor")
    tensor_gradient = cast(Any, gradient)
    if tensor_gradient.numel() <= 0:
        raise TypeError(f"{name} gradient is not a non-empty tensor")
    return tensor_gradient.detach().float().clone()


def _shared_host_family_gradient_snapshot(
    selected_host: Mapping[str, tuple[str, Any]],
    *,
    family_name: str,
    torch_module: Any,
) -> Any:
    """Flatten the same shared early/middle/late host surface for one family."""

    depths = ("early", "middle", "late")
    if set(selected_host) != set(depths):
        raise RuntimeError("shared family-gradient probe omitted an exact depth surface")
    snapshots = tuple(
        _parameter_gradient_snapshot(
            selected_host[depth][1],
            name=f"{family_name} {depth} shared-host",
            torch_module=torch_module,
        ).reshape(-1)
        for depth in depths
    )
    return torch_module.cat(snapshots)


def _backward_isolated_objective_family(
    family_terms: Mapping[str, Any],
    *,
    selected_name: str,
    torch_module: Any,
) -> None:
    """Differentiate one family while traversing the complete FSDP2 objective graph."""

    names = ("action", "predictive", "structural")
    if set(family_terms) != set(names):
        raise RuntimeError("native objective omitted an exact family gradient surface")
    if selected_name not in names:
        raise ValueError("native objective selected an unsupported gradient family")
    terms = tuple(family_terms[name] for name in names)
    for name, term in zip(names, terms, strict=True):
        if (
            not isinstance(term, torch_module.Tensor)
            or term.ndim != 0
            or not term.requires_grad
            or not bool(torch_module.isfinite(term).item())
        ):
            raise RuntimeError(f"native {name} family is not a finite attached scalar")
    cotangents = tuple(
        torch_module.ones_like(term) if name == selected_name else torch_module.zeros_like(term)
        for name, term in zip(names, terms, strict=True)
    )
    torch_module.autograd.backward(terms, grad_tensors=cotangents)


def _weighted_behavior_future_contribution(
    result: Any,
    *,
    predictive_family_weight: float,
    torch_module: Any,
) -> Any:
    """Recover the exact behavior term contribution used by the optimizer."""

    terms = result.objective.predictive_terms
    matches = tuple(term for term in terms if term.name == "rollout/vision/binding")
    if len(matches) != 1:
        raise RuntimeError("behavior objective requires exactly one causal rollout term")
    active_weight = 0.0
    for term in terms:
        if term.sample_weight is None:
            active = bool(term.valid.any().item())
        else:
            active = bool(term.sample_weight.masked_select(term.valid).sum().item() > 0)
        if active:
            active_weight += float(term.weight)
    behavior_term = matches[0]
    if active_weight <= 0 or not bool(behavior_term.valid.any().item()):
        raise RuntimeError("behavior objective has no active causal rollout mass")
    normalized = result.objective.objective.normalized_terms[behavior_term.name]
    contribution = (
        float(predictive_family_weight) * float(behavior_term.weight) * normalized / active_weight
    )
    if (
        not isinstance(contribution, torch_module.Tensor)
        or contribution.ndim != 0
        or not contribution.requires_grad
        or not bool(torch_module.isfinite(contribution).item())
    ):
        raise RuntimeError("weighted behavior contribution is not a finite attached scalar")
    return contribution


def _backward_behavior_total_host(
    behavior_term: Any,
    *,
    selected_host: Mapping[str, tuple[str, Any]],
    torch_module: Any,
) -> dict[str, Any]:
    """Measure the complete behavior gradient through FSDP2 backward hooks."""

    torch_module.autograd.backward(behavior_term)
    return {
        depth: _parameter_gradient_snapshot(
            parameter,
            name=f"behavior {depth} total-host",
            torch_module=torch_module,
        )
        for depth, (_path, parameter) in selected_host.items()
    }


def _backward_behavior_via_posterior_host(
    behavior_term: Any,
    *,
    behavior_rows: Any,
    selected_host: Mapping[str, tuple[str, Any]],
    torch_module: Any,
) -> tuple[Any, dict[str, Any]]:
    """Measure only credit entering the host through deploy posterior rows.

    FSDP2 replaces sharded parameters with temporary unsharded tensors during
    forward.  Explicit parameter VJPs taken after resharding therefore do not
    name the tensors used by that graph.  Ordinary backward is the supported
    boundary: its FSDP hooks reduce credit back into the registered shards.
    """

    posterior_credit = torch_module.autograd.grad(
        behavior_term,
        behavior_rows,
    )[0]
    if not bool(torch_module.isfinite(posterior_credit).all().item()) or not bool(
        (posterior_credit != 0).any().item()
    ):
        raise RuntimeError("behavior loss produced no finite deploy-posterior credit")
    torch_module.autograd.backward(
        behavior_rows,
        grad_tensors=posterior_credit.detach(),
    )
    snapshots = {
        depth: _parameter_gradient_snapshot(
            parameter,
            name=f"behavior {depth} primary-host",
            torch_module=torch_module,
        )
        for depth, (_path, parameter) in selected_host.items()
    }
    return posterior_credit, snapshots


def _predictive_host_gradient_parameters(policy: Any) -> dict[str, tuple[str, Any]]:
    """Select stable early/middle/late shared-host parameters after FSDP2 wrapping."""

    try:
        layers = policy.model.qwenvl_with_expert.qwenvl.model.language_model.layers
    except AttributeError as error:
        raise RuntimeError("LingBot policy lost its shared language-layer contract") from error
    if len(layers) != PREDICTIVE_SHARED_HOST_LAYERS:
        raise RuntimeError("LingBot predictive-host depth differs from the released contract")
    indices = {"early": 0, "middle": len(layers) // 2, "late": len(layers) - 1}
    names_by_identity = {id(parameter): name for name, parameter in policy.named_parameters()}
    selected: dict[str, tuple[str, Any]] = {}
    for depth, index in indices.items():
        try:
            parameter = layers[index].input_layernorm.weight
        except AttributeError as error:
            raise RuntimeError(
                f"LingBot shared layer {index} lost its input normalization parameter"
            ) from error
        name = names_by_identity.get(id(parameter))
        if name is None or not name.endswith(f"layers.{index}.input_layernorm.weight"):
            raise RuntimeError("LingBot predictive-host probe parameter is not uniquely registered")
        if not parameter.requires_grad:
            raise RuntimeError("LingBot predictive-host probe selected a frozen parameter")
        if tuple(parameter.shape) != (PREDICTIVE_SHARED_HOST_WIDTH,):
            raise RuntimeError("LingBot predictive-host width differs from the released contract")
        selected[depth] = (name, parameter)
    if len({id(parameter) for _name, parameter in selected.values()}) != len(selected):
        raise RuntimeError("LingBot predictive-host probe selected duplicate parameters")
    return selected


def _distributed_predictive_host_gradient_diagnostics(
    *,
    host_gradients: Mapping[str, tuple[str, Any]],
    decomposition_gradients: Mapping[str, Mapping[str, Any]] | None = None,
    probe: str = "lingbot.language_model.input_layernorm",
    device: Any,
    dist: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Measure whether an isolated predictive backward reaches the shared host."""

    depths = ("early", "middle", "late")
    if set(host_gradients) != set(depths):
        raise RuntimeError("predictive-host probe omitted an exact depth surface")
    statistics = []
    local_connected = []
    local_finite = []
    local_elements = []
    for depth in depths:
        gradient = host_gradients[depth][1]
        if not torch_module.is_tensor(gradient) or gradient.numel() <= 0:
            raise RuntimeError(f"predictive-host probe captured a malformed {depth} gradient")
        detached = gradient.detach().float().reshape(-1)
        statistic = detached.square().sum()
        statistics.append(statistic)
        local_connected.append(statistic.new_tensor(int(statistic.item() > 0)))
        local_finite.append(statistic.new_tensor(int(torch_module.isfinite(detached).all().item())))
        local_elements.append(statistic.new_tensor(int(detached.numel())))
    packed = torch_module.stack((*statistics, *local_connected, *local_finite, *local_elements)).to(
        device=device,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = packed.detach().cpu().tolist()
    world_size = int(dist.get_world_size())
    connected_counts = values[3:6]
    finite_counts = values[6:9]
    element_counts = values[9:12]
    if any(int(round(count)) != world_size for count in finite_counts):
        raise FloatingPointError("predictive loss produced a non-finite shared-host gradient")
    if any(int(round(count)) != PREDICTIVE_SHARED_HOST_WIDTH for count in element_counts):
        raise RuntimeError("predictive-host probe observed an incomplete FSDP2 gradient shard")
    norms = {
        depth: math.sqrt(max(float(value), 0.0))
        for depth, value in zip(depths, values[:3], strict=True)
    }
    elements = dict.fromkeys(depths, PREDICTIVE_SHARED_HOST_WIDTH)
    missing = tuple(
        depth
        for depth, connected in zip(depths, connected_counts, strict=True)
        if int(round(connected)) != world_size or norms[depth] <= 0
    )
    if missing:
        raise RuntimeError(
            "predictive loss did not reach the shared LingBot host at depths " + ",".join(missing)
        )
    if probe not in {
        "lingbot.language_model.input_layernorm",
        "lingbot.language_model.input_layernorm.via_primary_posterior_vjp",
    }:
        raise ValueError("predictive-host gradient probe identity is unsupported")
    decomposition = None
    if decomposition_gradients is not None:
        components = ("total", "via_posterior", "direct")
        if set(decomposition_gradients) != set(components) or any(
            set(decomposition_gradients[component]) != set(depths) for component in components
        ):
            raise RuntimeError("behavior-host gradient decomposition is incomplete")
        packed_statistics = []
        for depth in depths:
            total = decomposition_gradients["total"][depth].reshape(-1).float()
            via = decomposition_gradients["via_posterior"][depth].reshape(-1).float()
            direct = decomposition_gradients["direct"][depth].reshape(-1).float()
            if total.shape != via.shape or total.shape != direct.shape:
                raise RuntimeError("behavior-host gradient decomposition shapes differ")
            if not all(
                bool(torch_module.isfinite(value).all().item()) for value in (total, via, direct)
            ):
                raise FloatingPointError("behavior-host gradient decomposition is non-finite")
            closure = total - via - direct
            packed_statistics.extend(
                (
                    total.square().sum(),
                    via.square().sum(),
                    direct.square().sum(),
                    (total * via).sum(),
                    (total * direct).sum(),
                    closure.square().sum(),
                )
            )
        packed_decomposition = torch_module.stack(packed_statistics).to(device=device)
        dist.all_reduce(packed_decomposition, op=dist.ReduceOp.SUM)
        reduced = packed_decomposition.detach().cpu().tolist()
        by_depth: dict[str, dict[str, float]] = {}
        for index, depth in enumerate(depths):
            total_sq, via_sq, direct_sq, total_via, total_direct, closure_sq = (
                float(value) for value in reduced[index * 6 : (index + 1) * 6]
            )
            total_norm = math.sqrt(max(total_sq, 0.0))
            via_norm = math.sqrt(max(via_sq, 0.0))
            direct_norm = math.sqrt(max(direct_sq, 0.0))
            if total_norm <= 0 or via_norm <= 0:
                raise RuntimeError("behavior loss did not reach a decomposed shared-host path")
            by_depth[depth] = {
                "total_norm": total_norm,
                "via_posterior_norm": via_norm,
                "direct_norm": direct_norm,
                "via_to_total_norm_ratio": via_norm / total_norm,
                "total_via_cosine": total_via / (total_norm * via_norm),
                "total_direct_cosine": (
                    0.0 if direct_norm == 0 else total_direct / (total_norm * direct_norm)
                ),
                "closure_error_norm": math.sqrt(max(closure_sq, 0.0)),
            }
        decomposition = {
            "components": list(components),
            "depths": by_depth,
            "identity": "weighted_behavior_total=direct+via_primary_posterior",
        }
    return {
        "all_finite": True,
        "decomposition": decomposition,
        "gradient_elements": elements,
        "gradient_norms": norms,
        "parameter_paths": {depth: host_gradients[depth][0] for depth in depths},
        "probe": probe,
        "world_size": world_size,
    }


def _moe_routing_bias_snapshot(policy: Any, *, torch_module: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, buffer in policy.named_buffers():
        if not name.endswith("e_score_correction_bias"):
            continue
        local = buffer
        to_local = getattr(local, "to_local", None)
        if callable(to_local):
            local = to_local()
        if not torch_module.is_tensor(local):
            raise TypeError("LingBot MoE routing bias is not tensor-like")
        local_tensor = cast(Any, local)
        values[name] = local_tensor.detach().clone()
    if not values:
        raise RuntimeError("fixed-batch probe found no LingBot MoE routing bias")
    return values


def _reset_moe_probe_counters(policy: Any, *, torch_module: Any) -> None:
    observed = 0
    with torch_module.no_grad():
        for name, buffer in policy.named_buffers():
            if not name.endswith("tokens_per_expert"):
                continue
            local = buffer
            to_local = getattr(local, "to_local", None)
            if callable(to_local):
                local = to_local()
            if not torch_module.is_tensor(local):
                raise TypeError("LingBot MoE token counter is not tensor-like")
            local_tensor = cast(Any, local)
            local_tensor.zero_()
            observed += 1
    if observed == 0:
        raise RuntimeError("fixed-batch probe found no LingBot MoE token counters")


def _moe_routing_bias_matches(
    policy: Any,
    before: Mapping[str, Any],
    *,
    torch_module: Any,
) -> bool:
    after = _moe_routing_bias_snapshot(policy, torch_module=torch_module)
    return set(before) == set(after) and all(
        bool(torch_module.equal(before[name], after[name])) for name in before
    )


def _select_fixed_batch_plan(
    *,
    stream_plan: Any,
    dataset: Any,
    rank: int,
    world_size: int,
    total_planned_steps: int,
    device: Any,
    dtype: Any,
    dist: Any,
    torch_module: Any,
    build_planned_batch: Callable[..., Any],
    target_has_nontrivial_shuffle: Callable[[str], bool],
) -> tuple[int, Any]:
    """Choose the first global pair with a nontrivial target-control support."""

    for candidate in range(total_planned_steps):
        planned = build_planned_batch(
            stream_plan,
            dataset,
            optimizer_step=candidate,
            rank=rank,
            world_size=world_size,
            gradient_accumulation_steps=1,
            accumulation_index=0,
            device=device,
            dtype=dtype,
        )
        local_usable = all(
            dataset.available_future_transitions_by_key(key) >= 1
            and target_has_nontrivial_shuffle(key)
            for key in planned.training.routing.sample_keys
        )
        globally_usable = torch_module.tensor(
            int(local_usable),
            dtype=torch_module.int32,
            device=device,
        )
        dist.all_reduce(globally_usable, op=dist.ReduceOp.MIN)
        if bool(globally_usable.item()):
            return candidate, planned
    raise RuntimeError("stream plan contains no globally usable fixed two-frame probe batch")


def _run_predictive_fixed_batch_arm(
    *,
    args: argparse.Namespace,
    rank: int,
    device: Any,
    dist: Any,
    torch_module: Any,
    policy: Any,
    graph: Any,
    optimizer: Any,
    optimizer_contract: Any,
    trainable_scope: FixedBatchTrainableScope,
    stream_plan: Any,
    dataset: Any,
    collate_planned: Callable[[Any], Any],
    build_planned_batch: Callable[..., Any],
    build_continuation_batch: Callable[..., Any],
    run_full_objective: Callable[..., Any],
    predictive_cache: Any,
    current_grid_cache: Any,
    physical_sidecar: Any,
    task_identity_resolver: Callable[..., Any],
    patch_size: int,
    merge_size: int,
    objective_config: Any,
    structural_config: Any,
    derive_subseed_fn: Callable[..., int],
    patch_sha256: str,
    execution_sha256: str,
    implementation_sha256: str,
    model_family_sha256: str,
    dataset_contract_report: Mapping[str, Any],
) -> None:
    """Execute one fresh equal-budget arm and publish no trainable checkpoint."""

    arm = args.predictive_fixed_batch_arm
    curve_points = args.predictive_fixed_batch_curve_points
    optimizer_updates = curve_points - 1
    output_path = args.predictive_fixed_batch_output
    if arm not in PREDICTIVE_FIXED_BATCH_ARMS or output_path is None:
        raise RuntimeError("fixed-batch arm lost its validated command contract")
    fixed_sample_step, planned = _select_fixed_batch_plan(
        stream_plan=stream_plan,
        dataset=dataset,
        rank=rank,
        world_size=FULL_WORLD_SIZE,
        total_planned_steps=args.total_planned_steps,
        device=device,
        dtype=torch_module.bfloat16,
        dist=dist,
        torch_module=torch_module,
        build_planned_batch=build_planned_batch,
        target_has_nontrivial_shuffle=lambda key: (
            current_grid_cache.supported_current_summary_count(
                source_global_index=dataset.source_global_index_by_key(
                    dataset.future_sample_keys(key, count=1)[0]
                ),
                physical_sidecar=physical_sidecar,
                minimum_visible_fraction=predictive_cache.contract.minimum_visible_fraction,
            )
            >= 3
        ),
    )
    primary_batch = collate_planned(planned)
    continuation_batch = collate_planned(
        build_continuation_batch(
            planned,
            dataset,
            offset=1,
            device=device,
            dtype=torch_module.bfloat16,
        )
    )
    if primary_batch.routing.batch_size != 1:
        raise RuntimeError("fixed-batch two-rank contract requires one sample per rank")
    if (continuation_batch.controls.reset & continuation_batch.controls.token_valid).any():
        raise RuntimeError("fixed-batch observation pair crosses an episode reset")

    capacity_seeds = tuple(
        derive_subseed_fn(args.seed, "fixed-batch-capacity-censor", key)
        for key in primary_batch.routing.sample_keys
    )
    shuffled_cache = (
        ShuffledCurrentGridTargetCache(current_grid_cache) if arm == "shuffled_target" else None
    )
    selected_cache = current_grid_cache if shuffled_cache is None else shuffled_cache
    probe_shared_host = arm in {"full_host", "shuffled_target"}
    shared_host_gradient_probe: dict[str, Any] | None = None
    before_bias = _moe_routing_bias_snapshot(policy, torch_module=torch_module)
    optimizer.zero_grad(set_to_none=True)
    local_losses: list[float] = []
    global_losses: list[float] = []
    local_shuffle_distances: list[float] = []
    global_shuffle_distances: list[float] = []
    step_times: list[float] = []
    peak_reserved_bytes = 0
    fixed_row_bindings: tuple[Any, ...] = tuple(() for _ in range(primary_batch.routing.batch_size))
    total_started = time.perf_counter()
    for curve_index in range(curve_points):
        torch_module.cuda.reset_peak_memory_stats(device)
        if shuffled_cache is not None:
            shuffled_cache.begin_curve_point(curve_index)
        started = time.perf_counter()
        result = run_full_objective(
            policy,
            graph=graph,
            batches=(primary_batch, continuation_batch),
            previous_state=None,
            previous_state_valid=None,
            predictive_cache=predictive_cache,
            current_grid_cache=selected_cache,
            physical_sidecar=physical_sidecar,
            capacity=args.capacity,
            task_identity_resolver=task_identity_resolver,
            patch_size=patch_size,
            merge_size=merge_size,
            objective_config=objective_config,
            structural_config=structural_config,
            predictive_term_weight=args.predictive_term_weight,
            current_grid_term_weight=args.current_grid_term_weight,
            omitted_static_term_weight=args.omitted_static_term_weight,
            predictive_loss_power=args.predictive_loss_power,
            minimum_supervised_fraction=args.minimum_supervised_fraction,
            capacity_seeds=capacity_seeds,
            prior_row_bindings_by_batch=fixed_row_bindings,
        )
        observed_row_bindings = result.objective.row_bindings_by_batch
        if (
            not isinstance(observed_row_bindings, tuple)
            or len(observed_row_bindings) != primary_batch.routing.batch_size
        ):
            raise RuntimeError("fixed-batch objective returned malformed row bindings")
        if curve_index == 0:
            fixed_row_bindings = observed_row_bindings
        elif observed_row_bindings != fixed_row_bindings:
            raise RuntimeError("fixed-batch objective changed its frozen row gauge")
        predictive_loss = result.objective.objective.family_terms.get("predictive")
        if (
            predictive_loss is None
            or predictive_loss.ndim != 0
            or not predictive_loss.requires_grad
            or not bool(torch_module.isfinite(predictive_loss).item())
        ):
            raise RuntimeError("fixed-batch arm produced no finite differentiable predictive loss")
        local_loss = float(predictive_loss.detach().float().item())
        local_shuffle = 0.0 if shuffled_cache is None else float(shuffled_cache.maximum_distance)
        if curve_index < optimizer_updates:
            predictive_loss.backward()
            if curve_index == 0 and probe_shared_host:
                shared_host_gradient_probe = _distributed_predictive_host_gradient_diagnostics(
                    host_gradients={
                        depth: (
                            path,
                            _parameter_gradient_snapshot(
                                parameter,
                                name=f"predictive {depth} shared-host",
                                torch_module=torch_module,
                            ),
                        )
                        for depth, (path, parameter) in _predictive_host_gradient_parameters(
                            policy
                        ).items()
                    },
                    device=device,
                    dist=dist,
                    torch_module=torch_module,
                )
            successful_step, gradient_metrics = _optimizer_attempt(
                policy=policy,
                optimizer=optimizer,
                global_step=curve_index,
                max_grad_norm=args.max_grad_norm,
                device=device,
                dist=dist,
                torch_module=torch_module,
            )
            if successful_step != curve_index + 1:
                raise RuntimeError("fixed-batch optimizer update overflowed or was skipped")
            if (
                float(gradient_metrics.get("native_graph_norm", 0.0)) <= 0
                or float(gradient_metrics.get("predictive_readout_norm", 0.0)) <= 0
            ):
                raise RuntimeError("fixed-batch predictive loss missed its native prediction path")
            _reset_moe_probe_counters(policy, torch_module=torch_module)
        reduced = torch_module.tensor(
            [local_loss, local_shuffle],
            dtype=torch_module.float64,
            device=device,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced = reduced / FULL_WORLD_SIZE
        local_losses.append(local_loss)
        local_shuffle_distances.append(local_shuffle)
        global_losses.append(float(reduced[0].item()))
        global_shuffle_distances.append(float(reduced[1].item()))
        step_times.append(time.perf_counter() - started)
        peak_reserved_bytes = max(
            peak_reserved_bytes,
            int(torch_module.cuda.max_memory_reserved(device)),
        )

    total_time_s = time.perf_counter() - total_started
    local_bias_unchanged = torch_module.tensor(
        int(_moe_routing_bias_matches(policy, before_bias, torch_module=torch_module)),
        dtype=torch_module.int32,
        device=device,
    )
    dist.all_reduce(local_bias_unchanged, op=dist.ReduceOp.MIN)
    if not bool(local_bias_unchanged.item()):
        raise RuntimeError("fixed-batch probe changed LingBot MoE routing bias")
    if (shared_host_gradient_probe is None) != (arm in {"native_graph_only", "readout_only"}):
        raise RuntimeError("fixed-batch arm violated its shared-host gradient contract")
    local_report = {
        "rank": rank,
        "frame_sample_keys": [
            primary_batch.routing.sample_keys[0],
            continuation_batch.routing.sample_keys[0],
        ],
        "frame_source_digests": [
            primary_batch.source_digest,
            continuation_batch.source_digest,
        ],
        "loss_curve": local_losses,
        "shuffle_distance_curve": local_shuffle_distances,
        "step_times_s": step_times,
        "peak_reserved_bytes": peak_reserved_bytes,
    }
    rank_reports: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
    dist.all_gather_object(rank_reports, local_report)
    total_times: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
    dist.all_gather_object(total_times, total_time_s)

    publish_error: list[str | None] = [None]
    report: dict[str, object] | None = None
    if rank == 0:
        try:
            provenance = {
                "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "patch_sha256": patch_sha256,
                "execution_contract_sha256": execution_sha256,
                "implementation_sha256": implementation_sha256,
                "model_family_sha256": model_family_sha256,
                "plan_sha256": stream_plan.plan_sha256,
                "dataset_manifest_sha256": dataset_contract_report["manifest_sha256"],
                "physical_sidecar_manifest_sha256": physical_sidecar.manifest_sha256,
                "predictive_cache_manifest_sha256": predictive_cache.manifest_sha256,
                "current_grid_cache_manifest_sha256": current_grid_cache.manifest_sha256,
                "seed": args.seed,
                "fixed_sample_global_step": fixed_sample_step,
                "frame_sample_keys_by_rank": [
                    list(item["frame_sample_keys"]) for item in rank_reports
                ],
                "frame_source_digests_by_rank": [
                    list(item["frame_source_digests"]) for item in rank_reports
                ],
                "objective": {
                    "optimized_family": "predictive",
                    "target": "prior_to_current_object_summary",
                    "window": "fixed_two_frame_local_bptt",
                    "labels_are_loss_side_only": True,
                },
                "optimizer": {
                    "algorithm": optimizer_contract.algorithm,
                    "learning_rate_hex": optimizer_contract.learning_rate.hex(),
                    "weight_decay_hex": optimizer_contract.weight_decay.hex(),
                    "scheduler": optimizer_contract.scheduler,
                    "moe_load_balance_hook_enabled": False,
                    "update_count": optimizer_updates,
                },
            }
            report = {
                "schema": PREDICTIVE_FIXED_BATCH_ARM_REPORT_SCHEMA,
                "status": "PASS",
                "arm": arm,
                "subject_sha256": fixed_batch_probe_subject(
                    provenance,
                    curve_point_count=curve_points,
                ),
                "provenance": provenance,
                "trainable_scope": trainable_scope.as_dict(),
                "curve_point_count": curve_points,
                "optimizer_update_count": optimizer_updates,
                "global_loss_curve": global_losses,
                "global_shuffle_distance_curve": global_shuffle_distances,
                "rank_reports": rank_reports,
                "shared_host_gradient_probe": shared_host_gradient_probe,
                "moe_routing_bias_unchanged": True,
                "maximum_peak_reserved_bytes": max(
                    int(item["peak_reserved_bytes"]) for item in rank_reports
                ),
                "total_time_s": max(float(value) for value in total_times),
            }
            validate_predictive_fixed_batch_arm_report(report)
            output = Path(output_path)
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"fixed-batch output already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
            _write_text_durable(output, payload)
        except BaseException as error:
            publish_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(publish_error, src=0)
    if publish_error[0] is not None:
        raise RuntimeError(f"fixed-batch arm publication failed: {publish_error[0]}")
    dist.barrier()
    if rank == 0:
        if report is None:
            raise RuntimeError("rank zero lost the fixed-batch arm report")
        print(json.dumps(report, indent=2, sort_keys=True))


def _distributed_raise_if_local_probe_error(
    *,
    dist: Any,
    rank: int,
    world_size: int,
    stage: str,
    local_error: BaseException | None,
) -> None:
    """Make rank-local probe validation fail collectively before tensor collectives."""

    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or not 0 <= rank < world_size
        or not isinstance(stage, str)
        or not stage
    ):
        raise ValueError("distributed probe error synchronization arguments are invalid")
    payload: dict[str, object] | None = None
    if local_error is not None:
        payload = {
            "rank": rank,
            "type": type(local_error).__name__,
            "message": str(local_error)[:4096],
        }
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)
    failures: list[dict[str, object]] = []
    for peer_rank, item in enumerate(gathered):
        if item is None:
            continue
        if (
            not isinstance(item, Mapping)
            or set(item) != {"rank", "type", "message"}
            or item["rank"] != peer_rank
            or not isinstance(item["type"], str)
            or not item["type"]
            or not isinstance(item["message"], str)
        ):
            failures.append(
                {
                    "rank": peer_rank,
                    "type": "MalformedDistributedError",
                    "message": repr(item)[:4096],
                }
            )
        else:
            failures.append(dict(item))
    if failures:
        raise RuntimeError(
            f"{stage} failed on one or more ranks before the next tensor collective: "
            f"{json.dumps(failures, sort_keys=True)}"
        )


def _distributed_action_state_sha256(
    local_sha256: str,
    *,
    rank: int,
    world_size: int,
    dist: Any,
) -> str:
    """Bind rank-local FSDP action shards into one ordered distributed digest."""

    _require_sha256("local frozen action state", local_sha256)
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or not 0 <= rank < world_size
    ):
        raise ValueError("distributed action-state digest rank arguments are invalid")
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_sha256)
    for peer_rank, digest in enumerate(gathered):
        _require_sha256(f"rank {peer_rank} frozen action state", digest)
    return _canonical_digest(
        {
            "rank_local_action_state_sha256": gathered,
            "world_size": world_size,
        }
    )


def _select_relation_geometry_source_sample(
    *,
    args: argparse.Namespace,
    stream_plan: Any,
    dataset: Any,
    physical_sidecar: Any,
    task_identity_resolver: Callable[[str], tuple[str, ...] | None],
) -> RelationProbeSampleSelection:
    """Resolve a fixed probe pair before loading the policy or allocating GPU weights."""

    def source_metadata(sample_key: str) -> RelationProbeSampleMetadata:
        task_key = dataset.task_key_by_key(sample_key)
        future = dataset.available_future_transitions_by_key(sample_key)
        target_identity_keys = task_identity_resolver(task_key)
        inventory_identity_keys: tuple[str, ...] = ()
        target_supervised_pixel_counts: tuple[int, ...] | None = None
        if target_identity_keys is not None:
            locator = dataset.locator_by_key(sample_key)
            frame = physical_sidecar(
                locator.segment_index,
                locator.global_index,
            )
            inventory_identity_keys = tuple(frame.identity_keys)
            owner_by_identity = {
                identity_key: owner_index
                for owner_index, identity_key in enumerate(
                    inventory_identity_keys,
                    start=1,
                )
            }
            target_supervised_pixel_counts = tuple(
                sum(
                    int(
                        (
                            (camera.owner_index == owner_by_identity.get(identity_key, -1))
                            & camera.owner_supervised
                        ).sum()
                    )
                    for camera in frame.cameras
                )
                for identity_key in target_identity_keys
            )
        return RelationProbeSampleMetadata(
            sample_key=sample_key,
            task_key=task_key,
            available_future_transitions=future,
            target_identity_keys=target_identity_keys,
            inventory_identity_keys=inventory_identity_keys,
            target_supervised_pixel_counts=target_supervised_pixel_counts,
        )

    return select_relation_geometry_probe_sample(
        selection_start_global_step=args.relation_geometry_fixed_batch_sample_step,
        total_planned_steps=args.total_planned_steps,
        capacity=args.capacity,
        sample_keys_for_global_step=lambda step: tuple(
            transition.sample.sample_key
            for transition in stream_plan.global_batch(step).transitions
        ),
        metadata_for_sample_key=source_metadata,
    )


def _relation_probe_iou_metrics(
    *,
    visual_artifacts: list[dict[str, Any]],
    task_diagnostics: list[dict[str, Any]],
) -> tuple[float, float]:
    """Return macro and task-conditioned IoU from one generated evidence point."""

    if len(visual_artifacts) != 1 or len(task_diagnostics) != 1:
        raise RuntimeError("relation probe requires one local sample per rank")
    row_iou = visual_artifacts[0].get("row_matched_soft_iou")
    target_rows = task_diagnostics[0].get("target_rows")
    if (
        not isinstance(row_iou, list)
        or not row_iou
        or not isinstance(target_rows, list)
        or not target_rows
        or any(
            isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(row_iou)
            for row in target_rows
        )
    ):
        raise RuntimeError("relation probe lacks one materialized task-object row")
    macro_values = [float(value) for value in row_iou if value is not None]
    task_values = [float(row_iou[row]) for row in target_rows if row_iou[row] is not None]
    if not macro_values or len(task_values) != len(target_rows):
        raise RuntimeError("relation probe task object has no measurable ownership target")
    return sum(macro_values) / len(macro_values), sum(task_values) / len(task_values)


def _relation_probe_objective_iou_metrics(
    *,
    objective: Any,
    structural_sensor_valid: Any,
    task_diagnostics: list[dict[str, Any]],
    matched_row_soft_iou_fn: Callable[..., list[float | None]],
) -> tuple[float, float]:
    """Measure the same soft-IoU as visual evidence without writing an image."""

    if len(task_diagnostics) != 1:
        raise RuntimeError("relation depth probe requires one local task diagnostic")
    row_iou = matched_row_soft_iou_fn(
        objective=objective,
        structural_sensor_valid=structural_sensor_valid,
        batch_index=0,
    )
    return _relation_probe_iou_metrics(
        visual_artifacts=[{"row_matched_soft_iou": row_iou}],
        task_diagnostics=task_diagnostics,
    )


def _local_gradient_norm(value: Any, *, name: str, torch_module: Any) -> float:
    if not torch_module.is_tensor(value) or value.numel() <= 0:
        raise RuntimeError(f"{name} gradient snapshot is malformed")
    # The bilinear diagnostic has 6.5M FP32 elements. Squaring in FP32 can
    # overflow or underflow even when every gradient element is finite.
    norm = float(value.detach().double().square().sum().sqrt().item())
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"{name} gradient did not carry positive finite mass: norm={norm!r}")
    return norm


def _all_reduce_external_relation_candidate_gradients(
    *,
    readout: Any,
    candidate_id: str,
    diagnostic_rank: int | None,
    require_positive: bool,
    dist: Any,
    world_size: int,
    torch_module: Any,
) -> float:
    """Average one external candidate's three gradients in one ordered bucket."""

    if not isinstance(candidate_id, str) or not candidate_id:
        raise RuntimeError("external relation candidate ID is missing")
    gradients = []
    for parameter in relation_depth_trainable_parameters(readout):
        gradient = parameter.grad
        if (
            gradient is None
            or gradient.shape != parameter.shape
            or not bool(torch_module.isfinite(gradient).all().item())
        ):
            raise RuntimeError("external relation candidate has a missing or invalid gradient")
        gradients.append(gradient)
    flat = torch_module.cat(tuple(value.reshape(-1) for value in gradients))
    local_norm = float(flat.detach().double().square().sum().sqrt().item())
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for gradient in gradients:
        count = gradient.numel()
        gradient.copy_(flat[offset : offset + count].view_as(gradient))
        offset += count
    if offset != flat.numel():
        raise RuntimeError("external relation gradient bucket was not consumed exactly")
    global_norm = float(flat.detach().double().square().sum().sqrt().item())
    if not math.isfinite(global_norm) or (require_positive and global_norm <= 0):
        raise RuntimeError(
            "external relation distributed candidate "
            f"{candidate_id!r} gradient is invalid: "
            f"global_norm={global_norm!r}; pre-reduction local_norm={local_norm!r}; "
            f"require_positive={require_positive!r}"
        )
    if diagnostic_rank is not None:
        print(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "event": "external_relation_gradient_diagnostic",
                    "global_norm": global_norm,
                    "local_norm": local_norm,
                    "rank": diagnostic_rank,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return global_norm


def _run_external_relation_fixed_batch_arm(
    *,
    args: argparse.Namespace,
    rank: int,
    device: Any,
    dist: Any,
    torch_module: Any,
    policy: Any,
    graph: Any,
    trainable_scope: RelationGeometryTrainableScope,
    sample_selection: RelationProbeSampleSelection,
    stream_plan: Any,
    dataset: Any,
    collate_planned: Callable[[Any], Any],
    build_planned_batch: Callable[..., Any],
    build_continuation_batch: Callable[..., Any],
    context_type: Callable[..., Any],
    run_policy_diagnostic: Callable[..., Any],
    run_observation_diagnostic: Callable[..., Any],
    compose_objective: Callable[..., Any],
    physical_sidecar: Any,
    task_identity_resolver: Callable[..., Any],
    patch_size: int,
    merge_size: int,
    objective_config: Any,
    structural_config: Any,
    derive_subseed_fn: Callable[..., int],
    temporal_batch_seed_fn: Callable[..., int],
    matched_row_soft_iou_fn: Callable[..., list[float | None]],
    render_relation_visuals: Callable[..., list[dict[str, Any]]],
    patch_sha256: str,
    execution_sha256: str,
    implementation_sha256: str,
    model_family_sha256: str,
    dataset_contract_report: Mapping[str, Any],
) -> None:
    """Fit preregistered external readouts on detached native host states."""

    output_path = args.relation_geometry_fixed_batch_output
    visual_root = args.relation_geometry_fixed_batch_visual_root
    arm = args.relation_geometry_fixed_batch_arm
    is_bilinear_probe = arm == RELATION_BILINEAR_PROBE_ARM
    if arm not in {
        RELATION_BILINEAR_PROBE_ARM,
        RELATION_DEPTH_PROBE_ARM,
    }:
        raise RuntimeError("external relation runner received an unsupported arm")
    probe_name = "relation-bilinear" if is_bilinear_probe else "relation-depth"
    curve_names = (
        RELATION_BILINEAR_PROBE_CURVE_NAMES
        if is_bilinear_probe
        else RELATION_DEPTH_PROBE_CURVE_NAMES
    )
    curve_point_count = (
        RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT
        if is_bilinear_probe
        else RELATION_DEPTH_PROBE_CURVE_POINT_COUNT
    )
    update_count = (
        RELATION_BILINEAR_PROBE_UPDATE_COUNT
        if is_bilinear_probe
        else RELATION_DEPTH_PROBE_UPDATE_COUNT
    )
    visual_points = (
        RELATION_BILINEAR_PROBE_VISUAL_POINTS
        if is_bilinear_probe
        else RELATION_DEPTH_PROBE_VISUAL_POINTS
    )
    weight_decay = (
        RELATION_BILINEAR_PROBE_WEIGHT_DECAY
        if is_bilinear_probe
        else RELATION_DEPTH_PROBE_WEIGHT_DECAY
    )
    if (
        args.relation_geometry_fixed_batch_curve_points != curve_point_count
        or output_path is None
        or visual_root is None
    ):
        raise RuntimeError(f"{probe_name} arm lost its preregistered command contract")
    if (
        not isinstance(sample_selection, RelationProbeSampleSelection)
        or sample_selection.selection_start_global_step
        != args.relation_geometry_fixed_batch_sample_step
        or sample_selection.capacity != args.capacity
    ):
        raise RuntimeError(f"{probe_name} arm lost its source-only sample selection")

    sample_step = sample_selection.selected_global_step
    planned = primary_batch = continuation_batch = None
    setup_error: BaseException | None = None
    try:
        planned = build_planned_batch(
            stream_plan,
            dataset,
            optimizer_step=sample_step,
            rank=rank,
            world_size=FULL_WORLD_SIZE,
            gradient_accumulation_steps=1,
            accumulation_index=0,
            device=device,
            dtype=torch_module.bfloat16,
        )
        primary_batch = collate_planned(planned)
        continuation_batch = collate_planned(
            build_continuation_batch(
                planned,
                dataset,
                offset=1,
                device=device,
                dtype=torch_module.bfloat16,
            )
        )
        if primary_batch.routing.batch_size != 1:
            raise RuntimeError(f"{probe_name} probe requires one sample on each rank")
        expected_key = sample_selection.samples_by_rank[rank].sample_key
        if primary_batch.routing.sample_keys != (expected_key,):
            raise RuntimeError(f"{probe_name} local shard differs from source selection")
        if (continuation_batch.controls.reset & continuation_batch.controls.token_valid).any():
            raise RuntimeError(f"{probe_name} two-frame observation crosses an episode reset")
    except BaseException as error:
        setup_error = error
    _distributed_raise_if_local_probe_error(
        dist=dist,
        rank=rank,
        world_size=FULL_WORLD_SIZE,
        stage=f"{probe_name} fixed-observation materialization",
        local_error=setup_error,
    )
    if planned is None or primary_batch is None or continuation_batch is None:
        raise RuntimeError(f"{probe_name} fixed observation vanished after validation")

    global_sample_keys = tuple(sample.sample_key for sample in sample_selection.samples_by_rank)
    temporal_seed = temporal_batch_seed_fn(
        parent_seed=args.seed,
        comparison_id=FULL_COMPARISON_ID,
        optimizer_step=sample_step,
        sample_keys=global_sample_keys,
    )
    capacity_seeds = tuple(
        derive_subseed_fn(temporal_seed, "capacity-censor", key)
        for key in primary_batch.routing.sample_keys
    )
    forward_seed = derive_subseed_fn(
        temporal_seed,
        "relation-fixed-forward",
        primary_batch.routing.sample_keys[0],
    )
    probe_seed = derive_subseed_fn(
        temporal_seed,
        "relation-depth-probe-bank",
    )

    total_started = time.perf_counter()
    torch_module.cuda.reset_peak_memory_stats(device)
    before_bias = _moe_routing_bias_snapshot(policy, torch_module=torch_module)
    primary_forward = continuation_context = captures = None
    depth_capture: LingBotRelationDepthCapture | None = None
    capture_error: BaseException | None = None
    try:
        torch_module.manual_seed(forward_seed)
        torch_module.cuda.manual_seed_all(forward_seed)
        depth_capture = LingBotRelationDepthCapture(policy)
        with depth_capture:
            primary_forward = run_policy_diagnostic(
                policy,
                model_inputs=primary_batch.model_inputs,
                context=context_type(
                    controls=primary_batch.controls,
                    modalities=primary_batch.modalities,
                ),
            )
            posterior = primary_forward.context.posterior_state
            if posterior is None:
                raise RuntimeError(f"{probe_name} primary forward omitted posterior state")
            continuation_context = context_type(
                controls=continuation_batch.controls,
                previous_state=posterior,
                previous_state_valid=torch_module.ones(
                    primary_batch.routing.batch_size,
                    dtype=torch_module.bool,
                    device=device,
                ),
                modalities=continuation_batch.modalities,
            )
            continuation_context = run_observation_diagnostic(
                policy,
                model_inputs=continuation_batch.model_inputs,
                context=continuation_context,
            )
        captures = depth_capture.snapshot(expected_forward_count=2)
        _reset_moe_probe_counters(policy, torch_module=torch_module)
    except BaseException as error:
        capture_error = error
    _distributed_raise_if_local_probe_error(
        dist=dist,
        rank=rank,
        world_size=FULL_WORLD_SIZE,
        stage=f"{probe_name} frozen host capture",
        local_error=capture_error,
    )
    if (
        primary_forward is None
        or continuation_context is None
        or captures is None
        or depth_capture is None
    ):
        raise RuntimeError(f"{probe_name} host evidence vanished after validation")

    contexts = (primary_forward.context, continuation_context)
    production_relations = tuple(context.relation_output for context in contexts)
    if any(value is None for value in production_relations):
        raise RuntimeError(f"{probe_name} host omitted a production relation output")
    fixed_row_bindings: tuple[Any, ...] = ((),)
    production_objective = None
    depth_inputs_by_surface: dict[str, tuple[Any, ...]] = {}
    bank = candidates = initialization_sha256 = None
    probe_setup_error: BaseException | None = None
    try:
        production_objective = compose_objective(
            official_policy_loss=primary_forward.official_total_loss,
            requests_by_time=(
                primary_batch.structural_target_requests,
                continuation_batch.structural_target_requests,
            ),
            model_inputs_by_time=(
                primary_batch.model_inputs,
                continuation_batch.model_inputs,
            ),
            relations=production_relations,
            physical_sidecar=physical_sidecar,
            capacity=args.capacity,
            task_identity_resolver=task_identity_resolver,
            patch_size=patch_size,
            merge_size=merge_size,
            objective_config=objective_config,
            structural_config=structural_config,
            require_policy_loss_grad=False,
            minimum_supervised_fraction=args.minimum_supervised_fraction,
            capacity_seeds=capacity_seeds,
            prior_row_bindings_by_batch=fixed_row_bindings,
        )
        fixed_row_bindings = production_objective.row_bindings_by_batch
        if len(fixed_row_bindings) != 1 or not fixed_row_bindings[0]:
            raise RuntimeError(f"{probe_name} production gauge contains no physical objects")
        for surface_name, values in captures.items():
            depth_inputs_by_surface[surface_name] = tuple(
                relation_depth_inputs(
                    hidden.float(),
                    context=context,
                    capacity=args.capacity,
                )
                for hidden, context in zip(values, contexts, strict=True)
            )
        host_layer_count = len(depth_capture.surfaces) and (
            depth_capture.surfaces[-1].layer_index + 1
        )
        if is_bilinear_probe:
            bank, candidates, initialization_sha256 = build_relation_bilinear_probe_bank(
                host_width=graph.config.host_width,
                seed=probe_seed,
                device=device,
                dtype=torch_module.float32,
            )
        else:
            bank, candidates, initialization_sha256 = build_relation_depth_probe_bank(
                host_width=graph.config.host_width,
                num_layers=host_layer_count,
                seed=probe_seed,
                device=device,
                dtype=torch_module.float32,
            )
    except BaseException as error:
        probe_setup_error = error
    _distributed_raise_if_local_probe_error(
        dist=dist,
        rank=rank,
        world_size=FULL_WORLD_SIZE,
        stage=f"{probe_name} external probe setup",
        local_error=probe_setup_error,
    )
    if (
        production_objective is None
        or bank is None
        or candidates is None
        or initialization_sha256 is None
    ):
        raise RuntimeError(f"{probe_name} external probes vanished after validation")

    def validated_readout(candidate_id: str) -> SharedRelationReadout:
        readout = bank[candidate_id]
        if not isinstance(readout, SharedRelationReadout):
            raise RuntimeError(f"{probe_name} candidate readout has the wrong type")
        return readout

    def candidate_surface_name(
        candidate: RelationBilinearCandidate | RelationDepthCandidate,
    ) -> str:
        if isinstance(candidate, RelationBilinearCandidate):
            return "final"
        if isinstance(candidate, RelationDepthCandidate):
            return candidate.surface.name
        raise TypeError(f"{probe_name} candidate descriptor has the wrong type")

    parameter_groups = [
        {
            "params": relation_depth_trainable_parameters(
                validated_readout(candidate.candidate_id)
            ),
            "lr": candidate.learning_rate,
            "weight_decay": weight_decay,
        }
        for candidate in candidates
    ]
    probe_optimizer = torch_module.optim.AdamW(parameter_groups)
    probe_scheduler = torch_module.optim.lr_scheduler.CosineAnnealingLR(
        probe_optimizer,
        T_max=update_count,
        eta_min=0.0,
    )
    probe_optimizer.zero_grad(set_to_none=True)
    local_candidate_reports: dict[str, dict[str, Any]] = {
        candidate.candidate_id: {
            "rank": rank,
            "curves": {name: [] for name in curve_names},
            "gradient_norm_at_first_update": None,
            "visual_artifacts_by_point": [],
            "evaluation_times_s": [],
        }
        for candidate in candidates
    }

    for curve_point in range(curve_point_count):
        ownership_losses = []
        point_matrix: list[list[float]] = []
        point_error: BaseException | None = None
        try:
            for candidate in candidates:
                candidate_started = time.perf_counter()
                readout = validated_readout(candidate.candidate_id)
                surface_name = candidate_surface_name(candidate)
                relation_outputs = tuple(
                    inputs.read(readout) for inputs in depth_inputs_by_surface[surface_name]
                )
                composed = compose_objective(
                    official_policy_loss=primary_forward.official_total_loss,
                    requests_by_time=(
                        primary_batch.structural_target_requests,
                        continuation_batch.structural_target_requests,
                    ),
                    model_inputs_by_time=(
                        primary_batch.model_inputs,
                        continuation_batch.model_inputs,
                    ),
                    relations=relation_outputs,
                    physical_sidecar=physical_sidecar,
                    capacity=args.capacity,
                    task_identity_resolver=task_identity_resolver,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    objective_config=objective_config,
                    structural_config=structural_config,
                    require_policy_loss_grad=False,
                    minimum_supervised_fraction=args.minimum_supervised_fraction,
                    capacity_seeds=capacity_seeds,
                    prior_row_bindings_by_batch=fixed_row_bindings,
                )
                if composed.row_bindings_by_batch != fixed_row_bindings:
                    raise RuntimeError(f"{probe_name} candidate changed the production row gauge")
                normalized = composed.objective.normalized_terms
                ownership = normalized.get("set/ownership")
                ownership_nll = normalized.get("set/ownership_nll")
                if any(
                    value is None
                    or value.ndim != 0
                    or not value.requires_grad
                    or not bool(torch_module.isfinite(value).item())
                    for value in (ownership, ownership_nll)
                ):
                    raise RuntimeError(f"{probe_name} candidate lost ownership supervision")
                task_diagnostics = list(build_task_row_diagnostics(composed))
                validate_task_row_diagnostics(
                    task_diagnostics,
                    expected_batch_size=primary_batch.routing.batch_size,
                )
                macro_iou, task_iou = _relation_probe_objective_iou_metrics(
                    objective=composed,
                    structural_sensor_valid=relation_outputs[0].structural_valid,
                    task_diagnostics=task_diagnostics,
                    matched_row_soft_iou_fn=matched_row_soft_iou_fn,
                )
                point_values = {
                    "ownership": float(ownership.detach().item()),
                    "ownership_nll": float(ownership_nll.detach().item()),
                    "macro_soft_iou": macro_iou,
                    "task_soft_iou": task_iou,
                }
                if any(not math.isfinite(value) or value < 0 for value in point_values.values()):
                    raise FloatingPointError(f"{probe_name} candidate emitted an invalid metric")
                report = local_candidate_reports[candidate.candidate_id]
                for name in curve_names:
                    curves = report.get("curves")
                    if not isinstance(curves, dict):
                        raise RuntimeError(f"{probe_name} candidate curve report is malformed")
                    curve = curves.get(name)
                    if not isinstance(curve, list):
                        raise RuntimeError(f"{probe_name} candidate curve is malformed")
                    curve.append(point_values[name])
                if curve_point in visual_points:
                    candidate_root = Path(visual_root) / candidate.candidate_id
                    artifacts = render_relation_visuals(
                        output_root=candidate_root,
                        global_step=curve_point + 1,
                        rank=rank,
                        host_items=planned.training.host_items,
                        model_inputs=primary_batch.model_inputs,
                        objective=composed,
                        structural_sensor_valid=relation_outputs[0].structural_valid,
                        sample_keys=primary_batch.routing.sample_keys,
                        merge_size=merge_size,
                    )
                    visual_reports = report.get("visual_artifacts_by_point")
                    if not isinstance(visual_reports, list):
                        raise RuntimeError(f"{probe_name} candidate visual report is malformed")
                    visual_reports.append(
                        {
                            "curve_point": curve_point,
                            "artifacts": [
                                {
                                    **artifact,
                                    "path": (
                                        Path(candidate.candidate_id) / artifact["path"]
                                    ).as_posix(),
                                }
                                for artifact in artifacts
                            ],
                        }
                    )
                evaluation_times = report.get("evaluation_times_s")
                if not isinstance(evaluation_times, list):
                    raise RuntimeError(f"{probe_name} candidate timing report is malformed")
                evaluation_times.append(time.perf_counter() - candidate_started)
                ownership_losses.append(ownership)
                point_matrix.append([point_values[name] for name in curve_names])
        except BaseException as error:
            point_error = error
        _distributed_raise_if_local_probe_error(
            dist=dist,
            rank=rank,
            world_size=FULL_WORLD_SIZE,
            stage=f"{probe_name} curve point {curve_point} evidence",
            local_error=point_error,
        )
        if len(ownership_losses) != len(candidates) or len(point_matrix) != len(candidates):
            raise RuntimeError(f"{probe_name} candidate evidence vanished after validation")

        reduced_metrics = torch_module.tensor(
            point_matrix,
            dtype=torch_module.float64,
            device=device,
        )
        dist.all_reduce(reduced_metrics, op=dist.ReduceOp.SUM)
        reduced_metrics.div_(FULL_WORLD_SIZE)
        if rank == 0 and (curve_point % 10 == 0 or curve_point in visual_points):
            best_task_index = int(reduced_metrics[:, 3].argmax().item())
            print(
                json.dumps(
                    {
                        "event": f"{probe_name.replace('-', '_')}_probe_progress",
                        "curve_point": curve_point,
                        "best_task_candidate": candidates[best_task_index].candidate_id,
                        "best_task_soft_iou": float(reduced_metrics[best_task_index, 3].item()),
                        "best_task_ownership": float(reduced_metrics[best_task_index, 0].item()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if curve_point < update_count:
            backward_error: BaseException | None = None
            try:
                for ownership_loss in ownership_losses:
                    ownership_loss.backward()
            except BaseException as error:
                backward_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                stage=f"{probe_name} curve point {curve_point} backward",
                local_error=backward_error,
            )
            gradient_validation_error: BaseException | None = None
            try:
                for candidate in candidates:
                    for parameter in relation_depth_trainable_parameters(
                        validated_readout(candidate.candidate_id)
                    ):
                        if parameter.grad is None or not bool(
                            torch_module.isfinite(parameter.grad).all().item()
                        ):
                            raise RuntimeError(
                                f"{probe_name} candidate has an invalid local gradient"
                            )
            except BaseException as error:
                gradient_validation_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                stage=f"{probe_name} curve point {curve_point} gradient validation",
                local_error=gradient_validation_error,
            )
            gradient_norms = [
                _all_reduce_external_relation_candidate_gradients(
                    readout=validated_readout(candidate.candidate_id),
                    candidate_id=candidate.candidate_id,
                    diagnostic_rank=rank if curve_point == 0 else None,
                    require_positive=curve_point == 0,
                    dist=dist,
                    world_size=FULL_WORLD_SIZE,
                    torch_module=torch_module,
                )
                for candidate in candidates
            ]
            if curve_point == 0:
                for candidate, norm in zip(candidates, gradient_norms, strict=True):
                    local_candidate_reports[candidate.candidate_id][
                        "gradient_norm_at_first_update"
                    ] = norm
            update_error: BaseException | None = None
            try:
                probe_optimizer.step()
                probe_scheduler.step()
                probe_optimizer.zero_grad(set_to_none=True)
            except BaseException as error:
                update_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                stage=f"{probe_name} curve point {curve_point} optimizer update",
                local_error=update_error,
            )

    final_error: BaseException | None = None
    bias_unchanged = False
    try:
        if any(
            report["gradient_norm_at_first_update"] is None
            for report in local_candidate_reports.values()
        ):
            raise RuntimeError(f"{probe_name} probe omitted first-update gradient evidence")
        if any(abs(group["lr"]) > 1e-15 for group in probe_optimizer.param_groups):
            raise RuntimeError(f"{probe_name} cosine schedule did not finish at zero")
        bias_unchanged = _moe_routing_bias_matches(
            policy,
            before_bias,
            torch_module=torch_module,
        )
        if not bias_unchanged:
            raise RuntimeError(f"{probe_name} probe changed LingBot MoE routing bias")
    except BaseException as error:
        final_error = error
    _distributed_raise_if_local_probe_error(
        dist=dist,
        rank=rank,
        world_size=FULL_WORLD_SIZE,
        stage=f"{probe_name} final local validation",
        local_error=final_error,
    )

    local_payload = {
        "rank": rank,
        "forward_seed": forward_seed,
        "probe_seed": probe_seed,
        "frame_sample_keys": [
            primary_batch.routing.sample_keys[0],
            continuation_batch.routing.sample_keys[0],
        ],
        "frame_source_digests": [
            primary_batch.source_digest,
            continuation_batch.source_digest,
        ],
        "row_bindings": [list(value) for value in fixed_row_bindings[0]],
        "official_action": float(primary_forward.official_action_loss.detach().float().item()),
        "candidate_initialization_sha256": initialization_sha256,
        "candidate_reports": [
            local_candidate_reports[candidate.candidate_id] for candidate in candidates
        ],
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(device)),
        "total_time_s": time.perf_counter() - total_started,
    }
    rank_payloads: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
    dist.all_gather_object(rank_payloads, local_payload)

    publish_error: list[str | None] = [None]
    report: dict[str, object] | None = None
    if rank == 0:
        try:
            if [item["rank"] for item in rank_payloads] != [0, 1]:
                raise RuntimeError(f"{probe_name} rank payload order changed")
            if len({item["candidate_initialization_sha256"] for item in rank_payloads}) != 1:
                raise RuntimeError(f"{probe_name} candidate initialization differs by rank")
            candidate_reports = []
            trainable_numel = (
                graph.config.host_width * graph.config.host_width + graph.config.host_width + 1
            )
            for candidate_index, candidate in enumerate(candidates):
                rank_reports = [
                    item["candidate_reports"][candidate_index] for item in rank_payloads
                ]
                global_curves = {
                    name: [
                        sum(rank_report["curves"][name][point] for rank_report in rank_reports)
                        / FULL_WORLD_SIZE
                        for point in range(curve_point_count)
                    ]
                    for name in curve_names
                }
                recovery = relation_depth_recovery_summary(
                    global_curves=global_curves,
                    rank_task_curves=[
                        rank_report["curves"]["task_soft_iou"] for rank_report in rank_reports
                    ],
                )
                candidate_reports.append(
                    {
                        "candidate": candidate.as_dict(),
                        "trainable_numel": trainable_numel,
                        "global_curves": global_curves,
                        "rank_reports": rank_reports,
                        "recovery": recovery,
                    }
                )
            reference_source = (
                RELATION_BILINEAR_PROBE_GLOBAL_REFERENCES
                if is_bilinear_probe
                else RELATION_DEPTH_PROBE_GLOBAL_REFERENCES
            )
            references = {
                "point_zero": dict(
                    cast(
                        Mapping[str, float],
                        reference_source["point_zero"],
                    )
                ),
                "structural_full_host_point_40": dict(
                    cast(
                        Mapping[str, float],
                        reference_source["structural_full_host_point_40"],
                    )
                ),
                "rank_task_soft_iou": [
                    dict(value)
                    for value in cast(
                        Sequence[Mapping[str, float]],
                        reference_source["rank_task_soft_iou"],
                    )
                ],
            }
            provenance = {
                "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "patch_sha256": patch_sha256,
                "execution_contract_sha256": execution_sha256,
                "implementation_sha256": implementation_sha256,
                "model_family_sha256": model_family_sha256,
                "plan_sha256": stream_plan.plan_sha256,
                "dataset_manifest_sha256": dataset_contract_report["manifest_sha256"],
                "physical_sidecar_manifest_sha256": physical_sidecar.manifest_sha256,
                "seed": args.seed,
                "fixed_sample_global_step": sample_step,
                "sample_selection": sample_selection.as_dict(),
                "forward_seed_by_rank": [int(item["forward_seed"]) for item in rank_payloads],
                "probe_seed_by_rank": [int(item["probe_seed"]) for item in rank_payloads],
                "frame_sample_keys_by_rank": [
                    list(item["frame_sample_keys"]) for item in rank_payloads
                ],
                "frame_source_digests_by_rank": [
                    list(item["frame_source_digests"]) for item in rank_payloads
                ],
                "row_bindings_by_rank": [list(item["row_bindings"]) for item in rank_payloads],
                "official_action_by_rank": [
                    float(item["official_action"]) for item in rank_payloads
                ],
                "candidate_initialization_sha256": initialization_sha256,
                "host_width": graph.config.host_width,
                "host_layer_count": depth_capture.surfaces[-1].layer_index + 1,
                "surfaces": [surface.as_dict() for surface in depth_capture.surfaces],
                "capture": {
                    "intermediate_hook": ("next_layer_compute_kqv_input_after_block_and_deepstack"),
                    "final_hook": "post_final_norm",
                    "forward_count": 2,
                    "feature_dtype": "float32",
                    "policy_grad_enabled": False,
                    "prediction_queries": "absent",
                },
                "objective": {
                    "optimized_term": "set/ownership",
                    "observed_terms": list(curve_names),
                    "window": "fixed_two_frame_detached_host",
                    "labels_are_loss_side_only": True,
                    "row_gauge": "production_point_zero_then_frozen",
                    "official_policy_loss": "observed_not_optimized",
                },
                "optimizer": {
                    "algorithm": "torch.optim.AdamW",
                    "learning_rate_hex_grid": [
                        value.hex()
                        for value in (
                            RELATION_BILINEAR_PROBE_LEARNING_RATES
                            if is_bilinear_probe
                            else RELATION_DEPTH_PROBE_LEARNING_RATES
                        )
                    ],
                    "weight_decay_hex": weight_decay.hex(),
                    "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
                    "warmup_updates": 0,
                    "update_count": update_count,
                    "distributed_gradient": "rank_sum_div_world_size",
                },
                "global_references": references,
            }
            schema = (
                RELATION_BILINEAR_PROBE_SCHEMA if is_bilinear_probe else RELATION_DEPTH_PROBE_SCHEMA
            )
            subject_sha256 = (
                relation_bilinear_probe_subject(
                    provenance,
                    curve_point_count=curve_point_count,
                )
                if is_bilinear_probe
                else relation_depth_probe_subject(
                    provenance,
                    curve_point_count=curve_point_count,
                )
            )
            report = {
                "schema": schema,
                "status": "PASS",
                "arm": arm,
                "subject_sha256": subject_sha256,
                "provenance": provenance,
                "policy_parameter_boundary": trainable_scope.as_dict(),
                "candidate_initialization_sha256": initialization_sha256,
                "curve_point_count": curve_point_count,
                "optimizer_update_count": update_count,
                "candidates": candidate_reports,
                "maximum_peak_reserved_bytes": max(
                    int(item["peak_reserved_bytes"]) for item in rank_payloads
                ),
                "total_time_s": max(float(item["total_time_s"]) for item in rank_payloads),
            }
            if is_bilinear_probe:
                report["mode_decisions"] = relation_bilinear_decisions(candidate_reports)
                validate_relation_bilinear_probe_report(report)
            else:
                report["depth_decisions"] = relation_depth_decisions(
                    candidate_reports,
                    num_layers=depth_capture.surfaces[-1].layer_index + 1,
                )
                validate_relation_depth_probe_report(report)
            output = Path(output_path)
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"{probe_name} report already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_text_durable(
                output,
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
        except BaseException as error:
            publish_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(publish_error, src=0)
    if publish_error[0] is not None:
        raise RuntimeError(f"{probe_name} publication failed: {publish_error[0]}")
    dist.barrier()
    if rank == 0:
        if report is None:
            raise RuntimeError(f"rank zero lost the {probe_name} report")
        print(json.dumps(report, indent=2, sort_keys=True))


def _run_relation_geometry_fixed_batch_arm(
    *,
    args: argparse.Namespace,
    rank: int,
    device: Any,
    dist: Any,
    torch_module: Any,
    policy: Any,
    graph: Any,
    optimizer: Any,
    optimizer_contract: Any,
    trainable_scope: RelationGeometryTrainableScope,
    sample_selection: RelationProbeSampleSelection,
    stream_plan: Any,
    dataset: Any,
    collate_planned: Callable[[Any], Any],
    build_planned_batch: Callable[..., Any],
    build_continuation_batch: Callable[..., Any],
    run_relation_objective: Callable[..., Any],
    physical_sidecar: Any,
    task_identity_resolver: Callable[..., Any],
    patch_size: int,
    merge_size: int,
    objective_config: Any,
    structural_config: Any,
    derive_subseed_fn: Callable[..., int],
    temporal_batch_seed_fn: Callable[..., int],
    render_relation_visuals: Callable[..., list[dict[str, Any]]],
    patch_sha256: str,
    execution_sha256: str,
    implementation_sha256: str,
    model_family_sha256: str,
    dataset_contract_report: Mapping[str, Any],
) -> None:
    """Fit only ownership on one fixed observation and publish no checkpoint."""

    arm = args.relation_geometry_fixed_batch_arm
    curve_points = args.relation_geometry_fixed_batch_curve_points
    output_path = args.relation_geometry_fixed_batch_output
    visual_root = args.relation_geometry_fixed_batch_visual_root
    if arm not in RELATION_GEOMETRY_FIXED_BATCH_ARMS or output_path is None or visual_root is None:
        raise RuntimeError("relation-geometry arm lost its validated command contract")
    if (
        not isinstance(sample_selection, RelationProbeSampleSelection)
        or sample_selection.selection_start_global_step
        != args.relation_geometry_fixed_batch_sample_step
        or sample_selection.capacity != args.capacity
    ):
        raise RuntimeError("relation-geometry arm lost its source-only sample selection")
    selection_payload = sample_selection.as_dict()
    sample_step = sample_selection.selected_global_step
    planned = primary_batch = continuation_batch = None
    setup_error: BaseException | None = None
    try:
        planned = build_planned_batch(
            stream_plan,
            dataset,
            optimizer_step=sample_step,
            rank=rank,
            world_size=FULL_WORLD_SIZE,
            gradient_accumulation_steps=1,
            accumulation_index=0,
            device=device,
            dtype=torch_module.bfloat16,
        )
        primary_batch = collate_planned(planned)
        continuation_batch = collate_planned(
            build_continuation_batch(
                planned,
                dataset,
                offset=1,
                device=device,
                dtype=torch_module.bfloat16,
            )
        )
        if primary_batch.routing.batch_size != 1:
            raise RuntimeError("relation probe requires one sample on each of two ranks")
        expected_sample_key = sample_selection.samples_by_rank[rank].sample_key
        if primary_batch.routing.sample_keys != (expected_sample_key,):
            raise RuntimeError("relation local shard differs from selected global sample")
        if (continuation_batch.controls.reset & continuation_batch.controls.token_valid).any():
            raise RuntimeError("relation probe two-frame observation crosses an episode reset")
    except BaseException as error:
        setup_error = error
    _distributed_raise_if_local_probe_error(
        dist=dist,
        rank=rank,
        world_size=FULL_WORLD_SIZE,
        stage="relation fixed-observation materialization",
        local_error=setup_error,
    )
    if planned is None or primary_batch is None or continuation_batch is None:
        raise RuntimeError("relation fixed observation vanished after collective validation")

    global_sample_keys = tuple(sample.sample_key for sample in sample_selection.samples_by_rank)
    temporal_seed = temporal_batch_seed_fn(
        parent_seed=args.seed,
        comparison_id=FULL_COMPARISON_ID,
        optimizer_step=sample_step,
        sample_keys=global_sample_keys,
    )
    capacity_seeds = tuple(
        derive_subseed_fn(temporal_seed, "capacity-censor", key)
        for key in primary_batch.routing.sample_keys
    )
    forward_seed = derive_subseed_fn(
        temporal_seed,
        "relation-fixed-forward",
        primary_batch.routing.sample_keys[0],
    )

    curve_names = (
        "ownership",
        "ownership_nll",
        "macro_soft_iou",
        "task_soft_iou",
        "action",
    )
    local_curves: dict[str, list[float]] = {name: [] for name in curve_names}
    global_curves: dict[str, list[float]] = {name: [] for name in curve_names}
    task_diagnostics_by_point: list[list[dict[str, Any]]] = []
    visual_artifacts_by_point: list[list[dict[str, Any]]] = []
    step_times: list[float] = []
    peak_reserved_bytes = 0
    fixed_row_bindings: tuple[Any, ...] = ((),)
    gradient_probe: dict[str, Any] | None = None
    before_bias = _moe_routing_bias_snapshot(policy, torch_module=torch_module)
    optimizer.zero_grad(set_to_none=True)
    total_started = time.perf_counter()
    for curve_index in range(curve_points):
        torch_module.manual_seed(forward_seed)
        torch_module.cuda.manual_seed_all(forward_seed)
        torch_module.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = run_relation_objective(
            policy,
            graph=graph,
            batches=(primary_batch, continuation_batch),
            previous_state=None,
            previous_state_valid=None,
            physical_sidecar=physical_sidecar,
            capacity=args.capacity,
            task_identity_resolver=task_identity_resolver,
            patch_size=patch_size,
            merge_size=merge_size,
            objective_config=objective_config,
            structural_config=structural_config,
            minimum_supervised_fraction=args.minimum_supervised_fraction,
            capacity_seeds=capacity_seeds,
            prior_row_bindings_by_batch=fixed_row_bindings,
        )
        ownership_loss = None
        task_diagnostics = None
        visual_artifacts = None
        point_values = None
        evidence_error: BaseException | None = None
        try:
            observed_row_bindings = result.objective.row_bindings_by_batch
            if not isinstance(observed_row_bindings, tuple) or len(observed_row_bindings) != 1:
                raise RuntimeError("relation probe objective returned malformed row bindings")
            if curve_index == 0:
                fixed_row_bindings = observed_row_bindings
            elif observed_row_bindings != fixed_row_bindings:
                raise RuntimeError("relation probe changed its frozen row gauge")
            if not fixed_row_bindings[0]:
                raise RuntimeError(
                    "relation probe selected an observation with no physical objects"
                )

            normalized = result.objective.objective.normalized_terms
            ownership_loss = normalized.get("set/ownership")
            ownership_nll = normalized.get("set/ownership_nll")
            action_loss = result.objective.objective.family_terms.get("action")
            if any(
                value is None
                or value.ndim != 0
                or not value.requires_grad
                or not bool(torch_module.isfinite(value).item())
                for value in (ownership_loss, ownership_nll)
            ):
                raise RuntimeError("relation probe lost an attached ownership diagnostic")
            if (
                action_loss is None
                or action_loss.ndim != 0
                or not bool(torch_module.isfinite(action_loss).item())
            ):
                raise RuntimeError("relation probe lost its observed action diagnostic")
            relation = result.primary.context.relation_output
            if relation is None:
                raise RuntimeError("relation probe forward omitted its relation output")
            task_diagnostics = list(build_task_row_diagnostics(result.objective))
            validate_task_row_diagnostics(
                task_diagnostics,
                expected_batch_size=primary_batch.routing.batch_size,
            )
            visual_artifacts = render_relation_visuals(
                output_root=Path(visual_root),
                global_step=curve_index + 1,
                rank=rank,
                host_items=planned.training.host_items,
                model_inputs=primary_batch.model_inputs,
                objective=result.objective,
                structural_sensor_valid=relation.structural_valid,
                sample_keys=primary_batch.routing.sample_keys,
                merge_size=merge_size,
            )
            macro_iou, task_iou = _relation_probe_iou_metrics(
                visual_artifacts=visual_artifacts,
                task_diagnostics=task_diagnostics,
            )
            point_values = {
                "ownership": float(ownership_loss.detach().float().item()),
                "ownership_nll": float(ownership_nll.detach().float().item()),
                "macro_soft_iou": macro_iou,
                "task_soft_iou": task_iou,
                "action": float(action_loss.detach().float().item()),
            }
            for name, value in point_values.items():
                if not math.isfinite(value) or value < 0:
                    raise FloatingPointError(f"relation probe produced invalid {name}")
        except BaseException as error:
            evidence_error = error
        _distributed_raise_if_local_probe_error(
            dist=dist,
            rank=rank,
            world_size=FULL_WORLD_SIZE,
            stage=f"relation curve point {curve_index} evidence",
            local_error=evidence_error,
        )
        if (
            ownership_loss is None
            or task_diagnostics is None
            or visual_artifacts is None
            or point_values is None
        ):
            raise RuntimeError("relation point evidence vanished after collective validation")
        for name, value in point_values.items():
            local_curves[name].append(value)
        reduced = torch_module.tensor(
            [point_values[name] for name in curve_names],
            dtype=torch_module.float64,
            device=device,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced = reduced / FULL_WORLD_SIZE
        for index, name in enumerate(curve_names):
            global_curves[name].append(float(reduced[index].item()))
        task_diagnostics_by_point.append(task_diagnostics)
        visual_artifacts_by_point.append(visual_artifacts)

        if curve_index < curve_points - 1:
            ownership_loss.backward()
            gradient_error: BaseException | None = None
            try:
                if curve_index == 0:
                    relation_gradient = _parameter_gradient_snapshot(
                        graph.relation_readout.projection.weight,
                        name="ownership relation projection",
                        torch_module=torch_module,
                    )
                    gradient_probe = {
                        "relation_projection_norm": _local_gradient_norm(
                            relation_gradient,
                            name="ownership relation projection",
                            torch_module=torch_module,
                        ),
                        "object_query_norm": None,
                        "shared_host_norms": {},
                    }
                    if arm == "structural_full_host":
                        object_query_gradient = _parameter_gradient_snapshot(
                            graph.object_queries,
                            name="ownership object queries",
                            torch_module=torch_module,
                        )
                        gradient_probe["object_query_norm"] = _local_gradient_norm(
                            object_query_gradient,
                            name="ownership object queries",
                            torch_module=torch_module,
                        )
                        gradient_probe["shared_host_norms"] = {
                            depth: _local_gradient_norm(
                                _parameter_gradient_snapshot(
                                    parameter,
                                    name=f"ownership {depth} shared host",
                                    torch_module=torch_module,
                                ),
                                name=f"ownership {depth} shared host",
                                torch_module=torch_module,
                            )
                            for depth, (_path, parameter) in (
                                _predictive_host_gradient_parameters(policy).items()
                            )
                        }
            except BaseException as error:
                gradient_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                stage=f"relation curve point {curve_index} gradient evidence",
                local_error=gradient_error,
            )
            successful_step, gradient_metrics = _optimizer_attempt(
                policy=policy,
                optimizer=optimizer,
                global_step=curve_index,
                max_grad_norm=args.max_grad_norm,
                device=device,
                dist=dist,
                torch_module=torch_module,
            )
            update_error: BaseException | None = None
            try:
                if successful_step != curve_index + 1:
                    raise RuntimeError("relation probe optimizer update overflowed or was skipped")
                if float(gradient_metrics.get("native_graph_norm", 0.0)) <= 0:
                    raise RuntimeError("ownership loss missed the installed native graph")
            except BaseException as error:
                update_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                stage=f"relation curve point {curve_index} optimizer validation",
                local_error=update_error,
            )
        counter_error: BaseException | None = None
        try:
            _reset_moe_probe_counters(policy, torch_module=torch_module)
        except BaseException as error:
            counter_error = error
        _distributed_raise_if_local_probe_error(
            dist=dist,
            rank=rank,
            world_size=FULL_WORLD_SIZE,
            stage=f"relation curve point {curve_index} counter reset",
            local_error=counter_error,
        )
        step_times.append(time.perf_counter() - started)
        peak_reserved_bytes = max(
            peak_reserved_bytes,
            int(torch_module.cuda.max_memory_reserved(device)),
        )

    total_time_s = time.perf_counter() - total_started
    final_local_error: BaseException | None = None
    bias_unchanged = False
    try:
        if gradient_probe is None:
            raise RuntimeError("relation probe omitted its first-update gradient evidence")
        bias_unchanged = _moe_routing_bias_matches(
            policy,
            before_bias,
            torch_module=torch_module,
        )
    except BaseException as error:
        final_local_error = error
    _distributed_raise_if_local_probe_error(
        dist=dist,
        rank=rank,
        world_size=FULL_WORLD_SIZE,
        stage="relation final local validation",
        local_error=final_local_error,
    )
    local_bias_unchanged = torch_module.tensor(
        int(bias_unchanged),
        dtype=torch_module.int32,
        device=device,
    )
    dist.all_reduce(local_bias_unchanged, op=dist.ReduceOp.MIN)
    if not bool(local_bias_unchanged.item()):
        raise RuntimeError("relation probe changed LingBot MoE routing bias")
    local_report = {
        "rank": rank,
        "frame_sample_keys": [
            primary_batch.routing.sample_keys[0],
            continuation_batch.routing.sample_keys[0],
        ],
        "frame_source_digests": [
            primary_batch.source_digest,
            continuation_batch.source_digest,
        ],
        "forward_seed": forward_seed,
        "row_bindings": [list(value) for value in fixed_row_bindings[0]],
        "curves": local_curves,
        "task_diagnostics_by_point": task_diagnostics_by_point,
        "visual_artifacts_by_point": visual_artifacts_by_point,
        "gradient_probe": gradient_probe,
        "step_times_s": step_times,
        "peak_reserved_bytes": peak_reserved_bytes,
    }
    rank_reports: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
    dist.all_gather_object(rank_reports, local_report)
    total_times: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
    dist.all_gather_object(total_times, total_time_s)

    publish_error: list[str | None] = [None]
    report: dict[str, object] | None = None
    if rank == 0:
        try:
            provenance = {
                "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "patch_sha256": patch_sha256,
                "execution_contract_sha256": execution_sha256,
                "implementation_sha256": implementation_sha256,
                "model_family_sha256": model_family_sha256,
                "plan_sha256": stream_plan.plan_sha256,
                "dataset_manifest_sha256": dataset_contract_report["manifest_sha256"],
                "physical_sidecar_manifest_sha256": physical_sidecar.manifest_sha256,
                "seed": args.seed,
                "fixed_sample_global_step": sample_step,
                "sample_selection": selection_payload,
                "forward_seed_by_rank": [int(item["forward_seed"]) for item in rank_reports],
                "frame_sample_keys_by_rank": [
                    list(item["frame_sample_keys"]) for item in rank_reports
                ],
                "frame_source_digests_by_rank": [
                    list(item["frame_source_digests"]) for item in rank_reports
                ],
                "objective": {
                    "optimized_term": "set/ownership",
                    "observed_terms": list(curve_names),
                    "window": "fixed_two_frame_local_bptt",
                    "labels_are_loss_side_only": True,
                    "row_gauge": "initial_assignment_then_frozen",
                    "forward_randomness": "fixed_per_rank_torch_seed",
                    "official_policy_loss": "observed_not_optimized",
                    "predictive_queries": "absent",
                },
                "optimizer": {
                    "algorithm": optimizer_contract.algorithm,
                    "learning_rate_hex": optimizer_contract.learning_rate.hex(),
                    "weight_decay_hex": optimizer_contract.weight_decay.hex(),
                    "scheduler": optimizer_contract.scheduler,
                    "moe_load_balance_hook_enabled": False,
                    "update_count": curve_points - 1,
                },
            }
            report = {
                "schema": RELATION_GEOMETRY_ARM_REPORT_SCHEMA,
                "status": "PASS",
                "arm": arm,
                "subject_sha256": relation_geometry_probe_subject(
                    provenance,
                    curve_point_count=curve_points,
                ),
                "provenance": provenance,
                "trainable_scope": trainable_scope.as_dict(),
                "curve_point_count": curve_points,
                "optimizer_update_count": curve_points - 1,
                "global_curves": global_curves,
                "rank_reports": rank_reports,
                "moe_routing_bias_unchanged": True,
                "maximum_peak_reserved_bytes": max(
                    int(item["peak_reserved_bytes"]) for item in rank_reports
                ),
                "total_time_s": max(float(value) for value in total_times),
            }
            validate_relation_geometry_arm_report(report)
            output = Path(output_path)
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"relation probe report already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_text_durable(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
        except BaseException as error:
            publish_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(publish_error, src=0)
    if publish_error[0] is not None:
        raise RuntimeError(f"relation probe publication failed: {publish_error[0]}")
    dist.barrier()
    if rank == 0:
        if report is None:
            raise RuntimeError("rank zero lost the relation probe report")
        print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = _parse_args()
    _validate_paths_and_args(args)
    behavior_graph_sha256 = _behavior_graph_digest(args)
    behavior_g0_evidence = (
        None
        if args.behavior_causal_probe_evidence is None
        else load_behavior_causal_probe_evidence(
            args.behavior_causal_probe_evidence,
            expected_sha256=args.behavior_causal_probe_evidence_sha256,
        )
    )
    behavior_g1_predecessor = None
    behavior_g1_predecessor_sha256 = None
    if args.behavior_posterior_control_probe_output is not None:
        behavior_g1_predecessor, behavior_g1_predecessor_sha256 = (
            load_behavior_g1_predecessor_report(
                args.run_dir,
                load_global_step=args.load_global_step,
                expected_sha256=args.behavior_g1_predecessor_report_sha256,
            )
        )
    behavior_g2_evidence = None
    behavior_g2_evidence_sha256 = None
    if args.behavior_posterior_control_probe_evidence is not None:
        behavior_g2_evidence, behavior_g2_evidence_sha256 = (
            load_behavior_posterior_control_probe_evidence(
                args.behavior_posterior_control_probe_evidence,
                expected_sha256=(args.behavior_posterior_control_probe_evidence_sha256),
            )
        )
    require_persistent_run_root(args.run_dir)
    representation_stage = args.training_stage == NATIVE_REPRESENTATION_STAGE
    behavior_action_training = args.behavior_conditioned_prediction and not representation_stage
    representation_split = (
        RepresentationTrialSplit.load(args.representation_split)
        if representation_stage or behavior_action_training
        else None
    )
    _require_runtime_storage_capacity(
        args.run_dir,
        checkpoint_required=args.checkpoint_publication == "always",
    )
    physical_visual_acceptance = load_calvin_physical_visual_acceptance(
        args.physical_visual_acceptance,
        expected_sha256=args.physical_visual_acceptance_sha256,
        expected_dataset_manifest_sha256=_sha256(args.dataset_manifest),
        expected_sidecar_manifest_sha256=args.physical_sidecar_manifest_sha256,
    )
    _validate_training_supervision_policy(
        physical_visual_acceptance=physical_visual_acceptance,
        minimum_supervised_fraction=args.minimum_supervised_fraction,
    )
    physical_visual_acceptance_sha256 = _require_sha256(
        "physical visual acceptance sha256",
        args.physical_visual_acceptance_sha256,
    )
    if _BOOTSTRAPPED_CUDA_ALLOCATOR is None:
        _configure_cuda_allocator(args.cuda_allocator)
    elif args.cuda_allocator != _BOOTSTRAPPED_CUDA_ALLOCATOR:
        raise RuntimeError("CUDA allocator pre-bootstrap differs from parsed arguments")
    fsdp2_placement = validate_fsdp2_placement(args.fsdp2_placement)
    full_cpu_offload = fsdp2_placement == FSDP2_CPU_OFFLOAD
    selective_embedding_offload = fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD
    root = Path(__file__).resolve().parents[1]
    predictive_report = load_predictive_build_report(
        args.predictive_cache_build_report,
        expected_sha256=args.predictive_cache_build_report_sha256,
    )
    if Path(predictive_report["output_root"]).resolve() != args.predictive_cache_root.resolve():
        raise ValueError("predictive build report and cache root differ")
    if predictive_report["physical_visual_acceptance_sha256"] != physical_visual_acceptance_sha256:
        raise ValueError("predictive cache predates the accepted physical visual review")
    current_grid_report = load_current_grid_build_report(
        args.current_grid_cache_build_report,
        expected_sha256=args.current_grid_cache_build_report_sha256,
    )
    if Path(current_grid_report["output_root"]).resolve() != args.current_grid_cache_root.resolve():
        raise ValueError("current-grid build report and cache root differ")
    if (
        current_grid_report["physical_visual_acceptance_sha256"]
        != physical_visual_acceptance_sha256
    ):
        raise ValueError("current-grid cache predates the accepted physical visual review")
    cache_producer_patch_sha256 = _cache_producer_patch_sha256(
        predictive_report,
        current_grid_report,
    )
    if physical_visual_acceptance["sidecar_manifest_sha256"] != (
        args.physical_sidecar_manifest_sha256
    ):
        raise RuntimeError("physical visual acceptance belongs to another sidecar")
    patch_report = verify_native_patch(
        root=root,
        checkout=args.source_checkout,
        check_apply=True,
    )
    patch_sha256 = _require_sha256(
        "native patch report sha256",
        patch_report.get("patch_sha256"),
    )
    prepared_source = validate_prepared_native_source(
        checkout=args.source_checkout,
        patch_path=args.patch,
    )
    expected_hashes = patch_report.get("patched_source_sha256")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("native full patch verifier returned no source hashes")
    actual_hashes = prepared_source.get("patched_source_sha256")
    if not isinstance(actual_hashes, dict):
        raise RuntimeError("native full prepared source returned no source hashes")
    if actual_hashes != expected_hashes:
        raise RuntimeError("native full LingBot source differs from immutable patch replay")
    validate_checkpoint(args.checkpoint_dir)
    processor_report = validate_processor(args.processor_dir)

    if os.environ.get("WORLD_SIZE") != str(FULL_WORLD_SIZE):
        raise RuntimeError("native full training requires torchrun with exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(FULL_WORLD_SIZE):
        raise RuntimeError("native full training requires both processes on one two-GPU host")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import numpy as np
    import torch
    import torch.distributed as dist

    from picf_next.lingbot_native.torch_dcp_compat import (
        install_torch_2_8_sparse_optimizer_state_backport,
    )

    install_torch_2_8_sparse_optimizer_state_backport(torch)

    from lingbotvla.checkpoint import build_checkpointer
    from lingbotvla.data import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        LingbotVlaV2Policy,
    )
    from lingbotvla.models.vla.lingbot_vla.moe_load_balance import (
        build_moe_load_balance_hook,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
        apply_lingbot_qwen2_patch,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )
    from lingbotvla.optim import build_muon_optimizer
    from transformers import AutoConfig, __version__ as transformers_version
    from transformers.modeling_utils import no_init_weights

    from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.calvin_physical_supervision_schema import (
        CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
    from picf_next.data.dataset_manifest import (
        DatasetFileManifest,
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.eval.calvin_task_relevance import calvin_exact_task_loss_identities
    from picf_next.lingbot_native.calvin import (
        build_native_calvin_continuation_batch,
        build_native_calvin_training_stream_plan,
        build_planned_native_calvin_batch,
        collate_native_calvin_training_batch,
        materialize_native_flow_randomness,
    )
    from picf_next.lingbot_native.calvin_objective import (
        NativeStructuralLossConfig,
        compose_native_calvin_objective,
    )
    from picf_next.lingbot_native.controls import (
        ExecutedControlBatch,
        concatenate_executed_controls,
    )
    from picf_next.lingbot_native.current_grid_cache import (
        LingBotCurrentGridTargetCache,
        current_correction_summary_query_schema_digest,
        current_grid_query_schema_digest,
        omitted_static_summary_query_schema_digest,
    )
    from picf_next.lingbot_native.full_training import (
        NativeFullObjectiveStepResult,
        NativeOvershootFactory,
        NativeRepresentationObjectiveStepResult,
        _future_loss_inputs,
        make_native_future_request,
        native_current_correction_target,
        run_native_calvin_full_objective,
        run_native_calvin_representation_objective,
        run_native_calvin_relation_probe_objective,
    )
    from picf_next.lingbot_native.fixed_observation import (
        FixedObservationPairPlan,
        apply_fixed_observation_pair,
        build_fixed_observation_pair_plan,
        load_fixed_observation_audit,
    )
    from picf_next.lingbot_native.fixed_observation_evaluation import (
        FixedObservationEvaluationPlan,
        build_fixed_observation_evaluation_plan,
    )
    from picf_next.lingbot_native.host import (
        LingBotNativeContext,
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        LingBotNativePriorStepper,
        install_lingbot_native_graph,
    )
    from picf_next.lingbot_native.objective import NativeObjectiveConfig
    from picf_next.lingbot_native.prediction import PredictionSource
    from picf_next.lingbot_native.predictive_cache import (
        LINGBOT_PREDICTIVE_TARGET_SPACE,
        LingBotPredictiveTargetCache,
        native_predictive_query_schema_digest,
    )
    from picf_next.lingbot_native.predictive_plan import (
        build_native_predictive_coverage_plan,
    )
    from picf_next.lingbot_native.predictive_probes import (
        WRONG_TIME_SOURCE,
        NativeCorrectionCounterfactualPredictions,
        behavior_posterior_control_diagnostics,
        predictive_correction_counterfactual_diagnostics,
        run_native_behavior_causal_probe,
        run_native_behavior_posterior_control_forwards,
        run_native_correction_counterfactual_forwards,
    )
    from picf_next.lingbot_native.public_vl_evidence import cpu_tensor_sha256
    from picf_next.lingbot_native.representation_evaluation import (
        build_representation_evaluation_plan,
        build_representation_warm_evaluation_plan,
    )
    from picf_next.lingbot_native.representation_evaluation_runtime import (
        fixed_observation_training_pair_fingerprint,
        run_fixed_observation_checkpoint_evaluation,
        run_representation_checkpoint_evaluation,
    )
    from picf_next.lingbot_native.representation_intervention import (
        RepresentationTaskInterventionPlan,
        apply_representation_task_intervention,
        build_representation_task_intervention_plan,
    )
    from picf_next.lingbot_native.source_mask import (
        sample_qwen_packed_patch_mask,
        sample_qwen_whole_view_omission,
    )
    from picf_next.lingbot_native.representation_stage import (
        configure_native_representation_parameter_scope,
        native_representation_action_state_changes,
        native_representation_action_state_manifest_sha256,
        native_representation_frozen_action_state_manifest,
        verify_native_representation_parameter_scope,
    )
    from picf_next.lingbot_native.state import NativePosteriorState
    from picf_next.lingbot_native.supervision import (
        SequenceAssignment,
        assignment_binding_start_phase,
        assignment_binding_valid_at_phase,
    )
    from picf_next.lingbot_native.temporal import (
        NativeLaneConfig,
        NativeTrainingLaneBank,
        TemporalEstimatorConfig,
        native_temporal_batch_seed,
        rollout_native_prior_prediction,
        sample_temporal_batch_plan,
    )
    from picf_next.lingbot_native.training import (
        NativeTrainingLaneCoordinator,
        audit_native_optimizer_coverage,
        run_native_policy_diagnostic_forward,
        run_native_policy_observation_diagnostic_forward,
    )
    from picf_next.lingbot_native.visual_audit import (
        matched_row_soft_iou,
        render_native_relation_visuals,
    )
    from picf_next.training.control import (
        FrozenEpisodeStreamPlan,
        FrozenResetMixtureStreamPlan,
        derive_subseed,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    run_lease = None
    try:
        run_lease = acquire_distributed_run_lease(
            args.run_dir,
            rank=rank,
            distributed=dist,
        )
        if torch.cuda.device_count() != FULL_WORLD_SIZE:
            raise RuntimeError("native full process sees a CUDA topology other than two devices")
        properties = torch.cuda.get_device_properties(device)
        if "A100" not in properties.name or properties.total_memory < 39 * 1024**3:
            raise RuntimeError("native full training requires two A100 devices of at least 39 GiB")

        dataset_contract: list[Any] = [None]
        rank_zero_manifest: DatasetFileManifest | None = None
        if rank == 0:
            try:
                rank_zero_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
                norm_stats = json.loads(args.norm_stats.read_text())
                validate_lingbot_calvin_norm_stats(norm_stats)
                source = norm_stats["source"]
                if (
                    source["dataset_id"] != rank_zero_manifest.dataset_id
                    or source["dataset_revision"] != rank_zero_manifest.dataset_revision
                    or source["dataset_tree_sha256"] != rank_zero_manifest.tree_sha256
                    or rank_zero_manifest.split_name != args.dataset_split.name
                ):
                    raise ValueError("native full CALVIN manifest and normalization differ")
                dataset_contract[0] = {
                    "status": "PASS",
                    "manifest_sha256": _sha256(args.dataset_manifest),
                    "normalization_sha256": _sha256(args.norm_stats),
                    "validation": validate_dataset_runtime_binding(
                        rank_zero_manifest,
                        args.dataset_split,
                        dataset_id=source["dataset_id"],
                        dataset_revision=source["dataset_revision"],
                        split_name=args.dataset_split.name,
                    ),
                }
            except BaseException as error:
                dataset_contract[0] = {
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(dataset_contract, src=0)
        dataset_contract_report = dataset_contract[0]
        if (
            not isinstance(dataset_contract_report, dict)
            or dataset_contract_report.get("status") != "PASS"
        ):
            raise RuntimeError(f"native full dataset contract failed: {dataset_contract_report}")
        dataset_manifest = (
            rank_zero_manifest
            if rank_zero_manifest is not None
            else load_dataset_file_manifest(args.dataset_manifest.resolve())
        )
        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=dataset_manifest.dataset_id,
            dataset_revision=dataset_manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=dataset_manifest,
        )

        physical_sidecar = CalvinPhysicalSupervisionSidecar(
            args.physical_sidecar_root,
            index,
            manifest_path=args.physical_sidecar_manifest,
            expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
        )
        if physical_sidecar.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            raise RuntimeError("native full training requires all-source physical supervision")

        temporal_config = TemporalEstimatorConfig(
            local_bptt_probability=args.local_bptt_probability,
            overshoot_probability=args.overshoot_probability,
            source_mask_probability=args.source_mask_probability,
            maximum_optimizer_lag=args.maximum_optimizer_lag,
        )

        training = load_lingbot_training_config(args.training_config)
        optimizer_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=args.learning_rate,
        )
        require_lingbot_exact_resume_contract(optimizer_contract)
        merged, _ = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=args.total_planned_steps,
        )
        merged["use_cache"] = False
        merged["use_compile"] = False
        merged["attention_implementation"] = "eager"
        merged["vit_attn_implementation"] = "eager"
        config = LingbotVLAV2Config(**merged)
        for key, value in merged.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        if bool(config.train_expert_only) or bool(config.freeze_vision_encoder):
            raise RuntimeError(
                "native predictive training requires the complete trainable VLM host"
            )
        # QWEN_PROCESSOR_REVISION is an exact commit and this load is local-only.
        qwen_config = AutoConfig.from_pretrained(  # nosec B615
            args.processor_dir,
            revision=QWEN_PROCESSOR_REVISION,
            local_files_only=True,
        )
        vision_config = qwen_config.vision_config
        training_projection = _validate_lingbot_projection_processor(
            physical_visual_acceptance=physical_visual_acceptance,
            processor_report=processor_report,
            vision_config=vision_config,
            dataset_tree_sha256=dataset_manifest.tree_sha256,
            transformers_version=transformers_version,
        )
        _merge_qwen_config(config, qwen_config)
        config.tokenizer_path = str(args.processor_dir.resolve())

        dataset = CalvinStatefulTransitionDataset(index, action_horizon=config.chunk_size)
        reset_mixture = reset_mixture_values(args)
        stream_plan = build_native_calvin_training_stream_plan(
            dataset,
            comparison_id=FULL_COMPARISON_ID,
            seed=args.seed,
            global_batch_size=FULL_WORLD_SIZE,
            total_steps=args.total_planned_steps,
            lane_interleave_factor=args.lane_interleave_factor,
            excluded_source_episode_indices=(
                representation_split.evaluation_source_episode_indices
                if representation_split is not None
                and representation_split.schema == REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
                else ()
            ),
            reset_numerator=(None if reset_mixture is None else reset_mixture[0]),
            reset_denominator=(None if reset_mixture is None else reset_mixture[1]),
        )
        if representation_split is not None:
            split_identity = (
                representation_split.dataset_id,
                representation_split.dataset_revision,
                representation_split.dataset_manifest_sha256,
                representation_split.comparison_id,
                representation_split.stream_plan_sha256,
            )
            plan_identity = (
                stream_plan.dataset_id,
                stream_plan.dataset_revision,
                stream_plan.dataset_manifest_sha256,
                stream_plan.comparison_id,
                stream_plan.plan_sha256,
            )
            if split_identity != plan_identity:
                raise RuntimeError(
                    "representation source split differs from the active frozen stream"
                )
        if isinstance(stream_plan, FrozenResetMixtureStreamPlan):
            noncausal_audits = tuple(
                step
                for step in args.gradient_audit_steps
                if stream_plan.component_for_step(step - 1) != "causal"
            )
            if noncausal_audits:
                raise RuntimeError(
                    "reset-mixture family audits require a factual causal prior: "
                    f"{noncausal_audits}"
                )
        representation_task_intervention: RepresentationTaskInterventionPlan | None = None
        if args.representation_task_intervention_plan is not None:
            if not isinstance(stream_plan, FrozenEpisodeStreamPlan):
                raise RuntimeError("legacy task intervention requires its frozen causal stream")
            representation_task_intervention = RepresentationTaskInterventionPlan.load(
                args.representation_task_intervention_plan
            )
            rebuilt_task_intervention = build_representation_task_intervention_plan(
                stream_plan,
                dataset,
                task_identity_resolver=calvin_exact_task_loss_identities,
            )
            if representation_task_intervention != rebuilt_task_intervention:
                raise RuntimeError(
                    "representation task intervention differs from source-only reconstruction"
                )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "artifact_sha256": (representation_task_intervention.artifact_sha256),
                            "event": "representation_task_intervention_bound",
                            "file_sha256": (args.representation_task_intervention_plan_sha256),
                            "intervened_primary_slots": (
                                representation_task_intervention.exact_slot_count
                            ),
                            "factual_primary_slots": (
                                representation_task_intervention.inexact_slot_count
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        fixed_observation_pair_plan: FixedObservationPairPlan | None = None
        fixed_observation_training_audit = None
        if args.fixed_observation_pair_plan is not None:
            if not isinstance(stream_plan, FrozenResetMixtureStreamPlan):
                raise RuntimeError(
                    "fixed-observation pairing requires a frozen reset-mixture stream"
                )
            if args.fixed_observation_training_audit is None:
                raise RuntimeError("fixed-observation training audit vanished")
            fixed_observation_training_audit = load_fixed_observation_audit(
                args.fixed_observation_training_audit,
                expected_file_sha256=(args.fixed_observation_training_audit_sha256),
                expected_partition="training",
            )
            if fixed_observation_training_audit.dataset_manifest_file_sha256 != _sha256(
                args.dataset_manifest
            ):
                raise RuntimeError("fixed-observation audit belongs to another dataset manifest")
            fixed_observation_pair_plan = FixedObservationPairPlan.load(
                args.fixed_observation_pair_plan
            )
            rebuilt_fixed_observation_pair_plan = build_fixed_observation_pair_plan(
                stream_plan,
                dataset,
                fixed_observation_training_audit,
            )
            if fixed_observation_pair_plan != rebuilt_fixed_observation_pair_plan:
                raise RuntimeError(
                    "fixed-observation pair plan differs from source-only reconstruction"
                )
            if representation_task_intervention is not None:
                raise RuntimeError(
                    "fixed-observation and legacy task interventions were both bound"
                )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "artifact_sha256": (fixed_observation_pair_plan.artifact_sha256),
                            "audit_artifact_sha256": (
                                fixed_observation_training_audit.report_artifact_sha256
                            ),
                            "audit_file_sha256": (args.fixed_observation_training_audit_sha256),
                            "event": "fixed_observation_pair_plan_bound",
                            "file_sha256": (args.fixed_observation_pair_plan_sha256),
                            "pair_count": len(fixed_observation_pair_plan.pairs),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        fixed_observation_evaluation_plan: FixedObservationEvaluationPlan | None = None
        if args.fixed_observation_evaluation_plan is not None:
            if (
                fixed_observation_pair_plan is None
                or fixed_observation_training_audit is None
                or args.fixed_observation_validation_audit is None
                or args.fixed_observation_heldout_audit is None
            ):
                raise RuntimeError("fixed-observation evaluation lost its training contracts")
            validation_audit = load_fixed_observation_audit(
                args.fixed_observation_validation_audit,
                expected_file_sha256=args.fixed_observation_validation_audit_sha256,
                expected_partition="validation",
            )
            heldout_audit = load_fixed_observation_audit(
                args.fixed_observation_heldout_audit,
                expected_file_sha256=args.fixed_observation_heldout_audit_sha256,
                expected_partition="heldout",
            )
            fixed_observation_evaluation_plan = FixedObservationEvaluationPlan.load(
                args.fixed_observation_evaluation_plan
            )
            rebuilt_fixed_observation_evaluation_plan = build_fixed_observation_evaluation_plan(
                training_audit=fixed_observation_training_audit,
                validation_audit=validation_audit,
                heldout_audit=heldout_audit,
                training_pair_plan=fixed_observation_pair_plan,
            )
            if fixed_observation_evaluation_plan != rebuilt_fixed_observation_evaluation_plan:
                raise RuntimeError(
                    "fixed-observation evaluation plan differs from source-only reconstruction"
                )
            if (
                fixed_observation_evaluation_plan.dataset_tree_sha256
                != dataset_manifest.tree_sha256
            ):
                raise RuntimeError("fixed-observation evaluation belongs to another dataset tree")
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "artifact_sha256": (fixed_observation_evaluation_plan.artifact_sha256),
                            "event": "fixed_observation_evaluation_plan_bound",
                            "file_sha256": (args.fixed_observation_evaluation_plan_sha256),
                            "heldout_items": len(
                                fixed_observation_evaluation_plan.items_for(
                                    "heldout",
                                    0,
                                )
                            )
                            + len(
                                fixed_observation_evaluation_plan.items_for(
                                    "heldout",
                                    1,
                                )
                            ),
                            "validation_items": len(
                                fixed_observation_evaluation_plan.items_for(
                                    "validation",
                                    0,
                                )
                            )
                            + len(
                                fixed_observation_evaluation_plan.items_for(
                                    "validation",
                                    1,
                                )
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        def apply_training_prompt_overlay(planned: Any) -> Any:
            if fixed_observation_pair_plan is not None:
                return apply_fixed_observation_pair(
                    planned,
                    fixed_observation_pair_plan,
                    dataset,
                )
            if representation_task_intervention is not None:
                return apply_representation_task_intervention(
                    planned,
                    representation_task_intervention,
                    dataset,
                )
            return planned

        representation_evaluation_plan: RepresentationEvaluationPlan | None = None
        representation_warm_evaluation_plan: RepresentationEvaluationPlan | None = None
        representation_evaluation_baseline: dict[str, Any] | None = None
        if args.representation_evaluation_plan is not None:
            if representation_split is None:
                raise RuntimeError("representation evaluation lost its source split")
            representation_evaluation_plan = RepresentationEvaluationPlan.load(
                args.representation_evaluation_plan
            )
            rebuilt_evaluation_plan = build_representation_evaluation_plan(
                representation_split,
                dataset,
                task_identity_resolver=calvin_exact_task_loss_identities,
                evaluation_reference_plan_sha256=(
                    representation_evaluation_plan.evaluation_reference_plan_sha256
                ),
            )
            if representation_evaluation_plan != rebuilt_evaluation_plan:
                raise RuntimeError(
                    "representation evaluation plan differs from source-only reconstruction"
                )
        if args.representation_warm_evaluation_plan is not None:
            if representation_split is None:
                raise RuntimeError("warm representation evaluation lost its source split")
            representation_warm_evaluation_plan = RepresentationEvaluationPlan.load(
                args.representation_warm_evaluation_plan
            )
            rebuilt_warm_evaluation_plan = build_representation_warm_evaluation_plan(
                representation_split,
                dataset,
                task_identity_resolver=calvin_exact_task_loss_identities,
                history_transitions=representation_warm_evaluation_plan.history_transitions,
            )
            if representation_warm_evaluation_plan != rebuilt_warm_evaluation_plan:
                raise RuntimeError(
                    "warm representation evaluation differs from source-only reconstruction"
                )
        if args.representation_evaluation_baseline is not None:
            if representation_evaluation_plan is None:
                raise RuntimeError("representation baseline lost its evaluation plan")
            representation_evaluation_baseline = load_representation_evaluation_baseline(
                args.representation_evaluation_baseline
            )
            validate_representation_baseline_plan(
                representation_evaluation_baseline,
                candidate_plan=representation_evaluation_plan,
            )
        relation_probe_sample_selection: RelationProbeSampleSelection | None = None
        if args.relation_geometry_fixed_batch_arm is not None:
            selection_error: BaseException | None = None
            try:
                relation_probe_sample_selection = _select_relation_geometry_source_sample(
                    args=args,
                    stream_plan=stream_plan,
                    dataset=dataset,
                    physical_sidecar=physical_sidecar,
                    task_identity_resolver=calvin_exact_task_loss_identities,
                )
            except BaseException as error:
                selection_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                stage="relation source-only sample selection",
                local_error=selection_error,
            )
            if relation_probe_sample_selection is None:
                raise RuntimeError("relation sample selection vanished after collective validation")
            selection_payload = relation_probe_sample_selection.as_dict()
            gathered_selections: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
            dist.all_gather_object(gathered_selections, selection_payload)
            if any(value != selection_payload for value in gathered_selections):
                raise RuntimeError("relation source-only sample selection differs across ranks")
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "relation_probe_sample_selected",
                            "selection": selection_payload,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if predictive_report["stream_plan_sha256"] != stream_plan.plan_sha256:
            raise RuntimeError("predictive cache and training stream plans differ")
        if predictive_report["temporal_estimator_sha256"] != temporal_config.digest:
            raise RuntimeError("predictive cache and temporal estimators differ")
        expected_predictive_coverage = build_native_predictive_coverage_plan(
            stream_plan,
            temporal_config,
            source_global_index_for_sample=dataset.source_global_index_by_key,
            required_horizons=(1,) if args.behavior_conditioned_prediction else (),
        )
        if (
            predictive_report["pair_keys_sha256"] != expected_predictive_coverage.pair_keys_sha256
            or predictive_report["coverage_sha256"] != expected_predictive_coverage.coverage_sha256
            or predictive_report["expected_record_count"] != len(expected_predictive_coverage.pairs)
        ):
            raise RuntimeError("predictive cache does not cover the exact training objective")
        query_schema_sha256 = native_predictive_query_schema_digest(
            target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
            route_id=0,
            horizons=temporal_config.overshoot_horizons,
        )
        load_predictive_target_audit(
            args.predictive_target_audit,
            expected_sha256=args.predictive_target_audit_sha256,
            predictive_report=predictive_report,
            dataset_tree_sha256=dataset_manifest.tree_sha256,
            physical_sidecar_manifest_sha256=physical_sidecar.manifest_sha256,
            query_schema_sha256=query_schema_sha256,
            stream_plan_sha256=stream_plan.plan_sha256,
            temporal_estimator_sha256=temporal_config.digest,
        )
        load_predictive_teacher_causality_audit(
            args.predictive_teacher_causality_audit,
            expected_sha256=args.predictive_teacher_causality_audit_sha256,
            predictive_report=predictive_report,
            current_grid_report=current_grid_report,
            dataset_tree_sha256=dataset_manifest.tree_sha256,
            physical_sidecar_manifest_sha256=physical_sidecar.manifest_sha256,
            patch_sha256=cache_producer_patch_sha256,
            horizons=temporal_config.overshoot_horizons,
        )
        predictive_cache = LingBotPredictiveTargetCache.load(
            args.predictive_cache_root,
            manifest_sha256=predictive_report["cache_manifest_sha256"],
            dataset_tree_sha256=dataset_manifest.tree_sha256,
            physical_sidecar_manifest_sha256=physical_sidecar.manifest_sha256,
            encoder_digest=predictive_report["teacher_encoder_digest"],
            query_schema_sha256=query_schema_sha256,
            coverage_sha256=predictive_report["coverage_sha256"],
            memory_capacity=args.predictive_cache_memory_shards,
        )
        if (
            predictive_cache.contract.stream_plan_sha256 != stream_plan.plan_sha256
            or predictive_cache.contract.temporal_estimator_sha256 != temporal_config.digest
            or predictive_cache.contract.lingbot_source_commit != LINGBOT_NATIVE_SOURCE_COMMIT
            or predictive_cache.contract.lingbot_checkpoint_revision != LINGBOT_CHECKPOINT_REVISION
        ):
            raise RuntimeError("native full predictive cache provenance differs")
        if current_grid_report["stream_plan_sha256"] != stream_plan.plan_sha256:
            raise RuntimeError("current-grid cache and training stream plans differ")
        if current_grid_report["temporal_estimator_sha256"] != temporal_config.digest:
            raise RuntimeError("current-grid cache and temporal estimators differ")
        current_grid_cache = LingBotCurrentGridTargetCache.load(
            args.current_grid_cache_root,
            manifest_sha256=current_grid_report["cache_manifest_sha256"],
            dataset_tree_sha256=dataset_manifest.tree_sha256,
            physical_sidecar_manifest_sha256=physical_sidecar.manifest_sha256,
            encoder_digest=current_grid_report["teacher_encoder_digest"],
            coverage_sha256=current_grid_report["coverage_sha256"],
            memory_capacity=args.current_grid_cache_memory_shards,
        )
        if (
            current_grid_cache.contract.stream_plan_sha256 != stream_plan.plan_sha256
            or current_grid_cache.contract.temporal_estimator_sha256 != temporal_config.digest
            or current_grid_cache.contract.lingbot_source_commit != LINGBOT_NATIVE_SOURCE_COMMIT
            or current_grid_cache.contract.lingbot_checkpoint_revision
            != LINGBOT_CHECKPOINT_REVISION
        ):
            raise RuntimeError("native full current-grid cache provenance differs")
        if current_grid_cache.contract.hidden_size != predictive_cache.contract.hidden_size:
            raise RuntimeError("current and future DINO target widths differ")
        if (
            args.predictive_fixed_batch_arm is None
            and args.relation_geometry_fixed_batch_arm is None
            and not args.behavior_conditioned_prediction
        ):
            _validate_gradient_audit_target_coverage(
                stream_plan=stream_plan,
                audit_steps=args.gradient_audit_steps,
                source_global_index_for_sample=dataset.source_global_index_by_key,
                target_has_support=lambda *, source_global_index: (
                    current_grid_cache.has_supported_current_summary(
                        source_global_index=source_global_index,
                        physical_sidecar=physical_sidecar,
                        minimum_visible_fraction=(
                            predictive_cache.contract.minimum_visible_fraction
                        ),
                    )
                ),
            )
        load_predictive_temporal_audit(
            args.predictive_temporal_audit,
            expected_sha256=args.predictive_temporal_audit_sha256,
            predictive_report=predictive_report,
            current_grid_report=current_grid_report,
            physical_sidecar_manifest_sha256=physical_sidecar.manifest_sha256,
            horizons=predictive_cache.contract.horizons,
        )

        execution_sha256, implementation_sha256 = _execution_contract_digest(
            root=root,
            args=args,
            patched_source_sha256=actual_hashes,
            predictive_report=predictive_report,
            current_grid_report=current_grid_report,
            query_schema_sha256={
                "controlled_future_rollout": query_schema_sha256,
                "current_correction": current_correction_summary_query_schema_digest(
                    route_id=0,
                    address_width=FULL_PREDICTION_ADDRESS_WIDTH,
                ),
                "current_random_grid": current_grid_query_schema_digest(route_id=0),
                "omitted_static": omitted_static_summary_query_schema_digest(route_id=0),
            },
            temporal_metadata=temporal_config.metadata,
            optimizer_contract=optimizer_contract.metadata,
            behavior_graph_sha256=behavior_graph_sha256,
        )
        _require_unchanged_behavior_graph(
            args,
            expected_sha256=behavior_graph_sha256,
        )
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        init_parallel_state(
            dp_size=FULL_WORLD_SIZE,
            dp_replicate_size=1,
            dp_shard_size=FULL_WORLD_SIZE,
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=1,
            dp_mode="fsdp2",
        )
        processor = build_processor(str(args.processor_dir.resolve()))
        complete_prompt_tasks = [segment.instruction for segment in index.segments]
        fixed_observation_prompt_pairs: list[tuple[str, str]] = []
        if fixed_observation_pair_plan is not None:
            for pair in fixed_observation_pair_plan.pairs:
                instructions = (pair.variants[0].instruction, pair.variants[1].instruction)
                complete_prompt_tasks.extend(instructions)
                fixed_observation_prompt_pairs.append(instructions)
        if fixed_observation_evaluation_plan is not None:
            for item in fixed_observation_evaluation_plan.items:
                instructions = (item.variants[0].instruction, item.variants[1].instruction)
                complete_prompt_tasks.extend(instructions)
                fixed_observation_prompt_pairs.append(instructions)
        prompt_tokenization_audit = audit_complete_prompt_tokenization(
            tuple(complete_prompt_tasks),
            processor.tokenizer,
            maximum_tokens=int(config.tokenizer_max_length),
            use_qwen3_chat_template=bool(getattr(config, "use_qwen3_chat_template", False)),
        )
        if fixed_observation_prompt_pairs:
            validate_distinct_prompt_tokenizations(
                prompt_tokenization_audit,
                tuple(fixed_observation_prompt_pairs),
            )
        prompt_audit_digests: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
        dist.all_gather_object(
            prompt_audit_digests,
            prompt_tokenization_audit.artifact_sha256,
        )
        if any(
            digest != prompt_tokenization_audit.artifact_sha256 for digest in prompt_audit_digests
        ):
            raise RuntimeError("complete prompt tokenization audit differs across ranks")
        prompt_audit_error: list[str | None] = [None]
        if rank == 0:
            try:
                prompt_audit_path = _bind_complete_prompt_tokenization_audit(
                    args.run_dir,
                    prompt_tokenization_audit,
                )
                print(
                    json.dumps(
                        {
                            "artifact_sha256": prompt_tokenization_audit.artifact_sha256,
                            "event": "complete_prompt_tokenization_bound",
                            "maximum_observed_tokens": (
                                prompt_tokenization_audit.maximum_observed_tokens
                            ),
                            "maximum_tokens": prompt_tokenization_audit.maximum_tokens,
                            "path": str(prompt_audit_path.resolve()),
                            "prompt_count": prompt_tokenization_audit.prompt_count,
                            "truncation_count": 0,
                            "unique_prompt_count": len(prompt_tokenization_audit.prompts),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                prompt_audit_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(prompt_audit_error, src=0)
        if prompt_audit_error[0] is not None:
            raise RuntimeError(
                f"complete prompt tokenization publication failed: {prompt_audit_error[0]}"
            )
        dist.barrier()
        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
        with init_empty_weights(), no_init_weights():
            policy = LingbotVlaV2Policy(config=config, eval=False).to(torch.float32)
        load_model_weights(
            policy,
            str(args.checkpoint_dir.resolve()),
            str(device),
            post_training=True,
            adanorm_time=bool(config.adanorm_time),
        )
        alignment_teacher_prune = strip_targetless_alignment_teacher_heads(policy)
        policy.train()
        graph_config = LingBotNativeGraphConfig.from_policy(
            policy,
            capacity=args.capacity,
            maximum_control_tokens=args.maximum_control_tokens,
            predictive_target_widths=(
                (
                    LINGBOT_PREDICTIVE_TARGET_SPACE,
                    predictive_cache.contract.hidden_size,
                ),
            ),
            prediction_address_width=FULL_PREDICTION_ADDRESS_WIDTH,
            relation_supervision_layers=args.relation_supervision_layers,
        )
        graph = LingBotNativeGraph(graph_config, device=device, dtype=torch.float32).train()
        install_lingbot_native_graph(policy, graph)
        representation_parameter_scope = (
            configure_native_representation_parameter_scope(policy)
            if representation_stage
            else None
        )
        fixed_batch_trainable_scope = (
            None
            if args.predictive_fixed_batch_arm is None
            else configure_fixed_batch_trainable_scope(
                policy,
                graph,
                arm=args.predictive_fixed_batch_arm,
            )
        )
        relation_geometry_trainable_scope = (
            None
            if args.relation_geometry_fixed_batch_arm is None
            else configure_relation_geometry_trainable_scope(
                policy,
                graph,
                arm=(
                    "existing_readout_frozen_host"
                    if args.relation_geometry_fixed_batch_arm
                    in {
                        RELATION_BILINEAR_PROBE_ARM,
                        RELATION_DEPTH_PROBE_ARM,
                    }
                    else args.relation_geometry_fixed_batch_arm
                ),
            )
        )
        model_family_sha256 = _canonical_digest(
            {
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "complete_prompt_tokenization_sha256": (prompt_tokenization_audit.artifact_sha256),
                "execution_sha256": (
                    None if args.behavior_conditioned_prediction else execution_sha256
                ),
                "behavior_graph_sha256": behavior_graph_sha256,
                "graph": asdict(graph_config),
                "plan_sha256": stream_plan.plan_sha256,
                "training_stage": args.training_stage,
                "representation_split_sha256": (
                    None if representation_split is None else representation_split.artifact_sha256
                ),
                "representation_task_intervention_sha256": (
                    None
                    if representation_task_intervention is None
                    else representation_task_intervention.artifact_sha256
                ),
                "fixed_observation_pair_plan_sha256": (
                    None
                    if fixed_observation_pair_plan is None
                    else fixed_observation_pair_plan.artifact_sha256
                ),
                "fixed_observation_evaluation_plan_sha256": (
                    None
                    if fixed_observation_evaluation_plan is None
                    else fixed_observation_evaluation_plan.artifact_sha256
                ),
                "representation_parameter_scope": (
                    None
                    if representation_parameter_scope is None
                    else representation_parameter_scope.as_dict()
                ),
                "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
            }
        )
        behavior_resume_implementation_sha256 = implementation_sha256
        behavior_resume_execution_sha256 = execution_sha256
        behavior_context_model_family_sha256 = model_family_sha256
        if behavior_g1_predecessor is not None:
            if (
                behavior_g1_predecessor["implementation_sha256"] != implementation_sha256
                or behavior_g1_predecessor["execution_contract_sha256"] != execution_sha256
                or behavior_g1_predecessor["model_family_sha256"] != model_family_sha256
                or behavior_g1_predecessor["plan_sha256"] != stream_plan.plan_sha256
            ):
                raise RuntimeError(
                    "behavior G2 predecessor belongs to another implementation, model or stream"
                )
            behavior_resume_implementation_sha256 = behavior_g1_predecessor["implementation_sha256"]
            behavior_resume_execution_sha256 = behavior_g1_predecessor["execution_contract_sha256"]
        if behavior_g2_evidence is not None:
            if behavior_graph_sha256 is None or behavior_g2_evidence_sha256 is None:
                raise RuntimeError("joint behavior/action training lost its G2 receipt")
            expected_g2_context = {
                "behavior_graph_sha256": behavior_graph_sha256,
                "current_implementation_sha256": implementation_sha256,
                "g1_predecessor_implementation_sha256": implementation_sha256,
                "g0_evidence_sha256": args.behavior_causal_probe_evidence_sha256,
                "patch_sha256": patch_sha256,
                "plan_sha256": stream_plan.plan_sha256,
            }
            if any(
                behavior_g2_evidence.get(field) != expected
                for field, expected in expected_g2_context.items()
            ):
                raise RuntimeError("behavior G2 evidence belongs to another action context")
            behavior_resume_implementation_sha256 = behavior_g2_evidence[
                "g1_predecessor_implementation_sha256"
            ]
            behavior_resume_execution_sha256 = behavior_g2_evidence[
                "g1_predecessor_execution_contract_sha256"
            ]
            behavior_context_model_family_sha256 = behavior_g2_evidence["model_family_sha256"]
        if behavior_g0_evidence is not None:
            if behavior_graph_sha256 is None:
                raise RuntimeError("behavior G1 lost its frozen behavior graph")
            require_behavior_causal_probe_context(
                behavior_g0_evidence,
                patch_sha256=patch_sha256,
                implementation_sha256=behavior_resume_implementation_sha256,
                model_family_sha256=behavior_context_model_family_sha256,
                plan_sha256=stream_plan.plan_sha256,
                behavior_graph_sha256=behavior_graph_sha256,
            )
        training_authorization = None
        if args.phase == "resume" and not representation_stage:
            if args.authorization_manifest is None:
                raise RuntimeError("validated resumed run lost its authorization manifest")
            training_authorization = validate_training_authorization(
                args.authorization_manifest,
                expected_sha256=args.authorization_manifest_sha256,
                input_global_step=args.load_global_step,
                requested_global_step=args.load_global_step + args.invocation_steps,
                total_planned_steps=args.total_planned_steps,
                visual_audit_every=args.visual_audit_every,
                execution_contract_sha256=execution_sha256,
                implementation_sha256=implementation_sha256,
                model_family_sha256=model_family_sha256,
                expected_fsdp2_placement=fsdp2_placement,
                expected_cuda_allocator=args.cuda_allocator,
                expected_predictive_objective=IMPLEMENTED_PREDICTIVE_OBJECTIVE,
                expected_predictive_visible_support_weighting=(
                    IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING
                ),
                expected_predictive_minimum_visible_fraction=(
                    predictive_cache.contract.minimum_visible_fraction
                ),
            )
        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=optimizer_contract.enable_mixed_precision,
            enable_fp32=optimizer_contract.enable_fp32,
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=full_cpu_offload,
            enable_shared_embedding_offload=selective_embedding_offload,
            fsdp_kwargs={},
            basic_modules=policy._no_split_modules,
            enable_reentrant=False,
            enable_forward_prefetch=False,
            fsdp_llm_blocks=False,
            ignore_norm=False,
            use_depth_align=False,
            split_fused_experts_from_decoder_fsdp=False,
            vlm_fsdp=True,
            use_future_image=False,
        )
        register_native_fsdp_forward_methods(policy)
        if representation_parameter_scope is not None:
            representation_parameter_scope = verify_native_representation_parameter_scope(
                policy,
                expected=representation_parameter_scope,
            )
        if fixed_batch_trainable_scope is not None:
            fixed_batch_trainable_scope = verify_fixed_batch_trainable_scope(
                policy,
                graph,
                expected=fixed_batch_trainable_scope,
            )
        if relation_geometry_trainable_scope is not None:
            relation_geometry_trainable_scope = verify_relation_geometry_trainable_scope(
                policy,
                graph,
                expected=relation_geometry_trainable_scope,
            )
        representation_released_action_state_manifest = (
            None
            if representation_parameter_scope is None
            else native_representation_frozen_action_state_manifest(
                policy,
                expected=representation_parameter_scope,
            )
        )
        representation_released_action_state_local_sha256 = (
            None
            if representation_released_action_state_manifest is None
            else native_representation_action_state_manifest_sha256(
                representation_released_action_state_manifest
            )
        )
        representation_released_action_state_sha256 = (
            None
            if representation_released_action_state_local_sha256 is None
            else _distributed_action_state_sha256(
                representation_released_action_state_local_sha256,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                dist=dist,
            )
        )
        parameter_storage = _validate_fsdp2_parameter_storage(
            policy,
            torch,
            expected_placement=fsdp2_placement,
        )
        action_fsdp2_topology = _validate_action_fsdp2_topology(policy)
        vlm_fsdp2_topology = _validate_vlm_fsdp2_topology(policy)
        if representation_stage:
            optimizer = build_lingbot_representation_optimizer(
                policy,
                optimizer_contract,
                build_muon_optimizer=build_muon_optimizer,
            )
        elif fixed_batch_trainable_scope is None and relation_geometry_trainable_scope is None:
            optimizer = build_lingbot_official_optimizer(
                policy,
                optimizer_contract,
                build_muon_optimizer=build_muon_optimizer,
                build_moe_load_balance_hook=build_moe_load_balance_hook,
            )
        else:
            optimizer = build_lingbot_fixed_batch_probe_optimizer(
                policy,
                optimizer_contract,
                build_muon_optimizer=build_muon_optimizer,
            )
        parameter_manifest = audit_native_optimizer_coverage(
            modules={"policy": policy},
            optimizer=optimizer,
        )
        checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
        prior_stepper = LingBotNativePriorStepper(policy, graph)

        lane_config = NativeLaneConfig(
            model_digest=model_family_sha256,
            schema_digest=stream_plan.plan_sha256,
            capacity=args.capacity,
            host_width=graph_config.host_width,
            maximum_optimizer_lag=args.maximum_optimizer_lag,
            device=str(device),
            dtype=torch.bfloat16,
        )
        global_step = 0
        resume_rng: dict[str, bytes] | None = None
        loaded_boundary: dict[str, str] | None = None
        representation_parameter_scope_sha256 = (
            None
            if representation_parameter_scope is None
            else _canonical_digest(representation_parameter_scope.as_dict())
        )
        if args.phase == "fresh":
            bank = NativeTrainingLaneBank(lane_config)
        else:
            checkpoint_dir = args.run_dir / "checkpoints" / f"global_step_{args.load_global_step}"
            if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
                raise FileNotFoundError(checkpoint_dir)
            if representation_stage:
                if (
                    representation_split is None
                    or representation_parameter_scope is None
                    or representation_parameter_scope_sha256 is None
                    or representation_released_action_state_sha256 is None
                    or representation_released_action_state_local_sha256 is None
                ):
                    raise RuntimeError("representation resume lost its frozen contracts")
                checkpoint_report_path = checkpoint_dir / "native_representation_report.json"
                checkpoint_report_error: BaseException | None = None
                try:
                    if checkpoint_report_path.is_symlink() or not checkpoint_report_path.is_file():
                        raise ValueError(
                            "representation checkpoint lacks its immutable report before load"
                        )
                    if behavior_g1_predecessor is None:
                        checkpoint_report = json.loads(
                            checkpoint_report_path.read_text(encoding="utf-8")
                        )
                    else:
                        reloaded_predecessor, reloaded_sha256 = load_behavior_g1_predecessor_report(
                            args.run_dir,
                            load_global_step=args.load_global_step,
                            expected_sha256=args.behavior_g1_predecessor_report_sha256,
                        )
                        if (
                            reloaded_predecessor != behavior_g1_predecessor
                            or reloaded_sha256 != behavior_g1_predecessor_sha256
                        ):
                            raise RuntimeError("behavior G1 predecessor changed during startup")
                        checkpoint_report = reloaded_predecessor
                    validate_representation_objective_report(
                        checkpoint_report,
                        expected_saved_global_step=args.load_global_step,
                        expected_digests={
                            "execution_contract_sha256": behavior_resume_execution_sha256,
                            "implementation_sha256": behavior_resume_implementation_sha256,
                            "model_family_sha256": model_family_sha256,
                            "representation_split_sha256": (representation_split.artifact_sha256),
                            "representation_split_file_sha256": (args.representation_split_sha256),
                            "representation_parameter_scope_sha256": (
                                representation_parameter_scope_sha256
                            ),
                            "representation_frozen_action_state_sha256": (
                                representation_released_action_state_sha256
                            ),
                        },
                        expected_behavior_conditioning_sha256=(
                            _behavior_conditioning_digest(
                                args,
                                behavior_graph_sha256=behavior_graph_sha256,
                            )
                        ),
                        require_initial_probe=(
                            args.load_global_step == 1 and not args.behavior_conditioned_prediction
                        ),
                        expected_checkpoint_publication="always",
                        expected_fsdp2_placement=fsdp2_placement,
                        expected_cuda_allocator=args.cuda_allocator,
                    )
                except BaseException as error:
                    checkpoint_report_error = error
                _distributed_raise_if_local_probe_error(
                    dist=dist,
                    rank=rank,
                    world_size=FULL_WORLD_SIZE,
                    stage="representation checkpoint report validation",
                    local_error=checkpoint_report_error,
                )
            state = {"model": policy, "optimizer": optimizer, "extra_state": {}}
            checkpointer.load(str(checkpoint_dir), state)
            prior_planned = build_planned_native_calvin_batch(
                stream_plan,
                dataset,
                optimizer_step=args.load_global_step - 1,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                gradient_accumulation_steps=1,
                accumulation_index=0,
                device=device,
                dtype=torch.bfloat16,
            )
            prior_planned = apply_training_prompt_overlay(prior_planned)
            resume_kwargs = {
                "expected_global_step": args.load_global_step,
                "expected_implementation_sha256": behavior_resume_implementation_sha256,
                "expected_model_family_sha256": model_family_sha256,
                "expected_execution_sha256": behavior_resume_execution_sha256,
                "expected_plan_sha256": stream_plan.plan_sha256,
                "expected_temporal_sha256": temporal_config.digest,
                "expected_source_digest": prior_planned.source_digest,
                "expected_behavior_conditioning_sha256": (
                    _behavior_conditioning_digest(
                        args,
                        behavior_graph_sha256=behavior_graph_sha256,
                    )
                ),
                "rank": rank,
            }
            if representation_stage:
                if (
                    representation_split is None
                    or representation_parameter_scope is None
                    or representation_parameter_scope_sha256 is None
                ):
                    raise RuntimeError("representation resume lost its frozen contracts")
                extra = _validate_representation_resume_extra(
                    state["extra_state"],
                    **resume_kwargs,
                    expected_representation_split_sha256=(representation_split.artifact_sha256),
                    expected_parameter_scope_sha256=(representation_parameter_scope_sha256),
                )
                loaded_action_state_manifest = native_representation_frozen_action_state_manifest(
                    policy,
                    expected=representation_parameter_scope,
                )
                loaded_action_state_sha256 = native_representation_action_state_manifest_sha256(
                    loaded_action_state_manifest
                )
                if (
                    representation_released_action_state_local_sha256 is None
                    or loaded_action_state_sha256
                    != representation_released_action_state_local_sha256
                    or loaded_action_state_sha256
                    != extra["representation_frozen_action_state_sha256"]
                ):
                    changed = (
                        ()
                        if representation_released_action_state_manifest is None
                        else native_representation_action_state_changes(
                            representation_released_action_state_manifest,
                            loaded_action_state_manifest,
                        )
                    )
                    raise RuntimeError(
                        "native representation checkpoint changed frozen action state: "
                        + ", ".join(changed)
                    )
            else:
                extra = _validate_resume_extra(
                    state["extra_state"],
                    **resume_kwargs,
                )
            optimizer_summary = _validate_optimizer_state(
                optimizer,
                torch,
                expected_step=args.load_global_step,
            )
            if any(
                optimizer_summary[name] != extra[name]
                for name in ("optimizer_state_entries", "optimizer_local_moment_elements")
            ):
                raise RuntimeError("native full restored optimizer summary differs")
            bank = NativeTrainingLaneBank.deserialize(lane_config, extra["lane_snapshot"])
            resume_rng = extra["rank_rng_state"]
            loaded_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=bank.serialize(),
                rank_rng_state=resume_rng,
                torch_module=torch,
            )
            if loaded_boundary != extra["boundary_sha256"]:
                raise RuntimeError("native full restored checkpoint boundary differs")
            global_step = args.load_global_step

        rank_seed = args.seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)
        if resume_rng is not None:
            _restore_rank_rng(resume_rng, torch, np, device=device)
            if _rank_rng_digest(_capture_rank_rng(torch, np, device=device)) != (
                _rank_rng_digest(resume_rng)
            ):
                raise RuntimeError("native full restored process RNG differs")

        data_config_payload = json.loads(args.data_config.read_text())
        if data_config_payload.get("cameras") != [
            slot.runtime_camera_name for slot in LINGBOT_CALVIN_CAMERA_SLOTS
        ]:
            raise RuntimeError("LingBot CALVIN camera-slot configuration changed")
        feature_transform = FeatureTransform(
            str(args.robot_config.resolve()),
            official_lingbot_data_config(data_config_payload),
            config,
            processor,
            chunk_size=config.chunk_size,
            norm_stats_path=str(args.norm_stats.resolve()),
            use_depth_align=False,
            image_augment=False,
            use_future_image=False,
        )
        collator = VLADataCollatorWithPacking()

        def collate_planned(planned: Any) -> Any:
            collated = collate_native_calvin_training_batch(
                planned.training,
                feature_transform=feature_transform,
                collator=collator,
                augmentation_seeds=planned.augmentation_seeds,
                source_digest=planned.source_digest,
            )
            _validate_lingbot_calvin_projection_batch(
                collated.model_inputs,
                projection=training_projection,
            )
            return materialize_native_flow_randomness(
                _collated_to_device(
                    collated,
                    device=device,
                    torch_module=torch,
                ),
                planned,
            )

        objective_config = NativeObjectiveConfig(
            predictive_weight=args.predictive_weight,
            structural_weight=args.structural_weight,
            action_weight=0.0 if representation_stage else 1.0,
        )
        structural_config = NativeStructuralLossConfig(
            support_weight=args.support_weight,
            existence_weight=args.existence_weight,
            task_weight=args.task_weight,
            dense_task_weight=args.dense_task_weight,
            ownership_weight=args.ownership_weight,
            task_relation_estimator=args.task_relation_estimator,
            ownership_estimator=_ownership_estimator_from_args(args),
        )
        patch_size = int(vision_config.patch_size)
        merge_size = int(vision_config.spatial_merge_size)

        def evaluate_representation_checkpoint(
            checkpoint_step: int,
        ) -> dict[str, object] | None:
            if (
                representation_evaluation_plan is None
                or representation_split is None
                or representation_parameter_scope is None
            ):
                raise RuntimeError("scheduled representation evaluation lost its contracts")
            evaluation_rng = _capture_rank_rng(torch, np, device=device)
            evaluation_rng_sha256 = _rank_rng_digest(evaluation_rng)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "representation_checkpoint_evaluation_started",
                            "global_step": checkpoint_step,
                            "plan_sha256": representation_evaluation_plan.artifact_sha256,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            snapshot = None
            warm_snapshot = None
            fixed_observation_snapshot = None
            try:
                snapshot = run_representation_checkpoint_evaluation(
                    policy,
                    expected_scope=representation_parameter_scope,
                    plan=representation_evaluation_plan,
                    checkpoint_global_step=checkpoint_step,
                    implementation_sha256=implementation_sha256,
                    model_family_sha256=model_family_sha256,
                    representation_split_sha256=representation_split.artifact_sha256,
                    dataset=dataset,
                    collate_planned=collate_planned,
                    physical_sidecar=physical_sidecar,
                    capacity=args.capacity,
                    task_identity_resolver=calvin_exact_task_loss_identities,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    structural_config=structural_config,
                    minimum_supervised_fraction=args.minimum_supervised_fraction,
                    output_root=(
                        args.run_dir
                        / "representation_evaluations"
                        / f"global_step_{checkpoint_step}"
                    ),
                    rank=rank,
                    world_size=FULL_WORLD_SIZE,
                    dist_module=dist,
                    device=device,
                )
                if representation_warm_evaluation_plan is not None:
                    warm_snapshot = run_representation_checkpoint_evaluation(
                        policy,
                        expected_scope=representation_parameter_scope,
                        plan=representation_warm_evaluation_plan,
                        checkpoint_global_step=checkpoint_step,
                        implementation_sha256=implementation_sha256,
                        model_family_sha256=model_family_sha256,
                        representation_split_sha256=representation_split.artifact_sha256,
                        dataset=dataset,
                        collate_planned=collate_planned,
                        physical_sidecar=physical_sidecar,
                        capacity=args.capacity,
                        task_identity_resolver=calvin_exact_task_loss_identities,
                        patch_size=patch_size,
                        merge_size=merge_size,
                        structural_config=structural_config,
                        minimum_supervised_fraction=args.minimum_supervised_fraction,
                        output_root=(
                            args.run_dir
                            / "representation_warm_evaluations"
                            / f"global_step_{checkpoint_step}"
                        ),
                        rank=rank,
                        world_size=FULL_WORLD_SIZE,
                        dist_module=dist,
                        device=device,
                    )
                if fixed_observation_evaluation_plan is not None:
                    fixed_observation_snapshot = run_fixed_observation_checkpoint_evaluation(
                        policy,
                        expected_scope=representation_parameter_scope,
                        plan=fixed_observation_evaluation_plan,
                        checkpoint_global_step=checkpoint_step,
                        implementation_sha256=implementation_sha256,
                        model_family_sha256=model_family_sha256,
                        representation_split_sha256=(representation_split.artifact_sha256),
                        dataset=dataset,
                        collate_planned=collate_planned,
                        physical_sidecar=physical_sidecar,
                        capacity=args.capacity,
                        task_identity_resolver=calvin_exact_task_loss_identities,
                        patch_size=patch_size,
                        merge_size=merge_size,
                        structural_config=structural_config,
                        minimum_supervised_fraction=(args.minimum_supervised_fraction),
                        output_root=(
                            args.run_dir
                            / "fixed_observation_evaluations"
                            / f"global_step_{checkpoint_step}"
                        ),
                        rank=rank,
                        world_size=FULL_WORLD_SIZE,
                        dist_module=dist,
                        device=device,
                    )
            finally:
                _restore_rank_rng(evaluation_rng, torch, np, device=device)
            if _rank_rng_digest(_capture_rank_rng(torch, np, device=device)) != (
                evaluation_rng_sha256
            ):
                raise RuntimeError("representation checkpoint evaluation changed process RNG")
            torch.cuda.empty_cache()
            if rank == 0:
                if snapshot is None:
                    raise RuntimeError("rank zero representation evaluation returned no snapshot")
                print(
                    json.dumps(
                        {
                            "artifact_sha256": snapshot["artifact_sha256"],
                            "event": "representation_checkpoint_evaluation_completed",
                            "global_step": checkpoint_step,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if representation_warm_evaluation_plan is not None:
                    if warm_snapshot is None:
                        raise RuntimeError(
                            "rank zero warm representation evaluation returned no snapshot"
                        )
                    print(
                        json.dumps(
                            {
                                "artifact_sha256": warm_snapshot["artifact_sha256"],
                                "event": ("representation_warm_checkpoint_evaluation_completed"),
                                "global_step": checkpoint_step,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if fixed_observation_evaluation_plan is not None:
                    if fixed_observation_snapshot is None:
                        raise RuntimeError(
                            "rank zero fixed-observation evaluation returned no snapshot"
                        )
                    print(
                        json.dumps(
                            {
                                "artifact_sha256": (fixed_observation_snapshot["artifact_sha256"]),
                                "event": ("fixed_observation_checkpoint_evaluation_completed"),
                                "global_step": checkpoint_step,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            return snapshot

        if fixed_batch_trainable_scope is not None:
            _run_predictive_fixed_batch_arm(
                args=args,
                rank=rank,
                device=device,
                dist=dist,
                torch_module=torch,
                policy=policy,
                graph=graph,
                optimizer=optimizer,
                optimizer_contract=optimizer_contract,
                trainable_scope=fixed_batch_trainable_scope,
                stream_plan=stream_plan,
                dataset=dataset,
                collate_planned=collate_planned,
                build_planned_batch=build_planned_native_calvin_batch,
                build_continuation_batch=build_native_calvin_continuation_batch,
                run_full_objective=run_native_calvin_full_objective,
                predictive_cache=predictive_cache,
                current_grid_cache=current_grid_cache,
                physical_sidecar=physical_sidecar,
                task_identity_resolver=calvin_exact_task_loss_identities,
                patch_size=patch_size,
                merge_size=merge_size,
                objective_config=objective_config,
                structural_config=structural_config,
                derive_subseed_fn=derive_subseed,
                patch_sha256=patch_sha256,
                execution_sha256=execution_sha256,
                implementation_sha256=implementation_sha256,
                model_family_sha256=model_family_sha256,
                dataset_contract_report=dataset_contract_report,
            )
            return
        if relation_geometry_trainable_scope is not None:
            if relation_probe_sample_selection is None:
                raise RuntimeError("relation probe has no preflighted source-only sample")
            relation_objective_config = NativeObjectiveConfig(
                predictive_weight=0.0,
                structural_weight=args.structural_weight,
            )
            if args.relation_geometry_fixed_batch_arm in {
                RELATION_BILINEAR_PROBE_ARM,
                RELATION_DEPTH_PROBE_ARM,
            }:
                _run_external_relation_fixed_batch_arm(
                    args=args,
                    rank=rank,
                    device=device,
                    dist=dist,
                    torch_module=torch,
                    policy=policy,
                    graph=graph,
                    trainable_scope=relation_geometry_trainable_scope,
                    sample_selection=relation_probe_sample_selection,
                    stream_plan=stream_plan,
                    dataset=dataset,
                    collate_planned=collate_planned,
                    build_planned_batch=build_planned_native_calvin_batch,
                    build_continuation_batch=build_native_calvin_continuation_batch,
                    context_type=LingBotNativeContext,
                    run_policy_diagnostic=run_native_policy_diagnostic_forward,
                    run_observation_diagnostic=(run_native_policy_observation_diagnostic_forward),
                    compose_objective=compose_native_calvin_objective,
                    physical_sidecar=physical_sidecar,
                    task_identity_resolver=calvin_exact_task_loss_identities,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    objective_config=relation_objective_config,
                    structural_config=structural_config,
                    derive_subseed_fn=derive_subseed,
                    temporal_batch_seed_fn=native_temporal_batch_seed,
                    matched_row_soft_iou_fn=matched_row_soft_iou,
                    render_relation_visuals=render_native_relation_visuals,
                    patch_sha256=patch_sha256,
                    execution_sha256=execution_sha256,
                    implementation_sha256=implementation_sha256,
                    model_family_sha256=model_family_sha256,
                    dataset_contract_report=dataset_contract_report,
                )
            else:
                _run_relation_geometry_fixed_batch_arm(
                    args=args,
                    rank=rank,
                    device=device,
                    dist=dist,
                    torch_module=torch,
                    policy=policy,
                    graph=graph,
                    optimizer=optimizer,
                    optimizer_contract=optimizer_contract,
                    trainable_scope=relation_geometry_trainable_scope,
                    sample_selection=relation_probe_sample_selection,
                    stream_plan=stream_plan,
                    dataset=dataset,
                    collate_planned=collate_planned,
                    build_planned_batch=build_planned_native_calvin_batch,
                    build_continuation_batch=build_native_calvin_continuation_batch,
                    run_relation_objective=run_native_calvin_relation_probe_objective,
                    physical_sidecar=physical_sidecar,
                    task_identity_resolver=calvin_exact_task_loss_identities,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    objective_config=relation_objective_config,
                    structural_config=structural_config,
                    derive_subseed_fn=derive_subseed,
                    temporal_batch_seed_fn=native_temporal_batch_seed,
                    render_relation_visuals=render_native_relation_visuals,
                    patch_sha256=patch_sha256,
                    execution_sha256=execution_sha256,
                    implementation_sha256=implementation_sha256,
                    model_family_sha256=model_family_sha256,
                    dataset_contract_report=dataset_contract_report,
                )
            return
        initial_evaluation_snapshot: dict[str, object] | None = None
        if global_step in args.representation_evaluation_steps:
            initial_evaluation_snapshot = evaluate_representation_checkpoint(global_step)
        if representation_evaluation_baseline is not None and global_step == 0:
            baseline_error: BaseException | None = None
            if rank == 0:
                try:
                    if (
                        initial_evaluation_snapshot is None
                        or representation_evaluation_plan is None
                    ):
                        raise RuntimeError("step-zero baseline replay lost its evaluation snapshot")
                    step_zero_root = args.run_dir / "representation_evaluations" / "global_step_0"
                    replay_report = build_representation_baseline_replay_report(
                        baseline=representation_evaluation_baseline,
                        candidate_snapshot=initial_evaluation_snapshot,
                        candidate_plan=representation_evaluation_plan,
                        candidate_visual_root=step_zero_root,
                    )
                    write_representation_baseline_replay_report(
                        step_zero_root / "representation_baseline_replay_report.json",
                        replay_report,
                    )
                except BaseException as error:
                    baseline_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                stage="representation step-zero baseline replay",
                local_error=baseline_error,
            )
            dist.barrier()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "baseline_artifact_sha256": (
                                representation_evaluation_baseline["artifact_sha256"]
                            ),
                            "event": "representation_step_zero_baseline_replay_passed",
                            "global_step": 0,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        coordinator = NativeTrainingLaneCoordinator(bank)
        step_reports: list[dict[str, Any]] = []
        final_source_digest = ""

        def ring_exchange_controls(value: ExecutedControlBatch) -> ExecutedControlBatch:
            return ExecutedControlBatch(
                values=_distributed_ring_exchange_tensor(
                    value.values,
                    dist=dist,
                    torch_module=torch,
                ),
                field_valid=_distributed_ring_exchange_tensor(
                    value.field_valid,
                    dist=dist,
                    torch_module=torch,
                ),
                token_valid=_distributed_ring_exchange_tensor(
                    value.token_valid,
                    dist=dist,
                    torch_module=torch,
                ),
                delta_time=_distributed_ring_exchange_tensor(
                    value.delta_time,
                    dist=dist,
                    torch_module=torch,
                ),
                reset=_distributed_ring_exchange_tensor(
                    value.reset,
                    dist=dist,
                    torch_module=torch,
                ),
                acknowledged=_distributed_ring_exchange_tensor(
                    value.acknowledged,
                    dist=dist,
                    torch_module=torch,
                ),
            )

        for _invocation_index in range(args.invocation_steps):
            # End-to-end optimizer iteration timing includes data/cache materialization,
            # temporal planning, every forward/backward, counterfactual probes and update.
            # Periodic visualization remains a separately reported post-step diagnostic.
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            estimator_component = (
                stream_plan.component_for_step(global_step)
                if isinstance(stream_plan, FrozenResetMixtureStreamPlan)
                else "causal"
            )
            posterior_committed = estimator_component == "causal"
            posterior_bank_sha256_before = _posterior_bank_digest(bank)
            _emit_step_progress(
                "step_started",
                rank=rank,
                global_step=global_step + 1,
                details={"input_global_step": global_step},
            )
            planned = None
            primary_batch = None
            fixed_observation_fingerprint: dict[str, object] | None = None
            batch_materialization_error: BaseException | None = None
            try:
                planned = build_planned_native_calvin_batch(
                    stream_plan,
                    dataset,
                    optimizer_step=global_step,
                    rank=rank,
                    world_size=FULL_WORLD_SIZE,
                    gradient_accumulation_steps=1,
                    accumulation_index=0,
                    device=device,
                    dtype=torch.bfloat16,
                )
                planned = apply_training_prompt_overlay(planned)
                if planned.training.routing.batch_size != 1:
                    raise RuntimeError("native full two-rank contract requires one sample per rank")
                primary_batch = collate_planned(planned)
                if planned.fixed_observation_pair_sha256 is not None:
                    fixed_observation_fingerprint = fixed_observation_training_pair_fingerprint(
                        primary_batch
                    )
            except BaseException as error:
                batch_materialization_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                stage="training batch materialization",
                local_error=batch_materialization_error,
            )
            if planned is None or primary_batch is None:
                raise RuntimeError("distributed batch materialization lost its local result")
            attempt = coordinator.begin(optimizer_step=global_step, source_weight_version=0)
            prepared = attempt.prepare(primary_batch.routing)
            local_metadata = {
                "sample_keys": primary_batch.routing.sample_keys,
                "state_ages": prepared.next_state_ages,
                "available_future": tuple(
                    dataset.available_future_transitions_by_key(key)
                    for key in primary_batch.routing.sample_keys
                ),
                "optimizer_lags": prepared.optimizer_lags,
                "fixed_observation_pair_sha256": (planned.fixed_observation_pair_sha256),
                "fixed_observation_fingerprint": fixed_observation_fingerprint,
            }
            gathered_metadata: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
            dist.all_gather_object(gathered_metadata, local_metadata)
            fixed_observation_pair_verified = validate_fixed_observation_training_rank_metadata(
                gathered_metadata,
                expected_world_size=FULL_WORLD_SIZE,
            )
            _emit_step_progress(
                "step_batch_ready",
                rank=rank,
                global_step=global_step + 1,
                details={
                    "fixed_observation_pair_verified": fixed_observation_pair_verified,
                    "frame_indices": list(primary_batch.routing.frame_indices),
                    "sample_keys": list(primary_batch.routing.sample_keys),
                },
            )
            global_sample_keys = tuple(
                key for item in gathered_metadata for key in item["sample_keys"]
            )
            temporal_seed = native_temporal_batch_seed(
                parent_seed=args.seed,
                comparison_id=FULL_COMPARISON_ID,
                optimizer_step=global_step,
                sample_keys=global_sample_keys,
            )
            temporal = sample_temporal_batch_plan(
                temporal_config,
                seed=temporal_seed,
                state_ages=tuple(
                    value for item in gathered_metadata for value in item["state_ages"]
                ),
                available_future_steps=tuple(
                    value for item in gathered_metadata for value in item["available_future"]
                ),
                optimizer_lags=tuple(
                    value for item in gathered_metadata for value in item["optimizer_lags"]
                ),
            )
            if planned.fixed_observation_pair_sha256 is not None and estimator_component != "reset":
                raise RuntimeError("fixed-observation overlay reached a causal estimator step")
            temporal = _fixed_observation_primary_temporal_plan(
                temporal,
                pair_plan_sha256=planned.fixed_observation_pair_sha256,
            )
            source_mask = None
            omitted_static_view = None
            if temporal.source_masked_branch:
                source_seed = derive_subseed(
                    temporal_seed,
                    f"{args.source_prediction_mode}-source-omission",
                    primary_batch.routing.sample_keys[0],
                )
                if args.source_prediction_mode == "current_grid":
                    source_mask = sample_qwen_packed_patch_mask(
                        images=primary_batch.model_inputs["images"],
                        image_valid=primary_batch.model_inputs["img_masks"],
                        image_grid_thw=primary_batch.model_inputs["image_grid_thw"],
                        spatial_merge_size=merge_size,
                        probability=args.source_mask_token_fraction,
                        seed=source_seed,
                        eligible_view_indices=(0,),
                    )
                    if source_mask.query_count <= 0:
                        raise RuntimeError("enabled current-grid branch sampled no query")
                else:
                    omitted_static_view = sample_qwen_whole_view_omission(
                        images=primary_batch.model_inputs["images"],
                        image_valid=primary_batch.model_inputs["img_masks"],
                        image_grid_thw=primary_batch.model_inputs["image_grid_thw"],
                        seed=source_seed,
                        eligible_view_indices=(0,),
                    )

            local_count, overshoot_count = _temporal_execution_counts(
                local_bptt_steps=temporal.local_bptt_steps,
                overshoot_horizon=temporal.overshoot_horizon,
            )
            behavior_horizon = 1 if args.behavior_conditioned_prediction else 0
            maximum_offset = max(local_count - 1, overshoot_count, behavior_horizon)
            continuation_batches: dict[int, Any] = {}
            for offset in range(1, maximum_offset + 1):
                continuation_batches[offset] = collate_planned(
                    build_native_calvin_continuation_batch(
                        planned,
                        dataset,
                        offset=offset,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                )
            full_batches = (primary_batch,) + tuple(
                continuation_batches[offset] for offset in range(1, local_count)
            )
            overshoot_factory: NativeOvershootFactory | None = None
            build_overshoot: NativeOvershootFactory | None = None
            controls: tuple[Any, ...] = ()
            behavior_prediction_controls: ExecutedControlBatch | None = None
            behavior_prediction_horizon: int | None = None
            if args.behavior_conditioned_prediction:
                behavior_prediction_horizon = behavior_horizon
                controls = tuple(
                    continuation_batches[offset].controls
                    for offset in range(1, behavior_prediction_horizon + 1)
                )
                behavior_prediction_controls = concatenate_executed_controls(controls)
            elif temporal.overshoot_horizon is not None:
                horizon = temporal.overshoot_horizon
                controls = tuple(
                    continuation_batches[offset].controls for offset in range(1, horizon + 1)
                )

                def overshoot_callback(
                    state: Any,
                    *,
                    horizon: int = horizon,
                    controls: tuple[Any, ...] = controls,
                ) -> Any:
                    request = make_native_future_request(
                        source=PredictionSource.PRIOR,
                        batch_size=state.batch_size,
                        horizon=horizon,
                        valid=torch.ones(state.batch_size, dtype=torch.bool, device=device),
                        device=device,
                        dtype=state.rows.dtype,
                        route_id=predictive_cache.contract.route_id,
                        address_width=graph.config.prediction_address_width,
                    )
                    return rollout_native_prior_prediction(
                        prior_stepper,
                        state,
                        controls,
                        request=request,
                        target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
                    )

                build_overshoot = overshoot_callback
                overshoot_factory = build_overshoot
                del overshoot_callback

            capacity_seeds = tuple(
                derive_subseed(temporal_seed, "capacity-censor", key)
                for key in primary_batch.routing.sample_keys
            )
            family_gradient_diagnostics = None
            relation_surface_gradient_diagnostics = None
            predictive_host_gradient_diagnostics = None
            predictive_counterfactual_report = None
            predictive_counterfactual_weight_boundary = None
            gradient_audit = global_step + 1 in args.gradient_audit_steps
            objective_runner = (
                run_native_calvin_representation_objective
                if representation_stage
                else run_native_calvin_full_objective
            )
            objective_common_kwargs = dict(
                graph=graph,
                previous_state=prepared.previous_state,
                previous_state_valid=prepared.previous_state_valid,
                predictive_cache=predictive_cache,
                current_grid_cache=current_grid_cache,
                physical_sidecar=physical_sidecar,
                capacity=args.capacity,
                task_identity_resolver=calvin_exact_task_loss_identities,
                patch_size=patch_size,
                merge_size=merge_size,
                objective_config=objective_config,
                structural_config=structural_config,
                behavior_prediction_controls=behavior_prediction_controls,
                behavior_prediction_horizon=behavior_prediction_horizon,
                predictive_term_weight=args.predictive_term_weight,
                current_grid_term_weight=args.current_grid_term_weight,
                omitted_static_term_weight=args.omitted_static_term_weight,
                predictive_loss_power=args.predictive_loss_power,
                minimum_supervised_fraction=args.minimum_supervised_fraction,
                capacity_seeds=capacity_seeds,
                prior_row_bindings_by_batch=prepared.previous_row_bindings,
            )
            run_step_objective: Callable[[], Any] = partial(
                objective_runner,
                policy,
                batches=full_batches,
                source_mask=source_mask,
                omitted_static_view=omitted_static_view,
                overshoot_factory=overshoot_factory,
                **objective_common_kwargs,
            )
            run_gradient_audit_objective: Callable[[], Any] = partial(
                objective_runner,
                policy,
                batches=(primary_batch,),
                source_mask=None,
                omitted_static_view=None,
                overshoot_factory=None,
                **objective_common_kwargs,
            )

            if args.behavior_causal_probe_output is not None:
                if behavior_prediction_controls is None or behavior_prediction_horizon is None:
                    raise RuntimeError("behavior causal G0 lost its validated control contract")
                behavior_request = make_native_future_request(
                    source=PredictionSource.PRIOR,
                    batch_size=primary_batch.routing.batch_size,
                    horizon=behavior_prediction_horizon,
                    valid=torch.ones(
                        primary_batch.routing.batch_size,
                        dtype=torch.bool,
                        device=device,
                    ),
                    device=device,
                    dtype=primary_batch.controls.values.dtype,
                    route_id=predictive_cache.contract.route_id,
                    address_width=graph.config.prediction_address_width,
                )
                peer_prediction_controls = ring_exchange_controls(behavior_prediction_controls)
                probe_started = time.perf_counter()
                local_probe = run_native_behavior_causal_probe(
                    policy,
                    graph=graph,
                    model_inputs=primary_batch.model_inputs,
                    controls=primary_batch.controls,
                    previous_state=prepared.previous_state,
                    previous_state_valid=prepared.previous_state_valid,
                    request=behavior_request,
                    prediction_controls=behavior_prediction_controls,
                    peer_prediction_controls=peer_prediction_controls,
                    modalities=primary_batch.modalities,
                )
                _require_unchanged_behavior_graph(
                    args,
                    expected_sha256=behavior_graph_sha256,
                )
                local_probe_report = {
                    "rank": rank,
                    **local_probe.as_dict(),
                    "elapsed_s": time.perf_counter() - probe_started,
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
                gathered_probes: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
                dist.all_gather_object(gathered_probes, local_probe_report)
                probe_payload = {
                    "schema": BEHAVIOR_CAUSAL_PROBE_DISTRIBUTED_SCHEMA,
                    "status": "PASS",
                    "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                    "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                    "patch_sha256": patch_sha256,
                    "behavior_graph_sha256": behavior_graph_sha256,
                    "implementation_sha256": implementation_sha256,
                    "model_family_sha256": model_family_sha256,
                    "plan_sha256": stream_plan.plan_sha256,
                    "weight_boundary": "released_pre_optimizer",
                    "optimizer_updates": 0,
                    "sample_keys_by_rank": _behavior_probe_sample_keys_by_rank(gathered_metadata),
                    "rank_reports": gathered_probes,
                }
                validate_behavior_causal_probe_evidence(probe_payload)
                publication_error: list[str | None] = [None]
                if rank == 0:
                    try:
                        output = args.behavior_causal_probe_output
                        output.parent.mkdir(parents=True, exist_ok=True)
                        _write_text_durable(
                            output,
                            json.dumps(probe_payload, indent=2, sort_keys=True) + "\n",
                        )
                    except (OSError, RuntimeError, TypeError, ValueError) as error:
                        publication_error[0] = f"{type(error).__name__}: {error}"
                dist.broadcast_object_list(publication_error, src=0)
                if publication_error[0] is not None:
                    raise RuntimeError(
                        "behavior causal G0 publication failed: " + publication_error[0]
                    )
                attempt.abort()
                optimizer.zero_grad(set_to_none=True)
                dist.barrier()
                if rank == 0:
                    print(json.dumps(probe_payload, indent=2, sort_keys=True), flush=True)
                return

            behavior_posterior_gradient_norm = 0.0
            behavior_posterior_gradient_elements = 0
            try:
                if gradient_audit:
                    audit_rng_state = _capture_rank_rng(torch, np, device=device)
                    audit_rng_sha256 = _rank_rng_digest(audit_rng_state)
                    routing_bias = _moe_routing_bias_snapshot(policy, torch_module=torch)
                    family_gradients: dict[str, Any] = {}
                    relation_component_gradients: dict[str, Any] | None = None
                    host_gradients: dict[str, tuple[str, Any]] | None = None
                    behavior_host_decomposition: dict[str, dict[str, Any]] | None = None
                    family_shared_host = (
                        _predictive_host_gradient_parameters(policy)
                        if representation_stage
                        else None
                    )
                    for family_name in ("action", "predictive", "structural"):
                        posterior_host_vjp: dict[str, tuple[str, Any]] | None = None
                        if family_name == "predictive" and args.behavior_conditioned_prediction:
                            selected_host = _predictive_host_gradient_parameters(policy)

                            _restore_rank_rng(audit_rng_state, torch, np, device=device)
                            optimizer.zero_grad(set_to_none=True)
                            _emit_step_progress(
                                "gradient_audit_replay_started",
                                rank=rank,
                                global_step=global_step + 1,
                                details={"component": "total_host", "family": "predictive"},
                            )
                            total_result = run_gradient_audit_objective()
                            total_term = _weighted_behavior_future_contribution(
                                total_result,
                                predictive_family_weight=args.predictive_weight,
                                torch_module=torch,
                            )
                            total_snapshots = _backward_behavior_total_host(
                                total_term,
                                selected_host=selected_host,
                                torch_module=torch,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            _reset_moe_probe_counters(policy, torch_module=torch)
                            del total_result, total_term
                            if not _moe_routing_bias_matches(
                                policy,
                                routing_bias,
                                torch_module=torch,
                            ):
                                raise RuntimeError(
                                    "total behavior-gradient replay changed LingBot "
                                    "MoE routing bias"
                                )

                            _restore_rank_rng(audit_rng_state, torch, np, device=device)
                            optimizer.zero_grad(set_to_none=True)
                            _emit_step_progress(
                                "gradient_audit_replay_started",
                                rank=rank,
                                global_step=global_step + 1,
                                details={"component": "via_posterior", "family": "predictive"},
                            )
                            via_result = run_gradient_audit_objective()
                            via_term = _weighted_behavior_future_contribution(
                                via_result,
                                predictive_family_weight=args.predictive_weight,
                                torch_module=torch,
                            )
                            behavior_state = _objective_result_posterior_state(via_result)
                            posterior_credit, via_snapshots = _backward_behavior_via_posterior_host(
                                via_term,
                                behavior_rows=behavior_state.rows,
                                selected_host=selected_host,
                                torch_module=torch,
                            )
                            behavior_posterior_gradient_norm = float(
                                posterior_credit.detach().float().norm().item()
                            )
                            behavior_posterior_gradient_elements = int(posterior_credit.numel())
                            behavior_host_decomposition = {
                                "total": total_snapshots,
                                "via_posterior": via_snapshots,
                                "direct": {
                                    depth: total_snapshots[depth] - via_snapshots[depth]
                                    for depth in selected_host
                                },
                            }
                            posterior_host_vjp = {
                                depth: (
                                    selected_host[depth][0],
                                    via_snapshots[depth],
                                )
                                for depth in selected_host
                            }
                            optimizer.zero_grad(set_to_none=True)
                            _reset_moe_probe_counters(policy, torch_module=torch)
                            del via_result, via_term, behavior_state, posterior_credit
                            if not _moe_routing_bias_matches(
                                policy,
                                routing_bias,
                                torch_module=torch,
                            ):
                                raise RuntimeError(
                                    "posterior behavior-gradient replay changed LingBot "
                                    "MoE routing bias"
                                )

                        _restore_rank_rng(audit_rng_state, torch, np, device=device)
                        optimizer.zero_grad(set_to_none=True)
                        _emit_step_progress(
                            "gradient_audit_replay_started",
                            rank=rank,
                            global_step=global_step + 1,
                            details={"component": "isolated_family", "family": family_name},
                        )
                        diagnostic_result = run_gradient_audit_objective()
                        family_terms = diagnostic_result.objective.objective.family_terms
                        if family_name == "structural":
                            relation_component_gradients = _relation_surface_component_gradients(
                                diagnostic_result,
                                torch_module=torch,
                            )
                        _backward_isolated_objective_family(
                            family_terms,
                            selected_name=family_name,
                            torch_module=torch,
                        )
                        family_gradients[family_name] = (
                            _shared_host_family_gradient_snapshot(
                                family_shared_host,
                                family_name=family_name,
                                torch_module=torch,
                            )
                            if family_shared_host is not None
                            else _parameter_gradient_snapshot(
                                graph.object_queries,
                                name=f"{family_name} object-query",
                                torch_module=torch,
                            )
                        )
                        if family_name == "predictive":
                            host_gradients = (
                                posterior_host_vjp
                                if posterior_host_vjp is not None
                                else {
                                    depth: (
                                        path,
                                        _parameter_gradient_snapshot(
                                            parameter,
                                            name=f"predictive {depth} shared-host",
                                            torch_module=torch,
                                        ),
                                    )
                                    for depth, (
                                        path,
                                        parameter,
                                    ) in _predictive_host_gradient_parameters(policy).items()
                                }
                            )
                        optimizer.zero_grad(set_to_none=True)
                        _reset_moe_probe_counters(policy, torch_module=torch)
                        del diagnostic_result, family_terms
                        if not _moe_routing_bias_matches(
                            policy,
                            routing_bias,
                            torch_module=torch,
                        ):
                            raise RuntimeError(
                                "isolated family-gradient audit changed LingBot MoE routing bias"
                            )
                    if host_gradients is None:
                        raise RuntimeError(
                            "predictive family audit captured no shared-host gradient"
                        )
                    if relation_component_gradients is None:
                        raise RuntimeError(
                            "structural family audit captured no relation-surface gradients"
                        )
                    family_gradient_diagnostics = _distributed_family_gradient_diagnostics(
                        family_gradients=family_gradients,
                        probe=(
                            REPRESENTATION_FAMILY_GRADIENT_PROBE
                            if representation_stage
                            else "picf_native_graph.object_queries"
                        ),
                        device=device,
                        dist=dist,
                        torch_module=torch,
                    )
                    relation_surface_gradient_diagnostics = (
                        _distributed_relation_surface_gradient_diagnostics(
                            component_gradients=relation_component_gradients,
                            device=device,
                            dist=dist,
                            torch_module=torch,
                        )
                    )
                    predictive_host_gradient_diagnostics = (
                        _distributed_predictive_host_gradient_diagnostics(
                            host_gradients=host_gradients,
                            decomposition_gradients=behavior_host_decomposition,
                            probe=(
                                "lingbot.language_model.input_layernorm.via_primary_posterior_vjp"
                                if args.behavior_conditioned_prediction
                                else "lingbot.language_model.input_layernorm"
                            ),
                            device=device,
                            dist=dist,
                            torch_module=torch,
                        )
                    )
                    _restore_rank_rng(audit_rng_state, torch, np, device=device)
                    if _rank_rng_digest(_capture_rank_rng(torch, np, device=device)) != (
                        audit_rng_sha256
                    ):
                        raise RuntimeError("isolated family-gradient audit changed process RNG")
                    optimizer.zero_grad(set_to_none=True)
                    del (
                        audit_rng_state,
                        family_gradients,
                        host_gradients,
                        relation_component_gradients,
                        routing_bias,
                    )
                _emit_step_progress(
                    "objective_started",
                    rank=rank,
                    global_step=global_step + 1,
                )
                result = None
                objective_error: BaseException | None = None
                try:
                    result = run_step_objective()
                except BaseException as error:
                    objective_error = error
                objective_failures = _distributed_pre_backward_failures(
                    objective_error,
                    rank=rank,
                    expected_world_size=FULL_WORLD_SIZE,
                    dist=dist,
                )
                if objective_failures:
                    if objective_error is not None:
                        raise objective_error
                    details = "; ".join(
                        f"rank {item['rank']} {item['type']}: {item['message']}"
                        for item in objective_failures
                    )
                    raise RuntimeError(f"peer objective failed before backward: {details}")
                if result is None:
                    raise RuntimeError("all ranks reported success without an objective result")
                _emit_step_progress(
                    "objective_ready",
                    rank=rank,
                    global_step=global_step + 1,
                )
                if args.behavior_posterior_control_probe_output is not None:
                    if (
                        not isinstance(result, NativeRepresentationObjectiveStepResult)
                        or behavior_prediction_controls is None
                        or behavior_g1_predecessor is None
                        or behavior_g1_predecessor_sha256 is None
                    ):
                        raise RuntimeError("behavior G2 lost its validated representation boundary")
                    future_inputs = _future_loss_inputs(
                        branches=result.future_branches,
                        cache=predictive_cache,
                        track_identity_keys=result.objective.track_identity_keys_by_batch,
                        weight=1.0,
                        loss_power=args.predictive_loss_power,
                    )
                    if len(future_inputs) != 1:
                        raise RuntimeError(
                            "behavior G2 requires exactly one controlled-future target"
                        )
                    future_input = future_inputs[0]
                    current_state = result.primary.posterior_state
                    if current_state is None:
                        raise RuntimeError("behavior G2 objective omitted the current posterior")
                    factual_state = NativePosteriorState(current_state.rows.detach().clone())
                    identity_source_phase = future_input.identity_source_phase
                    row_binding_valid = (
                        assignment_binding_valid_at_phase(
                            result.objective.assignment,
                            result.objective.targets,
                            source_phase=identity_source_phase,
                        )
                        .detach()
                        .clone()
                    )
                    assignment = SequenceAssignment(
                        result.objective.assignment.row_to_track.detach().clone(),
                        assignment_binding_start_phase(
                            result.objective.assignment,
                            result.objective.targets,
                        )
                        .detach()
                        .clone(),
                    )
                    target = future_input.target
                    request = future_input.request
                    training_prediction = future_input.prediction.detach().clone()
                    target_identity_keys = result.objective.track_identity_keys_by_batch
                    del current_state, future_input, future_inputs, result
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()

                    peer_state = NativePosteriorState(
                        _distributed_ring_exchange_tensor(
                            factual_state.rows,
                            dist=dist,
                            torch_module=torch,
                        )
                    )
                    peer_prediction_controls = ring_exchange_controls(behavior_prediction_controls)
                    rng_before = _capture_rank_rng(torch, np, device=device)
                    rng_before_sha256 = _rank_rng_digest(rng_before)
                    optimizer_before = _validate_optimizer_state(
                        optimizer,
                        torch,
                        expected_step=args.load_global_step,
                    )
                    lane_before = _posterior_bank_digest(bank)
                    routing_bias_before = _moe_routing_bias_snapshot(
                        policy,
                        torch_module=torch,
                    )
                    probe_started = time.perf_counter()
                    factorial_predictions = run_native_behavior_posterior_control_forwards(
                        policy,
                        graph=graph,
                        factual_state=factual_state,
                        peer_state=peer_state,
                        request=request,
                        prediction_controls=behavior_prediction_controls,
                        peer_prediction_controls=peer_prediction_controls,
                    )
                    factual_probe_prediction = factorial_predictions.prediction_for(
                        "factual",
                        "factual",
                    )
                    training_prediction_bit_identical = torch.equal(
                        training_prediction,
                        factual_probe_prediction,
                    )
                    if not training_prediction_bit_identical:
                        raise RuntimeError(
                            "behavior G2 factual probe differs from the attached training branch"
                        )
                    training_prediction_sha256 = cpu_tensor_sha256(
                        training_prediction.detach().to(device="cpu")
                    )
                    factual_prediction_sha256 = cpu_tensor_sha256(
                        factual_probe_prediction.detach().to(device="cpu")
                    )
                    factorial_diagnostics = behavior_posterior_control_diagnostics(
                        factorial_predictions,
                        target=target,
                        assignment=assignment,
                        row_binding_valid=row_binding_valid,
                        loss_power=args.predictive_loss_power,
                    )
                    rng_after_sha256 = _rank_rng_digest(_capture_rank_rng(torch, np, device=device))
                    optimizer_after = _validate_optimizer_state(
                        optimizer,
                        torch,
                        expected_step=args.load_global_step,
                    )
                    lane_after = _posterior_bank_digest(bank)
                    if (
                        rng_after_sha256 != rng_before_sha256
                        or optimizer_after != optimizer_before
                        or lane_after != lane_before
                        or any(parameter.grad is not None for parameter in policy.parameters())
                        or not _moe_routing_bias_matches(
                            policy,
                            routing_bias_before,
                            torch_module=torch,
                        )
                    ):
                        raise RuntimeError(
                            "behavior G2 changed RNG, optimizer, lane, gradients or MoE bias"
                        )
                    _require_unchanged_behavior_graph(
                        args,
                        expected_sha256=behavior_graph_sha256,
                    )
                    valid_importance = target.importance[target.valid]
                    local_factorial_report = {
                        "rank": rank,
                        "sample_keys": list(primary_batch.routing.sample_keys),
                        "peer_sample_keys": list(
                            gathered_metadata[(rank + 1) % FULL_WORLD_SIZE]["sample_keys"]
                        ),
                        "tasks": [item["task"] for item in planned.training.host_items],
                        "elapsed_s": time.perf_counter() - probe_started,
                        "diagnostics": factorial_diagnostics.as_dict(),
                        "training_prediction_sha256": training_prediction_sha256,
                        "factual_prediction_sha256": factual_prediction_sha256,
                        "training_prediction_bit_identical": (training_prediction_bit_identical),
                        "target": {
                            "modality": target.modality,
                            "source_batch_digest": target.source_batch_digest,
                            "target_data_digest": target.target_data_digest,
                            "encoder_digest": target.encoder_digest,
                            "query_schema_digest": target.query_schema_digest,
                            "validity_semantics": target.validity_semantics,
                            "track_identity_keys": [list(keys) for keys in target_identity_keys],
                            "valid_count": int(target.valid.sum().item()),
                            "importance_sum": float(valid_importance.sum().item()),
                            "importance_min": float(valid_importance.min().item()),
                            "importance_max": float(valid_importance.max().item()),
                        },
                        "assignment": {
                            "row_to_track": assignment.row_to_track.tolist(),
                            "binding_start_phase": (
                                assignment.binding_start_phase.tolist()
                                if assignment.binding_start_phase is not None
                                else None
                            ),
                            "identity_source_phase": identity_source_phase,
                            "row_binding_valid": row_binding_valid.tolist(),
                            "sha256": _canonical_digest(
                                {
                                    "row_to_track": assignment.row_to_track.tolist(),
                                    "binding_start_phase": (
                                        assignment.binding_start_phase.tolist()
                                        if assignment.binding_start_phase is not None
                                        else None
                                    ),
                                    "identity_source_phase": identity_source_phase,
                                    "row_binding_valid": row_binding_valid.tolist(),
                                }
                            ),
                        },
                        "request_sha256": _canonical_digest(
                            {
                                "route_ids": request.route_ids.tolist(),
                                "horizons": request.horizons.tolist(),
                                "addresses": request.addresses.float().tolist(),
                                "valid": request.valid.tolist(),
                            }
                        ),
                        "rng_sha256": rng_before_sha256,
                        "rng_unchanged": True,
                        "optimizer_state_unchanged": True,
                        "posterior_bank_unchanged": True,
                        "moe_routing_bias_unchanged": True,
                        "loaded_boundary_sha256": loaded_boundary,
                        "factual_repeat_bit_identical": True,
                        "loss_only_labels_visible_to_model": False,
                        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    }
                    gathered_factorials: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
                    dist.all_gather_object(gathered_factorials, local_factorial_report)
                    posterior_margin_names = ("zero", "batch_shift")
                    control_margin_names = ("zero", "batch_shift")
                    aggregate_posterior_margins = {
                        name: sum(
                            report["diagnostics"]["posterior_margins_at_factual_control"][name]
                            for report in gathered_factorials
                        )
                        / FULL_WORLD_SIZE
                        for name in posterior_margin_names
                    }
                    aggregate_control_margins = {
                        name: sum(
                            report["diagnostics"]["control_margins_at_factual_posterior"][name]
                            for report in gathered_factorials
                        )
                        / FULL_WORLD_SIZE
                        for name in control_margin_names
                    }
                    scientific_pass = all(
                        value > 0
                        for value in (
                            *aggregate_posterior_margins.values(),
                            *aggregate_control_margins.values(),
                        )
                    )
                    probe_payload = {
                        "schema": BEHAVIOR_POSTERIOR_CONTROL_PROBE_DISTRIBUTED_SCHEMA,
                        "status": "PASS",
                        "scientific_status": "PASS" if scientific_pass else "FAIL",
                        "scientific_rule": (
                            "both rank-mean factual-axis margins must be positive for every "
                            "zero and batch-shift intervention"
                        ),
                        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                        "patch_sha256": patch_sha256,
                        "behavior_graph_sha256": behavior_graph_sha256,
                        "current_implementation_sha256": implementation_sha256,
                        "current_execution_contract_sha256": execution_sha256,
                        "model_family_sha256": model_family_sha256,
                        "plan_sha256": stream_plan.plan_sha256,
                        "g0_evidence_sha256": args.behavior_causal_probe_evidence_sha256,
                        "g1_predecessor_report_sha256": behavior_g1_predecessor_sha256,
                        "g1_predecessor_implementation_sha256": (
                            behavior_resume_implementation_sha256
                        ),
                        "g1_predecessor_execution_contract_sha256": (
                            behavior_resume_execution_sha256
                        ),
                        "input_global_step": args.load_global_step,
                        "weight_boundary": "loaded_g1_step2_pre_optimizer",
                        "optimizer_updates": 0,
                        "checkpoint_publication": "never",
                        "loaded_boundary_sha256_by_rank": [
                            report["loaded_boundary_sha256"] for report in gathered_factorials
                        ],
                        "aggregate_posterior_margins_at_factual_control": (
                            aggregate_posterior_margins
                        ),
                        "aggregate_control_margins_at_factual_posterior": (
                            aggregate_control_margins
                        ),
                        "rank_reports": gathered_factorials,
                    }
                    publication_error: list[str | None] = [None]
                    if rank == 0:
                        try:
                            output = args.behavior_posterior_control_probe_output
                            output.parent.mkdir(parents=True, exist_ok=True)
                            _write_text_durable(
                                output,
                                json.dumps(probe_payload, indent=2, sort_keys=True) + "\n",
                            )
                        except (OSError, RuntimeError, TypeError, ValueError) as error:
                            publication_error[0] = f"{type(error).__name__}: {error}"
                    dist.broadcast_object_list(publication_error, src=0)
                    if publication_error[0] is not None:
                        raise RuntimeError(
                            "behavior G2 publication failed: " + publication_error[0]
                        )
                    attempt.abort()
                    optimizer.zero_grad(set_to_none=True)
                    dist.barrier()
                    if rank == 0:
                        print(json.dumps(probe_payload, indent=2, sort_keys=True), flush=True)
                    return
                _emit_step_progress(
                    "backward_started",
                    rank=rank,
                    global_step=global_step + 1,
                )
                result.objective.objective.total.backward()
                _emit_step_progress(
                    "backward_completed",
                    rank=rank,
                    global_step=global_step + 1,
                )
                if args.behavior_conditioned_prediction:
                    behavior_state = _objective_result_posterior_state(result)
                    if behavior_state.rows.grad is None:
                        raise RuntimeError(
                            "behavior objective did not retain deploy-posterior credit"
                        )
                    total_posterior_credit = behavior_state.rows.grad.detach().float()
                    if not bool(torch.isfinite(total_posterior_credit).all().item()):
                        raise FloatingPointError("behavior deploy-posterior credit is non-finite")
                    if not bool((total_posterior_credit != 0).any().item()):
                        raise RuntimeError(
                            "behavior objective produced zero deploy-posterior credit"
                        )
                    if not gradient_audit:
                        behavior_posterior_gradient_norm = float(
                            total_posterior_credit.norm().item()
                        )
                        behavior_posterior_gradient_elements = int(total_posterior_credit.numel())
                    if (
                        behavior_posterior_gradient_norm <= 0
                        or behavior_posterior_gradient_elements <= 0
                    ):
                        raise RuntimeError(
                            "behavior objective did not persist deploy-posterior credit"
                        )
                if gradient_audit and not args.behavior_conditioned_prediction:
                    factual_prior_ready = _distributed_uniform_boolean(
                        bool(prepared.previous_state_valid.all().item()),
                        name="predictive factual-prior availability",
                        device=device,
                        dist=dist,
                        torch_module=torch,
                    )
                    if not factual_prior_ready:
                        raise RuntimeError(
                            "a predictive counterfactual audit requires valid factual prior state"
                        )
                    has_wrong_time = bool(prepared.wrong_time_state_valid.all().item())
                    execute_wrong_time = _distributed_any_boolean(
                        has_wrong_time,
                        name="predictive wrong-time availability",
                        device=device,
                        dist=dist,
                        torch_module=torch,
                    )
                    mature = all(frame >= 2 for frame in primary_batch.routing.frame_indices)
                    if mature and not has_wrong_time:
                        raise RuntimeError(
                            "a mature predictive audit lost its checkpointed wrong-time state"
                        )
                    if not mature and has_wrong_time:
                        raise RuntimeError(
                            "a predictive audit exposed a confounded wrong-time state"
                        )
                    factual_controls = primary_batch.controls
                    peer_state = NativePosteriorState(
                        _distributed_ring_exchange_tensor(
                            prepared.previous_state.rows,
                            dist=dist,
                            torch_module=torch,
                        )
                    )
                    peer_state_valid = _distributed_ring_exchange_tensor(
                        prepared.previous_state_valid,
                        dist=dist,
                        torch_module=torch,
                    )
                    peer_controls = ExecutedControlBatch(
                        values=_distributed_ring_exchange_tensor(
                            factual_controls.values,
                            dist=dist,
                            torch_module=torch,
                        ),
                        field_valid=_distributed_ring_exchange_tensor(
                            factual_controls.field_valid,
                            dist=dist,
                            torch_module=torch,
                        ),
                        token_valid=_distributed_ring_exchange_tensor(
                            factual_controls.token_valid,
                            dist=dist,
                            torch_module=torch,
                        ),
                        delta_time=_distributed_ring_exchange_tensor(
                            factual_controls.delta_time,
                            dist=dist,
                            torch_module=torch,
                        ),
                        reset=_distributed_ring_exchange_tensor(
                            factual_controls.reset,
                            dist=dist,
                            torch_module=torch,
                        ),
                        acknowledged=_distributed_ring_exchange_tensor(
                            factual_controls.acknowledged,
                            dist=dist,
                            torch_module=torch,
                        ),
                    )
                    correction_branch = result.correction_branches[0]
                    correction_row_binding_valid = assignment_binding_valid_at_phase(
                        result.objective.assignment,
                        result.objective.targets,
                        source_phase=correction_branch.identity_source_phase,
                    )
                    counterfactual_predictions = run_native_correction_counterfactual_forwards(
                        policy,
                        model_inputs=primary_batch.model_inputs,
                        controls=factual_controls,
                        previous_state=prepared.previous_state,
                        previous_state_valid=prepared.previous_state_valid,
                        request=correction_branch.request,
                        modalities=primary_batch.modalities,
                        wrong_batch_state=peer_state,
                        wrong_batch_state_valid=peer_state_valid,
                        # Every rank executes the same optional FSDP forward when
                        # any rank has a clean predecessor. Ranks without one use
                        # their typed invalid placeholder and remove that arm from
                        # the persisted evidence below.
                        wrong_time_state=(
                            prepared.wrong_time_state if execute_wrong_time else None
                        ),
                        wrong_time_state_valid=(
                            prepared.wrong_time_state_valid if execute_wrong_time else None
                        ),
                        wrong_controls=peer_controls,
                    )
                    if execute_wrong_time and not has_wrong_time:
                        counterfactual_predictions = NativeCorrectionCounterfactualPredictions(
                            request=counterfactual_predictions.request,
                            factual=counterfactual_predictions.factual,
                            interventions=tuple(
                                item
                                for item in counterfactual_predictions.interventions
                                if item[0] != WRONG_TIME_SOURCE
                            ),
                        )
                    correction_target = native_current_correction_target(
                        branch=correction_branch,
                        cache=current_grid_cache,
                        physical_sidecar=physical_sidecar,
                        track_identity_keys=result.objective.track_identity_keys_by_batch,
                        minimum_visible_fraction=(
                            predictive_cache.contract.minimum_visible_fraction
                        ),
                        device=device,
                    )
                    counterfactual_diagnostics = predictive_correction_counterfactual_diagnostics(
                        counterfactual_predictions,
                        target=correction_target,
                        assignment=result.objective.assignment,
                        row_binding_valid=correction_row_binding_valid,
                        loss_power=args.predictive_loss_power,
                    )
                    predictive_counterfactual_report = counterfactual_diagnostics.as_dict()
                    predictive_counterfactual_weight_boundary = "pre_update_post_backward"
                    del (
                        correction_branch,
                        correction_target,
                        counterfactual_diagnostics,
                        counterfactual_predictions,
                        factual_controls,
                        peer_controls,
                        peer_state,
                        peer_state_valid,
                    )
                if isinstance(result, NativeRepresentationObjectiveStepResult):
                    if not representation_stage:
                        raise RuntimeError(
                            "action stage returned a representation-only objective result"
                        )
                    primary_context = result.primary
                    reported_official_action_loss: float | None = None
                    reported_official_moe_regularizer: float | None = None
                    reported_official_policy_loss: float | None = None
                elif isinstance(result, NativeFullObjectiveStepResult):
                    if representation_stage:
                        raise RuntimeError(
                            "representation stage returned an action objective result"
                        )
                    primary_context = result.primary.context
                    reported_official_action_loss = float(
                        result.primary.official_action_loss.detach().float().item()
                    )
                    reported_official_moe_regularizer = float(
                        result.primary.official_moe_regularizer.detach().float().item()
                    )
                    reported_official_policy_loss = float(
                        result.primary.official_total_loss.detach().float().item()
                    )
                else:
                    raise TypeError("native objective runner returned another result type")
                posterior = primary_context.posterior_state
                if posterior is None:
                    raise RuntimeError("native full objective produced no committable posterior")
                if posterior_committed:
                    attempt.stage(
                        prepared,
                        posterior,
                        row_bindings_by_batch=result.objective.row_bindings_by_batch,
                    )
                else:
                    attempt.discard(prepared)
            except BaseException:
                attempt.abort()
                optimizer.zero_grad(set_to_none=True)
                raise

            gradient_metrics: dict[str, float | int | bool] = {}

            def optimizer_attempt(
                *,
                step: int = global_step,
                metrics: dict[str, float | int | bool] = gradient_metrics,
                behavior_norm: float = behavior_posterior_gradient_norm,
                behavior_elements: int = behavior_posterior_gradient_elements,
            ) -> int | None:
                successful_step, measured = _optimizer_attempt(
                    policy=policy,
                    optimizer=optimizer,
                    global_step=step,
                    max_grad_norm=args.max_grad_norm,
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )
                metrics.update(measured)
                metrics["behavior_posterior_gradient_norm"] = behavior_norm
                metrics["behavior_posterior_gradient_elements"] = behavior_elements
                return successful_step

            if posterior_committed:
                published = attempt.finish(optimizer_attempt)
            else:
                published = attempt.finish_stateless(optimizer_attempt)
            dist.barrier()
            if not published:
                raise RuntimeError("native full optimizer update overflowed or was skipped")
            posterior_bank_sha256_after = _posterior_bank_digest(bank)
            if posterior_committed:
                if posterior_bank_sha256_after == posterior_bank_sha256_before:
                    raise RuntimeError("causal optimizer update did not publish posterior state")
            elif posterior_bank_sha256_after != posterior_bank_sha256_before:
                raise RuntimeError("stateless reset optimizer update changed the posterior bank")
            if float(gradient_metrics.get("native_graph_norm", 0.0)) <= 0:
                raise RuntimeError("native full objective produced no native-graph gradient")
            action_output_norm = float(gradient_metrics.get("action_output_norm", 0.0))
            if representation_stage and action_output_norm != 0:
                raise RuntimeError("native representation objective reached frozen action output")
            if not representation_stage and action_output_norm <= 0:
                raise RuntimeError("native full objective produced no action-output gradient")
            step_seconds = time.perf_counter() - started
            global_step += 1
            final_source_digest = primary_batch.source_digest
            objective = result.objective.objective
            task_row_diagnostics = list(build_task_row_diagnostics(result.objective))
            prior_row_bindings = [
                [list(pair) for pair in bindings] for bindings in prepared.previous_row_bindings
            ]
            row_bindings = [
                [list(pair) for pair in bindings]
                for bindings in result.objective.row_bindings_by_batch
            ]
            row_binding_birth_count = sum(
                len(current) - len(prior)
                for prior, current in zip(
                    prepared.previous_row_bindings,
                    result.objective.row_bindings_by_batch,
                    strict=True,
                )
            )
            visual_artifacts: list[dict[str, Any]] = []
            relation = None
            visual_started = time.perf_counter()
            if args.visual_audit_every > 0 and global_step % args.visual_audit_every == 0:
                relation = primary_context.relation_output
                if relation is None:
                    raise RuntimeError("native full visual audit lost the relation output")
                visual_artifacts = render_native_relation_visuals(
                    output_root=args.run_dir,
                    global_step=global_step,
                    rank=rank,
                    host_items=planned.training.host_items,
                    model_inputs=primary_batch.model_inputs,
                    objective=result.objective,
                    structural_sensor_valid=relation.structural_valid,
                    sample_keys=primary_batch.routing.sample_keys,
                    merge_size=merge_size,
                )
            visual_seconds = time.perf_counter() - visual_started
            step_reports.append(
                {
                    "global_step": global_step,
                    "estimator_component": estimator_component,
                    "posterior_committed": posterior_committed,
                    "posterior_bank_sha256_before": posterior_bank_sha256_before,
                    "posterior_bank_sha256_after": posterior_bank_sha256_after,
                    "sample_keys": list(primary_batch.routing.sample_keys),
                    "lane_ids": list(primary_batch.routing.lane_ids),
                    "frame_indices": list(primary_batch.routing.frame_indices),
                    "state_ages": list(prepared.next_state_ages),
                    **(
                        {
                            "fixed_observation_pair_sha256": (
                                planned.fixed_observation_pair_sha256
                            ),
                            "fixed_observation_fingerprint": fixed_observation_fingerprint,
                        }
                        if representation_stage
                        else {}
                    ),
                    "temporal_plan_sha256": temporal.digest,
                    "local_bptt_steps": local_count,
                    "overshoot_horizon": overshoot_count,
                    "source_masked_branch": temporal.source_masked_branch,
                    "source_mask_digest": None if source_mask is None else source_mask.digest,
                    "source_mask_query_count": (
                        0 if source_mask is None else source_mask.query_count
                    ),
                    "source_prediction_mode": (
                        None if not temporal.source_masked_branch else args.source_prediction_mode
                    ),
                    "omitted_static_digest": (
                        None if omitted_static_view is None else omitted_static_view.digest
                    ),
                    "objective_total": float(objective.total.detach().float().item()),
                    "official_action_loss": reported_official_action_loss,
                    "official_moe_regularizer": reported_official_moe_regularizer,
                    "official_policy_loss": reported_official_policy_loss,
                    "normalized_terms": {
                        name: float(value.detach().float().item())
                        for name, value in objective.normalized_terms.items()
                    },
                    "valid_counts": objective.valid_counts,
                    "task_row_diagnostics": task_row_diagnostics,
                    "prior_row_bindings": prior_row_bindings,
                    "row_bindings": row_bindings,
                    "row_binding_birth_count": row_binding_birth_count,
                    "visual_artifacts": visual_artifacts,
                    "visual_audit_seconds": visual_seconds,
                    "gradient_metrics": gradient_metrics,
                    "family_gradient_diagnostics": family_gradient_diagnostics,
                    "relation_surface_gradient_diagnostics": (
                        relation_surface_gradient_diagnostics
                    ),
                    "predictive_host_gradient_diagnostics": (predictive_host_gradient_diagnostics),
                    "predictive_counterfactual_diagnostics": (predictive_counterfactual_report),
                    "predictive_counterfactual_weight_boundary": (
                        predictive_counterfactual_weight_boundary
                    ),
                    "step_time_s": step_seconds,
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
            completed = step_reports[-1]
            del (
                attempt,
                continuation_batches,
                family_gradient_diagnostics,
                full_batches,
                objective,
                omitted_static_view,
                optimizer_attempt,
                overshoot_factory,
                planned,
                posterior,
                prepared,
                primary_batch,
                primary_context,
                prior_row_bindings,
                relation,
                relation_surface_gradient_diagnostics,
                result,
                row_bindings,
                row_binding_birth_count,
                run_step_objective,
                source_mask,
                task_row_diagnostics,
            )
            del build_overshoot, controls
            _trim_cuda_allocator_after_gradient_audit(
                gradient_audit=gradient_audit,
                torch_module=torch,
            )
            _emit_step_progress(
                "step_completed",
                rank=rank,
                global_step=global_step,
                details={
                    "objective_total": completed["objective_total"],
                    "official_action_loss": completed["official_action_loss"],
                    "official_policy_loss": completed["official_policy_loss"],
                    "post_release_cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "post_release_cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "step_time_s": completed["step_time_s"],
                    "temporal_plan_sha256": completed["temporal_plan_sha256"],
                },
            )

        if not final_source_digest:
            raise RuntimeError("native full invocation completed no optimizer step")
        if global_step in args.representation_evaluation_steps:
            evaluate_representation_checkpoint(global_step)
        maximum_peak_reserved_bytes = int(args.maximum_peak_reserved_gib * 1024**3)
        local_peak = max(item["peak_cuda_reserved_bytes"] for item in step_reports)
        local_memory_ok = torch.tensor(
            int(local_peak <= maximum_peak_reserved_bytes),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(local_memory_ok, op=dist.ReduceOp.MIN)
        if not bool(local_memory_ok.item()):
            raise RuntimeError("native full invocation exceeded the CUDA reservation budget")

        optimizer_summary = _validate_optimizer_state(
            optimizer,
            torch,
            expected_step=global_step,
        )
        representation_frozen_action_state_sha256 = None
        representation_frozen_action_state_local_sha256 = None
        if representation_stage:
            if (
                representation_parameter_scope is None
                or representation_released_action_state_sha256 is None
                or representation_released_action_state_local_sha256 is None
            ):
                raise RuntimeError("representation action-state contract vanished")
            representation_frozen_action_state_manifest = (
                native_representation_frozen_action_state_manifest(
                    policy,
                    expected=representation_parameter_scope,
                )
            )
            representation_frozen_action_state_local_sha256 = (
                native_representation_action_state_manifest_sha256(
                    representation_frozen_action_state_manifest
                )
            )
            if (
                representation_frozen_action_state_local_sha256
                != representation_released_action_state_local_sha256
            ):
                if representation_released_action_state_manifest is None:
                    raise RuntimeError("representation released action-state manifest vanished")
                changed = native_representation_action_state_changes(
                    representation_released_action_state_manifest,
                    representation_frozen_action_state_manifest,
                )
                raise RuntimeError(
                    "native representation optimizer changed frozen action state: "
                    + ", ".join(changed)
                )
            representation_frozen_action_state_sha256 = _distributed_action_state_sha256(
                representation_frozen_action_state_local_sha256,
                rank=rank,
                world_size=FULL_WORLD_SIZE,
                dist=dist,
            )
            if (
                representation_frozen_action_state_sha256
                != representation_released_action_state_sha256
            ):
                raise RuntimeError("distributed frozen action state changed during training")
        _require_unchanged_behavior_graph(
            args,
            expected_sha256=behavior_graph_sha256,
        )
        publish_checkpoint = args.checkpoint_publication == "always"
        extra_state = None
        saved_boundary = None
        if publish_checkpoint:
            rank_rng_state = _capture_rank_rng(torch, np, device=device)
            lane_snapshot = bank.serialize()
            saved_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=lane_snapshot,
                rank_rng_state=rank_rng_state,
                torch_module=torch,
            )
            extra_state = {
                "boundary_sha256": saved_boundary,
                "behavior_conditioning_sha256": _behavior_conditioning_digest(
                    args,
                    behavior_graph_sha256=behavior_graph_sha256,
                ),
                "execution_contract_sha256": execution_sha256,
                "global_step": global_step,
                "implementation_sha256": implementation_sha256,
                "lane_snapshot": lane_snapshot,
                "model_family_sha256": model_family_sha256,
                "next_optimizer_step": global_step,
                **optimizer_summary,
                "plan_sha256": stream_plan.plan_sha256,
                "rank": rank,
                "rank_rng_state": rank_rng_state,
                "schema": (
                    REPRESENTATION_EXTRA_STATE_SCHEMA
                    if representation_stage
                    else FULL_EXTRA_STATE_SCHEMA
                ),
                "source_digest": final_source_digest,
                "temporal_estimator_sha256": temporal_config.digest,
                "world_size": FULL_WORLD_SIZE,
            }
            if representation_stage:
                if representation_split is None or representation_parameter_scope_sha256 is None:
                    raise RuntimeError("representation checkpoint lost its frozen contracts")
                extra_state.update(
                    {
                        "training_stage": NATIVE_REPRESENTATION_STAGE,
                        "representation_split_sha256": (representation_split.artifact_sha256),
                        "representation_parameter_scope_sha256": (
                            representation_parameter_scope_sha256
                        ),
                        "representation_frozen_action_state_sha256": (
                            representation_frozen_action_state_local_sha256
                        ),
                    }
                )
        gathered_reports: list[Any] = [None for _ in range(FULL_WORLD_SIZE)]
        dist.all_gather_object(
            gathered_reports,
            {
                "rank": rank,
                "steps": step_reports,
                "saved_boundary_sha256": saved_boundary,
                "loaded_boundary_sha256": loaded_boundary,
            },
        )

        checkpoint_root = args.run_dir / "checkpoints"
        output_checkpoint = checkpoint_root / f"global_step_{global_step}"
        staging_checkpoint = checkpoint_root / f".global_step_{global_step}.incomplete"
        report_filename = (
            "native_representation_report.json"
            if representation_stage
            else "native_full_report.json"
        )
        report_path = args.run_dir / (
            f"native_representation_step_{global_step}.json"
            if representation_stage
            else f"native_full_step_{global_step}.json"
        )
        publication_paths = (
            (output_checkpoint, staging_checkpoint, report_path)
            if publish_checkpoint
            else (report_path,)
        )
        conflict = torch.tensor(
            int(any(path.exists() or path.is_symlink() for path in publication_paths)),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(conflict, op=dist.ReduceOp.MAX)
        if bool(conflict.item()):
            raise FileExistsError(f"native full publication path exists: {report_path}")
        if publish_checkpoint:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            precheckpoint_error: list[str | None] = [None]
            if rank == 0:
                try:
                    require_checkpoint_write_capacity(checkpoint_root)
                except BaseException as error:
                    precheckpoint_error[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(precheckpoint_error, src=0)
            if precheckpoint_error[0] is not None:
                raise RuntimeError(
                    "native full pre-checkpoint capacity validation failed: "
                    f"{precheckpoint_error[0]}"
                )
            if extra_state is None:
                raise RuntimeError("native full checkpoint publication lost its extra state")
            checkpointer.save(
                str(staging_checkpoint),
                {"model": policy, "optimizer": optimizer, "extra_state": extra_state},
                global_steps=None,
            )
            dist.barrier()
        report = None
        if rank == 0:
            report = {
                "schema": (
                    REPRESENTATION_REPORT_SCHEMA if representation_stage else FULL_REPORT_SCHEMA
                ),
                "phase": args.phase,
                "status": "PASS",
                "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                "patch_sha256": patch_sha256,
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "execution_contract_sha256": execution_sha256,
                "implementation_sha256": implementation_sha256,
                "model_family_sha256": model_family_sha256,
                "plan_sha256": stream_plan.plan_sha256,
                "temporal_estimator_sha256": temporal_config.digest,
                "dataset_contract": dataset_contract_report,
                "physical_sidecar_manifest_sha256": physical_sidecar.manifest_sha256,
                "predictive_cache_manifest_sha256": predictive_cache.manifest_sha256,
                "predictive_teacher_causality_audit_sha256": (
                    args.predictive_teacher_causality_audit_sha256
                ),
                "predictive_target_audit_sha256": args.predictive_target_audit_sha256,
                "predictive_temporal_audit_sha256": args.predictive_temporal_audit_sha256,
                "current_grid_cache_manifest_sha256": current_grid_cache.manifest_sha256,
                "input_global_step": args.load_global_step,
                "saved_global_step": global_step,
                "checkpoint_dir": str(output_checkpoint.resolve()),
                "full_shard": publish_checkpoint,
                "fsdp2_placement": fsdp2_placement,
                "cuda_allocator": args.cuda_allocator,
                "gradient_checkpointing": True,
                "action_loss_enabled": not representation_stage,
                "predictive_correction_loss_enabled": (not args.behavior_conditioned_prediction),
                "behavior_future_loss_enabled": args.behavior_conditioned_prediction,
                "behavior_conditioning": _behavior_conditioning_contract(
                    args,
                    behavior_graph_sha256=behavior_graph_sha256,
                ),
                "structural_set_loss_enabled": True,
                "current_source_mask_enabled": args.source_prediction_mode == "current_grid",
                "omitted_static_binding_enabled": args.source_prediction_mode == "omitted_static",
                "source_prediction_mode": args.source_prediction_mode,
                "objective_contract": {
                    "predictive_family_weight": args.predictive_weight,
                    "structural_family_weight": args.structural_weight,
                    "predictive_term_weight": args.predictive_term_weight,
                    "current_grid_term_weight": args.current_grid_term_weight,
                    "omitted_static_term_weight": args.omitted_static_term_weight,
                    "support_weight": args.support_weight,
                    "existence_weight": args.existence_weight,
                    "task_weight": args.task_weight,
                    "task_relation_estimator": args.task_relation_estimator,
                    "dense_task_weight": args.dense_task_weight,
                    "ownership_weight": args.ownership_weight,
                    "ownership_estimator": _ownership_estimator_from_args(args),
                    "family_reduction": "active_weighted_mean",
                },
                "evidence_profile": args.evidence_profile,
                "gradient_audit_steps": list(args.gradient_audit_steps),
                "gradient_audit_temporal_scope": GRADIENT_AUDIT_TEMPORAL_SCOPE,
                "complete_adr74_objective": False,
                "training_authorization": (
                    None
                    if training_authorization is None
                    else {
                        **training_authorization,
                        "manifest_path": str(args.authorization_manifest.resolve()),
                        "manifest_sha256": args.authorization_manifest_sha256,
                    }
                ),
                "long_training_authorized": (
                    training_authorization is not None and training_authorization["stage"] == "long"
                ),
                "parameter_storage": parameter_storage,
                "alignment_teacher_prune": alignment_teacher_prune,
                "action_fsdp2_topology": action_fsdp2_topology,
                "vlm_fsdp2_topology": vlm_fsdp2_topology,
                "maximum_peak_reserved_bytes": maximum_peak_reserved_bytes,
                "parameter_manifest": {
                    "parameter_count": parameter_manifest.parameter_count,
                    "trainable_numel": parameter_manifest.trainable_numel,
                    "schema_sha256": parameter_manifest.schema_sha256,
                },
                "rank_reports": gathered_reports,
            }
            if representation_stage:
                if (
                    representation_split is None
                    or representation_parameter_scope is None
                    or representation_frozen_action_state_sha256 is None
                ):
                    raise RuntimeError("representation report lost its frozen contracts")
                report.update(
                    {
                        "checkpoint_publication": args.checkpoint_publication,
                        "training_stage": NATIVE_REPRESENTATION_STAGE,
                        "representation_split_sha256": (representation_split.artifact_sha256),
                        "representation_split_file_sha256": (args.representation_split_sha256),
                        "representation_parameter_scope": (
                            representation_parameter_scope.as_dict()
                        ),
                        "representation_parameter_scope_sha256": (
                            representation_parameter_scope_sha256
                        ),
                        "representation_frozen_action_state_sha256": (
                            representation_frozen_action_state_sha256
                        ),
                        "visual_audit_every": args.visual_audit_every,
                    }
                )
                report["objective_contract"]["action_family_weight"] = 0.0
        publish_error: list[str | None] = [None]
        representation_validation_kwargs: dict[str, Any] | None = None
        if rank == 0:
            try:
                if report is None:
                    raise RuntimeError("rank zero did not construct the native full report")
                validation_kwargs = {
                    "expected_saved_global_step": global_step,
                    "expected_digests": {
                        "execution_contract_sha256": execution_sha256,
                        "implementation_sha256": implementation_sha256,
                        "model_family_sha256": model_family_sha256,
                    },
                    "require_initial_probe": (
                        args.phase == "fresh"
                        and publish_checkpoint
                        and not args.behavior_conditioned_prediction
                    ),
                    "require_source_evidence": (
                        training_authorization is not None
                        and training_authorization["stage"] == "long"
                    ),
                    "expected_fsdp2_placement": fsdp2_placement,
                    "expected_cuda_allocator": args.cuda_allocator,
                }
                if representation_stage:
                    if (
                        representation_split is None
                        or representation_parameter_scope_sha256 is None
                        or representation_frozen_action_state_sha256 is None
                    ):
                        raise RuntimeError("representation publication lost its frozen contracts")
                    representation_validation_kwargs = {
                        name: measured
                        for name, measured in validation_kwargs.items()
                        if name != "require_source_evidence"
                    }
                    representation_validation_kwargs["expected_behavior_conditioning_sha256"] = (
                        _behavior_conditioning_digest(
                            args,
                            behavior_graph_sha256=behavior_graph_sha256,
                        )
                    )
                    representation_validation_kwargs["expected_digests"].update(
                        {
                            "representation_split_sha256": (representation_split.artifact_sha256),
                            "representation_split_file_sha256": (args.representation_split_sha256),
                            "representation_parameter_scope_sha256": (
                                representation_parameter_scope_sha256
                            ),
                            "representation_frozen_action_state_sha256": (
                                representation_frozen_action_state_sha256
                            ),
                        }
                    )
                    validate_representation_objective_report(
                        report,
                        **representation_validation_kwargs,
                        require_checkpoint_copy=False,
                        expected_checkpoint_publication=args.checkpoint_publication,
                    )
                else:
                    validate_full_objective_report(
                        report,
                        **validation_kwargs,
                        require_checkpoint_copy=False,
                    )
                payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
                if publish_checkpoint:
                    _write_text_durable(staging_checkpoint / report_filename, payload)
                    _fsync_tree(staging_checkpoint)
                    os.replace(staging_checkpoint, output_checkpoint)
                    descriptor = os.open(checkpoint_root, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    try:
                        if representation_stage:
                            if representation_validation_kwargs is None:
                                raise RuntimeError(
                                    "representation publication lost validation arguments"
                                )
                            validate_representation_objective_report(
                                report,
                                **representation_validation_kwargs,
                                expected_checkpoint_publication=args.checkpoint_publication,
                            )
                        else:
                            validate_full_objective_report(report, **validation_kwargs)
                    except BaseException:
                        os.replace(output_checkpoint, staging_checkpoint)
                        rollback_descriptor = os.open(checkpoint_root, os.O_RDONLY)
                        try:
                            os.fsync(rollback_descriptor)
                        finally:
                            os.close(rollback_descriptor)
                        raise
                    try:
                        _write_text_durable(report_path, payload)
                    except BaseException:
                        os.replace(output_checkpoint, staging_checkpoint)
                        rollback_descriptor = os.open(checkpoint_root, os.O_RDONLY)
                        try:
                            os.fsync(rollback_descriptor)
                        finally:
                            os.close(rollback_descriptor)
                        raise
                else:
                    _write_text_durable(report_path, payload)
            except BaseException as error:
                publish_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(publish_error, src=0)
        if publish_error[0] is not None:
            raise RuntimeError(f"native full checkpoint publication failed: {publish_error[0]}")
        dist.barrier()
        if rank == 0:
            if report is None:
                raise RuntimeError("rank zero lost the native full report before publication")
            payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
            print(payload, end="")
    finally:
        if run_lease is not None:
            run_lease.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
