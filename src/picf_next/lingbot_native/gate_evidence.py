"""Fail-closed evidence contracts for LingBot-native empirical gates.

These validators are deployment infrastructure, not model components.  They
bind every learned claim to one checkpoint and require the registered paired,
hierarchical comparison design before a report can authorize more training.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_SOURCE_COMMIT,
)
from picf_next.data.dataset_manifest import (
    DATASET_RUNTIME_BINDING_FIELDS,
    validate_dataset_runtime_binding_report,
)
from picf_next.lingbot_native.capacity import (
    MINIMUM_LINGBOT_FREE_STORAGE_BYTES,
    MINIMUM_LINGBOT_HOST_MEMORY_BYTES,
)
from picf_next.lingbot_native.empirical_statistics import (
    EMPIRICAL_COMPARISON_RULES,
    EMPIRICAL_REPORT_FIELDS,
    EMPIRICAL_REPORT_SCHEMAS,
    build_empirical_gate_report_from_observations,
)
from picf_next.lingbot_native.empirical_statistics import (
    EMPIRICAL_REQUIRED_ARMS as EMPIRICAL_REQUIRED_ARMS,
)
from picf_next.lingbot_native.empirical_statistics import (
    EMPIRICAL_REQUIRED_CHECKS as EMPIRICAL_REQUIRED_CHECKS,
)
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    validate_fsdp2_placement,
    validate_fsdp2_storage_report,
)

COMPLETE_ADR74_MISSING_CAPABILITIES: tuple[str, ...] = ()

_SUBJECT_FIELDS = {
    "input_full_report_sha256",
    "saved_global_step",
    "execution_contract_sha256",
    "implementation_sha256",
    "model_family_sha256",
}
_G0_FIELDS = {
    "schema",
    "phase",
    "source_commit",
    "patch_sha256",
    "patched_source_sha256",
    "checkpoint_revision",
    "execution_contract_sha256",
    "implementation_sha256",
    "model_family_sha256",
    "plan_sha256",
    "dataset_contract",
    "input_global_step",
    "saved_global_step",
    "checkpoint_dir",
    "status",
    "full_shard",
    "fsdp2_placement",
    "cuda_allocator",
    "gradient_checkpointing",
    "auxiliary_target_losses_enabled",
    "parameter_storage",
    "maximum_peak_reserved_bytes",
    "parameter_manifest",
    "rank_reports",
}
_G0_DATASET_FIELDS = {
    "status",
    "manifest_sha256",
    "normalization_sha256",
    "validation",
}
_G0_DATASET_VALIDATION_FIELDS = set(DATASET_RUNTIME_BINDING_FIELDS)
_G0_PARAMETER_MANIFEST_FIELDS = {"parameter_count", "trainable_numel", "schema_sha256"}
_G0_RANK_FIELDS = {
    "rank",
    "sample_keys",
    "lane_ids",
    "episode_keys",
    "frame_indices",
    "official_action_loss",
    "official_moe_regularizer",
    "official_policy_loss",
    "gradient_metrics",
    "optimizer_state",
    "step_time_s",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "saved_boundary_sha256",
    "loaded_boundary_sha256",
    "resume_boundary_verified",
    "resume_runtime_rng_verified",
}
_G0_GRADIENT_FIELDS = {
    "all_finite",
    "native_graph_norm",
    "native_graph_elements",
    "action_output_norm",
    "action_output_elements",
    "preclip_global_norm",
}
_G0_TASK_QUERY_GRADIENT_FIELDS = {
    "task_query_norm",
    "task_query_elements",
}
_G0_OPTIMIZER_FIELDS = {"optimizer_state_entries", "optimizer_local_moment_elements"}
_BOUNDARY_FIELDS = {
    "lane_snapshot_sha256",
    "model_local_state_sha256",
    "optimizer_local_state_sha256",
    "rank_rng_state_sha256",
}
_PREFLIGHT_FIELDS = {
    "schema",
    "status",
    "static_contract_pass",
    "local_tests_executed",
    "local_deployment_pass",
    "g0_action_only_static_ready",
    "future_structural_runner_static_ready",
    "complete_adr74_static_ready",
    "complete_adr74_missing_capabilities",
    "released_weight_omitted_static_binding_validated",
    "full_objective_static_ready",
    "full_objective_missing_files",
    "cloud_runtime_ready",
    "cloud_hardware_ready",
    "cloud_model_assets_ready",
    "cloud_data_ready",
    "cloud_assets_ready",
    "cloud_g0_ready",
    "authorized_gates",
    "long_training_authorized",
    "scientific_acceptance",
    "root",
    "python",
    "source_checkout",
    "static_checks",
    "commands",
    "import_origin",
    "import_origin_valid",
    "python_version",
    "python_major_minor",
    "package_versions",
    "expected_cloud_runtime",
    "host_import_probe",
    "gpu_inventory",
    "hardware_capacity",
    "checkpoint",
    "processor",
    "g0_data",
}
_PREFLIGHT_STATIC_FIELDS = {
    "required_files",
    "implementation_files",
    "implementation_sha256",
    "full_implementation_files",
    "full_implementation_sha256",
    "patch_replay",
    "lingbot_requirements_sha256",
    "lingbot_depth_requirements_sha256",
    "released_training_config_sha256",
    "released_optimizer_contract",
    "calvin_environment",
}
_PREFLIGHT_CALVIN_ENVIRONMENT_FIELDS = {
    "status",
    "calvin_env_root",
    "calvin_commit",
    "calvin_env_commit",
    "calvin_requirements_sha256",
    "calvin_setup_sha256",
}
_EXPECTED_PREFLIGHT_OPTIMIZER = {
    "adamw_betas": [0.9, 0.95],
    "adamw_eps": 1e-8,
    "algorithm": "lingbot_distributed_muon_with_adamw_fallback",
    "bias_centering": False,
    "bias_update_interval": 1,
    "bias_update_speed": 0.0,
    "builder": "lingbotvla.optim.build_muon_optimizer",
    "enable_fp32": True,
    "enable_mixed_precision": True,
    "learning_rate": 1e-4,
    "moe_hook": ("lingbotvla.models.vla.lingbot_vla.moe_load_balance.build_moe_load_balance_hook"),
    "muon_adjust_lr_fn": "match_rms_adamw",
    "muon_exclude_name_patterns": [],
    "muon_momentum": 0.95,
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "routed_scaling_factor": 4.0,
    "router_activation": "sigmoid",
    "router_z_loss_coeff": 1e-4,
    "scheduler": "constant",
    "scheduler_implementation": "constant_identity_no_state",
    "scheduler_start_lr": 0.0,
    "scheduler_warmup_ratio": 0.0,
    "sequence_wise_loss_coeff": 1e-3,
    "sequence_wise_mode": "per_sequence",
    "token_moe_layers": list(range(36)),
    "token_num_experts": 32,
    "token_top_k": 4,
    "use_moe": True,
    "use_moe_expert_lr": True,
    "use_shared_expert_gate": False,
    "weight_decay": 0.0,
}
_PREFLIGHT_OPTIMIZER_FIELDS = set(_EXPECTED_PREFLIGHT_OPTIMIZER)
_PATCH_REPLAY_FIELDS = {
    "apply_checked",
    "commit",
    "patch",
    "patch_sha256",
    "patched_sources",
    "patched_source_sha256",
    "verification_source",
}
_COMMAND_FIELDS = {"command", "returncode", "stdout_tail", "stderr_tail", "passed"}
_GPU_FIELDS = {"index", "name", "memory_mib"}
_PREFLIGHT_HARDWARE_CAPACITY_FIELDS = {
    "host_memory_bytes",
    "minimum_host_memory_bytes",
    "persistent_storage_root",
    "free_storage_bytes",
    "minimum_free_storage_bytes",
}
_CHECKPOINT_ASSET_FIELDS = {
    "checkpoint_id",
    "checkpoint_revision",
    "checkpoint_dir",
    "required_files",
    "checkpoint_assets",
}
_PROCESSOR_ASSET_FIELDS = {
    "processor_id",
    "processor_revision",
    "processor_dir",
    "required_processor_files",
    "processor_assets",
}
_PREFLIGHT_G0_DATA_FIELDS = {
    "ready",
    "status",
    "dataset_split",
    "dataset_manifest_sha256",
    "norm_stats_sha256",
    "validation",
}

_SMOKE_FIELDS = {
    "schema",
    "implementation_sha256",
    "source_commit",
    "source_patch_sha256",
    "patched_source_sha256",
    "source_diff_sha256",
    "checkpoint_revision",
    "checkpoint_assets",
    "processor_revision",
    "processor_assets",
    "config",
    "config_sha256",
    "image",
    "image_sha256",
    "task",
    "alternate_task",
    "target_only_fields_present",
    "device",
    "device_name",
    "dtype",
    "num_steps",
    "native_graph",
    "input_shapes",
    "moe_inference_backend",
    "official_action_sha256",
    "official_repeat_action_sha256",
    "targetless_action_sha256",
    "installed_neutral_action_sha256",
    "alignment_teacher_prune",
    "official_repeat_action_bitwise_equal",
    "official_repeat_action_max_abs_error",
    "targetless_action_bitwise_equal",
    "targetless_action_max_abs_error",
    "neutral_action_bitwise_equal",
    "neutral_action_max_abs_error",
    "official_routes",
    "official_repeat_routes",
    "targetless_routes",
    "installed_neutral_routes",
    "official_repeat_route_bitwise_equal",
    "targetless_route_bitwise_equal",
    "neutral_route_bitwise_equal",
    "first_action_sha256",
    "second_action_sha256",
    "first_prior_sha256",
    "first_posterior_sha256",
    "second_prior_sha256",
    "second_posterior_sha256",
    "native_actions_finite",
    "native_relations_finite",
    "session_snapshot_bytes",
    "session_snapshot_sha256",
    "session_snapshot_roundtrip_exact",
    "prompt_invariant_physical_posterior_bitwise_equal",
    "prompt_invariant_physical_posterior_max_abs_error",
    "alternate_action_sha256",
    "timings",
    "cuda_memory_bytes",
    "pid",
    "status",
    "failures",
}
_ALIGNMENT_PRUNE_FIELDS = {
    "schema",
    "removed",
    "removed_numel",
    "removed_storage_bytes",
    "retained_query_components",
}
_ALIGNMENT_PRUNE_ITEM_FIELDS = {"name", "parameter_count", "numel", "storage_bytes"}
_ALIGNMENT_TEACHER_HEAD_NAMES = {
    "current_video_align_head",
    "depth_align_head",
    "future_depth_align_head",
    "future_video_align_head",
    "future_video_cls_head",
}
_ROUTE_TRACE_FIELDS = {"sha256", "calls", "tokens", "layers"}
_MOE_INFERENCE_BACKEND_FIELDS = {
    "schema",
    "selected",
    "fused_fallback_available",
    "robby_available_before_selection",
    "robby_disabled",
}
_SMOKE_NATIVE_GRAPH_FIELDS = {
    "capacity",
    "host_width",
    "executed_action_dim",
    "num_layers",
    "maximum_control_tokens",
}
_ASSET_RECORD_FIELDS = {"path", "bytes", "sha256"}
_CUDA_MEMORY_FIELDS = {"allocated", "reserved", "peak_allocated", "peak_reserved"}


def _exact_dict(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields differ from the frozen schema")
    return cast(dict[str, Any], value)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boundary(value: object, *, name: str) -> dict[str, Any]:
    boundary = _exact_dict(value, name=name, fields=_BOUNDARY_FIELDS)
    for field, digest in boundary.items():
        _sha256(digest, name=f"{name} {field}")
    return boundary


def _real_absolute_path(value: object, *, name: str, directory: bool) -> Path:
    path = Path(value) if isinstance(value, str) else None
    expected_type = (
        path.is_dir() if path is not None and directory else path.is_file() if path else False
    )
    if path is None or not path.is_absolute() or path.is_symlink() or not expected_type:
        kind = "directory" if directory else "file"
        raise ValueError(f"{name} must be one real absolute {kind}")
    return path


def _validate_command(value: object, *, name: str) -> dict[str, Any]:
    report = _exact_dict(value, name=name, fields=_COMMAND_FIELDS)
    command = report["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
        or isinstance(report["returncode"], bool)
        or not isinstance(report["returncode"], int)
        or not isinstance(report["stdout_tail"], str)
        or not isinstance(report["stderr_tail"], str)
        or report["passed"] is not (report["returncode"] == 0)
    ):
        raise ValueError(f"{name} command result was not recomputed")
    return report


def validate_g0_report(
    value: object,
    *,
    schema: str,
    phase: str,
    source_commit: str,
    checkpoint_revision: str,
    world_size: int,
    require_checkpoint_copy: bool = True,
    expected_fsdp2_placement: str = FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    expected_cuda_allocator: str = "native",
) -> dict[str, Any]:
    """Recompute the complete two-rank released-weight update evidence contract."""

    if phase not in {"fresh", "resume"}:
        raise ValueError("G0 phase must be fresh or resume")
    _positive_integer(world_size, name="G0 world size")
    if not isinstance(require_checkpoint_copy, bool):
        raise TypeError("G0 checkpoint-copy requirement must be boolean")
    placement = validate_fsdp2_placement(expected_fsdp2_placement)
    if expected_cuda_allocator not in {
        "native",
        "expandable-segments",
        "cuda-malloc-async",
    }:
        raise ValueError("G0 CUDA allocator expectation is unsupported")
    report = _exact_dict(value, name="G0 report", fields=_G0_FIELDS)
    if (
        report["schema"] != schema
        or report["status"] != "PASS"
        or report["phase"] != phase
        or report["source_commit"] != source_commit
        or report["checkpoint_revision"] != checkpoint_revision
    ):
        raise ValueError("G0 report identity differs from the frozen released-weight run")
    expected_input = 0 if phase == "fresh" else 1
    if (
        report["input_global_step"] != expected_input
        or report["saved_global_step"] != expected_input + 1
    ):
        raise ValueError("G0 report does not bind the registered optimizer interval")
    if any(
        report[field] is not expected
        for field, expected in {
            "full_shard": True,
            "gradient_checkpointing": True,
            "auxiliary_target_losses_enabled": False,
        }.items()
    ):
        raise ValueError("G0 report differs from the isolated FSDP2 execution contract")
    if report["fsdp2_placement"] != placement:
        raise ValueError("G0 report differs from the expected FSDP2 placement contract")
    if report["cuda_allocator"] != expected_cuda_allocator:
        raise ValueError("G0 report differs from the expected CUDA allocator contract")
    for field in (
        "patch_sha256",
        "execution_contract_sha256",
        "implementation_sha256",
        "model_family_sha256",
        "plan_sha256",
    ):
        _sha256(report[field], name=f"G0 {field}")

    patched = report["patched_source_sha256"]
    if not isinstance(patched, dict) or not patched:
        raise ValueError("G0 report has no patched-source manifest")
    for relative, digest in patched.items():
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            path is None
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("G0 patched-source manifest contains an invalid path")
        _sha256(digest, name=f"G0 patched source {relative}")

    dataset = _exact_dict(
        report["dataset_contract"],
        name="G0 dataset contract",
        fields=_G0_DATASET_FIELDS,
    )
    if dataset["status"] != "PASS":
        raise ValueError("G0 dataset contract did not pass")
    _sha256(dataset["manifest_sha256"], name="G0 dataset manifest")
    _sha256(dataset["normalization_sha256"], name="G0 normalization")
    validation = _exact_dict(
        dataset["validation"],
        name="G0 dataset validation",
        fields=_G0_DATASET_VALIDATION_FIELDS,
    )
    try:
        validate_dataset_runtime_binding_report(validation)
    except ContractError as error:
        raise ValueError("G0 dataset verified-read contract changed") from error

    validate_fsdp2_storage_report(
        report["parameter_storage"],
        expected_placement=placement,
    )
    manifest = _exact_dict(
        report["parameter_manifest"],
        name="G0 parameter manifest",
        fields=_G0_PARAMETER_MANIFEST_FIELDS,
    )
    _positive_integer(manifest["parameter_count"], name="G0 parameter count")
    _positive_integer(manifest["trainable_numel"], name="G0 trainable elements")
    _sha256(manifest["schema_sha256"], name="G0 parameter schema")
    maximum_reserved = _positive_integer(
        report["maximum_peak_reserved_bytes"],
        name="G0 maximum reserved bytes",
    )

    rank_reports = report["rank_reports"]
    if not isinstance(rank_reports, list) or len(rank_reports) != world_size:
        raise ValueError("G0 report lacks one rank report per process")
    observed_ranks: set[int] = set()
    observed_samples: set[str] = set()
    for raw_rank in rank_reports:
        rank_report = _exact_dict(raw_rank, name="G0 rank report", fields=_G0_RANK_FIELDS)
        rank = rank_report["rank"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= world_size
            or rank in observed_ranks
        ):
            raise ValueError("G0 rank report has an invalid or duplicate rank")
        string_sequences = [
            rank_report["sample_keys"],
            rank_report["episode_keys"],
        ]
        integer_sequences = [
            rank_report["lane_ids"],
            rank_report["frame_indices"],
        ]
        sequences = [*string_sequences, *integer_sequences]
        if (
            any(not isinstance(sequence, list) or not sequence for sequence in sequences)
            or len({len(sequence) for sequence in sequences}) != 1
            or any(
                not isinstance(item, str) or not item
                for sequence in string_sequences
                for item in sequence
            )
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for sequence in integer_sequences
                for item in sequence
            )
        ):
            raise ValueError("G0 rank routing provenance is malformed")
        sample_keys = set(cast(list[str], rank_report["sample_keys"]))
        if len(sample_keys) != len(rank_report["sample_keys"]) or sample_keys & observed_samples:
            raise ValueError("G0 distributed ranks reused one sample key")
        observed_samples.update(sample_keys)

        action = _finite(rank_report["official_action_loss"], name="G0 action loss")
        moe = _finite(rank_report["official_moe_regularizer"], name="G0 MoE regularizer")
        policy = _finite(rank_report["official_policy_loss"], name="G0 policy loss")
        if min(action, moe, policy) < 0 or not math.isclose(
            policy,
            action + moe,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError("G0 official policy loss is not action plus MoE regularization")
        raw_gradients = rank_report["gradient_metrics"]
        if not isinstance(raw_gradients, dict):
            raise ValueError("G0 gradient metrics must be a dictionary")
        gradient_fields = set(raw_gradients)
        valid_gradient_fields = {
            frozenset(_G0_GRADIENT_FIELDS),
            frozenset(_G0_GRADIENT_FIELDS | _G0_TASK_QUERY_GRADIENT_FIELDS),
        }
        if frozenset(gradient_fields) not in valid_gradient_fields:
            raise ValueError("G0 gradient metrics fields differ from the frozen schema")
        gradients = raw_gradients
        if gradients["all_finite"] is not True:
            raise ValueError("G0 gradients are non-finite")
        for name in ("native_graph_norm", "action_output_norm", "preclip_global_norm"):
            if _finite(gradients[name], name=f"G0 {name}") <= 0:
                raise ValueError(f"G0 {name} must be positive")
        for name in ("native_graph_elements", "action_output_elements"):
            _positive_integer(gradients[name], name=f"G0 {name}")
        if gradient_fields >= _G0_TASK_QUERY_GRADIENT_FIELDS:
            if _finite(gradients["task_query_norm"], name="G0 task_query_norm") <= 0:
                raise ValueError("G0 task_query_norm must be positive")
            _positive_integer(
                gradients["task_query_elements"],
                name="G0 task_query_elements",
            )
        optimizer = _exact_dict(
            rank_report["optimizer_state"],
            name="G0 optimizer state",
            fields=_G0_OPTIMIZER_FIELDS,
        )
        for name, measured in optimizer.items():
            _positive_integer(measured, name=f"G0 {name}")
        if _finite(rank_report["step_time_s"], name="G0 step time") <= 0:
            raise ValueError("G0 step time must be positive")
        allocated = _positive_integer(
            rank_report["peak_cuda_allocated_bytes"],
            name="G0 peak allocated bytes",
        )
        reserved = _positive_integer(
            rank_report["peak_cuda_reserved_bytes"],
            name="G0 peak reserved bytes",
        )
        if allocated > reserved or reserved > maximum_reserved:
            raise ValueError("G0 CUDA memory report violates its registered budget")
        _boundary(rank_report["saved_boundary_sha256"], name="G0 saved boundary")
        loaded = rank_report["loaded_boundary_sha256"]
        expected_resume = phase == "resume"
        if expected_resume:
            _boundary(loaded, name="G0 loaded boundary")
        elif loaded is not None:
            raise ValueError("fresh G0 report unexpectedly contains a loaded boundary")
        if (
            rank_report["resume_boundary_verified"] is not expected_resume
            or rank_report["resume_runtime_rng_verified"] is not expected_resume
        ):
            raise ValueError("G0 resume boundary or runtime RNG was not verified exactly")
        observed_ranks.add(rank)
    if observed_ranks != set(range(world_size)):
        raise ValueError("G0 rank coverage is incomplete")

    checkpoint_value = report["checkpoint_dir"]
    checkpoint = Path(checkpoint_value) if isinstance(checkpoint_value, str) else None
    if checkpoint is None or not checkpoint.is_absolute() or checkpoint.is_symlink():
        raise ValueError("G0 checkpoint directory must be one real absolute path")
    if require_checkpoint_copy:
        checkpoint_report = checkpoint / "native_g0_report.json"
        if (
            not checkpoint.is_dir()
            or checkpoint_report.is_symlink()
            or not checkpoint_report.is_file()
        ):
            raise ValueError("G0 checkpoint does not contain its report copy")
        try:
            checkpoint_payload = json.loads(checkpoint_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("G0 checkpoint report copy is unreadable") from error
        if checkpoint_payload != report:
            raise ValueError("G0 checkpoint report copy differs from its gate evidence")
    return report


def validate_preflight_report(
    value: object,
    *,
    schema: str,
    source_commit: str,
    checkpoint_id: str,
    checkpoint_revision: str,
    checkpoint_assets: list[dict[str, object]],
    processor_id: str,
    processor_revision: str,
    processor_assets: list[dict[str, object]],
    world_size: int,
) -> dict[str, Any]:
    """Recompute the exact cloud-ready preflight rather than trust its PASS field."""

    report = _exact_dict(value, name="cloud preflight", fields=_PREFLIGHT_FIELDS)
    if report["schema"] != schema or report["status"] != "PASS":
        raise ValueError("cloud preflight schema or status changed")
    required_true = (
        "static_contract_pass",
        "local_tests_executed",
        "local_deployment_pass",
        "g0_action_only_static_ready",
        "future_structural_runner_static_ready",
        "full_objective_static_ready",
        "cloud_runtime_ready",
        "cloud_hardware_ready",
        "cloud_model_assets_ready",
        "cloud_data_ready",
        "cloud_assets_ready",
        "cloud_g0_ready",
        "import_origin_valid",
    )
    if any(report[field] is not True for field in required_true):
        raise ValueError("cloud preflight did not pass every required readiness surface")
    if (
        report["complete_adr74_static_ready"] is not True
        or report["complete_adr74_missing_capabilities"]
        != list(COMPLETE_ADR74_MISSING_CAPABILITIES)
        or report["released_weight_omitted_static_binding_validated"] is not False
        or report["full_objective_missing_files"] != []
        or report["authorized_gates"]
        != [
            "G0_full_weight_neutral_parity",
            "G0_two_rank_full_update_and_cold_resume",
        ]
        or report["long_training_authorized"] is not False
        or report["scientific_acceptance"] != "PENDING_G1_G8"
    ):
        raise ValueError("cloud preflight made an unsupported completeness claim")

    root = _real_absolute_path(report["root"], name="preflight root", directory=True)
    _real_absolute_path(report["python"], name="preflight Python", directory=False)
    _real_absolute_path(
        report["source_checkout"],
        name="preflight source checkout",
        directory=True,
    )
    import_origin = _real_absolute_path(
        report["import_origin"],
        name="preflight import origin",
        directory=False,
    )
    if not import_origin.is_relative_to(root / "src"):
        raise ValueError("preflight imported picf_next outside the audited source tree")

    static = _exact_dict(
        report["static_checks"],
        name="preflight static checks",
        fields=_PREFLIGHT_STATIC_FIELDS,
    )
    _positive_integer(static["required_files"], name="preflight required files")
    for name in ("implementation_files", "full_implementation_files"):
        paths = static[name]
        if (
            not isinstance(paths, list)
            or not paths
            or len(set(paths)) != len(paths)
            or any(
                not isinstance(path, str)
                or PurePosixPath(path).is_absolute()
                or any(part in {"", ".", ".."} for part in PurePosixPath(path).parts)
                for path in paths
            )
        ):
            raise ValueError(f"preflight {name} is malformed")
    for name in (
        "implementation_sha256",
        "full_implementation_sha256",
        "lingbot_requirements_sha256",
        "lingbot_depth_requirements_sha256",
        "released_training_config_sha256",
    ):
        _sha256(static[name], name=f"preflight {name}")
    calvin = _exact_dict(
        static["calvin_environment"],
        name="preflight CALVIN environment",
        fields=_PREFLIGHT_CALVIN_ENVIRONMENT_FIELDS,
    )
    if (
        calvin["status"] != "PASS"
        or calvin["calvin_commit"] != CALVIN_SOURCE_COMMIT
        or calvin["calvin_env_commit"] != CALVIN_ENV_SOURCE_COMMIT
    ):
        raise ValueError("preflight CALVIN source identity differs")
    _real_absolute_path(
        calvin["calvin_env_root"],
        name="preflight CALVIN environment root",
        directory=True,
    )
    _sha256(
        calvin["calvin_requirements_sha256"],
        name="preflight CALVIN requirements",
    )
    _sha256(calvin["calvin_setup_sha256"], name="preflight CALVIN setup")
    optimizer = _exact_dict(
        static["released_optimizer_contract"],
        name="preflight released optimizer",
        fields=_PREFLIGHT_OPTIMIZER_FIELDS,
    )
    if optimizer != _EXPECTED_PREFLIGHT_OPTIMIZER:
        raise ValueError("preflight released optimizer contract differs")
    patch = _exact_dict(
        static["patch_replay"],
        name="preflight patch replay",
        fields=_PATCH_REPLAY_FIELDS,
    )
    if (
        patch["apply_checked"] is not True
        or patch["commit"] != source_commit
        or patch["patch"] != "references/patches/lingbot_vla2_picf_native.patch"
        or patch["verification_source"] != "immutable_commit_archive"
    ):
        raise ValueError("preflight did not replay the immutable LingBot patch")
    _sha256(patch["patch_sha256"], name="preflight patch")
    patched_sources = patch["patched_sources"]
    patched_digests = patch["patched_source_sha256"]
    if (
        not isinstance(patched_sources, list)
        or not patched_sources
        or len(set(patched_sources)) != len(patched_sources)
        or not isinstance(patched_digests, dict)
        or set(patched_digests) != set(patched_sources)
    ):
        raise ValueError("preflight patched-source coverage is inconsistent")
    for relative, digest in patched_digests.items():
        _sha256(digest, name=f"preflight patched source {relative}")

    commands = report["commands"]
    if not isinstance(commands, list) or len(commands) < 6:
        raise ValueError("preflight did not execute the complete local command suite")
    validated_commands = [_validate_command(item, name="preflight command") for item in commands]
    if any(item["passed"] is not True for item in validated_commands):
        raise ValueError("preflight command suite contains a failure")
    host_import = _validate_command(report["host_import_probe"], name="host import probe")
    if host_import["passed"] is not True:
        raise ValueError("preflight host runtime imports failed")

    if (
        report["python_major_minor"] != [3, 12]
        or not isinstance(report["python_version"], str)
        or not report["python_version"].startswith("3.12")
        or not isinstance(report["package_versions"], dict)
        or not report["package_versions"]
        or report["package_versions"] != report["expected_cloud_runtime"]
        or any(
            not isinstance(name, str) or not name or not isinstance(version, str) or not version
            for name, version in report["package_versions"].items()
        )
    ):
        raise ValueError("preflight Python or exact package lock differs")
    gpus = report["gpu_inventory"]
    if not isinstance(gpus, list) or len(gpus) != world_size:
        raise ValueError("preflight GPU inventory differs from the two-rank contract")
    for expected_index, raw_gpu in enumerate(gpus):
        gpu = _exact_dict(raw_gpu, name="preflight GPU", fields=_GPU_FIELDS)
        if (
            gpu["index"] != expected_index
            or not isinstance(gpu["name"], str)
            or "A100" not in gpu["name"]
            or isinstance(gpu["memory_mib"], bool)
            or not isinstance(gpu["memory_mib"], int)
            or gpu["memory_mib"] < 40_000
        ):
            raise ValueError("preflight GPU is not one indexed A100 40GB device")

    hardware = _exact_dict(
        report["hardware_capacity"],
        name="preflight hardware capacity",
        fields=_PREFLIGHT_HARDWARE_CAPACITY_FIELDS,
    )
    host_memory = _positive_integer(
        hardware["host_memory_bytes"],
        name="preflight host memory",
    )
    minimum_host_memory = _positive_integer(
        hardware["minimum_host_memory_bytes"],
        name="preflight minimum host memory",
    )
    free_storage = _positive_integer(
        hardware["free_storage_bytes"],
        name="preflight free storage",
    )
    minimum_free_storage = _positive_integer(
        hardware["minimum_free_storage_bytes"],
        name="preflight minimum free storage",
    )
    _real_absolute_path(
        hardware["persistent_storage_root"],
        name="preflight persistent storage root",
        directory=True,
    )
    if (
        minimum_host_memory != MINIMUM_LINGBOT_HOST_MEMORY_BYTES
        or minimum_free_storage != MINIMUM_LINGBOT_FREE_STORAGE_BYTES
        or host_memory < minimum_host_memory
        or free_storage < minimum_free_storage
    ):
        raise ValueError("preflight host or persistent-storage capacity is insufficient")

    checkpoint = _exact_dict(
        report["checkpoint"],
        name="preflight checkpoint",
        fields=_CHECKPOINT_ASSET_FIELDS,
    )
    processor = _exact_dict(
        report["processor"],
        name="preflight processor",
        fields=_PROCESSOR_ASSET_FIELDS,
    )
    for asset, directory_field, count_field, identity, revision, expected_assets in (
        (
            checkpoint,
            "checkpoint_dir",
            "required_files",
            checkpoint_id,
            checkpoint_revision,
            checkpoint_assets,
        ),
        (
            processor,
            "processor_dir",
            "required_processor_files",
            processor_id,
            processor_revision,
            processor_assets,
        ),
    ):
        _real_absolute_path(asset[directory_field], name=directory_field, directory=True)
        manifest_field = "checkpoint_assets" if asset is checkpoint else "processor_assets"
        id_field = "checkpoint_id" if asset is checkpoint else "processor_id"
        revision_field = "checkpoint_revision" if asset is checkpoint else "processor_revision"
        _validate_asset_manifest(asset[manifest_field], name=f"preflight {manifest_field}")
        if (
            asset[id_field] != identity
            or asset[revision_field] != revision
            or asset[manifest_field] != expected_assets
            or _positive_integer(asset[count_field], name=count_field) != len(expected_assets)
        ):
            raise ValueError("preflight model asset identity is incomplete")
    g0_data = _exact_dict(
        report["g0_data"],
        name="preflight G0 data",
        fields=_PREFLIGHT_G0_DATA_FIELDS,
    )
    if g0_data["ready"] is not True or g0_data["status"] != "PASS":
        raise ValueError("preflight G0 data did not pass")
    _real_absolute_path(
        g0_data["dataset_split"],
        name="preflight dataset split",
        directory=True,
    )
    _sha256(g0_data["dataset_manifest_sha256"], name="preflight dataset manifest")
    _sha256(g0_data["norm_stats_sha256"], name="preflight normalization")
    data_validation = _exact_dict(
        g0_data["validation"],
        name="preflight dataset validation",
        fields=_G0_DATASET_VALIDATION_FIELDS,
    )
    try:
        validate_dataset_runtime_binding_report(data_validation)
    except ContractError as error:
        raise ValueError("preflight dataset verified-read contract changed") from error
    return report


def _validate_asset_manifest(value: object, *, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} is empty")
    observed: set[str] = set()
    for raw in value:
        item = _exact_dict(raw, name=name, fields=_ASSET_RECORD_FIELDS)
        relative = item["path"]
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            path is None
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or relative in observed
        ):
            raise ValueError(f"{name} contains an invalid path")
        _positive_integer(item["bytes"], name=f"{name} bytes")
        _sha256(item["sha256"], name=f"{name} digest")
        observed.add(relative)


def _validate_route_trace(value: object, *, name: str) -> dict[str, Any]:
    trace = _exact_dict(value, name=name, fields=_ROUTE_TRACE_FIELDS)
    _sha256(trace["sha256"], name=f"{name} digest")
    for field in ("calls", "tokens", "layers"):
        _positive_integer(trace[field], name=f"{name} {field}")
    return trace


def validate_full_weight_smoke_report(
    value: object,
    *,
    schema: str,
    implementation_sha256: str,
    source_commit: str,
    checkpoint_revision: str,
    checkpoint_assets: list[dict[str, object]],
    processor_revision: str,
    processor_assets: list[dict[str, object]],
) -> dict[str, Any]:
    """Recompute the released-weight neutral parity and state-isolation smoke."""

    report = _exact_dict(value, name="released-weight smoke", fields=_SMOKE_FIELDS)
    if (
        report["schema"] != schema
        or report["implementation_sha256"] != implementation_sha256
        or report["status"] != "PASS"
        or report["source_commit"] != source_commit
        or report["checkpoint_revision"] != checkpoint_revision
        or report["processor_revision"] != processor_revision
        or report["failures"] != []
        or report["target_only_fields_present"] != []
    ):
        raise ValueError("released-weight smoke identity, status, or leak check failed")
    for field in (
        "implementation_sha256",
        "source_patch_sha256",
        "source_diff_sha256",
        "config_sha256",
        "image_sha256",
        "official_action_sha256",
        "official_repeat_action_sha256",
        "targetless_action_sha256",
        "installed_neutral_action_sha256",
        "first_action_sha256",
        "second_action_sha256",
        "first_prior_sha256",
        "first_posterior_sha256",
        "second_prior_sha256",
        "second_posterior_sha256",
        "session_snapshot_sha256",
        "alternate_action_sha256",
    ):
        _sha256(report[field], name=f"released-weight smoke {field}")
    prune = _exact_dict(
        report["alignment_teacher_prune"],
        name="released-weight alignment teacher prune",
        fields=_ALIGNMENT_PRUNE_FIELDS,
    )
    if prune["schema"] != "picf-next.targetless-alignment-teacher-prune.v1":
        raise ValueError("released-weight alignment teacher prune schema changed")
    removed = prune["removed"]
    retained = prune["retained_query_components"]
    if (
        not isinstance(removed, list)
        or not removed
        or not isinstance(retained, list)
        or not retained
        or any(not isinstance(name, str) or not name for name in retained)
        or len(set(retained)) != len(retained)
    ):
        raise ValueError("released-weight alignment teacher prune is incomplete")
    removed_numel = 0
    removed_storage = 0
    removed_names: set[str] = set()
    for raw in removed:
        item = _exact_dict(
            raw,
            name="released-weight removed teacher head",
            fields=_ALIGNMENT_PRUNE_ITEM_FIELDS,
        )
        name = item["name"]
        if (
            not isinstance(name, str)
            or name not in _ALIGNMENT_TEACHER_HEAD_NAMES
            or name in removed_names
        ):
            raise ValueError("released-weight removed teacher-head identity is invalid")
        _positive_integer(item["parameter_count"], name=f"{name} parameter count")
        removed_numel += _positive_integer(item["numel"], name=f"{name} elements")
        removed_storage += _positive_integer(item["storage_bytes"], name=f"{name} bytes")
        removed_names.add(name)
    reported_numel = _positive_integer(
        prune["removed_numel"], name="released-weight removed teacher-head elements"
    )
    reported_storage = _positive_integer(
        prune["removed_storage_bytes"], name="released-weight removed teacher-head bytes"
    )
    if (
        removed_numel != reported_numel
        or removed_storage != reported_storage
        or "depth_align_head" not in removed_names
        or "depth_align_embs" not in retained
    ):
        raise ValueError("released-weight alignment teacher prune totals or topology differ")
    patched = report["patched_source_sha256"]
    if not isinstance(patched, dict) or not patched:
        raise ValueError("released-weight smoke has no patched-source manifest")
    for path, digest in patched.items():
        if not isinstance(path, str) or not path:
            raise ValueError("released-weight smoke has an invalid patched source")
        _sha256(digest, name=f"released-weight patched source {path}")
    _validate_asset_manifest(report["checkpoint_assets"], name="checkpoint asset manifest")
    _validate_asset_manifest(report["processor_assets"], name="processor asset manifest")
    if (
        report["checkpoint_assets"] != checkpoint_assets
        or report["processor_assets"] != processor_assets
    ):
        raise ValueError("released-weight smoke assets differ from the pinned revisions")
    config = _real_absolute_path(report["config"], name="smoke config", directory=False)
    image = _real_absolute_path(report["image"], name="smoke image", directory=False)
    if (
        _file_sha256(config) != report["config_sha256"]
        or _file_sha256(image) != report["image_sha256"]
    ):
        raise ValueError("released-weight smoke input file differs from its digest")
    if (
        not isinstance(report["task"], str)
        or not report["task"].strip()
        or not isinstance(report["alternate_task"], str)
        or not report["alternate_task"].strip()
        or report["task"] == report["alternate_task"]
        or report["device"] != "cuda:0"
        or not isinstance(report["device_name"], str)
        or "A100" not in report["device_name"]
        or report["dtype"] != "torch.bfloat16"
        or _positive_integer(report["num_steps"], name="smoke steps") < 2
    ):
        raise ValueError("released-weight smoke task or execution device is invalid")
    graph = _exact_dict(
        report["native_graph"],
        name="smoke native graph",
        fields=_SMOKE_NATIVE_GRAPH_FIELDS,
    )
    for field, measured in graph.items():
        _positive_integer(measured, name=f"smoke graph {field}")
    shapes = report["input_shapes"]
    if (
        not isinstance(shapes, dict)
        or not shapes
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
                for dimension in shape
            )
            for name, shape in shapes.items()
        )
    ):
        raise ValueError("released-weight smoke input shapes are malformed")
    backend = _exact_dict(
        report["moe_inference_backend"],
        name="smoke MoE inference backend",
        fields=_MOE_INFERENCE_BACKEND_FIELDS,
    )
    if (
        backend["schema"] != "picf-next.lingbot-moe-inference-backend.v1"
        or backend["selected"] != "fused_moe_forward"
        or backend["fused_fallback_available"] is not True
        or backend["robby_available_before_selection"] is not True
        or backend["robby_disabled"] is not True
    ):
        raise ValueError("released-weight smoke did not select the deterministic MoE backend")
    required_true = (
        "official_repeat_action_bitwise_equal",
        "official_repeat_route_bitwise_equal",
        "targetless_action_bitwise_equal",
        "targetless_route_bitwise_equal",
        "neutral_action_bitwise_equal",
        "neutral_route_bitwise_equal",
        "native_actions_finite",
        "native_relations_finite",
        "session_snapshot_roundtrip_exact",
        "prompt_invariant_physical_posterior_bitwise_equal",
    )
    if any(report[field] is not True for field in required_true):
        raise ValueError("released-weight smoke did not pass parity or isolation")
    official_routes = _validate_route_trace(
        report["official_routes"], name="official action route trace"
    )
    official_repeat_routes = _validate_route_trace(
        report["official_repeat_routes"], name="official repeat action route trace"
    )
    targetless_routes = _validate_route_trace(
        report["targetless_routes"], name="targetless action route trace"
    )
    neutral_routes = _validate_route_trace(
        report["installed_neutral_routes"], name="neutral action route trace"
    )
    if (
        report["official_action_sha256"] != report["official_repeat_action_sha256"]
        or report["official_action_sha256"] != report["targetless_action_sha256"]
        or report["official_action_sha256"] != report["installed_neutral_action_sha256"]
        or official_routes != official_repeat_routes
        or official_routes != targetless_routes
        or official_routes != neutral_routes
        or _finite(
            report["official_repeat_action_max_abs_error"],
            name="official repeat action error",
        )
        != 0
        or _finite(report["targetless_action_max_abs_error"], name="targetless action error") != 0
        or _finite(report["neutral_action_max_abs_error"], name="neutral action error") != 0
        or _finite(
            report["prompt_invariant_physical_posterior_max_abs_error"],
            name="prompt posterior error",
        )
        != 0
    ):
        raise ValueError("released-weight smoke parity booleans disagree with measurements")
    _positive_integer(report["session_snapshot_bytes"], name="session snapshot bytes")
    timings = report["timings"]
    if (
        not isinstance(timings, dict)
        or not timings
        or any(
            not isinstance(name, str)
            or not name
            or _finite(measured, name=f"smoke timing {name}") <= 0
            for name, measured in timings.items()
        )
    ):
        raise ValueError("released-weight smoke timings are malformed")
    memory = _exact_dict(
        report["cuda_memory_bytes"],
        name="smoke CUDA memory",
        fields=_CUDA_MEMORY_FIELDS,
    )
    if any(
        isinstance(measured, bool) or not isinstance(measured, int) or measured < 0
        for measured in memory.values()
    ) or not (
        memory["allocated"] <= memory["peak_allocated"] <= memory["peak_reserved"]
        and memory["reserved"] <= memory["peak_reserved"]
    ):
        raise ValueError("released-weight smoke CUDA memory is inconsistent")
    _positive_integer(report["pid"], name="smoke process id")
    return report


def validate_gate_subject(value: object) -> dict[str, Any]:
    subject = _exact_dict(value, name="gate subject", fields=_SUBJECT_FIELDS)
    for field in _SUBJECT_FIELDS - {"saved_global_step"}:
        _sha256(subject[field], name=f"gate subject {field}")
    step = subject["saved_global_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("gate subject saved step must be a positive integer")
    return subject


def validate_empirical_gate_report(
    value: object,
    *,
    gate: str,
    schema: str,
) -> dict[str, Any]:
    """Validate a G2-G6 machine report and recompute every PASS decision."""

    if gate not in EMPIRICAL_COMPARISON_RULES:
        raise ValueError("empirical report gate is unsupported")
    report = _exact_dict(
        value,
        name=f"{gate} empirical report",
        fields=EMPIRICAL_REPORT_FIELDS,
    )
    if schema != EMPIRICAL_REPORT_SCHEMAS[gate] or report["schema"] != schema:
        raise ValueError(f"{gate} empirical report schema changed")
    if report["gate"] != gate:
        raise ValueError(f"{gate} empirical report schema or gate changed")
    validate_gate_subject(report["subject"])
    observations = _exact_dict(
        report["observations"],
        name=f"{gate} observation reference",
        fields={"schema", "path", "sha256", "record_count"},
    )
    observations_path = (
        Path(observations["path"]) if isinstance(observations["path"], str) else None
    )
    if observations_path is None:
        raise ValueError(f"{gate} observation path is invalid")
    expected = build_empirical_gate_report_from_observations(
        observations_path,
        report_schema=schema,
        expected_sha256=observations["sha256"],
    )
    if expected != report:
        raise ValueError(f"{gate} empirical report was not recomputed from raw observations")
    return report


_VISUAL_FIELDS = {
    "schema",
    "status",
    "gate",
    "subject",
    "criteria_sha256",
    "artifact_root",
    "artifact_manifest_sha256",
    "reviewer",
    "reviewed_at_utc",
    "artifacts",
    "coverage",
    "checks",
    "failures",
    "long_training_authorized",
}
_VISUAL_ARTIFACT_FIELDS = {
    "path",
    "sha256",
    "bytes",
    "global_step",
    "rank",
    "sample_key",
    "task",
    "status",
    "observations",
}
_VISUAL_COVERAGE_FIELDS = {"artifact_count", "ranks", "global_steps", "tasks"}
_VISUAL_CHECKS = {
    "all_artifacts_reviewed",
    "all_panels_legible",
    "object_alignment_acceptable",
    "task_anchor_alignment_acceptable",
    "no_catastrophic_off_object_collapse",
    "context_and_no_object_behavior_acceptable",
    "no_label_or_mask_input_leak",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_g2_visual_review(value: object, *, schema: str) -> dict[str, Any]:
    """Validate a complete hash-bound review of current LingBot visual artifacts."""

    report = _exact_dict(value, name="G2 visual review", fields=_VISUAL_FIELDS)
    if report["schema"] != schema or report["gate"] != "G2":
        raise ValueError("G2 visual review schema or gate changed")
    validate_gate_subject(report["subject"])
    _sha256(report["criteria_sha256"], name="G2 visual review criteria")
    artifact_manifest_sha256 = _sha256(
        report["artifact_manifest_sha256"],
        name="G2 visual artifact manifest",
    )
    root_value = report["artifact_root"]
    root = Path(root_value) if isinstance(root_value, str) else None
    if root is None or not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("G2 visual artifact root must be one real absolute directory")
    reviewer = report["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("G2 visual review requires a named reviewer")
    try:
        reviewed_at = datetime.fromisoformat(cast(str, report["reviewed_at_utc"]))
    except (TypeError, ValueError) as error:
        raise ValueError("G2 visual review timestamp is invalid") from error
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() != timezone.utc.utcoffset(reviewed_at):
        raise ValueError("G2 visual review timestamp must be UTC")

    artifacts = report["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) < 12:
        raise ValueError("G2 visual review requires at least 12 artifacts")
    if artifact_manifest_sha256 != _canonical_sha256(artifacts):
        raise ValueError("G2 visual artifact manifest digest was not recomputed exactly")
    observed_paths: set[str] = set()
    observed_ranks: set[int] = set()
    observed_steps: set[int] = set()
    observed_tasks: set[str] = set()
    failures: list[str] = []
    for raw in artifacts:
        artifact = _exact_dict(raw, name="G2 visual artifact", fields=_VISUAL_ARTIFACT_FIELDS)
        relative_text = artifact["path"]
        if not isinstance(relative_text, str) or relative_text in observed_paths:
            raise ValueError("G2 visual artifact path is invalid or duplicated")
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("G2 visual artifact path escapes its root")
        path = root.joinpath(*relative.parts)
        expected_digest = _sha256(artifact["sha256"], name="G2 visual artifact")
        byte_count = artifact["bytes"]
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or path.stat().st_size != byte_count
            or _file_sha256(path) != expected_digest
        ):
            raise ValueError("G2 visual artifact content differs from the reviewed file")
        step = artifact["global_step"]
        rank = artifact["rank"]
        task = artifact["task"]
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in (0, 1)
            or not isinstance(task, str)
            or not task.strip()
            or not isinstance(artifact["sample_key"], str)
            or not artifact["sample_key"]
        ):
            raise ValueError("G2 visual artifact provenance is malformed")
        if artifact["status"] not in {"PASS", "FAIL"}:
            raise ValueError("G2 visual artifact review status is invalid")
        if not isinstance(artifact["observations"], str) or not artifact["observations"].strip():
            raise ValueError("G2 visual artifact requires substantive observations")
        if artifact["status"] == "FAIL":
            failures.append(f"artifact:{relative_text}")
        observed_paths.add(relative_text)
        observed_ranks.add(rank)
        observed_steps.add(step)
        observed_tasks.add(task)

    coverage = _exact_dict(
        report["coverage"],
        name="G2 visual coverage",
        fields=_VISUAL_COVERAGE_FIELDS,
    )
    expected_coverage = {
        "artifact_count": len(artifacts),
        "ranks": sorted(observed_ranks),
        "global_steps": sorted(observed_steps),
        "tasks": sorted(observed_tasks),
    }
    if coverage != expected_coverage:
        raise ValueError("G2 visual coverage was not recomputed exactly")
    if observed_ranks != {0, 1} or len(observed_steps) < 6 or len(observed_tasks) < 4:
        raise ValueError("G2 visual review lacks two-rank, six-step, or four-task coverage")
    checks = _exact_dict(report["checks"], name="G2 visual checks", fields=_VISUAL_CHECKS)
    if any(not isinstance(passed, bool) for passed in checks.values()):
        raise ValueError("G2 visual review checks must be boolean")
    failures.extend(f"check:{name}" for name, passed in checks.items() if not passed)
    failures.sort()
    expected_status = "PASS" if not failures else "FAIL"
    if (
        report["status"] != expected_status
        or report["failures"] != failures
        or report["long_training_authorized"] is not False
    ):
        raise ValueError("G2 visual review decision was not recomputed exactly")
    return report


_G7_FIELDS = {
    "schema",
    "status",
    "gate",
    "subject",
    "criteria_sha256",
    "protocol_sha256",
    "dataset_name",
    "dataset_manifest_sha256",
    "split_manifest_sha256",
    "embodiment_schema_sha256",
    "interface_schema_sha256",
    "environment_lock_sha256",
    "registered_arms",
    "registered_metrics",
    "paired_seed_count",
    "checkpoint_policy",
    "adapter_policy",
    "target_availability_audited",
    "executable_commands",
    "preregistered_before_evaluation",
    "failures",
    "long_training_authorized",
}


def validate_g7_protocol(value: object, *, schema: str) -> dict[str, Any]:
    """Validate that the second-dataset test is executable before the long run."""

    report = _exact_dict(value, name="G7 protocol", fields=_G7_FIELDS)
    if report["schema"] != schema or report["gate"] != "G7_PROTOCOL":
        raise ValueError("G7 protocol schema or gate changed")
    validate_gate_subject(report["subject"])
    for field in (
        "criteria_sha256",
        "protocol_sha256",
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "embodiment_schema_sha256",
        "interface_schema_sha256",
        "environment_lock_sha256",
    ):
        _sha256(report[field], name=f"G7 {field}")
    if not isinstance(report["dataset_name"], str) or not report["dataset_name"].strip():
        raise ValueError("G7 protocol requires a named second dataset")
    metrics = report["registered_metrics"]
    commands = report["executable_commands"]
    if (
        report["registered_arms"] != ["A", "H", "M", "O", "C"]
        or not isinstance(metrics, list)
        or len(metrics) < 3
        or any(not isinstance(item, str) or not item.strip() for item in metrics)
        or len(set(metrics)) != len(metrics)
        or not isinstance(commands, list)
        or not commands
        or any(not isinstance(item, str) or not item.strip() for item in commands)
    ):
        raise ValueError("G7 protocol arms, metrics, or commands are incomplete")
    seeds = report["paired_seed_count"]
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 5:
        raise ValueError("G7 protocol requires at least five paired seeds")
    if (
        report["checkpoint_policy"] != "single_checkpoint_interface"
        or report["adapter_policy"] != "typed_projection_only"
        or report["target_availability_audited"] is not True
        or report["preregistered_before_evaluation"] is not True
        or report["failures"] != []
        or report["status"] != "PASS"
        or report["long_training_authorized"] is not False
    ):
        raise ValueError("G7 protocol is not an executable passed protocol")
    return report
