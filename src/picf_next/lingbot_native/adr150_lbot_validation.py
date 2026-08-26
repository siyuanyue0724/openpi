"""Strict validation for the frozen ADR-150 matched-LBOT report.

The baseline producer layout lives in ``tools/run_lingbot_vla2_official_lbot.py``.
This module deliberately defines the fail-closed v2 contract one field ahead of
that v1 producer: a report is ineligible until it also publishes typed full-modal
action-adoption evidence.  Validation remains independent of the executable so
it cannot initialize CUDA or inherit mutable producer state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import validate_dataset_runtime_binding_report

LBOT_REPORT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-lbot.v2"
LBOT_SNAPSHOT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-lbot-snapshot.v1"
VALIDATION_SCHEMA = "picf-next.adr150-matched-lbot-validation.v2"
FULL_MODAL_ACTION_ADOPTION_SCHEMA = "picf-next.adr150-full-modal-action-adoption.v1"
# Frozen reports created before the control-arm rename retain their original
# schema bytes. They are accepted as inputs only; every new artifact uses LBOT.
LEGACY_LBOT_REPORT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-p0.v2"
ACCEPTED_LBOT_REPORT_SCHEMAS = frozenset(
    {LBOT_REPORT_SCHEMA, LEGACY_LBOT_REPORT_SCHEMA}
)
_ARCHITECTURE = "released_lingbot_vla2_action_policy"
_ATTENTION_IMPLEMENTATIONS = frozenset({"eager", "flex_cached"})
_MODALITIES = ("anytouch", "sonata", "vjepa")
_PRESENCE_SUBSETS = (
    ("none", ()),
    ("A", ("anytouch",)),
    ("S", ("sonata",)),
    ("V", ("vjepa",)),
    ("AS", ("anytouch", "sonata")),
    ("AV", ("anytouch", "vjepa")),
    ("SV", ("sonata", "vjepa")),
    ("ASV", _MODALITIES),
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "status",
        "architecture_identity",
        "picf_graph_installed",
        "physical_sidecar_read",
        "task_scorer_present",
        "action_suffix_executed",
        "posterior_present",
        "physical_event_stream",
        "minimum_future_source_frames",
        "maximum_control_tokens",
        "checkpoint_published",
        "source_commit",
        "source_patch_sha256",
        "patched_source_sha256",
        "implementation_files",
        "implementation_sha256",
        "checkpoint_revision",
        "checkpoint_assets",
        "processor_revision",
        "processor_assets",
        "dataset_contract",
        "plan_sha256",
        "curve_mode",
        "registered_evaluation_steps",
        "representation_split_sha256",
        "evaluation_plan_sha256",
        "evaluation_snapshots",
        "model_family_sha256",
        "lingbot_base_family",
        "lingbot_base_family_sha256",
        "world_size",
        "steps",
        "seed",
        "max_grad_norm",
        "official_output_arity",
        "optimizer_contract",
        "qwen_vision_geometry",
        "fsdp2_placement",
        "cuda_allocator",
        "attention_implementation",
        "gradient_checkpointing",
        "parameter_storage",
        "parameter_manifest",
        "alignment_teacher_prune",
        "maximum_peak_reserved_bytes",
        "rank_reports",
        "full_modal_action_adoption",
    }
)
_LINGBOT_BASE_FAMILY_TOP_LEVEL_FIELDS = frozenset(
    {"lingbot_base_family", "lingbot_base_family_sha256"}
)
_PRE_FUTURE_FILTER_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS.difference(
    {"minimum_future_source_frames"}
)
_PRE_ATTENTION_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS.difference(
    {"attention_implementation"}
)
_PRE_FUTURE_FILTER_AND_ATTENTION_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS.difference(
    {"minimum_future_source_frames", "attention_implementation"}
)
_CURRENT_TOP_LEVEL_FIELD_SETS = (
    _TOP_LEVEL_FIELDS,
    _PRE_FUTURE_FILTER_TOP_LEVEL_FIELDS,
    _PRE_ATTENTION_TOP_LEVEL_FIELDS,
    _PRE_FUTURE_FILTER_AND_ATTENTION_TOP_LEVEL_FIELDS,
)
_ACCEPTED_TOP_LEVEL_FIELD_SETS = tuple(
    fields
    for current in _CURRENT_TOP_LEVEL_FIELD_SETS
    for fields in (current, current.difference(_LINGBOT_BASE_FAMILY_TOP_LEVEL_FIELDS))
)
_LINGBOT_BASE_FAMILY_FIELDS = frozenset(
    {
        "schema",
        "architecture",
        "source_commit",
        "native_patch_sha256",
        "checkpoint_revision",
        "checkpoint_assets",
        "processor_revision",
        "processor_assets",
        "attention_implementation",
        "trainable_scope",
        "optimizer_contract",
        "maximum_control_tokens",
        "artifact_sha256",
    }
)
_OPTIMIZER_FIELDS = frozenset(
    {
        "algorithm",
        "learning_rate",
        "weight_decay",
        "adamw_betas",
        "adamw_eps",
        "muon_momentum",
        "muon_nesterov",
        "muon_ns_steps",
        "muon_adjust_lr_fn",
        "muon_exclude_name_patterns",
        "use_moe",
        "use_moe_expert_lr",
        "token_moe_layers",
        "token_num_experts",
        "token_top_k",
        "bias_update_speed",
        "bias_centering",
        "bias_update_interval",
        "sequence_wise_loss_coeff",
        "sequence_wise_mode",
        "router_z_loss_coeff",
        "router_activation",
        "routed_scaling_factor",
        "use_shared_expert_gate",
        "enable_fp32",
        "enable_mixed_precision",
        "scheduler",
        "scheduler_warmup_ratio",
        "scheduler_start_lr",
    }
)
_STORAGE_FIELDS = frozenset(
    {
        "parameter_tensors",
        "local_elements",
        "master_dtype",
        "placement",
        "cpu_parameter_tensors",
        "cpu_local_elements",
        "cuda_parameter_tensors",
        "cuda_local_elements",
        "selective_cpu_parameter_names",
    }
)
_PARAMETER_MANIFEST_FIELDS = frozenset({"parameter_count", "trainable_numel", "schema_sha256"})
_PRUNE_FIELDS = frozenset(
    {"schema", "removed", "removed_numel", "removed_storage_bytes", "retained_query_components"}
)
_PRUNED_ITEM_FIELDS = frozenset({"name", "parameter_count", "numel", "storage_bytes"})
_SNAPSHOT_FIELDS = frozenset(
    {
        "artifact_sha256",
        "file_sha256",
        "path",
        "checkpoint_global_step",
        "evaluation_input_sha256",
        "partition_summaries",
    }
)
_PARTITION_SUMMARY_FIELDS = frozenset(
    {
        "sample_count",
        "mean_action_loss",
        "mean_total_loss",
        "mean_moe_regularizer",
        "mean_forward_seconds",
    }
)
_RANK_REPORT_FIELDS = frozenset({"rank", "steps"})
_STEP_FIELDS = frozenset(
    {
        "global_step",
        "sample_keys",
        "lane_ids",
        "frame_indices",
        "reset",
        "source_digest",
        "augmentation_seeds",
        "flow_noise_seeds",
        "flow_timestep_seeds",
        "total_loss",
        "action_loss",
        "moe_regularizer",
        "official_output_arity",
        "picf_graph_installed",
        "gradient_metrics",
        "step_time_s",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
    }
)
_GRADIENT_FIELDS = frozenset(
    {
        "all_finite",
        "vlm_host_norm",
        "vlm_host_elements",
        "action_expert_norm",
        "action_expert_elements",
        "action_output_norm",
        "action_output_elements",
        "preclip_global_norm",
    }
)
_ASSET_FIELDS = frozenset({"path", "bytes", "sha256"})
_DATASET_FIELDS = frozenset({"status", "manifest_sha256", "normalization_sha256", "validation"})
_IMPLEMENTATION_IDENTITY_FIELDS = frozenset({"implementation_files", "implementation_sha256"})
_SOURCE_IDENTITY_FIELDS = frozenset(
    {"source_commit", "source_patch_sha256", "patched_source_sha256"}
)
_MODEL_IDENTITY_FIELDS = frozenset(
    {
        "model_family_sha256",
        "checkpoint_revision",
        "checkpoint_assets",
        "qwen_vision_geometry",
        "parameter_storage",
        "parameter_manifest",
        "alignment_teacher_prune",
    }
)
_PROCESSOR_IDENTITY_FIELDS = frozenset({"processor_revision", "processor_assets"})
_ACTION_ADOPTION_FIELDS = frozenset(
    {
        "schema",
        "status",
        "action_loss_only",
        "nonzero_gradient_min_norm",
        "presence_subsets",
        "modality_interventions",
        "active_anytouch_sample_keys",
        "dcp_cold_restore",
    }
)
_PRESENCE_SUBSET_FIELDS = frozenset(
    {
        "name",
        "present_modalities",
        "sample_keys",
        "adapter_action_only_gradients",
        "host_action_only_gradients",
        "qwen_action_expert_action_only_gradient",
        "action_out_proj_action_only_gradient",
    }
)
_ADAPTER_GRADIENT_FIELDS = frozenset({"value", "metadata"})
_GRADIENT_MEASUREMENT_FIELDS = frozenset({"norm", "elements"})
_HOST_GRADIENT_FIELDS = frozenset({"0", "18", "35"})
_MODALITY_INTERVENTION_FIELDS = frozenset(
    {
        "modality",
        "sample_keys",
        "token_permutations",
        "valid_before",
        "valid_after",
        "factual_repeat",
        "value_zero",
        "metadata_zero",
        "value_only_permutation",
        "joint_value_metadata_valid_permutation",
    }
)
_MAXIMUM_METRIC_FIELDS = frozenset({"measured_max_abs_action_drift", "maximum_allowed"})
_MINIMUM_METRIC_FIELDS = frozenset({"measured_max_abs_action_drift", "minimum_required"})
_DCP_COLD_RESTORE_FIELDS = frozenset(
    {
        "checkpoint_global_step",
        "optimizer_step",
        "save_process_sha256",
        "restore_process_sha256",
        "checkpoint_artifact_sha256",
        "saved_boundary",
        "restored_boundary",
        "uninterrupted_next_step",
        "restored_next_step",
    }
)
_BOUNDARY_DIGEST_FIELDS = frozenset(
    {"model_sha256", "optimizer_sha256", "lane_sha256", "rng_sha256"}
)
_CONTINUATION_FIELDS = frozenset(
    {
        "global_step",
        "model_sha256",
        "optimizer_sha256",
        "lane_sha256",
        "rng_sha256",
        "action_output_sha256",
        "action_loss",
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedADR150LBOTReport:
    """Canonical report plus content-addressed validation evidence."""

    canonical_report: dict[str, Any]
    report_sha256: str
    validation_report: dict[str, Any]


def _mapping(value: object, name: str, fields: frozenset[str] | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be one string-keyed mapping")
    if fields is not None and set(value) != fields:
        missing = sorted(fields.difference(value))
        extra = sorted(set(value).difference(fields))
        raise ContractError(f"{name} fields differ from schema; missing={missing}, extra={extra}")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be one JSON array")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be one nonempty string")
    return value


def _sha256(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ContractError(f"{name} must be one lowercase SHA-256")
    return result


def _git_revision(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise ContractError(f"{name} must be one lowercase 40-character Git revision")
    return result


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be one integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} must be at least {minimum}")
    return value


def _number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be one finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ContractError(f"{name} must be at least {minimum}")
    if strictly_positive and result <= 0:
        raise ContractError(f"{name} must be positive")
    return result


def _boolean(value: object, name: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be one boolean")
    if expected is not None and value is not expected:
        raise ContractError(f"{name} differs from the ADR-150 contract")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ContractError("ADR-150 LBOT report is not finite canonical JSON") from error


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_copy(value: object) -> Any:
    return json.loads(_canonical_json(value).decode("ascii"))


def _same_float(actual: object, expected: object, name: str) -> None:
    left = _number(actual, name)
    right = _number(expected, f"expected {name}")
    if left.hex() != right.hex():
        raise ContractError(f"{name} differs from its frozen value")


def _require_canonical_equal(actual: object, expected: object, name: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ContractError(f"{name} differs from its frozen identity")


def _relative_path(value: object, name: str) -> str:
    result = _text(value, name)
    parts = result.split("/")
    if result.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ContractError(f"{name} must be one canonical relative path")
    return result


def _validate_hash_mapping(value: object, name: str) -> dict[str, str]:
    payload = _mapping(value, name)
    if not payload:
        raise ContractError(f"{name} must not be empty")
    result: dict[str, str] = {}
    for path, digest in payload.items():
        canonical_path = _relative_path(path, f"{name} path")
        result[canonical_path] = _sha256(digest, f"{name}[{canonical_path}]")
    return result


def _validate_assets(value: object, name: str) -> list[dict[str, Any]]:
    assets = _list(value, name)
    if not assets:
        raise ContractError(f"{name} must not be empty")
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(assets):
        item = _mapping(raw, f"{name}[{index}]", _ASSET_FIELDS)
        validated.append(
            {
                "path": _relative_path(item["path"], f"{name}[{index}].path"),
                "bytes": _integer(item["bytes"], f"{name}[{index}].bytes", minimum=1),
                "sha256": _sha256(item["sha256"], f"{name}[{index}].sha256"),
            }
        )
    paths = [item["path"] for item in validated]
    if paths != sorted(set(paths)):
        raise ContractError(f"{name} paths must be sorted and unique")
    return validated


def _validate_dataset_contract(value: object, name: str) -> dict[str, Any]:
    payload = _mapping(value, name, _DATASET_FIELDS)
    if payload["status"] != "PASS":
        raise ContractError(f"{name} status must be PASS")
    _sha256(payload["manifest_sha256"], f"{name}.manifest_sha256")
    _sha256(payload["normalization_sha256"], f"{name}.normalization_sha256")
    validate_dataset_runtime_binding_report(payload["validation"])
    return _canonical_copy(payload)


def _validate_optimizer(value: object, *, expected_learning_rate: float) -> dict[str, Any]:
    payload = _mapping(value, "optimizer_contract", _OPTIMIZER_FIELDS)
    if payload["algorithm"] != "lingbot_distributed_muon_with_adamw_fallback":
        raise ContractError("optimizer algorithm differs from released LingBot")
    _same_float(payload["learning_rate"], expected_learning_rate, "optimizer learning rate")
    for name in (
        "weight_decay",
        "bias_update_speed",
        "sequence_wise_loss_coeff",
        "router_z_loss_coeff",
        "scheduler_warmup_ratio",
        "scheduler_start_lr",
    ):
        _number(payload[name], f"optimizer {name}", minimum=0.0)
    for name in ("adamw_eps", "muon_momentum", "routed_scaling_factor"):
        _number(payload[name], f"optimizer {name}", strictly_positive=True)
    betas = _list(payload["adamw_betas"], "optimizer adamw_betas")
    if len(betas) != 2:
        raise ContractError("optimizer adamw_betas must contain exactly two values")
    for index, beta in enumerate(betas):
        measured = _number(beta, f"optimizer adamw_betas[{index}]")
        if not 0 <= measured < 1:
            raise ContractError("optimizer AdamW beta is outside [0, 1)")
    for name in (
        "muon_nesterov",
        "bias_centering",
        "use_shared_expert_gate",
    ):
        _boolean(payload[name], f"optimizer {name}")
    for name in ("use_moe", "use_moe_expert_lr", "enable_fp32", "enable_mixed_precision"):
        _boolean(payload[name], f"optimizer {name}", True)
    for name in ("muon_ns_steps", "token_num_experts", "token_top_k", "bias_update_interval"):
        _integer(payload[name], f"optimizer {name}", minimum=1)
    if payload["token_top_k"] > payload["token_num_experts"]:
        raise ContractError("optimizer token_top_k exceeds token_num_experts")
    layers = _list(payload["token_moe_layers"], "optimizer token_moe_layers")
    if not layers or any(_integer(item, "optimizer MoE layer", minimum=0) < 0 for item in layers):
        raise ContractError("optimizer token_moe_layers must not be empty")
    if layers != sorted(set(layers)):
        raise ContractError("optimizer token_moe_layers must be sorted and unique")
    excludes = _list(payload["muon_exclude_name_patterns"], "optimizer exclusion patterns")
    if any(not isinstance(item, str) for item in excludes):
        raise ContractError("optimizer exclusion patterns must be strings")
    if payload["muon_adjust_lr_fn"] not in (None, "original", "match_rms_adamw"):
        raise ContractError("optimizer Muon LR adjustment is unsupported")
    if payload["sequence_wise_mode"] not in ("per_sequence", "global"):
        raise ContractError("optimizer sequence-wise mode is unsupported")
    if payload["router_activation"] not in ("softmax", "sigmoid"):
        raise ContractError("optimizer router activation is unsupported")
    if (
        payload["scheduler"] != "constant"
        or float(payload["scheduler_warmup_ratio"]) != 0.0
        or float(payload["scheduler_start_lr"]) != 0.0
        or payload["bias_update_interval"] != 1
    ):
        raise ContractError("optimizer does not define an exact resumable identity schedule")
    return _canonical_copy(payload)


def _validate_lingbot_base_family(
    value: object,
    *,
    expected_learning_rate: float,
) -> dict[str, Any]:
    payload = _mapping(value, "lingbot_base_family", _LINGBOT_BASE_FAMILY_FIELDS)
    semantic = dict(payload)
    artifact_sha256 = _sha256(
        semantic.pop("artifact_sha256"),
        "LingBot base-family artifact",
    )
    if artifact_sha256 != _canonical_digest(semantic):
        raise ContractError("LingBot base-family artifact digest is inconsistent")
    if semantic["schema"] != "picf-next.lingbot-base-family.v1":
        raise ContractError("LingBot base-family schema differs")
    if semantic["architecture"] != _ARCHITECTURE:
        raise ContractError("LingBot base-family architecture differs")
    attention_implementation = _text(
        semantic["attention_implementation"],
        "LingBot base-family attention implementation",
    )
    if attention_implementation not in _ATTENTION_IMPLEMENTATIONS:
        raise ContractError("LingBot base-family attention implementation is unsupported")
    trainable_scope = _text(
        semantic["trainable_scope"],
        "LingBot base-family trainable scope",
    )
    if trainable_scope != "full-host":
        raise ContractError("LingBot base-family trainable scope differs")
    result = {
        "schema": semantic["schema"],
        "architecture": semantic["architecture"],
        "source_commit": _git_revision(
            semantic["source_commit"],
            "LingBot base-family source commit",
        ),
        "native_patch_sha256": _sha256(
            semantic["native_patch_sha256"],
            "LingBot base-family native patch",
        ),
        "checkpoint_revision": _git_revision(
            semantic["checkpoint_revision"],
            "LingBot base-family checkpoint revision",
        ),
        "checkpoint_assets": _validate_assets(
            semantic["checkpoint_assets"],
            "LingBot base-family checkpoint assets",
        ),
        "processor_revision": _git_revision(
            semantic["processor_revision"],
            "LingBot base-family processor revision",
        ),
        "processor_assets": _validate_assets(
            semantic["processor_assets"],
            "LingBot base-family processor assets",
        ),
        "attention_implementation": attention_implementation,
        "trainable_scope": trainable_scope,
        "optimizer_contract": _validate_optimizer(
            semantic["optimizer_contract"],
            expected_learning_rate=expected_learning_rate,
        ),
        "maximum_control_tokens": _integer(
            semantic["maximum_control_tokens"],
            "LingBot base-family maximum control tokens",
            minimum=1,
        ),
    }
    return {**result, "artifact_sha256": artifact_sha256}


def _validate_geometry(value: object) -> dict[str, int]:
    payload = _mapping(
        value,
        "qwen_vision_geometry",
        frozenset({"patch_size", "spatial_merge_size"}),
    )
    return {
        "patch_size": _integer(payload["patch_size"], "Qwen patch size", minimum=1),
        "spatial_merge_size": _integer(
            payload["spatial_merge_size"], "Qwen spatial merge size", minimum=1
        ),
    }


def _validate_storage(value: object, *, expected_placement: str) -> dict[str, Any]:
    payload = _mapping(value, "parameter_storage", _STORAGE_FIELDS)
    if payload["master_dtype"] != "float32" or payload["placement"] != expected_placement:
        raise ContractError("FSDP2 storage dtype or placement differs")
    total_tensors = _integer(payload["parameter_tensors"], "parameter tensors", minimum=1)
    total_elements = _integer(payload["local_elements"], "local elements", minimum=1)
    cpu_tensors = _integer(payload["cpu_parameter_tensors"], "CPU parameter tensors", minimum=0)
    cuda_tensors = _integer(payload["cuda_parameter_tensors"], "CUDA parameter tensors", minimum=0)
    cpu_elements = _integer(payload["cpu_local_elements"], "CPU local elements", minimum=0)
    cuda_elements = _integer(payload["cuda_local_elements"], "CUDA local elements", minimum=0)
    names = _list(payload["selective_cpu_parameter_names"], "selective CPU parameter names")
    if any(not isinstance(item, str) or not item for item in names) or names != sorted(set(names)):
        raise ContractError("selective CPU parameter names must be sorted unique strings")
    if (
        total_tensors != cpu_tensors + cuda_tensors
        or total_elements != cpu_elements + cuda_elements
    ):
        raise ContractError("FSDP2 storage totals do not equal CPU plus CUDA partitions")
    if expected_placement == "selective-embedding-offload":
        if not names or cpu_tensors != len(names) or cuda_tensors <= 0:
            raise ContractError("selective FSDP2 placement has an invalid CPU/CUDA partition")
    elif names:
        raise ContractError("non-selective FSDP2 placement declared selective CPU parameters")
    return _canonical_copy(payload)


def _validate_parameter_manifest(value: object) -> dict[str, Any]:
    payload = _mapping(value, "parameter_manifest", _PARAMETER_MANIFEST_FIELDS)
    _integer(payload["parameter_count"], "parameter manifest count", minimum=1)
    _integer(payload["trainable_numel"], "parameter manifest trainable numel", minimum=1)
    _sha256(payload["schema_sha256"], "parameter manifest schema")
    return _canonical_copy(payload)


def _validate_prune(value: object) -> dict[str, Any]:
    payload = _mapping(value, "alignment_teacher_prune", _PRUNE_FIELDS)
    if payload["schema"] != "picf-next.targetless-alignment-teacher-prune.v1":
        raise ContractError("alignment teacher prune schema differs")
    removed = _list(payload["removed"], "alignment teacher removed heads")
    measured_numel = 0
    measured_bytes = 0
    names: list[str] = []
    for index, raw in enumerate(removed):
        item = _mapping(raw, f"removed head {index}", _PRUNED_ITEM_FIELDS)
        names.append(_text(item["name"], f"removed head {index} name"))
        _integer(item["parameter_count"], f"removed head {index} parameter count", minimum=0)
        measured_numel += _integer(item["numel"], f"removed head {index} numel", minimum=0)
        measured_bytes += _integer(item["storage_bytes"], f"removed head {index} bytes", minimum=0)
    if names != sorted(set(names)):
        raise ContractError("removed alignment teacher heads must be sorted and unique")
    removed_numel = _integer(payload["removed_numel"], "removed parameter numel", minimum=0)
    removed_bytes = _integer(
        payload["removed_storage_bytes"], "removed parameter storage bytes", minimum=0
    )
    if removed_numel != measured_numel or removed_bytes != measured_bytes:
        raise ContractError("alignment teacher prune totals differ from removed heads")
    retained = _list(payload["retained_query_components"], "retained query components")
    if any(not isinstance(item, str) or not item for item in retained) or retained != sorted(
        set(retained)
    ):
        raise ContractError("retained query components must be sorted unique strings")
    return _canonical_copy(payload)


def _validate_partition_summaries(value: object, name: str) -> dict[str, Any]:
    payload = _mapping(value, name, frozenset({"validation", "heldout"}))
    for partition in ("validation", "heldout"):
        summary = _mapping(
            payload[partition],
            f"{name}.{partition}",
            _PARTITION_SUMMARY_FIELDS,
        )
        _integer(summary["sample_count"], f"{name}.{partition}.sample_count", minimum=1)
        _number(summary["mean_action_loss"], f"{name}.{partition}.mean_action_loss", minimum=0)
        _number(summary["mean_total_loss"], f"{name}.{partition}.mean_total_loss")
        _number(
            summary["mean_moe_regularizer"],
            f"{name}.{partition}.mean_moe_regularizer",
        )
        _number(
            summary["mean_forward_seconds"],
            f"{name}.{partition}.mean_forward_seconds",
            strictly_positive=True,
        )
    return _canonical_copy(payload)


def _validate_evaluation_snapshots(
    value: object,
    *,
    expected_steps: tuple[int, ...],
) -> list[dict[str, Any]]:
    snapshots = _list(value, "evaluation_snapshots")
    if len(snapshots) != len(expected_steps):
        raise ContractError("evaluation snapshot count differs from registered steps")
    inputs: set[str] = set()
    for index, (raw, expected_step) in enumerate(zip(snapshots, expected_steps, strict=True)):
        item = _mapping(raw, f"evaluation_snapshots[{index}]", _SNAPSHOT_FIELDS)
        if (
            _integer(item["checkpoint_global_step"], "evaluation checkpoint", minimum=0)
            != expected_step
        ):
            raise ContractError("evaluation snapshot step differs from registered order")
        _sha256(item["artifact_sha256"], "evaluation snapshot artifact")
        _sha256(item["file_sha256"], "evaluation snapshot file")
        _text(item["path"], "evaluation snapshot path")
        inputs.add(_sha256(item["evaluation_input_sha256"], "evaluation input"))
        _validate_partition_summaries(
            item["partition_summaries"], f"evaluation_snapshots[{index}].partition_summaries"
        )
    if len(inputs) != 1:
        raise ContractError("evaluation inputs differ across registered checkpoints")
    return _canonical_copy(snapshots)


def _validate_string_list(value: object, name: str, *, nonempty: bool = True) -> list[str]:
    values = _list(value, name)
    if nonempty and not values:
        raise ContractError(f"{name} must not be empty")
    if any(not isinstance(item, str) or not item for item in values):
        raise ContractError(f"{name} must contain nonempty strings")
    return values


def _validate_integer_list(value: object, name: str, *, length: int) -> list[int]:
    values = _list(value, name)
    if len(values) != length:
        raise ContractError(f"{name} length differs from the local batch")
    return [_integer(item, f"{name} item", minimum=0) for item in values]


def _validate_gradient_metrics(value: object, name: str) -> None:
    payload = _mapping(value, name, _GRADIENT_FIELDS)
    _boolean(payload["all_finite"], f"{name}.all_finite", True)
    for component in ("vlm_host", "action_expert", "action_output"):
        _number(payload[f"{component}_norm"], f"{name}.{component}_norm", strictly_positive=True)
        _integer(payload[f"{component}_elements"], f"{name}.{component}_elements", minimum=1)
    _number(payload["preclip_global_norm"], f"{name}.preclip_global_norm", strictly_positive=True)


def _validate_action_gradient(
    value: object,
    name: str,
    *,
    required: bool,
    minimum_norm: float,
) -> dict[str, Any]:
    payload = _mapping(value, name, _GRADIENT_MEASUREMENT_FIELDS)
    elements = _integer(payload["elements"], f"{name}.elements", minimum=0)
    raw_norm = payload["norm"]
    if raw_norm is None:
        if elements != 0:
            raise ContractError(f"{name} null gradient must have zero elements")
        norm: float | None = None
    else:
        norm = _number(raw_norm, f"{name}.norm", minimum=0.0)
        if elements == 0:
            raise ContractError(f"{name} measured gradient must have positive elements")
    if required:
        if norm is None or norm < minimum_norm:
            raise ContractError(f"{name} must carry a nonzero action-only gradient")
    elif norm not in (None, 0.0):
        raise ContractError(f"{name} must be absent or exactly zero")
    return {"norm": norm, "elements": elements}


def _validate_presence_subsets(
    value: object,
    *,
    minimum_norm: float,
) -> tuple[list[dict[str, Any]], set[str]]:
    subsets = _list(value, "full-modal presence subsets")
    if len(subsets) != len(_PRESENCE_SUBSETS):
        raise ContractError("full-modal evidence must contain exactly eight presence subsets")
    validated: list[dict[str, Any]] = []
    anytouch_sample_keys: set[str] = set()
    for index, ((expected_name, expected_present), raw) in enumerate(
        zip(_PRESENCE_SUBSETS, subsets, strict=True)
    ):
        name = f"full-modal presence subsets[{index}]"
        payload = _mapping(raw, name, _PRESENCE_SUBSET_FIELDS)
        if payload["name"] != expected_name:
            raise ContractError("full-modal presence subsets must use the exact canonical order")
        present = _validate_string_list(
            payload["present_modalities"], f"{name}.present_modalities", nonempty=False
        )
        if tuple(present) != expected_present:
            raise ContractError(f"{name} declares the wrong present modalities")
        sample_keys = _validate_string_list(payload["sample_keys"], f"{name}.sample_keys")
        if sample_keys != sorted(set(sample_keys)):
            raise ContractError(f"{name}.sample_keys must be sorted and unique")
        if "anytouch" in expected_present:
            anytouch_sample_keys.update(sample_keys)

        raw_adapters = _mapping(
            payload["adapter_action_only_gradients"],
            f"{name}.adapter_action_only_gradients",
            frozenset(_MODALITIES),
        )
        adapters: dict[str, Any] = {}
        for modality in _MODALITIES:
            adapter = _mapping(
                raw_adapters[modality],
                f"{name}.{modality} adapter",
                _ADAPTER_GRADIENT_FIELDS,
            )
            modality_required = modality in expected_present
            adapters[modality] = {
                branch: _validate_action_gradient(
                    adapter[branch],
                    f"{name}.{modality}.{branch}",
                    required=modality_required,
                    minimum_norm=minimum_norm,
                )
                for branch in ("value", "metadata")
            }

        raw_host = _mapping(
            payload["host_action_only_gradients"],
            f"{name}.host_action_only_gradients",
            _HOST_GRADIENT_FIELDS,
        )
        host = {
            layer: _validate_action_gradient(
                raw_host[layer],
                f"{name}.host layer {layer}",
                required=True,
                minimum_norm=minimum_norm,
            )
            for layer in ("0", "18", "35")
        }
        action_expert = _validate_action_gradient(
            payload["qwen_action_expert_action_only_gradient"],
            f"{name}.qwen action expert",
            required=True,
            minimum_norm=minimum_norm,
        )
        action_output = _validate_action_gradient(
            payload["action_out_proj_action_only_gradient"],
            f"{name}.action_out_proj",
            required=True,
            minimum_norm=minimum_norm,
        )
        validated.append(
            {
                "name": expected_name,
                "present_modalities": present,
                "sample_keys": sample_keys,
                "adapter_action_only_gradients": adapters,
                "host_action_only_gradients": host,
                "qwen_action_expert_action_only_gradient": action_expert,
                "action_out_proj_action_only_gradient": action_output,
            }
        )
    return validated, anytouch_sample_keys


def _validate_maximum_metric(value: object, name: str) -> dict[str, float]:
    payload = _mapping(value, name, _MAXIMUM_METRIC_FIELDS)
    measured = _number(
        payload["measured_max_abs_action_drift"],
        f"{name}.measured_max_abs_action_drift",
        minimum=0.0,
    )
    maximum = _number(payload["maximum_allowed"], f"{name}.maximum_allowed", minimum=0.0)
    if measured > maximum:
        raise ContractError(f"{name} exceeds its registered maximum")
    return {"measured_max_abs_action_drift": measured, "maximum_allowed": maximum}


def _validate_minimum_metric(value: object, name: str) -> dict[str, float]:
    payload = _mapping(value, name, _MINIMUM_METRIC_FIELDS)
    measured = _number(
        payload["measured_max_abs_action_drift"],
        f"{name}.measured_max_abs_action_drift",
        minimum=0.0,
    )
    minimum = _number(
        payload["minimum_required"], f"{name}.minimum_required", strictly_positive=True
    )
    if measured < minimum:
        raise ContractError(f"{name} does not reach its registered minimum")
    return {"measured_max_abs_action_drift": measured, "minimum_required": minimum}


def _validate_modality_interventions(
    value: object,
) -> tuple[list[dict[str, Any]], set[str]]:
    interventions = _list(value, "full-modal modality interventions")
    if len(interventions) != len(_MODALITIES):
        raise ContractError("full-modal evidence must contain exactly three modality interventions")
    validated: list[dict[str, Any]] = []
    anytouch_sample_keys: set[str] = set()
    for index, (expected_modality, raw) in enumerate(zip(_MODALITIES, interventions, strict=True)):
        name = f"full-modal modality interventions[{index}]"
        payload = _mapping(raw, name, _MODALITY_INTERVENTION_FIELDS)
        if payload["modality"] != expected_modality:
            raise ContractError("modality interventions must use the exact canonical order")
        sample_keys = _validate_string_list(payload["sample_keys"], f"{name}.sample_keys")
        if sample_keys != sorted(set(sample_keys)):
            raise ContractError(f"{name}.sample_keys must contain sorted unique keys")
        if expected_modality == "anytouch":
            anytouch_sample_keys.update(sample_keys)

        raw_permutations = _list(payload["token_permutations"], f"{name}.token_permutations")
        valid_before = _list(payload["valid_before"], f"{name}.valid_before")
        valid_after = _list(payload["valid_after"], f"{name}.valid_after")
        if not (len(raw_permutations) == len(valid_before) == len(valid_after) == len(sample_keys)):
            raise ContractError(f"{name} token evidence differs from its sample count")
        permutations: list[list[int]] = []
        validated_before: list[list[bool]] = []
        validated_after: list[list[bool]] = []
        observed_nonidentity = False
        observed_valid = False
        for sample_index, (raw_permutation, raw_before, raw_after) in enumerate(
            zip(raw_permutations, valid_before, valid_after, strict=True)
        ):
            row_name = f"{name} sample {sample_index}"
            before = _list(raw_before, f"{row_name}.valid_before")
            after = _list(raw_after, f"{row_name}.valid_after")
            if (
                len(after) != len(before)
                or any(not isinstance(item, bool) for item in (*before, *after))
            ):
                raise ContractError(f"{row_name} valid masks must be equal boolean rows")
            permutation = _validate_integer_list(
                raw_permutation,
                f"{row_name}.token_permutation",
                length=len(before),
            )
            if sorted(permutation) != list(range(len(before))):
                raise ContractError(f"{row_name} token permutation must be a bijection")
            if after != [before[source] for source in permutation]:
                raise ContractError(
                    f"{row_name}.valid_after does not follow the registered token permutation"
                )
            observed_nonidentity |= permutation != list(range(len(before)))
            observed_valid |= any(before)
            permutations.append(permutation)
            validated_before.append(before)
            validated_after.append(after)
        if not observed_valid or not observed_nonidentity:
            raise ContractError(
                f"{name} must contain valid evidence and a nonidentity within-sample permutation"
            )

        factual_repeat = _validate_maximum_metric(
            payload["factual_repeat"], f"{name}.factual_repeat"
        )
        value_zero = _validate_minimum_metric(payload["value_zero"], f"{name}.value_zero")
        metadata_zero = _validate_minimum_metric(payload["metadata_zero"], f"{name}.metadata_zero")
        value_only_permutation = _validate_minimum_metric(
            payload["value_only_permutation"], f"{name}.value_only_permutation"
        )
        joint_permutation = _validate_maximum_metric(
            payload["joint_value_metadata_valid_permutation"],
            f"{name}.joint_value_metadata_valid_permutation",
        )
        stability_ceiling = max(
            factual_repeat["maximum_allowed"], joint_permutation["maximum_allowed"]
        )
        if any(
            effect["minimum_required"] <= stability_ceiling
            for effect in (value_zero, metadata_zero, value_only_permutation)
        ):
            raise ContractError(
                f"{name} intervention minimums must exceed repeat/equivariance maxima"
            )
        validated.append(
            {
                "modality": expected_modality,
                "sample_keys": sample_keys,
                "token_permutations": permutations,
                "valid_before": validated_before,
                "valid_after": validated_after,
                "factual_repeat": factual_repeat,
                "value_zero": value_zero,
                "metadata_zero": metadata_zero,
                "value_only_permutation": value_only_permutation,
                "joint_value_metadata_valid_permutation": joint_permutation,
            }
        )
    return validated, anytouch_sample_keys


def _validate_digest_state(value: object, name: str) -> dict[str, str]:
    payload = _mapping(value, name, _BOUNDARY_DIGEST_FIELDS)
    result = {field: _sha256(payload[field], f"{name}.{field}") for field in sorted(payload)}
    if len(set(result.values())) != len(result):
        raise ContractError(f"{name} contains duplicate typed state digests")
    return result


def _validate_continuation(value: object, name: str) -> dict[str, Any]:
    payload = _mapping(value, name, _CONTINUATION_FIELDS)
    result: dict[str, Any] = {
        "global_step": _integer(payload["global_step"], f"{name}.global_step", minimum=1),
        "action_loss": _number(payload["action_loss"], f"{name}.action_loss", minimum=0.0),
    }
    for field in sorted(_CONTINUATION_FIELDS.difference({"global_step", "action_loss"})):
        result[field] = _sha256(payload[field], f"{name}.{field}")
    digests = [result[field] for field in result if field.endswith("_sha256")]
    if len(set(digests)) != len(digests):
        raise ContractError(f"{name} contains duplicate typed continuation digests")
    return result


def _validate_dcp_cold_restore(value: object) -> dict[str, Any]:
    payload = _mapping(value, "DCP cold restore", _DCP_COLD_RESTORE_FIELDS)
    checkpoint_step = _integer(
        payload["checkpoint_global_step"], "DCP checkpoint global step", minimum=1
    )
    optimizer_step = _integer(payload["optimizer_step"], "DCP optimizer step", minimum=1)
    if checkpoint_step != 1 or optimizer_step != 1:
        raise ContractError("DCP cold-restore evidence must cross optimizer step 1")
    save_process = _sha256(payload["save_process_sha256"], "DCP save process")
    restore_process = _sha256(payload["restore_process_sha256"], "DCP restore process")
    if save_process == restore_process:
        raise ContractError("DCP restore must be observed in a distinct cold process")
    saved = _validate_digest_state(payload["saved_boundary"], "DCP saved boundary")
    restored = _validate_digest_state(payload["restored_boundary"], "DCP restored boundary")
    _require_canonical_equal(restored, saved, "DCP cold-restored boundary")
    uninterrupted = _validate_continuation(
        payload["uninterrupted_next_step"], "DCP uninterrupted next step"
    )
    restored_next = _validate_continuation(payload["restored_next_step"], "DCP restored next step")
    if uninterrupted["global_step"] != checkpoint_step + 1:
        raise ContractError("DCP continuation does not advance exactly one optimizer step")
    _require_canonical_equal(restored_next, uninterrupted, "DCP next-step continuation")
    for field in ("model_sha256", "optimizer_sha256"):
        if uninterrupted[field] == saved[field]:
            raise ContractError(f"DCP next step did not update {field.removesuffix('_sha256')}")
    return {
        "checkpoint_global_step": checkpoint_step,
        "optimizer_step": optimizer_step,
        "save_process_sha256": save_process,
        "restore_process_sha256": restore_process,
        "checkpoint_artifact_sha256": _sha256(
            payload["checkpoint_artifact_sha256"], "DCP checkpoint artifact"
        ),
        "saved_boundary": saved,
        "restored_boundary": restored,
        "uninterrupted_next_step": uninterrupted,
        "restored_next_step": restored_next,
    }


def _validate_full_modal_action_adoption(value: object) -> dict[str, Any]:
    payload = _mapping(value, "full-modal action adoption", _ACTION_ADOPTION_FIELDS)
    if payload["schema"] != FULL_MODAL_ACTION_ADOPTION_SCHEMA or payload["status"] != "PASS":
        raise ContractError("full-modal action-adoption schema or status differs")
    _boolean(payload["action_loss_only"], "full-modal action_loss_only", True)
    minimum_norm = _number(
        payload["nonzero_gradient_min_norm"],
        "full-modal nonzero gradient minimum",
        strictly_positive=True,
    )
    subsets, subset_anytouch_keys = _validate_presence_subsets(
        payload["presence_subsets"], minimum_norm=minimum_norm
    )
    interventions, intervention_anytouch_keys = _validate_modality_interventions(
        payload["modality_interventions"]
    )
    active_keys = _validate_string_list(
        payload["active_anytouch_sample_keys"], "active AnyTouch sample keys"
    )
    if active_keys != sorted(set(active_keys)):
        raise ContractError("active AnyTouch sample keys must be sorted and unique")
    if not set(active_keys).issubset(subset_anytouch_keys.intersection(intervention_anytouch_keys)):
        raise ContractError(
            "active AnyTouch samples must occur in both present-subset and intervention evidence"
        )
    return {
        "schema": FULL_MODAL_ACTION_ADOPTION_SCHEMA,
        "status": "PASS",
        "action_loss_only": True,
        "nonzero_gradient_min_norm": minimum_norm,
        "presence_subsets": subsets,
        "modality_interventions": interventions,
        "active_anytouch_sample_keys": active_keys,
        "dcp_cold_restore": _validate_dcp_cold_restore(payload["dcp_cold_restore"]),
    }


def validate_full_modal_action_adoption(value: object) -> dict[str, Any]:
    """Validate and canonicalize standalone ADR-150 action-adoption evidence."""

    return _canonical_copy(_validate_full_modal_action_adoption(value))


def _validate_rank_reports(
    value: object,
    *,
    world_size: int,
    steps: int,
    maximum_peak_reserved_bytes: int,
) -> list[dict[str, Any]]:
    reports = _list(value, "rank_reports")
    if len(reports) != world_size:
        raise ContractError("rank report count differs from world size")
    sample_keys_by_step: list[set[str]] = [set() for _ in range(steps)]
    for expected_rank, raw_rank in enumerate(reports):
        rank_report = _mapping(raw_rank, f"rank_reports[{expected_rank}]", _RANK_REPORT_FIELDS)
        if _integer(rank_report["rank"], "rank report rank", minimum=0) != expected_rank:
            raise ContractError("rank reports must be ordered and contiguous")
        rank_steps = _list(rank_report["steps"], f"rank {expected_rank} steps")
        if len(rank_steps) != steps:
            raise ContractError("rank report optimizer-step count differs")
        for offset, raw_step in enumerate(rank_steps):
            name = f"rank {expected_rank} step {offset + 1}"
            step = _mapping(raw_step, name, _STEP_FIELDS)
            if _integer(step["global_step"], f"{name}.global_step", minimum=1) != offset + 1:
                raise ContractError("rank report optimizer steps are not contiguous")
            sample_keys = _validate_string_list(step["sample_keys"], f"{name}.sample_keys")
            if len(sample_keys) != len(set(sample_keys)):
                raise ContractError("one rank consumed duplicate samples in one optimizer step")
            overlap = sample_keys_by_step[offset].intersection(sample_keys)
            if overlap:
                raise ContractError("distributed ranks consumed overlapping samples")
            sample_keys_by_step[offset].update(sample_keys)
            local_batch = len(sample_keys)
            _validate_integer_list(step["lane_ids"], f"{name}.lane_ids", length=local_batch)
            _validate_integer_list(
                step["frame_indices"], f"{name}.frame_indices", length=local_batch
            )
            resets = _list(step["reset"], f"{name}.reset")
            if len(resets) != local_batch or any(not isinstance(item, bool) for item in resets):
                raise ContractError(f"{name}.reset must be one boolean per sample")
            _sha256(step["source_digest"], f"{name}.source_digest")
            for seeds in ("augmentation_seeds", "flow_noise_seeds", "flow_timestep_seeds"):
                _validate_integer_list(step[seeds], f"{name}.{seeds}", length=local_batch)
            _number(step["total_loss"], f"{name}.total_loss")
            _number(step["action_loss"], f"{name}.action_loss", minimum=0)
            _number(step["moe_regularizer"], f"{name}.moe_regularizer")
            if _integer(step["official_output_arity"], f"{name}.official_output_arity") != 11:
                raise ContractError("rank step changed the released output arity")
            _boolean(step["picf_graph_installed"], f"{name}.picf_graph_installed", False)
            _validate_gradient_metrics(step["gradient_metrics"], f"{name}.gradient_metrics")
            _number(step["step_time_s"], f"{name}.step_time_s", strictly_positive=True)
            allocated = _integer(
                step["peak_cuda_allocated_bytes"], f"{name}.peak_cuda_allocated_bytes", minimum=0
            )
            reserved = _integer(
                step["peak_cuda_reserved_bytes"], f"{name}.peak_cuda_reserved_bytes", minimum=0
            )
            if allocated > reserved or reserved > maximum_peak_reserved_bytes:
                raise ContractError("rank step violates the registered CUDA reservation budget")
    return _canonical_copy(reports)


def _validated_expected_identity(
    *,
    implementation: object,
    source: object,
    model: object,
    processor: object,
    dataset: object,
    expected_placement: str,
) -> dict[str, Any]:
    impl = _mapping(
        implementation,
        "expected implementation identity",
        _IMPLEMENTATION_IDENTITY_FIELDS,
    )
    implementation_files = _validate_hash_mapping(
        impl["implementation_files"], "expected implementation files"
    )
    implementation_sha256 = _sha256(impl["implementation_sha256"], "expected implementation digest")
    if _canonical_digest(implementation_files) != implementation_sha256:
        raise ContractError("expected implementation digest is inconsistent with its file map")

    src = _mapping(source, "expected source identity", _SOURCE_IDENTITY_FIELDS)
    source_identity = {
        "source_commit": _git_revision(src["source_commit"], "expected source commit"),
        "source_patch_sha256": _sha256(src["source_patch_sha256"], "expected source patch"),
        "patched_source_sha256": _validate_hash_mapping(
            src["patched_source_sha256"], "expected patched source files"
        ),
    }

    mdl = _mapping(model, "expected model identity", _MODEL_IDENTITY_FIELDS)
    model_identity = {
        "model_family_sha256": _sha256(mdl["model_family_sha256"], "expected model family"),
        "checkpoint_revision": _git_revision(
            mdl["checkpoint_revision"], "expected checkpoint revision"
        ),
        "checkpoint_assets": _validate_assets(
            mdl["checkpoint_assets"], "expected checkpoint assets"
        ),
        "qwen_vision_geometry": _validate_geometry(mdl["qwen_vision_geometry"]),
        "parameter_storage": _validate_storage(
            mdl["parameter_storage"], expected_placement=expected_placement
        ),
        "parameter_manifest": _validate_parameter_manifest(mdl["parameter_manifest"]),
        "alignment_teacher_prune": _validate_prune(mdl["alignment_teacher_prune"]),
    }

    proc = _mapping(processor, "expected processor identity", _PROCESSOR_IDENTITY_FIELDS)
    processor_identity = {
        "processor_revision": _git_revision(
            proc["processor_revision"], "expected processor revision"
        ),
        "processor_assets": _validate_assets(proc["processor_assets"], "expected processor assets"),
    }
    return {
        "implementation": {
            "implementation_files": implementation_files,
            "implementation_sha256": implementation_sha256,
        },
        "source": source_identity,
        "model": model_identity,
        "processor": processor_identity,
        "dataset": _validate_dataset_contract(dataset, "expected dataset identity"),
    }


def validate_adr150_matched_lbot_report(
    report: object,
    *,
    expected_plan_sha256: str,
    expected_representation_split_sha256: str,
    expected_evaluation_plan_sha256: str,
    expected_seed: int,
    expected_implementation_identity: Mapping[str, object],
    expected_source_identity: Mapping[str, object],
    expected_model_identity: Mapping[str, object],
    expected_processor_identity: Mapping[str, object],
    expected_dataset_identity: Mapping[str, object],
    expected_optimizer_contract: Mapping[str, object],
    expected_world_size: int = 4,
    expected_steps: int = 200,
    expected_learning_rate: float = 1e-4,
    expected_max_grad_norm: float = 1.0,
    expected_registered_evaluation_steps: Sequence[int] = (0, 20, 100, 200),
    expected_minimum_future_source_frames: int = 0,
    expected_maximum_control_tokens: int = 64,
    expected_fsdp2_placement: str = "selective-embedding-offload",
    expected_cuda_allocator: str = "native",
    expected_attention_implementation: str = "eager",
    expected_maximum_peak_reserved_bytes: int = 39 * 1024**3,
) -> ValidatedADR150LBOTReport:
    """Validate one matched four-GPU LBOT report against independently frozen inputs.

    Identity mappings are deliberately complete and exact.  The model identity
    contains the checkpoint assets, geometry, parameter storage/manifest and
    targetless-prune report; the dataset identity is the complete producer
    ``dataset_contract`` object.
    """

    plan_sha256 = _sha256(expected_plan_sha256, "expected plan")
    split_sha256 = _sha256(expected_representation_split_sha256, "expected split")
    evaluation_sha256 = _sha256(expected_evaluation_plan_sha256, "expected evaluation")
    world_size = _integer(expected_world_size, "expected world size", minimum=1)
    steps = _integer(expected_steps, "expected steps", minimum=1)
    seed = _integer(expected_seed, "expected seed", minimum=0)
    maximum_control_tokens = _integer(
        expected_maximum_control_tokens, "expected maximum control tokens", minimum=1
    )
    minimum_future_source_frames = _integer(
        expected_minimum_future_source_frames,
        "expected minimum future source frames",
        minimum=0,
    )
    maximum_peak_reserved_bytes = _integer(
        expected_maximum_peak_reserved_bytes,
        "expected maximum peak reserved bytes",
        minimum=1,
    )
    if world_size != 4 or steps != 200:
        raise ContractError("ADR-150 matched LBOT requires exactly four ranks and 200 steps")
    evaluation_steps = tuple(
        _integer(value, "expected evaluation step", minimum=0)
        for value in expected_registered_evaluation_steps
    )
    if (
        not evaluation_steps
        or evaluation_steps != tuple(sorted(set(evaluation_steps)))
        or evaluation_steps[0] != 0
        or evaluation_steps[-1] != steps
    ):
        raise ContractError("expected evaluation steps must be unique, sorted and span 0..steps")
    _number(expected_learning_rate, "expected learning rate", strictly_positive=True)
    _number(expected_max_grad_norm, "expected max grad norm", strictly_positive=True)
    if not isinstance(expected_fsdp2_placement, str) or not expected_fsdp2_placement:
        raise ContractError("expected FSDP2 placement must be one nonempty string")
    if not isinstance(expected_cuda_allocator, str) or not expected_cuda_allocator:
        raise ContractError("expected CUDA allocator must be one nonempty string")
    if expected_attention_implementation not in _ATTENTION_IMPLEMENTATIONS:
        raise ContractError("expected attention implementation is unsupported")

    expected_identity = _validated_expected_identity(
        implementation=expected_implementation_identity,
        source=expected_source_identity,
        model=expected_model_identity,
        processor=expected_processor_identity,
        dataset=expected_dataset_identity,
        expected_placement=expected_fsdp2_placement,
    )
    frozen_optimizer = _validate_optimizer(
        expected_optimizer_contract,
        expected_learning_rate=expected_learning_rate,
    )
    report_fields = set(report) if isinstance(report, Mapping) else set()
    accepted_fields = next(
        (fields for fields in _ACCEPTED_TOP_LEVEL_FIELD_SETS if report_fields == fields),
        _TOP_LEVEL_FIELDS,
    )
    payload = _mapping(report, "ADR-150 matched LBOT report", accepted_fields)
    _canonical_json(payload)

    observed_report_schema = _text(payload["schema"], "schema")
    if observed_report_schema not in ACCEPTED_LBOT_REPORT_SCHEMAS:
        raise ContractError("schema is not an accepted released LingBot control report")

    fixed_values = {
        "status": "PASS",
        "architecture_identity": _ARCHITECTURE,
        "picf_graph_installed": False,
        "physical_sidecar_read": False,
        "task_scorer_present": False,
        "action_suffix_executed": True,
        "posterior_present": False,
        "physical_event_stream": True,
        "maximum_control_tokens": maximum_control_tokens,
        "checkpoint_published": False,
        "curve_mode": True,
        "world_size": world_size,
        "steps": steps,
        "seed": seed,
        "official_output_arity": 11,
        "fsdp2_placement": expected_fsdp2_placement,
        "cuda_allocator": expected_cuda_allocator,
        "gradient_checkpointing": True,
        "maximum_peak_reserved_bytes": maximum_peak_reserved_bytes,
    }
    for name, expected in fixed_values.items():
        actual = payload[name]
        if isinstance(expected, bool):
            _boolean(actual, name, expected)
        elif isinstance(expected, int):
            if _integer(actual, name) != expected:
                raise ContractError(f"{name} differs from its frozen value")
        elif not isinstance(actual, str) or actual != expected:
            raise ContractError(f"{name} differs from its frozen value")
    observed_minimum_future_source_frames = _integer(
        payload.get("minimum_future_source_frames", 0),
        "minimum_future_source_frames",
        minimum=0,
    )
    if observed_minimum_future_source_frames != minimum_future_source_frames:
        raise ContractError("minimum_future_source_frames differs from its frozen value")
    observed_attention_implementation = _text(
        payload.get("attention_implementation", "eager"),
        "attention_implementation",
    )
    if observed_attention_implementation != expected_attention_implementation:
        raise ContractError("attention_implementation differs from its frozen value")

    for name, expected in (
        ("plan_sha256", plan_sha256),
        ("representation_split_sha256", split_sha256),
        ("evaluation_plan_sha256", evaluation_sha256),
    ):
        if _sha256(payload[name], name) != expected:
            raise ContractError(f"{name} differs from its frozen value")
    _same_float(payload["max_grad_norm"], expected_max_grad_norm, "max_grad_norm")

    observed_evaluation_steps = _list(
        payload["registered_evaluation_steps"], "registered_evaluation_steps"
    )
    if tuple(observed_evaluation_steps) != evaluation_steps or any(
        isinstance(value, bool) or not isinstance(value, int) for value in observed_evaluation_steps
    ):
        raise ContractError("registered evaluation steps differ from the frozen curve")
    _validate_evaluation_snapshots(payload["evaluation_snapshots"], expected_steps=evaluation_steps)

    observed_implementation = {
        "implementation_files": _validate_hash_mapping(
            payload["implementation_files"], "implementation files"
        ),
        "implementation_sha256": _sha256(payload["implementation_sha256"], "implementation digest"),
    }
    if (
        _canonical_digest(observed_implementation["implementation_files"])
        != observed_implementation["implementation_sha256"]
    ):
        raise ContractError("implementation digest is inconsistent with the reported file map")
    observed_source = {
        "source_commit": _git_revision(payload["source_commit"], "source commit"),
        "source_patch_sha256": _sha256(payload["source_patch_sha256"], "source patch"),
        "patched_source_sha256": _validate_hash_mapping(
            payload["patched_source_sha256"], "patched source files"
        ),
    }
    observed_model = {
        "model_family_sha256": _sha256(payload["model_family_sha256"], "model family"),
        "checkpoint_revision": _git_revision(payload["checkpoint_revision"], "checkpoint revision"),
        "checkpoint_assets": _validate_assets(payload["checkpoint_assets"], "checkpoint assets"),
        "qwen_vision_geometry": _validate_geometry(payload["qwen_vision_geometry"]),
        "parameter_storage": _validate_storage(
            payload["parameter_storage"], expected_placement=expected_fsdp2_placement
        ),
        "parameter_manifest": _validate_parameter_manifest(payload["parameter_manifest"]),
        "alignment_teacher_prune": _validate_prune(payload["alignment_teacher_prune"]),
    }
    observed_processor = {
        "processor_revision": _git_revision(payload["processor_revision"], "processor revision"),
        "processor_assets": _validate_assets(payload["processor_assets"], "processor assets"),
    }
    observed_dataset = _validate_dataset_contract(payload["dataset_contract"], "dataset_contract")
    for name, actual, expected in (
        ("implementation", observed_implementation, expected_identity["implementation"]),
        ("source", observed_source, expected_identity["source"]),
        ("model", observed_model, expected_identity["model"]),
        ("processor", observed_processor, expected_identity["processor"]),
        ("dataset", observed_dataset, expected_identity["dataset"]),
    ):
        _require_canonical_equal(actual, expected, name)

    expected_model_family = _canonical_digest(
        {
            "architecture": _ARCHITECTURE,
            "checkpoint_revision": observed_model["checkpoint_revision"],
            "implementation_sha256": observed_implementation["implementation_sha256"],
            "plan_sha256": plan_sha256,
        }
    )
    if observed_model["model_family_sha256"] != expected_model_family:
        raise ContractError("model family digest is inconsistent with its producer preimage")

    observed_optimizer = _validate_optimizer(
        payload["optimizer_contract"], expected_learning_rate=expected_learning_rate
    )
    _require_canonical_equal(observed_optimizer, frozen_optimizer, "optimizer contract")
    if "lingbot_base_family" in payload:
        observed_base_family = _validate_lingbot_base_family(
            payload["lingbot_base_family"],
            expected_learning_rate=expected_learning_rate,
        )
        expected_base_family_semantic = {
            "schema": "picf-next.lingbot-base-family.v1",
            "architecture": _ARCHITECTURE,
            "source_commit": observed_source["source_commit"],
            "native_patch_sha256": observed_source["source_patch_sha256"],
            "checkpoint_revision": observed_model["checkpoint_revision"],
            "checkpoint_assets": observed_model["checkpoint_assets"],
            "processor_revision": observed_processor["processor_revision"],
            "processor_assets": observed_processor["processor_assets"],
            "attention_implementation": observed_attention_implementation,
            "trainable_scope": "full-host",
            "optimizer_contract": observed_optimizer,
            "maximum_control_tokens": maximum_control_tokens,
        }
        expected_base_family = {
            **expected_base_family_semantic,
            "artifact_sha256": _canonical_digest(expected_base_family_semantic),
        }
        _require_canonical_equal(
            observed_base_family,
            expected_base_family,
            "LingBot base family",
        )
        if (
            _sha256(payload["lingbot_base_family_sha256"], "LingBot base-family digest")
            != observed_base_family["artifact_sha256"]
        ):
            raise ContractError("LingBot base-family top-level digest differs")
    _validate_rank_reports(
        payload["rank_reports"],
        world_size=world_size,
        steps=steps,
        maximum_peak_reserved_bytes=maximum_peak_reserved_bytes,
    )
    full_modal_action_adoption = _validate_full_modal_action_adoption(
        payload["full_modal_action_adoption"]
    )

    canonical_report = _canonical_copy(payload)
    report_sha256 = _canonical_digest(canonical_report)
    expected_contract = {
        "plan_sha256": plan_sha256,
        "representation_split_sha256": split_sha256,
        "evaluation_plan_sha256": evaluation_sha256,
        "world_size": world_size,
        "steps": steps,
        "seed": seed,
        "learning_rate": float(expected_learning_rate),
        "max_grad_norm": float(expected_max_grad_norm),
        "registered_evaluation_steps": list(evaluation_steps),
        "minimum_future_source_frames": minimum_future_source_frames,
        "maximum_control_tokens": maximum_control_tokens,
        "fsdp2_placement": expected_fsdp2_placement,
        "cuda_allocator": expected_cuda_allocator,
        "maximum_peak_reserved_bytes": maximum_peak_reserved_bytes,
        "full_modal_action_adoption_schema": FULL_MODAL_ACTION_ADOPTION_SCHEMA,
        "identities": expected_identity,
        "optimizer_contract": frozen_optimizer,
    }
    validation_payload = {
        "schema": VALIDATION_SCHEMA,
        "status": "PASS",
        "lbot_report_schema": observed_report_schema,
        "lbot_report_sha256": report_sha256,
        "expected_contract_sha256": _canonical_digest(expected_contract),
        "world_size": world_size,
        "steps": steps,
        "registered_evaluation_steps": list(evaluation_steps),
        "full_modal_action_adoption_sha256": _canonical_digest(full_modal_action_adoption),
    }
    validation_report = {
        **validation_payload,
        "artifact_sha256": _canonical_digest(validation_payload),
    }
    return ValidatedADR150LBOTReport(
        canonical_report=canonical_report,
        report_sha256=report_sha256,
        validation_report=validation_report,
    )
