"""Fail-closed pairing for factual and blocked LTOP core-pilot curves."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from picf_next.lingbot_native.ltop_core_pilot import (
    LTOP_CORE_PILOT_CHECKPOINT_STEP,
    LTOP_CORE_PILOT_DIAGNOSTICS_EVERY,
    LTOP_CORE_PILOT_METRICS_EVERY,
    LTOP_CORE_PILOT_SCHEMA,
    LTOP_CORE_PILOT_TOTAL_STEPS,
    LTOP_CORE_PILOT_WORLD_SIZE,
    LTOPCorePilotArm,
    matched_arm_contract,
)

LTOP_CORE_PAIR_SCHEMA: Final = "picf-next.ltop-core-paired-curves.v1"
LTOP_CORE_PAIR_JOURNAL_SCHEMA: Final = "picf-next.ltop-core-pilot-rank-journal.v1"
LTOP_CORE_PAIR_OPTIMIZER_INITIALIZATION_SCHEMA: Final = (
    "picf-next.ltop-core-pilot-optimizer-initialization.v1"
)
LTOP_CORE_PAIR_METRICS_SCHEMA: Final = "picf-next.ltop-core-pilot-metrics.v1"
LTOP_CORE_PAIR_CHECKPOINT_SCHEMA: Final = "picf-next.ltop-core-pilot-checkpoint.v1"

_EXPECTED_CADENCE: Final = {
    "total_steps": LTOP_CORE_PILOT_TOTAL_STEPS,
    "metrics_every": LTOP_CORE_PILOT_METRICS_EVERY,
    "diagnostics_every": LTOP_CORE_PILOT_DIAGNOSTICS_EVERY,
    "checkpoint_step": LTOP_CORE_PILOT_CHECKPOINT_STEP,
}
_EXPECTED_METRIC_WINDOWS: Final = LTOP_CORE_PILOT_TOTAL_STEPS // LTOP_CORE_PILOT_METRICS_EVERY

_REPORT_KEYS: Final = frozenset(
    {
        "schema",
        "status",
        "failures",
        "mode",
        "arm",
        "arm_contract",
        "architecture_identity",
        "world_size",
        "steps",
        "cadence",
        "seed",
        "capacity",
        "task_query_count",
        "stage_checkpoint",
        "g2_report_sha256",
        "g3_report_sha256",
        "dataset_contract",
        "stream_plan_sha256",
        "representation_split_sha256",
        "evaluation_plan_sha256",
        "execution_contract_sha256",
        "offline_labels_sha256",
        "physical_sidecar_manifest_sha256",
        "source_identity",
        "runtime_environment_contract",
        "action_inference_contract",
        "training_contract",
        "checkpoint",
        "scientific_boundary",
        "rank_reports",
    }
)
_REPORT_INVARIANTS: Final = (
    "mode",
    "architecture_identity",
    "world_size",
    "steps",
    "cadence",
    "seed",
    "capacity",
    "task_query_count",
    "stage_checkpoint",
    "g2_report_sha256",
    "g3_report_sha256",
    "dataset_contract",
    "stream_plan_sha256",
    "representation_split_sha256",
    "evaluation_plan_sha256",
    "execution_contract_sha256",
    "offline_labels_sha256",
    "physical_sidecar_manifest_sha256",
    "source_identity",
    "runtime_environment_contract",
    "action_inference_contract",
    "training_contract",
)
_REPORT_DIGEST_FIELDS: Final = (
    "g2_report_sha256",
    "g3_report_sha256",
    "stream_plan_sha256",
    "representation_split_sha256",
    "evaluation_plan_sha256",
    "execution_contract_sha256",
    "offline_labels_sha256",
    "physical_sidecar_manifest_sha256",
)

_RANK_REPORT_KEYS: Final = frozenset(
    {
        "rank",
        "metric_reports",
        "diagnostics",
        "all_gradients_finite",
        "action_loss_first_100_mean",
        "action_loss_last_100_mean",
        "optimizer_parameter_manifest",
        "optimizer_initialization",
        "journal",
        "stage_restore",
        "checkpoint",
        "timings",
        "cuda_memory_bytes",
    }
)
_OPTIMIZER_MANIFEST_KEYS: Final = frozenset(
    {"canonical_names", "parameter_count", "trainable_numel", "schema_sha256"}
)
_OPTIMIZER_INITIALIZATION_KEYS: Final = frozenset(
    {
        "schema",
        "rank",
        "fresh_zero_state",
        "state_entry_count",
        "parameter_manifest_sha256",
        "parameter_groups_sha256",
        "optimizer_state_sha256",
        "model_local_state_sha256",
        "rank_rng_state_sha256",
    }
)
_JOURNAL_RECEIPT_KEYS: Final = frozenset({"schema", "rank", "path", "file_sha256", "record_count"})
_CHECKPOINT_RECEIPT_KEYS: Final = frozenset({"path", "manifest_sha256"})

_STAGE_RESTORE_KEYS: Final = frozenset(
    {
        "rank",
        "hostname",
        "pid",
        "expected_model_local_state_sha256",
        "actual_model_local_state_sha256",
        "digest_match",
        "meta_state_names_before_load",
        "meta_state_names_after_load",
        "fsdp2_storage_before_load",
        "fsdp2_storage_after_load",
        "timings",
        "cuda_memory_bytes",
    }
)
_STAGE_STABLE_FIELDS: Final = (
    "rank",
    "expected_model_local_state_sha256",
    "actual_model_local_state_sha256",
    "digest_match",
    "meta_state_names_before_load",
    "meta_state_names_after_load",
    "fsdp2_storage_before_load",
    "fsdp2_storage_after_load",
)

_STEP_MATCHED_FIELDS: Final = (
    "global_step",
    "sample_keys",
    "lane_ids",
    "frame_indices",
    "reset",
    "source_digest",
    "augmentation_seeds",
    "flow_noise_seeds",
    "flow_timestep_seeds",
    "target_identity",
    "target_row",
    "model_input_sha256",
    "controls_sha256",
    "prior_controls_sha256",
    "structural_targets_sha256",
    "normalized_forward_input_sha256",
)
_STEP_DIGEST_FIELDS: Final = (
    "source_digest",
    "model_input_sha256",
    "controls_sha256",
    "prior_controls_sha256",
    "structural_targets_sha256",
    "normalized_forward_input_sha256",
    "forward_input_sha256",
)
_STEP_KEYS: Final = frozenset(
    {
        *_STEP_MATCHED_FIELDS,
        "forward_input_sha256",
        "executed_object_read_action_intervention",
        "total_loss",
        "action_loss",
        "moe_regularizer",
        "physical_set_loss",
        "task_address_loss",
        "gradient_metrics",
        "step_time_s",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
    }
)

_METRIC_RECEIPT_KEYS: Final = frozenset({"path", "file_sha256", "start_step", "end_step", "means"})
_METRIC_ARTIFACT_KEYS: Final = frozenset(
    {"schema", "arm", "start_step", "end_step", "sample_count", "means", "rank_windows"}
)
_METRIC_MEAN_FIELDS: Final = frozenset(
    {
        "total_loss",
        "action_loss",
        "moe_regularizer",
        "physical_set_loss",
        "task_address_loss",
        "step_time_s",
    }
)
_CHECKPOINT_MANIFEST_KEYS: Final = frozenset(
    {
        "schema",
        "status",
        "global_step",
        "arm",
        "g2_report_sha256",
        "g3_report_sha256",
        "stream_plan_sha256",
        "rank_boundaries",
    }
)
_CHECKPOINT_BOUNDARY_KEYS: Final = frozenset(
    {
        "model_local_state_sha256",
        "optimizer_local_state_sha256",
        "lane_snapshot_sha256",
        "rank_rng_state_sha256",
    }
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(payload: bytes, *, name: str) -> Any:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not ASCII JSON") from error

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_non_finite_constant(value: str) -> Any:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        result = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON") from error
    _require_finite_json(result, name=name)
    return result


def _require_finite_json(value: Any, *, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json(child, name=f"{name}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_json(child, name=f"{name}[{index}]")


def _read_regular_bytes(path: Path, *, name: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{name} is absent or not a direct regular file: {path}")
    return path.read_bytes()


def _load_json_file(path: Path, *, name: str) -> tuple[dict[str, Any], str]:
    payload = _read_regular_bytes(path, name=name)
    value = _strict_json(payload, name=name)
    return _require_mapping(value, name=name), _sha256_bytes(payload)


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _require_list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _require_exact_keys(value: dict[str, Any], expected: frozenset[str], *, name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{name} fields differ: missing={missing}, unexpected={unexpected}")


def _require_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_finite_number(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _resolve_expected_file(
    report_path: Path,
    *,
    declared: Any,
    expected: Path,
    name: str,
) -> Path:
    if not isinstance(declared, str) or not declared:
        raise TypeError(f"{name} path must be a non-empty string")
    candidate = Path(declared)
    if not candidate.is_absolute():
        candidate = report_path.parent / candidate
    if candidate.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError(f"{name} path is outside its registered location")
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"{name} is absent or not a direct regular file: {candidate}")
    return candidate


def _load_report(path: Path, *, arm: LTOPCorePilotArm) -> tuple[dict[str, Any], str]:
    report, digest = _load_json_file(path, name=f"LTOP core-pilot {arm.value} report")
    _require_exact_keys(report, _REPORT_KEYS, name=f"LTOP core-pilot {arm.value} report")
    expected = {
        "schema": LTOP_CORE_PILOT_SCHEMA,
        "status": "PASS",
        "failures": [],
        "mode": "pilot",
        "arm": arm.value,
        "arm_contract": matched_arm_contract(arm),
        "world_size": LTOP_CORE_PILOT_WORLD_SIZE,
        "steps": LTOP_CORE_PILOT_TOTAL_STEPS,
        "cadence": _EXPECTED_CADENCE,
    }
    for field, value in expected.items():
        if report[field] != value:
            raise ValueError(f"LTOP core-pilot {arm.value} report violates {field}")
    for field in _REPORT_DIGEST_FIELDS:
        _require_sha256(report[field], name=f"LTOP core-pilot {arm.value}.{field}")
    for field in ("source_identity", "runtime_environment_contract"):
        value = _require_mapping(report[field], name=f"LTOP core-pilot {arm.value}.{field}")
        if not value:
            raise ValueError(f"LTOP core-pilot {arm.value}.{field} is empty")
    for field in ("dataset_contract", "action_inference_contract", "training_contract"):
        _require_mapping(report[field], name=f"LTOP core-pilot {arm.value}.{field}")
    _require_int(report["seed"], name=f"LTOP core-pilot {arm.value}.seed")
    _require_int(report["capacity"], name=f"LTOP core-pilot {arm.value}.capacity", minimum=1)
    _require_int(
        report["task_query_count"],
        name=f"LTOP core-pilot {arm.value}.task_query_count",
        minimum=1,
    )
    if not isinstance(report["architecture_identity"], str) or not report["architecture_identity"]:
        raise ValueError(f"LTOP core-pilot {arm.value} architecture identity is empty")
    if not isinstance(report["stage_checkpoint"], str) or not report["stage_checkpoint"]:
        raise ValueError(f"LTOP core-pilot {arm.value} stage checkpoint is empty")
    if not isinstance(report["scientific_boundary"], str) or not report["scientific_boundary"]:
        raise ValueError(f"LTOP core-pilot {arm.value} scientific boundary is empty")
    _require_mapping(report["checkpoint"], name=f"LTOP core-pilot {arm.value}.checkpoint")

    rank_reports = _require_list(
        report["rank_reports"], name=f"LTOP core-pilot {arm.value}.rank_reports"
    )
    if len(rank_reports) != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError(f"LTOP core-pilot {arm.value} omits one rank")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rank_reports):
        rank_report = _require_mapping(
            raw, name=f"LTOP core-pilot {arm.value}.rank_reports[{index}]"
        )
        _require_exact_keys(
            rank_report,
            _RANK_REPORT_KEYS,
            name=f"LTOP core-pilot {arm.value}.rank_reports[{index}]",
        )
        rank = _require_int(
            rank_report["rank"], name=f"LTOP core-pilot {arm.value}.rank_reports[{index}].rank"
        )
        if rank not in range(LTOP_CORE_PILOT_WORLD_SIZE):
            raise ValueError(f"LTOP core-pilot {arm.value} rank {rank} is invalid")
        if rank_report["all_gradients_finite"] is not True:
            raise ValueError(f"LTOP core-pilot {arm.value} rank {rank} has non-finite gradients")
        for field in ("action_loss_first_100_mean", "action_loss_last_100_mean"):
            _require_finite_number(
                rank_report[field],
                name=f"LTOP core-pilot {arm.value} rank {rank}.{field}",
                nonnegative=True,
            )
        _require_list(
            rank_report["metric_reports"],
            name=f"LTOP core-pilot {arm.value} rank {rank}.metric_reports",
        )
        _require_list(
            rank_report["diagnostics"],
            name=f"LTOP core-pilot {arm.value} rank {rank}.diagnostics",
        )
        for field in (
            "optimizer_parameter_manifest",
            "optimizer_initialization",
            "journal",
            "stage_restore",
            "checkpoint",
            "timings",
            "cuda_memory_bytes",
        ):
            _require_mapping(
                rank_report[field], name=f"LTOP core-pilot {arm.value} rank {rank}.{field}"
            )
        normalized.append(rank_report)
    if {item["rank"] for item in normalized} != set(range(LTOP_CORE_PILOT_WORLD_SIZE)):
        raise ValueError(f"LTOP core-pilot {arm.value} rank set is invalid")
    return report, digest


def _rank_map(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {item["rank"]: item for item in report["rank_reports"]}


def _validate_optimizer_manifest(value: Any, *, arm: LTOPCorePilotArm, rank: int) -> dict[str, Any]:
    manifest = _require_mapping(value, name=f"{arm.value} rank {rank} optimizer manifest")
    _require_exact_keys(
        manifest, _OPTIMIZER_MANIFEST_KEYS, name=f"{arm.value} rank {rank} optimizer manifest"
    )
    names = _require_list(
        manifest["canonical_names"], name=f"{arm.value} rank {rank} optimizer canonical_names"
    )
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{arm.value} rank {rank} optimizer canonical_names are invalid")
    if len(set(names)) != len(names):
        raise ValueError(f"{arm.value} rank {rank} optimizer canonical_names contain duplicates")
    parameter_count = _require_int(
        manifest["parameter_count"],
        name=f"{arm.value} rank {rank} optimizer parameter_count",
        minimum=1,
    )
    if parameter_count != len(names):
        raise ValueError(f"{arm.value} rank {rank} optimizer parameter_count is inconsistent")
    _require_int(
        manifest["trainable_numel"],
        name=f"{arm.value} rank {rank} optimizer trainable_numel",
        minimum=1,
    )
    _require_sha256(
        manifest["schema_sha256"], name=f"{arm.value} rank {rank} optimizer schema_sha256"
    )
    return manifest


def _validate_stage_restore(value: Any, *, arm: LTOPCorePilotArm, rank: int) -> dict[str, Any]:
    restore = _require_mapping(value, name=f"{arm.value} rank {rank} stage_restore")
    _require_exact_keys(restore, _STAGE_RESTORE_KEYS, name=f"{arm.value} rank {rank} stage_restore")
    if restore["rank"] != rank:
        raise ValueError(f"{arm.value} rank {rank} stage_restore rank differs")
    if not isinstance(restore["hostname"], str) or not restore["hostname"]:
        raise ValueError(f"{arm.value} rank {rank} stage_restore hostname is empty")
    _require_int(restore["pid"], name=f"{arm.value} rank {rank} stage_restore pid", minimum=1)
    expected_digest = _require_sha256(
        restore["expected_model_local_state_sha256"],
        name=f"{arm.value} rank {rank} expected model digest",
    )
    actual_digest = _require_sha256(
        restore["actual_model_local_state_sha256"],
        name=f"{arm.value} rank {rank} actual model digest",
    )
    if restore["digest_match"] is not True or expected_digest != actual_digest:
        raise ValueError(f"{arm.value} rank {rank} stage restore did not match its model digest")
    for field in ("meta_state_names_before_load", "meta_state_names_after_load"):
        names = _require_list(restore[field], name=f"{arm.value} rank {rank} stage_restore.{field}")
        if names:
            raise ValueError(f"{arm.value} rank {rank} stage restore retained meta parameters")
    for field in ("fsdp2_storage_before_load", "fsdp2_storage_after_load"):
        _require_mapping(restore[field], name=f"{arm.value} rank {rank} stage_restore.{field}")
    timings = _require_mapping(
        restore["timings"], name=f"{arm.value} rank {rank} stage_restore.timings"
    )
    _require_exact_keys(
        timings,
        frozenset({"model_build_s", "dcp_load_s"}),
        name=f"{arm.value} rank {rank} stage_restore.timings",
    )
    for field in timings:
        _require_finite_number(
            timings[field],
            name=f"{arm.value} rank {rank} stage_restore.timings.{field}",
            nonnegative=True,
        )
    memory = _require_mapping(
        restore["cuda_memory_bytes"],
        name=f"{arm.value} rank {rank} stage_restore.cuda_memory_bytes",
    )
    expected_memory_fields = frozenset({"allocated", "reserved", "peak_allocated", "peak_reserved"})
    _require_exact_keys(
        memory,
        expected_memory_fields,
        name=f"{arm.value} rank {rank} stage_restore.cuda_memory_bytes",
    )
    for field in memory:
        _require_int(
            memory[field],
            name=f"{arm.value} rank {rank} stage_restore.cuda_memory_bytes.{field}",
            minimum=0,
        )
    return {field: restore[field] for field in _STAGE_STABLE_FIELDS}


def _validate_optimizer_initialization(
    value: Any,
    *,
    arm: LTOPCorePilotArm,
    rank: int,
    manifest: dict[str, Any],
    stage_restore: dict[str, Any],
) -> dict[str, Any]:
    receipt = _require_mapping(value, name=f"{arm.value} rank {rank} optimizer_initialization")
    _require_exact_keys(
        receipt,
        _OPTIMIZER_INITIALIZATION_KEYS,
        name=f"{arm.value} rank {rank} optimizer_initialization",
    )
    expected = {
        "schema": LTOP_CORE_PAIR_OPTIMIZER_INITIALIZATION_SCHEMA,
        "rank": rank,
        "fresh_zero_state": True,
        "state_entry_count": 0,
        "parameter_manifest_sha256": manifest["schema_sha256"],
        "model_local_state_sha256": stage_restore["actual_model_local_state_sha256"],
    }
    for field, expected_value in expected.items():
        if receipt[field] != expected_value:
            raise ValueError(f"{arm.value} rank {rank} optimizer initialization violates {field}")
    for field in (
        "parameter_manifest_sha256",
        "parameter_groups_sha256",
        "optimizer_state_sha256",
        "model_local_state_sha256",
        "rank_rng_state_sha256",
    ):
        _require_sha256(
            receipt[field],
            name=f"{arm.value} rank {rank} optimizer_initialization.{field}",
        )
    return receipt


def _validate_step_record(
    value: Any,
    *,
    arm: LTOPCorePilotArm,
    rank: int,
    step: int,
) -> dict[str, Any]:
    record = _require_mapping(value, name=f"{arm.value} rank {rank} journal step {step}")
    _require_exact_keys(record, _STEP_KEYS, name=f"{arm.value} rank {rank} journal step {step}")
    if record["global_step"] != step:
        raise ValueError(f"{arm.value} rank {rank} journal is not contiguous at step {step}")
    sample_keys = _require_list(
        record["sample_keys"], name=f"{arm.value} rank {rank} step {step}.sample_keys"
    )
    if not sample_keys or any(not isinstance(item, str) or not item for item in sample_keys):
        raise ValueError(f"{arm.value} rank {rank} step {step} sample_keys are invalid")
    batch_size = len(sample_keys)
    for field in ("lane_ids", "frame_indices", "reset"):
        values = _require_list(record[field], name=f"{arm.value} rank {rank} step {step}.{field}")
        if len(values) != batch_size:
            raise ValueError(f"{arm.value} rank {rank} step {step}.{field} has wrong batch size")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in record["lane_ids"]):
        raise TypeError(f"{arm.value} rank {rank} step {step} lane_ids are invalid")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in record["frame_indices"]):
        raise TypeError(f"{arm.value} rank {rank} step {step} frame_indices are invalid")
    if any(not isinstance(item, bool) for item in record["reset"]):
        raise TypeError(f"{arm.value} rank {rank} step {step} reset flags are invalid")
    for field in ("augmentation_seeds", "flow_noise_seeds", "flow_timestep_seeds"):
        values = _require_list(record[field], name=f"{arm.value} rank {rank} step {step}.{field}")
        if len(values) != batch_size or any(
            isinstance(item, bool) or not isinstance(item, int) for item in values
        ):
            raise TypeError(f"{arm.value} rank {rank} step {step}.{field} is invalid")
    for field in _STEP_DIGEST_FIELDS:
        _require_sha256(record[field], name=f"{arm.value} rank {rank} step {step}.{field}")
    expected_intervention = "factual" if arm is LTOPCorePilotArm.FACTUAL else "blocked"
    if record["executed_object_read_action_intervention"] != expected_intervention:
        raise ValueError(
            f"{arm.value} rank {rank} step {step} executed the wrong "
            "OBJECT_READ->ACTION intervention"
        )
    target_identity = record["target_identity"]
    target_row = record["target_row"]
    if target_identity is None:
        if target_row is not None:
            raise ValueError(
                f"{arm.value} rank {rank} step {step} has a row without an exact target"
            )
    else:
        if not isinstance(target_identity, str) or not target_identity:
            raise ValueError(f"{arm.value} rank {rank} step {step} target_identity is invalid")
        _require_int(
            target_row,
            name=f"{arm.value} rank {rank} step {step}.target_row",
            minimum=0,
        )
    for field in (
        "total_loss",
        "action_loss",
        "moe_regularizer",
        "physical_set_loss",
        "task_address_loss",
    ):
        _require_finite_number(
            record[field],
            name=f"{arm.value} rank {rank} step {step}.{field}",
            nonnegative=field == "action_loss",
        )
    _require_mapping(
        record["gradient_metrics"],
        name=f"{arm.value} rank {rank} step {step}.gradient_metrics",
    )
    _require_finite_number(
        record["step_time_s"],
        name=f"{arm.value} rank {rank} step {step}.step_time_s",
        nonnegative=True,
    )
    for field in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
        _require_int(record[field], name=f"{arm.value} rank {rank} step {step}.{field}", minimum=0)
    return record


def _load_journal(
    report_path: Path,
    *,
    arm: LTOPCorePilotArm,
    rank: int,
    receipt_value: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = _require_mapping(receipt_value, name=f"{arm.value} rank {rank} journal receipt")
    _require_exact_keys(
        receipt, _JOURNAL_RECEIPT_KEYS, name=f"{arm.value} rank {rank} journal receipt"
    )
    if receipt["schema"] != LTOP_CORE_PAIR_JOURNAL_SCHEMA or receipt["rank"] != rank:
        raise ValueError(f"{arm.value} rank {rank} journal receipt identity is invalid")
    if receipt["record_count"] != LTOP_CORE_PILOT_TOTAL_STEPS:
        raise ValueError(f"{arm.value} rank {rank} journal receipt count is invalid")
    declared_sha = _require_sha256(
        receipt["file_sha256"], name=f"{arm.value} rank {rank} journal receipt SHA"
    )
    expected_path = report_path.parent / "metrics" / "rank_journal" / f"rank_{rank}.jsonl"
    path = _resolve_expected_file(
        report_path,
        declared=receipt["path"],
        expected=expected_path,
        name=f"{arm.value} rank {rank} journal",
    )
    payload = _read_regular_bytes(path, name=f"{arm.value} rank {rank} journal")
    actual_sha = _sha256_bytes(payload)
    if actual_sha != declared_sha:
        raise ValueError(f"{arm.value} rank {rank} journal SHA differs from its report")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{arm.value} rank {rank} journal is not ASCII") from error
    lines = text.splitlines()
    if len(lines) != LTOP_CORE_PILOT_TOTAL_STEPS or any(not line for line in lines):
        raise ValueError(f"{arm.value} rank {rank} journal is incomplete")
    records = [
        _validate_step_record(
            _strict_json(line.encode("ascii"), name=f"{arm.value} rank {rank} journal line {step}"),
            arm=arm,
            rank=rank,
            step=step,
        )
        for step, line in enumerate(lines, start=1)
    ]
    return records, {
        "rank": rank,
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "record_count": len(records),
    }


def _validate_checkpoint(
    report_path: Path,
    *,
    arm: LTOPCorePilotArm,
    report: dict[str, Any],
    rank_reports: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    receipt = _require_mapping(report["checkpoint"], name=f"{arm.value} checkpoint receipt")
    _require_exact_keys(receipt, _CHECKPOINT_RECEIPT_KEYS, name=f"{arm.value} checkpoint receipt")
    for rank, rank_report in rank_reports.items():
        if rank_report["checkpoint"] != receipt:
            raise ValueError(f"{arm.value} rank {rank} checkpoint receipt differs from rank zero")
    declared_path = receipt["path"]
    if not isinstance(declared_path, str) or not declared_path:
        raise TypeError(f"{arm.value} checkpoint path must be a non-empty string")
    path = Path(declared_path)
    if not path.is_absolute():
        path = report_path.parent / path
    expected = report_path.parent / "checkpoints" / f"global_step_{LTOP_CORE_PILOT_CHECKPOINT_STEP}"
    if path.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError(f"{arm.value} checkpoint path is outside its registered location")
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError(f"{arm.value} checkpoint directory is absent: {path}")
    manifest_path = path / "ltop_core_pilot_checkpoint.json"
    manifest, actual_sha = _load_json_file(manifest_path, name=f"{arm.value} checkpoint manifest")
    declared_sha = _require_sha256(
        receipt["manifest_sha256"], name=f"{arm.value} checkpoint manifest SHA"
    )
    if actual_sha != declared_sha:
        raise ValueError(f"{arm.value} checkpoint manifest SHA differs from its report")
    _require_exact_keys(
        manifest,
        _CHECKPOINT_MANIFEST_KEYS,
        name=f"{arm.value} checkpoint manifest",
    )
    expected_fields = {
        "schema": LTOP_CORE_PAIR_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": LTOP_CORE_PILOT_CHECKPOINT_STEP,
        "arm": arm.value,
        "g2_report_sha256": report["g2_report_sha256"],
        "g3_report_sha256": report["g3_report_sha256"],
        "stream_plan_sha256": report["stream_plan_sha256"],
    }
    for field, expected_value in expected_fields.items():
        if manifest[field] != expected_value:
            raise ValueError(f"{arm.value} checkpoint manifest violates {field}")
    boundaries = _require_list(manifest["rank_boundaries"], name=f"{arm.value} rank_boundaries")
    if len(boundaries) != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError(f"{arm.value} checkpoint omits one rank boundary")
    seen: set[int] = set()
    for item in boundaries:
        boundary_item = _require_mapping(item, name=f"{arm.value} checkpoint rank boundary")
        _require_exact_keys(
            boundary_item,
            frozenset({"rank", "boundary"}),
            name=f"{arm.value} checkpoint rank boundary",
        )
        rank = _require_int(boundary_item["rank"], name=f"{arm.value} checkpoint boundary rank")
        boundary = _require_mapping(
            boundary_item["boundary"], name=f"{arm.value} checkpoint rank {rank} boundary"
        )
        _require_exact_keys(
            boundary, _CHECKPOINT_BOUNDARY_KEYS, name=f"{arm.value} checkpoint rank {rank} boundary"
        )
        for field in _CHECKPOINT_BOUNDARY_KEYS:
            _require_sha256(boundary[field], name=f"{arm.value} checkpoint rank {rank}.{field}")
        seen.add(rank)
    if seen != set(range(LTOP_CORE_PILOT_WORLD_SIZE)):
        raise ValueError(f"{arm.value} checkpoint rank boundary set is invalid")
    return {"path": str(path.resolve()), "manifest_sha256": actual_sha}


def _validate_metric_artifacts(
    report_path: Path,
    *,
    arm: LTOPCorePilotArm,
    rank_reports: dict[int, dict[str, Any]],
    journals: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    receipts = rank_reports[0]["metric_reports"]
    if len(receipts) != _EXPECTED_METRIC_WINDOWS:
        raise ValueError(f"{arm.value} does not contain exactly 20 metric artifacts")
    for rank in range(1, LTOP_CORE_PILOT_WORLD_SIZE):
        if rank_reports[rank]["metric_reports"] != receipts:
            raise ValueError(f"{arm.value} rank {rank} metric receipts differ from rank zero")

    curve: list[dict[str, float]] = []
    evidence: list[dict[str, Any]] = []
    for index, raw_receipt in enumerate(receipts):
        start_step = index * LTOP_CORE_PILOT_METRICS_EVERY + 1
        end_step = (index + 1) * LTOP_CORE_PILOT_METRICS_EVERY
        receipt = _require_mapping(raw_receipt, name=f"{arm.value} metric receipt {index}")
        _require_exact_keys(
            receipt,
            _METRIC_RECEIPT_KEYS,
            name=f"{arm.value} metric receipt {index}",
        )
        if receipt["start_step"] != start_step or receipt["end_step"] != end_step:
            raise ValueError(f"{arm.value} metric receipt {index} has a non-contiguous window")
        declared_sha = _require_sha256(
            receipt["file_sha256"], name=f"{arm.value} metric receipt {index} SHA"
        )
        expected_path = (
            report_path.parent / "metrics" / f"steps_{start_step:08d}_{end_step:08d}.json"
        )
        path = _resolve_expected_file(
            report_path,
            declared=receipt["path"],
            expected=expected_path,
            name=f"{arm.value} metric artifact {index}",
        )
        artifact, actual_sha = _load_json_file(path, name=f"{arm.value} metric artifact {index}")
        if actual_sha != declared_sha:
            raise ValueError(f"{arm.value} metric artifact {index} SHA differs from its report")
        _require_exact_keys(
            artifact, _METRIC_ARTIFACT_KEYS, name=f"{arm.value} metric artifact {index}"
        )
        expected_fields = {
            "schema": LTOP_CORE_PAIR_METRICS_SCHEMA,
            "arm": arm.value,
            "start_step": start_step,
            "end_step": end_step,
        }
        for field, expected_value in expected_fields.items():
            if artifact[field] != expected_value:
                raise ValueError(f"{arm.value} metric artifact {index} violates {field}")
        means = _require_mapping(
            artifact["means"], name=f"{arm.value} metric artifact {index}.means"
        )
        _require_exact_keys(
            means,
            _METRIC_MEAN_FIELDS,
            name=f"{arm.value} metric artifact {index}.means",
        )
        for field, value in means.items():
            _require_finite_number(
                value,
                name=f"{arm.value} metric artifact {index}.means.{field}",
                nonnegative=field == "action_loss",
            )
        if receipt["means"] != means:
            raise ValueError(f"{arm.value} metric receipt {index} means differ from its artifact")

        rank_windows = _require_list(
            artifact["rank_windows"], name=f"{arm.value} metric artifact {index}.rank_windows"
        )
        if len(rank_windows) != LTOP_CORE_PILOT_WORLD_SIZE:
            raise ValueError(f"{arm.value} metric artifact {index} omits one rank window")
        window_by_rank: dict[int, list[Any]] = {}
        for raw_window in rank_windows:
            window = _require_mapping(
                raw_window, name=f"{arm.value} metric artifact {index} rank window"
            )
            _require_exact_keys(
                window,
                frozenset({"rank", "steps"}),
                name=f"{arm.value} metric artifact {index} rank window",
            )
            rank = _require_int(window["rank"], name=f"{arm.value} metric artifact {index} rank")
            steps = _require_list(
                window["steps"], name=f"{arm.value} metric artifact {index} rank {rank}.steps"
            )
            if rank in window_by_rank:
                raise ValueError(f"{arm.value} metric artifact {index} duplicates rank {rank}")
            window_by_rank[rank] = steps
        if set(window_by_rank) != set(range(LTOP_CORE_PILOT_WORLD_SIZE)):
            raise ValueError(f"{arm.value} metric artifact {index} rank set is invalid")
        for rank in range(LTOP_CORE_PILOT_WORLD_SIZE):
            expected_records = journals[rank][start_step - 1 : end_step]
            if window_by_rank[rank] != expected_records:
                raise ValueError(
                    f"{arm.value} metric artifact {index} rank {rank} differs from its journal"
                )
        expected_sample_count = sum(
            len(record["sample_keys"])
            for rank in range(LTOP_CORE_PILOT_WORLD_SIZE)
            for record in journals[rank][start_step - 1 : end_step]
        )
        if artifact["sample_count"] != expected_sample_count:
            raise ValueError(f"{arm.value} metric artifact {index} sample_count is invalid")
        action_values = [
            float(record["action_loss"])
            for rank in range(LTOP_CORE_PILOT_WORLD_SIZE)
            for record in journals[rank][start_step - 1 : end_step]
        ]
        action_mean = sum(action_values) / len(action_values)
        if not math.isfinite(action_mean) or means["action_loss"] != action_mean:
            raise ValueError(f"{arm.value} metric artifact {index} action mean is not reproducible")
        curve.append({"end_step": end_step, "action_loss": action_mean})
        evidence.append(
            {
                "start_step": start_step,
                "end_step": end_step,
                "path": str(path.resolve()),
                "sha256": actual_sha,
            }
        )
    return curve, evidence


def compose_ltop_core_pair(*, factual_path: Path, blocked_path: Path) -> dict[str, Any]:
    """Verify exact paired execution and retain reproducible action-loss curves."""

    factual, factual_report_sha = _load_report(factual_path, arm=LTOPCorePilotArm.FACTUAL)
    blocked, blocked_report_sha = _load_report(blocked_path, arm=LTOPCorePilotArm.BLOCKED)
    for field in _REPORT_INVARIANTS:
        if factual[field] != blocked[field]:
            raise ValueError(f"LTOP core-pilot pair differs at {field}")

    reports = {
        LTOPCorePilotArm.FACTUAL: factual,
        LTOPCorePilotArm.BLOCKED: blocked,
    }
    report_paths = {
        LTOPCorePilotArm.FACTUAL: factual_path,
        LTOPCorePilotArm.BLOCKED: blocked_path,
    }
    rank_reports = {arm: _rank_map(report) for arm, report in reports.items()}
    optimizer_manifests: dict[LTOPCorePilotArm, dict[int, dict[str, Any]]] = {}
    optimizer_initializations: dict[LTOPCorePilotArm, dict[int, dict[str, Any]]] = {}
    stage_restores: dict[LTOPCorePilotArm, dict[int, dict[str, Any]]] = {}
    journals: dict[LTOPCorePilotArm, dict[int, list[dict[str, Any]]]] = {}
    journal_evidence: dict[LTOPCorePilotArm, list[dict[str, Any]]] = {}
    checkpoint_evidence: dict[LTOPCorePilotArm, dict[str, Any]] = {}

    for arm in (LTOPCorePilotArm.FACTUAL, LTOPCorePilotArm.BLOCKED):
        optimizer_manifests[arm] = {}
        optimizer_initializations[arm] = {}
        stage_restores[arm] = {}
        journals[arm] = {}
        journal_evidence[arm] = []
        for rank in range(LTOP_CORE_PILOT_WORLD_SIZE):
            rank_report = rank_reports[arm][rank]
            manifest = _validate_optimizer_manifest(
                rank_report["optimizer_parameter_manifest"], arm=arm, rank=rank
            )
            stage_restore = _validate_stage_restore(
                rank_report["stage_restore"], arm=arm, rank=rank
            )
            optimizer_initialization = _validate_optimizer_initialization(
                rank_report["optimizer_initialization"],
                arm=arm,
                rank=rank,
                manifest=manifest,
                stage_restore=stage_restore,
            )
            records, evidence = _load_journal(
                report_paths[arm],
                arm=arm,
                rank=rank,
                receipt_value=rank_report["journal"],
            )
            optimizer_manifests[arm][rank] = manifest
            optimizer_initializations[arm][rank] = optimizer_initialization
            stage_restores[arm][rank] = stage_restore
            journals[arm][rank] = records
            journal_evidence[arm].append(evidence)
        if optimizer_manifests[arm][0] != optimizer_manifests[arm][1]:
            raise ValueError(f"{arm.value} optimizer manifests differ between ranks")
        checkpoint_evidence[arm] = _validate_checkpoint(
            report_paths[arm],
            arm=arm,
            report=reports[arm],
            rank_reports=rank_reports[arm],
        )

    for rank in range(LTOP_CORE_PILOT_WORLD_SIZE):
        if (
            optimizer_manifests[LTOPCorePilotArm.FACTUAL][rank]
            != optimizer_manifests[LTOPCorePilotArm.BLOCKED][rank]
        ):
            raise ValueError(f"LTOP core-pilot rank {rank} pair differs at optimizer manifest")
        if (
            stage_restores[LTOPCorePilotArm.FACTUAL][rank]
            != stage_restores[LTOPCorePilotArm.BLOCKED][rank]
        ):
            raise ValueError(f"LTOP core-pilot rank {rank} pair differs at stable stage restore")
        if (
            optimizer_initializations[LTOPCorePilotArm.FACTUAL][rank]
            != optimizer_initializations[LTOPCorePilotArm.BLOCKED][rank]
        ):
            raise ValueError(
                f"LTOP core-pilot rank {rank} pair differs at optimizer initialization"
            )
        for index, (factual_step, blocked_step) in enumerate(
            zip(
                journals[LTOPCorePilotArm.FACTUAL][rank],
                journals[LTOPCorePilotArm.BLOCKED][rank],
                strict=True,
            ),
            start=1,
        ):
            for field in _STEP_MATCHED_FIELDS:
                if factual_step[field] != blocked_step[field]:
                    raise ValueError(f"LTOP core-pilot rank {rank} step {index} differs at {field}")
            if factual_step["forward_input_sha256"] == blocked_step["forward_input_sha256"]:
                raise ValueError(
                    f"LTOP core-pilot rank {rank} step {index} raw forward digest did not change"
                )

    curves: dict[str, list[dict[str, float]]] = {}
    metric_evidence: dict[str, list[dict[str, Any]]] = {}
    for arm in (LTOPCorePilotArm.FACTUAL, LTOPCorePilotArm.BLOCKED):
        curve, evidence = _validate_metric_artifacts(
            report_paths[arm],
            arm=arm,
            rank_reports=rank_reports[arm],
            journals=journals[arm],
        )
        curves["factual" if arm is LTOPCorePilotArm.FACTUAL else "blocked"] = curve
        metric_evidence[arm.value] = evidence

    return {
        "schema": LTOP_CORE_PAIR_SCHEMA,
        "status": "PASS",
        "failures": [],
        "pair_contract": {
            "treatment": LTOPCorePilotArm.FACTUAL.value,
            "control": LTOPCorePilotArm.BLOCKED.value,
            "only_permitted_difference": "typed-object-read-to-action-edge",
        },
        "factual_report": str(factual_path.resolve()),
        "factual_report_sha256": factual_report_sha,
        "blocked_report": str(blocked_path.resolve()),
        "blocked_report_sha256": blocked_report_sha,
        "invariants": {field: factual[field] for field in _REPORT_INVARIANTS},
        "journal_evidence": {
            arm.value: journal_evidence[arm]
            for arm in (LTOPCorePilotArm.FACTUAL, LTOPCorePilotArm.BLOCKED)
        },
        "metric_evidence": metric_evidence,
        "checkpoint_evidence": {
            arm.value: checkpoint_evidence[arm]
            for arm in (LTOPCorePilotArm.FACTUAL, LTOPCorePilotArm.BLOCKED)
        },
        "action_loss_curves": curves,
        "action_loss_factual_minus_blocked": [
            {
                "end_step": factual_item["end_step"],
                "difference": factual_item["action_loss"] - blocked_item["action_loss"],
            }
            for factual_item, blocked_item in zip(curves["factual"], curves["blocked"], strict=True)
        ],
        "scientific_boundary": (
            "PASS proves paired execution integrity only. Mediator value requires the "
            "registered causal evaluator and LBOT-JOINT remains an external calibration."
        ),
    }
