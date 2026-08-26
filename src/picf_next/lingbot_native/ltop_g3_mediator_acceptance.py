"""Fail-closed composition of the complete ADR165 mediator G3 evidence set."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from picf_next.artifact_io import directory_tree_sha256

MEDIATOR_ACCEPTANCE_SCHEMA: Final = "picf-next.ltop-g3-mediator-acceptance.v2"
TRAINING_SCHEMA: Final = "picf-next.ltop-g3-training-phase.v1"
ARM_VALIDATION_SCHEMA: Final = "picf-next.ltop-g3-mediator-trial-arm-validation.v1"
ACTION_EVALUATION_SCHEMA: Final = "picf-next.ltop-g3-evaluation-phase.v1"
RETENTION_SCHEMA: Final = "picf-next.ltop-g3-representation-retention.v1"

EXPECTED_WORLD_SIZE: Final = 2
EXPECTED_TRAINING_STEPS: Final = 256
EXPECTED_EVALUATION_STEPS: Final = 128
EXPECTED_EVAL_EVERY: Final = 32
EXPECTED_COUNT_PER_ARM_PER_RANK: Final = 128
EXPECTED_ARM_WINDOW: Final = 16
MAXIMUM_LAST_TO_FIRST_RATIO: Final = 0.95
ACTION_INFORMATION_SET_POLICY: Final = "fixed-counterbalanced-50-50"
TRAINING_CHECKPOINT_SCHEMA: Final = "picf-next.ltop-g3-training-checkpoint.v2"
MODEL_TREE_SCHEMA: Final = "picf-next.ltop-g3-model-dcp-tree.v1"
MODEL_ONLY_CHECKPOINT_FORMAT: Final = "lingbot-fsdp2-dcp-model-only"

_ARM_VALUES: Final = ("factual", "mediator-required")
_ARM_LABELS: Final = ("FACTUAL", "MEDIATOR_REQUIRED")
_PARTITIONS: Final = ("validation", "heldout")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_FIELDS: Final = (
    "architecture_identity",
    "runtime_source_contract",
    "stage_checkpoint",
    "g2_report_sha256",
    "capacity",
    "task_query_count",
    "execution_contract_sha256",
    "offline_labels_sha256",
    "physical_sidecar_manifest_sha256",
    "dataset_contract",
)


class MediatorAcceptanceContractError(ValueError):
    """Raised when one ADR165 acceptance artifact fails its registered contract."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MediatorAcceptanceContractError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MediatorAcceptanceContractError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MediatorAcceptanceContractError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediatorAcceptanceContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MediatorAcceptanceContractError(f"{name} must be finite")
    return result


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise MediatorAcceptanceContractError(f"{name} must be one lowercase SHA-256")
    return value


def _absolute_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise MediatorAcceptanceContractError(f"{name} must be a non-empty absolute path")
    return value


def _read_report(path: Path, *, name: str) -> tuple[dict[str, Any], str, str]:
    if path.is_symlink() or not path.is_file():
        raise MediatorAcceptanceContractError(f"{name} must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MediatorAcceptanceContractError(f"{name} is not valid UTF-8 JSON") from error
    report = dict(_mapping(value, name=name))
    return report, _sha256_bytes(payload), str(path.resolve())


def _require_pass(report: Mapping[str, Any], *, name: str, schema: str) -> None:
    expected = {"schema": schema, "status": "PASS", "failures": []}
    for field, value in expected.items():
        if report.get(field) != value:
            raise MediatorAcceptanceContractError(f"{name} {field} differs from the PASS contract")


def _require_phase(
    report: Mapping[str, Any],
    *,
    name: str,
    schema: str,
    mode: str,
    phase: str,
    steps: int,
) -> None:
    _require_pass(report, name=name, schema=schema)
    expected = {
        "mode": mode,
        "phase": phase,
        "world_size": EXPECTED_WORLD_SIZE,
        "steps": steps,
        "eval_every": EXPECTED_EVAL_EVERY,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise MediatorAcceptanceContractError(
                f"{name} {field} differs from the registered phase contract"
            )


def _rank_map(report: Mapping[str, Any], *, name: str) -> dict[int, Mapping[str, Any]]:
    values = _sequence(report.get("rank_reports"), name=f"{name}.rank_reports")
    if len(values) != EXPECTED_WORLD_SIZE:
        raise MediatorAcceptanceContractError(f"{name} must contain exactly two rank reports")
    result: dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        item = _mapping(value, name=f"{name}.rank_reports[{index}]")
        rank = _integer(item.get("rank"), name=f"{name}.rank_reports[{index}].rank")
        if rank not in (0, 1) or rank in result:
            raise MediatorAcceptanceContractError(f"{name} rank set is invalid")
        result[rank] = item
    if set(result) != {0, 1}:
        raise MediatorAcceptanceContractError(f"{name} rank set is incomplete")
    return result


def _validate_identity(
    training: Mapping[str, Any],
    action: Mapping[str, Any],
    retention: Mapping[str, Any],
) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for field in _IDENTITY_FIELDS:
        value = training.get(field)
        if field in {
            "g2_report_sha256",
            "execution_contract_sha256",
            "offline_labels_sha256",
            "physical_sidecar_manifest_sha256",
        }:
            _sha256(value, name=f"training.{field}")
        elif field == "stage_checkpoint":
            _absolute_path(value, name="training.stage_checkpoint")
        elif field in {"capacity", "task_query_count"}:
            _integer(value, name=f"training.{field}", minimum=1)
        elif field == "architecture_identity":
            if not isinstance(value, str) or not value:
                raise MediatorAcceptanceContractError(
                    "training.architecture_identity must be non-empty"
                )
        else:
            if not isinstance(value, Mapping) or not value:
                raise MediatorAcceptanceContractError(
                    f"training.{field} must be a non-empty object"
                )
        for report_name, report in (("action", action), ("retention", retention)):
            if report.get(field) != value:
                raise MediatorAcceptanceContractError(
                    f"{report_name}.{field} differs from training.{field}"
                )
        identity[field] = value
    return identity


def _validate_training_schedule(training: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    contract = _mapping(training.get("training_contract"), name="training.training_contract")
    trial = _mapping(
        contract.get("action_information_set_trial"),
        name="training.training_contract.action_information_set_trial",
    )
    if trial.get("single_forward_per_optimizer_step") is not True:
        raise MediatorAcceptanceContractError(
            "training mediator trial must use one forward per optimizer step"
        )
    schedule = _mapping(
        trial.get("schedule"),
        name="training.training_contract.action_information_set_trial.schedule",
    )
    schedule_sha256 = _sha256(schedule.get("sha256"), name="training schedule sha256")
    unsealed = dict(schedule)
    del unsealed["sha256"]
    if _sha256_bytes(_canonical_json(unsealed).encode("ascii")) != schedule_sha256:
        raise MediatorAcceptanceContractError("training schedule sealed SHA-256 is invalid")
    expected = {
        "design": "scene-prompt-stratified-two-period-crossover",
        "single_forward_per_optimizer_step": True,
        "steps": EXPECTED_TRAINING_STEPS,
        "scene_count": 8,
        "prompts_per_scene": 2,
        "cycle_steps": 16,
        "arm_counts": {
            "factual": EXPECTED_COUNT_PER_ARM_PER_RANK,
            "mediator-required": EXPECTED_COUNT_PER_ARM_PER_RANK,
        },
    }
    for field, value in expected.items():
        if schedule.get(field) != value:
            raise MediatorAcceptanceContractError(
                f"training schedule {field} differs from the fixed counterbalance contract"
            )
    entries = _sequence(schedule.get("entries"), name="training schedule entries")
    if len(entries) != EXPECTED_TRAINING_STEPS:
        raise MediatorAcceptanceContractError("training schedule entry count is not 256")
    entry_arms: list[str] = []
    for index, value in enumerate(entries, start=1):
        entry = _mapping(value, name=f"training schedule entry {index}")
        if entry.get("global_step") != index or entry.get("arm") not in _ARM_VALUES:
            raise MediatorAcceptanceContractError(
                f"training schedule entry {index} is not a valid fixed assignment"
            )
        entry_arms.append(cast(str, entry["arm"]))
    if {arm: entry_arms.count(arm) for arm in _ARM_VALUES} != expected["arm_counts"]:
        raise MediatorAcceptanceContractError("training schedule entries are not 128/128")
    cell_counts = _mapping(
        schedule.get("cell_arm_counts"),
        name="training schedule cell_arm_counts",
    )
    if len(cell_counts) != 16 or any(
        counts != {"factual": 8, "mediator-required": 8} for counts in cell_counts.values()
    ):
        raise MediatorAcceptanceContractError(
            "training schedule is not balanced within every scene-prompt cell"
        )
    return schedule_sha256, contract


def _validate_training(
    training: Mapping[str, Any],
) -> tuple[
    str,
    Mapping[str, Any],
    dict[int, Mapping[str, Any]],
    str,
    list[str],
    str,
]:
    _require_phase(
        training,
        name="training report",
        schema=TRAINING_SCHEMA,
        mode="mediator-trial",
        phase="training",
        steps=EXPECTED_TRAINING_STEPS,
    )
    schedule_sha256, _ = _validate_training_schedule(training)
    checkpoint = _mapping(training.get("checkpoint"), name="training.checkpoint")
    checkpoint_path = _absolute_path(checkpoint.get("path"), name="training.checkpoint.path")
    if checkpoint.get("optimizer_saved") is not False:
        raise MediatorAcceptanceContractError(
            "training checkpoint must be the registered model-only receipt"
        )
    if checkpoint.get("format") != MODEL_ONLY_CHECKPOINT_FORMAT:
        raise MediatorAcceptanceContractError("training checkpoint format differs")
    checkpoint_directory = Path(checkpoint_path)
    if checkpoint_directory.is_symlink() or not checkpoint_directory.is_dir():
        raise MediatorAcceptanceContractError("training checkpoint must be one real directory")
    root_entries = {path.name for path in checkpoint_directory.iterdir()}
    if root_entries != {"model", "ltop_g3_training_checkpoint.json"}:
        raise MediatorAcceptanceContractError(
            "training checkpoint root differs from model-only ABI"
        )
    model_directory = checkpoint_directory / "model"
    if model_directory.is_symlink() or not model_directory.is_dir():
        raise MediatorAcceptanceContractError("training checkpoint model directory is absent")
    model_names = {path.name for path in model_directory.iterdir() if path.is_file()}
    if ".metadata" not in model_names or not any(name.endswith(".distcp") for name in model_names):
        raise MediatorAcceptanceContractError("training checkpoint omits DCP model payloads")
    manifest_path = checkpoint_directory / "ltop_g3_training_checkpoint.json"
    manifest, manifest_sha256, _ = _read_report(
        manifest_path,
        name="training checkpoint manifest",
    )
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise MediatorAcceptanceContractError("training checkpoint manifest SHA-256 differs")
    if checkpoint.get("model_tree_schema") != MODEL_TREE_SCHEMA:
        raise MediatorAcceptanceContractError("training checkpoint model-tree schema differs")
    try:
        model_tree_sha256 = directory_tree_sha256(
            model_directory,
            schema=MODEL_TREE_SCHEMA,
        )
    except ValueError as error:
        raise MediatorAcceptanceContractError(
            "training checkpoint model tree is invalid"
        ) from error
    if checkpoint.get("model_tree_sha256") != model_tree_sha256:
        raise MediatorAcceptanceContractError("training checkpoint model tree SHA-256 differs")
    manifest_expected = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": EXPECTED_TRAINING_STEPS,
        "optimizer_saved": False,
        "format": MODEL_ONLY_CHECKPOINT_FORMAT,
        "world_size": EXPECTED_WORLD_SIZE,
        "model_tree_schema": MODEL_TREE_SCHEMA,
        "model_tree_sha256": model_tree_sha256,
        "action_information_set_schedule_sha256": schedule_sha256,
        "source_stage_checkpoint": training.get("stage_checkpoint"),
        "g2_report_sha256": training.get("g2_report_sha256"),
        "runtime_source_contract": training.get("runtime_source_contract"),
    }
    for field, value in manifest_expected.items():
        if manifest.get(field) != value:
            raise MediatorAcceptanceContractError(f"training checkpoint manifest {field} differs")

    ranks = _rank_map(training, name="training report")
    expected_counts = {
        "factual": EXPECTED_COUNT_PER_ARM_PER_RANK,
        "mediator-required": EXPECTED_COUNT_PER_ARM_PER_RANK,
    }
    for rank, report in ranks.items():
        if report.get("action_information_set_counts") != expected_counts:
            raise MediatorAcceptanceContractError(f"training rank {rank} is not 128/128")
        if report.get("action_information_set_schedule_sha256") != schedule_sha256:
            raise MediatorAcceptanceContractError(f"training rank {rank} schedule SHA-256 differs")
        if report.get("all_gradients_finite") is not True:
            raise MediatorAcceptanceContractError(
                f"training rank {rank} reports non-finite gradients"
            )
        action_losses = _sequence(
            report.get("action_losses"),
            name=f"training rank {rank} action_losses",
        )
        if len(action_losses) != EXPECTED_TRAINING_STEPS:
            raise MediatorAcceptanceContractError(
                f"training rank {rank} action loss trace is incomplete"
            )
        for index, value in enumerate(action_losses):
            _finite_number(value, name=f"training rank {rank} action_losses[{index}]")
        journal = _mapping(report.get("arm_journal"), name=f"training rank {rank} journal")
        if journal.get("rank") != rank or journal.get("record_count") != EXPECTED_TRAINING_STEPS:
            raise MediatorAcceptanceContractError(
                f"training rank {rank} journal receipt is incomplete"
            )
        _absolute_path(journal.get("path"), name=f"training rank {rank} journal path")
        _sha256(journal.get("file_sha256"), name=f"training rank {rank} journal sha256")
    training_model_digests = [
        _sha256(
            ranks[rank].get("training_final_model_local_state_sha256"),
            name=f"training rank {rank} terminal model local state sha256",
        )
        for rank in (0, 1)
    ]
    for rank, training_digest in enumerate(training_model_digests):
        post_save_digest = _sha256(
            ranks[rank].get("post_checkpoint_save_model_local_state_sha256"),
            name=f"training rank {rank} post-checkpoint-save model local state sha256",
        )
        if post_save_digest != training_digest:
            raise MediatorAcceptanceContractError(
                f"training rank {rank} model state changed during checkpoint save"
            )
    if checkpoint.get("training_final_model_local_state_sha256_by_rank") != (
        training_model_digests
    ):
        raise MediatorAcceptanceContractError(
            "training checkpoint receipt terminal model digests differ"
        )
    if manifest.get("training_final_model_local_state_sha256_by_rank") != (training_model_digests):
        raise MediatorAcceptanceContractError(
            "training checkpoint manifest terminal model digests differ"
        )
    expected_counts_by_rank = [
        dict(ranks[rank]["action_information_set_counts"]) for rank in (0, 1)
    ]
    if manifest.get("action_information_set_counts_by_rank") != expected_counts_by_rank:
        raise MediatorAcceptanceContractError("training checkpoint manifest arm counts differ")
    return (
        checkpoint_path,
        checkpoint,
        ranks,
        schedule_sha256,
        training_model_digests,
        model_tree_sha256,
    )


def _cold_model_evidence(
    report: Mapping[str, Any],
    *,
    name: str,
    expected_model_digests: Sequence[str],
    expected_model_tree_sha256: str,
    expected_runtime_schedule_sha256_by_rank: Sequence[str],
) -> list[str]:
    """Validate one independent cold load before and after all forwards."""

    ranks = _rank_map(report, name=name)
    values: list[str] = []
    for rank in (0, 1):
        cold_loaded = _sha256(
            ranks[rank].get("cold_loaded_model_local_state_sha256"),
            name=f"{name} rank {rank} cold-loaded model local state sha256",
        )
        post_evaluation = _sha256(
            ranks[rank].get("post_evaluation_model_local_state_sha256"),
            name=f"{name} rank {rank} post-evaluation model local state sha256",
        )
        if cold_loaded != post_evaluation:
            raise MediatorAcceptanceContractError(
                f"{name} rank {rank} mutated persistent model state"
            )
        if cold_loaded != expected_model_digests[rank]:
            raise MediatorAcceptanceContractError(
                f"{name} rank {rank} differs from the training terminal model state"
            )
        if (
            ranks[rank].get("trained_model_local_state_sha256") is not None
            and ranks[rank].get("trained_model_local_state_sha256") != cold_loaded
        ):
            raise MediatorAcceptanceContractError(
                f"{name} rank {rank} legacy model digest alias differs"
            )
        if (
            _sha256(
                ranks[rank].get("trained_checkpoint_model_tree_sha256"),
                name=f"{name} rank {rank} checkpoint model tree sha256",
            )
            != expected_model_tree_sha256
        ):
            raise MediatorAcceptanceContractError(
                f"{name} rank {rank} consumed another checkpoint tree"
            )
        if (
            _sha256(
                ranks[rank].get("runtime_schedule_sha256"),
                name=f"{name} rank {rank} runtime schedule sha256",
            )
            != expected_runtime_schedule_sha256_by_rank[rank]
        ):
            raise MediatorAcceptanceContractError(
                f"{name} rank {rank} runtime schedule differs from training"
            )
        values.append(cold_loaded)
    return values


def _validate_arm_window(summary: Mapping[str, Any], *, name: str, count: int) -> None:
    expected = {
        "count": count,
        "expected_count": count,
        "balanced_count_pass": True,
        "action_loss_finite": True,
        "all_reported_losses_finite": True,
        "window_gate_pass": True,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise MediatorAcceptanceContractError(f"{name}.{field} differs from contract")
    if summary.get("maximum_last_to_first_ratio") != MAXIMUM_LAST_TO_FIRST_RATIO:
        raise MediatorAcceptanceContractError(f"{name} uses a different improvement threshold")
    ratio = _finite_number(summary.get("last_to_first_ratio"), name=f"{name}.ratio")
    improvement = _finite_number(
        summary.get("relative_improvement"),
        name=f"{name}.relative_improvement",
    )
    if not 0.0 <= ratio <= MAXIMUM_LAST_TO_FIRST_RATIO or not 0.05 <= improvement <= 1.0:
        raise MediatorAcceptanceContractError(f"{name} did not improve by at least five percent")
    if not math.isclose(improvement, 1.0 - ratio, rel_tol=0.0, abs_tol=1.0e-12):
        raise MediatorAcceptanceContractError(f"{name} ratio and improvement disagree")
    for window_name in ("first_window", "last_window"):
        window = _mapping(summary.get(window_name), name=f"{name}.{window_name}")
        expected_window_count = EXPECTED_ARM_WINDOW * (2 if count == 256 else 1)
        if window.get("count") != expected_window_count or window.get("finite") is not True:
            raise MediatorAcceptanceContractError(f"{name}.{window_name} is incomplete")
        mean_action_loss = _finite_number(
            window.get("mean_action_loss"),
            name=f"{name}.{window_name}.mean_action_loss",
        )
        if mean_action_loss < 0.0 or (window_name == "first_window" and mean_action_loss == 0.0):
            raise MediatorAcceptanceContractError(
                f"{name}.{window_name}.mean_action_loss is outside the loss domain"
            )


def _validate_arm_evidence(
    arm: Mapping[str, Any],
    *,
    training_path: str,
    training_sha256: str,
    training_ranks: Mapping[int, Mapping[str, Any]],
    schedule_sha256: str,
) -> dict[str, Any]:
    _require_pass(arm, name="arm validation", schema=ARM_VALIDATION_SCHEMA)
    thresholds = _mapping(arm.get("thresholds"), name="arm validation thresholds")
    expected_thresholds = {
        "expected_count_per_arm_per_rank": EXPECTED_COUNT_PER_ARM_PER_RANK,
        "expected_world_size": EXPECTED_WORLD_SIZE,
        "maximum_last_to_first_ratio": MAXIMUM_LAST_TO_FIRST_RATIO,
        "window_size": EXPECTED_ARM_WINDOW,
    }
    if thresholds != expected_thresholds:
        raise MediatorAcceptanceContractError("arm validation thresholds differ from contract")

    final_report = _mapping(arm.get("final_report"), name="arm validation final_report")
    expected_final = {
        "consistent": True,
        "file_sha256": training_sha256,
        "path": training_path,
        "runner_failures": [],
        "runner_status": "PASS",
        "schema": TRAINING_SCHEMA,
    }
    for field, value in expected_final.items():
        if final_report.get(field) != value:
            raise MediatorAcceptanceContractError(
                f"arm validation final_report.{field} is not bound to training"
            )
    inputs = _mapping(arm.get("inputs"), name="arm validation inputs")
    if inputs.get("report") != training_path:
        raise MediatorAcceptanceContractError(
            "arm validation input report path is not the training report"
        )
    journal_dir = _absolute_path(inputs.get("journal_dir"), name="arm validation journal_dir")

    global_summary = _mapping(arm.get("global"), name="arm validation global")
    expected_global = {
        "balanced_arms_pass": True,
        "finite_pass": True,
        "record_count": EXPECTED_TRAINING_STEPS * EXPECTED_WORLD_SIZE,
        "window_gates_pass": True,
    }
    for field, value in expected_global.items():
        if global_summary.get(field) != value:
            raise MediatorAcceptanceContractError(
                f"arm validation global.{field} differs from contract"
            )
    schedule = _mapping(global_summary.get("schedule"), name="arm validation schedule")
    if (
        schedule.get("consistent") is not True
        or schedule.get("entries_consistent_across_ranks") is not True
        or schedule.get("sha256") != schedule_sha256
        or schedule.get("rank_digests") != [schedule_sha256, schedule_sha256]
    ):
        raise MediatorAcceptanceContractError(
            "arm validation global schedule is not bound to training"
        )
    global_arms = _mapping(global_summary.get("arms"), name="arm validation global arms")
    if set(global_arms) != set(_ARM_LABELS):
        raise MediatorAcceptanceContractError("arm validation global arm set is invalid")
    for label in _ARM_LABELS:
        _validate_arm_window(
            _mapping(global_arms[label], name=f"arm validation global {label}"),
            name=f"arm validation global {label}",
            count=EXPECTED_COUNT_PER_ARM_PER_RANK * EXPECTED_WORLD_SIZE,
        )

    rank_values = _sequence(arm.get("ranks"), name="arm validation ranks")
    if len(rank_values) != EXPECTED_WORLD_SIZE:
        raise MediatorAcceptanceContractError("arm validation must contain two rank summaries")
    journal_receipts: list[dict[str, Any]] = []
    for expected_rank, value in enumerate(rank_values):
        summary = _mapping(value, name=f"arm validation rank {expected_rank}")
        if summary.get("rank") != expected_rank:
            raise MediatorAcceptanceContractError("arm validation rank order is invalid")
        for field in ("balanced_arms_pass", "finite_pass", "window_gates_pass"):
            if summary.get(field) is not True:
                raise MediatorAcceptanceContractError(
                    f"arm validation rank {expected_rank}.{field} is not true"
                )
        rank_schedule = _mapping(
            summary.get("schedule"),
            name=f"arm validation rank {expected_rank} schedule",
        )
        if (
            rank_schedule.get("consistent") is not True
            or rank_schedule.get("sha256") != schedule_sha256
            or rank_schedule.get("digests") != [schedule_sha256]
        ):
            raise MediatorAcceptanceContractError(
                f"arm validation rank {expected_rank} schedule differs"
            )
        validator_journal = _mapping(
            summary.get("journal"),
            name=f"arm validation rank {expected_rank} journal",
        )
        training_journal = _mapping(
            training_ranks[expected_rank].get("arm_journal"),
            name=f"training rank {expected_rank} journal",
        )
        for field in ("path", "file_sha256", "record_count"):
            if validator_journal.get(field) != training_journal.get(field):
                raise MediatorAcceptanceContractError(
                    f"arm validation rank {expected_rank} journal {field} differs from training"
                )
        journal_path = _absolute_path(
            validator_journal.get("path"),
            name=f"arm validation rank {expected_rank} journal path",
        )
        if str(Path(journal_path).parent) != journal_dir:
            raise MediatorAcceptanceContractError(
                f"arm validation rank {expected_rank} journal is outside journal_dir"
            )
        journal_sha256 = _sha256(
            validator_journal.get("file_sha256"),
            name=f"arm validation rank {expected_rank} journal sha256",
        )
        if validator_journal.get("record_count") != EXPECTED_TRAINING_STEPS:
            raise MediatorAcceptanceContractError(
                f"arm validation rank {expected_rank} journal is incomplete"
            )
        rank_arms = _mapping(
            summary.get("arms"),
            name=f"arm validation rank {expected_rank} arms",
        )
        if set(rank_arms) != set(_ARM_LABELS):
            raise MediatorAcceptanceContractError(
                f"arm validation rank {expected_rank} arm set is invalid"
            )
        for label in _ARM_LABELS:
            _validate_arm_window(
                _mapping(
                    rank_arms[label],
                    name=f"arm validation rank {expected_rank} {label}",
                ),
                name=f"arm validation rank {expected_rank} {label}",
                count=EXPECTED_COUNT_PER_ARM_PER_RANK,
            )
        journal_receipts.append(
            {
                "rank": expected_rank,
                "path": journal_path,
                "sha256": journal_sha256,
                "record_count": EXPECTED_TRAINING_STEPS,
            }
        )
    return {
        "journal_dir": journal_dir,
        "journal_receipts": journal_receipts,
        "schedule_sha256": schedule_sha256,
        "thresholds": dict(thresholds),
        "global_arms": {label: dict(global_arms[label]) for label in _ARM_LABELS},
    }


def _validate_action_summary(action: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = _mapping(action.get("thresholds"), name="action thresholds")
    if thresholds.get("bitwise_factual_replay") is not True:
        raise MediatorAcceptanceContractError("action factual replay threshold is absent")
    if thresholds.get("mean_factual_target_minus_distractor_strictly_positive") is not True:
        raise MediatorAcceptanceContractError("action factual effect threshold is absent")
    if thresholds.get("mean_blocked_path_did_strictly_positive") is not True:
        raise MediatorAcceptanceContractError("action blocked DID threshold is absent")
    positive_fraction = _finite_number(
        thresholds.get("positive_sample_fraction_minimum"),
        name="action positive sample fraction",
    )
    if positive_fraction != 0.625:
        raise MediatorAcceptanceContractError("action positive sample fraction must be 0.625")
    inference_contract = _mapping(
        action.get("action_inference_contract"),
        name="action inference contract",
    )
    if inference_contract.get("surface") != "policy.sample_actions":
        raise MediatorAcceptanceContractError("action did not use policy.sample_actions")
    if inference_contract.get("fixed_noise") is not True:
        raise MediatorAcceptanceContractError("action evaluation did not use fixed noise")

    ranks = _rank_map(action, name="action report")
    partition_summaries: dict[str, Any] = {}
    for partition in _PARTITIONS:
        scores: list[Mapping[str, Any]] = []
        for rank, report in ranks.items():
            history = _sequence(report.get("history"), name=f"action rank {rank} history")
            if len(history) != 1:
                raise MediatorAcceptanceContractError(
                    f"action rank {rank} must contain one cold evaluation receipt"
                )
            receipt = _mapping(history[0], name=f"action rank {rank} receipt")
            partition_report = _mapping(
                receipt.get(partition),
                name=f"action rank {rank} {partition}",
            )
            if (
                _finite_number(
                    partition_report.get("max_replay_floor_rms"),
                    name=f"action rank {rank} {partition} replay floor",
                )
                != 0.0
            ):
                raise MediatorAcceptanceContractError(
                    f"action rank {rank} {partition} factual replay is not bitwise stable"
                )
            scenes = _sequence(
                partition_report.get("scenes"),
                name=f"action rank {rank} {partition} scenes",
            )
            if not scenes:
                raise MediatorAcceptanceContractError(
                    f"action rank {rank} {partition} contains no scenes"
                )
            scores.extend(
                _mapping(
                    _mapping(scene, name=f"action rank {rank} {partition} scene").get("score"),
                    name=f"action rank {rank} {partition} scene score",
                )
                for scene in scenes
            )
        factual = [
            _finite_number(
                score.get("mean_factual_target_minus_distractor"),
                name=f"action {partition} factual effect",
            )
            for score in scores
        ]
        did = [
            _finite_number(
                score.get("mean_blocked_path_difference_in_differences"),
                name=f"action {partition} blocked DID",
            )
            for score in scores
        ]
        sample_count = 0
        positive_factual = 0
        positive_did = 0
        for score in scores:
            sample_keys = _sequence(
                score.get("sample_keys"),
                name=f"action {partition} score sample_keys",
            )
            invalid_sample_key = any(
                not isinstance(value, str) or not value for value in sample_keys
            )
            if not sample_keys or invalid_sample_key:
                raise MediatorAcceptanceContractError(f"action {partition} sample keys are invalid")
            sample_count += len(sample_keys)
            positive_factual += _integer(
                score.get("positive_factual_count"),
                name=f"action {partition} positive factual count",
            )
            positive_did += _integer(
                score.get("positive_blocked_path_did_count"),
                name=f"action {partition} positive DID count",
            )
        factual_mean = sum(factual) / len(factual)
        did_mean = sum(did) / len(did)
        minimum_positive = math.ceil(positive_fraction * sample_count)
        if factual_mean <= 0.0 or did_mean <= 0.0:
            raise MediatorAcceptanceContractError(
                f"action {partition} aggregate target mediation is nonpositive"
            )
        if positive_factual < minimum_positive or positive_did < minimum_positive:
            raise MediatorAcceptanceContractError(
                f"action {partition} positive count is below the registered threshold"
            )
        partition_summaries[partition] = {
            "scene_count": len(scores),
            "sample_count": sample_count,
            "mean_factual_target_minus_distractor": factual_mean,
            "mean_blocked_path_difference_in_differences": did_mean,
            "minimum_positive_count": minimum_positive,
            "positive_factual_count": positive_factual,
            "positive_blocked_path_did_count": positive_did,
        }
    return {
        "action_inference_contract": dict(inference_contract),
        "thresholds": dict(thresholds),
        "partitions": partition_summaries,
    }


def _validate_retention_summary(retention: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping(
        retention.get("representation_retention_contract"),
        name="retention representation contract",
    )
    if contract.get("optimizer_updates") != 0:
        raise MediatorAcceptanceContractError("retention evaluation performed optimizer updates")
    thresholds = _mapping(retention.get("thresholds"), name="retention thresholds")
    ranks = _rank_map(retention, name="retention report")
    partition_summaries: dict[str, Any] = {}
    for partition in _PARTITIONS:
        reports: list[Mapping[str, Any]] = []
        for rank, report in ranks.items():
            history = _sequence(report.get("history"), name=f"retention rank {rank} history")
            if len(history) != 1:
                raise MediatorAcceptanceContractError(
                    f"retention rank {rank} must contain one cold receipt"
                )
            receipt = _mapping(history[0], name=f"retention rank {rank} receipt")
            value = _mapping(
                receipt.get(partition),
                name=f"retention rank {rank} {partition}",
            )
            if value.get("scene_count") != 4 or value.get("prompt_count") != 8:
                raise MediatorAcceptanceContractError(
                    f"retention rank {rank} {partition} scene/prompt axis is incomplete"
                )
            if value.get("shared_row_gauge") is not True:
                raise MediatorAcceptanceContractError(
                    f"retention rank {rank} {partition} row gauge changed"
                )
            if (
                _finite_number(
                    value.get("physical_prompt_drift_max_abs"),
                    name=f"retention rank {rank} {partition} physical drift",
                )
                > 1.0e-5
            ):
                raise MediatorAcceptanceContractError(
                    f"retention rank {rank} {partition} physical rows drifted"
                )
            self_checks = _mapping(
                value.get("metric_self_checks"),
                name=f"retention rank {rank} {partition} self checks",
            )
            if (
                _finite_number(
                    self_checks.get("matched_row_permutation_max_abs_error"),
                    name=f"retention rank {rank} {partition} permutation error",
                )
                > 1.0e-6
            ):
                raise MediatorAcceptanceContractError(
                    f"retention rank {rank} {partition} permutation check failed"
                )
            reports.append(value)
        prompt_count = sum(
            _integer(value.get("prompt_count"), name=f"retention {partition} prompt_count")
            for value in reports
        )
        positive_count = sum(
            _integer(
                value.get("positive_margin_count"),
                name=f"retention {partition} positive_margin_count",
            )
            for value in reports
        )
        mean_margin = sum(
            _finite_number(value.get("mean_margin"), name=f"retention {partition} mean_margin")
            for value in reports
        ) / len(reports)
        minimum_positive = 12 if partition == "validation" else 10
        minimum_margin = 0.02 if partition == "validation" else 0.0
        if positive_count < minimum_positive:
            raise MediatorAcceptanceContractError(
                f"retention {partition} positive margin count is below threshold"
            )
        if mean_margin < minimum_margin or (
            partition == "heldout" and mean_margin <= minimum_margin
        ):
            raise MediatorAcceptanceContractError(
                f"retention {partition} mean margin is below threshold"
            )
        partition_summaries[partition] = {
            "scene_count": sum(cast(int, value["scene_count"]) for value in reports),
            "prompt_count": prompt_count,
            "positive_margin_count": positive_count,
            "minimum_positive_margin_count": minimum_positive,
            "mean_margin": mean_margin,
            "minimum_mean_margin": minimum_margin,
        }
    robustness = _mapping(
        retention.get("scene_level_robustness"),
        name="retention scene_level_robustness",
    )
    if set(robustness) != set(_PARTITIONS):
        raise MediatorAcceptanceContractError("retention scene robustness is incomplete")
    return {
        "representation_retention_contract": dict(contract),
        "thresholds": dict(thresholds),
        "partitions": partition_summaries,
        "scene_level_robustness": dict(robustness),
    }


def compose_ltop_g3_mediator_acceptance(
    *,
    training_path: Path,
    arm_validation_path: Path,
    action_evaluation_path: Path,
    retention_path: Path,
) -> dict[str, Any]:
    """Compose four process-isolated PASS artifacts into one immutable G3 ABI."""

    raw_paths = [training_path, arm_validation_path, action_evaluation_path, retention_path]
    if len({str(path.absolute()) for path in raw_paths}) != len(raw_paths):
        raise MediatorAcceptanceContractError("acceptance inputs must be four distinct files")
    training, training_sha256, training_absolute = _read_report(
        training_path,
        name="training report",
    )
    arm, arm_sha256, arm_absolute = _read_report(
        arm_validation_path,
        name="arm validation",
    )
    action, action_sha256, action_absolute = _read_report(
        action_evaluation_path,
        name="cold action evaluation",
    )
    retention, retention_sha256, retention_absolute = _read_report(
        retention_path,
        name="cold retention",
    )

    (
        checkpoint_path,
        checkpoint,
        training_ranks,
        schedule_sha256,
        training_model_digests,
        model_tree_sha256,
    ) = _validate_training(training)
    arm_summary = _validate_arm_evidence(
        arm,
        training_path=training_absolute,
        training_sha256=training_sha256,
        training_ranks=training_ranks,
        schedule_sha256=schedule_sha256,
    )

    _require_phase(
        action,
        name="cold action evaluation",
        schema=ACTION_EVALUATION_SCHEMA,
        mode="gate",
        phase="evaluation",
        steps=EXPECTED_EVALUATION_STEPS,
    )
    _require_phase(
        retention,
        name="cold retention",
        schema=RETENTION_SCHEMA,
        mode="gate",
        phase="retention",
        steps=EXPECTED_EVALUATION_STEPS,
    )
    if action.get("checkpoint") is not None or retention.get("checkpoint") is not None:
        raise MediatorAcceptanceContractError(
            "cold action and retention reports must not publish a new checkpoint"
        )
    for name, report in (("action", action), ("retention", retention)):
        if report.get("trained_checkpoint") != checkpoint_path:
            raise MediatorAcceptanceContractError(
                f"{name}.trained_checkpoint differs from training checkpoint path"
            )
    training_seed = _integer(training.get("seed"), name="training.seed")
    for name, report in (("action", action), ("retention", retention)):
        if report.get("seed") != training_seed:
            raise MediatorAcceptanceContractError(f"{name}.seed differs from training.seed")
    runtime_schedules = [
        _sha256(
            training_ranks[rank].get("runtime_schedule_sha256"),
            name=f"training rank {rank} runtime schedule sha256",
        )
        for rank in (0, 1)
    ]
    action_model_digests = _cold_model_evidence(
        action,
        name="cold action evaluation",
        expected_model_digests=training_model_digests,
        expected_model_tree_sha256=model_tree_sha256,
        expected_runtime_schedule_sha256_by_rank=runtime_schedules,
    )
    retention_model_digests = _cold_model_evidence(
        retention,
        name="cold retention",
        expected_model_digests=training_model_digests,
        expected_model_tree_sha256=model_tree_sha256,
        expected_runtime_schedule_sha256_by_rank=runtime_schedules,
    )
    if action_model_digests != retention_model_digests:
        raise MediatorAcceptanceContractError(
            "cold action and retention restored different model states"
        )
    identity = _validate_identity(training, action, retention)
    action_summary = _validate_action_summary(action)
    retention_summary = _validate_retention_summary(retention)

    return {
        "schema": MEDIATOR_ACCEPTANCE_SCHEMA,
        "status": "PASS",
        "failures": [],
        "world_size": EXPECTED_WORLD_SIZE,
        **identity,
        "checkpoint": dict(checkpoint),
        "training_final_model_local_state_sha256_by_rank": training_model_digests,
        "cold_load_model_local_state_sha256_by_rank": {
            "action_evaluation": action_model_digests,
            "retention": retention_model_digests,
        },
        "checkpoint_identity": {
            "manifest_sha256": checkpoint["manifest_sha256"],
            "model_tree_schema": MODEL_TREE_SCHEMA,
            "model_tree_sha256": model_tree_sha256,
        },
        "training_contract": {
            "mode": "mediator-trial",
            "steps": EXPECTED_TRAINING_STEPS,
            "eval_every": EXPECTED_EVAL_EVERY,
            "action_information_set_policy": ACTION_INFORMATION_SET_POLICY,
            "schedule_sha256": schedule_sha256,
            "seed": training_seed,
            "runtime_schedule_sha256_by_rank": runtime_schedules,
        },
        "evidence": {
            "training_report": {
                "path": training_absolute,
                "sha256": training_sha256,
            },
            "arm_validation": {
                "path": arm_absolute,
                "sha256": arm_sha256,
            },
            "cold_action_evaluation": {
                "path": action_absolute,
                "sha256": action_sha256,
            },
            "cold_retention": {
                "path": retention_absolute,
                "sha256": retention_sha256,
            },
        },
        "arm_validation_summary": arm_summary,
        "action_summary": action_summary,
        "retention_summary": retention_summary,
    }
