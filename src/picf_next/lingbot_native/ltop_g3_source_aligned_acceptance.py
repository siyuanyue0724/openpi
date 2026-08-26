"""Fail-closed composition of the complete ADR170 source-aligned G3 evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from picf_next.artifact_io import directory_tree_sha256
from picf_next.lingbot_native.ltop_g3_mediator_acceptance import (
    ACTION_EVALUATION_SCHEMA,
    EXPECTED_ARM_WINDOW,
    EXPECTED_COUNT_PER_ARM_PER_RANK,
    EXPECTED_EVAL_EVERY,
    EXPECTED_EVALUATION_STEPS,
    EXPECTED_TRAINING_STEPS,
    EXPECTED_WORLD_SIZE,
    MAXIMUM_LAST_TO_FIRST_RATIO,
    MODEL_ONLY_CHECKPOINT_FORMAT,
    MODEL_TREE_SCHEMA,
    RETENTION_SCHEMA,
    TRAINING_SCHEMA,
    MediatorAcceptanceContractError,
    _absolute_path,
    _canonical_json,
    _cold_model_evidence,
    _finite_number,
    _mapping,
    _rank_map,
    _read_report,
    _require_phase,
    _sequence,
    _sha256,
    _sha256_bytes,
    _validate_action_summary,
    _validate_arm_evidence,
    _validate_identity,
    _validate_retention_summary,
)
from picf_next.lingbot_native.task_address_learning import (
    action_consumable_task_address_depth_contract,
)

SOURCE_ALIGNED_ACCEPTANCE_SCHEMA: Final = "picf-next.ltop-g3-source-aligned-acceptance.v1"
SOURCE_ACTION_SCHEDULE_SCHEMA: Final = "picf-next.ltop-g3-source-action-counterbalance.v1"
TRAINING_CHECKPOINT_SCHEMA: Final = "picf-next.ltop-g3-training-checkpoint.v5"
ACTION_SUPERVISION_SCHEMA: Final = "picf-next.task-action-supervision.v1"
PICF_SOURCE_CONTRACT_SCHEMA: Final = "picf-next.g3-picf-source-contract.v1"
EXPECTED_LINGBOT_HOST_LAYERS: Final = 36
PICF_CRITICAL_SOURCE_FILES: Final = {
    "tools/run_lingbot_vla2_ltop_g3_action_mediation.py",
    "src/picf_next/lingbot_native/task_address_learning.py",
    "src/picf_next/lingbot_native/task_action_supervision.py",
}
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_PATTERN: Final = re.compile(r"[0-9a-f]{40}\Z")

EXPECTED_ACTION_SUPERVISION: Final = {
    "schema": ACTION_SUPERVISION_SCHEMA,
    "official_action_loss": "immutable-source-task-action-pairs-only",
    "crossed_prompt_action_loss": False,
    "crossed_prompts": "representation-and-causal-evaluation-only",
    "ambiguous_source_task_address_loss": False,
    "unobservable_source_target_address_loss": False,
    "unobservable_source_target_policy": ("disable-address-only-with-explicit-loss-side-receipt"),
}

SourceAlignedAcceptanceContractError = MediatorAcceptanceContractError


def _validate_picf_source_contract(value: object, *, name: str) -> dict[str, Any]:
    contract = _mapping(value, name=name)
    if set(contract) != {
        "schema",
        "repository_commit",
        "repository_tree",
        "worktree_clean",
        "critical_file_sha256",
    }:
        raise SourceAlignedAcceptanceContractError(f"{name} fields differ")
    if contract.get("schema") != PICF_SOURCE_CONTRACT_SCHEMA:
        raise SourceAlignedAcceptanceContractError(f"{name} schema differs")
    for field in ("repository_commit", "repository_tree"):
        item = contract.get(field)
        if not isinstance(item, str) or not _GIT_OBJECT_PATTERN.fullmatch(item):
            raise SourceAlignedAcceptanceContractError(f"{name}.{field} is malformed")
    if contract.get("worktree_clean") is not True:
        raise SourceAlignedAcceptanceContractError(f"{name} is not a clean worktree")
    files = _mapping(contract.get("critical_file_sha256"), name=f"{name}.critical files")
    if set(files) != PICF_CRITICAL_SOURCE_FILES:
        raise SourceAlignedAcceptanceContractError(f"{name} critical source set differs")
    for path, digest in files.items():
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise SourceAlignedAcceptanceContractError(
                f"{name} critical source digest is malformed for {path}"
            )
    return dict(contract)


def _validate_source_schedule(training: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    contract = _mapping(training.get("training_contract"), name="training.training_contract")
    if contract.get("action_supervision") != EXPECTED_ACTION_SUPERVISION:
        raise SourceAlignedAcceptanceContractError(
            "training action supervision differs from the ADR170 source-aligned contract"
        )
    depth = _mapping(
        contract.get("task_address_supervision_depth"),
        name="training.training_contract.task_address_supervision_depth",
    )
    try:
        expected_depth = action_consumable_task_address_depth_contract(depth.get("layer_count"))
    except (TypeError, ValueError) as error:
        raise SourceAlignedAcceptanceContractError(
            "training task-address supervision depth is malformed"
        ) from error
    if dict(depth) != expected_depth:
        raise SourceAlignedAcceptanceContractError(
            "training task-address supervision depth is not action-consumable"
        )
    if depth.get("layer_count") != EXPECTED_LINGBOT_HOST_LAYERS:
        raise SourceAlignedAcceptanceContractError(
            "training task-address supervision depth differs from LingBot's host graph"
        )
    trial = _mapping(
        contract.get("action_information_set_trial"),
        name="training.training_contract.action_information_set_trial",
    )
    if trial.get("single_forward_per_optimizer_step") is not True:
        raise SourceAlignedAcceptanceContractError(
            "ADR170 training must use one forward per optimizer step"
        )
    schedule = _mapping(
        trial.get("schedule"),
        name="training.training_contract.action_information_set_trial.schedule",
    )
    schedule_sha256 = _sha256(schedule.get("sha256"), name="training schedule sha256")
    unsealed = dict(schedule)
    del unsealed["sha256"]
    if _sha256_bytes(_canonical_json(unsealed).encode("ascii")) != schedule_sha256:
        raise SourceAlignedAcceptanceContractError("training schedule sealed SHA-256 is invalid")
    expected = {
        "schema": SOURCE_ACTION_SCHEDULE_SCHEMA,
        "design": "source-task-action-scene-stratified-two-period-crossover",
        "single_forward_per_optimizer_step": True,
        "action_labels": "immutable-source-trajectory-only",
        "crossed_prompts_used_for_action_loss": False,
        "assignment": "(scene_index + period_index) mod 2",
        "steps": EXPECTED_TRAINING_STEPS,
        "scene_count": 8,
        "periods_per_cycle": 2,
        "cycle_steps": 16,
        "arm_counts": {
            "factual": EXPECTED_COUNT_PER_ARM_PER_RANK,
            "mediator-required": EXPECTED_COUNT_PER_ARM_PER_RANK,
        },
    }
    for field, value in expected.items():
        if schedule.get(field) != value:
            raise SourceAlignedAcceptanceContractError(
                f"training source-action schedule {field} differs"
            )
    entries = _sequence(schedule.get("entries"), name="training schedule entries")
    if len(entries) != EXPECTED_TRAINING_STEPS:
        raise SourceAlignedAcceptanceContractError("training schedule entry count is not 256")
    arm_counts = {"factual": 0, "mediator-required": 0}
    scene_counts: dict[str, dict[str, int]] = {}
    for expected_step, raw in enumerate(entries, start=1):
        entry = _mapping(raw, name=f"training schedule entry {expected_step}")
        if entry.get("global_step") != expected_step:
            raise SourceAlignedAcceptanceContractError(
                "training schedule global steps are not contiguous"
            )
        arm = entry.get("arm")
        if arm not in arm_counts:
            raise SourceAlignedAcceptanceContractError("training schedule contains an unknown arm")
        scene_key = entry.get("scene_key")
        source_task_key = entry.get("source_task_key")
        if not isinstance(scene_key, str) or not scene_key:
            raise SourceAlignedAcceptanceContractError("training schedule scene key is invalid")
        if not isinstance(source_task_key, str) or not source_task_key:
            raise SourceAlignedAcceptanceContractError(
                "training schedule source task key is invalid"
            )
        arm_counts[arm] += 1
        counts = scene_counts.setdefault(scene_key, {"factual": 0, "mediator-required": 0})
        counts[arm] += 1
    if arm_counts != expected["arm_counts"]:
        raise SourceAlignedAcceptanceContractError("training schedule arm counts differ")
    if len(scene_counts) != expected["scene_count"] or any(
        counts != {"factual": 16, "mediator-required": 16} for counts in scene_counts.values()
    ):
        raise SourceAlignedAcceptanceContractError(
            "training schedule does not counterbalance every source scene"
        )
    if schedule.get("scene_arm_counts") != scene_counts:
        raise SourceAlignedAcceptanceContractError(
            "training schedule scene-arm summary differs from entries"
        )
    return schedule_sha256, schedule


def _validate_action_supervision_history(
    report: Mapping[str, Any],
    *,
    rank: int,
) -> None:
    if report.get("action_supervision_schema") != ACTION_SUPERVISION_SCHEMA:
        raise SourceAlignedAcceptanceContractError(
            f"training rank {rank} action supervision schema differs"
        )
    history = _sequence(
        report.get("action_supervision_history"),
        name=f"training rank {rank} action supervision history",
    )
    if len(history) != EXPECTED_TRAINING_STEPS:
        raise SourceAlignedAcceptanceContractError(
            f"training rank {rank} action supervision history is incomplete"
        )
    for step, raw in enumerate(history, start=1):
        receipt = _mapping(raw, name=f"training rank {rank} action receipt {step}")
        expected = {
            "schema": ACTION_SUPERVISION_SCHEMA,
            "scope": "factual-action",
            "official_action_loss_enabled": True,
        }
        for field, value in expected.items():
            if receipt.get(field) != value:
                raise SourceAlignedAcceptanceContractError(
                    f"training rank {rank} action receipt {step} violates {field}"
                )
        for source, candidate in (
            ("source_task_key", "candidate_task_key"),
            ("source_instruction_sha256", "candidate_instruction_sha256"),
            ("source_action_targets_sha256", "candidate_action_targets_sha256"),
        ):
            if receipt.get(source) != receipt.get(candidate):
                raise SourceAlignedAcceptanceContractError(
                    f"training rank {rank} action receipt {step} is not source-aligned"
                )
        for field in (
            "source_instruction_sha256",
            "source_action_targets_sha256",
        ):
            _sha256(receipt.get(field), name=f"training rank {rank} action receipt {step} {field}")


def _validate_task_address_history(report: Mapping[str, Any], *, rank: int) -> None:
    history = _sequence(
        report.get("task_address_supervision_history"),
        name=f"training rank {rank} task-address history",
    )
    if len(history) != EXPECTED_TRAINING_STEPS:
        raise SourceAlignedAcceptanceContractError(
            f"training rank {rank} task-address history is incomplete"
        )
    enabled_by_reason = {
        "bound-current-frame-target": True,
        "no-singleton-source-target": False,
        "unobservable-current-frame-target": False,
    }
    for step, raw in enumerate(history, start=1):
        receipt = _mapping(raw, name=f"training rank {rank} task-address receipt {step}")
        reason = receipt.get("reason")
        if (
            reason not in enabled_by_reason
            or receipt.get("enabled") is not enabled_by_reason[reason]
        ):
            raise SourceAlignedAcceptanceContractError(
                f"training rank {rank} task-address receipt {step} is incoherent"
            )


def _validate_training(
    training: Mapping[str, Any],
) -> tuple[
    str,
    Mapping[str, Any],
    dict[int, Mapping[str, Any]],
    str,
    list[str],
    str,
    dict[str, Any],
]:
    _require_phase(
        training,
        name="training report",
        schema=TRAINING_SCHEMA,
        mode="mediator-trial",
        phase="training",
        steps=EXPECTED_TRAINING_STEPS,
    )
    schedule_sha256, _ = _validate_source_schedule(training)
    checkpoint = _mapping(training.get("checkpoint"), name="training.checkpoint")
    source_contract = _validate_picf_source_contract(
        training.get("picf_source_contract"),
        name="training.picf_source_contract",
    )
    if checkpoint.get("picf_source_contract") != source_contract:
        raise SourceAlignedAcceptanceContractError(
            "training checkpoint PICF source identity differs"
        )
    checkpoint_path = _absolute_path(checkpoint.get("path"), name="training.checkpoint.path")
    if checkpoint.get("optimizer_saved") is not False:
        raise SourceAlignedAcceptanceContractError("training checkpoint must be model-only")
    if checkpoint.get("format") != MODEL_ONLY_CHECKPOINT_FORMAT:
        raise SourceAlignedAcceptanceContractError("training checkpoint format differs")
    if checkpoint.get("action_supervision_schema") != ACTION_SUPERVISION_SCHEMA:
        raise SourceAlignedAcceptanceContractError(
            "training checkpoint action supervision schema differs"
        )
    depth_contract = _mapping(
        training.get("training_contract", {}).get("task_address_supervision_depth"),
        name="training task-address supervision depth",
    )
    if checkpoint.get("task_address_supervision_depth") != depth_contract:
        raise SourceAlignedAcceptanceContractError(
            "training checkpoint task-address supervision depth differs"
        )
    checkpoint_directory = Path(checkpoint_path)
    if checkpoint_directory.is_symlink() or not checkpoint_directory.is_dir():
        raise SourceAlignedAcceptanceContractError("training checkpoint must be one real directory")
    if {path.name for path in checkpoint_directory.iterdir()} != {
        "model",
        "ltop_g3_training_checkpoint.json",
    }:
        raise SourceAlignedAcceptanceContractError(
            "training checkpoint root differs from model-only ABI"
        )
    model_directory = checkpoint_directory / "model"
    if model_directory.is_symlink() or not model_directory.is_dir():
        raise SourceAlignedAcceptanceContractError("training checkpoint model directory is absent")
    model_names = {path.name for path in model_directory.iterdir() if path.is_file()}
    if ".metadata" not in model_names or not any(name.endswith(".distcp") for name in model_names):
        raise SourceAlignedAcceptanceContractError("training checkpoint omits DCP model payloads")
    manifest_path = checkpoint_directory / "ltop_g3_training_checkpoint.json"
    manifest, manifest_sha256, _ = _read_report(
        manifest_path,
        name="training checkpoint manifest",
    )
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise SourceAlignedAcceptanceContractError("training checkpoint manifest SHA-256 differs")
    if checkpoint.get("model_tree_schema") != MODEL_TREE_SCHEMA:
        raise SourceAlignedAcceptanceContractError("training checkpoint model-tree schema differs")
    try:
        model_tree_sha256 = directory_tree_sha256(model_directory, schema=MODEL_TREE_SCHEMA)
    except ValueError as error:
        raise SourceAlignedAcceptanceContractError(
            "training checkpoint model tree is invalid"
        ) from error
    if checkpoint.get("model_tree_sha256") != model_tree_sha256:
        raise SourceAlignedAcceptanceContractError("training checkpoint model tree SHA-256 differs")
    manifest_expected = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": EXPECTED_TRAINING_STEPS,
        "optimizer_saved": False,
        "format": MODEL_ONLY_CHECKPOINT_FORMAT,
        "world_size": EXPECTED_WORLD_SIZE,
        "model_tree_schema": MODEL_TREE_SCHEMA,
        "model_tree_sha256": model_tree_sha256,
        "action_supervision_schema": ACTION_SUPERVISION_SCHEMA,
        "picf_source_contract": source_contract,
        "task_address_supervision_depth": dict(depth_contract),
        "action_information_set_schedule_sha256": schedule_sha256,
        "source_stage_checkpoint": training.get("stage_checkpoint"),
        "g2_report_sha256": training.get("g2_report_sha256"),
        "runtime_source_contract": training.get("runtime_source_contract"),
    }
    for field, value in manifest_expected.items():
        if manifest.get(field) != value:
            raise SourceAlignedAcceptanceContractError(
                f"training checkpoint manifest {field} differs"
            )

    ranks = _rank_map(training, name="training report")
    expected_counts = {
        "factual": EXPECTED_COUNT_PER_ARM_PER_RANK,
        "mediator-required": EXPECTED_COUNT_PER_ARM_PER_RANK,
    }
    for rank, report in ranks.items():
        if report.get("action_information_set_counts") != expected_counts:
            raise SourceAlignedAcceptanceContractError(f"training rank {rank} is not 128/128")
        if report.get("action_information_set_schedule_sha256") != schedule_sha256:
            raise SourceAlignedAcceptanceContractError(
                f"training rank {rank} schedule SHA-256 differs"
            )
        if report.get("all_gradients_finite") is not True:
            raise SourceAlignedAcceptanceContractError(
                f"training rank {rank} reports non-finite gradients"
            )
        action_losses = _sequence(
            report.get("action_losses"),
            name=f"training rank {rank} action_losses",
        )
        if len(action_losses) != EXPECTED_TRAINING_STEPS:
            raise SourceAlignedAcceptanceContractError(
                f"training rank {rank} action loss trace is incomplete"
            )
        for index, value in enumerate(action_losses):
            _finite_number(value, name=f"training rank {rank} action_losses[{index}]")
        _validate_action_supervision_history(report, rank=rank)
        _validate_task_address_history(report, rank=rank)
        journal = _mapping(report.get("arm_journal"), name=f"training rank {rank} journal")
        if journal.get("rank") != rank or journal.get("record_count") != EXPECTED_TRAINING_STEPS:
            raise SourceAlignedAcceptanceContractError(
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
        if (
            _sha256(
                ranks[rank].get("post_checkpoint_save_model_local_state_sha256"),
                name=f"training rank {rank} post-save model local state sha256",
            )
            != training_digest
        ):
            raise SourceAlignedAcceptanceContractError(
                f"training rank {rank} model state changed during checkpoint save"
            )
    if checkpoint.get("training_final_model_local_state_sha256_by_rank") != training_model_digests:
        raise SourceAlignedAcceptanceContractError(
            "training checkpoint receipt terminal model digests differ"
        )
    if manifest.get("training_final_model_local_state_sha256_by_rank") != training_model_digests:
        raise SourceAlignedAcceptanceContractError(
            "training checkpoint manifest terminal model digests differ"
        )
    expected_counts_by_rank = [
        dict(ranks[rank]["action_information_set_counts"]) for rank in (0, 1)
    ]
    if manifest.get("action_information_set_counts_by_rank") != expected_counts_by_rank:
        raise SourceAlignedAcceptanceContractError("training checkpoint manifest arm counts differ")
    return (
        checkpoint_path,
        checkpoint,
        ranks,
        schedule_sha256,
        training_model_digests,
        model_tree_sha256,
        source_contract,
    )


def compose_ltop_g3_source_aligned_acceptance(
    *,
    training_path: Path,
    arm_validation_path: Path,
    factual_action_path: Path,
    mediator_action_path: Path,
    retention_path: Path,
) -> dict[str, Any]:
    """Compose the five immutable ADR170 artifacts into one long-run gate."""

    raw_paths = [
        training_path,
        arm_validation_path,
        factual_action_path,
        mediator_action_path,
        retention_path,
    ]
    resolved = [path.resolve() for path in raw_paths]
    if len(set(resolved)) != len(resolved):
        raise SourceAlignedAcceptanceContractError(
            "ADR170 source-aligned evidence paths must be distinct"
        )
    training, training_sha256, training_absolute = _read_report(
        training_path,
        name="training report",
    )
    arm, arm_sha256, arm_absolute = _read_report(
        arm_validation_path,
        name="arm validation",
    )
    factual, factual_sha256, factual_absolute = _read_report(
        factual_action_path,
        name="cold factual action evaluation",
    )
    mediator, mediator_sha256, mediator_absolute = _read_report(
        mediator_action_path,
        name="cold mediator-required action evaluation",
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
        source_contract,
    ) = _validate_training(training)
    arm_summary = _validate_arm_evidence(
        arm,
        training_path=training_absolute,
        training_sha256=training_sha256,
        training_ranks=training_ranks,
        schedule_sha256=schedule_sha256,
    )

    for name, report, expected_information_set in (
        ("cold factual action evaluation", factual, "factual"),
        ("cold mediator-required action evaluation", mediator, "mediator-required"),
    ):
        _require_phase(
            report,
            name=name,
            schema=ACTION_EVALUATION_SCHEMA,
            mode="gate",
            phase="evaluation",
            steps=EXPECTED_EVALUATION_STEPS,
        )
        if report.get("evaluation_action_information_set") != expected_information_set:
            raise SourceAlignedAcceptanceContractError(
                f"{name} used another action information set"
            )
        if report.get("evaluation_scenes_per_partition") != 4:
            raise SourceAlignedAcceptanceContractError(
                f"{name} did not declare the full four-scene scope"
            )
        if report.get("evaluation_scope") != "full":
            raise SourceAlignedAcceptanceContractError(
                f"{name} did not declare the formal full scope"
            )
        if report.get("checkpoint") is not None:
            raise SourceAlignedAcceptanceContractError(f"{name} published a new checkpoint")
        for rank, rank_report in _rank_map(report, name=name).items():
            history = _sequence(rank_report.get("history"), name=f"{name} rank {rank} history")
            if len(history) != 1:
                raise SourceAlignedAcceptanceContractError(
                    f"{name} rank {rank} must contain one cold receipt"
                )
            receipt = _mapping(history[0], name=f"{name} rank {rank} receipt")
            for partition in ("validation", "heldout"):
                partition_report = _mapping(
                    receipt.get(partition),
                    name=f"{name} rank {rank} {partition}",
                )
                scenes = _sequence(
                    partition_report.get("scenes"),
                    name=f"{name} rank {rank} {partition} scenes",
                )
                if len(scenes) != 4:
                    raise SourceAlignedAcceptanceContractError(
                        f"{name} rank {rank} {partition} is not the full four-scene gate"
                    )
    _require_phase(
        retention,
        name="cold retention",
        schema=RETENTION_SCHEMA,
        mode="gate",
        phase="retention",
        steps=EXPECTED_EVALUATION_STEPS,
    )
    if retention.get("checkpoint") is not None:
        raise SourceAlignedAcceptanceContractError("cold retention published a new checkpoint")
    for name, report in (
        ("factual action", factual),
        ("mediator-required action", mediator),
        ("retention", retention),
    ):
        if report.get("trained_checkpoint") != checkpoint_path:
            raise SourceAlignedAcceptanceContractError(
                f"{name}.trained_checkpoint differs from training checkpoint path"
            )
        if report.get("seed") != training.get("seed"):
            raise SourceAlignedAcceptanceContractError(f"{name}.seed differs from training.seed")
        cold_source_contract = _validate_picf_source_contract(
            report.get("picf_source_contract"),
            name=f"{name}.picf_source_contract",
        )
        if cold_source_contract != source_contract:
            raise SourceAlignedAcceptanceContractError(
                f"{name}.picf_source_contract differs from training"
            )

    identity = _validate_identity(training, factual, retention)
    if _validate_identity(training, mediator, retention) != identity:
        raise SourceAlignedAcceptanceContractError(
            "cold action arms do not share one source identity"
        )
    runtime_schedules = [
        _sha256(
            training_ranks[rank].get("runtime_schedule_sha256"),
            name=f"training rank {rank} runtime schedule sha256",
        )
        for rank in (0, 1)
    ]
    cold_model_digests: dict[str, list[str]] = {}
    for name, report in (
        ("factual_action", factual),
        ("mediator_required_action", mediator),
        ("retention", retention),
    ):
        cold_model_digests[name] = _cold_model_evidence(
            report,
            name=f"cold {name}",
            expected_model_digests=training_model_digests,
            expected_model_tree_sha256=model_tree_sha256,
            expected_runtime_schedule_sha256_by_rank=runtime_schedules,
        )
    if len({tuple(values) for values in cold_model_digests.values()}) != 1:
        raise SourceAlignedAcceptanceContractError(
            "cold ADR170 evidence restored different model states"
        )
    factual_summary = _validate_action_summary(factual)
    mediator_summary = _validate_action_summary(mediator)
    retention_summary = _validate_retention_summary(retention)
    supervision = dict(
        _mapping(training.get("training_contract"), name="training contract")["action_supervision"]
    )
    return {
        "schema": SOURCE_ALIGNED_ACCEPTANCE_SCHEMA,
        "status": "PASS",
        "failures": [],
        "world_size": EXPECTED_WORLD_SIZE,
        **identity,
        "checkpoint": dict(checkpoint),
        "training_final_model_local_state_sha256_by_rank": training_model_digests,
        "cold_load_model_local_state_sha256_by_rank": cold_model_digests,
        "checkpoint_identity": {
            "manifest_sha256": checkpoint["manifest_sha256"],
            "manifest_schema": TRAINING_CHECKPOINT_SCHEMA,
            "model_tree_schema": MODEL_TREE_SCHEMA,
            "model_tree_sha256": model_tree_sha256,
            "picf_source_contract": source_contract,
        },
        "training_contract": {
            "mode": "mediator-trial",
            "steps": EXPECTED_TRAINING_STEPS,
            "eval_every": EXPECTED_EVAL_EVERY,
            "schedule_schema": SOURCE_ACTION_SCHEDULE_SCHEMA,
            "schedule_sha256": schedule_sha256,
            "action_supervision": supervision,
            "seed": training["seed"],
            "runtime_schedule_sha256_by_rank": runtime_schedules,
            "arm_validation": {
                "window_size": EXPECTED_ARM_WINDOW,
                "maximum_last_to_first_ratio": MAXIMUM_LAST_TO_FIRST_RATIO,
            },
        },
        "evidence": {
            "training_report": {"path": training_absolute, "sha256": training_sha256},
            "arm_validation": {"path": arm_absolute, "sha256": arm_sha256},
            "cold_action_factual": {"path": factual_absolute, "sha256": factual_sha256},
            "cold_action_mediator_required": {
                "path": mediator_absolute,
                "sha256": mediator_sha256,
            },
            "cold_retention": {"path": retention_absolute, "sha256": retention_sha256},
        },
        "arm_validation_summary": arm_summary,
        "action_summary": {
            "factual": factual_summary,
            "mediator_required": mediator_summary,
        },
        "retention_summary": retention_summary,
    }
