from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from picf_next.lingbot_native.ltop_core_pilot import (
    load_accepted_g3_source_aligned_gate,
)
from picf_next.lingbot_native.ltop_g3_source_aligned_acceptance import (
    ACTION_SUPERVISION_SCHEMA,
    PICF_CRITICAL_SOURCE_FILES,
    PICF_SOURCE_CONTRACT_SCHEMA,
    SOURCE_ACTION_SCHEDULE_SCHEMA,
    SOURCE_ALIGNED_ACCEPTANCE_SCHEMA,
    TRAINING_CHECKPOINT_SCHEMA,
    SourceAlignedAcceptanceContractError,
    compose_ltop_g3_source_aligned_acceptance,
)
from picf_next.lingbot_native.task_address_learning import (
    action_consumable_task_address_depth_contract,
)
from tests.lingbot_native.test_ltop_g3_mediator_acceptance import (
    _action_report,
    _arm_validation,
    _identity,
    _retention_report,
    _sha256,
    _training_report,
    _write,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _source_schedule() -> dict[str, object]:
    entries = []
    scene_arm_counts = {
        f"scene-{scene}": {"factual": 0, "mediator-required": 0} for scene in range(8)
    }
    for step in range(1, 257):
        cycle_index, cycle_offset = divmod(step - 1, 16)
        period_index, scene_index = divmod(cycle_offset, 8)
        arm = "mediator-required" if (scene_index + period_index) % 2 else "factual"
        scene_key = f"scene-{scene_index}"
        scene_arm_counts[scene_key][arm] += 1
        entries.append(
            {
                "global_step": step,
                "cycle_index": cycle_index,
                "cycle_offset": cycle_offset,
                "period_index": period_index,
                "scene_index": scene_index,
                "scene_key": scene_key,
                "source_task_key": f"task-{scene_index}",
                "arm": arm,
            }
        )
    schedule: dict[str, object] = {
        "schema": SOURCE_ACTION_SCHEDULE_SCHEMA,
        "design": "source-task-action-scene-stratified-two-period-crossover",
        "single_forward_per_optimizer_step": True,
        "action_labels": "immutable-source-trajectory-only",
        "crossed_prompts_used_for_action_loss": False,
        "assignment": "(scene_index + period_index) mod 2",
        "steps": 256,
        "scene_count": 8,
        "periods_per_cycle": 2,
        "cycle_steps": 16,
        "arm_counts": {"factual": 128, "mediator-required": 128},
        "scene_arm_counts": scene_arm_counts,
        "entries": entries,
    }
    schedule["sha256"] = hashlib.sha256(_canonical_json(schedule).encode("ascii")).hexdigest()
    return schedule


def _source_contract() -> dict[str, object]:
    return {
        "schema": PICF_SOURCE_CONTRACT_SCHEMA,
        "repository_commit": "a" * 40,
        "repository_tree": "b" * 40,
        "worktree_clean": True,
        "critical_file_sha256": {
            path: hashlib.sha256(path.encode("ascii")).hexdigest()
            for path in PICF_CRITICAL_SOURCE_FILES
        },
    }


def _source_training(tmp_path: Path, identity: dict[str, object]) -> dict[str, object]:
    training = _training_report(tmp_path, identity)
    schedule = _source_schedule()
    supervision = {
        "schema": ACTION_SUPERVISION_SCHEMA,
        "official_action_loss": "immutable-source-task-action-pairs-only",
        "crossed_prompt_action_loss": False,
        "crossed_prompts": "representation-and-causal-evaluation-only",
        "ambiguous_source_task_address_loss": False,
        "unobservable_source_target_address_loss": False,
        "unobservable_source_target_policy": (
            "disable-address-only-with-explicit-loss-side-receipt"
        ),
    }
    training["training_contract"]["action_information_set_trial"]["schedule"] = schedule
    training["training_contract"]["action_supervision"] = supervision
    depth_contract = action_consumable_task_address_depth_contract(36)
    training["training_contract"]["task_address_supervision_depth"] = depth_contract
    source_contract = _source_contract()
    training["picf_source_contract"] = source_contract
    action_receipt = {
        "schema": ACTION_SUPERVISION_SCHEMA,
        "scope": "factual-action",
        "official_action_loss_enabled": True,
        "sample_key": "sample",
        "source_task_key": "task",
        "candidate_task_key": "task",
        "source_instruction_sha256": "1" * 64,
        "candidate_instruction_sha256": "1" * 64,
        "source_action_targets_sha256": "2" * 64,
        "candidate_action_targets_sha256": "2" * 64,
    }
    for rank_report in training["rank_reports"]:
        rank_report["action_information_set_schedule_sha256"] = schedule["sha256"]
        rank_report["action_supervision_schema"] = ACTION_SUPERVISION_SCHEMA
        rank_report["action_supervision_history"] = [
            copy.deepcopy(action_receipt) for _ in range(256)
        ]
        rank_report["task_address_supervision_history"] = [
            {
                "global_step": step,
                "scene_key": "scene",
                "source_task_key": "task",
                "source_target_identity": "target",
                "enabled": True,
                "reason": "bound-current-frame-target",
            }
            for step in range(1, 257)
        ]
    checkpoint_path = Path(training["checkpoint"]["path"])
    manifest_path = checkpoint_path / "ltop_g3_training_checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["schema"] = TRAINING_CHECKPOINT_SCHEMA
    manifest["action_supervision_schema"] = ACTION_SUPERVISION_SCHEMA
    manifest["picf_source_contract"] = source_contract
    manifest["task_address_supervision_depth"] = depth_contract
    manifest["action_information_set_schedule_sha256"] = schedule["sha256"]
    _write(manifest_path, manifest)
    training["checkpoint"]["action_supervision_schema"] = ACTION_SUPERVISION_SCHEMA
    training["checkpoint"]["picf_source_contract"] = source_contract
    training["checkpoint"]["task_address_supervision_depth"] = depth_contract
    training["checkpoint"]["manifest_sha256"] = _sha256(manifest_path)
    return training


def _full_action_report(
    identity: dict[str, object],
    training: dict[str, object],
    *,
    information_set: str,
) -> dict[str, object]:
    report = _action_report(identity, training)
    report["picf_source_contract"] = copy.deepcopy(training["picf_source_contract"])
    report["evaluation_action_information_set"] = information_set
    report["evaluation_scenes_per_partition"] = 4
    report["evaluation_scope"] = "full"
    for rank_report in report["rank_reports"]:
        for partition in ("validation", "heldout"):
            scene = rank_report["history"][0][partition]["scenes"][0]
            rank_report["history"][0][partition]["scenes"] = [
                copy.deepcopy(scene) for _ in range(4)
            ]
    return report


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    identity = _identity(tmp_path)
    training = _source_training(tmp_path, identity)
    training_path = tmp_path / "training.json"
    _write(training_path, training)
    arm_path = tmp_path / "arm.json"
    factual_path = tmp_path / "factual.json"
    mediator_path = tmp_path / "mediator.json"
    retention_path = tmp_path / "retention.json"
    _write(arm_path, _arm_validation(training_path, training))
    _write(
        factual_path,
        _full_action_report(identity, training, information_set="factual"),
    )
    _write(
        mediator_path,
        _full_action_report(identity, training, information_set="mediator-required"),
    )
    retention = _retention_report(identity, training)
    retention["picf_source_contract"] = copy.deepcopy(training["picf_source_contract"])
    retention["evaluation_action_information_set"] = None
    retention["representation_retention_contract"]["scientific_action_evidence"] = False
    _write(retention_path, retention)
    return {
        "training": training_path,
        "arm": arm_path,
        "factual": factual_path,
        "mediator": mediator_path,
        "retention": retention_path,
    }


def _compose(paths: dict[str, Path]) -> dict[str, object]:
    return compose_ltop_g3_source_aligned_acceptance(
        training_path=paths["training"],
        arm_validation_path=paths["arm"],
        factual_action_path=paths["factual"],
        mediator_action_path=paths["mediator"],
        retention_path=paths["retention"],
    )


def test_source_aligned_acceptance_binds_v5_checkpoint_and_both_action_arms(
    tmp_path: Path,
) -> None:
    report = _compose(_artifacts(tmp_path))

    assert report["schema"] == SOURCE_ALIGNED_ACCEPTANCE_SCHEMA
    assert report["status"] == "PASS"
    assert report["checkpoint_identity"]["manifest_schema"] == TRAINING_CHECKPOINT_SCHEMA
    assert set(report["action_summary"]) == {"factual", "mediator_required"}
    assert set(report["evidence"]) == {
        "training_report",
        "arm_validation",
        "cold_action_factual",
        "cold_action_mediator_required",
        "cold_retention",
    }


def test_source_aligned_acceptance_rejects_duplicate_or_wrong_action_arms(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    with pytest.raises(SourceAlignedAcceptanceContractError, match="distinct"):
        compose_ltop_g3_source_aligned_acceptance(
            training_path=paths["training"],
            arm_validation_path=paths["arm"],
            factual_action_path=paths["factual"],
            mediator_action_path=paths["factual"],
            retention_path=paths["retention"],
        )

    mediator = json.loads(paths["mediator"].read_text(encoding="ascii"))
    mediator["evaluation_action_information_set"] = "factual"
    _write(paths["mediator"], mediator)
    with pytest.raises(SourceAlignedAcceptanceContractError, match="another action information"):
        _compose(paths)


def test_source_aligned_acceptance_rejects_quick_scope_declaration(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    factual = json.loads(paths["factual"].read_text(encoding="ascii"))
    factual["evaluation_scenes_per_partition"] = 1
    factual["evaluation_scope"] = "quick"
    _write(paths["factual"], factual)

    with pytest.raises(SourceAlignedAcceptanceContractError, match="four-scene scope"):
        _compose(paths)


def test_source_aligned_acceptance_rejects_legacy_checkpoint_schema(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    training = json.loads(paths["training"].read_text(encoding="ascii"))
    manifest_path = Path(training["checkpoint"]["path"]) / "ltop_g3_training_checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["schema"] = "picf-next.ltop-g3-training-checkpoint.v2"
    _write(manifest_path, manifest)
    training["checkpoint"]["manifest_sha256"] = _sha256(manifest_path)
    _write(paths["training"], training)
    _write(paths["arm"], _arm_validation(paths["training"], training))

    with pytest.raises(SourceAlignedAcceptanceContractError, match="manifest schema differs"):
        _compose(paths)


def test_source_aligned_acceptance_rejects_non_consumable_address_depth(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    training = json.loads(paths["training"].read_text(encoding="ascii"))
    training["training_contract"]["task_address_supervision_depth"]["producer_layer_index"] = 35
    _write(paths["training"], training)
    _write(paths["arm"], _arm_validation(paths["training"], training))

    with pytest.raises(SourceAlignedAcceptanceContractError, match="not action-consumable"):
        _compose(paths)


def test_source_aligned_acceptance_rejects_wrong_host_depth(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    training = json.loads(paths["training"].read_text(encoding="ascii"))
    depth = action_consumable_task_address_depth_contract(37)
    training["training_contract"]["task_address_supervision_depth"] = depth
    training["checkpoint"]["task_address_supervision_depth"] = depth
    manifest_path = Path(training["checkpoint"]["path"]) / "ltop_g3_training_checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["task_address_supervision_depth"] = depth
    _write(manifest_path, manifest)
    training["checkpoint"]["manifest_sha256"] = _sha256(manifest_path)
    _write(paths["training"], training)
    _write(paths["arm"], _arm_validation(paths["training"], training))

    with pytest.raises(SourceAlignedAcceptanceContractError, match="host graph"):
        _compose(paths)


def test_source_aligned_acceptance_rejects_cold_source_drift(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    factual = json.loads(paths["factual"].read_text(encoding="ascii"))
    factual["picf_source_contract"]["repository_commit"] = "c" * 40
    _write(paths["factual"], factual)

    with pytest.raises(SourceAlignedAcceptanceContractError, match="differs from training"):
        _compose(paths)


def test_long_loader_recomposes_live_source_aligned_evidence(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    acceptance_path = tmp_path / "acceptance.json"
    _write(acceptance_path, _compose(paths))

    accepted = load_accepted_g3_source_aligned_gate(acceptance_path)

    assert accepted.path == acceptance_path.resolve()
    assert (
        accepted.checkpoint_path
        == Path(
            json.loads(paths["training"].read_text(encoding="ascii"))["checkpoint"]["path"]
        ).resolve()
    )

    factual = json.loads(paths["factual"].read_text(encoding="ascii"))
    factual["status"] = "FAIL"
    _write(paths["factual"], factual)
    with pytest.raises(ValueError, match="changed after acceptance"):
        load_accepted_g3_source_aligned_gate(acceptance_path)
