from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from picf_next.artifact_io import directory_tree_sha256
from picf_next.lingbot_native.ltop_core_pilot import (
    LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY,
    LTOP_CORE_PILOT_G3_SCHEMA,
    LTOPCoreLongCadence,
    LTOPCorePilotArm,
    LTOPCorePilotCadence,
    LTOPCorePilotSmokeCadence,
    LTOPCoreRestartSmokeCadence,
    load_accepted_g3_gate,
    load_accepted_g3_mediator_gate,
    matched_arm_contract,
)
from picf_next.lingbot_native.ltop_g3_mediator_acceptance import (
    MEDIATOR_ACCEPTANCE_SCHEMA,
    MODEL_ONLY_CHECKPOINT_FORMAT,
    MODEL_TREE_SCHEMA,
    TRAINING_CHECKPOINT_SCHEMA,
)


def _g3_report() -> dict[str, object]:
    return {
        "schema": LTOP_CORE_PILOT_G3_SCHEMA,
        "status": "PASS",
        "failures": [],
        "mode": "gate",
        "steps": 128,
        "eval_every": 32,
        "world_size": 2,
    }


def test_production_cadence_has_no_step_zero_or_step_one_checkpoint() -> None:
    cadence = LTOPCorePilotCadence()
    assert [step for step in range(2_001) if cadence.metrics_due(step)][-1] == 2_000
    assert [step for step in range(2_001) if cadence.diagnostics_due(step)] == list(
        range(250, 2_001, 250)
    )
    assert not cadence.checkpoint_due(0)
    assert not cadence.checkpoint_due(1)
    assert cadence.checkpoint_due(2_000)


def test_production_cadence_rejects_override() -> None:
    with pytest.raises(ValueError, match="frozen"):
        LTOPCorePilotCadence(total_steps=128)


def test_engineering_smoke_exercises_io_at_step_two_only() -> None:
    cadence = LTOPCorePilotSmokeCadence()
    assert not cadence.metrics_due(1)
    assert cadence.metrics_due(2)
    assert cadence.diagnostics_due(2)
    assert cadence.checkpoint_due(2)


def test_restart_smoke_has_two_real_checkpoint_boundaries() -> None:
    cadence = LTOPCoreRestartSmokeCadence()
    assert cadence.total_steps == 4
    assert [step for step in range(5) if cadence.metrics_due(step)] == [2, 4]
    assert [step for step in range(5) if cadence.diagnostics_due(step)] == [2, 4]
    assert [step for step in range(5) if cadence.checkpoint_due(step)] == [2, 4]


def test_long_cadence_keeps_2k_resumable_boundaries_without_stopping() -> None:
    cadence = LTOPCoreLongCadence()
    assert cadence.total_steps == 30_000
    assert cadence.checkpoint_step == 30_000
    assert [step for step in range(30_001) if cadence.metrics_due(step)][-1] == 30_000
    assert [step for step in range(30_001) if cadence.diagnostics_due(step)][-1] == 30_000
    assert [step for step in range(30_001) if cadence.checkpoint_due(step)] == list(
        range(2_000, 30_001, 2_000)
    )


def test_long_cadence_rejects_override() -> None:
    with pytest.raises(ValueError, match="frozen"):
        LTOPCoreLongCadence(total_steps=2_000)


def test_g3_gate_preflight_accepts_only_registered_pass(tmp_path) -> None:
    path = tmp_path / "g3.json"
    path.write_text(json.dumps(_g3_report()), encoding="ascii")
    accepted = load_accepted_g3_gate(path)
    assert accepted.report["status"] == "PASS"
    assert len(accepted.file_sha256) == 64

    failed = _g3_report()
    failed["status"] = "FAIL"
    path.write_text(json.dumps(failed), encoding="ascii")
    with pytest.raises(ValueError, match="failure-free"):
        load_accepted_g3_gate(path)


def test_mediator_g3_loader_revalidates_live_evidence_and_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = checkpoint / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"minimal DCP metadata\n")
    (model / "__0_0.distcp").write_bytes(b"minimal DCP model shard\n")
    model_tree_sha256 = directory_tree_sha256(model, schema=MODEL_TREE_SCHEMA)
    schedule_sha256 = "a" * 64
    training_digests = ["b" * 64, "c" * 64]
    manifest = checkpoint / "ltop_g3_training_checkpoint.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": TRAINING_CHECKPOINT_SCHEMA,
                "status": "PASS",
                "global_step": 256,
                "optimizer_saved": False,
                "format": MODEL_ONLY_CHECKPOINT_FORMAT,
                "world_size": 2,
                "model_tree_schema": MODEL_TREE_SCHEMA,
                "model_tree_sha256": model_tree_sha256,
                "training_final_model_local_state_sha256_by_rank": training_digests,
                "action_information_set_schedule_sha256": schedule_sha256,
            }
        ),
        encoding="ascii",
    )
    evidence = {}
    for index, name in enumerate(
        ("training_report", "arm_validation", "cold_action_evaluation", "cold_retention")
    ):
        evidence_path = tmp_path / f"evidence-{index}.json"
        evidence_path.write_text(f'{{"index":{index}}}', encoding="ascii")
        evidence[name] = {
            "path": str(evidence_path.resolve()),
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        }
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "schema": MEDIATOR_ACCEPTANCE_SCHEMA,
                "status": "PASS",
                "failures": [],
                "world_size": 2,
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "format": MODEL_ONLY_CHECKPOINT_FORMAT,
                    "optimizer_saved": False,
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "model_tree_schema": MODEL_TREE_SCHEMA,
                    "model_tree_sha256": model_tree_sha256,
                    "training_final_model_local_state_sha256_by_rank": training_digests,
                },
                "checkpoint_identity": {
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "model_tree_schema": MODEL_TREE_SCHEMA,
                    "model_tree_sha256": model_tree_sha256,
                },
                "training_final_model_local_state_sha256_by_rank": training_digests,
                "training_contract": {
                    "mode": "mediator-trial",
                    "steps": 256,
                    "eval_every": 32,
                    "action_information_set_policy": "fixed-counterbalanced-50-50",
                    "schedule_sha256": schedule_sha256,
                },
                "evidence": evidence,
            }
        ),
        encoding="ascii",
    )

    monkeypatch.setattr(
        "picf_next.lingbot_native.ltop_core_pilot.compose_ltop_g3_mediator_acceptance",
        lambda **_kwargs: json.loads(acceptance.read_text(encoding="ascii")),
    )
    accepted = load_accepted_g3_mediator_gate(acceptance)

    assert accepted.checkpoint_path == checkpoint.resolve()
    assert accepted.training_final_model_local_state_sha256_by_rank == ("b" * 64, "c" * 64)
    assert accepted.checkpoint_model_tree_sha256 == model_tree_sha256
    assert LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY == ("rank-step-counterbalanced-50-50")

    Path(evidence["cold_retention"]["path"]).write_text("changed", encoding="ascii")
    with pytest.raises(ValueError, match="changed after acceptance"):
        load_accepted_g3_mediator_gate(acceptance)


def test_matched_arms_change_only_the_typed_action_edge() -> None:
    factual = matched_arm_contract(LTOPCorePilotArm.FACTUAL)
    blocked = matched_arm_contract(LTOPCorePilotArm.BLOCKED)
    differing = {key for key in factual if factual[key] != blocked[key]}
    assert differing == {"arm", "object_read_action_intervention"}
    assert factual["training_objective"] == blocked["training_objective"]
    assert factual["start_state"] == blocked["start_state"]
