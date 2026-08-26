from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
)
from picf_next.lingbot_native.predictive_decision import (
    IMPLEMENTED_PREDICTIVE_OBJECTIVE,
    IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
)
from tools.build_lingbot_native_gate_decision import (
    _parse_evidence,
    build_training_gate_decision,
)
from tools.build_lingbot_predictive_objective_decision import (
    build_predictive_objective_decision,
)
from tools.run_lingbot_vla2_native_full import (
    TRAINING_GATE_EVIDENCE_SCHEMAS,
    load_training_gate_decision,
)


def _write_pass(path: Path, *, schema: str, **extra: object) -> Path:
    path.write_text(json.dumps({"schema": schema, "status": "PASS", **extra}, sort_keys=True))
    return path


def _write_predictive_owner_decision(path: Path) -> Path:
    record = path.with_suffix(".md")
    record.write_text("owner-reviewed ADR-82 fixture")
    value = build_predictive_objective_decision(
        reviewer="local-test",
        temporal_objective=IMPLEMENTED_PREDICTIVE_OBJECTIVE,
        visible_support_weighting=IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
        minimum_visible_fraction=0.0,
        decision_record=record,
    )
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _gate_evidence(
    tmp_path: Path,
    gate: str,
    preflight_report_factory,
    smoke_report_factory,
) -> tuple[tuple[str, Path], ...]:
    values = []
    for name, schema in TRAINING_GATE_EVIDENCE_SCHEMAS[gate]:
        if name in {"preflight", "static_causality", "frozen_local_contract"}:
            values.append((name, preflight_report_factory(tmp_path / f"{gate}.{name}.json")))
            continue
        if name == "predictive_objective_decision":
            values.append(
                (name, _write_predictive_owner_decision(tmp_path / f"{gate}.{name}.json"))
            )
            continue
        if name in {"neutral", "released_isolation"}:
            values.append((name, smoke_report_factory(tmp_path / f"{gate}.{name}.json")))
            continue
        extra: dict[str, object] = {}
        if name == "fresh_update":
            extra.update(phase="fresh", input_global_step=0, saved_global_step=1)
        if name == "cold_resume":
            extra.update(phase="resume", input_global_step=1, saved_global_step=2)
        if name in {"fresh_update", "cold_resume"}:
            extra.update(
                full_shard=True,
                fsdp2_placement=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
                gradient_checkpointing=True,
                auxiliary_target_losses_enabled=False,
                execution_contract_sha256="a" * 64,
                implementation_sha256="b" * 64,
                model_family_sha256="c" * 64,
                plan_sha256="d" * 64,
                rank_reports=[
                    {
                        "rank": rank,
                        "resume_boundary_verified": name == "cold_resume",
                        "resume_runtime_rng_verified": name == "cold_resume",
                    }
                    for rank in range(2)
                ],
            )
        values.append((name, _write_pass(tmp_path / f"{gate}.{name}.json", schema=schema, **extra)))
    return tuple(values)


def test_gate_decision_binds_criteria_reviewer_and_passed_evidence(
    tmp_path: Path,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    criteria = tmp_path / "criteria.md"
    criteria.write_text("frozen G0 acceptance criteria")
    evidence = _gate_evidence(
        tmp_path,
        "G1",
        preflight_report_factory,
        smoke_report_factory,
    )
    value = build_training_gate_decision(
        gate="G1",
        reviewer="owner-review",
        criteria=criteria,
        evidence=evidence,
    )
    path = tmp_path / "G0.decision.json"
    payload = json.dumps(value, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    assert (
        load_training_gate_decision(
            path,
            expected_gate="G1",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )[1]
        == value
    )

    evidence[0][1].write_text(json.dumps({"status": "FAIL"}))
    with pytest.raises(ValueError, match="evidence differs"):
        load_training_gate_decision(path, expected_gate="G1")


def test_gate_decision_rejects_unpassed_or_reused_evidence(
    tmp_path: Path,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    criteria = tmp_path / "criteria.md"
    criteria.write_text("criteria")
    evidence = list(
        _gate_evidence(
            tmp_path,
            "G1",
            preflight_report_factory,
            smoke_report_factory,
        )
    )
    evidence[0][1].write_text(
        json.dumps(
            {
                "schema": TRAINING_GATE_EVIDENCE_SCHEMAS["G1"][0][1],
                "status": "FAIL",
                "cloud_g0_ready": True,
            }
        )
    )
    with pytest.raises(ValueError, match="passed report schema"):
        build_training_gate_decision(
            gate="G1",
            reviewer="reviewer",
            criteria=criteria,
            evidence=tuple(evidence),
        )

    passed = _write_pass(
        tmp_path / "passed.json",
        schema=TRAINING_GATE_EVIDENCE_SCHEMAS["G1"][0][1],
        cloud_g0_ready=True,
    )
    with pytest.raises(ValueError, match="distinct files"):
        build_training_gate_decision(
            gate="G1",
            reviewer="reviewer",
            criteria=criteria,
            evidence=(("static_causality", passed), ("released_isolation", passed)),
        )
    with pytest.raises(ValueError, match="unique"):
        _parse_evidence([f"same={passed}", f"same={evidence[0][1]}"])


def test_gate_decision_rejects_wrong_gate_kind_and_tampering(
    tmp_path: Path,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    criteria = tmp_path / "criteria.md"
    criteria.write_text("criteria")
    evidence = _gate_evidence(
        tmp_path,
        "G2_PROTOCOL",
        preflight_report_factory,
        smoke_report_factory,
    )
    value = build_training_gate_decision(
        gate="G2_PROTOCOL",
        reviewer="reviewer",
        criteria=criteria,
        evidence=evidence,
    )
    value["decision_kind"] = "empirical"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value, sort_keys=True))
    with pytest.raises(ValueError, match="does not pass"):
        load_training_gate_decision(path, expected_gate="G2_PROTOCOL")


def test_incomplete_empirical_gate_cannot_authorize_long_training(tmp_path: Path) -> None:
    criteria = tmp_path / "criteria.md"
    criteria.write_text("frozen G2 criteria")
    evidence = tuple(
        (
            name,
            _write_pass(
                tmp_path / f"G2.{name}.json",
                schema=schema,
                **(
                    {
                        "cuda_allocator": "native",
                        "fsdp2_placement": FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
                    }
                    if name == "pilot_train"
                    else {}
                ),
            ),
        )
        for name, schema in TRAINING_GATE_EVIDENCE_SCHEMAS["G2"]
    )
    with pytest.raises(ValueError, match="fields differ from schema"):
        build_training_gate_decision(
            gate="G2",
            reviewer="reviewer",
            criteria=criteria,
            evidence=evidence,
        )
