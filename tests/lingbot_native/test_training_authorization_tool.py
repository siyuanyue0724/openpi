from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_GPU_SHARDED,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    SELECTIVE_EMBEDDING_PARAMETER,
)
from picf_next.lingbot_native.predictive_decision import (
    IMPLEMENTED_PREDICTIVE_OBJECTIVE,
    IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
    PREDICTIVE_OBJECTIVE_POSTERIOR_FUTURE_IDENTITY,
)
from tools.build_lingbot_native_gate_decision import build_training_gate_decision
from tools.build_lingbot_native_training_authorization import (
    _parse_prerequisites,
    build_training_authorization,
)
from tools.build_lingbot_predictive_objective_decision import (
    build_predictive_objective_decision,
)
from tools.run_lingbot_vla2_native_full import (
    TRAINING_GATE_EVIDENCE_SCHEMAS,
    training_authorization_acceptance_subject,
    validate_full_objective_report,
    validate_training_authorization,
)


def _write_pass(path: Path, **extra: object) -> Path:
    path.write_text(json.dumps({"status": "PASS", **extra}, sort_keys=True))
    return path


def _storage_for_placement(placement: str) -> dict[str, object]:
    if placement == FSDP2_CPU_OFFLOAD:
        cpu_tensors, cpu_elements = 2, 10
        cuda_tensors, cuda_elements = 0, 0
        names: list[str] = []
    elif placement == FSDP2_GPU_SHARDED:
        cpu_tensors, cpu_elements = 0, 0
        cuda_tensors, cuda_elements = 2, 10
        names = []
    else:
        cpu_tensors, cpu_elements = 1, 4
        cuda_tensors, cuda_elements = 1, 6
        names = [SELECTIVE_EMBEDDING_PARAMETER]
    return {
        "parameter_tensors": 2,
        "local_elements": 10,
        "master_dtype": "float32",
        "placement": placement,
        "cpu_parameter_tensors": cpu_tensors,
        "cpu_local_elements": cpu_elements,
        "cuda_parameter_tensors": cuda_tensors,
        "cuda_local_elements": cuda_elements,
        "selective_cpu_parameter_names": names,
    }


def _write_predictive_owner_decision(
    path: Path,
    *,
    temporal_objective: str = IMPLEMENTED_PREDICTIVE_OBJECTIVE,
    minimum_visible_fraction: float = 0.0,
) -> Path:
    record = path.with_suffix(".md")
    record.write_text("owner-reviewed ADR-82 fixture")
    value = build_predictive_objective_decision(
        reviewer="local-test",
        temporal_objective=temporal_objective,
        visible_support_weighting=IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
        minimum_visible_fraction=minimum_visible_fraction,
        decision_record=record,
    )
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _write_gate_decision(
    tmp_path: Path,
    gate: str,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
    *,
    predictive_objective: str = IMPLEMENTED_PREDICTIVE_OBJECTIVE,
    predictive_minimum_visible_fraction: float = 0.0,
    fsdp2_placement: str = FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    cuda_allocator: str = "native",
) -> Path:
    criteria = tmp_path / "criteria.md"
    criteria.write_text("frozen criteria")
    evidence = []
    for name, schema in TRAINING_GATE_EVIDENCE_SCHEMAS[gate]:
        if name in {"preflight", "static_causality", "frozen_local_contract"}:
            evidence.append((name, preflight_report_factory(tmp_path / f"{gate}.{name}.json")))
            continue
        if name == "predictive_objective_decision":
            evidence.append(
                (
                    name,
                    _write_predictive_owner_decision(
                        tmp_path / f"{gate}.{name}.json",
                        temporal_objective=predictive_objective,
                        minimum_visible_fraction=predictive_minimum_visible_fraction,
                    ),
                )
            )
            continue
        if name in {"neutral", "released_isolation"}:
            evidence.append((name, smoke_report_factory(tmp_path / f"{gate}.{name}.json")))
            continue
        if name in {"fresh_update", "cold_resume"}:
            report_path = g0_report_factory(
                tmp_path / f"{gate}.{name}.json",
                phase="fresh" if name == "fresh_update" else "resume",
            )
            if fsdp2_placement != FSDP2_SELECTIVE_EMBEDDING_OFFLOAD or cuda_allocator != "native":
                report = json.loads(report_path.read_text())
                report["parameter_storage"] = _storage_for_placement(fsdp2_placement)
                report["fsdp2_placement"] = fsdp2_placement
                report["cuda_allocator"] = cuda_allocator
                payload = json.dumps(report, sort_keys=True)
                report_path.write_text(payload)
                checkpoint_report = Path(report["checkpoint_dir"]) / "native_g0_report.json"
                checkpoint_report.write_text(payload)
            evidence.append(
                (
                    name,
                    report_path,
                )
            )
            continue
        extra: dict[str, object] = {"schema": schema}
        evidence.append((name, _write_pass(tmp_path / f"{gate}.{name}.json", **extra)))
    value = build_training_gate_decision(
        gate=gate,
        reviewer="local-test",
        criteria=criteria,
        evidence=tuple(evidence),
    )
    path = tmp_path / f"{gate}.decision.json"
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def test_authorization_builder_and_runner_validate_one_exact_manifest(
    tmp_path: Path,
    full_objective_report_factory,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    digest = "a" * 64
    input_report = full_objective_report_factory(
        tmp_path / "native_full_step_1.json",
        digest=digest,
    )
    prerequisites = tuple(
        (
            gate,
            _write_gate_decision(
                tmp_path,
                gate,
                g0_report_factory,
                preflight_report_factory,
                smoke_report_factory,
            ),
        )
        for gate in ("G0", "G1", "G2_PROTOCOL")
    )
    value = build_training_authorization(
        stage="pilot",
        input_full_report=input_report,
        maximum_global_step=120,
        total_planned_steps=30_000,
        visual_audit_every=20,
        prerequisites=prerequisites,
    )
    path = tmp_path / "authorization.json"
    payload = json.dumps(value, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    assert (
        validate_training_authorization(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            input_global_step=1,
            requested_global_step=120,
            total_planned_steps=30_000,
            visual_audit_every=20,
            execution_contract_sha256=digest,
            implementation_sha256=digest,
            model_family_sha256=digest,
        )
        == value
    )


def test_authorization_preserves_explicit_gpu_sharded_topology(
    tmp_path: Path,
    full_objective_report_factory,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    digest = "7" * 64
    input_report = full_objective_report_factory(
        tmp_path / "native-full-gpu.json",
        digest=digest,
    )
    input_value = json.loads(input_report.read_text())
    input_value["fsdp2_placement"] = FSDP2_GPU_SHARDED
    input_value["parameter_storage"] = _storage_for_placement(FSDP2_GPU_SHARDED)
    payload = json.dumps(input_value, sort_keys=True)
    input_report.write_text(payload)
    checkpoint_report = Path(input_value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)
    prerequisites = tuple(
        (
            gate,
            _write_gate_decision(
                tmp_path,
                gate,
                g0_report_factory,
                preflight_report_factory,
                smoke_report_factory,
                fsdp2_placement=FSDP2_GPU_SHARDED,
            ),
        )
        for gate in ("G0", "G1", "G2_PROTOCOL")
    )

    authorization = build_training_authorization(
        stage="pilot",
        input_full_report=input_report,
        maximum_global_step=20,
        total_planned_steps=30_000,
        visual_audit_every=5,
        prerequisites=prerequisites,
    )
    path = tmp_path / "gpu-authorization.json"
    authorization_payload = json.dumps(authorization, sort_keys=True).encode("ascii")
    path.write_bytes(authorization_payload)
    kwargs = {
        "expected_sha256": hashlib.sha256(authorization_payload).hexdigest(),
        "input_global_step": 1,
        "requested_global_step": 20,
        "total_planned_steps": 30_000,
        "visual_audit_every": 5,
        "execution_contract_sha256": digest,
        "implementation_sha256": digest,
        "model_family_sha256": digest,
    }

    with pytest.raises(ValueError, match="FSDP2 execution contract"):
        validate_training_authorization(path, **kwargs)
    assert (
        validate_training_authorization(
            path,
            **kwargs,
            expected_fsdp2_placement=FSDP2_GPU_SHARDED,
        )
        == authorization
    )


def test_authorization_preserves_explicit_allocator_contract(
    tmp_path: Path,
    full_objective_report_factory,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    digest = "6" * 64
    input_report = full_objective_report_factory(
        tmp_path / "native-full-expandable.json",
        digest=digest,
    )
    input_value = json.loads(input_report.read_text())
    input_value["cuda_allocator"] = "expandable-segments"
    payload = json.dumps(input_value, sort_keys=True)
    input_report.write_text(payload)
    checkpoint_report = Path(input_value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)
    prerequisites = tuple(
        (
            gate,
            _write_gate_decision(
                tmp_path,
                gate,
                g0_report_factory,
                preflight_report_factory,
                smoke_report_factory,
                cuda_allocator="expandable-segments",
            ),
        )
        for gate in ("G0", "G1", "G2_PROTOCOL")
    )

    authorization = build_training_authorization(
        stage="pilot",
        input_full_report=input_report,
        maximum_global_step=20,
        total_planned_steps=30_000,
        visual_audit_every=5,
        prerequisites=prerequisites,
    )
    path = tmp_path / "expandable-authorization.json"
    authorization_payload = json.dumps(authorization, sort_keys=True).encode("ascii")
    path.write_bytes(authorization_payload)
    kwargs = {
        "expected_sha256": hashlib.sha256(authorization_payload).hexdigest(),
        "input_global_step": 1,
        "requested_global_step": 20,
        "total_planned_steps": 30_000,
        "visual_audit_every": 5,
        "execution_contract_sha256": digest,
        "implementation_sha256": digest,
        "model_family_sha256": digest,
    }

    with pytest.raises(ValueError, match="CUDA allocator contract"):
        validate_training_authorization(path, **kwargs)
    assert (
        validate_training_authorization(
            path,
            **kwargs,
            expected_cuda_allocator="expandable-segments",
        )
        == authorization
    )


def test_authorization_builder_rejects_missing_or_reused_gate_reports(
    tmp_path: Path,
    full_objective_report_factory,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    with pytest.raises(ValueError, match="exactly"):
        _parse_prerequisites([f"G0={tmp_path / 'g0.json'}"], stage="pilot")

    digest = "b" * 64
    input_report = full_objective_report_factory(
        tmp_path / "input.json",
        digest=digest,
    )
    shared = _write_gate_decision(
        tmp_path,
        "G0",
        g0_report_factory,
        preflight_report_factory,
        smoke_report_factory,
    )
    with pytest.raises(ValueError, match="distinct"):
        build_training_authorization(
            stage="pilot",
            input_full_report=input_report,
            maximum_global_step=20,
            total_planned_steps=30_000,
            visual_audit_every=5,
            prerequisites=(("G0", shared), ("G1", shared), ("G2_PROTOCOL", shared)),
        )


def test_authorization_rejects_unimplemented_objective_and_runtime_support_mismatch(
    tmp_path: Path,
    full_objective_report_factory,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    digest = "9" * 64
    input_report = full_objective_report_factory(tmp_path / "input.json", digest=digest)

    wrong_root = tmp_path / "wrong-objective"
    wrong_root.mkdir()
    wrong_objective = tuple(
        (
            gate,
            _write_gate_decision(
                wrong_root,
                gate,
                g0_report_factory,
                preflight_report_factory,
                smoke_report_factory,
                predictive_objective=PREDICTIVE_OBJECTIVE_POSTERIOR_FUTURE_IDENTITY,
            ),
        )
        for gate in ("G0", "G1", "G2_PROTOCOL")
    )
    with pytest.raises(ValueError, match="another temporal objective"):
        build_training_authorization(
            stage="pilot",
            input_full_report=input_report,
            maximum_global_step=20,
            total_planned_steps=30_000,
            visual_audit_every=5,
            prerequisites=wrong_objective,
        )

    threshold_root = tmp_path / "wrong-threshold"
    threshold_root.mkdir()
    threshold_prerequisites = tuple(
        (
            gate,
            _write_gate_decision(
                threshold_root,
                gate,
                g0_report_factory,
                preflight_report_factory,
                smoke_report_factory,
                predictive_minimum_visible_fraction=0.125,
            ),
        )
        for gate in ("G0", "G1", "G2_PROTOCOL")
    )
    authorization = build_training_authorization(
        stage="pilot",
        input_full_report=input_report,
        maximum_global_step=20,
        total_planned_steps=30_000,
        visual_audit_every=5,
        prerequisites=threshold_prerequisites,
    )
    path = tmp_path / "threshold-authorization.json"
    payload = json.dumps(authorization, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="another support threshold"):
        validate_training_authorization(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            input_global_step=1,
            requested_global_step=20,
            total_planned_steps=30_000,
            visual_audit_every=5,
            execution_contract_sha256=digest,
            implementation_sha256=digest,
            model_family_sha256=digest,
            expected_predictive_objective=IMPLEMENTED_PREDICTIVE_OBJECTIVE,
            expected_predictive_visible_support_weighting=(
                IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING
            ),
            expected_predictive_minimum_visible_fraction=0.0,
        )


def test_authorization_builder_rejects_zero_family_gradient_evidence(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "c" * 64
    input_report = full_objective_report_factory(
        tmp_path / "input.json",
        digest=digest,
        input_step=1,
    )
    value = json.loads(input_report.read_text())
    value["rank_reports"][0]["steps"][0]["family_gradient_diagnostics"]["gradient_norms"][
        "structural"
    ] = 0.0
    input_report.write_text(json.dumps(value, sort_keys=True))
    with pytest.raises(ValueError, match="structural family-gradient norm must be positive"):
        validate_full_objective_report(
            value,
            require_initial_probe=False,
        )


def test_authorization_builder_rejects_impossible_family_gradient_cosine(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "d" * 64
    input_report = full_objective_report_factory(
        tmp_path / "input.json",
        digest=digest,
        input_step=1,
    )
    value = json.loads(input_report.read_text())
    value["rank_reports"][0]["steps"][0]["family_gradient_diagnostics"]["cosines"][
        "action__structural"
    ] = 1.5
    payload = json.dumps(value, sort_keys=True)
    input_report.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)
    with pytest.raises(ValueError, match="outside"):
        validate_full_objective_report(
            value,
            require_initial_probe=False,
        )


def test_resumed_report_recursively_validates_its_pilot_authorization(
    tmp_path: Path,
    full_objective_report_factory,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    digest = "e" * 64
    initial = full_objective_report_factory(
        tmp_path / "initial.json",
        digest=digest,
    )
    prerequisites = tuple(
        (
            gate,
            _write_gate_decision(
                tmp_path,
                gate,
                g0_report_factory,
                preflight_report_factory,
                smoke_report_factory,
            ),
        )
        for gate in ("G0", "G1", "G2_PROTOCOL")
    )
    authorization = build_training_authorization(
        stage="pilot",
        input_full_report=initial,
        maximum_global_step=120,
        total_planned_steps=30_000,
        visual_audit_every=20,
        prerequisites=prerequisites,
    )
    manifest = tmp_path / "pilot-authorization.json"
    payload = json.dumps(authorization, sort_keys=True)
    manifest.write_text(payload)
    embedded = {
        **authorization,
        "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
    }
    resumed_path = full_objective_report_factory(
        tmp_path / "resumed.json",
        digest=digest,
        input_step=1,
        training_authorization=embedded,
    )
    resumed = json.loads(resumed_path.read_text())
    assert (
        validate_full_objective_report(
            resumed,
            expected_saved_global_step=2,
            require_initial_probe=False,
            require_source_evidence=True,
        )
        == resumed
    )

    manifest.write_text(payload + " ")
    with pytest.raises(ValueError, match="manifest differs"):
        validate_full_objective_report(
            resumed,
            expected_saved_global_step=2,
            require_initial_probe=False,
            require_source_evidence=True,
        )


def test_builder_renews_verified_pilot_without_widening_its_authority(
    tmp_path: Path,
    full_objective_report_factory,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    digest = "b" * 64
    initial = full_objective_report_factory(
        tmp_path / "initial.json",
        digest=digest,
    )
    prerequisites = tuple(
        (
            gate,
            _write_gate_decision(
                tmp_path,
                gate,
                g0_report_factory,
                preflight_report_factory,
                smoke_report_factory,
            ),
        )
        for gate in ("G0", "G1", "G2_PROTOCOL")
    )
    prior = build_training_authorization(
        stage="pilot",
        input_full_report=initial,
        maximum_global_step=120,
        total_planned_steps=30_000,
        visual_audit_every=20,
        prerequisites=prerequisites,
    )
    prior_manifest = tmp_path / "prior-pilot.json"
    prior_payload = json.dumps(prior, sort_keys=True)
    prior_manifest.write_text(prior_payload)
    embedded = {
        **prior,
        "manifest_path": str(prior_manifest),
        "manifest_sha256": hashlib.sha256(prior_payload.encode("ascii")).hexdigest(),
    }
    resumed = full_objective_report_factory(
        tmp_path / "resumed.json",
        digest=digest,
        input_step=1,
        training_authorization=embedded,
    )

    renewed = build_training_authorization(
        stage="pilot",
        input_full_report=resumed,
        maximum_global_step=120,
        total_planned_steps=30_000,
        visual_audit_every=20,
        prerequisites=prerequisites,
    )
    assert renewed["input_global_step"] == 2
    assert renewed["maximum_global_step"] == 120
    assert renewed["input_full_report"]["path"] == str(resumed.resolve())
    renewed_manifest = tmp_path / "renewed-pilot.json"
    renewed_payload = json.dumps(renewed, sort_keys=True).encode("ascii")
    renewed_manifest.write_bytes(renewed_payload)
    assert (
        validate_training_authorization(
            renewed_manifest,
            expected_sha256=hashlib.sha256(renewed_payload).hexdigest(),
            input_global_step=2,
            requested_global_step=120,
            total_planned_steps=30_000,
            visual_audit_every=20,
            execution_contract_sha256=digest,
            implementation_sha256=digest,
            model_family_sha256=digest,
        )
        == renewed
    )

    with pytest.raises(ValueError, match="same prior pilot and visual cadence"):
        build_training_authorization(
            stage="pilot",
            input_full_report=resumed,
            maximum_global_step=120,
            total_planned_steps=30_000,
            visual_audit_every=10,
            prerequisites=prerequisites,
        )
    with pytest.raises(ValueError, match="cannot widen"):
        build_training_authorization(
            stage="pilot",
            input_full_report=resumed,
            maximum_global_step=121,
            total_planned_steps=30_000,
            visual_audit_every=20,
            prerequisites=prerequisites,
        )


def test_long_segments_inherit_one_immutable_evaluated_checkpoint_subject() -> None:
    digest = "f" * 64
    initial = {
        "saved_global_step": 120,
        "execution_contract_sha256": digest,
        "implementation_sha256": digest,
        "model_family_sha256": digest,
        "long_training_authorized": False,
    }
    root = training_authorization_acceptance_subject(
        stage="long",
        input_report=initial,
        input_report_sha256=digest,
    )
    assert root == {
        "input_full_report_sha256": digest,
        "saved_global_step": 120,
        "execution_contract_sha256": digest,
        "implementation_sha256": digest,
        "model_family_sha256": digest,
    }
    continuation = {
        **initial,
        "saved_global_step": 320,
        "long_training_authorized": True,
        "training_authorization": {"acceptance_subject": root},
    }
    assert (
        training_authorization_acceptance_subject(
            stage="long",
            input_report=continuation,
            input_report_sha256="a" * 64,
        )
        == root
    )
    assert (
        training_authorization_acceptance_subject(
            stage="pilot",
            input_report=initial,
            input_report_sha256=digest,
        )
        is None
    )
