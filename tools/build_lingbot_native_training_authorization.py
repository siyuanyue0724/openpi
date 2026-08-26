#!/usr/bin/env python3
# ruff: noqa: E402
"""Build a hash-bound pilot/long authorization from passed gate reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from tools.repository_import import bind_entrypoint_to_own_repository
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot native training authorization builder",
)

from picf_next.lingbot_native.fsdp2_placement import validate_fsdp2_placement

try:
    from tools.run_lingbot_vla2_native_full import (
        FULL_REPORT_SCHEMA,
        IMPLEMENTED_PREDICTIVE_OBJECTIVE,
        IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
        TRAINING_AUTHORIZATION_GATES,
        TRAINING_AUTHORIZATION_SCHEMA,
        load_training_gate_decision,
        pilot_authorization_requires_initial_probe,
        predictive_objective_decision_from_gate_decision,
        training_authorization_acceptance_subject,
        validate_full_objective_report,
    )
    from tools.run_lingbot_vla2_native_g0 import _write_text_durable
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from run_lingbot_vla2_native_full import (  # type: ignore[no-redef]
        FULL_REPORT_SCHEMA,
        IMPLEMENTED_PREDICTIVE_OBJECTIVE,
        IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
        TRAINING_AUTHORIZATION_GATES,
        TRAINING_AUTHORIZATION_SCHEMA,
        load_training_gate_decision,
        pilot_authorization_requires_initial_probe,
        predictive_objective_decision_from_gate_decision,
        training_authorization_acceptance_subject,
        validate_full_objective_report,
    )
    from run_lingbot_vla2_native_g0 import _write_text_durable  # type: ignore[no-redef]


def _read_passed_report(path: Path, *, name: str) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be one real JSON file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise ValueError(f"{name} has not passed")
    return payload, value


def _parse_prerequisites(values: list[str], *, stage: str) -> tuple[tuple[str, Path], ...]:
    parsed: dict[str, Path] = {}
    for value in values:
        gate, separator, path = value.partition("=")
        if not separator or not gate or not path or gate in parsed:
            raise ValueError("each prerequisite must be one unique GATE=/real/report.json")
        parsed[gate] = Path(path).expanduser()
    expected = TRAINING_AUTHORIZATION_GATES[stage]
    if set(parsed) != set(expected):
        raise ValueError(f"{stage} prerequisites must be exactly {list(expected)}")
    return tuple((gate, parsed[gate]) for gate in expected)


def build_training_authorization(
    *,
    stage: str,
    input_full_report: Path,
    maximum_global_step: int,
    total_planned_steps: int,
    visual_audit_every: int,
    prerequisites: tuple[tuple[str, Path], ...],
) -> dict[str, Any]:
    """Build the exact manifest later revalidated by the distributed runner."""

    if stage not in TRAINING_AUTHORIZATION_GATES:
        raise ValueError("training authorization stage is unsupported")
    integers = (maximum_global_step, total_planned_steps, visual_audit_every)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("training authorization limits must be integers")
    if maximum_global_step <= 0 or maximum_global_step > total_planned_steps:
        raise ValueError("training authorization maximum step is outside the frozen plan")
    if stage == "pilot" and maximum_global_step > 200:
        raise ValueError("pilot authorization cannot exceed 200 optimizer steps")
    if visual_audit_every <= 0:
        raise ValueError("training authorization requires periodic visual audit")
    input_payload, input_report = _read_passed_report(
        input_full_report,
        name="input full-objective report",
    )
    if input_report.get("schema") != FULL_REPORT_SCHEMA:
        raise ValueError("input full-objective report uses an unrecognized schema")
    provenance_fields = (
        "execution_contract_sha256",
        "implementation_sha256",
        "model_family_sha256",
    )
    required_input = (
        "saved_global_step",
        *provenance_fields,
        "fsdp2_placement",
        "cuda_allocator",
    )
    if any(name not in input_report for name in required_input):
        raise ValueError("input full-objective report lacks checkpoint provenance")
    input_step = input_report["saved_global_step"]
    if (
        isinstance(input_step, bool)
        or not isinstance(input_step, int)
        or input_step <= 0
        or input_step >= maximum_global_step
    ):
        raise ValueError("input full-objective report has an invalid saved step")
    for name in provenance_fields:
        value = input_report[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("input full-objective report has invalid provenance digests")
    expected_fsdp2_placement = validate_fsdp2_placement(input_report["fsdp2_placement"])
    expected_cuda_allocator = input_report["cuda_allocator"]
    if expected_cuda_allocator not in {
        "native",
        "expandable-segments",
        "cuda-malloc-async",
    }:
        raise ValueError("input full-objective report has no explicit CUDA allocator")
    initial_pilot = pilot_authorization_requires_initial_probe(
        stage=stage,
        input_report=input_report,
        maximum_global_step=maximum_global_step,
        visual_audit_every=visual_audit_every,
    )
    validate_full_objective_report(
        input_report,
        expected_saved_global_step=input_step,
        expected_digests={name: input_report[name] for name in provenance_fields},
        require_initial_probe=initial_pilot,
        require_mature_wrong_time=stage == "long",
        require_source_evidence=stage == "long",
        expected_fsdp2_placement=expected_fsdp2_placement,
        expected_cuda_allocator=expected_cuda_allocator,
    )
    expected_gates = TRAINING_AUTHORIZATION_GATES[stage]
    if tuple(gate for gate, _path in prerequisites) != expected_gates:
        raise ValueError("training prerequisite order differs from the frozen ladder")
    input_report_path = input_full_report.resolve()
    prerequisite_paths = tuple(path.resolve() for _gate, path in prerequisites)
    if (
        len(set(prerequisite_paths)) != len(prerequisite_paths)
        or input_report_path in prerequisite_paths
    ):
        raise ValueError("training authorization reports must be distinct files")
    reports: list[dict[str, str]] = []
    input_report_sha256 = hashlib.sha256(input_payload).hexdigest()
    acceptance_subject = training_authorization_acceptance_subject(
        stage=stage,
        input_report=input_report,
        input_report_sha256=input_report_sha256,
    )
    owner_decision = None
    for (gate, path), resolved in zip(prerequisites, prerequisite_paths, strict=True):
        payload, gate_report = load_training_gate_decision(path, expected_gate=gate)
        if (
            gate == "G0"
            and gate_report["subject"].get("fsdp2_placement") != expected_fsdp2_placement
        ):
            raise ValueError("G0 decision uses another FSDP2 placement")
        if gate == "G0" and gate_report["subject"].get("cuda_allocator") != (
            expected_cuda_allocator
        ):
            raise ValueError("G0 decision uses another CUDA allocator")
        if gate == "G2_PROTOCOL":
            owner_decision = predictive_objective_decision_from_gate_decision(
                gate_report,
                expected_temporal_objective=IMPLEMENTED_PREDICTIVE_OBJECTIVE,
                expected_visible_support_weighting=(
                    IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING
                ),
            )
        if stage == "long" and gate in {"G2", "G3", "G4", "G5", "G6", "G7_PROTOCOL"}:
            subject = gate_report["subject"]
            if acceptance_subject is None:
                raise RuntimeError("long training acceptance subject was unexpectedly absent")
            if any(subject.get(name) != expected for name, expected in acceptance_subject.items()):
                raise ValueError(f"{gate} decision targets another evaluated checkpoint")
        reports.append(
            {
                "gate": gate,
                "path": str(resolved),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if owner_decision is None:
        raise ValueError("training authorization lacks a predictive owner decision")
    return {
        "schema": TRAINING_AUTHORIZATION_SCHEMA,
        "status": "PASS",
        "stage": stage,
        "input_global_step": input_step,
        "maximum_global_step": maximum_global_step,
        "visual_audit_every": visual_audit_every,
        "execution_contract_sha256": input_report["execution_contract_sha256"],
        "implementation_sha256": input_report["implementation_sha256"],
        "model_family_sha256": input_report["model_family_sha256"],
        "predictive_objective": owner_decision.temporal_objective,
        "predictive_claim_scope": owner_decision.claim_scope,
        "predictive_visible_support_weighting": owner_decision.visible_support_weighting,
        "predictive_minimum_visible_fraction_hex": (owner_decision.minimum_visible_fraction.hex()),
        "acceptance_subject": acceptance_subject,
        "input_full_report_sha256": input_report_sha256,
        "input_full_report": {
            "path": str(input_report_path),
            "sha256": input_report_sha256,
        },
        "prerequisite_reports": reports,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(TRAINING_AUTHORIZATION_GATES), required=True)
    parser.add_argument("--input-full-report", type=Path, required=True)
    parser.add_argument("--maximum-global-step", type=int, required=True)
    parser.add_argument("--total-planned-steps", type=int, default=30_000)
    parser.add_argument("--visual-audit-every", type=int, required=True)
    parser.add_argument(
        "--prerequisite",
        action="append",
        default=[],
        metavar="GATE=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    prerequisites = _parse_prerequisites(args.prerequisite, stage=args.stage)
    value = build_training_authorization(
        stage=args.stage,
        input_full_report=args.input_full_report,
        maximum_global_step=args.maximum_global_step,
        total_planned_steps=args.total_planned_steps,
        visual_audit_every=args.visual_audit_every,
        prerequisites=prerequisites,
    )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    _write_text_durable(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
                "stage": args.stage,
                "maximum_global_step": args.maximum_global_step,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
