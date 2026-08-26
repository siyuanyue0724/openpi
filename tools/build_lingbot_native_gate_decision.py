#!/usr/bin/env python3
# ruff: noqa: E402
"""Build one immutable, evidence-bound LingBot native acceptance decision."""

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
    entrypoint_name="LingBot native gate decision builder",
)

try:
    from tools.run_lingbot_vla2_native_full import (
        TRAINING_GATE_DECISION_KINDS,
        TRAINING_GATE_DECISION_SCHEMA,
        TRAINING_GATE_EVIDENCE_SCHEMAS,
        training_gate_decision_subject,
        validate_training_gate_evidence,
    )
    from tools.run_lingbot_vla2_native_g0 import _write_text_durable
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from run_lingbot_vla2_native_full import (  # type: ignore[no-redef]
        TRAINING_GATE_DECISION_KINDS,
        TRAINING_GATE_DECISION_SCHEMA,
        TRAINING_GATE_EVIDENCE_SCHEMAS,
        training_gate_decision_subject,
        validate_training_gate_evidence,
    )
    from run_lingbot_vla2_native_g0 import _write_text_durable  # type: ignore[no-redef]


def _read_real_file(path: Path, *, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be one real file")
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"{name} cannot be empty")
    return payload


def _parse_evidence(values: list[str]) -> tuple[tuple[str, Path], ...]:
    parsed: list[tuple[str, Path]] = []
    observed: set[str] = set()
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path or name in observed:
            raise ValueError("each evidence item must be one unique NAME=/real/report.json")
        observed.add(name)
        parsed.append((name, Path(path).expanduser()))
    if not parsed:
        raise ValueError("a gate decision requires at least one evidence report")
    return tuple(parsed)


def build_training_gate_decision(
    *,
    gate: str,
    reviewer: str,
    criteria: Path,
    evidence: tuple[tuple[str, Path], ...],
) -> dict[str, Any]:
    """Bind a PASS decision to frozen criteria and passed evidence reports."""

    decision_kind = TRAINING_GATE_DECISION_KINDS.get(gate)
    if decision_kind is None:
        raise ValueError("training gate is not part of the frozen acceptance ladder")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("a gate decision requires an explicit reviewer")
    if not evidence:
        raise ValueError("a gate decision requires at least one evidence report")

    criteria_payload = _read_real_file(criteria, name="gate criteria")
    criteria_path = criteria.resolve()
    evidence_paths = tuple(path.resolve() for _name, path in evidence)
    evidence_names = tuple(name for name, _path in evidence)
    if len(set(evidence_names)) != len(evidence_names) or any(
        not name.strip() for name in evidence_names
    ):
        raise ValueError("gate evidence names must be nonempty and distinct")
    if len(set(evidence_paths)) != len(evidence_paths) or criteria_path in evidence_paths:
        raise ValueError("gate criteria and evidence must be distinct files")
    expected_evidence = TRAINING_GATE_EVIDENCE_SCHEMAS[gate]
    if evidence_names != tuple(name for name, _schema in expected_evidence):
        raise ValueError("gate evidence coverage or order differs from the frozen schema")

    reports: list[dict[str, str]] = []
    validated_evidence: list[tuple[str, bytes, dict[str, Any]]] = []
    for (name, path), resolved in zip(evidence, evidence_paths, strict=True):
        payload = _read_real_file(path, name=f"{name} evidence")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{name} evidence is not valid JSON") from error
        validate_training_gate_evidence(gate=gate, name=name, value=value)
        validated_evidence.append((name, payload, value))
        reports.append(
            {
                "name": name,
                "path": str(resolved),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    criteria_sha256 = hashlib.sha256(criteria_payload).hexdigest()
    subject = training_gate_decision_subject(
        gate=gate,
        criteria_sha256=criteria_sha256,
        evidence=tuple(validated_evidence),
    )
    return {
        "schema": TRAINING_GATE_DECISION_SCHEMA,
        "status": "PASS",
        "gate": gate,
        "decision_kind": decision_kind,
        "subject": subject,
        "reviewer": reviewer.strip(),
        "criteria": {
            "path": str(criteria_path),
            "sha256": criteria_sha256,
        },
        "evidence": reports,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        choices=tuple(TRAINING_GATE_DECISION_KINDS),
        required=True,
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    value = build_training_gate_decision(
        gate=args.gate,
        reviewer=args.reviewer,
        criteria=args.criteria,
        evidence=_parse_evidence(args.evidence),
    )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    _write_text_durable(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
                "gate": args.gate,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
