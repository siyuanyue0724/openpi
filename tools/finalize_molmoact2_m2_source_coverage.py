#!/usr/bin/env python3
"""Finalize all-source M2 through training and external visual review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from picf_next.training.molmoact2_m2 import M2_GATE  # noqa: E402
from picf_next.training.molmoact2_m2_source_coverage import (  # noqa: E402
    M2_SOURCE_COVERAGE_GATE,
)
from tools import audit_molmoact2_m2_source_coverage_external as external  # noqa: E402
from tools import run_molmoact2_m2_source_coverage_cloud as source_runner  # noqa: E402
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _canonical_sha256,
    _is_under_mnt,
    _sha256,
    _write_json_atomic,
)

_REVIEW_SCHEMA = "picf-next.molmoact2-m2-source-coverage-visual-review.v1"
_VISUAL_DECISIONS = (
    "physical_object_ownership_accepted",
    "multi_camera_accepted",
    "occlusion_cases_accepted",
    "fragmentation_accepted",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser(
        "training",
        help="bind review of every training-run visual before external evaluation",
    )
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--visual-review", type=Path, required=True)
    final = subparsers.add_parser(
        "external",
        help="bind external visual review and issue the immutable M2 gate decision",
    )
    final.add_argument("--training-run", type=Path, required=True)
    final.add_argument("--external-run", type=Path, required=True)
    final.add_argument("--visual-review", type=Path, required=True)
    return parser.parse_args()


def validate_source_coverage_visual_review(
    review: dict[str, Any],
    *,
    run_dir: Path,
    stage: str,
) -> None:
    expected = {
        "schema",
        "stage",
        "status",
        "gate",
        "run_dir",
        "machine_decision_sha256",
        "visual_artifacts_sha256",
        "inspected_files",
        "reviewer",
        "findings",
        *_VISUAL_DECISIONS,
    }
    if set(review) != expected:
        raise ValueError("M2 source-coverage visual review fields differ from schema")
    if stage not in {"training", "external"}:
        raise ValueError("M2 source-coverage visual stage is invalid")
    if (
        review["schema"] != _REVIEW_SCHEMA
        or review["stage"] != stage
        or review["gate"] != M2_SOURCE_COVERAGE_GATE
        or review["status"] not in {"PASS", "FAIL"}
        or Path(review["run_dir"]).expanduser().resolve() != run_dir
    ):
        raise ValueError("M2 source-coverage visual review identity changed")
    if review["machine_decision_sha256"] != _sha256(run_dir / "machine_decision.json"):
        raise ValueError("M2 source-coverage review binds another machine decision")
    visual_path = run_dir / "visual_artifacts.json"
    if review["visual_artifacts_sha256"] != _sha256(visual_path):
        raise ValueError("M2 source-coverage review binds different visual artifacts")
    manifest = json.loads(visual_path.read_text())
    if (
        manifest.get("schema") != "picf-next.molmoact2-m2-visual-artifacts.v1"
        or manifest.get("gate") != M2_SOURCE_COVERAGE_GATE
        or manifest.get("all_splits_present") is not True
        or manifest.get("all_learned_segments_present") is not True
        or manifest.get("camera_views_per_artifact") != 2
    ):
        raise ValueError("M2 source-coverage visual artifact coverage changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("M2 source-coverage visual artifact manifest is empty")
    if manifest.get("artifacts_sha256") != _canonical_sha256(artifacts):
        raise ValueError("M2 source-coverage visual artifact checksum changed")
    expected_files = []
    for row in artifacts:
        if not isinstance(row, dict):
            raise ValueError("M2 source-coverage visual artifact row is malformed")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("M2 source-coverage visual artifact path is unsafe")
        expected_files.append(relative)
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            raise ValueError(f"M2 source-coverage visual artifact changed: {relative}")
    if len(set(expected_files)) != len(expected_files):
        raise ValueError("M2 source-coverage visual artifact paths are not unique")
    if review["inspected_files"] != expected_files:
        raise ValueError("M2 source-coverage review must inspect every artifact in order")
    if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
        raise ValueError("M2 source-coverage visual reviewer cannot be empty")
    if (
        not isinstance(review["findings"], list)
        or not review["findings"]
        or any(not isinstance(item, str) or not item.strip() for item in review["findings"])
    ):
        raise ValueError("M2 source-coverage review requires substantive findings")
    if any(not isinstance(review[name], bool) for name in _VISUAL_DECISIONS):
        raise ValueError("M2 source-coverage visual decisions must be boolean")
    if review["status"] == "PASS" and not all(review[name] for name in _VISUAL_DECISIONS):
        raise ValueError("M2 source-coverage visuals cannot pass a rejected criterion")


def finalize_training_visuals(
    *,
    run_dir: Path,
    visual_review_path: Path,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    visual_review_path = visual_review_path.expanduser().resolve()
    if not _is_under_mnt(run_dir):
        raise RuntimeError("M2 source-coverage evidence must persist under /mnt")
    decision_path = run_dir / "training_visual_decision.json"
    if decision_path.exists():
        raise FileExistsError("M2 source-coverage training visuals are already final")
    machine = source_runner.validate_source_coverage_machine_decision(run_dir)
    if machine["status"] != "PASS_PENDING_VISUAL_REVIEW":
        raise ValueError("failed source machine checks cannot enter visual review")
    review = json.loads(visual_review_path.read_text())
    validate_source_coverage_visual_review(review, run_dir=run_dir, stage="training")
    destination = run_dir / "training_visual_review.json"
    if destination.exists():
        if json.loads(destination.read_text()) != review:
            raise FileExistsError("training run already contains another visual review")
    else:
        _write_json_atomic(destination, review)
    status = review["status"]
    decision = {
        "schema": ("picf-next.molmoact2-m2-source-coverage-training-visual-decision.v1"),
        "gate": M2_SOURCE_COVERAGE_GATE,
        "status": status,
        "required_report_sha256": {
            "machine_decision.json": _sha256(run_dir / "machine_decision.json"),
            "training_visual_review.json": _sha256(destination),
        },
        "external_validation_authorized": status == "PASS",
        "later_gates_authorized": [],
    }
    _write_json_atomic(decision_path, decision)
    return decision


def finalize_external_visuals(
    *,
    training_run: Path,
    external_run: Path,
    visual_review_path: Path,
) -> dict[str, Any]:
    training_run = training_run.expanduser().resolve()
    external_run = external_run.expanduser().resolve()
    visual_review_path = visual_review_path.expanduser().resolve()
    if not _is_under_mnt(training_run) or not _is_under_mnt(external_run):
        raise RuntimeError("M2 source-coverage evidence must persist under /mnt")
    if external_run != training_run / "external_validation":
        raise ValueError("external M2 run is not uniquely bound to its training run")
    final_path = training_run / "gate_decision.json"
    external_final_path = external_run / "gate_decision.json"
    if final_path.exists():
        raise FileExistsError("M2 source-coverage external review is already final")
    training_visual = source_runner.validate_source_coverage_training_visual_decision(training_run)
    if training_visual["status"] != "PASS":
        raise ValueError("failed training visuals cannot enter external finalization")
    machine = external.validate_external_machine_decision(
        external_run,
        training_run=training_run,
    )
    if machine["status"] != "PASS_PENDING_VISUAL_REVIEW":
        raise ValueError("failed external machine checks cannot be overridden")
    review = json.loads(visual_review_path.read_text())
    validate_source_coverage_visual_review(review, run_dir=external_run, stage="external")
    destination = external_run / "visual_review.json"
    if destination.exists():
        if json.loads(destination.read_text()) != review:
            raise FileExistsError("external run already contains another visual review")
    else:
        _write_json_atomic(destination, review)
    status = review["status"]
    external_decision = {
        "schema": "picf-next.molmoact2-m2-source-coverage-external-gate-decision.v1",
        "gate": M2_SOURCE_COVERAGE_GATE,
        "status": status,
        "required_report_sha256": {
            "machine_decision.json": _sha256(external_run / "machine_decision.json"),
            "visual_review.json": _sha256(destination),
        },
        "later_gates_authorized": [],
    }
    if external_final_path.exists():
        if json.loads(external_final_path.read_text()) != external_decision:
            raise FileExistsError("external run already contains another gate decision")
    else:
        _write_json_atomic(external_final_path, external_decision)
    final_decision = {
        "schema": "picf-next.molmoact2-m2-source-coverage-gate-decision.v1",
        "gate": M2_SOURCE_COVERAGE_GATE,
        "base_gate": M2_GATE,
        "status": status,
        "required_report_sha256": {
            "training_visual_decision.json": _sha256(
                training_run / "training_visual_decision.json"
            ),
            "external_validation/gate_decision.json": _sha256(external_final_path),
        },
        "later_gates_authorized": ["M3_structural_probe"] if status == "PASS" else [],
    }
    _write_json_atomic(final_path, final_decision)
    return final_decision


def main() -> None:
    args = _parse_args()
    if args.command == "training":
        decision = finalize_training_visuals(
            run_dir=args.run_dir,
            visual_review_path=args.visual_review,
        )
    else:
        decision = finalize_external_visuals(
            training_run=args.training_run,
            external_run=args.external_run,
            visual_review_path=args.visual_review,
        )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
