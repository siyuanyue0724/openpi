#!/usr/bin/env python3
"""Finalize M2 only after a run-bound review of every anchor overlay."""

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
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _canonical_sha256,
    _is_under_mnt,
    _sha256,
    _write_json_atomic,
    validate_m2_machine_decision,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    return parser.parse_args()


def validate_m2_visual_review(review: dict[str, Any], *, run_dir: Path) -> None:
    expected = {
        "schema",
        "status",
        "gate",
        "run_dir",
        "machine_decision_sha256",
        "visual_artifacts_sha256",
        "inspected_files",
        "reviewer",
        "findings",
        "physical_object_ownership_accepted",
        "multi_camera_accepted",
        "occlusion_cases_accepted",
        "fragmentation_accepted",
    }
    if set(review) != expected:
        raise ValueError("M2 visual review fields differ from schema")
    if (
        review["schema"] != "picf-next.molmoact2-m2-visual-review.v1"
        or review["gate"] != M2_GATE
        or review["status"] not in {"PASS", "FAIL"}
        or Path(review["run_dir"]).resolve() != run_dir
    ):
        raise ValueError("M2 visual review identity or status changed")
    if review["machine_decision_sha256"] != _sha256(run_dir / "machine_decision.json"):
        raise ValueError("M2 visual review is bound to a different machine decision")
    visual_path = run_dir / "visual_artifacts.json"
    if review["visual_artifacts_sha256"] != _sha256(visual_path):
        raise ValueError("M2 visual review is bound to different artifacts")
    manifest = json.loads(visual_path.read_text())
    if (
        manifest.get("schema") != "picf-next.molmoact2-m2-visual-artifacts.v1"
        or manifest.get("gate") != M2_GATE
        or manifest.get("all_splits_present") is not True
        or manifest.get("all_learned_segments_present") is not True
        or manifest.get("camera_views_per_artifact") != 2
    ):
        raise ValueError("M2 visual artifact coverage or identity changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("M2 visual artifact manifest is empty")
    if manifest.get("artifacts_sha256") != _canonical_sha256(artifacts):
        raise ValueError("M2 visual artifact manifest checksum changed")
    expected_files = []
    for row in artifacts:
        if not isinstance(row, dict):
            raise ValueError("M2 visual artifact row is malformed")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("M2 visual artifact path is unsafe")
        expected_files.append(relative)
    if len(set(expected_files)) != len(expected_files):
        raise ValueError("M2 visual artifact paths are not unique")
    if review["inspected_files"] != expected_files:
        raise ValueError("M2 visual review must inspect every artifact in manifest order")
    for row in artifacts:
        path = run_dir / row["path"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(f"M2 visual artifact changed: {row['path']}")
    if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
        raise ValueError("M2 visual reviewer cannot be empty")
    if (
        not isinstance(review["findings"], list)
        or not review["findings"]
        or any(not isinstance(value, str) or not value.strip() for value in review["findings"])
    ):
        raise ValueError("M2 visual review requires substantive findings")
    decisions = (
        "physical_object_ownership_accepted",
        "multi_camera_accepted",
        "occlusion_cases_accepted",
        "fragmentation_accepted",
    )
    if any(not isinstance(review[name], bool) for name in decisions):
        raise ValueError("M2 visual decisions must be boolean")
    if review["status"] == "PASS" and not all(review[name] for name in decisions):
        raise ValueError("M2 cannot pass while a visual criterion is rejected")


def finalize_m2(*, run_dir: Path, visual_review_path: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    visual_review_path = visual_review_path.expanduser().resolve()
    if not _is_under_mnt(run_dir):
        raise RuntimeError("M2 final evidence must persist under /mnt")
    if (run_dir / "gate_decision.json").exists():
        raise FileExistsError("M2 already has an immutable final gate decision")
    machine = validate_m2_machine_decision(run_dir)
    if machine["status"] != "PASS_PENDING_VISUAL_REVIEW":
        raise ValueError("failed M2 machine checks cannot be overridden by visual review")
    review = json.loads(visual_review_path.read_text())
    validate_m2_visual_review(review, run_dir=run_dir)
    destination = run_dir / "visual_review.json"
    if destination.exists():
        if json.loads(destination.read_text()) != review:
            raise FileExistsError("M2 run already contains a different visual review")
    else:
        _write_json_atomic(destination, review)
    required = dict(machine["required_report_sha256"])
    required.update(
        {
            "machine_decision.json": _sha256(run_dir / "machine_decision.json"),
            "visual_review.json": _sha256(destination),
        }
    )
    status = review["status"]
    decision = {
        "schema": "picf-next.molmoact2-m2-gate-decision.v1",
        "status": status,
        "gate": M2_GATE,
        "required_report_sha256": required,
        "later_gates_authorized": ["M3_structural_probe"] if status == "PASS" else [],
    }
    _write_json_atomic(run_dir / "gate_decision.json", decision)
    return decision


def main() -> None:
    args = _parse_args()
    decision = finalize_m2(
        run_dir=args.run_dir,
        visual_review_path=args.visual_review,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
