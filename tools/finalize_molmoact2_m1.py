#!/usr/bin/env python3
"""Finalize M1 only after an immutable, run-bound visual review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.run_molmoact2_m1_cloud import (  # noqa: E402
    _is_under_mnt,
    _sha256,
    _write_json_atomic,
    validate_m1_machine_decision,
    validate_m1_visual_review,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    return parser.parse_args()


def finalize_m1(*, run_dir: Path, visual_review_path: Path) -> dict:
    run_dir = run_dir.expanduser().resolve()
    visual_review_path = visual_review_path.expanduser().resolve()
    if not _is_under_mnt(run_dir):
        raise RuntimeError(f"M1 final evidence must be persisted under /mnt: {run_dir}")
    if (run_dir / "gate_decision.json").exists():
        raise FileExistsError("M1 already has an immutable final gate decision")
    machine_decision = validate_m1_machine_decision(run_dir)
    review = json.loads(visual_review_path.read_text())
    validate_m1_visual_review(review, run_dir=run_dir)
    review_destination = run_dir / "visual_review.json"
    if review_destination.exists():
        if json.loads(review_destination.read_text()) != review:
            raise FileExistsError("M1 run contains a different immutable visual review")
    else:
        _write_json_atomic(review_destination, review)

    status = review["status"]
    required_hashes = dict(machine_decision["required_report_sha256"])
    required_hashes.update(
        {
            "machine_decision.json": _sha256(run_dir / "machine_decision.json"),
            "visual_review.json": _sha256(run_dir / "visual_review.json"),
        }
    )
    decision = {
        "schema": "picf-next.molmoact2-m1-gate-decision.v1",
        "status": status,
        "gate": "M1_typed_full_manifest",
        "required_report_sha256": required_hashes,
        "later_gates_authorized": ["M2_representation_smoke"] if status == "PASS" else [],
    }
    _write_json_atomic(run_dir / "gate_decision.json", decision)
    return decision


def main() -> None:
    args = _parse_args()
    decision = finalize_m1(
        run_dir=args.run_dir,
        visual_review_path=args.visual_review,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
