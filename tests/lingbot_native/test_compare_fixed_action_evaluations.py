from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "adr152" / "tools" / "compare_fixed_action_evaluations.py"


def _sample(ordinal: int, *, partition: str, action_loss: float) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "partition": partition,
        "sample_key": f"sample-{ordinal}",
        "task_key": f"task-{ordinal}",
        "segment_index": ordinal,
        "source_episode_index": ordinal + 10,
        "source_global_index": ordinal + 100,
        "transition_index": ordinal + 1_000,
        "source_digest": f"source-{ordinal}",
        "model_inputs_sha256": f"input-{ordinal}",
        "prior_control_chunk_count": 2,
        "action_loss": action_loss,
    }


def _write_evaluation(path: Path, *, offset: float = 0.0) -> None:
    samples = [
        _sample(0, partition="heldout", action_loss=0.4 + offset),
        _sample(1, partition="heldout", action_loss=0.6 + offset),
        _sample(2, partition="validation", action_loss=0.2 + offset),
        _sample(3, partition="validation", action_loss=0.8 + offset),
    ]
    path.write_text(
        json.dumps(
            {
                "schema": "picf-next.adr149-cold-action-snapshot/v1",
                "status": "PASS",
                "checkpoint_global_step": 20,
                "evaluation_input_sha256": "evaluation-input",
                "evaluation_plan_sha256": "evaluation-plan",
                "representation_split_sha256": "representation-split",
                "stream_plan_sha256": "stream-plan",
                "state_mode": "cold_reset",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )


def test_exact_action_comparison_reports_paired_change(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "report.json"
    _write_evaluation(reference)
    _write_evaluation(candidate, offset=-0.05)

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["status"] == "PASS"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pairing"]["sample_count"] == 4
    assert report["pairing"]["mismatch_count"] == 0
    assert report["comparisons"]["all"]["candidate_minus_reference_mean"] == pytest.approx(
        -0.05
    )
    assert report["comparisons"]["heldout"]["candidate_lower_fraction"] == 1.0
    assert report["comparisons"]["validation"]["sample_count"] == 2


def test_action_comparison_rejects_input_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "report.json"
    _write_evaluation(reference)
    _write_evaluation(candidate)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["samples"][0]["model_inputs_sha256"] = "changed"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "sample pairing changed" in completed.stderr
    assert not output.exists()
