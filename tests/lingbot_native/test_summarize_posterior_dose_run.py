from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "adr152" / "tools" / "summarize_posterior_dose_run.py"


def _record(step: int, *, rank: int) -> dict[str, Any]:
    factual = 0.5 / step + rank * 0.01
    routed = factual + 0.02
    return {
        "global_step": step,
        "source_masked_branch": True,
        "omitted_static_branch": True,
        "official_action_loss": factual,
        "omitted_static_action_loss": routed,
        "effective_training_action_loss": (factual + routed) / 2,
        "step_time_s": 10.0 + step,
        "peak_cuda_reserved_bytes": 2**30 * (20 + rank),
        "gradient_metrics": {
            "all_finite": True,
            "preclip_global_norm": 2.0 + step,
        },
    }


def _write_run(run_dir: Path, *, omit_last: bool = False) -> None:
    (run_dir / "metrics" / "rank_journal").mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"execution_contract": {"acceptance_mode": "posterior-adoption-dose"}}),
        encoding="utf-8",
    )
    for rank in (0, 1):
        records = [_record(step, rank=rank) for step in range(1, 5)]
        if omit_last and rank == 1:
            records.pop()
        (run_dir / "metrics" / "rank_journal" / f"rank_{rank}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )


def test_complete_dose_ledger_reports_three_action_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output = tmp_path / "summary.json"
    _write_run(run_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--run-dir",
            str(run_dir),
            "--stop-step",
            "4",
            "--window-boundary",
            "2",
            "--window-boundary",
            "4",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["status"] == "PASS"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["contract"]["rank_step_record_count"] == 8
    assert report["contract"]["routed_full_step_equivalents"] == 2
    assert report["overall_action"]["routed_minus_factual"]["mean"] == pytest.approx(
        0.02
    )
    assert len(report["windows"]) == 2
    assert report["runtime"]["maximum_peak_reserved_gib"] == 21


def test_dose_ledger_rejects_missing_rank_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output = tmp_path / "summary.json"
    _write_run(run_dir, omit_last=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--run-dir",
            str(run_dir),
            "--stop-step",
            "4",
            "--window-boundary",
            "4",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "dose ledger is incomplete" in completed.stderr
    assert not output.exists()
