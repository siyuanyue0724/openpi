from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_watch_metrics_renders_recent_summary(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 100,
                        "loss_total": 3.0,
                        "loss_action": 2.0,
                        "loss_pt": 0.30,
                        "tactile_active_rate": 0.10,
                        "tactile_contact_prob_mean": 0.50,
                    }
                ),
                json.dumps(
                    {
                        "step": 200,
                        "loss_total": 2.5,
                        "loss_action": 1.8,
                        "loss_pt": 0.25,
                        "tactile_active_rate": 0.05,
                        "tactile_contact_prob_mean": 0.45,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/picf_watch_metrics.py",
            str(metrics),
            "--window",
            "2",
            "--spark-width",
            "8",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    assert "step=200" in output
    assert "loss_total:" in output
    assert "loss_action:" in output
    assert "loss_pt:" in output


def test_watch_metrics_handles_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/picf_watch_metrics.py",
            str(missing),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Waiting for metrics file" in result.stdout
