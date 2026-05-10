from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_plot_metrics_writes_png(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    rows = []
    for step in range(1, 61):
        rows.append(
            json.dumps(
                {
                    "step": step * 100,
                    "loss_total": 4.0 - 0.01 * step,
                    "loss_action": 2.8 - 0.008 * step,
                    "loss_alignment": 0.8 - 0.002 * step,
                    "loss_pt": 0.7 - 0.005 * step,
                    "loss_action_pos": 0.2,
                    "loss_action_rot": 0.15,
                    "loss_action_gripper": 0.9,
                    "loss_visual_real": 0.3,
                    "loss_visual_latent": 0.1,
                    "loss_tactile_real": 0.2,
                    "loss_point_real": 0.4,
                    "loss_anchor_pv": 1.3,
                    "loss_pv_weak": 3.1,
                    "loss_semantic_future_aux": 0.0,
                    "tactile_active_rate": 0.1,
                    "tactile_contact_prob_mean": 0.6,
                    "projective_candidate_density": 0.03,
                    "steps_per_sec": 0.2,
                }
            )
        )
    metrics.write_text("\n".join(rows), encoding="utf-8")
    output = tmp_path / "trend.png"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/picf_plot_metrics.py",
            str(metrics),
            "--output",
            str(output),
            "--smoothing-window",
            "50",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert "saved_plot=" in result.stdout
