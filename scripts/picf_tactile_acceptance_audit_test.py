from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_tactile_acceptance_audit_warns_for_subideal_front_ratio(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {
                "tau_on": 0.23,
                "tau_off": 0.22,
                "active_rate_tau_on": 0.01,
                "negative_active_rate_tau_on": 0.0,
                "negative_pool_size": 32,
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "d_nn_trimmed_mean": 0.01,
                "front_ratio": 0.55,
                "recommended_pt_bag_radius_m": 0.045,
            }
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "loss_total": 3.0,
                        "loss_action": 2.0,
                        "loss_pt": 0.7,
                        "tactile_contact_prob_mean": 0.8,
                        "tactile_active_rate": 0.4,
                    }
                ),
                json.dumps(
                    {
                        "loss_total": 4.0,
                        "loss_action": 2.8,
                        "loss_pt": 0.69,
                        "tactile_contact_prob_mean": 0.75,
                        "tactile_active_rate": 0.5,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/picf_tactile_acceptance_audit.py",
            "--contact-stats",
            str(stats),
            "--fingertip-calibration",
            str(calibration),
            "--metrics",
            str(metrics),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["overall_status"] == "warn"
    assert any(check["name"] == "fingertip_front_ratio" and check["status"] == "warn" for check in payload["checks"])


def test_tactile_acceptance_audit_fails_for_invalid_threshold_order(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {
                "tau_on": 0.2,
                "tau_off": 0.22,
                "active_rate_tau_on": 0.01,
                "negative_active_rate_tau_on": 0.1,
                "negative_pool_size": 32,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/picf_tactile_acceptance_audit.py",
            "--contact-stats",
            str(stats),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["overall_status"] == "fail"
