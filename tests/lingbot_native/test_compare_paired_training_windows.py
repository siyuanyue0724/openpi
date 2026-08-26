from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "adr141" / "tools" / "compare_paired_training_windows.py"


def _step(global_step: int, action: float, entity: float) -> dict[str, Any]:
    return {
        "global_step": global_step,
        "sample_keys": [f"sample-{global_step}"],
        "frame_indices": [global_step - 1],
        "lane_ids": [0],
        "state_ages": [global_step - 1],
        "local_bptt_steps": 1,
        "overshoot_horizon": 0,
        "source_masked_branch": False,
        "omitted_static_branch": False,
        "temporal_plan_sha256": f"plan-{global_step}",
        "official_action_loss": action,
        "normalized_terms": {
            "set/frame_000/entities": entity,
            "set/frame_000/existence_focal": 0.1,
            "set/frame_000/mask_dice": entity / 2,
            "set/frame_000/mask_focal": entity / 4,
            "set/frame_000/ownership_nll": entity / 3,
        },
        "family_terms": {"predictive": 0.01},
        "gradient_metrics": {
            "action_output_norm": 1.0,
            "native_graph_norm": 2.0,
            "predictive_readout_norm": 0.5,
            "relation_projection_norm": 1.5,
            "preclip_global_norm": 3.0,
            "all_finite": True,
        },
    }


def _write_metric(path: Path, *, entity_scale: float = 1.0) -> None:
    payload = {
        "rank_reports": [
            {
                "rank": rank,
                "steps": [
                    _step(step, action=0.5 / step, entity=entity_scale * (3.0 / step))
                    for step in (1, 2)
                ],
            }
            for rank in (0, 1)
        ]
    }
    path.write_text(json.dumps(payload))


def test_exact_stream_comparison_reports_paired_improvement(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    _write_metric(baseline)
    _write_metric(candidate, entity_scale=0.5)

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--window-size",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["status"] == "PASS"
    report = json.loads(output.read_text())
    assert report["pairing"]["record_count"] == 4
    assert report["pairing"]["mismatch_count"] == 0
    assert report["overall_losses"]["action"]["relative_change_percent"] == 0
    assert report["overall_losses"]["entity_total"]["relative_change_percent"] == -50
    assert report["overall_losses"]["entity_total"]["candidate_median"] == 1.125
    assert report["state_age_strata"]["reset"]["record_count"] == 2
    assert report["state_age_strata"]["continuation"]["record_count"] == 2
    assert len(report["windows"]) == 2


def test_comparison_rejects_nonidentical_stream(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    _write_metric(baseline)
    _write_metric(candidate)
    payload = json.loads(candidate.read_text())
    payload["rank_reports"][0]["steps"][0]["sample_keys"] = ["different"]
    candidate.write_text(json.dumps(payload))

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--baseline",
            str(baseline),
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
    assert "paired stream mismatch" in completed.stderr
    assert not output.exists()
