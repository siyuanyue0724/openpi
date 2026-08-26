from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "adr152" / "tools" / "compare_posterior_dose_training_windows.py"


def _step(step: int, *, high_dose: bool, entity_scale: float) -> dict[str, Any]:
    factual = 0.5 / step
    return {
        "global_step": step,
        "sample_keys": [f"sample-{step}"],
        "frame_indices": [step - 1],
        "lane_ids": [0],
        "augmentation_seeds": [100 + step],
        "flow_noise_seeds": [200 + step],
        "flow_timestep_seeds": [300 + step],
        "optimizer_lags": [step - 1],
        "local_bptt_steps": 1,
        "overshoot_horizon": 0,
        "reset": [step == 1],
        "source_digest": f"source-{step}",
        "causal_ablation_mode": "none",
        "posterior_input_mode": "causal_lane",
        "source_masked_branch": high_dose,
        "omitted_static_branch": high_dose,
        "omitted_static_action_branch": high_dose,
        "omitted_static_action_loss": factual + 0.02 if high_dose else None,
        "temporal_plan_sha256": "high-dose" if high_dose else "regular-dose",
        "official_action_loss": factual,
        "normalized_terms": {
            "set/frame_000/entities": 4.0 * entity_scale / step,
            "set/frame_000/existence_focal": 0.1,
            "set/frame_000/mask_dice": 0.8 * entity_scale,
            "set/frame_000/mask_focal": 1.0 * entity_scale,
            "set/frame_000/ownership_nll": 2.0 * entity_scale,
        },
        "family_terms": {"predictive": 0.01 * entity_scale},
    }


def _write(path: Path, *, high_dose: bool, entity_scale: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "rank_reports": [
                    {
                        "rank": rank,
                        "steps": [
                            _step(step, high_dose=high_dose, entity_scale=entity_scale)
                            for step in (1, 2)
                        ],
                    }
                    for rank in (0, 1)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_compares_route_dose_on_exact_stochastic_stream(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "report.json"
    _write(baseline, high_dose=False)
    _write(candidate, high_dose=True, entity_scale=0.5)

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--window-boundary",
            "1",
            "--window-boundary",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["status"] == "PASS"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pairing"]["mismatch_count"] == 0
    assert report["route_dose"]["baseline"]["source_masked_fraction"] == 0
    assert report["route_dose"]["candidate"]["source_masked_fraction"] == 1
    assert report["route_dose"]["candidate_routed_minus_factual_action"][
        "mean"
    ] == pytest.approx(0.02)
    assert report["overall_losses"]["entity_total"][
        "relative_change_percent"
    ] == pytest.approx(-50)
    assert len(report["windows"]) == 2


def test_rejects_changed_training_sample(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "report.json"
    _write(baseline, high_dose=False)
    _write(candidate, high_dose=True)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["rank_reports"][0]["steps"][0]["augmentation_seeds"] = [999]
    candidate.write_text(json.dumps(payload), encoding="utf-8")

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
    assert "paired stochastic stream mismatch" in completed.stderr
    assert not output.exists()
