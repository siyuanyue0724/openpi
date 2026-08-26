from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "adr147" / "compare_matched_lbot_action_prefix.py"
PLAN_SHA = "a" * 64
PAIR_FIELDS = (
    "sample_keys",
    "frame_indices",
    "lane_ids",
    "reset",
    "source_digest",
    "augmentation_seeds",
    "flow_noise_seeds",
    "flow_timestep_seeds",
)


def _records(*, lbot: bool, mismatch: bool = False) -> list[dict[str, object]]:
    reports = []
    for rank in range(4):
        steps = []
        for step in range(1, 5):
            sample = f"sample-{rank}-{step}"
            if mismatch and rank == 2 and step == 3:
                sample = "wrong"
            item: dict[str, object] = {
                "global_step": step,
                "sample_keys": [sample],
                "frame_indices": [step - 1],
                "lane_ids": [rank],
                "reset": [step == 1],
                "source_digest": f"source-{rank}-{step}",
                "augmentation_seeds": [1000 + rank * 10 + step],
                "flow_noise_seeds": [2000 + rank * 10 + step],
                "flow_timestep_seeds": [3000 + rank * 10 + step],
            }
            loss_field = "action_loss" if lbot else "official_action_loss"
            item[loss_field] = 1.0 / step if lbot else 0.8 / step
            steps.append(item)
        reports.append({"rank": rank, "steps": steps})
    return reports


def _write_inputs(
    tmp_path: Path,
    *,
    mismatch: bool = False,
    baseline_schema: str = "picf-next.lingbot-vla2-official-calvin-lbot.v1",
) -> tuple[Path, Path]:
    baseline = tmp_path / "lbot.json"
    baseline.write_text(
        json.dumps(
            {
                "schema": baseline_schema,
                "status": "PASS",
                "picf_graph_installed": False,
                "world_size": 4,
                "steps": 4,
                "seed": 7,
                "max_grad_norm": 1.0,
                "optimizer_contract": {"learning_rate": 1e-4},
                "plan_sha256": PLAN_SHA,
                "rank_reports": _records(lbot=True),
            }
        )
    )
    run = tmp_path / "candidate"
    (run / "metrics").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "world_size": 4,
                "stream_plan_sha256": PLAN_SHA,
                "execution_contract": {
                    "seed": 7,
                    "learning_rate": (1e-4).hex(),
                    "max_grad_norm": (1.0).hex(),
                },
            }
        )
    )
    (run / "metrics" / "steps_00000001_00000004.json").write_text(
        json.dumps(
            {
                "schema": "picf-next.task-independent-full-metrics/v1",
                "start_global_step": 1,
                "end_global_step": 4,
                "rank_reports": _records(lbot=False, mismatch=mismatch),
            }
        )
    )
    return baseline, run


def test_exact_four_rank_prefix_produces_action_only_comparison(tmp_path: Path) -> None:
    baseline, run = _write_inputs(tmp_path)
    output = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--baseline-report",
            str(baseline),
            "--candidate-run-dir",
            str(run),
            "--steps",
            "4",
            "--window-size",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["status"] == "PASS"
    report = json.loads(output.read_text())
    assert report["contract"]["record_count"] == 16
    assert report["contract"]["pair_mismatch_count"] == 0
    assert report["overall_action"]["relative_change_percent"] == pytest.approx(-20.0)
    assert len(report["windows"]) == 2


def test_legacy_control_report_remains_read_only_compatible(tmp_path: Path) -> None:
    baseline, run = _write_inputs(
        tmp_path,
        baseline_schema="picf-next.lingbot-vla2-official-calvin-p0.v1",
    )
    output = tmp_path / "comparison.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--baseline-report",
            str(baseline),
            "--candidate-run-dir",
            str(run),
            "--steps",
            "4",
            "--window-size",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["status"] == "PASS"


def test_comparison_fails_closed_on_one_stream_mismatch(tmp_path: Path) -> None:
    baseline, run = _write_inputs(tmp_path, mismatch=True)
    output = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--baseline-report",
            str(baseline),
            "--candidate-run-dir",
            str(run),
            "--steps",
            "4",
            "--window-size",
            "2",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "paired action stream mismatch" in completed.stderr
    assert not output.exists()


def test_both_runners_publish_every_exact_pair_field() -> None:
    baseline = (ROOT / "tools" / "run_lingbot_vla2_official_lbot.py").read_text()
    candidate = (ROOT / "tools" / "run_lingbot_vla2_task_independent_full.py").read_text()

    for field in PAIR_FIELDS:
        assert f'"{field}"' in baseline
        assert f'"{field}"' in candidate
