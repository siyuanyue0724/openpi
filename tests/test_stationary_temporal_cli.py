# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tools.train_stationary_molmoact2_calvin_temporal import (
    _accelerator_runtime_kwargs,
    _reconcile_metrics,
    _reduce_diagnostic_totals,
    _validate_scheduler_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/train_stationary_molmoact2_calvin_temporal.py"


def test_stationary_temporal_cli_definition_is_local_and_long_train_closed() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--mode", "definition"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report["optimizer_steps"] == 200
    assert report["world_size"] == 2
    assert report["prefix_lengths"] == [0, 8, 32, 128]
    assert report["train_length"] == 2
    assert report["required_future_horizon"] == 2
    assert report["action_weight"] == 0.0
    assert report["long_training_authorized"] is False


def test_stationary_temporal_cli_bootstraps_its_src_layout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    shutil.copytree(ROOT / "src", checkout / "src")
    shutil.copytree(ROOT / "configs", checkout / "configs")
    (checkout / "tools").mkdir()
    shutil.copy2(TOOL, checkout / "tools" / TOOL.name)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(checkout / "tools" / TOOL.name), "--mode", "definition"],
        cwd=checkout,
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert json.loads(result.stdout)["optimizer_steps"] == 200


def test_stationary_resume_reconciles_only_records_newer_than_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "".join(
            json.dumps({"optimizer_step": step, "metrics": {"loss": 1.0}}) + "\n"
            for step in range(1, 4)
        )
    )
    _reconcile_metrics(path, completed_steps=2)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["optimizer_step"] for record in records] == [1, 2]


def test_stationary_resume_rejects_a_metrics_gap(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"optimizer_step": 2, "metrics": {"loss": 1.0}}) + "\n")
    with pytest.raises(ValueError, match="does not cover checkpoint progress exactly"):
        _reconcile_metrics(path, completed_steps=2)


def test_stationary_accelerator_keeps_scheduler_on_global_optimizer_clock() -> None:
    assert _accelerator_runtime_kwargs() == {
        "mixed_precision": "bf16",
        "gradient_accumulation_steps": 1,
        "step_scheduler_with_optimizer": False,
    }


def test_stationary_scheduler_epoch_must_equal_completed_global_steps() -> None:
    scheduler = type(
        "Scheduler",
        (),
        {"state_dict": lambda self: {"last_epoch": 17}},
    )()
    assert _validate_scheduler_epoch(scheduler, completed_steps=17) == 17
    with pytest.raises(RuntimeError, match="scheduler=17, optimizer=9"):
        _validate_scheduler_epoch(scheduler, completed_steps=9)


def test_stationary_diagnostics_are_global_totals_with_stable_names() -> None:
    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def reduce(value: torch.Tensor, *, reduction: str) -> torch.Tensor:
            assert reduction == "sum"
            return value * 2.0

    reduced = _reduce_diagnostic_totals(
        Accelerator(),
        {
            "lifecycle_detection_positive_target_mass": 3.5,
            "lifecycle_detection_negative_target_mass": 1,
        },
    )

    assert reduced == {
        "picf_lifecycle_detection_positive_target_mass": 7.0,
        "picf_lifecycle_detection_negative_target_mass": 2.0,
    }


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf")])
def test_stationary_diagnostics_reject_malformed_values(value: object) -> None:
    class Accelerator:
        device = torch.device("cpu")

    with pytest.raises(ValueError, match="diagnostic is malformed"):
        _reduce_diagnostic_totals(Accelerator(), {"bad": value})  # type: ignore[dict-item]
