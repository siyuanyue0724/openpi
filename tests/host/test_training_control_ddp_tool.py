from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_two_process_training_control_smoke(tmp_path: Path) -> None:
    if importlib.util.find_spec("accelerate") is None:
        pytest.skip("accelerate is not installed")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "ddp-smoke"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment["OMP_NUM_THREADS"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(root / "tools/smoke_training_control_ddp.py"),
            "--output-dir",
            str(output),
        ],
        cwd=root,
        env=environment,
        check=True,
        timeout=120,
    )
    report = json.loads((output / "report.json").read_text())
    assert report["schema"] == "picf-next.training-control-ddp-smoke.v1"
    assert report["world_size"] == 2
    assert report["rank_partition_exact"] is True
    assert report["checkpoint_resume_exact"] is True
    assert report["rank_local_checkpoint_state_exact"] is True
    assert report["single_process_gradient_equivalent"] is True
    assert report["checkpoint_collision_failed_closed"] is True
    assert report["invalid_manifest_failed_closed"] is True
    assert report["nonfinite_loss_failed_closed"] is True
    assert report["rank_local_forward_error_failed_closed"] is True
    assert report["rank_local_prepare_error_failed_closed"] is True
    assert report["rank_local_update_error_failed_closed"] is True
