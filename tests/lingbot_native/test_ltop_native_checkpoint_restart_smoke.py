from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _lingbot_source_checkout(repository_root: Path) -> Path:
    configured = os.environ.get("PICF_LINGBOT_NATIVE_SOURCE")
    candidates = [
        Path(configured) if configured else None,
        repository_root.parent
        / "remote_audit/adr152/upstream/lingbot-vla-v2",
    ]
    for candidate in candidates:
        if candidate is not None and (
            candidate / "lingbotvla/checkpoint/checkpointer.py"
        ).is_file():
            return candidate.resolve()
    pytest.skip("a LingBot source checkout is required for the native DCP integration smoke")


def test_ltop_native_dcp_survives_a_real_cold_process_restart(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source_checkout = _lingbot_source_checkout(repository_root)
    runner = repository_root / "tools/smoke_ltop_native_checkpoint_resume.py"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(runner),
    ]
    for phase in ("fresh", "resume"):
        completed = subprocess.run(
            [
                *command,
                "--phase",
                phase,
                "--source-checkout",
                str(source_checkout),
                "--run-dir",
                str(tmp_path),
            ],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    receipt = json.loads((tmp_path / "cold_resume_receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["global_step"] == 2
    assert receipt["continued_global_step"] == 3
    assert [value["rank"] for value in receipt["rank_loads"]] == [0, 1]
    assert all(value["runtime_rng_verified"] for value in receipt["rank_loads"])
    assert all(value["continued_global_step"] == 3 for value in receipt["rank_loads"])
    assert all(
        value["optimizer_state"]["optimizer_state_entries"] == 4
        for value in receipt["rank_loads"]
    )
    assert all(
        value["step3_optimizer_state"]["optimizer_state_entries"] == 4
        for value in receipt["rank_loads"]
    )
    assert all(len(value["step3_input_sha256"]) == 64 for value in receipt["rank_loads"])
