from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.build_lingbot_representation_natural_prompt_launch import (
    INTERVENTION_OPTIONS,
    derive_natural_prompt_launch,
    parse_launch_text,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/build_lingbot_representation_natural_prompt_launch.py"
COMMITTED_LAUNCH = ROOT / "configs/cloud/adr120_host_match_natural_prompt_arm_n.sh"
COMMITTED_REPORT = (
    ROOT / "configs/cloud/adr120_host_match_natural_prompt_arm_n.launch-contract.json"
)


def _baseline_launch(*, gradient_audits: str = "9,17,20,50,100,200") -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
RUN_DIR='/mnt/picf-next/runs/adr117'
LOG='/mnt/picf-next/logs/adr117.log'
echo $$ > "$RUN_DIR/launcher.pid"
exec env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_ALLOC_CONF \\
  PYTHONPATH=/mnt/picf-next/adr117/src:/mnt/picf-next/adr117 \\
  CUDA_VISIBLE_DEVICES=0,1 \\
  /opt/picf/bin/python -m torch.distributed.run \\
  --standalone --nproc_per_node=2 \\
  /mnt/picf-next/adr117/tools/run_lingbot_vla2_native_full.py \\
  --phase fresh \\
  --training-stage representation \\
  --checkpoint-publication never \\
  --representation-split /mnt/picf-next/split.json \\
  --representation-task-intervention-plan /mnt/picf-next/donor.json \\
  --representation-task-intervention-plan-sha256 {"a" * 64} \\
  --representation-evaluation-steps 0,200 \\
  --run-dir "$RUN_DIR" \\
  --load-global-step 0 \\
  --invocation-steps 200 \\
  --total-planned-steps 200 \\
  --seed 20260721 \\
  --capacity 16 \\
  --maximum-optimizer-lag 8 \\
  --lane-interleave-factor 8 \\
  --gradient-audit-steps {gradient_audits} \\
  --visual-audit-every 1 \\
  >"$LOG" 2>&1
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_arm_n_launch_removes_only_donor_intervention_and_round_trips() -> None:
    baseline_text = _baseline_launch()
    candidate_text, report = derive_natural_prompt_launch(
        baseline_text,
        baseline_sha256=_sha256(baseline_text),
        run_dir="/mnt/picf-next/runs/adr120-arm-n",
        log="/mnt/picf-next/logs/adr120-arm-n.log",
    )
    baseline = parse_launch_text(baseline_text)
    candidate = parse_launch_text(candidate_text)

    assert candidate.prefix == baseline.prefix
    assert candidate.runner == baseline.runner
    assert candidate.option_pairs == tuple(
        pair for pair in baseline.option_pairs if pair[0] not in INTERVENTION_OPTIONS
    )
    assert candidate.options["--gradient-audit-steps"] == "9,17,20,50,100,200"
    assert candidate.options["--visual-audit-every"] == "1"
    assert report["other_runner_delta_count"] == 0
    assert report["training_state_delta"] == ("donor_intervention_absent_use_natural_source_prompt")


def test_arm_n_launch_rejects_hash_drift_and_diagnostic_schedule_drift() -> None:
    baseline_text = _baseline_launch()
    kwargs = {
        "run_dir": "/mnt/picf-next/runs/adr120-arm-n",
        "log": "/mnt/picf-next/logs/adr120-arm-n.log",
    }
    with pytest.raises(ValueError, match="SHA-256 differs"):
        derive_natural_prompt_launch(
            baseline_text,
            baseline_sha256="0" * 64,
            **kwargs,
        )

    changed = _baseline_launch(gradient_audits="20,100,200")
    with pytest.raises(ValueError, match="gradient-audit-steps differs"):
        derive_natural_prompt_launch(
            changed,
            baseline_sha256=_sha256(changed),
            **kwargs,
        )

    with pytest.raises(ValueError, match="canonical persistent"):
        derive_natural_prompt_launch(
            baseline_text,
            baseline_sha256=_sha256(baseline_text),
            run_dir="/mnt/picf-next/runs/../adr120-arm-n",
            log=kwargs["log"],
        )
    with pytest.raises(ValueError, match="canonical persistent"):
        derive_natural_prompt_launch(
            baseline_text,
            baseline_sha256=_sha256(baseline_text),
            run_dir=kwargs["run_dir"],
            log="/mnt/picf-next/runs/adr120-arm-n.log",
        )


def test_arm_n_launch_cli_writes_executable_fail_closed_artifacts(tmp_path: Path) -> None:
    baseline_text = _baseline_launch()
    baseline = tmp_path / "baseline.sh"
    output = tmp_path / "candidate.sh"
    report = tmp_path / "candidate.json"
    baseline.write_text(baseline_text, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--baseline-launch",
            str(baseline),
            "--baseline-launch-sha256",
            _sha256(baseline_text),
            "--run-dir",
            "/mnt/picf-next/runs/adr120-arm-n",
            "--log",
            "/mnt/picf-next/logs/adr120-arm-n.log",
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.stat().st_mode & 0o111
    assert subprocess.run(["bash", "-n", str(output)], check=False).returncode == 0
    assert '"status": "PASS"' in report.read_text(encoding="utf-8")

    repeated = subprocess.run(
        completed.args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr


def test_committed_arm_n_launch_is_bound_to_its_reviewed_contract() -> None:
    launch_text = COMMITTED_LAUNCH.read_text(encoding="utf-8")
    report = json.loads(COMMITTED_REPORT.read_text(encoding="utf-8"))
    launch = parse_launch_text(launch_text)

    assert COMMITTED_LAUNCH.stat().st_mode & 0o111 == 0
    assert _sha256(launch_text) == report["candidate_launch_sha256"]
    assert report["baseline_launch_sha256"] == (
        "b5dd6223d45bbef64cc9c56e7b63a9000ca3c4b89401863f48417497d572eeef"
    )
    assert report["other_runner_delta_count"] == 0
    assert report["unchanged_runner_option_count"] == len(launch.option_pairs) == 54
    assert report["candidate_run_dir"] == launch.run_dir
    assert report["candidate_log"] == launch.log
    assert not set(INTERVENTION_OPTIONS) & set(launch.options)
    assert launch.options["--gradient-audit-steps"] == "9,17,20,50,100,200"
    assert launch.options["--visual-audit-every"] == "1"
