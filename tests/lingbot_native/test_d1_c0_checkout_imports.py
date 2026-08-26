from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_TOOLS = (
    ROOT / "tools/run_calvin_physical_supervision_parallel.py",
    ROOT / "tools/build_calvin_physical_supervision.py",
    ROOT / "tools/audit_calvin_physical_supervision.py",
    ROOT / "tools/finalize_calvin_physical_visual_review.py",
    ROOT / "tools/probe_lingbot_calvin_projection.py",
    ROOT / "tools/build_lingbot_calvin_predictive_cache.py",
    ROOT / "tools/build_lingbot_calvin_current_grid_cache.py",
    ROOT / "tools/build_lingbot_representation_split.py",
    ROOT / "tools/build_lingbot_representation_evaluation_plan.py",
    ROOT / "tools/build_lingbot_representation_evaluation_baseline.py",
    ROOT / "tools/build_lingbot_entity_evaluation_plan.py",
    ROOT / "tools/plan_lingbot_calvin_artifact_capacity.py",
    ROOT / "tools/preflight_lingbot_native.py",
    ROOT / "tools/audit_lingbot_predictive_targets.py",
    ROOT / "tools/audit_lingbot_dino_teacher_causality.py",
    ROOT / "tools/audit_lingbot_predictive_temporal_targets.py",
    ROOT / "tools/build_lingbot_native_gate_decision.py",
    ROOT / "tools/build_lingbot_native_training_authorization.py",
    ROOT / "tools/build_lingbot_predictive_objective_decision.py",
)


@pytest.mark.parametrize("tool", PRODUCTION_TOOLS, ids=lambda path: path.stem)
def test_d1_c0_tool_imports_its_own_checkout_over_stale_editable(
    tmp_path: Path,
    tool: Path,
) -> None:
    stale = tmp_path / "stale"
    package = stale / "picf_next"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('stale editable imported')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(stale)

    result = subprocess.run(
        [sys.executable, str(tool), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "stale editable imported" not in result.stderr
