from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_picf_loss_audit_summarizes_weighted_losses(tmp_path: Path) -> None:
    log = tmp_path / "loss.log"
    rows = [
        {
            "loss_total": 2.0,
            "loss_action": 1.0,
            "loss_visual_real": 0.2,
            "loss_alignment": 0.1,
        },
        {
            "loss_total": 4.0,
            "loss_action": 3.0,
            "loss_visual_real": 0.4,
            "loss_alignment": 0.3,
        },
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/picf_loss_audit.py", "--log", str(log), "--tail", "10"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(result.stdout)
    assert payload["num_rows"] == 2
    assert payload["means"]["loss_total"] == 3.0
    assert payload["means"]["loss_action"] == 2.0
    assert payload["ratios_to_total"]["loss_action"] == (2.0 / 3.0)
