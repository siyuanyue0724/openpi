from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from openpi.picf.test_utils import build_mini_calvin_dataset


def test_precompute_tactile_contact_calibration_smoke(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)
    output_dir = tmp_path / "artifacts"
    script_path = Path(__file__).with_name("precompute_tactile_contact_calibration.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--calvin-root",
            calvin_root,
            "--split",
            "training",
            "--backend",
            "dir",
            "--sample-stride",
            "1",
            "--max-frames",
            "4",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["sampled_frames"] == 4
    assert 0.0 <= float(summary["active_rate_tau_on"]) <= 1.0

    backgrounds = np.load(output_dir / "tactile_backgrounds.npz")
    assert set(backgrounds.files) == {"digit", "gelsight_mini"}

    stats = json.loads((output_dir / "tactile_contact_stats.json").read_text())
    assert stats["score_mode"] == "rgb_only"
    assert stats["sampled_frames"] == 4
    assert stats["tau_on"] >= stats["tau_off"]

    calibration = json.loads((output_dir / "tactile_fingertip_calibration.json").read_text())
    assert len(calibration["u_open_local"]) == 3
    assert len(calibration["o_local"]) == 3
    assert 0.035 <= float(calibration["recommended_pt_bag_radius_m"]) <= 0.055
