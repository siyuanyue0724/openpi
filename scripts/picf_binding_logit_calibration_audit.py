#!/usr/bin/env python3
"""Audit PICF binding-signature logit calibration.

This script is intentionally lightweight: it verifies the code-level dataflow
and the core math invariant without importing the full training stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _contains(path: str, *needles: str) -> tuple[bool, list[str]]:
    text = (ROOT / path).read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def _calibrate(score: np.ndarray, *, min_std: float = 0.05, clip: float = 4.0) -> np.ndarray:
    centered = score - score.mean(axis=1, keepdims=True) - score.mean(axis=0, keepdims=True) + score.mean()
    std = centered.std()
    if std < min_std:
        return np.zeros_like(centered)
    return np.clip(centered / std, -clip, clip)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()

    checks: list[dict[str, object]] = []

    for name, path, needles in (
        (
            "config_exposes_calibration",
            "src/openpi/picf/core/config.py",
            (
                "binding_signature_score_calibration_enabled: bool = True",
                'binding_signature_score_calibration_mode: str = "double_center_zscore"',
                "binding_signature_score_min_std: float = 0.05",
                "binding_signature_score_clip: float = 4.0",
            ),
        ),
        (
            "pipeline_applies_calibrated_combined_score",
            "src/openpi/picf/core/pipeline.py",
            (
                "def _calibrate_pairwise_binding_score",
                "centered = score - score.mean(dim=1, keepdim=True)",
                "binding_signature_combined_score_mean",
                "binding_signature_calibrated_score_std",
                "logits = logits + (bind_gate[:, None] * calibrated_score)",
                "nn.init.orthogonal_(self.binding_low_rank_left.weight)",
            ),
        ),
        (
            "trainer_logs_calibrated_metrics",
            "scripts/picf_core_train.py",
            (
                "posterior_binding_signature_calibrated_score_std",
                "--binding-signature-score-calibration-mode",
                "binding_signature_score_calibration_enabled=bool",
            ),
        ),
        (
            "readme_links_math_followthrough",
            "src/openpi/picf/README_v2.2.md",
            (
                "binding-logit calibration update",
                "PICF_AQR_OWM_BINDING_LOGIT_CALIBRATION_20260515_TEMP.md",
            ),
        ),
    ):
        passed, missing = _contains(path, *needles)
        checks.append({"name": name, "pass": passed, "missing": missing})

    common = np.full((4, 4), 0.73, dtype=np.float64)
    common_cal = _calibrate(common)
    checks.append(
        {
            "name": "math_common_mode_maps_to_zero",
            "pass": bool(np.allclose(common_cal, 0.0)),
            "max_abs": float(np.abs(common_cal).max()),
        }
    )

    relative = np.array(
        [
            [2.0, 0.2, 0.1],
            [0.1, 2.2, 0.2],
            [0.0, 0.1, 1.8],
        ],
        dtype=np.float64,
    )
    rel_cal = _calibrate(relative, min_std=1e-4, clip=10.0)
    checks.append(
        {
            "name": "math_relative_pairs_survive_calibration",
            "pass": bool(np.array_equal(rel_cal.argmax(axis=1), np.array([0, 1, 2])) and rel_cal.std() > 0.9),
            "argmax": rel_cal.argmax(axis=1).tolist(),
            "std": float(rel_cal.std()),
        }
    )

    passed = all(bool(c["pass"]) for c in checks)
    print(json.dumps({"pass": passed, "checks": checks}, indent=2, sort_keys=True))
    if args.fail_on_fail and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
