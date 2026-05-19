#!/usr/bin/env python3
"""Audit posterior object-file continuity diagnostics.

The runtime binding path calibrates IsSameObject-style pairwise scores by
removing row/column common mode. Posterior continuity diagnostics must use the
same relative-score semantics; otherwise a raw cosine matrix can report false
swaps when every active file shares the same task/background signature.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(errors="ignore")


def _contains(text: str, *needles: str) -> bool:
    return all(needle in text for needle in needles)


def _double_center_zscore(score: torch.Tensor, min_std: float = 0.05) -> torch.Tensor:
    centered = score - score.mean(dim=1, keepdim=True) - score.mean(dim=0, keepdim=True) + score.mean()
    std = torch.std(centered, unbiased=False)
    if bool((std < min_std).item()):
        return torch.zeros_like(centered)
    return torch.clamp(centered / std, min=-4.0, max=4.0)


def _swap_rate(score: torch.Tensor) -> float:
    eye = torch.eye(score.shape[0], dtype=torch.bool)
    self_score = torch.diagonal(score, 0)
    best_other = score.masked_fill(eye, -1.0).max(dim=-1).values
    return float((best_other > self_score + 0.05).to(dtype=torch.float32).mean().item())


def main() -> int:
    pipeline = _read("src/openpi/picf/core/pipeline.py")
    train = _read("scripts/picf_core_train.py")
    report = _read("scripts/picf_anchor_run_diagnostic_report.py")
    verifier = _read("scripts/verify_picf_owm_contract.py")

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "pipeline_calibrates_file_continuity_matrix",
            _contains(
                pipeline,
                "calibrated_file_sim = self._calibrate_pairwise_binding_score(file_sim)",
                "posterior_active_file_calibrated_potential_swap_rate",
                "posterior_file_calibrated_signature_score_std",
            ),
            "Posterior file diagnostics must expose calibrated continuity metrics.",
        )
    )
    checks.append(
        (
            "trainer_surfaces_calibrated_file_metrics",
            _contains(
                train,
                "posterior_active_file_calibrated_potential_swap_rate",
                "posterior_file_calibrated_signature_score_std",
            ),
            "Training logs must include calibrated file-continuity metrics.",
        )
    )
    checks.append(
        (
            "diagnostic_report_prefers_calibrated_file_metrics",
            _contains(
                report,
                "calibrated_active_file_swap",
                "raw potential swap is high, but calibrated file-signature",
                "posterior active-file calibrated potential swap is high",
            ),
            "Run diagnostics must not treat raw common-mode signature overlap as confirmed file swaps.",
        )
    )
    checks.append(
        (
            "contract_requires_calibrated_file_metrics",
            _contains(
                verifier,
                "posterior_active_file_calibrated_potential_swap_rate",
                "posterior_file_calibrated_signature_score_std",
            ),
            "OWM contract must require calibrated file-continuity debug keys.",
        )
    )

    common_mode = torch.tensor([[0.90, 0.97], [0.90, 0.97]], dtype=torch.float32)
    calibrated_common = _double_center_zscore(common_mode)
    checks.append(
        (
            "math_rejects_common_mode_raw_swap",
            _swap_rate(common_mode) > 0.0 and float(calibrated_common.abs().max().item()) == 0.0,
            "Double-centering must reject raw swaps created only by row/column common mode.",
        )
    )

    real_swap = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32)
    calibrated_swap = _double_center_zscore(real_swap)
    checks.append(
        (
            "math_preserves_real_relative_swap",
            _swap_rate(calibrated_swap) == 1.0,
            "Calibrated continuity must still detect a real relative cross-file preference.",
        )
    )

    results = [
        {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        for name, passed, detail in checks
    ]
    print(json.dumps({"checks": results}, indent=2))
    failed = [item for item in results if item["status"] != "PASS"]
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
