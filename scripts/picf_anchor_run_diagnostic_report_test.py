from __future__ import annotations

import json
from pathlib import Path

from scripts.picf_anchor_run_diagnostic_report import summarize


def test_anchor_run_diagnostic_uses_active_file_continuity(tmp_path: Path) -> None:
    row = {
        "step": 180,
        "aqr_same_role_support_overlap_max": 0.98,
        "aqr_active_same_role_support_overlap_max": 0.04,
        "aqr_same_role_object_core_overlap_max": 0.91,
        "aqr_active_same_role_object_core_overlap_max": 0.05,
        "posterior_identity_switch_rate": 0.76,
        "posterior_binding_signature_calibrated_score_std": 0.0,
        "posterior_binding_signature_calibrated_top1_margin_mean": 0.0,
        "posterior_active_file_fraction": 0.4,
        "posterior_active_file_best_other_signature_margin_mean": 0.12,
        "posterior_active_file_potential_swap_rate": 0.0,
    }
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = summarize(tmp_path)
    findings = "\n".join(report["findings"])

    assert "raw same-role support overlap is high" in findings
    assert "posterior active-file continuity is healthy" in findings
