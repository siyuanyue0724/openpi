#!/usr/bin/env python3
"""Summarize PICF anchor/binding diagnostic runs from metrics and overlays.

This is intentionally read-only. It turns the failure mode observed in short
CALVIN diagnostics into explicit, reproducible checks:

* raw same-role overlap can rise because reserve anchors reuse candidates;
* active-owner overlap is the relevant owner-separation signal;
* calibrated pairwise binding evidence should be used only when its matrix has
  nontrivial relative dispersion;
* high posterior identity switch after healthy active-owner separation requires
  an active-owner/posterior-object-file continuity audit, not another scalar
  loss patch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _last_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except Exception:
        return default


def _overlay_counts(run_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    overlay_dir = run_dir / "anchor_overlays"
    for path in sorted(overlay_dir.glob("step_*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        anchors = payload.get("anchors", [])
        source_counts: dict[str, int] = {}
        active_graph = 0.0
        posterior_count = 0
        for anchor in anchors:
            source = str(anchor.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
            if source == "graph":
                active_graph += float(anchor.get("active") or 0.0)
            if source == "posterior":
                posterior_count += 1
        out.append(
            {
                "file": path.name,
                "step": payload.get("step"),
                "prompt": payload.get("prompt"),
                "source_counts": source_counts,
                "active_graph_sum": active_graph,
                "posterior_count": posterior_count,
            }
        )
    return out


def summarize(run_dir: Path) -> dict[str, Any]:
    rows = _json_rows(run_dir / "metrics.jsonl")
    keep_keys = [
        "step",
        "loss_total",
        "loss_action_default_equiv",
        "aqr_same_role_support_overlap_max",
        "aqr_active_same_role_support_overlap_max",
        "aqr_same_role_object_core_overlap_max",
        "aqr_active_same_role_object_core_overlap_max",
        "aqr_effective_anchor_count",
        "aqr_active_anchor_count",
        "posterior_identity_switch_rate",
        "posterior_identity_switch_rate_stable",
        "posterior_recycle_rate",
        "posterior_binding_signature_calibrated_score_std",
        "posterior_binding_signature_calibrated_top1_margin_mean",
        "posterior_active_file_fraction",
        "posterior_active_file_self_signature_sim_mean",
        "posterior_active_file_best_other_signature_margin_mean",
        "posterior_active_file_potential_swap_rate",
        "posterior_file_calibrated_signature_score_std",
        "posterior_active_file_calibrated_self_signature_sim_mean",
        "posterior_active_file_calibrated_best_other_signature_margin_mean",
        "posterior_active_file_calibrated_potential_swap_rate",
    ]
    metrics = [{key: row.get(key) for key in keep_keys if key in row} for row in rows]
    last = rows[-1] if rows else {}
    raw_overlap = _last_float(last, "aqr_same_role_support_overlap_max")
    active_overlap = _last_float(last, "aqr_active_same_role_support_overlap_max")
    raw_core = _last_float(last, "aqr_same_role_object_core_overlap_max")
    active_core = _last_float(last, "aqr_active_same_role_object_core_overlap_max")
    identity_switch = _last_float(last, "posterior_identity_switch_rate")
    calib_std = _last_float(last, "posterior_binding_signature_calibrated_score_std")
    calib_margin = _last_float(last, "posterior_binding_signature_calibrated_top1_margin_mean")
    active_file_fraction = _last_float(last, "posterior_active_file_fraction")
    active_file_swap = _last_float(last, "posterior_active_file_potential_swap_rate")
    active_file_margin = _last_float(last, "posterior_active_file_best_other_signature_margin_mean")
    calibrated_file_std = _last_float(last, "posterior_file_calibrated_signature_score_std")
    calibrated_active_file_swap = _last_float(last, "posterior_active_file_calibrated_potential_swap_rate")
    calibrated_active_file_margin = _last_float(
        last,
        "posterior_active_file_calibrated_best_other_signature_margin_mean",
    )

    findings: list[str] = []
    if raw_overlap > 0.8 and active_overlap < 0.15:
        findings.append(
            "raw same-role support overlap is high while active-owner overlap is low; "
            "reserve/inactive anchors are the likely source of raw overlap."
        )
    if raw_core > 0.7 and active_core < 0.15:
        findings.append(
            "raw object-core overlap is high while active-owner object-core overlap is low; "
            "active owners are still separated in geometry/support space."
        )
    if calib_std <= 1e-6 and calib_margin <= 1e-6:
        findings.append(
            "calibrated pairwise binding evidence is off for the last row; the score matrix "
            "did not have enough relative dispersion to trust as identity evidence."
        )
    if identity_switch > 0.5 and active_overlap < 0.15:
        findings.append(
            "identity switch remains high despite healthy active-owner overlap; next audit "
            "must isolate active-owner object-file continuity before changing losses."
        )
    if active_file_fraction == active_file_fraction:
        calibrated_present = calibrated_active_file_swap == calibrated_active_file_swap
        if (
            calibrated_present
            and active_file_fraction > 0.0
            and calibrated_active_file_swap <= 0.1
            and calibrated_active_file_margin >= 0.0
        ):
            findings.append(
                "posterior active-file continuity is healthy under calibrated relative identity scores; "
                "raw signature swap should be treated as common-mode evidence unless overlays disagree."
            )
        elif calibrated_present and active_file_fraction > 0.0 and calibrated_active_file_swap > 0.25:
            findings.append(
                "posterior active-file calibrated potential swap is high; this is a real object-file "
                "continuity target, not only raw common-mode signature overlap."
            )
        elif active_file_fraction > 0.0 and active_file_swap <= 0.1 and active_file_margin >= 0.0:
            findings.append(
                "posterior active-file continuity is healthy even if row-id switch is high; "
                "do not treat observation-row churn as confirmed object-file collapse."
            )
        elif active_file_fraction > 0.0 and active_file_swap > 0.25:
            if calibrated_file_std == calibrated_file_std and calibrated_file_std <= 1e-6:
                findings.append(
                    "posterior active-file raw potential swap is high, but calibrated file-signature "
                    "dispersion is zero; treat this as a raw common-mode diagnostic, not a confirmed "
                    "object-file swap."
                )
            else:
                findings.append(
                    "posterior active-file potential swap is high; the remaining target is "
                    "object-file update/continuity, not raw support separation."
                )
    if not findings:
        findings.append("no predefined anchor failure signature detected; inspect overlays manually.")

    return {
        "run_dir": str(run_dir),
        "metric_rows": metrics,
        "overlay_rows": _overlay_counts(run_dir),
        "last_step": last.get("step"),
        "findings": findings,
        "recommended_next_step": (
            "Run with posterior active-file continuity metrics enabled before launching "
            "another architecture variant."
            if any("identity switch remains high" in item for item in findings)
            else "Proceed according to overlay/video inspection."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()
    if args.fail_on_missing and not (args.run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"missing metrics.jsonl under {args.run_dir}")
    print(json.dumps(summarize(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
