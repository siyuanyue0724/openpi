#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.verify_picf_owm_contract import run_checks


OWM_KEYS: tuple[str, ...] = (
    "loss_slot_jepa",
    "loss_support_pred",
    "loss_binding_consistency",
    "loss_aqr_denoising",
    "aqr_temporal_support_entropy_mean",
    "aqr_temporal_support_time_mass_t0",
    "aqr_temporal_support_time_mass_t1",
    "aqr_temporal_view_mass_0",
    "aqr_temporal_view_mass_1",
    "aqr_pg_support_entropy_mean",
    "aqr_pg_support_max",
    "aqr_pg_support_peak_mean",
    "aqr_tracklet_support_entropy_mean",
    "aqr_tracklet_support_max",
    "aqr_proposal_support_entropy_mean",
    "aqr_proposal_support_max",
    "aqr_local_support_entropy_mean",
    "aqr_effective_anchor_count",
    "aqr_same_role_support_overlap_max",
    "posterior_identity_switch_rate",
    "posterior_recycle_rate",
    "owm_tracklet_tokens",
    "owm_tracklet_valid_fraction",
    "owm_proposal_tokens",
    "owm_proposal_valid_fraction",
    "owm_posterior_support_signature_mean",
    "evidence_cache_trust_mean",
    "evidence_cache_age_mean",
    "innovation_norm_visual",
    "innovation_norm_point",
    "innovation_norm_tactile",
    "owm_ordinal_active",
    "owm_ordinal_target_rank",
    "owm_ordinal_confidence",
)

OWM_ARG_KEYS: tuple[str, ...] = (
    "aqr_vjepa_temporal_mode",
    "aqr_vjepa_temporal_tokens",
    "aqr_vjepa_temporal_include_delta",
    "evidence_cache_enabled",
    "evidence_cache_len",
    "evidence_cache_read_weight",
    "evidence_cache_innovation_downweight",
    "evidence_cache_address_weight",
    "tracklet_memory_enabled",
    "tracklet_max_tokens",
    "tracklet_read_weight",
    "proposal_memory_enabled",
    "proposal_max_tokens",
    "proposal_read_weight",
    "bind_support_signature_weight",
    "bind_address_weight",
    "local_refinement_enabled",
    "local_refinement_topk",
    "local_refinement_weight",
    "slot_jepa_enabled",
    "support_prediction_enabled",
    "ordinal_relation_enabled",
    "lambda_slot_jepa",
    "lambda_support_pred",
    "lambda_binding_consistency",
    "lambda_aqr_denoising",
)


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(path: Path, *, tail: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records[-max(int(tail), 1) :]


def _select_keys(record: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: record[key] for key in keys if key in record}


def _diagnostic_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir / "diagnostics"
    if not root.is_dir():
        return []
    artifacts: list[dict[str, Any]] = []
    for diag_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = _read_json(diag_dir / "metadata.json")
        files = sorted(
            str(path.relative_to(run_dir))
            for path in diag_dir.iterdir()
            if path.is_file() and path.name != "metadata.json"
        )
        artifacts.append(
            {
                "diagnostic_dir": str(diag_dir.relative_to(run_dir)),
                "metadata": metadata,
                "files": files,
            }
        )
    return artifacts


def _contract_verifier_snapshot() -> dict[str, Any]:
    checks = run_checks()
    return {
        "ok": all(check.ok for check in checks),
        "checks": [check.__dict__ for check in checks],
    }


def build_bundle(run_dir: Path, *, tail: int) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    args_payload = _read_json(run_dir / "args.json") or {}
    metrics_tail = _read_metrics(run_dir / "metrics.jsonl", tail=tail)
    latest = metrics_tail[-1] if metrics_tail else {}
    return {
        "contract": "PICF-AQR-OWM evidence bundle",
        "run_dir": str(run_dir),
        "args_owm": _select_keys(args_payload, OWM_ARG_KEYS) if isinstance(args_payload, dict) else {},
        "latest_owm_metrics": _select_keys(latest, OWM_KEYS) if isinstance(latest, dict) else {},
        "metrics_tail": [
            {
                "step": record.get("step"),
                "owm": _select_keys(record, OWM_KEYS),
            }
            for record in metrics_tail
            if isinstance(record, dict)
        ],
        "diagnostics": _diagnostic_artifacts(run_dir),
        "contract_verifier": _contract_verifier_snapshot(),
        "audit_rules": {
            "posterior_authoritative": True,
            "cache_auxiliary_only": True,
            "future_targets_are_loss_only": True,
            "last_two_mean_is_ablation_only": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a PICF-AQR-OWM JSON evidence bundle from a training run directory.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tail", type=int, default=50)
    args = parser.parse_args(argv)

    bundle = build_bundle(args.run_dir, tail=args.tail)
    output = args.output or (args.run_dir / "owm_evidence_bundle.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "diagnostics": len(bundle["diagnostics"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
