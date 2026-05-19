#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AuditCheck:
    name: str
    ok: bool
    detail: str


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def _contains(source: str, *needles: str) -> bool:
    return all(needle in source for needle in needles)


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def _ema(prev: np.ndarray, inst: np.ndarray, rate: np.ndarray) -> np.ndarray:
    return _normalize_rows(((1.0 - rate[:, None]) * prev) + (rate[:, None] * inst))


def _double_center_zscore(score: np.ndarray, min_std: float = 0.05) -> tuple[np.ndarray, float]:
    centered = score - score.mean(axis=1, keepdims=True) - score.mean(axis=0, keepdims=True) + score.mean()
    std = float(centered.std())
    if std < min_std:
        return np.zeros_like(centered), std
    return centered / std, std


def _dispersion_gate(inst: np.ndarray, min_std: float = 0.05, min_margin: float = 0.25, temp: float = 0.10) -> tuple[np.ndarray, float, np.ndarray]:
    inst = _normalize_rows(inst)
    calibrated, std = _double_center_zscore(inst @ inst.T, min_std=min_std)
    self_score = np.diag(calibrated)
    other = calibrated.copy()
    np.fill_diagonal(other, -1.0)
    best_other = other.max(axis=1)
    margin = self_score - best_other
    gate = 1.0 / (1.0 + np.exp(-(margin - min_margin) / max(temp, 1e-8)))
    if std < min_std:
        gate = np.zeros_like(gate)
    return gate, std, margin


def _math_checks() -> list[AuditCheck]:
    prev = _normalize_rows(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    inst = _normalize_rows(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    low_rate = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    birth_rate = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    trusted_rate = np.asarray([0.2, 0.2, 0.2], dtype=np.float64)
    kept = _ema(prev, inst, low_rate)
    reset = _ema(prev, inst, birth_rate)
    trusted = _ema(prev, inst, trusted_rate)
    common_mode = _normalize_rows(np.ones((3, 3), dtype=np.float64))
    common_gate, common_std, _ = _dispersion_gate(common_mode)
    relative_gate, relative_std, relative_margin = _dispersion_gate(prev)
    gated_rate = trusted_rate * relative_gate
    gated = _ema(prev, inst, gated_rate)
    return [
        AuditCheck(
            "math_low_trust_keeps_previous_signature",
            bool(np.allclose(kept, prev, atol=1e-8)),
            "When support/owner trust is zero, file identity descriptors must not be overwritten.",
        ),
        AuditCheck(
            "math_birth_or_recycle_resets_to_instant_signature",
            bool(np.allclose(reset, inst, atol=1e-8)),
            "Birth/recycle is the only path that should fully replace a file signature.",
        ),
        AuditCheck(
            "math_trusted_measurement_moves_but_does_not_jump",
            bool(np.all((trusted * prev).sum(axis=-1) > 0.7) and np.all((trusted * inst).sum(axis=-1) > 0.1)),
            "Trusted measurements should move file descriptors gradually, preserving continuity.",
        ),
        AuditCheck(
            "math_outputs_remain_unit_normalized",
            bool(np.allclose(np.linalg.norm(trusted, axis=-1), 1.0, atol=1e-7)),
            "Signature memory update must preserve cosine/probe geometry.",
        ),
        AuditCheck(
            "math_common_mode_measurement_is_rejected_by_dispersion_gate",
            bool(np.allclose(common_gate, 0.0, atol=1e-8) and common_std < 0.05),
            "A common-mode binding signature must not overwrite object-file memory.",
        ),
        AuditCheck(
            "math_relative_measurement_can_update_when_dispersed",
            bool(relative_std >= 0.05 and np.all(relative_margin > 0.25) and np.all(gated_rate > 0.0)),
            "Only a relative same-object matrix with diagonal margin may update file identity.",
        ),
        AuditCheck(
            "math_dispersion_gated_update_remains_slow_memory",
            bool(np.all((gated * prev).sum(axis=-1) > 0.7)),
            "Even accepted relative evidence must update gradually rather than jump.",
        ),
    ]


def run_checks() -> list[AuditCheck]:
    config = _read("src/openpi/picf/core/config.py")
    contracts = _read("src/openpi/picf/core/contracts.py")
    pipeline = _read("src/openpi/picf/core/pipeline.py")
    trainer = _read("scripts/picf_core_train.py")
    evidence = _read("scripts/picf_owm_evidence_bundle.py")
    docs = _read("temp/audits_20260519/posterior_file_continuity_metric_followthrough.md")

    checks = [
        AuditCheck(
            "config_exposes_binding_signature_memory_knobs",
            _contains(
                config,
                "posterior_binding_signature_memory_enabled: bool = True",
                "posterior_binding_signature_update_rate",
                "posterior_binding_signature_update_max_rate",
                "posterior_binding_signature_min_support",
                "posterior_binding_signature_owner_weight",
                "posterior_binding_signature_dispersion_gate_enabled: bool = True",
                "posterior_binding_signature_measurement_min_std",
                "posterior_binding_signature_measurement_margin_min",
                "posterior_binding_signature_measurement_margin_temperature",
            ),
            "The belief-state signature update must be a documented, configurable contract.",
        ),
        AuditCheck(
            "contracts_expose_signature_memory_diagnostics",
            _contains(
                contracts,
                "binding_signature_update_rate",
                "binding_signature_measurement_trust",
                "binding_signature_memory_keep_rate",
                "binding_signature_measurement_score_std",
                "binding_signature_measurement_margin",
                "binding_signature_measurement_dispersion_gate",
            ),
            "Posterior state must carry the signature-memory diagnostics into logs and bundles.",
        ),
        AuditCheck(
            "pipeline_keeps_identity_as_latent_memory_state",
            _contains(
                pipeline,
                "instant_binding_signature",
                "previous.posterior.binding_signature",
                "binding_signature_measurement_trust",
                "calibrated_instant_score",
                "binding_signature_measurement_dispersion_gate",
                "stable_file_gate",
                "reset_rate",
                "((1.0 - binding_signature_update_rate[:, None]) * previous_binding_signature)",
                "binding_signature_memory_keep_rate",
            ),
            "Posterior signatures must be EMA-updated from trusted measurements instead of overwritten.",
        ),
        AuditCheck(
            "trainer_and_bundle_log_signature_memory",
            _contains(
                trainer,
                "posterior_binding_signature_update_rate_mean",
                "posterior_binding_signature_measurement_trust_mean",
                "posterior_binding_signature_memory_keep_rate_mean",
                "--posterior-binding-signature-memory-enabled",
                "--posterior-binding-signature-dispersion-gate-enabled",
                "posterior_binding_signature_measurement_score_std",
                "posterior_binding_signature_measurement_dispersion_gate_mean",
            )
            and _contains(
                evidence,
                "posterior_binding_signature_update_rate_mean",
                "posterior_binding_signature_measurement_trust_mean",
                "posterior_binding_signature_memory_keep_rate_mean",
                "posterior_binding_signature_measurement_score_std",
                "posterior_binding_signature_measurement_dispersion_gate_mean",
            ),
            "Training logs and evidence bundles must expose whether the memory update is active.",
        ),
        AuditCheck(
            "docs_record_file_continuity_root_fix",
            _contains(
                docs,
                "posterior file continuity",
                "binding_signature",
                "calibrated",
            ),
            "The math follow-through must describe the posterior file-continuity target.",
        ),
    ]
    checks.extend(_math_checks())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()
    checks = run_checks()
    payload = {
        "pass": all(check.ok for check in checks),
        "checks": [{"name": check.name, "pass": check.ok, "detail": check.detail} for check in checks],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_fail and not payload["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
