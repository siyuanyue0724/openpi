#!/usr/bin/env python3
"""Audit the PICF binding-signature common-mode repair.

This script is intentionally lightweight and does not import torch. It checks
the code-level dataflow and runs a NumPy-only sanity test for the core math:

    signature_j = normalize(sum_i support_ji * normalize(W z_i))

If W z_i contains a large shared component, different supports can still produce
nearly identical signatures. Centering W z_i inside each typed memory before
support pooling removes that global component and makes the signature represent
relative same-object evidence instead of scene/modality bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _contains(path: Path, *needles: str) -> tuple[bool, list[str]]:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def _synthetic_common_mode() -> dict[str, float]:
    rng = np.random.default_rng(17)
    supports = np.zeros((4, 16), dtype=np.float64)
    supports[0, :4] = 0.25
    supports[1, 4:8] = 0.25
    supports[2, 8:12] = 0.25
    supports[3, 12:16] = 0.25
    common = rng.normal(size=(1, 32))
    residual = rng.normal(size=(16, 32))
    residual[:4] += 1.0
    residual[4:8] -= 1.0
    residual[8:12, 2:6] += 1.0
    residual[12:16, 2:6] -= 1.0
    tokens = (6.0 * common) + residual

    raw_keys = _normalize(tokens)
    centered_keys = _normalize(tokens - tokens.mean(axis=0, keepdims=True))
    raw_sig = _normalize(supports @ raw_keys)
    centered_sig = _normalize(supports @ centered_keys)
    offdiag = ~np.eye(raw_sig.shape[0], dtype=bool)
    raw_cos = raw_sig @ raw_sig.T
    centered_cos = centered_sig @ centered_sig.T
    return {
        "raw_offdiag_cos_mean": float(raw_cos[offdiag].mean()),
        "raw_offdiag_cos_max": float(raw_cos[offdiag].max()),
        "centered_offdiag_cos_mean": float(centered_cos[offdiag].mean()),
        "centered_offdiag_cos_max": float(centered_cos[offdiag].max()),
        "mean_drop": float(raw_cos[offdiag].mean() - centered_cos[offdiag].mean()),
    }


def run() -> dict[str, Any]:
    root = _repo_root()
    config = root / "src/openpi/picf/core/config.py"
    pipeline = root / "src/openpi/picf/core/pipeline.py"
    trainer = root / "scripts/picf_core_train.py"
    tests = root / "src/openpi/picf/core/pipeline_test.py"
    checks: list[dict[str, Any]] = []

    ok, missing = _contains(
        config,
        "binding_signature_centering_enabled: bool = True",
        "binding_signature_centering_min_tokens",
    )
    checks.append({"name": "config_exposes_centered_binding_signature", "pass": ok, "missing": missing})

    ok, missing = _contains(
        pipeline,
        "def _binding_keys(self, tokens: torch.Tensor | None, *, center: bool = False)",
        "binding_signature_centering_enabled",
        "projected - projected.mean(dim=0, keepdim=True)",
        "self._binding_keys(tokens, center=True)",
    )
    checks.append({"name": "pipeline_centers_binding_keys_before_support_pooling", "pass": ok, "missing": missing})

    ok, missing = _contains(
        trainer,
        "--binding-signature-centering-enabled",
        "binding_signature_centering_enabled=bool",
        "--binding-signature-centering-min-tokens",
    )
    checks.append({"name": "trainer_threads_centering_flags", "pass": ok, "missing": missing})

    ok, missing = _contains(
        pipeline,
        "prev.binding_signature",
        "obs.binding_signature",
        "bind_embedding_signature_weight",
        "_binding_signature_quadratic_scores",
        "bind_quadratic_signature_weight",
        "bind_low_rank_signature_weight",
        "innovation_decay",
    )
    checks.append({"name": "binding_logits_use_gated_quadratic_pairwise_signature", "pass": ok, "missing": missing})

    ok, missing = _contains(
        tests,
        "test_binding_signature_centering_removes_common_mode",
        "torch.dot(raw[0], raw[1]) > 0.95",
        "torch.dot(centered[0], centered[1]) < -0.95",
    )
    checks.append({"name": "unit_test_covers_common_mode_case", "pass": ok, "missing": missing})

    synthetic = _synthetic_common_mode()
    checks.append(
        {
            "name": "numpy_common_mode_sanity",
            "pass": synthetic["raw_offdiag_cos_mean"] > 0.95 and synthetic["mean_drop"] > 0.25,
            "metrics": synthetic,
        }
    )
    return {"checks": checks, "pass": all(bool(check["pass"]) for check in checks)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.fail_on_fail and not result["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
