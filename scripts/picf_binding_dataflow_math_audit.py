#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOTS = [
    Path("/tmp/vit-object-binding"),
    Path("/tmp/picf_paper_code_20260515/vit-object-binding"),
]


@dataclass(frozen=True)
class AuditCheck:
    name: str
    ok: bool
    detail: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains(source: str, *needles: str) -> bool:
    return all(needle in source for needle in needles)


def _double_center_zscore(score: np.ndarray, *, min_std: float = 0.05, clip: float = 4.0) -> np.ndarray:
    centered = score - score.mean(axis=1, keepdims=True) - score.mean(axis=0, keepdims=True) + score.mean()
    std = float(centered.std())
    if std < min_std:
        return np.zeros_like(centered)
    out = centered / std
    return np.clip(out, -clip, clip)


def _paper_root() -> Path | None:
    for root in PAPER_ROOTS:
        if (root / "src/utils/models.py").exists() and (root / "src/trainer.py").exists():
            return root
    return None


def _math_checks() -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    common = np.full((4, 4), 3.0)
    checks.append(
        AuditCheck(
            "math_common_mode_zero",
            bool(np.allclose(_double_center_zscore(common), 0.0)),
            "A constant raw IsSameObject matrix must not become identity evidence.",
        )
    )

    row_col = np.arange(4, dtype=np.float64)[:, None] + np.arange(4, dtype=np.float64)[None, :]
    checks.append(
        AuditCheck(
            "math_row_column_bias_zero",
            bool(np.allclose(_double_center_zscore(row_col), 0.0, atol=1e-8)),
            "Pure row/column saliency bias must be removed because it is not pair identity.",
        )
    )

    rel = np.eye(4, dtype=np.float64) * 2.0
    cal = _double_center_zscore(rel)
    checks.append(
        AuditCheck(
            "math_relative_pairs_preserved",
            bool(np.all(np.argmax(cal, axis=1) == np.arange(4)) and cal.std() > 0.5),
            "A relative diagonal same-object signal must survive calibration.",
        )
    )

    noisy_tiny = np.eye(4, dtype=np.float64) * 1e-4
    checks.append(
        AuditCheck(
            "math_low_dispersion_rejected",
            bool(np.allclose(_double_center_zscore(noisy_tiny), 0.0)),
            "Near-constant low-dispersion matrices should not be amplified as noise.",
        )
    )
    return checks


def run_checks() -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    paper = _paper_root()
    if paper is None:
        checks.append(AuditCheck("paper_code_available", False, "vit-object-binding code snapshot not found in /tmp."))
        paper_models = paper_trainer = ""
    else:
        paper_models = _read(paper / "src/utils/models.py")
        paper_trainer = _read(paper / "src/trainer.py")
        checks.append(
            AuditCheck(
                "paper_code_available",
                True,
                f"Using {paper} for code-level math comparison.",
            )
        )

    checks.append(
        AuditCheck(
            "paper_quadratic_probe_family_present",
            _contains(
                paper_models,
                "class DiagonalQuadraticProbe",
                "class QuadraticProbe",
                "class QuadraticFixedRankProbe",
                "forward_pairwise",
                "W_sym",
            ),
            "Paper code must expose diagonal, full, and fixed-rank quadratic pairwise probes.",
        )
    )
    checks.append(
        AuditCheck(
            "paper_pairwise_bce_calibration_present",
            _contains(paper_trainer, "BCEWithLogitsLoss", "labels_pairwise = labels.unsqueeze(1) == labels.unsqueeze(2)"),
            "Paper probe scores are calibrated logits trained from instance-mask IsSameObject labels.",
        )
    )

    pipeline = _read(REPO_ROOT / "src/openpi/picf/core/pipeline.py")
    config = _read(REPO_ROOT / "src/openpi/picf/core/config.py")
    contracts = _read(REPO_ROOT / "src/openpi/picf/core/contracts.py")
    trainer = _read(REPO_ROOT / "scripts/picf_core_train.py")
    replay = _read(REPO_ROOT / "src/openpi/picf/replay/calvin_replay.py")
    serve = _read(REPO_ROOT / "scripts/serve_picf_policy.py")
    readme_v22 = _read(REPO_ROOT / "src/openpi/picf/README_v2.2.md")
    report = _read(REPO_ROOT / "docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md")

    checks.extend(
        [
            AuditCheck(
                "picf_exposes_binding_calibration_config",
                _contains(
                    config,
                    "binding_signature_score_calibration_enabled",
                    'binding_signature_score_calibration_mode: str = "double_center_zscore"',
                    "binding_signature_score_min_std",
                    "binding_signature_score_clip",
                ),
                "Runtime must expose explicit calibration knobs instead of implicit raw-cosine behavior.",
            ),
            AuditCheck(
                "picf_builds_support_weighted_signatures",
                _contains(pipeline, "def _support_binding_signature", "normalized_weights @ self._binding_keys(tokens, center=True)"),
                "Binding signatures must be support-weighted typed evidence, not only learned slot IDs.",
            ),
            AuditCheck(
                "picf_reimplements_quadratic_family_natively",
                _contains(
                    pipeline,
                    "def _binding_signature_quadratic_scores",
                    "binding_quadratic_diag",
                    "binding_low_rank_left",
                    "binding_low_rank_right",
                    "low_rank_score",
                ),
                "PICF should reimplement the equation family without importing or copying unlicensed paper code.",
            ),
            AuditCheck(
                "picf_calibrates_before_binding_logit",
                _contains(
                    pipeline,
                    "def _calibrate_pairwise_binding_score",
                    "score - score.mean(dim=1, keepdim=True) - score.mean(dim=0, keepdim=True) + score.mean()",
                    "std < min_std",
                    "calibrated_score = self._calibrate_pairwise_binding_score(combined_score)",
                    "logits = logits + (bind_gate[:, None] * calibrated_score)",
                ),
                "Raw pairwise scores must be converted into relative assignment logits before posterior binding.",
            ),
            AuditCheck(
                "picf_preserves_belief_filter_gates",
                _contains(pipeline, "bind_gate = bind_gate * innovation_decay", "prev.alpha", "prev.recycle_gate"),
                "Binding evidence must be gated by posterior trust rather than hard-locking identity.",
            ),
            AuditCheck(
                "picf_observation_to_posterior_signature_dataflow",
                _contains(
                    pipeline,
                    "obs_binding_signature",
                    "binding_signature=obs_binding_signature",
                    "binding_cond @ obs_anchors.binding_signature",
                    "instant_binding_signature",
                    "posterior_binding_signature_memory_enabled",
                    "posterior_binding_signature_dispersion_gate_enabled",
                    "calibrated_instant_score",
                    "binding_signature_measurement_dispersion_gate",
                    "binding_signature_update_rate",
                    "binding_signature=binding_signature",
                ),
                "Observation anchor signatures must reach posterior object files through a trusted memory update, not a blind overwrite.",
            ),
            AuditCheck(
                "picf_posterior_signature_memory_is_trust_gated",
                _contains(
                    pipeline,
                    "previous.posterior.binding_signature",
                    "posterior_binding_signature_update_rate",
                    "posterior_binding_signature_min_support",
                    "posterior_binding_signature_measurement_min_std",
                    "posterior_binding_signature_measurement_margin_min",
                    "assignment_trust",
                    "owner_reliability",
                    "binding_signature_measurement_score_std",
                    "binding_signature_measurement_margin",
                    "stable_file_gate",
                    "reset_rate",
                    "((1.0 - binding_signature_update_rate[:, None]) * previous_binding_signature)",
                )
                and _contains(
                    config,
                    "posterior_binding_signature_memory_enabled: bool = True",
                    "posterior_binding_signature_update_rate",
                    "posterior_binding_signature_update_max_rate",
                    "posterior_binding_signature_min_support",
                )
                and _contains(
                    contracts,
                    "binding_signature_update_rate",
                    "binding_signature_measurement_trust",
                    "binding_signature_memory_keep_rate",
                ),
                "Posterior file signatures must be latent state with assignment/owner/support/recycle-gated measurement updates.",
            ),
            AuditCheck(
                "picf_measures_posterior_file_continuity_not_only_obs_row_ids",
                _contains(
                    pipeline,
                    "posterior_file_self_signature_sim_mean",
                    "posterior_active_file_self_signature_sim_mean",
                    "posterior_active_file_potential_swap_rate",
                    "posterior_active_file_calibrated_potential_swap_rate",
                    "posterior_file_calibrated_signature_score_std",
                    "observation-anchor row ids",
                )
                and _contains(trainer, "posterior_active_file_self_signature_sim_mean", "posterior_active_file_calibrated_potential_swap_rate")
                and _contains(readme_v22, "posterior object-file continuity"),
                "Identity diagnostics must include posterior file self-continuity, because observation-anchor row ids are not stable object ids.",
            ),
            AuditCheck(
                "contracts_store_runtime_diagnostics",
                _contains(
                    contracts,
                    "binding_signature_combined_score_abs_mean",
                    "binding_signature_calibrated_score_std",
                    "binding_signature_calibrated_top1_margin_mean",
                ),
                "Posterior state must preserve calibrated binding diagnostics for evidence bundles and logs.",
            ),
            AuditCheck(
                "train_replay_serve_thread_optional_mvtrack_fields",
                _contains(trainer, "tracklet_xy=frame.get(\"tracklet_xy\")", "proposal_centers_xy=frame.get(\"proposal_centers_xy\")")
                and _contains(replay, "tracklet_xy=frame.get(\"tracklet_xy\")", "proposal_centers_xy=frame.get(\"proposal_centers_xy\")")
                and _contains(serve, "tracklet_xy=_optional_array", "proposal_centers_xy=_optional_array"),
                "Optional tracklet/proposal fields must be dataflow-valid even if current CALVIN has none.",
            ),
            AuditCheck(
                "metrics_and_docs_record_limits",
                _contains(trainer, "posterior_binding_signature_calibrated_top1_margin_mean", "owm_tracklet_tokens", "loss_action_default_equiv")
                and _contains(readme_v22, "binding-logit calibration update")
                and _contains(report, "Paper-code boundary", "posterior_identity_switch_rate stays around 0.70"),
                "Docs/metrics must expose both the repair and the remaining identity-continuity limit.",
            ),
        ]
    )
    checks.extend(_math_checks())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()
    checks = run_checks()
    payload = {
        "pass": all(c.ok for c in checks),
        "checks": [{"name": c.name, "pass": c.ok, "detail": c.detail} for c in checks],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_fail and not payload["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
