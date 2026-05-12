#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Finding:
    name: str
    status: str
    severity: str
    detail: str
    evidence: list[str]


class Source:
    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.path = REPO_ROOT / relpath
        self.text = self.path.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()

    def contains(self, *needles: str) -> bool:
        return all(needle in self.text for needle in needles)

    def refs(self, *needles: str, limit: int = 8) -> list[str]:
        refs: list[str] = []
        for needle in needles:
            for index, line in enumerate(self.lines, start=1):
                if needle in line:
                    refs.append(f"{self.relpath}:{index}: {line.strip()}")
                    break
            if len(refs) >= limit:
                break
        return refs

    def regex_refs(self, pattern: str, limit: int = 8) -> list[str]:
        rx = re.compile(pattern)
        refs: list[str] = []
        for index, line in enumerate(self.lines, start=1):
            if rx.search(line):
                refs.append(f"{self.relpath}:{index}: {line.strip()}")
                if len(refs) >= limit:
                    break
        return refs


def _finding(name: str, ok: bool, *, severity: str, detail: str, evidence: Iterable[str] = ()) -> Finding:
    return Finding(
        name=name,
        status="PASS" if ok else ("WARN" if severity == "warn" else "FAIL"),
        severity=severity,
        detail=detail,
        evidence=list(evidence),
    )


def _read_metrics(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return None if not vals else sum(vals) / len(vals)


def _nonzero_count(rows: list[dict], key: str) -> int:
    return sum(1 for row in rows if isinstance(row.get(key), (int, float)) and abs(float(row[key])) > 1e-12)


def _metric_trend_findings(rows: list[dict]) -> list[Finding]:
    if not rows:
        return [
            Finding(
                name="runtime_metrics_available",
                status="WARN",
                severity="warn",
                detail="No metrics JSONL was provided; runtime loss/diagnostic conclusions are not evaluated.",
                evidence=[],
            )
        ]
    n = max(1, len(rows) // 4)
    first = rows[:n]
    last = rows[-n:]
    findings: list[Finding] = [
        Finding(
            name="runtime_metrics_available",
            status="PASS",
            severity="info",
            detail=f"Loaded {len(rows)} metric rows; first step={rows[0].get('step')} last step={rows[-1].get('step')}.",
            evidence=[],
        )
    ]
    if _mean(first, "loss_action") is not None and _mean(last, "loss_action") is not None:
        f = _mean(first, "loss_action") or 0.0
        l = _mean(last, "loss_action") or 0.0
        findings.append(
            _finding(
                "runtime_action_loss_trend",
                l < f,
                severity="fail",
                detail=f"Action loss first-quartile mean={f:.6g}, last-quartile mean={l:.6g}.",
            )
        )
    if _mean(first, "loss_anchor_pv") is not None and _mean(last, "loss_anchor_pv") is not None:
        f = _mean(first, "loss_anchor_pv") or 0.0
        l = _mean(last, "loss_anchor_pv") or 0.0
        findings.append(
            _finding(
                "runtime_anchor_pv_not_worsening",
                l <= f * 1.05,
                severity="fail",
                detail=(
                    "Anchor PV is the point/visual routing-vs-projection constraint. "
                    f"first-quartile mean={f:.6g}, last-quartile mean={l:.6g}, ratio={l / max(f, 1e-12):.3f}."
                ),
            )
        )
    if _mean(first, "loss_pv_weak") is not None and _mean(last, "loss_pv_weak") is not None:
        f = _mean(first, "loss_pv_weak") or 0.0
        l = _mean(last, "loss_pv_weak") or 0.0
        findings.append(
            _finding(
                "runtime_pv_embedding_alignment_trend",
                l < f,
                severity="warn",
                detail=f"PV weak embedding loss first-quartile mean={f:.6g}, last-quartile mean={l:.6g}.",
            )
        )
    owm_keys = [key for key in rows[-1] if key.startswith("owm_") or key.startswith("aqr_temporal_") or key.startswith("posterior_identity")]
    findings.append(
        _finding(
            "runtime_current_run_has_owm_debug_keys",
            bool(owm_keys),
            severity="warn",
            detail=(
                "OWM debug keys indicate that the checkpoint was produced by the current final graph. "
                f"Found keys: {', '.join(sorted(owm_keys)[:12]) if owm_keys else '<none>'}."
            ),
        )
    )
    if _mean(last, "loss_mapg_support_diversity") is not None:
        val = _mean(last, "loss_mapg_support_diversity") or 0.0
        findings.append(
            _finding(
                "runtime_same_role_support_pressure_low",
                val < 0.35,
                severity="warn",
                detail=f"Last-quartile raw support-diversity loss={val:.6g}; high values indicate same-role support collapse pressure remains.",
            )
        )
    return findings


def _eval_findings(eval_dir: Path | None) -> list[Finding]:
    if eval_dir is None or not eval_dir.exists():
        return [
            Finding(
                name="calvin_eval_artifacts_available",
                status="WARN",
                severity="warn",
                detail="No CALVIN eval directory was provided; anchor drift/same-role overlap conclusions are not evaluated.",
                evidence=[],
            )
        ]
    findings = [
        Finding(
            name="calvin_eval_artifacts_available",
            status="PASS",
            severity="info",
            detail=f"CALVIN eval directory exists: {eval_dir}",
            evidence=[],
        )
    ]
    drift = eval_dir / "analysis_samples" / "anchor_drift_diag_ep0_ep1.txt"
    if drift.is_file():
        text = drift.read_text(encoding="utf-8")
        findings.append(
            _finding(
                "calvin_same_role_overlap_not_collapsed",
                "same_role_visual_overlap_max n=" in text and "mean=1.00" not in text,
                severity="fail",
                detail="same_role_*_overlap_max should be materially below 1.0 if anchors are separated.",
                evidence=[str(drift)],
            )
        )
        m = re.search(r"posterior\.pixel jump n=\d+ mean=([0-9.]+)", text)
        if m:
            mean_jump = float(m.group(1))
            findings.append(
                _finding(
                    "calvin_posterior_anchor_jump_reasonable",
                    mean_jump < 8.0,
                    severity="fail",
                    detail=f"Posterior pixel mean jump={mean_jump:.3f}; high values indicate unstable posterior anchor localization.",
                    evidence=[str(drift)],
                )
            )
    else:
        findings.append(
            Finding(
                name="calvin_anchor_drift_diag_present",
                status="WARN",
                severity="warn",
                detail="anchor_drift_diag_ep0_ep1.txt not found.",
                evidence=[],
            )
        )
    return findings


def run_static_checks() -> list[Finding]:
    contracts = Source("src/openpi/picf/core/contracts.py")
    config = Source("src/openpi/picf/core/config.py")
    pipeline = Source("src/openpi/picf/core/pipeline.py")
    training = Source("src/openpi/picf/core/training.py")
    trainer = Source("scripts/picf_core_train.py")
    evidence_bundle = Source("scripts/picf_owm_evidence_bundle.py")
    wrapper = Source("src/openpi/picf/vjepa/wrapper.py")
    burnin_body = pipeline.text.split("def recurrent_burnin_step", 1)[1].split("def _predictive_state", 1)[0]

    checks: list[Finding] = []
    checks.append(
        _finding(
            "production_defaults_use_direct_owm_profile",
            config.contains(
                "aqr_mapg_enabled: bool = True",
                "mapg_enabled: bool = False",
                "vl_anchor_router_enabled: bool = False",
                "aqr_pg_grounding_enabled: bool = False",
                "aqr_pg_image_support_enabled: bool = True",
                'aqr_vjepa_temporal_mode: str = "last_two_tokens"',
                "evidence_cache_read_weight: float = 0.05",
                "local_refinement_role_competition_enabled: bool = False",
                "local_refinement_coverage_seed_enabled",
            )
            and trainer.contains(
                "_LOSS_DEFAULTS = PicfTransitionLossConfig()",
                'default="paligemma"',
                "default=_SPEC_DEFAULTS.aqr_mapg_enabled",
                "default=_LOSS_DEFAULTS.lambda_mapg_cycle",
                "default=_LOSS_DEFAULTS.lambda_mapg_support_diversity",
                "default=_LOSS_DEFAULTS.lambda_slot_jepa",
            ),
            severity="fail",
            detail=(
                "No separate flags should be required for the latest OWM training profile: "
                "AQR is on, legacy routers are off, PaliGemma semantic mode is the CLI default, "
                "and loss defaults come from PicfTransitionLossConfig."
            ),
            evidence=config.refs(
                "aqr_mapg_enabled",
                "mapg_enabled",
                "aqr_vjepa_temporal_mode",
                "evidence_cache_read_weight",
                "local_refinement_role_competition_enabled",
                "local_refinement_coverage_seed_enabled",
            )
            + trainer.refs('default="paligemma"', "_LOSS_DEFAULTS", "default=_LOSS_DEFAULTS.lambda_mapg_cycle"),
        )
    )
    checks.append(
        _finding(
            "temporal_vjepa_preserves_time",
            wrapper.contains("def recent_maps") and pipeline.contains("fmap.recent_maps", "PicfTemporalVisualSupportState", "vjepa_temporal_priors"),
            severity="fail",
            detail="V-JEPA must provide recent maps and AQR must route over temporal visual support instead of only last-two mean.",
            evidence=wrapper.refs("def recent_maps") + pipeline.refs("fmap.recent_maps", "PicfTemporalVisualSupportState", "vjepa_temporal_priors"),
        )
    )
    checks.append(
        _finding(
            "pg_image_support_first_class",
            pipeline.contains("for index, (start, end) in enumerate(semantic.image_token_ranges)", "pg_priors[rows]", "pg_priors=pg_priors"),
            severity="fail",
            detail="PG image support must consume all image ranges/views and survive as graph.pg_priors.",
            evidence=pipeline.refs("for index, (start, end) in enumerate(semantic.image_token_ranges)", "pg_priors[rows]", "pg_priors=pg_priors"),
        )
    )
    checks.append(
        _finding(
            "mvtrack_proposal_memory_is_optional_typed_evidence",
            contracts.contains("class PicfPseudoProposalState", "proposal: PicfPseudoProposalState | None")
            and config.contains("proposal_memory_enabled: bool = True", "proposal_read_weight: float = 0.15")
            and pipeline.contains("aqr_proposal_reader", "proposal_priors", "graph_proposal_weights", "proposal_signature"),
            severity="fail",
            detail=(
                "Optional proposal memory must be typed evidence only: no proposal data is a no-op, "
                "and provided proposal boxes/tokens route through AQR rather than overwriting posterior identity."
            ),
            evidence=contracts.refs("class PicfPseudoProposalState", "proposal: PicfPseudoProposalState | None")
            + config.refs("proposal_memory_enabled", "proposal_read_weight")
            + pipeline.refs("aqr_proposal_reader", "proposal_priors", "graph_proposal_weights", "proposal_signature"),
        )
    )
    checks.append(
        _finding(
            "projective_geometry_reaches_alignment_losses",
            pipeline.contains("projective_compatibility", "projective_candidate_mask", "projective_attention_bias")
            and training.contains("anchor_pv", "_routing_consistency", "_mapg_cycle_loss"),
            severity="fail",
            detail=(
                "Projection creates point-visual compatibility. anchor_pv constrains observation-anchor routing; "
                "mapg_cycle is the AQR-graph bidirectional point/visual projection constraint and must be weighted in training."
            ),
            evidence=pipeline.refs("projective_compatibility", "projective_candidate_mask", "projective_attention_bias")
            + training.refs("def _routing_consistency", "def _mapg_cycle_loss", "loss_anchor_pv"),
        )
    )
    checks.append(
        _finding(
            "graph_pv_consistency_is_bidirectional",
            training.contains("point_from_visual", "visual_from_point", "point_cycle", "visual_cycle")
            and training.contains("lambda_mapg_cycle: float = 0.02", "lambda_mapg_support_diversity: float = 0.01"),
            severity="fail",
            detail=(
                "Graph PV consistency must directly compare graph.point_priors with visual->point projection "
                "and graph.visual_priors with point->visual projection. A pure visual->point->visual cycle can pass while point priors drift."
            ),
            evidence=training.refs("point_from_visual", "visual_from_point", "point_cycle", "visual_cycle", "lambda_mapg_cycle: float"),
        )
    )
    checks.append(
        _finding(
            "posterior_precision_update_intact",
            pipeline.contains("lambda_prior", "eta_prior", "lambda_meas", "eta_meas", "var_post", "mu_post"),
            severity="fail",
            detail="Posterior correction must retain precision-form prior+measurement fusion.",
            evidence=pipeline.refs("lambda_prior", "eta_prior", "lambda_meas", "eta_meas", "mu_post"),
        )
    )
    checks.append(
        _finding(
            "cache_read_is_previous_only_and_innovation_gated",
            pipeline.contains("getattr(previous.predictive, \"evidence_cache\", None)", "innovation_at_write", "evidence_cache_innovation_downweight", "source_factor"),
            severity="fail",
            detail="Evidence cache must read previous carry only and downweight stale/high-innovation entries.",
            evidence=pipeline.refs("getattr(previous.predictive, \"evidence_cache\", None)", "innovation_at_write", "evidence_cache_innovation_downweight", "source_factor"),
        )
    )
    checks.append(
        _finding(
            "cache_read_weight_scales_residual",
            pipeline.contains("q_before_cache", "cache_read - q_before_cache", "evidence_cache_read_weight")
            and "cache_bias = cache_bias + math.log(max(float(self.config.evidence_cache_read_weight)" not in pipeline.text,
            severity="fail",
            detail=(
                "evidence_cache_read_weight must scale the cache residual. Adding log(weight) as a constant "
                "attention bias is a softmax-invariant no-op once weight is positive."
            ),
            evidence=pipeline.refs("q_before_cache", "cache_read - q_before_cache", "evidence_cache_read_weight"),
        )
    )
    checks.append(
        _finding(
            "cache_skips_immediate_previous_posterior_duplicate",
            pipeline.contains("immediate_posterior", "valid = valid & ~immediate_posterior")
            and pipeline.contains("aqr_posterior_reader", "cache_read_active")
            and pipeline.contains("cache_roles", "role_mask"),
            severity="fail",
            detail=(
                "The cache must not re-read the newest posterior cache row, because previous.posterior.tokens "
                "already has a dedicated AQR posterior branch. Cache read should provide older episodic context "
                "and apply role-aware filtering before attention."
            ),
            evidence=pipeline.refs("immediate_posterior", "valid = valid & ~immediate_posterior", "cache_roles", "role_mask", "aqr_posterior_reader"),
        )
    )
    checks.append(
        _finding(
            "state_only_burnin_uses_aqr_graph_when_enabled",
            "if bool(self.config.aqr_mapg_enabled):" in burnin_body
            and "anchor_prior_graph = self._build_aqr_anchor_graph(" in burnin_body,
            severity="fail",
            detail=(
                "State-only burn-in must not use the legacy MAPG builder when AQR is enabled; "
                "otherwise burn-in and suffix posterior updates use different measurement models."
            ),
            evidence=pipeline.refs("def recurrent_burnin_step", "same AQR measurement model as the trainable suffix"),
        )
    )
    default_weight_match = re.search(r"evidence_cache_read_weight:\s*float\s*=\s*([0-9.]+)", config.text)
    default_weight = float(default_weight_match.group(1)) if default_weight_match else 0.0
    checks.append(
        _finding(
            "cache_read_weight_nonzero_by_default",
            default_weight > 0.0,
            severity="warn",
            detail=f"Default evidence_cache_read_weight={default_weight}. A zero default means cache is written but never read unless explicitly overridden.",
            evidence=config.refs("evidence_cache_read_weight"),
        )
    )
    checks.append(
        _finding(
            "slot_jepa_uses_detached_next_posterior",
            training.contains("future.posterior_tokens", "target_slots = future.posterior_tokens.detach()", "posterior_support_summary"),
            severity="fail",
            detail="Slot-JEPA/support prediction targets must come from detached next posterior, not future input leakage.",
            evidence=training.refs("future.posterior_tokens", "target_slots = future.posterior_tokens.detach()", "posterior_support_summary")
            + trainer.refs("future_targets_from_current_targets(current_targets, availability, posterior=posterior)"),
        )
    )
    checks.append(
        _finding(
            "aqr_denoising_is_training_only_and_guarded",
            config.contains("lambda_aqr_denoising: float = 0.0")
            and training.contains("lambda_aqr_denoising: float = 0.0", "_aqr_support_denoising_loss", "cfg.lambda_aqr_denoising")
            and trainer.contains("--lambda-aqr-denoising", "_LOSS_DEFAULTS.lambda_aqr_denoising", "loss_aqr_denoising"),
            severity="fail",
            detail=(
                "AQR denoising must be a guarded training-only support auxiliary with default zero weight; "
                "it must not introduce inference-time query/path changes."
            ),
            evidence=config.refs("lambda_aqr_denoising")
            + training.refs("lambda_aqr_denoising", "_aqr_support_denoising_loss", "cfg.lambda_aqr_denoising")
            + trainer.refs("--lambda-aqr-denoising", "loss_aqr_denoising"),
        )
    )
    checks.append(
        _finding(
            "binding_consistency_includes_temporal_matching",
            training.contains(
                "def _binding_consistency_loss",
                "future.posterior_tokens",
                "assign_row = torch.softmax",
                "matched_target",
                "matched_current",
                "future_tokens.detach()",
            )
            and "binding_entropy.mean()" not in training.text,
            severity="warn",
            detail=(
                "Binding consistency must include detached, permutation-tolerant next-posterior temporal matching, "
                "not only current-step binding entropy/sharpness."
            ),
            evidence=training.refs("def _binding_consistency_loss", "future.posterior_tokens", "fn.cross_entropy(logits, labels"),
        )
    )
    checks.append(
        _finding(
            "posterior_address_drift_false_positive_removed",
            "posterior_address_drift_mean" not in pipeline.text
            and "posterior_address_drift_mean" not in trainer.text
            and "posterior_address_drift_mean" not in evidence_bundle.text,
            severity="fail",
            detail=(
                "Current slot_address is a persistent carrier, not an identity-quality metric. "
                "Do not log address drift as an acceptance signal; rely on posterior_identity_switch_rate, recycle rate, and support overlap."
            ),
            evidence=pipeline.refs("posterior_address_drift_mean")
            + trainer.refs("posterior_address_drift_mean")
            + evidence_bundle.refs("posterior_address_drift_mean"),
        )
    )
    checks.append(
        _finding(
            "unused_ordinal_threshold_and_loss_key_removed",
            "ordinal_confidence_threshold" not in config.text
            and "ordinal_confidence_threshold" not in trainer.text
            and "ordinal_confidence_threshold" not in evidence_bundle.text
            and "ordinal_loss_active" not in pipeline.text
            and "ordinal_loss_active" not in trainer.text
            and "ordinal_loss_active" not in evidence_bundle.text,
            severity="fail",
            detail=(
                "Ordinal/relation is only a prompt-gated diagnostic until a real rank target exists. "
                "Do not expose unused confidence thresholds or loss-looking debug keys."
            ),
            evidence=config.refs("ordinal_confidence_threshold")
            + trainer.refs("ordinal_confidence_threshold", "ordinal_loss_active")
            + pipeline.refs("ordinal_loss_active")
            + evidence_bundle.refs("ordinal_confidence_threshold", "ordinal_loss_active"),
        )
    )
    checks.append(
        _finding(
            "trainer_logs_owm_debug_metrics",
            trainer.contains(
                "OWM_DEBUG_METRIC_KEYS",
                "aqr_same_role_support_overlap_max",
                "posterior_identity_switch_rate",
                "evidence_cache_trust_mean",
                "posterior_recycle_logit_mean",
                "posterior_dustbin_mass_raw",
                "posterior_address_update_rate_mean",
            ),
            severity="fail",
            detail="Training metrics must carry the OWM diagnostics needed to detect anchor collapse and cache misuse.",
            evidence=trainer.refs(
                "OWM_DEBUG_METRIC_KEYS",
                "aqr_same_role_support_overlap_max",
                "posterior_identity_switch_rate",
                "evidence_cache_trust_mean",
                "posterior_recycle_logit_mean",
                "posterior_dustbin_mass_raw",
                "posterior_address_update_rate_mean",
            ),
        )
    )
    checks.append(
        _finding(
            "contracts_expose_final_state_objects",
            contracts.contains("PicfTemporalVisualSupportState", "PicfEvidenceCacheState", "vjepa_temporal_priors", "slot_prediction_tokens", "ordinal_scores"),
            severity="fail",
            detail="Contracts must expose temporal support, fixed cache, temporal priors, slot prediction, and ordinal state.",
            evidence=contracts.refs("class PicfTemporalVisualSupportState", "class PicfEvidenceCacheState", "vjepa_temporal_priors", "slot_prediction_tokens", "ordinal_scores"),
        )
    )
    return checks


def render_markdown(findings: list[Finding]) -> str:
    fail_count = sum(1 for item in findings if item.status == "FAIL")
    warn_count = sum(1 for item in findings if item.status == "WARN")
    lines = [
        "# PICF-AQR-OWM Strict Diagnosis Temp",
        "",
        "This file is generated by `scripts/picf_owm_strict_diagnose.py`.",
        "",
        f"Summary: {fail_count} FAIL, {warn_count} WARN, {len(findings) - fail_count - warn_count} PASS/INFO.",
        "",
        "## Findings",
        "",
    ]
    for item in findings:
        lines.append(f"### {item.status}: {item.name}")
        lines.append("")
        lines.append(item.detail)
        if item.evidence:
            lines.append("")
            lines.append("Evidence:")
            for ref in item.evidence:
                lines.append(f"- `{ref}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict PICF-AQR-OWM static/runtime datapath diagnosis.")
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()

    findings = run_static_checks()
    findings.extend(_metric_trend_findings(_read_metrics(args.metrics_jsonl)))
    findings.extend(_eval_findings(args.eval_dir))

    payload = {"ok": not any(item.status == "FAIL" for item in findings), "findings": [item.__dict__ for item in findings]}
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(findings), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_fail and not payload["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
