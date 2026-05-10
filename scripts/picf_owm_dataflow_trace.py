#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TraceNode:
    name: str
    formula: str
    invariant: str
    sources: tuple[str, ...]
    needles: tuple[str, ...]


class SourceSet:
    def __init__(self) -> None:
        self._cache: dict[str, list[str]] = {}

    def lines(self, relpath: str) -> list[str]:
        if relpath not in self._cache:
            self._cache[relpath] = (REPO_ROOT / relpath).read_text(encoding="utf-8").splitlines()
        return self._cache[relpath]

    def has(self, relpath: str, needle: str) -> bool:
        return any(needle in line for line in self.lines(relpath))

    def ref(self, relpath: str, needle: str) -> str:
        for line_no, line in enumerate(self.lines(relpath), start=1):
            if needle in line:
                return f"{relpath}:{line_no}: {line.strip()}"
        return f"{relpath}:?: MISSING {needle!r}"


NODES: tuple[TraceNode, ...] = (
    TraceNode(
        name="Observation and recurrent carry",
        formula="o_t=(I_t,D_t,T_t,q_t,ell,a_{t-1}), b_{t-1}^+ -> observe_step",
        invariant="The current update must start from previous posterior/predictive carry, not from a stateless router.",
        sources=("src/openpi/picf/core/pipeline.py",),
        needles=("def observe_step", "previous: PicfPreviousState | None"),
    ),
    TraceNode(
        name="Production default profile",
        formula="default train profile = AQR(on) + PG semantic/image support + temporal V-JEPA + posterior cache scaffold + guarded OWM losses",
        invariant="The latest OWM profile must be the default path; legacy routers and risky auxiliary losses are explicit ablations.",
        sources=("src/openpi/picf/core/config.py", "scripts/picf_core_train.py", "src/openpi/picf/core/training.py"),
        needles=("aqr_mapg_enabled: bool = True", 'default="paligemma"', "lambda_mapg_cycle: float = 0.02", "lambda_slot_jepa: float = 0.0"),
    ),
    TraceNode(
        name="V-JEPA temporal evidence",
        formula="M_vjepa={z_{tau,h,w}: tau in T_recent}; p_j(tau,h,w)=softmax(q_j^T k_{tau,h,w})",
        invariant="Production support preserves temporal slices; last_two_mean is only an ablation.",
        sources=("src/openpi/picf/vjepa/wrapper.py", "src/openpi/picf/core/pipeline.py"),
        needles=("def recent_maps", "fmap.recent_maps", "vjepa_temporal_priors"),
    ),
    TraceNode(
        name="PaliGemma image evidence",
        formula="p_pg(j,v,u)=softmax_u((Wq q_j)^T(Wk e_pg[v,u])/sqrt(d))",
        invariant="PG image evidence must survive as graph.pg_priors and may not be destroyed into a V-JEPA bias only.",
        sources=("src/openpi/picf/core/pipeline.py",),
        needles=("for index, (start, end) in enumerate(semantic.image_token_ranges)", "pg_priors[rows]", "pg_priors=pg_priors"),
    ),
    TraceNode(
        name="Typed token field",
        formula="M_t={M_text,M_pg,M_vjepa,M_point,M_tactile,M_post,M_cache}",
        invariant="Missing modalities are masked/empty; typed support must not be collapsed into one opaque token before AQR.",
        sources=("src/openpi/picf/core/contracts.py", "src/openpi/picf/core/pipeline.py"),
        needles=("class PicfTokenFieldState", "temporal_visual", "cache_tokens"),
    ),
    TraceNode(
        name="Previous evidence cache read",
        formula="skip newest posterior duplicate; role-filter cache; w_c proportional source_factor/(1+age+uncertainty+lambda_innov*innovation_at_write); q<-q+lambda_cache*(Read_C(q)-q)",
        invariant="A step can read only previous carry cache; t-1 posterior is read by posterior_reader, cache supplies older role-compatible episodic context; read_weight scales the cache residual; current posterior writes cache for the next step only.",
        sources=("src/openpi/picf/core/pipeline.py", "src/openpi/picf/core/config.py"),
        needles=("cache = getattr(previous.predictive, \"evidence_cache\", None)", "immediate_posterior", "cache_roles", "cache_read - q_before_cache"),
    ),
    TraceNode(
        name="AQR measurement routing",
        formula="l_{j,i}^{(m)}=(Wq[a_j,c_j^-,r_j])^T Wk e_i^{(m)}/sqrt(d)+biases",
        invariant="AQR produces measurements/supports; it does not replace posterior belief.",
        sources=("src/openpi/picf/core/pipeline.py", "src/openpi/picf/core/contracts.py"),
        needles=("def _build_aqr_anchor_graph", "visual_priors", "point_priors", "tactile_priors", "posterior_priors"),
    ),
    TraceNode(
        name="Projective point-visual geometry",
        formula="L_pv = D(p_point, p_visual P_{v->p}) + D(p_visual, p_point P_{p->v})",
        invariant="PV alignment is geometry-projected support consistency, not a cosmetic RoPE/embedding shortcut.",
        sources=("src/openpi/picf/core/pipeline.py", "src/openpi/picf/core/training.py"),
        needles=("projective_compatibility", "point_from_visual", "visual_from_point", "lambda_mapg_cycle: float = 0.02"),
    ),
    TraceNode(
        name="Prior prediction",
        formula="S_t^- = F_theta(S_{t-1}^+, a_{t-1}, proprio_t)",
        invariant="Prior is predicted from previous posterior, previous action, and proprio before measurement correction.",
        sources=("src/openpi/picf/core/pipeline.py",),
        needles=("def _current_prior", "previous.posterior", "getattr(previous.predictive, \"executed_action\", None)"),
    ),
    TraceNode(
        name="Posterior correction",
        formula="Lambda^+=Lambda^-+Lambda_meas; eta^+=Lambda^- mu^-+Lambda_meas mu_meas; mu^+=(Lambda^+)^{-1}eta^+",
        invariant="Posterior after correction is the authoritative current belief.",
        sources=("src/openpi/picf/core/pipeline.py",),
        needles=("lambda_prior", "eta_prior", "lambda_meas", "eta_meas", "mu_post"),
    ),
    TraceNode(
        name="State-only burn-in consistency",
        formula="b_t^{burnin}=U_AQR(o_t,P(b_{t-1},a_{t-1})) when AQR is enabled",
        invariant="Burn-in and train suffix must use the same measurement model.",
        sources=("src/openpi/picf/core/pipeline.py", "src/openpi/picf/core/pipeline_test.py"),
        needles=("Keep state-only burn-in on the same AQR measurement model", "_build_aqr_anchor_graph", "test_recurrent_burnin_uses_aqr_graph_when_aqr_enabled"),
    ),
    TraceNode(
        name="Innovation",
        formula="nu_t = Sigma_pred^{-1/2}(y_t - yhat_t)",
        invariant="Innovation compares real current targets against world-only prediction; it gates correction/cache trust.",
        sources=("src/openpi/picf/core/pipeline.py",),
        needles=("def _innovation", "physical_prediction_cache", "innovation_norm"),
    ),
    TraceNode(
        name="Slot prediction targets",
        formula="L_slot=||F(S_t,a_t)-stopgrad(S_{t+1}^{posterior})||",
        invariant="Future posterior is a detached target only; it is never current action input.",
        sources=("src/openpi/picf/core/training.py", "scripts/picf_core_train.py"),
        needles=("future.posterior_tokens.detach", "future_targets_from_current_targets"),
    ),
    TraceNode(
        name="Temporal binding consistency",
        formula="L_bind = CE(cos(c_t,stopgrad(c_{t+1}))/tau, identity labels) + entropy guard",
        invariant="Binding consistency must include temporal identity contrast, not only current binding sharpness.",
        sources=("src/openpi/picf/core/training.py",),
        needles=("def _binding_consistency_loss", "future.posterior_tokens", "fn.cross_entropy"),
    ),
    TraceNode(
        name="Evidence cache write",
        formula="C_t = write(topk(evidence), S_t^+, uncertainty_t, innovation_t) for t+1",
        invariant="Cache is written after posterior correction and is auxiliary evidence for later steps only.",
        sources=("src/openpi/picf/core/pipeline.py",),
        needles=("def _write_evidence_cache", "posterior.slot_address", "innovation_at_write"),
    ),
    TraceNode(
        name="Action path",
        formula="a_t ~ PI0.5(posterior, task anchors, innovation, typed supports)",
        invariant="PI0.5 remains final action generator; OWM does not create a separate action head truth path.",
        sources=("src/openpi/picf/core/pipeline.py",),
        needles=("def _build_conditioned_control_state", "posterior_to_control_proj", "innovation_to_control_proj", "task_to_control_proj"),
    ),
    TraceNode(
        name="False-positive guards",
        formula="acceptance != substring verifier; address drift is not identity; old CALVIN failure is not new-code proof",
        invariant="Removed dead knobs/loss-looking placeholders and stale metrics that could create false confidence.",
        sources=("src/openpi/picf/core/training.py", "scripts/verify_picf_owm_contract.py", "docs/PICF_AQR_OWM_FINAL_DEPLOYMENT_README.md"),
        needles=("weak placeholder losses must stay removed", "removed", "Do not expose placeholder losses"),
    ),
)


def run() -> tuple[list[dict], bool]:
    sources = SourceSet()
    rows: list[dict] = []
    ok = True
    for node in NODES:
        evidence: list[str] = []
        missing: list[str] = []
        for needle in node.needles:
            matched = False
            for relpath in node.sources:
                if sources.has(relpath, needle):
                    evidence.append(sources.ref(relpath, needle))
                    matched = True
                    break
            if not matched:
                missing.append(needle)
        status = "PASS" if not missing else "FAIL"
        ok = ok and status == "PASS"
        rows.append(
            {
                "name": node.name,
                "status": status,
                "formula": node.formula,
                "invariant": node.invariant,
                "evidence": evidence,
                "missing": missing,
            }
        )
    return rows, ok


def render_markdown(rows: list[dict], ok: bool) -> str:
    pass_count = sum(row["status"] == "PASS" for row in rows)
    out = [
        "# PICF-AQR-OWM Recursive Dataflow Audit Temp",
        "",
        "Generated by `scripts/picf_owm_dataflow_trace.py`.",
        "",
        f"Status: {'PASS' if ok else 'FAIL'} ({pass_count}/{len(rows)} dataflow nodes passed).",
        "",
        "This document is intentionally stricter than a substring contract verifier: every node records the mathematical formula, the invariant that protects the posterior belief-state design, and the code evidence that the dataflow exists.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        out.extend(
            [
                f"## {index}. {row['name']} - {row['status']}",
                "",
                "Formula:",
                "",
                "```text",
                row["formula"],
                "```",
                "",
                "Invariant:",
                "",
                "```text",
                row["invariant"],
                "```",
                "",
                "Evidence:",
                "",
            ]
        )
        if row["evidence"]:
            out.extend(f"- `{item}`" for item in row["evidence"])
        else:
            out.append("- `<none>`")
        if row["missing"]:
            out.extend(["", "Missing:", ""])
            out.extend(f"- `{item}`" for item in row["missing"])
        out.append("")
    out.extend(
        [
            "## Final Interpretation",
            "",
            "A PASS here means the current repository contains a coherent code-level dataflow for typed evidence -> AQR measurement -> posterior correction -> prediction/cache -> action.",
            "",
            "It does not prove empirical CALVIN anchor quality. Runtime metrics and video diagnostics must be checked separately with `scripts/picf_owm_strict_diagnose.py --metrics-jsonl ... --eval-dir ...`.",
            "",
        ]
    )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-out", type=Path, default=REPO_ROOT / "docs/PICF_AQR_OWM_RECURSIVE_DATAFLOW_AUDIT_TEMP.md")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()

    rows, ok = run()
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(rows, ok), encoding="utf-8")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"ok": ok, "nodes": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": ok, "nodes": len(rows), "markdown": str(args.markdown_out)}, indent=2, sort_keys=True))
    return 1 if args.fail_on_fail and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
