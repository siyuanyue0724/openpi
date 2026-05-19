#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    severity: str
    detail: str
    evidence: list[str]

    @property
    def status(self) -> str:
        if self.ok:
            return "PASS"
        return "WARN" if self.severity == "warn" else "FAIL"


class Source:
    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.path = REPO_ROOT / relpath
        self.text = self.path.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()
        self.tree = ast.parse(self.text, filename=relpath) if relpath.endswith(".py") else None

    def contains(self, *needles: str) -> bool:
        return all(needle in self.text for needle in needles)

    def refs(self, *needles: str, limit: int = 24) -> list[str]:
        refs: list[str] = []
        for needle in needles:
            for line_no, line in enumerate(self.lines, start=1):
                if needle in line:
                    refs.append(f"{self.relpath}:{line_no}: {line.strip()}")
                    break
            if len(refs) >= limit:
                break
        return refs

    def node_source(self, name: str) -> str:
        if self.tree is None:
            raise ValueError(f"{self.relpath} is not Python")
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                start = int(getattr(node, "lineno", 1))
                end = int(getattr(node, "end_lineno", start))
                return "\n".join(self.lines[start - 1 : end])
        raise KeyError(f"{name} not found in {self.relpath}")

    def node_ref(self, name: str) -> list[str]:
        if self.tree is None:
            return []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                line_no = int(getattr(node, "lineno", 1))
                return [f"{self.relpath}:{line_no}: {self.lines[line_no - 1].strip()}"]
        return [f"{self.relpath}:?: missing function {name}"]


def make_check(name: str, ok: bool, *, detail: str, evidence: Iterable[str], severity: str = "fail") -> Check:
    return Check(name=name, ok=ok, severity=severity, detail=detail, evidence=list(evidence))


def ordered(text: str, *needles: str) -> bool:
    cursor = -1
    for needle in needles:
        nxt = text.find(needle, cursor + 1)
        if nxt < 0:
            return False
        cursor = nxt
    return True


def run_checks() -> list[Check]:
    config = Source("src/openpi/picf/core/config.py")
    contracts = Source("src/openpi/picf/core/contracts.py")
    pipeline = Source("src/openpi/picf/core/pipeline.py")
    tests = Source("src/openpi/picf/core/pipeline_test.py")
    trainer = Source("scripts/picf_core_train.py")
    bundle = Source("scripts/picf_owm_evidence_bundle.py")
    verifier = Source("scripts/verify_picf_owm_contract.py")
    professor = Source("scripts/picf_owm_professor_grade_audit.py")
    readme = Source("src/openpi/picf/README_v2.2.md")
    report = Source("docs/PICF_AQR_OWM_EXPERIMENT_REPORT_20260511_TEMP.md")

    owner_from_graph = pipeline.node_source("_observation_owner_active_from_graph")
    posterior_owner_bias = pipeline.node_source("_posterior_owner_active_binding_bias")
    posterior_update = pipeline.node_source("_posterior_update")
    build_obs = pipeline.node_source("_build_observation_anchors")
    debug = pipeline.text

    checks: list[Check] = []
    checks.append(
        make_check(
            "config_defaults_enable_owner_gate",
            config.contains(
                "posterior_owner_active_gate_enabled: bool = True",
                "posterior_owner_active_min: float = 0.25",
                "posterior_owner_active_bias: float = -1.0e4",
            ),
            detail="Owner/reserve posterior gating must be a production default, not a forgotten launch-only flag.",
            evidence=config.refs("posterior_owner_active_gate_enabled", "posterior_owner_active_min", "posterior_owner_active_bias"),
        )
    )
    checks.append(
        make_check(
            "observation_contract_carries_owner_active",
            contracts.contains("owner_active: torch.Tensor | None = None"),
            detail="The active owner decision must be carried on observation anchors, because posterior binding consumes observation anchors, not graph columns.",
            evidence=contracts.refs("owner_active"),
        )
    )
    checks.append(
        make_check(
            "owner_assignment_uses_margin_novelty_not_row_sum",
            all(
                needle in owner_from_graph
                for needle in (
                    "used",
                    "active_assignment",
                    "winner_margin",
                    "soft_owner",
                    "duplicate_terms",
                    "duplicate_score",
                    "torch.argsort(priority",
                    "active_cols",
                    "row_score = graph_assignment[:, col]",
                    "row_score = row_score.masked_fill(used, -1.0)",
                    "owner[row] = 1.0",
                    "torch.unique(roles).tolist()",
                )
            )
            and "graph_assignment @ anchor_active" not in owner_from_graph,
            detail=(
                "Owner extraction must not use row-stochastic active-column mass as reliability. It must keep unique "
                "graph-column owners while using winner margin and same-object novelty to demote duplicate reserve rows."
            ),
            evidence=pipeline.node_ref("_observation_owner_active_from_graph")
            + pipeline.refs(
                "active_assignment",
                "winner_margin",
                "soft_owner",
                "duplicate_terms",
                "duplicate_score",
                "used",
                "row_score = graph_assignment[:, col]",
                "owner[row] = 1.0",
                "torch.unique(roles).tolist()",
            ),
        )
    )
    checks.append(
        make_check(
            "owner_active_is_returned_by_observation_anchors",
            "owner_active = self._observation_owner_active_from_graph" in build_obs
            and "owner_active=owner_active" in build_obs,
            detail="The AQR graph owner state must become PicfObservationAnchorState.owner_active before posterior update.",
            evidence=pipeline.node_ref("_build_observation_anchors") + pipeline.refs("owner_active = self._observation_owner_active_from_graph", "owner_active=owner_active"),
        )
    )
    checks.append(
        make_check(
            "posterior_bias_masks_reserve_rows_with_role_fallback",
            all(
                needle in posterior_owner_bias
                for needle in (
                    "posterior_owner_active_gate_enabled",
                    "posterior_owner_active_min",
                    "eligible = owner_score >= threshold",
                    "for role_value in torch.unique(roles).tolist()",
                    "eligible[row] = True",
                    "posterior_owner_active_bias",
                    "masked_fill(~eligible, penalty)",
                )
            ),
            detail=(
                "Reserve rows must receive a large negative posterior-binding bias, while a per-role fallback keeps the measurement model non-empty."
            ),
            evidence=pipeline.node_ref("_posterior_owner_active_binding_bias")
            + pipeline.refs("eligible = owner_score >= threshold", "eligible[row] = True", "posterior_owner_active_bias"),
        )
    )
    checks.append(
        make_check(
            "posterior_applies_owner_bias_before_other_evidence_biases",
            ordered(
                posterior_update,
                "role_bias = self._posterior_binding_role_bias",
                "owner_bias = self._posterior_owner_active_binding_bias",
                "vl_bias = self._posterior_vl_binding_bias",
                "graph_bias = self._posterior_mapg_binding_bias",
                "occupancy_bias = self._posterior_occupancy_binding_bias",
                "binding_raw = self._sinkhorn_dustbin",
            )
            and "bind_logits = bind_logits + owner_bias" in posterior_update,
            detail=(
                "Owner/reserve eligibility is a measurement-eligibility mask. It must be applied before softer VL/graph/occupancy priors and before Sinkhorn-dustbin normalization."
            ),
            evidence=pipeline.node_ref("_posterior_update")
            + pipeline.refs("owner_bias = self._posterior_owner_active_binding_bias", "bind_logits = bind_logits + owner_bias"),
        )
    )
    checks.append(
        make_check(
            "diagnostics_surface_owner_gate_runtime_state",
            all(
                needle in debug
                for needle in (
                    "posterior_owner_active_score_mean",
                    "posterior_owner_active_score_max",
                    "posterior_owner_active_eligible_fraction",
                    "posterior_owner_active_gate_enabled",
                )
            ),
            detail="Runtime metrics must reveal whether the owner gate is active and how many observation rows remain eligible.",
            evidence=pipeline.refs(
                "posterior_owner_active_score_mean",
                "posterior_owner_active_score_max",
                "posterior_owner_active_eligible_fraction",
                "posterior_owner_active_gate_enabled",
            ),
        )
    )
    checks.append(
        make_check(
            "trainer_threads_cli_metrics_and_startup_contract",
            trainer.contains(
                "--posterior-owner-active-gate-enabled",
                "--posterior-owner-active-min",
                "--posterior-owner-active-bias",
                "posterior_owner_active_eligible_fraction",
                "posterior_owner_active_gate_enabled=%s",
            ),
            detail="Launch flags, startup log, and metrics keys must all include the owner gate. Otherwise a run can silently omit the repair.",
            evidence=trainer.refs(
                "--posterior-owner-active-gate-enabled",
                "--posterior-owner-active-min",
                "--posterior-owner-active-bias",
                "posterior_owner_active_eligible_fraction",
                "posterior_owner_active_gate_enabled=%s",
            ),
        )
    )
    checks.append(
        make_check(
            "evidence_bundle_and_verifiers_capture_owner_gate",
            bundle.contains("posterior_owner_active_eligible_fraction", "posterior_owner_active_min")
            and verifier.contains("posterior_owner_active_eligible_fraction", "--posterior-owner-active-gate-enabled")
            and professor.contains("active_owner_state_reaches_posterior_measurement_eligibility"),
            detail="Reviewer handoff scripts must carry the owner gate; otherwise this can regress without being seen.",
            evidence=bundle.refs("posterior_owner_active_eligible_fraction", "posterior_owner_active_min")
            + verifier.refs("posterior_owner_active_eligible_fraction", "--posterior-owner-active-gate-enabled")
            + professor.refs("active_owner_state_reaches_posterior_measurement_eligibility"),
        )
    )
    checks.append(
        make_check(
            "targeted_tests_cover_margin_novelty_owner_and_posterior_mask",
            tests.contains(
                "test_observation_owner_active_uses_margin_and_novelty_not_row_sum",
                "test_posterior_owner_active_binding_bias_masks_reserve_rows",
                "assert float(owner.mean().item()) < 0.70",
                "bias[:, 2:]",
            ),
            detail="Unit tests must verify winner-margin/novelty owner reliability, unique owner peaks, and posterior reserve-row masking.",
            evidence=tests.refs(
                "test_observation_owner_active_uses_margin_and_novelty_not_row_sum",
                "test_posterior_owner_active_binding_bias_masks_reserve_rows",
                "assert float(owner.mean().item()) < 0.70",
                "bias[:, 2:]",
            ),
        )
    )
    checks.append(
        make_check(
            "readme_and_experiment_report_record_math_and_limits",
            (
                readme.contains("owner/reserve posterior gate", "Owner_i", "posterior_owner_active_eligible_fraction")
                or readme.contains("owner/reserve posterior gate", "owner_i", "posterior_owner_active_eligible_fraction")
            )
            and report.contains("Owner/Reserve Posterior Gate", "The posterior binding logits then receive", "posterior_owner_active_eligible_fraction"),
            detail=(
                "The mathematical reason for the repair and its acceptance metrics must be documented in reviewer-facing files."
            ),
            evidence=readme.refs("owner/reserve posterior gate", "Owner_i", "posterior_owner_active_eligible_fraction")
            + report.refs("Owner/Reserve Posterior Gate", "The posterior binding logits then receive", "posterior_owner_active_eligible_fraction"),
        )
    )
    checks.append(
        make_check(
            "no_claim_that_owner_gate_creates_missing_evidence",
            readme.contains("does not create missing object evidence") or report.contains("does not create missing object evidence"),
            detail=(
                "The repair is a capacity/measurement eligibility correction. It must not be documented as creating sub-token evidence or solving ordinal/fine-instance grounding."
            ),
            evidence=readme.refs("does not create missing object evidence") + report.refs("does not create missing object evidence"),
            severity="warn",
        )
    )
    return checks


def render_markdown(checks: list[Check]) -> str:
    passed = sum(check.status == "PASS" for check in checks)
    failed = sum(check.status == "FAIL" for check in checks)
    warned = sum(check.status == "WARN" for check in checks)
    lines = [
        "# PICF-AQR-OWM Owner/Reserve Follow-Through Audit",
        "",
        f"Summary: pass={passed} warn={warned} fail={failed} total={len(checks)}",
        "",
        "This audit is intentionally narrower than the general OWM verifier. It checks the cross-layer chain that failed in the A7 long run:",
        "",
        "```text",
        "AQR graph active owners -> observation-anchor owner_active -> posterior binding eligibility -> metrics/evidence/docs",
        "```",
        "",
        "Mathematical invariant:",
        "",
        "```text",
        "inactive fixed-capacity reserve rows may remain in the graph for capacity accounting,",
        "but they must not update object posterior slots as ordinary measurements.",
        "```",
        "",
    ]
    for check in checks:
        lines.extend(
            [
                f"## {check.status} {check.name}",
                "",
                check.detail,
                "",
                "Evidence:",
            ]
        )
        if check.evidence:
            lines.extend(f"- {item}" for item in check.evidence)
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--markdown", action="store_true", help="Emit markdown evidence.")
    parser.add_argument("--fail-on-fail", action="store_true", help="Exit non-zero if any FAIL exists.")
    args = parser.parse_args()

    checks = run_checks()
    failed = [check for check in checks if check.status == "FAIL"]
    warned = [check for check in checks if check.status == "WARN"]
    passed = [check for check in checks if check.status == "PASS"]

    if args.json:
        print(
            json.dumps(
                {
                    "pass": len(passed),
                    "warn": len(warned),
                    "fail": len(failed),
                    "checks": [
                        {
                            "name": check.name,
                            "status": check.status,
                            "severity": check.severity,
                            "detail": check.detail,
                            "evidence": check.evidence,
                        }
                        for check in checks
                    ],
                },
                indent=2,
            )
        )
    elif args.markdown:
        print(render_markdown(checks))
    else:
        for check in checks:
            print(f"{check.status} {check.name}: {check.detail}")
            for item in check.evidence[:8]:
                print(f"  - {item}")
        print(f"SUMMARY pass={len(passed)} warn={len(warned)} fail={len(failed)} total={len(checks)}")

    if args.fail_on_fail and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
