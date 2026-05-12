#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
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
        self.tree = ast.parse(self.text, filename=relpath) if relpath.endswith(".py") else None

    def contains(self, *needles: str) -> bool:
        return all(needle in self.text for needle in needles)

    def lacks(self, *needles: str) -> bool:
        return all(needle not in self.text for needle in needles)

    def regex(self, pattern: str) -> bool:
        return re.search(pattern, self.text, flags=re.MULTILINE | re.DOTALL) is not None

    def refs(self, *needles: str, limit: int = 10) -> list[str]:
        refs: list[str] = []
        for needle in needles:
            for line_no, line in enumerate(self.lines, start=1):
                if needle in line:
                    refs.append(f"{self.relpath}:{line_no}: {line.strip()}")
                    break
            if len(refs) >= limit:
                break
        return refs

    def node_source(self, name: str, *, kind: type[ast.AST] = ast.FunctionDef) -> str:
        if self.tree is None:
            raise ValueError(f"{self.relpath} is not Python")
        for node in ast.walk(self.tree):
            if isinstance(node, kind) and getattr(node, "name", None) == name:
                start = int(getattr(node, "lineno", 1))
                end = int(getattr(node, "end_lineno", start))
                return "\n".join(self.lines[start - 1 : end])
        raise KeyError(f"{name} not found in {self.relpath}")

    def node_refs(self, name: str, *, kind: type[ast.AST] = ast.FunctionDef) -> list[str]:
        if self.tree is None:
            return []
        for node in ast.walk(self.tree):
            if isinstance(node, kind) and getattr(node, "name", None) == name:
                line = int(getattr(node, "lineno", 1))
                return [f"{self.relpath}:{line}: {self.lines[line - 1].strip()}"]
        return []


def _finding(name: str, ok: bool, *, severity: str, detail: str, evidence: Iterable[str] = ()) -> Finding:
    return Finding(
        name=name,
        status="PASS" if ok else ("WARN" if severity == "warn" else "FAIL"),
        severity=severity,
        detail=detail,
        evidence=list(evidence),
    )


def _ordered(text: str, *needles: str) -> bool:
    pos = -1
    for needle in needles:
        nxt = text.find(needle, pos + 1)
        if nxt < 0:
            return False
        pos = nxt
    return True


def _all_python_parse() -> list[Finding]:
    relpaths = [
        "src/openpi/picf/contracts.py",
        "src/openpi/picf/core/contracts.py",
        "src/openpi/picf/core/config.py",
        "src/openpi/picf/core/pipeline.py",
        "src/openpi/picf/core/training.py",
        "scripts/picf_core_train.py",
        "scripts/verify_picf_owm_contract.py",
        "scripts/picf_owm_strict_diagnose.py",
        "scripts/picf_owm_dataflow_trace.py",
        "scripts/picf_owm_evidence_bundle.py",
    ]
    findings: list[Finding] = []
    for relpath in relpaths:
        try:
            Source(relpath)
        except SyntaxError as exc:
            findings.append(
                Finding(
                    name=f"python_ast_parse_{relpath}",
                    status="FAIL",
                    severity="fail",
                    detail=f"AST parse failed: {exc}",
                    evidence=[relpath],
                )
            )
        else:
            findings.append(
                Finding(
                    name=f"python_ast_parse_{relpath}",
                    status="PASS",
                    severity="info",
                    detail="Python source parses cleanly.",
                    evidence=[relpath],
                )
            )
    return findings


def run_static_checks() -> list[Finding]:
    root_readme = Source("README.md")
    picf_readme = Source("src/openpi/picf/README.md")
    readme_v22 = Source("src/openpi/picf/README_v2.2.md")
    mvtrack_doc = Source("docs/PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md")
    contracts = Source("src/openpi/picf/core/contracts.py")
    obs_contracts = Source("src/openpi/picf/contracts.py")
    config = Source("src/openpi/picf/core/config.py")
    pipeline = Source("src/openpi/picf/core/pipeline.py")
    training = Source("src/openpi/picf/core/training.py")
    trainer = Source("scripts/picf_core_train.py")
    verifier = Source("scripts/verify_picf_owm_contract.py")
    strict = Source("scripts/picf_owm_strict_diagnose.py")
    dataflow = Source("scripts/picf_owm_dataflow_trace.py")
    evidence = Source("scripts/picf_owm_evidence_bundle.py")

    visual_maps = pipeline.node_source("_visual_maps")
    build_tokens = pipeline.node_source("_build_token_field")
    cache_read = pipeline.node_source("_previous_evidence_cache_tokens")
    aqr_graph = pipeline.node_source("_build_aqr_anchor_graph")
    binding = pipeline.node_source("_binding_logits")
    posterior = pipeline.node_source("_posterior_update")
    recurrent_burnin = pipeline.node_source("recurrent_burnin_step")
    predictive_state = pipeline.node_source("_predictive_state")
    matched_loss = training.node_source("_matched_prediction_loss")
    binding_loss = training.node_source("_binding_consistency_loss")
    denoise_loss = training.node_source("_aqr_support_denoising_loss")
    transition_loss = training.node_source("compute_transition_loss")

    code_sources = [contracts, obs_contracts, config, pipeline, training, trainer]
    code_text = "\n".join(src.text for src in code_sources)
    banned_active_knobs = (
        "aqr_temporal_memory_tokens",
        "posterior_address_drift_mean",
        "ordinal_confidence_threshold",
        "lambda_cross_modal_align",
        "lambda_ordinal_relation",
        "lambda_innovation_calib",
    )

    checks: list[Finding] = []
    checks.extend(_all_python_parse())
    checks.append(
        _finding(
            "readme_routing_is_canonical",
            root_readme.contains("src/openpi/picf/README_v2.2.md")
            and picf_readme.contains("README_v2.2.md")
            and readme_v22.contains("PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md"),
            severity="fail",
            detail="Repository and PICF README files must route reviewers to README_v2.2, which then links MVTrack.",
            evidence=root_readme.refs("README_v2.2.md") + picf_readme.refs("README_v2.2.md") + readme_v22.refs("PICF_AQR_OWM_MVTRACK_DEPLOYMENT_README.md"),
        )
    )
    checks.append(
        _finding(
            "docs_do_not_overclaim_behavior_completion",
            readme_v22.contains("code-level runtime completion")
            and readme_v22.contains("not a replacement")
            and mvtrack_doc.contains("behavior acceptance")
            and mvtrack_doc.contains("It does not create nonexistent information."),
            severity="fail",
            detail="Docs must distinguish code-level deployment from CALVIN/video behavior proof and information-limit guarantees.",
            evidence=readme_v22.refs("code-level runtime completion", "not a replacement")
            + mvtrack_doc.refs("behavior acceptance", "It does not create nonexistent information."),
        )
    )
    checks.append(
        _finding(
            "production_defaults_are_guarded_mvtrack",
            config.contains(
                "aqr_mapg_enabled: bool = True",
                "mapg_enabled: bool = False",
                "vl_anchor_router_enabled: bool = False",
                "aqr_pg_grounding_enabled: bool = False",
                "evidence_cache_read_weight: float = 0.05",
                "tracklet_memory_enabled: bool = True",
                "proposal_memory_enabled: bool = True",
                "local_refinement_enabled: bool = True",
                "local_refinement_role_competition_enabled: bool = False",
                "local_refinement_coverage_seed_enabled",
                "lambda_aqr_denoising: float = 0.0",
            )
            and training.contains(
                "lambda_slot_jepa: float = 0.0",
                "lambda_support_pred: float = 0.0",
                "lambda_binding_consistency: float = 0.0",
                "lambda_aqr_denoising: float = 0.0",
            ),
            severity="fail",
            detail="Runtime evidence branches may default on, but high-risk OWM training losses must stay zero by default.",
            evidence=config.refs("aqr_mapg_enabled", "tracklet_memory_enabled", "proposal_memory_enabled", "lambda_aqr_denoising")
            + training.refs("lambda_slot_jepa", "lambda_aqr_denoising"),
        )
    )
    checks.append(
        _finding(
            "stale_or_placeholder_knobs_absent_from_active_code",
            all(needle not in code_text for needle in banned_active_knobs),
            severity="fail",
            detail="Known stale/deceptive knobs and placeholder losses must not re-enter active code paths.",
            evidence=[needle for needle in banned_active_knobs if needle in code_text],
        )
    )
    checks.append(
        _finding(
            "mvtrack_contracts_are_complete",
            contracts.contains(
                "class PicfTrackletSupportState",
                "class PicfPseudoProposalState",
                "class PicfCacheReadState",
                "slot_content: torch.Tensor",
                "score: torch.Tensor",
                "proposal: PicfPseudoProposalState | None",
                "graph_proposal_weights",
                "proposal_signature",
                "ordinal_target_rank",
            )
            and obs_contracts.contains(
                "tracklet_xy",
                "proposal_centers_xy",
                "proposal_boxes_xyxy",
                "proposal_objectness",
            ),
            severity="fail",
            detail="Contracts must expose tracklets, optional proposals, dataclass cache-read metadata, support signatures, and weak ordinal fields.",
            evidence=contracts.refs("class PicfTrackletSupportState", "class PicfPseudoProposalState", "class PicfCacheReadState", "proposal_signature")
            + obs_contracts.refs("proposal_centers_xy", "tracklet_xy"),
        )
    )
    checks.append(
        _finding(
            "multiview_vjepa_uses_wrist_without_static_geometry_leak",
            "self.clip_buffers" in visual_maps
            and "_observation_rgb_for_view(observation, view_name)" in visual_maps
            and "rgb_gripper" in pipeline.text
            and "temporal_visual_view_embedding" in build_tokens
            and "non_static = view_ids != 0" in build_tokens
            and "temporal_ray = torch.where(non_static[:, None], torch.zeros_like(temporal_ray), temporal_ray)" in build_tokens
            and "temporal_cam = torch.where(non_static[:, None], torch.zeros_like(temporal_cam), temporal_cam)" in build_tokens,
            severity="fail",
            detail="Static and wrist V-JEPA views must be typed by view embedding; wrist must not inherit static camera ray/pose without extrinsics.",
            evidence=pipeline.node_refs("_visual_maps") + pipeline.refs("temporal_visual_view_embedding", "non_static = view_ids != 0"),
        )
    )
    checks.append(
        _finding(
            "tracklet_and_proposal_are_optional_typed_memory",
            build_tokens.count("if bool(self.config.tracklet_memory_enabled)") == 1
            and build_tokens.count("if bool(self.config.proposal_memory_enabled)") == 1
            and "observation.tracklet_xy is not None" in build_tokens
            and "observation.proposal_centers_xy" in build_tokens
            and "PicfTrackletSupportState(" in build_tokens
            and "PicfPseudoProposalState(" in build_tokens
            and "tracklet_read_weight" in aqr_graph
            and "proposal_read_weight" in aqr_graph,
            severity="fail",
            detail="Tracklet/proposal branches must be missing-modality no-ops and residual-gated typed evidence, not required hard inputs.",
            evidence=pipeline.refs("tracklet_memory_enabled", "proposal_memory_enabled", "tracklet_read_weight", "proposal_read_weight"),
        )
    )
    checks.append(
        _finding(
            "trainer_replay_serve_feed_optional_mvtrack_fields_when_present",
            trainer.contains(
                "_MVTRACK_TRACKLET_KEYS",
                "_MVTRACK_PROPOSAL_KEYS",
                "_read_npz_required_optional",
                "load_tracklet_fields",
                "tracklet_xy=frame.get(\"tracklet_xy\")",
                "proposal_centers_xy=frame.get(\"proposal_centers_xy\")",
            )
            and Source("src/openpi/picf/replay/calvin_replay.py").contains(
                "_MVTRACK_TRACKLET_KEYS",
                "_read_npz_required_optional",
                "tracklet_xy=frame.get(\"tracklet_xy\")",
                "proposal_centers_xy=frame.get(\"proposal_centers_xy\")",
            )
            and Source("scripts/serve_picf_policy.py").contains(
                "_optional_array",
                "tracklet_xy=_optional_array",
                "proposal_centers_xy=_optional_array",
            ),
            severity="fail",
            detail="Tracklet/proposal runtime branches must be fed by train/replay/serve when optional episode or service fields are present.",
            evidence=trainer.refs("_MVTRACK_TRACKLET_KEYS", "tracklet_xy=frame.get")
            + Source("src/openpi/picf/replay/calvin_replay.py").refs("_MVTRACK_TRACKLET_KEYS", "tracklet_xy=frame.get")
            + Source("scripts/serve_picf_policy.py").refs("_optional_array", "tracklet_xy=_optional_array"),
        )
    )
    checks.append(
        _finding(
            "local_refinement_uses_existing_typed_memory_not_visual_only",
            "_add_local_component(visual_priors, token_field.visual_tokens" in aqr_graph
            and "_add_local_component(" in aqr_graph
            and "vjepa_temporal_priors" in aqr_graph
            and "point_priors" in aqr_graph
            and "tracklet_priors" in aqr_graph
            and "proposal_priors" in aqr_graph
            and "local_token_indices" in aqr_graph
            and "local_refinement_weight" in aqr_graph
            and "_coverage_seed_selection_scores" in aqr_graph
            and "local_read" in aqr_graph,
            severity="fail",
            detail=(
                "Local refinement must aggregate top-k evidence from existing typed memories and apply "
                "guarded seeded candidate proposal before top-k selection when the diagnostic is enabled."
            ),
            evidence=pipeline.refs("_add_local_component", "_coverage_seed_selection_scores", "local_refinement_weight"),
        )
    )
    checks.append(
        _finding(
            "cache_read_state_is_dataclass_not_tuple_contract",
            "-> PicfCacheReadState" in cache_read
            and "return PicfCacheReadState(" in cache_read
            and "score=score.reshape" in cache_read
            and "slot_content=flat_tokens" in cache_read
            and "cache_read_state = self._previous_evidence_cache_tokens(previous)" in aqr_graph
            and "cache_tokens = cache_read_state.tokens" in aqr_graph,
            severity="fail",
            detail="Address-aware cache must flow through PicfCacheReadState metadata, not an unlabeled tuple that can drift from the contract.",
            evidence=pipeline.node_refs("_previous_evidence_cache_tokens") + pipeline.refs("cache_read_state = self._previous_evidence_cache_tokens"),
        )
    )
    checks.append(
        _finding(
            "cache_math_is_causal_address_aware_and_residual_gated",
            "previous.predictive" in cache_read
            and "immediate_posterior" in cache_read
            and "valid = valid & ~immediate_posterior" in cache_read
            and "innovation_cost" in cache_read
            and "evidence_cache_address_weight" in aqr_graph
            and "_aqr_cache_query_addresses" in pipeline.text
            and "previous.posterior.slot_address" in pipeline.text
            and "evidence_cache_content_weight" in aqr_graph
            and "evidence_cache_role_weight" in aqr_graph
            and "q_before_cache" in aqr_graph
            and "cache_read - q_before_cache" in aqr_graph
            and "math.log(max(float(self.config.evidence_cache_read_weight)" not in aqr_graph,
            severity="fail",
            detail="Cache must read only previous state, skip duplicate posterior, include age/uncertainty/innovation/address/content/role scoring, and scale the output residual.",
            evidence=pipeline.refs("immediate_posterior", "evidence_cache_address_weight", "cache_read - q_before_cache"),
        )
    )
    checks.append(
        _finding(
            "cache_write_is_after_posterior_correction",
            _ordered(recurrent_burnin, "innovation_token, innovation_norm", "posterior = self._posterior_update", "evidence_cache = self._write_evidence_cache")
            and _ordered(predictive_state, "posterior", "evidence_cache = self._write_evidence_cache"),
            severity="fail",
            detail="Innovation may be computed before posterior for identity gating, but evidence cache writes must happen after posterior correction.",
            evidence=pipeline.node_refs("recurrent_burnin_step") + pipeline.node_refs("_predictive_state"),
        )
    )
    checks.append(
        _finding(
            "binding_uses_support_overlap_then_gated_address",
            "support_terms" in binding
            and "tracklet_signature" in binding
            and "proposal_signature" in binding
            and "bind_support_signature_weight" in binding
            and "slot_address" in binding
            and "anchor_address" in binding
            and "bind_address_weight" in binding
            and "prev.recycle_gate" in binding
            and "innovation_norm" in binding
            and "_innovation_risk_scalar" in binding
            and "torch.exp(-float(self.config.bind_address_innovation_downweight)" in binding,
            severity="fail",
            detail="Identity binding must first use current support overlap and only then gated address inertia, downweighted by recycle/innovation risk.",
            evidence=pipeline.node_refs("_binding_logits") + pipeline.refs("bind_support_signature_weight", "bind_address_weight"),
        )
    )
    checks.append(
        _finding(
            "posterior_updates_address_slowly_and_resets_through_recycle",
            "base_address" in posterior
            and "obs_address = binding_cond @ obs_anchors.anchor_address" in posterior
            and "address_update_rate" in posterior
            and "1.0 - recycle.clamp" in posterior
            and "_measurement_innovation_norm(x_prior, S_prior, obs_anchors)" in posterior
            and "_innovation_risk_scalar(identity_innovation_norm)" in posterior
            and "address_update_max_rate" in posterior
            and "slot_address=slot_address" in posterior,
            severity="fail",
            detail="Slot address must be a slow, measurement-innovation-gated identity state rather than a hard immutable ID or predictive-cache side effect.",
            evidence=pipeline.node_refs("_posterior_update") + pipeline.refs("address_update_rate", "slot_address=slot_address"),
        )
    )
    checks.append(
        _finding(
            "recycle_diagnostics_explain_temporal_identity_failures",
            "recycle_logits=recycle_logits" in posterior
            and "recycle_support_mass_raw=support_mass_raw" in posterior
            and "recycle_dustbin_raw_mass=dustbin_raw.sum" in posterior
            and "identity_innovation_risk=identity_innovation_risk" in posterior
            and "address_update_rate=address_update_rate" in posterior
            and strict.contains("posterior_recycle_logit_mean", "posterior_dustbin_mass_raw", "posterior_address_update_rate_mean"),
            severity="fail",
            detail="Recycle saturation must be diagnosable from logits, input mass, dustbin transfer, innovation risk, and address-update rate before changing the posterior math.",
            evidence=pipeline.node_refs("_posterior_update")
            + pipeline.refs("recycle_logits", "recycle_dustbin_raw_mass", "address_update_rate")
            + strict.refs("posterior_recycle_logit_mean", "posterior_dustbin_mass_raw", "posterior_address_update_rate_mean"),
        )
    )
    checks.append(
        _finding(
            "matched_predictive_losses_are_permutation_tolerant_and_detached",
            "target.detach()" in matched_loss
            and "cost = 1.0 - (pred_n @ target_n.T)" in matched_loss
            and "assign_row = torch.softmax" in matched_loss
            and "assign_col = torch.softmax" in matched_loss
            and "assign.detach()" in matched_loss
            and "future.posterior_tokens.detach()" in transition_loss
            and "future.posterior_support_summary.detach()" in transition_loss
            and "fn.mse_loss(slot_tokens[:slot_count], future.posterior_tokens[:slot_count])" not in training.text,
            severity="fail",
            detail="Slot-JEPA/support prediction must use detached, permutation-tolerant matching instead of same-index future slot supervision.",
            evidence=training.node_refs("_matched_prediction_loss") + training.refs("future.posterior_tokens.detach", "future.posterior_support_summary.detach"),
        )
    )
    checks.append(
        _finding(
            "binding_consistency_temporal_term_is_matched_not_index_ce",
            "assign_row = torch.softmax" in binding_loss
            and "assign_col = torch.softmax" in binding_loss
            and "matched_target" in binding_loss
            and "matched_current" in binding_loss
            and "labels = torch.arange(slot_count" not in binding_loss,
            severity="fail",
            detail="Binding consistency must not assume current slot j is future slot j before the guarded loss is enabled.",
            evidence=training.node_refs("_binding_consistency_loss"),
        )
    )
    checks.append(
        _finding(
            "denoising_aux_is_training_only_guarded_and_not_runtime_query",
            "_aqr_support_denoising_loss" in training.text
            and "lambda_aqr_denoising: float = 0.0" in training.text
            and "cfg.lambda_aqr_denoising * aqr_denoising" in transition_loss
            and "never creates inference-time queries" in denoise_loss
            and "_aqr_support_denoising_loss" not in pipeline.text,
            severity="fail",
            detail="Denoising must remain a zero-default training-only support auxiliary, not an inference-time posterior/action source.",
            evidence=training.node_refs("_aqr_support_denoising_loss") + training.refs("lambda_aqr_denoising"),
        )
    )
    checks.append(
        _finding(
            "ordinal_is_weak_gated_and_does_not_rewrite_posterior",
            "ordinal_target_rank" in pipeline.text
            and "ordinal_selected_slot" in pipeline.text
            and "ordinal_confidence" in pipeline.text
            and "ordinal_weak_target_enabled" in config.text
            and "ordinal" not in posterior.lower(),
            severity="fail",
            detail="Ordinal relation support may expose weak task targets, but must not mutate posterior truth.",
            evidence=pipeline.refs("ordinal_target_rank", "ordinal_selected_slot", "ordinal_confidence"),
        )
    )
    checks.append(
        _finding(
            "trainer_and_bundle_expose_new_runtime_controls",
            trainer.contains(
                "--tracklet-memory-enabled",
                "--proposal-memory-enabled",
                "--local-refinement-enabled",
                "--lambda-aqr-denoising",
                "loss_aqr_denoising",
            )
            and evidence.contains(
                "proposal_memory_enabled",
                "lambda_aqr_denoising",
                "aqr_proposal_support_entropy_mean",
                "loss_aqr_denoising",
            ),
            severity="fail",
            detail="CLI, startup logs, metrics, and evidence bundles must expose MVTrack controls and denoising metrics for auditability.",
            evidence=trainer.refs("--proposal-memory-enabled", "--lambda-aqr-denoising", "loss_aqr_denoising")
            + evidence.refs("proposal_memory_enabled", "loss_aqr_denoising"),
        )
    )
    checks.append(
        _finding(
            "audit_scripts_cover_runtime_c_invariants",
            verifier.contains("proposal_priors", "lambda_aqr_denoising")
            and strict.contains("mvtrack_proposal_memory_is_optional_typed_evidence", "aqr_denoising_is_training_only_and_guarded")
            and dataflow.contains("Optional proposal typed memory", "Training-only support denoising"),
            severity="fail",
            detail="Verifier, strict diagnose, and dataflow trace must cover runtime-c proposal and denoising invariants.",
            evidence=verifier.refs("proposal_priors", "lambda_aqr_denoising")
            + strict.refs("mvtrack_proposal_memory_is_optional_typed_evidence", "aqr_denoising_is_training_only_and_guarded")
            + dataflow.refs("Optional proposal typed memory", "Training-only support denoising"),
        )
    )
    return checks


def write_markdown(findings: list[Finding], path: Path) -> None:
    fail_count = sum(1 for finding in findings if finding.status == "FAIL")
    warn_count = sum(1 for finding in findings if finding.status == "WARN")
    pass_count = sum(1 for finding in findings if finding.status == "PASS")
    lines = [
        "# PICF-AQR-OWM MVTrack Deep Audit",
        "",
        "This file is generated by `scripts/picf_owm_mvtrack_deep_audit.py`.",
        "",
        "## Summary",
        "",
        f"- PASS: {pass_count}",
        f"- WARN: {warn_count}",
        f"- FAIL: {fail_count}",
        "",
        "## Findings",
        "",
    ]
    for finding in findings:
        lines.extend(
            [
                f"### {finding.status}: {finding.name}",
                "",
                f"- Failure severity if violated: `{finding.severity}`",
                f"- Detail: {finding.detail}",
            ]
        )
        if finding.evidence:
            lines.append("- Evidence:")
            for ref in finding.evidence:
                lines.append(f"  - `{ref}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--fail-on-fail", action="store_true")
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    findings = run_static_checks()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps([finding.__dict__ for finding in findings], indent=2), encoding="utf-8")
    if args.markdown_out is not None:
        write_markdown(findings, args.markdown_out)
    for finding in findings:
        print(f"{finding.status} {finding.name}: {finding.detail}")
    has_fail = any(finding.status == "FAIL" for finding in findings)
    has_warn = any(finding.status == "WARN" for finding in findings)
    if args.fail_on_fail and has_fail:
        return 1
    if args.fail_on_warn and has_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
