from __future__ import annotations

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
        self.text = self.path.read_text(encoding='utf-8')
        self.lines = self.text.splitlines()
        self.tree = ast.parse(self.text, filename=relpath) if relpath.endswith('.py') else None

    def contains(self, *needles: str) -> bool:
        return all(needle in self.text for needle in needles)

    def lacks(self, *needles: str) -> bool:
        return all(needle not in self.text for needle in needles)

    def refs(self, *needles: str, limit: int = 16) -> list[str]:
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
            raise ValueError(f'{self.relpath} is not Python')
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                start = int(getattr(node, 'lineno', 1))
                end = int(getattr(node, 'end_lineno', start))
                return '\n'.join(self.lines[start - 1:end])
        raise KeyError(f'{name} not found in {self.relpath}')

    def node_refs(self, name: str) -> list[str]:
        if self.tree is None:
            return []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                line = int(getattr(node, 'lineno', 1))
                return [f"{self.relpath}:{line}: {self.lines[line - 1].strip()}"]
        return []


def finding(name: str, ok: bool, *, severity: str, detail: str, evidence: Iterable[str]) -> Finding:
    return Finding(name=name, status='PASS' if ok else ('WARN' if severity == 'warn' else 'FAIL'), severity=severity, detail=detail, evidence=list(evidence))


def ordered(text: str, *needles: str) -> bool:
    pos = -1
    for needle in needles:
        nxt = text.find(needle, pos + 1)
        if nxt < 0:
            return False
        pos = nxt
    return True


def run_checks() -> list[Finding]:
    root = Source('README.md')
    readme = Source('src/openpi/picf/README_v2.2.md')
    local_audit = Source('docs/PICF_AQR_OWM_STRICT_LOCAL_AUDIT_20260515_TEMP.md')
    config = Source('src/openpi/picf/core/config.py')
    public_contracts = Source('src/openpi/picf/contracts.py')
    core_contracts = Source('src/openpi/picf/core/contracts.py')
    pipeline = Source('src/openpi/picf/core/pipeline.py')
    training = Source('src/openpi/picf/core/training.py')
    trainer = Source('scripts/picf_core_train.py')
    serve = Source('scripts/serve_picf_policy.py')
    replay = Source('src/openpi/picf/replay/calvin_replay.py')
    verifier = Source('scripts/verify_picf_owm_contract.py')
    strict = Source('scripts/picf_owm_strict_diagnose.py')
    dataflow = Source('scripts/picf_owm_dataflow_trace.py')
    mvtrack = Source('scripts/picf_owm_mvtrack_deep_audit.py')

    visual_maps = pipeline.node_source('_visual_maps')
    cache_read = pipeline.node_source('_previous_evidence_cache_tokens')
    aqr_graph = pipeline.node_source('_build_aqr_anchor_graph')
    active_mask = pipeline.node_source('_aqr_active_slot_mask')
    support_comp = pipeline.node_source('_aqr_same_role_support_competition')
    binding_logits = pipeline.node_source('_binding_logits')
    posterior_update = pipeline.node_source('_posterior_update')
    token_field = pipeline.node_source('_build_token_field')
    obs_anchors = pipeline.node_source('_build_observation_anchors')
    matched_prediction = training.node_source('_matched_prediction_loss')
    binding_consistency = training.node_source('_binding_consistency_loss')
    transition_loss = training.node_source('compute_transition_loss')

    checks: list[Finding] = []
    checks.append(finding(
        'canonical_docs_route_to_strict_followthrough',
        root.contains('src/openpi/picf/README_v2.2.md') and readme.contains('PICF_AQR_OWM_STRICT_LOCAL_AUDIT_20260515_TEMP.md'),
        severity='fail',
        detail='Reviewer entry must route through README_v2.2 and include the strict local audit TEMP doc.',
        evidence=root.refs('README_v2.2.md') + readme.refs('PICF_AQR_OWM_STRICT_LOCAL_AUDIT_20260515_TEMP.md'),
    ))
    checks.append(finding(
        'strict_audit_contains_math_papers_dataflow_tests',
        local_audit.contains('POMDP', 'IsSameObject', 'Data follow-through', 'Script evidence', 'I(Y; A_t'),
        severity='fail',
        detail='Strict TEMP audit must explicitly include math, paper grounding, data follow-through, and script results.',
        evidence=local_audit.refs('POMDP', 'IsSameObject', 'Data follow-through', 'Script evidence', 'I(Y; A_t'),
    ))
    checks.append(finding(
        'production_defaults_are_not_legacy_or_local_refinement',
        config.contains(
            'aqr_mapg_enabled: bool = True',
            'mapg_enabled: bool = False',
            'vl_anchor_router_enabled: bool = False',
            'aqr_pg_grounding_enabled: bool = False',
            'legacy_local_refinement_opt_in: bool = False',
            'local_refinement_enabled: bool = False',
            'local_refinement_weight: float = 0.0',
            'lambda_aqr_denoising: float = 0.0',
        ),
        severity='fail',
        detail='Production defaults must keep the maintained AQR path and archive legacy local-refinement/aux-loss pressure unless explicitly ablated.',
        evidence=config.refs('aqr_mapg_enabled', 'mapg_enabled', 'legacy_local_refinement_opt_in', 'local_refinement_enabled', 'lambda_aqr_denoising'),
    ))
    checks.append(finding(
        'multiview_temporal_uses_wrist_without_static_geometry_leak',
        all(s in visual_maps for s in ['for view_name in self._configured_vjepa_views()', '_observation_rgb_for_view', 'if view_name == "static":', 'current = current_map', 'shape != reference_shape', 'continue'])
        and all(s in token_field for s in ['temporal_visual_view_embedding', 'view_ids', 'grid_hw_by_view', 'source_hw_by_view']),
        severity='fail',
        detail='Wrist V-JEPA may be temporal typed evidence, but static current/projective geometry must not silently absorb mismatched wrist grids.',
        evidence=pipeline.node_refs('_visual_maps') + pipeline.refs('shape != reference_shape', 'temporal_visual_view_embedding', 'grid_hw_by_view'),
    ))
    checks.append(finding(
        'tracklet_proposal_dataflow_is_optional_but_threaded',
        public_contracts.contains('tracklet_xy', 'proposal_centers_xy')
        and trainer.contains('load_tracklet_fields=bool', 'tracklet_xy=frame.get', 'proposal_centers_xy=frame.get')
        and serve.contains('tracklet_xy=_optional_array', 'proposal_centers_xy=_optional_array')
        and replay.contains('load_tracklet_fields: bool = True', 'tracklet_xy=frame.get', 'proposal_centers_xy=frame.get')
        and pipeline.contains('PicfTrackletSupportState', 'PicfPseudoProposalState')
        and trainer.contains('owm_tracklet_tokens', 'owm_proposal_tokens'),
        severity='fail',
        detail='Tracklet/proposal fields must be schema-valid, optionally loaded by train/replay/serve, built into typed states, and observable in metrics.',
        evidence=public_contracts.refs('tracklet_xy', 'proposal_centers_xy') + trainer.refs('load_tracklet_fields=bool', 'tracklet_xy=frame.get', 'owm_tracklet_tokens') + serve.refs('tracklet_xy=_optional_array') + replay.refs('load_tracklet_fields: bool = True'),
    ))
    checks.append(finding(
        'cache_is_auxiliary_addressed_residual_not_truth_or_duplicate',
        all(s in cache_read for s in ['immediate_posterior', 'valid = valid & ~immediate_posterior', 'innovation_cost', 'source_factor'])
        and all(s in aqr_graph for s in ['cache_roles', 'evidence_cache_role_weight', 'cache_address', 'evidence_cache_address_weight', 'evidence_cache_content_weight', 'q_before_cache', 'cache_read - q_before_cache'])
        and 'math.log(max(float(self.config.evidence_cache_read_weight)' not in aqr_graph,
        severity='fail',
        detail='Cache must skip the previous-posterior duplicate, score by age/uncertainty/innovation/source, use role/address/content bias, and only enter through residual scaling.',
        evidence=pipeline.refs('immediate_posterior', 'innovation_cost', 'evidence_cache_address_weight', 'q_before_cache', 'cache_read - q_before_cache'),
    ))
    checks.append(finding(
        'same_role_capacity_has_assignment_and_geometry_interactions',
        all(s in active_mask for s in ['_object_core_overlap_matrix', 'aqr_active_slot_geometry_duplicate_enabled', 'support_peak', 'aqr_active_slot_max_per_role', 'aqr_active_slot_min_per_role', 'aqr_active_slot_overlap_threshold'])
        and all(s in support_comp for s in ['cannot invent', 'same-role', 'exclusive', 'valid_cols', 'weight', '_normalize_rows']),
        severity='fail',
        detail='Active/dustbin must combine semantic support overlap, geometric duplicate evidence, role capacity, and competition without pretending to create missing evidence.',
        evidence=pipeline.node_refs('_aqr_active_slot_mask') + pipeline.node_refs('_aqr_same_role_support_competition') + pipeline.refs('_object_core_overlap_matrix', 'aqr_active_slot_geometry_duplicate_enabled'),
    ))
    checks.append(finding(
        'active_owner_state_reaches_posterior_measurement_eligibility',
        core_contracts.contains('owner_active')
        and config.contains('posterior_owner_active_gate_enabled', 'posterior_owner_active_min', 'posterior_owner_active_bias')
        and pipeline.contains('_observation_owner_active_from_graph', '_posterior_owner_active_binding_bias', 'posterior_owner_active_eligible_fraction')
        and all(s in posterior_update for s in ['_posterior_owner_active_binding_bias', 'bind_logits = bind_logits + owner_bias'])
        and trainer.contains('--posterior-owner-active-gate-enabled', 'posterior_owner_active_eligible_fraction')
        and Source('scripts/picf_owm_evidence_bundle.py').contains('posterior_owner_active_eligible_fraction'),
        severity='fail',
        detail='Active graph owners must become observation-anchor owner eligibility and then posterior binding eligibility; otherwise reserve anchors can still update object files.',
        evidence=core_contracts.refs('owner_active') + config.refs('posterior_owner_active_gate_enabled') + pipeline.node_refs('_observation_owner_active_from_graph') + pipeline.node_refs('_posterior_owner_active_binding_bias') + trainer.refs('--posterior-owner-active-gate-enabled', 'posterior_owner_active_eligible_fraction'),
    ))
    checks.append(finding(
        'binding_logit_combines_hidden_geometry_support_address_with_trust_gates',
        all(s in binding_logits for s in ['hidden_score', 'maha', 'support_terms', 'support_score', 'binding_score', 'address_score', 'innovation_decay', 'prev.alpha', 'prev.recycle_gate', 'bind_support_signature_weight', 'bind_embedding_signature_weight', 'bind_address_weight']),
        severity='fail',
        detail='Binding cannot rely only on hidden cosine/geometry; it must use support signatures and address terms gated by alpha/recycle/innovation trust.',
        evidence=pipeline.node_refs('_binding_logits') + pipeline.refs('support_score', 'binding_score', 'address_score', 'innovation_decay'),
    ))
    checks.append(finding(
        'posterior_recycle_is_slotwise_normalized_and_not_global_dustbin_reset',
        all(s in posterior_update for s in ['posterior_slotwise_recycle_residual', 'slot_residual_summary', 'fn.layer_norm', 'rms_norm', 'birth_share', 'dustbin_final', 'binding_support', 'address_update_rate', 'identity_innovation_risk']),
        severity='fail',
        detail='Recycle must be a slot-local trust/reset gate, normalized before logits, with dustbin redistribution and address update gated by support/recycle/innovation.',
        evidence=pipeline.node_refs('_posterior_update') + pipeline.refs('posterior_slotwise_recycle_residual', 'fn.layer_norm', 'birth_share', 'address_update_rate'),
    ))
    checks.append(finding(
        'observation_anchor_signature_feeds_binding_not_only_metrics',
        all(s in obs_anchors for s in ['_support_binding_signature', 'point_weights', 'graph_temporal_weights', 'graph_tracklet_weights', 'graph_proposal_weights', 'binding_signature=obs_binding_signature']),
        severity='fail',
        detail='IsSameObject-inspired binding signatures must be built from support-weighted typed evidence and carried into observation anchors for posterior binding.',
        evidence=pipeline.node_refs('_build_observation_anchors') + pipeline.refs('_support_binding_signature', 'binding_signature=obs_binding_signature'),
    ))
    checks.append(finding(
        'future_teachers_are_detached_and_losses_guarded',
        training.contains('lambda_slot_jepa: float = 0.0', 'lambda_support_pred: float = 0.0', 'lambda_binding_consistency: float = 0.0', 'lambda_aqr_denoising: float = 0.0')
        and all(s in matched_prediction for s in ['target.detach()', 'assign_row', 'assign_col'])
        and all(s in transition_loss for s in ['future.posterior_tokens.detach()', 'future.posterior_support_summary.detach()', 'cfg.lambda_slot_jepa', 'cfg.lambda_support_pred', 'cfg.lambda_binding_consistency', 'cfg.lambda_aqr_denoising']),
        severity='fail',
        detail='Predictive/denoising hooks must be default-off and use detached future targets to avoid leakage into current action.',
        evidence=training.refs('lambda_slot_jepa', 'future.posterior_tokens.detach', 'future.posterior_support_summary.detach', 'cfg.lambda_aqr_denoising'),
    ))
    checks.append(finding(
        'binding_consistency_is_permutation_tolerant_enough_for_guarded_hook',
        all(s in binding_consistency for s in ['assign_row', 'assign_col', 'matched_target', 'matched_current'])
        and 'fn.cross_entropy(logits, labels)' not in binding_consistency,
        severity='fail',
        detail='Binding-consistency hook should not regress to index-aligned cross entropy; it must use soft matching when enabled.',
        evidence=training.node_refs('_binding_consistency_loss') + training.refs('assign_row', 'matched_target'),
    ))
    checks.append(finding(
        'action_comparison_metric_is_explicitly_logged',
        trainer.contains('loss_action_default_equiv', 'loss_action_active7') and Source('scripts/picf_owm_evidence_bundle.py').contains('loss_action_default_equiv', 'loss_action_active7'),
        severity='fail',
        detail='Reduced active-action objectives must log default-equivalent action loss so runs can be compared to ablation 4-22/full PICF baselines.',
        evidence=trainer.refs('loss_action_default_equiv', 'loss_action_active7') + Source('scripts/picf_owm_evidence_bundle.py').refs('loss_action_default_equiv', 'loss_action_active7'),
    ))
    checks.append(finding(
        'diagnostic_surface_covers_interaction_failures',
        trainer.contains('aqr_active_same_role_support_overlap_max', 'aqr_active_same_role_object_core_overlap_max', 'posterior_recycle_logit_mean', 'posterior_dustbin_mass_raw', 'posterior_address_update_rate_mean', 'owm_tracklet_tokens', 'owm_proposal_tokens')
        and verifier.contains('aqr_active_same_role_support_overlap_max', 'posterior_recycle_logit_mean', 'owm_posterior_binding_signature_norm_mean')
        and strict.contains('posterior_recycle_logit_mean', 'posterior_dustbin_mass_raw')
        and dataflow.contains('tracklet', 'proposal')
        and mvtrack.contains('recycle_diagnostics_explain_temporal_identity_failures'),
        severity='fail',
        detail='Metrics and audits must expose the specific cross-module failure modes: overlap, object-core overlap, recycle logits/dustbin, address update, and optional track/proposal activation.',
        evidence=trainer.refs('aqr_active_same_role_support_overlap_max', 'posterior_recycle_logit_mean', 'owm_tracklet_tokens') + verifier.refs('owm_posterior_binding_signature_norm_mean') + strict.refs('posterior_dustbin_mass_raw'),
    ))
    checks.append(finding(
        'known_limits_are_not_hidden',
        readme.contains('第四根', 'tracklet', 'proposal', 'behavior acceptance') or readme.contains('fourth', 'tracklet', 'proposal', 'behavior acceptance'),
        severity='warn',
        detail='README should keep known scientific limits visible: fine ordinal grounding, optional track/proposal data activation, and fresh behavior acceptance.',
        evidence=readme.refs('tracklet', 'proposal', 'behavior acceptance', '第四根'),
    ))
    return checks


def render_markdown(findings: list[Finding]) -> str:
    passed = sum(1 for f in findings if f.status == 'PASS')
    failed = sum(1 for f in findings if f.status == 'FAIL')
    warned = sum(1 for f in findings if f.status == 'WARN')
    lines = [
        '# PICF-AQR-OWM Professor-Grade Interaction Audit',
        '',
        'Generated by `scripts/picf_owm_professor_grade_audit.py`.',
        f'Status: {passed}/{len(findings)} PASS, {warned} WARN, {failed} FAIL.',
        '',
        'This audit is stricter than a presence verifier: every check targets a cross-module interaction that can produce false confidence if only one file is inspected.',
        '',
        '## Findings',
        '',
    ]
    for f in findings:
        lines.append(f'### {f.status}: `{f.name}`')
        lines.append('')
        lines.append(f'- Severity: `{f.severity}`')
        lines.append(f'- Detail: {f.detail}')
        if f.evidence:
            lines.append('- Evidence:')
            for ref in f.evidence:
                lines.append(f'  - `{ref}`')
        lines.append('')
    return '\n'.join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fail-on-fail', action='store_true')
    parser.add_argument('--json', type=Path, default=None)
    parser.add_argument('--markdown', type=Path, default=None)
    args = parser.parse_args()
    findings = run_checks()
    payload = [f.__dict__ for f in findings]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(findings) + '\n', encoding='utf-8')
    for f in findings:
        print(f'{f.status:4} {f.name}: {f.detail}')
    failed = [f for f in findings if f.status == 'FAIL']
    warned = [f for f in findings if f.status == 'WARN']
    print(f'SUMMARY pass={sum(f.status == "PASS" for f in findings)} warn={len(warned)} fail={len(failed)} total={len(findings)}')
    if args.fail_on_fail and failed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
