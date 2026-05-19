#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditResult:
    name: str
    ok: bool
    detail: str


def _assign(
    scores: list[list[float]],
    row_specific: list[list[float]],
    *,
    candidate_quality: list[float] | None = None,
    valid: list[bool] | None = None,
    temperature: float = 0.35,
    background_prior: float = 0.25,
    background_quality_weight: float = 2.0,
    max_rows_per_candidate: int = 1,
    row_capacity: float = 1.25,
    row_capacity_iters: int = 10,
    eps: float = 1e-8,
) -> tuple[list[list[float]], list[float], list[float], list[list[float]]]:
    """Reference math for PICF object-candidate slot/background assignment."""

    if valid is None:
        valid = [True for _ in scores[0]]
    if candidate_quality is None:
        candidate_quality = [0.0 for _ in scores[0]]
    row_count = len(scores)
    col_count = len(scores[0])
    masked_scores = [
        [
            max(float(scores[j][p]), 0.0) if row_specific[j][p] > 0.0 and valid[p] else -1.0e4
            for p in range(col_count)
        ]
        for j in range(row_count)
    ]
    if int(max_rows_per_candidate) > 0 and int(max_rows_per_candidate) < row_count:
        k = int(max_rows_per_candidate)
        for p in range(col_count):
            ranked = sorted(range(row_count), key=lambda j: masked_scores[j][p], reverse=True)
            keep = set(j for j in ranked[:k] if masked_scores[j][p] > -9999.0)
            for j in range(row_count):
                if j not in keep:
                    masked_scores[j][p] = -1.0e4
    scaled = [[value / max(float(temperature), eps) for value in row] for row in masked_scores]
    bg_logits = [
        math.log(float(background_prior)) - max(float(background_quality_weight), 0.0) * min(max(float(q), 0.0), 1.0)
        for q in candidate_quality
    ]
    max_col = [max(max(scaled[j][p] for j in range(row_count)), bg_logits[p]) for p in range(col_count)]
    slot_exp = [
        [math.exp(scaled[j][p] - max_col[p]) if valid[p] else 0.0 for p in range(col_count)]
        for j in range(row_count)
    ]
    bg_exp = [math.exp(bg_logits[p] - max_col[p]) if valid[p] else 0.0 for p in range(col_count)]
    if float(row_capacity) > 0.0 and int(row_capacity_iters) > 0:
        cap = float(row_capacity)
        for _ in range(int(row_capacity_iters)):
            denom_i = [max(sum(slot_exp[j][p] for j in range(row_count)) + bg_exp[p], eps) for p in range(col_count)]
            assignment_i = [[slot_exp[j][p] / denom_i[p] for p in range(col_count)] for j in range(row_count)]
            for j in range(row_count):
                row_mass = sum(assignment_i[j])
                if row_mass > cap:
                    scale = cap / max(row_mass, eps)
                    for p in range(col_count):
                        slot_exp[j][p] *= scale
    denom = [max(sum(slot_exp[j][p] for j in range(row_count)) + bg_exp[p], eps) for p in range(col_count)]
    assignment = [[slot_exp[j][p] / denom[p] for p in range(col_count)] for j in range(row_count)]
    background = [bg_exp[p] / denom[p] for p in range(col_count)]
    coverage = [sum(assignment[j][p] for j in range(row_count)) for p in range(col_count)]
    norm = [math.sqrt(max(sum(v * v for v in assignment[j]), eps)) for j in range(row_count)]
    duplicate = []
    for j in range(row_count):
        row = []
        for k in range(row_count):
            if j == k:
                row.append(0.0)
            else:
                row.append(sum(assignment[j][p] * assignment[k][p] for p in range(col_count)) / (norm[j] * norm[k]))
        duplicate.append(row)
    return assignment, coverage, background, duplicate


def _active_object_weight(
    active: list[float] | None,
    downstream_weight: list[float] | None,
    row_count: int,
) -> list[float]:
    if active is not None and len(active) >= row_count:
        return [min(max(float(value), 0.0), 1.0) for value in active[:row_count]]
    if downstream_weight is not None and len(downstream_weight) >= row_count:
        return [1.0 if min(max(float(value), 0.0), 1.0) > 0.5 else 0.0 for value in downstream_weight[:row_count]]
    return [1.0 for _ in range(row_count)]


def _active_pair_duplicate_max(duplicate: list[list[float]], active: list[float]) -> float:
    best = 0.0
    for i in range(len(duplicate)):
        for j in range(i + 1, len(duplicate)):
            if active[i] > 0.5 and active[j] > 0.5:
                best = max(best, float(duplicate[i][j]))
    return best


def _denoise_active_rows(peaks: list[float], active: list[float], *, uniform: float = 0.10) -> int:
    return sum(1 for peak, row_active in zip(peaks, active) if row_active > 0.5 and peak > min(uniform + 0.05, 0.95))


def _denoise_confirmed_rows(
    peaks: list[float],
    active: list[float],
    confirmation: list[float],
    *,
    threshold: float = 0.05,
    uniform: float = 0.10,
) -> int:
    return sum(
        1
        for peak, row_active, row_confirmed in zip(peaks, active, confirmation)
        if row_active > 0.5 and row_confirmed > threshold and peak > min(uniform + 0.05, 0.95)
    )


def _weighted_bce(routing: list[list[float]], target: list[list[float]], weight: list[list[float]]) -> float:
    total = 0.0
    mass = 0.0
    for i, row in enumerate(routing):
        for j, pred in enumerate(row):
            y = min(max(float(target[i][j]), 1e-6), 1.0 - 1e-6)
            p = min(max(float(pred), 1e-6), 1.0 - 1e-6)
            w = max(float(weight[i][j]), 0.0)
            total += w * (-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))
            mass += w
    return total / max(mass, 1e-8)


def _js(p: list[float], q: list[float], eps: float = 1e-8) -> float:
    p_sum = max(sum(max(v, 0.0) for v in p), eps)
    q_sum = max(sum(max(v, 0.0) for v in q), eps)
    pp = [max(v, 0.0) / p_sum for v in p]
    qq = [max(v, 0.0) / q_sum for v in q]
    m = [(a + b) * 0.5 for a, b in zip(pp, qq, strict=True)]
    kl_p = sum(a * (math.log(max(a, eps)) - math.log(max(mm, eps))) for a, mm in zip(pp, m, strict=True))
    kl_q = sum(b * (math.log(max(b, eps)) - math.log(max(mm, eps))) for b, mm in zip(qq, m, strict=True))
    return 0.5 * (kl_p + kl_q)


def _distributional_object_pv_loss(
    point: list[list[float]],
    visual: list[list[float]],
    compat: list[list[float]],
    row_weight: list[float],
) -> float:
    total = 0.0
    mass = 0.0
    for j, weight in enumerate(row_weight):
        if weight <= 0.0:
            continue
        v_hat = [sum(point[j][p] * compat[p][u] for p in range(len(compat))) for u in range(len(compat[0]))]
        p_hat = [sum(visual[j][u] * compat[p][u] for u in range(len(compat[0]))) for p in range(len(compat))]
        total += weight * 0.5 * (_js(visual[j], v_hat) + _js(point[j], p_hat))
        mass += weight
    return total / max(mass, 1e-8)


def _all(values: list[bool]) -> bool:
    return all(bool(value) for value in values)


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def _row_normalize(a: list[list[float]], eps: float = 1e-8) -> list[list[float]]:
    out = []
    for row in a:
        total = max(sum(row), eps)
        out.append([value / total for value in row])
    return out


def _owner_transport(
    *,
    pre_topk_scores: list[list[float]],
    roles: list[int],
    query_types: list[int],
    coverage: list[float],
    assignment: list[list[float]],
    proposal_to_point: list[list[float]],
    owner_roles: tuple[int, ...] = (1,),
    min_share: float = 0.65,
) -> tuple[list[list[float]], list[list[float]]]:
    """Reference owner-geometry transport used by the runtime audit.

    Object/contact dual-role assignment is allowed to let a contact bridge
    explain the same inspected candidate.  The missing leg caught by the
    runtime probe was different: the role-1 object file could fail to receive
    stable geometry.  Owner transport selects a bounded role-1 owner per
    covered candidate and transports the candidate mask into that owner's point
    prior without making the contact bridge the object identity.
    """

    row_count = len(pre_topk_scores)
    candidate_count = len(pre_topk_scores[0]) if row_count else 0
    point_count = len(proposal_to_point[0]) if proposal_to_point else 0
    owner_assignment = [[0.0 for _ in range(candidate_count)] for _ in range(row_count)]
    owner_rows = [
        row
        for row, (role, query_type) in enumerate(zip(roles, query_types, strict=True))
        if int(query_type) == 0 and int(role) in owner_roles and int(role) != 0
    ]
    for p in range(candidate_count):
        if coverage[p] <= 1e-8 or not owner_rows:
            continue
        best = max(owner_rows, key=lambda row: pre_topk_scores[row][p])
        if pre_topk_scores[best][p] <= -9999.0:
            continue
        owner_assignment[best][p] = max(float(assignment[best][p]), float(coverage[p]) * float(min_share))
    owner_point = _row_normalize(_matmul(owner_assignment, proposal_to_point))
    return owner_assignment, owner_point


def _column_max_normalize(a: list[list[float]], floor: float = 1e-4, eps: float = 1e-8) -> list[list[float]]:
    if not a:
        return []
    col_count = len(a[0])
    out = [[0.0 for _ in range(col_count)] for _ in range(len(a))]
    for p in range(col_count):
        values = [max(float(a[j][p]), 0.0) if float(a[j][p]) >= floor else 0.0 for j in range(len(a))]
        denom = max(max(values), eps)
        for j, value in enumerate(values):
            out[j][p] = value / denom if denom > eps else 0.0
    return out


def _combine_positive_sources(
    *,
    proposal_prior: list[list[float]] | None = None,
    point_overlap: list[list[float]] | None = None,
    seed: list[list[float]] | None = None,
    owner: list[float] | None = None,
    proposal_weight: float = 0.75,
    point_weight: float = 1.0,
    seed_weight: float = 1.25,
    owner_weight: float = 0.5,
    floor: float = 0.01,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    source = proposal_prior or point_overlap or seed
    if source is None:
        raise ValueError("at least one row-specific source is required")
    row_count = len(source)
    col_count = len(source[0])
    scores = [[0.0 for _ in range(col_count)] for _ in range(row_count)]
    row_specific = [[0.0 for _ in range(col_count)] for _ in range(row_count)]
    quality = [0.0 for _ in range(col_count)]

    def add(src: list[list[float]] | None, weight: float) -> None:
        if src is None or weight == 0.0:
            return
        normalized = _column_max_normalize(src, floor=floor)
        for j in range(row_count):
            for p in range(col_count):
                if src[j][p] >= floor:
                    row_specific[j][p] = 1.0
                scores[j][p] += weight * normalized[j][p]
        for p in range(col_count):
            q = max(max(float(src[j][p]), 0.0) for j in range(row_count))
            quality[p] = max(quality[p], q)

    add(proposal_prior, proposal_weight)
    add(point_overlap, point_weight)
    add(seed, seed_weight)
    if owner is not None:
        owner_max = max(max(owner), 1e-8)
        owner_norm = [max(float(value), 0.0) / owner_max for value in owner]
        for j in range(row_count):
            for p in range(col_count):
                scores[j][p] += owner_weight * owner_norm[p]
        quality = [max(quality[p], owner_norm[p]) for p in range(col_count)]
    return scores, row_specific, [min(max(value, 0.0), 1.0) for value in quality]


def run_audit() -> list[AuditResult]:
    results: list[AuditResult] = []
    repo_root = Path(__file__).resolve().parents[1]
    config_text = (repo_root / "src/openpi/picf/core/config.py").read_text(encoding="utf-8")
    pipeline_text = (repo_root / "src/openpi/picf/core/pipeline.py").read_text(encoding="utf-8")
    train_text = (repo_root / "scripts/picf_core_train.py").read_text(encoding="utf-8")
    owner_probe_script = repo_root / "run_a7_object_owner_only_pull_probe_1000_20260519.sh"
    owner_probe_text = owner_probe_script.read_text(encoding="utf-8") if owner_probe_script.exists() else ""

    results.append(
        AuditResult(
            "config_exposes_dual_role_object_candidate_rows",
            "object_candidate_eligible_roles" in config_text and "(1, 2)" in config_text,
            "Object candidates should be explainable by role 1 object owners and role 2 contact bridges by default.",
        )
    )
    results.append(
        AuditResult(
            "pipeline_uses_object_candidate_rows_not_effector_rows",
            "_object_candidate_physical_rows" in pipeline_text
            and "if int(role) == 0:\n                continue" in pipeline_text
            and pipeline_text.count("self._object_candidate_physical_rows") >= 5,
            "Proposal/point/visual object-candidate routing must use role-filtered object/contact rows and explicitly exclude role 0.",
        )
    )
    results.append(
        AuditResult(
            "train_cli_exposes_object_candidate_roles",
            "--object-candidate-eligible-roles" in train_text and "object_candidate_eligible_roles=_parse_int_tuple" in train_text,
            "Training launchers must be able to configure object/contact candidate roles rather than baking in one role.",
        )
    )
    results.append(
        AuditResult(
            "config_exposes_role1_owner_transport",
            "object_candidate_owner_transport_enabled" in config_text
            and "object_candidate_owner_roles: tuple[int, ...] = (1,)" in config_text
            and "object_candidate_owner_min_share" in config_text
            and "object_candidate_owner_point_mix" in config_text,
            "Confirmed object/contact candidates need an explicit role-1 owner geometry transport leg, not only dual-role support assignment.",
        )
    )
    results.append(
        AuditResult(
            "pipeline_returns_owner_assignment_and_point_priors",
            "object_candidate_owner_assignment" in pipeline_text
            and "object_candidate_owner_point_priors" in pipeline_text
            and "pre_topk_slot_logits" in pipeline_text
            and "object_candidate_owner_min_share" in pipeline_text,
            "The runtime graph must expose owner assignment and transported point priors for diagnostics and object-pull training.",
        )
    )
    results.append(
        AuditResult(
            "train_cli_exposes_owner_transport_flags",
            "--object-candidate-owner-transport-enabled" in train_text
            and "--object-candidate-owner-roles" in train_text
            and "object_candidate_owner_min_share" in train_text,
            "Remote launches must be able to enable and inspect owner transport explicitly.",
        )
    )
    results.append(
        AuditResult(
            "config_exposes_object_owner_tactile_and_role_layout",
            "tactile_attach_to_object_owner: bool = True" in config_text
            and 'aqr_role_layout: str = "structured"' in config_text,
            "Contact/tactile evidence must be configurable as object-owner evidence, and graph role layout must be launch-selectable.",
        )
    )
    results.append(
        AuditResult(
            "pipeline_object_only_layout_removes_effector_role0",
            'if layout in {"object_only", "object"}:' in pipeline_text
            and "return torch.ones((count,), device=self.device, dtype=torch.long)" in pipeline_text,
            "The object-owner probe needs a real role-1-only graph layout, not only hidden blue/effector overlays.",
        )
    )
    results.append(
        AuditResult(
            "pipeline_tactile_can_attach_to_object_owner",
            "attach_to_object = bool(getattr(self.config, \"tactile_attach_to_object_owner\", True))" in pipeline_text
            and "if attach_to_object:" in pipeline_text
            and "if role_int != 1:" in pipeline_text
            and "elif role_int == 1:" in pipeline_text,
            "When enabled, tactile/contact tokens must be readable by role-1 object owners and blocked from non-object rows.",
        )
    )
    results.append(
        AuditResult(
            "aqr_graph_tactile_seed_and_reader_follow_object_owner_flag",
            "active_roles = (roles == 1) if attach_to_object else ((roles == 0) | (roles == 2))" in pipeline_text
            and "tactile_roles = (roles == 1) if attach_to_object else ((roles == 0) | (roles == 2))" in pipeline_text,
            "AQR tactile seed priors and tactile reader bias must follow tactile_attach_to_object_owner, not only the public/fused read path.",
        )
    )
    results.append(
        AuditResult(
            "train_cli_exposes_object_owner_tactile_and_role_layout",
            "--tactile-attach-to-object-owner" in train_text
            and "--aqr-role-layout" in train_text
            and "aqr_role_layout=str(_arg_or_default" in train_text,
            "Remote launchers must expose the object-owner tactile routing and role-layout switches.",
        )
    )
    results.append(
        AuditResult(
            "object_pull_closes_graph_to_posterior_belief",
            "anchor_object_pull_posterior_weight" in train_text
            and "anchor_object_pull_posterior_weight" in (repo_root / "src/openpi/picf/core/training.py").read_text(encoding="utf-8")
            and "posterior.x.to(device=reference.device" in (repo_root / "src/openpi/picf/core/training.py").read_text(encoding="utf-8"),
            "The diagnostic object-pull objective must test belief/posterior ownership, not only graph-anchor geometry.",
        )
    )
    results.append(
        AuditResult(
            "owner_only_probe_disables_effector_competition",
            "--aqr-role-layout object_only" in owner_probe_text
            and "--effector-persistent-anchors 0" in owner_probe_text
            and "--effector-observation-anchors 0" in owner_probe_text
            and "--task-effector-queries 0" in owner_probe_text
            and "--tactile-attach-to-object-owner" in owner_probe_text
            and "--object-candidate-eligible-roles 1" in owner_probe_text
            and "--anchor-object-pull-allowed-roles 1" in owner_probe_text,
            "The clean object-pull probe must remove blue/effector role competition and test only role-1 object ownership.",
        )
    )
    results.append(
        AuditResult(
            "owner_only_probe_has_single_owner_posterior_file",
            "--posterior-file-competition-max-per-role 1" in owner_probe_text
            and "--anchor-object-pull-posterior-weight 1.0" in owner_probe_text,
            "A single sidecar object owner should be explained by one active posterior file in the clean pull probe.",
        )
    )

    logits = [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]
    row_specific = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assignment, coverage, background, _ = _assign(logits, row_specific)
    results.append(
        AuditResult(
            "task_owner_only_cannot_clone_to_all_slots",
            _all([v < 1e-6 for row in assignment for v in row])
            and _all([v < 1e-6 for v in coverage])
            and _all([v > 0.999 for v in background]),
            "Without row-specific support, task-level proposal scores must go to background rather than symmetric slot rows.",
        )
    )

    logits = [[5.0, 0.0], [0.0, 5.0], [0.0, 0.0]]
    row_specific = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    assignment, coverage, background, duplicate = _assign(logits, row_specific, candidate_quality=[1.0, 1.0])
    col_mass = [coverage[p] + background[p] for p in range(len(coverage))]
    results.append(
        AuditResult(
            "candidate_columns_conserve_slot_background_mass",
            _all([abs(v - 1.0) < 1e-5 for v in col_mass]),
            "Every candidate must be explained by slots plus explicit background residual.",
        )
    )
    results.append(
        AuditResult(
            "row_specific_support_assigns_distinct_candidates",
            bool(assignment[0][0] > 0.95 and assignment[1][1] > 0.95 and max(v for row in duplicate for v in row) < 0.05),
            "Distinct row-specific evidence should assign distinct candidates to distinct slots.",
        )
    )

    logits = [[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]]
    row_specific = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assignment, _, _, duplicate = _assign(
        logits,
        row_specific,
        candidate_quality=[1.0, 1.0],
        max_rows_per_candidate=0,
        row_capacity_iters=0,
    )
    results.append(
        AuditResult(
            "duplicate_candidate_explanations_are_visible",
            bool(duplicate[0][1] > 0.95),
            "Two rows explaining the same candidate must surface as high duplicate overlap.",
        )
    )
    assignment, coverage, background, duplicate = _assign(logits, row_specific, candidate_quality=[1.0, 1.0])
    results.append(
        AuditResult(
            "candidate_top1_suppresses_raw_same_candidate_clones",
            bool(assignment[0][0] > 0.95 and assignment[1][0] < 1e-6 and max(v for row in duplicate for v in row) < 0.05),
            "Default candidate top-1 competition should prevent two raw rows from cloning the same proposal candidate.",
        )
    )
    assignment, coverage, background, _ = _assign(
        scores=[[5.0], [4.8], [0.0]],
        row_specific=[[1.0], [1.0], [0.0]],
        candidate_quality=[1.0],
        max_rows_per_candidate=2,
        row_capacity=1.25,
        row_capacity_iters=10,
    )
    results.append(
        AuditResult(
            "candidate_top2_allows_object_and_contact_bridge",
            bool(assignment[0][0] > 0.45 and assignment[1][0] > 0.25 and assignment[2][0] < 1e-6 and background[0] < 0.05),
            "A single inspected object candidate may be explained by one object owner plus one contact bridge, but not by arbitrary extra rows.",
        )
    )
    owner_assignment, owner_point = _owner_transport(
        pre_topk_scores=[[4.6], [5.0], [0.0]],
        roles=[1, 2, 1],
        query_types=[0, 0, 1],
        coverage=coverage,
        assignment=assignment,
        proposal_to_point=[[0.15, 0.85, 0.0]],
        owner_roles=(1,),
        min_share=0.65,
    )
    results.append(
        AuditResult(
            "owner_transport_gives_role1_geometry_even_when_contact_bridge_scores_higher",
            bool(
                owner_assignment[0][0] >= 0.60
                and owner_assignment[1][0] < 1e-8
                and owner_assignment[2][0] < 1e-8
                and sum(owner_point[0][1:2]) > 0.80
            ),
            "Role-2 contact may share candidate support, but role-1 object owners must receive bounded transported mask geometry.",
        )
    )
    results.append(
        AuditResult(
            "posterior_update_closes_owner_responsibility_to_file_geometry",
            "def _posterior_owner_transport_measurement" in pipeline_text
            and "obs_owner_weight = obs_graph * row_strength[None, :]" in pipeline_text
            and "post_owner_weight = torch.clamp(binding_support" in pipeline_text
            and "posterior_owner_transport_enabled" in config_text
            and "--posterior-owner-transport-enabled" in train_text
            and "--posterior-owner-transport-roles" in train_text
            and "--posterior-owner-transport-activates-file" in train_text
            and "posterior_owner_transport_inactive_prior" in config_text
            and "owner_file_gate = torch.clamp(torch.maximum(owner_file_gate, owner_activation)" in pipeline_text
            and "owner_candidate = owner_conf_for_gate >= active_threshold" in pipeline_text
            and "100.0 * owner_conf_for_gate" in pipeline_text
            and "def _cap_file_gate_by_role" in pipeline_text
            and "posterior_owner_transport_precision_gain" in config_text
            and "--posterior-owner-transport-precision-gain" in train_text
            and "standard_precision = torch.linalg.pinv(S + jitter)" in pipeline_text
            and "owner_precision = torch.linalg.pinv(owner_S_aligned + jitter)" in pipeline_text
            and "fused_x = torch.matmul(fused_S, fused_eta[:, :, None]).squeeze(-1)" in pipeline_text
            and "owner_transport_confidence=owner_transport_confidence" in pipeline_text
            and "posterior_owner_transport_confidence_mean" in train_text,
            "Accepted graph object responsibility must be transported through observation/posterior assignment and fused as a high-precision posterior measurement.",
        )
    )
    crowded = [[5.0, 5.0, 5.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    crowded_support = [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assignment, _, background, _ = _assign(
        crowded,
        crowded_support,
        candidate_quality=[1.0, 1.0, 1.0],
        max_rows_per_candidate=2,
        row_capacity=1.25,
        row_capacity_iters=10,
    )
    results.append(
        AuditResult(
            "row_capacity_limits_one_slot_from_eating_all_candidates",
            bool(sum(assignment[0]) <= 1.26 and max(background) < 0.95),
            "Soft row capacity should stop one high-scoring row from absorbing every proposal candidate while preserving non-background alternatives.",
        )
    )

    active = _active_object_weight(active=[1.0, 0.0, 1.0], downstream_weight=None, row_count=3)
    duplicate = [
        [0.0, 0.99, 0.05],
        [0.99, 0.0, 0.98],
        [0.05, 0.98, 0.0],
    ]
    results.append(
        AuditResult(
            "active_object_scope_ignores_reserve_duplicates",
            bool(_active_pair_duplicate_max(duplicate, active) < 0.10),
            "Reserve/no-object rows may duplicate context evidence, but object-level duplicate penalties must only see active object rows.",
        )
    )
    active_from_downstream = _active_object_weight(active=None, downstream_weight=[1.0, 0.15, 0.0, 0.8], row_count=4)
    results.append(
        AuditResult(
            "downstream_weight_fallback_excludes_context_rows",
            active_from_downstream == [1.0, 0.0, 0.0, 1.0],
            "When only downstream weights are present, low-weight context rows are not optimized as active object files.",
        )
    )
    results.append(
        AuditResult(
            "denoising_active_object_scope_excludes_no_object_peaks",
            _denoise_active_rows([0.9, 0.92, 0.88], [1.0, 0.0, 0.0]) == 1,
            "AQR denoising should not train reserve/no-object rows even when their support priors have sharp peaks.",
        )
    )
    results.append(
        AuditResult(
            "denoising_confirmed_object_scope_excludes_unconfirmed_active_rows",
            _denoise_confirmed_rows([0.9, 0.92, 0.88], [1.0, 1.0, 0.0], [0.8, 0.0, 0.0]) == 1,
            "AQR denoising should require confirmed object-candidate/proposal/point evidence, not only an active row flag.",
        )
    )
    dense_weight = [[0.25, 0.25], [0.25, 0.25]]
    object_weight = [[0.0, 1.0], [0.0, 0.0]]
    dense_loss = _weighted_bce([[0.9, 0.9], [0.9, 0.9]], [[0.0, 1.0], [0.0, 0.0]], dense_weight)
    object_loss = _weighted_bce([[0.9, 0.9], [0.9, 0.9]], [[0.0, 1.0], [0.0, 0.0]], object_weight)
    results.append(
        AuditResult(
            "object_pv_normalizes_by_confirmed_object_mass_not_dense_floor",
            bool(object_loss < dense_loss and abs(object_loss - (-math.log(0.9))) < 1e-5),
            "Object anchor-PV should average over confirmed object edges; dense/background coverage belongs to pv_weak.",
        )
    )

    compat = [[1.0, 0.0], [0.0, 1.0]]
    matched = _distributional_object_pv_loss(
        point=[[1.0, 0.0], [0.0, 1.0]],
        visual=[[1.0, 0.0], [0.0, 1.0]],
        compat=compat,
        row_weight=[1.0, 1.0],
    )
    mismatched = _distributional_object_pv_loss(
        point=[[1.0, 0.0], [0.0, 1.0]],
        visual=[[0.0, 1.0], [1.0, 0.0]],
        compat=compat,
        row_weight=[1.0, 1.0],
    )
    results.append(
        AuditResult(
            "distributional_object_pv_penalizes_slot_projective_mismatch",
            bool(matched < 1e-6 and mismatched > 0.2),
            "Anchor-PV should compare each object's point support to the same object's visual support transported through projective geometry.",
        )
    )

    proposal_to_point = [[0.8, 0.2, 0.0, 0.0], [0.0, 0.0, 0.3, 0.7]]
    assignment = [[1.0, 0.0], [0.0, 1.0]]
    point_prior = _row_normalize(_matmul(assignment, proposal_to_point))
    results.append(
        AuditResult(
            "candidate_mask_transport_preserves_object_support",
            bool(sum(point_prior[0][:2]) > 0.99 and sum(point_prior[1][2:]) > 0.99),
            "Assigned candidates must transport their mask support into the corresponding point priors.",
        )
    )

    proposal_prior = [[0.85], [0.01], [0.0]]
    point_overlap = [[0.05], [0.01], [0.0]]
    scores, row_specific, quality = _combine_positive_sources(
        proposal_prior=proposal_prior,
        point_overlap=point_overlap,
        owner=[1.0],
    )
    assignment, coverage, background, _ = _assign(scores, row_specific, candidate_quality=quality)
    results.append(
        AuditResult(
            "runtime_scale_candidate_not_absorbed_by_background",
            bool(coverage[0] > 0.5 and assignment[0][0] > assignment[1][0] and background[0] < 0.5),
            "A high-quality task proposal with weak but row-specific point overlap must remain explainable by a slot; weak overlap is not negative evidence.",
        )
    )

    scores, row_specific, quality = _combine_positive_sources(proposal_prior=None, point_overlap=None, seed=[[0.0], [0.0]], owner=[1.0])
    assignment, coverage, background, _ = _assign(scores, row_specific, candidate_quality=quality)
    results.append(
        AuditResult(
            "task_quality_still_needs_row_specific_support",
            bool(coverage[0] < 1e-6 and background[0] > 0.999),
            "Candidate quality can lower background only after a row-specific support source exists; task score alone must not clone into all rows.",
        )
    )

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the PICF object-candidate slot/background assignment math.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    results = run_audit()
    failed = [result for result in results if not result.ok]
    if args.json:
        print(json.dumps({"ok": not failed, "results": [result.__dict__ for result in results]}, indent=2, sort_keys=True))
    else:
        for result in results:
            status = "PASS" if result.ok else "FAIL"
            print(f"{status} {result.name}: {result.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
