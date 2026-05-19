#!/usr/bin/env python3
"""Audit the object-routed anchor-PV repair.

The old anchor-PV target treated every projective point/visual edge as equally
binding-relevant for object slots.  That is mathematically too strong: dense
PV correspondence is a perception invariant, while object-slot routing is a
sparse belief assignment.  This audit verifies that the production loss keeps
the dense PV weak objective but gates the object-slot PV term by AQR support
with no object-loss floor.  Background evidence is not discarded; it is retained
in the dense `pv_weak` branch instead of being optimized as object identity.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _check(name: str, ok: bool) -> tuple[str, bool]:
    return name, bool(ok)


def main() -> int:
    training = _read("src/openpi/picf/core/training.py")
    trainer = _read("scripts/picf_core_train.py")
    readme = _read("src/openpi/picf/README_v2.2.md")
    doc_path = ROOT / "docs/PICF_AQR_OWM_PROPOSAL_PV_BINDING_REPAIR_20260516_TEMP.md"
    doc = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""

    checks = [
        _check(
            "loss_config_exposes_object_gate",
            "anchor_pv_object_gate_enabled" in training
            and "anchor_pv_object_gate_floor" in training
            and "anchor_pv_object_normalize_by_object_mass" in training,
        ),
        _check(
            "trainer_cli_exposes_object_gate",
            "--anchor-pv-object-gate-enabled" in trainer
            and "--anchor-pv-object-gate-floor" in trainer
            and "--anchor-pv-object-normalize-by-object-mass" in trainer,
        ),
        _check(
            "distributional_object_pv_replaces_dense_edge_bce",
            "def _object_projective_distribution_loss" in training
            and "v_hat_j = normalize(p_j C)" in training
            and "anchor_pv_object_distribution_loss" in training
            and "--anchor-pv-object-distribution-loss" in trainer,
        ),
        _check(
            "object_gate_uses_aqr_point_visual_priors",
            "point_priors" in training
            and "visual_priors" in training
            and "object_pair" in training
            and "target_weight.sum()" in training,
        ),
        _check(
            "dense_pv_weak_branch_remains",
            "pv_weak" in training
            and "point_align_embeddings" in training
            and "visual_align_embeddings" in training,
        ),
        _check(
            "blind_sam_sidecar_is_archived",
            "Blind automatic SAM is rejected" in readme
            and "scripts/picf_contact_motion_sidecar_precompute.py" in readme
            and "--mvtrack-sidecar-root" in trainer
            and "--allow-legacy-blind-sam-sidecar" in trainer
            and "proposal_centers_xy=frame.get" in trainer,
        ),
        _check(
            "proposal_diagnostic_can_restrict_to_covered_segments",
            "--calvin-segment-indices" in trainer
            and "segment_indices=calvin_segment_indices" in trainer,
        ),
        _check(
            "overlay_reports_task_and_sidecar_proposals",
            "name=\"task\"" in trainer
            and "\"proposals\": proposal_records" in trainer
            and "variant_name=\"sidecar_proposals\"" in trainer
            and "variant_name=\"mask_only\"" in trainer
            and "variant_name=\"mask_active\"" in trainer
            and "variant_name=\"mask_with_gray\"" in trainer
            and "--preview-count" in _read("scripts/picf_contact_motion_sidecar_precompute.py"),
        ),
        _check(
            "object_pull_is_role_scoped",
            "anchor_object_pull_allowed_roles" in training
            and "roles_t == int(role)" in training
            and "--anchor-object-pull-allowed-roles" in trainer
            and "anchor_object_pull_allowed_roles=_parse_int_tuple" in trainer
            and "role-0 effector" in readme,
        ),
        _check(
            "repair_doc_links_math_and_dataflow",
            "L_{anchor\\_pv}" in doc
            and "object-routed" in doc
            and "SAM" in doc
            and "IsSameObject" in doc,
        ),
    ]

    width = max(len(name) for name, _ in checks)
    failed = 0
    for name, ok in checks:
        print(f"{name:<{width}} : {'PASS' if ok else 'FAIL'}")
        failed += int(not ok)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
