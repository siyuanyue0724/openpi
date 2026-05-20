#!/usr/bin/env python3
"""Audit the PICF latest-slot deployment closure.

This script is intentionally static and conservative.  It checks whether the
current codebase contains the belief-state-compatible invariants taken from
recent slot/object-binding/visuo-tactile work, and whether rejected mechanisms
are documented as rejected rather than silently treated as production paths.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _all(text: str, needles: tuple[str, ...]) -> bool:
    return all(needle in text for needle in needles)


def run(repo_root: Path) -> list[Check]:
    src = repo_root / "src/openpi/picf"
    core = src / "core"
    pipeline = _read(core / "pipeline.py")
    config = _read(core / "config.py")
    contracts = _read(core / "contracts.py")
    observation_contracts = _read(src / "contracts.py")
    training = _read(core / "training.py")
    trainer = _read(repo_root / "scripts/picf_core_train.py")
    run_audit = _read(repo_root / "scripts/picf_run_contract_audit.py")
    object_audit = _read(repo_root / "scripts/picf_object_candidate_slot_binding_audit.py")
    readme = _read(src / "README_v2.2.md")
    issue_tracker = _read(repo_root / "docs/PICF_AQR_OWM_OPEN_ISSUE_TRACKER_20260517_TEMP.md")
    closure_doc = repo_root / "temp/audits_20260519/latest_slot_full_deployment_closure_20260519.md"
    closure_text = _read(closure_doc) if closure_doc.exists() else ""
    final_audit_doc = repo_root / "docs/PICF_AQR_OWM_LATEST_SLOT_FINAL_AUDIT_20260520_TEMP.md"
    final_audit_text = _read(final_audit_doc) if final_audit_doc.exists() else ""
    direct_owner_doc = repo_root / "temp/audits_20260520/posterior_owner_transport_direct_write_through_followthrough.md"
    direct_owner_text = _read(direct_owner_doc) if direct_owner_doc.exists() else ""
    comprehensive_script = repo_root / "scripts/experiments/picf_aqr_owm_202605_active/run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh"
    comprehensive = _read(comprehensive_script) if comprehensive_script.exists() else ""
    actionaware_script = repo_root / "scripts/experiments/picf_aqr_owm_202605_active/run_a7_actionaware_after_dedup_smoke300_20260520.sh"
    actionaware = _read(actionaware_script) if actionaware_script.exists() else ""

    checks: list[Check] = []

    checks.append(
        Check(
            "paper_code_snapshots_present_for_nontrivial_comparison",
            all(
                (repo_root / path).exists()
                for path in (
                    "temp/paper_code_20260518/MetaSlot",
                    "temp/paper_code_20260518/AdaSlot",
                    "temp/paper_code_20260518/object-centric-learning-framework",
                    "temp/paper_code_20260518/slot-attention-video",
                    "temp/external_repos/SlotLifter",
                )
            ),
            "Local paper-code snapshots must exist so the audit is not based on memory-only summaries.",
        )
    )
    checks.append(
        Check(
            "slot_attention_competition_invariant_is_picf_native",
            _all(
                pipeline,
                (
                    "aqr_same_role_support_competition_enabled",
                    "_proposal_object_candidate_assignment",
                    "object_candidate_background",
                    "row_capacity",
                    "_normalize_rows",
                ),
            ),
            "PICF must implement slot-axis competition, explicit background residual, and row-capacity allocation.",
        )
    )
    checks.append(
        Check(
            "qasa_style_slot_quality_is_first_class_and_guarded",
            _all(
                contracts,
                ("class PicfSlotQualityState", "object_quality", "no_object_prob", "duplicate_prob"),
            )
            and _all(
                pipeline,
                ("def _build_slot_quality_state", "aqr_slot_quality_head", "slot_quality.active_weight", "slot_quality.context_weight"),
            )
            and _all(training, ("def _slot_quality_loss", "lambda_slot_quality")),
            "Adaptive object/no-object/duplicate slot quality must be explicit state and optional guarded loss.",
        )
    )
    checks.append(
        Check(
            "metaslot_duplicate_and_dynamic_count_principle_without_hard_vq_truth",
            _all(
                pipeline,
                (
                    "_slot_duplicate_risk",
                    "file_competition_active",
                    "posterior_file_competition",
                    "posterior_birth_competition",
                    "active_weight",
                    "context_weight",
                ),
            )
            and "global VQ prototype codebook" in closure_text
            and ("Not copied" in closure_text or "not copied" in closure_text),
            "MetaSlot's dynamic-count/dedup invariant should be present, while hard VQ posterior truth is rejected.",
        )
    )
    checks.append(
        Check(
            "object_binding_pairwise_subspace_is_runtime_signal_not_placeholder",
            _all(
                pipeline,
                (
                    "binding_signature_proj",
                    "binding_quadratic_diag",
                    "binding_low_rank_left",
                    "def _binding_signature_quadratic_scores",
                    "def _calibrate_pairwise_binding_score",
                    "binding_signature_calibrated_top1_margin_mean",
                ),
            )
            and _all(
                contracts,
                (
                    "binding_signature_quadratic_score_mean",
                    "binding_signature_low_rank_score_mean",
                    "binding_signature_calibrated_score_std",
                ),
            ),
            "Object Binding paper's pairwise/quadratic subspace must be used inside binding diagnostics/runtime.",
        )
    )
    checks.append(
        Check(
            "posterior_binding_signature_is_persistent_gated_memory",
            _all(
                config,
                (
                    "posterior_binding_signature_memory_enabled",
                    "posterior_binding_signature_dispersion_gate_enabled",
                    "posterior_binding_signature_measurement_margin_min",
                ),
            )
            and _all(
                pipeline,
                (
                    "previous.posterior.binding_signature",
                    "binding_signature_measurement_trust",
                    "binding_signature_measurement_dispersion_gate",
                    "binding_signature_update_rate",
                ),
            ),
            "Binding signature must not be blind per-frame overwrite; it must be gated posterior file state.",
        )
    )
    checks.append(
        Check(
            "slotvla_style_object_evidence_is_sidecar_measurement_not_box_truth",
            _all(
                observation_contracts,
                ("proposal_centers_xy", "proposal_boxes_xyxy", "tracklet_xy", "tracklet_confidence"),
            )
            and _all(
                trainer,
                ("_read_mvtrack_sidecar_fields", "tracklet_xy", "proposal_centers_xy", "load_tracklet_fields"),
            )
            and _all(
                pipeline,
                (
                    "proposal_priors",
                    "tracklet_priors",
                    "object_candidate_owner_assignment",
                    "object_candidate_owner_point_priors",
                    "object_candidate_owner_x",
                    "object_candidate_owner_geometry_mix",
                    "aqr_context_slot_active_support_overlap_enabled",
                    "max_support_to_active",
                ),
            ),
            "SlotVLA-style object evidence must enter as optional typed sidecar measurements under posterior authority.",
        )
    )
    checks.append(
        Check(
            "posterior_owner_transport_preserves_candidate_identity_until_file_write",
            _all(
                config,
                (
                    "posterior_owner_transport_direct_candidate_assignment",
                    "posterior_owner_transport_direct_candidate_min_score",
                ),
            )
            and _all(
                pipeline,
                (
                    "slot_graph_weight",
                    "direct_candidate_assignment",
                    "direct_role_has_assignment",
                    "owner_transport_dist_after_fusion",
                ),
            )
            and _all(
                trainer,
                (
                    "--posterior-owner-transport-direct-candidate-assignment",
                    "posterior_owner_transport_active_dist_after_fusion_mean",
                    "owner_transport_dist_after_fusion",
                ),
            )
            and direct_owner_doc.exists()
            and _all(
                direct_owner_text,
                (
                    "candidate \\rightarrow graph\\ owner \\rightarrow posterior\\ file",
                    "direct candidate/file assignment",
                    "posterior_owner_transport_dist_after_fusion",
                    "old obs-averaged owner transport remains only as fallback",
                ),
            ),
            "Accepted graph-owner candidate identity must survive until posterior-file write-through; pre-fusion distance is not the closure metric.",
        )
    )
    checks.append(
        Check(
            "tactile_visuotactile_binding_routes_to_object_owner_not_gripper_owner",
            _all(config, ("tactile_attach_to_object_owner: bool = True", "posterior_owner_transport_roles: tuple[int, ...] = (1,)"))
            and _all(
                pipeline,
                (
                    "attach_to_object = bool(getattr(self.config, \"tactile_attach_to_object_owner\", True))",
                    "if role_int != 1:",
                    "posterior_owner_transport_enabled",
                    "posterior_owner_transport_precision_gain",
                ),
            ),
            "Tactile/contact evidence must attach to the task object owner, with role-0 effector prevented from owning the object.",
        )
    )
    checks.append(
        Check(
            "object_only_probe_and_comprehensive_validation_are_separated",
            _all(
                run_audit,
                (
                    "anchor_capability_probe",
                    "slot_comprehensive_frozen_policy_validation",
                    "formal_frozen_pretrain_cotrain",
                ),
            )
            and _all(
                comprehensive,
                (
                    "--picf-trainable-scope all",
                    "--perception-finetune-mode frozen",
                    "TRAINING_STRATEGY=\"${TRAINING_STRATEGY:-ddp}\"",
                    "--training-strategy \"${TRAINING_STRATEGY}\"",
                    "ACTION_LOSS_WEIGHT=\"${ACTION_LOSS_WEIGHT:-0.0}\"",
                    "--lambda-action-pos \"${ACTION_POS_WEIGHT}\"",
                    "SEMANTIC_TRAINABLE=\"${SEMANTIC_TRAINABLE:-0}\"",
                    "--lambda-slot-jepa 0.0",
                    "--lambda-anchor-object-pull 0.35",
                ),
            )
            and _all(
                actionaware,
                (
                    "SEMANTIC_TRAINABLE=\"${SEMANTIC_TRAINABLE:-1}\"",
                    "TRAINING_STRATEGY=\"${TRAINING_STRATEGY:-fsdp_full_shard}\"",
                    "PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\"",
                    "ACTION_LOSS_WEIGHT=\"${ACTION_LOSS_WEIGHT:-0.50}\"",
                    "run_a7_slot_comprehensive_frozen_policy_1000_20260519.sh",
                ),
            ),
            "Anchor-only probes must not be conflated with comprehensive frozen-policy validation.",
        )
    )
    checks.append(
        Check(
            "blind_sam_is_rejected_not_production_default",
            "Blind automatic SAM is rejected" in readme
            and "blind SAM" in issue_tracker
            and "has_sam_like_sidecar" in run_audit
            and "--allow-legacy-blind-sam-sidecar" in trainer,
            "Blind SAM may remain only as archived legacy/reproduction evidence, never as the maintained object source.",
        )
    )
    checks.append(
        Check(
            "latest_slot_closure_doc_is_linked_from_readme",
            "latest_slot_full_deployment_closure_20260519.md" in readme
            and closure_doc.exists()
            and "Final closure verdict" in closure_text,
            "README_v2.2 must route reviewers to the current closure verdict.",
        )
    )
    checks.append(
        Check(
            "latest_slot_final_audit_records_no_new_mandatory_module",
            "PICF_AQR_OWM_LATEST_SLOT_FINAL_AUDIT_20260520_TEMP.md" in readme
            and final_audit_doc.exists()
            and _all(
                final_audit_text,
                (
                    "No new mandatory slot module was found",
                    "slot-axis evidence competition",
                    "object/background residual explanation",
                    "Object-Binding-style pairwise/quadratic same-object subspace",
                    "tactile/contact routing to object owner",
                    "blind SAM proposals",
                    "hard visual VQ posterior truth",
                    "behavior acceptance still requires longer co-training plus CALVIN/video evidence",
                ),
            ),
            "The current README must link the final latest-slot audit and record both accepted and rejected mechanisms.",
        )
    )
    checks.append(
        Check(
            "audits_cover_static_math_and_runtime_contracts",
            "posterior_update_closes_owner_responsibility_to_file_geometry" in object_audit
            and "candidate_top1_suppresses_raw_same_candidate_clones" in object_audit
            and "owner_only_probe_disables_effector_competition" in object_audit,
            "Executable audits must cover object-owner transport, clone suppression, and effector/object leakage.",
        )
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fail-on-fail", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks = run(args.repo_root.resolve())
    ok = all(check.ok for check in checks)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "checks": [check.__dict__ for check in checks],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for check in checks:
            status = "PASS" if check.ok else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
        print(f"SUMMARY: {sum(1 for check in checks if check.ok)}/{len(checks)} PASS")
    if args.fail_on_fail and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
