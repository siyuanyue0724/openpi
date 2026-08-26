from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.finalize_molmoact2_m1 import finalize_m1
from tools.run_molmoact2_m1_cloud import (
    _EXPECTED_FILE_MANIFEST_SHA256,
    _EXPECTED_TRAIN_MANIFEST_SHA256,
    _EXPECTED_VALIDATION_MANIFEST_SHA256,
    _M1_MACHINE_REQUIRED_REPORTS,
    _is_under_mnt,
    _sha256,
    build_sample_plan,
    build_visual_artifact_manifest,
    validate_full_audit,
    validate_m1_ddp_report,
    validate_m1_machine_decision,
    validate_m1_visual_review,
    validate_split_summary,
    validate_visual_artifact_manifest,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _passing_full_audit() -> dict:
    representatives = [
        {
            "task_index": task_index,
            "task": f"task {task_index}",
            "episode_index": task_index,
            "length": 9,
            "phase_global_indices": [task_index * 10, task_index * 10 + 4, task_index * 10 + 8],
            "panel": f"task_{task_index:02d}_episode_{task_index:04d}.png",
        }
        for task_index in range(40)
    ]
    return {
        "status": "PASS_WITH_UPSTREAM_LOCATOR_BUG_MITIGATED",
        "dataset_id": "allenai/MolmoAct2-LIBERO-Dataset",
        "dataset_revision": "fe3ead447f44c0ea950396360b304cc2fb6be8f8",
        "tree": {
            "files": 385,
            "data_shards": 379,
            "bytes": 34_935_776_578,
            "hashes_verified": True,
            "canonical_file_manifest_sha256": _EXPECTED_FILE_MANIFEST_SHA256,
        },
        "rows": {
            "episodes": 1693,
            "frames": 273465,
            "tasks": 40,
            "fps": 10,
            "state_shape": [8],
            "action_shape": [7],
        },
        "upstream_episode_locator": {
            "mismatched_episodes": 373,
            "raw_files_modified": False,
        },
        "visual_sample": {
            "decoded_images": 240,
            "representatives": representatives,
        },
    }


def _passing_split() -> dict:
    return {
        "schema": "picf-next.libero-episode-split.v1",
        "dataset_id": "allenai/MolmoAct2-LIBERO-Dataset",
        "dataset_revision": "fe3ead447f44c0ea950396360b304cc2fb6be8f8",
        "train_episodes": 1518,
        "validation_episodes": 175,
        "train_frames": 245510,
        "validation_frames": 27955,
        "tasks_each_arm": 40,
        "train_manifest_sha256": _EXPECTED_TRAIN_MANIFEST_SHA256,
        "validation_manifest_sha256": _EXPECTED_VALIDATION_MANIFEST_SHA256,
        "locator_fields_used": False,
        "episode_task_stats_used": False,
    }


def _passing_ddp_report() -> dict:
    return {
        "schema": "picf-next.molmoact2-m1-ddp.v1",
        "status": "PASS",
        "gate": "M1_typed_full_manifest",
        "world_size": 2,
        "dataset": {
            "id": "allenai/MolmoAct2-LIBERO-Dataset",
            "revision": "fe3ead447f44c0ea950396360b304cc2fb6be8f8",
            "selected_episodes": 40,
            "selected_tasks": 40,
            "selected_representative_rows": 120,
            "official_loader": "lerobot.datasets.io_utils.load_nested_dataset",
            "loader_discovers_all_physical_shards_once": True,
            "episode_filter_field": "episode_index",
            "episode_locator_fields_used": False,
        },
        "typed_contract": {
            "state_shape": [8],
            "action_shape": [10, 7],
            "delta_t_s": 0.1,
            "metadata_state_names_trusted": False,
        },
        "processor": {
            "factory": "make_molmoact2_pre_post_processors",
            "all_representatives_processed": True,
            "action_mode": "continuous",
            "action_horizon": 10,
        },
        "no_leak": {
            "representative_rows_checked": 120,
            "target_free_action_is_none": True,
            "target_free_labels_absent": True,
            "targetful_labels_absent_for_continuous_mode": True,
            "observation_tensors_exactly_equal_with_and_without_action_target": True,
        },
        "continuation": {
            "checkpoint_resume_exact": True,
            "rank_local_cursor_exact": True,
            "rng_exact": True,
            "loader_processor_trace_exact": True,
            "optimizer_scheduler_model_exact": True,
            "corrupted_checkpoint_failed_closed_on_all_ranks": True,
            "model_sha256": "a" * 64,
        },
        "resources": [
            {"device_name": "NVIDIA A100-PCIE-40GB"},
            {"device_name": "NVIDIA A100-PCIE-40GB"},
        ],
    }


def test_m1_full_audit_and_split_validators_are_fail_closed() -> None:
    validate_full_audit(_passing_full_audit())
    validate_split_summary(_passing_split())

    audit = _passing_full_audit()
    audit["tree"]["hashes_verified"] = False
    with pytest.raises(ValueError, match="tree field changed"):
        validate_full_audit(audit)

    split = _passing_split()
    split["locator_fields_used"] = True
    with pytest.raises(ValueError, match="split changed"):
        validate_split_summary(split)


def test_m1_sample_plan_covers_all_tasks_and_phases() -> None:
    audit = _passing_full_audit()
    overlay = [
        {"episode_index": episode_index, "mismatch": episode_index % 3 == 0}
        for episode_index in range(40)
    ]
    plan = build_sample_plan(audit_report=audit, locator_overlay=overlay)

    assert plan["schema"] == "picf-next.molmoact2-m1-sample-plan.v1"
    assert len(plan["representatives"]) == 120
    assert {row["task_index"] for row in plan["representatives"]} == set(range(40))
    assert {row["phase"] for row in plan["representatives"]} == {
        "start",
        "middle",
        "end",
    }
    assert plan["representatives_sha256"] == _canonical_sha256(plan["representatives"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["dataset"].update(episode_locator_fields_used=True),
            "loader contract",
        ),
        (
            lambda report: report["processor"].update(action_mode="both"),
            "processor contract",
        ),
        (
            lambda report: report["no_leak"].update(target_free_action_is_none=False),
            "no-leak",
        ),
        (
            lambda report: report["continuation"].update(checkpoint_resume_exact=False),
            "continuation",
        ),
        (
            lambda report: report.update(resources=[{"device_name": "NVIDIA A100"}]),
            "resource reports",
        ),
    ],
)
def test_m1_ddp_report_acceptance_is_fail_closed(mutation, message: str) -> None:
    report = _passing_ddp_report()
    validate_m1_ddp_report(report)
    changed = copy.deepcopy(report)
    mutation(changed)
    with pytest.raises(ValueError, match=message):
        validate_m1_ddp_report(changed)


def test_m1_cloud_paths_must_be_persistent() -> None:
    assert _is_under_mnt(Path("/mnt/picf-next/runs"))
    assert not _is_under_mnt(Path("/tmp/picf-next/runs"))


def _write_visual_fixture(run_dir: Path) -> tuple[dict, dict]:
    audit_dir = run_dir / "full_audit"
    audit_dir.mkdir(parents=True)
    audit = _passing_full_audit()
    audit["visual_sample"]["overview"] = "all_40_tasks.png"
    (audit_dir / "full_audit.json").write_text(json.dumps(audit))
    (audit_dir / "episode_file_locator_overlay.json").write_text("[]\n")
    (audit_dir / "all_40_tasks.png").write_bytes(b"overview")
    for representative in audit["visual_sample"]["representatives"]:
        (audit_dir / representative["panel"]).write_bytes(
            f"task-{representative['task_index']}".encode()
        )
    manifest = build_visual_artifact_manifest(
        audit_dir=audit_dir,
        audit_report=audit,
    )
    (run_dir / "visual_artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    review = {
        "schema": "picf-next.molmoact2-m1-visual-review.v1",
        "reviewer": "Codex visual inspection",
        "date": "2026-07-16 Asia/Shanghai",
        "status": "PASS",
        "machine_report_sha256": _sha256(audit_dir / "full_audit.json"),
        "visual_artifacts_sha256": _sha256(run_dir / "visual_artifacts.json"),
        "overview_sha256": manifest["overview"]["sha256"],
        "locator_overlay_file_sha256": manifest["locator_overlay_sha256"],
        "overview_tasks_reviewed": 40,
        "decoded_images_reviewed": 240,
        "individually_enlarged_task_indices": [2, 12, 22, 32],
        "checks": {
            "task_text_matches_external_trajectory": True,
            "start_middle_end_order_is_coherent": True,
            "external_and_wrist_camera_roles_are_consistent": True,
            "button_drawer_container_and_spatial_relation_tasks_are_coherent": True,
            "obvious_cross_episode_or_cross_task_row_mix": False,
        },
        "observations": ["The deterministic representative trajectories are coherent."],
    }
    return manifest, review


def test_m1_visual_evidence_is_run_bound_and_fail_closed(tmp_path: Path) -> None:
    manifest, review = _write_visual_fixture(tmp_path)
    validate_visual_artifact_manifest(run_dir=tmp_path, manifest=manifest)
    validate_m1_visual_review(review, run_dir=tmp_path)

    changed_review = copy.deepcopy(review)
    changed_review["overview_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not bound"):
        validate_m1_visual_review(changed_review, run_dir=tmp_path)

    changed_review = copy.deepcopy(review)
    changed_review["individually_enlarged_task_indices"] = [2, 12, 32]
    with pytest.raises(ValueError, match="LIBERO suite"):
        validate_m1_visual_review(changed_review, run_dir=tmp_path)

    (tmp_path / manifest["panels"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after machine audit"):
        validate_visual_artifact_manifest(run_dir=tmp_path, manifest=manifest)


def test_m1_machine_decision_covers_exact_report_inventory(tmp_path: Path) -> None:
    _manifest, _review = _write_visual_fixture(tmp_path)
    for relative in _M1_MACHINE_REQUIRED_REPORTS:
        path = tmp_path / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative + "\n")
    hashes = {relative: _sha256(tmp_path / relative) for relative in _M1_MACHINE_REQUIRED_REPORTS}
    decision = {
        "schema": "picf-next.molmoact2-m1-machine-decision.v1",
        "status": "PASS_PENDING_VISUAL_REVIEW",
        "gate": "M1_typed_full_manifest",
        "required_report_sha256": hashes,
        "later_gates_authorized": [],
    }
    (tmp_path / "machine_decision.json").write_text(json.dumps(decision))
    validate_m1_machine_decision(tmp_path)

    changed = copy.deepcopy(decision)
    changed["later_gates_authorized"] = ["M2_representation_smoke"]
    (tmp_path / "machine_decision.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="not awaiting visual review"):
        validate_m1_machine_decision(tmp_path)


def test_m1_finalizer_is_recoverable_and_authorizes_only_m2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, review = _write_visual_fixture(tmp_path)
    for relative in _M1_MACHINE_REQUIRED_REPORTS:
        path = tmp_path / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative + "\n")
    hashes = {relative: _sha256(tmp_path / relative) for relative in _M1_MACHINE_REQUIRED_REPORTS}
    machine_decision = {
        "schema": "picf-next.molmoact2-m1-machine-decision.v1",
        "status": "PASS_PENDING_VISUAL_REVIEW",
        "gate": "M1_typed_full_manifest",
        "required_report_sha256": hashes,
        "later_gates_authorized": [],
    }
    (tmp_path / "machine_decision.json").write_text(json.dumps(machine_decision))
    review_path = tmp_path / "review-source.json"
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr("tools.finalize_molmoact2_m1._is_under_mnt", lambda _path: True)

    decision = finalize_m1(run_dir=tmp_path, visual_review_path=review_path)

    assert decision["status"] == "PASS"
    assert decision["later_gates_authorized"] == ["M2_representation_smoke"]
    assert json.loads((tmp_path / "visual_review.json").read_text()) == review
    assert json.loads((tmp_path / "gate_decision.json").read_text()) == decision
    with pytest.raises(FileExistsError, match="final gate decision"):
        finalize_m1(run_dir=tmp_path, visual_review_path=review_path)
