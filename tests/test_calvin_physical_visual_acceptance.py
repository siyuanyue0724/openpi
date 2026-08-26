from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_visual_acceptance import (
    CALVIN_PHYSICAL_AUDIT_SCHEMA,
    CALVIN_PHYSICAL_VISUAL_REVIEW_SCHEMA,
    build_calvin_physical_visual_acceptance,
    load_calvin_physical_visual_acceptance,
)
from picf_next.data.lingbot_calvin_projection import projection_payload_sha256
from picf_next.data.token_supervision_policy import (
    build_known_pixel_token_supervision_policy,
    token_supervision_policy_sha256,
)


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _projection(dataset_sha: str) -> dict[str, object]:
    def view(source_field: str, shape: list[int], digit: str) -> dict[str, object]:
        return {
            "source_field": source_field,
            "source_shape": shape,
            "image_grid_thw": [1, 16, 16],
            "merged_grid_hw": [8, 8],
            "raw_patch_count": 256,
            "merged_token_count": 64,
            "pixel_values_shape": [256, 1536],
            "source_rgb_sha256": [digit * 64] * 3,
        }

    return {
        "schema": "picf-next.lingbot-calvin-qwen-projection.v1",
        "status": "PASS",
        "runtime_input": False,
        "processor_id": "Qwen/Qwen3-VL-4B-Instruct",
        "processor_revision": "c" * 40,
        "processor_assets_sha256": "d" * 64,
        "processor_config_sha256": "e" * 64,
        "processor_preprocessor_config_sha256": "f" * 64,
        "dataset_manifest_sha256": dataset_sha,
        "dataset_tree_sha256": "0" * 64,
        "source_frame_count": 20,
        "sample_global_indices": [0, 10, 19],
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "views": {
            "static": view("rgb_static", [200, 200, 3], "1"),
            "gripper": view("rgb_gripper", [84, 84, 3], "2"),
        },
        "transformers_version": "5.0.0",
    }


def _artifacts(tmp_path: Path) -> tuple[Path, str, Path, str, str]:
    sidecar_sha = "a" * 64
    dataset_sha = "b" * 64
    records = []
    for index in (7, 11):
        panel = tmp_path / f"step{index:07d}_task.png"
        panel.write_bytes(f"panel-{index}".encode("ascii"))
        records.append(
            {
                "global_index": index,
                "selection_reasons": ["temporal"],
                "task_annotations": [],
                "identity_keys": ["object"],
                "visible_identity_keys": ["object"],
                "panel": panel.name,
                "panel_sha256": _sha256(panel),
                "cameras": {"static": {}, "gripper": {}},
                "scanned_metrics": {"static": {}, "gripper": {}},
            }
        )
    projection = _projection(dataset_sha)
    supervision_policy = build_known_pixel_token_supervision_policy()
    audit = {
        "format": CALVIN_PHYSICAL_AUDIT_SCHEMA,
        "mode": "full_tail",
        "runtime_input": False,
        "task_used_for_owner_selection": False,
        "task_used_for_audit_selection": True,
        "selection_affects_training": False,
        "coverage": "all_source_frames",
        "dataset_manifest_sha256": dataset_sha,
        "sidecar_manifest_sha256": sidecar_sha,
        "training_projection_contract_sha256": "9" * 64,
        "training_projection_payload_sha256": projection_payload_sha256(projection),
        "training_projection": projection,
        "training_supervision_policy_sha256": token_supervision_policy_sha256(supervision_policy),
        "training_supervision_policy": supervision_policy,
        "frame_count": 20,
        "first_global_index": 0,
        "last_global_index": 19,
        "full_shard_schema_validation": True,
        "manifest_summary_match": True,
        "manifest_summary_absolute_error": 0.0,
        "distributions": {},
        "selection_contract": {
            "tail_per_metric": 4,
            "tail_directions": {
                "rgb_mae": ["high"],
                "depth_mae_m": ["high"],
                "depth_p95_m": ["high"],
                "known_pixel_fraction": ["low"],
                "raw_object_pixel_fraction": ["high"],
                "known_object_pixel_fraction": ["low", "high"],
                "known_owner_retention": ["low"],
            },
            "temporal_strata": 16,
            "one_median_occurrence_midpoint_per_task": True,
            "deduplicated": True,
        },
        "record_count": len(records),
        "records": records,
    }
    audit_path = tmp_path / "audit_manifest.json"
    audit_sha = _write_json(audit_path, audit)
    review = {
        "schema": CALVIN_PHYSICAL_VISUAL_REVIEW_SCHEMA,
        "reviewer": "Codex image review",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_manifest_sha256": audit_sha,
        "sidecar_manifest_sha256": sidecar_sha,
        "rows": [
            {
                "global_index": record["global_index"],
                "panel": record["panel"],
                "panel_sha256": record["panel_sha256"],
                "verdict": "PASS",
                "observations": "Both camera views and overlays match the visible scene.",
                "context_expanded": False,
            }
            for record in records
        ],
        "checks": {
            "every_panel_opened_original_resolution": True,
            "both_camera_views_reviewed": True,
            "task_annotation_matches_scene": True,
            "visible_owner_assignment_is_correct": True,
            "unknown_regions_do_not_paint_hidden_objects": True,
            "training_token_overlay_is_consistent": True,
            "partially_observed_tokens_are_visually_distinct": True,
            "ambiguous_cases_expanded": True,
        },
        "status": "PASS",
        "findings": "Every deterministic full-tail panel passed the recorded visual checks.",
    }
    review_path = tmp_path / "visual_review.json"
    review_sha = _write_json(review_path, review)
    return audit_path, audit_sha, review_path, review_sha, sidecar_sha


def test_visual_acceptance_rehashes_every_panel_and_loads(tmp_path: Path) -> None:
    audit_path, audit_sha, review_path, review_sha, sidecar_sha = _artifacts(tmp_path)
    acceptance = build_calvin_physical_visual_acceptance(
        audit_manifest_path=audit_path,
        audit_manifest_sha256=audit_sha,
        dataset_manifest_sha256="b" * 64,
        sidecar_manifest_sha256=sidecar_sha,
        review_path=review_path,
        review_sha256=review_sha,
    )
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_sha = _write_json(acceptance_path, acceptance)
    assert (
        acceptance["training_supervision_policy"]["unknown_pixel_semantics"]
        == "zero-loss-mass-never-context"
    )
    assert (
        load_calvin_physical_visual_acceptance(
            acceptance_path,
            expected_sha256=acceptance_sha,
            expected_dataset_manifest_sha256="b" * 64,
            expected_sidecar_manifest_sha256=sidecar_sha,
        )
        == acceptance
    )

    (tmp_path / "step0000007_task.png").write_bytes(b"changed")
    with pytest.raises(ContractError, match="panel differs"):
        build_calvin_physical_visual_acceptance(
            audit_manifest_path=audit_path,
            audit_manifest_sha256=audit_sha,
            dataset_manifest_sha256="b" * 64,
            sidecar_manifest_sha256=sidecar_sha,
            review_path=review_path,
            review_sha256=review_sha,
        )


def test_visual_acceptance_rejects_token_supervision_policy_drift(tmp_path: Path) -> None:
    audit_path, _audit_sha, review_path, _review_sha, sidecar_sha = _artifacts(tmp_path)
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    audit["training_supervision_policy"]["unknown_pixel_semantics"] = "unknown-becomes-context"
    audit_sha = _write_json(audit_path, audit)
    review = json.loads(review_path.read_text(encoding="ascii"))
    review["audit_manifest_sha256"] = audit_sha
    review_sha = _write_json(review_path, review)

    with pytest.raises(ContractError, match="semantics changed"):
        build_calvin_physical_visual_acceptance(
            audit_manifest_path=audit_path,
            audit_manifest_sha256=audit_sha,
            dataset_manifest_sha256="b" * 64,
            sidecar_manifest_sha256=sidecar_sha,
            review_path=review_path,
            review_sha256=review_sha,
        )


def test_visual_acceptance_rejects_skipped_or_false_review(tmp_path: Path) -> None:
    audit_path, audit_sha, review_path, _review_sha, sidecar_sha = _artifacts(tmp_path)
    review = json.loads(review_path.read_text(encoding="ascii"))
    review["rows"].pop()
    review_sha = _write_json(review_path, review)
    with pytest.raises(ContractError, match="skipped"):
        build_calvin_physical_visual_acceptance(
            audit_manifest_path=audit_path,
            audit_manifest_sha256=audit_sha,
            dataset_manifest_sha256="b" * 64,
            sidecar_manifest_sha256=sidecar_sha,
            review_path=review_path,
            review_sha256=review_sha,
        )

    audit_path, audit_sha, review_path, _review_sha, sidecar_sha = _artifacts(tmp_path)
    review = json.loads(review_path.read_text(encoding="ascii"))
    review["rows"][0]["verdict"] = "FAIL"
    review["status"] = "FAIL"
    review_sha = _write_json(review_path, review)
    with pytest.raises(ContractError, match="did not pass"):
        build_calvin_physical_visual_acceptance(
            audit_manifest_path=audit_path,
            audit_manifest_sha256=audit_sha,
            dataset_manifest_sha256="b" * 64,
            sidecar_manifest_sha256=sidecar_sha,
            review_path=review_path,
            review_sha256=review_sha,
        )


def test_visual_acceptance_binds_dataset_and_preregistered_selection(tmp_path: Path) -> None:
    audit_path, audit_sha, review_path, review_sha, sidecar_sha = _artifacts(tmp_path)
    with pytest.raises(ContractError, match="machine contract"):
        build_calvin_physical_visual_acceptance(
            audit_manifest_path=audit_path,
            audit_manifest_sha256=audit_sha,
            dataset_manifest_sha256="c" * 64,
            sidecar_manifest_sha256=sidecar_sha,
            review_path=review_path,
            review_sha256=review_sha,
        )

    audit = json.loads(audit_path.read_text(encoding="ascii"))
    audit["selection_contract"]["tail_per_metric"] = 3
    audit_sha = _write_json(audit_path, audit)
    with pytest.raises(ContractError, match="selection contract changed"):
        build_calvin_physical_visual_acceptance(
            audit_manifest_path=audit_path,
            audit_manifest_sha256=audit_sha,
            dataset_manifest_sha256="b" * 64,
            sidecar_manifest_sha256=sidecar_sha,
            review_path=review_path,
            review_sha256=review_sha,
        )
