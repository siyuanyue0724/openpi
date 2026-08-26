from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from picf_next.eval.stationary_lifecycle import (
    build_stationary_lifecycle_calibration,
    validate_stationary_lifecycle_calibration,
)
from picf_next.eval.stationary_replay import (
    STATIONARY_FIXED_REPLAY_METRICS,
    compare_stationary_replay_summaries,
)
from picf_next.eval.stationary_runtime import (
    build_stationary_runtime_probe,
    validate_stationary_runtime_probe,
)
from picf_next.eval.stationary_visual import (
    validate_stationary_visual_artifacts,
    validate_stationary_visual_review,
)

_PREFIXES = (0, 8, 32, 128)
_SPLITS = ("validation", "heldout")
_MODELS = ("fresh_m2", "candidate")
_CAMERAS = ["observation.images.image", "observation.images.wrist_image"]
_PANELS = [
    "source",
    "loss_only_target",
    "fresh_m2_discovery",
    "candidate_discovery",
    "fresh_m2_persistent_posterior",
    "candidate_persistent_posterior",
]
_VISUAL_CHECKS = {
    "all_manifest_artifacts_reviewed": True,
    "all_camera_panels_legible": True,
    "candidate_object_identity_alignment_acceptable": True,
    "no_catastrophic_off_object_collapse": True,
    "occlusion_uncertainty_not_misrepresented_as_fresh_observation": True,
    "no_mask_or_identity_input_leak": True,
    "no_task_text_input_leak": True,
    "task_annotation_present_or_explicitly_independent": True,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")


def _replay() -> dict[str, object]:
    fresh = {
        name: (0.5 if "iou" in name or "coverage" in name else 1.0)
        for name in STATIONARY_FIXED_REPLAY_METRICS
    }
    candidate = {
        name: (0.6 if "iou" in name or "coverage" in name else 0.9)
        for name in STATIONARY_FIXED_REPLAY_METRICS
    }
    comparisons = compare_stationary_replay_summaries(
        fresh_m2=fresh,
        candidate=candidate,
        absolute_tolerance=1e-6,
    )
    measurements = []
    for split_index, split in enumerate(_SPLITS):
        for optimizer_step, prefix_length in enumerate(_PREFIXES):
            for rank in (0, 1):
                start = split_index * 1000 + optimizer_step * 200 + rank * 20
                clip = {
                    "optimizer_step": optimizer_step,
                    "source_range_index": 0,
                    "start_global_index": start,
                    "prefix_length": prefix_length,
                    "train_length": 2,
                    "train_start_global_index": start + prefix_length,
                    "stop_global_index": start + prefix_length + 2,
                }
                for model, metrics in (("fresh_m2", fresh), ("candidate", candidate)):
                    measurements.append(
                        {
                            "clip": clip,
                            "metrics": metrics,
                            "model": model,
                            "optimizer_step": optimizer_step,
                            "rank": rank,
                            "split": split,
                        }
                    )
    checks = {
        f"{split}_{name}": passed for split in _SPLITS for name, passed in comparisons.items()
    }
    return {
        "schema": "picf-next.stationary-fixed-checkpoint-replay.v2",
        "status": "PASS",
        "protocol": {
            "comparison": "same-frozen-clips-fresh-m2-vs-stage-b-candidate.v1",
            "observation_inputs": "task-independent-cached-native-token-bank",
            "target_use": "post-forward-loss-and-evaluation-only",
            "split_names": list(_SPLITS),
            "prefix_lengths": list(_PREFIXES),
            "train_length": 2,
            "world_size": 2,
            "optimizer_steps_per_split": 4,
            "seed": 20260720,
        },
        "bindings": {
            "audit_code_revision": "a" * 40,
            "candidate_code_revision": "b" * 40,
            "candidate_checkpoint_sha256": "c" * 64,
            "candidate_report_sha256": "d" * 64,
            "dataset_manifest_sha256": "e" * 64,
            "feature_cache_manifest_sha256": "f" * 64,
            "foundation_recipe_sha256": "0" * 64,
            "m2_checkpoint_sha256": "1" * 64,
            "m2_report_sha256": "2" * 64,
            "physical_sidecar_manifest_sha256": "3" * 64,
            "source_coverage_recipe_sha256": "4" * 64,
            "stage_recipe_sha256": "5" * 64,
        },
        "plans": {
            split: {"plan_sha256": character * 64, "source_ranges": [[0, 200]]}
            for split, character in (("validation", "6"), ("heldout", "7"))
        },
        "thresholds": {
            "absolute_tolerance": 1e-6,
            "lower_is_better": [
                "loss_total",
                "loss_set",
                "loss_dynamics",
                "loss_dynamics_survival",
                "loss_dynamics_visibility",
                "loss_binding",
                "assignment_conflicts_per_clip",
            ],
            "higher_is_better": [
                "discovery_soft_iou",
                "posterior_soft_iou",
                "posterior_identity_coverage",
            ],
        },
        "splits": {
            split: {
                "clip_count": 8,
                "models": {"fresh_m2": fresh, "candidate": candidate},
                "comparisons": comparisons,
            }
            for split in _SPLITS
        },
        "checks": checks,
        "failed_checks": [],
        "measurements": measurements,
        "long_training_authorized": False,
    }


def _runtime_rows(replay: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "model": row["model"],
            "split": row["split"],
            "optimizer_step": row["optimizer_step"],
            "rank": row["rank"],
            "prefix_length": row["clip"]["prefix_length"],
            "transition_count": (row["clip"]["prefix_length"] + row["clip"]["train_length"]),
            "elapsed_seconds": 0.01,
            "peak_allocated_bytes": 1024,
        }
        for row in replay["measurements"]
    ]


def _visual_evidence(
    root: Path,
    *,
    fixed_replay_sha256: str,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    visuals = root / "visuals"
    visuals.mkdir()
    artifacts = []
    for split in _SPLITS:
        for prefix_length in _PREFIXES:
            for rank in (0, 1):
                path = visuals / (
                    f"{split}_prefix{prefix_length:03d}_rank{rank}_task_independent.png"
                )
                path.write_bytes(f"{split}:{prefix_length}:{rank}".encode("ascii"))
                artifacts.append(
                    {
                        "bytes": path.stat().st_size,
                        "cameras": _CAMERAS,
                        "global_index": prefix_length + rank + 1,
                        "optimizer_step": _PREFIXES.index(prefix_length),
                        "panels": _PANELS,
                        "path": f"visuals/{path.name}",
                        "prefix_length": prefix_length,
                        "rank": rank,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "split": split,
                        "tasks": [],
                        "lifecycle_targets": [
                            {
                                "identity_key": "object:visible",
                                "currently_measurable": True,
                                "conditional_detection_target": 1.0,
                                "conditional_detection_supervised": True,
                                "ever_measurable_before_final": True,
                                "last_measurable_global_index": prefix_length + rank + 1,
                                "terminal_unmeasurable_frames": 0,
                                "seen_then_unmeasurable": False,
                                "candidate_posterior_identity_retained": True,
                                "candidate_posterior_map_present": True,
                                "candidate_posterior_existence": 0.9,
                            },
                            {
                                "identity_key": "object:occluded",
                                "currently_measurable": prefix_length < 32,
                                "conditional_detection_target": (
                                    1.0 if prefix_length < 32 else 0.0
                                ),
                                "conditional_detection_supervised": True,
                                "ever_measurable_before_final": True,
                                "last_measurable_global_index": (
                                    prefix_length + rank + 1
                                    if prefix_length < 32
                                    else prefix_length + rank - 7
                                ),
                                "terminal_unmeasurable_frames": (0 if prefix_length < 32 else 8),
                                "seen_then_unmeasurable": prefix_length >= 32,
                                "candidate_posterior_identity_retained": True,
                                "candidate_posterior_map_present": prefix_length < 32,
                                "candidate_posterior_existence": 0.4,
                            },
                        ],
                    }
                )
    manifest = {
        "schema": "picf-next.stationary-replay-visual-artifacts.v3",
        "status": "PENDING_HUMAN_REVIEW",
        "candidate_checkpoint_sha256": "c" * 64,
        "fixed_checkpoint_replay_sha256": fixed_replay_sha256,
        "artifact_count": len(artifacts),
        "required_split_prefix_rank_coverage": [
            {"split": split, "prefix_length": prefix_length, "rank": rank}
            for split in _SPLITS
            for prefix_length in _PREFIXES
            for rank in (0, 1)
        ],
        "artifacts": artifacts,
        "artifacts_sha256": _canonical_sha256(artifacts),
        "mask_or_identity_visible_to_model": False,
        "task_text_visible_to_stationary_model": False,
    }
    manifest_path = root / "visual_artifacts.json"
    _write_json(manifest_path, manifest)
    review = {
        "schema": "picf-next.stationary-replay-visual-review.v1",
        "status": "PASS",
        "reviewer": "test-reviewer",
        "reviewed_at_utc": "2026-07-19T00:00:00+00:00",
        "bindings": {
            "visual_artifacts_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "candidate_checkpoint_sha256": "c" * 64,
            "fixed_checkpoint_replay_sha256": fixed_replay_sha256,
        },
        "reviewed_artifacts": [
            {
                "path": artifact["path"],
                "sha256": artifact["sha256"],
                "status": "PASS",
                "observations": "Both camera rows and all six panels were inspected.",
            }
            for artifact in artifacts
        ],
        "checks": _VISUAL_CHECKS,
        "failed_checks": [],
        "findings": ["No catastrophic off-object collapse was observed."],
        "long_training_authorized": False,
    }
    return manifest, manifest_path, review


def test_lifecycle_calibration_is_an_exact_projection_of_replay() -> None:
    replay = _replay()
    report = build_stationary_lifecycle_calibration(
        replay,
        fixed_replay_sha256="8" * 64,
    )
    validate_stationary_lifecycle_calibration(
        report,
        fixed_replay=replay,
        fixed_replay_sha256="8" * 64,
    )

    changed = copy.deepcopy(report)
    changed["splits"]["heldout"]["models"]["candidate"]["survival_log_loss"] = 0.0
    with pytest.raises(ValueError, match="differs from fixed replay"):
        validate_stationary_lifecycle_calibration(
            changed,
            fixed_replay=replay,
            fixed_replay_sha256="8" * 64,
        )


def test_runtime_probe_recomputes_coverage_memory_and_decision() -> None:
    replay = _replay()
    report = build_stationary_runtime_probe(
        replay,
        fixed_replay_sha256="8" * 64,
        candidate_recurrent_state_serialized=False,
        device_name="NVIDIA A100-SXM4-40GB",
        total_memory_bytes=40 * 2**30,
        measurements=_runtime_rows(replay),
    )
    assert report["status"] == "PASS"
    validate_stationary_runtime_probe(
        report,
        fixed_replay=replay,
        fixed_replay_sha256="8" * 64,
        candidate_recurrent_state_serialized=False,
    )

    changed = copy.deepcopy(report)
    changed["measurements"].pop()
    with pytest.raises(ValueError, match="coverage changed"):
        validate_stationary_runtime_probe(
            changed,
            fixed_replay=replay,
            fixed_replay_sha256="8" * 64,
            candidate_recurrent_state_serialized=False,
        )

    changed = copy.deepcopy(report)
    changed["measurements"] = None
    with pytest.raises(ValueError, match="must be one list"):
        validate_stationary_runtime_probe(
            changed,
            fixed_replay=replay,
            fixed_replay_sha256="8" * 64,
            candidate_recurrent_state_serialized=False,
        )


def test_visual_review_binds_every_file_and_rejects_false_pass(tmp_path: Path) -> None:
    manifest, manifest_path, review = _visual_evidence(
        tmp_path,
        fixed_replay_sha256="8" * 64,
    )
    validated = validate_stationary_visual_artifacts(manifest, evidence_root=tmp_path)
    validate_stationary_visual_review(
        review,
        manifest=validated,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        evidence_root=tmp_path,
    )

    changed_review = copy.deepcopy(review)
    changed_review["reviewed_artifacts"][0]["status"] = "FAIL"
    with pytest.raises(ValueError, match="not recomputed exactly"):
        validate_stationary_visual_review(
            changed_review,
            manifest=manifest,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            evidence_root=tmp_path,
        )

    first = tmp_path / manifest["artifacts"][0]["path"]
    first.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte count changed|content changed"):
        validate_stationary_visual_artifacts(manifest, evidence_root=tmp_path)


def test_visual_artifacts_require_seen_occlusion_in_both_splits(tmp_path: Path) -> None:
    manifest, _, _ = _visual_evidence(tmp_path, fixed_replay_sha256="8" * 64)
    for artifact in manifest["artifacts"]:
        if artifact["split"] == "heldout":
            for lifecycle in artifact["lifecycle_targets"]:
                lifecycle["currently_measurable"] = True
                lifecycle["conditional_detection_target"] = 1.0
                lifecycle["last_measurable_global_index"] = artifact["global_index"]
                lifecycle["terminal_unmeasurable_frames"] = 0
                lifecycle["seen_then_unmeasurable"] = False
    manifest["artifacts_sha256"] = _canonical_sha256(manifest["artifacts"])
    with pytest.raises(ValueError, match="seen-then-unmeasurable coverage"):
        validate_stationary_visual_artifacts(manifest, evidence_root=tmp_path)


def test_visual_artifacts_reject_lost_seen_occluded_identity(tmp_path: Path) -> None:
    manifest, _, _ = _visual_evidence(tmp_path, fixed_replay_sha256="8" * 64)
    for artifact in manifest["artifacts"]:
        if artifact["split"] == "heldout" and artifact["prefix_length"] >= 32:
            lifecycle = artifact["lifecycle_targets"][1]
            lifecycle["candidate_posterior_identity_retained"] = False
            lifecycle["candidate_posterior_map_present"] = False
            lifecycle["candidate_posterior_existence"] = None
    manifest["artifacts_sha256"] = _canonical_sha256(manifest["artifacts"])
    with pytest.raises(ValueError, match="failed to retain"):
        validate_stationary_visual_artifacts(manifest, evidence_root=tmp_path)
