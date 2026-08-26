# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from picf_next.eval.stationary_lifecycle import build_stationary_lifecycle_calibration
from picf_next.eval.stationary_replay import (
    STATIONARY_FIXED_REPLAY_METRICS,
    compare_stationary_replay_summaries,
)
from picf_next.eval.stationary_runtime import build_stationary_runtime_probe
from picf_next.training.stage_checkpoints import StationaryTemporalCheckpointProvenance, sha256_file
from picf_next.training.stationary_acceptance import (
    validate_stationary_candidate_metrics,
    validate_stationary_temporal_acceptance,
)
from tools.finalize_stationary_temporal import finalize_stationary_temporal

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


def _provenance() -> StationaryTemporalCheckpointProvenance:
    return StationaryTemporalCheckpointProvenance(
        stage_recipe_sha256="1" * 64,
        source_coverage_recipe_sha256="2" * 64,
        foundation_recipe_sha256="3" * 64,
        m2_checkpoint_sha256="4" * 64,
        feature_cache_manifest_sha256="5" * 64,
        dataset_manifest_sha256="6" * 64,
        physical_sidecar_manifest_sha256="7" * 64,
        clip_plan_sha256="8" * 64,
        trainable_parameter_scope_sha256="9" * 64,
        frozen_parameter_scope_sha256="a" * 64,
        code_revision="b" * 40,
        optimizer_steps=200,
        state_parameter_version=200,
    )


def _fixed_replay(
    provenance: StationaryTemporalCheckpointProvenance,
    *,
    checkpoint_sha256: str,
    candidate_report_sha256: str,
) -> dict[str, object]:
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
    checks = {
        f"{split}_{name}": passed
        for split in ("validation", "heldout")
        for name, passed in comparisons.items()
    }
    measurements = []
    for split_index, split in enumerate(("validation", "heldout")):
        for optimizer_step, prefix_length in enumerate((0, 8, 32, 128)):
            for rank in range(2):
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
    return {
        "schema": "picf-next.stationary-fixed-checkpoint-replay.v2",
        "status": "PASS",
        "protocol": {
            "comparison": "same-frozen-clips-fresh-m2-vs-stage-b-candidate.v1",
            "observation_inputs": "task-independent-cached-native-token-bank",
            "target_use": "post-forward-loss-and-evaluation-only",
            "split_names": ["validation", "heldout"],
            "prefix_lengths": [0, 8, 32, 128],
            "train_length": 2,
            "world_size": 2,
            "optimizer_steps_per_split": 4,
            "seed": 20260720,
        },
        "bindings": {
            "audit_code_revision": "c" * 40,
            "candidate_code_revision": provenance.code_revision,
            "candidate_checkpoint_sha256": checkpoint_sha256,
            "candidate_report_sha256": candidate_report_sha256,
            "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
            "feature_cache_manifest_sha256": provenance.feature_cache_manifest_sha256,
            "foundation_recipe_sha256": provenance.foundation_recipe_sha256,
            "m2_checkpoint_sha256": provenance.m2_checkpoint_sha256,
            "m2_report_sha256": "d" * 64,
            "physical_sidecar_manifest_sha256": provenance.physical_sidecar_manifest_sha256,
            "source_coverage_recipe_sha256": provenance.source_coverage_recipe_sha256,
            "stage_recipe_sha256": provenance.stage_recipe_sha256,
        },
        "plans": {
            split: {"plan_sha256": character * 64, "source_ranges": [[0, 200]]}
            for split, character in (("validation", "e"), ("heldout", "f"))
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
            for split in ("validation", "heldout")
        },
        "checks": checks,
        "failed_checks": [],
        "measurements": measurements,
        "long_training_authorized": False,
    }


def _publish_acceptance(root: Path) -> tuple[Path, Path]:
    provenance = _provenance()
    checkpoint = root / "stationary_temporal_core_accepted.pt"
    torch.save(
        {
            "schema": "picf-next.stationary-temporal-core.v1",
            "provenance": provenance.to_dict(),
            "core": {"posterior_filter.fixture": torch.zeros(1)},
            "objective": {"fixture": torch.zeros(1)},
        },
        checkpoint,
    )
    metrics = root / "candidate_metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps(
                {
                    "optimizer_step": step,
                    "metrics": {
                        "prefix_length": (0, 8, 32, 128)[(step - 1) % 4],
                        "picf_lifecycle_survival_positive_target_mass": 3.0,
                        "picf_lifecycle_survival_negative_target_mass": 1.0,
                        "picf_lifecycle_detection_positive_target_mass": 3.0,
                        "picf_lifecycle_detection_negative_target_mass": 1.0,
                    },
                },
                sort_keys=True,
            )
            + "\n"
            for step in range(1, 201)
        ),
        encoding="ascii",
    )
    candidate = root / "candidate_report.json"
    candidate.write_text(
        json.dumps(
            {
                "schema": "picf-next.stationary-temporal-candidate-report.v1",
                "status": "CANDIDATE_REQUIRES_FIXED_CHECKPOINT_AUDIT",
                "stage_recipe_sha256": provenance.stage_recipe_sha256,
                "source_coverage_recipe_sha256": provenance.source_coverage_recipe_sha256,
                "foundation_recipe_sha256": provenance.foundation_recipe_sha256,
                "structural_recipe_sha256": "c" * 64,
                "clip_plan_sha256": provenance.clip_plan_sha256,
                "optimizer_steps": 200,
                "world_size": 2,
                "prefix_lengths": [0, 8, 32, 128],
                "train_length": 2,
                "required_future_horizon": 2,
                "action_weight": 0.0,
                "checkpoint_sha256": sha256_file(checkpoint),
                "metrics_sha256": sha256_file(metrics),
                "completed_optimizer_steps": 200,
                "long_training_authorized": False,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    fixed_replay = _fixed_replay(
        provenance,
        checkpoint_sha256=sha256_file(checkpoint),
        candidate_report_sha256=sha256_file(candidate),
    )
    fixed_path = root / "fixed_checkpoint_replay.json"
    fixed_path.write_text(json.dumps(fixed_replay, sort_keys=True), encoding="ascii")
    fixed_sha256 = sha256_file(fixed_path)

    lifecycle = build_stationary_lifecycle_calibration(
        fixed_replay,
        fixed_replay_sha256=fixed_sha256,
    )
    (root / "lifecycle_calibration.json").write_text(
        json.dumps(lifecycle, sort_keys=True),
        encoding="ascii",
    )
    runtime_rows = [
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
        for row in fixed_replay["measurements"]
    ]
    runtime = build_stationary_runtime_probe(
        fixed_replay,
        fixed_replay_sha256=fixed_sha256,
        candidate_recurrent_state_serialized=False,
        device_name="NVIDIA A100-SXM4-40GB",
        total_memory_bytes=40 * 2**30,
        measurements=runtime_rows,
    )
    (root / "runtime_probe.json").write_text(
        json.dumps(runtime, sort_keys=True),
        encoding="ascii",
    )

    visuals = root / "visuals"
    visuals.mkdir()
    artifacts = []
    for split in ("validation", "heldout"):
        for prefix_length in (0, 8, 32, 128):
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
                        "optimizer_step": (0, 8, 32, 128).index(prefix_length),
                        "panels": _PANELS,
                        "path": f"visuals/{path.name}",
                        "prefix_length": prefix_length,
                        "rank": rank,
                        "sha256": sha256_file(path),
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
    visual_manifest = {
        "schema": "picf-next.stationary-replay-visual-artifacts.v3",
        "status": "PENDING_HUMAN_REVIEW",
        "candidate_checkpoint_sha256": sha256_file(checkpoint),
        "fixed_checkpoint_replay_sha256": fixed_sha256,
        "artifact_count": len(artifacts),
        "required_split_prefix_rank_coverage": [
            {"split": split, "prefix_length": prefix_length, "rank": rank}
            for split in ("validation", "heldout")
            for prefix_length in (0, 8, 32, 128)
            for rank in (0, 1)
        ],
        "artifacts": artifacts,
        "artifacts_sha256": _canonical_sha256(artifacts),
        "mask_or_identity_visible_to_model": False,
        "task_text_visible_to_stationary_model": False,
    }
    visual_manifest_path = root / "visual_artifacts.json"
    visual_manifest_path.write_text(
        json.dumps(visual_manifest, sort_keys=True),
        encoding="ascii",
    )
    visual_review = {
        "schema": "picf-next.stationary-replay-visual-review.v1",
        "status": "PASS",
        "reviewer": "test-reviewer",
        "reviewed_at_utc": "2026-07-19T00:00:00+00:00",
        "bindings": {
            "visual_artifacts_sha256": sha256_file(visual_manifest_path),
            "candidate_checkpoint_sha256": sha256_file(checkpoint),
            "fixed_checkpoint_replay_sha256": fixed_sha256,
        },
        "reviewed_artifacts": [
            {
                "path": artifact["path"],
                "sha256": artifact["sha256"],
                "status": "PASS",
                "observations": "Both cameras and all six panels were inspected.",
            }
            for artifact in artifacts
        ],
        "checks": _VISUAL_CHECKS,
        "failed_checks": [],
        "findings": ["No catastrophic off-object collapse in this fixture."],
        "long_training_authorized": False,
    }
    (root / "visual_review.json").write_text(
        json.dumps(visual_review, sort_keys=True),
        encoding="ascii",
    )
    artifact_names = (
        "stationary_temporal_core_accepted.pt",
        "candidate_metrics.jsonl",
        "candidate_report.json",
        "fixed_checkpoint_replay.json",
        "lifecycle_calibration.json",
        "runtime_probe.json",
        "visual_artifacts.json",
        "visual_review.json",
    )
    checks = {
        "candidate_metrics_detection_support_validated": True,
        "candidate_report_validated": True,
        "fixed_checkpoint_replay_passed": True,
        "full_stationary_checkpoint_hash_bound": True,
        "lifecycle_calibration_passed": True,
        "no_recurrent_state_serialized": True,
        "runtime_probe_passed": True,
        "visual_review_passed": True,
    }
    report = root / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "picf-next.stationary-temporal-acceptance.v1",
                "status": "ACCEPTED_FOR_M4_ACTION_ADOPTION",
                "provenance": provenance.to_dict(),
                "artifacts_sha256": {name: sha256_file(root / name) for name in artifact_names},
                "decision": {
                    "status": "PASS",
                    "checks": checks,
                    "failed_checks": [],
                    "later_gates_authorized": ["M4_action_adoption"],
                    "long_training_authorized": False,
                },
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )
    return report, checkpoint


def test_stationary_acceptance_binds_complete_evidence_package(tmp_path: Path) -> None:
    report, checkpoint = _publish_acceptance(tmp_path)

    accepted = validate_stationary_temporal_acceptance(
        report_path=report,
        checkpoint_path=checkpoint,
    )

    assert accepted.checkpoint_sha256 == sha256_file(checkpoint)
    assert accepted.provenance == _provenance()
    assert accepted.contract_dict()["stage_authorized"] == "M4_action_adoption"


def test_stationary_candidate_metrics_require_negatives_in_every_prefix(tmp_path: Path) -> None:
    root = tmp_path / "accepted"
    root.mkdir()
    _publish_acceptance(root)
    metrics = root / "candidate_metrics.jsonl"
    records = [json.loads(line) for line in metrics.read_text().splitlines()]
    for record in records:
        if record["metrics"]["prefix_length"] == 128:
            record["metrics"]["picf_lifecycle_detection_negative_target_mass"] = 0.0
    metrics.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="every prefix bucket"):
        validate_stationary_candidate_metrics(metrics, expected_steps=200)


def test_stationary_acceptance_rejects_changed_evidence_or_failed_gate(tmp_path: Path) -> None:
    report, checkpoint = _publish_acceptance(tmp_path)
    (tmp_path / "fixed_checkpoint_replay.json").write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="absent or changed"):
        validate_stationary_temporal_acceptance(
            report_path=report,
            checkpoint_path=checkpoint,
        )

    other = tmp_path / "failed"
    other.mkdir()
    report, checkpoint = _publish_acceptance(other)
    payload = json.loads(report.read_text(encoding="ascii"))
    payload["decision"]["checks"]["visual_review_passed"] = False
    payload["decision"]["failed_checks"] = ["visual_review_passed"]
    report.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="did not pass exactly"):
        validate_stationary_temporal_acceptance(
            report_path=report,
            checkpoint_path=checkpoint,
        )


def test_stationary_acceptance_rejects_rehashed_empty_pass_evidence(tmp_path: Path) -> None:
    report, checkpoint = _publish_acceptance(tmp_path)
    lifecycle = tmp_path / "lifecycle_calibration.json"
    lifecycle.write_text('{"status":"PASS"}\n', encoding="ascii")
    payload = json.loads(report.read_text(encoding="ascii"))
    payload["artifacts_sha256"]["lifecycle_calibration.json"] = sha256_file(lifecycle)
    report.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match="differs from fixed replay"):
        validate_stationary_temporal_acceptance(
            report_path=report,
            checkpoint_path=checkpoint,
        )


def test_stationary_finalizer_publishes_one_atomic_validated_package(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _report, accepted_checkpoint = _publish_acceptance(source)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    shutil.copyfile(
        accepted_checkpoint,
        candidate / "stationary_temporal_core_candidate.pt",
    )
    shutil.copyfile(source / "candidate_metrics.jsonl", candidate / "metrics.jsonl")
    shutil.copyfile(source / "candidate_report.json", candidate / "report.json")
    persistence = tmp_path / "persistent"
    persistence.mkdir()
    output = persistence / "accepted"

    result = finalize_stationary_temporal(
        candidate_run_dir=candidate,
        replay_dir=source,
        visual_review_path=source / "visual_review.json",
        output_dir=output,
        persistent_root=persistence,
    )

    assert result["status"] == "ACCEPTED_FOR_M4_ACTION_ADOPTION"
    assert output.is_dir()
    assert not any(path.name.startswith(".accepted.incomplete") for path in persistence.iterdir())
    validate_stationary_temporal_acceptance(
        report_path=output / "report.json",
        checkpoint_path=output / "stationary_temporal_core_accepted.pt",
    )
