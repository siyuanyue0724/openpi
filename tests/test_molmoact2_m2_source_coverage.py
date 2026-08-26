from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("olmo.hf_model.modeling_molmoact2")

from picf_next.training.molmoact2_m2_source_coverage import (
    M2_SOURCE_COVERAGE_GATE,
    load_molmoact2_m2_source_coverage_recipe,
    m2_source_coverage_report,
)
from tools.audit_molmoact2_m2_source_coverage_external import (
    _external_acceptance,
    _external_rows,
)
from tools.finalize_molmoact2_m2_source_coverage import (
    finalize_external_visuals,
    finalize_training_visuals,
    validate_source_coverage_visual_review,
)
from tools.run_molmoact2_m2_cloud import _canonical_sha256, _sha256

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs/training/molmoact2_calvin_m2_source_coverage.json"


def _visual_review(run_dir: Path, *, stage: str) -> dict:
    visual_dir = run_dir / "visuals"
    visual_dir.mkdir(parents=True)
    image = visual_dir / "sample.png"
    image.write_bytes(b"png")
    artifacts = [
        {
            "path": "visuals/sample.png",
            "sha256": _sha256(image),
        }
    ]
    visual_manifest = {
        "schema": "picf-next.molmoact2-m2-visual-artifacts.v1",
        "gate": M2_SOURCE_COVERAGE_GATE,
        "artifacts": artifacts,
        "artifacts_sha256": _canonical_sha256(artifacts),
        "all_splits_present": True,
        "all_learned_segments_present": True,
        "camera_views_per_artifact": 2,
    }
    (run_dir / "visual_artifacts.json").write_text(json.dumps(visual_manifest))
    (run_dir / "machine_decision.json").write_text("{}")
    return {
        "schema": "picf-next.molmoact2-m2-source-coverage-visual-review.v1",
        "stage": stage,
        "status": "PASS",
        "gate": M2_SOURCE_COVERAGE_GATE,
        "run_dir": str(run_dir.resolve()),
        "machine_decision_sha256": _sha256(run_dir / "machine_decision.json"),
        "visual_artifacts_sha256": _sha256(run_dir / "visual_artifacts.json"),
        "inspected_files": ["visuals/sample.png"],
        "reviewer": "test reviewer",
        "findings": ["Object ownership and both cameras were inspected."],
        "physical_object_ownership_accepted": True,
        "multi_camera_accepted": True,
        "occlusion_cases_accepted": True,
        "fragmentation_accepted": True,
    }


def test_source_coverage_recipe_is_exact_and_declares_every_pipeline_change() -> None:
    recipe = load_molmoact2_m2_source_coverage_recipe(_CONFIG)
    report = m2_source_coverage_report(recipe, repository_root=_ROOT)

    assert recipe.gate == M2_SOURCE_COVERAGE_GATE
    assert report["split_frame_counts"] == {
        "train": 1800,
        "validation": 400,
        "heldout": 391,
    }
    assert report["guard_frame_count"] == 180
    assert report["target_probe_frame_count"] == 2771
    assert report["external_validation"]["frame_count"] == 1675
    assert report["external_validation"]["checkpoint_reselection_authorized"] is False
    assert report["external_validation"]["threshold_reselection_authorized"] is False
    assert report["external_validation"]["thresholds"] == {
        "minimum_count_mae_improvement_fraction_vs_random": 0.1,
        "minimum_exact_count_accuracy": 0.25,
        "minimum_geometry_mae_improvement_fraction_vs_random": 0.1,
        "minimum_mean_object_dice": 0.35,
        "minimum_ownership_accuracy": 0.6,
        "minimum_ownership_accuracy_improvement_vs_all_context": 0.05,
        "minimum_random_dice_margin": 0.05,
    }
    assert report["candidate_under_test"] == (
        "task-independent all-source coverage with corrected token-measurable v3 targets"
    )
    assert report["single_variable_source_coverage_attribution_authorized"] is False
    assert len(report["declared_differences_vs_historical_sparse_m2"]) == 4
    assert report["unchanged_trainable_runtime_modules"] == ["projector", "discovery"]
    assert report["long_training_authorized"] is False
    assert recipe.split.split_name(358482) == "train"
    assert recipe.split.split_name(360500) == "validation"
    assert recipe.split.split_name(361252) == "heldout"
    with pytest.raises(KeyError):
        recipe.split.split_name(360300)


def test_source_coverage_recipe_rejects_unaccounted_or_short_guard(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG.read_text())
    payload["split"]["guard_ranges"][0][1] -= 1
    path = tmp_path / "gap.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="partition"):
        load_molmoact2_m2_source_coverage_recipe(path)

    payload = json.loads(_CONFIG.read_text())
    payload["split"]["minimum_guard_frames"] = 91
    path = tmp_path / "short.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="shorter"):
        load_molmoact2_m2_source_coverage_recipe(path)

    payload = json.loads(_CONFIG.read_text())
    payload["split"]["train_ranges"][0][1] = payload["split"]["guard_ranges"][0][1]
    payload["split"]["guard_ranges"].pop(0)
    path = tmp_path / "no-separating-guard.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="separated"):
        load_molmoact2_m2_source_coverage_recipe(path)


def test_external_rows_cover_every_frame_once_in_deterministic_blocks() -> None:
    rows = _external_rows(1000, 1251, block_frames=100)

    assert len(rows) == 251
    assert rows[0] == (1000, "external_validation", 0)
    assert rows[99] == (1099, "external_validation", 0)
    assert rows[100] == (1100, "external_validation", 1)
    assert rows[-1] == (1250, "external_validation", 2)


def test_external_acceptance_reuses_base_thresholds_without_reselection() -> None:
    recipe = load_molmoact2_m2_source_coverage_recipe(_CONFIG).load_base_m2(_ROOT)
    actual = {
        "exact_count_accuracy": 0.25,
        "mean_object_dice": 0.36,
        "ownership_accuracy": 0.61,
        "count_mae": 0.89,
        "geometry_mae_physical": 0.89,
        "nonfinite_metric_count": 0,
    }
    random = {
        "mean_object_dice": 0.30,
        "count_mae": 1.0,
        "geometry_mae_physical": 1.0,
    }
    all_context = {"ownership_accuracy": 0.55}
    intervention = {
        "all_dense_features_exact": True,
        "maximum_absolute_error": 0.0,
    }

    accepted = _external_acceptance(
        recipe=recipe,
        actual=actual,
        random=random,
        all_context=all_context,
        intervention=intervention,
    )
    assert accepted["status"] == "PASS_PENDING_VISUAL_REVIEW"
    assert accepted["failed_checks"] == []

    actual["exact_count_accuracy"] = 0.24
    rejected = _external_acceptance(
        recipe=recipe,
        actual=actual,
        random=random,
        all_context=all_context,
        intervention=intervention,
    )
    assert rejected["status"] == "FAIL"
    assert rejected["failed_checks"] == ["external_exact_count_accuracy"]


def test_source_visual_review_cannot_skip_an_artifact(tmp_path: Path) -> None:
    review = _visual_review(tmp_path, stage="training")
    validate_source_coverage_visual_review(
        review,
        run_dir=tmp_path.resolve(),
        stage="training",
    )

    review["inspected_files"] = []
    with pytest.raises(ValueError, match="every artifact"):
        validate_source_coverage_visual_review(
            review,
            run_dir=tmp_path.resolve(),
            stage="training",
        )


def test_source_external_protocol_requires_training_visuals_and_one_bound_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_run = tmp_path / "training"
    training_run.mkdir()
    training_review = _visual_review(training_run, stage="training")
    training_review_path = tmp_path / "training-review.json"
    training_review_path.write_text(json.dumps(training_review))

    from tools import finalize_molmoact2_m2_source_coverage as finalizer

    monkeypatch.setattr(finalizer, "_is_under_mnt", lambda _path: True)
    monkeypatch.setattr(
        finalizer.source_runner,
        "validate_source_coverage_machine_decision",
        lambda _path: {"status": "PASS_PENDING_VISUAL_REVIEW"},
    )
    training_decision = finalize_training_visuals(
        run_dir=training_run,
        visual_review_path=training_review_path,
    )
    assert training_decision["external_validation_authorized"] is True

    external_run = training_run / "external_validation"
    external_run.mkdir()
    external_review = _visual_review(external_run, stage="external")
    external_review_path = tmp_path / "external-review.json"
    external_review_path.write_text(json.dumps(external_review))
    monkeypatch.setattr(
        finalizer.source_runner,
        "validate_source_coverage_training_visual_decision",
        lambda _path: {"status": "PASS"},
    )
    monkeypatch.setattr(
        finalizer.external,
        "validate_external_machine_decision",
        lambda _path, training_run: {"status": "PASS_PENDING_VISUAL_REVIEW"},
    )
    final = finalize_external_visuals(
        training_run=training_run,
        external_run=external_run,
        visual_review_path=external_review_path,
    )
    assert final["status"] == "PASS"
    assert final["later_gates_authorized"] == ["M3_structural_probe"]

    wrong_external = tmp_path / "other-external"
    wrong_external.mkdir()
    with pytest.raises(ValueError, match="uniquely bound"):
        finalize_external_visuals(
            training_run=training_run,
            external_run=wrong_external,
            visual_review_path=external_review_path,
        )
