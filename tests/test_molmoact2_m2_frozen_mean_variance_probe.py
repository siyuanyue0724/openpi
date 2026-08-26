from __future__ import annotations

import pytest

from tools.run_molmoact2_m2_frozen_mean_variance_probe import (
    _complete_pass_plan,
    _decision,
    _learning_rate_multiplier,
    _representation_metric_drift,
)


def _metrics(spearman: float | None, calibration_loss: float) -> dict:
    return {
        "uncertainty_error_spearman": spearman,
        "losses": {"loss_geometry_calibration": calibration_loss},
    }


def test_complete_pass_plan_visits_every_key_once_per_pass() -> None:
    keys = [f"sample-{index}" for index in range(11)]
    plan = _complete_pass_plan(keys, batch_size=4, seed=17, passes=2)

    assert len(plan) == 6
    assert sorted(key for batch in plan[:3] for key in batch) == sorted(keys)
    assert sorted(key for batch in plan[3:] for key in batch) == sorted(keys)
    assert plan == _complete_pass_plan(keys, batch_size=4, seed=17, passes=2)


def test_learning_rate_multiplier_has_exact_warmup_and_cosine_endpoints() -> None:
    assert _learning_rate_multiplier(
        1,
        total_steps=100,
        warmup_steps=10,
        final_multiplier=0.1,
    ) == pytest.approx(0.1)
    assert _learning_rate_multiplier(
        10,
        total_steps=100,
        warmup_steps=10,
        final_multiplier=0.1,
    ) == pytest.approx(1.0)
    assert _learning_rate_multiplier(
        100,
        total_steps=100,
        warmup_steps=10,
        final_multiplier=0.1,
    ) == pytest.approx(0.1)


def test_representation_metric_drift_uses_every_preregistered_field() -> None:
    reference = {
        "count_mae": 0.1,
        "exact_count_accuracy": 0.2,
        "ownership_accuracy": 0.3,
        "token_ownership_accuracy": 0.4,
        "context_accuracy": 0.5,
        "mean_object_dice": 0.6,
        "geometry_mae_model_chart": 0.7,
        "geometry_mae_physical": 0.8,
        "fragmentation_excess_per_object": 0.9,
        "maximum_active_query_pair_dice": 1.0,
    }
    candidate = dict(reference)
    candidate["mean_object_dice"] += 0.01

    drift = _representation_metric_drift(reference, candidate)

    assert len(drift) == len(reference)
    assert drift["mean_object_dice"] == pytest.approx(0.01)
    assert max(value for name, value in drift.items() if name != "mean_object_dice") == 0.0


def test_decision_supports_two_timescales_only_with_alignment_and_isolation() -> None:
    baseline = {split: _metrics(None, 0.6) for split in ("train", "validation", "heldout")}
    aligned = {
        "train": _metrics(0.5, 0.2),
        "validation": _metrics(0.4, 0.2),
        "heldout": _metrics(0.3, 0.2),
    }
    acceptance = {
        "minimum_uncertainty_error_spearman": 0.0,
        "minimum_aligned_control_heldout_spearman_margin": 0.15,
        "maximum_representation_metric_absolute_drift": 1e-6,
    }

    decision = _decision(
        baseline_reset=baseline,
        aligned=aligned,
        control_heldout=_metrics(0.0, 0.4),
        frozen_state_exact=True,
        representation_metric_maximum_drift=0.0,
        acceptance=acceptance,
    )

    assert decision["status"] == "SUPPORTS_TWO_TIMESCALE"
    assert decision["failed_checks"] == []
    assert decision["later_gates_authorized"] == []
    assert decision["production_training_changes_authorized"] == []


def test_decision_rejects_a_rank_gain_shared_by_deranged_targets() -> None:
    baseline = {split: _metrics(None, 0.6) for split in ("train", "validation", "heldout")}
    aligned = {split: _metrics(0.3, 0.2) for split in ("train", "validation", "heldout")}
    acceptance = {
        "minimum_uncertainty_error_spearman": 0.0,
        "minimum_aligned_control_heldout_spearman_margin": 0.15,
        "maximum_representation_metric_absolute_drift": 1e-6,
    }

    decision = _decision(
        baseline_reset=baseline,
        aligned=aligned,
        control_heldout=_metrics(0.25, 0.3),
        frozen_state_exact=True,
        representation_metric_maximum_drift=0.0,
        acceptance=acceptance,
    )

    assert decision["status"] == "DOES_NOT_SUPPORT_TWO_TIMESCALE"
    assert decision["failed_checks"] == ["aligned_beats_deranged_target_control"]


def test_decision_treats_constant_control_as_zero_ranking_capability() -> None:
    baseline = {split: _metrics(None, 0.6) for split in ("train", "validation", "heldout")}
    aligned = {split: _metrics(0.3, 0.2) for split in ("train", "validation", "heldout")}
    acceptance = {
        "minimum_uncertainty_error_spearman": 0.0,
        "minimum_aligned_control_heldout_spearman_margin": 0.15,
        "maximum_representation_metric_absolute_drift": 1e-6,
    }

    decision = _decision(
        baseline_reset=baseline,
        aligned=aligned,
        control_heldout=_metrics(None, 0.3),
        frozen_state_exact=True,
        representation_metric_maximum_drift=0.0,
        acceptance=acceptance,
    )

    assert decision["status"] == "SUPPORTS_TWO_TIMESCALE"
    assert decision["control_heldout_effective_rank"] == 0.0
    assert decision["control_undefined_rank_semantics"] == "zero-ranking-capability"
