from __future__ import annotations

import copy

import pytest

from picf_next.lingbot_native.representation_cluster_statistics import (
    REPRESENTATION_CLUSTER_METRICS,
    build_representation_cluster_curve,
    build_representation_cluster_thresholds,
    validate_representation_cluster_curve,
    validate_representation_cluster_thresholds,
)
from picf_next.lingbot_native.representation_factor_oracle import (
    build_representation_factor_oracle,
)
from tests.lingbot_native.test_representation_factor_oracle import _snapshot


def _bundle(
    *,
    step: int,
    task_logits: tuple[float, float],
    ownership_blend: float,
    omit_target_rows: bool = False,
):
    plan, snapshot = _snapshot(
        checkpoint_global_step=step,
        task_logits=task_logits,
        stratified_target_area=True,
        ownership_blend=ownership_blend,
        omit_target_rows=omit_target_rows,
    )
    factor = build_representation_factor_oracle(
        snapshot,
        plan=plan,
        partition="heldout",
    )
    return plan, snapshot, factor


def test_cluster_thresholds_are_deterministic_and_recomputed() -> None:
    plan, baseline, baseline_factor = _bundle(
        step=0,
        task_logits=(0.0, 0.0),
        ownership_blend=0.0,
    )
    _, learned, learned_factor = _bundle(
        step=200,
        task_logits=(-1.0, 1.0),
        ownership_blend=0.5,
    )
    sources = {
        "historical_step_0": (baseline, baseline_factor),
        "historical_step_200": (learned, learned_factor),
    }
    first = build_representation_cluster_thresholds(
        sources,
        plan=plan,
        replicates=200,
        bootstrap_seed=17,
    )
    second = build_representation_cluster_thresholds(
        sources,
        plan=plan,
        replicates=200,
        bootstrap_seed=17,
    )

    assert first == second
    assert set(first["materiality_thresholds"]) == set(REPRESENTATION_CLUSTER_METRICS)
    assert first["is_run_to_run_variance_estimate"] is False
    assert (
        validate_representation_cluster_thresholds(
            first,
            sources=sources,
            plan=plan,
        )
        == first
    )


def test_cluster_curve_is_paired_and_reports_E_minus_M_progress() -> None:
    plan, baseline, baseline_factor = _bundle(
        step=0,
        task_logits=(0.0, 0.0),
        ownership_blend=0.0,
    )
    _, historical, historical_factor = _bundle(
        step=200,
        task_logits=(-0.5, 0.5),
        ownership_blend=0.25,
    )
    thresholds = build_representation_cluster_thresholds(
        {
            "historical_step_0": (baseline, baseline_factor),
            "historical_step_200": (historical, historical_factor),
        },
        plan=plan,
        replicates=200,
        bootstrap_seed=19,
    )
    _, arm_m, factor_m = _bundle(
        step=200,
        task_logits=(-1.0, 1.0),
        ownership_blend=0.5,
    )
    _, arm_e, factor_e = _bundle(
        step=200,
        task_logits=(-3.0, 3.0),
        ownership_blend=0.9,
    )
    arms = {
        "M": {0: (baseline, baseline_factor), 200: (arm_m, factor_m)},
        "E": {0: (baseline, baseline_factor), 200: (arm_e, factor_e)},
    }
    curve = build_representation_cluster_curve(
        arms,
        thresholds,
        plan=plan,
    )

    target_brier = curve["metrics"]["factor_target_joint_brier"]
    assert target_brier["arm_progress"]["M"]["200"]["estimate"] > 0.0
    assert target_brier["arm_progress"]["E"]["200"]["estimate"] > 0.0
    assert target_brier["E_minus_M_progress"]["200"]["estimate"] > 0.0
    assert curve["authorizes_model_adoption"] is False
    assert (
        validate_representation_cluster_curve(
            curve,
            arms=arms,
            thresholds=thresholds,
            plan=plan,
        )
        == curve
    )


def test_cluster_curve_rejects_changed_metric_eligibility() -> None:
    plan, baseline, baseline_factor = _bundle(
        step=0,
        task_logits=(0.0, 0.0),
        ownership_blend=0.0,
    )
    _, learned, learned_factor = _bundle(
        step=200,
        task_logits=(-1.0, 1.0),
        ownership_blend=0.5,
    )
    thresholds = build_representation_cluster_thresholds(
        {
            "historical_step_0": (baseline, baseline_factor),
            "historical_step_200": (learned, learned_factor),
        },
        plan=plan,
        replicates=100,
        bootstrap_seed=23,
    )
    _, missing, missing_factor = _bundle(
        step=200,
        task_logits=(-2.0, 2.0),
        ownership_blend=0.75,
        omit_target_rows=True,
    )
    arms = {
        "M": {0: (baseline, baseline_factor), 200: (learned, learned_factor)},
        "E": {0: (baseline, baseline_factor), 200: (missing, missing_factor)},
    }
    with pytest.raises(ValueError, match="eligibility changed"):
        build_representation_cluster_curve(arms, thresholds, plan=plan)


def test_cluster_threshold_hash_tampering_is_rejected() -> None:
    plan, baseline, baseline_factor = _bundle(
        step=0,
        task_logits=(0.0, 0.0),
        ownership_blend=0.0,
    )
    _, learned, learned_factor = _bundle(
        step=200,
        task_logits=(-1.0, 1.0),
        ownership_blend=0.5,
    )
    thresholds = build_representation_cluster_thresholds(
        {
            "historical_step_0": (baseline, baseline_factor),
            "historical_step_200": (learned, learned_factor),
        },
        plan=plan,
        replicates=100,
        bootstrap_seed=29,
    )
    tampered = copy.deepcopy(thresholds)
    tampered["materiality_thresholds"]["token_auc"] = 1.0
    with pytest.raises(ValueError, match="artifact changed"):
        validate_representation_cluster_thresholds(tampered)
