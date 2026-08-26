from __future__ import annotations

import json
from pathlib import Path

import pytest

from picf_next.training.counterfactual_measurement import (
    deterministic_cycle,
    deterministic_cycle_exposure_counts,
    formal_counterfactual_measurement_acceptance,
    formal_counterfactual_measurement_occam_acceptance,
    load_counterfactual_measurement_recipe,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/molmoact2_calvin_m2_counterfactual_smoke.json"
FORMAL_CONFIG = ROOT / "configs/training/molmoact2_calvin_m2_counterfactual_formal.json"
PHASE_B_CONFIG = ROOT / "configs/training/molmoact2_calvin_m2_counterfactual_phase_b.json"
STATE_COVERAGE_CONFIG = (
    ROOT / "configs/training/molmoact2_calvin_m2_counterfactual_state_coverage.json"
)


def test_counterfactual_measurement_recipe_is_hash_bound_to_m2() -> None:
    recipe = load_counterfactual_measurement_recipe(CONFIG)

    assert recipe.optimization.steps == 20
    assert recipe.optimization.pair_count_per_step == 2
    assert recipe.foundation_m2_path(ROOT).name == "molmoact2_calvin_m2_representation.json"
    assert len(recipe.recipe_sha256) == 64


def test_formal_recipe_keeps_natural_replay_dominant_and_cycles_pair_bank() -> None:
    recipe = load_counterfactual_measurement_recipe(FORMAL_CONFIG)

    assert recipe.optimization.steps == 160
    assert recipe.optimization.natural_count_per_step == 6
    assert recipe.optimization.pair_count_per_step == 1
    assert recipe.acceptance.minimum_pairs == 8
    assert recipe.recipe_sha256 == (
        "67b53e7d3cd9019c1bba9ac78d0b5fb6b4482b86b3b9a2e5ac2ec311b472b868"
    )


def test_phase_b_recipe_balances_per_item_exposure_and_uses_occam_rule() -> None:
    recipe = load_counterfactual_measurement_recipe(PHASE_B_CONFIG)

    assert recipe.optimization.steps == 320
    assert recipe.optimization.pair_count_per_step == 2
    assert recipe.optimization.natural_count_per_step == 6
    pair_exposure = deterministic_cycle_exposure_counts(
        tuple(f"pair-{index}" for index in range(32)),
        count=recipe.optimization.pair_count_per_step,
        seed=recipe.optimization.seed + 101,
        steps=recipe.optimization.steps,
    )
    natural_exposure = deterministic_cycle_exposure_counts(
        tuple(f"natural-{index}" for index in range(96)),
        count=recipe.optimization.natural_count_per_step,
        seed=recipe.optimization.seed,
        steps=recipe.optimization.steps,
    )

    assert set(pair_exposure.values()) == {20}
    assert set(natural_exposure.values()) == {20}
    assert recipe.decision_rule == "occam-complete-set-v1"


def test_state_coverage_recipe_doubles_support_without_changing_exposure_ratio() -> None:
    phase_b = load_counterfactual_measurement_recipe(PHASE_B_CONFIG)
    recipe = load_counterfactual_measurement_recipe(STATE_COVERAGE_CONFIG)

    assert recipe.optimization.steps == phase_b.optimization.steps
    assert recipe.optimization.pair_count_per_step == 4
    assert recipe.optimization.natural_count_per_step == 12
    assert recipe.optimization.natural_replay_pool_size == 192
    pair_exposure = deterministic_cycle_exposure_counts(
        tuple(f"pair-{index}" for index in range(64)),
        count=recipe.optimization.pair_count_per_step,
        seed=recipe.optimization.seed + 101,
        steps=recipe.optimization.steps,
    )
    natural_exposure = deterministic_cycle_exposure_counts(
        tuple(f"natural-{index}" for index in range(192)),
        count=recipe.optimization.natural_count_per_step,
        seed=recipe.optimization.seed,
        steps=recipe.optimization.steps,
    )

    assert set(pair_exposure.values()) == {20}
    assert set(natural_exposure.values()) == {20}
    assert recipe.optimization.natural_count_per_step == (
        3 * recipe.optimization.pair_count_per_step
    )
    assert recipe.decision_rule == phase_b.decision_rule


def test_counterfactual_recipe_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["shortcut"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unknown=.*shortcut"):
        load_counterfactual_measurement_recipe(path)


def test_deterministic_cycle_is_reproducible_and_covers_values() -> None:
    values = ("a", "b", "c")
    first = deterministic_cycle(values, count=2, seed=7, step=1)
    repeated = deterministic_cycle(values, count=2, seed=7, step=1)
    covered = {
        item
        for step in range(1, 7)
        for item in deterministic_cycle(values, count=2, seed=7, step=step)
    }

    assert first == repeated
    assert covered == set(values)


def test_deterministic_cycle_exposure_rejects_duplicate_values() -> None:
    with pytest.raises(ValueError, match="unique"):
        deterministic_cycle_exposure_counts(
            ("a", "a"),
            count=1,
            seed=7,
            steps=2,
        )


def _pair_metrics(
    *,
    removed_loss: float,
    removed_existence: float,
) -> dict[str, object]:
    pairs = {}
    for index in range(8):
        identity = "a" if index < 4 else "b"
        factual_count = 3 if identity == "a" else 4
        pairs[f"pair-{index}"] = {
            "target_identity_key": f"object/{identity}",
            "factual": {
                "target_existence": 0.9 if identity == "a" else 0.85,
                "target_soft_dice": 0.6 if identity == "a" else 0.55,
                "active_count": factual_count,
                "target_count": factual_count,
            },
            "removed": {
                "maximum_unmatched_existence": removed_existence,
                "active_count": factual_count - 1,
                "target_count": factual_count - 1,
            },
        }
    return {
        "pairs": pairs,
        "mean_removed_loss": removed_loss,
        "mean_removed_maximum_unmatched_existence": removed_existence,
    }


def test_formal_acceptance_requires_both_unseen_partitions_and_natural_replay() -> None:
    recipe = load_counterfactual_measurement_recipe(CONFIG)
    prior = {
        partition: _pair_metrics(removed_loss=2.0, removed_existence=0.9)
        for partition in ("validation", "heldout")
    }
    actual = {
        partition: _pair_metrics(removed_loss=1.0, removed_existence=0.1)
        for partition in ("validation", "heldout")
    }
    control = {
        partition: _pair_metrics(removed_loss=1.8, removed_existence=0.7)
        for partition in ("validation", "heldout")
    }
    natural_prior = {"losses": {"loss_total": 2.0}, "count_mae": 0.3}
    natural_actual = {"losses": {"loss_total": 2.1}, "count_mae": 0.3}

    passed = formal_counterfactual_measurement_acceptance(
        recipe=recipe,
        prior_pairs=prior,
        actual_pairs=actual,
        control_pairs=control,
        prior_natural=natural_prior,
        actual_natural=natural_actual,
    )

    assert passed["status"] == "PASS_COUNTERFACTUAL_MEASUREMENT"
    assert passed["later_gates_authorized"] == ["M3_stationary_temporal_revalidation"]

    actual["heldout"] = _pair_metrics(removed_loss=1.0, removed_existence=0.5)
    failed = formal_counterfactual_measurement_acceptance(
        recipe=recipe,
        prior_pairs=prior,
        actual_pairs=actual,
        control_pairs=control,
        prior_natural=natural_prior,
        actual_natural=natural_actual,
    )

    assert failed["status"] == "FAIL"
    assert "heldout_removed_targets_rejected" in failed["failed_checks"]
    assert failed["later_gates_authorized"] == []


def test_occam_acceptance_selects_sufficient_factual_baseline() -> None:
    recipe = load_counterfactual_measurement_recipe(PHASE_B_CONFIG)
    prior = {
        partition: _pair_metrics(removed_loss=2.0, removed_existence=0.9)
        for partition in ("validation", "heldout")
    }
    actual = {
        partition: _pair_metrics(removed_loss=1.0, removed_existence=0.1)
        for partition in ("validation", "heldout")
    }
    control = {
        partition: _pair_metrics(removed_loss=1.1, removed_existence=0.1)
        for partition in ("validation", "heldout")
    }
    natural_prior = {"losses": {"loss_total": 2.0}, "count_mae": 0.3}
    natural_actual = {"losses": {"loss_total": 1.9}, "count_mae": 0.2}
    natural_control = {"losses": {"loss_total": 1.8}, "count_mae": 0.2}

    decision = formal_counterfactual_measurement_occam_acceptance(
        recipe=recipe,
        prior_pairs=prior,
        actual_pairs=actual,
        control_pairs=control,
        prior_natural=natural_prior,
        actual_natural=natural_actual,
        control_natural=natural_control,
    )

    assert decision["status"] == "PASS_FACTUAL_BASELINE"
    assert decision["selected_candidate"] == "factual_only_control"
    assert decision["failed_checks"] == []


def test_occam_acceptance_selects_counterfactual_only_for_added_value() -> None:
    recipe = load_counterfactual_measurement_recipe(PHASE_B_CONFIG)
    prior = {
        partition: _pair_metrics(removed_loss=2.0, removed_existence=0.9)
        for partition in ("validation", "heldout")
    }
    actual = {
        partition: _pair_metrics(removed_loss=1.0, removed_existence=0.1)
        for partition in ("validation", "heldout")
    }
    control = {
        partition: _pair_metrics(removed_loss=1.5, removed_existence=0.7)
        for partition in ("validation", "heldout")
    }
    natural = {"losses": {"loss_total": 1.9}, "count_mae": 0.2}
    natural_prior = {"losses": {"loss_total": 2.0}, "count_mae": 0.3}

    decision = formal_counterfactual_measurement_occam_acceptance(
        recipe=recipe,
        prior_pairs=prior,
        actual_pairs=actual,
        control_pairs=control,
        prior_natural=natural_prior,
        actual_natural=natural,
        control_natural=natural,
    )

    assert decision["status"] == "PASS_COUNTERFACTUAL_MEASUREMENT"
    assert decision["selected_candidate"] == "counterfactual"
    assert decision["corrected_control_identity_count"] == 2
    assert len(decision["corrected_control_pairs"]) == 16


def test_occam_acceptance_does_not_hide_incomplete_counterfactual_sets() -> None:
    recipe = load_counterfactual_measurement_recipe(PHASE_B_CONFIG)
    prior = {
        partition: _pair_metrics(removed_loss=2.0, removed_existence=0.9)
        for partition in ("validation", "heldout")
    }
    actual = {
        partition: _pair_metrics(removed_loss=1.0, removed_existence=0.1)
        for partition in ("validation", "heldout")
    }
    actual["heldout"]["pairs"]["pair-0"]["removed"]["active_count"] = 1
    control = {
        partition: _pair_metrics(removed_loss=1.5, removed_existence=0.7)
        for partition in ("validation", "heldout")
    }
    natural = {"losses": {"loss_total": 1.9}, "count_mae": 0.2}
    natural_prior = {"losses": {"loss_total": 2.0}, "count_mae": 0.3}

    decision = formal_counterfactual_measurement_occam_acceptance(
        recipe=recipe,
        prior_pairs=prior,
        actual_pairs=actual,
        control_pairs=control,
        prior_natural=natural_prior,
        actual_natural=natural,
        control_natural=natural,
    )

    assert decision["status"] == "FAIL"
    assert decision["selected_candidate"] is None
    assert "counterfactual_heldout_removed_set_cardinality_exact" in decision["failed_checks"]


def test_occam_acceptance_requires_preregistered_partition_support() -> None:
    recipe = load_counterfactual_measurement_recipe(PHASE_B_CONFIG)
    prior = {
        partition: _pair_metrics(removed_loss=2.0, removed_existence=0.9)
        for partition in ("validation", "heldout")
    }
    actual = {
        partition: _pair_metrics(removed_loss=1.0, removed_existence=0.1)
        for partition in ("validation", "heldout")
    }
    control = {
        partition: _pair_metrics(removed_loss=1.5, removed_existence=0.7)
        for partition in ("validation", "heldout")
    }
    del prior["heldout"]["pairs"]["pair-7"]
    del actual["heldout"]["pairs"]["pair-7"]
    del control["heldout"]["pairs"]["pair-7"]
    natural = {"losses": {"loss_total": 1.9}, "count_mae": 0.2}
    natural_prior = {"losses": {"loss_total": 2.0}, "count_mae": 0.3}

    decision = formal_counterfactual_measurement_occam_acceptance(
        recipe=recipe,
        prior_pairs=prior,
        actual_pairs=actual,
        control_pairs=control,
        prior_natural=natural_prior,
        actual_natural=natural,
        control_natural=natural,
    )

    assert decision["status"] == "FAIL"
    assert "counterfactual_heldout_pair_count_sufficient" in decision["failed_checks"]


def test_occam_acceptance_does_not_call_factual_repair_counterfactual_value() -> None:
    recipe = load_counterfactual_measurement_recipe(PHASE_B_CONFIG)
    prior = {
        partition: _pair_metrics(removed_loss=2.0, removed_existence=0.9)
        for partition in ("validation", "heldout")
    }
    actual = {
        partition: _pair_metrics(removed_loss=1.0, removed_existence=0.1)
        for partition in ("validation", "heldout")
    }
    control = {
        partition: _pair_metrics(removed_loss=1.0, removed_existence=0.1)
        for partition in ("validation", "heldout")
    }
    for partition in ("validation", "heldout"):
        for row in control[partition]["pairs"].values():
            row["factual"]["target_existence"] = 0.1
    natural = {"losses": {"loss_total": 1.9}, "count_mae": 0.2}
    natural_prior = {"losses": {"loss_total": 2.0}, "count_mae": 0.3}

    decision = formal_counterfactual_measurement_occam_acceptance(
        recipe=recipe,
        prior_pairs=prior,
        actual_pairs=actual,
        control_pairs=control,
        prior_natural=natural_prior,
        actual_natural=natural,
        control_natural=natural,
    )

    assert decision["status"] == "FAIL"
    assert decision["counterfactual_added_value"] is False
