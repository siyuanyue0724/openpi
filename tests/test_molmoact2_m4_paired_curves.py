from __future__ import annotations

import pytest

from tools.audit_molmoact2_m4_paired_curves import (
    _intervention_summary,
    _moving_block_mean_interval,
    _paired_summary,
    _task_segments,
    _validated_curve,
)


def _metric_row(step: int, loss: float) -> dict[str, object]:
    return {
        "attempted_optimizer_steps": step,
        "metrics": {
            "action_flow_loss": loss,
            "loss": loss,
            "picf_loss_action": loss,
            "system_optimizer_loss": loss,
        },
        "optimizer_step_skipped": False,
        "successful_optimizer_steps": step,
    }


def test_validated_curve_rejects_skips_and_alias_drift() -> None:
    assert _validated_curve([_metric_row(1, 0.5)], arm="A") == [0.5]

    skipped = _metric_row(1, 0.5)
    skipped["optimizer_step_skipped"] = True
    with pytest.raises(ValueError, match="skipped"):
        _validated_curve([skipped], arm="A")

    drift = _metric_row(1, 0.5)
    drift["metrics"]["picf_loss_action"] = 0.4
    with pytest.raises(ValueError, match="aliases"):
        _validated_curve([drift], arm="C")


def test_moving_block_bootstrap_is_deterministic_and_temporally_blocked() -> None:
    values = [0.01] * 20 + [-0.01] * 20
    first = _moving_block_mean_interval(values, block_steps=10, replicates=1_000, seed=7)
    second = _moving_block_mean_interval(values, block_steps=10, replicates=1_000, seed=7)

    assert first == second
    assert first["lower_95"] < 0.0 < first["upper_95"]
    assert first["interpretation"].endswith("not_generalization_ci")


def test_paired_summary_retains_every_window_direction() -> None:
    arm_a = [1.0, 1.0, 1.0, 1.0]
    arm_c = [0.9, 1.1, 0.8, 1.2]

    report = _paired_summary(
        arm_a,
        arm_c,
        window_steps=2,
        block_steps=2,
        bootstrap_replicates=1_000,
        bootstrap_seed=3,
    )

    assert report["arm_c_wins"] == 2
    assert report["arm_c_losses"] == 2
    assert report["exact_ties"] == 0
    assert len(report["windows"]) == 2
    assert report["windows"][0]["start_step_one_based"] == 1
    assert report["windows"][1]["stop_step_one_based_inclusive"] == 4


def test_task_segments_split_only_at_episode_identity_boundaries() -> None:
    contexts = [
        {
            "lanes": [
                {
                    "episode_instance_id": "episode-a",
                    "lane_id": "lane-0",
                    "task": "task-a",
                    "task_index": 0,
                    "task_key": "a",
                    "transition_index": step,
                }
            ],
            "optimizer_step_one_based": step + 1,
        }
        for step in range(2)
    ]
    contexts.append(
        {
            "lanes": [
                {
                    "episode_instance_id": "episode-b",
                    "lane_id": "lane-0",
                    "task": "task-b",
                    "task_index": 1,
                    "task_key": "b",
                    "transition_index": 0,
                }
            ],
            "optimizer_step_one_based": 3,
        }
    )

    segments = _task_segments(contexts, [-0.1, 0.2, 0.3])

    assert len(segments) == 2
    assert segments[0]["start_step_one_based"] == 1
    assert segments[0]["stop_step_one_based_inclusive"] == 2
    assert segments[0]["mean_loss_delta_c_minus_a"] == pytest.approx(0.05)
    assert segments[1]["lanes"][0]["task"] == "task-b"


def test_intervention_summary_keeps_missing_directions_as_failed_gates() -> None:
    rank_count = 2

    def condition(delta: float, positive_ranks: int) -> dict[str, float | int]:
        return {
            "action_loss": 0.2 + delta,
            "loss_delta_from_baseline": delta,
            "positive_loss_delta_ranks": positive_ranks,
            "velocity_rms_from_baseline": abs(delta),
        }

    report = {
        "aggregate": {
            "all_baseline_replays_exact": True,
            "conditions": {
                "baseline": condition(0.0, 0),
                "joint_row_permutation": condition(0.0, 0),
                "remove_max_prior_row": condition(0.01, 2),
                "stale_previous_frame": condition(0.01, 1),
                "without_posterior": condition(-0.01, 1),
                "wrong_address": condition(0.01, 2),
            },
            "maximum_causal_velocity_rms": 0.01,
            "rank_count": rank_count,
        },
        "checkpoint": {"completed_optimizer_steps": 200},
        "gates": {},
        "plan": {"checkpoint_plan_sha256": "p" * 64},
        "recipe_sha256": "r" * 64,
        "schema": "picf-next.m4-action-intervention-audit.v1",
    }
    contract = {
        "plan": {"plan_sha256": "p" * 64},
        "recipe_sha256": "r" * 64,
    }

    _, checks = _intervention_summary(report, paired_contract=contract, completed_steps=200)

    assert checks["correct_beats_wrong_address_all_ranks"] is True
    assert checks["correct_beats_removed_max_prior_all_ranks"] is True
    assert checks["correct_beats_stale_previous_all_ranks"] is False
    assert checks["correct_beats_no_posterior_all_ranks"] is False
    assert checks["oracle_intervention_present"] is False
