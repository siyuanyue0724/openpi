from __future__ import annotations

import copy

import pytest

from tools.compare_adr224_source_update_curves import (
    FROZEN_ARM,
    JOINT_ARM,
    _decision,
    _step_one_equivalence,
    compare_training_records,
    validate_completed_run,
    validate_matched_manifests,
)


def _manifest(arm: str) -> dict[str, object]:
    return {
        "status": "PASS",
        "schema": "run/v1",
        "world_size": 2,
        "global_batch_size": 2,
        "implementation_sha256": "a" * 64,
        "model_family_sha256": "b" * 64,
        "stream_plan_sha256": "c" * 64,
        "representation_split_artifact_sha256": "d" * 64,
        "evaluation_plan_artifact_sha256": "e" * 64,
        "parameter_manifest": {"count": 10},
        "objective": {"name": "action+world"},
        "trainable_scope": "full-host",
        "action_fsdp2_topology": {"blocks": 28},
        "vlm_fsdp2_topology": {"blocks": 36},
        "early_stop_step": 20,
        "execution_contract": {
            "seed": 7,
            "wla_host_evidence_arm": "picf_full",
            "videomt_stage_pq": {
                "source_update_arm": arm,
                "source_forward_and_backward_graph": "unchanged_complete_joint_graph",
            },
        },
    }


def _record(*, step: int, arm: str, action_loss: float) -> dict[str, object]:
    return {
        "global_step": step,
        "wla_host_evidence_arm": "picf_full",
        "sample_keys": [f"sample-{step}"],
        "source_digest": f"source-{step}",
        "temporal_plan_sha256": f"temporal-{step}",
        "augmentation_seeds": [step],
        "flow_noise_seeds": [step + 1],
        "flow_timestep_seeds": [step + 2],
        "frame_indices": [step - 1],
        "lane_ids": [0],
        "reset": [step == 1],
        "local_bptt_steps": 1,
        "optimizer_lags": [0],
        "official_action_loss": action_loss,
        "official_policy_loss": action_loss + 0.1,
        "objective_total": action_loss + 1.0,
        "training_objective_total": action_loss + 1.0,
        "posterior_bank_sha256": f"posterior-{step}",
        "gradient_metrics": {
            "source_update_arm": arm,
            "source_update_applied": arm == JOINT_ARM,
            "source_scheduler_step": step if arm == JOINT_ARM else 0,
        },
        "wla_action_world": {
            "metrics": {"loss_action": action_loss, "loss_world": 0.5},
            "target_source_global_indices": [step + 8],
            "target_source_rgb_sha256": [f"rgb-{step}"],
            "world_loss_weight": 0.1,
            "optimizer_contract": {"name": "AdamW"},
        },
        "videomt_source_objective": {
            "global_indices": [[step, step + 1]],
            "query_count": 200,
            "total": 1.0,
        },
    }


def _curve(arm: str, scale_after_step_one: float) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (step, rank): _record(
            step=step,
            arm=arm,
            action_loss=(
                1.0 + rank / 100.0
                if step == 1
                else scale_after_step_one * (1.0 + rank / 100.0 + step / 1000.0)
            ),
        )
        for step in range(1, 21)
        for rank in range(2)
    }


def test_source_update_manifest_match_allows_only_registered_intervention() -> None:
    frozen = _manifest(FROZEN_ARM)
    joint = _manifest(JOINT_ARM)
    expected = copy.deepcopy((frozen, joint))

    validate_matched_manifests(frozen, joint)

    assert (frozen, joint) == expected
    joint["execution_contract"]["seed"] = 8
    with pytest.raises(ValueError, match="beyond the source-update arm"):
        validate_matched_manifests(frozen, joint)


def test_source_update_manifest_accepts_immutable_declarations() -> None:
    frozen = _manifest(FROZEN_ARM)
    joint = _manifest(JOINT_ARM)
    frozen["status"] = joint["status"] = "DECLARED"

    validate_matched_manifests(frozen, joint)


def test_source_update_completion_requires_matching_early_stop_summary(tmp_path) -> None:
    manifest = _manifest(FROZEN_ARM)
    manifest.update(
        {
            "schema": "run/v19",
            "status": "DECLARED",
            "declared_total_steps": 30_000,
        }
    )
    summary = {
        "schema": "run/v19",
        "status": "EARLY_STOP",
        "completed_global_step": 20,
        "declared_total_steps": 30_000,
    }
    (tmp_path / "run_summary_step_00000020.json").write_text(
        __import__("json").dumps(summary), encoding="ascii"
    )

    receipt = validate_completed_run(tmp_path, manifest)

    assert receipt["status"] == "EARLY_STOP"
    assert receipt["completed_global_step"] == 20


def test_source_update_completion_fails_closed_on_partial_summary(tmp_path) -> None:
    manifest = _manifest(FROZEN_ARM)
    manifest.update(
        {
            "schema": "run/v19",
            "status": "DECLARED",
            "declared_total_steps": 30_000,
        }
    )
    (tmp_path / "run_summary_step_00000020.json").write_text(
        '{"schema":"run/v19","status":"EARLY_STOP",'
        '"completed_global_step":19,"declared_total_steps":30000}',
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="completed_global_step"):
        validate_completed_run(tmp_path, manifest)


def test_source_update_curve_requires_exact_step_one_and_tracks_frozen_candidate() -> None:
    report = compare_training_records(
        frozen_records=_curve(FROZEN_ARM, 0.8),
        joint_records=_curve(JOINT_ARM, 1.0),
        windows=((1, 5), (16, 20)),
        bootstrap_replicates=100,
    )

    assert report["step_one_forward_equivalence"]["exact"] is True
    assert report["overall"]["paired_geometric_relative_delta"] < -0.15


def test_step_one_accepts_only_float32_scale_roundoff() -> None:
    frozen = _curve(FROZEN_ARM, 1.0)
    joint = _curve(JOINT_ARM, 1.0)
    frozen[(1, 0)]["videomt_source_objective"]["total"] += 2.0**-23

    report = _step_one_equivalence(frozen, joint)

    assert report["exact"] is False
    assert report["float32_numerically_equivalent"] is True
    frozen[(1, 0)]["videomt_source_objective"]["total"] += 1.0e-4
    report = _step_one_equivalence(frozen, joint)
    assert report["float32_numerically_equivalent"] is False


def test_source_update_curve_rejects_a_frozen_scheduler_step() -> None:
    frozen = _curve(FROZEN_ARM, 0.8)
    joint = _curve(JOINT_ARM, 1.0)
    frozen[(2, 0)]["gradient_metrics"]["source_scheduler_step"] = 1

    with pytest.raises(ValueError, match="applied a source update"):
        compare_training_records(
            frozen_records=frozen,
            joint_records=joint,
            windows=((1, 20),),
            bootstrap_replicates=10,
        )


def test_source_update_decision_rejects_any_registered_window_regression() -> None:
    training = {
        "step_one_forward_equivalence": {"exact": True},
        "overall": {"paired_geometric_relative_delta": -0.03},
        "windows": [
            {"paired_geometric_relative_delta": -0.04},
            {"paired_geometric_relative_delta": 0.021},
        ],
    }
    heldout = {
        "learning_difference_in_differences": [
            {
                "overall": {
                    "learning_ratio_of_ratios_delta": -0.03,
                    "learning_ratio_of_ratios_delta_episode_bootstrap_95": [
                        -0.04,
                        -0.02,
                    ],
                }
            }
        ]
    }

    assert _decision(training, heldout) == (
        "REJECTS_COORDINATE_MOTION_AS_PRIMARY_EARLY_ROOT_CAUSE"
    )
