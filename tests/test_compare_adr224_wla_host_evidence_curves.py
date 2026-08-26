from __future__ import annotations

import copy

import pytest

from tools.compare_adr224_wla_host_evidence_curves import (
    FULL_ARM,
    MASKED_ARM,
    compare_heldout_snapshots,
    compare_records,
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
        "early_stop_step": 100,
        "execution_contract": {"seed": 7, "wla_host_evidence_arm": arm},
    }


def _record(*, step: int, arm: str, action_loss: float) -> dict[str, object]:
    return {
        "global_step": step,
        "wla_host_evidence_arm": arm,
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
        },
    }


def _curve(arm: str, scale: float) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (step, rank): _record(
            step=step,
            arm=arm,
            action_loss=scale * (1.0 + rank / 100.0 + step / 1000.0),
        )
        for step in range(1, 101)
        for rank in range(2)
    }


def test_matched_curve_reports_material_early_lead_without_long_train_authority() -> None:
    report = compare_records(
        full_records=_curve(FULL_ARM, 0.8),
        masked_records=_curve(MASKED_ARM, 1.0),
        windows=((1, 20), (51, 100)),
        bootstrap_replicates=100,
        minimum_relative_lead=0.05,
    )

    assert report["decision"] == "PICF_EARLY_MATERIAL_OPTIMIZATION_LEAD"
    assert report["authorizes_long_train"] is False
    assert report["overall"]["ratio_of_means"] == pytest.approx(0.8)


def test_matched_curve_rejects_randomness_drift() -> None:
    full = _curve(FULL_ARM, 0.8)
    masked = _curve(MASKED_ARM, 1.0)
    masked[(50, 1)]["flow_noise_seeds"] = [999]

    with pytest.raises(ValueError, match="flow_noise_seeds"):
        compare_records(
            full_records=full,
            masked_records=masked,
            windows=((1, 100),),
            bootstrap_replicates=10,
            minimum_relative_lead=0.05,
        )


def test_manifest_match_allows_only_registered_evidence_arm() -> None:
    full = _manifest(FULL_ARM)
    masked = _manifest(MASKED_ARM)
    expected = copy.deepcopy((full, masked))

    validate_matched_manifests(full, masked)

    assert (full, masked) == expected
    masked["execution_contract"]["seed"] = 8
    with pytest.raises(ValueError, match="beyond the evidence arm"):
        validate_matched_manifests(full, masked)


def _heldout_snapshot(scale: float, *, step: int = 100) -> dict[str, object]:
    return {
        "schema": "action/v1",
        "status": "PASS",
        "checkpoint_global_step": step,
        "architecture_identity": "native",
        "state_mode": "cold_reset",
        "implementation_sha256": "a" * 64,
        "model_family_sha256": "b" * 64,
        "lingbot_base_family_sha256": "c" * 64,
        "stream_plan_sha256": "d" * 64,
        "representation_split_sha256": "e" * 64,
        "evaluation_plan_sha256": "f" * 64,
        "evaluation_input_sha256": "0" * 64,
        "samples": [
            {
                "partition": "heldout",
                "ordinal": index,
                "task_key": "task",
                "segment_index": index,
                "source_episode_index": index // 2,
                "source_global_index": 100 + index,
                "transition_index": index,
                "sample_key": f"sample-{index}",
                "source_digest": f"source-{index}",
                "model_inputs_sha256": f"inputs-{index}",
                "native_source_rgb_sha256": f"rgb-{index}",
                "native_source_query_count": 200,
                "prior_control_chunk_count": 1,
                "action_backend": "wla_complete",
                "action_loss": scale * (1.0 + index / 10.0),
            }
            for index in range(8)
        ],
    }


def test_heldout_comparison_is_paired_and_episode_bootstrapped() -> None:
    report = compare_heldout_snapshots(
        full_snapshots={100: _heldout_snapshot(0.8)},
        masked_snapshots={100: _heldout_snapshot(1.0)},
        bootstrap_replicates=100,
    )

    summary = report["steps"][0]["overall"]
    assert summary["paired_geometric_relative_delta"] == pytest.approx(-0.2)
    assert summary["source_episode_count"] == 4


def test_heldout_comparison_reports_learning_difference_in_differences() -> None:
    report = compare_heldout_snapshots(
        full_snapshots={
            0: _heldout_snapshot(1.0, step=0),
            100: _heldout_snapshot(0.8),
        },
        masked_snapshots={
            0: _heldout_snapshot(1.0, step=0),
            100: _heldout_snapshot(0.9),
        },
        bootstrap_replicates=100,
    )

    summary = report["learning_difference_in_differences"][0]["overall"]
    assert summary["full_paired_geometric_relative_change"] == pytest.approx(-0.2)
    assert summary["masked_paired_geometric_relative_change"] == pytest.approx(-0.1)
    assert summary["learning_ratio_of_ratios_delta"] == pytest.approx(0.8 / 0.9 - 1.0)
    assert summary["full_learning_wins"] == 8


def test_heldout_comparison_rejects_backend_drift() -> None:
    full = _heldout_snapshot(0.8)
    masked = _heldout_snapshot(1.0)
    masked["samples"][0]["action_backend"] = "released_lingbot"

    with pytest.raises(ValueError, match="complete WLA action"):
        compare_heldout_snapshots(
            full_snapshots={100: full},
            masked_snapshots={100: masked},
            bootstrap_replicates=10,
        )
