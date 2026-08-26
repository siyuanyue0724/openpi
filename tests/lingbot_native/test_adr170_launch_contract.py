from __future__ import annotations

from pathlib import Path


def test_adr170_source_aligned_smoke_is_bounded_and_persistent() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "adr170/run_ltop_g3_source_aligned_smoke_2gpu.sh").read_text(encoding="utf-8")

    assert "--nproc_per_node=2" in source
    assert "--mode smoke" in source
    assert "--phase combined" in source
    assert "--steps 8" in source
    assert "--eval-every 8" in source
    assert "--progress-every 1" in source
    assert "task_action_supervision.py" in source
    assert "source-aligned smoke requires an exact clean PICF checkout" in source
    assert '[[ "$run_root" == /mnt/*' in source
    assert "timeout --signal=TERM --kill-after=60s" in source


def test_adr170_source_aligned_trial_and_cold_gates_require_v5_supervision() -> None:
    root = Path(__file__).resolve().parents[2]
    trial = (root / "adr170/run_ltop_g3_source_aligned_trial_2gpu_256.sh").read_text(
        encoding="utf-8"
    )
    contract = (root / "adr170/source_aligned_cold_evaluation_contract.sh").read_text(
        encoding="utf-8"
    )
    action = (root / "adr170/run_ltop_g3_source_aligned_cold_action_2gpu.sh").read_text(
        encoding="utf-8"
    )
    retention = (root / "adr170/run_ltop_g3_source_aligned_retention_2gpu.sh").read_text(
        encoding="utf-8"
    )
    acceptance = (root / "adr170/run_ltop_g3_source_aligned_cold_acceptance_2gpu.sh").read_text(
        encoding="utf-8"
    )

    assert "--steps 256" in trial
    assert "--phase training" in trial
    assert "ltop_g3_source_aligned_trial_training_report.json" in trial
    assert "task_action_supervision.py" in trial
    assert "timeout --signal=TERM --kill-after=60s" in trial
    assert 'CHECKPOINT_MANIFEST_SCHEMA = "picf-next.ltop-g3-training-checkpoint.v5"' in contract
    assert 'ACTION_SUPERVISION_SCHEMA = "picf-next.task-action-supervision.v1"' in contract
    assert (
        'TASK_ADDRESS_DEPTH_SCHEMA = "picf-next.action-consumable-task-address-depth.v1"'
        in contract
    )
    assert "require_action_consumable_depth" in contract
    assert "require_picf_source_contract" in contract
    assert "immutable-source-task-action-pairs-only" in contract
    assert "adr170_resolve_source_aligned_trial_checkpoint" in action
    assert "adr170_validate_cold_report" in action
    assert "PICF_G3_EVALUATION_ACTION_INFORMATION_SET" in action
    assert "default_evaluation_timeout_seconds=3600" in action
    assert "default_evaluation_timeout_seconds=14400" in action
    assert (
        "evaluation_timeout_seconds=${evaluation_timeout_seconds:-$default_evaluation_timeout_seconds}"
        in action
    )
    assert "adr170_resolve_source_aligned_trial_checkpoint" in retention
    assert "adr170_validate_cold_report" in retention
    assert "PICF_G3_EVALUATION_ACTION_INFORMATION_SET=factual" in acceptance
    assert "PICF_G3_EVALUATION_ACTION_INFORMATION_SET=mediator-required" in acceptance
    assert "PICF_G3_EVALUATION_SCENES_PER_PARTITION:-4" in acceptance
    assert (
        acceptance.count(
            'PICF_G3_EVALUATION_SCENES_PER_PARTITION="$evaluation_scenes_per_partition"'
        )
        == 2
    )
    assert "validate_ltop_g3_mediator_trial.py" in acceptance
    assert "compose_ltop_g3_source_aligned_acceptance.py" in acceptance
    assert "ltop_g3_source_aligned_acceptance.json" in acceptance
    assert contract.count('"unobservable_source_target_address_loss": False') == 2
    assert contract.count('"disable-address-only-with-explicit-loss-side-receipt"') == 2
    assert acceptance.index("action-factual-$evaluation_scope") < acceptance.index(
        "action-mediator-required-$evaluation_scope"
    )
    assert acceptance.rindex("action-mediator-required-$evaluation_scope") < acceptance.rindex(
        "retention"
    )
