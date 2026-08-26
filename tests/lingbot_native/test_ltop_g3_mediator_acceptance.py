from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from picf_next.artifact_io import directory_tree_sha256
from picf_next.lingbot_native.ltop_g3_mediator_acceptance import (
    ACTION_INFORMATION_SET_POLICY,
    ARM_VALIDATION_SCHEMA,
    MEDIATOR_ACCEPTANCE_SCHEMA,
    MediatorAcceptanceContractError,
    compose_ltop_g3_mediator_acceptance,
)
from tools.compose_ltop_g3_mediator_acceptance import main

MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"
TRAINING_CHECKPOINT_SCHEMA = "picf-next.ltop-g3-training-checkpoint.v2"
MODEL_ONLY_CHECKPOINT_FORMAT = "lingbot-fsdp2-dcp-model-only"
TRAINING_SEED = 20260813
TRAINING_MODEL_DIGESTS = ["c" * 64, "d" * 64]
RUNTIME_SCHEDULE_DIGESTS = ["8" * 64, "9" * 64]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")


def _identity(tmp_path: Path) -> dict[str, object]:
    return {
        "architecture_identity": "lingbot_task_query_object_value_read_v1",
        "runtime_source_contract": {
            "native_patch_sha256": "1" * 64,
            "runtime_hotfix_sha256": "2" * 64,
        },
        "stage_checkpoint": str((tmp_path / "g2-checkpoint").resolve()),
        "g2_report_sha256": "3" * 64,
        "capacity": 16,
        "task_query_count": 4,
        "execution_contract_sha256": "4" * 64,
        "offline_labels_sha256": "5" * 64,
        "physical_sidecar_manifest_sha256": "6" * 64,
        "dataset_contract": {
            "dataset_tree_sha256": "7" * 64,
            "dataset_file_count": 17,
        },
    }


def _schedule() -> dict[str, object]:
    entries = []
    for step in range(1, 257):
        cycle_index, cycle_offset = divmod(step - 1, 16)
        scene_index, prompt_index = divmod(cycle_offset, 2)
        required = bool((scene_index + prompt_index + cycle_index) % 2)
        entries.append(
            {
                "global_step": step,
                "cycle_index": cycle_index,
                "cycle_offset": cycle_offset,
                "scene_index": scene_index,
                "scene_key": f"scene-{scene_index}",
                "prompt_index": prompt_index,
                "prompt_key": f"prompt-{prompt_index}",
                "arm": "mediator-required" if required else "factual",
            }
        )
    schedule: dict[str, object] = {
        "schema": "picf-next.ltop-g3-mediator-trial-schedule.v1",
        "design": "scene-prompt-stratified-two-period-crossover",
        "single_forward_per_optimizer_step": True,
        "assignment": "(scene_index + prompt_index + cycle_index) mod 2",
        "steps": 256,
        "scene_count": 8,
        "prompts_per_scene": 2,
        "cycle_steps": 16,
        "arm_counts": {"factual": 128, "mediator-required": 128},
        "cell_arm_counts": {
            f"scene-{scene}::prompt-{prompt}": {
                "factual": 8,
                "mediator-required": 8,
            }
            for scene in range(8)
            for prompt in range(2)
        },
        "entries": entries,
    }
    schedule["sha256"] = hashlib.sha256(_canonical_json(schedule).encode("ascii")).hexdigest()
    return schedule


def _training_report(tmp_path: Path, identity: dict[str, object]) -> dict[str, object]:
    schedule = _schedule()
    journal_dir = (tmp_path / "rank_journal").resolve()
    checkpoint_path = (tmp_path / "checkpoint-model-only").resolve()
    model_path = checkpoint_path / "model"
    model_path.mkdir(parents=True)
    (model_path / ".metadata").write_bytes(b"minimal DCP metadata\n")
    (model_path / "__0_0.distcp").write_bytes(b"minimal DCP model shard\n")
    model_tree_sha256 = directory_tree_sha256(
        model_path,
        schema=MODEL_TREE_SCHEMA,
    )
    counts = {"factual": 128, "mediator-required": 128}
    rank_reports = [
        {
            "rank": rank,
            "action_losses": [1.0] * 224 + [0.8] * 32,
            "all_gradients_finite": True,
            "action_information_set_counts": dict(counts),
            "action_information_set_schedule_sha256": schedule["sha256"],
            "runtime_schedule_sha256": RUNTIME_SCHEDULE_DIGESTS[rank],
            "training_final_model_local_state_sha256": TRAINING_MODEL_DIGESTS[rank],
            "post_checkpoint_save_model_local_state_sha256": TRAINING_MODEL_DIGESTS[rank],
            "arm_journal": {
                "schema": "picf-next.ltop-g3-arm-journal-receipt.v1",
                "rank": rank,
                "path": str(journal_dir / f"rank_{rank}.jsonl"),
                "file_sha256": ("a" if rank == 0 else "b") * 64,
                "record_count": 256,
            },
        }
        for rank in (0, 1)
    ]
    manifest = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": 256,
        "optimizer_saved": False,
        "format": MODEL_ONLY_CHECKPOINT_FORMAT,
        "world_size": 2,
        "model_tree_schema": MODEL_TREE_SCHEMA,
        "model_tree_sha256": model_tree_sha256,
        "training_final_model_local_state_sha256_by_rank": TRAINING_MODEL_DIGESTS,
        "action_information_set_schedule_sha256": schedule["sha256"],
        "action_information_set_counts_by_rank": [dict(counts), dict(counts)],
        "source_stage_checkpoint": identity["stage_checkpoint"],
        "g2_report_sha256": identity["g2_report_sha256"],
        "runtime_source_contract": identity["runtime_source_contract"],
    }
    manifest_path = checkpoint_path / "ltop_g3_training_checkpoint.json"
    _write(manifest_path, manifest)
    report = {
        "schema": "picf-next.ltop-g3-training-phase.v1",
        "status": "PASS",
        "failures": [],
        "mode": "mediator-trial",
        "phase": "training",
        "world_size": 2,
        "steps": 256,
        "eval_every": 32,
        "seed": TRAINING_SEED,
        **identity,
        "trained_checkpoint": None,
        "checkpoint": {
            "path": str(checkpoint_path),
            "format": MODEL_ONLY_CHECKPOINT_FORMAT,
            "optimizer_saved": False,
            "manifest_sha256": _sha256(manifest_path),
            "model_tree_schema": MODEL_TREE_SCHEMA,
            "model_tree_sha256": model_tree_sha256,
            "training_final_model_local_state_sha256_by_rank": TRAINING_MODEL_DIGESTS,
        },
        "training_contract": {
            "optimizer": {"algorithm": "released"},
            "action_information_set_trial": {
                "single_forward_per_optimizer_step": True,
                "schedule": schedule,
                "evaluation_intervention_enum_is_separate": True,
            },
        },
        "rank_reports": rank_reports,
    }
    return report


def _arm_window(*, count: int, window_count: int) -> dict[str, object]:
    return {
        "action_loss_finite": True,
        "all_reported_losses_finite": True,
        "balanced_count_pass": True,
        "count": count,
        "expected_count": count,
        "first_window": {
            "count": window_count,
            "finite": True,
            "mean_action_loss": 1.0,
        },
        "last_to_first_ratio": 0.8,
        "last_window": {
            "count": window_count,
            "finite": True,
            "mean_action_loss": 0.8,
        },
        "maximum_last_to_first_ratio": 0.95,
        "relative_improvement": 0.2,
        "window_gate_pass": True,
    }


def _arm_validation(
    training_path: Path,
    training: dict[str, object],
) -> dict[str, object]:
    schedule = training["training_contract"]["action_information_set_trial"]["schedule"]
    schedule_sha256 = schedule["sha256"]
    ranks = training["rank_reports"]
    journal_dir = str(Path(ranks[0]["arm_journal"]["path"]).parent)
    return {
        "schema": ARM_VALIDATION_SCHEMA,
        "status": "PASS",
        "failures": [],
        "inputs": {
            "journal_dir": journal_dir,
            "report": str(training_path.resolve()),
        },
        "thresholds": {
            "expected_count_per_arm_per_rank": 128,
            "expected_world_size": 2,
            "maximum_last_to_first_ratio": 0.95,
            "window_size": 16,
        },
        "final_report": {
            "consistent": True,
            "file_sha256": _sha256(training_path),
            "path": str(training_path.resolve()),
            "runner_failures": [],
            "runner_status": "PASS",
            "schema": "picf-next.ltop-g3-training-phase.v1",
        },
        "global": {
            "aggregation": "pooled-rank-local-arm-windows",
            "arms": {
                "FACTUAL": _arm_window(count=256, window_count=32),
                "MEDIATOR_REQUIRED": _arm_window(count=256, window_count=32),
            },
            "balanced_arms_pass": True,
            "finite_pass": True,
            "record_count": 512,
            "schedule": {
                "consistent": True,
                "entries_consistent_across_ranks": True,
                "rank_digests": [schedule_sha256, schedule_sha256],
                "sha256": schedule_sha256,
            },
            "window_gates_pass": True,
        },
        "ranks": [
            {
                "rank": rank,
                "arms": {
                    "FACTUAL": _arm_window(count=128, window_count=16),
                    "MEDIATOR_REQUIRED": _arm_window(count=128, window_count=16),
                },
                "balanced_arms_pass": True,
                "finite_pass": True,
                "journal": {
                    "path": ranks[rank]["arm_journal"]["path"],
                    "file_sha256": ranks[rank]["arm_journal"]["file_sha256"],
                    "record_count": 256,
                },
                "schedule": {
                    "consistent": True,
                    "digests": [schedule_sha256],
                    "sha256": schedule_sha256,
                },
                "window_gates_pass": True,
            }
            for rank in (0, 1)
        ],
    }


def _cold_model_identity(
    training: dict[str, object],
    *,
    rank: int,
) -> dict[str, object]:
    training_rank = training["rank_reports"][rank]
    model_digest = training_rank["training_final_model_local_state_sha256"]
    return {
        "cold_loaded_model_local_state_sha256": model_digest,
        "post_evaluation_model_local_state_sha256": model_digest,
        "trained_checkpoint_model_tree_sha256": training["checkpoint"]["model_tree_sha256"],
        "runtime_schedule_sha256": training_rank["runtime_schedule_sha256"],
        "trained_model_local_state_sha256": model_digest,
    }


def _action_report(
    identity: dict[str, object],
    training: dict[str, object],
) -> dict[str, object]:
    score = {
        "sample_keys": ["sample-a", "sample-b"],
        "mean_factual_target_minus_distractor": 0.4,
        "mean_blocked_path_difference_in_differences": 0.3,
        "positive_factual_count": 2,
        "positive_blocked_path_did_count": 2,
    }
    return {
        "schema": "picf-next.ltop-g3-evaluation-phase.v1",
        "status": "PASS",
        "failures": [],
        "mode": "gate",
        "phase": "evaluation",
        "world_size": 2,
        "steps": 128,
        "eval_every": 32,
        "seed": training["seed"],
        **identity,
        "trained_checkpoint": training["checkpoint"]["path"],
        "checkpoint": None,
        "action_inference_contract": {
            "surface": "policy.sample_actions",
            "fixed_noise": True,
        },
        "thresholds": {
            "bitwise_factual_replay": True,
            "action_loss_improvement_ratio_maximum": 0.95,
            "mean_factual_target_minus_distractor_strictly_positive": True,
            "mean_blocked_path_did_strictly_positive": True,
            "positive_sample_fraction_minimum": 0.625,
        },
        "rank_reports": [
            {
                "rank": rank,
                **_cold_model_identity(training, rank=rank),
                "history": [
                    {
                        "step": 128,
                        **{
                            partition: {
                                "max_replay_floor_rms": 0.0,
                                "scenes": [{"score": copy.deepcopy(score)}],
                            }
                            for partition in ("validation", "heldout")
                        },
                    }
                ],
            }
            for rank in (0, 1)
        ],
    }


def _retention_report(
    identity: dict[str, object],
    training: dict[str, object],
) -> dict[str, object]:
    partition = {
        "scene_count": 4,
        "prompt_count": 8,
        "positive_margin_count": 8,
        "mean_margin": 0.2,
        "shared_row_gauge": True,
        "physical_prompt_drift_max_abs": 0.0,
        "metric_self_checks": {"matched_row_permutation_max_abs_error": 0.0},
    }
    return {
        "schema": "picf-next.ltop-g3-representation-retention.v1",
        "status": "PASS",
        "failures": [],
        "mode": "gate",
        "phase": "retention",
        "world_size": 2,
        "steps": 128,
        "eval_every": 32,
        "seed": training["seed"],
        **identity,
        "trained_checkpoint": training["checkpoint"]["path"],
        "checkpoint": None,
        "representation_retention_contract": {
            "optimizer_updates": 0,
            "purpose": "representation retention",
        },
        "thresholds": {
            "validation_positive_prompt_margins_global_minimum": 12,
            "validation_mean_margin_minimum": 0.02,
            "heldout_positive_prompt_margins_global_minimum": 10,
            "heldout_mean_margin_strictly_positive": True,
        },
        "scene_level_robustness": {
            "validation": {"interpretation": "ROBUST_POSITIVE"},
            "heldout": {"interpretation": "ROBUST_POSITIVE"},
        },
        "rank_reports": [
            {
                "rank": rank,
                **_cold_model_identity(training, rank=rank),
                "history": [
                    {
                        "step": 128,
                        "validation": copy.deepcopy(partition),
                        "heldout": copy.deepcopy(partition),
                    }
                ],
            }
            for rank in (0, 1)
        ],
    }


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    identity = _identity(tmp_path)
    training = _training_report(tmp_path, identity)
    training_path = tmp_path / "training.json"
    _write(training_path, training)
    arm_path = tmp_path / "arm.json"
    action_path = tmp_path / "action.json"
    retention_path = tmp_path / "retention.json"
    _write(arm_path, _arm_validation(training_path, training))
    _write(action_path, _action_report(identity, training))
    _write(retention_path, _retention_report(identity, training))
    return {
        "training": training_path,
        "arm": arm_path,
        "action": action_path,
        "retention": retention_path,
    }


def _compose(paths: dict[str, Path]) -> dict[str, object]:
    return compose_ltop_g3_mediator_acceptance(
        training_path=paths["training"],
        arm_validation_path=paths["arm"],
        action_evaluation_path=paths["action"],
        retention_path=paths["retention"],
    )


def test_composer_emits_fixed_loader_abi_and_receipts(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    training = json.loads(paths["training"].read_text())

    result = _compose(paths)

    assert result["schema"] == MEDIATOR_ACCEPTANCE_SCHEMA
    assert result["status"] == "PASS"
    assert result["failures"] == []
    assert result["world_size"] == 2
    assert result["checkpoint"] == training["checkpoint"]
    assert result["training_final_model_local_state_sha256_by_rank"] == (TRAINING_MODEL_DIGESTS)
    assert result["cold_load_model_local_state_sha256_by_rank"] == {
        "action_evaluation": TRAINING_MODEL_DIGESTS,
        "retention": TRAINING_MODEL_DIGESTS,
    }
    assert result["checkpoint_identity"] == {
        "manifest_sha256": training["checkpoint"]["manifest_sha256"],
        "model_tree_schema": MODEL_TREE_SCHEMA,
        "model_tree_sha256": training["checkpoint"]["model_tree_sha256"],
    }
    assert result["training_contract"] == {
        "mode": "mediator-trial",
        "steps": 256,
        "eval_every": 32,
        "action_information_set_policy": ACTION_INFORMATION_SET_POLICY,
        "schedule_sha256": training["training_contract"]["action_information_set_trial"][
            "schedule"
        ]["sha256"],
        "seed": TRAINING_SEED,
        "runtime_schedule_sha256_by_rank": RUNTIME_SCHEDULE_DIGESTS,
    }
    assert set(result["evidence"]) == {
        "training_report",
        "arm_validation",
        "cold_action_evaluation",
        "cold_retention",
    }
    for evidence in result["evidence"].values():
        assert Path(evidence["path"]).is_absolute()
        assert evidence["sha256"] == _sha256(Path(evidence["path"]))
    assert result["action_summary"]["partitions"]["heldout"][
        "mean_factual_target_minus_distractor"
    ] == pytest.approx(0.4)
    assert result["retention_summary"]["partitions"]["validation"]["positive_margin_count"] == 16


@pytest.mark.parametrize(
    "field",
    [
        "architecture_identity",
        "runtime_source_contract",
        "stage_checkpoint",
        "g2_report_sha256",
        "capacity",
        "task_query_count",
        "execution_contract_sha256",
        "offline_labels_sha256",
        "physical_sidecar_manifest_sha256",
        "dataset_contract",
    ],
)
def test_composer_rejects_every_identity_drift(tmp_path: Path, field: str) -> None:
    paths = _artifacts(tmp_path)
    action = json.loads(paths["action"].read_text())
    action[field] = {"different": True} if isinstance(action[field], dict) else "different"
    _write(paths["action"], action)

    with pytest.raises(MediatorAcceptanceContractError, match=field):
        _compose(paths)


@pytest.mark.parametrize("artifact", ["training", "arm", "action", "retention"])
def test_composer_rejects_any_nonpass_input(tmp_path: Path, artifact: str) -> None:
    paths = _artifacts(tmp_path)
    report = json.loads(paths[artifact].read_text())
    report["status"] = "FAIL"
    report["failures"] = ["injected"]
    _write(paths[artifact], report)

    with pytest.raises(MediatorAcceptanceContractError, match="status"):
        _compose(paths)


@pytest.mark.parametrize("artifact", ["action", "retention"])
def test_composer_rejects_checkpoint_binding_drift(tmp_path: Path, artifact: str) -> None:
    paths = _artifacts(tmp_path)
    report = json.loads(paths[artifact].read_text())
    report["trained_checkpoint"] = str((tmp_path / "other-checkpoint").resolve())
    _write(paths[artifact], report)

    with pytest.raises(MediatorAcceptanceContractError, match="trained_checkpoint"):
        _compose(paths)


def test_composer_rejects_arm_report_hash_drift(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    arm = json.loads(paths["arm"].read_text())
    arm["final_report"]["file_sha256"] = "f" * 64
    _write(paths["arm"], arm)

    with pytest.raises(MediatorAcceptanceContractError, match="file_sha256"):
        _compose(paths)


def test_composer_rejects_missing_cold_rank_local_initialization_identity(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    action = json.loads(paths["action"].read_text())
    action["rank_reports"][1]["cold_loaded_model_local_state_sha256"] = None
    _write(paths["action"], action)

    with pytest.raises(MediatorAcceptanceContractError, match="cold-loaded model local state"):
        _compose(paths)


def test_composer_rejects_disagreement_between_independent_cold_loads(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    retention = json.loads(paths["retention"].read_text())
    retention_rank = retention["rank_reports"][1]
    retention_rank["cold_loaded_model_local_state_sha256"] = "e" * 64
    retention_rank["post_evaluation_model_local_state_sha256"] = "e" * 64
    retention_rank["trained_model_local_state_sha256"] = "e" * 64
    _write(paths["retention"], retention)

    with pytest.raises(MediatorAcceptanceContractError, match="training terminal model state"):
        _compose(paths)


def test_composer_rejects_live_model_tree_byte_tamper(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    training = json.loads(paths["training"].read_text())
    model_shard = Path(training["checkpoint"]["path"]) / "model" / "__0_0.distcp"
    model_shard.write_bytes(model_shard.read_bytes() + b"tampered")

    with pytest.raises(MediatorAcceptanceContractError, match="model tree SHA-256 differs"):
        _compose(paths)


def test_composer_rejects_post_forward_model_digest_drift(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    action = json.loads(paths["action"].read_text())
    action["rank_reports"][0]["post_evaluation_model_local_state_sha256"] = "e" * 64
    _write(paths["action"], action)

    with pytest.raises(MediatorAcceptanceContractError, match="mutated persistent model state"):
        _compose(paths)


def test_composer_rejects_seed_drift(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    action = json.loads(paths["action"].read_text())
    action["seed"] = TRAINING_SEED + 1
    _write(paths["action"], action)

    with pytest.raises(MediatorAcceptanceContractError, match=r"action\.seed"):
        _compose(paths)


def test_composer_rejects_runtime_schedule_drift(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    retention = json.loads(paths["retention"].read_text())
    retention["rank_reports"][1]["runtime_schedule_sha256"] = "e" * 64
    _write(paths["retention"], retention)

    with pytest.raises(MediatorAcceptanceContractError, match="runtime schedule differs"):
        _compose(paths)


def test_composer_rejects_live_manifest_hash_drift(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    training = json.loads(paths["training"].read_text())
    manifest_path = Path(training["checkpoint"]["path"]) / "ltop_g3_training_checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tampered_after_receipt"] = True
    _write(manifest_path, manifest)

    with pytest.raises(MediatorAcceptanceContractError, match="manifest SHA-256 differs"):
        _compose(paths)


def test_composer_rejects_journal_receipt_drift(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    arm = json.loads(paths["arm"].read_text())
    arm["ranks"][1]["journal"]["file_sha256"] = "e" * 64
    _write(paths["arm"], arm)

    with pytest.raises(MediatorAcceptanceContractError, match="journal file_sha256"):
        _compose(paths)


@pytest.mark.parametrize("label", ["FACTUAL", "MEDIATOR_REQUIRED"])
def test_composer_rejects_arm_with_less_than_five_percent_improvement(
    tmp_path: Path,
    label: str,
) -> None:
    paths = _artifacts(tmp_path)
    arm = json.loads(paths["arm"].read_text())
    arm["global"]["arms"][label]["last_to_first_ratio"] = 0.96
    arm["global"]["arms"][label]["relative_improvement"] = 0.04
    _write(paths["arm"], arm)

    with pytest.raises(MediatorAcceptanceContractError, match="five percent"):
        _compose(paths)


def test_composer_rejects_action_gate_claim_with_negative_effect(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    action = json.loads(paths["action"].read_text())
    for rank in action["rank_reports"]:
        rank["history"][0]["heldout"]["scenes"][0]["score"][
            "mean_factual_target_minus_distractor"
        ] = -0.1
    _write(paths["action"], action)

    with pytest.raises(MediatorAcceptanceContractError, match="nonpositive"):
        _compose(paths)


def test_composer_rejects_retention_gate_claim_below_floor(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    retention = json.loads(paths["retention"].read_text())
    for rank in retention["rank_reports"]:
        rank["history"][0]["heldout"]["mean_margin"] = -0.1
    _write(paths["retention"], retention)

    with pytest.raises(MediatorAcceptanceContractError, match="mean margin"):
        _compose(paths)


def test_composer_does_not_mutate_inputs(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    before = {name: path.read_bytes() for name, path in paths.items()}

    _compose(paths)

    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_cli_publishes_once_and_never_replaces_output(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    output = tmp_path / "acceptance.json"
    arguments = [
        "--training-report",
        str(paths["training"]),
        "--arm-validation",
        str(paths["arm"]),
        "--action-evaluation-report",
        str(paths["action"]),
        "--retention-report",
        str(paths["retention"]),
        "--output",
        str(output),
    ]

    assert main(arguments) == 0
    first = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2
    assert output.read_bytes() == first
    assert json.loads(first)["schema"] == MEDIATOR_ACCEPTANCE_SCHEMA


def test_composer_rejects_symlink_input(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    symlink = tmp_path / "action-link.json"
    symlink.symlink_to(paths["action"])
    paths["action"] = symlink

    with pytest.raises(MediatorAcceptanceContractError, match="non-symlink"):
        _compose(paths)
