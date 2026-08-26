from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from picf_next.lingbot_native.ltop_core_pair import (
    LTOP_CORE_PAIR_CHECKPOINT_SCHEMA,
    LTOP_CORE_PAIR_JOURNAL_SCHEMA,
    LTOP_CORE_PAIR_METRICS_SCHEMA,
    LTOP_CORE_PAIR_OPTIMIZER_INITIALIZATION_SCHEMA,
    compose_ltop_core_pair,
)
from picf_next.lingbot_native.ltop_core_pilot import (
    LTOP_CORE_PILOT_SCHEMA,
    LTOPCorePilotArm,
    matched_arm_contract,
)

_TOTAL_STEPS = 2_000
_WINDOW = 100
_MEAN_FIELDS = (
    "total_loss",
    "action_loss",
    "moe_regularizer",
    "physical_set_loss",
    "task_address_loss",
    "step_time_s",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.test-tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any, *, allow_nan: bool = False) -> None:
    encoded = (
        json.dumps(payload, allow_nan=allow_nan, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")
    _write_bytes_atomic(path, encoded)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def _relative_metric_path(start_step: int, end_step: int) -> str:
    return f"metrics/steps_{start_step:08d}_{end_step:08d}.json"


def _step_record(*, arm: LTOPCorePilotArm, rank: int, step: int) -> dict[str, Any]:
    blocked_offset = 0.05 if arm is LTOPCorePilotArm.BLOCKED else 0.0
    action_loss = 1.0 / (step + rank + 10) + blocked_offset
    exact_target = step % 17 != 0
    return {
        "global_step": step,
        "sample_keys": [f"sample-{rank}-{step}"],
        "lane_ids": [rank],
        "frame_indices": [step],
        "reset": [step == 1],
        "source_digest": _sha(f"source-{rank}-{step}"),
        "augmentation_seeds": [step * 3 + rank],
        "flow_noise_seeds": [step * 5 + rank],
        "flow_timestep_seeds": [step * 7 + rank],
        "total_loss": action_loss + 0.3,
        "action_loss": action_loss,
        "moe_regularizer": 0.01,
        "physical_set_loss": 0.2,
        "task_address_loss": 0.1,
        "target_identity": "blue_block" if exact_target else None,
        "target_row": 1 if exact_target else None,
        "gradient_metrics": {
            "all_finite": True,
            "preclip_global_norm": 1.0 + rank / 10,
        },
        "step_time_s": 0.2 + blocked_offset,
        "peak_cuda_allocated_bytes": 1_000 + rank,
        "peak_cuda_reserved_bytes": 2_000 + rank,
        "model_input_sha256": _sha(f"model-input-{rank}-{step}"),
        "controls_sha256": _sha(f"controls-{rank}-{step}"),
        "prior_controls_sha256": _sha(f"prior-controls-{rank}-{step}"),
        "structural_targets_sha256": _sha(f"structural-targets-{rank}-{step}"),
        "normalized_forward_input_sha256": _sha(f"normalized-forward-{rank}-{step}"),
        "forward_input_sha256": _sha(f"{arm.value}-forward-{rank}-{step}"),
        "executed_object_read_action_intervention": (
            "factual" if arm is LTOPCorePilotArm.FACTUAL else "blocked"
        ),
    }


def _stage_restore(*, arm: LTOPCorePilotArm, rank: int) -> dict[str, Any]:
    model_digest = _sha(f"initial-model-rank-{rank}")
    volatile_offset = 0 if arm is LTOPCorePilotArm.FACTUAL else 100
    return {
        "rank": rank,
        "hostname": f"host-{arm.value}",
        "pid": 10_000 + volatile_offset + rank,
        "expected_model_local_state_sha256": model_digest,
        "actual_model_local_state_sha256": model_digest,
        "digest_match": True,
        "meta_state_names_before_load": [],
        "meta_state_names_after_load": [],
        "fsdp2_storage_before_load": {"layout": "dtensor", "rank": rank},
        "fsdp2_storage_after_load": {"layout": "dtensor", "rank": rank},
        "timings": {
            "model_build_s": 1.0 + volatile_offset,
            "dcp_load_s": 2.0 + volatile_offset,
        },
        "cuda_memory_bytes": {
            "allocated": 1_000 + volatile_offset,
            "reserved": 2_000 + volatile_offset,
            "peak_allocated": 3_000 + volatile_offset,
            "peak_reserved": 4_000 + volatile_offset,
        },
    }


def _optimizer_manifest() -> dict[str, Any]:
    return {
        "canonical_names": ["policy.action", "policy.shared_host"],
        "parameter_count": 2,
        "trainable_numel": 128,
        "schema_sha256": _sha("optimizer-manifest"),
    }


def _optimizer_initialization(*, rank: int) -> dict[str, Any]:
    return {
        "schema": LTOP_CORE_PAIR_OPTIMIZER_INITIALIZATION_SCHEMA,
        "rank": rank,
        "fresh_zero_state": True,
        "state_entry_count": 0,
        "parameter_manifest_sha256": _sha("optimizer-manifest"),
        "parameter_groups_sha256": _sha("optimizer-parameter-groups"),
        "optimizer_state_sha256": _sha(f"empty-optimizer-state-{rank}"),
        "model_local_state_sha256": _sha(f"initial-model-rank-{rank}"),
        "rank_rng_state_sha256": _sha(f"initial-rng-rank-{rank}"),
    }


def _checkpoint_manifest(*, arm: LTOPCorePilotArm) -> dict[str, Any]:
    return {
        "schema": LTOP_CORE_PAIR_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": _TOTAL_STEPS,
        "arm": arm.value,
        "g2_report_sha256": _sha("g2-report"),
        "g3_report_sha256": _sha("g3-report"),
        "stream_plan_sha256": _sha("stream-plan"),
        "rank_boundaries": [
            {
                "rank": rank,
                "boundary": {
                    "model_local_state_sha256": _sha(f"{arm.value}-final-model-{rank}"),
                    "optimizer_local_state_sha256": _sha(f"{arm.value}-final-optimizer-{rank}"),
                    "lane_snapshot_sha256": _sha(f"lane-{rank}"),
                    "rank_rng_state_sha256": _sha(f"{arm.value}-final-rng-{rank}"),
                },
            }
            for rank in (0, 1)
        ],
    }


def _write_run(root: Path, *, arm: LTOPCorePilotArm) -> Path:
    records_by_rank = {
        rank: [_step_record(arm=arm, rank=rank, step=step) for step in range(1, 2_001)]
        for rank in (0, 1)
    }
    journal_receipts: dict[int, dict[str, Any]] = {}
    for rank, records in records_by_rank.items():
        relative_path = f"metrics/rank_journal/rank_{rank}.jsonl"
        journal_path = root / relative_path
        payload = "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n" for record in records
        ).encode("ascii")
        _write_bytes_atomic(journal_path, payload)
        journal_receipts[rank] = {
            "schema": LTOP_CORE_PAIR_JOURNAL_SCHEMA,
            "rank": rank,
            "path": relative_path,
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "record_count": _TOTAL_STEPS,
        }

    metric_receipts: list[dict[str, Any]] = []
    for start_step in range(1, _TOTAL_STEPS + 1, _WINDOW):
        end_step = start_step + _WINDOW - 1
        window_records = [
            record for rank in (0, 1) for record in records_by_rank[rank][start_step - 1 : end_step]
        ]
        means = {
            field: sum(float(record[field]) for record in window_records) / len(window_records)
            for field in _MEAN_FIELDS
        }
        artifact = {
            "schema": LTOP_CORE_PAIR_METRICS_SCHEMA,
            "arm": arm.value,
            "start_step": start_step,
            "end_step": end_step,
            "sample_count": sum(len(record["sample_keys"]) for record in window_records),
            "means": means,
            "rank_windows": [
                {
                    "rank": rank,
                    "steps": records_by_rank[rank][start_step - 1 : end_step],
                }
                for rank in (0, 1)
            ],
        }
        relative_path = _relative_metric_path(start_step, end_step)
        metric_path = root / relative_path
        _write_json(metric_path, artifact)
        metric_receipts.append(
            {
                "path": relative_path,
                "file_sha256": hashlib.sha256(metric_path.read_bytes()).hexdigest(),
                "start_step": start_step,
                "end_step": end_step,
                "means": means,
            }
        )

    checkpoint_relative = "checkpoints/global_step_2000"
    checkpoint_manifest_path = root / checkpoint_relative / "ltop_core_pilot_checkpoint.json"
    _write_json(checkpoint_manifest_path, _checkpoint_manifest(arm=arm))
    checkpoint_receipt = {
        "path": checkpoint_relative,
        "manifest_sha256": hashlib.sha256(checkpoint_manifest_path.read_bytes()).hexdigest(),
    }

    manifest = _optimizer_manifest()
    rank_reports = []
    for rank in (0, 1):
        action_losses = [record["action_loss"] for record in records_by_rank[rank]]
        rank_reports.append(
            {
                "rank": rank,
                "metric_reports": copy.deepcopy(metric_receipts),
                "diagnostics": [],
                "all_gradients_finite": True,
                "action_loss_first_100_mean": sum(action_losses[:100]) / 100,
                "action_loss_last_100_mean": sum(action_losses[-100:]) / 100,
                "optimizer_parameter_manifest": copy.deepcopy(manifest),
                "optimizer_initialization": _optimizer_initialization(rank=rank),
                "journal": journal_receipts[rank],
                "stage_restore": _stage_restore(arm=arm, rank=rank),
                "checkpoint": copy.deepcopy(checkpoint_receipt),
                "timings": {"mean_wall_s_per_optimizer_step": 0.3},
                "cuda_memory_bytes": {
                    "allocated": 10_000,
                    "reserved": 20_000,
                    "peak_allocated": 30_000,
                    "peak_reserved": 40_000,
                },
            }
        )

    report = {
        "schema": LTOP_CORE_PILOT_SCHEMA,
        "status": "PASS",
        "failures": [],
        "mode": "pilot",
        "arm": arm.value,
        "arm_contract": matched_arm_contract(arm),
        "architecture_identity": "lingbot-vla2-ltop-core",
        "world_size": 2,
        "steps": _TOTAL_STEPS,
        "cadence": {
            "total_steps": _TOTAL_STEPS,
            "metrics_every": 100,
            "diagnostics_every": 250,
            "checkpoint_step": _TOTAL_STEPS,
        },
        "seed": 20260813,
        "capacity": 8,
        "task_query_count": 4,
        "stage_checkpoint": "/mnt/stage",
        "g2_report_sha256": _sha("g2-report"),
        "g3_report_sha256": _sha("g3-report"),
        "dataset_contract": {"manifest_sha256": _sha("dataset")},
        "stream_plan_sha256": _sha("stream-plan"),
        "representation_split_sha256": _sha("representation-split"),
        "evaluation_plan_sha256": _sha("evaluation-plan"),
        "execution_contract_sha256": _sha("execution-contract"),
        "offline_labels_sha256": _sha("offline-labels"),
        "physical_sidecar_manifest_sha256": _sha("physical-sidecar"),
        "source_identity": {
            "git_commit": "a" * 40,
            "source_tree_sha256": _sha("source-tree"),
        },
        "runtime_environment_contract": {
            "torch": "2.8.0",
            "cuda": "12.8",
            "deterministic_algorithms": True,
        },
        "action_inference_contract": {"surface": "separate-fresh-process-evaluator"},
        "training_contract": {
            "optimizer": "released",
            "fresh_optimizer_after_strict_model_only_restore": True,
        },
        "checkpoint": checkpoint_receipt,
        "scientific_boundary": "paired execution integrity only",
        "rank_reports": rank_reports,
    }
    path = root / "ltop_core_pilot_report.json"
    _write_json(path, report)
    return path


@pytest.fixture(scope="module")
def pair_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ltop-core-pair-template")
    _write_run(root / "factual", arm=LTOPCorePilotArm.FACTUAL)
    _write_run(root / "blocked", arm=LTOPCorePilotArm.BLOCKED)
    return root


@pytest.fixture
def pair_paths(pair_template: Path, tmp_path: Path) -> tuple[Path, Path]:
    for arm in ("factual", "blocked"):
        shutil.copytree(pair_template / arm, tmp_path / arm, copy_function=os.link)
    return (
        tmp_path / "factual" / "ltop_core_pilot_report.json",
        tmp_path / "blocked" / "ltop_core_pilot_report.json",
    )


def _mutate_report(path: Path, mutation: Any) -> dict[str, Any]:
    report = _load_json(path)
    mutation(report)
    _write_json(path, report)
    return report


def _journal_records(report_path: Path, rank: int) -> list[dict[str, Any]]:
    path = report_path.parent / "metrics" / "rank_journal" / f"rank_{rank}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def _replace_journal(
    report_path: Path,
    *,
    rank: int,
    records: list[dict[str, Any]],
) -> None:
    journal_path = report_path.parent / "metrics" / "rank_journal" / f"rank_{rank}.jsonl"
    payload = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n" for record in records
    ).encode("ascii")
    _write_bytes_atomic(journal_path, payload)
    report = _load_json(report_path)
    report["rank_reports"][rank]["journal"]["file_sha256"] = hashlib.sha256(payload).hexdigest()
    _write_json(report_path, report)


def _replace_metric_artifact(
    report_path: Path,
    *,
    index: int,
    artifact: dict[str, Any],
    allow_nan: bool = False,
    synchronize_receipt_means: bool = False,
) -> None:
    start_step = index * _WINDOW + 1
    end_step = start_step + _WINDOW - 1
    artifact_path = report_path.parent / _relative_metric_path(start_step, end_step)
    _write_json(artifact_path, artifact, allow_nan=allow_nan)
    report = _load_json(report_path)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    for rank_report in report["rank_reports"]:
        receipt = rank_report["metric_reports"][index]
        receipt["file_sha256"] = digest
        if synchronize_receipt_means:
            receipt["means"] = artifact["means"]
    _write_json(report_path, report)


def test_pair_composer_accepts_real_stage_shape_and_recomputes_metrics(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths

    result = compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)

    assert result["status"] == "PASS"
    assert len(result["action_loss_curves"]["factual"]) == 20
    assert len(result["metric_evidence"][LTOPCorePilotArm.FACTUAL.value]) == 20
    assert all(item["difference"] < 0 for item in result["action_loss_factual_minus_blocked"])


def test_pair_composer_rejects_duplicate_report_key(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths
    payload = factual.read_bytes()
    _write_bytes_atomic(factual, b'{"schema":"duplicate",' + payload[1:])

    with pytest.raises(ValueError, match="duplicate JSON key 'schema'"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_non_finite_report_json(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths
    report = _load_json(factual)
    report["seed"] = float("nan")
    _write_json(factual, report, allow_nan=True)

    with pytest.raises(ValueError, match="non-finite JSON constant NaN"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_wrong_arm_contract(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths
    _mutate_report(
        factual,
        lambda report: report["arm_contract"].update(
            {"object_read_action_intervention": "blocked"}
        ),
    )

    with pytest.raises(ValueError, match="violates arm_contract"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("world_size", 4, "violates world_size"),
        (
            "cadence",
            {
                "total_steps": 2_000,
                "metrics_every": 200,
                "diagnostics_every": 250,
                "checkpoint_step": 2_000,
            },
            "violates cadence",
        ),
    ],
)
def test_pair_composer_rejects_non_registered_topology_or_cadence(
    pair_paths: tuple[Path, Path], field: str, value: Any, message: str
) -> None:
    factual, blocked = pair_paths
    _mutate_report(factual, lambda report: report.__setitem__(field, value))

    with pytest.raises(ValueError, match=message):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_ignores_only_volatile_stage_fields(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths
    _mutate_report(
        blocked,
        lambda report: report["rank_reports"][0]["stage_restore"].update(
            {
                "hostname": "another-host",
                "pid": 999_999,
                "timings": {"model_build_s": 999.0, "dcp_load_s": 888.0},
                "cuda_memory_bytes": {
                    "allocated": 9,
                    "reserved": 10,
                    "peak_allocated": 11,
                    "peak_reserved": 12,
                },
            }
        ),
    )

    assert compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)["status"] == "PASS"


def test_pair_composer_rejects_false_stage_digest_match(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths
    _mutate_report(
        blocked,
        lambda report: report["rank_reports"][0]["stage_restore"].update({"digest_match": False}),
    )

    with pytest.raises(ValueError, match="did not match its model digest"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_stable_stage_mismatch(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths
    changed = _sha("other-initial-model")

    def mutate(report: dict[str, Any]) -> None:
        restore = report["rank_reports"][0]["stage_restore"]
        restore["expected_model_local_state_sha256"] = changed
        restore["actual_model_local_state_sha256"] = changed
        report["rank_reports"][0]["optimizer_initialization"]["model_local_state_sha256"] = changed

    _mutate_report(blocked, mutate)

    with pytest.raises(ValueError, match="differs at stable stage restore"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_missing_optimizer_initialization(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths
    _mutate_report(
        blocked,
        lambda report: report["rank_reports"][0].pop("optimizer_initialization"),
    )

    with pytest.raises(ValueError, match=r"missing=\['optimizer_initialization'\]"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_nonzero_optimizer_initialization(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths
    _mutate_report(
        blocked,
        lambda report: report["rank_reports"][0]["optimizer_initialization"].update(
            {"state_entry_count": 1}
        ),
    )

    with pytest.raises(ValueError, match="violates state_entry_count"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_optimizer_initialization_mismatch(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths
    _mutate_report(
        blocked,
        lambda report: report["rank_reports"][0]["optimizer_initialization"].update(
            {"rank_rng_state_sha256": _sha("different-initial-rng")}
        ),
    )

    with pytest.raises(ValueError, match="differs at optimizer initialization"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_journal_sha_mismatch(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths
    _mutate_report(
        blocked,
        lambda report: report["rank_reports"][0]["journal"].update(
            {"file_sha256": _sha("wrong-journal")}
        ),
    )

    with pytest.raises(ValueError, match="journal SHA differs"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_checkpoint_sha_mismatch(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths

    def mutate(report: dict[str, Any]) -> None:
        report["checkpoint"]["manifest_sha256"] = _sha("wrong-checkpoint")
        for rank_report in report["rank_reports"]:
            rank_report["checkpoint"]["manifest_sha256"] = _sha("wrong-checkpoint")

    _mutate_report(blocked, mutate)

    with pytest.raises(ValueError, match="checkpoint manifest SHA differs"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_input_sha256", _sha("different-model-input"), "model_input_sha256"),
        ("flow_noise_seeds", [987_654], "flow_noise_seeds"),
        (
            "normalized_forward_input_sha256",
            _sha("different-normalized-forward"),
            "normalized_forward_input_sha256",
        ),
    ],
)
def test_pair_composer_rejects_changed_input_or_randomness_receipt(
    pair_paths: tuple[Path, Path], field: str, value: Any, message: str
) -> None:
    factual, blocked = pair_paths
    records = _journal_records(blocked, 0)
    records[730][field] = value
    _replace_journal(blocked, rank=0, records=records)

    with pytest.raises(ValueError, match=rf"step 731 differs at {message}"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_wrong_executed_intervention(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths
    records = _journal_records(blocked, 0)
    records[0]["executed_object_read_action_intervention"] = "factual"
    _replace_journal(blocked, rank=0, records=records)

    with pytest.raises(ValueError, match="executed the wrong OBJECT_READ->ACTION intervention"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_requires_raw_forward_digest_to_change(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths
    factual_records = _journal_records(factual, 0)
    blocked_records = _journal_records(blocked, 0)
    blocked_records[0]["forward_input_sha256"] = factual_records[0]["forward_input_sha256"]
    _replace_journal(blocked, rank=0, records=blocked_records)

    with pytest.raises(ValueError, match="raw forward digest did not change"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_missing_metric_window(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths

    def mutate(report: dict[str, Any]) -> None:
        for rank_report in report["rank_reports"]:
            rank_report["metric_reports"].pop()

    _mutate_report(blocked, mutate)

    with pytest.raises(ValueError, match="exactly 20 metric artifacts"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_rank_metric_receipt_mismatch(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths
    _mutate_report(
        blocked,
        lambda report: report["rank_reports"][1]["metric_reports"][0].update(
            {"file_sha256": _sha("different-rank-receipt")}
        ),
    )

    with pytest.raises(ValueError, match="rank 1 metric receipts differ"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_metric_artifact_schema(pair_paths: tuple[Path, Path]) -> None:
    factual, blocked = pair_paths
    artifact_path = blocked.parent / _relative_metric_path(1, 100)
    artifact = _load_json(artifact_path)
    artifact["schema"] = "wrong-schema"
    _replace_metric_artifact(blocked, index=0, artifact=artifact)

    with pytest.raises(ValueError, match="metric artifact 0 violates schema"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_non_finite_metric_artifact(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths
    artifact_path = blocked.parent / _relative_metric_path(1, 100)
    artifact = _load_json(artifact_path)
    artifact["means"]["action_loss"] = float("nan")
    _replace_metric_artifact(blocked, index=0, artifact=artifact, allow_nan=True)

    with pytest.raises(ValueError, match="non-finite JSON constant NaN"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_unreproducible_metric_mean(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths
    artifact_path = blocked.parent / _relative_metric_path(1, 100)
    artifact = _load_json(artifact_path)
    artifact["means"]["action_loss"] += 0.5
    _replace_metric_artifact(
        blocked,
        index=0,
        artifact=artifact,
        synchronize_receipt_means=True,
    )

    with pytest.raises(ValueError, match="action mean is not reproducible"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)


def test_pair_composer_rejects_metric_rank_window_not_bound_to_journal(
    pair_paths: tuple[Path, Path],
) -> None:
    factual, blocked = pair_paths
    artifact_path = blocked.parent / _relative_metric_path(1, 100)
    artifact = _load_json(artifact_path)
    artifact["rank_windows"][0]["steps"][0]["action_loss"] += 0.1
    _replace_metric_artifact(blocked, index=0, artifact=artifact)

    with pytest.raises(ValueError, match="rank 0 differs from its journal"):
        compose_ltop_core_pair(factual_path=factual, blocked_path=blocked)
