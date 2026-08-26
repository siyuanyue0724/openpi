from __future__ import annotations

import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from picf_next.lingbot_native.empirical_producers import (
    build_empirical_observations_from_producer,
    write_empirical_episode_artifact,
    write_empirical_producer_bundle,
)
from picf_next.lingbot_native.empirical_statistics import (
    EMPIRICAL_COMPARISON_SPECS,
    EMPIRICAL_EVALUATION_PLAN_SCHEMA,
    build_empirical_gate_report_from_observations,
)
from picf_next.lingbot_native.fsdp2_placement import FSDP2_GPU_SHARDED
from picf_next.lingbot_native.gate_evidence import (
    EMPIRICAL_COMPARISON_RULES,
    EMPIRICAL_REQUIRED_ARMS,
    EMPIRICAL_REQUIRED_CHECKS,
    validate_empirical_gate_report,
    validate_full_weight_smoke_report,
    validate_g0_report,
    validate_g2_visual_review,
    validate_g7_protocol,
    validate_preflight_report,
)
from tools import (
    build_lingbot_native_empirical_observations as empirical_observations_tool,
)
from tools import build_lingbot_native_empirical_report as empirical_report_tool
from tools import build_lingbot_native_evaluation_plan as evaluation_plan_tool
from tools.bootstrap_lingbot_vla2 import (
    CHECKPOINT_ASSET_CONTRACT,
    LINGBOT_CHECKPOINT_ID,
    LINGBOT_CHECKPOINT_REVISION,
    PROCESSOR_ASSET_CONTRACT,
    QWEN_PROCESSOR_ID,
    QWEN_PROCESSOR_REVISION,
    asset_contract_manifest,
)
from tools.bootstrap_lingbot_vla2_native import LINGBOT_NATIVE_SOURCE_COMMIT
from tools.build_lingbot_native_gate_decision import build_training_gate_decision
from tools.preflight_lingbot_native import PREFLIGHT_REPORT_SCHEMA
from tools.run_lingbot_vla2_native_full import (
    TRAINING_GATE_EVIDENCE_SCHEMAS,
    load_training_gate_decision,
)
from tools.run_lingbot_vla2_native_g0 import G0_REPORT_SCHEMA


def _digest(value: bytes = b"contract") -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _subject() -> dict[str, object]:
    return {
        "input_full_report_sha256": _digest(b"full-report"),
        "saved_global_step": 120,
        "execution_contract_sha256": _digest(b"execution"),
        "implementation_sha256": _digest(b"implementation"),
        "model_family_sha256": _digest(b"model"),
    }


def _producer_arrays(gate: str) -> dict[str, np.ndarray]:
    time, tokens, rows = 5, 6, 3
    target_masks = np.zeros((time, rows, tokens), dtype=np.float32)
    for row in range(rows):
        target_masks[:, row, 2 * row : 2 * row + 2] = 1
    mask_valid = np.ones_like(target_masks, dtype=np.bool_)
    candidate_support = np.transpose(target_masks, (0, 2, 1)).copy()
    generic_support = np.full_like(candidate_support, 0.5)
    if gate == "G2":
        return {
            "c_support": candidate_support,
            "m_support": generic_support,
            "target_masks": target_masks,
            "mask_valid": mask_valid,
            "c_existence": np.ones((time, rows), dtype=np.float32),
            "m_existence": np.full((time, rows), 0.5, dtype=np.float32),
            "target_existence": np.ones((time, rows), dtype=np.float32),
            "existence_valid": np.ones((time, rows), dtype=np.bool_),
            "c_task_relevance": np.ones(rows, dtype=np.float32),
            "m_task_relevance": np.zeros(rows, dtype=np.float32),
            "c_dense_task_grounding": np.ones((time, tokens), dtype=np.float32),
            "m_dense_task_grounding": np.zeros((time, tokens), dtype=np.float32),
            "target_task_relevance": np.ones(rows, dtype=np.float32),
            "task_valid": np.ones(rows, dtype=np.bool_),
            "track_valid": np.ones(rows, dtype=np.bool_),
            "capacity_censored": np.zeros(rows, dtype=np.bool_),
            "inventory_exhaustive": np.ones(time, dtype=np.bool_),
        }
    if gate == "G3":
        target_masks[2] = 0
        mask_valid[2] = False
        return {
            "c_support": np.transpose(target_masks, (0, 2, 1)).copy(),
            "o_support": generic_support,
            "target_masks": target_masks,
            "mask_valid": mask_valid,
            "c_existence": np.ones((time, rows), dtype=np.float32),
            "o_existence": np.full((time, rows), 0.5, dtype=np.float32),
            "target_existence": np.ones((time, rows), dtype=np.float32),
            "existence_valid": np.ones((time, rows), dtype=np.bool_),
            "track_valid": np.ones(rows, dtype=np.bool_),
            "capacity_censored": np.zeros(rows, dtype=np.bool_),
            "inventory_exhaustive": np.ones(time, dtype=np.bool_),
            "state_age": np.asarray([1, 8, 32, 64, 128], dtype=np.int64),
        }
    if gate == "G4":
        return {
            "same_entity_similarity": np.ones(3, dtype=np.float32),
            "hard_negative_similarity": np.zeros(3, dtype=np.float32),
            "all_available_quality": np.ones(3, dtype=np.float32),
            "missing_modality_quality": np.ones(3, dtype=np.float32),
            "corrupt_modality_quality": np.ones(3, dtype=np.float32),
            "whole_static_omission_trial": np.asarray([True, False, False]),
        }
    if gate == "G5":
        steps = np.asarray([0, 10, 20], dtype=np.int64)
        return {
            "steps": steps,
            "action_loss_a": np.full(3, 0.12, dtype=np.float32),
            "action_loss_h": np.full(3, 0.80, dtype=np.float32),
            "action_loss_m": np.full(3, 0.80, dtype=np.float32),
            "action_loss_o": np.full(3, 0.80, dtype=np.float32),
            "action_loss_c": np.full(3, 0.10, dtype=np.float32),
            "action_loss_c_row_intervened": np.full(3, 0.80, dtype=np.float32),
        }
    if gate == "G6":
        return {
            "sequence_length": np.asarray([5], dtype=np.int64),
            "successful_prefix_a": np.asarray([1], dtype=np.int64),
            "successful_prefix_o": np.asarray([1], dtype=np.int64),
            "successful_prefix_c": np.asarray([5], dtype=np.int64),
            "successful_prefix_c_row_intervened": np.asarray([0], dtype=np.int64),
            "recovery_o": np.zeros(3, dtype=np.bool_),
            "recovery_c": np.ones(3, dtype=np.bool_),
            "reset_session_isolation": np.ones(1, dtype=np.bool_),
        }
    raise ValueError(f"unsupported producer fixture gate: {gate}")


def _acceptance_fixture(gate: str) -> dict[str, float]:
    return {
        name: (-0.05 if rule == "lower_ge" else 0.0 if rule == "upper_le" else 0.5)
        for name, rule in EMPIRICAL_COMPARISON_RULES[gate].items()
    }


def _g2_no_object_arrays() -> dict[str, np.ndarray]:
    time, tokens, rows = 5, 6, 3
    return {
        "c_support": np.zeros((time, tokens, rows), dtype=np.float32),
        "m_support": np.full((time, tokens, rows), 0.5, dtype=np.float32),
        "target_masks": np.zeros((time, 0, tokens), dtype=np.float32),
        "mask_valid": np.zeros((time, 0, tokens), dtype=np.bool_),
        "c_existence": np.zeros((time, rows), dtype=np.float32),
        "m_existence": np.full((time, rows), 0.5, dtype=np.float32),
        "target_existence": np.zeros((time, 0), dtype=np.float32),
        "existence_valid": np.zeros((time, 0), dtype=np.bool_),
        "c_task_relevance": np.zeros(rows, dtype=np.float32),
        "m_task_relevance": np.zeros(rows, dtype=np.float32),
        "c_dense_task_grounding": np.zeros((time, tokens), dtype=np.float32),
        "m_dense_task_grounding": np.zeros((time, tokens), dtype=np.float32),
        "target_task_relevance": np.zeros(0, dtype=np.float32),
        "task_valid": np.zeros(0, dtype=np.bool_),
        "track_valid": np.zeros(0, dtype=np.bool_),
        "capacity_censored": np.zeros(0, dtype=np.bool_),
        "inventory_exhaustive": np.ones(time, dtype=np.bool_),
    }


def test_g0_routing_provenance_matches_native_calvin_static_abi() -> None:
    source_path = Path(__file__).resolve().parents[2] / "src/picf_next/lingbot_native/calvin.py"
    module = ast.parse(source_path.read_text())
    routing_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeCALVINRouting"
    )
    annotations = {
        node.target.id: ast.unparse(node.annotation)
        for node in routing_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert annotations["sample_keys"] == "tuple[str, ...]"
    assert annotations["episode_keys"] == "tuple[str, ...]"
    assert annotations["lane_ids"] == "tuple[int, ...]"
    assert annotations["frame_indices"] == "tuple[int, ...]"


@pytest.mark.parametrize("phase", ["fresh", "resume"])
def test_g0_report_recomputes_rank_and_checkpoint_evidence(
    tmp_path: Path,
    g0_report_factory,
    phase: str,
) -> None:
    path = g0_report_factory(tmp_path / f"g0-{phase}.json", phase=phase)
    report = json.loads(path.read_text())
    assert (
        validate_g0_report(
            report,
            schema=G0_REPORT_SCHEMA,
            phase=phase,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            world_size=2,
        )
        == report
    )

    wrong_data_contract = json.loads(json.dumps(report))
    wrong_data_contract["dataset_contract"]["validation"]["dataset_verification_mode"] = "wrong"
    with pytest.raises(ValueError, match="verified-read contract"):
        validate_g0_report(
            wrong_data_contract,
            schema=G0_REPORT_SCHEMA,
            phase=phase,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            world_size=2,
        )

    wrong_lane_type = json.loads(json.dumps(report))
    wrong_lane_type["rank_reports"][0]["lane_ids"] = ["lane-0"]
    with pytest.raises(ValueError, match="routing provenance"):
        validate_g0_report(
            wrong_lane_type,
            schema=G0_REPORT_SCHEMA,
            phase=phase,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            world_size=2,
        )

    wrong_sample_type = json.loads(json.dumps(report))
    wrong_sample_type["rank_reports"][0]["sample_keys"] = [0]
    with pytest.raises(ValueError, match="routing provenance"):
        validate_g0_report(
            wrong_sample_type,
            schema=G0_REPORT_SCHEMA,
            phase=phase,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            world_size=2,
        )

    bool_frame = json.loads(json.dumps(report))
    bool_frame["rank_reports"][0]["frame_indices"] = [True]
    with pytest.raises(ValueError, match="routing provenance"):
        validate_g0_report(
            bool_frame,
            schema=G0_REPORT_SCHEMA,
            phase=phase,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            world_size=2,
        )

    report["rank_reports"][0]["official_policy_loss"] = 0.5
    checkpoint_report = Path(report["checkpoint_dir"]) / "native_g0_report.json"
    checkpoint_report.write_text(json.dumps(report, sort_keys=True))
    with pytest.raises(ValueError, match="action plus MoE"):
        validate_g0_report(
            report,
            schema=G0_REPORT_SCHEMA,
            phase=phase,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            world_size=2,
        )


def test_g0_gate_rejects_a_resume_from_another_rank_boundary(
    tmp_path: Path,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    criteria = tmp_path / "criteria.md"
    criteria.write_text("frozen G0 criteria")
    preflight = preflight_report_factory(tmp_path / "preflight.json")
    neutral = smoke_report_factory(tmp_path / "neutral.json")
    fresh = g0_report_factory(tmp_path / "fresh.json", phase="fresh")
    resumed = g0_report_factory(tmp_path / "resumed.json", phase="resume")
    resumed_report = json.loads(resumed.read_text())
    resumed_report["rank_reports"][0]["loaded_boundary_sha256"] = {
        name: "b" * 64 for name in resumed_report["rank_reports"][0]["loaded_boundary_sha256"]
    }
    payload = json.dumps(resumed_report, sort_keys=True)
    resumed.write_text(payload)
    checkpoint_report = Path(resumed_report["checkpoint_dir"]) / "native_g0_report.json"
    checkpoint_report.write_text(payload)
    with pytest.raises(ValueError, match="did not load the fresh"):
        build_training_gate_decision(
            gate="G0",
            reviewer="local-test",
            criteria=criteria,
            evidence=(
                ("preflight", preflight),
                ("neutral", neutral),
                ("fresh_update", fresh),
                ("cold_resume", resumed),
            ),
        )


def test_g0_report_supports_prepublication_semantic_validation(
    tmp_path: Path,
    g0_report_factory,
) -> None:
    path = g0_report_factory(tmp_path / "g0-fresh.json", phase="fresh")
    report = json.loads(path.read_text())
    checkpoint = Path(report["checkpoint_dir"])
    checkpoint_report = checkpoint / "native_g0_report.json"
    checkpoint_report.unlink()
    checkpoint.rmdir()
    kwargs = {
        "schema": G0_REPORT_SCHEMA,
        "phase": "fresh",
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "world_size": 2,
    }

    assert validate_g0_report(report, **kwargs, require_checkpoint_copy=False) == report
    with pytest.raises(ValueError, match="does not contain its report copy"):
        validate_g0_report(report, **kwargs)


def test_g0_report_accepts_only_a_complete_positive_task_query_gradient_family(
    tmp_path: Path,
    g0_report_factory,
) -> None:
    path = g0_report_factory(tmp_path / "g0-ltop.json", phase="fresh")
    report = json.loads(path.read_text())
    kwargs = {
        "schema": G0_REPORT_SCHEMA,
        "phase": "fresh",
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "world_size": 2,
        "require_checkpoint_copy": False,
    }
    for rank_report in report["rank_reports"]:
        rank_report["gradient_metrics"].update(
            {
                "task_query_norm": 0.5,
                "task_query_elements": 32,
            }
        )

    assert validate_g0_report(report, **kwargs) == report

    incomplete = json.loads(json.dumps(report))
    del incomplete["rank_reports"][0]["gradient_metrics"]["task_query_elements"]
    with pytest.raises(ValueError, match="fields differ"):
        validate_g0_report(incomplete, **kwargs)

    zero = json.loads(json.dumps(report))
    zero["rank_reports"][0]["gradient_metrics"]["task_query_norm"] = 0.0
    with pytest.raises(ValueError, match="task_query_norm must be positive"):
        validate_g0_report(zero, **kwargs)


def test_g0_report_requires_an_explicit_nondefault_gpu_placement_contract(
    tmp_path: Path,
    g0_report_factory,
) -> None:
    path = g0_report_factory(tmp_path / "g0-gpu.json", phase="fresh")
    report = json.loads(path.read_text())
    report["fsdp2_placement"] = FSDP2_GPU_SHARDED
    report["parameter_storage"] = {
        "parameter_tensors": 2,
        "local_elements": 10,
        "master_dtype": "float32",
        "placement": FSDP2_GPU_SHARDED,
        "cpu_parameter_tensors": 0,
        "cpu_local_elements": 0,
        "cuda_parameter_tensors": 2,
        "cuda_local_elements": 10,
        "selective_cpu_parameter_names": [],
    }
    kwargs = {
        "schema": G0_REPORT_SCHEMA,
        "phase": "fresh",
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "world_size": 2,
        "require_checkpoint_copy": False,
    }

    with pytest.raises(ValueError, match="placement contract"):
        validate_g0_report(report, **kwargs)
    assert (
        validate_g0_report(
            report,
            **kwargs,
            expected_fsdp2_placement=FSDP2_GPU_SHARDED,
        )
        == report
    )


def test_g0_report_requires_an_explicit_nondefault_allocator_contract(
    tmp_path: Path,
    g0_report_factory,
) -> None:
    path = g0_report_factory(tmp_path / "g0-expandable.json", phase="fresh")
    report = json.loads(path.read_text())
    report["cuda_allocator"] = "expandable-segments"
    kwargs = {
        "schema": G0_REPORT_SCHEMA,
        "phase": "fresh",
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "world_size": 2,
        "require_checkpoint_copy": False,
    }

    with pytest.raises(ValueError, match="CUDA allocator contract"):
        validate_g0_report(report, **kwargs)
    assert (
        validate_g0_report(
            report,
            **kwargs,
            expected_cuda_allocator="expandable-segments",
        )
        == report
    )


def test_preflight_and_smoke_recompute_pass_fields(
    tmp_path: Path,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    preflight_path = preflight_report_factory(tmp_path / "preflight.json")
    preflight = json.loads(preflight_path.read_text())
    assert (
        validate_preflight_report(
            preflight,
            schema=PREFLIGHT_REPORT_SCHEMA,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=2,
        )
        == preflight
    )
    low_storage = json.loads(json.dumps(preflight))
    low_storage["hardware_capacity"]["free_storage_bytes"] = 250 * 2**30 - 1
    with pytest.raises(ValueError, match="capacity is insufficient"):
        validate_preflight_report(
            low_storage,
            schema=PREFLIGHT_REPORT_SCHEMA,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=2,
        )
    low_memory = json.loads(json.dumps(preflight))
    low_memory["hardware_capacity"]["host_memory_bytes"] = 128 * 2**30 - 1
    with pytest.raises(ValueError, match="capacity is insufficient"):
        validate_preflight_report(
            low_memory,
            schema=PREFLIGHT_REPORT_SCHEMA,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=2,
        )
    lowered_reserve = json.loads(json.dumps(preflight))
    lowered_reserve["hardware_capacity"]["minimum_free_storage_bytes"] -= 1
    with pytest.raises(ValueError, match="capacity is insufficient"):
        validate_preflight_report(
            lowered_reserve,
            schema=PREFLIGHT_REPORT_SCHEMA,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=2,
        )
    wrong_data_contract = json.loads(json.dumps(preflight))
    wrong_data_contract["g0_data"]["validation"]["dataset_full_tree_rescanned"] = True
    with pytest.raises(ValueError, match="verified-read contract"):
        validate_preflight_report(
            wrong_data_contract,
            schema=PREFLIGHT_REPORT_SCHEMA,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=2,
        )
    wrong_optimizer = json.loads(json.dumps(preflight))
    wrong_optimizer["static_checks"]["released_optimizer_contract"]["router_z_loss_coeff"] = 0.0
    with pytest.raises(ValueError, match="released optimizer contract differs"):
        validate_preflight_report(
            wrong_optimizer,
            schema=PREFLIGHT_REPORT_SCHEMA,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=2,
        )
    unsupported_complete = dict(preflight)
    unsupported_complete["complete_adr74_static_ready"] = False
    with pytest.raises(ValueError, match="unsupported completeness claim"):
        validate_preflight_report(
            unsupported_complete,
            schema=PREFLIGHT_REPORT_SCHEMA,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=2,
        )
    preflight["commands"][0]["returncode"] = 1
    with pytest.raises(ValueError, match="not recomputed"):
        validate_preflight_report(
            preflight,
            schema=PREFLIGHT_REPORT_SCHEMA,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=2,
        )

    preflight = json.loads(preflight_path.read_text())
    preflight["checkpoint"]["checkpoint_assets"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="asset identity is incomplete"):
        validate_preflight_report(
            preflight,
            schema=PREFLIGHT_REPORT_SCHEMA,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_id=LINGBOT_CHECKPOINT_ID,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_id=QWEN_PROCESSOR_ID,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
            world_size=2,
        )

    smoke_path = smoke_report_factory(tmp_path / "smoke.json")
    smoke = json.loads(smoke_path.read_text())
    implementation_sha256 = smoke["implementation_sha256"]
    assert (
        validate_full_weight_smoke_report(
            smoke,
            schema="picf-next.lingbot-vla2-native-full-weight-smoke.v4",
            implementation_sha256=implementation_sha256,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        )
        == smoke
    )
    smoke["implementation_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identity, status, or leak check failed"):
        validate_full_weight_smoke_report(
            smoke,
            schema="picf-next.lingbot-vla2-native-full-weight-smoke.v4",
            implementation_sha256=implementation_sha256,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        )

    smoke = json.loads(smoke_path.read_text())
    smoke["installed_neutral_action_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="booleans disagree"):
        validate_full_weight_smoke_report(
            smoke,
            schema="picf-next.lingbot-vla2-native-full-weight-smoke.v4",
            implementation_sha256=implementation_sha256,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        )

    smoke = json.loads(smoke_path.read_text())
    smoke["moe_inference_backend"]["selected"] = "robby_moe_forward"
    with pytest.raises(ValueError, match="deterministic MoE backend"):
        validate_full_weight_smoke_report(
            smoke,
            schema="picf-next.lingbot-vla2-native-full-weight-smoke.v4",
            implementation_sha256=implementation_sha256,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        )

    smoke = json.loads(smoke_path.read_text())
    smoke["official_repeat_action_bitwise_equal"] = False
    smoke["official_repeat_action_max_abs_error"] = 0.125
    with pytest.raises(ValueError, match="did not pass parity"):
        validate_full_weight_smoke_report(
            smoke,
            schema="picf-next.lingbot-vla2-native-full-weight-smoke.v4",
            implementation_sha256=implementation_sha256,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        )

    smoke = json.loads(smoke_path.read_text())
    smoke["processor_assets"][0]["bytes"] += 1
    with pytest.raises(ValueError, match="assets differ from the pinned revisions"):
        validate_full_weight_smoke_report(
            smoke,
            schema="picf-next.lingbot-vla2-native-full-weight-smoke.v4",
            implementation_sha256=implementation_sha256,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        )

    smoke = json.loads(smoke_path.read_text())
    smoke["targetless_routes"] = [[0, 1]]
    with pytest.raises(ValueError, match="targetless action route trace fields differ"):
        validate_full_weight_smoke_report(
            smoke,
            schema="picf-next.lingbot-vla2-native-full-weight-smoke.v4",
            implementation_sha256=implementation_sha256,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        )

    smoke = json.loads(smoke_path.read_text())
    smoke["alignment_teacher_prune"]["removed_numel"] = True
    with pytest.raises(ValueError, match="teacher-head elements must be a positive integer"):
        validate_full_weight_smoke_report(
            smoke,
            schema="picf-next.lingbot-vla2-native-full-weight-smoke.v4",
            implementation_sha256=implementation_sha256,
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_assets=asset_contract_manifest(CHECKPOINT_ASSET_CONTRACT),
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_assets=asset_contract_manifest(PROCESSOR_ASSET_CONTRACT),
        )


def _empirical_report(
    tmp_path: Path,
    gate: str,
    *,
    criteria: Path | None = None,
) -> dict[str, object]:
    root = tmp_path / gate.lower()
    root.mkdir(parents=True)
    if criteria is None:
        criteria = root / "criteria.md"
        criteria.write_text(f"frozen {gate} criteria")
    dataset = root / "dataset.json"
    split = root / "split.json"
    dataset.write_text('{"dataset":"fixture"}')
    split.write_text('{"split":"heldout"}')
    design = {
        "arms": list(EMPIRICAL_REQUIRED_ARMS[gate]),
        "paired_seed_count": 5,
        "bootstrap_replicates": 1_000,
        "bootstrap_seed": 741,
        "confidence_level": 0.95,
        "top_level_unit": "seed",
        "nested_units": ["task", "episode"],
        "aggregation": "equal_seed_task_episode_mean",
        "frames_treated_as_independent": False,
    }
    bounds = _acceptance_fixture(gate)
    metric_config = {"action_loss_threshold": 0.2} if gate == "G5" else {}
    plan = root / "evaluation-plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": EMPIRICAL_EVALUATION_PLAN_SCHEMA,
                "gate": gate,
                "design": design,
                "metric_config": metric_config,
                "acceptance_bounds": bounds,
                "required_checks": sorted(EMPIRICAL_REQUIRED_CHECKS[gate]),
            },
            sort_keys=True,
        )
    )
    protocol = {
        "criteria_path": str(criteria.resolve()),
        "criteria_sha256": _digest(criteria.read_bytes()),
        "dataset_manifest_path": str(dataset.resolve()),
        "dataset_manifest_sha256": _digest(dataset.read_bytes()),
        "split_manifest_path": str(split.resolve()),
        "split_manifest_sha256": _digest(split.read_bytes()),
        "evaluation_plan_path": str(plan.resolve()),
        "evaluation_plan_sha256": _digest(plan.read_bytes()),
        "preregistered_before_evaluation": True,
    }
    check_artifact = root / "check-evidence.json"
    check_artifact.write_text(
        json.dumps(
            {"checks": {name: True for name in EMPIRICAL_REQUIRED_CHECKS[gate]}},
            sort_keys=True,
        )
    )
    check_reference = {
        "path": str(check_artifact.resolve()),
        "sha256": _digest(check_artifact.read_bytes()),
    }
    episode_references = []
    for seed in range(5):
        for task_index in range(2):
            for episode_index in range(2):
                artifact = root / f"episode-{seed}-{task_index}-{episode_index}.npz"
                reference = write_empirical_episode_artifact(
                    artifact,
                    gate=gate,
                    arrays=(
                        _g2_no_object_arrays()
                        if gate == "G2" and task_index == 1
                        else _producer_arrays(gate)
                    ),
                )
                episode_references.append(
                    {
                        "seed": seed,
                        "task": f"task-{task_index}",
                        "episode": f"episode-{episode_index}",
                        **reference,
                    }
                )
    producer = root / "producer.json"
    write_empirical_producer_bundle(
        producer,
        bundle={
            "schema": "picf-next.lingbot-native-empirical-producer.v2",
            "gate": gate,
            "subject": _subject(),
            "protocol": protocol,
            "design": design,
            "check_evidence": {name: check_reference for name in EMPIRICAL_REQUIRED_CHECKS[gate]},
            "episodes": episode_references,
        },
    )
    raw_observations = build_empirical_observations_from_producer(
        producer,
        expected_sha256=_digest(producer.read_bytes()),
    )
    observations = root / "observations.json"
    observations.write_text(json.dumps(raw_observations, sort_keys=True))
    schema = (
        TRAINING_GATE_EVIDENCE_SCHEMAS[gate][1][1]
        if gate == "G2"
        else TRAINING_GATE_EVIDENCE_SCHEMAS[gate][0][1]
    )
    return build_empirical_gate_report_from_observations(
        observations,
        report_schema=schema,
        expected_sha256=_digest(observations.read_bytes()),
    )


@pytest.mark.parametrize("gate", ["G2", "G3", "G4", "G5", "G6"])
def test_empirical_gate_recomputes_paired_hierarchical_decisions(
    tmp_path: Path,
    gate: str,
) -> None:
    report = _empirical_report(tmp_path, gate)
    schema = (
        TRAINING_GATE_EVIDENCE_SCHEMAS[gate][1][1]
        if gate == "G2"
        else TRAINING_GATE_EVIDENCE_SCHEMAS[gate][0][1]
    )
    assert validate_empirical_gate_report(report, gate=gate, schema=schema) == report

    report["comparisons"][0]["ci_lower"] = -1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="not recomputed from raw observations"):
        validate_empirical_gate_report(report, gate=gate, schema=schema)


def test_empirical_gate_decision_is_bound_to_criteria_and_checkpoint(tmp_path: Path) -> None:
    criteria = tmp_path / "criteria.md"
    criteria.write_text("frozen G3 criteria")
    report = _empirical_report(tmp_path, "G3", criteria=criteria)
    evidence = tmp_path / "g3.json"
    evidence.write_text(json.dumps(report, sort_keys=True))
    decision = build_training_gate_decision(
        gate="G3",
        reviewer="owner",
        criteria=criteria,
        evidence=(("temporal_evaluation", evidence),),
    )
    path = tmp_path / "G3.decision.json"
    path.write_text(json.dumps(decision, sort_keys=True))
    assert load_training_gate_decision(path, expected_gate="G3")[1] == decision

    report["subject"] = {**_subject(), "saved_global_step": 121}
    evidence.write_text(json.dumps(report, sort_keys=True))
    with pytest.raises(ValueError, match="evidence differs"):
        load_training_gate_decision(path, expected_gate="G3")


def test_empirical_gate_rejects_observations_changed_after_report(tmp_path: Path) -> None:
    report = _empirical_report(tmp_path, "G3")
    observations = Path(report["observations"]["path"])  # type: ignore[index]
    raw = json.loads(observations.read_text())
    raw["records"][0]["candidate"] += 0.25
    observations.write_text(json.dumps(raw, sort_keys=True))

    with pytest.raises(ValueError, match="expected digest"):
        validate_empirical_gate_report(
            report,
            gate="G3",
            schema=TRAINING_GATE_EVIDENCE_SCHEMAS["G3"][0][1],
        )


def test_empirical_gate_rejects_episode_array_changed_after_report(tmp_path: Path) -> None:
    report = _empirical_report(tmp_path, "G3")
    observations = json.loads(Path(report["observations"]["path"]).read_text())  # type: ignore[index]
    producer = json.loads(Path(observations["producer"]["path"]).read_text())
    episode = Path(producer["episodes"][0]["path"])
    with np.load(episode, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["c_support"] = arrays["c_support"].copy()
    arrays["c_support"][0, 0, 0] = 0.25
    with episode.open("wb") as stream:
        cast(Any, np.savez_compressed)(stream, **arrays)

    with pytest.raises(ValueError, match="hash-bound real file"):
        validate_empirical_gate_report(
            report,
            gate="G3",
            schema=TRAINING_GATE_EVIDENCE_SCHEMAS["G3"][0][1],
        )


def test_empirical_gate_rejects_acceptance_plan_changed_after_report(tmp_path: Path) -> None:
    report = _empirical_report(tmp_path, "G4")
    plan = Path(report["protocol"]["evaluation_plan_path"])  # type: ignore[index]
    value = json.loads(plan.read_text())
    first = next(iter(value["acceptance_bounds"]))
    value["acceptance_bounds"][first] = -100.0
    plan.write_text(json.dumps(value, sort_keys=True))

    with pytest.raises(ValueError, match="hash-bound real file"):
        validate_empirical_gate_report(
            report,
            gate="G4",
            schema=TRAINING_GATE_EVIDENCE_SCHEMAS["G4"][0][1],
        )


def test_empirical_builder_rejects_manually_reversed_or_incomplete_records(
    tmp_path: Path,
) -> None:
    report = _empirical_report(tmp_path, "G5")
    observations = Path(report["observations"]["path"])  # type: ignore[index]
    raw = json.loads(observations.read_text())
    raw["records"][0]["candidate_label"] = "reversed"
    observations.write_text(json.dumps(raw, sort_keys=True))
    with pytest.raises(ValueError, match="not recomputed from the hash-bound producer"):
        build_empirical_gate_report_from_observations(
            observations,
            report_schema=TRAINING_GATE_EVIDENCE_SCHEMAS["G5"][0][1],
            expected_sha256=_digest(observations.read_bytes()),
        )

    raw["records"][0]["candidate_label"] = EMPIRICAL_COMPARISON_SPECS["G5"][
        raw["records"][0]["comparison"]
    ][2]
    first_comparison = raw["records"][0]["comparison"]
    raw["records"] = [
        record
        for record in raw["records"]
        if not (record["comparison"] == first_comparison and record["seed"] == 4)
    ]
    observations.write_text(json.dumps(raw, sort_keys=True))
    with pytest.raises(ValueError, match="not recomputed from the hash-bound producer"):
        build_empirical_gate_report_from_observations(
            observations,
            report_schema=TRAINING_GATE_EVIDENCE_SCHEMAS["G5"][0][1],
            expected_sha256=_digest(observations.read_bytes()),
        )


def test_empirical_checks_are_recomputed_from_hash_bound_artifacts(tmp_path: Path) -> None:
    report = _empirical_report(tmp_path, "G2")
    observations = Path(report["observations"]["path"])  # type: ignore[index]
    raw = json.loads(observations.read_text())
    check_name = next(iter(raw["check_evidence"]))
    check_artifact = Path(raw["check_evidence"][check_name]["path"])
    check_value = json.loads(check_artifact.read_text())
    check_value["checks"][check_name] = False
    check_artifact.write_text(json.dumps(check_value, sort_keys=True))

    with pytest.raises(ValueError, match="hash-bound real file"):
        validate_empirical_gate_report(
            report,
            gate="G2",
            schema=TRAINING_GATE_EVIDENCE_SCHEMAS["G2"][1][1],
        )

    changed_digest = _digest(check_artifact.read_bytes())
    producer_value = json.loads(Path(raw["producer"]["path"]).read_text())
    for reference in producer_value["check_evidence"].values():
        reference["sha256"] = changed_digest
    changed_producer = observations.parent / "producer-failed-check.json"
    write_empirical_producer_bundle(changed_producer, bundle=producer_value)
    changed_observations = build_empirical_observations_from_producer(
        changed_producer,
        expected_sha256=_digest(changed_producer.read_bytes()),
    )
    observations.write_text(json.dumps(changed_observations, sort_keys=True))
    failed = build_empirical_gate_report_from_observations(
        observations,
        report_schema=TRAINING_GATE_EVIDENCE_SCHEMAS["G2"][1][1],
        expected_sha256=_digest(observations.read_bytes()),
    )
    assert failed["status"] == "FAIL"
    assert f"check:{check_name}" in failed["failures"]


def test_empirical_builder_rejects_manually_unbalanced_episode_plan(tmp_path: Path) -> None:
    report = _empirical_report(tmp_path, "G6")
    observations = Path(report["observations"]["path"])  # type: ignore[index]
    raw = json.loads(observations.read_text())
    first = raw["records"][0]
    raw["records"] = [
        record
        for record in raw["records"]
        if not (
            record["comparison"] == first["comparison"]
            and record["seed"] == first["seed"]
            and record["task"] == first["task"]
            and record["episode"] == first["episode"]
        )
    ]
    observations.write_text(json.dumps(raw, sort_keys=True))

    with pytest.raises(ValueError, match="not recomputed from the hash-bound producer"):
        build_empirical_gate_report_from_observations(
            observations,
            report_schema=TRAINING_GATE_EVIDENCE_SCHEMAS["G6"][0][1],
            expected_sha256=_digest(observations.read_bytes()),
        )


def test_empirical_producer_rejects_reused_artifact_and_missing_seed(
    tmp_path: Path,
) -> None:
    report = _empirical_report(tmp_path, "G4")
    observations = json.loads(Path(report["observations"]["path"]).read_text())  # type: ignore[index]
    producer_path = Path(observations["producer"]["path"])
    original = json.loads(producer_path.read_text())

    reused = json.loads(producer_path.read_text())
    reused["episodes"][1]["path"] = reused["episodes"][0]["path"]
    reused["episodes"][1]["sha256"] = reused["episodes"][0]["sha256"]
    with pytest.raises(ValueError, match="reuses one episode artifact"):
        write_empirical_producer_bundle(
            producer_path.parent / "producer-reused.json",
            bundle=reused,
        )

    original["episodes"] = [episode for episode in original["episodes"] if episode["seed"] != 4]
    with pytest.raises(ValueError, match="does not cover every paired seed"):
        write_empirical_producer_bundle(
            producer_path.parent / "producer-missing-seed.json",
            bundle=original,
        )


def test_empirical_report_cli_publishes_recomputed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _empirical_report(tmp_path, "G6")
    observations = Path(expected["observations"]["path"])  # type: ignore[index]
    output = tmp_path / "g6-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_lingbot_native_empirical_report.py",
            "--gate",
            "G6",
            "--observations",
            str(observations),
            "--observations-sha256",
            _digest(observations.read_bytes()),
            "--output",
            str(output),
            "--require-pass",
        ],
    )

    empirical_report_tool.main()

    assert json.loads(output.read_text()) == expected
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_empirical_observations_cli_recomputes_producer_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _empirical_report(tmp_path, "G4")
    source = Path(report["observations"]["path"])  # type: ignore[index]
    expected = json.loads(source.read_text())
    producer = Path(expected["producer"]["path"])
    output = tmp_path / "recomputed-observations.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_lingbot_native_empirical_observations.py",
            "--producer",
            str(producer),
            "--producer-sha256",
            _digest(producer.read_bytes()),
            "--output",
            str(output),
        ],
    )

    empirical_observations_tool.main()

    assert json.loads(output.read_text()) == expected
    assert json.loads(capsys.readouterr().out)["gate"] == "G4"


def test_evaluation_plan_cli_requires_every_preregistered_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "g3-plan.json"
    argv = [
        "build_lingbot_native_evaluation_plan.py",
        "--gate",
        "G3",
        "--bootstrap-seed",
        "741",
        "--output",
        str(output),
    ]
    for name, rule in EMPIRICAL_COMPARISON_RULES["G3"].items():
        argv.extend(["--acceptance-bound", f"{name}={0.0 if rule.startswith('lower') else 0.2}"])
    monkeypatch.setattr(sys, "argv", argv)

    evaluation_plan_tool.main()

    value = json.loads(output.read_text())
    assert value["schema"] == EMPIRICAL_EVALUATION_PLAN_SCHEMA
    assert value["design"]["frames_treated_as_independent"] is False
    assert set(value["acceptance_bounds"]) == set(EMPIRICAL_COMPARISON_RULES["G3"])
    assert json.loads(capsys.readouterr().out)["gate"] == "G3"

    with pytest.raises(ValueError, match="coverage is incomplete"):
        evaluation_plan_tool._acceptance_bounds([], gate="G3")

    with pytest.raises(ValueError, match="differs from the frozen schema"):
        evaluation_plan_tool._metric_config([], gate="G5")
    assert evaluation_plan_tool._metric_config(["action_loss_threshold=0.07"], gate="G5") == {
        "action_loss_threshold": 0.07
    }


def test_g5_evaluation_plan_cli_freezes_threshold_and_every_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "g5-plan.json"
    argv = [
        "build_lingbot_native_evaluation_plan.py",
        "--gate",
        "G5",
        "--bootstrap-seed",
        "741",
        "--metric-config",
        "action_loss_threshold=0.07",
        "--output",
        str(output),
    ]
    for name, rule in EMPIRICAL_COMPARISON_RULES["G5"].items():
        bound = -0.05 if rule == "lower_ge" else 0.0 if rule == "upper_le" else 0.05
        argv.extend(["--acceptance-bound", f"{name}={bound}"])
    monkeypatch.setattr(sys, "argv", argv)

    evaluation_plan_tool.main()

    plan = json.loads(output.read_text())
    assert plan["metric_config"] == {"action_loss_threshold": 0.07}
    assert "action_C_vs_H" in plan["acceptance_bounds"]
    assert "raw_history_compute_memory_latency_matched" in plan["required_checks"]


def test_g2_visual_review_rehashes_every_artifact_and_coverage(tmp_path: Path) -> None:
    root = tmp_path / "visual-root"
    root.mkdir()
    artifacts = []
    for step_index, step in enumerate((20, 40, 60, 80, 100, 120)):
        for rank in (0, 1):
            relative = f"visuals/step_{step}/rank_{rank}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"step={step};rank={rank}".encode()
            path.write_bytes(payload)
            artifacts.append(
                {
                    "path": relative,
                    "sha256": _digest(payload),
                    "bytes": len(payload),
                    "global_step": step,
                    "rank": rank,
                    "sample_key": f"episode/{step}/{rank}",
                    "task": f"task-{step_index % 4}",
                    "status": "PASS",
                    "observations": "Object and task anchor panels are aligned.",
                }
            )
    report = {
        "schema": TRAINING_GATE_EVIDENCE_SCHEMAS["G2"][2][1],
        "status": "PASS",
        "gate": "G2",
        "subject": _subject(),
        "criteria_sha256": _digest(b"criteria"),
        "artifact_root": str(root.resolve()),
        "artifact_manifest_sha256": _canonical_digest(artifacts),
        "reviewer": "human-reviewer",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "coverage": {
            "artifact_count": 12,
            "ranks": [0, 1],
            "global_steps": [20, 40, 60, 80, 100, 120],
            "tasks": ["task-0", "task-1", "task-2", "task-3"],
        },
        "checks": {
            "all_artifacts_reviewed": True,
            "all_panels_legible": True,
            "object_alignment_acceptable": True,
            "task_anchor_alignment_acceptable": True,
            "no_catastrophic_off_object_collapse": True,
            "context_and_no_object_behavior_acceptable": True,
            "no_label_or_mask_input_leak": True,
        },
        "failures": [],
        "long_training_authorized": False,
    }
    schema = TRAINING_GATE_EVIDENCE_SCHEMAS["G2"][2][1]
    assert validate_g2_visual_review(report, schema=schema) == report
    (root / artifacts[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content differs"):
        validate_g2_visual_review(report, schema=schema)


def test_g7_protocol_requires_one_checkpoint_interface_and_executable_commands() -> None:
    schema = TRAINING_GATE_EVIDENCE_SCHEMAS["G7_PROTOCOL"][0][1]
    report = {
        "schema": schema,
        "status": "PASS",
        "gate": "G7_PROTOCOL",
        "subject": _subject(),
        "criteria_sha256": _digest(b"criteria"),
        "protocol_sha256": _digest(b"protocol"),
        "dataset_name": "LIBERO",
        "dataset_manifest_sha256": _digest(b"dataset"),
        "split_manifest_sha256": _digest(b"split"),
        "embodiment_schema_sha256": _digest(b"embodiment"),
        "interface_schema_sha256": _digest(b"interface"),
        "environment_lock_sha256": _digest(b"environment"),
        "registered_arms": ["A", "H", "M", "O", "C"],
        "registered_metrics": ["success", "recovery", "object_identity"],
        "paired_seed_count": 5,
        "checkpoint_policy": "single_checkpoint_interface",
        "adapter_policy": "typed_projection_only",
        "target_availability_audited": True,
        "executable_commands": ["python -m picf_next.eval.second_dataset"],
        "preregistered_before_evaluation": True,
        "failures": [],
        "long_training_authorized": False,
    }
    assert validate_g7_protocol(report, schema=schema) == report
    report["adapter_policy"] = "dataset_specific_controller"
    with pytest.raises(ValueError, match="not an executable"):
        validate_g7_protocol(report, schema=schema)
