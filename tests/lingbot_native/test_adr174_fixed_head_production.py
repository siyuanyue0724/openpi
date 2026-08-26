from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from picf_next.lingbot_native.action_posterior_learning import (
    action_posterior_target_mass_loss,
)
from tools.run_lingbot_vla2_ltop_adr172_direct_posterior import (
    ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES,
    ADR172_GUIDEDVLA_UPSTREAM_CONTRACT,
)
from tools.run_lingbot_vla2_ltop_core_pilot import (
    ADR172_ACTION_SUPERVISION_SCHEMA,
    ADR172_COLD_REPORT_SCHEMA,
    ADR172_COLD_VALIDATION_SCHEMA,
    ADR172_PICF_CRITICAL_SOURCE_FILES,
    ADR172_PICF_SOURCE_CONTRACT_SCHEMA,
    ADR172_RETENTION_REPORT_SCHEMA,
    ADR172_TRAINING_CHECKPOINT_FORMAT,
    ADR172_TRAINING_CHECKPOINT_SCHEMA,
    ADR172_TRAINING_MODEL_TREE_SCHEMA,
    ADR172_TRAINING_REPORT_SCHEMA,
    ADR174_FIXED_HEAD_INDICES,
    ADR174_FIXED_HEAD_LAYERS,
    ADR174_FIXED_HEAD_WEIGHT,
    G2_ARCHITECTURE,
    _cadence_for_mode,
    _fixed_head_objective_contract,
    _load_accepted_adr172_fixed_head_training,
    _load_adr172_fixed_head_evidence,
    _validate_adr172_training_evidence_binding,
)
from tools.run_lingbot_vla2_ltop_core_pilot import (
    ADR172_GUIDEDVLA_UPSTREAM_CONTRACT as PRODUCTION_UPSTREAM_CONTRACT,
)


def _source_contract() -> dict[str, object]:
    return {
        "schema": ADR172_PICF_SOURCE_CONTRACT_SCHEMA,
        "repository_commit": "1" * 40,
        "repository_tree": "2" * 40,
        "worktree_clean": True,
        "critical_file_sha256": {
            name: f"{index + 1:064x}"
            for index, name in enumerate(ADR172_PICF_CRITICAL_SOURCE_FILES)
        },
    }


def _training_contract() -> dict[str, object]:
    objective = _fixed_head_objective_contract()
    return {
        "optimizer": {"name": "test"},
        "deploy_time_module_added": False,
        "direct_posterior_adoption": {
            "route": objective["route"],
            "registered_layer_indices": objective["registered_layer_indices"],
            "head_scope": objective["head_scope"],
            "head_indices": objective["head_indices"],
            "upstream_contract": objective["upstream_contract"],
            "single_forward_per_optimizer_step": True,
            "deploy_time_module_added": False,
        },
        "loss_weights": {
            "official": 1.0,
            "physical_set": 1.0,
            "direct_grounding": ADR174_FIXED_HEAD_WEIGHT,
        },
    }


def _report(*, schema: str, phase: str) -> dict[str, object]:
    source = _source_contract()
    return {
        "schema": schema,
        "status": "PASS",
        "failures": [],
        "mode": "gate",
        "phase": phase,
        "world_size": 2,
        "architecture_identity": G2_ARCHITECTURE,
        "capacity": 16,
        "task_query_count": 4,
        "stage_checkpoint": "/mnt/stage",
        "trained_checkpoint": "/mnt/trained",
        "g2_report_sha256": "3" * 64,
        "runtime_source_contract": {
            "native_patch_sha256": "4" * 64,
            "runtime_hotfix_sha256": "5" * 64,
            "runtime_patched_source_sha256": {"model.py": "6" * 64},
        },
        "picf_source_contract": source,
        "trained_picf_source_contract": source,
        "dataset_contract": {"dataset": "calvin"},
        "execution_contract_sha256": "7" * 64,
        "offline_labels_sha256": "8" * 64,
        "physical_sidecar_manifest_sha256": "9" * 64,
        "training_contract": _training_contract(),
        "rank_reports": [
            {
                "rank": rank,
                "trained_checkpoint_model_tree_sha256": "a" * 64,
                "trained_model_local_state_sha256": character * 64,
            }
            for rank, character in enumerate(("b", "c"))
        ],
    }


def _write_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    cold = tmp_path / "cold.json"
    retention = tmp_path / "retention.json"
    cold.write_text(
        json.dumps(_report(schema=ADR172_COLD_REPORT_SCHEMA, phase="evaluation")),
        encoding="ascii",
    )
    retention.write_text(
        json.dumps(_report(schema=ADR172_RETENTION_REPORT_SCHEMA, phase="retention")),
        encoding="ascii",
    )
    validation = tmp_path / "cold-validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema": ADR172_COLD_VALIDATION_SCHEMA,
                "status": "PASS",
                "failures": [],
                "source_report": str(cold.resolve()),
                "source_report_sha256": hashlib.sha256(cold.read_bytes()).hexdigest(),
            }
        ),
        encoding="ascii",
    )
    return cold, validation, retention


def _write_training_acceptance(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = _source_contract()
    model_digests = ["b" * 64, "c" * 64]
    model_tree = "a" * 64
    manifest = {
        "schema": ADR172_TRAINING_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "format": ADR172_TRAINING_CHECKPOINT_FORMAT,
        "optimizer_saved": False,
        "world_size": 2,
        "global_step": 256,
        "action_supervision_schema": ADR172_ACTION_SUPERVISION_SCHEMA,
        "direct_action_causal_surface": "native-action-to-current-posterior-row-kv",
        "direct_grounding_weight": ADR174_FIXED_HEAD_WEIGHT,
        "direct_posterior_head_scope": "guidedvla-fixed-object-heads-0-1",
        "direct_posterior_head_indices": list(ADR174_FIXED_HEAD_INDICES),
        "direct_posterior_registered_layer_indices": list(ADR174_FIXED_HEAD_LAYERS),
        "direct_grounding_upstream_contract": ADR172_GUIDEDVLA_UPSTREAM_CONTRACT,
        "source_stage_checkpoint": "/mnt/stage",
        "g2_report_sha256": "3" * 64,
        "runtime_source_contract": {
            "native_patch_sha256": "4" * 64,
            "runtime_hotfix_sha256": "5" * 64,
            "runtime_patched_source_sha256": {"model.py": "6" * 64},
        },
        "picf_source_contract": source,
        "model_tree_schema": ADR172_TRAINING_MODEL_TREE_SCHEMA,
        "model_tree_sha256": model_tree,
        "training_final_model_local_state_sha256_by_rank": model_digests,
    }
    manifest_path = checkpoint / "ltop_g3_training_checkpoint.json"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    report = _report(schema=ADR172_TRAINING_REPORT_SCHEMA, phase="training")
    report.update(
        {
            "mode": "direct-trial",
            "steps": 256,
            "eval_every": 32,
            "direct_action_causal_surface": "native-action-to-current-posterior-row-kv",
            "checkpoint": {
                "format": ADR172_TRAINING_CHECKPOINT_FORMAT,
                "optimizer_saved": False,
                "action_supervision_schema": ADR172_ACTION_SUPERVISION_SCHEMA,
                "direct_grounding_weight": ADR174_FIXED_HEAD_WEIGHT,
                "direct_posterior_head_scope": "guidedvla-fixed-object-heads-0-1",
                "direct_posterior_head_indices": list(ADR174_FIXED_HEAD_INDICES),
                "direct_posterior_registered_layer_indices": list(ADR174_FIXED_HEAD_LAYERS),
                "model_tree_schema": ADR172_TRAINING_MODEL_TREE_SCHEMA,
                "model_tree_sha256": model_tree,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "path": str(checkpoint.resolve()),
                "picf_source_contract": source,
                "training_final_model_local_state_sha256_by_rank": model_digests,
            },
        }
    )
    path = tmp_path / "training.json"
    path.write_text(json.dumps(report), encoding="ascii")
    return path


def test_production_contract_exactly_matches_validated_adr172_primitive() -> None:
    contract = _fixed_head_objective_contract()
    assert ADR174_FIXED_HEAD_LAYERS == (32, 35)
    assert ADR174_FIXED_HEAD_INDICES == ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES == (0, 1)
    assert ADR174_FIXED_HEAD_WEIGHT == 0.001
    assert PRODUCTION_UPSTREAM_CONTRACT == ADR172_GUIDEDVLA_UPSTREAM_CONTRACT
    assert contract["replaces_old_task_address_objective"] is True
    assert contract["deploy_time_module_added"] is False


def test_only_selected_native_action_heads_receive_objective_gradient() -> None:
    logits = torch.randn(1, 4, 2, 3, requires_grad=True)
    attention = logits.softmax(dim=-1)
    result = action_posterior_target_mass_loss(
        attention,
        target_row_weights=torch.tensor([[1.0, 0.0, 0.0]]),
        target_valid=torch.tensor([True]),
        head_indices=torch.tensor(ADR174_FIXED_HEAD_INDICES),
    )
    result.loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, :2]).item() > 0
    assert torch.count_nonzero(logits.grad[:, 2:]).item() == 0


def test_evidence_loader_byte_binds_cold_and_retention_provenance(tmp_path: Path) -> None:
    cold, validation, retention = _write_evidence(tmp_path)
    evidence = _load_adr172_fixed_head_evidence(
        cold_report_path=cold,
        cold_validation_path=validation,
        retention_report_path=retention,
    )
    assert evidence["cold_report_sha256"] == hashlib.sha256(cold.read_bytes()).hexdigest()
    assert evidence["physical_retention_report_sha256"] == hashlib.sha256(
        retention.read_bytes()
    ).hexdigest()
    assert evidence["objective"] == _fixed_head_objective_contract()

    cold.write_text(cold.read_text(encoding="ascii") + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="not byte-bound"):
        _load_adr172_fixed_head_evidence(
            cold_report_path=cold,
            cold_validation_path=validation,
            retention_report_path=retention,
        )


def test_evidence_loader_rejects_upstream_or_fixed_head_drift(tmp_path: Path) -> None:
    cold, validation, retention = _write_evidence(tmp_path)
    payload = json.loads(retention.read_text(encoding="ascii"))
    payload["training_contract"]["direct_posterior_adoption"]["upstream_contract"][
        "repository_commit"
    ] = "0" * 40
    retention.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="adoption contract differs"):
        _load_adr172_fixed_head_evidence(
            cold_report_path=cold,
            cold_validation_path=validation,
            retention_report_path=retention,
        )


def test_training_acceptance_byte_binds_exact_adr172_checkpoint(tmp_path: Path) -> None:
    path = _write_training_acceptance(tmp_path)
    accepted = _load_accepted_adr172_fixed_head_training(path)
    assert accepted.checkpoint_path == (tmp_path / "checkpoint").resolve()
    assert accepted.checkpoint_model_tree_sha256 == "a" * 64
    assert accepted.training_final_model_local_state_sha256_by_rank == (
        "b" * 64,
        "c" * 64,
    )

    manifest = tmp_path / "checkpoint" / "ltop_g3_training_checkpoint.json"
    manifest.write_text(manifest.read_text(encoding="ascii") + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="not byte-bound"):
        _load_accepted_adr172_fixed_head_training(path)


def test_training_acceptance_and_cold_evidence_must_name_same_model(tmp_path: Path) -> None:
    accepted = _load_accepted_adr172_fixed_head_training(_write_training_acceptance(tmp_path))
    report = accepted.report
    evidence = {
        "trained_checkpoint": str(accepted.checkpoint_path),
        "trained_checkpoint_identity": {
            "model_tree_sha256": accepted.checkpoint_model_tree_sha256,
            "model_local_state_sha256_by_rank": list(
                accepted.training_final_model_local_state_sha256_by_rank
            ),
        },
        "runtime_source_contract": report["runtime_source_contract"],
        "dataset_contract": report["dataset_contract"],
        "stage_checkpoint": report["stage_checkpoint"],
        "g2_report_sha256": report["g2_report_sha256"],
        "execution_contract_sha256": report["execution_contract_sha256"],
        "offline_labels_sha256": report["offline_labels_sha256"],
        "physical_sidecar_manifest_sha256": report["physical_sidecar_manifest_sha256"],
        "training_contract": report["training_contract"],
        "trained_picf_source_contract": report["trained_picf_source_contract"],
    }
    _validate_adr172_training_evidence_binding(accepted, evidence)
    evidence["trained_checkpoint"] = "/mnt/another-checkpoint"
    with pytest.raises(ValueError, match="differ at trained_checkpoint"):
        _validate_adr172_training_evidence_binding(accepted, evidence)


def test_runner_replaces_old_task_address_objective_without_stacking() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = (repository_root / "tools/run_lingbot_vla2_ltop_core_pilot.py").read_text()
    assert "action_posterior_target_mass_loss(" in source
    assert "task_address_target_coverage" not in source
    assert "task_address_weight" not in source
    assert '"task_address_loss"' not in source


def test_restart_smoke_is_two_then_cold_resume_then_four() -> None:
    cadence = _cadence_for_mode("restart-smoke")
    assert cadence.total_steps == 4
    assert [step for step in range(5) if cadence.checkpoint_due(step)] == [2, 4]
    repository_root = Path(__file__).resolve().parents[2]
    launcher = (repository_root / "adr174/run_fixed_head_ltop_production_2gpu.sh").read_text()
    wrapper = (repository_root / "adr174/run_fixed_head_ltop_restart_smoke_2gpu.sh").read_text()
    assert "total_steps=4" in launcher
    assert "segment_steps=2" in launcher
    assert "current_phase=resume" in launcher
    assert "resume_${total_steps}_verify.log" in launcher
    assert "--action-information-set-policy factual-only" in launcher
    assert "--adr172-cold-validation" in launcher
    assert "adr172-g2b-callback-compatible-2b1b5da-v1" in launcher
    assert "lingbot-vla-v2-adr172-callback-v1" in launcher
    assert "adr160-g2b-confirm" not in launcher
    assert "PICF_LTOP_MODE=restart-smoke" in wrapper
