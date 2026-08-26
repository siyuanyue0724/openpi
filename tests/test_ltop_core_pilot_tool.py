from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from picf_next.lingbot_native.ltop_core_pilot import (
    LTOP_CORE_LONG_TOTAL_STEPS,
    LTOPCoreLongCadence,
    LTOPCorePilotArm,
)
from picf_next.lingbot_native.training import NativeParameterManifest
from tools.run_lingbot_vla2_ltop_core_pilot import (
    CORE_PILOT_CHECKPOINT_EXTRA_SCHEMA,
    CORE_PILOT_CHECKPOINT_PROVENANCE_SCHEMA,
    CORE_PILOT_CHECKPOINT_SCHEMA,
    CORE_PILOT_COLD_RESUME_SCHEMA,
    _action_information_set_for_step,
    _action_information_set_metric_summary,
    _all_gather_checkpoint_provenance_rank_receipts,
    _arm_contract_for_mode,
    _checkpoint_provenance_sha256,
    _detached_prior_boundary,
    _expected_action_information_set_counts,
    _forward_input_receipt,
    _long_action_information_set_schedule_contract,
    _optimizer_initialization_receipt,
    _prepare_rank_metric_journal,
    _prune_superseded_long_checkpoints,
    _require_accepted_g3_dataset_contract,
    _require_rolling_checkpoint_capacity,
    _resolve_run_interval,
    _scientific_boundary_for_mode,
    _tensor_mapping_sha256,
    _validate_checkpoint_manifest,
    _validate_checkpoint_provenance_rank_receipts,
    _validate_long_action_information_set_schedule_contract,
    _validate_resume_extra,
)


def _source() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "tools/run_lingbot_vla2_ltop_core_pilot.py").read_text(encoding="utf-8")


def test_core_pilot_is_fixed_head_evidence_gated_and_restores_the_exact_stage() -> None:
    source = _source()

    assert "load_accepted_g3_gate" in source
    assert "_load_accepted_adr172_fixed_head_training" in source
    assert "_load_adr172_fixed_head_evidence" in source
    assert "_validate_adr172_training_evidence_binding" in source
    assert "_validate_adr172_fixed_head_runtime_binding" in source
    assert "load_accepted_g3_source_aligned_gate" not in source
    assert "prepare_lingbot_vla2_ltop_stage_transfer" in source
    assert "open_lingbot_vla2_ltop_stage_runtime" in source
    assert 'accepted_g3.report.get("g2_report_sha256")' in source
    assert 'accepted_g3.report.get("stage_checkpoint")' in source


def test_core_pilot_consumes_the_frozen_physical_2k_stream() -> None:
    source = _source()

    assert "CalvinPhysicalTransitionDataset" in source
    assert "load_frozen_episode_stream_plan" in source
    assert "build_native_calvin_physical_episode_domain" in source
    assert "build_planned_native_calvin_batch" in source
    assert "plan.total_steps != cadence.total_steps" in source
    assert "evaluation overlaps a training source episode" in source


def test_core_pilot_fixed_head_objective_uses_native_action_attention() -> None:
    source = _source()
    objective = source[
        source.index("def objective_for_batch(") : source.index("def publish_training_visual(")
    ]

    assert "run_native_policy_training_forward" in source
    assert "build_lingbot_official_optimizer" in source
    assert "ObjectReadActionIntervention.FACTUAL" in source
    assert "ObjectReadActionIntervention.BLOCKED" in source
    assert "official_loss_weight * result.official_total_loss" in objective
    assert 'physical_set_weight * physical["set_loss"].total' in objective
    assert "ADR174_FIXED_HEAD_WEIGHT * action_posterior_loss" in objective
    assert "action_posterior_target_mass_loss" in objective
    assert "RegisteredActionPosteriorReceiptCollector" in source
    assert "resolve_task_address_target_row" in source
    assert 'allow_unobservable=True' in source
    assert '"action_posterior_supervision_reason"' in source
    assert "task_address_weight * address_loss" not in objective
    assert "action_consumable_task_address" not in source
    assert "row_mass_by_layer" not in source
    assert 'captured["final_row_mass"]' not in source
    assert "graph.config.num_layers" in source
    assert "LTOP core-pilot target identity is unbound" not in source
    assert "torch.nn.Linear" not in source
    assert "nn.Linear" not in source


def test_core_pilot_reuses_training_forward_for_visuals_and_one_final_checkpoint() -> None:
    source = _source()

    assert "LTOPCorePilotCadence()" in source
    assert "LTOPCorePilotSmokeCadence()" in source
    assert "publish_metrics_window" in source
    assert "publish_training_visual" in source
    assert "render_task_independent_entity_visuals" in source
    assert "policy.sample_actions(" not in source
    assert "run_native_policy_diagnostic_forward" not in source
    assert '"extra_model_forward": False' in source
    assert '"weight_boundary": "pre_update_training_forward"' in source
    assert 'f".global_step_{step}.incomplete"' in source
    assert "if not cadence.checkpoint_due(step)" in source
    assert "_require_rolling_checkpoint_capacity(checkpoint_root)" in source
    loop = source[source.index("for optimizer_step in range(") :]
    assert loop.index("publish_training_visual(") < loop.index("publish_checkpoint(")


def test_core_pilot_saves_then_delegates_causal_evaluation_to_a_fresh_process() -> None:
    source = _source()
    loop_and_tail = source[source.index("for optimizer_step in range(") :]

    assert "checkpoint_report = publish_checkpoint(" in loop_and_tail
    assert "record(cadence.total_steps)" not in loop_and_tail
    assert '"surface": "separate-fresh-process-evaluator"' in source
    assert '"executed_in_training_process": False' in source
    assert "if run_lease is not None:" in source


def test_core_pilot_emits_strict_pairing_receipts() -> None:
    source = _source()

    for field in (
        '"model_input_sha256"',
        '"controls_sha256"',
        '"prior_controls_sha256"',
        '"structural_targets_sha256"',
        '"normalized_forward_input_sha256"',
        '"forward_input_sha256"',
        '"executed_object_read_action_intervention"',
        '"executed_action_information_set"',
        '"optimizer_initialization"',
        '"source_identity"',
        '"runtime_environment_contract"',
        '"journal"',
    ):
        assert field in source


def test_core_pilot_resume_uses_full_native_dcp_and_prunes_only_after_cold_pass() -> None:
    source = _source()
    resume = source[source.index('if args.phase == "resume":') : source.index("def collate(")]

    assert "require_lingbot_exact_resume_contract(runtime.optimizer_contract)" in source
    assert 'state = {"model": policy, "optimizer": optimizer, "extra_state": {}}' in resume
    assert "checkpointer.load(str(checkpoint_dir), state)" in resume
    assert "_validate_resume_extra(" in resume
    assert "_validate_optimizer_state(" in resume
    assert "_restore_rank_rng(" in resume
    assert "resume_runtime_rng_verified" in resume
    assert "_prepare_rank_metric_journal(" in resume
    assert resume.index("_write_or_validate_cold_resume_receipt(") < resume.index(
        "_prune_superseded_long_checkpoints("
    )


def test_checkpoint_save_gathers_provenance_before_dcp_write() -> None:
    source = _source()
    checkpoint = source[
        source.index("def publish_checkpoint(") : source.index("invocation_report_path =")
    ]

    assert checkpoint.index("_all_gather_checkpoint_provenance_rank_receipts(") < checkpoint.index(
        "checkpointer.save("
    )
    assert '"provenance_rank_receipts": save_provenance_rank_receipts' in checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance() -> dict[str, object]:
    return {
        "schema": CORE_PILOT_CHECKPOINT_PROVENANCE_SCHEMA,
        "source_identity": {"picf_commit": "a" * 40},
    }


def _boundary() -> dict[str, str]:
    return {
        "lane_snapshot_sha256": "a" * 64,
        "model_local_state_sha256": "b" * 64,
        "optimizer_local_state_sha256": "c" * 64,
        "rank_rng_state_sha256": "d" * 64,
    }


def test_long_checkpoint_pruning_requires_cold_successor_receipt(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    predecessor = checkpoint_root / "global_step_2000"
    successor = checkpoint_root / "global_step_4000"
    predecessor.mkdir(parents=True)
    successor.mkdir()
    (predecessor / "ltop_core_pilot_checkpoint.json").write_text(
        '{"global_step":2000}\n',
        encoding="ascii",
    )
    (successor / "ltop_core_pilot_checkpoint.json").write_text(
        '{"global_step":4000}\n',
        encoding="ascii",
    )

    receipt = checkpoint_root / "cold_resume_receipts/global_step_4000.json"
    with pytest.raises(FileNotFoundError, match="verification receipt"):
        _prune_superseded_long_checkpoints(
            checkpoint_root,
            verified_step=4_000,
            verification_receipt=receipt,
        )
    assert predecessor.is_dir()

    receipt.parent.mkdir()
    receipt.write_text(
        json.dumps(
            {
                "schema": CORE_PILOT_COLD_RESUME_SCHEMA,
                "status": "PASS",
                "global_step": 4_000,
                "checkpoint_manifest_sha256": _sha256(
                    successor / "ltop_core_pilot_checkpoint.json"
                ),
            }
        ),
        encoding="ascii",
    )
    receipts = _prune_superseded_long_checkpoints(
        checkpoint_root,
        verified_step=4_000,
        verification_receipt=receipt,
    )

    assert len(receipts) == 1
    assert not predecessor.exists()
    assert successor.is_dir()
    assert (checkpoint_root / "pruned_receipts/global_step_2000.json").is_file()


def test_long_checkpoint_pruning_recovers_a_pre_receipt_tombstone(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    predecessor = checkpoint_root / "global_step_2000"
    successor = checkpoint_root / "global_step_4000"
    predecessor.mkdir(parents=True)
    successor.mkdir()
    (predecessor / "ltop_core_pilot_checkpoint.json").write_text(
        '{"global_step":2000}\n',
        encoding="ascii",
    )
    successor_manifest = successor / "ltop_core_pilot_checkpoint.json"
    successor_manifest.write_text('{"global_step":4000}\n', encoding="ascii")
    receipt = checkpoint_root / "cold_resume_receipts/global_step_4000.json"
    receipt.parent.mkdir()
    receipt.write_text(
        json.dumps(
            {
                "schema": CORE_PILOT_COLD_RESUME_SCHEMA,
                "status": "PASS",
                "global_step": 4_000,
                "checkpoint_manifest_sha256": _sha256(successor_manifest),
            }
        ),
        encoding="ascii",
    )
    tombstone = checkpoint_root / ".global_step_2000.pruned_by_4000"
    os.replace(predecessor, tombstone)

    receipts = _prune_superseded_long_checkpoints(
        checkpoint_root,
        verified_step=4_000,
        verification_receipt=receipt,
    )

    assert len(receipts) == 1
    assert not tombstone.exists()
    assert (checkpoint_root / "pruned_receipts/global_step_2000.json").is_file()


def test_rolling_capacity_uses_measured_predecessor_plus_margin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "global_step_2000").mkdir(parents=True)
    monkeypatch.setattr(
        "tools.run_lingbot_vla2_ltop_core_pilot._checkpoint_tree_size",
        lambda _path: 70 * 2**30,
    )
    monkeypatch.setattr(
        "tools.run_lingbot_vla2_ltop_core_pilot.require_checkpoint_write_capacity",
        lambda _path: 85 * 2**30,
    )
    with pytest.raises(RuntimeError, match="86.00 GiB is required"):
        _require_rolling_checkpoint_capacity(checkpoint_root)

    monkeypatch.setattr(
        "tools.run_lingbot_vla2_ltop_core_pilot.require_checkpoint_write_capacity",
        lambda _path: 86 * 2**30,
    )
    report = _require_rolling_checkpoint_capacity(checkpoint_root)
    assert report == {
        "free_bytes": 86 * 2**30,
        "reference_checkpoint_bytes": 70 * 2**30,
        "required_free_bytes": 86 * 2**30,
    }


def test_rolling_capacity_reserves_a_real_first_full_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    monkeypatch.setattr(
        "tools.run_lingbot_vla2_ltop_core_pilot.require_checkpoint_write_capacity",
        lambda _path: 79 * 2**30,
    )
    with pytest.raises(RuntimeError, match="80.00 GiB is required"):
        _require_rolling_checkpoint_capacity(checkpoint_root)

    monkeypatch.setattr(
        "tools.run_lingbot_vla2_ltop_core_pilot.require_checkpoint_write_capacity",
        lambda _path: 80 * 2**30,
    )
    assert _require_rolling_checkpoint_capacity(checkpoint_root) == {
        "free_bytes": 80 * 2**30,
        "reference_checkpoint_bytes": None,
        "required_free_bytes": 80 * 2**30,
    }


def test_rank_journal_resume_truncates_tail_and_requires_contiguous_steps(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "metrics/rank_journal/rank_0.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        "".join(
            json.dumps({"global_step": step, "source_digest": f"stream-{step}"}) + "\n"
            for step in range(1, 6)
        ),
        encoding="ascii",
    )

    handle = _prepare_rank_metric_journal(
        journal,
        phase="resume",
        load_global_step=3,
        expected_boundary_source_digest="stream-3",
    )
    handle.write(json.dumps({"global_step": 4, "source_digest": "stream-4"}) + "\n")
    handle.close()
    assert [json.loads(line)["global_step"] for line in journal.read_text().splitlines()] == [
        1,
        2,
        3,
        4,
    ]

    journal.write_text(
        '{"global_step":1,"source_digest":"stream-1"}\n'
        '{"global_step":3,"source_digest":"stream-3"}\n',
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="contiguous"):
        _prepare_rank_metric_journal(
            journal,
            phase="resume",
            load_global_step=3,
            expected_boundary_source_digest="stream-3",
        )

    journal.write_text(
        '{"global_step":1,"source_digest":"stream-1"}\n'
        '{"global_step":2,"source_digest":"wrong-stream"}\n',
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="frozen stream boundary differs"):
        _prepare_rank_metric_journal(
            journal,
            phase="resume",
            load_global_step=2,
            expected_boundary_source_digest="stream-2",
        )


def test_resume_contract_binds_manifest_extra_stream_and_provenance() -> None:
    provenance = _provenance()
    provenance_sha256 = _checkpoint_provenance_sha256(provenance)
    manifest = {
        "schema": CORE_PILOT_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": 2_000,
        "next_optimizer_step": 2_000,
        "world_size": 2,
        "arm": "ltop-ec-factual",
        "provenance": provenance,
        "provenance_sha256": provenance_sha256,
        "provenance_rank_receipts": [
            {"rank": 0, "checkpoint_provenance_sha256": provenance_sha256},
            {"rank": 1, "checkpoint_provenance_sha256": provenance_sha256},
        ],
        "rank_boundaries": [
            {"rank": 0, "boundary": _boundary()},
            {"rank": 1, "boundary": _boundary()},
        ],
    }
    assert (
        _validate_checkpoint_manifest(
            manifest,
            expected_global_step=2_000,
            expected_arm="ltop-ec-factual",
            expected_provenance=provenance,
        )
        == manifest
    )
    extra = {
        "schema": CORE_PILOT_CHECKPOINT_EXTRA_SCHEMA,
        "rank": 0,
        "world_size": 2,
        "global_step": 2_000,
        "next_optimizer_step": 2_000,
        "source_digest": "stream-step-1999",
        "provenance": provenance,
        "provenance_sha256": provenance_sha256,
        "rank_rng_state": {"opaque": b"rng"},
        "lane_snapshot": _detached_prior_boundary(2_000),
        "boundary_sha256": _boundary(),
        "optimizer_state_entries": 2,
        "optimizer_local_moment_elements": 4,
    }
    assert (
        _validate_resume_extra(
            extra,
            expected_global_step=2_000,
            expected_source_digest="stream-step-1999",
            expected_provenance=provenance,
            rank=0,
        )
        == extra
    )
    with pytest.raises(ValueError, match="stream boundary"):
        _validate_resume_extra(
            extra,
            expected_global_step=2_000,
            expected_source_digest="another-stream",
            expected_provenance=provenance,
            rank=0,
        )


def test_run_interval_allows_only_registered_fresh_resume_boundaries() -> None:
    cadence = LTOPCoreLongCadence()
    assert _resolve_run_interval(
        argparse.Namespace(phase="fresh", load_global_step=0, stop_after_step=2_000),
        cadence,
    ) == (0, 2_000)
    assert _resolve_run_interval(
        argparse.Namespace(phase="resume", load_global_step=2_000, stop_after_step=4_000),
        cadence,
    ) == (2_000, 4_000)
    assert _resolve_run_interval(
        argparse.Namespace(phase="resume", load_global_step=4_000, stop_after_step=4_000),
        cadence,
    ) == (4_000, 4_000)
    with pytest.raises(ValueError, match="registered checkpoint boundary"):
        _resolve_run_interval(
            argparse.Namespace(phase="resume", load_global_step=1_999, stop_after_step=4_000),
            cadence,
        )


def test_normalized_forward_receipt_removes_only_the_registered_intervention() -> None:
    input_receipt = {
        "schema": "input",
        "model_input_sha256": "a" * 64,
        "controls_sha256": "b" * 64,
        "prior_controls_sha256": "c" * 64,
        "structural_targets_sha256": "d" * 64,
    }

    factual = _forward_input_receipt(input_receipt, intervention="factual")
    blocked = _forward_input_receipt(input_receipt, intervention="blocked")

    assert factual["normalized_forward_input_sha256"] == blocked["normalized_forward_input_sha256"]
    assert factual["forward_input_sha256"] != blocked["forward_input_sha256"]


def test_long_information_sets_are_rank_counterbalanced_across_resume_boundary() -> None:
    values = {}
    for step in (0, 1, 1_999, 2_000):
        values[step] = tuple(
            _action_information_set_for_step(
                policy="rank-step-counterbalanced-50-50",
                optimizer_step=step,
                rank=rank,
                factual="factual",
                mediator_required="mediator-required",
            )
            for rank in (0, 1)
        )
        assert set(values[step]) == {"factual", "mediator-required"}
    assert values[0] != values[1]
    assert values[1_999] != values[2_000]
    assert _expected_action_information_set_counts(
        policy="rank-step-counterbalanced-50-50",
        load_global_step=0,
        stop_global_step=2_000,
        rank=0,
    ) == {"factual": 1_000, "mediator-required": 1_000}
    assert _expected_action_information_set_counts(
        policy="rank-step-counterbalanced-50-50",
        load_global_step=2_000,
        stop_global_step=4_000,
        rank=1,
    ) == {"factual": 1_000, "mediator-required": 1_000}


def test_long_schedule_contract_is_content_addressed_and_covers_all_30k_steps() -> None:
    contract = _long_action_information_set_schedule_contract()

    assert _validate_long_action_information_set_schedule_contract(contract) == contract
    assert contract["total_steps"] == LTOP_CORE_LONG_TOTAL_STEPS == 30_000
    assert contract["world_size"] == 2
    assert contract["zero_based_domain"] == {
        "optimizer_step": {"start_inclusive": 0, "stop_exclusive": 30_000},
        "rank": {"start_inclusive": 0, "stop_exclusive": 2},
    }
    assert contract["formula"] == {
        "expression": "(optimizer_step + rank) % 2",
        "remainder_to_executed_information_set": {
            "0": "factual",
            "1": "mediator-required",
        },
    }
    observed = {
        0: {"factual": 0, "mediator-required": 0},
        1: {"factual": 0, "mediator-required": 0},
    }
    for optimizer_step in range(LTOP_CORE_LONG_TOTAL_STEPS):
        per_rank = tuple(
            _action_information_set_for_step(
                policy="rank-step-counterbalanced-50-50",
                optimizer_step=optimizer_step,
                rank=rank,
                factual="factual",
                mediator_required="mediator-required",
            )
            for rank in range(2)
        )
        assert set(per_rank) == {"factual", "mediator-required"}
        for rank, information_set in enumerate(per_rank):
            observed[rank][information_set] += 1
    assert contract["per_rank_counts"] == [{"rank": rank, **observed[rank]} for rank in range(2)]

    tampered = copy.deepcopy(contract)
    tampered["per_rank_counts"][0]["factual"] -= 1
    with pytest.raises(ValueError, match="digest differs"):
        _validate_long_action_information_set_schedule_contract(tampered)


def test_long_schedule_every_prefix_and_suffix_add_to_the_30k_contract() -> None:
    for rank in range(2):
        total = _expected_action_information_set_counts(
            policy="rank-step-counterbalanced-50-50",
            load_global_step=0,
            stop_global_step=LTOP_CORE_LONG_TOTAL_STEPS,
            rank=rank,
        )
        for split in range(LTOP_CORE_LONG_TOTAL_STEPS + 1):
            prefix = _expected_action_information_set_counts(
                policy="rank-step-counterbalanced-50-50",
                load_global_step=0,
                stop_global_step=split,
                rank=rank,
            )
            suffix = _expected_action_information_set_counts(
                policy="rank-step-counterbalanced-50-50",
                load_global_step=split,
                stop_global_step=LTOP_CORE_LONG_TOTAL_STEPS,
                rank=rank,
            )
            assert {
                information_set: prefix[information_set] + suffix[information_set]
                for information_set in total
            } == total


def test_accepted_g3_dataset_contract_must_equal_runtime_exactly() -> None:
    contract = {
        "dataset_id": "calvin",
        "dataset_revision": "abc123",
        "dataset_tree_sha256": "a" * 64,
        "split_name": "training",
    }

    assert (
        _require_accepted_g3_dataset_contract(
            accepted_dataset_contract=copy.deepcopy(contract),
            runtime_dataset_contract=contract,
        )
        == contract
    )
    changed = copy.deepcopy(contract)
    changed["split_name"] = "validation"
    with pytest.raises(ValueError, match="differs from runtime"):
        _require_accepted_g3_dataset_contract(
            accepted_dataset_contract=contract,
            runtime_dataset_contract=changed,
        )
    with pytest.raises(ValueError, match="accepted G3 dataset contract is absent"):
        _require_accepted_g3_dataset_contract(
            accepted_dataset_contract=None,
            runtime_dataset_contract=contract,
        )


def test_checkpoint_provenance_all_gather_requires_exact_cross_rank_equality() -> None:
    digest = "a" * 64
    receipts = [
        {"rank": 1, "checkpoint_provenance_sha256": digest},
        {"rank": 0, "checkpoint_provenance_sha256": digest},
    ]
    expected = [
        {"rank": 0, "checkpoint_provenance_sha256": digest},
        {"rank": 1, "checkpoint_provenance_sha256": digest},
    ]

    class FakeDistributed:
        def __init__(self, gathered: list[dict[str, object]]) -> None:
            self.gathered = gathered
            self.local: dict[str, object] | None = None

        def all_gather_object(self, output: list[object], local: dict[str, object]) -> None:
            self.local = local
            output[:] = copy.deepcopy(self.gathered)

    distributed = FakeDistributed(receipts)
    assert (
        _all_gather_checkpoint_provenance_rank_receipts(
            distributed=distributed,
            rank=0,
            checkpoint_provenance_sha256=digest,
        )
        == expected
    )
    assert distributed.local == {
        "rank": 0,
        "checkpoint_provenance_sha256": digest,
    }
    assert (
        _validate_checkpoint_provenance_rank_receipts(
            receipts,
            expected_provenance_sha256=digest,
        )
        == expected
    )

    mismatched = copy.deepcopy(receipts)
    mismatched[0]["checkpoint_provenance_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="differs across ranks"):
        _all_gather_checkpoint_provenance_rank_receipts(
            distributed=FakeDistributed(mismatched),
            rank=0,
            checkpoint_provenance_sha256=digest,
        )


def test_arm_contract_start_state_depends_on_execution_generation() -> None:
    arm = LTOPCorePilotArm.FACTUAL

    for mode in ("pilot", "smoke"):
        assert _arm_contract_for_mode(arm, mode)["start_state"] == (
            "same-accepted-g2b-model-only-checkpoint"
        )
    for mode in ("long", "restart-smoke"):
        assert _arm_contract_for_mode(arm, mode)["start_state"] == (
            "same-accepted-mediator-g3-model-only-checkpoint"
        )


def test_metric_summary_separates_executed_information_set_arms() -> None:
    rank_windows = [
        {
            "rank": 0,
            "steps": (
                {
                    "global_step": 1,
                    "executed_action_information_set": "factual",
                    "action_loss": 1.0,
                    "total_loss": 10.0,
                },
                {
                    "global_step": 2,
                    "executed_action_information_set": "mediator-required",
                    "action_loss": 2.0,
                    "total_loss": 20.0,
                },
            ),
        },
        {
            "rank": 1,
            "steps": (
                {
                    "global_step": 1,
                    "executed_action_information_set": "mediator-required",
                    "action_loss": 5.0,
                    "total_loss": 50.0,
                },
                {
                    "global_step": 2,
                    "executed_action_information_set": "factual",
                    "action_loss": 7.0,
                    "total_loss": 70.0,
                },
            ),
        },
    ]

    summary = _action_information_set_metric_summary(
        rank_windows,
        policy="rank-step-counterbalanced-50-50",
        fields=("action_loss", "total_loss"),
    )
    assert summary == {
        "policy": "rank-step-counterbalanced-50-50",
        "arms": {
            "factual": {
                "count": 2,
                "means": {"action_loss": 4.0, "total_loss": 40.0},
            },
            "mediator-required": {
                "count": 2,
                "means": {"action_loss": 3.5, "total_loss": 35.0},
            },
        },
    }
    wrong = copy.deepcopy(rank_windows)
    wrong[0]["steps"][0]["executed_action_information_set"] = "mediator-required"
    with pytest.raises(ValueError, match="differs from schedule"):
        _action_information_set_metric_summary(
            wrong,
            policy="rank-step-counterbalanced-50-50",
            fields=("action_loss", "total_loss"),
        )


def test_scientific_boundary_describes_fixed_head_factual_route() -> None:
    long_boundary = _scientific_boundary_for_mode("long")

    assert "factual direct-posterior" in long_boundary
    assert "Deployment benefit still requires" in long_boundary
    assert "MEDIATOR_REQUIRED" not in long_boundary
    assert "historical" in _scientific_boundary_for_mode("pilot").lower()


def test_model_input_receipt_is_order_independent_and_content_sensitive() -> None:
    first = {
        "tokens": torch.tensor([[1, 2]], dtype=torch.int64),
        "pixels": torch.tensor([[0.25, 0.5]], dtype=torch.float32),
    }
    reordered = {"pixels": first["pixels"].clone(), "tokens": first["tokens"].clone()}
    changed = {"pixels": first["pixels"].clone(), "tokens": first["tokens"].clone()}
    changed["pixels"][0, 1] = 0.75

    assert _tensor_mapping_sha256(first, torch_module=torch) == _tensor_mapping_sha256(
        reordered,
        torch_module=torch,
    )
    assert _tensor_mapping_sha256(first, torch_module=torch) != _tensor_mapping_sha256(
        changed,
        torch_module=torch,
    )


def test_optimizer_initialization_receipt_requires_fresh_zero_state() -> None:
    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    manifest = NativeParameterManifest(
        canonical_names=("policy.bias", "policy.weight"),
        parameter_count=2,
        trainable_numel=3,
        schema_sha256="a" * 64,
    )

    receipt = _optimizer_initialization_receipt(
        rank=0,
        optimizer=optimizer,
        policy=policy,
        optimizer_manifest=manifest,
        model_local_state_sha256="b" * 64,
        rank_rng_sha256="c" * 64,
        torch_module=torch,
    )

    assert set(receipt) == {
        "schema",
        "rank",
        "fresh_zero_state",
        "state_entry_count",
        "parameter_manifest_sha256",
        "parameter_groups_sha256",
        "optimizer_state_sha256",
        "model_local_state_sha256",
        "rank_rng_state_sha256",
    }
    assert receipt["rank"] == 0
    assert receipt["fresh_zero_state"] is True
    assert receipt["state_entry_count"] == 0
    assert receipt["parameter_manifest_sha256"] == manifest.schema_sha256
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    with pytest.raises(RuntimeError, match="empty state"):
        _optimizer_initialization_receipt(
            rank=0,
            optimizer=optimizer,
            policy=policy,
            optimizer_manifest=manifest,
            model_local_state_sha256="b" * 64,
            rank_rng_sha256="c" * 64,
            torch_module=torch,
        )
