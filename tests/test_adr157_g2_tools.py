from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import torch

from tools.build_adr157_g2_contract import _build_pair, _canonical_bytes
from tools.score_adr157_g2_offline import _action_table, _normalized_effect

ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = ROOT / "tools/run_adr157_g2_fixed_observation.py"
SCORER = ROOT / "tools/score_adr157_g2_offline.py"


def _instruction(value: str) -> dict[str, object]:
    return {
        "instruction": value,
        "instruction_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def test_g2_builder_separates_execution_from_offline_labels(tmp_path: Path) -> None:
    items = []
    rows = []
    for partition in ("validation", "heldout"):
        for ordinal in range(2):
            sample_key = f"{partition}/sample-{ordinal}"
            source_global_index = (0 if partition == "validation" else 100) + ordinal
            sensor_hashes = {"rgb_static": f"{ordinal + 1:064x}"}
            state_hash = f"{ordinal + 11:064x}"
            variants = []
            for variant_index in range(2):
                variant = _instruction(f"{partition} instruction {ordinal}/{variant_index}")
                variant.update(
                    {
                        "target_identity_key": (
                            f"calvin/{partition}/object-{ordinal}-{variant_index}"
                        ),
                        "target_mass": float(variant_index + 1),
                        "task_key": f"task-{partition}-{ordinal}-{variant_index}",
                    }
                )
                variants.append(variant)
            items.append(
                {
                    "group": {
                        "source_sensor_sha256": sensor_hashes,
                        "source_global_index": source_global_index,
                        "source_state_sha256": state_hash,
                        "stateful_sample_key": sample_key,
                    },
                    "ordinal": ordinal,
                    "partition": partition,
                    "replay_seed": ordinal + 17,
                    "variants": variants,
                }
            )
            rows.append(
                {
                    "ordinal": ordinal,
                    "partition": partition,
                    "sample_key": sample_key,
                    "source_sensor_sha256": sensor_hashes,
                    "source_state_sha256": state_hash,
                }
            )
    plan = {
        "items": items,
        "schema": "picf-next.lingbot-fixed-observation-evaluation-plan.v2",
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(_canonical_bytes(plan) + b"\n")
    rebind = {
        "current_dataset_manifest_file_sha256": "a" * 64,
        "current_dataset_tree_sha256": "b" * 64,
        "old_plan_file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "rows": rows,
        "schema": "picf-next.adr154-current-dataset-fixed-x-source-rebind/v1",
        "status": "PASS",
        "validated_item_count": len(items),
    }
    rebind_path = tmp_path / "rebind.json"
    rebind_path.write_bytes(_canonical_bytes(rebind) + b"\n")
    coverage = {
        "artifact_sha256": "c" * 64,
        "dataset_tree_sha256": rebind["current_dataset_tree_sha256"],
        "records": [
            {
                "sample_key": item["group"]["stateful_sample_key"],
                "source_global_index": item["group"]["source_global_index"],
            }
            for item in items
        ],
        "records_sha256": "d" * 64,
        "schema": "picf-next.dense-evidence-coverage-plan/v1",
    }
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_bytes(_canonical_bytes(coverage) + b"\n")

    _build_pair(
        plan_path=plan_path,
        rebind_path=rebind_path,
        coverage_path=coverage_path,
        output_dir=tmp_path,
        name="g2-smoke-4",
        per_partition=2,
        world_size=4,
    )

    execution = json.loads((tmp_path / "g2-smoke-4.execution.json").read_text())
    labels = json.loads((tmp_path / "g2-smoke-4.labels.json").read_text())
    encoded_execution = json.dumps(execution, sort_keys=True)
    assert "target_identity_key" not in encoded_execution
    assert "target_mass" not in encoded_execution
    assert sorted(item["execution_rank"] for item in execution["items"]) == [0, 1, 2, 3]
    assert sorted(item["source_global_index"] for item in execution["items"]) == [0, 1, 100, 101]
    assert execution["provenance"]["selection_policy"] == "full_modal_mass_balanced_v1"
    assert (
        labels["source_execution_sha256"] == hashlib.sha256(_canonical_bytes(execution)).hexdigest()
    )
    assert all(
        "target_identity_key" in prompt for item in labels["items"] for prompt in item["prompts"]
    )


def test_g2_builder_excludes_sources_without_full_modal_coverage(tmp_path: Path) -> None:
    items = []
    rows = []
    covered_indices = []
    for partition_index, partition in enumerate(("validation", "heldout")):
        for ordinal in range(3):
            source_global_index = partition_index * 100 + ordinal
            sample_key = f"{partition}/sample-{ordinal}"
            sensor_hashes = {"rgb_static": f"{source_global_index + 1:064x}"}
            state_hash = f"{source_global_index + 11:064x}"
            items.append(
                {
                    "group": {
                        "source_global_index": source_global_index,
                        "source_sensor_sha256": sensor_hashes,
                        "source_state_sha256": state_hash,
                        "stateful_sample_key": sample_key,
                    },
                    "ordinal": ordinal,
                    "partition": partition,
                    "replay_seed": source_global_index + 17,
                    "variants": [
                        {
                            **_instruction(f"{partition} {ordinal} first"),
                            "target_identity_key": f"calvin/{partition}/first-{ordinal}",
                            "target_mass": 1.0,
                            "task_key": f"first-{partition}-{ordinal}",
                        },
                        {
                            **_instruction(f"{partition} {ordinal} second"),
                            "target_identity_key": f"calvin/{partition}/second-{ordinal}",
                            "target_mass": 1.0 + ordinal,
                            "task_key": f"second-{partition}-{ordinal}",
                        },
                    ],
                }
            )
            rows.append(
                {
                    "ordinal": ordinal,
                    "partition": partition,
                    "sample_key": sample_key,
                    "source_sensor_sha256": sensor_hashes,
                    "source_state_sha256": state_hash,
                }
            )
            if ordinal:
                covered_indices.append(source_global_index)
    plan = {
        "items": items,
        "schema": "picf-next.lingbot-fixed-observation-evaluation-plan.v2",
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(_canonical_bytes(plan) + b"\n")
    rebind = {
        "current_dataset_manifest_file_sha256": "a" * 64,
        "current_dataset_tree_sha256": "b" * 64,
        "old_plan_file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "rows": rows,
        "schema": "picf-next.adr154-current-dataset-fixed-x-source-rebind/v1",
        "status": "PASS",
        "validated_item_count": len(items),
    }
    rebind_path = tmp_path / "rebind.json"
    rebind_path.write_bytes(_canonical_bytes(rebind) + b"\n")
    coverage = {
        "artifact_sha256": "c" * 64,
        "dataset_tree_sha256": rebind["current_dataset_tree_sha256"],
        "records": [
            {"sample_key": f"covered/{index}", "source_global_index": index}
            for index in covered_indices
        ],
        "records_sha256": "d" * 64,
        "schema": "picf-next.dense-evidence-coverage-plan/v1",
    }
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_bytes(_canonical_bytes(coverage) + b"\n")

    _build_pair(
        plan_path=plan_path,
        rebind_path=rebind_path,
        coverage_path=coverage_path,
        output_dir=tmp_path,
        name="covered",
        per_partition=2,
        world_size=4,
    )

    execution = json.loads((tmp_path / "covered.execution.json").read_text())
    assert {item["source_global_index"] for item in execution["items"]} == set(covered_indices)


def test_g2_gpu_executor_is_label_free_and_uses_the_shared_action_root() -> None:
    source = EXECUTOR.read_text(encoding="utf-8")
    module = ast.parse(source)
    imported_modules = {
        node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom)
    }
    assert "picf_next.data.calvin_physical_supervision_sidecar" not in imported_modules
    assert "picf_next.lingbot_native.calvin_entity_set" not in imported_modules
    for fragment in (
        "run_native_v3_prior_chain",
        "native_context_from_prior_trace",
        "run_native_policy_observation_diagnostic_forward",
        "run_native_frozen_posterior_action_forward",
        "factual-repeat",
        "label_blind_visibility_removal_arms",
        'build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp").load',
        '"status": "ACTION_RECEIPTS_SEALED"',
        '"status": "SEALED_AFTER_ACTIONS"',
    ):
        assert fragment in source
    assert "_task_independent_current_frame_context" not in source
    diagnostic = source[
        source.index("def _instrumented_diagnostic(") : source.index(
            "native_training.run_native_policy_diagnostic_forward = _instrumented_diagnostic"
        )
    ]
    assert diagnostic.index("run_native_v3_prior_chain") < diagnostic.index(
        "_original_observation("
    )
    assert source.index('"status": "ACTION_RECEIPTS_SEALED"') < source.index(
        '"status": "SEALED_AFTER_ACTIONS"'
    )
    assert "len(prompts) * len(arms) * int(flow.config.num_steps)" in source
    assert "int(flow.config.num_steps) + 1" not in source


def test_g2_checkpoint_restore_matches_the_released_runner_collective_order() -> None:
    source = EXECUTOR.read_text(encoding="utf-8")
    restore = source[
        source.index("def _restore_then_build_official_optimizer(") : source.index(
            "def _restore_forward_bindings()"
        )
    ]
    diagnostic = source[
        source.index("def _instrumented_diagnostic(") : source.index(
            "native_training.run_native_policy_diagnostic_forward = _instrumented_diagnostic"
        )
    ]
    assert '"model_checkpoint_load_started"' in restore
    assert '"model_checkpoint_dcp_returned"' in restore
    assert '"model_checkpoint_cuda_synchronized"' in restore
    assert "torch.cuda.synchronize" in restore
    assert restore.index("_original_optimizer_builder(") < restore.index(
        'build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp").load'
    )
    assert "dist.barrier()" not in restore
    assert "build_checkpointer" not in diagnostic
    assert "if not _checkpoint_restored:" in diagnostic
    assert source.index(
        "runner.build_lingbot_official_optimizer = _restore_then_build_official_optimizer"
    ) < source.index("runner.main()")


def test_g2_offline_scorer_opens_labels_after_sealed_receipts() -> None:
    source = SCORER.read_text(encoding="utf-8")
    assert "CalvinPhysicalSupervisionSidecar" in source
    assert 'parser.add_argument("--offline-labels-sha256", required=True)' in source
    assert "_sha256(args.offline_labels) != args.offline_labels_sha256" in source
    assert 'aggregate.get("status") != "ACTION_RECEIPTS_SEALED"' in source
    assert source.index('aggregate.get("status") != "ACTION_RECEIPTS_SEALED"') < source.index(
        "CalvinDatasetIndex.load("
    )


def test_g2_action_effect_uses_the_factual_action_norm() -> None:
    factual = torch.tensor([[3.0, 4.0]])
    changed = torch.tensor([[0.0, 4.0]])
    assert abs(_normalized_effect(changed, factual) - 0.6) < 1e-6
    execution = {
        "receipt_count": 2,
        "receipts": [
            {"action_key": "a", "arm_name": "factual", "prompt_name": "p"},
            {"action_key": "b", "arm_name": "remove-row-0", "prompt_name": "p"},
        ],
    }
    assert set(_action_table(execution, {"a": factual, "b": changed})) == {
        ("p", "factual"),
        ("p", "remove-row-0"),
    }
