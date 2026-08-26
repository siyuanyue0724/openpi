from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from picf_next.lingbot_native.task_address_learning import task_address_row_coverage
from tools.run_lingbot_vla2_ltop_g2_core import (
    _computed_representation_failures,
    _local_representation_contract_items,
    _physical_relation_prompt_drift,
    _scene_level_robustness,
    _scene_metrics,
    _stage_transfer_checkpoint_state,
    _validate_representation_execution_provenance,
    _validate_representation_item_source,
)


def _representation_contracts() -> tuple[dict[str, object], dict[str, object]]:
    items = []
    labels = []
    source_index = 0
    for partition in ("validation", "heldout"):
        for ordinal in range(8):
            item_id = f"{partition}-{ordinal:04d}"
            prompts = [
                {"name": f"{item_id}/prompt-{index}"}
                for index in range(2)
            ]
            items.append(
                {
                    "execution_rank": source_index % 4,
                    "item_id": item_id,
                    "ordinal": ordinal,
                    "partition": partition,
                    "prompts": prompts,
                }
            )
            labels.append({"item_id": item_id, "prompts": prompts})
            source_index += 1
    return (
        {"items": items, "world_size": 4},
        {"items": labels},
    )


def test_representation_runtime_rebind_is_balanced_and_auditable() -> None:
    execution, labels = _representation_contracts()
    rank0, schedule0 = _local_representation_contract_items(execution, labels, rank=0)
    rank1, schedule1 = _local_representation_contract_items(execution, labels, rank=1)

    assert schedule0 == schedule1
    assert schedule0["source_world_size"] == 4
    assert schedule0["runtime_world_size"] == 2
    assert len(schedule0["sha256"]) == 64
    for selected in (rank0, rank1):
        assert sum(item["partition"] == "validation" for item, _label in selected) == 4
        assert sum(item["partition"] == "heldout" for item, _label in selected) == 4
    assert {item["item_id"] for item, _label in rank0}.isdisjoint(
        item["item_id"] for item, _label in rank1
    )


def test_scene_metric_reports_row_permutation_as_an_implementation_self_check() -> None:
    first = torch.tensor(
        [[[0.80, 0.10, 0.05, 0.05]] * 4],
        dtype=torch.float32,
    )
    second = torch.tensor(
        [[[0.10, 0.80, 0.05, 0.05]] * 4],
        dtype=torch.float32,
    )

    metrics = _scene_metrics(
        (first, second),
        (0, 1),
        task_address_row_coverage=task_address_row_coverage,
        torch_module=torch,
    )

    assert metrics["mean_margin"] > 0
    assert (
        metrics["metric_self_checks"]["matched_row_permutation_max_abs_error"]
        <= 1e-7
    )


def test_representation_provenance_is_bound_to_dataset_and_live_source() -> None:
    execution = {
        "world_size": 4,
        "provenance": {
            "current_dataset_manifest_file_sha256": "a" * 64,
            "current_dataset_tree_sha256": "b" * 64,
        },
    }
    _validate_representation_execution_provenance(
        execution,
        dataset_manifest_file_sha256="a" * 64,
        dataset_tree_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="another dataset tree"):
        _validate_representation_execution_provenance(
            execution,
            dataset_manifest_file_sha256="a" * 64,
            dataset_tree_sha256="c" * 64,
        )

    request = SimpleNamespace(
        sample_key="sample",
        source_global_index=3,
        source_sensor_hash_by_field={"rgb_static": "d" * 64},
    )
    item = {
        "sample_key": "sample",
        "source_global_index": 3,
        "source_sensor_sha256": {"rgb_static": "d" * 64},
        "source_state_sha256": "e" * 64,
    }
    _validate_representation_item_source(
        item,
        request=request,
        canonical_source_global_index=3,
        sidecar_source_state_sha256="e" * 64,
    )
    item_without_redundant_index = dict(item)
    item_without_redundant_index.pop("source_global_index")
    _validate_representation_item_source(
        item_without_redundant_index,
        request=request,
        canonical_source_global_index=3,
        sidecar_source_state_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="sample key"):
        _validate_representation_item_source(
            item,
            request=request,
            canonical_source_global_index=4,
            sidecar_source_state_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="sensor hashes"):
        _validate_representation_item_source(
            item,
            request=SimpleNamespace(
                sample_key="sample",
                source_global_index=3,
                source_sensor_hash_by_field={"rgb_static": "f" * 64},
            ),
            canonical_source_global_index=3,
            sidecar_source_state_sha256="e" * 64,
        )


def test_physical_prompt_drift_covers_row_embeddings() -> None:
    fields = {
        "support_logits": torch.zeros(1),
        "visible_support": torch.zeros(1),
        "ownership": torch.zeros(1),
        "ownership_log_probability": torch.zeros(1),
        "existence": torch.zeros(1),
        "existence_logits": torch.zeros(1),
        "row_embeddings": torch.zeros(2, 3),
        "relation_temperature": torch.zeros(1),
        "sensor_valid": torch.ones(1, dtype=torch.bool),
        "structural_sensor_valid": None,
    }
    left = SimpleNamespace(**fields)
    right_fields = dict(fields)
    right_fields["row_embeddings"] = fields["row_embeddings"].clone()
    right_fields["row_embeddings"][1, 2] = 0.25
    right = SimpleNamespace(**right_fields)

    assert _physical_relation_prompt_drift(left, right) == 0.25


def test_scene_level_robustness_does_not_treat_prompts_as_independent() -> None:
    reports = []
    for rank in range(2):
        scenes = [
            {"item_id": f"heldout-{rank * 4 + index:04d}", "mean_margin": margin}
            for index, margin in enumerate((0.1, 0.2, 0.3, -0.1))
        ]
        reports.append({"history": [{"heldout": {"scenes": scenes}}]})

    diagnostic = _scene_level_robustness(
        reports,
        partition="heldout",
        seed=7,
        bootstrap_samples=1_000,
    )

    assert diagnostic["scene_count"] == 8
    assert diagnostic["positive_scene_count"] == 6
    assert len(diagnostic["scene_margins"]) == 8


def test_stage_transfer_checkpoint_is_model_only() -> None:
    policy = object()

    state = _stage_transfer_checkpoint_state(policy)

    assert state == {"model": policy}
    assert "optimizer" not in state
    assert "extra_state" not in state


def test_representation_gate_rejects_scope_provenance_or_frozen_action_drift() -> None:
    def partition(*, initial: bool, shared_row_gauge: bool = True) -> dict[str, object]:
        return {
            "mean_margin": 0.0 if initial else 0.1,
            "mean_target_nll": 1.0 if initial else 0.5,
            "mean_physical_set_loss": 10.0 if initial else 5.0,
            "physical_prompt_drift_max_abs": 0.0,
            "positive_margin_count": 0 if initial else 8,
            "prompts": [{"margin": 0.0 if initial else 0.1} for _ in range(8)],
            "shared_row_gauge": shared_row_gauge,
            "metric_self_checks": {
                "matched_row_permutation_max_abs_error": 0.0,
            },
        }

    ranks = []
    for rank in range(2):
        ranks.append(
            {
                "rank": rank,
                "history": [
                    {
                        "validation": partition(initial=True),
                        "heldout": partition(initial=True),
                    },
                    {
                        "validation": partition(
                            initial=False,
                            shared_row_gauge=rank != 0,
                        ),
                        "heldout": partition(initial=False),
                    },
                ],
                "all_gradients_finite": True,
                "runtime_schedule_sha256": ("e" if rank else "c") * 64,
                "optimizer_parameter_manifest": {
                    "schema_sha256": ("f" if rank else "d") * 64
                },
                "gradient_norms": [1.0],
                "gradient_metrics_history": [
                    {
                        "native_graph_norm": 1.0,
                        "task_query_norm": 1.0,
                        "shared_host_norm": 1.0,
                        "action_output_norm": 0.0,
                    }
                ],
                "frozen_action_state": {
                    "before_sha256": "a" * 64,
                    "after_sha256": ("b" if rank == 1 else "a") * 64,
                    "changed_tensors": ["action"] if rank == 1 else [],
                    "tensor_names": ["action"],
                },
            }
        )

    failures = _computed_representation_failures({"rank_reports": ranks})

    assert "crossed prompts did not preserve one canonical physical row gauge" in failures
    assert "data-parallel ranks used different runtime execution schedules" in failures
    assert "data-parallel ranks optimized different parameter schemas" in failures
    assert "rank 1: frozen action state changed" in failures
