from __future__ import annotations

from copy import deepcopy

import pytest
from torch import nn

from picf_next.lingbot_native.host import (
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)
from picf_next.lingbot_native.relation_geometry_probe import (
    RELATION_GEOMETRY_ARM_REPORT_SCHEMA,
    RELATION_GEOMETRY_SAMPLE_SELECTION_RULE,
    RelationProbeSampleMetadata,
    RelationProbeSampleSelection,
    configure_relation_geometry_trainable_scope,
    relation_geometry_probe_subject,
    select_relation_geometry_probe_sample,
    validate_relation_geometry_arm_report,
    validate_relation_probe_sample_selection,
    verify_relation_geometry_trainable_scope,
)


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.host = nn.Linear(8, 8)
        self.picf_native_graph = LingBotNativeGraph(
            LingBotNativeGraphConfig(
                capacity=3,
                host_width=8,
                executed_action_dim=2,
                num_layers=3,
                prediction_address_width=2,
                predictive_target_widths=(("dino_video", 4),),
            )
        )


class _NestedPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.host = nn.Linear(8, 8)
        self.model = nn.Module()
        self.model.qwenvl_with_expert = nn.Module()
        self.model.qwenvl_with_expert.picf_native_graph = LingBotNativeGraph(
            LingBotNativeGraphConfig(
                capacity=3,
                host_width=8,
                executed_action_dim=2,
                num_layers=3,
                prediction_address_width=2,
                predictive_target_widths=(("dino_video", 4),),
            )
        )


def test_relation_geometry_scope_freezes_everything_except_ownership_readout() -> None:
    policy = _Policy()
    graph = policy.picf_native_graph
    scope = configure_relation_geometry_trainable_scope(
        policy,
        graph,
        arm="existing_readout_frozen_host",
    )

    assert scope.parameter_names == (
        "picf_native_graph.relation_readout.no_object",
        "picf_native_graph.relation_readout.projection.weight",
        "picf_native_graph.relation_readout.temperature_parameter",
    )
    assert all(
        parameter.requires_grad == (name in scope.parameter_names)
        for name, parameter in policy.named_parameters()
    )
    assert verify_relation_geometry_trainable_scope(policy, graph, expected=scope) == scope

    graph.relation_readout.existence_projection.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="parameter boundary"):
        verify_relation_geometry_trainable_scope(policy, graph, expected=scope)


def test_relation_geometry_scope_uses_graph_logical_names_under_real_host_nesting() -> None:
    policy = _NestedPolicy()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    scope = configure_relation_geometry_trainable_scope(
        policy,
        graph,
        arm="existing_readout_frozen_host",
    )

    assert scope.parameter_names == (
        "picf_native_graph.relation_readout.no_object",
        "picf_native_graph.relation_readout.projection.weight",
        "picf_native_graph.relation_readout.temperature_parameter",
    )
    assert {name for name, parameter in policy.named_parameters() if parameter.requires_grad} == {
        "model.qwenvl_with_expert.picf_native_graph.relation_readout.no_object",
        "model.qwenvl_with_expert.picf_native_graph.relation_readout.projection.weight",
        "model.qwenvl_with_expert.picf_native_graph.relation_readout.temperature_parameter",
    }
    assert (
        verify_relation_geometry_trainable_scope(
            policy,
            graph,
            expected=scope,
        )
        == scope
    )


def test_relation_geometry_full_host_scope_preserves_production_trainability() -> None:
    policy = _Policy()
    graph = policy.picf_native_graph
    expected_names = tuple(
        sorted(name for name, parameter in policy.named_parameters() if parameter.requires_grad)
    )

    scope = configure_relation_geometry_trainable_scope(
        policy,
        graph,
        arm="structural_full_host",
    )

    assert scope.parameter_names == expected_names
    assert verify_relation_geometry_trainable_scope(policy, graph, expected=scope) == scope


def test_relation_probe_selects_earliest_source_only_globally_eligible_step() -> None:
    sample_keys_by_step = {
        0: ("ambiguous", "good-rank1"),
        1: ("missing-target", "good-rank1"),
        2: ("invisible-target", "good-rank1"),
        3: ("overflow", "good-rank1"),
        4: ("good-rank0", "good-rank1"),
    }
    metadata = {
        "ambiguous": RelationProbeSampleMetadata(
            sample_key="ambiguous",
            task_key="place_in_drawer",
            available_future_transitions=5,
            target_identity_keys=None,
            inventory_identity_keys=(),
            target_supervised_pixel_counts=None,
        ),
        "missing-target": RelationProbeSampleMetadata(
            sample_key="missing-target",
            task_key="push_blue_block_left",
            available_future_transitions=5,
            target_identity_keys=("movable/block_blue",),
            inventory_identity_keys=("movable/block_red",),
            target_supervised_pixel_counts=(0,),
        ),
        "overflow": RelationProbeSampleMetadata(
            sample_key="overflow",
            task_key="push_blue_block_left",
            available_future_transitions=5,
            target_identity_keys=("movable/block_blue",),
            inventory_identity_keys=(
                "movable/block_blue",
                "movable/block_red",
                "movable/block_pink",
            ),
            target_supervised_pixel_counts=(20,),
        ),
        "invisible-target": RelationProbeSampleMetadata(
            sample_key="invisible-target",
            task_key="push_blue_block_left",
            available_future_transitions=5,
            target_identity_keys=("movable/block_blue",),
            inventory_identity_keys=("movable/block_blue", "movable/block_red"),
            target_supervised_pixel_counts=(0,),
        ),
        "good-rank0": RelationProbeSampleMetadata(
            sample_key="good-rank0",
            task_key="push_blue_block_left",
            available_future_transitions=5,
            target_identity_keys=("movable/block_blue",),
            inventory_identity_keys=("movable/block_blue", "movable/block_red"),
            target_supervised_pixel_counts=(20,),
        ),
        "good-rank1": RelationProbeSampleMetadata(
            sample_key="good-rank1",
            task_key="turn_on_led",
            available_future_transitions=1,
            target_identity_keys=("part/table/button_link",),
            inventory_identity_keys=("part/table/button_link",),
            target_supervised_pixel_counts=(10,),
        ),
    }

    selected = select_relation_geometry_probe_sample(
        selection_start_global_step=0,
        total_planned_steps=5,
        capacity=2,
        sample_keys_for_global_step=sample_keys_by_step.__getitem__,
        metadata_for_sample_key=metadata.__getitem__,
    )

    assert selected.selected_global_step == 4
    assert selected.inspected_step_count == 5
    assert tuple(sample.sample_key for sample in selected.samples_by_rank) == (
        "good-rank0",
        "good-rank1",
    )
    assert validate_relation_probe_sample_selection(selected.as_dict()) == selected


def test_relation_probe_selection_fails_closed_without_eligible_pair() -> None:
    metadata = RelationProbeSampleMetadata(
        sample_key="ambiguous",
        task_key="place_in_drawer",
        available_future_transitions=2,
        target_identity_keys=None,
        inventory_identity_keys=(),
        target_supervised_pixel_counts=None,
    )
    with pytest.raises(RuntimeError, match="no two-rank exact-task"):
        select_relation_geometry_probe_sample(
            selection_start_global_step=0,
            total_planned_steps=2,
            capacity=16,
            sample_keys_for_global_step=lambda step: (
                f"ambiguous-{step}-rank0",
                f"ambiguous-{step}-rank1",
            ),
            metadata_for_sample_key=lambda key: RelationProbeSampleMetadata(
                sample_key=key,
                task_key=metadata.task_key,
                available_future_transitions=metadata.available_future_transitions,
                target_identity_keys=metadata.target_identity_keys,
                inventory_identity_keys=metadata.inventory_identity_keys,
                target_supervised_pixel_counts=metadata.target_supervised_pixel_counts,
            ),
        )


def _report() -> dict[str, object]:
    policy = _Policy()
    scope = configure_relation_geometry_trainable_scope(
        policy,
        policy.picf_native_graph,
        arm="existing_readout_frozen_host",
    )
    points = 3
    provenance = {
        "source_commit": "source",
        "checkpoint_revision": "checkpoint",
        "patch_sha256": "1" * 64,
        "execution_contract_sha256": "2" * 64,
        "implementation_sha256": "3" * 64,
        "model_family_sha256": "4" * 64,
        "plan_sha256": "5" * 64,
        "dataset_manifest_sha256": "6" * 64,
        "physical_sidecar_manifest_sha256": "7" * 64,
        "seed": 11,
        "fixed_sample_global_step": 0,
        "sample_selection": RelationProbeSampleSelection(
            selection_start_global_step=0,
            selected_global_step=0,
            inspected_step_count=1,
            capacity=3,
            samples_by_rank=(
                RelationProbeSampleMetadata(
                    sample_key="rank0/current",
                    task_key="push_blue_block_left",
                    available_future_transitions=2,
                    target_identity_keys=("movable/block_blue",),
                    inventory_identity_keys=("movable/block_blue",),
                    target_supervised_pixel_counts=(20,),
                ),
                RelationProbeSampleMetadata(
                    sample_key="rank1/current",
                    task_key="turn_on_led",
                    available_future_transitions=2,
                    target_identity_keys=("part/table/button_link",),
                    inventory_identity_keys=("part/table/button_link",),
                    target_supervised_pixel_counts=(10,),
                ),
            ),
        ).as_dict(),
        "forward_seed_by_rank": [21, 22],
        "frame_sample_keys_by_rank": [
            ["rank0/current", "rank0/next"],
            ["rank1/current", "rank1/next"],
        ],
        "frame_source_digests_by_rank": [
            ["8" * 64, "9" * 64],
            ["a" * 64, "b" * 64],
        ],
        "objective": {
            "optimized_term": "set/ownership",
            "observed_terms": [
                "ownership",
                "ownership_nll",
                "macro_soft_iou",
                "task_soft_iou",
                "action",
            ],
            "window": "fixed_two_frame_local_bptt",
            "labels_are_loss_side_only": True,
            "row_gauge": "initial_assignment_then_frozen",
            "forward_randomness": "fixed_per_rank_torch_seed",
            "official_policy_loss": "observed_not_optimized",
            "predictive_queries": "absent",
        },
        "optimizer": {
            "algorithm": "lingbot_distributed_muon_with_adamw_fallback",
            "learning_rate_hex": (1e-4).hex(),
            "weight_decay_hex": (0.0).hex(),
            "scheduler": "constant",
            "moe_load_balance_hook_enabled": False,
            "update_count": points - 1,
        },
    }
    rank_curves = [
        {
            "ownership": [1.4, 1.2, 1.0],
            "ownership_nll": [10.0, 8.0, 6.0],
            "macro_soft_iou": [0.01, 0.02, 0.03],
            "task_soft_iou": [0.008, 0.018, 0.028],
            "action": [0.2, 0.2, 0.2],
        },
        {
            "ownership": [1.2, 1.0, 0.8],
            "ownership_nll": [8.0, 6.0, 4.0],
            "macro_soft_iou": [0.02, 0.03, 0.04],
            "task_soft_iou": [0.01, 0.02, 0.03],
            "action": [0.3, 0.3, 0.3],
        },
    ]
    rank_reports = []
    for rank, curves in enumerate(rank_curves):
        rank_reports.append(
            {
                "rank": rank,
                "frame_sample_keys": provenance["frame_sample_keys_by_rank"][rank],
                "frame_source_digests": provenance["frame_source_digests_by_rank"][rank],
                "forward_seed": provenance["forward_seed_by_rank"][rank],
                "row_bindings": [["object/a", 0]],
                "curves": curves,
                "task_diagnostics_by_point": [[{"point": point}] for point in range(points)],
                "visual_artifacts_by_point": [[{"point": point}] for point in range(points)],
                "gradient_probe": {"relation_projection_norm": 1.0},
                "step_times_s": [1.0, 1.0, 1.0],
                "peak_reserved_bytes": rank + 1,
            }
        )
    global_curves = {
        name: [
            (rank_curves[0][name][point] + rank_curves[1][name][point]) / 2
            for point in range(points)
        ]
        for name in rank_curves[0]
    }
    return {
        "schema": RELATION_GEOMETRY_ARM_REPORT_SCHEMA,
        "status": "PASS",
        "arm": "existing_readout_frozen_host",
        "subject_sha256": relation_geometry_probe_subject(
            provenance,
            curve_point_count=points,
        ),
        "provenance": provenance,
        "trainable_scope": scope.as_dict(),
        "curve_point_count": points,
        "optimizer_update_count": points - 1,
        "global_curves": global_curves,
        "rank_reports": rank_reports,
        "moe_routing_bias_unchanged": True,
        "maximum_peak_reserved_bytes": 2,
        "total_time_s": 3.0,
    }


def test_relation_geometry_report_recomputes_rank_means_and_subject() -> None:
    report = _report()
    assert validate_relation_geometry_arm_report(report)["status"] == "PASS"

    tampered = deepcopy(report)
    tampered["global_curves"]["task_soft_iou"][2] = 0.9  # type: ignore[index]
    with pytest.raises(ValueError, match="rank mean"):
        validate_relation_geometry_arm_report(tampered)

    tampered = deepcopy(report)
    tampered["provenance"]["objective"]["labels_are_loss_side_only"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="objective"):
        validate_relation_geometry_arm_report(tampered)

    tampered = deepcopy(report)
    tampered["rank_reports"][0]["row_bindings"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="row bindings"):
        validate_relation_geometry_arm_report(tampered)

    tampered = deepcopy(report)
    tampered["provenance"]["sample_selection"]["rule"] = (  # type: ignore[index]
        RELATION_GEOMETRY_SAMPLE_SELECTION_RULE + "-tampered"
    )
    with pytest.raises(ValueError, match="selection rule"):
        validate_relation_geometry_arm_report(tampered)

    tampered = deepcopy(report)
    tampered["provenance"]["sample_selection"]["samples_by_rank"][0][  # type: ignore[index]
        "sample_key"
    ] = "another/current"
    with pytest.raises(ValueError, match="executed current frames"):
        validate_relation_geometry_arm_report(tampered)
