from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import picf_next.lingbot_native.capacity as capacity
import tools.run_lingbot_vla2_native_full as full_runner
from picf_next.data.lingbot_calvin_projection import processor_assets_sha256
from picf_next.data.token_supervision_policy import (
    build_known_pixel_token_supervision_policy,
)
from picf_next.lingbot_native.calvin import NativeCALVINRouting
from picf_next.lingbot_native.capacity import MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES
from picf_next.lingbot_native.current_grid_cache import (
    current_correction_summary_query_schema_digest,
    current_grid_query_schema_digest,
    omitted_static_summary_query_schema_digest,
)
from picf_next.lingbot_native.fixed_batch_probe import (
    configure_fixed_batch_trainable_scope,
    validate_predictive_fixed_batch_arm_report,
)
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_GPU_SHARDED,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
)
from picf_next.lingbot_native.host import LingBotNativeGraph, LingBotNativeGraphConfig
from picf_next.lingbot_native.predictive_cache import (
    PredictiveCacheContract,
    native_predictive_coverage_digest,
    native_predictive_query_schema_digest,
)
from picf_next.lingbot_native.predictive_decision import (
    IMPLEMENTED_PREDICTIVE_OBJECTIVE,
    IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
    PREDICTIVE_OBJECTIVE_CLAIMS,
)
from picf_next.lingbot_native.predictive_diagnostics import (
    PREDICTIVE_TARGET_AUDIT_SCHEMA,
    PREDICTIVE_TEMPORAL_AUDIT_SCHEMA,
    PREDICTIVE_TEMPORAL_FEATURE_PAIRING,
    TEACHER_CAUSALITY_AUDIT_SCHEMA,
    predictive_latent_diagnostics,
    predictive_target_pretraining_readiness,
    predictive_temporal_diagnostics,
    predictive_temporal_pretraining_readiness,
    predictive_visible_support_diagnostics,
)
from picf_next.lingbot_native.prompt_tokenization import (
    CompletePromptTokenizationAudit,
    PromptTokenizationEntry,
)
from picf_next.lingbot_native.relation_bilinear_probe import (
    RELATION_BILINEAR_PROBE_ARM,
    FullRankBilinearRelationReadout,
)
from picf_next.lingbot_native.relation_depth_probe import RELATION_DEPTH_PROBE_ARM
from picf_next.lingbot_native.relation_geometry_probe import (
    configure_relation_geometry_trainable_scope,
    validate_relation_geometry_arm_report,
)
from picf_next.lingbot_native.relations import SharedRelationReadout
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
    REPRESENTATION_TRIAL_SPLIT_SCHEMA,
    RepresentationEvaluationSegment,
    RepresentationTrialSplit,
)
from picf_next.lingbot_native.supervision import (
    TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
    TOKEN_MICRO_OWNERSHIP,
)
from picf_next.lingbot_native.task_relation import (
    GLOBAL_MULTIPOSITIVE_TASK_RELATION,
    HOST_NATIVE_FACTORIZED_TASK_RELATION,
    HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION,
    LOCAL_BALANCED_TASK_RELATION,
)
from picf_next.lingbot_native.temporal import (
    NativeLaneConfig,
    NativeTrainingLaneBank,
    TemporalBatchPlan,
)
from tools.bootstrap_lingbot_vla2 import (
    LINGBOT_CHECKPOINT_REVISION,
    QWEN_PROCESSOR_ID,
    QWEN_PROCESSOR_REVISION,
)
from tools.bootstrap_lingbot_vla2_native import (
    CHECKOUT_RELATIVE_PATH,
    LINGBOT_NATIVE_SOURCE_COMMIT,
)
from tools.build_lingbot_native_gate_decision import build_training_gate_decision
from tools.build_lingbot_predictive_objective_decision import (
    build_predictive_objective_decision,
)
from tools.run_lingbot_vla2_native_full import (
    BEHAVIOR_CAUSAL_PROBE_DISTRIBUTED_SCHEMA,
    BEHAVIOR_CAUSAL_PROBE_SCHEMA,
    BEHAVIOR_POSTERIOR_CONTROL_PROBE_DISTRIBUTED_SCHEMA,
    FULL_COMPARISON_ID,
    FULL_EXTRA_STATE_SCHEMA,
    FULL_WORLD_SIZE,
    REPRESENTATION_EXTRA_STATE_SCHEMA,
    TRAINING_AUTHORIZATION_SCHEMA,
    TRAINING_GATE_EVIDENCE_SCHEMAS,
    _advance_report_row_binding_continuity,
    _all_reduce_external_relation_candidate_gradients,
    _backward_behavior_total_host,
    _backward_behavior_via_posterior_host,
    _backward_isolated_objective_family,
    _behavior_conditioning_digest,
    _behavior_graph_digest,
    _behavior_probe_sample_keys_by_rank,
    _cache_producer_patch_sha256,
    _distributed_action_state_sha256,
    _distributed_any_boolean,
    _distributed_family_gradient_diagnostics,
    _distributed_pre_backward_failures,
    _distributed_predictive_host_gradient_diagnostics,
    _distributed_raise_if_local_probe_error,
    _distributed_relation_surface_gradient_diagnostics,
    _distributed_ring_exchange_tensor,
    _distributed_uniform_boolean,
    _emit_step_progress,
    _execution_contract_digest,
    _fixed_observation_primary_temporal_plan,
    _full_implementation_digest,
    _full_implementation_paths,
    _local_gradient_norm,
    _local_import_modules,
    _moe_routing_bias_matches,
    _moe_routing_bias_snapshot,
    _objective_result_posterior_state,
    _parameter_gradient_snapshot,
    _parse_args,
    _parse_nonnegative_layer_set,
    _parse_nonnegative_step_set,
    _parse_positive_step_set,
    _posterior_bank_digest,
    _predictive_host_gradient_parameters,
    _recompute_report_objective,
    _relation_surface_component_gradients,
    _require_runtime_storage_capacity,
    _require_unchanged_behavior_graph,
    _reset_moe_probe_counters,
    _run_predictive_fixed_batch_arm,
    _run_relation_geometry_fixed_batch_arm,
    _select_fixed_batch_plan,
    _select_relation_geometry_source_sample,
    _shared_host_family_gradient_snapshot,
    _temporal_execution_counts,
    _trim_cuda_allocator_after_gradient_audit,
    _validate_action_fsdp2_topology,
    _validate_gradient_audit_target_coverage,
    _validate_lingbot_calvin_projection_batch,
    _validate_lingbot_projection_processor,
    _validate_paths_and_args,
    _validate_representation_resume_extra,
    _validate_resume_extra,
    _validate_step_row_bindings,
    _validate_training_supervision_policy,
    _validate_vlm_fsdp2_topology,
    _weighted_behavior_future_contribution,
    load_behavior_causal_probe_evidence,
    load_behavior_posterior_control_probe_evidence,
    load_current_grid_build_report,
    load_predictive_build_report,
    load_predictive_target_audit,
    load_predictive_teacher_causality_audit,
    load_predictive_temporal_audit,
    require_behavior_causal_probe_context,
    validate_behavior_causal_probe_evidence,
    validate_behavior_posterior_control_probe_evidence,
    validate_full_objective_report,
    validate_representation_objective_report,
    validate_training_authorization,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/run_lingbot_vla2_native_full.py"


def test_arm_nr_runner_reads_the_native_lane_bank_digest_property() -> None:
    bank = NativeTrainingLaneBank(
        NativeLaneConfig(
            model_digest="model",
            schema_digest="schema",
            capacity=2,
            host_width=3,
            maximum_optimizer_lag=4,
        )
    )

    assert _posterior_bank_digest(bank) == bank.digest
    with pytest.raises(ValueError, match="posterior bank digest"):
        _posterior_bank_digest(SimpleNamespace(digest=lambda: bank.digest))


def test_objective_result_posterior_state_supports_both_native_result_shapes() -> None:
    posterior = object()
    representation = SimpleNamespace(primary=SimpleNamespace(posterior_state=posterior))
    action = SimpleNamespace(
        primary=SimpleNamespace(context=SimpleNamespace(posterior_state=posterior))
    )

    assert _objective_result_posterior_state(representation) is posterior
    assert _objective_result_posterior_state(action) is posterior

    with pytest.raises(RuntimeError, match="omitted the deploy posterior"):
        _objective_result_posterior_state(SimpleNamespace(primary=SimpleNamespace()))


def test_full_runner_requires_live_checkpoint_storage_before_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "new-run"
    monkeypatch.setattr(
        capacity.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES),
    )
    assert _require_runtime_storage_capacity(run_dir) == MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES
    monkeypatch.setattr(
        capacity.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_LINGBOT_CHECKPOINT_WRITE_FREE_BYTES - 1),
    )
    with pytest.raises(RuntimeError, match="checkpoint filesystem"):
        _require_runtime_storage_capacity(run_dir)


def test_behavior_probe_requires_only_bounded_evidence_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "new-run"
    free_bytes = capacity.MINIMUM_LINGBOT_EVIDENCE_WRITE_FREE_BYTES
    monkeypatch.setattr(
        capacity.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=free_bytes),
    )
    assert (
        _require_runtime_storage_capacity(
            run_dir,
            checkpoint_required=False,
        )
        == free_bytes
    )


def test_native_run_root_must_be_a_real_descendant_of_persistent_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent = tmp_path / "mnt"
    persistent.mkdir()
    run_root = persistent / "runs/trial"
    run_root.mkdir(parents=True)
    monkeypatch.setattr(capacity, "PERSISTENT_CLOUD_ROOT", persistent)

    assert capacity.require_persistent_run_root(run_root) == run_root.resolve()
    with pytest.raises(RuntimeError, match="strict descendant"):
        capacity.require_persistent_run_root(tmp_path)
    with pytest.raises(RuntimeError, match="strict descendant"):
        capacity.require_persistent_run_root(persistent)

    alias = persistent / "alias"
    alias.symlink_to(run_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        capacity.require_persistent_run_root(alias)


def _training_projection_fixture() -> tuple[dict[str, object], dict[str, object]]:
    assets = [
        {"path": "config.json", "bytes": 1, "sha256": "1" * 64},
        {"path": "preprocessor_config.json", "bytes": 1, "sha256": "2" * 64},
    ]

    def view(source_field: str, shape: list[int], digit: str) -> dict[str, object]:
        return {
            "source_field": source_field,
            "source_shape": shape,
            "image_grid_thw": [1, 16, 16],
            "merged_grid_hw": [8, 8],
            "raw_patch_count": 256,
            "merged_token_count": 64,
            "pixel_values_shape": [256, 1536],
            "source_rgb_sha256": [digit * 64] * 3,
        }

    projection = {
        "schema": "picf-next.lingbot-calvin-qwen-projection.v1",
        "status": "PASS",
        "runtime_input": False,
        "processor_id": QWEN_PROCESSOR_ID,
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "processor_assets_sha256": processor_assets_sha256(assets),
        "processor_config_sha256": "1" * 64,
        "processor_preprocessor_config_sha256": "2" * 64,
        "dataset_manifest_sha256": "3" * 64,
        "dataset_tree_sha256": "4" * 64,
        "source_frame_count": 20,
        "sample_global_indices": [0, 10, 19],
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "views": {
            "static": view("rgb_static", [200, 200, 3], "5"),
            "gripper": view("rgb_gripper", [84, 84, 3], "6"),
        },
        "transformers_version": "5.0.0",
    }
    report = {
        "processor_id": QWEN_PROCESSOR_ID,
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "processor_assets": assets,
    }
    return projection, report


def test_full_runner_binds_d1_projection_to_exact_processor_and_batches() -> None:
    projection, processor_report = _training_projection_fixture()
    measured = _validate_lingbot_projection_processor(
        physical_visual_acceptance={
            "dataset_manifest_sha256": "3" * 64,
            "training_projection": projection,
        },
        processor_report=processor_report,
        vision_config=SimpleNamespace(
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        dataset_tree_sha256="4" * 64,
        transformers_version="5.0.0",
    )
    batch_size = 2
    _validate_lingbot_calvin_projection_batch(
        {
            "image_grid_thw": torch.tensor([[[1, 16, 16], [1, 16, 16], [1, 16, 16]]]).expand(
                batch_size, -1, -1
            ),
            "img_masks": torch.tensor([[True, True, False]]).expand(batch_size, -1),
        },
        projection=measured,
    )


def test_full_runner_rejects_projection_processor_or_camera_slot_drift() -> None:
    projection, processor_report = _training_projection_fixture()
    acceptance = {
        "dataset_manifest_sha256": "3" * 64,
        "training_projection": projection,
    }
    with pytest.raises(RuntimeError, match="vision geometry"):
        _validate_lingbot_projection_processor(
            physical_visual_acceptance=acceptance,
            processor_report=processor_report,
            vision_config=SimpleNamespace(
                patch_size=14,
                spatial_merge_size=2,
                temporal_patch_size=2,
            ),
            dataset_tree_sha256="4" * 64,
            transformers_version="5.0.0",
        )
    wrong_grid = torch.tensor(
        [[[1, 16, 16], [1, 18, 18], [1, 16, 16]]],
        dtype=torch.long,
    )
    with pytest.raises(RuntimeError, match="accepted camera projection"):
        _validate_lingbot_calvin_projection_batch(
            {
                "image_grid_thw": wrong_grid,
                "img_masks": torch.tensor([[True, True, False]]),
            },
            projection=projection,
        )


def test_full_runner_binds_d1_token_supervision_measure_to_runtime() -> None:
    policy = build_known_pixel_token_supervision_policy()
    acceptance = {"training_supervision_policy": policy}

    assert (
        _validate_training_supervision_policy(
            physical_visual_acceptance=acceptance,
            minimum_supervised_fraction=0.0,
        )
        == policy
    )
    with pytest.raises(RuntimeError, match="differs from training"):
        _validate_training_supervision_policy(
            physical_visual_acceptance=acceptance,
            minimum_supervised_fraction=1.0,
        )


def test_full_runner_registers_cpu_and_cuda_collective_backends() -> None:
    source = TOOL.read_text()
    assert 'dist.init_process_group(backend="cpu:gloo,cuda:nccl")' in source
    assert 'dist.init_process_group(backend="nccl")' not in source


def test_temporal_report_records_executed_counts_not_optional_sampling_sentinels() -> None:
    assert _temporal_execution_counts(local_bptt_steps=None, overshoot_horizon=None) == (1, 0)
    assert _temporal_execution_counts(local_bptt_steps=4, overshoot_horizon=64) == (4, 64)
    with pytest.raises(ValueError, match="local BPTT"):
        _temporal_execution_counts(local_bptt_steps=1, overshoot_horizon=None)
    with pytest.raises(ValueError, match="overshoot"):
        _temporal_execution_counts(local_bptt_steps=None, overshoot_horizon=0)


def test_fixed_observation_temporal_restriction_is_primary_only() -> None:
    natural = TemporalBatchPlan(
        seed=7,
        state_ages=(0, 0),
        local_bptt_steps=4,
        overshoot_horizon=None,
        source_masked_branch=False,
    )

    assert (
        _fixed_observation_primary_temporal_plan(
            natural,
            pair_plan_sha256=None,
        )
        is natural
    )
    fixed = _fixed_observation_primary_temporal_plan(
        natural,
        pair_plan_sha256="a" * 64,
    )
    assert fixed == TemporalBatchPlan(
        seed=natural.seed,
        state_ages=natural.state_ages,
        local_bptt_steps=None,
        overshoot_horizon=None,
        source_masked_branch=False,
    )

    source_only = TemporalBatchPlan(
        seed=8,
        state_ages=(0, 0),
        local_bptt_steps=None,
        overshoot_horizon=None,
        source_masked_branch=True,
    )
    assert (
        _fixed_observation_primary_temporal_plan(
            source_only,
            pair_plan_sha256="b" * 64,
        )
        == source_only
    )


def test_distributed_ring_exchange_returns_the_peer_tensor() -> None:
    class Dist:
        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def all_gather(outputs: list[torch.Tensor], value: torch.Tensor) -> None:
            outputs[0].copy_(value)
            outputs[1].copy_(value + 10)

    value = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(
        _distributed_ring_exchange_tensor(value, dist=Dist(), torch_module=torch),
        value + 10,
    )


@pytest.mark.parametrize("bilinear", (False, True))
def test_external_relation_candidate_gradients_are_rank_meaned_in_one_bucket(
    bilinear: bool,
) -> None:
    class Dist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def all_reduce(value: torch.Tensor, *, op: str) -> None:
            assert op == "sum"
            value.mul_(2)

    readout = (
        FullRankBilinearRelationReadout(4, mode="unconstrained")
        if bilinear
        else SharedRelationReadout(4)
    )
    expected_square_sum = 0.0
    for value, parameter in enumerate(
        (
            readout.projection.weight,
            readout.no_object,
            readout.temperature_parameter,
        ),
        start=1,
    ):
        parameter.grad = torch.full_like(parameter, float(value))
        expected_square_sum += parameter.numel() * value**2

    norm = _all_reduce_external_relation_candidate_gradients(
        readout=readout,
        candidate_id="test_candidate",
        diagnostic_rank=None,
        require_positive=True,
        dist=Dist,
        world_size=2,
        torch_module=torch,
    )

    assert norm == pytest.approx(expected_square_sum**0.5)
    for value, parameter in enumerate(
        (
            readout.projection.weight,
            readout.no_object,
            readout.temperature_parameter,
        ),
        start=1,
    ):
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, float(value)))


def test_external_relation_candidate_allows_finite_zero_after_first_update() -> None:
    class Dist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def all_reduce(value: torch.Tensor, *, op: str) -> None:
            assert op == "sum"

    readout = FullRankBilinearRelationReadout(4, mode="unconstrained")
    for parameter in (
        readout.projection.weight,
        readout.no_object,
        readout.temperature_parameter,
    ):
        parameter.grad = torch.zeros_like(parameter)

    assert (
        _all_reduce_external_relation_candidate_gradients(
            readout=readout,
            candidate_id="saturated_candidate",
            diagnostic_rank=None,
            require_positive=False,
            dist=Dist,
            world_size=1,
            torch_module=torch,
        )
        == 0.0
    )
    with pytest.raises(RuntimeError, match="require_positive=True"):
        _all_reduce_external_relation_candidate_gradients(
            readout=readout,
            candidate_id="untrainable_candidate",
            diagnostic_rank=None,
            require_positive=True,
            dist=Dist,
            world_size=1,
            torch_module=torch,
        )


@pytest.mark.parametrize("value", (1e-30, 1e20))
def test_local_gradient_norm_is_stable_outside_fp32_square_range(value: float) -> None:
    gradient = torch.tensor([value, value], dtype=torch.float32)

    norm = _local_gradient_norm(gradient, name="test", torch_module=torch)

    assert norm == pytest.approx(math.sqrt(2.0) * value)


def test_distributed_boolean_consensus_rejects_optional_forward_divergence() -> None:
    class Dist:
        class ReduceOp:
            MIN = "min"
            MAX = "max"

        @staticmethod
        def all_reduce(value: torch.Tensor, *, op: str) -> None:
            if op == "min":
                value.zero_()
            elif op == "max":
                value.fill_(1)
            else:
                raise AssertionError(op)

    with pytest.raises(RuntimeError, match="wrong-time availability"):
        _distributed_uniform_boolean(
            True,
            name="wrong-time availability",
            device=torch.device("cpu"),
            dist=Dist,
            torch_module=torch,
        )


def test_distributed_boolean_consensus_returns_uniform_value() -> None:
    class Dist:
        class ReduceOp:
            MIN = "min"
            MAX = "max"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op in {"min", "max"}

    assert _distributed_uniform_boolean(
        True,
        name="factual prior",
        device=torch.device("cpu"),
        dist=Dist,
        torch_module=torch,
    )
    assert not _distributed_uniform_boolean(
        False,
        name="wrong-time state",
        device=torch.device("cpu"),
        dist=Dist,
        torch_module=torch,
    )


def test_distributed_boolean_union_synchronizes_optional_forward() -> None:
    class Dist:
        class ReduceOp:
            MAX = "max"

        @staticmethod
        def all_reduce(value: torch.Tensor, *, op: str) -> None:
            assert op == "max"
            value.fill_(1)

    assert _distributed_any_boolean(
        False,
        name="wrong-time state",
        device=torch.device("cpu"),
        dist=Dist,
        torch_module=torch,
    )


def test_distributed_pre_backward_failure_exchange_uses_ranked_cpu_evidence() -> None:
    class Dist:
        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def all_gather_object(outputs: list[object], local: object) -> None:
            outputs[:] = [
                local,
                {
                    "rank": 1,
                    "type": "RuntimeError",
                    "message": "rank-local objective failed",
                },
            ]

    assert _distributed_pre_backward_failures(
        None,
        rank=0,
        expected_world_size=2,
        dist=Dist,
    ) == (
        {
            "rank": 1,
            "type": "RuntimeError",
            "message": "rank-local objective failed",
        },
    )


def test_distributed_pre_backward_failure_exchange_preserves_local_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Dist:
        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def all_gather_object(outputs: list[object], local: object) -> None:
            outputs[:] = [local, None]

    error = ValueError("local objective failed")
    assert _distributed_pre_backward_failures(
        error,
        rank=0,
        expected_world_size=2,
        dist=Dist,
    ) == (
        {
            "rank": 0,
            "type": "ValueError",
            "message": "local objective failed",
        },
    )
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["event"] == "local_distributed_failure_before_exchange"
    assert diagnostic["rank"] == 0
    assert diagnostic["type"] == "ValueError"
    assert diagnostic["message"] == "local objective failed"
    assert "ValueError: local objective failed" in diagnostic["traceback"]


def test_distributed_pre_backward_failure_exchange_accepts_all_rank_success() -> None:
    class Dist:
        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def get_rank() -> int:
            return 1

        @staticmethod
        def all_gather_object(outputs: list[object], local: object) -> None:
            assert local is None
            outputs[:] = [None, None]

    assert (
        _distributed_pre_backward_failures(
            None,
            rank=1,
            expected_world_size=2,
            dist=Dist,
        )
        == ()
    )


def test_distributed_pre_backward_failure_exchange_supports_registered_four_rank_topology() -> None:
    class Dist:
        @staticmethod
        def get_world_size() -> int:
            return 4

        @staticmethod
        def get_rank() -> int:
            return 2

        @staticmethod
        def all_gather_object(outputs: list[object], local: object) -> None:
            assert local is None
            outputs[:] = [
                None,
                {
                    "rank": 1,
                    "type": "RuntimeError",
                    "message": "rank-one objective failed",
                },
                None,
                {
                    "rank": 3,
                    "type": "ValueError",
                    "message": "rank-three objective failed",
                },
            ]

    assert _distributed_pre_backward_failures(
        None,
        rank=2,
        expected_world_size=4,
        dist=Dist,
    ) == (
        {
            "rank": 1,
            "type": "RuntimeError",
            "message": "rank-one objective failed",
        },
        {
            "rank": 3,
            "type": "ValueError",
            "message": "rank-three objective failed",
        },
    )


def test_distributed_pre_backward_failure_exchange_rejects_unregistered_topology() -> None:
    class Dist:
        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def get_rank() -> int:
            return 0

    with pytest.raises(RuntimeError, match="registered topology"):
        _distributed_pre_backward_failures(
            None,
            rank=0,
            expected_world_size=4,
            dist=Dist,
        )


def test_distributed_pre_backward_failure_exchange_rejects_malformed_rank_evidence() -> None:
    class Dist:
        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def all_gather_object(outputs: list[object], local: object) -> None:
            outputs[:] = [local, {"rank": 0, "type": "RuntimeError", "message": "wrong rank"}]

    with pytest.raises(RuntimeError, match="malformed rank evidence"):
        _distributed_pre_backward_failures(
            None,
            rank=0,
            expected_world_size=2,
            dist=Dist,
        )


def test_relation_probe_local_validation_error_is_raised_on_every_rank() -> None:
    class Dist:
        @staticmethod
        def all_gather_object(outputs: list[object], local: object) -> None:
            outputs[:] = [
                local,
                {
                    "rank": 1,
                    "type": "RuntimeError",
                    "message": "task row absent",
                },
            ]

    with pytest.raises(RuntimeError, match="task row absent"):
        _distributed_raise_if_local_probe_error(
            dist=Dist,
            rank=0,
            world_size=2,
            stage="relation evidence",
            local_error=None,
        )


def test_distributed_action_state_digest_binds_rank_order() -> None:
    local = "a" * 64
    peer = "b" * 64

    class Dist:
        @staticmethod
        def all_gather_object(outputs: list[object], value: object) -> None:
            assert value == local
            outputs[:] = [local, peer]

    measured = _distributed_action_state_sha256(
        local,
        rank=0,
        world_size=2,
        dist=Dist,
    )
    assert measured == full_runner._canonical_digest(
        {
            "rank_local_action_state_sha256": [local, peer],
            "world_size": 2,
        }
    )

    class ReversedDist:
        @staticmethod
        def all_gather_object(outputs: list[object], value: object) -> None:
            assert value == local
            outputs[:] = [peer, local]

    assert measured != _distributed_action_state_sha256(
        local,
        rank=0,
        world_size=2,
        dist=ReversedDist,
    )


def test_relation_probe_local_validation_exchange_accepts_global_success() -> None:
    class Dist:
        @staticmethod
        def all_gather_object(outputs: list[object], local: object) -> None:
            assert local is None
            outputs[:] = [None, None]

    _distributed_raise_if_local_probe_error(
        dist=Dist,
        rank=1,
        world_size=2,
        stage="relation evidence",
        local_error=None,
    )


def test_relation_probe_source_selection_precedes_weight_loading_and_has_no_model_inputs() -> None:
    assert tuple(inspect.signature(_select_relation_geometry_source_sample).parameters) == (
        "args",
        "stream_plan",
        "dataset",
        "physical_sidecar",
        "task_identity_resolver",
    )
    source = _source()
    selection = source.index(
        "relation_probe_sample_selection = _select_relation_geometry_source_sample("
    )
    assert selection < source.index("with init_empty_weights(), no_init_weights():")
    assert selection < source.index("load_model_weights(")


def test_gradient_audit_schedule_requires_prior_and_current_targets_on_every_rank() -> None:
    keys = (("a", "b"), ("c", "d"), ("e", "f"))
    sources = {key: index for index, key in enumerate("abcdef")}

    class Plan:
        total_steps = len(keys)

        @staticmethod
        def global_batch(optimizer_step: int) -> SimpleNamespace:
            return SimpleNamespace(
                transitions=tuple(
                    SimpleNamespace(
                        sample=SimpleNamespace(sample_key=key),
                        transition_index=0 if key == "c" else 1,
                    )
                    for key in keys[optimizer_step]
                )
            )

    with pytest.raises(ValueError, match="state-bootstrap"):
        _validate_gradient_audit_target_coverage(
            stream_plan=Plan(),
            audit_steps=(1, 3),
            source_global_index_for_sample=sources.__getitem__,
            target_has_support=lambda **_: True,
        )
    _validate_gradient_audit_target_coverage(
        stream_plan=Plan(),
        audit_steps=(3,),
        source_global_index_for_sample=sources.__getitem__,
        target_has_support=lambda **_: True,
    )
    with pytest.raises(ValueError, match=r"step=2.*next_eligible=3"):
        _validate_gradient_audit_target_coverage(
            stream_plan=Plan(),
            audit_steps=(2,),
            source_global_index_for_sample=sources.__getitem__,
            target_has_support=lambda **_: True,
        )
    with pytest.raises(ValueError, match=r"zero_supported_target_mass.*next_eligible=None"):
        _validate_gradient_audit_target_coverage(
            stream_plan=Plan(),
            audit_steps=(3,),
            source_global_index_for_sample=sources.__getitem__,
            target_has_support=lambda *, source_global_index: source_global_index != sources["e"],
        )


def test_report_objective_treats_unobserved_predictive_family_as_zero() -> None:
    _recompute_report_objective(
        item={
            "normalized_terms": {
                "action": 0.21,
                "correction/dino_video": 0.0,
                "set/support": 0.09,
                "set/existence": 0.0,
                "set/task": 0.0,
                "set/task_dense": 0.0,
            },
            "valid_counts": {
                "action": 1,
                "correction/dino_video": 0,
                "set/support": 1,
                "set/existence": 0,
                "set/task": 0,
                "set/task_dense": 0,
            },
            "official_action_loss": 0.2,
            "official_moe_regularizer": 0.01,
            "official_policy_loss": 0.21,
            "objective_total": 0.21036,
        },
        mode="omitted_static",
        contract={
            "predictive_family_weight": 0.004,
            "structural_family_weight": 0.004,
            "predictive_term_weight": 1.0,
            "current_grid_term_weight": 1.0,
            "omitted_static_term_weight": 1.0,
            "support_weight": 1.0,
            "existence_weight": 1.0,
            "task_weight": 1.0,
            "dense_task_weight": 1.0,
            "ownership_weight": 1.0,
        },
    )


def test_report_objective_ignores_zero_weight_ownership_nll() -> None:
    _recompute_report_objective(
        item={
            "normalized_terms": {
                "action": 0.21,
                "set/ownership": 0.09,
                "set/ownership_nll": 999.0,
                "set/support": 0.0,
                "set/existence": 0.0,
                "set/task": 0.0,
                "set/task_dense": 0.0,
            },
            "valid_counts": {
                "action": 1,
                "set/ownership": 1,
                "set/ownership_nll": 128,
                "set/support": 0,
                "set/existence": 0,
                "set/task": 0,
                "set/task_dense": 0,
            },
            "official_action_loss": 0.2,
            "official_moe_regularizer": 0.01,
            "official_policy_loss": 0.21,
            "objective_total": 0.21036,
        },
        mode="omitted_static",
        contract={
            "predictive_family_weight": 0.004,
            "structural_family_weight": 0.004,
            "predictive_term_weight": 1.0,
            "current_grid_term_weight": 1.0,
            "omitted_static_term_weight": 1.0,
            "support_weight": 0.0,
            "existence_weight": 1.0,
            "task_weight": 1.0,
            "dense_task_weight": 1.0,
            "ownership_weight": 1.0,
        },
    )


def test_report_objective_recomputes_entity_conditional_ownership_mixture() -> None:
    common_item = {
        "normalized_terms": {
            "action": 0.21,
            "set/ownership": 0.1,
            "set/ownership_nll": 999.0,
            "set/ownership_entity": 0.3,
            "set/support": 0.0,
            "set/existence": 0.0,
            "set/task": 0.0,
            "set/task_dense": 0.0,
        },
        "valid_counts": {
            "action": 1,
            "set/ownership": 1,
            "set/ownership_nll": 1,
            "set/ownership_entity": 2,
            "set/support": 0,
            "set/existence": 0,
            "set/task": 0,
            "set/task_dense": 0,
        },
        "official_action_loss": 0.2,
        "official_moe_regularizer": 0.01,
        "official_policy_loss": 0.21,
        "objective_total": 0.2108,
    }
    contract = {
        "predictive_family_weight": 0.004,
        "structural_family_weight": 0.004,
        "predictive_term_weight": 1.0,
        "current_grid_term_weight": 1.0,
        "omitted_static_term_weight": 1.0,
        "support_weight": 0.0,
        "existence_weight": 1.0,
        "task_weight": 1.0,
        "dense_task_weight": 1.0,
        "ownership_weight": 1.0,
        "ownership_estimator": TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
    }

    _recompute_report_objective(
        item=common_item,
        mode="omitted_static",
        contract=contract,
    )

    missing_entity = deepcopy(common_item)
    missing_entity["normalized_terms"].pop("set/ownership_entity")
    missing_entity["valid_counts"].pop("set/ownership_entity")
    with pytest.raises(ValueError, match="structural term schema differs"):
        _recompute_report_objective(
            item=missing_entity,
            mode="omitted_static",
            contract=contract,
        )

    with pytest.raises(ValueError, match="structural term schema differs"):
        _recompute_report_objective(
            item=common_item,
            mode="omitted_static",
            contract={
                **contract,
                "ownership_estimator": TOKEN_MICRO_OWNERSHIP,
            },
        )


def test_report_objective_recomputes_shared_multi_depth_ownership_weight() -> None:
    normalized = {
        "action": 0.21,
        "set/ownership": 0.1,
        "set/ownership_q1": 0.2,
        "set/ownership_q2": 0.3,
        "set/ownership_q3": 0.4,
        "set/ownership_nll": 100.0,
        "set/ownership_nll_q1": 200.0,
        "set/ownership_nll_q2": 300.0,
        "set/ownership_nll_q3": 400.0,
        "set/support": 0.0,
        "set/existence": 0.0,
        "set/task": 0.0,
        "set/task_dense": 0.0,
    }
    _recompute_report_objective(
        item={
            "normalized_terms": normalized,
            "valid_counts": {
                name: 0
                if name in {"set/support", "set/existence", "set/task", "set/task_dense"}
                else 1
                for name in normalized
            },
            "official_action_loss": 0.2,
            "official_moe_regularizer": 0.01,
            "official_policy_loss": 0.21,
            "objective_total": 0.211,
        },
        mode="omitted_static",
        contract={
            "predictive_family_weight": 0.004,
            "structural_family_weight": 0.004,
            "predictive_term_weight": 1.0,
            "current_grid_term_weight": 1.0,
            "omitted_static_term_weight": 1.0,
            "support_weight": 0.0,
            "existence_weight": 1.0,
            "task_weight": 1.0,
            "dense_task_weight": 1.0,
            "ownership_weight": 1.0,
        },
    )


def test_report_objective_rejects_noncontiguous_ownership_depths() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _recompute_report_objective(
            item={
                "normalized_terms": {
                    "action": 0.21,
                    "set/ownership": 0.1,
                    "set/ownership_q1": 0.2,
                    "set/ownership_q3": 0.4,
                },
                "valid_counts": {
                    "action": 1,
                    "set/ownership": 1,
                    "set/ownership_q1": 1,
                    "set/ownership_q3": 1,
                },
                "official_action_loss": 0.2,
                "official_moe_regularizer": 0.01,
                "official_policy_loss": 0.21,
                "objective_total": 0.211,
            },
            mode="omitted_static",
            contract={
                "predictive_family_weight": 0.004,
                "structural_family_weight": 0.004,
                "predictive_term_weight": 1.0,
                "current_grid_term_weight": 1.0,
                "omitted_static_term_weight": 1.0,
                "support_weight": 0.0,
                "existence_weight": 1.0,
                "task_weight": 1.0,
                "dense_task_weight": 1.0,
                "ownership_weight": 1.0,
            },
        )


def test_step_row_bindings_are_monotonic_and_assignment_exact() -> None:
    diagnostic = {
        "task_logits": [1.0, -1.0],
        "identity_keys": ["pink_block", "drawer"],
        "row_to_track": [0, 1],
        "binding_start_phase": [0, 1],
    }
    _validate_step_row_bindings(
        prior_value=[[["drawer", 1]]],
        current_value=[[["drawer", 1], ["pink_block", 0]]],
        reported_birth_count=1,
        task_row_diagnostics=[diagnostic],
        expected_batch_size=1,
    )

    with pytest.raises(ValueError, match="removed or rebound"):
        _validate_step_row_bindings(
            prior_value=[[["drawer", 0]]],
            current_value=[[["drawer", 1], ["pink_block", 0]]],
            reported_birth_count=1,
            task_row_diagnostics=[diagnostic],
            expected_batch_size=1,
        )

    future_birth_diagnostic = {
        "task_logits": [1.0, -1.0],
        "identity_keys": ["pink_block", "drawer"],
        "row_to_track": [0, 1],
        "binding_start_phase": [1, 3],
    }
    _validate_step_row_bindings(
        prior_value=[[]],
        current_value=[[["pink_block", 0]]],
        reported_birth_count=1,
        task_row_diagnostics=[future_birth_diagnostic],
        expected_batch_size=1,
    )

    with pytest.raises(ValueError, match="loss-only future"):
        _validate_step_row_bindings(
            prior_value=[[]],
            current_value=[[["drawer", 1], ["pink_block", 0]]],
            reported_birth_count=2,
            task_row_diagnostics=[future_birth_diagnostic],
            expected_batch_size=1,
        )

    with pytest.raises(ValueError, match="birth count"):
        _validate_step_row_bindings(
            prior_value=[[["drawer", 1]]],
            current_value=[[["drawer", 1], ["pink_block", 0]]],
            reported_birth_count=0,
            task_row_diagnostics=[diagnostic],
            expected_batch_size=1,
        )

    with pytest.raises(ValueError, match="assignment and persisted"):
        _validate_step_row_bindings(
            prior_value=[[]],
            current_value=[[["drawer", 0], ["pink_block", 1]]],
            reported_birth_count=2,
            task_row_diagnostics=[diagnostic],
            expected_batch_size=1,
        )


def test_report_row_bindings_link_temporal_lanes_and_reset_cleanly() -> None:
    lane_bindings: dict[int, dict[str, int]] = {}
    _advance_report_row_binding_continuity(
        lane_bindings,
        lane_ids=[7],
        frame_indices=[0],
        state_ages=[0],
        step_bindings=(({}, {"object/a": 1}),),
    )
    _advance_report_row_binding_continuity(
        lane_bindings,
        lane_ids=[7],
        frame_indices=[1],
        state_ages=[1],
        step_bindings=(({"object/a": 1}, {"object/a": 1, "object/b": 0}),),
    )

    with pytest.raises(ValueError, match="cross-step"):
        _advance_report_row_binding_continuity(
            lane_bindings,
            lane_ids=[7],
            frame_indices=[2],
            state_ages=[2],
            step_bindings=(({"object/a": 0, "object/b": 1}, {"object/a": 0, "object/b": 1}),),
        )
    with pytest.raises(ValueError, match="reset lane retained"):
        _advance_report_row_binding_continuity(
            lane_bindings,
            lane_ids=[7],
            frame_indices=[0],
            state_ages=[0],
            step_bindings=(({"object/a": 1}, {"object/a": 0}),),
        )

    _advance_report_row_binding_continuity(
        lane_bindings,
        lane_ids=[7],
        frame_indices=[0],
        state_ages=[0],
        step_bindings=(({}, {"object/a": 0}),),
    )
    assert lane_bindings == {7: {"object/a": 0}}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _passing_predictive_target_audit() -> tuple[dict[str, object], dict[str, object]]:
    digest = "b" * 64
    dataset_digest = "c" * 64
    sidecar_digest = "d" * 64
    stream_digest = "e" * 64
    temporal_digest = "f" * 64
    pair_digest = "1" * 64
    horizons = (1, 2, 64)
    query_digest = native_predictive_query_schema_digest(
        target_space="dino_video",
        route_id=0,
        horizons=horizons,
    )
    coverage_digest = native_predictive_coverage_digest(
        dataset_tree_sha256=dataset_digest,
        stream_plan_sha256=stream_digest,
        temporal_estimator_sha256=temporal_digest,
        pair_keys_sha256=pair_digest,
        expected_record_count=8,
        horizons=horizons,
    )
    contract = PredictiveCacheContract(
        dataset_id="calvin",
        dataset_revision="test",
        split_name="training",
        dataset_tree_sha256=dataset_digest,
        physical_sidecar_manifest_sha256=sidecar_digest,
        lingbot_source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
        lingbot_checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
        teacher_config_sha256="2" * 64,
        teacher_checkpoint_sha256="3" * 64,
        query_schema_sha256=query_digest,
        horizons=horizons,
        stream_plan_sha256=stream_digest,
        temporal_estimator_sha256=temporal_digest,
        pair_keys_sha256=pair_digest,
        coverage_sha256=coverage_digest,
        expected_record_count=8,
    )
    diagnostics = predictive_latent_diagnostics(
        torch.tensor(
            [
                [2.0, -2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, -2.0],
                [2.1, -1.9, 0.1, -0.1],
                [0.1, -0.1, 2.1, -1.9],
            ]
        ),
        identity_keys=("object/a", "object/b", "object/a", "object/b"),
        target_group_keys=("frame/1", "frame/1", "frame/2", "frame/2"),
    )
    ready, failures = predictive_target_pretraining_readiness(diagnostics)
    assert ready and not failures
    report: dict[str, object] = {
        "cache_contract": asdict(contract),
        "cache_manifest_sha256": digest,
        "diagnostics": diagnostics.as_dict(),
        "encoder_digest": contract.encoder_digest,
        "horizon_record_counts": {"1": 4, "2": 4, "64": 0},
        "identity_count": 2,
        "interpretation": {
            "numerical_status": "no_obvious_numerical_collapse",
            "pretraining_readiness": "PASS",
            "pretraining_readiness_failures": [],
            "retrieval_is_computable": True,
            "scientific_acceptance": False,
            "scientific_acceptance_reason": (
                "target statistics cannot establish source-conditioned learnability, "
                "shared-host gradient reach, object semantics or action benefit"
            ),
        },
        "maximum_samples": 4,
        "sample_selection": "lowest-sha256-priority-without-replacement/v1",
        "sample_selection_sha256": "4" * 64,
        "sampled_target_count": 4,
        "scanned_object_target_count": 4,
        "scanned_record_count": 8,
        "schema": PREDICTIVE_TARGET_AUDIT_SCHEMA,
        "supported_object_target_count": 4,
        "visible_support_diagnostics": {
            "supported_count": 4,
            "sampled_count": 4,
            "minimum_visible_image_fraction": 0.1,
            "mean_visible_image_fraction": 0.2,
            "maximum_visible_image_fraction": 0.3,
            "sampled_p05_visible_image_fraction": 0.11,
            "sampled_median_visible_image_fraction": 0.2,
            "sampled_p95_visible_image_fraction": 0.29,
        },
        "zero_support_object_target_count": 0,
    }
    predictive_report: dict[str, object] = {
        "cache_manifest_sha256": digest,
        "coverage_sha256": coverage_digest,
        "expected_record_count": 8,
        "pair_keys_sha256": pair_digest,
        "stream_plan_sha256": stream_digest,
        "teacher_encoder_digest": contract.encoder_digest,
        "temporal_estimator_sha256": temporal_digest,
    }
    return report, predictive_report


def _passing_predictive_temporal_audit() -> tuple[
    dict[str, object], dict[str, object], dict[str, object], str, tuple[int, ...]
]:
    target_report, predictive_report = _passing_predictive_target_audit()
    predictive_report = dict(predictive_report)
    predictive_report["expected_record_count"] = 4
    contract = target_report["cache_contract"]
    assert isinstance(contract, dict)
    sidecar_digest = str(contract["physical_sidecar_manifest_sha256"])
    horizons = (1, 2)
    current_report: dict[str, object] = {
        "cache_manifest_sha256": "5" * 64,
        "expected_record_count": 4,
        "teacher_encoder_digest": "6" * 64,
    }
    current = torch.tensor(
        [
            [2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, -2.0],
            [1.8, -2.1, 0.1, 0.0],
            [0.1, 0.0, 1.9, -2.2],
        ]
    )
    future = current + torch.tensor(
        [
            [0.2, 0.0, 0.1, 0.0],
            [0.0, -0.2, 0.0, 0.1],
            [0.0, 0.1, -0.2, 0.0],
            [-0.1, 0.0, 0.0, 0.2],
        ]
    )
    diagnostics = predictive_temporal_diagnostics(
        current,
        future,
        identity_keys=("object/a", "object/b", "object/a", "object/b"),
        horizons=(1, 1, 2, 2),
    )
    ready, failures = predictive_temporal_pretraining_readiness(diagnostics)
    assert ready and not failures
    current_diagnostics = predictive_latent_diagnostics(
        current,
        identity_keys=("object/a", "object/b", "object/a", "object/b"),
        target_group_keys=("frame-1", "frame-1", "frame-2", "frame-2"),
    )
    current_ready, current_failures = predictive_target_pretraining_readiness(current_diagnostics)
    assert current_ready and not current_failures
    current_support = predictive_visible_support_diagnostics(
        torch.full((4,), 0.5),
        supported_count=4,
        total_importance=2.0,
        minimum_importance=0.5,
        maximum_importance=0.5,
    )
    report: dict[str, object] = {
        "current_cache_manifest_sha256": current_report["cache_manifest_sha256"],
        "current_correction_diagnostics": current_diagnostics.as_dict(),
        "current_correction_identity_count": 2,
        "current_correction_sample_selection_sha256": "8" * 64,
        "current_correction_sampled_target_count": 4,
        "current_correction_scanned_object_target_count": 4,
        "current_correction_supported_object_target_count": 4,
        "current_correction_visible_support_diagnostics": current_support.as_dict(),
        "current_correction_zero_support_object_target_count": 0,
        "current_encoder_digest": current_report["teacher_encoder_digest"],
        "diagnostics": diagnostics.as_dict(),
        "feature_pairing": PREDICTIVE_TEMPORAL_FEATURE_PAIRING,
        "future_cache_manifest_sha256": predictive_report["cache_manifest_sha256"],
        "future_encoder_digest": predictive_report["teacher_encoder_digest"],
        "horizon_supported_pair_counts": {"1": 2, "2": 2},
        "interpretation": {
            "controlled_future_temporal_pretraining_readiness": "PASS",
            "controlled_future_temporal_pretraining_readiness_failures": [],
            "current_correction_pretraining_readiness": "PASS",
            "current_correction_pretraining_readiness_failures": [],
            "pretraining_readiness": "PASS",
            "pretraining_readiness_failures": [],
            "scientific_acceptance": False,
            "scientific_acceptance_reason": (
                "target-bank statistics do not establish source-conditioned prediction, "
                "action conditioning or action benefit"
            ),
        },
        "matched_future_record_count": 4,
        "maximum_samples": 4,
        "physical_sidecar_manifest_sha256": sidecar_digest,
        "sample_selection": "lowest-sha256-priority-without-replacement/v1",
        "sample_selection_sha256": "7" * 64,
        "sampled_pair_count": 4,
        "scanned_current_record_count": current_report["expected_record_count"],
        "schema": PREDICTIVE_TEMPORAL_AUDIT_SCHEMA,
        "supported_aligned_pair_count": 4,
    }
    return report, predictive_report, current_report, sidecar_digest, horizons


def _source() -> str:
    return TOOL.read_text()


def _gate_evidence(
    tmp_path: Path,
    gate: str,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> tuple[tuple[str, Path], ...]:
    values = []
    for name, schema in TRAINING_GATE_EVIDENCE_SCHEMAS[gate]:
        if name in {"preflight", "static_causality", "frozen_local_contract"}:
            values.append((name, preflight_report_factory(tmp_path / f"{gate}.{name}.json")))
            continue
        if name == "predictive_objective_decision":
            record = tmp_path / f"{gate}.{name}.md"
            record.write_text("owner-reviewed ADR-82 fixture")
            report = build_predictive_objective_decision(
                reviewer="local-test",
                temporal_objective=IMPLEMENTED_PREDICTIVE_OBJECTIVE,
                visible_support_weighting=IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
                minimum_visible_fraction=0.0,
                decision_record=record,
            )
            path = tmp_path / f"{gate}.{name}.json"
            path.write_text(json.dumps(report, sort_keys=True))
            values.append((name, path))
            continue
        if name in {"neutral", "released_isolation"}:
            values.append((name, smoke_report_factory(tmp_path / f"{gate}.{name}.json")))
            continue
        if name in {"fresh_update", "cold_resume"}:
            values.append(
                (
                    name,
                    g0_report_factory(
                        tmp_path / f"{gate}.{name}.json",
                        phase="fresh" if name == "fresh_update" else "resume",
                    ),
                )
            )
            continue
        report = {"schema": schema, "status": "PASS"}
        path = tmp_path / f"{gate}.{name}.json"
        path.write_text(json.dumps(report, sort_keys=True))
        values.append((name, path))
    return tuple(values)


def _runner_args(tmp_path: Path) -> argparse.Namespace:
    directories = {
        name: tmp_path / name
        for name in (
            "checkpoint",
            "current-grid",
            "physical",
            "predictive",
            "processor",
            "run",
            "source",
            "split",
        )
    }
    for path in directories.values():
        path.mkdir()
    files = {
        name: tmp_path / f"{name}.json"
        for name in (
            "data",
            "current-grid-report",
            "manifest",
            "norm",
            "patch",
            "physical-sidecar-manifest",
            "physical-visual-acceptance",
            "predictive-report",
            "predictive-teacher-causality-audit",
            "predictive-target-audit",
            "predictive-temporal-audit",
            "robot",
            "training",
        )
    }
    for path in files.values():
        path.write_text("{}")
    digest = "1" * 64
    return argparse.Namespace(
        phase="fresh",
        source_checkout=directories["source"],
        patch=files["patch"],
        training_config=files["training"],
        robot_config=files["robot"],
        data_config=files["data"],
        checkpoint_dir=directories["checkpoint"],
        processor_dir=directories["processor"],
        dataset_split=directories["split"],
        dataset_manifest=files["manifest"],
        norm_stats=files["norm"],
        physical_sidecar_root=directories["physical"],
        physical_sidecar_manifest=files["physical-sidecar-manifest"],
        physical_sidecar_manifest_sha256=digest,
        physical_visual_acceptance=files["physical-visual-acceptance"],
        physical_visual_acceptance_sha256=digest,
        predictive_cache_root=directories["predictive"],
        predictive_cache_build_report=files["predictive-report"],
        predictive_cache_build_report_sha256=digest,
        predictive_teacher_causality_audit=files["predictive-teacher-causality-audit"],
        predictive_teacher_causality_audit_sha256=digest,
        predictive_target_audit=files["predictive-target-audit"],
        predictive_target_audit_sha256=digest,
        predictive_temporal_audit=files["predictive-temporal-audit"],
        predictive_temporal_audit_sha256=digest,
        current_grid_cache_root=directories["current-grid"],
        current_grid_cache_build_report=files["current-grid-report"],
        current_grid_cache_build_report_sha256=digest,
        run_dir=directories["run"],
        authorization_manifest=None,
        authorization_manifest_sha256=None,
        load_global_step=0,
        invocation_steps=1,
        total_planned_steps=10,
        checkpoint_publication="always",
        seed=7,
        capacity=16,
        maximum_control_tokens=8,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        maximum_peak_reserved_gib=39.0,
        fsdp2_placement="cpu-offload",
        cuda_allocator="native",
        local_bptt_probability=0.1,
        overshoot_probability=0.05,
        source_mask_probability=0.1,
        behavior_conditioned_prediction=False,
        behavior_causal_probe_output=None,
        behavior_causal_probe_evidence=None,
        behavior_causal_probe_evidence_sha256=None,
        behavior_posterior_control_probe_output=None,
        behavior_posterior_control_probe_evidence=None,
        behavior_posterior_control_probe_evidence_sha256=None,
        behavior_g1_predecessor_report_sha256=None,
        source_prediction_mode="omitted_static",
        source_mask_token_fraction=0.0625,
        maximum_optimizer_lag=8,
        lane_interleave_factor=1,
        predictive_weight=1.0,
        structural_weight=1.0,
        evidence_profile="acceptance",
        gradient_audit_steps=(2, 3),
        support_weight=0.0,
        existence_weight=1.0,
        task_weight=1.0,
        task_relation_estimator=LOCAL_BALANCED_TASK_RELATION,
        dense_task_weight=1.0,
        ownership_weight=1.0,
        relation_supervision_layers=(),
        predictive_term_weight=1.0,
        current_grid_term_weight=1.0,
        omitted_static_term_weight=1.0,
        predictive_loss_power=1.0,
        minimum_supervised_fraction=1.0,
        predictive_cache_memory_shards=2,
        current_grid_cache_memory_shards=2,
        visual_audit_every=0,
        predictive_fixed_batch_arm=None,
        predictive_fixed_batch_curve_points=0,
        predictive_fixed_batch_output=None,
        relation_geometry_fixed_batch_arm=None,
        relation_geometry_fixed_batch_curve_points=0,
        relation_geometry_fixed_batch_sample_step=0,
        relation_geometry_fixed_batch_output=None,
        relation_geometry_fixed_batch_visual_root=None,
        representation_evaluation_plan=None,
        representation_evaluation_plan_sha256=None,
        representation_evaluation_baseline=None,
        representation_evaluation_baseline_sha256=None,
        representation_warm_evaluation_plan=None,
        representation_warm_evaluation_plan_sha256=None,
        representation_evaluation_steps=(),
        representation_split=None,
        representation_split_sha256=None,
        representation_task_intervention_plan=None,
        representation_task_intervention_plan_sha256=None,
        fixed_observation_pair_plan=None,
        fixed_observation_pair_plan_sha256=None,
        fixed_observation_training_audit=None,
        fixed_observation_training_audit_sha256=None,
        fixed_observation_evaluation_plan=None,
        fixed_observation_evaluation_plan_sha256=None,
        fixed_observation_validation_audit=None,
        fixed_observation_validation_audit_sha256=None,
        fixed_observation_heldout_audit=None,
        fixed_observation_heldout_audit_sha256=None,
        training_stage="action",
    )


def _behavior_g0_evidence() -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema": BEHAVIOR_CAUSAL_PROBE_DISTRIBUTED_SCHEMA,
        "status": "PASS",
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "patch_sha256": digest,
        "behavior_graph_sha256": digest,
        "implementation_sha256": digest,
        "model_family_sha256": digest,
        "plan_sha256": digest,
        "weight_boundary": "released_pre_optimizer",
        "optimizer_updates": 0,
        "sample_keys_by_rank": [["rank-0-sample"], ["rank-1-sample"]],
        "rank_reports": [
            {
                "schema": BEHAVIOR_CAUSAL_PROBE_SCHEMA,
                "status": "PASS",
                "rank": rank,
                "cold_deploy_omits_future_controls": True,
                "deploy_bit_identical": True,
                "fresh_primary_rerun_bit_identical": True,
                "deploy_isolation": "separate_same_weight_auxiliary_forward",
                "deploy_tensor_count": 97,
                "horizon": 1,
                "intervention_prediction_changed": {
                    "peer_replace_shuffle": True,
                    "reverse": False,
                    "zero": True,
                },
                "elapsed_s": 1.0,
                "peak_cuda_allocated_bytes": 1,
                "peak_cuda_reserved_bytes": 2,
            }
            for rank in range(FULL_WORLD_SIZE)
        ],
    }


def _behavior_g2_evidence() -> dict[str, object]:
    digest = "b" * 64
    posterior = {"zero": 0.2, "batch_shift": 0.1}
    control = {"zero": 0.3, "batch_shift": 0.05}
    rank_reports = []
    for rank in range(FULL_WORLD_SIZE):
        boundary = {
            "lane_snapshot_sha256": f"{rank + 1}" * 64,
            "model_local_state_sha256": f"{rank + 2}" * 64,
            "optimizer_local_state_sha256": f"{rank + 3}" * 64,
            "rank_rng_state_sha256": f"{rank + 4}" * 64,
        }
        rank_reports.append(
            {
                "rank": rank,
                "sample_keys": [f"rank-{rank}-sample"],
                "peer_sample_keys": [f"rank-{1 - rank}-sample"],
                "tasks": [f"task-{rank}"],
                "elapsed_s": 1.0,
                "training_prediction_sha256": digest,
                "factual_prediction_sha256": digest,
                "training_prediction_bit_identical": True,
                "diagnostics": {
                    "posterior_margins_at_factual_control": posterior,
                    "control_margins_at_factual_posterior": control,
                },
                "target": {
                    "modality": "vision",
                    "source_batch_digest": digest,
                    "target_data_digest": digest,
                    "encoder_digest": digest,
                    "query_schema_digest": digest,
                    "validity_semantics": "fixture support",
                    "track_identity_keys": [["object/a", "object/b"]],
                    "valid_count": 2,
                    "importance_sum": 2.0,
                    "importance_min": 1.0,
                    "importance_max": 1.0,
                },
                "assignment": {
                    "row_to_track": [[0, 1]],
                    "binding_start_phase": [[1, 3]],
                    "identity_source_phase": 1,
                    "row_binding_valid": [[True, False]],
                    "sha256": full_runner._canonical_digest(
                        {
                            "row_to_track": [[0, 1]],
                            "binding_start_phase": [[1, 3]],
                            "identity_source_phase": 1,
                            "row_binding_valid": [[True, False]],
                        }
                    ),
                },
                "request_sha256": digest,
                "rng_sha256": digest,
                "rng_unchanged": True,
                "optimizer_state_unchanged": True,
                "posterior_bank_unchanged": True,
                "moe_routing_bias_unchanged": True,
                "loaded_boundary_sha256": boundary,
                "factual_repeat_bit_identical": True,
                "loss_only_labels_visible_to_model": False,
                "peak_cuda_allocated_bytes": 1,
                "peak_cuda_reserved_bytes": 2,
            }
        )
    return {
        "schema": BEHAVIOR_POSTERIOR_CONTROL_PROBE_DISTRIBUTED_SCHEMA,
        "status": "PASS",
        "scientific_status": "PASS",
        "scientific_rule": (
            "both rank-mean factual-axis margins must be positive for every zero and batch-shift "
            "intervention"
        ),
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "patch_sha256": digest,
        "behavior_graph_sha256": digest,
        "current_implementation_sha256": digest,
        "current_execution_contract_sha256": digest,
        "model_family_sha256": digest,
        "plan_sha256": digest,
        "g0_evidence_sha256": digest,
        "g1_predecessor_report_sha256": digest,
        "g1_predecessor_implementation_sha256": digest,
        "g1_predecessor_execution_contract_sha256": digest,
        "input_global_step": 2,
        "weight_boundary": "loaded_g1_step2_pre_optimizer",
        "optimizer_updates": 0,
        "checkpoint_publication": "never",
        "loaded_boundary_sha256_by_rank": [
            report["loaded_boundary_sha256"] for report in rank_reports
        ],
        "aggregate_posterior_margins_at_factual_control": posterior,
        "aggregate_control_margins_at_factual_posterior": control,
        "rank_reports": rank_reports,
    }


def _representation_split(
    path: Path,
    *,
    training_steps: int,
    reference_evaluation: bool = False,
) -> RepresentationTrialSplit:
    digest = "9" * 64
    segment = {
        "task_key": "move_blue_block",
        "segment_index": 1,
        "source_start": 10,
        "source_end": 20,
    }
    split = RepresentationTrialSplit(
        dataset_id="calvin",
        dataset_revision="fixture",
        dataset_manifest_sha256=digest,
        comparison_id=FULL_COMPARISON_ID,
        stream_plan_sha256=digest,
        partition_seed=7,
        training_steps=training_steps,
        training_sample_count=training_steps * FULL_WORLD_SIZE,
        training_sample_keys_sha256=digest,
        training_source_global_indices_sha256=digest,
        training_segment_indices=(0,),
        training_source_episode_indices=(0,),
        segments_per_task=1,
        validation_segments=(
            RepresentationEvaluationSegment(
                **segment,
                source_episode_index=1,
            ),
        ),
        heldout_segments=(
            RepresentationEvaluationSegment(
                **{
                    **segment,
                    "segment_index": 2,
                    "source_start": 20,
                    "source_end": 30,
                },
                source_episode_index=2,
            ),
        ),
        schema=(
            REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
            if reference_evaluation
            else REPRESENTATION_TRIAL_SPLIT_SCHEMA
        ),
        evaluation_reference_split_artifact_sha256=("8" * 64 if reference_evaluation else None),
    )
    split.write(path)
    return split


def _resume_extra() -> dict[str, object]:
    digest = "a" * 64
    return {
        "boundary_sha256": {
            "lane_snapshot_sha256": digest,
            "model_local_state_sha256": digest,
            "optimizer_local_state_sha256": digest,
            "rank_rng_state_sha256": digest,
        },
        "behavior_conditioning_sha256": None,
        "execution_contract_sha256": digest,
        "global_step": 3,
        "implementation_sha256": digest,
        "lane_snapshot": b"lane",
        "model_family_sha256": digest,
        "next_optimizer_step": 3,
        "optimizer_local_moment_elements": 16,
        "optimizer_state_entries": 2,
        "plan_sha256": digest,
        "rank": 0,
        "rank_rng_state": {"python_json": b"[]"},
        "schema": FULL_EXTRA_STATE_SCHEMA,
        "source_digest": digest,
        "temporal_estimator_sha256": digest,
        "world_size": FULL_WORLD_SIZE,
    }


def _validate_resume(value: object) -> dict[str, object]:
    digest = "a" * 64
    return _validate_resume_extra(
        value,
        expected_global_step=3,
        expected_implementation_sha256=digest,
        expected_model_family_sha256=digest,
        expected_execution_sha256=digest,
        expected_plan_sha256=digest,
        expected_temporal_sha256=digest,
        expected_source_digest=digest,
        expected_behavior_conditioning_sha256=None,
        rank=0,
    )


def test_full_runner_delays_every_accelerator_and_host_import() -> None:
    tree = ast.parse(_source())
    top_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_from_imports = {
        (node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    assert {"lingbotvla", "numpy", "torch", "transformers"}.isdisjoint(
        top_imports | top_from_imports
    )


def test_full_runner_defaults_to_pinned_native_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PICF_LINGBOT_NATIVE_SOURCE", raising=False)
    monkeypatch.setattr(sys, "argv", [str(TOOL), "--phase", "fresh"])
    args = _parse_args()
    expected = ROOT / CHECKOUT_RELATIVE_PATH
    assert args.source_checkout == expected
    assert args.training_config == expected / "configs/vla/robotwin/robotwin.yaml"
    assert args.source_prediction_mode == "omitted_static"
    assert args.training_stage == "action"
    assert args.checkpoint_publication == "always"
    assert args.representation_split is None
    assert args.representation_split_sha256 is None
    assert args.representation_task_intervention_plan is None
    assert args.representation_task_intervention_plan_sha256 is None
    assert args.fixed_observation_pair_plan is None
    assert args.fixed_observation_pair_plan_sha256 is None
    assert args.fixed_observation_training_audit is None
    assert args.fixed_observation_training_audit_sha256 is None
    assert args.fixed_observation_evaluation_plan is None
    assert args.fixed_observation_evaluation_plan_sha256 is None
    assert args.fixed_observation_validation_audit is None
    assert args.fixed_observation_validation_audit_sha256 is None
    assert args.fixed_observation_heldout_audit is None
    assert args.fixed_observation_heldout_audit_sha256 is None
    assert args.predictive_weight is None
    assert args.structural_weight is None
    assert args.evidence_profile == "acceptance"
    assert args.gradient_audit_steps is None
    assert args.visual_audit_every == 0
    assert args.learning_rate == 1e-4
    assert args.capacity == 16
    assert args.maximum_control_tokens == 8
    assert args.ownership_estimator == TOKEN_MICRO_OWNERSHIP
    assert args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD
    assert args.cuda_allocator == "native"
    assert args.local_bptt_probability == 0.10
    assert args.overshoot_probability == 0.05
    assert args.source_mask_probability == 0.10
    assert args.behavior_conditioned_prediction is False
    assert args.behavior_causal_probe_output is None
    assert args.behavior_causal_probe_evidence is None
    assert args.behavior_causal_probe_evidence_sha256 is None
    assert args.behavior_posterior_control_probe_output is None
    assert args.behavior_posterior_control_probe_evidence is None
    assert args.behavior_posterior_control_probe_evidence_sha256 is None
    assert args.behavior_g1_predecessor_report_sha256 is None
    assert args.source_mask_token_fraction == 0.0625
    assert args.lane_interleave_factor == 1
    assert args.support_weight == 0.0
    assert args.existence_weight == 1.0
    assert args.task_weight == 1.0
    assert args.dense_task_weight == 1.0
    assert args.ownership_weight == 1.0
    assert args.predictive_term_weight == 1.0
    assert args.current_grid_term_weight == 1.0
    assert args.omitted_static_term_weight == 1.0
    assert args.predictive_loss_power == 1.0
    assert args.minimum_supervised_fraction == 0.0
    assert FULL_WORLD_SIZE == 2


def test_full_runner_requires_explicit_entity_conditional_ownership_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--phase",
            "fresh",
            "--ownership-estimator",
            TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
        ],
    )

    assert _parse_args().ownership_estimator == TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP


def test_representation_runner_requires_exact_split_and_forbids_action_authorization(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "representation-split.json"
    _representation_split(split_path, training_steps=args.total_planned_steps)
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    _validate_paths_and_args(args)

    args.authorization_manifest = tmp_path / "authorization.json"
    args.authorization_manifest.write_text("{}")
    args.authorization_manifest_sha256 = _sha(args.authorization_manifest.read_bytes())
    with pytest.raises(ValueError, match="cannot consume action authorization"):
        _validate_paths_and_args(args)

    args.authorization_manifest = None
    args.authorization_manifest_sha256 = None
    args.representation_split_sha256 = "0" * 64
    with pytest.raises(ValueError, match="split SHA-256 differs"):
        _validate_paths_and_args(args)

    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.training_stage = "action"
    with pytest.raises(ValueError, match="cannot consume a representation source split"):
        _validate_paths_and_args(args)


def test_behavior_causal_g0_is_bounded_before_any_optimizer_update(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "representation-split.json"
    _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.lane_interleave_factor = 8
    args.evidence_profile = "loss_visual_trial"
    args.checkpoint_publication = "never"
    args.behavior_conditioned_prediction = True
    args.behavior_causal_probe_output = args.run_dir / "behavior_causal_probe.json"

    _validate_paths_and_args(args)

    args.invocation_steps = 2
    with pytest.raises(ValueError, match="one fresh representation-only"):
        _validate_paths_and_args(args)
    args.invocation_steps = 1
    args.behavior_causal_probe_output = tmp_path / "outside.json"
    with pytest.raises(ValueError, match="below the run directory"):
        _validate_paths_and_args(args)


def test_behavior_causal_g1_requires_the_exact_g0_receipt_and_two_step_audit(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "representation-split.json"
    _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    evidence_path = tmp_path / "behavior-g0.json"
    evidence_path.write_text(
        json.dumps(_behavior_g0_evidence(), sort_keys=True),
        encoding="ascii",
    )
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.lane_interleave_factor = 8
    args.evidence_profile = "loss_visual_trial"
    args.gradient_audit_steps = (1, 2)
    args.behavior_conditioned_prediction = True
    args.behavior_causal_probe_evidence = evidence_path
    args.behavior_causal_probe_evidence_sha256 = _sha(evidence_path.read_bytes())

    _validate_paths_and_args(args)

    args.gradient_audit_steps = (1,)
    with pytest.raises(ValueError, match="frozen two-step gradient audit"):
        _validate_paths_and_args(args)
    args.gradient_audit_steps = (1, 2)

    args.phase = "resume"
    args.load_global_step = 1
    _validate_paths_and_args(args)

    args.load_global_step = 2
    with pytest.raises(ValueError, match="exact cold-resume step"):
        _validate_paths_and_args(args)


def test_behavior_g2_is_only_the_loaded_step2_zero_update_factorial(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "representation-split.json"
    _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    evidence_path = tmp_path / "behavior-g0.json"
    evidence_path.write_text(
        json.dumps(_behavior_g0_evidence(), sort_keys=True),
        encoding="ascii",
    )
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.lane_interleave_factor = 8
    args.evidence_profile = "loss_visual_trial"
    args.gradient_audit_steps = (1, 2)
    args.behavior_conditioned_prediction = True
    args.behavior_causal_probe_evidence = evidence_path
    args.behavior_causal_probe_evidence_sha256 = _sha(evidence_path.read_bytes())
    args.phase = "resume"
    args.load_global_step = 2
    args.checkpoint_publication = "never"
    args.behavior_posterior_control_probe_output = args.run_dir / "g2-factorial.json"
    args.behavior_g1_predecessor_report_sha256 = "a" * 64

    _validate_paths_and_args(args)

    args.load_global_step = 1
    with pytest.raises(ValueError, match="exact loaded step-2 G1 boundary"):
        _validate_paths_and_args(args)
    args.load_global_step = 2
    args.checkpoint_publication = "always"
    with pytest.raises(ValueError, match="exact loaded step-2 G1 boundary"):
        _validate_paths_and_args(args)


def test_behavior_g2_receipt_is_recomputed_and_hash_bound(tmp_path: Path) -> None:
    evidence = _behavior_g2_evidence()
    assert validate_behavior_posterior_control_probe_evidence(evidence) == evidence

    path = tmp_path / "behavior-g2.json"
    path.write_text(json.dumps(evidence, sort_keys=True), encoding="ascii")
    expected_sha256 = _sha(path.read_bytes())
    loaded, actual = load_behavior_posterior_control_probe_evidence(
        path,
        expected_sha256=expected_sha256,
    )
    assert loaded == evidence
    assert actual == expected_sha256

    tampered = deepcopy(evidence)
    tampered["aggregate_control_margins_at_factual_posterior"] = {
        **tampered["aggregate_control_margins_at_factual_posterior"],
        "batch_shift": -0.1,
    }
    with pytest.raises(ValueError, match="differ from rank evidence"):
        validate_behavior_posterior_control_probe_evidence(tampered)

    tampered = deepcopy(evidence)
    tampered["g1_predecessor_implementation_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="differs from its G1 predecessor"):
        validate_behavior_posterior_control_probe_evidence(tampered)

    tampered = deepcopy(evidence)
    tampered["g1_predecessor_execution_contract_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="differs from its G1 predecessor"):
        validate_behavior_posterior_control_probe_evidence(tampered)

    tampered = deepcopy(evidence)
    tampered["rank_reports"][0]["factual_prediction_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="differs from its training prediction"):
        validate_behavior_posterior_control_probe_evidence(tampered)

    tampered = deepcopy(evidence)
    tampered["rank_reports"][0]["target"] = {}
    with pytest.raises(ValueError, match="target evidence"):
        validate_behavior_posterior_control_probe_evidence(tampered)

    tampered = deepcopy(evidence)
    tampered["rank_reports"][0]["assignment"]["row_binding_valid"] = [[True, True]]
    with pytest.raises(ValueError, match="causal source cut"):
        validate_behavior_posterior_control_probe_evidence(tampered)


def test_joint_behavior_action_requires_exact_g0_g2_receipts_and_visuals(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.total_planned_steps = full_runner.BEHAVIOR_ACTION_EVIDENCE_MAXIMUM_STEPS
    split_path = tmp_path / "representation-split.json"
    _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    g0_path = tmp_path / "behavior-g0.json"
    g0_path.write_text(json.dumps(_behavior_g0_evidence(), sort_keys=True), encoding="ascii")
    g2_path = tmp_path / "behavior-g2.json"
    g2_path.write_text(json.dumps(_behavior_g2_evidence(), sort_keys=True), encoding="ascii")

    args.training_stage = "action"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.behavior_conditioned_prediction = True
    args.behavior_causal_probe_evidence = g0_path
    args.behavior_causal_probe_evidence_sha256 = _sha(g0_path.read_bytes())
    args.behavior_posterior_control_probe_evidence = g2_path
    args.behavior_posterior_control_probe_evidence_sha256 = _sha(g2_path.read_bytes())
    args.evidence_profile = "loss_visual_trial"
    args.visual_audit_every = 1
    args.gradient_audit_steps = (2, 10, 20)
    args.lane_interleave_factor = 8

    _validate_paths_and_args(args)
    graph_sha256 = _behavior_graph_digest(args)
    assert graph_sha256 is not None
    conditioning = full_runner._behavior_conditioning_contract(
        args,
        behavior_graph_sha256=graph_sha256,
    )
    assert conditioning["protocol"] == "g2_approved_joint_action"
    assert conditioning["g2_evidence_sha256"] == _sha(g2_path.read_bytes())

    args.invocation_steps = full_runner.BEHAVIOR_ACTION_EVIDENCE_MAXIMUM_STEPS
    _validate_paths_and_args(args)

    args.invocation_steps = full_runner.BEHAVIOR_ACTION_EVIDENCE_MAXIMUM_STEPS + 1
    with pytest.raises(ValueError, match="bounded checkpoint publication"):
        _validate_paths_and_args(args)

    args.invocation_steps = 1
    args.visual_audit_every = 0
    with pytest.raises(ValueError, match="learned-anchor visuals"):
        _validate_paths_and_args(args)


def test_joint_behavior_action_discrimination_trial_is_exact_fresh_sixty_steps(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.total_planned_steps = 200
    split_path = tmp_path / "representation-split.json"
    _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    g0_path = tmp_path / "behavior-g0.json"
    g0_path.write_text(json.dumps(_behavior_g0_evidence(), sort_keys=True), encoding="ascii")
    g2_path = tmp_path / "behavior-g2.json"
    g2_path.write_text(json.dumps(_behavior_g2_evidence(), sort_keys=True), encoding="ascii")

    args.training_stage = "action"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.behavior_conditioned_prediction = True
    args.behavior_causal_probe_evidence = g0_path
    args.behavior_causal_probe_evidence_sha256 = _sha(g0_path.read_bytes())
    args.behavior_posterior_control_probe_evidence = g2_path
    args.behavior_posterior_control_probe_evidence_sha256 = _sha(g2_path.read_bytes())
    args.evidence_profile = "behavior_discrimination_trial"
    args.visual_audit_every = 1
    args.gradient_audit_steps = full_runner.BEHAVIOR_ACTION_DISCRIMINATION_AUDIT_STEPS
    args.lane_interleave_factor = 8
    args.invocation_steps = full_runner.BEHAVIOR_ACTION_DISCRIMINATION_STEPS

    _validate_paths_and_args(args)

    args.invocation_steps -= 1
    with pytest.raises(ValueError, match="bounded checkpoint publication"):
        _validate_paths_and_args(args)
    args.invocation_steps += 2
    with pytest.raises(ValueError, match="bounded checkpoint publication"):
        _validate_paths_and_args(args)
    args.invocation_steps = full_runner.BEHAVIOR_ACTION_DISCRIMINATION_STEPS
    args.phase = "resume"
    args.load_global_step = 20
    args.invocation_steps = 40
    args.authorization_manifest = tmp_path / "authorization.json"
    args.authorization_manifest.write_text("{}", encoding="ascii")
    args.authorization_manifest_sha256 = _sha(args.authorization_manifest.read_bytes())
    with pytest.raises(ValueError, match="bounded checkpoint publication"):
        _validate_paths_and_args(args)


def test_joint_behavior_action_report_accepts_the_behavior_future_graph(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "b" * 64
    path = full_objective_report_factory(tmp_path / "joint-action.json", digest=digest)
    value = json.loads(path.read_text())
    value["predictive_correction_loss_enabled"] = False
    value["behavior_future_loss_enabled"] = True
    value["behavior_conditioning"] = {
        "schema": full_runner.BEHAVIOR_JOINT_CONDITIONING_SCHEMA,
        "protocol": "g2_approved_joint_action",
        "horizon": 1,
        "isolation": "separate_same_weight_auxiliary_forward",
        "behavior_graph_sha256": digest,
        "g0_evidence_sha256": digest,
        "g2_evidence_sha256": digest,
    }
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["normalized_terms"]["rollout/vision/binding"] = 0.1
        step["valid_counts"]["rollout/vision/binding"] = 1
        step["normalized_terms"].pop("correction/dino_video")
        step["valid_counts"].pop("correction/dino_video")
        step["objective_total"] = 0.21072
        step["gradient_metrics"]["behavior_posterior_gradient_norm"] = 1.0
        step["gradient_metrics"]["behavior_posterior_gradient_elements"] = 16

    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    (Path(value["checkpoint_dir"]) / "native_full_report.json").write_text(payload)

    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=False,
        )["status"]
        == "PASS"
    )


def test_behavior_causal_g0_receipt_is_strict_and_hash_bound(tmp_path: Path) -> None:
    evidence = _behavior_g0_evidence()
    assert validate_behavior_causal_probe_evidence(evidence) == evidence

    path = tmp_path / "behavior-g0.json"
    path.write_text(json.dumps(evidence, sort_keys=True), encoding="ascii")
    expected_sha256 = _sha(path.read_bytes())
    assert load_behavior_causal_probe_evidence(path, expected_sha256=expected_sha256) == evidence

    changed = deepcopy(evidence)
    changed["rank_reports"][0]["fresh_primary_rerun_bit_identical"] = False
    with pytest.raises(ValueError, match="deploy isolation"):
        validate_behavior_causal_probe_evidence(changed)

    changed = deepcopy(evidence)
    changed["sample_keys_by_rank"] = [["same"], ["same"]]
    with pytest.raises(ValueError, match="overlapping samples"):
        validate_behavior_causal_probe_evidence(changed)

    changed = deepcopy(evidence)
    changed["rank_reports"][0]["horizon"] = 2
    with pytest.raises(ValueError, match="horizon differs"):
        validate_behavior_causal_probe_evidence(changed)

    with pytest.raises(ValueError, match="expected digest"):
        load_behavior_causal_probe_evidence(path, expected_sha256="b" * 64)


def test_behavior_causal_g1_rejects_any_changed_scientific_context() -> None:
    evidence = _behavior_g0_evidence()
    digest = "a" * 64
    kwargs = {
        "patch_sha256": digest,
        "implementation_sha256": digest,
        "model_family_sha256": digest,
        "plan_sha256": digest,
        "behavior_graph_sha256": digest,
    }
    require_behavior_causal_probe_context(evidence, **kwargs)
    for field in tuple(kwargs):
        changed = dict(kwargs)
        changed[field] = "b" * 64
        with pytest.raises(RuntimeError, match="scientific context"):
            require_behavior_causal_probe_context(evidence, **changed)


def test_behavior_causal_g0_hashes_and_decodes_one_open_file_description(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _behavior_g0_evidence()
    path = tmp_path / "behavior-g0.json"
    payload = json.dumps(evidence, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    replacement = tmp_path / "replacement.json"
    replacement.write_text("{}", encoding="ascii")
    real_open = full_runner.os.open

    def open_then_replace(open_path, flags):
        descriptor = real_open(open_path, flags)
        replacement.replace(path)
        return descriptor

    monkeypatch.setattr(full_runner.os, "open", open_then_replace)
    assert (
        load_behavior_causal_probe_evidence(
            path,
            expected_sha256=_sha(payload),
        )
        == evidence
    )


def test_behavior_causal_g0_normalizes_distributed_tuple_provenance() -> None:
    assert _behavior_probe_sample_keys_by_rank(
        (
            {"sample_keys": ("rank-0-a", "rank-0-b")},
            {"sample_keys": ("rank-1-a",)},
        )
    ) == [["rank-0-a", "rank-0-b"], ["rank-1-a"]]


def test_behavior_graph_binds_scientific_graph_but_not_g0_g1_protocol(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "representation-split.json"
    _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.behavior_conditioned_prediction = True

    baseline = _behavior_graph_digest(args)
    assert baseline is not None

    args.data_config.write_text('{"changed": true}', encoding="ascii")
    assert _behavior_graph_digest(args) != baseline
    args.data_config.write_text("{}", encoding="ascii")
    assert _behavior_graph_digest(args) == baseline

    args.predictive_weight = 2.0
    assert _behavior_graph_digest(args) != baseline
    args.predictive_weight = 1.0
    args.capacity += 1
    assert _behavior_graph_digest(args) != baseline
    args.capacity -= 1

    args.behavior_causal_probe_output = args.run_dir / "g0-output.json"
    g0_conditioning = _behavior_conditioning_digest(
        args,
        behavior_graph_sha256=baseline,
    )
    assert _behavior_graph_digest(args) == baseline

    args.behavior_causal_probe_output = None
    args.behavior_causal_probe_evidence = tmp_path / "g0-evidence.json"
    args.behavior_causal_probe_evidence_sha256 = "f" * 64
    g1_fresh_conditioning = _behavior_conditioning_digest(
        args,
        behavior_graph_sha256=baseline,
    )
    assert _behavior_graph_digest(args) == baseline
    assert g1_fresh_conditioning != g0_conditioning

    args.phase = "resume"
    args.load_global_step = 1
    assert _behavior_graph_digest(args) == baseline
    assert (
        _behavior_conditioning_digest(
            args,
            behavior_graph_sha256=baseline,
        )
        == g1_fresh_conditioning
    )

    args.data_config.write_text('{"drifted": true}', encoding="ascii")
    with pytest.raises(RuntimeError, match="changed during"):
        _require_unchanged_behavior_graph(args, expected_sha256=baseline)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("structural_weight", 2.0),
        ("support_weight", 0.1),
        ("existence_weight", 2.0),
        ("task_weight", 2.0),
        ("dense_task_weight", 2.0),
        ("ownership_weight", 2.0),
        ("predictive_term_weight", 2.0),
        ("current_grid_term_weight", 2.0),
        ("omitted_static_term_weight", 2.0),
        ("predictive_loss_power", 2.0),
        ("minimum_supervised_fraction", 0.5),
        ("task_relation_estimator", HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION),
        ("seed", 8),
        ("total_planned_steps", 11),
        ("lane_interleave_factor", 2),
        ("local_bptt_probability", 0.2),
        ("overshoot_probability", 0.2),
        ("source_mask_probability", 0.2),
        ("source_prediction_mode", "current_grid"),
        ("source_mask_token_fraction", 0.1),
        ("maximum_control_tokens", 9),
        ("maximum_optimizer_lag", 9),
        ("relation_supervision_layers", (8,)),
        ("fsdp2_placement", "gpu-sharded"),
        ("cuda_allocator", "expandable-segments"),
    ],
)
def test_behavior_graph_binds_each_objective_sampling_and_topology_surface(
    tmp_path: Path,
    field: str,
    changed: object,
) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "representation-split.json"
    _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    args.training_stage = "representation"
    args.representation_split = split_path
    args.behavior_conditioned_prediction = True
    baseline = _behavior_graph_digest(args)
    setattr(args, field, changed)
    assert _behavior_graph_digest(args) != baseline


@pytest.mark.parametrize(
    "field",
    [
        "physical_sidecar_manifest_sha256",
        "physical_visual_acceptance_sha256",
        "predictive_cache_build_report_sha256",
        "predictive_teacher_causality_audit_sha256",
        "predictive_target_audit_sha256",
        "predictive_temporal_audit_sha256",
        "current_grid_cache_build_report_sha256",
    ],
)
def test_behavior_graph_binds_each_declared_target_artifact(
    tmp_path: Path,
    field: str,
) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "representation-split.json"
    _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    args.training_stage = "representation"
    args.representation_split = split_path
    args.behavior_conditioned_prediction = True
    baseline = _behavior_graph_digest(args)
    setattr(args, field, "2" * 64)
    assert _behavior_graph_digest(args) != baseline


def test_runner_rejects_legacy_global_task_retrieval_before_model_loading(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.task_relation_estimator = GLOBAL_MULTIPOSITIVE_TASK_RELATION

    with pytest.raises(ValueError, match="rejected legacy interface"):
        _validate_paths_and_args(args)


def test_runner_accepts_preregistered_host_native_row_competition(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.task_relation_estimator = HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION

    _validate_paths_and_args(args)


def test_runner_accepts_factorized_relation_only_without_independent_dense_loss(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.task_relation_estimator = HOST_NATIVE_FACTORIZED_TASK_RELATION
    args.dense_task_weight = 0.0

    _validate_paths_and_args(args)

    args.dense_task_weight = 1.0
    with pytest.raises(ValueError, match="requires dense_task_weight=0"):
        _validate_paths_and_args(args)


def test_runner_rejects_zero_dense_weight_for_non_factorized_relation(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.task_relation_estimator = HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION
    args.dense_task_weight = 0.0

    with pytest.raises(ValueError, match="requires dense_task_weight>0"):
        _validate_paths_and_args(args)


def test_runner_binds_one_immutable_complete_prompt_contract(tmp_path: Path) -> None:
    task = "move the blue block"
    audit = CompletePromptTokenizationAudit(
        prompt_count=2,
        maximum_tokens=72,
        use_qwen3_chat_template=True,
        prompts=(
            PromptTokenizationEntry(
                task=task,
                task_sha256=hashlib.sha256(task.encode("utf-8")).hexdigest(),
                formatted_prompt_sha256="a" * 64,
                token_count=11,
                token_ids_sha256="b" * 64,
            ),
        ),
    )
    run_dir = tmp_path / "prompt-run"
    run_dir.mkdir()

    path = full_runner._bind_complete_prompt_tokenization_audit(run_dir, audit)
    assert CompletePromptTokenizationAudit.load(path) == audit
    assert full_runner._bind_complete_prompt_tokenization_audit(run_dir, audit) == path

    changed = CompletePromptTokenizationAudit(
        prompt_count=2,
        maximum_tokens=73,
        use_qwen3_chat_template=True,
        prompts=audit.prompts,
    )
    with pytest.raises(RuntimeError, match="contract changed"):
        full_runner._bind_complete_prompt_tokenization_audit(run_dir, changed)


def test_representation_checkpoint_suppression_requires_fresh_final_evidence(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.checkpoint_publication = "never"
    with pytest.raises(ValueError, match="restricted to fresh representation evidence"):
        _validate_paths_and_args(args)

    args.total_planned_steps = 30_000
    split_path = tmp_path / "representation-split.json"
    _representation_split(split_path, training_steps=args.total_planned_steps)
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.invocation_steps = 20
    args.visual_audit_every = 4
    args.evidence_profile = "loss_visual_trial"
    args.gradient_audit_steps = (60, 120)
    with pytest.raises(ValueError, match="requires immutable final representation evaluation"):
        _validate_paths_and_args(args)

    evaluation_plan = tmp_path / "representation-evaluation-plan.json"
    evaluation_plan.write_text("{}", encoding="ascii")
    args.representation_evaluation_plan = evaluation_plan
    args.representation_evaluation_plan_sha256 = _sha(evaluation_plan.read_bytes())
    args.representation_evaluation_steps = (20,)
    _validate_paths_and_args(args)

    args.phase = "resume"
    args.load_global_step = 1
    args.invocation_steps = 19
    with pytest.raises(ValueError, match="restricted to fresh representation evidence"):
        _validate_paths_and_args(args)


def test_representation_task_intervention_requires_complete_stage_bound_pair(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    intervention = tmp_path / "task-intervention.json"
    intervention.write_text("{}", encoding="ascii")
    intervention_sha256 = _sha(intervention.read_bytes())

    args.representation_task_intervention_plan = intervention
    with pytest.raises(ValueError, match="requires its plan, digest and stage"):
        _validate_paths_and_args(args)

    args.representation_task_intervention_plan_sha256 = intervention_sha256
    with pytest.raises(ValueError, match="requires its plan, digest and stage"):
        _validate_paths_and_args(args)

    split_path = tmp_path / "representation-split.json"
    _representation_split(split_path, training_steps=args.total_planned_steps)
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    _validate_paths_and_args(args)

    args.representation_task_intervention_plan_sha256 = "0" * 64
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        _validate_paths_and_args(args)


def test_full_runner_interleaving_is_bounded_by_optimizer_lag(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.total_planned_steps = 20
    args.lane_interleave_factor = 8
    args.gradient_audit_steps = (9, 17)
    _validate_paths_and_args(args)

    args.lane_interleave_factor = 9
    with pytest.raises(ValueError, match="exceeds the frozen maximum optimizer lag"):
        _validate_paths_and_args(args)

    args.lane_interleave_factor = 0
    with pytest.raises(ValueError, match="lane_interleave_factor must be positive"):
        _validate_paths_and_args(args)


def test_adr121_runner_requires_the_exact_reset_warm_and_temporal_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _runner_args(tmp_path)
    args.training_stage = "representation"
    args.total_planned_steps = 200
    args.invocation_steps = 200
    args.checkpoint_publication = "never"
    args.lane_interleave_factor = 8
    args.maximum_optimizer_lag = 16
    args.gradient_audit_steps = (18, 34, 50, 100, 200)
    args.visual_audit_every = 1
    args.representation_evaluation_steps = (0, 200)
    args.reset_mixture_numerator = 1
    args.reset_mixture_denominator = 2

    split_path = tmp_path / "adr121-representation-split.json"
    split = _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())

    evaluation_plan_path = tmp_path / "adr121-reset-evaluation-plan.json"
    evaluation_plan_path.write_text("{}")
    args.representation_evaluation_plan = evaluation_plan_path
    args.representation_evaluation_plan_sha256 = _sha(evaluation_plan_path.read_bytes())
    baseline_path = tmp_path / "adr121-step-zero-baseline.json"
    baseline_path.write_text("{}")
    args.representation_evaluation_baseline = baseline_path
    args.representation_evaluation_baseline_sha256 = _sha(baseline_path.read_bytes())
    warm_plan_path = tmp_path / "adr121-warm-evaluation-plan.json"
    warm_plan_path.write_text("{}")
    args.representation_warm_evaluation_plan = warm_plan_path
    args.representation_warm_evaluation_plan_sha256 = _sha(warm_plan_path.read_bytes())

    reset_plan = SimpleNamespace()
    warm_plan = SimpleNamespace(
        schema=full_runner.REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA,
        history_transitions=8,
        representation_split_sha256=split.artifact_sha256,
    )
    monkeypatch.setattr(
        full_runner.RepresentationEvaluationPlan,
        "load",
        classmethod(lambda _cls, path: warm_plan if Path(path) == warm_plan_path else reset_plan),
    )
    monkeypatch.setattr(
        full_runner,
        "load_representation_evaluation_baseline",
        lambda _path: {"status": "PASS"},
    )
    monkeypatch.setattr(
        full_runner,
        "validate_representation_baseline_plan",
        lambda _baseline, *, candidate_plan: None,
    )
    _validate_paths_and_args(args)

    fixed_pair_plan = tmp_path / "fixed-observation-pair-plan.json"
    fixed_pair_plan.write_text("{}")
    fixed_training_audit = tmp_path / "fixed-observation-training-audit.json"
    fixed_training_audit.write_text("{}")
    fixed_evaluation_plan = tmp_path / "fixed-observation-evaluation-plan.json"
    fixed_evaluation_plan.write_text("{}")
    fixed_validation_audit = tmp_path / "fixed-observation-validation-audit.json"
    fixed_validation_audit.write_text("{}")
    fixed_heldout_audit = tmp_path / "fixed-observation-heldout-audit.json"
    fixed_heldout_audit.write_text("{}")
    args.fixed_observation_pair_plan = fixed_pair_plan
    args.fixed_observation_pair_plan_sha256 = _sha(fixed_pair_plan.read_bytes())
    args.fixed_observation_training_audit = fixed_training_audit
    args.fixed_observation_training_audit_sha256 = _sha(fixed_training_audit.read_bytes())
    args.fixed_observation_evaluation_plan = fixed_evaluation_plan
    args.fixed_observation_evaluation_plan_sha256 = _sha(fixed_evaluation_plan.read_bytes())
    args.fixed_observation_validation_audit = fixed_validation_audit
    args.fixed_observation_validation_audit_sha256 = _sha(fixed_validation_audit.read_bytes())
    args.fixed_observation_heldout_audit = fixed_heldout_audit
    args.fixed_observation_heldout_audit_sha256 = _sha(fixed_heldout_audit.read_bytes())
    _validate_paths_and_args(args)

    intervention = tmp_path / "legacy-task-intervention.json"
    intervention.write_text("{}")
    args.representation_task_intervention_plan = intervention
    args.representation_task_intervention_plan_sha256 = _sha(intervention.read_bytes())
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_paths_and_args(args)
    args.representation_task_intervention_plan = None
    args.representation_task_intervention_plan_sha256 = None

    args.maximum_optimizer_lag = 8
    with pytest.raises(ValueError, match="twice the lane interleave factor"):
        _validate_paths_and_args(args)
    args.maximum_optimizer_lag = 16

    args.gradient_audit_steps = (18, 34, 50, 100, 199)
    with pytest.raises(ValueError, match="exact 18,34,50,100,200"):
        _validate_paths_and_args(args)
    args.gradient_audit_steps = (18, 34, 50, 100, 200)

    args.representation_warm_evaluation_plan = None
    args.representation_warm_evaluation_plan_sha256 = None
    with pytest.raises(ValueError, match="must be provided together"):
        _validate_paths_and_args(args)


def _matched_medium_horizon_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str,
    load_step: int,
    invocation_steps: int,
    evaluation_steps: tuple[int, ...],
) -> argparse.Namespace:
    args = _runner_args(tmp_path)
    args.training_stage = "representation"
    args.phase = phase
    args.load_global_step = load_step
    args.invocation_steps = invocation_steps
    args.total_planned_steps = full_runner.MATCHED_MEDIUM_HORIZON_TOTAL_STEPS
    args.checkpoint_publication = "always"
    args.lane_interleave_factor = 8
    args.maximum_optimizer_lag = 16
    args.reset_mixture_numerator = 1
    args.reset_mixture_denominator = 2
    args.evidence_profile = full_runner.MATCHED_MEDIUM_HORIZON_PROFILE
    args.gradient_audit_steps = full_runner.MATCHED_MEDIUM_HORIZON_AUDIT_STEPS
    args.visual_audit_every = full_runner.MATCHED_MEDIUM_HORIZON_VISUAL_CADENCE
    args.task_relation_estimator = HOST_NATIVE_FACTORIZED_TASK_RELATION
    args.ownership_estimator = TOKEN_MICRO_OWNERSHIP
    args.dense_task_weight = 0.0
    args.representation_evaluation_steps = evaluation_steps

    split_path = tmp_path / "adr135-representation-split.json"
    split = _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())

    def bind_artifact(name: str) -> tuple[Path, str]:
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="ascii")
        return path, _sha(path.read_bytes())

    evaluation_plan, evaluation_plan_sha256 = bind_artifact("adr135-evaluation-plan")
    args.representation_evaluation_plan = evaluation_plan
    args.representation_evaluation_plan_sha256 = evaluation_plan_sha256
    baseline, baseline_sha256 = bind_artifact("adr135-step-zero-baseline")
    args.representation_evaluation_baseline = baseline
    args.representation_evaluation_baseline_sha256 = baseline_sha256
    warm_plan, warm_plan_sha256 = bind_artifact("adr135-warm-plan")
    args.representation_warm_evaluation_plan = warm_plan
    args.representation_warm_evaluation_plan_sha256 = warm_plan_sha256

    fixed_paths: dict[str, Path] = {}
    for name in (
        "fixed-observation-pair-plan",
        "fixed-observation-training-audit",
        "fixed-observation-evaluation-plan",
        "fixed-observation-validation-audit",
        "fixed-observation-heldout-audit",
    ):
        fixed_paths[name], _ = bind_artifact(name)
    args.fixed_observation_pair_plan = fixed_paths["fixed-observation-pair-plan"]
    args.fixed_observation_pair_plan_sha256 = _sha(args.fixed_observation_pair_plan.read_bytes())
    args.fixed_observation_training_audit = fixed_paths["fixed-observation-training-audit"]
    args.fixed_observation_training_audit_sha256 = _sha(
        args.fixed_observation_training_audit.read_bytes()
    )
    args.fixed_observation_evaluation_plan = fixed_paths["fixed-observation-evaluation-plan"]
    args.fixed_observation_evaluation_plan_sha256 = _sha(
        args.fixed_observation_evaluation_plan.read_bytes()
    )
    args.fixed_observation_validation_audit = fixed_paths["fixed-observation-validation-audit"]
    args.fixed_observation_validation_audit_sha256 = _sha(
        args.fixed_observation_validation_audit.read_bytes()
    )
    args.fixed_observation_heldout_audit = fixed_paths["fixed-observation-heldout-audit"]
    args.fixed_observation_heldout_audit_sha256 = _sha(
        args.fixed_observation_heldout_audit.read_bytes()
    )

    candidate_plan = SimpleNamespace()
    warm = SimpleNamespace(
        schema=full_runner.REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA,
        history_transitions=8,
        representation_split_sha256=split.artifact_sha256,
    )
    monkeypatch.setattr(
        full_runner.RepresentationEvaluationPlan,
        "load",
        classmethod(lambda _cls, path: warm if Path(path) == warm_plan else candidate_plan),
    )
    monkeypatch.setattr(
        full_runner,
        "load_representation_evaluation_baseline",
        lambda _path: {"status": "PASS"},
    )
    monkeypatch.setattr(
        full_runner,
        "validate_representation_baseline_plan",
        lambda _baseline, *, candidate_plan: None,
    )
    if phase == "resume":
        step_zero = args.run_dir / "representation_evaluations" / "global_step_0"
        step_zero.mkdir(parents=True)
        (step_zero / "representation_evaluation_snapshot.json").write_text(
            "{}",
            encoding="ascii",
        )
        (step_zero / "representation_baseline_replay_report.json").write_text(
            "{}",
            encoding="ascii",
        )
        monkeypatch.setattr(
            full_runner,
            "build_representation_baseline_replay_report",
            lambda **_kwargs: {},
        )
        monkeypatch.setattr(
            full_runner,
            "load_representation_baseline_replay_report",
            lambda _path: {},
        )
    return args


@pytest.mark.parametrize(
    ("phase", "load_step", "invocation_steps", "evaluation_steps"),
    full_runner.MATCHED_MEDIUM_HORIZON_SEGMENTS,
)
def test_adr135_runner_accepts_only_registered_medium_horizon_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    load_step: int,
    invocation_steps: int,
    evaluation_steps: tuple[int, ...],
) -> None:
    args = _matched_medium_horizon_args(
        tmp_path,
        monkeypatch,
        phase=phase,
        load_step=load_step,
        invocation_steps=invocation_steps,
        evaluation_steps=evaluation_steps,
    )

    _validate_paths_and_args(args)


@pytest.mark.parametrize(
    ("phase", "load_step", "invocation_steps", "evaluation_steps"),
    full_runner.CONTENT_ADDRESSED_SET_MEDIUM_HORIZON_SEGMENTS,
)
def test_adr136_runner_accepts_only_registered_content_addressed_set_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    load_step: int,
    invocation_steps: int,
    evaluation_steps: tuple[int, ...],
) -> None:
    args = _matched_medium_horizon_args(
        tmp_path,
        monkeypatch,
        phase=phase,
        load_step=load_step,
        invocation_steps=invocation_steps,
        evaluation_steps=evaluation_steps,
    )

    _validate_paths_and_args(args)


def test_adr135_runner_rejects_unregistered_medium_horizon_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _matched_medium_horizon_args(
        tmp_path,
        monkeypatch,
        phase="resume",
        load_step=200,
        invocation_steps=300,
        evaluation_steps=(500,),
    )

    args.visual_audit_every = 100
    with pytest.raises(ValueError, match="explicitly registered"):
        _validate_paths_and_args(args)
    args.visual_audit_every = full_runner.MATCHED_MEDIUM_HORIZON_VISUAL_CADENCE

    args.gradient_audit_steps = (18, 34, 50, 100, 200, 500)
    with pytest.raises(ValueError, match="exact 18,34,50,100,200,500,1000"):
        _validate_paths_and_args(args)
    args.gradient_audit_steps = full_runner.MATCHED_MEDIUM_HORIZON_AUDIT_STEPS

    args.invocation_steps = 299
    with pytest.raises(ValueError, match="explicitly registered"):
        _validate_paths_and_args(args)


def test_interleaved_representation_requires_fixed_reference_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _runner_args(tmp_path)
    args.total_planned_steps = 20
    legacy_split_path = tmp_path / "legacy-representation-split.json"
    _representation_split(legacy_split_path, training_steps=args.total_planned_steps)
    args.training_stage = "representation"
    args.representation_split = legacy_split_path
    args.representation_split_sha256 = _sha(legacy_split_path.read_bytes())
    args.lane_interleave_factor = 8
    args.gradient_audit_steps = (9, 17)

    with pytest.raises(ValueError, match="fixed reference evaluation bank"):
        _validate_paths_and_args(args)

    reference_split_path = tmp_path / "reference-representation-split.json"
    _representation_split(
        reference_split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    args.representation_split = reference_split_path
    args.representation_split_sha256 = _sha(reference_split_path.read_bytes())
    with pytest.raises(ValueError, match="checkpoint-boundary evaluation"):
        _validate_paths_and_args(args)

    evaluation_plan_path = tmp_path / "representation-evaluation-plan.json"
    evaluation_plan_path.write_text("{}")
    args.representation_evaluation_plan = evaluation_plan_path
    args.representation_evaluation_plan_sha256 = _sha(evaluation_plan_path.read_bytes())
    args.representation_evaluation_steps = (0, 1)
    with pytest.raises(ValueError, match="exact K1 step-zero replay"):
        _validate_paths_and_args(args)

    baseline_path = tmp_path / "representation-evaluation-baseline.json"
    baseline_path.write_text("{}")
    args.representation_evaluation_baseline = baseline_path
    args.representation_evaluation_baseline_sha256 = _sha(baseline_path.read_bytes())
    candidate_plan = SimpleNamespace()
    monkeypatch.setattr(
        full_runner.RepresentationEvaluationPlan,
        "load",
        classmethod(lambda _cls, _path: candidate_plan),
    )
    monkeypatch.setattr(
        full_runner,
        "load_representation_evaluation_baseline",
        lambda _path: {"status": "PASS"},
    )
    monkeypatch.setattr(
        full_runner,
        "validate_representation_baseline_plan",
        lambda _baseline, *, candidate_plan: None,
    )
    _validate_paths_and_args(args)


def test_representation_resume_needs_no_action_authorization(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "representation-split.json"
    _representation_split(split_path, training_steps=args.total_planned_steps)
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.phase = "resume"
    args.load_global_step = 1

    _validate_paths_and_args(args)


def test_full_runner_training_config_follows_an_overridden_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--phase",
            "fresh",
            "--source-checkout",
            str(source),
        ],
    )

    args = _parse_args()

    assert args.source_checkout == source
    assert args.training_config == source / "configs/vla/robotwin/robotwin.yaml"


def test_gradient_audit_step_parser_accepts_only_sorted_unique_positive_steps() -> None:
    assert _parse_positive_step_set("2,3,20,120") == (2, 3, 20, 120)
    for invalid in ("", "0", "-1", "3,2", "2,2", "2,invalid"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_positive_step_set(invalid)


def test_representation_evaluation_step_parser_accepts_sorted_unique_boundaries() -> None:
    assert _parse_nonnegative_step_set("0,20,50,100,200") == (0, 20, 50, 100, 200)
    for invalid in ("", "-1", "20,0", "20,20", "20,invalid"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_nonnegative_step_set(invalid)


def test_representation_evaluation_requires_complete_boundary_bound_plan(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "representation-split.json"
    plan_path = tmp_path / "representation-evaluation-plan.json"
    _representation_split(split_path, training_steps=args.total_planned_steps)
    plan_path.write_text("{}")
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.representation_evaluation_plan = plan_path
    args.representation_evaluation_plan_sha256 = _sha(plan_path.read_bytes())
    args.representation_evaluation_steps = (0, 1)
    _validate_paths_and_args(args)

    args.representation_evaluation_steps = (0, 2)
    with pytest.raises(ValueError, match="checkpoint boundaries"):
        _validate_paths_and_args(args)

    args.representation_evaluation_steps = ()
    with pytest.raises(ValueError, match="requires its plan"):
        _validate_paths_and_args(args)


def test_factorized_relation_step_zero_baseline_source_uses_one_checkpointed_lane(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    split_path = tmp_path / "joint-baseline-source-split.json"
    plan_path = tmp_path / "joint-baseline-source-evaluation-plan.json"
    _representation_split(
        split_path,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    plan_path.write_text("{}", encoding="ascii")
    args.training_stage = "representation"
    args.representation_split = split_path
    args.representation_split_sha256 = _sha(split_path.read_bytes())
    args.representation_evaluation_plan = plan_path
    args.representation_evaluation_plan_sha256 = _sha(plan_path.read_bytes())
    args.representation_evaluation_steps = (0,)
    args.task_relation_estimator = HOST_NATIVE_FACTORIZED_TASK_RELATION
    args.dense_task_weight = 0.0
    args.evidence_profile = "acceptance"
    _validate_paths_and_args(args)
    assert args.checkpoint_publication == "always"
    assert args.lane_interleave_factor == 1
    assert args.representation_evaluation_baseline is None
    assert args.representation_warm_evaluation_plan is None
    assert getattr(args, "reset_mixture_numerator", None) is None
    assert getattr(args, "reset_mixture_denominator", None) is None

    args.phase = "resume"
    args.load_global_step = 1
    args.representation_evaluation_steps = (2,)
    _validate_paths_and_args(args)


def test_factorized_relation_baseline_uses_the_frozen_k1_reference_domain() -> None:
    baseline = (ROOT / "configs/cloud/adr132_joint_relation_j2_baseline.sh").read_text()
    candidate = (ROOT / "configs/cloud/adr132_joint_relation_j2.sh").read_text()

    assert "--lane-interleave-factor 1" in baseline
    assert "--maximum-optimizer-lag 8" in baseline
    assert "lingbot-representation-k1-200-reference-split.json" in baseline
    assert "--representation-split-sha256 392fd6b9" in baseline
    assert "representation-evaluation-plan-3b6d367-v3.json" in baseline
    assert "--representation-evaluation-plan-sha256 9518c1e6" in baseline
    assert "adr132-k1-current-estimator-predictive-20260804T090614+0800" in baseline
    assert "adr132-k1-current-estimator-current-grid-20260804T090614+0800" in baseline
    assert "lingbot-predictive-representation200-b67289e" not in baseline
    assert "lingbot-current-grid-representation200-b67289e" not in baseline
    assert "--lane-interleave-factor 8" in candidate
    assert "representation-k8-reset-mixture-adr121-e6dbcf6-20260731.split.json" in candidate
    assert "adr132-k8-current-estimator-predictive-20260804T094015+0800" in candidate
    assert "adr132-k8-current-estimator-current-grid-20260804T094015+0800" in candidate
    assert "cffffa03453456e9a6df6ce500d131856791c0dd8969b33a40e28e8649c247ba" in candidate
    assert "b635eb35d1c1313bc22b1d40127e7379ed430a03e8e6188ba6bb589b25bae77e" in candidate
    assert "9dbd7b2ffa14cbb8ff19494db94fa35ea562f1c766b861c0d28cd42d737acf10" in candidate
    assert "db291d41a15e86f155aa1bcb335b37275f0d26c0c1f51fc5559c4120af1191d7" in candidate
    assert "3b7e9a5aed48d7930983d545211aa8141f534fc1f207d5a28d6670647b8bb987" in candidate
    assert "representation-k8-reset-mixture-adr121-e6dbcf6-predictive" not in candidate
    assert "representation-k8-reset-mixture-adr121-e6dbcf6-current-grid" not in candidate
    assert "--representation-evaluation-steps 0,200" in candidate
    assert "--evidence-profile acceptance" in candidate
    assert "--local-bptt-probability 0.10" in candidate
    assert "--overshoot-probability 0.05" in candidate
    assert "--source-mask-probability 0.10" in candidate
    assert "--representation-evaluation-steps 0,20,50,100,200" not in candidate
    assert "--evidence-profile loss_visual_trial" not in candidate


def test_adr134_scripts_change_only_the_registered_ownership_estimator() -> None:
    baseline = (ROOT / "configs/cloud/adr134_entity_conditional_j2_baseline.sh").read_text()
    candidate = (ROOT / "configs/cloud/adr134_entity_conditional_j2.sh").read_text()
    chain = (ROOT / "configs/cloud/adr134_entity_conditional_j2_chain.sh").read_text()
    estimator = "--ownership-estimator token_micro_entity_conditional_equal"

    assert baseline.count(estimator) == 1
    assert candidate.count(estimator) == 1
    for script in (baseline, candidate):
        assert "--task-relation-estimator host_native_factorized_task_physical_ownership" in script
        assert "--dense-task-weight 0" in script
        assert "--support-weight 0" in script
        assert "CUDA_VISIBLE_DEVICES=0,1" in script
        assert "--nproc_per_node=2" in script
    assert "lingbot-representation-k1-200-reference-split.json" in baseline
    assert "--lane-interleave-factor 1" in baseline
    assert "fresh:0" in baseline and "resume:1" in baseline
    assert '--representation-evaluation-steps "$EVALUATION_STEPS"' in baseline
    assert '--load-global-step "$LOAD_GLOBAL_STEP"' in baseline
    assert "--evidence-profile acceptance" in baseline
    assert "representation-k8-reset-mixture-adr121-e6dbcf6-20260731.split.json" in candidate
    assert "--lane-interleave-factor 8" in candidate
    assert "--representation-evaluation-steps 0,200" in candidate
    assert "--checkpoint-publication never" in candidate
    assert "adr134_entity_conditional_j2_baseline.sh" in chain
    assert "adr134_entity_conditional_j2.sh" in chain
    assert "adr134-entity-step0-baseline-v1" in chain
    assert "adr134-entity-j2-v1" in chain
    assert "PHASE=resume LOAD_GLOBAL_STEP=1" in chain
    assert "phase=baseline_resume_running" in chain


def test_adr132_k8_bundle_publication_is_atomic_and_identity_bound() -> None:
    script = (ROOT / "configs/cloud/adr132_rebuild_j2_k8_temporal_bundle.sh").read_text()

    for field in (
        "BUNDLE_SCHEMA=picf-next.adr132-k8-temporal-bundle.v1",
        "STREAM_SHA256=$STREAM_SHA256",
        "TEMPORAL_SHA256=$TEMPORAL_SHA256",
        "SPLIT_SHA256=$SPLIT_SHA256",
        "VISUAL_MANIFEST_SHA256=$VISUAL_MANIFEST_SHA256",
    ):
        assert field in script
    assert 'sync -f "$BUNDLE_TMP"' in script
    assert 'mv "$BUNDLE_TMP" "$BUNDLE"' in script
    assert 'sync -f "$ROOT"' in script
    assert "bundle_sha256=%s" in script
    assert "trap 'on_signal INT 130' INT" in script
    assert "trap 'on_signal TERM 143' TERM" in script


def test_relation_supervision_parser_accepts_only_sorted_unique_nonnegative_layers() -> None:
    assert _parse_nonnegative_layer_set("0,8,17,26") == (0, 8, 17, 26)
    for invalid in ("", "-1", "8,0", "8,8", "8,invalid"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_nonnegative_layer_set(invalid)


def test_relation_supervision_runner_accepts_only_preregistered_training_depths(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.relation_supervision_layers = (8, 17, 26)
    _validate_paths_and_args(args)
    args.relation_supervision_layers = (8, 16, 26)
    with pytest.raises(ValueError, match="preregistered"):
        _validate_paths_and_args(args)


def test_full_runner_consumes_every_declared_cli_argument() -> None:
    tree = ast.parse(_source())
    declared: set[str] = set()
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr != "add_argument"
            or not node.args
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)
            or not node.args[0].value.startswith("--")
        ):
            continue
        destination = node.args[0].value[2:].replace("-", "_")
        for keyword in node.keywords:
            if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                destination = keyword.value.value
        assert isinstance(destination, str)
        declared.add(destination)

    consumed = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Load)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }
    assert len(declared) == 108
    assert consumed == declared


def test_full_runner_validators_are_read_only_and_broad_exceptions_fail_closed() -> None:
    tree = ast.parse(_source())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    validator_names = {
        "_validate_full_visual_artifact",
        "validate_full_objective_report",
        "_validate_paths_and_args",
        "load_predictive_target_audit",
        "load_predictive_temporal_audit",
    }
    forbidden_calls = {
        "commit",
        "mkdir",
        "remove",
        "rename",
        "replace",
        "rmtree",
        "save",
        "unlink",
        "write_bytes",
        "write_text",
    }
    for name in validator_names:
        calls = {
            node.func.attr
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert calls.isdisjoint(forbidden_calls), name

    handlers = [
        handler
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
    ]
    assert len(handlers) == 28
    reraised = []
    converted = []
    for handler in handlers:
        body = "\n".join(ast.unparse(node) for node in handler.body)
        if any(isinstance(node, ast.Raise) for node in ast.walk(handler)):
            reraised.append(body)
        else:
            converted.append(body)
    assert len(reraised) == 3
    assert any("attempt.abort()" in body and "optimizer.zero_grad" in body for body in reraised)
    assert (
        sum("os.replace(output_checkpoint, staging_checkpoint)" in body for body in reraised) == 2
    )
    assert len(converted) == 25
    assert any("dataset_contract[0]" in body and "'status': 'FAIL'" in body for body in converted)
    assert sum("objective_error = error" in body for body in converted) == 1
    assert sum("precheckpoint_error[0]" in body for body in converted) == 1
    assert sum("publish_error[0]" in body for body in converted) == 4
    expected_conversions = {
        "selection_error = error": 1,
        "setup_error = error": 2,
        "capture_error = error": 1,
        "probe_setup_error = error": 1,
        "point_error = error": 1,
        "backward_error = error": 1,
        "gradient_validation_error = error": 1,
        "evidence_error = error": 1,
        "gradient_error = error": 1,
        "update_error = error": 2,
        "counter_error = error": 1,
        "final_error = error": 1,
        "final_local_error = error": 1,
        "batch_materialization_error = error": 1,
    }
    for marker, expected_count in expected_conversions.items():
        assert sum(marker == body.strip() for body in converted) == expected_count


def test_full_runner_source_contains_real_transaction_and_fail_closed_claims() -> None:
    source = _source()
    required = (
        "verify_native_patch(",
        "validate_prepared_native_source(",
        "sys.dont_write_bytecode = True",
        "require_persistent_run_root(args.run_dir)",
        "acquire_distributed_run_lease(",
        "register_native_fsdp_forward_methods(policy)",
        "strip_targetless_alignment_teacher_heads(policy)",
        "native predictive training requires the complete trainable VLM host",
        "build_lingbot_official_optimizer(",
        "build_lingbot_representation_optimizer(",
        "install_torch_2_8_sparse_optimizer_state_backport(torch)",
        "enable_fp32=optimizer_contract.enable_fp32",
        "LingBotPredictiveTargetCache.load(",
        "LingBotCurrentGridTargetCache.load(",
        "sample_qwen_packed_patch_mask(",
        "sample_qwen_whole_view_omission(",
        "objective_runner = (",
        "run_native_calvin_representation_objective",
        "run_native_calvin_full_objective",
        "run_step_objective: Callable[[], Any] = partial(",
        "run_gradient_audit_objective: Callable[[], Any] = partial(",
        "                del overshoot_callback",
        "rollout_native_prior_prediction(",
        "sample_temporal_batch_plan(",
        "attempt.finish(optimizer_attempt)",
        "require_checkpoint_write_capacity(checkpoint_root)",
        "dist.broadcast_object_list(precheckpoint_error, src=0)",
        "native full pre-checkpoint capacity validation failed",
        "checkpointer.save(",
        "checkpointer.load(",
        "_validate_action_fsdp2_topology(policy)",
        "_validate_vlm_fsdp2_topology(policy)",
        '"action_fsdp2_topology": action_fsdp2_topology',
        '"vlm_fsdp2_topology": vlm_fsdp2_topology',
        '"alignment_teacher_prune": alignment_teacher_prune',
        '"omitted_static_binding_enabled": args.source_prediction_mode == "omitted_static"',
        '"complete_adr74_objective": False',
        '"long_training_authorized": (',
        "validate_training_authorization(",
        "_distributed_family_gradient_diagnostics(",
        "_distributed_predictive_host_gradient_diagnostics(",
        "_parameter_gradient_snapshot(",
        "load_predictive_target_audit(",
        "load_predictive_teacher_causality_audit(",
        "load_predictive_temporal_audit(",
        "render_native_relation_visuals(",
        "validate_representation_objective_report(",
        "native_representation_report.json",
    )
    for fragment in required:
        assert fragment in source
    assert source.rindex("require_checkpoint_write_capacity(checkpoint_root)") < source.index(
        "checkpointer.save("
    )
    assert source.rindex("dist.broadcast_object_list(precheckpoint_error, src=0)") < source.index(
        "checkpointer.save("
    )
    assert source.index("install_torch_2_8_sparse_optimizer_state_backport(torch)") < source.index(
        "from lingbotvla.checkpoint import build_checkpointer"
    )
    assert source.index(
        "representation checkpoint lacks its immutable report before load"
    ) < source.index("checkpointer.load(")
    assert source.rindex("_write_text_durable(report_path, payload)") < source.index(
        "dist.broadcast_object_list(publish_error, src=0)", source.index("report_filename = (")
    )
    assert source.count("torch_module.autograd.grad(") == 1
    assert "posterior_credit = torch_module.autograd.grad(" in source
    assert "_backward_behavior_total_host(" in source
    assert "_backward_behavior_via_posterior_host(" in source
    assert "allow_unused=True" not in source
    for forbidden in (
        "picf_next.unified",
        "semantic_scorer",
        "confidence_controller",
        "action_layer_adapter",
    ):
        assert forbidden not in source


def test_full_runner_uses_only_lingbot_layer_checkpointing_for_overshoot() -> None:
    tree = ast.parse(_source())
    rollout_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "rollout_native_prior_prediction"
    ]
    assert len(rollout_calls) == 1
    assert {keyword.arg for keyword in rollout_calls[0].keywords} == {
        "request",
        "target_name",
    }

    parallelize_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_parallelize_model"
    ]
    assert len(parallelize_calls) == 1
    gradient_checkpointing = next(
        keyword.value
        for keyword in parallelize_calls[0].keywords
        if keyword.arg == "enable_gradient_checkpointing"
    )
    assert isinstance(gradient_checkpointing, ast.Constant)
    assert gradient_checkpointing.value is True


def test_full_runner_family_audit_uses_isolated_graphs_and_restores_rng() -> None:
    source = _source()
    audit_runner_start = source.index(
        "            run_gradient_audit_objective: Callable[[], Any] = partial("
    )
    audit_runner_end = source.index(
        "\n\n            if args.behavior_causal_probe_output is not None:",
        audit_runner_start,
    )
    audit_runner = source[audit_runner_start:audit_runner_end]
    assert "batches=(primary_batch,)" in audit_runner
    assert "source_mask=None" in audit_runner
    assert "omitted_static_view=None" in audit_runner
    assert "overshoot_factory=None" in audit_runner

    audit_start = source.index(
        "                    audit_rng_state = _capture_rank_rng(torch, np, device=device)"
    )
    main_forward = source.index("                result = run_step_objective()", audit_start)
    audit = source[audit_start:main_forward]
    assert 'for family_name in ("action", "predictive", "structural"):' in audit
    assert audit.count("diagnostic_result = run_gradient_audit_objective()") == 1
    assert audit.count("total_result = run_gradient_audit_objective()") == 1
    assert audit.count("via_result = run_gradient_audit_objective()") == 1
    assert "run_step_objective()" not in audit
    assert audit.count("_backward_isolated_objective_family(") == 1
    assert "family_term.backward()" not in audit
    assert "torch.autograd.grad(" not in audit
    assert "retain_graph=True" not in audit
    assert audit.count("_backward_behavior_total_host(") == 1
    assert audit.count("_backward_behavior_via_posterior_host(") == 1
    assert 'family_name == "predictive" and args.behavior_conditioned_prediction' in audit
    assert audit.count("_restore_rank_rng(audit_rng_state, torch, np, device=device)") == 4
    assert audit.index("optimizer.zero_grad(set_to_none=True)") < audit.index(
        "diagnostic_result = run_gradient_audit_objective()"
    )
    assert audit.index("_backward_isolated_objective_family(") < audit.index(
        "_parameter_gradient_snapshot("
    )
    main_forward = source.index("                result = run_step_objective()", audit_start)
    main_backward = source.index(
        "                result.objective.objective.total.backward()",
        audit_start,
    )
    failure_exchange = source.index(
        "                objective_failures = _distributed_pre_backward_failures(",
        main_forward,
    )
    objective_ready = source.index(
        '                    "objective_ready",',
        failure_exchange,
    )
    assert main_forward < failure_exchange < objective_ready < main_backward


def test_behavior_host_decomposition_uses_backward_parameter_hooks() -> None:
    helper_source = inspect.getsource(_backward_behavior_via_posterior_host)
    assert "retain_graph=True" not in helper_source

    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    selected = {"early": ("shared.weight", parameter)}

    total_rows = 2.0 * parameter
    total_term = (total_rows + 3.0 * parameter).sum()
    total = _backward_behavior_total_host(
        total_term,
        selected_host=selected,
        torch_module=torch,
    )
    parameter.grad = None

    via_rows = 2.0 * parameter
    via_term = (via_rows + 3.0 * parameter).sum()
    posterior_credit, via = _backward_behavior_via_posterior_host(
        via_term,
        behavior_rows=via_rows,
        selected_host=selected,
        torch_module=torch,
    )

    assert posterior_credit.item() == pytest.approx(1.0)
    assert total["early"].item() == pytest.approx(5.0)
    assert via["early"].item() == pytest.approx(2.0)
    assert (total["early"] - via["early"]).item() == pytest.approx(3.0)


def test_behavior_runner_requires_exact_horizon_one_cache_coverage_before_loading() -> None:
    source = _source()
    coverage = source.index("        expected_predictive_coverage = ")
    cache_load = source.index("        predictive_cache = LingBotPredictiveTargetCache.load(")
    block = source[coverage:cache_load]

    assert coverage < cache_load
    assert "required_horizons=(1,) if args.behavior_conditioned_prediction else ()" in block
    assert 'predictive_report["pair_keys_sha256"]' in block
    assert 'predictive_report["coverage_sha256"]' in block
    assert 'predictive_report["expected_record_count"]' in block
    assert "predictive cache does not cover the exact training objective" in block


def test_full_runner_counterfactual_and_lane_commit_order_is_frozen() -> None:
    source = _source()
    loop_start = source.rindex("            started = time.perf_counter()")
    assert loop_start < source.index(
        "            planned = build_planned_native_calvin_batch(", loop_start
    )
    loop_end = source.index("            step_reports.append(", loop_start)
    step = source[loop_start:loop_end]
    ordered = (
        "result.objective.objective.total.backward()",
        "run_native_correction_counterfactual_forwards(",
        'predictive_counterfactual_weight_boundary = "pre_update_post_backward"',
        "row_bindings_by_batch=result.objective.row_bindings_by_batch,",
        "attempt.finish(optimizer_attempt)",
        "step_seconds = time.perf_counter() - started",
    )
    positions = tuple(step.index(fragment) for fragment in ordered)
    assert positions == tuple(sorted(positions))


def test_full_runner_validates_text_and_vision_fsdp2_blocks(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    text_type = type("Qwen3VLTextDecoderLayer", (nn.Module,), {})
    vision_type = type("Qwen3VLVisionBlock", (nn.Module,), {})
    modules = {
        "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0": text_type(),
        "model.qwenvl_with_expert.qwenvl.model.visual.blocks.0": vision_type(),
    }
    for module in modules.values():
        module.reshard = lambda: None
        module.unshard = lambda: None

    class Policy:
        _lingbot_vlm_fsdp2_topology = {
            "text": ("model.qwenvl_with_expert.qwenvl.model.language_model.layers.0",),
            "vision": ("model.qwenvl_with_expert.qwenvl.model.visual.blocks.0",),
        }

        def get_submodule(self, path: str) -> object:
            return modules[path]

    topology = _validate_vlm_fsdp2_topology(Policy())
    assert topology == {
        "text_block_count": 1,
        "text_block_paths": ["model.qwenvl_with_expert.qwenvl.model.language_model.layers.0"],
        "vision_block_count": 1,
        "vision_block_paths": ["model.qwenvl_with_expert.qwenvl.model.visual.blocks.0"],
    }
    path = full_objective_report_factory(tmp_path / "full.json", digest="a" * 64)
    report = json.loads(path.read_text())
    report["vlm_fsdp2_topology"] = topology
    assert (
        validate_full_objective_report(
            report,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )


def test_full_runner_rejects_incomplete_vlm_fsdp2_topology() -> None:
    class Policy:
        _lingbot_vlm_fsdp2_topology = {"text": ("text.0",), "vision": ()}

        def get_submodule(self, path: str) -> object:
            raise AssertionError(path)

    with pytest.raises(RuntimeError, match="no vision blocks"):
        _validate_vlm_fsdp2_topology(Policy())


def test_full_runner_rejects_nested_text_projection_fsdp2() -> None:
    text_type = type("Qwen3VLTextDecoderLayer", (nn.Module,), {})
    vision_type = type("Qwen3VLVisionBlock", (nn.Module,), {})
    text = text_type()
    text.projection = nn.Linear(4, 4, bias=False)
    vision = vision_type()
    for module in (text, vision, text.projection):
        module.reshard = lambda: None
        module.unshard = lambda: None
    modules = {"text.0": text, "vision.0": vision}

    class Policy:
        _lingbot_vlm_fsdp2_topology = {
            "text": ("text.0",),
            "vision": ("vision.0",),
        }

        def get_submodule(self, path: str) -> nn.Module:
            return modules[path]

    with pytest.raises(RuntimeError, match="nested FSDP2 modules"):
        _validate_vlm_fsdp2_topology(Policy())


def test_full_runner_validates_action_block_fsdp2_without_nested_groups() -> None:
    action_type = type("Qwen2DecoderLayer", (nn.Module,), {})
    block = action_type()
    block.projection = nn.Linear(4, 4, bias=False)
    block.reshard = lambda: None
    block.unshard = lambda: None

    class _ActionModel:
        layers = (block,)

    class _Expert:
        model = _ActionModel()

    class _Host:
        qwen_expert = _Expert()

    class _Root:
        qwenvl_with_expert = _Host()

    class Policy:
        model = _Root()

    assert _validate_action_fsdp2_topology(Policy()) == {
        "schema": "picf-next.lingbot-action-block-fsdp2.v1",
        "block_count": 1,
        "block_paths": ["model.qwenvl_with_expert.qwen_expert.model.layers.0"],
        "maximum_block_bf16_bytes_upper_bound": 32,
    }


def test_full_runner_rejects_nested_action_projection_fsdp2() -> None:
    action_type = type("Qwen2DecoderLayer", (nn.Module,), {})
    block = action_type()
    block.projection = nn.Linear(4, 4, bias=False)
    block.reshard = lambda: None
    block.unshard = lambda: None
    block.projection.reshard = lambda: None
    block.projection.unshard = lambda: None

    class _ActionModel:
        layers = (block,)

    class _Expert:
        model = _ActionModel()

    class _Host:
        qwen_expert = _Expert()

    class _Root:
        qwenvl_with_expert = _Host()

    class Policy:
        model = _Root()

    with pytest.raises(RuntimeError, match="nested FSDP2 modules"):
        _validate_action_fsdp2_topology(Policy())


def test_full_runner_paths_and_source_mask_require_published_cache(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    _validate_paths_and_args(args)
    args.physical_sidecar_manifest.unlink()
    with pytest.raises(FileNotFoundError, match="source/config/data files are absent"):
        _validate_paths_and_args(args)
    args.physical_sidecar_manifest.write_text("{}")
    _validate_paths_and_args(args)
    args.current_grid_cache_root = None
    args.current_grid_cache_build_report = None
    args.current_grid_cache_build_report_sha256 = None
    with pytest.raises(ValueError, match="paths are absent"):
        _validate_paths_and_args(args)
    args.current_grid_cache_root = tmp_path / "current-grid"
    args.current_grid_cache_build_report = tmp_path / "current-grid-report.json"
    args.current_grid_cache_build_report_sha256 = "1" * 64
    _validate_paths_and_args(args)
    args.source_mask_token_fraction = 0.0
    args.source_prediction_mode = "current_grid"
    with pytest.raises(ValueError, match="positive token fraction"):
        _validate_paths_and_args(args)
    args.source_mask_token_fraction = 0.0625
    args.source_mask_probability = 0.0
    args.source_prediction_mode = "unsupported"
    with pytest.raises(ValueError, match="mode is unsupported"):
        _validate_paths_and_args(args)
    args.source_prediction_mode = "omitted_static"
    with pytest.raises(ValueError, match="source_mask_probability must be positive"):
        _validate_paths_and_args(args)
    args.source_mask_probability = 0.1
    args.fsdp2_placement = "unsupported"
    with pytest.raises(ValueError, match="FSDP2 placement"):
        _validate_paths_and_args(args)
    args.fsdp2_placement = "cpu-offload"
    args.cuda_allocator = "unsupported"
    with pytest.raises(ValueError, match="CUDA allocator"):
        _validate_paths_and_args(args)
    args.cuda_allocator = "native"
    args.phase = "resume"
    args.load_global_step = 1
    args.invocation_steps = 10
    args.authorization_manifest = tmp_path / "authorization.json"
    args.authorization_manifest.write_text("{}")
    args.authorization_manifest_sha256 = _sha(args.authorization_manifest.read_bytes())
    args.visual_audit_every = 1
    with pytest.raises(ValueError, match="exceeds the frozen stream plan"):
        _validate_paths_and_args(args)


def test_full_runner_rejects_implicit_family_weights(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.predictive_weight = None
    with pytest.raises(ValueError, match="must be explicitly provided"):
        _validate_paths_and_args(args)
    args.predictive_weight = 0.004
    args.structural_weight = None
    with pytest.raises(ValueError, match="must be explicitly provided"):
        _validate_paths_and_args(args)

    args.structural_weight = 0.004
    args.gradient_audit_steps = None
    with pytest.raises(ValueError, match="audit steps must be explicitly provided"):
        _validate_paths_and_args(args)


def test_full_runner_rejects_step_one_gradient_audit_before_asset_loading(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.gradient_audit_steps = (1, 2, 3)
    with pytest.raises(ValueError, match="fresh state-bootstrap step"):
        _validate_paths_and_args(args)


def test_full_runner_gradient_audit_schedule_tracks_lane_interleave(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.total_planned_steps = 20
    args.lane_interleave_factor = 8
    args.gradient_audit_steps = (2, 3, 20)
    with pytest.raises(ValueError, match="first/second recurrent correction steps 9/17"):
        _validate_paths_and_args(args)

    args.gradient_audit_steps = (9, 17)
    _validate_paths_and_args(args)


def test_full_runner_loss_visual_trial_is_bounded_and_defers_family_audits(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.phase = "resume"
    args.load_global_step = 1
    args.invocation_steps = 19
    args.total_planned_steps = 30_000
    args.authorization_manifest = tmp_path / "authorization.json"
    args.authorization_manifest.write_text("{}")
    args.authorization_manifest_sha256 = _sha(args.authorization_manifest.read_bytes())
    args.visual_audit_every = 4
    args.evidence_profile = "loss_visual_trial"
    args.gradient_audit_steps = (60, 120)
    _validate_paths_and_args(args)

    args.gradient_audit_steps = (20, 60)
    with pytest.raises(ValueError, match="must lie after its saved step"):
        _validate_paths_and_args(args)
    args.gradient_audit_steps = (60, 120)
    args.invocation_steps = 20
    with pytest.raises(ValueError, match="cannot exceed global step 20"):
        _validate_paths_and_args(args)
    args.invocation_steps = 19
    args.evidence_profile = "acceptance"
    with pytest.raises(ValueError, match="first/second recurrent correction steps 2/3"):
        _validate_paths_and_args(args)


def test_full_runner_fixed_batch_mode_is_fresh_isolated_and_immutable(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.predictive_fixed_batch_arm = "full_host"
    args.predictive_fixed_batch_curve_points = 8
    args.predictive_fixed_batch_output = tmp_path / "fixed-batch.json"
    _validate_paths_and_args(args)

    args.phase = "resume"
    args.load_global_step = 1
    with pytest.raises(ValueError, match="fresh released checkpoint"):
        _validate_paths_and_args(args)
    args.phase = "fresh"
    args.load_global_step = 0
    args.predictive_fixed_batch_curve_points = 1
    with pytest.raises(ValueError, match="at least two"):
        _validate_paths_and_args(args)
    args.predictive_fixed_batch_curve_points = 8
    args.predictive_fixed_batch_output.write_text("{}")
    with pytest.raises(FileExistsError, match="already exists"):
        _validate_paths_and_args(args)


def test_relation_geometry_fixed_batch_mode_is_isolated_and_immutable(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.relation_geometry_fixed_batch_arm = "existing_readout_frozen_host"
    args.relation_geometry_fixed_batch_curve_points = 8
    args.relation_geometry_fixed_batch_output = tmp_path / "relation.json"
    args.relation_geometry_fixed_batch_visual_root = tmp_path / "relation-visuals"
    _validate_paths_and_args(args)

    args.predictive_fixed_batch_arm = "full_host"
    args.predictive_fixed_batch_curve_points = 8
    args.predictive_fixed_batch_output = tmp_path / "predictive.json"
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_paths_and_args(args)
    args.predictive_fixed_batch_arm = None
    args.predictive_fixed_batch_curve_points = 0
    args.predictive_fixed_batch_output = None

    args.relation_geometry_fixed_batch_sample_step = args.total_planned_steps
    with pytest.raises(ValueError, match="outside the frozen plan"):
        _validate_paths_and_args(args)
    args.relation_geometry_fixed_batch_sample_step = 0
    args.relation_geometry_fixed_batch_visual_root.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _validate_paths_and_args(args)


@pytest.mark.parametrize(
    "arm",
    (RELATION_DEPTH_PROBE_ARM, RELATION_BILINEAR_PROBE_ARM),
)
def test_external_relation_fixed_batch_mode_requires_preregistered_curve_length(
    tmp_path: Path,
    arm: str,
) -> None:
    args = _runner_args(tmp_path)
    args.relation_geometry_fixed_batch_arm = arm
    args.relation_geometry_fixed_batch_curve_points = 41
    args.relation_geometry_fixed_batch_output = tmp_path / "depth.json"
    args.relation_geometry_fixed_batch_visual_root = tmp_path / "depth-visuals"
    _validate_paths_and_args(args)

    args.relation_geometry_fixed_batch_curve_points = 40
    with pytest.raises(ValueError, match="exactly 41"):
        _validate_paths_and_args(args)


def test_fixed_batch_returns_before_constructing_streaming_training_coordinator() -> None:
    main = next(
        node
        for node in ast.parse(TOOL.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    fixed_batch_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_predictive_fixed_batch_arm"
    )
    fixed_batch_branch = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If)
        and fixed_batch_call in tuple(ast.walk(node))
        and any(isinstance(child, ast.Return) for child in node.body)
    )
    coordinator_assignment = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "coordinator" for target in node.targets
        )
    )
    assert fixed_batch_branch.end_lineno is not None
    assert coordinator_assignment.lineno > fixed_batch_branch.end_lineno


def test_relation_probe_returns_before_constructing_streaming_training_coordinator() -> None:
    main = next(
        node
        for node in ast.parse(TOOL.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    probe_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_relation_geometry_fixed_batch_arm"
    )
    probe_branch = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If) and probe_call in tuple(ast.walk(node))
    )
    coordinator_assignment = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "coordinator" for target in node.targets
        )
    )
    assert any(isinstance(node, ast.Return) for node in probe_branch.body)
    assert probe_branch.end_lineno is not None
    assert coordinator_assignment.lineno > probe_branch.end_lineno


def test_zero_update_probes_skip_unrelated_long_run_gradient_schedule_validation() -> None:
    main = next(
        node
        for node in ast.parse(TOOL.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    coverage_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_gradient_audit_target_coverage"
    )
    branch = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If) and coverage_call in tuple(ast.walk(node))
    )
    assert ast.unparse(branch.test) == (
        "args.predictive_fixed_batch_arm is None and args.relation_geometry_fixed_batch_arm is "
        "None and (not args.behavior_conditioned_prediction)"
    )


def test_fixed_batch_curve_observes_every_update_without_an_unobserved_final_update() -> None:
    function = next(
        node
        for node in ast.parse(TOOL.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_predictive_fixed_batch_arm"
    )
    optimizer_updates = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "optimizer_updates"
            for target in node.targets
        )
    )
    assert isinstance(optimizer_updates.value, ast.BinOp)
    assert isinstance(optimizer_updates.value.left, ast.Name)
    assert optimizer_updates.value.left.id == "curve_points"
    assert isinstance(optimizer_updates.value.op, ast.Sub)
    assert isinstance(optimizer_updates.value.right, ast.Constant)
    assert optimizer_updates.value.right.value == 1

    optimizer_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_optimizer_attempt"
    )
    update_branch = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and optimizer_call in tuple(ast.walk(node))
    )
    assert isinstance(update_branch.test, ast.Compare)
    assert ast.unparse(update_branch.test) == "curve_index < optimizer_updates"

    loss_record = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "local_losses"
        and node.func.attr == "append"
    )
    assert update_branch.end_lineno is not None
    assert loss_record.lineno > update_branch.end_lineno


def test_fixed_batch_probe_preserves_moe_bias_and_resets_only_counters() -> None:
    class Moe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("e_score_correction_bias", torch.tensor([0.2, -0.1]))
            self.register_buffer("tokens_per_expert", torch.tensor([3.0, 4.0]))

    policy = Moe()
    before = _moe_routing_bias_snapshot(policy, torch_module=torch)
    _reset_moe_probe_counters(policy, torch_module=torch)
    assert torch.count_nonzero(policy.tokens_per_expert) == 0
    assert _moe_routing_bias_matches(policy, before, torch_module=torch)
    policy.e_score_correction_bias.add_(1)
    assert not _moe_routing_bias_matches(policy, before, torch_module=torch)


def test_fixed_batch_plan_selects_first_global_two_frame_pair() -> None:
    class Dataset:
        @staticmethod
        def available_future_transitions_by_key(key: str) -> int:
            return int(not key.endswith("0"))

        @staticmethod
        def future_sample_keys(key: str, *, count: int) -> tuple[str, ...]:
            assert count == 1
            return (f"{key}-next",)

        @staticmethod
        def source_global_index_by_key(key: str) -> int:
            assert key.endswith("-next")
            return int(key.removeprefix("sample-").removesuffix("-next"))

    class Dist:
        class ReduceOp:
            MIN = "min"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op == "min"

    def build_planned(
        _stream: object,
        _dataset: object,
        *,
        optimizer_step: int,
        **_kwargs: object,
    ) -> object:
        routing = SimpleNamespace(sample_keys=(f"sample-{optimizer_step}",))
        return SimpleNamespace(training=SimpleNamespace(routing=routing))

    dataset = Dataset()
    selected, planned = _select_fixed_batch_plan(
        stream_plan=object(),
        dataset=dataset,
        rank=0,
        world_size=2,
        total_planned_steps=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dist=Dist(),
        torch_module=torch,
        build_planned_batch=build_planned,
        target_has_nontrivial_shuffle=lambda key: (
            dataset.source_global_index_by_key(dataset.future_sample_keys(key, count=1)[0]) >= 2
        ),
    )
    assert selected == 2
    assert planned.training.routing.sample_keys == ("sample-2",)


def test_fixed_batch_plan_scans_the_complete_frozen_plan() -> None:
    class Dataset:
        @staticmethod
        def available_future_transitions_by_key(_key: str) -> int:
            return 1

    class Dist:
        class ReduceOp:
            MIN = "min"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op == "min"

    def build_planned(
        _stream: object,
        _dataset: object,
        *,
        optimizer_step: int,
        **_kwargs: object,
    ) -> object:
        routing = SimpleNamespace(sample_keys=(f"sample-{optimizer_step}",))
        return SimpleNamespace(training=SimpleNamespace(routing=routing))

    selected, planned = _select_fixed_batch_plan(
        stream_plan=object(),
        dataset=Dataset(),
        rank=0,
        world_size=2,
        total_planned_steps=1026,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dist=Dist(),
        torch_module=torch,
        build_planned_batch=build_planned,
        target_has_nontrivial_shuffle=lambda key: key == "sample-1025",
    )

    assert selected == 1025
    assert planned.training.routing.sample_keys == ("sample-1025",)


def test_fixed_batch_readout_arm_executes_and_publishes_one_complete_report(
    tmp_path: Path,
) -> None:
    class Policy(nn.Module):
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
            self.register_buffer("e_score_correction_bias", torch.tensor([0.2, -0.1]))
            self.register_buffer("tokens_per_expert", torch.tensor([3.0, 4.0]))

    class FakeCuda:
        @staticmethod
        def manual_seed_all(_seed: int) -> None:
            return None

        @staticmethod
        def reset_peak_memory_stats(_device: torch.device) -> None:
            return None

        @staticmethod
        def max_memory_reserved(_device: torch.device) -> int:
            return 0

    class TorchProxy:
        Tensor = torch.Tensor
        autograd = torch.autograd
        bfloat16 = torch.bfloat16
        cuda = FakeCuda()
        float64 = torch.float64
        int32 = torch.int32
        nn = torch.nn

        def __getattr__(self, name: str) -> object:
            return getattr(torch, name)

    class Dist:
        class ReduceOp:
            MIN = "min"
            SUM = "sum"

        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def all_reduce(value: torch.Tensor, *, op: str) -> None:
            if op == Dist.ReduceOp.SUM:
                value.mul_(2)
            else:
                assert op == Dist.ReduceOp.MIN

        @staticmethod
        def all_gather_object(outputs: list[object], value: object) -> None:
            outputs[0] = value
            if isinstance(value, dict):
                peer = dict(value)
                peer["rank"] = 1
                outputs[1] = peer
            else:
                outputs[1] = value

        @staticmethod
        def broadcast_object_list(_values: list[object], *, src: int) -> None:
            assert src == 0

        @staticmethod
        def barrier() -> None:
            return None

    class Dataset:
        @staticmethod
        def available_future_transitions_by_key(_key: str) -> int:
            return 1

        @staticmethod
        def future_sample_keys(_key: str, *, count: int) -> tuple[str, ...]:
            assert count == 1
            return ("continuation",)

        @staticmethod
        def source_global_index_by_key(key: str) -> int:
            assert key == "continuation"
            return 1

    class CurrentGridCache:
        manifest_sha256 = "4" * 64

        @staticmethod
        def supported_current_summary_count(**_kwargs: object) -> int:
            return 3

    def batch(*, continuation: bool) -> SimpleNamespace:
        return SimpleNamespace(
            controls=SimpleNamespace(
                reset=torch.tensor([False]),
                token_valid=torch.tensor([True]),
            ),
            routing=SimpleNamespace(
                batch_size=1,
                sample_keys=("continuation" if continuation else "primary",),
            ),
            source_digest=("2" if continuation else "1") * 64,
        )

    primary_plan = SimpleNamespace(
        continuation=False,
        training=SimpleNamespace(routing=SimpleNamespace(sample_keys=("primary",))),
    )
    continuation_plan = SimpleNamespace(continuation=True)
    policy = Policy()
    graph = policy.picf_native_graph
    scope = configure_fixed_batch_trainable_scope(policy, graph, arm="readout_only")
    optimizer = torch.optim.SGD(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    output = tmp_path / "readout-only.json"
    args = SimpleNamespace(
        capacity=3,
        current_grid_term_weight=1.0,
        max_grad_norm=1.0,
        minimum_supervised_fraction=1.0,
        omitted_static_term_weight=1.0,
        predictive_fixed_batch_arm="readout_only",
        predictive_fixed_batch_curve_points=3,
        predictive_fixed_batch_output=output,
        predictive_loss_power=1.0,
        predictive_term_weight=1.0,
        seed=7,
        total_planned_steps=1,
    )

    observed_prior_row_bindings: list[object] = []

    def run_full_objective(model: Policy, **kwargs: object) -> SimpleNamespace:
        prior_row_bindings = kwargs["prior_row_bindings_by_batch"]
        observed_prior_row_bindings.append(prior_row_bindings)
        weight = model.picf_native_graph.predictive_readouts["dino_video"].weight
        loss = weight.square().mean()
        objective = SimpleNamespace(family_terms={"predictive": loss})
        row_bindings = (("object/a", 0),) if prior_row_bindings == ((),) else prior_row_bindings[0]
        return SimpleNamespace(
            objective=SimpleNamespace(
                objective=objective,
                row_bindings_by_batch=(row_bindings,),
            )
        )

    _run_predictive_fixed_batch_arm(
        args=args,
        rank=0,
        device=torch.device("cpu"),
        dist=Dist(),
        torch_module=TorchProxy(),
        policy=policy,
        graph=graph,
        optimizer=optimizer,
        optimizer_contract=SimpleNamespace(
            algorithm="lingbot_distributed_muon_with_adamw_fallback",
            learning_rate=0.01,
            scheduler="constant",
            weight_decay=0.0,
        ),
        trainable_scope=scope,
        stream_plan=SimpleNamespace(plan_sha256="3" * 64),
        dataset=Dataset(),
        collate_planned=lambda planned: batch(continuation=planned.continuation),
        build_planned_batch=lambda *_args, **_kwargs: primary_plan,
        build_continuation_batch=lambda *_args, **_kwargs: continuation_plan,
        run_full_objective=run_full_objective,
        predictive_cache=SimpleNamespace(
            contract=SimpleNamespace(minimum_visible_fraction=0.1),
            manifest_sha256="5" * 64,
        ),
        current_grid_cache=CurrentGridCache(),
        physical_sidecar=SimpleNamespace(manifest_sha256="6" * 64),
        task_identity_resolver=lambda *_args, **_kwargs: (),
        patch_size=14,
        merge_size=2,
        objective_config=object(),
        structural_config=object(),
        derive_subseed_fn=lambda *_args: 11,
        patch_sha256="7" * 64,
        execution_sha256="8" * 64,
        implementation_sha256="9" * 64,
        model_family_sha256="a" * 64,
        dataset_contract_report={"manifest_sha256": "b" * 64},
    )

    report = validate_predictive_fixed_batch_arm_report(json.loads(output.read_text()))
    assert report["arm"] == "readout_only"
    assert report["curve_point_count"] == 3
    assert report["optimizer_update_count"] == 2
    assert report["shared_host_gradient_probe"] is None
    assert report["moe_routing_bias_unchanged"] is True
    assert report["global_loss_curve"][-1] < report["global_loss_curve"][0]
    assert torch.count_nonzero(policy.tokens_per_expert) == 0
    assert observed_prior_row_bindings == [
        ((),),
        ((("object/a", 0),),),
        ((("object/a", 0),),),
    ]


def test_relation_geometry_readout_arm_executes_updates_and_publishes_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Policy(nn.Module):
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
            self.register_buffer("e_score_correction_bias", torch.tensor([0.2, -0.1]))
            self.register_buffer("tokens_per_expert", torch.tensor([3.0, 4.0]))

    class FakeCuda:
        @staticmethod
        def manual_seed_all(_seed: int) -> None:
            return None

        @staticmethod
        def reset_peak_memory_stats(_device: torch.device) -> None:
            return None

        @staticmethod
        def max_memory_reserved(_device: torch.device) -> int:
            return 0

    class TorchProxy:
        Tensor = torch.Tensor
        autograd = torch.autograd
        bfloat16 = torch.bfloat16
        cuda = FakeCuda()
        float64 = torch.float64
        int32 = torch.int32
        nn = torch.nn

        def __getattr__(self, name: str) -> object:
            return getattr(torch, name)

    class Dist:
        class ReduceOp:
            MIN = "min"
            SUM = "sum"

        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def all_reduce(value: torch.Tensor, *, op: str) -> None:
            if op == Dist.ReduceOp.SUM:
                value.mul_(2)
            else:
                assert op == Dist.ReduceOp.MIN

        @staticmethod
        def all_gather_object(outputs: list[object], value: object) -> None:
            outputs[0] = value
            if isinstance(value, dict) and "frame_sample_keys" in value:
                peer = deepcopy(value)
                peer["rank"] = 1
                peer["frame_sample_keys"] = ["rank1/current", "rank1/next"]
                peer["frame_source_digests"] = ["3" * 64, "4" * 64]
                peer["forward_seed"] = 12
                outputs[1] = peer
            else:
                outputs[1] = value

        @staticmethod
        def broadcast_object_list(_values: list[object], *, src: int) -> None:
            assert src == 0

        @staticmethod
        def barrier() -> None:
            return None

    class Dataset:
        @staticmethod
        def available_future_transitions_by_key(_key: str) -> int:
            return 1

        @staticmethod
        def task_key_by_key(key: str) -> str:
            return "push_blue_block_left" if key == "rank0/current" else "turn_on_led"

        @staticmethod
        def locator_by_key(key: str) -> SimpleNamespace:
            return SimpleNamespace(
                segment_index=0 if key == "rank0/current" else 1,
                global_index=10 if key == "rank0/current" else 20,
            )

    class PhysicalSidecar:
        manifest_sha256 = "7" * 64

        @staticmethod
        def __call__(_segment_index: int, _global_index: int) -> SimpleNamespace:
            return SimpleNamespace(
                identity_keys=("object/a",),
                cameras=(
                    SimpleNamespace(
                        owner_index=torch.tensor([[1]], dtype=torch.uint8),
                        owner_supervised=torch.tensor([[True]], dtype=torch.bool),
                    ),
                ),
            )

    class StreamPlan:
        plan_sha256 = "5" * 64

        @staticmethod
        def global_batch(_step: int) -> SimpleNamespace:
            return SimpleNamespace(
                transitions=(
                    SimpleNamespace(sample=SimpleNamespace(sample_key="rank0/current")),
                    SimpleNamespace(sample=SimpleNamespace(sample_key="rank1/current")),
                )
            )

    def batch(*, continuation: bool) -> SimpleNamespace:
        return SimpleNamespace(
            controls=SimpleNamespace(
                reset=torch.tensor([False]),
                token_valid=torch.tensor([True]),
            ),
            routing=SimpleNamespace(
                batch_size=1,
                sample_keys=("rank0/next" if continuation else "rank0/current",),
            ),
            source_digest=("2" if continuation else "1") * 64,
            model_inputs={},
        )

    primary_plan = SimpleNamespace(
        continuation=False,
        training=SimpleNamespace(host_items=({"task": "touch object a"},)),
    )
    continuation_plan = SimpleNamespace(continuation=True)
    policy = Policy()
    graph = policy.picf_native_graph
    scope = configure_relation_geometry_trainable_scope(
        policy,
        graph,
        arm="existing_readout_frozen_host",
    )
    optimizer = torch.optim.SGD(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    output = tmp_path / "relation-readout.json"
    visual_root = tmp_path / "relation-visuals"
    args = SimpleNamespace(
        capacity=3,
        current_grid_term_weight=1.0,
        max_grad_norm=10.0,
        minimum_supervised_fraction=1.0,
        omitted_static_term_weight=1.0,
        predictive_loss_power=1.0,
        predictive_term_weight=1.0,
        relation_geometry_fixed_batch_arm="existing_readout_frozen_host",
        relation_geometry_fixed_batch_curve_points=3,
        relation_geometry_fixed_batch_output=output,
        relation_geometry_fixed_batch_sample_step=0,
        relation_geometry_fixed_batch_visual_root=visual_root,
        seed=7,
        total_planned_steps=1,
    )
    observed_prior_row_bindings: list[object] = []
    observed_argument_names: list[set[str]] = []
    stream_plan = StreamPlan()
    dataset = Dataset()
    physical_sidecar = PhysicalSidecar()

    def task_identity_resolver(_task_key: str) -> tuple[str, ...]:
        return ("object/a",)

    sample_selection = _select_relation_geometry_source_sample(
        args=args,
        stream_plan=stream_plan,
        dataset=dataset,
        physical_sidecar=physical_sidecar,
        task_identity_resolver=task_identity_resolver,
    )

    def run_relation_objective(model: Policy, **kwargs: object) -> SimpleNamespace:
        prior_row_bindings = kwargs["prior_row_bindings_by_batch"]
        observed_prior_row_bindings.append(prior_row_bindings)
        observed_argument_names.append(set(kwargs))
        weight = model.picf_native_graph.relation_readout.projection.weight
        ownership = weight.square().mean()
        nll = ownership + weight.sum() * 0 + 1.0
        action = weight.sum() * 0 + 0.2
        objective = SimpleNamespace(
            normalized_terms={
                "set/ownership": ownership,
                "set/ownership_nll": nll,
            },
            family_terms={"action": action},
        )
        relation = SimpleNamespace(structural_valid=torch.tensor([[True]]))
        return SimpleNamespace(
            primary=SimpleNamespace(context=SimpleNamespace(relation_output=relation)),
            objective=SimpleNamespace(
                objective=objective,
                row_bindings_by_batch=((("object/a", 0),),),
            ),
        )

    monkeypatch.setattr(
        full_runner,
        "build_task_row_diagnostics",
        lambda _objective: ({"target_rows": [0]},),
    )
    monkeypatch.setattr(
        full_runner,
        "validate_task_row_diagnostics",
        lambda value, *, expected_batch_size: value,
    )

    _run_relation_geometry_fixed_batch_arm(
        args=args,
        rank=0,
        device=torch.device("cpu"),
        dist=Dist(),
        torch_module=TorchProxy(),
        policy=policy,
        graph=graph,
        optimizer=optimizer,
        optimizer_contract=SimpleNamespace(
            algorithm="lingbot_distributed_muon_with_adamw_fallback",
            learning_rate=0.01,
            scheduler="constant",
            weight_decay=0.0,
        ),
        trainable_scope=scope,
        sample_selection=sample_selection,
        stream_plan=stream_plan,
        dataset=dataset,
        collate_planned=lambda planned: batch(continuation=planned.continuation),
        build_planned_batch=lambda *_args, **_kwargs: primary_plan,
        build_continuation_batch=lambda *_args, **_kwargs: continuation_plan,
        run_relation_objective=run_relation_objective,
        physical_sidecar=physical_sidecar,
        task_identity_resolver=task_identity_resolver,
        patch_size=14,
        merge_size=2,
        objective_config=object(),
        structural_config=object(),
        derive_subseed_fn=lambda *_args: 11,
        temporal_batch_seed_fn=lambda **_kwargs: 13,
        render_relation_visuals=lambda **kwargs: [
            {
                "path": f"rank_{kwargs['rank']}/point_{kwargs['global_step']}.png",
                "row_matched_soft_iou": [0.01],
            }
        ],
        patch_sha256="1" * 64,
        execution_sha256="2" * 64,
        implementation_sha256="3" * 64,
        model_family_sha256="4" * 64,
        dataset_contract_report={"manifest_sha256": "6" * 64},
    )

    report = validate_relation_geometry_arm_report(json.loads(output.read_text()))
    assert report["arm"] == "existing_readout_frozen_host"
    assert report["global_curves"]["ownership"][-1] < report["global_curves"]["ownership"][0]
    assert report["rank_reports"][0]["row_bindings"] == [["object/a", 0]]
    assert len(report["rank_reports"][0]["visual_artifacts_by_point"]) == 3
    assert observed_prior_row_bindings == [
        ((),),
        ((("object/a", 0),),),
        ((("object/a", 0),),),
    ]
    assert all(
        {
            "predictive_cache",
            "current_grid_cache",
            "predictive_term_weight",
            "optimize_official_policy_loss",
        }.isdisjoint(names)
        for names in observed_argument_names
    )
    assert report["provenance"]["objective"]["predictive_queries"] == "absent"


def test_full_runner_rejects_zero_weight_required_families(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.predictive_weight = 0.0
    with pytest.raises(ValueError, match="predictive_weight must be positive"):
        _validate_paths_and_args(args)

    args.predictive_weight = 0.004
    args.structural_weight = 0.0
    with pytest.raises(ValueError, match="structural_weight must be positive"):
        _validate_paths_and_args(args)


def test_full_runner_requires_categorical_calvin_ownership_without_dead_support_weight(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    _validate_paths_and_args(args)
    args.support_weight = 1.0
    with pytest.raises(ValueError, match="exclusive ownership requires support_weight=0"):
        _validate_paths_and_args(args)


def test_family_gradient_diagnostics_measure_norms_and_conflicts() -> None:
    class Dist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op == "sum"

        @staticmethod
        def get_world_size() -> int:
            return 1

    result = _distributed_family_gradient_diagnostics(
        family_gradients={
            "action": torch.tensor([2.0, 4.0]),
            "predictive": torch.tensor([1.0, 1.0]),
            "structural": torch.tensor([-1.0, 1.0]),
        },
        device=torch.device("cpu"),
        dist=Dist,
        torch_module=torch,
    )
    assert result["probe"] == "picf_native_graph.object_queries"
    assert result["gradient_norms"] == pytest.approx(
        {"action": 20**0.5, "predictive": 2**0.5, "structural": 2**0.5}
    )
    assert result["cosines"]["predictive__structural"] == pytest.approx(0.0)

    shared = _distributed_family_gradient_diagnostics(
        family_gradients={
            "action": torch.zeros(2),
            "predictive": torch.tensor([1.0, 1.0]),
            "structural": torch.tensor([-1.0, 1.0]),
        },
        probe=full_runner.REPRESENTATION_FAMILY_GRADIENT_PROBE,
        device=torch.device("cpu"),
        dist=Dist,
        torch_module=torch,
    )
    assert shared["probe"] == full_runner.REPRESENTATION_FAMILY_GRADIENT_PROBE
    assert shared["gradient_norms"]["action"] == 0.0


def test_family_gradient_diagnostics_reject_inconsistent_local_shards() -> None:
    class Dist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op == "sum"

        @staticmethod
        def get_world_size() -> int:
            return 1

    with pytest.raises(RuntimeError, match="inconsistent local shapes"):
        _distributed_family_gradient_diagnostics(
            family_gradients={
                "action": torch.ones(2),
                "predictive": torch.ones(3),
                "structural": torch.ones(2),
            },
            device=torch.device("cpu"),
            dist=Dist,
            torch_module=torch,
        )


def test_representation_family_gradient_probe_uses_one_shared_host_surface() -> None:
    early = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    middle = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    late = torch.nn.Parameter(torch.tensor([5.0, 6.0]))
    loss = early.sum() + 2.0 * middle.sum() + 3.0 * late.sum()
    loss.backward()

    snapshot = _shared_host_family_gradient_snapshot(
        {
            "early": ("layers.0.input_layernorm.weight", early),
            "middle": ("layers.18.input_layernorm.weight", middle),
            "late": ("layers.35.input_layernorm.weight", late),
        },
        family_name="structural",
        torch_module=torch,
    )

    assert snapshot.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_relation_surface_component_gradients_are_read_only_and_exact() -> None:
    match = torch.tensor([[[0.25, -0.5], [0.5, -0.25]]], requires_grad=True)
    rows = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]], requires_grad=True)
    normalized_terms = {
        "set/task": 2.0 * match.sum(),
        "set/task_dense": -4.0 * match.sum() + 5.0 * rows.sum(),
        "set/ownership": rows.square().sum(),
    }
    result = SimpleNamespace(
        final_relation=SimpleNamespace(
            match_embeddings=match,
            row_embeddings=rows,
        ),
        objective=SimpleNamespace(
            structural_terms=(
                SimpleNamespace(name="set/task", weight=0.5),
                SimpleNamespace(name="set/task_dense", weight=0.25),
                SimpleNamespace(name="set/ownership", weight=2.0),
            ),
            objective=SimpleNamespace(normalized_terms=normalized_terms),
        ),
    )
    gradients = _relation_surface_component_gradients(result, torch_module=torch)
    assert tuple(gradients) == (
        "task@match_embeddings",
        "task_dense@match_embeddings",
        "task_dense@row_embeddings",
        "ownership@row_embeddings",
    )
    torch.testing.assert_close(gradients["task@match_embeddings"], torch.ones_like(match))
    torch.testing.assert_close(gradients["task_dense@match_embeddings"], -torch.ones_like(match))
    torch.testing.assert_close(gradients["task_dense@row_embeddings"], 1.25 * torch.ones_like(rows))
    torch.testing.assert_close(gradients["ownership@row_embeddings"], 4.0 * rows)
    assert match.grad is None
    assert rows.grad is None


def test_factorized_relation_surface_gradients_keep_task_and_ownership_separate() -> None:
    match = torch.tensor([[[0.25, -0.5], [0.5, -0.25]]], requires_grad=True)
    rows = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]], requires_grad=True)
    normalized_terms = {
        "set/task_row": 2.0 * match.sum(),
        "set/ownership": rows.square().sum(),
    }
    result = SimpleNamespace(
        final_relation=SimpleNamespace(
            match_embeddings=match,
            row_embeddings=rows,
        ),
        objective=SimpleNamespace(
            structural_terms=(
                SimpleNamespace(name="set/task_row", weight=0.5),
                SimpleNamespace(name="set/task_dense", weight=0.0),
                SimpleNamespace(name="set/ownership", weight=2.0),
            ),
            objective=SimpleNamespace(normalized_terms=normalized_terms),
        ),
    )
    gradients = _relation_surface_component_gradients(result, torch_module=torch)
    assert tuple(gradients) == (
        "task_row@match_embeddings",
        "ownership@row_embeddings",
    )
    torch.testing.assert_close(gradients["task_row@match_embeddings"], torch.ones_like(match))
    torch.testing.assert_close(gradients["ownership@row_embeddings"], 4.0 * rows)
    assert match.grad is None
    assert rows.grad is None


def test_factorized_relation_surface_gradients_reject_a_disconnected_task_path() -> None:
    match = torch.randn(1, 2, 4, requires_grad=True)
    rows = torch.randn(1, 2, 4, requires_grad=True)
    disconnected = torch.tensor(1.0, requires_grad=True)
    result = SimpleNamespace(
        final_relation=SimpleNamespace(
            match_embeddings=match,
            row_embeddings=rows,
        ),
        objective=SimpleNamespace(
            structural_terms=(
                SimpleNamespace(name="set/task_row", weight=1.0),
                SimpleNamespace(name="set/task_dense", weight=0.0),
                SimpleNamespace(name="set/ownership", weight=1.0),
            ),
            objective=SimpleNamespace(
                normalized_terms={
                    "set/task_row": disconnected,
                    "set/ownership": rows.square().mean(),
                }
            ),
        ),
    )

    with pytest.raises(RuntimeError):
        _relation_surface_component_gradients(result, torch_module=torch)


def test_relation_surface_gradient_diagnostics_measure_component_conflict() -> None:
    class Dist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op == "sum"

        @staticmethod
        def get_world_size() -> int:
            return 1

    result = _distributed_relation_surface_gradient_diagnostics(
        component_gradients={
            "task@match_embeddings": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "task_dense@match_embeddings": torch.tensor([-1.0, -1.0, -1.0, -1.0]),
            "task_dense@row_embeddings": torch.tensor([1.5, 1.5, 1.5, 1.5]),
            "ownership@row_embeddings": torch.tensor([4.0, -4.0, 8.0, -8.0]),
        },
        device=torch.device("cpu"),
        dist=Dist,
        torch_module=torch,
    )
    assert result["probe"] == "final_relation.match_embeddings+row_embeddings"
    assert result["gradient_elements"] == {
        "task@match_embeddings": 4,
        "task_dense@match_embeddings": 4,
        "task_dense@row_embeddings": 4,
        "ownership@row_embeddings": 4,
    }
    assert result["cosines"]["task__task_dense@match_embeddings"] == pytest.approx(-1.0)
    assert result["cosines"]["task_dense__ownership@row_embeddings"] == pytest.approx(0.0)


def test_factorized_relation_surface_diagnostics_have_no_fabricated_cross_surface_pair() -> None:
    class Dist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op == "sum"

        @staticmethod
        def get_world_size() -> int:
            return 1

    result = _distributed_relation_surface_gradient_diagnostics(
        component_gradients={
            "task_row@match_embeddings": torch.ones(4),
            "ownership@row_embeddings": torch.tensor([1.0, -1.0, 1.0, -1.0]),
        },
        device=torch.device("cpu"),
        dist=Dist,
        torch_module=torch,
    )
    assert result["gradient_elements"] == {
        "task_row@match_embeddings": 4,
        "ownership@row_embeddings": 4,
    }
    assert result["cosines"] == {}
    assert result["dot_products"] == {}


def test_parameter_gradient_snapshot_reads_only_completed_backward() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    with pytest.raises(RuntimeError, match="did not reach"):
        _parameter_gradient_snapshot(parameter, name="unit", torch_module=torch)
    parameter.square().sum().backward()
    snapshot = _parameter_gradient_snapshot(parameter, name="unit", torch_module=torch)
    assert torch.equal(snapshot, torch.tensor([2.0, 4.0]))
    parameter.grad.zero_()
    assert torch.equal(snapshot, torch.tensor([2.0, 4.0]))


@pytest.mark.parametrize("selected_name", ("action", "predictive", "structural"))
def test_isolated_family_backward_is_exact_and_traverses_every_root(selected_name: str) -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.5, -1.5]))
    family_terms = {
        "action": (2.0 * parameter).sum(),
        "predictive": parameter.square().sum(),
        "structural": parameter.pow(3).sum(),
    }
    observed_cotangents: dict[str, torch.Tensor] = {}
    for name, term in family_terms.items():
        term.register_hook(
            lambda gradient, name=name: observed_cotangents.setdefault(
                name, gradient.detach().clone()
            )
        )

    _backward_isolated_objective_family(
        family_terms,
        selected_name=selected_name,
        torch_module=torch,
    )

    expected = {
        "action": torch.full_like(parameter, 2.0),
        "predictive": 2.0 * parameter.detach(),
        "structural": 3.0 * parameter.detach().square(),
    }
    assert torch.equal(parameter.grad, expected[selected_name])
    assert set(observed_cotangents) == set(family_terms)
    for name, cotangent in observed_cotangents.items():
        assert torch.equal(
            cotangent,
            torch.ones_like(cotangent) if name == selected_name else torch.zeros_like(cotangent),
        )


def test_behavior_posterior_credit_uses_exact_weighted_optimizer_contribution() -> None:
    behavior = torch.tensor(2.0, requires_grad=True)
    valid = torch.ones(1, dtype=torch.bool)
    behavior_term = SimpleNamespace(
        name="rollout/vision/binding",
        weight=3.0,
        valid=valid,
        sample_weight=None,
    )
    source_term = SimpleNamespace(
        name="xmod/vision/representation",
        weight=1.0,
        valid=valid,
        sample_weight=None,
    )
    result = SimpleNamespace(
        objective=SimpleNamespace(
            predictive_terms=(behavior_term, source_term),
            objective=SimpleNamespace(
                normalized_terms={
                    "rollout/vision/binding": behavior,
                    "xmod/vision/representation": torch.tensor(1.0),
                }
            ),
        )
    )

    contribution = _weighted_behavior_future_contribution(
        result,
        predictive_family_weight=0.004,
        torch_module=torch,
    )
    assert contribution.item() == pytest.approx(0.006)
    assert torch.autograd.grad(contribution, behavior)[0].item() == pytest.approx(0.003)


def test_predictive_host_gradient_probe_covers_early_middle_and_late_layers() -> None:
    class Node(torch.nn.Module):
        pass

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(2560, elementwise_affine=True)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.input_layernorm(value)

    class Dist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op == "sum"

        @staticmethod
        def get_world_size() -> int:
            return 1

    policy = Node()
    policy.model = Node()
    policy.model.qwenvl_with_expert = Node()
    policy.model.qwenvl_with_expert.qwenvl = Node()
    policy.model.qwenvl_with_expert.qwenvl.model = Node()
    language_model = Node()
    language_model.layers = torch.nn.ModuleList(Layer() for _ in range(36))
    policy.model.qwenvl_with_expert.qwenvl.model.language_model = language_model
    selected = _predictive_host_gradient_parameters(policy)
    assert set(selected) == {"early", "middle", "late"}
    loss = sum(parameter.square().sum() for _name, parameter in selected.values())
    loss.backward()
    result = _distributed_predictive_host_gradient_diagnostics(
        host_gradients={
            depth: (
                name,
                _parameter_gradient_snapshot(
                    parameter,
                    name=f"predictive {depth}",
                    torch_module=torch,
                ),
            )
            for depth, (name, parameter) in selected.items()
        },
        device=torch.device("cpu"),
        dist=Dist,
        torch_module=torch,
    )
    assert result["all_finite"] is True
    assert result["decomposition"] is None
    assert all(value > 0 for value in result["gradient_norms"].values())
    assert all(value == 2560 for value in result["gradient_elements"].values())

    total = {
        depth: _parameter_gradient_snapshot(
            parameter,
            name=f"behavior total {depth}",
            torch_module=torch,
        )
        for depth, (_name, parameter) in selected.items()
    }
    via = {depth: value * 0.25 for depth, value in total.items()}
    direct = {depth: total[depth] - via[depth] for depth in total}
    decomposed = _distributed_predictive_host_gradient_diagnostics(
        host_gradients={depth: (selected[depth][0], via[depth]) for depth in selected},
        decomposition_gradients={
            "total": total,
            "via_posterior": via,
            "direct": direct,
        },
        probe="lingbot.language_model.input_layernorm.via_primary_posterior_vjp",
        device=torch.device("cpu"),
        dist=Dist,
        torch_module=torch,
    )
    assert decomposed["decomposition"]["depths"]["early"][
        "via_to_total_norm_ratio"
    ] == pytest.approx(0.25)

    with pytest.raises(RuntimeError, match="did not reach"):
        _distributed_predictive_host_gradient_diagnostics(
            host_gradients={
                depth: (name, torch.zeros_like(parameter))
                for depth, (name, parameter) in selected.items()
            },
            device=torch.device("cpu"),
            dist=Dist,
            torch_module=torch,
        )

    language_model.layers[18].input_layernorm.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen parameter"):
        _predictive_host_gradient_parameters(policy)


def test_full_runner_requires_authorization_and_visuals_for_multi_step(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.phase = "resume"
    args.load_global_step = 1
    with pytest.raises(ValueError, match="authorization-manifest"):
        _validate_paths_and_args(args)

    args.invocation_steps = 2
    with pytest.raises(ValueError, match="authorization-manifest"):
        _validate_paths_and_args(args)

    args.authorization_manifest = tmp_path / "authorization.json"
    args.authorization_manifest.write_text("{}")
    args.authorization_manifest_sha256 = _sha(args.authorization_manifest.read_bytes())
    with pytest.raises(ValueError, match="visual audit"):
        _validate_paths_and_args(args)
    args.visual_audit_every = 1
    _validate_paths_and_args(args)


def test_full_runner_rejects_future_invocation_outputs_before_model_load(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    args.phase = "resume"
    args.load_global_step = 1
    args.invocation_steps = 4
    args.authorization_manifest = tmp_path / "authorization.json"
    args.authorization_manifest.write_text("{}")
    args.authorization_manifest_sha256 = _sha(args.authorization_manifest.read_bytes())
    args.visual_audit_every = 1

    accepted_visual = args.run_dir / "visuals/step_00000001/rank_0/accepted.png"
    accepted_visual.parent.mkdir(parents=True)
    accepted_visual.write_bytes(b"accepted")
    _validate_paths_and_args(args)

    orphaned_visual = args.run_dir / "visuals/step_00000002/rank_0/orphaned.png"
    orphaned_visual.parent.mkdir(parents=True)
    orphaned_visual.write_bytes(b"orphaned")
    with pytest.raises(FileExistsError, match="pre-existing output artifacts"):
        _validate_paths_and_args(args)
    orphaned_visual.unlink()
    orphaned_visual.parent.rmdir()
    orphaned_visual.parent.parent.rmdir()

    output_checkpoint = args.run_dir / "checkpoints/global_step_5"
    output_checkpoint.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="global_step_5"):
        _validate_paths_and_args(args)


def test_full_runner_progress_events_are_compact_rank_and_step_bound(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for event in (
        "gradient_audit_replay_started",
        "objective_started",
        "objective_ready",
        "backward_started",
        "backward_completed",
    ):
        _emit_step_progress(event, rank=0, global_step=6)
        assert json.loads(capsys.readouterr().out) == {
            "event": f"native_full_{event}",
            "global_step": 6,
            "rank": 0,
        }
    _emit_step_progress(
        "step_completed",
        rank=1,
        global_step=7,
        details={"official_action_loss": 0.25},
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "event": "native_full_step_completed",
        "global_step": 7,
        "official_action_loss": 0.25,
        "rank": 1,
    }
    with pytest.raises(ValueError, match="unsupported"):
        _emit_step_progress("unknown", rank=0, global_step=1)
    with pytest.raises(ValueError, match="reserved"):
        _emit_step_progress(
            "step_completed",
            rank=0,
            global_step=1,
            details={"rank": 1},
        )


def test_full_runner_every_literal_progress_event_is_allowlisted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    events = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_emit_step_progress"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }
    assert "gradient_audit_replay_started" in events
    for event in sorted(events):
        _emit_step_progress(event, rank=0, global_step=1)
        assert json.loads(capsys.readouterr().out)["event"] == f"native_full_{event}"


def test_full_runner_trims_cuda_cache_only_after_gradient_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("trim"))

    _trim_cuda_allocator_after_gradient_audit(
        gradient_audit=False,
        torch_module=torch,
    )
    assert calls == []
    _trim_cuda_allocator_after_gradient_audit(
        gradient_audit=True,
        torch_module=torch,
    )
    assert calls == ["trim"]
    with pytest.raises(TypeError, match="boolean"):
        _trim_cuda_allocator_after_gradient_audit(
            gradient_audit=1,  # type: ignore[arg-type]
            torch_module=torch,
        )


def test_full_runner_releases_every_graph_owner_before_allocator_trim() -> None:
    module = ast.parse(TOOL.read_text(encoding="utf-8"))
    main = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    deleted_names = {
        name.id
        for node in ast.walk(main)
        if isinstance(node, ast.Delete)
        for target in node.targets
        for name in ast.walk(target)
        if isinstance(name, ast.Name)
    }
    assert {
        "audit_rng_state",
        "build_overshoot",
        "correction_branch",
        "correction_target",
        "counterfactual_diagnostics",
        "counterfactual_predictions",
        "family_gradients",
        "full_batches",
        "host_gradients",
        "objective",
        "planned",
        "posterior",
        "primary_batch",
        "relation",
        "result",
        "routing_bias",
        "run_step_objective",
    } <= deleted_names

    source = TOOL.read_text(encoding="utf-8")
    graph_release = source.index("del (", source.index("completed = step_reports[-1]"))
    allocator_trim = source.index(
        "_trim_cuda_allocator_after_gradient_audit(",
        graph_release,
    )
    progress = source.index(
        '_emit_step_progress(\n                "step_completed"', allocator_trim
    )
    assert graph_release < allocator_trim < progress


def test_visual_audit_cadence_does_not_change_model_or_resume_contract(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    kwargs = {
        "root": ROOT,
        "args": args,
        "patched_source_sha256": {"model.py": "a" * 64},
        "predictive_report": {"cache_manifest_sha256": "b" * 64},
        "current_grid_report": {"cache_manifest_sha256": "c" * 64},
        "query_schema_sha256": {
            "controlled_future_rollout": "d" * 64,
            "current_correction": current_correction_summary_query_schema_digest(
                route_id=0,
                address_width=2,
            ),
            "current_random_grid": current_grid_query_schema_digest(route_id=0),
            "omitted_static": omitted_static_summary_query_schema_digest(route_id=0),
        },
        "temporal_metadata": {"schema": "temporal"},
        "optimizer_contract": {"algorithm": "official-muon"},
    }
    first = _execution_contract_digest(**kwargs)
    args.visual_audit_every = 200
    assert _execution_contract_digest(**kwargs) == first

    args.relation_supervision_layers = (8, 17, 26)
    multi_depth = _execution_contract_digest(**kwargs)
    assert multi_depth[0] != first[0]
    assert multi_depth[1] == first[1]
    args.relation_supervision_layers = ()

    args.task_relation_estimator = GLOBAL_MULTIPOSITIVE_TASK_RELATION
    assert _execution_contract_digest(**kwargs) != first
    args.task_relation_estimator = HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION
    host_native = _execution_contract_digest(**kwargs)
    assert host_native != first
    args.task_relation_estimator = LOCAL_BALANCED_TASK_RELATION
    assert _execution_contract_digest(**kwargs) == first

    args.fsdp2_placement = "gpu-sharded"
    assert _execution_contract_digest(**kwargs) != first
    args.fsdp2_placement = "cpu-offload"

    args.cuda_allocator = "expandable-segments"
    assert _execution_contract_digest(**kwargs) != first
    args.cuda_allocator = "native"

    args.lane_interleave_factor = 8
    assert _execution_contract_digest(**kwargs) != first
    args.lane_interleave_factor = 1

    behavior_evidence = tmp_path / "behavior-g0.json"
    behavior_evidence.write_text("first", encoding="ascii")
    representation_split = tmp_path / "behavior-representation-split.json"
    _representation_split(
        representation_split,
        training_steps=args.total_planned_steps,
        reference_evaluation=True,
    )
    args.training_stage = "representation"
    args.representation_split = representation_split
    args.representation_split_sha256 = _sha(representation_split.read_bytes())
    args.behavior_conditioned_prediction = True
    args.behavior_causal_probe_evidence = behavior_evidence
    args.behavior_causal_probe_evidence_sha256 = _sha(behavior_evidence.read_bytes())
    kwargs["behavior_graph_sha256"] = _behavior_graph_digest(args)
    behavior = _execution_contract_digest(**kwargs)
    assert behavior != first
    behavior_evidence.write_text("second", encoding="ascii")
    assert _execution_contract_digest(**kwargs) == behavior
    args.behavior_causal_probe_evidence_sha256 = _sha(behavior_evidence.read_bytes())
    assert _execution_contract_digest(**kwargs) != behavior
    args.behavior_conditioned_prediction = False
    args.behavior_causal_probe_evidence = None
    args.behavior_causal_probe_evidence_sha256 = None
    args.training_stage = "action"
    args.representation_split = None
    args.representation_split_sha256 = None
    kwargs["behavior_graph_sha256"] = None

    intervention = tmp_path / "task-intervention.json"
    intervention.write_text("{}", encoding="ascii")
    args.representation_task_intervention_plan = intervention
    assert _execution_contract_digest(**kwargs) != first
    args.representation_task_intervention_plan = None

    fixed_files = []
    for name in (
        "fixed_observation_pair_plan",
        "fixed_observation_training_audit",
        "fixed_observation_evaluation_plan",
        "fixed_observation_validation_audit",
        "fixed_observation_heldout_audit",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="ascii")
        setattr(args, name, path)
        fixed_files.append((name, path))
    fixed = _execution_contract_digest(**kwargs)
    assert fixed != first
    name, path = fixed_files[-1]
    path.write_text(f"{name}-changed", encoding="ascii")
    assert _execution_contract_digest(**kwargs) != fixed
    for name, _path in fixed_files:
        setattr(args, name, None)

    changed = dict(kwargs)
    changed["query_schema_sha256"] = {
        **kwargs["query_schema_sha256"],
        "current_correction": "e" * 64,
    }
    assert _execution_contract_digest(**changed) != first


def test_execution_contract_rejects_missing_or_malformed_query_schemas(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    kwargs = {
        "root": ROOT,
        "args": args,
        "patched_source_sha256": {"model.py": "a" * 64},
        "predictive_report": {"cache_manifest_sha256": "b" * 64},
        "current_grid_report": {"cache_manifest_sha256": "c" * 64},
        "query_schema_sha256": {
            "controlled_future_rollout": "d" * 64,
            "current_correction": "e" * 64,
            "current_random_grid": "f" * 64,
        },
        "temporal_metadata": {"schema": "temporal"},
        "optimizer_contract": {"algorithm": "official-muon"},
    }
    with pytest.raises(RuntimeError, match="query schemas differ"):
        _execution_contract_digest(**kwargs)

    kwargs["query_schema_sha256"] = {
        **kwargs["query_schema_sha256"],
        "omitted_static": "not-a-digest",
    }
    with pytest.raises(ValueError, match="omitted_static query schema"):
        _execution_contract_digest(**kwargs)


def test_training_authorization_is_hash_bound_and_requires_ordered_passes(
    tmp_path: Path,
    full_objective_report_factory,
    g0_report_factory,
    preflight_report_factory,
    smoke_report_factory,
) -> None:
    digest = "a" * 64
    input_report = full_objective_report_factory(
        tmp_path / "native_full_step_1.json",
        digest=digest,
    )
    reports = []
    criteria = tmp_path / "criteria.md"
    criteria.write_text("frozen criteria")
    for gate in ("G0", "G1", "G2_PROTOCOL"):
        path = tmp_path / f"{gate}.decision.json"
        path.write_text(
            json.dumps(
                build_training_gate_decision(
                    gate=gate,
                    reviewer="local-test",
                    criteria=criteria,
                    evidence=_gate_evidence(
                        tmp_path,
                        gate,
                        g0_report_factory,
                        preflight_report_factory,
                        smoke_report_factory,
                    ),
                ),
                sort_keys=True,
            )
        )
        reports.append({"gate": gate, "path": str(path), "sha256": _sha(path.read_bytes())})
    authorization = {
        "schema": TRAINING_AUTHORIZATION_SCHEMA,
        "status": "PASS",
        "stage": "pilot",
        "input_global_step": 1,
        "maximum_global_step": 200,
        "visual_audit_every": 20,
        "execution_contract_sha256": digest,
        "implementation_sha256": digest,
        "model_family_sha256": digest,
        "predictive_objective": IMPLEMENTED_PREDICTIVE_OBJECTIVE,
        "predictive_claim_scope": PREDICTIVE_OBJECTIVE_CLAIMS[IMPLEMENTED_PREDICTIVE_OBJECTIVE],
        "predictive_visible_support_weighting": (IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING),
        "predictive_minimum_visible_fraction_hex": (0.0).hex(),
        "acceptance_subject": None,
        "input_full_report_sha256": _sha(input_report.read_bytes()),
        "input_full_report": {
            "path": str(input_report),
            "sha256": _sha(input_report.read_bytes()),
        },
        "prerequisite_reports": reports,
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(authorization, sort_keys=True))
    kwargs = {
        "expected_sha256": _sha(path.read_bytes()),
        "input_global_step": 1,
        "requested_global_step": 120,
        "total_planned_steps": 30_000,
        "visual_audit_every": 20,
        "execution_contract_sha256": digest,
        "implementation_sha256": digest,
        "model_family_sha256": digest,
    }
    assert validate_training_authorization(path, **kwargs) == authorization

    reports[0], reports[1] = reports[1], reports[0]
    path.write_text(json.dumps(authorization, sort_keys=True))
    kwargs["expected_sha256"] = _sha(path.read_bytes())
    with pytest.raises(ValueError, match="order differs"):
        validate_training_authorization(path, **kwargs)

    reports[0], reports[1] = reports[1], reports[0]
    reports[0]["path"] = str(input_report)
    reports[0]["sha256"] = _sha(input_report.read_bytes())
    path.write_text(json.dumps(authorization, sort_keys=True))
    kwargs["expected_sha256"] = _sha(path.read_bytes())
    with pytest.raises(ValueError, match="report differs"):
        validate_training_authorization(path, **kwargs)


def test_long_authorization_requires_executed_source_branch_on_every_rank(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "d" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest)
    value = json.loads(path.read_text())
    step = value["rank_reports"][1]["steps"][0]
    step["source_masked_branch"] = False
    step["source_prediction_mode"] = None
    step["omitted_static_digest"] = None
    step["normalized_terms"].pop("xmod/vision/representation")
    step["valid_counts"].pop("xmod/vision/representation")
    step["objective_total"] = 0.21036
    path.write_text(json.dumps(value, sort_keys=True))
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(json.dumps(value, sort_keys=True))

    with pytest.raises(ValueError, match="on every rank"):
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_source_evidence=True,
        )


@pytest.mark.parametrize("input_step", [0, 1])
def test_representation_report_accepts_exact_action_isolation(
    tmp_path: Path,
    representation_objective_report_factory,
    input_step: int,
) -> None:
    digest = "e" * 64
    path = representation_objective_report_factory(
        tmp_path / f"representation-{input_step}.json",
        digest=digest,
        input_step=input_step,
    )
    value = json.loads(path.read_text())

    assert (
        validate_representation_objective_report(
            value,
            require_initial_probe=input_step == 0,
            expected_digests={
                "representation_split_sha256": digest,
                "representation_split_file_sha256": digest,
                "representation_parameter_scope_sha256": value[
                    "representation_parameter_scope_sha256"
                ],
                "representation_frozen_action_state_sha256": digest,
            },
        )["status"]
        == "PASS"
    )


def test_representation_report_binds_shared_host_family_gradient_probe(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-shared-host.json",
        digest="e" * 64,
        input_step=1,
    )
    value = json.loads(path.read_text())

    stale = deepcopy(value)
    stale["rank_reports"][0]["steps"][0]["family_gradient_diagnostics"]["probe"] = (
        "picf_native_graph.object_queries"
    )
    with pytest.raises(ValueError, match="probe contract"):
        validate_representation_objective_report(
            stale,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )

    disconnected = deepcopy(value)
    disconnected["rank_reports"][0]["steps"][0]["family_gradient_diagnostics"]["gradient_norms"][
        "structural"
    ] = 0.0
    with pytest.raises(ValueError, match="structural family-gradient norm"):
        validate_representation_objective_report(
            disconnected,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_behavior_representation_report_binds_interval_conditioning_and_gradient_paths(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "behavior-representation.json",
        digest="e" * 64,
        input_step=1,
        behavior=True,
    )
    value = json.loads(path.read_text())
    conditioning_sha256 = full_runner._canonical_digest(value["behavior_conditioning"])
    validate_representation_objective_report(
        value,
        require_initial_probe=False,
        expected_behavior_conditioning_sha256=conditioning_sha256,
    )

    changed = deepcopy(value)
    changed["input_global_step"] = 2
    changed["saved_global_step"] = 3
    with pytest.raises(ValueError, match="bounded G1 interval"):
        validate_representation_objective_report(
            changed,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )

    changed = deepcopy(value)
    changed["behavior_conditioning"]["behavior_graph_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="conditioning contract differs"):
        validate_representation_objective_report(
            changed,
            require_initial_probe=False,
            require_checkpoint_copy=False,
            expected_behavior_conditioning_sha256=conditioning_sha256,
        )

    changed = deepcopy(value)
    changed["rank_reports"][0]["steps"][0]["predictive_host_gradient_diagnostics"]["decomposition"][
        "depths"
    ]["early"]["closure_error_norm"] = 2.0
    with pytest.raises(ValueError, match="does not close"):
        validate_representation_objective_report(
            changed,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_representation_report_distinguishes_non_resumable_evidence(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    digest = "a" * 64
    path = representation_objective_report_factory(
        tmp_path / "representation-no-checkpoint.json",
        digest=digest,
    )
    value = json.loads(path.read_text())
    checkpoint_dir = Path(value["checkpoint_dir"])
    (checkpoint_dir / "native_representation_report.json").unlink()
    checkpoint_dir.rmdir()
    value["checkpoint_publication"] = "never"
    value["full_shard"] = False
    for rank_report in value["rank_reports"]:
        rank_report["saved_boundary_sha256"] = None

    validate_representation_objective_report(
        value,
        require_initial_probe=False,
        require_checkpoint_copy=False,
        expected_checkpoint_publication="never",
    )
    with pytest.raises(ValueError, match="is not a checkpoint"):
        validate_representation_objective_report(
            value,
            require_initial_probe=False,
            expected_checkpoint_publication="never",
        )
    with pytest.raises(ValueError, match="mode differs"):
        validate_representation_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_representation_report_binds_global_task_relation_estimator(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-global-task-relation.json",
        digest="e" * 64,
    )
    value = json.loads(path.read_text())
    value["objective_contract"]["task_relation_estimator"] = GLOBAL_MULTIPOSITIVE_TASK_RELATION
    assert (
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )

    value["objective_contract"]["task_relation_estimator"] = HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION
    assert (
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )

    value["objective_contract"]["task_relation_estimator"] = "unbound"
    with pytest.raises(ValueError, match="task relation estimator"):
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )


def test_representation_report_binds_entity_conditional_ownership_contract(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-entity-ownership.json",
        digest="4" * 64,
    )
    value = json.loads(path.read_text())
    value["objective_contract"]["task_relation_estimator"] = LOCAL_BALANCED_TASK_RELATION
    value["objective_contract"]["ownership_estimator"] = TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["normalized_terms"]["set/ownership"] = 0.07
        step["normalized_terms"]["set/ownership_nll"] = 0.07
        step["normalized_terms"]["set/ownership_entity"] = 0.11
        step["valid_counts"]["set/ownership_entity"] = 2

    assert (
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )

    value["rank_reports"][0]["steps"][0]["normalized_terms"].pop("set/ownership_entity")
    value["rank_reports"][0]["steps"][0]["valid_counts"].pop("set/ownership_entity")
    with pytest.raises(ValueError, match="structural term schema differs"):
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )


def test_representation_report_accepts_factorized_relation_gradient_contract(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-joint-task-relation.json",
        digest="e" * 64,
        input_step=1,
        behavior=True,
    )
    value = json.loads(path.read_text())
    value["objective_contract"]["task_relation_estimator"] = HOST_NATIVE_FACTORIZED_TASK_RELATION
    value["objective_contract"]["dense_task_weight"] = 0.0
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["normalized_terms"]["set/task_row"] = step["normalized_terms"].pop("set/task")
        step["valid_counts"]["set/task_row"] = step["valid_counts"].pop("set/task")
        step["relation_surface_gradient_diagnostics"] = {
            "all_finite": True,
            "cosines": {},
            "dot_products": {},
            "gradient_elements": {
                "task_row@match_embeddings": 81920,
                "ownership@row_embeddings": 81920,
            },
            "gradient_norms": {
                "task_row@match_embeddings": 1.0,
                "ownership@row_embeddings": 1.5,
            },
            "probe": "final_relation.match_embeddings+row_embeddings",
            "world_size": FULL_WORLD_SIZE,
        }

    validate_representation_objective_report(
        value,
        require_initial_probe=False,
        require_checkpoint_copy=False,
    )

    changed = deepcopy(value)
    changed["rank_reports"][0]["steps"][0]["gradient_metrics"]["match_projection_norm"] = 0.0
    with pytest.raises(ValueError, match="match_projection gradient norm"):
        validate_representation_objective_report(
            changed,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )

    changed = deepcopy(value)
    for rank_report in changed["rank_reports"]:
        step = rank_report["steps"][0]
        step["normalized_terms"]["set/task"] = step["normalized_terms"].pop("set/task_row")
        step["valid_counts"]["set/task"] = step["valid_counts"].pop("set/task_row")
    with pytest.raises(ValueError, match="structural term schema differs"):
        validate_representation_objective_report(
            changed,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_adr135_representation_report_binds_the_registered_profile(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-adr135.json",
        digest="e" * 64,
    )
    value = json.loads(path.read_text())
    value["evidence_profile"] = full_runner.MATCHED_MEDIUM_HORIZON_PROFILE
    value["visual_audit_every"] = full_runner.MATCHED_MEDIUM_HORIZON_VISUAL_CADENCE
    value["gradient_audit_steps"] = list(full_runner.MATCHED_MEDIUM_HORIZON_AUDIT_STEPS)
    value["objective_contract"]["task_relation_estimator"] = HOST_NATIVE_FACTORIZED_TASK_RELATION
    value["objective_contract"]["dense_task_weight"] = 0.0
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["normalized_terms"]["set/task_row"] = step["normalized_terms"].pop("set/task")
        step["valid_counts"]["set/task_row"] = step["valid_counts"].pop("set/task")

    validate_representation_objective_report(
        value,
        require_initial_probe=True,
        require_checkpoint_copy=False,
    )

    changed = deepcopy(value)
    changed["visual_audit_every"] = 100
    with pytest.raises(ValueError, match="medium-horizon representation report"):
        validate_representation_objective_report(
            changed,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )

    changed = deepcopy(value)
    changed["objective_contract"]["task_relation_estimator"] = LOCAL_BALANCED_TASK_RELATION
    with pytest.raises(ValueError, match="dense_task_weight"):
        validate_representation_objective_report(
            changed,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )


def test_factorized_representation_report_rejects_legacy_relation_gradient_schema(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-factorized-with-legacy-gradients.json",
        digest="e" * 64,
        input_step=1,
        behavior=True,
    )
    value = json.loads(path.read_text())
    value["objective_contract"]["task_relation_estimator"] = HOST_NATIVE_FACTORIZED_TASK_RELATION
    value["objective_contract"]["dense_task_weight"] = 0.0
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["normalized_terms"]["set/task_row"] = step["normalized_terms"].pop("set/task")
        step["valid_counts"]["set/task_row"] = step["valid_counts"].pop("set/task")

    with pytest.raises(
        ValueError,
        match="relation-surface gradient schema differs from task relation estimator",
    ):
        validate_representation_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_legacy_full_report_rejects_factorized_relation_gradient_schema(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    path = full_objective_report_factory(
        tmp_path / "full-legacy-with-factorized-gradients.json",
        digest="e" * 64,
        input_step=1,
    )
    value = json.loads(path.read_text())
    for rank_report in value["rank_reports"]:
        rank_report["steps"][0]["relation_surface_gradient_diagnostics"] = {
            "all_finite": True,
            "cosines": {},
            "dot_products": {},
            "gradient_elements": {
                "task_row@match_embeddings": 81920,
                "ownership@row_embeddings": 81920,
            },
            "gradient_norms": {
                "task_row@match_embeddings": 1.0,
                "ownership@row_embeddings": 1.5,
            },
            "probe": "final_relation.match_embeddings+row_embeddings",
            "world_size": FULL_WORLD_SIZE,
        }

    with pytest.raises(
        ValueError,
        match="relation-surface gradient schema differs from task relation estimator",
    ):
        validate_full_objective_report(
            value,
            require_initial_probe=False,
        )


def test_representation_bootstrap_distinguishes_zero_gradient_allocation_from_activity(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-inactive-predictive.json",
        digest="f" * 64,
    )
    value = json.loads(path.read_text())
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["source_masked_branch"] = False
        step["source_prediction_mode"] = None
        step["omitted_static_digest"] = None
        step["normalized_terms"]["xmod/vision/representation"] = 0.0
        step["valid_counts"]["xmod/vision/representation"] = 0
        step["objective_total"] = 0.004 * 0.09
        step["gradient_metrics"]["predictive_readout_norm"] = 0.0
        assert step["gradient_metrics"]["predictive_readout_elements"] > 0

    assert (
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )

    for rank_report in value["rank_reports"]:
        rank_report["steps"][0]["gradient_metrics"]["predictive_readout_norm"] = 1e-6
    with pytest.raises(ValueError, match="inactive predictive family"):
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )


def test_representation_report_allows_zero_valid_sampled_source_branch(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-zero-valid-source.json",
        digest="f" * 64,
        input_step=1,
    )
    value = json.loads(path.read_text())
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["normalized_terms"]["xmod/vision/representation"] = 0.0
        step["valid_counts"]["xmod/vision/representation"] = 0
        # The active correction term is 0.1 and the structural family is 0.09.
        step["objective_total"] = 0.004 * 0.1 + 0.004 * 0.09

    assert (
        validate_representation_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )


def test_full_report_zero_valid_source_does_not_satisfy_long_source_evidence(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    path = full_objective_report_factory(
        tmp_path / "full-zero-valid-source.json",
        digest="f" * 64,
        input_step=0,
    )
    value = json.loads(path.read_text())
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["normalized_terms"]["xmod/vision/representation"] = 0.0
        step["valid_counts"]["xmod/vision/representation"] = 0
        step["objective_total"] = 0.21 + 0.004 * 0.09
        step["gradient_metrics"]["predictive_readout_norm"] = 0.0

    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_source_evidence=False,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )
    with pytest.raises(ValueError, match="lacks executed source-branch evidence"):
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_source_evidence=True,
            require_checkpoint_copy=False,
        )


@pytest.mark.parametrize(
    ("field", "measured", "message"),
    [
        ("official_action_loss", 0.1, "official action loss"),
        ("action_output_norm", 0.1, "frozen action output"),
    ],
)
def test_representation_report_rejects_action_leakage(
    tmp_path: Path,
    representation_objective_report_factory,
    field: str,
    measured: float,
    message: str,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / f"representation-{field}.json",
        digest="a" * 64,
    )
    value = json.loads(path.read_text())
    step = value["rank_reports"][0]["steps"][0]
    if field == "action_output_norm":
        step["gradient_metrics"][field] = measured
    else:
        step[field] = measured

    with pytest.raises(ValueError, match=message):
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )


def test_representation_report_binds_split_and_parameter_scope(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    digest = "b" * 64
    path = representation_objective_report_factory(
        tmp_path / "representation.json",
        digest=digest,
    )
    value = json.loads(path.read_text())

    with pytest.raises(ValueError, match="another implementation or model"):
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
            expected_digests={"representation_split_sha256": "c" * 64},
        )

    value["representation_parameter_scope"]["action_frozen_numel"] += 1
    with pytest.raises(ValueError, match="parameter scope is malformed"):
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )


def test_representation_checkpoint_requires_stage_specific_report_copy(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation.json",
        digest="d" * 64,
    )
    value = json.loads(path.read_text())
    checkpoint_dir = Path(value["checkpoint_dir"])
    representation_copy = checkpoint_dir / "native_representation_report.json"
    representation_copy.rename(checkpoint_dir / "native_full_report.json")

    with pytest.raises(ValueError, match="immutable report copy"):
        validate_representation_objective_report(
            value,
            require_initial_probe=True,
        )


def test_full_report_rejects_cross_rank_sample_overlap(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "8" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest, input_step=1)
    value = json.loads(path.read_text())
    value["rank_reports"][1]["steps"][0]["sample_keys"] = ["sample-0"]
    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)

    with pytest.raises(ValueError, match="ranks consumed overlapping samples"):
        validate_full_objective_report(value, require_initial_probe=False)


def _fixed_observation_report_fingerprint(*, language: str, task: str) -> dict[str, object]:
    return {
        "batch_size": 1,
        "controls_sha256": "1" * 64,
        "language_masks_sha256": "2" * 64,
        "language_tokens_sha256": language * 64,
        "modalities_sha256": None,
        "non_language_model_inputs_sha256": "3" * 64,
        "routing_source_sha256": "4" * 64,
        "schema": "picf-next.fixed-observation-training-pair-fingerprint.v1",
        "structural_source_sha256": "5" * 64,
        "task_keys": [task],
    }


def test_representation_report_allows_only_proven_fixed_observation_overlap(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-fixed-observation.json",
        digest="8" * 64,
        input_step=1,
    )
    value = json.loads(path.read_text())
    first = value["rank_reports"][0]["steps"][0]
    second = value["rank_reports"][1]["steps"][0]
    second["sample_keys"] = list(first["sample_keys"])

    with pytest.raises(ValueError, match="ranks consumed overlapping samples"):
        validate_representation_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )

    pair_sha256 = "6" * 64
    first["fixed_observation_pair_sha256"] = pair_sha256
    second["fixed_observation_pair_sha256"] = pair_sha256
    first["fixed_observation_fingerprint"] = _fixed_observation_report_fingerprint(
        language="a",
        task="lift_blue_block_table",
    )
    second["fixed_observation_fingerprint"] = _fixed_observation_report_fingerprint(
        language="b",
        task="turn_on_led",
    )
    assert (
        validate_representation_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )

    second["fixed_observation_fingerprint"]["controls_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="changed non-language contracts"):
        validate_representation_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_representation_report_rejects_fixed_observation_nonoverlap(
    tmp_path: Path,
    representation_objective_report_factory,
) -> None:
    path = representation_objective_report_factory(
        tmp_path / "representation-fixed-observation-nonoverlap.json",
        digest="9" * 64,
        input_step=1,
    )
    value = json.loads(path.read_text())
    first = value["rank_reports"][0]["steps"][0]
    second = value["rank_reports"][1]["steps"][0]
    pair_sha256 = "6" * 64
    first["fixed_observation_pair_sha256"] = pair_sha256
    second["fixed_observation_pair_sha256"] = pair_sha256
    first["fixed_observation_fingerprint"] = _fixed_observation_report_fingerprint(
        language="a",
        task="lift_blue_block_table",
    )
    second["fixed_observation_fingerprint"] = _fixed_observation_report_fingerprint(
        language="b",
        task="turn_on_led",
    )

    with pytest.raises(ValueError, match="did not consume the same source samples"):
        validate_representation_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_current_report_schema_requires_and_validates_estimator_transactions(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    path = full_objective_report_factory(tmp_path / "full.json", digest="5" * 64)
    value = json.loads(path.read_text())
    reset = deepcopy(value)
    for rank_report in reset["rank_reports"]:
        step = rank_report["steps"][0]
        step["estimator_component"] = "reset"
        step["posterior_committed"] = False
        step["posterior_bank_sha256_after"] = step["posterior_bank_sha256_before"]
    assert (
        validate_full_objective_report(
            reset,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )

    missing = deepcopy(value)
    del missing["rank_reports"][0]["steps"][0]["posterior_bank_sha256_after"]
    with pytest.raises(ValueError, match="step fields differ"):
        validate_full_objective_report(
            missing,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )

    unchanged = deepcopy(value)
    first = unchanged["rank_reports"][0]["steps"][0]
    first["posterior_bank_sha256_after"] = first["posterior_bank_sha256_before"]
    with pytest.raises(ValueError, match="did not publish posterior state"):
        validate_full_objective_report(
            unchanged,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )


def test_fresh_report_allows_only_causal_local_bptt_tail_correction(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    path = full_objective_report_factory(tmp_path / "full.json", digest="6" * 64)
    value = json.loads(path.read_text())
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["local_bptt_steps"] = 2
        step["normalized_terms"]["correction/dino_video"] = 0.1
        step["valid_counts"]["correction/dino_video"] = 1
        step["objective_total"] = 0.21072
    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )

    value["rank_reports"][0]["steps"][0]["local_bptt_steps"] = 1
    with pytest.raises(ValueError, match="fabricated prior-correction support"):
        validate_full_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )


def test_full_report_routing_accepts_native_calvin_producer_types(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "7" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest)
    value = json.loads(path.read_text())
    routing = NativeCALVINRouting(
        lane_ids=(7,),
        episode_keys=("episode-7",),
        frame_indices=(0,),
        reset=(True,),
        sample_keys=("native-sample-7",),
        optimizer_step=0,
    )
    step = value["rank_reports"][0]["steps"][0]
    step["sample_keys"] = list(routing.sample_keys)
    step["lane_ids"] = list(routing.lane_ids)
    step["frame_indices"] = list(routing.frame_indices)
    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )

    step["lane_ids"] = ["lane-7"]
    with pytest.raises(ValueError, match="routing provenance is malformed"):
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_full_report_loss_visual_trial_cannot_seed_long_training(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    path = full_objective_report_factory(tmp_path / "trial.json", digest="a" * 64)
    value = json.loads(path.read_text())
    value["evidence_profile"] = "loss_visual_trial"
    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )
    with pytest.raises(ValueError, match="cannot originate"):
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_source_evidence=True,
            require_checkpoint_copy=False,
        )


def test_full_report_requires_explicit_nondefault_gpu_placement_contract(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "1" * 64
    path = full_objective_report_factory(tmp_path / "full-gpu.json", digest=digest)
    value = json.loads(path.read_text())
    value["fsdp2_placement"] = FSDP2_GPU_SHARDED
    value["parameter_storage"] = {
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

    with pytest.raises(ValueError, match="FSDP2 execution contract"):
        validate_full_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )
    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
            expected_fsdp2_placement=FSDP2_GPU_SHARDED,
        )["status"]
        == "PASS"
    )


def test_full_report_requires_explicit_nondefault_allocator_contract(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "2" * 64
    path = full_objective_report_factory(tmp_path / "full-expandable.json", digest=digest)
    value = json.loads(path.read_text())
    value["cuda_allocator"] = "expandable-segments"

    with pytest.raises(ValueError, match="CUDA allocator contract"):
        validate_full_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )
    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
            expected_cuda_allocator="expandable-segments",
        )["status"]
        == "PASS"
    )


def test_full_report_recomputes_registered_objective(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "7" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest, input_step=1)
    value = json.loads(path.read_text())
    value["rank_reports"][0]["steps"][0]["objective_total"] = 0.3
    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)

    with pytest.raises(ValueError, match="not recomputed"):
        validate_full_objective_report(value, require_initial_probe=False)


def test_full_report_rejects_ambiguous_gradient_audit_temporal_scope(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    path = full_objective_report_factory(tmp_path / "full.json", digest="a" * 64)
    value = json.loads(path.read_text())
    value["gradient_audit_temporal_scope"] = "full-sampled-objective"

    with pytest.raises(ValueError, match="gradient-audit temporal scope changed"):
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_full_report_accepts_conserved_shared_multi_depth_ownership(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    path = full_objective_report_factory(tmp_path / "full-depth.json", digest="8" * 64)
    value = json.loads(path.read_text())
    ownership = {
        "set/ownership": 0.06,
        "set/ownership_q1": 0.08,
        "set/ownership_q2": 0.1,
        "set/ownership_q3": 0.12,
    }
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["normalized_terms"].update(ownership)
        step["valid_counts"].update({name: 1 for name in ownership})
        for suffix in ("", "_q1", "_q2", "_q3"):
            step["normalized_terms"][f"set/ownership_nll{suffix}"] = 999.0
            step["valid_counts"][f"set/ownership_nll{suffix}"] = 1
    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )


def test_full_report_rejects_zero_predictive_host_gradient(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "2" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest, input_step=1)
    value = json.loads(path.read_text())
    value["rank_reports"][0]["steps"][0]["predictive_host_gradient_diagnostics"]["gradient_norms"][
        "middle"
    ] = 0.0
    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)

    with pytest.raises(ValueError, match="middle gradient norm must be positive"):
        validate_full_objective_report(value, require_initial_probe=False)


def test_full_report_rejects_current_observation_leak_into_prior_correction(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "3" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest, input_step=1)
    value = json.loads(path.read_text())
    interventions = value["rank_reports"][0]["steps"][0]["predictive_counterfactual_diagnostics"][
        "interventions"
    ]
    zero_observation = next(
        item for item in interventions if item["name"] == "zero_current_observation"
    )
    zero_observation["normalized_prediction_l1"] = 0.01
    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)

    with pytest.raises(ValueError, match="read the current observation"):
        validate_full_objective_report(value, require_initial_probe=False)


def test_full_report_rejects_incomplete_predictive_host_shard_coverage(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "6" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest, input_step=1)
    value = json.loads(path.read_text())
    value["rank_reports"][0]["steps"][0]["predictive_host_gradient_diagnostics"][
        "gradient_elements"
    ]["early"] = 2559
    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)

    with pytest.raises(ValueError, match="early gradient must cover exactly 2560 elements"):
        validate_full_objective_report(value, require_initial_probe=False)


def test_full_report_requires_exclusive_calvin_ownership_supervision(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "3" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest)
    value = json.loads(path.read_text())
    step = value["rank_reports"][0]["steps"][0]
    step["valid_counts"]["set/ownership"] = 0
    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)

    with pytest.raises(ValueError, match="active exclusive ownership"):
        validate_full_objective_report(value, require_initial_probe=True)

    value = json.loads(
        full_objective_report_factory(tmp_path / "full-2.json", digest=digest).read_text()
    )
    step = value["rank_reports"][0]["steps"][0]
    step["valid_counts"]["set/support"] = 1
    payload = json.dumps(value, sort_keys=True)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)

    with pytest.raises(ValueError, match="forbidden independent support"):
        validate_full_objective_report(value, require_initial_probe=True)


def test_full_report_can_be_semantically_validated_before_atomic_publication(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "4" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest)
    value = json.loads(path.read_text())
    value["checkpoint_dir"] = str((tmp_path / "pending" / "global_step_1").resolve())

    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=True,
            require_checkpoint_copy=False,
        )["status"]
        == "PASS"
    )
    with pytest.raises(ValueError, match="no real checkpoint directory"):
        validate_full_objective_report(value, require_initial_probe=True)


def test_full_report_bootstrap_and_wrong_time_phases_are_causally_separated(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "4" * 64
    fresh = json.loads(
        full_objective_report_factory(tmp_path / "fresh.json", digest=digest).read_text()
    )
    audited = json.loads(
        full_objective_report_factory(
            tmp_path / "audited.json",
            digest=digest,
            input_step=1,
        ).read_text()
    )
    fresh["gradient_audit_steps"] = [1]
    audit_fields = (
        "family_gradient_diagnostics",
        "relation_surface_gradient_diagnostics",
        "predictive_host_gradient_diagnostics",
    )
    for fresh_rank, audited_rank in zip(
        fresh["rank_reports"],
        audited["rank_reports"],
        strict=True,
    ):
        fresh_step = fresh_rank["steps"][0]
        audited_step = audited_rank["steps"][0]
        for field in audit_fields:
            fresh_step[field] = deepcopy(audited_step[field])
    with pytest.raises(ValueError, match="before recurrent bootstrap"):
        validate_full_objective_report(fresh, require_initial_probe=True)

    first_path = full_objective_report_factory(
        tmp_path / "first.json",
        digest=digest,
        input_step=1,
    )
    first = json.loads(first_path.read_text())
    intervention = first["rank_reports"][0]["steps"][0]["predictive_counterfactual_diagnostics"][
        "interventions"
    ]
    assert all(item["name"] != "wrong_time_source" for item in intervention)
    intervention.append(
        {
            "name": "wrong_time_source",
            "loss": 0.16,
            "loss_margin_over_factual": 0.06,
            "normalized_prediction_l1": 0.6,
        }
    )
    intervention.sort(key=lambda item: item["name"])
    with pytest.raises(ValueError, match="fabricated wrong-time"):
        validate_full_objective_report(
            first,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )

    mature_path = full_objective_report_factory(
        tmp_path / "mature.json",
        digest=digest,
        input_step=2,
    )
    mature = json.loads(mature_path.read_text())
    assert all(
        any(
            item["name"] == "wrong_time_source"
            for item in rank_report["steps"][0]["predictive_counterfactual_diagnostics"][
                "interventions"
            ]
        )
        for rank_report in mature["rank_reports"]
    )
    for rank_report in mature["rank_reports"]:
        interventions = rank_report["steps"][0]["predictive_counterfactual_diagnostics"][
            "interventions"
        ]
        interventions[:] = [item for item in interventions if item["name"] != "wrong_time_source"]
    with pytest.raises(ValueError, match="omitted wrong-time"):
        validate_full_objective_report(
            mature,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_full_report_rejects_obsolete_state_changing_refresh_metadata(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "5" * 64
    path = full_objective_report_factory(tmp_path / "mature.json", digest=digest, input_step=2)
    value = json.loads(path.read_text())
    for rank_report in value["rank_reports"]:
        step = rank_report["steps"][0]
        step["refresh"] = True
        step["refresh_replay_steps"] = 2
        step["refresh_seconds"] = 1.0
    with pytest.raises(ValueError, match="step fields differ"):
        validate_full_objective_report(
            value,
            require_initial_probe=False,
            require_checkpoint_copy=False,
        )


def test_full_report_allows_one_rank_without_predictive_target_off_audit(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "6" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest)
    value = json.loads(path.read_text())
    value["gradient_audit_steps"] = [2]
    for rank_report in value["rank_reports"]:
        rank_report["steps"][0]["family_gradient_diagnostics"] = None
        rank_report["steps"][0]["predictive_host_gradient_diagnostics"] = None
        rank_report["steps"][0]["predictive_counterfactual_diagnostics"] = None
    step = value["rank_reports"][1]["steps"][0]
    step["source_masked_branch"] = False
    step["source_prediction_mode"] = None
    step["omitted_static_digest"] = None
    step["normalized_terms"].pop("xmod/vision/representation")
    step["valid_counts"].pop("xmod/vision/representation")
    step["normalized_terms"]["correction/dino_video"] = 0.0
    step["valid_counts"]["correction/dino_video"] = 0
    step["objective_total"] = 0.21036
    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)

    assert validate_full_objective_report(value, require_initial_probe=False)["status"] == "PASS"


def test_full_report_audit_requires_predictive_target_on_every_rank(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "5" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest, input_step=1)
    value = json.loads(path.read_text())
    step = value["rank_reports"][1]["steps"][0]
    step["source_masked_branch"] = False
    step["source_prediction_mode"] = None
    step["omitted_static_digest"] = None
    step["normalized_terms"].pop("xmod/vision/representation")
    step["valid_counts"].pop("xmod/vision/representation")
    step["normalized_terms"]["correction/dino_video"] = 0.0
    step["valid_counts"]["correction/dino_video"] = 0
    step["objective_total"] = 0.21036
    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)

    with pytest.raises(ValueError, match="audit lacked predictive targets on every rank"):
        validate_full_objective_report(value, require_initial_probe=False)


def test_full_report_visual_artifact_is_file_and_provenance_bound(
    tmp_path: Path,
    full_objective_report_factory,
) -> None:
    digest = "9" * 64
    path = full_objective_report_factory(tmp_path / "full.json", digest=digest)
    value = json.loads(path.read_text())
    run_root = Path(value["checkpoint_dir"]).parent.parent
    visual = run_root / "visuals" / "rank0.png"
    visual.parent.mkdir(parents=True)
    visual.write_bytes(b"audited-png")
    step = value["rank_reports"][0]["steps"][0]
    step["visual_audit_seconds"] = 0.1
    step["visual_artifacts"] = [
        {
            "schema": "picf-next.lingbot-native-relation-visual.v5",
            "path": visual.relative_to(run_root).as_posix(),
            "sha256": _sha(visual.read_bytes()),
            "bytes": visual.stat().st_size,
            "global_step": 1,
            "input_weight_global_step": 0,
            "weight_boundary": "pre_update_forward",
            "rank": 0,
            "batch_index": 0,
            "sample_key": "sample-0",
            "task": "pick up the red block",
            "identity_keys": ["red_block"],
            "source_time": 0,
            "source_side": "posterior",
            "source_phase": 1,
            "binding_start_phase": [1, 2],
            "source_binding_valid": [True, False],
            "row_to_track": [0, -1],
            "sequence_row_to_track": [0, -1],
            "row_existence": [0.9, 0.1],
            "row_task_relevance": [0.8, 0.2],
            "row_matched_soft_iou": [0.75, 0.5],
            "anchor_surface": "task_object_probability.max(row)",
            "views": [
                {
                    "name": "static",
                    "merged_grid": [2, 2],
                    "source_shape": [224, 224, 3],
                    "token_count": 4,
                }
            ],
            "loss_only_labels_visible_to_model": False,
        }
    ]
    payload = json.dumps(value, sort_keys=True)
    path.write_text(payload)
    checkpoint_report = Path(value["checkpoint_dir"]) / "native_full_report.json"
    checkpoint_report.write_text(payload)
    assert (
        validate_full_objective_report(
            value,
            require_initial_probe=True,
        )
        == value
    )

    visual.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from its PNG"):
        validate_full_objective_report(
            value,
            require_initial_probe=True,
        )


def test_predictive_build_report_is_exact_and_tamper_evident(tmp_path: Path) -> None:
    digest = "b" * 64
    report = {
        "cache_manifest_sha256": digest,
        "coverage_sha256": digest,
        "expected_record_count": 8,
        "output_root": "/mnt/predictive",
        "pair_keys_sha256": digest,
        "patch_sha256": digest,
        "physical_visual_acceptance_sha256": digest,
        "stream_plan_sha256": digest,
        "teacher_encoder_digest": digest,
        "temporal_estimator_sha256": digest,
    }
    path = tmp_path / "report.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    assert load_predictive_build_report(path, expected_sha256=_sha(payload)) == report

    path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="expected digest"):
        load_predictive_build_report(path, expected_sha256=_sha(payload))


def test_teacher_causality_audit_is_exact_and_tamper_evident(tmp_path: Path) -> None:
    _target_report, predictive_report = _passing_predictive_target_audit()
    digest = "a" * 64
    temporal = predictive_temporal_diagnostics(
        torch.tensor([[2.0, -2.0, 0.0, 0.0], [0.0, 0.0, 2.0, -2.0]]),
        torch.tensor([[2.2, -2.0, 0.1, 0.0], [0.0, -0.2, 2.0, -1.9]]),
        identity_keys=("object/a", "object/b"),
        horizons=(1, 2),
    )
    temporal_ready, temporal_failures = predictive_temporal_pretraining_readiness(temporal)
    assert temporal_ready and not temporal_failures
    current_report = {
        "cache_manifest_sha256": digest,
        "teacher_encoder_digest": digest,
    }
    report = {
        "current_cache_manifest_sha256": current_report["cache_manifest_sha256"],
        "current_encoder_digest": current_report["teacher_encoder_digest"],
        "dataset_tree_sha256": digest,
        "diagnostics": {
            "current_cache_patch_elements": 2 * 256 * 1024,
            "current_cache_patch_mismatch_count": 0,
            "current_patch_elements": 2 * 256 * 1024,
            "current_patch_mismatch_count": 0,
            "future_feature_elements": 2 * 1024,
            "future_feature_mismatch_count": 0,
            "future_importance_elements": 2,
            "future_importance_mismatch_count": 0,
            "maximum_current_cache_patch_absolute_error": 0.0,
            "maximum_current_patch_absolute_error": 0.0,
            "maximum_future_feature_absolute_error": 0.0,
            "maximum_future_importance_absolute_error": 0.0,
            "sample_selection_sha256": digest,
            "sampled_horizon_record_counts": {"1": 1, "2": 1, "64": 0},
            "sampled_record_count": 2,
            "same_call_supported_pair_count": temporal.pair_count,
            "same_call_temporal_diagnostics": temporal.as_dict(),
            "same_call_temporal_pretraining_readiness": "PASS",
            "same_call_temporal_pretraining_readiness_failures": [],
            "status": "PASS",
        },
        "patch_sha256": digest,
        "physical_sidecar_manifest_sha256": digest,
        "predictive_cache_manifest_sha256": predictive_report["cache_manifest_sha256"],
        "predictive_encoder_digest": predictive_report["teacher_encoder_digest"],
        "scanned_record_count": predictive_report["expected_record_count"],
        "schema": TEACHER_CAUSALITY_AUDIT_SCHEMA,
    }
    path = tmp_path / "teacher-causality.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    kwargs = {
        "expected_sha256": _sha(payload),
        "predictive_report": predictive_report,
        "current_grid_report": current_report,
        "dataset_tree_sha256": digest,
        "physical_sidecar_manifest_sha256": digest,
        "patch_sha256": digest,
        "horizons": (1, 2, 64),
    }

    assert load_predictive_teacher_causality_audit(path, **kwargs) == report

    report["diagnostics"]["current_patch_mismatch_count"] = 1
    path.write_text(json.dumps(report, sort_keys=True))
    with pytest.raises(ValueError, match="mismatch_count is nonzero"):
        load_predictive_teacher_causality_audit(
            path,
            **{**kwargs, "expected_sha256": _sha(path.read_bytes())},
        )

    report["diagnostics"]["current_patch_mismatch_count"] = 0
    report["diagnostics"]["current_cache_patch_mismatch_count"] = 1
    path.write_text(json.dumps(report, sort_keys=True))
    with pytest.raises(ValueError, match="current_cache_patch_mismatch_count is nonzero"):
        load_predictive_teacher_causality_audit(
            path,
            **{**kwargs, "expected_sha256": _sha(path.read_bytes())},
        )


def test_predictive_target_audit_is_recomputed_and_tamper_evident(tmp_path: Path) -> None:
    report, predictive_report = _passing_predictive_target_audit()
    contract = report["cache_contract"]
    assert isinstance(contract, dict)
    path = tmp_path / "predictive-target-audit.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    kwargs = {
        "expected_sha256": _sha(payload),
        "predictive_report": predictive_report,
        "dataset_tree_sha256": contract["dataset_tree_sha256"],
        "physical_sidecar_manifest_sha256": contract["physical_sidecar_manifest_sha256"],
        "query_schema_sha256": contract["query_schema_sha256"],
        "stream_plan_sha256": contract["stream_plan_sha256"],
        "temporal_estimator_sha256": contract["temporal_estimator_sha256"],
    }
    assert load_predictive_target_audit(path, **kwargs) == json.loads(payload)

    path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="expected digest"):
        load_predictive_target_audit(path, **kwargs)


def test_predictive_target_audit_requires_every_declared_horizon(tmp_path: Path) -> None:
    report, predictive_report = _passing_predictive_target_audit()
    contract = report["cache_contract"]
    assert isinstance(contract, dict)
    report["horizon_record_counts"] = {"1": 8}
    path = tmp_path / "missing-horizon-target-audit.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="horizon accounting"):
        load_predictive_target_audit(
            path,
            expected_sha256=_sha(payload),
            predictive_report=predictive_report,
            dataset_tree_sha256=contract["dataset_tree_sha256"],
            physical_sidecar_manifest_sha256=contract["physical_sidecar_manifest_sha256"],
            query_schema_sha256=contract["query_schema_sha256"],
            stream_plan_sha256=contract["stream_plan_sha256"],
            temporal_estimator_sha256=contract["temporal_estimator_sha256"],
        )


def test_predictive_target_audit_rejects_inconsistent_visible_support(tmp_path: Path) -> None:
    report, predictive_report = _passing_predictive_target_audit()
    contract = report["cache_contract"]
    visible_support = report["visible_support_diagnostics"]
    assert isinstance(contract, dict) and isinstance(visible_support, dict)
    visible_support["supported_count"] = 5
    path = tmp_path / "inconsistent-visible-support-target-audit.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="differ from audit coverage"):
        load_predictive_target_audit(
            path,
            expected_sha256=_sha(payload),
            predictive_report=predictive_report,
            dataset_tree_sha256=contract["dataset_tree_sha256"],
            physical_sidecar_manifest_sha256=contract["physical_sidecar_manifest_sha256"],
            query_schema_sha256=contract["query_schema_sha256"],
            stream_plan_sha256=contract["stream_plan_sha256"],
            temporal_estimator_sha256=contract["temporal_estimator_sha256"],
        )


def test_predictive_target_audit_rejects_collapsed_targets(tmp_path: Path) -> None:
    report, predictive_report = _passing_predictive_target_audit()
    contract = report["cache_contract"]
    assert isinstance(contract, dict)
    collapsed = predictive_latent_diagnostics(
        torch.tensor([[2.0, -1.0, 0.0]]).expand(4, -1).clone(),
        identity_keys=("a", "b", "a", "b"),
        target_group_keys=("f1", "f1", "f2", "f2"),
    )
    ready, failures = predictive_target_pretraining_readiness(collapsed)
    assert not ready
    report["diagnostics"] = collapsed.as_dict()
    interpretation = report["interpretation"]
    assert isinstance(interpretation, dict)
    interpretation.update(
        {
            "numerical_status": "obvious_target_collapse",
            "pretraining_readiness": "FAIL",
            "pretraining_readiness_failures": list(failures),
            "retrieval_is_computable": collapsed.retrieval_query_count > 0,
        }
    )
    path = tmp_path / "collapsed-target-audit.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="failed pretraining readiness"):
        load_predictive_target_audit(
            path,
            expected_sha256=_sha(payload),
            predictive_report=predictive_report,
            dataset_tree_sha256=contract["dataset_tree_sha256"],
            physical_sidecar_manifest_sha256=contract["physical_sidecar_manifest_sha256"],
            query_schema_sha256=contract["query_schema_sha256"],
            stream_plan_sha256=contract["stream_plan_sha256"],
            temporal_estimator_sha256=contract["temporal_estimator_sha256"],
        )


def test_predictive_temporal_audit_is_recomputed_and_tamper_evident(
    tmp_path: Path,
) -> None:
    report, predictive_report, current_report, sidecar_digest, horizons = (
        _passing_predictive_temporal_audit()
    )
    path = tmp_path / "predictive-temporal-audit.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    kwargs = {
        "expected_sha256": _sha(payload),
        "predictive_report": predictive_report,
        "current_grid_report": current_report,
        "physical_sidecar_manifest_sha256": sidecar_digest,
        "horizons": horizons,
    }
    assert load_predictive_temporal_audit(path, **kwargs) == json.loads(payload)

    path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="expected digest"):
        load_predictive_temporal_audit(path, **kwargs)


def test_predictive_temporal_audit_requires_every_declared_horizon(
    tmp_path: Path,
) -> None:
    report, predictive_report, current_report, sidecar_digest, horizons = (
        _passing_predictive_temporal_audit()
    )
    report["horizon_supported_pair_counts"] = {"1": 4}
    path = tmp_path / "missing-horizon-temporal-audit.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="horizon accounting"):
        load_predictive_temporal_audit(
            path,
            expected_sha256=_sha(payload),
            predictive_report=predictive_report,
            current_grid_report=current_report,
            physical_sidecar_manifest_sha256=sidecar_digest,
            horizons=horizons,
        )


def test_predictive_temporal_audit_allows_source_plan_to_skip_short_horizon(
    tmp_path: Path,
) -> None:
    report, predictive_report, current_report, sidecar_digest, horizons = (
        _passing_predictive_temporal_audit()
    )
    current = torch.tensor(
        [
            [2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, -2.0],
            [1.8, -2.1, 0.1, 0.0],
            [0.1, 0.0, 1.9, -2.2],
        ]
    )
    future = current + torch.tensor(
        [
            [0.2, 0.0, 0.1, 0.0],
            [0.0, -0.2, 0.0, 0.1],
            [0.0, 0.1, -0.2, 0.0],
            [-0.1, 0.0, 0.0, 0.2],
        ]
    )
    diagnostics = predictive_temporal_diagnostics(
        current,
        future,
        identity_keys=("object/a", "object/b", "object/a", "object/b"),
        horizons=(2, 2, 2, 2),
    )
    ready, failures = predictive_temporal_pretraining_readiness(diagnostics)
    assert ready and not failures
    report["diagnostics"] = diagnostics.as_dict()
    report["horizon_supported_pair_counts"] = {"1": 0, "2": 4}
    path = tmp_path / "zero-short-horizon-temporal-audit.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)

    assert load_predictive_temporal_audit(
        path,
        expected_sha256=_sha(payload),
        predictive_report=predictive_report,
        current_grid_report=current_report,
        physical_sidecar_manifest_sha256=sidecar_digest,
        horizons=horizons,
    ) == json.loads(payload)


def test_predictive_temporal_audit_requires_complete_future_cache_join(
    tmp_path: Path,
) -> None:
    report, predictive_report, current_report, sidecar_digest, horizons = (
        _passing_predictive_temporal_audit()
    )
    report["matched_future_record_count"] = 3
    path = tmp_path / "partial-future-join-temporal-audit.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="coverage or sample accounting"):
        load_predictive_temporal_audit(
            path,
            expected_sha256=_sha(payload),
            predictive_report=predictive_report,
            current_grid_report=current_report,
            physical_sidecar_manifest_sha256=sidecar_digest,
            horizons=horizons,
        )


def test_predictive_temporal_audit_rejects_current_copy_targets(tmp_path: Path) -> None:
    report, predictive_report, current_report, sidecar_digest, horizons = (
        _passing_predictive_temporal_audit()
    )
    unchanged = torch.tensor(
        [
            [2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, -2.0],
            [1.8, -2.1, 0.1, 0.0],
            [0.1, 0.0, 1.9, -2.2],
        ]
    )
    diagnostics = predictive_temporal_diagnostics(
        unchanged,
        unchanged.clone(),
        identity_keys=("object/a", "object/b", "object/a", "object/b"),
        horizons=(1, 1, 2, 2),
    )
    ready, failures = predictive_temporal_pretraining_readiness(diagnostics)
    assert not ready
    report["diagnostics"] = diagnostics.as_dict()
    interpretation = report["interpretation"]
    assert isinstance(interpretation, dict)
    interpretation.update(
        {
            "controlled_future_temporal_pretraining_readiness": "FAIL",
            "controlled_future_temporal_pretraining_readiness_failures": list(failures),
            "pretraining_readiness": "FAIL",
            "pretraining_readiness_failures": [
                f"controlled_future:{failure}" for failure in failures
            ],
        }
    )
    path = tmp_path / "unchanged-temporal-audit.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="failed pretraining readiness"):
        load_predictive_temporal_audit(
            path,
            expected_sha256=_sha(payload),
            predictive_report=predictive_report,
            current_grid_report=current_report,
            physical_sidecar_manifest_sha256=sidecar_digest,
            horizons=horizons,
        )


def test_current_grid_build_report_is_exact_and_tamper_evident(tmp_path: Path) -> None:
    digest = "c" * 64
    report = {
        "cache_manifest_sha256": digest,
        "coverage_sha256": digest,
        "expected_record_count": 7,
        "output_root": "/mnt/current-grid",
        "patch_sha256": digest,
        "physical_visual_acceptance_sha256": digest,
        "source_keys_sha256": digest,
        "stream_plan_sha256": digest,
        "teacher_encoder_digest": digest,
        "temporal_estimator_sha256": digest,
    }
    path = tmp_path / "current-grid-report.json"
    payload = json.dumps(report, sort_keys=True).encode("ascii")
    path.write_bytes(payload)
    assert load_current_grid_build_report(path, expected_sha256=_sha(payload)) == report

    donor_report = {
        **report,
        "content_identical_donor": {
            "donor_cache_manifest_sha256": "a" * 64,
            "donor_content_manifest_sha256": "b" * 64,
            "official_source_receipt_sha256": "d" * 64,
            "reused_record_count": 3,
            "target_dataset_manifest_sha256": "e" * 64,
        },
    }
    donor_payload = json.dumps(donor_report, sort_keys=True).encode("ascii")
    path.write_bytes(donor_payload)
    assert (
        load_current_grid_build_report(path, expected_sha256=_sha(donor_payload))
        == donor_report
    )

    invalid_donor = dict(donor_report)
    invalid_donor["content_identical_donor"] = {
        **donor_report["content_identical_donor"],
        "reused_record_count": 8,
    }
    path.write_text(json.dumps(invalid_donor), encoding="ascii")
    with pytest.raises(ValueError, match="outside coverage"):
        load_current_grid_build_report(path, expected_sha256=_sha(path.read_bytes()))

    malformed = dict(report)
    malformed["unexpected"] = 1
    path.write_text(json.dumps(malformed), encoding="ascii")
    with pytest.raises(ValueError, match="fields differ"):
        load_current_grid_build_report(path, expected_sha256=_sha(path.read_bytes()))


def test_target_caches_share_producer_patch_independent_of_training_consumer() -> None:
    predictive_report = {"patch_sha256": "a" * 64}
    current_grid_report = {"patch_sha256": "a" * 64}
    assert (
        _cache_producer_patch_sha256(predictive_report, current_grid_report)
        == predictive_report["patch_sha256"]
    )

    current_grid_report["patch_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="different producer patches"):
        _cache_producer_patch_sha256(predictive_report, current_grid_report)


def test_full_runner_resume_extra_binds_implementation_and_boundary_hashes() -> None:
    value = _resume_extra()
    assert _validate_resume(value) == value

    changed = dict(value)
    changed["implementation_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="provenance"):
        _validate_resume(changed)

    changed = dict(value)
    changed["boundary_sha256"] = dict(value["boundary_sha256"])
    changed["boundary_sha256"]["lane_snapshot_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _validate_resume(changed)


def test_representation_resume_extra_binds_stage_split_and_parameter_scope() -> None:
    digest = "a" * 64
    value = {
        **_resume_extra(),
        "schema": REPRESENTATION_EXTRA_STATE_SCHEMA,
        "training_stage": "representation",
        "representation_split_sha256": digest,
        "representation_parameter_scope_sha256": digest,
        "representation_frozen_action_state_sha256": digest,
    }
    kwargs = {
        "expected_global_step": 3,
        "expected_implementation_sha256": digest,
        "expected_model_family_sha256": digest,
        "expected_execution_sha256": digest,
        "expected_plan_sha256": digest,
        "expected_temporal_sha256": digest,
        "expected_source_digest": digest,
        "expected_representation_split_sha256": digest,
        "expected_parameter_scope_sha256": digest,
        "expected_behavior_conditioning_sha256": None,
        "rank": 0,
    }
    assert (
        _validate_representation_resume_extra(value, **kwargs)["training_stage"] == "representation"
    )

    changed = deepcopy(value)
    changed["representation_split_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="checkpoint provenance differs"):
        _validate_representation_resume_extra(changed, **kwargs)

    changed = deepcopy(value)
    changed["training_stage"] = "joint_adoption"
    with pytest.raises(ValueError, match="checkpoint stage differs"):
        _validate_representation_resume_extra(changed, **kwargs)


def test_full_implementation_digest_covers_the_training_closure() -> None:
    paths = {str(path.relative_to(ROOT)) for path in _full_implementation_paths(ROOT)}
    required = {
        "references/patches/lingbot_vla2_picf_native.patch",
        "src/picf_next/artifact_io.py",
        "src/picf_next/lingbot_native/capacity.py",
        "src/picf_next/lingbot_native/calvin.py",
        "src/picf_next/lingbot_native/current_grid_cache.py",
        "src/picf_next/lingbot_native/empirical_producers.py",
        "src/picf_next/lingbot_native/empirical_statistics.py",
        "src/picf_next/lingbot_native/fixed_batch_probe.py",
        "src/picf_next/lingbot_native/full_training.py",
        "src/picf_next/lingbot_native/predictive_cache.py",
        "src/picf_next/lingbot_native/predictive_diagnostics.py",
        "src/picf_next/lingbot_native/predictive_plan.py",
        "src/picf_next/lingbot_native/predictive_probes.py",
        "src/picf_next/lingbot_native/relation_bilinear_probe.py",
        "src/picf_next/lingbot_native/relation_depth_probe.py",
        "src/picf_next/lingbot_native/relation_geometry_probe.py",
        "src/picf_next/lingbot_native/relation_gradient_diagnostics.py",
        "src/picf_next/lingbot_native/relation_precision_audit.py",
        "src/picf_next/lingbot_native/representation_intervention.py",
        "src/picf_next/lingbot_native/representation_split.py",
        "src/picf_next/lingbot_native/representation_stage.py",
        "src/picf_next/lingbot_native/stage_control.py",
        "src/picf_next/lingbot_native/task_relation.py",
        "src/picf_next/lingbot_native/temporal.py",
        "src/picf_next/lingbot_native/training.py",
        "tools/audit_lingbot_predictive_targets.py",
        "tools/audit_lingbot_predictive_temporal_targets.py",
        "tools/audit_lingbot_dino_teacher_causality.py",
        "tools/build_lingbot_calvin_current_grid_cache.py",
        "tools/build_lingbot_calvin_predictive_cache.py",
        "tools/build_lingbot_representation_split.py",
        "tools/build_lingbot_representation_task_intervention.py",
        "tools/lingbot_vla2_runtime_helpers.py",
        "tools/run_lingbot_vla2_native_full.py",
    }
    assert required <= paths
    assert {
        "src/picf_next/association.py",
        "src/picf_next/hosts/lingbot_vla2.py",
        "src/picf_next/posterior.py",
        "src/picf_next/unified/objective.py",
    }.isdisjoint(paths)
    digest = _full_implementation_digest(ROOT)
    assert len(digest) == 64
    assert digest == _full_implementation_digest(ROOT)


def _is_exact_native_g0_rank_failure_exchange(
    *,
    relative: str,
    tree: ast.Module,
    handler: ast.ExceptHandler,
) -> bool:
    """Recognize only the fail-closed native G0 rank-error exchange boundary."""

    if relative != "tools/run_lingbot_vla2_native_g0.py" or handler.name != "error":
        return False
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_distributed_rank_local_call"
    ]
    if len(functions) != 1:
        return False
    function = functions[0]
    if function.end_lineno is None or not (
        function.lineno <= handler.lineno <= function.end_lineno
    ):
        return False
    if len(handler.body) != 1 or not isinstance(handler.body[0], ast.Assign):
        return False
    assignment = handler.body[0]
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or assignment.targets[0].id != "rank_local_error"
        or not isinstance(assignment.value, ast.Name)
        or assignment.value.id != "error"
    ):
        return False
    owning_tries = [
        node for node in function.body if isinstance(node, ast.Try) and handler in node.handlers
    ]
    if len(owning_tries) != 1:
        return False
    try_node = owning_tries[0]
    exchange_calls = [
        node
        for node in function.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_distributed_phase_error"
    ]
    if len(exchange_calls) != 1:
        return False
    exchange = exchange_calls[0]
    exchange_keywords = {
        keyword.arg: keyword.value for keyword in exchange.value.keywords if keyword.arg is not None
    }
    expected_keywords = {
        "error": "rank_local_error",
        "phase": "phase",
        "rank": "rank",
        "dist_module": "dist_module",
    }
    if set(exchange_keywords) != set(expected_keywords) or any(
        not isinstance(exchange_keywords[name], ast.Name) or exchange_keywords[name].id != value
        for name, value in expected_keywords.items()
    ):
        return False
    result_returns = [
        node
        for node in function.body
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "result"
    ]
    return (
        len(result_returns) == 1
        and try_node.end_lineno is not None
        and exchange.end_lineno is not None
        and try_node.end_lineno < exchange.lineno
        and exchange.end_lineno < result_returns[0].lineno
    )


def test_full_training_closure_cannot_reach_historical_semantic_authorities() -> None:
    forbidden_prefixes = (
        "picf_next.association",
        "picf_next.hosts.lingbot_unified",
        "picf_next.hosts.lingbot_vla2",
        "picf_next.hosts.molmoact2",
        "picf_next.models",
        "picf_next.posterior",
        "picf_next.training.molmoact2",
        "picf_next.training.stateful_runner",
        "picf_next.training.stationary",
        "picf_next.unified",
    )
    dynamic_imports: set[tuple[str, str, str]] = set()
    converted_base_exceptions: set[tuple[str, str]] = set()

    for path in _full_implementation_paths(ROOT):
        if path.suffix != ".py":
            continue
        relative = str(path.relative_to(ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                pytest.fail(
                    f"production assertion can disappear under -O: {relative}:{node.lineno}"
                )
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                pytest.fail(
                    f"bare exception handler in production closure: {relative}:{node.lineno}"
                )
            if (
                isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id == "Exception"
            ):
                pytest.fail(
                    f"ambiguous broad exception in production closure: {relative}:{node.lineno}"
                )
            if (
                isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id == "BaseException"
                and not any(isinstance(item, ast.Raise) for item in ast.walk(node))
            ):
                body = "\n".join(ast.unparse(item) for item in node.body)
                if "dataset_contract[0]" in body and "'status': 'FAIL'" in body:
                    conversion = "dataset_contract_failure"
                elif "precheckpoint_error[0]" in body:
                    conversion = "precheckpoint_validation_failure"
                elif "publish_error[0]" in body:
                    conversion = "checkpoint_publication_failure"
                elif "objective_error = error" in body:
                    conversion = "pre_backward_objective_failure"
                elif "forward_error = error" in body or "state_error = error" in body:
                    conversion = "action_diagnostic_transactional_rethrow"
                elif "gradient_metrics_error = error" in body:
                    conversion = "distributed_gradient_traversal_failure"
                elif _is_exact_native_g0_rank_failure_exchange(
                    relative=relative,
                    tree=tree,
                    handler=node,
                ):
                    conversion = "native_g0_rank_local_failure_exchange"
                elif any(
                    marker in body
                    for marker in (
                        "selection_error = error",
                        "enter_error = error",
                        "close_error = error",
                        "setup_error[0]",
                        "setup_error = error",
                        "prepare_error = error",
                        "seed_error = error",
                        "factual_postcheck_error = error",
                        "shuffled_postcheck_error = error",
                        "capture_error = error",
                        "probe_setup_error = error",
                        "point_error = error",
                        "backward_error = error",
                        "gradient_validation_error = error",
                        "evidence_error = error",
                        "control_assembly_error = error",
                        "gradient_error = error",
                        "update_error = error",
                        "counter_error = error",
                        "final_error = error",
                        "final_local_error = error",
                        "baseline_error = error",
                        "batch_materialization_error = error",
                        "checkpoint_report_error = error",
                    )
                ):
                    conversion = "relation_probe_distributed_local_failure"
                else:
                    pytest.fail(
                        "BaseException is swallowed outside an exact distributed failure "
                        f"conversion: {relative}:{node.lineno}"
                    )
                converted_base_exceptions.add((relative, conversion))
            if isinstance(node, ast.Import):
                imported_modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules = () if node.module is None else (node.module,)
                if node.module == "picf_next":
                    pytest.fail(
                        "production closure may not consume legacy root-package lazy exports: "
                        f"{relative}:{node.lineno}"
                    )
            else:
                imported_modules = ()
            for module in imported_modules:
                if any(
                    module == prefix or module.startswith(f"{prefix}.")
                    for prefix in forbidden_prefixes
                ):
                    pytest.fail(
                        f"historical semantic module is production-reachable: "
                        f"{relative}:{node.lineno}:{module}"
                    )
            if not isinstance(node, ast.Call):
                continue
            is_import_module = (
                isinstance(node.func, ast.Name) and node.func.id in {"__import__", "import_module"}
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"__import__", "import_module"}
            )
            if not is_import_module:
                continue
            if len(node.args) != 1:
                pytest.fail(
                    f"unbounded dynamic import in production closure: {relative}:{node.lineno}"
                )
            argument = ast.dump(node.args[0], include_attributes=False)
            function_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            dynamic_imports.add((relative, function_name, argument))

    assert dynamic_imports == {
        (
            "src/picf_next/__init__.py",
            "import_module",
            "Name(id='module_name', ctx=Load())",
        ),
        (
            "tools/build_lingbot_calvin_predictive_cache.py",
            "import_module",
            "Constant(value='lingbotvla.models.vla.vision_models.module_utils')",
        ),
    }
    assert converted_base_exceptions == {
        (
            "src/picf_next/lingbot_native/representation_evaluation_runtime.py",
            "action_diagnostic_transactional_rethrow",
        ),
        (
            "src/picf_next/lingbot_native/representation_evaluation_runtime.py",
            "checkpoint_publication_failure",
        ),
        (
            "src/picf_next/lingbot_native/representation_evaluation_runtime.py",
            "relation_probe_distributed_local_failure",
        ),
        ("tools/run_lingbot_vla2_native_full.py", "checkpoint_publication_failure"),
        ("tools/run_lingbot_vla2_native_full.py", "dataset_contract_failure"),
        ("tools/run_lingbot_vla2_native_full.py", "pre_backward_objective_failure"),
        ("tools/run_lingbot_vla2_native_full.py", "precheckpoint_validation_failure"),
        (
            "tools/run_lingbot_vla2_native_full.py",
            "relation_probe_distributed_local_failure",
        ),
        ("tools/run_lingbot_vla2_native_g0.py", "checkpoint_publication_failure"),
        ("tools/run_lingbot_vla2_native_g0.py", "dataset_contract_failure"),
        (
            "tools/run_lingbot_vla2_native_g0.py",
            "distributed_gradient_traversal_failure",
        ),
        (
            "tools/run_lingbot_vla2_native_g0.py",
            "native_g0_rank_local_failure_exchange",
        ),
        ("tools/run_lingbot_vla2_native_g0.py", "precheckpoint_validation_failure"),
    }


def test_full_implementation_closure_resolves_relative_imports_and_excludes_legacy_models() -> None:
    legacy = ROOT / "src/picf_next/models/dynamics_loss.py"
    assert {
        "picf_next.models.core",
        "picf_next.models.temporal",
    } <= set(_local_import_modules(ROOT, legacy))

    paths = set(_full_implementation_paths(ROOT))
    assert ROOT / "src/picf_next/data/rollout_targets.py" in paths
    assert legacy not in paths
    assert all("/src/picf_next/models/" not in path.as_posix() for path in paths)
