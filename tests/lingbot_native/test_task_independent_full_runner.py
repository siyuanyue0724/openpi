from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import tools.run_lingbot_vla2_task_independent_full as full_runner
from picf_next.lingbot_native.temporal import (
    TemporalEstimatorConfig,
    sample_temporal_batch_plan,
)
from tools.run_lingbot_vla2_task_independent_full import (
    ADR148_ENTITY_WEIGHT,
    ADR148_PREDICTIVE_WEIGHT,
    ADR148_SOURCE_MASK_PROBABILITY,
    ADR149_ENTITY_WEIGHT,
    ADR149_FILTER_WEIGHT,
    ADR149_OMITTED_STATIC_PROBABILITY,
    ADR178_ARCHITECTURE_PROFILE,
    ADR178_NATIVE_ATTENTION_WEIGHT,
    ADR178_RELATION_SUPERVISION_LAYERS,
    ADR193_ANCHOR_CHECK_STEPS,
    ADR193_ARCHITECTURE_PROFILE,
    ADR204_ARCHITECTURE_PROFILE,
    ADR205_ARCHITECTURE_PROFILE,
    ADR207_ACTION_EFFECT_MIN_ABS_DRIFT,
    ADR207_ANCHOR_CHECK_STEPS,
    ADR207_ARCHITECTURE_PROFILE,
    ADR207_MODALITY_INTERVENTION_SCHEMA,
    ADR207_MODALITY_INTERVENTION_STEPS,
    ADR207_MODALITY_INTERVENTIONS,
    ADR225_ARCHITECTURE_PROFILE,
    CAUSAL_ARM_STOP_STEP,
    CAUSAL_BRANCH_STEP,
    CHECKPOINT_EVERY,
    LINGBOT_COMPILE_UPSTREAM_DEFAULT,
    METRICS_EVERY,
    POSTERIOR_ADOPTION_DOSE_ACTION_EVALUATION_STEPS,
    POSTERIOR_ADOPTION_DOSE_SOURCE_MASK_PROBABILITY,
    POSTERIOR_ADOPTION_DOSE_STOP_STEP,
    POSTERIOR_ADOPTION_STOP_STEP,
    PRODUCTION_LOCAL_BPTT_PROBABILITY,
    RUNNER_SCHEMA,
    SUPPORTED_WORLD_SIZES,
    TOTAL_STEPS,
    TRAINABLE_SCOPE_FROZEN_VISION_HOST,
    TRAINABLE_SCOPE_FULL_HOST,
    TWO_PASS_ACTION_EVALUATION_SCHEMA,
    TWO_PASS_ACTION_EVALUATION_STEPS,
    TWO_PASS_FILTER_DIAGNOSTIC_SCHEMA,
    TWO_PASS_FILTER_DIAGNOSTIC_STEPS,
    V3_DISTRIBUTED_PRIOR_SCHEDULE,
    VIDEOMT_FSDP2_PLACEMENT_CPU_OFFLOAD,
    VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED,
    VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS,
    VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
    VIDEOMT_RELEASED_CHECKPOINT_SHA256,
    VIDEOMT_STAGE_PQ_DISABLED,
    VIDEOMT_STAGE_PQ_FROZEN_RELEASED_EVAL_C5,
    VIDEOMT_STAGE_PQM_FROZEN_ADAPTED_EVAL_C5,
    VIDEOMT_STAGE_PQM_FROZEN_RELEASED_EVAL_C5,
    VIDEOMT_STAGE_PQMR_FROZEN_ADAPTED_EVAL_C5,
    VIDEOMT_STAGE_PQRF_FROZEN_ADAPTED_EVAL_C5,
    VISUAL_EVERY,
    ProductionCadence,
    _acceptance_checkpoint_due,
    _acceptance_terminal_evidence_due,
    _action_evaluation_active,
    _adr178_direct_action_posterior_active,
    _adr193_implicit_multimodal_anchor_active,
    _adr204_full_source_final_only_active,
    _adr205_released_query_propagation_active,
    _anchor_evaluation_due,
    _cache_manifest,
    _causal_checkpoint_due,
    _current_correction_assets_required,
    _dense_evidence_training_step_prefix,
    _direct_action_posterior_targets,
    _disabled_auxiliary_digest,
    _distributed_prior_host_step_schedule,
    _emit_progress,
    _evaluation_visual_sample_keys,
    _execution_contract,
    _external_stop_requested,
    _nonnegative_finite_float,
    _objective_posterior_inputs,
    _physical_step_observability,
    _picf_optimizer_learning_rate_stratification_active,
    _positive_finite_float,
    _posterior_adoption_route_active,
    _predictive_assets_required,
    _prepare_rank_metric_journal,
    _pretrained_object_memory_step_report,
    _prune_resume_publications,
    _registered_action_evaluation_steps,
    _resolve_current_grid_cache_coverage,
    _runtime_world_size,
    _scientific_terminal_checkpoint_due,
    _staged_row_bindings,
    _summarize_adr207_modality_interventions,
    _trainable_scope_receipt,
    _validate_acceptance_args,
    _validate_auxiliary_cache_args,
    _validate_causal_ablation_args,
    _validate_current_cache_build_binding,
    _validate_dense_evidence_args,
    _validate_engineering_smoke_args,
    _validate_frozen_stream_args,
    _validate_picf_architecture_profile,
    _validate_posterior_adoption_dose_step,
    _validate_production_temporal_estimator,
    _validate_videomt_stage_pq_args,
)


def test_production_cadence_is_exact_and_nonzero() -> None:
    cadence = ProductionCadence()

    assert (TOTAL_STEPS, METRICS_EVERY, VISUAL_EVERY, CHECKPOINT_EVERY) == (
        30_000,
        100,
        250,
        2_000,
    )
    assert cadence.metrics_due(100)
    assert cadence.visual_due(250)
    assert cadence.checkpoint_due(2_000)
    assert not cadence.metrics_due(0)
    assert not cadence.visual_due(1)
    assert not cadence.checkpoint_due(1)
    assert not cadence.checkpoint_due(1_999)


class _RuntimeBufferPolicy(torch.nn.Module):
    def __init__(self, *, include_released_action_buffers: bool) -> None:
        super().__init__()
        if include_released_action_buffers:
            self.register_buffer("avg_topk_sigmoid_score", torch.tensor([0.25]))
            self.register_buffer("tokens_per_expert", torch.tensor([3.0, 4.0]))


def test_action_runtime_buffer_snapshot_is_backend_exact() -> None:
    released = _RuntimeBufferPolicy(include_released_action_buffers=True)
    snapshot = full_runner._snapshot_action_backend_runtime_buffers(
        released,
        action_backend="lingbot_released",
    )
    released.avg_topk_sigmoid_score.fill_(0.75)
    released.tokens_per_expert.zero_()
    full_runner._restore_action_backend_runtime_buffers(
        snapshot,
        action_backend="lingbot_released",
        torch_module=torch,
    )
    torch.testing.assert_close(released.avg_topk_sigmoid_score, torch.tensor([0.25]))
    torch.testing.assert_close(released.tokens_per_expert, torch.tensor([3.0, 4.0]))

    complete_wla = _RuntimeBufferPolicy(include_released_action_buffers=False)
    wla_snapshot = full_runner._snapshot_action_backend_runtime_buffers(
        complete_wla,
        action_backend="wla_complete",
    )
    assert wla_snapshot.values == ()
    full_runner._restore_action_backend_runtime_buffers(
        wla_snapshot,
        action_backend="wla_complete",
        torch_module=torch,
    )


def test_action_runtime_buffer_snapshot_rejects_backend_contract_drift() -> None:
    released = _RuntimeBufferPolicy(include_released_action_buffers=True)
    complete_wla = _RuntimeBufferPolicy(include_released_action_buffers=False)

    with pytest.raises(RuntimeError, match="no official action-MoE"):
        full_runner._snapshot_action_backend_runtime_buffers(
            complete_wla,
            action_backend="lingbot_released",
        )
    with pytest.raises(RuntimeError, match="unexpectedly retained"):
        full_runner._snapshot_action_backend_runtime_buffers(
            released,
            action_backend="wla_complete",
        )

    wla_snapshot = full_runner._snapshot_action_backend_runtime_buffers(
        complete_wla,
        action_backend="wla_complete",
    )
    with pytest.raises(RuntimeError, match="different backend"):
        full_runner._restore_action_backend_runtime_buffers(
            wla_snapshot,
            action_backend="lingbot_released",
            torch_module=torch,
        )


@pytest.mark.parametrize(
    ("source_update_arm", "expected_source_steps", "expected_scheduler_steps"),
    (("joint", 1, 1), ("frozen-coordinate-control", 0, 0)),
)
def test_source_update_control_changes_only_the_source_optimizer_transaction(
    monkeypatch: pytest.MonkeyPatch,
    source_update_arm: str,
    expected_source_steps: int,
    expected_scheduler_steps: int,
) -> None:
    class _Optimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 4.0e-6}]
            self.step_count = 0
            self.zero_count = 0

        def step(self) -> None:
            self.step_count += 1

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none
            self.zero_count += 1

    class _Scheduler:
        def __init__(self) -> None:
            self.last_epoch = 0
            self.step_count = 0

        def step(self) -> None:
            self.step_count += 1
            self.last_epoch += 1

        def get_last_lr(self) -> list[float]:
            return [1.0e-6]

    source_optimizer = _Optimizer()
    source_scheduler = _Scheduler()
    host_auxiliary_scheduler = _Scheduler()
    source_model = argparse.Namespace(parameters=lambda: ())
    monkeypatch.setattr(
        full_runner,
        "_distributed_complete_source_gradient_metrics",
        lambda *args, **kwargs: {
            "all_finite_and_present": True,
            "global_l2_norm": 7.0,
            "global_max_abs": 2.0,
            "rank_failures": [],
        },
    )
    monkeypatch.setattr(
        full_runner,
        "clip_lingbot_distributed_l2_grad_norm_",
        lambda *args, **kwargs: 7.0,
    )
    monkeypatch.setattr(
        full_runner,
        "_optimizer_attempt",
        lambda **kwargs: (3, {"preclip_global_norm": 5.0}),
    )

    successful, metrics = full_runner._joint_host_source_optimizer_attempt(
        policy=object(),
        host_optimizer=object(),
        source_model=source_model,
        source_optimizer=source_optimizer,
        source_scheduler=source_scheduler,
        host_auxiliary_scheduler=host_auxiliary_scheduler,
        source_update_arm=source_update_arm,
        global_step=2,
        max_grad_norm=1.0,
        device=torch.device("cpu"),
        dist=object(),
        torch_module=torch,
    )

    assert successful == 3
    assert source_optimizer.step_count == expected_source_steps
    assert source_scheduler.step_count == expected_scheduler_steps
    assert source_optimizer.zero_count == 1
    assert host_auxiliary_scheduler.step_count == 1
    assert metrics["source_update_arm"] == source_update_arm
    assert metrics["source_update_applied"] is (source_update_arm == "joint")
    assert metrics["source_preclip_global_norm"] == 7.0


def test_scientific_terminal_boundary_respects_the_2k_checkpoint_cadence() -> None:
    assert not _scientific_terminal_checkpoint_due(stop_after_step=250, global_step=250)
    assert not _scientific_terminal_checkpoint_due(stop_after_step=250, global_step=249)
    assert _scientific_terminal_checkpoint_due(stop_after_step=2_000, global_step=2_000)
    assert _scientific_terminal_checkpoint_due(stop_after_step=30_000, global_step=30_000)
    with pytest.raises(ValueError, match="outside"):
        _scientific_terminal_checkpoint_due(stop_after_step=0, global_step=0)


def test_fixed_entity_visuals_select_distinct_tasks_per_partition() -> None:
    items = (
        argparse.Namespace(partition="validation", task_key="a", sample_key="v0"),
        argparse.Namespace(partition="validation", task_key="a", sample_key="v1"),
        argparse.Namespace(partition="validation", task_key="b", sample_key="v2"),
        argparse.Namespace(partition="heldout", task_key="c", sample_key="h0"),
        argparse.Namespace(partition="heldout", task_key="d", sample_key="h1"),
    )
    assert _evaluation_visual_sample_keys(
        items,
        partitions=("validation", "heldout"),
        per_partition=2,
    ) == ("v0", "v2", "h0", "h1")


def test_native_videomt_anchor_summary_is_object_weighted_and_keeps_ranked_metrics() -> None:
    def ranked_proposals(
        *,
        mean_soft_iou: float,
        mean_binary_iou: float,
        recall_at_50: float,
    ) -> list[dict[str, object]]:
        return [
            {
                "top_k": top_k,
                "query_indices": [0],
                "soft_ious": [mean_soft_iou],
                "binary_ious": [mean_binary_iou],
                "foreground_probabilities": [0.5],
                "mean_soft_iou": mean_soft_iou,
                "mean_binary_iou": mean_binary_iou,
                "recall_at_50": recall_at_50,
            }
            for top_k in (10, 25, 50, 100, 200)
        ]

    samples = [
        {
            "partition": "heldout",
            "soft_ious": [0.2, 0.8],
            "binary_ious": [0.4, 0.9],
            "foreground_probabilities": [0.3, 0.7],
            "ranked_proposals": ranked_proposals(
                mean_soft_iou=0.25,
                mean_binary_iou=0.35,
                recall_at_50=0.5,
            ),
        },
        {
            "partition": "heldout",
            "soft_ious": [0.5],
            "binary_ious": [0.6],
            "foreground_probabilities": [0.9],
            "ranked_proposals": ranked_proposals(
                mean_soft_iou=0.45,
                mean_binary_iou=0.55,
                recall_at_50=1.0,
            ),
        },
    ]
    summary = full_runner._summarize_native_videomt_anchor_partition(
        samples,
        partition="heldout",
    )

    assert summary["sample_count"] == 2
    assert summary["object_observation_count"] == 3
    assert summary["mean_soft_iou"] == pytest.approx(0.5)
    assert summary["recall_at_50"] == pytest.approx(2 / 3)
    assert summary["ranked_proposals"]["10"]["mean_soft_iou"] == pytest.approx(0.35)


def test_native_videomt_anchor_summary_rejects_legacy_mapping_proposals() -> None:
    with pytest.raises(TypeError, match="source list ABI"):
        full_runner._summarize_native_videomt_anchor_partition(
            [
                {
                    "partition": "heldout",
                    "soft_ious": [0.8],
                    "binary_ious": [0.9],
                    "foreground_probabilities": [0.7],
                    "ranked_proposals": {"10": {}},
                }
            ],
            partition="heldout",
        )


def test_videomt_stage_pq_mode_is_explicit_and_fail_closed() -> None:
    args = argparse.Namespace(
        videomt_stage_pq_mode=VIDEOMT_STAGE_PQ_DISABLED,
        videomt_checkpoint=None,
        videomt_dinov3_bundle=None,
        videomt_idle_placement=VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
        videomt_fsdp2_placement=VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED,
        posterior_architecture="two_pass_v3",
    )
    _validate_videomt_stage_pq_args(args)

    args.videomt_checkpoint = Path("checkpoint.pth")
    with pytest.raises(ValueError, match="disabled.*forbids"):
        _validate_videomt_stage_pq_args(args)
    args.videomt_checkpoint = None
    args.videomt_idle_placement = VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
    with pytest.raises(ValueError, match="disabled.*idle placement"):
        _validate_videomt_stage_pq_args(args)
    args.videomt_idle_placement = VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT
    args.videomt_fsdp2_placement = VIDEOMT_FSDP2_PLACEMENT_CPU_OFFLOAD
    with pytest.raises(ValueError, match="disabled.*FSDP2 placement"):
        _validate_videomt_stage_pq_args(args)
    args.videomt_fsdp2_placement = VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED

    args.videomt_stage_pq_mode = VIDEOMT_STAGE_PQ_FROZEN_RELEASED_EVAL_C5
    args.videomt_idle_placement = VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT
    args.videomt_checkpoint = None
    with pytest.raises(ValueError, match="requires checkpoint and DINOv3"):
        _validate_videomt_stage_pq_args(args)

    args.videomt_checkpoint = Path("checkpoint.pth")
    args.videomt_dinov3_bundle = Path("dinov3")
    _validate_videomt_stage_pq_args(args)
    args.videomt_idle_placement = VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
    _validate_videomt_stage_pq_args(args)
    args.posterior_architecture = "layerwise_v2"
    with pytest.raises(ValueError, match="requires the two_pass_v3"):
        _validate_videomt_stage_pq_args(args)


def test_videomt_stage_pqm_requires_the_complete_multimodal_host_contract() -> None:
    args = _causal_args("current_frame_branch")
    args.videomt_stage_pq_mode = VIDEOMT_STAGE_PQM_FROZEN_RELEASED_EVAL_C5
    args.videomt_checkpoint = Path("checkpoint.pth")
    args.videomt_dinov3_bundle = Path("dinov3")
    args.videomt_idle_placement = VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT
    args.posterior_architecture = "two_pass_v3"
    args.dense_evidence_mode = "none"
    with pytest.raises(ValueError, match="requires V-JEPA, AnyTouch and Sonata"):
        _validate_videomt_stage_pq_args(args)

    args.dense_evidence_mode = "calvin_full_v1"
    _validate_videomt_stage_pq_args(args)
    contract = _execution_contract(args)["videomt_stage_pq"]
    assert contract["host_boundary"] == (
        "latest_all_200_queries_plus_complete_class_mask_relation_"
        "no_selection_pooling_or_local_decoder"
    )


def test_adapted_videomt_stage_pqm_requires_an_authenticated_full_checkpoint() -> None:
    args = _causal_args("current_frame_branch")
    args.videomt_stage_pq_mode = VIDEOMT_STAGE_PQM_FROZEN_ADAPTED_EVAL_C5
    args.videomt_checkpoint = Path("released.pth")
    args.videomt_dinov3_bundle = Path("dinov3")
    args.videomt_adapted_checkpoint = None
    args.videomt_adapted_checkpoint_sha256 = None
    args.videomt_idle_placement = VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
    args.posterior_architecture = "two_pass_v3"
    args.dense_evidence_mode = "calvin_full_v1"
    with pytest.raises(ValueError, match="adapted checkpoint and SHA-256"):
        _validate_videomt_stage_pq_args(args)

    args.videomt_adapted_checkpoint = Path("adapted.pt")
    args.videomt_adapted_checkpoint_sha256 = "a" * 64
    _validate_videomt_stage_pq_args(args)
    contract = _execution_contract(args)["videomt_stage_pq"]
    assert contract["donor_execution_mode"] == (
        "complete_calvin_adapted_eval_graph_fp32_frozen"
    )
    assert contract["checkpoint_sha256"] == "a" * 64
    assert contract["released_checkpoint_sha256"] == VIDEOMT_RELEASED_CHECKPOINT_SHA256

    args.videomt_stage_pq_mode = VIDEOMT_STAGE_PQMR_FROZEN_ADAPTED_EVAL_C5
    _validate_videomt_stage_pq_args(args)
    contract = _execution_contract(args)["videomt_stage_pq"]
    assert "mask_embeddings_and_dense_mask_features" in contract["host_boundary"]
    assert contract["pixel_row_composition"].startswith("row_mask_logit=dot")
    assert contract["local_object_selector_decoder_or_lifecycle_head"] is False

    args.videomt_stage_pq_mode = VIDEOMT_STAGE_PQRF_FROZEN_ADAPTED_EVAL_C5
    _validate_videomt_stage_pq_args(args)
    contract = _execution_contract(args)["videomt_stage_pq"]
    assert "complete_frozen_blocks20_23" in contract["host_boundary"]
    assert contract["pixel_row_composition"].startswith("prepend(tied_projected")
    assert contract["local_object_selector_decoder_or_lifecycle_head"] is False


def test_videomt_stage_pq_asset_receipt_authenticates_the_declared_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "released.pth"
    checkpoint.write_bytes(b"released-checkpoint")
    checkpoint_sha256 = full_runner._sha256(checkpoint)
    bundle = tmp_path / "dinov3"
    bundle.mkdir()
    config = bundle / "config.json"
    config.write_text('{"model_type":"dinov3_vit"}\n', encoding="utf-8")
    (bundle / "model.safetensors").write_bytes(b"converted-weights")
    (bundle / "conversion_receipt.json").write_text(
        json.dumps(
            {
                "published_checkpoint": {"sha256": checkpoint_sha256},
                "converted_tensor_count": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        full_runner,
        "VIDEOMT_RELEASED_CHECKPOINT_BYTES",
        checkpoint.stat().st_size,
    )
    monkeypatch.setattr(
        full_runner,
        "VIDEOMT_RELEASED_CHECKPOINT_SHA256",
        checkpoint_sha256,
    )
    monkeypatch.setattr(full_runner, "VIDEOMT_DINOV3_CONFIG_BYTES", config.stat().st_size)
    monkeypatch.setattr(
        full_runner,
        "VIDEOMT_DINOV3_CONFIG_SHA256",
        full_runner._sha256(config),
    )
    args = argparse.Namespace(
        videomt_stage_pq_mode=VIDEOMT_STAGE_PQ_FROZEN_RELEASED_EVAL_C5,
        videomt_checkpoint=checkpoint,
        videomt_dinov3_bundle=bundle,
    )

    receipt = full_runner._videomt_stage_pq_asset_receipt(args)

    assert receipt["active"] is True
    assert receipt["checkpoint_sha256"] == checkpoint_sha256
    assert receipt["converted_tensor_count"] == 4


def test_videomt_stage_pq_contract_forbids_a_simplified_donor_boundary() -> None:
    args = _causal_args("current_frame_branch")
    args.videomt_stage_pq_mode = VIDEOMT_STAGE_PQ_FROZEN_RELEASED_EVAL_C5
    args.videomt_checkpoint = Path("checkpoint.pth")
    args.videomt_dinov3_bundle = Path("dinov3")
    args.videomt_idle_placement = VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
    args.posterior_architecture = "two_pass_v3"

    contract = _execution_contract(args)["videomt_stage_pq"]

    assert contract == {
        "mode": VIDEOMT_STAGE_PQ_FROZEN_RELEASED_EVAL_C5,
        "active": True,
        "donor_execution_mode": "complete_released_eval_graph_fp32_frozen",
        "temporal_adapter": "five_real_raw_episode_frames_t_minus_4_through_t_no_padding",
        "host_boundary": (
            "latest_all_200_queries_no_selection_pooling_resampling_or_second_norm"
        ),
        "idle_placement": VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS,
        "idle_placement_changes_tensor_semantics": False,
        "fsdp2_placement": VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED,
        "fsdp2_placement_changes_model_semantics": False,
        "short_prefix_policy": "empty_absent_stream_counted",
        "query_count": 200,
        "query_width": 1024,
        "checkpoint_sha256": VIDEOMT_RELEASED_CHECKPOINT_SHA256,
        "released_checkpoint_sha256": VIDEOMT_RELEASED_CHECKPOINT_SHA256,
    }


def test_adr207_execution_contract_declares_trainable_future_source_semantics() -> None:
    args = _causal_args("current_frame_branch")
    args.causal_ablation_mode = "none"
    args.videomt_stage_pq_mode = full_runner.VIDEOMT_NATIVE_TRAINABLE_ADAPTED_CAUSAL_C5
    args.videomt_checkpoint = Path("released.pth")
    args.videomt_dinov3_bundle = Path("dinov3")
    args.videomt_adapted_checkpoint = Path("adapted.pt")
    args.videomt_adapted_checkpoint_sha256 = "a" * 64
    args.videomt_idle_placement = VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT
    args.videomt_fsdp2_placement = VIDEOMT_FSDP2_PLACEMENT_CPU_OFFLOAD
    args.posterior_architecture = "two_pass_v3"
    args.picf_architecture_profile = ADR207_ARCHITECTURE_PROFILE
    args.videomt_source_update_arm = "frozen-coordinate-control"
    args.attention_implementation = full_runner.LINGBOT_ATTENTION_FLEX_CACHED
    args.dense_evidence_mode = "calvin_full_v1"
    args.capacity = 200

    execution_contract = _execution_contract(args)
    contract = execution_contract["videomt_stage_pq"]

    assert execution_contract["attention_implementation"] == "flex_cached"
    assert contract["donor_execution_mode"] == (
        "complete_calvin_adapted_train_graph_fsdp2_joint"
    )
    assert contract["temporal_adapter"] == (
        "five_real_raw_episode_frames_t_through_t_plus_4_no_padding"
    )
    assert contract["short_prefix_policy"] == (
        "stream_domain_requires_four_future_frames"
    )
    assert contract["query_count"] == 200
    assert contract["fsdp2_placement"] == VIDEOMT_FSDP2_PLACEMENT_CPU_OFFLOAD
    assert contract["fsdp2_placement_changes_model_semantics"] is False
    assert contract["host_boundary"] == (
        "all_200_native_source_queries_and_source_masks_same_index_as_"
        "shared_host_posterior_rows_no_assignment_or_reverse_projection"
    )
    assert contract["pixel_row_composition"] == (
        "source_mask(query_i,pixel);posterior_row_i=same_native_query_"
        "address;no_host_to_source_mask_decode"
    )
    assert contract["posterior_query_integration"] == (
        "same_index_source_to_host_width_projection"
    )
    assert contract["source_update_arm"] == "frozen-coordinate-control"
    assert contract["source_forward_and_backward_graph"] == (
        "unchanged_complete_joint_graph"
    )
    assert _execution_contract(args)["object_action_information_contract"] == (
        "native_source_query_i_as_shared_host_posterior_row_i_to_official_action"
    )


def test_full_runner_executes_the_exact_videomt_graph_without_query_reduction() -> None:
    source = Path("tools/run_lingbot_vla2_task_independent_full.py").read_text(
        encoding="utf-8"
    )

    for required in (
        "load_exact_videomt(",
        "dtype=torch.float32",
        "videomt_runtime.requires_grad_(False)",
        "VidEoMTStagePQMR(videomt_runtime)",
        "VidEoMTStagePQM(videomt_runtime)",
        "VidEoMTStageP(videomt_runtime)",
        "with_videomt_row_mask_query_modality_spec(modality_specs)",
        "with_videomt_query_modality_spec(modality_specs)",
        "prepare_calvin_stage_pq_c5(",
        "with torch.no_grad():",
        "host_dtype=torch.bfloat16",
        "resume=False",
        '"host_injected_output": (',
        "VIDEOMT_STAGE_PQM_HOST_OUTPUT",
        "object_query_spatial_specs=object_query_spatial_specs",
        "enable_frozen_vision_offload=(",
        "FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD",
    ):
        assert required in source
    assert "CALVIN_FULL_DENSE_MODALITIES = (\"anytouch\", \"sonata\", \"vjepa\")" in source
    assert "VIDEOMT_QUERY_MODALITY" not in source
    assert "verify_muon_collective_hotfix(" in source
    assert "validate_prepared_native_source_with_muon_collective_hotfix(" in source
    assert "verify_selective_class_cpu_offload(" in source
    assert "validate_prepared_native_source_with_selective_class_cpu_offload(" in source
    assert (
        "verify_selective_trainable_vision_with_vlm_selective_class_cpu_offload("
        in source
    )
    assert (
        "validate_prepared_native_source_with_trainable_vision_and_vlm_selective_class_offload("
        in source
    )
    assert "verify_selective_frozen_vision_offload(" in source
    assert "validate_prepared_native_source_with_selective_frozen_vision_offload(" in source
    assert '"runtime_hotfix_sha256": patch_report["runtime_hotfix_sha256"]' in source


def test_adr178_profile_is_direct_full_modal_and_fail_closed() -> None:
    args = argparse.Namespace(
        picf_architecture_profile=ADR178_ARCHITECTURE_PROFILE,
        posterior_architecture="two_pass_v3",
        dense_evidence_mode="calvin_full_v1",
        dense_token_bridge="lingbot_task_token_resampler_v1",
        task_query_count=0,
        relation_supervision_layers=ADR178_RELATION_SUPERVISION_LAYERS,
        learning_rate=1e-4,
        picf_learning_rate_multiplier=2.0,
        modality_bridge_learning_rate_multiplier=0.5,
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
    )

    _validate_picf_architecture_profile(args)
    assert ADR178_NATIVE_ATTENTION_WEIGHT == 0.001
    assert _picf_optimizer_learning_rate_stratification_active(args)
    args.task_query_count = 1
    with pytest.raises(ValueError, match="changed frozen fields"):
        _validate_picf_architecture_profile(args)


def test_legacy_profile_does_not_rewrite_official_optimizer_groups() -> None:
    args = argparse.Namespace(picf_architecture_profile="legacy")

    assert not _picf_optimizer_learning_rate_stratification_active(args)


def test_adr193_profile_is_full_modal_without_a_private_task_or_action_selector() -> None:
    args = argparse.Namespace(
        picf_architecture_profile=ADR193_ARCHITECTURE_PROFILE,
        posterior_architecture="two_pass_v3",
        dense_evidence_mode="calvin_full_v1",
        dense_token_bridge="lingbot_task_token_resampler_v1",
        task_query_count=0,
        relation_supervision_layers=ADR178_RELATION_SUPERVISION_LAYERS,
        learning_rate=1e-4,
        picf_learning_rate_multiplier=2.0,
        modality_bridge_learning_rate_multiplier=0.5,
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
    )

    _validate_picf_architecture_profile(args)
    assert _picf_optimizer_learning_rate_stratification_active(args)
    assert _adr193_implicit_multimodal_anchor_active(args)
    assert not _adr178_direct_action_posterior_active(args)
    args.task_query_count = 1
    with pytest.raises(ValueError, match="changed frozen fields"):
        _validate_picf_architecture_profile(args)


def test_anchor_checks_are_registered_per_architecture_profile() -> None:
    args = argparse.Namespace(
        picf_architecture_profile=ADR193_ARCHITECTURE_PROFILE,
    )

    assert ADR193_ANCHOR_CHECK_STEPS == (50, 100)
    assert _anchor_evaluation_due(args=args, global_step=50)
    assert _anchor_evaluation_due(args=args, global_step=100)
    assert not _anchor_evaluation_due(args=args, global_step=49)
    assert not _anchor_evaluation_due(args=args, global_step=101)

    contract_args = _causal_args("current_frame_branch")
    contract_args.picf_architecture_profile = ADR193_ARCHITECTURE_PROFILE
    assert _execution_contract(contract_args)["anchor_evaluation_steps"] == [50, 100]

    args.picf_architecture_profile = ADR207_ARCHITECTURE_PROFILE
    contract_args.picf_architecture_profile = ADR207_ARCHITECTURE_PROFILE
    assert ADR207_ANCHOR_CHECK_STEPS == (50, 100, 200)
    assert _anchor_evaluation_due(args=args, global_step=50)
    assert _anchor_evaluation_due(args=args, global_step=100)
    assert _anchor_evaluation_due(args=args, global_step=200)
    assert not _anchor_evaluation_due(args=args, global_step=250)
    assert _execution_contract(contract_args)["anchor_evaluation_steps"] == [50, 100, 200]

    args.picf_architecture_profile = ADR178_ARCHITECTURE_PROFILE
    assert not _anchor_evaluation_due(args=args, global_step=50)
    contract_args.picf_architecture_profile = ADR178_ARCHITECTURE_PROFILE
    assert _execution_contract(contract_args)["anchor_evaluation_steps"] == []

    args.picf_architecture_profile = ADR225_ARCHITECTURE_PROFILE
    contract_args.picf_architecture_profile = ADR225_ARCHITECTURE_PROFILE
    assert _anchor_evaluation_due(args=args, global_step=50)
    assert _execution_contract(contract_args)["anchor_evaluation_steps"] == [50, 100, 200]


def test_adr204_profile_keeps_implicit_anchor_contract_with_final_readout_only() -> None:
    args = argparse.Namespace(
        picf_architecture_profile=ADR204_ARCHITECTURE_PROFILE,
        posterior_architecture="two_pass_v3",
        dense_evidence_mode="calvin_full_v1",
        dense_token_bridge="lingbot_task_token_resampler_v1",
        task_query_count=0,
        relation_supervision_layers=(),
        learning_rate=1e-4,
        picf_learning_rate_multiplier=2.0,
        modality_bridge_learning_rate_multiplier=0.5,
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
    )

    _validate_picf_architecture_profile(args)
    assert _adr193_implicit_multimodal_anchor_active(args)
    assert _adr204_full_source_final_only_active(args)
    assert _picf_optimizer_learning_rate_stratification_active(args)
    assert _anchor_evaluation_due(args=args, global_step=50)
    args.relation_supervision_layers = ADR178_RELATION_SUPERVISION_LAYERS
    with pytest.raises(ValueError, match="changed frozen fields"):
        _validate_picf_architecture_profile(args)


def test_adr205_profile_reuses_final_only_contract_with_released_query_propagation() -> None:
    args = argparse.Namespace(
        picf_architecture_profile=ADR205_ARCHITECTURE_PROFILE,
        posterior_architecture="two_pass_v3",
        dense_evidence_mode="calvin_full_v1",
        dense_token_bridge="lingbot_task_token_resampler_v1",
        task_query_count=0,
        relation_supervision_layers=(),
        learning_rate=1e-4,
        picf_learning_rate_multiplier=2.0,
        modality_bridge_learning_rate_multiplier=0.5,
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
    )

    _validate_picf_architecture_profile(args)
    assert _adr193_implicit_multimodal_anchor_active(args)
    assert not _adr204_full_source_final_only_active(args)
    assert _adr205_released_query_propagation_active(args)
    assert _picf_optimizer_learning_rate_stratification_active(args)
    assert _anchor_evaluation_due(args=args, global_step=50)
    args.relation_supervision_layers = ADR178_RELATION_SUPERVISION_LAYERS
    with pytest.raises(ValueError, match="changed frozen fields"):
        _validate_picf_architecture_profile(args)


def test_adr207_profile_is_one_complete_200_query_full_modal_contract() -> None:
    args = argparse.Namespace(
        picf_architecture_profile=ADR207_ARCHITECTURE_PROFILE,
        posterior_architecture="two_pass_v3",
        dense_evidence_mode="calvin_full_v1",
        dense_token_bridge="exact_tokens_v1",
        videomt_stage_pq_mode="trainable-adapted-native-query-causal-c5",
        videomt_idle_placement=VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
        videomt_fsdp2_placement=VIDEOMT_FSDP2_PLACEMENT_CPU_OFFLOAD,
        trainable_scope=TRAINABLE_SCOPE_FULL_HOST,
        capacity=200,
        task_query_count=0,
        relation_supervision_layers=(),
        learning_rate=1e-4,
        picf_learning_rate_multiplier=1.0,
        modality_bridge_learning_rate_multiplier=1.0,
        entity_weight=0.0,
        predictive_weight=0.0,
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
        source_mask_probability=0.0,
        lingbot_compile_mode=LINGBOT_COMPILE_UPSTREAM_DEFAULT,
    )

    _validate_picf_architecture_profile(args)
    args.capacity = 16
    with pytest.raises(ValueError, match="changed frozen fields"):
        _validate_picf_architecture_profile(args)

    contract_args = _causal_args("current_frame_branch")
    contract_args.picf_architecture_profile = ADR207_ARCHITECTURE_PROFILE
    contract_args.posterior_architecture = "two_pass_v3"
    contract_args.capacity = 200
    contract_args.acceptance_mode = "none"
    contract_args.lingbot_compile_mode = LINGBOT_COMPILE_UPSTREAM_DEFAULT
    contract = _execution_contract(contract_args)
    assert contract["lingbot_compile_mode"] == LINGBOT_COMPILE_UPSTREAM_DEFAULT
    assert contract["native_query_modality_intervention_steps"] == [200, 2_000]
    assert contract["native_query_modality_interventions"] == list(
        ADR207_MODALITY_INTERVENTIONS
    )
    assert contract["native_query_modality_intervention_scope"] == (
        "fixed_source_queries_frozen_training_stream_wiring"
    )


def test_adr225_profile_replaces_only_the_random_query_projection() -> None:
    args = argparse.Namespace(
        picf_architecture_profile=ADR225_ARCHITECTURE_PROFILE,
        posterior_architecture="two_pass_v3",
        dense_evidence_mode="calvin_full_v1",
        dense_token_bridge="exact_tokens_v1",
        videomt_stage_pq_mode="trainable-adapted-native-query-causal-c5",
        videomt_idle_placement=VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
        trainable_scope=TRAINABLE_SCOPE_FULL_HOST,
        capacity=200,
        task_query_count=0,
        relation_supervision_layers=(),
        learning_rate=1e-4,
        picf_learning_rate_multiplier=1.0,
        modality_bridge_learning_rate_multiplier=1.0,
        entity_weight=0.0,
        predictive_weight=0.0,
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
        source_mask_probability=0.0,
        lingbot_compile_mode="disabled",
    )

    _validate_picf_architecture_profile(args)
    assert full_runner._adr225_pretrained_object_memory_active(args)
    assert full_runner._objective_profile(args) == (
        "adr225_pretrained_native_object_memory_joint_action"
    )
    assert full_runner._object_action_information_contract(args) == (
        "videomt_mask_pooled_native_qwen3_object_memory_to_official_action"
    )

    contract_args = _causal_args("current_frame_branch")
    contract_args.picf_architecture_profile = ADR225_ARCHITECTURE_PROFILE
    contract_args.posterior_architecture = "two_pass_v3"
    contract_args.capacity = 200
    contract_args.videomt_stage_pq_mode = (
        full_runner.VIDEOMT_NATIVE_TRAINABLE_ADAPTED_CAUSAL_C5
    )
    contract_args.videomt_adapted_checkpoint_sha256 = "a" * 64
    contract = _execution_contract(contract_args)["videomt_stage_pq"]
    assert contract["host_boundary"] == (
        "all_200_source_masks_pool_same_index_pretrained_qwen3_visual_cells_"
        "through_exact_copied_native_merger_mlp_as_shared_host_posterior_"
        "rows_with_original_qwen3_prefix_retained_no_selection"
    )
    assert contract["posterior_query_integration"] == (
        "same_index_pretrained_qwen3_object_memory_no_random_source_query_projection"
    )
    assert contract["object_memory_source_primitive"] == (
        "unipixel_cache_native_merger_input_mask_mean_then_native_merger_mlp"
    )
    assert contract["object_memory_external_inference_model"] is None
    assert contract["object_memory_required_ablation"] == (
        "source_faithful_binary_mask_mean_vs_soft_posterior_mean"
    )


def test_adr225_step_report_fails_closed_and_summarizes_mask_support() -> None:
    context = argparse.Namespace(
        object_memory_support_mass=torch.tensor([[0.25, 0.0], [0.50, 0.75]]),
        object_memory_query_valid=torch.tensor([[True, False], [True, True]]),
        object_memory_capture_generation=7,
    )

    report = _pretrained_object_memory_step_report(
        context,
        capacity=2,
        torch_module=torch,
    )

    assert report["schema"] == "picf-next.pretrained-object-memory-step/v1"
    assert report["active"] is True
    assert report["capture_generation"] == 7
    assert report["query_capacity"] == 2
    assert report["valid_query_count"] == 3
    assert report["valid_query_count_by_batch"] == [1, 2]
    assert report["mean_valid_support_mass"] == pytest.approx(0.5)
    assert report["maximum_valid_support_mass"] == pytest.approx(0.75)
    assert report["zero_support_valid_query_count"] == 0

    context.object_memory_capture_generation = 0
    with pytest.raises(ValueError, match="capture generation"):
        _pretrained_object_memory_step_report(
            context,
            capacity=2,
            torch_module=torch,
        )


def test_adr193_step250_gate_also_runs_the_matched_action_curve() -> None:
    args = _acceptance_args("none")
    args.phase = "fresh"
    args.load_global_step = 0
    args.stop_after_step = 250
    args.picf_architecture_profile = ADR193_ARCHITECTURE_PROFILE

    assert _registered_action_evaluation_steps(args) == (0, 20, 100, 200)
    assert _action_evaluation_active(args)
    assert _execution_contract(args)["action_evaluation_steps"] == (0, 20, 100, 200)


def test_adr178_step250_combines_action_curve_causal_gate_and_visuals() -> None:
    args = argparse.Namespace(
        acceptance_mode="none",
        phase="fresh",
        picf_architecture_profile=ADR178_ARCHITECTURE_PROFILE,
        posterior_architecture="two_pass_v3",
        stop_after_step=250,
    )

    assert _registered_action_evaluation_steps(args) == (0, 20, 100, 200)
    assert _action_evaluation_active(args)


def test_direct_action_targets_use_exact_hungarian_rows_and_fail_closed() -> None:
    weights, valid, audit = _direct_action_posterior_targets(
        bindings_by_batch=(
            (("movable/block_blue", 3), ("part/table/slide_link", 5)),
            (("movable/block_blue", 2),),
        ),
        structural_target_requests=(
            argparse.Namespace(task_key="push_blue_block_left"),
            argparse.Namespace(task_key="push_into_drawer"),
        ),
        capacity=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        torch_module=torch,
    )

    assert valid.tolist() == [True, False]
    assert weights[0, 3].item() == 1.0
    assert torch.count_nonzero(weights[1]) == 0
    assert audit[0]["selected_rows"] == [3]
    assert audit[1]["target_valid"] is False


def test_adr177_launcher_defaults_to_the_accepted_30k_two_card_scope() -> None:
    launcher = Path("adr177/run_upgraded_full_modal.sh").read_text(encoding="utf-8")

    assert "STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-30000}" in launcher
    assert "STOP_AFTER_STEP <= 30000" in launcher
    assert "TRAINABLE_SCOPE=${PICF_TRAINABLE_SCOPE:-frozen-vision-host}" in launcher
    assert "CUDA_ALLOCATOR=${PICF_CUDA_ALLOCATOR:-expandable-segments}" in launcher
    assert "/mnt/picf-next/adr177/contracts/full-modal-2gpu-30k-v1" in launcher
    assert "/mnt/picf-next/adr150/caches/calvin-official-30k-v1" in launcher
    for modality_cache in ("ANYTOUCH_CACHE", "SONATA_CACHE", "VJEPA_CACHE"):
        assert f'--dense-evidence-cache-root "${modality_cache}"' in launcher
    for supplement in (
        "ANYTOUCH_SUPPLEMENT",
        "SONATA_SUPPLEMENT",
        "VJEPA_SUPPLEMENT",
    ):
        assert f'--dense-evidence-supplement-cache-root "${supplement}"' in launcher


def test_adr178_launcher_uses_direct_action_posterior_without_task_queries() -> None:
    launcher = Path("adr178/run_direct_action_posterior_full_modal.sh").read_text(encoding="utf-8")

    assert "STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-250}" in launcher
    assert "adr178_direct_action_posterior_full_modal_v1" in launcher
    assert "--task-query-count 0" in launcher
    assert 'RELATION_SUPERVISION_LAYERS=${PICF_RELATION_SUPERVISION_LAYERS-8,17,26}' in launcher
    assert '--relation-supervision-layers "$RELATION_SUPERVISION_LAYERS"' in launcher
    assert "/mnt/picf-next/adr177/contracts/full-modal-2gpu-30k-v1" in launcher
    for modality_cache in ("ANYTOUCH_CACHE", "SONATA_CACHE", "VJEPA_CACHE"):
        assert f'--dense-evidence-cache-root "${modality_cache}"' in launcher


def test_adr193_launcher_selects_the_clean_implicit_anchor_profile() -> None:
    launcher = Path("adr193/run_implicit_multimodal_anchor_2gpu.sh").read_text(encoding="utf-8")

    assert "PICF_ARCHITECTURE_PROFILE:-adr193_implicit_multimodal_anchor_v1" in launcher
    assert "PICF_STOP_AFTER_STEP:-250" in launcher
    assert "adr193/source-freezes/adr193-implicit-anchor-v2" in launcher
    assert "../adr178/run_direct_action_posterior_full_modal.sh" in launcher


def test_adr204_final_only_launcher_disables_only_intermediate_relation_reads() -> None:
    launcher = Path(
        "adr204/run_full_source_row_refinement_final_only_2gpu.sh"
    ).read_text(encoding="utf-8")

    assert "adr204_full_source_final_only_v1" in launcher
    assert "export PICF_RELATION_SUPERVISION_LAYERS=" in launcher
    assert "full-source-row-refinement-v5" in launcher
    assert "frozen-adapted-eval-causal-c5-pqrf" in launcher
    assert "../adr193/run_implicit_multimodal_anchor_2gpu.sh" in launcher


def test_adr205_launcher_uses_released_query_propagation_profile() -> None:
    launcher = Path("adr204/run_released_query_propagation_2gpu.sh").read_text(
        encoding="utf-8"
    )

    assert "adr205_released_query_propagation_v1" in launcher
    assert "export PICF_RELATION_SUPERVISION_LAYERS=" in launcher
    assert "released-query-propagation-v7-block-checkpoint" in launcher
    assert "frozen-adapted-eval-causal-c5-pqrf" in launcher
    assert "../adr193/run_implicit_multimodal_anchor_2gpu.sh" in launcher


def test_adr207_launcher_freezes_the_complete_native_query_contract() -> None:
    launcher = Path("adr207/run_native_videomt_query_posterior_2gpu.sh").read_text(
        encoding="utf-8"
    )
    base = Path("adr178/run_direct_action_posterior_full_modal.sh").read_text(
        encoding="utf-8"
    )

    required = (
        "source-freezes/native-query-posterior-v18",
        "source-freeze.receipt.json",
        "status --porcelain=v1 --untracked-files=all",
        "PICF_ARCHITECTURE_PROFILE=adr207_native_videomt_query_posterior_v1",
        "lingbot-vla-v2-adr207-native-muon",
        "PICF_VIDEOMT_STAGE_PQ_MODE=trainable-adapted-native-query-causal-c5",
        "PICF_VIDEOMT_FSDP2_PLACEMENT:-cpu-offload",
        "PICF_TRAINABLE_SCOPE=full-host",
        "PICF_DENSE_TOKEN_BRIDGE=exact_tokens_v1",
        "PICF_CAPACITY=200",
        "PICF_RELATION_SUPERVISION_LAYERS=",
        "PICF_PICF_LEARNING_RATE_MULTIPLIER=1.0",
        "PICF_MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER=1.0",
        "PICF_ENTITY_WEIGHT=0.0",
        "PICF_PREDICTIVE_WEIGHT=0.0",
        "PICF_SOURCE_MASK_PROBABILITY=0.0",
        "PICF_STOP_AFTER_STEP:-30000",
        "PICF_ATTENTION_IMPLEMENTATION=flex_cached",
        "PICF_LINGBOT_COMPILE_MODE=upstream-default",
        "PICF_USE_DENSE_SUPPLEMENT=0",
        "PICF_USE_CURRENT_GRID_CACHE=0",
        "videomt-calvin-adapted-step250-v1.pt",
        "4437d8632c4e3877adcf5cfec5bf6e673445ad9d3d2de3a3afdd924651b5bd5d",
        "videomt-torch280-functorch-v1",
        "857d364103403df8aafc97674e97e518acf781bc8fc080840ca11c99f25aacd0",
        'assert torch.__version__ == functorch.__version__ == "2.8.0+cu128"',
    )
    for value in required:
        assert value in launcher
    assert "trainable-adapted-native-query-causal-c5" in base
    assert '--dense-token-bridge "$DENSE_TOKEN_BRIDGE"' in base
    assert '--capacity "$CAPACITY"' in base
    assert 'PICF_ATTENTION_IMPLEMENTATION:-eager' in base
    assert '--attention-implementation "$ATTENTION_IMPLEMENTATION"' in base
    assert '--lingbot-compile-mode "$LINGBOT_COMPILE_MODE"' in base
    assert '"${CURRENT_GRID_ARGS[@]}"' in base
    assert '"${DENSE_SUPPLEMENT_ARGS[@]}"' in base
    assert '--videomt-fsdp2-placement "$VIDEOMT_FSDP2_PLACEMENT"' in base
    assert 'PICF_VIDEOMT_SOURCE_UPDATE_ARM:-joint' in base
    assert '--videomt-source-update-arm "$VIDEOMT_SOURCE_UPDATE_ARM"' in base
    assert 'PHASE=${PICF_PHASE:-fresh}' in base
    assert 'LOAD_GLOBAL_STEP=${PICF_LOAD_GLOBAL_STEP:-0}' in base
    assert '--phase "$PHASE"' in base
    assert '--load-global-step "$LOAD_GLOBAL_STEP"' in base
    assert 'LOAD_GLOBAL_STEP=${2:-0}' in launcher
    assert "PICF_PHASE=resume" in launcher
    assert "LOAD_GLOBAL_STEP % 2000 == 0" in launcher


def test_adr225_launcher_changes_only_the_pretrained_object_memory_boundary() -> None:
    launcher = Path("adr225/run_pretrained_object_memory_2gpu.sh").read_text(
        encoding="utf-8"
    )
    freeze = Path("adr225/freeze_source.sh").read_text(encoding="utf-8")

    required = (
        "source-freezes/pretrained-object-memory-v1",
        "source-freeze.receipt.json",
        "status --porcelain=v1 --untracked-files=all",
        "PICF_ARCHITECTURE_PROFILE=adr225_pretrained_native_object_memory_v1",
        "PICF_VIDEOMT_STAGE_PQ_MODE=trainable-adapted-native-query-causal-c5",
        "PICF_VIDEOMT_FSDP2_PLACEMENT:-cpu-offload",
        "PICF_TRAINABLE_SCOPE=full-host",
        "PICF_DENSE_TOKEN_BRIDGE=exact_tokens_v1",
        "PICF_CAPACITY=200",
        "PICF_RELATION_SUPERVISION_LAYERS=",
        "PICF_PICF_LEARNING_RATE_MULTIPLIER=1.0",
        "PICF_MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER=1.0",
        "PICF_ENTITY_WEIGHT=0.0",
        "PICF_PREDICTIVE_WEIGHT=0.0",
        "PICF_SOURCE_MASK_PROBABILITY=0.0",
        "PICF_STOP_AFTER_STEP:-250",
        "PICF_ADR225_LONG_RUN_AUTHORIZED:-0",
        "STOP_AFTER_STEP > 2000",
        "explicit Gate-D authorization",
        "PICF_ATTENTION_IMPLEMENTATION=flex_cached",
        "PICF_LINGBOT_COMPILE_MODE=disabled",
        "PICF_USE_DENSE_SUPPLEMENT=0",
        "PICF_USE_CURRENT_GRID_CACHE=0",
        "/mnt/picf-next/adr207/contracts/native-query-posterior-${WORLD_SIZE}gpu-30k-v1",
        "/mnt/picf-next/adr207/caches/native-query-posterior-${WORLD_SIZE}gpu-30k-v1",
        "videomt-calvin-adapted-step250-v1.pt",
        "4437d8632c4e3877adcf5cfec5bf6e673445ad9d3d2de3a3afdd924651b5bd5d",
        "videomt-torch280-functorch-v1",
        "857d364103403df8aafc97674e97e518acf781bc8fc080840ca11c99f25aacd0",
        'assert torch.__version__ == functorch.__version__ == "2.8.0+cu128"',
        "adr178/run_direct_action_posterior_full_modal.sh",
    )
    for value in required:
        assert value in launcher
    assert 'LOAD_GLOBAL_STEP=${2:-0}' in launcher
    assert "PICF_PHASE=resume" in launcher
    assert "LOAD_GLOBAL_STEP % 2000 == 0" in launcher
    assert "status --porcelain=v1 --untracked-files=all" in freeze
    assert "source must be clean before an immutable freeze" in freeze


def test_adr207_restores_released_whole_model_compile_after_fsdp() -> None:
    source = Path("tools/run_lingbot_vla2_task_independent_full.py").read_text(
        encoding="utf-8"
    )

    fsdp = source.index("policy = build_parallelize_model(")
    compile_model = source.index("policy = torch.compile(policy)")
    optimizer = source.index("build_lingbot_official_optimizer(", compile_model)
    assert fsdp < compile_model < optimizer
    assert 'merged["use_compile"] = (' in source
    assert '"ordering": "fsdp2_then_whole_model_compile_then_optimizer"' in source


def test_adr207_contract_builder_filters_every_training_sample_for_future_four() -> None:
    builder = Path("adr207/prepare_contracts_2gpu.sh").read_text(encoding="utf-8")

    assert "source-freezes/native-query-posterior-v18" in builder
    assert "source-freeze.receipt.json" in builder
    assert 'git -C "$REPO" status --porcelain=v1 --untracked-files=all' in builder
    assert "--physical-event-stream" in builder
    assert builder.count("--minimum-future-source-frames 4") == 2
    assert "--allow-reference-budget-change" in builder
    assert '--global-batch-size "$WORLD_SIZE"' in builder
    assert "--total-steps 30000" in builder
    assert '--world-size "$WORLD_SIZE"' in builder
    assert 'WORLD_SIZE=${PICF_WORLD_SIZE:-2}' in builder
    assert '"global_batch_size": world_size' in builder
    assert '"world_size": world_size' in builder
    assert "--comparison-id lingbot-vla2-native-picf-full" in builder
    assert "picf-next.adr207-contract-freeze/v1" in builder


def test_adr207_matched_lingbot_uses_the_identical_future_filtered_stream() -> None:
    launcher = Path("adr207/run_matched_lingbot_2gpu.sh").read_text(encoding="utf-8")

    for value in (
        "source-freezes/native-query-posterior-v18",
        "source-freeze.receipt.json",
        "lingbot-vla-v2-adr207-native-muon",
        "--runtime-hotfix",
        "lingbot_vla2_distributed_muon_collective_alignment.patch",
        "native-query-posterior-${WORLD_SIZE}gpu-30k-v1",
        "--physical-event-stream",
        "--minimum-future-source-frames 4",
        "--maximum-control-tokens 64",
        "--seed 20260721",
        "--learning-rate 1e-4",
        "PICF_ATTENTION_IMPLEMENTATION:-flex_cached",
        '--attention-implementation "$ATTENTION_IMPLEMENTATION"',
        "--lingbot-compile-mode upstream-default",
        "--trainable-scope full-host",
        "--fsdp2-placement selective-embedding-offload",
        "--cuda-allocator expandable-segments",
        '--nproc-per-node="$WORLD_SIZE"',
        "official_lbot_steps_$STEPS.json",
        "videomt-torch280-functorch-v1",
        "857d364103403df8aafc97674e97e518acf781bc8fc080840ca11c99f25aacd0",
        'assert torch.__version__ == functorch.__version__ == "2.8.0+cu128"',
    ):
        assert value in launcher
    for forbidden in (
        "dense-evidence-cache",
        "physical-sidecar",
        "task_independent_full",
        "PICF_CAPACITY",
    ):
        assert forbidden not in launcher


def test_adr207_four_gpu_wrappers_select_one_shared_world_size_contract() -> None:
    candidate = Path("adr207/run_native_videomt_query_posterior_4gpu.sh").read_text(
        encoding="utf-8"
    )
    baseline = Path("adr207/run_matched_lingbot_4gpu.sh").read_text(encoding="utf-8")
    contracts = Path("adr207/prepare_contracts_4gpu.sh").read_text(encoding="utf-8")
    base = Path("adr178/run_direct_action_posterior_full_modal.sh").read_text(
        encoding="utf-8"
    )

    for wrapper in (candidate, baseline, contracts):
        assert "export PICF_WORLD_SIZE=4" in wrapper
    assert "run_native_videomt_query_posterior_2gpu.sh" in candidate
    assert "run_matched_lingbot_2gpu.sh" in baseline
    assert "prepare_contracts_2gpu.sh" in contracts
    assert 'WORLD_SIZE=${PICF_WORLD_SIZE:-2}' in base
    assert '--nproc-per-node="$WORLD_SIZE"' in base
    assert "CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3" in base


def test_adr207_curve_comparison_launcher_is_fail_closed() -> None:
    launcher = Path("adr207/compare_matched_action_curves.sh").read_text(encoding="utf-8")

    for value in (
        "source-freezes/native-query-posterior-v18",
        "source-freeze.receipt.json",
        "0,20,100,200",
        "0,20,100,200,500,1000,1500,2000",
        "tools.compare_adr207_matched_action_curves",
        "--picf-run-dir",
        "--lbot-run-dir",
        "--steps",
        "--output",
        "git -C \"$REPO\" status --porcelain=v1 --untracked-files=all",
    ):
        assert value in launcher
    assert '[[ "$OUTPUT" == /mnt/* && ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]' in launcher


def test_adr207_modality_intervention_gate_aggregates_the_mature_contract() -> None:
    reports = []
    for rank in range(2):
        modalities = {}
        for modality in ("anytouch", "sonata", "vjepa"):
            modalities[modality] = {
                intervention: {
                    "changed_elements": 4,
                    "max_abs": (
                        0.0
                        if intervention == "joint_permutation"
                        else ADR207_ACTION_EFFECT_MIN_ABS_DRIFT * 2
                    ),
                    "rms": 0.0 if intervention == "joint_permutation" else 1e-5,
                }
                for intervention in ADR207_MODALITY_INTERVENTIONS
            }
        reports.append(
            {
                "sample_key": f"sample-{rank}",
                "factual_repeat": {"max_abs": 0.0},
                "modalities": modalities,
            }
        )

    summary = _summarize_adr207_modality_interventions(
        tuple(reports),
        checkpoint_global_step=ADR207_MODALITY_INTERVENTION_STEPS[0],
        expected_world_size=2,
    )

    assert summary["schema"] == ADR207_MODALITY_INTERVENTION_SCHEMA
    assert summary["status"] == "PASS"
    assert summary["source_query_count"] == 200
    assert summary["sample_keys"] == ["sample-0", "sample-1"]


def test_adr207_modality_intervention_gate_rejects_an_inert_dense_path() -> None:
    reports = tuple(
        {
            "sample_key": f"sample-{rank}",
            "factual_repeat": {"max_abs": 0.0},
            "modalities": {
                modality: {
                    intervention: {
                        "changed_elements": 4,
                        "max_abs": 0.0,
                        "rms": 0.0,
                    }
                    for intervention in ADR207_MODALITY_INTERVENTIONS
                }
                for modality in ("anytouch", "sonata", "vjepa")
            },
        }
        for rank in range(2)
    )

    with pytest.raises(ValueError, match="did not affect action"):
        _summarize_adr207_modality_interventions(
            reports,
            checkpoint_global_step=ADR207_MODALITY_INTERVENTION_STEPS[0],
            expected_world_size=2,
        )


def test_adr207_cache_publication_reuses_authenticated_full_modal_donors() -> None:
    publisher = Path("adr207/build_dense_cache_2gpu_contract.sh").read_text(
        encoding="utf-8"
    )
    orchestrator = Path("adr207/prepare_dense_caches_2gpu.sh").read_text(
        encoding="utf-8"
    )

    assert "tools/republish_calvin_frozen_evidence_cache.py" in publisher
    assert "source-freezes/native-query-posterior-v18" in publisher
    assert "source-freeze.receipt.json" in publisher
    assert 'git -C "$REPO" status --porcelain=v1 --untracked-files=all' in publisher
    assert "--donor-coverage-plan" in publisher
    assert "--donor-cache-root" in publisher
    assert "--coverage-plan" in publisher
    for modality in ("anytouch", "sonata", "vjepa"):
        assert modality in publisher
        assert modality in orchestrator
    assert "CUDA_VISIBLE_DEVICES=0" in orchestrator
    assert "CUDA_VISIBLE_DEVICES=1" in orchestrator


def test_adr207_source_freeze_excludes_training_outputs_and_commits_exact_code() -> None:
    freezer = Path("adr207/freeze_source.sh").read_text(encoding="utf-8")

    for excluded in (
        "/.venv*/",
        "/.audit-logs/",
        "/artifacts/",
        "/evidence/",
        "/.tmp*/",
    ):
        assert excluded in freezer
    assert "source-freeze.receipt.json" in freezer
    assert "--exclude='/source-freeze.receipt.json'" in freezer
    assert "picf-next.adr207-source-freeze/v2" in freezer
    assert "source-freezes/native-query-posterior-v18" in freezer
    assert "git -C \"$STAGING\" commit" in freezer
    assert "chmod -R a-w" in freezer


def test_adr199_launcher_enables_only_the_complete_released_videomt_arm() -> None:
    launcher = Path("adr199/run_full_transplant_stage_pq_2gpu.sh").read_text(
        encoding="utf-8"
    )
    base = Path("adr178/run_direct_action_posterior_full_modal.sh").read_text(
        encoding="utf-8"
    )

    assert "adr193_implicit_multimodal_anchor_v1" in launcher
    assert "PICF_STOP_AFTER_STEP:-250" in launcher
    assert "full-transplant-stage-pq-f6e0740a-v8" in launcher
    assert "ADR-199 forbids a non-v8 PICF_REPO override" in launcher
    assert "ADR-199 forbids a different prepared LingBot checkout" in launcher
    assert "PICF_VIDEOMT_STAGE_PQ_MODE=frozen-released-eval-causal-c5" in launcher
    assert "PICF_VIDEOMT_IDLE_PLACEMENT=cpu-between-forwards" in launcher
    assert "PICF_FSDP2_PLACEMENT=selective-embedding-frozen-vision-offload" in launcher
    assert "PICF_RUNTIME_PYTHON_OVERLAY" in launcher
    assert 'PYTHONPATH="$PICF_REPO/src:$PICF_REPO:$PICF_RUNTIME_PYTHON_OVERLAY"' in launcher
    assert "PICF_VIDEOMT_CHECKPOINT is required" in launcher
    assert "PICF_VIDEOMT_DINOV3_BUNDLE is required" in launcher
    assert "lingbot-vla-v2-adr199-prepared-frozen-visual-root-v1" in launcher
    assert "official-source-234f6f0a-v1/lingbot-vla-v2" in launcher
    assert "export PICF_LINGBOT_NATIVE_SOURCE=$SOURCE" in launcher
    assert "export PICF_LINGBOT_REPOSITORY=$LINGBOT_REPOSITORY" in launcher
    assert '--lingbot-repository "$LINGBOT_REPOSITORY"' in launcher
    assert '--prepared-lingbot-checkout "$SOURCE"' in launcher
    assert "../adr193/run_implicit_multimodal_anchor_2gpu.sh" in launcher
    assert "PICF_VIDEOMT_STAGE_PQ_MODE:-disabled" in base
    assert "frozen-released-eval-causal-c5-pqm" in base
    assert "frozen-adapted-eval-causal-c5-pqm" in base
    assert "frozen-adapted-eval-causal-c5-pqmr" in base
    assert "frozen-adapted-eval-causal-c5-pqrf" in base
    assert '--videomt-stage-pq-mode "$VIDEOMT_STAGE_PQ_MODE"' in base
    assert '--videomt-checkpoint "$PICF_VIDEOMT_CHECKPOINT"' in base
    assert '--videomt-dinov3-bundle "$PICF_VIDEOMT_DINOV3_BUNDLE"' in base
    assert '--videomt-adapted-checkpoint "$PICF_VIDEOMT_ADAPTED_CHECKPOINT"' in base
    assert (
        '--videomt-adapted-checkpoint-sha256 '
        '"$PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256"'
    ) in base
    assert 'PICF_VIDEOMT_IDLE_PLACEMENT:-cuda-resident' in base
    assert 'PICF_FSDP2_PLACEMENT:-selective-embedding-offload' in base
    assert '--fsdp2-placement "$FSDP2_PLACEMENT"' in base
    assert '--videomt-idle-placement "$VIDEOMT_IDLE_PLACEMENT"' in base
    assert '--videomt-fsdp2-placement "$VIDEOMT_FSDP2_PLACEMENT"' in base
    assert "RUNTIME_PYTHON_OVERLAY=${PICF_RUNTIME_PYTHON_OVERLAY:-}" in base
    assert "IFS=':' read -r -a RUNTIME_PYTHON_OVERLAYS" in base
    assert 'for overlay in "${RUNTIME_PYTHON_OVERLAYS[@]}"' in base
    assert 'RUNTIME_PYTHON_OVERLAY:+:$RUNTIME_PYTHON_OVERLAY' in base
    assert '"${VIDEOMT_STAGE_PQ_ARGS[@]}"' in base


def test_adr200_launcher_selects_the_complete_query_mask_large_model_path() -> None:
    launcher = Path("adr200/run_full_pqm_2gpu.sh").read_text(encoding="utf-8")

    assert "adr200/source-freezes/full-pqm-v1" in launcher
    assert "frozen-released-eval-causal-c5-pqm" in launcher
    assert "adr193_implicit_multimodal_anchor_v1" in launcher
    assert "selective-embedding-frozen-vision-offload" in launcher
    assert "PICF_STOP_AFTER_STEP:-250" in launcher
    assert "PICF_VIDEOMT_IDLE_PLACEMENT:-cpu-between-forwards" in launcher
    assert "PICF_VIDEOMT_CHECKPOINT is required" in launcher
    assert "PICF_VIDEOMT_DINOV3_BUNDLE is required" in launcher
    assert "audit_full_transplant_contract.py" in launcher
    assert "../adr193/run_implicit_multimodal_anchor_2gpu.sh" in launcher


def test_adr202_launcher_selects_the_complete_adapted_query_mask_path() -> None:
    launcher = Path("adr202/run_adapted_full_pqm_2gpu.sh").read_text(encoding="utf-8")

    assert "adr202/source-freezes/adapted-full-pqm-v2" in launcher
    assert "frozen-adapted-eval-causal-c5-pqm" in launcher
    assert "adr193_implicit_multimodal_anchor_v1" in launcher
    assert "selective-embedding-frozen-vision-offload" in launcher
    assert "PICF_VIDEOMT_ADAPTED_CHECKPOINT is required" in launcher
    assert "PICF_VIDEOMT_ADAPTED_CHECKPOINT_SHA256 is required" in launcher
    assert "PICF_REUSED_FULL_TRANSPLANT_AUDIT" in launcher
    assert "picf-next.reused-full-transplant-audit/v1" in launcher
    assert "audit_full_transplant_contract.py" in launcher
    assert "../adr193/run_implicit_multimodal_anchor_2gpu.sh" in launcher


def test_adr203_launcher_selects_the_source_faithful_row_mask_path() -> None:
    launcher = Path("adr203/run_source_faithful_row_mask_2gpu.sh").read_text(
        encoding="utf-8"
    )

    assert "adr203/source-freezes/source-faithful-row-mask-v2" in launcher
    assert "frozen-adapted-eval-causal-c5-pqmr" in launcher
    assert "adr193_implicit_multimodal_anchor_v1" in launcher
    assert "selective-embedding-frozen-vision-offload" in launcher
    assert "PICF_STOP_AFTER_STEP:-100" in launcher
    assert "PICF_VIDEOMT_ADAPTED_CHECKPOINT is required" in launcher
    assert "../adr193/run_implicit_multimodal_anchor_2gpu.sh" in launcher


def test_adr204_launcher_selects_complete_source_row_refinement() -> None:
    launcher = Path("adr204/run_full_source_row_refinement_2gpu.sh").read_text(
        encoding="utf-8"
    )

    assert "adr204/source-freezes/full-source-row-refinement-v3" in launcher
    assert "frozen-adapted-eval-causal-c5-pqrf" in launcher
    assert "adr193_implicit_multimodal_anchor_v1" in launcher
    assert "selective-embedding-frozen-vision-offload" in launcher
    assert "PICF_STOP_AFTER_STEP:-250" in launcher
    assert "PICF_VIDEOMT_ADAPTED_CHECKPOINT is required" in launcher
    assert "../adr193/run_implicit_multimodal_anchor_2gpu.sh" in launcher


def test_task_addressed_cold_paths_use_the_canonical_lane_state() -> None:
    runner = Path("tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")

    assert "def cold_previous_state(" in runner
    assert "previous_memory=cold_previous_state(collated_batch)" in runner
    assert "previous_state=cold_previous_state(batch)" in runner


def test_task_addressed_causal_interventions_preserve_the_routing_gauge() -> None:
    runner = Path("tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")
    diagnostic = runner[
        runner.index("        def run_v3_control_intervention(") : runner.index(
            "        def run_full_modal_action_adoption_phase()"
        )
    ]

    assert "gathered_prior_addresses" in diagnostic
    assert "gathered_state_addresses" in diagnostic
    assert "layerwise_prior_trace_with_tensor(" in diagnostic
    assert "persistent_state_with_tensor(" in diagnostic
    assert "episode_address_state=peer_prior_address" in diagnostic
    assert "episode_address_state=peer_state_address" in diagnostic
    assert '"wrong_row": factual_prior.permute_rows' not in diagnostic
    assert '("wrong_row", factual_state.permute_rows' not in diagnostic


def test_training_step_releases_every_graph_bearing_alias_before_the_next_step() -> None:
    runner = Path("tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")
    release_start = runner.index(
        "            result = None\n            native_joint_result = None"
    )
    release_end = runner.index("            causal_report = run_causal_diagnostic(")
    release = runner[release_start:release_end]

    for name in (
        "result",
        "native_joint_result",
        "videomt_transaction",
        "videomt_source_batch",
        "objective",
        "frame_losses",
        "omitted_policy",
        "effective_action_loss",
        "effective_policy_loss",
        "primary_policy",
        "source_objective",
        "native_relation",
        "report_videomt_source_objective",
        "posterior",
        "captured",
        "sequential_plan",
        "replayed_prior",
        "omitted_result",
    ):
        assert f"            {name} = None" in release
    assert "torch.cuda.empty_cache()" in release
    assert '"step_graph_released"' in release


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("total_steps", 2),
        ("metrics_every", 1),
        ("visual_every", 1),
        ("checkpoint_every", 1),
    ),
)
def test_production_cadence_rejects_runtime_redefinition(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="frozen at 30k/100/250/2000"):
        ProductionCadence(**{field: value})


def test_production_temporal_estimator_uses_detached_real_age_lanes() -> None:
    assert RUNNER_SCHEMA == "picf-next.task-independent-full-runner/v19"
    assert PRODUCTION_LOCAL_BPTT_PROBABILITY == 0.0
    _validate_production_temporal_estimator(PRODUCTION_LOCAL_BPTT_PROBABILITY)

    with pytest.raises(ValueError, match="local_bptt_probability=0"):
        _validate_production_temporal_estimator(0.10)


def _toy_lingbot_scope_policy() -> torch.nn.Module:
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.qwenvl_with_expert = torch.nn.Module()
    policy.model.qwenvl_with_expert.qwenvl = torch.nn.Module()
    policy.model.qwenvl_with_expert.qwenvl.model = torch.nn.Module()
    policy.model.qwenvl_with_expert.qwenvl.model.visual = torch.nn.Linear(3, 4)
    policy.model.qwenvl_with_expert.qwen_expert = torch.nn.Linear(4, 5)
    return policy


def test_trainable_scope_receipt_proves_the_visual_freeze_only() -> None:
    policy = _toy_lingbot_scope_policy()
    full = _trainable_scope_receipt(policy, scope=TRAINABLE_SCOPE_FULL_HOST)
    assert full["forward_model_complete"] is True
    assert full["visual_forward_enabled"] is True
    assert full["visual_numel"] == 16
    assert full["trainable_visual_numel"] == 16

    for parameter in policy.model.qwenvl_with_expert.qwenvl.model.visual.parameters():
        parameter.requires_grad_(False)
    frozen = _trainable_scope_receipt(
        policy,
        scope=TRAINABLE_SCOPE_FROZEN_VISION_HOST,
    )
    assert frozen["visual_numel"] == 16
    assert frozen["trainable_visual_numel"] == 0
    assert frozen["trainable_numel"] < frozen["total_numel"]

    with pytest.raises(RuntimeError, match="violate the declared trainable scope"):
        _trainable_scope_receipt(policy, scope=TRAINABLE_SCOPE_FULL_HOST)


@pytest.mark.parametrize("value", ("0", "-0.1", "nan", "inf", "-inf"))
def test_objective_weights_must_be_positive_and_finite(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite and greater than zero"):
        _positive_finite_float(value)


def test_objective_weight_parser_preserves_explicit_value() -> None:
    assert _positive_finite_float("0.08") == pytest.approx(0.08)


@pytest.mark.parametrize("value", ("-0.1", "nan", "inf", "-inf"))
def test_nonnegative_weight_parser_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite and non-negative"):
        _nonnegative_finite_float(value)


def _causal_args(mode: str) -> argparse.Namespace:
    branch = mode == "current_frame_branch"
    return argparse.Namespace(
        causal_ablation_mode=mode,
        phase="fresh" if branch else "resume",
        load_global_step=0 if branch else CAUSAL_BRANCH_STEP,
        stop_after_step=CAUSAL_BRANCH_STEP if branch else CAUSAL_ARM_STOP_STEP,
        seed=20260721,
        capacity=16,
        maximum_control_tokens=8,
        maximum_optimizer_lag=8,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        maximum_peak_reserved_gib=39.0,
        entity_weight=0.08,
        predictive_weight=0.0,
        mask_focal_weight=1.0,
        mask_dice_weight=1.0,
        existence_weight=1.0,
        ownership_weight=1.0,
        predictive_loss_power=1.0,
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
        source_mask_probability=0.0,
        source_mask_token_fraction=0.0625,
        source_prediction_mode="omitted_static",
        minimum_supervised_fraction=0.0,
        fsdp2_placement="selective-embedding-offload",
        cuda_allocator="native",
    )


def _adr221_wsa_edge_diagnostic_args() -> argparse.Namespace:
    args = _causal_args("current_frame_branch")
    args.causal_ablation_mode = "none"
    args.phase = "fresh"
    args.load_global_step = 0
    args.stop_after_step = 1
    args.acceptance_mode = "none"
    args.posterior_architecture = "two_pass_v3"
    args.picf_architecture_profile = full_runner.ADR221_ARCHITECTURE_PROFILE
    args.adr221_wsa_edge_diagnostic = True
    args.engineering_force_omitted_static_step = 0
    args.engineering_force_causal_diagnostic_step = 0
    args.run_dir = Path("/mnt/picf-next/adr221/runs/diagnostics/edge")
    return args


def test_adr221_wsa_edge_diagnostic_is_measurement_only_and_step_zero_registered() -> None:
    args = _adr221_wsa_edge_diagnostic_args()

    full_runner._validate_adr221_wsa_edge_diagnostic_args(args)
    assert _registered_action_evaluation_steps(args) == (0,)
    assert _action_evaluation_active(args)
    assert _execution_contract(args)["wsa_future_to_action_edge_diagnostic"] == {
        "active": True,
        "intervention": "block_future_to_action",
        "scope": "measurement_only_fixed_source_host_replay",
        "source_host_batch_reused_by_identity": True,
        "paired_host_rng_replayed_exactly": True,
        "posterior_must_be_exact_equal": True,
        "optimization_graph_changed": False,
    }
    launcher = Path("adr178/run_direct_action_posterior_full_modal.sh").read_text()
    assert "PICF_ADR221_WSA_EDGE_DIAGNOSTIC" in launcher
    assert "MINIMUM_STOP_AFTER_STEP=1" in launcher
    assert 'WSA_EDGE_DIAGNOSTIC_ARGS=(--adr221-wsa-edge-diagnostic)' in launcher

    args.stop_after_step = 20
    with pytest.raises(ValueError, match="requires profile/fresh/load/stop"):
        full_runner._validate_adr221_wsa_edge_diagnostic_args(args)


def test_adr222_registers_world_token_route_without_reusing_measurement_mode() -> None:
    args = _adr221_wsa_edge_diagnostic_args()
    args.picf_architecture_profile = full_runner.ADR222_ARCHITECTURE_PROFILE
    args.adr221_wsa_edge_diagnostic = False
    args.stop_after_step = 20

    assert full_runner._adr222_world_token_adoption_active(args)
    assert full_runner._posterior_adoption_route_active_for_args(args)
    contract = _execution_contract(args)
    assert contract["posterior_adoption_route"] is True
    assert contract["posterior_adoption_dose"] is False
    assert contract["wsa_action_coupling"] == (
        "auxiliary_world_decoder_no_future_action_keys"
    )
    assert contract["wsa_future_to_action_edge_diagnostic"] == {"active": False}


def test_adr221_wsa_edge_summary_requires_exact_source_and_posterior() -> None:
    def sample(
        partition: str,
        standard: float,
        factual: float,
        blocked: float,
    ) -> dict[str, object]:
        return {
            "partition": partition,
            "action_loss": standard,
            "wsa_future_to_action_intervention": {
                "intervention": "block_future_to_action",
                "source_host_batch_reused_by_identity": True,
                "posterior_exact_equal": True,
                "factual_action_loss": factual,
                "blocked_action_loss": blocked,
                "blocked_minus_factual_action_loss": blocked - factual,
            },
        }

    samples = [
        sample("validation", 0.55, 0.5, 0.4),
        sample("validation", 0.35, 0.3, 0.35),
        sample("heldout", 0.65, 0.6, 0.45),
    ]
    summary = full_runner._summarize_adr221_wsa_edge_intervention(
        samples,
        partition="validation",
    )
    assert summary["schema"] == full_runner.ADR221_WSA_EDGE_INTERVENTION_SCHEMA
    assert summary["sample_count"] == 2
    assert summary["standard_action_loss_mean"] == pytest.approx(0.45)
    assert summary["factual_action_loss_mean"] == pytest.approx(0.4)
    assert summary["paired_factual_minus_standard_action_loss_mean"] == pytest.approx(-0.05)
    assert summary["blocked_action_loss_mean"] == pytest.approx(0.375)
    assert summary["blocked_minus_factual_action_loss_mean"] == pytest.approx(-0.025)
    assert summary["blocked_improved_fraction"] == pytest.approx(0.5)

    intervention = samples[0]["wsa_future_to_action_intervention"]
    assert isinstance(intervention, dict)
    intervention["posterior_exact_equal"] = False
    with pytest.raises(ValueError, match="changed the emitted posterior"):
        full_runner._summarize_adr221_wsa_edge_intervention(
            samples,
            partition="validation",
        )


@pytest.mark.parametrize(
    "mode",
    ("current_frame_branch", "zero_state", "recurrent_state"),
)
def test_adr146_modes_have_one_shared_execution_contract(mode: str) -> None:
    args = _causal_args(mode)
    _validate_causal_ablation_args(args)
    assert _execution_contract(args)["experiment"] == "adr146-recurrence-only"


def test_adr146_rejects_prediction_or_the_wrong_resume_boundary() -> None:
    predicted = _causal_args("recurrent_state")
    predicted.predictive_weight = 0.08
    with pytest.raises(ValueError, match="weights 0.08/0.0"):
        _validate_causal_ablation_args(predicted)

    wrong_boundary = _causal_args("zero_state")
    wrong_boundary.load_global_step = 2_000
    with pytest.raises(ValueError, match="phase/load/stop"):
        _validate_causal_ablation_args(wrong_boundary)


def test_layerwise_v2_is_a_fresh_or_2k_resume_core_without_auxiliaries() -> None:
    args = _causal_args("current_frame_branch")
    args.posterior_architecture = "layerwise_v2"
    args.causal_ablation_mode = "none"
    args.phase = "fresh"
    args.stop_after_step = TOTAL_STEPS
    for name in (
        "predictive_cache_root",
        "predictive_cache_build_report",
        "predictive_cache_build_report_sha256",
        "current_grid_cache_root",
        "current_grid_cache_build_report",
        "current_grid_cache_build_report_sha256",
    ):
        setattr(args, name, None)

    _validate_causal_ablation_args(args)
    _validate_auxiliary_cache_args(args)
    assert not _predictive_assets_required(args)
    assert not _current_correction_assets_required(args)
    assert _execution_contract(args)["posterior_architecture"] == "layerwise_v2"
    assert _execution_contract(args)["objective_profile"] == "adr147_recurrent_core"
    assert len(_disabled_auxiliary_digest("predictive")) == 64
    assert _disabled_auxiliary_digest("predictive") != _disabled_auxiliary_digest("current_grid")
    assert _disabled_auxiliary_digest("dense_evidence") not in {
        _disabled_auxiliary_digest("predictive"),
        _disabled_auxiliary_digest("current_grid"),
    }
    with pytest.raises(ValueError, match="unknown auxiliary cache family"):
        _disabled_auxiliary_digest("unknown")

    args.phase = "resume"
    args.load_global_step = CHECKPOINT_EVERY
    _validate_causal_ablation_args(args)


def test_layerwise_v2_rejects_legacy_cache_inputs_and_auxiliary_branches() -> None:
    args = _causal_args("current_frame_branch")
    args.posterior_architecture = "layerwise_v2"
    args.causal_ablation_mode = "none"
    args.phase = "fresh"
    args.stop_after_step = TOTAL_STEPS
    args.predictive_cache_root = Path("retired-cache")
    for name in (
        "predictive_cache_build_report",
        "predictive_cache_build_report_sha256",
        "current_grid_cache_root",
        "current_grid_cache_build_report",
        "current_grid_cache_build_report_sha256",
    ):
        setattr(args, name, None)
    with pytest.raises(ValueError, match="forbids predictive/current-grid"):
        _validate_auxiliary_cache_args(args)

    args.predictive_cache_root = None
    args.source_mask_probability = 0.1
    with pytest.raises(ValueError, match="keeps local/source auxiliaries disabled"):
        _validate_causal_ablation_args(args)


def test_layerwise_v2_predictive_correction_requires_only_current_assets() -> None:
    args = _causal_args("current_frame_branch")
    args.posterior_architecture = "layerwise_v2"
    args.causal_ablation_mode = "none"
    args.phase = "fresh"
    args.stop_after_step = TOTAL_STEPS
    args.entity_weight = ADR148_ENTITY_WEIGHT
    args.predictive_weight = ADR148_PREDICTIVE_WEIGHT
    args.source_mask_probability = ADR148_SOURCE_MASK_PROBABILITY
    args.predictive_cache_root = None
    args.predictive_cache_build_report = None
    args.predictive_cache_build_report_sha256 = None
    args.current_grid_cache_root = Path("current-cache")
    args.current_grid_cache_build_report = Path("current-cache.build-report.json")
    args.current_grid_cache_build_report_sha256 = "a" * 64

    _validate_causal_ablation_args(args)
    _validate_auxiliary_cache_args(args)
    assert not _predictive_assets_required(args)
    assert _current_correction_assets_required(args)
    assert _execution_contract(args)["objective_profile"] == ("adr148_prior_current_correction")

    args.current_grid_cache_root = None
    with pytest.raises(ValueError, match="requires every current-cache"):
        _validate_auxiliary_cache_args(args)

    args.current_grid_cache_root = Path("current-cache")
    args.predictive_cache_root = Path("future-cache")
    with pytest.raises(ValueError, match="forbids future predictive-cache"):
        _validate_auxiliary_cache_args(args)

    args.predictive_cache_root = None
    args.overshoot_probability = 0.05
    with pytest.raises(ValueError, match="requires entity/predictive"):
        _validate_causal_ablation_args(args)


def test_two_pass_v3_has_one_fail_closed_filter_profile() -> None:
    args = _causal_args("current_frame_branch")
    args.posterior_architecture = "two_pass_v3"
    args.causal_ablation_mode = "none"
    args.phase = "fresh"
    args.stop_after_step = TOTAL_STEPS
    args.entity_weight = ADR149_ENTITY_WEIGHT
    args.predictive_weight = ADR149_FILTER_WEIGHT
    args.source_mask_probability = ADR149_OMITTED_STATIC_PROBABILITY
    args.predictive_cache_root = None
    args.predictive_cache_build_report = None
    args.predictive_cache_build_report_sha256 = None
    args.current_grid_cache_root = Path("current-cache")
    args.current_grid_cache_build_report = Path("current-cache.build-report.json")
    args.current_grid_cache_build_report_sha256 = "a" * 64

    _validate_causal_ablation_args(args)
    _validate_auxiliary_cache_args(args)
    assert not _predictive_assets_required(args)
    assert _current_correction_assets_required(args)
    contract = _execution_contract(args)
    assert contract["posterior_architecture"] == "two_pass_v3"
    assert contract["objective_profile"] == "adr149_action_visible_two_pass_filter"
    assert contract["physical_event_stream"] is True
    assert contract["prompt_overlay"] == "deterministic_plan_episode_sample_candidates_v1"
    assert contract["control_receipt"] == (
        "exact_raw_actions_chunked_without_semantic_compression_v1"
    )
    assert contract["distributed_prior_host_schedule"] == V3_DISTRIBUTED_PRIOR_SCHEDULE


def test_forced_omitted_static_smoke_is_engineering_only_and_bounded() -> None:
    args = _causal_args("current_frame_branch")
    args.posterior_architecture = "two_pass_v3"
    args.causal_ablation_mode = "none"
    args.phase = "fresh"
    args.acceptance_mode = "none"
    args.load_global_step = 0
    args.stop_after_step = 1
    args.dense_evidence_mode = "calvin_full_v1"
    args.source_mask_probability = ADR149_OMITTED_STATIC_PROBABILITY
    args.source_prediction_mode = "omitted_static"
    args.run_dir = Path("/mnt/picf-next/adr176/diagnostics/forced-omitted")
    args.engineering_force_omitted_static_step = 1

    _validate_engineering_smoke_args(args)
    assert _execution_contract(args)["engineering_force_omitted_static_step"] == 1

    args.stop_after_step = 3
    _validate_engineering_smoke_args(args)

    args.stop_after_step = 4
    with pytest.raises(ValueError, match="forced engineering smoke requires"):
        _validate_engineering_smoke_args(args)

    args.stop_after_step = 1
    args.run_dir = Path("/mnt/picf-next/adr176/scientific-arm")
    with pytest.raises(ValueError, match="under diagnostics"):
        _validate_engineering_smoke_args(args)

    args.run_dir = Path("/mnt/picf-next/adr176/diagnostics/forced-omitted")
    args.engineering_force_omitted_static_step = 33
    with pytest.raises(ValueError, match=r"integer in \[1, 32\]"):
        _validate_engineering_smoke_args(args)

    args.local_bptt_probability = 0.1
    with pytest.raises(ValueError, match="ADR-149 requires"):
        _validate_causal_ablation_args(args)

    args.local_bptt_probability = 0.0
    args.predictive_cache_root = Path("legacy-future-cache")
    with pytest.raises(ValueError, match="forbids legacy future-cache"):
        _validate_auxiliary_cache_args(args)


def test_forced_causal_diagnostic_smoke_is_early_bounded_and_exclusive() -> None:
    args = _causal_args("current_frame_branch")
    args.posterior_architecture = "two_pass_v3"
    args.causal_ablation_mode = "none"
    args.phase = "fresh"
    args.acceptance_mode = "none"
    args.load_global_step = 0
    args.stop_after_step = 4
    args.dense_evidence_mode = "calvin_full_v1"
    args.source_mask_probability = ADR149_OMITTED_STATIC_PROBABILITY
    args.source_prediction_mode = "omitted_static"
    args.run_dir = Path("/mnt/picf-next/adr176/diagnostics/forced-causal")
    args.engineering_force_omitted_static_step = 0
    args.engineering_force_causal_diagnostic_step = 4

    _validate_engineering_smoke_args(args)
    contract = _execution_contract(args)
    assert contract["engineering_force_causal_diagnostic_step"] == 4

    args.engineering_force_causal_diagnostic_step = 2
    with pytest.raises(ValueError, match=r"requires step in \[3, 32\]"):
        _validate_engineering_smoke_args(args)

    args.engineering_force_causal_diagnostic_step = 4
    args.engineering_force_omitted_static_step = 1
    with pytest.raises(ValueError, match="cannot force two independent branches"):
        _validate_engineering_smoke_args(args)


def _acceptance_args(mode: str) -> argparse.Namespace:
    args = _causal_args("current_frame_branch")
    args.posterior_architecture = "two_pass_v3"
    args.causal_ablation_mode = "none"
    args.entity_weight = ADR149_ENTITY_WEIGHT
    args.predictive_weight = ADR149_FILTER_WEIGHT
    args.source_mask_probability = ADR149_OMITTED_STATIC_PROBABILITY
    args.dense_evidence_mode = "calvin_full_v1"
    args.acceptance_mode = mode
    if mode in {"action-adoption-presence", "action-adoption-interventions"}:
        args.phase, args.load_global_step, args.stop_after_step = "fresh", 0, 1
    elif mode == "posterior-adoption-route":
        args.phase, args.load_global_step, args.stop_after_step = (
            "fresh",
            0,
            POSTERIOR_ADOPTION_STOP_STEP,
        )
    elif mode == "posterior-adoption-dose":
        args.phase, args.load_global_step, args.stop_after_step = (
            "fresh",
            0,
            POSTERIOR_ADOPTION_DOSE_STOP_STEP,
        )
        args.source_mask_probability = POSTERIOR_ADOPTION_DOSE_SOURCE_MASK_PROBABILITY
    elif mode == "dcp-uninterrupted":
        args.phase, args.load_global_step, args.stop_after_step = "fresh", 0, 2
    else:
        args.phase, args.load_global_step, args.stop_after_step = "resume", 1, 2
    return args


@pytest.mark.parametrize(
    "mode",
    (
        "action-adoption-presence",
        "action-adoption-interventions",
        "posterior-adoption-route",
        "posterior-adoption-dose",
        "dcp-uninterrupted",
        "dcp-restored",
    ),
)
def test_adr150_acceptance_modes_are_exact_and_two_pass(mode: str) -> None:
    args = _acceptance_args(mode)
    _validate_acceptance_args(args)
    _validate_causal_ablation_args(args)
    assert _acceptance_checkpoint_due(mode=mode, global_step=1) == (mode == "dcp-uninterrupted")
    assert _acceptance_checkpoint_due(
        mode=mode,
        global_step=POSTERIOR_ADOPTION_STOP_STEP,
    ) == (mode == "posterior-adoption-route")
    assert _acceptance_checkpoint_due(
        mode=mode,
        global_step=POSTERIOR_ADOPTION_DOSE_STOP_STEP,
    ) == (mode == "posterior-adoption-dose")


def test_posterior_adoption_routes_are_explicit_in_the_execution_contract() -> None:
    args = _acceptance_args("posterior-adoption-route")
    contract = _execution_contract(args)
    assert contract["acceptance_mode"] == "posterior-adoption-route"
    assert contract["posterior_adoption_route"] is True
    assert contract["posterior_adoption_dose"] is False
    assert contract["posterior_adoption_factual_branch"] is True
    assert contract["posterior_adoption_routed_action_every_step"] is False

    args = _acceptance_args("posterior-adoption-dose")
    contract = _execution_contract(args)
    assert contract["acceptance_mode"] == "posterior-adoption-dose"
    assert contract["posterior_adoption_route"] is True
    assert contract["posterior_adoption_dose"] is True
    assert contract["posterior_adoption_factual_branch"] is True
    assert contract["posterior_adoption_routed_action_every_step"] is True
    assert contract["action_evaluation_steps"] == (0, 20, 100, 200)

    args.acceptance_mode = "none"
    assert _execution_contract(args)["posterior_adoption_route"] is False


def test_dcp_save_and_restore_share_one_checkpoint_execution_identity() -> None:
    uninterrupted = _acceptance_args("dcp-uninterrupted")
    restored = _acceptance_args("dcp-restored")
    assert _execution_contract(uninterrupted) == _execution_contract(restored)
    assert _execution_contract(uninterrupted)["acceptance_mode"] == "dcp-cold-restore"


def test_adr150_acceptance_rejects_wrong_boundary_or_partial_graph() -> None:
    args = _acceptance_args("dcp-restored")
    args.load_global_step = 2_000
    with pytest.raises(ValueError, match="phase/load/stop"):
        _validate_acceptance_args(args)

    args = _acceptance_args("dcp-uninterrupted")
    args.dense_evidence_mode = "none"
    with pytest.raises(ValueError, match="calvin_full_v1"):
        _validate_acceptance_args(args)


def test_posterior_adoption_dose_is_one_exact_high_dose_contract() -> None:
    args = _acceptance_args("posterior-adoption-dose")

    _validate_acceptance_args(args)
    _validate_causal_ablation_args(args)
    assert _posterior_adoption_route_active(args.acceptance_mode)
    assert _action_evaluation_active(args)
    assert _registered_action_evaluation_steps(args) == (
        POSTERIOR_ADOPTION_DOSE_ACTION_EVALUATION_STEPS
    )
    assert _acceptance_terminal_evidence_due(
        mode=args.acceptance_mode,
        global_step=POSTERIOR_ADOPTION_DOSE_STOP_STEP,
    )
    assert not _acceptance_terminal_evidence_due(
        mode=args.acceptance_mode,
        global_step=199,
    )

    for field, value in (
        ("phase", "resume"),
        ("load_global_step", 1),
        ("stop_after_step", 201),
        ("posterior_architecture", "layerwise_v2"),
        ("dense_evidence_mode", "none"),
        ("causal_ablation_mode", "zero_state"),
        ("entity_weight", 0.081),
        ("predictive_weight", 0.005),
        ("local_bptt_probability", 0.1),
        ("overshoot_probability", 0.1),
        ("source_mask_probability", ADR149_OMITTED_STATIC_PROBABILITY),
        ("source_prediction_mode", "current_grid"),
    ):
        invalid = _acceptance_args("posterior-adoption-dose")
        setattr(invalid, field, value)
        with pytest.raises(ValueError):
            _validate_acceptance_args(invalid)
            _validate_causal_ablation_args(invalid)


def test_fresh_two_pass_action_curve_stops_at_an_exact_registered_boundary() -> None:
    args = _acceptance_args("none")
    args.phase = "fresh"
    args.load_global_step = 0
    args.stop_after_step = 1_500

    expected = (0, 20, 100, 200, 500, 1_000, 1_500)
    assert _registered_action_evaluation_steps(args) == expected
    assert _action_evaluation_active(args)
    assert _execution_contract(args)["action_evaluation_steps"] == expected

    args.stop_after_step = 1_499
    assert _registered_action_evaluation_steps(args) == TWO_PASS_ACTION_EVALUATION_STEPS
    assert not _action_evaluation_active(args)


def test_posterior_adoption_dose_selects_the_omitted_branch_for_every_seed() -> None:
    config = TemporalEstimatorConfig(
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
        source_mask_probability=POSTERIOR_ADOPTION_DOSE_SOURCE_MASK_PROBABILITY,
        maximum_optimizer_lag=8,
    )

    for seed in range(256):
        plan = sample_temporal_batch_plan(
            config,
            seed=seed,
            state_ages=(seed,),
            available_future_steps=(0,),
            optimizer_lags=(0,),
        )
        assert plan.source_masked_branch is True
        assert plan.local_bptt_steps is None
        assert plan.overshoot_horizon is None


def test_posterior_adoption_dose_step_requires_both_action_branches() -> None:
    route = torch.ones(2, dtype=torch.bool)
    result = argparse.Namespace(
        primary=object(),
        omitted_static_branch=object(),
        omitted_static_policy=object(),
    )
    values = {
        "mode": "posterior-adoption-dose",
        "source_masked_branch": True,
        "omitted_static_view": object(),
        "posterior_adoption_route": route,
        "expected_batch_size": 2,
        "result": result,
    }

    _validate_posterior_adoption_dose_step(**values)

    failures = (
        {"source_masked_branch": False},
        {"omitted_static_view": None},
        {"posterior_adoption_route": torch.tensor([True, False])},
        {"expected_batch_size": 1},
        {"result": argparse.Namespace(primary=None)},
        {
            "result": argparse.Namespace(
                primary=object(),
                omitted_static_branch=None,
                omitted_static_policy=None,
            )
        },
    )
    for changes in failures:
        with pytest.raises(RuntimeError, match="posterior-adoption-dose"):
            _validate_posterior_adoption_dose_step(**(values | changes))


def test_posterior_adoption_dose_launcher_is_exact_and_syntactically_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = root / "adr150" / "run_full_modal_acceptance_4gpu.sh"
    source = launcher.read_text(encoding="utf-8")

    subprocess.run(("bash", "-n", str(launcher)), check=True)
    for required in (
        "posterior-adoption-dose)",
        "ACCEPTANCE_MODE=posterior-adoption-dose",
        "STOP_AFTER_STEP=200",
        "SOURCE_MASK_PROBABILITY=1.0",
        '--source-mask-probability "$SOURCE_MASK_PROBABILITY"',
        "--posterior-architecture two_pass_v3",
        "--dense-evidence-mode calvin_full_v1",
        "--causal-ablation-mode none",
        "--entity-weight 0.08",
        "--predictive-weight 0.004",
        "--local-bptt-probability 0.0",
        "--overshoot-probability 0.0",
        "--source-prediction-mode omitted_static",
    ):
        assert required in source

    rejected = subprocess.run(
        ("bash", str(launcher), "posterior-adoption-dose", "/tmp/not-persistent"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "must persist under /mnt" in rejected.stderr


def test_full_dense_evidence_is_exact_three_cache_two_pass_only() -> None:
    args = _causal_args("current_frame_branch")
    args.posterior_architecture = "two_pass_v3"
    args.dense_evidence_mode = "calvin_full_v1"
    args.dense_evidence_cache_root = [Path("vjepa"), Path("sonata"), Path("anytouch")]
    args.dense_evidence_cache_manifest_sha256 = ["c" * 64, "a" * 64, "b" * 64]
    args.dense_evidence_supplement_cache_root = []
    args.dense_evidence_supplement_cache_manifest_sha256 = []
    args.dense_evidence_coverage_plan = Path("coverage.json")
    args.dense_evidence_coverage_plan_sha256 = "d" * 64
    args.stream_plan = Path("stream.json")
    args.stream_plan_sha256 = "e" * 64
    args.representation_split = Path("split.json")
    args.representation_split_sha256 = "f" * 64
    args.evaluation_plan = Path("evaluation.json")
    args.evaluation_plan_sha256 = "1" * 64
    args.dense_token_bridge = "lingbot_task_token_resampler_v1"

    _validate_dense_evidence_args(args)
    assert _execution_contract(args)["dense_token_bridge"] == ("lingbot_task_token_resampler_v1")
    assert _execution_contract(args)["dense_evidence_cache_manifest_sha256"] == (
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    assert _execution_contract(args)["dense_evidence_coverage_plan_file_sha256"] == "d" * 64
    assert _execution_contract(args)["native_relation_surfaces"] == [
        {
            "name": "anytouch",
            "geometry_kind": "contact_sites",
            "layout": "anytouch2.calvin.contact-sites.v1",
            "target_kind": "none",
        },
        {
            "name": "sonata",
            "geometry_kind": "world_points",
            "layout": "sonata.calvin.world-points.v1",
            "target_kind": "none",
        },
        {
            "name": "vjepa",
            "geometry_kind": "image_grid",
            "layout": "vjepa21.calvin.static-gripper.24x24.v1",
            "target_kind": "calvin_vjepa21_visible_owner_v1",
        },
    ]

    args.dense_evidence_supplement_cache_root = [
        Path("supplement-vjepa"),
        Path("supplement-sonata"),
        Path("supplement-anytouch"),
    ]
    args.dense_evidence_supplement_cache_manifest_sha256 = [
        "4" * 64,
        "2" * 64,
        "3" * 64,
    ]
    _validate_dense_evidence_args(args)
    assert _execution_contract(args)["dense_evidence_supplement_cache_manifest_sha256"] == (
        "2" * 64,
        "3" * 64,
        "4" * 64,
    )
    args.dense_evidence_supplement_cache_root.pop()
    with pytest.raises(ValueError, match="supplements require exactly three"):
        _validate_dense_evidence_args(args)
    args.dense_evidence_supplement_cache_root.append(Path("supplement-anytouch"))
    runner_source = (
        Path(__file__)
        .parents[2]
        .joinpath("tools/run_lingbot_vla2_task_independent_full.py")
        .read_text(encoding="utf-8")
    )
    assert "validate_calvin_evidence_timestamps(" in runner_source
    assert "physical_dataset.timestamp_s_by_key(canonical_key)" not in runner_source
    assert "index.source_episode(source_global_index)" in runner_source
    assert (
        "source_global_index_by_sample_key(canonical_key) != source_global_index"
        in runner_source
    )
    assert "dense_source_identity(transition.sample.sample_key)[0]" in runner_source
    assert "prior_row_bindings_by_batch=empty_bindings" in runner_source
    assert "prior_row_bindings_by_batch=(empty_bindings,)" not in runner_source
    assert (
        "evaluation_dataset.source_global_index_by_key(transition.sample.sample_key)"
        not in runner_source
    )

    args.posterior_architecture = "layerwise_v2"
    with pytest.raises(ValueError, match="two_pass_v3"):
        _validate_dense_evidence_args(args)
    args.posterior_architecture = "two_pass_v3"
    args.dense_evidence_cache_root.pop()
    with pytest.raises(ValueError, match="exactly three"):
        _validate_dense_evidence_args(args)

    args.dense_evidence_cache_root.append(Path("anytouch"))
    args.dense_evidence_coverage_plan = None
    with pytest.raises(ValueError, match="coverage plan"):
        _validate_dense_evidence_args(args)

    args.dense_evidence_mode = "none"
    with pytest.raises(ValueError, match="forbids cache roots"):
        _validate_dense_evidence_args(args)


def test_dense_evidence_prefix_must_cover_complete_invocation_steps() -> None:
    coverage = argparse.Namespace(training_visit_count=8_000)
    stream = argparse.Namespace(total_steps=TOTAL_STEPS, global_batch_size=4)

    assert (
        _dense_evidence_training_step_prefix(
            coverage,
            stream,
            stop_after_step=2_000,
        )
        == 2_000
    )
    with pytest.raises(ValueError, match="does not cover this invocation"):
        _dense_evidence_training_step_prefix(
            coverage,
            stream,
            stop_after_step=2_001,
        )


@pytest.mark.parametrize("visits", (0, 7_999, True))
def test_dense_evidence_prefix_rejects_partial_global_steps(visits: object) -> None:
    with pytest.raises(ValueError, match="complete global steps"):
        _dense_evidence_training_step_prefix(
            argparse.Namespace(training_visit_count=visits),
            argparse.Namespace(total_steps=TOTAL_STEPS, global_batch_size=4),
            stop_after_step=1,
        )


def test_distributed_prior_host_schedule_uses_rank_max_without_shortening() -> None:
    class FakeReduceOp:
        MAX = object()

    class FakeDist:
        ReduceOp = FakeReduceOp

        @staticmethod
        def all_reduce(value: torch.Tensor, *, op: object) -> None:
            assert op is FakeReduceOp.MAX
            value.copy_(torch.tensor([3, 5], dtype=value.dtype, device=value.device))

    assert _distributed_prior_host_step_schedule(
        (1, 4),
        device="cpu",
        dist=FakeDist,
        torch_module=torch,
    ) == (3, 5)

    with pytest.raises(ValueError, match="positive integers"):
        _distributed_prior_host_step_schedule(
            (0,),
            device="cpu",
            dist=FakeDist,
            torch_module=torch,
        )


def test_adr149_physical_step_observability_binds_the_complete_two_pass_receipt() -> None:
    planned = argparse.Namespace(
        physical_prompt_selection_sha256="a" * 64,
        training=argparse.Namespace(
            physical_control_span_sha256=("b" * 64,),
            selected_segment_indices=(17,),
        ),
    )
    primary = argparse.Namespace(
        prior_control_chunks=(
            argparse.Namespace(token_count=64),
            argparse.Namespace(token_count=3),
        ),
        effective_prior_control_chunks=(
            argparse.Namespace(token_count=64),
            argparse.Namespace(token_count=3),
        ),
    )
    egress = argparse.Namespace(
        source_digest="c" * 64,
        effective_prior_control_chunks=(argparse.Namespace(token_count=9),),
    )
    result = argparse.Namespace(
        v3_prior_traces=(object(), object()),
        attached_egress=object(),
        filter_phase_branches=tuple(object() for _ in range(5)),
    )

    receipt = _physical_step_observability(
        active=True,
        planned=planned,
        primary_batch=primary,
        sequence_batch_count=2,
        egress_batch=egress,
        prior_host_steps_by_batch=(3, 2),
        prior_gradient_suffix_steps_by_batch=(1, 2),
        egress_prior_host_steps=2,
        result=result,
    )

    assert receipt == {
        "active": True,
        "physical_prompt_selection_sha256": "a" * 64,
        "physical_control_span_sha256": ["b" * 64],
        "selected_segment_indices": [17],
        "prior_control_chunk_count": 2,
        "prior_control_chunk_token_counts": [64, 3],
        "prior_host_steps_by_batch": [3, 2],
        "prior_gradient_suffix_steps_by_batch": [1, 2],
        "sequence_batch_count": 2,
        "v3_prior_trace_count": 2,
        "filter_phase_branch_count": 5,
        "expected_filter_phase_branch_count": 5,
        "attached_egress_result": True,
        "egress_source_digest": "c" * 64,
        "egress_prior_host_steps": 2,
    }


def test_adr149_physical_step_observability_rejects_a_missing_prior_trace() -> None:
    planned = argparse.Namespace(
        physical_prompt_selection_sha256="a" * 64,
        training=argparse.Namespace(
            physical_control_span_sha256=("b" * 64,),
            selected_segment_indices=(17,),
        ),
    )
    primary = argparse.Namespace(
        prior_control_chunks=(argparse.Namespace(token_count=1),),
        effective_prior_control_chunks=(argparse.Namespace(token_count=1),),
    )
    result = argparse.Namespace(
        v3_prior_traces=(),
        attached_egress=None,
        filter_phase_branches=(object(), object()),
    )

    with pytest.raises(RuntimeError, match="one prior trace"):
        _physical_step_observability(
            active=True,
            planned=planned,
            primary_batch=primary,
            sequence_batch_count=1,
            egress_batch=None,
            prior_host_steps_by_batch=(1,),
            prior_gradient_suffix_steps_by_batch=(1,),
            egress_prior_host_steps=None,
            result=result,
        )


def test_adr149_physical_receipt_is_validated_before_backward() -> None:
    runner = (
        Path(__file__).resolve().parents[2] / "tools" / "run_lingbot_vla2_task_independent_full.py"
    ).read_text(encoding="utf-8")

    receipt = runner.index("physical_observability = _physical_step_observability(")
    backward = runner.index("training_total.backward()")
    optimizer = runner.index("if not attempt.finish(optimizer_attempt):")
    assert receipt < backward < optimizer


def test_legacy_runner_still_requires_every_cache_input() -> None:
    args = _causal_args("recurrent_state")
    args.posterior_architecture = "legacy_v1"
    for name in (
        "predictive_cache_root",
        "predictive_cache_build_report",
        "predictive_cache_build_report_sha256",
        "current_grid_cache_root",
        "current_grid_cache_build_report",
        "current_grid_cache_build_report_sha256",
    ):
        setattr(args, name, None)
    assert _predictive_assets_required(args)
    assert _current_correction_assets_required(args)
    with pytest.raises(ValueError, match="legacy_v1 requires every"):
        _validate_auxiliary_cache_args(args)


def test_current_cache_report_is_bound_to_consumed_manifest_contract(tmp_path: Path) -> None:
    contract = argparse.Namespace(
        coverage_sha256="1" * 64,
        expected_record_count=19,
        source_keys_sha256="2" * 64,
        stream_plan_sha256="3" * 64,
        encoder_digest="4" * 64,
        temporal_estimator_sha256="5" * 64,
    )
    report = {
        "cache_manifest_sha256": "6" * 64,
        "coverage_sha256": contract.coverage_sha256,
        "expected_record_count": contract.expected_record_count,
        "output_root": str(tmp_path),
        "patch_sha256": "7" * 64,
        "source_keys_sha256": contract.source_keys_sha256,
        "stream_plan_sha256": contract.stream_plan_sha256,
        "teacher_encoder_digest": contract.encoder_digest,
        "temporal_estimator_sha256": contract.temporal_estimator_sha256,
    }
    _validate_current_cache_build_binding(
        report=report,
        contract=contract,
        manifest_sha256="6" * 64,
        output_root=tmp_path,
    )

    # The report patch identifies the cache producer, not the current
    # training consumer. Consumer-only changes must not invalidate content-
    # identical target banks.
    report["patch_sha256"] = "9" * 64
    _validate_current_cache_build_binding(
        report=report,
        contract=contract,
        manifest_sha256="6" * 64,
        output_root=tmp_path,
    )

    report["cache_manifest_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="cache_manifest_sha256"):
        _validate_current_cache_build_binding(
            report=report,
            contract=contract,
            manifest_sha256="6" * 64,
            output_root=tmp_path,
        )


def _current_coverage_fixture(*, source_mask_probability: float):
    temporal = TemporalEstimatorConfig(
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
        source_mask_probability=source_mask_probability,
        maximum_optimizer_lag=8,
    )
    expected = argparse.Namespace(
        dataset_tree_sha256="1" * 64,
        stream_plan_sha256="2" * 64,
        temporal_estimator_sha256=temporal.digest,
        coverage_sha256="3" * 64,
        source_keys_sha256="4" * 64,
        source_global_indices=(10, 20, 30),
    )
    contract = argparse.Namespace(
        dataset_tree_sha256=expected.dataset_tree_sha256,
        stream_plan_sha256=expected.stream_plan_sha256,
        temporal_estimator_sha256=temporal.digest,
        coverage_sha256=expected.coverage_sha256,
        source_keys_sha256=expected.source_keys_sha256,
        expected_record_count=len(expected.source_global_indices),
    )
    return temporal, expected, contract


def test_current_cache_coverage_accepts_exact_temporal_contract() -> None:
    temporal, expected, contract = _current_coverage_fixture(source_mask_probability=0.1)

    coverage, binding = _resolve_current_grid_cache_coverage(
        acceptance_mode="posterior-adoption-route",
        contract=contract,
        expected=expected,
        temporal_config=temporal,
    )

    assert coverage == contract.coverage_sha256
    assert binding["mode"] == "exact_temporal_and_source_coverage"


def test_posterior_adoption_dose_reuses_only_the_exact_registered_source_set() -> None:
    run_temporal, expected, contract = _current_coverage_fixture(source_mask_probability=1.0)
    donor_temporal = replace(
        run_temporal,
        source_mask_probability=ADR149_OMITTED_STATIC_PROBABILITY,
    )
    contract.temporal_estimator_sha256 = donor_temporal.digest
    contract.coverage_sha256 = "5" * 64

    coverage, binding = _resolve_current_grid_cache_coverage(
        acceptance_mode="posterior-adoption-dose",
        contract=contract,
        expected=expected,
        temporal_config=run_temporal,
    )

    assert coverage == contract.coverage_sha256
    assert binding == {
        "mode": "exact_source_set_reuse_for_route_dose",
        "cache_temporal_estimator_sha256": donor_temporal.digest,
        "run_temporal_estimator_sha256": run_temporal.digest,
        "source_keys_sha256": expected.source_keys_sha256,
        "record_count": 3,
        "content_invariance": (
            "same source RGB set and frozen teacher; only route sampling probability differs"
        ),
    }

    contract.source_keys_sha256 = "6" * 64
    with pytest.raises(RuntimeError, match="changed content coverage"):
        _resolve_current_grid_cache_coverage(
            acceptance_mode="posterior-adoption-dose",
            contract=contract,
            expected=expected,
            temporal_config=run_temporal,
        )


def test_non_dose_mode_rejects_temporal_only_cache_alias() -> None:
    run_temporal, expected, contract = _current_coverage_fixture(source_mask_probability=1.0)
    contract.temporal_estimator_sha256 = replace(
        run_temporal,
        source_mask_probability=ADR149_OMITTED_STATIC_PROBABILITY,
    ).digest

    with pytest.raises(RuntimeError, match="exact 30k plan"):
        _resolve_current_grid_cache_coverage(
            acceptance_mode="posterior-adoption-route",
            contract=contract,
            expected=expected,
            temporal_config=run_temporal,
        )


def test_zero_treatment_clears_model_state_and_loss_side_gauge() -> None:
    prepared = argparse.Namespace(
        previous_state=object(),
        previous_state_valid=torch.ones(2, dtype=torch.bool),
        previous_row_bindings=((("a", 0),), (("b", 1),)),
    )
    state, valid, bindings = _objective_posterior_inputs(
        mode="zero_state",
        prepared=prepared,
        torch_module=torch,
    )
    assert state is None
    assert valid.tolist() == [False, False]
    assert bindings == ((), ())
    assert _staged_row_bindings(
        mode="zero_state",
        observed=prepared.previous_row_bindings,
    ) == ((), ())


def test_recurrent_treatment_preserves_model_state_and_loss_side_gauge() -> None:
    prepared = argparse.Namespace(
        previous_state=object(),
        previous_state_valid=torch.tensor([True, False]),
        previous_row_bindings=((("a", 0),), ()),
    )
    state, valid, bindings = _objective_posterior_inputs(
        mode="recurrent_state",
        prepared=prepared,
        torch_module=torch,
    )
    assert state is prepared.previous_state
    assert valid is prepared.previous_state_valid
    assert bindings is prepared.previous_row_bindings
    assert (
        _staged_row_bindings(
            mode="recurrent_state",
            observed=bindings,
        )
        is bindings
    )


def test_adr146_only_publishes_registered_branch_and_endpoint_checkpoints() -> None:
    assert _causal_checkpoint_due(mode="current_frame_branch", global_step=200)
    assert not _causal_checkpoint_due(mode="current_frame_branch", global_step=199)
    assert _causal_checkpoint_due(mode="zero_state", global_step=300)
    assert _causal_checkpoint_due(mode="recurrent_state", global_step=300)
    assert not _causal_checkpoint_due(mode="zero_state", global_step=250)


def test_zero_predictive_weight_keeps_graph_width_bound_to_validated_contract() -> None:
    runner = (
        Path(__file__).resolve().parents[2] / "tools" / "run_lingbot_vla2_task_independent_full.py"
    ).read_text(encoding="utf-8")

    assert "predictive_contract.hidden_size" in runner
    assert "predictive_cache.contract.hidden_size" not in runner


def test_production_launchers_bind_explicit_objective_weights() -> None:
    root = Path(__file__).resolve().parents[2]
    production = (root / "adr141" / "run_task_independent_full.sh").read_text()
    candidate = (root / "adr141" / "launch_task_independent_full_v5.sh").read_text()

    assert "[[ $# -ne 6 ]]" in production
    assert '--entity-weight "$ENTITY_WEIGHT"' in production
    assert '--predictive-weight "$PREDICTIVE_WEIGHT"' in production
    assert 'fresh "$run_dir" 30000 0 0.08 0.08' in candidate
    assert 'publish_pointer "$run_dir" "$ROOT/ACTIVE_RUN_DIR"' in candidate
    assert 'publish_pointer "$log" "$ROOT/ACTIVE_TRAIN_LOG"' in candidate


def test_runtime_world_size_accepts_only_supported_single_host_topologies() -> None:
    assert SUPPORTED_WORLD_SIZES == (2, 4)
    assert _runtime_world_size({}) == 2
    assert _runtime_world_size({"WORLD_SIZE": "2"}) == 2
    assert _runtime_world_size({"WORLD_SIZE": "4"}) == 4

    for value in ("0", "3", "04", "eight"):
        with pytest.raises(RuntimeError):
            _runtime_world_size({"WORLD_SIZE": value})


def test_four_rank_runner_requires_one_complete_frozen_stream_contract() -> None:
    args = argparse.Namespace(
        stream_plan=None,
        stream_plan_sha256=None,
        representation_split=None,
        representation_split_sha256=None,
        evaluation_plan=None,
        evaluation_plan_sha256=None,
    )
    assert not _validate_frozen_stream_args(args, world_size=2)
    with pytest.raises(ValueError, match="four-rank training requires"):
        _validate_frozen_stream_args(args, world_size=4)

    args.stream_plan = Path("stream.json")
    with pytest.raises(ValueError, match="requires plan, split"):
        _validate_frozen_stream_args(args, world_size=4)

    args.stream_plan_sha256 = "a" * 64
    args.representation_split = Path("split.json")
    args.representation_split_sha256 = "b" * 64
    args.evaluation_plan = Path("evaluation.json")
    args.evaluation_plan_sha256 = "c" * 64
    assert _validate_frozen_stream_args(args, world_size=4)


def test_adr147_launcher_is_layerwise_multi_gpu_and_has_no_legacy_cache_surface() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "adr147" / "run_layerwise_v2.sh").read_text()

    assert "[[ $# -ne 4 ]]" in launcher
    assert "WORLD_SIZE=${PICF_WORLD_SIZE:-2}" in launcher
    assert '--nproc-per-node="$WORLD_SIZE"' in launcher
    assert "2) GPU_LIST=0,1" in launcher
    assert "GPU_LIST=0,1,2,3" in launcher
    assert '--stream-plan "$STREAM_PLAN"' in launcher
    assert '--representation-split "$REPRESENTATION_SPLIT"' in launcher
    assert "--posterior-architecture layerwise_v2" in launcher
    assert "--causal-ablation-mode none" in launcher
    assert "--entity-weight 0.08" in launcher
    assert "--predictive-weight 0.0" in launcher
    assert "--local-bptt-probability 0.0" in launcher
    assert "--overshoot-probability 0.0" in launcher
    assert "--source-mask-probability 0.0" in launcher
    assert "predictive-cache" not in launcher
    assert "current-grid-cache" not in launcher
    assert "status --porcelain=v1 --untracked-files=all" in launcher
    assert "PICF_EXPECTED_GIT_DIFF_SHA256" in launcher
    assert "diff HEAD --binary --no-ext-diff" in launcher
    assert "ls-files --others --exclude-standard" in launcher


def test_checkpoint_precedes_optional_causal_diagnostics_at_shared_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tools" / "run_lingbot_vla2_task_independent_full.py").read_text()
    loop = runner[runner.index("        while global_step < args.stop_after_step:") :]

    assert loop.index("            if checkpoint_due:") < loop.index(
        "            causal_report = run_causal_diagnostic("
    )
    assert '"causal_diagnostic_variant_started"' in runner
    assert '"causal_diagnostic_variant_completed"' in runner


def test_layerwise_causal_diagnostic_covers_predictive_correction() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")
    required = (
        "correction_diagnostic = bool(",
        "run_task_independent_calvin_joint_sequence_objective(",
        '"predictive_family_loss"',
        'action_loss_route = "routed_omitted_static"',
        '"factual_official_action_loss"',
        '("correction/", "filter_prior/", "filter_posterior/")',
        'report["correction_diagnostic_active"] = correction_diagnostic',
    )
    for value in required:
        assert value in runner


def test_two_pass_diagnostic_crosses_prior_state_with_observation_availability() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")

    assert TWO_PASS_FILTER_DIAGNOSTIC_SCHEMA == "picf-next.adr149-two-pass-filter-diagnostic/v1"
    assert TWO_PASS_FILTER_DIAGNOSTIC_STEPS == (250, 500, 1_000, 2_000)
    required = (
        'args.posterior_architecture == "two_pass_v3"',
        "sample_qwen_whole_view_omission(",
        "omitted_static_view=diagnostic_omission",
        '"omitted_static_action_loss"',
        '"two_pass_filter_diagnostics"',
        "snapshot_official_runtime_buffers()",
        "restore_official_runtime_buffers(runtime_snapshot)",
        "def run_v3_control_intervention(",
        "zero_executed_control(chunk)",
        "native_context_from_prior_trace(",
        '"Pass A controls differ; Pass B observation, controls, action target, "',
        "gathered_prior_rows = [",
        '"wrong_row": layerwise_prior_trace_with_tensor(',
        '"cross_batch": layerwise_prior_trace_with_tensor(',
        "episode_address_state=peer_prior_address",
        '"direct_prior_intervention": {',
        '"Pass A is held factual; only the prior trace supplied to Pass B "',
        'report["control_intervention"] = run_v3_control_intervention(',
    )
    for value in required:
        assert value in runner


def test_two_pass_action_curve_uses_the_frozen_lbot_input_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")

    assert TWO_PASS_ACTION_EVALUATION_SCHEMA == "picf-next.adr149-cold-action-snapshot/v2"
    assert TWO_PASS_ACTION_EVALUATION_STEPS == (0, 20, 100, 200, 500, 1_000, 1_500, 2_000)
    required = (
        "EntityEvaluationPlan.load(args.evaluation_plan)",
        "build_entity_evaluation_plan(",
        "build_native_calvin_replay_batch(",
        "_evaluation_replay_seed(",
        '"state_mode": "cold_reset"',
        '"evaluation_input_sha256"',
        "_summarize_action_partition(",
        "summarize_entity_evaluation_partition(",
        '"physical_sidecar_read_during_model_forward": False',
        '"physical_sidecar_read_after_model_forward_for_metrics": True',
        '"picf-next.adr202-heldout-entity-snapshot/v1"',
        "run_registered_action_evaluations(0)",
    )
    for value in required:
        assert value in runner

    causal_warm_required = (
        "ADR210_CAUSAL_WARM_ACTION_EVALUATION_STEPS = (100,)",
        "build_distributed_causal_warm_evaluation_schedule(",
        "run_causal_warm_native_videomt_lingbot_evaluation(",
        '"state_mode": "causal_warm_four_past_frames"',
        '"action_suffix_executed_only_at_current": True',
        '"causal-warm current input differs from cold field {name}"',
        '"history_transitions": ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS',
    )
    for value in causal_warm_required:
        assert value in runner


def test_action_interventions_pair_the_official_training_forward_with_backward() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")
    start = runner.index("            def diagnostic_action_output(")
    end = runner.index("            def maximum_action_drift(", start)
    intervention = runner[start:end]

    assert "result = action_objective(batch)" in intervention
    assert "result.primary.official_action_loss.backward()" in intervention
    assert "optimizer.zero_grad(set_to_none=True)" in intervention
    assert "gc.collect()" not in intervention
    assert "torch.no_grad()" not in intervention


def test_matched_wla_action_evaluation_fails_closed_on_backend_dispatch() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tools/run_lingbot_vla2_task_independent_full.py").read_text(
        encoding="utf-8"
    )

    assert 'result.action_backend != WLA_COMPLETE_ACTION_BACKEND' in runner
    assert '"action_backend": result.action_backend' in runner
    assert "matched WLA action evaluation executed another action backend" in runner


def test_large_model_artifacts_are_validated_once_after_distributed_init() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tools" / "run_lingbot_vla2_task_independent_full.py").read_text()

    init = runner.index("dist.init_process_group(")
    rank_zero_validation = runner.index(
        '                    "checkpoint": validate_checkpoint(args.checkpoint_dir),'
    )
    broadcast = runner.index("        dist.broadcast_object_list(artifact_contract, src=0)")
    assert init < rank_zero_validation < broadcast
    assert runner.count("validate_checkpoint(args.checkpoint_dir)") == 1
    assert runner.count("validate_processor(args.processor_dir)") == 1


def test_four_gpu_restore_rejects_wrong_hardware_before_runtime_extraction() -> None:
    root = Path(__file__).resolve().parents[2]
    restore = (root / "adr147" / "restore_four_gpu_runtime.sh").read_text()

    hardware_probe = restore.index("mapfile -t GPU_ROWS")
    runtime_restore = restore.index("RUNTIME_STAGE=$(mktemp -d")
    assert hardware_probe < runtime_restore


def test_four_gpu_restore_never_trusts_an_existing_mutable_runtime() -> None:
    root = Path(__file__).resolve().parents[2]
    restore = (root / "adr147" / "restore_four_gpu_runtime.sh").read_text()

    assert 'sha256sum -c "$ARCHIVE.sha256"' in restore
    assert '[[ ! -L "$RUNTIME" ]]' in restore
    assert 'rm -rf "$RUNTIME"' in restore
    assert 'mv -T "${EXTRACTED_ROOTS[0]}" "$RUNTIME"' in restore
    assert 'if [[ ! -x "$RUNTIME/bin/python" ]]' not in restore


def test_four_gpu_launcher_requires_accepted_matched_lbot() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "adr147" / "launch_four_gpu_30k.sh").read_text()

    assert "usage: $0 RUN_DIR MATCHED_LBOT_REPORT" in launcher
    assert '[[ "$MATCHED_LBOT_REPORT" == /mnt/* && -f "$MATCHED_LBOT_REPORT" ]]' in launcher
    assert 'report.get("status") != "PASS"' in launcher
    assert 'report.get("world_size") != 4 or report.get("steps") != 200' in launcher
    assert 'report.get("plan_sha256") != plan.get("plan_sha256")' in launcher
    assert launcher.index('"$REPO/adr147/restore_four_gpu_runtime.sh"') < launcher.index(
        '"$PYTHON" - "$MATCHED_LBOT_REPORT"'
    )
    assert launcher.index('"$PYTHON" - "$MATCHED_LBOT_REPORT"') < launcher.index(
        'exec "$REPO/adr147/run_layerwise_v2.sh"'
    )


@pytest.mark.parametrize(
    ("phase", "stop_step", "load_step", "message"),
    (
        ("fresh", "08", "0", "canonical non-negative decimal"),
        ("fresh", "30001", "0", "no greater than 30000"),
        ("fresh", "2000", "2000", "greater than load step"),
        ("resume", "4000", "1", "positive 2000-step boundary"),
    ),
)
def test_adr147_launcher_fails_before_environment_checks_for_invalid_boundaries(
    tmp_path: Path,
    phase: str,
    stop_step: str,
    load_step: str,
    message: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            "bash",
            str(root / "adr147" / "run_layerwise_v2.sh"),
            phase,
            str(tmp_path / "run"),
            stop_step,
            load_step,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_adr147_launcher_rejects_an_unregistered_world_size_before_host_checks(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            "bash",
            str(root / "adr147" / "run_layerwise_v2.sh"),
            "fresh",
            str(tmp_path / "run"),
            "30000",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PICF_WORLD_SIZE": "3"},
    )

    assert completed.returncode == 2
    assert "PICF_WORLD_SIZE must be exactly 2 or 4" in completed.stderr


def test_external_stop_contract_distinguishes_immediate_and_checkpoint_requests(
    tmp_path: Path,
) -> None:
    assert not _external_stop_requested(run_dir=tmp_path, checkpoint_due=False)

    (tmp_path / "STOP_AFTER_CHECKPOINT").write_text("stop after durable checkpoint\n")
    assert not _external_stop_requested(run_dir=tmp_path, checkpoint_due=False)
    assert _external_stop_requested(run_dir=tmp_path, checkpoint_due=True)

    (tmp_path / "STOP").write_text("stop after durable evidence\n")
    assert _external_stop_requested(run_dir=tmp_path, checkpoint_due=False)


def test_cache_manifest_respects_distinct_publication_schemas(tmp_path) -> None:
    predictive = tmp_path / "predictive"
    predictive.mkdir()
    (predictive / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "predictive",
                "complete": True,
                "contract": {},
                "shards": [],
            }
        ),
        encoding="utf-8",
    )
    current = tmp_path / "current"
    current.mkdir()
    (current / "manifest.json").write_text(
        json.dumps({"schema": "current", "contract": {}, "shards": []}),
        encoding="utf-8",
    )

    assert _cache_manifest(predictive, require_complete_field=True)[0]["complete"] is True
    assert _cache_manifest(current, require_complete_field=False)[0]["schema"] == "current"
    with pytest.raises(ValueError, match="incomplete"):
        _cache_manifest(current, require_complete_field=True)


def test_progress_evidence_is_rank_and_step_explicit(capsys) -> None:
    _emit_progress(
        "batch_ready",
        rank=1,
        global_step=15,
        details={"local_bptt_steps": 3},
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema": "picf-next.task-independent-full-progress/v1",
        "event": "batch_ready",
        "rank": 1,
        "global_step": 15,
        "local_bptt_steps": 3,
    }


@pytest.mark.parametrize(
    "event",
    [
        "step_graph_released",
        "causal_diagnostic_variant_started",
        "causal_diagnostic_variant_completed",
        "causal_diagnostic_variant_released",
    ],
)
def test_progress_evidence_accepts_release_events(event, capsys) -> None:
    _emit_progress(
        event,
        rank=0,
        global_step=250,
        details={"variant": "wrong_time"},
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema": "picf-next.task-independent-full-progress/v1",
        "event": event,
        "rank": 0,
        "global_step": 250,
        "variant": "wrong_time",
    }


def test_causal_diagnostic_releases_graph_tensors_between_variants() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")
    diagnostic = runner[
        runner.index("        def run_causal_diagnostic(") : runner.index(
            "        def run_full_modal_action_adoption_phase()"
        )
    ]
    for token in (
        "factual_action_loss,",
        "omitted_policy,",
        "action_loss,",
        "gc.collect()",
        "torch.cuda.empty_cache()",
        '"causal_diagnostic_variant_released"',
    ):
        assert token in diagnostic


def test_progress_evidence_rejects_reserved_overrides() -> None:
    with pytest.raises(ValueError, match="override reserved fields"):
        _emit_progress(
            "step_started",
            rank=0,
            global_step=1,
            details={"rank": 1},
        )


def test_resume_truncates_uncheckpointed_metric_and_visual_outputs(tmp_path) -> None:
    journal = tmp_path / "metrics" / "rank_journal" / "rank_0.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        "\n".join(json.dumps({"global_step": step}) for step in range(1, 2_101)) + "\n",
        encoding="utf-8",
    )
    metric = tmp_path / "metrics" / "steps_00002001_00002100.json"
    metric.write_text(json.dumps({"end_global_step": 2_100}), encoding="utf-8")
    visual_directories = (
        "entity_visuals",
        "native_videomt_query_visuals",
    )
    visuals = []
    for directory in visual_directories:
        visual = tmp_path / directory / "step_00002100"
        visual.mkdir(parents=True)
        (visual / "artifact.png").write_bytes(b"not-an-image")
        visuals.append(visual)
    summary = tmp_path / "run_summary_step_00002100.json"
    summary.write_text("{}", encoding="utf-8")
    diagnostic = tmp_path / "layerwise_causal_diagnostics" / "step_00002100"
    diagnostic.mkdir(parents=True)
    (diagnostic / "rank_0.json").write_text("{}", encoding="utf-8")
    two_pass_diagnostic = tmp_path / "two_pass_filter_diagnostics" / "step_00002100"
    two_pass_diagnostic.mkdir(parents=True)
    (two_pass_diagnostic / "rank_0.json").write_text("{}", encoding="utf-8")
    action_evaluation = tmp_path / "action_evaluations" / "step_00002100"
    action_evaluation.mkdir(parents=True)
    (action_evaluation / "distributed.json").write_text("{}", encoding="utf-8")
    heldout_directories = (
        "heldout_entity_evaluation",
        "heldout_entity_evaluations",
        "heldout_native_videomt_anchor_evaluation",
        "heldout_native_videomt_anchor_evaluations",
        "native_videomt_modality_interventions",
    )
    heldout_publications = []
    for directory in heldout_directories:
        publication = tmp_path / directory / "step_00002100"
        publication.mkdir(parents=True)
        (publication / "distributed.json").write_text("{}", encoding="utf-8")
        heldout_publications.append(publication)
    action_curve = tmp_path / "action_evaluation_curve_step_00002100.json"
    action_curve.write_text("{}", encoding="utf-8")

    handle = _prepare_rank_metric_journal(
        journal,
        phase="resume",
        load_global_step=2_000,
    )
    handle.close()
    _prune_resume_publications(tmp_path, load_global_step=2_000)

    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2_000
    assert records[-1]["global_step"] == 2_000
    assert not metric.exists()
    assert all(not visual.exists() for visual in visuals)
    assert not summary.exists()
    assert not diagnostic.exists()
    assert not two_pass_diagnostic.exists()
    assert not action_evaluation.exists()
    assert all(not publication.exists() for publication in heldout_publications)
    assert not action_curve.exists()


def test_visual_metadata_is_durable_before_optional_causal_diagnostics() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")
    loop = runner[runner.index("        while global_step < args.stop_after_step:") :]
    manifest_write = loop.index('"picf-next.task-independent-entity-visual-manifest/v1"')
    causal_diagnostic = loop.index("            causal_report = run_causal_diagnostic(")
    assert manifest_write < causal_diagnostic
    assert '/ "artifacts.json"' in loop


def test_dense_coverage_replay_preserves_causal_warm_history_contract() -> None:
    runner = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "run_lingbot_vla2_task_independent_full.py"
    ).read_text(encoding="utf-8")
    start = runner.index("reproduced_dense_coverage = build_calvin_dense_evidence_coverage_plan(")
    end = runner.index("                if reproduced_dense_coverage !=", start)
    replay = runner[start:end]

    assert (
        "dense_evidence_coverage.evaluation_history_transition_count"
        in replay
    )
    assert "schema=dense_evidence_coverage.schema" in replay


def test_adr210_launcher_uses_warm_v2_coverage_and_real_30k_boundary() -> None:
    launcher = (
        Path(__file__).resolve().parents[2]
        / "adr210"
        / "run_causal_warm_action_gate_4gpu.sh"
    ).read_text(encoding="utf-8")

    assert "contracts/causal-warm-4gpu-30k-v2" in launcher
    assert "caches/causal-warm-history-v2" in launcher
    assert "PICF_ADR210_ENABLE_DENSE_SUPPLEMENT=1" in launcher
    assert "PICF_STOP_AFTER_STEP=30000" in launcher
    assert "PICF_STOP_AFTER_STEP=100" not in launcher


def test_sequential_omitted_backward_preserves_one_optimizer_transaction() -> None:
    runner = (
        Path(__file__).resolve().parents[2] / "tools" / "run_lingbot_vla2_task_independent_full.py"
    ).read_text(encoding="utf-8")

    factual_backward = runner.index("factual_backward_loss.backward()")
    spill_factual = runner.index(
        "spill_fsdp2_factual_gradients_to_cpu(",
        factual_backward,
    )
    spill_failure_exchange = runner.index(
        "spill_failures = _distributed_pre_backward_failures(",
        spill_factual,
    )
    restore_prior = runner.index(
        "restore_rank_execution_state(prior_entry_execution_state)",
        factual_backward,
    )
    replay_prior = runner.index("run_native_v3_prior_chain(", restore_prior)
    restore_omitted = runner.index("restore_rank_execution_state(omitted_entry_execution_state)")
    omitted_backward = runner.index("omitted_result.backward_loss.backward()")
    merge_factual = runner.index(
        "merge_fsdp2_factual_gradients_from_cpu(",
        omitted_backward,
    )
    merge_failure_exchange = runner.index(
        "merge_failures = _distributed_pre_backward_failures(",
        merge_factual,
    )
    transaction_failure_exchange = runner.index(
        "backward_failures = _distributed_pre_backward_failures(",
        merge_failure_exchange,
    )
    stage = runner.index("attempt.stage(", merge_factual)
    optimizer = runner.index("if not attempt.finish(optimizer_attempt):", stage)

    assert (
        factual_backward
        < spill_factual
        < spill_failure_exchange
        < restore_prior
        < replay_prior
        < restore_omitted
        < omitted_backward
        < merge_factual
        < merge_failure_exchange
        < stage
        < transaction_failure_exchange
        < optimizer
    )
    assert "set_fsdp2_is_last_backward" not in runner
    assert 'buffer.detach().to(device="cpu", copy=True)' in runner
    assert '"factual_backward_completed"' in runner
    assert '"factual_gradients_spilled"' in runner
    assert '"factual_gradients_merged"' in runner
    assert '"--sequential-factual-gradient-storage"' in runner


def test_full_runner_uses_rank_safe_lingbot_gradient_clipping() -> None:
    runner = (
        Path(__file__).resolve().parents[2] / "tools" / "run_lingbot_vla2_native_full.py"
    ).read_text(encoding="utf-8")
    start = runner.index("def _optimizer_attempt(")
    end = runner.index("\ndef _emit_step_progress(", start)
    optimizer_attempt = runner[start:end]

    assert "clip_lingbot_distributed_l2_grad_norm_(" in optimizer_attempt
    assert "torch_module.nn.utils.clip_grad_norm_(" not in optimizer_attempt
