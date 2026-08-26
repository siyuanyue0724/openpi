#!/usr/bin/env python3
"""Validate and compare the exact ADR-207 and matched LingBot action curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.compare_adr176_matched_action_curves import (
    LBOT_SCHEMA,
    _canonical_sha256,
    _load_snapshot,
    _parse_steps,
    _snapshot_path,
    compare_curves,
)

ADR207_ACTION_SCHEMA = "picf-next.adr207-cold-native-query-action-snapshot/v1"
ADR207_ANCHOR_SCHEMA = "picf-next.adr207-heldout-native-query-anchor/v1"
ADR207_INTERVENTION_SCHEMA = (
    "picf-next.adr207-heldout-native-query-modality-intervention/v1"
)
ADR207_REPORT_SCHEMA = "picf-next.adr207-matched-action-curve-comparison/v1"
LBOT_RUN_REPORT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-lbot.v1"
PICF_RUN_MANIFEST_SCHEMA = "picf-next.task-independent-full-runner/v18"
PICF_CHECKPOINT_SCHEMA = "picf-next.task-independent-full-checkpoint/v7"
ADR207_ARCHITECTURE = "native_videomt_query_posterior_v1"
LBOT_ARCHITECTURE = "released_lingbot_vla2_action_policy"
ADR207_PROFILE = "adr207_native_videomt_query_posterior_v1"
ADR207_SOURCE_MODE = "trainable-adapted-native-query-causal-c5"
ADR207_ADAPTED_SOURCE_SHA256 = (
    "4437d8632c4e3877adcf5cfec5bf6e673445ad9d3d2de3a3afdd924651b5bd5d"
)
REGISTERED_CURVES = (
    (0, 20, 100, 200),
    (0, 20, 100, 200, 500, 1_000, 1_500, 2_000),
)
INTERVENTION_STEPS = frozenset({200, 2_000})
SOURCE_HARD_IOU_MINIMUM = 0.70
SOURCE_RECALL_AT_50_MINIMUM = 0.85


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picf-run-dir", type=Path, required=True)
    parser.add_argument("--lbot-run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=_parse_steps, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_string(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not an object")
    return value


def _validate_lingbot_base_family(value: object, *, name: str) -> str:
    identity = dict(_mapping(value, name=name))
    observed = _sha256_string(
        identity.pop("artifact_sha256", None),
        name=f"{name}.artifact_sha256",
    )
    if identity.get("schema") != "picf-next.lingbot-base-family.v1":
        raise ValueError(f"{name} schema differs")
    if identity.get("architecture") != LBOT_ARCHITECTURE:
        raise ValueError(f"{name} architecture differs")
    if identity.get("attention_implementation") != "flex_cached":
        raise ValueError(f"{name} attention implementation differs")
    if identity.get("trainable_scope") != "full-host":
        raise ValueError(f"{name} trainable scope differs")
    if identity.get("maximum_control_tokens") != 64:
        raise ValueError(f"{name} control-token budget differs")
    for field in (
        "source_commit",
        "native_patch_sha256",
        "checkpoint_revision",
        "processor_revision",
    ):
        digest = identity.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name}.{field} is not a lowercase digest")
    for field in ("checkpoint_assets", "processor_assets"):
        assets = identity.get(field)
        if not isinstance(assets, list) or not assets:
            raise ValueError(f"{name}.{field} is absent")
    optimizer = _mapping(identity.get("optimizer_contract"), name=f"{name} optimizer")
    learning_rate = float(optimizer.get("learning_rate", math.nan))
    if not math.isfinite(learning_rate) or learning_rate != 1e-4:
        raise ValueError(f"{name} optimizer learning rate differs")
    expected = _canonical_sha256(identity)
    if observed != expected:
        raise ValueError(f"{name} semantic identity differs")
    return observed


def _validate_evaluation_digest(snapshot: Mapping[str, Any], *, adr207: bool) -> None:
    samples = snapshot.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("action snapshot has no samples")
    fields = ["sample_key", "source_digest", "model_inputs_sha256"]
    if adr207:
        fields.append("native_source_rgb_sha256")
    inputs = [{field: sample.get(field) for field in fields} for sample in samples]
    observed = _sha256_string(
        snapshot.get("evaluation_input_sha256"),
        name="evaluation_input_sha256",
    )
    if observed != _canonical_sha256(inputs):
        raise ValueError("action snapshot evaluation-input SHA-256 differs")


def _validate_companion_receipt(
    receipt_value: object,
    *,
    expected_path: Path,
    expected_schema: str,
    expected_step: int,
) -> dict[str, Any]:
    receipt = _mapping(receipt_value, name=f"{expected_schema} receipt")
    if receipt.get("status", "PASS") != "PASS":
        raise ValueError(f"{expected_schema} receipt did not pass")
    if expected_path.is_symlink() or not expected_path.is_file():
        raise ValueError(f"required direct companion snapshot is absent: {expected_path}")
    embedded_path = receipt.get("path")
    if (
        not isinstance(embedded_path, str)
        or Path(embedded_path).resolve() != expected_path.resolve()
    ):
        raise ValueError(f"{expected_schema} receipt points to another artifact")
    companion = json.loads(expected_path.read_text(encoding="ascii"))
    if not isinstance(companion, dict) or companion.get("schema") != expected_schema:
        raise ValueError(f"unexpected companion snapshot schema: {expected_path}")
    if (
        companion.get("status") != "PASS"
        or companion.get("checkpoint_global_step") != expected_step
    ):
        raise ValueError(f"companion is not a passing step-{expected_step} result")
    semantic = dict(companion)
    artifact_sha256 = semantic.pop("artifact_sha256", None)
    if artifact_sha256 != _canonical_sha256(semantic):
        raise ValueError(f"companion semantic SHA-256 differs: {expected_path}")
    if receipt.get("artifact_sha256") != companion["artifact_sha256"]:
        raise ValueError(f"{expected_schema} receipt has another semantic identity")
    if receipt.get("file_sha256") != _file_sha256(expected_path):
        raise ValueError(f"{expected_schema} receipt has another file identity")
    return companion


def _validate_adr207_snapshot(
    snapshot: dict[str, Any],
    *,
    run_dir: Path,
    step: int,
) -> dict[str, dict[str, float]]:
    expected = {
        "architecture_identity": ADR207_ARCHITECTURE,
        "picf_graph_installed": True,
        "physical_sidecar_read_during_model_forward": False,
        "physical_sidecar_read_after_model_forward_for_metrics": True,
        "task_scorer_present": False,
        "action_suffix_executed": True,
        "state_mode": "cold_reset",
    }
    for field, value in expected.items():
        if snapshot.get(field) != value:
            raise ValueError(f"ADR-207 action snapshot differs for {field}")
    base_family_sha256 = _sha256_string(
        snapshot.get("lingbot_base_family_sha256"),
        name="ADR-207 LingBot base family",
    )
    _validate_evaluation_digest(snapshot, adr207=True)
    for sample in snapshot["samples"]:
        if sample.get("native_source_query_count") != 200:
            raise ValueError("ADR-207 action snapshot did not expose all 200 source queries")
        _sha256_string(
            sample.get("native_source_rgb_sha256"),
            name="native_source_rgb_sha256",
        )
        _sha256_string(sample.get("model_inputs_sha256"), name="model_inputs_sha256")
        if (
            sample.get("prior_trace_finite") is not True
            or sample.get("posterior_finite") is not True
        ):
            raise ValueError("ADR-207 action snapshot contains a non-finite posterior trace")

    anchor_path = (
        run_dir
        / "heldout_native_videomt_anchor_evaluations"
        / f"step_{step:08d}"
        / "distributed.json"
    )
    anchor = _validate_companion_receipt(
        snapshot.get("heldout_anchor_evaluation"),
        expected_path=anchor_path,
        expected_schema=ADR207_ANCHOR_SCHEMA,
        expected_step=step,
    )
    for field in (
        "stream_plan_sha256",
        "representation_split_sha256",
        "evaluation_plan_sha256",
        "evaluation_input_sha256",
        "lingbot_base_family_sha256",
    ):
        if anchor.get(field) != snapshot.get(field):
            raise ValueError(f"ADR-207 action/anchor snapshots differ for {field}")
    if anchor.get("source_query_count") != 200:
        raise ValueError("ADR-207 held-out anchor snapshot did not expose all 200 queries")
    partition_summaries = _mapping(
        anchor.get("partition_summaries"),
        name="ADR-207 held-out anchor partition summaries",
    )
    source_spatial_gate: dict[str, dict[str, float]] = {}
    for partition in ("validation", "heldout"):
        summary = _mapping(
            partition_summaries.get(partition),
            name=f"ADR-207 {partition} anchor summary",
        )
        metrics: dict[str, float] = {}
        for field in ("mean_soft_iou", "mean_binary_iou", "recall_at_50"):
            metric = float(summary.get(field, math.nan))
            if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
                raise ValueError(f"ADR-207 {partition} anchor {field} is invalid")
            metrics[field] = metric
        if metrics["mean_binary_iou"] < SOURCE_HARD_IOU_MINIMUM:
            raise ValueError(f"ADR-207 {partition} anchor hard-IoU gate failed")
        if metrics["recall_at_50"] < SOURCE_RECALL_AT_50_MINIMUM:
            raise ValueError(f"ADR-207 {partition} anchor Recall@0.5 gate failed")
        source_spatial_gate[partition] = metrics

    intervention = snapshot.get("full_modal_action_intervention")
    if step in INTERVENTION_STEPS:
        intervention_path = (
            run_dir
            / "native_videomt_modality_interventions"
            / f"step_{step:08d}"
            / "distributed.json"
        )
        intervention_snapshot = _validate_companion_receipt(
            intervention,
            expected_path=intervention_path,
            expected_schema=ADR207_INTERVENTION_SCHEMA,
            expected_step=step,
        )
        if intervention_snapshot.get("architecture_identity") != ADR207_ARCHITECTURE:
            raise ValueError("ADR-207 modality intervention used another architecture")
        if intervention_snapshot.get("source_query_count") != 200:
            raise ValueError("ADR-207 modality intervention did not freeze all 200 queries")
        if intervention_snapshot.get("lingbot_base_family_sha256") != base_family_sha256:
            raise ValueError("ADR-207 modality intervention used another LingBot base family")
        modalities = _mapping(
            intervention_snapshot.get("modalities"),
            name="ADR-207 modality intervention modalities",
        )
        if set(modalities) != {"anytouch", "sonata", "vjepa"}:
            raise ValueError("ADR-207 modality intervention did not cover every dense modality")
    elif intervention is not None:
        raise ValueError("ADR-207 modality intervention was published at an unregistered step")
    return source_spatial_gate


def _validate_lbot_snapshot(snapshot: dict[str, Any]) -> None:
    expected = {
        "architecture_identity": LBOT_ARCHITECTURE,
        "picf_graph_installed": False,
        "physical_sidecar_read": False,
        "task_scorer_present": False,
        "action_suffix_executed": True,
        "posterior_present": False,
    }
    for field, value in expected.items():
        if snapshot.get(field) != value:
            raise ValueError(f"matched LingBot snapshot differs for {field}")
    _sha256_string(
        snapshot.get("lingbot_base_family_sha256"),
        name="matched LingBot base family",
    )
    _validate_evaluation_digest(snapshot, adr207=False)
    for sample in snapshot["samples"]:
        _sha256_string(sample.get("model_inputs_sha256"), name="model_inputs_sha256")


def _validate_picf_run_manifest(
    *,
    run_dir: Path,
    steps: tuple[int, ...],
    snapshots: Sequence[dict[str, Any]],
) -> tuple[str, int, str]:
    path = run_dir / "run_manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required direct ADR-207 run manifest is absent: {path}")
    value = _mapping(
        json.loads(path.read_text(encoding="ascii")),
        name="ADR-207 run manifest",
    )
    expected = {
        "schema": PICF_RUN_MANIFEST_SCHEMA,
        "status": "DECLARED",
        "declared_total_steps": 30_000,
        "metrics_every": 100,
        "visual_every": 250,
        "checkpoint_every": 2_000,
        "gradient_accumulation_steps": 1,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(f"ADR-207 run manifest differs for {field}")
    world_size = value.get("world_size")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size not in {2, 4}
        or value.get("global_batch_size") != world_size
    ):
        raise ValueError("ADR-207 run manifest has an unsupported topology")
    early_stop = value.get("early_stop_step")
    if (
        isinstance(early_stop, bool)
        or not isinstance(early_stop, int)
        or early_stop < steps[-1]
        or early_stop not in {200, 2_000, 30_000}
    ):
        raise ValueError("ADR-207 run manifest has an unregistered early-stop boundary")
    for manifest_field, snapshot_field in (
        ("stream_plan_sha256", "stream_plan_sha256"),
        ("representation_split_artifact_sha256", "representation_split_sha256"),
        ("evaluation_plan_artifact_sha256", "evaluation_plan_sha256"),
        ("implementation_sha256", "implementation_sha256"),
        ("model_family_sha256", "model_family_sha256"),
    ):
        if value.get(manifest_field) != snapshots[0].get(snapshot_field):
            raise ValueError(f"ADR-207 run manifest differs for {manifest_field}")
    base_family_sha256 = _validate_lingbot_base_family(
        value.get("lingbot_base_family"),
        name="ADR-207 LingBot base family",
    )
    if value.get("lingbot_base_family_sha256") != base_family_sha256 or any(
        snapshot.get("lingbot_base_family_sha256") != base_family_sha256
        for snapshot in snapshots
    ):
        raise ValueError("ADR-207 run and snapshots use different LingBot base families")
    action_evaluation = _mapping(
        value.get("action_evaluation"),
        name="ADR-207 action evaluation manifest",
    )
    registered_steps = action_evaluation.get("registered_steps")
    if (
        not isinstance(registered_steps, list)
        or tuple(registered_steps[: len(steps)]) != steps
        or action_evaluation.get("state_mode") != "cold_reset"
    ):
        raise ValueError("ADR-207 run manifest changed its registered action curve")

    dense = _mapping(value.get("dense_evidence"), name="ADR-207 dense-evidence manifest")
    if (
        dense.get("mode") != "calvin_full_v1"
        or set(dense.get("modalities", ())) != {"anytouch", "sonata", "vjepa"}
        or int(dense.get("record_count", 0)) <= 0
        or dense.get("semantic_owner") != "shared_lingbot_host_and_posterior_rows"
    ):
        raise ValueError("ADR-207 run manifest does not install all dense modalities")
    auxiliary = _mapping(
        value.get("auxiliary_caches_enabled"),
        name="ADR-207 auxiliary-cache manifest",
    )
    if auxiliary != {
        "future": False,
        "current_filter_target": False,
        "dense_observation": True,
        "videomt_stage_pq": True,
    }:
        raise ValueError("ADR-207 run manifest enabled a retired or omitted cache path")
    physical = _mapping(
        value.get("physical_stream_semantics"),
        name="ADR-207 physical-stream manifest",
    )
    eligibility = _mapping(
        physical.get("native_videomt_source_eligibility"),
        name="ADR-207 source eligibility receipt",
    )
    if (
        physical.get("active") is not True
        or physical.get("maximum_control_tokens_per_prior_pass") != 64
        or physical.get("prior_gradient_control_tokens") != 8
        or eligibility.get("required_future_source_frames") != 4
    ):
        raise ValueError("ADR-207 run manifest changed the causal source/host boundary")
    trainable_scope = _mapping(
        value.get("trainable_scope"),
        name="ADR-207 LingBot trainable scope",
    )
    if (
        trainable_scope.get("scope") != "full-host"
        or trainable_scope.get("forward_model_complete") is not True
        or trainable_scope.get("visual_forward_enabled") is not True
        or trainable_scope.get("trainable_visual_numel") != trainable_scope.get("visual_numel")
    ):
        raise ValueError("ADR-207 run manifest does not train the complete LingBot host")
    if _mapping(value.get("lingbot_compile"), name="ADR-207 compile receipt") != {
        "mode": "upstream-default",
        "enabled": True,
        "ordering": "fsdp2_then_whole_model_compile_then_optimizer",
        "backend": "torch_compile_upstream_default",
    }:
        raise ValueError("ADR-207 run manifest does not use the released compile path")

    source = _mapping(value.get("videomt_stage_pq"), name="ADR-207 source manifest")
    runtime = _mapping(source.get("runtime"), name="ADR-207 source runtime receipt")
    if (
        runtime.get("mode") != ADR207_SOURCE_MODE
        or runtime.get("active") is not True
        or runtime.get("parameter_numel") != 315_986_985
        or runtime.get("training") is not True
        or runtime.get("requires_grad") is not True
        or runtime.get("optimizer_membership") is not True
        or runtime.get("released_training_only_auxiliary_outputs_active") is not True
        or not isinstance(runtime.get("fsdp2"), Mapping)
        or not isinstance(runtime.get("optimizer"), Mapping)
    ):
        raise ValueError("ADR-207 run manifest does not train the complete VidEoMT source")

    execution = _mapping(value.get("execution_contract"), name="ADR-207 execution contract")
    expected_execution = {
        "world_size": world_size,
        "global_batch_size": world_size,
        "gradient_accumulation_steps": 1,
        "trainable_scope": "full-host",
        "physical_event_stream": True,
        "seed": 20260721,
        "capacity": 200,
        "posterior_architecture": "two_pass_v3",
        "picf_architecture_profile": ADR207_PROFILE,
        "task_query_count": 0,
        "relation_supervision_layers": [],
        "objective_profile": "adr207_complete_source_native_query_joint_action",
        "object_action_information_contract": (
            "native_source_query_i_as_shared_host_posterior_row_i_to_official_action"
        ),
        "maximum_control_tokens": 64,
        "prior_gradient_control_tokens": 8,
        "maximum_optimizer_lag": 8,
        "learning_rate": 1e-4.hex(),
        "picf_learning_rate_multiplier": 1.0.hex(),
        "modality_bridge_learning_rate_multiplier": 1.0.hex(),
        "max_grad_norm": 1.0.hex(),
        "entity_weight": 0.0.hex(),
        "predictive_weight": 0.0.hex(),
        "local_bptt_probability": 0.0.hex(),
        "overshoot_probability": 0.0.hex(),
        "source_mask_probability": 0.0.hex(),
        "dense_evidence_mode": "calvin_full_v1",
        "dense_token_bridge": "exact_tokens_v1",
        "fsdp2_placement": "selective-embedding-offload",
        "fsdp2_backward_prefetch": "disabled",
        "sequential_factual_gradient_storage": "gpu",
        "omitted_static_rematerialization": "none",
        "cuda_allocator": "expandable-segments",
        "lingbot_compile_mode": "upstream-default",
    }
    for field, expected_value in expected_execution.items():
        if execution.get(field) != expected_value:
            raise ValueError(f"ADR-207 execution contract differs for {field}")
    if execution.get("native_query_modality_intervention_steps") != [200, 2_000]:
        raise ValueError("ADR-207 execution contract changed its modality intervention steps")
    if set(execution.get("native_query_modality_interventions", ())) != {
        "value_zero",
        "metadata_zero",
        "value_permutation",
        "joint_permutation",
    }:
        raise ValueError("ADR-207 execution contract changed its modality interventions")
    native_relations = execution.get("native_relation_surfaces")
    if not isinstance(native_relations, list) or {
        row.get("name") for row in native_relations if isinstance(row, Mapping)
    } != {"anytouch", "sonata", "vjepa"}:
        raise ValueError("ADR-207 execution contract changed its native relation surfaces")
    source_contract = _mapping(
        execution.get("videomt_stage_pq"),
        name="ADR-207 source execution contract",
    )
    expected_source_contract = {
        "mode": ADR207_SOURCE_MODE,
        "active": True,
        "donor_execution_mode": "complete_calvin_adapted_train_graph_fsdp2_joint",
        "temporal_adapter": "five_real_raw_episode_frames_t_through_t_plus_4_no_padding",
        "short_prefix_policy": "stream_domain_requires_four_future_frames",
        "query_count": 200,
        "query_width": 1024,
        "local_object_selector_decoder_or_lifecycle_head": False,
        "posterior_query_integration": "same_index_source_to_host_width_projection",
        "checkpoint_sha256": ADR207_ADAPTED_SOURCE_SHA256,
    }
    for field, expected_value in expected_source_contract.items():
        if source_contract.get(field) != expected_value:
            raise ValueError(f"ADR-207 source execution contract differs for {field}")
    execution_sha256 = _canonical_sha256(execution)
    if value.get("execution_contract_sha256") != execution_sha256:
        raise ValueError("ADR-207 run manifest execution-contract SHA-256 differs")
    if any(
        snapshot.get("execution_contract_sha256") != execution_sha256
        for snapshot in snapshots
    ):
        raise ValueError("ADR-207 action curve belongs to another execution contract")
    return _canonical_sha256(value), world_size, base_family_sha256


def _validate_lbot_run_report(
    report: object,
    *,
    run_dir: Path,
    steps: tuple[int, ...],
    snapshots: Sequence[dict[str, Any]],
    expected_world_size: int,
    expected_base_family_sha256: str,
) -> None:
    value = _mapping(report, name="matched LingBot run report")
    expected = {
        "schema": LBOT_RUN_REPORT_SCHEMA,
        "status": "PASS",
        "architecture_identity": LBOT_ARCHITECTURE,
        "picf_graph_installed": False,
        "physical_sidecar_read": False,
        "task_scorer_present": False,
        "action_suffix_executed": True,
        "posterior_present": False,
        "physical_event_stream": True,
        "minimum_future_source_frames": 4,
        "maximum_control_tokens": 64,
        "checkpoint_published": False,
        "curve_mode": True,
        "registered_evaluation_steps": list(steps),
        "world_size": expected_world_size,
        "steps": steps[-1],
        "seed": 20260721,
        "max_grad_norm": 1.0,
        "fsdp2_placement": "selective-embedding-offload",
        "cuda_allocator": "expandable-segments",
        "lingbot_compile": {
            "mode": "upstream-default",
            "enabled": True,
            "ordering": "fsdp2_then_whole_model_compile_then_optimizer",
            "backend": "torch_compile_upstream_default",
        },
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(f"matched LingBot run report differs for {field}")
    base_family_sha256 = _validate_lingbot_base_family(
        value.get("lingbot_base_family"),
        name="matched LingBot base family",
    )
    if (
        base_family_sha256 != expected_base_family_sha256
        or value.get("lingbot_base_family_sha256") != base_family_sha256
        or any(
            snapshot.get("lingbot_base_family_sha256") != base_family_sha256
            for snapshot in snapshots
        )
    ):
        raise ValueError("candidate and matched LingBot base families differ")
    if value.get("plan_sha256") != snapshots[0].get("stream_plan_sha256"):
        raise ValueError("matched LingBot run report binds another stream plan")
    for field in ("representation_split_sha256", "evaluation_plan_sha256"):
        if value.get(field) != snapshots[0].get(field):
            raise ValueError(f"matched LingBot run report differs for {field}")
    optimizer = _mapping(value.get("optimizer_contract"), name="LingBot optimizer contract")
    learning_rate = float(optimizer.get("learning_rate", math.nan))
    if not math.isfinite(learning_rate) or learning_rate != 1e-4:
        raise ValueError("matched LingBot optimizer learning rate is not 1e-4")
    rank_reports = value.get("rank_reports")
    if not isinstance(rank_reports, list) or {
        report.get("rank") for report in rank_reports if isinstance(report, Mapping)
    } != set(range(expected_world_size)):
        raise ValueError("matched LingBot run report has another rank topology")
    receipts = value.get("evaluation_snapshots")
    if not isinstance(receipts, list) or len(receipts) != len(steps):
        raise ValueError("matched LingBot run report has another evaluation curve")
    for step, snapshot, raw_receipt in zip(steps, snapshots, receipts, strict=True):
        receipt = _mapping(raw_receipt, name=f"LingBot step-{step} receipt")
        snapshot_path = _snapshot_path(run_dir, treatment="lbot", step=step)
        if snapshot_path.is_symlink() or not snapshot_path.is_file():
            raise ValueError(f"required direct LingBot snapshot is absent: {snapshot_path}")
        if receipt.get("checkpoint_global_step") != step:
            raise ValueError("matched LingBot snapshot receipt step differs")
        embedded_path = receipt.get("path")
        if (
            not isinstance(embedded_path, str)
            or Path(embedded_path).resolve() != snapshot_path.resolve()
        ):
            raise ValueError("matched LingBot snapshot receipt points to another artifact")
        if receipt.get("artifact_sha256") != snapshot.get("artifact_sha256"):
            raise ValueError("matched LingBot snapshot receipt semantic identity differs")
        if receipt.get("file_sha256") != _file_sha256(snapshot_path):
            raise ValueError("matched LingBot snapshot receipt file identity differs")
        if receipt.get("evaluation_input_sha256") != snapshot.get("evaluation_input_sha256"):
            raise ValueError("matched LingBot snapshot receipt input identity differs")


def _validate_picf_joint_checkpoint(
    *,
    run_dir: Path,
    step: int,
    snapshots: Sequence[dict[str, Any]],
    expected_world_size: int,
) -> str | None:
    if step < 2_000:
        return None
    checkpoint_dir = run_dir / "checkpoints" / f"global_step_{step}"
    incomplete = checkpoint_dir.parent / f".global_step_{step}.incomplete"
    if incomplete.exists() or incomplete.is_symlink():
        raise ValueError("ADR-207 checkpoint has an incomplete sibling")
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"required direct ADR-207 checkpoint is absent: {checkpoint_dir}")
    report_path = checkpoint_dir / "task_independent_checkpoint.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError("ADR-207 checkpoint report is absent")
    payload_files = [
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file() and path != report_path
    ]
    if not payload_files or any(path.is_symlink() for path in payload_files):
        raise ValueError("ADR-207 checkpoint has no direct DCP payload")
    report = _mapping(
        json.loads(report_path.read_text(encoding="ascii")),
        name="ADR-207 checkpoint report",
    )
    expected = {
        "schema": PICF_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": step,
        "implementation_sha256": snapshots[-1].get("implementation_sha256"),
        "model_family_sha256": snapshots[-1].get("model_family_sha256"),
        "stream_plan_sha256": snapshots[-1].get("stream_plan_sha256"),
        "execution_contract_sha256": snapshots[-1].get("execution_contract_sha256"),
        "joint_source_active": True,
    }
    for field, expected_value in expected.items():
        if report.get(field) != expected_value:
            raise ValueError(f"ADR-207 checkpoint differs for {field}")
    boundaries = report.get("rank_boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != expected_world_size:
        raise ValueError("ADR-207 checkpoint rank-boundary count differs")
    required_boundary_fields = {
        "model_local_state_sha256",
        "optimizer_local_state_sha256",
        "lane_snapshot_sha256",
        "rank_rng_state_sha256",
        "source_model_local_state_sha256",
        "source_optimizer_local_state_sha256",
    }
    observed_ranks: set[int] = set()
    for row in boundaries:
        item = _mapping(row, name="ADR-207 rank checkpoint boundary")
        rank = item.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("ADR-207 checkpoint rank is invalid")
        observed_ranks.add(rank)
        boundary = _mapping(item.get("boundary"), name="ADR-207 checkpoint boundary")
        if not required_boundary_fields.issubset(boundary):
            raise ValueError("ADR-207 checkpoint boundary omits joint source state")
        for field in required_boundary_fields:
            _sha256_string(boundary[field], name=f"ADR-207 checkpoint {field}")
    if observed_ranks != set(range(expected_world_size)):
        raise ValueError("ADR-207 checkpoint rank set differs")
    return _file_sha256(report_path)


def compare_adr207_curves(
    *,
    picf_snapshots: list[dict[str, Any]],
    lbot_snapshots: list[dict[str, Any]],
    lbot_run_report: object,
    picf_run_dir: Path,
    lbot_run_dir: Path,
    steps: tuple[int, ...],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if steps not in REGISTERED_CURVES:
        raise ValueError("ADR-207 comparison requires the registered 200- or 2000-step curve")
    if len(picf_snapshots) != len(steps) or len(lbot_snapshots) != len(steps):
        raise ValueError("ADR-207 snapshots differ from the registered curve")
    source_spatial_curve = {
        str(step): _validate_adr207_snapshot(snapshot, run_dir=picf_run_dir, step=step)
        for step, snapshot in zip(steps, picf_snapshots, strict=True)
    }
    (
        picf_run_manifest_sha256,
        world_size,
        lingbot_base_family_sha256,
    ) = _validate_picf_run_manifest(
        run_dir=picf_run_dir,
        steps=steps,
        snapshots=picf_snapshots,
    )
    picf_checkpoint_report_sha256 = _validate_picf_joint_checkpoint(
        run_dir=picf_run_dir,
        step=steps[-1],
        snapshots=picf_snapshots,
        expected_world_size=world_size,
    )
    for snapshot in lbot_snapshots:
        _validate_lbot_snapshot(snapshot)
    _validate_lbot_run_report(
        lbot_run_report,
        run_dir=lbot_run_dir,
        steps=steps,
        snapshots=lbot_snapshots,
        expected_world_size=world_size,
        expected_base_family_sha256=lingbot_base_family_sha256,
    )
    picf_model_family_identities = {
        _sha256_string(
            snapshot.get("model_family_sha256"),
            name="ADR-207 treatment model family",
        )
        for snapshot in picf_snapshots
    }
    lbot_model_family_identities = {
        _sha256_string(
            snapshot.get("model_family_sha256"),
            name="matched LingBot treatment model family",
        )
        for snapshot in lbot_snapshots
    }
    if len(picf_model_family_identities) != 1 or len(lbot_model_family_identities) != 1:
        raise ValueError("a treatment changed model family within its action curve")

    mature = compare_curves(
        picf_snapshots=picf_snapshots,
        lbot_snapshots=lbot_snapshots,
        steps=steps,
        bootstrap_replicates=bootstrap_replicates,
    )
    mature.pop("artifact_sha256")
    mature["schema"] = ADR207_REPORT_SCHEMA
    mature["validated_contract"] = {
        "candidate_architecture": ADR207_ARCHITECTURE,
        "candidate_source_query_count": 200,
        "candidate_full_modal_interventions": sorted(INTERVENTION_STEPS.intersection(steps)),
        "lbot_architecture": LBOT_ARCHITECTURE,
        "minimum_future_source_frames": 4,
        "world_size": world_size,
        "lingbot_base_family_sha256": lingbot_base_family_sha256,
        "candidate_model_family_sha256": next(iter(picf_model_family_identities)),
        "lbot_model_family_sha256": next(iter(lbot_model_family_identities)),
        "picf_run_manifest_sha256": picf_run_manifest_sha256,
        "picf_joint_checkpoint_report_sha256": picf_checkpoint_report_sha256,
        "sample_pairing_includes_model_inputs_sha256": True,
        "source_spatial_curve": source_spatial_curve,
        "source_spatial_thresholds": {
            "mean_binary_iou_minimum": SOURCE_HARD_IOU_MINIMUM,
            "recall_at_50_minimum": SOURCE_RECALL_AT_50_MINIMUM,
        },
    }
    return {**mature, "artifact_sha256": _canonical_sha256(mature)}


def main() -> None:
    args = _parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if args.steps not in REGISTERED_CURVES:
        raise ValueError("ADR-207 comparison requires the registered 200- or 2000-step curve")
    picf_snapshots = [
        _load_snapshot(
            _snapshot_path(args.picf_run_dir, treatment="picf", step=step),
            expected_schema=ADR207_ACTION_SCHEMA,
            expected_step=step,
        )
        for step in args.steps
    ]
    lbot_snapshots = [
        _load_snapshot(
            _snapshot_path(args.lbot_run_dir, treatment="lbot", step=step),
            expected_schema=LBOT_SCHEMA,
            expected_step=step,
        )
        for step in args.steps
    ]
    lbot_report_path = args.lbot_run_dir / f"official_lbot_steps_{args.steps[-1]}.json"
    lbot_run_report = json.loads(lbot_report_path.read_text(encoding="ascii"))
    report = compare_adr207_curves(
        picf_snapshots=picf_snapshots,
        lbot_snapshots=lbot_snapshots,
        lbot_run_report=lbot_run_report,
        picf_run_dir=args.picf_run_dir,
        lbot_run_dir=args.lbot_run_dir,
        steps=args.steps,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
