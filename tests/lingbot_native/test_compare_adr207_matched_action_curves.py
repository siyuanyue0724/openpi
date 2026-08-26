from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.compare_adr176_matched_action_curves import LBOT_SCHEMA, _canonical_sha256
from tools.compare_adr207_matched_action_curves import (
    ADR207_ACTION_SCHEMA,
    ADR207_ADAPTED_SOURCE_SHA256,
    ADR207_ANCHOR_SCHEMA,
    ADR207_ARCHITECTURE,
    ADR207_INTERVENTION_SCHEMA,
    ADR207_PROFILE,
    ADR207_REPORT_SCHEMA,
    ADR207_SOURCE_MODE,
    LBOT_ARCHITECTURE,
    LBOT_RUN_REPORT_SCHEMA,
    PICF_CHECKPOINT_SCHEMA,
    _validate_picf_joint_checkpoint,
    compare_adr207_curves,
)

STEPS = (0, 20, 100, 200)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_artifact(
    path: Path,
    payload: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    artifact = {**payload, "artifact_sha256": _canonical_sha256(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="ascii")
    receipt = {
        "status": "PASS",
        "path": str(path),
        "artifact_sha256": artifact["artifact_sha256"],
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return artifact, receipt


def _sample(
    *,
    ordinal: int,
    partition: str,
    action_loss: float,
    adr207: bool,
    world_size: int,
) -> dict[str, object]:
    sample = {
        "ordinal": ordinal,
        "partition": partition,
        "rank": ordinal % world_size,
        "sample_key": f"sample-{partition}-{ordinal}",
        "segment_index": ordinal,
        "source_digest": _sha(f"source-{ordinal}"),
        "source_episode_index": ordinal,
        "source_global_index": ordinal,
        "task_key": f"task-{ordinal}",
        "transition_index": ordinal,
        "model_inputs_sha256": _sha(f"model-input-{ordinal}"),
        "action_loss": action_loss,
    }
    if adr207:
        sample.update(
            {
                "native_source_rgb_sha256": _sha(f"rgb-{ordinal}"),
                "native_source_query_count": 200,
                "prior_trace_finite": True,
                "posterior_finite": True,
            }
        )
    return sample


def _evaluation_sha(samples: list[dict[str, object]], *, adr207: bool) -> str:
    fields = ["sample_key", "source_digest", "model_inputs_sha256"]
    if adr207:
        fields.append("native_source_rgb_sha256")
    return _canonical_sha256(
        [{field: sample[field] for field in fields} for sample in samples]
    )


def _base_family() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "picf-next.lingbot-base-family.v1",
        "architecture": LBOT_ARCHITECTURE,
        "source_commit": "1" * 40,
        "native_patch_sha256": _sha("native-patch"),
        "checkpoint_revision": "2" * 40,
        "checkpoint_assets": [{"path": "model.safetensors", "sha256": _sha("model")}],
        "processor_revision": "3" * 40,
        "processor_assets": [{"path": "tokenizer.json", "sha256": _sha("processor")}],
        "attention_implementation": "flex_cached",
        "trainable_scope": "full-host",
        "optimizer_contract": {"learning_rate": 1e-4},
        "maximum_control_tokens": 64,
    }
    return {**payload, "artifact_sha256": _canonical_sha256(payload)}


def _execution_contract(*, world_size: int = 2) -> dict[str, object]:
    return {
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
        "native_query_modality_intervention_steps": [200, 2_000],
        "native_query_modality_interventions": [
            "value_zero",
            "metadata_zero",
            "value_permutation",
            "joint_permutation",
        ],
        "native_relation_surfaces": [
            {"name": name} for name in ("anytouch", "sonata", "vjepa")
        ],
        "videomt_stage_pq": {
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
        },
    }


def _build_curve(root: Path, *, world_size: int = 2) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    Path,
    Path,
]:
    picf_run = root / "picf"
    lbot_run = root / "lbot"
    picf_snapshots: list[dict[str, object]] = []
    lbot_snapshots: list[dict[str, object]] = []
    lbot_receipts: list[dict[str, object]] = []
    contracts = {
        "stream_plan_sha256": _sha("stream"),
        "representation_split_sha256": _sha("split"),
        "evaluation_plan_sha256": _sha("evaluation"),
    }
    base_family = _base_family()
    base_family_sha256 = str(base_family["artifact_sha256"])
    picf_model_family_sha256 = _sha("picf-model-family")
    lbot_model_family_sha256 = _sha("lbot-model-family")
    execution_contract = _execution_contract(world_size=world_size)
    execution_contract_sha256 = _canonical_sha256(execution_contract)
    for step in STEPS:
        lbot_samples = [
            _sample(
                ordinal=0,
                partition="validation",
                action_loss=0.50 - step / 2_000,
                adr207=False,
                world_size=world_size,
            ),
            _sample(
                ordinal=1,
                partition="validation",
                action_loss=0.55 - step / 2_000,
                adr207=False,
                world_size=world_size,
            ),
            _sample(
                ordinal=2,
                partition="heldout",
                action_loss=0.60 - step / 2_000,
                adr207=False,
                world_size=world_size,
            ),
            _sample(
                ordinal=3,
                partition="heldout",
                action_loss=0.65 - step / 2_000,
                adr207=False,
                world_size=world_size,
            ),
        ]
        picf_samples = [
            _sample(
                ordinal=index,
                partition=str(sample["partition"]),
                action_loss=float(sample["action_loss"]) - step / 2_000,
                adr207=True,
                world_size=world_size,
            )
            for index, sample in enumerate(lbot_samples)
        ]
        anchor_payload = {
            "schema": ADR207_ANCHOR_SCHEMA,
            "status": "PASS",
            "checkpoint_global_step": step,
            "architecture_identity": ADR207_ARCHITECTURE,
            "source_query_count": 200,
            "model_family_sha256": picf_model_family_sha256,
            "lingbot_base_family_sha256": base_family_sha256,
            **contracts,
            "evaluation_input_sha256": _evaluation_sha(picf_samples, adr207=True),
            "samples": [{"sample_key": sample["sample_key"]} for sample in picf_samples],
            "partition_summaries": {
                partition: {
                    "mean_soft_iou": 0.78,
                    "mean_binary_iou": 0.81,
                    "recall_at_50": 0.92,
                }
                for partition in ("validation", "heldout")
            },
        }
        _, anchor_receipt = _write_artifact(
            picf_run
            / "heldout_native_videomt_anchor_evaluations"
            / f"step_{step:08d}"
            / "distributed.json",
            anchor_payload,
        )
        intervention_receipt = None
        if step == 200:
            intervention_payload = {
                "schema": ADR207_INTERVENTION_SCHEMA,
                "status": "PASS",
                "checkpoint_global_step": step,
                "architecture_identity": ADR207_ARCHITECTURE,
                "source_query_count": 200,
                "model_family_sha256": picf_model_family_sha256,
                "lingbot_base_family_sha256": base_family_sha256,
                "modalities": {name: {} for name in ("anytouch", "sonata", "vjepa")},
            }
            _, intervention_receipt = _write_artifact(
                picf_run
                / "native_videomt_modality_interventions"
                / f"step_{step:08d}"
                / "distributed.json",
                intervention_payload,
            )
        picf_payload = {
            "schema": ADR207_ACTION_SCHEMA,
            "status": "PASS",
            "checkpoint_global_step": step,
            "architecture_identity": ADR207_ARCHITECTURE,
            "picf_graph_installed": True,
            "physical_sidecar_read_during_model_forward": False,
            "physical_sidecar_read_after_model_forward_for_metrics": True,
            "task_scorer_present": False,
            "action_suffix_executed": True,
            "state_mode": "cold_reset",
            "implementation_sha256": _sha("picf-implementation"),
            "model_family_sha256": picf_model_family_sha256,
            "lingbot_base_family_sha256": base_family_sha256,
            "execution_contract_sha256": execution_contract_sha256,
            **contracts,
            "evaluation_input_sha256": _evaluation_sha(picf_samples, adr207=True),
            "full_modal_action_intervention": intervention_receipt,
            "heldout_anchor_evaluation": anchor_receipt,
            "samples": picf_samples,
        }
        picf_path = picf_run / "action_evaluations" / f"step_{step:08d}" / "distributed.json"
        picf_snapshot, _ = _write_artifact(picf_path, picf_payload)
        picf_snapshots.append(picf_snapshot)

        lbot_payload = {
            "schema": LBOT_SCHEMA,
            "status": "PASS",
            "checkpoint_global_step": step,
            "architecture_identity": LBOT_ARCHITECTURE,
            "picf_graph_installed": False,
            "physical_sidecar_read": False,
            "task_scorer_present": False,
            "action_suffix_executed": True,
            "posterior_present": False,
            "implementation_sha256": _sha("lbot-implementation"),
            "model_family_sha256": lbot_model_family_sha256,
            "lingbot_base_family_sha256": base_family_sha256,
            **contracts,
            "evaluation_input_sha256": _evaluation_sha(lbot_samples, adr207=False),
            "samples": lbot_samples,
        }
        lbot_path = lbot_run / f"action_evaluation_step_{step:06d}.json"
        lbot_snapshot, lbot_receipt = _write_artifact(lbot_path, lbot_payload)
        lbot_receipt.update(
            {
                "checkpoint_global_step": step,
                "evaluation_input_sha256": lbot_snapshot["evaluation_input_sha256"],
            }
        )
        lbot_snapshots.append(lbot_snapshot)
        lbot_receipts.append(lbot_receipt)

    picf_manifest = {
        "schema": "picf-next.task-independent-full-runner/v18",
        "status": "DECLARED",
        "declared_total_steps": 30_000,
        "early_stop_step": 200,
        "metrics_every": 100,
        "visual_every": 250,
        "checkpoint_every": 2_000,
        "world_size": world_size,
        "global_batch_size": world_size,
        "gradient_accumulation_steps": 1,
        "stream_plan_sha256": contracts["stream_plan_sha256"],
        "representation_split_artifact_sha256": contracts["representation_split_sha256"],
        "evaluation_plan_artifact_sha256": contracts["evaluation_plan_sha256"],
        "implementation_sha256": _sha("picf-implementation"),
        "model_family_sha256": picf_model_family_sha256,
        "lingbot_base_family": base_family,
        "lingbot_base_family_sha256": base_family_sha256,
        "action_evaluation": {
            "registered_steps": list(STEPS),
            "state_mode": "cold_reset",
        },
        "dense_evidence": {
            "mode": "calvin_full_v1",
            "modalities": ["anytouch", "sonata", "vjepa"],
            "record_count": 8,
            "semantic_owner": "shared_lingbot_host_and_posterior_rows",
        },
        "auxiliary_caches_enabled": {
            "future": False,
            "current_filter_target": False,
            "dense_observation": True,
            "videomt_stage_pq": True,
        },
        "physical_stream_semantics": {
            "active": True,
            "maximum_control_tokens_per_prior_pass": 64,
            "prior_gradient_control_tokens": 8,
            "native_videomt_source_eligibility": {
                "required_future_source_frames": 4,
            },
        },
        "trainable_scope": {
            "scope": "full-host",
            "forward_model_complete": True,
            "visual_forward_enabled": True,
            "visual_numel": 10,
            "trainable_visual_numel": 10,
        },
        "lingbot_compile": {
            "mode": "upstream-default",
            "enabled": True,
            "ordering": "fsdp2_then_whole_model_compile_then_optimizer",
            "backend": "torch_compile_upstream_default",
        },
        "videomt_stage_pq": {
            "runtime": {
                "mode": ADR207_SOURCE_MODE,
                "active": True,
                "parameter_numel": 315_986_985,
                "training": True,
                "requires_grad": True,
                "optimizer_membership": True,
                "released_training_only_auxiliary_outputs_active": True,
                "fsdp2": {},
                "optimizer": {},
            }
        },
        "execution_contract": execution_contract,
        "execution_contract_sha256": execution_contract_sha256,
    }
    (picf_run / "run_manifest.json").write_text(
        json.dumps(picf_manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    lbot_report = {
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
        "registered_evaluation_steps": list(STEPS),
        "world_size": world_size,
        "steps": 200,
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
        "plan_sha256": contracts["stream_plan_sha256"],
        "representation_split_sha256": contracts["representation_split_sha256"],
        "evaluation_plan_sha256": contracts["evaluation_plan_sha256"],
        "optimizer_contract": {"learning_rate": 1e-4},
        "model_family_sha256": lbot_model_family_sha256,
        "lingbot_base_family": base_family,
        "lingbot_base_family_sha256": base_family_sha256,
        "rank_reports": [{"rank": rank} for rank in range(world_size)],
        "evaluation_snapshots": lbot_receipts,
    }
    (lbot_run / "official_lbot_steps_200.json").write_text(
        json.dumps(lbot_report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return picf_snapshots, lbot_snapshots, lbot_report, picf_run, lbot_run


def _compare(root: Path, *, mutate_report=None) -> dict[str, object]:
    picf, lbot, report, picf_run, lbot_run = _build_curve(root)
    if mutate_report is not None:
        mutate_report(report)
    return compare_adr207_curves(
        picf_snapshots=picf,
        lbot_snapshots=lbot,
        lbot_run_report=report,
        picf_run_dir=picf_run,
        lbot_run_dir=lbot_run,
        steps=STEPS,
        bootstrap_replicates=100,
    )


def test_compare_adr207_curves_reuses_mature_curve_statistics(tmp_path: Path) -> None:
    result = _compare(tmp_path)

    assert result["schema"] == ADR207_REPORT_SCHEMA
    assert result["decision"] == "PICF_ACTION_ADVANTAGE"
    assert result["validated_contract"]["candidate_source_query_count"] == 200
    assert result["validated_contract"]["candidate_full_modal_interventions"] == [200]
    assert result["validated_contract"]["minimum_future_source_frames"] == 4
    assert result["validated_contract"]["world_size"] == 2
    assert (
        result["validated_contract"]["candidate_model_family_sha256"]
        != result["validated_contract"]["lbot_model_family_sha256"]
    )
    assert result["validated_contract"]["source_spatial_curve"]["200"]["heldout"][
        "mean_binary_iou"
    ] == pytest.approx(0.81)
    semantic = copy.deepcopy(result)
    artifact_sha256 = semantic.pop("artifact_sha256")
    assert artifact_sha256 == _canonical_sha256(semantic)


def test_compare_adr207_curves_accepts_four_gpu_matched_topology(tmp_path: Path) -> None:
    picf, lbot, report, picf_run, lbot_run = _build_curve(tmp_path, world_size=4)

    result = compare_adr207_curves(
        picf_snapshots=picf,
        lbot_snapshots=lbot,
        lbot_run_report=report,
        picf_run_dir=picf_run,
        lbot_run_dir=lbot_run,
        steps=STEPS,
        bootstrap_replicates=10,
    )

    assert result["validated_contract"]["world_size"] == 4


def test_compare_adr207_curves_rejects_different_lingbot_base_family(tmp_path: Path) -> None:
    picf, lbot, report, picf_run, lbot_run = _build_curve(tmp_path)
    changed = dict(report["lingbot_base_family"])
    changed.pop("artifact_sha256")
    changed["source_commit"] = "4" * 40
    changed_sha256 = _canonical_sha256(changed)
    report["lingbot_base_family"] = {**changed, "artifact_sha256": changed_sha256}
    report["lingbot_base_family_sha256"] = changed_sha256

    with pytest.raises(ValueError, match="base families differ"):
        compare_adr207_curves(
            picf_snapshots=picf,
            lbot_snapshots=lbot,
            lbot_run_report=report,
            picf_run_dir=picf_run,
            lbot_run_dir=lbot_run,
            steps=STEPS,
            bootstrap_replicates=10,
        )


def test_compare_adr207_curves_rejects_non_future_filtered_lbot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="minimum_future_source_frames"):
        _compare(
            tmp_path,
            mutate_report=lambda report: report.update({"minimum_future_source_frames": 0}),
        )


def test_compare_adr207_curves_rejects_candidate_hyperparameter_drift(tmp_path: Path) -> None:
    picf, lbot, report, picf_run, lbot_run = _build_curve(tmp_path)
    manifest_path = picf_run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["execution_contract"]["learning_rate"] = 2e-4.hex()
    manifest["execution_contract_sha256"] = _canonical_sha256(
        manifest["execution_contract"]
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="learning_rate"):
        compare_adr207_curves(
            picf_snapshots=picf,
            lbot_snapshots=lbot,
            lbot_run_report=report,
            picf_run_dir=picf_run,
            lbot_run_dir=lbot_run,
            steps=STEPS,
            bootstrap_replicates=10,
        )


def test_compare_adr207_curves_rejects_source_spatial_regression(tmp_path: Path) -> None:
    picf, lbot, report, picf_run, lbot_run = _build_curve(tmp_path)
    anchor_path = (
        picf_run
        / "heldout_native_videomt_anchor_evaluations"
        / "step_00000200"
        / "distributed.json"
    )
    anchor = json.loads(anchor_path.read_text(encoding="ascii"))
    anchor.pop("artifact_sha256")
    anchor["partition_summaries"]["heldout"]["mean_binary_iou"] = 0.69
    _, receipt = _write_artifact(anchor_path, anchor)
    picf[-1]["heldout_anchor_evaluation"].update(receipt)

    with pytest.raises(ValueError, match="hard-IoU gate failed"):
        compare_adr207_curves(
            picf_snapshots=picf,
            lbot_snapshots=lbot,
            lbot_run_report=report,
            picf_run_dir=picf_run,
            lbot_run_dir=lbot_run,
            steps=STEPS,
            bootstrap_replicates=10,
        )


def test_compare_adr207_curves_rejects_incomplete_modality_gate(tmp_path: Path) -> None:
    picf, lbot, report, picf_run, lbot_run = _build_curve(tmp_path)
    intervention_path = (
        picf_run
        / "native_videomt_modality_interventions"
        / "step_00000200"
        / "distributed.json"
    )
    intervention = json.loads(intervention_path.read_text(encoding="ascii"))
    intervention.pop("artifact_sha256")
    intervention["modalities"].pop("sonata")
    _write_artifact(intervention_path, intervention)
    picf[-1]["full_modal_action_intervention"]["artifact_sha256"] = json.loads(
        intervention_path.read_text(encoding="ascii")
    )["artifact_sha256"]
    picf[-1]["full_modal_action_intervention"]["file_sha256"] = hashlib.sha256(
        intervention_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="every dense modality"):
        compare_adr207_curves(
            picf_snapshots=picf,
            lbot_snapshots=lbot,
            lbot_run_report=report,
            picf_run_dir=picf_run,
            lbot_run_dir=lbot_run,
            steps=STEPS,
            bootstrap_replicates=10,
        )


def test_compare_adr207_curves_rejects_unregistered_partial_curve(tmp_path: Path) -> None:
    picf, lbot, report, picf_run, lbot_run = _build_curve(tmp_path)

    with pytest.raises(ValueError, match="registered 200- or 2000-step curve"):
        compare_adr207_curves(
            picf_snapshots=picf[:3],
            lbot_snapshots=lbot[:3],
            lbot_run_report=report,
            picf_run_dir=picf_run,
            lbot_run_dir=lbot_run,
            steps=STEPS[:3],
            bootstrap_replicates=10,
        )


def test_adr207_step2000_checkpoint_requires_joint_source_boundaries(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "global_step_2000"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"dcp")
    snapshot = {
        "implementation_sha256": _sha("implementation"),
        "model_family_sha256": _sha("model-family"),
        "stream_plan_sha256": _sha("stream"),
        "execution_contract_sha256": _sha("execution"),
    }
    boundary = {
        field: _sha(field)
        for field in (
            "model_local_state_sha256",
            "optimizer_local_state_sha256",
            "lane_snapshot_sha256",
            "rank_rng_state_sha256",
            "source_model_local_state_sha256",
            "source_optimizer_local_state_sha256",
        )
    }
    report = {
        "schema": PICF_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": 2_000,
        "implementation_sha256": snapshot["implementation_sha256"],
        "model_family_sha256": snapshot["model_family_sha256"],
        "stream_plan_sha256": snapshot["stream_plan_sha256"],
        "execution_contract_sha256": snapshot["execution_contract_sha256"],
        "joint_source_active": True,
        "rank_boundaries": [
            {"rank": rank, "boundary": boundary} for rank in range(2)
        ],
    }
    report_path = checkpoint / "task_independent_checkpoint.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    assert _validate_picf_joint_checkpoint(
        run_dir=tmp_path,
        step=2_000,
        snapshots=[snapshot],
        expected_world_size=2,
    ) == hashlib.sha256(report_path.read_bytes()).hexdigest()

    report["rank_boundaries"][0]["boundary"].pop("source_model_local_state_sha256")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="joint source state"):
        _validate_picf_joint_checkpoint(
            run_dir=tmp_path,
            step=2_000,
            snapshots=[snapshot],
            expected_world_size=2,
        )
