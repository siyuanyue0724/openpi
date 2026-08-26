from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import DATASET_RUNTIME_VERIFICATION_MODE
from picf_next.lingbot_native.adr150_lbot_validation import (
    FULL_MODAL_ACTION_ADOPTION_SCHEMA,
    LBOT_REPORT_SCHEMA,
    LEGACY_LBOT_REPORT_SCHEMA,
    VALIDATION_SCHEMA,
    validate_adr150_matched_lbot_report,
)


def _sha(character: str) -> str:
    return character * 64


def _implementation_files() -> dict[str, str]:
    return {
        "src/picf_next/lingbot_native/calvin.py": _sha("1"),
        "tools/run_lingbot_vla2_official_lbot.py": _sha("2"),
    }


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dataset_contract() -> dict[str, Any]:
    return {
        "status": "PASS",
        "manifest_sha256": _sha("3"),
        "normalization_sha256": _sha("4"),
        "validation": {
            "dataset_file_count": 12,
            "dataset_total_size_bytes": 4096,
            "dataset_tree_sha256": _sha("5"),
            "dataset_manifest_self_consistent": True,
            "dataset_full_tree_rescanned": False,
            "dataset_runtime_verified_read_required": True,
            "dataset_runtime_probe_file_count": 2,
            "dataset_runtime_probe_sha256": _sha("6"),
            "dataset_verification_mode": DATASET_RUNTIME_VERIFICATION_MODE,
        },
    }


def _storage() -> dict[str, Any]:
    return {
        "parameter_tensors": 10,
        "local_elements": 1000,
        "master_dtype": "float32",
        "placement": "selective-embedding-offload",
        "cpu_parameter_tensors": 1,
        "cpu_local_elements": 100,
        "cuda_parameter_tensors": 9,
        "cuda_local_elements": 900,
        "selective_cpu_parameter_names": ["policy.model.shared_embedding.weight"],
    }


def _prune() -> dict[str, Any]:
    return {
        "schema": "picf-next.targetless-alignment-teacher-prune.v1",
        "removed": [
            {
                "name": "current_video_align_head",
                "parameter_count": 2,
                "numel": 32,
                "storage_bytes": 128,
            }
        ],
        "removed_numel": 32,
        "removed_storage_bytes": 128,
        "retained_query_components": ["current_video_align_embs"],
    }


def _optimizer() -> dict[str, Any]:
    return {
        "algorithm": "lingbot_distributed_muon_with_adamw_fallback",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1e-8,
        "muon_momentum": 0.95,
        "muon_nesterov": True,
        "muon_ns_steps": 5,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "muon_exclude_name_patterns": ["embed"],
        "use_moe": True,
        "use_moe_expert_lr": True,
        "token_moe_layers": [1, 2],
        "token_num_experts": 32,
        "token_top_k": 1,
        "bias_update_speed": 0.001,
        "bias_centering": False,
        "bias_update_interval": 1,
        "sequence_wise_loss_coeff": 0.0,
        "sequence_wise_mode": "per_sequence",
        "router_z_loss_coeff": 0.0,
        "router_activation": "softmax",
        "routed_scaling_factor": 1.0,
        "use_shared_expert_gate": True,
        "enable_fp32": True,
        "enable_mixed_precision": True,
        "scheduler": "constant",
        "scheduler_warmup_ratio": 0.0,
        "scheduler_start_lr": 0.0,
    }


def _partition_summaries() -> dict[str, Any]:
    return {
        partition: {
            "sample_count": 4,
            "mean_action_loss": 0.1,
            "mean_total_loss": 0.2,
            "mean_moe_regularizer": 0.01,
            "mean_forward_seconds": 0.5,
        }
        for partition in ("validation", "heldout")
    }


def _step(rank: int, global_step: int) -> dict[str, Any]:
    return {
        "global_step": global_step,
        "sample_keys": [f"rank-{rank}-step-{global_step}"],
        "lane_ids": [rank],
        "frame_indices": [global_step - 1],
        "reset": [global_step == 1],
        "source_digest": hashlib.sha256(f"{rank}:{global_step}".encode()).hexdigest(),
        "augmentation_seeds": [global_step * 10 + rank],
        "flow_noise_seeds": [global_step * 20 + rank],
        "flow_timestep_seeds": [global_step * 30 + rank],
        "total_loss": 0.5,
        "action_loss": 0.4,
        "moe_regularizer": 0.01,
        "official_output_arity": 11,
        "picf_graph_installed": False,
        "gradient_metrics": {
            "all_finite": True,
            "vlm_host_norm": 1.0,
            "vlm_host_elements": 10,
            "action_expert_norm": 2.0,
            "action_expert_elements": 20,
            "action_output_norm": 3.0,
            "action_output_elements": 30,
            "preclip_global_norm": 4.0,
        },
        "step_time_s": 1.0,
        "peak_cuda_allocated_bytes": 100,
        "peak_cuda_reserved_bytes": 200,
    }


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _gradient(norm: float | None, *, elements: int = 10) -> dict[str, Any]:
    return {"norm": norm, "elements": 0 if norm is None else elements}


def _full_modal_action_adoption() -> dict[str, Any]:
    subset_specs = (
        ("none", ()),
        ("A", ("anytouch",)),
        ("S", ("sonata",)),
        ("V", ("vjepa",)),
        ("AS", ("anytouch", "sonata")),
        ("AV", ("anytouch", "vjepa")),
        ("SV", ("sonata", "vjepa")),
        ("ASV", ("anytouch", "sonata", "vjepa")),
    )
    subsets = []
    for subset_name, present in subset_specs:
        adapters = {}
        for modality_index, modality in enumerate(("anytouch", "sonata", "vjepa"), start=1):
            if modality in present:
                adapters[modality] = {
                    "value": _gradient(0.1 * modality_index),
                    "metadata": _gradient(0.2 * modality_index),
                }
            else:
                adapters[modality] = {
                    "value": _gradient(None),
                    "metadata": _gradient(0.0),
                }
        subsets.append(
            {
                "name": subset_name,
                "present_modalities": list(present),
                "sample_keys": ["probe-0", "touch-active-0"],
                "adapter_action_only_gradients": adapters,
                "host_action_only_gradients": {
                    "0": _gradient(0.7),
                    "18": _gradient(0.8),
                    "35": _gradient(0.9),
                },
                "qwen_action_expert_action_only_gradient": _gradient(1.0),
                "action_out_proj_action_only_gradient": _gradient(1.1),
            }
        )

    interventions = []
    for modality in ("anytouch", "sonata", "vjepa"):
        interventions.append(
            {
                "modality": modality,
                "sample_keys": ["probe-1", "touch-active-0"],
                "token_permutations": [[1, 0, 2], [0, 2, 1]],
                "valid_before": [[True, True, False], [False, True, True]],
                "valid_after": [[True, True, False], [False, True, True]],
                "factual_repeat": {
                    "measured_max_abs_action_drift": 1e-8,
                    "maximum_allowed": 1e-6,
                },
                "value_zero": {
                    "measured_max_abs_action_drift": 0.2,
                    "minimum_required": 0.01,
                },
                "metadata_zero": {
                    "measured_max_abs_action_drift": 0.1,
                    "minimum_required": 0.01,
                },
                "value_only_permutation": {
                    "measured_max_abs_action_drift": 0.3,
                    "minimum_required": 0.01,
                },
                "joint_value_metadata_valid_permutation": {
                    "measured_max_abs_action_drift": 1e-8,
                    "maximum_allowed": 1e-6,
                },
            }
        )

    boundary = {
        "model_sha256": _digest("boundary-model"),
        "optimizer_sha256": _digest("boundary-optimizer"),
        "lane_sha256": _digest("boundary-lane"),
        "rng_sha256": _digest("boundary-rng"),
    }
    continuation = {
        "global_step": 2,
        "model_sha256": _digest("continuation-model"),
        "optimizer_sha256": _digest("continuation-optimizer"),
        "lane_sha256": _digest("continuation-lane"),
        "rng_sha256": _digest("continuation-rng"),
        "action_output_sha256": _digest("continuation-action-output"),
        "action_loss": 0.25,
    }
    return {
        "schema": FULL_MODAL_ACTION_ADOPTION_SCHEMA,
        "status": "PASS",
        "action_loss_only": True,
        "nonzero_gradient_min_norm": 1e-9,
        "presence_subsets": subsets,
        "modality_interventions": interventions,
        "active_anytouch_sample_keys": ["touch-active-0"],
        "dcp_cold_restore": {
            "checkpoint_global_step": 1,
            "optimizer_step": 1,
            "save_process_sha256": _digest("save-process"),
            "restore_process_sha256": _digest("restore-process"),
            "checkpoint_artifact_sha256": _digest("dcp-checkpoint"),
            "saved_boundary": boundary,
            "restored_boundary": copy.deepcopy(boundary),
            "uninterrupted_next_step": continuation,
            "restored_next_step": copy.deepcopy(continuation),
        },
    }


def _identities() -> dict[str, Any]:
    implementation_files = _implementation_files()
    implementation_sha256 = _canonical_digest(implementation_files)
    plan_sha256 = _sha("7")
    checkpoint_revision = "8" * 40
    model_family_sha256 = _canonical_digest(
        {
            "architecture": "released_lingbot_vla2_action_policy",
            "checkpoint_revision": checkpoint_revision,
            "implementation_sha256": implementation_sha256,
            "plan_sha256": plan_sha256,
        }
    )
    return {
        "plan_sha256": plan_sha256,
        "split_sha256": _sha("9"),
        "evaluation_sha256": _sha("a"),
        "implementation": {
            "implementation_files": implementation_files,
            "implementation_sha256": implementation_sha256,
        },
        "source": {
            "source_commit": "b" * 40,
            "source_patch_sha256": _sha("c"),
            "patched_source_sha256": {
                "lingbotvla/model.py": _sha("d"),
            },
        },
        "model": {
            "model_family_sha256": model_family_sha256,
            "checkpoint_revision": checkpoint_revision,
            "checkpoint_assets": [
                {"path": "config.json", "bytes": 10, "sha256": _sha("e")},
                {"path": "model.safetensors", "bytes": 20, "sha256": _sha("f")},
            ],
            "qwen_vision_geometry": {"patch_size": 14, "spatial_merge_size": 2},
            "parameter_storage": _storage(),
            "parameter_manifest": {
                "parameter_count": 10,
                "trainable_numel": 4000,
                "schema_sha256": _sha("0"),
            },
            "alignment_teacher_prune": _prune(),
        },
        "processor": {
            "processor_revision": "1" * 40,
            "processor_assets": [{"path": "tokenizer.json", "bytes": 30, "sha256": _sha("2")}],
        },
        "dataset": _dataset_contract(),
        "optimizer": _optimizer(),
    }


def _report() -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _identities()
    evaluation_steps = (0, 20, 100, 200)
    base_family_semantic = {
        "schema": "picf-next.lingbot-base-family.v1",
        "architecture": "released_lingbot_vla2_action_policy",
        "source_commit": identity["source"]["source_commit"],
        "native_patch_sha256": identity["source"]["source_patch_sha256"],
        "checkpoint_revision": identity["model"]["checkpoint_revision"],
        "checkpoint_assets": copy.deepcopy(identity["model"]["checkpoint_assets"]),
        "processor_revision": identity["processor"]["processor_revision"],
        "processor_assets": copy.deepcopy(identity["processor"]["processor_assets"]),
        "attention_implementation": "eager",
        "trainable_scope": "full-host",
        "optimizer_contract": copy.deepcopy(identity["optimizer"]),
        "maximum_control_tokens": 64,
    }
    base_family_sha256 = _canonical_digest(base_family_semantic)
    report = {
        "schema": LBOT_REPORT_SCHEMA,
        "status": "PASS",
        "architecture_identity": "released_lingbot_vla2_action_policy",
        "picf_graph_installed": False,
        "physical_sidecar_read": False,
        "task_scorer_present": False,
        "action_suffix_executed": True,
        "posterior_present": False,
        "physical_event_stream": True,
        "minimum_future_source_frames": 0,
        "maximum_control_tokens": 64,
        "checkpoint_published": False,
        "source_commit": identity["source"]["source_commit"],
        "source_patch_sha256": identity["source"]["source_patch_sha256"],
        "patched_source_sha256": copy.deepcopy(identity["source"]["patched_source_sha256"]),
        "implementation_files": copy.deepcopy(identity["implementation"]["implementation_files"]),
        "implementation_sha256": identity["implementation"]["implementation_sha256"],
        "checkpoint_revision": identity["model"]["checkpoint_revision"],
        "checkpoint_assets": copy.deepcopy(identity["model"]["checkpoint_assets"]),
        "processor_revision": identity["processor"]["processor_revision"],
        "processor_assets": copy.deepcopy(identity["processor"]["processor_assets"]),
        "dataset_contract": copy.deepcopy(identity["dataset"]),
        "plan_sha256": identity["plan_sha256"],
        "curve_mode": True,
        "registered_evaluation_steps": list(evaluation_steps),
        "representation_split_sha256": identity["split_sha256"],
        "evaluation_plan_sha256": identity["evaluation_sha256"],
        "evaluation_snapshots": [
            {
                "artifact_sha256": hashlib.sha256(f"artifact:{step}".encode()).hexdigest(),
                "file_sha256": hashlib.sha256(f"file:{step}".encode()).hexdigest(),
                "path": f"/mnt/lbot/action_evaluation_step_{step:06d}.json",
                "checkpoint_global_step": step,
                "evaluation_input_sha256": _sha("3"),
                "partition_summaries": _partition_summaries(),
            }
            for step in evaluation_steps
        ],
        "model_family_sha256": identity["model"]["model_family_sha256"],
        "lingbot_base_family": {
            **base_family_semantic,
            "artifact_sha256": base_family_sha256,
        },
        "lingbot_base_family_sha256": base_family_sha256,
        "world_size": 4,
        "steps": 200,
        "seed": 20260721,
        "max_grad_norm": 1.0,
        "official_output_arity": 11,
        "optimizer_contract": copy.deepcopy(identity["optimizer"]),
        "qwen_vision_geometry": copy.deepcopy(identity["model"]["qwen_vision_geometry"]),
        "fsdp2_placement": "selective-embedding-offload",
        "cuda_allocator": "native",
        "attention_implementation": "eager",
        "gradient_checkpointing": True,
        "parameter_storage": copy.deepcopy(identity["model"]["parameter_storage"]),
        "parameter_manifest": copy.deepcopy(identity["model"]["parameter_manifest"]),
        "alignment_teacher_prune": copy.deepcopy(identity["model"]["alignment_teacher_prune"]),
        "maximum_peak_reserved_bytes": 39 * 1024**3,
        "rank_reports": [
            {"rank": rank, "steps": [_step(rank, step) for step in range(1, 201)]}
            for rank in range(4)
        ],
        "full_modal_action_adoption": _full_modal_action_adoption(),
    }
    return report, identity


def _validate(
    report: dict[str, Any],
    identity: dict[str, Any],
    *,
    attention_implementation: str = "eager",
):
    return validate_adr150_matched_lbot_report(
        report,
        expected_plan_sha256=identity["plan_sha256"],
        expected_representation_split_sha256=identity["split_sha256"],
        expected_evaluation_plan_sha256=identity["evaluation_sha256"],
        expected_seed=20260721,
        expected_implementation_identity=identity["implementation"],
        expected_source_identity=identity["source"],
        expected_model_identity=identity["model"],
        expected_processor_identity=identity["processor"],
        expected_dataset_identity=identity["dataset"],
        expected_optimizer_contract=identity["optimizer"],
        expected_attention_implementation=attention_implementation,
    )


def test_valid_report_returns_stable_canonical_evidence() -> None:
    report, identity = _report()
    result = _validate(report, identity)
    reordered = dict(reversed(list(report.items())))
    repeated = _validate(reordered, identity)

    assert result.report_sha256 == repeated.report_sha256
    assert result.report_sha256 == _canonical_digest(report)
    assert result.canonical_report == report
    assert result.canonical_report is not report
    assert result.validation_report["schema"] == VALIDATION_SCHEMA
    assert result.validation_report["status"] == "PASS"
    validation_payload = dict(result.validation_report)
    artifact = validation_payload.pop("artifact_sha256")
    assert artifact == _canonical_digest(validation_payload)


def test_legacy_control_schema_is_read_only_compatible() -> None:
    report, identity = _report()
    report["schema"] = LEGACY_LBOT_REPORT_SCHEMA

    result = _validate(report, identity)

    assert result.canonical_report["schema"] == LEGACY_LBOT_REPORT_SCHEMA
    assert result.validation_report["lbot_report_schema"] == LEGACY_LBOT_REPORT_SCHEMA


def test_pre_base_family_report_is_read_only_compatible() -> None:
    report, identity = _report()
    report.pop("lingbot_base_family")
    report.pop("lingbot_base_family_sha256")

    result = _validate(report, identity)

    assert "lingbot_base_family" not in result.canonical_report


def test_validator_v2_schema_is_one_explicit_field_ahead_of_the_v1_producer() -> None:
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "tools/run_lingbot_vla2_official_lbot.py").read_text(encoding="utf-8"))
    candidates: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "report" for target in node.targets
        ):
            continue
        keys = {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "architecture_identity" in keys:
            candidates.append(keys)
    report, _identity = _report()
    assert candidates == [set(report).difference({"full_modal_action_adoption"})]


def test_minimum_future_source_frames_is_exactly_validated() -> None:
    report, identity = _report()
    report["minimum_future_source_frames"] = 4

    with pytest.raises(ContractError, match="minimum_future_source_frames differs"):
        _validate(report, identity)

    result = validate_adr150_matched_lbot_report(
        report,
        expected_plan_sha256=identity["plan_sha256"],
        expected_representation_split_sha256=identity["split_sha256"],
        expected_evaluation_plan_sha256=identity["evaluation_sha256"],
        expected_seed=20260721,
        expected_implementation_identity=identity["implementation"],
        expected_source_identity=identity["source"],
        expected_model_identity=identity["model"],
        expected_processor_identity=identity["processor"],
        expected_dataset_identity=identity["dataset"],
        expected_optimizer_contract=identity["optimizer"],
        expected_minimum_future_source_frames=4,
    )
    assert result.canonical_report["minimum_future_source_frames"] == 4


def test_historical_report_without_future_filter_field_is_read_only_compatible() -> None:
    report, identity = _report()
    del report["minimum_future_source_frames"]

    result = _validate(report, identity)

    assert "minimum_future_source_frames" not in result.canonical_report
    assert result.report_sha256 == _canonical_digest(report)


def test_attention_implementation_is_exact_and_historical_eager_is_compatible() -> None:
    report, identity = _report()
    report["attention_implementation"] = "flex_cached"
    base_family = dict(report["lingbot_base_family"])
    base_family.pop("artifact_sha256")
    base_family["attention_implementation"] = "flex_cached"
    base_family_sha256 = _canonical_digest(base_family)
    report["lingbot_base_family"] = {
        **base_family,
        "artifact_sha256": base_family_sha256,
    }
    report["lingbot_base_family_sha256"] = base_family_sha256

    result = _validate(report, identity, attention_implementation="flex_cached")
    assert result.canonical_report["attention_implementation"] == "flex_cached"

    with pytest.raises(ContractError, match="attention_implementation differs"):
        _validate(report, identity)

    del report["attention_implementation"]
    del report["lingbot_base_family"]
    del report["lingbot_base_family_sha256"]
    historical = _validate(report, identity)
    assert "attention_implementation" not in historical.canonical_report


def test_lingbot_base_family_must_match_top_level_attention() -> None:
    report, identity = _report()
    report["attention_implementation"] = "flex_cached"

    with pytest.raises(ContractError, match="LingBot base family differs"):
        _validate(report, identity, attention_implementation="flex_cached")


@pytest.mark.parametrize("field", ["schema", "rank_reports", "dataset_contract"])
def test_missing_top_level_critical_field_is_rejected(field: str) -> None:
    report, identity = _report()
    del report[field]

    with pytest.raises(ContractError, match="fields differ"):
        _validate(report, identity)


def test_extra_top_level_or_nested_critical_field_is_rejected() -> None:
    report, identity = _report()
    report["claimed_safe"] = True
    with pytest.raises(ContractError, match="extra"):
        _validate(report, identity)

    report, identity = _report()
    report["optimizer_contract"]["unregistered_scheduler_state"] = 0
    with pytest.raises(ContractError, match="optimizer_contract fields differ"):
        _validate(report, identity)

    report, identity = _report()
    report["checkpoint_assets"][0]["untrusted_origin"] = "elsewhere"
    with pytest.raises(ContractError, match="checkpoint assets.*fields differ"):
        _validate(report, identity)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("plan_sha256",), _sha("4")),
        (("representation_split_sha256",), _sha("4")),
        (("evaluation_plan_sha256",), _sha("4")),
        (("seed",), 20260722),
        (("max_grad_norm",), 0.5),
        (("optimizer_contract", "learning_rate"), 2e-4),
        (("optimizer_contract", "weight_decay"), 0.02),
        (("implementation_sha256",), _sha("4")),
        (("source_commit",), "4" * 40),
        (("model_family_sha256",), _sha("4")),
        (("processor_revision",), "4" * 40),
        (("dataset_contract", "manifest_sha256"), _sha("4")),
    ],
)
def test_stale_or_forged_frozen_identity_is_rejected(
    path: tuple[str, ...], replacement: object
) -> None:
    report, identity = _report()
    target: dict[str, Any] = report
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = replacement

    with pytest.raises(ContractError):
        _validate(report, identity)


@pytest.mark.parametrize("replacement", [True, 4.0, "4"])
def test_world_size_requires_exact_integer_type(replacement: object) -> None:
    report, identity = _report()
    report["world_size"] = replacement

    with pytest.raises(ContractError):
        _validate(report, identity)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_nested_metric_is_rejected(value: float) -> None:
    report, identity = _report()
    report["rank_reports"][0]["steps"][0]["total_loss"] = value

    with pytest.raises(ContractError, match="finite"):
        _validate(report, identity)


def test_rank_step_forgery_and_cross_rank_overlap_are_rejected() -> None:
    report, identity = _report()
    report["rank_reports"][0]["steps"][10]["global_step"] = 99
    with pytest.raises(ContractError, match="not contiguous"):
        _validate(report, identity)

    report, identity = _report()
    report["rank_reports"][1]["steps"][0]["sample_keys"] = report["rank_reports"][0]["steps"][0][
        "sample_keys"
    ]
    with pytest.raises(ContractError, match="overlapping"):
        _validate(report, identity)


def test_evaluation_curve_must_match_registered_steps_and_inputs() -> None:
    report, identity = _report()
    report["registered_evaluation_steps"] = [0, 20, 200]
    with pytest.raises(ContractError, match="registered evaluation"):
        _validate(report, identity)

    report, identity = _report()
    report["evaluation_snapshots"][2]["evaluation_input_sha256"] = _sha("4")
    with pytest.raises(ContractError, match="evaluation inputs differ"):
        _validate(report, identity)


def test_expected_identity_must_itself_be_self_consistent() -> None:
    report, identity = _report()
    stale = copy.deepcopy(identity)
    stale["implementation"]["implementation_files"]["new.py"] = _sha("4")

    with pytest.raises(ContractError, match="expected implementation digest"):
        _validate(report, stale)


def test_validation_evidence_addresses_full_modal_action_adoption() -> None:
    report, identity = _report()
    result = _validate(report, identity)

    assert result.validation_report["full_modal_action_adoption_sha256"] == _canonical_digest(
        report["full_modal_action_adoption"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "action_loss_only",
        "presence_subsets",
        "modality_interventions",
        "active_anytouch_sample_keys",
        "dcp_cold_restore",
    ],
)
def test_missing_full_modal_action_adoption_field_is_rejected(field: str) -> None:
    report, identity = _report()
    del report["full_modal_action_adoption"][field]

    with pytest.raises(ContractError, match="fields differ"):
        _validate(report, identity)


def test_extra_full_modal_action_adoption_fields_are_rejected_at_every_level() -> None:
    mutations = (
        lambda adoption: adoption.__setitem__("claimed_complete", True),
        lambda adoption: adoption["presence_subsets"][0].__setitem__("aux_gradient", 1.0),
        lambda adoption: adoption["presence_subsets"][0]["adapter_action_only_gradients"][
            "anytouch"
        ]["value"].__setitem__("finite", True),
        lambda adoption: adoption["modality_interventions"][0]["factual_repeat"].__setitem__(
            "mean_drift", 0.0
        ),
        lambda adoption: adoption["dcp_cold_restore"].__setitem__("warm_restore", False),
    )
    for mutate in mutations:
        report, identity = _report()
        mutate(report["full_modal_action_adoption"])
        with pytest.raises(ContractError, match="fields differ"):
            _validate(report, identity)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema", "picf-next.adr150-full-modal-action-adoption.v0"),
        ("status", "PARTIAL"),
        ("action_loss_only", False),
        ("nonzero_gradient_min_norm", 0.0),
        ("nonzero_gradient_min_norm", math.nan),
    ],
)
def test_action_adoption_identity_and_threshold_must_be_exact(
    field: str, replacement: object
) -> None:
    report, identity = _report()
    report["full_modal_action_adoption"][field] = replacement

    with pytest.raises(ContractError):
        _validate(report, identity)


def test_presence_subsets_require_the_exact_unique_eight_subset_lattice() -> None:
    report, identity = _report()
    report["full_modal_action_adoption"]["presence_subsets"].pop()
    with pytest.raises(ContractError, match="exactly eight"):
        _validate(report, identity)

    report, identity = _report()
    subsets = report["full_modal_action_adoption"]["presence_subsets"]
    subsets[1] = copy.deepcopy(subsets[0])
    with pytest.raises(ContractError, match="canonical order"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["presence_subsets"][4]["present_modalities"] = [
        "sonata",
        "anytouch",
    ]
    with pytest.raises(ContractError, match="wrong present modalities"):
        _validate(report, identity)


@pytest.mark.parametrize("branch", ["value", "metadata"])
def test_present_adapter_value_and_metadata_require_action_only_gradient(branch: str) -> None:
    report, identity = _report()
    gradient = report["full_modal_action_adoption"]["presence_subsets"][1][
        "adapter_action_only_gradients"
    ]["anytouch"][branch]
    gradient.update(norm=0.0, elements=10)

    with pytest.raises(ContractError, match="nonzero action-only gradient"):
        _validate(report, identity)


@pytest.mark.parametrize("branch", ["value", "metadata"])
def test_absent_adapter_value_and_metadata_reject_nonzero_gradient(branch: str) -> None:
    report, identity = _report()
    gradient = report["full_modal_action_adoption"]["presence_subsets"][0][
        "adapter_action_only_gradients"
    ]["vjepa"][branch]
    gradient.update(norm=0.1, elements=10)

    with pytest.raises(ContractError, match="absent or exactly zero"):
        _validate(report, identity)


@pytest.mark.parametrize(
    "path",
    [
        ("host_action_only_gradients", "0"),
        ("host_action_only_gradients", "18"),
        ("host_action_only_gradients", "35"),
        ("qwen_action_expert_action_only_gradient",),
        ("action_out_proj_action_only_gradient",),
    ],
)
def test_host_and_action_path_gradients_are_required_in_every_subset(path: tuple[str, ...]) -> None:
    report, identity = _report()
    target = report["full_modal_action_adoption"]["presence_subsets"][0]
    for field in path:
        target = target[field]
    target.update(norm=0.0, elements=10)

    with pytest.raises(ContractError, match="nonzero action-only gradient"):
        _validate(report, identity)


def test_gradient_nan_null_shape_and_duplicate_sample_keys_are_rejected() -> None:
    report, identity = _report()
    report["full_modal_action_adoption"]["presence_subsets"][7]["host_action_only_gradients"]["18"][
        "norm"
    ] = math.nan
    with pytest.raises(ContractError, match="finite"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["presence_subsets"][0]["adapter_action_only_gradients"][
        "anytouch"
    ]["value"].update(norm=None, elements=2)
    with pytest.raises(ContractError, match="null gradient"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["presence_subsets"][0]["sample_keys"] = [
        "probe-0",
        "probe-0",
    ]
    with pytest.raises(ContractError, match="sorted and unique"):
        _validate(report, identity)


def test_interventions_require_exact_unique_modalities_and_valid_permutation() -> None:
    report, identity = _report()
    report["full_modal_action_adoption"]["modality_interventions"].pop()
    with pytest.raises(ContractError, match="exactly three"):
        _validate(report, identity)

    report, identity = _report()
    interventions = report["full_modal_action_adoption"]["modality_interventions"]
    interventions[1] = copy.deepcopy(interventions[0])
    with pytest.raises(ContractError, match="canonical order"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["modality_interventions"][0]["token_permutations"] = [
        [0, 1, 2],
        [0, 1, 2],
    ]
    with pytest.raises(ContractError, match="nonidentity within-sample"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["modality_interventions"][0]["valid_after"][0] = [
        True,
        False,
        True,
    ]
    with pytest.raises(ContractError, match="does not follow"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["modality_interventions"][0]["token_permutations"][0] = [
        1,
        1,
        2,
    ]
    with pytest.raises(ContractError, match="must be a bijection"):
        _validate(report, identity)


def test_interventions_allow_sample_local_missing_modality_rows() -> None:
    report, identity = _report()
    anytouch = report["full_modal_action_adoption"]["modality_interventions"][0]
    anytouch["token_permutations"][0] = []
    anytouch["valid_before"][0] = []
    anytouch["valid_after"][0] = []
    _validate(report, identity)


@pytest.mark.parametrize(
    ("probe", "measured", "message"),
    [
        ("factual_repeat", 2e-6, "exceeds"),
        ("value_zero", 0.0, "does not reach"),
        ("metadata_zero", 0.0, "does not reach"),
        ("value_only_permutation", 0.0, "does not reach"),
        ("joint_value_metadata_valid_permutation", 2e-6, "exceeds"),
    ],
)
def test_each_modality_intervention_must_satisfy_its_registered_threshold(
    probe: str, measured: float, message: str
) -> None:
    for modality_index in range(3):
        report, identity = _report()
        report["full_modal_action_adoption"]["modality_interventions"][modality_index][probe][
            "measured_max_abs_action_drift"
        ] = measured
        with pytest.raises(ContractError, match=message):
            _validate(report, identity)


def test_intervention_nan_duplicate_samples_and_zero_minimum_are_rejected() -> None:
    report, identity = _report()
    report["full_modal_action_adoption"]["modality_interventions"][2]["value_zero"][
        "measured_max_abs_action_drift"
    ] = math.nan
    with pytest.raises(ContractError, match="finite"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["modality_interventions"][1]["sample_keys"] = [
        "probe-1",
        "probe-1",
    ]
    with pytest.raises(ContractError, match="sorted unique"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["modality_interventions"][0]["metadata_zero"][
        "minimum_required"
    ] = 0.0
    with pytest.raises(ContractError, match="positive"):
        _validate(report, identity)

    report, identity = _report()
    intervention = report["full_modal_action_adoption"]["modality_interventions"][0]
    intervention["value_zero"]["minimum_required"] = intervention["factual_repeat"][
        "maximum_allowed"
    ]
    with pytest.raises(ContractError, match="must exceed repeat/equivariance maxima"):
        _validate(report, identity)


def test_active_anytouch_sample_is_required_unique_and_grounded_in_both_probes() -> None:
    report, identity = _report()
    report["full_modal_action_adoption"]["active_anytouch_sample_keys"] = []
    with pytest.raises(ContractError, match="must not be empty"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["active_anytouch_sample_keys"] = [
        "touch-active-0",
        "touch-active-0",
    ]
    with pytest.raises(ContractError, match="sorted and unique"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["active_anytouch_sample_keys"] = ["unseen-touch"]
    with pytest.raises(ContractError, match="both present-subset and intervention"):
        _validate(report, identity)


@pytest.mark.parametrize("field", ["model_sha256", "optimizer_sha256", "lane_sha256", "rng_sha256"])
def test_dcp_cold_restore_requires_exact_typed_boundary_equality(field: str) -> None:
    report, identity = _report()
    report["full_modal_action_adoption"]["dcp_cold_restore"]["restored_boundary"][field] = _digest(
        f"forged-{field}"
    )

    with pytest.raises(ContractError, match="cold-restored boundary"):
        _validate(report, identity)


def test_dcp_requires_distinct_process_step1_and_nonduplicate_typed_digests() -> None:
    report, identity = _report()
    dcp = report["full_modal_action_adoption"]["dcp_cold_restore"]
    dcp["restore_process_sha256"] = dcp["save_process_sha256"]
    with pytest.raises(ContractError, match="distinct cold process"):
        _validate(report, identity)

    report, identity = _report()
    report["full_modal_action_adoption"]["dcp_cold_restore"]["optimizer_step"] = 2
    with pytest.raises(ContractError, match="optimizer step 1"):
        _validate(report, identity)

    report, identity = _report()
    boundary = report["full_modal_action_adoption"]["dcp_cold_restore"]["saved_boundary"]
    boundary["rng_sha256"] = boundary["lane_sha256"]
    with pytest.raises(ContractError, match="duplicate typed state digests"):
        _validate(report, identity)


def test_dcp_next_step_requires_exact_continuation_and_real_model_optimizer_update() -> None:
    report, identity = _report()
    report["full_modal_action_adoption"]["dcp_cold_restore"]["restored_next_step"][
        "action_loss"
    ] += 1e-6
    with pytest.raises(ContractError, match="next-step continuation"):
        _validate(report, identity)

    report, identity = _report()
    dcp = report["full_modal_action_adoption"]["dcp_cold_restore"]
    for branch in ("uninterrupted_next_step", "restored_next_step"):
        dcp[branch]["model_sha256"] = dcp["saved_boundary"]["model_sha256"]
    with pytest.raises(ContractError, match="did not update model"):
        _validate(report, identity)

    report, identity = _report()
    dcp = report["full_modal_action_adoption"]["dcp_cold_restore"]
    dcp["uninterrupted_next_step"]["action_loss"] = math.nan
    dcp["restored_next_step"]["action_loss"] = math.nan
    with pytest.raises(ContractError, match="finite"):
        _validate(report, identity)
