from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

from picf_next.lingbot_native.adr175_assembly import (
    Adr175AssemblyContractIdentity,
    assemble_adr175_arm_reports,
)
from picf_next.lingbot_native.adr175_validation import (
    ADR175_AMBIGUOUS_TASKS,
    ADR175_ARMS,
    ADR175_EXACT_TASK_TARGETS,
    ADR175_MILESTONES,
    canonical_sha256,
    validate_adr175_matched_three_arm,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _implementation_files() -> dict[str, str]:
    return {"tools/runner.py": _digest("runner")}


def _implementation_sha256() -> str:
    return canonical_sha256(_implementation_files())


def _prefix_inputs() -> tuple[list[str], list[str]]:
    sample_keys: list[str] = []
    prompt_receipts: list[str] = []
    for step in range(1, 2001):
        for rank in (0, 1):
            sample_keys.append(f"sample-{step}-rank-{rank}")
            prompt_receipts.append(_digest(f"prompt-{step}-rank-{rank}"))
    return sample_keys, prompt_receipts


def _contract() -> Adr175AssemblyContractIdentity:
    sample_keys, prompt_receipts = _prefix_inputs()
    return Adr175AssemblyContractIdentity(
        artifact_sha256="b" * 64,
        matched_arm_input_sha256="f" * 64,
        dataset_manifest_sha256="7" * 64,
        stream_plan_sha256="8" * 64,
        representation_split_artifact_sha256="9" * 64,
        entity_evaluation_plan_artifact_sha256="a" * 64,
        training_prefix_steps=2000,
        plan_total_steps=2000,
        global_batch_size=2,
        training_prefix_sample_count=4000,
        training_prefix_sample_keys_sha256=canonical_sha256(sample_keys),
        training_prefix_prompt_receipts_sha256=canonical_sha256(prompt_receipts),
    )


def _rank_reports(arm: str) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "steps": [
                {
                    "global_step": step,
                    "adr175_arm": arm,
                    "sample_keys": [f"sample-{step}-rank-{rank}"],
                    "source_digest": _digest(f"source-{step}-rank-{rank}"),
                    "frame_indices": [0],
                    "reset": [True],
                    "optimizer_lags": [0],
                    "policy_forward_absent": False,
                    "picf_graph_active": arm != "lbot",
                    "peak_cuda_reserved_bytes": 100,
                    "objective_total": 1.0,
                    "official_action_loss": 1.0,
                    "official_policy_loss": 1.0,
                    "action_family": 1.0,
                    "gradient_metrics": {
                        "all_finite": True,
                        "preclip_global_norm": 1.0,
                        "vlm_host_norm": 1.0,
                        "action_expert_norm": 1.0,
                        "native_graph_norm": 1.0 if arm != "lbot" else 0.0,
                    },
                    "adr175_input_receipt": {
                        "sample_sha256": canonical_sha256(
                            {
                                "sample_keys": [f"sample-{step}-rank-{rank}"],
                                "source_digest": _digest(f"source-{step}-rank-{rank}"),
                            }
                        ),
                        "action_target_sha256": _digest(f"action-{step}-rank-{rank}"),
                        "noise_sha256": _digest(f"noise-{step}-rank-{rank}"),
                        "time_sha256": _digest(f"time-{step}-rank-{rank}"),
                        "prompt_sha256": _digest(f"prompt-{step}-rank-{rank}"),
                    },
                }
                for step in range(1, 2001)
            ],
        }
        for rank in (0, 1)
    ]


def _snapshot(arm: str, step: int) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    tasks = [task_key for task_key, _keys in ADR175_EXACT_TASK_TARGETS]
    tasks.extend(ADR175_AMBIGUOUS_TASKS)
    progress = step / 2000.0
    for partition_index, partition in enumerate(("validation", "heldout")):
        for task_index, task_key in enumerate(tasks):
            replicate_count = 1 if partition == "validation" else 2
            for replicate in range(replicate_count):
                exact = task_key not in ADR175_AMBIGUOUS_TASKS
                target_identity_keys = dict(ADR175_EXACT_TASK_TARGETS).get(
                    task_key, (f"context/{task_key}",)
                )
                if arm == "physical-set":
                    selectivity = 0.30 + 0.10 * progress if exact else None
                    entity_score = 0.20 + 0.30 * progress
                elif arm == "native-attention":
                    selectivity = 0.40 + 0.10 * progress if exact else None
                    entity_score = 0.20 + 0.40 * progress
                else:
                    selectivity = None
                    entity_score = None
                samples.append(
                    {
                        "ordinal": len(samples),
                        "partition": partition,
                        "task_key": task_key,
                        "sample_key": f"{partition}/{task_key}/{replicate}",
                        "segment_index": replicate,
                        "source_episode_index": (
                            partition_index * 1000 + task_index * 3 + replicate
                        ),
                        "target_valid": exact if arm != "lbot" else False,
                        "posterior_adoption": 0.55 if exact and arm != "lbot" else None,
                        "conditional_selectivity": selectivity,
                        "official_action_loss": 1.0 if arm == "lbot" else 1.01,
                        "entity_evidence": (
                            None
                            if entity_score is None
                            else {
                                "rows": [
                                    {
                                        "identity_key": identity_key,
                                        "support_soft_iou_efficiency": entity_score,
                                    }
                                    for identity_key in target_identity_keys
                                ],
                                "target_visible_count": len(target_identity_keys),
                            }
                        ),
                    }
                )
    return {
        "schema": "picf-next.adr175-evaluation-snapshot.v1",
        "status": "PASS",
        "arm": arm,
        "checkpoint_global_step": step,
        "evaluation_input_sha256": "6" * 64,
        "implementation_sha256": _implementation_sha256(),
        "model_family_sha256": _digest(
            "model-family-treatment" if arm != "lbot" else "model-family-lbot"
        ),
        "stream_plan_sha256": "8" * 64,
        "representation_split_sha256": "9" * 64,
        "entity_evaluation_plan_sha256": "a" * 64,
        "partition_summaries": {"heldout": {"count": 68}, "validation": {"count": 68}},
        "samples": samples,
    }


def _raw_report(arm: str) -> dict[str, object]:
    optimizer_manifest = {
        "schema": "picf-next.adr175-shared-optimizer-manifest.v1",
        "expected_update_count": 2000,
        "groups": [],
        "shared_parameter_count": 0,
    }
    partition_summaries = {"heldout": {"count": 68}, "validation": {"count": 68}}
    treatment = arm != "lbot"
    return {
        "schema": "picf-next.lingbot-vla2-task-independent-p1-report.v1",
        "status": "PASS",
        "steps": 2000,
        "architecture_identity": "entity-posterior" if treatment else "released-lbot",
        "relation_interface": "physical-entities" if treatment else None,
        "source_commit": "2838c1862bbec1ea47942fb61512130f635eb595",
        "source_patch_sha256": "2" * 64,
        "patched_source_sha256": {"model.py": "3" * 64},
        "implementation_files": _implementation_files(),
        "implementation_sha256": _implementation_sha256(),
        "checkpoint_revision": "checkpoint-revision",
        "checkpoint_assets": [{"path": "model", "sha256": "4" * 64}],
        "processor_revision": "processor-revision",
        "processor_assets": [{"path": "processor", "sha256": "5" * 64}],
        "dataset_contract": {
            "manifest_sha256": "0" * 64,
            "status": "PASS",
            "validation": {"dataset_tree_sha256": "7" * 64},
        },
        "physical_sidecar_manifest_sha256": "6" * 64,
        "plan_sha256": "8" * 64,
        "representation_split_sha256": "9" * 64,
        "entity_evaluation_plan_sha256": "a" * 64,
        "evaluation_snapshots": [
            {
                "arm": arm,
                "step": step,
                "artifact_sha256": _digest(f"snapshot-{arm}-{step}"),
                "checkpoint_global_step": step,
                "evaluation_input_sha256": "6" * 64,
                "file_sha256": _digest(f"snapshot-file-{arm}-{step}"),
                "partition_summaries": partition_summaries,
            }
            for step in ADR175_MILESTONES
        ],
        "rank_reports": _rank_reports(arm),
        "model_family_sha256": _digest(
            "model-family-treatment" if treatment else "model-family-lbot"
        ),
        "graph": {"capacity": 16} if treatment else None,
        "parameter_manifest": {"schema_sha256": "d" * 64},
        "parameter_scope": "full_joint_action",
        "parameter_storage": {"placement": "selective-embedding-offload"},
        "action_suffix_executed": True,
        "alignment_teacher_prune": {"schema": "prune.v1"},
        "checkpoint_published": False,
        "cuda_allocator": "native",
        "curve_mode": True,
        "fsdp2_placement": "selective-embedding-offload",
        "gradient_checkpointing": True,
        "maximum_peak_reserved_bytes": 1000,
        "posterior_input_mode": "current_frame_joint_action",
        "registered_evaluation_steps": list(ADR175_MILESTONES),
        "representation_parameter_scope": None,
        "seed": 20260816,
        "task_scorer_present": False,
        "world_size": 2,
        "objective": {"action_weight": 1.0, "entity_weight": 0.08},
        "qwen_vision_geometry": {"visual_lattice": 8},
        "adr175": {
            "arm": arm,
            "contract_artifact_sha256": "b" * 64,
            "contract_file_sha256": "c" * 64,
            "matched_arm_input_sha256": "f" * 64,
            "shared_initialization_sha256": "c" * 64,
            "shared_optimizer_contract_sha256": canonical_sha256(optimizer_manifest),
            "shared_optimizer_manifest": optimizer_manifest,
            "picf_graph_sha256": None if not treatment else "d" * 64,
            "picf_initialization_sha256": None if not treatment else "e" * 64,
        },
    }


def _snapshots() -> dict[tuple[str, int], dict[str, object]]:
    return {(arm, step): _snapshot(arm, step) for arm in ADR175_ARMS for step in ADR175_MILESTONES}


def _loader(
    snapshots: dict[tuple[str, int], dict[str, object]],
) -> Any:
    def load(receipt: object) -> dict[str, Any]:
        assert isinstance(receipt, dict)
        return snapshots[(str(receipt["arm"]), int(receipt["step"]))]

    return load


def test_adr175_assembly_builds_validator_ready_task_macro_reports() -> None:
    snapshots = _snapshots()

    reports = assemble_adr175_arm_reports(
        {arm: _raw_report(arm) for arm in ADR175_ARMS},
        broad_support_contract=_contract(),
        raw_report_file_sha256_by_arm={arm: _digest(f"raw-{arm}") for arm in ADR175_ARMS},
        snapshot_loader=_loader(snapshots),
    )
    result = validate_adr175_matched_three_arm(tuple(reports.values()))

    assert result.status == "PASS"
    assert reports["lbot"]["exact_strata"] is None
    assert len(reports["native-attention"]["exact_strata"]) == 29
    assert reports["native-attention"]["heldout_selectivity_bootstrap"]["raw_lower_bound"] > 0
    assert reports["native-attention"]["heldout_selectivity_bootstrap"]["cluster_count"] == 58
    assert (
        reports["native-attention"]["heldout_selectivity_bootstrap"]["resampling_scheme"]
        == "paired_global_source_episode_bayesian"
    )


def test_adr175_assembly_scores_an_invisible_exact_partition_conservatively() -> None:
    snapshots = _snapshots()
    task_key = ADR175_EXACT_TASK_TARGETS[0][0]
    for arm in ("physical-set", "native-attention"):
        for step in ADR175_MILESTONES:
            sample = next(
                sample
                for sample in snapshots[(arm, step)]["samples"]  # type: ignore[index]
                if sample["task_key"] == task_key  # type: ignore[index]
                and sample["partition"] == "validation"  # type: ignore[index]
            )
            sample["target_valid"] = False  # type: ignore[index]
            sample["posterior_adoption"] = None  # type: ignore[index]
            sample["conditional_selectivity"] = None  # type: ignore[index]
            sample["entity_evidence"]["rows"] = [  # type: ignore[index]
                {
                    "identity_key": "different/visible_entity",
                    "support_soft_iou_efficiency": 0.2,
                }
            ]
            sample["entity_evidence"]["target_visible_count"] = 1  # type: ignore[index]

    reports = assemble_adr175_arm_reports(
        {arm: _raw_report(arm) for arm in ADR175_ARMS},
        broad_support_contract=_contract(),
        raw_report_file_sha256_by_arm={arm: _digest(f"raw-{arm}") for arm in ADR175_ARMS},
        snapshot_loader=_loader(snapshots),
    )

    for arm in ("physical-set", "native-attention"):
        stratum = next(
            row
            for row in reports[arm]["exact_strata"]  # type: ignore[index]
            if row["task_key"] == task_key  # type: ignore[index]
        )
        assert stratum["validation_score"] == 0.0  # type: ignore[index]
        assert stratum["heldout_score"] > 0.0  # type: ignore[index]


def test_adr175_assembly_scores_observable_resolution_failure_as_zero() -> None:
    snapshots = _snapshots()
    task_key = ADR175_EXACT_TASK_TARGETS[0][0]
    sample = next(
        sample
        for sample in snapshots[("native-attention", 2000)]["samples"]  # type: ignore[index]
        if sample["task_key"] == task_key  # type: ignore[index]
        and sample["partition"] == "validation"  # type: ignore[index]
    )
    sample["target_valid"] = False  # type: ignore[index]
    sample["posterior_adoption"] = None  # type: ignore[index]
    sample["conditional_selectivity"] = None  # type: ignore[index]

    reports = assemble_adr175_arm_reports(
        {arm: _raw_report(arm) for arm in ADR175_ARMS},
        broad_support_contract=_contract(),
        raw_report_file_sha256_by_arm={arm: _digest(f"raw-{arm}") for arm in ADR175_ARMS},
        snapshot_loader=_loader(snapshots),
    )

    stratum = next(
        row
        for row in reports["native-attention"]["exact_strata"]  # type: ignore[index]
        if row["task_key"] == task_key  # type: ignore[index]
    )
    assert stratum["validation_censored"] is False  # type: ignore[index]
    assert stratum["validation_score"] == 0.0  # type: ignore[index]


def test_adr175_assembly_rejects_cross_arm_observability_drift() -> None:
    snapshots = _snapshots()
    task_key = ADR175_EXACT_TASK_TARGETS[0][0]
    for step in ADR175_MILESTONES:
        sample = next(
            sample
            for sample in snapshots[("native-attention", step)]["samples"]  # type: ignore[index]
            if sample["task_key"] == task_key  # type: ignore[index]
            and sample["partition"] == "validation"  # type: ignore[index]
        )
        sample["target_valid"] = False  # type: ignore[index]
        sample["posterior_adoption"] = None  # type: ignore[index]
        sample["conditional_selectivity"] = None  # type: ignore[index]
        sample["entity_evidence"]["rows"] = [  # type: ignore[index]
            {
                "identity_key": "different/visible_entity",
                "support_soft_iou_efficiency": 0.2,
            }
        ]
        sample["entity_evidence"]["target_visible_count"] = 1  # type: ignore[index]

    with pytest.raises(ValueError, match="observability differs across arms"):
        assemble_adr175_arm_reports(
            {arm: _raw_report(arm) for arm in ADR175_ARMS},
            broad_support_contract=_contract(),
            raw_report_file_sha256_by_arm={arm: _digest(f"raw-{arm}") for arm in ADR175_ARMS},
            snapshot_loader=_loader(snapshots),
        )


def test_adr175_assembly_rejects_evaluation_sample_multiplicity_drift() -> None:
    snapshots = _snapshots()
    duplicate = copy.deepcopy(snapshots[("lbot", 0)]["samples"][0])  # type: ignore[index]
    duplicate["sample_key"] = "extra-validation-sample"  # type: ignore[index]
    duplicate["source_episode_index"] = 99999  # type: ignore[index]
    snapshots[("lbot", 0)]["samples"].append(duplicate)  # type: ignore[index]

    with pytest.raises(ValueError, match="requires 1 samples"):
        assemble_adr175_arm_reports(
            {arm: _raw_report(arm) for arm in ADR175_ARMS},
            broad_support_contract=_contract(),
            raw_report_file_sha256_by_arm={arm: _digest(f"raw-{arm}") for arm in ADR175_ARMS},
            snapshot_loader=_loader(snapshots),
        )


def test_adr175_assembly_rejects_prefix_sample_order_drift() -> None:
    raw_reports = {arm: _raw_report(arm) for arm in ADR175_ARMS}
    step = raw_reports["native-attention"]["rank_reports"][0]["steps"][0]  # type: ignore[index]
    step["sample_keys"] = ["different-sample"]  # type: ignore[index]
    step["adr175_input_receipt"]["sample_sha256"] = canonical_sha256(  # type: ignore[index]
        {
            "sample_keys": ["different-sample"],
            "source_digest": step["source_digest"],  # type: ignore[index]
        }
    )

    with pytest.raises(ValueError, match="sample order differs"):
        assemble_adr175_arm_reports(
            raw_reports,
            broad_support_contract=_contract(),
            raw_report_file_sha256_by_arm={arm: _digest(f"raw-{arm}") for arm in ADR175_ARMS},
            snapshot_loader=_loader(_snapshots()),
        )


def test_adr175_assembly_rejects_snapshot_parent_drift() -> None:
    snapshots = _snapshots()
    snapshots[("native-attention", 250)]["implementation_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="parent raw run"):
        assemble_adr175_arm_reports(
            {arm: _raw_report(arm) for arm in ADR175_ARMS},
            broad_support_contract=_contract(),
            raw_report_file_sha256_by_arm={arm: _digest(f"raw-{arm}") for arm in ADR175_ARMS},
            snapshot_loader=_loader(snapshots),
        )


def test_adr175_assembly_rejects_shared_runtime_drift() -> None:
    raw_reports = {arm: _raw_report(arm) for arm in ADR175_ARMS}
    raw_reports["physical-set"]["seed"] = 1

    with pytest.raises(ValueError, match="shared contracts differ"):
        assemble_adr175_arm_reports(
            raw_reports,
            broad_support_contract=_contract(),
            raw_report_file_sha256_by_arm={arm: _digest(f"raw-{arm}") for arm in ADR175_ARMS},
            snapshot_loader=_loader(_snapshots()),
        )


def test_adr175_assembly_rejects_treatment_layout_drift() -> None:
    raw_reports = {arm: _raw_report(arm) for arm in ADR175_ARMS}
    native_storage = copy.deepcopy(raw_reports["native-attention"]["parameter_storage"])
    native_storage["placement"] = "different"  # type: ignore[index]
    raw_reports["native-attention"]["parameter_storage"] = native_storage

    with pytest.raises(ValueError, match="treatment contracts differ"):
        assemble_adr175_arm_reports(
            raw_reports,
            broad_support_contract=_contract(),
            raw_report_file_sha256_by_arm={arm: _digest(f"raw-{arm}") for arm in ADR175_ARMS},
            snapshot_loader=_loader(_snapshots()),
        )
