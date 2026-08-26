from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.calvin_geometry_schema import calvin_source_state_sha256
from picf_next.data.calvin_target_request import (
    native_calvin_structural_target_request,
)
from picf_next.data.dataset_manifest import (
    DATASET_RUNTIME_VERIFICATION_MODE,
    build_dataset_file_manifest,
)
from picf_next.data.token_supervision_policy import (
    build_known_pixel_token_supervision_policy,
    token_supervision_policy_sha256,
)
from picf_next.lingbot_native.calvin import (
    audit_native_calvin_model_inputs,
    build_native_calvin_continuation_batch,
    build_native_calvin_training_stream_plan,
    build_planned_native_calvin_batch,
    collate_native_calvin_training_batch,
)
from picf_next.lingbot_native.fixed_observation import (
    CALVIN_FIXED_OBSERVATION_AUDIT_SCHEMA,
    CALVIN_FIXED_OBSERVATION_AUDIT_SCHEMA_V2,
    FixedObservationAudit,
    FixedObservationGroup,
    FixedObservationPairPlan,
    FixedObservationVariant,
    apply_fixed_observation_pair,
    build_fixed_observation_pair_plan,
    load_fixed_observation_audit,
    load_native_vl_grounding_audit,
    validate_fixed_observation_group_source,
    validate_fixed_observation_group_source_index,
)
from picf_next.lingbot_native.fixed_observation_evaluation import (
    FixedObservationEvaluationItem,
    FixedObservationEvaluationPlan,
    build_fixed_observation_evaluation_plan,
    build_fixed_observation_evaluation_sample,
    build_fixed_observation_evaluation_snapshot,
    build_fixed_observation_forward_equivalence_probe,
    fixed_observation_evaluation_mass_strata,
    fixed_observation_prompt_switch_metrics,
)
from picf_next.lingbot_native.fixed_observation_gate import (
    build_fixed_observation_numeric_gate,
    validate_fixed_observation_numeric_gate,
    write_fixed_observation_numeric_gate,
)
from picf_next.lingbot_native.fixed_observation_training_contract import (
    validate_fixed_observation_training_pair_fingerprints,
    validate_fixed_observation_training_rank_metadata,
)
from picf_next.lingbot_native.representation_evaluation import (
    build_representation_ownership_row,
    build_representation_token_evidence,
    summarize_representation_ownership_rows,
)
from picf_next.lingbot_native.representation_evaluation_runtime import (
    _prepare_fixed_observation_evaluation_pair,
    fixed_observation_training_pair_fingerprint,
)
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
)
from picf_next.lingbot_native.task_diagnostics import TASK_ROW_DIAGNOSTIC_SCHEMA
from picf_next.lingbot_native.visual_audit import NATIVE_VISUAL_AUDIT_SCHEMA
from picf_next.training.control import FrozenResetMixtureStreamPlan
from tests.test_calvin_data import _frame

_SHA = {
    name: hashlib.sha256(name.encode("ascii")).hexdigest()
    for name in (
        "applicability-artifact",
        "applicability-file",
        "audit-artifact",
        "dataset-file",
        "dataset-tree",
        "physical-sidecar",
        "projection-contract",
        "projection-payload",
        "split-artifact",
        "split-file",
        "stream-plan",
        "visual",
    )
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _variant_record(task: str, target: str, mass: float) -> dict[str, object]:
    instruction = f"perform the complete {task} instruction"
    return {
        "fixed_x_diagnostic_eligible": True,
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "proof": f"simulator-proof:{task}",
        "support": {
            "identity_key": target,
            "maximum_target_probability": 0.75,
            "measurable": True,
            "object_row_addressable": True,
            "positive_token_count": 2,
            "strict_categorical_winner_token_count": 1,
            "strict_object_winner_token_count": 2,
            "target_mass": mass,
            "views": [
                {
                    "camera_name": "static",
                    "maximum_target_probability": 0.75,
                    "measurable": True,
                    "merged_grid_hw": [8, 8],
                    "object_row_addressable": True,
                    "positive_token_count": 2,
                    "strict_categorical_winner_token_count": 1,
                    "strict_object_winner_token_count": 2,
                    "target_mass": mass,
                }
            ],
        },
        "target_identity_key": target,
        "task_key": task,
    }


def _report() -> dict[str, object]:
    variants = [
        _variant_record("turn_on_led", "part/table/button_link", 1.0),
        _variant_record("move_slider_left", "part/table/slide_link", 0.5),
    ]
    policy = build_known_pixel_token_supervision_policy()
    content = {
        "acceptance_scope": {
            "fixed_x_evaluation_bank_authorized": False,
            "fixed_x_partition_artifact_authorized": True,
            "fixed_x_training_stream_plan_authorized": True,
            "raw_owner_visibility_proven": True,
            "representation_partition_isolation_proven": True,
            "source_state_and_sensor_hash_binding_proven": True,
            "stateful_reset_addressability_proven": True,
            "token_grid_measurability_proven_for_retained_variants": True,
            "training_authorized": False,
        },
        "applicability_artifact_sha256": _SHA["applicability-artifact"],
        "applicability_report_sha256": _SHA["applicability-file"],
        "dataset_manifest_sha256": _SHA["dataset-file"],
        "dataset_runtime_binding": {
            "dataset_manifest_self_consistent": True,
            "dataset_runtime_verified_read_required": True,
            "dataset_tree_sha256": _SHA["dataset-tree"],
            "dataset_verification_mode": DATASET_RUNTIME_VERIFICATION_MODE,
        },
        "group_count": 1,
        "groups": [
            {
                "fixed_x_group_eligible": True,
                "retained_target_identity_keys": [
                    "part/table/button_link",
                    "part/table/slide_link",
                ],
                "retained_task_keys": ["turn_on_led", "move_slider_left"],
                "scene": "test-scene",
                "source_global_index": 10,
                "source_sensor_sha256": {
                    name: hashlib.sha256(name.encode("ascii")).hexdigest()
                    for name in (
                        "depth_gripper",
                        "depth_static",
                        "rgb_gripper",
                        "rgb_static",
                    )
                },
                "source_state_sha256": hashlib.sha256(b"state").hexdigest(),
                "stateful_reset_binding": {
                    "language_segment_index": 0,
                    "source_episode_index": 0,
                    "source_instruction_sha256": hashlib.sha256(b"source instruction").hexdigest(),
                    "source_task_key": "source_task",
                    "stateful_episode_key": "calvin-language-segment-00000000",
                    "stateful_sample_key": (
                        "calvin-language-segment-00000000/transition-00000000-frame-00000010"
                    ),
                    "transition_index": 0,
                },
                "variants": variants,
            }
        ],
        "leakage_contract": {
            "model_input_contains_applicability_proof": False,
            "model_input_contains_complete_natural_instruction": True,
            "model_input_contains_identity_or_owner": False,
            "model_input_contains_representation_split_metadata": False,
            "model_input_contains_simulator_state": False,
            "model_input_contains_stateful_binding": False,
            "model_input_contains_target": False,
            "model_input_contains_task_key": False,
        },
        "measurement_contract": {
            "absolute_pixel_or_probability_threshold": None,
            "context_is_not_an_object_row": True,
            "fixed_x_retention_rule": (
                "target-owner-mass-strictly-exceeds-every-other-physical-object-in-"
                "at-least-one-supervised-merged-token"
            ),
            "model_input": False,
            "projection": "exact-pinned-qwen3vl-patch-and-spatial-merger-addresses",
            "target_measure": "known-owner-mass-conditioned-within-token",
        },
        "physical_sidecar_manifest_sha256": _SHA["physical-sidecar"],
        "representation_split": {
            "artifact_sha256": _SHA["split-artifact"],
            "comparison_id": "fixed-X-test",
            "file_sha256": _SHA["split-file"],
            "partition": "training",
            "partition_segment_count": 1,
            "partition_source_episode_count": 1,
            "schema": REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
            "stream_plan_sha256": _SHA["stream-plan"],
        },
        "rejected_groups": [],
        "rejected_visual_artifacts": [],
        "schema": CALVIN_FIXED_OBSERVATION_AUDIT_SCHEMA,
        "source_group_count": 1,
        "status": "PASS",
        "summary": {
            "addressable_target_histogram": {
                "part/table/button_link": 1,
                "part/table/slide_link": 1,
            },
            "addressable_task_histogram": {
                "move_slider_left": 1,
                "turn_on_led": 1,
            },
            "addressable_variant_count": 2,
            "dropped_variant_count": 0,
            "eligible_group_count": 1,
            "ineligible_group_count": 0,
            "retained_target_histogram": {
                "part/table/button_link": 1,
                "part/table/slide_link": 1,
            },
            "retained_task_histogram": {
                "move_slider_left": 1,
                "turn_on_led": 1,
            },
            "retained_variant_count": 2,
            "source_variant_count": 2,
            "stranded_addressable_variant_count": 0,
        },
        "training_projection_contract_sha256": _SHA["projection-contract"],
        "training_projection_payload_sha256": _SHA["projection-payload"],
        "training_supervision_policy": policy,
        "training_supervision_policy_sha256": token_supervision_policy_sha256(policy),
        "visual_artifacts": [
            {
                "file": "source_0000010.png",
                "png_sha256": _SHA["visual"],
                "source_global_index": 10,
            }
        ],
    }
    return {**content, "artifact_sha256": _canonical_sha256(content)}


def _write_report(path: Path, value: dict[str, object]) -> str:
    content = {name: child for name, child in value.items() if name != "artifact_sha256"}
    rebound = {**content, "artifact_sha256": _canonical_sha256(content)}
    payload = (
        json.dumps(
            rebound,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _report_with_one_rejected_group() -> dict[str, object]:
    report = _report()
    eligible_groups = report["groups"]
    assert isinstance(eligible_groups, list)
    rejected = json.loads(json.dumps(eligible_groups[0]))
    rejected["fixed_x_group_eligible"] = False
    rejected["source_global_index"] = 11
    rejected["source_state_sha256"] = hashlib.sha256(b"rejected-state").hexdigest()
    reset = rejected["stateful_reset_binding"]
    reset["stateful_sample_key"] = (
        "calvin-language-segment-00000000/transition-00000000-frame-00000011"
    )
    variants = rejected["variants"]
    second = variants[1]
    second["fixed_x_diagnostic_eligible"] = False
    support = second["support"]
    support["object_row_addressable"] = False
    support["strict_categorical_winner_token_count"] = 0
    support["strict_object_winner_token_count"] = 0
    views = support["views"]
    views[0]["object_row_addressable"] = False
    views[0]["strict_categorical_winner_token_count"] = 0
    views[0]["strict_object_winner_token_count"] = 0
    rejected["retained_task_keys"] = ["turn_on_led"]
    rejected["retained_target_identity_keys"] = ["part/table/button_link"]
    report["rejected_groups"] = [rejected]
    report["rejected_visual_artifacts"] = [
        {
            "file": "source_0000011.png",
            "png_sha256": hashlib.sha256(b"rejected-visual").hexdigest(),
            "source_global_index": 11,
        }
    ]
    report["source_group_count"] = 2
    summary = report["summary"]
    summary.update(
        {
            "addressable_target_histogram": {
                "part/table/button_link": 2,
                "part/table/slide_link": 1,
            },
            "addressable_task_histogram": {
                "move_slider_left": 1,
                "turn_on_led": 2,
            },
            "addressable_variant_count": 3,
            "dropped_variant_count": 2,
            "ineligible_group_count": 1,
            "source_variant_count": 4,
            "stranded_addressable_variant_count": 1,
        }
    )
    return report


def _many_segment_dataset(
    tmp_path: Path,
    *,
    segment_count: int = 12,
) -> CalvinStatefulTransitionDataset:
    split = tmp_path / "training"
    (split / ".hydra").mkdir(parents=True)
    (split / "lang_annotations").mkdir()
    (split / ".hydra" / "merged_config.yaml").write_text(
        "env:\n  control_freq: 30\n",
        encoding="ascii",
    )
    start = 10
    end = start + 2 * segment_count - 1
    np.save(split / "ep_start_end_ids.npy", np.asarray([[start, end]], dtype=np.int64))
    np.save(split / "ep_lens.npy", np.asarray(end - start + 1, dtype=np.int64))
    annotations = {
        "language": {
            "ann": [f"source instruction {index:02d}" for index in range(segment_count)],
            "task": [f"source_task_{index:02d}" for index in range(segment_count)],
        },
        "info": {
            "indx": [(start + 2 * index, start + 2 * index + 1) for index in range(segment_count)]
        },
    }
    np.save(
        split / "lang_annotations" / "auto_lang_ann.npy",
        annotations,  # type: ignore[arg-type]
    )
    for global_index in range(start, end + 1):
        frame = _frame()
        absolute_x = min((global_index - start) * 0.00025, 0.019)
        frame["actions"][0] = absolute_x
        frame["rel_actions"][0] = absolute_x / 0.02
        frame["scene_obs"][0] = float(global_index)
        np.savez(split / f"episode_{global_index:07d}.npz", **frame)
    relative_paths = (
        ".hydra/merged_config.yaml",
        "ep_lens.npy",
        "ep_start_end_ids.npy",
        "lang_annotations/auto_lang_ann.npy",
        *(f"episode_{index:07d}.npz" for index in range(start, end + 1)),
    )
    manifest = build_dataset_file_manifest(
        split,
        dataset_id="calvin-fixed-X-test",
        dataset_revision="sha256:fixed-X-test",
        split_name="training",
        relative_paths=relative_paths,
    )
    index = CalvinDatasetIndex.load(
        split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        dataset_manifest=manifest,
    )
    return CalvinStatefulTransitionDataset(index, action_horizon=1)


def _pair_fixture(
    tmp_path: Path,
    *,
    total_steps: int = 4,
) -> tuple[
    CalvinStatefulTransitionDataset,
    FrozenResetMixtureStreamPlan,
    FixedObservationAudit,
    FixedObservationPairPlan,
]:
    dataset = _many_segment_dataset(
        tmp_path,
        segment_count=max(12, 2 * total_steps),
    )
    stream = build_native_calvin_training_stream_plan(
        dataset,
        comparison_id="fixed-X-test",
        seed=71,
        global_batch_size=2,
        total_steps=total_steps,
        reset_numerator=1,
        reset_denominator=2,
    )
    assert isinstance(stream, FrozenResetMixtureStreamPlan)
    groups = []
    for sample_key in stream.reset_sample_keys:
        sample = dataset.by_key(sample_key)
        locator = dataset.locator_by_key(sample_key)
        segment = dataset.index.segments[locator.segment_index]
        source_index = sample.record.global_index
        source_arrays = dataset.index.validated_source_frame_arrays(
            source_index,
            fields=("robot_obs", "scene_obs"),
        )
        request = native_calvin_structural_target_request(sample)
        groups.append(
            FixedObservationGroup(
                scene="test-scene",
                source_global_index=source_index,
                source_state_sha256=calvin_source_state_sha256(
                    source_arrays["scene_obs"],
                    source_arrays["robot_obs"],
                ),
                source_sensor_sha256=request.source_sensor_sha256,
                source_episode_index=dataset.index.source_episode(source_index).index,
                source_task_key=segment.task_key,
                source_instruction_sha256=hashlib.sha256(
                    segment.instruction.encode("utf-8")
                ).hexdigest(),
                stateful_episode_key=sample.episode_key,
                stateful_sample_key=sample.sample_key,
                variants=(
                    FixedObservationVariant(
                        task_key="task_alpha",
                        instruction="perform the complete task alpha instruction",
                        instruction_sha256=hashlib.sha256(
                            b"perform the complete task alpha instruction"
                        ).hexdigest(),
                        target_identity_key="physical/alpha",
                        target_mass=1.0,
                    ),
                    FixedObservationVariant(
                        task_key="task_beta",
                        instruction="perform the complete task beta instruction",
                        instruction_sha256=hashlib.sha256(
                            b"perform the complete task beta instruction"
                        ).hexdigest(),
                        target_identity_key="physical/beta",
                        target_mass=0.5,
                    ),
                ),
            )
        )
    audit = FixedObservationAudit(
        partition="training",
        report_file_sha256=_SHA["applicability-file"],
        report_artifact_sha256=_SHA["audit-artifact"],
        dataset_manifest_file_sha256=_SHA["dataset-file"],
        dataset_tree_sha256=stream.dataset_manifest_sha256,
        representation_split_file_sha256=_SHA["split-file"],
        representation_split_artifact_sha256=_SHA["split-artifact"],
        comparison_id=stream.comparison_id,
        stream_plan_sha256=stream.plan_sha256,
        training_projection_contract_sha256=_SHA["projection-contract"],
        training_projection_payload_sha256=_SHA["projection-payload"],
        groups=tuple(sorted(groups, key=lambda item: item.source_global_index)),
    )
    plan = build_fixed_observation_pair_plan(stream, dataset, audit)
    return dataset, stream, audit, plan


def _evaluation_audit(
    training: FixedObservationAudit,
    *,
    partition: str,
    source_offset: int,
) -> FixedObservationAudit:
    groups = tuple(
        replace(
            group,
            source_global_index=source_offset + ordinal,
            source_state_sha256=hashlib.sha256(
                f"{partition}-state-{ordinal}".encode("ascii")
            ).hexdigest(),
            source_episode_index=source_offset + ordinal,
            stateful_episode_key=(f"calvin-language-segment-{source_offset + ordinal:08d}"),
            stateful_sample_key=(
                f"calvin-language-segment-{source_offset + ordinal:08d}/"
                f"transition-00000000-frame-{source_offset + ordinal:08d}"
            ),
        )
        for ordinal, group in enumerate(training.groups)
    )
    return replace(
        training,
        partition=partition,
        report_file_sha256=hashlib.sha256(f"{partition}-audit-file".encode("ascii")).hexdigest(),
        report_artifact_sha256=hashlib.sha256(
            f"{partition}-audit-artifact".encode("ascii")
        ).hexdigest(),
        groups=groups,
    )


def test_sparse_fixed_observation_source_binding_matches_materialized_map(
    tmp_path: Path,
) -> None:
    dataset, _stream, audit, _plan = _pair_fixture(tmp_path)
    group = audit.groups[0]
    native_horizon_dataset = CalvinStatefulTransitionDataset(
        dataset.index,
        action_horizon=50,
    )

    sparse = validate_fixed_observation_group_source_index(dataset.index, group)
    materialized = validate_fixed_observation_group_source(
        native_horizon_dataset,
        group,
        action_horizon=native_horizon_dataset.action_horizon,
    )

    assert sparse.sample_key == materialized.sample_key == group.stateful_sample_key
    assert sparse.episode_key == materialized.episode_key == group.stateful_episode_key
    assert sparse.record.global_index == materialized.record.global_index
    assert sparse.record.task_index == materialized.record.task_index
    assert sparse.host_sample.action.shape == (1, 7)
    assert materialized.host_sample.action.shape == (50, 7)
    assert materialized.host_sample.action_is_pad.shape == (50,)

    with pytest.raises(ValueError, match="malformed stateful episode key"):
        validate_fixed_observation_group_source_index(
            dataset.index,
            replace(group, stateful_episode_key="calvin-language-segment-invalid"),
        )
    with pytest.raises(ValueError, match="immutable stateful sample"):
        validate_fixed_observation_group_source_index(
            dataset.index,
            replace(group, stateful_sample_key=f"{group.stateful_sample_key}-tampered"),
        )


class _PromptAwareTransform:
    def apply(self, item, policy_eval=False):
        assert not policy_eval
        instruction = item["task"]
        prompt_code = sum(instruction.encode("utf-8"))
        actions = item["action.lingbot"]
        return {
            "action_is_pad": item["action.lingbot_is_pad"],
            "action_joint_mask": torch.ones(55, dtype=torch.bool),
            "actions": actions,
            "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
            "images": torch.stack(
                (
                    item["observation.images.camera_top"],
                    torch.nn.functional.interpolate(
                        item["observation.images.camera_wrist_left"][None],
                        size=(200, 200),
                        mode="nearest",
                    )[0],
                )
            ),
            "img_masks": torch.ones(2, dtype=torch.bool),
            "joint_mask": torch.ones(actions.shape, dtype=torch.bool),
            "lang_masks": torch.ones(1, dtype=torch.bool),
            "lang_tokens": torch.asarray([prompt_code], dtype=torch.long),
            "state": item["observation.state.lingbot"],
            "state_joint_mask": torch.ones(55, dtype=torch.bool),
        }


def _collator(items):
    return {name: torch.stack([item[name] for item in items]) for name in items[0]}


def _task_diagnostic(
    *,
    target_index: int,
    logits: tuple[float, float],
) -> dict[str, object]:
    identities = ["physical/alpha", "physical/beta"]
    probabilities = [1.0 / (1.0 + math.exp(-value)) for value in logits]
    other = 1 - target_index
    margin = logits[target_index] - logits[other]
    probability_margin = probabilities[target_index] - probabilities[other]
    targets = [0.0, 0.0]
    targets[target_index] = 1.0
    assignment = {
        "identity_keys": identities,
        "sequence_time_count": 1,
        "source_phase": 1,
        "binding_start_phase": [1, 1],
        "source_binding_valid": [True, True],
        "row_to_track": [0, 1],
    }
    return {
        "schema": TASK_ROW_DIAGNOSTIC_SCHEMA,
        "exact_task": True,
        "identity_keys": identities,
        "track_task_targets": targets,
        "track_task_valid": [True, True],
        "capacity_censored": [False, False],
        "sequence_time_count": 1,
        "source_time": 0,
        "source_side": "posterior",
        "source_phase": 1,
        "binding_start_phase": [1, 1],
        "source_binding_valid": [True, True],
        "row_to_track": [0, 1],
        "assignment_sha256": _canonical_sha256(assignment),
        "row_task_targets": targets,
        "row_task_valid": [True, True],
        "task_logits": list(logits),
        "task_probabilities": probabilities,
        "target_rows": [target_index],
        "target_identity_keys": [identities[target_index]],
        "materialized_target_identity_keys": [identities[target_index]],
        "unmaterialized_target_identity_keys": [],
        "known_negative_rows": [other],
        "minimum_target_logit": logits[target_index],
        "maximum_known_negative_logit": logits[other],
        "target_vs_hardest_negative_logit_margin": margin,
        "minimum_target_probability": probabilities[target_index],
        "maximum_known_negative_probability": probabilities[other],
        "target_vs_hardest_negative_probability_margin": probability_margin,
        "worst_target_rank": 1 if margin > 0.0 else 2,
        "all_targets_beat_known_negatives": margin > 0.0,
    }


def test_audit_loader_recomputes_contract_and_rejects_resigned_semantic_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.json"
    report = _report()
    file_sha256 = _write_report(path, report)
    audit = load_fixed_observation_audit(
        path,
        expected_file_sha256=file_sha256,
        expected_partition="training",
    )

    assert audit.partition == "training"
    assert audit.task_keys == ("move_slider_left", "turn_on_led")
    assert audit.target_identity_keys == (
        "part/table/button_link",
        "part/table/slide_link",
    )

    leakage = report["leakage_contract"]
    assert isinstance(leakage, dict)
    leakage["model_input_contains_target"] = True
    changed_sha256 = _write_report(path, report)
    with pytest.raises(ValueError, match="leakage contract"):
        load_fixed_observation_audit(
            path,
            expected_file_sha256=changed_sha256,
            expected_partition="training",
        )

    report = _report()
    measurement = report["measurement_contract"]
    assert isinstance(measurement, dict)
    measurement["absolute_pixel_or_probability_threshold"] = 0.1
    changed_sha256 = _write_report(path, report)
    with pytest.raises(ValueError, match="measurement contract"):
        load_fixed_observation_audit(
            path,
            expected_file_sha256=changed_sha256,
            expected_partition="training",
        )

    report = _report()
    policy = report["training_supervision_policy"]
    assert isinstance(policy, dict)
    policy["minimum_observed_fraction_hex"] = (0.1).hex()
    report["training_supervision_policy_sha256"] = token_supervision_policy_sha256(policy)
    changed_sha256 = _write_report(path, report)
    with pytest.raises(ValueError, match="supervision policy"):
        load_fixed_observation_audit(
            path,
            expected_file_sha256=changed_sha256,
            expected_partition="training",
        )

    report = _report()
    groups = report["groups"]
    assert isinstance(groups, list)
    variants = groups[0]["variants"]
    assert isinstance(variants, list)
    support = variants[0]["support"]
    assert isinstance(support, dict)
    support["strict_object_winner_token_count"] = 0
    changed_sha256 = _write_report(path, report)
    with pytest.raises(ValueError, match="token support derivations"):
        load_fixed_observation_audit(
            path,
            expected_file_sha256=changed_sha256,
            expected_partition="training",
        )

    report = _report()
    groups = report["groups"]
    assert isinstance(groups, list)
    variants = groups[0]["variants"]
    assert isinstance(variants, list)
    variants[0]["proof"] = ""
    changed_sha256 = _write_report(path, report)
    with pytest.raises(ValueError, match="applicability proof"):
        load_fixed_observation_audit(
            path,
            expected_file_sha256=changed_sha256,
            expected_partition="training",
        )

    report = _report()
    groups = report["groups"]
    assert isinstance(groups, list)
    variants = groups[0]["variants"]
    assert isinstance(variants, list)
    variants[0]["task_key"] = "move_slider_right"
    changed_sha256 = _write_report(path, report)
    with pytest.raises(ValueError, match="pinned target protocol"):
        load_fixed_observation_audit(
            path,
            expected_file_sha256=changed_sha256,
            expected_partition="training",
        )

    report = _report()
    groups = report["groups"]
    assert isinstance(groups, list)
    variants = groups[0]["variants"]
    assert isinstance(variants, list)
    for variant in variants:
        support = variant["support"]
        assert isinstance(support, dict)
        views = support["views"]
        assert isinstance(views, list)
        views.append(
            {
                "camera_name": "gripper",
                "maximum_target_probability": 0.0,
                "measurable": False,
                "merged_grid_hw": [8, 8],
                "object_row_addressable": False,
                "positive_token_count": 0,
                "strict_categorical_winner_token_count": 0,
                "strict_object_winner_token_count": 0,
                "target_mass": 0.0,
            }
        )
    changed_sha256 = _write_report(path, report)
    assert load_fixed_observation_audit(
        path,
        expected_file_sha256=changed_sha256,
        expected_partition="training",
    ).task_keys == ("move_slider_left", "turn_on_led")


def test_audit_loader_accepts_content_audited_rejected_group_population(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit-with-rejected-group.json"
    report = _report_with_one_rejected_group()
    file_sha256 = _write_report(path, report)

    audit = load_fixed_observation_audit(
        path,
        expected_file_sha256=file_sha256,
        expected_partition="training",
    )

    assert len(audit.groups) == 1
    assert audit.groups[0].source_global_index == 10

    report["rejected_groups"] = []
    report["rejected_visual_artifacts"] = []
    report["source_group_count"] = 1
    changed_sha256 = _write_report(path, report)
    with pytest.raises(ValueError, match="summary differs"):
        load_fixed_observation_audit(
            path,
            expected_file_sha256=changed_sha256,
            expected_partition="training",
        )


def test_audit_loader_rejects_stranded_unique_addressable_semantics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit-with-stranded-semantics.json"
    report = _report_with_one_rejected_group()
    rejected_groups = report["rejected_groups"]
    first_variant = rejected_groups[0]["variants"][0]
    first_variant.update(
        {
            "instruction": "open the drawer",
            "instruction_sha256": hashlib.sha256(b"open the drawer").hexdigest(),
            "proof": "proof:open_drawer",
            "target_identity_key": "part/table/drawer_link",
            "task_key": "open_drawer",
        }
    )
    support = first_variant["support"]
    support["identity_key"] = "part/table/drawer_link"
    rejected_groups[0]["retained_task_keys"] = ["open_drawer"]
    rejected_groups[0]["retained_target_identity_keys"] = ["part/table/drawer_link"]
    summary = report["summary"]
    summary["addressable_task_histogram"] = {
        "move_slider_left": 1,
        "open_drawer": 1,
        "turn_on_led": 1,
    }
    summary["addressable_target_histogram"] = {
        "part/table/button_link": 1,
        "part/table/drawer_link": 1,
        "part/table/slide_link": 1,
    }
    file_sha256 = _write_report(path, report)

    with pytest.raises(ValueError, match="lose addressable task or target coverage"):
        load_fixed_observation_audit(
            path,
            expected_file_sha256=file_sha256,
            expected_partition="training",
        )


def test_audit_loader_replays_legacy_v2_report(tmp_path: Path) -> None:
    path = tmp_path / "legacy-v2-audit.json"
    report = _report()
    report["schema"] = CALVIN_FIXED_OBSERVATION_AUDIT_SCHEMA_V2
    for field in (
        "rejected_groups",
        "rejected_visual_artifacts",
        "source_group_count",
    ):
        del report[field]
    summary = report["summary"]
    for field in (
        "addressable_target_histogram",
        "addressable_task_histogram",
        "addressable_variant_count",
        "stranded_addressable_variant_count",
    ):
        del summary[field]
    file_sha256 = _write_report(path, report)

    audit = load_fixed_observation_audit(
        path,
        expected_file_sha256=file_sha256,
        expected_partition="training",
    )

    assert len(audit.groups) == 1
    assert audit.groups[0].source_global_index == 10


def test_native_vl_audit_retains_measurable_nonwinning_targets(tmp_path: Path) -> None:
    report = _report()
    groups = report["groups"]
    assert isinstance(groups, list)
    variants = groups[0]["variants"]
    assert isinstance(variants, list)
    occluded = _variant_record("open_drawer", "part/table/drawer_link", 0.25)
    occluded["fixed_x_diagnostic_eligible"] = False
    support = occluded["support"]
    assert isinstance(support, dict)
    support["object_row_addressable"] = False
    support["strict_categorical_winner_token_count"] = 0
    support["strict_object_winner_token_count"] = 0
    views = support["views"]
    assert isinstance(views, list)
    views[0]["object_row_addressable"] = False
    views[0]["strict_categorical_winner_token_count"] = 0
    views[0]["strict_object_winner_token_count"] = 0
    variants.append(occluded)
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["dropped_variant_count"] = 1
    summary["source_variant_count"] = 3
    path = tmp_path / "audit-with-nonwinning-target.json"
    file_sha256 = _write_report(path, report)

    fixed = load_fixed_observation_audit(
        path,
        expected_file_sha256=file_sha256,
        expected_partition="training",
    )
    native = load_native_vl_grounding_audit(
        path,
        expected_file_sha256=file_sha256,
        expected_partition="training",
    )

    assert fixed.task_keys == ("move_slider_left", "turn_on_led")
    assert native.task_keys == ("move_slider_left", "open_drawer", "turn_on_led")
    assert native.object_row_addressable_variant_count == 2
    assert native.measurable_variant_count == native.source_variant_count == 3
    assert native.groups[0].visible_camera_names == (("static",), ("static",), ("static",))


def test_pair_plan_is_deterministic_content_addressed_and_source_disjoint(
    tmp_path: Path,
) -> None:
    dataset, stream, audit, plan = _pair_fixture(tmp_path)
    replay = build_fixed_observation_pair_plan(stream, dataset, audit)

    assert replay == plan
    assert replay.artifact_sha256 == plan.artifact_sha256
    assert len(plan.pairs) == stream.reset_step_count
    assert plan.available_task_keys == ("task_alpha", "task_beta")
    assert plan.available_target_identity_keys == ("physical/alpha", "physical/beta")
    selected_sources = {item.group.source_global_index for item in plan.pairs}
    causal_sources = {
        dataset.source_global_index_by_key(transition.sample.sample_key)
        for step in range(stream.total_steps)
        if stream.component_for_step(step) == "causal"
        for transition in stream.global_batch(step).transitions
    }
    assert selected_sources.isdisjoint(causal_sources)

    path = tmp_path / "pair-plan.json"
    plan.write(path)
    assert FixedObservationPairPlan.load(path) == plan
    value = json.loads(path.read_text(encoding="ascii"))
    value["pairs"][0]["variants"][0]["task_key"] = "tampered"
    with pytest.raises(ValueError):
        FixedObservationPairPlan.from_dict(value)
    with pytest.raises(ValueError, match="signed range"):
        replace(plan.pairs[0], augmentation_seed=2**63)


def test_pair_overlay_changes_only_true_prompt_and_loss_side_target(
    tmp_path: Path,
) -> None:
    dataset, stream, _audit, plan = _pair_fixture(tmp_path)
    reset_step = next(
        step for step in range(stream.total_steps) if stream.component_for_step(step) == "reset"
    )
    natural = tuple(
        build_planned_native_calvin_batch(
            stream,
            dataset,
            optimizer_step=reset_step,
            rank=rank,
            world_size=2,
            gradient_accumulation_steps=1,
            accumulation_index=0,
        )
        for rank in range(2)
    )
    paired = tuple(apply_fixed_observation_pair(item, plan, dataset) for item in natural)

    assert paired[0].training.routing.sample_keys == paired[1].training.routing.sample_keys
    assert paired[0].augmentation_seeds == paired[1].augmentation_seeds
    assert paired[0].flow_noise_seeds == paired[1].flow_noise_seeds
    assert paired[0].flow_timestep_seeds == paired[1].flow_timestep_seeds
    assert paired[0].training.host_items[0]["task"] != paired[1].training.host_items[0]["task"]
    assert (
        paired[0].training.structural_target_requests[0].task_key
        != paired[1].training.structural_target_requests[0].task_key
    )
    for name, first in paired[0].training.host_items[0].items():
        if name == "task":
            continue
        second = paired[1].training.host_items[0][name]
        assert isinstance(first, torch.Tensor)
        torch.testing.assert_close(first, second)
    assert all(
        changed.source_digest != original.source_digest
        for changed, original in zip(paired, natural, strict=True)
    )
    assert all(item.fixed_observation_pair_sha256 == plan.artifact_sha256 for item in paired)
    with pytest.raises(ValueError, match="only once"):
        apply_fixed_observation_pair(paired[0], plan, dataset)
    with pytest.raises(ValueError, match="another prompt intervention"):
        apply_fixed_observation_pair(
            replace(natural[0], task_intervention_sha256="a" * 64),
            plan,
            dataset,
        )
    with pytest.raises(ValueError, match="continuation is forbidden"):
        build_native_calvin_continuation_batch(paired[0], dataset, offset=1)

    causal_step = next(
        step for step in range(stream.total_steps) if stream.component_for_step(step) == "causal"
    )
    causal = build_planned_native_calvin_batch(
        stream,
        dataset,
        optimizer_step=causal_step,
        rank=0,
        world_size=2,
        gradient_accumulation_steps=1,
        accumulation_index=0,
    )
    assert apply_fixed_observation_pair(causal, plan, dataset) is causal


def test_pair_overlay_preserves_native_action_horizon(tmp_path: Path) -> None:
    dataset, stream, _audit, plan = _pair_fixture(tmp_path)
    native_horizon_dataset = CalvinStatefulTransitionDataset(
        dataset.index,
        action_horizon=50,
    )
    reset_step = plan.pairs[0].optimizer_step

    natural = build_planned_native_calvin_batch(
        stream,
        native_horizon_dataset,
        optimizer_step=reset_step,
        rank=0,
        world_size=2,
        gradient_accumulation_steps=1,
        accumulation_index=0,
    )
    paired = apply_fixed_observation_pair(natural, plan, native_horizon_dataset)

    assert natural.training.host_items[0]["action.lingbot"].shape == (50, 55)
    assert paired.training.host_items[0]["action.lingbot"].shape == (50, 55)
    assert paired.training.host_items[0]["action.lingbot_is_pad"].shape == (50,)


def test_pair_forward_inputs_expose_only_shared_sensors_and_full_prompt(
    tmp_path: Path,
) -> None:
    dataset, stream, _audit, plan = _pair_fixture(tmp_path)
    reset_step = plan.pairs[0].optimizer_step
    batches = []
    for rank in range(2):
        natural = build_planned_native_calvin_batch(
            stream,
            dataset,
            optimizer_step=reset_step,
            rank=rank,
            world_size=2,
            gradient_accumulation_steps=1,
            accumulation_index=0,
        )
        paired = apply_fixed_observation_pair(natural, plan, dataset)
        batches.append(
            collate_native_calvin_training_batch(
                paired.training,
                feature_transform=_PromptAwareTransform(),
                collator=_collator,
                augmentation_seeds=paired.augmentation_seeds,
                source_digest=paired.source_digest,
            )
        )

    for batch in batches:
        audit_native_calvin_model_inputs(batch.model_inputs)
        assert not set(batch.model_inputs) & {
            "identity",
            "owner",
            "proof",
            "sample_key",
            "simulator_state",
            "target",
            "task_key",
        }
    assert not torch.equal(
        batches[0].model_inputs["lang_tokens"],
        batches[1].model_inputs["lang_tokens"],
    )
    for name in set(batches[0].model_inputs) - {"lang_tokens"}:
        torch.testing.assert_close(
            batches[0].model_inputs[name],
            batches[1].model_inputs[name],
        )

    fingerprints = tuple(fixed_observation_training_pair_fingerprint(batch) for batch in batches)
    validate_fixed_observation_training_pair_fingerprints(fingerprints)

    changed_nonlanguage_inputs = dict(batches[1].model_inputs)
    changed_nonlanguage_inputs["images"] = changed_nonlanguage_inputs["images"] + 1
    changed_nonlanguage_batch = replace(
        batches[1],
        model_inputs=changed_nonlanguage_inputs,
    )
    with pytest.raises(ValueError, match="non-language contracts"):
        validate_fixed_observation_training_pair_fingerprints(
            (
                fingerprints[0],
                fixed_observation_training_pair_fingerprint(changed_nonlanguage_batch),
            )
        )

    changed_controls = dict(fingerprints[1])
    changed_controls["controls_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="controls_sha256"):
        validate_fixed_observation_training_pair_fingerprints((fingerprints[0], changed_controls))

    same_task = dict(fingerprints[1])
    same_task["task_keys"] = fingerprints[0]["task_keys"]
    with pytest.raises(ValueError, match="same loss-side task"):
        validate_fixed_observation_training_pair_fingerprints((fingerprints[0], same_task))

    same_language = dict(fingerprints[1])
    same_language["language_tokens_sha256"] = fingerprints[0]["language_tokens_sha256"]
    with pytest.raises(ValueError, match="same tokenized language"):
        validate_fixed_observation_training_pair_fingerprints((fingerprints[0], same_language))

    fixed_metadata = tuple(
        {
            "fixed_observation_fingerprint": fingerprint,
            "fixed_observation_pair_sha256": "a" * 64,
        }
        for fingerprint in fingerprints
    )
    assert validate_fixed_observation_training_rank_metadata(fixed_metadata) is True
    causal_metadata = (
        {
            "fixed_observation_fingerprint": None,
            "fixed_observation_pair_sha256": None,
        },
        {
            "fixed_observation_fingerprint": None,
            "fixed_observation_pair_sha256": None,
        },
    )
    assert validate_fixed_observation_training_rank_metadata(causal_metadata) is False

    mismatched_activation = dict(fixed_metadata[1])
    mismatched_activation["fixed_observation_pair_sha256"] = None
    mismatched_activation["fixed_observation_fingerprint"] = None
    with pytest.raises(ValueError, match="activation differs"):
        validate_fixed_observation_training_rank_metadata(
            (fixed_metadata[0], mismatched_activation)
        )

    causal_with_evidence = dict(causal_metadata[0])
    causal_with_evidence["fixed_observation_fingerprint"] = fingerprints[0]
    with pytest.raises(ValueError, match="causal training step"):
        validate_fixed_observation_training_rank_metadata(
            (causal_with_evidence, causal_metadata[1])
        )


def test_pair_builder_rejects_source_state_drift(tmp_path: Path) -> None:
    dataset, stream, audit, _plan = _pair_fixture(tmp_path)
    changed_group = replace(audit.groups[0], source_state_sha256="f" * 64)
    changed = replace(audit, groups=(changed_group, *audit.groups[1:]))
    with pytest.raises(ValueError, match="source-state hash"):
        build_fixed_observation_pair_plan(stream, dataset, changed)

    changed_group = replace(audit.groups[0], source_instruction_sha256="f" * 64)
    changed = replace(audit, groups=(changed_group, *audit.groups[1:]))
    with pytest.raises(ValueError, match="immutable stateful sample"):
        build_fixed_observation_pair_plan(stream, dataset, changed)

    sensors = dict(audit.groups[0].source_sensor_sha256)
    sensors["rgb_static"] = "f" * 64
    changed_group = replace(
        audit.groups[0],
        source_sensor_sha256=tuple(sorted(sensors.items())),
    )
    changed = replace(audit, groups=(changed_group, *audit.groups[1:]))
    with pytest.raises(ValueError, match="sensor hashes"):
        build_fixed_observation_pair_plan(stream, dataset, changed)


def test_evaluation_plan_is_source_disjoint_balanced_and_tamper_evident(
    tmp_path: Path,
) -> None:
    _dataset, _stream, training, pair_plan = _pair_fixture(tmp_path)
    validation = _evaluation_audit(
        training,
        partition="validation",
        source_offset=1_000,
    )
    heldout = _evaluation_audit(
        training,
        partition="heldout",
        source_offset=2_000,
    )

    plan = build_fixed_observation_evaluation_plan(
        training_audit=training,
        validation_audit=validation,
        heldout_audit=heldout,
        training_pair_plan=pair_plan,
    )
    replay = build_fixed_observation_evaluation_plan(
        training_audit=training,
        validation_audit=validation,
        heldout_audit=heldout,
        training_pair_plan=pair_plan,
    )

    assert replay == plan
    assert replay.artifact_sha256 == plan.artifact_sha256
    assert all(0 <= item.replay_seed < 2**63 for item in plan.items)
    for partition in ("validation", "heldout"):
        rank_counts = [len(plan.items_for(partition, rank)) for rank in range(plan.world_size)]
        assert sum(rank_counts) == len(validation.groups)
        assert max(rank_counts) - min(rank_counts) <= 1
        assert set(plan.task_histogram[partition]) == {
            "task_alpha",
            "task_beta",
        }
        assert set(plan.target_histogram[partition]) == {
            "physical/alpha",
            "physical/beta",
        }
    source_episodes = {
        partition: {
            item.group.source_episode_index for item in plan.items if item.partition == partition
        }
        for partition in ("validation", "heldout")
    }
    training_sources = {group.source_episode_index for group in training.groups}
    assert training_sources.isdisjoint(source_episodes["validation"])
    assert training_sources.isdisjoint(source_episodes["heldout"])
    assert source_episodes["validation"].isdisjoint(source_episodes["heldout"])

    path = tmp_path / "evaluation-plan.json"
    plan.write(path)
    assert FixedObservationEvaluationPlan.load(path) == plan
    value = json.loads(path.read_text(encoding="ascii"))
    value["items"][0]["replay_seed"] += 1
    with pytest.raises(ValueError, match="artifact SHA-256"):
        FixedObservationEvaluationPlan.from_dict(value)
    with pytest.raises(ValueError, match="signed range"):
        replace(plan.items[0], replay_seed=2**63)

    odd_validation = replace(validation, groups=validation.groups[:3])
    odd_heldout = replace(heldout, groups=heldout.groups[:3])
    with pytest.raises(ValueError, match="combined rank load must be equal"):
        build_fixed_observation_evaluation_plan(
            training_audit=training,
            validation_audit=odd_validation,
            heldout_audit=odd_heldout,
            training_pair_plan=pair_plan,
        )

    changed_manifest_validation = replace(
        validation,
        dataset_manifest_file_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="different source splits"):
        build_fixed_observation_evaluation_plan(
            training_audit=training,
            validation_audit=changed_manifest_validation,
            heldout_audit=heldout,
            training_pair_plan=pair_plan,
        )

    overlapping_group = replace(
        validation.groups[0],
        source_episode_index=training.groups[0].source_episode_index,
    )
    overlapping_validation = replace(
        validation,
        groups=(overlapping_group, *validation.groups[1:]),
    )
    with pytest.raises(ValueError, match="share source episodes"):
        build_fixed_observation_evaluation_plan(
            training_audit=training,
            validation_audit=overlapping_validation,
            heldout_audit=heldout,
            training_pair_plan=pair_plan,
        )


def test_fixed_observation_runtime_pair_changes_only_language_and_loss_target(
    tmp_path: Path,
) -> None:
    dataset, _stream, audit, _pair_plan = _pair_fixture(tmp_path)
    group = audit.groups[0]
    item = FixedObservationEvaluationItem(
        partition="validation",
        ordinal=0,
        rank=0,
        group=group,
        variants=(group.variants[0], group.variants[1]),
        replay_seed=313,
    )

    def collate(planned):
        return collate_native_calvin_training_batch(
            planned.training,
            feature_transform=_PromptAwareTransform(),
            collator=_collator,
            augmentation_seeds=planned.augmentation_seeds,
            source_digest=planned.source_digest,
        )

    planned, batches, capacity_seed, non_language_sha256, language_sha256 = (
        _prepare_fixed_observation_evaluation_pair(
            item=item,
            checkpoint_global_step=7,
            rank=0,
            dataset=dataset,
            device=torch.device("cpu"),
            collate_planned=collate,
        )
    )

    assert planned[0].source_digest == planned[1].source_digest
    assert batches[0].source_digest == batches[1].source_digest
    assert (
        batches[0].structural_target_requests[0].task_key,
        batches[1].structural_target_requests[0].task_key,
    ) == ("task_alpha", "task_beta")
    assert non_language_sha256
    assert len(set(language_sha256)) == 2
    assert capacity_seed >= 0
    for name in set(batches[0].model_inputs) - {"lang_tokens", "lang_masks"}:
        torch.testing.assert_close(
            batches[0].model_inputs[name],
            batches[1].model_inputs[name],
        )


def test_fixed_observation_runtime_pair_rejects_prompt_dependent_nonlanguage_input(
    tmp_path: Path,
) -> None:
    dataset, _stream, audit, _pair_plan = _pair_fixture(tmp_path)
    group = audit.groups[0]
    item = FixedObservationEvaluationItem(
        partition="validation",
        ordinal=0,
        rank=0,
        group=group,
        variants=(group.variants[0], group.variants[1]),
        replay_seed=313,
    )

    def collate(planned):
        batch = collate_native_calvin_training_batch(
            planned.training,
            feature_transform=_PromptAwareTransform(),
            collator=_collator,
            augmentation_seeds=planned.augmentation_seeds,
            source_digest=planned.source_digest,
        )
        if planned.training.structural_target_requests[0].task_key == "task_beta":
            model_inputs = dict(batch.model_inputs)
            model_inputs["images"] = model_inputs["images"] + 1
            batch = replace(batch, model_inputs=model_inputs)
        return batch

    with pytest.raises(ValueError, match="changed model input 'images'"):
        _prepare_fixed_observation_evaluation_pair(
            item=item,
            checkpoint_global_step=7,
            rank=0,
            dataset=dataset,
            device=torch.device("cpu"),
            collate_planned=collate,
        )


def test_forward_equivalence_probe_is_content_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    _dataset, _stream, training, pair_plan = _pair_fixture(tmp_path)
    plan = build_fixed_observation_evaluation_plan(
        training_audit=training,
        validation_audit=_evaluation_audit(
            training,
            partition="validation",
            source_offset=1_000,
        ),
        heldout_audit=_evaluation_audit(
            training,
            partition="heldout",
            source_offset=2_000,
        ),
        training_pair_plan=pair_plan,
    )
    item = plan.items_for("validation", 0)[0]
    probe = build_fixed_observation_forward_equivalence_probe(
        plan=plan,
        item=item,
        checkpoint_global_step=7,
        model_inputs_sha256="a" * 64,
        relation_sha256="b" * 64,
        repeated_relation_sha256="b" * 64,
        repeat_forward_seconds=0.25,
    )

    assert probe["same_model_inputs"] is True
    assert probe["same_forward_seed"] is True
    assert probe["same_previous_state"] is True
    assert probe["relation_outputs_equal"] is True

    with pytest.raises(ValueError, match="not reproducible"):
        build_fixed_observation_forward_equivalence_probe(
            plan=plan,
            item=item,
            checkpoint_global_step=7,
            model_inputs_sha256="a" * 64,
            relation_sha256="b" * 64,
            repeated_relation_sha256="c" * 64,
            repeat_forward_seconds=0.25,
        )


def test_prompt_switch_metrics_recompute_bidirectional_diagonal_advantage() -> None:
    first = FixedObservationVariant(
        task_key="task_alpha",
        instruction="perform the complete task alpha instruction",
        instruction_sha256=hashlib.sha256(
            b"perform the complete task alpha instruction"
        ).hexdigest(),
        target_identity_key="physical/alpha",
        target_mass=1.0,
    )
    second = FixedObservationVariant(
        task_key="task_beta",
        instruction="perform the complete task beta instruction",
        instruction_sha256=hashlib.sha256(
            b"perform the complete task beta instruction"
        ).hexdigest(),
        target_identity_key="physical/beta",
        target_mass=0.5,
    )
    results = (
        {
            "variant": first.as_dict(),
            "own_target_token_evidence": build_representation_token_evidence(
                (2.0, -1.0),
                (1.0, 0.0),
            ),
            "alternate_target_token_evidence": build_representation_token_evidence(
                (2.0, -1.0),
                (0.0, 1.0),
            ),
            "task_row_diagnostic": _task_diagnostic(
                target_index=0,
                logits=(2.0, -1.0),
            ),
            "relation_sha256": "a" * 64,
        },
        {
            "variant": second.as_dict(),
            "own_target_token_evidence": build_representation_token_evidence(
                (-1.0, 2.0),
                (0.0, 1.0),
            ),
            "alternate_target_token_evidence": build_representation_token_evidence(
                (-1.0, 2.0),
                (1.0, 0.0),
            ),
            "task_row_diagnostic": _task_diagnostic(
                target_index=1,
                logits=(-1.0, 2.0),
            ),
            "relation_sha256": "b" * 64,
        },
    )

    metrics = fixed_observation_prompt_switch_metrics(results)

    assert metrics["dense_variant_diagonal_advantages"] == [6.0, 6.0]
    assert metrics["dense_mean_diagonal_advantage"] == 6.0
    assert metrics["dense_bidirectional_positive"] is True
    assert metrics["fractional_auc_variant_diagonal_advantages"] == [1.0, 1.0]
    assert metrics["fractional_auc_bidirectional_positive"] is True
    assert metrics["row_variant_diagonal_advantages"] == [3.0, 3.0]
    assert metrics["row_mean_diagonal_advantage"] == 3.0
    assert metrics["row_bidirectional_positive"] is True
    assert metrics["relation_output_changed"] is True


def test_mass_strata_are_frozen_from_plan_not_checkpoint_outputs(
    tmp_path: Path,
) -> None:
    _dataset, _stream, training, pair_plan = _pair_fixture(tmp_path)
    plan = build_fixed_observation_evaluation_plan(
        training_audit=training,
        validation_audit=_evaluation_audit(
            training,
            partition="validation",
            source_offset=1_000,
        ),
        heldout_audit=_evaluation_audit(
            training,
            partition="heldout",
            source_offset=2_000,
        ),
        training_pair_plan=pair_plan,
    )

    strata = fixed_observation_evaluation_mass_strata(plan)

    assert set(strata) == {(item.partition, item.ordinal) for item in plan.items}
    assert set(strata.values()) <= {"lower_third", "middle_third", "upper_third"}
    for partition in ("validation", "heldout"):
        counts = {
            name: sum(
                value == name
                for (item_partition, _ordinal), value in strata.items()
                if item_partition == partition
            )
            for name in ("lower_third", "middle_third", "upper_third")
        }
        assert max(counts.values()) - min(counts.values()) <= 1


def _fixed_x_relation_sha256(
    item: FixedObservationEvaluationItem,
    *,
    checkpoint_global_step: int,
    variant_index: int,
) -> str:
    return _canonical_sha256(
        {
            "checkpoint_global_step": checkpoint_global_step,
            "ordinal": item.ordinal,
            "partition": item.partition,
            "variant_index": variant_index,
        }
    )


def _fixed_x_variant_result(
    item: FixedObservationEvaluationItem,
    *,
    checkpoint_global_step: int,
    variant_index: int,
    prompt_target_scale: float,
    target_tag: str = "fixed",
) -> dict[str, object]:
    variant = item.variants[variant_index]
    own_mass = (1.0, 0.0) if variant_index == 0 else (0.0, 1.0)
    alternate_mass = (0.0, 1.0) if variant_index == 0 else (1.0, 0.0)
    logits = (prompt_target_scale, 0.0) if variant_index == 0 else (0.0, prompt_target_scale)
    ownership_prediction = (0.8, 0.2) if variant_index == 0 else (0.2, 0.8)
    ownership = build_representation_ownership_row(
        row_index=variant_index,
        track_index=variant_index,
        identity_key=variant.target_identity_key,
        is_task_target=True,
        prediction=ownership_prediction,
        target=own_mass,
        weight=(1.0, 1.0),
    )
    return {
        "variant": variant.as_dict(),
        "own_target_token_evidence": build_representation_token_evidence(
            logits,
            own_mass,
        ),
        "alternate_target_token_evidence": build_representation_token_evidence(
            logits,
            alternate_mass,
        ),
        "task_row_diagnostic": _task_diagnostic(
            target_index=variant_index,
            logits=logits,
        ),
        "ownership_rows": [ownership],
        "ownership_summary": summarize_representation_ownership_rows([ownership]),
        "relation_sha256": _fixed_x_relation_sha256(
            item,
            checkpoint_global_step=checkpoint_global_step,
            variant_index=variant_index,
        ),
        "target_sha256": _canonical_sha256(
            {
                "identity_key": variant.target_identity_key,
                "mass": list(own_mass),
                "target_tag": target_tag,
            }
        ),
        "forward_seconds": 0.1,
        "instruction_sha256": variant.instruction_sha256,
        "visual_artifact": {
            "schema": NATIVE_VISUAL_AUDIT_SCHEMA,
            "bytes": 10,
            "global_step": checkpoint_global_step,
            "input_weight_global_step": checkpoint_global_step,
            "loss_only_labels_visible_to_model": False,
            "path": (
                f"{item.partition}/step_{checkpoint_global_step:07d}/"
                f"item_{item.ordinal:05d}/variant_{variant_index}.png"
            ),
            "rank": item.rank,
            "sample_key": item.group.stateful_sample_key,
            "sha256": _canonical_sha256(
                {
                    "checkpoint_global_step": checkpoint_global_step,
                    "ordinal": item.ordinal,
                    "partition": item.partition,
                    "variant_index": variant_index,
                    "visual": True,
                }
            ),
            "task": variant.instruction,
            "weight_boundary": "fixed_observation_checkpoint_evaluation",
        },
    }


def _fixed_x_snapshot(
    plan: FixedObservationEvaluationPlan,
    *,
    checkpoint_global_step: int,
    prompt_target_scale: float,
    prompt_target_scale_by_task: Mapping[str, float] | None = None,
    action_state_sha256: str = "d" * 64,
    model_input_tag: str = "fixed",
    target_tag: str = "fixed",
) -> dict[str, object]:
    strata = fixed_observation_evaluation_mass_strata(plan)
    samples = []
    for item in plan.items:
        samples.append(
            build_fixed_observation_evaluation_sample(
                checkpoint_global_step=checkpoint_global_step,
                item=item,
                mass_stratum=strata[(item.partition, item.ordinal)],
                variant_results=tuple(
                    _fixed_x_variant_result(
                        item,
                        checkpoint_global_step=checkpoint_global_step,
                        variant_index=index,
                        prompt_target_scale=(
                            prompt_target_scale
                            if prompt_target_scale_by_task is None
                            else prompt_target_scale_by_task[item.variants[index].task_key]
                        ),
                        target_tag=target_tag,
                    )
                    for index in range(2)
                ),
                source_digest=_canonical_sha256(
                    {
                        "ordinal": item.ordinal,
                        "partition": item.partition,
                        "source": True,
                    }
                ),
                non_language_model_inputs_sha256=_canonical_sha256(
                    {
                        "model_input_tag": model_input_tag,
                        "ordinal": item.ordinal,
                        "partition": item.partition,
                        "non_language": True,
                    }
                ),
                language_model_inputs_sha256=tuple(
                    _canonical_sha256(
                        {
                            "instruction_sha256": variant.instruction_sha256,
                            "language": True,
                            "model_input_tag": model_input_tag,
                        }
                    )
                    for variant in item.variants
                ),
                peak_cuda_reserved_bytes=1,
            )
        )
    probes = []
    for rank in range(plan.world_size):
        item = (plan.items_for("validation", rank) + plan.items_for("heldout", rank))[0]
        relation_sha256 = _fixed_x_relation_sha256(
            item,
            checkpoint_global_step=checkpoint_global_step,
            variant_index=0,
        )
        probes.append(
            build_fixed_observation_forward_equivalence_probe(
                plan=plan,
                item=item,
                checkpoint_global_step=checkpoint_global_step,
                model_inputs_sha256=_canonical_sha256(
                    {
                        "checkpoint_global_step": checkpoint_global_step,
                        "rank": rank,
                        "repeat": True,
                    }
                ),
                relation_sha256=relation_sha256,
                repeated_relation_sha256=relation_sha256,
                repeat_forward_seconds=0.1,
            )
        )
    return build_fixed_observation_evaluation_snapshot(
        checkpoint_global_step=checkpoint_global_step,
        implementation_sha256="a" * 64,
        model_family_sha256="b" * 64,
        representation_split_sha256="c" * 64,
        plan=plan,
        representation_frozen_action_state_sha256=action_state_sha256,
        samples=samples,
        forward_equivalence_probes=probes,
    )


def test_fixed_observation_numeric_gate_is_recomputed_fail_closed_and_atomic(
    tmp_path: Path,
) -> None:
    _dataset, _stream, training, pair_plan = _pair_fixture(
        tmp_path,
        total_steps=12,
    )
    plan = build_fixed_observation_evaluation_plan(
        training_audit=training,
        validation_audit=_evaluation_audit(
            training,
            partition="validation",
            source_offset=1_000,
        ),
        heldout_audit=_evaluation_audit(
            training,
            partition="heldout",
            source_offset=2_000,
        ),
        training_pair_plan=pair_plan,
    )
    assert all(
        len(tuple(item for item in plan.items if item.partition == partition)) >= 6
        for partition in ("validation", "heldout")
    )
    baseline = _fixed_x_snapshot(
        plan,
        checkpoint_global_step=0,
        prompt_target_scale=-1.0,
    )
    decision = _fixed_x_snapshot(
        plan,
        checkpoint_global_step=200,
        prompt_target_scale=2.0,
    )
    assert isinstance(baseline["samples"], list)
    assert isinstance(baseline["forward_equivalence_probes"], list)
    assert isinstance(decision["samples"], list)
    assert isinstance(decision["forward_equivalence_probes"], list)
    with pytest.raises(ValueError, match="sample checkpoint differs from its snapshot"):
        build_fixed_observation_evaluation_snapshot(
            checkpoint_global_step=200,
            implementation_sha256="a" * 64,
            model_family_sha256="b" * 64,
            representation_split_sha256="c" * 64,
            plan=plan,
            representation_frozen_action_state_sha256="d" * 64,
            samples=baseline["samples"],
            forward_equivalence_probes=decision["forward_equivalence_probes"],
        )
    with pytest.raises(ValueError, match="repeat probe checkpoint differs from its snapshot"):
        build_fixed_observation_evaluation_snapshot(
            checkpoint_global_step=200,
            implementation_sha256="a" * 64,
            model_family_sha256="b" * 64,
            representation_split_sha256="c" * 64,
            plan=plan,
            representation_frozen_action_state_sha256="d" * 64,
            samples=decision["samples"],
            forward_equivalence_probes=baseline["forward_equivalence_probes"],
        )

    gate = build_fixed_observation_numeric_gate(
        baseline,
        decision,
        plan=plan,
    )

    assert gate["status"] == "PASS_PENDING_STANDARD_GATES_AND_VISUAL_REVIEW"
    assert gate["authorizes_action_or_long_training"] is False
    assert gate["visual_review_required"] is True
    for partition in ("validation", "heldout"):
        result = gate["partitions"][partition]
        assert result["status"] == "PASS"
        assert set(result["mass_strata"]) == {
            "lower_third",
            "middle_third",
            "upper_third",
        }
        assert result["source_episode_count"] >= 6
        assert result["source_episode_exact_sign_tests"]["row"]["one_sided_pvalue"] <= 0.05
        assert (
            result["source_episode_improvement_exact_sign_tests"]["row"]["one_sided_pvalue"] <= 0.05
        )

    failed_decision = _fixed_x_snapshot(
        plan,
        checkpoint_global_step=200,
        prompt_target_scale=-0.5,
    )
    failed_gate = build_fixed_observation_numeric_gate(
        baseline,
        failed_decision,
        plan=plan,
    )
    assert failed_gate["status"] == "FAIL"

    changed_action = _fixed_x_snapshot(
        plan,
        checkpoint_global_step=200,
        prompt_target_scale=2.0,
        action_state_sha256="e" * 64,
    )
    changed_action_gate = build_fixed_observation_numeric_gate(
        baseline,
        changed_action,
        plan=plan,
    )
    assert changed_action_gate["status"] == "FAIL"
    assert changed_action_gate["invariant_checks"]["frozen_action_state_unchanged"] is False

    wrong_step = _fixed_x_snapshot(
        plan,
        checkpoint_global_step=199,
        prompt_target_scale=2.0,
    )
    with pytest.raises(ValueError, match="exactly checkpoints 0 and 200"):
        build_fixed_observation_numeric_gate(
            baseline,
            wrong_step,
            plan=plan,
        )

    changed_inputs = _fixed_x_snapshot(
        plan,
        checkpoint_global_step=200,
        prompt_target_scale=2.0,
        model_input_tag="changed",
    )
    with pytest.raises(ValueError, match="model inputs changed between checkpoints"):
        build_fixed_observation_numeric_gate(
            baseline,
            changed_inputs,
            plan=plan,
        )

    changed_targets = _fixed_x_snapshot(
        plan,
        checkpoint_global_step=200,
        prompt_target_scale=2.0,
        target_tag="changed",
    )
    with pytest.raises(ValueError, match="supervision targets changed between checkpoints"):
        build_fixed_observation_numeric_gate(
            baseline,
            changed_targets,
            plan=plan,
        )

    tampered = json.loads(json.dumps(gate))
    tampered["status"] = "FAIL"
    with pytest.raises(ValueError, match="not recomputed"):
        validate_fixed_observation_numeric_gate(
            tampered,
            baseline_snapshot=baseline,
            decision_snapshot=decision,
            plan=plan,
        )

    path = tmp_path / "fixed-X-numeric-gate.json"
    write_fixed_observation_numeric_gate(path, gate)
    assert json.loads(path.read_text(encoding="ascii")) == gate
    with pytest.raises(FileExistsError):
        write_fixed_observation_numeric_gate(path, gate)


def test_fixed_observation_numeric_gate_clusters_source_episodes(
    tmp_path: Path,
) -> None:
    _dataset, _stream, training, pair_plan = _pair_fixture(
        tmp_path,
        total_steps=12,
    )
    plan = build_fixed_observation_evaluation_plan(
        training_audit=training,
        validation_audit=_evaluation_audit(
            training,
            partition="validation",
            source_offset=1_000,
        ),
        heldout_audit=_evaluation_audit(
            training,
            partition="heldout",
            source_offset=2_000,
        ),
        training_pair_plan=pair_plan,
    )
    clustered_plan = replace(
        plan,
        items=tuple(
            replace(
                item,
                group=replace(
                    item.group,
                    source_episode_index=(
                        (1_000 if item.partition == "validation" else 2_000) + item.ordinal % 2
                    ),
                ),
            )
            for item in plan.items
        ),
    )
    baseline = _fixed_x_snapshot(
        clustered_plan,
        checkpoint_global_step=0,
        prompt_target_scale=-1.0,
    )
    decision = _fixed_x_snapshot(
        clustered_plan,
        checkpoint_global_step=200,
        prompt_target_scale=2.0,
    )

    gate = build_fixed_observation_numeric_gate(
        baseline,
        decision,
        plan=clustered_plan,
    )

    assert gate["status"] == "FAIL"
    for partition in ("validation", "heldout"):
        result = gate["partitions"][partition]
        assert result["sample_count"] >= 6
        assert result["source_episode_count"] == 2
        assert result["source_episode_exact_sign_tests"]["row"]["one_sided_pvalue"] == 0.25
        assert (
            result["source_episode_improvement_exact_sign_tests"]["row"]["one_sided_pvalue"] == 0.25
        )


def test_fixed_observation_numeric_gate_does_not_share_breadth_between_variants(
    tmp_path: Path,
) -> None:
    _dataset, _stream, training, pair_plan = _pair_fixture(
        tmp_path,
        total_steps=12,
    )
    plan = build_fixed_observation_evaluation_plan(
        training_audit=training,
        validation_audit=_evaluation_audit(
            training,
            partition="validation",
            source_offset=1_000,
        ),
        heldout_audit=_evaluation_audit(
            training,
            partition="heldout",
            source_offset=2_000,
        ),
        training_pair_plan=pair_plan,
    )
    baseline = _fixed_x_snapshot(
        plan,
        checkpoint_global_step=0,
        prompt_target_scale=-1.0,
    )
    decision = _fixed_x_snapshot(
        plan,
        checkpoint_global_step=200,
        prompt_target_scale=0.0,
        prompt_target_scale_by_task={
            "task_alpha": 3.0,
            "task_beta": -1.0,
        },
    )

    gate = build_fixed_observation_numeric_gate(
        baseline,
        decision,
        plan=plan,
    )

    assert gate["status"] == "FAIL"
    for partition in ("validation", "heldout"):
        result = gate["partitions"][partition]
        assert result["decision_mean_advantages"]["row"] > 0.0
        assert result["breadth"]["positive_task_count"] == 1
        assert result["breadth"]["task_count"] == 2
        assert result["checks"]["task_breadth"]["passed"] is False
