from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from picf_next.lingbot_native.fixed_observation import (
    FixedObservationAudit,
    FixedObservationGroup,
    FixedObservationPair,
    FixedObservationPairPlan,
    FixedObservationVariant,
    NativeVLGroundingAudit,
    NativeVLGroundingGroup,
)
from picf_next.lingbot_native.vl_curriculum import (
    NATIVE_VL_CURRICULUM_LATTICES,
    NativeVLGroundingCurriculumPlan,
    build_native_vl_grounding_curriculum,
)
from tools.build_lingbot_native_vl_curriculum import _positive_int


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _variant(group_index: int, variant_index: int) -> FixedObservationVariant:
    instruction = f"perform task {variant_index} for group {group_index}"
    return FixedObservationVariant(
        task_key=f"task_{variant_index}",
        instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        target_identity_key=f"object_{variant_index}",
        target_mass=float(variant_index + 1),
    )


def _group(group_index: int, variant_count: int) -> FixedObservationGroup:
    return FixedObservationGroup(
        scene=f"scene-{group_index}",
        source_global_index=100 + group_index,
        source_state_sha256=_sha(f"state-{group_index}"),
        source_sensor_sha256=tuple(
            (name, _sha(f"{name}-{group_index}"))
            for name in ("depth_gripper", "depth_static", "rgb_gripper", "rgb_static")
        ),
        source_episode_index=10 + group_index,
        source_task_key=f"source_task_{group_index}",
        source_instruction_sha256=_sha(f"source-instruction-{group_index}"),
        stateful_episode_key=f"episode-{group_index}",
        stateful_sample_key=f"sample-{group_index}",
        variants=tuple(_variant(group_index, index) for index in range(variant_count)),
    )


def _native_group(
    group: FixedObservationGroup,
    *camera_names: tuple[str, ...],
) -> NativeVLGroundingGroup:
    support = camera_names or tuple(("static",) for _variant in group.variants)
    return NativeVLGroundingGroup(group=group, visible_camera_names=support)


def _inputs() -> tuple[FixedObservationPairPlan, NativeVLGroundingAudit]:
    groups = (_group(0, 4), _group(1, 3))
    audit = FixedObservationAudit(
        partition="training",
        report_file_sha256=_sha("audit-file"),
        report_artifact_sha256=_sha("audit-artifact"),
        dataset_manifest_file_sha256=_sha("manifest-file"),
        dataset_tree_sha256=_sha("manifest-tree"),
        representation_split_file_sha256=_sha("split-file"),
        representation_split_artifact_sha256=_sha("split-artifact"),
        comparison_id="curriculum-test",
        stream_plan_sha256=_sha("stream-plan"),
        training_projection_contract_sha256=_sha("projection-contract"),
        training_projection_payload_sha256=_sha("projection-payload"),
        groups=groups,
    )
    pair_group = replace(groups[0], variants=groups[0].variants[:2])
    fixed_group = replace(groups[0], variants=groups[0].variants[:3])
    pair = FixedObservationPair(
        optimizer_step=0,
        lane_ids=("rank-0", "rank-1"),
        group=pair_group,
        variants=(pair_group.variants[0], pair_group.variants[1]),
        augmentation_seed=11,
        flow_noise_seed=12,
        flow_timestep_seed=13,
    )
    pair_plan = FixedObservationPairPlan(
        dataset_id="calvin",
        dataset_revision="test",
        dataset_manifest_sha256=audit.dataset_tree_sha256,
        comparison_id=audit.comparison_id,
        seed=20260801,
        stream_plan_sha256=audit.stream_plan_sha256,
        component_schedule_sha256=_sha("component-schedule"),
        audit_report_file_sha256=audit.report_file_sha256,
        audit_artifact_sha256=audit.report_artifact_sha256,
        representation_split_file_sha256=audit.representation_split_file_sha256,
        representation_split_artifact_sha256=audit.representation_split_artifact_sha256,
        training_projection_contract_sha256=audit.training_projection_contract_sha256,
        training_projection_payload_sha256=audit.training_projection_payload_sha256,
        candidate_group_count=len(groups),
        available_task_keys=tuple(sorted(variant.task_key for variant in pair.variants)),
        available_target_identity_keys=tuple(
            sorted(variant.target_identity_key for variant in pair.variants)
        ),
        pairs=(pair,),
    )
    return pair_plan, NativeVLGroundingAudit(
        fixed_x_audit=replace(audit, groups=(fixed_group, groups[1])),
        groups=tuple(_native_group(group) for group in groups),
        source_variant_count=7,
    )


def _canonical_artifact_sha256(value: dict[str, object]) -> str:
    content = {name: child for name, child in value.items() if name != "artifact_sha256"}
    payload = json.dumps(
        content,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_curriculum_builder_requires_positive_preregistered_counts() -> None:
    assert _positive_int(216, name="groups") == 216
    for invalid in (True, 0, -1, 1.5, "216"):
        with pytest.raises(ValueError, match="positive integer"):
            _positive_int(invalid, name="groups")


def test_curriculum_is_exhaustive_deterministic_and_rank_balanced() -> None:
    pair_plan, audit = _inputs()
    kwargs = {"pair_plan_file_sha256": _sha("pair-plan-file")}

    plan = build_native_vl_grounding_curriculum(pair_plan, audit, **kwargs)
    assert plan == build_native_vl_grounding_curriculum(pair_plan, audit, **kwargs)
    assert plan.groups == tuple(group.group for group in audit.groups)
    assert plan.source_variant_count == 7
    assert plan.object_row_addressable_variant_count == 6
    assert plan.visual_lattices == NATIVE_VL_CURRICULUM_LATTICES
    assert len(plan.steps) == 4
    assert Counter(batch.visual_lattice for step in plan.steps for batch in step.batches) == {
        8: 4,
        14: 4,
    }
    assert plan.rank_task_histograms[0] == plan.rank_task_histograms[1]
    assert plan.rank_target_histograms[0] == plan.rank_target_histograms[1]

    exposure: Counter[tuple[int, int, int]] = Counter()
    for step in plan.steps:
        group, batches = plan.resolve_step(step.optimizer_step)
        assert tuple(lattice for lattice, _camera, _variants in batches) == (
            NATIVE_VL_CURRICULUM_LATTICES
        )
        for batch, (lattice, camera, variants) in zip(step.batches, batches, strict=True):
            assert camera == "static"
            assert variants[0].task_key != variants[1].task_key
            assert variants[0].target_identity_key != variants[1].target_identity_key
            for variant_index in batch.variant_indices:
                exposure[(step.group_index, variant_index, lattice)] += 1
        assert group == plan.groups[step.group_index]

    for lattice in NATIVE_VL_CURRICULUM_LATTICES:
        assert [exposure[(0, index, lattice)] for index in range(4)] == [1, 1, 1, 1]
    odd_duplicates = []
    for lattice in NATIVE_VL_CURRICULUM_LATTICES:
        counts = [exposure[(1, index, lattice)] for index in range(3)]
        assert sorted(counts) == [1, 1, 2]
        odd_duplicates.append(counts.index(2))
    assert odd_duplicates[0] == odd_duplicates[1]


def test_curriculum_repairs_camera_incompatible_hash_pair_with_perfect_matching() -> None:
    pair_plan, audit = _inputs()
    first = audit.groups[0]
    conflict = replace(
        first,
        visible_camera_names=(
            ("gripper", "static"),
            ("gripper",),
            ("gripper", "static"),
            ("static",),
        ),
    )
    audit = replace(audit, groups=(conflict, *audit.groups[1:]))

    plan = build_native_vl_grounding_curriculum(
        pair_plan,
        audit,
        pair_plan_file_sha256=_sha("pair-plan-file"),
    )
    support = conflict.visible_camera_names
    first_group_steps = tuple(step for step in plan.steps if step.group_index == 0)
    assert len(first_group_steps) == 2
    for step in first_group_steps:
        batch = step.batches[0]
        left, right = batch.variant_indices
        assert batch.camera_name in set(support[left]).intersection(support[right])
        assert frozenset((left, right)) != frozenset((1, 3))


def test_curriculum_round_trip_and_semantic_tamper_rejection(tmp_path: Path) -> None:
    pair_plan, audit = _inputs()
    plan = build_native_vl_grounding_curriculum(
        pair_plan,
        audit,
        pair_plan_file_sha256=_sha("pair-plan-file"),
    )
    path = tmp_path / "curriculum.json"
    plan.write(path)
    assert NativeVLGroundingCurriculumPlan.load(path) == plan

    value = json.loads(path.read_text(encoding="ascii"))
    value["steps"] = value["steps"][:-1]
    for optimizer_step, step in enumerate(value["steps"]):
        step["optimizer_step"] = optimizer_step
    value["artifact_sha256"] = _canonical_artifact_sha256(value)
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(ValueError, match="coverage|step count"):
        NativeVLGroundingCurriculumPlan.load(tampered)

    value = json.loads(path.read_text(encoding="ascii"))
    value["object_row_addressable_variant_count"] = 0
    value["artifact_sha256"] = _canonical_artifact_sha256(value)
    tampered_count = tmp_path / "tampered-count.json"
    tampered_count.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(ValueError, match="must be positive"):
        NativeVLGroundingCurriculumPlan.load(tampered_count)

    value = json.loads(path.read_text(encoding="ascii"))
    value["steps"][0]["batches"][1]["camera_name"] = "gripper"
    value["artifact_sha256"] = _canonical_artifact_sha256(value)
    tampered_camera = tmp_path / "tampered-camera.json"
    tampered_camera.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(ValueError, match="rank-balanced scale pair"):
        NativeVLGroundingCurriculumPlan.load(tampered_camera)


def test_curriculum_rejects_cross_evidence_inputs() -> None:
    pair_plan, audit = _inputs()
    changed = replace(
        audit,
        fixed_x_audit=replace(
            audit.fixed_x_audit,
            training_projection_payload_sha256=_sha("different-projection-payload"),
        ),
    )
    with pytest.raises(ValueError, match="different evidence"):
        build_native_vl_grounding_curriculum(
            pair_plan,
            changed,
            pair_plan_file_sha256=_sha("pair-plan-file"),
        )


def test_curriculum_rejects_fixed_x_filtered_audit() -> None:
    pair_plan, audit = _inputs()
    with pytest.raises(TypeError, match="measurable-target audit"):
        build_native_vl_grounding_curriculum(
            pair_plan,
            audit.fixed_x_audit,  # type: ignore[arg-type]
            pair_plan_file_sha256=_sha("pair-plan-file"),
        )


def test_measurable_target_audit_rejects_reordered_fixed_x_evidence() -> None:
    _pair_plan, audit = _inputs()
    first = audit.groups[0]
    reordered = replace(
        first,
        group=replace(
            first.group,
            variants=(
                first.group.variants[1],
                first.group.variants[0],
                *first.group.variants[2:],
            ),
        ),
        visible_camera_names=(
            first.visible_camera_names[1],
            first.visible_camera_names[0],
            *first.visible_camera_names[2:],
        ),
    )
    with pytest.raises(ValueError, match="changed fixed-X evidence"):
        replace(audit, groups=(reordered, *audit.groups[1:]))


def test_curriculum_rejects_symlink(tmp_path: Path) -> None:
    pair_plan, audit = _inputs()
    plan = build_native_vl_grounding_curriculum(
        pair_plan,
        audit,
        pair_plan_file_sha256=_sha("pair-plan-file"),
    )
    path = tmp_path / "curriculum.json"
    plan.write(path)
    link = tmp_path / "curriculum-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="real file"):
        NativeVLGroundingCurriculumPlan.load(link)
