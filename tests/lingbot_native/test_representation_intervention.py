from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.lingbot_native.calvin import (
    build_native_calvin_continuation_batch,
    build_native_calvin_stream_plan,
    build_planned_native_calvin_batch,
)
from picf_next.lingbot_native.representation_intervention import (
    RepresentationTaskInterventionPlan,
    apply_representation_task_intervention,
    build_representation_task_intervention_plan,
)
from tests.test_calvin_data import _split_manifest, _write_split
from tests.test_lingbot_calvin import _dataset


def _resolver(task_key: str) -> tuple[str, ...]:
    return {
        "move_block": ("physical/block",),
        "turn_on_light": ("physical/switch",),
    }[task_key]


def _plan(tmp_path: Path):
    dataset = _dataset(tmp_path)
    stream = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-intervention-test",
        seed=37,
        global_batch_size=2,
        total_steps=3,
    )
    plan = build_representation_task_intervention_plan(
        stream,
        dataset,
        task_identity_resolver=_resolver,
    )
    return dataset, stream, plan


def test_intervention_plan_is_exact_deterministic_and_content_addressed(
    tmp_path: Path,
) -> None:
    dataset, stream, plan = _plan(tmp_path)
    replay = build_representation_task_intervention_plan(
        stream,
        dataset,
        task_identity_resolver=_resolver,
    )

    assert plan == replay
    assert plan.artifact_sha256 == replay.artifact_sha256
    assert plan.exact_slot_count == 6
    assert plan.inexact_slot_count == 0
    assert {item.task_key for item in plan.slots} == {"move_block", "turn_on_light"}
    assert all(item.intervened for item in plan.slots)
    assert all(
        set(item.target_identity_keys).isdisjoint(item.donor_target_identity_keys or ())
        for item in plan.slots
    )
    assert sorted(item.task_key for item in plan.slots) == sorted(
        item.donor_task_key for item in plan.slots
    )

    path = tmp_path / "intervention.json"
    plan.write(path)
    assert RepresentationTaskInterventionPlan.load(path) == plan
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["slots"][0]["donor"]["task_key"] = "tampered"
    with pytest.raises(ValueError, match="donor metadata"):
        RepresentationTaskInterventionPlan.from_dict(payload)


def test_intervention_plan_fails_closed_without_disjoint_targets(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    stream = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-intervention-impossible",
        seed=41,
        global_batch_size=2,
        total_steps=2,
    )
    with pytest.raises(ValueError, match="cannot form"):
        build_representation_task_intervention_plan(
            stream,
            dataset,
            task_identity_resolver=lambda _task: ("physical/shared",),
        )


def test_intervention_plan_rejects_cross_visit_donor_swap(tmp_path: Path) -> None:
    _, _, plan = _plan(tmp_path)
    slots = list(plan.slots)
    episode = slots[0].episode_instance_id
    indices = [index for index, item in enumerate(slots) if item.episode_instance_id == episode]
    first_index, second_index = indices[:2]
    first = slots[first_index]
    second = slots[second_index]

    def donor_fields(item):
        return {
            "donor_optimizer_step": item.donor_optimizer_step,
            "donor_lane_id": item.donor_lane_id,
            "donor_episode_instance_id": item.donor_episode_instance_id,
            "donor_sample_key": item.donor_sample_key,
            "donor_task_key": item.donor_task_key,
            "donor_instruction_sha256": item.donor_instruction_sha256,
            "donor_target_identity_keys": item.donor_target_identity_keys,
        }

    slots[first_index] = replace(first, **donor_fields(second))
    slots[second_index] = replace(second, **donor_fields(first))
    with pytest.raises(ValueError, match="visit count or ordinal"):
        replace(plan, slots=tuple(slots))


def test_intervention_rotates_eligible_donor_targets_between_visits(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    annotations = {
        "language": {
            "ann": ["query alpha", "query beta", "query gamma"],
            "task": ["task_alpha", "task_beta", "task_gamma"],
        },
        "info": {"indx": [(10, 14), (11, 15), (12, 16)]},
    }
    np.save(split / "lang_annotations" / "auto_lang_ann.npy", annotations)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    stream = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-intervention-rotation",
        seed=43,
        global_batch_size=3,
        total_steps=3,
    )
    identities = {
        "task_alpha": ("physical/alpha",),
        "task_beta": ("physical/beta",),
        "task_gamma": ("physical/gamma",),
    }
    plan = build_representation_task_intervention_plan(
        stream,
        dataset,
        task_identity_resolver=identities.__getitem__,
    )

    by_episode = {}
    for item in plan.slots:
        by_episode.setdefault(item.episode_instance_id, []).append(item)
    for items in by_episode.values():
        ordered = sorted(items, key=lambda item: item.optimizer_step)
        donors = [item.donor_target_identity_keys for item in ordered]
        assert len(set(donors)) >= 2
        assert all(
            previous != current for previous, current in zip(donors, donors[1:], strict=False)
        )


def test_intervention_changes_only_prompt_and_loss_side_task(tmp_path: Path) -> None:
    dataset, stream, intervention = _plan(tmp_path)
    natural = build_planned_native_calvin_batch(
        stream,
        dataset,
        optimizer_step=0,
        rank=0,
        world_size=2,
        gradient_accumulation_steps=1,
        accumulation_index=0,
    )
    natural_digest = natural.source_digest
    changed = apply_representation_task_intervention(
        natural,
        intervention,
        dataset,
    )
    transition = natural.plan_microbatch.transitions[0]
    slot = intervention.slot_for(transition, optimizer_step=0)

    assert changed.task_intervention_sha256 == intervention.artifact_sha256
    assert changed.source_digest != natural_digest
    assert changed.training.routing == natural.training.routing
    assert changed.training.controls is natural.training.controls
    assert changed.augmentation_seeds == natural.augmentation_seeds
    assert changed.flow_noise_seeds == natural.flow_noise_seeds
    assert changed.flow_timestep_seeds == natural.flow_timestep_seeds
    assert changed.training.host_items[0]["task"] != natural.training.host_items[0]["task"]
    assert changed.training.structural_target_requests[0].task_key == slot.donor_task_key
    assert natural.training.structural_target_requests[0].task_key == slot.task_key
    for name, natural_value in natural.training.host_items[0].items():
        if name == "task":
            continue
        changed_value = changed.training.host_items[0][name]
        assert isinstance(natural_value, torch.Tensor)
        assert changed_value.data_ptr() == natural_value.data_ptr()

    with pytest.raises(ValueError, match="only once"):
        apply_representation_task_intervention(changed, intervention, dataset)
    with pytest.raises(ValueError, match="stream differ"):
        apply_representation_task_intervention(
            replace(natural, plan_sha256="f" * 64),
            intervention,
            dataset,
        )


def test_natural_prompt_and_task_remain_source_exact_across_continuation(
    tmp_path: Path,
) -> None:
    dataset, stream, intervention = _plan(tmp_path)
    natural = build_planned_native_calvin_batch(
        stream,
        dataset,
        optimizer_step=0,
        rank=0,
        world_size=2,
        gradient_accumulation_steps=1,
        accumulation_index=0,
    )
    transition = natural.plan_microbatch.transitions[0]
    locator = dataset.locator_by_key(transition.sample.sample_key)
    segment = dataset.index.segments[locator.segment_index]
    request = natural.training.structural_target_requests[0]

    assert natural.task_intervention_sha256 is None
    assert natural.training.host_items[0]["task"] == segment.instruction
    assert request.task_key == segment.task_key

    continuation = build_native_calvin_continuation_batch(
        natural,
        dataset,
        offset=1,
    )
    assert continuation.task_intervention_sha256 is None
    assert continuation.training.host_items[0]["task"] == segment.instruction
    assert continuation.training.structural_target_requests[0].task_key == segment.task_key

    changed = apply_representation_task_intervention(
        natural,
        intervention,
        dataset,
    )
    assert changed.source_digest != natural.source_digest
    assert changed.task_intervention_sha256 == intervention.artifact_sha256


def test_intervention_propagates_one_query_across_continuation_frames(
    tmp_path: Path,
) -> None:
    dataset, stream, intervention = _plan(tmp_path)
    natural = build_planned_native_calvin_batch(
        stream,
        dataset,
        optimizer_step=0,
        rank=0,
        world_size=2,
        gradient_accumulation_steps=1,
        accumulation_index=0,
    )
    changed = apply_representation_task_intervention(
        natural,
        intervention,
        dataset,
    )
    continuation = build_native_calvin_continuation_batch(
        changed,
        dataset,
        offset=1,
    )

    assert continuation.task_intervention_sha256 == intervention.artifact_sha256
    assert continuation.training.host_items[0]["task"] == (changed.training.host_items[0]["task"])
    assert continuation.training.structural_target_requests[0].task_key == (
        changed.training.structural_target_requests[0].task_key
    )
    assert (
        continuation.source_digest
        != build_native_calvin_continuation_batch(
            natural,
            dataset,
            offset=1,
        ).source_digest
    )
